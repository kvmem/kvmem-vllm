"""CPU-only FPS feasibility on saved, real 256K FP8 Keys.

No model or CUDA context is started. Timing includes owned Key reads, CPU FPS,
and representative publication; checking the saved reference happens outside
timing. File cache is warm, so this is not a disk or serving throughput test.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import ModuleType

import numpy as np
import torch

import benchmark_query_fps_publication as prior


def frozen_module(root):
    source = root/'source'
    source.mkdir()
    previous = prior.REPO/'artifacts/query-fps-publication-component-20261023-r2/source'
    for directory, name in [('new_module', 'query_key_fps.py'), ('new_base', 'query_window_scoring.py')]:
        shutil.copyfile(previous/directory/name, source/name)
    shutil.copyfile(__file__, source/Path(__file__).name)
    shutil.copyfile(prior.__file__, source/Path(prior.__file__).name)
    package = ModuleType('_cpu_fps_frozen')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__+'.query_key_fps', source/'query_key_fps.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trial(module, keys, meta, offsets, threads, blocks, *, full):
    torch.set_num_threads(threads)
    authority = prior.OwnedMmapAuthority(keys, meta)
    cache = module.RawKeyFPSCache(authority.archive, keys.shape[0], keys.shape[2], keys.shape[3],
        dtype=torch.float16, max_bytes=12*1024**3)
    records = authority.records if full else [authority.records[i] for i in blocks]
    with cache.use(), torch.inference_mode():
        wall = time.perf_counter(); cpu = time.process_time()
        values, indices, stats = cache.snapshot(authority, records, optimized=True, device='cpu')
        seconds = time.perf_counter()-wall; cpu_seconds = time.process_time()-cpu
        expected = offsets if full else offsets[:, blocks]
        prior.require(np.array_equal(indices.numpy(), expected), 'CPU representatives differ from saved offsets')
        # Validate every selected Key directly against the original FP8 bytes.
        starts = authority.begins if full else authority.begins[blocks]
        heads = np.arange(keys.shape[2])[None, None, :]
        for layer in range(keys.shape[0]):
            for begin in range(0, len(records), 32):
                end = min(begin+32, len(records))
                selected = indices[layer, begin:end].numpy().astype(np.int64)
                raw = np.array(keys[layer, starts[begin:end, None, None]+selected, heads, :], copy=True)
                expected_values = torch.from_numpy(raw).view(torch.float8_e4m3fn).half()
                prior.require(torch.equal(values[layer, begin:end], expected_values), 'CPU representative Key bytes changed')
        tokens = sum(r.count for r in records)
        result = dict(threads=threads, blocks=len(records), tokens=tokens, seconds=seconds,
            cpu_seconds=cpu_seconds, tokens_per_second=tokens/seconds, average_cpu_cores=cpu_seconds/seconds,
            cache_bytes=cache.bytes, stats=stats, exact_offsets=True, exact_original_keys=True)
        del values, indices
    cache.clear()
    prior.require(authority.archive._query_window_cache_bytes == 0, 'Cache budget leak')
    gc.collect()
    return result


def run(root, threads=(1, 2, 4, 8), repeats=3):
    if not threads or any(n < 1 for n in threads) or repeats < 1:
        raise ValueError('Positive CPU threads and repeats required')
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: prior.save(root/name, value)
    save('status.json', dict(status='running', pid=os.getpid()))
    try:
        module = frozen_module(root)
        paths = dict(keys=prior.CAPTURE/'raw_keys.npy', metadata=prior.CAPTURE/'archive.json', offsets=prior.OFFSETS)
        proof = {name: dict(path=str(path.resolve()), sha256=prior.sha(path)) for name, path in paths.items()}
        proof['source'] = {p.name: prior.sha(p) for p in (root/'source').iterdir() if p.is_file()}
        save('proof.json', proof)
        keys = np.load(paths['keys'], mmap_mode='r')
        meta = prior.read(paths['metadata']); offsets = np.load(paths['offsets'])
        prior.require(all(n == 32 for n in meta['counts']), 'This CPU qualification requires the sealed 32-token fixture')
        # Predetermined coverage across the entire history, independent of scores.
        blocks = np.unique(np.linspace(0, len(meta['counts'])-1, 512, dtype=int)).tolist()
        save('environment.json', dict(key_shape=list(keys.shape), torch=torch.__version__,
            allowed_cpus=sorted(os.sched_getaffinity(0)), cpu_count=os.cpu_count(),
            scope='CPU component, real fixed 256K FP8 Keys, warm file cache, owned mmap reads; no model or GPU.',
            sweep_threads=list(threads), selected_history_blocks=blocks))
        sweep = []
        for count in threads:
            row = trial(module, keys, meta, offsets, count, blocks, full=False)
            sweep.append(row); save('sweep.json', sweep)
            print(json.dumps(dict(stage='sweep', threads=count, seconds=row['seconds'], tokens_per_second=row['tokens_per_second'])), flush=True)
        # Select by measured construction speed, never by answer quality.
        best = min(sweep, key=lambda row: row['seconds'])['threads']
        results = []
        for repeat in range(repeats):
            row = trial(module, keys, meta, offsets, best, [], full=True)
            row['repeat'] = repeat; results.append(row); save('full.json', results)
            print(json.dumps(dict(stage='full', repeat=repeat, threads=best, seconds=row['seconds'], tokens_per_second=row['tokens_per_second'])), flush=True)
        prior.require(not torch.cuda.is_initialized(), 'CPU test unexpectedly initialized CUDA')
        for name, path in paths.items():
            prior.require(prior.sha(path) == proof[name]['sha256'], 'Input changed during measurement')
        median = statistics.median(row['seconds'] for row in results)
        save('summary.json', dict(status='passed', full_repeats=repeats, threads=best,
            tokens=sum(meta['counts']), median_seconds=median, tokens_per_second=sum(meta['counts'])/median,
            cpu_seconds_median=statistics.median(row['cpu_seconds'] for row in results),
            exact_offsets_and_raw_keys=True, cuda_initialized=False,
            limitation='CPU component feasibility only; model concurrency, NUMA contention and native archive copies still need live testing.'))
        save('status.json', dict(status='completed', at=time.time()))
    except BaseException as error:
        save('failure.json', dict(error=repr(error), at=time.time()))
        save('status.json', dict(status='failed', at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--full-repeats', type=int, default=3)
    args = parser.parse_args()
    run(args.root.resolve(), args.threads, args.full_repeats)
