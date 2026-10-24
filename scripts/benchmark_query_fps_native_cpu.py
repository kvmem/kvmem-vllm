#!/usr/bin/env python3
"""CPU-only, fixed-real-Key cold-cache benchmark of Torch and native FPS.

Each measured trial calls the production cache.snapshot including owned Key
reads, ambiguity fallback, representative conversion and publication. Warmup
and exhaustive equality checks are separate. Inputs are mmap-backed and warmed
by hashing; neither live archive throughput nor model throughput is measured.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import sys
from threading import RLock
import time
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
CASE = 'niah_multikey_2-0011'
CAPTURE = REPO/'artifacts/ruler-retrieval-diagnostic-20261023-r4/compressed/cases'/CASE/'diagnostic'
OFFSETS = REPO/'artifacts/query-key-fps-fixed-operands-20261023-r1'/CASE/'representative-token-offsets.npy'
CACHE_BYTES = 12*1024**3
SCOPE = ('CPU component only: original 261952-token FP8 Keys; empty derived cache and owned mmap '
    'Key slabs. Timing includes reading, FPS, ambiguity repair, conversion and publication. '
    'Input hashes warm OS file caches; no model, live archive, score calculation or GPU is run.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def frozen_modules(root):
    """Load a private source/binary copy, never a mutable in-flight module."""
    source = root/'source'
    package_dir = source/'vllm_kvmem'
    package_dir.mkdir(parents=True)
    for name in ('query_key_fps.py', 'query_window_scoring.py', 'query_fps_cpu.py'):
        shutil.copy2(REPO/'src/vllm_kvmem'/name, package_dir/name)
    libraries = list((REPO/'src/vllm_kvmem').glob('_native*.so'))
    require(len(libraries) == 1, 'Expected exactly one locally compiled native extension')
    shutil.copy2(libraries[0], package_dir/libraries[0].name)
    shutil.copytree(REPO/'csrc', source/'csrc', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO/'third_party/qw3', source/'third_party/qw3',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('setup.py', 'pyproject.toml'):
        shutil.copy2(REPO/name, source/name)
    shutil.copy2(__file__, source/Path(__file__).name)
    name = '_native_fps_component_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    package = ModuleType(name)
    package.__path__, package.__package__ = [str(package_dir)], name
    sys.modules[name] = package
    module = importlib.import_module(name+'.query_key_fps')
    native = importlib.import_module(name+'._native')
    require(hasattr(native, 'KeyFPSCPUPool'), 'Native CPU FPS pool has not been compiled')
    return module, native


class OwnedKeys:
    """Bounded owned copies of original Keys; no target/offset enters reading."""
    def __init__(self, keys, metadata):
        require(keys.dtype == np.uint8 and keys.ndim == 4, 'Expected [L,N,H,D] original FP8 bytes')
        counts, starts = metadata['counts'], metadata['starts']
        require(len(counts) == len(starts) and sum(counts) == keys.shape[1] and
            all(type(n) is int and n == 32 for n in counts), 'Expected accepted full 32-token records')
        require(starts == sorted(set(starts)), 'Original logical positions are not ordered')
        self.keys, self.history_end = keys, metadata['history']
        self.begins = np.r_[0, np.cumsum(counts)]
        self.records = [NS(history_begin=start, count=count, handle=NS(slot=i, generation=1))
            for i, (start, count) in enumerate(zip(starts, counts))]
        self.archive = NS(_lock=RLock())

    def read_score_keys(self, handles, *, pin_memory=False):
        require(not pin_memory, 'CPU benchmark must not allocate CUDA-pinned memory')
        layers, _, heads, dim = self.keys.shape
        slots = [h.slot for h in handles]
        require(0 < len(slots) <= 32 and len(set(slots)) == len(slots) and
            all(type(h.slot) is int and 0 <= h.slot < len(self.records) and h.generation == 1
                for h in handles), 'Invalid owned original-Key read')
        slab = torch.empty((len(slots), layers, 32, heads, dim), dtype=torch.float8_e4m3fn)
        target = slab.view(torch.uint8).numpy()
        started = time.perf_counter()
        for i, slot in enumerate(slots):
            lo, hi = int(self.begins[slot]), int(self.begins[slot+1])
            np.copyto(target[i], self.keys[:, lo:hi], casting='no')
        return slab, dict(key_read_bytes=slab.numel(), value_read_bytes=0, draft_key_read_bytes=0,
            key_read_batches=1, key_resident_bytes=0, key_region_seconds=time.perf_counter()-started)


def validate_every_representative(values, indices, keys, metadata, offsets, block_ids):
    """Exact saved offsets and selected original FP8 bytes, outside timing."""
    require(values.device.type == indices.device.type == 'cpu', 'CPU cache expected')
    selected_ids = np.asarray(block_ids, dtype=np.int64)
    expected_offsets = offsets[:, selected_ids]
    require(np.array_equal(indices.numpy(), expected_offsets), 'Representative offset differs from original CPU FPS')
    begins = np.r_[0, np.cumsum(metadata['counts'])][selected_ids]
    heads = np.arange(keys.shape[2])[None, None, :]
    digest = hashlib.sha256()
    for layer in range(keys.shape[0]):
        for begin in range(0, len(selected_ids), 32):
            end = min(begin+32, len(selected_ids))
            chosen = indices[layer, begin:end].numpy().astype(np.int64)
            require(np.all((chosen >= 0) & (chosen < 32)), 'Invalid real-token representative offset')
            raw = np.array(keys[layer, begins[begin:end, None, None]+chosen, heads, :], copy=True)
            actual = values[layer, begin:end]
            expected = torch.from_numpy(raw).view(torch.float8_e4m3fn).to(actual.dtype)
            require(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
                'Representative cache bytes differ from original selected Keys')
            # e4m3fn values embed exactly in cache dtype; signed zero is checked too.
            recovered = actual.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            require(np.array_equal(recovered, raw), 'Original selected FP8 Key bytes changed')
            digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
    return dict(exact_offsets=True, exact_cache_bytes=True, exact_original_fp8_bytes=True,
        checked_offsets=int(indices.numel()), cache_value_sha256=digest.hexdigest())


def trial(module, keys, metadata, offsets, block_ids, *, pool=None):
    require(torch.get_num_threads() == 1, 'Only the benchmark process Torch reference must stay at one thread')
    authority = OwnedKeys(keys, metadata)
    records = [authority.records[i] for i in block_ids]
    cache = module.RawKeyFPSCache(authority.archive, keys.shape[0], keys.shape[2], keys.shape[3],
        dtype=torch.float16, max_bytes=CACHE_BYTES)
    try:
        with torch.inference_mode(), cache.use():
            before = torch.get_num_threads()
            started, cpu_started = time.perf_counter(), time.process_time()
            values, indices, stats = cache.snapshot(authority, records, optimized=True,
                device='cpu', cpu_pool=pool)
            seconds, cpu_seconds = time.perf_counter()-started, time.process_time()-cpu_started
            require(before == torch.get_num_threads() == 1, 'Native pool changed global Torch thread count')
            require(stats['cache_hit_blocks'] == 0 and stats['cache_rebuilt_blocks'] == len(records),
                'Cold construction unexpectedly reused rows')
            require(stats.get('fps_gpu_groups') == 0 and stats.get('construction_device') == 'cpu',
                'Construction was not entirely on CPU')
            groups = keys.shape[0]*len(records)*keys.shape[2]
            if pool is None:
                require(stats.get('fps_cpu_native_groups') == 0 and
                    stats.get('fps_cpu_reference_groups') == groups,
                    'Torch reference did not build every original group')
            else:
                require(stats.get('fps_cpu_native_groups') == groups and
                    stats.get('fps_cpu_pool_busy_groups') == 0 and
                    stats.get('fps_cpu_unsupported_fallback_groups') == 0,
                    'Native benchmark silently used a capacity/unsupported fallback')
            checked = validate_every_representative(values, indices, keys, metadata, offsets, block_ids)
            tokens = sum(row.count for row in records)
            result = dict(seconds=seconds, cpu_seconds=cpu_seconds, average_cpu_cores=cpu_seconds/seconds,
                tokens=tokens, blocks=len(records), tokens_per_second=tokens/seconds,
                torch_threads=before, cache_bytes=cache.bytes, stats=stats, validation=checked,
                process_cumulative_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            del values, indices
    finally:
        cache.clear()
    require(authority.archive._query_window_cache_bytes == 0, 'Derived cache budget leak')
    require(not torch.cuda.is_initialized(), 'CPU component initialized CUDA')
    gc.collect()
    return result


def initial_arms(workers):
    require(workers and all(type(n) is int and 1 <= n <= 16 for n in workers)
        and len(set(workers)) == len(workers), 'Use distinct native worker counts from one to sixteen')
    return [('torch_1', None), *((f'native_{n}', n) for n in workers)]


def confirmation_order(best, repeats):
    require(type(repeats) is int and 0 <= repeats <= 3, 'Use zero to three interleaved confirmations')
    return [(round_, arm) for round_ in range(repeats) for arm in
        ((best, 'torch_1') if round_%2 == 0 else ('torch_1', best))]


def summarize(results):
    names = list(dict.fromkeys(row['arm'] for row in results))
    arms = {}
    for name in names:
        selected = [row for row in results if row['arm'] == name]
        median = statistics.median(row['seconds'] for row in selected)
        arms[name] = dict(trials=len(selected), seconds=[row['seconds'] for row in selected],
            median_seconds=median, tokens_per_second=selected[0]['tokens']/median,
            mean_cpu_cores=statistics.fmean(row['average_cpu_cores'] for row in selected),
            offsets_checked_per_trial=selected[0]['validation']['checked_offsets'])
    for name in names:
        arms[name]['speedup_vs_torch1'] = arms['torch_1']['median_seconds']/arms[name]['median_seconds']
    return dict(scope=SCOPE, arms=arms, exact_offsets_and_original_keys=True,
        model_or_gpu_started=False, torch_reference_threads=1,
        limitation='Private pool construction/warmup and exact checks are separately recorded, outside cold-cache timing. '
            'Different arms have declared repeat counts; no model throughput or concurrent prefill benefit is implied.')


def run(root, *, capture=CAPTURE, offsets_path=OFFSETS, workers=(4, 8, 16), confirms=2):
    arms = initial_arms(workers)
    confirmation_order('placeholder', confirms)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root/'status.json', dict(status='preparing', pid=os.getpid(), at=time.time()))
    try:
        # This standalone child process owns the setting; serving processes are untouched.
        torch.set_num_threads(1)
        module, native = frozen_modules(root)
        paths = dict(keys=Path(capture)/'raw_keys.npy', metadata=Path(capture)/'archive.json', offsets=Path(offsets_path))
        source = {str(p.relative_to(root/'source')):sha(p) for p in (root/'source').rglob('*') if p.is_file()}
        proof = dict(inputs={name:dict(path=str(path.resolve()), sha256=sha(path)) for name,path in paths.items()}, source=source)
        save(root/'proof.json', proof)
        keys = np.load(paths['keys'], mmap_mode='r')
        metadata, offsets = read(paths['metadata']), np.load(paths['offsets'])
        require(keys.shape == (16, 261952, 4, 256) and len(metadata['counts']) == 8186 and
            all(n == 32 for n in metadata['counts']), 'Expected the exact saved 261952-token geometry')
        require(offsets.shape == (16, 8186, 4, 4), 'Saved independent CPU offsets have another geometry')
        all_blocks = list(range(len(metadata['counts'])))
        warm_blocks = np.unique(np.linspace(0, len(all_blocks)-1, 64, dtype=int)).tolist()
        save(root/'environment.json', dict(scope=SCOPE, torch=torch.__version__,
            allowed_cpus=sorted(os.sched_getaffinity(0)), cpu_count=os.cpu_count(),
            torch_threads=1, cuda_initialized=torch.cuda.is_initialized(),
            native_worker_candidates=list(workers), initial_order=[name for name,_ in arms],
            confirmation_repeats=confirms, worker_selection='Fastest complete native snapshot; timing only, never answers.',
            warmup_original_blocks=warm_blocks, parallel_backend=torch.__config__.parallel_info()))
        results, warmups = [], []

        def measured(arm, workers_, stage, repeat):
            pool = native.KeyFPSCPUPool(threads=workers_, max_submissions=2) if workers_ is not None else None
            try:
                warm = trial(module, keys, metadata, offsets, warm_blocks, pool=pool)
                warm.update(arm=arm, stage=stage, repeat=repeat, native_workers=workers_)
                warmups.append(warm);save(root/'warmup.json', warmups)
                value = trial(module, keys, metadata, offsets, all_blocks, pool=pool)
                value.update(arm=arm, stage=stage, repeat=repeat, native_workers=workers_,
                    pool_stats=pool.stats() if pool is not None else None)
                results.append(value);save(root/'full.json', results)
                print(json.dumps(dict(arm=arm, stage=stage, repeat=repeat, seconds=value['seconds'],
                    tokens_per_second=value['tokens_per_second'], exact_offsets_and_keys=True)), flush=True)
                return value
            finally:
                if pool is not None:
                    pool.close()

        save(root/'status.json', dict(status='running', pid=os.getpid(), at=time.time()))
        for name, count in arms:
            measured(name, count, 'initial', 0)
        best = min((row for row in results if row['native_workers'] is not None), key=lambda r:r['seconds'])
        save(root/'selection.json', dict(arm=best['arm'], native_workers=best['native_workers'],
            based_on='All declared full-input initial native wall times; no generation or score is used.'))
        for repeat, name in confirmation_order(best['arm'], confirms):
            measured(name, None if name=='torch_1' else best['native_workers'], 'confirmation', repeat)
        for name, path in paths.items():
            require(sha(path) == proof['inputs'][name]['sha256'], 'Saved source input changed during measurement')
        for name, digest in source.items():
            require(sha(root/'source'/name) == digest, 'Frozen benchmark source or binary changed')
        require(not torch.cuda.is_initialized(), 'CPU component initialized CUDA')
        save(root/'summary.json', summarize(results))
        save(root/'status.json', dict(status='completed', at=time.time(), trials=len(results)))
    except BaseException as error:
        save(root/'failure.json', dict(error=repr(error), at=time.time()))
        save(root/'status.json', dict(status='failed', at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--offsets', type=Path, default=OFFSETS)
    parser.add_argument('--workers', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--confirm-repeats', type=int, default=2)
    args = parser.parse_args()
    run(args.root, capture=args.capture, offsets_path=args.offsets, workers=args.workers,
        confirms=args.confirm_repeats)
