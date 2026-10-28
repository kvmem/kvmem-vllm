"""Fixed operands, old frozen scorer versus bounded per-layer Key residency.

Synthetic FP8-representable Keys cover MAB history/Query geometry. These are
component measurements, not model throughput or benchmark answer accuracy.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from benchmark_niah import save, sha
from vllm_kvmem import query_key_fps as current


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve previous component results')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('vllm_kvmem._frozen_fps_reference',args.reference)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(814)
    results = []
    for blocks,rows in [(12864,264),(23584,272)]:
        q = torch.randn(args.layers,rows,24,256,device='cuda',dtype=torch.bfloat16)
        reps = torch.empty(args.layers,blocks,4,4,256,dtype=torch.float16)
        for layer in range(args.layers):
            reps[layer].copy_(torch.randn(blocks,4,4,256).to(torch.float8_e4m3fn).half())
        ids = torch.arange(4,dtype=torch.int8).view(1,1,4,1).expand(args.layers,blocks,4,4).clone()
        ids[:,13,2:] = -1
        mask = torch.ones(blocks,dtype=torch.bool); mask[::23] = False
        measurements = []; reference = None
        for repeat in range(args.repeats+1):
            for name,module in [('old',old),('new',current)] if repeat%2==0 else [('new',current),('old',old)]:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated(); stats = {}
                started = time.perf_counter()
                kwargs = dict(query_half=True,optimized=True)
                if name == 'new': kwargs['execution_stats'] = stats
                scores = module.key_fps_scores(q,reps,ids,mask,**kwargs)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-started
                if reference is None: reference = scores.clone()
                if not torch.equal(reference,scores):
                    raise AssertionError(f'Scores differ: {name}, max error {(reference-scores).abs().max().item()}')
                result = dict(arm=name,repeat=repeat,seconds=elapsed,warmup=repeat==0,
                    exact_scores=True,peak_scratch_bytes=torch.cuda.max_memory_allocated()-base,stats=stats)
                measurements.append(result); print(json.dumps(dict(blocks=blocks,rows=rows,**result)),flush=True)
                del scores
        medians = {name:statistics.median(r['seconds'] for r in measurements if r['arm']==name and not r['warmup'])
            for name in ('old','new')}
        results.append(dict(blocks=blocks,rows=rows,layers=args.layers,measurements=measurements,
            medians=medians,speedup=medians['old']/medians['new']))
        del q,reps,ids,reference
    save(args.output,dict(scope=__doc__,results=results,exact_scores=True,
        source_sha256=sha(Path(current.__file__)),reference_sha256=sha(args.reference),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',type=int,default=16)
    p.add_argument('--repeats',type=int,default=2)
    run(p.parse_args())
