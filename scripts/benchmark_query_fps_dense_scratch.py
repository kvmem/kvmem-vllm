"""Verify the unchanged dense FPS scorer's bounded layer cache on fixed operands.

No model answers are generated. Uses the missing-counter pilot's full geometry,
measures incremental CUDA allocation, and counts CPU-to-GPU Key layer copies.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from benchmark_niah import save,sha
from vllm_kvmem import query_key_fps as fps


def run(output):
    if output.exists():raise FileExistsError('Preserve prior component measurements')
    expected='451e9fc3048823765471e7f91dd4e4ca45910327c8017305552493f421ef9b68'
    assert sha(Path(fps.__file__))==expected,'Only verify the exact frozen scorer'
    torch.set_num_threads(1);torch.manual_seed(31439);torch.backends.cuda.matmul.allow_tf32=False
    layers,rows,heads,kvheads,dim,blocks,slots,chunk=16,3718,24,4,256,8224,4,8
    queries=torch.randn(layers,rows,heads,dim,device='cuda',dtype=torch.bfloat16)
    reps=torch.empty(layers,blocks,slots,kvheads,dim,dtype=torch.float16)
    for layer in range(layers):reps[layer].copy_(torch.randn(blocks,slots,kvheads,dim).to(torch.float8_e4m3fn).half())
    indices=torch.arange(4,dtype=torch.int8).view(1,1,4,1).expand(layers,blocks,slots,kvheads).clone()
    mask=torch.ones(blocks,dtype=torch.bool);mask[:41]=False
    # Warm both the full eight-row chunk and final six-row chunk without
    # retaining outputs; the measured call below uses all 16 x 3718 rows.
    fps.key_fps_scores(queries[:1,:14],reps[:1],indices[:1],mask,query_half=True,optimized=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    original_to=torch.Tensor.to;copies=[]
    def traced_to(t,*args,**kwargs):
        result=original_to(t,*args,**kwargs)
        if (t.device.type=='cpu' and result.device.type=='cuda' and t.dtype==torch.float16 and
                tuple(t.shape)==(blocks,slots,kvheads,dim)):
            copies.append(t.numel()*t.element_size())
        return result
    stats={};began=time.perf_counter()
    try:
        torch.Tensor.to=traced_to
        scores=fps.key_fps_scores(queries,reps,indices,mask,query_half=True,optimized=True,execution_stats=stats)
        torch.cuda.synchronize()
    finally:torch.Tensor.to=original_to
    seconds=time.perf_counter()-began;peak=torch.cuda.max_memory_allocated()-base
    bound=(2*blocks*slots*kvheads*dim*4+3*chunk*heads*blocks*slots*4+
        3*chunk*heads*blocks*4+2*chunk*heads*dim*4+blocks*slots*kvheads*2+blocks*8+
        blocks*slots*kvheads*dim*reps.element_size())
    assert stats=={},'Expected original dense branch without execution counters'
    assert len(copies)==layers and sum(copies)==reps.numel()*reps.element_size()
    assert 0<peak<=bound<=fps.SCRATCH_LIMIT and torch.isfinite(scores).all() and (scores>=0).all()
    assert scores[:41].count_nonzero()==0
    mass=scores.double().sum().item();assert abs(mass-rows)/rows<1e-4
    report=dict(status='passed',source_sha256=expected,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        layers=layers,query_tokens=rows,query_heads=heads,kv_heads=kvheads,head_dim=dim,blocks=blocks,
        representatives=slots,query_chunk=chunk,seconds=seconds,
        key_layer_copies=len(copies),key_upload_bytes=sum(copies),one_copy_per_layer=True,
        peak_incremental_cuda_allocated_bytes=peak,derived_scratch_bound_bytes=bound,
        scratch_limit_bytes=fps.SCRATCH_LIMIT,seed_mass=mass,masked_mass=0.,original_stats=stats,
        scope='Fixed synthetic operands at actual pilot geometry. Incremental live torch CUDA allocation '
            'excludes resident input Q and model/KV, allocator reserved memory and external-library allocations. '
            'This verifies the dense branch/cache bound, not a model-throughput or accuracy measurement.')
    save(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
