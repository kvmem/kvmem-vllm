"""Attribute capture functionalization copies in matched14K diagnostic traces.

No GPU runs. Source-level logical copied bytes are not profiler DRAM counters.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

from analyze_matched_14k_nsight import digest, require, statistics, NORM, FA2, LAUNCH_GEOMETRY
from compare_prefill_nsight import load

COPY_NAMES = ["triton_poi_fused_as_strided_clone_slice_view_4",
              "triton_poi_fused_as_strided_clone_slice_view_5",
              "triton_poi_fused_as_strided_clone_slice_view_5",
              "triton_poi_fused_6", "triton_poi_fused_7"]


def brief(row):
    return dict(name=row['name'], kind=row['kind'], step=row['step'],
        correlation_id=row['correlationId'], stream=row['streamId'],
        launch_ns=row['api']['start'], start_ns=row['start'], end_ns=row['end'],
        ms=(row['end']-row['start'])/1e6, bytes=row.get('bytes'))


def analyze(root, wrapper):
    raw, result = {}, {}
    for name in ['native', 'kvmem']:
        _, shapes, _, data = load(root/name)
        require([(s['query_tokens'], s['active_seq_len']) for s in shapes] ==
            [(14336,16384),(14336,30720)], 'Not matched14K')
        require(all(r['api'] is not None for r in data), 'Unknown activity launch')
        raw[name]=sorted(data,key=lambda r:(r['api']['start'],r['start']))
    src=wrapper.read_text()
    for proof in ['triton_poi_fused_as_strided_clone_slice_view_4.run(arg22_1, buf33,',
                  'triton_poi_fused_as_strided_clone_slice_view_5.run(arg23_1, buf34,',
                  'triton_poi_fused_as_strided_clone_slice_view_5.run(arg21_1, buf35,',
                  'triton_poi_fused_6.run(buf33, arg22_1,',
                  'triton_poi_fused_7.run(buf34, buf35, arg23_1, arg21_1']:
        require(proof in src,'Expected AOT capture copy pattern missing')
    require('5710696ae8a4ad64ffa38e9993589119712c2678c47189f65e799d11dae46df5' in
            (root/'kvmem/server.log').read_text(), 'Actual loaded AOT wrapper differs')
    copy_rows=[]; projections=[]
    for step in range(2):
        ks=[r for r in raw['kvmem'] if r['step']==step and r['kind']=='kernel']
        ns=[r for r in ks if NORM in r['name']]
        require(len(ns)==20,'Expected target16+MTPprefill1+MTPdecode3 norms')
        for index,norm in enumerate(ns[:16]):
            i=ks.index(norm)
            five=ks[i-3:i]+ks[i+1:i+3]
            require([r['name'] for r in five]==COPY_NAMES,'Capture copy launch adjacency changed')
            require(all(r['streamId']==norm['streamId'] for r in five),'Different capture stream')
            copy_rows.extend(five)
            projections.append(dict(step=step,target_fa2_ordinal=index,
                target_model_layer=4*index+3,norm=brief(norm),copies=[brief(r) for r in five],
                copies_ms=sum(r['end']-r['start'] for r in five)/1e6))
    require(not any(r['name'] in COPY_NAMES for r in raw['native']), 'Native has same named capture copies')
    expected=Counter({'triton_poi_fused_as_strided_clone_slice_view_4':32,
        'triton_poi_fused_as_strided_clone_slice_view_5':64,'triton_poi_fused_6':32,'triton_poi_fused_7':32})
    require(Counter(r['name'] for r in copy_rows)==expected,'Incomplete capture projection pairing')
    result['capture_functionalization']=dict(total=statistics(copy_rows),
        by_kernel={n:statistics([r for r in copy_rows if r['name']==n]) for n in expected},
        per_projection=projections,source=str(wrapper),source_lines=[1709,1754],
        source_array_shapes={'raw_Q':[16424,24,256],'raw_K':[16424,4,256],'raw_V':[16424,4,256]},
        logical_copied_bytes=2*32*14336*(24+4+4)*256*2,
        logical_byte_formula='clone and copyback (2) * target layer instances (32) * Q tokens (14336) * (24+4+4) heads * 256 dimensions * BF16 bytes (2)',
        estimated_read_plus_write_bytes=2*2*32*14336*(24+4+4)*256*2,
        byte_limit='Calculated logical copy extent, not measured DRAM traffic; cache effects are unknown.',
        optimization_limit='Could remove functionalization copies by changing mutation capture contract; raw Q/K/V producer writes and lifetime constraints remain necessary. Not a measured speedup.')
    for arm,seq in raw.items():
        with sqlite3.connect('file:'+str((root/arm/'trace.sqlite').resolve())+'?mode=ro',uri=True) as db:
            kinds=dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        d2d=[r for r in seq if r['kind']=='memcpy' and kinds[r['copyKind']]=='Device-to-Device']
        gdn=[r for r in d2d if r['bytes']==176160768]
        require(len(gdn)==96,'Expected common GDN output copies')
        gdn_proof=[]
        for r in gdn:
            i=seq.index(r)
            require('index_put_kernel' in seq[i-1]['name'],'GDN state write expected before output')
            require('direct_copy_kernel_cuda' in seq[i+1]['name'] and
                    seq[i+2]['name']=='layer_norm_fwd_kernel','GDN norm/output sequence missing')
            require(any('CPDeltaRulePrefill' in x['name'] for x in seq[max(0,i-5):i]),'No preceding GDN')
            gdn_proof.append(dict(copy=brief(r),preceding=[brief(x) for x in seq[i-4:i]],
                                 following=[brief(x) for x in seq[i+1:i+3]]))
        buckets=defaultdict(list)
        for r in d2d: buckets[r['bytes']].append(r)
        direct=[r for r in seq if r['kind']=='kernel' and 'direct_copy_kernel_cuda' in r['name']]
        result[arm]=dict(d2d=statistics(d2d),d2d_by_size={str(n):statistics(v) for n,v in buckets.items()},
            gdn_output_d2d=statistics(gdn),gdn_output_trace_proofs=gdn_proof,
            other_d2d=[brief(r) for r in d2d if r['bytes']!=176160768],
            direct_copy=statistics(direct),direct_copy_by_name={n:statistics([r for r in direct if r['name']==n])
                for n in sorted({r['name'] for r in direct})})
    gs={a: [r for r in rows if r['kind']=='kernel' and 'gemm' in r['name'].lower()] for a,rows in raw.items()}
    require(len(gs['native'])==len(gs['kvmem'])==640,'GEMM launch count mismatch')
    equal=[];different=[]
    for step in range(2):
        left=[r for r in gs['native'] if r['step']==step];right=[r for r in gs['kvmem'] if r['step']==step]
        for i,(l,r) in enumerate(zip(left,right,strict=True)):
            same=(l['name']==r['name'] and all(l[k]==r[k] for k in LAUNCH_GEOMETRY))
            (equal if same else different).append(dict(step=step,gemm_ordinal=i,
                native=brief(l),kvmem=brief(r),native_geometry={k:l[k] for k in LAUNCH_GEOMETRY},
                kvmem_geometry={k:r[k] for k in LAUNCH_GEOMETRY}))
    require(len(different)==128,'Unexpected number of layout-changing GEMMs')
    for pair in different:
        for arm in ['native','kvmem']:
            r=next(r for r in gs[arm] if r['correlationId']==pair[arm]['correlation_id'])
            kernels=[k for k in raw[arm] if k['kind']=='kernel'];j=kernels.index(r)
            require(kernels[j+1]['name']=='triton_poi_fused_mul_scaled_fp4_quant_silu_slice_2',
                    'Expected changed GEMM immediately preceding MLP SiLU/quantization')
    result['gemm_audit']=dict(actual_count=640,exact_name_and_launch_geometry_pairs=len(equal),
        changed_layout_pairs=len(different),changed_pairs=different,
        totals={a:statistics(v) for a,v in gs.items()},
        changed_totals={a:sum(p[a]['ms'] for p in different) for a in ['native','kvmem']},
        explanation='128 target MLP gate/up NVFP4 GEMMs immediately precede SiLU/FP4 quantization. Output layout and grid orientation differ; same LinearKernel classes do not prove identical binaries. Other 512 pairs match exact template/launch geometry.',
        linear_backend_lines={a:[line for line in (root/a/'server.log').read_text().splitlines()
            if ('Selected FlashInfer' in line or 'Using FlashInferCutlass' in line or 'GemmRunner' in line)]
            for a in ['native','kvmem']})
    sources=[Path(__file__),Path(__file__).with_name('analyze_matched_14k_nsight.py'),
        Path(__file__).with_name('compare_prefill_nsight.py'),wrapper]
    sources += [root/a/name for a in ['native','kvmem'] for name in ['trace.sqlite','server.log','plan.json']]
    gdn_source=Path(importlib.util.find_spec('vllm').origin).parent/'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
    sources.append(gdn_source)
    result['gdn_source']=dict(path=str(gdn_source),lines=[1537,1590],
        operation='GDN recurrent state update, output assignment core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0), then gated RMS norm. 176160768 bytes = 14336*48*128*BF16(2).')
    result['sources_sha256']={str(p.resolve()):digest(p) for p in sources}
    result['limitations']=['Single profiled two-step diagnostic, not throughput.',
        'Kernel duration sums are not additional wall time; categories can overlap with DMA.',
        'Common GDN/direct copies remain normal model work; removing them is a separate model optimization.',
        'The additional 146800640-byte copies occur in target/MTP input preparation; no exact Python source ownership is asserted solely from the size.',
        'GEMM output layout differs for128 calls; aggregate measured GEMM delta must remain separate from capture attribution.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--wrapper',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Use a fresh output path')
    a.output.write_text(json.dumps(analyze(a.root,a.wrapper),indent=2)+'\n')
