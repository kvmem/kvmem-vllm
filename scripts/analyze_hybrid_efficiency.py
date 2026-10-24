"""Compare completed, matched shape pilots without treating quality as a gate."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from benchmark_niah import save,sha
from graph_v2_protocol import require
from run_hybrid_suite_queue import completed


def read(path):return json.loads(Path(path).read_text())


def scoring_allocation(campaign,query,fps):
    """Interpret missing old dense-path counters only for the verified source.

    That implementation caches an entire layer when its conservative scratch
    estimate fits 512 MiB; only its tiled branch emitted execution counters.
    Keep reconstructed bounds distinct from observed CUDA allocator peaks.
    """
    path,bound=fps.get('scoring_path'),fps.get('scoring_scratch_bound_bytes')
    if path is not None or bound is not None:
        require(path is not None and type(bound) is int and bound>0,
            'Partial or invalid runtime scoring allocation counters')
        return dict(path=path,bound_bytes=bound,proof=dict(kind='runtime_reported'))
    name='src/vllm_kvmem/query_key_fps.py'
    supported='451e9fc3048823765471e7f91dd4e4ca45910327c8017305552493f421ef9b68'
    recorded=read(campaign/'runtime-proof.json')['files'].get(name)
    require(recorded==supported and sha(campaign/'runtime'/name)==supported,
        'Missing scoring counters from an unverified frozen implementation')
    profile=read(campaign/'profile.json');k=profile['kvmem']
    require(k.get('query_fps_optimized') is True and fps.get('optimized') is True and
        fps.get('cache_key_dtype')=='torch.float16' and fps.get('representatives_per_block')==4 and
        fps.get('scoring_device','').startswith('cuda:') and query['query_score_tokens']>0,
        'Dense scoring proof lacks original dtype/device/optimization geometry')
    config=Path(profile.get('hf_config_path') or profile['model'])
    if config.is_dir():config=config/'config.json'
    model=read(config);model=model.get('text_config',model)
    heads,kvheads,dim=(model[n] for n in ('num_attention_heads','num_key_value_heads','head_dim'))
    blocks=fps['cache_hit_blocks']+fps['cache_rebuilt_blocks']
    require(all(type(v) is int and v>0 for v in (heads,kvheads,dim,blocks)) and
        heads%kvheads==0 and 0<query['score_mask_count']<=blocks,'Invalid dense GQA/block domain')
    slots,chunk,key_bytes=4,8,2
    # Exact upper-bound expression and default Query chunk in the hash-bound
    # old scorer. Layer count and total Query length do not multiply scratch.
    scratch=(2*blocks*slots*kvheads*dim*4+3*chunk*heads*blocks*slots*4+
        3*chunk*heads*blocks*4+2*chunk*heads*dim*4+blocks*slots*kvheads*2+
        blocks*8+blocks*slots*kvheads*dim*key_bytes)
    require(scratch<=512*1024**2,'Missing counters cannot be explained by the bounded dense branch')
    return dict(path='dense_layer_cache',bound_bytes=scratch,proof=dict(
        kind='derived_from_frozen_source_and_recorded_geometry',source_sha256=supported,
        model_config_sha256=sha(config),blocks=blocks,query_heads=heads,kv_heads=kvheads,
        head_dim=dim,representatives=slots,query_chunk=chunk,key_element_bytes=key_bytes,
        runtime_counter_missing=True,measured_peak=False))


def case_metrics(path):
    v=read(path/'result.json');q=v['graph_v2']['query'];fps=q['query_key_fps_scoring']
    graph=[g for g in v['graph_v2']['graph'] if g['phase']=='query']
    require(len(graph)==1,'Missing unique Query graph event')
    graph=graph[0];metrics=v['metrics'];usage=v['usage']
    branches=graph.get('query_quota_propagation',{})
    compute={key:branches[key]['scoring_seconds'] for key in ('mean_k','key_fps_4') if key in branches}
    allocation=scoring_allocation(path.parents[3],q,fps)
    return dict(input_tokens=v['prompt_tokens'],query_tokens=q['query_score_tokens'],
        replay_tokens=q['query_tokens'],prefix_reused=v['prefix_reused'],
        request_seconds=v['seconds'],ttft_seconds=metrics['time_to_first_token_ms']/1000,
        decode_seconds=metrics['generation_time_ms']/1000,
        mean_itl_ms=metrics['mean_itl_ms'],
        query_refresh_seconds=q['refresh_finished_at']-q['refresh_started_at'],
        fps_cache_build_seconds=fps['cache_build_seconds'],fps_scoring_seconds=fps['scoring_seconds'],
        mean_scoring_seconds=fps['mean_quota_scoring_seconds'],
        graph_selection_seconds=graph['graph_selection_seconds'],graph_branch_seconds=compute,
        graph_branch_backends={key:branches[key]['backend'] for key in compute},
        assembly_seconds=q['assembly_seconds'],
        cache_hit_blocks=fps['cache_hit_blocks'],cache_prebuilt_hit_blocks=fps['cache_prebuilt_hit_blocks'],
        cache_rebuilt_blocks=fps['cache_rebuilt_blocks'],scoring_path=allocation['path'],
        cache_bytes=fps['cache_bytes'],cache_capacity_bytes=fps['cache_capacity_bytes'],
        scoring_scratch_bound_bytes=allocation['bound_bytes'],scoring_allocation_proof=allocation['proof'],
        output_tokens=usage['completion_tokens'],finish_reason=v['finish_reason'],
        output_text_sha256=hashlib.sha256(v['raw_completion'].encode()).hexdigest(),
        mtp=metrics.get('speculative_decoding'),request=read(path/'request.json'),
        result_sha256=sha(path/'result.json'))


def compare(root,*,output=None):
    off,on=root/'graph-off',root/'graph-on'
    for p in (off,on):completed(p)
    require(read(off/'runtime-proof.json')==read(on/'runtime-proof.json'),'Pilot engine/client sources differ')
    a,b=read(off/'profile.json'),read(on/'profile.json')
    require(a['kvmem']['graph_alpha']==0 and b['kvmem']['graph_alpha']==.2,'Pilot alpha pair differs')
    a['kvmem']['graph_alpha']=.2
    require(a==b,'Pilot profiles differ beyond graph alpha')
    fa,fb=read(off/'fixture.json'),read(on/'fixture.json')
    require(fa['case_ids']==fb['case_ids'] and fa['rows']==fb['rows'],'Original matched cohort differs')
    comparisons=[]
    for ident in fa['case_ids']:
        left=case_metrics(off/'run/answers/cases'/ident);right=case_metrics(on/'run/answers/cases'/ident)
        before,after=deepcopy(left['request']),deepcopy(right['request'])
        for r in (before,after):
            r.pop('session_id',None);r.pop('cache_salt',None)
        require(before==after,'Original input/seed/Query/prefix/sampling/output limit changed: '+ident)
        require(left['prefix_reused']==right['prefix_reused'],'Paired cache lifecycle differs')
        require(set(right['graph_branch_backends'])=={'mean_k','key_fps_4'} and
            set(right['graph_branch_backends'].values())=={'cpp_v2'},'Unexpected graph reference fallback: '+ident)
        comparisons.append(dict(id=ident,off=left,on=right,
            output_changed=left['output_text_sha256']!=right['output_text_sha256'],
            delta_seconds={k:right[k]-left[k] for k in ('ttft_seconds','query_refresh_seconds',
                'fps_cache_build_seconds','fps_scoring_seconds','mean_scoring_seconds',
                'graph_selection_seconds','decode_seconds','request_seconds')}))
    phases=('ttft_seconds','query_refresh_seconds','fps_scoring_seconds','graph_selection_seconds')
    summary={k:{arm:statistics.median(v[arm][k] for v in comparisons) for arm in ('off','on')} for k in phases}
    report=dict(status='requires_efficiency_review_before_full',cases=len(comparisons),
        matched_inputs_seeds_sampling=True,same_frozen_sources=True,only_profile_change='graph_alpha 0 -> 0.2',
        no_accuracy_gate=True,per_case=comparisons,median_seconds=summary,
        inference_audit_sha256={p.name:sha(p/'execution-audit.json') for p in (off,on)},
        full_campaigns_started=False,
        scope='Single off/on pass, identical finite shape cohort and original per-case sampling. '
            'Query intervals include host work/queueing/replay; graph_selection includes both propagation and native quota/picker plus full diagnostic export. '
            'GPU scoring wall intervals are not pure kernel timings. Compare decode with output length/MTP separately; generated trajectories may differ. '
            'Warm-prefix requests do not prefill their entire served prompt again. No total-speedup claim from per-request timing alone.')
    save(output if output is not None else root/'comparison.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=compare(args.root.resolve(),output=args.output)
    print(json.dumps(dict(cases=result['cases'],median_seconds=result['median_seconds']),indent=2))
