"""Actual static-aligned FA2 qualification; both arms share operands and plans."""
import math
import os
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.native_attention import NativeAttentionStats, fa2_overlay_capability

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                            reason='Explicit native static-aligned GPU qualification')
NAME='model.layers.3.attn'


def assert_bits(left,right):
    assert left.shape==right.shape and left.dtype==right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


def geometry(tmp_path,*,qlens=(257,),lengths=(2051,),prefix=0,page=16,
             heads=24,kvheads=4,split=None,graph_plan=False):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(7326)
    dim=256;total=sum(qlens);counts=[math.ceil(n/page) for n in lengths]
    physical=sum(counts)+7
    cache=(torch.randn(physical,kvheads,page,2*dim,device='cuda')*.3).to(torch.float8_e4m3fn)
    key,value=cache.transpose(1,2).split(dim,-1)
    table=torch.randperm(physical,device='cuda',dtype=torch.int32)[:sum(counts)]
    query=torch.randn(prefix+total,heads,dim,device='cuda',dtype=torch.bfloat16)
    qo=[0];kv=[0]
    for q,n in zip(qlens,counts):qo.append(qo[-1]+q);kv.append(kv[-1]+n)
    ints=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
    kwargs={}
    if graph_plan:
        kwargs=dict(use_cuda_graph=True,qo_indptr_buf=ints(qo),paged_kv_indptr_buf=ints(kv),
            paged_kv_indices_buf=table.clone(),
            paged_kv_last_page_len_buf=ints([(n-1)%page+1 for n in lengths]))
    # Explicit one-page split-K produces far more partitions than auto planning.
    workspace_bytes=(512 if split is not None else 256)<<20
    wrapper=BatchPrefillWithPagedKVCacheWrapper(torch.zeros(workspace_bytes,device='cuda',dtype=torch.uint8),
                                               'NHD',backend='fa2',**kwargs)
    wrapper.plan(ints(qo),ints(kv),table,ints([(n-1)%page+1 for n in lengths]),heads,kvheads,dim,page,
        causal=True,sm_scale=dim**-.5,q_data_type=query.dtype,kv_data_type=key.dtype,
        fixed_split_size=split)
    metadata=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=query.dtype,num_decode_tokens=prefix,
                num_prefills=len(qlens),num_prefill_tokens=sum(qlens))
    arms={}
    for gate in (False,True):
        class Impl:
            def forward(self,layer,q,k,v,cache,meta,out):
                wrapper.run(q[prefix:],(key,value),q_scale=1.,k_scale=.37,v_scale=.61,out=out[prefix:])
                return out
        impl=Impl();original=impl.forward
        layer=NS(num_heads=heads,head_dim=dim,q_norm=NS(weight=query),
                 attn=NS(layer_name=NAME,impl=impl,_q_scale_float=1.))
        runner=NS(model=NS(named_modules=lambda layer=layer:[(NAME,layer)]),_kvmem_layers=[NAME],
                  max_num_tokens=prefix+total,max_num_reqs=len(qlens)+2)
        options=NS(action_native_scores='reuse',action_layers='all',action_native_max_rows=total,
            action_native_max_bytes=2<<30,action_native_prefill_aligned_pieces=False,
            action_native_prefill_cta_gate=False,action_native_prefill_static_aligned=gate,
            memory=NS(select_budget=max(lengths)+32,gen_budget=128,block_tokens=32))
        bank=NativeAttentionStats(runner,options,tmp_path/str(gate))
        assert bank.disabled is None,bank.disabled
        output=torch.zeros_like(query)
        def execute(impl=impl,layer=layer,output=output):
            return impl.forward(layer.attn,query,None,None,None,metadata,output)
        def baseline(original=original,layer=layer):
            return original(layer.attn,query,None,None,None,metadata,torch.zeros_like(query))
        arms[gate]=NS(bank=bank,execute=execute,baseline=baseline,output=output)
    return NS(arms=arms,query=query,wrapper=wrapper,qlens=qlens,lengths=lengths,prefix=prefix,
              qo=qo,heads=heads,kvheads=kvheads)


def plans(env,rows,*,partial=False,owners=None):
    result={}
    for i,(qlen,length,sampled) in enumerate(zip(env.qlens,env.lengths,rows)):
        starts=([0,32,37] if partial else [0,32])+list(range(64,length,32))
        counts=[b-a for a,b in zip(starts,starts[1:]+[length])]
        result[str(i)]=dict(index=(owners[i] if owners else i+int(bool(env.prefix))),
            begin=env.prefix+env.qo[i],rows=list(sampled),starts=starts,counts=counts,
            local_begin=length-qlen,count=qlen)
    return result


def run(arm,current,*,execute=None):
    arm.bank.lses[NAME].fill_(float('nan'))
    arm.bank.arm(current)
    (execute or arm.execute)()
    masses={rid:arm.bank.score(NAME,rid,True).clone() for rid,p in current.items() if p['rows']}
    assert all(value is not None for value in masses.values()),arm.bank.reasons
    return masses,arm.output.clone(),arm.bank.lses[NAME].clone()


def compare_pair(env,current):
    reference=run(env.arms[False],current)
    candidate=run(env.arms[True],current)
    for rid in reference[0]:assert_bits(candidate[0][rid],reference[0][rid])
    assert_bits(candidate[1],reference[1]);assert_bits(candidate[2],reference[2])
    assert_bits(candidate[1],env.arms[False].baseline())
    assert env.arms[False].bank.allocated_bytes==env.arms[True].bank.allocated_bytes
    assert not any(k.startswith('fallback:') for arm in env.arms.values() for k in arm.bank.counts)
    destination=os.getenv('KVMEM_STATIC_ALIGNED_PROOF')
    if destination:
        evidence=dict(qlens=env.qlens,lengths=env.lengths,prefix=env.prefix,
            plan_info=list(env.wrapper._plan_info),bitwise_outputs_scores_lse=True,
            arms={str(enabled):dict(counts=dict(arm.bank.static_prefill_counts),
                generic_source=arm.bank.source_fingerprint,
                static_source=arm.bank.static_prefill_source_fingerprint,
                static_unavailable=arm.bank.static_prefill_unavailable,
                allocated_bytes=arm.bank.allocated_bytes) for enabled,arm in env.arms.items()})
        with Path(destination).open('a') as stream:stream.write(json.dumps(evidence)+'\n')


@pytest.mark.parametrize('pattern',['none','all','sparse','boundary'])
def test_partial_final_page_and_gqa_rows_match_bitwise(tmp_path,pattern):
    env=geometry(tmp_path)
    boundary=env.wrapper._plan_info[3]//(env.heads//env.kvheads)
    rows={'none':[], 'all':range(257),'sparse':range(31,257,32),
          'boundary':[0,boundary-1,boundary,boundary+1,256]}[pattern]
    compare_pair(env,plans(env,[rows]))
    stats=env.arms[True].bank.static_prefill_counts
    if rows:assert stats['static_dispatch_calls']>0
    else:assert stats['generic_dispatch:no_sampled_frame']>0


@pytest.mark.parametrize('heads,kvheads,qlen,page',[(24,4,2048,16),(8,8,257,32)])
def test_actual_2k_or_mha_and_larger_page_match_bitwise(tmp_path,heads,kvheads,qlen,page):
    env=geometry(tmp_path,qlens=(qlen,),lengths=(4099,),heads=heads,kvheads=kvheads,page=page)
    compare_pair(env,plans(env,[range(31,qlen,32)]))
    assert env.arms[True].bank.static_prefill_counts['static_dispatch_calls']>0


def test_ragged_multiowner_with_unsampled_peer_uses_exact_owner_table(tmp_path):
    env=geometry(tmp_path,qlens=(129,161),lengths=(2051,3079))
    compare_pair(env,plans(env,[[0,10,31,128],[]]))
    assert env.arms[True].bank.static_prefill_counts['static_dispatch_calls']>0
    mixed=plans(env,[[0,10,31,128],[]]);mixed['1']=plans(env,[[],[]],partial=True)['1']
    compare_pair(env,mixed)
    assert env.arms[True].bank.static_prefill_counts['generic_dispatch:unaligned_logical_segments']>0


@pytest.mark.parametrize('page,reason',[(16,None),(24,'unaligned_kv_page')])
def test_actual_split_k_uses_static_only_when_page_proves_all_split_boundaries(tmp_path,page,reason):
    env=geometry(tmp_path,qlens=(129,),lengths=(259,),page=page,split=1)
    assert env.wrapper._plan_info[14], 'Must exercise actual split-K attention'
    compare_pair(env,plans(env,[[0,31,63,128]]))
    stats=env.arms[True].bank.static_prefill_counts
    assert stats['static_dispatch_calls' if reason is None else 'generic_dispatch:'+reason]>0


@pytest.mark.parametrize('case,reason',[
    ('partial','unaligned_logical_segments'),('mixed','noncontiguous_owner_frame'),
    ('small','small_prefill'),('owner','noncontiguous_owner_frame')])
def test_ineligible_call_keeps_generic_native_scores(tmp_path,case,reason):
    if fa2_overlay_capability(torch.cuda.get_device_capability()) and case == 'mixed':
        pytest.skip('decode-prefix rows come from the SM12x XQA producer; an FA2-only GPU (SM80, SM89 opt-in) needs an FA2 decode wrapper for them and declines the whole batch by design')
    qlen=5 if case=='small' else 129
    env=geometry(tmp_path,qlens=(qlen,),lengths=(259,),prefix=3 if case=='mixed' else 0)
    current=plans(env,[[0,qlen-1]],partial=case=='partial',owners=(2,) if case=='owner' else None)
    compare_pair(env,current)
    stats=env.arms[True].bank.static_prefill_counts
    assert stats['generic_dispatch:'+reason]>0
    assert not stats['static_dispatch_calls']


def test_graph_always_uses_generic_across_rows_segments_owner_changes(tmp_path):
    env=geometry(tmp_path)
    arm=env.arms[True]
    initial=plans(env,[[0,31,256]])
    compare_pair(env,initial)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):arm.execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):arm.execute()
    assert arm.bank.static_prefill_counts['generic_dispatch:cuda_graph_capture']>0
    before=arm.bank.static_prefill_counts['static_dispatch_calls']
    pointers=tuple(t.data_ptr() for t in (arm.bank.rows,arm.bank.requests,arm.bank.pieces,arm.bank.lses[NAME]))
    for current in (plans(env,[[10,11,31,63]],partial=True,owners=(2,)),
                    plans(env,[[0,256]]),plans(env,[[]])):
        env.query.mul_(.75)
        expected=run(env.arms[False],current)
        actual=run(arm,current,execute=graph.replay)
        for rid in expected[0]:assert_bits(actual[0][rid],expected[0][rid])
        assert_bits(actual[1],expected[1]);assert_bits(actual[2],expected[2])
        assert pointers==tuple(t.data_ptr() for t in (arm.bank.rows,arm.bank.requests,arm.bank.pieces,arm.bank.lses[NAME]))
    assert arm.bank.static_prefill_counts['static_dispatch_calls']==before
    arm.bank.disarm();arm.bank.lses[NAME].fill_(float('nan'));graph.replay()
    assert torch.isnan(arm.bank.lses[NAME]).all().item()


def test_graph_planned_split_and_invalid_padded_ctas_keep_generic(tmp_path):
    env=geometry(tmp_path,qlens=(5,),lengths=(259,),page=24,split=1,graph_plan=True)
    info=env.wrapper._plan_info
    assert info[13] and info[14]
    mask=env.wrapper._int_workspace_buffer[info[12]:info[12]+info[0]].view(torch.bool)
    assert mask.any().item() and (~mask).any().item()
    compare_pair(env,plans(env,[[0,4]],partial=True))
    assert env.arms[True].bank.static_prefill_counts['generic_dispatch:graph_or_unknown_wrapper']>0


def test_missing_or_nonfinite_native_producer_still_invalidates_frame(tmp_path):
    env=geometry(tmp_path,qlens=(129,),lengths=(259,))
    arm=env.arms[True];current=plans(env,[[0,31,128]])
    run(arm,current)
    assert arm.bank.static_prefill_counts['static_dispatch_calls']>0
    arm.bank.arm(current)
    assert arm.bank.score(NAME,'0',True) is None
    assert arm.bank.validation_failures==1
    arm.bank.arm(current);arm.execute();arm.bank.lses[NAME].fill_(float('nan'))
    assert arm.bank.score(NAME,'0',True) is None
    assert arm.bank.validation_failures==2
