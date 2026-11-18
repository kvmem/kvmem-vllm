"""Actual FA2 CTA-gate qualification; both arms share operands and plans."""
import math
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.native_attention import NativeAttentionStats, fa2_overlay_capability

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                            reason='Explicit native CTA-gate GPU qualification')
NAME='model.layers.3.attn'


def assert_bits(left,right):
    assert left.shape==right.shape and left.dtype==right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


def geometry(tmp_path,*,qlens=(257,),lengths=(2051,),prefix=0,page=16,aligned=False,
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
    wrapper=BatchPrefillWithPagedKVCacheWrapper(torch.zeros(256<<20,device='cuda',dtype=torch.uint8),
                                               'NHD',backend='fa2',**kwargs)
    wrapper.plan(ints(qo),ints(kv),table,ints([(n-1)%page+1 for n in lengths]),heads,kvheads,dim,page,
        causal=True,sm_scale=dim**-.5,q_data_type=query.dtype,kv_data_type=key.dtype,
        fixed_split_size=split)
    metadata=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=query.dtype,num_decode_tokens=prefix)
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
            action_native_max_bytes=2<<30,action_native_prefill_aligned_pieces=aligned,
            action_native_prefill_cta_gate=gate,
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


def plans(env,rows,*,partial=True,owners=None):
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


@pytest.mark.parametrize('aligned',[False,True])
@pytest.mark.parametrize('pattern',['none','all','sparse','boundary'])
def test_all_empty_sparse_and_gqa_boundary_ctas_match_bitwise(tmp_path,aligned,pattern):
    env=geometry(tmp_path,aligned=aligned)
    tile=env.wrapper._plan_info[3];group=env.heads//env.kvheads
    boundary=tile//group
    rows={'none':[], 'all':range(257),'sparse':range(31,257,32),
          'boundary':[0,boundary-1,boundary,boundary+1,256]}[pattern]
    compare_pair(env,plans(env,[rows],partial=not aligned))


def test_actual_2k_geometry_and_gqa_one(tmp_path):
    for heads,kvheads,qlen in [(24,4,2048),(8,8,257)]:
        env=geometry(tmp_path,qlens=(qlen,),lengths=(4099,),heads=heads,kvheads=kvheads)
        compare_pair(env,plans(env,[range(31,qlen,32)]))


@pytest.mark.parametrize('aligned',[False,True])
def test_ragged_multiple_owners_and_decode_prefix_use_exact_global_rows(tmp_path,aligned):
    if fa2_overlay_capability(torch.cuda.get_device_capability()):
        pytest.skip('decode-prefix rows come from the SM12x XQA producer; an FA2-only GPU (SM80, SM89 opt-in) needs an FA2 decode wrapper for them and declines the whole batch by design')
    env=geometry(tmp_path,qlens=(65,97),lengths=(2051,3079),prefix=3,aligned=aligned)
    compare_pair(env,plans(env,[[0,10,11,31,64],[0,20,21,63,96]],partial=not aligned,owners=(2,1)))


@pytest.mark.parametrize('aligned',[False,True])
def test_graph_replay_empty_to_sampled_to_empty_and_owner_replacement(tmp_path,aligned):
    env=geometry(tmp_path,aligned=aligned)
    arm=env.arms[True]
    empty=plans(env,[[]]);run(arm,empty)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):arm.execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):arm.execute()
    pointers=tuple(t.data_ptr() for t in (arm.bank.rows,arm.bank.requests,arm.bank.pieces,arm.bank.lses[NAME]))
    for current in (plans(env,[[10,11,31,63]],owners=(2,)),plans(env,[[0,256]],partial=False),empty):
        env.query.mul_(.75)
        expected=run(env.arms[False],current)
        actual=run(arm,current,execute=graph.replay)
        for rid in expected[0]:assert_bits(actual[0][rid],expected[0][rid])
        assert_bits(actual[1],expected[1]);assert_bits(actual[2],expected[2])
        assert pointers==tuple(t.data_ptr() for t in (arm.bank.rows,arm.bank.requests,arm.bank.pieces,arm.bank.lses[NAME]))
    arm.bank.disarm();arm.bank.lses[NAME].fill_(float('nan'));graph.replay()
    assert torch.isnan(arm.bank.lses[NAME]).all().item()


def test_split_k_unaligned_chunks_and_padded_invalid_ctas_do_not_deadlock(tmp_path):
    env=geometry(tmp_path,qlens=(5,),lengths=(259,),page=24,split=1,graph_plan=True)
    info=env.wrapper._plan_info
    assert info[13] and info[14], 'Must use CUDA-graph planned split KV'
    mask=env.wrapper._int_workspace_buffer[info[12]:info[12]+info[0]].view(torch.bool)
    assert mask.any().item() and (~mask).any().item(), 'Must execute both valid and padded invalid CTAs'
    compare_pair(env,plans(env,[[0,4]]))


def test_missing_or_nonfinite_producer_still_invalidates_frame(tmp_path):
    env=geometry(tmp_path,qlens=(33,),lengths=(259,))
    arm=env.arms[True];current=plans(env,[[0,31,32]])
    run(arm,current)
    arm.bank.arm(current)
    assert arm.bank.score(NAME,'0',True) is None
    assert arm.bank.validation_failures==1
    arm.bank.arm(current);arm.execute();arm.bank.lses[NAME].fill_(float('nan'))
    assert arm.bank.score(NAME,'0',True) is None
    assert arm.bank.validation_failures==2
