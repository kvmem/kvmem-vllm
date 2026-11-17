"""Opt-in real FA2 qualification; no model launch or throughput claim."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

import test_native_attention_stats_gpu as existing
from vllm_kvmem.serving.native_attention import NativeAttentionStats, fa2_overlay_capability

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit aligned FA2 CUDA qualification')


def geometry(monkeypatch,tmp_path,*,enabled=True,qlen=5,page=16):
    original=existing.NativeAttentionStats
    def configured(runner,options,output):
        options.action_native_prefill_aligned_pieces=enabled
        return original(runner,options,output)
    with monkeypatch.context() as patch:
        patch.setattr(existing,'NativeAttentionStats',configured)
        return existing.geometry(tmp_path,'prefill',qlen=qlen,dim=256,heads=24,
                                 kvheads=4,page=page,scale=.37)


def segments(plan,aligned):
    starts=list(range(0,259,32)) if aligned else [0,32,37]+list(range(64,259,32))
    return dict(plan,starts=starts,counts=[b-a for a,b in zip(starts,starts[1:]+[259])])


@pytest.mark.parametrize('aligned,qlen',[(True,5),(False,5),(True,32)])
def test_old_module_new_generic_and_new_aligned_match_exactly(monkeypatch,tmp_path,aligned,qlen):
    old=geometry(monkeypatch,tmp_path/'old',enabled=False,qlen=qlen)
    new=geometry(monkeypatch,tmp_path/'new',enabled=True,qlen=qlen)
    results=[]
    for item,force_generic in [(old,False),(new,True),(new,False)]:
        bank,plan,execute,_,_,meta,output,*_=item
        plan=segments(plan,aligned)
        bank.arm({'r':plan})
        if force_generic: bank.aligned_pieces.zero_()
        execute()
        masses=bank.score('model.layers.3.attn','r',True)
        assert masses is not None,bank.reasons
        torch.testing.assert_close(output,meta.baseline(),rtol=0,atol=0)
        results.append((masses.clone(),output.clone()))
    for actual in results[1:]:
        assert torch.equal(actual[0],results[0][0])
        assert torch.equal(actual[1],results[0][1])
    assert bool(new[0].aligned_pieces[0].item())==aligned


def test_same_graph_changes_alignment_owner_index_and_disarms_without_stale_lse(monkeypatch,tmp_path):
    bank,plan,execute,_,q,_,output,*_=geometry(monkeypatch,tmp_path)
    name='model.layers.3.attn'
    aligned=segments(plan,True)
    bank.arm({'first':aligned});execute();assert bank.score(name,'first',True) is not None
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): execute()
    pointers=(bank.aligned_pieces.data_ptr(),bank.rows.data_ptr(),bank.ranges.data_ptr(),bank.lses[name].data_ptr())
    for owner,index,alignment in [('aligned',0,True),('partial',0,False),('replacement',1,True)]:
        q.mul_(.75)
        current=dict(segments(plan,alignment),index=index)
        # New binary's generic path is the same replay shape and arithmetic.
        bank.arm({owner:current});bank.aligned_pieces.zero_();execute()
        expected=bank.score(name,owner,True).clone();expected_output=output.clone()
        bank.arm({owner:current});graph.replay()
        actual=bank.score(name,owner,True)
        assert torch.equal(actual,expected)
        assert torch.equal(output,expected_output)
        assert bank.aligned_pieces.tolist()==([int(alignment),0] if index==0 else [0,1])
        assert pointers==(bank.aligned_pieces.data_ptr(),bank.rows.data_ptr(),bank.ranges.data_ptr(),bank.lses[name].data_ptr())
    bank.disarm();bank.lses[name].fill_(float('nan'));graph.replay()
    assert bank.aligned_pieces.tolist()==[0,0]
    assert torch.isnan(bank.lses[name]).all().item()


def mixed_geometry(tmp_path):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(741)
    page,dim,heads,kvheads,length,qlen,prefix=16,256,24,4,259,5,2
    cache=(torch.randn(22,kvheads,page,2*dim,device='cuda')*.4).to(torch.float8_e4m3fn)
    key,value=cache.transpose(1,2).split(dim,-1)
    query=torch.randn(prefix+2*qlen,heads,dim,device='cuda',dtype=torch.bfloat16)
    output=torch.zeros_like(query)
    tables=torch.stack([torch.randperm(22,device='cuda',dtype=torch.int32)[:17] for _ in range(2)])
    wrapper=BatchPrefillWithPagedKVCacheWrapper(torch.zeros(32<<20,device='cuda',dtype=torch.uint8),'NHD',backend='fa2')
    wrapper.plan(torch.tensor([0,5,10],device='cuda',dtype=torch.int32),
        torch.tensor([0,17,34],device='cuda',dtype=torch.int32),tables.flatten(),
        torch.tensor([3,3],device='cuda',dtype=torch.int32),heads,kvheads,dim,page,
        causal=True,sm_scale=dim**-.5,q_data_type=query.dtype,kv_data_type=key.dtype)
    class Impl:
        def forward(self,layer,q,k,v,cache,meta,out):
            wrapper.run(q[prefix:],(key,value),q_scale=1.,k_scale=.7,v_scale=.3,out=out[prefix:])
            return out
    impl=Impl()
    layer=NS(num_heads=heads,head_dim=dim,q_norm=NS(weight=query),
             attn=NS(layer_name='model.layers.3.attn',impl=impl,_q_scale_float=1.))
    runner=NS(model=NS(named_modules=lambda:[('model.layers.3.attn',layer)]),
              _kvmem_layers=['model.layers.3.attn'],max_num_tokens=64,max_num_reqs=3)
    options=NS(action_native_scores='reuse',action_layers='all',action_native_max_rows=32,
        action_native_max_bytes=64<<20,action_native_prefill_aligned_pieces=True,
        memory=NS(select_budget=512,gen_budget=128,block_tokens=32))
    bank=NativeAttentionStats(runner,options,tmp_path)
    meta=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=query.dtype,num_decode_tokens=prefix)
    plans={}
    for i,aligned in enumerate((True,False)):
        plans[str(i)]=segments(dict(index=i+1,begin=prefix+i*qlen,rows=list(range(qlen)),
                                   local_begin=length-qlen,count=qlen),aligned)
    return bank,plans,lambda:impl.forward(layer.attn,query,None,None,None,meta,output),output


def test_mixed_decode_prefix_two_prefill_owners_use_global_requests_mapping(tmp_path):
    if fa2_overlay_capability(torch.cuda.get_device_capability()):
        pytest.skip('decode-prefix rows come from the SM12x XQA producer; an FA2-only GPU (SM80, SM89 opt-in) needs an FA2 decode wrapper for them and declines the whole batch by design')
    bank,plans,execute,output=mixed_geometry(tmp_path)
    name='model.layers.3.attn'
    bank.arm(plans)
    assert bank.aligned_pieces.tolist()==[0,1,0]
    assert bank.requests[:10].tolist()==[1]*5+[2]*5
    bank.aligned_pieces.zero_();execute()
    expected={rid:bank.score(name,rid,True).clone() for rid in plans}
    expected_output=output.clone()
    bank.arm(plans);execute()
    actual={rid:bank.score(name,rid,True) for rid in plans}
    assert all(torch.equal(actual[rid],expected[rid]) for rid in plans)
    assert torch.equal(output,expected_output)


def test_nonfinite_native_stats_still_reject_whole_frame(monkeypatch,tmp_path):
    bank,plan,execute,*_=geometry(monkeypatch,tmp_path)
    bank.arm({'r':segments(plan,True)});execute()
    bank.lses['model.layers.3.attn'].fill_(float('nan'))
    assert bank.score('model.layers.3.attn','r',True) is None
    assert bank.validation_failures==1


def test_unaligned_split_k_chunk_keeps_original_merge_path(monkeypatch,tmp_path):
    # A 24-token page with split_size=1 makes alternating split-K starts
    # unaligned to 16. Even aligned tile bases must retain CAS when merge=True.
    bank,plan,execute,_,q,meta,_,(key,_),table=geometry(monkeypatch,tmp_path,page=24)
    wrapper=meta.prefill.wrapper
    wrapper.plan(torch.tensor([0,5],device='cuda',dtype=torch.int32),
        torch.tensor([0,11],device='cuda',dtype=torch.int32),table[0,:11],
        torch.tensor([19],device='cuda',dtype=torch.int32),24,4,256,24,causal=True,
        sm_scale=256**-.5,q_data_type=q.dtype,kv_data_type=key.dtype,fixed_split_size=1)
    assert wrapper._plan_info[14], 'The qualification must exercise split-K CAS merging'
    current=segments(plan,True)
    bank.arm({'r':current});bank.aligned_pieces.zero_();execute()
    expected=bank.score('model.layers.3.attn','r',True).clone()
    bank.arm({'r':current});execute()
    actual=bank.score('model.layers.3.attn','r',True)
    assert bank.aligned_pieces[0].item()==1
    assert torch.equal(actual,expected)
