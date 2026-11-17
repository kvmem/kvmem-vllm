"""Real native FA2/XQA QK reuse, including graph replay and partial MTP acceptance."""
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.serving.native_attention import NativeAttentionStats, fa2_overlay_capability

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
                                reason='opt-in native attention CUDA qualification')


def geometry(tmp_path, phase, qlen=5, dim=128, heads=8, kvheads=2, scale=.7, page=32, qscale=1.):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper, xqa_batch_decode_with_kv_cache
    if phase == 'decode' and torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('This fixture exercises the SM12x XQA decode producer')
    torch.manual_seed(192)
    length = 259
    physical = (length+page-1)//page + 2
    cache = (torch.randn(physical, kvheads, page, 2*dim, device='cuda')*.4).to(torch.float8_e4m3fn)
    k, v = cache.transpose(1, 2).split(dim, -1)
    table = torch.randperm(physical, device='cuda', dtype=torch.int32)[None]
    q = torch.randn(qlen, heads, dim, device='cuda', dtype=torch.bfloat16)
    workspace = torch.zeros(32*1024*1024, device='cuda', dtype=torch.uint8)
    lengths = torch.tensor([length], device='cuda', dtype=torch.int32)
    mask = torch.tensor([[((1 << (row+1))-1) & 65535, ((1 << (row+1))-1) >> 16]
                         for row in range(qlen)], device='cuda', dtype=torch.uint16) if phase=='decode' and qlen>1 else None
    wrapper = None
    if phase == 'prefill':
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD', backend='fa2')
        qo = torch.tensor([0, qlen], device='cuda', dtype=torch.int32)
        kv = torch.tensor([0, (length+page-1)//page], device='cuda', dtype=torch.int32)
        wrapper.plan(qo, kv, table[0, :kv[-1].item()], torch.tensor([length%page], device='cuda', dtype=torch.int32),
            heads, kvheads, dim, page, causal=True, sm_scale=dim**-.5,
            q_data_type=q.dtype, kv_data_type=k.dtype)
    elif phase == 'fa2-decode':
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD', use_tensor_cores=True, backend='fa2')
        pages = (length+page-1)//page
        wrapper.plan(torch.tensor([0, pages], device='cuda', dtype=torch.int32), table[0, :pages],
            torch.tensor([length%page], device='cuda', dtype=torch.int32), heads, kvheads, dim, page,
            sm_scale=dim**-.5, q_data_type=q.dtype, kv_data_type=k.dtype, q_len_per_req=qlen)

    class Impl:
        def forward(self, layer, query, key, value, kv_cache, meta, output):
            if wrapper is not None:
                return wrapper.run(query, (k, v), q_scale=qscale, k_scale=scale, v_scale=.3, out=output)
            return xqa_batch_decode_with_kv_cache(query=query, kv_cache=(k,v), workspace_buffer=workspace,
                block_tables=table, seq_lens=lengths,
                max_seq_len=length, bmm1_scale=dim**-.5*scale, bmm2_scale=.3, kv_layout='NHD',
                out=output, q_len_per_req=qlen, mask=mask)

    impl = Impl()
    original = impl.forward
    layer = SimpleNamespace(num_heads=heads, head_dim=dim, q_norm=SimpleNamespace(weight=q),
        attn=SimpleNamespace(layer_name='model.layers.3.attn', impl=impl, _q_scale_float=qscale))
    runner = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [('model.layers.3.attn', layer)]),
        _kvmem_layers=['model.layers.3.attn'], max_num_tokens=64, max_num_reqs=2)
    options = SimpleNamespace(action_native_scores='reuse', action_layers='all', action_native_max_rows=32,
        action_native_max_bytes=64*1024*1024, memory=SimpleNamespace(select_budget=512, gen_budget=128, block_tokens=32))
    bank = NativeAttentionStats(runner, options, tmp_path)
    assert bank.disabled is None
    meta = SimpleNamespace(prefill=SimpleNamespace(wrapper=wrapper) if phase=='prefill' else None,
        decode=SimpleNamespace(wrapper=wrapper) if phase=='fa2-decode' else None,
        q_data_type_prefill=q.dtype, q_data_type_decode=q.dtype,
        num_decode_tokens=0 if phase=='prefill' else qlen, lengths=lengths)
    # Nonaligned logical segments, partial tail and noncontiguous physical pages.
    starts = [0, 32, 37] + list(range(64, length, 32))
    counts = [b-a for a,b in zip(starts, starts[1:]+[length])]
    rows = list(range(qlen))
    plan = dict(index=0, begin=0, rows=rows, starts=starts, counts=counts, local_begin=length-qlen, count=qlen)
    output = torch.empty_like(q)
    meta.baseline = lambda: original(layer.attn, q, None, None, None, meta, torch.empty_like(q))

    def execute():
        return impl.forward(layer.attn, q, None, None, None, meta, output)

    def oracle():
        visible_length = int(lengths[0].item())
        p = torch.arange(visible_length, device='cuda')
        keys = k[table[0, p//page].long(), p%page].float().repeat_interleave(heads//kvheads, 1)
        logits = torch.einsum('rhd,thd->rht', q.float(), keys) * (dim**-.5*scale*(qscale if phase=='prefill' else 1.))
        logits.masked_fill_(p[None,None] > (visible_length-qlen+torch.arange(qlen,device='cuda'))[:,None,None], -torch.inf)
        probs = logits.softmax(-1).mean(1)
        return torch.stack([probs[:,s:s+n].sum(-1) for s,n in zip(starts,counts)], -1)

    return bank, plan, execute, oracle, q, meta, output, (k,v), table


@pytest.mark.parametrize('phase,qlen', [('prefill',5), ('decode',1), ('decode',5)])
@pytest.mark.parametrize('dim', [128, 256])
def test_native_qk_side_output_matches_full_causal_gqa(tmp_path, phase, qlen, dim):
    bank, plan, execute, oracle, _, meta, output, *_ = geometry(tmp_path, phase, qlen, dim)
    bank.arm({'r':plan})
    execute()
    actual = bank.score('model.layers.3.attn', 'r', phase=='prefill')
    assert actual is not None, bank.reasons
    # Native operands/tensor-core arithmetic may differ from the FP32 oracle.
    torch.testing.assert_close(actual, oracle(), rtol=.02, atol=2e-4)
    torch.testing.assert_close(actual.sum(-1), torch.ones(qlen, device='cuda'), atol=2e-5, rtol=0)
    torch.testing.assert_close(output, meta.baseline(), rtol=0, atol=0)


@pytest.mark.parametrize('phase,qlen', [('prefill',32), ('decode',1), ('decode',5)])
def test_model_head_geometry_and_native_sixteen_token_pages(tmp_path, phase, qlen):
    bank, plan, execute, oracle, *_ = geometry(tmp_path, phase, qlen, 256, heads=24, kvheads=4, page=16)
    bank.arm({'r':plan}); execute()
    actual = bank.score('model.layers.3.attn', 'r', phase=='prefill')
    assert actual is not None, bank.reasons
    torch.testing.assert_close(actual, oracle(), rtol=.02, atol=2e-4)


def test_unquantized_prefill_q_scale_domain_mismatch_is_explicit_fallback(tmp_path):
    bank, plan, execute, _, _, meta, output, *_ = geometry(tmp_path,'prefill',5,qscale=.25)
    bank.arm({'r':plan});execute()
    assert bank.score('model.layers.3.attn','r',True) is None
    assert bank.counts['fallback:prefill:q_scale_domain_mismatch'] == 1
    torch.testing.assert_close(output,meta.baseline(),rtol=0,atol=0)


def test_sm80_mtp_can_consume_fa2_scores_only_for_a_covered_batch(tmp_path):
    if not fa2_overlay_capability(torch.cuda.get_device_capability()):
        pytest.skip('SM80 FA2 MTP verification (SM80, or SM89 with KVMEM_NATIVE_FA2_SM89=1)')
    bank, plan, execute, oracle, _, meta, _, *_ = geometry(tmp_path, 'prefill', 5)
    name = 'model.layers.3.attn'
    bank.arm({'r': plan})
    execute()
    torch.testing.assert_close(bank.score(name, 'r', False), oracle(), rtol=.02, atol=2e-4)
    # The next forward has no instrumented FA2 wrapper. An earlier MTP result
    # must not be used for these rows, even on the same request and layer.
    meta.prefill = None
    bank.arm({'r': plan})
    execute()
    assert bank.score(name, 'r', False) is None
    assert bank.counts['fallback:decode:backend_not_observed'] == 1


@pytest.mark.parametrize('qlen', [1, 5])
def test_sm80_tensor_core_decode_and_prefill_use_correct_packed_rows(tmp_path, qlen):
    if not fa2_overlay_capability(torch.cuda.get_device_capability()):
        pytest.skip('SM80 FA2 mixed-batch statistics (SM80, or SM89 with KVMEM_NATIVE_FA2_SM89=1)')
    _, dec_plan, _, dec_oracle, dec_q, dec_meta, *_ = geometry(tmp_path, 'fa2-decode', qlen)
    bank, pre_plan, _, pre_oracle, pre_q, pre_meta, *_ = geometry(tmp_path, 'prefill', 7)
    name = 'model.layers.3.attn'
    query = torch.cat([dec_q, pre_q])
    baseline = torch.cat([dec_meta.baseline(), pre_meta.baseline()])

    def mixed_forward(layer, q, key, value, kv_cache, metadata, out):
        out[:qlen].copy_(dec_meta.baseline())
        out[qlen:].copy_(pre_meta.baseline())
        return out

    impl = SimpleNamespace(forward=mixed_forward)
    bank._wrap(impl, name)
    meta = SimpleNamespace(prefill=pre_meta.prefill, decode=dec_meta.decode,
        q_data_type_prefill=query.dtype, q_data_type_decode=query.dtype, num_decode_tokens=qlen)
    bank.arm({'decode': dec_plan, 'prefill': dict(pre_plan, begin=qlen, index=1, rows=[1, 5])})
    actual = impl.forward(SimpleNamespace(_q_scale_float=1.), query, None, None, None, meta, torch.empty_like(query))
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(bank.score(name, 'decode', False), dec_oracle(), rtol=.02, atol=2e-4)
    torch.testing.assert_close(bank.score(name, 'prefill', True), pre_oracle()[[1, 5]], rtol=.02, atol=2e-4)


def test_deferred_missing_producer_cannot_publish_stale_graph_mass(tmp_path):
    bank,plan,execute,_,*_ = geometry(tmp_path,'decode',5)
    name = 'model.layers.3.attn'
    bank.arm({'r':plan});execute()
    assert bank.score(name,'r',False) is not None
    previous = bank.counts['reuse:decode']
    # Keep a previously observed backend, but omit the producer this frame.
    bank.arm({'r':plan})
    assert bank.score(name,'r',False,defer_validation=True) is not None
    assert not bank.validate()
    assert bank.counts['reuse:decode'] == previous
    assert bank.validation_failures == 1
    assert bank.score(name,'r',False,defer_validation=True) is None
    assert bank.counts['fallback:decode:invalid_native_mass'] == 1


def test_graph_many_tiny_segments_with_different_causal_quad_tails(tmp_path):
    bank,plan,execute,_,q,_,_,(k,_),table=geometry(tmp_path,'decode',5)
    starts=list(range(0,224,32))+list(range(224,259))
    counts=[b-a for a,b in zip(starts,starts[1:]+[259])]
    plan=dict(plan,starts=starts,counts=counts)
    bank.arm({'r':plan});execute()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):execute()
    bank.arm({'r':plan});graph.replay()
    actual=bank.score('model.layers.3.attn','r',False)
    positions=torch.arange(259,device='cuda')
    keys=k[table[0,positions//32].long(),positions%32].float().repeat_interleave(4,1)
    logits=torch.einsum('rhd,thd->rht',q.float(),keys)*128**-.5*.7
    logits.masked_fill_(positions[None,None]>(254+torch.arange(5,device='cuda'))[:,None,None],-torch.inf)
    probs=logits.softmax(-1).mean(1)
    expected=torch.stack([probs[:,s:s+n].sum(-1) for s,n in zip(starts,counts)],-1)
    assert actual is not None
    torch.testing.assert_close(actual,expected,rtol=.02,atol=2e-4)


@pytest.mark.parametrize('produce,interval',[(False,1),(True,16)])
def test_actual_decode_candidates_keep_exact_scope_and_safe_fallback(tmp_path,produce,interval):
    from vllm_kvmem.serving.action_attention import ActionHistory,ActionRuntime,action_metadata
    bank,plan,execute,oracle,q,_,_,(k,v),table = geometry(tmp_path,'decode',5)
    bank.arm({'r':plan});execute()  # observe a supported backend
    graph=None
    if produce:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):execute()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):execute()
    options=bank.options
    options.action_scope,options.action_interval,options.action_decay='non-thinking',interval,.5
    options.prefill_attention=True
    state=ActionHistory(options,[10,11,12,13,14]);state.begin('r',[14])
    if produce:
        state.parser.prepare([20]*13);state.parser.commit(13)
    action=ActionRuntime.__new__(ActionRuntime)
    action.options,action.ids=options,[10,11,12,13,14]
    action.runtime=SimpleNamespace(config=options.memory,output=tmp_path,
        drain_requests=lambda *a,**k:True,
        ensure_authority=lambda r:SimpleNamespace(action_history=state,block_graph=None))
    action.native,action.metadata,action.native_plans=bank,None,None
    action.capture_seconds=action.candidate_samples=action.packets=0
    action.metadata_early_batches=action.metadata_wait_seconds=0
    action._window_segments=lambda *unused:(plan['starts'],plan['counts'],plan['starts'])
    name='model.layers.3.attn'
    cache=torch.cat((k,v),-1).transpose(1,2)
    trig=torch.cat((torch.ones(259,32,device='cuda'),torch.zeros(259,32,device='cuda')),-1)
    module=SimpleNamespace(num_kv_heads=2,head_dim=128,rotary_emb=SimpleNamespace(cos_sin_cache=trig),
        attn=SimpleNamespace(layer_name=name,kv_cache=cache,impl=SimpleNamespace(scale=128**-.5),
                             _k_scale=torch.tensor(.7,device='cuda'),_q_scale=torch.ones((),device='cuda')))
    action.layers,action.groups=[(0,module)],{name:0}
    action.q_dtypes=action.prefill_q_dtypes={name:q.dtype}
    action.runner=SimpleNamespace(req_states=SimpleNamespace(req_id_to_index={'r':0}),
        block_tables=SimpleNamespace(block_tables=[SimpleNamespace(gpu=table)]))
    from vllm_kvmem.serving.native_attention_planning import make_decode_frame
    action.native_cache={'r':dict(generation=0,offset=0,
        **make_decode_frame(plan['starts'],plan['counts'],plan['starts'],254,254,5,bank.context,32))}
    positions=torch.arange(254,259,device='cuda')
    batch=SimpleNamespace(num_reqs=1,req_ids=['r'],is_prefilling_np=[False],query_start_loc_np=[0,5],
        num_computed_tokens_np=[254],seq_lens_cpu_upper_bound=[259],
        input_ids=torch.tensor([20,21,22,23,24],device='cuda'),positions=positions)
    # Exercise the concurrent-owner preparation algorithm with one small owner.
    action.metadata=action_metadata(batch,positions,True)
    bank.lses[name].fill_(1000.)  # future pieces must never contaminate this replay
    assert action._prepare_decode_candidates(batch)
    if produce:graph.replay()
    # Missing native forward must discard NaNs and recompute now. A valid
    # producer retains the original one-in-sixteen scope after collecting five.
    packet=action.capture([SimpleNamespace(query=q,positions=positions)],batch)['r']
    packet.event.synchronize()
    if produce:assert packet.rows==[3]
    torch.testing.assert_close(packet.masses,oracle()[packet.rows].cpu(),rtol=.02,atol=2e-4)
    assert bank.validation_failures==int(not produce)
    assert bank.counts['fallback:decode:invalid_native_mass']==int(not produce)
    assert action.candidate_samples==len(packet.rows) and action.packets==1
    state.accept(2,packet.rows,packet.history,packet.masses.tolist())
    assert state.samples==(2 if interval==1 else 0)  # sampled row 3 was rejected


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_graph_replay_reuses_stable_storage_and_clears_old_rows(tmp_path, phase):
    bank, plan, execute, oracle, q, meta, output, _, _ = geometry(tmp_path, phase, 5)
    bank.arm({'r':plan})
    execute()
    assert bank.score('model.layers.3.attn', 'r', phase=='prefill') is not None, bank.reasons
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    address = bank.lses['model.layers.3.attn'].data_ptr()
    q.mul_(-2.)
    plan = dict(plan, rows=[1, 4])
    bank.arm({'r':plan})
    graph.replay()
    actual = bank.score('model.layers.3.attn', 'r', phase=='prefill')
    assert actual is not None, bank.reasons
    torch.testing.assert_close(actual, oracle()[[1,4]], rtol=.02, atol=2e-4)
    assert bank.lses['model.layers.3.attn'].data_ptr() == address
    if phase == 'decode':
        meta.lengths.fill_(241)
        starts = [s for s in plan['starts'] if s < 241]
        counts = [b-a for a,b in zip(starts,starts[1:]+[241])]
        bank.arm({'new-owner':dict(plan,starts=starts,counts=counts,local_begin=236)})
        graph.replay()
        actual = bank.score('model.layers.3.attn','new-owner',False)
        assert actual is not None
        # A shorter context and changed owner must not normalize against old pieces.
        expected = oracle()[:, :len(starts)][[1,4]]
        torch.testing.assert_close(actual,expected,rtol=.02,atol=2e-4)
        assert bank.score('model.layers.3.attn','r',False) is None
    bank.disarm()
    graph.replay()
    assert bank.score('model.layers.3.attn', 'r', phase=='prefill') is None


def test_native_candidates_only_commit_the_accepted_mtp_prefix(tmp_path):
    from vllm_kvmem.serving.action_attention import ActionHistory
    bank, plan, execute, oracle, *_ = geometry(tmp_path, 'decode', 5)
    bank.arm({'r':plan}); execute()
    mass = bank.score('model.layers.3.attn', 'r', False)
    assert mass is not None, bank.reasons
    h = ActionHistory(SimpleNamespace(action_scope='non-thinking', action_interval=1, action_decay=.5), [10,11,12,13,14])
    h.begin('r', [14]); h.parser.prepare([20,21,22,23,24])
    h.accept(2, plan['rows'], list(range(len(plan['starts']))), mass.cpu().tolist())
    assert h.samples == 2
    expected = mass[:2].sum(0).cpu().tolist()
    assert [h.mass.get(i,0.) for i in range(len(expected))] == pytest.approx(expected)


def test_ragged_xqa_separates_request_rows_and_contexts(tmp_path):
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from vllm_kvmem.serving.native_attention import _active
    bank, plan, _, _, _, _, _, (k,v), table = geometry(tmp_path, 'decode', 5)
    q = torch.randn(7,8,128,device='cuda',dtype=torch.bfloat16)
    table = torch.cat((table,table.roll(2,1)),0)
    lengths = torch.tensor([259,249],device='cuda',dtype=torch.int32)
    cu = torch.tensor([0,5,7],device='cuda',dtype=torch.int32)
    mask = torch.tensor([[(1<<(r+1))-1,0] for n in (5,2) for r in range(n)],device='cuda',dtype=torch.uint16)
    second_starts = [s for s in plan['starts'] if s < 249]
    second = dict(plan,index=1,begin=5,rows=[0,1],local_begin=247,count=2,starts=second_starts,
                  counts=[b-a for a,b in zip(second_starts,second_starts[1:]+[249])])
    bank.arm({'a':dict(plan,rows=[0,4]),'b':second})
    workspace = torch.zeros(32*1024*1024,device='cuda',dtype=torch.uint8)
    token = _active.set((bank,'model.layers.3.attn'))
    try:
        xqa_batch_decode_with_kv_cache(query=q,kv_cache=(k,v),workspace_buffer=workspace,
            block_tables=table,seq_lens=lengths,max_seq_len=259,bmm1_scale=128**-.5*.7,
            bmm2_scale=.3,out=torch.empty_like(q),kv_layout='NHD',q_len_per_req=5,q_cu_seq_lens=cu,mask=mask)
    finally:
        _active.reset(token)
    for i,(request,begin,n,length,rows) in enumerate((('a',0,5,259,[0,4]),('b',5,2,249,[0,1]))):
        actual = bank.score('model.layers.3.attn',request,False)
        assert actual is not None,bank.reasons
        positions = torch.arange(length,device='cuda')
        keys = k[table[i,positions//32].long(),positions%32].float().repeat_interleave(4,1)
        logits = torch.einsum('rhd,thd->rht',q[begin:begin+n].float(),keys)*128**-.5*.7
        logits.masked_fill_(positions[None,None] > (length-n+torch.arange(n,device='cuda'))[:,None,None],-torch.inf)
        p = logits.softmax(-1).mean(1)
        current = plan if request=='a' else second
        starts,counts = current['starts'],current['counts']
        expected = torch.stack([p[:,s:s+c].sum(-1) for s,c in zip(starts,counts)],-1)[rows]
        torch.testing.assert_close(actual,expected,rtol=.02,atol=2e-4)


def test_native_scores_preserve_query_pressure_mandatory_and_checkpoint_policy(tmp_path):
    from vllm_kvmem.serving.action_attention import ActionHistory
    from vllm_kvmem.core.config import KVMemConfig
    from vllm_kvmem.core.memory import RequestMemory
    from vllm_kvmem.serving.prefill_attention import pressure_seed
    from test_core import append
    bank, plan, execute, oracle, q, _, _, (k,_), table = geometry(tmp_path,'decode',5)
    q.fill_(.125)
    positions = torch.arange(259,device='cuda')
    segments = torch.bucketize(positions,torch.tensor(plan['starts'],device='cuda'),right=True)-1
    keys = torch.zeros_like(k,dtype=torch.bfloat16)
    keys[table[0,positions//32].long(),positions%32] = ((segments.float()-4)*.125).bfloat16()[:,None,None]
    k.copy_(keys)
    bank.arm({'r':plan});execute()
    native = bank.score('model.layers.3.attn','r',False)
    assert native is not None,bank.reasons
    reference = oracle()
    history = [0,160,32,224,96,256,64,192,128,288]

    def consume(values):
        cfg = KVMemConfig(block_tokens=32,select_budget=128,gen_budget=512,repository_blocks=32,
            sink_blocks=1,recent_blocks=0,action_attention=True,action_fraction=.5)
        memory = RequestMemory('policy',cfg)
        spans = append(memory,320)
        options = SimpleNamespace(action_scope='non-thinking',action_interval=1,action_decay=.5)
        h = ActionHistory(options,[10,11,12,13,14])
        authority = SimpleNamespace(memory=memory,history_end=320,action_history=h,
            records={s.handle.slot:SimpleNamespace(history_begin=i*32) for i,s in enumerate(spans)})
        h.begin('r',[14]);h.parser.prepare([20,21,22,23,24])
        h.accept(2,plan['rows'],history,values.cpu().tolist());h.finish(authority)
        snapshot = h.snapshot()
        query_scores = {s.handle.slot:float(10-i) for i,s in enumerate(spans)}
        query = memory.preview_selection(query_scores,(spans[7].handle.slot,))
        seed, source = pressure_seed(authority,memory._store.working_records())
        pressure = memory.preview_selection({},(spans[7].handle.slot,),action=False,pressure_scores=seed)
        assert source == 'output'
        assert {spans[0].handle.slot,spans[7].handle.slot} <= set(query) & set(pressure)
        h.begin('future',[14]);h.parser.prepare([20]);h.accept(1,[0],[0],[[1.]])
        h.finish(authority)
        h.restore(snapshot,authority)
        assert h.snapshot() == snapshot
        assert memory.preview_selection(query_scores,(spans[7].handle.slot,)) == query
        assert memory.close()
        return snapshot,query,pressure

    old,new = consume(reference),consume(native)
    assert old[1:] == new[1:]
    assert new[0][0] == pytest.approx(old[0][0],abs=2e-5)


def test_resolve_keeps_exact_plans_armed_beside_decode_candidates(tmp_path):
    """A mixed batch arms one exact (prefill) plan and one decode candidate."""
    import numpy as np
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from vllm_kvmem.serving.action_attention import ActionRuntime
    from vllm_kvmem.serving.native_attention import _active
    from vllm_kvmem.serving.native_attention_planning import make_decode_frame
    bank, plan, _, _, _, _, _, (k,v), table = geometry(tmp_path, 'decode', 5)
    name = 'model.layers.3.attn'
    q = torch.randn(7,8,128,device='cuda',dtype=torch.bfloat16)
    table = torch.cat((table,table.roll(2,1)),0)
    lengths = torch.tensor([259,249],device='cuda',dtype=torch.int32)
    cu = torch.tensor([0,5,7],device='cuda',dtype=torch.int32)
    mask = torch.tensor([[(1<<(r+1))-1,0] for n in (5,2) for r in range(n)],device='cuda',dtype=torch.uint16)
    second_starts = [s for s in plan['starts'] if s < 249]
    second_counts = [b-a for a,b in zip(second_starts,second_starts[1:]+[249])]
    exact = dict(plan,rows=[0,4],pipeline=True)
    action = ActionRuntime.__new__(ActionRuntime)
    action.runner = SimpleNamespace(_kvmem_window_adapter=None)
    action.native_cache = {'b':dict(generation=0,offset=0,
        **make_decode_frame(second_starts,second_counts,second_starts,247,247,2,bank.context,32))}
    batch = SimpleNamespace(req_ids=['a','b'],query_start_loc_np=[0,5,7],
        num_computed_tokens_np=[254,247],seq_lens_cpu_upper_bound=[259,249])
    candidate = action._decode_candidate(batch,1,'b')
    assert candidate['rows'] == [0,1] and candidate['begin'] == 5
    bank.arm({'a':exact,'b':candidate})
    assert bank.active_rows == 4
    workspace = torch.zeros(32*1024*1024,device='cuda',dtype=torch.uint8)
    token = _active.set((bank,name))
    try:
        xqa_batch_decode_with_kv_cache(query=q,kv_cache=(k,v),workspace_buffer=workspace,
            block_tables=table,seq_lens=lengths,max_seq_len=259,bmm1_scale=128**-.5*.7,
            bmm2_scale=.3,out=torch.empty_like(q),kv_layout='NHD',q_len_per_req=5,q_cu_seq_lens=cu,mask=mask)
    finally:
        _active.reset(token)
    sampled = dict(index=1,begin=5,rows=[1],local_begin=247,count=2,starts=np.array(second_starts),
        counts=np.array(second_counts),history=list(second_starts))
    bank.resolve({'a':exact,'b':sampled})
    assert not bank.request_reasons and bank.reduction_rows == 3
    assert 'a' not in bank.selected_slots and bank.selected_slots['b'] == (2,1)
    assert bank.selected[:3].tolist() == [0,1,3]
    for i,(request,begin,n,length,rows,current) in enumerate((('a',0,5,259,[0,4],plan),
            ('b',5,2,249,[1],dict(starts=second_starts,counts=second_counts)))):
        actual = bank.score(name,request,False)
        assert actual is not None,(bank.reasons,bank.counts)
        positions = torch.arange(length,device='cuda')
        keys = k[table[i,positions//32].long(),positions%32].float().repeat_interleave(4,1)
        logits = torch.einsum('rhd,thd->rht',q[begin:begin+n].float(),keys)*128**-.5*.7
        logits.masked_fill_(positions[None,None] > (length-n+torch.arange(n,device='cuda'))[:,None,None],-torch.inf)
        p = logits.softmax(-1).mean(1)
        expected = torch.stack([p[:,s:s+c].sum(-1) for s,c in zip(current['starts'],current['counts'])],-1)[rows]
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual,expected,rtol=.02,atol=2e-4)
    bank.validate()
    assert bank.validation_failures == 0
    # A different exact plan object was never armed: per-request fallback.
    bank.arm({'a':exact,'b':candidate})
    bank.resolve({'a':dict(exact),'b':sampled})
    assert bank.request_reasons == {'a':'candidate_domain:candidate cached layout missing'}
