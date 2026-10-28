import pytest
from vllm_kvmem.request_plan import RequestPlan


def test_query_replay_limit_is_independent_of_full_generation_budget(monkeypatch):
    import json
    from vllm_kvmem.http_sessions import normalize_query_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(
        block_tokens=32, gen_budget=131072, select_budget=131072,
        query_replay_budget=16384, max_model_len=1048576)))
    payload = dict(prompt=[1]*20000, kvmem_query_span=[19000,19999], max_tokens=131072)
    assert normalize_query_plan(payload) and payload['max_tokens'] == 131072
    with pytest.raises(ValueError, match='Query replay budget'):
        normalize_query_plan(dict(prompt=[1]*20000, kvmem_query_span=[0,17000], max_tokens=1))
    with pytest.raises(ValueError, match='within G'):
        normalize_query_plan(dict(prompt=[1], max_tokens=131073))


def test_full_query_and_partial_tail_are_one_atomic_replay():
    plan=RequestPlan.parse({'kvmem_query_span':[9001,11003], 'kvmem_reselect':'force'},12000,128,4096)
    assert plan.replay_begin==8960
    assert plan.admit(8832,4096,12000,0,12288)==128
    assert plan.admit(8960,1024,12000,0,12288)==0
    assert plan.admit(8960,4096,12000,0,12288)==2043
    assert plan.action(8960,12000,0,12288)=='query'
    with pytest.raises(RuntimeError,match='split'):
        plan.admit(9984,4096,12000,0,12288)


def test_pressure_keeps_complete_query_headroom_without_shortening_history():
    p=RequestPlan.parse({'kvmem_query_span':[25003,27103]},28001,128,4096)
    h=offset=0;events=[]
    while h<28001:
        n=p.admit(h,4096,28001,offset,12288)
        assert n>0 and h-offset+n<=12288
        action=p.action(h,28001,offset,12288)
        if h>=8192 and action:
            events.append((action,h,n));offset=h-8192
        h+=n
    assert h==28001
    queries=[e for e in events if e[0]=='query']
    assert queries==[('query',24960,2143)]
    assert any(e[0]=='pressure' for e in events)


def test_message_roles_and_complete_range_preserve_explicit_role_envelope():
    p=RequestPlan.parse({'kvmem_message_spans':[
        {'role':'tool','begin':0,'end':4000}, {'role':'user','begin':4001,'end':6003}],
        'kvmem_query_message_range':[1,2]},6100,128,4096)
    assert p.query==(4001,6003)
    tool = RequestPlan.parse({'kvmem_message_spans':[{'role':'tool','begin':0,'end':40}],
        'kvmem_query_message_range':[0,1]},40,128,4096)
    assert tool.query == (0,40)


@pytest.mark.parametrize('payload',[{'kvmem_reselect':'force'},
    {'kvmem_reselect':'off','kvmem_query_span':[1,3]},
    {'kvmem_prefill_window':'semantic_chunk','kvmem_query_span':[1,3]},
    {'kvmem_query_span':[True,3]}, {'kvmem_query_span':[1,9000]},
    {'kvmem_reselect':'truncate'}, {'unsupported':True}])
def test_invalid_plans_fail_explicitly(payload):
    with pytest.raises(ValueError): RequestPlan.parse(payload,9000,128,4096)


def test_keep_selected_exhaustion_and_off_do_not_invent_query():
    p=RequestPlan.parse({'kvmem_reselect':'off','kvmem_prefill_window':'keep_selected'},12000,128,4096)
    assert p.query is None
    assert p.action(12160,12000,0,12288)=='exhausted'
    p=RequestPlan.parse({'kvmem_reselect':'off'},25000,128,4096)
    assert p.action(12160,25000,0,12288)=='pressure'


def test_native_http_moves_plan_to_the_supported_metadata_channel(monkeypatch):
    import json
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'block_tokens':128,'gen_budget':4096,'select_budget':8192}))
    from vllm_kvmem.http_sessions import normalize_query_plan
    payload={'prompt':list(range(12000)), 'max_tokens':128, 'kvmem_query_span':[9001,11003],
        'kvmem_reselect':'force'}
    assert normalize_query_plan(payload)
    assert payload['kv_transfer_params']['kvmem_request_plan']['kvmem_query_span']==[9001,11003]
    assert 'kvmem_query_span' not in payload


def test_index_finalization_matches_original_serial_content_sum():
    import torch
    from vllm_kvmem import _native
    raw=torch.randn(3,128,2,32,generator=torch.Generator().manual_seed(9)).bfloat16()
    raw[:,0]=2**24;raw[:,1]=1;raw[:,2]=-2**24
    sums=torch.full((2,2,32),float('nan'))
    _native.finalize_mean_bf16(raw.view(torch.uint8).numpy(),sums.numpy(),128)
    expected=torch.zeros_like(sums)
    for t in range(128): expected.add_(raw[:2,t].float())
    assert torch.equal(sums,expected)
    assert torch.equal((sums/128).half(),(expected/128).half())


def test_native_http_unbounded_null_output_is_normalized_within_g(monkeypatch):
    import json
    from vllm_kvmem.http_sessions import normalize_query_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'block_tokens':128,'gen_budget':4096,'select_budget':8192}))
    payload={'prompt':[1,2,3], 'max_tokens':None}
    assert normalize_query_plan(payload) and payload['max_tokens']==16


def test_off_pressure_visits_boundary_before_native_mtp_page_allocation():
    p=RequestPlan.parse({'kvmem_reselect':'off'},65001,128,4096)
    h=offset=0;pressures=0
    while h<65001:
        n=p.admit(h,4128,65001,offset,12288)
        assert n>0 and h-offset+n+7<=12288
        if p.action(h,65001,offset,12288)=='pressure':
            pressures+=1;offset=h-8192
        h+=n
    assert h==65001 and pressures>0


def test_planning_rejects_g_without_pressure_replay_and_mtp_headroom():
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.options import RuntimeOptions
    with pytest.raises(ValueError,match='three blocks'):
        RuntimeOptions(memory=KVMemConfig(select_budget=8192,gen_budget=256),request_planning=True)


@pytest.mark.parametrize('prompt,query',[(51634,None),(65001,None),(36865,None),
    (40960,(36864,40960)),(41011,(36864,40960)),(41011,(36001,41011))])
def test_prefill_reserves_entire_generation_before_decode(prompt,query):
    from dataclasses import replace
    k,g,block=36864,16384,32
    p=replace(RequestPlan.parse({'kvmem_query_span':query} if query else
        {'kvmem_reselect':'off'},prompt,block,g),output_tokens=g)
    h=offset=0;events=[]
    while h<prompt:
        # Atomic Query credit may exceed an ordinary chunk.
        desired=8192 if h==p.replay_begin else 1024
        n=p.admit(h,desired,prompt,offset,k+g)
        assert n>0 and h-offset+n+7<=k+g
        action=p.action(h,prompt,offset,k+g)
        budget=p.selection_budget(h,prompt,k+g)
        if action and h>=budget:
            offset=h-budget;events.append((action,h,budget))
        h+=n
    assert events
    assert prompt-offset+g<=k+g-block
    for h in range(prompt,prompt+g):
        assert p.action(h,prompt,offset,k+g) is None


def test_warm_tool_preserves_pages_if_actual_output_fits():
    from dataclasses import replace
    p=replace(RequestPlan(None,block_tokens=32,gen_budget=16384,
        mandatory=((0,128,'query'),)),output_tokens=512)
    assert p.action(51616,51634,0,53248) is None
    assert p.admit(51200,1024,51634,0,53248)==434
    # A full G output needs a final, smaller selection instead.
    p=replace(p,output_tokens=16384)
    assert p.admit(51200,1024,51634,0,53248)==416
    assert p.action(51616,51634,0,53248)=='pressure'
    assert p.selection_budget(51616,51634,53248)==36800


def test_rolling_window_defers_output_reserve_until_actual_decode_pressure():
    from dataclasses import replace
    p = RequestPlan(None, block_tokens=32, gen_budget=16384,
        mandatory=((0,128,'query'),), output_tokens=16384)
    rolling = replace(p, rolling_active_window=True)
    assert p.action(51616,51634,0,53248) == 'pressure'
    assert rolling.action(51616,51634,0,53248) is None
    assert p.admit(51200,1024,51634,0,53248) == 416
    assert rolling.admit(51200,1024,51634,0,53248) == 434
    assert rolling.selection_budget(51616,51634,53248) == 36864
    assert rolling.output_tokens == p.output_tokens == 16384


def test_rolling_continuation_preserves_stable_query_and_pages_until_they_fill():
    from dataclasses import replace
    p = RequestPlan((64,96), block_tokens=32, gen_budget=16384,
        mandatory=((64,96,'query'),), output_tokens=16384)
    previous = ((64,96,'query'),)
    old, changed = p.continuation(51634,16384,51600,50576,53248,(64,96),previous)
    rolling, rolling_changed = replace(p, rolling_active_window=True).continuation(
        51634,16384,51600,50576,53248,(64,96),previous)
    assert not changed and not rolling_changed
    assert old.query is rolling.query is None
    assert old.prefill_window == 'pressure'
    assert rolling.prefill_window == 'keep_selected'
    assert rolling.mandatory == previous
    assert rolling.output_tokens == 16384
    # Native MTP clips every possible accepted suffix at the same sealed point.
    from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy
    policy = SpeculativeQueryPolicy(block_tokens=32,select_budget=36864,
        gen_budget=16384,max_query_tokens=5)
    offset, boundary = 1024, 53248 - 32
    for local in range(boundary - 5, boundary):
        admitted = policy.admit(offset+local,5,51634,offset)
        assert 1 <= admitted <= 5 and local + admitted <= boundary
        for accepted in range(1,admitted+1):
            next_history = offset + local + accepted
            action = rolling.action(next_history,51634,offset,53248)
            assert action == ('pressure' if local+accepted == boundary else None)
    assert replace(rolling,rolling_active_window=False).action(
        offset+boundary,51634,offset,53248) == 'exhausted'
    assert rolling.selection_budget(offset+boundary,51634,53248) == 36864


def test_dflash8_tool_rounds_share_window_and_clip_every_acceptance_at_capacity():
    from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy
    p=RequestPlan((64,96),block_tokens=32,gen_budget=16384,
        output_tokens=16384,rolling_active_window=True)
    # A 129-token tool call and its 920-token tool result do not reserve a fresh G.
    resumed,changed=p.continuation(151841,16384,150921,36937,53248,(64,96))
    assert not changed and resumed.query is None and resumed.prefill_window=='keep_selected'
    assert resumed.action(151840,151841,150921-36937,53248) is None
    policy=SpeculativeQueryPolicy(32,36864,16384,8)
    boundary=53248-32
    for left in range(1,9):
        h=boundary-left
        admitted=policy.admit(h,8,1,0)
        assert admitted==left
        for accepted in range(1,admitted+1):
            assert resumed.action(h+accepted,1,0,53248)==('pressure' if accepted==left else None)


def test_rolling_plan_is_server_only_and_survives_public_payload_roundtrip(monkeypatch):
    import json
    from types import SimpleNamespace
    from vllm_kvmem.request_plan import request_plan
    from vllm_kvmem.session_control import SessionLedger
    request = SimpleNamespace(num_prompt_tokens=80, max_tokens=32,
        sampling_params=SimpleNamespace(max_tokens=32),
        kv_transfer_params={'kvmem_request_plan':{'kvmem_reselect':'off'}})
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'rolling_active_window':True}))
    plan = request_plan(request,4,32)
    assert plan.rolling_active_window and plan.output_reserve == 0
    assert plan.output_tokens == request.max_tokens == request.sampling_params.max_tokens == 32
    assert 'rolling_active_window' not in plan.payload()
    SessionLedger.set_plan(request,plan)
    restored = request_plan(request,4,32)
    assert restored.payload() == plan.payload()
    assert restored.rolling_active_window and restored.output_tokens == 32
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG','{}')
    assert not request_plan(request,4,32).rolling_active_window
    with pytest.raises(ValueError,match='Unknown'):
        RequestPlan.parse({'rolling_active_window':True},80,4,32)
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'rolling_active_window':'false'}))
    with pytest.raises(ValueError,match='server boolean'):
        request_plan(request,4,32)


@pytest.mark.parametrize('rolling',[False,True])
def test_only_rolling_window_can_generate_past_g_in_one_request(monkeypatch,rolling):
    import json
    from vllm_kvmem.http_sessions import normalize_query_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps(dict(
        block_tokens=32,gen_budget=32768,select_budget=65536,rolling_active_window=rolling)))
    payload = {'prompt':[1,2,3], 'max_tokens':32768}
    assert normalize_query_plan(payload) and payload['max_tokens'] == 32768
    longer = {'prompt':[1,2,3], 'max_tokens':131072}
    if rolling:
        assert normalize_query_plan(longer) and longer['max_tokens']==131072
    else:
        with pytest.raises(ValueError,match='within G'):
            normalize_query_plan(longer)
