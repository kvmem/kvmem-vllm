from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_kvmem.session_control import SessionLedger, SessionNotice


def request(name='a', session='s', tokens=None, salt='tenant'):
    return SimpleNamespace(request_id=name, session_id=session, cache_salt=salt,
        prompt_token_ids=list(range(25)) if tokens is None else tokens,
        num_computed_tokens=16, num_prompt_tokens=25)


def ledger(capacity=2):
    events = []
    result = SessionLedger(capacity, 4,
        lambda *args: events.append(('prepare', args)),
        lambda *args: events.append(('abort', args)),
        lambda *args: events.append(('commit', args)))
    return result, events


def first_turn(control, req=None):
    req = req or request()
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    binding, capture = commands.bindings[0], commands.captures[0]
    control.finished(req)
    notice = SessionNotice('retired', binding.namespace, binding.generation, req.request_id, capture.position, 12)
    control.consume(notice)
    return binding, notice


def close_request(name='close', session='s'):
    req = request(name, session=session, tokens=[777])
    req.num_prompt_tokens = 1
    req.num_computed_tokens = 0
    req.max_tokens = 1
    req.kv_transfer_params = {'kvmem_release_session':True}
    return req


def test_owner_view_keeps_retiring_namespace_until_validated_empty_retirement():
    control, _ = ledger(capacity=1)
    req = request()
    control.lookup(req, 0)
    control.allocated(req, 0)
    binding = control.take_commands().bindings[0]
    assert control.owner_snapshot() == [dict(namespace=['tenant','s'],
        generation=binding.generation,request_id='a',phase='active')]
    control.finished(req)
    assert control.owner_snapshot()[0]['phase'] == 'retiring'
    with pytest.raises(ValueError,match='Stale'):
        control.consume(SessionNotice('retired',binding.namespace,binding.generation+1,'a'))
    assert len(control.owner_snapshot()) == 1
    control.consume(SessionNotice('retired',binding.namespace,binding.generation,'a'))
    assert control.owner_snapshot() == []


def test_explicit_release_fences_old_generation_and_keeps_no_replacement_checkpoint():
    control, events = ledger()
    binding, _ = first_turn(control)
    peer, _ = first_turn(control, request('peer', session='other'))
    control.terminal_capture = True
    req = close_request()
    assert control.lookup(req, 0) is None
    eviction, = control.take_commands().evictions
    assert (eviction.namespace, eviction.generation) == (binding.namespace, binding.generation)
    assert control.lookup(req, 0) is None  # No allocation before worker fence.
    with pytest.raises(ValueError, match='Stale'):
        control.consume(SessionNotice('evicted', binding.namespace, binding.generation+1, ''))
    control.consume(SessionNotice('evicted', binding.namespace, binding.generation, ''))
    assert control.lookup(req, 0) == control.lookup(req, 0) == 0
    assert control.lookup(request('successor'), 0) is None
    control.allocated(req, 0)
    control.capture_before_final_prefill(req, 1)
    control.capture_decode_boundary(req)
    control.finished(req)
    commands = control.take_commands()
    assert not commands.bindings and not commands.captures and not commands.terminals
    assert set(control.entries) == {peer.namespace}
    assert not control.requests and not control.uncached_sessions and not control.uncached_requests
    assert events == []  # No import of closed authority.
    assert control.lookup(request('successor'), 0) == 0


def test_release_waits_for_active_turn_and_can_close_absent_namespace_at_capacity():
    control, _ = ledger(capacity=1)
    req = request(); control.lookup(req, 0); control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands(); binding = commands.bindings[0]
    closing = close_request()
    assert control.lookup(closing, 0) is None and not control.take_commands().evictions
    # An absent namespace needs no checkpoint capacity and cannot evict a peer.
    absent = close_request('absent', session='absent')
    assert control.lookup(absent, 0) == 0
    control.allocated(absent, 0); control.finished(absent)
    assert not control.take_commands().evictions
    assert control.entries[binding.namespace].request_id == req.request_id
    control.finished(req)
    assert control.lookup(closing, 0) is None
    control.consume(SessionNotice('retired', binding.namespace, binding.generation, req.request_id, 16, 12))
    assert control.lookup(closing, 0) is None
    assert len(control.take_commands().evictions) == 1


def test_cancel_release_before_allocation_releases_namespace_serialization():
    control, _ = ledger()
    req = close_request()
    assert control.lookup(req, 0) == 0
    control.finished(req)
    assert not control.entries and not control.uncached_sessions
    assert control.lookup(request('new'), 0) == 0


def test_hit_requires_worker_retirement_then_fresh_exact_prefix_allocation():
    control, events = ledger()
    a, b = request(), request('b')
    assert control.lookup(a, 0) == 0
    control.allocated(a, 0)
    control.capture_before_final_prefill(a, 9)
    commands = control.take_commands()
    assert len(commands.captures) == 1
    assert control.lookup(b, 0) is None
    control.finished(a)
    assert control.lookup(b, 0) is None and control.has_pending_work()
    binding = commands.bindings[0]
    retired = SessionNotice('retired', binding.namespace, binding.generation, 'a', 16, 12)
    control.consume(retired)
    assert not control.has_pending_work()
    assert control.lookup(b, 0) == 16
    assert control.lookup(b, 0) == 16 and len(events) == 1
    control.allocated(b, 16)
    restored = replace(retired, kind='restored', request_id='b')
    control.consume(restored)
    assert events[-1] == ('commit', ('b', 16))
    with pytest.raises(ValueError, match='import completion'):
        control.consume(restored)
    with pytest.raises(ValueError, match='another native'):
        control.consume(retired)
    control.finished(b)
    control.consume(replace(retired, request_id='b', history=0, window=0))
    assert not control.entries and not control.requests


def test_decode_checkpoint_retains_accepted_output_and_failed_replacement_keeps_previous_prefix():
    control, events = ledger()
    req = request()
    control.lookup(req, 0); control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    binding = commands.bindings[0]
    def notice(kind, position, window=0):
        return SessionNotice(kind, binding.namespace, binding.generation, req.request_id,
                             position, window)
    control.consume(notice('captured', 16))
    req.num_computed_tokens = 32
    req.all_token_ids = req.prompt_token_ids + [101, 102, 103, 104, 105, 106, 107, 108]
    control.capture_decode_boundary(req)
    command = control.take_commands().captures[0]
    assert command.position == 32 and command.tokens[-8:] == tuple(range(101, 109))
    control.consume(notice('captured', 32))
    req.num_computed_tokens = 36; req.all_token_ids += [201, 202, 203, 204]
    control.capture_decode_boundary(req)
    control.take_commands()
    control.consume(notice('capture_deferred', 36))
    control.finished(req)
    control.consume(notice('retired', 32, 12))
    continuation = request('new', tokens=req.all_token_ids + [400, 500])
    assert control.lookup(continuation, 0) == 32
    assert events[-1] == ('prepare', ('new', 32, 12))


def test_terminal_checkpoint_uses_full_accepted_identity_at_a_partial_block():
    control, events = ledger()
    control.terminal_capture = True
    req = request(); control.lookup(req, 0); control.allocated(req, 0)
    binding = control.take_commands().bindings[0]
    req.all_token_ids = req.prompt_token_ids + [700, 701, 702, 703, 704]
    req.num_computed_tokens = 29; req.num_in_flight_tokens = 0
    req.num_tokens = 30; req.num_output_tokens = 5
    req.status = SimpleNamespace(name='FINISHED_LENGTH_CAPPED')
    req.is_finished = lambda: True
    control.finished(req)
    terminal = control.take_commands().terminals[0]
    assert terminal.position == 29 and terminal.tokens[-5:] == tuple(range(700, 705))
    control.consume(SessionNotice('captured', binding.namespace, binding.generation, 'a', 29))
    control.consume(SessionNotice('retired', binding.namespace, binding.generation, 'a', 29, 13))
    assert control.lookup(request('new', tokens=req.all_token_ids + [999]), 0) == 29
    assert events[-1] == ('prepare', ('new', 29, 13))


@pytest.mark.parametrize('p_saved,m_saved', [(True,True),(True,False),(False,True),(False,False)])
def test_one_token_finish_keeps_pending_p_and_m_acknowledgments_separate(p_saved,m_saved):
    control, _ = ledger()
    control.terminal_capture = True
    req = request()
    control.lookup(req,0); control.allocated(req,0)
    control.capture_before_final_prefill(req,9)
    binding = control.take_commands().bindings[0]
    req.all_token_ids = req.prompt_token_ids+[700]
    req.num_computed_tokens = 25; req.num_in_flight_tokens = 0
    req.num_tokens = 26; req.num_output_tokens = 1
    req.status = SimpleNamespace(name='FINISHED_LENGTH_CAPPED')
    req.is_finished = lambda: True
    control.finished(req)  # P acknowledgment arrives after this transition.
    assert control.take_commands().terminals[0].position == 25
    for position,saved in [(16,p_saved),(25,m_saved)]:
        notice = SessionNotice('captured' if saved else 'capture_deferred',
            binding.namespace,binding.generation,'a',position)
        control.consume(notice)
        with pytest.raises(ValueError,match='acknowledgment'):
            control.consume(notice)
    position = 25 if m_saved else 16 if p_saved else 0
    control.consume(SessionNotice('retired',binding.namespace,binding.generation,
                                  'a',position,12 if position else 0))
    if position:
        assert control.entries[binding.namespace].history == position
    else:
        assert not control.entries


def test_prefix_miss_and_lru_eviction_hold_capacity_until_generation_ack():
    control, _ = ledger(capacity=1)
    binding, retired = first_turn(control)
    changed = list(range(25)); changed[3] = 999
    req = request('b', tokens=changed)
    assert control.lookup(req, 0) is None
    assert control.lookup(req, 0) is None
    commands = control.take_commands()
    assert len(commands.evictions) == 1 and control.has_pending_work()
    with pytest.raises(ValueError, match='Stale'):
        control.consume(SessionNotice('evicted', binding.namespace, binding.generation + 1, ''))
    control.consume(SessionNotice('evicted', binding.namespace, binding.generation, ''))
    assert control.lookup(req, 0) == 0
    assert control.entries[binding.namespace].generation > binding.generation
    with pytest.raises(ValueError, match='Stale'):
        control.consume(retired)
    # Cancelling a waiting cold request creates no phantom worker ownership.
    control.finished(req)
    assert not control.entries and not control.has_pending_work()
    first_turn(control)
    assert control.lookup(request('other', session='new'), 0) is None
    assert len(control.take_commands().evictions) == 1


def planned_idle_session():
    control, events = ledger()
    binding, _ = first_turn(control)
    control.gen_budget, control.select_budget = 32, 64
    entry = control.entries[binding.namespace]
    entry.query = (8, 15)
    entry.mandatory = ((0, 4, 'root'), (8, 15, 'query'))
    return control, events, binding, entry


@pytest.mark.parametrize('explicit_pins', [False, True])
def test_short_prefix_replacement_does_not_inherit_old_query_pins(explicit_pins):
    from vllm_kvmem.request_plan import request_plan
    control, _, binding, _ = planned_idle_session()
    req = request('reset', tokens=[999])
    req.num_prompt_tokens = 1
    fields = {'kvmem_reselect':'off'}
    if explicit_pins:
        fields['kvmem_mandatory_spans'] = []
    req.kv_transfer_params = {'kvmem_request_plan':fields}
    assert control.lookup(req, 0) is None
    control.consume(SessionNotice('evicted', binding.namespace, binding.generation, ''))
    assert control.lookup(req, 0) == 0
    plan = request_plan(req, 4, 32)
    assert plan.query is None and not plan.mandatory


def test_exact_continuation_keeps_old_query_and_mandatory_pins():
    from vllm_kvmem.request_plan import request_plan
    control, events, _, entry = planned_idle_session()
    req = request('next', tokens=list(range(25))+[999])
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.kv_transfer_params = {'kvmem_request_plan':{'kvmem_query_span':[8, 15]}}
    assert control.lookup(req, 0) == 16
    plan = request_plan(req, 4, 32)
    assert plan.query is None
    assert plan.mandatory == ((0, 4, 'root'), (8, 15, 'query'))
    assert entry.query == (8, 15)
    assert events[-1] == ('prepare', ('next', 16, 12))


@pytest.mark.parametrize('cancel_before_allocation', [False, True])
@pytest.mark.parametrize('max_output,expected_window', [(16,63),(32,55)])
def test_larger_output_after_partial_terminal_compacts_during_import(
        cancel_before_allocation,max_output,expected_window):
    from vllm_kvmem.prompt_checkpoints import token_bytes
    control, events, binding, entry = planned_idle_session()
    entry.history = entry.window = 95
    entry.tokens = token_bytes(list(range(96)), 95)
    req = request('larger-output', tokens=list(range(96)))
    req.num_prompt_tokens = 96
    req.max_tokens = max_output
    req.kv_transfer_params = {'kvmem_request_plan': {'kvmem_reselect': 'off'}}
    assert control.lookup(req, 0) == 95
    assert entry.pressure_restore and entry.restore_window == expected_window
    assert entry.history == entry.window == 95  # Durable M remains unchanged.
    assert events[-1] == ('prepare', ('larger-output', 95, expected_window))
    assert expected_window + 1 + req.max_tokens <= 96 - 4
    from vllm_kvmem.restore_admission import RestoreAdmission
    gate = RestoreAdmission()
    assert gate.lookup(req, entry)
    assert gate.permits[req.request_id].window == expected_window
    assert gate.lookup(req, entry)  # Repeated prefix lookup preserves the permit.
    if cancel_before_allocation:
        gate.rollback(req)
        control.finished(req)
        assert entry.phase == 'idle' and entry.history == entry.window == 95
        assert not entry.pressure_restore and entry.restore_window == 0
    else:
        gate.before_allocate(req, 95)
        gate.allocated(req)
        control.allocated(req, 95)
        new_binding = control.take_commands().bindings[-1]
        assert new_binding.pressure_restore and new_binding.window == expected_window
        gate.dispatch(SimpleNamespace(request_id=req.request_id, command_id=-1,
            generation=new_binding.generation, history_end=95, window_tokens=expected_window, resident=False))
        control.consume(SessionNotice('restored', binding.namespace, binding.generation,
            req.request_id, 95, expected_window))
        gate.complete(req.request_id, -1)
        assert events[-1] == ('commit', ('larger-output', 95))


@pytest.mark.parametrize('rolling,history,expected_window,pressure', [
    (False,79,55,True), (True,79,79,False), (True,95,63,True)])
def test_rolling_partial_m_preserves_selected_pages_when_only_output_reserve_overflows(
        monkeypatch,rolling,history,expected_window,pressure):
    import json
    from vllm_kvmem.prompt_checkpoints import token_bytes
    from vllm_kvmem.request_plan import request_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'rolling_active_window':rolling}))
    control, events, _, entry = planned_idle_session()
    entry.history = entry.window = history
    entry.tokens = token_bytes(list(range(history+1)),history)
    req = request('next',tokens=list(range(history+1)))
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.max_tokens = 32
    req.kv_transfer_params = {'kvmem_request_plan':{'kvmem_reselect':'off'}}
    assert control.lookup(req,0) == history
    assert entry.pressure_restore == pressure
    assert events[-1] == ('prepare',('next',history,expected_window))
    plan = request_plan(req,4,32)
    assert plan.rolling_active_window == rolling
    assert plan.prefill_window == ('pressure' if pressure else 'keep_selected')
    assert plan.mandatory == ((0,4,'root'),(8,15,'query'))
    assert plan.output_tokens == req.max_tokens == 32
    assert entry.history == entry.window == history
    control.finished(req)  # Cancellation before allocation retains the original M.
    assert entry.phase == 'idle' and entry.history == entry.window == history
    assert entry.restore_window == 0 and not entry.pressure_restore


@pytest.mark.parametrize('window,suffix,pressure', [
    (87,12,False), (88,12,False), (89,12,True), (90,12,True),
    (91,12,True), (92,12,True), (90,160,True), (90,1,False), (90,2,False)])
def test_near_full_terminal_append_reserves_prefill_probe_before_import(
        monkeypatch,window,suffix,pressure):
    import json
    from vllm_kvmem.prompt_checkpoints import token_bytes
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps({'rolling_active_window':True}))
    control, events, _, entry = planned_idle_session()
    history = window + 128
    entry.history, entry.window = history, window
    entry.tokens = token_bytes(list(range(history+1)), history)
    req = request('near-full', tokens=list(range(history+suffix)))
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.max_tokens = 32
    req.kv_transfer_params = {'kvmem_request_plan':{'kvmem_reselect':'off'}}
    assert control.lookup(req,0) == history
    expected = 64 - (-history % 4) if pressure else window
    assert entry.pressure_restore == pressure
    assert (entry.restore_window or entry.window) == expected
    assert events[-1] == ('prepare', ('near-full',history,expected))
    assert entry.history == history and entry.window == window
    control.finished(req)
    assert entry.phase == 'idle' and entry.restore_window == 0
    assert entry.history == history and entry.window == window


def test_dsh_terminal_capacity_failure_keeps_native_lookahead_inside_allocation(monkeypatch):
    """Real DSH step132: partial M at local53212, then628 new prompt tokens."""
    import json
    from vllm_kvmem.prompt_checkpoints import token_bytes
    from vllm_kvmem.request_plan import request_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'rolling_active_window':True}))
    control, _, _, entry = planned_idle_session()
    control.block_tokens, control.select_budget, control.gen_budget = 32,36864,16384
    entry.history, entry.window = 102204,53212
    entry.tokens = token_bytes(list(range(entry.history+1)),entry.history)
    req = request('dsh-failure',tokens=list(range(102832)))
    req.num_prompt_tokens, req.max_tokens = 102832,16384
    req.kv_transfer_params = {'kvmem_request_plan':{'kvmem_reselect':'off'}}
    assert control.lookup(req,0) == 102204
    assert entry.pressure_restore and entry.restore_window == 36860
    plan = request_plan(req,32,16384)
    history, offset = entry.history, entry.history-entry.restore_window
    while history < req.num_prompt_tokens:
        count = plan.admit(history,1024,req.num_prompt_tokens,offset,53248)
        assert count > 0
        # Native allocate_slots includes seven draft lookahead rows BEFORE
        # connector metadata can execute an ordinary synchronous pressure.
        assert history-offset+count+7 <= 53248
        history += count


@pytest.mark.parametrize('history,window,pressure', [
    (161467,111419,True),  # Actual r8 import: five tokens before a 4492-token Query.
    (161472,111424,True),  # Already at the atomic replay boundary.
    (161467,110139,False), # Exactly at the safe pre-Query pressure boundary.
    (161467,110171,True),  # One page beyond that boundary.
    (161472,110144,False), # Query and native lookahead fit without import selection.
])
@pytest.mark.parametrize('cancel', [False,True])
def test_warm_new_query_reserves_atomic_replay_before_import(monkeypatch,history,window,pressure,cancel):
    import json
    from vllm_kvmem.prompt_checkpoints import token_bytes
    from vllm_kvmem.request_plan import request_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'rolling_active_window':True}))
    control, events, binding, entry = planned_idle_session()
    control.block_tokens, control.select_budget, control.gen_budget = 32,98304,16384
    entry.history, entry.window = history,window
    entry.tokens = token_bytes(list(range(165971)),history)
    entry.mandatory = ()
    req = request('tb-query',tokens=list(range(165971)))
    req.num_prompt_tokens, req.max_tokens = 165971,131072
    req.kv_transfer_params = {'kvmem_request_plan':{'kvmem_query_span':[161472,165964],
        'kvmem_mandatory_spans':[]}}
    assert control.lookup(req,0) == history
    expected = 98304 - (-history % 32) if pressure else window
    assert entry.pressure_restore == pressure
    assert events[-1] == ('prepare',('tb-query',history,expected))
    assert (entry.history,entry.window) == (history,window)
    plan = request_plan(req,32,16384)
    assert plan.query == (161472,165964) and req.max_tokens == 131072
    if cancel:
        control.finished(req)
        assert entry.phase == 'idle' and not entry.pressure_restore
        assert (entry.history,entry.window) == (history,window)
        return
    offset = history - expected
    if history < plan.replay_begin:
        count = plan.admit(history,2048,req.num_prompt_tokens,offset,114688)
        assert history+count == plan.replay_begin
        assert history-offset+count+7 <= 114688
        history += count
    count = plan.admit(history,16424,req.num_prompt_tokens,offset,114688)
    assert count == 4492 and plan.action(history,req.num_prompt_tokens,offset,114688) == 'query'
    assert history-offset+count+7 <= 114688
    control.allocated(req,entry.history)
    new_binding = control.take_commands().bindings[-1]
    assert new_binding.pressure_restore == pressure and new_binding.window == expected


@pytest.mark.parametrize('session', ['s',None])
@pytest.mark.parametrize('rolling,prompt,expected', [
    (False,80,'pressure'), (True,80,'keep_selected'), (True,93,'pressure')])
def test_rolling_cold_plan_reserves_input_and_native_headroom(
        monkeypatch,session,rolling,prompt,expected):
    import json
    from vllm_kvmem.request_plan import request_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'rolling_active_window':rolling}))
    control, _ = ledger()
    control.gen_budget, control.select_budget = 32,64
    req = request(session=session,tokens=list(range(prompt)))
    req.num_prompt_tokens = prompt
    req.max_tokens = 32
    req.kv_transfer_params = {'kvmem_request_plan':{
        'kvmem_reselect':'off','kvmem_prefill_window':'keep_selected'}}
    assert control.lookup(req,0) == 0
    plan = request_plan(req,4,32)
    assert plan.prefill_window == expected
    assert plan.output_tokens == req.max_tokens == 32


def test_changed_branch_inherits_only_pins_in_verified_prompt_checkpoint():
    from vllm_kvmem.prompt_checkpoints import token_bytes
    from vllm_kvmem.request_plan import request_plan
    control, events, _, entry = planned_idle_session()
    entry.prompt_history = entry.prompt_window = 8
    entry.prompt_tokens = token_bytes(list(range(25)), 8)
    req = request('branch', tokens=list(range(8))+[999, 1000])
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.kv_transfer_params = {'kvmem_request_plan':{'kvmem_reselect':'off'}}
    assert control.lookup(req, 0) == 8
    plan = request_plan(req, 4, 32)
    assert plan.mandatory == ((0, 4, 'root'),)
    assert plan.query is None and entry.query is None
    assert events[-1] == ('prepare', ('branch', 8, 8))


def test_cancel_before_allocation_restores_idle_lease_and_aborts_only_preparation():
    control, events = ledger()
    binding, _ = first_turn(control)
    req = request('b')
    assert control.lookup(req, 0) == 16
    control.finished(req)
    assert events[-1] == ('abort', ('b',))
    assert control.entries[binding.namespace].phase == 'idle'
    assert control.lookup(request('c'), 0) == 16


def test_full_busy_cache_allows_uncached_session_but_serializes_its_turns():
    control, _ = ledger(capacity=1)
    a = request()
    assert control.lookup(a, 0) == 0
    control.allocated(a, 0)
    assert control.lookup(request('b'), 0) is None
    unrelated = request('c', salt='other-tenant')
    assert control.lookup(unrelated, 0) == 0
    control.allocated(unrelated, 0)
    assert len(control.entries) == 1 and 'c' not in control.requests
    assert control.lookup(unrelated, 0) == 0
    continuation = request('d', salt='other-tenant')
    assert control.lookup(continuation, 0) is None
    assert [binding.request_id for binding in control.take_commands().bindings] == ['a']
    control.finished(unrelated)
    assert not control.uncached_sessions and not control.uncached_requests
    assert control.lookup(continuation, 0) == 0
    control.finished(continuation)


def test_uncached_session_lease_survives_a_checkpoint_slot_becoming_available():
    control, _ = ledger(capacity=1)
    cached = request()
    assert control.lookup(cached, 0) == 0
    control.allocated(cached, 0)
    binding = control.take_commands().bindings[0]
    uncached = request('b', session='overflow')
    assert control.lookup(uncached, 0) == 0
    control.allocated(uncached, 0)
    control.finished(cached)
    control.consume(SessionNotice('retired', binding.namespace, binding.generation, 'a'))
    assert not control.entries
    assert control.lookup(request('c', session='overflow'), 0) is None
    # A retry after admission pressure stays uncached for this whole request.
    assert control.lookup(uncached, 0) == 0 and not control.entries
    control.finished(uncached)
    continuation = request('c', session='overflow')
    assert control.lookup(continuation, 0) == 0
    assert ('tenant', 'overflow') in control.entries
    control.finished(continuation)
    assert not control.entries and not control.uncached_sessions and not control.uncached_requests


def test_cancel_uncached_request_before_allocation_releases_only_its_lease():
    control, events = ledger(capacity=1)
    cached = request()
    assert control.lookup(cached, 0) == 0
    uncached = request('b', session='overflow')
    assert control.lookup(uncached, 0) == 0
    # Waiting requests that never acquired ownership cannot release the owner.
    waiting = request('c', session='overflow')
    assert control.lookup(waiting, 0) is None
    control.finished(waiting)
    assert control.lookup(waiting, 0) is None
    control.finished(uncached)
    assert control.lookup(waiting, 0) == 0
    control.finished(waiting)
    assert list(control.requests) == ['a'] and not events
    assert not control.uncached_sessions and not control.uncached_requests


def test_checkpoint_candidate_requires_sealed_final_prefill_boundary():
    control, _ = ledger()
    req = request()
    control.lookup(req, 0); control.allocated(req, 0)
    control.capture_before_final_prefill(req, 4)
    req.num_computed_tokens = 17
    control.capture_before_final_prefill(req, 8)
    req.num_computed_tokens = 16
    control.capture_before_final_prefill(req, 9)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    assert [c.position for c in commands.captures] == [16]
    control.finished(req)
    b = commands.bindings[0]
    with pytest.raises(ValueError, match='different prompt'):
        control.consume(SessionNotice('retired', b.namespace, b.generation, req.request_id, 12, 12))
    control.consume(SessionNotice('retired', b.namespace, b.generation, req.request_id))
    assert not control.entries  # Worker may decline a checkpoint on byte pressure.


def test_native_connector_metadata_roundtrip_for_exact_checkpoint_ack(monkeypatch):
    from test_connector import pair
    from vllm.v1.outputs import KVConnectorOutput
    from vllm_kvmem.connector import ControlCompletions
    scheduler, _ = pair()
    control, events = ledger()
    scheduler.sessions = control
    req = request()
    assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
    scheduler.update_state_after_alloc(req, None, 0)
    control.capture_before_final_prefill(req, 9)
    metadata = scheduler.build_connector_meta(None)
    binding = metadata.sessions.bindings[0]
    scheduler.request_finished_all_groups(req, ())
    assert scheduler.has_pending_push_work()
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        sessions=[SessionNotice('retired', binding.namespace, binding.generation, 'a', 16, 12)])))
    assert scheduler.get_num_new_matched_tokens(request('b'), 0) == (16, False)
    assert events == [('prepare', ('b', 16, 12))]


@pytest.mark.parametrize('cancel_before_restore', [False, True])
def test_retained_import_survives_cancellation_without_another_capture(cancel_before_restore):
    control, events = ledger()
    binding, original = first_turn(control)
    req = request('b')
    assert control.lookup(req, 0) == original.history
    control.allocated(req, original.history)
    control.take_commands()
    if cancel_before_restore:
        control.finished(req)
    restored = replace(original, kind='restored', request_id='b', retained=True)
    control.consume(restored)
    assert events[-1] == ('commit', ('b', original.history))
    if not cancel_before_restore:
        control.finished(req)
    control.consume(replace(original, request_id='b'))
    entry = control.entries[binding.namespace]
    assert entry.phase == 'idle' and entry.history == original.history
    assert entry.tokens and not control.requests
    assert control.lookup(request('c'), 0) == original.history
    assert events[-1] == ('prepare', ('c', original.history, original.window))


def test_graph_only_metadata_survives_scheduler_session_binding():
    control, _ = ledger()
    control.gen_budget = 64
    control.select_budget = 64
    req = request(tokens=list(range(24)))
    req.num_prompt_tokens = 24
    req.kv_transfer_params = dict(kvmem_request_plan=dict(
        kvmem_locality_spans=[dict(begin=0,end=12,role='user'),dict(begin=12,end=24,role='tool')],
        kvmem_locality_provenance=dict(schema_version=1,source='renderer')))
    control.lookup(req,0)
    control.allocated(req,0)
    binding, = control.take_commands().bindings
    assert binding.locality_spans == ((0,12,'user'),(12,24,'tool'))
    assert dict(binding.locality_provenance)['source'] == 'renderer'
    assert binding.message_spans == binding.mandatory_spans == ()
    assert binding.tokens == tuple(req.prompt_token_ids)
