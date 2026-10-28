from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from vllm.v1.outputs import KVConnectorOutput
from vllm_kvmem.connector import ControlCompletions
from vllm_kvmem.scheduler import KVMemScheduler
from test_connector import pair
from test_window_manager import make_manager


def setup_scheduler():
    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler.session_capture_decode = False
    scheduler.connector, _ = pair()
    scheduler.connector.scheduler_completion = scheduler._complete_refresh
    manager, pool = make_manager()
    second, second_pool = make_manager()
    scheduler.window_managers = {0: manager, 1: second}
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(
        single_type_managers=(manager, second)))
    request = SimpleNamespace(request_id="a", num_in_flight_tokens=0, num_prompt_tokens=64,
                              num_computed_tokens=64, next_decode_eligible_step=7,
                              is_finished=lambda: False)
    scheduler.requests = {"a": request}
    scheduler.transactions = {}
    scheduler.table_updates = {}
    for m in (manager, second):
        m.allocate_new_blocks("a", 64, 64)
        m.allocate_new_blocks("b", 32, 32)
    return scheduler, request, (pool, second_pool)


def test_native_prefill_cadence_releases_regularly_and_never_waits_on_unavailable_decode():
    scheduler, request, _ = setup_scheduler()
    scheduler._bounded_target = True
    scheduler.prefill_cadence = 4
    scheduler._mixed_steps = scheduler._cadence_requested = 0
    scheduler.current_step = 7
    scheduler.prefill_capacity_bound = False
    request.is_prefill_chunk = False
    prefill = SimpleNamespace(is_prefill_chunk=True)
    scheduler.running = [request, prefill]
    scheduler.waiting = []
    assert [scheduler._throttle_mixed_prefill() for _ in range(8)] == [True, True, True, False] * 2
    assert scheduler._cadence_requested == 6
    request.next_decode_eligible_step = 8
    assert not scheduler._throttle_mixed_prefill()
    request.next_decode_eligible_step = 7
    request.num_in_flight_tokens = 1
    assert not scheduler._throttle_mixed_prefill()
    request.num_in_flight_tokens = 0
    scheduler.transactions['a'] = object()
    assert not scheduler._throttle_mixed_prefill()
    scheduler.transactions.clear()
    scheduler.connector.publication_blocked = {'a'}
    assert not scheduler._throttle_mixed_prefill()
    scheduler.connector.publication_blocked.clear()
    scheduler.prefill_capacity_bound = True
    assert not scheduler._throttle_mixed_prefill()
    scheduler.prefill_capacity_bound = False
    scheduler.running = [prefill]
    assert not scheduler._throttle_mixed_prefill()
    scheduler.running = [request]
    assert not scheduler._throttle_mixed_prefill()
    scheduler.waiting = [object()]
    assert scheduler._throttle_mixed_prefill()
    scheduler.prefill_cadence = 1
    assert not scheduler._throttle_mixed_prefill()


def test_legacy_entry_in_query_profile_leaves_mtp_allocation_headroom(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy
    scheduler, request, _ = setup_scheduler()
    scheduler._bounded_target = scheduler.request_planning = True
    scheduler.query_policy = SpeculativeQueryPolicy(16,32,48,4)
    request.num_prompt_tokens = 200
    monkeypatch.setattr(Scheduler,'_reserve_prefill_lookahead',lambda self,r,h,n:n)
    tokens = scheduler._reserve_prefill_lookahead(request,32,52)
    assert tokens==32 and 32+tokens+7<=scheduler.query_policy.capacity
    scheduler.query_policy = SpeculativeQueryPolicy(16,32,256,4)
    assert scheduler._reserve_prefill_lookahead(request,32,272)==128


@pytest.mark.parametrize('history',[0,48])
@pytest.mark.parametrize('planned',[False,True])
def test_native_padded_one_token_tail_retains_full_verification_shape(monkeypatch,history,planned):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy
    scheduler,request,_=setup_scheduler()
    scheduler._bounded_target=True;scheduler.num_spec_tokens=4
    scheduler.query_policy=SpeculativeQueryPolicy(16,32,48,5)
    request.num_prompt_tokens=history+1;request.max_tokens=1
    request.kv_transfer_params={'kvmem_request_plan':{'kvmem_reselect':'off'}} if planned else {}
    monkeypatch.setattr(Scheduler,'_reserve_prefill_lookahead',lambda self,r,h,n:n)
    assert scheduler._reserve_prefill_lookahead(request,history,5)==5
    assert scheduler._reserve_prefill_lookahead(request,history,1)==1


def test_suspended_refresh_commits_all_group_tables_and_resumes_request():
    scheduler, request, pools = setup_scheduler()
    scheduler.begin_refresh(request, 32)
    assert request.next_decode_eligible_step == 2**62
    assert [p.get_num_free_blocks() for p in pools] == [10, 10]
    command = scheduler.connector.commands[0]
    scheduler.connector.update_connector_output(KVConnectorOutput(
        kv_connector_worker_meta=ControlCompletions([("a", command.command_id)])))
    assert request.next_decode_eligible_step == 7
    assert [p.get_num_free_blocks() for p in pools] == [12, 12]
    assert not scheduler.transactions
    assert all(len(row) == 2 for row in scheduler.table_updates["a"])
    assert all(m.offsets["a"] == 32 and len(m.req_to_blocks["b"]) == 2
               for m in scheduler.window_managers.values())


def test_second_group_prepare_failure_rolls_back_first_without_suspending():
    scheduler, request, pools = setup_scheduler()
    scheduler.window_managers[1].free("a")
    with pytest.raises(ValueError, match="unallocated"):
        scheduler.begin_refresh(request, 32)
    assert request.next_decode_eligible_step == 7
    assert not scheduler.window_managers[0].pending
    assert not scheduler.transactions
    assert not scheduler.connector.pending
    assert pools[0].get_num_free_blocks() == 10


def test_commit_validates_all_groups_before_freeing_any_page():
    scheduler, request, pools = setup_scheduler()
    scheduler.begin_refresh(request, 32)
    command = scheduler.connector.commands[0]
    # Simulate an illegal native mutation while the worker uses the old table.
    manager = scheduler.window_managers[1]
    manager.native.allocate_new_blocks("a", 80, 80)
    with pytest.raises(RuntimeError, match="allocation changed"):
        scheduler._complete_refresh("a", command.command_id)
    assert pools[0].get_num_free_blocks() == 10
    assert len(scheduler.window_managers[0].req_to_blocks["a"]) == 4
    assert request.next_decode_eligible_step == 2**62


def test_cancelled_request_commits_before_free_but_never_publishes_worker_table():
    scheduler, request, pools = setup_scheduler()
    scheduler.begin_refresh(request, 32)
    command = scheduler.connector.commands[0]
    request.is_finished = lambda: True
    scheduler._complete_refresh("a", command.command_id)
    assert not scheduler.table_updates
    for m in scheduler.window_managers.values():
        m.free("a")
    assert [p.get_num_free_blocks() for p in pools] == [14, 14]


def test_refresh_pages_cannot_be_preempted_before_ack():
    scheduler, request, _ = setup_scheduler()
    scheduler.begin_refresh(request, 32)
    assert not scheduler._request_blocks_can_be_freed(request)


@pytest.mark.parametrize('native_field', [False, True])
def test_published_rows_include_blocks_allocated_in_resume_step(monkeypatch, tmp_path, native_field):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.connector import ControlMetadata
    scheduler, request, _ = setup_scheduler()
    scheduler.begin_refresh(request, 32)
    command = scheduler.connector.commands[0]
    scheduler._complete_refresh("a", command.command_id)
    scheduler._probe_submitted = True
    monkeypatch.setenv("KVMEM_RUNNER_PROBE_DIR", str(tmp_path))

    def allocate_on_resume(self, throttle):
        for manager in self.window_managers.values():
            manager.allocate_new_blocks("a", 80, 80)
        return (SimpleNamespace(block_table_updates=None) if native_field else
                SimpleNamespace(kv_connector_metadata=ControlMetadata()))

    monkeypatch.setattr(Scheduler, "schedule", allocate_on_resume)
    output = scheduler.schedule()
    updates = output.block_table_updates if native_field else output.kv_connector_metadata.block_table_updates
    assert all(len(row) == 3 for row in updates["a"])
    assert not scheduler.table_updates


def test_query_command_precedes_counter_advance_and_ack_uses_actual_selected_extent(monkeypatch, tmp_path):
    from vllm_kvmem.admission import QueryPolicy
    scheduler, request, pools = setup_scheduler()
    scheduler._bounded_target = True
    scheduler.query_policy = QueryPolicy(16, 32, 32)
    request.num_computed_tokens = 48
    request.num_prompt_tokens = 80
    output = SimpleNamespace(num_scheduled_tokens={"a": 16}, has_sync_kv_loads=False)
    monkeypatch.setenv("KVMEM_RUNNER_PROBE_DIR", str(tmp_path))
    metadata = scheduler._build_kv_connector_meta(scheduler.connector, output)
    command, = metadata.commands
    assert output.has_sync_kv_loads
    assert request.num_computed_tokens == 48
    assert all(ticket.history_end == 48 and ticket.replay_tokens == 16 for _, ticket in command.groups)
    assert all(len(ticket.retained_block_ids) == 3 for _, ticket in command.groups)
    request.num_computed_tokens += 16  # Native scheduler advance, not plugin accounting.
    scheduler.connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=
        ControlCompletions([("a", command.command_id)], {"a": 16})))
    assert all(m.offsets["a"] == 32 for m in scheduler.window_managers.values())
    assert all(len(row) == 2 for row in scheduler.table_updates["a"])
    assert request.num_computed_tokens == 64 and request.next_decode_eligible_step == 7
    assert [p.get_num_free_blocks() for p in pools] == [12, 12]


def test_invalid_worker_extent_cannot_partially_free_native_groups():
    scheduler, request, pools = setup_scheduler()
    scheduler.begin_refresh(request, 32)
    command = scheduler.connector.commands[0]
    with pytest.raises(ValueError, match="selected extent"):
        scheduler._complete_refresh("a", command.command_id, selected_tokens=33)
    assert [p.get_num_free_blocks() for p in pools] == [10, 10]
    assert request.request_id in scheduler.transactions


def test_lifecycle_snapshot_does_not_confuse_released_credits_with_deferred_physical_pages(tmp_path, monkeypatch):
    import json
    from test_capacity import make_gate, request
    gate, _, pool = make_gate(blocks=5)
    req = request('a')
    assert gate.allocate_slots(req, 64) is not None
    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler._bounded_target = True
    scheduler.kv_cache_manager = gate
    scheduler.requests = {'a': req}
    scheduler.transactions = {}
    scheduler.connector = SimpleNamespace(pending={}, sessions=None)
    monkeypatch.setenv('KVMEM_LIFECYCLE_TRACE', '1')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    scheduler._write_lifecycle_snapshot()
    read = lambda: json.loads((tmp_path / 'scheduler_resources.json').read_text())
    assert read()['owned_blocks'] == 4 and read()['reservations'] == ['a']
    held = gate.pop_blocks_for_free(req)
    scheduler.requests.clear()
    scheduler._write_lifecycle_snapshot()
    assert not read()['reservations'] and read()['free_blocks'] == 0
    pool.free_blocks(reversed(held))
    scheduler._write_lifecycle_snapshot()
    assert read()['free_blocks'] == read()['total_blocks'] - 1


def test_import_and_query_refresh_share_one_native_forward_but_commit_in_order(monkeypatch, tmp_path):
    from vllm_kvmem.session_control import SessionLedger, SessionBinding, SessionNotice
    from vllm_kvmem.admission import QueryPolicy
    from test_session_control import request as session_request, first_turn
    scheduler, req, pools = setup_scheduler()
    for manager in scheduler.window_managers.values():
        manager.free('a')
    events = []
    def prepare(request_id, history, window):
        for manager in scheduler.window_managers.values():
            manager.prepare_import(request_id, history, window)
    def commit(request_id, history):
        events.append('import')
        for manager in scheduler.window_managers.values():
            assert manager.commit_import(request_id, history, SimpleNamespace(query=lambda: True))
    sessions = SessionLedger(2, 4, prepare, lambda _: None, commit)
    scheduler.connector.sessions = sessions
    old = session_request('old', tokens=list(range(65)))
    old.num_computed_tokens, old.num_prompt_tokens = 48, 65
    assert sessions.lookup(old, 0) == 0
    sessions.allocated(old, 0)
    sessions.capture_before_final_prefill(old, 17)
    binding = sessions.take_commands().bindings[0]
    sessions.finished(old)
    sessions.consume(SessionNotice('retired', binding.namespace, binding.generation, 'old', 48, 32))
    new = session_request('new', tokens=list(range(65)))
    new.num_computed_tokens, new.num_prompt_tokens = 48, 65
    new.num_in_flight_tokens, new.next_decode_eligible_step = 0, 7
    new.is_finished = lambda: False
    scheduler.requests = {'new': new}
    assert sessions.lookup(new, 0) == 48
    for manager in scheduler.window_managers.values():
        manager.add_local_computed_blocks('new', (), 0, 48)
        manager.allocate_external_computed_blocks('new', 0, 48)
        manager.allocate_new_blocks('new', 64, 64)
    sessions.allocated(new, 48)
    scheduler._bounded_target = True
    scheduler.query_policy = QueryPolicy(16, 32, 32)
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    output = SimpleNamespace(num_scheduled_tokens={'new': 16}, has_sync_kv_loads=True)
    meta = scheduler._build_kv_connector_meta(scheduler.connector, output)
    assert len(meta.sessions.bindings) == len(meta.commands) == 1
    command = meta.commands[0]
    scheduler.connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        completed=[('new', command.command_id)], selected_windows={'new': 16},
        sessions=[SessionNotice('restored', binding.namespace, binding.generation, 'new', 48, 32)])))
    assert events == ['import']
    assert all(not m.imports and not m.pending and m.offsets['new'] == 32
               for m in scheduler.window_managers.values())
    assert all(len(row) == 2 for row in scheduler.table_updates['new'])
    for manager in scheduler.window_managers.values():
        manager.free('new'); manager.free('b')
    assert [p.get_num_free_blocks() for p in pools] == [16, 16]
