"""CPU integration of readiness gates with the pinned native scheduling loop.

Real Request, FCFS queues, KVCacheManager, KVMem connector/session/restore gate
and Scheduler.schedule are used. No GPU runner is created. Cached-request
serialization is irrelevant here and replaced with an empty output record;
the native admission, skipping, allocation, metadata and counter paths run.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip('vllm')
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

from vllm_kvmem.active_residency import ActiveResidency
from vllm_kvmem.capacity import ReservedKVCacheManager
from vllm_kvmem.connector import ControlMetadata
from vllm_kvmem.restore_admission import RestoreAdmission
from vllm_kvmem.scheduler import KVMemScheduler
from vllm_kvmem.session_control import SessionLedger, SessionNotice
from vllm_kvmem.window_manager import WorkingSetManager
from test_connector import pair


def request(rid, *, session=None, prompt=33):
    return Request(rid, list(range(prompt)), SamplingParams(max_tokens=8), None,
                   session_id=session, cache_salt='tenant')


def native_fixture(monkeypatch, tmp_path, *, cold_busy=lambda: False, max_seqs=8):
    monkeypatch.delenv('KVMEM_BOUNDED_TARGET', raising=False)
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    config = KVCacheConfig(num_blocks=129, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['target'], FullAttentionSpec(block_size=16, num_kv_heads=2,
                                                     head_size=64, dtype=torch.uint8))])
    native = KVCacheManager(config, 1024, 512, 16, enable_caching=False)
    window = WorkingSetManager(native.coordinator.single_type_managers[0], 32, 32)
    native.coordinator.single_type_managers = (window,)
    cfg = NS(speculative_config=None, max_concurrent_batches=1,
             cache_config=NS(enable_prefix_caching=False, mamba_cache_mode='none'))
    cache = ReservedKVCacheManager(native, cfg)
    now = [10.]
    residency = ActiveResidency({0: window}, native.block_pool, ttl_ms=1000,
                               max_fraction=.5, clock=lambda: now[0])
    gate = RestoreAdmission(residency=residency, cold_busy=cold_busy)
    connector, _ = pair()

    def abort(rid):
        # A provisional hot lease must be returned before import prep ends.
        assert rid not in window.resident_imports
        window.abort_import_preparation(rid)

    sessions = SessionLedger(32, 16, window.prepare_import, abort,
                             lambda *_: None)
    connector.sessions = sessions
    connector.restore_admission = gate
    connector.active_residency = residency
    cache.active_residency = residency
    cache.restore_admission = gate
    cache.resident_entry = sessions.requests.get

    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler.vllm_config = cfg
    scheduler.scheduler_config = NS(max_num_seqs=max_seqs, max_num_batched_tokens=512,
        long_prefill_token_threshold=0, enable_chunked_prefill=True)
    scheduler.kv_cache_config = config
    scheduler.kv_cache_manager = cache
    scheduler.window_managers = {0: window}
    scheduler.connector, scheduler.ec_connector = connector, None
    scheduler.running = []
    scheduler.waiting, scheduler.skipped_waiting = FCFSRequestQueue(), FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.can_skip_drafts = False
    scheduler.deferred_waiting = set()
    scheduler.local_prefill_interval = 1
    scheduler.last_prefill_step = -1
    scheduler.num_executed_steps = 0
    scheduler.max_num_active_reqs = max_seqs
    scheduler.adaptive_long_prefill_threshold = False
    scheduler.aux_output_connector = None
    scheduler.is_mm_encoder_only = False
    scheduler.prefix_replay_tokens = 0
    scheduler.prefix_replay_group_ids = ()
    scheduler._kv_fetch_stages = None
    scheduler.requests, scheduler.transactions, scheduler.table_updates = {}, {}, {}
    scheduler.current_step = 0
    scheduler.max_num_scheduled_tokens = 512
    scheduler.max_num_running_reqs = max_seqs
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_model_len = 1024
    scheduler.block_size = 16
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.num_prefill_lookahead = 1
    scheduler.num_lookahead_tokens = scheduler.num_spec_tokens = 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.need_mamba_block_aligned_split = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.is_encoder_decoder = False
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.lora_config = None
    scheduler.log_stats = False
    scheduler.prefill_capacity_bound = False
    scheduler.prefill_cadence = 1
    scheduler._bounded_target = False
    scheduler._probe_submitted = True
    scheduler.connector_prefix_cache_stats = None
    scheduler._inflight_prefills = set()
    scheduler.use_v2_model_runner = True
    scheduler.model_uses_mrope = False
    scheduler.dynamic_sd_lookup = None
    scheduler.reset_preempted_req_ids, scheduler.finished_req_ids = set(), set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler.encoder_cache_manager = NS(get_freed_mm_hashes=lambda: [],
                                         get_manager_metadata=lambda: None)
    scheduler._make_cached_request_data = lambda *args: CachedRequestData.make_empty()
    # This fixture tests admission rather than the already separately qualified
    # worker restore transaction. Still run native connector metadata creation.
    scheduler._build_kv_connector_meta = Scheduler._build_kv_connector_meta.__get__(scheduler)
    return scheduler, window, sessions, residency, gate, now


def enqueue(scheduler, *requests):
    for req in requests:
        scheduler.requests[req.request_id] = req
        scheduler.waiting.add_request(req)


def checkpoint(sessions, window, residency, session, *, hot=False):
    old = request('old-' + session, session=session, prompt=49)
    assert sessions.lookup(old, 0) == 0
    sessions.allocated(old, 0)
    old.num_computed_tokens = 32
    sessions.capture_before_final_prefill(old, 17)
    binding = sessions.take_commands().bindings[0]
    sessions.finished(old)
    notice = SessionNotice('retired', binding.namespace, binding.generation,
                           old.request_id, 32, 32)
    sessions.consume(notice)
    if hot:
        window.allocate_new_blocks(old.request_id, 32, 32)
        residency.consume(replace(notice, resident_ready=True))
        window.free(old.request_id)
    return sessions.entries[binding.namespace]


def decode(scheduler, req):
    scheduler.requests[req.request_id] = req
    assert scheduler.kv_cache_manager.allocate_slots(req, req.num_prompt_tokens) is not None
    req.num_computed_tokens = req.num_prompt_tokens
    req.append_output_token_ids(42)
    req.status = RequestStatus.RUNNING
    scheduler.running.append(req)


def test_native_waiting_skips_busy_cold_then_admits_hot_and_fresh(monkeypatch, tmp_path):
    s, window, sessions, residency, gate, _ = native_fixture(monkeypatch, tmp_path, cold_busy=lambda: True)
    checkpoint(sessions, window, residency, 'cold')
    checkpoint(sessions, window, residency, 'hot', hot=True)
    cold, hot, fresh = request('cold', session='cold'), request('hot', session='hot'), request('fresh')
    enqueue(s, cold, hot, fresh)
    output = s.schedule()
    assert set(output.num_scheduled_tokens) == {'hot', 'fresh'}
    assert [r.request_id for r in s.running] == ['hot', 'fresh']
    assert s._waiting_requests() == [cold]
    assert cold.status == RequestStatus.WAITING and cold.num_computed_tokens == 0
    assert 'cold' not in s.kv_cache_manager.reservations and not window.req_to_blocks.get('cold')
    assert set(gate.permits) == {'hot'} and gate.permits['hot'].allocated
    assert not window.resident_imports and not residency.entries


def test_eight_cold_waiters_cannot_fill_all_native_slots(monkeypatch, tmp_path):
    s, window, sessions, residency, gate, _ = native_fixture(monkeypatch, tmp_path)
    cold = []
    for i in range(8):
        checkpoint(sessions, window, residency, f'cold-{i}')
        cold.append(request(f'cold-{i}', session=f'cold-{i}'))
    fresh = [request(f'fresh-{i}') for i in range(7)]
    enqueue(s, *cold, *fresh)
    output = s.schedule()
    expected = {'cold-0', *(r.request_id for r in fresh)}
    assert set(output.num_scheduled_tokens) == expected
    assert {r.request_id for r in s.running} == expected and len(s.running) == 8
    assert {r.request_id for r in s._waiting_requests()} == {r.request_id for r in cold[1:]}
    assert set(gate.permits) == {'cold-0'}
    assert all(not window.req_to_blocks.get(r.request_id) and
               r.request_id not in s.kv_cache_manager.reservations for r in cold[1:])


def test_expired_hot_view_uses_busy_cold_lane_and_does_not_block_fresh(monkeypatch, tmp_path):
    s, window, sessions, residency, gate, now = native_fixture(monkeypatch, tmp_path, cold_busy=lambda: True)
    checkpoint(sessions, window, residency, 'expired', hot=True)
    now[0] += 2
    expired, fresh = request('expired', session='expired'), request('fresh')
    enqueue(s, expired, fresh)
    output = s.schedule()
    assert set(output.num_scheduled_tokens) == {'fresh'}
    assert s._waiting_requests() == [expired]
    assert not residency.entries and not window.resident_imports and not gate.permits


@pytest.mark.parametrize('reason', ['token_gate', 'allocator_failure'])
def test_native_early_break_returns_provisional_hot_pages_and_lane(monkeypatch, tmp_path, reason):
    s, window, sessions, residency, gate, _ = native_fixture(monkeypatch, tmp_path)
    entry = checkpoint(sessions, window, residency, 'hot', hot=True)
    view = residency.entries[entry.namespace]
    hot = request('hot', session='hot')
    enqueue(s, hot)
    if reason == 'token_gate':
        s._reserve_prefill_lookahead = lambda *args: 0
    else:
        monkeypatch.setattr(s.kv_cache_manager.native, 'allocate_slots', lambda *args, **kwargs: None)
    output = s.schedule()
    assert output.total_num_scheduled_tokens == 0 and not s.running
    assert not gate.permits and not window.resident_imports
    assert residency.entries[entry.namespace] is view
    assert not window.req_to_blocks.get('hot') and not s.kv_cache_manager.reservations
    assert sessions.requests['hot'].phase == 'lookup'


def test_cancel_before_native_allocation_rolls_back_before_session_abort(monkeypatch, tmp_path):
    s, window, sessions, residency, gate, _ = native_fixture(monkeypatch, tmp_path)
    entry = checkpoint(sessions, window, residency, 'hot', hot=True)
    hot = request('hot', session='hot')
    assert s.connector.get_num_new_matched_tokens(hot, 0) == (32, False)
    assert 'hot' in window.resident_imports and 'hot' in gate.permits
    assert s.connector.request_finished_all_groups(hot, ()) == (False, None)
    assert not gate.permits and not window.resident_imports and not window.imports
    assert residency.entries[entry.namespace].history == 32
    assert entry.phase == 'idle' and 'hot' not in sessions.requests


def test_blocked_running_row_does_not_prevent_peer_decode_or_new_waiter(monkeypatch, tmp_path):
    s, _, _, _, _, _ = native_fixture(monkeypatch, tmp_path)
    b, a, fresh = request('blocked', prompt=32), request('ready', prompt=32), request('fresh')
    decode(s, b)
    decode(s, a)
    enqueue(s, fresh)
    s.connector.publication_blocked = {'blocked'}
    output = s.schedule()
    assert output.num_scheduled_tokens == {'ready': 1, 'fresh': 33}
    assert b.num_computed_tokens == 32 and b.num_in_flight_tokens == 0
    assert a.num_computed_tokens == 33 and a.num_in_flight_tokens == 1
    assert s.running == [b, a, fresh]


def test_all_blocked_still_emits_zero_token_control_then_resumes(monkeypatch, tmp_path):
    s, _, _, _, _, _ = native_fixture(monkeypatch, tmp_path)
    a, b = request('a', prompt=32), request('b', prompt=32)
    decode(s, a)
    decode(s, b)
    s.connector.publication_blocked = {'a', 'b'}
    output = s.schedule()
    assert output.total_num_scheduled_tokens == 0 and not output.num_scheduled_tokens
    assert isinstance(output.kv_connector_metadata, ControlMetadata)
    assert s.connector.has_pending_push_work()
    assert a.num_in_flight_tokens == b.num_in_flight_tokens == 0
    # Readiness feedback can arrive on the connector-only iteration. A future
    # public forward must not be required to break this dependency cycle.
    s.connector.publication_blocked.clear()
    next_output = s.schedule()
    assert next_output.num_scheduled_tokens == {'a': 1, 'b': 1}
    assert not s.connector.has_pending_push_work()


def test_stale_blocked_waiting_id_does_not_turn_native_skip_into_break(monkeypatch, tmp_path):
    s, _, _, _, _, _ = native_fixture(monkeypatch, tmp_path)
    stale, fresh = request('stale'), request('fresh')
    enqueue(s, stale, fresh)
    s.connector.publication_blocked = {'stale'}
    output = s.schedule()
    assert output.num_scheduled_tokens == {'stale': 33, 'fresh': 33}
