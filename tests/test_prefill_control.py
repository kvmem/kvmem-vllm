from types import SimpleNamespace

import pytest

from vllm_kvmem.prefill_control import BatchPrefillBudget, PrefillController
from vllm_kvmem.step_timing import StepTiming


def options():
    return SimpleNamespace(prefill_min_tokens=16, prefill_max_tokens=128,
        memory=SimpleNamespace(block_tokens=16), decode_step_target_ms=50,
        prefill_feedback_timeout_ms=5000)


def test_batch_credits_are_shared_fair_and_reset_without_delaying_idle_prefill():
    budget = BatchPrefillBudget(512, 32)
    for _ in range(5):
        budget.begin(decode_ready=True, contenders=2)
        assert budget.available('first') == 256
        budget.charge('first', 256)
        budget.charge('first', 256)  # Native retry must not double-charge.
        assert budget.available('second') == 256
        budget.charge('second', 256)
        assert budget.available('third') == 0
        with pytest.raises(ValueError, match='shared batch'):
            budget.charge('third', 1)
    budget.begin(decode_ready=False, contenders=2)
    assert budget.available('first') is None
    budget.charge('first', 8192)
    assert budget.available('second') is None


@pytest.mark.parametrize('contenders', [1, 2, 4, 7, 8])
def test_per_request_cap_does_not_divide_a_single_chunks_credits_between_peers(contenders):
    budget = BatchPrefillBudget(2048, 32, per_request_tokens=256)
    budget.begin(decode_ready=True, contenders=contenders)
    for index in range(contenders):
        assert budget.available(str(index)) == 256
        budget.charge(str(index), 256)
        budget.charge(str(index), 256)
    assert budget.remaining == 2048 - contenders*256
    budget.begin(decode_ready=False, contenders=contenders)
    assert budget.available('first') is None
    budget.charge('first', 8192)


def test_per_request_cap_never_increases_shared_budget_or_changes_zero_default():
    capped = BatchPrefillBudget(512, 32, per_request_tokens=256)
    original = BatchPrefillBudget(512, 32)
    for budget in [capped, original]:
        budget.begin(decode_ready=True, contenders=8)
        for index in range(8):
            assert budget.available(str(index)) == 64
            budget.charge(str(index), 64)
        assert budget.remaining == 0
    for invalid in [True, -32, 17, 1024]:
        with pytest.raises(ValueError):BatchPrefillBudget(512, 32, per_request_tokens=invalid)


def test_per_request_limit_parses_and_requires_an_enabled_aligned_batch_budget():
    from test_options import config
    from vllm_kvmem.options import RuntimeOptions
    values=dict(block_tokens=32,repository_blocks=8192,request_planning=True,prefill_batch_tokens=2048,prefill_request_tokens=256)
    assert RuntimeOptions.from_vllm(config({'kvmem':values})).prefill_request_tokens==256
    for invalid in [True,-32,17,4096]:
        with pytest.raises(ValueError,match='Per-request'):
            RuntimeOptions.from_vllm(config({'kvmem':values|dict(prefill_request_tokens=invalid)}))
    with pytest.raises(ValueError,match='Per-request'):
        RuntimeOptions.from_vllm(config({'kvmem':values|dict(prefill_batch_tokens=0)}))


@pytest.mark.parametrize('per_request', [False, True])
def test_scheduler_shared_budget_keeps_decode_and_atomic_query_independent(monkeypatch,per_request):
    from copy import copy
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.admission import QueryPolicy
    from test_scheduler_transactions import setup_scheduler
    scheduler, first, _ = setup_scheduler()
    scheduler._bounded_target = scheduler.request_planning = True
    scheduler.query_policy = QueryPolicy(32, 131072, 131072)
    first.num_prompt_tokens = 100000
    first.kv_transfer_params = {'kvmem_request_plan': {'kvmem_query_span': [19028, 23022]}}
    second = copy(first); second.request_id = 'b'
    scheduler.prefill_batch = BatchPrefillBudget(512, 32, per_request_tokens=256 if per_request else 0)
    scheduler.prefill_batch.begin(decode_ready=True, contenders=2)
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self,r,h,n:n)
    assert scheduler._reserve_prefill_lookahead(first, 32768, 2048) == 256
    assert scheduler._reserve_prefill_lookahead(second, 32768, 2048) == 256
    assert scheduler.prefill_batch.remaining == 0
    # Replaying Query preserves the full original reservation, even when the
    # ordinary budget is exhausted. It is visible as separate maintenance work.
    assert scheduler._reserve_prefill_lookahead(second, 19008, 16404) == 4014
    assert scheduler._reserve_prefill_lookahead(first, 100000, 5) == 5
    assert scheduler.prefill_batch.remaining == 0
    scheduler.prefill_batch.begin(decode_ready=False, contenders=2)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS', '2048')
    assert scheduler._reserve_prefill_lookahead(first, 32768, 16404) == 2048


def test_shared_credits_preserve_native_mtp_placeholder_tail(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.admission import QueryPolicy
    from test_scheduler_transactions import setup_scheduler
    scheduler, request, _ = setup_scheduler()
    scheduler._bounded_target = scheduler.request_planning = True
    scheduler.query_policy = QueryPolicy(16, 32, 64)
    scheduler.num_spec_tokens = 4
    request.num_prompt_tokens = 65
    request.kv_transfer_params = {'kvmem_request_plan': {'kvmem_reselect':'off'}}
    scheduler.prefill_batch = BatchPrefillBudget(16, 16)
    scheduler.prefill_batch.begin(decode_ready=True, contenders=1)
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self,r,h,n:n)
    scheduler.prefill_batch.charge('peer', 12)
    assert scheduler._reserve_prefill_lookahead(request, 64, 5) == 0
    assert request.num_computed_tokens == 64
    scheduler.prefill_batch.begin(decode_ready=True, contenders=1)
    assert scheduler._reserve_prefill_lookahead(request, 64, 5) == 5
    assert scheduler.prefill_batch.remaining == 11


def feedback(sequence, ms=80, **kw):
    return dict(sequence=sequence, step_ms=ms, model_ms=ms, maintenance_ms=0., submitted_at=10.,
                decode_requests=2, prefill_tokens=128, prefill_requests=1) | kw


def test_variable_prefill_cost_selects_feasible_chunk_from_matching_decode_baseline():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 30, prefill_tokens=0, prefill_requests=0),
                     feedback(2, 94)])  # 30 ms fixed + .5 ms/token.
    assert control.limit == 128  # One chunk size cannot identify marginal cost.
    control.observe([feedback(3, 62, prefill_tokens=64)])
    assert control.cap(decode_ready=True) == 32
    assert control.cap(decode_ready=False) == 128
    assert control.token_ms == .5 and control.fixed_ms == 30


def test_fixed_maintenance_does_not_cause_prefill_shrink():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 30, prefill_tokens=0),
                     feedback(2, 1040, model_ms=40, maintenance_ms=1000)])
    assert control.limit == 128 and control.decreases == 0
    assert control.last_maintenance_ms == 1000


def test_unattainable_fixed_mixed_cost_does_not_lock_chunks_at_floor():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 30, prefill_tokens=0), feedback(2, 180)])
    assert control.limit == 128
    # A naturally shorter chunk saves only 10 ms: fixed cost exceeds the target.
    control.observe([feedback(3, 170, prefill_tokens=64)])
    assert control.limit == 128 and control.unattainable == 1
    for n in range(4, 20):
        control.observe([feedback(n, 180)])
    assert control.limit == 128 and control.decreases == 0


def test_cost_fit_uses_only_matching_occupancy_and_requires_split_timing():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 30, prefill_tokens=0, decode_requests=1), feedback(2, 180)])
    assert control.limit == 128
    control.observe([feedback(3, 62, prefill_tokens=64, decode_requests=1)])
    assert control.limit == 128  # Never fit a slope across different occupancies.
    row = feedback(4, 62, prefill_tokens=64); row.pop('model_ms')
    control.observe([row])
    assert control.limit == 128 and control.observations == 2


def test_fixed_intercept_can_be_measured_without_a_pure_decode_sample():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 94), feedback(2, 62, prefill_tokens=64)])
    assert control.limit == 32
    assert control.fixed_ms == 30 and control.token_ms == .5


def test_changing_occupancy_and_expired_samples_never_restart_blind_shrinking():
    now = [10.]
    control = PrefillController(options(), clock=lambda:now[0])
    for sequence in range(1, 40):
        control.observe([feedback(sequence, 180, submitted_at=now[0],
                                  decode_requests=1+sequence % 7)])
        assert control.limit == 128
        now[0] += 1.
    assert control.decreases == 0


def test_unknown_new_occupancy_resets_previously_justified_small_cap():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 94), feedback(2, 62, prefill_tokens=64)])
    assert control.limit == 32
    control.observe([feedback(3, 180, decode_requests=3)])
    assert control.limit == 128 and control.fixed_ms is None and control.token_ms is None


def test_total_prefill_cost_is_shared_across_prefill_requests():
    control = PrefillController(options(), clock=lambda:10.)
    control.observe([feedback(1, 30, prefill_tokens=0),
        feedback(2, 94, prefill_tokens=256, prefill_requests=2),
        feedback(3, 62, prefill_tokens=128, prefill_requests=2)])
    assert control.limit == 32  # Total 64 tokens, .25 ms each; not 64 per request.


def test_feedback_is_fenced_by_sequence_age_and_mixed_public_work():
    clock = [10.]
    control = PrefillController(options(), clock=lambda:clock[0])
    control.observe([feedback(1, 30, prefill_tokens=0), feedback(2, 94),
                     feedback(3, 62, prefill_tokens=64)])
    assert control.limit == 32
    control.observe([feedback(3), feedback(0)])
    for sequence, field in ((4, 'decode_requests'), (5, 'prefill_tokens')):
        row = feedback(sequence); row[field] = 0
        control.observe([row])
    assert control.limit == 32 and control.observations == 2
    clock[0] = 16.
    control.observe([feedback(6)])
    assert control.cap(decode_ready=True) == 128
    assert control.observations == 2


def test_step_timing_never_waits_for_gpu_and_includes_sample_completion(tmp_path):
    class Event:
        def __init__(self): self.ready = False; self.recorded = False
        def record(self): self.recorded = True
        def query(self): return self.ready
        def elapsed_time(self, end):
            assert end.ready and self.recorded and end.recorded
            return 42.
    timing = StepTiming(1, tmp_path/'steps.jsonl', event_factory=Event, clock=lambda:10.)
    timing.begin()
    timing.model_begin(SimpleNamespace(num_scheduled_tokens=[1024,5,5], is_prefilling_np=[True,False,False]))
    assert not timing.poll() and timing.active is not None
    timing.end()
    assert not timing.poll() and len(timing.pending) == 1
    timing.pending[0]['end'].ready = True
    timing.pending[0]['model'].ready = True
    row, = timing.poll()
    assert row['step_ms'] == 42. and row['decode_requests'] == 2 and row['prefill_tokens'] == 1024
    assert row['prefill_requests'] == 1 and row['maintenance_ms'] == row['model_ms'] == 42.
    assert not timing.pending and (tmp_path/'steps.jsonl').is_file()


def test_pending_feedback_is_bounded_when_device_does_not_complete():
    class Event:
        def record(self): pass
        def query(self): return False
    timing = StepTiming(1, None, event_factory=Event)
    for _ in range(100): timing.begin(); timing.end()
    assert len(timing.pending) == 16 and timing.dropped == 84
    sampled = StepTiming(4, None, event_factory=Event)
    for _ in range(8): sampled.begin(); sampled.end()
    assert len(sampled.pending) == 2


def test_latency_only_connector_feedback_reaches_scheduler_without_session_work():
    from test_connector import pair, poll
    scheduler, worker = pair()
    seen = []
    scheduler.scheduler_step_feedback = seen.extend
    rows = [feedback(1)]
    worker.step_timings = lambda: list(rows)
    assert not poll(scheduler, worker).is_empty()
    assert seen == rows
    rows.clear()
    assert poll(scheduler, worker).is_empty()


def test_adaptive_chunks_never_split_explicit_query_replay(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.admission import QueryPolicy
    from test_scheduler_transactions import setup_scheduler
    scheduler, request, _ = setup_scheduler()
    scheduler._bounded_target = scheduler.request_planning = True
    scheduler.query_policy = QueryPolicy(16, 32, 64)
    scheduler.prefill_controller = PrefillController(options(), clock=lambda:10.)
    scheduler.prefill_controller.limit = 16; scheduler.prefill_controller.last_feedback = 10.
    scheduler._decode_ready = lambda:True
    request.num_prompt_tokens = 80
    request.kv_transfer_params = {'kvmem_request_plan':{'kvmem_query_span':[48,79]}}
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self,r,h,n:n)
    assert scheduler._reserve_prefill_lookahead(request, 0, 48) == 16
    # Query and its aligned replay reservation retain original semantics.
    assert scheduler._reserve_prefill_lookahead(request, 48, 32) == 31


@pytest.mark.parametrize('adaptive', [False, True])
def test_query_suffix_resumes_chunk_cap_after_checkpoint_fallback(monkeypatch, adaptive):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.admission import QueryPolicy
    from test_scheduler_transactions import setup_scheduler
    scheduler, request, _ = setup_scheduler()
    scheduler._bounded_target = scheduler.request_planning = True
    scheduler.query_policy = QueryPolicy(32, 131072, 16384)
    request.num_prompt_tokens = 100000
    request.kv_transfer_params = {'kvmem_request_plan': {'kvmem_query_span': [19028, 23022]}}
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self,r,h,n:n)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS', '2048')
    if adaptive:
        scheduler.prefill_controller = type('Cap', (), {'cap': lambda self, **kw: 1024})()
        scheduler._decode_ready = lambda: True
    cap = 1024 if adaptive else 2048
    assert scheduler._reserve_prefill_lookahead(request, 0, 16424) == cap
    # Query starts at its sealed block; even a 4K Query remains indivisible.
    assert scheduler._reserve_prefill_lookahead(request, 19008, 16424) == 4014
    # Immediate unaligned suffix and later aligned suffix both use normal caps.
    assert scheduler._reserve_prefill_lookahead(request, 23022, 16424) <= cap
    assert scheduler._reserve_prefill_lookahead(request, 32768, 16424) == cap


@pytest.mark.parametrize('changed', [dict(step_latency_interval=0), dict(prefill_min_tokens=17),
    dict(prefill_max_tokens=65536), dict(decode_step_target_ms=float('nan')), dict(adaptive_prefill=1)])
def test_invalid_control_configuration_fails_before_launch(changed):
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.options import RuntimeOptions
    cfg = dict(memory=KVMemConfig(block_tokens=16, select_budget=32, gen_budget=256, sink_blocks=0, recent_blocks=0),
               request_planning=True, adaptive_prefill=True, step_latency_interval=4,
               prefill_min_tokens=16, prefill_max_tokens=128)
    RuntimeOptions(**cfg)
    with pytest.raises(ValueError): RuntimeOptions(**(cfg | changed))


@pytest.mark.parametrize('pressure_only', [False, True])
def test_elastic_v2_controls_reach_native_options_without_changing_model(tmp_path, pressure_only):
    import json
    from test_serving_launcher import nvfp4_profile
    from vllm_kvmem.elastic_serve import build_launch
    model = nvfp4_profile(tmp_path)
    users = tmp_path/'users.json'
    users.write_text(json.dumps({'users':[{'id':'alice','api_key':'test-only-key'}]}))
    cfg = {key:model[key] for key in ('model','tokenizer')}
    cfg.update(users_file=users.name, context_mode='meank', draft_method='mtp', active_context_tokens=32768,
        execution_mode='compiled_graph', request_planning=True, archive_backend='cpu',
        async_refresh=True, async_restore=True, maintenance_max_steps=8, session_checkpoint_policy='pm',
        adaptive_prefill=True, step_latency_interval=4, decode_step_target_ms=60.,
        prefill_min_tokens=128, prefill_max_tokens=1024, prefill_batch_tokens=512,prefill_request_tokens=256,
        max_num_seqs=4, max_inflight=4, active_resident_ttl_ms=5000., active_resident_max_fraction=.5,
        maintenance_transfer_bytes_per_second=8*1024**3, maintenance_transfer_burst_bytes=64*1024**2)
    cfg['active_resident_pressure_only'] = pressure_only
    if pressure_only: cfg['active_resident_ttl_ms'] = 0.
    resolved, _ = build_launch(cfg, tmp_path, tmp_path/'run')
    native = resolved['mean_k_backend_profile']
    for key in ('async_restore','adaptive_prefill','step_latency_interval','decode_step_target_ms',
                'prefill_min_tokens','prefill_max_tokens','prefill_batch_tokens','prefill_request_tokens',
                'active_resident_ttl_ms','active_resident_pressure_only','active_resident_max_fraction',
                'maintenance_transfer_bytes_per_second','maintenance_transfer_burst_bytes'):
        assert native['kvmem'][key] == cfg[key]
    assert native['draft_method'] == 'mtp' and native['execution_mode'] == 'compiled_graph'
