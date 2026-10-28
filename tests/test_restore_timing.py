from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_kvmem.restore_timing import RestoreTiming


def test_disabled_observer_never_reads_either_clock():
    def forbidden():
        pytest.fail('Disabled restore observation must not read a clock')
    timing = RestoreTiming(clock=forbidden, thread_clock=forbidden)
    with timing.stage('unused'):
        pass
    assert not timing.summary()['enabled']
    assert timing.summary()['stages'] == [] and timing.summary()['totals'] == {}


def test_nested_and_repeated_stages_keep_cpu_and_wall_intervals_separate():
    wall = iter([10., 11., 12., 16., 20., 25.])
    cpu = iter([1., 2., 3., 4., 7., 9.])
    timing = RestoreTiming(True, clock=lambda: next(wall), thread_clock=lambda: next(cpu))
    with timing.stage('outer'):
        with timing.stage('inner'):
            pass
    with timing.stage('outer'):
        pass
    report = timing.summary()
    assert report['totals']['outer'] == dict(calls=2, wall_seconds=11., thread_cpu_seconds=5.)
    assert report['totals']['inner'] == dict(calls=1, wall_seconds=1., thread_cpu_seconds=1.)
    assert [r['name'] for r in report['stages']] == ['inner', 'outer', 'outer']
    assert report['stages'][1]['started_at'] == 10. and report['stages'][1]['finished_at'] == 16.
    assert 'Nested stages overlap' in report['scope']
    report['stages'][0]['wall_seconds'] = -1
    report['totals']['outer']['calls'] = 99
    assert timing.summary()['stages'][0]['wall_seconds'] == 1.
    assert timing.summary()['totals']['outer']['calls'] == 2


@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt])
def test_exception_is_recorded_and_propagates_without_replacement(error_type):
    wall, cpu = iter([7., 8.]), iter([2., 2.25])
    timing = RestoreTiming(True, clock=lambda: next(wall), thread_clock=lambda: next(cpu))
    error = error_type('original')
    with pytest.raises(error_type) as caught:
        with timing.stage('failed'):
            raise error
    assert caught.value is error
    assert timing.summary()['stages'] == [dict(name='failed', started_at=7., finished_at=8.,
        wall_seconds=1., thread_cpu_seconds=.25, status='error', error_type=error_type.__name__)]


def test_noop_summary_does_not_create_stages_or_read_clocks():
    timing = RestoreTiming(True, clock=lambda: pytest.fail('no clock without a stage'),
                           thread_clock=lambda: pytest.fail('no clock without a stage'))
    assert timing.summary()['stages'] == [] and timing.summary()['totals'] == {}


@pytest.mark.parametrize('enabled', [None, False, True])
def test_memory_restore_timing_preserves_plan_and_fenced_commit(enabled):
    from test_core import Event, append, small
    from vllm_kvmem.memory import Phase
    memory = small()
    append(memory, 32)
    snapshot = memory.checkpoint_window()
    timing = None if enabled is None else RestoreTiming(enabled)
    plan = memory.restore_window(snapshot, timing=timing)
    assert plan.total_window_tokens == 32
    assert [(row.block_id, row.to_base, row.skip, row.working_k_resident)
            for row in plan.remaps] == [(0, 0, False, False), (1, 16, False, False)]
    assert memory.phase is Phase.ASSEMBLING
    fence = Event()
    memory.attach_completion(fence)
    assert not memory.commit_selection()
    fence.done = True
    assert memory.commit_selection() and memory.phase is Phase.READY
    assert memory._store.working_records() == list(snapshot.records)
    if enabled:
        assert set(timing.summary()['totals']) == {
            'window_validate', 'saved_handles', 'current_handles', 'residency_reset', 'native_select'}
    elif timing is not None:
        assert timing.summary()['stages'] == []
    assert memory.close()


def test_memory_validation_error_is_observed_without_mutation():
    from test_core import append, small
    from vllm_kvmem.memory import Phase
    memory = small()
    append(memory, 32)
    snapshot = memory.checkpoint_window()
    bad = replace(snapshot, records=(snapshot.records[0], (*snapshot.records[1][:-1], 0)))
    before = memory._store.working_records()
    timing = RestoreTiming(True)
    with pytest.raises(ValueError, match='dense frame'):
        memory.restore_window(bad, timing=timing)
    assert memory.phase is Phase.READY and memory._store.working_records() == before
    assert set(timing.summary()['totals']) == {'window_validate'}
    assert timing.summary()['stages'][0]['status'] == 'error'
    assert memory.close()


def test_pool_claim_miss_and_prefix_comparison_are_optional_observation():
    from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool, CheckpointPhase, token_bytes
    pool = PromptCheckpointPool(1, 1024)
    timing = RestoreTiming(True)
    assert pool.claim(('tenant', 'missing'), 'new', [1, 2, 3], timing=timing) is None
    assert timing.summary()['stages'] == []
    checkpoint = SimpleNamespace(phase=CheckpointPhase.IDLE, model=SimpleNamespace(query=lambda: True),
                                 claim_graph_history=None, claim_graph_end=0,
                                 request_id='old', position=2, tokens=token_bytes([1, 2, 3], 2))
    pool.entries['session'] = checkpoint
    assert pool.claim('session', 'new', [1, 9, 3], timing=timing) is None
    assert checkpoint.phase is CheckpointPhase.IDLE
    assert pool.claim('session', 'new', [1, 2, 3], timing=timing) is checkpoint
    assert checkpoint.phase is CheckpointPhase.CLAIMED
    assert timing.summary()['totals']['token_prefix_compare']['calls'] == 2


def test_pool_prepare_observes_action_and_nested_window_without_changing_ownership(tmp_path):
    from test_prompt_checkpoints import saved, ready
    from test_core import Event
    pool, checkpoint, authority, archive, _, _ = saved(tmp_path)
    action_calls = []
    authority.action_history = SimpleNamespace(restore=lambda *a, **kw: action_calls.append((a, kw)))
    pool.retire(checkpoint, Event(True))
    ready(pool, checkpoint)
    timing = RestoreTiming(True)
    assert pool.claim(checkpoint.namespace, 'new', list(range(12)), timing=timing) is checkpoint
    plan = pool.prepare_claim_window(checkpoint, timing=timing)
    assert plan.total_window_tokens == 8 and len(action_calls) == 1
    assert checkpoint.restore_started
    assert set(timing.summary()['totals']) == {'token_prefix_compare', 'action_restore', 'window_restore',
        'window_validate', 'saved_handles', 'current_handles', 'residency_reset', 'native_select', 'restore_records'}
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True)) is authority
    del plan
    assert authority.close() and archive.close()
