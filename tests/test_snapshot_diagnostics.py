from types import SimpleNamespace

import pytest

from vllm_kvmem.serving.snapshot_diagnostics import SnapshotCadence


def test_periodic_snapshots_publish_state_transitions_and_forced_final_state():
    now = [10.]
    cadence = SnapshotCadence(1., clock=lambda: now[0])
    assert cadence.due(('active', 'a'))
    for _ in range(100):
        assert not cadence.due(('active', 'a'))
    now[0] += .2
    assert cadence.due(('retiring', 'a'))
    assert not cadence.due(('retiring', 'a'))
    now[0] += 1.
    assert cadence.due(('retiring', 'a'))
    assert cadence.due(('retiring', 'a'), force=True)


def test_zero_interval_keeps_per_step_diagnostics(monkeypatch):
    monkeypatch.setenv('KVMEM_SNAPSHOT_INTERVAL_SECONDS', '0')
    cadence = SnapshotCadence(clock=lambda: 1.)
    assert all(cadence.due('same') for _ in range(4))


@pytest.mark.parametrize('interval', [-1., float('inf'), float('nan')])
def test_invalid_snapshot_interval_is_rejected(interval):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        SnapshotCadence(interval)


def test_session_snapshot_hashes_immutable_prefix_once_and_invalidates_on_replace(monkeypatch):
    import hashlib
    from dataclasses import replace
    from test_session_control import ledger, first_turn

    control, _ = ledger()
    binding, _ = first_turn(control)
    original = hashlib.sha256
    hashed = []

    def sha256(tokens):
        hashed.append(tokens)
        return original(tokens)

    monkeypatch.setattr('vllm_kvmem.serving.session_control.hashlib.sha256', sha256)
    first = control.snapshot()
    for _ in range(50):
        assert control.snapshot() == first
    assert len(hashed) == 1
    entry = control.entries[binding.namespace]
    newer = entry.tokens[:-4] + b'\xff\x00\x00\x00'
    entry.tokens = newer
    assert control.snapshot()[0]['prefix_sha256'] == original(newer).hexdigest()
    assert len(hashed) == 2
    # A recycled namespace with a new generation must not reuse the old hash.
    control.entries[binding.namespace] = replace(entry, generation=entry.generation+1,
                                                tokens=b'\x02\x00\x00\x00')
    assert control.snapshot()[0]['prefix_sha256'] == original(b'\x02\x00\x00\x00').hexdigest()
    assert len(hashed) == 3


def make_scheduler(tmp_path, monkeypatch):
    pytest.importorskip('vllm')
    from vllm_kvmem.serving.scheduler import KVMemScheduler
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
    scheduler._scheduled_queries = {'prefill': {'32': 1}, 'decode': {'5': 1}}
    scheduler.prefill_cadence = 1
    scheduler._cadence_requested = 0
    scheduler._snapshot_cadences = {
        key: SnapshotCadence(60., clock=lambda: 0.) for key in ('resources', 'queries')}
    monkeypatch.setenv('KVMEM_LIFECYCLE_TRACE', '1')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    return scheduler, req, pool


def test_steady_steps_skip_page_walk_but_final_deferred_free_is_immediately_visible(tmp_path, monkeypatch):
    import json
    scheduler, req, pool = make_scheduler(tmp_path, monkeypatch)
    assert scheduler._write_lifecycle_snapshot()
    report = lambda: json.loads((tmp_path/'scheduler_resources.json').read_text())
    assert report()['owned_blocks'] == 4
    groups = scheduler.kv_cache_manager.coordinator.single_type_managers
    scheduler.kv_cache_manager.coordinator.single_type_managers = None
    # This would fail if the unchanged snapshot still walked every native page.
    assert not scheduler._write_lifecycle_snapshot()
    scheduler.kv_cache_manager.coordinator.single_type_managers = groups
    held = scheduler.kv_cache_manager.pop_blocks_for_free(req)
    scheduler.requests.clear()
    assert scheduler._write_lifecycle_snapshot()
    assert report()['free_blocks'] == 0 and not report()['reservations']
    pool.free_blocks(reversed(held))
    assert scheduler._write_lifecycle_snapshot()
    assert report()['free_blocks'] == report()['total_blocks']-1


def test_retirement_update_flushes_every_histogram_count_before_engine_goes_idle(tmp_path, monkeypatch):
    import json
    from vllm.v1.core.sched.scheduler import Scheduler
    scheduler, req, _ = make_scheduler(tmp_path, monkeypatch)
    scheduler._write_lifecycle_snapshot()
    scheduler._write_scheduled_queries()
    report = lambda: json.loads((tmp_path/'scheduled_queries.json').read_text())
    scheduler._scheduled_queries['decode']['5'] += 99
    scheduler._write_scheduled_queries()
    assert report()['queries']['decode']['5'] == 1

    def finish(self):
        self.kv_cache_manager.free(req)
        self.requests.clear()
        return 'finished'

    monkeypatch.setattr(Scheduler, 'update_from_output', finish)
    assert scheduler.update_from_output() == 'finished'
    assert report()['queries']['decode']['5'] == 100
    assert json.loads((tmp_path/'scheduler_resources.json').read_text())['requests'] == 0


def test_shutdown_flushes_latest_counters_even_without_lifecycle_trace(tmp_path, monkeypatch):
    import json
    from vllm.v1.core.sched.scheduler import Scheduler
    scheduler, _, _ = make_scheduler(tmp_path, monkeypatch)
    scheduler._write_scheduled_queries()
    scheduler._scheduled_queries['decode']['5'] += 9
    monkeypatch.delenv('KVMEM_LIFECYCLE_TRACE')
    monkeypatch.setattr(Scheduler, 'shutdown', lambda self: None)
    scheduler.shutdown()
    assert json.loads((tmp_path/'scheduled_queries.json').read_text())['queries']['decode']['5'] == 10


def test_new_query_shape_is_published_without_waiting_for_period(tmp_path, monkeypatch):
    import json
    scheduler, _, _ = make_scheduler(tmp_path, monkeypatch)
    scheduler._write_scheduled_queries()
    scheduler._scheduled_queries['prefill']['64'] = 1
    scheduler._write_scheduled_queries()
    assert json.loads((tmp_path/'scheduled_queries.json').read_text())['queries']['prefill']['64'] == 1


def test_periodic_query_counts_do_not_wait_for_disk_and_final_state_is_exact(tmp_path, monkeypatch):
    import json
    from threading import Event
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.serving import report_writer

    scheduler, _, _ = make_scheduler(tmp_path, monkeypatch)
    monkeypatch.setenv('KVMEM_LIFECYCLE_TRACE', '0')
    monkeypatch.setattr(Scheduler, 'shutdown', lambda self: None)
    now = [0.]
    scheduler._snapshot_cadences['queries'] = SnapshotCadence(1., clock=lambda: now[0])
    scheduler.prefill_decode_burst = True
    scheduler._burst_stats = {'bursts': 1}
    scheduler._write_scheduled_queries()
    write = report_writer.write_report
    writing, release = Event(), Event()
    snapshots = []

    def hold_periodic_write(output, name, report, **kwargs):
        writing.set()
        assert release.wait(timeout=5)
        snapshots.append(report)
        write(output, name, report, **kwargs)

    monkeypatch.setattr(report_writer, 'write_report', hold_periodic_write)
    now[0] += 1.
    scheduler._scheduled_queries['decode']['5'] = 2
    try:
        scheduler._write_scheduled_queries()
        assert writing.wait(timeout=5)
        scheduler._scheduled_queries['decode']['5'] = 3
        scheduler._burst_stats['bursts'] = 2
    finally:
        release.set()
        scheduler.shutdown()
    assert [(r['queries']['decode']['5'], r['decode_burst']['bursts']) for r in snapshots] == [(2, 1), (3, 2)]
    assert json.loads((tmp_path/'scheduled_queries.json').read_text()) == snapshots[-1]
