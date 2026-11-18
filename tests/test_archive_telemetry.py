"""In-step metrics survive unsampled publication/reclamation and stay bounded."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
from threading import RLock
from types import SimpleNamespace

import pytest

from vllm_kvmem import _native
from vllm_kvmem.core.archive_telemetry import ArchiveResourceReporter
from vllm_kvmem.core.memory_storage import MemoryArchive, MemoryTier


def test_cpu_publication_peak_survives_unsampled_full_reclamation(tmp_path):
    archive = MemoryArchive(str(tmp_path), 8, 4)
    repo = _native.Repository(4, 4)
    leases = [repo.reserve(i*4, 4).commit() for i in range(3)]
    try:
        archive.write('a', leases, b'x'*24).result(5)
        # No io_usage sampling occurred while any payload was resident.
        for lease in leases:
            handle = lease.handle
            assert repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda r: archive.reclaim('a', r)) == _native.ReclaimResult.RECLAIMED
        metrics = archive.io_usage()
        assert metrics['cpu_resident_bytes'] == 0
        assert metrics['cpu_peak_bytes'] == 24
        assert metrics['cpu_peak_scope'] == 'lifetime_published_payload_bytes'
    finally:
        assert archive.close()
    closed = archive.io_usage()
    assert closed['cpu_authority_bytes'] == closed['cpu_resident_bytes'] == archive.history_used_bytes() == 0
    assert closed['cpu_peak_bytes'] == 24


def test_failed_publication_does_not_inflate_published_payload_peak():
    class PartialFailure(dict):
        def update(self, values):
            self[next(iter(values))] = next(iter(values.values()))
            raise MemoryError('partial insert')
    tier = MemoryTier(4, 3)
    allocations = [tier.allocate(i) for i in range(3)]
    tier.publish_records(allocations[:1], [b'abcd'])
    assert tier.cpu_peak_bytes == 4
    tier.records = PartialFailure(tier.records)
    with pytest.raises(MemoryError):
        tier.publish_records(allocations[1:], [b'efgh', b'ijkl'])
    assert len(tier.records) == 1 and tier.cpu_peak_bytes == 4


def test_compact_reporter_throttles_in_step_counters_and_host_reads_independently(tmp_path):
    now, calls = [0.], []
    class Archive:
        def io_usage(self):
            calls.append('archive')
            return dict(cpu_resident_bytes=8, cpu_peak_bytes=16)
    def host():
        calls.append('host')
        return dict(worker=dict(rss_bytes=100))
    reporter = ArchiveResourceReporter(tmp_path, Archive(), clock=lambda: now[0], memory_sampler=host)
    (tmp_path/'archive.json').write_text('{"completed":["preserved"]}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: reporter.publish(2, 0), range(100))) == 1
    assert calls.count('archive') == calls.count('host') == 1
    now[0] = .2
    assert reporter.publish(2, 1)
    assert calls.count('archive') == 2 and calls.count('host') == 1
    now[0] = 1.2
    assert reporter.publish(1, 1)
    assert calls.count('archive') == 3 and calls.count('host') == 2
    row = json.loads((tmp_path/'archive_resources.json').read_text())
    assert row['resource_sample'] == 3 and row['active_requests'] == 1
    assert row['cpu_resident_bytes'] == 8
    assert json.loads((tmp_path/'archive.json').read_text()) == dict(completed=['preserved'])
    assert not (tmp_path/'archive_resources.tmp').exists()


def test_worker_feedback_publishes_lightweight_metrics_outside_publication_lock():
    from vllm_kvmem.serving.capture_runtime import CaptureRuntime
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    lock = RLock()
    runtime._publication_lock = lambda: lock
    runtime._raise_publication_error = lambda: None
    runtime._busy_requests = lambda: set()
    runtime.authorities, runtime.retiring = {}, {'retiring': object()}
    calls = []
    def publish(active, retiring):
        assert not lock._is_owned()
        calls.append((active, retiring))
    runtime._archive_resource_reporter = SimpleNamespace(publish=publish)
    assert runtime.archive_capacity() == {}
    assert calls == [(0, 1)]  # No finish/retirement report was needed.


def test_native_arena_without_exact_high_water_abi_does_not_invent_one(tmp_path):
    archive = MemoryArchive(str(tmp_path), 8, 2, native_payloads=True)
    try:
        metrics = archive.io_usage()
        assert metrics['cpu_resident_bytes'] == 0
        assert metrics['cpu_peak_bytes'] is None
        assert metrics['cpu_peak_scope'] == 'unavailable_for_native_payload_arena'
    finally:
        deadline = time.monotonic() + 5
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
    closed = archive.io_usage()
    assert closed['cpu_authority_bytes'] == closed['cpu_resident_bytes'] == closed['cpu_index_bytes'] == 0
    assert closed['cpu_peak_bytes'] is None


def test_checkpoint_storage_observations_are_separate_from_history(tmp_path):
    checkpoints = {'storage': dict(cpu_resident_bytes=16, ssd_used_bytes=32)}
    reporter = ArchiveResourceReporter(tmp_path,
        SimpleNamespace(io_usage=lambda: dict(cpu_resident_bytes=8, ssd_used_bytes=64)),
        memory_sampler=lambda: {}, checkpoint_sampler=lambda: checkpoints)
    reporter.publish(1, 0)
    report = json.loads((tmp_path/'archive_resources.json').read_text())
    assert report['cpu_resident_bytes'] == 8 and report['ssd_used_bytes'] == 64
    assert report['request_checkpoints'] == checkpoints


def test_always_installed_publication_feedback_observes_live_cpu_without_repository_gc(tmp_path):
    from vllm_kvmem.serving.capture_runtime import CaptureRuntime
    archive = MemoryArchive(str(tmp_path), 8, 4)
    repo = _native.Repository(4, 4)
    leases = [repo.reserve(i*4, 4).commit() for i in range(2)]
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = SimpleNamespace(repository_gc=False, cpu_scatter_min_tokens=32)
    runtime.authorities, runtime.retiring = {'a': object()}, {}
    runtime.queue = [SimpleNamespace(request_ids=('a',), request_tokens=(64,),
        tokens=64, published=set())]
    def forbidden():
        pytest.fail('Repository-GC archive feedback is disabled for this configuration')
    runtime.archive_capacity = forbidden
    def host():
        assert not runtime._publication_lock()._is_owned()
        return {}
    now = [0.]
    runtime._archive_resource_reporter = ArchiveResourceReporter(tmp_path, archive,
        clock=lambda: now[0], memory_sampler=host)
    try:
        archive.write('a', leases[:1], b'a'*8).result(5)
        assert runtime.publication_blocked_requests() == {'a'}
        first = json.loads((tmp_path/'archive_resources.json').read_text())
        assert first['cpu_resident_bytes'] == 8 and first['active_requests'] == 1
        archive.write('a', leases[1:], b'b'*8).result(5)
        now[0] = .2
        assert runtime.publication_blocked_requests() == {'a'}
        second = json.loads((tmp_path/'archive_resources.json').read_text())
        assert second['cpu_resident_bytes'] == 16 and second['resource_sample'] == 2
        assert not (tmp_path/'archive.json').exists()  # No retirement/full report.
    finally:
        assert archive.close()
