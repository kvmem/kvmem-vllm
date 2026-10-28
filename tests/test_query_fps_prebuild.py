"""CPU qualification of optional FPS prebuild and accepted-history ownership."""
from contextlib import contextmanager
from threading import Event, Thread, get_ident
import time

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, _Ready
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.query_fps_prebuild import FPSPrebuildManager
from vllm_kvmem.query_key_fps import select_key_representatives


def complete_close(close):
    # Native retirement releases payload, Python index ownership, and scope
    # metadata in distinct nonblocking polls, including with policy mode OFF.
    deadline, retry = time.monotonic()+5, Event()
    while not close():
        assert time.monotonic() < deadline, 'native cleanup did not release every reader/index owner'
        retry.wait(.001)


@contextmanager
def harness(tmp_path, *, native=False, **manager_options):
    # Include a draft layer to prove that the immutable K reader takes only
    # target scoring layers. This fixture never imports or starts a model.
    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=192,
        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 2*32*1*4*2, 64,
        native_payloads=native)
    if native:
        gc = _native.HistoricalGcConfig()
        gc.mode, gc.background = _native.HistoricalGcMode.OFF, False
        archive.enable_history_gc(gc)
    manager = FPSPrebuildManager(device='cpu', **manager_options)
    authorities, gates = [], []

    def create(name='first'):
        authority = RequestAuthority(name, config, archive, 2, 1, 4,
            raw_dtype=torch.float8_e4m3fn, scoring_layers=1, resident_records=16,
            query_score_mode='key_fps_4', query_window_cache_bytes=1 << 20)
        authority.fps_prebuild = manager.bind(authority)
        authorities.append(authority)
        return authority

    try:
        yield manager, archive, create, gates
    finally:
        for gate in gates:
            gate.set()
        manager.close()
        for authority in authorities:
            complete_close(authority.close)
        if native:
            usage = archive.io_usage()
            assert usage['cpu_authority_bytes'] == usage['cpu_payload_readers'] == 0
            assert usage['cpu_index_owners'] == usage['cpu_index_bytes'] == 0
        complete_close(archive.close)


def append(authority, count, *, candidates=0, seed=4):
    raw = torch.randn(2, count+candidates, 1, 4,
        generator=torch.Generator().manual_seed(seed)).to(torch.float8_e4m3fn)
    values = torch.full(raw.shape, 199, dtype=torch.uint8)
    begin = authority.history_end
    assert authority.append(raw, values, history_begin=begin, local_rope_begin=begin,
        accepted=count)
    return raw[:, :count].clone()


def drain_writes(authority):
    for job, _ in tuple(authority.writes):
        job.result(5)
    authority.poll()


def cache_snapshot(authority):
    with authority.fps_prebuild.query(), authority.index_snapshot() as (records, _):
        with authority.query_window_cache.use():
            keys, offsets, stats = authority.query_window_cache.snapshot(authority, records,
                optimized=True, device='cpu')
            return keys.clone(), offsets.clone(), stats


def expected(raw):
    count = raw.shape[1]
    blocks = (count+31)//32
    padded = torch.zeros(1, blocks, 32, 1, 4, dtype=raw.dtype)
    counts = []
    for block in range(blocks):
        n = min(32, count-block*32)
        counts.append(n)
        padded[:, block, :n].copy_(raw[:1, block*32:block*32+n])
    keys, offsets = select_key_representatives(padded, counts)
    return keys.float(), offsets.to(torch.int8)


def gate_first_read(monkeypatch, archive, gates, namespace='first'):
    started, release = Event(), Event()
    gates.append(release)
    real = archive.read_region
    calls = []

    def gated(name, leases, *args, **kwargs):
        calls.append((name, tuple((lease.handle.slot, lease.handle.generation) for lease in leases)))
        if name == namespace and not started.is_set():
            started.set()
            assert release.wait(10), 'test did not release immutable FPS read'
        return real(name, leases, *args, **kwargs)

    monkeypatch.setattr(archive, 'read_region', gated)
    return started, release, calls


def test_only_published_accepted_full_blocks_are_prebuilt(tmp_path, monkeypatch):
    with harness(tmp_path) as (manager, archive, create, _):
        authority = create()
        visible = Event()
        real_write = archive.write

        class GatedPublication:
            def __init__(self, job): self.job = job
            def done(self): return visible.is_set() and self.job.done()
            def result(self, timeout=None): return self.job.result(timeout)

        monkeypatch.setattr(archive, 'write',
            lambda *args, **kwargs:GatedPublication(real_write(*args, **kwargs)))
        raw = append(authority, 35, candidates=29)
        authority.poll()
        assert manager.wait_idle(timeout=5)
        assert not any(authority.query_window_cache.signatures)
        assert authority.history_end == 35 and not next(iter(authority.records.values())).archived
        visible.set()
        drain_writes(authority)
        assert manager.wait_idle(timeout=5)
        keys, offsets, stats = cache_snapshot(authority)
        wanted, wanted_offsets = expected(raw)
        assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
        assert stats['cache_hit_blocks'] == 1 and stats['cache_rebuilt_blocks'] == 1
        assert offsets[:, -1, -1].eq(-1).all()


def test_worker_uses_immutable_descriptor_not_mutable_authority_methods(tmp_path, monkeypatch):
    with harness(tmp_path) as (manager, _, create, _):
        authority = create()
        producer_thread, real_poll = get_ident(), authority.poll

        def poll():
            assert get_ident() == producer_thread, 'prebuilder called authority.poll'
            return real_poll()

        monkeypatch.setattr(authority, 'poll', poll)
        monkeypatch.setattr(authority, 'read_score_keys',
            lambda *args, **kwargs:pytest.fail('prebuilder used mutable authority K reader'))
        monkeypatch.setattr(authority, 'read_records',
            lambda *args, **kwargs:pytest.fail('prebuilder used mutable authority record reader'))
        raw = append(authority, 64)
        drain_writes(authority)
        assert manager.wait_idle(timeout=5)
        keys, offsets, stats = cache_snapshot(authority)
        wanted, wanted_offsets = expected(raw)
        assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
        assert stats['cache_hit_blocks'] == 2 and stats['cache_rebuilt_blocks'] == 0


def test_full_queue_defers_without_blocking_publication_or_unbounded_leases(tmp_path, monkeypatch):
    with harness(tmp_path, max_pending_records=2, batch_records=1) as (manager, archive, create, gates):
        authority = create()
        started, release, _ = gate_first_read(monkeypatch, archive, gates)
        append(authority, 32)
        drain_writes(authority)
        assert started.wait(5)
        # This producer turn must return while the first background read waits.
        append(authority, 96, seed=5)
        drain_writes(authority)
        assert authority.history_end == 128
        assert not manager.wait_idle(timeout=.01)
        stats = authority.fps_prebuild.snapshot()
        assert stats['deferred_records'] > 0
        release.set()
        assert manager.wait_idle(timeout=5)
        _, _, stats = cache_snapshot(authority)
        assert stats['cache_hit_blocks'] + stats['cache_rebuilt_blocks'] == 4


def test_query_waits_only_for_its_owner_and_cancels_only_its_queued_work(tmp_path, monkeypatch):
    with harness(tmp_path, batch_records=1) as (manager, archive, create, gates):
        first, second = create(), create('second')
        started, release, _ = gate_first_read(monkeypatch, archive, gates)
        append(first, 32)
        drain_writes(first)
        assert started.wait(5)
        append(second, 32, seed=6)
        drain_writes(second)
        first_entered, second_entered = Event(), Event()
        errors = []

        def query(owner, entered):
            try:
                with owner.query(): entered.set()
            except BaseException as error:
                errors.append(error)

        t1 = Thread(target=query, args=(first.fps_prebuild, first_entered))
        t2 = Thread(target=query, args=(second.fps_prebuild, second_entered))
        t1.start()
        t2.start()
        try:
            assert second_entered.wait(2), 'unrelated owner Query waited for global prebuild queue'
            assert not first_entered.wait(.02)
            release.set()
            assert first_entered.wait(5)
        finally:
            release.set()
            t1.join(5)
            t2.join(5)
        assert not errors and not t1.is_alive() and not t2.is_alive()
        assert manager.wait_idle(timeout=5)
        # The second owner can enqueue again after leaving its Query guard.
        second.fps_prebuild.offer(list(second.records.values()))
        assert manager.wait_idle(timeout=5)
        _, _, stats = cache_snapshot(second)
        assert stats['cache_hit_blocks'] == 1 and stats['cache_rebuilt_blocks'] == 0


def test_cancel_invalidates_inflight_result_and_exact_query_repairs_cache(tmp_path, monkeypatch):
    with harness(tmp_path) as (manager, archive, create, gates):
        authority = create()
        started, release, _ = gate_first_read(monkeypatch, archive, gates)
        raw = append(authority, 32)
        drain_writes(authority)
        assert started.wait(5)
        authority.fps_prebuild.cancel()
        release.set()
        assert manager.wait_idle(timeout=5)
        assert not any(authority.query_window_cache.signatures)
        keys, offsets, stats = cache_snapshot(authority)
        wanted, wanted_offsets = expected(raw)
        assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
        assert stats['cache_rebuilt_blocks'] == 1


def test_close_returns_until_inflight_descriptor_is_released(tmp_path, monkeypatch):
    with harness(tmp_path) as (manager, archive, create, gates):
        authority = create()
        started, release, _ = gate_first_read(monkeypatch, archive, gates)
        append(authority, 32)
        drain_writes(authority)
        assert started.wait(5)
        assert not authority.close()
        assert authority.closing
        release.set()
        assert manager.wait_idle(timeout=5)
        assert authority.close()
        assert not authority.records and not any(authority.query_window_cache.signatures)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0


@pytest.mark.parametrize('preserve_prefix',[False,True])
def test_rewind_discards_old_inflight_suffix_and_new_branch_uses_new_keys(tmp_path, monkeypatch, preserve_prefix):
    with harness(tmp_path, batch_records=1) as (manager, archive, create, gates):
        authority = create()
        if preserve_prefix:
            from types import SimpleNamespace
            authority.graph_options = SimpleNamespace(query_mean_quota_fraction=.5)
        prefix = append(authority, 32)
        drain_writes(authority)
        assert manager.wait_idle(timeout=5)
        checkpoint = authority.memory.checkpoint()
        try:
            started, release, _ = gate_first_read(monkeypatch, archive, gates)
            append(authority, 32, seed=6)
            drain_writes(authority)
            assert started.wait(5)
            assert authority.begin_rewind(checkpoint) is not None
            assert any(sig is not None and sig[0] == 0 for sig in authority.query_window_cache.signatures) is preserve_prefix
            authority.attach_rewind_completion(_Ready())
            assert not authority.poll_rewind()
            assert authority.history_end == 32
            release.set()
            assert manager.wait_idle(timeout=5)
            assert authority.poll_rewind()
            assert all(sig is None or sig[0] < 32 for sig in authority.query_window_cache.signatures)
            suffix = append(authority, 32, seed=87)
            drain_writes(authority)
            assert manager.wait_idle(timeout=5)
            keys, offsets, _ = cache_snapshot(authority)
            wanted, wanted_offsets = expected(torch.cat((prefix, suffix), dim=1))
            assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
        finally:
            checkpoint = None


def test_optional_worker_failure_is_recorded_and_falls_back_to_exact_query(tmp_path, monkeypatch):
    with harness(tmp_path) as (manager, archive, create, _):
        authority = create()
        real = archive.read_region

        def fail(*args, **kwargs):
            raise OSError('injected prebuild archive failure')

        monkeypatch.setattr(archive, 'read_region', fail)
        raw = append(authority, 64)
        drain_writes(authority)
        assert manager.wait_idle(timeout=5)
        assert authority.fps_prebuild.snapshot()['failed_records'] > 0
        assert not any(authority.query_window_cache.signatures)
        monkeypatch.setattr(archive, 'read_region', real)
        keys, offsets, stats = cache_snapshot(authority)
        wanted, wanted_offsets = expected(raw)
        assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
        assert stats['cache_rebuilt_blocks'] == 2


def test_busy_archive_defers_optional_work_without_reporting_a_failed_build(tmp_path, monkeypatch):
    with harness(tmp_path) as (manager, archive, create, _):
        authority = create()
        real = archive.read_region
        monkeypatch.setattr(archive, 'read_region', lambda *args, **kwargs: None)
        raw = append(authority, 64)
        drain_writes(authority)
        assert manager.wait_idle(timeout=5)
        report = authority.fps_prebuild.snapshot()
        assert report['deferred_records'] == report['skip_archive_queue_records'] == 2
        assert not report.get('failed_records') and report['last_error'] is None
        monkeypatch.setattr(archive, 'read_region', real)
        keys, offsets, stats = cache_snapshot(authority)
        wanted, wanted_offsets = expected(raw)
        assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
        assert stats['cache_rebuilt_blocks'] == 2


def test_two_owners_with_same_native_slots_keep_distinct_keys(tmp_path):
    with harness(tmp_path) as (manager, _, create, _):
        first, second = create(), create('second')
        one, two = append(first, 64, seed=21), append(second, 64, seed=34)
        drain_writes(first)
        drain_writes(second)
        assert manager.wait_idle(timeout=5)
        assert [r.handle.slot for r in first.records.values()] == [r.handle.slot for r in second.records.values()]
        for authority, raw in ((first, one), (second, two)):
            keys, offsets, stats = cache_snapshot(authority)
            wanted, wanted_offsets = expected(raw)
            assert torch.equal(keys, wanted) and torch.equal(offsets, wanted_offsets)
            assert stats['cache_hit_blocks'] == 2 and stats['cache_rebuilt_blocks'] == 0


def test_borrowed_native_payload_reader_survives_permanent_retirement(tmp_path):
    with harness(tmp_path, native=True) as (manager, archive, create, _):
        authority = create()
        raw = append(authority, 32)
        drain_writes(authority)
        assert manager.wait_idle(timeout=5)
        handle = next(iter(authority.records.values())).handle
        lease = authority.memory.repository.pin(handle)
        sources = archive.borrow_read_sources(authority.memory.request_id, [lease])
        assert len(sources) == 1 and sources[0]
        try:
            authority.memory.repository.retire_permanently(handle)
            assert not authority.memory.repository.pin(handle)
            amount = 1*32*1*4
            destination = torch.zeros(1, 32, 1, 4, dtype=torch.float8_e4m3fn)
            archive.read_region(authority.memory.request_id, [lease], 0, amount,
                destination=destination.view(torch.uint8).numpy().reshape(-1),
                _sources=sources).result(5)
            assert torch.equal(destination.view(torch.uint8), raw[:1].view(torch.uint8))
        finally:
            sources.clear()
            lease.reset()
