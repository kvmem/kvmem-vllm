"""Independent native MeanK owners retain aliases and exact release credits."""
import gc
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.authority import Record, RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.historical_gc import HistoricalAuthorityGc
from vllm_kvmem.memory_storage import MemoryArchive, SplitHistoricalIndexOwner


def allocate(arena, shape=(2, 2), **kwargs):
    return MemoryArchive.allocate_index(SimpleNamespace(_native_payload_arena=arena),
        shape, mean=True, split_sums=True, **kwargs)


def close(arena):
    deadline = time.monotonic() + 5
    while not arena.close():
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_sum_reaper_releases_bytes_while_mean_and_snapshot_remain_exact():
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    sums, owner = allocate(arena)
    mean = owner.mean_view()
    sums.copy_(torch.tensor([[3., 5.], [17., -31.]]))
    torch.div(sums, 32, out=mean)
    expected = (sums / 32).half()
    alias = sums.detach().view(-1)
    fence = owner.completion
    assert arena.stats['external_owners'] == 2
    assert arena.stats['external_bytes'] == 24
    owner.release_sums()
    owner.release_sums()  # Repeated release does not double free/discount.
    del sums
    gc.collect()
    arena.advance()
    assert arena.stats['external_bytes'] == 24 and not fence.query()
    assert torch.equal(alias, torch.tensor([3., 5., 17., -31.]))
    del alias
    gc.collect()
    arena.advance()
    assert arena.stats['external_bytes'] == 8
    assert arena.stats['external_released_bytes'] == 16
    assert arena.stats['external_owners'] == 1 and not fence.query()
    with pytest.raises(RuntimeError, match='released'):
        owner.sum_view()
    del owner
    arena.advance()
    assert arena.stats['external_bytes'] == 8 and not fence.query()
    assert torch.equal(mean, expected)
    del mean
    gc.collect()
    arena.advance()
    assert fence.query() and arena.stats['external_bytes'] == 0
    assert arena.stats['external_released_bytes'] == 24
    close(arena)


def test_gc_completion_waits_for_sum_alias_even_when_mean_owner_finishes_first():
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    sums, owner = allocate(arena)
    mean = owner.mean_view()
    sums.copy_(torch.arange(4.).view(2, 2))
    collector = object.__new__(HistoricalAuthorityGc)
    collector.pending_index = {}
    record = SimpleNamespace(index_owner=owner)
    collector.defer_index((7, 19), record)
    owner.release_sums()
    del owner, record, mean
    gc.collect()
    arena.advance()
    assert not collector.index_released()
    assert arena.stats['external_bytes'] == 16
    assert torch.equal(sums, torch.arange(4.).view(2, 2))
    del sums
    gc.collect()
    arena.advance()
    assert collector.index_released() and not collector.pending_index
    assert arena.stats['external_bytes'] == arena.stats['external_owners'] == 0
    close(arena)


def test_split_failure_releases_first_owner_even_if_exception_traceback_is_retained():
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    calls = 0
    def fail_second(size):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError('injected second allocation failure')
        return arena._allocate_index(size)
    with pytest.raises(MemoryError, match='second allocation') as error:
        allocate(SimpleNamespace(_allocate_index=fail_second))
    assert error.traceback is not None
    gc.collect()
    arena.advance()
    assert arena.stats['external_owners'] == arena.stats['external_bytes'] == 0
    assert arena.stats['external_released_bytes'] == 16
    close(arena)


@pytest.mark.parametrize('stopped', [False, True])
def test_retirement_notification_is_generation_exact_once_while_index_fence_drains(stopped):
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    sums, owner = allocate(arena)
    handle = n.Handle(3, 17)
    key = (3, 17)
    records = {key: Record(handle, 352, 32, None, None, sums,
        archived=True, mean_key=owner.mean_view(), index_owner=owner)}
    del sums, owner
    event = SimpleNamespace(handle=handle, cpu=True)
    events, acknowledged = [event], []
    def acknowledge(value):
        acknowledged.append(value)
        events.remove(value)
        return True
    collector = object.__new__(HistoricalAuthorityGc)
    collector.scope = SimpleNamespace(completions=lambda _: list(events), acknowledge=acknowledge)
    collector.memory = SimpleNamespace(request_id='test', repository=SimpleNamespace(pin=lambda _: None))
    collector.archive = SimpleNamespace(retire_cpu_record=lambda *_: True)
    collector.stopped = stopped
    collector.reclaimed_records = 0
    collector.pending_index, collector.retired_index = {}, {}
    retired = []
    collector.drain(records, retired=retired)
    assert retired == [(352, 3, 17)] and not records
    assert collector.reclaimed_records == 1
    assert not acknowledged and key in collector.pending_index
    # Reliable CPU completions may be returned repeatedly until the reaper
    # actually frees the two index buffers. Notifications cannot repeat.
    collector.drain(records, retired=retired)
    assert retired == [(352, 3, 17)]
    gc.collect()
    arena.advance()
    collector.drain(records, retired=retired)
    assert retired == [(352, 3, 17)] and acknowledged == [event]
    assert not collector.pending_index and not collector.retired_index
    assert arena.stats['external_owners'] == arena.stats['external_bytes'] == 0
    close(arena)


def test_native_payload_reader_and_mean_snapshot_have_independent_retirement():
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    raw_handle = n.HistoricalPayloadHandle(0, 1)
    arena.publish(raw_handle, b'x' * 64)
    reader = arena.acquire(raw_handle)
    sums, owner = allocate(arena)
    means = owner.mean_view()
    sums.fill_(16.)
    torch.div(sums, 32, out=means)
    fence = owner.completion
    owner.release_sums()
    del sums, owner
    assert arena.retire(raw_handle) == n.HistoricalPayloadRetireResult.COMMITTED
    arena.advance()
    assert arena.stats['used_bytes'] == 64 and arena.stats['external_bytes'] == 8
    assert bytes(memoryview(reader)) == b'x' * 64
    assert not fence.query()
    del reader
    arena.advance()
    assert arena.stats['used_bytes'] == 0 and arena.stats['external_bytes'] == 8
    assert torch.equal(means, torch.full((2, 2), .5, dtype=torch.float16))
    del means
    gc.collect()
    arena.advance()
    assert fence.query() and arena.stats['external_bytes'] == 0
    close(arena)


@pytest.mark.parametrize('shape, error', [
    ((0, 2), ValueError), ((True, 2), ValueError), ((), ValueError),
    ((sys.maxsize // 6 + 1,), OverflowError), ((sys.maxsize, sys.maxsize), OverflowError),
])
def test_split_shape_and_allocation_overflow_fail_before_native_allocation(shape, error):
    def unexpected(_):
        pytest.fail('invalid geometry reached native allocation')
    with pytest.raises(error):
        SplitHistoricalIndexOwner(SimpleNamespace(_allocate_index=unexpected), shape)


def test_legacy_contiguous_owner_remains_buffer_compatible():
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    archive = SimpleNamespace(_native_payload_arena=arena)
    sums, owner = MemoryArchive.allocate_index(archive, (2, 2), mean=True)
    means = torch.frombuffer(owner, dtype=torch.float16, count=4, offset=16).view(2, 2)
    sums.fill_(12.)
    torch.div(sums, 32, out=means)
    assert torch.equal(means, torch.full((2, 2), .375, dtype=torch.float16))
    assert arena.stats['external_owners'] == 1 and arena.stats['external_bytes'] == 24
    with pytest.raises(ValueError, match='stored MeanK'):
        MemoryArchive.allocate_index(archive, (2, 2), mean=False, split_sums=True)
    del sums, means, owner
    close(arena)


def test_authority_sealed_sums_release_with_native_gc_and_live_partial_tail(tmp_path):
    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 2*32*8*2, 64, native_payloads=True)
    settings = n.HistoricalGcConfig()
    settings.mode, settings.background = n.HistoricalGcMode.OFF, False
    settings.ready_low, settings.ready_high = 2, 4
    worker = archive.enable_history_gc(settings)
    authority = RequestAuthority('compact', config, archive, 2, 1, 8,
        raw_dtype=torch.float8_e4m3fn, mean_k_backend='qw3_cuda',
        cpu_release_sealed_sums=True)
    raw = torch.randn(2, 97, 1, 8, generator=torch.Generator().manual_seed(29)).to(torch.float8_e4m3fn)
    values = torch.zeros(raw.shape, dtype=torch.uint8)
    try:
        assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
        authority.publish_sealed(lambda: None)
        worker.advance()
        authority.poll()
        records = sorted(authority.records.values(), key=lambda r: r.history_begin)
        assert [r.key_sum is None for r in records] == [True, True, True, False]
        assert archive.io_usage()['cpu_index_bytes'] == (3*2 + 6) * (2*8)
        assert archive.io_usage()['cpu_index_released_bytes'] == 3*4*(2*8)
        for record in records[:3]:
            expected = raw[:, record.history_begin:record.history_begin+32].float().mean(1).half()
            assert torch.equal(record.mean_key, expected)
        assert torch.equal(records[-1].key_sum, raw[:, 96].float())
        snapshot = records[0].mean_key.detach()
        expected_snapshot = snapshot.clone()
        records.clear()
        del record
        assert not authority.close()
        worker.advance()
        assert torch.equal(snapshot, expected_snapshot)
        del snapshot
    finally:
        deadline = time.monotonic()+5
        while not authority.close():
            worker.advance()
            assert time.monotonic() < deadline
            time.sleep(.001)
        deadline = time.monotonic()+5
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
    assert archive.io_usage()['cpu_index_bytes'] == archive.io_usage()['cpu_authority_bytes'] == 0


def test_full_arena_split_indices_use_two_descriptors_but_original_byte_charge():
    # Requires the matching native build: four record slots admit eight split
    # descriptors, with exactly the old four * six-byte-per-element payload.
    arena = n.HistoricalPayloadArena(4, 64, background=False)
    indices = [allocate(arena) for _ in range(4)]
    assert arena.stats['external_owners'] == 8
    assert arena.stats['external_bytes'] == 4 * 4 * 6
    with pytest.raises(RuntimeError, match='capacity'):
        arena._allocate_index(1)
    means = [owner.mean_view() for _, owner in indices]
    fences = [owner.completion for _, owner in indices]
    for sums, owner in indices:
        sums.fill_(2.)
        torch.div(sums, 32, out=owner.mean_view())
        owner.release_sums()
    del sums, owner
    indices.clear()
    gc.collect()
    arena.advance()
    assert arena.stats['external_bytes'] == 4 * 4 * 2
    assert arena.stats['external_owners'] == 4
    assert not any(fence.query() for fence in fences)
    assert all(torch.equal(mean, torch.full((2, 2), .0625, dtype=torch.float16)) for mean in means)
    means.clear()
    gc.collect()
    arena.advance()
    assert all(fence.query() for fence in fences)
    assert arena.stats['external_owners'] == arena.stats['external_bytes'] == 0
    close(arena)
