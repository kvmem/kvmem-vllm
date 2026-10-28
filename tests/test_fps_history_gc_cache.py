"""Retired archive generations must not accumulate in the derived FPS cache."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event, RLock
from types import SimpleNamespace as NS

import pytest
import torch

import vllm_kvmem.query_key_fps as module


def case(*, compact=False, blocks=96):
    archive = NS(_lock=RLock())
    # Eight cache rows, while logical history and repository generations grow.
    row_bytes = 2*4*2*8*(1 if compact else 2)+2048+2*4*2
    cache = module.RawKeyFPSCache(archive, 2, 2, 8, dtype=torch.float16,
                                  max_bytes=8*row_bytes, compact_fp8=compact)
    raw = torch.randn(2, blocks, 32, 2, 8,
        generator=torch.Generator().manual_seed(518)).to(torch.float8_e4m3fn)
    records = [NS(history_begin=i*32, count=32,
        handle=NS(slot=i % 8, generation=i // 8+1)) for i in range(blocks)]
    lookup = {(r.handle.slot, r.handle.generation):i for i, r in enumerate(records)}
    reads = []
    def read(handles):
        rows = [lookup[(h.slot, h.generation)] for h in handles]
        reads.append(rows)
        return [(raw[:, row], None) for row in rows]
    authority = NS(history_end=blocks*32, read_records=read)
    return NS(archive=archive, cache=cache, raw=raw, records=records,
              authority=authority, reads=reads)


def identity(record):
    return record.history_begin, record.handle.slot, record.handle.generation


def build(c, rows, *, epoch=None):
    raw = c.raw.view(torch.uint8).index_select(1, torch.tensor(rows)).view(c.raw.dtype)
    return c.cache.prebuild([c.records[i] for i in rows], raw,
        epoch=c.cache.epoch if epoch is None else epoch, optimized=True, device='cpu')


def check_snapshot(c, rows):
    with c.cache.use():
        keys, offsets, stats = c.cache.snapshot(c.authority, [c.records[i] for i in rows])
        raw = c.raw.view(torch.uint8).index_select(1, torch.tensor(rows)).view(c.raw.dtype)
        expected, wanted_offsets = module.select_key_representatives(raw, [32]*len(rows))
        assert torch.equal(keys.float(), expected.float())
        assert torch.equal(offsets, wanted_offsets.to(torch.int8))
        assert stats['cache_hit_blocks'] == stats['cache_prebuilt_hit_blocks'] == len(rows)
        assert stats['cache_rebuilt_blocks'] == 0
    assert not c.reads


@pytest.mark.parametrize('compact', [False, True])
def test_retirement_reuses_noncontiguous_rows_without_rebuilding_live_history(compact):
    c = case(compact=compact)
    try:
        assert build(c, list(range(8)))['status'] == 'published'
        pointer = c.cache.tensor.untyped_storage().data_ptr()
        charged = c.archive._query_window_cache_bytes
        assert c.cache.retire_records([identity(c.records[1]), identity(c.records[5])]) == 2
        assert build(c, [8, 9])['status'] == 'published'
        check_snapshot(c, [0, 2, 3, 4, 6, 7, 8, 9])
        assert c.cache.tensor.untyped_storage().data_ptr() == pointer
        assert c.cache.capacity == 8
        assert c.archive._query_window_cache_bytes == charged == c.cache.bytes
    finally:
        c.cache.clear()


@pytest.mark.parametrize('compact', [False, True])
def test_repeated_gc_stays_bounded_without_query_time_compaction(compact):
    c = case(compact=compact)
    try:
        assert build(c, list(range(8)))['status'] == 'published'
        pointer = c.cache.tensor.untyped_storage().data_ptr()
        live = list(range(8))
        for begin in range(8, 96, 2):
            victims = live[2:4]  # Keep two old hot/sink-like rows throughout.
            assert c.cache.retire_records([identity(c.records[i]) for i in victims]) == 2
            live = live[:2]+live[4:]+[begin, begin+1]
            result = build(c, [begin, begin+1])
            assert result['status'] == 'published' and result['cache_rebuilt_blocks'] == 2
            assert c.cache.capacity == len(c.cache._slot_by_start) == 8
            assert not c.cache._free_slots
            assert len(c.cache._prebuilt_signatures) == 8
            assert c.archive._query_window_cache_bytes == c.cache.bytes <= c.cache.max_bytes
            assert c.cache.tensor.untyped_storage().data_ptr() == pointer
        # 3K logical history used only eight cache rows, with an original hole.
        assert live == [0, 1, 90, 91, 92, 93, 94, 95]
        check_snapshot(c, live)
    finally:
        c.cache.clear()
    assert c.archive._query_window_cache_bytes == 0


def test_delayed_old_identity_cannot_remove_replacement_at_same_logical_start():
    c = case()
    try:
        build(c, list(range(8)))
        old = identity(c.records[2])
        c.records[2].handle.generation += 100
        assert build(c, [2])['status'] == 'published'
        assert c.cache.retire_records([old, (old[0], 99, old[2])]) == 0
        assert c.cache._slot_by_start[old[0]] == 2
        check_snapshot(c, list(range(8)))
        assert c.cache.retire_records([identity(c.records[2])]*2) == 1
        assert c.cache._free_slots == [2]
    finally:
        c.cache.clear()


def test_unpublished_retired_row_invalidates_running_and_queued_prebuild(monkeypatch):
    c = case()
    build(c, list(range(8)))
    c.cache.retire_records([identity(c.records[i]) for i in (6, 7)])
    entered, release = Event(), Event()
    original = module.select_key_representatives_optimized
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'select_key_representatives_optimized', blocked)
    epoch = c.cache.epoch
    try:
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(build, c, [8, 9], epoch=epoch)
            try:
                assert entered.wait(5)
                # Row 8 has no published signature yet. Its retirement must
                # still fence the job without waiting for expensive CPU FPS.
                assert pool.submit(c.cache.retire_records,
                    [identity(c.records[8])]).result(timeout=1) == 0
                assert c.cache.epoch == epoch+1
            finally:
                release.set()
            assert pending.result(timeout=5)['status'] == 'stale'
        assert build(c, [8], epoch=epoch)['status'] == 'stale'
        assert c.cache._slot_by_start == {i*32:i for i in range(6)}
        assert build(c, [9, 10])['status'] == 'published'
        check_snapshot(c, [0, 1, 2, 3, 4, 5, 9, 10])
    finally:
        c.cache.clear()


def test_retirement_during_growth_discards_unpublished_slab_and_reuses_old_hole(monkeypatch):
    c = case()
    build(c, [0, 1])
    entered, release = Event(), Event()
    original = torch.Tensor.copy_
    pointer = c.cache.tensor.untyped_storage().data_ptr()
    charged = c.cache.bytes
    def blocked(self, other, *args, **kwargs):
        if other.untyped_storage().data_ptr() == pointer:
            entered.set()
            assert release.wait(5)
        return original(self, other, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', blocked)
    try:
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(build, c, [2])
            try:
                assert entered.wait(5)
                assert pool.submit(c.cache.retire_records,
                    [identity(c.records[0])]).result(timeout=1) == 1
            finally:
                release.set()
            assert pending.result(timeout=5)['status'] == 'stale'
        assert c.cache.capacity == 2 and c.cache._free_slots == [0]
        assert c.archive._query_window_cache_bytes == c.cache.bytes == charged
        assert build(c, [3])['status'] == 'published'
        check_snapshot(c, [1, 3])
    finally:
        c.cache.clear()


def test_clear_after_retirement_keeps_old_view_charged_until_last_use_exits():
    c = case()
    build(c, list(range(8)))
    with c.cache.use():
        keys, offsets, _ = c.cache.snapshot(c.authority, c.records[:8])
        expected, expected_offsets = keys.clone(), offsets.clone()
        charged = c.cache.bytes
        assert c.cache.retire_records([identity(c.records[2])]) == 1
        c.cache.clear()
        assert c.cache.bytes == 0 and not c.cache._free_slots
        assert c.archive._query_window_cache_bytes == charged
        assert torch.equal(keys, expected) and torch.equal(offsets, expected_offsets)
    assert c.archive._query_window_cache_bytes == 0 and not c.cache._retired


@pytest.mark.parametrize('compact', [False, True])
def test_retirement_preserves_active_views_and_charge_until_reuse_is_safe(compact):
    c = case(compact=compact)
    build(c, list(range(8)))
    try:
        with ThreadPoolExecutor(2) as pool:
            with c.cache.use():
                keys, offsets, _ = c.cache.snapshot(c.authority, c.records[:8])
                expected, expected_offsets = keys.float().clone(), offsets.clone()
                charged = c.archive._query_window_cache_bytes
                assert pool.submit(c.cache.retire_records,
                    [identity(c.records[1])]).result(timeout=1) == 1
                pending = pool.submit(build, c, [8])
                with pytest.raises(TimeoutError):
                    pending.result(timeout=.05)
                assert torch.equal(keys.float(), expected) and torch.equal(offsets, expected_offsets)
                assert c.archive._query_window_cache_bytes == charged
            assert pending.result(timeout=5)['status'] == 'published'
        check_snapshot(c, [0, 2, 3, 4, 5, 6, 7, 8])
    finally:
        c.cache.clear()
    assert c.archive._query_window_cache_bytes == 0


def test_failed_reused_row_publication_returns_slot_and_keeps_other_rows(monkeypatch):
    c = case()
    build(c, list(range(8)))
    c.cache.retire_records([identity(c.records[3])])
    original = torch.Tensor.copy_
    pointer = c.cache.indices.untyped_storage().data_ptr()
    def fail(self, *args, **kwargs):
        if self.untyped_storage().data_ptr() == pointer:
            raise OSError('injected reused-row offset failure')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', fail)
    try:
        with pytest.raises(OSError, match='reused-row'):
            build(c, [8])
        assert c.cache._free_slots == [3]
        assert 256 not in c.cache._slot_by_start
        monkeypatch.setattr(torch.Tensor, 'copy_', original)
        assert build(c, [8])['status'] == 'published'
        check_snapshot(c, [0, 1, 2, 4, 5, 6, 7, 8])
    finally:
        c.cache.clear()


def test_reused_rows_do_not_mistake_matching_endpoint_span_for_contiguous_order():
    c = case()
    try:
        # Reserve the slab in one batch, then use Query order to establish a
        # noncanonical physical layout without adding any cache capacity.
        build(c, list(range(8)))
        with c.cache.use(), c.cache._lock:
            c.cache._swap_slots(0, 6)
            c.cache._swap_slots(1, 5)
        assert c.cache.retire_records([identity(c.records[3])]) == 1
        c.records[5].handle.generation += 100
        c.records[6].handle.generation += 100
        # Dirty original starts [160,192,256] map to slots [1,0,3]. Their
        # endpoint span has length three but cannot use tensor[:, 1:4].
        result = build(c, [5, 6, 8])
        assert result['fps_publication_index_copy_batches'] == 1
        assert result['fps_publication_slice_batches'] == 0
        check_snapshot(c, [0, 1, 2, 4, 5, 6, 7, 8])
    finally:
        c.cache.clear()


@pytest.mark.parametrize('entry', [(1, 0, 1), (0, -1, 1), (0, 0, 0),
                                  (0, 0, True), (0, 0), ('0', 0, 1)])
def test_invalid_retirement_batch_is_rejected_before_epoch_or_cache_changes(entry):
    c = case()
    try:
        build(c, list(range(8)))
        epoch = c.cache.epoch
        with pytest.raises(ValueError, match='retired FPS'):
            c.cache.retire_records([identity(c.records[0]), entry])
        assert c.cache.epoch == epoch
        check_snapshot(c, list(range(8)))
    finally:
        c.cache.clear()
