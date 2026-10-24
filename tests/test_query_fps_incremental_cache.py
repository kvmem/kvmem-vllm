"""Accepted-block FPS prebuild must remain exact, bounded and cancelable."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, RLock
from types import SimpleNamespace as NS

import pytest
import torch

import vllm_kvmem.query_key_fps as module


@pytest.mark.parametrize('records, maximum, expected', [
    (0, 100000, 0), (1, 100000, 1), (33, 100000, 64),
    (4097, 100000, 8192), (8192, 100000, 8192),
    (8193, 100000, 9216), (8448, 100000, 9216),
    (9216, 100000, 9216), (9217, 100000, 10240),
    (16384, 100000, 16384), (16385, 100000, 17408),
    (8193, 9000, 9000), (9000, 9000, 9000),
])
def test_cache_capacity_does_not_double_after_full_256k_history(records, maximum, expected):
    assert module.fps_cache_capacity(records, maximum) == expected


def test_cache_capacity_rejects_requests_outside_budget_without_allocating():
    with pytest.raises(MemoryError, match='query_window_cache_bytes'):
        module.fps_cache_capacity(8193, 8192)
    for records, maximum in ((-1, 10), (1, -1), (1., 10), (True, 10)):
        with pytest.raises(ValueError, match='capacity'):
            module.fps_cache_capacity(records, maximum)


def fixture(blocks=8, *, archive=None, budget=1 << 20):
    raw = torch.randn(2, blocks, 32, 2, 8,
        generator=torch.Generator().manual_seed(172)).to(torch.float8_e4m3fn)
    records = [NS(history_begin=i*32, count=32, handle=NS(slot=i, generation=1))
        for i in range(blocks)]
    reads = []
    def read(handles):
        reads.append(tuple(h.slot for h in handles))
        return [(raw[:, h.slot, :records[h.slot].count], None) for h in handles]
    archive = archive or NS(_lock=RLock())
    cache = module.KeyFPSCache(archive, 2, 2, 8, dtype=torch.float16, max_bytes=budget)
    authority = NS(history_end=blocks*32, read_records=read, query_window_cache=cache,
        mean_k_backend='torch_reference')
    return NS(raw=raw, records=records, reads=reads, archive=archive, cache=cache, authority=authority)


def prebuild(case, rows, *, epoch=None):
    keys = case.raw.view(torch.uint8).index_select(1, torch.tensor(rows)).view(case.raw.dtype)
    return case.cache.prebuild([case.records[i] for i in rows], keys,
        epoch=case.cache.epoch if epoch is None else epoch, optimized=True, device='cpu')


def snapshot(case, rows=None):
    records = case.records if rows is None else [case.records[i] for i in rows]
    return case.cache.snapshot(case.authority, records)


def exact(case, rows=None):
    rows = list(range(len(case.records))) if rows is None else rows
    raw = case.raw.view(torch.uint8).index_select(1, torch.tensor(rows)).view(case.raw.dtype)
    keys, ids = module.select_key_representatives(raw, [case.records[i].count for i in rows])
    return keys.half(), ids.to(torch.int8)


def test_incremental_growth_preserves_prior_rows_and_query_never_reads_archive():
    case = fixture()
    try:
        for rows in ([0], [1], [2, 3], [4, 5, 6, 7]):
            result = prebuild(case, rows)
            assert result['status'] == 'published' and result['cache_rebuilt_blocks'] == len(rows)
            assert result['fps_gpu_groups'] == 0
            assert result['fps_cpu_fallback_groups'] == 4*len(rows)
        keys, ids, stats = snapshot(case)
        wanted, wanted_ids = exact(case)
        assert torch.equal(keys, wanted) and torch.equal(ids, wanted_ids)
        assert stats['cache_prebuilt_hit_blocks'] == stats['cache_hit_blocks'] == 8
        assert stats['cache_rebuilt_blocks'] == stats['cache_growth_rebuilt_blocks'] == 0
        assert not case.reads
        assert prebuild(case, [0, 1])['status'] == 'hit'
        assert case.archive._query_window_cache_bytes == case.cache.bytes
    finally:
        case.cache.clear()


def test_out_of_order_batches_and_logical_holes_reconcile_without_recomputation():
    case = fixture()
    try:
        assert prebuild(case, [2, 4])['status'] == 'published'
        assert prebuild(case, [0, 1])['status'] == 'published'
        rows = [0, 1, 2, 4]
        keys, ids, stats = snapshot(case, rows)
        wanted, wanted_ids = exact(case, rows)
        assert torch.equal(keys, wanted) and torch.equal(ids, wanted_ids)
        assert stats['cache_prebuilt_hit_blocks'] == 4 and not case.reads
        assert case.cache._slot_by_start == {case.records[i].history_begin:j for j, i in enumerate(rows)}
    finally:
        case.cache.clear()


@pytest.mark.parametrize('blocks, batches, query_rows', [
    (4, [[2, 3]], [0, 1, 2, 3]),
    (6, [[2, 4]], [0, 1, 2, 3, 4, 5]),
    (8, [[6, 7], [2, 4]], list(range(8))),
    (8, [[5, 7], [0, 3]], list(range(8))),
    (8, [[4, 6]], [0, 2, 4, 6]),
    (6, [[1, 2, 3, 4]], [0, 2, 3]),
    (4, [[3], [1]], [1, 2, 3]),
    (6, [[4, 5]], [0, 1, 2]),
])
def test_sparse_prebuild_survives_missing_prefix_insertion_growth_and_query_holes(
        blocks, batches, query_rows):
    case = fixture(blocks)
    try:
        for batch in batches:
            assert prebuild(case, batch)['status'] == 'published'
        built = {row for batch in batches for row in batch}
        expected_hits = len(built.intersection(query_rows))
        keys, ids, stats = snapshot(case, query_rows)
        expected, expected_ids = exact(case, query_rows)
        assert torch.equal(keys, expected) and torch.equal(ids, expected_ids)
        assert stats['cache_hit_blocks'] == stats['cache_prebuilt_hit_blocks'] == expected_hits
        assert stats['cache_rebuilt_blocks'] == len(query_rows)-expected_hits
        missing = tuple(row for row in query_rows if row not in built)
        assert case.reads == ([missing] if missing else [])
        assert case.cache._slot_by_start == {row*32:i for i, row in enumerate(query_rows)}
        # The canonical result is reusable without a second archive read.
        _, _, hot = snapshot(case, query_rows)
        assert hot['cache_hit_blocks'] == len(query_rows) and hot['cache_rebuilt_blocks'] == 0
    finally:
        case.cache.clear()


def test_only_new_rows_are_selected_and_partial_tail_is_query_only(monkeypatch):
    case = fixture(3)
    prebuild(case, [0])
    counts = []
    original = module.select_key_representatives_optimized
    def select(keys, valid_counts, **kwargs):
        counts.append(keys.shape[1])
        return original(keys, valid_counts, **kwargs)
    monkeypatch.setattr(module, 'select_key_representatives_optimized', select)
    assert prebuild(case, [0, 1])['cache_rebuilt_blocks'] == 1
    case.records[2].count = 3
    case.authority.history_end = 67
    with pytest.raises(ValueError, match='sealed'):
        prebuild(case, [2])
    keys, ids, stats = snapshot(case)
    wanted, wanted_ids = exact(case)
    assert torch.equal(keys, wanted) and torch.equal(ids, wanted_ids)
    assert counts == [1] and stats['cache_prebuilt_hit_blocks'] == 2
    assert case.reads == [(2,)]
    case.cache.clear()


def test_background_growth_budget_failure_skips_before_fps_then_query_repairs(monkeypatch):
    row_bytes = 2*4*2*8*2+2048+2*4*2
    case = fixture(2, budget=2*row_bytes)
    assert prebuild(case, [0])['status'] == 'published'
    original = module.select_key_representatives_optimized
    monkeypatch.setattr(module, 'select_key_representatives_optimized',
        lambda *args, **kwargs: pytest.fail('Budget rejection must precede FPS'))
    result = prebuild(case, [1])
    assert result['status'] == 'skipped' and result['skip_reason'] == 'cache_budget'
    assert case.cache.capacity == 1 and case.cache.signatures[0] is not None
    monkeypatch.setattr(module, 'select_key_representatives_optimized', original)
    keys, ids, stats = snapshot(case)
    wanted, wanted_ids = exact(case)
    assert torch.equal(keys, wanted) and torch.equal(ids, wanted_ids)
    assert stats['cache_growth_rebuilt_blocks'] == 1 and stats['cache_rebuilt_blocks'] == 2
    assert case.archive._query_window_cache_bytes == 2*row_bytes
    case.cache.clear()


def test_prebuild_never_evicts_other_owner_and_query_still_can():
    row_bytes = 2*4*2*8*2+2048+2*4*2
    a = fixture(3, budget=4*row_bytes)
    b = fixture(2, archive=a.archive, budget=4*row_bytes)
    prebuild(a, [0, 1]); prebuild(b, [0, 1])
    before = b.cache.epoch
    assert prebuild(a, [2])['status'] == 'skipped'
    assert b.cache.bytes == 2*row_bytes and b.cache.epoch == before
    snapshot(a)
    assert b.cache.bytes == 0 and b.cache.epoch > before
    assert a.archive._query_window_cache_bytes <= 4*row_bytes
    a.cache.clear(); b.cache.clear()


def test_clear_does_not_wait_for_selection_and_stale_work_never_publishes(monkeypatch):
    case = fixture(2)
    entered, resume = Event(), Event()
    original = module.select_key_representatives_optimized
    def blocked(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'select_key_representatives_optimized', blocked)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(prebuild, case, [0, 1])
        try:
            assert entered.wait(5)
            old_epoch = case.cache.epoch
            pool.submit(case.cache.clear).result(timeout=1)
            assert case.cache.epoch == old_epoch+1 and case.cache.bytes == 0
        finally:
            resume.set()
        assert future.result(timeout=5)['status'] == 'stale'
    assert not case.cache.signatures and case.archive._query_window_cache_bytes == 0


def test_clear_keeps_active_query_view_charged_until_consumer_finishes():
    case = fixture(2)
    prebuild(case, [0, 1])
    with case.cache.use():
        keys, ids, _ = snapshot(case)
        expected, expected_ids = keys.clone(), ids.clone()
        reserved = case.archive._query_window_cache_bytes
        with ThreadPoolExecutor(1) as pool:
            pool.submit(case.cache.clear).result(timeout=1)
        assert case.cache.tensor is None and case.cache.bytes == 0
        assert case.archive._query_window_cache_bytes == reserved
        assert torch.equal(keys, expected) and torch.equal(ids, expected_ids)
    assert case.archive._query_window_cache_bytes == 0


def test_growth_copy_does_not_block_clear_and_unpublished_capacity_is_released(monkeypatch):
    case = fixture(2)
    prebuild(case, [0])
    entered, resume = Event(), Event()
    original = torch.Tensor.copy_
    old_pointer = case.cache.tensor.untyped_storage().data_ptr()
    def blocked(self, other, *args, **kwargs):
        if other.untyped_storage().data_ptr() == old_pointer:
            entered.set()
            assert resume.wait(5)
        return original(self, other, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', blocked)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(prebuild, case, [1])
        try:
            assert entered.wait(5)
            pool.submit(case.cache.clear).result(timeout=1)
            assert case.cache.tensor is None
        finally:
            resume.set()
        assert future.result(timeout=5)['status'] == 'stale'
    assert case.cache.bytes == case.archive._query_window_cache_bytes == 0


def test_rewind_invalidates_queued_epoch_but_keeps_verified_prefix():
    case = fixture(4)
    prebuild(case, [0, 1, 2, 3])
    old_epoch = case.cache.epoch
    assert case.cache.rewind(64) == old_epoch+1
    assert prebuild(case, [2, 3], epoch=old_epoch)['status'] == 'stale'
    assert all(case.cache.signatures[:2]) and case.cache.signatures[2:4] == [None, None]
    case.authority.history_end = 64
    _, _, stats = snapshot(case, [0, 1])
    assert stats['cache_hit_blocks'] == stats['cache_prebuilt_hit_blocks'] == 2
    assert not case.reads
    case.cache.clear()


def test_canonical_reorder_failure_invalidates_all_touched_rows_then_repairs(monkeypatch):
    case = fixture(2)
    prebuild(case, [1]); prebuild(case, [0])
    original = torch.Tensor.copy_
    pointer = case.cache.indices.untyped_storage().data_ptr()
    def fail_offsets(self, *args, **kwargs):
        if self.untyped_storage().data_ptr() == pointer:
            raise OSError('canonical offsets failed')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', fail_offsets)
    with pytest.raises(OSError, match='canonical offsets'):
        snapshot(case)
    assert case.cache.signatures[:2] == [None, None]
    monkeypatch.setattr(torch.Tensor, 'copy_', original)
    keys, ids, stats = snapshot(case)
    wanted, wanted_ids = exact(case)
    assert torch.equal(keys, wanted) and torch.equal(ids, wanted_ids)
    assert stats['cache_rebuilt_blocks'] == 2 and case.reads == [(0, 1)]
    case.cache.clear()


def test_prebuild_publication_failure_is_retryable_without_publishing_any_row(monkeypatch):
    case = fixture(2)
    original = torch.Tensor.copy_
    def fail_offsets(self, *args, **kwargs):
        if case.cache.indices is not None and self.untyped_storage().data_ptr() == case.cache.indices.untyped_storage().data_ptr():
            raise OSError('offset publication failed')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', fail_offsets)
    with pytest.raises(OSError, match='publication'):
        prebuild(case, [0, 1])
    assert all(sig is None for sig in case.cache.signatures)
    assert not case.cache._slot_by_start
    monkeypatch.setattr(torch.Tensor, 'copy_', original)
    assert prebuild(case, [0, 1])['status'] == 'published'
    keys, ids, stats = snapshot(case)
    wanted, wanted_ids = exact(case)
    assert torch.equal(keys, wanted) and torch.equal(ids, wanted_ids)
    assert stats['cache_prebuilt_hit_blocks'] == 2
    case.cache.clear()


def test_query_holds_publication_mutex_and_records_manager_wait_context():
    case = fixture(2)
    prebuild(case, [0, 1])
    calls = []
    @contextmanager
    def query():
        calls.append('enter')
        yield
        calls.append('exit')
    case.authority.fps_prebuild = NS(query=query, snapshot=lambda:dict(query_wait_seconds=.01))
    q = torch.randn(2, 2, 6, 8, generator=torch.Generator().manual_seed(91))
    score, stats = module.score_authority_fps(case.authority, case.records, q,
        torch.zeros(2, 2, 2, 8), [True, True])
    keys, ids = exact(case)
    expected = module.key_fps_scores(q, keys, ids, torch.tensor([True, True]))
    assert torch.equal(score, expected)
    assert calls == ['enter', 'exit'] and stats['prebuild']['query_wait_seconds'] == .01
    assert stats['cache_prebuilt_hit_blocks'] == 2
    case.cache.clear()


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('bound_owner', [False, True])
def test_cpu_configuration_keeps_query_misses_on_cpu_while_qk_stays_on_query_device(
        monkeypatch, optimized, bound_owner):
    case = fixture(2)
    prebuild(case, [0])
    case.authority.graph_options = NS(query_fps_optimized=optimized,
        query_fps_prebuild=True, query_fps_prebuild_device='cpu')
    if bound_owner:
        case.authority.fps_prebuild = NS(manager=NS(device=torch.device('cpu')),
            query=lambda:module.nullcontext(), snapshot=lambda:dict(device='cpu'))
    def read_keys(handles, *, pin_memory=False):
        assert not pin_memory, 'CPU representative construction must not allocate CUDA staging'
        selected = case.raw.view(torch.uint8).index_select(1,
            torch.tensor([h.slot for h in handles])).view(case.raw.dtype)
        return selected.permute(1, 0, 2, 3, 4), {}
    case.authority.read_score_keys = read_keys
    old_snapshot = case.cache.snapshot
    def snapshot_on_cpu(*args, **kwargs):
        assert torch.device(kwargs['device']).type == 'cpu'
        return old_snapshot(*args, **kwargs)
    monkeypatch.setattr(case.cache, 'snapshot', snapshot_on_cpu)
    # A device-only Query descriptor avoids any GPU use. The scorer stub
    # asserts that dispatch still sees the original GPU Query unchanged.
    queries = NS(device=torch.device('cuda:0'))
    def score_on_query_device(q, representatives, indices, mask, **kwargs):
        assert q is queries and q.device.type == 'cuda'
        expected, expected_ids = exact(case)
        assert torch.equal(representatives, expected) and torch.equal(indices, expected_ids)
        return torch.tensor([.25, .75])
    monkeypatch.setattr(module, 'key_fps_scores', score_on_query_device)
    scores, stats = module.score_authority_fps(case.authority, case.records, queries,
        torch.zeros(2, 2, 2, 8), [True, True])
    assert scores.tolist() == [.25, .75]
    assert stats['construction_device'] == 'cpu' and stats['scoring_device'] == 'cuda:0'
    assert stats['cache_prebuilt_hit_blocks'] == 1 and stats['cache_rebuilt_blocks'] == 1
    assert stats['fps_cpu_groups'] == stats['fps_cpu_reference_groups'] == 4
    assert stats['fps_gpu_groups'] == stats['fps_cpu_ambiguity_groups'] == 0
    assert stats['fps_cpu_unsupported_fallback_groups'] == 0
    case.cache.clear()


def test_configured_cpu_rejects_misbound_gpu_worker_without_doing_work():
    case = fixture(2)
    case.authority.graph_options = NS(query_fps_optimized=True,
        query_fps_prebuild=True, query_fps_prebuild_device='cpu')
    case.authority.fps_prebuild = NS(manager=NS(device=torch.device('cuda:0')))
    with pytest.raises(ValueError, match='non-CPU prebuild worker'):
        module.score_authority_fps(case.authority, case.records, NS(device=torch.device('cuda:0')),
            torch.zeros(2, 2, 2, 8), [True, True])
    assert not case.reads and case.cache.bytes == 0


def test_cpu_reference_stat_is_distinct_from_gpu_ambiguity_fallback():
    case = fixture(2)
    result = prebuild(case, [0, 1])
    assert result['construction_device'] == 'cpu'
    assert result['fps_cpu_reference_groups'] == result['fps_cpu_groups'] == 8
    assert result['fps_cpu_fallback_groups'] == 8  # Preserved legacy aggregate.
    assert result['fps_cpu_ambiguity_groups'] == result['fps_gpu_groups'] == 0
    assert result['fps_cpu_unsupported_fallback_groups'] == 0
    assert result['fps_fallback_reason'] == 'CPU reference construction requested'
    case.cache.clear()
