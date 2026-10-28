"""CPU integration: native LRU + compact FP8 + native FPS + graph-free quota.

Only the CUDA MeanK scorer is replaced by its CPU reference. Archive ownership,
native FPS pool, cache publication, score masks, quota/halo picker, generation
leases and GC are real. These tests do not qualify GPU/model execution.
"""
from contextlib import contextmanager
from types import SimpleNamespace as NS
import time

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority, _Ready, query_score_frame
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.query_boundary_history import QueryBoundaryHistory
from vllm_kvmem.query_fps_prebuild import FPSPrebuildManager
from vllm_kvmem.query_hybrid_recall import preview_hybrid_selection
from vllm_kvmem.query_key_fps import key_fps_scores, select_key_representatives
from vllm_kvmem.retrieval import mean_key_scores


def eventually(callback):
    deadline = time.monotonic()+10
    while not callback():
        assert time.monotonic() < deadline, 'native ownership did not settle'
        time.sleep(.001)


def advance(authority, worker, count=32):
    for _ in range(count):
        worker.advance()
        authority.poll()


def make_authority(archive, manager, name, *, owners=1):
    config = KVMemConfig(block_tokens=32, select_budget=224, gen_budget=160,
        repository_blocks=16, sink_blocks=1, recent_blocks=0,
        action_attention=True, action_fraction=0.)
    # Room for sixteen compact rows; long history must recycle those rows.
    row_bytes = 4*1*8+2048+4
    authority = RequestAuthority(name, config, archive, 2, 1, 8,
        raw_dtype=torch.float8_e4m3fn, scoring_layers=1, resident_records=16,
        mean_k_backend='qw3_cuda', cpu_release_sealed_sums=True,
        query_score_mode='key_fps_4', query_fps_cache_fp8=True,
        query_window_cache_bytes=owners*16*row_bytes)
    authority.query_boundaries = QueryBoundaryHistory(name, max_spans=32)
    authority.graph_options = NS(graph_mode='off', graph_alpha=0.,
        query_fps_halo_blocks=1, query_mean_quota_fraction=.5, graph_trace=False,
        query_fps_optimized=True, query_fps_prebuild=True,
        query_fps_prebuild_device='cpu', query_fps_cpu_backend='native')
    authority.fps_prebuild = manager.bind(authority)
    return authority


@contextmanager
def harness(tmp_path, monkeypatch, *, owners=1, background=False):
    archive = MemoryArchive(str(tmp_path), 2*32*1*8*2, owners*32, native_payloads=True)
    gc = n.HistoricalGcConfig()
    gc.mode, gc.background = n.HistoricalGcMode.DELETE, background
    gc.max_scopes, gc.scope_capacity = owners, 16
    gc.ready_low, gc.ready_high = 2, 4
    gc.scan_blocks, gc.delete_blocks, gc.poll_ms = 16, 4, 1
    worker = archive.enable_history_gc(gc)
    manager = FPSPrebuildManager(device='cpu', cpu_backend='native', cpu_workers=2*owners)
    authorities = [make_authority(archive, manager, f'hybrid-lru-{i}', owners=owners)
                   for i in range(owners)]
    # Preserve the native score representation's FP16 Query input conversion.
    monkeypatch.setattr('vllm_kvmem.cuda_scoring.mean_key_scores',
        lambda q, means, mask: mean_key_scores(q.half(), means, mask))
    expected = [{} for _ in range(owners)]
    try:
        # Match the model worker: the background builder publishes inference
        # tensors, and foreground canonical row swaps run in inference mode.
        with torch.inference_mode():
            yield (authorities[0] if owners == 1 else authorities), archive, worker, manager, (
                expected[0] if owners == 1 else expected)
    finally:
        manager.close()
        for authority in authorities:
            eventually(authority.close)
            assert authority.memory.repository.ready_slots == 16
        usage = archive.io_usage()
        assert usage['cpu_authority_bytes'] == usage['cpu_payload_readers'] == 0
        assert usage['cpu_index_bytes'] == usage['cpu_index_owners'] == 0
        assert archive._query_window_cache_bytes == 0
        eventually(archive.close)


def append(authority, expected, count, local):
    begin = authority.history_end
    generator = torch.Generator().manual_seed(739+begin)
    # Include rejected draft rows; no archive/index/GC history may include them.
    raw = torch.randn(2, count+4, 1, 8, generator=generator).to(torch.float8_e4m3fn)
    values = torch.randint(0, 256, raw.shape, generator=generator, dtype=torch.uint8)
    assert authority.append(raw, values, history_begin=begin, local_rope_begin=local,
                            accepted=count)
    for offset in range(0, count, 32):
        expected[begin+offset] = (raw[:, offset:offset+32].clone(), values[:, offset:offset+32].clone())
    for job, _ in tuple(authority.writes):
        job.result(5)
    authority.poll()
    authority.query_boundaries.update_message_spans(
        [(begin, begin+count, 'tool' if begin else 'user')], validated_prefix=begin)
    assert authority.query_boundaries.accepted_through == begin+count


def exact_live_selection(authority, manager, expected, query):
    """Independent original-FP8 oracle and exact native picker comparison."""
    assert manager.wait_idle(10)
    with authority.fps_prebuild.query(), authority.index_snapshot() as (records, means):
        starts = [record.history_begin for record in records]
        payloads = authority.read_records([record.handle for record in records])
        for record, (actual_k, actual_v) in zip(records, payloads):
            original_k, original_v = expected[record.history_begin]
            assert torch.equal(actual_k.view(torch.uint8), original_k.view(torch.uint8))
            assert torch.equal(actual_v, original_v)
            assert record.key_sum is None and record.mean_key.dtype == torch.float16
        raw = torch.stack([expected[start][0][:1] for start in starts], dim=1)
        expected_mean = raw.float().mean(dim=2).half()
        assert torch.equal(means, expected_mean)
        representatives, offsets = select_key_representatives(raw, [32]*len(records))
        with authority.query_window_cache.use():
            cached, cached_offsets, cache_stats = authority.query_window_cache.snapshot(
                authority, records, optimized=True, device='cpu', cpu_pool=manager.cpu_pool)
            assert cached.dtype == torch.float8_e4m3fn
            assert torch.equal(cached.view(torch.uint8), representatives.view(torch.uint8))
            assert torch.equal(cached_offsets, offsets.to(torch.int8))
            assert cache_stats['cache_rebuilt_blocks'] == 0
            del cached, cached_offsets
        with query_score_frame(authority.memory, records) as frame:
            mask = torch.tensor(frame.score_mask, dtype=torch.bool)
            fps = key_fps_scores(query, representatives, offsets, mask, query_half=True).tolist()
            mean = mean_key_scores(query.half(), expected_mean, mask).tolist()
        preview = preview_hybrid_selection(authority.memory, records, fps, mean,
            mean_fraction=.5, message_spans=authority.query_boundaries.message_spans,
            message_metadata_present=authority.query_boundaries.has_metadata)
        wanted = set(preview['selected_slots'])
        plan = authority.select_for_queries(query)
        assert {row.block_id for row in plan.remaps} == wanted
        assert plan.total_window_tokens <= 224
        assert authority.query_quota_propagation is None
        assert authority.query_selection_report['propagation_applied'] is False
        assert authority.query_selection_report['selection_backend'] == 'native_query_no_graph'
        assert authority.query_quota_selection['mean_fraction'] == .5
        assert not hasattr(authority, 'block_graph')
        local = plan.total_window_tokens
        authority.memory.attach_completion(_Ready())
        assert authority.memory.commit_selection()
        del plan, payloads, records, means
    return local


def test_compact_hybrid_recycles_many_repository_generations_without_changing_live_bytes(tmp_path, monkeypatch):
    with harness(tmp_path, monkeypatch) as (authority, archive, worker, manager, expected):
        append(authority, expected, 384, 0)
        query = torch.randn(1, 3, 2, 8, generator=torch.Generator().manual_seed(59))
        peaks = []
        stale = {}
        for _ in range(24):
            local = exact_live_selection(authority, manager, expected, query)
            authority.gc.scope.request_capacity(8)
            advance(authority, worker)
            assert len(authority.records) == 8
            assert archive.tier.used_slots == 8
            for key in authority.records:
                if key[0] in stale and stale[key[0]] != key[1]:
                    assert not authority.memory.repository.pin(n.Handle(key[0], stale[key[0]]))
                stale[key[0]] = key[1]
            assert len(authority.query_window_cache._slot_by_start) == 8
            assert len(authority.query_boundaries.message_spans) <= 8
            usage = archive.io_usage()
            assert usage['cpu_authority_bytes'] == 8*authority.record_bytes
            assert usage['cpu_index_bytes'] == 8*1*1*8*2  # Only sealed FP16 MeanK remains.
            peaks.append((usage['cpu_authority_bytes'], usage['cpu_index_bytes'],
                          archive._query_window_cache_bytes))
            authority.gc.scope.request_capacity(0)
            append(authority, expected, 128, local)
        exact_live_selection(authority, manager, expected, query)
        assert authority.history_end == 3456 > 6*16*32
        assert authority.gc_reclaimed_records == 96
        assert len(set(peaks)) == 1  # Real allocator/index accounting stays flat.
        assert authority.query_window_cache.capacity == 16
        assert manager.cpu_pool.stats()['groups'] > 100
        assert authority.fps_prebuild.snapshot().get('errors', 0) == 0
        starts = sorted(record.history_begin for record in authority.records.values())
        assert starts[0] == 0 and any(b-a > 32 for a, b in zip(starts, starts[1:]))


def test_gc_cannot_retire_query_inputs_during_scoring_and_native_quota_pick(tmp_path, monkeypatch):
    with harness(tmp_path, monkeypatch) as (authority, archive, worker, manager, expected):
        append(authority, expected, 384, 0)
        assert manager.wait_idle(10)
        authority.gc.scope.request_capacity(8)
        before = authority.gc.scope.stats['retired']
        original_mean = mean_key_scores
        observations = []
        def score(q, means, mask):
            # FPS has already consumed the same pinned index. Force background
            # retirement between the FPS branch and MeanK quota construction.
            advance(authority, worker)
            assert authority.gc.scope.stats['retired'] == before
            observations.append(True)
            return original_mean(q.half(), means, mask)
        monkeypatch.setattr('vllm_kvmem.cuda_scoring.mean_key_scores', score)
        plan = authority.select_for_queries(torch.ones(1, 2, 2, 8))
        assert observations and plan.total_window_tokens <= 224
        authority.memory.attach_completion(_Ready())
        assert authority.memory.commit_selection()
        del plan
        advance(authority, worker)
        assert authority.gc.scope.stats['retired'] == before+4
        assert len(authority.records) == archive.tier.used_slots == 8
        assert authority.query_window_cache.epoch > 0


def test_query_exception_releases_all_policy_holds_for_later_gc(tmp_path, monkeypatch):
    with harness(tmp_path, monkeypatch) as (authority, _, worker, manager, expected):
        append(authority, expected, 384, 0)
        assert manager.wait_idle(10)
        authority.gc.scope.request_capacity(8)
        def fail(*_):
            advance(authority, worker)
            assert authority.gc.scope.stats['retired'] == 0
            raise OSError('injected second-branch scorer failure')
        monkeypatch.setattr('vllm_kvmem.cuda_scoring.mean_key_scores', fail)
        with pytest.raises(OSError, match='second-branch'):
            authority.select_for_queries(torch.ones(1, 2, 2, 8))
        advance(authority, worker)
        assert authority.gc.scope.stats['retired'] == 4
        assert len(authority.records) == 8


def test_delayed_old_retirement_during_fps_miss_does_not_invalidate_query(tmp_path, monkeypatch):
    with harness(tmp_path, monkeypatch) as (authority, _, worker, manager, expected):
        append(authority, expected, 384, 0)
        assert manager.wait_idle(10)
        old_handle = next(r.handle for r in authority.records.values() if r.history_begin == 32)
        # This reader was admitted before retirement. Its release can produce
        # an old completion while a later Query is already scoring live rows.
        old_read = [authority.gc.scope.admit_read(old_handle)]
        assert old_read[0] is not None
        authority.gc.scope.request_capacity(8)
        advance(authority, worker)
        assert authority.gc.scope.stats['retired'] == 4
        assert authority.gc.scope.stats['cpu_released'] == 3
        assert not authority.memory.repository.pin(old_handle)
        cache = authority.query_window_cache
        assert 32 in cache._slot_by_start
        epoch = cache.epoch
        # Simulate one unbuilt derived row, leaving the old pending generation
        # and every other live cache row in place. The real miss constructs FPS.
        cache.signatures[cache._slot_by_start[352]] = None
        original_read = authority.read_score_keys
        observed = []
        def read(handles, **kwargs):
            assert authority._index_snapshot_depth > 0
            old_read.clear()
            for _ in range(32):
                worker.advance()
            result = original_read(handles, **kwargs)  # Actual poll receives the completion.
            assert authority.gc.scope.stats['cpu_released'] == 4
            assert authority._pending_cache_retire
            assert cache.epoch == epoch  # Active FPS views must remain publishable.
            observed.append(True)
            return result
        monkeypatch.setattr(authority, 'read_score_keys', read)
        plan = authority.select_for_queries(torch.ones(1, 2, 2, 8))
        assert observed and authority.query_key_fps_stats['cache_rebuilt_blocks'] == 1
        assert cache.epoch == epoch+1 and not authority._pending_cache_retire
        assert 32 not in cache._slot_by_start
        assert len(authority.records) == 8
        authority.memory.attach_completion(_Ready())
        assert authority.memory.commit_selection()
        del plan


def test_eight_owners_share_bounded_archive_and_native_fps_pool_over_long_histories(tmp_path, monkeypatch):
    with harness(tmp_path, monkeypatch, owners=8) as (authorities, archive, worker, manager, expected):
        query = torch.randn(1, 2, 2, 8, generator=torch.Generator().manual_seed(619))
        for authority, originals in zip(authorities, expected):
            append(authority, originals, 384, 0)
        peaks = []
        for _ in range(24):
            locals_ = [exact_live_selection(authority, manager, originals, query)
                       for authority, originals in zip(authorities, expected)]
            for authority in authorities:
                authority.gc.scope.request_capacity(8)
            for _ in range(32):
                worker.advance()
                for authority in authorities:
                    authority.poll()
            assert all(len(authority.records) == 8 for authority in authorities)
            assert archive.tier.used_slots == 8*8
            usage = archive.io_usage()
            assert usage['cpu_authority_bytes'] == 8*8*authorities[0].record_bytes
            assert usage['cpu_index_bytes'] == 8*8*1*1*8*2
            assert archive._query_window_cache_bytes == sum(
                authority.query_window_cache.bytes for authority in authorities)
            peaks.append((usage['cpu_authority_bytes'], usage['cpu_index_bytes'],
                          archive._query_window_cache_bytes))
            for authority, originals, local in zip(authorities, expected, locals_):
                authority.gc.scope.request_capacity(0)
                append(authority, originals, 128, local)
        for authority, originals in zip(authorities, expected):
            exact_live_selection(authority, manager, originals, query)
            assert authority.history_end == 3456 > 6*16*32
            assert authority.gc_reclaimed_records == 96
            assert authority.query_window_cache.capacity == 16
            assert authority.fps_prebuild.snapshot().get('errors', 0) == 0
            assert len(authority.query_boundaries.message_spans) <= 12
        assert len(set(peaks)) == 1
        assert manager.cpu_pool.stats()['workers'] == 16
        assert manager.cpu_pool.stats()['groups'] > 8*100


def test_real_background_gc_protects_queries_and_recycles_compact_fps_rows(tmp_path, monkeypatch):
    with harness(tmp_path, monkeypatch, background=True) as (authority, archive, _, manager, expected):
        append(authority, expected, 384, 0)
        query = torch.randn(1, 2, 2, 8, generator=torch.Generator().manual_seed(967))
        scope = authority.gc.scope
        observations = []
        def score(q, means, mask):
            assert authority._index_snapshot_depth > 0
            before = scope.stats
            scope.request_capacity(8)
            # No manual worker.advance(): require actual background ticks
            # while every FPS/MeanK input is protected by the Query snapshot.
            eventually(lambda: scope.stats['ticks'] >= before['ticks']+8)
            assert scope.stats['retired'] == before['retired']
            observations.append(scope.stats['ticks'])
            return mean_key_scores(q.half(), means, mask)
        monkeypatch.setattr('vllm_kvmem.cuda_scoring.mean_key_scores', score)
        slab = None
        for _ in range(12):
            # The original-FP8 oracle also verifies real native FPS, means,
            # cache hits and the exact 50:50 native picker before committing.
            local = exact_live_selection(authority, manager, expected, query)
            current = authority.query_window_cache.tensor.untyped_storage().data_ptr()
            slab = current if slab is None else slab
            assert current == slab
            def collected():
                authority.poll()  # Drains receipts; does not choose/advance policy.
                stats = scope.stats
                return (len(authority.records) <= 8 and
                        stats['cpu_released'] == stats['retired'] and
                        authority.gc_reclaimed_records == stats['retired'])
            eventually(collected)
            scope.request_capacity(0)
            live = {r.history_begin for r in authority.records.values()}
            assert 0 in live and len(live) < 12
            assert set(authority.query_window_cache._slot_by_start) == live
            assert archive.io_usage()['cpu_authority_bytes'] <= 8*authority.record_bytes
            assert authority.query_window_cache.capacity == 16
            assert archive._query_window_cache_bytes == authority.query_window_cache.bytes
            # Wait for background descriptor/index release before a new
            # accepted packet, without relying on one exact retirement count.
            eventually(lambda: (authority.poll(), authority.memory.repository.ready_slots >= 4)[1])
            append(authority, expected, 128, local)
        assert len(observations) == 12
        assert authority.history_end > 3*16*32
        assert authority.gc_reclaimed_records >= 12
        assert manager.wait_idle(10)
        assert authority.fps_prebuild.snapshot().get('errors', 0) == 0
