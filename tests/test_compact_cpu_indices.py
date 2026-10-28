"""Compact storage must preserve original Keys, page scores and rollback."""
from dataclasses import replace
from types import SimpleNamespace as NS
from threading import RLock

import pytest
import torch

from vllm_kvmem.authority import RequestAuthority, _Ready, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.options import RuntimeOptions
import vllm_kvmem.query_key_fps as fps
from test_query_fps_incremental_cache import fixture, prebuild, snapshot, exact
from test_graph_options import runtime, model_profile, worker_payload
from test_query_fps_prebuild_options import fps_settings
from vllm_kvmem.serve import build_launch


def test_every_finite_original_fp8_code_roundtrips_through_old_cache_exactly():
    codes = torch.arange(256, dtype=torch.uint8)
    codes = codes[(codes != 127) & (codes != 255)]
    assert torch.equal(codes.view(torch.float8_e4m3fn).half().to(torch.float8_e4m3fn).view(torch.uint8), codes)


def compact_case(blocks=8):
    case = fixture(blocks)
    case.cache = fps.KeyFPSCache(case.archive, 2, 2, 8, dtype=torch.float16,
                                 max_bytes=1 << 20, compact_fp8=True)
    case.authority.query_window_cache = case.cache
    return case


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('tiled', [False, True])
def test_sparse_prebuild_lazy_tail_and_scores_remain_exact(monkeypatch, optimized, tiled):
    case = compact_case()
    try:
        for group in ([6, 7], [2, 4], [0]):
            assert prebuild(case, group)['status'] == 'published'
        case.records[7].count = 3
        case.cache.rewind(7*32+3)
        selected = [0, 1, 2, 4, 6, 7]
        keys, ids, stats = snapshot(case, selected)
        old_keys, old_ids = exact(case, selected)
        assert keys.dtype == torch.float8_e4m3fn
        assert torch.equal(keys.half(), old_keys) and torch.equal(ids, old_ids)
        assert case.cache.bytes == case.cache.capacity*(2*4*2*8+2048+2*4*2)
        q = torch.randn(2, 11, 4, 8, generator=torch.Generator().manual_seed(37))
        mask = torch.tensor([True, False, True, True, False, True])
        if tiled:
            monkeypatch.setattr(fps, 'SCRATCH_LIMIT', 10000)
        expected = fps.key_fps_scores(q, old_keys, old_ids, mask, optimized=optimized)
        result = fps.key_fps_scores(q, keys, ids, mask, optimized=optimized)
        assert torch.equal(result, expected)
    finally:
        case.cache.clear()


def test_compact_cache_clear_retains_live_snapshot_and_charges_until_release():
    case = compact_case()
    with case.cache.use():
        prebuild(case, list(range(8)))
        values, ids, _ = snapshot(case)
        old = values.half().clone()
        charged = case.archive._query_window_cache_bytes
        case.cache.clear()
        assert case.archive._query_window_cache_bytes == charged
        assert torch.equal(values.half(), old)
    assert case.archive._query_window_cache_bytes == 0


def test_compact_cache_refuses_non_fp8_original_keys():
    case = compact_case()
    with pytest.raises(ValueError, match='original FP8'):
        case.cache.prebuild(case.records[:1], case.raw[:, :1].half(), epoch=case.cache.epoch)
    case.cache.clear()


def test_native_cpu_prebuild_and_lazy_repair_keep_exact_original_offsets():
    from test_query_fps_cpu_integration import pool
    case = compact_case()
    def read(handles, *, pin_memory=False):
        rows = torch.tensor([h.slot for h in handles])
        return case.raw.view(torch.uint8).index_select(1, rows).view(case.raw.dtype).permute(1,0,2,3,4), {}
    case.authority.read_score_keys = read
    try:
        with pool() as native:
            built = case.cache.prebuild(case.records[2:4], case.raw[:,2:4],
                epoch=case.cache.epoch, cpu_pool=native)
            assert built['status'] == 'published' and built['fps_cpu_native_groups'] > 0
            with case.cache.use():
                keys, ids, stats = case.cache.snapshot(case.authority, case.records,
                    optimized=True, device='cpu', cpu_pool=native)
                wanted, offsets = exact(case)
                assert torch.equal(keys.half(), wanted) and torch.equal(ids, offsets)
                assert stats['cache_prebuilt_hit_blocks'] == 2
                assert stats['fps_cpu_native_groups'] > 0
    finally:
        case.cache.clear()


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('async_seal', [False, True])
def test_released_sums_preserve_means_bytes_partial_tail_and_rewind(tmp_path, batched, async_seal):
    cfg = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=128,
                     repository_blocks=32, sink_blocks=1, recent_blocks=0)
    g = torch.Generator().manual_seed(73)
    raw = (torch.randn(2, 101, 2, 16, generator=g)*3).to(torch.float8_e4m3fn)
    values = torch.randint(0, 256, raw.shape, dtype=torch.uint8, generator=g)
    outputs = []
    for compact in (False, True):
        archive = MemoryArchive(str(tmp_path/str(compact)), 2*32*2*16*2, 64)
        archive.async_seal = async_seal
        a = RequestAuthority('test', cfg, archive, 2, 2, 16, raw_dtype=raw.dtype,
            scoring_layers=1, mean_k_backend='qw3_cuda', cpu_release_sealed_sums=compact)
        saved = None
        try:
            def append(begin, count):
                kwargs = dict(history_begin=begin, local_rope_begin=begin)
                args = (raw[:, begin:begin+count], values[:, begin:begin+count])
                if batched: assert append_authority_batch([(a, *args, kwargs)]) == 1
                else: assert a.append(*args, **kwargs)
                a.publish_sealed(lambda: None)
            append(0, 64)
            saved = a.memory.checkpoint()
            append(64, 37)
            for record in a.records.values():
                assert (record.key_sum is None) == (compact and record.count == 32)
            before = a.mean_keys()[1].clone()
            assert a.begin_rewind(saved) is not None
            a.attach_rewind_completion(_Ready())
            assert a.poll_rewind()
            assert a.history_end == 64
            append(64, 11)
            records = sorted(a.records.values(), key=lambda r:r.history_begin)
            stored = [a.read_record(r.handle) for r in records]
            outputs.append((before, a.mean_keys()[1].clone(),
                            [(k.view(torch.uint8).clone(), v.clone()) for k,v in stored]))
        finally:
            saved = None
            assert a.close() and archive.close()
    for i in (0, 1): assert torch.equal(outputs[0][i], outputs[1][i])
    for old, new in zip(outputs[0][2], outputs[1][2]):
        assert all(torch.equal(x,y) for x,y in zip(old,new))


@pytest.mark.parametrize('field', ['cpu_release_sealed_sums', 'query_fps_cache_fp8'])
def test_options_are_explicit_cpu_only_and_preserved_by_launcher(tmp_path, field):
    assert getattr(RuntimeOptions(), field) is False
    payload = fps_settings(archive_backend='cpu', mean_k_backend='qw3_cuda', **{field:True})
    opt = runtime(payload)
    assert getattr(opt, field) is True
    with pytest.raises(ValueError, match='Compact CPU indices'):
        replace(opt, archive_backend='nvme')
    with pytest.raises(ValueError, match='boolean'):
        runtime(dict(payload, **{field:1}))
    profile = model_profile(tmp_path)
    profile['kvmem'] = payload
    assert getattr(runtime(worker_payload(build_launch(profile, tmp_path, tmp_path/'out'))), field) is True


@pytest.mark.skipif(__import__('os').environ.get('KVMEM_TEST_CUDA') != '1', reason='explicit GPU qualification')
@pytest.mark.parametrize('tiled', [False, True])
def test_cuda_scores_bitwise_equal_after_compact_storage(monkeypatch, tiled):
    case = compact_case(48)
    try:
        keys, ids, _ = snapshot(case)
        q = torch.randn(2, 19, 4, 8, generator=torch.Generator().manual_seed(81)).cuda()
        mask = torch.arange(48)%3 != 0
        if tiled: monkeypatch.setattr(fps, 'SCRATCH_LIMIT', 80000)
        new = fps.key_fps_scores(q, keys, ids, mask, optimized=True)
        old = fps.key_fps_scores(q, keys.half(), ids, mask, optimized=True)
        torch.cuda.synchronize()
        assert torch.equal(new, old)
    finally:
        case.cache.clear()


@pytest.mark.skipif(__import__('os').environ.get('KVMEM_TEST_CUDA') != '1', reason='explicit GPU qualification')
@pytest.mark.parametrize('tiled', [False, True])
def test_cuda_real_head_geometry_preserves_score_and_picker_order(monkeypatch, tiled):
    generator = torch.Generator().manual_seed(89)
    keys = torch.randn(16, 48, 4, 4, 256, generator=generator).to(torch.float8_e4m3fn)
    ids = torch.arange(4, dtype=torch.int8).view(1, 1, 4, 1).expand(16, 48, 4, 4)
    queries = torch.randn(16, 19, 32, 256, generator=generator).cuda()
    mask = torch.arange(48)%5 != 0
    if tiled:
        monkeypatch.setattr(fps, 'SCRATCH_LIMIT', 800000)
    before = fps.key_fps_scores(queries, keys.half(), ids, mask, optimized=True)
    after = fps.key_fps_scores(queries, keys, ids, mask, optimized=True)
    torch.cuda.synchronize()
    assert torch.equal(before, after)
    assert torch.equal(before.argsort(descending=True), after.argsort(descending=True))
