"""Native CPU construction must publish the original reference Key selection."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.query_key_fps import (
    select_key_representatives, select_key_representatives_optimized, score_authority_fps)
from test_query_fps_incremental_cache import fixture
from test_query_fps_prebuild import harness, append, drain_writes


@contextmanager
def pool(workers=2):
    value = _native.KeyFPSCPUPool(workers, 2)
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize('dim', [8, 256])
@pytest.mark.parametrize('cache_dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_native_wrapper_matches_partial_gqa_groups_and_ignores_nan_padding(dim, cache_dtype):
    counts = [0, 1, 2, 3, 4, 31, 32]
    raw = torch.randn(2, len(counts), 32, 3, dim, generator=torch.Generator().manual_seed(188))
    for b, count in enumerate(counts):
        raw[:, b, count:] = float('nan')
    keys = raw.to(torch.float8_e4m3fn)
    wanted, wanted_ids = select_key_representatives(keys, counts)
    threads = torch.get_num_threads()
    with pool() as native:
        values, ids, stats = select_key_representatives_optimized(keys, counts,
            device='cpu', cache_dtype=cache_dtype, cpu_pool=native)
    assert torch.get_num_threads() == threads
    assert torch.equal(ids, wanted_ids)
    assert torch.equal(values, wanted.to(cache_dtype))
    assert stats['fps_cpu_workers'] == 2 and stats['fps_cpu_backend'] == 'native'
    assert stats['fps_cpu_native_groups'] == stats['fps_cpu_groups'] == 42
    assert stats['fps_cpu_reference_groups'] == stats['fps_cpu_ambiguity_groups']
    assert stats['fps_cpu_exact_cache_conversion'] and not stats['fps_device_cache_conversion']


def test_ambiguous_groups_recompute_from_all_original_tokens_and_tie_order():
    keys = torch.zeros(1, 2, 32, 2, 8, dtype=torch.float8_e4m3fn)
    with pool() as native:
        values, ids, stats = select_key_representatives_optimized(keys, [32, 2],
            device='cpu', cache_dtype=torch.float16, cpu_pool=native)
    expected, expected_ids = select_key_representatives(keys, [32, 2])
    assert torch.equal(ids, expected_ids) and torch.equal(values, expected.half())
    assert stats['fps_cpu_ambiguity_groups'] == stats['fps_cpu_reference_groups'] == 4
    assert ids[0, 0, :, 0].tolist() == [0, 1, 2, 3]
    assert ids[0, 1, :, 0].tolist() == [0, 1, -1, -1]


def test_busy_falls_back_but_real_worker_error_remains_visible():
    keys = torch.randn(1, 2, 32, 2, 8).to(torch.float8_e4m3fn)
    def busy(*args):
        raise _native.KeyFPSCPUBusyError('busy')
    fake = NS(stats=lambda:{'workers':2}, select=busy)
    values, ids, stats = select_key_representatives_optimized(keys, [32, 17], device='cpu', cpu_pool=fake)
    expected, expected_ids = select_key_representatives(keys, [32, 17])
    assert torch.equal(ids, expected_ids) and torch.equal(values.float(), expected.float())
    assert stats['fps_cpu_pool_busy_groups'] == stats['fps_cpu_reference_groups'] == 4
    def broken(*args):
        raise ValueError('broken native input')
    fake.select = broken
    with pytest.raises(ValueError, match='broken native input'):
        select_key_representatives_optimized(keys, [32, 17], device='cpu', cpu_pool=fake)


def test_native_pool_rejects_gpu_route_before_any_device_work():
    keys = torch.randn(1, 1, 32, 1, 8).to(torch.float8_e4m3fn)
    fake = NS(stats=lambda:{'workers':2})
    with pytest.raises(ValueError, match='bound to GPU'):
        select_key_representatives_optimized(keys, [32], device='cuda:0', cpu_pool=fake)


def test_native_sparse_prebuild_hits_survive_query_layout_and_read_only_missing_blocks():
    case = fixture(4)
    def read(handles, *, pin_memory=False):
        assert not pin_memory
        rows = [h.slot for h in handles]
        case.reads.append(tuple(rows))
        return case.raw.view(torch.uint8).index_select(1, torch.tensor(rows)).view(case.raw.dtype).permute(1,0,2,3,4), {}
    case.authority.read_score_keys = read
    with pool() as native:
        raw = case.raw[:, 2:]
        built = case.cache.prebuild(case.records[2:], raw, epoch=case.cache.epoch,
            optimized=True, device='cpu', cpu_pool=native)
        assert built['cache_rebuilt_blocks'] == 2
        with case.cache.use():
            values, ids, stats = case.cache.snapshot(case.authority, case.records,
                optimized=True, device='cpu', cpu_pool=native)
            wanted, wanted_ids = select_key_representatives(case.raw, [32]*4)
            assert torch.equal(values, wanted.half()) and torch.equal(ids, wanted_ids.to(torch.int8))
        assert stats['cache_prebuilt_hit_blocks'] == 2 and stats['cache_rebuilt_blocks'] == 2
        assert case.reads == [(0,1)]
        assert stats['fps_cpu_native_groups'] == 8
    case.cache.clear()


def test_live_owner_background_and_query_share_native_pool_without_changing_torch_threads(tmp_path):
    threads = torch.get_num_threads()
    with harness(tmp_path, cpu_backend='native', cpu_workers=2) as (manager, archive, create, _):
        owner = create()
        append(owner, 35, candidates=29)
        drain_writes(owner)
        assert manager.wait_idle(10)
        owner.graph_options = NS(query_fps_optimized=True, query_fps_prebuild=True,
            query_fps_prebuild_device='cpu', query_fps_cpu_backend='native')
        owner.mean_k_backend = 'torch_reference'
        with owner.index_snapshot() as (records, _):
            queries = torch.randn(1, 2, 2, 4, generator=torch.Generator().manual_seed(135))
            scores, stats = score_authority_fps(owner, records, queries,
                torch.zeros(1, len(records), 1, 4), [True]*len(records))
        assert torch.isfinite(scores).all()
        assert stats['cache_prebuilt_hit_blocks'] == 1 and stats['cache_rebuilt_blocks'] == 1
        assert stats['fps_cpu_native_groups'] == 1 and stats['fps_cpu_workers'] == 2
        assert stats['prebuild']['fps_cpu_native_groups'] == 1
        assert stats['prebuild']['cpu_backend'] == 'native' and stats['prebuild']['cpu_workers'] == 2
        assert manager.cpu_pool.stats()['groups'] == 2
        assert torch.get_num_threads() == threads
