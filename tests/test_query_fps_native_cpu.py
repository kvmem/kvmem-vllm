"""The private CPU FPS pool is bounded and exact after its ambiguity guard."""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import time
import weakref

import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.query_key_fps import select_key_representatives


def raw_fixture(dim=64, counts=(32, 31, 3, 1, 0)):
    keys = torch.randn(2, len(counts), 32, 3, dim,
        generator=torch.Generator().manual_seed(481)).to(torch.float8_e4m3fn)
    return keys, keys.view(torch.uint8).numpy(), list(counts)


def checked_select(pool, keys, raw, counts):
    expected_keys, expected_indices = select_key_representatives(keys, counts)
    indices, uncertain = pool.select(raw, counts)
    assert indices.dtype == np.int8 and uncertain.dtype == np.bool_
    actual = torch.from_numpy(indices).long()
    uncertain = torch.from_numpy(uncertain)
    assert uncertain.shape == (keys.shape[0], keys.shape[1], keys.shape[3])
    exact_groups = ~uncertain[:, :, None, :].expand_as(actual)
    assert torch.equal(actual[exact_groups], expected_indices[exact_groups])
    # The binding's contract leaves whole ambiguous groups to the established
    # Torch reference, starting again from the original complete token group.
    actual = torch.where(uncertain[:, :, None, :], expected_indices, actual)
    assert torch.equal(actual, expected_indices)
    selected = keys.float().gather(2, actual.clamp_min(0)[..., None].expand(
        *actual.shape, keys.shape[-1]))
    selected = torch.where(actual[..., None] >= 0, selected, 0).to(keys.dtype)
    assert torch.equal(selected.view(torch.uint8), expected_keys.view(torch.uint8))
    return indices, uncertain.numpy()


@pytest.mark.parametrize('threads', [1, 4, 8, 16, 32])
@pytest.mark.parametrize('dim', [1, 17, 256])
def test_raw_fp8_indices_and_keys_match_reference_with_partial_and_empty_groups(threads, dim):
    keys, raw, counts = raw_fixture(dim)
    original = raw.copy()
    pool = _native.KeyFPSCPUPool(threads=threads)
    try:
        before_threads = torch.get_num_threads()
        first, mask = checked_select(pool, keys, raw, counts)
        second, again = checked_select(pool, keys, raw, counts)
        assert np.array_equal(first, second) and np.array_equal(mask, again)
        assert np.array_equal(raw, original) and torch.get_num_threads() == before_threads
        stats = pool.stats()
        assert stats['workers'] == threads and stats['active'] == 0
        assert stats['peak'] == 1 and stats['completed_jobs'] == 2
        assert stats['groups'] == 2*keys.shape[0]*keys.shape[1]*keys.shape[3]
        assert stats['scratch_bytes'] == threads*32*4096*4
    finally:
        pool.close()


def test_dense_axis_permutations_and_readonly_input_preserve_original_coordinates():
    keys, _, counts = raw_fixture(31)
    raw = keys.view(torch.uint8).permute(1, 0, 2, 3, 4).contiguous().permute(1, 0, 2, 3, 4).numpy()
    assert not raw.flags.c_contiguous
    raw.setflags(write=False)
    pool = _native.KeyFPSCPUPool(threads=4)
    try:
        checked_select(pool, keys, raw, counts)
    finally:
        pool.close()


def test_maximum_dimension_is_supported_without_unbounded_group_scratch():
    keys = torch.randn(1, 1, 32, 1, 4096,
        generator=torch.Generator().manual_seed(948)).to(torch.float8_e4m3fn)
    pool = _native.KeyFPSCPUPool(threads=2)
    try:
        checked_select(pool, keys, keys.view(torch.uint8).numpy(), [32])
    finally:
        pool.close()


def test_ties_choose_lowest_original_offsets_and_raise_whole_group_guard():
    raw = np.zeros((1, 3, 32, 2, 16), dtype=np.uint8)
    pool = _native.KeyFPSCPUPool(threads=4)
    try:
        indices, ambiguous = pool.select(raw, [32, 3, 1])
        assert indices[0, 0, :, 0].tolist() == [0, 1, 2, 3]
        assert indices[0, 1, :, 1].tolist() == [0, 1, 2, -1]
        assert indices[0, 2, :, 0].tolist() == [0, -1, -1, -1]
        assert ambiguous.tolist() == [[[True, True], [True, True], [False, False]]]
    finally:
        pool.close()


@pytest.mark.parametrize('code', [127, 255])
def test_valid_nan_rejected_padding_nan_ignored_and_pool_recovers(code):
    raw = np.zeros((2, 2, 32, 2, 32), dtype=np.uint8)
    raw[:, 1, 31] = code
    pool = _native.KeyFPSCPUPool(threads=4)
    try:
        pool.select(raw, [32, 31])
        with pytest.raises(ValueError, match='Nonfinite original FP8'):
            pool.select(raw, [32, 32])
        assert pool.stats()['active'] == 0 and pool.stats()['failed_jobs'] == 1
        pool.select(raw, [0, 0])
        assert pool.stats()['completed_jobs'] == 2
    finally:
        pool.close()


def test_all_finite_fp8_codes_include_signed_zero_and_extreme_values():
    raw = np.arange(256, dtype=np.uint8).reshape(1, 1, 32, 1, 8)
    raw[raw == 127] = 0
    raw[raw == 255] = 128
    keys = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn)
    pool = _native.KeyFPSCPUPool(threads=1)
    try:
        checked_select(pool, keys, raw, [32])
    finally:
        pool.close()


@pytest.mark.parametrize('threads, submissions', [(0, 2), (33, 2), (4, 0), (4, 3)])
def test_pool_bounds_reject_unbounded_workers_or_submissions(threads, submissions):
    with pytest.raises(ValueError, match='workers'):
        _native.KeyFPSCPUPool(threads=threads, max_submissions=submissions)


def test_invalid_geometry_stride_and_counts_fail_before_submission():
    raw = np.zeros((1, 1, 32, 1, 16), dtype=np.uint8)
    bad = [
        (raw.astype(np.float32), [32]), (raw[0], [32]), (raw[..., ::-1], [32]),
        (raw[..., ::2], [32]), (np.broadcast_to(raw[..., :1], raw.shape), [32]),
        (np.zeros((1, 1, 33, 1, 1), np.uint8), [32]),
        (np.zeros((1, 33, 1, 1, 1), np.uint8), [1]*33),
        (np.zeros((1, 1, 1, 1, 4097), np.uint8), [1]),
        (np.zeros((0, 1, 1, 1, 1), np.uint8), [1]),
        (raw, [-1]), (raw, [33]), (raw, []),
    ]
    pool = _native.KeyFPSCPUPool(threads=2)
    try:
        for array, counts in bad:
            with pytest.raises(ValueError):
                pool.select(array, counts)
        assert pool.stats()['jobs'] == 0
    finally:
        pool.close()


def wait_active(pool, expected):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if pool.stats()['active'] == expected:
            return
        time.sleep(.001)
    pytest.fail(f'Expected {expected} active native submissions; got {pool.stats()}')


def test_two_concurrent_submissions_are_bounded_and_close_drains_without_gil_deadlock():
    # Large enough to observe admission, yet bounded by the same 64 MiB input
    # ceiling as serving. One shared immutable buffer is used by both jobs.
    raw = np.zeros((16, 32, 32, 4, 1024), dtype=np.uint8)
    pool = _native.KeyFPSCPUPool(threads=1, max_submissions=2)
    with ThreadPoolExecutor(2) as callers:
        one = callers.submit(pool.select, raw, [32]*32)
        two = callers.submit(pool.select, raw, [32]*32)
        wait_active(pool, 2)
        with pytest.raises(_native.KeyFPSCPUBusyError):
            pool.select(raw, [32]*32)
        pool.close()
        first, a = one.result(timeout=5)
        second, b = two.result(timeout=5)
    assert np.array_equal(first, second) and a.all() and b.all()
    stats = pool.stats()
    assert stats['peak'] == 2 and stats['active'] == 0 and stats['closed']
    assert stats['completed_jobs'] == 2 and stats['busy_rejections'] == 1
    with pytest.raises(RuntimeError, match='closed'):
        pool.select(raw[:, :1].copy(), [32])
    pool.close()


def test_input_owner_lives_until_native_job_finishes_and_destructor_joins_workers():
    raw = np.zeros((16, 32, 32, 4, 1024), dtype=np.uint8)
    owner = weakref.ref(raw)
    before = {entry.name for entry in Path('/proc/self/task').iterdir()}
    pool = _native.KeyFPSCPUPool(threads=1)
    worker_ids = {entry.name for entry in Path('/proc/self/task').iterdir()}-before
    assert worker_ids
    with ThreadPoolExecutor(1) as callers:
        future = callers.submit(pool.select, raw, [32]*32)
        del raw
        wait_active(pool, 1)
        gc.collect()
        assert owner() is not None
        future.result(timeout=5)
    gc.collect()
    assert owner() is None
    del pool
    gc.collect()
    assert not worker_ids.intersection(entry.name for entry in Path('/proc/self/task').iterdir())
