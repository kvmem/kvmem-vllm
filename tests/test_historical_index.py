"""Tensor aliases retain native storage; only the C++ reaper frees its bytes."""
import gc
import time
import weakref

import pytest
import torch

from vllm_kvmem import _native as n


def close(arena):
    deadline = time.monotonic()+5
    while not arena.close():
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_tensor_alias_keeps_native_buffer_after_record_and_python_wrapper_drop():
    arena = n.HistoricalPayloadArena(2, 32, background=False)
    owner = arena._allocate_index(16)
    fence = owner.completion
    tensor = torch.frombuffer(owner, dtype=torch.float32)
    assert torch.equal(tensor, torch.zeros(4))
    tensor.copy_(torch.arange(4.))
    wrapper = weakref.ref(tensor)
    alias = tensor.detach().view(2,2)
    del owner, tensor
    gc.collect()
    # The buffer owner has no Tensor references. Aliases alone keep bytes safe.
    assert wrapper() is None
    assert torch.equal(alias, torch.arange(4.).view(2,2))
    arena.advance()
    assert not fence.query() and arena.stats['external_bytes'] == 16
    del alias
    gc.collect()
    assert wrapper() is None and not fence.query()
    arena.advance()
    assert fence.query() and arena.stats['external_bytes'] == 0
    assert arena.stats['external_released_bytes'] == 16
    close(arena)


def test_capacity_rejection_keeps_original_storage_and_bytes_bound():
    arena = n.HistoricalPayloadArena(1, 32, background=False)
    owner = arena._allocate_index(32)
    mean_owner = arena._allocate_index(16)
    with pytest.raises(RuntimeError, match='capacity'):
        arena._allocate_index(1)
    assert arena.stats['external_owners'] == 2 and arena.stats['external_bytes'] == 48
    del owner, mean_owner
    arena.advance()
    close(arena)


def test_sum_and_mean_views_release_without_a_python_finalizer_on_native_worker():
    arena = n.HistoricalPayloadArena(1, 32, background=False)
    owner = arena._allocate_index(24)
    sums = torch.frombuffer(owner,dtype=torch.float32,count=4)
    means = torch.frombuffer(owner,dtype=torch.float16,count=4,offset=16)
    sums.copy_(torch.arange(4.))
    torch.div(sums,3,out=means)
    assert torch.equal(means,(sums/3).half())
    fence = owner.completion
    assert not arena.close() and not fence.query()
    del owner,sums
    assert not fence.query()
    del means
    close(arena)
    assert fence.query() and arena.stats['external_released_bytes'] == 24
