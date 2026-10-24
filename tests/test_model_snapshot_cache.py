"""Cached state must equal the CPU authority and survive eviction in flight."""
from dataclasses import replace
import gc
import os

import pytest
import torch

from vllm_kvmem.model_snapshot_cache import ModelSnapshotCache
from test_session_state import fixture


def snapshot(adapter):
    value = adapter.capture('a', 8, max_bytes=4096)
    return value, value.payload.clone()


def test_capacity_identity_and_checkpoint_release():
    adapter, *_ = fixture(draft=False)
    a, packed = snapshot(adapter)
    cache = ModelSnapshotCache(a.bytes)
    assert cache.put(a, packed)
    assert cache.get(a) is packed
    b = replace(a)
    assert cache.get(b) is None  # Same position/content is not an identity proof.
    assert cache.put(b, packed.clone())
    assert cache.get(a) is None and cache.stats()['evictions'] == 1
    assert cache.bytes == a.bytes
    del b
    gc.collect()
    assert cache.bytes == 0 and not cache.entries


@pytest.mark.parametrize('draft', [False, True])
def test_restore_cached_clone_equals_cpu_after_native_source_overwrite_and_eviction(draft):
    adapter, conv, ssm, kv, _ = fixture(draft=draft)
    original, packed = snapshot(adapter)
    adapter.snapshot_cache = ModelSnapshotCache(original.bytes)
    adapter.snapshot_cache.put(original, packed)
    before = conv.clone(), ssm.clone(), kv.clone()
    prepared = adapter.prepare_restore(original, 'b')
    assert prepared.cached_payload is packed
    # A new snapshot may evict the old cache while its prepared copy is pending.
    other = replace(original)
    adapter.snapshot_cache.put(other, packed.clone())
    conv.fill_(-11); ssm.fill_(-22); kv.fill_(-33)
    adapter.submit_restore(prepared)
    actual = conv.clone(), ssm.clone(), kv.clone()
    conv.fill_(-11); ssm.fill_(-22); kv.fill_(-33)
    adapter.restore(original, 'b')  # Evicted snapshot falls back to CPU.
    assert all(torch.equal(a, b) for a, b in zip(actual, (conv, ssm, kv)))
    assert torch.equal(conv[3], before[0][1])
    assert torch.equal(ssm[3], before[1][1])


def test_disabled_oversize_and_recycled_destination():
    adapter, conv, ssm, *_ = fixture(draft=False)
    value, packed = snapshot(adapter)
    assert not ModelSnapshotCache(0).put(value, packed)
    assert not ModelSnapshotCache(value.bytes - 1).put(value, packed)
    adapter.snapshot_cache = ModelSnapshotCache(value.bytes)
    adapter.snapshot_cache.put(value, packed)
    prepared = adapter.prepare_restore(value, 'b')
    before = conv.clone(), ssm.clone()
    adapter.runner._kvmem_request_generations['b'] += 1
    with pytest.raises(ValueError, match='recycled'):
        adapter.submit_restore(prepared)
    assert torch.equal(conv, before[0]) and torch.equal(ssm, before[1])


def test_bad_schema_rejected_and_cache_reusable():
    adapter, *_ = fixture(draft=False)
    value, packed = snapshot(adapter)
    cache = ModelSnapshotCache(value.bytes)
    with pytest.raises(ValueError, match='schema'):
        cache.put(value, packed[:-1])
    assert cache.bytes == 0 and cache.put(value, packed)


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='Queued GPU qualification only')
def test_cuda_terminal_snapshot_replay_eviction_and_public_neighbour():
    from types import SimpleNamespace
    adapter, conv, ssm, *_ = fixture('cuda', draft=False)
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    adapter.runner.req_states.prefill_len.np[0] = 5
    adapter.runner._kvmem_capture_runtime = SimpleNamespace(
        last_steps={'a':[{'history_begin':7,'accepted':1,'scheduled':1}]},
        authorities={'a':SimpleNamespace(history_end=8)})
    adapter.snapshot_cache = ModelSnapshotCache(4096)
    before = conv.clone(), ssm.clone()
    state = adapter.capture('a', 8, max_bytes=4096, terminal=True)
    state.completion.synchronize()
    assert state.query()
    prepared = adapter.prepare_restore(state, 'b')
    assert prepared.cached_payload is not None
    conv[1].fill_(-11); ssm[1].fill_(-22)
    stream = torch.cuda.Stream()
    ready = torch.cuda.Event(); ready.record()
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        completion = adapter.submit_restore(prepared)
    completion.synchronize()
    assert torch.equal(conv[3], before[0][1]) and torch.equal(ssm[3], before[1][1])
    assert torch.equal(conv[5:], before[0][5:]) and torch.equal(ssm[5:], before[1][5:])
    # Capture/replay copies only into the leased destination; cache eviction
    # cannot release a tensor still owned by the prepared graph's lifetime.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        adapter.submit_restore(prepared)
    adapter.snapshot_cache.capacity = 0
    with adapter.snapshot_cache.lock:
        adapter.snapshot_cache._drop(id(state))
    conv[3].fill_(-33); ssm[3].fill_(-44)
    graph.replay(); torch.cuda.synchronize()
    assert torch.equal(conv[3], before[0][1]) and torch.equal(ssm[3], before[1][1])
