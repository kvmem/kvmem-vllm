"""Exact host snapshots for transient synchronous query replay."""
import os

import pytest
import torch

from vllm_kvmem.core.state import StateCheckpoints

gpu = pytest.mark.skipif(
    os.getenv("KVMEM_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Opt-in CUDA checkpoint gate",
)


def test_host_snapshot_preserves_schema_budget_and_identity():
    state = {"conv": torch.arange(48).reshape(4, 12).float().t(),
             "accepted": torch.tensor([2], dtype=torch.int32)}
    store = StateCheckpoints("owner", 4096, capacity=1, offload_to_cpu=True)
    checkpoint = store.capture(7, state)
    expected = {name: value.clone() for name, value in state.items()}
    for tensor in state.values():
        tensor.zero_()
    with pytest.raises(BufferError):
        store.capture(8, state)
    with pytest.raises(ValueError, match="State schema"):
        store.restore(checkpoint, dict(state, accepted=state["accepted"].float()))
    assert not state["conv"].any()
    store.restore(checkpoint, state)
    assert all(torch.equal(state[name], value) for name, value in expected.items())
    assert store.new_branch()
    with pytest.raises(ValueError, match="expired"):
        store.restore(checkpoint, state)


@gpu
def test_pinned_capture_restores_strided_multi_owner_bytes_and_counters():
    pool = torch.arange(2 * 8 * 16, device="cuda").reshape(2, 8, 16).float()
    bf16 = pool.to(torch.bfloat16)
    counters = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    state = {f"{i}/ssm": pool[i].t() for i in range(2)}
    state.update({f"{i}/conv": bf16[i, :, ::2] for i in range(2)})
    state["accepted"] = counters
    expected = {name: tensor.cpu().clone() for name, tensor in state.items()}
    store = StateCheckpoints("two-owners", 100000, offload_to_cpu=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        checkpoint = store.capture(200000, state)
        for tensor in state.values():
            tensor.fill_(-3)
        completed = store.restore(checkpoint, state)
    completed.synchronize()
    assert all(t.device.type == "cpu" and t.is_pinned()
               for t in checkpoint.tensors.values())
    for name, tensor in state.items():
        assert torch.equal(tensor.cpu().contiguous().view(torch.uint8),
                           expected[name].contiguous().view(torch.uint8))
    with pytest.raises(ValueError, match="State schema"):
        store.restore(checkpoint, {name: tensor.cpu() for name, tensor in state.items()})
    assert store.release(200000)


@gpu
def test_host_snapshot_avoids_device_duplicate_and_survives_graph_replay():
    tensor = torch.arange(4 * 1024 * 1024, device="cuda", dtype=torch.float32)
    state = {"ssm": tensor}
    expected = tensor.cpu()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tensor.add_(3)
    tensor.copy_(expected)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    store = StateCheckpoints("graph-owner", tensor.numel() * tensor.element_size(),
                             offload_to_cpu=True)
    checkpoint = store.capture(1, state)
    checkpoint.event.synchronize()
    assert torch.cuda.memory_allocated() - before < 1024 * 1024
    graph.replay()
    store.restore(checkpoint, state).synchronize()
    assert torch.equal(tensor.cpu().view(torch.uint8), expected.view(torch.uint8))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(tensor.cpu(), expected + 3)
    assert store.release(1)
