"""KVMEM_PORTABLE_SCORE_LAYERS: page_attention_layers (one fused pass over every
layerwise layer) equals torch.stack of per-layer page_attention bit for bit."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
                                reason='set KVMEM_RUN_GPU_TESTS=1 on a CUDA host')

LAYERS, ROWS, QH, KH, DIM, PAGE, BLOCK = 16, 5, 12, 2, 256, 16, 32


def _setup(q_dtype, length, seed=0):
    from vllm_kvmem.core.codec import native_vllm_fp8_views
    torch.manual_seed(seed)
    dev = 'cuda'
    pages = (length + PAGE - 1) // PAGE + 3
    # One buffer for all layers, each layer a BHN(2D) cache view like vLLM's.
    raw = (torch.randn(LAYERS, pages, KH, PAGE, 2*DIM, device=dev) * 0.7).to(torch.float8_e4m3fn)
    keys = [native_vllm_fp8_views(raw[layer], KH, DIM)[0] for layer in range(LAYERS)]
    table = torch.randperm(pages, device=dev, dtype=torch.int32)[:(length + PAGE - 1)//PAGE]
    positions = torch.tensor([length - ROWS + i for i in range(ROWS)], device=dev, dtype=torch.int64)
    raw_q = torch.randn(LAYERS, ROWS, QH, DIM, device=dev, dtype=torch.bfloat16)
    cos_sin = torch.randn(length + 8, 128, device=dev, dtype=torch.float32)
    key_scales = torch.rand(LAYERS, device=dev) + 0.5
    q_scales = (torch.rand(LAYERS, device=dev) + 0.5) if q_dtype == torch.float8_e4m3fn else torch.ones(LAYERS, device=dev)
    # A complete partition of [0, length) in BLOCK-token segments, the last one partial.
    starts = torch.arange(0, length, BLOCK, device=dev, dtype=torch.int32)
    counts = torch.clamp(length - starts, max=BLOCK).to(torch.int32)
    return raw_q, positions, keys, table, cos_sin, key_scales, q_scales, starts, counts


@pytest.mark.parametrize('q_dtype', [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize('length', [37, 3000, 33001])
def test_layer_batched_scores_equal_per_layer_page_attention(q_dtype, length):
    from vllm_kvmem.serving.action_scoring import page_attention, page_attention_layers
    raw_q, positions, keys, table, cos_sin, key_scales, q_scales, starts, counts = _setup(q_dtype, length)
    scale = DIM ** -0.5
    expected = torch.stack([
        page_attention(raw_q[layer], positions, keys[layer], table, cos_sin, key_scales[layer], starts, counts,
                       length, scale=scale, q_dtype=q_dtype, q_scale=q_scales[layer], block_tokens=BLOCK,
                       fused=True, complete_partition=True)
        for layer in range(LAYERS)])
    actual = page_attention_layers(raw_q, positions, keys, table, cos_sin, key_scales, q_scales, starts, counts,
                                   length, scale=scale, q_dtype=q_dtype, block_tokens=BLOCK)
    torch.cuda.synchronize()
    assert actual is not None and actual.shape == (LAYERS, ROWS, starts.numel())
    assert torch.equal(actual, expected)
    assert actual._action_scoring_backend == 'fused'


def test_layer_batched_scores_decline_what_page_attention_would_not_fuse():
    from vllm_kvmem.serving.action_scoring import page_attention_layers
    raw_q, positions, keys, table, cos_sin, key_scales, q_scales, starts, counts = _setup(torch.bfloat16, 3000)
    kwargs = dict(scale=DIM ** -0.5, q_dtype=torch.bfloat16)
    assert page_attention_layers(raw_q, positions, keys, table, cos_sin, key_scales, q_scales, starts, counts,
                                 3000, block_tokens=256, **kwargs) is None
    empty = starts[:0]
    assert page_attention_layers(raw_q, positions, keys, table, cos_sin, key_scales, q_scales, empty, empty,
                                 3000, block_tokens=BLOCK, **kwargs) is None
