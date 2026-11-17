"""GPU checks: QSA attention logits taken from the B12X kernel (explicit opt-in).

The score-storing variant (adapters/qsa_scores.py) must compute the attention
bitwise like the vendor launch, store the logits that attention normalized, and
give the same per-block attention mass as the torch recomputation
(serving/qsa_pressure.py). Geometry follows the server: 3,008-token pages
holding K and V interleaved in one padded page.
"""
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='explicit GPU opt-in')

HEADS, KV_HEADS, DIM, PAGE, WIDTH = 24, 2, 256, 3008, 2051
STRIDE = PAGE * KV_HEADS * DIM * 2 + PAGE // 4 * 128 * 2
PAGES, TABLE = 10, 4


def _modules():
    import importlib
    sparse = importlib.import_module('b12x.attention.qsa._sparse_gqa')
    forward = importlib.import_module('b12x.attention.paged._selected_forward')
    return sparse, forward


def _inputs(rows, seed):
    import torch
    torch.manual_seed(seed)
    device = 'cuda'
    backing = (torch.randn(PAGES * STRIDE, device=device) * 0.5).to(torch.float8_e4m3fn)
    key = backing.as_strided((PAGES, PAGE, KV_HEADS, DIM), (STRIDE, KV_HEADS * DIM, DIM, 1))
    value = backing.as_strided((PAGES, PAGE, KV_HEADS, DIM), (STRIDE, KV_HEADS * DIM, DIM, 1), PAGE * KV_HEADS * DIM)
    table = torch.randperm(PAGES, device=device)[:2 * TABLE].view(2, TABLE).to(torch.int32).contiguous()
    half = rows // 2
    request_ids = torch.tensor([0] * half + [1] * (rows - half), dtype=torch.int32, device=device)
    positions = torch.cat((9000 + torch.arange(half), 11000 + torch.arange(rows - half))).to(device)
    selected = torch.stack([torch.randperm(TABLE * PAGE)[:WIDTH] for _ in range(rows)]).to(torch.int32)
    selected[:, -5:] = -1                                      # padding columns
    query = torch.randn(rows, HEADS, DIM, dtype=torch.bfloat16, device=device)
    return dict(query=query, key_cache=key, value_cache=value,
                k_descale=torch.tensor([1.3], device=device), v_descale=torch.tensor([0.9], device=device),
                block_table=table, request_ids=request_ids, selected_positions=selected.to(device).contiguous(),
                query_positions=positions.to(torch.int64).contiguous())


def _vendor(sparse, inputs, direct_kv_warps):
    """The vendor's prepared launch (as _qsa_attention_op calls it); returns the output."""
    import importlib
    import torch
    rows = int(inputs['query'].shape[0])
    prepared = sparse.compile_sparse_paged_gqa(
        query=inputs['query'], key_cache=inputs['key_cache'], value_cache=inputs['value_cache'],
        request_ids=inputs['request_ids'], selected_positions=inputs['selected_positions'],
        direct_kv_warps=direct_kv_warps)
    if prepared.get('merge') is None:
        # Outside compile-only planning the merge launcher compiles but returns
        # nothing; the server's prepared plan holds this same carrier.
        forward = importlib.import_module('b12x.attention.paged._selected_forward')
        prepared['merge'] = next(raw for key, raw in forward._MERGE_CACHE.items()
                                 if key[1] == HEADS and key[4] is False)
    splits = 1 if rows > sparse.split_row_limit(KV_HEADS) else (64 if rows == 1 else 32 if rows <= 4 else 16)
    output = torch.empty(rows, HEADS, DIM, dtype=torch.bfloat16, device='cuda')
    partial_output = torch.empty(rows, splits, HEADS, DIM, dtype=torch.float32, device='cuda')
    partial_lse = torch.empty(rows, splits, HEADS, dtype=torch.float32, device='cuda')
    sparse.launch_sparse_paged_gqa(**inputs, output=output, output_lse=None, partial_output=partial_output,
                                   partial_lse=partial_lse, softmax_scale=DIM ** -0.5, block_n=16,
                                   splits=splits, direct_kv_warps=direct_kv_warps, _prepared=prepared)
    return output, prepared, splits, partial_output, partial_lse


def _reference_logits(inputs, rows):
    """[len(rows), heads, width] q.k (FP8 K, no scale) with the kernel's -inf mask."""
    import torch
    query, key = inputs['query'], inputs['key_cache']
    out = []
    for row in rows:
        pos = inputs['selected_positions'][row].long()
        valid = (pos >= 0) & (pos <= inputs['query_positions'][row])
        safe = pos.clamp(min=0)
        pages = inputs['block_table'][int(inputs['request_ids'][row])].long()[safe // PAGE]
        k = key[pages, safe % PAGE].float()                                   # [W, kv, D]
        k = k[:, torch.arange(HEADS, device=k.device) // (HEADS // KV_HEADS)]  # [W, H, D]
        logits = torch.einsum('hd,whd->hw', query[row].float(), k)
        out.append(logits.masked_fill(~valid[None], float('-inf')))
    return torch.stack(out)


@pytest.mark.parametrize('rows,direct_kv_warps', [(6, 2), (40, 2), (160, 2), (160, 4)])
def test_variant_attention_is_bitwise_the_vendor_attention(rows, direct_kv_warps):
    import torch
    from vllm_kvmem.adapters import qsa_scores
    sparse, forward = _modules()
    inputs = _inputs(rows, seed=rows + direct_kv_warps)
    want, prepared, splits, partial_output, partial_lse = _vendor(sparse, inputs, direct_kv_warps)
    module = qsa_scores.load_module(forward)
    direct = rows > sparse.split_row_limit(KV_HEADS)
    raw = qsa_scores.compile_variant(module, query=inputs['query'], key_cache=inputs['key_cache'],
                                     value_cache=inputs['value_cache'], request_ids=inputs['request_ids'],
                                     direct=direct, return_lse=False, kv_warps=direct_kv_warps,
                                     selection_width=WIDTH)
    marked = sorted({0, 3, rows // 2, rows - 1})
    slots = torch.full((rows,), -1, dtype=torch.int32, device='cuda')
    slots[marked] = torch.arange(len(marked), dtype=torch.int32, device='cuda')
    scores = torch.empty(len(marked), HEADS, WIDTH, dtype=torch.float32, device='cuda')
    got = torch.empty_like(want)
    qsa_scores.launch_scores(module, raw, prepared['merge'], **inputs, output=got, output_lse=None,
                             partial_output=partial_output, partial_lse=partial_lse, softmax_scale=DIM ** -0.5,
                             splits=splits, direct=direct, score_rows=slots, scores=scores)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), 'attention output changed'
    reference = _reference_logits(inputs, marked)
    assert torch.equal(torch.isinf(scores), torch.isinf(reference))
    finite = torch.isfinite(reference)
    assert torch.allclose(scores[finite], reference[finite], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('rows', [40, 160])
def test_sampled_logits_give_the_recomputed_masses(monkeypatch, rows):
    import torch
    from vllm_kvmem.adapters import qsa_scores
    from vllm_kvmem.serving.qsa_pressure import QSALayerPressure
    sparse, forward = _modules()
    monkeypatch.setattr(qsa_scores, '_vendor_modules', lambda: (sparse, forward))
    inputs = _inputs(rows, seed=rows)
    binding = SimpleNamespace(selected_positions=inputs['selected_positions'], main_k_cache=inputs['key_cache'],
                              main_v_cache=inputs['value_cache'], k_descale=inputs['k_descale'],
                              v_descale=inputs['v_descale'], main_block_table=inputs['block_table'])
    sampled = list(range(1, rows, 7))
    scores = qsa_scores.sampled_logits(binding, inputs, sampled)
    torch.cuda.synchronize()
    reference = _reference_logits(inputs, sampled)
    assert torch.equal(torch.isinf(scores), torch.isinf(reference))
    finite = torch.isfinite(reference)
    assert torch.allclose(scores[finite], reference[finite], rtol=1e-4, atol=1e-3)

    starts = list(range(0, TABLE * PAGE, 96))
    pressure = QSALayerPressure.__new__(QSALayerPressure)
    capture = SimpleNamespace(scores=scores, rows=sampled)
    subset = sampled[::2]
    from_kernel = QSALayerPressure._kernel_masses(pressure, binding, inputs, capture, subset, starts)
    recomputed = QSALayerPressure._masses(pressure, binding, inputs, subset, starts)
    assert torch.allclose(from_kernel, recomputed, atol=2e-6), (from_kernel - recomputed).abs().max()
    assert torch.allclose(from_kernel.sum(1), torch.ones(len(subset), device='cuda'), atol=1e-5)
