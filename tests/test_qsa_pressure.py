"""QSA per-layer pressure: sampled rows' attention mass over window blocks."""
import math
from types import SimpleNamespace

import torch

from vllm_kvmem.serving.qsa_pressure import QSALayerPressure


def _reference_masses(query, key_cache, descale, block_table, request_ids, selected, positions, starts, rows):
    """b12x sparse_paged_gqa_reference's probabilities, summed per window block."""
    heads, dim = query.shape[1], query.shape[2]
    kv_heads, page = key_cache.shape[2], key_cache.shape[1]
    head_to_kv = torch.arange(heads) // (heads // kv_heads)
    out = torch.zeros(len(rows), len(starts), dtype=torch.float64)
    for i, row in enumerate(rows):
        pos = selected[row].long()
        pos = pos[(pos >= 0) & (pos <= positions[row])]
        pages = block_table[int(request_ids[row]), pos // page].long()
        k = key_cache[pages, pos % page].double()[:, head_to_kv] * float(descale)
        logits = torch.einsum('hd,khd->hk', query[row].double(), k) * dim ** -0.5
        p = torch.softmax(logits, dim=-1).mean(0)
        for value, at in zip(p.tolist(), pos.tolist()):
            block = max(j for j, s in enumerate(starts) if s <= at)
            out[i, block] += value
    return out


def test_masses_match_the_sparse_attention_reference():
    torch.manual_seed(0)
    rows_total, heads, kv_heads, dim, page, pages, width = 12, 8, 2, 16, 16, 12, 24
    key_cache = (torch.randn(pages, page, kv_heads, dim) * 0.5).to(torch.float8_e4m3fn)
    block_table = torch.tensor([[3, 1, 7, 5, 9], [2, 4, 6, 8, 10]], dtype=torch.int32)
    request_ids = torch.tensor([0] * 6 + [1] * 6, dtype=torch.int32)
    positions = torch.tensor([40, 41, 42, 43, 44, 45, 60, 61, 62, 63, 64, 65], dtype=torch.int64)
    selected = torch.randint(0, 70, (rows_total, width), dtype=torch.int32)
    selected[:, -3:] = -1                                   # padding entries
    query = torch.randn(rows_total, heads, dim, dtype=torch.bfloat16)
    binding = SimpleNamespace(selected_positions=selected, main_k_cache=key_cache,
                              k_descale=torch.tensor(1.75), main_block_table=block_table)
    inputs = dict(query=query, query_positions=positions, request_ids=request_ids)
    starts = [0, 32, 40, 64]                                # window blocks (frame positions)
    rows = [1, 4, 7, 10, 11]
    pressure = QSALayerPressure.__new__(QSALayerPressure)
    got = QSALayerPressure._masses(pressure, binding, inputs, rows, starts).double()
    want = _reference_masses(query, key_cache, 1.75, block_table, request_ids, selected, positions, starts, rows)
    assert torch.allclose(got, want, atol=1e-5), (got, want)
    assert torch.allclose(got.sum(1), torch.ones(len(rows), dtype=torch.float64), atol=1e-5)
