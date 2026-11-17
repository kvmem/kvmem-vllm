"""Layer-batched native reductions (KVMEM_NATIVE_SCORE_STACK) write bitwise the same row LSE,
segment masses and valid flags as the per-layer _row_lse/_segment_scores launches."""
import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
                                reason='opt-in CUDA qualification')


@pytest.mark.parametrize('layers,heads,width,rows,segments', [(16, 12, 2067, 20, 700), (4, 6, 300, 7, 40)])
def test_layered_kernels_equal_per_layer_kernels(layers, heads, width, rows, segments):
    import triton
    from vllm_kvmem.serving.native_attention import (_row_lse, _segment_scores, _row_lse_layers,
                                                     _segment_scores_layers)
    torch.manual_seed(5)
    dev = 'cuda'
    max_rows, max_segments, requests = rows + 3, segments + 5, 3
    lse = torch.randn(layers, max_rows, heads, width, device=dev) * 4
    lse[:, :, :, width - 7:] = -float('inf')                    # padded tail columns
    lse[3 % layers, 1, 2, :] = float('nan')                     # one invalid row in one layer
    meta = torch.zeros(max_rows, 3, dtype=torch.int32, device=dev)
    ranges = torch.zeros(requests, max_segments, 4, dtype=torch.int32, device=dev)
    for request in range(requests):
        cursor = 0
        for segment in range(segments):
            count = 16 + (segment * 7 + request) % 17
            ranges[request, segment, 2], ranges[request, segment, 3] = cursor, count
            cursor += count
    for row in range(max_rows):
        meta[row] = torch.tensor([row % requests, width - 3 - row % 5, segments - row % 4])
    control = torch.tensor([rows], dtype=torch.int32, device=dev)
    selected = torch.tensor(list(range(0, rows, 2)) + list(range(1, rows, 2)) + [max_rows - 1],
                            dtype=torch.int32, device=dev)          # last row >= control: skipped
    n = selected.numel()
    block = triton.next_power_of_2(width)
    b = triton.next_power_of_2(math.ceil(32 / 16) + 1)
    # Per-layer reference: one shared row-LSE scratch, as NativeAttentionStats._reduce uses.
    ref_masses = torch.full((layers, max_rows, max_segments), -1., device=dev)
    ref_valid = torch.ones(layers, dtype=torch.int32, device=dev)
    ref_row = torch.full((layers, max_rows, heads), -1., device=dev)
    scratch = torch.full((max_rows, heads), -1., device=dev)
    for layer in range(layers):
        _row_lse[(n, heads)](lse[layer], meta, control, selected, ref_valid[layer], scratch,
                             heads, width, block, num_warps=8)
        _segment_scores[(n, segments)](lse[layer], scratch, ranges, meta, control, selected, ref_masses[layer],
                                       heads, width, max_segments, triton.next_power_of_2(heads), b)
        ref_row[layer] = scratch
    masses = torch.full_like(ref_masses, -1.)
    valid = torch.ones_like(ref_valid)
    row_lse = torch.full_like(ref_row, -1.)
    _row_lse_layers[(n, heads, layers)](lse, meta, control, selected, valid, row_lse, lse.stride(0),
                                        row_lse.stride(0), heads, width, block, num_warps=8)
    _segment_scores_layers[(n, segments, layers)](lse, row_lse, ranges, meta, control, selected, masses,
                                                  lse.stride(0), row_lse.stride(0), masses.stride(0), heads, width,
                                                  max_segments, triton.next_power_of_2(heads), b)
    torch.cuda.synchronize()
    assert torch.equal(valid, ref_valid) and valid[3 % layers].item() == 0 and int(valid.sum()) == layers - 1
    assert torch.equal(torch.nan_to_num(row_lse, nan=7.), torch.nan_to_num(ref_row, nan=7.))
    assert torch.equal(torch.nan_to_num(masses, nan=7.), torch.nan_to_num(ref_masses, nan=7.))
    assert (masses[:, :rows, :segments - 3] != -1).all()        # every selected row/segment was written
