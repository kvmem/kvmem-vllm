"""KVMEM_FA2_VERIFY_GRAPH on real FlashInfer FA2 (fp8 KV): a graph-mode verification
wrapper planned with uniform rows per request (padded requests included, without KV),
re-planned between CUDA graph replays, equals an eager wrapper on the real requests."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
                                reason='set KVMEM_RUN_GPU_TESTS=1 on a CUDA host')

QH, KH, D, PAGE, WIDTH = 12, 2, 256, 16, 2


def _plan_kwargs(qo, indptr, indices, last, lens):
    return dict(qo_indptr=qo, paged_kv_indptr=indptr, paged_kv_indices=indices, paged_kv_last_page_len=last,
                seq_lens=lens, num_qo_heads=QH, num_kv_heads=KH, head_dim_qk=D, page_size=PAGE, causal=True,
                sm_scale=D ** -0.5, window_left=-1, logits_soft_cap=None, q_data_type=torch.bfloat16,
                kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16)


def test_uniform_graph_plan_matches_eager_with_padding_and_replans():
    flashinfer = pytest.importorskip('flashinfer')
    from vllm_kvmem.adapters import fa2_verify_graph as fvg
    fvg._check_plan_abi(flashinfer.BatchPrefillWithPagedKVCacheWrapper)
    Uniform = fvg.uniform_wrapper_class(flashinfer.BatchPrefillWithPagedKVCacheWrapper)
    dev, n = 'cuda', 4
    torch.manual_seed(0)
    pages_total = n * (9000 // PAGE + 2)
    kv = (torch.randn(pages_total, 2, PAGE, KH, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    workspace = lambda: torch.zeros(128 << 20, dtype=torch.uint8, device=dev)
    wrapper = Uniform(workspace(), 'NHD', use_cuda_graph=True,
                      qo_indptr_buf=torch.zeros(n + 1, dtype=torch.int32, device=dev),
                      paged_kv_indptr_buf=torch.zeros(n + 1, dtype=torch.int32, device=dev),
                      paged_kv_indices_buf=torch.zeros(pages_total, dtype=torch.int32, device=dev),
                      paged_kv_last_page_len_buf=torch.zeros(n, dtype=torch.int32, device=dev), backend='fa2')
    wrapper._max_total_num_rows = n * WIDTH
    q = torch.zeros(n * WIDTH, QH, D, dtype=torch.bfloat16, device=dev)
    out = torch.zeros_like(q)
    graph = None
    for real in ([3000, 9000, 5000], [700, 8900, 128, 4097], [16]):
        lens = real + [0] * (n - len(real))
        pages = [(x + PAGE - 1) // PAGE for x in lens]
        indptr = torch.zeros(n + 1, dtype=torch.int32)
        indptr[1:] = torch.tensor(pages, dtype=torch.int32).cumsum(0)
        indices = torch.randperm(pages_total, dtype=torch.int32)[:int(indptr[-1])].to(dev)
        last = torch.tensor([(x % PAGE) or (PAGE if x else 0) for x in lens], dtype=torch.int32)
        klens = torch.tensor(lens, dtype=torch.int32)
        qo = torch.arange(0, (n + 1) * WIDTH, WIDTH, dtype=torch.int32)
        wrapper.plan_uniform(WIDTH, **_plan_kwargs(qo, indptr, indices, last, klens))
        q.normal_()
        if graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                wrapper.run(q, kv, k_scale=1.0, v_scale=1.0, out=out)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                wrapper.run(q, kv, k_scale=1.0, v_scale=1.0, out=out)
        graph.replay()
        r = len(real)
        eager = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace(), 'NHD', backend='fa2')
        eager.plan(**_plan_kwargs(qo[:r + 1], indptr[:r + 1], indices, last[:r], klens[:r]))
        reference = eager.run(q[:r * WIDTH], kv, k_scale=1.0, v_scale=1.0)
        torch.cuda.synchronize()
        assert torch.equal(out[:r * WIDTH], reference), real
        assert torch.isfinite(out[r * WIDTH:]).all(), real
