"""MeanK shared-load tail regression; also run under compute-sanitizer memcheck.

Ordinary CUDA execution can hide an over-read inside the hardware's rounded
shared allocation. In particular, 4801 eligible pages exposed a 12-byte invalid
read in the original sm_120 binary. Mask compaction determines the page count,
not the original history length.
"""
import pytest

torch = pytest.importorskip('torch')
from vllm_kvmem.cuda_scoring import mean_key_scores
from vllm_kvmem.retrieval import mean_key_scores as reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA qualification')


@pytest.mark.parametrize('eligible_count', [1, 2, 3, 5, 4791, 4800, 4801, 4802, 4803,
                                           8191, 8192, 8193])
def test_compacted_tail_stays_inside_shared_allocation(eligible_count):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rng = torch.Generator(device='cuda').manual_seed(1000)
        blocks = eligible_count + 41
        queries = torch.randn(2, 10, 24, 256, generator=rng, device='cuda').half()
        means = torch.randn(2, blocks, 4, 256, generator=rng, device='cuda').half()
        eligible = torch.ones(blocks, dtype=torch.bool, device='cuda')
        eligible[:41] = False
        actual = mean_key_scores(queries, means, eligible)
        expected = reference(queries, means, eligible)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-5)
    assert torch.count_nonzero(actual[~eligible]).item() == 0
    assert abs(actual.sum().item()-10) < 1e-4
