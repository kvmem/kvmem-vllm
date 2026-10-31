import pytest
torch=pytest.importorskip('torch')
from vllm_kvmem.retrieval import mean_key_scores as reference
from vllm_kvmem.cuda_scoring import mean_key_scores

@pytest.mark.parametrize('empty_axis', range(4))
def test_empty_query_geometry_is_rejected_before_chunking(empty_axis):
    shape = [2, 3, 4, 8]
    shape[empty_axis] = 0
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty(2, 5, 2, 8)
    with pytest.raises(ValueError, match='nonempty geometry'):
        mean_key_scores(q, k, torch.ones(5, dtype=torch.bool))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')
@pytest.mark.parametrize('tokens,blocks,heads,kvheads',[(1,17,8,2),(33,257,16,4),(1025,41,8,2),
    (3,8200,8,2),(3,8201,8,2),(257,20000,8,2),(3,32768,8,2)])
def test_original_fp16_kernel_matches_page_softmax_definition(tokens,blocks,heads,kvheads):
    g=torch.Generator(device='cuda').manual_seed(831)
    q=torch.randn(2,tokens,heads,128,device='cuda',generator=g).half()
    k=torch.randn(2,blocks,kvheads,128,device='cuda',generator=g).half()
    eligible=torch.ones(blocks,device='cuda',dtype=torch.bool)
    eligible[:3]=False;eligible[-4:]=False;eligible[8]=False
    actual=mean_key_scores(q,k,eligible)
    expected=reference(q,k,eligible)
    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=2e-5)
    assert torch.equal(actual[~eligible],torch.zeros_like(actual[~eligible]))
    assert abs(actual.sum().item()-tokens)<tokens*1e-5

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')
def test_no_eligible_history_has_zero_mass():
    q=torch.zeros(1,1,4,128,device='cuda',dtype=torch.float16)
    k=torch.zeros(1,4,2,128,device='cuda',dtype=torch.float16)
    assert torch.equal(mean_key_scores(q,k,torch.zeros(4,device='cuda',dtype=torch.bool)),
        torch.zeros(4,device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')
def test_large_history_qwen_head_geometry_on_current_nondefault_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g=torch.Generator(device='cuda').manual_seed(79)
        q=torch.randn(2,7,24,256,device='cuda',generator=g).half()
        k=torch.randn(2,24000,4,256,device='cuda',generator=g).half()
        eligible=torch.ones(24000,device='cuda',dtype=torch.bool)
        eligible[:41]=False;eligible[12000:12007]=False
        actual=mean_key_scores(q,k,eligible)
        expected=reference(q,k,eligible)
    stream.synchronize()
    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=2e-5)
    assert abs(actual.sum().item()-7)<7e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA qualification')
@pytest.mark.parametrize('blocks', [41, 8201])
def test_bounded_bf16_conversion_keeps_full_page_softmax(monkeypatch, blocks):
    import vllm_kvmem.cuda_scoring as scoring
    from types import SimpleNamespace
    g = torch.Generator(device='cuda').manual_seed(319)
    q = torch.randn(2, 37, 8, 32, device='cuda', generator=g).bfloat16()
    k = torch.randn(2, blocks, 2, 32, device='cuda', generator=g).half()
    eligible = torch.ones(blocks, device='cuda', dtype=torch.bool)
    eligible[:3] = False
    eligible[-4:] = False
    expected = scoring.mean_key_scores(q.half(), k, eligible)
    original = scoring.kernel()
    observed = []
    def score(chunk, means, scale):
        observed.append(chunk.numel() * chunk.element_size())
        assert chunk.dtype == torch.float16 and chunk.is_contiguous()
        assert means.shape[1] == int(eligible.sum())
        return original.score(chunk, means, scale)
    bound = 2 * 5 * 8 * 32 * 2
    monkeypatch.setattr(scoring, 'QUERY_CONVERSION_BYTES', bound)
    monkeypatch.setattr(scoring, 'kernel', lambda: SimpleNamespace(score=score))
    actual = scoring.mean_key_scores(q, k, eligible)
    assert len(observed) == 8 and max(observed) <= bound
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-5)
    assert torch.equal(actual[~eligible], torch.zeros_like(actual[~eligible]))
    assert abs(actual.sum().item() - 37) < 37e-5
