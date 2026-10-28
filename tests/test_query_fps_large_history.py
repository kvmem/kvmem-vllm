"""Force Key tiling and compare full-domain normalization and native ranks."""
import pytest
import torch
import os

import vllm_kvmem.query_key_fps as fps


@pytest.mark.parametrize('query_half', [False,True])
@pytest.mark.parametrize('optimized', [False,True])
def test_tiled_long_history_matches_original_full_domain(query_half,optimized,monkeypatch):
    torch.manual_seed(17)
    q = torch.randn(2,13,4,8)
    keys = torch.randn(2,47,4,2,8).half()
    indices = torch.arange(4,dtype=torch.int8).view(1,1,4,1).expand(2,47,4,2).clone()
    indices[:,9,2:] = -1
    indices[:,17] = -1
    mask = torch.ones(47,dtype=torch.bool); mask[[3,17]] = False
    reference = fps.key_fps_scores(q,keys,indices,mask,query_half=query_half,optimized=optimized)
    monkeypatch.setattr(fps,'SCRATCH_LIMIT',35000)
    actual = fps.key_fps_scores(q,keys,indices,mask,query_half=query_half,optimized=optimized)
    torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-6)
    assert actual.argsort(descending=True)[:10].tolist() == reference.argsort(descending=True)[:10].tolist()
    assert actual[[3,17]].eq(0).all()
    assert abs(actual.sum().item()-13) < 1e-5


def test_tiling_rejects_missing_representatives_and_real_capacity_exhaustion(monkeypatch):
    q = torch.ones(1,1,2,8)
    keys = torch.ones(1,300,4,1,8)
    indices = torch.zeros(1,300,4,1,dtype=torch.int8)
    mask = torch.ones(300,dtype=torch.bool)
    monkeypatch.setattr(fps,'SCRATCH_LIMIT',30000)
    indices[:,10] = -1
    with pytest.raises(ValueError,match='no valid'): fps.key_fps_scores(q,keys,indices,mask)
    indices[:,10] = 0
    monkeypatch.setattr(fps,'SCRATCH_LIMIT',100)
    with pytest.raises(MemoryError,match='bounded scratch'): fps.key_fps_scores(q,keys,indices,mask)


@pytest.mark.skipif(os.getenv('KVMEM_TEST_CUDA') != '1', reason='Explicit GPU component opt-in')
@pytest.mark.parametrize('rows', [1, 17, 264])
def test_layer_residency_preserves_exact_scores_and_bounds_transfer(rows):
    torch.manual_seed(43)
    # Force the long-history path with the actual target GQA/Key geometry.
    q = torch.randn(1,rows,24,256,device='cuda',dtype=torch.bfloat16)
    reps = torch.randn(1,12864,4,4,256).to(torch.float8_e4m3fn).half()
    ids = torch.arange(4,dtype=torch.int8).view(1,1,4,1).expand(1,12864,4,4).clone()
    mask = torch.ones(12864,dtype=torch.bool); mask[::19] = False
    ids[:,7,2:] = -1
    old_stats, new_stats = {}, {}
    old = fps.key_fps_scores(q,reps,ids,mask,query_half=True,optimized=False,execution_stats=old_stats)
    new = fps.key_fps_scores(q,reps,ids,mask,query_half=True,optimized=True,execution_stats=new_stats)
    # optimized=False can take the original full-layer path at this size;
    # directly qualify the old tiled arithmetic as well.
    reference = fps._key_fps_tiled_scores(q,reps,ids,mask.to('cuda'),scale=256**-.5,
        query_half=True,query_chunk=8)
    assert torch.equal(new,reference)
    torch.testing.assert_close(new,old,rtol=2e-6,atol=2e-6)
    assert new_stats['scoring_cached_layers'] == 1
    assert new_stats['scoring_key_upload_bytes'] == reps.numel()*reps.element_size()
    assert new_stats['scoring_scratch_bound_bytes'] <= fps.SCRATCH_LIMIT
    assert new[~mask].eq(0).all()
