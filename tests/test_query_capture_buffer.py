"""Exact Query values and ownership across reusable forward buffers."""
from types import SimpleNamespace
import os

import pytest

torch = pytest.importorskip('torch')
from vllm_kvmem.query_capture_buffer import (
    append_query_chunk, query_for_submit, release_query_buffer)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='Explicit GPU qualification only'))])
def test_partial_quanta_keep_exact_values_after_capture_reuse_without_full_copy(dtype, device, monkeypatch):
    job = SimpleNamespace(query_rows=[], guided_rows=[])
    source = torch.arange(2*4*3*2, device=device).reshape(2,4,3,2).to(dtype)
    expected = []
    for lo, hi, shift in [(2,4,0), (0,4,100), (0,1,200)]:
        source.copy_(torch.arange(source.numel(), device=device).reshape_as(source)+shift)
        expected.append(source[:,lo:hi].clone())
        append_query_chunk(job, source, lo, hi, 7)
        source.fill_(-1)  # The next model forward reuses its capture storage.
    reference = torch.cat(expected, dim=1)
    pointers = {row.untyped_storage().data_ptr() for row in job.query_rows}
    assert pointers == {job.query_buffer.untyped_storage().data_ptr()}
    monkeypatch.setattr(torch, 'cat', lambda *a, **k: pytest.fail('Full Query copy at submit'))
    assert torch.equal(query_for_submit(job), reference)
    assert query_for_submit(job) is job.query_buffer
    job.query_rows.clear()
    release_query_buffer(job)
    assert job.query_buffer is None and job.query_buffer_tokens == 0


def test_incomplete_or_overlong_capture_cannot_be_published():
    job = SimpleNamespace(query_rows=[], guided_rows=[])
    source = torch.ones(2,3,1,2)
    append_query_chunk(job, source, 0, 2, 4)
    with pytest.raises(ValueError, match='incomplete'):
        query_for_submit(job)
    with pytest.raises(ValueError, match='exceeds'):
        append_query_chunk(job, source, 0, 3, 4)
    assert job.query_buffer_tokens == 2
    append_query_chunk(job, source, 0, 2, 4)
    assert torch.equal(query_for_submit(job), torch.ones(2,4,1,2))


def test_guided_query_and_fallback_retain_their_original_values():
    job = SimpleNamespace(query_rows=[], guided_rows=[])
    original = torch.ones(2,3,1,2)
    append_query_chunk(job, original, 0, 3, 3)
    job.guided_rows = [original+4, original+8]
    assert torch.equal(query_for_submit(job), torch.cat(job.guided_rows, dim=1))
    job.guided_rows.clear()
    assert torch.equal(query_for_submit(job), original)
