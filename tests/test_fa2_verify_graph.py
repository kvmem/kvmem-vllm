"""KVMEM_FA2_VERIFY_GRAPH (adapters/fa2_verify_graph.py): which batches take the
verification graph path, when the backend advertises UNIFORM_BATCH, and that
the patch installs on the pinned vLLM FlashInfer backend."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.adapters import fa2_verify_graph as fvg


def common(lens, actual=None):
    loc = torch.zeros(len(lens) + 1, dtype=torch.int32)
    loc[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    return NS(num_reqs=len(lens), max_query_len=max(lens), query_start_loc_cpu=loc,
              num_actual_tokens=sum(lens) if actual is None else actual)


@pytest.mark.parametrize('lens,want', [
    ([5] * 8, 5),
    ([5] * 7 + [0], 5),                 # FULL-graph padding: trailing empty requests
    ([5] * 3 + [0] * 5, 5),
    ([3, 3], 3),                        # fewer drafts, still uniform
    ([5], 5),
    ([1] * 8, 0),                       # single-token decodes stay on vLLM's decode path
    ([5, 5, 4], 0),                     # ragged
    ([5, 0, 5], 0),                     # padding only at the end
    ([6, 6], 0),                        # longer than 1 + k: a prefill chunk
    ([5, 5, 40], 0),                    # mixed with a prefill
    ([0, 0], 0),
])
def test_uniform_query_len(lens, want):
    assert fvg.uniform_query_len(common(lens), 5) == want


def test_uniform_query_len_checks_actual_tokens():
    assert fvg.uniform_query_len(common([5, 5, 0], actual=10), 5) == 5
    assert fvg.uniform_query_len(common([5, 5, 0], actual=11), 5) == 0


def config(**kw):
    spec = NS(num_speculative_tokens=kw.get('k', 4), parallel_drafting=kw.get('parallel', False),
              enable_adaptive_verification=False)
    return NS(speculative_config=spec if kw.get('spec', True) else None,
              parallel_config=NS(decode_context_parallel_size=kw.get('dcp', 1)),
              attention_config=NS(use_non_causal=kw.get('noncausal', False)))


def test_config_gate():
    assert fvg._config_eligible(config())
    assert fvg.max_verify_len(config(k=3)) == 4
    for bad in (dict(spec=False), dict(k=0), dict(parallel=True), dict(dcp=2), dict(noncausal=True)):
        assert not fvg._config_eligible(config(**bad))


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_FA2_VERIFY_GRAPH', 'yes')
    with pytest.raises(ValueError):
        fvg.enabled()
    monkeypatch.setenv('KVMEM_FA2_VERIFY_GRAPH', '1')
    assert fvg.enabled()
    monkeypatch.delenv('KVMEM_FA2_VERIFY_GRAPH')
    assert not fvg.enabled()


def test_install_patches_the_pinned_backend():
    fi = pytest.importorskip('vllm.v1.attention.backends.flashinfer')
    from vllm.v1.attention.backend import AttentionCGSupport
    fvg.install(fi)
    fvg.install(fi)                                         # idempotent
    builder = fi.FlashInferMetadataBuilder
    assert builder._kvmem_fa2_verify_graph and builder.build.__module__ == fvg.__name__
    # Only the single-token-decode answer is upgraded, and only for eligible configs.
    original = builder.get_cudagraph_support
    assert AttentionCGSupport.UNIFORM_BATCH.value > AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE.value


def test_install_refuses_an_unknown_backend():
    with pytest.raises(RuntimeError):
        fvg.install(NS())
