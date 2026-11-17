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
    assert fvg.uniform_query_len(common([5, 5, 0], actual=9), 5) == 0


@pytest.mark.parametrize('lens,actual', [
    ([5] * 3 + [0] * 5, 40),            # FULL graph: 3 requests padded to the 8-request capture
    ([5, 0], 10),
    ([2] * 5 + [0] * 3, 16),            # DFlash2-style k=1
    ([5] * 16, 85),                     # request count capped at max_num_seqs, tokens padded further
])
def test_uniform_query_len_accepts_full_graph_padding(lens, actual):
    """The bug behind the multi-request corruption: vLLM passes the padded token
    count for FULL graphs, and these batches fell through to a stale plan."""
    c = common(lens, actual=actual)
    assert fvg.padded(c)
    assert fvg.uniform_query_len(c, max(lens)) == max(lens)


class _Builder:
    """Just enough of FlashInferMetadataBuilder for install() and the dispatch in build()."""
    def __init__(self, *a):
        self.calls = []

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return 'support'

    def build(self, common_prefix_len, common, fast_build=False):
        self.calls.append(common)
        return 'vllm'

    def _compute_flashinfer_kv_metadata(self, *a):
        raise AssertionError('not reached')

    def _get_workspace_buffer(self):
        raise AssertionError('not reached')


def _patched_builder():
    module = NS(FlashInferMetadataBuilder=type('B', (_Builder,), {}),
                **{name: object() for name in fvg._REQUIRED if name != 'FlashInferMetadataBuilder'})
    fvg.install(module)
    builder = object.__new__(module.FlashInferMetadataBuilder)
    builder.calls = []
    builder._kvmem_verify = dict(limit=5, graph={}, eager=None)
    return builder


@pytest.mark.parametrize('lens,actual', [
    ([5, 4, 0, 0], 20),                 # ragged and padded
    ([5, 0, 5, 0], 20),                 # padding not trailing
    ([6, 6, 0, 0], 24),                 # wider than 1 + k
])
def test_padded_multi_token_batch_outside_the_verify_path_fails_closed(lens, actual):
    c = common(lens, actual=actual)
    c.causal = True
    with pytest.raises(RuntimeError, match='stale|uniform verification'):
        _patched_builder().build(0, c)


def test_padded_non_causal_multi_token_batch_fails_closed():
    c = common([2, 2, 0, 0], actual=8)
    c.causal = False                    # e.g. a non-causal parallel draft on this backend
    with pytest.raises(RuntimeError, match='uniform verification'):
        _patched_builder().build(0, c)


@pytest.mark.parametrize('lens,actual', [
    ([5, 4], 9),                        # ragged, PIECEWISE (exact count): vLLM's prefill path
    ([1] * 4 + [0] * 4, 8),             # single-token FULL decode: vLLM's own graphs
])
def test_unpadded_or_single_token_batches_keep_vllms_build(lens, actual):
    c = common(lens, actual=actual)
    c.causal = True
    builder = _patched_builder()
    assert builder.build(0, c) == 'vllm' and builder.calls == [c]


def config(**kw):
    spec = NS(num_speculative_tokens=kw.get('k', 4), parallel_drafting=kw.get('parallel', False),
              enable_adaptive_verification=False, method=kw.get('method', 'mtp'))
    return NS(speculative_config=spec if kw.get('spec', True) else None,
              parallel_config=NS(decode_context_parallel_size=kw.get('dcp', 1)),
              attention_config=NS(use_non_causal=kw.get('noncausal', False)))


def test_config_gate():
    assert fvg._config_eligible(config())
    assert fvg.max_verify_len(config(k=3)) == 4
    for bad in (dict(spec=False), dict(k=0), dict(parallel=True), dict(dcp=2), dict(noncausal=True)):
        assert not fvg._config_eligible(config(**bad))
    # DFlash drafts in parallel; its target verifies the uniform 1 + k layout.
    assert fvg._config_eligible(config(parallel=True, method='dflash', k=1))
    assert not fvg._config_eligible(config(parallel=True, method='dspark'))


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
