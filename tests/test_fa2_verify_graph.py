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


class _FakeFA2Module:
    """FlashInfer's compiled FA2 module: records plan arguments."""
    def __init__(self):
        self.plans = []

    def plan(self, *args):
        self.plans.append(args)
        return 'plan-info'

    def paged_run(self, *args):
        return 'ran'


class _FakePrefillWrapper:
    """BatchPrefillWithPagedKVCacheWrapper in CUDA-graph mode, reduced to the FA2 plan ABI."""
    def __init__(self, float_workspace, layout, use_cuda_graph=False, **buffers):
        self.use_cuda_graph, self.buffers, self.kwargs = use_cuda_graph, buffers, None
        self._cached_module = None

    def plan(self, **kwargs):
        if self._cached_module is None or not isinstance(self._cached_module, (_FakeFA2Module, fvg._UniformPlanModule)):
            self._cached_module = _FakeFA2Module()
        self.kwargs = kwargs
        args = [None] * 11 + [self.use_cuda_graph] + [None] * 4
        args.append(-1)  # fixed_split_size
        args.append(False)  # disable_split_kv
        args.append(0)  # num_colocated_ctas
        args.append(0)  # uniform_q_len
        self._plan_info = self._cached_module.plan(*args)


def _fake_module(builder_base=None):
    from contextlib import nullcontext
    module = NS(FlashInferMetadataBuilder=type('B', (builder_base or _Builder,), {}),
                BatchPrefillWithPagedKVCacheWrapper=_FakePrefillWrapper,
                FlashInferMetadata=lambda **kw: NS(**kw), FIDecode=lambda wrapper: NS(wrapper=wrapper),
                get_flashinfer_layout_string=lambda layout: 'NHD', PIN_MEMORY=False,
                gpu_sync_allowed=nullcontext,
                get_seq_lens=lambda indptr, last, page: (indptr[1:] - indptr[:-1] - 1).clamp(min=0) * page + last)
    for name in fvg._REQUIRED:
        assert hasattr(module, name), name
    fvg.install(module)
    return module


def _patched_builder():
    module = _fake_module()
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


def test_uniform_plan_module_passes_the_row_count_only_to_the_fa2_planner():
    module = _FakeFA2Module()
    proxy = fvg._UniformPlanModule(module, 2)
    args = (None,) * 11 + (True,) + (None,) * 4 + (-1, False, 0, 0)
    assert proxy.plan(*args) == 'plan-info' and module.plans[-1][-1] == 2 and module.plans[-1][:-1] == args[:-1]
    assert proxy.paged_run() == 'ran'                       # everything else is the real module
    for bad in (args[:-1], args[:11] + (False,) + args[12:], args[:-1] + (3,), args[:-2] + (1, 0)):
        with pytest.raises(RuntimeError, match='plan arguments'):
            proxy.plan(*bad)


def test_uniform_wrapper_swaps_modules_like_the_native_overlay():
    cls = fvg.uniform_wrapper_class(_FakePrefillWrapper)
    wrapper = cls(None, 'NHD', use_cuda_graph=True)
    wrapper.plan_uniform(2)
    real = wrapper._cached_module
    assert isinstance(real, _FakeFA2Module) and real.plans[-1][-1] == 2
    wrapper.plan()                                          # an ordinary plan stays worst case
    assert real.plans[-1][-1] == 0
    # The overlay saves the module, installs its own for one run, then restores it.
    overlay = _FakeFA2Module()
    saved = wrapper._cached_module
    wrapper._cached_module = overlay
    assert wrapper._cached_module is overlay
    wrapper._cached_module = saved
    assert wrapper._cached_module is real
    # A proxy assigned back (saved during a uniform plan) is unwrapped.
    wrapper._kvmem_uniform_rows = 2
    proxied = wrapper._cached_module
    wrapper._kvmem_uniform_rows = 0
    wrapper._cached_module = proxied
    assert wrapper._cached_module is real

    def fail(**kw):
        raise ValueError('plan failed')
    wrapper.plan = fail
    with pytest.raises(ValueError):
        wrapper.plan_uniform(2)
    assert wrapper._kvmem_uniform_rows == 0


def test_installed_flashinfer_plan_abi_is_the_one_patched():
    prefill = pytest.importorskip('flashinfer.prefill')
    fvg._check_plan_abi(prefill.BatchPrefillWithPagedKVCacheWrapper)
    with pytest.raises(RuntimeError, match='ABI changed'):
        fvg._check_plan_abi(_Builder)


class _KVBuilder(_Builder):
    """FlashInferMetadataBuilder state the verification build reads."""
    def _compute_flashinfer_kv_metadata(self, num_blocks, seq_lens, block_table, num_reqs, page_size):
        self.paged_kv_indptr.cpu[0] = 0
        self.paged_kv_indptr.cpu[1:num_reqs + 1] = torch.as_tensor(num_blocks).cumsum(0)
        last = torch.as_tensor(seq_lens) % page_size
        self.paged_kv_last_page_len.cpu[:num_reqs] = torch.where((last == 0) & (torch.as_tensor(seq_lens) != 0),
                                                                 page_size, last)
        return torch.zeros(int(self.paged_kv_indptr.cpu[num_reqs]), dtype=torch.int32)

    def _get_workspace_buffer(self):
        return None


def _kv_builder(limit):
    module = _fake_module(_KVBuilder)
    b = object.__new__(module.FlashInferMetadataBuilder)
    b.calls, b.page_size, b.device = [], 16, 'cpu'
    b._kvmem_verify = dict(limit=limit, graph={}, eager=None)
    b.enable_cuda_graph, b._decode_cudagraph_max_bs, b.kv_cache_layout = True, 64, 'NHD'
    b.paged_kv_indptr = NS(cpu=torch.zeros(33, dtype=torch.int32), gpu=torch.zeros(33, dtype=torch.int32))
    b.paged_kv_last_page_len = NS(cpu=torch.zeros(32, dtype=torch.int32), gpu=torch.zeros(32, dtype=torch.int32))
    b.paged_kv_indices = torch.zeros(1024, dtype=torch.int32)
    b.num_qo_heads, b.num_kv_heads, b.head_dim, b.sm_scale, b.window_left = 12, 2, 256, 1.0, -1
    b.logits_soft_cap, b.q_data_type_decode, b.q_data_type_prefill = None, torch.bfloat16, torch.bfloat16
    b.kv_cache_dtype, b.model_config = torch.float8_e4m3fn, NS(dtype=torch.bfloat16)
    b.prefill_fixed_split_size, b.disable_split_kv = -1, False
    return b


def _batch(lens, actual, seq_lens):
    c = common(lens, actual=actual)
    c.causal, c.seq_lens = True, torch.tensor(seq_lens, dtype=torch.int32)
    c.block_table_tensor, c.slot_mapping = None, None
    return c


def test_full_graph_batch_plans_every_request_at_the_verification_width():
    """3 requests padded to a 4-request graph: the padded request gets 2 query rows
    and no KV, and the FA2 scheduler is told every request has exactly 2 rows."""
    b = _kv_builder(limit=2)
    meta = b.build(0, _batch([2, 2, 2, 0], 8, [40, 33, 18, 0]))
    wrapper = meta.decode.wrapper
    assert b._kvmem_verify['graph'][4] is wrapper and wrapper._max_total_num_rows == 8
    assert wrapper.kwargs['qo_indptr'].tolist() == [0, 2, 4, 6, 8]
    assert wrapper.kwargs['seq_lens'].tolist() == [40, 33, 18, 0]
    assert wrapper._cached_module.plans[-1][-1] == 2
    assert meta.num_decode_tokens == 8 and meta.num_decodes == 4


def test_unpadded_graph_sized_batch_also_plans_uniform_rows():
    b = _kv_builder(limit=2)
    meta = b.build(0, _batch([2, 2], 4, [40, 33]))
    assert meta.decode.wrapper.kwargs['qo_indptr'].tolist() == [0, 2, 4]
    assert meta.decode.wrapper._cached_module.plans[-1][-1] == 2


def test_narrower_than_full_width_keeps_the_worst_case_plan():
    """Only full-width batches replay a FULL graph; others keep the exact qo_indptr."""
    b = _kv_builder(limit=5)
    meta = b.build(0, _batch([3, 3], 6, [40, 33]))
    assert meta.decode.wrapper.kwargs['qo_indptr'].tolist() == [0, 3, 6]
    assert meta.decode.wrapper._cached_module.plans[-1][-1] == 0
