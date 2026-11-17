"""KVMEM_FA2_VERIFY_GRAPH=1: FULL CUDA graphs for speculative verification on
FlashInfer's native FA2 path (GPUs without an XQA/trtllm-gen decode kernel,
such as the A800, sm80).

vLLM 0.30's FlashInfer backend serves a verification batch (every request
scheduling the same 1 + k tokens) as decodes only through XQA/trtllm-gen. On
other GPUs its builder reports UNIFORM_SINGLE_TOKEN_DECODE, and with
speculative decoding vLLM then downgrades the whole model to PIECEWISE graphs:
every attention and GDN core of the target forward and of each draft step runs
eagerly between graph pieces, and the engine thread spends the step launching
kernels.

With this adapter the builder reports UNIFORM_BATCH there. A pure uniform
verification batch (trailing zero-length requests allowed: vLLM pads FULL
graphs that way) is built as decodes served by a causal FA2
BatchPrefillWithPagedKVCacheWrapper in CUDA-graph mode, one per padded request
count, planned into its fixed buffers before every replay; batches outside the
graph range use one eager wrapper of the same kind. Every other batch (prefill
or mixed, ragged decode, single-token decode, cascade) is built by vLLM
unchanged, so the drafts' single-token steps take vLLM's existing FULL decode
graphs.

The rows run the same FA2 prefill kernel family as on the PIECEWISE path, but a
graph-mode plan tiles for the padded batch and always splits the KV, so results
are numerically close, not bitwise equal, to PIECEWISE. Extra GPU memory: the
FULL graphs' memory pool and an 8 MiB plan workspace per captured verification
batch size (user decision 2026-11-16 for the A800).
"""
import importlib.abc
import importlib.machinery
import sys

from ..core import settings as _settings

TARGET = 'vllm.v1.attention.backends.flashinfer'
_REQUIRED = ('FlashInferMetadataBuilder', 'FlashInferMetadata', 'FIDecode', 'BatchPrefillWithPagedKVCacheWrapper',
             'get_flashinfer_layout_string', 'get_seq_lens', 'PIN_MEMORY', 'gpu_sync_allowed')


def enabled():
    value = _settings.get('KVMEM_FA2_VERIFY_GRAPH', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_FA2_VERIFY_GRAPH must be 0 or 1')
    return value == '1'


def _config_eligible(vllm_config):
    """Speculative decoding on a single decode-context rank: the only case vLLM downgrades for."""
    speculative = vllm_config.speculative_config
    return (speculative is not None and (speculative.num_speculative_tokens or 0) > 0 and
            not getattr(speculative, 'parallel_drafting', False) and
            not getattr(speculative, 'enable_adaptive_verification', False) and
            vllm_config.parallel_config.decode_context_parallel_size == 1 and
            not vllm_config.attention_config.use_non_causal)


def max_verify_len(vllm_config):
    return 1 + vllm_config.speculative_config.num_speculative_tokens


def uniform_query_len(common, limit):
    """L when every request schedules L tokens (2 <= L <= limit), except
    trailing zero-length padding; else 0."""
    num_reqs = common.num_reqs
    if num_reqs <= 0 or common.max_query_len < 2 or common.max_query_len > limit:
        return 0
    loc = common.query_start_loc_cpu[:num_reqs + 1]
    lens = (loc[1:] - loc[:-1]).tolist()
    width = lens[0]
    if width < 2:
        return 0
    real = 0
    while real < num_reqs and lens[real] == width:
        real += 1
    if any(n != 0 for n in lens[real:]) or real * width != common.num_actual_tokens:
        return 0
    return width


def install(module):
    """Patch the imported FlashInfer backend module (idempotent)."""
    missing = [name for name in _REQUIRED if not hasattr(module, name)]
    builder = getattr(module, 'FlashInferMetadataBuilder', None)
    for name in ('build', 'get_cudagraph_support', '_compute_flashinfer_kv_metadata', '_get_workspace_buffer'):
        if builder is not None and not hasattr(builder, name):
            missing.append('FlashInferMetadataBuilder.' + name)
    if missing:
        raise RuntimeError('KVMEM_FA2_VERIFY_GRAPH: FlashInfer backend lacks ' + ', '.join(missing))
    if getattr(builder, '_kvmem_fa2_verify_graph', False):
        return
    import torch
    from vllm.v1.attention.backend import AttentionCGSupport

    original_support = builder.get_cudagraph_support.__func__
    original_init = builder.__init__
    original_build = builder.build

    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        support = original_support(cls, vllm_config, kv_cache_spec)
        if support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE and _config_eligible(vllm_config):
            return AttentionCGSupport.UNIFORM_BATCH
        return support

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        original_init(self, kv_cache_spec, layer_names, vllm_config, device)
        self._kvmem_verify = None
        if self.flashinfer_trtllm_api_decode_kernel is not None or not _config_eligible(vllm_config):
            return
        # The builder advertised UNIFORM_BATCH for this configuration, so a
        # case the verification path cannot serve must fail here, not run wrong.
        if self.has_sinks or self.use_dcp or self.is_kvcache_nvfp4:
            raise ValueError('KVMEM_FA2_VERIFY_GRAPH supports FA2 attention without sinks, DCP or NVFP4 KV')
        self._kvmem_verify = dict(limit=max_verify_len(vllm_config), graph={}, eager=None)

    def _verify_wrapper(self, num_reqs, limit):
        state = self._kvmem_verify
        layout = module.get_flashinfer_layout_string(self.kv_cache_layout)
        graph = self.enable_cuda_graph and num_reqs * limit <= self._decode_cudagraph_max_bs
        if not graph:
            if state['eager'] is None:
                state['eager'] = module.BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(), layout, backend='auto')
                state['eager']._kvmem_verify = True
            return state['eager']
        wrapper = state['graph'].get(num_reqs)
        if wrapper is None:
            wrapper = module.BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), layout, use_cuda_graph=True,
                qo_indptr_buf=torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device),
                paged_kv_indptr_buf=self.paged_kv_indptr.gpu[:num_reqs + 1],
                paged_kv_indices_buf=self.paged_kv_indices,
                paged_kv_last_page_len_buf=self.paged_kv_last_page_len.gpu[:num_reqs],
                backend='auto')
            # Plan every replay of this graph for its largest batch, whatever
            # batch is planned first, so the kernel grid never changes.
            wrapper._max_total_num_rows = num_reqs * limit
            wrapper._kvmem_verify = True
            state['graph'][num_reqs] = wrapper
        return wrapper

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        state = getattr(self, '_kvmem_verify', None)
        if state is None or common_prefix_len > 0 or not common_attn_metadata.causal:
            return original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        width = uniform_query_len(common_attn_metadata, state['limit'])
        if not width:
            return original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        common = common_attn_metadata
        num_reqs, page_size = common.num_reqs, self.page_size
        with module.gpu_sync_allowed():
            seq_lens_cpu = common.seq_lens.cpu()
        seq_lens_np = seq_lens_cpu.numpy()
        num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        paged_kv_indices = self._compute_flashinfer_kv_metadata(
            num_blocks_np, seq_lens_np, common.block_table_tensor, num_reqs, page_size)
        qo_indptr_cpu = common.query_start_loc_cpu[:num_reqs + 1]
        paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[:num_reqs + 1]
        last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]
        kv_lens_cpu = module.get_seq_lens(paged_kv_indptr_cpu, last_page_len_cpu, page_size)
        if module.PIN_MEMORY:
            qo_indptr_cpu = qo_indptr_cpu.pin_memory()
            paged_kv_indptr_cpu = paged_kv_indptr_cpu.pin_memory()
            last_page_len_cpu = last_page_len_cpu.pin_memory()
            kv_lens_cpu = kv_lens_cpu.pin_memory()
        wrapper = _verify_wrapper(self, num_reqs, state['limit'])
        wrapper.plan(
            qo_indptr=qo_indptr_cpu, paged_kv_indptr=paged_kv_indptr_cpu, paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=last_page_len_cpu, seq_lens=kv_lens_cpu,
            num_qo_heads=self.num_qo_heads, num_kv_heads=self.num_kv_heads, head_dim_qk=self.head_dim,
            page_size=page_size, causal=True, sm_scale=self.sm_scale, window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap, q_data_type=self.q_data_type_decode,
            kv_data_type=self.kv_cache_dtype, o_data_type=self.model_config.dtype,
            fixed_split_size=self.prefill_fixed_split_size, disable_split_kv=self.disable_split_kv)
        self._kvmem_verify_builds = getattr(self, '_kvmem_verify_builds', 0) + 1
        return module.FlashInferMetadata(
            num_actual_tokens=common.num_actual_tokens, slot_mapping=common.slot_mapping,
            q_data_type_prefill=self.q_data_type_prefill, q_data_type_decode=self.q_data_type_decode,
            num_decodes=num_reqs, num_decode_tokens=common.num_actual_tokens, num_prefills=0,
            num_prefill_tokens=0, causal=True, use_cascade=False, prefill=None,
            decode=module.FIDecode(wrapper=wrapper), cascade_wrapper=None)

    builder.get_cudagraph_support = classmethod(get_cudagraph_support)
    builder.__init__ = __init__
    builder.build = build
    builder._kvmem_fa2_verify_graph = True


class _Loader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        install(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError('Cannot locate the FlashInfer attention backend')
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def enable():
    """Patch the FlashInfer backend now if it is imported, else when it is."""
    if TARGET in sys.modules:
        install(sys.modules[TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
