"""Flash Next QSA: selector keys in pages carved from the KV pool (qsa_split_selector).

The B12X kernels take the main K/V cache and the compressed selector cache as
independent operands with independent block tables. The vendor integration ties
them: the selector cache is a view of each main page's tail and one block table
feeds both. This module unties them without changing any kernel or any value
the kernels compute:

- ``qsa_compressed_cache_view`` returns, while a QSA layer binds the real pool,
  a dense ``[P, 752, 128]`` BF16 view over the layer's share of the carved
  native blocks (core/compressed_pages.py) instead of the page tails;
- ``_ensure_qsa_context_buffers`` gives every context its own compressed table;
- ``_stage_runtime_metadata`` fills each batch row's compressed table from the
  request's state slot, after the vendor fills the main table from vLLM's;
- ``QSAWindowAssembler`` rebuilds selector keys through the same slot rows.

Selector keys then depend only on the request's frame, never on which main
pages are resident, which is what streaming (P2) needs. With everything still
resident the selected tokens, K/V bytes and outputs are unchanged.
"""
import contextvars
import importlib
from types import MethodType

from ..core.compressed_pages import frame_tokens, place_carves, plan_carve, plan_streaming_carve

VENDOR_MODULE = 'vllm.models.qwen4_exp.nvidia.b12x_qsa'
INDEX_ELEMENT_BYTES = 2  # BF16 selector keys

_BINDING = contextvars.ContextVar('kvmem_qsa_selector_binding', default=None)


def plan_qsa_carves(vllm_config, kv_cache_config, options, draft_tokens):
    """Placed (selector, streaming or None, main_end): the scheduler's view; SelectorState derives the same."""
    selector = selector_carve_for(vllm_config, kv_cache_config, options, draft_tokens)
    streaming = (plan_streaming_carve(selector, hot_pages=options.qsa_hot_pages,
                                      max_rows=vllm_config.scheduler_config.max_num_batched_tokens)
                 if getattr(options, 'qsa_streaming', False) else None)
    return place_carves(selector, streaming, kv_cache_config.num_blocks)


def selector_carve_for(vllm_config, kv_cache_config, options, draft_tokens):
    """The unplaced selector carve geometry (plan_qsa_carves places it)."""
    from .cache_groups import full_attention_groups
    groups = full_attention_groups(kv_cache_config)
    page_tokens = {kv_cache_config.kv_cache_groups[i].kv_cache_spec.block_size for i in groups}
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    if len(page_tokens) != 1 or len(sizes) != 1:
        raise ValueError('QSA selector pages require one attention page size and one native backing allocation')
    native_page_bytes, remainder = divmod(sizes.pop(), kv_cache_config.num_blocks)
    if remainder:
        raise ValueError('QSA selector pages require integral native pages')
    text = vllm_config.model_config.hf_text_config
    return plan_carve(layers=len(groups), slots=vllm_config.scheduler_config.max_num_seqs,
                      frame_tokens=frame_tokens(options.memory, draft_tokens), page_tokens=page_tokens.pop(),
                      native_page_bytes=native_page_bytes, compress_ratio=int(text.indexer_compress_ratio),
                      index_head_dim=int(text.indexer_head_dim), index_element_bytes=INDEX_ELEMENT_BYTES)


class SelectorState:
    """Worker side of the carve: per-layer selector views and the slot table."""

    def __init__(self, layers, frame, slots, streaming=None):
        self.layer_count = layers
        self.frame = frame
        self.slots = slots
        self.streaming = streaming          # (hot_pages, max_rows) with qsa_streaming
        self.carve = None
        self.streaming_carve = None
        self.main_end = None                # QSA main K/V views stop here (first selector block)
        self.views = {}
        self.slot_table = None
        self.pool = None                    # (storage pointer, blocks) of the bound pool

    def carved_view(self, index, layer, kv_cache, compress_ratio, index_head_dim):
        """The layer's dense selector view, or None for a cache that is not the native pool."""
        import torch
        page_tokens = int(kv_cache.shape[2])
        native_page_bytes = int(kv_cache.stride(0)) * kv_cache.element_size()
        carve = plan_carve(layers=self.layer_count, slots=self.slots, frame_tokens=self.frame,
                           page_tokens=page_tokens, native_page_bytes=native_page_bytes,
                           compress_ratio=compress_ratio, index_head_dim=index_head_dim,
                           index_element_bytes=INDEX_ELEMENT_BYTES)
        streaming = (plan_streaming_carve(carve, hot_pages=self.streaming[0], max_rows=self.streaming[1])
                     if self.streaming is not None else None)
        tail = carve.blocks + (streaming.blocks if streaming is not None else 0)
        if int(kv_cache.shape[0]) < tail + 2:
            # CUDA-graph memory profiling binds a small throwaway cache.
            return None
        carve, streaming, main_end = place_carves(carve, streaming, int(kv_cache.shape[0]))
        pool = (kv_cache.untyped_storage().data_ptr(), int(kv_cache.shape[0]))
        if self.carve is not None and pool != self.pool:
            # A later binding of a different pool (vLLM binds more than once at
            # startup): placement depends on the pool size, so the last one wins.
            self.carve = self.streaming_carve = self.main_end = self.slot_table = None
            self.views = {}
        if self.carve is None:
            self.carve, self.streaming_carve, self.main_end, self.pool = carve, streaming, main_end, pool
        elif carve != self.carve or streaming != self.streaming_carve:
            raise ValueError('QSA layers disagree on the selector carve geometry')
        if layer.max_seqs != carve.slots:
            raise ValueError('QSA state slots differ from max_num_seqs')
        storage = kv_cache.untyped_storage()
        start = (kv_cache.storage_offset() * kv_cache.element_size() +
                 carve.layer_first_block(index) * carve.native_page_bytes)
        length = carve.blocks_per_layer * carve.native_page_bytes
        if start + length > storage.nbytes():
            raise ValueError('QSA selector carve exceeds the native pool')
        raw = torch.empty(0, dtype=torch.uint8, device=kv_cache.device)
        raw.set_(storage, start, (length,), (1,))
        pages = carve.blocks_per_layer * carve.per_native_page
        view = raw.view(torch.bfloat16).view(pages, carve.page_tokens // compress_ratio, index_head_dim)
        # The vendor sets max_seq_len before taking the view; its compressed
        # table width is ceil(max_seq_len rounded up to a group / page tokens).
        width = -(-(-(-int(layer.max_seq_len) // compress_ratio) * compress_ratio) // carve.page_tokens)
        if self.slot_table is None:
            # Built at bind time: graph capture later only indexes it.
            rows = [carve.slot_row(slot, width) for slot in range(carve.slots)]
            self.slot_table = torch.tensor(rows, dtype=torch.int32, device=kv_cache.device)
        elif self.slot_table.shape[1] != width:
            raise ValueError('QSA layers disagree on the compressed table width')
        self.views[index] = view
        return view

    def table(self, width, device):
        """Slot rows cut to a context's table width (prefill contexts plan narrower tables)."""
        if self.slot_table is None or self.slot_table.shape[1] < width or self.slot_table.device != device:
            raise RuntimeError('QSA selector slot table does not match the staged compressed table')
        return self.slot_table[:, :width]

    def slot_row(self, slot):
        """[1, width] compressed table row of one state slot, for selector rebuilds."""
        if self.slot_table is None:
            raise RuntimeError('QSA selector table is not bound')
        return self.slot_table[slot:slot + 1]


def install_qsa_split_selector(runner, options):
    if not getattr(options, 'qsa_split_selector', False):
        return None
    from .engine_api import engine_root
    from .qsa_contract import check_qsa_split_sources
    check_qsa_split_sources(engine_root())
    vendor = importlib.import_module(VENDOR_MODULE)
    layers = list(runner._kvmem_qsa_layers.items())
    draft = int(getattr(runner.vllm_config.speculative_config, 'num_speculative_tokens', 0) or 0)
    streaming = ((options.qsa_hot_pages, runner.vllm_config.scheduler_config.max_num_batched_tokens)
                 if getattr(options, 'qsa_streaming', False) else None)
    state = SelectorState(len(layers), frame_tokens(options.memory, draft),
                          runner.vllm_config.scheduler_config.max_num_seqs, streaming)
    original_view = vendor.qsa_compressed_cache_view

    def view(kv_cache, *, compress_ratio, index_head_dim):
        active = _BINDING.get()
        result = None
        if active is not None:
            result = state.carved_view(*active, kv_cache, compress_ratio, index_head_dim)
        if result is None:
            return original_view(kv_cache, compress_ratio=compress_ratio, index_head_dim=index_head_dim)
        return result

    vendor.qsa_compressed_cache_view = view
    runtime = None
    if getattr(options, 'qsa_streaming', False):
        from .qsa_split import install_split
        from .qsa_stream_runtime import StreamingRuntime
        runtime = StreamingRuntime(runner, options, state)
        install_split(runner).append(runtime)
        runner._kvmem_qsa_stream = runtime
    for index, (_, layer) in enumerate(layers):
        _patch_layer(layer, index, state, runtime)
    runner._kvmem_qsa_selector = state
    return state


def bind_qsa_streaming(runner):
    """After the real KV pool is bound: carve maps/staging, pin the host pool, wrap draft reuse."""
    runtime = getattr(runner, '_kvmem_qsa_stream', None)
    if runtime is None or runtime.carve is not None:
        return runtime
    runtime.bind()
    vendor = importlib.import_module(VENDOR_MODULE)
    api = vendor.get_b12x_qsa()
    if api is None:
        raise RuntimeError('QSA streaming needs the bound B12X QSA API')
    from .qsa_stream_runtime import install_reuse_wrapper
    install_reuse_wrapper(runtime, getattr(api, '_api', api))
    # Staging steps may replay graphs: the captured hook carries the device-side
    # fill (install_eager_staging remains available to force them eager).
    return runtime


def _patch_layer(layer, index, state, runtime=None):
    bind = layer.bind_kv_cache
    ensure = layer._ensure_qsa_context_buffers
    stage = layer._stage_runtime_metadata
    bind_context = layer._bind_qsa_context

    def bind_qsa_context(self, *args, **kwargs):
        # B12X main K/V views end before the carved selector pages it may write.
        full, end = self.kv_cache, state.main_end
        if end is None or index not in state.views or int(full.shape[0]) <= end:
            return bind_context(*args, **kwargs)
        self.kv_cache = full[:end]
        try:
            return bind_context(*args, **kwargs)
        finally:
            self.kv_cache = full

    def bind_kv_cache(self, kv_cache):
        token = _BINDING.set((index, self))
        try:
            return bind(kv_cache)
        finally:
            _BINDING.reset(token)

    def ensure_context_buffers(self, context):
        result = ensure(context)
        if (index in state.views and context.compressed_block_table is not None and
                context.compressed_block_table is context.main_block_table):
            context.compressed_block_table = context.main_block_table.clone().fill_(-1)
        return result

    def stage_runtime_metadata(self, metadata, rows, *, row_capacity, main_block_table=None,
                               compressed_block_table=None):
        staged = stage(metadata, rows, row_capacity=row_capacity, main_block_table=main_block_table,
                       compressed_block_table=compressed_block_table)
        if (index in state.views and compressed_block_table is not None and
                compressed_block_table is not main_block_table):
            _, slots, _, _ = self._require_qsa_metadata(metadata)
            requests = int(metadata.seq_lens.shape[0])
            table = state.table(int(compressed_block_table.shape[1]), compressed_block_table.device)
            compressed_block_table.fill_(-1)
            if requests:
                compressed_block_table[:requests].copy_(
                    table.index_select(0, slots[:requests].long().clamp(min=0)))
            if runtime is not None and runtime.carve is not None:
                main = main_block_table if main_block_table is not None else self._main_block_table
                # skip_topk: an MTP draft round reusing anchor selections (_run_reused_qsa).
                runtime.rewrite_main_table(index, metadata, main, slots, requests,
                                           reuse=bool(getattr(self, 'skip_topk', False)))
        return staged

    layer.bind_kv_cache = MethodType(bind_kv_cache, layer)
    layer._bind_qsa_context = MethodType(bind_qsa_context, layer)
    layer._ensure_qsa_context_buffers = MethodType(ensure_context_buffers, layer)
    layer._stage_runtime_metadata = MethodType(stage_runtime_metadata, layer)
