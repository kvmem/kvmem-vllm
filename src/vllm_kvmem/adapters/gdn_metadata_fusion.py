"""One launch for the GDN metadata of every recurrent cache group (Flash Next).

Flash Next with MTP has 13 QSA and 37 GDN layers, so vLLM's hybrid allocator
makes 50 cache groups of one layer each. Every step build_attn_metadata builds
the first GDN group's metadata and calls ``update_block_table`` for each other
GDN group. Without prefix caching (``mamba_cache_mode`` 'none') that is the
generic path: a worklist copy-and-refresh launch, two fills, a CPU-mask gather
of the group's state indices and six copies into the group's graph buffers,
issued from Python per group. KVMem schedules synchronously, so these ~36
sequences of small launches (about 10 ms per decode step at 16 requests) run
while the GPU waits.

For pure speculative-decode batches (every request verifies its drafts; no
prefill, no plain decode, no prefill checkpoints) this module returns the same
metadata object per group as the vendor path and writes the same bytes into
the same buffers, but issues the device writes of all groups as one Triton
launch when build_attn_metadata returns. Other batches, graph capture, and
builds outside the main forward's build_attn_metadata (the MTP drafter) keep
the vendor path. The GPU test compares every buffer bitwise with the vendor.
"""
import sys
import threading
from dataclasses import replace

import torch

MODEL_STATE_MODULE = 'vllm.models.qwen4_exp.nvidia.model_state'
GDN_MODULE = 'vllm.v1.attention.backends.gdn_attn'

_STATE = threading.local()
_INSTALLED = {}
_KERNEL = {}
STATS = dict(fused_steps=0, fused_groups=0, vendor_groups=0)

# Per-group pointer table columns.
_WORKLISTS = 14
(P_STATE, P_SPEC_STATE, P_CHECKPOINT, P_TABLE, P_LIVE, P_SPEC_TENSOR, P_MASKS, P_NON_SPEC_TOK, P_SPEC_TOK,
 P_SPEC_QSL, P_ACCEPTED) = range(_WORKLISTS, _WORKLISTS + 11)
GROUP_COLUMNS = _WORKLISTS + 11
# Shared source table columns.
(S_REQUEST_ROWS, S_SPEC_ROWS, S_CHECKPOINT_COLUMNS, S_CHECKPOINT_OFFSETS, S_MASKS, S_NON_SPEC_TOK, S_SPEC_TOK,
 S_SPEC_QSL, S_ACCEPTED) = range(_WORKLISTS, _WORKLISTS + 9)
SOURCE_COLUMNS = _WORKLISTS + 9


def _kernel():
    found = _KERNEL.get('fused')
    if found is not None:
        return found
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=[
        'num_non_spec', 'num_spec', 'num_reqs', 'live0', 'live1', 'mask_bytes', 'non_spec_tok_bytes',
        'spec_tok_bytes', 'spec_qsl_bytes', 'accepted_bytes', 'state_stride', 'state_column_stride',
        'table_stride', 'table_column_stride'])
    def _fused_gdn_group_update(
        groups, sources,
        num_non_spec, num_spec, num_reqs, live0, live1,
        mask_bytes, non_spec_tok_bytes, spec_tok_bytes, spec_qsl_bytes, accepted_bytes,
        state_stride, state_column_stride, table_stride, table_column_stride,
        WORKLIST_BYTES: tl.constexpr, GROUP_COLUMNS: tl.constexpr, MAX_SEQS: tl.constexpr,
        STATE_COLUMNS: tl.constexpr, SPEC_COLUMNS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        group = groups + tl.program_id(1) * GROUP_COLUMNS
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        # Worklists: byte copies from the source group's mixed metadata.
        for field in tl.static_range(len(WORKLIST_BYTES)):
            source = tl.load(sources + field).to(tl.pointer_type(tl.uint8))
            target = tl.load(group + field).to(tl.pointer_type(tl.uint8))
            inside = index < WORKLIST_BYTES[field]
            tl.store(target + index, tl.load(source + index, inside, other=0), inside)
        # State refresh: _copy_worklists_and_refresh_states with this group's table.
        states = tl.load(group + 17).to(tl.pointer_type(tl.int32))
        request_rows = tl.load(sources + 14).to(tl.pointer_type(tl.int64))
        spec_rows = tl.load(sources + 15).to(tl.pointer_type(tl.int64))
        checkpoint_columns = tl.load(sources + 16).to(tl.pointer_type(tl.int64))
        checkpoint_offsets = tl.load(sources + 17).to(tl.pointer_type(tl.int32))
        out_states = tl.load(group + 14).to(tl.pointer_type(tl.int32))
        out_spec_states = tl.load(group + 15).to(tl.pointer_type(tl.int32))
        out_checkpoint = tl.load(group + 16).to(tl.pointer_type(tl.int32))
        row = tl.load(request_rows + index, index < num_non_spec, other=0).to(tl.int64)
        state = tl.load(states + row * state_stride, index < num_non_spec, other=0)
        tl.store(out_states + index, state, index < MAX_SEQS)
        column = tl.load(checkpoint_columns + index, index < num_non_spec, other=0).to(tl.int64)
        offset = tl.load(checkpoint_offsets + index, index < num_non_spec, other=0)
        checkpoint = tl.load(states + row * table_stride + column * table_column_stride,
                             (index < num_non_spec) & (offset > 0), other=0)
        tl.store(out_checkpoint + index, checkpoint, index < MAX_SEQS)
        spec_row = index // STATE_COLUMNS
        spec_column = index % STATE_COLUMNS
        srow = tl.load(spec_rows + spec_row, spec_row < num_spec, other=0).to(tl.int64)
        sstate = tl.load(states + srow * state_stride + spec_column * state_column_stride,
                         spec_row < num_spec, other=0)
        tl.store(out_spec_states + index, sstate, spec_row < MAX_SEQS)
        # Prefill live counts.
        live = tl.load(group + 18).to(tl.pointer_type(tl.int32))
        tl.store(live + index, tl.where(index == 0, live0, live1).to(tl.int32), index < 2)
        # Graph-owned speculative state indices: rows [0, num_reqs) of the table.
        spec_tensor = tl.load(group + 19).to(tl.pointer_type(tl.int32))
        srow2 = (index // SPEC_COLUMNS).to(tl.int64)
        scol2 = (index % SPEC_COLUMNS).to(tl.int64)
        inside = index < num_reqs * SPEC_COLUMNS
        value = tl.load(states + srow2 * table_stride + scol2 * table_column_stride, inside, other=0)
        tl.store(spec_tensor + index, value, inside)
        # Batch fields shared by every group: byte copies from the first group's buffers.
        for pair in tl.static_range(5):
            source = tl.load(sources + 18 + pair).to(tl.pointer_type(tl.uint8))
            target = tl.load(group + 20 + pair).to(tl.pointer_type(tl.uint8))
            if pair == 0:
                limit = mask_bytes
            elif pair == 1:
                limit = non_spec_tok_bytes
            elif pair == 2:
                limit = spec_tok_bytes
            elif pair == 3:
                limit = spec_qsl_bytes
            else:
                limit = accepted_bytes
            inside = index < limit
            tl.store(target + index, tl.load(source + index, inside, other=0), inside)

    _KERNEL['fused'] = _fused_gdn_group_update
    return _fused_gdn_group_update


def _all_spec(metadata):
    cache = getattr(_STATE, 'mask_cache', None)
    if cache is None or cache[0] is not metadata:
        mask = metadata.spec_sequence_masks_cpu
        # The vendor gathers the table rows the mask selects: all of them, in order.
        result = mask is not None and int(mask.shape[0]) == metadata.num_reqs and bool(mask.all())
        cache = _STATE.mask_cache = (metadata, result)
    return cache[1]


def eligible(builder, metadata, block_table):
    mixed = getattr(builder, '_b12x_mixed', None)
    source = metadata.b12x_mixed
    if mixed is None or source is None or source is mixed or metadata.is_uniform_spec_decode:
        return False
    if getattr(builder.vllm_config.cache_config, 'mamba_cache_mode', 'none') not in ('all', 'none'):
        return False
    if getattr(builder, '_b12x_prefill_live_counts', None) is None or not mixed.state_indices.is_cuda:
        return False
    n = metadata.num_reqs
    if not (metadata.num_prefills == 0 and metadata.num_decodes == 0 and metadata.num_spec_decodes == n > 0
            and builder.use_full_cuda_graph and metadata.prefill_checkpoint is None
            and metadata.num_spec_decodes <= builder.decode_cudagraph_max_bs
            and metadata.num_spec_decode_tokens <= builder.decode_cudagraph_max_bs):
        return False
    if (mixed.max_tokens, mixed.max_seqs, mixed.state_columns) != (source.max_tokens, source.max_seqs,
                                                                  source.state_columns):
        return False
    fields = (('spec_sequence_masks', builder.spec_sequence_masks), ('spec_token_indx', builder.spec_token_indx),
              ('non_spec_token_indx', builder.non_spec_token_indx),
              ('spec_query_start_loc', builder.spec_query_start_loc),
              ('num_accepted_tokens', builder.num_accepted_tokens))
    for name, target in fields:
        value = getattr(metadata, name)
        if value is None or value.dtype != target.dtype or not value.is_contiguous():
            return False
    spec = builder.spec_state_indices_tensor
    if (block_table.dtype != torch.int32 or block_table.dim() != 2 or int(block_table.shape[0]) != n
            or int(block_table.shape[1]) < builder.state_index_columns or spec.dtype != torch.int32
            or not spec.is_contiguous() or int(spec.shape[1]) != builder.state_index_columns):
        return False
    return _all_spec(metadata)


def _views(builder, metadata):
    key = (metadata.num_reqs, int(metadata.non_spec_token_indx.shape[0]), int(metadata.spec_token_indx.shape[0]))
    cache = builder.__dict__.setdefault('_kvmem_fused_views', {})
    found = cache.get(key)
    if found is None:
        n, l1, l2 = key
        found = cache[key] = (builder.spec_state_indices_tensor[:n], builder.spec_sequence_masks[:n],
                              builder.spec_token_indx[:l2], builder.non_spec_token_indx[:l1],
                              builder.spec_query_start_loc[:n + 1], builder.num_accepted_tokens[:n])
    return found


def defer(builder, metadata, block_table):
    """The vendor generic path's result for one group; its device writes are queued."""
    mixed, source = builder._b12x_mixed, metadata.b12x_mixed
    mixed._num_non_spec = source._num_non_spec
    mixed._num_spec = source._num_spec
    spec, masks, spec_tok, non_spec_tok, spec_qsl, accepted = _views(builder, metadata)
    _STATE.pending.append((builder, metadata, block_table))
    result = replace(metadata, b12x_mixed=mixed, b12x_prefill_live_counts=builder._b12x_prefill_live_counts,
                   spec_state_indices_tensor=spec, non_spec_state_indices_tensor=block_table.new_empty((0,)),
                   prefill_state_indices=metadata.prefill_state_indices, spec_sequence_masks=masks,
                   spec_token_indx=spec_tok, non_spec_token_indx=non_spec_tok, spec_query_start_loc=spec_qsl,
                   non_spec_query_start_loc=metadata.non_spec_query_start_loc, num_accepted_tokens=accepted,
                   prefill_checkpoint=None)
    if not hasattr(_STATE, 'results'):
        _STATE.results = []
    _STATE.results.append(result)
    return result


def _group_row(builder, block_table):
    mixed = builder._b12x_mixed
    return ([value.data_ptr() for value in mixed._worklists] +
            [mixed.state_indices.data_ptr(), mixed.spec_state_indices.data_ptr(),
             mixed.checkpoint.state_indices.data_ptr(), block_table.data_ptr(),
             builder._b12x_prefill_live_counts.data_ptr(), builder.spec_state_indices_tensor.data_ptr(),
             builder.spec_sequence_masks.data_ptr(), builder.non_spec_token_indx.data_ptr(),
             builder.spec_token_indx.data_ptr(), builder.spec_query_start_loc.data_ptr(),
             builder.num_accepted_tokens.data_ptr()])


def _source_row(metadata):
    source = metadata.b12x_mixed
    return ([value.data_ptr() for value in source._worklists] +
            [source.request_rows.data_ptr(), source.spec_request_rows.data_ptr(),
             source.checkpoint_columns.data_ptr(), source.checkpoint.checkpoint_offsets.data_ptr(),
             metadata.spec_sequence_masks.data_ptr(), metadata.non_spec_token_indx.data_ptr(),
             metadata.spec_token_indx.data_ptr(), metadata.spec_query_start_loc.data_ptr(),
             metadata.num_accepted_tokens.data_ptr()])


def _table(cache_key, rows, device):
    cache = _INSTALLED.setdefault('tables', {})
    found = cache.get(cache_key)
    if found is None:
        if len(cache) > 64:
            cache.clear()
        found = cache[cache_key] = torch.tensor(rows, dtype=torch.int64, device=device)
    return found


def _check_enabled():
    from ..core import settings as _settings
    return _settings.get('KVMEM_GDN_FUSION_CHECK', '0') == '1'


def _snapshot(builder):
    mixed = builder._b12x_mixed
    out = {k: v.clone() for k, v in builder.__dict__.items() if isinstance(v, torch.Tensor)}
    out.update({'mixed.' + k: v.clone() for k, v in mixed.__dict__.items() if isinstance(v, torch.Tensor)})
    out['mixed.checkpoint.state_indices'] = mixed.checkpoint.state_indices.clone()
    return out


def _check(pending, results):
    """Diagnostic: rerun vLLM's update on each fused group and compare every buffer and field."""
    update = _INSTALLED['update']
    for (builder, metadata, block_table), fused in zip(pending, results):
        before = _snapshot(builder)
        vendor = update(builder, metadata, block_table, None)
        after = _snapshot(builder)
        bad = [k for k in before if not torch.equal(before[k], after[k])]
        for field in vendor.__dataclass_fields__:
            x, y = getattr(vendor, field), getattr(fused, field)
            if isinstance(x, torch.Tensor):
                if not (isinstance(y, torch.Tensor) and x.shape == y.shape and x.dtype == y.dtype and
                        torch.equal(x, y)):
                    bad.append('field.' + field)
            elif field not in ('b12x_mixed',) and type(x) in (int, bool, type(None)) and x != y:
                bad.append('field.' + field)
        STATS['checked_groups'] = STATS.get('checked_groups', 0) + 1
        if bad:
            STATS['mismatched_groups'] = STATS.get('mismatched_groups', 0) + 1
            mismatches = STATS.setdefault('mismatch_fields', {})
            for k in bad:
                mismatches[k] = mismatches.get(k, 0) + 1


def flush():
    """Issue the queued groups' device writes: one launch per (source, table strides)."""
    pending = getattr(_STATE, 'pending', None)
    if not pending:
        return
    _STATE.pending = []
    results = getattr(_STATE, 'results', [])
    _STATE.results = []
    batches = {}
    for builder, metadata, block_table in pending:
        key = (id(metadata), tuple(block_table.stride()), builder.state_index_columns)
        batches.setdefault(key, (metadata, []))[1].append((builder, block_table))
    kernel = _kernel()
    for (_, strides, spec_columns), (metadata, groups) in batches.items():
        source = metadata.b12x_mixed
        group_key = tuple((id(builder), block_table.data_ptr()) for builder, block_table in groups)
        table = _table(('groups', group_key), [_group_row(b, t) for b, t in groups], source.state_indices.device)
        source_key = ('source', id(source), metadata.spec_sequence_masks.data_ptr(),
                      metadata.non_spec_token_indx.data_ptr(), metadata.spec_token_indx.data_ptr(),
                      metadata.spec_query_start_loc.data_ptr(), metadata.num_accepted_tokens.data_ptr())
        sources = _table(source_key, [_source_row(metadata)], source.state_indices.device)
        n = metadata.num_reqs
        worklist_bytes = tuple(value.numel() * value.element_size() for value in source._worklists)
        sizes = (max(worklist_bytes), source.max_seqs * source.state_columns, source.max_seqs,
                 n * spec_columns, 4 * int(metadata.spec_token_indx.shape[0]),
                 4 * int(metadata.non_spec_token_indx.shape[0]), 4 * (n + 1))
        block = 1024
        grid = (-(-max(sizes) // block), len(groups))
        kernel[grid](
            table, sources, source._num_non_spec, source._num_spec, n, metadata.num_prefills,
            metadata.num_prefill_tokens,
            n * metadata.spec_sequence_masks.element_size(),
            int(metadata.non_spec_token_indx.shape[0]) * metadata.non_spec_token_indx.element_size(),
            int(metadata.spec_token_indx.shape[0]) * metadata.spec_token_indx.element_size(),
            (n + 1) * metadata.spec_query_start_loc.element_size(),
            n * metadata.num_accepted_tokens.element_size(),
            strides[0], strides[1], strides[0], strides[1],
            WORKLIST_BYTES=worklist_bytes, GROUP_COLUMNS=GROUP_COLUMNS, MAX_SEQS=source.max_seqs,
            STATE_COLUMNS=source.state_columns, SPEC_COLUMNS=spec_columns, BLOCK=block)
        STATS['fused_steps'] += 1
        STATS['fused_groups'] += len(groups)
    if _check_enabled():
        _check(pending, results)
        if STATS['fused_steps'] % 2000 == 1 or (STATS['fused_steps'] % 100 == 0 and _check_enabled()):
            import logging
            logging.getLogger('vllm_kvmem').warning('gdn metadata fusion: %s', dict(STATS))


def install():
    """Patch GDN update_block_table and wrap the main forward's build_attn_metadata."""
    if _INSTALLED.get('done'):
        return True
    import importlib
    try:
        gdn = importlib.import_module(GDN_MODULE)
        state = sys.modules.get(MODEL_STATE_MODULE) or importlib.import_module(MODEL_STATE_MODULE)
        builder_cls = gdn.GDNAttentionMetadataBuilder
        state.build_attn_metadata
    except (ImportError, AttributeError):
        return False   # not the Flash Next vendor runtime: keep vLLM's per-group path
    original_update = builder_cls.update_block_table
    original_build = state.build_attn_metadata

    def update_block_table(self, metadata, blk_table, slot_mapping):
        if getattr(_STATE, 'active', False) and eligible(self, metadata, blk_table):
            return defer(self, metadata, blk_table)
        flush()
        STATS['vendor_groups'] += 1
        if STATS['vendor_groups'] % 20000 == 1:
            import logging
            logging.getLogger('vllm_kvmem').warning('gdn metadata fusion (vendor path): %s', dict(STATS))
        return original_update(self, metadata, blk_table, slot_mapping)

    def build_attn_metadata(*args, **kwargs):
        if kwargs.get('for_cudagraph_capture') or torch.cuda.is_current_stream_capturing():
            return original_build(*args, **kwargs)
        _STATE.active, _STATE.pending, _STATE.mask_cache, _STATE.results = True, [], None, []
        try:
            result = original_build(*args, **kwargs)
            flush()
            return result
        finally:
            _STATE.active, _STATE.pending, _STATE.mask_cache, _STATE.results = False, [], None, []

    builder_cls.update_block_table = update_block_table
    state.build_attn_metadata = build_attn_metadata
    _INSTALLED.update(done=True, update=original_update, build=original_build)
    return True
