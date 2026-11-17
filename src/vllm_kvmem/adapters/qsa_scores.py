"""QSA attention logits straight from the B12X sparse attention kernel.

KVMem's per-layer pressure (serving/qsa_pressure.py) needs, for a few sampled
prefill rows per layer, the softmax QSA's attention gives each selected token.
The B12X selected-position kernel (attention/paged/_selected_forward.py)
computes exactly the logits behind it: for every tile of selected positions,
QK^T into an FP32 accumulator, masked to -inf where the position is invalid or
in the future, then the online softmax. This module builds a variant of that
kernel that also stores the masked FP32 accumulator of the rows a caller marks
(``score_rows[row] >= 0``) into ``scores[slot, q_head, column]``.

``sampled_logits`` launches the variant on the sampled rows alone (gathered
query, selected positions and positions; output discarded) next to the layer's
own vendor attention, which runs untouched. Each logit comes from the
same QK MMA over the same inputs, so it is the logit the layer's attention
normalized. (Storing the rows inside the layer's own launch is as fast on the
GPU, but its per-call row map and launch wrapper stalled the server's host.)

The variant is generated from the installed vendor source at first use: six
anchored insertions (two pointer parameters, their forwarding, the per-row slot
load and the store before ``online_softmax``) plus a distinct compile-cache
name, written to a file and imported as a sibling module, so CuTe's
source-based tracing works and nothing of the vendor kernel is vendored here.
The insertions leave every arithmetic instruction of the attention unchanged
(the GPU test checks its output is bitwise equal to the vendor launch). An
anchor that no longer matches or a frozen kernel resolution raises; the caller
then recomputes the logits instead.

Probabilities follow as softmax(scores * head_dim**-0.5 * k_descale) over each
row's columns: the kernel's own normalization (it folds the same scale and
k_descale into exp2, LOG2_E included).
"""
import hashlib
import importlib
import importlib.util
import inspect
import os
import re
import sys
import tempfile
import threading

import torch

VENDOR_MODULE = 'vllm.models.qwen4_exp.nvidia.b12x_qsa'

_STORE = '''\
            if score_slot >= Int32(0):
                for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                    for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                        coordinate = score_coordinates[m, n]
                        score_column = tile * Int32(self.tile_n) + coordinate[1]
                        if (coordinate[0] < Int32(self.heads_per_kv)) & (
                            score_column < Int32(self.selection_width)
                        ):
                            score_offset = (
                                score_slot.to(Int64) * Int64(self.q_heads)
                                + (kv_head * Int32(self.heads_per_kv) + coordinate[0]).to(Int64)
                            ) * Int64(self.selection_width) + score_column.to(Int64)
                            scores[score_offset] = score_mn[m, n]
'''

# (anchor, replacement, occurrences); a regex anchor is a compiled pattern.
_PATCHES = (
    ("        output: cute.Pointer,\n        num_cache_pages: Int64,\n",
     "        output: cute.Pointer,\n        score_rows: cute.Pointer,\n        scores: cute.Pointer,\n"
     "        num_cache_pages: Int64,\n", 2),
    ("            output,\n            num_cache_pages,\n            table_batch,\n",
     "            output,\n            score_rows,\n            scores,\n            num_cache_pages,\n"
     "            table_batch,\n", 1),
    ("        request_valid = (request_id >= Int64(0)) & (request_id < table_batch)\n",
     "        request_valid = (request_id >= Int64(0)) & (request_id < table_batch)\n"
     "        score_slot = Int32(score_rows[row])\n", 1),
    (re.compile(r"( +)softmax,\n( +)is_first="), r"\1softmax,\1scores,\1score_slot,\n\2is_first=", 2),
    ("        softmax: Softmax,\n        is_first: cutlass.Constexpr,\n",
     "        softmax: Softmax,\n        scores: cute.Pointer,\n        score_slot: Int32,\n"
     "        is_first: cutlass.Constexpr,\n", 1),
    ("            row_scale = softmax.online_softmax(\n",
     _STORE + "            row_scale = softmax.online_softmax(\n", 1),
    ('"attention.paged.selected_forward"', '"kvmem.qsa_scores.selected_forward"', 1),
    ("                _fake_pointer(BFloat16),\n                Int64(1),\n",
     "                _fake_pointer(BFloat16),\n                _fake_pointer(Int32),\n"
     "                _fake_pointer(Float32),\n                Int64(1),\n", 1),
)

_LOCK = threading.Lock()
_STATE = {}            # 'module' | 'error' -> ...


def patch_source(source):
    """The vendor kernel source with the score store; ValueError if an anchor moved."""
    for anchor, replacement, count in _PATCHES:
        if isinstance(anchor, re.Pattern):
            found = len(anchor.findall(source))
            if found != count:
                raise ValueError(f'QSA score anchor {anchor.pattern!r} matched {found} times, expected {count}')
            source = anchor.sub(replacement, source)
        else:
            found = source.count(anchor)
            if found != count:
                raise ValueError(f'QSA score anchor {anchor[:48]!r} matched {found} times, expected {count}')
            source = source.replace(anchor, replacement)
    return source


def _vendor_modules():
    """(sparse launch module, selected-forward module) as the running vendor code imports them."""
    vendor = sys.modules.get(VENDOR_MODULE) or importlib.import_module(VENDOR_MODULE)
    api = vendor.get_b12x_qsa()
    api = getattr(api, '_api', api)
    if api is None:
        raise RuntimeError('B12X QSA API is not available')
    package = sys.modules[api.run.__module__].__package__              # ...attention.qsa
    sparse = importlib.import_module(package + '._sparse_gqa')
    forward = importlib.import_module(package.rsplit('.', 1)[0] + '.paged._selected_forward')
    return sparse, forward


def load_module(forward=None):
    """The patched selected-forward module (cached); raises if it cannot be built."""
    with _LOCK:
        module = _STATE.get('module')
        if module is not None:
            return module
        if 'error' in _STATE:
            raise RuntimeError(_STATE['error'])
        try:
            if forward is None:
                _, forward = _vendor_modules()
            source = patch_source(inspect.getsource(forward))
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            directory = os.path.join(tempfile.gettempdir(), 'kvmem-qsa-scores')
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f'selected_forward_scores_{digest}.py')
            if not os.path.exists(path):
                fd, partial = tempfile.mkstemp(dir=directory, suffix='.py')
                with os.fdopen(fd, 'w') as handle:
                    handle.write(source)
                os.replace(partial, path)
            name = forward.__package__ + '._kvmem_selected_forward_scores'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as error:   # noqa: BLE001 - reported, the caller recomputes
            _STATE['error'] = f'{type(error).__name__}: {error}'
            raise
        _STATE['module'] = module
        return module


def compile_variant(module, *, query, key_cache, value_cache, request_ids, direct, return_lse, kv_warps,
                    selection_width):
    _, raw = module._compile(query=query, key_cache=key_cache, value_cache=value_cache,
                             request_ids=request_ids, direct_output=direct,
                             return_lse=return_lse if direct else True,
                             kv_warps=kv_warps if direct else 2, selection_width=selection_width)
    return raw


def launch_scores(module, raw, merge, *, query, key_cache, value_cache, k_descale, v_descale, block_table,
                  request_ids, selected_positions, query_positions, output, output_lse, partial_output,
                  partial_lse, softmax_scale, splits, direct, score_rows, scores):
    """The vendor ``_prepared`` launch (_sparse_gqa.launch_sparse_paged_gqa) plus the two score pointers."""
    BFloat16, Float32, Int32, Int64 = module.BFloat16, module.Float32, module.Int32, module.Int64
    pointer, fake = module._pointer, module._fake_pointer
    stream = module.current_cuda_stream()
    rows = int(query.shape[0])
    request_type = Int32 if request_ids.dtype == torch.int32 else Int64
    kv_type = module.Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
    module.run_compiled(raw, (
        pointer(query, BFloat16), pointer(key_cache, kv_type), pointer(value_cache, kv_type),
        pointer(k_descale, Float32) if k_descale is not None else fake(Float32),
        pointer(v_descale, Float32) if v_descale is not None else fake(Float32),
        pointer(block_table, Int32), pointer(request_ids, request_type),
        pointer(selected_positions, Int32), pointer(query_positions, Int64),
        fake(Float32) if direct else pointer(partial_output, Float32),
        (pointer(output_lse, Float32) if output_lse is not None else fake(Float32))
        if direct else pointer(partial_lse, Float32),
        pointer(output, BFloat16), pointer(score_rows, Int32), pointer(scores, Float32),
        int(key_cache.shape[0]), int(block_table.shape[0]), int(block_table.shape[1]),
        float(softmax_scale), rows, 1 if direct else int(splits), stream))
    if not direct:
        module.run_compiled(merge, (
            pointer(partial_output, Float32), pointer(partial_lse, Float32), pointer(output, BFloat16),
            pointer(output_lse, Float32) if output_lse is not None else fake(Float32),
            rows, int(splits), stream))


def sampled_logits(binding, inputs, rows, kv_warps=4):
    """[len(rows), q_heads, width] masked QK logits of the given query rows, from the
    score variant launched on those rows alone (direct entry, output discarded).

    The layer's own attention launch is left to the vendor kernel. The variant reads
    the same query, selected positions, positions, block table and K/V cache, and
    computes each logit with the same instruction sequence (one QK MMA per
    tile), so the logits are the ones the layer's attention normalizes."""
    module = load_module()
    query = inputs['query']
    device = query.device
    index = torch.tensor(rows, dtype=torch.int64, device=device)
    picked = query.index_select(0, index)
    request_ids = inputs['request_ids'].index_select(0, index)
    selected = binding.selected_positions.index_select(0, index)
    width = int(selected.shape[1])
    raw = compile_variant(module, query=picked, key_cache=binding.main_k_cache, value_cache=binding.main_v_cache,
                          request_ids=request_ids, direct=True, return_lse=False, kv_warps=kv_warps,
                          selection_width=width)
    scores = torch.empty((len(rows), int(query.shape[1]), width), dtype=torch.float32, device=device)
    launch_scores(module, raw, None, direct=True, splits=1, query=picked, key_cache=binding.main_k_cache,
                  value_cache=binding.main_v_cache, k_descale=binding.k_descale, v_descale=binding.v_descale,
                  block_table=binding.main_block_table, request_ids=request_ids, selected_positions=selected,
                  query_positions=inputs['query_positions'].index_select(0, index),
                  output=torch.empty_like(picked), output_lse=None, partial_output=None, partial_lse=None,
                  softmax_scale=int(query.shape[2]) ** -0.5,
                  score_rows=torch.arange(len(rows), dtype=torch.int32, device=device), scores=scores)
    return scores
