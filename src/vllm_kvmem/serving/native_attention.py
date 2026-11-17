"""Bounded native-QK side outputs for FlashInfer FA2 and SM12x XQA.

One runner, one CUDA stream and one public batch own the scratch. Each selected
layer retains compact LSE pieces at stable addresses. Normalization runs outside
the graph only for the actual sampled rows, before Action's existing D2H.
"""
from contextvars import ContextVar
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import wraps
import os
import importlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from threading import Lock, get_ident

import numpy as np
import torch
import triton
import triton.language as tl

from .native_attention_build import UnsupportedNativeStats, source_overlay, prefill_module, xqa_module
from ..core import settings as _settings


def statistic_layer_names(runner, options):
    """Capture every pressure layer without changing decode's selected layers.

    Layerwise pressure consumes physical segment masses separately and maps them
    back through each layer's own history. It must not inherit spread3's producer
    scope merely because decode still uses that policy.
    """
    from .. import _native
    names = tuple(runner._kvmem_layers)
    ids = [int(re.search(r'layers\.(\d+)\.', name).group(1)) for name in names]
    selected = set(_native.resolve_action_layers(options.action_layers, ids))
    decode = tuple(name for name, number in zip(names, ids) if number in selected)
    pressure = getattr(runner, '_kvmem_layerwise_pressure', None)
    if pressure is None:
        return decode, decode
    if tuple(pressure.names) != names:
        raise RuntimeError('Native pressure statistics must cover the ordered target layers, excluding MTP')
    return names, names if getattr(pressure, 'rolling_decode', False) else decode


def piece_layout(starts, counts, length):
    """Dense sixteen-token pieces split at every KVMem logical-segment boundary."""
    cursor = pieces = 0
    result = []
    for start, count in zip(starts, counts, strict=True):
        if start != cursor or count <= 0:
            raise ValueError('Native attention segments must partition the full context')
        n = (start + count + 15) // 16 - start // 16
        result.append((start, count, pieces, n))
        cursor += count
        pieces += n
    if cursor != length or not length:
        raise ValueError('Native attention segments omit visible context')
    return result, pieces


def _layout_array(starts, counts, length):
    start, count = np.asarray(starts, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    if (start.ndim != 1 or start.shape != count.shape or not start.size or
            start[0] != 0 or np.any(count <= 0) or
            not np.array_equal(start[1:], start[:-1] + count[:-1]) or start[-1]+count[-1] != length):
        raise ValueError('Native attention segments must partition the full context')
    sizes = (start+count+15)//16 - start//16
    ends = np.cumsum(sizes)
    return np.stack((start,count,ends-sizes,sizes),axis=1).astype(np.int32), int(ends[-1])


def aligned_piece_layout(layout, length):
    """Whether the visible partition's piece IDs equal dense token_index // 16.

    Cached candidate layouts can extend past the current visible prefix. Only
    that prefix matters; its last segment/count and causal tail may be partial.
    No physical page IDs or original-history positions enter this predicate.
    """
    layout = np.asarray(layout)
    if (layout.ndim != 2 or layout.shape[1] != 4 or not len(layout) or
            not np.issubdtype(layout.dtype, np.integer) or length <= 0):
        return False
    end = int(np.searchsorted(layout[:, 0], length, side='left'))
    live = layout[:end]
    if not len(live):
        return False
    starts, counts, first, pieces = live.T
    return bool(starts[0] == 0 and np.all(counts > 0) and
        np.array_equal(starts[1:], starts[:-1] + counts[:-1]) and
        starts[-1] + counts[-1] >= length and np.all(starts % 16 == 0) and
        np.array_equal(first, starts // 16) and
        np.array_equal(pieces, (starts + counts + 15) // 16 - starts // 16))


def static_aligned_frame(plans):
    """Freeze CPU geometry proved against the metadata uploaded by this arm.

    All owners must partition the complete visible prefix into dense pieces;
    only its final causal tail may be partial. No device metadata is read.
    """
    if not plans or not any(len(plan.get('rows', ())) for plan in plans.values()):
        return None, 'no_sampled_frame'
    spans = []
    try:
        for plan in plans.values():
            count, begin = int(plan['count']), int(plan['begin'])
            length = int(plan['local_begin']) + count
            if count <= 0 or begin < 0:
                return None, 'invalid_frame_geometry'
            layout = (plan['_layout'] if '_layout' in plan else
                      _layout_array(plan['starts'], plan['counts'], length)[0])
            if not aligned_piece_layout(layout, length):
                return None, 'unaligned_logical_segments'
            spans.append((int(plan['index']), begin, count))
        spans.sort()
        cursor = 0
        for index, (owner, begin, count) in enumerate(spans):
            if owner != index or begin != cursor:
                return None, 'noncontiguous_owner_frame'
            cursor += count
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return None, 'invalid_frame_geometry'
    return tuple(spans), None


@triton.jit
def _map_pieces(Ranges, Pieces, SEGMENTS: tl.constexpr, CONTEXT: tl.constexpr, BLOCK: tl.constexpr):
    request, segment = tl.program_id(0), tl.program_id(1)
    base = (request * SEGMENTS + segment) * 4
    start, count, piece = tl.load(Ranges + base), tl.load(Ranges + base + 1), tl.load(Ranges + base + 2)
    offset = tl.arange(0, BLOCK)
    token = start + offset
    tl.store(Pieces + request * CONTEXT + token, piece + token // 16 - start // 16, mask=offset < count)


def batched_arm():
    """KVMEM_NATIVE_ARM_BATCHED: arm() builds its row lists in bulk and clears every
    layer's LSE rows with one launch over a stacked buffer (same values written)."""
    import os
    value = _settings.get('KVMEM_NATIVE_ARM_BATCHED', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_ARM_BATCHED must be 0 or 1')
    return value == '1'


@triton.jit
def _clear_lse(Lse, Meta, Control, HEADS: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if row < tl.load(Control):
        width = tl.load(Meta + row * 3 + 1)
        x = tile * TILE + tl.arange(0, TILE)
        tl.store(Lse + (row * HEADS + head) * WIDTH + x, -float('inf'), mask=(x < width) & (x < WIDTH))


@triton.jit
def _clear_lse_layers(Lse, Meta, Control, LAYER_STRIDE, TILES, HEADS: tl.constexpr, WIDTH: tl.constexpr,
                      TILE: tl.constexpr):
    """_clear_lse for every layer of a stacked LSE buffer (axis 2 = layer * TILES + tile)."""
    row, head, packed = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer, tile = packed // TILES, packed % TILES
    if row < tl.load(Control):
        width = tl.load(Meta + row * 3 + 1)
        x = tile * TILE + tl.arange(0, TILE)
        tl.store(Lse + layer.to(tl.int64) * LAYER_STRIDE + (row * HEADS + head) * WIDTH + x, -float('inf'),
                 mask=(x < width) & (x < WIDTH))


@triton.jit
def _row_lse(Lse, Meta, Control, Selected, Valid, RowLse, HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, head = tl.load(Selected+tl.program_id(0)), tl.program_id(1)
    if row < tl.load(Control):
        width = tl.load(Meta + row * 3 + 1)
        x = tl.arange(0, BLOCK)
        values = tl.load(Lse + (row * HEADS + head) * WIDTH + x, mask=(x < width) & (x < WIDTH), other=-float('inf'))
        maximum = tl.max(values, 0)
        total = tl.sum(tl.exp(values - maximum), 0)
        value=maximum+tl.log(total)
        tl.store(RowLse + row * HEADS + head, value)
        if not ((value > -float('inf')) & (value < float('inf'))):
            tl.atomic_min(Valid,0)


@triton.jit
def _segment_scores(Lse, RowLse, Ranges, Meta, Control, Selected, Out,
                    HEADS: tl.constexpr, WIDTH: tl.constexpr, SEGMENTS: tl.constexpr,
                    H: tl.constexpr, B: tl.constexpr):
    row, segment = tl.load(Selected+tl.program_id(0)), tl.program_id(1)
    if row < tl.load(Control):
        request, segments = tl.load(Meta + row * 3), tl.load(Meta + row * 3 + 2)
        if segment < segments:
            row_width=tl.load(Meta+row*3+1)
            base = (request * SEGMENTS + segment) * 4
            first, count = tl.load(Ranges + base + 2), tl.load(Ranges + base + 3)
            h, p = tl.arange(0, H), tl.arange(0, B)
            normalizer = tl.load(RowLse + row * HEADS + h, mask=h < HEADS, other=0.)
            mass=0.
            for tile in range(tl.cdiv(count,B)):
                column=tile*B+p
                logits = tl.load(Lse + (row * HEADS + h[:, None]) * WIDTH + first + column[None, :],
                                 mask=(h[:, None] < HEADS) & (column[None, :] < count) &
                                      (first+column[None,:]<row_width), other=-float('inf'))
                probability = tl.exp(logits - normalizer[:, None])
                mass += tl.sum(tl.sum(probability, 1), 0) / HEADS
            tl.store(Out + row * SEGMENTS + segment, mass)


@triton.jit
def _row_lse_layers(Lse, Meta, Control, Selected, Valid, RowLse, LSE_STRIDE, ROW_STRIDE,
                    HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """_row_lse for every layer of a stacked bank at once (program_id(2) = layer).

    The per-row arithmetic is _row_lse's, on that layer's LSE, valid flag and row-LSE slice."""
    layer = tl.program_id(2)
    Lse += layer * LSE_STRIDE
    RowLse += layer * ROW_STRIDE
    row, head = tl.load(Selected+tl.program_id(0)), tl.program_id(1)
    if row < tl.load(Control):
        width = tl.load(Meta + row * 3 + 1)
        x = tl.arange(0, BLOCK)
        values = tl.load(Lse + (row * HEADS + head) * WIDTH + x, mask=(x < width) & (x < WIDTH), other=-float('inf'))
        maximum = tl.max(values, 0)
        total = tl.sum(tl.exp(values - maximum), 0)
        value=maximum+tl.log(total)
        tl.store(RowLse + row * HEADS + head, value)
        if not ((value > -float('inf')) & (value < float('inf'))):
            tl.atomic_min(Valid + layer,0)


@triton.jit
def _segment_scores_layers(Lse, RowLse, Ranges, Meta, Control, Selected, Out, LSE_STRIDE, ROW_STRIDE, OUT_STRIDE,
                           HEADS: tl.constexpr, WIDTH: tl.constexpr, SEGMENTS: tl.constexpr,
                           H: tl.constexpr, B: tl.constexpr):
    """_segment_scores for every layer of a stacked bank at once (program_id(2) = layer)."""
    layer = tl.program_id(2)
    Lse += layer * LSE_STRIDE
    RowLse += layer * ROW_STRIDE
    Out += layer * OUT_STRIDE
    row, segment = tl.load(Selected+tl.program_id(0)), tl.program_id(1)
    if row < tl.load(Control):
        request, segments = tl.load(Meta + row * 3), tl.load(Meta + row * 3 + 2)
        if segment < segments:
            row_width=tl.load(Meta+row*3+1)
            base = (request * SEGMENTS + segment) * 4
            first, count = tl.load(Ranges + base + 2), tl.load(Ranges + base + 3)
            h, p = tl.arange(0, H), tl.arange(0, B)
            normalizer = tl.load(RowLse + row * HEADS + h, mask=h < HEADS, other=0.)
            mass=0.
            for tile in range(tl.cdiv(count,B)):
                column=tile*B+p
                logits = tl.load(Lse + (row * HEADS + h[:, None]) * WIDTH + first + column[None, :],
                                 mask=(h[:, None] < HEADS) & (column[None, :] < count) &
                                      (first+column[None,:]<row_width), other=-float('inf'))
                probability = tl.exp(logits - normalizer[:, None])
                mass += tl.sum(tl.sum(probability, 1), 0) / HEADS
            tl.store(Out + row * SEGMENTS + segment, mass)


_active: ContextVar = ContextVar('kvmem_native_attention', default=None)


@dataclass
class NativeValidation:
    """One private same-forward completion fence; no buffer may be rearmed."""
    dirty: bool
    event: object = None
    recorded: bool = False
    resolved: bool = False
    error: object = None


@dataclass(frozen=True)
class DetachedValidationResult:
    invalid_names: tuple[str, ...]

    @property
    def valid(self):
        return not self.invalid_names


@dataclass(eq=False)
class DetachedNativeValidation:
    """One frame's independently owned download, with no reference to its bank.

    The caller owns the bounded flags/event lease. Resolving on a publication
    thread never reads live native scratch or updates native counters. A failed
    fence remains quarantined even if a later CUDA operation would succeed.
    """
    names: tuple[str, ...]
    dirty: bool
    frame_success: tuple[tuple[str, int], ...]
    host_flags: object
    event: object
    origin: object = field(repr=False)
    sequence: int
    recorded: bool = False
    resolved: bool = False
    applied: bool = False
    error: object = None
    result: object = None
    _lock: object = field(default_factory=Lock, repr=False)

    def resolve(self):
        with self._lock:
            if self.error is not None:
                raise RuntimeError('Detached native validation is quarantined') from self.error
            if self.resolved or not self.recorded:
                raise RuntimeError('Detached native validation is already resolved or incomplete')
            try:
                self.event.synchronize()
                flags = self.host_flags.tolist() if self.dirty else (1,) * len(self.names)
                if len(flags) != len(self.names):
                    raise ValueError('Detached native flags have an incomplete layer domain')
                result = DetachedValidationResult(tuple(name for name, valid in
                    zip(self.names, flags, strict=True) if not valid))
            except BaseException as error:
                self.error = error
                raise
            self.result = result
            self.resolved = True
            return result


def _install_xqa_dispatch():
    module = importlib.import_module('flashinfer.xqa')
    original = module.get_xqa_module
    if getattr(original, '_kvmem_native_stats', False):
        return

    @wraps(original)
    def dispatch(*args, **kwargs):
        active = _active.get()
        if active is None:
            return original(*args, **kwargs)
        bank, name = active
        q_dtype, kv_dtype, page, dim, group, sliding, out_dtype, q_len = args[:8]
        ragged = args[8] if len(args) > 8 else kwargs.get('use_ragged_q', False)
        if sliding or dim > 256 or kv_dtype != torch.float8_e4m3fn:
            bank.reasons[name, 'decode'] = 'unsupported_xqa_geometry'
            return original(*args, **kwargs)
        try:
            geometry = q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged
            built = bank.xqa_modules.get(geometry)
            if built is None:
                if torch.cuda.is_current_stream_capturing():
                    bank.reasons[name, 'decode'] = 'xqa_specialization_not_warmed'
                    return original(*args, **kwargs)
                built = xqa_module(bank.directory, *geometry)
                bank.xqa_modules[geometry] = built
        except (UnsupportedNativeStats, RuntimeError) as error:
            bank.reasons[name, 'decode'] = 'xqa_build:' + str(error).splitlines()[0][:160]
            return original(*args, **kwargs)
        bank.reasons[name, 'decode'] = None

        def run(run_sm90, sm, kh, window, qscale, output, oscale, q, sinks,
                k, v, ksf, vsf, table, maxlen, lengths, batch, kvscale,
                semaphores, workspace, pdl, qlen, qcu, mask):
            # SM12x uses this warp-specialized path, with the original scales.
            built.xqa_wrapper(False, sm, kh, window,
                1.0 if isinstance(qscale, torch.Tensor) else qscale,
                qscale if isinstance(qscale, torch.Tensor) else None,
                output, oscale, q, sinks, k, v, ksf, vsf, table, maxlen, lengths, batch,
                1.0 if isinstance(kvscale, torch.Tensor) else kvscale,
                kvscale if isinstance(kvscale, torch.Tensor) else None,
                qlen, qcu, mask, semaphores, workspace, pdl,
                bank.rows, bank.requests, bank.pieces, bank.lses[name], bank.context, bank.width)
        return SimpleNamespace(xqa=run)

    dispatch._kvmem_native_stats = True
    module.get_xqa_module = dispatch


def _batched_select():
    """Opt-in KVMEM_NATIVE_SCORE_BATCHED_SELECT=1 (read once): see NativeAttentionStats.score."""
    global _BATCHED_SELECT
    if _BATCHED_SELECT is None:
        value = _settings.get('KVMEM_NATIVE_SCORE_BATCHED_SELECT', '0')
        if value not in ('0', '1'):
            raise ValueError('KVMEM_NATIVE_SCORE_BATCHED_SELECT must be 0 or 1')
        _BATCHED_SELECT = value == '1'
    return _BATCHED_SELECT


_BATCHED_SELECT = None


def score_stack_enabled():
    """KVMEM_NATIVE_SCORE_STACK=1 (read once): layerwise capture takes every layer's
    native masses for a request from NativeAttentionStats.score_stack (one gather
    per frame, same values) instead of one score() call, gather and stack per layer."""
    global _SCORE_STACK
    if _SCORE_STACK is None:
        value = _settings.get('KVMEM_NATIVE_SCORE_STACK', '0')
        if value not in ('0', '1'):
            raise ValueError('KVMEM_NATIVE_SCORE_STACK must be 0 or 1')
        _SCORE_STACK = value == '1'
    return _SCORE_STACK


_SCORE_STACK = None


def window_frame_enabled():
    """KVMEM_NATIVE_WINDOW_FRAME=1 (read once): a decode owner's canonical window
    segments come from its frozen candidate frame once the native store confirms
    they are equal (no per-step Python rows or NumPy canonicalization)."""
    global _WINDOW_FRAME
    if _WINDOW_FRAME is None:
        value = _settings.get('KVMEM_NATIVE_WINDOW_FRAME', '0')
        if value not in ('0', '1'):
            raise ValueError('KVMEM_NATIVE_WINDOW_FRAME must be 0 or 1')
        _WINDOW_FRAME = value == '1'
    return _WINDOW_FRAME


_WINDOW_FRAME = None


class NativeAttentionStats:
    fa2_only = False

    def __init__(self, runner, options, output):
        from .. import _native
        self.options = options
        self.directory = str(Path(output) / 'native_attention_build')
        self.output = Path(output) / 'native_attention_stats.json'
        self.mode = options.action_native_scores
        self.reasons, self.counts = {}, Counter()
        self.xqa_modules, self.prefill_modules = {}, {}
        self.aligned_prefill_enabled = bool(getattr(options, 'action_native_prefill_aligned_pieces', False))
        self.aligned_prefill_counts = Counter()
        self.cta_gate_enabled = bool(getattr(options, 'action_native_prefill_cta_gate', False))
        self.cta_gate_counts = Counter()
        self.static_prefill_enabled = bool(getattr(options, 'action_native_prefill_static_aligned', False))
        if self.static_prefill_enabled and (self.aligned_prefill_enabled or self.cta_gate_enabled):
            raise ValueError('Static aligned FA2 cannot combine runtime aligned pieces or CTA gate')
        # Same host qualification and outputs; only the sampled-row capture
        # inside the static FA2 kernel uses the quad reduce-scatter variant.
        from ..core.defaults import legacy
        default_quad = getattr(options, 'action_native_prefill_static_quad', False) or (
            self.static_prefill_enabled and not legacy())
        flag = _settings.get('KVMEM_NATIVE_PREFILL_STATIC_QUAD', '1' if default_quad else '0')
        if flag not in ('0', '1'):
            raise ValueError('KVMEM_NATIVE_PREFILL_STATIC_QUAD must be 0 or 1')
        self.static_quad_enabled = flag == '1'
        if self.static_quad_enabled and not self.static_prefill_enabled:
            raise ValueError('Static quad FA2 requires static aligned FA2')
        self.static_prefill_counts = Counter()
        self.static_prefill_source_fingerprint = None
        self.static_prefill_unavailable = None
        self.prefill_source_fingerprint = None
        self.plans, self.slots = {}, {}
        self.reduced = set()
        self._gathered = {}
        self.invalid = set()
        self.validation_failures = 0
        self.armed_rows = self.normalized_rows = 0
        self.disabled = None
        self.source_fingerprint = None
        self.report_calls = 0
        self.report_lock = Lock()
        self._detached_owner_thread = get_ident()
        self.allocated_bytes = 0
        self.gpu_bytes = self.host_bytes = 0
        self.verification = {'comparisons': 0, 'max_abs': 0., 'l1_max': 0.}
        self.rows = None
        self.disarm()
        modules = dict(runner.model.named_modules())
        producer_names, decode_names = statistic_layer_names(runner, options)
        layers = [modules[name] for name in producer_names]
        self.names = [layer.attn.layer_name for layer in layers]
        self.decode_names = {modules[name].attn.layer_name for name in decode_names}
        # `unavailable`: this platform (GPU architecture or installed attention
        # sources) cannot run the overlay at all. Portable consumers may then
        # use the reference scorer instead of failing (action_attention).
        self.unavailable = False
        capability = torch.cuda.get_device_capability()
        self.fa2_only = capability == (8, 0)
        if _settings.get('KVMEM_NATIVE_ATTENTION_FORCE_UNSUPPORTED') == '1':
            # Frozen campaigns use this switch to reproduce the original fallback receipt.
            self.disabled = 'unsupported_architecture_requires_sm12x'
        elif not self.fa2_only and capability[0] != 12:
            self.disabled = 'unsupported_architecture_requires_sm80_or_sm12x'
        if self.disabled:
            self.unavailable = True
            return
        try:
            self.source_fingerprint = source_overlay(self.directory).name
            self.prefill_source_fingerprint = (source_overlay(self.directory, prefill_cta_gate=True).name
                if self.cta_gate_enabled else self.source_fingerprint)
        except UnsupportedNativeStats as error:
            self.disabled = str(error)
            self.unavailable = True
            return
        if self.static_prefill_enabled:
            try:
                self.static_prefill_source_fingerprint = source_overlay(
                    self.directory, prefill_static_aligned=True).name
            except UnsupportedNativeStats as error:
                self.static_prefill_unavailable = str(error)
        geometry = {(layer.num_heads, layer.head_dim) for layer in layers}
        if len(geometry) != 1:
            self.disabled = 'nonuniform_head_geometry'
            return
        self.heads, _ = geometry.pop()
        self.max_rows = options.action_native_max_rows
        self.context = options.memory.select_budget + options.memory.gen_budget + runner.max_num_tokens
        self.max_segments = math.ceil(self.context / options.memory.block_tokens) + 2 * runner.max_num_reqs + 136
        self.width = math.ceil(self.context / 16) + self.max_segments
        shapes = {'rows': ((runner.max_num_tokens,), torch.int32),
            'requests': ((self.max_rows,), torch.int32),
            'pieces': ((runner.max_num_reqs, self.context), torch.int32),
            'ranges': ((runner.max_num_reqs, self.max_segments, 4), torch.int32),
            'meta': ((self.max_rows, 3), torch.int32), 'control': ((1,), torch.int32),
            'selected': ((self.max_rows,),torch.int32),
            'row_lse': ((self.max_rows, self.heads), torch.float32),
            'valid_flags': ((len(layers),), torch.int32)}
        if self.aligned_prefill_enabled:
            shapes['aligned_pieces'] = ((runner.max_num_reqs,), torch.int32)
        self.allocated_bytes = sum(math.prod(shape) * 4 for shape, _ in shapes.values())
        self.allocated_bytes += len(layers) * self.max_rows * self.max_segments * 4
        self.allocated_bytes += len(layers) * self.max_rows * self.heads * self.width * 4
        self.gpu_bytes = self.allocated_bytes
        host_fields = ('rows', 'requests', 'ranges', 'meta', 'valid_flags', 'selected')
        if self.aligned_prefill_enabled:
            host_fields += ('aligned_pieces',)
        self.host_bytes = sum(math.prod(shapes[field][0])*4 for field in host_fields)
        self.allocated_bytes += self.host_bytes
        if self.allocated_bytes > options.action_native_max_bytes:
            if (self.mode == 'reuse' and getattr(options, 'prefill_native_plan_arm', False) and
                    getattr(options, 'prefill_physical_coalescing', False)):
                # Logical prefill groups have no fallback when the overlay is
                # available but too large (only an unsupported platform uses the
                # portable scorer), so the first prefill would fail. Say so now.
                raise ValueError(
                    f'Native prefill scores need {self.allocated_bytes} bytes for {len(layers)} layers x '
                    f'{self.max_rows} rows x {self.heads} heads x width {self.width}, above '
                    f'action_native_max_bytes={options.action_native_max_bytes}; raise action_native_max_bytes '
                    'or turn off prefill_physical_coalescing')
            self.disabled = 'startup_buffer_capacity'
            self.allocated_bytes = 0
            self.gpu_bytes = self.host_bytes = 0
            return
        device = layers[0].q_norm.weight.device
        for name, (shape, dtype) in shapes.items():
            setattr(self, name, torch.empty(shape, dtype=dtype, device=device))
        self.host = {name: torch.empty(shapes[name][0],dtype=torch.int32,pin_memory=True) for name in host_fields}
        self.upload_event = None
        # One allocation, per-layer contiguous views (the same bytes per layer),
        # so score_stack() can gather every layer's selected rows at once.
        self._mass_stack = torch.empty((len(self.names), self.max_rows, self.max_segments),
                                       dtype=torch.float32, device=device)
        self.masses = {name: self._mass_stack[i] for i, name in enumerate(self.names)}
        self._lse_stack = None
        if batched_arm():
            # One allocation, per-layer contiguous views: the same bytes per layer,
            # so arm() can clear every layer's LSE rows with a single launch.
            self._lse_stack = torch.empty((len(self.names), self.max_rows, self.heads, self.width),
                                          dtype=torch.float32, device=device)
            self.lses = {name: self._lse_stack[i] for i, name in enumerate(self.names)}
        else:
            self.lses = {name: torch.empty((self.max_rows, self.heads, self.width), dtype=torch.float32, device=device)
                         for name in self.names}
        self._row_lse_stack = None
        if self._lse_stack is not None and score_stack_enabled():
            # Opt-in score stack: per-layer row-LSE scratch (layers x rows x heads
            # floats, ~0.3 MB) so every layer reduces in one launch per kernel.
            self._row_lse_stack = torch.empty((len(self.names), self.max_rows, self.heads),
                                              dtype=torch.float32, device=device)
            self.gpu_bytes += self._row_lse_stack.numel() * 4
            self.allocated_bytes += self._row_lse_stack.numel() * 4
        self.ranges.zero_()
        self.disarm()
        if not self.fa2_only:
            _install_xqa_dispatch()
        for layer in layers:
            self._wrap(layer.attn.impl, layer.attn.layer_name)

    def disarm(self):
        self._check_detached_validation(allow_submitted=True)
        if getattr(self, '_validation_ticket', None) is not None:
            raise RuntimeError('Native attention still owns unresolved validation')
        if self.rows is not None:
            self.rows.fill_(-1)
            self.control.zero_()
            if getattr(self, 'aligned_prefill_enabled', False):
                self.aligned_pieces.zero_()
        self.plans, self.slots = {}, {}
        self.reduced = set()
        self._gathered = {}
        self._gathered_stack = None
        self.invalid = set()
        self.validation_dirty = False
        self.frame_invalid = False
        self.frame_success = Counter()
        self._detached_frame_submitted = False
        self.selected_slots,self.request_reasons={},{}
        self.static_prefill_frame = None
        self.static_prefill_reason = 'unarmed_frame'
        if self.rows is not None:
            self.valid_flags.fill_(1)

    def arm(self, plans):
        self.disarm()
        if self.disabled:
            return
        total = sum(len(plan['rows']) for plan in plans.values())
        if total > self.max_rows:
            self.batch_reason = 'sample_row_capacity'
            return
        self.batch_reason = None
        meta, requests, row_indices, sample_ids = [], [], [], []
        # Pinned metadata has one bounded owner. Its previous DMA must finish
        # before the CPU modifies it; graph/input tensor addresses never change.
        if self.upload_event is not None and not self.upload_event.query():
            self.upload_event.synchronize()
        self.host['ranges'].zero_()
        self.host['rows'].fill_(-1)
        aligned_counts = Counter()
        if getattr(self, 'aligned_prefill_enabled', False):
            self.host['aligned_pieces'].zero_()
        for request_id, plan in plans.items():
            if not plan['rows']:
                continue
            if '_layout' in plan:
                layout,pieces=plan['_layout'],plan['_width']
            else:
                layout, pieces = _layout_array(plan['starts'], plan['counts'], plan['local_begin'] + plan['count'])
            if (len(layout) > self.max_segments or pieces > self.width or
                    plan['local_begin'] + plan['count'] > self.context):
                self.disarm()
                self.batch_reason = 'segment_or_context_capacity'
                return
            i, first, n = plan['index'], len(meta), len(plan['rows'])
            self.host['ranges'].numpy()[i, :len(layout)] = layout
            if getattr(self, 'aligned_prefill_enabled', False):
                aligned = aligned_piece_layout(layout, plan['local_begin'] + plan['count'])
                self.host['aligned_pieces'][i] = int(aligned)
                kind = 'aligned' if aligned else 'generic'
                aligned_counts[kind + '_candidate_owners'] += 1
                aligned_counts[kind + '_candidate_rows'] += n
            segments=plan.get('_segments',len(layout))
            self.slots[request_id] = first, n, segments
            if batched_arm():
                base = len(meta)
                sample_ids.extend(range(base, base + n))
                row_indices.extend(plan['begin'] + row for row in plan['rows'])
                meta.extend([(i, pieces, segments)] * n)
                requests.extend([i] * n)
                continue
            for row in plan['rows']:
                sample_ids.append(len(meta))
                row_indices.append(plan['begin'] + row)
                meta.append((i, pieces, segments))
                requests.append(i)
        if meta:
            self.armed_rows += len(meta)
            self.host['meta'].numpy()[:len(meta)] = meta
            self.host['requests'].numpy()[:len(meta)] = requests
            self.host['rows'].numpy()[row_indices] = sample_ids
            upload_names = ('rows', 'ranges')
            if getattr(self, 'aligned_prefill_enabled', False):
                upload_names += ('aligned_pieces',)
                self.aligned_prefill_counts.update(aligned_counts)
            for name in upload_names:
                getattr(self,name).copy_(self.host[name],non_blocking=True)
            for name in ('meta','requests'):
                getattr(self,name)[:len(meta)].copy_(self.host[name][:len(meta)],non_blocking=True)
            self.upload_event = torch.cuda.Event()
            self.upload_event.record()
            self.control.fill_(len(meta))
            _map_pieces[(max(plan['index'] for plan in plans.values())+1, max(m[2] for m in meta))](self.ranges, self.pieces,
                self.max_segments, self.context, triton.next_power_of_2(self.options.memory.block_tokens))
            self.active_rows, self.active_segments = len(meta), max(m[2] for m in meta)
            self.reduction_rows=len(meta)
            self.host['selected'].numpy()[:len(meta)]=np.arange(len(meta),dtype=np.int32)
            self.selected[:len(meta)].copy_(self.host['selected'][:len(meta)],non_blocking=True)
            self.upload_event.record()
            tiles = triton.cdiv(max(m[1] for m in meta), 256)
            if self._lse_stack is not None:
                stack = self._lse_stack
                _clear_lse_layers[(len(meta), self.heads, tiles * stack.shape[0])](
                    stack, self.meta, self.control, stack.stride(0), tiles, self.heads, self.width, 256)
            else:
                for lse in self.lses.values():
                    _clear_lse[(len(meta), self.heads, tiles)](
                        lse, self.meta, self.control, self.heads, self.width, 256)
        self.plans = plans
        if getattr(self, 'static_prefill_enabled', False):
            self.static_prefill_frame, self.static_prefill_reason = static_aligned_frame(plans)
            key = ('qualified_candidate_frames' if self.static_prefill_frame is not None else
                   'candidate_fallback:' + self.static_prefill_reason)
            self.static_prefill_counts[key] += 1

    def resolve(self, plans):
        """Bind exact sampled rows/logical segments after native forward."""
        self._check_detached_validation()
        if getattr(self, '_validation_ticket', None) is not None:
            raise RuntimeError('Planning cannot modify a queued native validation frame')
        if self.disabled or getattr(self,'batch_reason',None):
            return
        if self.upload_event is not None and not self.upload_event.query():
            self.upload_event.synchronize()
        selected=[];maximum=0
        for request_id,plan in plans.items():
            if not plan['rows']:
                continue
            if request_id not in self.slots:
                self.request_reasons[request_id]='candidate_not_prepared'
                continue
            candidate=self.plans[request_id]
            if '_layout' not in candidate and plan is candidate:
                # An exact plan armed beside candidates (a prefill owner in a
                # mixed batch) keeps every armed row and its own segments.
                first,n,segments=self.slots[request_id]
                selected.extend(range(first,first+n))
                maximum=max(maximum,segments)
                continue
            try:
                if '_layout' in candidate and plan.get('_framed') is candidate:
                    # KVMEM_NATIVE_WINDOW_FRAME: these rows are this frame's prefix,
                    # already confirmed equal to the live window by the native store.
                    pass
                elif '_layout' in candidate:
                    extent=plan['local_begin']+plan['count']
                    end=int(np.searchsorted(candidate['starts'],extent-1,side='right'))
                    expected=np.stack((candidate['starts'][:end],
                        np.minimum(candidate['counts'][:end],extent-candidate['starts'][:end]),
                        candidate['history'][:end]),axis=1)
                    actual=np.stack((plan['starts'],plan['counts'],plan['history']),axis=1)
                    if not np.array_equal(expected,actual):
                        raise ValueError('candidate logical history changed')
                else:
                    raise ValueError('candidate cached layout missing')
                first,n,_=self.slots[request_id]
                source_rows=self.plans[request_id]['rows']
                row_ids=[first+source_rows.index(row) for row in plan['rows']]
            except (ValueError,IndexError) as error:
                self.request_reasons[request_id]='candidate_domain:'+str(error)
                continue
            self.selected_slots[request_id]=(len(selected),len(row_ids))
            selected.extend(row_ids)
            segments=len(plan['starts'])
            self.slots[request_id]=(first,n,segments)
            maximum=max(maximum,self.plans[request_id].get('_segments',segments))
        self.reduction_rows=len(selected)
        self.active_segments=maximum
        if selected:
            self.host['selected'].numpy()[:len(selected)]=selected
            self.selected[:len(selected)].copy_(self.host['selected'][:len(selected)],non_blocking=True)
            self.upload_event=torch.cuda.Event();self.upload_event.record()

    def _static_prefill_qualification(self, wrapper, run_args, metadata):
        if getattr(self, 'static_prefill_unavailable', None):
            return self.static_prefill_unavailable
        # Never bake a changing CPU qualification into a CUDA graph. A graph
        # captures the generic module even if this particular frame is aligned.
        if torch.cuda.is_current_stream_capturing():
            return 'cuda_graph_capture'
        if getattr(wrapper, 'is_cuda_graph_enabled', None) is not False:
            return 'graph_or_unknown_wrapper'
        plan_info = run_args[2]
        # FlashInfer's TVM FFI returns a host Array implementing Sequence, not
        # necessarily a Python list. Tensor plans remain ineligible: no device
        # scalar inspection or synchronization is permitted here.
        if (not isinstance(plan_info, Sequence) or len(plan_info) != 15 or
                type(plan_info[13]) not in (int, bool) or plan_info[13]):
            return 'graph_or_unknown_plan'
        frame = getattr(self, 'static_prefill_frame', None)
        if frame is None or getattr(self, 'disabled', None) or getattr(self, 'batch_reason', None):
            return getattr(self, 'static_prefill_reason', 'unarmed_frame')
        query, key = run_args[3:5]
        # Pure prefill only; mixed/decode owners retain the existing native
        # implementation. The checked host spans must cover the actual FA2 Q.
        decodes = getattr(metadata, 'num_decode_tokens', None)
        owners = getattr(metadata, 'num_prefills', None)
        tokens = getattr(metadata, 'num_prefill_tokens', None)
        if any(type(value) is not int for value in (decodes, owners, tokens)):
            return 'unknown_owner_metadata'
        if decodes != 0:
            return 'mixed_decode_prefix'
        if (owners != len(frame) or tokens != query.shape[0] or
                sum(item[2] for item in frame) != query.shape[0]):
            return 'owner_or_query_domain'
        if query.shape[0] < 128:
            return 'small_prefill'
        layout = getattr(wrapper, '_kv_layout', None)
        if layout not in ('NHD', 'HND') or len(key.shape) != 4:
            return 'unknown_page_geometry'
        page = key.shape[1 if layout == 'NHD' else 2]
        if page <= 0 or page % 16:
            return 'unaligned_kv_page'
        # Source-checked FlashInfer plans express every KV split in whole
        # pages (scheduler.cuh: kv_chunk_size *= page_size). With window=-1,
        # kv_start_idx=0. Thus every nonterminal split starts at a multiple of
        # 16 regardless of partition_kv; a final partial page needs no CAS.
        if type(run_args[14]) is not int or run_args[14] != -1:
            return 'sliding_window'
        return None

    def _fa2_stats_module(self, wrapper, name, layer, phase, q_dtype, dim, out_dtype, attn_metadata, row_offset):
        # Both source-pinned wrappers call FA2 paged_run with the same 16 base
        # arguments and effective QK scale. Decode uses a noncausal mask for
        # one query at the end of its KV prefix and a causal mask for MTP.
        unsupported = f'unsupported_{phase}_backend_or_mask'
        if (getattr(wrapper, '_backend', None) != 'fa2' or
                getattr(wrapper, '_jit_module', None) is not None or
                q_dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)):
            return None, unsupported
        if phase == 'decode':
            if (not getattr(wrapper, 'use_tensor_cores', False) or
                    wrapper._q_len_per_req < 1 or wrapper._window_right != 0):
                return None, unsupported
        elif wrapper._custom_mask_buf is not None or not wrapper._causal:
            return None, unsupported
        if (wrapper._pos_encoding_mode != 'NONE' or wrapper._window_left != -1 or
                wrapper._logits_soft_cap not in (0, None) or dim > 256 or
                (self.fa2_only and getattr(wrapper, '_cached_kv_data_type', None) != torch.float8_e4m3fn)):
            return None, unsupported
        # Unquantized Q must have the same scale domain as the sampled scorer.
        if q_dtype in (torch.bfloat16, torch.float16) and getattr(layer, '_q_scale_float', 1.0) != 1.0:
            return None, 'q_scale_domain_mismatch'
        geometry = (q_dtype, torch.float8_e4m3fn,
                    out_dtype, wrapper._paged_kv_indptr_buf.dtype, dim)
        aligned_enabled = getattr(self, 'aligned_prefill_enabled', False)
        module_key = geometry + ('aligned16',) if aligned_enabled else geometry
        cta_gate = getattr(self, 'cta_gate_enabled', False)
        if cta_gate:
            module_key += ('cta_gate',)
        built = self.prefill_modules.get(module_key)
        if built is None:
            if torch.cuda.is_current_stream_capturing():
                raise UnsupportedNativeStats('prefill_specialization_not_warmed')
            build_options = {}
            if aligned_enabled:
                build_options['aligned_pieces'] = True
            if cta_gate:
                build_options['cta_gate'] = True
            built = prefill_module(self.directory, *geometry, **build_options)
            self.prefill_modules[module_key] = built

        def paged_run(*run_args):
            # Original FA2 ABI: 16 common parameters, then optional
            # mask/ALiBi fields, softcap and the effective QK scale.
            tensors = (self.rows, self.requests, self.pieces, self.lses[name])
            if aligned_enabled:
                tensors += (self.aligned_pieces,)
                self.aligned_prefill_counts['prefill_dispatch_calls'] += 1
            if cta_gate:
                self.cta_gate_counts['prefill_dispatch_calls'] += 1
            selected_module = built
            if phase == 'prefill' and getattr(self, 'static_prefill_enabled', False):
                reason = self._static_prefill_qualification(wrapper, run_args, attn_metadata)
                if reason is None:
                    quad = getattr(self, 'static_quad_enabled', False)
                    static_key = geometry + ('static_aligned16',) + (('quad_store',) if quad else ())
                    selected_module = self.prefill_modules.get(static_key)
                    if selected_module is None:
                        try:
                            selected_module = prefill_module(self.directory, *geometry,
                                static_aligned=True, **({'quad_store': True} if quad else {}))
                            self.prefill_modules[static_key] = selected_module
                        except (UnsupportedNativeStats, RuntimeError) as error:
                            reason = 'build:' + str(error).splitlines()[0][:160]
                            selected_module = built
                self.static_prefill_counts[
                    'static_dispatch_calls' if reason is None else 'generic_dispatch:' + reason] += 1
            selected_module.paged_run(*run_args[:16], *tensors,
                self.context, self.width, row_offset, run_args[23])
        return SimpleNamespace(paged_run=paged_run), None

    def _wrap(self, impl, name):
        original = impl.forward

        @wraps(original)
        def forward(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            if attn_metadata is None:
                return original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            # vLLM puts the native decode rows first; MTP verification may
            # instead be scheduled in the FA2 prefill suffix. The logical
            # prompt/decode label does not establish which wrapper produced a row.
            prefix = getattr(attn_metadata, 'num_decode_tokens', 0)
            prefill = getattr(attn_metadata, 'prefill', None)
            parts = []
            if self.fa2_only and prefix:
                parts.append(('decode', getattr(attn_metadata, 'decode', None), 0))
            if prefill is not None:
                parts.append(('prefill', prefill, prefix))

            saved = []
            bank_reason = None if parts else 'backend_not_observed'
            token = _active.set(None if self.fa2_only else (self, name))
            try:
                for phase, metadata, offset in parts:
                    wrapper = getattr(metadata, 'wrapper', None)
                    q_dtype = getattr(attn_metadata, 'q_data_type_' + phase, None)
                    try:
                        module, reason = self._fa2_stats_module(wrapper, name, layer, phase, q_dtype,
                            query.shape[-1], output.dtype, attn_metadata, offset)
                    except (UnsupportedNativeStats, RuntimeError) as error:
                        module, reason = None, 'prefill_build:' + str(error).splitlines()[0][:160]
                    bank_reason = bank_reason or reason
                    self.reasons[name, phase] = reason
                    if module is not None:
                        saved.append((wrapper, wrapper._cached_module))
                        wrapper._cached_module = module
                if self.fa2_only:
                    # The bank reduces all sampled owners together. Advertise both
                    # logical phases only when all physical rows can produce scores.
                    prefill_dtype = getattr(attn_metadata, 'q_data_type_prefill', None)
                    if getattr(attn_metadata, 'q_data_type_decode', prefill_dtype) != prefill_dtype:
                        bank_reason = 'q_scale_domain_mismatch'
                    self.reasons[name, 'prefill'] = self.reasons[name, 'decode'] = bank_reason
                return original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            finally:
                _active.reset(token)
                for wrapper, previous in reversed(saved):
                    wrapper._cached_module = previous
        impl.forward = forward

    def _check_detached_validation(self, *, allow_submitted=False):
        failed = getattr(self, '_detached_enqueue_failure', None)
        if failed is not None:
            raise RuntimeError('Native attention owns a failed detached enqueue') from failed.error
        if not allow_submitted and getattr(self, '_detached_frame_submitted', False):
            raise RuntimeError('Native attention frame was detached; rearm before reusing it')

    def can_detach_validation(self):
        """Only a complete reuse frame may take the asynchronous route.

        A missing producer discovered by the device flags is handled later.
        Known reference/mixed-layer fallbacks retain the synchronous API.
        """
        if (getattr(self, '_detached_enqueue_failure', None) is not None or
                getattr(self, '_detached_frame_submitted', False) or
                getattr(self, '_validation_ticket', None) is not None or
                self.mode != 'reuse' or self.disabled or self.frame_invalid or
                self.invalid or getattr(self, 'batch_reason', None) or
                any(self.request_reasons.values())):
            return False
        owners = sum(bool(plan['rows']) for plan in self.plans.values())
        if not owners:
            return not self.validation_dirty and not self.frame_success and not self.reduced
        return (self.validation_dirty and self.reduced == set(self.names) and
                self.frame_success == Counter({'reuse:prefill': owners * len(self.names)}) and
                all(request_id in self.slots for request_id, plan in self.plans.items() if plan['rows']))

    def detach_validation(self, host_flags, event):
        """Enqueue flags and a complete-score fence on the producer stream.

        All score reductions and mass downloads must already be on this same
        stream; subsequent arm/disarm must use it too. This method never waits.
        The caller supplies distinct bounded pinned storage and an exclusive
        event, held through successful resolution. Counter application reads
        only the frozen result, so it does not extend this DMA lease.
        No second GPU bank is used.
        """
        self._check_detached_validation()
        if not self.can_detach_validation():
            raise RuntimeError('Detached validation requires a complete native prefill reuse frame')
        thread = self.__dict__.setdefault('_detached_owner_thread', get_ident())
        if thread != get_ident():
            raise RuntimeError('Detached validation submission is engine-thread owned')
        if (host_flags.device.type != 'cpu' or host_flags.dtype != torch.int32 or
                tuple(host_flags.shape) != (len(self.names),) or
                not host_flags.is_contiguous() or not host_flags.is_pinned()):
            raise ValueError('Detached validation needs independent pinned int32 layer flags')
        if event is None or not callable(getattr(event, 'record', None)) or not callable(getattr(event, 'synchronize', None)):
            raise ValueError('Detached validation needs an exclusive completion event')
        pointer, size = host_flags.data_ptr(), len(self.names) * 4
        def aliases(other):
            address = other.data_ptr()
            return pointer < address + size and address < pointer + size
        shared = self.host.get('valid_flags')
        if shared is not None and aliases(shared):
            raise ValueError('Detached flags cannot alias the bank host scratch')
        pending = self.__dict__.setdefault('_detached_pending', {})
        if any((old.error is not None or not old.resolved) and
               (old.event is event or aliases(old.host_flags)) for old in pending.values()):
            raise ValueError('Detached validation flags/event are still leased by another frame')
        origin = self.__dict__.setdefault('_detached_origin', object())
        sequence = self.__dict__.get('_detached_sequence', 0) + 1
        ticket = DetachedNativeValidation(tuple(self.names), bool(self.validation_dirty),
            tuple(self.frame_success.items()), host_flags, event, origin, sequence)
        self._detached_sequence = sequence
        pending[sequence] = ticket
        try:
            if ticket.dirty:
                host_flags.copy_(self.valid_flags, non_blocking=True)
            event.record()
            ticket.recorded = True
        except BaseException as error:
            ticket.error = error
            self._detached_enqueue_failure = ticket
            raise
        self.validation_dirty = False
        self._detached_frame_submitted = True
        return ticket

    def apply_detached_validation(self, ticket, result):
        """Apply this detached frame's counters once, without changing live plans.

        Invalid native output has the existing whole-frame fallback policy.
        The caller must complete that reference scoring before publication;
        this method records the fallback decision, not its GPU completion.
        """
        if getattr(self, '_detached_owner_thread', None) != get_ident():
            raise RuntimeError('Detached validation statistics are engine-thread owned')
        pending = getattr(self, '_detached_pending', {})
        if (not isinstance(ticket, DetachedNativeValidation) or
                ticket.origin is not getattr(self, '_detached_origin', None) or
                pending.get(ticket.sequence) is not ticket):
            raise RuntimeError('Detached validation belongs to another bank or was already applied')
        with ticket._lock:
            if (ticket.error is not None or not ticket.resolved or ticket.applied or
                    ticket.result is not result):
                raise RuntimeError('Detached validation cannot apply an unresolved or foreign result')
            if not result.valid:
                success = Counter(dict(ticket.frame_success))
                self.counts.subtract(success)
                self.counts.update({'fallback:'+key.split(':', 1)[1]+':invalid_native_mass': value
                                    for key, value in success.items()})
                self.validation_failures += 1
            ticket.applied = True
            del pending[ticket.sequence]
        return result.valid

    def enqueue_validation(self):
        """Fence all scores/D2H, even if every layer used reference scoring.

        The one live ticket prevents rearm or host-flag reuse. The same-forward
        MTP caller may enqueue independent proposal work before resolving it.
        A failed copy/record retains the ticket and its buffers for quarantine.
        """
        self._check_detached_validation()
        if getattr(self, '_validation_ticket', None) is not None:
            raise RuntimeError('Native attention validation already queued')
        ticket = NativeValidation(bool(self.validation_dirty))
        self._validation_ticket = ticket
        try:
            if ticket.dirty:
                self.host['valid_flags'].copy_(self.valid_flags, non_blocking=True)
            ticket.event = torch.cuda.Event()
            ticket.event.record()
            ticket.recorded = True
        except BaseException as error:
            ticket.error = error
            raise
        return ticket

    def resolve_validation(self, ticket):
        if (getattr(self, '_validation_ticket', None) is not ticket or
                ticket.resolved or not ticket.recorded or ticket.event is None):
            raise RuntimeError('Native attention validation ticket is stale or incomplete')
        if ticket.error is not None:
            raise RuntimeError('Native attention validation ticket is quarantined') from ticket.error
        try:
            ticket.event.synchronize()
            if ticket.dirty:
                flags = self.host['valid_flags'].tolist()
                self.validation_dirty = False
                self.invalid.update(name for name, valid in zip(self.names, flags) if not valid)
                if self.invalid:
                    self.frame_invalid = True
                    self.validation_failures += 1
                    self.counts.subtract(self.frame_success)
                    self.frame_success.clear()
        except BaseException as error:
            ticket.error = error
            raise
        ticket.resolved = True
        self._validation_ticket = None
        return not self.frame_invalid

    def validate(self):
        """Original synchronous API; only dirty frames require a new fence."""
        self._check_detached_validation()
        if getattr(self, '_validation_ticket', None) is not None:
            raise RuntimeError('Queued native validation requires its original ticket')
        if self.validation_dirty:
            return self.resolve_validation(self.enqueue_validation())
        return not self.frame_invalid

    def _reduce(self, name):
        """Normalize one layer's LSE rows into segment masses, once per frame."""
        if name in self.reduced:
            return
        self.normalized_rows += self.reduction_rows
        lse = self.lses[name]
        _row_lse[(self.reduction_rows, self.heads)](lse, self.meta, self.control,self.selected,
            self.valid_flags[self.names.index(name)],self.row_lse,
            self.heads, self.width, triton.next_power_of_2(self.width), num_warps=8)
        _segment_scores[(self.reduction_rows, self.active_segments)](lse, self.row_lse, self.ranges,
            self.meta, self.control,self.selected,self.masses[name], self.heads, self.width, self.max_segments,
            triton.next_power_of_2(self.heads), triton.next_power_of_2(math.ceil(self.options.memory.block_tokens / 16) + 1))
        self.reduced.add(name)
        # GPU-only check here: a host readback inside the request/layer
        # loop would prevent enqueueing the remaining scoring work.
        self.validation_dirty = True

    def _reduce_layers(self, names):
        """Every producer layer's reduction in one launch per kernel when none is reduced yet.

        Same per-element arithmetic as _reduce on each layer (the layered kernels
        offset into the stacked LSE, row-LSE scratch, masses and valid flags);
        otherwise falls back to _reduce per layer."""
        stacks = (getattr(self, '_lse_stack', None), getattr(self, '_row_lse_stack', None),
                  getattr(self, '_mass_stack', None))
        if self.reduced or tuple(names) != tuple(self.names) or any(stack is None for stack in stacks):
            for name in names:
                self._reduce(name)
            return
        lse, row_lse, masses = stacks
        layers = len(self.names)
        self.normalized_rows += self.reduction_rows * layers
        _row_lse_layers[(self.reduction_rows, self.heads, layers)](lse, self.meta, self.control, self.selected,
            self.valid_flags, row_lse, lse.stride(0), row_lse.stride(0),
            self.heads, self.width, triton.next_power_of_2(self.width), num_warps=8)
        _segment_scores_layers[(self.reduction_rows, self.active_segments, layers)](lse, row_lse, self.ranges,
            self.meta, self.control, self.selected, masses, lse.stride(0), row_lse.stride(0), masses.stride(0),
            self.heads, self.width, self.max_segments,
            triton.next_power_of_2(self.heads), triton.next_power_of_2(math.ceil(self.options.memory.block_tokens / 16) + 1))
        self.reduced.update(self.names)
        self.validation_dirty = True

    def score_stack(self, names, request_id, prefill):
        """Every listed layer's masses for one request as one [layers, rows, segments] view.

        Same reductions, values and counters as score(name, ..., defer_validation=True)
        for each name, in order. Returns None, counting nothing, when any layer would
        fall back (the caller then takes the per-layer path) or the names are not
        the producer layers in order. The selected rows of every layer are gathered
        with one index_select per frame instead of one per request and layer.
        """
        self._check_detached_validation()
        if getattr(self, '_validation_ticket', None) is not None:
            raise RuntimeError('Scoring cannot modify a queued native validation frame')
        stack = getattr(self, '_mass_stack', None)
        phase = 'prefill' if prefill else 'decode'
        if (stack is None or tuple(names) != tuple(self.names) or self.disabled or
                getattr(self, 'batch_reason', None) or self.request_reasons.get(request_id) or
                request_id not in self.slots or self.frame_invalid or
                any(self.reasons.get((name, phase), 'backend_not_observed') for name in names)):
            return None
        self._reduce_layers(names)
        first, rows, segments = self.slots[request_id]
        if request_id in self.selected_slots:
            selected_begin, selected_count = self.selected_slots[request_id]
            gathered = self._gathered_stack
            if gathered is None:
                gathered = self._gathered_stack = stack.index_select(
                    1, self.selected[:self.reduction_rows].long())
            result = gathered[:, selected_begin:selected_begin+selected_count, :segments]
        else:
            result = stack[:, first:first+rows, :segments]
        self.counts[self.mode + ':' + phase] += len(names)
        self.frame_success[self.mode + ':' + phase] += len(names)
        return result

    def score(self, name, request_id, prefill, *, defer_validation=False):
        self._check_detached_validation()
        if getattr(self, '_validation_ticket', None) is not None:
            raise RuntimeError('Scoring cannot modify a queued native validation frame')
        phase = 'prefill' if prefill else 'decode'
        reason = self.disabled or getattr(self, 'batch_reason', None)
        if not reason:
            reason=self.request_reasons.get(request_id)
        if not reason and request_id not in self.slots:
            reason = 'unprepared_forward'
        if not reason:
            reason = self.reasons.get((name, phase), 'backend_not_observed')
        if not reason and self.frame_invalid:
            reason = 'invalid_native_mass'
        if reason:
            self.counts['fallback:' + phase + ':' + reason] += 1
            return None
        self._reduce(name)
        if not defer_validation:
            self.validate()
        first, rows, segments = self.slots[request_id]
        result = self.masses[name][first:first+rows, :segments]
        if request_id in self.selected_slots:
            selected_begin,selected_count=self.selected_slots[request_id]
            if _batched_select():
                # Opt-in: gather every selected row of this layer once per frame;
                # each request reads a view of its rows (same values, read-only).
                gathered=self._gathered.get(name)
                if gathered is None:
                    gathered=self._gathered[name]=self.masses[name].index_select(0,
                        self.selected[:self.reduction_rows].long())
                result=gathered[selected_begin:selected_begin+selected_count,:segments]
            else:
                result=self.masses[name][:,:segments].index_select(0,
                    self.selected[selected_begin:selected_begin+selected_count].long())
        # A missed producer (including a graph captured before instrumentation)
        # cannot publish uninitialized or stale mass. prepare() clears LSE before
        # every public forward, including graph replays; the initial empty rows
        # disable statistics during private forwards and warmup.
        if self.frame_invalid:
            self.counts['fallback:' + phase + ':invalid_native_mass'] += 1
            return None
        self.counts[self.mode + ':' + phase] += 1
        self.frame_success[self.mode + ':' + phase] += 1
        return result

    def compare(self, native, reference):
        difference = (native - reference).abs()
        self.verification['comparisons'] += 1
        self.verification['max_abs'] = max(self.verification['max_abs'], float(difference.max().item()))
        self.verification['l1_max'] = max(self.verification['l1_max'], float(difference.sum(-1).max().item()))

    def report(self, *, force=False):
        self.report_calls += 1
        if not force and self.report_calls != 1 and self.report_calls % 128:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.report_lock:
            temporary = self.output.with_suffix('.tmp')
            temporary.write_text(json.dumps({'requested_mode': self.mode, 'allocated_bytes': self.allocated_bytes,
                'producer_layers': self.names,
                'decode_score_layers': sorted(getattr(self, 'decode_names', self.names)),
                'persistent_gpu_bytes': self.gpu_bytes, 'persistent_pinned_host_bytes': self.host_bytes,
                'disabled_reason': self.disabled, 'layer_phase_counts': dict(self.counts),
                'source_fingerprint': self.source_fingerprint, 'verification': self.verification,
                'validation_failures': self.validation_failures,
                'armed_candidate_rows': self.armed_rows,
                'normalized_layer_sample_rows': self.normalized_rows,
                'prefill_cta_gate': dict(enabled=getattr(self, 'cta_gate_enabled', False),
                    counts=dict(getattr(self, 'cta_gate_counts', {})),
                    source_fingerprint=getattr(self, 'prefill_source_fingerprint', None),
                    dispatch_counts_exclude_graph_replays=True,
                    device_skipped_ctas_not_counted=True),
                'prefill_aligned_pieces': dict(enabled=getattr(self, 'aligned_prefill_enabled', False),
                    counts=dict(getattr(self, 'aligned_prefill_counts', {})),
                    candidate_counts_are_host_qualification=True,
                    dispatch_counts_exclude_graph_replays=True),
                'prefill_static_aligned': dict(enabled=getattr(self, 'static_prefill_enabled', False),
                    quad_store=getattr(self, 'static_quad_enabled', False),
                    counts=dict(getattr(self, 'static_prefill_counts', {})),
                    source_fingerprint=getattr(self, 'static_prefill_source_fingerprint', None),
                    unavailable=getattr(self, 'static_prefill_unavailable', None),
                    graph_dispatches_use_generic=True,
                    dispatch_counts_exclude_graph_replays=True)}, indent=2) + '\n')
            temporary.replace(self.output)


def install_native_attention(runner, options, output):
    if getattr(options, 'action_native_scores', 'off') == 'off':
        return None
    return NativeAttentionStats(runner, options, output)
