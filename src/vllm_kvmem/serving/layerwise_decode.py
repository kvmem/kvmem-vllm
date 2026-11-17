"""Accepted-token feedback for one native response across layerwise refreshes.

This pressure history is independent of the legacy non-thinking action parser.
Rows are sampled in original consumed-token coordinates, including thinking;
only scheduler-confirmed rows publish mass. No draft or rejected Q contributes.
Opt-in KVMEM_LAYERWISE_DECODE_SCOPE=non-thinking samples only non-thinking
output, and KVMEM_LAYERWISE_DECODE_PREFILL=1 adds accepted prefill rows.
"""
from dataclasses import dataclass, field

import numpy as np

from .. import _native
from ..core import settings as _settings


def sample_rows(begin, count, interval):
    if begin < 0 or count < 0 or interval <= 0:
        raise ValueError('Invalid rolling attention sampling coordinates')
    return [row for row in range(count) if (begin + row + 1) % interval == 0]


@dataclass
class DecodeHistory:
    mass: dict = field(default_factory=dict)
    samples: int = 0
    end: int = 0
    chunks: int = 0

    def prepare(self, pending, accepted, logical, masses):
        return _prepare_history(self, self.mass, pending, accepted, logical, masses)

    def scores(self):
        return {key: value / self.samples for key, value in self.mass.items()} if self.samples else {}


def decode_scope():
    """KVMEM_LAYERWISE_DECODE_SCOPE: which accepted output tokens score rolling decode.

    'all' (default): every action_interval-th accepted output token, thinking
    included (the original rule). 'non-thinking': only accepted output outside
    <think>...</think>, every decode_interval()-th such token, chosen by the
    native ActionParser 'non-thinking' scope on the same verifier tokens and
    acceptance as the legacy parser. Keys (candidates) are unchanged: every
    accepted history block, thinking included.
    """
    import os
    value = _settings.get('KVMEM_LAYERWISE_DECODE_SCOPE', 'all')
    if value not in ('all', 'non-thinking'):
        raise ValueError('KVMEM_LAYERWISE_DECODE_SCOPE must be all or non-thinking')
    return value


def decode_interval(options):
    """KVMEM_LAYERWISE_DECODE_INTERVAL: sampling interval of the 'non-thinking'
    scope, counted over non-thinking output tokens (default: action_interval)."""
    import os
    value = _settings.get('KVMEM_LAYERWISE_DECODE_INTERVAL')
    if value is None:
        return options.action_interval
    if not value.isdigit() or not 1 <= int(value) <= 4096:
        raise ValueError('KVMEM_LAYERWISE_DECODE_INTERVAL must be an integer in [1, 4096]')
    return int(value)


def prefill_fold_enabled():
    """KVMEM_LAYERWISE_DECODE_PREFILL=1: accepted public-prefill attention rows
    (tool results, user input; the rows the prefill history already samples,
    minus rows inside <think>...</think>) also score rolling decode."""
    import os
    value = _settings.get('KVMEM_LAYERWISE_DECODE_PREFILL', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_DECODE_PREFILL must be 0 or 1')
    return value == '1'


def prefill_fold_weight(options):
    """KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT: samples one folded prefill row
    counts as. Default: prefill stride / decode interval (rounded, >= 1), so a
    prefill token and a non-thinking output token weigh about the same."""
    import os
    value = _settings.get('KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT')
    if value is not None:
        if not value.isdigit() or not 1 <= int(value) <= 4096:
            raise ValueError('KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT must be an integer in [1, 4096]')
        return int(value)
    stride = getattr(options, 'prefill_attention_stride', 0) or 32
    return max(1, round(stride / decode_interval(options)))


def query_source_label():
    """Evidence label of the rolling decode score source."""
    if decode_scope() == 'all':
        return 'accepted_output_including_thinking', 'all_accepted_including_thinking'
    if prefill_fold_enabled():
        return 'accepted_non_thinking_output_and_prefill', 'accepted_non_thinking_and_prefill'
    return 'accepted_non_thinking_output', 'accepted_non_thinking'


def _wanted_rows(pending, accepted):
    """Accepted rows this packet must fold: the non-thinking mask chosen when the
    packet was planned, or the original every-interval rule."""
    mask = getattr(pending, 'layerwise_mask', None)
    if mask is not None:
        return {row for row in range(accepted) if (mask >> row) & 1}
    return set(sample_rows(pending.history_begin, accepted, pending.state.options.action_interval))


def _prepare_history(history, source, pending, accepted, logical, masses):
    """Stage the next accepted-mass state without mutating ``history``."""
    if not 0 <= accepted <= pending.candidate_count:
        raise ValueError('Invalid rolling attention accepted prefix')
    begin = pending.history_begin
    wanted = _wanted_rows(pending, accepted)
    indices = [i for i, row in enumerate(pending.rows)
               if row in wanted and begin + row >= history.end]
    if wanted - set(pending.rows):
        raise RuntimeError('Rolling attention omitted an accepted sampled row')
    if not indices:
        return history
    values = masses.numpy() if hasattr(masses, 'numpy') else masses
    cap = getattr(pending.state.options, 'action_commit_max_bytes', 16777216)
    if mass_map_enabled():
        if not isinstance(source, (dict, _native.DecodeMassMap)):
            source = history.mass
        # A row of the layer-batch identity array (KVMEM_NATIVE_DECODE_LAYER_BATCH)
        # is read natively; a list goes through the original entry point.
        prepare = (_native.prepare_decode_attention_map_targets if type(logical) is np.ndarray
                   else _native.prepare_decode_attention_map)
        mass_map, samples = prepare(source, history.samples, indices, logical, values, cap)
        return NativeDecodeHistory(mass_map, samples, begin + accepted, history.chunks + 1)
    if isinstance(source, _native.DecodeMassMap if hasattr(_native, 'DecodeMassMap') else ()):
        source = history.mass
    prepare = (_native.prepare_decode_attention_sparse if sparse_mass_enabled()
               else _native.prepare_decode_attention)
    mass, samples = prepare(source, history.samples, indices, logical, values, cap)
    return DecodeHistory(mass, samples, begin + accepted, history.chunks + 1)


class NativeDecodeHistory:
    """DecodeHistory whose accepted mass stays in a native DecodeMassMap
    (KVMEM_NATIVE_DECODE_MASS_MAP). ``mass`` materializes the identical dict
    (same keys, insertion order and float values) on first use, for
    checkpoints, swaps, historical GC and scores. Instances are never mutated."""

    __hash__ = None

    def __init__(self, mass_map, samples, end, chunks):
        self._map, self._mass = mass_map, None
        self.samples, self.end, self.chunks = samples, end, chunks

    @property
    def mass(self):
        if self._mass is None:
            self._mass = self._map.to_dict()
        return self._mass

    def prepare(self, pending, accepted, logical, masses):
        return _prepare_history(self, self._map, pending, accepted, logical, masses)

    def scores(self):
        return {key: value / self.samples for key, value in self.mass.items()} if self.samples else {}

    def __eq__(self, other):
        fields = ('mass', 'samples', 'end', 'chunks')
        if not all(hasattr(other, f) for f in fields):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) for f in fields)

    def __repr__(self):
        return (f'NativeDecodeHistory(mass=<{len(self._map)} keys>, samples={self.samples}, '
                f'end={self.end}, chunks={self.chunks})')


def mass_map_enabled():
    """KVMEM_NATIVE_DECODE_MASS_MAP=1: keep accepted decode mass natively."""
    import os
    value = _settings.get('KVMEM_NATIVE_DECODE_MASS_MAP', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_MASS_MAP must be 0 or 1')
    return value == '1' and hasattr(_native, 'prepare_decode_attention_map')


def record_map(records):
    cursor, result = 0, []
    for r in records:
        result.append((r.handle.slot, r.handle.generation, r.history_begin, r.count, cursor))
        cursor += r.count
    return tuple(result)


def sparse_mass_enabled():
    """KVMEM_NATIVE_DECODE_SPARSE_MASS=1: prepare_decode_attention_sparse, which
    returns the same dictionary but writes only changed entries after copy()."""
    import os
    value = _settings.get('KVMEM_NATIVE_DECODE_SPARSE_MASS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_SPARSE_MASS must be 0 or 1')
    return value == '1' and hasattr(_native, 'prepare_decode_attention_sparse')


def native_segments_enabled():
    """KVMEM_NATIVE_DECODE_SEGMENTS=1: compute segment_identities in C++.

    The native helper returns None for any input outside its exact int64 domain
    and for every input that would raise; the Python code below then runs and
    raises its original error. Equivalence: tests/test_native_decode_segments.py.
    """
    import os
    value = _settings.get('KVMEM_NATIVE_DECODE_SEGMENTS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_SEGMENTS must be 0 or 1')
    return value == '1' and hasattr(_native, 'decode_segment_identities')


def segment_identities(records, begin, local, count, starts, counts):
    """Map either split or canonically merged native partial-tail segments.

    A committed partial block and its current speculative continuation may be
    two window_segments entries but one native score column. A score column
    must still belong entirely to one original block in each target layer.
    """
    if native_segments_enabled():
        result = _native.decode_segment_identities(records, begin, local, count, starts, counts, 32)
        if result is not None:
            return result
    return _segment_identities_python(records, begin, local, count, starts, counts)


def _segment_identities_python(records, begin, local, count, starts, counts):
    from .action_attention import window_segments
    ls, lc, lh = window_segments(records, begin, local, count, 32)
    cursor = index = 0
    result = []
    for start, size in zip(starts, counts, strict=True):
        start, size = int(start), int(size)
        if start != cursor or size <= 0:
            raise RuntimeError('Rolling score segments do not partition the native window')
        end = start + size
        identity = None
        while cursor < end:
            if index >= len(ls) or not ls[index] <= cursor < ls[index]+lc[index]:
                raise RuntimeError('Rolling layer map changed physical segment geometry')
            if identity is not None and identity != lh[index]:
                raise RuntimeError('One rolling score column crosses different logical blocks')
            identity = lh[index]
            cursor = min(end, ls[index]+lc[index])
            if cursor == ls[index]+lc[index]: index += 1
        # Native segment arrays carry NumPy scalars. Logical coordinates also
        # feed the legacy graph, whose canonical key contract is Python int.
        result.append(int(identity))
    if cursor != local+count or index != len(ls):
        raise RuntimeError('Rolling score segments omit part of the native window')
    return result


def layer_map_enabled():
    """KVMEM_NATIVE_DECODE_LAYER_MAP=1: per-layer decode identities straight from
    the frozen pressure frame (parsed once per frame), without building record lists."""
    import os
    value = _settings.get('KVMEM_NATIVE_DECODE_LAYER_MAP', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_LAYER_MAP must be 0 or 1')
    return value == '1' and hasattr(_native, 'decode_identities_from_map')


def layer_batch_enabled():
    """KVMEM_NATIVE_DECODE_LAYER_BATCH=1: all layers' decode identities in one
    native call, kept as one int64 [layers, columns] array that the accepted-mass
    update reads directly (no per-layer list of Python ints). Layer 0 is also
    returned as the usual list of ints for ``pending.history``.

    Effective only together with KVMEM_NATIVE_DECODE_LAYER_MAP=1 and
    KVMEM_NATIVE_DECODE_MASS_MAP=1, whose code it builds on. Any input the batch
    call does not reproduce exactly returns None and the per-layer path runs.
    Equivalence: tests/test_native_decode_segments.py.
    """
    import os
    value = _settings.get('KVMEM_NATIVE_DECODE_LAYER_BATCH', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_LAYER_BATCH must be 0 or 1')
    return (value == '1' and hasattr(_native, 'decode_identities_from_maps') and
            hasattr(_native, 'prepare_decode_attention_map_targets') and
            layer_map_enabled() and mass_map_enabled())


def prepare_batch_enabled():
    """KVMEM_NATIVE_DECODE_PREPARE_BATCH=1: stage every layer's accepted mass of
    one request in one native call (prepare_decode_attention_maps_targets) instead
    of one prepare_decode_attention_map_targets call per layer.

    The layers run on the native publication pool (KVMEM_NATIVE_PUBLISH_THREADS)
    without the GIL, map copy included; each layer's arithmetic is unchanged, so
    the staged maps are bitwise equal to the per-layer loop. Effective only with
    KVMEM_NATIVE_DECODE_LAYER_BATCH=1 (whose identity array it reads) and only
    when every layer is a NativeDecodeHistory folding the same rows; anything
    else runs the per-layer loop. Equivalence: tests/test_native_decode_layer_batch.py.
    """
    value = _settings.get('KVMEM_NATIVE_DECODE_PREPARE_BATCH', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_PREPARE_BATCH must be 0 or 1')
    return value == '1' and hasattr(_native, 'prepare_decode_attention_maps_targets')


def _prepare_batch(histories, pending, accepted, logical, masses):
    """prepare_accept's staged histories from one native call, or None to run the per-layer loop.

    Makes _prepare_history's checks in its order (they are the same for every
    layer), then batches only when every layer folds the same non-empty rows.
    """
    if (type(logical) is not np.ndarray or logical.ndim != 2 or not histories or
            any(type(h) is not NativeDecodeHistory for h in histories) or not mass_map_enabled()):
        return None
    if not 0 <= accepted <= pending.candidate_count:
        raise ValueError('Invalid rolling attention accepted prefix')
    begin = pending.history_begin
    wanted = _wanted_rows(pending, accepted)
    end = histories[0].end
    if any(h.end != end or type(h.samples) is not int for h in histories):
        return None
    indices = [i for i, row in enumerate(pending.rows) if row in wanted and begin + row >= end]
    if wanted - set(pending.rows):
        raise RuntimeError('Rolling attention omitted an accepted sampled row')
    if not indices:
        return None
    values = masses.numpy() if hasattr(masses, 'numpy') else masses
    cap = getattr(pending.state.options, 'action_commit_max_bytes', 16777216)
    maps = _native.prepare_decode_attention_maps_targets([h._map for h in histories], indices, logical, values, cap)
    if maps is None:
        return None
    return [NativeDecodeHistory(mass_map, h.samples + len(indices), begin + accepted, h.chunks + 1)
            for mass_map, h in zip(maps, histories)]


_LAYER_MAP_CACHE_LIMIT = 1024


def _fused_identities(controller, rid, begin, local, count, starts, counts, batch=False):
    """All layers' identities via the native fused path, or None to run Python.

    Equals segment_identities(controller.records(rid, authority, begin, local, layer), ...)
    for every layer; any input the native helper cannot reproduce exactly (including
    every error case) returns None so the original per-layer code runs and raises.

    With ``batch`` the result is (layer-0 list, int64 [layers, columns] array)
    from one native call, or None to run the per-layer native path.
    """
    from .layerwise_pressure import LayerwisePrefillPressure
    if getattr(type(controller), 'records', None) is not LayerwisePrefillPressure.records:
        return None
    frame = controller.resolved_maps(rid)
    if frame is None:
        return None
    origin, layers = frame
    cache = controller.__dict__.setdefault('_decode_layer_maps', {})
    if batch:
        count_layers = len(controller.names)
        hit = cache.get((rid, None))
        # One entry per request: the per-layer maps the arrays were parsed from
        # (identity-checked, like the per-layer entries below) and the arrays.
        if (hit is None or len(hit[0]) != count_layers or
                any(hit[0][layer] is not layers[layer] for layer in range(count_layers))):
            sources = [layers[layer] for layer in range(count_layers)]
            parsed = []
            for records in sources:
                one = _native.parse_layer_map(records)
                if one is None:
                    return None
                parsed.append(one)
            if len(cache) >= _LAYER_MAP_CACHE_LIMIT:
                cache.clear()
            cache[(rid, None)] = hit = (sources, parsed)
        return _native.decode_identities_from_maps(hit[1], origin, begin, local, count, starts, counts)
    result = []
    for layer in range(len(controller.names)):
        records = layers[layer]
        hit = cache.get((rid, layer))
        if hit is None or hit[0] is not records:
            parsed = _native.parse_layer_map(records)
            if parsed is None:
                return None
            if len(cache) >= _LAYER_MAP_CACHE_LIMIT:
                cache.clear()
            cache[(rid, layer)] = hit = (records, parsed)
        identities = _native.decode_identities_from_map(hit[1], origin, begin, local, count, starts, counts)
        if identities is None:
            return None
        result.append(identities)
    return result


def attach_packet(controller, pending, rid, authority, starts, counts, masses):
    begin, count = pending.history_begin, pending.candidate_count
    local = sum(map(int, counts)) - count
    logical = history = None
    if layer_batch_enabled():
        fused = _fused_identities(controller, rid, begin, local, count, starts, counts, batch=True)
        if fused is not None:
            history, logical = fused
    if logical is None and layer_map_enabled():
        logical = _fused_identities(controller, rid, begin, local, count, starts, counts)
    if logical is None:
        logical = []
        for layer in range(len(controller.names)):
            records = controller.records(rid, authority, begin, local, layer)
            logical.append(segment_identities(records, begin, local, count, starts, counts))
    # Before the first decode refresh the shared frame still describes the
    # initial global Query preselection. Feedback must describe real layer zero.
    pending.history = logical[0] if history is None else history
    pending.layerwise_decode = controller, rid, logical, masses
    controller.decode_histories.setdefault(rid, [DecodeHistory() for _ in controller.names])


def prepare_accept(controller, pending, accepted):
    owner, rid, logical, masses = pending.layerwise_decode
    if owner is not controller or masses.shape[0] != len(controller.names):
        raise RuntimeError('Rolling feedback belongs to a different layer domain')
    histories = controller.decode_histories[rid]
    staged = _prepare_batch(histories, pending, accepted, logical, masses) if prepare_batch_enabled() else None
    if staged is None:
        staged = [history.prepare(pending, accepted, logical[i], masses[i])
                  for i, history in enumerate(histories)]
    if len({(h.samples, h.end, h.chunks) for h in staged}) != 1:
        raise RuntimeError('Rolling layers disagree on scheduler acceptance')
    return rid, staged, 'decode'


def fold_prefill(controller, rid, pending, accepted, logical, masses):
    """Staged decode histories with this accepted prefill chunk's rows, or None.

    Opt-in KVMEM_LAYERWISE_DECODE_PREFILL. Uses the rows the prefill history
    already sampled (no extra scoring), minus rows inside <think>...</think>
    (``pending.prefill_thinking``, set when the chunk was planned). Like the
    prefill history, only a fully accepted chunk counts. Each kept row is
    validated like a decode row and counts as prefill_fold_weight() samples;
    rows before a history's end (replays, restores) are skipped.
    """
    end, count = pending.prefill[0], pending.prefill[1]
    if accepted != count or not pending.rows:
        return None
    begin = end - count
    excluded = getattr(pending, 'prefill_thinking', None) or ()
    histories = controller.decode_histories.get(rid)
    if histories is None:
        histories = [DecodeHistory() for _ in controller.names]
    if len(histories) != len(controller.names) or masses.shape[0] != len(controller.names):
        raise RuntimeError('Rolling prefill feedback belongs to a different layer domain')
    weight = prefill_fold_weight(pending.state.options)
    values = masses.numpy() if hasattr(masses, 'numpy') else np.asarray(masses)
    staged, changed = [], False
    for layer, history in enumerate(histories):
        indices = [i for i, row in enumerate(pending.rows)
                   if row not in excluded and begin + row >= history.end]
        if not indices:
            staged.append(history)
            continue
        rows = np.asarray(values[layer], dtype=np.float64)[indices]
        targets = logical[layer]
        if rows.ndim != 2 or rows.shape[1] != len(targets):
            raise RuntimeError('Rolling prefill feedback has the wrong score geometry')
        if (not np.isfinite(rows).all() or (rows < 0).any() or
                (np.abs(rows.sum(axis=1) - 1.) > .005).any()):
            raise RuntimeError('Action attention mass is invalid or does not sum to one')
        column = rows.sum(axis=0) * weight
        mass = dict(history.mass)
        for key, value in zip(targets, column.tolist()):
            if value:
                key = int(key)
                mass[key] = mass.get(key, 0.) + value
        samples = history.samples + weight * len(indices)
        if mass_map_enabled():
            staged.append(NativeDecodeHistory(_native.DecodeMassMap.from_dict(mass), samples, end,
                                              history.chunks + 1))
        else:
            staged.append(DecodeHistory(mass, samples, end, history.chunks + 1))
        changed = True
    if not changed:
        return None
    if len({(h.samples, h.end, h.chunks) for h in staged}) != 1:
        raise RuntimeError('Rolling layers disagree on prefill acceptance')
    return staged


def prefill_thinking_rows(state, begin, tokens, rows):
    """Rows of a prefill chunk at or inside <think>...</think>.

    The thinking state at the chunk start is the end state of this request's
    preceding chunk; a request's first prefill chunk starts outside thinking
    (prompts resume at a message boundary). '<|im_start|>' resets the state,
    as in ActionHistory.begin.
    """
    if not rows:
        return frozenset()
    ids = state.token_ids
    think_open, think_close, header = ids[2], ids[3], ids[4]
    states = state.__dict__.setdefault('_layerwise_prefill_thinking', {})
    thinking = states.get(begin, False)
    wanted, inside = set(rows), set()
    for row, token in enumerate(tokens):
        if token == think_open:
            thinking = True
            marked = True
        elif token == think_close:
            marked, thinking = True, False
        elif token == header:
            thinking = marked = False
        else:
            marked = thinking
        if marked and row in wanted:
            inside.add(row)
    if len(states) > 16:
        # Chunks only move forward; a re-plan of this chunk still finds `begin`.
        for key in [key for key in states if key < begin]:
            del states[key]
    states[begin + len(tokens)] = thinking
    return frozenset(inside)


def validate_configuration(profile, *, query_mode, optimized, pressure, single_assembly, original_rope):
    kv = profile.get('kvmem', {})
    if (query_mode != 'sequential' or not optimized or not pressure or
            not single_assembly or original_rope):
        raise ValueError('Layerwise rolling requires optimized sequential Query, per-layer pressure, '
                         'single assembly and compact positions')
    if not all(kv.get(key) for key in ('rolling_active_window', 'request_planning',
                                      'async_refresh', 'action_attention', 'prefill_attention')):
        raise ValueError('Layerwise rolling requires rolling request planning and accepted attention')
    if kv.get('pressure_policy') != 'attention' or kv.get('repository_gc', False):
        raise ValueError('Layerwise rolling requires attention pressure and retained CPU history')
    if profile.get('draft_method', 'mtp') != 'mtp' and not (
            profile.get('draft_method') == 'dflash' and profile.get('experimental_dflash_async_refresh')):
        raise ValueError('Layerwise rolling currently requires the MTP draft contract '
                         '(DFlash2 needs experimental_dflash_async_refresh)')
    # Opt-in score sources: reject malformed values at launch, not mid-run.
    from types import SimpleNamespace
    scope_options = SimpleNamespace(action_interval=kv.get('action_interval', 16),
                                    prefill_attention_stride=kv.get('prefill_attention_stride', 32))
    decode_scope(); decode_interval(scope_options)
    if prefill_fold_enabled():
        prefill_fold_weight(scope_options)
