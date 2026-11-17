"""CPU identity/state snapshots for frozen P-prefix layerwise pressure reuse.

This module does not publish a restore or write GPU pages. The session owner
must rebuild the prepared per-layer records and wait for its completion fence
before installing the maps/histories or admitting a public Query.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
import time

from .prefill_attention import PrefillHistory
from ..core import settings as _settings


def sessions_enabled():
    """KVMEM_LAYERWISE_SESSIONS: per-layer checkpoints of multi-turn sessions.

    Without it only frozen pre-Query P prefixes are captured (MemoryAgentBench).
    With it, non-frozen P prefixes and terminal M checkpoints also carry every
    layer's map, prefill attention history and (rolling) decode history.
    """
    value = _settings.get('KVMEM_LAYERWISE_SESSIONS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_SESSIONS must be 0 or 1')
    return value == '1'


def fast_state():
    """KVMEM_CHECKPOINT_FAST_STATE: O(history) checkpoint work without Python loops.

    Session captures and restores validated, copied, sorted and hashed every
    layer's map rows and one attention entry per history block in Python on the
    engine thread. With the flag a checkpoint keeps each layer history as its
    immutable native state, and the row checks, handle sets and evidence text
    run in the prefill-state extension. Results, error messages and evidence
    digests are unchanged; any input outside the plain shapes the extension
    handles takes the original code.
    """
    value = _settings.get('KVMEM_CHECKPOINT_FAST_STATE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_CHECKPOINT_FAST_STATE must be 0 or 1')
    return value == '1'


_NATIVE = []


def fast_native():
    """The prefill-state extension when KVMEM_CHECKPOINT_FAST_STATE can use it."""
    if not fast_state():
        return None
    if not _NATIVE:
        try:
            from .. import _prefill_state_native as module
        except ImportError:
            module = None
        if module is not None and not all(hasattr(module, name) for name in (
                'resolve_rows', 'same_identity', 'unique_handles', 'identity_json')):
            module = None
        if module is not None and not hasattr(module.State, 'keys_below'):
            module = None
        _NATIVE.append(module)
    return _NATIVE[0]


def fast_turn():
    """KVMEM_SESSION_FAST_TURN: per-turn session capture/restore work kept native.

    Needs KVMEM_CHECKPOINT_FAST_STATE. Every finished turn captures an M
    checkpoint and every next turn restores it, on the engine thread, in steps
    where the GPU waits. With the flag:

    * decode-history rows come straight from the native mass map
      (DecodeMassMap.sorted_rows) and are restored straight into one
      (DecodeMassMap.from_rows), without a dictionary of every attended block;
    * GC holds pass (slot, generation) tuples to the native scope
      (protect_pairs / protect_current_pairs) instead of one Handle object per
      record;
    * a capture computes the unique handle pairs of its maps once, for the hold
      and for the session ledger's protected-row count;
    * a capture resolves against the working records it already read.

    The checkpoint contents, the held records, errors and evidence are the
    same; inputs outside the plain shapes take the original code.
    """
    value = _settings.get('KVMEM_SESSION_FAST_TURN', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_SESSION_FAST_TURN must be 0 or 1')
    return value == '1' and fast_state()


def capture_index():
    """KVMEM_CHECKPOINT_CAPTURE_INDEX: per-capture O(history) Python work removed.

    Every session P/M capture runs on the engine thread. It rebuilt a
    {history_begin: record} dict over the whole archive for _resolve and turned
    each layer's native decode mass map into a tuple of (key, mass) Python
    tuples. With the flag the begin index is kept on the authority's record dict
    and extended for appended records (anything else rebuilds it), and decode
    rows stay native (MassRows) until something iterates them; a restore copies
    the map in key order natively instead of rebuilding it from rows. Needs
    KVMEM_SESSION_FAST_TURN; results and error messages are unchanged.
    """
    value = _settings.get('KVMEM_CHECKPOINT_CAPTURE_INDEX', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_CHECKPOINT_CAPTURE_INDEX must be 0 or 1')
    return value == '1' and fast_turn()


def records_by_begin(authority):
    """{record.history_begin: record for record in authority.records.values()}.

    With KVMEM_CHECKPOINT_CAPTURE_INDEX the dict is cached on the record dict
    (VersionedRecords.begin_index) and extended from its begin_log when only
    __setitem__ ran since: each logged key's record is added when its begin is
    new (or maps to the same object). A key whose begin already maps to another
    record, a missing key, or any other mutation (log reset to None) rebuilds
    the dict, so the result always equals the comprehension, including which
    record wins a duplicated begin.
    """
    current = authority.records
    if not capture_index() or not hasattr(current, 'begin_log'):
        return {record.history_begin: record for record in current.values()}
    cached, log = current.begin_index, current.begin_log
    if cached is not None:
        version, index = cached
        if version == current.version and log is not None and not log:
            return index
        if log is not None and len(log) == current.version - version:
            additions, ok = {}, True
            for key in log:
                record = dict.get(current, key)
                if record is None:
                    ok = False
                    break
                begin = record.history_begin
                old = additions.get(begin, index.get(begin))
                if old is not None and old is not record:
                    ok = False
                    break
                additions[begin] = record
            if ok:
                index.update(additions)
                current.begin_log = []
                current.begin_index = (current.version, index)
                return index
    index = {record.history_begin: record for record in current.values()}
    current.begin_log = []
    current.begin_index = (current.version, index)
    return index


class MassRows:
    """A rolling decode history's sorted (key, mass) rows, kept as its native map.

    Stands in for the row tuple inside LayerwisePrefix.decode
    (KVMEM_CHECKPOINT_CAPTURE_INDEX). Native mass maps are never mutated (every
    decode feedback builds a new one), so the map can be shared with the
    history it came from and with the history a restore builds; the tuple is
    built only when something iterates the rows.
    """
    __slots__ = ('map', '_rows')

    def __init__(self, mass_map):
        self.map, self._rows = mass_map, None

    def rows(self):
        if self._rows is None:
            self._rows = self.map.sorted_rows()
        return self._rows

    def max_key(self):
        return self.map.max_key()

    def __len__(self):
        return len(self.map)

    def __bool__(self):
        return len(self.map) > 0

    def __iter__(self):
        return iter(self.rows())

    def __getitem__(self, index):
        return self.rows()[index]

    def __eq__(self, other):
        if isinstance(other, MassRows):
            return self.rows() == other.rows()
        if isinstance(other, tuple):
            return self.rows() == other
        return NotImplemented

    def __hash__(self):
        return hash(self.rows())

    def __repr__(self):
        return f'MassRows({len(self)} rows)'


class StateRows:
    """A layer history's sorted (key, mass, exposure) rows, kept as its native state.

    Stands in for the row tuple inside LayerwisePrefix.histories; the tuple is
    built only when something iterates it.
    """
    __slots__ = ('state', '_rows')

    def __init__(self, state):
        self.state, self._rows = state, None

    def rows(self):
        if self._rows is None:
            self._rows = self.state.rows()
        return self._rows

    def __len__(self):
        return self.state.size()

    def __iter__(self):
        return iter(self.rows())

    def __getitem__(self, index):
        return self.rows()[index]

    def __eq__(self, other):
        if isinstance(other, StateRows):
            return self.rows() == other.rows()
        if isinstance(other, tuple):
            return self.rows() == other
        return NotImplemented

    def __hash__(self):
        return hash(self.rows())

    def __repr__(self):
        return f'StateRows({len(self)} rows)'


def _native_state(history):
    """The history's immutable native state (queued folds applied), or None."""
    state = getattr(history, '_buffered_state', None)
    return state if hasattr(state, 'keys_below') else None


def _same_scores(shared, history):
    """shared.snapshot() == history.snapshot()."""
    if fast_native() is not None:
        mine, theirs = _native_state(shared), _native_state(history)
        if mine is not None and theirs is not None:
            return ((shared.end, shared.samples, shared.chunks) ==
                    (history.end, history.samples, history.chunks) and mine.same_state(theirs))
    return shared.snapshot() == history.snapshot()


def _same_identity(first, second):
    """_identity(first) == _identity(second)."""
    native = fast_native()
    same = native.same_identity(first, second) if native is not None else None
    return _identity(first) == _identity(second) if same is None else same


def _restored_history(controller, options, snapshot, position):
    """A fresh layer history holding one LayerwisePrefix.histories entry."""
    rows, end, samples, chunks = snapshot
    if type(rows) is StateRows and type(position) is int and fast_native() is not None:
        if end > position or not rows.state.keys_below(position):
            raise ValueError('Layerwise restore cannot reuse future attention evidence')
        history = controller.new_history(options)
        from .buffered_prefill_attention import BufferedPrefillHistory
        if (isinstance(history, BufferedPrefillHistory) and
                type(history).restore is BufferedPrefillHistory.restore):
            # The state restore() would rebuild from the sorted rows, shared:
            # native states are immutable, a commit returns a new one.
            history._buffered_state = rows.state
            history.end, history.samples, history.chunks = end, samples, chunks
            history.successful_buffered_restores += 1
            return history
    else:
        if end > position or any(begin >= position for begin, _, _ in rows):
            raise ValueError('Layerwise restore cannot reuse future attention evidence')
        history = controller.new_history(options)
    history.restore(({begin: (mass, exposure) for begin, mass, exposure in rows},
                     end, samples, chunks), position)
    return history


def map_pairs(snapshot):
    """Every (slot, generation) of a checkpoint's maps, for its lifetime.

    The set session_worker._history_rows built from the rows on every poll;
    None when the fast path is off or the rows are not plain.
    """
    store = getattr(snapshot, '__dict__', None)
    cached = store.get('_map_pairs') if store is not None else None
    if cached is None:
        native = fast_native()
        if native is None or store is None:
            return None
        cached = native.unique_handles(snapshot.maps, False)
        if cached is None:
            return None
        object.__setattr__(snapshot, '_map_pairs', cached)
    return cached


def _maps_evidence(maps, yield_gil=False):
    """(layer_tokens, map_sha256) of LayerwisePrefix.evidence()."""
    native = fast_native()
    if native is not None:
        digest, tokens = hashlib.sha256(b'['), []
        for rows in maps:
            item = native.identity_json(rows)
            if item is None:
                break
            digest.update(b', ' + item[0] if tokens else item[0])
            tokens.append(item[1])
            if yield_gil:
                time.sleep(0)
        else:
            digest.update(b']')
            return tokens, digest.hexdigest()
    return ([sum(r[3] for r in rows) for rows in maps],
            _list_sha256((_identity(rows) for rows in maps), yield_gil))


def _attention_sha256(histories, yield_gil=False):
    """_list_sha256(histories); native-state rows are serialized by the state."""
    digest, first = hashlib.sha256(b'['), True
    for item in histories:
        rows = item[0] if item else None
        if type(rows) is StateRows:
            encode = getattr(rows.state, 'rows_json', None)
            digest.update(b'[' if first else b', [')
            digest.update(encode() if encode is not None else json.dumps(rows.rows()).encode())
            digest.update((''.join(', ' + json.dumps(value) for value in item[1:]) + ']').encode())
        else:
            digest.update((json.dumps(item) if first else ', ' + json.dumps(item)).encode())
        first = False
        if yield_gil:
            time.sleep(0)  # Let the engine thread take the GIL between layers.
    digest.update(b']')
    return digest.hexdigest()


@dataclass(frozen=True)
class LayerwisePrefix:
    position: int
    names: tuple
    maps: tuple
    histories: tuple
    # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: native policy hold on every
    # record of ``maps`` for this checkpoint's lifetime (dropped with it).
    gc_hold: object = field(default=None, compare=False, repr=False)
    # Opt-in KVMEM_LAYERWISE_SESSIONS: rolling-decode attention per layer,
    # ((key, mass) rows, samples, end, chunks), or None when it never ran.
    decode: tuple = None
    terminal: bool = False

    @property
    def bytes(self):
        # Serialized integer identities and FP64 mass/exposure, matching the
        # checkpoint pool's explicit payload accounting (not Python overhead).
        return 8 + sum(len(rows) * 40 for rows in self.maps) + sum(
            24 + len(snapshot[0]) * 24 for snapshot in self.histories) + sum(
            32 + len(snapshot[0]) * 16 for snapshot in (self.decode or ()))

    def evidence(self):
        if any(type(snapshot[0]) is StateRows for snapshot in self.histories):
            return self.evidence_resolver()()
        return dict(accepted_history=self.position, layer_names=self.names,
            layer_tokens=[sum(r[3] for r in rows) for rows in self.maps],
            # Handles can change on rewind. Identity is original text plus its
            # dense destination, not an implementation-specific archive slot.
            map_sha256=hashlib.sha256(json.dumps([_identity(rows) for rows in self.maps]).encode()).hexdigest(),
            attention_sha256=hashlib.sha256(json.dumps(self.histories).encode()).hexdigest())

    def evidence_resolver(self):
        """evidence() computed when called; holds the immutable maps/histories only."""
        position, names, maps, histories = self.position, self.names, self.maps, self.histories
        def resolve(yield_gil=False):
            if fast_state():
                tokens, map_sha256 = _maps_evidence(maps, yield_gil)
                return dict(accepted_history=position, layer_names=names, layer_tokens=tokens,
                    map_sha256=map_sha256, attention_sha256=_attention_sha256(histories, yield_gil))
            return dict(accepted_history=position, layer_names=names,
                layer_tokens=[sum(r[3] for r in rows) for rows in maps],
                map_sha256=_list_sha256((_identity(rows) for rows in maps), yield_gil),
                attention_sha256=_attention_sha256(histories, yield_gil))
        return resolve


def _identity(rows):
    return tuple(tuple(row[2:]) for row in rows)


def _list_sha256(items, yield_gil=False):
    """sha256(json.dumps(list(items))) hashed one item at a time."""
    digest, first = hashlib.sha256(b'['), True
    for item in items:
        digest.update((json.dumps(item) if first else ', ' + json.dumps(item)).encode())
        first = False
        if yield_gil:
            time.sleep(0)  # Let the engine thread take the GIL between layers.
    digest.update(b']')
    return digest.hexdigest()


def lazy_evidence():
    """KVMEM_CHECKPOINT_LAZY_EVIDENCE: hash checkpoint evidence off the engine step.

    The capture/restore row keeps its timestamps, bytes and position in
    layerwise_query.jsonl; its digests are computed by the deferred evidence
    writer (or at the next synchronous emit) instead of before the step runs.
    """
    value = _settings.get('KVMEM_CHECKPOINT_LAZY_EVIDENCE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_CHECKPOINT_LAZY_EVIDENCE must be 0 or 1')
    return value == '1'


def emit_prefix_evidence(query, event, snapshot, **fields):
    if lazy_evidence():
        query.defer_lazy(event, snapshot.evidence_resolver(), **fields)
    else:
        query.emit(event, **fields, **snapshot.evidence())


def layer0_diverged(shared, layer_zero):
    """Whether layer zero's own window differs from the native (shared) frame.

    A probe-mode final Query selects each target layer with its own same-forward
    Query while the native frame (and the MTP draft) keep the global selection;
    the next pressure realigns them. Session checkpoints may be taken in between
    (KVMEM_LAYERWISE_SESSIONS). Lengths and partial-block geometry must agree.
    """
    if _same_identity(shared, layer_zero):
        return False
    partial = lambda rows: tuple((r[2], r[3], r[4]) for r in rows if r[3] != 32)
    if (sum(r[3] for r in shared) != sum(r[3] for r in layer_zero) or
            partial(shared) != partial(layer_zero)):
        raise ValueError('Layerwise checkpoint differs from the native layer-zero frame')
    return True


def _resolve(authority, maps, position, *, planned_records=None, allow_layer0_divergence=False,
             want_chosen=True):
    """Resolve original coordinates against the retained immutable archive."""
    by_begin = records_by_begin(authority)
    native = fast_native()
    if native is not None:
        block_tokens = getattr(getattr(getattr(authority, 'memory', None), 'config', None),
                               'block_tokens', None)
        fast = (native.resolve_rows(maps, by_begin, position, block_tokens, want_chosen)
                if type(position) is int and type(block_tokens) is int else None)
        if fast is not None:            # same checks and errors as the loop below
            resolved, canonical = fast
            shared = (tuple(authority.memory._store.working_records()) if planned_records is None
                      else tuple(planned_records))
            if not _same_identity(shared, canonical[0]):
                if not allow_layer0_divergence:
                    raise ValueError('Layerwise checkpoint differs from the native layer-zero frame')
                layer0_diverged(shared, canonical[0])
            return resolved, canonical
    resolved, canonical = [], []
    for rows in maps:
        chosen, mapped, seen, cursor = [], [], set(), 0
        for _, _, begin, count, dense in rows:
            if (begin in seen or dense != cursor or begin < 0 or count <= 0 or
                    begin + count > position):
                raise ValueError('Invalid layerwise checkpoint coordinates')
            record = by_begin.get(begin)
            if record is None or record.count != count:
                raise ValueError('Layerwise checkpoint source was not retained')
            chosen.append(record)
            mapped.append((record.handle.slot, record.handle.generation, begin, count, dense))
            seen.add(begin)
            cursor += count
        resolved.append(tuple(chosen))
        canonical.append(tuple(mapped))
    if not canonical or len({sum(r[3] for r in rows) for rows in canonical}) != 1:
        raise ValueError('Layerwise checkpoint has inconsistent physical lengths')
    partial = [tuple((r[2], r[3], r[4]) for r in rows if r[3] != authority.memory.config.block_tokens)
               for rows in canonical]
    if any(rows != partial[0] for rows in partial):
        raise ValueError('Layerwise checkpoint has inconsistent partial-block geometry')
    shared = (tuple(authority.memory._store.working_records()) if planned_records is None
              else tuple(planned_records))
    if _identity(shared) != _identity(canonical[0]):
        if not allow_layer0_divergence:
            raise ValueError('Layerwise checkpoint differs from the native layer-zero frame')
        layer0_diverged(shared, canonical[0])
    return tuple(resolved), tuple(canonical)


MAX_RETIRED = 64


def retire_state(controller, request_id, authority=None):
    """Keep a finished request's per-layer state for its terminal capture.

    The native finish runs before the session's terminal (M) capture. The
    resolved maps, prefill and decode histories move out of the live tables
    into a bounded stash; the request's window hold stays until the capture
    takes its own checkpoint hold (or the stash entry is dropped).
    """
    from .layerwise_gc_holds import holds_of
    holds = holds_of(controller)
    frame = controller.resolved_maps(request_id) if request_id in controller.maps else None
    controller.maps.pop(request_id, None)
    histories = controller.histories.pop(request_id, None)
    decode = controller.decode_histories.pop(request_id, None)
    from .layerwise_pressure import _drop_record_index
    _drop_record_index(controller, request_id, authority)
    owners = controller.__dict__.get('_history_owners', {})
    if histories is None:
        holds.release(request_id)
        owners.pop(request_id, None)
        return
    holds.retired[request_id] = (frame, histories, decode)
    while len(holds.retired) > MAX_RETIRED:
        oldest = next(iter(holds.retired))
        holds.retired.pop(oldest)
        holds.release(oldest)
        owners.pop(oldest, None)


def _consume_retired(controller, request_id):
    from .layerwise_gc_holds import holds_of
    holds = holds_of(controller)
    if holds.retired.pop(request_id, None) is not None:
        holds.release(request_id)
    controller.__dict__.get('_history_owners', {}).pop(request_id, None)


def capture(controller, request_id, authority, position, *, terminal=False, committed=False):
    """Per-layer snapshot; a finished request's stashed state is used if live state is gone."""
    reinstated = False
    if sessions_enabled() and request_id not in controller.histories:
        from .layerwise_gc_holds import holds_of
        stash = holds_of(controller).retired.get(request_id)
        if stash is not None:
            frame, histories, decode = stash
            if frame is not None:
                controller.maps[request_id] = frame
            controller.histories[request_id] = histories
            if decode is not None:
                controller.decode_histories[request_id] = decode
            reinstated = True
    early = False
    if fast_state():
        # checkpoint_hold waits for the GC worker to acknowledge its pause;
        # asked for here, the acknowledgement arrives while the rows are built.
        from .layerwise_gc_holds import pause, resume
        early = pause(authority, key='checkpoint')
    try:
        return _capture(controller, request_id, authority, position, terminal=terminal, committed=committed)
    finally:
        if early:
            resume(authority, key='checkpoint')
        if reinstated:
            controller.maps.pop(request_id, None)
            controller.histories.pop(request_id, None)
            controller.decode_histories.pop(request_id, None)


def _capture(controller, request_id, authority, position, *, terminal=False, committed=False):
    if position != authority.history_end or position <= 0:
        raise ValueError('Layerwise checkpoint must use the exact accepted boundary')
    if (terminal or committed) and not sessions_enabled():
        raise ValueError('Layerwise session checkpoints require KVMEM_LAYERWISE_SESSIONS')
    if not (terminal or committed) and request_id in controller.query.finished:
        raise ValueError('Layerwise P checkpoint must precede the final Query selection')
    histories = controller.histories.get(request_id)
    if histories is None or len(histories) != len(controller.names):
        raise ValueError('Missing independent cumulative attention histories')
    shared = tuple(authority.memory._store.working_records())
    extent = sum(r[3] for r in shared)
    maps = tuple(tuple(controller.records(request_id, authority, position, extent, i))
                 for i in range(len(controller.names)))
    quick = fast_turn()
    # KVMEM_SESSION_FAST_TURN: _resolve compares against the working records
    # read above instead of reading them again (nothing ran in between).
    _, maps = _resolve(authority, maps, position, planned_records=shared if quick else None,
                       allow_layer0_divergence=sessions_enabled(), want_chosen=False)
    snapshots = []
    fast = fast_native() is not None and type(position) is int
    for history in histories:
        state = _native_state(history) if fast else None
        if state is not None:
            end, samples, chunks = history.end, history.samples, history.chunks
            if end > position or not state.keys_below(position):
                raise ValueError('Layerwise checkpoint contains future attention evidence')
            snapshots.append((StateRows(state if state.is_sorted() else state.sorted()),
                              end, samples, chunks))
            continue
        values, end, samples, chunks = history.snapshot()
        if end > position or any(begin >= position for begin in values):
            raise ValueError('Layerwise checkpoint contains future attention evidence')
        snapshots.append((tuple((key, mass, exposure) for key, (mass, exposure)
                                in sorted(values.items())), end, samples, chunks))
    if not _same_scores(authority.action_history.prefill, histories[0]):
        raise ValueError('Layerwise checkpoint differs from shared layer-zero scores')
    decode = None
    decode_histories = controller.decode_histories.get(request_id)
    if decode_histories is not None:
        if not terminal and not sessions_enabled():
            raise ValueError('A frozen P checkpoint cannot carry decode attention')
        if len(decode_histories) != len(controller.names):
            raise ValueError('Missing per-layer rolling decode histories')
        decode = _decode_snapshot(decode_histories, position)
    from .layerwise_gc_holds import checkpoint_hold
    # A diverged layer zero leaves the native frame (restored into the MTP draft
    # and, before the per-layer rebuild, every layer) outside the layer maps.
    diverged = not _same_identity(shared, maps[0])
    held = maps + ((shared,) if diverged else ())
    every = pairs = None
    if quick and not diverged:
        every, pairs = _capture_pairs(authority, maps)
    prefix = LayerwisePrefix(position, tuple(controller.names), maps, tuple(snapshots),
                             gc_hold=checkpoint_hold(authority, held, pairs), decode=decode, terminal=terminal)
    if every is not None:
        object.__setattr__(prefix, '_map_pairs', every)     # what map_pairs() would compute
    if decode is not None and fast_state():
        # Rows built just above: sorted (begin, mass) tuples with unique begins.
        object.__setattr__(prefix, '_decode_sorted', True)
    return prefix


def _capture_pairs(authority, maps):
    """(map_pairs value, checkpoint_hold handles) of a capture's maps, or (None, None).

    One native pass over the rows (KVMEM_SESSION_FAST_TURN): the sorted unique
    (slot, generation) of every row, and the same list without the (-1, -1)
    placeholder, which is what window_handles(maps) returns.
    """
    from .layerwise_gc_holds import scope_of
    native = fast_native()
    if native is None or scope_of(authority) is None:
        return None, None
    every = native.unique_handles(maps, False)
    if every is None:
        return None, None
    from bisect import bisect_left
    index = bisect_left(every, (-1, -1))
    if index < len(every) and every[index] == (-1, -1):
        return every, every[:index] + every[index + 1:]
    return every, every


def _native_decode_rows(history):
    """sorted(history.mass.items()) from a NativeDecodeHistory's map, cached on it."""
    from .layerwise_decode import NativeDecodeHistory
    if type(history) is not NativeDecodeHistory:
        return None
    rows = history.__dict__.get('_sorted_rows')
    if rows is None:
        if capture_index() and hasattr(history._map, 'sorted_copy'):
            # KVMEM_CHECKPOINT_CAPTURE_INDEX: the same rows, built on demand.
            rows = history._sorted_rows = MassRows(history._map)
            return rows
        sorted_rows = getattr(history._map, 'sorted_rows', None)
        if sorted_rows is None:
            return None
        rows = history._sorted_rows = sorted_rows()       # instances are never mutated
    return rows


def _native_restored_decode(decode):
    """Checkpoint decode rows restored straight into native mass maps, or None.

    Equal to DecodeHistory(dict(rows), ...) followed by the conversion its
    first prepare() performs (DecodeMassMap.from_dict).
    """
    from . import layerwise_decode
    from_rows = getattr(getattr(layerwise_decode._native, 'DecodeMassMap', None), 'from_rows', None)
    if from_rows is None or not layerwise_decode.mass_map_enabled():
        return None
    histories = []
    for rows, samples, end, chunks in decode:
        # MassRows (KVMEM_CHECKPOINT_CAPTURE_INDEX): the map from_rows would
        # rebuild from its rows (entries in key order), copied natively.
        mass_map = rows.map.sorted_copy() if type(rows) is MassRows else from_rows(rows)
        if mass_map is None:
            return None
        history = layerwise_decode.NativeDecodeHistory(mass_map, samples, end, chunks)
        history._sorted_rows = rows                           # the next capture's rows
        histories.append(history)
    return histories


def _decode_snapshot(decode_histories, position):
    decode = []
    quick = fast_state() and type(position) is int
    for history in decode_histories:
        rows = _decode_rows(history, position) if quick else None
        if rows is None:
            if history.end > position or any(begin >= position for begin in history.mass):
                raise ValueError('Layerwise checkpoint contains future decode attention')
            rows = tuple(sorted(history.mass.items()))
        decode.append((rows, history.samples, history.end, history.chunks))
    return tuple(decode)


def _decode_rows(history, position):
    """Sorted (begin, mass) rows of a decode history (KVMEM_CHECKPOINT_FAST_STATE).

    A history restored from a checkpoint this process captured and not replaced
    since (decode feedback and pruning create new objects) still equals the
    rows it was built from. Otherwise the future-evidence check uses max();
    None leaves the original expressions (and their errors) to the caller.
    """
    if history.end > position:
        raise ValueError('Layerwise checkpoint contains future decode attention')
    if fast_turn():
        rows = _native_decode_rows(history)
        if rows is not None:
            last = rows.max_key() if type(rows) is MassRows else (rows[-1][0] if rows else None)
            if last is not None and last >= position:
                raise ValueError('Layerwise checkpoint contains future decode attention')
            return rows
    source = getattr(history, '__dict__', {}).get('_checkpoint_rows')
    mass = history.mass
    if source is not None and source[0] is mass and len(mass) == len(source[1]):
        rows = source[1]
        if rows and rows[-1][0] >= position:
            raise ValueError('Layerwise checkpoint contains future decode attention')
        return rows
    if type(mass) is not dict:
        return None
    try:
        late = bool(mass) and max(mass) >= position
    except TypeError:
        return None
    if late:
        raise ValueError('Layerwise checkpoint contains future decode attention')
    return tuple(sorted(mass.items()))


def prepare_restore(controller, request_id, authority, snapshot, options, *, planned_records=None):
    """Validate all CPU state without publishing partially restored owners."""
    if snapshot is None:
        raise ValueError('Layerwise restore requires an explicit per-layer checkpoint')
    if (tuple(controller.names) != snapshot.names or
            len(snapshot.maps) != len(snapshot.names) or
            len(snapshot.histories) != len(snapshot.names)):
        raise ValueError('Layerwise checkpoint layer order or count changed')
    if authority.history_end != snapshot.position:
        raise ValueError('Layerwise restore must first rewind to its exact P prefix')
    if (request_id in controller.maps or request_id in controller.histories or
            request_id in controller.query.finished):
        raise ValueError('Layerwise restore requires a fresh native request owner')
    chosen, maps = _resolve(authority, snapshot.maps, snapshot.position, planned_records=planned_records,
                            allow_layer0_divergence=sessions_enabled())
    histories = []
    for entry in snapshot.histories:
        histories.append(_restored_history(controller, options, entry, snapshot.position))
    reconcile_pruned(authority, histories)
    return chosen, (snapshot.position, maps), histories


def reconcile_pruned(authority, histories):
    """Apply a retirement prune that the restoring request's histories missed.

    Historical GC prunes the shared prefill history as soon as a retirement
    drains (RequestAuthority._prune_history_metadata), and with it the layer
    histories of bound requests. A restoring request is bound only once its
    state is installed, so a drain between the checkpoint claim and that point
    (e.g. authority.poll() inside the draft rebuild's packed read) pruned the
    shared history alone. The same rule is applied here; any other difference
    is still an error. Returns the live set when a prune was applied.
    """
    shared = authority.action_history.prefill
    if _same_scores(shared, histories[0]):
        return None
    live = getattr(authority, 'history_live', None)
    live = live() if live is not None else None
    if live is not None:
        end = authority.history_end
        for history in histories:
            values = history.values
            if any(key not in live and key < end for key in values):
                history.values = {k: v for k, v in values.items() if k in live or k >= end}
    if not _same_scores(shared, histories[0]):
        mine, theirs = set(histories[0].values), set(shared.values)
        raise ValueError('Restored layer-zero scores differ from the shared checkpoint '
                         f'(layer-only keys {len(mine - theirs)}, shared-only keys {len(theirs - mine)})')
    return live


def restored_decode(snapshot):
    """Per-layer rolling decode histories of a session checkpoint, or None."""
    if snapshot.decode is None:
        return None
    from .layerwise_decode import DecodeHistory
    if len(snapshot.decode) != len(snapshot.names):
        raise ValueError('Layerwise checkpoint decode history count changed')
    if getattr(snapshot, '__dict__', {}).get('_decode_sorted') and fast_turn():
        histories = _native_restored_decode(snapshot.decode)
        if histories is not None:
            return histories
    histories = [DecodeHistory(dict(rows), samples, end, chunks)
                 for rows, samples, end, chunks in snapshot.decode]
    if getattr(snapshot, '__dict__', {}).get('_decode_sorted') and fast_state():
        # The next capture of an unchanged history returns these rows again.
        for history, item in zip(histories, snapshot.decode):
            if type(item[0]) is tuple and len(history.mass) == len(item[0]):
                history._checkpoint_rows = (history.mass, item[0])
    return histories


def install_restored(controller, request_id, authority, frame, histories, snapshot):
    """Publish restored per-layer CPU state after every GPU write completed."""
    bind = getattr(controller, 'bind_history_owner', None)
    if bind is not None:
        bind(request_id, authority)
    controller.maps[request_id] = frame
    controller.histories[request_id] = histories
    decode = restored_decode(snapshot)
    if decode is not None:
        controller.decode_histories[request_id] = decode
    # GPU writes since prepare_restore (per-layer and draft assembly) poll the
    # authority; a retirement drained there pruned the shared history only.
    live = reconcile_pruned(authority, histories)
    prune = getattr(controller, 'prune_layer_histories', None)
    if live is not None and prune is not None:
        prune(authority, live)             # same rule for the decode histories
    # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: hold every restored layer window.
    # The selection pause first excludes an in-flight retirement decision.
    from .layerwise_gc_holds import holds_of, wait_paused
    wait_paused(authority)
    holds_of(controller).protect(request_id, authority, frame[1])


def capture_for_pool(adapter, request_id, authority, position, *, terminal=False, committed=False):
    runner = getattr(adapter, 'runner', None)
    controller = getattr(runner, '_kvmem_layerwise_pressure', None)
    if controller is None:
        return None
    worker = getattr(runner, '_kvmem_sessions', None)
    options = getattr(worker, 'options', None)
    if not sessions_enabled() and (terminal or not getattr(options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise checkpoints currently require frozen pre-Query P prefixes')
    # committed: a deferred P capture taken after a selection committed at its
    # position (KVMEM_DEFERRED_P_CAPTURE), as refresh_committed_snapshot does.
    snapshot = capture(controller, request_id, authority, position, terminal=terminal,
                       **({'committed': True} if committed else {}))
    if terminal and sessions_enabled():
        # The checkpoint now holds its records; the finished request's stash goes.
        _consume_retired(controller, request_id)
    return snapshot


def _rebuild_draft(controller, state, records):
    """Rebuild the MTP draft layers from layer zero's window; returns the count.

    A probe-mode final Query leaves the draft (and the native frame) on the
    global selection while layer zero has its own. A session restore makes the
    new turn start aligned: draft and native frame follow layer zero.
    """
    runner = getattr(controller, 'runner', None)
    names = tuple(getattr(runner, '_kvmem_archive_layers', controller.names))
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Draft rebuild changed the target archive layer order')
    count = 0
    for index, name in enumerate(names[len(controller.names):], len(controller.names)):
        controller.query.assemble_layer_pinned(state, index, records, layer_name=name)
        count += 1
    return count


def realign_after_restore(worker, request_id, checkpoint):
    """KVMEM_LAYERWISE_SESSIONS: after a restore is published, commit layer
    zero's window as the native frame (bookkeeping only; its pages, and the
    draft's, already hold it) and let the retained checkpoint adopt it.

    The shared attention state, fast metadata and native skip hints all
    describe the native frame, so the new turn must not start diverged.
    """
    controller = getattr(getattr(worker, 'runner', None), '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None or snapshot is None or not sessions_enabled():
        return False
    frame = controller.maps.get(request_id)
    if frame is None or isinstance(frame, PendingMapsType()) or worker.bindings[request_id].pressure_restore:
        return False
    authority = checkpoint.authority
    from .tensor_parallel import agree_flag
    if not agree_flag(layer0_diverged(tuple(authority.memory._store.working_records()), frame[1][0]),
                      f'layer-zero realign for {request_id}'):
        return False
    from .layerwise_no_probe import commit_layer_zero
    records = [authority.records[(row[0], row[1])] for row in frame[1][0]]
    commit_layer_zero(controller, dict(authority=authority, layer_zero_records=records,
                                       selected_tokens=sum(r.count for r in records)))
    worker.selection_committed(request_id, checkpoint.position)
    controller.query.emit('pressure_prefix_realigned', request_id=request_id,
                          accepted_history=checkpoint.position)
    return True


def query_frame_committed(runner, request_id):
    """KVMEM_LAYERWISE_SESSIONS: a committed Query (non-pressure) selection
    replaced the whole native frame. Until the per-layer Query completes, every
    layer resumes from that frame, so earlier per-layer maps are stale; drop
    them before a P checkpoint at this boundary re-takes its snapshot.
    """
    controller = getattr(runner, '_kvmem_layerwise_pressure', None)
    if controller is None or not sessions_enabled():
        return False
    frame = controller.maps.pop(request_id, None)
    if isinstance(frame, PendingMapsType()):
        frame.deferred.discard()
    return frame is not None


def PendingMapsType():
    from .layerwise_pressure import PendingMaps
    return PendingMaps


def restore_sync(worker, request_id, checkpoint):
    """Rebuild distinct target-layer pages before publishing a synchronous P hit.

    Native restore already rebuilt layer zero and the draft. Reuse the existing
    per-layer archive assembly for the remaining target layers. CPU ownership is
    installed only after every GPU write completes; failures never publish a hit.
    """
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None:
        raise ValueError('Layerwise prefix cannot be restored by a shared-pressure worker')
    if worker.bindings[request_id].pressure_restore:
        raise ValueError('A compacting layerwise restore uses restore_pressure_sync')
    if not sessions_enabled() and (checkpoint.kind != 'P' or
            not getattr(worker.options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise restore requires the original frozen P window')
    chosen, frame, histories = prepare_restore(controller, request_id,
        checkpoint.authority, snapshot, worker.options)
    # Synchronous restore and causal Query both run on the model thread.
    # Reuse their one-layer arena; background pressure assembly can overlap
    # this restore and must retain its separate scratch/DMA ownership.
    state = dict(authority=checkpoint.authority, request_id=request_id,
        selected_tokens=sum(r.count for r in chosen[0]), assembly_arena='layerwise')
    # Native restore wrote its own (shared) frame; a diverged layer zero
    # (session checkpoint after a probe-mode Query) is rebuilt like the others.
    diverged = sessions_enabled() and layer0_diverged(
        tuple(checkpoint.authority.memory._store.working_records()), frame[1][0])
    first = 0 if diverged else 1
    for layer, records in enumerate(chosen[first:], first):
        controller.query.assemble_layer_pinned(state, layer, records)
    if diverged:
        _rebuild_draft(controller, state, chosen[0])
    import torch
    torch.cuda.current_stream(worker.runner.device).synchronize()
    install_restored(controller, request_id, checkpoint.authority, frame, histories, snapshot)
    emit_prefix_evidence(controller.query, 'pressure_prefix_restored', snapshot, request_id=request_id,
        actual_method='accumulated_attention_per_layer', fallback_reason=None)


def require_supported_restore(worker, request_id, checkpoint, *, asynchronous=False):
    """Reject unqualified paths before modifying any restore destination."""
    controller = getattr(getattr(worker, 'runner', None), '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None or snapshot is None:
        raise ValueError('Layerwise pressure restore is missing its matching checkpoint/controller')
    if sessions_enabled():
        # Session checkpoints (P or M; copy, resident or compacting). An
        # asynchronous import finishes its per-layer part on the model thread
        # (finish_async_restore) before the restore is published.
        if checkpoint.kind not in ('P', 'M'):
            raise ValueError('Unknown layerwise checkpoint kind')
        return
    if (asynchronous or checkpoint.kind != 'P' or
            worker.bindings[request_id].pressure_restore or
            not getattr(worker.options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise prefix requires synchronous frozen P restore')


def oracle_record_groups(worker, request_id, authority):
    """Group equal frames for independent full-byte restore verification."""
    names = tuple(worker.runtime.names)
    shared = tuple(authority.memory._store.working_records())
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    if controller is None:
        return ((shared, tuple(range(len(names)))),)
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Layerwise restore oracle changed target archive order')
    frame = controller.maps.get(request_id)
    if frame is None or frame[0] != authority.history_end:
        raise ValueError('Layerwise restore oracle needs the exact restored frame')
    _, maps = _resolve(authority, frame[1], frame[0], allow_layer0_divergence=sessions_enabled())
    # The MTP draft follows layer zero (a diverged session restore rebuilds it so).
    maps += (maps[0],) * (len(names) - len(maps))
    groups = {}
    for index, records in enumerate(maps):
        groups.setdefault(records, []).append(index)
    return tuple((records, tuple(indices)) for records, indices in groups.items())


def refresh_committed_snapshot(worker, request_id, checkpoint):
    """A selection committed exactly at a live P checkpoint (non-frozen sessions).

    The pool adopts the newly committed shared window for that P; the per-layer
    snapshot is re-taken at the same position so its maps stay the windows the
    restored turn will actually resume with.
    """
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    if controller is None or getattr(checkpoint, 'layerwise_pressure', None) is None:
        return
    if not sessions_enabled():
        raise ValueError('Frozen layerwise P checkpoint cannot adopt a later shared window')
    checkpoint.layerwise_pressure = capture(controller, request_id, checkpoint.authority,
                                            checkpoint.position, committed=True)


def restore_pressure_sync(worker, request_id, checkpoint):
    """Compacting session restore (KVMEM_LAYERWISE_SESSIONS).

    The native compacting import already wrote one shared, budget-sized window
    into every layer. Every layer resumes from that shared frame (no per-layer
    map), with its own restored prefill and decode attention histories; the
    next layerwise pressure refresh selects each layer separately again.
    """
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None or snapshot is None or not sessions_enabled():
        raise ValueError('Compacting layerwise restore requires KVMEM_LAYERWISE_SESSIONS and its checkpoint')
    authority = checkpoint.authority
    if (tuple(controller.names) != snapshot.names or len(snapshot.histories) != len(snapshot.names)):
        raise ValueError('Layerwise checkpoint layer order or count changed')
    if authority.history_end != snapshot.position:
        raise ValueError('Layerwise restore must first rewind to its exact checkpoint prefix')
    if (request_id in controller.maps or request_id in controller.histories or
            request_id in controller.query.finished):
        raise ValueError('Layerwise restore requires a fresh native request owner')
    histories = []
    for entry in snapshot.histories:
        histories.append(_restored_history(controller, worker.options, entry, snapshot.position))
    live = reconcile_pruned(authority, histories)
    bind = getattr(controller, 'bind_history_owner', None)
    if bind is not None:
        bind(request_id, authority)
    controller.histories[request_id] = histories
    decode = restored_decode(snapshot)
    if decode is not None:
        controller.decode_histories[request_id] = decode
    prune = getattr(controller, 'prune_layer_histories', None)
    if live is not None and prune is not None:
        prune(authority, live)             # same rule for the decode histories
    shared = tuple(authority.memory._store.working_records())
    from .layerwise_gc_holds import holds_of, wait_paused
    wait_paused(authority)
    holds_of(controller).protect(request_id, authority, [shared])
    controller.query.emit('pressure_prefix_restored', request_id=request_id,
        actual_method='shared_compacted_frame_per_layer_histories', fallback_reason=None,
        accepted_history=snapshot.position, selected_tokens=sum(r[3] for r in shared))


def finish_async_restore(worker, request_id, checkpoint, *, resident, pressure=None, assembler=None):
    """Per-layer completion of an asynchronous session import, on the model thread.

    The background import wrote one shared window into every layer (or, for a
    resident import, left the terminal pages untouched). Before publication:
    a compacting import resumes from that shared frame; a resident one already
    holds every layer's terminal window, so only CPU state is installed; a copy
    import rebuilds target layers 1..N from the snapshot like restore_sync.
    """
    controller = getattr(getattr(worker, 'runner', None), '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if pressure is None:
        pressure = getattr(worker.bindings[request_id], 'pressure_restore', False)
    if pressure:
        restore_pressure_sync(worker, request_id, checkpoint)
        return
    if resident:
        if controller is None or snapshot is None or not sessions_enabled():
            raise ValueError('Resident layerwise import requires KVMEM_LAYERWISE_SESSIONS and its checkpoint')
        early = False
        if fast_state():
            # install_restored waits for the GC worker to acknowledge the
            # selection pause; asked for here, it arrives during the checks.
            from .layerwise_gc_holds import pause, resume
            early = pause(checkpoint.authority)
        try:
            chosen, frame, histories = prepare_restore(controller, request_id,
                checkpoint.authority, snapshot, worker.options)
            if layer0_diverged(tuple(checkpoint.authority.memory._store.working_records()), frame[1][0]):
                state = dict(authority=checkpoint.authority, request_id=request_id,
                    selected_tokens=sum(r.count for r in chosen[0]), assembly_arena='layerwise')
                if _rebuild_draft(controller, state, chosen[0]):
                    import torch
                    torch.cuda.current_stream(worker.runner.device).synchronize()
                    # Verification: the draft no longer equals the terminal pages.
                    worker.__dict__.setdefault('layerwise_draft_rebuilt', set()).add(request_id)
            install_restored(controller, request_id, checkpoint.authority, frame, histories, snapshot)
        except BaseException:
            if early:
                resume(checkpoint.authority)    # install's window hold never released it
            raise
        from .pressure_key_reuse import restored as _reuse_restored
        _reuse_restored(controller, request_id, resident=True)
        emit_prefix_evidence(controller.query, 'pressure_prefix_restored', snapshot, request_id=request_id,
            actual_method='resident_terminal_pages_per_layer', fallback_reason=None)
        return
    from .layerwise_restore_efficiency import finish_direct_session_restore
    if finish_direct_session_restore(worker, request_id, checkpoint, assembler):
        return      # opt-in KVMEM_RESTORE_DIRECT_LAYERS: every layer already written
    restore_sync(worker, request_id, checkpoint)
