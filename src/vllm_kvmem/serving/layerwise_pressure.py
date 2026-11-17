"""Opt-in experiment: independent accepted prefill attention per owner/layer.

The native shared working frame represents layer zero only. Other layers keep
their own logical block maps; newly appended records still share original token
coordinates. No cold blocks enter a pressure candidate set. Final Query keeps
the existing causal per-layer Mean-K implementation.
"""
from copy import copy
import math
import os
import time

import torch

from .action_attention import ActionRuntime, window_segments
from .prefill_attention import PrefillHistory
from ..adapters.original_rope import enabled as original_rope
from ..core import settings as _settings
from ..core.block_runs import encode_starts
from .tensor_parallel import active as _tp_active


def _decode_labels():
    from .layerwise_decode import query_source_label
    return query_source_label()


class _LayerwiseAcceptance:
    """Publish per-layer histories only with the accepted graph transaction."""
    def __init__(self, transaction, controller, staged):
        self.transaction, self.controller, self.staged = transaction, controller, staged

    def __getattr__(self, name):
        return getattr(self.transaction, name)

    def commit(self):
        self.transaction.commit()
        self.controller.commit_accept(self.staged)
        self.staged = None

    def abort(self):
        self.transaction.abort()
        self.staged = None


def extend_records(records, origin, end, block_tokens=32):
    """Extend a frozen layer map with accepted ordinary input, without handles."""
    if end < origin:
        raise ValueError('Layer map cannot move back across its pressure boundary')
    result = list(records)
    cursor = sum(r[3] for r in result)
    logical = origin
    while logical < end:
        n = min(block_tokens-logical % block_tokens, end-logical)
        if (result and result[-1][2]+result[-1][3] == logical and
                result[-1][4]+result[-1][3] == cursor and
                result[-1][2]//block_tokens == logical//block_tokens):
            slot, gen, begin, size, pos = result[-1]
            result[-1] = slot, gen, begin, size+n, pos
        else:
            result.append((-1, -1, logical, n, cursor))
        cursor += n
        logical += n
    return result


class _Release:
    """Release an already acquired lock on exit (DeferredPreparedCommit.value)."""
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        return self.lock

    def __exit__(self, *exc):
        self.lock.release()
        return False


class DeferredPreparedCommit:
    """Thread-safe one-shot builder for a PreparedPressureCommit.

    The producer builds it after its per-layer upload loop; any consumer that
    needs the maps earlier builds it itself. Errors are re-raised to every caller.

    With layer-0-first (lazy) selection, ``selected`` is set once the producer
    has selected every layer. A consumer can reach the maps before that: with
    KVMEM_LAYER0_FAST no layer guard holds the next forward, so its capture can
    ask for them while later layers are still being selected. It waits for the
    selection (holding no lock) instead of building from a partial plan.
    """
    SELECTION_TIMEOUT = 120.

    def __init__(self, build, layer_zero_starts, selected=None):
        import threading
        self._build, self._lock = build, threading.Lock()
        self._value = self._error = None
        self.layer_zero_starts = layer_zero_starts
        self._selected = selected

    @property
    def ready(self):
        return self._value is not None

    def release(self, error):
        """The producer stopped before selecting every layer: fail, never block, consumers."""
        with self._lock:
            if self._value is None and self._error is None:
                self._error = error
        if self._selected is not None:
            self._selected.set()

    def discard(self):
        """The owner finished before anyone needed these maps: drop the builder.

        Its closure references the job (and through it the plan, histories and
        source rights); dropping it breaks that reference cycle immediately.
        """
        with self._lock:
            if self._value is None and self._error is None:
                self._error = RuntimeError('Deferred pressure maps were discarded with their request')
            self._build = None

    def value(self):
        from .gil_gate import engine_waiting
        selected = self._selected
        if selected is not None and not selected.is_set():
            with engine_waiting():      # opt-in gate: the engine waits on background work
                if not selected.wait(self.SELECTION_TIMEOUT):
                    raise RuntimeError('Lazy layer pressure selection did not finish before its maps were needed')
        if not self._lock.acquire(blocking=False):
            with engine_waiting():
                self._lock.acquire()
        with _Release(self._lock):
            if self._value is None and self._error is None:
                try:
                    self._value = self._build()
                except BaseException as error:
                    self._error = error
            if self._error is not None:
                raise self._error
            self._build = None
            return self._value


class PendingMaps:
    """A committed pressure owner whose per-layer maps are still being built."""
    def __init__(self, deferred, job):
        self.deferred, self.job = deferred, job

    def resolve(self):
        prepared = self.deferred.value()
        prepared.validate(self.job)
        return self.job.history, tuple(prepared.maps)


def native_lookups():
    value = _settings.get('KVMEM_PRESSURE_NATIVE_LOOKUPS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_NATIVE_LOOKUPS must be 0 or 1')
    return value == '1'


def native_layer_pick():
    """KVMEM_PRESSURE_NATIVE_LAYER_PICK: one native call per layer for the
    fast-prepare candidate loop, preview pick, layer checks and archive order."""
    value = _settings.get('KVMEM_PRESSURE_NATIVE_LAYER_PICK', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_NATIVE_LAYER_PICK must be 0 or 1')
    return value == '1'


def native_lookup_scores(history, mapping):
    """Opt-in: history.scores().get(begin, 0.) for every mapping row, computed
    natively from the same doubles (same IEEE division), or None when off."""
    if not native_lookups() or not getattr(history, 'native_score_export', False):
        return None
    state = history._buffered_state
    if not hasattr(state, 'scores_for'):
        return None
    return state.scores_for([row[2] for row in mapping])


# Age-adjusted per-layer pressure (strategy_pressure_scorer=lift, strategy/builtin.py).
# Native: the layer histories are lift states (new_history), whose exported scores are
# accumulated attention / expected attention for each block's distance, so every
# selection path below runs unchanged.
LIFT = 'lift'


def pressure_scorer(owner, options):
    """The profile's pressure scorer, or None for the built-in attention path."""
    name = getattr(options, 'strategy_pressure_scorer', 'attention')
    cache = owner.__dict__.setdefault('_strategy_scorers', {})
    if name not in cache:
        from ..strategy import PRESSURE_SCORER, resolve
        cache[name] = resolve(PRESSURE_SCORER, name)
    return cache[name]


def strategy_scores(scorer, layer, mapping, history, decode):
    """A strategy's scores for one layer's window, checked against the contract."""
    from ..strategy import PressureLayerView, check_pressure_scores
    view = PressureLayerView(layer, mapping, history, None, decode)
    return list(check_pressure_scores(view, scorer.scores(view)))


def same_snapshot(global_history, layer_zero):
    """global_history.snapshot() == layer_zero.snapshot(); opt-in native compare."""
    from .prefill_attention import PrefillHistory
    state = getattr(layer_zero, '_buffered_state', None) if native_lookups() else None
    other = getattr(global_history, '_buffered_state', None) if state is not None else None
    if other is not None and hasattr(state, 'same_state'):
        # Both native (KVMEM_GLOBAL_PREFILL_BUFFERED): compare states directly.
        return ((global_history.end, global_history.samples, global_history.chunks) ==
                (layer_zero.end, layer_zero.samples, layer_zero.chunks) and state.same_state(other))
    if (state is None or not hasattr(state, 'same_values') or type(global_history) is not PrefillHistory):
        return global_history.snapshot() == layer_zero.snapshot()
    return ((global_history.end, global_history.samples, global_history.chunks) ==
            (layer_zero.end, layer_zero.samples, layer_zero.chunks) and
            state.same_values(global_history.values))


def record_index(controller, authority):
    """Opt-in KVMEM_PRESSURE_RECORD_INDEX_CACHE: reuse the sorted records,
    their slots and lookup tables while the authority's record set is unchanged
    (VersionedRecords.version). Identical objects and order; None when off."""
    value = _settings.get('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_RECORD_INDEX_CACHE must be 0 or 1')
    current = authority.records
    version = getattr(current, 'version', None)
    if value != '1' or version is None:
        return None
    cache = controller.__dict__.setdefault('_record_index_cache', {})
    key = authority.memory.request_id
    entry = cache.get(key)
    if entry is not None and entry[0] is current and entry[1] == version:
        return entry[2]
    log = current.log
    if (entry is not None and entry[0] is current and log is not None and
            len(log) == version - entry[1]):
        # Only appends since the cached index: extend copies when every new
        # record starts after the last indexed one (the sorted order is then
        # unchanged); anything else falls through to a full rebuild.
        records, record_slots, record_positions, by_begin = (list(entry[2][0]), list(entry[2][1]),
                                                             dict(entry[2][2]), dict(entry[2][3]))
        last = records[-1].history_begin if records else None
        extended = True
        for record_key_ in log:
            record = current.get(record_key_)
            if record is None:
                extended = False
                break
            slot = record.handle.slot
            if slot in record_positions or (last is not None and record.history_begin <= last):
                extended = False
                break
            record_positions[slot] = len(records)
            records.append(record)
            record_slots.append(slot)
            by_begin[record.history_begin] = (record, slot)
            last = record.history_begin
        if extended:
            index = (records, record_slots, record_positions, by_begin)
            aligned = entry[3] and all(r.history_begin % 32 == 0 for r in records[len(entry[2][0]):])
            aligned = aligned and len(by_begin) == len(records)
            current.log = []
            cache[key] = (current, version, index, aligned)
            return index
    records = sorted(current.values(), key=lambda r: r.history_begin)
    record_slots = [r.handle.slot for r in records]
    record_positions = {slot: i for i, slot in enumerate(record_slots)}
    if len(record_positions) != len(records):
        raise RuntimeError('Pressure archive contains duplicate physical slots')
    by_begin = {r.history_begin: (r, s) for r, s in zip(records, record_slots)}
    index = (records, record_slots, record_positions, by_begin)
    aligned = len(by_begin) == len(records) and all(r.history_begin % 32 == 0 for r in records)
    current.log = []        # start logging appends after this full build
    cache[key] = (current, version, index, aligned)
    return index


def _drop_record_index(controller, request_id, authority=None):
    """Forget a finished owner's cached record index.

    The cache is keyed by the authority's native namespace, which for a
    restored session is the session's first request, not the finishing one.
    Its Record list must not outlive the turn: it keeps retired records'
    index storage alive, so historical GC could never release their slots.
    """
    cache = controller.__dict__.get('_record_index_cache', {})
    cache.pop(request_id, None)
    controller.__dict__.get('_identity_tables', {}).pop(request_id, None)
    key = getattr(getattr(authority, 'memory', None), 'request_id', None)
    if key is not None:
        cache.pop(key, None)
        controller.__dict__.get('_identity_tables', {}).pop(key, None)


def record_index_aligned(controller, authority):
    """True when the cached index's records start at unique multiples of 32."""
    entry = controller.__dict__.get('_record_index_cache', {}).get(authority.memory.request_id)
    return bool(entry is not None and entry[0] is authority.records and entry[3])


class LayerwisePrefillPressure:
    def __init__(self, query):
        self.query, self.runner, self.names = query, query.runner, query.names
        if not query.optimized:
            raise ValueError('Layerwise pressure requires optimized layerwise Query execution')
        if not query.native_batches and not self.runner.model_config.enforce_eager:
            raise ValueError('Compiled batches require the native layerwise Query dispatcher')
        self.maps = {}
        self.histories = {}
        self.rolling_decode = _settings.get('KVMEM_LAYERWISE_ROLLING_DECODE') == '1'
        self.decode_histories = {}
        self.options = None
        self.single_assembly = _settings.get('KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY') == '1'
        self.fast_metadata = _settings.get('KVMEM_LAYERWISE_PRESSURE_FAST_METADATA') == '1'
        self.buffered_history = _settings.get('KVMEM_LAYERWISE_BUFFERED_HISTORY') == '1'
        batch_accept = _settings.get('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT', '0')
        if batch_accept not in ('0', '1'):
            raise ValueError('Layerwise pressure batch acceptance must be 0 or 1')
        self.batch_accept = batch_accept == '1'
        from .deferred_fold import enabled as _deferred_fold
        self.deferred_fold = _deferred_fold()
        if self.deferred_fold and not self.batch_accept:
            raise ValueError('Deferred layer folds require batched layer acceptance')
        if self.batch_accept:
            if not self.buffered_history:
                raise ValueError('Batched layer acceptance requires native buffered histories')
            from .buffered_prefill_attention import require_batch_accept
            require_batch_accept()
        # New execution candidates remain opt-in until full throughput gates
        # pass; a successful component test alone must not change defaults.
        self.prepared_reads = _settings.get('KVMEM_LAYERWISE_PRESSURE_PREPARED_READS') == '1'
        self.post_forward_reap = _settings.get('KVMEM_LAYERWISE_PRESSURE_POST_FORWARD_REAP') == '1'
        self.submit_on_prepare = _settings.get('KVMEM_LAYERWISE_PRESSURE_SUBMIT_ON_PREPARE') == '1'
        self.reuse_layer_views = _settings.get('KVMEM_LAYERWISE_PRESSURE_REUSE_LAYER_VIEWS') == '1'
        self.resident_values = _settings.get('KVMEM_LAYERWISE_PRESSURE_RESIDENT_VALUES') == '1'
        batch_selection = _settings.get('KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION', '0')
        if batch_selection not in ('0', '1'):
            raise ValueError('Layerwise pressure batch selection must be 0 or 1')
        self.batch_selection = batch_selection == '1'
        # Opt-in: one pass per layer builds candidates and picker scores from
        # the same accepted values (identical floats, order and checks).
        fast_prepare = _settings.get('KVMEM_LAYERWISE_PRESSURE_FAST_PREPARE', '0')
        if fast_prepare not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_FAST_PREPARE must be 0 or 1')
        self.fast_prepare = fast_prepare == '1'
        # Opt-in: build the prepared 17-layer commit maps after the plan/ACK on
        # the producer instead of inside selection. Same maps; consumers resolve.
        deferred_maps = _settings.get('KVMEM_LAYERWISE_PRESSURE_DEFERRED_MAPS', '0')
        if deferred_maps not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_DEFERRED_MAPS must be 0 or 1')
        self.deferred_maps = deferred_maps == '1'
        pipeline = _settings.get('KVMEM_LAYERWISE_PRESSURE_PIPELINE', '0')
        if pipeline not in ('0', '1'):
            raise ValueError('Layerwise pressure pipeline must be 0 or 1')
        handoff = _settings.get('KVMEM_LAYERWISE_PRESSURE_PIPELINE_HANDOFF', '0')
        if handoff not in ('0', '1'):
            raise ValueError('Layerwise pressure pipeline handoff must be 0 or 1')
        self.pipeline_handoff = handoff == '1'
        prepared_commit = _settings.get('KVMEM_LAYERWISE_PRESSURE_PREPARED_COMMIT', '0')
        preforward = _settings.get('KVMEM_LAYERWISE_PRESSURE_PIPELINE_PREFORWARD', '0')
        progress = _settings.get('KVMEM_LAYERWISE_PRESSURE_PIPELINE_PROGRESS', '0')
        if prepared_commit not in ('0', '1') or preforward not in ('0', '1') or progress not in ('0', '1'):
            raise ValueError('Prepared commit and pre-forward pressure flags must be 0 or 1')
        self.prepared_commit = prepared_commit == '1'
        self.pipeline_preforward = preforward == '1'
        self.pipeline_progress = progress == '1'
        if self.prepared_commit and not self.single_assembly:
            raise ValueError('Prepared pressure commit requires the single-assembly path')
        if self.pipeline_preforward and not self.pipeline_handoff:
            raise ValueError('Pre-forward pressure preparation requires guarded early handoff')
        if self.pipeline_progress and (not self.pipeline_handoff or self.pipeline_preforward):
            raise ValueError('Completion progress requires the original bounded handoff without early enqueue')
        if self.pipeline_handoff and (pipeline != '1' or not self.post_forward_reap or
                                      not self.submit_on_prepare):
            raise ValueError('Early handoff requires pipeline and original pressure prepare/reap paths')
        # Opt-in: acknowledge an already-prepared plan at the next schedule
        # instead of the next batch's feedback (synchronous single process).
        schedule_ack = _settings.get('KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK', '0')
        if schedule_ack not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK must be 0 or 1')
        self.schedule_ack = schedule_ack == '1'
        if self.schedule_ack and not self.pipeline_handoff:
            raise ValueError('Schedule-time pressure ACK requires the guarded early handoff')
        self.pipeline = None
        from .pressure_key_reuse import enabled as _gpu_reuse
        if pipeline == '1':
            if (not self.single_assembly or not self.prepared_reads or not self.reuse_layer_views or
                    not query.paired_reads or not query.native_batches or
                    _settings.get('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING', '0') != '0' or
                    (self.resident_values and not _gpu_reuse())):
                raise ValueError('Pipeline requires the qualified single-assembly immutable-reader path')
            from .pressure_pipeline import PressurePipeline
            self.pipeline = PressurePipeline(self)
        # Opt-in concurrent prefill-pressure producers. Each lane has its own
        # worker thread, CUDA stream, staging scratch and transfer credit; the
        # engine thread still owns every commit. Selection and bytes are the
        # same per owner; only how many owners assemble at once changes.
        lanes = _settings.get('KVMEM_LAYERWISE_PRESSURE_LANES', '1')
        if lanes not in ('1', '2'):
            raise ValueError('Layerwise pressure lanes must be 1 or 2')
        self.lanes = int(lanes)
        if self.lanes > 1 and (self.pipeline is None or not self.pipeline_handoff or
                               self.batch_selection or self.resident_values or self.parent_staging_requested()):
            raise ValueError('Concurrent pressure lanes require the guarded pipeline handoff '
                             'without batch selection, resident values or parent staging')
        self.install_acceptance()

    @staticmethod
    def parent_staging_requested():
        return _settings.get('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING', '0') != '0'

    def new_history(self, options):
        if getattr(self, 'buffered_history', False):
            if getattr(self, 'deferred_fold', False):
                from .deferred_fold import DeferredFoldHistory
                history = DeferredFoldHistory(options)
            else:
                from .buffered_prefill_attention import BufferedPrefillHistory
                history = BufferedPrefillHistory(options)
        else:
            history = PrefillHistory(options)
        if getattr(options, 'strategy_pressure_scorer', 'attention') == LIFT:
            # Age-adjusted pressure: the native state also accumulates each block's
            # expected attention for its distance, in the same pass, batches and folds.
            # Every layer history starts as one, restored ones included: a session
            # restore shares the checkpoint's native state (KVMEM_CHECKPOINT_FAST_STATE),
            # and any other restore path is refused rather than silently dropping it.
            if not getattr(history, 'native_score_export', False):
                raise ValueError('The lift pressure scorer needs native buffered histories with native '
                                 'score export (KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT=1)')
            from .._prefill_state_native import State
            history._buffered_state = State.lift_empty()
        return history

    def history_execution(self, history):
        backend = getattr(history, 'storage_backend', 'native_dictionary')
        commits = getattr(history, 'successful_buffered_commits', 0)
        restores = getattr(history, 'successful_buffered_restores', 0)
        if getattr(self, 'buffered_history', False) and (backend != 'native_buffered' or commits + restores <= 0):
            raise RuntimeError('Layer pressure did not execute the configured native history backend')
        result = dict(attention_state_backend=backend, buffered_commits=commits, buffered_restores=restores)
        if getattr(self, 'batch_accept', False):
            result['buffered_batch_commits'] = history.successful_buffered_batch_commits
        return result

    def _history(self, rid, options):
        if self.options is None:
            self.options = options
            # Memory suites retain their original graph metadata for the
            # initial Query path. Pressure itself still uses only the supplied
            # per-layer cumulative attention, never graph propagation.
            if (options.pressure_policy != 'attention' or
                    options.guided_query_tokens or options.query_history or
                    not options.prefill_attention):
                raise ValueError('Layerwise pressure requires cumulative attention without guidance/history fusion')
        if rid not in self.histories:
            self.histories[rid] = [self.new_history(options) for _ in self.names]
        return self.histories[rid]

    def scoring_layers(self, action):
        # CPU history also archives MTP's draft-attention layer. It has no
        # public target-layer Query and must never join this score tensor.
        if tuple(action.runtime.names[:len(self.names)]) != self.names:
            raise RuntimeError('Target attention layers changed their archive order')
        return [(i, action.runtime.layers[i]) for i in range(len(self.names))]

    def resolved_maps(self, rid):
        frame = self.maps.get(rid)
        if isinstance(frame, PendingMaps):
            frame = self.maps[rid] = frame.resolve()
        return frame

    def records(self, rid, authority, history_begin, local_begin, layer):
        return self._layer_records(self.resolved_maps(rid), authority, history_begin, local_begin, layer)

    def _layer_records(self, frame, authority, history_begin, local_begin, layer, working=None):
        if frame is None:
            # Without previous maps every layer starts from the native frame:
            # lazy selection passes the snapshot taken before the layer-0 commit.
            result = list(authority.memory._store.working_records() if working is None else working)
        else:
            origin, layers = frame
            result = extend_records(layers[layer], origin, history_begin)
        if sum(r[3] for r in result) != local_begin:
            raise RuntimeError('Layer-specific physical history length differs from the native frame')
        return result

    def bind_history_owner(self, rid, authority):
        """Prune this request's layer histories whenever its authority prunes
        retired blocks from the shared attention state (historical GC)."""
        owners = self.__dict__.setdefault('_history_owners', {})
        if owners.get(rid) is authority:
            return
        owners[rid] = authority
        pruners = authority.__dict__.setdefault('history_pruners', [])
        if not any(p is self for p in pruners):
            pruners.append(self)

    def prune_layer_histories(self, authority, live):
        """The shared rule of RequestAuthority._prune_history_metadata, per layer."""
        end = authority.history_end
        keep = lambda key: key in live or key >= end
        owners = self.__dict__.get('_history_owners', {})
        from .layerwise_gc_holds import holds_of
        retired = holds_of(self).retired
        for rid, owner in list(owners.items()):
            if owner is not authority:
                continue
            stash = retired.get(rid)
            groups = [self.histories.get(rid), stash[1] if stash else None]
            for histories in groups:
                for history in histories or ():
                    values = history.values
                    if any(not keep(key) for key in values):
                        history.values = {k: v for k, v in values.items() if keep(k)}
            from .layerwise_decode import DecodeHistory
            for table, key in ((self.decode_histories, rid),):
                rows = table.get(key)
                if rows:
                    table[key] = [h if all(keep(k) for k in h.mass) else
                                  DecodeHistory({k: v for k, v in h.mass.items() if keep(k)}, h.samples, h.end, h.chunks)
                                  for h in rows]
            if stash and stash[2]:
                decode = [h if all(keep(k) for k in h.mass) else
                          DecodeHistory({k: v for k, v in h.mass.items() if keep(k)}, h.samples, h.end, h.chunks)
                          for h in stash[2]]
                retired[rid] = (stash[0], stash[1], decode)
            if rid not in self.histories and rid not in self.decode_histories and rid not in retired:
                del owners[rid]

    def attach_packet(self, pending, rid, authority, starts, counts, masses):
        if len(masses.shape) != 3 or masses.shape[0] != len(self.names):
            raise RuntimeError('Missing independent prefill scores for a full-attention layer')
        self._history(rid, pending.state.options)
        self.bind_history_owner(rid, authority)
        if pending.prefill is None:
            if not self.rolling_decode:
                raise RuntimeError('Decode feedback requires explicit layerwise rolling mode')
            from .layerwise_decode import attach_packet
            return attach_packet(self, pending, rid, authority, starts, counts, masses)
        if self.fast_metadata and rid in self.maps:
            from .layerwise_pressure_efficiency import packet_histories
            histories = packet_histories(self.resolved_maps(rid), pending, starts, counts)
            pending.layerwise_pressure = (self, rid, histories, masses)
            return
        end, count, _, local = pending.prefill
        histories = []
        for layer in range(len(self.names)):
            if rid not in self.maps:
                # The original immutable pipeline plan can include a preceding
                # accepted chunk whose CPU archive publication is still pending.
                # All layers share that plan before the first pressure event.
                histories.append(list(pending.history))
                continue
            records = self.records(rid, authority, end-count, local, layer)
            ls, lc, lh = window_segments(records, end-count, local, count, 32)
            if list(ls) != list(starts) or list(lc) != list(counts):
                raise RuntimeError('Layer map changed causal segment geometry')
            histories.append(lh)
        if histories[0] != list(pending.history):
            raise RuntimeError('Shared metadata no longer describes physical layer zero')
        pending.layerwise_pressure = (self, rid, histories, masses)

    def prepare_accept(self, pending, accepted):
        if getattr(pending, 'layerwise_decode', None) is not None:
            from .layerwise_decode import prepare_accept
            return prepare_accept(self, pending, accepted)
        marker = getattr(pending, 'layerwise_pressure', None)
        if marker is None:
            return None
        owner, rid, logical, masses = marker
        if owner is not self:
            raise RuntimeError('Foreign layerwise attention packet')
        end, count, starts, local = pending.prefill
        # Caller waits for the original score-download fence before this reads.
        staged = [copy(history) for history in self.histories[rid]]
        if getattr(self, 'batch_accept', False):
            from .buffered_prefill_attention import accept_buffered_batch
            deferred = None
            if getattr(self, 'deferred_fold', False):
                from . import deferred_fold as deferred
            if deferred is not None and deferred.deferrable(staged, end, count, accepted, pending.rows,
                                                            logical, masses):
                # Layer 0 now; layer i is folded, bit-identically, on first use.
                accepted_flags = deferred.accept_deferred(staged, end, count, accepted, pending.rows,
                    starts, logical, local, masses)
            else:
                accepted_flags = accept_buffered_batch(staged, end, count, accepted, pending.rows,
                    starts, logical, local, masses)
        else:
            accepted_flags = [history.accept(end, count, accepted, pending.rows,
                starts, logical[layer], local, masses[layer]) for layer, history in enumerate(staged)]
        if len(set(accepted_flags)) != 1:
            raise RuntimeError('Layer attention histories disagree on accepted sampling boundary')
        if getattr(self, 'rolling_decode', False):
            from .layerwise_decode import prefill_fold_enabled, fold_prefill
            if prefill_fold_enabled():
                # Opt-in KVMEM_LAYERWISE_DECODE_PREFILL: the same accepted rows
                # also score rolling decode; published with this transaction.
                decode = fold_prefill(self, rid, pending, accepted, logical, masses)
                if decode is not None:
                    return rid, staged, None, decode
        return rid, staged

    def commit_accept(self, prepared):
        if prepared is not None:
            if len(prepared) == 4:
                rid, histories, _, decode = prepared
                self.histories[rid] = histories
                self.decode_histories[rid] = decode
                return
            rid, histories = prepared[:2]
            target = self.decode_histories if len(prepared) == 3 else self.histories
            target[rid] = histories

    def prepare(self, refresh, job):
        from .request_plan import RequestPlan, selection_config
        rid, authority = job.command.request_id, job.authority
        decode = job.command.pressure_stage == 'decode' and self.rolling_decode
        if job.command.pressure_stage != 'prefill' and not decode:
            raise RuntimeError('This short-output experiment cannot reinterpret decode pressure')
        # Strategy layer: None runs the built-in accumulated-attention scores unchanged.
        scorer = pressure_scorer(self, refresh.options)
        trace_scores = getattr(refresh.options, 'strategy_trace_scores', False)
        from ..core.publication_trace import emit as _trace, enabled as _tracing
        _tr = _tracing()
        _sec = {}
        _clock = [time.perf_counter()]
        def _lap(name):
            if _tr:
                now = time.perf_counter(); _sec[name] = _sec.get(name, 0.) + now - _clock[0]; _clock[0] = now
        _sealed_started = time.perf_counter()
        if getattr(job, 'promised_lease', None) is None:
            authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
        # A promised refresh selects on record metadata only. _assemble repeats
        # this read barrier after selection and before any payload read, so the
        # just-appended records seal while selection runs (pressure_promise).
        _trace('prepare_seal_wait', rid=rid, seconds=time.perf_counter()-_sealed_started)
        _lap('seal_wait')
        cfg = selection_config(refresh.options.memory, job.command.groups[0][1].window_tokens,
            selection='pressure', pressure_policy='attention')
        fit = RequestPlan(None, block_tokens=32, gen_budget=cfg.gen_budget,
            mandatory=job.command.mandatory_spans).mandatory_fit(job.history, cfg)
        required = set(fit['hard_blocks'])
        _lap('plan_fit')
        # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: no retirement for this request
        # from the record snapshot until every layer's window is held.
        from .layerwise_gc_holds import wait_paused as _gc_wait_paused, holds_of as _gc_holds
        _gc_wait_paused(authority)
        index = record_index(self, authority)
        if index is None:
            records = sorted(authority.records.values(), key=lambda r:r.history_begin)
            # Each handle.slot read crosses pybind; read once per pressure, not per layer.
            record_slots = [r.handle.slot for r in records]
            record_positions = {slot:i for i,slot in enumerate(record_slots)}
            if len(record_positions) != len(records):
                raise RuntimeError('Pressure archive contains duplicate physical slots')
            by_begin = {r.history_begin:(r, s) for r, s in zip(records, record_slots)}
        else:
            records, record_slots, record_positions, by_begin = index
        if index is not None and record_index_aligned(self, authority):
            # Every record starts at a unique multiple of 32, so the records with
            # begin//32 in required are exactly by_begin[32*b], in sorted order.
            found = [by_begin.get(32 * b) for b in sorted(required)]
            if any(entry is None for entry in found):
                raise RuntimeError('Missing original mandatory pressure blocks')
            mandatory = tuple(entry[1] for entry in found)
            mandatory_set = set(mandatory)
        else:
            mandatory = tuple(s for r, s in zip(records, record_slots) if r.history_begin//32 in required)
            mandatory_set = set(mandatory)
            if {r.history_begin//32 for r, s in zip(records, record_slots) if s in mandatory_set} != required:
                raise RuntimeError('Missing original mandatory pressure blocks')
        _lap('records_sort_slots_mandatory')
        global_history = authority.action_history.prefill
        histories = (self.decode_histories if decode else self.histories).get(rid)
        if histories is None or (not decode and not same_snapshot(global_history, histories[0])):
            raise RuntimeError('Accepted layer-zero attention differs from the shared control state')
        _lap('history_snapshot_check')
        current = list(authority.memory._store.working_records())
        local = sum(r[3] for r in current)
        job.layerwise_layer0_diverged = False
        if not decode and rid in self.maps:
            from .layerwise_checkpoints import sessions_enabled, layer0_diverged
            if sessions_enabled() and layer0_diverged(
                    current, self.records(rid, authority, job.history, local, 0)):
                # Session resumed after a probe-mode Query: layer zero's pages
                # differ from the native frame, so the native skip hints of
                # unmoved rows do not apply to it (as for rolling decode).
                if not getattr(self, 'single_assembly', False):
                    raise ValueError('A diverged layer-zero window needs single pressure assembly')
                job.layerwise_layer0_diverged = True
        chosen, evidence, source_maps = [], [], []
        prepare_commit = not decode and getattr(self, 'prepared_commit', False)
        selected_slot_groups = []
        capture_values = not decode and getattr(self, 'resident_values', False)
        batched = None
        if not decode and getattr(self, 'batch_selection', False):
            if scorer is not None or getattr(authority.memory, 'picker', None) is not None:
                raise ValueError('Pressure batch selection implements the attention scorer and native picker only')
            from .pressure_selection_batch import prepare_pressure_batch
            batched = prepare_pressure_batch(self, job, histories, local, by_begin,
                                             mandatory, cfg.select_budget)
        _lap('working_records_batch')
        # Opt-in KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST (promised refreshes only):
        # select layer 0 now; the producer selects every other layer just before
        # uploading it. Inputs are frozen here (previous maps, histories, records,
        # mandatory set); the native picker ignores residency, so the committed
        # layer-0 host map cannot change any later layer's choice.
        from .pressure_key_reuse import enabled as _gpu_reuse
        # Opt-in KVMEM_LAYERWISE_PRESSURE_GPU_REUSE keeps layer-0-first with
        # resident sources: every layer's old window comes from the frozen
        # snapshot below, independent of the (lazy) selections themselves.
        lazy = (getattr(job, 'promised_lease', None) is not None and not decode and batched is None and
                (not capture_values or _gpu_reuse()) and prepare_commit and getattr(self, 'deferred_maps', False) and
                _settings.get('KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST') == '1')
        maps_snapshot = self.resolved_maps(rid) if lazy else None
        eager_sources = lazy and capture_values
        if eager_sources:
            source_maps.extend(tuple(self._layer_records(maps_snapshot, authority, job.history, local, layer, current))
                               for layer in range(len(histories)))
        def select_layer(layer):
            history = histories[layer]
            fused = None
            if batched is None:
                mapping = (self._layer_records(maps_snapshot, authority, job.history, local, layer, current)
                           if lazy else self.records(rid, authority, job.history, local, layer))
                _lap('layer_records_map')
                fused = None
                custom = None if scorer is None else strategy_scores(scorer, layer, mapping, history, decode)
                traced = custom
                # The fused native call is the native picker; another picker takes the plain path.
                if (getattr(self, 'fast_prepare', False) and native_layer_pick() and
                        getattr(authority.memory, 'picker', None) is None):
                    looked_up = native_lookup_scores(history, mapping) if custom is None else custom
                    if trace_scores and looked_up is not None:
                        traced = looked_up
                    _lap('layer_history_scores')
                    store = authority.memory._store
                    if (looked_up is not None and hasattr(store, 'pressure_layer_pick') and
                            type(records) is list and type(by_begin) is dict and
                            type(record_positions) is dict):
                        from ..core.memory import Phase
                        fused = store.pressure_layer_pick(
                            mapping if type(mapping) is list else list(mapping), by_begin,
                            looked_up if type(looked_up) is list else list(looked_up),
                            list(mandatory), cfg.select_budget, record_positions, records,
                            lambda: authority.memory._require(Phase.READY))
                if fused is not None:
                    # Same loop, checks, store calls and archive order, natively.
                    candidates, candidate_slots, ordered_slots, picked = fused
                    _lap('layer_native_select')
                elif getattr(self, 'fast_prepare', False):
                    looked_up = native_lookup_scores(history, mapping) if custom is None else custom
                    if looked_up is None:
                        scores = history.scores()
                        score_of = scores.get
                    else:
                        # scores().get(begin, 0.) per mapping row, without the dict.
                        score_of = iter(looked_up).__next__
                    _lap('layer_history_scores')
                    candidates, candidate_slots, supplied = [], [], []
                    positive = False
                    lookup, isfinite = by_begin.get, math.isfinite
                    for _, _, begin, n, _ in mapping:
                        entry = lookup(begin)
                        if entry is None or entry[0].count != n:
                            raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
                        record, slot = entry
                        score = score_of(begin, 0.) if looked_up is None else score_of()
                        if not isfinite(score) or score < 0:
                            raise RuntimeError('Invalid layer cumulative attention score')
                        positive = positive or score > 0
                        candidates.append(record)
                        candidate_slots.append(slot)
                        supplied.append((slot, score))
                    if not positive:
                        raise RuntimeError('Layer pressure has no accepted attention; refusing a recency fallback')
                    if trace_scores:
                        traced = [score for _, score in supplied]
                    _lap('layer_candidates_supplied')
                else:
                    candidates, candidate_slots = [], []
                    for _, _, begin, n, _ in mapping:
                        entry = by_begin.get(begin)
                        if entry is None or entry[0].count != n:
                            raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
                        candidates.append(entry[0])
                        candidate_slots.append(entry[1])
                    _lap('layer_candidates')
                    if custom is None:
                        scores = history.scores()
                        _lap('layer_history_scores')
                        supplied = [(s, scores.get(r.history_begin, 0.)) for r, s in zip(candidates, candidate_slots)]
                    else:
                        supplied = list(zip(candidate_slots, custom))
                    if trace_scores:
                        traced = [score for _, score in supplied]
                    if any(not math.isfinite(s) or s < 0 for _,s in supplied):
                        raise RuntimeError('Invalid layer cumulative attention score')
                    if not any(s > 0 for _,s in supplied):
                        raise RuntimeError('Layer pressure has no accepted attention; refusing a recency fallback')
                    _lap('layer_supplied_checks')
                if fused is None:
                    slots = set(authority.memory.preview_selection({}, mandatory, action=False,
                        pressure_scores=supplied, max_tokens=cfg.select_budget))
                    _lap('layer_native_picker')
            else:
                mapping, candidates, candidate_slots, slots = batched[layer]
            if capture_values and not eager_sources:
                source_maps.append(tuple(mapping))
            if fused is None:
                allowed = set(candidate_slots) | mandatory_set
                if not mandatory_set <= slots or not slots <= allowed or 32*len(slots) > cfg.select_budget:
                    raise RuntimeError('Layer pressure violated candidates, mandatory blocks or Active budget')
                # Preserve the original archive order without scanning cold records
                # again for every layer. The native preview still owns all ranking.
                ordered_slots = sorted(slots, key=record_positions.__getitem__)
                picked = [records[record_positions[s]] for s in ordered_slots]
                _lap('layer_order_pick')
            if prepare_commit:
                selected_slot_groups.append(ordered_slots)
            if layer == 0:
                job.layerwise_zero_slots = ordered_slots   # slots of chosen[0], same order
            chosen.append(picked)
            if _tp_active():
                # Reduced per-layer histories make the pick rank-identical; check
                # it before any rank assembles this layer (producer: private group).
                from .tensor_parallel import agree_layer_selection
                agree_layer_selection(rid, layer, picked, 'decode pressure' if decode else 'pressure')
            evidence.append(dict(layer=layer, layer_name=self.names[layer],
                candidate_block_count=len(candidates), candidate_runs=encode_starts([r.history_begin for r in candidates]),
                selected_starts=[r.history_begin for r in picked], selected_tokens=sum(r.count for r in picked),
                charged_tokens=32*len(picked), accepted_score_end=history.end,
                query_samples=history.samples, sampled_chunks=history.chunks,
                actual_method=(('lift_per_layer' if getattr(refresh.options, 'strategy_pressure_scorer', '') == LIFT
                                else 'accumulated_attention_per_layer') if scorer is None
                               else scorer.name + '_per_layer'), fallback_reason=None,
                **({'candidate_scores': [float(v) for v in traced],
                    # What an offline replay needs to pick again (python -m vllm_kvmem.strategy.replay).
                    'mandatory_starts': sorted(32 * b for b in required), 'budget_tokens': cfg.select_budget,
                    'sink_tokens': cfg.sink_blocks * cfg.block_tokens}
                   if trace_scores and batched is None and traced is not None else {}),
                **({'pressure_selection_backend': 'native_batch_snapshot'} if batched is not None else {}),
                **(dict(attention_state_backend='native_decode',
                        score_scope=_decode_labels()[1]) if decode else
                   self.history_execution(history))))
            _lap('layer_evidence')
        if lazy:
            select_layer(0)
        else:
            if getattr(self, 'deferred_fold', False):
                # Apply queued folds with the original layer-parallel batches.
                from .deferred_fold import force_all
                force_all(histories)
                _lap('deferred_fold_force')
            for layer in range(len(histories)):
                select_layer(layer)
            _gc_holds(self).protect(rid, authority, chosen)
        def geometry_check():
            lengths = {sum(r.count for r in group) for group in chosen}
            partial = [{(r.history_begin,r.count) for r in group if r.count != 32} for group in chosen]
            if len(lengths) != 1 or any(p != partial[0] for p in partial):
                raise RuntimeError('Layer pressure cannot change native causal length or partial-block geometry')
        deferred_maps = prepare_commit and getattr(self, 'deferred_maps', False)
        commit_table = None
        if prepare_commit:
            from .pressure_commit import identity_table
            commit_table = identity_table(self, authority.memory.request_id, records, record_slots)
        if not deferred_maps:
            geometry_check()
        job.layerwise_pressure_plan = chosen, evidence
        selected_all = None
        if deferred_maps:
            from .pressure_commit import prepare_commit as build_commit
            def build():
                geometry_check()
                return build_commit(job, chosen, records, record_slots, selected_slot_groups,
                                    table=commit_table)
            if lazy and len(chosen) < len(histories):
                import threading
                selected_all = threading.Event()
            job.layerwise_prepared_commit = DeferredPreparedCommit(
                build, frozenset(r.history_begin for r in chosen[0]), selected_all)
        elif prepare_commit:
            from .pressure_commit import prepare_commit as build_commit
            job.layerwise_prepared_commit = build_commit(
                job, chosen, records, record_slots, selected_slot_groups, table=commit_table)
        _lap('geometry_check_prepared_commit')
        if lazy:
            def select_next(layer):
                if layer != len(chosen) or layer >= len(histories):
                    raise RuntimeError('Lazy layer pressure selection lost its layer order')
                select_layer(layer)
                if len(chosen) == len(histories):
                    _gc_holds(self).protect(rid, authority, chosen)
                    if selected_all is not None:
                        selected_all.set()
                    if _tr:
                        _trace('prepare_sections', rid=rid, records=len(records), layers=len(histories),
                               lazy=True, **{k: round(v, 6) for k, v in _sec.items()})
                    if getattr(job, 'layerwise_evidence_pending', False):
                        job.layerwise_evidence_pending = False
                        self._emit_selected(job)
            job.layerwise_select_layer = select_next
            job.layerwise_layers_total = len(histories)
        elif _tr:
            _trace('prepare_sections', rid=rid, records=len(records), layers=len(histories),
                   **{k: round(v, 6) for k, v in _sec.items()})
        if capture_values:
            job.layerwise_pressure_value_sources = (rid, job.history, local, tuple(source_maps))
        if decode:
            if any(history.end > job.history for history in histories):
                raise RuntimeError('Rolling pressure contains future or rejected Query feedback')
            parser = getattr(authority.action_history, 'parser', None)
            job.layerwise_decode_thinking = bool(parser.state[0]) if parser is not None else None
            # The final Query can select a new layer-zero set while the native
            # host frame still describes its global preselection. Admit exactly
            # the independent preview; never rank stale shared-frame identities.
            job.layerwise_decode_selection = tuple(r.handle.slot for r in chosen[0])
        else:
            from .layerwise_checkpoints import sessions_enabled
            if sessions_enabled():
                # KVMEM_LAYERWISE_SESSIONS: after a decoded turn the shared
                # control state also carries output attention, so the global
                # pressure seed no longer equals layer zero's own prefill
                # history. Admit exactly layer zero's selection (as rolling
                # decode does); single-turn prefill is unchanged by this.
                job.layerwise_decode_selection = tuple(r.handle.slot for r in chosen[0])

    def committed(self, job):
        rid, authority = job.command.request_id, job.authority
        chosen, evidence = job.layerwise_pressure_plan
        actual = {r[2] for r in authority.memory._store.working_records()}
        prepared = deferred = None
        if job.command.pressure_stage == 'prefill' and getattr(self, 'prepared_commit', False):
            prepared = getattr(job, 'layerwise_prepared_commit', None)
            if prepared is None:
                raise RuntimeError('Missing the configured prepared pressure maps')
            if isinstance(prepared, DeferredPreparedCommit):
                deferred = prepared
                prepared = deferred.value() if deferred.ready else None
            if prepared is not None:
                prepared.validate(job)
            if not (getattr(job, 'layerwise_single_assembly_done', False) or
                    getattr(job, 'layerwise_pipeline_published', False)):
                raise RuntimeError('Prepared maps need completed assembly or guarded pipeline publication')
        if deferred is not None and prepared is None:
            expected_zero = deferred.layer_zero_starts
        else:
            expected_zero = prepared.layer_zero_starts if prepared else {r.history_begin for r in chosen[0]}
        if actual != expected_zero:
            raise RuntimeError('Native pressure selection disagrees with the layer-zero preview')
        state = dict(authority=authority, request_id=rid, selected_tokens=job.selected_tokens)
        maps = prepared.maps if prepared is not None else []
        pending = deferred is not None and prepared is None
        for layer, picked in (() if prepared is not None or pending else enumerate(chosen)):
            if layer and not (getattr(job, 'layerwise_single_assembly_done', False) or
                              getattr(job, 'layerwise_pipeline_published', False)):
                self.query.assemble_layer_pinned(state, layer, picked)
            cursor = 0
            mapping = []
            for r in picked:
                mapping.append((r.handle.slot,r.handle.generation,r.history_begin,r.count,cursor))
                cursor += r.count
            if cursor != job.selected_tokens:
                raise RuntimeError('Layer pressure assembly changed the unchanged native token extent')
            maps.append(tuple(mapping))
        # Single-pass assembly completes on the existing refresh fence before
        # the native transaction commits. It needs no second main-thread wait.
        if not (getattr(job, 'layerwise_single_assembly_done', False) or
                getattr(job, 'layerwise_pipeline_published', False)):
            torch.cuda.current_stream(self.runner.device).synchronize()
        self.maps[rid] = PendingMaps(deferred, job) if pending else (job.history, tuple(maps))
        if len(chosen) != getattr(job, 'layerwise_layers_total', len(chosen)):
            # Layer-0-first: the evidence row needs every layer's selection.
            job.layerwise_evidence_pending = True
            return
        self._emit_selected(job)

    def _emit_selected(self, job):
        rid = job.command.request_id
        chosen, evidence = job.layerwise_pressure_plan
        # Opt-in: the large per-layer evidence row is written after the ACK with
        # its original timestamps, bytes and row order (deferred_evidence.py).
        emit = (self.query.defer if job.command.pressure_stage == 'prefill' and
                getattr(self.query, 'deferred_writer', None) is not None else self.query.emit)
        emit('pressure_selected', request_id=rid, accepted_history=job.history,
            actual_method='accumulated_attention_per_layer', event_type=job.command.pressure_stage+'_pressure',
            configured_method='accumulated_attention_per_layer',
            query_source=(_decode_labels()[0] if job.command.pressure_stage == 'decode'
                          else 'accepted_public_prefill'),
            thinking_at_refresh=getattr(job, 'layerwise_decode_thinking', None),
            selected_tokens=job.selected_tokens, layers=evidence, fallback_reason=None)
        if original_rope():
            emit('pressure_rope', request_id=rid, accepted_history=job.history,
                rope_position_policy='original', key_source='record.history_begin',
                score_query_source='accepted_original_position', physical_slots='dense')

    @staticmethod
    def install_acceptance():
        if getattr(ActionRuntime, '_layerwise_pressure_accept_installed', False):
            return
        original, finish = ActionRuntime.accept, ActionRuntime.finish
        graph_prepare = ActionRuntime.prepare_accept
        def accept(runtime, packet, rid, accepted):
            pending = packet.action_inputs.get(rid)
            controller = getattr(runtime.runner, '_kvmem_layerwise_pressure', None)
            prepared = None
            if controller is not None and (getattr(pending, 'layerwise_pressure', None) is not None or
                                           getattr(pending, 'layerwise_decode', None) is not None):
                runtime._wait_packet(pending)
                prepared = controller.prepare_accept(pending, accepted)
            result = original(runtime, packet, rid, accepted)
            if controller is not None:
                controller.commit_accept(prepared)
            return result
        def prepare_graph(runtime, packet, rid, accepted, **kwargs):
            pending = packet.action_inputs.get(rid)
            controller = getattr(getattr(runtime, 'runner', None), '_kvmem_layerwise_pressure', None)
            staged = None
            if controller is not None and (getattr(pending, 'layerwise_pressure', None) is not None or
                                           getattr(pending, 'layerwise_decode', None) is not None):
                _t0 = time.perf_counter()
                runtime._wait_packet(pending)
                _t1 = time.perf_counter()
                staged = controller.prepare_accept(pending, accepted)
                _t2 = time.perf_counter()
                transaction = graph_prepare(runtime, packet, rid, accepted, **kwargs)
                from ..core.publication_trace import emit as _trace
                _trace('accept_split', rid=rid, wait_scores=_t1-_t0, layerwise_fold=_t2-_t1,
                       action_graph_prepare=time.perf_counter()-_t2)
            else:
                transaction = graph_prepare(runtime, packet, rid, accepted, **kwargs)
            return (_LayerwiseAcceptance(transaction, controller, staged)
                    if staged is not None else transaction)
        def finished(runtime, rid, authority):
            result = finish(runtime, rid, authority)
            controller = getattr(runtime.runner, '_kvmem_layerwise_pressure', None)
            if controller is not None:
                from .pressure_key_reuse import forget as _forget_reuse
                _forget_reuse(controller, rid)
                from .layerwise_checkpoints import sessions_enabled, retire_state
                if sessions_enabled():
                    # The terminal (M) capture runs after this finish: keep the
                    # request's per-layer state (and window hold) until then.
                    retire_state(controller, rid, authority)
                    return result
                frame = controller.maps.pop(rid, None)
                if isinstance(frame, PendingMaps):
                    frame.deferred.discard()          # break the builder <-> job cycle
                frame = None
                from .layerwise_gc_holds import holds_of
                holds_of(controller).release(rid)
                controller.histories.pop(rid, None)
                _drop_record_index(controller, rid, authority)
                controller.decode_histories.pop(rid, None)
                controller.__dict__.get('_history_owners', {}).pop(rid, None)
            return result
        ActionRuntime.accept, ActionRuntime.finish = accept, finished
        ActionRuntime.prepare_accept = prepare_graph
        ActionRuntime._layerwise_pressure_accept_installed = True
