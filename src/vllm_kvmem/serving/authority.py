"""Accepted-token raw-K / FP8-V authority with immutable sealed records.

CPU record buffers and admitted archive writes have a shared per-request bound.
The caller retries after poll() on backpressure. An incomplete tail is never
published to the immutable archive; repeated decode appends extend it in memory.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import hashlib
import math
import time

import torch

from .. import _native
from ..core.memory import RequestMemory
from ..core import settings as _settings


class _Ready:
    def query(self):
        return True


@dataclass
class Record:
    handle: object
    history_begin: int
    count: int
    raw: torch.Tensor | None
    values: torch.Tensor | None
    key_sum: torch.Tensor | None
    archived: bool = False
    submitted: bool = False
    digest: str | None = None
    scatter_views: tuple | None = field(default=None, repr=False)
    archive_views: tuple | None = field(default=None, repr=False)
    mean_key: torch.Tensor | None = field(default=None, repr=False)
    staging_slot: int | None = field(default=None, repr=False)
    index_owner: object | None = field(default=None, repr=False)

    def __del__(self):
        # Keep the native buffer owner until every Python tensor field
        # has dropped its reference. Its destruction only queues native work.
        owner = getattr(self, 'index_owner', None)
        self.key_sum = self.mean_key = self.scatter_views = self.archive_views = None
        self.index_owner = None
        del owner


def record_key(handle):
    return handle.slot, handle.generation


def _contiguous_sums(records):
    """torch.stack([r.key_sum ...]) without the copy when the sums already are
    consecutive views of one block (same values, same order)."""
    sums = [r.key_sum for r in records]
    first = sums[0]
    base = first._base
    if base is not None and base.dim() == first.dim() + 1 and base.is_contiguous():
        step = first.numel()
        start = first.storage_offset() - base.storage_offset()
        if (start % step == 0 and all(s._base is base and s.is_contiguous() and
                s.storage_offset() == first.storage_offset() + i * step for i, s in enumerate(sums))):
            index = start // step
            if index + len(sums) <= base.shape[0]:
                return base[index:index + len(sums)]
    return torch.stack(sums)


class VersionedRecords(dict):
    """The authority's record dict with a mutation counter (cache invalidation).

    Behaves exactly like a dict; ``version`` changes on every insert/remove so
    per-refresh indexes (sorted records, slots) can be reused while unchanged.
    """
    __slots__ = ('version', 'log', 'begin_log', 'begin_index')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = 0
        # Keys set since the index consumer last reset it; None after any
        # other kind of mutation (the consumer then rebuilds from scratch).
        self.log = None
        # The same for a second, independent consumer: the checkpoint capture's
        # {history_begin: record} index (layerwise_checkpoints.records_by_begin,
        # KVMEM_CHECKPOINT_CAPTURE_INDEX), cached here as (version, dict).
        self.begin_log = None
        self.begin_index = None

    def _changed(self):
        self.version += 1
        self.log = None
        # Drop the cached begin index with its record references: a removed
        # record must die now (its native index owner gates GC index release).
        self.begin_log = self.begin_index = None

    def __setitem__(self, key, value):
        self.version += 1
        if self.log is not None:
            self.log.append(key)
        if self.begin_log is not None:
            # Only new keys extend the begin index; replacing a key's record
            # would leave the old record's begin behind, so it forces a rebuild.
            if dict.__contains__(self, key):
                self.begin_log = self.begin_index = None
            else:
                self.begin_log.append(key)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._changed()
        super().__delitem__(key)

    def pop(self, *args):
        self._changed()
        return super().pop(*args)

    def popitem(self):
        self._changed()
        return super().popitem()

    def clear(self):
        self._changed()
        super().clear()

    def update(self, *args, **kwargs):
        self._changed()
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        self._changed()
        return super().setdefault(key, default)

    def __ior__(self, other):
        self._changed()
        return super().__ior__(other)


SCORE_DOMAIN_VERSION = 'qw3-native-v2'


# Global Query scorers (strategy layer). Each returns (scores, mean_scores):
# one score per record, and the MeanK scores a MeanK quota needs, or None.
def _global_meank(authority, records, queries, means, score_mask, eligible, guided_pressure, mean_key_scores):
    from .tensor_parallel import mean_head_scores
    return mean_head_scores(mean_key_scores(queries, means.to(queries.device), eligible)).cpu().tolist(), None


def _global_overlap(authority, records, queries, means, score_mask, eligible, guided_pressure, mean_key_scores):
    from .query_window_scoring import score_authority_windows
    from .tensor_parallel import mean_head_scores
    if guided_pressure:
        raise ValueError('16/8 window mode is qualified for the public Query only')
    values, stats = score_authority_windows(authority, records, queries, means, score_mask)
    authority.query_window_stats = stats
    return mean_head_scores(values).cpu().tolist(), None


def _global_key_fps(authority, records, queries, means, score_mask, eligible, guided_pressure, mean_key_scores):
    from .query_key_fps import score_authority_fps
    from .tensor_parallel import mean_head_scores
    if guided_pressure:
        raise ValueError('4-Key FPS mode is qualified for the public Query only')
    values, stats = score_authority_fps(authority, records, queries, means, score_mask)
    authority.query_key_fps_stats = stats
    scores, mean_scores = mean_head_scores(values).cpu().tolist(), None
    if getattr(getattr(authority, 'graph_options', None), 'query_mean_quota_fraction', 0):
        mean_started = time.perf_counter()
        mean_scores = mean_head_scores(mean_key_scores(
            queries, means.to(queries.device), eligible)).cpu().tolist()
        stats['mean_quota_scoring_seconds'] = time.perf_counter()-mean_started
    return scores, mean_scores


def _bind_global_query_scorers():
    from ..strategy import GLOBAL, QUERY_SCORER, bind
    for name, implementation in (('meank', _global_meank), ('overlap', _global_overlap),
                                 ('key_fps', _global_key_fps)):
        bind(QUERY_SCORER, name, GLOBAL, implementation)


_bind_global_query_scorers()


def global_query_scorer(authority):
    """The global Query scorer: a bound built-in, or a strategy object behind the contract.

    query_score_mode names the built-in one. On the global path a strategy that
    is not built in (strategy_query_scorer) replaces it.
    """
    from ..strategy import GLOBAL, QUERY_SCORE_MODES, QUERY_SCORER, native, resolve
    options = getattr(authority, 'graph_options', None)
    name = getattr(options, 'strategy_query_scorer', 'meank')
    if getattr(options, 'strategy_path', 'global') == GLOBAL and name not in QUERY_SCORE_MODES.values():
        cache = authority.__dict__.setdefault('_strategy_scorers', {})
        if name not in cache:
            cache[name] = resolve(QUERY_SCORER, name, GLOBAL)
        authority.query_strategy_name = name    # reported as the Query scoring method
        return _object_scorer(cache[name])
    return native(QUERY_SCORER, QUERY_SCORE_MODES[authority.query_score_mode], GLOBAL)


def _object_scorer(scorer):
    def scores(authority, records, queries, means, score_mask, eligible, guided_pressure, mean_key_scores):
        from ..strategy import QueryLayerView, check_query_scores
        if guided_pressure:
            raise ValueError('A strategy Query scorer is qualified for the public Query only')
        view = QueryLayerView(None, records, tuple(score_mask), queries, means,
                              max((r.history_begin + r.count for r in records), default=0), {})
        return list(check_query_scores(view, scorer.scores(view))), None
    return scores


@contextmanager
def query_score_frame(memory, records, mandatory=(), max_tokens=None):
    """Keep the original qw3 score-frame leases through scoring and selection."""
    frame = memory._store.query_score_frame([r.handle for r in records],
        [r.count for r in records], memory.history_end, list(mandatory), max_tokens or 0)
    try:
        yield frame
    finally:
        frame.close()


class MeanIndexStats:
    """Small shared counter; independent owner publishers can finish together."""
    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._counts = dict(completed_batches=0, completed_records=0,
                            vectorized_records=0, fallback_records=0)

    def record(self, records, vectorized, enabled):
        with self._lock:
            self._counts['completed_batches'] += 1
            self._counts['completed_records'] += records
            self._counts['vectorized_records'] += vectorized
            self._counts['fallback_records'] += records-vectorized if enabled else 0

    def snapshot(self):
        with self._lock:
            return dict(self._counts)


class RequestAuthority:
    def __init__(self, request_id, config, archive, layers, heads, head_dim,
                 raw_dtype=torch.bfloat16, resident_records=16, gc_config=None, gc_progress=None,
                 scoring_layers=None, mean_k_backend='torch_reference',
                 cpu_mean_index_simd=False, mean_index_stats=None,
                 query_score_mode='mean_k', query_window_cache_bytes=2147483648,
                 cpu_release_sealed_sums=False, query_fps_cache_fp8=False,
                 cpu_reuse_record_views=False, record_view_stats=None,
                 cpu_fuse_record_mean=False, fused_mean_stats=None):
        if min(layers, heads, head_dim, resident_records) <= 0 or raw_dtype not in (
                torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            raise ValueError("Invalid authority geometry or staging budget")
        self.memory = RequestMemory(request_id, config)
        self.archive = archive
        self.shape = (layers, config.block_tokens, heads, head_dim)
        self.scoring_layers = layers if scoring_layers is None else scoring_layers
        if type(self.scoring_layers) is not int or not 0 < self.scoring_layers <= layers:
            raise ValueError('MeanK scoring must cover a positive target layer prefix')
        self.raw_dtype = raw_dtype
        if mean_k_backend not in ('torch_reference','qw3_cuda'):
            raise ValueError('Invalid MeanK backend')
        self.mean_k_backend = mean_k_backend
        if type(cpu_release_sealed_sums) is not bool or type(query_fps_cache_fp8) is not bool:
            raise ValueError('Compact CPU index settings must be boolean')
        if cpu_release_sealed_sums and mean_k_backend != 'qw3_cuda':
            raise ValueError('Releasing sealed sums requires a stored native MeanK')
        if query_fps_cache_fp8 and (raw_dtype != torch.float8_e4m3fn or
                mean_k_backend != 'qw3_cuda' or query_score_mode != 'key_fps_4'):
            raise ValueError('Compact FPS cache requires original FP8 Keys and native FPS scoring')
        self.cpu_release_sealed_sums = cpu_release_sealed_sums
        if query_score_mode not in ('mean_k', 'overlap_16_8', 'key_fps_4') or (query_score_mode != 'mean_k' and config.block_tokens != 32):
            raise ValueError('Invalid Query window mode or physical block geometry')
        self.query_score_mode = query_score_mode
        self.query_window_cache = None
        if query_score_mode == 'overlap_16_8':
            from .query_window_scoring import WindowMeanCache
            self.query_window_cache = WindowMeanCache(archive, self.scoring_layers, heads, head_dim,
                dtype=torch.float16 if mean_k_backend == 'qw3_cuda' else torch.float32,
                max_bytes=query_window_cache_bytes)
        elif query_score_mode == 'key_fps_4':
            from .query_key_fps import KeyFPSCache
            self.query_window_cache = KeyFPSCache(archive, self.scoring_layers, heads, head_dim,
                dtype=torch.float16 if mean_k_backend == 'qw3_cuda' else torch.float32,
                max_bytes=query_window_cache_bytes, compact_fp8=query_fps_cache_fp8)
        if type(cpu_mean_index_simd) is not bool:
            raise ValueError('cpu_mean_index_simd must be boolean')
        if cpu_mean_index_simd and mean_k_backend != 'qw3_cuda':
            raise ValueError('cpu_mean_index_simd requires mean_k_backend=qw3_cuda')
        self.cpu_mean_index_simd = cpu_mean_index_simd
        if type(cpu_fuse_record_mean) is not bool:
            raise ValueError('cpu_fuse_record_mean must be boolean')
        if cpu_fuse_record_mean and (mean_k_backend != 'qw3_cuda' or
                raw_dtype not in (torch.bfloat16, torch.float8_e4m3fn) or cpu_mean_index_simd):
            raise ValueError('Fused record mean requires native BF16/FP8 indexing with SIMD disabled')
        self.cpu_fuse_record_mean = cpu_fuse_record_mean
        self.fused_mean_stats = dict(records=0, batches=0) if fused_mean_stats is None else fused_mean_stats
        if type(cpu_reuse_record_views) is not bool:
            raise ValueError('cpu_reuse_record_views must be boolean')
        self.cpu_reuse_record_views = cpu_reuse_record_views
        # Opt-in KVMEM_AUTHORITY_BATCHED_MEANS: pages completed by one append
        # skip redundant zero-filling (the batched native finalizer writes every
        # sum) and compute their FP16 MeanK in one stacked FP32 division.
        import os
        batched_means = _settings.get('KVMEM_AUTHORITY_BATCHED_MEANS', '0')
        if batched_means not in ('0', '1'):
            raise ValueError('KVMEM_AUTHORITY_BATCHED_MEANS must be 0 or 1')
        self._fast_record_init = (batched_means == '1' and mean_k_backend == 'qw3_cuda' and
                                  raw_dtype in (torch.bfloat16, torch.float8_e4m3fn) and
                                  callable(getattr(_native, 'finalize_mean_batch', None)))
        self.record_view_stats = (dict(slot_exports=0, bound_records=0,
            finalize_records=0, sealed_records=0) if record_view_stats is None else record_view_stats)
        self.mean_index_stats = MeanIndexStats() if mean_index_stats is None else mean_index_stats
        self.raw_element_bytes = raw_dtype.itemsize
        self.record_elements = layers * config.block_tokens * heads * head_dim
        self.raw_record_bytes = self.record_elements * self.raw_element_bytes
        self.value_record_bytes = self.record_elements
        self.record_bytes = self.raw_record_bytes + self.value_record_bytes
        if archive.record_bytes != self.record_bytes:
            raise ValueError("Archive record format does not match model geometry")
        self.resident_records = resident_records
        self.records: dict[tuple[int, int], Record] = VersionedRecords()
        from .tensor_parallel import active, admit_window, selected_slots
        if active():
            self.memory.tp_records = self.records
            # core.memory calls serving's collectives through these (it imports no serving module).
            self.memory.tp_selected_slots, self.memory.tp_admit_window = selected_slots, admit_window
        # These contain only live, not-yet-published generations, so ordinary
        # decode bookkeeping does not grow with archived history. GC can only
        # remove archived records; rewind explicitly removes its live suffix.
        self._unsubmitted: dict[tuple[int, int], Record] = {}
        self._resident_count = 0
        # Counts archived-flag flips; with records.version it tells a reader
        # that no record's archived state changed (session_worker._history_rows).
        self.archived_epoch = 0
        # Reuse only temporary raw/V staging. Immutable archive bytes and each
        # record's MeanK state remain independently owned. Repeatedly freeing
        # large raw/V tensors between persistent small index allocations can
        # leave many GiB of resident-but-free allocator holes during prefill.
        self._record_buffers = []
        # These exports belong to the existing bounded staging slots, never to
        # an archived generation. A slot remains busy until its write finishes.
        self._record_buffer_views = []
        self._free_record_buffers = []
        self._busy_record_buffers = set()
        # The staging pool may be borrowed by a streaming publication worker
        # (streaming_publication.py) while the scatter thread publishes.
        import threading
        self._record_buffer_lock = threading.RLock()
        # history_begin -> (raw, values, slot, sums): complete pages filled and
        # summed layer by layer during the forward that produced them.
        self._prefilled = {}
        self.writes = []
        self.closing = False
        self.discarded_records: dict[tuple[int, int], Record] = {}
        self._rewind_position = None
        self._rewind_committed = False
        self._index_snapshot_depth = 0
        self._pending_cache_retire = []
        self.gc = None
        self.redundancy_history = None
        from .pressure_query_capture import pressure_query_required, PressureQueryWindow
        self.pressure_query_window = PressureQueryWindow() if pressure_query_required() else None
        if getattr(archive, '_history_worker', None) is not None:
            if gc_config is not None:
                raise ValueError('Historical KV management and legacy repository GC cannot share an authority')
            from ..core.historical_gc import HistoricalAuthorityGc
            self.gc = HistoricalAuthorityGc(self.memory, archive)
            if active():
                # Leader-decided GC: records leave this index only in the
                # rank-synchronous tensor_parallel.gc_sync, never at a
                # rank-local completion time.
                from .tensor_parallel import register_gc_authority
                self._tp_history_gc = True
                register_gc_authority(self)
        elif gc_config is not None:
            from ..core.repository_gc import AuthorityGc
            self.gc = AuthorityGc(self.memory, archive, gc_config, gc_progress)

    @property
    def history_end(self):
        return self.memory.history_end

    def _drain_gc(self):
        # A rewound unpublished tail can be retired by the background worker.
        # Its generation-bound tombstone, rather than foreground reclaim(),
        # proves that the temporary buffer no longer has any readers.
        staged = [(key, record) for key, record in self.discarded_records.items()
                  if record.staging_slot is not None]
        retired = []
        finalizing = getattr(self.gc, 'stopped', False)
        if getattr(self.gc, 'native_history', False):
            if self.__dict__.get('_tp_history_gc', False):
                # Finalization and close may release live records at local times.
                self.gc.drain(self.records, self.discarded_records, retired=retired,
                              synchronized=not finalizing and not self.closing)
            else:
                self.gc.drain(self.records, self.discarded_records, retired=retired)
        else:
            self.gc.drain(self.records, self.discarded_records)
        if retired and not finalizing:
            self._trace_evictions(retired)
        if retired:
            self._pending_cache_retire.extend(retired)
            self._drain_cache_retirements()
            self._prune_history_metadata()
        for key, record in staged:
            if key not in self.discarded_records:
                self._return_record_buffers(record)

    def retire_synchronized(self, keys):
        """Tensor-parallel GC: remove the records every rank just retired.

        Runs inside tensor_parallel.gc_sync while the native round still holds
        its prepared records, so no CPU completion can precede the removal.
        """
        retired = []
        self.gc.retire_synchronized(self.records, keys, retired)
        return retired

    def finish_synchronized_retirements(self, retired):
        """The metadata pruning drain applies after a retirement, at the same step on every rank."""
        if not retired:
            return
        if not getattr(self.gc, 'stopped', False):
            self._trace_evictions(retired)
        self._pending_cache_retire.extend(retired)
        self._drain_cache_retirements()
        self._prune_history_metadata()

    def _trace_evictions(self, retired):
        """Strategy layer: the historical-GC evictor's decisions join the decision trace.

        A completion does not say why a record's CPU backing was released. Only a
        rise of the scope's policy-retirement counter marks evictions; releases at
        session hand-off or request end leave it unchanged and are not logged.
        """
        path = getattr(self.archive, 'history_scope_metrics_path', None)
        if path is None:
            return
        stats = getattr(getattr(self.gc, 'scope', None), 'stats', None)
        decided = stats.get('retired', 0) if isinstance(stats, dict) else 0
        if decided <= getattr(self, '_traced_policy_retired', 0):
            return
        self._traced_policy_retired = decided
        import json
        from pathlib import Path
        row = dict(event='evicted', request_id=self.memory.request_id, at=time.time(),
                   policy=getattr(self.archive, '_history_policy', None),
                   history_begins=sorted(int(entry[0]) for entry in retired))
        with (Path(path).parent / 'historical_gc_evictions.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    def _drain_cache_retirements(self):
        if self._index_snapshot_depth or not self._pending_cache_retire:
            return
        retired, self._pending_cache_retire = self._pending_cache_retire, []
        cache = self.query_window_cache
        if cache is not None and hasattr(cache, 'retire_records'):
            cache.retire_records(retired)

    def history_live(self):
        """History begins whose attention metadata survives a retirement prune."""
        live = {r.history_begin for r in self.records.values()}
        if self.memory._store is not None:
            live.update(row[2] for row in self.memory._store.working_records())
        return live

    def _prune_history_metadata(self):
        """Drop metadata only for permanently missing logical blocks.

        Accepted live-window attention can still contain a retired CPU block
        until the next GPU refresh. Those small rows remain until that refresh;
        saved future feedback on a verified P/M replay remains valid too.
        """
        if self.closing:
            return
        live = self.history_live()
        action = getattr(self, 'action_history', None)
        if action is not None:
            for name in ('scores', 'mass'):
                values = getattr(action, name)
                setattr(action, name, {k: v for k, v in values.items()
                    if k in live or k >= self.history_end})
            prefill = getattr(action, 'prefill', None)
            if prefill is not None:
                prefill.values = {k: v for k, v in prefill.values.items()
                    if k in live or k >= self.history_end}
        boundaries = getattr(self, 'query_boundaries', None)
        if boundaries is not None:
            boundaries.prune(sorted((r.history_begin, r.history_begin+r.count)
                for r in self.records.values()), future_begin=self.history_end)
        # Per-layer attention histories (layerwise pressure) prune by the same
        # rule, so layer zero stays identical to the shared prefill history.
        for owner in tuple(self.__dict__.get('history_pruners', ())):
            owner.prune_layer_histories(self, live)

    def poll(self):
        if self.gc is not None:
            self._drain_gc()
        thinking = getattr(self, 'thinking_history', None)
        if thinking is not None:
            thinking.retry(self.memory.repository)
        remaining = []
        newly_archived = []
        completed = 0
        for job, keys in self.writes:
            if not job.done():
                remaining.append((job, keys))
                continue
            result = job.result()  # Failed I/O is fatal, never a successfully archived record.
            digests = result.get('record_digests') if isinstance(result, dict) else None
            if digests is not None and len(digests) != len(keys):
                raise RuntimeError('Archived checksum count differs from sealed records')
            for index, key in enumerate(keys):
                record = self.records[key]
                if digests is not None:
                    record.digest = digests[index]
                if not record.archived:
                    self._resident_count -= 1
                    self.archived_epoch += 1
                    self._return_record_buffers(record)
                    newly_archived.append(record)
                record.archived = True
                redundancy = getattr(self, 'redundancy_history', None)
                if redundancy is not None:
                    redundancy.publish_key(record, self.memory.repository)
                thinking = getattr(self, 'thinking_history', None)
                if thinking is not None:
                    thinking.publish(record, self.memory.repository)
                # Publish only after durable write completion, never on CPU
                # cache admission or asynchronous I/O submission.
                flags = _native.GC_RAW_BACKED | _native.GC_VALUE_BACKED | _native.GC_INDEXED
                if not getattr(self.gc, 'native_history', False):
                    self.memory._store.gc_flags(record.handle, flags, flags)
            completed += 1
        self.writes = remaining
        prebuild = getattr(self, 'fps_prebuild', None)
        if prebuild is not None and newly_archived and not self.closing and self._rewind_position is None:
            prebuild.offer([r for r in newly_archived if r.count == self.shape[1]])
        for key, record in list(self.discarded_records.items()):
            if getattr(self.gc, 'native_history', False):
                break  # The native worker owns reclamation of these generations.
            result = self.memory.repository.reclaim(record.handle,
                lambda entry: self.archive.reclaim(self.memory.request_id, entry))
            if result == _native.ReclaimResult.RECLAIMED:
                self._return_record_buffers(record)
                del self.discarded_records[key]
        if self.gc is not None:
            # The background reclaimer may have finished a discarded generation
            # between our first drain and reclaim attempt. Its tombstone owns
            # index removal; neither worker fabricates a second release.
            self._drain_gc()
        return completed

    def _borrow_record_buffers(self):
        with self._record_buffer_lock:
            return self._borrow_record_buffers_locked()

    def _borrow_record_buffers_locked(self):
        if self._free_record_buffers:
            slot = self._free_record_buffers.pop()
        else:
            if len(self._record_buffers) >= self.resident_records:
                raise RuntimeError('Record staging exceeded its resident credit')
            buffers = (torch.empty(self.shape, dtype=self.raw_dtype),
                       torch.empty(self.shape, dtype=torch.uint8))
            slot = len(self._record_buffers)
            self._record_buffers.append(buffers)
            if self.cpu_reuse_record_views:
                raw, values = buffers
                raw_bytes, value_bytes = raw.view(torch.uint8).numpy(), values.numpy()
                self._record_buffer_views.append((raw_bytes, value_bytes,
                    (memoryview(raw_bytes).cast('B'), memoryview(value_bytes).cast('B'))))
                self.record_view_stats['slot_exports'] += 1
        if slot in self._busy_record_buffers:
            raise RuntimeError('Record staging slot is still owned by publication')
        self._busy_record_buffers.add(slot)
        return (*self._record_buffers[slot], slot)

    def _return_record_buffers(self, record):
        slot = record.staging_slot
        if slot is None:
            return
        self._release_staging_slot(slot)
        record.staging_slot = None

    def _release_staging_slot(self, slot):
        with self._record_buffer_lock:
            if slot not in self._busy_record_buffers:
                raise RuntimeError('Record staging ownership was released twice')
            self._busy_record_buffers.remove(slot)
            self._free_record_buffers.append(slot)

    def register_prefilled(self, pages):
        """Opt-in streaming publication: pages {history_begin: (raw, values, slot, sums)}."""
        with self._record_buffer_lock:
            if any(begin in self._prefilled for begin in pages):
                raise RuntimeError('A streamed page was registered twice')
            self._prefilled.update(pages)

    def discard_prefilled(self, begins=None):
        """Return unadopted streamed pages (aborted or clipped publication)."""
        with self._record_buffer_lock:
            keys = list(self._prefilled) if begins is None else [b for b in begins if b in self._prefilled]
            for begin in keys:
                slot = self._prefilled.pop(begin)[2]
                if slot is not None:   # zero-copy pages own no staging slot
                    self._release_staging_slot(slot)
            return len(keys)

    @property
    def record_staging_bytes(self):
        return len(self._record_buffers) * self.record_bytes

    def _require_open(self):
        if self.closing:
            raise RuntimeError('Authority is closing')
        if self._rewind_position is not None:
            raise RuntimeError('Authority rewind has not finished suffix cleanup')

    def flush(self):
        """Submit complete blocks only; false means archive queue/capacity pressure."""
        self._require_open()
        self.poll()
        records = list(self._unsubmitted.values())
        if not records:
            return True
        if self.cpu_release_sealed_sums and any(r.mean_key is None for r in records):
            raise RuntimeError('Cannot release an unfinished MeanK sum')
        leases = [self.memory.repository.pin(r.handle) for r in records]
        try:
            if not all(leases):
                raise RuntimeError("Authority record lost its repository generation")
            # Join tensor buffer views directly into the immutable I/O snapshot.
            # Avoid raw/V tobytes, per-record concatenations and their copies;
            # row layout and per-record checksums remain byte-for-byte identical.
            views = []
            for record in records:
                packed = getattr(record, 'packed', None)
                if packed is not None:
                    # The seal adopts these private bytes (memory_storage zero copy).
                    whole = memoryview(packed)
                    views.extend((whole[:self.raw_record_bytes], whole[self.raw_record_bytes:]))
                elif self.cpu_reuse_record_views:
                    if record.archive_views is None:
                        raise RuntimeError('Prepared record lost its archive views')
                    views.extend(record.archive_views)
                else:
                    views.extend((memoryview(record.raw.view(torch.uint8).numpy()).cast('B'),
                                  memoryview(record.values.numpy()).cast('B')))
            if getattr(self.archive, 'async_seal', False):
                from ..core.publication_trace import emit as _trace
                _trace('seal_submitted', rid=self.memory.request_id, records=len(records))
                job = self.archive.write_from_views(self.memory.request_id, leases, views)
            else:
                payload = b"".join(views)
                view = memoryview(payload)
                for i, record in enumerate(records):
                    begin = i * self.record_bytes
                    record.digest = hashlib.sha256(view[begin:begin+self.record_bytes]).hexdigest()
                job = self.archive.write(self.memory.request_id, leases, payload)
        finally:
            for lease in leases:
                lease.reset()
        if job is None:
            return False
        keys = []
        for record in records:
            record.submitted = True
            record.packed = None
            record.scatter_views = None
            record.archive_views = None
            if self.cpu_reuse_record_views:
                self.record_view_stats['sealed_records'] += 1
            record.raw = record.values = None
            if self.cpu_release_sealed_sums:
                # This sealed generation is immutable; reads use its finalized
                # MeanK. A rewind rebuilds any new partial tail from original K.
                record.key_sum = None
                if record.index_owner is not None:
                    record.index_owner.release_sums()
            key = record_key(record.handle)
            del self._unsubmitted[key]
            keys.append(key)
        self.writes.append((job, keys))
        return True

    def _prepare_append(self, raw, values, *, history_begin, local_rope_begin, accepted=None):
        """Reserve a CPU-ready accepted prefix; no copied rows are published here.

        raw/values: [layers, candidate tokens, KV heads, head_dim]. The caller
        must have observed its DMA fence before passing these CPU tensors.
        """
        self._require_open()
        if (raw.device.type != "cpu" or values.device.type != "cpu" or
                raw.dtype != self.raw_dtype or values.dtype != torch.uint8 or
                raw.ndim != 4 or raw.shape != values.shape or
                (raw.shape[0], raw.shape[2], raw.shape[3]) != (self.shape[0], self.shape[2], self.shape[3])):
            raise ValueError("Captured payload does not match the authority codec")
        count = raw.shape[1] if accepted is None else accepted
        if type(count) is not int or not 0 <= count <= raw.shape[1]:
            raise ValueError("Accepted prefix outside candidate rows")
        if history_begin != self.history_end:
            raise ValueError("Capture skipped or repeated logical history")
        if count == 0:
            return (raw, values, count, [])
        waiter = getattr(self, '_prefill_wait', None)
        if waiter is not None:
            # Opt-in prescatter: this span's pages are being filled by a worker.
            waiter(history_begin, count)
        self.flush()
        block = self.shape[1]
        # Includes I/O-owned byte snapshots after torch staging was released.
        tail = self.history_end % block
        new_records = (tail + count + block - 1) // block - bool(tail)
        if self._resident_count + new_records > self.resident_records:
            return None
        spans = self.memory.begin_append(count, local_rope_begin)
        if spans is None:
            return None
        if self.gc is not None:
            # Recycled reservations become ready after their release callback
            # queued the old generation. Drop it before installing a new index.
            self._drain_gc()
        # Validate the complete write map before mutating any existing tail.
        for span in spans:
            key = record_key(span.handle)
            offset = span.storage_begin % block
            previous = self.records.get(key)
            if ((offset and previous is None) or
                    (previous is not None and (previous.count != offset or previous.submitted))):
                raise RuntimeError("Repository append no longer matches the authoritative tail")
        rows = []
        # Batched MeanK: one contiguous FP32 block for this append's new full
        # pages, handed out as consecutive views (one allocation, and the means
        # later come from one slice instead of a stack of separate tensors).
        sum_block = sum_next = None
        if self._fast_record_init and not getattr(self.gc, 'native_history', False):
            fresh = sum(1 for span in spans if span.token_count == block and
                        span.storage_begin % block == 0 and record_key(span.handle) not in self.records)
            if fresh > 1:
                sum_block = torch.empty((fresh, self.scoring_layers, self.shape[2], self.shape[3]),
                                        dtype=torch.float32)
                sum_next = 0
        for span in spans:
            key = record_key(span.handle)
            offset = span.storage_begin % block
            start = span.history_begin - history_begin
            n = span.token_count
            record = self.records.get(key)
            if record is None:
                # pybind field views retain their Append/Store owner. Persist a
                # value handle, otherwise an index entry pins the entire store
                # and prevents request cancellation from reclaiming any record.
                handle = _native.Handle(span.handle.slot, span.handle.generation)
                index_shape = (self.scoring_layers, self.shape[2], self.shape[3])
                index_owner = None
                if getattr(self.gc, 'native_history', False):
                    sums, index_owner = self.archive.allocate_index(index_shape,
                        mean=self.mean_k_backend == 'qw3_cuda',
                        split_sums=self.cpu_release_sealed_sums)
                elif self._fast_record_init and n == block:
                    # Completed by this append: the batched MeanK finalizer
                    # initializes and writes every sum before any read.
                    if sum_block is not None and offset == 0 and sum_next < sum_block.shape[0]:
                        sums = sum_block[sum_next]
                        sum_next += 1
                    else:
                        sums = torch.empty(index_shape, dtype=torch.float32)
                else:
                    sums = torch.zeros(index_shape, dtype=torch.float32)
                prefilled = None
                # Native-history GC owns its own sum storage; never adopt there.
                if self._prefilled and n == block and offset == 0 and index_owner is None:
                    with self._record_buffer_lock:
                        prefilled = self._prefilled.pop(span.history_begin, None)
                if prefilled is not None:
                    # Bytes and MeanK sums were produced layer by layer from the
                    # same capture rows during the forward (streaming_publication).
                    record_raw, record_values, slot, sums, packed = prefilled
                else:
                    record_raw, record_values, slot = self._borrow_record_buffers()
                    packed = None
                try:
                    record = Record(handle, span.history_begin, 0,
                        record_raw, record_values, sums, staging_slot=slot)
                    record.index_owner = index_owner
                    record.prefilled = prefilled is not None
                    # Zero-copy streamed page: its private archive bytes.
                    record.packed = packed
                    if self.cpu_reuse_record_views:
                        raw_bytes, value_bytes, archive_views = self._record_buffer_views[slot]
                        record.scatter_views = (raw_bytes, value_bytes, sums.numpy())
                        record.archive_views = archive_views
                        self.record_view_stats['bound_records'] += 1
                    self.records[key] = record
                except BaseException:
                    if slot is not None:
                        self._release_staging_slot(slot)
                    raise
                self._resident_count += 1
            rows.append((key, record, offset, start, n))
        return raw, values, count, rows

    def _commit_prepared_append(self, prepared, *, finalized_records=()):
        raw, values, count, rows = prepared
        if not count:
            return
        import time as _time
        _c0 = _time.perf_counter()
        finalize_batch = getattr(_native, 'finalize_mean_batch', None)
        batched_index = (self.mean_k_backend == 'qw3_cuda' and
                         self.raw_dtype in (torch.bfloat16, torch.float8_e4m3fn) and
                         callable(finalize_batch))
        if batched_index:
            # Source scatter has completed and every destination is owned by
            # this append. Keep token-order FP32 addition and the existing
            # FP16 conversion; only batch the small native calls/GIL handoffs.
            jobs = [(*(record.scatter_views[0::2] if self.cpu_reuse_record_views else
                        (record.raw.view(torch.uint8).numpy(), record.key_sum.numpy())),
                     record.count+n, self.raw_dtype == torch.float8_e4m3fn)
                    for key, record, _, _, n in rows
                    if record.count+n == self.shape[1] and key not in finalized_records]
            vectorized = 0
            if jobs:
                if self.cpu_reuse_record_views:
                    self.record_view_stats['finalize_records'] += len(jobs)
                if self.cpu_mean_index_simd:
                    vectorized = finalize_batch(jobs, vectorized=True)
                    if type(vectorized) is not int or not 0 <= vectorized <= len(jobs):
                        raise RuntimeError('Invalid native MeanK SIMD execution count')
                else:
                    finalize_batch(jobs)
            if jobs or finalized_records:
                self.mean_index_stats.record(len(jobs)+len(finalized_records), vectorized, self.cpu_mean_index_simd)
        _c1 = _time.perf_counter()
        batched_means = []
        for key, record, offset, start, n in rows:
            record.count += n
            if record.count == self.shape[1]:
                if self.mean_k_backend == 'qw3_cuda':
                    if self.raw_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
                        raise ValueError('Original content-index finalization requires a qualified raw source')
                    finalize = (_native.finalize_mean_fp8 if self.raw_dtype == torch.float8_e4m3fn
                                else _native.finalize_mean_bf16)
                    if not batched_index and key not in finalized_records:
                        finalize(record.raw.view(torch.uint8).numpy(), record.key_sum.numpy(), record.count)
                    if record.index_owner is not None:
                        if self.cpu_release_sealed_sums:
                            record.mean_key = record.index_owner.mean_view()
                        else:
                            record.mean_key = torch.frombuffer(record.index_owner, dtype=torch.float16,
                                count=record.key_sum.numel(), offset=record.key_sum.numel()*4).view(record.key_sum.shape)
                        torch.div(record.key_sum, record.count, out=record.mean_key)
                    elif self._fast_record_init:
                        batched_means.append(record)
                    else:
                        record.mean_key = (record.key_sum / record.count).to(torch.float16)
                self._unsubmitted[key] = record
        if batched_means:
            # One element-wise FP32 division and FP16 rounding for the whole
            # append; every completed page has count == block_tokens.
            means = (_contiguous_sums(batched_means) / self.shape[1]).to(torch.float16)
            for record, mean in zip(batched_means, means):
                record.mean_key = mean
        _c2 = _time.perf_counter()
        self.memory.attach_completion(_Ready())
        assert self.memory.commit_append(count)
        thinking = getattr(self, 'thinking_history', None)
        if thinking is not None:
            thinking.note_append(self.history_end-count, count, rows)
        boundaries = getattr(self, 'query_boundaries', None)
        if boundaries is not None:
            boundaries.advance(self.history_end-count, self.history_end)
        if getattr(self.gc, 'native_history', False):
            # One accepted public packet stamps the current GPU window. The
            # native observer does not scan historical records per decode.
            self.memory._store.gc_begin_step()
        _c3 = _time.perf_counter()
        self.flush()
        from ..core.publication_trace import emit as _trace
        _trace('commit_append_split', records=len(rows), finalize_mean=_c1-_c0, record_loop=_c2-_c1,
               store_commit=_c3-_c2, flush_seal_submit=_time.perf_counter()-_c3)

    def append(self, raw, values, *, history_begin, local_rope_begin, accepted=None):
        prepared = self._prepare_append(raw, values, history_begin=history_begin,
            local_rope_begin=local_rope_begin, accepted=accepted)
        if prepared is None:
            return False
        raw, values, count, rows = prepared
        for key, record, offset, start, n in rows:
            record.raw[:, offset:offset+n].copy_(raw[:, start:start+n])
            record.values[:, offset:offset+n].copy_(values[:, start:start+n])
            record.key_sum.add_(raw[:record.key_sum.shape[0], start:start+n].float().sum(dim=1))
        self._commit_prepared_append(prepared)
        return True

    def publish_sealed(self, wait_for_progress):
        """Complete the read barrier, including writes refused by admission.

        Append may keep sealed CPU records when another namespace fills the
        shared queue. Waiting only for this authority's admitted jobs misses
        those records; retry admission before any published-only packed read.
        """
        while not self.flush():
            wait_for_progress()
        for job, _ in tuple(self.writes):
            job.result()
        self.poll()

    @contextmanager
    def index_snapshot(self):
        """Pin every scored generation until selection has acquired its leases."""
        self._require_open()
        self.poll()
        leases, holds, ordered = [], [], []
        scope = getattr(self.memory, '_history_scope', None)
        self._index_snapshot_depth += 1
        try:
            for record in sorted(self.records.values(), key=lambda r: r.history_begin):
                # Ordinary pins retain metadata but do not prevent permanent
                # CPU retirement. A policy hold protects FPS misses and the
                # complete multi-stage picker without changing LRU heat.
                if scope is not None:
                    while True:
                        hold = scope.protect([record.handle])
                        if hold is not None:
                            holds.append(hold)
                            break
                        check = self.memory.repository.pin(record.handle)
                        if not check:
                            break  # Retired before this snapshot admitted it.
                        check.reset()
                        # Only a concurrent policy gate can reject a still
                        # live generation. Yield instead of spinning under GIL.
                        time.sleep(0)
                    if hold is None:
                        continue
                lease = self.memory.repository.pin(record.handle)
                if lease:
                    leases.append(lease)
                    ordered.append(record)
            if not ordered:
                raise ValueError("No accepted tokens in the index")
            yield ordered, torch.stack([r.mean_key if r.mean_key is not None else
                (r.key_sum / r.count).to(torch.float16 if self.mean_k_backend=='qw3_cuda' else torch.float32)
                for r in ordered], dim=1)
        finally:
            for lease in leases:
                lease.reset()
            holds.clear()
            self._index_snapshot_depth -= 1
            # An older retirement receipt can arrive during a Key read's
            # poll(). Its block was excluded before scoring. Defer the cache
            # epoch bump until the held Query views finish; it cannot invalidate
            # the still-live rows being scored by this transaction.
            self._drain_cache_retirements()
            if self.gc is not None and not getattr(self.gc, 'native_history', False):
                self.gc.worker.notify_progress()

    def mean_keys(self):
        """Detached diagnostic copy; use index_snapshot() for retrieval/selection."""
        with self.index_snapshot() as (records, means):
            return [r.handle for r in records], means

    def read_record(self, handle):
        return self.read_records([handle])[0]

    def read_score_keys(self, handles, *, pin_memory=False):
        """Owned, bounded raw-K slab for the scoring layers only.

        The [record, layer, token, KV head, dim] destination is written directly
        by archive region gathers. V and appended draft layers are excluded.
        Read leases and source owners survive every asynchronous gather until
        its completion; the returned tensor owns independent copied storage.
        """
        self._require_open()
        if not handles or len(handles) > 32 or type(pin_memory) is not bool:
            raise ValueError('Scoring Key reads require one to 32 records')
        keys = [record_key(handle) for handle in handles]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate scoring Key read handle')
        self.poll()
        records = [self.records[key] for key in keys]
        layers, tokens, heads, dim = self.scoring_layers, *self.shape[1:]
        amount = layers*tokens*heads*dim*self.raw_element_bytes
        if len(handles)*amount > 64*1024**2:
            raise MemoryError('Scoring Key read scratch exceeds 64 MiB')
        admitted = []
        stats = dict(key_read_bytes=0, value_read_bytes=0, draft_key_read_bytes=0,
            key_read_batches=0, key_resident_bytes=0, key_region_seconds=0.)
        try:
            for handle in handles:
                admitted.append(self.memory.admitted_read(handle))
            if not all(entry[0] for entry in admitted):
                raise ValueError('Stale authority scoring Key handle')
            archived = [i for i, record in enumerate(records) if record.raw is None]
            if any(not records[i].archived for i in archived):
                raise BlockingIOError('Immutable scoring Key write is still in flight')
            result = torch.zeros((len(records), layers, tokens, heads, dim),
                dtype=self.raw_dtype, pin_memory=pin_memory)
            for i, record in enumerate(records):
                if record.raw is not None:
                    result[i, :, :record.count].copy_(record.raw[:layers, :record.count])
                    stats['key_resident_bytes'] += layers*record.count*heads*dim*self.raw_element_bytes
            # Usually all records are immutable. Separate only contiguous runs
            # around resident tails so no second packed slab or clone is needed.
            cursor = 0
            while cursor < len(archived):
                stop = cursor+1
                while stop < len(archived) and archived[stop] == archived[stop-1]+1:
                    stop += 1
                indices = archived[cursor:stop]
                sources = [admitted[i][1] for i in indices]
                kwargs = {'_sources': sources} if sources and all(sources) else {}
                destination = result[indices[0]:indices[-1]+1].view(torch.uint8).numpy().reshape(-1)
                started = time.perf_counter()
                job = self.archive.read_region(self.memory.request_id,
                    [admitted[i][0] for i in indices], 0, amount, destination=destination, **kwargs)
                if job is None:
                    raise BlockingIOError('Archive scoring Key read queue is full')
                _, counters = job.result()
                stats['key_region_seconds'] += time.perf_counter()-started
                stats['key_read_bytes'] += len(indices)*amount
                stats['key_read_batches'] += 1
                for name in ('read_queue_wait_seconds', 'cache_copy_seconds', 'cache_hit_bytes'):
                    stats[name] = stats.get(name, 0)+counters.get(name, 0)
                cursor = stop
            return result, stats
        finally:
            for lease, _ in admitted:
                lease.reset()

    def read_score_kv(self, handles, *, destination=None):
        """Gather target-layer K/V straight into owned block-major score slabs.

        Region reads skip draft layers and keep their native leases until both
        operands finish. A caller may supply contiguous CPU slabs to avoid a
        second packed copy; no returned view aliases the archive or a live tail.
        """
        self._require_open()
        if not handles or len(handles) > 32:
            raise ValueError('Scoring K/V reads require one to 32 records')
        keys = [record_key(handle) for handle in handles]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate scoring K/V read handle')
        self.poll()
        records = [self.records[key] for key in keys]
        shape = (len(records), self.scoring_layers, *self.shape[1:])
        elements = math.prod(shape[1:])
        if len(records)*elements*(self.raw_element_bytes+1) > 64*1024**2:
            raise MemoryError('Scoring K/V read scratch exceeds 64 MiB')
        if destination is None:
            result = (torch.empty(shape, dtype=self.raw_dtype),
                      torch.empty(shape, dtype=torch.uint8))
        else:
            if (not isinstance(destination, tuple) or len(destination) != 2 or
                    any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or
                        tuple(t.shape) != shape or not t.is_contiguous() for t in destination) or
                    destination[0].dtype != self.raw_dtype or destination[1].dtype != torch.uint8):
                raise ValueError('Scoring K/V destination must be contiguous CPU raw-K/uint8-V slabs')
            result = destination
        stats = dict(key_read_bytes=0, value_read_bytes=0, draft_key_read_bytes=0,
                     draft_value_read_bytes=0, kv_region_batches=0, kv_resident_bytes=0)
        admitted = []
        try:
            for handle in handles:
                admitted.append(self.memory.admitted_read(handle))
            if not all(entry[0] for entry in admitted):
                raise ValueError('Stale authority scoring K/V handle')
            archived = [i for i, record in enumerate(records) if record.raw is None]
            if any(not records[i].archived for i in archived):
                raise BlockingIOError('Immutable scoring K/V write is still in flight')
            for i, record in enumerate(records):
                if record.raw is not None:
                    result[0][i, :, :record.count].copy_(record.raw[:self.scoring_layers, :record.count])
                    result[1][i, :, :record.count].copy_(record.values[:self.scoring_layers, :record.count])
                    stats['kv_resident_bytes'] += (self.scoring_layers*record.count*
                        math.prod(self.shape[2:])*(self.raw_element_bytes+1))
            cursor = 0
            while cursor < len(archived):
                stop = cursor+1
                while stop < len(archived) and archived[stop] == archived[stop-1]+1:
                    stop += 1
                indices = archived[cursor:stop]
                sources = [admitted[i][1] for i in indices]
                kwargs = {'_sources': sources} if sources and all(sources) else {}
                for target, offset, amount, counter in (
                        (result[0], 0, elements*self.raw_element_bytes, 'key_read_bytes'),
                        (result[1], self.raw_record_bytes, elements, 'value_read_bytes')):
                    output = target[indices[0]:indices[-1]+1].view(torch.uint8).numpy().reshape(-1)
                    job = self.archive.read_region(self.memory.request_id,
                        [admitted[i][0] for i in indices], offset, amount, destination=output, **kwargs)
                    if job is None:
                        raise BlockingIOError('Archive scoring K/V read queue is full')
                    _, counters = job.result()
                    stats[counter] += len(indices)*amount
                    stats['kv_region_batches'] += 1
                    for name in ('read_queue_wait_seconds', 'cache_copy_seconds', 'cache_hit_bytes'):
                        stats[name] = stats.get(name, 0)+counters.get(name, 0)
                cursor = stop
            for i, record in enumerate(records):
                if record.count < self.shape[1]:
                    for target in result:
                        target[i, :, record.count:].zero_()
            return result, stats
        finally:
            for lease, _ in admitted:
                lease.reset()

    def read_records(self, handles):
        """Read a bounded batch through one coalescible native archive I/O job."""
        self._require_open()
        self.poll()
        records = [self.records[record_key(handle)] for handle in handles]
        admitted = [self.memory.admitted_read(handle) for handle in handles]
        leases = [entry[0] for entry in admitted]
        try:
            if not all(leases):
                raise ValueError("Stale authority handle")
            disk_indices = [i for i, record in enumerate(records) if record.raw is None]
            if any(not records[i].archived for i in disk_indices):
                raise BlockingIOError("Immutable record write is still in flight")
            result = [None] * len(records)
            for i, record in enumerate(records):
                if record.raw is not None:
                    result[i] = (record.raw[:, :record.count].clone(), record.values[:, :record.count].clone())
            if not disk_indices:
                return result
            sources = [admitted[i][1] for i in disk_indices]
            kwargs = {'_sources': sources} if sources and all(sources) else {}
            job = self.archive.read(self.memory.request_id, [leases[i] for i in disk_indices], **kwargs)
            if job is None:
                raise BlockingIOError("Archive read queue is full")
            data, _ = job.result()
            elements = self.record_elements
            for packet_index, i in enumerate(disk_indices):
                offset = packet_index * self.record_bytes
                raw = torch.frombuffer(data, dtype=self.raw_dtype, count=elements, offset=offset).reshape(self.shape)
                values = torch.frombuffer(data, dtype=torch.uint8, count=elements,
                                          offset=offset + self.raw_record_bytes).reshape(self.shape)
                count = records[i].count
                result[i] = (raw[:, :count].clone(), values[:, :count].clone())
            return result
        finally:
            for lease in leases:
                lease.reset()

    def raw_layer_hole(self, first, count):
        """(offset, bytes) of raw-plane archive layers [first, first + count) in a record,
        or (0, 0) when that range is not 4 KiB aligned (then it is read)."""
        per_layer = self.shape[1] * self.shape[2] * self.shape[3] * self.raw_element_bytes
        offset, size = first * per_layer, count * per_layer
        if count <= 0 or first + count > self.shape[0] or offset % 4096 or size % 4096:
            return 0, 0
        return offset, size

    def start_direct_read(self, handles, destination, hole=(0, 0)):
        """SSD tier: start reading records' archive bytes into a pinned, 4 KiB aligned uint8 tensor.

        Record i lands at destination[i * record_bytes:] in the archive layout
        (raw plane, then values). Records still held in memory are copied from
        RAM; the rest are read with O_DIRECT, except hole = (offset, bytes) of
        every record (bytes the caller does not use). Returns (pending, counts):
        pending.wait() blocks on the drive and releases the leases.
        """
        self._require_open()
        self.poll()
        records = [self.records[record_key(handle)] for handle in handles]
        admitted = [self.memory.admitted_read(handle) for handle in handles]
        leases = [entry[0] for entry in admitted]
        pending = None
        try:
            if not all(leases):
                raise ValueError("Stale authority handle")
            if any(record.raw is None and not record.archived for record in records):
                raise BlockingIOError("Immutable record write is still in flight")
            for i, record in enumerate(records):
                if record.raw is None:
                    continue
                base, count = i * self.record_bytes, record.count
                raw = destination[base:base + self.raw_record_bytes].view(self.raw_dtype).view(self.shape)
                raw[:, :count].copy_(record.raw[:, :count])
                values = destination[base + self.raw_record_bytes:base + self.record_bytes].view(self.shape)
                values[:, :count].copy_(record.values[:, :count])
            pending = self.archive.read_direct(
                self.memory.request_id, [lease if record.raw is None else None
                                         for lease, record in zip(leases, records)], destination, hole)
            return _HeldDirectRead(pending, leases), [record.count for record in records]
        finally:
            if pending is None:
                for lease in leases:
                    lease.reset()

    def start_group_read(self, handles, groups, destination):
        """SSD tier, archive_layout=layer: start reading packed layer groups of records.

        Group groups[k] of record i lands at destination[offset_k + i * group_bytes]
        where the groups are laid out one after another (plane-major), each over
        all records. Records still held in memory are packed from RAM. Returns
        (pending, counts, offsets); pending.wait() blocks on the drive and
        releases the leases.
        """
        self._require_open()
        self.poll()
        group_bytes = self.archive.layer_group_bytes
        offsets, cursor = [], 0
        for group in groups:
            offsets.append(cursor)
            cursor += len(handles) * group_bytes[group]
        if cursor > destination.numel():
            raise ValueError('Layer group destination is smaller than the batch')
        records = [self.records[record_key(handle)] for handle in handles]
        admitted = [self.memory.admitted_read(handle) for handle in handles]
        leases = [entry[0] for entry in admitted]
        pending = None
        try:
            if not all(leases):
                raise ValueError("Stale authority handle")
            if any(record.raw is None and not record.archived for record in records):
                raise BlockingIOError("Immutable record write is still in flight")
            in_memory = [i for i, record in enumerate(records) if record.raw is not None]
            if in_memory:
                from ..core.layer_major import pack_record_groups
                for i in in_memory:
                    record = records[i]
                    whole = torch.zeros(self.record_bytes, dtype=torch.uint8)
                    whole[:self.raw_record_bytes].view(self.raw_dtype).view(self.shape)[:, :record.count].copy_(
                        record.raw[:, :record.count])
                    whole[self.raw_record_bytes:].view(self.shape)[:, :record.count].copy_(
                        record.values[:, :record.count])
                    for group, offset in zip(groups, offsets):
                        size = group_bytes[group]
                        pack_record_groups(whole, self.archive.tier.groups[group],
                                           destination[offset + i * size:offset + (i + 1) * size])
            pending = self.archive.read_groups(
                self.memory.request_id, [lease if record.raw is None else None
                                         for lease, record in zip(leases, records)],
                groups, offsets, destination)
            return _HeldDirectRead(pending, leases), [record.count for record in records], offsets
        finally:
            if pending is None:
                for lease in leases:
                    lease.reset()

    def select_for_queries(self, queries, mandatory_handles=(), *, observe_only=False, max_tokens=None,
                           guided_pressure=False):
        """Reference MeanK-to-native-selection bridge over accepted history only.

        The returned plan still needs fenced GPU assembly and commit_selection.
        Queries must come from the model's content-frame query probe/replay.
        """
        self._require_open()
        from ..core.retrieval import mean_key_scores
        if self.mean_k_backend == 'qw3_cuda':
            from ..core.cuda_scoring import mean_key_scores
        from .graph_integration import query_snapshot, score_and_select
        from .tensor_parallel import mean_head_scores, verify_query_domain
        with self.index_snapshot() as (records, means), (
                nullcontext(None) if observe_only else query_snapshot(self)) as graph_snapshot:
            # MTP records travel with target KV but never alter MeanK scoring.
            means = means[:getattr(self, 'scoring_layers', self.shape[0])]
            handles = [r.handle for r in records]
            keys = [record_key(h) for h in handles]
            mandatory = {record_key(h) for h in mandatory_handles}
            if not mandatory.issubset(keys):
                raise ValueError("Mandatory selection uses a stale or foreign record")
            # Native policy owns the softmax domain. Mandatory identity does
            # not remove a logit; fixed picker keeps instead mask transport.
            with query_score_frame(self.memory, records,
                    tuple(slot for slot, _ in mandatory), max_tokens) as frame:
                score_mask = frame.score_mask
                transport_mask = frame.transport_mask
                verify_query_domain(self.memory.request_id, records, score_mask, queries.shape[1])
                eligible = torch.tensor(score_mask, dtype=torch.bool, device=queries.device)
                # Strategy layer: the global Query scorer, looked up by name.
                scorer = global_query_scorer(self)
                scores, mean_scores = scorer(self, records, queries, means, score_mask, eligible,
                                             guided_pressure, mean_key_scores)
            # index_snapshot retains the same generation-qualified records
            # through the remaining synchronous picker/assembly preparation.
            self.query_score_domain = dict(version=SCORE_DOMAIN_VERSION,
                score_mask=tuple(score_mask), transport_mask=tuple(transport_mask))
            history = getattr(self, 'query_history', None)
            if history is not None and history.policy.open:
                scores_by_start = history.preview({r.history_begin: score
                    for r, score in zip(records, scores)})
                scores = [scores_by_start.get(r.history_begin, 0.0) for r in records]
            if observe_only:
                return
            if guided_pressure:
                from .guided_pressure import select_guided_pressure
                return select_guided_pressure(self, records, scores, graph_snapshot,
                    tuple(slot for slot, _ in mandatory), max_tokens)
            graph_options = getattr(self, 'graph_options', None)
            if graph_snapshot is None and self.query_score_mode == 'key_fps_4' and (
                    getattr(graph_options, 'graph_mode', None) == 'off'):
                from .query_no_graph import select_without_graph
                return select_without_graph(self, records, scores, transport_mask,
                    tuple(slot for slot, _ in mandatory), max_tokens, mean_scores=mean_scores)
            if graph_snapshot is not None:
                from .graph_integration import record_query_inputs
                record_query_inputs(self, records, scores, transport_mask,
                    graph_snapshot, tuple(slot for slot, _ in mandatory), max_tokens)
            if graph_snapshot is not None and graph_options.graph_mode in ('locality', 'attention', 'multi'):
                return score_and_select(self, records, scores, transport_mask,
                    graph_snapshot, tuple(slot for slot, _ in mandatory), max_tokens, mean_scores=mean_scores)
            action = getattr(self,'action_history',None)
            if action is not None:
                # A verified M-prefix continuation may replay from an older P.
                # Apply saved logical scores to pages that have now reappeared.
                self.memory.set_action_scores(action.scores)
            prefill = getattr(action,'prefill',None)
            by_slot = {h.slot: score for h, score in zip(handles, scores)}
            pins = tuple(slot for slot, _ in mandatory)
            prefill_scores = prefill.scores() if prefill is not None and not action.samples else None
            if graph_snapshot is not None:
                from .graph_integration import emit_graph_report
                graph = self.block_graph
                selected = set(self.memory.preview_selection(by_slot, pins,
                    prefill_scores=prefill_scores, max_tokens=max_tokens))
                report = dict(mode='collect', revision=graph_snapshot.revision,
                    phase='query', seed_source=self.query_score_mode, score_domain_version=SCORE_DOMAIN_VERSION,
                    scoring_version=getattr(graph_options,'graph_scoring_version','v1'),
                    alpha=getattr(graph_options,'graph_alpha',.2),
                    pressure_policy=getattr(graph_options,'pressure_policy','recency'),
                    score_mask_count=sum(score_mask), transport_mask_count=sum(transport_mask),
                    locality_provenance=dict(getattr(graph_snapshot,'locality_provenance',())),
                    accepted_through=graph_snapshot.accepted_through,
                    observed_sources=len(graph_snapshot.rows),
                    edges=sum(len(row.targets) for row in graph_snapshot.rows),
                    graph_bytes=graph.bytes, graph_live_bytes=graph.live_bytes,
                    snapshot_bytes=graph_snapshot.bytes, graph_global_bytes=graph.budget.used,
                    selected_tokens=sum(r.count for r in records if r.handle.slot in selected), selection_unchanged=True)
                if graph_options.graph_trace:
                    report['selected_starts'] = [r.history_begin for r in records if r.handle.slot in selected][:256]
                self.graph_selection_report = report
                emit_graph_report(self, report)
            plan = self.memory.select(by_slot, pins, prefill_scores=prefill_scores, max_tokens=max_tokens)
            return plan

    def begin_rewind(self, checkpoint):
        """Quiesce publication and prepare a block-aligned branch rollback.

        None means existing archive publication must drain first; an empty tuple
        is a valid no-suffix transaction. The caller must also quiesce GPU/model
        users and supply their cleanup fence before poll_rewind can commit.
        """
        self._require_open()
        if not self.memory.checkpoint_valid(checkpoint):
            raise ValueError('Stale or foreign authority checkpoint')
        if checkpoint.position % self.shape[1]:
            raise ValueError('Rewind requires a sealed-record boundary')
        if not self.flush() or self.writes:
            return None
        prebuild = getattr(self, 'fps_prebuild', None)
        preserve_prefix = (self.query_score_mode == 'key_fps_4' and
            bool(getattr(getattr(self, 'graph_options', None), 'query_mean_quota_fraction', 0)))
        if prebuild is not None:
            prebuild.cancel(history_end=checkpoint.position) if preserve_prefix else prebuild.cancel()
        redundancy = getattr(self, 'redundancy_history', None)
        if redundancy is not None:
            # Invalidate queued decisions before retiring a suffix that may
            # contain their retained representative. Validation/drain above
            # leaves rejected or deferred rewinds untouched.
            redundancy.invalidate()
        spans = self.memory.begin_rewind(checkpoint)
        thinking = getattr(self, 'thinking_history', None)
        if thinking is not None:
            thinking.rewind(checkpoint.position)
        recall_cache = getattr(self, '_recall_feature_cache', None)
        if recall_cache is not None:
            recall_cache.clear()
        pressure_queries = getattr(self, 'pressure_query_window', None)
        if pressure_queries is not None:
            pressure_queries.rewind(checkpoint.position)
        graph = getattr(self, 'block_graph', None)
        if graph is not None:
            graph.invalidate()
        boundaries = getattr(self, 'query_boundaries', None)
        if boundaries is not None:
            boundaries.invalidate()
        self._rewind_position = checkpoint.position
        self._rewind_committed = False
        if self.query_window_cache is not None:
            if preserve_prefix:
                if prebuild is None:
                    self.query_window_cache.rewind(checkpoint.position)
            else:
                self.query_window_cache.clear()
        return spans

    def attach_rewind_completion(self, fence):
        if self._rewind_position is None or self._rewind_committed:
            raise RuntimeError('No uncommitted authority rewind')
        self.memory.attach_completion(fence)

    def poll_rewind(self):
        """Retire the old index, then yield until all suffix I/O readers release."""
        if self._rewind_position is None or self.closing:
            raise RuntimeError('No live authority rewind')
        if not self._rewind_committed:
            if not self.memory.commit_rewind():
                return False
            self._rewind_committed = True
            boundaries = getattr(self, 'query_boundaries', None)
            if boundaries is not None:
                boundaries.rewind(self._rewind_position)
            for key, record in list(self.records.items()):
                if record.history_begin >= self._rewind_position:
                    self.discarded_records[key] = self.records.pop(key)
                    self._unsubmitted.pop(key, None)
                    if not record.archived:
                        self._resident_count -= 1
            if self.gc is not None:
                # GC accounting is branch-relative. Previously collected suffix
                # records are not holes in the now-shorter accepted prefix.
                self.gc.reclaimed_records = self.history_end // self.shape[1] - len(self.records)
                if self.gc.reclaimed_records < 0:
                    raise RuntimeError('Rewound index exceeds its sealed prefix')
        self.poll()
        if self.discarded_records:
            return False
        self._rewind_position = None
        self._rewind_committed = False
        return True

    def close(self):
        self.closing = True
        if getattr(self, '_prefilled', None):
            self.discard_prefilled()
        recall_cache = getattr(self, '_recall_feature_cache', None)
        if recall_cache is not None:
            recall_cache.clear()
        pressure_queries = getattr(self, 'pressure_query_window', None)
        if pressure_queries is not None:
            pressure_queries.close()
        redundancy = getattr(self, 'redundancy_history', None)
        if redundancy is not None:
            redundancy.invalidate()
        thinking = getattr(self, 'thinking_history', None)
        if thinking is not None:
            thinking.close()
        prebuild = getattr(self, 'fps_prebuild', None)
        if prebuild is not None:
            prebuild.cancel(permanent=True)
        if self.query_window_cache is not None:
            self.query_window_cache.clear()
        self.guided_seed = None
        historical = getattr(self.gc, 'native_history', False)
        graph = getattr(self, 'block_graph', None)
        if graph is not None:
            graph.close()
        boundaries = getattr(self, 'query_boundaries', None)
        if boundaries is not None:
            boundaries.close()
        if not historical:
            self.stop_gc()
        self.poll()
        # Diagnostic: the first step a failed close stopped at (eviction audits).
        self.close_blocker = 'writes' if self.writes else None
        if self.writes or not self.memory.close(background_retire=historical):
            self.close_blocker = self.close_blocker or 'memory'
            return False
        if historical and not self.gc.close():
            self.close_blocker = getattr(self.gc, 'close_blocker', 'historical_gc')
            return False
        all_reclaimed = True
        for records in (self.records, self.discarded_records):
            for key, record in list(records.items()):
                if historical:
                    released = self.archive.retire_cpu_record(self.memory.request_id, record.handle)
                else:
                    result = self.memory.repository.reclaim(record.handle,
                        lambda entry: self.archive.reclaim(self.memory.request_id, entry))
                    released = result == _native.ReclaimResult.RECLAIMED
                if released:
                    if historical:
                        self.gc.defer_index(key, record)
                    self._return_record_buffers(record)
                    del records[key]
                    if records is self.records:
                        self._unsubmitted.pop(key, None)
                        if not record.archived:
                            self._resident_count -= 1
                else:
                    all_reclaimed = False
                    self.close_blocker = 'record_release'
        if all_reclaimed:
            if historical and not self.gc.index_released():
                self.close_blocker = 'index_release'
                return False
            if self._busy_record_buffers:
                raise RuntimeError('Authority closed with live record staging')
            self._record_buffers.clear()
            self._record_buffer_views.clear()
            self._free_record_buffers.clear()
            if historical:
                self.archive.unregister_history_scope(self.memory.request_id, self.gc.scope)
        return all_reclaimed

    def stop_gc(self):
        """Freeze the live index before final checksum iteration and teardown."""
        if self.gc is not None:
            self.gc.stop()
            self._drain_gc()

    @property
    def gc_reclaimed_records(self):
        return self.gc.reclaimed_records if self.gc is not None else 0


# The fast reduction is qualified against the pinned ATen outer-reduction
# kernel on x86; other Torch versions/CPUs use the reference operation.
_FUSED_DECODE_SUM = (torch.__version__.split('+')[0].startswith('2.13.') and
                     torch.backends.cpu.get_cpu_capability() in ('AVX2', 'AVX512'))


def append_authority_batch(entries):
    """Copy a capacity-admitted prefix of one packet in one GIL-free call.

    Each entry is (authority, raw, values, append keyword arguments). The caller
    has already clipped rejected/EOS/cancelled rows and observed the DMA fence.
    Completed original-MeanK pages are copied without an incremental reduction;
    their finalizer recomputes the sum in token order before publication. Small
    qualified BF16/FP8 decode reductions use the same FP32 addition order; other
    geometries retain the reference Torch reduction. No prefix is published
    before all admitted copies complete. Backpressure consumes only the returned prefix.
    """
    if len({id(entry[0]) for entry in entries}) != len(entries):
        raise ValueError('A history batch must contain distinct authority owners')
    plans, jobs = [], []
    finalized = {}
    prefilled_keys = {}
    import time as _time
    _t0 = _time.perf_counter()
    for authority, raw, values, kwargs in entries:
        # Later queued packets can start beyond an EOS/cancelled prefix. They
        # are fully discarded, as on the reference drain path, without opening
        # a transaction or requiring authority.history_end to reach those rows.
        if kwargs.get('accepted') == 0:
            plans.append((authority, (raw, values, 0, [])))
            continue
        prepared = authority._prepare_append(raw, values, **kwargs)
        if prepared is None:
            break
        plans.append((authority, prepared))
        source_raw = raw.view(torch.uint8).numpy()
        source_values = values.numpy()
        for key, record, offset, start, n in prepared[3]:
            if getattr(record, 'prefilled', False):
                # Streamed during the forward: bytes are in place and the page's
                # MeanK sums are final; no copy or finalizer runs for it here.
                from .streaming_publication import verify_enabled, verify_page
                if verify_enabled():
                    verify_page(record, raw, values, start, n)
                prefilled_keys.setdefault(authority, set()).add(key)
                continue
            # The original MeanK finalizer recomputes every completed page in
            # token order. Its result does not use the incremental sum, so do
            # not widen/reduce a prefix that is about to be overwritten.
            copy_only = (authority.mean_k_backend == 'qw3_cuda' and
                         raw.dtype in (torch.bfloat16, torch.float8_e4m3fn) and
                         offset + n == authority.shape[1])
            fused = (raw.dtype in (torch.bfloat16, torch.float8_e4m3fn) and n <= 4 and
                     raw.shape[2] * raw.shape[3] % 64 == 0 and _FUSED_DECODE_SUM)
            delta = (None if copy_only or fused else
                     raw[:record.key_sum.shape[0], start:start+n].float().sum(dim=1).numpy())
            if record.scatter_views is None:
                record.scatter_views = (record.raw.view(torch.uint8).numpy(),
                                        record.values.numpy(), record.key_sum.numpy())
            destination_raw, destination_values, sums = record.scatter_views
            job = (source_raw, source_values, destination_raw,
                   destination_values, delta, sums, start, offset, n)
            if copy_only and authority.cpu_fuse_record_mean:
                jobs.append((*job, True, True))
                finalized.setdefault(authority, set()).add(key)
            else:
                jobs.append((*job, True) if copy_only else job)
    _t1 = _time.perf_counter()
    observed = _native.scatter_history_batch(jobs)
    _t2 = _time.perf_counter()
    expected = sum(len(keys) for keys in finalized.values())
    if expected and (type(observed) is not int or observed != expected):
        raise RuntimeError('Native fused record mean did not execute all requested pages')
    for authority, prepared in plans:
        keys = finalized.get(authority, ())
        if keys:
            authority.fused_mean_stats['records'] += len(keys)
            authority.fused_mean_stats['batches'] += 1
        streamed = prefilled_keys.get(authority)
        if streamed:
            keys = set(keys) | streamed
            authority.streamed_pages = getattr(authority, 'streamed_pages', 0) + len(streamed)
        authority._commit_prepared_append(prepared, finalized_records=keys)
    from ..core.publication_trace import emit as _trace
    _trace('append_batch', records=len(jobs), prepare=_t1-_t0, scatter=_t2-_t1,
           commit=_time.perf_counter()-_t2)
    return len(plans)


class _HeldDirectRead:
    """Authority leases stay held until the drive has written the destination."""
    def __init__(self, pending, leases):
        self._pending, self._leases = pending, leases

    def wait(self):
        try:
            return self._pending.wait()
        finally:
            leases, self._leases = self._leases, ()
            for lease in leases:
                lease.reset()
