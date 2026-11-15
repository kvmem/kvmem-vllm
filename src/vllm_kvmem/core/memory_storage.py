"""Bounded immutable CPU authority using the existing publication/lease protocol.

There is no disk fallback, eviction, compression, or raw-K precision change.
The owner must reserve each request's complete history before admission.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, local
from collections import Counter
import hashlib
import json
import time

from .. import _native
from .storage import AsyncArchive
from . import settings as _settings

CPU_REGION_GATHER_PARALLEL_MIN_BYTES = 32 * 1024**2


def region_gather_parallel_min_bytes():
    """Opt-in KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB: split smaller layer-region
    copies (8 MiB per K or V chunk) across the existing gather workers. Each
    worker copies a disjoint record range with the same native routine, so the
    bytes are identical; only the copy's latency on the refresh path changes."""
    import os
    value = _settings.get('KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB')
    if value is None:
        return globals()['CPU_REGION_GATHER_PARALLEL_MIN_BYTES']   # read at call time
    if not value.isdigit() or not 1 <= int(value) <= 1024:
        raise ValueError('KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB must be an integer in [1, 1024]')
    return int(value) * 1024**2


def _skip_digest():
    """Opt-in: skip per-record SHA-256 that nothing reads while verification is off.

    Digests are consumed only by verify_archive_on_finish checks (finish,
    retirement, preserved sessions) and a session-restore diagnostic. Callers
    must only enable this with verify_archive_on_finish=False.
    """
    import os
    return _settings.get('KVMEM_ARCHIVE_SKIP_DIGEST') == '1'


class SplitHistoricalIndexOwner:
    """Independent sum/mean native buffers with a joint release completion.

    Tensor views retain the respective native owner through the buffer
    protocol. This wrapper stores no tensors, so releasing sums cannot create
    a Python ownership cycle or move their deallocation onto the caller.
    """
    __slots__ = ('shape', '_elements', '_sum_owner', '_mean_owner', 'completion')

    def __init__(self, arena, shape):
        import math
        import sys
        from .historical_gc import IndexReleaseFence
        self.shape = tuple(shape)
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError('Split historical index requires positive dimensions')
        self._elements = math.prod(self.shape)
        if self._elements > sys.maxsize // 6:
            raise OverflowError('Split historical index allocation is too large')
        self._sum_owner = self._mean_owner = None
        try:
            self._sum_owner = arena._allocate_index(self._elements * 4)
            self._mean_owner = arena._allocate_index(self._elements * 2)
            self.completion = IndexReleaseFence(
                self._sum_owner.completion, self._mean_owner.completion)
        except BaseException:
            # Even an exception traceback retaining this wrapper must not
            # strand a successful first allocation after the second failed.
            self._sum_owner = self._mean_owner = None
            raise

    def sum_view(self):
        import torch
        if self._sum_owner is None:
            raise RuntimeError('Historical index sums have been released')
        return torch.frombuffer(self._sum_owner, dtype=torch.float32,
            count=self._elements).view(self.shape)

    def mean_view(self):
        import torch
        return torch.frombuffer(self._mean_owner, dtype=torch.float16,
            count=self._elements).view(self.shape)

    def release_sums(self):
        """Drop our sum ownership; outstanding tensor aliases remain valid."""
        self._sum_owner = None


def gather_records_python(destination, sources, record_bytes, offset, length, begin, end):
    """Reference gather for immutable CPU records; also supports older extensions."""
    view = memoryview(destination)
    if (type(record_bytes) is not int or record_bytes <= 0 or
            any(type(value) is not int for value in (offset, length, begin, end)) or
            not 0 <= offset < offset+length <= record_bytes or
            not 0 <= begin <= end <= len(sources) or
            view.readonly or view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous or
            view.nbytes != len(sources)*length or
            any(not _immutable_record(source, record_bytes) for source in sources[begin:end])):
        raise ValueError('Invalid immutable CPU record gather')
    view = view.cast('B')
    for i in range(begin, end):
        view[i*length:(i+1)*length] = memoryview(sources[i])[offset:offset+length]
    return (end-begin)*length


def _immutable_record(source, record_bytes):
    if type(source) is bytes:
        return len(source) == record_bytes
    native_read = getattr(_native, 'HistoricalPayloadRead', ())
    return isinstance(source, native_read) and bool(source) and memoryview(source).nbytes == record_bytes


def gather_records_native(destination, sources, record_bytes, offset, length, begin, end):
    """Copy disjoint ranges with the existing GIL-releasing native memcpy path.

    Missing functions in an older extension use the reference path. A present
    native function's validation or copy error must propagate, never be hidden
    by retrying a partly written destination through another backend.
    """
    if offset == 0 and length == record_bytes and all(type(source) is bytes for source in sources[begin:end]):
        native = getattr(_native, 'gather_cached_records', None)
        if callable(native):
            return native(destination, sources, record_bytes, begin, end)
    else:
        native = getattr(_native, 'gather_record_region', None)
        if callable(native):
            return native(destination, sources, record_bytes, offset, length, begin, end)
    return gather_records_python(destination, sources, record_bytes, offset, length, begin, end)


def copy_record_slices(source, offsets, record_bytes):
    """Own each sealed record without retaining its whole source batch.

    The native copy releases the GIL after allocating independent immutable
    bytes. Older extensions retain the reference implementation; errors from
    an available native implementation must propagate before publication.
    """
    native = getattr(_native, 'copy_record_slices', None)
    if callable(native):
        return native(source, offsets, record_bytes)
    return [bytes(source[offset:offset+record_bytes]) for offset in offsets]


@dataclass(frozen=True)
class MemoryAllocation:
    block_id: int
    slot: int
    generation: int


@dataclass(frozen=True)
class MemorySpan:
    slot: int
    offset: int
    size: int


class MemoryTier:
    def __init__(self, record_bytes, capacity):
        self.record_bytes, self.capacity = record_bytes, capacity
        self.allocations, self.records = {}, {}
        self.generation = 0
        self.cpu_peak_bytes = 0
        self.lock = RLock()
        self.gather = None

    @property
    def used_slots(self):
        with self.lock:
            return len(self.allocations)

    def allocate(self, block_id):
        with self.lock:
            if not 0 <= block_id < self.capacity or block_id in self.allocations:
                raise ValueError('Invalid or already owned CPU archive slot')
            self.generation += 1
            value = MemoryAllocation(block_id, block_id, self.generation)
            self.allocations[block_id] = value
            return value

    def release(self, allocation):
        with self.lock:
            if self.allocations.get(allocation.slot) != allocation:
                return False
            del self.allocations[allocation.slot]
            self.records.pop(allocation.slot, None)
            return True

    def publish_records(self, allocations, records):
        """Atomically install independent immutable bytes for exact generations."""
        started = time.perf_counter_ns()
        allocations, records = tuple(allocations), tuple(records)
        pending = {}
        try:
            if (len(allocations) != len(records) or
                    any(type(a) is not MemoryAllocation for a in allocations) or
                    len({a.slot for a in allocations}) != len(allocations) or
                    any(type(record) is not bytes or len(record) != self.record_bytes for record in records)):
                raise ValueError('Invalid owned CPU record publication')
            pending = {a.slot: record for a, record in zip(allocations, records)}
            with self.lock:
                if any(self.allocations.get(a.slot) != a for a in allocations):
                    raise ValueError('CPU archive allocation changed during record copy')
                if any(slot in self.records for slot in pending):
                    raise ValueError('Cannot overwrite immutable CPU authority')
                try:
                    self.records.update(pending)
                    self.cpu_peak_bytes = max(self.cpu_peak_bytes,
                                              len(self.records) * self.record_bytes)
                except BaseException:
                    for slot in pending:
                        self.records.pop(slot, None)
                    raise
            return dict(bytes=len(records)*self.record_bytes, syscalls=0,
                        duration_ns=time.perf_counter_ns()-started)
        finally:
            records = ()
            pending.clear()

    def io(self, spans, buffer, write):
        start = time.perf_counter_ns()
        view = memoryview(buffer).cast('B')
        spans = tuple(spans)
        with self.lock:
            if len({s.slot for s in spans}) != len(spans):
                raise ValueError('Duplicate CPU archive I/O span')
            for span in spans:
                if (span.slot not in self.allocations or span.size != self.record_bytes
                        or span.offset < 0 or span.offset+span.size > view.nbytes):
                    raise ValueError('Invalid CPU archive I/O extent')
                if write and span.slot in self.records:
                    raise ValueError('Cannot overwrite immutable CPU authority')
                if not write and span.slot not in self.records:
                    raise KeyError('CPU archive record is not published')
            if write:
                allocations = tuple(self.allocations[span.slot] for span in spans)
            else:
                if view.readonly:
                    raise ValueError('CPU archive read destination is readonly')
                # Borrow immutable bytes under the publication lock, then let
                # native memcpy release the GIL outside it. Archive read leases
                # protect the generation; these references retain the payload
                # even if a direct tier caller releases its slot concurrently.
                sources = [self.records[span.slot] for span in spans]
        read_stats = {}
        if write:
            # Copying a long prefill snapshot must neither hold the publication
            # lock nor monopolize the engine's GIL. Keep exact per-record owners
            # rather than slices that retain an unaccounted full batch slab.
            started = time.perf_counter()
            records = copy_record_slices(view, [span.offset for span in spans], self.record_bytes)
            if (len(records) != len(spans) or
                    any(type(record) is not bytes or len(record) != self.record_bytes for record in records)):
                raise RuntimeError('Invalid owned CPU record copy')
            copy_seconds = time.perf_counter() - started
            self.publish_records(allocations, records)
            read_stats = dict(seal_record_copy_seconds=copy_seconds)
        else:
            copy_started = time.perf_counter()
            packed = (view.nbytes == len(spans)*self.record_bytes and
                      all(span.offset == i*self.record_bytes for i, span in enumerate(spans)))
            if not sources:
                tasks = 0
            elif packed:
                if self.gather is not None:
                    tasks = self.gather(view, sources, 0, self.record_bytes)
                else:
                    gather_records_native(view, sources, self.record_bytes, 0, self.record_bytes, 0, len(sources))
                    tasks = 1
            else:
                # The archive always requests packed records. Preserve the
                # tier's more general ordered/gapped destination-span contract.
                for span, source in zip(spans, sources):
                    view[span.offset:span.offset+span.size] = source
                tasks = 1
            read_stats = dict(cache_gather_tasks=tasks,
                              cache_copy_seconds=time.perf_counter()-copy_started)
        # Transfer/assembly callers sum these counters across packed regions.
        # Backend identity belongs in io_usage(), not this numeric contract.
        return {'bytes':sum(s.size for s in spans), 'syscalls':0,
                'duration_ns':time.perf_counter_ns()-start, **read_stats}


def _buffer_address(buffer):
    import numpy as np
    return np.frombuffer(buffer, dtype=np.uint8).__array_interface__['data'][0]


def _prepacked_records(views, record_bytes):
    """Per record pair: its bytes object when the pair is exactly that object's
    two halves (in order, complete), else None. None overall when no pair is."""
    result, found = [], False
    for i in range(0, len(views) - 1, 2):
        first, second = views[i], views[i+1]
        owner = getattr(first, 'obj', None)
        record = None
        if (type(owner) is bytes and getattr(second, 'obj', None) is owner and
                len(owner) == record_bytes and first.nbytes + second.nbytes == record_bytes):
            base = _buffer_address(owner)
            if _buffer_address(first) == base and _buffer_address(second) == base + first.nbytes:
                record, found = owner, True
        result.append(record)
    return result if found else None


class MemoryArchive(AsyncArchive):
    def __init__(self, *args, gather_workers=8, cpu_direct_record_seal=False, cpu_seal_hash_workers=1,
                 cpu_seal_pack_workers=1, cpu_seal_serial_allocate=False,
                 native_payloads=False, payload_capacity=None, **kwargs):
        if type(gather_workers) is not int or not 1 <= gather_workers <= 16:
            raise ValueError('CPU gather workers must be between 1 and 16')
        region_gather_parallel_min_bytes()   # validate the opt-in override early
        if type(cpu_direct_record_seal) is not bool:
            raise ValueError('cpu_direct_record_seal must be boolean')
        if type(cpu_seal_hash_workers) is not int or not 1 <= cpu_seal_hash_workers <= 8:
            raise ValueError('cpu_seal_hash_workers must be an integer in [1, 8]')
        if type(cpu_seal_pack_workers) is not int or not 1 <= cpu_seal_pack_workers <= 8:
            raise ValueError('cpu_seal_pack_workers must be an integer in [1, 8]')
        if cpu_seal_pack_workers > 1 and (not cpu_direct_record_seal or native_payloads):
            raise ValueError('cpu_seal_pack_workers > 1 requires direct CPU records without native payloads')
        if type(cpu_seal_serial_allocate) is not bool:
            raise ValueError('cpu_seal_serial_allocate must be boolean')
        if cpu_seal_serial_allocate:
            if not cpu_direct_record_seal or native_payloads or cpu_seal_pack_workers != 1:
                raise ValueError('cpu_seal_serial_allocate requires direct CPU records, no native payloads, and pack_workers=1')
            if not callable(getattr(_native, 'PrivateRecordPack', None)):
                raise ValueError('cpu_seal_serial_allocate requires the updated native extension')
        super().__init__(*args, **kwargs, memory_only=True, native_cpu_payloads=native_payloads,
                         **({'payload_capacity': payload_capacity} if payload_capacity is not None else {}))
        self._native_payload_arena = getattr(self.tier, 'payloads', None)
        self._history_worker = None
        self._history_scopes = {}
        self._history_totals = Counter()
        self._history_mode = None
        self._history_policy = None
        self._history_policy_params = {}
        # Optional terminal-only audit. Context is bounded by active scopes;
        # completed rows are written once and never retained in memory.
        self.history_scope_metrics_path = None
        self._history_scope_context = {}
        self._history_scope_metrics_records = 0
        self._history_scope_metrics_errors = 0
        self._history_scope_metrics_last_error = None
        self.cpu_direct_record_seal = cpu_direct_record_seal
        self.cpu_seal_hash_workers = cpu_seal_hash_workers
        self.cpu_seal_pack_workers = cpu_seal_pack_workers
        self.cpu_seal_serial_allocate = cpu_seal_serial_allocate
        self._seal_serial_allocate_batches_completed = 0
        self._seal_pack_parallel_batches_completed = 0
        self._seal_pack_parallel_shards_completed = 0
        self._seal_hash_parallel_batches_completed = 0
        self._seal_hash_parallel_shards_completed = 0
        self._seal_totals = dict(seal_batches=0, seal_records=0, seal_payload_bytes=0,
            seal_direct_records=0, seal_direct_bytes=0, seal_pack_seconds=0.,
            seal_hash_seconds=0., seal_cache_seconds=0., seal_record_copy_seconds=0.,
            seal_record_publish_seconds=0.)
        self.gather_workers = gather_workers
        # Archive jobs wait for their copies; using the archive executor for
        # these tasks would deadlock when its workers are all occupied by reads.
        from .portable import cap_workers
        self._gather_executor = ThreadPoolExecutor(max_workers=cap_workers(gather_workers),
                                                   thread_name_prefix='kvmem-cpu-gather')
        self._gather_control = RLock()
        self._gather_unusable = None
        self._gather_shutdown_started = False
        self._gather_shutdown_done = Event()
        self._gather_worker = local()
        self._region_gather_counts = dict(inline_calls=0, parallel_calls=0,
                                         inline_bytes=0, parallel_bytes=0)
        self.tier.gather = self._gather

    def _seal_views(self, keys, entries, spans, views):
        if self._native_payload_arena is not None:
            sources = []
            try:
                started = time.perf_counter()
                allocations = [entry.allocation for entry in entries]
                stats = self.tier.publish_record_views(allocations,
                    [views[2*i:2*i+2] for i in range(len(entries))])
                copied = time.perf_counter()
                sources = self.tier.borrow_records(allocations)
                digests = None if _skip_digest() else self._record_digests(sources)
                from .publication_trace import emit as _trace
                _trace('seal_done', path='native', records=len(entries), pack=copied-started,
                       hash=time.perf_counter()-copied, ns=str(keys[0])[:60] if keys else None)
                stats.update(record_digests=digests, seal_pack_seconds=copied-started,
                    seal_hash_seconds=time.perf_counter()-copied, seal_cache_seconds=0.,
                    seal_record_copy_seconds=0., seal_record_publish_seconds=0.,
                    seal_direct_records=len(entries), seal_direct_bytes=len(entries)*self.record_bytes)
                return stats
            finally:
                views = None
                sources.clear()
        if not self.cpu_direct_record_seal:
            try:
                return super()._seal_views(keys, entries, spans, views)
            finally:
                views = None
        records = []
        record = None
        try:
            started = time.perf_counter()
            records = self._pack_records(views)
            if (len(records) != len(entries) or
                    any(type(record) is not bytes or len(record) != self.record_bytes for record in records)):
                raise RuntimeError('Invalid owned CPU sealed record')
            packed = time.perf_counter()
            digests = None if _skip_digest() else self._record_digests(records)
            hashed = time.perf_counter()
            from .publication_trace import emit as _trace
            _trace('seal_done', path='direct', records=len(records), pack=packed-started,
                   hash=hashed-packed, ns=str(keys[0])[:60] if keys else None)
            # No batch slab or record views survive publication. The exact
            # bytes objects hashed above become the immutable authority owners.
            stats = self.tier.publish_records([e.allocation for e in entries], records)
            stats.update(record_digests=digests, seal_pack_seconds=packed-started,
                seal_hash_seconds=hashed-packed, seal_cache_seconds=0.,
                seal_record_copy_seconds=0., seal_record_publish_seconds=time.perf_counter()-hashed,
                seal_direct_records=len(records), seal_direct_bytes=len(records)*self.record_bytes)
            return stats
        finally:
            views = record = None
            records.clear()

    def _pack_range(self, views, start, end):
        """Own one disjoint record shard; native memcpy releases the GIL."""
        self._gather_worker.active = True
        pairs = None
        try:
            pairs = views[2*start:2*end]
            return _native.pack_record_pairs(pairs, self.record_bytes)
        finally:
            # A failed Future must not retain sealed tensor buffer exports.
            views = pairs = None
            self._gather_worker.active = False

    def _pack_records(self, views):
        # Zero-copy streamed pages (streaming_publication): a pair of views that
        # is exactly one private, complete bytes record is adopted as is.
        adopted = subset = copied = None
        try:
            adopted = _prepacked_records(views, self.record_bytes)
            if adopted is None:
                return self._pack_copied(views)
            missing = [i for i, record in enumerate(adopted) if record is None]
            if missing:
                subset = [v for i in missing for v in views[2*i:2*i+2]]
                copied = self._pack_copied(subset)
                for i, record in zip(missing, copied):
                    adopted[i] = record
            with self._lock:
                self._seal_zero_copy_records = (getattr(self, '_seal_zero_copy_records', 0) +
                                                len(adopted) - len(missing))
            return adopted
        finally:
            # A failed future's traceback must not retain any source export.
            views = subset = copied = adopted = None

    def _pack_copied(self, views):
        """Pack an admitted batch with bounded shards and one copy per record.

        write_from_views already validates the complete source extent and owns
        generation leases plus 2x byte admission. Workers only allocate private
        independent records; hashing and publication wait for every shard.
        """
        if self.cpu_seal_serial_allocate:
            from .private_record_pack import serial_allocate_pack
            try:
                return serial_allocate_pack(self, views, workers=min(8, self.gather_workers))
            finally:
                views = None
        jobs, parts, error, job, records = [], [], None, None, None
        try:
            if getattr(self._gather_worker, 'active', False):
                raise RuntimeError('CPU gather worker cannot recursively pack')
            pack_pairs = getattr(_native, 'pack_record_pairs', None)
            if not callable(pack_pairs):
                # Older extensions retain their original serial copy path.
                records = []
                for i in range(0, len(views), 2):
                    records.append(_native.pack_record_views(views[i:i+2]))
                return records
            if self.cpu_seal_pack_workers == 1:
                return pack_pairs(views, self.record_bytes)
            workers = min(self.cpu_seal_pack_workers, len(views)//2)
            with self._gather_control:
                if self._gather_unusable is not None:
                    raise RuntimeError('CPU gather pool unavailable: '+self._gather_unusable)
                if workers > 1:
                    try:
                        for worker in range(workers):
                            jobs.append(self._gather_executor.submit(self._pack_range, views,
                                (len(views)//2)*worker//workers,
                                (len(views)//2)*(worker+1)//workers))
                    except BaseException as failure:
                        error = failure
                        self._gather_unusable = 'submission failed ('+type(failure).__name__+')'
            if error is not None:
                # submit can start a task without returning its Future. Drain
                # the whole poisoned pool before source leases are released.
                self._stop_gather_pool()
            if workers <= 1:
                return pack_pairs(views, self.record_bytes)
            for job in jobs:
                try:
                    parts.append(job.result())
                except BaseException as failure:
                    if error is None:
                        error = failure
            if error is not None:
                raise error
            records = [record for part in parts for record in part]
            with self._lock:
                self._seal_pack_parallel_batches_completed += 1
                self._seal_pack_parallel_shards_completed += len(jobs)
            return records
        finally:
            views = job = records = None
            jobs.clear()
            parts.clear()

    def _hash_range(self, records, start, end):
        self._gather_worker.active = True
        try:
            return super()._record_digests(records[start:end])
        finally:
            records = None
            self._gather_worker.active = False

    def _record_digests(self, records):
        jobs, error, parts = [], None, []
        try:
            if self.cpu_seal_hash_workers == 1:
                return super()._record_digests(records)
            if getattr(self._gather_worker, 'active', False):
                raise RuntimeError('CPU gather worker cannot recursively seal')
            workers = min(self.cpu_seal_hash_workers, len(records))
            with self._gather_control:
                if self._gather_unusable is not None:
                    raise RuntimeError('CPU gather pool unavailable: '+self._gather_unusable)
                if workers > 1:
                    try:
                        # At most eight shards per admitted archive job; reuse
                        # the existing bounded gather pool, never one task per record.
                        for worker in range(workers):
                            jobs.append(self._gather_executor.submit(self._hash_range, records,
                                len(records)*worker//workers, len(records)*(worker+1)//workers))
                    except BaseException as failure:
                        error = failure
                        self._gather_unusable = 'submission failed ('+type(failure).__name__+')'
            if error is not None:
                # submit can enqueue/start work and then raise without returning
                # its Future. Cancel unseen queued tasks and join running ones.
                self._stop_gather_pool()
            if workers <= 1:
                return super()._record_digests(records)
            for job in jobs:
                try:
                    parts.append(job.result())
                except BaseException as failure:
                    if error is None:
                        error = failure
            if error is not None:
                raise error
            # These counters describe completed parallel hashing, before the
            # caller's publication transaction; failed shard batches add none.
            with self._lock:
                self._seal_hash_parallel_batches_completed += 1
                self._seal_hash_parallel_shards_completed += len(jobs)
            return tuple(digest for part in parts for digest in part)
        finally:
            # All workers have drained before callers release source owners or
            # roll back allocations; failed archive Futures retain no records here.
            records = None
            jobs.clear()

    def _copy_range(self, *args):
        self._gather_worker.active = True
        try:
            return gather_records_native(*args)
        finally:
            self._gather_worker.active = False

    def _stop_gather_pool(self):
        # Only archive I/O/caller threads may join the copy pool. A callback
        # cannot recursively gather or close its own pool and join itself.
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot close its own pool')
        with self._gather_control:
            self._gather_unusable = self._gather_unusable or 'closed'
            owner = not self._gather_shutdown_started
            self._gather_shutdown_started = True
        if owner:
            # submit() can put a task in the queue and then fail to start a
            # thread, without returning its Future. Cancel unseen queued tasks
            # and join all running copies before releasing the caller's buffers
            # or archive leases. wait=True alone leaves tasks queued when the
            # very first worker failed to start.
            self._gather_executor.shutdown(wait=True, cancel_futures=True)
            self._gather_shutdown_done.set()
        else:
            self._gather_shutdown_done.wait()

    def _gather(self, destination, sources, offset, length):
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot recursively gather')
        workers = min(self.gather_workers, len(sources))
        region = length != self.record_bytes
        # A layer's 256-record K or V region is only 8 MiB. Reuse the native
        # inline copy below 32 MiB instead of submitting eight 1-MiB jobs.
        # Full-record reads retain their existing parallel threshold.
        threshold = region_gather_parallel_min_bytes() if region else 8*1024**2
        parallel = workers > 1 and len(sources)*length >= threshold
        jobs, error = [], None
        with self._gather_control:
            # Reject all subsequent reads, including the inline small-copy
            # path, after a submission failure. Never resurrect the executor.
            if self._gather_unusable is not None:
                raise RuntimeError('CPU gather pool unavailable: '+self._gather_unusable)
            if parallel:
                try:
                    for worker in range(workers):
                        jobs.append(self._gather_executor.submit(self._copy_range,
                            destination, sources, self.record_bytes, offset, length,
                            len(sources)*worker//workers, len(sources)*(worker+1)//workers))
                except BaseException as failure:
                    error = failure
                    # Store no traceback/source buffer in the permanent state.
                    # The lock prevents another read from submitting while the
                    # failing submit's unseen task is being drained/cancelled.
                    self._gather_unusable = 'submission failed ('+type(failure).__name__+')'
        if error is not None:
            self._stop_gather_pool()
        if not sources:
            return 0
        if not parallel:
            gather_records_native(destination, sources, self.record_bytes, offset, length, 0, len(sources))
            if region:
                with self._gather_control:
                    self._region_gather_counts['inline_calls'] += 1
                    self._region_gather_counts['inline_bytes'] += len(sources)*length
            return 1
        # No destination, immutable source reference or archive lease may be
        # released while another worker is still copying, including on failure.
        for job in jobs:
            try:
                job.result()
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error
        if region:
            with self._gather_control:
                self._region_gather_counts['parallel_calls'] += 1
                self._region_gather_counts['parallel_bytes'] += len(sources)*length
        return len(jobs)

    def _gather_region(self, destination, sources, offset, length):
        return self._gather(destination, sources, offset, length)

    def _paired_copy_range(self, destination, sources, regions, begin, end):
        """One disjoint destination shard on the existing bounded copy pool."""
        self._gather_worker.active = True
        try:
            return _native.gather_record_regions(destination, sources, self.record_bytes,
                                                regions, begin, end)
        finally:
            destination = sources = regions = None
            self._gather_worker.active = False

    def _gather_regions_paired(self, destination, sources, regions, *, workers=1):
        """One explicitly requested native copy; keep pool failure semantics."""
        if type(workers) is not int or workers not in (1,2) or workers > self.gather_workers:
            raise ValueError('Paired gather permits one or two existing CPU workers')
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot recursively gather')
        started = time.perf_counter_ns()
        region_count = len(regions)
        actual_workers = min(workers, max(1,len(sources)))
        jobs, parts, error, job = [], [], None, None
        try:
            with self._gather_control:
                if self._gather_unusable is not None:
                    raise RuntimeError('CPU gather pool unavailable: '+self._gather_unusable)
                if actual_workers > 1:
                    try:
                        for worker in range(actual_workers):
                            jobs.append(self._gather_executor.submit(self._paired_copy_range,
                                destination, sources, regions, len(sources)*worker//actual_workers,
                                len(sources)*(worker+1)//actual_workers))
                    except BaseException as failure:
                        error = failure
                        self._gather_unusable = 'submission failed ('+type(failure).__name__+')'
            if error is not None:
                # submit may start work without returning its Future. Join
                # even that unseen producer before releasing caller storage.
                self._stop_gather_pool()
            if actual_workers == 1:
                stats = _native.gather_record_regions(destination, sources, self.record_bytes,
                                                     regions, 0, len(sources))
            else:
                for job in jobs:
                    try:
                        parts.append(job.result())
                    except BaseException as failure:
                        if error is None:
                            error = failure
                if error is not None:
                    raise error
                # These native durations are SUMS of shard work, not elapsed
                # wall time; the independently recorded gather wall may be
                # shorter because the two memcpy calls overlap.
                stats = {key:sum(part[key] for part in parts) for key in
                    ('copied_bytes','source_prepare_ns','copy_ns','gil_release_ns',
                     'gil_reacquire_ns','total_ns')}
            stats.update(worker_count=actual_workers,
                         gather_wall_ns=time.perf_counter_ns()-started)
        finally:
            # All returned Futures and all unseen submissions have drained.
            # Do not retain their arguments through a saved failure traceback.
            destination = sources = regions = job = None
            jobs.clear();parts.clear()
        with self._gather_control:
            kind = 'parallel' if actual_workers > 1 else 'inline'
            self._region_gather_counts[kind+'_calls'] += region_count
            self._region_gather_counts[kind+'_bytes'] += stats['copied_bytes']
        return stats

    def _region_sources(self, keys):
        # Native region gather reads immutable record buffers directly, avoiding
        # a second cache copy or a full-record fallback slab.
        if self._native_payload_arena is not None:
            return self.tier.borrow_records([self._entries[key].allocation for key in keys])
        with self.tier.lock:
            return [self.tier.records[self._entries[key].allocation.slot] for key in keys]

    def prepare_region_reader(self, namespace, leases, *, keys=None, adopt=False):
        """Pin each immutable no-GC source once for one layerwise pressure plan.

        Opt-in (KVMEM_PRESSURE_READER_ADOPT): ``keys`` are the callers' already
        validated (namespace, slot, generation) of each lease, and ``adopt``
        transfers the live leases instead of cloning them (the caller then must
        not reset them). Sources, keys and lease lifetimes are otherwise identical.
        """
        from .prepared_regions import PreparedRegionReader
        from .archive_flags import layerwise_gc
        arena = self._native_payload_arena is not None
        if (arena and not layerwise_gc()) or not leases:
            raise ValueError('Prepared CPU regions require nonempty no-GC immutable sources')
        if keys is None:
            keys = [self.key(namespace, lease) for lease in leases]
        elif len(keys) != len(leases) or not namespace:
            raise ValueError('Prepared CPU source keys do not match their leases')
        if len(set(keys)) != len(keys):
            raise ValueError('Prepared CPU source identities must be unique')
        owned = []
        native = None
        if adopt:
            from .archive_flags import native_reader
            if native_reader():
                native = getattr(_native, 'archive_region_sources', None)
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            if arena:
                sources = self._historical_region_sources(keys)
            elif native is not None:
                # Same entry checks, sources and errors in one native call.
                with self.tier.lock:
                    sources = native(self._entries, self.tier.records, list(keys), self.record_bytes)
            else:
                entries = [self._entries.get(key) for key in keys]
                if any(entry is None or not entry.ready for entry in entries):
                    raise KeyError('Prepared CPU record is absent, stale or unpublished')
                sources = self._region_sources(keys)
                if any(type(source) is not bytes or len(source) != self.record_bytes for source in sources):
                    raise ValueError('Prepared CPU regions require exact immutable record bytes')
            try:
                if adopt:
                    native_set = hasattr(leases, 'valid')
                    if not (leases.valid() if native_set else all(leases)):
                        raise ValueError('Stale prepared CPU source generation')
                    owned = leases if native_set else list(leases)
                else:
                    for lease in leases:
                        owned.append(lease.clone())
                        if not owned[-1]:
                            raise ValueError('Stale prepared CPU source generation')
                result = PreparedRegionReader(self, namespace, keys, sources, owned)
            except BaseException:
                if not adopt:      # adopted leases still belong to the caller
                    for lease in owned:
                        lease.reset()
                raise
            self._prepared_region_readers = getattr(self, '_prepared_region_readers', 0)+1
            return result

    def extend_region_reader(self, reader, namespace, leases, keys):
        """KVMEM_PRESSURE_READER_EXTEND: prepare_region_reader's adopt path for
        further records of the same plan (same entry checks, sources, unique
        identities and pin validity), appended to an open native reader."""
        from .archive_flags import layerwise_gc
        arena = self._native_payload_arena is not None
        if (arena and not layerwise_gc()) or not keys or not hasattr(leases, 'valid'):
            raise ValueError('Prepared CPU regions require nonempty no-GC immutable sources')
        if len(keys) != len(leases) or not namespace or reader.namespace != namespace:
            raise ValueError('Prepared CPU source keys do not match their leases')
        if len(set(keys)) != len(keys) or not set(keys).isdisjoint(reader.keys):
            raise ValueError('Prepared CPU source identities must be unique')
        native = getattr(_native, 'archive_region_sources', None)
        if native is None and not arena:
            raise RuntimeError('Prepared reader extension requires the native source export')
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            if arena:
                sources = self._historical_region_sources(keys)
            else:
                with self.tier.lock:
                    sources = native(self._entries, self.tier.records, list(keys), self.record_bytes)
            if not leases.valid():
                raise ValueError('Stale prepared CPU source generation')
            return reader.extend(keys, sources, leases)

    def _historical_region_sources(self, keys):
        """KVMEM_LAYERWISE_HISTORICAL_GC: one native read right per published
        record (the same rights the per-read path admits), held by the prepared
        reader until it closes. Absent, unpublished or retired records raise
        KeyError before any right is kept."""
        entries = [self._entries.get(key) for key in keys]
        if any(entry is None or not entry.ready for entry in entries):
            raise KeyError('Prepared CPU record is absent, stale or unpublished')
        sources = self.tier.borrow_records([entry.allocation for entry in entries])
        if any(not source or memoryview(source).nbytes != self.record_bytes for source in sources):
            raise ValueError('Prepared CPU regions require exact immutable record bytes')
        return sources

    def _admit_read_sources(self, entries):
        if self._native_payload_arena is not None:
            return self.tier.borrow_records([entry.allocation for entry in entries])
        return None

    def _validate_read_sources(self, entries, sources):
        if self._native_payload_arena is None or len(entries) != len(sources):
            raise ValueError('Invalid pre-admitted CPU read extent')
        for entry, source in zip(entries, sources):
            # The payload of a shared prefix may live in another descriptor's slot.
            expected = self.tier.records.get(entry.allocation.slot)
            if (not isinstance(source, _native.HistoricalPayloadRead) or not source or expected is None or
                    (source.handle.slot, source.handle.generation) !=
                    (expected.slot, expected.generation) or
                    not self._native_payload_arena.owns(source)):
                raise ValueError('Pre-admitted CPU read has a foreign allocation')
        return [source.clone() for source in sources]

    def record_payloads(self, namespace, leases):
        """Prefix images (serving/prefix_images.py): the immutable bytes object of
        each published record, by identity, no copy.

        The caller holds ``leases`` (repository pins of these generations) for the
        whole call; readiness and record size are checked under the archive lock.
        Only the bytes tier owns per-record ``bytes`` objects.
        """
        if self._native_payload_arena is not None:
            raise ValueError('Record payload objects exist only in the bytes CPU archive')
        if not leases:
            raise ValueError('Record payload lookup requires leases')
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError('Archive record missing, stale, or not yet published')
            with self.tier.lock:
                payloads = [self.tier.records.get(entry.allocation.slot) for entry in entries]
        if any(type(payload) is not bytes or len(payload) != self.record_bytes for payload in payloads):
            raise ValueError('Archive record payload is not one immutable bytes record')
        return payloads

    def alias_records(self, namespace, leases, payloads):
        """Prefix images: publish existing immutable record ``bytes`` under new keys.

        Each lease (a pinned, committed generation of the importing request's own
        repository) gets an ordinary new allocation in this archive whose payload
        is the given bytes object itself; nothing is copied. The entries are
        ready on return, as after a completed write. False when the archive has
        too few free records (nothing is published). Release is the ordinary
        reclaim path; it drops only this allocation's reference to the bytes.
        """
        if self._native_payload_arena is not None:
            return self._alias_native(namespace, leases, tuple(payloads))
        payloads = tuple(payloads)
        if not leases or len(leases) != len(payloads):
            raise ValueError('Aliased records need one payload per lease')
        if any(type(payload) is not bytes or len(payload) != self.record_bytes for payload in payloads):
            raise ValueError('Aliased records must be complete immutable bytes records')
        keys = [self.key(namespace, lease) for lease in leases]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate archive record in one alias batch')
        from .storage import _Entry
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            if any(key in self._entries for key in keys):
                raise ValueError('Cannot overwrite immutable archive records')
            if len(self._free_ids) < len(keys):
                return False
            entries = []
            try:
                for key in keys:
                    block_id = self._free_ids.pop()
                    try:
                        allocation = self.tier.allocate(block_id)
                        if allocation.slot < 0:
                            raise RuntimeError('Archive allocator capacity invariant violated')
                    except BaseException:
                        self._free_ids.append(block_id)
                        raise
                    entry = _Entry(allocation)
                    self._entries[key] = entry
                    self._namespace_counts[namespace] = self._namespace_counts.get(namespace, 0) + 1
                    entries.append(entry)
                self.tier.publish_records([entry.allocation for entry in entries], payloads)
            except BaseException:
                for key in keys:
                    self._release(key)
                raise
            for entry in entries:
                entry.ready = True
            self.aliased_records = getattr(self, 'aliased_records', 0) + len(keys)
        return True

    def record_payload_handles(self, namespace, leases):
        """Shared prefixes on the native arena: the payload handle of each published
        record (another descriptor's payload for an aliased record)."""
        if self._native_payload_arena is None:
            raise ValueError('Payload handles exist only with native CPU payloads')
        if not leases:
            raise ValueError('Record payload lookup requires leases')
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError('Archive record missing, stale, or not yet published')
            handles = self.tier.payload_handles([entry.allocation for entry in entries])
        if any(handle is None for handle in handles):
            raise KeyError('Archive record has no published payload')
        return handles

    def _alias_native(self, namespace, leases, handles):
        """alias_records on the native arena: a new descriptor per lease owning a
        share of an existing live payload. Nothing is copied; the arena's resident
        bytes do not change. The importing namespace's historical GC scope, when it
        has one, learns each record's backing and that it is shared."""
        if not leases or len(leases) != len(handles):
            raise ValueError('Aliased records need one payload per lease')
        if any(type(handle) is not _native.HistoricalPayloadHandle for handle in handles):
            raise ValueError('Native aliased records require payload handles')
        keys = [self.key(namespace, lease) for lease in leases]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate archive record in one alias batch')
        from .storage import _Entry
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            if any(key in self._entries for key in keys):
                raise ValueError('Cannot overwrite immutable archive records')
            if len(self._free_ids) < len(keys):
                return False
            entries = []
            try:
                for key, handle in zip(keys, handles):
                    block_id = self._free_ids.pop()
                    try:
                        allocation = self.tier.alias(block_id, handle)
                    except BaseException:
                        self._free_ids.append(block_id)
                        raise
                    entry = _Entry(allocation)
                    self._entries[key] = entry
                    self._namespace_counts[namespace] = self._namespace_counts.get(namespace, 0) + 1
                    entries.append(entry)
                context = self._backing_context(namespace)
            except BaseException:
                for key in keys:
                    self._release(key)
                raise
        try:
            if context is not None:
                self._publish_backing(context, leases, entries)
                for lease in leases:
                    context.mark_shared(lease)
        except BaseException:
            with self._lock:
                for key in keys:
                    self._release(key)
            raise
        with self._lock:
            for entry in entries:
                entry.ready = True
            self._mark_shared_keys(namespace, keys, aliases=True)
            self.aliased_records = getattr(self, 'aliased_records', 0) + len(keys)
        return True

    # --- shared-prefix pool accounting (docs/prefix-sharing.md, phase 2) ------

    def _mark_shared_keys(self, namespace, keys, *, aliases=False):
        shared = self.__dict__.setdefault('_namespace_shared', {}).setdefault(namespace, set())
        alias_counts = self.__dict__.setdefault('_namespace_aliases', {})
        for key in keys:
            if key[1:] not in shared:
                shared.add(key[1:])
                if aliases:
                    alias_counts[namespace] = alias_counts.get(namespace, 0) + 1
                    self.__dict__.setdefault('_alias_keys', set()).add(key)

    def namespace_shared_records(self, namespace):
        """(slot, generation) of this namespace's descriptors charged to the shared-prefix
        pool, and how many of them are aliases of another session's payloads."""
        with self._lock:
            shared = self.__dict__.get('_namespace_shared', {}).get(namespace)
            return (frozenset(shared) if shared else frozenset(),
                    self.__dict__.get('_namespace_aliases', {}).get(namespace, 0))

    def pool_records(self):
        records = getattr(self.tier, 'pool_records', None)
        return records() if records is not None else 0

    def share_to_pool(self, namespace, leases):
        """Registration of a prefix image on the native arena: each record's payload
        gains a pool owner (returned handles, held by the image) and the publisher's
        descriptor is charged to the pool from now on. The namespace's historical GC
        scope marks the records shared, so its byte budget no longer counts them.
        Returns the handles and how many records moved from private to pool charge."""
        if self._native_payload_arena is None:
            raise ValueError('Pool owners exist only with native CPU payloads')
        if not leases:
            raise ValueError('Pool sharing requires leases')
        keys = [self.key(namespace, lease) for lease in leases]
        handles = []
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError('Archive record missing, stale, or not yet published')
            try:
                for entry in entries:
                    handle = self.tier.payload_handles([entry.allocation])[0]
                    if handle is None or not self.tier.share_pool(handle):
                        raise ValueError('A pooled CPU payload must be live')
                    handles.append(handle)
            except BaseException:
                for handle in handles:
                    self.tier.release_pool(handle)
                raise
            transferred = sum(bool(self.tier.charge_pool(entry.allocation)) for entry in entries)
            self._mark_shared_keys(namespace, keys)
            context = self._history_scopes.get(namespace)
        if context is not None:
            for lease in leases:
                context.mark_shared(lease)
        return handles, transferred

    def release_pool(self, handles):
        """An image leaves: drop its pool owner of each payload."""
        for handle in handles:
            self.tier.release_pool(handle)

    def _release(self, key):
        released = super()._release(key)
        if released:
            shared = self.__dict__.get('_namespace_shared', {}).get(key[0])
            if shared is not None and key[1:] in shared:
                shared.discard(key[1:])
                if not shared:
                    del self._namespace_shared[key[0]]
                aliases = self.__dict__.get('_alias_keys')
                if aliases is not None and key in aliases:
                    aliases.discard(key)
                    remaining = self._namespace_aliases[key[0]] - 1
                    if remaining:
                        self._namespace_aliases[key[0]] = remaining
                    else:
                        del self._namespace_aliases[key[0]]
        return released

    def _payload_room(self, count):
        room = getattr(self.tier, 'payload_room', None)
        return True if room is None else room(count)

    def enable_history_gc(self, config):
        """Install one native manager for this CPU arena before serving."""
        if self._native_payload_arena is None:
            raise ValueError('Historical GC requires native CPU payload ownership')
        if self._history_worker is not None:
            raise RuntimeError('Historical GC manager is already installed')
        with self._lock:
            if self._pending or self._entries or self._closed:
                raise RuntimeError('Historical GC must be configured before archive use')
        self._history_worker = _native.HistoricalGcWorker(config, self._native_payload_arena)
        self._history_background = config.background
        self._history_mode = config.mode.name.lower()
        self._history_policy = config.policy
        self._history_policy_params = dict(config.policy_params)
        return self._history_worker

    def register_history_scope(self, namespace, repository):
        if self._history_worker is None:
            raise RuntimeError('Historical GC manager is not installed')
        if not namespace:
            raise ValueError('Historical namespace must be nonempty')
        with self._lock:
            if self._closed or namespace in self._history_scopes:
                raise ValueError('Historical namespace is closed or already registered')
        # No archive lock is held across native metadata allocation. Registration
        # has bounded admission; a busy/full native registry returns no scope.
        scope = self._history_worker.register_scope(repository)
        if scope is None:
            raise BufferError('Historical GC scope capacity is exhausted')
        with self._lock:
            if self._closed or namespace in self._history_scopes:
                scope.close()
                raise ValueError('Historical namespace changed during registration')
            self._history_scopes[namespace] = scope
        return scope

    def unregister_history_scope(self, namespace, scope):
        with self._lock:
            if self._history_scopes.get(namespace) is scope:
                stats = dict(scope.stats)
                self._history_totals.update(stats)
                del self._history_scopes[namespace]
                context = self._history_scope_context.pop(namespace, {})
                if self.history_scope_metrics_path is not None:
                    # close() has retained native final_stats_ before returning
                    # its metadata to the pool. Policy retirement counters do
                    # not include shutdown-only CPU/slot release. The identity
                    # guard above makes repeated unregister calls a no-op.
                    row = dict(event='history_scope_closed', at=time.time(), namespace=namespace,
                        mode=self._history_mode, policy=self._history_policy, stats=stats, **context)
                    try:
                        with self.history_scope_metrics_path.open('a') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                        self._history_scope_metrics_records += 1
                    except Exception as error:
                        # Observability must not strand a closed authority. A
                        # bounded error status makes a missing audit explicit.
                        self._history_scope_metrics_errors += 1
                        self._history_scope_metrics_last_error = repr(error)

    def set_history_scope_context(self, namespace, session_namespace, generation):
        """Bind a stable session to the original native request, for auditing."""
        with self._lock:
            if self.history_scope_metrics_path is not None and namespace in self._history_scopes:
                self._history_scope_context[namespace] = dict(
                    session_namespace=list(session_namespace), session_generation=generation)

    def history_stats(self):
        if self._history_worker is None:
            return None
        with self._lock:
            totals = self._history_totals.copy()
            features = Counter()
            for scope in self._history_scopes.values():
                totals.update(scope.stats)
                if self._history_policy == 'redundancy':
                    features.update(scope.feature_stats)
            scopes = len(self._history_scopes)
            metrics = (dict(path=str(self.history_scope_metrics_path),
                records=self._history_scope_metrics_records, errors=self._history_scope_metrics_errors,
                last_error=self._history_scope_metrics_last_error)
                if self.history_scope_metrics_path is not None else None)
        result = dict(mode=self._history_mode, policy=self._history_policy,
                    policy_params=dict(self._history_policy_params), active_scopes=scopes, **dict(totals),
                    metadata=self._history_worker.storage_stats)
        if self._history_policy == 'redundancy':
            result['active_features'] = dict(features)
            result['feature_reserved_bytes'] = getattr(self, '_historical_feature_reserved_bytes', 0)
        if metrics is not None:
            result['scope_metrics'] = metrics
        return result

    def allocate_index(self, shape, *, mean, split_sums=False):
        import math
        import torch
        if type(split_sums) is not bool or split_sums and not mean:
            raise ValueError('Split historical sums require a stored MeanK')
        if split_sums:
            owner = SplitHistoricalIndexOwner(self._native_payload_arena, shape)
            return owner.sum_view(), owner
        elements = math.prod(shape)
        owner = self._native_payload_arena._allocate_index(elements*(6 if mean else 4))
        sums = torch.frombuffer(owner, dtype=torch.float32, count=elements).view(shape)
        return sums, owner

    def history_used_bytes(self):
        if self._native_payload_arena is None:
            return self.tier.used_slots*self.record_bytes
        stats = self._native_payload_arena.stats
        return stats['used_bytes'] + stats['external_bytes']

    def _backing_context(self, namespace):
        # Called under the archive admission lock. The I/O job captures this
        # exact scope; it never discovers a replacement namespace on completion.
        return self._history_scopes.get(namespace)

    def _publish_backing(self, context, leases, entries):
        sources = self.tier.borrow_records([entry.allocation for entry in entries])
        try:
            for lease, source in zip(leases, sources):
                if not context.publish(lease, source, 0):
                    raise RuntimeError('Historical backing publication must retry')
        finally:
            sources.clear()

    def retire_cpu_record(self, namespace, handle):
        """Backend-only CPU cleanup after irreversible repository retirement.

        Existing GPU metadata leases do not delay these bytes. The arena itself
        fences actual CPU readers; this method only polls and recycles small
        allocation descriptors once native payload release is complete.
        """
        if self._native_payload_arena is None:
            raise RuntimeError('Independent CPU retirement requires native payload ownership')
        key = namespace, handle.slot, handle.generation
        if not self._lock.acquire(blocking=False):
            return False
        try:
            entry = self._entries.get(key)
            if entry is None:
                return True
            if not entry.ready:
                return False
            self.tier.retire_payload(entry.allocation)
            return self._release(key)
        finally:
            self._lock.release()

    def io_usage(self):
        with self._lock:
            hash_usage = dict(cpu_seal_hash_workers=self.cpu_seal_hash_workers,
                cpu_seal_serial_allocate=self.cpu_seal_serial_allocate,
                cpu_seal_serial_copy_workers=min(8,self.gather_workers) if self.cpu_seal_serial_allocate else 0,
                seal_serial_allocate_batches_completed=self._seal_serial_allocate_batches_completed,
                seal_hash_parallel_batches_completed=self._seal_hash_parallel_batches_completed,
                seal_hash_parallel_shards_completed=self._seal_hash_parallel_shards_completed,
                cpu_seal_pack_workers=self.cpu_seal_pack_workers,
                seal_pack_parallel_batches_completed=self._seal_pack_parallel_batches_completed,
                seal_pack_parallel_shards_completed=self._seal_pack_parallel_shards_completed)
        with self._gather_control:
            gather_usage = dict(cpu_region_gather_parallel_min_bytes=region_gather_parallel_min_bytes(),
                **{'cpu_region_gather_'+key:value for key,value in self._region_gather_counts.items()})
        native = self._native_payload_arena.stats if self._native_payload_arena is not None else None
        with self.tier.lock:
            resident_bytes = (native['used_bytes'] if native else
                              len(self.tier.records) * self.record_bytes)
            # The native arena can reclaim independently; its current ABI has
            # no exact high-water counter. Do not label a sampled value exact.
            peak_bytes = None if native else self.tier.cpu_peak_bytes
        return {**super().io_usage(), 'archive_backend':'cpu', 'disk_read_bytes':0,
                'disk_write_bytes':0, **hash_usage, **gather_usage,
                'cpu_authority_bytes':native['used_bytes'] if native else self.tier.used_slots*self.record_bytes,
                'cpu_resident_bytes': resident_bytes,
                'cpu_peak_bytes': peak_bytes,
                'cpu_peak_scope': ('unavailable_for_native_payload_arena' if native else
                                   'lifetime_published_payload_bytes'),
                **({'cpu_native_payloads': True, 'cpu_payload_readers': native['readers'],
                    'cpu_payload_released_bytes': native['released_bytes'],
                    'cpu_index_owners': native['external_owners'],
                    'cpu_index_bytes': native['external_bytes'],
                    'cpu_index_released_bytes': native['external_released_bytes']} if native else {})}

    def close(self):
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot close its own pool')
        if self._history_worker is not None:
            with self._lock:
                if self._pending:
                    return False
            self._history_worker.close()
            for namespace, scope in tuple(self._history_scopes.items()):
                if not scope.close():
                    return False
                self.unregister_history_scope(namespace, scope)
            if not self._history_worker.close():
                return False
        if not super().close():
            return False
        self._stop_gather_pool()
        if self._native_payload_arena is not None and not self._native_payload_arena.close():
            return False
        return True
