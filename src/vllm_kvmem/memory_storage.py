"""Bounded immutable CPU authority using the existing publication/lease protocol.

There is no disk fallback, eviction, compression, or raw-K precision change.
The owner must reserve each request's complete history before admission.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, local
from collections import Counter
import hashlib
import time

from . import _native
from .storage import AsyncArchive


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


class MemoryArchive(AsyncArchive):
    def __init__(self, *args, gather_workers=8, cpu_direct_record_seal=False, cpu_seal_hash_workers=1,
                 cpu_seal_pack_workers=1, cpu_seal_serial_allocate=False,
                 native_payloads=False, **kwargs):
        if type(gather_workers) is not int or not 1 <= gather_workers <= 16:
            raise ValueError('CPU gather workers must be between 1 and 16')
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
        super().__init__(*args, **kwargs, memory_only=True, native_cpu_payloads=native_payloads)
        self._native_payload_arena = getattr(self.tier, 'payloads', None)
        self._history_worker = None
        self._history_scopes = {}
        self._history_totals = Counter()
        self._history_mode = None
        self._history_policy = None
        self._history_policy_params = {}
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
        self._gather_executor = ThreadPoolExecutor(max_workers=gather_workers,
                                                   thread_name_prefix='kvmem-cpu-gather')
        self._gather_control = RLock()
        self._gather_unusable = None
        self._gather_shutdown_started = False
        self._gather_shutdown_done = Event()
        self._gather_worker = local()
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
                digests = self._record_digests(sources)
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
            digests = self._record_digests(records)
            hashed = time.perf_counter()
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
        parallel = workers > 1 and len(sources)*length >= 8*1024**2
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
        return len(jobs)

    def _gather_region(self, destination, sources, offset, length):
        return self._gather(destination, sources, offset, length)

    def _region_sources(self, keys):
        # Native region gather reads immutable record buffers directly, avoiding
        # a second cache copy or a full-record fallback slab.
        if self._native_payload_arena is not None:
            return self.tier.borrow_records([self._entries[key].allocation for key in keys])
        with self.tier.lock:
            return [self.tier.records[self._entries[key].allocation.slot] for key in keys]

    def _admit_read_sources(self, entries):
        if self._native_payload_arena is not None:
            return self.tier.borrow_records([entry.allocation for entry in entries])
        return None

    def _validate_read_sources(self, entries, sources):
        if self._native_payload_arena is None or len(entries) != len(sources):
            raise ValueError('Invalid pre-admitted CPU read extent')
        for entry, source in zip(entries, sources):
            if (not isinstance(source, _native.HistoricalPayloadRead) or not source or
                    (source.handle.slot, source.handle.generation) !=
                    (entry.allocation.slot, entry.allocation.generation) or
                    not self._native_payload_arena.owns(source)):
                raise ValueError('Pre-admitted CPU read has a foreign allocation')
        return [source.clone() for source in sources]

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
                self._history_totals.update(scope.stats)
                del self._history_scopes[namespace]

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
        result = dict(mode=self._history_mode, policy=self._history_policy,
                    policy_params=dict(self._history_policy_params), active_scopes=scopes, **dict(totals),
                    metadata=self._history_worker.storage_stats)
        if self._history_policy == 'redundancy':
            result['active_features'] = dict(features)
            result['feature_reserved_bytes'] = getattr(self, '_historical_feature_reserved_bytes', 0)
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
        native = self._native_payload_arena.stats if self._native_payload_arena is not None else None
        return {**super().io_usage(), 'archive_backend':'cpu', 'disk_read_bytes':0,
                'disk_write_bytes':0, **hash_usage,
                'cpu_authority_bytes':native['used_bytes'] if native else self.tier.used_slots*self.record_bytes,
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
