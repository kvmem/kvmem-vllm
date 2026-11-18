"""Bounded asynchronous NVMe archive over the original coalescing I/O engine.

Records are immutable after publication. Request namespace plus repository
generation is mandatory. No LRU eviction of authoritative data is permitted;
repository GC releases records only after all consumers have dropped leases.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Event, Lock
import hashlib
import time

from .. import _native
from . import settings as _settings


def read_destination(destination, size):
    """Retain the caller's exclusive writable byte buffer until I/O completes."""
    if destination is None:
        return None
    try:
        view = memoryview(destination)
    except TypeError as error:
        raise ValueError("Read destination must export writable contiguous bytes") from error
    if view.readonly or view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous or view.nbytes != size:
        raise ValueError("Read destination must have the exact writable contiguous byte extent")
    return view.cast('B')


@dataclass
class _Entry:
    allocation: object
    ready: bool = False


class ArchiveJob:
    """Once admitted, I/O drains even if its request is cancelled."""
    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        return self._future.result(timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        return False


def _submit_archive_job(executor, work):
    """Do not touch I/O sources until submission returns an owned completion.

    ThreadPoolExecutor.submit can enqueue work and then fail while starting a
    thread, leaving no Future for the caller to drain. The gate makes such a
    hidden task harmless: its caller performs the original admission rollback,
    and the task only exits. In particular, never join the pool while holding
    the archive lock needed by other accepted jobs' completion paths.
    """
    ready = Event()
    callback = [work]
    accepted = False

    def run():
        ready.wait()
        if accepted:
            return callback.pop()()

    try:
        future = executor.submit(run)
        job = ArchiveJob(future)
    except BaseException:
        # A queued/running wrapper cannot retain archive leases or the pinned
        # destination through its closure after the admission is rolled back.
        callback.clear()
        raise
    else:
        accepted = True
        return job
    finally:
        work = None
        ready.set()


def prefix_pool_bytes():
    """KVMEM_PREFIX_POOL_BYTES: CPU budget of the shared-prefix pool (docs/prefix-sharing.md,
    phase 2). 0 (the default) disables sharing payloads on the native arena."""
    value = _settings.get('KVMEM_PREFIX_POOL_BYTES', '0')
    if not value.isdigit():
        raise ValueError('KVMEM_PREFIX_POOL_BYTES must be a non-negative integer')
    return int(value)


def make_shared_archive(options, directory, record_bytes, max_requests, *, max_query_tokens=0, layer_groups=None):
    """One process-wide arena; each admitted authority has a lifetime partition."""
    from .query_window_budget import split_cpu_history_budget
    from .historical_feature_budget import split_historical_feature_budget
    history_bytes, query_cache_bytes = split_cpu_history_budget(options)
    history_bytes, feature_bytes = split_historical_feature_budget(
        options, history_bytes, max_requests + options.session_capacity)
    capacity = options.memory.repository_blocks * (max_requests + options.session_capacity)
    if options.archive_backend == 'cpu':
        # Immutable authority, not an evictable read cache. No file or mmap is
        # opened. Callers must admit full histories within this byte budget.
        from .memory_storage import MemoryArchive
        charge = record_bytes
        if getattr(options, 'historical_gc', None) is not None:
            from .historical_gc import historical_record_charge
            charge = historical_record_charge(record_bytes, options.memory.block_tokens)
        descriptors = capacity
        capacity = min(capacity, history_bytes // charge)
        if capacity <= 0:
            raise ValueError('CPU history budget cannot fit one record after Query cache reservation')
        sharing = {}
        if getattr(options, 'historical_gc', None) is not None and prefix_pool_bytes() and descriptors > capacity:
            # Shared prefixes: aliases take descriptors, not payload bytes. The
            # byte budget bounds payload slots; descriptors keep the logical bound.
            sharing = dict(payload_capacity=capacity)
            capacity = descriptors
        archive = MemoryArchive(str(directory), record_bytes, capacity, **sharing,
                               io_max_bytes=options.archive_io_max_bytes,
                               gather_workers=options.cpu_gather_workers,
                               cpu_direct_record_seal=getattr(options, 'cpu_direct_record_seal', False),
                               cpu_seal_hash_workers=getattr(options, 'cpu_seal_hash_workers', 1),
                               cpu_seal_pack_workers=getattr(options, 'cpu_seal_pack_workers', 1),
                               cpu_seal_serial_allocate=getattr(options, 'cpu_seal_serial_allocate', False),
                               native_payloads=getattr(options, 'historical_gc', None) is not None)
        archive.async_seal = options.async_seal
        archive.history_record_charge = charge
        archive._query_window_cache_reserved_bytes = query_cache_bytes
        archive._historical_feature_reserved_bytes = feature_bytes
        archive._query_window_cache_bytes = 0
        if getattr(options, 'historical_gc', None) is not None:
            archive.enable_history_gc(options.historical_gc.native(options.memory, max_requests+options.session_capacity))
        return archive
    if options.archive_io_max_bytes:
        block = options.memory.block_tokens
        # Refuse an impossible serving configuration before opening the arena.
        # Region fallback holds its output, a complete-record scratch slab and
        # immutable cache copies. K/V serving regions are disjoint and cover at
        # most one record. Even an all-CPU tiered read takes this fallback path.
        read_records = (options.memory.select_budget + options.memory.gen_budget) // block
        if options.query_score_mode != 'mean_k':
            read_records = max(read_records, 32)  # Bounded scoring-Key batches.
        read_bytes = 3 * read_records * record_bytes
        capture_bytes = max(16, (max_query_tokens + block - 1) // block + 1) * record_bytes
        if options.async_seal:
            capture_bytes *= 2  # Owned source tensors plus immutable snapshot.
        largest_batch = max(read_bytes, capture_bytes)
        if options.archive_io_max_bytes < largest_batch:
            raise ValueError('Archive I/O byte capacity cannot fit one full serving batch')
    cls, kwargs = AsyncArchive, {}
    if options.archive_backend == 'tiered':
        from .archive_limits import tiered_capacity_records
        from .tiered_storage import TieredArchive
        capacity = tiered_capacity_records(options, record_bytes, max_requests)
        archive = TieredArchive(str(directory), record_bytes, capacity,
            cpu_bytes=history_bytes, ssd_max_bytes=options.nvme_max_bytes,
            segment_bytes=options.tiered_segment_bytes,
            cpu_low_watermark=options.tiered_cpu_low_watermark,
            cpu_high_watermark=options.tiered_cpu_high_watermark,
            io_max_bytes=options.archive_io_max_bytes)
        archive.async_seal = options.async_seal
        archive._query_window_cache_reserved_bytes = query_cache_bytes
        archive._historical_feature_reserved_bytes = feature_bytes
        archive._query_window_cache_bytes = 0
        return archive
    if options.cpu_read_cache_bytes:
        from .cached_storage import CachedArchive
        cls = CachedArchive
        kwargs.update(cache_bytes=history_bytes,
                      gather_workers=options.cpu_gather_workers,
                      namespace_bytes=(min(options.cpu_history_cache_bytes_per_request,
                                           options.memory.repository_blocks * record_bytes)
                          if options.cpu_history_cache_bytes_per_request else
                          (options.memory.select_budget + options.memory.gen_budget)
                          // options.memory.block_tokens * record_bytes))
    if getattr(options, 'archive_layout', 'record') == 'layer':
        if layer_groups is None:
            raise ValueError('archive_layout=layer needs the model\'s layer groups')
        kwargs.update(layer_groups=layer_groups, layer_threads=options.archive_direct_threads)
    archive = cls(str(directory), record_bytes, capacity, preallocate=options.nvme_preallocate,
               max_bytes=options.nvme_max_bytes, io_max_bytes=options.archive_io_max_bytes, **kwargs)
    if getattr(options, 'archive_direct_reads', False):
        archive.enable_direct_reads(options.archive_direct_threads)
    archive.async_seal = options.async_seal
    archive._query_window_cache_reserved_bytes = query_cache_bytes
    archive._query_window_cache_bytes = 0
    return archive


class DirectRead:
    """One O_DIRECT batch in flight; wait() returns its stats and releases its leases."""
    def __init__(self, archive, ticket, owned):
        self.archive, self.ticket, self._owned = archive, ticket, owned
        self.stats, self._finished = None, False

    def wait(self):
        if not self._finished:
            self._finished = True
            try:
                self.stats = self.archive._direct.wait(self.ticket)
            finally:
                self._owned.clear()
                self.archive._direct_done()
        if self.stats is None:
            raise RuntimeError('Direct archive read failed')
        return self.stats


class AsyncArchive:
    def __init__(self, directory: str, record_bytes: int, capacity: int,
                 workers: int = 2, max_pending: int = 4, *, preallocate: bool = False,
                 max_bytes: int = 0, io_max_bytes: int = 0, memory_only: bool = False,
                 native_cpu_payloads: bool = False, payload_capacity: int | None = None,
                 layer_groups=None, layer_threads: int = 8):
        if min(record_bytes, capacity, workers, max_pending) <= 0:
            raise ValueError("Archive capacities must be positive")
        self.record_bytes = record_bytes
        if type(preallocate) is not bool or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Invalid NVMe reservation/quota option')
        self.capacity_bytes = record_bytes * capacity
        if max_bytes and self.capacity_bytes > max_bytes:
            raise ValueError('Shared NVMe arena exceeds configured byte quota')
        if type(io_max_bytes) is not int or io_max_bytes < 0 or 0 < io_max_bytes < record_bytes:
            raise ValueError('Archive I/O byte capacity must fit at least one record')
        self.io_max_bytes = io_max_bytes
        self._io_bytes = self._io_peak_bytes = self._io_deferrals = 0
        self.preallocated = preallocate
        self.memory_only = memory_only
        self.span = _native.IoSpan
        if type(native_cpu_payloads) is not bool or native_cpu_payloads and not memory_only:
            raise ValueError('Native CPU payloads require the CPU archive backend')
        if payload_capacity is not None and not native_cpu_payloads:
            raise ValueError('A separate CPU payload capacity requires native CPU payloads')
        if memory_only:
            from .memory_storage import MemoryTier, MemorySpan
            if native_cpu_payloads:
                from .native_memory_storage import NativeMemoryTier
                self.tier = NativeMemoryTier(record_bytes, capacity, payload_capacity)
            else:
                self.tier = MemoryTier(record_bytes, capacity)
            self.span = MemorySpan
        elif layer_groups is not None:
            # SSD tier, archive_layout=layer: per-layer groups, unpadded (core/layer_major.py).
            if preallocate:
                raise ValueError('The layer-major archive does not preallocate')
            from .layer_major import LayerMajorTier
            self.tier = LayerMajorTier(directory, capacity, record_bytes, layer_groups, threads=layer_threads)
            self.span = LayerMajorTier.span
        else:
            self.tier = (_native.NvmeTier(directory, self.capacity_bytes, record_bytes, True)
                         if preallocate else _native.NvmeTier(directory, self.capacity_bytes, record_bytes))
        from .portable import cap_workers
        self.executor = ThreadPoolExecutor(max_workers=cap_workers(workers), thread_name_prefix="kvmem-io")
        self.max_pending = max_pending
        self._pending = 0
        self._lock = Lock()
        self._capacity_changed = Condition(self._lock)
        self._entries = {}
        self._free_ids = list(reversed(range(capacity)))
        self._namespace_counts = {}
        self._closed = False
        self._seal_totals = None
        self._direct = None          # SSD tier: O_DIRECT reader (enable_direct_reads)
        self._direct_pending = 0
        self.layer_major = layer_groups is not None and not memory_only

    @staticmethod
    def key(namespace: str, lease):
        if not namespace or not lease:
            raise ValueError("Archive access requires a namespace and a live lease")
        h = lease.handle
        return namespace, h.slot, h.generation

    def _check_extent(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError('Archive I/O extent must be positive')
        if self.io_max_bytes and size > self.io_max_bytes:
            # Retrying an impossible batch would otherwise deadlock serving.
            raise ValueError('Archive I/O batch exceeds configured byte capacity')

    def _has_capacity(self, size):
        return (self._pending < self.max_pending and
                (not self.io_max_bytes or self._io_bytes + size <= self.io_max_bytes))

    def _admit(self, size):
        if self._closed:
            raise RuntimeError("Archive is closed")
        self._check_extent(size)
        if not self._has_capacity(size):
            self._io_deferrals += 1
            return False
        self._pending += 1
        self._io_bytes += size
        self._io_peak_bytes = max(self._io_peak_bytes, self._io_bytes)
        return True

    def _complete(self, size):
        """Caller holds the archive lock; I/O and its generation leases drained."""
        self._pending -= 1
        self._io_bytes -= size
        self._capacity_changed.notify_all()

    def io_usage(self):
        with self._lock:
            return dict(archive_io_max_bytes=self.io_max_bytes,
                        archive_io_pending_bytes=self._io_bytes,
                        archive_io_peak_bytes=self._io_peak_bytes,
                        archive_io_deferrals=self._io_deferrals,
                        query_window_cache_reserved_bytes=getattr(self, '_query_window_cache_reserved_bytes', 0),
                        historical_feature_reserved_bytes=getattr(self, '_historical_feature_reserved_bytes', 0),
                        query_window_cache_bytes=getattr(self, '_query_window_cache_bytes', 0),
                        **(self._seal_totals or {}))

    def wait_for_capacity(self, timeout, *, size=None):
        """Host retirement may wait for I/O admission; model threads need not."""
        size = self.record_bytes if size is None else size
        self._check_extent(size)
        with self._capacity_changed:
            ready = self._capacity_changed.wait_for(
                lambda: self._closed or self._has_capacity(size), timeout)
            if self._closed:
                raise RuntimeError('Archive closed while waiting for retirement I/O')
            return ready

    def wait_for_idle(self, timeout):
        """Drain shared work, including jobs owned by detached retirement."""
        with self._capacity_changed:
            ready = self._capacity_changed.wait_for(lambda: not self._pending or self._closed, timeout)
            if self._closed:
                raise RuntimeError('Archive closed while waiting for shared I/O')
            return ready

    def write(self, namespace: str, leases: list, payload: bytes) -> ArchiveJob | None:
        return self._write(namespace, leases, payload=payload)

    def write_from_views(self, namespace: str, leases: list, views) -> ArchiveJob | None:
        """Admit sealed, exclusively owned tensors before packing/hashing.

        A successful call retains buffer exports and cloned generation leases.
        The caller must never mutate sealed sources. Both source and snapshot
        bytes count against I/O admission until the background write completes.
        """
        views = tuple(memoryview(v).cast('B') for v in views)
        if (len(views) != 2 * len(leases) or not leases or
                any(not v.c_contiguous or v.ndim != 1 for v in views) or
                any(views[2*i].nbytes + views[2*i+1].nbytes != self.record_bytes
                    for i in range(len(leases)))):
            raise ValueError('Sealed views must cover each complete raw-K/V record')
        return self._write(namespace, leases, views=views)

    def _write(self, namespace, leases, *, payload=None, views=None):
        deferred = views is not None
        if deferred:
            size = len(leases) * self.record_bytes
        else:
            size = len(payload)
        if size != len(leases) * self.record_bytes or not leases:
            raise ValueError("Write requires complete, fixed-size records")
        keys = [self.key(namespace, lease) for lease in leases]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate archive record in one batch")
        admitted_bytes = size * (2 if deferred else 1)
        with self._lock:
            if any(key in self._entries for key in keys):
                raise ValueError("Cannot overwrite immutable archive records")
            self._check_extent(admitted_bytes)
            if (len(self._free_ids) < len(keys) or not self._payload_room(len(keys)) or
                    not self._admit(admitted_bytes)):
                return None
            entries, owned = [], []
            try:
                # Snapshot only after byte admission. Refused writers do not
                # allocate another full payload while the queue is saturated.
                if not deferred:
                    payload = bytes(payload)
                for key in keys:
                    block_id = self._free_ids.pop()
                    try:
                        allocation = self.tier.allocate(block_id)
                        if allocation.slot < 0:
                            raise RuntimeError("Archive allocator capacity invariant violated")
                    except BaseException:
                        self._free_ids.append(block_id)
                        raise
                    entry = _Entry(allocation)
                    self._entries[key] = entry
                    self._namespace_counts[namespace] = self._namespace_counts.get(namespace, 0) + 1
                    entries.append(entry)
                owned = [lease.clone() for lease in leases]
                backing_context = self._backing_context(namespace)
                if not deferred:
                    self._cache_write(keys, payload)
            except BaseException:
                owned.clear()
                for key in keys:
                    self._release(key)
                self._complete(admitted_bytes)
                raise
        # Direct CPU seals publish owned allocations without using I/O spans.
        # Let the generic seal path construct them only when it needs them.
        spans = (None if deferred else
                 [self.span(e.allocation.slot, i * self.record_bytes, self.record_bytes)
                  for i, e in enumerate(entries)])

        def work():
            nonlocal views, backing_context
            try:
                if deferred:
                    stats = self._seal_views(keys, entries, spans, views)
                else:
                    stats = self.tier.io(spans, payload, True)
                if backing_context is not None:
                    self._publish_backing(backing_context, owned, entries)
                with self._lock:
                    if deferred and self._seal_totals is not None:
                        increments = {**stats, 'seal_batches': 1,
                                      'seal_records': len(entries), 'seal_payload_bytes': size}
                        self._seal_totals = {name: value + increments.get(name, 0)
                                            for name, value in self._seal_totals.items()}
                    for entry in entries:
                        entry.ready = True
                return stats
            except BaseException:
                with self._lock:
                    for key in keys:
                        self._release(key)
                raise
            finally:
                views = None
                owned.clear()
                backing_context = None
                with self._lock:
                    self._complete(admitted_bytes)
        # Executor submission and close are serialized; close cannot shut down
        # the pool while an admitted task still needs to be submitted.
        with self._lock:
            try:
                return _submit_archive_job(self.executor, work)
            except BaseException:
                owned.clear()
                for key in keys:
                    self._release(key)
                self._complete(admitted_bytes)
                raise

    def _payload_room(self, count):
        """Whether the payload store can take ``count`` more records (shared prefixes)."""
        return True

    def _backing_context(self, namespace):
        return None

    def _publish_backing(self, context, leases, entries):
        raise NotImplementedError('Archive backing publication is not implemented')

    def _seal_views(self, keys, entries, spans, views):
        """The original whole-batch snapshot path, including cached/SSD writes."""
        snapshot = byte_view = None
        records = []
        try:
            if spans is None:
                spans = [self.span(e.allocation.slot, i * self.record_bytes, self.record_bytes)
                         for i, e in enumerate(entries)]
            started = time.perf_counter()
            snapshot = _native.pack_record_views(views)
            packed = time.perf_counter()
            byte_view = memoryview(snapshot)
            records = [byte_view[i*self.record_bytes:(i+1)*self.record_bytes] for i in range(len(keys))]
            digests = self._record_digests(records)
            hashed = time.perf_counter()
            with self._lock:
                self._cache_write(keys, snapshot)
            cached = time.perf_counter()
            stats = self.tier.io(spans, snapshot, True)
            stats.update(record_digests=digests, seal_pack_seconds=packed-started,
                         seal_hash_seconds=hashed-packed, seal_cache_seconds=cached-hashed)
            return stats
        finally:
            # A retained failed Future must not retain its source buffer exports.
            views = snapshot = byte_view = None
            records.clear()

    def _record_digests(self, records):
        """Hash complete immutable records in their publication order."""
        record = None
        try:
            digests = []
            for record in records:
                digests.append(hashlib.sha256(record).hexdigest())
            return tuple(digests)
        finally:
            # Failed Futures may retain this frame after admission rolls back.
            records = record = None

    def _cache_write(self, keys, payload):
        """Optional cache admission under lock, before asynchronous publication."""

    def read(self, namespace: str, leases: list, *, destination=None, _sources=None) -> ArchiveJob | None:
        if not leases:
            raise ValueError("Empty read")
        destination = read_destination(destination, len(leases) * self.record_bytes)
        size = len(leases) * self.record_bytes
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(e is None or not e.ready for e in entries):
                raise KeyError("Archive record missing, stale, or not yet published")
            if not self._admit(size):
                return None
            sources = None
            owned = []
            try:
                owned = [lease.clone() for lease in leases]
                sources = (self._admit_read_sources(entries) if _sources is None else
                           self._validate_read_sources(entries, _sources))
            except BaseException:
                if sources is not None:
                    sources.clear()
                owned.clear()
                self._complete(size)
                raise
        spans = [self.span(e.allocation.slot, i * self.record_bytes, self.record_bytes)
                 for i, e in enumerate(entries)]

        def work():
            try:
                payload = bytearray(len(entries) * self.record_bytes) if destination is None else destination
                stats = (self.tier.io(spans, payload, False) if sources is None else
                         self.tier.read_from_sources(spans, payload, sources))
                return payload, stats
            finally:
                if sources is not None:
                    sources.clear()
                owned.clear()
                with self._lock:
                    self._complete(size)
        with self._lock:
            try:
                return _submit_archive_job(self.executor, work)
            except BaseException:
                if sources is not None:
                    sources.clear()
                owned.clear()
                self._complete(size)
                raise

    def enable_direct_reads(self, threads=8):
        """SSD tier: O_DIRECT reads of whole records into caller-owned pinned memory."""
        if self.memory_only:
            raise ValueError('Direct reads need the NVMe archive')
        with self._lock:
            if self._closed:
                raise RuntimeError('Archive is closed')
            if self._direct is None:
                self._direct = (self.tier.arena if self.layer_major else
                                _native.DirectRecordReader(self.tier, threads=threads))
            return self._direct

    @property
    def layer_group_bytes(self):
        """Packed bytes of each layer group (archive_layout=layer), else None."""
        return self.tier.group_bytes if self.layer_major else None

    def read_groups(self, namespace, leases, groups, offsets, destination):
        """Start O_DIRECT reads of packed layer groups (archive_layout=layer).

        For the k-th group, record i lands at destination[offsets[k] + i * group_bytes:];
        a None lease is skipped (the caller fills it). Leases are held until wait().
        """
        if not self.layer_major:
            raise ValueError('Layer group reads need archive_layout=layer')
        keys = [None if lease is None else self.key(namespace, lease) for lease in leases]
        with self._lock:
            if self._direct is None or self._closed:
                raise RuntimeError('Direct reads are not enabled on this archive')
            entries = [None if key is None else self._entries.get(key) for key in keys]
            if any(key is not None and (e is None or not e.ready) for key, e in zip(keys, entries)):
                raise KeyError('Archive record missing, stale, or not yet published')
            owned = [lease.clone() for lease in leases if lease is not None]
            try:
                ticket = self._direct.submit_groups([-1 if e is None else e.allocation.slot for e in entries],
                                                    list(groups), list(offsets),
                                                    destination.data_ptr(), destination.numel())
            except BaseException:
                owned.clear()
                raise
            self._direct_pending += 1
        return DirectRead(self, ticket, owned)

    @property
    def direct_reads(self):
        return self._direct is not None

    def read_direct(self, namespace, leases, destination, hole=(0, 0)):
        """Start reading records into a 4 KiB aligned contiguous uint8 CPU tensor (pinned).

        leases[i] lands at destination[i * record_bytes:]; a None lease leaves
        that record to the caller. hole = (offset, bytes) inside every record is
        not read. Generation leases are held until wait().
        """
        if len(leases) * self.record_bytes > destination.numel():
            raise ValueError('Direct read destination is smaller than the batch')
        keys = [None if lease is None else self.key(namespace, lease) for lease in leases]
        with self._lock:
            if self._direct is None or self._closed:
                raise RuntimeError('Direct reads are not enabled on this archive')
            if self.layer_major:
                raise ValueError('Whole-record direct reads need archive_layout=record')
            entries = [None if key is None else self._entries.get(key) for key in keys]
            if any(key is not None and (e is None or not e.ready) for key, e in zip(keys, entries)):
                raise KeyError('Archive record missing, stale, or not yet published')
            owned = [lease.clone() for lease in leases if lease is not None]
            try:
                ticket = self._direct.submit([-1 if e is None else e.allocation.slot for e in entries],
                                             destination.data_ptr(), destination.numel(), *hole)
            except BaseException:
                owned.clear()
                raise
            self._direct_pending += 1
        return DirectRead(self, ticket, owned)

    def _direct_done(self):
        with self._lock:
            self._direct_pending -= 1
            self._capacity_changed.notify_all()

    def borrow_read_sources(self, namespace, leases):
        """Retain immutable payload read rights for a bounded deferred reader.

        Metadata leases alone do not fence native CPU payload retirement. This
        admission is short and copies no data; the caller releases every source
        after cancellation or after its independently admitted I/O completes.
        """
        if not leases or len(leases) > 32:
            raise ValueError('Deferred archive readers require one to 32 leases')
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError('Deferred archive record missing, stale or unpublished')
            return self._admit_read_sources(entries)

    def _admit_read_sources(self, entries):
        return None

    def _validate_read_sources(self, entries, sources):
        raise ValueError('Pre-admitted CPU reads require native payload ownership')

    def _release(self, key) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            return True
        if not self.tier.release(entry.allocation):
            return False
        self._free_ids.append(entry.allocation.block_id)
        del self._entries[key]
        remaining = self._namespace_counts[key[0]] - 1
        if remaining:
            self._namespace_counts[key[0]] = remaining
        else:
            del self._namespace_counts[key[0]]
        return True

    def namespace_allocated_records(self, namespace):
        """Includes retired/publishing descriptors until actual safe release."""
        with self._lock:
            return self._namespace_counts.get(namespace, 0)

    def _region_sources(self, keys):
        return [None] * len(keys)

    def _gather_region(self, destination, sources, offset, length):
        _native.gather_record_region(destination, sources, self.record_bytes,
                                     offset, length, 0, len(sources))
        return 1

    def read_region(self, namespace, leases, offset, length, *, destination=None, _sources=None, inline=False):
        """Read only raw K or V from immutable CPU records.

        The unchanged NVMe tier reads whole records on a cache miss. Its bounded
        fallback slab is charged separately; only the requested field is copied
        to the caller. Namespace, generation, readiness and leases stay mandatory.
        """
        return self._read_regions(namespace, leases, ((offset, length),),
                                  destination=destination, _sources=_sources, inline=inline)

    def read_regions(self, namespace, leases, regions, *, destination=None, _sources=None, inline=False,
                     paired_native=False):
        """One admitted read of paired fields, packed field-major then record-major.

        All generation checks, immutable source leases and I/O admission cover
        both fields. A cache miss reads its complete record only once. The
        existing gather backend and its failure-draining rules are unchanged.
        """
        regions = tuple(regions)
        if len(regions) != 2:
            raise ValueError('Paired archive read requires exactly two regions')
        return self._read_regions(namespace, leases, regions,
                                  destination=destination, _sources=_sources, inline=inline,
                                  paired_native=paired_native)

    def _read_regions(self, namespace, leases, regions, *, destination=None, _sources=None, inline=False,
                      paired_native=False):
        # Only a synchronous CPU-archive consumer may remove the executor
        # round trip. Admission, generation leases, gather and failure draining
        # are identical; asynchronous/pre-read and disk callers keep the pool.
        if type(inline) is not bool or inline and (not self.memory_only or destination is None):
            raise ValueError('Inline region reads require CPU authority and an owned destination')
        completed = ArchiveJob(Future()) if inline else None
        regions = tuple(tuple(region) for region in regions)
        if type(paired_native) is not bool or paired_native and (
                not self.memory_only or destination is None or len(regions) != 2):
            raise ValueError('Native paired gather requires CPU authority, paired regions and owned staging')
        if paired_native and not callable(getattr(_native, 'gather_record_regions', None)):
            raise RuntimeError('Native paired gather requires the updated native extension')
        if (not leases or not regions or any(len(region) != 2 for region in regions) or
                any(type(offset) is not int or type(length) is not int or
                    offset < 0 or length <= 0 or offset + length > self.record_bytes
                    for offset, length in regions)):
            raise ValueError('Invalid archive record region')
        row_bytes = sum(length for _, length in regions)
        size = len(leases) * row_bytes
        destination = read_destination(destination, size)
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError('Archive record missing, stale, or not yet published')
            sources = (self._region_sources(keys) if _sources is None else
                       self._validate_read_sources(entries, _sources))
            misses = [i for i, source in enumerate(sources) if source is None]
            # A fallback record slab and immutable cache copies can coexist.
            admitted_bytes = size + 2 * len(misses) * self.record_bytes
            if not self._admit(admitted_bytes):
                return None
            try:
                owned = [lease.clone() for lease in leases]
            except BaseException:
                self._complete(admitted_bytes)
                raise
        spans = [self.span(entries[i].allocation.slot, j*self.record_bytes, self.record_bytes)
                 for j, i in enumerate(misses)]
        submitted_at = time.perf_counter()

        def work():
            try:
                started = time.perf_counter()
                data = bytearray(size) if destination is None else destination
                fallback = bytearray(len(misses)*self.record_bytes)
                stats = self.tier.io(spans, fallback, False)
                read_at = time.perf_counter()
                view = memoryview(fallback)
                for j, i in enumerate(misses):
                    sources[i] = view[j*self.record_bytes:(j+1)*self.record_bytes]
                gather_stats = None
                if paired_native:
                    gather_stats = self._gather_regions_paired(memoryview(data), sources, regions)
                    tasks = 1
                else:
                    tasks, cursor = 0, 0
                    for offset, length in regions:
                        end = cursor + len(sources) * length
                        tasks += self._gather_region(memoryview(data)[cursor:end], sources, offset, length)
                        cursor = end
                copied_at = time.perf_counter()
                if misses:
                    with self._lock:
                        for i in misses:
                            self._cache_write([keys[i]], bytes(sources[i]))
                stats.update(logical_bytes=size, cache_hit_bytes=(len(keys)-len(misses))*row_bytes,
                    cache_gather_tasks=tasks,
                    read_queue_wait_seconds=started-submitted_at,
                    native_read_call_seconds=read_at-started, cache_copy_seconds=copied_at-read_at,
                    cache_fill_seconds=time.perf_counter()-copied_at)
                if len(regions) == 1:
                    stats.update(region_offset=regions[0][0], region_bytes=regions[0][1])
                else:
                    stats.update(regions=regions, region_bytes=row_bytes, region_read_count=len(regions))
                if inline:
                    stats['inline_cpu_region_read'] = True
                if gather_stats is not None:
                    stats['paired_native_gather'] = True
                    stats.update({'paired_native_'+key: value for key,value in gather_stats.items()})
                return data, stats
            finally:
                sources.clear()
                owned.clear()
                with self._lock:
                    self._complete(admitted_bytes)

        if inline:
            # Do not hold the archive lock while copying or joining gather
            # tasks. The caller already waits for this read before its DMA;
            # the native gather releases the GIL and retains the same sources.
            try:
                result = work()
            except BaseException as error:
                completed._future.set_exception(error)
            else:
                completed._future.set_result(result)
            return completed
        with self._lock:
            try:
                return _submit_archive_job(self.executor, work)
            except BaseException:
                sources.clear()
                owned.clear()
                self._complete(admitted_bytes)
                raise

    def reclaim(self, namespace: str, record) -> bool:
        """Only call from Repository.reclaim's release callback (leases == 0)."""
        key = namespace, record.handle.slot, record.handle.generation
        if not self._lock.acquire(blocking=False):
            return False
        try:
            return self._release(key)
        finally:
            self._lock.release()

    def close(self) -> bool:
        with self._lock:
            if self._pending or self._direct_pending or getattr(self, '_prepared_region_readers', 0):
                return False
            if self._closed:
                return True
            self._closed = True
            if self._direct is not None:
                self._direct.close()
                self._direct = None
            self._capacity_changed.notify_all()
            self.executor.shutdown(wait=False)
            self._entries.clear()
            self._namespace_counts.clear()
            self._free_ids.clear()
            if self.memory_only:
                self._closed_cpu_peak_bytes = self.tier.cpu_peak_bytes
            self.tier = None
            return True
