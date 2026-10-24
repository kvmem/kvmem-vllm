"""Bounded, speculative FPS construction from accepted immutable CPU records.

The worker never visits live authority records or polls its owner. Queue entries
hold generation-qualified leases and (where needed) native CPU payload readers.
An unsuccessful optimization leaves exact on-demand FPS construction in charge.
"""
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import Condition, Thread
import time

import torch
from . import _native


@dataclass(frozen=True)
class SealedFPSRecord:
    history_begin: int
    count: int
    handle: object


@dataclass
class _Batch:
    owner: object
    epoch: int
    cache_epoch: int
    records: tuple
    leases: list
    sources: object
    queued_at: float

    def release(self):
        if self.sources is not None:
            self.sources.clear()
        for lease in self.leases:
            lease.reset()
        self.leases.clear()


class FPSPrebuildOwner:
    def __init__(self, manager, authority):
        self.manager, self.cache = manager, authority.query_window_cache
        self.archive, self.memory = authority.archive, authority.memory
        self.namespace = authority.memory.request_id
        self.shape = (authority.scoring_layers, *authority.shape[1:])
        self.dtype = authority.raw_dtype
        self.epoch = 0
        self.paused = 0
        self.closed = False
        self.stats = Counter(fps_gpu_groups=0, fps_cpu_groups=0, fps_cpu_reference_groups=0,
            fps_cpu_ambiguity_groups=0, fps_cpu_unsupported_fallback_groups=0,
            fps_cpu_native_groups=0, fps_cpu_pool_busy_groups=0)
        self.last_error = None

    def offer(self, records):
        """Called by the accepted-history publisher, after archive completion."""
        return self.manager.offer(self, records)

    def cancel(self, permanent=False, *, history_end=None):
        if history_end is not None and (permanent or type(history_end) is not int or history_end < 0):
            raise ValueError('Invalid prefix-preserving FPS cancellation')
        with self.manager.condition:
            self.epoch += 1
            self.closed |= permanent
            self.manager._discard(self)
            self.manager.condition.notify_all()
        # The worker can already be inside FPS after its owner-epoch check.
        # Invalidating the cache epoch also fences that late publication.
        if history_end is None:
            self.cache.clear()
        else:
            self.cache.rewind(history_end)

    @contextmanager
    def query(self):
        # Only the querying owner waits, on its background refresh lane. The
        # public model scheduler and other owners never join this condition.
        started = time.perf_counter()
        with self.manager.condition:
            self.paused += 1
            self.manager._discard(self)
            self.manager.condition.wait_for(lambda: self.manager.active is None or
                self.manager.active.owner is not self)
            self.stats['query_wait_seconds'] += time.perf_counter()-started
            self.stats['queries'] += 1
        try:
            yield self
        finally:
            with self.manager.condition:
                self.paused -= 1

    def snapshot(self):
        with self.manager.condition:
            return dict(enabled=True, device=str(self.manager.device), **dict(self.stats),
                cpu_backend=self.manager.cpu_backend, cpu_workers=self.manager.cpu_workers,
                queue_capacity_records=self.manager.maximum,
                last_error=self.last_error, pending_records=sum(len(b.records) for b in
                    self.manager.queue if b.owner is self),
                active_records=(len(self.manager.active.records) if self.manager.active is not None
                    and self.manager.active.owner is self else 0))


class FPSPrebuildManager:
    def __init__(self, *, device='cpu', max_pending_records=None, batch_records=32,
                 cpu_backend='torch', cpu_workers=8):
        # Eight 2048-token prefill chunks can publish 512 sealed blocks at
        # once. Reserve two bounded bursts of metadata/read leases for the
        # fast native worker; queued jobs never own copied Key slabs.
        if max_pending_records is None:
            max_pending_records = 1024 if cpu_backend == 'native' else 256
        if not 1 <= batch_records <= 32 or max_pending_records < batch_records:
            raise ValueError('Invalid bounded FPS prebuild queue')
        self.device = torch.device(device)
        if cpu_backend not in ('torch', 'native') or type(cpu_workers) is not int or not 1 <= cpu_workers <= 32:
            raise ValueError('Invalid FPS CPU worker configuration')
        if cpu_backend == 'native' and self.device.type != 'cpu':
            raise ValueError('Native FPS CPU workers require CPU construction')
        self.cpu_backend, self.cpu_workers = cpu_backend, cpu_workers if cpu_backend == 'native' else 0
        # Shared by background construction and Query misses. The pool owns
        # a bounded set of threads; no model-wide Torch/OpenMP setting changes.
        self.cpu_pool = (_native.KeyFPSCPUPool(cpu_workers, 2) if cpu_backend == 'native' else None)
        self.maximum, self.batch_records = max_pending_records, batch_records
        self.condition = Condition()
        self.queue = deque()
        self.active = None
        self.pending_records = 0
        self.closed = False
        self.worker = Thread(target=self._work, name='kvmem-fps-prebuild', daemon=True)
        self.worker.start()

    def bind(self, authority):
        if authority.query_score_mode != 'key_fps_4' or authority.query_window_cache is None:
            raise ValueError('FPS prebuild requires the representative Key cache')
        return FPSPrebuildOwner(self, authority)

    def _discard(self, owner):
        kept = deque()
        while self.queue:
            batch = self.queue.popleft()
            if batch.owner is owner:
                self.pending_records -= len(batch.records)
                owner.stats['cancelled_records'] += len(batch.records)
                batch.release()
            else:
                kept.append(batch)
        self.queue = kept

    def offer(self, owner, records):
        accepted = 0
        for begin in range(0, len(records), self.batch_records):
            group = records[begin:begin+self.batch_records]
            if any(r.count != 32 or not r.archived for r in group):
                raise ValueError('Only accepted, sealed and published blocks may be prebuilt')
            # No data copy, I/O, tensor allocation or device wait on this path.
            with self.condition:
                if self.closed or owner.closed or owner.paused or self.pending_records+len(group) > self.maximum:
                    owner.stats['deferred_records'] += len(group)
                    continue
                leases = []
                sources = None
                try:
                    leases = [owner.memory.repository.pin(r.handle) for r in group]
                    if not all(leases):
                        raise ValueError('Stale prebuild archive generation')
                    sources = owner.archive.borrow_read_sources(owner.namespace, leases)
                    batch = _Batch(owner, owner.epoch, owner.cache.epoch,
                        tuple(SealedFPSRecord(r.history_begin, r.count,
                            _native.Handle(r.handle.slot, r.handle.generation)) for r in group),
                        leases, sources, time.perf_counter())
                except Exception as error:
                    for lease in leases:
                        lease.reset()
                    if sources is not None:
                        sources.clear()
                    owner.stats['failed_records'] += len(group)
                    owner.stats['errors'] += 1
                    owner.last_error = repr(error)
                    continue
                self.queue.append(batch)
                self.pending_records += len(group)
                owner.stats['queued_records'] += len(group)
                owner.stats['queue_peak_records'] = max(owner.stats['queue_peak_records'], self.pending_records)
                accepted += len(group)
                self.condition.notify()
        return accepted

    def _build(self, batch):
        owner = batch.owner
        amount = owner.dtype.itemsize
        for dimension in owner.shape:
            amount *= dimension
        if len(batch.records)*amount > 64*1024**2:
            raise MemoryError('FPS prebuild Key scratch exceeds 64 MiB')
        slab = torch.empty((len(batch.records), *owner.shape), dtype=owner.dtype,
            pin_memory=self.device.type == 'cuda')
        started = time.perf_counter()
        destination = slab.view(torch.uint8).numpy().reshape(-1)
        kwargs = {'_sources': batch.sources} if batch.sources is not None else {}
        job = owner.archive.read_region(owner.namespace, batch.leases, 0, amount,
            destination=destination, **kwargs)
        if job is None:
            # Optional background work yields on ordinary archive backpressure;
            # the authoritative Query path fills this unchanged exact row later.
            return dict(status='deferred', read_seconds=time.perf_counter()-started,
                skip_reason='archive_queue', key_read_bytes=0)
        job.result()
        read_seconds = time.perf_counter()-started
        with self.condition:
            if owner.closed or owner.epoch != batch.epoch:
                return dict(status='stale', read_seconds=read_seconds)
        result = owner.cache.prebuild(batch.records, slab.permute(1, 0, 2, 3, 4),
            epoch=batch.cache_epoch, optimized=True, device=self.device, cpu_pool=self.cpu_pool)
        return dict(result, read_seconds=read_seconds, key_read_bytes=len(batch.records)*amount)

    def _work(self):
        stream = None
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.closed)
                if not self.queue:
                    return
                batch = self.queue.popleft()
                self.active = batch
            started = time.perf_counter()
            try:
                if self.device.type == 'cuda' and stream is None:
                    stream = torch.cuda.Stream(device=self.device, priority=0)
                with torch.inference_mode(), (torch.cuda.stream(stream) if stream is not None else nullcontext()):
                    result = self._build(batch)
                with self.condition:
                    status = result['status']
                    batch.owner.stats['cpu_threads'] = torch.get_num_threads()
                    batch.owner.stats[status+'_records'] += (result['cache_rebuilt_blocks']
                        if status == 'published' else len(batch.records))
                    for key in ('fps_select_seconds', 'fps_publish_seconds', 'fps_gpu_groups',
                                'fps_cpu_fallback_groups', 'fps_cpu_groups', 'fps_cpu_reference_groups',
                                'fps_cpu_ambiguity_groups', 'fps_cpu_unsupported_fallback_groups',
                                'fps_cpu_native_groups', 'fps_cpu_pool_busy_groups',
                                'cache_hit_blocks'):
                        batch.owner.stats[key] += result.get(key, 0)
                    if result.get('skip_reason'):
                        batch.owner.stats['skip_'+result['skip_reason']+'_records'] += len(batch.records)
                    batch.owner.stats['read_seconds'] += result.get('read_seconds', 0.)
                    batch.owner.stats['key_read_bytes'] += result.get('key_read_bytes', 0)
            except Exception as error:
                with self.condition:
                    batch.owner.stats['failed_records'] += len(batch.records)
                    batch.owner.stats['errors'] += 1
                    batch.owner.last_error = repr(error)
            finally:
                batch.release()
                with self.condition:
                    batch.owner.stats['build_seconds'] += time.perf_counter()-started
                    batch.owner.stats['queue_wait_seconds'] += started-batch.queued_at
                    self.pending_records -= len(batch.records)
                    self.active = None
                    self.condition.notify_all()

    def wait_idle(self, timeout=30):
        with self.condition:
            return self.condition.wait_for(lambda: not self.pending_records, timeout)

    def close(self):
        with self.condition:
            self.closed = True
            while self.queue:
                batch = self.queue.popleft()
                self.pending_records -= len(batch.records)
                batch.owner.stats['cancelled_records'] += len(batch.records)
                batch.release()
            self.condition.notify_all()
        self.worker.join()
        if self.cpu_pool is not None:
            self.cpu_pool.close()
