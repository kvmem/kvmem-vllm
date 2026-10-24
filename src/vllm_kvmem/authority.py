"""Accepted-token raw-K / FP8-V authority with immutable sealed records.

CPU record buffers and admitted archive writes have a shared per-request bound.
The caller retries after poll() on backpressure. An incomplete tail is never
published to the immutable archive; repeated decode appends extend it in memory.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import hashlib
import time

import torch

from . import _native
from .memory import RequestMemory


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
    key_sum: torch.Tensor
    archived: bool = False
    submitted: bool = False
    digest: str | None = None
    scatter_views: tuple | None = field(default=None, repr=False)
    mean_key: torch.Tensor | None = field(default=None, repr=False)
    staging_slot: int | None = field(default=None, repr=False)
    index_owner: object | None = field(default=None, repr=False)

    def __del__(self):
        # Keep the native buffer owner until every Python tensor field
        # has dropped its reference. Its destruction only queues native work.
        owner = getattr(self, 'index_owner', None)
        self.key_sum = self.mean_key = self.scatter_views = None
        self.index_owner = None
        del owner


def record_key(handle):
    return handle.slot, handle.generation


SCORE_DOMAIN_VERSION = 'qw3-native-v2'


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
                 query_score_mode='mean_k', query_window_cache_bytes=2147483648):
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
                max_bytes=query_window_cache_bytes)
        if type(cpu_mean_index_simd) is not bool:
            raise ValueError('cpu_mean_index_simd must be boolean')
        if cpu_mean_index_simd and mean_k_backend != 'qw3_cuda':
            raise ValueError('cpu_mean_index_simd requires mean_k_backend=qw3_cuda')
        self.cpu_mean_index_simd = cpu_mean_index_simd
        self.mean_index_stats = MeanIndexStats() if mean_index_stats is None else mean_index_stats
        self.raw_element_bytes = raw_dtype.itemsize
        self.record_elements = layers * config.block_tokens * heads * head_dim
        self.raw_record_bytes = self.record_elements * self.raw_element_bytes
        self.value_record_bytes = self.record_elements
        self.record_bytes = self.raw_record_bytes + self.value_record_bytes
        if archive.record_bytes != self.record_bytes:
            raise ValueError("Archive record format does not match model geometry")
        self.resident_records = resident_records
        self.records: dict[tuple[int, int], Record] = {}
        # These contain only live, not-yet-published generations, so ordinary
        # decode bookkeeping does not grow with archived history. GC can only
        # remove archived records; rewind explicitly removes its live suffix.
        self._unsubmitted: dict[tuple[int, int], Record] = {}
        self._resident_count = 0
        # Reuse only temporary raw/V staging. Immutable archive bytes and each
        # record's MeanK state remain independently owned. Repeatedly freeing
        # large raw/V tensors between persistent small index allocations can
        # leave many GiB of resident-but-free allocator holes during prefill.
        self._record_buffers = []
        self._free_record_buffers = []
        self._busy_record_buffers = set()
        self.writes = []
        self.closing = False
        self.discarded_records: dict[tuple[int, int], Record] = {}
        self._rewind_position = None
        self._rewind_committed = False
        self.gc = None
        if getattr(archive, '_history_worker', None) is not None:
            if gc_config is not None:
                raise ValueError('Historical KV management and legacy repository GC cannot share an authority')
            from .historical_gc import HistoricalAuthorityGc
            self.gc = HistoricalAuthorityGc(self.memory, archive)
        elif gc_config is not None:
            from .repository_gc import AuthorityGc
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
        self.gc.drain(self.records, self.discarded_records)
        for key, record in staged:
            if key not in self.discarded_records:
                self._return_record_buffers(record)

    def poll(self):
        if self.gc is not None:
            self._drain_gc()
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
                    self._return_record_buffers(record)
                    newly_archived.append(record)
                record.archived = True
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
        if self._free_record_buffers:
            slot = self._free_record_buffers.pop()
        else:
            if len(self._record_buffers) >= self.resident_records:
                raise RuntimeError('Record staging exceeded its resident credit')
            buffers = (torch.empty(self.shape, dtype=self.raw_dtype),
                       torch.empty(self.shape, dtype=torch.uint8))
            slot = len(self._record_buffers)
            self._record_buffers.append(buffers)
        if slot in self._busy_record_buffers:
            raise RuntimeError('Record staging slot is still owned by publication')
        self._busy_record_buffers.add(slot)
        return (*self._record_buffers[slot], slot)

    def _return_record_buffers(self, record):
        slot = record.staging_slot
        if slot is None:
            return
        if slot not in self._busy_record_buffers:
            raise RuntimeError('Record staging ownership was released twice')
        self._busy_record_buffers.remove(slot)
        self._free_record_buffers.append(slot)
        record.staging_slot = None

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
        leases = [self.memory.repository.pin(r.handle) for r in records]
        try:
            if not all(leases):
                raise RuntimeError("Authority record lost its repository generation")
            # Join tensor buffer views directly into the immutable I/O snapshot.
            # Avoid raw/V tobytes, per-record concatenations and their copies;
            # row layout and per-record checksums remain byte-for-byte identical.
            views = []
            for record in records:
                views.extend((memoryview(record.raw.view(torch.uint8).numpy()).cast('B'),
                              memoryview(record.values.numpy()).cast('B')))
            if getattr(self.archive, 'async_seal', False):
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
            record.scatter_views = None
            record.raw = record.values = None
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
                        mean=self.mean_k_backend == 'qw3_cuda')
                else:
                    sums = torch.zeros(index_shape, dtype=torch.float32)
                record_raw, record_values, slot = self._borrow_record_buffers()
                try:
                    record = Record(handle, span.history_begin, 0,
                        record_raw, record_values, sums, staging_slot=slot)
                    record.index_owner = index_owner
                    self.records[key] = record
                except BaseException:
                    self._busy_record_buffers.remove(slot)
                    self._free_record_buffers.append(slot)
                    raise
                self._resident_count += 1
            rows.append((key, record, offset, start, n))
        return raw, values, count, rows

    def _commit_prepared_append(self, prepared):
        raw, values, count, rows = prepared
        if not count:
            return
        finalize_batch = getattr(_native, 'finalize_mean_batch', None)
        batched_index = (self.mean_k_backend == 'qw3_cuda' and
                         self.raw_dtype in (torch.bfloat16, torch.float8_e4m3fn) and
                         callable(finalize_batch))
        if batched_index:
            # Source scatter has completed and every destination is owned by
            # this append. Keep token-order FP32 addition and the existing
            # FP16 conversion; only batch the small native calls/GIL handoffs.
            jobs = [(record.raw.view(torch.uint8).numpy(), record.key_sum.numpy(),
                     record.count+n, self.raw_dtype == torch.float8_e4m3fn)
                    for _, record, _, _, n in rows
                    if record.count+n == self.shape[1]]
            if jobs:
                vectorized = 0
                if self.cpu_mean_index_simd:
                    vectorized = finalize_batch(jobs, vectorized=True)
                    if type(vectorized) is not int or not 0 <= vectorized <= len(jobs):
                        raise RuntimeError('Invalid native MeanK SIMD execution count')
                else:
                    finalize_batch(jobs)
                self.mean_index_stats.record(len(jobs), vectorized, self.cpu_mean_index_simd)
        for key, record, offset, start, n in rows:
            record.count += n
            if record.count == self.shape[1]:
                if self.mean_k_backend == 'qw3_cuda':
                    if self.raw_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
                        raise ValueError('Original content-index finalization requires a qualified raw source')
                    finalize = (_native.finalize_mean_fp8 if self.raw_dtype == torch.float8_e4m3fn
                                else _native.finalize_mean_bf16)
                    if not batched_index:
                        finalize(record.raw.view(torch.uint8).numpy(), record.key_sum.numpy(), record.count)
                    if record.index_owner is not None:
                        record.mean_key = torch.frombuffer(record.index_owner, dtype=torch.float16,
                            count=record.key_sum.numel(), offset=record.key_sum.numel()*4).view(record.key_sum.shape)
                        torch.div(record.key_sum, record.count, out=record.mean_key)
                    else:
                        record.mean_key = (record.key_sum / record.count).to(torch.float16)
                self._unsubmitted[key] = record
        self.memory.attach_completion(_Ready())
        assert self.memory.commit_append(count)
        if getattr(self.gc, 'native_history', False):
            # One accepted public packet stamps the current GPU window. The
            # native observer does not scan historical records per decode.
            self.memory._store.gc_begin_step()
        self.flush()

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
        leases, ordered = [], []
        try:
            for record in sorted(self.records.values(), key=lambda r: r.history_begin):
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

    def select_for_queries(self, queries, mandatory_handles=(), *, observe_only=False, max_tokens=None,
                           guided_pressure=False):
        """Reference MeanK-to-native-selection bridge over accepted history only.

        The returned plan still needs fenced GPU assembly and commit_selection.
        Queries must come from the model's content-frame query probe/replay.
        """
        self._require_open()
        from .retrieval import mean_key_scores
        if self.mean_k_backend == 'qw3_cuda':
            from .cuda_scoring import mean_key_scores
        from .graph_integration import query_snapshot, score_and_select
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
                eligible = torch.tensor(score_mask, dtype=torch.bool, device=queries.device)
                mean_scores = None
                if self.query_score_mode == 'overlap_16_8':
                    from .query_window_scoring import score_authority_windows
                    if guided_pressure:
                        raise ValueError('16/8 window mode is qualified for the public Query only')
                    values, stats = score_authority_windows(self, records, queries, means, score_mask)
                    self.query_window_stats = stats
                    scores = values.cpu().tolist()
                elif self.query_score_mode == 'key_fps_4':
                    from .query_key_fps import score_authority_fps
                    if guided_pressure:
                        raise ValueError('4-Key FPS mode is qualified for the public Query only')
                    values, stats = score_authority_fps(self, records, queries, means, score_mask)
                    self.query_key_fps_stats = stats
                    scores = values.cpu().tolist()
                    if getattr(self.graph_options, 'query_mean_quota_fraction', 0):
                        mean_started = time.perf_counter()
                        mean_scores = mean_key_scores(queries, means.to(queries.device), eligible).cpu().tolist()
                        stats['mean_quota_scoring_seconds'] = time.perf_counter()-mean_started
                else:
                    scores = mean_key_scores(queries, means.to(queries.device), eligible).cpu().tolist()
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
        spans = self.memory.begin_rewind(checkpoint)
        graph = getattr(self, 'block_graph', None)
        if graph is not None:
            graph.invalidate()
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
        if not historical:
            self.stop_gc()
        self.poll()
        if self.writes or not self.memory.close(background_retire=historical):
            return False
        if historical and not self.gc.close():
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
        if all_reclaimed:
            if historical and not self.gc.index_released():
                return False
            if self._busy_record_buffers:
                raise RuntimeError('Authority closed with live record staging')
            self._record_buffers.clear()
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
            jobs.append((*job, True) if copy_only else job)
    _native.scatter_history_batch(jobs)
    for authority, prepared in plans:
        authority._commit_prepared_append(prepared)
    return len(plans)
