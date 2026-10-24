"""Four real content Keys per physical block, selected by raw-Key k-center.

Distances choose representatives only. Retrieval uses the original Keys and
per-Query/head maximum dot products before the native block softmax domain.
The derived CPU cache shares the bounded archive pool with window scoring.
"""
import math
import time
from contextlib import contextmanager, nullcontext
from threading import RLock

import torch

from .query_window_scoring import WindowMeanCache


MODE = 'key_fps_4'
REPRESENTATIVES = 4
BLOCK_TOKENS = 32
SCRATCH_LIMIT = 512 * 1024**2


class StaleFPSCacheError(RuntimeError):
    """The owner changed history while a derived index was being built."""


def fps_cache_capacity(records, maximum):
    """Geometric startup, then bounded growth past the 256K history index.

    A single decoded block after 8192 history blocks must not double every
    concurrent owner's full cache. Above that boundary grow by 1024 rows,
    while still allowing a large direct snapshot to reserve its whole index.
    """
    if (type(records) is not int or type(maximum) is not int
            or records < 0 or maximum < 0):
        raise ValueError('Invalid FPS cache row capacity')
    if records > maximum:
        raise MemoryError('FPS index exceeds query_window_cache_bytes')
    if not records:
        return 0
    desired = (1 << (records-1).bit_length()) if records <= 8192 else ((records+1023)//1024)*1024
    return min(maximum, desired)


def _select_key_representatives(keys, valid_counts, *, track_ambiguity=False):
    """Return real Keys and original token offsets for each layer/block/head.

    Input is [layers, blocks, tokens <= 32, KV heads, dim]. Norms and direct
    squared differences use FP32. Argmax selects the lowest original offset
    on ties; already selected offsets are excluded even for identical Keys.
    Output slots beyond min(4, count) contain zero and index -1, which callers
    must exclude from maximum logits. Zero-norm *valid* Keys remain candidates.
    """
    if keys.ndim != 5 or not keys.is_floating_point():
        raise ValueError('Invalid raw-Key representative tensor')
    layers, blocks, tokens, heads, dim = keys.shape
    if not (layers > 0 and heads > 0 and dim > 0 and 0 < tokens <= BLOCK_TOKENS):
        raise ValueError('Invalid raw-Key representative geometry')
    counts = torch.as_tensor(valid_counts, device=keys.device)
    if (counts.shape != (blocks,) or counts.dtype not in
            (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError('Invalid raw-Key valid counts')
    if bool(((counts < 0) | (counts > tokens)).any()):
        raise ValueError('Raw-Key valid counts exceed the token dimension')
    # A cache builder calls this on at most 32 records. Reject unexpectedly
    # large GPU batches before creating the FP32/distance temporaries.
    if keys.device.type == 'cuda' and keys.numel()*4*5 > SCRATCH_LIMIT:
        raise MemoryError('Raw-Key selection scratch exceeds 512 MiB')
    available = (torch.arange(tokens, device=keys.device)[None, None, :, None]
        < counts[None, :, None, None]).expand(layers, blocks, tokens, heads).clone()
    raw = torch.where(available[..., None], keys.float(), 0.)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite original Key in representative index')
    criterion = raw.square().sum(-1)
    if not bool(torch.isfinite(criterion).all()):
        raise ValueError('Nonfinite FP32 raw-Key norm')
    nearest = None
    indices = torch.full((layers, blocks, REPRESENTATIVES, heads), -1,
        dtype=torch.long, device=keys.device)
    uncertain = torch.zeros((layers, blocks, heads), dtype=torch.bool, device=keys.device)
    for slot in range(REPRESENTATIVES):
        candidates = criterion.masked_fill(~available, -torch.inf)
        chosen = candidates.argmax(2)
        if track_ambiguity and tokens > 1:
            best = candidates.topk(2, dim=2).values
            # FP8 inputs give identical elementwise FP32 subtraction/squaring
            # on CPU and GPU implementations. Only summation order may
            # differ. This conservative bound exceeds the two sequential-sum
            # gamma_(dim-1) error bounds; taking min over distances preserves
            # their relative intervals. Ties/near ties use the CPU reference.
            tolerance = 4*dim*torch.finfo(torch.float32).eps*best[:, :, 0].abs()
            uncertain |= torch.isfinite(best[:, :, 1]) & (
                best[:, :, 0]-best[:, :, 1] <= tolerance)
        live = (counts > slot)[None, :, None].expand(layers, blocks, heads)
        indices[:, :, slot] = torch.where(live, chosen, -1)
        available.scatter_(2, chosen[:, :, None, :], False)
        if slot + 1 == REPRESENTATIVES:
            break
        selected = raw.gather(2, chosen[:, :, None, :, None].expand(
            layers, blocks, 1, heads, dim))
        # Direct differences avoid cancellation in ||k||²+||s||²-2k·s.
        distance = (raw-selected).square_().sum(-1)
        if not bool(torch.isfinite(distance).all()):
            raise ValueError('Nonfinite FP32 raw-Key distance')
        nearest = distance if nearest is None else torch.minimum(nearest, distance)
        criterion = nearest
    gather_indices = indices.clamp_min(0)[..., None].expand(
        layers, blocks, REPRESENTATIVES, heads, dim)
    # CPU gather does not support FP8. FP32 represents every finite FP8 value
    # exactly; casting the selected values back preserves the original Keys.
    source = raw if keys.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else keys
    result = source.gather(2, gather_indices)
    result = torch.where(indices[..., None] >= 0, result, 0.).to(keys.dtype)
    return result, indices, uncertain


def select_key_representatives(keys, valid_counts):
    """Reference raw-Key FPS; preserve CPU reduction and deterministic ties."""
    representatives, indices, _ = _select_key_representatives(keys, valid_counts)
    return representatives, indices


def select_key_representatives_optimized(keys, valid_counts, *, device, cache_dtype=None, cpu_pool=None):
    """FP8 GPU construction with CPU reference only for ambiguous groups.

    Returned real Keys and offsets live on CPU. ``cache_dtype`` optionally moves
    the finite FP8 -> cache conversion onto the GPU; its exact embedding is
    qualified only for FP16, BF16 and FP32. A group is one layer, original
    block and KV head. Near-tie groups are recomputed from their original Keys,
    never from an already-selected subset. Unsupported geometry/dtypes retain
    the reference path and report that fallback explicitly.
    """
    device = torch.device(device)
    if cache_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Invalid exact FP8 representative cache dtype')
    groups = keys.shape[0]*keys.shape[1]*keys.shape[3]
    stats = dict(fps_gpu_groups=0, fps_cpu_fallback_groups=0, fps_cpu_groups=0,
        fps_cpu_reference_groups=0, fps_cpu_ambiguity_groups=0,
        fps_cpu_native_groups=0, fps_cpu_pool_busy_groups=0,
        fps_cpu_backend='native' if cpu_pool is not None else 'torch',
        fps_cpu_workers=cpu_pool.stats()['workers'] if cpu_pool is not None else 0,
        fps_cpu_unsupported_fallback_groups=0, fps_device_cache_conversion=False,
        fps_cpu_exact_cache_conversion=False,
        fps_reduction_guard='4 * dim * FP32 epsilon relative top-two gap')
    if keys.device.type != 'cpu':
        raise ValueError('Optimized FPS requires an owned original CPU Key slab')
    if cpu_pool is not None and device.type != 'cpu':
        raise ValueError('Native CPU FPS pool cannot be bound to GPU construction')
    if cpu_pool is not None and keys.dtype == torch.float8_e4m3fn and keys.shape[-1] <= 4096:
        from .query_fps_cpu import select_native_cpu_representatives
        return select_native_cpu_representatives(keys, valid_counts, cpu_pool, cache_dtype=cache_dtype)
    if (device.type != 'cuda' or keys.dtype != torch.float8_e4m3fn or keys.shape[-1] > 4096):
        representatives, indices = select_key_representatives(keys, valid_counts)
        # Keep the old aggregate for saved audit compatibility. A deliberately
        # selected CPU reference path is distinct from a GPU ambiguity guard.
        stats.update(fps_cpu_fallback_groups=groups, fps_cpu_groups=groups,
            fps_cpu_reference_groups=groups,
            fps_cpu_unsupported_fallback_groups=groups if device.type != 'cpu' or cpu_pool is not None else 0,
            fps_fallback_reason=('native CPU construction requires FP8 e4m3fn and dim <= 4096' if cpu_pool is not None else
                'CPU reference construction requested' if device.type == 'cpu' else
                'GPU construction requires FP8 e4m3fn CPU Keys and dim <= 4096'))
        return representatives, indices, stats
    # The additional converted output can coexist with the selected FP8 values.
    # Include it before any device allocation, even though the FPS reduction
    # temporaries normally die before conversion starts.
    converted_bytes = (keys.shape[0]*keys.shape[1]*REPRESENTATIVES*keys.shape[3]*keys.shape[4]
        * torch.empty((), dtype=cache_dtype).element_size() if cache_dtype is not None else 0)
    if keys.numel()*4*5+converted_bytes > SCRATCH_LIMIT:
        raise MemoryError('Optimized FPS selection/conversion scratch exceeds 512 MiB')
    gpu = keys.to(device, non_blocking=keys.is_pinned())
    representatives, indices, uncertain = _select_key_representatives(gpu, valid_counts,
        track_ambiguity=True)
    if cache_dtype is not None:
        # _select_key_representatives already rejected every nonfinite valid
        # FP8 input. Every finite e4m3fn value embeds exactly in these formats;
        # no host FP32 round-trip validation or arithmetic is needed here.
        representatives = representatives.to(cache_dtype)
        stats['fps_device_cache_conversion'] = True
    representatives, indices = representatives.cpu(), indices.cpu()
    triples = uncertain.nonzero().cpu()
    stats.update(fps_gpu_groups=groups, fps_cpu_fallback_groups=len(triples),
        fps_cpu_groups=len(triples), fps_cpu_ambiguity_groups=len(triples),
        fps_fallback_reason='near-tie FP32 reduction guard' if len(triples) else None)
    if len(triples):
        layer, block, head = triples.unbind(1)
        # uint8 advanced indexing is supported for archived FP8 on CPU. The
        # selected group shape keeps each original token's dim reduction intact.
        subset = keys.view(torch.uint8)[layer, block, :, head, :].view(keys.dtype)
        subset = subset.unsqueeze(0).unsqueeze(3)
        counts = torch.as_tensor(valid_counts, device='cpu')[block]
        exact, exact_indices = select_key_representatives(subset, counts)
        if cache_dtype is None:
            representatives.view(torch.uint8)[layer, block, :, head, :] = exact[0, :, :, 0].view(torch.uint8)
        else:
            # Only the ambiguous groups use CPU FPS. Conversion of their
            # original finite FP8 values uses the same exact embedding.
            representatives[layer, block, :, head, :] = exact[0, :, :, 0].to(cache_dtype)
        indices[layer, block, :, head] = exact_indices[0, :, :, 0]
    return representatives, indices, stats


class RawKeyFPSCache(WindowMeanCache):
    """Owner-local real-Key index, with shared bounded/LRU archive storage.

    A row depends only on that block's logical location, handle generation and
    accepted count. Neither adjacent blocks nor Query/transport masks enter its
    signature. No archive record or read lease survives snapshot construction.
    """
    def __init__(self, archive, layers, heads, dim, *, dtype, max_bytes):
        if not (type(layers) is int and type(heads) is int and type(dim) is int
                and min(layers, heads, dim) > 0):
            raise ValueError('Invalid representative cache geometry')
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError('Representative cache requires a floating CPU index')
        self.indices = None
        self._operation_lock = RLock()
        self._epoch = 0
        self._retired = []
        self._slot_by_start, self._slot_starts = {}, []
        self._prebuilt_signatures = {}
        self._growth_rebuilt = 0
        super().__init__(archive, layers, heads, dim, dtype=dtype, max_bytes=max_bytes)
        # Token offsets fit in signed int8 (-1 means absent); count these bytes
        # in addition to the parent's bounded Python metadata allowance.
        self.row_bytes += layers * REPRESENTATIVES * heads

    @property
    def epoch(self):
        with self._lock:
            return self._epoch

    @contextmanager
    def use(self):
        # Mark active before waiting for the owner mutex: pool eviction holds
        # the shared lock and may only clear inactive owners. Never hold that
        # shared lock while waiting for another owner's operation mutex.
        with self._lock:
            self.active += 1
            self.touched = time.monotonic()
        try:
            with self._operation_lock:
                yield self
        finally:
            with self._lock:
                self.active -= 1
                if not self.active and self._retired:
                    self.archive._query_window_cache_bytes -= sum(item[-1] for item in self._retired)
                    self._retired.clear()

    def _drop_storage(self, *, retire):
        if self.bytes:
            if retire and self.active:
                # Existing Query views stay valid and charged until their use
                # ends, even though clear can return without waiting for them.
                self._retired.append((self.tensor, self.indices, self.signatures, self.bytes))
            else:
                self.archive._query_window_cache_bytes -= self.bytes
        self.tensor, self.indices = None, None
        self.signatures, self.capacity, self.bytes = [], 0, 0
        self._slot_by_start, self._slot_starts = {}, []
        self._prebuilt_signatures = {}

    def clear(self):
        with self._lock:
            self._epoch += 1
            self._drop_storage(retire=True)

    def rewind(self, history_end):
        """Invalidate queued work and the changed suffix without dropping prefix rows."""
        if type(history_end) is not int or history_end < 0:
            raise ValueError('Invalid FPS history rewind')
        with self._lock:
            self._epoch += 1
            for i, signature in enumerate(self.signatures):
                if signature is not None and signature[0]+signature[3] > history_end:
                    self.signatures[i] = None
                    self._prebuilt_signatures.pop(signature[0], None)
            return self._epoch

    def _reserve(self, records, *, speculative=False, expected_epoch=None):
        """Grow in bounded steps, preserving rows whenever the bounded peak fits.

        Background work neither evicts owners nor drops an existing index.
        Query construction retains the old lazy fallback if old+new slabs do
        not fit: release the unexposed old slab, then rebuild the missing rows.
        Both allocations and their metadata are charged during a preserving
        growth; a failed allocation leaves the original slab usable.
        """
        maximum = self.max_bytes//self.row_bytes
        if records > maximum:
            raise MemoryError('FPS index exceeds query_window_cache_bytes')
        with self._lock:
            if expected_epoch is not None and expected_epoch != self._epoch:
                raise StaleFPSCacheError('FPS history changed before cache growth')
            if records <= self.capacity:
                return
            capacity = fps_cache_capacity(records, maximum)
            amount = capacity*self.row_bytes
            if not speculative:
                for other in sorted(self.archive._query_window_caches, key=lambda c: c.touched):
                    if self.archive._query_window_cache_bytes+amount <= self.max_bytes:
                        break
                    if other is not self and not other.active:
                        other.clear()
            if self.archive._query_window_cache_bytes+amount > self.max_bytes:
                if speculative:
                    raise MemoryError('FPS background growth exceeds shared cache budget')
                # Only this operation owns mutable cache views. External clear
                # uses retirement instead; it must not discount live readers.
                self._growth_rebuilt += sum(s is not None for s in self.signatures)
                self._drop_storage(retire=False)
            if self.archive._query_window_cache_bytes+amount > self.max_bytes:
                raise MemoryError('FPS cache is occupied by active scoring owners')
            layers, heads, _ = self.geometry
            dim = self.geometry[2]
            self.archive._query_window_cache_bytes += amount
            old_tensor, old_indices = self.tensor, self.indices
            old_capacity, old_bytes, epoch = self.capacity, self.bytes, self._epoch
            old_signatures, old_starts = self.signatures, self._slot_starts
        # Slow full-prefix copying must not delay clear()/rewind(). The pending
        # new allocation remains charged, and an epoch check rejects its result
        # if the owner invalidates the index during the copy.
        try:
            tensor = torch.empty((layers, capacity, REPRESENTATIVES, heads, dim), dtype=self.dtype)
            indices = torch.full((layers, capacity, REPRESENTATIVES, heads), -1, dtype=torch.int8)
            signatures, starts = [None]*capacity, [None]*capacity
            if old_capacity:
                tensor[:, :old_capacity].copy_(old_tensor)
                indices[:, :old_capacity].copy_(old_indices)
                signatures[:old_capacity] = old_signatures
                starts[:old_capacity] = old_starts
        except BaseException:
            with self._lock:
                self.archive._query_window_cache_bytes -= amount
            raise
        with self._lock:
            if epoch != self._epoch:
                self.archive._query_window_cache_bytes -= amount
                raise StaleFPSCacheError('FPS history changed during cache growth')
            self.tensor, self.indices = tensor, indices
            self.signatures, self._slot_starts = signatures, starts
            self.capacity, self.bytes = capacity, amount
            self.archive._query_window_cache_bytes -= old_bytes

    def _swap_slots(self, left, right):
        """Swap under the pool lock; a copy failure invalidates both rows."""
        before_left, before_right = self.signatures[left], self.signatures[right]
        start_left, start_right = self._slot_starts[left], self._slot_starts[right]
        self.signatures[left] = self.signatures[right] = None
        # One representative row is temporary, independent of total history.
        tmp, ids = self.tensor[:, left].clone(), self.indices[:, left].clone()
        self.tensor[:, left].copy_(self.tensor[:, right])
        self.indices[:, left].copy_(self.indices[:, right])
        self.tensor[:, right].copy_(tmp)
        self.indices[:, right].copy_(ids)
        self.signatures[left], self.signatures[right] = before_right, before_left
        self._slot_starts[left], self._slot_starts[right] = start_right, start_left
        if start_left is not None:
            self._slot_by_start[start_left] = right
        if start_right is not None:
            self._slot_by_start[start_right] = left

    def _canonical_layout(self, starts):
        """Reorder only at Query time; append-only prebuild never scans history."""
        with self._lock:
            targets = {start:i for i, start in enumerate(starts)}
            for i, start in enumerate(starts):
                source = self._slot_by_start.get(start)
                if source == i:
                    continue
                if source is not None:
                    self._swap_slots(i, source)
                    continue
                previous = self._slot_starts[i]
                # A missing Query row may occupy a slot that already holds a
                # later, successfully prebuilt history block. Move that block
                # to its final position before installing the missing row.
                # Each swap places one known row; chains end at an empty slot
                # or a history block excluded from this Query's canonical set.
                while previous in targets:
                    target = targets[previous]
                    if target <= i:
                        raise RuntimeError('Inconsistent FPS canonical slot index')
                    self._swap_slots(i, target)
                    previous = self._slot_starts[i]
                self.signatures[i] = None
                if previous is not None:
                    self._slot_by_start.pop(previous, None)
                    self._prebuilt_signatures.pop(previous, None)
                self._slot_starts[i] = start
                self._slot_by_start[start] = i
            for i in range(len(starts), self.capacity):
                previous = self._slot_starts[i]
                if previous is not None:
                    self._slot_by_start.pop(previous, None)
                    self._prebuilt_signatures.pop(previous, None)
                self._slot_starts[i] = self.signatures[i] = None

    def _publish_batch(self, batch, representatives, indices, signatures, *, optimized=False,
                       expected_epoch=None):
        """Publish a whole batch only after both value and offset writes succeed.

        Any partial write leaves every affected signature invalid. Subsequent
        snapshots reconstruct all affected records before exposing those slots.
        """
        with self._lock:
            if expected_epoch is not None and expected_epoch != self._epoch:
                raise StaleFPSCacheError('FPS history changed before publication')
            for i in batch:
                self.signatures[i] = None
            contiguous = optimized and batch[-1]-batch[0]+1 == len(batch)
            if contiguous:
                begin, end = batch[0], batch[-1]+1
                self.tensor[:, begin:end].copy_(representatives)
                self.indices[:, begin:end].copy_(indices)
            else:
                slots = torch.tensor(batch, dtype=torch.long)
                self.tensor.index_copy_(1, slots, representatives)
                self.indices.index_copy_(1, slots, indices.to(torch.int8))
            for i in batch:
                self.signatures[i] = signatures[i]
            return contiguous

    def prebuild(self, records, raw_keys, *, epoch, optimized=True, device='cpu', cpu_pool=None):
        """Build at most 32 immutable, fully accepted blocks without authority IO.

        ``raw_keys`` owns [L, B, 32, Hkv, D] original content Keys. The caller
        holds archive leases until this call returns; no record/lease is kept.
        Raw FPS runs outside the owner/publication locks. A rewind, clear or
        eviction invalidates its epoch before anything can be published.
        Background capacity pressure skips work; the Query exact path repairs
        any missing row. Partial or speculative blocks are never admitted.
        """
        records = tuple(records)
        layers, heads, dim = self.geometry
        signatures = [(r.history_begin, r.handle.slot, r.handle.generation, r.count) for r in records]
        starts = [sig[0] for sig in signatures]
        if (not 0 < len(records) <= 32 or type(epoch) is not int
                or any(type(s) is not int or s < 0 or s % BLOCK_TOKENS for s in starts)
                or any(a >= b for a, b in zip(starts, starts[1:]))
                or any(type(r.count) is not int or r.count != BLOCK_TOKENS for r in records)
                or raw_keys.shape != (layers, len(records), BLOCK_TOKENS, heads, dim)
                or raw_keys.device.type != 'cpu' or not raw_keys.is_floating_point()):
            raise ValueError('FPS prebuild requires one to 32 sealed original blocks')
        scratch = (raw_keys.numel()*4*5 + layers*len(records)*REPRESENTATIVES*heads*dim
            * torch.empty((), dtype=self.dtype).element_size())
        if scratch > SCRATCH_LIMIT:
            raise MemoryError('FPS prebuild selection scratch exceeds 512 MiB')
        stats = dict(status='stale', optimized=optimized, cache_hit_blocks=0,
            cache_rebuilt_blocks=0, fps_select_seconds=0., fps_publish_seconds=0.,
            fps_gpu_groups=0, fps_cpu_fallback_groups=0, fps_cpu_groups=0,
            fps_cpu_reference_groups=0, fps_cpu_ambiguity_groups=0,
            fps_cpu_native_groups=0, fps_cpu_pool_busy_groups=0,
            fps_cpu_backend='native' if cpu_pool is not None else 'torch',
            fps_cpu_workers=cpu_pool.stats()['workers'] if cpu_pool is not None else 0,
            fps_cpu_unsupported_fallback_groups=0, cache_epoch=epoch,
            construction_device=str(torch.device(device)) if optimized else 'cpu',
            fps_selection_scratch_bound_bytes=scratch,
            fps_key_slab_bytes_max=raw_keys.numel()*raw_keys.element_size())
        with self.use():
            with self._lock:
                if epoch != self._epoch:
                    return stats
                dirty = [i for i, sig in enumerate(signatures) if (
                    self._slot_by_start.get(sig[0]) is None or
                    self.signatures[self._slot_by_start[sig[0]]] != sig)]
                stats['cache_hit_blocks'] = len(records)-len(dirty)
                if not dirty:
                    stats['status'] = 'hit'
                    return stats
                new = sum(signatures[i][0] not in self._slot_by_start for i in dirty)
                needed = len(self._slot_by_start)+new
            try:
                self._reserve(needed, speculative=True, expected_epoch=epoch)
            except StaleFPSCacheError:
                return stats
            except MemoryError as error:
                stats.update(status='skipped', skip_reason='cache_budget', detail=str(error))
                return stats
        # Only new rows enter FPS. FP8 CPU index_select is not universally
        # available; byte selection keeps every original value unchanged.
        if len(dirty) == len(records):
            raw = raw_keys
        else:
            select = torch.tensor(dirty, dtype=torch.long)
            raw = (raw_keys.view(torch.uint8).index_select(1, select).view(raw_keys.dtype)
                if raw_keys.element_size() == 1 else raw_keys.index_select(1, select))
        started = time.perf_counter()
        if optimized:
            representatives, indices, selection = select_key_representatives_optimized(
                raw, [BLOCK_TOKENS]*len(dirty), device=device, cache_dtype=self.dtype, cpu_pool=cpu_pool)
            stats.update(selection)
        else:
            representatives, indices = select_key_representatives(raw, [BLOCK_TOKENS]*len(dirty))
            stats.update(fps_cpu_groups=layers*len(dirty)*heads,
                fps_cpu_reference_groups=layers*len(dirty)*heads)
        if not (stats.get('fps_device_cache_conversion', False) or stats.get('fps_cpu_exact_cache_conversion', False)):
            converted = representatives.to(self.dtype)
            if not torch.equal(converted.float(), representatives.float()):
                raise ValueError('Representative cache dtype changes an original Key')
        else:
            if representatives.dtype != self.dtype or raw.dtype != torch.float8_e4m3fn:
                raise ValueError('Unqualified representative device conversion')
            converted = representatives
        stats['fps_select_seconds'] = time.perf_counter()-started
        published = time.perf_counter()
        with self.use():
            with self._lock:
                if epoch != self._epoch:
                    return stats
                new = sum(signatures[i][0] not in self._slot_by_start for i in dirty)
                needed = len(self._slot_by_start)+new
            try:
                self._reserve(needed, speculative=True, expected_epoch=epoch)
            except StaleFPSCacheError:
                return stats
            except MemoryError as error:
                stats.update(status='skipped', skip_reason='cache_budget', detail=str(error))
                return stats
            with self._lock:
                if epoch != self._epoch:
                    return stats
                slots, target_signatures = [], {}
                next_slot = len(self._slot_by_start)
                for i in dirty:
                    signature = signatures[i]
                    slot = self._slot_by_start.get(signature[0])
                    if slot is None:
                        slot, next_slot = next_slot, next_slot+1
                    slots.append(slot)
                    target_signatures[slot] = signature
                contiguous = self._publish_batch(slots, converted, indices, target_signatures,
                    optimized=optimized, expected_epoch=epoch)
                for slot in slots:
                    start = target_signatures[slot][0]
                    self._slot_by_start[start], self._slot_starts[slot] = slot, start
                    self._prebuilt_signatures[start] = target_signatures[slot]
                stats.update(status='published', cache_rebuilt_blocks=len(dirty),
                    cache_bytes=self.bytes, fps_publication_slice_batches=int(contiguous),
                    fps_publication_index_copy_batches=int(not contiguous))
        stats['fps_publish_seconds'] = time.perf_counter()-published
        return stats

    def snapshot(self, authority, records, *, optimized=False, device='cpu', cpu_pool=None):
        # Views must remain inside an outer use() until scoring finishes. This
        # inner use also makes direct snapshot construction mutually exclusive.
        with self.use():
            return self._snapshot(authority, records, optimized=optimized, device=device, cpu_pool=cpu_pool)

    def _snapshot(self, authority, records, *, optimized=False, device='cpu', cpu_pool=None):
        starts, counts = [r.history_begin for r in records], [r.count for r in records]
        if (any(type(s) is not int or s < 0 or s % BLOCK_TOKENS for s in starts)
                or any(type(n) is not int or not 0 < n <= BLOCK_TOKENS for n in counts)
                or any(a >= b for a, b in zip(starts, starts[1:]))
                or any(s+n > authority.history_end for s, n in zip(starts, counts))):
            raise ValueError('Invalid original 32-token representative history index')
        epoch = self.epoch
        growth_rebuilt = self._growth_rebuilt
        self._reserve(len(records), expected_epoch=epoch)
        with self._lock:
            if epoch != self._epoch:
                raise StaleFPSCacheError('FPS history changed before snapshot construction')
            self._canonical_layout(starts)
            signatures = [(r.history_begin, r.handle.slot, r.handle.generation, r.count)
                for r in records]
            dirty = [i for i, sig in enumerate(signatures) if self.signatures[i] != sig]
            prebuilt_hits = sum(self.signatures[i] == sig
                and self._prebuilt_signatures.get(sig[0]) == sig for i, sig in enumerate(signatures))
        layers, heads, dim = self.geometry
        work = dict(key_read_bytes=0, value_read_bytes=0, draft_key_read_bytes=0,
            key_read_batches=0, key_resident_bytes=0, key_region_seconds=0.,
            fps_gpu_groups=0, fps_cpu_fallback_groups=0, fps_cpu_groups=0,
            fps_cpu_reference_groups=0, fps_cpu_ambiguity_groups=0,
            fps_cpu_native_groups=0, fps_cpu_pool_busy_groups=0,
            fps_cpu_backend='native' if cpu_pool is not None else 'torch',
            fps_cpu_workers=cpu_pool.stats()['workers'] if cpu_pool is not None else 0,
            fps_cpu_unsupported_fallback_groups=0, fps_read_seconds=0.,
            fps_select_seconds=0., fps_publish_seconds=0.)
        work.update(fps_key_slab_bytes_max=0, fps_selection_scratch_bound_bytes=0)
        work.update(fps_device_cache_conversion_batches=0, fps_cpu_cache_conversion_batches=0, fps_publication_slice_batches=0,
            fps_publication_index_copy_batches=0, fps_publication_strict_check_batches=0)
        fallback_reasons = set()
        for begin in range(0, len(dirty), 32):
            batch = dirty[begin:begin+32]
            started = time.perf_counter()
            payloads = None
            exact_device_conversion = False
            if optimized:
                scratch = (layers*len(batch)*BLOCK_TOKENS*heads*dim*4*5
                    + layers*len(batch)*REPRESENTATIVES*heads*dim
                    * torch.empty((), dtype=self.dtype).element_size())
                if scratch > SCRATCH_LIMIT:
                    raise MemoryError('Optimized FPS selection scratch exceeds 512 MiB')
                work['fps_selection_scratch_bound_bytes'] = max(
                    work['fps_selection_scratch_bound_bytes'], scratch)
                packed, read_stats = authority.read_score_keys([records[i].handle for i in batch],
                    pin_memory=torch.device(device).type == 'cuda')
                work['fps_key_slab_bytes_max'] = max(work['fps_key_slab_bytes_max'],
                    packed.numel()*packed.element_size())
                raw = packed.permute(1, 0, 2, 3, 4)
                if raw.shape != (layers, len(batch), BLOCK_TOKENS, heads, dim):
                    raise ValueError('Scoring Key-only archive geometry changed')
                for name, value in read_stats.items():
                    work[name] = work.get(name, 0)+value
            else:
                payloads = authority.read_records([records[i].handle for i in batch])
                if len(payloads) != len(batch):
                    raise ValueError('Incomplete original-Key archive batch')
                dtype = payloads[0][0].dtype
                raw = torch.zeros((layers, len(batch), BLOCK_TOKENS, heads, dim), dtype=dtype)
                for row, (i, payload) in enumerate(zip(batch, payloads)):
                    key = payload[0]
                    if (key.device.type != 'cpu' or key.ndim != 4 or key.shape[0] < layers
                            or key.shape[1:] != (counts[i], heads, dim) or key.dtype != dtype):
                        raise ValueError('Original-Key archive geometry changed')
                    raw[:, row, :counts[i]].copy_(key[:layers])
            selected_at = time.perf_counter()
            work['fps_read_seconds'] += selected_at-started
            if optimized:
                representatives, indices, select_stats = select_key_representatives_optimized(
                    raw, [counts[i] for i in batch], device=device, cache_dtype=self.dtype, cpu_pool=cpu_pool)
                exact_device_conversion = (select_stats['fps_device_cache_conversion'] or
                    select_stats.get('fps_cpu_exact_cache_conversion', False))
                work['fps_device_cache_conversion_batches'] += int(select_stats['fps_device_cache_conversion'])
                work['fps_cpu_cache_conversion_batches'] += int(select_stats.get('fps_cpu_exact_cache_conversion', False))
                for key in ('fps_gpu_groups', 'fps_cpu_fallback_groups', 'fps_cpu_groups',
                        'fps_cpu_reference_groups', 'fps_cpu_ambiguity_groups',
                        'fps_cpu_native_groups', 'fps_cpu_pool_busy_groups', 'fps_cpu_unsupported_fallback_groups'):
                    work[key] += select_stats[key]
                if select_stats['fps_fallback_reason']:
                    fallback_reasons.add(select_stats['fps_fallback_reason'])
                del packed
            else:
                representatives, indices = select_key_representatives(raw, [counts[i] for i in batch])
                work['fps_cpu_groups'] += layers*len(batch)*heads
                work['fps_cpu_reference_groups'] += layers*len(batch)*heads
            published_at = time.perf_counter()
            work['fps_select_seconds'] += published_at-selected_at
            if exact_device_conversion:
                if representatives.dtype != self.dtype or raw.dtype != torch.float8_e4m3fn:
                    raise ValueError('Unqualified representative device conversion')
                converted = representatives
            else:
                # CPU/non-FP8/reference paths retain their strict original
                # conversion check; no unqualified dtype gets a fast path.
                converted = representatives.to(self.dtype)
                if not torch.equal(converted.float(), representatives.float()):
                    raise ValueError('Representative cache dtype changes an original Key')
                work['fps_publication_strict_check_batches'] += 1
            contiguous = self._publish_batch(batch, converted, indices, signatures,
                optimized=optimized, expected_epoch=epoch)
            with self._lock:
                for i in batch:
                    self._prebuilt_signatures.pop(signatures[i][0], None)
            work['fps_publication_slice_batches' if contiguous else
                'fps_publication_index_copy_batches'] += 1
            work['fps_publish_seconds'] += time.perf_counter()-published_at
            del payloads, raw, representatives, converted, indices
        with self._lock:
            if self._epoch != epoch:
                raise StaleFPSCacheError('FPS history changed during snapshot')
            if self.tensor is None:
                representatives = torch.empty((layers, 0, REPRESENTATIVES, heads, dim), dtype=self.dtype)
                indices = torch.empty((layers, 0, REPRESENTATIVES, heads), dtype=torch.int8)
            else:
                representatives, indices = self.tensor[:, :len(records)], self.indices[:, :len(records)]
        distinct = [min(REPRESENTATIVES, n) for n in counts]
        return representatives, indices, dict(**work, optimized=optimized,
            construction_device=str(torch.device(device)) if optimized else 'cpu',
            fps_fallback_reasons=sorted(fallback_reasons), cache_bytes=self.bytes,
            cache_epoch=epoch,
            cache_capacity_bytes=self.max_bytes, cache_hit_blocks=len(records)-len(dirty),
            cache_rebuilt_blocks=len(dirty), valid_representatives=sum(distinct)*layers*heads,
            cache_prebuilt_hit_blocks=prebuilt_hits,
            cache_growth_rebuilt_blocks=self._growth_rebuilt-growth_rebuilt,
            distinct_token_offsets_min=min(distinct, default=0),
            distinct_token_offsets_max=max(distinct, default=0))


# Keep the integration name short without creating a second cache type/pool.
KeyFPSCache = RawKeyFPSCache


def key_fps_scores(queries, representatives, indices, score_mask, *, scale=None,
                   query_half=False, query_chunk=8, optimized=False, execution_stats=None):
    """Real-Key logits -> per-head/Query block max -> block softmax -> mean/sum.

    Only one layer and up to eight Query rows are resident in scratch. Native
    score_mask owns the complete normalization domain; transport is not an input.
    """
    if queries.ndim != 4 or representatives.ndim != 5:
        raise ValueError('Invalid representative scoring tensor ranks')
    layers, rows, heads, dim = queries.shape
    rl, blocks, slots, kvheads, rd = representatives.shape
    if (rl != layers or rd != dim or slots != REPRESENTATIVES or not kvheads
            or heads % kvheads or not layers or not heads or not dim
            or indices.shape != (layers, blocks, REPRESENTATIVES, kvheads)
            or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or score_mask.shape != (blocks,) or score_mask.dtype != torch.bool
            or type(query_chunk) is not int or not 0 < query_chunk <= 8):
        raise ValueError('Invalid GQA/representative/block score geometry')
    if bool(((indices < -1) | (indices >= BLOCK_TOKENS)).any()):
        raise ValueError('Representative index is outside its original block')
    scale = dim**-.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError('Invalid Query scale')
    device = queries.device
    score_mask = score_mask.to(device)
    scores = torch.zeros(blocks, dtype=torch.float32, device=device)
    if not bool(score_mask.any()) or not rows:
        return scores
    # Includes a possible contiguous Key packing copy, logits, pooled logits,
    # probabilities, masks and Q conversions. No all-layer/all-Query logits.
    scratch = (2*blocks*REPRESENTATIVES*kvheads*dim*4
        + 3*query_chunk*heads*blocks*REPRESENTATIVES*4
        + 3*query_chunk*heads*blocks*4 + 2*query_chunk*heads*dim*4
        + blocks*REPRESENTATIVES*kvheads*2 + blocks*8)
    if optimized:
        scratch += blocks*REPRESENTATIVES*kvheads*dim*representatives.element_size()
    if scratch > SCRATCH_LIMIT:
        return _key_fps_tiled_scores(queries, representatives, indices, score_mask,
            scale=scale, query_half=query_half, query_chunk=query_chunk,
            cache_layer=optimized, execution_stats=execution_stats)
    for layer in range(layers):
        valid = (indices[layer] >= 0).permute(2, 0, 1).to(device)
        if bool((score_mask[None, :] & ~valid.any(-1)).any()):
            raise ValueError('Scorable block has no valid Key representative')
        if optimized:
            # Transfer the compact cached dtype first; convert and pack on the
            # destination device instead of host-side strided FP16 -> FP32 copy.
            source = representatives[layer].to(device=device)
            keys = source.permute(2, 3, 0, 1).reshape(
                kvheads, dim, blocks*REPRESENTATIVES).float().contiguous()
            del source
        else:
            keys = representatives[layer].permute(2, 3, 0, 1).reshape(
                kvheads, dim, blocks*REPRESENTATIVES).to(device=device, dtype=torch.float32).contiguous()
        for begin in range(0, rows, query_chunk):
            query = queries[layer, begin:begin+query_chunk]
            if query_half:
                query = query.half()
            n = query.shape[0]
            query = query.float().reshape(n, kvheads, heads//kvheads, dim)
            logits = torch.einsum('tkgd,kdw->tkgw', query, keys)
            logits.mul_(scale)
            logits = logits.reshape(n, kvheads, heads//kvheads, blocks, REPRESENTATIVES)
            logits.masked_fill_(~valid[None, :, None, :, :], -torch.inf)
            pooled = logits.amax(-1)
            pooled.masked_fill_(~score_mask, -torch.inf)
            probability = pooled.softmax(-1)
            scores.add_(probability.sum((0, 1, 2)), alpha=1/(layers*heads))
            del query, logits, pooled, probability
        del keys, valid
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite real-Key representative block probability')
    return scores


def _key_fps_tiled_scores(queries, representatives, indices, score_mask, *, scale,
                          query_half, query_chunk, cache_layer=False, execution_stats=None):
    """Bound Key scratch for long histories; normalize over *all* blocks.

    Max reduction is local to each original block. Only Key/logit tiles are
    bounded; block softmax is never applied separately to a tile.
    """
    layers, rows, heads, dim = queries.shape
    blocks, slots, kvheads = representatives.shape[1:4]
    device = queries.device
    scores = torch.zeros(blocks, dtype=torch.float32, device=device)
    uploaded_bytes = cached_layers = 0
    peak_bound = 0
    for layer in range(layers):
        valid = (indices[layer] >= 0).permute(2,0,1).to(device)
        if bool((score_mask[None,:] & ~valid.any(-1)).any()):
            raise ValueError('Scorable block has no valid Key representative')
        # Keep the compact source for one layer, not all owners/layers. The
        # exact same tile packing and FP32 dot/reduction order remain below.
        # Reserve it only if the original tile still fits; never change GEMM
        # shapes or exceed the shared scratch bound to enable this cache.
        n_max = min(rows, query_chunk)
        permanent_max = (3*n_max*heads*blocks*4 + 2*n_max*heads*dim*4
            + blocks*slots*kvheads*2 + blocks*8)
        per_block_max = (3*slots*kvheads*dim*4 + slots*kvheads*dim*representatives.element_size()
            + 3*n_max*heads*slots*4 + n_max*heads*4)
        tile_max = min(1024, blocks, (SCRATCH_LIMIT-permanent_max)//per_block_max)
        layer_bytes = representatives[layer].numel()*representatives.element_size()
        resident = None
        if (cache_layer and device.type == 'cuda' and representatives.device.type == 'cpu'
                and tile_max > 0 and
                layer_bytes+permanent_max+tile_max*per_block_max <= SCRATCH_LIMIT):
            resident = representatives[layer].to(device=device)
            uploaded_bytes += layer_bytes
            cached_layers += 1
        for begin in range(0,rows,query_chunk):
            query = queries[layer,begin:begin+query_chunk]
            if query_half:
                query = query.half()
            n = query.shape[0]
            query = query.float().reshape(n,kvheads,heads//kvheads,dim)
            permanent = (3*n*heads*blocks*4 + 2*n*heads*dim*4 + blocks*slots*kvheads*2 + blocks*8)
            per_block = (3*slots*kvheads*dim*4 + slots*kvheads*dim*representatives.element_size()
                + 3*n*heads*slots*4 + n*heads*4)
            tile = min(1024,blocks,(SCRATCH_LIMIT-permanent)//per_block)
            if tile < 1:
                raise MemoryError('Long-history FPS scoring cannot fit one Key tile in bounded scratch (512 MiB default)')
            # The final, shorter Q chunk can choose a larger tile than n_max.
            # A cached source must fit that tile as well.
            if resident is not None and layer_bytes+permanent+tile*per_block > SCRATCH_LIMIT:
                del resident
                resident = None
            peak_bound = max(peak_bound, permanent+tile*per_block+
                (layer_bytes if resident is not None else 0))
            pooled = torch.full((n,kvheads,heads//kvheads,blocks),-torch.inf,
                dtype=torch.float32,device=device)
            for left in range(0,blocks,tile):
                right = min(blocks,left+tile)
                if resident is None:
                    source = representatives[layer,left:right].to(device=device)
                    if representatives.device.type == 'cpu' and device.type == 'cuda':
                        uploaded_bytes += source.numel()*source.element_size()
                else:
                    source = resident[left:right]
                keys = source.permute(2,3,0,1).reshape(kvheads,dim,(right-left)*slots).float().contiguous()
                del source
                logits = torch.einsum('tkgd,kdw->tkgw',query,keys)
                logits.mul_(scale)
                logits = logits.reshape(n,kvheads,heads//kvheads,right-left,slots)
                logits.masked_fill_(~valid[None,:,None,left:right,:],-torch.inf)
                pooled[...,left:right] = logits.amax(-1)
                del keys,logits
            pooled.masked_fill_(~score_mask,-torch.inf)
            probability = pooled.softmax(-1)
            scores.add_(probability.sum((0,1,2)),alpha=1/(layers*heads))
            del query,pooled,probability
        del valid, resident
    if execution_stats is not None:
        execution_stats.update(scoring_path='tiled_layer_cache' if cached_layers else 'tiled_reference',
            scoring_cached_layers=cached_layers, scoring_key_upload_bytes=uploaded_bytes,
            scoring_scratch_bound_bytes=peak_bound, scoring_scratch_limit_bytes=SCRATCH_LIMIT)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite tiled representative block probability')
    return scores


def score_authority_fps(authority, records, queries, block_means, score_mask):
    started = time.perf_counter()
    cache = authority.query_window_cache
    layers, heads, dim = cache.geometry
    options = getattr(authority, 'graph_options', None)
    optimized = getattr(options, 'query_fps_optimized', False)
    if block_means.shape != (layers, len(records), heads, dim):
        raise ValueError('Original block index differs from the representative index')
    prebuild = getattr(authority, 'fps_prebuild', None)
    configured_cpu = (getattr(options, 'query_fps_prebuild', False)
        and getattr(options, 'query_fps_prebuild_device', 'cpu') == 'cpu')
    worker_device = getattr(getattr(prebuild, 'manager', None), 'device', None)
    if configured_cpu and worker_device is not None and torch.device(worker_device).type != 'cpu':
        raise ValueError('CPU FPS configuration has a non-CPU prebuild worker')
    cpu_construction = configured_cpu or (worker_device is not None
        and torch.device(worker_device).type == 'cpu')
    build_device = torch.device('cpu') if cpu_construction else queries.device
    cpu_pool = getattr(getattr(prebuild, 'manager', None), 'cpu_pool', None)
    if getattr(options, 'query_fps_cpu_backend', 'torch') == 'native' and cpu_pool is None:
        raise ValueError('Native CPU FPS configuration lacks its shared worker pool')
    with prebuild.query() if prebuild is not None else nullcontext():
        with cache.use():
            cache_started = time.perf_counter()
            representatives, indices, stats = cache.snapshot(authority, records,
                optimized=optimized, device=build_device, cpu_pool=cpu_pool)
            cache_seconds = time.perf_counter()-cache_started
            scoring_started = time.perf_counter()
            execution_stats = {}
            scores = key_fps_scores(queries, representatives, indices,
                torch.tensor(score_mask, dtype=torch.bool),
                query_half=authority.mean_k_backend == 'qw3_cuda', optimized=optimized,
                execution_stats=execution_stats)
            stats.update(execution_stats)
            if cache.epoch != stats['cache_epoch']:
                raise StaleFPSCacheError('FPS history changed during Query scoring')
            scoring_seconds = time.perf_counter()-scoring_started
            del representatives, indices
    if prebuild is not None:
        stats['prebuild'] = prebuild.snapshot()
    stats.update(method=MODE, representatives_per_block=REPRESENTATIVES,
        scoring_device=str(queries.device),
        fps_cpu_fallback_groups_semantics=('Legacy aggregate includes CPU reference routing; '
            'reference, ambiguity and unsupported fallback groups are reported separately'),
        block_tokens=BLOCK_TOKENS, selection='raw_key_farthest_first',
        distance='FP32 direct squared Euclidean on raw Keys',
        first_representative='maximum raw-Key squared norm',
        tie_break='lowest original token offset', fallback_blocks=0,
        normalization='block softmax after per-layer/per-Query/per-head representative max',
        key_normalization='none', cache_key_dtype=str(cache.dtype),
        rope='existing content frame; no additional RoPE',
        cache_build_seconds=cache_seconds,
        cache_timing_scope='snapshot: capacity checks, archive reads, FPS and publication',
        scoring_seconds=scoring_seconds,
        seconds=time.perf_counter()-started)
    return scores, stats
