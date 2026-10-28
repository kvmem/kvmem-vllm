"""Bounded immutable-source assembly into native vLLM FP8 working pages.

The scheduler must hold the request's physical pages through the returned CUDA
fence. Only after it completes may it trim the old allocation or resume execution.
Immutable CPU reads, H2D copies and native assembly use two bounded staging
slots. Each GPU slot remains fenced until its assembly consumers finish.
"""
from dataclasses import dataclass, field
from threading import RLock

import time

import torch

from .assembly import assemble_fp8
from .codec import native_vllm_fp8_views
from .memory import Phase
from .transfer import PackedTransfers


@dataclass
class LayerWindow:
    key: torch.Tensor
    value: torch.Tensor
    cos_sin: torch.Tensor
    key_scale: torch.Tensor
    kernel_page_table: torch.Tensor
    restore_group: int | None = None


@dataclass
class AssemblyResult:
    event: torch.cuda.Event
    owners: tuple
    rebuilt_tokens: int
    skipped_tokens: int
    chunks: int
    host_prepare_seconds: float = 0.0
    uploaded_bytes: int = 0
    read_stats: dict = field(default_factory=dict)
    raw_cache_hit_tokens: int = 0
    resident_v_reused_tokens: int = 0
    raw_cache_capacity_bytes: int = 0
    device_temp_bytes: int = 0
    split_tokens: int = 0
    partial_tokens: int = 0

    def query(self):
        return self.event.query()


class _UnfencedAssembly:
    """Retain a failed transaction when CUDA cannot prove its work has ended."""

    def query(self):
        return False


def _validate_destinations(remaps, by_slot, layers):
    """Check the whole plan against every layer before any cache writes.

    Layer extents are immutable for this assembly. Reduce them once, while
    retaining the original per-layer error order when a bound is exceeded.
    """
    bounds = None
    for remap in remaps:
        record = by_slot.get(remap.block_id)
        if record is None or record.count != remap.n_tokens or remap.to_base < 0:
            raise ValueError("Selected generation/extent no longer matches authority")
        if bounds is None:
            bounds = tuple((layer.kernel_page_table.numel() * layer.key.shape[1],
                            layer.cos_sin.shape[0]) for layer in layers)
            capacity = min((limit[0] for limit in bounds), default=float('inf'))
            rope = min((limit[1] for limit in bounds), default=float('inf'))
        end = remap.to_base + remap.n_tokens
        if end > capacity or end > rope:
            for layer_capacity, layer_rope in bounds:
                if end > layer_capacity:
                    raise ValueError("Selected working window exceeds native allocated pages")
                if end > layer_rope:
                    raise ValueError("Selected position exceeds the model RoPE table")


def _validate_resident_sources(remaps, layers):
    """Check resident V extents using immutable layer geometry once per plan.

    This only reads tensor shape/numel metadata. Source pages and V bytes stay
    on the device, and all source validation still precedes the first snapshot.
    """
    capacity = None
    for remap in remaps:
        if remap.from_base < 0:
            raise ValueError('Resident V source exceeds retained native pages')
        if capacity is None:
            capacity = min((layer.kernel_page_table.numel() * layer.value.shape[1]
                            for layer in layers), default=float('inf'))
        if remap.from_base + remap.n_tokens > capacity:
            raise ValueError('Resident V source exceeds retained native pages')


class _AssemblyStaging:
    """Runner-owned scratch; neither selected pages nor archived records live here.

    Host slots are protected by PackedTransfers' DMA events. Device slots have
    separate consumer events: completing H2D alone does not permit overwriting
    bytes that a re-RoPE kernel still reads. Allocate lazily so an all-skip plan
    needs no staging, and keep only two chunks across requests and refreshes.
    """
    slots = 2

    def __init__(self, slot_bytes, device, value_capacity_bytes=0):
        self.slot_bytes, self.device = slot_bytes, device
        self.transfers = None
        self.buffers = []
        self.ready = []
        self.cursor = 0
        self.owner = None
        self.value_capacity_bytes=value_capacity_bytes
        self.value_buffer=None
        self.value_ready=None
        self._lock = RLock()
        self.bandwidth_limiter = None
        self.fused_host = None
        self.fused_device = None
        self.fused_event = None
        self.fused_active = False
        self.fused_poisoned = False
        self.fused_binding = None
        self.fused_owners = ()

    def claim(self, owner):
        with self._lock:
            if self.fused_poisoned:
                raise RuntimeError('Assembly metadata has an unfenced failed upload')
            if self.owner is not None:
                raise RuntimeError("Assembly staging already has a CPU producer")
            self.owner = owner

    def poll_fused_table(self):
        """Whole-consumer completion also protects the pinned metadata upload."""
        with self._lock:
            if self.fused_poisoned or self.fused_active:
                return False
            if self.fused_event is not None and not self.fused_event.query():
                return False
            self.fused_owners = ()
            self.fused_binding = None
            self.fused_event = None
            return True

    def prepare_fused_table(self, owner, authority, plan, spec):
        with self._lock:
            if self.owner is not owner:
                raise RuntimeError('Fused metadata requires its current assembly owner')
            if self.fused_poisoned:
                raise RuntimeError('Assembly metadata has an unfenced failed upload')
            if not self.poll_fused_table():
                return None, 'busy'
            shape = (len(spec.rows),12)
            if self.fused_host is not None and tuple(self.fused_host.shape) != shape:
                return None, 'capacity'
            if self.fused_host is None:
                host = torch.empty(shape,dtype=torch.int64,device='cpu',pin_memory=True)
                device = torch.empty(shape,dtype=torch.int64,device=self.device)
                self.fused_host,self.fused_device = host,device
            stream = torch.cuda.current_stream(self.device)
            self.fused_binding = (id(authority),authority.memory.request_id,id(plan),
                getattr(getattr(owner,'_restore_lease',None),'generation',None),spec.signature,
                getattr(stream,'cuda_stream',id(stream)))
            # Opaque kernel pointers do not retain tensor owners automatically.
            # This includes the from_runner page-table clones and their lease.
            self.fused_owners = (authority,plan,getattr(owner,'_restore_lease',None),*spec.owners)
            self.fused_active = True
            self.fused_host.copy_(torch.tensor(spec.rows,dtype=torch.int64,device='cpu'))
            for tensor in spec.owners:
                tensor.record_stream(stream)
            try:
                self.fused_device.copy_(self.fused_host,non_blocking=True)
                self.fused_device.record_stream(stream)
            except BaseException as error:
                try:
                    stream.synchronize()
                except BaseException as fence_error:
                    self.fused_poisoned = True
                    error.add_note('Fused metadata failure fence also failed: '+repr(fence_error))
                raise
            return self.fused_device, None

    def finish_fused_table(self, owner, event):
        with self._lock:
            if self.fused_active:
                if self.owner is not owner:
                    raise RuntimeError('Stale fused metadata completion owner')
                self.fused_event = event
                self.fused_active = False

    def fused_capacity(self):
        return (0 if self.fused_device is None else self.fused_device.numel()*8,
                0 if self.fused_host is None else self.fused_host.numel()*8)

    def destination(self, size):
        if self.owner is None or not 0 < size <= self.slot_bytes:
            raise ValueError("Assembly destination requires an owned bounded slot")
        if self.transfers is None:
            self.transfers = PackedTransfers(self.slot_bytes, self.slots, self.device)
        self.transfers.bandwidth_limiter = self.bandwidth_limiter
        if not self.buffers:
            self.buffers = [torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
                            for _ in range(self.slots)]
            allocated = torch.cuda.Event()
            allocated.record(torch.cuda.current_stream(self.device))
            self.ready = [allocated] * self.slots
        slot = self.cursor % self.slots
        self.cursor += 1
        return self.buffers[slot][:size], self.ready[slot], slot

    def consumed(self, slot):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.ready[slot] = event

    def value_destination(self,size):
        if self.owner is None or not 0<size<=self.value_capacity_bytes:
            raise ValueError('Resident V snapshot exceeds its shared byte budget')
        if self.value_buffer is None:
            self.value_buffer=torch.empty(self.value_capacity_bytes,dtype=torch.uint8,device=self.device)
        if self.value_ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self.value_ready)
        return self.value_buffer[:size]

    def release(self, owner, failed_fence=None):
        with self._lock:
            if self.owner is not owner:
                raise RuntimeError("Stale assembly staging owner")
            if self.fused_active:
                if failed_fence is None:
                    # No proven completion permits metadata or its opaque
                    # pointer sources to be recycled after a failed submit.
                    self.fused_poisoned = True
                else:
                    self.finish_fused_table(owner,failed_fence)
            if failed_fence is not None:
                # The failure fence follows every submitted DMA and partial
                # kernel, including a chunk that failed before consumed().
                self.ready = [failed_fence] * len(self.buffers)
                if self.value_buffer is not None:
                    self.value_ready=failed_fence
            self.owner = None

    def close(self):
        with self._lock:
            if (self.owner is not None or not self.poll_fused_table() or
                    any(not event.query() for event in self.ready) or
                    (self.value_ready is not None and not self.value_ready.query())):
                return False
            if self.transfers is not None and not self.transfers.close():
                return False
            self.buffers.clear()
            self.ready.clear()
            self.transfers = None
            self.value_buffer=None
            self.value_ready=None
            self.fused_host = self.fused_device = None
            return True


class NativeWindowAssembler:
    def __init__(self, layers: tuple[LayerWindow, ...], block_tokens: int, chunk_blocks=32, packed=False,
                 direct_pinned_reads=False, *, staging=None, raw_cache=None, reuse_resident_v=False,
                 value_capacity_bytes=None, lease=None, raw_dtype=torch.bfloat16, read_prefetch=False,
                 fused_layers=False):
        if not layers or min(block_tokens, chunk_blocks) <= 0:
            raise ValueError("Invalid native assembly geometry")
        device = layers[0].key.device
        heads, dim = layers[0].key.shape[-2:]
        if lease is not None:
            lease.validate(lease.runner)
        checked_tables = set()
        for layer in layers:
            if (layer.key.device != device or layer.key.shape != layer.value.shape or
                    layer.key.shape[-2:] != (heads, dim) or layer.kernel_page_table.ndim != 1 or
                    layer.kernel_page_table.device != device):
                raise ValueError("Native layers disagree on cache geometry or ownership")
            if layer.key.dtype != torch.float8_e4m3fn or layer.value.dtype != torch.float8_e4m3fn:
                raise ValueError("Native assembly requires FP8 e4m3 cache views")
            signature = (layer.kernel_page_table.data_ptr(), layer.key.shape[0])
            if signature not in checked_tables:
                if lease is not None:
                    if layer.restore_group is None:
                        raise ValueError('Restore assembly omitted its native allocation group')
                    rows = lease.group_rows(layer.restore_group, page_count=layer.key.shape[0])
                    if len(rows) != layer.kernel_page_table.numel():
                        raise ValueError('Restore assembly table differs from its allocation extent')
                elif (not layer.kernel_page_table.numel() or
                      torch.any((layer.kernel_page_table <= 0) | (layer.kernel_page_table >= layer.key.shape[0])).item()):
                    raise ValueError("Native page table contains null or out-of-arena pages")
                checked_tables.add(signature)
        self.layers = layers
        self.block_tokens = block_tokens
        self.chunk_blocks = chunk_blocks
        self.packed = packed
        if direct_pinned_reads and not packed:
            raise ValueError("Direct pinned reads require packed assembly")
        self.direct_pinned_reads = direct_pinned_reads
        if (raw_cache is not None or reuse_resident_v) and not direct_pinned_reads:
            raise ValueError('Split raw/V assembly requires direct packed reads')
        self.raw_cache, self.reuse_resident_v = raw_cache, reuse_resident_v
        self.read_prefetch = read_prefetch
        self.read_prefetch_stats = {}
        self.fused_layers = fused_layers
        self.fused_stats = {}
        self.device = device
        self.heads, self.dim = heads, dim
        self.raw_dtype = raw_dtype
        raw_element_bytes = raw_dtype.itemsize
        slot_bytes = len(layers) * heads * dim * block_tokens * chunk_blocks * (raw_element_bytes + 1)
        if staging is not None and (staging.slot_bytes != slot_bytes or staging.device != device):
            raise ValueError("Shared assembly staging geometry changed")
        self._shared_staging = staging is not None
        if value_capacity_bytes is None:
            value_capacity_bytes=(len(layers)*heads*dim*min(l.kernel_page_table.numel()*l.value.shape[1]
                for l in layers)) if reuse_resident_v else 0
        if staging is not None and staging.value_capacity_bytes!=value_capacity_bytes:
            raise ValueError('Shared resident V staging geometry changed')
        self.staging = staging if staging is not None else _AssemblyStaging(slot_bytes, device,value_capacity_bytes)
        self._metadata_owners = []
        self._completion = None
        self.closed = False
        self._restore_lease = lease

    @classmethod
    def from_runner(cls, runner, request_id, layer_names, block_tokens=128, chunk_blocks=None, packed=False,
                    direct_pinned_reads=False, *, arena='main', lease=None):
        if arena not in ('main', 'refresh'):
            raise ValueError('Unknown assembly arena')
        options = getattr(getattr(runner, '_kvmem_capture_runtime', None), 'options', None)
        if chunk_blocks is None:
            chunk_blocks = getattr(options, 'assembly_chunk_blocks', 32)
        if type(chunk_blocks) is not int or not 1 <= chunk_blocks <= 256:
            raise ValueError('Assembly chunk blocks must be an integer in [1, 256]')
        if lease is not None:
            lease.validate(runner)
            if lease.request_id != request_id:
                raise ValueError('Assembly restore lease belongs to another request')
        modules = dict(runner.model.named_modules())
        modules.update(getattr(runner, '_kvmem_mtp_modules', {}))
        groups = {name: group_id for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                  for name in group.layer_names}
        index = runner.req_states.req_id_to_index[request_id]
        tables = {}
        layers = []
        for name in layer_names:
            module = modules[name]
            attn = module.attn
            if attn.kv_cache_dtype != "fp8_e4m3":
                raise ValueError("Native assembly currently qualifies FP8 e4m3")
            group_id = groups[attn.layer_name]
            if group_id not in tables:
                from .native_pages import kernel_table
                tables[group_id] = kernel_table(runner, group_id, index).clone()
            key, value = native_vllm_fp8_views(attn.kv_cache, module.num_kv_heads, module.head_dim)
            layers.append(LayerWindow(key, value, module.rotary_emb.cos_sin_cache,
                                      attn._k_scale, tables[group_id], group_id if lease is not None else None))
        staging_name = '_kvmem_assembly_staging' if arena == 'main' else '_kvmem_refresh_assembly_staging'
        cache_name = '_kvmem_gpu_raw_cache' if arena == 'main' else '_kvmem_refresh_gpu_raw_cache'
        staging = getattr(runner, staging_name, None)
        raw_cache = getattr(runner, cache_name, None)
        if direct_pinned_reads and options is not None and options.gpu_raw_k_cache_bytes and raw_cache is None:
            from .gpu_raw_cache import GpuRawCache
            # CPU background assembly and foreground checkpoint import have
            # separate scratch/index ownership, within the same raw-K budget.
            budget = options.gpu_raw_k_cache_bytes // (2 if getattr(options,'async_refresh',False) else 1)
            raw_cache = GpuRawCache(budget,
                (len(layers),block_tokens,module.num_kv_heads,module.head_dim), key.device,
                runner.max_num_reqs, options.memory.select_budget//block_tokens,
                dtype=torch.float8_e4m3fn)
            setattr(runner, cache_name, raw_cache)
        assembler = cls(tuple(layers), block_tokens, chunk_blocks, packed=packed,
                        direct_pinned_reads=direct_pinned_reads, staging=staging,
                        raw_cache=raw_cache if direct_pinned_reads else None,
                        reuse_resident_v=bool(direct_pinned_reads and options is not None and options.reuse_resident_v),
                        value_capacity_bytes=(len(layers)*module.num_kv_heads*module.head_dim*options.memory.select_budget
                            if options is not None and options.reuse_resident_v else 0), lease=lease,
                        raw_dtype=torch.float8_e4m3fn,
                        read_prefetch=getattr(options, 'assembly_read_prefetch', False),
                        fused_layers=getattr(options, 'assembly_fused_layers', False))
        if staging is None:
            # Publish shared scratch only after all geometry/ownership checks
            # pass; an invalid first request must not poison the runner pool.
            setattr(runner, staging_name, assembler.staging)
            assembler._shared_staging = True
        return assembler

    @property
    def transfers(self):
        return self.staging.transfers

    def assemble(self, authority, plan):
        if self.closed:
            raise RuntimeError("Native assembler is closed")
        if self._restore_lease is not None:
            self._restore_lease.validate(self._restore_lease.runner)
        self.staging.claim(self)
        self.staging.bandwidth_limiter = getattr(self, 'bandwidth_limiter', None)
        self.read_prefetch_stats = {f'assembly_prefetch_{name}': 0 for name in
            ('submitted', 'completed', 'abandoned', 'slot_busy', 'admission_busy', 'eligible', 'declined')}
        self.fused_stats = {f'assembly_fused_{name}': 0 for name in
                            ('chunks','layer_launches','fallback_chunks')}
        failed = None
        try:
            result = self._assemble(authority, plan)
            self.staging.finish_fused_table(self,result.event)
            if self.read_prefetch:
                result.read_stats.update(self.read_prefetch_stats)
            if self.fused_layers:
                gpu,pinned = self.staging.fused_capacity()
                self.fused_stats.update(assembly_fused_metadata_bytes=gpu+pinned,
                    assembly_fused_metadata_gpu_bytes=gpu,assembly_fused_metadata_pinned_bytes=pinned)
                result.read_stats.update(self.fused_stats)
                result.device_temp_bytes += gpu
            self._completion = result.event
            return result
        except BaseException as error:
            if authority.memory.phase is Phase.ASSEMBLING and authority.memory._plan is plan:
                # Even an I/O error after an earlier chunk must not permit the
                # half-rebuilt cache to commit, or staging/leases to be recycled
                # while submitted copies and kernels still use them.
                try:
                    stream = torch.cuda.current_stream(self.device)
                    if self.transfers is not None:
                        for transfer in self.transfers.pending.values():
                            stream.wait_event(transfer.event)
                    candidate = torch.cuda.Event()
                    candidate.record(stream)
                    failed = candidate
                except BaseException as fence_error:
                    # An unrecorded CUDA event may report ready. It cannot
                    # release opaque pointers, destination pages or scratch.
                    failed = _UnfencedAssembly()
                    self.staging.fused_poisoned = True
                    error.add_note('Assembly failure fence also failed: '+repr(fence_error))
                self._completion = failed
                authority.memory.fail_transaction(AssemblyResult(
                    failed, (self, plan, authority), 0, 0, 0))
            raise
        finally:
            # CPU record bytes have now been copied into completion-owned
            # staging. Later GPU assembly/compute does not read the archive.
            release_sources = getattr(plan, 'release_cpu_sources', None)
            if release_sources is not None:
                release_sources()
            self.staging.release(self, failed)
            if failed is not None and self.raw_cache is not None:
                self.raw_cache.drop_namespace(authority.memory.request_id)
                self.raw_cache.finish(failed)

    def _upload_metadata(self, values):
        # A pageable torch.tensor(..., device="cuda") after wait_event can
        # block the CPU behind the preceding large KV copy. Submit small pinned
        # metadata once before the chunk pipeline, and keep both ends fenced.
        host = torch.tensor(values, device="cpu", dtype=torch.int64, pin_memory=True)
        self._metadata_owners.append(host)
        device = host.to(self.device, non_blocking=True)
        self._metadata_owners.append(device)
        return device

    def _assemble(self, authority, plan):
        if authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
            raise ValueError("Native assembly requires an active original-core selection")
        if authority.shape != (len(self.layers), self.block_tokens, self.heads, self.dim):
            raise ValueError("Authority codec does not match native working pages")
        if (authority.raw_dtype != self.raw_dtype or
                (self.raw_cache is not None and self.raw_cache.dtype != authority.raw_dtype)):
            raise ValueError('Immutable raw-K codec differs from assembly/cache storage')
        by_slot = {record.handle.slot: record for record in authority.records.values()}
        remaps = list(plan.remaps)
        # Validate the entire destination before changing any GPU cache bytes.
        _validate_destinations(remaps, by_slot, self.layers)
        pending = [remap for remap in remaps if not remap.skip]
        if (pending and (self.raw_cache is not None or self.reuse_resident_v) and
                authority.raw_dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)):
            complete = [r for r in pending if r.n_tokens == self.block_tokens]
            partial = [r for r in pending if r.n_tokens != self.block_tokens]
            if complete:
                # Snapshot every resident full-page V source before any writes.
                # Partial records read CPU authority, so they can safely follow
                # full-page moves even if destinations overlap old sources.
                full = self._assemble_split(authority,plan,remaps,complete,by_slot)
                if not partial:
                    return full
                tail = self._assemble_generic(authority,plan,[],partial,by_slot)
                stats = dict(full.read_stats)
                for key, value in tail.read_stats.items():
                    stats[key] = stats.get(key, 0) + value
                # The final fence/owners cover BOTH subpaths. Do not publish a
                # checkpoint or recycle metadata after only the full-page DMA.
                return AssemblyResult(tail.event, (full, tail, *full.owners, *tail.owners),
                    full.rebuilt_tokens + tail.rebuilt_tokens, full.skipped_tokens,
                    full.chunks + tail.chunks,
                    full.host_prepare_seconds + tail.host_prepare_seconds,
                    full.uploaded_bytes + tail.uploaded_bytes, stats,
                    full.raw_cache_hit_tokens, full.resident_v_reused_tokens,
                    full.raw_cache_capacity_bytes, full.device_temp_bytes,
                    full.split_tokens, tail.rebuilt_tokens)
        return self._assemble_generic(authority,plan,remaps,pending,by_slot)

    def _assemble_generic(self, authority, plan, remaps, pending, by_slot):
        if self.read_prefetch:
            self.read_prefetch_stats['assembly_prefetch_declined'] += 1
        if self.fused_layers:
            self._count_fused_fallback('generic',(len(pending)+self.chunk_blocks-1)//self.chunk_blocks)
        rebuilt, chunks, owners = 0, 0, [self, plan, authority]
        host_prepare_seconds, uploaded_bytes = 0.0, 0
        read_stats = {}
        metadata_start = time.perf_counter()
        bases = self._upload_metadata([remap.to_base for remap in pending]) if pending and self.packed else None
        positions = None
        if pending and (not self.packed or any(r.n_tokens != self.block_tokens for r in pending)):
            positions = self._upload_metadata([position for remap in pending
                for position in range(remap.to_base, remap.to_base + remap.n_tokens)])
        host_prepare_seconds += time.perf_counter() - metadata_start
        position_begin = 0
        for start in range(0, len(pending), self.chunk_blocks):
            chunk = pending[start:start+self.chunk_blocks]
            handles = [by_slot[remap.block_id].handle for remap in chunk]
            prepare_start = time.perf_counter()
            transfer = None
            use_packed = self.packed and all(r.n_tokens == self.block_tokens for r in chunk)
            use_direct = self.direct_pinned_reads and use_packed
            if use_direct:
                from .packed_assembly import read_packed_records
                if any(remap.n_tokens != self.block_tokens for remap in chunk):
                    raise ValueError("Packed window assembly requires complete selected records")
                size = len(handles) * authority.record_bytes
                gpu, ready, slot = self.staging.destination(size)
                def fill(staging):
                    _, stats = read_packed_records(authority, handles, destination=staging)
                    for key, value in stats.items():
                        read_stats[key] = read_stats.get(key, 0) + value
                transfer = self.transfers.upload_from_reader(size, gpu, fill, ready=ready)
                if transfer is None:
                    next(iter(self.transfers.pending.values())).event.synchronize()
                    transfer = self.transfers.upload_from_reader(size, gpu, fill, ready=ready)
                count = len(chunk) * self.block_tokens
            elif use_packed:
                from .packed_assembly import read_packed_records
                if any(remap.n_tokens != self.block_tokens for remap in chunk):
                    raise ValueError("Packed window assembly requires complete selected records")
                payload, stats = read_packed_records(authority, handles)
                for key, value in stats.items():
                    read_stats[key] = read_stats.get(key, 0) + value
                count = len(chunk) * self.block_tokens
            else:
                payloads = authority.read_records(handles)
                raw = torch.cat([raw for raw, _ in payloads], dim=1).contiguous()
                values = torch.cat([value for _, value in payloads], dim=1).contiguous()
                payload = torch.cat((raw.view(torch.uint8).flatten(), values.flatten()))
                count = raw.shape[1]
            uploaded_bytes += size if use_direct else payload.numel()
            if not use_direct:
                gpu, ready, slot = self.staging.destination(payload.numel())
                transfer = self.transfers.upload(payload, gpu, ready=ready)
                if transfer is None:
                    # Only bounded staging is reused, after its own DMA completes.
                    next(iter(self.transfers.pending.values())).event.synchronize()
                    transfer = self.transfers.upload(payload, gpu, ready=ready)
            if transfer is None:
                raise RuntimeError("Native assembly staging did not make progress")
            # Includes CPU staging and waiting for a bounded slot, in both paths.
            host_prepare_seconds += time.perf_counter() - prepare_start
            torch.cuda.current_stream(self.device).wait_event(transfer.event)
            if use_packed:
                from .packed_assembly import assemble_packed_layer
                chunk_bases = bases[start:start+len(chunk)]
                for i, layer in enumerate(self.layers):
                    assemble_packed_layer(gpu, chunk_bases, layer, layer_index=i, layers=len(self.layers),
                                          block_tokens=self.block_tokens, raw_dtype=authority.raw_dtype)
            else:
                elements = raw.numel()
                raw_size = elements * authority.raw_element_bytes
                raw_gpu = gpu[:raw_size].view(authority.raw_dtype).reshape(raw.shape)
                values_gpu = gpu[raw_size:].view(torch.float8_e4m3fn).reshape(values.shape)
                chunk_positions = positions[position_begin:position_begin+count]
                for i, layer in enumerate(self.layers):
                    page_tokens = layer.key.shape[1]
                    slots = layer.kernel_page_table[chunk_positions // page_tokens].to(torch.int64) * page_tokens + chunk_positions % page_tokens
                    assemble_fp8(raw_gpu[i], values_gpu[i], chunk_positions, slots, layer.cos_sin,
                                 layer.key, layer.value, layer.key_scale)
            self.staging.consumed(slot)
            position_begin += count
            rebuilt += count
            chunks += 1
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        return AssemblyResult(done, tuple(owners), rebuilt,
                              sum(remap.n_tokens for remap in remaps if remap.skip), chunks,
                              host_prepare_seconds, uploaded_bytes, read_stats,
                              partial_tokens=sum(r.n_tokens for r in pending if r.n_tokens != self.block_tokens))

    def _count_fused_fallback(self, reason, chunks=1):
        self.fused_stats['assembly_fused_fallback_chunks'] += chunks
        name = 'assembly_fused_fallback:'+reason
        self.fused_stats[name] = self.fused_stats.get(name,0)+chunks

    def _prepare_fused_layers(self,authority,plan,remaps):
        if not getattr(self,'fused_layers',False):
            return None,None
        from .memory_storage import MemoryArchive
        from .split_assembly import prepare_fused_layer_rows
        if not self.direct_pinned_reads or self.raw_cache is not None or not isinstance(authority.archive,MemoryArchive):
            return None,'path'
        if any(remap.n_tokens != self.block_tokens for remap in remaps):
            return None,'partial'
        spec,reason = prepare_fused_layer_rows(self.layers)
        if spec is None:
            return None,reason
        table,reason = self.staging.prepare_fused_table(self,authority,plan,spec)
        return (None,reason) if table is None else ((table,spec),None)

    def _assemble_split(self,authority,plan,remaps,pending,by_slot):
        from .memory_storage import MemoryArchive
        from .packed_assembly import PackedReadAhead
        fused,fused_reason = self._prepare_fused_layers(authority,plan,remaps)
        kwargs = dict(fused=fused,fused_reason=fused_reason) if getattr(self,'fused_layers',False) else {}
        eligible = (self.read_prefetch and self.direct_pinned_reads and self.raw_cache is None and
                    isinstance(authority.archive, MemoryArchive))
        if not eligible:
            if self.read_prefetch:
                self.read_prefetch_stats['assembly_prefetch_declined'] += 1
            return self._assemble_split_chunks(authority,plan,remaps,pending,by_slot,None,**kwargs)
        self.read_prefetch_stats['assembly_prefetch_eligible'] += 1
        # The context drains an admitted writer even when a later layer fails,
        # before assemble() fails the transaction or releases plan CPU sources.
        with PackedReadAhead(authority, self.read_prefetch_stats) as ahead:
            return self._assemble_split_chunks(authority,plan,remaps,pending,by_slot,ahead,**kwargs)

    def _assemble_split_chunks(self,authority,plan,remaps,pending,by_slot,ahead,*,fused=None,fused_reason=None):
        from .packed_assembly import read_packed_records
        from .split_assembly import gather_resident_values, assemble_split_layer, assemble_split_layers, store_raw_records
        layers=len(self.layers)
        raw_bytes=authority.raw_record_bytes
        value_bytes=authority.value_record_bytes
        owners=[self,plan,authority]
        started=time.perf_counter()
        namespace=authority.memory.request_id
        keys=[(namespace,int(by_slot[r.block_id].handle.slot),int(by_slot[r.block_id].handle.generation))
              for r in pending]
        hits=[None]*len(pending)
        if self.raw_cache is not None:
            self.raw_cache.begin()
            hits=[self.raw_cache.index.get(key) for key in keys]
        resident=[i for i,r in enumerate(pending) if self.reuse_resident_v and r.working_k_resident]
        _validate_resident_sources((pending[i] for i in resident), self.layers)
        resident_index={i:j for j,i in enumerate(resident)}
        resident_values=None
        if resident:
            from_bases=self._upload_metadata([pending[i].from_base for i in resident])
            # Stage the whole read set before any writes; per-chunk staging
            # would corrupt cycles crossing a chunk boundary.
            size=len(resident)*value_bytes
            destination=self.staging.value_destination(size)
            resident_values=gather_resident_values(self.layers,from_bases,self.block_tokens,destination)
            owners.append(resident_values)
        host_seconds=time.perf_counter()-started
        uploaded,rebuilt,chunks=0,0,0
        stats={}
        groups={(h,v):[] for h,v in ((True,True),(True,False),(False,True),(False,False))}
        for i in range(len(pending)):
            groups[hits[i] is not None,i in resident_index].append(i)
        # Every cache hit consumes its old slot before any miss may evict it.
        for (hit,reuse),indices in groups.items():
            for begin in range(0,len(indices),self.chunk_blocks):
                chunk=indices[begin:begin+self.chunk_blocks]
                clock=time.perf_counter()
                bases=self._upload_metadata([pending[i].to_base for i in chunk])
                raw_indices=self._upload_metadata([hits[i] if hit else j for j,i in enumerate(chunk)])
                value_indices=self._upload_metadata([resident_index[i] if reuse else j for j,i in enumerate(chunk)])
                gpu=None
                slot=None
                stride=raw_bytes
                if not hit or not reuse:
                    region=((raw_bytes,value_bytes) if hit else (0,raw_bytes) if reuse else None)
                    stride=authority.record_bytes if region is None else region[1]
                    size=len(chunk)*stride
                    gpu,ready,slot=self.staging.destination(size)
                    handles=[by_slot[pending[i].block_id].handle for i in chunk]
                    def fill(buffer):
                        _,read=read_packed_records(authority,handles,destination=buffer,region=region)
                        for name,value in read.items():
                            stats[name]=stats.get(name,0)+value
                    prepared = (ahead.take(self.transfers,handles,region,gpu,ready)
                                if ahead is not None else None)
                    if prepared is not None:
                        transfer, read = prepared
                        for name,value in read.items():
                            stats[name]=stats.get(name,0)+value
                    else:
                        transfer=self.transfers.upload_from_reader(size,gpu,fill,ready=ready)
                    if transfer is None:
                        next(iter(self.transfers.pending.values())).event.synchronize()
                        transfer=self.transfers.upload_from_reader(size,gpu,fill,ready=ready)
                    if transfer is None:
                        raise RuntimeError('Split assembly staging did not make progress')
                    torch.cuda.current_stream(self.device).wait_event(transfer.event)
                    uploaded+=size
                    if ahead is not None:
                        next_chunk = indices[begin+self.chunk_blocks:begin+2*self.chunk_blocks]
                        if next_chunk:
                            next_handles = [by_slot[pending[i].block_id].handle for i in next_chunk]
                            # Try-only CPU admission precedes the current layer
                            # launches. It never waits for a slot or another
                            # owner's I/O, and never crosses a hit/reuse group.
                            ahead.start(self.transfers,next_handles,region,len(next_chunk)*stride)
                raw_source=self.raw_cache.tensor if hit else gpu
                values_source=resident_values if reuse else gpu
                raw_stride=raw_bytes if hit else stride
                value_stride=value_bytes if reuse or hit else stride
                value_offset=0 if reuse or hit else raw_bytes
                if not hit and self.raw_cache is not None:
                    protected=set()
                    cache_slots=[]
                    for i in chunk:
                        cached_slot=self.raw_cache.index.reserve(keys[i],protected)
                        cache_slots.append(-1 if cached_slot is None else cached_slot)
                        if cached_slot is not None:
                            protected.add(keys[i])
                    slots=self._upload_metadata(cache_slots)
                    store_raw_records(gpu,slots,self.raw_cache.tensor,stride)
                if fused is not None:
                    table,spec = fused
                    assemble_split_layers(raw_source,raw_indices,values_source,value_indices,bases,table,spec.geometry,
                        layers=layers,block_tokens=self.block_tokens,
                        raw_record_bytes=raw_stride,value_record_bytes=value_stride,value_offset=value_offset,
                        raw_dtype=authority.raw_dtype)
                    self.fused_stats['assembly_fused_chunks'] += 1
                    self.fused_stats['assembly_fused_layer_launches'] += layers
                else:
                    if getattr(self,'fused_layers',False):
                        self._count_fused_fallback(fused_reason or 'path')
                    for i,layer in enumerate(self.layers):
                        assemble_split_layer(raw_source,raw_indices,values_source,value_indices,bases,layer,
                            layer_index=i,layers=layers,block_tokens=self.block_tokens,
                            raw_record_bytes=raw_stride,value_record_bytes=value_stride,value_offset=value_offset,
                            raw_dtype=authority.raw_dtype)
                if slot is not None:
                    self.staging.consumed(slot)
                host_seconds+=time.perf_counter()-clock
                rebuilt+=len(chunk)*self.block_tokens
                chunks+=1
        done=torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        if resident_values is not None:
            self.staging.value_ready=done
        if self.raw_cache is not None:
            self.raw_cache.finish(done)
            owners.append(self.raw_cache)
        return AssemblyResult(done,tuple(owners),rebuilt,
            sum(r.n_tokens for r in remaps if r.skip),chunks,host_seconds,uploaded,stats,
            sum(hit is not None for hit in hits)*self.block_tokens,len(resident)*self.block_tokens,
            self.raw_cache.capacity_bytes if self.raw_cache is not None else 0,
            resident_values.numel() if resident_values is not None else 0, rebuilt)

    def close(self):
        if self.staging.owner is self:
            return False
        if self._completion is not None and not self._completion.query():
            return False
        if not self.staging.poll_fused_table():
            return False
        if not self._shared_staging and not self.staging.close():
            return False
        self._metadata_owners.clear()
        self.closed = True
        return True
