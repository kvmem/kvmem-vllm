"""Portable GDN and native DFlash context for sealed prompt checkpoints.

Only quiescent, non-final prefill boundaries are supported. Target attention KV
is reconstructed separately from WindowCheckpoint's immutable authority. Capture
owns a packed GPU snapshot until its bounded pinned CPU payload is ready. The
session owner must fence publication/retirement and verify exact prompt identity;
restore holds destination pages until its own completion before admitting work.
"""
from dataclasses import dataclass

import torch

from .native_state import NativeRecurrentState
from .checkpoint_timing import async_checkpoint_capture, checkpoint_stage, checkpoint_trace


@dataclass(frozen=True)
class PromptModelState:
    owner: object
    position: int
    draft_begin: int
    recurrent: dict[tuple[str, str], torch.Tensor]
    draft: dict[str, torch.Tensor]
    payload: torch.Tensor | None = None
    completion: object = None
    source_owners: tuple = ()

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (*self.recurrent.values(), *self.draft.values()))

    def query(self):
        completion = self.completion
        if completion is not None and not completion.query():
            return False
        # Only the independent packed snapshot is retained here. Releasing it
        # after D2H completion cannot expose a borrowed native state row.
        if self.source_owners:
            object.__setattr__(self, 'source_owners', ())
        if completion is not None:
            object.__setattr__(self, 'completion', None)
        return True


@dataclass
class PreparedModelRestore:
    snapshot: PromptModelState
    request_id: str
    index: int
    generation: int
    recurrent: dict
    draft: dict
    counter: torch.Tensor
    lease: object = None
    cached_payload: object = None


@dataclass
class ModelRestoreCompletion:
    event: object
    owners: tuple

    def query(self):
        return self.event.query()

    def synchronize(self):
        self.event.synchronize()


class PromptStateAdapter:
    def __init__(self, runner):
        from .draft_contract import DraftContract
        from .options import RuntimeOptions
        self.runner = runner
        self.recurrent = NativeRecurrentState(runner)
        self.contract = None
        if runner.speculative_config is not None:
            self.contract = DraftContract.from_vllm(runner.vllm_config,
                RuntimeOptions.from_vllm(runner.vllm_config).memory)
            self.contract.validate_native_groups(runner)
        self._identity = object()  # Snapshots never cross workers/model loads.
        options = RuntimeOptions.from_vllm(runner.vllm_config)
        self.capture_decode = options.session_capture_decode or options.session_checkpoint_policy=='pm'
        from .model_snapshot_cache import ModelSnapshotCache
        self.snapshot_cache = ModelSnapshotCache(options.session_gpu_checkpoint_bytes)

    def _prefill_index(self, request_id, position, *, terminal=False):
        index = self.runner.req_states.req_id_to_index[request_id]
        states = self.runner.req_states
        if (type(position) is not int or position <= 0 or
                (not terminal and (int(states.num_computed_tokens_np[index]) != position or
                    (int(states.prefill_len.np[index]) <= position and not getattr(self, 'capture_decode', False))))):
            raise ValueError('Model checkpoint requires the exact non-final prefill boundary')
        return index

    def _recurrent_views(self, request_id, *, lease=None):
        # borrow validates generation, null pages and the native state layout.
        if lease is not None:
            lease.validate(self.runner)
            if lease.request_id != request_id:
                raise ValueError('Recurrent restore lease belongs to another request')
        bundle = self.recurrent.borrow([request_id]) if lease is None else None
        result = {}
        for gid, layers in self.recurrent.groups.items():
            rows = bundle.page_rows[request_id, gid] if bundle is not None else lease.group_rows(gid)
            for name, (conv, ssm) in layers.items():
                if max(rows) >= min(conv.shape[0], ssm.shape[0]):
                    raise ValueError('Recurrent restore page exceeds its native cache arena')
                result[name, 'conv'] = conv[rows[0]]
                result[name, 'ssm'] = ssm[rows[0]]
        return result

    def _draft_views(self, request_id, begin, end):
        if self.contract is not None and self.contract.method == 'mtp':
            if begin != end:
                raise ValueError('MTP working KV belongs to the immutable window checkpoint')
            # MTP raw-K and FP8-V are part of target window assembly. Native
            # autoregressive proposal overwrites its temporary hidden inputs
            # with the next target batch; there is no extra persistent state.
            self.contract.validate_native_groups(self.runner)
            return {}
        if self.contract is None:
            if begin != end:
                raise ValueError('Target-only snapshot cannot contain draft history')
            return {}
        groups = self.contract.validate_native_groups(self.runner)
        index = self.runner.req_states.req_id_to_index[request_id]
        modules = {getattr(m, 'layer_name', None): m
                   for m in self.runner.speculator.model.modules()
                   if getattr(m, 'layer_name', None) is not None}
        result = {}
        for gid in groups:
            from .native_pages import kernel_block_size, kernel_table
            block_size = kernel_block_size(self.runner, gid)
            table = kernel_table(self.runner, gid, index)
            count = len(table)
            first, stop = begin // block_size, (end + block_size - 1) // block_size
            if not 0 <= first < stop <= count:
                raise ValueError('Draft checkpoint context is outside its native page table')
            with checkpoint_stage(self.runner, 'draft_page_table_gpu_read', sync=True):
                pages = table[first:stop].cpu().tolist()
            if any(p <= 0 for p in pages) or len(set(pages)) != len(pages):
                raise ValueError('Draft checkpoint has null or aliased context pages')
            logical = torch.arange(begin, end, device=table.device, dtype=torch.int64)
            physical, offset = table[logical // block_size].long(), logical % block_size
            for name in self.runner.kv_cache_config.kv_cache_groups[gid].layer_names:
                layer = modules.get(name)
                cache = getattr(layer, 'kv_cache', None)
                if (cache is None or cache.dtype != torch.bfloat16 or cache.ndim != 4 or
                        cache.shape[1] != layer.num_kv_heads or
                        cache.shape[2] != block_size or cache.shape[3] != 2 * layer.head_size or
                        cache.stride(-1) != 1 or cache.device != table.device or
                        max(pages) >= cache.shape[0]):
                    raise ValueError('Unqualified native draft BHN(2D) cache view')
                result[name] = (cache.transpose(1, 2), physical, offset)
        return result

    def capture(self, request_id, position, *, max_bytes, terminal=False):
        optimized = async_checkpoint_capture(self.runner)
        with checkpoint_trace(self.runner, request_id, position, terminal, optimized) as trace:
            result = self._capture(request_id, position, max_bytes=max_bytes,
                                   terminal=terminal, optimized=optimized)
            trace['snapshot_bytes'] = result.bytes
            return result

    def _capture(self, request_id, position, *, max_bytes, terminal, optimized):
        index = self._prefill_index(request_id, position, terminal=terminal)
        counter = self.runner.model_state.num_accepted_tokens_gpu
        canonicalize = terminal
        if not terminal:
            # This value also detects a prior canonicalization of the same
            # accepted step. Keep it until a qualified state-version witness
            # covers Query save/restore and all native counter writers.
            with checkpoint_stage(self.runner, 'accepted_counter_read', sync=True):
                canonicalize = int(counter[index].item()) != 1
        if canonicalize:
            if not terminal and not getattr(self, 'capture_decode', False):
                raise ValueError('Speculative terminal state requires canonicalization or replay')
            runtime = self.runner._kvmem_capture_runtime
            steps = runtime.last_steps.get(request_id, [])
            from .publication_step import native_publication_step
            step = native_publication_step(steps[-1]) if steps else None
            if step is None or step['history_begin'] + step['accepted'] != position:
                raise ValueError('Decode snapshot is missing its exact accepted sampling extent')
            if terminal and runtime.authorities[request_id].history_end != position:
                raise ValueError('Terminal state differs from the accepted KV authority')
            with checkpoint_stage(self.runner, 'canonicalize_submit'):
                event = self.recurrent.canonicalize_accepted(request_id,
                    retained_tokens=step['accepted'], query_tokens=step['scheduled'],
                    # Native speculative padding can follow a one-token prompt
                    # tail even when that first row is still public prefill.
                    speculative=step['history_begin'] + step['scheduled'] >
                        int(self.runner.req_states.prefill_len.np[index]))
            if event is not None and not optimized:
                with checkpoint_stage(self.runner, 'canonicalize_wait', sync=True):
                    event.synchronize()
            # Optimized capture immediately snapshots on the same stream.
            # Its packed clone and D2H completion preserve every dependency;
            # a host wait before those enqueues provides no additional safety.
        with checkpoint_stage(self.runner, 'recurrent_views'):
            recurrent = self._recurrent_views(request_id)
        begin = (max(0, position - self.contract.sliding_window + 1)
                 if self.contract and self.contract.method == 'dflash' else position)
        with checkpoint_stage(self.runner, 'draft_views'):
            draft = self._draft_views(request_id, begin, position)
        size = sum(t.numel() * t.element_size() for t in recurrent.values())
        size += sum((position - begin) * cache.shape[2] * cache.shape[3] * cache.element_size()
                    for cache, _, _ in draft.values())
        if type(max_bytes) is not int or max_bytes <= 0 or size > max_bytes:
            raise BufferError('Prompt model checkpoint exceeds its byte reservation')
        # The packed GPU allocation snapshots borrowed GDN/draft views before
        # the next target forward on this stream can overwrite them. CPU readers
        # and request retirement separately wait for the returned payload fence.
        tensors = list(recurrent.values()) + [cache[pages, offsets] for cache, pages, offsets in draft.values()]
        # One transfer/synchronization for all GDN layers and draft pages. Each
        # typed view owns a disjoint range of this CPU checkpoint buffer.
        with checkpoint_stage(self.runner, 'packed_snapshot'):
            packed = torch.cat([t.contiguous().view(torch.uint8).flatten() for t in tensors])
        completion = None
        if packed.device.type == 'cuda':
            # A persistent checkpoint owns this bounded pinned buffer. Pageable
            # snapshots otherwise add a host staging copy at every sealed page.
            with checkpoint_stage(self.runner, 'pinned_payload_allocation'):
                payload = torch.empty(packed.numel(),dtype=torch.uint8,device='cpu',pin_memory=True)
            completion = torch.cuda.Event()
        else:
            payload = packed.to('cpu',copy=True)
        copies, offset = [], 0
        for tensor in tensors:
            nbytes = tensor.numel() * tensor.element_size()
            copies.append(payload[offset:offset+nbytes].view(tensor.dtype).reshape(tensor.shape))
            offset += nbytes
        saved_recurrent = dict(zip(recurrent, copies[:len(recurrent)]))
        saved_draft = dict(zip(draft, copies[len(recurrent):]))
        snapshot = PromptModelState(self._identity, position, begin, saved_recurrent, saved_draft, payload,
                                    completion, (packed,) if completion is not None else ())
        if snapshot.bytes != size:
            raise RuntimeError('Checkpoint byte reservation disagrees with materialized state')
        if completion is not None:
            stream = torch.cuda.current_stream(packed.device)
            try:
                with checkpoint_stage(self.runner, 'payload_d2h_submit'):
                    payload.copy_(packed, non_blocking=True)
                    completion.record(stream)
            except BaseException:
                # An unsuccessful submission must not drop either DMA buffer.
                # This wait is failure-only; successful capture just publishes
                # its event and keeps the source allocation in the snapshot.
                with checkpoint_stage(self.runner, 'failed_submission_wait', sync=True):
                    stream.synchronize()
                raise
        cache = getattr(self, 'snapshot_cache', None)
        if terminal and cache is not None and packed.device.type == 'cuda':
            # Reuse the independent clone already required for D2H. Never
            # retain borrowed native rows or skip durable CPU publication.
            cache.put(snapshot, packed)
        return snapshot

    def restore(self, snapshot, request_id):
        return self.submit_restore(self.prepare_restore(snapshot, request_id))

    def prepare_restore(self, snapshot, request_id, *, lease=None):
        """Engine thread validates/borrows destinations before background DMA."""
        if not isinstance(snapshot, PromptModelState) or snapshot.owner is not self._identity:
            raise ValueError('Prompt state belongs to another worker/model load')
        if not snapshot.query():
            raise RuntimeError('Prompt model checkpoint payload is not ready')
        if lease is not None and self.contract is not None and self.contract.method != 'mtp':
            raise ValueError('CPU allocation restore currently supports target-only or native MTP state')
        index = self._prefill_index(request_id, snapshot.position)
        recurrent = self._recurrent_views(request_id, lease=lease) if lease is not None else self._recurrent_views(request_id)
        expected_begin = (max(0, snapshot.position - self.contract.sliding_window + 1)
                          if self.contract and self.contract.method == 'dflash' else snapshot.position)
        if snapshot.draft_begin != expected_begin:
            raise ValueError('Checkpoint draft context differs from the model window')
        draft = self._draft_views(request_id, snapshot.draft_begin, snapshot.position)
        if recurrent.keys() != snapshot.recurrent.keys() or draft.keys() != snapshot.draft.keys():
            raise ValueError('Incomplete prompt model checkpoint')
        # Validate every tensor before the first native state/page write.
        for name, destination in recurrent.items():
            source = snapshot.recurrent[name]
            if source.device.type != 'cpu' or source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError('Recurrent checkpoint schema changed')
        for name, (cache, _, _) in draft.items():
            source = snapshot.draft[name]
            shape = (snapshot.position - snapshot.draft_begin, cache.shape[2], cache.shape[3])
            if source.device.type != 'cpu' or tuple(source.shape) != shape or source.dtype != cache.dtype:
                raise ValueError('Draft checkpoint schema changed')
        cache = getattr(self, 'snapshot_cache', None)
        cached = cache.get(snapshot) if cache is not None else None
        if cached is not None and (cached.device != self.runner.model_state.num_accepted_tokens_gpu.device
                                   or cached.numel() != snapshot.bytes):
            raise ValueError('Cached model payload device or size changed')
        return PreparedModelRestore(snapshot, request_id, index,
            self.runner._kvmem_request_generations[request_id], recurrent, draft,
            self.runner.model_state.num_accepted_tokens_gpu[index:index+1], lease, cached)

    def validate_restore(self, prepared):
        if (self.runner._kvmem_request_generations.get(prepared.request_id) != prepared.generation or
                self.runner.req_states.req_id_to_index.get(prepared.request_id) != prepared.index):
            raise ValueError('Restore destination request was recycled before completion')
        if prepared.lease is not None:
            prepared.lease.validate(self.runner)

    def submit_restore(self, prepared, *, initialize_counter=True):
        """Uses only borrowed destinations; safe on the sole maintenance stream."""
        self.validate_restore(prepared)
        snapshot, recurrent, draft = prepared.snapshot, prepared.recurrent, prepared.draft
        if not snapshot.query():
            raise RuntimeError('Prompt model checkpoint payload is not ready')
        sources = [snapshot.recurrent[name] for name in recurrent] + [snapshot.draft[name] for name in draft]
        host = snapshot.payload
        offset = 0
        packed = host is not None and host.numel() == snapshot.bytes
        for source in sources:
            packed = packed and source.is_contiguous() and source.data_ptr() == host.data_ptr() + offset
            offset += source.numel() * source.element_size()
        if not packed:
            host = torch.cat([t.contiguous().view(torch.uint8).flatten() for t in sources])
        payload = prepared.cached_payload
        if payload is None:
            payload = host.to(prepared.counter.device, non_blocking=True)
        uploaded, offset = [], 0
        for source in sources:
            nbytes = source.numel() * source.element_size()
            uploaded.append(payload[offset:offset+nbytes].view(source.dtype).reshape(source.shape))
            offset += nbytes
        for (name, destination), source in zip(recurrent.items(), uploaded[:len(recurrent)]):
            destination.copy_(source)
        for (name, (cache, pages, offsets)), source in zip(draft.items(), uploaded[len(recurrent):]):
            cache[pages, offsets] = source
        counter = prepared.counter
        if initialize_counter:
            counter.fill_(1)
        if counter.device.type == 'cuda':
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(counter.device))
            return ModelRestoreCompletion(event, (prepared, host, payload))
        return None
