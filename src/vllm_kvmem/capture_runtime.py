"""Continuous accepted-prefix archive runtime for the native target runner.

Target layers share bounded packed D2H storage. Small decode requests can use an
independent pool/event from bulk prefill; rejection counts share their owner's
fenced packet. This stage provides
the immutable authority for refresh, but does not yet trigger retrieval/replay.
Finalization always drains I/O; diagnostic checksum readback is separately gated.
"""
from dataclasses import asdict, dataclass, field, replace
from contextlib import nullcontext
import hashlib
import json
import os
import time
from threading import Condition, Event, RLock, get_ident
from concurrent.futures import ThreadPoolExecutor

import torch

from .authority import RequestAuthority
from .diagnostics import CompletionHistory, completion_counts
from .options import RuntimeOptions
from .storage import make_shared_archive
from .transfer import PackedDownloads
from .capture_feedback import EarlyCaptureFeedback


@dataclass
class CapturedPacket:
    request_ids: tuple[str, ...]
    tokens: int
    download: object
    next_request: int = 0
    query_inputs: dict = field(default_factory=dict)
    # CPU metadata only; Q bytes share this packet's bounded DMA and lifetime.
    redundancy_queries: dict = field(default_factory=dict)
    # Experiment-only public Q sidecar; acceptance owns branch/row validation.
    pressure_queries: dict = field(default_factory=dict)
    action_inputs: dict = field(default_factory=dict)
    # Hand-built diagnostic packets are already scheduler-confirmed. Serving
    # submit explicitly sets this false until the next scheduler decision.
    confirmed: bool = True
    request_tokens: tuple[int, ...] = ()
    published: set[int] = field(default_factory=set)
    consuming: set[int] = field(default_factory=set)
    lane: str = 'bulk'
    source_copied: dict = field(default_factory=dict)
    source_feedback_bytes: dict = field(default_factory=dict)
    source_released: bool = False
    source_generation: int = 0
    # Fully validated, accounted graph/Action reservations survive source-copy
    # handoff; their owner locks are detached before another thread publishes.
    graph_updates: dict | None = None
    # Exact nonterminal prompt chunks with a frozen Action layout. Only these
    # owners may run one further prefill while this packet is being published.
    pipeline_requests: frozenset[str] = frozenset()
    # A whole-batch score barrier, including frozen reference fallback. It
    # precedes graph reservations and KV append even when graphs are disabled.
    action_validation: object = None
    # A single physical slot can carry exactly two original logical prefill
    # transactions. The owner remains unpublished until its cursor is done.
    logical_group: object = None
    logical_slices: tuple = ()
    logical_cursor: object = None
    logical_confirmation: tuple | None = None
    logical_payload: tuple | None = None


@dataclass
class _DeferredScatterStart:
    # Engine-thread-only intent, bounded to one public forward. No new queue,
    # resource slot, publication permission or GPU completion is introduced.
    rows: tuple
    batch: object = None


class CaptureRuntime(EarlyCaptureFeedback):
    def __init__(self, runner, output):
        self.runner = runner
        self.output = output
        self.options = RuntimeOptions.from_vllm(runner.vllm_config)
        self.config = self.options.memory
        modules = dict(runner.model.named_modules())
        modules.update(getattr(runner, '_kvmem_mtp_modules', {}))
        self.names = tuple(getattr(runner, '_kvmem_archive_layers', runner._kvmem_layers))
        self.layers = [modules[name] for name in self.names]
        heads = {module.num_kv_heads for module in self.layers}
        dims = {module.head_dim for module in self.layers}
        if len(heads) != 1 or len(dims) != 1:
            raise ValueError("First continuous archive requires uniform target KV geometry")
        if any(module.attn.kv_cache_dtype != "fp8_e4m3" for module in self.layers):
            raise ValueError("Continuous target archive currently qualifies FP8 e4m3")
        self.heads, self.dim = heads.pop(), dims.pop()
        self.scales = torch.stack([module.attn._v_scale.reshape(()) for module in self.layers])
        if not torch.all(torch.isfinite(self.scales) & (self.scales > 0)).item():
            raise ValueError("Invalid target V scales")
        self.row_elements = len(self.layers) * self.heads * self.dim
        self.raw_dtype = torch.float8_e4m3fn
        gc = getattr(self.options, 'historical_gc', None)
        self.redundancy_enabled = gc is not None and gc.policy == 'redundancy' and gc.mode != 'off'
        self.redundancy_q_stats = dict(packets=0, bytes=0, overflow_packets=0)
        q_slot_bytes = 0
        if self.redundancy_enabled:
            from .historical_gc import redundancy_layers, REDUNDANCY_Q_SLOT_BYTES
            unsupported = ('offline_prefill_coalescing', 'prefill_physical_coalescing', 'prefill_publication_pipeline',
                           'early_capture_slot_release', 'prefill_capture_ahead')
            if any(getattr(self.options, key, False) for key in unsupported):
                raise ValueError('Redundancy Q capture does not support coalescing or early/pipelined publication')
            self.redundancy_layers = redundancy_layers(len(runner._kvmem_layers))
            query_heads = {self.layers[i].num_heads for i in self.redundancy_layers}
            if len(query_heads) != 1:
                raise ValueError('Redundancy requires uniform target Query head geometry')
            self.redundancy_query_heads = query_heads.pop()
            q_slot_bytes = REDUNDANCY_Q_SLOT_BYTES
        from .pressure_query_capture import configured_method, PRESSURE_QUERY_WINDOW
        self.pressure_query_enabled = configured_method() is not None
        self.pressure_query_stats = dict(packets=0, bytes=0, accepted_rows=0)
        self.pressure_query_window = PRESSURE_QUERY_WINDOW
        if self.pressure_query_enabled:
            unsupported = ('offline_prefill_coalescing', 'prefill_physical_coalescing',
                           'prefill_publication_pipeline', 'early_capture_slot_release',
                           'prefill_capture_ahead')
            active = [key for key in unsupported if getattr(self.options, key, False)]
            if active:
                raise ValueError('Recall pressure Q capture requires disabled options: ' + ', '.join(active))
            self.pressure_query_layers = tuple(range(len(runner._kvmem_layers)))
            query_heads = {self.layers[i].num_heads for i in self.pressure_query_layers}
            if len(query_heads) != 1:
                raise ValueError('Recall pressure Q capture requires uniform target Query geometry')
            self.pressure_query_heads = query_heads.pop()
            q_slot_bytes += (runner.vllm_config.scheduler_config.max_num_seqs *
                PRESSURE_QUERY_WINDOW * len(self.pressure_query_layers) *
                self.pressure_query_heads * self.dim * 2)
        record_bytes = self.row_elements * self.config.block_tokens * 2
        if self.options.cpu_history_coordination:
            from .cache_manifest import archive_bytes_per_token
            if archive_bytes_per_token(runner.kv_cache_config) != self.row_elements * 2:
                raise ValueError('Worker CPU authority differs from scheduler credit geometry')
        output.mkdir(parents=True, exist_ok=True)
        max_tokens = runner.vllm_config.scheduler_config.max_num_batched_tokens
        max_reqs = runner.vllm_config.scheduler_config.max_num_seqs
        self._capture_pack_cache = None
        self.capture_pack_stats = {}
        if getattr(self.options, 'capture_fused_pack', False):
            from .capture_pack import PackLimits
            self._capture_pack_limits = PackLimits(len(self.layers), self.heads, self.dim, max_tokens, max_reqs)
        self._early_owner_limit = max_reqs
        self.gc_config = None
        self.gc_progress = Event()
        self.gc_wait_seconds = 0.0
        self.gc_wait_count = 0
        self._archive_wait_deadline = None
        self._cpu_scatter_executor = None
        self._cpu_scatter_future = None
        self._publication_condition = Condition(RLock())
        self._publication_error = None
        self._published_capacity = {}
        self._progress_lock = RLock()
        self.cpu_scatter_batches = 0
        self.cpu_scatter_seconds = self.cpu_scatter_join_seconds = 0.0
        self.prefill_pipeline_stats = {}
        if self.options.async_cpu_scatter and not getattr(runner,'_kvmem_bounded_target',False):
            raise ValueError('Background CPU scatter requires native bounded terminal-prefix control')
        if self.options.repository_gc:
            if not getattr(runner, '_kvmem_bounded_target', False):
                raise ValueError('Serving repository GC requires bounded execution')
            from .repository_gc import serving_gc_config
            self.gc_config = serving_gc_config(self.config, max_tokens,
                checkpoint_frames=int(bool(self.options.session_capacity)))
        self.archive = getattr(runner, '_kvmem_prepared_archive', None)
        if self.archive is None:
            self.archive = make_shared_archive(self.options, output / 'continuous_archive', record_bytes,
                max_reqs if getattr(runner, '_kvmem_bounded_target', False) else 1,
                max_query_tokens=max_tokens)
        self.downloads = PackedDownloads(max_tokens * self.row_elements * 2 + (4 * max_reqs + 1) * 8 + q_slot_bytes,
                                         2, runner.device)
        split_flag = os.getenv('KVMEM_SPLIT_CAPTURE_DMA',
            '1' if getattr(self.options, 'split_capture_dma', True) else '0')
        if split_flag not in ('0', '1'):
            raise ValueError('KVMEM_SPLIT_CAPTURE_DMA must be 0 or 1')
        self.split_capture_dma = split_flag == '1'
        if self.early_capture_enabled and not self.split_capture_dma:
            raise ValueError('Early capture slot release requires split capture DMA')
        self.decode_downloads = None
        self.decode_capture_limit = min(127, max(1, int(getattr(runner, 'decode_query_len', 1))))
        if self.split_capture_dma:
            # Two fixed small slots, not another full-batch pool. The original
            # bulk capacity remains sufficient for an all-prefill batch.
            small_tokens = min(max_tokens, max_reqs * self.decode_capture_limit)
            self.decode_downloads = PackedDownloads(
                small_tokens * self.row_elements * 2 + (4 * max_reqs + 1) * 8 + q_slot_bytes,
                2, runner.device)
        self.resident_records = max(16, (max_tokens + self.config.block_tokens - 1) // self.config.block_tokens + 1)
        self.authorities = {}
        self.queue = []
        self.accepted = {}
        self.rejected = {}
        self.last_steps = {}
        self.terminal_prefixes = {}
        self.terminal_discarded = {}
        self.completed = CompletionHistory(serving=bool(getattr(runner, '_kvmem_serving', False)))
        self.batches = 0
        self._progress_batch = -1
        self.retiring = {}
        self.retirement = None
        if self.options.async_archive_retirement:
            if not getattr(runner, '_kvmem_bounded_target', False):
                raise ValueError('Asynchronous retirement requires native bounded authority admission')
            from .retirement import RetirementQueue
            self.retirement = RetirementQueue(max_reqs, self.options.verify_archive_on_finish)
        self.action = None
        if self.config.action_attention:
            from .action_attention import ActionRuntime
            self.action = ActionRuntime(self)
        self.block_graph_budget = None
        if getattr(self.options, 'graph_mode', 'off') != 'off':
            from .block_graph import GraphBudget
            self.block_graph_budget = GraphBudget(self.options.graph_max_total_bytes)

    def submit(self, captured, batch, rejected):
        self._raise_publication_error()
        profile = getattr(getattr(self, 'action', None), 'profile', None)
        trace = profile.begin_batch(batch) if profile is not None else None
        try:
            with trace.stage('publication.batch_drain', kind='batch',
                             request_ids=list(batch.req_ids)) if trace else nullcontext():
                action = getattr(self, 'action', None)
                planned = getattr(action, 'prefill_plans', None)
                pipeline = (set(getattr(action, 'pipeline_requests', ()))
                    if self.prefill_pipeline_enabled and planned is not None and
                    planned[0] is batch else set())
                fenced = tuple(rid for rid in batch.req_ids if rid not in pipeline)
                if not self.drain_requests(fenced, wait=True):
                    raise RuntimeError('Public capture preceded scheduler confirmation')
                for request_id in pipeline:
                    if not self.prefill_pipeline_can_advance(request_id):
                        raise RuntimeError('Prefill pipeline lost its confirmed bounded predecessor')
        except BaseException:
            if profile is not None:
                profile.finish_batch(status='error')
            raise
        if tuple(c.name for c in captured) != self.names:
            raise ValueError("Continuous capture missed or reordered target layers")
        if any(c.key.dtype != torch.bfloat16 for c in captured):
            raise ValueError("Continuous Qwen target capture currently qualifies BF16 activations")
        groups = self._capture_groups(batch)
        if not getattr(self, 'split_capture_dma', False) and len(self.downloads.pending) == len(self.downloads.buffers):
            # Keep the original pre-Action backpressure order in the disabled
            # control arm. Split mode must enqueue decode before a bulk wait.
            self.ensure_download_slot(batch.req_ids)
        ahead = (getattr(self.options, 'prefill_capture_ahead', False) and
                 bool(batch.num_reqs) and pipeline == set(batch.req_ids) and
                 all(batch.is_prefilling_np) and len(groups) == 1 and groups[0][0] == 'bulk' and
                 all(planned[1].get(rid, {}).get('pipeline', False) for rid in batch.req_ids))
        staged = None
        if ahead:
            # Enqueue read-only packing and D2H while this forward is running.
            # Neither a publication packet nor mutable history is exposed until
            # Action's original metadata check/native validation/fallback returns.
            lane, indices = groups[0]
            pool = self._download_pool(lane)
            if len(pool.pending) == len(pool.buffers):
                self.ensure_download_slot((batch.req_ids[i] for i in indices), lane=lane)
            payload, t = self._capture_payload(captured, batch, rejected, indices)
            download = pool.download(payload)
            if download is None:
                raise RuntimeError("Capture queue failed to provide bounded staging after drain")
            staged = payload, t, download
            try:
                action_inputs = self.action.capture(captured, batch)
            except BaseException:
                # This private slot is absent from queue and has no CPU readers.
                # Recycle only after its own DMA; unrelated owners stay pending.
                # A failed CUDA fence keeps the slot quarantined in pool.pending.
                try:
                    download.event.synchronize()
                    if not pool.release(download):
                        raise RuntimeError('Failed to release unsubmitted prefill capture')
                except BaseException as cleanup_error:
                    with self._publication_lock():
                        self._publication_error = cleanup_error
                        self._publication_condition.notify_all()
                    raise
                self._pipeline_count('capture_ahead_discarded')
                raise
        else:
            action_inputs = self.action.capture(captured,batch) if getattr(self,'action',None) is not None else {}
        try:
            logical_group,logical_slices = self._logical_capture_inputs(action_inputs,batch)
        except BaseException as error:
            if staged is not None:
                download = staged[2]
                try:
                    download.event.synchronize()
                    if not self._download_pool(groups[0][0]).release(download):
                        raise RuntimeError('Invalid logical prefill retained an unsubmitted DMA')
                except BaseException as cleanup_error:
                    error.add_note('Logical capture cleanup also failed: '+repr(cleanup_error))
                    self.quarantine_publication(error)
                    raise error
            raise
        packets = {}
        for lane, indices in groups:
            redundancy_queries = {}
            pressure_queries = {}
            if staged is not None:
                payload, t, download = staged
            else:
                pool = self._download_pool(lane)
                if len(pool.pending) == len(pool.buffers):
                    self.ensure_download_slot((batch.req_ids[i] for i in indices), lane=lane)
                payload, t = self._capture_payload(captured, batch, rejected, indices)
                payload, redundancy_queries = self._capture_redundancy_queries(
                    payload, captured, batch, indices)
                payload, pressure_queries = self._capture_pressure_queries(
                    payload, captured, batch, indices)
                download = pool.download(payload)
                if download is None:
                    raise RuntimeError("Capture queue failed to provide bounded staging after drain")
            packet = CapturedPacket(tuple(batch.req_ids[i] for i in indices), t, download, confirmed=False,
                request_tokens=tuple(int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i])
                                     for i in indices), lane=lane)
            packet.action_inputs = {rid: action_inputs[rid] for rid in packet.request_ids if rid in action_inputs}
            packet.redundancy_queries = redundancy_queries
            packet.pressure_queries = pressure_queries
            packet.action_validation = getattr(action_inputs, 'validation', None)
            packet.pipeline_requests = frozenset(pipeline.intersection(packet.request_ids))
            if logical_group is not None:
                packet.logical_group,packet.logical_slices = logical_group,logical_slices
                self._logical_source_generation = getattr(self,'_logical_source_generation',0)+1
                packet.source_generation = self._logical_source_generation
            with self._publication_lock():
                if packet.pipeline_requests:
                    self._pipeline_count('eligible_packets')
                    self._pipeline_count('eligible_owners', len(packet.pipeline_requests))
                    previous = {rid: sum(rid == old.request_ids[i]
                        for old in self.queue for i in range(len(old.request_ids))
                        if i not in old.published) for rid in packet.pipeline_requests}
                    self._pipeline_count('submits_with_pending', sum(n > 0 for n in previous.values()))
                    self._pipeline_max('max_pending_per_owner', max(previous.values()) + 1)
                self.queue.append(packet)
                if staged is not None:
                    self._pipeline_count('capture_ahead_batches')
                if self.prefill_pipeline_enabled:
                    self._pipeline_max('max_bulk_pending', sum(p.lane == 'bulk' for p in self.queue))
                if self.early_capture_packet(packet):
                    self._early_source_submitted(packet)
            packets.update((rid, packet) for rid in packet.request_ids)
            self._publication_metric('submitted_bytes', payload.numel(), lane=lane)
            self._publication_metric('submitted_packets', 1, lane=lane)
        with self._publication_lock():
            self.batches += 1
        n = batch.num_reqs
        relevance_rows = []
        keep_guided = (getattr(self.options, 'guided_query_tokens', 0) and
                       getattr(self.options, 'guided_reselect', 'boundary') in ('middecode', 'both'))
        if (self.options.query_history or keep_guided) and batch.has_prefill:
            sessions = getattr(self.runner,'_kvmem_sessions',None)
            for i, request_id in enumerate(batch.req_ids):
                binding = sessions.bindings.get(request_id) if sessions is not None else None
                if binding is None or not batch.is_prefilling_np[i]:
                    continue
                start = int(batch.num_computed_tokens_np[i])
                end = start + int(batch.num_scheduled_tokens[i])
                q = binding.query_span
                if end >= len(binding.tokens) or (q is not None and start <= q[0] < q[1] <= end):
                    relevance_rows.append((i,request_id))
        if relevance_rows:
            # Preserve Q, but never drain the just-sampled batch: neighbouring
            # MTP rows may still be trimmed by the next scheduler terminal notice.
            for i, request_id in relevance_rows:
                binding = sessions.bindings[request_id]
                queries = None
                history = getattr(self.authorities.get(request_id),'query_history',None)
                if binding.query_span is not None and (history is None or
                        history.request_id != request_id or history.observation is None):
                    qb, qe = binding.query_span
                    start = int(batch.num_computed_tokens_np[i])
                    end = start + int(batch.num_scheduled_tokens[i])
                    if start <= qb < qe <= end:
                        at = int(batch.query_start_loc_np[i])
                        queries = torch.stack([c.query[at+qb-start:at+qe-start]
                            for c in captured[:len(self.runner._kvmem_layers)]]).clone()
                packets[request_id].query_inputs[request_id] = (binding,queries)

    def _logical_capture_inputs(self, action_inputs, batch):
        """Require explicit worker-bound plans; never infer them from token count."""
        groups = getattr(action_inputs,'logical_groups',{})
        slices = getattr(action_inputs,'logical_slices',{})
        if not groups and not slices:
            return None,()
        from .logical_prefill import LogicalPrefillGroup
        from .logical_action import LogicalActionSlice
        if (not getattr(self.options,'prefill_physical_coalescing',False) or
                self.early_capture_enabled or getattr(action_inputs,'validation',None) is not None or
                batch.num_reqs != 1 or len(groups) != 1 or set(groups) != set(slices) or
                set(groups) != set(batch.req_ids) or not bool(batch.is_prefilling_np[0])):
            raise RuntimeError('Logical prefill requires one ordinary fully validated physical owner')
        rid = batch.req_ids[0]
        group,parts = groups[rid],slices[rid]
        if (not isinstance(group,LogicalPrefillGroup) or group.request_id != rid or
                not 2 <= len(group.spans) <= getattr(self.options,'prefill_physical_max_chunks',2) or
                type(parts) is not tuple or len(parts) != len(group.spans) or
                any(not isinstance(part,LogicalActionSlice) or part.group is not group or
                    part.plan.span != span for part,span in zip(parts,group.spans)) or
                int(batch.num_computed_tokens_np[0]) != group.history_begin or
                int(batch.query_start_loc_np[1]-batch.query_start_loc_np[0]) != group.count or
                getattr(self.runner,'_kvmem_logical_prefill_step',None) != group.scheduler_step):
            raise RuntimeError('Logical prefill capture differs from its immutable worker plan')
        sessions = getattr(self.runner,'_kvmem_sessions',None)
        binding = sessions.bindings.get(rid) if sessions is not None else None
        if binding is not None:
            query = binding.query_span
            if (group.history_end >= len(binding.tokens) or
                    query is not None and group.history_begin <= query[0] < query[1] <= group.history_end):
                raise RuntimeError('Logical prefill cannot carry a Query seed or final-input observation')
        return group,parts

    def _capture_groups(self, batch):
        all_indices = tuple(range(batch.num_reqs))
        if not getattr(self, 'split_capture_dma', False):
            return [('bulk', all_indices)]
        small = tuple(i for i in all_indices if not batch.is_prefilling_np[i] and
            0 < int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i]) <= self.decode_capture_limit)
        large = tuple(i for i in all_indices if i not in small)
        # Enqueue decode's compact payload and ready event before doing any of
        # the bulk quantization/packing. Independent streams avoid a software
        # FIFO dependency; the physical copy engine/bandwidth is still shared.
        return ([('decode', small)] if small else []) + ([('bulk', large)] if large else [])

    def _capture_payload(self, captured, batch, rejected, indices):
        stats = self.__dict__.setdefault('capture_pack_stats', {})
        if getattr(getattr(self, 'options', None), 'capture_fused_pack', False):
            from .capture_pack import try_pack
            payload, reason = try_pack(self, captured, batch, rejected, indices)
            if payload is not None:
                stats['fused_batches'] = stats.get('fused_batches', 0) + 1
                stats['fused_payload_bytes'] = stats.get('fused_payload_bytes', 0) + payload.numel()
                return payload, batch.num_tokens
            name = 'fallback:' + reason
            stats[name] = stats.get(name, 0) + 1
        stats['reference_batches'] = stats.get('reference_batches', 0) + 1
        n = batch.num_reqs
        starts = batch.query_start_loc[:n+1].to(torch.int64)
        positions = captured[0].positions
        if positions.ndim == 2:
            positions = positions[0]
        if len(indices) == n:
            t = batch.num_tokens
            header = torch.cat((starts, batch.positions[starts[:-1]], positions[starts[:-1]],
                                rejected[:n].to(torch.int64)))
            keys = [c.key[:t] for c in captured]
            vals = [c.value[:t] for c in captured]
        else:
            # Stack existing device scalars: constructing a CUDA tensor from a
            # Python list here would add a blocking H2D copy after the model.
            original = torch.stack([starts[i] for i in indices])
            lengths = torch.stack([starts[i+1] for i in indices]) - original
            compact = torch.cat((starts.new_zeros(1), lengths.cumsum(0)))
            # CPU counts are upper bounds under adaptive MTP. Allocate that
            # bounded size, but gather/describe actual rows using GPU starts.
            # Padding after compact[-1] is never accepted or published.
            t = sum(int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i]) for i in indices)
            offsets = torch.arange(t, device=starts.device, dtype=torch.int64)
            owner = torch.searchsorted(compact[1:], offsets, right=True).clamp_max(len(indices)-1)
            rows = (original[owner] + offsets - compact[owner]).clamp(0, batch.num_tokens-1)
            header = torch.cat((compact, batch.positions[original], positions[original],
                                torch.stack([rejected[i] for i in indices]).to(torch.int64)))
            keys = [c.key.index_select(0, rows) for c in captured]
            vals = [c.value.index_select(0, rows) for c in captured]
        raw = torch.stack(keys).float().clamp(-448, 448).to(self.raw_dtype)
        values = (torch.stack(vals).float() / self.scales[:, None, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return torch.cat((header.view(torch.uint8), raw.view(torch.uint8).flatten(),
                          values.view(torch.uint8).flatten())), t

    def _capture_redundancy_queries(self, payload, captured, batch, indices):
        """Append bounded BF16 public Q, keeping head order and original rows.

        The CPU scheduler supplies only bounds; device starts locate actual MTP
        rows. Extra adaptive/padding rows are excluded by the later accepted
        prefix. No .item(), CPU readback or device synchronization occurs here.
        """
        if not getattr(self, 'redundancy_enabled', False):
            return payload, {}
        from .historical_gc import REDUNDANCY_Q_SLOT_BYTES
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        descriptors, total = {}, 0
        layers = self.redundancy_layers
        row_bytes = len(layers)*self.redundancy_query_heads*self.dim*2
        for i in indices:
            rid = batch.req_ids[i]
            binding = sessions.bindings.get(rid) if sessions is not None else None
            if binding is None:
                if rid not in getattr(sessions, 'release_requests', ()):
                    raise ValueError('Redundancy public capture requires an owned session binding')
                descriptors[rid] = dict(binding=None, count=0)
                continue
            # The native batch retains logical positions. WindowBatchAdapter
            # translates only attention/RoPE positions, not this input batch.
            history = int(batch.num_computed_tokens_np[i])
            count = int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i])
            prefill, offset = bool(batch.is_prefilling_np[i]), 0
            if prefill:
                query = binding.query_span
                if query is None:
                    count = 0
                else:
                    offset = max(0, query[0]-history)
                    count = max(0, min(history+count, query[1])-history-offset)
            descriptors[rid] = dict(binding=binding, prefill=prefill, row_offset=offset,
                count=count, byte_offset=payload.numel()+total, history_begin=history,
                batch_index=i, overflow=False)
            total += count*row_bytes
        self.redundancy_q_stats['packets'] += 1
        if total > REDUNDANCY_Q_SLOT_BYTES:
            self.redundancy_q_stats['overflow_packets'] += 1
            for row in descriptors.values():
                row['overflow'] = bool(row['count'])
                row['count'] = 0
            return payload, descriptors
        if not total:
            return payload, descriptors
        result = torch.empty(payload.numel()+total, dtype=torch.uint8, device=payload.device)
        result[:payload.numel()].copy_(payload)
        for row in descriptors.values():
            count = row['count']
            if not count:
                continue
            i, offset = row['batch_index'], row['row_offset']
            positions = (batch.query_start_loc[i].to(torch.int64) + offset +
                torch.arange(count, device=payload.device, dtype=torch.int64)).clamp(0, batch.num_tokens-1)
            at = row['byte_offset']
            target = result[at:at+count*row_bytes].view(torch.bfloat16).view(
                len(layers), count, self.redundancy_query_heads, self.dim)
            for out, layer in enumerate(layers):
                target[out].copy_(captured[layer].query.index_select(0, positions))
        self.redundancy_q_stats['bytes'] += total
        return result, descriptors

    def _accept_redundancy_queries(self, packet, request_id, history_begin, retained):
        if not getattr(self, 'redundancy_enabled', False):
            return
        row = packet.redundancy_queries.pop(request_id, None)
        if row is None:
            raise RuntimeError('Accepted public owner lost its redundancy Q descriptor')
        if row['binding'] is None:
            return  # Narrow session-release allowlist; never creates a Q bank.
        authority = self.authorities[request_id]
        history = authority.redundancy_history
        if row['prefill'] and row['history_begin'] != history_begin:
            # Do not associate a stale scheduler extent with another prefix.
            history.invalidate()
            return
        # Adaptive MTP's CPU computed count is an upper bound. Decode row zero
        # uses the actual device start and its accepted-header history below.
        values = None
        if row['count']:
            size = len(self.redundancy_layers)*row['count']*self.redundancy_query_heads*self.dim*2
            at = row['byte_offset']
            values = packet.download.payload[at:at+size].view(torch.bfloat16).view(
                len(self.redundancy_layers), row['count'], self.redundancy_query_heads, self.dim)
        history.accept(row['binding'], history_begin, retained, prefill=row['prefill'],
            row_offset=row['row_offset'], rows=values, overflow=row['overflow'])

    def _capture_pressure_queries(self, payload, captured, batch, indices):
        """Copy original public Q into the same bounded accepted-prefix DMA.

        Prefill keeps its last 128 scheduled rows. Decode keeps all possible
        accepted rows, locating them by device starts under adaptive MTP. The
        private replay/refresh suffix is never part of these public owners.
        """
        if not getattr(self, 'pressure_query_enabled', False):
            return payload, {}
        layers = self.pressure_query_layers
        row_bytes = len(layers) * self.pressure_query_heads * self.dim * 2
        descriptors, total = {}, 0
        for i in indices:
            rid = batch.req_ids[i]
            prefill = bool(batch.is_prefilling_np[i])
            scheduled = int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i])
            if scheduled <= 0 or not prefill and scheduled > self.pressure_query_window:
                raise ValueError('Public pressure Query capture exceeds its bounded row geometry')
            offset = max(0, scheduled-self.pressure_query_window) if prefill else 0
            count = scheduled-offset
            authority = self.authorities.get(rid)
            window = getattr(authority, 'pressure_query_window', None)
            descriptors[rid] = dict(prefill=prefill, row_offset=offset, count=count,
                history_begin=int(batch.num_computed_tokens_np[i]), batch_index=i,
                epoch=window.epoch if window is not None else 0,
                byte_offset=payload.numel()+total)
            total += count*row_bytes
        result = torch.empty(payload.numel()+total, dtype=torch.uint8, device=payload.device)
        result[:payload.numel()].copy_(payload)
        for row in descriptors.values():
            count, i = row['count'], row['batch_index']
            positions = (batch.query_start_loc[i].to(torch.int64) + row['row_offset'] +
                torch.arange(count, device=payload.device, dtype=torch.int64)).clamp(0, batch.num_tokens-1)
            at = row['byte_offset']
            target = result[at:at+count*row_bytes].view(torch.bfloat16).view(
                len(layers), count, self.pressure_query_heads, self.dim)
            for out, layer in enumerate(layers):
                target[out].copy_(captured[layer].query.index_select(0, positions))
        self.pressure_query_stats['packets'] += 1
        self.pressure_query_stats['bytes'] += total
        return result, descriptors

    def _accept_pressure_queries(self, packet, request_id, history_begin, retained):
        if not getattr(self, 'pressure_query_enabled', False):
            return
        row = packet.pressure_queries.pop(request_id, None)
        if row is None:
            raise RuntimeError('Accepted public owner lost its pressure Query descriptor')
        if row['prefill'] and row['history_begin'] != history_begin:
            raise RuntimeError('Prefill pressure Query capture differs from its accepted prefix')
        authority = self.authorities[request_id]
        window = getattr(authority, 'pressure_query_window', None)
        if window is None:
            raise RuntimeError('Recall experiment authority lacks its accepted pressure Query cache')
        size = len(self.pressure_query_layers)*row['count']*self.pressure_query_heads*self.dim*2
        at = row['byte_offset']
        values = packet.download.payload[at:at+size].view(torch.bfloat16).view(
            len(self.pressure_query_layers), row['count'], self.pressure_query_heads, self.dim)
        window.accept(history_begin, retained, values, row_offset=row['row_offset'],
            epoch=row['epoch'], prefill=row['prefill'])
        self.pressure_query_stats['accepted_rows'] += max(0, min(row['count'], retained-row['row_offset']))

    def _download_pool(self, lane='bulk'):
        return self.decode_downloads if lane == 'decode' else self.downloads

    def _pending_download_count(self):
        return len(self.downloads.pending) + (len(self.decode_downloads.pending)
            if getattr(self, 'decode_downloads', None) is not None else 0)

    def _publication_metric(self, name, value, *, lane=None):
        thread = 'background' if getattr(self, '_cpu_scatter_thread', None) == get_ident() else 'main'
        key = thread if lane is None else thread + '/' + lane
        with self._publication_lock():
            counters = self.__dict__.setdefault('_publication_metrics', {}).setdefault(key, {})
            counters[name] = counters.get(name, 0) + value

    def _publication_metrics_snapshot(self):
        # Caller holds the publication lock. Durations overlap GPU execution;
        # these host counters are never an additive GPU-idle measurement.
        return {key: dict(value) for key, value in getattr(self, '_publication_metrics', {}).items()}

    def _accept_query_input(self, packet, request_id):
        value = packet.query_inputs.pop(request_id,None)
        if value is None:
            return
        binding, queries = value
        authority = self.authorities[request_id]
        if (queries is not None and binding.query_span is not None and
                getattr(self.options, 'guided_query_tokens', 0) and
                getattr(self.options, 'guided_reselect', 'boundary') in ('middecode', 'both')):
            from .guided_seed import publish_seed
            publish_seed(self, authority, queries, binding.query_span[1])
        history = getattr(authority, 'query_history', None)
        if history is None:
            return
        history.begin(request_id,binding.tokens)
        if queries is not None and history.observation is None:
            history.query = queries
        if authority.history_end >= len(binding.tokens) and history.policy.open:
            if history.observation is None and history.query is not None:
                from .request_plan import RequestPlan
                fit = RequestPlan(None,block_tokens=self.config.block_tokens,
                    gen_budget=self.config.gen_budget,mandatory=binding.mandatory_spans).mandatory_fit(
                        authority.history_end,self.config)
                required = set(fit['hard_blocks'])
                records = [r for r in authority.records.values()
                           if r.history_begin//self.config.block_tokens in required]
                if {r.history_begin//self.config.block_tokens for r in records} != required:
                    raise ValueError('Required Query-history evidence is absent from the durable source')
                authority.select_for_queries(history.query, [r.handle for r in records], observe_only=True)
            observed = history.policy.has_observation
            history.commit()
            with (self.output / 'query_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_id': request_id,
                    'round': history.policy.round_count, 'input_end': history.policy.input_end,
                    'observed': observed, 'scored_pages': len(history.policy.scores),
                    'score_sha256':hashlib.sha256(json.dumps(sorted(history.policy.scores.items())).encode()).hexdigest()}) + '\n')

    def _publication_lock(self):
        # __new__ based CPU fixtures use the same locking protocol.
        self.__dict__.setdefault('_published_capacity', {})
        return self.__dict__.setdefault('_publication_condition', Condition(RLock()))

    def _raise_publication_error(self):
        error = getattr(self, '_publication_error', None)
        if error is not None:
            raise error

    def quarantine_publication(self, error):
        """Keep private/queued resources owned after a fatal submission failure."""
        with self._publication_lock():
            self._publication_error = error
            self._publication_condition.notify_all()

    def _busy_requests(self):
        return {packet.request_ids[i] for packet in self.queue for i in
                (packet.consuming | (set(packet.source_copied)-packet.published
                 if self.early_capture_packet(packet) else set()))}

    def _install_terminals(self, request_ids, terminal_prefixes):
        if terminal_prefixes is None:
            return
        for request_id in request_ids:
            known = request_id in self.authorities or any(
                rid == request_id and i not in packet.published
                for packet in self.queue for i, rid in enumerate(packet.request_ids))
            prefix = terminal_prefixes.get(request_id)
            if known and prefix is None:
                raise RuntimeError('Missing scheduler terminal prefix for captured request')
            if prefix is None or not known:
                continue
            authority = self.authorities.get(request_id)
            # history_end is published by native commit, never by reservation.
            if authority is not None and authority.history_end > prefix.history_end:
                raise RuntimeError('Terminal prefix arrived after excess history was published')
            self.terminal_prefixes[request_id] = prefix

    def begin_step(self, request_ids=(), terminal_prefixes=None):
        """Apply terminal clipping before authorizing earlier sample packets.

        Does not finish sessions or wait for CPU publication. The runner calls
        this once per scheduler decision, then receives session metadata and
        calls finish for the ended requests. Newly submitted packets remain
        unconfirmed until the following decision.
        """
        with self._publication_lock():
            self._raise_publication_error()
            self._install_terminals(request_ids, terminal_prefixes)
            action = getattr(self, 'action', None)
            if self.prefill_pipeline_enabled and request_ids and action is not None:
                action.invalidate_prefill_plans(request_ids)
            for packet in self.queue:
                if getattr(packet,'logical_group',None) is not None:
                    step = getattr(self.runner,'_kvmem_logical_prefill_step',None)
                    if type(step) is not int or step <= packet.logical_group.scheduler_step:
                        raise RuntimeError('Logical prefill confirmation must follow its scheduler decision')
                    packet.logical_confirmation = (step,self.terminal_prefixes.get(packet.request_ids[0]))
                packet.confirmed = True
            self._publication_condition.notify_all()

    def defer_terminals(self, output):
        """Keep ended owners' native indices until their CPU prefix is ready.

        begin_step must install terminal clipping first. Finished IDs are sent
        once by the scheduler, so save and replay them on a later worker step.
        This composes after the refresh runtime's own terminal deferral.
        """
        from .connector import ControlMetadata
        control = output.kv_connector_metadata
        if not isinstance(control, ControlMetadata):
            return output
        deferred = self.__dict__.setdefault('_deferred_terminals', {})
        finished, prefixes = set(output.finished_req_ids), dict(control.terminal_prefixes)
        with self._publication_lock():
            # Refresh may replay a previously hidden finished ID this step.
            self._install_terminals(finished, prefixes)
        # A small completed decode needs no worker handoff, but must never wait
        # for DMA or for an unrelated in-flight owner on this control path.
        small = (finished | set(deferred)) - self._large_pending_requests()
        self.drain_requests(small, wait=False)
        with self._publication_lock():
            for request_id in list(finished):
                if not self.publication_ready((request_id,)):
                    deferred[request_id] = prefixes.pop(request_id, None)
                    finished.remove(request_id)
            for request_id, prefix in list(deferred.items()):
                if self.publication_ready((request_id,)):
                    finished.add(request_id)
                    if prefix is not None:
                        prefixes[request_id] = prefix
                    del deferred[request_id]
        return replace(output, finished_req_ids=finished,
                       kv_connector_metadata=replace(control, terminal_prefixes=prefixes))

    def publication_ready(self, request_ids):
        """Whether every queued publication for these requests is complete."""
        wanted = set(request_ids)
        with self._publication_lock():
            self._raise_publication_error()
            return not any(any(request_id in wanted and i not in packet.published
                for i, request_id in enumerate(packet.request_ids)) for packet in self.queue)

    @property
    def prefill_pipeline_enabled(self):
        return bool(getattr(self.options, 'prefill_publication_pipeline', False))

    def _pipeline_count(self, name, value=1):
        counters = self.__dict__.setdefault('prefill_pipeline_stats', {})
        counters[name] = counters.get(name, 0) + value

    def _pipeline_max(self, name, value):
        counters = self.__dict__.setdefault('prefill_pipeline_stats', {})
        counters[name] = max(counters.get(name, 0), value)

    def prefill_pipeline_can_advance(self, request_id):
        """Planning may lead a confirmed ordinary prompt prefix by one chunk.

        This is not publication readiness: refresh, finish, restore and accepted
        decode still use their full owner fence. Immutable plans are prepared on
        the engine thread; the worker retains both old DMA sources until commit.
        """
        with self._publication_lock():
            self._raise_publication_error()
            if not self.prefill_pipeline_enabled or request_id in self.terminal_prefixes:
                return False
            pending = [packet for packet in self.queue
                for i, rid in enumerate(packet.request_ids)
                if rid == request_id and i not in packet.published]
            allowed = len(pending) <= 1 and all(packet.confirmed and
                request_id in packet.pipeline_requests for packet in pending)
            if allowed and pending:
                self._pipeline_count('advance_checks_with_pending')
            return allowed

    def _large_pending_requests(self):
        """Publication workers still own large packets when admission overlaps."""
        return self._publication_blocked_requests(allow_pipeline=False)

    def note_prefill_pipeline_forward(self, batch):
        """Nonblocking host evidence immediately before native model dispatch."""
        if not self.prefill_pipeline_enabled:
            return
        action = getattr(self, 'action', None)
        planned = getattr(action, 'prefill_plans', None)
        if planned is None or planned[0] is not batch:
            return
        with self._publication_lock():
            pending = [(packet, i) for packet in self.queue
                       for i, rid in enumerate(packet.request_ids)
                       if rid in action.pipeline_requests and i not in packet.published]
            if pending:
                self._pipeline_count('forwards_with_pending')
                self._pipeline_count('forwards_with_background_publication',
                                     int(any(i in packet.consuming for packet, i in pending)))

    def publication_blocked_requests(self):
        return self._publication_blocked_requests(allow_pipeline=True)

    def _publication_blocked_requests(self, *, allow_pipeline):
        """Large pending owners for scheduler admission, including newest DMA.

        A scheduler must still send confirmation/poll iterations when all such
        owners are blocked. Tiny decode prefixes can use request-local fences.
        """
        with self._publication_lock():
            self._raise_publication_error()
            threshold = self.options.cpu_scatter_min_tokens
            pending = {}
            packets = {}
            for packet in self.queue:
                sizes = packet.request_tokens or (packet.tokens,) * len(packet.request_ids)
                for i, request_id in enumerate(packet.request_ids):
                    if i not in packet.published:
                        pending[request_id] = pending.get(request_id, 0) + sizes[i]
                        packets.setdefault(request_id, []).append(packet)
            blocked = {request_id for request_id, count in pending.items() if count >= threshold}
            if (allow_pipeline and self.prefill_pipeline_enabled and
                    len(self.downloads.pending) < len(self.downloads.buffers)):
                blocked.difference_update(request_id for request_id, owned in packets.items()
                    if len(owned) == 1 and request_id in owned[0].pipeline_requests
                    and request_id not in self.terminal_prefixes)
            if self.early_capture_enabled:
                blocked.update(packet.request_ids[i] for packet in self.queue
                               if self.early_capture_packet(packet)
                               for i in set(packet.source_copied)-packet.published)
            return blocked

    def ensure_download_slot(self, request_ids, *, lane='bulk'):
        """Wait for one reusable bounded slot, never for the entire queue."""
        if self.early_capture_enabled and lane == 'bulk':
            return self._ensure_early_download_slot(lane)
        self.drain_requests(request_ids, wait=True)
        pool = self._download_pool(lane)
        condition = self._publication_lock()
        started = time.perf_counter()
        try:
            while True:
                with condition:
                    self._raise_publication_error()
                    if len(pool.pending) < len(pool.buffers):
                        return
                    owned = [p for p in self.queue if p.lane == lane]
                    available = [p for p in owned if p.confirmed and not p.consuming]
                    if available:
                        packet = min(available, key=lambda p: sum(
                            (p.request_tokens or (p.tokens,) * len(p.request_ids))[i]
                            for i in range(len(p.request_ids)) if i not in p.published))
                        needed = {request_id for i, request_id in enumerate(packet.request_ids)
                                  if i not in packet.published}
                    elif any(p.consuming for p in owned):
                        waited = time.perf_counter()
                        condition.wait()
                        self._publication_metric('slot_condition_wait_seconds', time.perf_counter()-waited, lane=lane)
                        continue
                    else:
                        raise RuntimeError('Bounded capture slots contain only unconfirmed samples')
                reclaimed = time.perf_counter()
                published = self.drain_requests(needed, wait=True)
                self._publication_metric('slot_reclaim_seconds', time.perf_counter()-reclaimed, lane=lane)
                if not published:
                    raise RuntimeError('Cannot reclaim an unconfirmed capture slot')
        finally:
            self._publication_metric('slot_wait_seconds', time.perf_counter()-started, lane=lane)
            self._publication_metric('slot_wait_calls', 1, lane=lane)

    def _scatter_start_count(self, name):
        counts = self.__dict__.setdefault('prefill_scatter_after_target_stats', {})
        counts[name] = counts.get(name, 0) + 1

    def defer_prefill_scatter(self, output, refresh=None):
        """Delay only a new worker for an exact ordinary pipeline successor.

        This runs after scheduler confirmation, terminal handling and session
        commands. Its CPU-only checks use the immutable planner, never mutable
        archive contents. All other work keeps the original immediate start.
        """
        if not getattr(self.options, 'prefill_scatter_after_target', False):
            return False
        self._raise_publication_error()
        if getattr(self, '_prefill_scatter_deferred', None) is not None:
            error = RuntimeError('Deferred scatter escaped its public forward')
            self.abort_prefill_scatter(error)
            raise error

        def decline(reason):
            self._scatter_start_count('eligibility_declines')
            self._scatter_start_count('declined:' + reason)
            return False

        future = getattr(self, '_cpu_scatter_future', None)
        if future is not None:
            if not future.done():
                self._scatter_start_count('already_running_batches')
                return decline('worker_running')
            try:
                self.join_cpu_scatter()
            except BaseException as error:
                self.quarantine_publication(error)
                raise
        scheduled = output.num_scheduled_tokens
        control = getattr(output, 'kv_connector_metadata', None)
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        if (not scheduled or control is None or getattr(refresh, 'jobs', None) or
                getattr(control, 'commands', None) or getattr(control, 'cancelled', None) or
                getattr(control, 'terminal_prefixes', None) or output.finished_req_ids or
                getattr(output, 'preempted_req_ids', None) or self.terminal_prefixes or
                getattr(self, 'retiring', None) or
                any(getattr(sessions, name, None) for name in
                    ('imports', 'captures', 'terminal_captures', 'retiring'))):
            return decline('lifecycle')
        commands = getattr(control, 'sessions', None)
        if any(getattr(commands, name, None) for name in
               ('bindings', 'captures', 'evictions', 'terminals', 'release_requests')):
            return decline('session_commands')
        cached = getattr(output, 'scheduled_cached_reqs', None)
        if (cached is None or output.scheduled_new_reqs or cached.resumed_req_ids or
                output.scheduled_spec_decode_tokens or
                getattr(output, 'scheduled_encoder_inputs', None) or
                getattr(output, 'has_sync_kv_loads', False)):
            return decline('nonordinary_batch')
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        generated = dict(zip(cached.req_ids, cached.num_output_tokens, strict=True))
        action = getattr(self, 'action', None)
        native = getattr(action, 'native', None)
        if (native is None or native.disabled or native.mode != 'reuse' or
                not callable(getattr(action, 'prefill_pipeline_ready', None))):
            return decline('native_plan')
        states = self.runner.req_states
        offsets = getattr(getattr(self.runner, '_kvmem_window_adapter', None), 'offsets', {})
        rows = []
        with self._publication_lock():
            # One old packet and one free bulk slot preserve the existing
            # one-chunk lookahead. Unrelated owners must never be held back.
            if (len(self.queue) != 1 or len(self.downloads.pending) != 1 or
                    len(self.downloads.pending) >= len(self.downloads.buffers)):
                return decline('slot_or_queue')
            packet = self.queue[0]
            if (packet.lane != 'bulk' or not packet.confirmed or packet.published or packet.consuming or
                    set(packet.request_ids) != set(scheduled) or
                    packet.pipeline_requests != frozenset(scheduled) or
                    (packet.tokens < self.options.cpu_scatter_min_tokens and
                     packet.action_validation is None)):
                return decline('predecessor')
            for request_id, count in scheduled.items():
                index = states.req_id_to_index.get(request_id)
                history = computed.get(request_id)
                if (index is None or history is None or generated.get(request_id) != 0 or count <= 0 or
                        history + count >= int(states.prompt_len.np[index])):
                    return decline('prompt_boundary')
                local = history - offsets.get(request_id, 0)
                authority = self.authorities.get(request_id)
                excluded = getattr(getattr(authority, 'action_history', None), 'prefill_replay', None)
                if (local < 0 or (excluded is not None and history < excluded[1] and
                                 history + count > excluded[0]) or
                        not action.prefill_pipeline_ready(request_id, history, local) or
                        not self.prefill_pipeline_can_advance(request_id)):
                    return decline('immutable_plan')
                rows.append((request_id, history, local, count))
            self._prefill_scatter_deferred = _DeferredScatterStart(tuple(rows))
        self._scatter_start_count('deferred_batches')
        return True

    def recheck_prefill_scatter(self, batch):
        """Restore immediate startup if prepared input differs from admission."""
        pending = getattr(self, '_prefill_scatter_deferred', None)
        if pending is None:
            return
        action = getattr(self, 'action', None)
        prepared = getattr(action, 'prefill_plans', None)
        native = getattr(action, 'native_plans', None)
        requests = {row[0] for row in pending.rows}
        valid = (set(batch.req_ids) == requests and all(batch.is_prefilling_np) and
                 prepared is not None and prepared[0] is batch and
                 native is not None and native[0] is batch and
                 getattr(action, 'pipeline_requests', set()) == requests)
        if valid:
            expected = {row[0]: row[1:] for row in pending.rows}
            for index, request_id in enumerate(batch.req_ids):
                begin, end = map(int, batch.query_start_loc_np[index:index+2])
                history, local, count = expected[request_id]
                plans = (prepared[1].get(request_id), native[1].get(request_id))
                if (int(batch.num_computed_tokens_np[index]) != history or end - begin != count or
                        any(plan is None or not plan.get('pipeline', False) or
                            (plan['history_begin'], plan['local_begin'], plan['count'], plan['index'], plan['begin']) !=
                            (history, local, count, index, begin) for plan in plans)):
                    valid = False
                    break
        if valid:
            pending.batch = batch
        else:
            self._scatter_start_count('eligibility_declines')
            self._scatter_start_count('declined:prepared_batch')
            self.resume_prefill_scatter(after_target=False)

    def resume_prefill_scatter(self, *, after_target, batch=None):
        pending = getattr(self, '_prefill_scatter_deferred', None)
        if pending is None:
            return False
        if after_target and (batch is None or pending.batch is not batch):
            error = RuntimeError('Deferred scatter requires its fresh prepared public target')
            self.abort_prefill_scatter(error)
            raise error
        try:
            started = self.start_cpu_scatter()
        except BaseException as error:
            self.abort_prefill_scatter(error)
            raise
        self._prefill_scatter_deferred = None
        if started:
            self._scatter_start_count('after_target_started_batches' if after_target else
                                      'resumed_before_target_batches')
        else:
            # A prepared-plan fallback can synchronously drain the old owner
            # before recheck. Track the consumed intent even without a worker.
            self._scatter_start_count('no_work_batches')
        return started

    def abort_prefill_scatter(self, error):
        if getattr(self, '_prefill_scatter_deferred', None) is not None:
            # Keep the queue and its DMA/score sources. A failed target never
            # authorizes new publication or a physical-page retirement ACK.
            self._prefill_scatter_deferred = None
            self._scatter_start_count('quarantined_batches')
            self.quarantine_publication(error)

    def advance_publication(self, scheduled_tokens, refresh=None, *, defer_scatter=False):
        """Give confirmed packets to the scatter worker before public compute.

        Large owners advance independently in the background. Tiny owners can
        publish through their foreground request fence. A no-forward iteration
        must also advance those tiny owners, since it has no later submit.
        Maintenance polls only jobs whose own publication fence is complete.
        """
        if self.options.async_cpu_scatter:
            if not defer_scatter:
                self.start_cpu_scatter()
            elif not scheduled_tokens or getattr(self, '_prefill_scatter_deferred', None) is None:
                raise RuntimeError('Scatter deferral needs one qualified public forward')
            # An empty scheduler step must advance tiny confirmed packets too;
            # large packets remain on their owning worker. Lifecycle readers
            # use request fences, not a no-forward packet-wide barrier.
            if not scheduled_tokens:
                self.drain_requests(self._small_pending_requests(), wait=False)
        else:
            self.drain(wait=False)
        if refresh is not None and refresh.jobs:
            self.drain_requests(set(refresh.jobs) & self._small_pending_requests(), wait=False)
            # pump checks each owner's publication fence before it reads any
            # mutable authority; an unrelated ready job can still advance.
            refresh.pump(tuple(scheduled_tokens))

    def _small_pending_requests(self):
        with self._publication_lock():
            return {request_id for packet in self.queue if packet.confirmed
                    for request_id in packet.request_ids} - self._large_pending_requests()

    def start_cpu_scatter(self):
        """Called only after applying the next scheduler's EOS/cancel decision.

        The worker claims one confirmed request at a time. Foreground callers
        can publish unrelated owners, including those sharing its DMA slot.
        Newly sampled packets remain unconfirmed until the next begin_step.
        """
        self._raise_publication_error()
        future = getattr(self, '_cpu_scatter_future', None)
        if future is not None:
            if not future.done():
                return False
            self.join_cpu_scatter()
        with self._publication_lock():
            queued = [packet for packet in self.queue if getattr(packet, 'confirmed', True)]
        if not self.options.async_cpu_scatter or not queued:
            return False
        feedback = any(self.early_capture_packet(packet) and
                       (packet.source_copied or packet.query_inputs) for packet in queued)
        feedback = feedback or any(getattr(packet, 'action_validation', None) is not None
                                   for packet in queued)
        if sum(packet.tokens for packet in queued)<self.options.cpu_scatter_min_tokens and not feedback:
            # Tiny decode packets cost less than a worker handoff/GIL contention.
            # submit drains them after current GPU work has been enqueued; keep
            # the same bounded DMA slots and next-scheduler terminal barrier.
            return False
        if getattr(self,'_cpu_scatter_executor',None) is None:
            self._cpu_scatter_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='kvmem-scatter')
        self.cpu_scatter_batches=getattr(self,'cpu_scatter_batches',0)+len(queued)
        def work():
            started=time.perf_counter()
            self._cpu_scatter_thread=get_ident()
            try:
                return self.drain(wait=True)
            except BaseException as error:
                with self._publication_lock():
                    self._publication_error = error
                    self._publication_condition.notify_all()
                raise
            finally:
                self._cpu_scatter_thread=None
                self.cpu_scatter_seconds=getattr(self,'cpu_scatter_seconds',0.0)+time.perf_counter()-started
        self._cpu_scatter_future=self._cpu_scatter_executor.submit(work)
        return True

    def join_cpu_scatter(self):
        future=getattr(self,'_cpu_scatter_future',None)
        if future is None:
            return
        started=time.perf_counter()
        try:
            future.result()
            self._cpu_scatter_future=None
        finally:
            self.cpu_scatter_join_seconds=getattr(self,'cpu_scatter_join_seconds',0.0)+time.perf_counter()-started

    def ensure_authority(self, request_id):
        with self._publication_lock():
            return self._ensure_authority_locked(request_id)

    def _ensure_authority_locked(self, request_id):
        authority = self.authorities.get(request_id)
        if authority is None:
            sessions = getattr(self.runner,'_kvmem_sessions',None)
            graph_enabled = (getattr(self.options, 'graph_mode', 'off') != 'off' and
                             request_id not in getattr(sessions, 'release_requests', ()))
            boundary_enabled = (getattr(self.options, 'historical_gc', None) is not None and
                self.options.query_score_mode == 'key_fps_4' and not graph_enabled and
                request_id not in getattr(sessions, 'release_requests', ()))
            binding = sessions.bindings.get(request_id) if sessions is not None else None
            if (graph_enabled or boundary_enabled) and binding is None:
                # Never allocate authority state for a graph with temporary
                # request ownership; its logical identities would be unstable.
                raise ValueError('Block graph requires an owned session binding')
            from .authority import MeanIndexStats
            if getattr(self, 'cpu_mean_index_stats', None) is None:
                self.cpu_mean_index_stats = MeanIndexStats()
            authority = RequestAuthority(request_id, self.config, self.archive,
                len(self.layers), self.heads, self.dim, raw_dtype=self.raw_dtype,
                resident_records=self.resident_records,
                gc_config=self.gc_config, gc_progress=self.gc_progress,
                scoring_layers=len(self.runner._kvmem_layers),mean_k_backend=self.options.mean_k_backend,
                query_score_mode=getattr(self.options, 'query_score_mode', 'mean_k'),
                query_window_cache_bytes=getattr(self.options, 'query_window_cache_bytes', 2147483648),
                cpu_mean_index_simd=getattr(self.options, 'cpu_mean_index_simd', False),
                cpu_release_sealed_sums=getattr(self.options, 'cpu_release_sealed_sums', False),
                query_fps_cache_fp8=getattr(self.options, 'query_fps_cache_fp8', False),
                mean_index_stats=self.cpu_mean_index_stats)
            authority.graph_options = self.options
            if (getattr(self, 'redundancy_enabled', False) and
                    request_id not in getattr(sessions, 'release_requests', ())):
                if binding is None or not getattr(authority.gc, 'native_history', False):
                    authority.close()
                    raise ValueError('Redundancy requires native history and an owned session binding')
                from .historical_gc import RedundancyHistory
                authority.redundancy_history = RedundancyHistory(authority.gc.scope,
                    len(self.runner._kvmem_layers), self.heads, self.redundancy_query_heads, self.dim)
            if boundary_enabled:
                from .query_boundary_history import QueryBoundaryHistory
                from .prompt_spans import install_selection_metadata
                try:
                    authority.query_boundaries = QueryBoundaryHistory(
                        (binding.namespace, binding.generation),
                        block_tokens=self.config.block_tokens,
                        max_spans=self.config.repository_blocks*self.config.block_tokens)
                    authority.graph_report_path = self.output/'block_graph.jsonl'
                    install_selection_metadata(authority, binding, validated_prefix=0)
                except BaseException:
                    authority.close()
                    raise
            if getattr(self.options, 'query_fps_prebuild', False):
                if getattr(self, 'fps_prebuild_manager', None) is None:
                    from .query_fps_prebuild import FPSPrebuildManager
                    device = (self.runner.device if self.options.query_fps_prebuild_device == 'cuda'
                              else 'cpu')
                    self.fps_prebuild_manager = FPSPrebuildManager(device=device,
                        cpu_backend=getattr(self.options, 'query_fps_cpu_backend', 'torch'),
                        cpu_workers=getattr(self.options, 'query_fps_cpu_workers', 8))
                authority.fps_prebuild = self.fps_prebuild_manager.bind(authority)
            if getattr(self,'action',None) is not None:
                from .action_attention import ActionHistory
                authority.action_history = ActionHistory(self.options,self.action.ids)
            if graph_enabled:
                from .block_graph import BlockGraph
                try:
                    authority.block_graph = BlockGraph((binding.namespace,binding.generation),
                        self.config.block_tokens,topk=self.options.graph_attention_topk,
                        min_mass=self.options.graph_attention_min_mass,
                        max_bytes_per_session=self.options.graph_max_bytes_per_session,
                        budget=self.block_graph_budget)
                    authority.graph_options = self.options
                    authority.graph_report_path = self.output/'block_graph.jsonl'
                    from .prompt_spans import install_graph_metadata
                    install_graph_metadata(authority.block_graph, binding, validated_prefix=0)
                except BaseException:
                    graph = getattr(authority,'block_graph',None)
                    if graph is not None:
                        graph.close()
                    authority.close()
                    raise
            if self.options.query_history and sessions is not None and request_id in sessions.bindings:
                from .query_history import QueryRelevance
                authority.query_history = QueryRelevance(self.options.query_history_decay)
            self.authorities[request_id] = authority
            self.accepted[request_id] = self.rejected[request_id] = 0
            self.last_steps[request_id] = []
            self._snapshot_capacity(request_id, authority)
        return authority

    def _snapshot_capacity(self, request_id, authority):
        from .archive_admission import ArchiveCapacity
        self._published_capacity[request_id] = ArchiveCapacity(
            authority.history_end, authority.memory.repository.ready_slots)

    def drain(self, wait=False):
        """Explicit all-request barrier; public paths use drain_requests."""
        return self.drain_requests(None, wait=wait)

    def _claim_publications(self, wanted, wait):
        """Claim independent owners; never hold the condition during CPU copy."""
        blocked = set()
        pending = False
        unconfirmed = False
        background = getattr(self, '_cpu_scatter_thread', None) == get_ident()
        for packet in self.queue:
            indices = []
            for i, request_id in enumerate(packet.request_ids):
                if i in packet.published or (wanted is not None and request_id not in wanted):
                    continue
                pending = True
                earlier = request_id in blocked
                blocked.add(request_id)
                if not packet.confirmed:
                    unconfirmed = True
                    continue
                if earlier or i in packet.consuming:
                    continue
                validation = getattr(packet, 'action_validation', None)
                if validation is not None and not wait and not validation.ready:
                    continue
                if not wait and not packet.source_released and not packet.download.query():
                    continue
                indices.append(i)
                # Large background requests have independent publication
                # fences. A short foreground decode may still batch its peers.
                if background or not self.options.batched_cpu_append:
                    break
            if indices:
                packet.consuming.update(indices)
                return (packet, indices), pending, unconfirmed
        return None, pending, unconfirmed

    def drain_requests(self, request_ids, wait=False):
        """Publish only the selected requests, preserving each owner's FIFO.

        Another thread may publish an unrelated owner from the same DMA packet.
        Bulk source release can precede scores only after every owner is copied;
        owner publication still waits for its scores. A false result means
        an unready DMA/capacity/in-flight owner, or an unconfirmed sample; even
        wait=True never bypasses the scheduler-confirmation boundary.
        """
        wanted = None if request_ids is None else set(request_ids)
        condition = self._publication_lock()
        while True:
            with condition:
                self._raise_publication_error()
                claim, pending, unconfirmed = self._claim_publications(wanted, wait)
                if claim is None:
                    if not pending:
                        return True
                    if not wait or unconfirmed:
                        return False
                    condition.wait()
                    continue
            packet, indices = claim
            started = time.perf_counter()
            try:
                consumed = self._publish_claim(packet, indices, wait)
            except BaseException as error:
                with condition:
                    self._publication_error = error
                    # Failed owners keep their DMA source alive. Every waiter
                    # receives the same fatal publication failure.
                    condition.notify_all()
                raise
            finally:
                self._publication_metric('claim_seconds', time.perf_counter()-started, lane=packet.lane)
                self._publication_metric('claim_calls', 1, lane=packet.lane)
            with condition:
                if getattr(packet,'logical_group',None) is not None and consumed:
                    # A logical owner's publication ACK follows successful
                    # source release. A failed finalizer keeps it quarantined
                    # and cannot make publication_ready report completion.
                    try:
                        if consumed != 1 or indices != [0] or not packet.logical_cursor.done:
                            raise RuntimeError('Incomplete logical owner reached source release')
                        if packet.logical_cursor is not None:
                            packet.logical_cursor.release()
                            packet.logical_payload = None
                            packet.logical_slices = ()
                            packet.action_inputs.clear()
                        if not self._download_pool(packet.lane).release(packet.download):
                            raise RuntimeError('Completed logical owner retained its DMA lease')
                    except BaseException as error:
                        packet.logical_cursor.fail(error)
                        self._publication_error = error
                        condition.notify_all()
                        raise
                    packet.published.update(indices)
                    packet.consuming.difference_update(indices)
                    packet.next_request = 1
                    self.queue[:] = [queued for queued in self.queue if queued is not packet]
                else:
                    packet.published.update(indices[:consumed])
                    packet.consuming.difference_update(indices)
                    while packet.next_request in packet.published:
                        packet.next_request += 1
                    if len(packet.published) == len(packet.request_ids):
                        assert not packet.consuming
                        if self.early_capture_packet(packet):
                            self._finish_early_packet(packet)
                        else:
                            assert self._download_pool(packet.lane).release(packet.download)
                        self.queue[:] = [queued for queued in self.queue if queued is not packet]
                condition.notify_all()
            if not consumed and not wait:
                return False
            # Only the engine thread writes diagnostics. It snapshots published
            # counters and never traverses an in-flight authority's records.
            if (getattr(self, '_cpu_scatter_thread', None) != get_ident() and
                    self.batches // 8 != self._progress_batch):
                self._progress_batch = self.batches // 8
                self._write_progress()

    @staticmethod
    def _graph_reservation(prepared):
        return getattr(prepared, 'graph_update', prepared)

    def _prepare_graph_acceptance(self, packet, index, accepted, history_begin):
        if getattr(packet,'logical_group',None) is not None:
            raise RuntimeError('A physical logical-prefill packet requires ordered child transactions')
        request_id = packet.request_ids[index]
        graph = getattr(self.authorities[request_id], 'block_graph', None)
        if graph is None:
            return None
        action = getattr(self, 'action', None)
        if action is not None:
            return action.prepare_accept(packet, request_id, accepted,
                                         graph=graph, history_begin=history_begin)
        begin = min(history_begin, graph.accepted_through) if not accepted else history_begin
        return graph.prepare(begin, begin + accepted)

    def _abort_graph_updates(self, packet, indices):
        updates = getattr(packet, 'graph_updates', None)
        if updates is None:
            return
        for index in indices:
            prepared = updates.pop(index, None)
            if prepared is not None:
                prepared.abort()
        if not updates:
            packet.graph_updates = None

    def _publish_claim(self, packet, indices, wait):
        try:
            if getattr(packet,'logical_group',None) is not None:
                return self._publish_logical_claim(packet,indices,wait)
            return self._publish_reserved_claim(packet, indices, wait)
        except BaseException:
            self._abort_graph_updates(packet, indices)
            raise

    def _publish_logical_claim(self, packet, indices, wait):
        """One claimed physical owner, bounded original reserve/append/commit steps."""
        from .logical_publication import LogicalPublicationCursor
        group = packet.logical_group
        if (list(indices) != [0] or packet.request_ids != (group.request_id,) or
                packet.tokens != group.count or packet.query_inputs or packet.action_validation is not None or
                self.early_capture_packet(packet) or packet.source_released):
            raise RuntimeError('Logical publication lost its qualified physical owner')
        cursor = packet.logical_cursor
        try:
            if not packet.download.query():
                if not wait:
                    return 0
                started = time.perf_counter()
                try:
                    packet.download.event.synchronize()
                finally:
                    self._publication_metric('dma_wait_seconds',time.perf_counter()-started,lane=packet.lane)
                    self._publication_metric('dma_wait_calls',1,lane=packet.lane)
            if packet.logical_payload is None:
                data = packet.download.payload
                header_bytes = 5*8
                raw_bytes = self.row_elements*group.count*self.raw_dtype.itemsize
                value_bytes = self.row_elements*group.count
                if data.numel() != header_bytes+raw_bytes+value_bytes:
                    raise ValueError('Logical physical payload has an invalid byte extent')
                header = data[:header_bytes].view(torch.int64).tolist()
                if header != [0,group.count,group.history_begin,group.local_begin,0]:
                    raise ValueError('Logical physical metadata differs from the admitted non-rejected span')
                shape = (len(self.layers),group.count,self.heads,self.dim)
                packet.logical_payload = (data[header_bytes:header_bytes+raw_bytes].view(self.raw_dtype).reshape(shape),
                                          data[header_bytes+raw_bytes:].reshape(shape))
            authority = self.ensure_authority(group.request_id)
            action = self.action
            if cursor is None:
                epoch = action._prefill_identity(group.request_id,authority.action_history)
                cursor = packet.logical_cursor = LogicalPublicationCursor(group,packet,authority,action,
                    packet.logical_slices,owner_epoch=epoch,source_generation=packet.source_generation)
            elif cursor.authority is not authority:
                raise RuntimeError('Logical publication authority was replaced')
            while not cursor.done:
                with self._publication_lock():
                    confirmation = packet.logical_confirmation
                if confirmation is None:
                    raise RuntimeError('Logical publication lacks a scheduler receipt')
                epoch = action._prefill_identity(group.request_id,authority.action_history)
                cursor.confirm(confirmation[0],owner_epoch=epoch,terminal_prefix=confirmation[1],
                               native_accepted=group.count)
                started = time.perf_counter()
                receipt = cursor.advance(*packet.logical_payload,owner_epoch=epoch)
                self._publication_metric('logical_transaction_seconds',time.perf_counter()-started,lane=packet.lane)
                if receipt is None:
                    if not wait:
                        return 0
                    self._wait_archive_progress({group.request_id})
                    continue
                # Only complete child transactions reach bookkeeping. A second
                # child failure preserves this prefix and never replays it.
                with self._publication_lock():
                    rid = group.request_id
                    # A cancellation can arrive while native append releases
                    # the GIL. Reconcile it before exposing this child receipt,
                    # including the final child which has no next loop visit.
                    latest = packet.logical_confirmation
                    if latest is None:
                        raise RuntimeError('Logical scheduler receipt disappeared during append')
                    cursor.confirm(latest[0],owner_epoch=epoch,terminal_prefix=latest[1],
                                   native_accepted=group.count)
                    self.accepted[rid] += receipt.accepted
                    self.rejected[rid] += receipt.rejected
                    self.terminal_discarded[rid] = self.terminal_discarded.get(rid,0)+receipt.terminal_discarded
                    steps = self.last_steps[rid]
                    steps.append(dict(history_begin=receipt.history_begin,scheduled=receipt.scheduled,
                        accepted=receipt.accepted,rejected=receipt.rejected,
                        terminal_discarded=receipt.terminal_discarded))
                    if receipt.owner_done:
                        # GDN/MTP checkpoint consumers need the actual physical
                        # forward extent. Keep original 2K transaction fields
                        # for logical diagnostics and attach its separate view.
                        steps[-1]['physical_step'] = dict(history_begin=group.history_begin,
                            scheduled=group.count,accepted=cursor.retained,rejected=0,
                            terminal_discarded=group.count-cursor.retained)
                    del steps[:max(0,len(steps)-self.runner.vllm_config.max_concurrent_batches-1)]
                    self._snapshot_capacity(rid,authority)
                    self.batched_append_calls = getattr(self,'batched_append_calls',0)+receipt.append_calls
                    self.batched_append_requests = getattr(self,'batched_append_requests',0)+receipt.append_requests
                    self._publication_metric('published_owners',1,lane=packet.lane)
                    self._publication_metric('published_tokens',receipt.accepted,lane=packet.lane)
                    self._publication_metric('logical_prefill_commits',1,lane=packet.lane)
                    if receipt.owner_done:
                        self._record_logical_publication(packet,cursor,latest[0])
                    cursor.ack(receipt)
            with self._publication_lock():
                getattr(self,'_archive_request_deadlines',{}).pop(frozenset(packet.request_ids),None)
            self._publication_metric('logical_prefill_physical_owners',1,lane=packet.lane)
            return 1
        except BaseException as error:
            if cursor is not None:
                cursor.fail(error)
            raise

    def _record_logical_publication(self, packet, cursor, confirmation_step):
        """Audit each completed physical group before its final acknowledgement.

        This small write is part of publication cost. A write failure follows
        the same fail-stop path as other bookkeeping after a child commit;
        the physical download is retained and never replayed or released.
        """
        group = packet.logical_group
        spans = []
        for part in packet.logical_slices:
            plan,span = part.plan,part.plan.span
            spans.append(dict(history_begin=span.history_begin,local_begin=span.local_begin,
                count=span.count,sampled_rows=len(plan.rows),segments=len(plan.starts),
                sampled_row_indices=plan.rows))
        row = dict(request_id=group.request_id,scheduler_step=group.scheduler_step,
            confirmation_step=confirmation_step,history_begin=group.history_begin,
            local_begin=group.local_begin,count=group.count,retained=cursor.retained,
            spans=spans,source_generation=packet.source_generation)
        with (self.output/'logical_prefill_publication.jsonl').open('a') as stream:
            stream.write(json.dumps(row,separators=(',',':'))+'\n')

    def _publish_reserved_claim(self, packet, indices, wait):
        metadata = self._copy_claim_metadata(packet, indices, wait)
        for i, accepted, discarded, scheduled, history_begin, rejected in metadata:
            request_id = packet.request_ids[i]
            if self.early_capture_packet(packet):
                self._early_event('semantic_owner_begin', packet, owner=request_id)
            action = getattr(self, 'action', None)
            updates = getattr(packet, 'graph_updates', None)
            prepared = updates.pop(i, None) if updates is not None else None
            if updates is not None and not updates:
                packet.graph_updates = None
            if prepared is None:
                # Compatibility for diagnostic/test metadata sources which
                # bypass raw archive copying. Production reserves before append.
                prepared = self._prepare_graph_acceptance(packet, i, accepted, history_begin)
            try:
                if prepared is not None:
                    # Early source copy can run on a different thread. Resume
                    # and validate ownership before Query/parser/legacy mutation.
                    self._graph_reservation(prepared).resume()
                started = time.perf_counter()
                self._accept_query_input(packet, request_id)
                self._publication_metric('owner_query_seconds', time.perf_counter()-started, lane=packet.lane)
                if prepared is not None:
                    started = time.perf_counter()
                    prepared.commit()
                    self._publication_metric('owner_action_seconds', time.perf_counter()-started, lane=packet.lane)
                elif action is not None:
                    started = time.perf_counter()
                    action.accept(packet, request_id, accepted)
                    self._publication_metric('owner_action_seconds', time.perf_counter()-started, lane=packet.lane)
                self._accept_redundancy_queries(packet, request_id, history_begin, accepted)
                self._accept_pressure_queries(packet, request_id, history_begin, accepted)
            except BaseException:
                if prepared is not None:
                    prepared.abort()
                raise
            self._publication_metric('published_owners', 1, lane=packet.lane)
            self._publication_metric('published_tokens', accepted, lane=packet.lane)
            with self._publication_lock():
                self.accepted[request_id] += accepted
                self.rejected[request_id] += rejected
                self.terminal_discarded[request_id] = self.terminal_discarded.get(request_id, 0) + discarded
                steps = self.last_steps[request_id]
                steps.append(dict(history_begin=history_begin, scheduled=scheduled, accepted=accepted,
                                  rejected=rejected, terminal_discarded=discarded))
                del steps[:max(0, len(steps)-self.runner.vllm_config.max_concurrent_batches-1)]
                self._snapshot_capacity(request_id, self.authorities[request_id])
                if self.early_capture_packet(packet):
                    self._early_event('semantic_owner_committed', packet, owner=request_id)
        if metadata:
            with self._publication_lock():
                getattr(self, '_archive_request_deadlines', {}).pop(
                    frozenset(packet.request_ids[i] for i in indices), None)
        return len(metadata)

    def _copy_claim_metadata(self, packet, indices, wait):
        try:
            return self._copy_reserved_metadata(packet, indices, wait)
        except BaseException:
            self._abort_graph_updates(packet, indices)
            raise

    def _copy_reserved_metadata(self, packet, indices, wait):
        if not self.early_capture_packet(packet):
            return self._copy_source_owners(packet, indices, wait)
        missing = [i for i in indices if i not in packet.source_copied]
        if missing:
            if packet.source_released:
                raise RuntimeError('Released DMA source still has uncopied owners')
            started = time.perf_counter()
            metadata = self._copy_source_owners(packet, missing, wait)
            # _copy_source_owners has returned: no local tensor view of staging
            # survives in these integer tuples. Async seal owns record copies.
            with self._publication_lock():
                for row in metadata:
                    i = row[0]
                    packet.source_feedback_bytes[i] = self._feedback_bytes(packet, i)
                    packet.source_copied[i] = row
                self._publication_metric('source_copy_seconds', time.perf_counter()-started, lane=packet.lane)
                self._publication_metric('source_copied_owners', len(metadata), lane=packet.lane)
                self._early_event('source_copied', packet,
                                  copied_indices=[row[0] for row in metadata],
                                  feedback_bytes=sum(packet.source_feedback_bytes.values()),
                                  owner_storage_bytes={packet.request_ids[row[0]]:
                                      self._feedback_storage_bytes(packet, row[0]) for row in metadata})
                self._try_release_source(packet)
        result = []
        for i in indices:
            if i not in packet.source_copied:
                break
            result.append(packet.source_copied[i])
        return result

    def _copy_source_owners(self, packet, indices, wait):
        validation = getattr(packet, 'action_validation', None)
        if validation is not None and not validation.resolve(wait=wait):
            return []
        if not packet.download.query():
            started = time.perf_counter()
            try:
                packet.download.event.synchronize()
            finally:
                self._publication_metric('dma_wait_seconds', time.perf_counter()-started, lane=packet.lane)
                self._publication_metric('dma_wait_calls', 1, lane=packet.lane)
        n, t = len(packet.request_ids), packet.tokens
        data = packet.download.payload
        header_bytes = (4 * n + 1) * 8
        header = data[:header_bytes].view(torch.int64).tolist()
        starts, histories = header[:n+1], header[n+1:2*n+1]
        local, rejected = header[2*n+1:3*n+1], header[3*n+1:]
        shape = (len(self.layers), t, self.heads, self.dim)
        raw_bytes = self.row_elements * t * self.raw_dtype.itemsize
        raw = data[header_bytes:header_bytes+raw_bytes].view(self.raw_dtype).reshape(shape)
        value_end = header_bytes+raw_bytes+self.row_elements*t
        values = data[header_bytes+raw_bytes:value_end].reshape(shape)
        entries, metadata = [], []
        for i in indices:
            request_id = packet.request_ids[i]
            authority = self.ensure_authority(request_id)
            begin, end = starts[i:i+2]
            accepted = end - begin - rejected[i]
            if not 0 <= accepted <= end - begin:
                raise ValueError('Native sampling rejection count exceeds captured rows')
            terminal = self.terminal_prefixes.get(request_id)
            retained = accepted if terminal is None else max(0, min(accepted, terminal.history_end-histories[i]))
            if getattr(self, 'pressure_query_enabled', False):
                query = packet.pressure_queries.get(request_id)
                window = getattr(authority, 'pressure_query_window', None)
                if query is None or window is None:
                    raise RuntimeError('Public capture lacks its accepted pressure Query metadata')
                window.validate_epoch(query['epoch'])
            if getattr(self, 'redundancy_enabled', False):
                query = packet.redundancy_queries.get(request_id)
                if query is None:
                    raise RuntimeError('Public capture lacks its redundancy Q metadata')
                if query['binding'] is not None:
                    authority.redundancy_history.begin(query['binding'], histories[i])
            entries.append((authority, raw[:, begin:end], values[:, begin:end],
                dict(history_begin=histories[i], local_rope_begin=local[i], accepted=retained)))
            metadata.append((i, retained, accepted-retained, end-begin, histories[i], rejected[i]))
        try:
            # Preflight every owner before native append can reserve/commit any
            # raw KV. Failed graph capacity leaves all history and parser state
            # unchanged. This is deliberately before append_authority_batch.
            for i, retained, _discarded, _scheduled, history_begin, _rejected in metadata:
                if getattr(packet, 'graph_updates', None) is not None and i in packet.graph_updates:
                    raise RuntimeError('Capture owner already has a graph reservation')
                prepared = self._prepare_graph_acceptance(packet, i, retained, history_begin)
                if prepared is not None:
                    self._graph_reservation(prepared).detach()
                    if getattr(packet, 'graph_updates', None) is None:
                        packet.graph_updates = {}
                    packet.graph_updates[i] = prepared
            while True:
                started = time.perf_counter()
                try:
                    if self.options.batched_cpu_append:
                        from .authority import append_authority_batch
                        consumed = append_authority_batch(entries)
                        if consumed:
                            self.batched_append_calls = getattr(self, 'batched_append_calls', 0) + 1
                            self.batched_append_requests = getattr(self, 'batched_append_requests', 0) + consumed
                    else:
                        authority, request_raw, request_values, kwargs = entries[0]
                        consumed = int(not kwargs['accepted'] or authority.append(request_raw, request_values, **kwargs))
                finally:
                    self._publication_metric('append_seconds', time.perf_counter()-started, lane=packet.lane)
                if consumed or not wait:
                    break
                self._wait_archive_progress({packet.request_ids[i] for i in indices})
            self._abort_graph_updates(packet, indices[consumed:])
            return metadata[:consumed]
        except BaseException:
            self._abort_graph_updates(packet, indices)
            raise

    def _write_progress(self):
        lock = self.__dict__.setdefault('_progress_lock', RLock())
        with lock, self._publication_lock():
            self._write_progress_locked()

    def _write_progress_locked(self):
        busy = self._busy_requests()
        progress = {"batches_submitted": self.batches,
            'redundancy_q_capture': dict(getattr(self, 'redundancy_q_stats', {})),
            'pressure_q_capture': dict(getattr(self, 'pressure_query_stats', {})),
            'batched_append_calls': getattr(self, 'batched_append_calls', 0),
            'batched_append_requests': getattr(self, 'batched_append_requests', 0),
            'full_graph_decode_calls': getattr(self.runner, '_kvmem_graph_decode_calls', 0),
            "accepted_history": {req: (self._published_capacity[req].history_end
                if req in busy and req in self._published_capacity else owner.history_end)
                for req, owner in self.authorities.items()},
            "archive_used_slots": self.archive.tier.used_slots,
            'historical_gc': self.archive.history_stats() if hasattr(self.archive, 'history_stats') else None,
            "pending_downloads": self._pending_download_count(),
            'split_capture_dma': getattr(self, 'split_capture_dma', False),
            'publication_metrics': self._publication_metrics_snapshot(),
            'prefill_publication_pipeline': self.prefill_pipeline_enabled,
            'prefill_capture_ahead': getattr(self.options, 'prefill_capture_ahead', False),
            'capture_fused_pack': getattr(self.options, 'capture_fused_pack', False),
            'capture_pack_stats': dict(getattr(self, 'capture_pack_stats', {})),
            'capture_opaque_projection': getattr(self.options, 'capture_opaque_projection', False),
            'capture_opaque_stats': (self.runner._kvmem_capture_opaque.snapshot()
                if getattr(self.runner, '_kvmem_capture_opaque', None) is not None else {}),
            'cpu_mean_index_simd': getattr(self.options, 'cpu_mean_index_simd', False),
            'query_score_mode': getattr(self.options, 'query_score_mode', 'mean_k'),
            'query_mean_quota_fraction': getattr(self.options, 'query_mean_quota_fraction', 0.),
            'query_window_cache_bytes': getattr(self.options, 'query_window_cache_bytes', 2147483648),
            'cpu_mean_index_stats': (self.cpu_mean_index_stats.snapshot()
                if getattr(self, 'cpu_mean_index_stats', None) is not None else {}),
            'prefill_action_before_mtp': getattr(self.options, 'prefill_action_before_mtp', False),
            'prefill_action_before_mtp_stats': dict(getattr(getattr(self, 'action', None), 'mtp_stage_stats', {})),
            'prefill_deferred_validation': getattr(self.options, 'prefill_deferred_validation', False),
            'prefill_scatter_after_target': getattr(self.options, 'prefill_scatter_after_target', False),
            'prefill_scatter_after_target_stats': dict(getattr(self, 'prefill_scatter_after_target_stats', {})),
            'prefill_deferred_validation_pool': (self.action.deferred_score_pool.snapshot()
                if getattr(getattr(self, 'action', None), 'deferred_score_pool', None) is not None else None),
            'capture_pack_metadata_capacity_bytes': (self._capture_pack_cache.limits.metadata_bytes
                if getattr(self, '_capture_pack_cache', None) is not None else 0),
            'prefill_pipeline_stats': dict(getattr(self, 'prefill_pipeline_stats', {})),
            'logical_prefill_stats': {'enabled':getattr(self.options,'prefill_physical_coalescing',False),
                **dict(getattr(getattr(self,'action',None),'logical_prefill_stats',{}))},
            'publication_metrics_scope': 'Cumulative host durations by thread/lane; overlapping, not GPU idle. claim includes append and owner scores; slot wait may include claims.',
            'capture_staging_capacity_bytes': sum(getattr(pool, 'capacity_bytes', 0)
                for pool in (self.downloads, getattr(self, 'decode_downloads', None)) if pool is not None),
            'cpu_scatter_batches':getattr(self,'cpu_scatter_batches',0),
            'cpu_scatter_seconds':getattr(self,'cpu_scatter_seconds',0.0),
            'cpu_scatter_join_seconds':getattr(self,'cpu_scatter_join_seconds',0.0),
            **completion_counts(self.completed),
            "retiring_requests": len(getattr(self, 'retiring', {}))}
        if self.early_capture_enabled:
            self._sample_early_publication(progress)
        temporary = self.output / "progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(self.output / "progress.json")

    def archive_capacity(self):
        """Nonblocking worker feedback, including DMA not yet incorporated.

        The scheduler debits any accepted/computing suffix beyond history_end;
        snapshotting an unfinished download cannot invent free archive space.
        """
        from .archive_admission import ArchiveCapacity
        # Do not drain the latest sample here: native EOS clipping is decided
        # by the scheduler after this output. execute_model applies that boundary
        # before draining on the following iteration.
        with self._publication_lock():
            self._raise_publication_error()
            busy = self._busy_requests()
            result = {}
            for request_id, authority in self.authorities.items():
                if request_id not in busy and not getattr(authority, '_refresh_background', False):
                    authority.poll()
                    self._snapshot_capacity(request_id, authority)
                # An in-flight append may have reserved rows but cannot make
                # its history/capacity visible before Action and Query commit.
                # The scheduler debits its suffix against this older prefix.
                value = self._published_capacity.get(request_id)
                result[request_id] = value if value is not None else ArchiveCapacity(0, 0)
            return result

    def _wait_archive_progress(self, request_ids=None):
        # Shared queue pressure may come from another request, even when this
        # authority has not admitted its first write yet.
        with self._publication_lock():
            owners = [a for request_id, a in self.authorities.items()
                if (request_ids is None or request_id in request_ids)
                and not getattr(a, '_refresh_background', False)]
        progressed = sum(authority.poll() for authority in owners)
        jobs = [job for authority in owners for job, _ in authority.writes]
        if not jobs:
            if progressed:
                if request_ids is None:
                    self._archive_wait_deadline = None
                else:
                    getattr(self, '_archive_request_deadlines', {}).pop(frozenset(request_ids), None)
                return
            if (request_ids is not None or getattr(self, 'retiring', {}) or
                    getattr(getattr(getattr(self, 'runner', None), '_kvmem_sessions', None), 'retiring', {})):
                # Detached finalizers share the same count/byte queue. Their
                # I/O is absent from active authority.writes; do not report a
                # false repository-capacity failure while they still own it.
                if self.archive.io_usage()['archive_io_pending_bytes']:
                    if not self.archive.wait_for_idle(30):
                        raise TimeoutError('Detached archive I/O did not release capacity')
                    return
            if self.options.repository_gc:
                # The CPU policy worker may be reclaiming durable generations
                # even with no outstanding archive write. Never run policy on
                # this engine thread or silently discard captured accepted rows.
                # This synchronous fallback is measured; scheduler-side capacity
                # feedback is still required to remove such serving stalls.
                started = time.monotonic()
                if request_ids is None:
                    self._archive_wait_deadline = self._archive_wait_deadline or started + 30
                    deadline = self._archive_wait_deadline
                else:
                    with self._publication_lock():
                        deadlines = self.__dict__.setdefault('_archive_request_deadlines', {})
                        deadline = deadlines.setdefault(frozenset(request_ids), started+30)
                if started >= deadline:
                    raise TimeoutError('Repository GC made no append progress for 30 seconds')
                self.gc_progress.wait(min(.05, deadline - started))
                self.gc_progress.clear()
                self.gc_wait_seconds += time.monotonic() - started
                self.gc_wait_count += 1
                return
            raise BufferError("Authority capacity exhausted; scheduler admission/GC must yield")
        for job in jobs:
            job.result()

    def finish(self, request_ids, terminal_prefixes=None):
        """Drain publication, optionally checksum records, then reclaim the request."""
        if not request_ids:
            return
        with self._publication_lock():
            self._install_terminals(request_ids, terminal_prefixes)
        if not self.drain_requests(request_ids, wait=True):
            raise RuntimeError('Terminal capture was not confirmed by begin_step')
        for request_id in request_ids:
            authority = self.authorities.get(request_id)
            if authority is None:
                continue
            terminal = getattr(self, 'terminal_prefixes', {}).pop(request_id, None)
            if terminal is not None and authority.history_end != terminal.history_end:
                raise RuntimeError('Final capture does not match scheduler terminal prefix')
            terminal_report = {
                'terminal_prefix': asdict(terminal) if terminal is not None else None,
                'terminal_discarded_tokens': getattr(self, 'terminal_discarded', {}).pop(request_id, 0)}
            if getattr(authority, 'fps_prebuild', None) is not None:
                terminal_report['query_fps_prebuild'] = authority.fps_prebuild.snapshot()
            if getattr(self,'action',None) is not None:
                terminal_report['action_attention'] = self.action.finish(request_id,authority)
            sessions = getattr(self.runner, '_kvmem_sessions', None)
            if sessions is not None and sessions.detach(request_id, authority, {
                    'request_id': request_id, 'accepted_tokens': self.accepted[request_id],
                    'rejected_tokens': self.rejected[request_id],
                    'last_steps': self.last_steps[request_id], **terminal_report}):
                with self._publication_lock():
                    del self.authorities[request_id], self.accepted[request_id], self.rejected[request_id]
                    del self.last_steps[request_id]
                    self._published_capacity.pop(request_id, None)
                    getattr(self, '_deferred_terminals', {}).pop(request_id, None)
                continue
            from .gpu_raw_cache import retire_raw_cache
            retire_raw_cache(self.runner, authority)
            if self.options.async_archive_retirement:
                # All GPU capture packets have drained above. Detach ownership
                # before another native batch; this worker touches CPU data only.
                enqueued_at = time.perf_counter()
                self.retirement.submit(request_id, authority)
                self.retiring[request_id] = {"request_id": request_id,
                    "retirement_enqueued_at": enqueued_at,
                    "accepted_tokens": self.accepted.pop(request_id),
                    "rejected_tokens": self.rejected.pop(request_id),
                    "last_steps": self.last_steps.pop(request_id), **terminal_report}
                with self._publication_lock():
                    del self.authorities[request_id]
                    self._published_capacity.pop(request_id, None)
                    getattr(self, '_deferred_terminals', {}).pop(request_id, None)
                continue
            authority.stop_gc()
            if not authority.flush():
                self._wait_archive_progress({request_id})
                if not authority.flush():
                    raise BufferError("Archive cannot finalize complete authority records")
            for job, _ in authority.writes:
                job.result()
            authority.poll()
            published = sum(record.archived for record in authority.records.values())
            if published + authority.gc_reclaimed_records != authority.history_end // self.config.block_tokens:
                raise RuntimeError("Finalized authority has unpublished complete archive records")
            verified = 0
            for record in authority.records.values():
                if record.archived and self.options.verify_archive_on_finish:
                    raw, values = authority.read_record(record.handle)
                    digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()
                                           + values.numpy().tobytes()).hexdigest()
                    if digest != record.digest:
                        raise RuntimeError("Archived raw K/V checksum differs from accepted capture")
                    verified += 1
            self.completed.append({"request_id": request_id, "history_end": authority.history_end,
                "accepted_tokens": self.accepted.pop(request_id),
                "rejected_tokens": self.rejected.pop(request_id),
                "last_steps": self.last_steps.pop(request_id),
                "records": len(authority.records), "sealed_records_archived": published,
                "sealed_records_verified": verified,
                "gc_reclaimed_records": authority.gc_reclaimed_records, **terminal_report})
            if not authority.close():
                raise RuntimeError("Finalized authority still has unexpected live storage leases")
            with self._publication_lock():
                del self.authorities[request_id]
                self._published_capacity.pop(request_id, None)
                getattr(self, '_deferred_terminals', {}).pop(request_id, None)
        self._write_archive_report()

    def retirement_done(self, request_id):
        """Called on the engine thread before the native connector releases pages."""
        if request_id in getattr(self, '_deferred_terminals', {}) or not self.publication_ready((request_id,)):
            return False
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        if sessions is not None:
            retained = sessions.retirement_done(request_id)
            if retained is not None:
                return retained
        if request_id in self.retiring:
            result = self.retirement.poll(request_id)
            if result is None:
                return False
            self.completed.append({**self.retiring.pop(request_id), **result,
                                   "retirement_ack_ready_at": time.perf_counter()})
            self._write_archive_report()
        done = request_id not in self.authorities
        if done and sessions is not None:
            sessions.ordinary_retired(request_id)
        return done

    def shutdown_retirement(self):
        """Graceful engine shutdown drains detached work before releasing storage."""
        self.join_cpu_scatter()
        if getattr(self, 'fps_prebuild_manager', None) is not None:
            self.fps_prebuild_manager.close()
        executor=getattr(self,'_cpu_scatter_executor',None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._cpu_scatter_executor=None
        if not self.options.async_archive_retirement:
            return
        self.drain(wait=True)
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        if sessions is not None:
            deadline = time.monotonic() + 180
            while not sessions.close():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Session checkpoint cleanup did not drain during shutdown')
                self.archive.wait_for_idle(.05)
        self.finish(set(self.authorities))
        for request_id in list(self.retiring):
            self.retirement.jobs[request_id].result(timeout=180)
            assert self.retirement_done(request_id)
        if not self.retirement.close() or not self.archive.close():
            raise RuntimeError('Shutdown retained archive I/O or retirement jobs')

    def _write_archive_report(self):
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        report = {"scope": "continuous target authority; refresh qualification is recorded separately",
                  "batches": self.batches, "completed": list(self.completed),
                  **completion_counts(self.completed),
                  "block_tokens": self.config.block_tokens,
                  "memory_config": asdict(self.config),
                  "cpu_cache_capacity_bytes": self.options.cpu_read_cache_bytes,
                  "cpu_cache_used_bytes": getattr(getattr(self.archive, 'read_cache', None), 'used_bytes', 0),
                  "packed_assembly": self.options.packed_assembly,
                  "direct_pinned_reads": self.options.direct_pinned_reads,
                  "async_archive_retirement": self.options.async_archive_retirement,
                  "terminal_prefix_required": bool(getattr(self.runner, '_kvmem_bounded_target', False)),
                  "repository_gc": self.options.repository_gc,
                  "historical_gc": self.archive.history_stats() if hasattr(self.archive, 'history_stats') else None,
                  "gc_wait_seconds": getattr(self, 'gc_wait_seconds', 0.0),
                  "gc_wait_count": getattr(self, 'gc_wait_count', 0),
                  "retiring_requests": len(getattr(self, 'retiring', {})),
                  "session_capacity": self.options.session_capacity,
                  "session_checkpoint_payload_bytes": sessions.pool.payload_bytes if sessions else 0,
                  "session_checkpoints": len(sessions.pool.entries) if sessions else 0,
                  "session_retiring_requests": len(sessions.retiring) if sessions else 0,
                  "session_restored_tokens": sessions.restored_tokens if sessions else 0,
                  "session_checkpoint_deferrals": sessions.checkpoint_deferrals if sessions else 0,
                  "archive_verification": "full_checksum" if self.options.verify_archive_on_finish else "publication_only",
                  "max_concurrent_batches": self.runner.vllm_config.max_concurrent_batches,
                  "decode_query_len": self.runner.decode_query_len,
                  "active_requests": len(self.authorities),
                  "block_graph_bytes": getattr(getattr(self, 'block_graph_budget', None), 'used', 0),
                  "block_graph_limit_bytes": getattr(getattr(self, 'block_graph_budget', None), 'maximum', 0),
                  "archive_used_slots": self.archive.tier.used_slots,
                  "nvme_capacity_bytes": 0 if self.options.archive_backend=='cpu' else self.archive.capacity_bytes,
                  "nvme_preallocated": self.archive.preallocated,
                  **self.archive.io_usage(),
                  "pending_downloads": self._pending_download_count()}
        temporary = self.output / 'archive.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(self.output / 'archive.json')
        self._write_progress()
