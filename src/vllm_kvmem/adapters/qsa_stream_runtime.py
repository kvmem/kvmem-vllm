"""qsa_streaming worker runtime (Flash Next).

Each streaming request owns ``hot_pages + 1`` native pages per QSA layer: hot
pages H0..H{h-1}, then the ring page R. Its frame K/V (target and MTP layers)
lives in a pinned host pool, laid out ``[slot][layer][K|V][frame token][bytes]``.
Selector keys stay resident in carved slot pages (adapters/qsa_streaming.py), so
QSA's selection never depends on which K/V pages are resident.

Writes: the worker block-table row of a streaming request maps every frame
column to R, so every K/V write lands at ``R[pos % page]``; the hook copies the
step's rows back to the pool before attention, so the pool holds every frame
position before its ring slot is reused.

Attention, per QSA layer run (adapters/qsa_split.py):

- decoding requests: the residency kernels keep groups QSA selected in the hot
  pages (misses copied from the pool), rewrite the selection to hot positions,
  and attention reads the staged table ``[H0..H{h-1}, R]`` with query positions
  that mask nothing (selection already emitted only causal positions);
- the prefilling request (at most one per step, scheduler.py): attention reads a
  frame-sized staging run, filled from the pool and the ring, with its real
  positions. Runs alternate by layer.

MTP draft steps that reuse anchor selections go through the same residency plan
on the binding's draft work positions.

Positions within ``RING_MARGIN`` of the newest row may already be overwritten in
R by speculative writes (MTP drafts run ahead of the verified rows); they are
read from the pool instead. Every value attention reads is the same byte that a
fully resident frame would hold, so outputs do not change.
"""
import mmap
import sys

import torch

from ..core.compressed_pages import MapAllocator
from .qsa_residency import kernels

RING_MARGIN = 8
STAGE_BLOCKS = 256
SPOOF = 1 << 62


class StreamingRuntime:
    def __init__(self, runner, options, selector):
        self.runner = runner
        self.options = options
        self.selector = selector
        self.hot_pages = options.qsa_hot_pages
        self.layers = list(runner._kvmem_qsa_layers.items())
        self.draft = int(getattr(runner.vllm_config.speculative_config, 'num_speculative_tokens', 0) or 0)
        # A request attends from hot pages only when its step is at most one
        # verify interval; any longer step (prefill chunk, Query replay, refresh
        # probe) attends from a staging run, whatever the vendor's prefill flag.
        self.verify_rows = self.draft + 1
        if self.draft + 1 > RING_MARGIN:
            raise ValueError('QSA streaming ring margin does not cover the draft lookahead')
        self.max_rows = int(runner.vllm_config.scheduler_config.max_num_batched_tokens)
        self.group_of = {}     # layer name -> KV cache group, known once the KV config exists (bind)
        self.carve = None
        self.ext = None
        self.hosts = {}        # slot -> per-layer [H0..H{h-1}, R] (host mirror)
        # KVMEM_QSA_STREAM_NO_PREFETCH_CHECK=1 (diagnostic): eager steps use the full
        # device staging fill instead of copy-stream prefetch; outputs must not change.
        from ..core import settings as _settings
        no_prefetch = _settings.get('KVMEM_QSA_STREAM_NO_PREFETCH_CHECK', '0')
        if no_prefetch not in ('0', '1'):
            raise ValueError('KVMEM_QSA_STREAM_NO_PREFETCH_CHECK must be 0 or 1')
        self.prefetch_enabled = no_prefetch == '0'
        self._pool_host = None
        self._installed_reuse = False
        self.eager_staging_steps = 0
        # Layer-wise prefetch (as Strata overlaps loads with compute): the pool
        # part of a QSA layer's staging run is copied on a side stream by the
        # DMA engine while the previous layers compute.
        self.step_output = None      # scheduler output of the step being executed
        self.step_slot = False       # False: not resolved yet for this step
        self.step_pages = 0          # staging pages to prefetch this step
        self.prefetched = {}         # layer index -> CUDA event (copy done)
        self.copy_stream = None
        self.mtp_layers = {i for i, (name, _) in enumerate(self.layers) if 'mtp' in name.lower()}
        self.prefetch_layers = 0

    # ------------------------------------------------------------------ binding
    def bind(self):
        """Bind maps, staging and the host pool once the real KV pool exists."""
        if self.carve is not None:
            return
        if self.selector.carve is None:
            raise RuntimeError('QSA streaming needs the bound selector carve')
        self.ext = kernels()
        self.group_of = {name: i for i, group in enumerate(self.runner.kv_cache_config.kv_cache_groups)
                         for name in group.layer_names}
        if any(layer.layer_name not in self.group_of for _, layer in self.layers):
            raise ValueError('Every QSA layer needs its own KV cache group')
        carve = self.carve = self.selector.streaming_carve     # placed by the selector binding
        if carve is None or (carve.hot_pages, carve.max_rows) != (self.hot_pages, self.max_rows):
            raise RuntimeError('QSA streaming carve was not placed with this runtime geometry')
        layer0 = self.layers[0][1]
        cache = layer0.kv_cache
        self.device = cache.device
        self.page_stride = int(cache.stride(0)) * cache.element_size()
        if self.page_stride != carve.native_page_bytes:
            raise ValueError('QSA streaming page geometry differs from the carve')
        self.token_bytes = layer0.num_kv_heads * layer0.head_dim       # FP8 E4M3: one byte per element
        self.page_tokens = carve.page_tokens
        self.plane_stride = self.page_tokens * self.token_bytes
        if (int(cache.shape[1]) != 2 or int(cache.shape[2]) != self.page_tokens or
                int(cache.stride(1)) * cache.element_size() != self.plane_stride or
                int(cache.shape[3]) * cache.element_size() != self.token_bytes):
            raise ValueError('QSA streaming requires [pages, 2, page, packed] native caches')
        self.ring_tokens = self.page_tokens
        self.hot_tokens = carve.hot_pages * self.page_tokens
        if self.max_rows > self.ring_tokens - RING_MARGIN:
            # One request's step (prefill chunk or atomic Query replay) must not wrap its ring page.
            raise ValueError(f'qsa_streaming needs max_num_batched_tokens <= {self.ring_tokens - RING_MARGIN} '
                             f'(got {self.max_rows}): lower query_replay_budget / prefill_chunk_tokens')
        for _, layer in self.layers:
            other = layer.kv_cache
            if (other.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr() or
                    other.storage_offset() != cache.storage_offset() or other.stride() != cache.stride()):
                raise ValueError('QSA streaming requires every QSA layer to view one native pool')
        storage = cache.untyped_storage()
        base = cache.storage_offset() * cache.element_size()
        start = base + carve.maps_first_block * carve.native_page_bytes
        raw = torch.empty(0, dtype=torch.uint8, device=self.device)
        raw.set_(storage, start, (carve.map_blocks * carve.native_page_bytes,), (1,))
        views = MapAllocator(raw, carve.map_plan()).allocate_all()
        self.maps = views
        self.streaming = views['streaming']
        self.streaming.zero_()
        self.decode_mask = views['decode_mask']
        self.stats = views['stats']
        self.stats.zero_()
        self.spoof = views['spoof']
        self.spoof.fill_(SPOOF)
        self.row_positions = views['row_positions']
        self.pool_base = views['pool_base']
        self.state = []
        for index in range(len(self.layers)):
            state = {key: views[(index, key)] for key in
                     ('group_slot', 'slot_group', 'referenced', 'hot_page_ids', 'copy_list', 'copy_count')}
            state['group_slot'].fill_(-1)
            state['slot_group'].fill_(-1)
            state['referenced'].zero_()
            state['hot_page_ids'].zero_()
            self.state.append(state)
        self.by_ptr = {layer._selected_positions.data_ptr(): index
                       for index, (_, layer) in enumerate(self.layers)}
        shared = self.ext.plan_shared_bytes(carve.max_groups, carve.hot_groups)
        if shared > 96 * 1024:
            raise ValueError('QSA residency maps exceed the plan kernel shared memory')
        self.query_start = torch.arange(carve.slots + 1, dtype=torch.int32, device=self.device)
        # Staged attention tables: staging runs (prefill) per buffer, -1 padded.
        width = max(int(getattr(layer, 'max_seq_len', 0)) for _, layer in self.layers)
        width = max(-(-width // self.page_tokens), carve.frame_pages)
        self.staging_tables = []
        for buffer in range(carve.staging_buffers):
            row = carve.staging_row(buffer) + [-1] * (width - carve.frame_pages)
            self.staging_tables.append(torch.tensor(row, dtype=torch.int32, device=self.device))
        self.staging_runs = [table[:carve.frame_pages] for table in self.staging_tables]
        self.staging_first = [carve.staging_row(buffer)[0] for buffer in range(carve.staging_buffers)]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # Pinned host pool (anonymous mapping: page aligned, no power-of-two rounding).
        slot_bytes = carve.pool_bytes_per_slot(self.token_bytes)
        total = slot_bytes * carve.slots
        self._mapping = mmap.mmap(-1, total)
        host = torch.frombuffer(self._mapping, dtype=torch.uint8)
        self._pool_host = host
        self._pool_dev = self.ext.host_register(host.data_ptr(), total)
        self.slot_bytes = slot_bytes
        self.layer_bytes = 2 * carve.frame_capacity_tokens * self.token_bytes
        self.plane_bytes = carve.frame_capacity_tokens * self.token_bytes
        self.pool_base.copy_(torch.tensor([self._pool_dev + s * slot_bytes for s in range(carve.slots)],
                                          dtype=torch.int64))
        self.pool_view = self.ext.device_view(self._pool_dev, total, self.device.index)

    def describe(self):
        carve = self.carve
        return dict(carve.describe(), pool_bytes=self.slot_bytes * carve.slots, ring_margin=RING_MARGIN)

    # -------------------------------------------------------------- lifecycle
    def admit(self, slot, layer_rows):
        """A request took ``slot``; ``layer_rows[i]`` are its native pages in QSA layer i."""
        if len(layer_rows) != len(self.layers):
            raise ValueError('QSA streaming admission needs pages for every QSA layer')
        for index, row in enumerate(layer_rows):
            if len(row) != self.hot_pages + 1 or len(set(row)) != len(row) or min(row) <= 0:
                raise ValueError('A streaming request must own exactly its hot and ring pages')
        self.reset_slot(slot)
        ids = torch.tensor(layer_rows, dtype=torch.int32)
        for index, state in enumerate(self.state):
            state['hot_page_ids'][slot].copy_(ids[index], non_blocking=False)
        self.hosts[slot] = [list(row) for row in layer_rows]
        self.streaming[slot] = 1

    def release(self, slot):
        if slot in self.hosts:
            self.streaming[slot] = 0
            self.reset_slot(slot)
            del self.hosts[slot]

    def reset_slot(self, slot):
        """Forget hot residency (admission, refresh, release)."""
        for state in self.state:
            state['group_slot'][slot].fill_(-1)
            state['slot_group'][slot].fill_(-1)
            state['referenced'][slot].zero_()

    def is_streaming(self, slot):
        return slot in self.hosts

    def owned_pages(self, slot, group_id):
        """The admitted [H0.., R] pages of the QSA layer in KV group ``group_id``."""
        for index, (_, layer) in enumerate(self.layers):
            if self.group_of[layer.layer_name] == group_id:
                return list(self.hosts[slot][index])
        raise KeyError(f'KV group {group_id} is not a QSA layer group')

    def ring_page(self, slot, index):
        return self.hosts[slot][index][self.hot_pages]

    def override_row(self, frame_columns, row, slot, index):
        """Worker block-table row of QSA layer ``index``: every frame column is the ring page."""
        return [self.ring_page(slot, index)] * frame_columns

    # ----------------------------------------------------------- staged table
    def rewrite_main_table(self, index, metadata, table, slots, num_reqs, reuse=False):
        """Decoding rows attend [H0.., R]; the prefilling row attends its staging run.

        Draft-reuse runs plan every streaming row as decoding, so they never stage.
        """
        if num_reqs <= 0:
            return
        width = int(table.shape[1])
        live = slots[:num_reqs]
        clamped = live.long().clamp(min=0)
        stream = self.streaming[clamped].bool() & (live >= 0)
        prefill = (torch.zeros_like(stream) if reuse else
                   self._staging_requests(metadata.query_start_loc, num_reqs))
        decode, staging = stream & ~prefill, stream & prefill
        rows = table[:num_reqs]
        hot = self.state[index]['hot_page_ids'][clamped]
        columns = min(width, self.hot_pages + 1)
        decode_rows = torch.full_like(rows, -1)
        decode_rows[:, :columns] = hot[:, :columns]
        run = self.staging_tables[index % self.carve.staging_buffers][:width]
        rows.copy_(torch.where(decode[:, None], decode_rows,
                               torch.where(staging[:, None], run[None, :].expand_as(rows), rows)))

    # ------------------------------------------------------------------ hooks
    def _layer(self, binding):
        if self.carve is None:
            # Not bound yet: vLLM's CUDA-graph memory profiling runs before the real pool exists.
            return None
        index = self.by_ptr.get(binding.selected_positions.data_ptr())
        return None if index is None else (index, self.layers[index][1], self.state[index])

    def _common(self, index, layer, state, binding, request_ids, query_positions, rows):
        cache = layer.kv_cache
        main = cache.view(torch.uint8) if cache.dtype != torch.uint8 else cache
        slots = binding.raw_state_slot_ids
        layer_offset = index * self.layer_bytes
        self.ext.residency_writeback(
            request_ids, query_positions, rows, slots, self.streaming, self.pool_base, layer_offset,
            self.plane_bytes, main, self.page_stride, self.plane_stride, state['hot_page_ids'], self.hot_pages,
            self.page_tokens, self.token_bytes, 64)
        return main, slots, layer_offset

    def _staging_requests(self, query_start_loc, requests):
        """Per request: more rows this step than one verify interval."""
        bounds = query_start_loc[:requests + 1].long()
        return (bounds[1:] - bounds[:-1]) > self.verify_rows

    def _decode_mask(self, slots, staging, requests):
        """streaming & not staging, per slot (slot -1 rows write nothing)."""
        mask = self.decode_mask
        mask.copy_(self.streaming)
        if staging is not None and requests:
            live = slots[:requests].long()
            valid = (live >= 0) & staging[:requests]
            # Zero prefilling slots; other rows target the always-zero dummy entry.
            target = torch.where(valid, live, torch.full_like(live, mask.shape[0] - 1))
            mask.index_put_((target,), torch.zeros_like(target, dtype=torch.uint8))
        return mask

    def _spoofed(self, request_ids, query_positions, slots, mask, rows):
        out = self.row_positions[:rows]
        requests = request_ids[:rows].long()
        slot = slots.long()[requests.clamp(min=0)]
        streaming = mask.bool()[slot.clamp(min=0)] & (slot >= 0) & (requests >= 0)
        out.copy_(torch.where(streaming, self.spoof[:rows], query_positions[:rows]))
        return out

    def __call__(self, binding, inputs, query_positions):
        """adapters/qsa_split.py hook for ordinary (selection + attention) runs."""
        found = self._layer(binding)
        if found is None:
            return None
        index, layer, state = found
        rows = int(inputs['query'].shape[0])
        if rows > self.max_rows:
            raise ValueError('QSA streaming rows exceed the planned capacity')
        request_ids = inputs['request_ids'][:rows]
        positions = inputs['query_positions'][:rows]
        main, slots, layer_offset = self._common(index, layer, state, binding, request_ids, positions, rows)
        requests = int(inputs['query_start_loc'].shape[0]) - 1
        staging = self._staging_requests(inputs['query_start_loc'], requests)
        mask = self._decode_mask(slots, staging, requests)
        selected = binding.selected_positions[:rows]
        state['copy_count'].zero_()
        self.ext.residency_plan(
            selected, int(selected.shape[1]), inputs['query_start_loc'], positions, slots, mask,
            state['group_slot'], state['slot_group'], state['referenced'], state['copy_list'], state['copy_count'],
            requests, self.hot_tokens, self.ring_tokens, RING_MARGIN, self.stats)
        self.ext.residency_copy(
            state['copy_list'], state['copy_count'], self.pool_base, layer_offset, self.plane_bytes, main,
            self.page_stride, self.plane_stride, state['hot_page_ids'], self.page_tokens, self.token_bytes, 64)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            # Graphs (decode and vLLM's small single-request prefill graphs) carry
            # the full device-side fill: at replay it finds the staging request
            # itself and copies its frame, or exits at once on a decode step.
            self.ext.stage_fill(
                inputs['query_start_loc'], requests, positions, slots, self.streaming, self.verify_rows,
                self.staging_runs[index % self.carve.staging_buffers], self.pool_base, layer_offset,
                self.plane_bytes, main, self.page_stride, self.plane_stride, state['hot_page_ids'],
                self.hot_pages, self.page_tokens, self.token_bytes, RING_MARGIN, self.stats, STAGE_BLOCKS,
                False)
        elif self.hosts and self.prefetch_enabled:
            slot = self._step_slot()
            if slot is not None:
                # The pool part arrives on the copy stream (prefetched during the
                # previous layers); the ring window is copied here, after this
                # layer's K/V write. Steps that stage never replay a graph.
                if index == 0:
                    self.prefetched.clear()      # a new forward of this step (refresh replays re-run it)
                done = self.prefetched.pop(index, None)
                if done is None:
                    done = self._prefetch(index, slot, main)
                torch.cuda.current_stream().wait_event(done)
                self.ext.stage_fill(
                    inputs['query_start_loc'], requests, positions, slots, self.streaming, self.verify_rows,
                    self.staging_runs[index % self.carve.staging_buffers], self.pool_base, layer_offset,
                    self.plane_bytes, main, self.page_stride, self.plane_stride, state['hot_page_ids'],
                    self.hot_pages, self.page_tokens, self.token_bytes, RING_MARGIN, self.stats, STAGE_BLOCKS,
                    True)
                following = index + 1
                if following < len(self.layers) and following not in self.mtp_layers:
                    # Buffer (index+1) % 2 was last read by layer index-1, enqueued before this point.
                    self.prefetched[following] = self._prefetch(following, slot, main)
            else:
                # The host saw no staging request; if the device finds one anyway
                # (same rule, so not expected), it copies the whole frame itself.
                self.ext.stage_fill(
                    inputs['query_start_loc'], requests, positions, slots, self.streaming, self.verify_rows,
                    self.staging_runs[index % self.carve.staging_buffers], self.pool_base, layer_offset,
                    self.plane_bytes, main, self.page_stride, self.plane_stride, state['hot_page_ids'],
                    self.hot_pages, self.page_tokens, self.token_bytes, RING_MARGIN, self.stats, STAGE_BLOCKS,
                    False)
        elif self.hosts:
            # Prefetch disabled (diagnostic): the device copies the whole frame itself.
            self.ext.stage_fill(
                inputs['query_start_loc'], requests, positions, slots, self.streaming, self.verify_rows,
                self.staging_runs[index % self.carve.staging_buffers], self.pool_base, layer_offset,
                self.plane_bytes, main, self.page_stride, self.plane_stride, state['hot_page_ids'],
                self.hot_pages, self.page_tokens, self.token_bytes, RING_MARGIN, self.stats, STAGE_BLOCKS,
                False)
        return self._spoofed(request_ids, positions, slots, mask, rows)

    def _step_slot(self):
        """The streaming request that stages this step (host data: no device read), or None."""
        if self.step_slot is not False:
            return self.step_slot
        slots = {}
        output = self.step_output
        for request_id, count in (getattr(output, 'num_scheduled_tokens', None) or {}).items():
            if count > self.verify_rows:
                slot = self.runner.req_states.req_id_to_index.get(request_id)
                if slot is not None and slot in self.hosts:
                    slots[slot] = (request_id, count)
        if len(slots) > 1:
            raise RuntimeError('More than one streaming request needs staging in one step')
        self.step_slot, self.step_pages = None, self.carve.frame_pages
        if slots:
            self.step_slot, (request_id, count) = slots.popitem()
            self.step_pages = self._frame_pages(output, request_id, count)
        return self.step_slot

    def _frame_pages(self, output, request_id, count):
        """Pages the staging request's frame can reach this step (host data), plus one spare.

        Frame end = logical end - window offset. A refresh inside the step only
        shrinks the frame, so the estimate stays an upper bound.
        """
        computed = None
        for request in getattr(output, 'scheduled_new_reqs', ()) or ():
            if request.req_id == request_id:
                computed = request.num_computed_tokens
        cached = getattr(output, 'scheduled_cached_reqs', None)
        if computed is None and cached is not None and request_id in cached.req_ids:
            computed = cached.num_computed_tokens[cached.req_ids.index(request_id)]
        if computed is None:
            return self.carve.frame_pages
        adapter = getattr(self.runner, '_kvmem_window_adapter', None)
        offset = adapter.offsets.get(request_id, 0) if adapter is not None else 0
        end = computed + count - offset
        return max(1, min(self.carve.frame_pages, -(-end // self.page_tokens) + 1))

    def begin_step(self, scheduler_output):
        self.step_output, self.step_slot = scheduler_output, False
        self.prefetched.clear()

    def _prefetch(self, index, slot, main):
        """Copy-engine H2D of the slot's pooled frame (all frame pages) into the layer's staging run."""
        ready = torch.cuda.Event()
        ready.record()
        stream = self.copy_stream
        stream.wait_event(ready)
        buffer = index % self.carve.staging_buffers
        pool = self._pool_dev + slot * self.slot_bytes + index * self.layer_bytes
        base = main.data_ptr() + self.staging_first[buffer] * self.page_stride
        with torch.cuda.stream(stream):
            for plane in (0, 1):
                self.ext.memcpy2d(base + plane * self.plane_stride, self.page_stride,
                                  pool + plane * self.plane_bytes, self.plane_stride,
                                  self.plane_stride, self.step_pages)
            done = torch.cuda.Event()
            done.record(stream)
        self.prefetch_layers += 1
        return done

    def reuse_attention(self, original, binding, *, query, request_ids, query_positions, reuse=False):
        """Wrapper of the vendor _run_attention: draft steps that reuse anchor selections."""
        found = self._layer(binding) if reuse else None
        if found is None or binding._draft_work_positions is None:
            return original(binding, query=query, request_ids=request_ids, query_positions=query_positions,
                            reuse=reuse)
        index, layer, state = found
        rows = int(query.shape[0])
        main, slots, layer_offset = self._common(index, layer, state, binding, request_ids, query_positions, rows)
        mask = self._decode_mask(slots, None, 0)
        selected = binding._draft_work_positions
        if int(selected.shape[0]) < rows or rows >= int(self.query_start.shape[0]):
            raise ValueError('QSA draft reuse rows exceed the residency plan')
        state['copy_count'].zero_()
        # One row per request, in request order: request r owns draft row r.
        self.ext.residency_plan(
            selected, int(selected.shape[1]), self.query_start, query_positions, slots, mask,
            state['group_slot'], state['slot_group'], state['referenced'], state['copy_list'], state['copy_count'],
            rows, self.hot_tokens, self.ring_tokens, RING_MARGIN, self.stats)
        self.ext.residency_copy(
            state['copy_list'], state['copy_count'], self.pool_base, layer_offset, self.plane_bytes, main,
            self.page_stride, self.plane_stride, state['hot_page_ids'], self.page_tokens, self.token_bytes, 64)
        positions = self._spoofed(request_ids, query_positions, slots, mask, rows)
        return original(binding, query=query, request_ids=request_ids, query_positions=positions, reuse=True)

    # ---------------------------------------------------------------- prefill
    def _fill_staging(self, index, main, layer_offset, slot, first, end):
        """Host-driven reference of stage_fill (tests): staging run <- frame [0, end), pool then ring."""
        buffer = index % self.carve.staging_buffers
        run = self.carve.staging_row(buffer)
        if end > len(run) * self.page_tokens:
            raise ValueError('Streaming prefill frame exceeds its staging run')
        if end - first > self.ring_tokens - RING_MARGIN:
            raise ValueError('Streaming prefill chunk exceeds the ring page')
        ring_start = max(0, end - self.ring_tokens + RING_MARGIN)
        main_base = main.data_ptr()
        pool = self._pool_dev + slot * self.slot_bytes + layer_offset
        pages = -(-ring_start // self.page_tokens)
        if pages:
            # Staging pages are consecutive native blocks: one strided copy per plane.
            for plane in (0, 1):
                self.ext.memcpy2d(main_base + run[0] * self.page_stride + plane * self.plane_stride,
                                  self.page_stride, pool + plane * self.plane_bytes,
                                  self.plane_stride, self.plane_stride, pages)
        ring = self.ring_page(slot, index)
        at = ring_start
        while at < end:
            page, offset = divmod(at, self.page_tokens)
            stop = min(end, (page + 1) * self.page_tokens)
            length = (stop - at) * self.token_bytes
            for plane in (0, 1):
                self.ext.memcpy2d(main_base + run[page] * self.page_stride + plane * self.plane_stride +
                                  offset * self.token_bytes, length,
                                  main_base + ring * self.page_stride + plane * self.plane_stride +
                                  offset * self.token_bytes, length, length, 1)
            at = stop

    # ---------------------------------------------------------------- refresh
    def pool_layer_views(self, slot, index, heads, dim):
        """(key, value) CUDA views over the slot's pooled frame in page layout, FP8."""
        start = slot * self.slot_bytes + index * self.layer_bytes
        shape = (self.carve.frame_pages, self.page_tokens, heads, dim)
        key = self.pool_view[start:start + self.plane_bytes]
        value = self.pool_view[start + self.plane_bytes:start + 2 * self.plane_bytes]
        return (key.view(torch.float8_e4m3fn).view(shape), value.view(torch.float8_e4m3fn).view(shape))

    def seed_ring(self, slot, index, main, frame_end):
        """Ring <- the newest page_tokens positions of a freshly assembled frame."""
        ring = self.ring_page(slot, index)
        pool = self._pool_dev + slot * self.slot_bytes + index * self.layer_bytes
        at = max(0, frame_end - self.ring_tokens)
        main_base = main.data_ptr()
        while at < frame_end:
            offset = at % self.page_tokens
            stop = min(frame_end, at + self.page_tokens - offset)
            length = (stop - at) * self.token_bytes
            for plane in (0, 1):
                self.ext.memcpy2d(main_base + ring * self.page_stride + plane * self.plane_stride +
                                  offset * self.token_bytes, length,
                                  pool + plane * self.plane_bytes + at * self.token_bytes, length, length, 1)
            at = stop

    def check(self):
        """Raise if a residency plan overflowed (read without a step sync: call at report points)."""
        needed, misses, steps, errors = self.stats.tolist()
        if errors:
            raise RuntimeError(f'QSA streaming residency errors: {errors}')
        return dict(needed_groups=needed, missed_groups=misses, planned_steps=steps,
                    eager_staging_steps=self.eager_staging_steps, prefetched_layers=self.prefetch_layers)

    def close(self):
        if self._pool_host is not None and self.ext is not None:
            self.ext.host_unregister(self._pool_host.data_ptr())
            self._pool_host = None


def install_reuse_wrapper(runtime, api):
    """Wrap the vendor module's _run_attention once (draft reuse goes through run())."""
    module = sys.modules[api.run.__module__]
    if getattr(module, '_kvmem_streaming_wrapped', False):
        return
    original = module._run_attention

    def run_attention(binding, *, query, request_ids, query_positions, reuse=False):
        return runtime.reuse_attention(original, binding, query=query, request_ids=request_ids,
                                       query_positions=query_positions, reuse=reuse)

    module._run_attention = run_attention
    module._kvmem_streaming_wrapped = True


# --------------------------------------------------------------------- runner
# vLLM's kernel warmup runs synthetic requests with made-up block IDs; they never
# come from the KVMem scheduler and run the ordinary resident path.
_WARMUP_PREFIXES = ('_warmup_', '_v2_mixed_warmup')


def _is_warmup(request_id):
    return request_id.startswith(_WARMUP_PREFIXES)


def _layer_rows(runtime, rows):
    return [list(rows[runtime.group_of[layer.layer_name]]) for _, layer in runtime.layers]


def _override(runtime, runner, slot, rows):
    """Stage the worker rows: QSA groups map every frame column to the ring page.

    vLLM applies all staged block-table writes of a step in ONE parallel launch,
    so two writes to the same row race. Native (and KVMem refresh) rows for this
    request are flushed first; the override then lands in a later, ordered launch.
    """
    runner.block_tables.apply_staged_writes()
    rows = list(rows)
    for index, (_, layer) in enumerate(runtime.layers):
        group = runtime.group_of[layer.layer_name]
        rows[group] = runtime.override_row(runtime.carve.frame_pages, rows[group], slot, index)
    runner.block_tables.append_block_ids(slot, tuple(rows), overwrite=True)


def after_add_requests(runner, scheduler_output):
    runtime = getattr(runner, '_kvmem_qsa_stream', None)
    if runtime is None or runtime.carve is None:
        return
    for request in scheduler_output.scheduled_new_reqs:
        if _is_warmup(request.req_id):
            continue
        slot = runner.req_states.req_id_to_index[request.req_id]
        rows = request.block_ids
        runtime.admit(slot, _layer_rows(runtime, rows))
        _override(runtime, runner, slot, rows)


def after_update_requests(runner, scheduler_output, table_updates):
    """Re-apply the ring rows after any native or KVMem row write to a streaming request."""
    runtime = getattr(runner, '_kvmem_qsa_stream', None)
    if runtime is None or runtime.carve is None:
        return
    cached = scheduler_output.scheduled_cached_reqs
    written = {**(getattr(scheduler_output, 'block_table_updates', None) or {}), **(table_updates or {})}
    resumed = set(getattr(cached, 'resumed_req_ids', ()) or ())
    for request_id, blocks in zip(cached.req_ids, cached.new_block_ids):
        if blocks is None or request_id in written or _is_warmup(request_id):
            continue
        qsa = [blocks[runtime.group_of[layer.layer_name]] for _, layer in runtime.layers]
        if any(qsa):
            if request_id not in resumed:
                raise RuntimeError('A streaming request received new attention pages after admission')
            written[request_id] = blocks
    for request_id, rows in written.items():
        slot = runner.req_states.req_id_to_index.get(request_id)
        if slot is None or _is_warmup(request_id):
            continue
        layer_rows = _layer_rows(runtime, rows)
        if not runtime.is_streaming(slot) or runtime.hosts[slot] != layer_rows:
            runtime.admit(slot, layer_rows)
        _override(runtime, runner, slot, rows)


def before_finish_requests(runner, scheduler_output):
    runtime = getattr(runner, '_kvmem_qsa_stream', None)
    if runtime is None or runtime.carve is None:
        return
    finished = False
    for request_id in scheduler_output.finished_req_ids:
        slot = runner.req_states.req_id_to_index.get(request_id)
        if slot is not None and runtime.is_streaming(slot):
            runtime.release(slot)
            finished = True
    if finished:
        # One device read per finished request: a plan overflow is a wrong answer, not a slowdown.
        report = runtime.check()
        output = getattr(runner, '_kvmem_probe_output', None)
        if output is not None:
            import json
            with (output / 'qsa_streaming.jsonl').open('a') as stream:
                stream.write(json.dumps(report) + '\n')


def install_eager_staging(runtime, manager):
    """Steps that stage a frame (any request with more rows than one verify interval) run eagerly.

    vLLM also captures FULL graphs for small single-request prefills; replaying
    one would skip the host-side staging fill and attend a stale staging run.
    Decode steps (at most one verify interval per request) keep their graphs.
    """
    if getattr(manager, '_kvmem_eager_staging', False):
        return
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    original = manager.dispatch

    def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None, num_ubatches=1):
        desc = original(num_reqs, num_tokens, uniform_token_count, num_active_loras,
                        max_query_len=max_query_len, num_ubatches=num_ubatches)
        if (desc.cg_mode != CUDAGraphMode.NONE and runtime.hosts and
                (max_query_len is None or max_query_len > runtime.verify_rows) and
                not (uniform_token_count is not None and uniform_token_count <= runtime.verify_rows)):
            runtime.eager_staging_steps += 1
            return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs,
                                            num_active_loras=desc.num_active_loras, num_ubatches=num_ubatches)
        return desc

    manager.dispatch = dispatch
    manager._kvmem_eager_staging = True
