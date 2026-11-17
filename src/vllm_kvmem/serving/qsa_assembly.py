"""Synchronous bounded QSA reconstruction from one atomic CPU authority.

Both target and MTP selector keys are rebuilt from lossless BF16 projections
using the installed B12X compression kernel, with dense selected positions.
No inverse rotation of quantized/pooled keys is used. Scratch is chunk-bounded;
only the final raw ring survives until the enclosing refresh commits.
"""
import time

import torch

from ..adapters.assembly import assemble_fp8
from ..core.memory import Phase
from ..core.native_pages import kernel_table
from .window_assembly import AssemblyResult, _UnfencedAssembly


class QSAWindowAssembler:
    def __init__(self, runner, request_id, names, block_tokens, chunk_blocks=None, **kwargs):
        if kwargs.get('lease') is not None or kwargs.get('arena', 'main') != 'main':
            raise ValueError('QSA currently requires synchronous main-arena refresh')
        self.runner, self.request_id = runner, request_id
        self.block_tokens = block_tokens
        options = getattr(getattr(runner, '_kvmem_capture_runtime', None), 'options', None)
        self.chunk_blocks = (getattr(options, 'assembly_chunk_blocks', 32)
                             if chunk_blocks is None else chunk_blocks)
        if type(self.chunk_blocks) is not int or not 1 <= self.chunk_blocks <= 256:
            raise ValueError('QSA assembly chunk blocks must be an integer in [1, 256]')
        self.closed = False
        self._completion = None
        self.index = runner.req_states.req_id_to_index[request_id]
        self.layers = tuple(runner._kvmem_qsa_layers.values())
        self.names = tuple(runner._kvmem_qsa_layers)
        expected = (*self.names, *('__qsa_index__.' + name for name in self.names))
        if tuple(names) != expected:
            raise ValueError('QSA assembly requires paired main and auxiliary archive layers')
        groups = {name: i for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                  for name in group.layer_names}
        self.tables = [kernel_table(runner, groups[layer.layer_name], self.index).clone()
                       for layer in self.layers]
        self.views = [layer.impl._kv_cache_views(layer.kv_cache) for layer in self.layers]
        # qsa_split_selector: selector keys live in the slot's carved pages, not in
        # the tails of the main pages (adapters/qsa_streaming.py).
        selector = getattr(runner, '_kvmem_qsa_selector', None)
        self.selector = selector if selector is not None and selector.views else None
        self.selector_table = self.selector.slot_row(self.index) if self.selector is not None else None
        # qsa_streaming: the frame is assembled into the slot's pinned host pool
        # (adapters/qsa_stream_runtime.py), never into resident frame pages.
        stream = getattr(runner, '_kvmem_qsa_stream', None)
        self.stream = stream if stream is not None and stream.is_streaming(self.index) else None
        if self.stream is not None:
            heads, dim = self.layers[0].num_kv_heads, self.layers[0].head_dim
            self.views = [stream.pool_layer_views(self.index, i, heads, dim) for i in range(len(self.layers))]
            self.tables = [None] * len(self.layers)

    def close(self):
        if self._completion is not None and not self._completion.query():
            return False
        self.closed = True
        return True

    def assemble(self, authority, plan, layer_records=None):
        """layer_records (qsa_layerwise_query): one record list per QSA layer, in
        frame order, each with the plan's token count; None assembles the plan."""
        from b12x.attention.qsa._kernels import launch_compress_completed_groups
        if self.closed or authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
            raise ValueError('QSA assembly requires an active owned selection')
        count_layers = len(self.layers)
        if authority.shape != (2 * count_layers, self.block_tokens,
                               self.layers[0].num_kv_heads, self.layers[0].head_dim):
            raise ValueError('QSA authority geometry differs from its paired archive format')
        remaps = sorted(plan.remaps, key=lambda r: r.to_base)
        records = {r.handle.slot: r for r in authority.records.values()}
        cursor = 0
        for remap in remaps:
            record = records.get(remap.block_id)
            if record is None or record.count != remap.n_tokens or remap.to_base != cursor:
                raise ValueError('QSA selection must describe the complete contiguous dense frame')
            cursor += remap.n_tokens
        if cursor != plan.total_window_tokens or cursor <= 0:
            raise ValueError('QSA selected frame extent mismatch')
        if layer_records is not None:
            if len(layer_records) != count_layers:
                raise ValueError('Layerwise QSA assembly needs one record list per QSA layer')
            for chosen in layer_records:
                if (sum(r.count for r in chosen) != cursor or
                        any(records.get(r.handle.slot) is not r for r in chosen) or
                        any(a.history_begin >= b.history_begin for a, b in zip(chosen, chosen[1:]))):
                    raise ValueError('A layer selection is not a history-ordered frame of the planned length')
        if self.selector is not None and cursor > self.selector.carve.pages_per_slot * self.selector.carve.page_tokens:
            raise ValueError('QSA selection exceeds the slot selector pages')
        for layer, table, (key, _) in zip(self.layers, self.tables, self.views):
            if cursor > layer.rotary_emb.cos_sin_cache.shape[0]:
                raise ValueError('QSA selection exceeds RoPE capacity')
            if layer._compressed_cache is None or key.shape[1] % layer.compress_ratio:
                raise ValueError('QSA selector tail is not bound/aligned')
            if self.stream is not None:
                if cursor > key.shape[0] * key.shape[1]:
                    raise ValueError('QSA selection exceeds the streaming host pool')
                continue
            if cursor > table.numel() * key.shape[1]:
                raise ValueError('QSA selection exceeds leased pages')
            pages = table[:(cursor+key.shape[1]-1)//key.shape[1]].tolist()
            if (len(set(pages)) != len(pages) or
                    any(page <= 0 or page >= key.shape[0] for page in pages)):
                raise ValueError('QSA selection contains null, duplicate or out-of-range pages')
        device = self.layers[0].device
        scratch = []
        for layer in self.layers:
            scratch.append((torch.zeros_like(layer._raw_k_ring[:1]),
                            torch.full_like(layer._raw_logical_positions[:1], -1),
                            torch.full_like(layer._raw_rope_positions[:1], -1)))
        owners, uploaded, chunks = [self, authority, plan], 0, 0
        started = time.perf_counter()
        # Work items: (layer or None for every layer, first frame position, record handles).
        items = []
        if layer_records is None:
            for at in range(0, len(remaps), self.chunk_blocks):
                chunk = remaps[at:at+self.chunk_blocks]
                items.append((None, chunk[0].to_base, [records[r.block_id].handle for r in chunk]))
        else:
            for layer_index, chosen in enumerate(layer_records):
                base = 0
                for at in range(0, len(chosen), self.chunk_blocks):
                    chunk = chosen[at:at+self.chunk_blocks]
                    items.append((layer_index, base, [r.handle for r in chunk]))
                    base += sum(r.count for r in chunk)
        handles = [item[2] for item in items]
        item_groups = [None if item[0] is None else [item[0]] for item in items]
        io = dict(archive_direct_bytes=0, archive_direct_requests=0, archive_direct_wait_s=0.0)
        archive = getattr(authority, 'archive', None)
        direct = getattr(archive, 'direct_reads', False)
        if direct and getattr(archive, 'layer_major', False):
            # SSD tier, archive_layout=layer: packed per-layer groups, no padding read.
            from .direct_assembly_reads import direct_layer_chunks
            first = self.layers[0]
            source = direct_layer_chunks(authority, handles, device, io, layers=count_layers,
                                         block_tokens=self.block_tokens,
                                         width=first.num_kv_heads * first.head_dim,
                                         selector_bytes=first.index_head_dim * 2, chunk_groups=item_groups)
        elif direct:
            # SSD tier: O_DIRECT reads into a pinned ring, several chunks ahead.
            from .direct_assembly_reads import direct_chunks
            # Selector (auxiliary) layers keep zeros in their raw plane: never read it.
            source = direct_chunks(authority, handles, device, io, unused_raw_layers=(count_layers, count_layers))
        else:
            source = self._buffered_chunks(authority, handles, device)
        try:
            for (only, start, _), (raw_of, values_of, n, size) in zip(items, source):
                uploaded += size
                positions = torch.arange(start, start+n, device=device, dtype=torch.int64)
                requests = torch.zeros(n, device=device, dtype=torch.int32)
                query_start = torch.tensor([0, n], device=device, dtype=torch.int32)
                slot = torch.zeros(1, device=device, dtype=torch.int32)
                for i, (layer, table, (key, value), ring_state) in enumerate(
                        zip(self.layers, self.tables, self.views, scratch)):
                    if only is not None and i != only:
                        continue
                    page = key.shape[1]
                    slots = (positions if table is None else
                             table[positions // page].long() * page + positions % page)
                    assemble_fp8(raw_of(i), values_of(i).view(torch.float8_e4m3fn),
                        positions, slots, layer.rotary_emb.cos_sin_cache, key, value, layer._k_scale)
                    selector = values_of(count_layers+i).reshape(n, -1)[:, :layer.index_head_dim*2]
                    selector = selector.contiguous().view(torch.bfloat16)
                    rope = positions[:, None].expand(-1, layer.position_axes).contiguous()
                    cos, sin = layer.rotary_emb.cos_sin_cache.chunk(2, dim=-1)
                    caps = layer._qsa_decode_context.caps
                    launch_compress_completed_groups(raw_index_key=selector,
                        query_positions=positions, rope_positions=rope,
                        request_ids=requests, query_start_loc=query_start,
                        raw_state_slot_ids=slot, raw_k_ring=ring_state[0],
                        raw_logical_positions=ring_state[1], raw_rope_positions=ring_state[2],
                        key_norm_weight=layer.indexer.k_layernorm.weight,
                        rope_cos=cos, rope_sin=sin, compressed_cache=layer._compressed_cache,
                        compressed_block_table=(table[None, :] if self.selector_table is None
                                                else self.selector_table), caps=caps)
                    keep = min(n, layer.raw_ring_capacity)
                    tail_positions, tail_raw = positions[-keep:], selector[-keep:]
                    ring_slots = tail_positions % layer.raw_ring_capacity
                    ring_state[0][0, ring_slots] = tail_raw
                    ring_state[1][0, ring_slots] = tail_positions
                    ring_state[2][0, ring_slots] = tail_positions[:, None].expand(-1, layer.position_axes)
                chunks += 1
            if self.stream is not None:
                # The ring holds the newest positions; hot residency describes the old frame.
                for i, layer in enumerate(self.layers):
                    cache = layer.kv_cache
                    self.stream.seed_ring(self.index, i, cache.view(torch.uint8) if cache.dtype != torch.uint8
                                          else cache, cursor)
                self.stream.reset_slot(self.index)
            # Stage only the bounded tail. The outer transaction first restores
            # provisional GDN/QSA state, then publishes these rebuilt rings.
            pending = []
            for layer, (ring, tags, _) in zip(self.layers, scratch):
                p = torch.arange(max(0, cursor-layer.raw_ring_capacity), cursor,
                                 device=device, dtype=torch.int64)
                pending.append((layer, self.index, p, ring[0, p % layer.raw_ring_capacity].clone(), cursor))
            event = torch.cuda.Event()
            event.record()
            self._completion = event
            self.runner._kvmem_qsa_pending_rings.extend(pending)
            extra = {'qsa_rebuilt_selector_tokens': cursor, 'qsa_auxiliary_codec': 'bf16_bytes_v1',
                     'qsa_layerwise': layer_records is not None}
            if direct:
                extra.update(io)
            return AssemblyResult(event, tuple(owners), cursor, 0, chunks,
                time.perf_counter()-started, uploaded, extra)
        except BaseException:
            try:
                event = torch.cuda.Event()
                event.record()
            except BaseException:
                event = _UnfencedAssembly()
            self._completion = event
            authority.memory.fail_transaction(AssemblyResult(event, tuple(owners), 0, 0, 0))
            raise
        finally:
            source.close()
            release = getattr(plan, 'release_cpu_sources', None)
            if release is not None:
                release()

    @staticmethod
    def _buffered_chunks(authority, handles, device):
        for chunk in handles:
            payloads = authority.read_records(chunk)
            raw = torch.cat([p[0] for p in payloads], dim=1).contiguous().to(device)
            values = torch.cat([p[1] for p in payloads], dim=1).contiguous().to(device)
            yield raw.__getitem__, values.__getitem__, raw.shape[1], raw.numel() + values.numel()
