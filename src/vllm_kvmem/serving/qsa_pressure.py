"""Per-layer pressure for Flash Next QSA (strategy_path=layer).

KVMem's per-layer pressure (serving/layerwise_pressure.py) keeps each attention
layer's window at K tokens while a long prompt is read. Every
prefill_attention_interval tokens it samples rows of the prefill chunk (every
prefill_attention_stride-th row, PrefillHistory.rows) and records the attention
mass each window block receives from them; a layer's history ranks its window
blocks by accumulated attention / time visible, or by the age-adjusted lift
(strategy_pressure_scorer=lift, prefill_attention_half_life). Those packets come
from FlashAttention statistics on the full-attention path.

QSA attends only to the positions its indexer selected, with a plain softmax
over them (b12x qsa reference.sparse_paged_gqa_reference: scale head_dim**-0.5,
K = FP8 cache x k_descale, no sinks). With qsa_pressure_scores=kernel the B12X
attention kernel itself stores the sampled rows' masked logits
(adapters/qsa_scores.py) and this module only normalizes them; with
'recompute' (or when the kernel variant is unavailable for a call) it rebuilds
the same logits from the selected positions, query and K cache. Either way the
softmax is summed per window block: the attention mass each block received.
Blocks QSA did not select received none. The packets feed the
same per-layer histories (BufferedPrefillHistory; a lift State for lift) and the
same native picker (MemoryOracle.preview_selection with pressure_scores) as the
full-attention path. The MTP draft layer takes layer 0's selection.
"""
import time

import torch

from .action_attention import window_segments


def enabled(options):
    return (getattr(options, 'strategy_path', 'global') == 'layer' and
            getattr(options, 'strategy_pressure_scorer', 'attention') in ('attention', 'lift'))


class QSALayerPressure:
    ROW_GROUP = 8          # sampled rows per recomputation batch (bounds the K gather)

    def __init__(self, runner, options):
        self.runner, self.options = runner, options
        self.lift = getattr(options, 'strategy_pressure_scorer', 'attention') == 'lift'
        qsa = runner._kvmem_qsa_layers
        drafts = set(getattr(runner, '_kvmem_mtp_modules', {}))
        self.names = [name for name in qsa if name not in drafts]
        self.draft_indices = [i for i, name in enumerate(qsa) if name in drafts]
        self.layer_count = len(qsa)
        self.by_ptr = {}       # built lazily: selection buffers are bound with the KV cache
        speculative = runner.vllm_config.speculative_config
        self.verify_rows = int(getattr(speculative, 'num_speculative_tokens', 0) or 0) + 1
        self.histories = {}    # rid -> [history per target layer]
        self.maps = {}         # rid -> [frame records per target layer] after a per-layer selection
        self.pending = {}      # rid -> {layer: packet}
        self.stats = dict(packets=0, rows=0, seconds=0.0, kernel_packets=0, recomputed_packets=0)
        self.kernel = getattr(options, 'qsa_pressure_scores', 'kernel') == 'kernel'
        self.kernel_error = None

    # ---------------------------------------------------------------- histories
    def _new_history(self):
        from .buffered_prefill_attention import BufferedPrefillHistory
        history = BufferedPrefillHistory(self.options)
        if self.lift:
            from .._prefill_state_native import State
            history._buffered_state = State.lift_empty()
        return history

    def histories_for(self, rid):
        found = self.histories.get(rid)
        if found is None:
            found = self.histories[rid] = [self._new_history() for _ in self.names]
        return found

    def frame(self, rid, authority, layer, history_begin, local_begin):
        """The layer's frame records (slot, generation, history_begin, count, position)
        covering frame positions [0, local_begin): its map from the last per-layer
        selection extended with the input accepted since, else the native frame."""
        found = self.maps.get(rid)
        if found is None:
            result = list(authority.memory._store.working_records())
        else:
            from .layerwise_pressure import extend_records
            origin, layers = found
            result = extend_records(layers[layer], origin, history_begin)
        if sum(r[3] for r in result) != local_begin:
            raise RuntimeError('QSA layer frame length differs from the native frame')
        return result

    def report(self):
        """Cumulative packet counters (kernel vs recomputed logits) for refresh reports."""
        out = dict(self.stats, seconds=round(self.stats['seconds'], 3), source='kernel' if self.kernel else 'recompute')
        if self.kernel_error:
            out['kernel_error'] = self.kernel_error
        return out

    def retire(self, rid):
        self.histories.pop(rid, None)
        self.maps.pop(rid, None)
        self.pending.pop(rid, None)

    # ------------------------------------------------------------------ packets
    def hook(self, binding, inputs, query_positions):
        """SplitQSA hook (after the streaming hook): record sampled rows' attention mass."""
        pointer = binding.selected_positions.data_ptr()
        layer = self.by_ptr.get(pointer)
        if layer is None:
            qsa = self.runner._kvmem_qsa_layers
            self.by_ptr = {qsa[name]._selected_positions.data_ptr(): i for i, name in enumerate(self.names)}
            layer = self.by_ptr.get(pointer)
            if layer is None:
                return None
        batch = getattr(self.runner, '_kvmem_last_batch', None)
        runtime = getattr(self.runner, '_kvmem_capture_runtime', None)
        if batch is None or runtime is None:
            return None
        starts_np = batch.query_start_loc_np
        adapter = getattr(self.runner, '_kvmem_window_adapter', None)
        packets = []
        for i, rid in enumerate(batch.req_ids):
            first, last = int(starts_np[i]), int(starts_np[i + 1])
            count = last - first
            authority = runtime.authorities.get(rid)
            if count <= self.verify_rows or authority is None:
                continue
            begin = int(batch.num_computed_tokens_np[i])
            local = begin - (adapter.offsets.get(rid, 0) if adapter is not None else 0)
            histories = self.histories_for(rid)
            rows = histories[layer].rows(begin, count, local)
            if not rows:
                continue
            started = time.perf_counter()
            starts, counts, keys = window_segments(self.frame(rid, authority, layer, begin, local),
                                                   begin, local, count, 32)
            packets.append((rid, layer, [first + r for r in rows], (begin + count, count, rows, starts, keys, local)))
            self.stats['packets'] += 1
            self.stats['rows'] += len(rows)
            self.stats['seconds'] += time.perf_counter() - started
        if not packets:
            return None
        started = time.perf_counter()
        logits = self._kernel_logits(binding, inputs, packets) if self.kernel else None
        self._finish(binding, inputs, packets, logits)
        self.stats['seconds'] += time.perf_counter() - started
        return None

    def _kernel_logits(self, binding, inputs, packets):
        """The sampled rows' logits from the B12X kernel variant (adapters/qsa_scores.py),
        launched on those rows alone before the layer's own attention; None to recompute."""
        if self.kernel_error is not None or torch.cuda.is_current_stream_capturing():
            return None
        from types import SimpleNamespace
        from ..adapters import qsa_scores
        rows = sorted({row for _, _, global_rows, _ in packets for row in global_rows})
        try:
            scores = qsa_scores.sampled_logits(binding, inputs, rows)
        except Exception as error:   # noqa: BLE001 - reported; this and later packets recompute
            self.kernel_error = f'{type(error).__name__}: {error}'
            return None
        return SimpleNamespace(scores=scores, rows=rows)

    def _finish(self, binding, inputs, packets, capture):
        for rid, layer, global_rows, packet in packets:
            if capture is not None:
                masses = self._kernel_masses(binding, inputs, capture, global_rows, packet[3])
                self.stats['kernel_packets'] += 1
            else:
                masses = self._masses(binding, inputs, global_rows, packet[3])
                self.stats['recomputed_packets'] += 1
            self.pending.setdefault(rid, {})[layer] = packet + (masses,)

    def _kernel_masses(self, binding, inputs, capture, rows, starts):
        """[rows, window blocks] attention mass from the kernel's logits (GPU, float32);
        capture.scores[i] holds the logits of query row capture.rows[i]."""
        import bisect
        device = capture.scores.device
        slots = [bisect.bisect_left(capture.rows, row) for row in rows]
        index = torch.tensor(rows, dtype=torch.int64, device=device)
        logits = capture.scores.index_select(0, torch.tensor(slots, dtype=torch.int64, device=device))
        scale = int(inputs['query'].shape[2]) ** -0.5
        if binding.k_descale is not None:
            logits = logits * (binding.k_descale.float() * scale)
        else:
            logits = logits * scale
        probability = torch.softmax(logits, dim=-1).nan_to_num_(0.0).mean(dim=1)   # [r, W]
        pos = binding.selected_positions.index_select(0, index).long()
        limit = inputs['query_positions'].index_select(0, index).long()
        valid = (pos >= 0) & (pos <= limit[:, None])
        safe = pos.clamp(min=0)
        start_t = torch.tensor(starts, dtype=torch.int64, device=device)
        block = torch.searchsorted(start_t, safe, right=True) - 1
        out = torch.zeros(len(rows), len(starts), dtype=torch.float32, device=device)
        out.scatter_add_(1, block.clamp(min=0), probability * valid)
        return out

    def _masses(self, binding, inputs, rows, starts):
        """[rows, window blocks] attention mass of QSA's sparse softmax (GPU, float32)."""
        device = binding.selected_positions.device
        index = torch.tensor(rows, dtype=torch.int64, device=device)
        query = inputs['query'].index_select(0, index).float()                # [R, H, D]
        selected = binding.selected_positions.index_select(0, index).long()   # [R, W]
        limit = inputs['query_positions'].index_select(0, index).long()
        requests = inputs['request_ids'].index_select(0, index).long()
        cache = binding.main_k_cache                                           # [pages, page, kv, D]
        page = int(cache.shape[1])
        kv_heads, dim = int(cache.shape[2]), int(cache.shape[3])
        heads = int(query.shape[1])
        group = heads // kv_heads
        scale = dim ** -0.5
        descale = binding.k_descale.float() if binding.k_descale is not None else None
        start_t = torch.tensor(starts, dtype=torch.int64, device=device)
        out = torch.zeros(len(rows), len(starts), dtype=torch.float32, device=device)
        for at in range(0, len(rows), self.ROW_GROUP):
            sl = slice(at, at + self.ROW_GROUP)
            pos = selected[sl]
            valid = (pos >= 0) & (pos <= limit[sl, None])
            safe = pos.clamp(min=0)
            table = binding.main_block_table.index_select(0, requests[sl])
            pages = table.gather(1, safe // page).long()
            keys = cache[pages, safe % page].float()                            # [r, W, kv, D]
            if descale is not None:
                keys = keys * descale
            q = query[sl].view(-1, kv_heads, group, dim)
            logits = torch.einsum('rkgd,rwkd->rkgw', q, keys) * scale
            logits = logits.masked_fill(~valid[:, None, None, :], float('-inf'))
            probability = torch.softmax(logits, dim=-1).nan_to_num(0.0).mean(dim=(1, 2))   # [r, W]
            block = torch.searchsorted(start_t, safe, right=True) - 1
            out[sl].scatter_add_(1, block.clamp(min=0), probability * valid)
        return out

    def flush(self):
        """Accept every recorded packet (the chunk is complete); one host copy per packet."""
        if not self.pending:
            return
        pending, self.pending = self.pending, {}
        for rid, layers in pending.items():
            histories = self.histories.get(rid)
            if histories is None:
                continue
            accepted = set()
            for layer, (end, count, rows, starts, keys, local, masses) in sorted(layers.items()):
                accepted.add(histories[layer].accept(end, count, count, rows, starts, keys, local,
                                                     masses.double().cpu().numpy()))
            if len(accepted) > 1:
                raise RuntimeError('QSA layer histories disagree on the accepted sampling boundary')

    # ---------------------------------------------------------------- selection
    def select(self, rid, authority, mandatory_slots, budget, history_begin, local_begin):
        """Per-layer pressure selections: one record list per QSA layer, in frame order.

        Candidates are each layer's current window; scores come from its history
        (scores_for: lift ranks blocks no row scored yet first); the native picker
        keeps sink, mandatory blocks and the budget. A layer with no attention
        recorded yet keeps the global window (reported)."""
        self.flush()
        histories = self.histories_for(rid)
        by_slot = {r.handle.slot: r for r in authority.records.values()}
        by_begin = {r.history_begin: r for r in authority.records.values()}
        chosen, fallback = [], []
        for layer, history in enumerate(histories):
            mapping = self.frame(rid, authority, layer, history_begin, local_begin)
            candidates = []
            for _, _, begin, n, _ in mapping:
                record = by_begin.get(begin)
                if record is None or record.count != n:
                    raise RuntimeError('QSA layer pressure candidate is absent from accepted history')
                candidates.append(record)
            scores = history._buffered_state.scores_for([r.history_begin for r in candidates])
            supplied = [(r.handle.slot, float(s)) for r, s in zip(candidates, scores)]
            if not any(s > 0 for _, s in supplied):
                fallback.append(layer)
                chosen.append(None)
                continue
            slots = set(authority.memory.preview_selection({}, tuple(mandatory_slots), action=False,
                                                           pressure_scores=supplied, max_tokens=budget))
            chosen.append(sorted((by_slot[s] for s in slots), key=lambda r: r.history_begin))
        return chosen, fallback

    def remember(self, rid, layer_records, origin):
        """Frame maps after a per-layer assembly (dense positions in frame order);
        origin is the history position the frames end at."""
        maps = []
        for records in layer_records[:len(self.names)]:
            position, frame = 0, []
            for r in records:
                frame.append((r.handle.slot, r.handle.generation, r.history_begin, r.count, position))
                position += r.count
            maps.append(frame)
        self.maps[rid] = (origin, maps)

    def forget(self, rid):
        """A global selection: every layer's frame is the working set again."""
        self.maps.pop(rid, None)
