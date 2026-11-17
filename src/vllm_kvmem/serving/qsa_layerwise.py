"""Per-layer Query selection for Flash Next QSA (qsa_layerwise_query).

KVMem's layerwise selection (serving/layerwise_query.py, the full-attention
path) gives every attention layer its own MeanK selection with the same token
count as the global one, so dense positions and causal lengths are unchanged and
only each layer's historical contents differ. The first N layers may keep the
global selection: shallow layers retrieve poorly on their own (LongMemEval-S,
K 8,192: 82.4 with shared shallow layers against 81.0 per-layer and 77.4
global; KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS).

Flash Next refreshes synchronously, and its refresh already runs a probe
forward that captures every QSA layer's Query. Here each layer scores the
candidates with its own probe Query against its own per-block mean keys, so all
layers' selections are known before assembly: the SSD tier then reads every
layer's blocks in one pipelined pass (archive_layout=layer reads each layer's
blocks as contiguous runs). The full-attention path instead scores each layer
with the Query of a second, layer-by-layer forward; this module does not.

The MTP draft layer takes the global selection (qsa_layerwise_mtp=global, as
the full-attention default) or layer 0's (layer0).
"""
import time

import torch


def layer_selections(authority, queries, mandatory_handles, max_tokens, *, layers, global_layers, draft_layers,
                     draft_selection):
    """Preview the global selection and every layer's own selection (memory must be READY).

    queries: the probe Queries, one per archive layer (layers QSA layers first).
    Returns (global_slots, per_layer) where per_layer[i] is a set of record slots
    with the global selection's token count, or None for the global selection.
    """
    from .authority import query_score_frame, record_key
    from .tensor_parallel import mean_head_scores
    from ..core.retrieval import mean_key_scores
    if authority.mean_k_backend == 'qw3_cuda':
        from ..core.cuda_scoring import mean_key_scores
    started = time.perf_counter()
    memory = authority.memory
    block = authority.shape[1]
    with authority.index_snapshot() as (records, means):
        pins = tuple(record_key(h)[0] for h in mandatory_handles)
        with query_score_frame(memory, records, pins, max_tokens) as frame:
            score_mask = tuple(frame.score_mask)
        eligible = torch.tensor(score_mask, dtype=torch.bool, device=queries.device)
        scoring = getattr(authority, 'scoring_layers', authority.shape[0])
        layer_means = means[:layers].to(queries.device)
        # The global scores, exactly as select_for_queries forms them.
        total = mean_head_scores(mean_key_scores(queries[:scoring], means[:scoring].to(queries.device),
                                                 eligible)).cpu().tolist()
        global_slots = set(memory.preview_selection({r.handle.slot: s for r, s in zip(records, total)}, pins,
                                                    max_tokens=max_tokens))
        selected_tokens = sum(r.count for r in records if r.handle.slot in global_slots)
        # Partial records keep identical inclusion in every layer (positions and lengths).
        fixed = set(pins) | {r.handle.slot for r in records if r.count != block and r.handle.slot in global_slots}
        excluded = {r.handle.slot for r in records if r.count != block and r.handle.slot not in global_slots}
        per_layer = []
        for index in range(layers):
            if index < global_layers or (index in draft_layers and draft_selection == 'global'):
                per_layer.append(None)
                continue
            if index in draft_layers:          # layer0: filled below
                per_layer.append('layer0')
                continue
            if index >= queries.shape[0]:
                raise ValueError('A non-draft QSA layer has no probe Query')
            scores = mean_head_scores(mean_key_scores(queries[index:index + 1], layer_means[index:index + 1],
                                                      eligible)).cpu().tolist()
            by_slot = {r.handle.slot: (-1.0 if r.handle.slot in excluded else s) for r, s in zip(records, scores)}
            chosen = set(memory.preview_selection(by_slot, tuple(fixed), action=False, max_tokens=max_tokens))
            if not fixed <= chosen or sum(r.count for r in records if r.handle.slot in chosen) != selected_tokens:
                raise RuntimeError('A layer selection changed the mandatory content or the causal length')
            per_layer.append(chosen)
        per_layer = [per_layer[0] if entry == 'layer0' else entry for entry in per_layer]
    report = dict(qsa_layerwise_seconds=time.perf_counter() - started, qsa_layerwise_global_layers=global_layers,
                  qsa_layerwise_layers=sum(entry is not None for entry in per_layer),
                  qsa_layerwise_overlap=[round(len(entry & global_slots) / max(1, len(global_slots)), 4)
                                         if entry is not None else 1.0 for entry in per_layer],
                  qsa_layerwise_union_blocks=len(set().union(global_slots, *[e for e in per_layer if e])))
    return global_slots, per_layer, report


def assembly_check():
    """KVMEM_QSA_LAYERWISE_ASSEMBLY_CHECK=1 (diagnostic): assemble layer by layer even
    when every layer takes the global selection; outputs must equal the global path."""
    from ..core import settings as _settings
    value = _settings.get('KVMEM_QSA_LAYERWISE_ASSEMBLY_CHECK', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_QSA_LAYERWISE_ASSEMBLY_CHECK must be 0 or 1')
    return value == '1'


def layer_records(authority, plan, global_slots, per_layer):
    """Per-layer record lists in frame order, or None when every layer takes the
    global selection (unless the assembly check is on)."""
    if all(entry is None for entry in per_layer) and not assembly_check():
        return None
    planned = {remap.block_id for remap in plan.remaps}
    if planned != global_slots:
        raise RuntimeError('The global Query selection differs from its preview')
    by_slot = {r.handle.slot: r for r in authority.records.values()}
    ordered_global = [by_slot[remap.block_id] for remap in sorted(plan.remaps, key=lambda r: r.to_base)]
    if any(a.history_begin >= b.history_begin for a, b in zip(ordered_global, ordered_global[1:])):
        raise RuntimeError('The QSA dense frame is not in history order')
    out = []
    for entry in per_layer:
        if entry is None:
            out.append(ordered_global)
        else:
            out.append(sorted((by_slot[slot] for slot in entry), key=lambda r: r.history_begin))
    return out


def pressure_layer_records(authority, plan, picked, fallback, *, draft_layers, draft_selection, scorer):
    """Per-layer pressure picks (serving/qsa_pressure.py) as one frame per QSA layer.

    picked[i] is target layer i's history-ordered record list, or None (no attention
    recorded yet: the global pick). Draft layers follow layer 0 or the global pick.
    Every frame must have the global pick's token count, or that layer keeps it."""
    by_slot = {r.handle.slot: r for r in authority.records.values()}
    ordered_global = [by_slot[remap.block_id] for remap in sorted(plan.remaps, key=lambda r: r.to_base)]
    total = plan.total_window_tokens
    out, kept_global = [], list(fallback)
    target = iter(picked)
    for index in range(len(picked) + len(draft_layers)):
        if index in draft_layers:
            out.append('draft')
            continue
        layer = next(target)
        if layer is None or sum(r.count for r in layer) != total:
            if layer is not None:
                kept_global.append(index)
            out.append(ordered_global)
        else:
            out.append(layer)
    first = out[0] if out[0] != 'draft' else ordered_global
    out = [(first if draft_selection == 'layer0' else ordered_global) if entry == 'draft' else entry
           for entry in out]
    report = dict(qsa_pressure_scorer=scorer, qsa_pressure_global_layers=sorted(set(kept_global)),
                  qsa_pressure_overlap=[round(len({r.handle.slot for r in entry} & {r.handle.slot for r in ordered_global})
                                              / max(1, len(ordered_global)), 4) for entry in out])
    return out, report
