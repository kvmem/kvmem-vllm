"""Separate MeanK and FPS4 quotas over the unchanged native block budget."""
import math

from .query_group_recall import build_halo_groups


def preview_hybrid_selection(memory, records, fps_scores, mean_scores, mandatory_slots=(),
                             *, max_tokens=None, mean_fraction=.5, message_spans=()):
    if (type(mean_fraction) not in (float, int) or not 0 < mean_fraction <= 1
            or len(mean_scores) != len(records) or len(fps_scores) != len(records)
            or any(not math.isfinite(s) or s < 0 for s in (*mean_scores, *fps_scores))
            or memory.config.block_tokens != 32 or memory.config.action_fraction != 0):
        raise ValueError('Invalid hybrid quota geometry, scores or action policy')
    budget = memory.config.select_budget if max_tokens is None else max_tokens
    if type(budget) is not int or budget <= 0:
        raise ValueError('Hybrid quota requires a positive native token budget')
    slots = [r.handle.slot for r in records]
    if len(set(slots)) != len(slots) or any(memory._store.handle(r.handle.slot) != r.handle for r in records):
        raise ValueError('Hybrid quota index has stale or duplicate handles')
    groups, group_scores, metadata = build_halo_groups(records, fps_scores,
        message_spans=message_spans)
    fixed = set(memory._store.fixed_selection_blocks(list(mandatory_slots), pressure=False, budget=budget))
    if not fixed.issubset(slots) or len(fixed) > budget//32:
        raise ValueError('Fixed content exceeds hybrid budget or current index')
    optional = max(0, min(len(records), budget//32)-len(fixed))
    mean_quota = math.floor(optional*mean_fraction)
    mean_by_slot = dict(zip(slots, mean_scores))
    # A zero native runtime budget means "use the configured budget". Never
    # pass zero to represent an empty first-stage selection.
    protected = (set(memory.preview_selection(mean_by_slot, tuple(fixed), action=False,
        prefill_scores=None, max_tokens=(len(fixed)+mean_quota)*32)) if mean_quota else fixed.copy())
    fps_selected = set(memory.preview_group_selection(groups, group_scores,
        tuple(protected), max_tokens=budget))
    # All original FPS probabilities are nonnegative. Negative singleton
    # priorities run strictly after every atomic FPS group. Rank by MeanK,
    # with the native chronological-later tie break, to fill stranded slots.
    rank = sorted(range(len(records)), key=lambda i: (mean_scores[i], records[i].history_begin), reverse=True)
    single_scores = [0.]*len(records)
    for ordinal, i in enumerate(rank):
        single_scores[i] = -float(ordinal+1)
    groups += [(r.history_begin, r.history_begin+r.count) for r in records]
    group_scores += single_scores
    selected = set(memory.preview_group_selection(groups, group_scores, tuple(protected), max_tokens=budget))
    if not fixed <= protected <= fps_selected <= selected or len(selected)*32 > budget:
        raise RuntimeError('Hybrid selection violated staged budget or fixed-content invariants')
    positions = dict(zip(slots, (r.history_begin for r in records)))
    starts = lambda values: sorted(positions[s] for s in values)
    quota = dict(mode='mean_fps_quota', mean_fraction=float(mean_fraction), budget=budget,
        fixed_blocks=len(fixed), optional_blocks=optional, mean_quota_blocks=mean_quota,
        mean_selected_blocks=len(protected-fixed), fps_added_blocks=len(fps_selected-protected),
        mean_fill_blocks=len(selected-fps_selected), selected_blocks=len(selected), charged_tokens=len(selected)*32,
        score_combination='separate_full_domain_scores; budget_quota; no_score_blend',
        fixed_starts=starts(fixed), mean_selected_starts=starts(protected-fixed),
        fps_added_starts=starts(fps_selected-protected), mean_fill_starts=starts(selected-fps_selected),
        selected_starts=starts(selected))
    evidence = dict(metadata, mode='mean_fps_quota', selected_starts=starts(selected),
        selected_tokens=sum(r.count for r in records if r.handle.slot in selected),
        selected_blocks=len(selected), charged_tokens=32*len(selected), budget=budget,
        fixed_starts=starts(fixed), quota_selection=quota)
    return dict(groups=groups, group_scores=group_scores, mandatory_slots=tuple(sorted(protected)),
        selected_slots=tuple(sorted(selected)), group_selection=evidence, quota_selection=quota)


def quota_summary(quota):
    return {key: value for key, value in quota.items() if not key.endswith('_starts')}
