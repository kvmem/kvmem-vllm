"""Bounded, answer-blind logical neighbors for atomic Query group picking.

These are retrieval units only. Original 32-token KV blocks and raw FPS scores
are unchanged. No recursive expansion or post-picker neighbor append occurs.
"""
from bisect import bisect_right
import math


def build_halo_groups(records, scores, *, radius=1, message_spans=(),
                      message_metadata_present=None):
    if type(radius) is not int or radius != 1 or len(records) != len(scores):
        raise ValueError('Expected one-block halo and one score per original record')
    starts = [r.history_begin for r in records]
    counts = [r.count for r in records]
    if (any(type(s) is not int or s < 0 or s % 32 for s in starts)
            or any(type(n) is not int or not 0 < n <= 32 for n in counts)
            or any(a >= b for a, b in zip(starts, starts[1:]))
            or any(not math.isfinite(s) for s in scores)):
        raise ValueError('Invalid original logical halo records/scores')
    from .block_graph import _spans
    messages = _spans(message_spans)
    if message_metadata_present is not None and type(message_metadata_present) is not bool:
        raise ValueError('Message metadata presence must be boolean')
    constrained = bool(messages) or bool(message_metadata_present)
    message_starts = [s for s, _, _ in messages]
    membership = []
    for start, count in zip(starts, counts):
        index = bisect_right(message_starts, start)-1
        membership.append(index if index >= 0 and start+count <= messages[index][1] else None)

    def adjacent(left, right):
        return (starts[left]+counts[left] == starts[right] and
            (not constrained or membership[left] is not None and membership[left] == membership[right]))

    spans, members = [], []
    for i, start in enumerate(starts):
        left = i-1 if i > 0 and adjacent(i-1, i) else i
        right = i+1 if i+1 < len(records) and adjacent(i, i+1) else i
        spans.append((starts[left], starts[right]+counts[right]))
        members.append(starts[left:right+1])
    metadata = dict(mode='adjacent_1', radius=radius, atomic_groups=True,
        post_picker_expansion=False, coordinate='original_logical_tokens_half_open',
        anchor_starts=starts, groups=[list(s) for s in spans], group_scores=list(scores),
        group_members=members, message_boundary_policy='same_complete_message_or_no_metadata',
        tie_break='native group index descending', group_count=len(spans))
    return spans, list(scores), metadata


def describe_halo_selection(metadata, records, selected_slots, baseline_slots, fixed_slots, budget):
    """Derive evidence from the native result without changing selection."""
    selected_slots, baseline_slots, fixed_slots = map(set,
        (selected_slots, baseline_slots, fixed_slots))
    by_slot = {r.handle.slot: r for r in records}
    if not (selected_slots | baseline_slots | fixed_slots).issubset(by_slot):
        raise RuntimeError('Halo picker returned a foreign generation/slot')
    if not fixed_slots.issubset(selected_slots):
        raise RuntimeError('Halo selection lost fixed content')
    selected = {by_slot[s].history_begin for s in selected_slots}
    baseline = {by_slot[s].history_begin for s in baseline_slots}
    covered = [anchor for anchor, members in zip(metadata['anchor_starts'], metadata['group_members'])
        if set(members).issubset(selected)]
    covered_members = {start for members in metadata['group_members'] if set(members).issubset(selected)
        for start in members}
    fixed = {by_slot[s].history_begin for s in fixed_slots}
    if not (selected-fixed).issubset(covered_members):
        raise RuntimeError('Halo selection contains an orphaned partial group')
    tokens = sum(by_slot[s].count for s in selected_slots)
    if len(selected_slots)*32 > budget or tokens > budget:
        raise RuntimeError('Halo selection exceeds the original KV block budget')
    return dict(metadata, selected_starts=sorted(selected), selected_tokens=tokens,
        selected_blocks=len(selected), charged_tokens=32*len(selected),
        fixed_starts=sorted(fixed), baseline_selected_starts=sorted(baseline),
        added_starts=sorted(selected-baseline), displaced_starts=sorted(baseline-selected),
        covered_anchor_starts=covered, budget=budget, unused_block_slots=budget//32-len(selected),
        all_fit=len(records) <= budget//32)


def halo_summary(metadata):
    """Keep large per-anchor arrays only in the bounded Query input capture."""
    return {key: value for key, value in metadata.items()
        if key not in ('groups', 'group_scores', 'group_members', 'anchor_starts')}
