"""Block start lists stored as runs, for the decision trace in layerwise_query.jsonl.

A pressure decision lists every candidate block of every layer. Candidates are
mostly long stretches of consecutive 32-token blocks, so the list is stored as
``[first_start, count]`` runs: a run covers ``first_start, first_start + 32, ...``
for ``count`` blocks. Any list round-trips exactly, in its original order, so
per-candidate fields such as ``candidate_scores`` stay aligned.

Older traces wrote the full ``candidate_starts`` list; :func:`candidate_starts`
reads either form.
"""
import numpy as np

BLOCK_TOKENS = 32


def encode_starts(starts, block=BLOCK_TOKENS):
    """``[[first_start, count], ...]`` for a list of block starts (order kept)."""
    values = np.asarray(starts, dtype=np.int64)
    if values.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(values) != block) + 1
    firsts = np.concatenate(([0], breaks))
    counts = np.diff(np.concatenate((firsts, [values.size])))
    return np.stack((values[firsts], counts), axis=1).tolist()


def decode_runs(runs, block=BLOCK_TOKENS):
    """The block starts a run list stands for."""
    starts = []
    for first, count in runs:
        starts.extend(range(first, first + count * block, block))
    return starts


def candidate_starts(row, block=BLOCK_TOKENS):
    """A trace row's candidate block starts, from ``candidate_runs`` or the older full list."""
    if 'candidate_runs' in row:
        return decode_runs(row['candidate_runs'], block)
    return list(row.get('candidate_starts', ()))
