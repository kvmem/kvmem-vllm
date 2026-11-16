"""Candidate block lists stored as runs in the decision trace (core/block_runs.py)."""
import json
import random

import pytest

from vllm_kvmem.core.block_runs import candidate_starts, decode_runs, encode_starts
from vllm_kvmem.strategy import trace


@pytest.mark.parametrize('starts', [
    [], [0], [0, 32, 64, 96], [0, 32, 96, 128, 1024],
    [64, 32, 0], [0, 0, 32], [96, 32, 64, 0, 128, 160],
])
def test_any_list_round_trips_in_order(starts):
    runs = encode_starts(starts)
    assert decode_runs(runs) == starts
    assert all(type(v) is int for run in runs for v in run)
    assert json.loads(json.dumps(runs)) == runs


def test_consecutive_blocks_collapse_to_few_runs():
    rng = random.Random(0)
    starts = sorted(rng.sample(range(0, 32 * 20000, 32), 18000))
    runs = encode_starts(starts)
    assert decode_runs(runs) == starts
    assert len(json.dumps(runs)) * 3 < len(json.dumps(starts))


def test_reader_accepts_new_runs_and_older_full_lists():
    assert candidate_starts(dict(candidate_runs=[[0, 3], [128, 2]])) == [0, 32, 64, 128, 160]
    assert candidate_starts(dict(candidate_starts=[5, 1])) == [5, 1]
    assert candidate_starts({}) == []


def test_trace_reads_both_formats_identically(tmp_path):
    def capture(form):
        run = tmp_path / form
        cap = run / 'run' / 'server' / 'capture'
        cap.mkdir(parents=True)
        starts = [0, 32, 64, 160]
        field = (dict(candidate_runs=encode_starts(starts)) if form == 'runs'
                 else dict(candidate_starts=starts))
        rows = [dict(event='pressure_selected', request_id='r', accepted_history=192, event_type='prefill_pressure',
                     layers=[dict(layer=0, actual_method='accumulated_attention_per_layer', selected_starts=[0, 160],
                                  candidate_scores=[.4, .3, .2, .1], mandatory_starts=[0], budget_tokens=64,
                                  sink_tokens=32, **field)]),
                dict(event='query_candidates', request_id='r', budget=64, history=192, candidate_counts=[32] * 4,
                     eligible=[True] * 4, fixed_starts=[0], excluded_starts=[], **field),
                dict(event='layer_selected', request_id='r', layer=0, selected_starts=[0, 64],
                     candidate_scores=[.1, .2, .3, .4])]
        (cap / 'layerwise_query.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return trace.read(run)
    runs, full = capture('runs'), capture('list')
    assert runs == full and len(runs) == 2
    assert [d.candidates for d in runs] == [(0, 32, 64, 160)] * 2
