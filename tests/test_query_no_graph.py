"""Unchanged native quota/halo choices without allocating any block graph."""
import json
from types import SimpleNamespace as NS

import pytest

from test_query_hybrid_recall import populated, index
from test_query_group_picker import finish
from vllm_kvmem.memory import Phase
from vllm_kvmem.query_boundary_history import QueryBoundaryHistory
from vllm_kvmem.query_hybrid_recall import preview_hybrid_selection, quota_summary
from vllm_kvmem.query_no_graph import select_without_graph


def authority(tmp_path, *, count=384, fraction=.5, halo=1, trace=False):
    memory, handles = populated(count=count, budget=224, repository=20)
    rows = index(handles, count)
    boundaries = QueryBoundaryHistory('hybrid')
    boundaries.advance(0, count)
    obj = NS(memory=memory, query_score_mode='key_fps_4', history_end=count,
        query_boundaries=boundaries,
        graph_options=NS(graph_mode='off', graph_alpha=0., query_fps_halo_blocks=halo,
            query_mean_quota_fraction=fraction, graph_trace=trace),
        query_report_path=tmp_path/'query_recall.jsonl',
        query_score_domain=dict(version='test', score_mask=[True]*len(rows)))
    return obj, rows


@pytest.mark.parametrize('fraction', [.25, .5, 1.])
@pytest.mark.parametrize('messages', [(), ((0, 128, 'user'), (128, 384, 'tool'))])
def test_native_hybrid_same_scores_budget_and_message_boundaries(tmp_path, fraction, messages):
    obj, rows = authority(tmp_path, fraction=fraction, trace=True)
    obj.query_boundaries.update_message_spans(messages)
    fps, mean = [0.,0.,0.,0.,0.,0.,0.,0.,100.,0.,0.,0.], [0.,10.,9.,8.,0.,0.,0.,0.,0.,0.,0.,0.]
    expected = preview_hybrid_selection(obj.memory, rows, fps, mean,
        mean_fraction=fraction, message_spans=messages)
    plan = select_without_graph(obj, rows, fps, [False]*len(rows), mean_scores=mean)
    assert {r.block_id for r in plan.remaps} == set(expected['selected_slots'])
    assert obj.query_quota_selection == quota_summary(expected['quota_selection'])
    assert obj.query_quota_propagation is None
    assert obj.query_selection_report['mode'] == 'off'
    assert obj.query_selection_report['propagation_applied'] is False
    assert not hasattr(obj, 'graph_selection_report')
    trace = json.loads((tmp_path/'query_recall_inputs.jsonl').read_text())
    assert trace['raw_scores'] == fps and trace['mean_scores'] == mean
    assert 'graph' not in trace
    finish(obj.memory)
    del plan
    assert obj.memory.close()


@pytest.mark.parametrize('halo', [0, 1])
def test_pure_fps_path_without_a_mean_quota(tmp_path, halo):
    obj, rows = authority(tmp_path, fraction=0, halo=halo)
    plan = select_without_graph(obj, rows, list(range(len(rows))), [True]*len(rows))
    assert plan.total_window_tokens <= 224
    assert obj.query_selection_report['propagation_applied'] is False
    finish(obj.memory)
    del plan
    assert obj.memory.close()


def test_pruning_unrelated_metadata_during_assembly_does_not_invalidate_selected_plan(tmp_path):
    obj, rows = authority(tmp_path)
    obj.query_boundaries.update_message_spans([(0, 64, 'user'), (128, 256, 'tool')])
    plan = select_without_graph(obj, rows, [1.]*len(rows), [True]*len(rows), mean_scores=[1.]*len(rows))
    # All selected groups retain their native leases. Dropping already retired
    # unrelated boundary metadata must not manufacture a stale-revision error.
    obj.query_boundaries.prune([(r.history_begin, r.history_begin+r.count) for r in rows])
    finish(obj.memory)
    del plan
    assert obj.memory.close()


def test_changed_message_semantics_invalidates_pending_assembly(tmp_path):
    obj, rows = authority(tmp_path)
    plan = select_without_graph(obj, rows, [1.]*len(rows), [True]*len(rows), mean_scores=[1.]*len(rows))
    obj.query_boundaries.update_message_spans([(0, 32, 'tool')])
    with pytest.raises(RuntimeError, match='Stale'):
        finish(obj.memory)
    del plan
    assert obj.memory.close()


def test_diagnostic_failure_happens_before_native_selection_transaction(tmp_path, monkeypatch):
    obj, rows = authority(tmp_path)
    before = obj.memory._store.working_records()
    def fail(*_): raise OSError('diagnostic disk failure')
    monkeypatch.setattr('vllm_kvmem.query_no_graph._write_evidence', fail)
    with pytest.raises(OSError, match='disk'):
        select_without_graph(obj, rows, [1.]*len(rows), [True]*len(rows), mean_scores=[1.]*len(rows))
    assert obj.memory.phase is Phase.READY and obj.memory._plan is None
    assert obj.memory._store.working_records() == before
    assert obj.memory.close()


def test_unknown_boundaries_stay_strict_after_every_known_span_is_collected(tmp_path):
    obj, rows = authority(tmp_path, fraction=.5, trace=True)
    obj.query_boundaries.update_message_spans([(0, 16, 'user')])
    obj.query_boundaries.prune([(32, 384)])
    plan = select_without_graph(obj, rows, [1.]*len(rows), [True]*len(rows), mean_scores=[1.]*len(rows))
    trace = json.loads((tmp_path/'query_recall_inputs.jsonl').read_text())
    assert trace['message_metadata_present'] and not trace['message_spans']
    assert all(len(group) == 1 for group in trace['query_group_selection']['group_members'])
    finish(obj.memory)
    del plan
    assert obj.memory.close()


def test_async_validator_tracks_only_selected_generations_after_score_snapshot_is_released(tmp_path):
    obj, rows = authority(tmp_path)
    plan = select_without_graph(obj, rows, [1.]*len(rows), [True]*len(rows), mean_scores=[1.]*len(rows))
    selected = {r.block_id for r in plan.remaps}
    unselected = next(r.handle.slot for r in rows if r.handle.slot not in selected)
    selected_slot = next(iter(selected))
    original = obj.memory._store
    missing = {unselected}

    class ReusedStore:
        def handle(self, slot):
            return None if slot in missing else original.handle(slot)

    try:
        obj.memory._store = ReusedStore()
        obj.memory._selection_validator()
        missing.add(selected_slot)
        with pytest.raises(RuntimeError, match='recycled'):
            obj.memory._selection_validator()
    finally:
        obj.memory._store = original
    finish(obj.memory)
    del plan
    assert obj.memory.close()
