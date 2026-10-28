"""Quota preservation and native atomic assembly, with no model dependency."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from test_query_group_picker import finish
from test_core import append
from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory
from vllm_kvmem.memory import Phase
from vllm_kvmem.query_hybrid_recall import preview_hybrid_selection


def populated(*, count=256, budget=128, repository=16, sink=1, recent=0):
    memory = RequestMemory('hybrid', KVMemConfig(block_tokens=32, select_budget=budget,
        gen_budget=256, repository_blocks=repository, sink_blocks=sink, recent_blocks=recent,
        action_fraction=0))
    spans = append(memory, count)
    return memory, [_native.Handle(s.handle.slot, s.handle.generation) for s in spans]


def index(handles, count):
    return [NS(handle=h, history_begin=i*32, count=min(32, count-i*32))
        for i, h in enumerate(handles)]


def test_mean_quota_survives_fps_competition_and_one_atomic_commit():
    memory, handles = populated(count=384, budget=224, repository=20)
    rows = index(handles, 384)
    try:
        mean = [0.,10.,9.,8.,0.,0.,0.,0.,0.,0.,0.,0.]
        fps = [0.,0.,0.,0.,0.,0.,0.,0.,100.,0.,0.,0.]
        preview = preview_hybrid_selection(memory, rows, fps, mean)
        quota = preview['quota_selection']
        assert quota['mean_quota_blocks'] == quota['mean_selected_blocks'] == 3
        assert quota['mean_selected_starts'] == [32,64,96]
        assert quota['fps_added_starts'] == [224,256,288]
        assert quota['mean_fill_blocks'] == 0
        assert quota['selected_starts'] == [0,32,64,96,224,256,288]
        assert memory.phase is Phase.READY and memory._plan is None
        plan = memory.select_groups(preview['groups'], preview['group_scores'],
            preview['mandatory_slots'], expected_slots=preview['selected_slots'])
        assert plan.total_window_tokens == 224
        finish(memory)
        del plan
    finally:
        assert memory.close()


def test_stranded_group_slots_fill_in_mean_order_without_post_picker_expansion():
    memory, handles = populated(count=384, budget=128, repository=20)
    rows = index(handles, 384)
    try:
        # One MeanK slot leaves two. No non-overlapping three-block FPS group
        # fits; boundary group at the end does fit and is admitted atomically.
        preview = preview_hybrid_selection(memory, rows, [0.]*12, [0.,0.,0.,9.,8.,7.,0.,0.,0.,0.,0.,0.])
        quota = preview['quota_selection']
        assert quota['mean_selected_starts'] == [96]
        assert quota['selected_starts'] == [0,96,320,352]
        assert quota['charged_tokens'] == 128
        # Separate messages make every FPS group a singleton, also bounded.
        other = preview_hybrid_selection(memory, rows, [0.]*12, [0.]*12,
            message_spans=[(i*32,(i+1)*32,'user') for i in range(12)])
        assert len(other['selected_slots']) == 4
        assert other['group_selection']['post_picker_expansion'] is False
    finally:
        assert memory.close()


@pytest.mark.parametrize('fraction', [.25,.5,1.])
def test_partial_tail_original_fixed_pins_and_all_fit(fraction):
    memory, handles = populated(count=197, budget=224, sink=1, recent=1)
    rows = index(handles,197)
    try:
        preview = preview_hybrid_selection(memory, rows, [1.]*7, [1.]*7,
            (handles[2].slot,), mean_fraction=fraction)
        assert set(preview['selected_slots']) == {h.slot for h in handles}
        assert preview['group_selection']['selected_tokens'] == 197
        assert preview['quota_selection']['charged_tokens'] == 224
        assert preview['quota_selection']['fixed_starts'] == list(range(0,197,32))
        assert memory._store.runtime_select_budget == 0
    finally:
        assert memory.close()


def test_zero_first_stage_quota_and_runtime_budget_do_not_expand_to_configured_budget():
    memory, handles = populated(count=384, budget=224, repository=20, sink=0)
    try:
        preview = preview_hybrid_selection(memory, index(handles,384), [1.]*12,
            [1.]*12, max_tokens=32)
        assert len(preview['selected_slots']) == 1
        assert preview['quota_selection']['mean_quota_blocks'] == 0
        assert preview['quota_selection']['mean_fill_blocks'] == 1
        assert memory._store.runtime_select_budget == 0
    finally:
        assert memory.close()


def test_actual_authority_exports_two_independent_full_domain_scores(tmp_path, monkeypatch):
    from test_query_key_fps_integration import actual
    from vllm_kvmem.authority import _Ready
    from vllm_kvmem.retrieval import mean_key_scores
    authority, archive, _ = actual(tmp_path, select_budget=128)
    authority.graph_options.query_fps_halo_blocks = 1
    authority.graph_options.query_mean_quota_fraction = .5
    authority.graph_options.graph_trace = True
    authority.graph_report_path = tmp_path/'block_graph.jsonl'
    monkeypatch.setattr('vllm_kvmem.query_key_fps.score_authority_fps',
        lambda *args: (torch.tensor([0., .5, .4, .9, .001, .002]), {}))
    queries = torch.ones(1,1,1,2)
    plan = authority.select_for_queries(queries)
    trace = json.loads((tmp_path/'block_graph_inputs.jsonl').read_text())
    with authority.index_snapshot() as (_, means):
        expected = mean_key_scores(queries, means, torch.tensor(trace['score_mask'])).tolist()
    assert trace['mean_scores'] == expected
    assert trace['raw_scores'] != expected
    assert trace['query_mean_quota_fraction'] == .5
    quota = trace['query_group_selection']['quota_selection']
    assert quota['mean_selected_blocks'] == quota['mean_quota_blocks'] == 1
    assert quota['charged_tokens'] <= 128
    assert authority.query_quota_selection['mode'] == 'mean_fps_quota'
    authority.memory.attach_completion(_Ready()); assert authority.memory.commit_selection()
    del plan
    assert authority.close() and archive.close()


def test_failure_precedes_assembly_and_does_not_mutate_another_owner(tmp_path, monkeypatch):
    from test_query_key_fps_integration import actual
    authority, archive, _ = actual(tmp_path, select_budget=128)
    authority.graph_options.query_fps_halo_blocks = 1
    authority.graph_options.query_mean_quota_fraction = .5
    before = authority.memory._store.working_records()
    def fail(*args, **kwargs):
        raise OSError('hybrid diagnostic failure')
    monkeypatch.setattr('vllm_kvmem.graph_integration.emit_graph_report', fail)
    with pytest.raises(OSError, match='hybrid diagnostic'):
        authority.select_for_queries(torch.ones(1,1,1,2))
    assert authority.memory.phase is Phase.READY and authority.memory._plan is None
    assert authority.memory._store.working_records() == before
    assert authority.memory._store.runtime_select_budget == 0
    assert authority.close() and archive.close()
