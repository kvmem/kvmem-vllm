"""Independent graph seeds precede the native quota, masks, and atomic picker."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.graph_scoring import score_graph
from vllm_kvmem.memory import Phase
from vllm_kvmem.query_hybrid_recall import preview_hybrid_selection, quota_summary
from vllm_kvmem.query_group_recall import halo_summary


@pytest.mark.parametrize('native', [False, True])
def test_actual_authority_propagates_both_original_seeds_before_reserving_quota(tmp_path, monkeypatch, native):
    from test_query_key_fps_integration import actual
    from vllm_kvmem.authority import _Ready
    authority, archive, graph = actual(tmp_path,select_budget=128)
    opts = authority.graph_options
    opts.query_fps_halo_blocks = 1
    opts.query_mean_quota_fraction = .5
    opts.graph_alpha = .2
    opts.graph_native_propagation = native
    opts.graph_trace = True
    authority.graph_report_path = tmp_path/'block_graph.jsonl'
    monkeypatch.setattr('vllm_kvmem.query_key_fps.score_authority_fps',
        lambda *args: (torch.tensor([0.,.05,.15,.25,.15,.4]), {}))
    plan = authority.select_for_queries(torch.ones(1,1,1,2))
    trace = json.loads((tmp_path/'block_graph_inputs.jsonl').read_text())
    report = json.loads((tmp_path/'block_graph.jsonl').read_text())
    assert report['query_quota_propagation']['order'] == 'independent_graph_then_quota'
    with graph.snapshot() as snapshot, authority.index_snapshot() as (records, _):
        vectors = {}
        for name,seed in [('key_fps_4',trace['raw_scores']),('mean_k',trace['mean_scores'])]:
            with score_graph(seed,trace['starts'],trace['transport_mask'],snapshot,
                    scoring_version='v2',alpha=.2,mode='multi',block_tokens=32) as reference:
                actual_scores = trace['quota_propagated_scores'][name]
                assert actual_scores == pytest.approx(reference.scores,rel=1e-13,abs=1e-14)
                assert actual_scores != seed
                assert all(keep or a == b for keep,a,b in zip(trace['transport_mask'],actual_scores,seed))
                assert report['query_quota_propagation'][name]['revision'] == snapshot.revision
                vectors[name] = actual_scores
        # The quota's newly protected MeanK pages must not be removed from the
        # original transport mask before scoring the FPS branch.
        import analyze_block_graph as offline
        cfg,_ = offline._validate_record(trace)
        replay,handles = offline._native_history(cfg,trace['starts'],trace['counts'])
        try:
            rows = [NS(handle=handles[r.history_begin],history_begin=r.history_begin,count=r.count) for r in records]
            expected = preview_hybrid_selection(replay,rows,vectors['key_fps_4'],vectors['mean_k'],
                max_tokens=128,message_spans=snapshot.message_spans)
            assert expected['group_selection'] == trace['query_group_selection']
            assert expected['quota_selection']['charged_tokens'] <= 128
        finally:
            assert replay.close()
    authority.memory.attach_completion(_Ready()); assert authority.memory.commit_selection()
    del plan
    assert authority.close() and archive.close()


def test_second_branch_failure_releases_first_branch_before_any_assembly(tmp_path, monkeypatch):
    from test_query_key_fps_integration import actual
    authority, archive, graph = actual(tmp_path,select_budget=128)
    authority.graph_options.query_fps_halo_blocks = 1
    authority.graph_options.query_mean_quota_fraction = .5
    authority.graph_options.graph_alpha = .2
    before = authority.memory._store.working_records()
    used = graph.budget.used
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise BufferError('second seed capacity')
        return score_graph(*args,**kwargs)
    monkeypatch.setattr('vllm_kvmem.graph_scoring.score_graph',fail_second)
    with pytest.raises(BufferError,match='second seed capacity'):
        authority.select_for_queries(torch.ones(1,1,1,2))
    assert len(calls) == 2
    assert authority.memory.phase is Phase.READY and authority.memory._plan is None
    assert authority.memory._store.working_records() == before
    assert authority.memory._store.runtime_select_budget == 0
    assert graph.budget.used == used
    assert authority.close() and archive.close()


def propagated_fixture():
    import analyze_block_graph as offline
    from test_ruler_hybrid_protocol import fixture
    query,graph,inputs,profile = fixture()
    profile['kvmem']['graph_alpha'] = .2
    inputs['alpha'] = .2
    cfg,snapshot = offline._validate_record(inputs)
    values, diagnostics = {}, {}
    for name,seed in [('key_fps_4',inputs['raw_scores']),('mean_k',inputs['mean_scores'])]:
        with score_graph(seed,inputs['starts'],inputs['transport_mask'],snapshot,
                scoring_version='v2',alpha=.2,mode='multi',block_tokens=32,
                cross_message_weight=inputs['cross_message_weight']) as result:
            values[name] = list(result.scores)
            diagnostics[name] = dict(result.diagnostics)
    inputs['quota_propagated_scores'] = values
    graph.update(diagnostics['key_fps_4'])
    graph.update(seed_source='key_fps_4',transport_mask_count=sum(inputs['transport_mask']),
        propagation_applied=not diagnostics['key_fps_4']['identity'])
    graph['query_quota_propagation'] = dict(order='independent_graph_then_quota',**diagnostics)
    query['query_quota_propagation'] = deepcopy(graph['query_quota_propagation'])
    memory, handles = offline._native_history(cfg,inputs['starts'],inputs['counts'])
    try:
        records = [NS(handle=handles[s],history_begin=s,count=n) for s,n in zip(inputs['starts'],inputs['counts'])]
        preview = preview_hybrid_selection(memory,records,values['key_fps_4'],values['mean_k'],
            tuple(handles[s].slot for s in inputs['mandatory_starts']),max_tokens=inputs['budget'],
            message_spans=inputs['graph']['message_spans'])
    finally:
        assert memory.close()
    inputs['query_group_selection'] = preview['group_selection']
    summary = halo_summary(preview['group_selection'])
    summary['quota_selection'] = quota_summary(preview['quota_selection'])
    for row in (query,graph):
        row['query_group_selection'] = deepcopy(summary)
        row['query_quota_selection'] = quota_summary(preview['quota_selection'])
    counts = dict(zip(inputs['starts'],inputs['counts']))
    query['selected_history_ranges'] = [[s,counts[s]] for s in preview['quota_selection']['selected_starts']]
    return query,graph,inputs,profile


def test_independent_full_vector_protocol_replay_and_native_quota():
    import ruler_hybrid_protocol as policy
    audit = policy.audit_selection(*propagated_fixture())
    assert audit['independent_graph_vectors_verified'] and audit['independent_mean_quota_verified']
    assert audit['graph_alpha'] == .2 and audit['propagation_applied']


@pytest.mark.parametrize('fault',['missing_branch','raw_mean','shared_vector','binding','mask','quota'])
def test_audit_rejects_partial_propagation_blending_mask_or_version_changes(fault):
    import ruler_hybrid_protocol as policy
    query,graph,inputs,profile = propagated_fixture()
    if fault == 'missing_branch': del graph['query_quota_propagation']['mean_k']
    elif fault == 'raw_mean': inputs['quota_propagated_scores']['mean_k'] = inputs['mean_scores']
    elif fault == 'shared_vector': inputs['quota_propagated_scores']['mean_k'] = inputs['quota_propagated_scores']['key_fps_4']
    elif fault == 'binding':
        graph['query_quota_propagation']['mean_k']['revision'] += 1
        query['query_quota_propagation'] = deepcopy(graph['query_quota_propagation'])
    elif fault == 'mask':
        graph['query_quota_propagation']['mean_k']['eligible'] -= 1
        query['query_quota_propagation'] = deepcopy(graph['query_quota_propagation'])
    else: inputs['query_group_selection']['quota_selection']['mean_quota_blocks'] += 1
    with pytest.raises(ValueError): policy.audit_selection(query,graph,inputs,profile)


def test_graph_profile_changes_only_alpha_and_has_explicit_experiment_identity():
    import ruler_hybrid_protocol as policy
    from test_ruler_fps_cpu_protocol import full_fixture
    _,data = full_fixture()
    base = policy.expected_profile(data['reference_profile'])
    graph = policy.expected_profile(data['reference_profile'],graph_alpha=.2)
    assert graph == dict(base,kvmem=dict(base['kvmem'],graph_alpha=.2))
    data['execution_profile'] = policy.GRAPH_PROFILE
    policy.validate_execution(graph,data)
    with pytest.raises(ValueError): policy.validate_execution(base,data)


def test_launch_roundtrip_preserves_explicit_hybrid_graph(tmp_path):
    from test_graph_options import model_profile,runtime,worker_payload
    from test_query_mean_quota_options import quota_settings
    from test_query_fps_prebuild_options import elastic_profile
    from vllm_kvmem.serve import build_launch
    from vllm_kvmem.elastic_serve import build_launch as build_elastic
    profile = model_profile(tmp_path)
    profile['kvmem'] = quota_settings(graph_alpha=.2)
    worker = worker_payload(build_launch(profile,tmp_path,tmp_path/'direct'))
    assert runtime(worker).graph_alpha == .2
    elastic = elastic_profile(tmp_path,profile)
    elastic.update(graph_alpha=.2,query_fps_halo_blocks=1,query_mean_quota_fraction=.5)
    resolved,_ = build_elastic(elastic,tmp_path,tmp_path/'elastic')
    actual = worker_payload(build_launch(resolved['mean_k_backend_profile'],tmp_path,tmp_path/'backend'))
    assert runtime(actual).graph_alpha == .2 and actual['query_mean_quota_fraction'] == .5


def test_memory_graph_arm_is_explicit_and_keeps_original_geometry():
    import hybrid_memory_protocol as policy
    from test_hybrid_memory_campaign import profile
    for suite in ('longmemeval_s','memoryagentbench'):
        value = profile(suite); value['kvmem']['graph_alpha'] = .2
        assert policy.infer_arm(value) == 'mean_fps_half_graph'
        policy.validate_profile(value,'mean_fps_half_graph',suite)
        with pytest.raises(ValueError): policy.validate_profile(value,'mean_fps_half',suite)
