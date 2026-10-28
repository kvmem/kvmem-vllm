from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

import analyze_block_graph as offline
import ruler_hybrid_protocol as policy
from test_ruler_fps_halo_answers import audit_fixture
from test_ruler_fps_cpu_protocol import full_fixture
from vllm_kvmem.query_hybrid_recall import preview_hybrid_selection, quota_summary
from vllm_kvmem.query_group_recall import halo_summary


def fixture():
    query, graph, inputs, profile = audit_fixture()
    assert query['query_score_tokens'].is_integer()
    query['query_score_tokens'] = int(query['query_score_tokens'])
    profile['kvmem']['query_mean_quota_fraction'] = .5
    inputs['query_mean_quota_fraction'] = .5
    inputs['mean_scores'] = list(reversed(inputs['raw_scores']))
    # Preserve the native score domain even when the reversed vector changes
    # which scorable blocks receive mass.
    mass = sum(inputs['raw_scores'])
    inputs['mean_scores'] = [float(keep) for keep in inputs['score_mask']]
    total = sum(inputs['mean_scores'])
    inputs['mean_scores'] = [mass*s/total for s in inputs['mean_scores']]
    cfg, _ = offline._validate_record(inputs)
    memory, handles = offline._native_history(cfg, inputs['starts'],inputs['counts'])
    try:
        records = [NS(handle=handles[s],history_begin=s,count=n)
            for s,n in zip(inputs['starts'],inputs['counts'])]
        result = preview_hybrid_selection(memory,records,inputs['raw_scores'],inputs['mean_scores'],
            tuple(handles[s].slot for s in inputs['mandatory_starts']),max_tokens=inputs['budget'],
            message_spans=inputs['graph']['message_spans'])
    finally:
        assert memory.close()
    evidence = result['group_selection']; inputs['query_group_selection'] = evidence
    brief = halo_summary(evidence); brief['quota_selection'] = quota_summary(result['quota_selection'])
    for row in (query,graph):
        row['query_group_selection'] = deepcopy(brief)
        row['query_quota_selection'] = quota_summary(result['quota_selection'])
    counts = dict(zip(inputs['starts'],inputs['counts']))
    query['selected_history_ranges'] = [[s,counts[s]] for s in evidence['selected_starts']]
    query['query_key_fps_scoring']['mean_quota_scoring_seconds'] = .01
    return query,graph,inputs,profile


def test_hybrid_native_replay_and_independent_mean_ranking():
    query,graph,inputs,profile = fixture()
    audit = policy.audit_selection(query,graph,inputs,profile)
    assert audit['native_group_picker_replay_equal']
    assert audit['independent_mean_quota_verified']
    assert audit['quota']['mean_fraction'] == .5


@pytest.mark.parametrize('damage',['ratio','mean_mass','mean_mask','quota','selection','summary'])
def test_audit_rejects_changed_scores_geometry_and_budget(damage):
    query,graph,inputs,profile = fixture()
    if damage == 'ratio': inputs['query_mean_quota_fraction'] = .25
    elif damage == 'mean_mass': inputs['mean_scores'][0] += 1.
    elif damage == 'mean_mask': inputs['score_mask'][0] = False; inputs['mean_scores'][0] = 1.
    elif damage == 'quota': inputs['query_group_selection']['quota_selection']['mean_quota_blocks'] += 1
    elif damage == 'selection': query['selected_history_ranges'].pop()
    else: graph['query_quota_selection']['fps_added_blocks'] += 1
    with pytest.raises(ValueError): policy.audit_selection(query,graph,inputs,profile)


def test_profile_only_adds_explicit_quota_to_cpu_fps_control():
    import ruler_fps_cpu_protocol as original
    profile, data = full_fixture()
    data['execution_profile'] = policy.PROFILE
    expected = policy.expected_profile(data['reference_profile'])
    assert expected == dict(profile, kvmem=dict(profile['kvmem'],query_mean_quota_fraction=.5))
    policy.validate_execution(expected,data)
    original.validate_execution(profile,dict(data,execution_profile=original.PROFILE))
    expected['kvmem']['query_mean_quota_fraction'] = .25
    with pytest.raises(ValueError): policy.validate_execution(expected,data)


def test_native_mean_mass_accounts_for_actual_fp32_reduction_geometry(tmp_path):
    import json
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(num_hidden_layers=64,num_attention_heads=24,
        layer_types=['linear_attention']*48+['full_attention']*16)))
    profile=dict(hf_config_path=str(config),kvmem=dict(mean_k_backend='qw3_cuda'))
    query=dict(query_score_tokens=3718,mean_k_backend='qw3_cuda')
    inputs=dict(score_mask=[True]*128)
    # Fixed-zero-QK native GPU output; every exact score is T/B. No input
    # quantization error is possible, but 1,427,712 FP32 atomic adds drift.
    mean=[29.349132537841797]*128;before=mean[:]
    fps=[3718/128]*128
    out=policy.audit_seed_mass(mean,fps,query,inputs,profile)
    assert out['route']=='atomic_fp32' and out['distributions']==3718*16*24
    assert out['mean_k_absolute_error']>1 and mean==before
    assert out['model_config_sha256'] and not out['score_vectors_changed']
    with pytest.raises(ValueError,match='roundoff bound'):
        policy.audit_seed_mass([s*.5 for s in mean],fps,query,inputs,profile)
    with pytest.raises(ValueError,match='FPS Query seed mass'):
        policy.audit_seed_mass(mean,[s*.99 for s in fps],query,inputs,profile)
    with pytest.raises(ValueError,match='geometry'):
        policy.audit_seed_mass(mean,fps,query,inputs,dict(kvmem=profile['kvmem']))
    tiled=policy.mean_key_mass_bound(3718,8193,16,24)
    assert tiled['route']=='tiled_fp32' and tiled['relative_bound']<.001
    with pytest.raises(ValueError,match='too loose'):
        policy.mean_key_mass_bound(1000000,128,16,24)
