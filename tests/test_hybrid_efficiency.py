from copy import deepcopy
import json

import pytest

from prepare_hybrid_efficiency import choose,prepare


def rows():
    values = [('event_q0000',755,142),('event_q0001',754,130),
        ('longbook_q0000',757,100),('shortbook_q0000',267,200),
        ('medium_q0000',411,175),('facts_q0000',337,9),
        ('1_longmemeval_s_q0000',365,41),('2_longmemeval_s_q0000',367,45)]
    return [dict(id=i,input_tokens=n,query_begin=n-q,query_end=n,gold_answer='unused') for i,n,q in values]


def test_pilot_selection_uses_shapes_not_scores_or_gold_and_keeps_one_warm_prefix():
    original=rows();before=deepcopy(original)
    selected=choose(original)
    assert original == before
    modified=deepcopy(original)
    for row in modified:row.update(gold_answer='different',score=0,answer='different')
    assert choose(modified) == selected
    by_id={v['id']:v for v in selected}
    assert {'event_q0000','event_q0001','longbook_q0000','shortbook_q0000','facts_q0000'} <= set(by_id)
    assert 'minimum_input' in by_id['shortbook_q0000']['reasons']
    assert 'maximum_query' in by_id['shortbook_q0000']['reasons']
    assert 'shortest_query_first_in_group' in by_id['facts_q0000']['reasons']


def test_invalid_shapes_and_overwrite_are_rejected(tmp_path):
    data=rows();data[0]['query_begin']=data[0]['query_end']
    with pytest.raises(ValueError,match='shape'):choose(data)
    with pytest.raises(ValueError,match='Duplicate'):choose(rows()+rows()[:1])
    marker=tmp_path/'response.json';marker.write_text('preserve')
    with pytest.raises(ValueError,match='fresh efficiency'):prepare(tmp_path,tmp_path,graph_alpha=.2)
    assert marker.read_text() == 'preserve'


def paired_results(root):
    from benchmark_niah import save
    for arm,alpha in [('graph-off',0.),('graph-on',.2)]:
        campaign=root/arm;case=campaign/'run/answers/cases/example_q0000'
        save(campaign/'status.json',dict(status='completed'))
        save(campaign/'execution-audit.json',dict(status='passed'))
        save(campaign/'run/cleanup.json',dict(remaining_owned_groups=[],gpu_processes=[]))
        save(campaign/'runtime-proof.json',dict(files={'engine':'same'}))
        save(campaign/'profile.json',dict(kvmem=dict(graph_alpha=alpha,query_mean_quota_fraction=.5)))
        save(campaign/'fixture.json',dict(case_ids=['example_q0000'],rows=[dict(id='example_q0000')]))
        fps=dict(cache_build_seconds=.01,scoring_seconds=.2,mean_quota_scoring_seconds=.05,
            cache_hit_blocks=100,cache_prebuilt_hit_blocks=100,cache_rebuilt_blocks=0,
            cache_bytes=1024,cache_capacity_bytes=12*1024**3,
            scoring_path='tiled_layer_cache',scoring_scratch_bound_bytes=32*1024**2)
        query=dict(query_score_tokens=10,query_tokens=32,refresh_finished_at=15.,refresh_started_at=10.,
            query_key_fps_scoring=fps,assembly_seconds=.1)
        graph=dict(phase='query',graph_selection_seconds=.02,
            query_quota_propagation=dict(mean_k=dict(backend='cpp_v2',scoring_seconds=.005),
                key_fps_4=dict(backend='cpp_v2',scoring_seconds=.005)))
        save(case/'result.json',dict(prompt_tokens=3200,prefix_reused=False,seconds=9.,
            metrics=dict(time_to_first_token_ms=8000.,generation_time_ms=1000.,mean_itl_ms=100.),
            usage=dict(completion_tokens=11),graph_v2=dict(query=query,graph=[graph]),
            finish_reason='stop',raw_completion=arm))
        save(case/'request.json',dict(seed=9,max_tokens=100,prompt_tokens=3200,session_id=arm,cache_salt=arm))


def test_efficiency_comparison_preserves_units_and_does_not_gate_on_changed_output(tmp_path):
    from analyze_hybrid_efficiency import compare
    paired_results(tmp_path)
    report=compare(tmp_path)
    case=report['per_case'][0]
    assert case['on']['ttft_seconds']==8 and case['on']['decode_seconds']==1
    assert case['output_changed'] is True
    assert report['no_accuracy_gate'] and not report['full_campaigns_started']
    assert report['status']=='requires_efficiency_review_before_full'


def test_efficiency_comparison_rejects_changed_seed_or_unfinished_run(tmp_path):
    from analyze_hybrid_efficiency import compare
    from benchmark_niah import save
    paired_results(tmp_path)
    save(tmp_path/'graph-on/run/answers/cases/example_q0000/request.json',dict(seed=10,max_tokens=100,prompt_tokens=3200))
    with pytest.raises(ValueError,match='sampling/output limit'):compare(tmp_path)
    save(tmp_path/'graph-on/status.json',dict(status='testing'))
    with pytest.raises(ValueError,match='did not complete'):compare(tmp_path)


def test_recovery_split_never_regenerates_saved_failed_audit_response(tmp_path):
    from benchmark_niah import save
    from recover_hybrid_efficiency import split_cases
    fixture=dict(case_ids=['a_q0000','a_q0001','b_q0000'],
        rows=[dict(id=i) for i in ['a_q0000','a_q0001','a_q0002','b_q0000']])
    for ident in ['a_q0000','b_q0000']:
        save(tmp_path/'cases'/ident/'response.json',dict(text='unexamined'))
        save(tmp_path/'cases'/ident/'result.json',dict(status='error',score=0))
    assert split_cases(fixture,tmp_path)==(['a_q0000','b_q0000'],['a_q0001','a_q0002'])
    save(tmp_path/'cases/a_q0001/attempt.json',dict(started=True))
    with pytest.raises(ValueError,match='attempted'):split_cases(fixture,tmp_path)


def test_recovery_preparation_refuses_existing_root(tmp_path):
    from recover_hybrid_efficiency import prepare
    marker=tmp_path/'response.json';marker.write_text('preserve')
    with pytest.raises(ValueError,match='new efficiency'):prepare(tmp_path,tmp_path,tmp_path)
    assert marker.read_text()=='preserve'


def test_audit_only_clone_preserves_engine_and_exact_original_rows(tmp_path,monkeypatch):
    from benchmark_niah import save,sha
    import run_hybrid_memory_campaign as campaign
    import recover_hybrid_efficiency as recovery
    source=tmp_path/'source';repo=tmp_path/'repo';target=tmp_path/'target'
    fixture=dict(rows=rows(),qualification=True,case_ids=['old'],cases=1)
    profile=dict(kvmem=dict(graph_alpha=0.))
    save(source/'fixture.json',fixture);save(source/'profile.json',profile)
    save(source/'status.json',dict(status='prepared'));save(source/'prepared-proof.json',{})
    (source/'runtime/src').mkdir(parents=True);(source/'runtime/src/engine.py').write_text('immutable engine')
    for name in recovery.PATCHES:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new audit only')
        p=source/'runtime'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('old audit')
    files={str(p.relative_to(source/'runtime')):sha(p) for p in (source/'runtime').rglob('*') if p.is_file()}
    save(source/'runtime-proof.json',dict(files=files));monkeypatch.setattr(recovery,'REPO',repo)
    monkeypatch.setattr(campaign,'verify',lambda p:(json.loads((p/'fixture.json').read_text()),
        json.loads((p/'profile.json').read_text())))
    recovery.clone(source,target,['event_q0000'],alpha=.2,warm=0)
    actual=json.loads((target/'fixture.json').read_text())
    assert actual['rows']==fixture['rows'] and actual['efficiency_expected_warm']==0
    assert (source/'runtime/src/engine.py').read_bytes()==(target/'runtime/src/engine.py').read_bytes()
    assert (source/'runtime/scripts/ruler_hybrid_protocol.py').read_text()=='old audit'
    assert json.loads((target/'audit-update.json').read_text())['engine_sources_identical']


def dense_allocation_fixture(tmp_path):
    from pathlib import Path
    import shutil
    from benchmark_niah import save,sha
    name='src/vllm_kvmem/query_key_fps.py';source=Path(__file__).resolve().parents[1]/name
    target=tmp_path/'runtime'/name;target.parent.mkdir(parents=True);shutil.copy2(source,target)
    save(tmp_path/'runtime-proof.json',dict(files={name:sha(target)}))
    save(tmp_path/'model/config.json',dict(num_attention_heads=24,num_key_value_heads=4,head_dim=256))
    save(tmp_path/'profile.json',dict(model=str(tmp_path/'model'),kvmem=dict(query_fps_optimized=True)))
    query=dict(query_score_tokens=3718,score_mask_count=8183)
    fps=dict(optimized=True,cache_key_dtype='torch.float16',representatives_per_block=4,
        scoring_device='cuda:0',cache_hit_blocks=8224,cache_rebuilt_blocks=0)
    return query,fps


def test_old_dense_scorer_is_verified_from_exact_source_and_geometry(tmp_path):
    from analyze_hybrid_efficiency import scoring_allocation
    query,fps=dense_allocation_fixture(tmp_path)
    result=scoring_allocation(tmp_path,query,fps)
    assert result['path']=='dense_layer_cache' and result['bound_bytes']==432317696
    assert result['proof']['runtime_counter_missing'] and not result['proof']['measured_peak']
    fps['cache_hit_blocks']=23584
    with pytest.raises(ValueError,match='bounded dense branch'):scoring_allocation(tmp_path,query,fps)


def test_missing_counters_do_not_allow_arbitrary_source_or_partial_diagnostics(tmp_path):
    from analyze_hybrid_efficiency import scoring_allocation
    query,fps=dense_allocation_fixture(tmp_path)
    fps['scoring_path']='tiled_layer_cache'
    with pytest.raises(ValueError,match='Partial'):scoring_allocation(tmp_path,query,fps)
    fps.pop('scoring_path')
    with (tmp_path/'runtime/src/vllm_kvmem/query_key_fps.py').open('a') as f:f.write('\n# different source\n')
    with pytest.raises(ValueError,match='unverified'):scoring_allocation(tmp_path,query,fps)


def test_reassessment_does_not_overwrite_original_comparison(tmp_path):
    from analyze_hybrid_efficiency import compare
    paired_results(tmp_path)
    original=tmp_path/'comparison.json';original.write_text('preserved failed assessment')
    output=tmp_path/'new-comparison.json';compare(tmp_path,output=output)
    assert original.read_text()=='preserved failed assessment' and output.is_file()
