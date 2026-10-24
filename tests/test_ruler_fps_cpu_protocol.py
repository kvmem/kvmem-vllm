"""Fixed-cohort, generic-task and bounded evidence audits without model work."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import ruler_fps_cpu_protocol as policy
from test_ruler_protocol import cohort
from test_graph_v2_ablation import base_profile
from test_ruler_fps_halo_answers import audit_fixture


def source_fixture():
    rows=cohort(20)
    for row in rows:
        row.update(id=f"{row['task']}-{row['index']:04d}",seed=42+row['index'],served_input_tokens=262144)
    return dict(cases=260,samples_per_task=20,target_length=262144,rows=rows,workers=8,
        cpu_payload_credits_bytes=82820130816,request_timeout_seconds=3600,starts_in_thinking=True,
        sampling=deepcopy(policy.SAMPLING))


def reference_profile():
    p=base_profile();p.update(prefill_chunk_tokens=2048,max_num_batched_tokens=16424)
    p['kvmem'].update(pressure_policy='attention',cpu_read_cache_bytes=100000000000,
        graph_trace=False,prefill_physical_coalescing=True)
    return p


def full_fixture():
    source=source_fixture();reference=reference_profile()
    fixture=dict(source,rows=policy.select_rows(source),cases=130,samples_per_task=10,
        execution_profile=policy.PROFILE,reference_profile=reference)
    return policy.expected_profile(reference),fixture


def test_original_first_ten_are_selected_for_all_thirteen_categories_without_gold_filter():
    source=source_fixture();original=deepcopy(source)
    source['rows'].reverse()
    rows=policy.select_rows(source)
    assert len(rows)==130
    for task in policy.ruler.TASK_FAMILIES:
        assert {row['index'] for row in rows if row['task']==task}==set(range(10))
    assert {row['id']:row for row in rows}=={row['id']:row for row in original['rows'] if row['index']<10}
    rows[0]['outputs'].append('local mutation')
    assert all('local mutation' not in row['outputs'] for row in source['rows'])
    source['rows'][0]['index']=99
    with pytest.raises(ValueError,match='ordinals'):policy.select_rows(source)


def test_cpu_cache_is_inside_original_reserve_and_eight_full_histories_still_fit():
    profile,fixture=full_fixture();audit=policy.validate_execution(profile,fixture)
    assert audit['client_history_credits_bytes']==82820130816
    assert audit['other_fixed_reserve_bytes']==4*1024**3
    assert audit['eight_largest_charged_histories_bytes']<=audit['client_history_credits_bytes']
    assert audit['client_history_credits_bytes']+audit['shared_fps_cache_bytes']<=100000000000
    assert profile['kvmem']['graph_trace'] is True and profile['kvmem']['graph_query_trace_max_bytes']==1024**3
    assert profile['kvmem']['query_fps_cpu_backend']=='native' and profile['kvmem']['query_fps_cpu_workers']==16
    assert fixture['reference_profile']['kvmem']['graph_trace'] is False


@pytest.mark.parametrize('field,value',[('max_tokens',4096),('thinking_token_budget',2048),('temperature',0.)])
def test_sampling_cannot_silently_change(field,value):
    profile,fixture=full_fixture();fixture['sampling'][field]=value
    with pytest.raises(ValueError,match='sampling'):policy.validate_execution(profile,fixture)


def cpu_stats(*,workers=8):
    counters=dict(fps_cpu_groups=2,fps_cpu_native_groups=2,fps_cpu_reference_groups=0,
        fps_cpu_ambiguity_groups=0,fps_cpu_pool_busy_groups=0,fps_cpu_unsupported_fallback_groups=0)
    return dict(construction_device='cpu',fps_gpu_groups=0,fps_cpu_backend='native',fps_cpu_workers=workers,
        **counters,prebuild=dict(enabled=True,device='cpu',cpu_backend='native',cpu_workers=workers,**counters,
        fps_gpu_groups=0,queued_records=8,published_records=8,failed_records=0,last_error=None),
        cache_prebuilt_hit_blocks=8,cache_rebuilt_blocks=2)


@pytest.mark.parametrize('damage',['missing_device','query_gpu','background_gpu','missing_background_counter','failure'])
def test_cpu_only_build_is_proven_for_query_misses_and_background(damage):
    stats=cpu_stats();assert policy.audit_cpu_fps(stats)['cache_rebuilt_blocks']==2
    if damage=='missing_device':stats.pop('construction_device')
    elif damage=='query_gpu':stats['fps_gpu_groups']=1
    elif damage=='background_gpu':stats['prebuild']['fps_gpu_groups']=1
    elif damage=='missing_background_counter':stats['prebuild'].pop('fps_gpu_groups')
    else:stats['prebuild']['last_error']='real failure'
    with pytest.raises(ValueError):policy.audit_cpu_fps(stats)


def test_native_cpu_ambiguity_is_recorded_and_cache_hit_zero_query_groups_is_valid():
    stats=cpu_stats()
    for name in tuple(stats):
        if name.endswith('_groups'):stats[name]=0
    stats['prebuild'].update(fps_cpu_ambiguity_groups=1,fps_cpu_reference_groups=1)
    audit=policy.audit_cpu_fps(stats)
    assert audit['query_native']['fps_cpu_groups']==0
    assert audit['background_native']['fps_cpu_ambiguity_groups']==1


def test_selected_sixteen_worker_campaign_requires_matching_actual_query_and_background_pools():
    profile,fixture=full_fixture();policy.validate_execution(profile,fixture)
    workers=profile['kvmem']['query_fps_cpu_workers']
    assert workers==16
    assert policy.audit_cpu_fps(cpu_stats(workers=16),workers=workers)['cpu_workers']==16
    with pytest.raises(ValueError,match='worker count'):
        policy.audit_cpu_fps(cpu_stats(workers=8),workers=workers)


@pytest.mark.parametrize('scope',['query','background'])
@pytest.mark.parametrize('fault',['backend','workers','missing_counter','bad_total','unsupported','busy'])
def test_native_cpu_pool_and_fallback_audit_rejects_silent_reference_execution(scope,fault):
    stats=cpu_stats();target=stats['prebuild'] if scope=='background' else stats
    prefix='cpu_' if scope=='background' else 'fps_cpu_'
    if fault=='backend':target[prefix+'backend']='torch'
    elif fault=='workers':target[prefix+'workers']=16
    elif fault=='missing_counter':target.pop('fps_cpu_reference_groups')
    elif fault=='bad_total':target['fps_cpu_groups']+=1
    else:
        target['fps_cpu_groups']+=1;target['fps_cpu_reference_groups']+=1
        target['fps_cpu_'+('unsupported_fallback_groups' if fault=='unsupported' else 'pool_busy_groups')]=1
    with pytest.raises(ValueError):policy.audit_cpu_fps(stats)


def test_qualification_is_the_original_first_admission_wave_without_smaller_inputs_or_sampling():
    source=source_fixture();reference=reference_profile();rows=policy.qualification_rows(source)
    fixture=dict(source,rows=rows,cases=8,samples_per_task=None,execution_profile=policy.QUALIFICATION_PROFILE,
        reference_profile=reference)
    profile=policy.expected_profile(reference,qualification=True)
    policy.validate_execution(profile,fixture)
    assert [(row['task'],row['index']) for row in rows]==[(task,0) for task in list(policy.ruler.TASK_FAMILIES)[:8]]
    full,_=full_fixture();full['port']=profile['port']
    assert full==profile and fixture['sampling']==source['sampling']
    fixture['rows'][-1]=dict(fixture['rows'][-1],index=1)
    with pytest.raises(ValueError,match='first eight'):policy.validate_execution(profile,fixture)


@pytest.mark.parametrize('task', ['qa_1','vt','cwe','niah_multikey_2'])
def test_generic_query_audit_replays_native_atomic_picker_without_target_spans(task):
    query,graph,inputs,profile=audit_fixture()
    profile['kvmem'].update(query_fps_cpu_backend='native',query_fps_cpu_workers=16)
    query['query_key_fps_scoring'].update(cpu_stats(workers=16))
    query.update(selection='query',request_id='cmpl-a-0',query_span=[180,190],mean_k_backend='qw3_cuda',
        query_history_enabled=False,action_history_pages=0,prefill_attention_samples=5,guided_query_tokens=0,
        selected_tokens=sum(n for _,n in query['selected_history_ranges']),score_domain_version=policy.SCORE_DOMAIN,
        scoring_version='v2',pressure_policy='attention')
    inputs['request_id']=graph['request_id']='cmpl-a-0';graph['phase']='query'
    pressure=dict(query,selection='pressure',seed_source='prefill',score_mask_count=6,transport_mask_count=0)
    pg=dict(request_id='cmpl-a-0',phase='pressure',seed_source='prefill',selected_tokens=pressure['selected_tokens'],
        score_mask_count=6,transport_mask_count=0,propagation_applied=False)
    events={'bounded_refresh.jsonl':[pressure,query],'block_graph.jsonl':[pg,graph],'block_graph_inputs.jsonl':[inputs]}
    evidence=policy.audit_case(Path('/unused'),dict(response_id='cmpl-a'),profile,[180,190],
        event_reader=lambda path:events[path.name])
    assert evidence['selection_audit']['native_group_picker_replay_equal']
    assert evidence['cpu_fps_audit']['construction_device']=='cpu'
    pg['propagation_applied']=True
    with pytest.raises(ValueError,match='cumulative attention'):
        policy.audit_case(Path('/unused'),dict(response_id='cmpl-a'),profile,[180,190],event_reader=lambda path:events[path.name])


def encoded(rid,value=1):return (json.dumps(dict(request_id=rid,phase='query',value=value),ensure_ascii=False)+'\n').encode()


def test_incremental_query_index_handles_interleaving_partial_lines_and_no_log_rotation(tmp_path):
    path=tmp_path/'shared.jsonl';one,two=encoded('cmpl-a-0','中文'),encoded('cmpl-b-0')
    path.write_bytes(one+two[:20]);index=policy.QueryTraceIndex(path,tmp_path/'index',max_records=2,max_bytes=2048)
    selected=index.take(dict(response_id='cmpl-a'),tmp_path/'a.jsonl')
    assert selected['value']=='中文' and path.read_bytes()==one+two[:20]
    assert index.offset==len(one) and len(index.files)==1
    with path.open('ab') as out:out.write(two[20:])
    assert index.take(dict(response_id='cmpl-b'),tmp_path/'b.jsonl')['request_id']=='cmpl-b-0'
    assert index.audit()['records']==2
    assert path.read_bytes()==one+two and (tmp_path/'a.jsonl').read_bytes()==one
    with pytest.raises(ValueError,match='already consumed'):index.take(dict(response_id='cmpl-a'),tmp_path/'again')


@pytest.mark.parametrize('fault',['duplicate','too_many','record_bytes','total_bytes','rotate'])
def test_trace_limits_reject_errors_without_overwriting_completed_request_files(tmp_path,fault):
    path=tmp_path/'shared.jsonl';path.write_bytes(encoded('cmpl-a-0'))
    index=policy.QueryTraceIndex(path,tmp_path/'index',max_records=1,max_bytes=1024,max_record_bytes=200)
    destination=tmp_path/'a.jsonl';index.take(dict(response_id='cmpl-a'),destination);before=destination.read_bytes()
    if fault=='rotate':path.unlink();path.write_bytes(encoded('other'))
    else:
        data=encoded('cmpl-a-0') if fault=='duplicate' else encoded('b', 'x'*220) if fault=='record_bytes' else encoded('b','x'*1500) if fault=='total_bytes' else encoded('b')
        with path.open('ab') as out:out.write(data)
    with pytest.raises(ValueError):index.audit()
    assert destination.read_bytes()==before
