"""A fixed eight-way quality-independent gate reuses the complete case audit."""
from copy import deepcopy

import pytest

import qualify_query_fps_native_cpu_live as qualification
import run_ruler_fps_cpu_campaign as campaign
import ruler_fps_cpu_protocol as policy
from test_ruler_fps_cpu_protocol import source_fixture,cpu_stats


def test_qualification_prepares_only_explicit_subset_and_refuses_existing_root(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(campaign,'prepare',lambda *args,**kwargs:calls.append((args,kwargs)))
    qualification.prepare(tmp_path/'new',tmp_path/'source')
    assert calls==[((tmp_path/'new',tmp_path/'source'),{'qualification':True})]


def test_qualification_score_never_claims_full_ruler_or_gates_low_scores():
    rows=policy.qualification_rows(source_fixture())
    fixture=dict(rows=rows,cases=8,execution_profile=policy.QUALIFICATION_PROFILE)
    results=[dict(id=row['id'],task=row['task'],official_case_score=0) for row in rows]
    score=campaign.score_results(fixture,results,None)
    assert score['macro_score']==0 and score['qualification_only'] and score['full_ruler_score'] is False
    with pytest.raises(ValueError,match='immutable full requested cohort'):
        campaign.score_results(fixture,results[:-1],None)


def metrics():
    audit=policy.audit_cpu_fps(cpu_stats())
    return [dict(client_started_at=100+i*.01,request_seconds=120,ttft_seconds=60,client_to_query_seconds=45,
        query_refresh_seconds=10,query_to_first_token_seconds=5,decode_stream_seconds=60,
        cpu_fps_audit=deepcopy(audit)) for i in range(8)]


def completed():return dict(cases=8,queue_peak_inflight=8,input_tokens=8*262144,completion_tokens=800)


def test_live_qualification_requires_actual_native_execution_cache_hits_and_eight_admitted_requests():
    summary=campaign.performance_summary(metrics(),completed(),qualification=True)
    assert summary['prebuilt_hit_blocks']==64 and summary['native_cpu_groups']==32
    assert summary['no_accuracy_gate'] and summary['concurrent_admission_peak']==8
    assert summary['request_wall_seconds']==pytest.approx(120.07)
    assert summary['phases_seconds']['query_refresh_seconds']['median']==10
    for damage in ('peak','cache','native'):
        rows,done=metrics(),completed()
        if damage=='peak':done['queue_peak_inflight']=7
        elif damage=='cache':
            for row in rows:row['cpu_fps_audit']['cache_prebuilt_hit_blocks']=0
        else:
            for row in rows:
                for scope in ('query_native','background_native'):
                    row['cpu_fps_audit'][scope]['fps_cpu_native_groups']=0
        with pytest.raises(ValueError):campaign.performance_summary(rows,done,qualification=True)
