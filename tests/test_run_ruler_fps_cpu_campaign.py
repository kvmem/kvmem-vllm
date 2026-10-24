"""Controller and public diagnostic-limit plumbing; never starts a model."""
import json

import pytest

import run_ruler_fps_cpu_campaign as campaign
from test_graph_options import model_profile,runtime,worker_payload
from test_query_fps_prebuild_options import fps_settings,elastic_profile
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch
from vllm_kvmem.elastic_serve import build_launch as build_elastic


def test_query_trace_limit_defaults_bounded_and_roundtrips_both_launchers(tmp_path):
    assert RuntimeOptions().graph_query_trace_max_bytes==16*1024**2
    local=model_profile(tmp_path);local['kvmem']=fps_settings(graph_query_trace_max_bytes=1024**3)
    worker=worker_payload(build_launch(local,tmp_path,tmp_path/'native'))
    assert runtime(worker).graph_query_trace_max_bytes==1024**3
    profile=elastic_profile(tmp_path,local);profile['graph_query_trace_max_bytes']=1024**3
    resolved,_=build_elastic(profile,tmp_path,tmp_path/'elastic')
    assert resolved['mean_k_backend_profile']['kvmem']['graph_query_trace_max_bytes']==1024**3
    replay,_=build_elastic(resolved,tmp_path,tmp_path/'replay')
    assert replay['mean_k_backend_profile']==resolved['mean_k_backend_profile']


@pytest.mark.parametrize('limit',[0,-1,True,'1GiB',None,1024**3+1])
def test_invalid_query_trace_limit_fails_before_launch(limit):
    with pytest.raises(ValueError,match='graph_query_trace_max_bytes'):
        runtime(fps_settings(graph_query_trace_max_bytes=limit))


def test_query_log_cap_applies_even_to_first_record_and_never_changes_scores(tmp_path):
    from test_graph_v2_ablation import export_fixture,export
    authority,rows,snapshot=export_fixture(tmp_path,nodes=4)
    authority.query_score_mode='key_fps_4';authority.graph_options.graph_query_trace_max_bytes=1
    with pytest.raises(BufferError,match='log byte capacity'):export(authority,rows,snapshot)
    assert not (tmp_path/'block_graph_inputs.jsonl').exists()
    authority.graph_options.graph_query_trace_max_bytes=65536
    export(authority,rows,snapshot)
    path=tmp_path/'block_graph_inputs.jsonl';original=path.read_bytes();record=json.loads(original)
    assert record['raw_scores']==[.25]*4 and record['starts']==[0,32,64,96]
    authority.graph_options.graph_query_trace_max_bytes=len(original)
    with pytest.raises(BufferError,match='log byte capacity'):export(authority,rows,snapshot)
    assert path.read_bytes()==original


def test_existing_campaign_root_is_refused_without_touching_successes(tmp_path):
    completed=tmp_path/'response.json';completed.write_text('immutable')
    with pytest.raises(ValueError,match='fresh campaign'):
        campaign.prepare(tmp_path,tmp_path/'nonexistent')
    assert completed.read_text()=='immutable'


def test_unfrozen_controller_cannot_launch_or_resume(tmp_path):
    with pytest.raises(ValueError,match='frozen controller'):
        campaign.run(tmp_path)
    assert not (tmp_path/'attempt.json').exists()


def test_source_fixture_changed_after_preparation_is_detected(tmp_path):
    path=tmp_path/'fixed.json';path.write_text('original')
    proof={'fixed.json':campaign.sha(path)}
    campaign.verify_files(tmp_path,proof)
    path.write_text('changed')
    with pytest.raises(ValueError,match='Frozen file changed'):campaign.verify_files(tmp_path,proof)
