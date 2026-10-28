import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

from scripts.benchmark_deepswe_interference import validate_profiles,check_frozen_entrypoints,OVERLAYS,validate_preserved_control,CLIENT_SHA,qualified_proof_arms


def profiles():
    control=dict(max_model_len=10_000_000,kvmem=dict(select_budget=98304,gen_budget=16384,
        graph_alpha=0,native_checkpoint_capture=False,checkpoint_pinned_prewarm_count=0,
        prefill_batch_tokens=0,host_stage_timing=False,restore_stage_timing=False))
    candidate=copy.deepcopy(control)
    candidate['kvmem'].update(native_checkpoint_capture=True,checkpoint_pinned_prewarm_count=10,prefill_batch_tokens=512)
    return control,candidate


def test_only_qualified_optimization_knobs_may_differ():
    a,b=profiles();validate_profiles(a,b)
    assert a['kvmem']['native_checkpoint_capture'] is False
    assert b['kvmem']['native_checkpoint_capture'] is True


@pytest.mark.parametrize('field,value',[('select_budget',36864),('gen_budget',1024),('graph_alpha',.2),
                                      ('host_stage_timing',True),('restore_stage_timing',True)])
def test_cannot_hide_a_quality_change_or_diagnostic_overhead(field,value):
    a,b=profiles();b['kvmem'][field]=value
    with pytest.raises(ValueError):validate_profiles(a,b)


def test_logical_limit_and_verification_must_match():
    a,b=profiles();b['max_model_len']=300_000
    with pytest.raises(ValueError):validate_profiles(a,b)
    a,b=profiles();a['verify_session_restore_digest']=b['verify_session_restore_digest']=True
    with pytest.raises(ValueError,match='verification'):validate_profiles(a,b)


def test_frozen_controller_imports_without_workspace_fallback(tmp_path):
    source=Path(__file__).resolve().parents[1]
    scripts=tmp_path/'scripts';scripts.mkdir()
    for rel in [*OVERLAYS,'scripts/run_agent_handoff_queue.py']:
        shutil.copy2(source/rel,scripts/Path(rel).name)
    assert check_frozen_entrypoints(tmp_path,sys.executable)['status']=='passed'
    (scripts/'queue_agent_interference.py').unlink()
    with pytest.raises(RuntimeError,match='Frozen entrypoint import failed'):
        check_frozen_entrypoints(tmp_path,sys.executable)


@pytest.mark.parametrize('corruption',['none','profile','tasks','duration','client','cleanup','audit','status'])
def test_reuse_requires_same_completed_window(tmp_path,corruption):
    profile=profiles()[0];expected=copy.deepcopy(profile);expected['port']=18878
    manifest=dict(tasks=[{'task_id':'fixed'}],duration_seconds=900,claude_sha256=CLIENT_SHA)
    files={'profile.json':expected,'manifest.json':manifest,'run.json':dict(status='window_complete',cleanup_errors=[]),
           'cleanup.json':dict(remaining_processes=[],remaining_owned_gpu_pids=[]),'execution-audit.json':dict(status='passed')}
    if corruption=='profile':expected['max_model_len']=512000
    if corruption=='tasks':manifest['tasks']=[]
    if corruption=='duration':manifest['duration_seconds']=300
    if corruption=='client':manifest['claude_sha256']='changed'
    if corruption=='cleanup':files['cleanup.json']['remaining_processes']=[123]
    if corruption=='audit':files['execution-audit.json']['status']='failed'
    if corruption=='status':files['run.json']['status']='running'
    for name,value in files.items():(tmp_path/name).write_text(json.dumps(value))
    args=(tmp_path,dict(tasks=[{'task_id':'fixed'}]),profile,900)
    if corruption=='none':validate_preserved_control(*args)
    else:
        with pytest.raises(ValueError):validate_preserved_control(*args)


@pytest.mark.parametrize('change',['none','different_profile','missing_digest'])
def test_correctness_only_gate_requires_exact_gpu_qualified_profile(tmp_path,monkeypatch,change):
    import scripts.benchmark_deepswe_interference as module
    profile=profiles()[1];profile.update(verify_session_restore=False,verify_session_restore_digest=False)
    qualified=copy.deepcopy(profile);qualified.update(verify_session_restore=True,verify_session_restore_digest=True)
    if change=='different_profile':qualified['kvmem']['select_budget']=32768
    if change=='missing_digest':qualified['verify_session_restore_digest']=False
    for name,value in [('qualification',qualified),('per_request256',profile)]:
        (tmp_path/name).mkdir();(tmp_path/name/'profile.json').write_text(json.dumps(value))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(correctness_only_candidates=['per_request256'])))
    seen=[];monkeypatch.setattr(module,'completed',lambda root,arm:seen.append(arm))
    if change=='none':
        assert qualified_proof_arms(tmp_path,'per_request256')==['qualification','control']
        assert seen==['qualification','control']
    else:
        with pytest.raises(ValueError):qualified_proof_arms(tmp_path,'per_request256')
