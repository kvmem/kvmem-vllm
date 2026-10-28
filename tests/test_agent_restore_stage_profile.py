import json

import pytest

from scripts import profile_agent_restore_stages as profile
from scripts import run_agent_handoff_queue as q
import monitor_agent_handoff as monitor


def traces(root):
    capture = root/'run/server/capture'
    capture.mkdir(parents=True)
    restores = [dict(request_id=f'r{i}') for i in range(56)]
    names = ['prepare_claim_window','model_state_prepare','background_submit','publish_restore']
    stages = [dict(request_id=f'r{i}',outcome='progress',timing=dict(stages=[dict(name=n) for n in names])) for i in range(56)]
    (capture/'async_restore.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in restores))
    path = capture/'restore_host_stages.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in stages))
    return path, stages


def test_profile_requires_complete_restore_stage_coverage(tmp_path):
    traces(tmp_path)
    result = profile.stage_audit(tmp_path)
    assert result['restores'] == 56
    assert result['performance_comparison_allowed'] is False


@pytest.mark.parametrize('change',['missing_request','missing_stage','error'])
def test_incomplete_stage_profile_is_not_accepted(tmp_path,change):
    path, rows = traces(tmp_path)
    if change == 'missing_request':rows.pop()
    if change == 'missing_stage':rows[0]['timing']['stages'].pop()
    if change == 'error':rows[0]['outcome']='error'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):profile.stage_audit(tmp_path)


def test_existing_monitor_tracks_diagnostic_engine_without_new_timer(tmp_path):
    def save(rel,value):
        path=tmp_path/rel;path.parent.mkdir(parents=True,exist_ok=True);q.save(path,value)
    save('manifest.json',dict(optimization_kind='graph_span_merge',optimization_arms=['before','after','profile']))
    save('status.json',dict(status='optimization_running',active='profile',at=1))
    save('optimization/profile/status.json',dict(status='testing'))
    save('optimization/profile/run/server-process.json',dict(pid=4,created=1))
    save('optimization/profile/run/server/capture/public_work.json',dict(pid=5,counter_id='p',
        completed=dict(public_batches=7,public_prefill_tokens=1000)))
    snap=monitor.inspect(tmp_path,now=2,probe=lambda r:dict(state='alive'),resource_probe=lambda _:dict(gpu=[dict(pid=5)]))
    assert not snap['issues']
    assert snap['optimization']['profile']['work']['public_prefill_tokens']==1000
    assert snap['gpu'][0]['matches_active_engine'] is True


def test_only_new_profile_arm_executes(tmp_path,monkeypatch):
    runtime=tmp_path/'optimization/profile/runtime';runtime.mkdir(parents=True)
    monkeypatch.setattr(profile,'ROOT',runtime)
    monkeypatch.setattr(profile,'validate',lambda _:dict())
    monkeypatch.setattr(q,'wait_dependency',lambda *_:None)
    monkeypatch.setattr(q,'gpu_pids',lambda:[])
    q.save(tmp_path/'status.json',dict(status='prepared'))
    q.save(tmp_path/'manifest.json',dict())
    calls=[]
    monkeypatch.setattr(profile.benchmark,'run_arm',lambda r,n,p:calls.append(n))
    monkeypatch.setattr(profile,'stage_audit',lambda _:dict(status='passed'))
    profile.run(tmp_path)
    assert calls==['profile']
    assert q.read(tmp_path/'status.json')['full_paired_test_passed'] is False


def test_monitor_requires_stage_audit_before_accepting_diagnostic_completion(tmp_path):
    def save(rel,value):
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);q.save(p,value)
    save('manifest.json',dict(optimization_kind='graph_span_merge',optimization_arms=['profile']))
    save('status.json',dict(status='needs_optimization_review'))
    p=tmp_path/'optimization/profile'
    for i in range(64):save(f'optimization/profile/answers/{i}.json',{})
    save('optimization/profile/status.json',dict(status='completed'))
    save('optimization/profile/execution-audit.json',dict(status='passed',responses=64,restores=56,exact_output_equal=True))
    save('optimization/profile/cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))
    save('optimization/profile/performance.json',{})
    save('optimization/profile/evidence-hashes.json',{})
    def check():return monitor.inspect(tmp_path,now=2,probe=lambda _:dict(state='missing_or_exited'),resource_probe=lambda _:dict(gpu=[]))
    assert not check()['optimization']['profile']['audit_and_cleanup_passed']
    assert any('component audit' in s for s in check()['issues'])
    save('optimization/profile/stage-audit.json',dict(status='passed',restores=56,timed_rows=168))
    assert check()['optimization']['profile']['audit_and_cleanup_passed']
