import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pytest

from scripts.recover_agent_handoff_startup import prepare, validate_startup_failure
from scripts import run_agent_handoff_queue as queue


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    queue.save(path,value)


def fixture(root):
    for arm in ('baseline','candidate'):
        write(root/arm/'status.json',dict(status='prepared'))
        write(root/arm/'profile.json',dict(K=36864,G=16384,temperature=.6,seed=1000))
        write(root/arm/'task-plan.json',dict(cases=[dict(id='fps',session_id='unused-session',
            workspace=str(root/arm/'workspaces/fps'),prompts=['Original prompt'])],sampling=dict(seed=1000)))
    write(root/'qualification-profile.json',dict(verify_session_restore=True))
    write(root/'qualification-fixture.json',[dict(tokens=[1,2,3],suffix=[4])])
    (root/'protocol.so').write_bytes(b'unchanged protocol')
    write(root/'manifest.json',dict(protocol=str(root/'protocol.so'),files={str(p.relative_to(root)):queue.sha(p)
        for p in root.rglob('*') if p.is_file() and p.name!='status.json'}))
    write(root/'status.json',dict(status='failed',error="TimeoutError('Owned service readiness timed out')"))
    write(root/'queue-process.json',dict(pid=1,created=1))
    write(root/'qualification/cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[],owned=[]))
    return root


def test_recovery_preserves_fixtures_sampling_old_evidence_and_changes_only_output_paths(tmp_path):
    source=fixture(tmp_path/'old');destination=tmp_path/'new'
    original={str(p.relative_to(source)):queue.sha(p) for p in source.rglob('*') if p.is_file()}
    repository=Path(__file__).resolve().parents[1]
    result=prepare(source,destination,repository,process_probe=lambda _:None)
    assert queue.verify_frozen(destination)['model_startup_timeout_seconds']==3600
    assert queue.sha(destination/'qualification-fixture.json')==original['qualification-fixture.json']
    old=queue.read(source/'candidate/task-plan.json');new=queue.read(destination/'candidate/task-plan.json')
    assert old['sampling']==new['sampling']
    assert old['cases'][0]['session_id']==new['cases'][0]['session_id']
    assert old['cases'][0]['prompts']==new['cases'][0]['prompts']
    assert new['cases'][0]['workspace']==str(destination/'candidate/workspaces/fps')
    assert all(queue.sha(source/p)==value for p,value in original.items())
    assert result['audit_sha256']==queue.sha(destination/'recovery-audit.json')
    with pytest.raises(ValueError,match='already exists'):
        prepare(source,destination,repository,process_probe=lambda _:None)


@pytest.mark.parametrize('damage',['answer','audit','public_work','attempted_arm','cleanup','live_queue','unknown_capture','source_change'])
def test_rejects_any_partially_run_or_unverified_source(tmp_path,damage):
    source=fixture(tmp_path/'old')
    probe=lambda _:None
    if damage=='answer':write(source/'qualification/0-0.json',dict(response='preserve'))
    if damage=='audit':write(source/'qualification/execution-audit.json',dict(status='passed'))
    if damage=='public_work':write(source/'qualification/server/capture/public_work.json',dict(completed=dict(public_batches=1)))
    if damage=='unknown_capture':write(source/'qualification/server/capture/unknown.json',{})
    if damage=='attempted_arm':write(source/'baseline/started.json',{})
    if damage=='cleanup':write(source/'qualification/cleanup.json',dict(remaining_processes=[1],remaining_owned_gpu_pids=[]))
    if damage=='live_queue':probe=lambda _:object()
    if damage=='source_change':write(source/'baseline/profile.json',dict(K=1))
    with pytest.raises(ValueError):validate_startup_failure(source,probe)


def test_readiness_accepts_slow_cold_start_without_requesting_generation(tmp_path,monkeypatch):
    clock=[0.];attempts=[]
    monkeypatch.setattr(queue.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Opener:
        def open(self,url,timeout):
            attempts.append(url)
            if clock[0]<902:raise OSError('not ready')
            return Response()
    monkeypatch.setattr(urllib.request,'build_opener',lambda *args:Opener())
    owned=SimpleNamespace(rows={},track=lambda:None)
    queue.ready_http('http://127.0.0.1:18877/health',SimpleNamespace(poll=lambda:None),owned,
        timeout=3600,progress_path=tmp_path/'startup.json')
    assert queue.read(tmp_path/'startup.json')['status']=='ready'
    assert queue.read(tmp_path/'startup.json')['elapsed_seconds']==902
    assert set(attempts)=={'http://127.0.0.1:18877/health'}
    clock[0]=0
    with pytest.raises(TimeoutError):
        queue.ready_http('http://127.0.0.1:18877/health',SimpleNamespace(poll=lambda:None),owned,timeout=900)


def test_readiness_exits_immediately_when_owned_process_dies(tmp_path):
    with pytest.raises(RuntimeError,match='exited'):
        queue.ready_http('unused',SimpleNamespace(poll=lambda:1),SimpleNamespace(track=lambda:None),
            timeout=3600,progress_path=tmp_path/'startup.json')
