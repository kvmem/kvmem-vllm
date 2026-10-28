import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import deepswe_window_snapshot as s


def fixture(tmp_path, monkeypatch, *, foreign=False, commit_failure=False):
    cell=tmp_path/'cases/task';cell.mkdir(parents=True)
    (cell/'agent.container.cid').write_text('a'*64)
    owner=hashlib.sha256(str(cell).encode()).hexdigest()[:16]
    commands=[]
    def run(args, **kwargs):
        commands.append(args)
        if args[1]=='inspect':
            value=dict(Config=dict(Labels={'com.qw3.deepswe.run-owner':'foreign' if foreign else owner,
                'com.qw3.deepswe.role':'agent'}),State=dict(Running=True,Paused=False))
            return SimpleNamespace(returncode=0,stdout=json.dumps([value]))
        return SimpleNamespace(returncode=0,stdout='')
    def output(args,**kwargs):
        commands.append(args)
        if commit_failure:raise RuntimeError('Disk write failed')
        return ('Flag --pause has been deprecated\n' if args[1]=='commit' else '')+'sha256:'+'a'*64+'\n'
    monkeypatch.setattr(s.subprocess,'run',run)
    monkeypatch.setattr(s.subprocess,'check_output',output)
    return commands


def test_owned_workspace_committed_before_cleanup(tmp_path,monkeypatch):
    commands=fixture(tmp_path,monkeypatch)
    result=s.snapshot(tmp_path,[])
    assert result['status']=='preserved'
    assert result['containers'][0]['image']=='sha256:'+'a'*64
    assert [c[1] for c in commands]==['inspect','pause','commit','image']
    assert json.loads((tmp_path/'window-workspaces.json').read_text())==result
    with pytest.raises(ValueError,match='already attempted'):s.snapshot(tmp_path,[])


def test_foreign_owner_never_paused_or_committed(tmp_path,monkeypatch):
    commands=fixture(tmp_path,monkeypatch,foreign=True)
    with pytest.raises(ValueError,match='Foreign'):s.snapshot(tmp_path,[])
    assert [c[1] for c in commands]==['inspect']
    record=json.loads((tmp_path/'window-workspaces.json').read_text())
    with pytest.raises(ValueError,match='incomplete'):s.release_supervisors(record)


def test_failed_commit_preserves_paused_container_and_failure_audit(tmp_path,monkeypatch):
    commands=fixture(tmp_path,monkeypatch,commit_failure=True)
    with pytest.raises(RuntimeError,match='Disk'):s.snapshot(tmp_path,[])
    record=json.loads((tmp_path/'window-workspaces.json').read_text())
    assert record['status']=='failed_preserved_in_place'
    assert record['containers'][0]['paused_or_stopped']
    assert not any(c[1] in ('rm','unpause') for c in commands)


def test_release_checks_creation_time_and_argv(monkeypatch):
    sent=[]
    p=SimpleNamespace(create_time=lambda:1.0,cmdline=lambda:['original'],send_signal=sent.append)
    monkeypatch.setattr(s.psutil,'Process',lambda pid:p)
    r=dict(status='preserved',supervisors=[dict(pid=42,created=2.0,argv=['original'])])
    with pytest.raises(ValueError,match='identity'):s.release_supervisors(r)
    assert not sent
    r['supervisors'][0]['created']=1.0
    s.release_supervisors(r)
    assert sent==[s.signal.SIGCONT]
