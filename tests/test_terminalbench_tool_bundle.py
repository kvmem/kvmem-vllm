import asyncio
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from scripts.terminalbench_tool_bundle import install_bundle, validate_bundle


def bundle(tmp_path, member='tool_1_amd64.deb', link=False, duplicate=False):
    data=b'public package bytes';digest=hashlib.sha256(data).hexdigest()
    archive=tmp_path/'packages.tar'
    with tarfile.open(archive,'w') as tar:
        m=tarfile.TarInfo(member);m.size=len(data)
        if link:m.type=tarfile.SYMTYPE;m.linkname='/etc/passwd';m.size=0
        tar.addfile(m,io.BytesIO(data))
        if duplicate:tar.addfile(m,io.BytesIO(data))
    proof=dict(task='task',image='original:tag',offline_install_passed=True,proxy_disabled=True,
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               files={member:dict(sha256=digest,apt_sha256=digest)})
    (tmp_path/'manifest.json').write_text(json.dumps(proof))
    return proof


@pytest.mark.parametrize('change', ['task','image','offline_install_passed','proxy_disabled','apt_sha256','archive'])
def test_reject_unqualified_changed_or_unbound_bundle(tmp_path,change):
    p=bundle(tmp_path)
    if change=='archive':(tmp_path/'packages.tar').write_bytes(b'changed')
    elif change=='apt_sha256':p['files']['tool_1_amd64.deb']['apt_sha256']='0'*64
    elif change in ('task','image'):p[change]='different'
    else:p[change]=False
    (tmp_path/'manifest.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):validate_bundle(tmp_path,'task','original:tag')


@pytest.mark.parametrize('kwargs',[dict(member='../bad.deb'),dict(link=True),dict(duplicate=True)])
def test_reject_unsafe_archive(tmp_path,kwargs):
    bundle(tmp_path,**kwargs)
    with pytest.raises(ValueError):validate_bundle(tmp_path,'task','original:tag')


@pytest.mark.parametrize('failure',[None,RuntimeError('Command timed out after 120 seconds'),RuntimeError('container disappeared'),asyncio.CancelledError()])
def test_offline_install_is_bounded_audited_and_propagates_failures(tmp_path,failure):
    proof=bundle(tmp_path);events=[];calls=[]
    async def upload(**kwargs):calls.append(('upload',kwargs))
    async def execute(**kwargs):
        calls.append(('exec',kwargs))
        if failure:raise failure
        return SimpleNamespace(return_code=0,stdout='tools installed',stderr='')
    env=SimpleNamespace(environment_dir=Path('/tasks/task/environment'),
                        task_env_config=SimpleNamespace(docker_image='original:tag'),
                        upload_file=upload,exec=execute)
    coro=install_bundle(SimpleNamespace(environment=env),tmp_path,lambda **e:events.append(e))
    if failure:
        with pytest.raises(type(failure)):asyncio.run(coro)
    else:
        assert asyncio.run(coro)==proof
        assert events[-1]['event']=='terminal_bundle_installed'
    assert calls[1][1]['timeout_sec']==120
    assert 'apt-get -y --no-download -o Dir::Cache::archives=' in calls[1][1]['command']
    assert 'apt-get update' not in calls[1][1]['command']
    assert proof['archive_sha256'] in calls[1][1]['command']
    if failure:assert not any(e['event']=='terminal_bundle_installed' for e in events)
