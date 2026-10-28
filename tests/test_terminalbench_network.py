from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.terminalbench_network import (
    DIRECT_CONTAINER_ENV, assert_direct_docker, reclaim_trial_networks, validate_owner,
    filesystem_diff_sha,
    copy_download_cache, cache_compose_override,
)


def objects():
    network = dict(Id='n'*64, Driver='bridge', Containers={}, Labels={
        'com.docker.compose.project':'example__env', 'com.docker.compose.network':'default'})
    container = dict(Id='c'*64, Image='sha256:base', Config=dict(Labels={
        'com.docker.compose.project':'example__env',
        'com.docker.compose.project.working_dir':'/tmp/owned/environment'}),
        State=dict(Status='exited', Running=False, Paused=False, Restarting=False))
    return network, container


@pytest.mark.parametrize('change', ['endpoint','project','container_project','path','active','paused','missing'])
def test_reclamation_rejects_unproven_or_active_ownership(change):
    network, container = objects()
    containers = [container]
    if change == 'endpoint': network['Containers']={'foreign':{}}
    if change == 'project': network['Labels']['com.docker.compose.project']='other'
    if change == 'container_project': container['Config']['Labels']['com.docker.compose.project']='other'
    if change == 'path': container['Config']['Labels']['com.docker.compose.project.working_dir']='/other'
    if change == 'active': container['State']['Running']=True
    if change == 'paused': container['State']['Paused']=True
    if change == 'missing': containers=[]
    with pytest.raises(ValueError): validate_owner(network, containers, 'example__env','/tmp/owned/environment')


@pytest.mark.parametrize('race', [False, True])
def test_reclamation_checks_again_and_never_removes_container_or_volume(tmp_path, race):
    network, container = objects()
    calls=[]; inspections=0
    def call(*args):
        nonlocal inspections
        calls.append(args)
        if args[:2] == ('network','ls'): return network['Id']
        if args[0] == 'ps': return container['Id']
        if args[:2] == ('network','inspect'):
            inspections+=1
            value=deepcopy(network)
            if race and inspections>1: value['Containers']={'new-active':{}}
            return json.dumps([value])
        if args[0] == 'inspect': return json.dumps([container])
        if args[0] == 'diff': return 'A /tmp/important-answer\n'
        if args[:2] == ('network','rm'): return network['Id']
        raise AssertionError(args)
    if race:
        with pytest.raises(ValueError,match='active endpoints'):
            reclaim_trial_networks('example__env','/tmp/owned/environment',tmp_path,call)
        assert not any(c[:2]==('network','rm') for c in calls)
    else:
        rows=reclaim_trial_networks('example__env','/tmp/owned/environment',tmp_path,call)
        assert rows[0]['retained_filesystem_diffs_unchanged']
        assert list(tmp_path.glob('*.removed.json'))
        assert inspections==2
    assert not any(c[0] in ('rm','rmi','volume','stop','kill','prune') for c in calls)


def test_empty_project_is_noop(tmp_path):
    assert reclaim_trial_networks('example__env','/tmp/owned/environment',tmp_path,lambda *a:'')==[]


def test_docker_diff_order_is_not_a_filesystem_change():
    assert filesystem_diff_sha('A /a\nC /b\n') == filesystem_diff_sha('C /b\nA /a\n')
    assert filesystem_diff_sha('A /a\n') != filesystem_diff_sha('D /a\n')


@pytest.mark.parametrize('source',['daemon','client','none'])
def test_direct_policy_checks_hidden_proxy_sources_without_changing_global_config(tmp_path,source):
    config=tmp_path/'config.json'
    data={'proxies':{'default':{'httpProxy':'secret'}}} if source=='client' else {}
    config.write_text(json.dumps(data)); before=config.read_bytes()
    call=lambda *a: json.dumps({'HTTPSProxy':'secret'} if source=='daemon' else {})
    if source=='none':
        assert assert_direct_docker({'DOCKER_CONFIG':str(tmp_path)},call)['client_proxy'] is False
    else:
        with pytest.raises(RuntimeError,match='proxy|proxies'):
            assert_direct_docker({'DOCKER_CONFIG':str(tmp_path)},call)
    assert config.read_bytes()==before
    assert all(DIRECT_CONTAINER_ENV[k]=='' for k in ['HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy'])


@pytest.mark.parametrize('damage',[None,'bytes','digest','path'])
def test_cache_only_copies_verified_public_archives(tmp_path,damage):
    import hashlib
    source=tmp_path/'source';source.mkdir()
    name='astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz'
    payload=b'public dependency fixture';digest=hashlib.sha256(payload).hexdigest()
    archive=source/name;archive.parent.mkdir(parents=True);archive.write_bytes(payload)
    proof=dict(sha256=digest,official_digest='sha256:'+digest,bytes=len(payload))
    if damage=='bytes':archive.write_bytes(b'tampered')
    if damage=='digest':proof['official_digest']='sha256:wrong'
    if damage=='path':name='../outside.tar.gz'
    (source/'manifest.json').write_text(json.dumps(dict(passed=True,proxy_disabled=True,files={name:proof})))
    (source/'not-a-dependency.txt').write_text('must never reach a task container')
    dest=tmp_path/'dest'
    if damage:
        with pytest.raises(ValueError):copy_download_cache(source,dest)
        assert not dest.exists()
    else:
        copy_download_cache(source,dest)
        assert (dest/name).read_bytes()==payload
        assert not (dest/'not-a-dependency.txt').exists()
        overlay=cache_compose_override(dest)
        assert overlay['services']['main']['volumes'][0]['read_only'] is True
