"""Run-scoped direct downloads and conservative reclamation of idle networks."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


PROXY_KEYS = tuple(prefix+suffix for suffix in ('proxy', 'PROXY')
                   for prefix in (('http_', 'https_', 'all_', 'ftp_', 'socks_', 'no_')
                                  if suffix == 'proxy' else
                                  ('HTTP_', 'HTTPS_', 'ALL_', 'FTP_', 'SOCKS_', 'NO_')))
DIRECT_CONTAINER_ENV = {key: '' for key in PROXY_KEYS}
CACHE_CONTAINER_PATH = '/opt/kvmem-download-cache'
CACHE_ENV = {
    'UV_INSTALLER_GITHUB_BASE_URL': 'file://'+CACHE_CONTAINER_PATH,
    'UV_PYTHON_INSTALL_MIRROR': 'file://'+CACHE_CONTAINER_PATH+'/astral-sh/python-build-standalone/releases/download',
}


def copy_download_cache(source, destination):
    """Copy only digest-bound public archives; never expose a host package cache."""
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/'manifest.json').read_text())
    if not manifest.get('passed') or not manifest.get('proxy_disabled') or not manifest.get('files'):
        raise ValueError('Missing direct-download cache proof')
    for name, proof in manifest['files'].items():
        p=Path(name)
        if (p.is_absolute() or '..' in p.parts or not name.startswith('astral-sh/') or
                not name.endswith('.tar.gz') or (source/p).is_symlink() or
                not (source/p).resolve().is_relative_to(source.resolve())):
            raise ValueError('Unsafe dependency archive path')
        digest=hashlib.sha256((source/p).read_bytes()).hexdigest()
        if digest != proof['sha256'] or 'sha256:'+digest != proof['official_digest']:
            raise ValueError('Dependency archive digest mismatch')
        if (source/p).stat().st_size != proof['bytes']:
            raise ValueError('Dependency archive size mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest['files']:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    shutil.copyfile(source/'manifest.json',destination/'manifest.json')
    return manifest


def cache_compose_override(cache_path):
    return {'services': {'main': {'volumes': [dict(type='bind',
        source=str(Path(cache_path).resolve()), target=CACHE_CONTAINER_PATH, read_only=True)]}}}


def direct_environment(env=None):
    return {key: value for key, value in (os.environ if env is None else env).items()
            if 'proxy' not in key.lower()}


def docker(*args):
    return subprocess.check_output(['docker', *args], env=direct_environment(),
                                   text=True, timeout=60)


def filesystem_diff_sha(value):
    # Docker walks overlay directories in unstable order. Compare the set of
    # changed paths, not traversal order; this is not a file-content hash.
    return hashlib.sha256('\n'.join(sorted(value.splitlines())).encode()).hexdigest()


def assert_direct_docker(env=None, call=docker):
    """Host CLI variables alone do not control daemon or injected build proxies."""
    env = os.environ if env is None else env
    path = Path(env.get('DOCKER_CONFIG', str(Path.home()/'.docker')))/'config.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    if any(any(values.values()) for values in config.get('proxies', {}).values()):
        raise RuntimeError('Docker client injects proxies; direct-download preflight failed')
    daemon = json.loads(call('info', '--format', '{{json .}}'))
    if any(daemon.get(k) for k in ('HTTPProxy', 'HTTPSProxy')):
        raise RuntimeError('Docker daemon uses a proxy; direct-download preflight failed')
    return dict(checked_at=time.time(), daemon_proxy=False, client_proxy=False,
                inherited_proxy_variables_removed=True,
                container_proxy_variables=list(DIRECT_CONTAINER_ENV))


def validate_owner(network, containers, project, environment_dir):
    """Never delete active, foreign, built-in, or unproven orphan networks."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]+__env', project):
        raise ValueError('Unexpected Harbor project name')
    labels = network.get('Labels') or {}
    if (labels.get('com.docker.compose.project') != project or
            not labels.get('com.docker.compose.network') or network.get('Driver') != 'bridge'):
        raise ValueError('Network ownership mismatch')
    if network.get('Containers'):
        raise ValueError('Network has active endpoints')
    if not containers:
        raise ValueError('No retained container proves network ownership')
    expected = str(Path(environment_dir).resolve())
    for container in containers:
        labels = container['Config'].get('Labels') or {}
        if (labels.get('com.docker.compose.project') != project or
                labels.get('com.docker.compose.project.working_dir') != expected):
            raise ValueError('Container ownership mismatch')
        state = container['State']
        if state.get('Running') or state.get('Paused') or state.get('Restarting'):
            raise ValueError('Owned container still active')
        if state.get('Status') not in ('exited', 'created', 'dead'):
            raise ValueError('Unrecognized container state')


def reclaim_trial_networks(project, environment_dir, audit_dir, call=docker):
    """Remove only this finished trial's empty networks, retaining all containers.

    Recheck full IDs and owner state immediately before non-forced removal.
    Docker also rejects a network that acquires an active endpoint in the race.
    Container filesystems and images are never deleted or committed.
    """
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    network_ids = call('network', 'ls', '-q', '--no-trunc', '--filter',
                       'label=com.docker.compose.project='+project).split()
    records = []
    for ident in network_ids:
        def inspect():
            network = json.loads(call('network', 'inspect', ident))[0]
            ids = call('ps', '-aq', '--no-trunc', '--filter',
                       'label=com.docker.compose.project='+project).split()
            containers = json.loads(call('inspect', *ids)) if ids else []
            if network['Id'] != ident:
                raise ValueError('Network identity changed')
            validate_owner(network, containers, project, environment_dir)
            return network, containers
        network, containers = inspect()
        retained = {c['Id']: dict(image=c['Image'], state=c['State']['Status'],
                    diff_sha256=filesystem_diff_sha(call('diff', c['Id'])))
                    for c in containers}
        row = dict(at=time.time(), network=network, retained_containers=retained,
                   project=project, environment_dir=str(environment_dir), status='checked')
        path = audit_dir/(ident+'.json')
        # Preserve evidence of every separate attempt, including failed removals.
        with path.open('x') as f:
            json.dump(row, f, indent=2)
        _, current = inspect()
        if {c['Id'] for c in current} != set(retained):
            raise ValueError('Container cohort changed during reclamation')
        call('network', 'rm', ident)
        for cid, before in retained.items():
            after = json.loads(call('inspect', cid))[0]
            if (after['Image'] != before['image'] or after['State']['Status'] != before['state'] or
                    filesystem_diff_sha(call('diff', cid)) != before['diff_sha256']):
                raise ValueError('Retained container changed during reclamation')
        row.update(status='removed', retained_filesystem_diffs_unchanged=True, finished_at=time.time())
        done = path.with_suffix('.removed.json')
        with done.open('x') as f:
            json.dump(row, f, indent=2)
        records.append(row)
    return records
