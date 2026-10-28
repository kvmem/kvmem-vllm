"""Verified public terminal dependencies; no task files or model requests."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import tarfile
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bundle(path, task, image):
    path = Path(path)
    proof = json.loads((path / 'manifest.json').read_text())
    if (proof.get('task') != task or proof.get('image') != image or
            proof.get('offline_install_passed') is not True or
            proof.get('proxy_disabled') is not True):
        raise ValueError('Terminal dependency bundle identity/qualification mismatch')
    files = proof['files']
    if not files or any(not re.fullmatch(r'[A-Za-z0-9_.+%:~\-]+\.deb', n) for n in files):
        raise ValueError('Unsafe or empty dependency archive')
    archive = path / 'packages.tar'
    if sha(archive) != proof['archive_sha256']:
        raise ValueError('Terminal dependency archive changed')
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError('Dependency archive membership mismatch')
        for m in members:
            if not m.isfile():
                raise ValueError('Dependency archive contains a link/non-file')
            digest = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
            if digest != files[m.name]['sha256'] or digest != files[m.name]['apt_sha256']:
                raise ValueError('Dependency does not match authenticated apt metadata')
    return proof


async def install_bundle(session, path, audit):
    """Install only a task-bound, prequalified bundle before original setup.

    The caller supplies the manifest-frozen bundle path. Dependencies are copied,
    never mounted from a host package cache. Original setup then checks the tools
    normally. Exec and total setup limits remain unchanged.
    """
    task = Path(session.environment.environment_dir).parent.name
    image = session.environment.task_env_config.docker_image
    proof = validate_bundle(path, task, image)
    remote = '/tmp/kvmem-terminal-tools-' + uuid.uuid4().hex
    archive = remote + '.tar'
    await session.environment.upload_file(source_path=Path(path) / 'packages.tar', target_path=archive)
    command = (
        f"printf '%s  %s\\n' {shlex.quote(proof['archive_sha256'])} {shlex.quote(archive)} | sha256sum -c - && "
        f"mkdir {remote} && tar -xf {archive} -C {remote} && "
        f"DEBIAN_FRONTEND=noninteractive apt-get -y --no-download -o Dir::Cache::archives={remote} install {remote}/*.deb > {remote}.log 2>&1; "
        f"rc=$?; cat {remote}.log 2>/dev/null; test $rc -eq 0 && tmux -V && asciinema --version"
    )
    try:
        result = await session.environment.exec(command=command, user='root', timeout_sec=120)
    except RuntimeError as error:
        # Preserve the original exception and remote log; never mistake a
        # wrapped Docker exec timeout for a successful/optional installation.
        audit(event='terminal_bundle_failed', task=task, error=str(error), remote_log=remote+'.log')
        raise
    audit(event='terminal_bundle_installed', task=task, archive_sha256=proof['archive_sha256'],
          return_code=result.return_code, stdout=result.stdout, stderr=result.stderr)
    if result.return_code:
        raise RuntimeError('Offline terminal dependency installation failed; preserved setup log')
    return proof
