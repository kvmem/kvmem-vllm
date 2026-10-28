"""Run only untouched, predeclared interference arms after byte qualification."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def completed(root, arm):
    directory = root/arm
    if read(directory/'status.json')['status'] != 'completed':
        raise ValueError('Arm did not complete: '+arm)
    if read(directory/'execution-audit.json')['status'] != 'passed':
        raise ValueError('Arm audit did not pass: '+arm)
    cleanup = read(directory/'cleanup.json')
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
        raise ValueError('Arm still owns resources: '+arm)
    for rel, expected in read(directory/'evidence-hashes.json').items():
        if hashlib.sha256((directory/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Completed evidence changed: '+rel)


def run(root):
    with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = root/'finite-queue-started.json'
        with started.open('x') as f:
            json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time(),
                at=time.time(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), f)
        qualification = root/'qualification'
        binding = read(qualification/'controller-process.json')
        while True:
            state = read(qualification/'status.json')
            if state['status'] == 'failed':
                raise RuntimeError('Qualification failed; no performance arms started')
            if (qualification/'cleanup.json').exists():
                break
            try:
                process = psutil.Process(binding['pid'])
                if abs(process.create_time()-binding['created']) > .1 or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    raise RuntimeError('Qualification exited without cleanup')
            except psutil.NoSuchProcess:
                raise RuntimeError('Qualification missing without cleanup') from None
            time.sleep(2)
        completed(root, 'qualification')
        manifest = read(root/'manifest.json')
        for arm in manifest.get('execution_order', ('control', 'capture', 'cap512', 'cap256')):
            if (root/arm/'started.json').exists():
                raise ValueError('Never rerun an attempted arm: '+arm)
            save(root/'status.json', dict(status='running', active=arm, at=time.time()))
            env = os.environ.copy()
            env['PYTHONPATH'] = ':'.join(str(root/'runtime'/p) for p in ('', 'src', 'scripts'))
            with (root/arm/'controller.log').open('x') as output:
                child = subprocess.Popen([manifest['python'],
                    str(root/'runtime/scripts/benchmark_agent_interference.py'),
                    '--output', str(root), '--arm', arm], cwd=root/'runtime', env=env,
                    stdout=output, stderr=subprocess.STDOUT)
                save(root/arm/'controller-process.json', dict(pid=child.pid,
                    created=psutil.Process(child.pid).create_time(), at=time.time()))
                if child.wait() != 0:
                    raise RuntimeError('Performance arm failed: '+arm)
            completed(root, arm)
        save(root/'status.json', dict(status='needs_analysis', at=time.time(),
            component_only=True, real_agent_gain_not_measured=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root.resolve())
    except BaseException as error:
        save(args.root/'finite-queue-failure.json', dict(error=repr(error), at=time.time()))
        raise
