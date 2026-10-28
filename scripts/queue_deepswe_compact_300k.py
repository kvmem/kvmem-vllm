"""User-requested fresh 300K cohort after exact compact-index qualification.

The failed 512K experiment is preserved, not resumed or counted as completed.
This finite queue does not create or restart a periodic monitor.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import psutil

from scripts import run_agent_handoff_queue as q
from scripts import queue_deepswe_fastpath as old
from scripts import benchmark_deepswe_first16 as runner

ROOT = Path(__file__).resolve().parents[1]
MODEL_OVERLAYS = tuple('src/vllm_kvmem/'+name for name in (
    'authority.py', 'capture_runtime.py', 'options.py', 'query_key_fps.py'))
OVERLAYS = MODEL_OVERLAYS + ('scripts/queue_deepswe_compact_300k.py',
    'tests/test_compact_cpu_indices.py', 'tests/test_deepswe_compact_queue.py')


def profile_from(profile):
    result = copy.deepcopy(profile)
    result['max_model_len'] = 307200
    result['kvmem'].update(cpu_release_sealed_sums=True, query_fps_cache_fp8=True)
    return result


def validate_workload(before, after):
    for key in ('tasks', 'sampling', 'seed', 'workers', 'attempts_per_task',
                'termination_policies', 'optimized_handoff', 'duration_seconds',
                'setup_sha256', 'protocol_provenance', 'claude_sha256', 'scoring'):
        if before[key] != after[key]:
            raise ValueError('Changed original workload: '+key)
    if after['profile'] != profile_from(before['profile']):
        raise ValueError('Only the requested 300K cap and compact storage may change')
    if after.get('carried_results'):
        raise ValueError('This new-configuration cohort does not import old answers')


def stopped_prior(source):
    if q.read(source/'status.json')['status'] != 'failed':
        raise ValueError('Expected preserved failed 512K experiment')
    if q.live(q.read(source/'queue-process.json')):
        raise ValueError('Prior controller is still running')
    cleanup = q.read(source/'cleanup.json')
    if any(cleanup[k] for k in ('remaining_processes', 'remaining_owned_gpu_pids')):
        raise ValueError('Prior cleanup was incomplete')
    if any(q.live(owner) for owner in cleanup['owned']):
        raise ValueError('Prior owned process is still running')


def preserved_hashes(source):
    """Hash private container-owned transcripts without changing permissions."""
    result, private = {}, []
    for path in source.rglob('*'):
        if not path.is_file() or path.is_relative_to(source/'runtime') or '__pycache__' in path.parts:
            continue
        relative = str(path.relative_to(source))
        try:
            result[relative] = q.sha(path)
        except PermissionError:
            if 'claude-config' not in path.relative_to(source).parts:
                raise
            private.append(relative)
    if private:
        owners = {(path.stat().st_uid, path.stat().st_gid) for path in (source/rel for rel in private)}
        if len(owners) != 1:
            raise ValueError('Private evidence has differing container owners')
        uid, gid = owners.pop()
        task = q.read(source/'first16/manifest.json')['tasks'][0]
        image = task['images'][task['agent_image']]
        output = subprocess.check_output(['docker', 'run', '--rm', '--network', 'none',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--user', f'{uid}:{gid}', '--mount', f'type=bind,src={source},dst=/evidence,readonly',
            '--entrypoint', 'sha256sum', image, '--',
            *('/evidence/'+rel for rel in private)], text=True)
        rows = output.splitlines()
        if len(rows) != len(private):
            raise ValueError('Incomplete private evidence hashes')
        for relative, row in zip(private, rows):
            digest, filename = row.split('  ', 1)
            if filename != '/evidence/'+relative or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid private evidence hash')
            result[relative] = digest
    return result


def prepare(source, output, validation):
    if output.exists():
        raise ValueError('Use a new root; no automatic retry or overwrite')
    previous = q.verify_frozen(source)
    stopped_prior(source)
    if not validation.is_file():
        raise ValueError('Offline validation report required')
    checks = q.read(validation)
    if checks.get('status') != 'passed' or checks.get('exit_code') != 0:
        raise ValueError('Offline validation has not passed')
    for rel in OVERLAYS:
        if q.sha(ROOT/rel) != checks['source_hashes'].get(rel):
            raise ValueError('Source changed after offline validation: '+rel)
    output.mkdir(parents=True)
    runtime = output/'runtime'
    shutil.copytree(source/'runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel in OVERLAYS:
        shutil.copy2(ROOT/rel, runtime/rel)
    directory = output/'first16'
    directory.mkdir()
    profile = profile_from(q.read(source/'first16/profile.json'))
    q.save(directory/'profile.json', profile)
    shutil.copytree(source/'first16/protocol', directory/'protocol')
    task = runner.prepare(SimpleNamespace(reference=old.REFERENCE,
        config=directory/'profile.json', output=directory, workers=8,
        bash_repeat_limit=32, carry_result=[], backend_kind='kvmem', duration_seconds=0,
        optimized_handoff=True))
    before = q.read(source/'first16/manifest.json')
    task['differences'] = [item for item in before['differences']
                          if 'current limit524288' not in item] + [
        'User requested a separate 307200-token cohort after two storage-only optimizations.',
        'Sealed FP32 sums are released after native MeanK publication; incomplete tails retain sums.',
        'FPS4 stores original finite FP8 values; representative selection and scoring geometry are unchanged.',
        'The failed 512K cohort remains preserved and unscored; this is not partial recovery.',
        'The existing 8GiB host-memory stop remains. Eight full 300K histories are not guaranteed to fit.']
    validate_workload(before, task)
    q.save(directory/'manifest.json', task)
    shutil.copy2(source/'first16/baseline-comparison.json', directory/'baseline-comparison.json')
    shutil.copy2(source/'previous-vllm-outcomes.json', output/'previous-vllm-outcomes.json')
    shutil.copy2(validation, output/'offline-validation.json')
    evidence = preserved_hashes(source)
    q.save(output/'prior-preservation.json', dict(source=str(source), status='failed_preserved',
        files=evidence, scope='New user-requested configuration; no previous response, patch or outcome changed.'))
    old_model = {str(p.relative_to(source/'runtime')):q.sha(p)
                 for folder in ('src','csrc') for p in (source/'runtime'/folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    model = {rel:q.sha(runtime/rel) for rel in old_model}
    if {rel for rel in model if model[rel] != old_model[rel]} != set(MODEL_OVERLAYS):
        raise ValueError('Unexpected model changes beyond compact CPU indices')
    manifest = dict(source=str(source), source_manifest_sha256=q.sha(source/'manifest.json'),
        python=previous['python'], external_files=previous['external_files'],
        idle_seconds=60, poll_seconds=15, bridge_port=previous['bridge_port'],
        old_model_files=old_model, model_files=model,
        scope='Fresh first16, 8 agents, 300K logical cap, compact CPU indices. No SSD or sampling/policy change.',
        files={str(p.relative_to(output)):q.sha(p) for p in output.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts})
    q.save(output/'manifest.json', manifest)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate(output)


def validate(root):
    m = q.verify_frozen(root)
    source = Path(m['source'])
    if q.sha(source/'manifest.json') != m['source_manifest_sha256']:
        raise ValueError('Prior source identity changed')
    if preserved_hashes(source) != q.read(root/'prior-preservation.json')['files']:
        raise ValueError('Preserved prior experiment changed')
    for path, digest in m['external_files'].items():
        if q.sha(Path(path)) != digest:
            raise ValueError('Original task/client changed')
    before, task = q.read(source/'first16/manifest.json'), q.read(root/'first16/manifest.json')
    validate_workload(before, task)
    if q.read(root/'first16/profile.json') != task['profile']:
        raise ValueError('Frozen profile differs from task manifest')
    return m


def run(root):
    if ROOT.resolve() != (root/'runtime').resolve():
        raise ValueError('Use the frozen controller')
    m = validate(root)
    if q.read(root/'status.json')['status'] != 'prepared' or (root/'queue-process.json').exists():
        raise ValueError('No automatic retry/restart')
    stopped_prior(Path(m['source']))
    q.save(root/'queue-process.json', dict(**q.identity(psutil.Process()), at=time.time(),
        manifest_sha256=q.sha(root/'manifest.json')))
    owned = q.Owned()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            idle = None
            while True:
                if (root/'STOP').exists(): raise InterruptedError('User stopped queue')
                pids = q.gpu_pids()
                idle = (idle or time.monotonic()) if not pids else None
                q.save(root/'status.json', dict(status='waiting_for_gpu', gpu_pids=pids, at=time.time()))
                if idle and time.monotonic()-idle >= m['idle_seconds']: break
                time.sleep(m['poll_seconds'])
            validate(root)
            old.verify_images(q.read(root/'first16/manifest.json'))
            for port in (18880, m['bridge_port']): q.require_free_port(port)
            if psutil.virtual_memory().available < 90*2**30:
                raise ValueError('Insufficient free memory before qualification')
            if q.gpu_pids(): raise RuntimeError('Foreign GPU owner appeared')
            env = q.environment(root/'runtime')
            env.update(MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            q.save(root/'status.json', dict(status='gpu_index_qualification', at=time.time()))
            gpu = owned.start([m['python'], '-B', '-m', 'pytest', '-q',
                'tests/test_compact_cpu_indices.py'], root/'runtime',
                dict(env, KVMEM_TEST_CUDA='1'), root/'gpu-component.log', root/'gpu-component-process.json')
            while gpu.poll() is None:
                owned.track()
                time.sleep(1)
            if gpu.returncode: raise RuntimeError('Compact-index GPU qualification failed')
            if q.gpu_pids(): raise RuntimeError('GPU component did not release GPU')
            q.save(root/'gpu-qualification.json', dict(status='passed',
                log_sha256=q.sha(root/'gpu-component.log'), model_files=m['model_files'], at=time.time()))
            directory = root/'first16'
            args = [m['python'], '-B', str(root/'runtime/scripts/benchmark_deepswe_first16.py'),
                '--config', str(directory/'profile.json'), '--output', str(directory),
                '--workers', '8', '--backend-kind', 'kvmem', '--duration-seconds', '0',
                '--bridge-port', str(m['bridge_port']), '--bash-repeat-limit', '32',
                '--use-prepared', '--optimized-handoff', '--startup-timeout-seconds', '3600']
            process = owned.start(args, root/'runtime', env, directory/'controller.log',
                                  directory/'controller-process.json')
            q.save(root/'status.json', dict(status='qualification_then_testing', at=time.time()))
            while process.poll() is None:
                owned.track()
                if set(q.gpu_pids())-set(owned.rows):
                    raise RuntimeError('Foreign GPU owner appeared')
                time.sleep(5)
            if process.returncode: raise RuntimeError('DeepSWE controller failed; preserve attempts')
            report = old.final_review(root)
        cleanup = owned.stop()
        q.save(root/'cleanup.json', cleanup)
        if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
            raise ValueError('Owned cleanup incomplete')
        q.save(root/'status.json', dict(status='completed', solved=report['solved'],
            scored=report['scored'], context_ceiling=report['context_ceiling'], at=time.time()))
    except BaseException as error:
        q.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        q.save(root/'cleanup.json', owned.stop())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offline-validation', type=Path)
    args = parser.parse_args()
    if args.source:
        prepare(args.source.resolve(), args.output.resolve(), args.offline_validation.resolve())
    else:
        run(args.output.resolve())
