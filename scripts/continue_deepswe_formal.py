"""User-authorized formal-only handoff; preserve the incomplete 10M GC trial."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import psutil

from scripts import run_agent_handoff_queue as q
from scripts import queue_deepswe_hybrid_lru as lru
from scripts import queue_deepswe_fastpath as old

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ('scripts/benchmark_deepswe_first16.py', 'scripts/continue_deepswe_formal.py')
PROOFS = ('gpu-qualification.json', 'first16/atomic-query-preflight.json',
          'first16/short-tail-preflight.json', 'first16/generation-preflight.json',
          'first16/cleanup-preflight.json')


def fresh_engine_archive(capture):
    # This branch starts a fresh owned service in an unused run directory. The
    # archive report is first written after a public request retires; it may
    # legitimately not exist yet. Never reuse this check for an adopted engine.
    work = q.read(capture/'public_work.json') if (capture/'public_work.json').exists() else {}
    if any(work.get(phase, {}).get('public_batches', 0) for phase in ('submitted', 'completed')):
        raise ValueError('Formal-only service unexpectedly processed earlier public requests')
    if (capture/'archive.json').exists():
        from scripts.benchmark_deepswe_first16 import wait_preflight_release
        return wait_preflight_release(capture)
    return {}


def validate_workload(before, after):
    expected = copy.deepcopy(before)
    expected.pop('long_gc_qualification')
    expected['preflight_reuse'] = 'preflight-reuse-proof.json'
    expected['differences'].append('User stopped the incomplete synthetic10M trial and requested direct formal tasks; passed preflights reused, not rerun.')
    if after != expected:
        raise ValueError('Formal handoff changed tasks, profile, sampling or tool/scoring limits')


def validate_reuse(root):
    manifest = q.verify_frozen(root)
    proof = q.read(root/'preflight-reuse-proof.json')
    source = Path(proof['source'])
    q.verify_frozen(source)
    lru.stopped_prior(source)
    if q.read(source/'first16/run.json').get('tasks'):
        raise ValueError('Formal tasks already attempted; separate partial recovery required')
    if q.read(source/'user-preflight-stop.json').get('reason') != 'user_requested_stop_long_gc_and_start_real_tasks':
        raise ValueError('Missing explicit user-directed transition')
    if proof.get('long_gc_status') != 'user_stopped_incomplete':
        raise ValueError('Incomplete long-GC trial cannot be reported as passed')
    for name, digest in proof['source_files'].items():
        if q.sha(source/name) != digest:
            raise ValueError('Preserved source evidence changed: '+name)
    for name in PROOFS:
        value = q.read(source/name)
        if not (value.get('passed') is True or value.get('status') == 'passed'):
            raise ValueError('Required completed proof missing: '+name)
    generation = q.read(source/'first16/generation-preflight.json')
    if (generation['concurrency'] != 8 or generation['generation_tokens'] != 16384
            or len(generation['results']) != 8
            or not all(r['cleanup_passed'] for r in generation['results'])):
        raise ValueError('Incomplete eight-way full generation proof')
    current = lru.inventory(root/'runtime')
    previous = q.read(source/'manifest.json')['runtime_files']
    delta = {name for name in current.keys() | previous.keys() if current.get(name) != previous.get(name)}
    if delta != set(OVERLAYS) or current != manifest['runtime_files']:
        raise ValueError('Only the formal controller handoff may differ')
    validate_workload(q.read(source/'first16/manifest.json'), q.read(root/'first16/manifest.json'))
    if q.read(root/'first16/profile.json') != q.read(source/'first16/profile.json'):
        raise ValueError('Inference profile changed')
    for path, digest in manifest['external_files'].items():
        if q.sha(Path(path)) != digest:
            raise ValueError('External task/client changed')
    return proof


def prepare(source, output):
    if output.exists():
        raise ValueError('Use a new root; never overwrite existing attempts')
    previous = q.verify_frozen(source)
    lru.stopped_prior(source)
    if q.read(source/'first16/run.json').get('tasks'):
        raise ValueError('Existing formal answers cannot be restarted')
    output.mkdir(parents=True)
    shutil.copytree(source/'runtime', output/'runtime',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in OVERLAYS:
        shutil.copy2(ROOT/name, output/'runtime'/name)
    directory = output/'first16'; directory.mkdir()
    for name in ('profile.json', 'tasks.txt', 'baseline-comparison.json'):
        shutil.copy2(source/'first16'/name, directory/name)
    for name in ('setup', 'protocol'):
        shutil.copytree(source/'first16'/name, directory/name)
    task = q.read(source/'first16/manifest.json')
    task.pop('long_gc_qualification')
    task['preflight_reuse'] = 'preflight-reuse-proof.json'
    task['differences'].append('User stopped the incomplete synthetic10M trial and requested direct formal tasks; passed preflights reused, not rerun.')
    q.save(directory/'manifest.json', task)
    shutil.copy2(source/'previous-vllm-outcomes.json', output/'previous-vllm-outcomes.json')
    preserved = lru.inventory(source)
    q.save(output/'preflight-reuse-proof.json', dict(source=str(source), source_files=preserved,
        long_gc_status='user_stopped_incomplete', formal_answers_previously_attempted=0,
        reused=list(PROOFS), at=time.time()))
    runtime = lru.inventory(output/'runtime')
    manifest = dict(python=previous['python'], source=str(source),
        bridge_port=previous['bridge_port'], external_files=previous['external_files'],
        runtime_files=runtime, model_files={k:v for k,v in runtime.items() if k.startswith(('src/', 'csrc/'))},
        files=lru.inventory(output), idle_seconds=60, poll_seconds=5,
        scope='User-directed formal-only DeepSWE16, 96K/16K, 10M allowance, eight agents; long-GC trial incomplete.')
    q.save(output/'manifest.json', manifest)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate_reuse(output)


def run(root):
    if ROOT.resolve() != (root/'runtime').resolve():
        raise ValueError('Use frozen controller')
    validate_reuse(root)
    manifest = q.read(root/'manifest.json')
    if (root/'queue-process.json').exists() or q.read(root/'status.json')['status'] != 'prepared':
        raise ValueError('No automatic restart/retry')
    q.save(root/'queue-process.json', dict(**q.identity(psutil.Process()), at=time.time(),
        manifest_sha256=q.sha(root/'manifest.json')))
    owned = q.Owned()
    with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
        try:
            lru.acquire_coordination(root, lock, 5)
            idle = None
            while True:
                lru.stop_requested(root)
                pids = q.gpu_pids()
                idle = (idle if idle is not None else time.monotonic()) if not pids else None
                q.save(root/'status.json', dict(status='waiting_for_gpu', gpu_pids=pids, at=time.time()))
                if idle is not None and time.monotonic()-idle >= 60:
                    break
                time.sleep(5)
            validate_reuse(root)
            task = q.read(root/'first16/manifest.json')
            old.verify_images(task)
            for port in (task['profile']['port'], manifest['bridge_port']):
                q.require_free_port(port)
            if psutil.virtual_memory().available < 90*2**30 or q.gpu_pids():
                raise ValueError('Insufficient free host/GPU resources')
            env = q.environment(root/'runtime')
            env.update(MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            directory = root/'first16'
            args = [manifest['python'], '-B', str(root/'runtime/scripts/benchmark_deepswe_first16.py'),
                '--config', str(directory/'profile.json'), '--output', str(directory),
                '--workers', '8', '--backend-kind', 'kvmem', '--duration-seconds', '0',
                '--bridge-port', str(manifest['bridge_port']), '--bash-repeat-limit', '32',
                '--use-prepared', '--optimized-handoff', '--startup-timeout-seconds', '3600']
            process = owned.start(args, root/'runtime', env, directory/'controller.log', directory/'controller-process.json')
            q.save(root/'status.json', dict(status='formal_starting', at=time.time(),
                long_gc_status='user_stopped_incomplete'))
            while process.poll() is None:
                lru.stop_requested(root); owned.track()
                if set(q.gpu_pids())-set(owned.rows):
                    raise RuntimeError('Foreign GPU owner appeared')
                lru.sample(root); time.sleep(5)
            lru.sample(root)
            if process.returncode:
                raise RuntimeError('Formal controller failed; preserve all attempted answers')
            report, gc = old.final_review(root), lru.lru_review(root)
            if gc['status'] == 'failed':
                raise RuntimeError('LRU policy failure')
            cleanup = owned.stop(); q.save(root/'cleanup.json', cleanup)
            if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
                raise RuntimeError('Owned cleanup incomplete')
            q.save(root/'status.json', dict(status='completed', at=time.time(),
                scored=report['scored'], solved=report['solved'], lru_verified=gc['status']=='passed',
                long_gc_status='user_stopped_incomplete'))
        except BaseException as error:
            q.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            q.save(root/'cleanup.json', owned.stop())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source:
        prepare(args.source.resolve(), args.output.resolve())
    else:
        run(args.output.resolve())
