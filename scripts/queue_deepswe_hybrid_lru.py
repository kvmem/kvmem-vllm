"""Fresh first16/300K hybrid-LRU experiment; finite queue, no retries or timer."""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import psutil

from scripts import benchmark_deepswe_first16 as runner
from scripts import queue_deepswe_compact_300k as compact
from scripts import queue_deepswe_fastpath as old
from scripts import run_agent_handoff_queue as q

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'artifacts/deepswe-hybrid-compact-300k-20261025-r3'
DEFAULT_PROOF = ROOT/'artifacts/hybrid-lru-20261025-r1/validation.json'
DEFAULT_PROFILE = DEFAULT_PROOF.with_name('profile.json')
RUNTIME_DIRECTORIES = ('src', 'csrc', 'scripts', 'benchmarks', 'tests')
RUNTIME_FILES = ('pyproject.toml',)
# This exact generated file is the only mismatch observed across all 989
# source-manifest entries. Do not waive other pytest caches or Python bytecode.
SOURCE_MUTABLE_CACHE = 'runtime/.pytest_cache/v/cache/nodeids'
EXCLUDED = {'artifacts', 'data', '.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'node_modules'}
WORKLOAD_FIELDS = ('tasks', 'sampling', 'seed', 'workers', 'attempts_per_task',
    'termination_policies', 'optimized_handoff', 'duration_seconds', 'setup_sha256',
    'protocol_provenance', 'claude_sha256', 'scoring', 'backend_kind', 'failure_policy',
    'agent_timeout', 'verifier_timeout', 'network', 'cpu_admission')
LRU = dict(mode='delete', policy='lru', policy_params={'keep_mandatory': True},
    poll_ms=10, scan_blocks=256, delete_blocks=64, ready_low=8192, ready_high=9216)
GPU_TESTS = ('tests/test_historical_hybrid_gpu.py', 'tests/test_compact_cpu_indices.py',
             'tests/test_historical_compact_index.py')
GC_COUNTERS = ('published', 'ticks', 'examined', 'decisions', 'retired', 'cpu_released',
    'slots_released', 'pending_slots', 'pending_cpu_bytes', 'retries', 'stale',
    'policy_failures', 'shadow_decisions', 'admission_attempts', 'admission_retries', 'active_scopes')
MEMORY_COUNTERS = ('cpu_authority_bytes', 'cpu_index_bytes', 'cpu_index_owners',
    'cpu_payload_readers', 'cpu_payload_released_bytes', 'query_window_cache_bytes',
    'query_window_cache_reserved_bytes', 'session_checkpoint_payload_bytes', 'session_checkpoints',
    'archive_used_slots', 'pending_downloads', 'active_requests', 'retiring_requests',
    'session_retiring_requests', 'archive_io_pending_bytes', 'archive_io_peak_bytes',
    'disk_read_bytes', 'disk_write_bytes', 'block_graph_bytes', 'gc_wait_count', 'gc_wait_seconds')
ADMISSION_COUNTERS = ('budget_bytes', 'capacity_bytes', 'used_bytes', 'reserved_bytes', 'available_bytes',
    'active_requests', 'waiting_requests', 'admitted_requests', 'blocked_requests',
    'retained_bytes', 'progress_reserve_bytes', 'active', 'waiting', 'running', 'completed',
    'leases', 'retiring', 'deferrals', 'admissions', 'last_observed_archive_bytes',
    'total_wait_seconds', 'oldest_wait_seconds')


def profile_from(before):
    result = copy.deepcopy(before)
    result['served_model_name'] = 'kvmem-hybrid-lru'
    result['kvmem'].update(graph_mode='off', cpu_seal_serial_allocate=False,
                           historical_gc=copy.deepcopy(LRU))
    return result


def validate_workload(before, after):
    for field in WORKLOAD_FIELDS:
        if before[field] != after[field]:
            raise ValueError('Changed original workload: '+field)
    if (len(after['tasks']), after['workers'], after['attempts_per_task'],
            after['duration_seconds'], after['profile']['max_model_len']) != (16, 8, 1, 0, 307200):
        raise ValueError('Expected original first16, eight agents, single attempt and 300K')
    if after['profile'] != profile_from(before['profile']):
        raise ValueError('Only historical LRU and its graph/seal/model-name compatibility may change')
    if after.get('carried_results'):
        raise ValueError('Fresh configuration may not import old answers')


def stopped_prior(source):
    if q.read(source/'status.json')['status'] not in ('failed', 'completed'):
        raise ValueError('Predecessor is not terminal; do not interrupt it')
    if q.live(q.read(source/'queue-process.json')):
        raise ValueError('Prior controller is still running')
    cleanup = q.read(source/'cleanup.json')
    if any(cleanup[k] for k in ('remaining_processes', 'remaining_owned_gpu_pids')):
        raise ValueError('Prior cleanup was incomplete')
    if any(q.live(owner) for owner in cleanup['owned']):
        raise ValueError('Prior owned process is still running')


def relative_file(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or str(path) in ('', '.'):
        raise ValueError('Expected safe relative evidence path')
    return path


def verify_source(source, *, accepted_cache_changes=None):
    """Check every original hash, with one explicitly audited generated cache."""
    manifest = q.read(source/'manifest.json')
    changes = []
    for relative, expected in sorted(manifest['files'].items()):
        path = source/relative_file(relative)
        actual = q.sha(path)
        if actual == expected:
            continue
        if relative != SOURCE_MUTABLE_CACHE:
            raise ValueError('Frozen predecessor source/profile changed: '+relative)
        nodes = q.read(path)
        if not isinstance(nodes, list) or not all(isinstance(node, str) for node in nodes):
            raise ValueError('Unexpected generated pytest nodeids cache format')
        changes.append(dict(path=relative, original_sha256=expected, current_sha256=actual,
            reason='Prior controller froze pytest nodeids before its GPU qualification updated this generated test-discovery cache; no source, configuration or result exception.'))
    if accepted_cache_changes is not None and changes != accepted_cache_changes:
        raise ValueError('Previously audited predecessor cache changed again')
    return manifest, changes


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if any(part in EXCLUDED for part in relative.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_symlink():
            raise ValueError('Frozen sources may not traverse symlinks: '+str(relative))
        if path.is_file():
            result[str(relative)] = q.sha(path)
    return result


def runtime_delta(before, after):
    return dict(added={k: after[k] for k in sorted(after.keys()-before.keys())},
        removed={k: before[k] for k in sorted(before.keys()-after.keys())},
        changed={k: dict(before=before[k], after=after[k]) for k in sorted(before.keys() & after.keys())
                 if before[k] != after[k]})


def verify_component_proof(validation, profile, *, defer_gpu=False):
    checks = q.read(validation)
    full = (checks.get('status') == 'component_validation_passed' and
            checks.get('cuda_hybrid', {}).get('passed', 0) > 0)
    pending = (defer_gpu and checks.get('status') == 'cpu_component_validation_passed' and
               checks.get('cuda_hybrid') == {'status': 'pending_frozen_queue'})
    if (not (full or pending) or
            checks.get('cpu', {}).get('passed', 0) <= 0 or
            checks.get('native_index', {}).get('ctest_passed', 0) <= 0):
        raise ValueError('Passed LRU CPU/native and passed or explicitly queued CUDA validation required')
    hashes = checks.get('sha256', {})
    required = ('src/vllm_kvmem/authority.py', 'src/vllm_kvmem/historical_gc.py',
        'src/vllm_kvmem/query_key_fps.py', 'tests/test_historical_hybrid_gpu.py')
    if not all(name in hashes for name in required):
        raise ValueError('Incomplete component source proof')
    for rel, expected in hashes.items():
        if q.sha(ROOT/relative_file(rel)) != expected:
            raise ValueError('Source changed after component validation: '+rel)
    profile_rel = str(profile.relative_to(ROOT))
    if hashes.get(profile_rel) != q.sha(profile):
        raise ValueError('Profile was not bound by component validation')
    return checks


def prepare(source, output, validation, profile, *, defer_gpu=False):
    if output.exists():
        raise ValueError('Use a new root; no automatic retry or overwrite')
    previous, source_cache_changes = verify_source(source)
    stopped_prior(source)
    checks = verify_component_proof(validation, profile, defer_gpu=defer_gpu)
    before = q.read(source/'first16/manifest.json')
    requested = q.read(profile)
    if requested != profile_from(before['profile']):
        raise ValueError('Qualified LRU profile changed the original workload')
    evidence = compact.preserved_hashes(source)
    output.mkdir(parents=True)
    runtime = output/'runtime'
    runtime.mkdir()
    q.save(output/'source-cache-audit.json', dict(
        source_manifest_sha256=q.sha(source/'manifest.json'),
        checked_files=len(previous['files']), changes=source_cache_changes,
        scope='Only the exact observed generated pytest nodeids file may differ; the original manifest and all prior files remain unchanged.'))
    # Explicit directories only: never copy the workspace or historical artifacts.
    ignore = shutil.ignore_patterns(*EXCLUDED, '*.pyc', '*.pyo')
    for folder in RUNTIME_DIRECTORIES:
        inventory(ROOT/folder)  # Reject symlink escapes before copying.
        shutil.copytree(ROOT/folder, runtime/folder, ignore=ignore)
    for name in RUNTIME_FILES:
        shutil.copy2(ROOT/name, runtime/name)
    directory = output/'first16'
    directory.mkdir()
    shutil.copytree(source/'first16/protocol', directory/'protocol')
    shutil.copytree(source/'first16/setup', directory/'setup')
    q.save(directory/'profile.json', requested)
    task = runner.prepare(SimpleNamespace(reference=old.REFERENCE,
        config=directory/'profile.json', output=directory, workers=8, bash_repeat_limit=32,
        carry_result=[], backend_kind='kvmem', duration_seconds=0, optimized_handoff=True))
    task['differences'] = [
        'Fresh same first16/8-agent/307200 logical-token cohort; previous answers are preserved.',
        'Historical LRU permanently drops cold CPU KV; fixed sink/mandatory/read/restore protections remain.',
        'Graph-free 50:50 MeanK/FPS4 and compatible native historical allocation; sampling/tool limits unchanged.',
        'Physical historical retirement does not remove the original logical prompt limit.',
        'Final LRU verification requires policy retirement and release observed during formal inference.',
        'Historical trajectories and code differ; this is not a controlled throughput-only A/B.']
    validate_workload(before, task)
    q.save(directory/'manifest.json', task)
    # runner.prepare must reproduce the already frozen shell and verifier setup.
    if inventory(directory/'setup') != inventory(source/'first16/setup'):
        raise ValueError('Original frozen task setup changed')
    for name in ('baseline-comparison.json',):
        shutil.copy2(source/'first16'/name, directory/name)
    shutil.copy2(source/'previous-vllm-outcomes.json', output/'previous-vllm-outcomes.json')
    shutil.copy2(validation, output/'component-validation.json')
    for rel in checks['sha256']:
        target = output/'component-evidence'/relative_file(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
    q.save(output/'prior-preservation.json', dict(source=str(source), files=evidence,
        status='original_terminal_outcome_preserved', scope='No old answer, patch, score or workspace is reused.'))
    old_files, runtime_files = inventory(source/'runtime'), inventory(runtime)
    q.save(output/'runtime-diff.json', runtime_delta(old_files, runtime_files))
    frozen_files = inventory(output)
    # Evidence copies can deliberately include artifact/doc paths; only the
    # runtime-directory copy excludes those names.
    frozen_files.update({'component-evidence/'+rel: q.sha(output/'component-evidence'/rel)
                         for rel in checks['sha256']})
    manifest = dict(source=str(source), source_manifest_sha256=q.sha(source/'manifest.json'),
        python=previous['python'], external_files=previous['external_files'],
        idle_seconds=60, poll_seconds=15, bridge_port=previous['bridge_port'],
        runtime_files=runtime_files, old_runtime_files=old_files,
        model_files={k:v for k,v in runtime_files.items() if k.startswith(('src/', 'csrc/'))},
        component_source_hashes=checks['sha256'],
        component_gpu_deferred=checks['status'] == 'cpu_component_validation_passed',
        gpu_qualification_before_inference=True,
        scope='Fresh first16/8-agent/300K, compact CPU indices and permanent historical LRU; no SSD.',
        files=frozen_files)
    q.save(output/'manifest.json', manifest)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate(output)


def validate(root):
    manifest = q.verify_frozen(root)
    if manifest.get('component_gpu_deferred') and not manifest.get('gpu_qualification_before_inference'):
        raise ValueError('Deferred GPU validation requires the frozen pre-inference gate')
    source = Path(manifest['source'])
    if q.sha(source/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Prior source identity changed')
    verify_source(source, accepted_cache_changes=q.read(root/'source-cache-audit.json')['changes'])
    if compact.preserved_hashes(source) != q.read(root/'prior-preservation.json')['files']:
        raise ValueError('Preserved prior experiment changed')
    if inventory(root/'runtime') != manifest['runtime_files']:
        raise ValueError('Frozen runtime inventory changed')
    if inventory(source/'runtime') != manifest['old_runtime_files']:
        raise ValueError('Prior runtime inventory changed')
    for relative, expected in manifest['component_source_hashes'].items():
        if q.sha(root/'component-evidence'/relative_file(relative)) != expected:
            raise ValueError('Component evidence copy changed: '+relative)
    for path, expected in manifest['external_files'].items():
        if q.sha(Path(path)) != expected:
            raise ValueError('Original task/client changed')
    task = q.read(root/'first16/manifest.json')
    validate_workload(q.read(source/'first16/manifest.json'), task)
    if q.read(root/'first16/profile.json') != task['profile']:
        raise ValueError('Frozen profile differs from task manifest')
    return manifest


def numeric_fields(value, names):
    return {name:value[name] for name in names if type(value.get(name)) in (int, float)
            and math.isfinite(value[name]) and value[name] >= 0}


def safe_read(path):
    try:
        value = q.read(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def sample(root):
    directory = root/'first16'
    capture = directory/'server/capture'
    archive, progress = safe_read(capture/'archive.json'), safe_read(capture/'progress.json')
    state, work = safe_read(directory/'run.json'), safe_read(capture/'public_work.json')
    running = state.get('status') == 'running' and bool(state.get('tasks')) and not state.get('ended_at')
    phase = 'formal' if running else 'cleanup' if state.get('ended_at') else 'model_qualification'
    histories = progress.get('accepted_history', {})
    histories = sorted(v for v in (histories.values() if isinstance(histories, dict) else ())
                       if type(v) is int and v >= 0)
    instance = work.get('counter_id')
    if not isinstance(instance, str) or len(instance) != 32 or any(c not in '0123456789abcdef' for c in instance):
        instance = None
    result = dict(at=time.time(), phase=phase,
        engine_instance=instance, engine_pid=work.get('pid') if type(work.get('pid')) is int else None,
        host_available_bytes=psutil.virtual_memory().available,
        memory=numeric_fields(archive, MEMORY_COUNTERS),
        progress=numeric_fields(progress, ('batches_submitted', 'full_graph_decode_calls',
            'archive_used_slots', 'pending_downloads', 'completed_requests', 'retiring_requests')),
        accepted_history=dict(owners=len(histories), total=sum(histories), maximum=max(histories, default=0)),
        historical_gc={})
    # No request IDs, prompts, tools, completion text, tokens or environment values.
    for origin, value in (('archive', archive), ('progress', progress)):
        gc = value.get('historical_gc') or {}
        if not isinstance(gc, dict):
            gc = {}
        row = numeric_fields(gc, GC_COUNTERS)
        row.update(mode=gc.get('mode') if gc.get('mode') in ('delete', 'off', 'shadow') else None,
                   policy=gc.get('policy') if gc.get('policy') == 'lru' else None)
        result['historical_gc'][origin] = row
    result['cpu_admission'] = {}
    scheduler = safe_read(capture/'scheduler_resources.json').get('cpu_history_admission', {})
    if isinstance(scheduler, dict):
        result['cpu_admission']['scheduler_resources'] = numeric_fields(scheduler, ADMISSION_COUNTERS)
    for name in ('bridge/cpu_admission.json', 'bridge/cpu-admission.json', 'server/capture/cpu_admission.json'):
        fields = numeric_fields(safe_read(directory/name), ADMISSION_COUNTERS)
        if fields:
            result['cpu_admission'][name] = fields
    with (root/'lru-samples.jsonl').open('a') as stream:
        stream.write(json.dumps(result, sort_keys=True)+'\n')
    return result


def lru_review(root):
    rows = [json.loads(line) for line in (root/'lru-samples.jsonl').read_text().splitlines()]
    instances = {}
    for row in rows:
        if row['phase'] != 'formal' or not row.get('engine_instance'):
            continue
        for origin, gc in row['historical_gc'].items():
            if gc.get('mode') != 'delete' or gc.get('policy') != 'lru':
                continue
            instances.setdefault((row['engine_instance'], origin), []).append(gc)
    windows = []
    for (instance, origin), samples in instances.items():
        counts = {name:max(s.get(name, 0) for s in samples)-min(s.get(name, 0) for s in samples)
                  for name in ('retired', 'cpu_released', 'slots_released')}
        windows.append(dict(engine_instance=instance, origin=origin, formal_samples=len(samples),
            increases=counts, observed=all(counts[name] > 0 for name in counts)))
    policy_failures = max((gc.get('policy_failures', 0) for row in rows if row['phase'] == 'formal'
        for gc in row['historical_gc'].values()), default=0)
    passed = any(window['observed'] for window in windows) and not policy_failures
    report = dict(status='failed' if policy_failures else 'passed' if passed else 'not_exercised',
        policy_failures=policy_failures,
        criterion='Policy retirement plus CPU and slot release increases within one engine instance during formal tasks; cleanup is excluded.',
        windows=windows, samples_sha256=q.sha(root/'lru-samples.jsonl'),
        phase_samples={phase:sum(row['phase'] == phase for row in rows)
                       for phase in ('model_qualification', 'formal', 'cleanup')})
    q.save(root/'lru-validation.json', report)
    return report


def stop_requested(root):
    if (root/'STOP').exists():
        raise InterruptedError('User stopped this finite queue')


def qualification_command(manifest, env):
    return ([manifest['python'], '-B', '-m', 'pytest', '-q', *GPU_TESTS],
            dict(env, KVMEM_RUN_GPU_TESTS='1', KVMEM_TEST_CUDA='1'))


def acquire_coordination(root, lock, poll_seconds):
    # Another qualification may have CPU-only gaps between GPU runs. Wait for
    # its shared lock as well as idle GPU; do not treat contention as failure.
    while True:
        stop_requested(root)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            return
        except BlockingIOError:
            q.save(root/'status.json', dict(status='waiting_for_gpu',
                reason='another_qualification_holds_lock', at=time.time()))
            time.sleep(poll_seconds)


def run(root):
    if ROOT.resolve() != (root/'runtime').resolve():
        raise ValueError('Use the frozen controller')
    manifest = validate(root)
    if q.read(root/'status.json')['status'] != 'prepared' or (root/'queue-process.json').exists():
        raise ValueError('No automatic retry/restart')
    stopped_prior(Path(manifest['source']))
    q.save(root/'queue-process.json', dict(**q.identity(psutil.Process()), at=time.time(),
        manifest_sha256=q.sha(root/'manifest.json')))
    owned = q.Owned()
    with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
        try:
            acquire_coordination(root, lock, manifest['poll_seconds'])
            idle = None
            while True:
                stop_requested(root)
                pids = q.gpu_pids()
                idle = (idle if idle is not None else time.monotonic()) if not pids else None
                q.save(root/'status.json', dict(status='waiting_for_gpu', gpu_pids=pids, at=time.time()))
                if idle is not None and time.monotonic()-idle >= manifest['idle_seconds']:
                    break
                time.sleep(manifest['poll_seconds'])
            validate(root)
            stopped_prior(Path(manifest['source']))
            old.verify_images(q.read(root/'first16/manifest.json'))
            for port in (18880, manifest['bridge_port']):
                q.require_free_port(port)
            if psutil.virtual_memory().available < 90*2**30:
                raise ValueError('Insufficient free memory before qualification')
            if q.gpu_pids():
                raise RuntimeError('Foreign GPU owner appeared')
            env = q.environment(root/'runtime')
            env.update(MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            q.save(root/'status.json', dict(status='gpu_lru_qualification', at=time.time()))
            gpu_args, gpu_env = qualification_command(manifest, env)
            gpu = owned.start(gpu_args, root/'runtime', gpu_env,
                root/'gpu-component.log', root/'gpu-component-process.json')
            while gpu.poll() is None:
                stop_requested(root)
                owned.track()
                time.sleep(1)
            if gpu.returncode:
                raise RuntimeError('Historical LRU GPU qualification failed')
            if q.gpu_pids():
                raise RuntimeError('GPU component did not release GPU')
            q.save(root/'gpu-qualification.json', dict(status='passed', tests=GPU_TESTS,
                log_sha256=q.sha(root/'gpu-component.log'), model_files=manifest['model_files'], at=time.time()))
            directory = root/'first16'
            args = [manifest['python'], '-B', str(root/'runtime/scripts/benchmark_deepswe_first16.py'),
                '--config', str(directory/'profile.json'), '--output', str(directory),
                '--workers', '8', '--backend-kind', 'kvmem', '--duration-seconds', '0',
                '--bridge-port', str(manifest['bridge_port']), '--bash-repeat-limit', '32',
                '--use-prepared', '--optimized-handoff', '--startup-timeout-seconds', '3600']
            process = owned.start(args, root/'runtime', env, directory/'controller.log',
                                  directory/'controller-process.json')
            q.save(root/'status.json', dict(status='qualification_then_testing', at=time.time()))
            while process.poll() is None:
                stop_requested(root)
                owned.track()
                if set(q.gpu_pids())-set(owned.rows):
                    raise RuntimeError('Foreign GPU owner appeared')
                sample(root)
                time.sleep(5)
            sample(root)
            if process.returncode:
                raise RuntimeError('DeepSWE controller failed; preserve attempts')
            report, lru = old.final_review(root), lru_review(root)
            if lru['status'] == 'failed':
                raise RuntimeError('Historical LRU policy failure observed; retain task results and diagnostics')
            cleanup = owned.stop()
            q.save(root/'cleanup.json', cleanup)
            if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
                raise ValueError('Owned cleanup incomplete')
            q.save(root/'status.json', dict(status='completed' if lru['status'] == 'passed' else 'completed_lru_not_exercised',
                lru_verified=lru['status'] == 'passed', solved=report['solved'], scored=report['scored'],
                context_ceiling=report['context_ceiling'], at=time.time()))
        except BaseException as error:
            q.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            q.save(root/'cleanup.json', owned.stop())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component-validation', type=Path, default=DEFAULT_PROOF)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--defer-gpu-component', action='store_true',
        help='Freeze passed CPU/native evidence now; CUDA qualification still must pass after GPU idle and before any model or task starts.')
    args = parser.parse_args()
    if args.source:
        prepare(args.source.resolve(), args.output.resolve(), args.component_validation.resolve(), args.profile.resolve(),
                defer_gpu=args.defer_gpu_component)
    else:
        run(args.output.resolve())
