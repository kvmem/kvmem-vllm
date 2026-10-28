"""Frozen 96K/16K, eight-owner near-10M GC gate followed by fresh DeepSWE16.

No automatic retries, timer, foreign process control, or reused task answers.
Only an explicit overlay is applied to the preserved completed LRU runtime.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from scripts import queue_deepswe_hybrid_lru as lru
from scripts import queue_deepswe_fastpath as old
from scripts import queue_deepswe_compact_300k as compact
from scripts import benchmark_deepswe_first16 as runner
from scripts import run_agent_handoff_queue as q
from scripts.deepswe_long_gc_preflight import SPEC

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ('src/vllm_kvmem/serve.py', 'scripts/benchmark_deepswe_first16.py',
    'scripts/deepswe_long_gc_preflight.py', 'scripts/queue_deepswe_lru_10m.py',
    'scripts/deepswe_dashboard.py', 'benchmarks/deepswe/dashboard/index.html',
    'tests/test_deepswe_lru_10m.py', 'src/vllm_kvmem/logical_capacity.py',
    'src/vllm_kvmem/integration.py', 'tests/test_logical_capacity.py')


def profile_from(before):
    result = copy.deepcopy(before)
    result.update(max_model_len=10_000_000, served_model_name='kvmem-hybrid-lru-96k')
    result['kvmem']['select_budget'] = 98304
    return result


def client_script(source):
    anchor = 'export MAX_THINKING_TOKENS=4096\n'
    if source.count(anchor) != 1 or 'export CLAUDE_CODE_MAX_OUTPUT_TOKENS=' in source:
        raise ValueError('Unexpected original pinned client setup')
    return source.replace(anchor, anchor+'export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384\n')


def validate_workload(before, after):
    for field in lru.WORKLOAD_FIELDS:
        if field in ('setup_sha256', 'cpu_admission'):
            continue
        if before[field] != after[field]:
            raise ValueError('Changed original workload: '+field)
    cpu = copy.deepcopy(before['cpu_admission']); cpu['max_history_tokens'] = 10_000_000
    if after['cpu_admission'] != cpu:
        raise ValueError('CPU credits/budget changed')
    if after['profile'] != profile_from(before['profile']):
        raise ValueError('Only K=96K and logical10M/model-name may change')
    if (len(after['tasks']), after['workers'], after['attempts_per_task'], after['duration_seconds']) != (16,8,1,0):
        raise ValueError('Expected original16, eight workers, one attempt, no duration truncation')
    if after.get('carried_results') or after.get('long_gc_qualification') != SPEC:
        raise ValueError('Fresh answers and exact mandatory long-GC gate required')
    if after.get('client_max_output_tokens') != 16384:
        raise ValueError('Client output reservation must match native G')


def failed_startup_proof(failed, source):
    q.verify_frozen(failed)
    lru.stopped_prior(failed)
    if q.read(failed/'status.json').get('status') != 'failed':
        raise ValueError('Only a stopped failed startup can be preserved here')
    if q.read(failed/'manifest.json')['source'] != str(source):
        raise ValueError('Failed startup has another workload source')
    if q.read(failed/'first16/profile.json') != profile_from(q.read(source/'first16/profile.json')):
        raise ValueError('Startup recovery cannot change the requested profile')
    if q.read(failed/'first16/run.json').get('tasks'):
        raise ValueError('Task answers need a separate audited partial recovery')
    if list((failed/'first16/preflight-requests').glob('*.json')):
        raise ValueError('Completed qualification requests must not be regenerated')
    work = failed/'first16/server/capture/public_work.json'
    if work.exists() and q.read(work).get('completed',{}).get('public_batches',0):
        raise ValueError('Not a startup-only failure')
    return dict(root=str(failed), files=lru.inventory(failed),
        public_requests=0, task_answers=0,
        reason='Native startup counted logical10M as resident GPU KV. New adapter retains native allocation checks for K+G and validates all eight owners.')


def prepare(source, output, validation, failed_startup=None):
    if output.exists():
        raise ValueError('Use a new root; preserve all prior attempts')
    previous = q.verify_frozen(source)
    lru.stopped_prior(source)
    if q.read(source/'status.json').get('status') != 'completed':
        raise ValueError('Expected completed LRU predecessor')
    proof = q.read(validation)
    if proof.get('status') != 'passed' or proof.get('exit_code') != 0:
        raise ValueError('Passed offline qualification required')
    for name in OVERLAYS:
        if proof['source_hashes'].get(name) != q.sha(ROOT/name):
            raise ValueError('Overlay changed after offline validation: '+name)
    before = q.read(source/'first16/manifest.json')
    if before['profile']['kvmem']['historical_gc'] != lru.LRU:
        raise ValueError('Expected unchanged source LRU policy')
    evidence = compact.preserved_hashes(source)
    startup = failed_startup_proof(failed_startup, source) if failed_startup else None
    output.mkdir(parents=True)
    runtime = output/'runtime'
    shutil.copytree(source/'runtime', runtime,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    for name in OVERLAYS:
        (runtime/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, runtime/name)
    directory = output/'first16'; directory.mkdir()
    shutil.copytree(source/'first16/protocol', directory/'protocol')
    profile = profile_from(before['profile'])
    q.save(directory/'profile.json', profile)
    task = runner.prepare(SimpleNamespace(reference=old.REFERENCE, config=directory/'profile.json',
        output=directory, workers=8, bash_repeat_limit=32, carry_result=[], backend_kind='kvmem',
        duration_seconds=0, optimized_handoff=True))
    for path in (directory/'setup').iterdir():
        original = source/'first16/setup'/path.name
        if path.read_bytes() != original.read_bytes():
            raise ValueError('Unrequested original task setup change')
    shell = directory/'setup/container_claude_agent.sh'
    shell.write_text(client_script(shell.read_text()))
    task['setup_sha256'] = {p.name:q.sha(p) for p in (directory/'setup').iterdir()}
    task['client_max_output_tokens'] = 16384
    task['long_gc_qualification'] = copy.deepcopy(SPEC)
    task['differences'] = [
        'User-authorized fresh 96K selection/16K generation, eight agents, decimal10M logical ceiling.',
        'Original16 tasks, images, protocol, thinking4096, sampling, seeds, guards and scoring unchanged.',
        'Client max output explicitly16384, matching the already-existing native output cap.',
        'LRU/50:50 recall/CPU100GB and all physical budgets except K remain unchanged; no SSD.',
        'Mandatory eight-owner synthetic actual near10M growth gate before formal tasks; outputs constrained only in that gate.',
        'Previous300K outcomes remain preserved. Larger K and longer logical allowance are not a single-variable performance comparison.']
    validate_workload(before, task); q.save(directory/'manifest.json', task)
    shutil.copy2(source/'first16/baseline-comparison.json', directory/'baseline-comparison.json')
    shutil.copy2(source/'previous-vllm-outcomes.json', output/'previous-vllm-outcomes.json')
    shutil.copy2(validation, output/'offline-validation.json')
    q.save(output/'prior-preservation.json', dict(source=str(source), files=evidence,
        status='completed_original_outcomes_preserved', scope='No old answers/workspaces are resumed or replaced.'))
    if startup:
        q.save(output/'startup-recovery-audit.json', startup)
    old_files, new_files = lru.inventory(source/'runtime'), lru.inventory(runtime)
    changed = {k for k in new_files if new_files[k] != old_files.get(k)}
    if changed != set(OVERLAYS):
        raise ValueError('Unexpected frozen runtime delta: '+repr(changed))
    q.save(output/'runtime-diff.json', lru.runtime_delta(old_files,new_files))
    manifest = dict(source=str(source), source_manifest_sha256=q.sha(source/'manifest.json'),
        python=previous['python'], external_files=previous['external_files'], bridge_port=previous['bridge_port'],
        idle_seconds=60, poll_seconds=15, runtime_files=new_files, old_runtime_files=old_files,
        model_files={k:v for k,v in new_files.items() if k.startswith(('src/','csrc/'))},
        overlay_files={k:q.sha(runtime/k) for k in OVERLAYS},
        component_gpu_deferred=True, gpu_qualification_before_inference=True,
        scope='96K/16K; real near10M eight-owner GC qualification then original DeepSWE16, one fresh attempt.',
        files=lru.inventory(output))
    q.save(output/'manifest.json',manifest)
    q.save(output/'status.json',dict(status='prepared',at=time.time()))
    validate(output)


def validate(root):
    manifest = q.verify_frozen(root)
    source = Path(manifest['source'])
    if q.sha(source/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Original manifest changed')
    if lru.inventory(source/'runtime') != manifest['old_runtime_files']:
        raise ValueError('Original runtime changed')
    if lru.inventory(root/'runtime') != manifest['runtime_files']:
        raise ValueError('Frozen runtime changed')
    if compact.preserved_hashes(source) != q.read(root/'prior-preservation.json')['files']:
        raise ValueError('Original experiment evidence changed')
    if (root/'startup-recovery-audit.json').exists():
        failure = q.read(root/'startup-recovery-audit.json')
        if lru.inventory(Path(failure['root'])) != failure['files']:
            raise ValueError('Preserved failed startup changed')
    for path, expected in manifest['external_files'].items():
        if q.sha(Path(path)) != expected:
            raise ValueError('Original external task/client changed')
    before, after = q.read(source/'first16/manifest.json'), q.read(root/'first16/manifest.json')
    validate_workload(before,after)
    if q.read(root/'first16/profile.json') != after['profile']:
        raise ValueError('Profile does not match manifest')
    for name, digest in after['setup_sha256'].items():
        path=root/'first16/setup'/name
        expected=(client_script((source/'first16/setup'/name).read_text()).encode()
                  if name=='container_claude_agent.sh' else (source/'first16/setup'/name).read_bytes())
        if path.read_bytes()!=expected or q.sha(path)!=digest:
            raise ValueError('Frozen client setup differs from declared output-cap change')
    return manifest


def run(root):
    # Reuse the existing owner-aware finite controller. All validation is this
    # experiment's stricter validator; the 300K validator is never weakened.
    lru.validate = validate
    lru.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--failed-startup',type=Path)
    args=parser.parse_args()
    if args.source:
        prepare(args.source.resolve(),args.output.resolve(),args.validation.resolve(),
                args.failed_startup.resolve() if args.failed_startup else None)
    else:
        run(args.output.resolve())
