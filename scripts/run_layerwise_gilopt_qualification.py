"""Finite fixed-order G4K qualification: baseline A, baseline B, then candidate.

No waiting daemon, retries, external grading or throughput acceptance. The
native eight-request/16424-token batch limits and all request controls stay
unchanged; one client owner at a time removes completion-refill batch changes.
"""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Copied launchers resolve analysis dependencies from their frozen baseline.
FROZEN_SCRIPTS = Path(__file__).resolve().parent/'baseline-a/runtime-v1/scripts'
if FROZEN_SCRIPTS.exists():
    sys.path.append(str(FROZEN_SCRIPTS))

from audit_layerwise_gilopt import (CHANGED, apply_patch, compare, read, save_new,
                                   sha, tree, verify_audit, verify_files)
from run_layerwise_matched_followup import identity


ARMS = ('baseline-a', 'baseline-b', 'optimized')
DRIVER = 'scripts/run_layerwise_memory.py'


def subset(fixture, ids):
    selected = deepcopy(fixture)
    rows = {r['id']: r for r in selected['rows']}
    if len(ids) != len(set(ids)) or any(cid not in rows for cid in ids):
        raise ValueError('Select distinct original case IDs')
    selected.update(rows=[rows[cid] for cid in ids], cases=len(ids), case_ids=list(ids),
                    qualification=True, performance_observation=False)
    bindings = {b['id']: b for b in selected['bindings']}
    selected['bindings'] = [bindings[cid] for cid in ids]
    return selected


def layerwise_events(result, case_dir):
    """Inline events (older runs) or a capture-log reference (benchmark_client_io.result_events)."""
    events = result['layerwise']['events']
    if isinstance(events, list):
        return events
    # Imported here: frozen baselines put older scripts first on the path.
    from benchmark_client_io import result_events
    return result_events(result, case_dir)


def choose_cases(baseline, *, exclude=(), refresh_only=False):
    fixture = read(baseline/'fixture.json')
    # Selection uses execution coverage only, never correctness or judge scores.
    long = []
    for row in fixture['rows']:
        result = read(baseline/'run/answers/cases'/row['id']/'result.json')
        if any(e.get('event_type') == 'decode_pressure'
               for e in layerwise_events(result, baseline/'run/answers/cases'/row['id'])):
            long.append(row['id'])
    if not long:
        raise ValueError('Reference has no accepted-output refresh sample')
    if refresh_only:
        remaining = [cid for cid in long if cid not in set(exclude)]
        if not remaining:
            raise ValueError('No unattempted original refresh cases remain; never resample for coverage')
        return remaining
    ordinary = next(row['id'] for row in fixture['rows'] if row['id'] not in long)
    return [ordinary, long[0]]


def coverage_extension(source, baseline):
    """Add only previously unattempted refresh IDs, not a retry of saved answers."""
    prior = verify(source)
    state = read(source/'status.json')
    if (Path(prior['baseline']) != baseline or state['status'] != 'not_qualified' or
            state['reason'] != 'missing_decode_refresh_coverage' or state['optimized_started']):
        raise ValueError('Coverage extension requires a preserved, stable but uncovered baseline pair')
    pair = compare(source/'baseline-a', source/'baseline-b', prior['case_ids'])
    if not pair['passed']:
        raise ValueError('Preserved baseline stability no longer passes')
    files = {str(source/n): sha(source/n) for n in ('plan.json', 'status.json', 'baseline-b-equivalence.json')}
    for check in pair['checks']:
        files.update(check['source_files'])
    return dict(source=str(source), excluded_case_ids=prior['case_ids'], files=files,
                same_case_retries=0, reason='Actual decode refresh was not covered; add remaining original refresh IDs only.')


def prepare(root, audit_path, previous, pointer, coverage_source=None):
    if root.exists():
        raise ValueError('Qualification needs a fresh directory; preserve previous attempts')
    observed = verify_audit(read(audit_path))
    baseline = Path(observed['baseline'])
    if previous.resolve() != Path(observed['optimized']).resolve():
        raise ValueError('Previous owner must be the audited optimization run')
    if any(root.is_relative_to(Path(observed[key])) for key in ('baseline', 'optimized')):
        raise ValueError('Never prepare inside an observed frozen campaign')
    import run_layerwise_memory as memory
    fixture, profile = memory.verify(baseline)
    if (fixture['suite'] != 'longmemeval_s' or profile['max_num_seqs'] != 8 or
            profile['max_num_batched_tokens'] != 16424 or
            [profile['kvmem'][k] for k in ('select_budget', 'gen_budget', 'block_tokens', 'sink_blocks', 'recent_blocks')]
            != [16384, 4096, 32, 41, 0]):
        raise ValueError('Expected the frozen 16K Active / 4K Gen eight-owner configuration')
    extension = coverage_extension(coverage_source, baseline) if coverage_source else None
    ids = (choose_cases(baseline, exclude=extension['excluded_case_ids'], refresh_only=True)
           if extension else choose_cases(baseline))
    patch = next(Path(p) for p in observed['files'] if p.endswith('/opt.patch'))
    root.mkdir(parents=True)
    shutil.copy2(audit_path, root/'observed-runtime-audit.json')
    shutil.copy2(patch, root/'opt.patch')
    launcher = Path(__file__).resolve()
    shutil.copy2(launcher, root/'launch.py')
    shutil.copy2(launcher.with_name('audit_layerwise_gilopt.py'), root/'audit_layerwise_gilopt.py')
    shutil.copy2(launcher.with_name('run_layerwise_matched_followup.py'), root/'run_layerwise_matched_followup.py')
    if extension:
        save_new(root/'coverage-source.json', extension)
    records = {}
    for name in ARMS:
        arm = root/name
        arm.mkdir()
        runtime = arm/'runtime-v1'
        shutil.copytree(baseline/'runtime-v1', runtime,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if name == 'optimized':
            apply_patch(baseline/'runtime-v1', root/'opt.patch', runtime)
        # Shared driver gains an explicit client-worker argument; inference
        # modules are frozen-baseline bytes, plus only the audited patch.
        shutil.copy2(launcher.with_name('run_layerwise_memory.py'), runtime/DRIVER)
        proof = tree(runtime)
        expected = dict(observed['baseline_runtime'])
        expected[DRIVER] = sha(runtime/DRIVER)
        if name == 'optimized':
            for module in CHANGED:
                expected[module] = observed['optimized_runtime'][module]
        if proof != expected:
            raise ValueError('Qualification changed an undeclared runtime file')
        save_new(arm/'fixture.json', subset(fixture, ids))
        shutil.copy2(baseline/'profile.json', arm/'profile.json')
        save_new(arm/'runtime-proof.json', proof)
        names = ('fixture.json', 'profile.json', 'runtime-proof.json')
        records[name] = {n: sha(arm/n) for n in names}
        save_new(arm/'status.json', dict(status='prepared', expected=len(ids)))
    process = read(previous/'process.json')
    plan = dict(baseline=str(baseline), previous=str(previous), case_ids=ids, arms=records,
        observed_runtime_audit_sha256=sha(root/'observed-runtime-audit.json'),
        files={n: sha(root/n) for n in ('launch.py', 'audit_layerwise_gilopt.py',
                                      'run_layerwise_matched_followup.py', 'opt.patch')},
        previous_processes={key: identity(process[key]) for key in ('pid', 'server_pid')},
        pointer=str(pointer), previous_pointer_sha256=sha(pointer),
        client_workers=1, native_max_num_seqs=8, native_batch_tokens=16424,
        diagnostic_only=True, external_grading=False, automatic_retry=False, scheduled_monitor=False,
        selection_rule='First original non-refresh case and first original decode-refresh case in fixture order; no scores used.',
        acceptance='Baseline A/B exact pressure/Query blocks and full tokens/text, then optimized exact match; '
                   'all arms must naturally exercise decode pressure. No resampling for missing coverage.')
    if extension:
        plan['files']['coverage-source.json'] = sha(root/'coverage-source.json')
        plan['selection_rule'] = 'All remaining original decode-refresh cases, excluding saved prior diagnostic IDs; no scores used.'
    save_new(root/'plan.json', plan)
    save_new(root/'prepared-proof.json', dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json', dict(status='prepared', case_ids=ids, diagnostic_only=True))
    verify(root)
    # No workspace PYTHONPATH fallback: test the copied launcher's complete
    # offline import/verification path before any model process can be started.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-B', str(root/'launch.py'), 'verify', '--root', str(root)],
                   cwd=root, env=env, check=True)
    return plan


def verify(root):
    if sha(root/'plan.json') != read(root/'prepared-proof.json')['plan_sha256']:
        raise ValueError('Qualification plan changed')
    plan = read(root/'plan.json')
    verify_files(root, plan['files'])
    if 'coverage-source.json' in plan['files']:
        extension = read(root/'coverage-source.json')
        verify_files(Path('/'), extension['files'])
        if extension['same_case_retries'] or set(plan['case_ids']) & set(extension['excluded_case_ids']):
            raise ValueError('Coverage extension must not resample previous diagnostic cases')
    if sha(root/'observed-runtime-audit.json') != plan['observed_runtime_audit_sha256']:
        raise ValueError('Observed runtime audit changed')
    audit = verify_audit(read(root/'observed-runtime-audit.json'))
    baseline = Path(plan['baseline'])
    fixture, profile = read(baseline/'fixture.json'), read(baseline/'profile.json')
    expected_fixture = subset(fixture, plan['case_ids'])
    for name, files in plan['arms'].items():
        arm = root/name
        verify_files(arm, files)
        proof = tree(arm/'runtime-v1')
        if proof != read(arm/'runtime-proof.json'):
            raise ValueError('Frozen qualification runtime changed: '+name)
        expected = dict(audit['baseline_runtime'])
        expected[DRIVER] = proof[DRIVER]
        if name == 'optimized':
            for module in CHANGED:
                expected[module] = audit['optimized_runtime'][module]
        if (proof != expected or read(arm/'fixture.json') != expected_fixture or
                read(arm/'profile.json') != profile):
            raise ValueError('Qualification changed frozen controls or inference modules')
    return plan


def handoff_ready(plan, *, gpu_processes):
    previous = Path(plan['previous'])
    for key, expected in plan['previous_processes'].items():
        if expected is not None:
            live = identity(expected['pid'])
            if live is not None:
                if live != expected:
                    raise RuntimeError('Previous PID was reused; inspect ownership before handoff')
                raise RuntimeError('Previous '+key+' is still live; no new GPU work started')
    if read(previous/'status.json')['status'] != 'completed':
        raise RuntimeError('Previous run is not successfully completed; no automatic wait/retry')
    cleanup = read(previous/'run/cleanup.json')
    if cleanup['gpu_processes'] or cleanup['remaining_owned_groups'] or gpu_processes():
        raise RuntimeError('GPU ownership is not released')
    if sha(Path(plan['pointer'])) != plan['previous_pointer_sha256']:
        raise RuntimeError('GPU scheduling pointer changed; leave the other experiment untouched')


def decode_coverage(arm, ids):
    return {cid: sum(e.get('event_type') == 'decode_pressure'
                     for e in layerwise_events(read(arm/'run/answers/cases'/cid/'result.json'),
                                              arm/'run/answers/cases'/cid))
            for cid in ids}


def require_predecessor(root, name):
    if name not in ARMS:
        raise ValueError('Unknown qualification arm')
    if name != 'baseline-a':
        previous = ARMS[ARMS.index(name)-1]
        if read(root/previous/'status.json')['status'] != 'completed':
            raise ValueError('Previous qualification arm is incomplete')
    if name == 'optimized':
        report = read(root/'baseline-b-equivalence.json')
        if not report['passed'] or not all(any(c.values()) for c in report['decode_coverage']):
            raise ValueError('Baseline stability and decode-refresh coverage must pass first')


def run_arm(root, name):
    plan = verify(root)
    require_predecessor(root, name)
    if read(Path(plan['pointer']))['queue_root'] != str(root):
        raise RuntimeError('This qualification does not own GPU scheduling')
    arm = root/name
    sys.path[:0] = [str(arm/'runtime-v1/src'), str(arm/'runtime-v1/scripts')]
    import run_layerwise_memory as memory
    # This is a strict, hash-bound verifier for a deliberately selected cohort,
    # not the old run_g4k.py read-only replacement for memory.verify.
    def verified(path):
        verify(root)
        if path.resolve() != arm.resolve():
            raise ValueError('Wrong qualification arm')
        return read(arm/'fixture.json'), read(arm/'profile.json')
    memory.verify = verified
    memory.run(arm, workers=plan['client_workers'])


def run(root):
    from run_recall_comparison import gpu_lock, termination_guard, stop_owned
    import benchmark_block_graph as lifecycle
    import recall_comparison_protocol as p
    plan = verify(root)
    if (root/'process.json').exists():
        raise ValueError('Preserve the first qualification attempt; no automatic retry')
    with termination_guard(), (root/'queue.lock').open('a+') as queue:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
            handoff_ready(plan, gpu_processes=lifecycle.gpu_processes)
            save_new(root/'process.json', dict(pid=os.getpid(), at=time.time()))
            p.save(Path(plan['pointer']), dict(queue_root=str(root), root=str(root),
                process=str(root/'process.json'), status=str(root/'status.json'),
                experiment_type='fixed-order-g4k-equivalence', at=time.time()))
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
        try:
            for name in ARMS:
                verify(root)
                if read(Path(plan['pointer']))['queue_root'] != str(root):
                    raise RuntimeError('Another owner acquired GPU scheduling')
                p.save(root/'status.json', dict(status='running', arm=name, at=time.time(), diagnostic_only=True))
                command = [sys.executable, '-B', str(root/'launch.py'), 'arm', '--root', str(root), '--arm', name]
                with (root/(name+'.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                p.save(root/'child.json', dict(pid=child.pid, arm=name, command=command, at=time.time()))
                try:
                    code = child.wait()
                except BaseException:
                    stop_owned(child)
                    raise
                if code:
                    raise RuntimeError('Technical failure; preserve first answers: '+name)
                cleanup = read(root/name/'run/cleanup.json')
                if cleanup['gpu_processes'] or cleanup['remaining_owned_groups'] or lifecycle.gpu_processes():
                    raise RuntimeError('Arm has not released its GPU')
                if name != 'baseline-a':
                    report = compare(root/'baseline-a', root/name, plan['case_ids'])
                    report['decode_coverage'] = [decode_coverage(root/a, plan['case_ids']) for a in ('baseline-a', name)]
                    save_new(root/(name+'-equivalence.json'), report)
                    if not report['passed'] or not all(any(c.values()) for c in report['decode_coverage']):
                        p.save(root/'status.json', dict(status='not_qualified', arm=name,
                            reason=('missing_decode_refresh_coverage' if report['passed'] else
                                    'baseline_instability' if name == 'baseline-b' else 'candidate_divergence'),
                            equivalence_passed=report['passed'], decode_coverage=report['decode_coverage'],
                            optimized_started=name == 'optimized', diagnostic_only=True, at=time.time()))
                        return
            verify(root)
            p.save(root/'status.json', dict(status='completed', exact_equivalence=True,
                diagnostic_only=True, performance_accepted=False, at=time.time()))
        except BaseException as error:
            p.save(root/'status.json', dict(status='failed', error=repr(error), diagnostic_only=True, at=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'run', 'arm'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--pointer', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--coverage-source', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        plan = prepare(root, args.audit.resolve(), args.previous.resolve(), args.pointer.resolve(),
                       args.coverage_source.resolve() if args.coverage_source else None)
        print(plan['case_ids'])
    elif args.action == 'arm':
        run_arm(root, args.arm)
    else:
        globals()[args.action](root)


if __name__ == '__main__':
    main()
