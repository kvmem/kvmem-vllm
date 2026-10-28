"""Run the untouched candidate after preserving a baseline resource-limit failure.

This does not resume, retry, or relabel the failed baseline. Its incomplete
cohort remains an explicit outcome, so the resulting queue cannot claim a fully
passed A/B test. The model, prompts, sampling, and turn limits are unchanged.
"""
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from scripts.run_agent_handoff_queue import (
    arm, gpu_pids, identity, live, read, require_idle_dsh, save, sha,
    verify_frozen, wait_dependency,
)


def validate_baseline(source, probe=live):
    plan = verify_frozen(source)
    if read(source/'status.json').get('status') != 'failed':
        raise ValueError('Source queue is not failed')
    if probe(read(source/'queue-process.json')) is not None:
        raise ValueError('Source queue is still alive')
    baseline = source/'baseline'
    cleanup = read(baseline/'cleanup.json')
    if (cleanup.get('remaining_processes') != [] or
            cleanup.get('remaining_owned_gpu_pids') != [] or
            any(probe(row) is not None for row in cleanup.get('owned', []))):
        raise ValueError('Source cleanup is not complete')
    audit = read(baseline/'execution-audit.json')
    if (audit.get('status') != 'failed' or set(audit.get('errors', [])) !=
            {'Incomplete/context cohort', 'Missing followups'}):
        raise ValueError('Unexpected baseline audit failure')
    tasks = read(baseline/'task-plan.json')
    if (len(tasks['cases']) != 8 or tasks['max_steps_per_turn'] != 240 or
            tasks['turn_timeout_seconds'] != 3600 or tasks['minimum_turns'] != 4):
        raise ValueError('Original task limits changed')
    failed = []
    for row in tasks['cases']:
        case = baseline/'cases'/row['id']
        state = read(case/'status.json')
        if (case/'failure.json').exists():
            failed.append(row['id'])
        elif state.get('status') != 'completed' or state.get('turns_completed', 0) < 4:
            raise ValueError('Unexpected incomplete task')
    if failed != ['r01-fps']:
        raise ValueError('Only the audited FPS limit is eligible')
    case = baseline/'cases/r01-fps'
    if read(case/'failure.json').get('error') != "RuntimeError('Predeclared turn resource limit; original attempt retained')":
        raise ValueError('Failure is not a predeclared resource limit')
    limit = read(case/'turn-04/limit.json')
    if limit['steps'] != 240 or not 0 < limit['elapsed'] < 3600:
        raise ValueError('Limit evidence changed')
    if read(case/'status.json').get('turns_completed') != 3:
        raise ValueError('Baseline followup count changed')
    evidence = read(baseline/'evidence-hashes.json')
    if not evidence or any(sha(baseline/rel) != expected for rel, expected in evidence.items()):
        raise ValueError('Baseline response evidence changed')
    if read(baseline/'browser-summary.json').get('checked') != 7:
        raise ValueError('Completed-project browser checks missing')
    candidate = source/'candidate'
    if read(candidate/'status.json').get('status') != 'prepared' or any(
            (candidate/p).exists() for p in ('started.json', 'cases', 'bridge', 'server', 'workspaces')):
        raise ValueError('Candidate has already been attempted')
    return plan


def verify_preserved_baseline(root):
    plan = read(root/'manifest.json')
    binding = plan['preserved_baseline']
    proof = root/binding['proof']
    if sha(proof) != binding['sha256']:
        raise ValueError('Baseline outcome proof changed')
    outcome = read(proof)
    if (outcome.get('status') != 'incomplete_predeclared_limit' or
            outcome.get('failed_cases') != ['r01-fps'] or
            outcome.get('completed_tasks') != 7 or outcome.get('retry_allowed') is not False):
        raise ValueError('Baseline failure cannot be relabeled as successful')
    source = Path(outcome['source'])
    if (root/'baseline').resolve() != (source/'baseline').resolve():
        raise ValueError('Baseline reference changed')
    for rel, expected in outcome['hashes'].items():
        if sha(source/rel) != expected:
            raise ValueError('Preserved baseline audit changed: '+rel)
    evidence = read(source/'baseline/evidence-hashes.json')
    if any(sha(source/'baseline'/rel) != expected for rel, expected in evidence.items()):
        raise ValueError('Preserved baseline answers changed')
    return outcome


def prepare(source, destination, repository, probe=live):
    source = source.resolve(); destination = destination.resolve()
    plan = validate_baseline(source, probe)
    if destination.exists():
        raise ValueError('Destination already exists')
    destination.mkdir()
    (destination/'baseline').symlink_to(source/'baseline', target_is_directory=True)
    for relative in plan['files']:
        if relative.startswith('baseline/'):
            continue
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    (destination/'qualification').symlink_to(source/'qualification', target_is_directory=True)
    shutil.copy2(source/'gpu-component.log', destination/'gpu-component.log')
    tasks = read(source/'candidate/task-plan.json')
    for row in tasks['cases']:
        row['workspace'] = str(destination/Path(row['workspace']).relative_to(source))
    save(destination/'candidate/task-plan.json', tasks)
    save(destination/'candidate/status.json', dict(status='prepared'))
    changed = ['candidate/task-plan.json']
    for relative in ('scripts/continue_agent_handoff_candidate.py',
                     'tests/test_agent_handoff_candidate_continuation.py'):
        target = destination/'candidate/runtime'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository/relative, target)
        changed.append(str(target.relative_to(destination)))
    proof_files = ['manifest.json', 'status.json', 'queue-process.json'] + [
        'baseline/'+p for p in ('execution-audit.json', 'cleanup.json', 'evidence-hashes.json',
                               'browser-summary.json', 'performance.json', 'task-plan.json',
                               'cases/r01-fps/failure.json', 'cases/r01-fps/turn-04/limit.json')]
    save(destination/'baseline-outcome.json', dict(
        status='incomplete_predeclared_limit', source=str(source), at=time.time(),
        completed_tasks=7, expected_tasks=8, completed_human_turns=31,
        failed_cases=['r01-fps'], retry_allowed=False, engine_failure_on_this_attempt=False,
        hashes={rel:sha(source/rel) for rel in proof_files},
        limitation='Baseline had an engine interruption and a final resource-limit failure. '
                   'Do not report this as a passed continuous eight-task A/B comparison.'))
    changed.append('baseline-outcome.json')
    plan['preserved_baseline'] = dict(proof='baseline-outcome.json',
                                     sha256=sha(destination/'baseline-outcome.json'))
    plan['execution_arms'] = ['candidate']
    plan['order'] = ['reuse byte-verified qualification', 'preserve incomplete baseline',
                     'candidate eight-agent suite', 'optimization review still required']
    plan['protocol'] = str(destination/'protocol.so')
    plan.pop('resume_arms', None)
    plan['files'] = {p:sha(destination/p) for p in sorted(set(plan['files']) | set(changed))}
    save(destination/'manifest.json', plan)
    save(destination/'status.json', dict(status='prepared', at=time.time()))
    verify_frozen(destination); verify_preserved_baseline(destination)
    return dict(root=str(destination), manifest_sha256=sha(destination/'manifest.json'))


def run(root):
    plan = verify_frozen(root)
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'gpu-started.json').exists():
            raise RuntimeError('Never restart an attempted candidate')
        save(root/'queue-process.json', dict(**identity(psutil.Process()), at=time.time(),
                                            manifest_sha256=sha(root/'manifest.json')))
        try:
            if plan.get('execution_arms') != ['candidate']:
                raise ValueError('This continuation never reruns the baseline')
            verify_preserved_baseline(root)
            subprocess.run([plan['browser_python'],
                str(root/'candidate/runtime/scripts/check_dsh_daily_browser.py'),
                '--root', str(root/'candidate'), '--probe'], check=True, timeout=60)
            wait_dependency(root, plan); verify_frozen(root)
            if gpu_pids():
                raise RuntimeError('GPU ownership changed before candidate startup')
            proof = plan['reuse_qualification']; prior = Path(proof['root'])
            if any(sha(prior/p) != expected for p, expected in proof['hashes'].items()):
                raise ValueError('Qualification proof changed')
            if read(root/'qualification/execution-audit.json').get('status') != 'passed':
                raise ValueError('Qualification not passed')
            require_idle_dsh(plan)
            save(root/'gpu-started.json', dict(at=time.time()))
            save(root/'status.json', dict(status='running', active='candidate', at=time.time()))
            arm(root, 'candidate', plan)
            verify_preserved_baseline(root)
            save(root/'comparison.json', {name:read(root/name/'performance.json')
                                         for name in ('baseline', 'candidate')})
            save(root/'status.json', dict(status='needs_optimization_review', at=time.time(),
                baseline_outcome='incomplete_predeclared_limit', candidate_outcome='completed',
                full_paired_test_passed=False))
        except BaseException as error:
            save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.root:
        run(args.root.resolve())
    else:
        if not args.source or not args.destination:
            parser.error('Preparation requires --source and --destination')
        print(prepare(args.source, args.destination, args.repository))
