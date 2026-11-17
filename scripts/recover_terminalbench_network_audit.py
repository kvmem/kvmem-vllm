"""Reconcile one pinned post-scoring Docker audit incident, then run untouched LRU.

Never retry an Agent, verifier, model request, or attempted suite. The original
controller must finish naturally. Original errors/results/events remain intact;
only the previously absent derived summary and a new controller are created.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

TRIAL = 'count-dataset-tokens__LFKAe6y'
CID = '80ea0d8d2669af55e499fc6335b6c1f98e5a0e0f2602e2a541a1f0abf6901b96'
NETWORK = '0b4970965b41acc28ab4af8e98c7d0fe0ace00219f3717820b0f74dca27d45b8'
ERROR = f"TimeoutExpired(['docker', 'diff', '{CID}'], 60)"
ALLOWED_ERROR = dict(trial=TRIAL, error=ERROR)
CHECKED_SHA = '903f63ed33b7032fd5019c8954492f1f0d379e739c145448a2ee41ef0e0f559e'
TASKS = 89
PYTHON = sys.executable


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(8*1024**2):
            h.update(part)
    return h.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    else:
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
        temporary.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def confined(root, relative):
    relative = Path(relative)
    path = root/relative
    require(not relative.is_absolute() and path.resolve().is_relative_to(root.resolve())
            and not path.is_symlink(), 'Evidence path leaves its experiment')
    return path


def process_alive(record):
    require(type(record.get('pid')) is int and type(record.get('created')) in (int, float),
            'Missing PID/creation-time identity')
    try:
        process = psutil.Process(record['pid'])
        return (process.create_time() == record['created'] and
                process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD))
    except psutil.NoSuchProcess:
        return False


def events(arm):
    return [json.loads(line) for line in (arm/'formal-events/events.jsonl').read_text().splitlines() if line]


def classify(raw):
    reward = (raw.get('verifier_result') or {}).get('rewards')
    require(isinstance(reward, dict) and type(reward.get('reward')) in (int, float)
            and reward['reward'] in (0, 1) and raw.get('exception_info') is None,
            'Every original attempt must have an ordinary binary score and no exception')
    return dict(task=raw['task_name'], trial=raw['trial_name'], rewards=reward,
                exception=None, status='scored', passed=reward['reward'] == 1)


def unique(rows, key):
    values = {row[key]: row for row in rows}
    require(len(values) == len(rows), 'Duplicate '+key)
    return values


def saved_config_matches(saved, expanded):
    """Harbor's config.json omits defaults; raw result config includes them."""
    if isinstance(saved, dict):
        return isinstance(expanded, dict) and all(
            key in expanded and saved_config_matches(value, expanded[key]) for key, value in saved.items())
    if isinstance(saved, list):
        return (isinstance(expanded, list) and len(saved) == len(expanded) and
                all(saved_config_matches(a, b) for a, b in zip(saved, expanded)))
    return type(saved) is type(expanded) and saved == expanded


def validate_network_receipt(arm, error, lifecycle):
    require(error == ALLOWED_ERROR, 'Unknown network error; no blanket exception is allowed')
    relative = f'formal-events/network-cleanup/{TRIAL}/{NETWORK}.json'
    checked_path = confined(arm, relative)
    checked = read(checked_path)
    require(sha(checked_path) == CHECKED_SHA, 'Pinned original checked evidence changed')
    proof_path = arm/'network-audit-recovery'/TRIAL/'result.json'
    proof = read(proof_path)
    require(proof.get('kind') == 'post-removal-filesystem-audit-only' and
            all(proof.get(key) is True for key in
                ('passed', 'network_absent', 'original_error_retained', 'no_task_execution', 'no_score_changes')),
            'Missing successful read-only post-removal reconciliation')
    require(proof.get('original_checked_file') == relative and proof.get('original_checked_sha256') == CHECKED_SHA,
            'Receipt does not bind the original checked evidence')
    project = TRIAL.lower()+'__env'
    network = checked.get('network', {})
    require(checked.get('status') == 'checked' and checked.get('project') == project
            and network.get('Id') == NETWORK and network.get('Containers') == {}
            and (network.get('Labels') or {}).get('com.docker.compose.project') == project,
            'Original audit did not identify the stopped trial empty network')
    expected_env = arm/'dataset/tasks/count-dataset-tokens/environment'
    require(Path(checked.get('environment_dir', '')).resolve() == expected_env.resolve(),
            'Network environment identity changed')
    require(set(checked.get('retained_containers', {})) == set(proof.get('containers', {})) == {CID},
            'Container cohort differs from the pinned incident')
    before, after = checked['retained_containers'][CID], proof['containers'][CID]
    state = after.get('state') or {}
    require(state.get('Status') == before['state'] == 'exited' and
            all(state.get(key) is False for key in ('Running', 'Paused', 'Restarting', 'OOMKilled'))
            and after.get('image') == before['image'] and after.get('matches_original') is True,
            'Retained stopped container identity/state differs')
    diff = proof_path.parent/(CID+'.diff.txt')
    diff_hash = hashlib.sha256('\n'.join(sorted(diff.read_text().splitlines())).encode()).hexdigest()
    require(diff_hash == before['diff_sha256'] == after.get('diff_sha256'),
            'Docker changed-path audit differs')
    failures = [row for row in lifecycle if row.get('event') == 'network_reclamation_failed']
    require(len(failures) == 1 and {key: failures[0].get(key) for key in ('trial', 'error')} == error,
            'Network error event is absent, duplicated, or different')
    ends = [row for row in lifecycle if row.get('trial') == TRIAL and row.get('event') == 'end']
    require(len(ends) == 1 and ends[0]['at'] <= checked['at'] <= failures[0]['at']
            <= proof['started_at'] <= proof['finished_at'], 'Post-scoring audit time order differs')
    paths = (checked_path, proof_path, diff)
    return dict(passed=True, incident=error, original_checked=checked, read_only_receipt=proof,
                source_hashes={str(path.relative_to(arm)): sha(path) for path in paths},
                scope='Changed-path set and container identity match; this is not a full file-content digest.')


def validate_results(arm):
    manifest = read(arm/'manifest.json'); status = read(arm/'formal-events/status.json')
    names = manifest['launched_task_names']
    require(manifest.get('tasks') == len(names) == len(set(names)) == TASKS, 'Need the original full 89-task cohort')
    require(not manifest.get('preserved_scored') and not (arm/'preserved-outcomes.json').exists(),
            'Inherited scores are not allowed')
    job = read(arm/'job.json')
    require(job.get('n_attempts') == 1 and job.get('retry', {}).get('max_retries') == 0,
            'Original job must allow exactly one attempt')
    require(status.get('expected') == status.get('completed') == status.get('released') == TASKS
            and status.get('active') == [] and status.get('release_errors') == []
            and status.get('verifier_cache_errors') == [] and status.get('network_errors') == [ALLOWED_ERROR],
            'Original job is incomplete or has a nonallowlisted error')
    cohort = unique(status['outcomes'], 'trial')
    require(len(cohort) == TASKS and {r['task'].removeprefix('terminal-bench/') for r in cohort.values()} == set(names),
            'Original result cohort differs')
    harbor = read(arm/'formal-events/harbor-result.json')
    stats = harbor.get('stats', {})
    require(harbor.get('finished_at') and harbor.get('n_total_trials') == stats.get('n_completed_trials') == TASKS
            and all(stats.get(key) == 0 for key in
                    ('n_errored_trials', 'n_running_trials', 'n_pending_trials', 'n_cancelled_trials', 'n_retries')),
            'Harbor did not finish all original first attempts normally')
    upstream = unique(harbor['trial_results'], 'trial_name')
    require(set(upstream) == set(cohort), 'Harbor trial cohort differs')
    native_job = read(arm/'jobs/formal/result.json')
    require(native_job == {k: v for k, v in harbor.items() if k != 'trial_results'},
            'Harbor aggregate files disagree')
    actual = {p.parent.name for p in (arm/'jobs/formal').glob('*/config.json')}
    require(actual == set(cohort), 'Extra, missing, or repeated attempted trial')
    result_paths = {p.parent.name for p in (arm/'jobs/formal').glob('*/result.json')}
    require(result_paths == set(cohort), 'Extra or missing raw trial result')
    sources = ['manifest.json', 'job.json', 'formal-events/status.json', 'formal-events/events.jsonl',
               'formal-events/harbor-result.json', 'jobs/formal/result.json', 'jobs/formal/config.json']
    lifecycle = events(arm)
    for kind in ('start', 'agent-start', 'agent-end', 'verification-start', 'end', 'kv_released'):
        selected = [row for row in lifecycle if row.get('event') == kind]
        require(Counter(row.get('trial') for row in selected) == Counter({trial: 1 for trial in cohort}),
                'Missing or repeated '+kind+' event')
        if kind == 'kv_released':
            require(all(row.get('released') is True for row in selected), 'Missing native KV release acknowledgment')
    for trial, expected in cohort.items():
        raw_path = f'jobs/formal/{trial}/result.json'; event_path = f'formal-events/{trial}.json'
        raw = read(confined(arm, raw_path)); copied = read(confined(arm, event_path))
        require(raw == copied == upstream[trial], 'Original raw/Harbor/event result differs: '+trial)
        require(classify(raw) == expected, 'Status classification differs from original score: '+trial)
        config_path = f'jobs/formal/{trial}/config.json'
        config = read(confined(arm, config_path))
        require(config.get('trial_name') == trial and Path(config['task']['path']).name ==
                expected['task'].removeprefix('terminal-bench/'), 'Trial config identity differs')
        require(saved_config_matches(config, raw.get('config')), 'Saved trial config differs from raw result config')
        sources.extend((raw_path, event_path, config_path))
    passed = sum(row['passed'] for row in cohort.values())
    require(status.get('passed') == passed, 'Status pass count differs')
    receipt = validate_network_receipt(arm, status['network_errors'][0], lifecycle)
    summary = dict(expected=TASKS, outcomes=list(cohort.values()), scored=TASKS, passed=passed,
                   task_timeouts=0, errors=0, retries=0)
    bindings = {name: sha(arm/name) for name in sources}
    bindings.update(receipt['source_hashes'])
    return dict(passed=True, summary=summary, network_reconciliation=receipt, source_hashes=bindings)


def immutable_inputs(pair):
    paths = ['run_pair.py', 'compare_terminalbench_policies.py', 'prelaunch-audit.json']
    for arm in ('thinking', 'lru'):
        paths.extend(arm+'/'+name for name in ('manifest.json', 'profile.json', 'job.json', 'agent-time-policy.json'))
    paths.extend(['thinking/runtime/scripts/finalize_terminalbench.py',
                  f'thinking/formal-events/network-cleanup/{TRIAL}/{NETWORK}.json',
                  f'thinking/network-audit-recovery/{TRIAL}/result.json',
                  f'thinking/network-audit-recovery/{TRIAL}/{CID}.diff.txt',
                  f'thinking/jobs/formal/{TRIAL}/result.json', f'thinking/formal-events/{TRIAL}.json'])
    return {name: sha(confined(pair, name)) for name in paths}


def unattempted_lru(pair):
    lru = pair/'lru'
    require(not any((lru/name).exists() for name in ('started', 'queue-process.json', 'formal-process.json',
                                                    'smoke-process.json', 'formal-events', 'jobs', 'model')),
            'LRU has already been attempted; never relaunch it')


def prepare(pair, recovery):
    pair, recovery = pair.resolve(), recovery.resolve()
    require(not recovery.exists(), 'Use a new recovery directory')
    require(read(pair/'pair-status.json').get('status') == 'running', 'Prepare while the original pair is running')
    status = read(pair/'thinking/formal-events/status.json')
    require(status.get('network_errors') == [ALLOWED_ERROR], 'Only the pinned incident may be reconciled')
    receipt = validate_network_receipt(pair/'thinking', ALLOWED_ERROR, events(pair/'thinking'))
    unattempted_lru(pair)
    plan = dict(schema=1, pair=str(pair), created_at=time.time(), expires_at=time.time()+7*86400,
                allowed_error=ALLOWED_ERROR, original_pair=read(pair/'pair-process.json'),
                original_suite=read(pair/'thinking/queue-process.json'),
                immutable_inputs=immutable_inputs(pair), receipt=receipt,
                task_retries_allowed=False, verifier_retries_allowed=False, expected_tasks=TASKS)
    recovery.mkdir(parents=True)
    save(recovery/'plan.json', plan, exclusive=True)
    return plan


def audit(pair, recovery, plan, live_probe=process_alive):
    require(str(pair.resolve()) == plan.get('pair') and plan.get('allowed_error') == ALLOWED_ERROR,
            'Recovery plan identifies another campaign/incident')
    for name, expected in plan['immutable_inputs'].items():
        require(sha(confined(pair, name)) == expected, 'Pinned recovery input changed: '+name)
    require(read(pair/'pair-process.json') == plan['original_pair'] and
            read(pair/'thinking/queue-process.json') == plan['original_suite'],
            'Original controller identity records changed')
    for owner in (plan['original_pair'], plan['original_suite']):
        require(not live_probe(owner), 'Original controller/suite must finish naturally first')
    arm = pair/'thinking'
    for name in ('queue-process.json', 'formal-process.json', 'model/server-process.json',
                 'bridge-process.json', 'gpu-monitor-process.json'):
        require(not live_probe(read(arm/name)), 'An original owned process remains alive: '+name)
    cleanup = read(arm/'cleanup.json')
    require(cleanup.get('remaining_processes') == cleanup.get('remaining_owned_gpu_pids') == [],
            'Original cleanup is incomplete')
    require(not any(live_probe(record) for record in cleanup.get('owned', [])), 'Original descendant remains alive')
    require(read(pair/'pair-status.json').get('status') == 'failed' and
            read(pair/'pair-status.json').get('error') ==
            "RuntimeError('thinking suite exited 1; preserve all attempts, no task retries')",
            'Original pair did not end at the expected post-job guard')
    expected = "RuntimeError('formal exited 1; preserve logs, no retry')"
    require(read(arm/'status.json').get('status') == 'failed' and read(arm/'status.json').get('error') == expected
            and read(arm/'failure.json').get('error') == expected,
            'Original suite failed for another reason')
    unattempted_lru(pair)
    require(not (arm/'formal-events/summary.json').exists(), 'A summary already exists; never replace it')
    result = validate_results(arm)
    result.update(pair=str(pair.resolve()), checked_at=time.time(),
                  plan_sha256=sha(recovery/'plan.json'), original_job_failure_retained=True,
                  scores_reused_from_other_campaigns=False, no_task_or_verifier_execution=True)
    return result


def runtime_env(runtime):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(map(str, (runtime/'src', runtime/'scripts', runtime)))
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    return env


def call_frozen_finalizer(arm, recovery):
    runtime = arm/'runtime'
    expected = read(arm/'manifest.json')['frozen']['runtime']['scripts/finalize_terminalbench.py']
    require(sha(runtime/'scripts/finalize_terminalbench.py') == expected, 'Frozen finalizer source changed')
    # Its CLI only runs needs_final_review; call review directly after our audit.
    command = [PYTHON, '-B', '-c',
               'from pathlib import Path; import sys; from finalize_terminalbench import review; review(Path(sys.argv[1]))', str(arm)]
    with (recovery/'finalizer.log').open('xb') as log:
        subprocess.run(command, cwd=runtime, env=runtime_env(runtime), stdout=log, stderr=subprocess.STDOUT, check=True)


def recover(pair, recovery, plan, finalizer=call_frozen_finalizer, live_probe=process_alive):
    result = audit(pair, recovery, plan, live_probe)
    original = recovery/'original'
    original.mkdir(exist_ok=False)
    files = ['pair-status.json', 'pair-process.json', 'pair-child.json', 'thinking/status.json',
             'thinking/failure.json', 'thinking/cleanup.json', 'thinking/formal-events/status.json',
             'thinking/formal-events/events.jsonl', 'thinking/formal-events/harbor-result.json']
    for name in files:
        target = original/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(confined(pair, name), target)
    originals = {name: sha(original/name) for name in files}
    result['archived_originals'] = originals
    save(recovery/'audit.json', result, exclusive=True)
    arm = pair/'thinking'
    receipt = arm/'formal-events/network-audit-reconciliation.json'
    save(receipt, dict(**result, recovery_directory=str(recovery.resolve()),
                       audit_sha256=sha(recovery/'audit.json')), exclusive=True)
    summary = dict(result['summary'], derived=dict(kind='post-scoring-network-audit-reconciliation',
        original_failure_preserved=True, original_network_errors_preserved=True,
        receipt=str(receipt.relative_to(arm)), receipt_sha256=sha(receipt)))
    save(arm/'formal-events/summary.json', summary, exclusive=True)
    finalizer(arm, recovery)
    # Finalization must not mutate raw evidence or erase historical errors.
    for name, expected in result['source_hashes'].items():
        require(sha(arm/name) == expected, 'Finalizer changed original evidence: '+name)
    require(read(arm/'status.json').get('status') == 'completed', 'Frozen final review did not complete')
    save(recovery/'recovered.json', dict(at=time.time(), passed=True, audit_sha256=sha(recovery/'audit.json'),
        derived_summary_sha256=sha(arm/'formal-events/summary.json'), original_files_preserved=True), exclusive=True)
    return result


def continue_pair(pair, recovery, plan):
    recover(pair, recovery, plan)
    original_pair = read(recovery/'original/pair-status.json')
    original_started = original_pair['started_at']
    unattempted_lru(pair)
    owner = psutil.Process()
    metadata = dict(kind='post-scoring-network-audit-continuation', directory=str(recovery.resolve()),
                    original_failure=str(recovery/'original/pair-status.json'),
                    audit_sha256=sha(recovery/'audit.json'), recovered_arm='thinking')
    save(pair/'pair-process.json', dict(pid=owner.pid, created=owner.create_time(), argv=owner.cmdline(),
                                       cwd=owner.cwd(), at=time.time(), recovery=metadata))
    save(pair/'pair-status.json', dict(status='running', arm='lru', started_at=original_started,
        at=time.time(), recovery_started_at=time.time(), recovery=metadata))
    save(pair/'thinking-completed.json', dict(at=time.time(), status=read(pair/'thinking/status.json'), recovery=metadata), exclusive=True)
    lru = pair/'lru'; runtime = lru/'runtime'
    with (pair/'lru-driver.log').open('xb') as log:
        process = subprocess.Popen([PYTHON, '-B', str(runtime/'scripts/run_terminalbench_suite.py'), 'run', str(lru)],
            cwd=runtime, env=runtime_env(runtime), stdout=log, stderr=subprocess.STDOUT)
        child = dict(arm='lru', pid=process.pid, created=psutil.Process(process.pid).create_time(), at=time.time(), recovery=metadata)
        save(recovery/'lru-launched.json', child, exclusive=True)
        save(pair/'pair-child.json', child)
        code = process.wait()
    require(code == 0 and read(lru/'status.json').get('status') == 'completed',
            'Untouched LRU ended unsuccessfully; preserve attempts, no automatic retry')
    save(pair/'lru-completed.json', dict(at=time.time(), status=read(lru/'status.json'), recovery=metadata), exclusive=True)
    require(sha(pair/'compare_terminalbench_policies.py') == plan['immutable_inputs']['compare_terminalbench_policies.py'],
            'Original paired comparator changed')
    with (pair/'comparison.log').open('xb') as log:
        subprocess.run([PYTHON, str(pair/'compare_terminalbench_policies.py'), '--baseline', str(lru),
            '--candidate', str(pair/'thinking'), '--output', str(pair/'accuracy-comparison.json')],
            check=True, stdout=log, stderr=subprocess.STDOUT)
    save(pair/'pair-status.json', dict(status='completed', at=time.time(), started_at=original_started,
        seconds=time.time()-original_started, result=str(pair/'accuracy-comparison.json'), recovery=metadata))


def run(pair, recovery):
    plan = read(recovery/'plan.json')
    with (recovery/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (recovery/'run-started.json').exists(), 'Continuation already attempted; manual audit required')
        require(time.time() < plan['expires_at'], 'Finite recovery plan already expired')
        save(recovery/'run-started.json', dict(at=time.time(), pid=os.getpid()), exclusive=True)
        try:
            while process_alive(plan['original_pair']) or process_alive(plan['original_suite']):
                require(time.time() < plan['expires_at'], 'Finite recovery wait expired; original work left intact')
                save(recovery/'status.json', dict(status='waiting_for_original_completion', at=time.time(),
                                                 no_task_cancellation=True))
                time.sleep(30)
            require(time.time() < plan['expires_at'], 'Finite recovery wait expired; no continuation started')
            continue_pair(pair, recovery, plan)
            save(recovery/'status.json', dict(status='completed', at=time.time()))
        except BaseException as error:
            save(recovery/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            # Never overwrite the old pair's failure until we have taken over.
            record = read(pair/'pair-process.json')
            if record.get('pid') == os.getpid() and record.get('recovery'):
                previous = read(pair/'pair-status.json')
                save(pair/'pair-status.json', dict(status='failed', error=repr(error), at=time.time(),
                    started_at=previous.get('started_at'), recovery=record['recovery']))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'audit', 'run'))
    parser.add_argument('--pair-root', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    args = parser.parse_args(); pair = args.pair_root.resolve(); recovery = args.recovery.resolve()
    if args.action == 'prepare':
        plan = prepare(pair, recovery)
        print(json.dumps(dict(prepared=True, plan=str(recovery/'plan.json'), expires_at=plan['expires_at'])))
    elif args.action == 'audit':
        print(json.dumps(audit(pair, recovery, read(recovery/'plan.json')), indent=2))
    else:
        run(pair, recovery)
