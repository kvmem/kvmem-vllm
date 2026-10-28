"""Finite fresh eight-task DSH comparison of the qualified restore fastpath.

Both arms share model/bridge code and existing optimizations. Only the fastpath
flag changes. No old sessions or failed answers are resumed, and no timer runs.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from scripts import run_agent_handoff_queue as q
from scripts.benchmark_restore_handoff_fastpath import validate as validate_component

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_FILES = ('scripts/benchmark_agent_fastpath_dsh.py',
                    'scripts/analyze_agent_fastpath_dsh.py',
                    'scripts/run_agent_handoff_queue.py',
                    'tests/test_agent_fastpath_dsh.py')


def task_plan(original, output, arm):
    plan = copy.deepcopy(original)
    if len(plan['cases']) != 8 or plan['minimum_turns'] != 4 or plan['max_steps_per_turn'] != 240:
        raise ValueError('Unexpected original task limits')
    if 'resume_cases' in plan:
        raise ValueError('Fresh tasks must not resume original sessions')
    for index, case in enumerate(plan['cases']):
        case['session_id'] = f'{output.name}-{arm}-{case["task"]}'
        case['workspace'] = str(output/arm/'workspaces'/case['id'])
        case['preview_port'] = 19600 + (10 if arm == 'candidate' else 0) + index
    return plan


def outcome(directory):
    """Keep declared task limits as outcomes, but never ignore engine faults."""
    audit = q.read(directory/'execution-audit.json')
    cleanup = q.read(directory/'cleanup.json')
    if any(cleanup[k] for k in ('remaining_processes','remaining_owned_gpu_pids')):
        raise ValueError('Arm cleanup incomplete')
    hashes = q.read(directory/'evidence-hashes.json')
    if not hashes or any(q.sha(directory/rel) != value for rel,value in hashes.items()):
        raise ValueError('Answer evidence changed')
    if audit['status'] == 'passed':
        return dict(status='completed', full_cohort_passed=True,
                    counts=audit['counts'], retry_allowed=False)
    if not set(audit['errors']).issubset({'Incomplete/context cohort','Missing followups'}):
        raise ValueError('Unexpected execution failure')
    failures = []
    plan = q.read(directory/'task-plan.json')
    for case in plan['cases']:
        path = directory/'cases'/case['id']
        failure = path/'failure.json'
        if not failure.exists():
            if q.read(path/'status.json')['status'] != 'completed':
                raise ValueError('Unclassified unfinished task')
            continue
        if q.read(failure).get('error') != "RuntimeError('Predeclared turn resource limit; original attempt retained')":
            raise ValueError('Unexpected task failure')
        limits = [q.read(p) for p in path.glob('turn-*/limit.json')]
        if len(limits) != 1 or not (limits[0]['steps'] >= plan['max_steps_per_turn'] or
                                   limits[0]['elapsed'] >= plan['turn_timeout_seconds']):
            raise ValueError('Missing original limit evidence')
        failures.append(case['id'])
    if not failures or audit['counts'].get('bridge_errors',0):
        raise ValueError('Limit outcome mixed with an infrastructure failure')
    return dict(status='completed_with_resource_limits',full_cohort_passed=False,
                failed_tasks=failures,counts=audit['counts'],retry_allowed=False)


def validate(root):
    plan = q.verify_frozen(root)
    source, tasks_source = Path(plan['source']), Path(plan['tasks_source'])
    for rel, digest in plan['component_proof'].items():
        if q.sha(source/rel) != digest:
            raise ValueError('Qualification/component evidence changed')
    if q.sha(tasks_source/'candidate/task-plan.json') != plan['original_tasks_sha256']:
        raise ValueError('Original task fixture changed')
    for rel, digest in plan['qualified_model_hashes'].items():
        if q.sha(root/'runtime'/rel) != digest:
            raise ValueError('Model differs from byte-qualified implementation')
    base = q.read(source/'optimization/before/profile.json')
    base['port'] = 18876
    for name in ('baseline','candidate'):
        expected = copy.deepcopy(base)
        if name == 'candidate':
            expected['kvmem']['restore_handoff_fastpath'] = True
        if q.read(root/name/'profile.json') != expected:
            raise ValueError('Unexpected model/recall/budget difference')
        if q.read(root/name/'task-plan.json') != task_plan(
                q.read(tasks_source/'candidate/task-plan.json'),root,name):
            raise ValueError('Task, prompt, cap or sampling changed')
        if (root/name/'runtime').resolve() != (root/'runtime').resolve():
            raise ValueError('Both arms must use the same runtime')
        if not q.optimized_bridge(plan,name):
            raise ValueError('Both arms must use the existing bridge optimization')
    for path, digest in plan['dsh_config_hashes'].items():
        if q.sha(Path(path)) != digest:
            raise ValueError('DSH configuration changed')
    return plan


def prepare(source, tasks_source, output):
    component = validate_component(source)
    if q.read(source/'status.json')['status'] != 'completed':
        raise ValueError('Finished component comparison required')
    for directory in [source/'qualification', source/'optimization/before',source/'optimization/after']:
        if q.read(directory/'execution-audit.json')['status'] != 'passed' or any(
                q.read(directory/'cleanup.json')[key] for key in ('remaining_processes','remaining_owned_gpu_pids')):
            raise ValueError('Qualification/component cleanup not passed')
    old = q.verify_frozen(tasks_source)
    if output.exists():
        raise ValueError('Never reuse an experiment root')
    output.mkdir(parents=True)
    runtime = output/'runtime'
    shutil.copytree(source/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in CONTROLLER_FILES:
        shutil.copy2(ROOT/rel,runtime/rel)
    shutil.copy2(tasks_source/'protocol.so',output/'protocol.so')
    for name in ('baseline','candidate'):
        directory = output/name;directory.mkdir()
        (directory/'runtime').symlink_to(runtime,target_is_directory=True)
        profile = q.read(source/'optimization'/('before' if name=='baseline' else 'after')/'profile.json')
        profile['port'] = 18876
        q.save(directory/'profile.json',profile)
        q.save(directory/'task-plan.json',task_plan(q.read(tasks_source/'candidate/task-plan.json'),output,name))
        q.save(directory/'status.json',dict(status='prepared'))
    plan = {k:old[k] for k in ('python','browser_python','dsh_url','bridge_port')}
    plan.update({k:component[k] for k in ('dependency_state','dependency_expected','dependency_controller',
                                         'idle_seconds','poll_seconds','model_startup_timeout_seconds')})
    dsh = psutil.Process(2869055)
    args = dsh.cmdline()
    if not any(x.endswith('/dsh') for x in args):
        raise ValueError('DSH identity changed')
    patches = [Path(args[i+1]) for i,arg in enumerate(args) if arg == '--patch']
    if not patches:
        raise ValueError('Missing recorded DSH profile')
    proof_files = ['manifest.json','qualification/execution-audit.json','qualification/cleanup.json',
                   'qualification/server/capture/verified_restores.jsonl','comparison.json']
    proof_files += [str(p.relative_to(source)) for p in (source/'qualification').glob('[0-9]-[0-9].json')]
    plan.update(source=str(source),tasks_source=str(tasks_source),protocol=str(output/'protocol.so'),
        optimized_bridge_arms=['baseline','candidate'],dsh_process=dict(**q.identity(dsh),argv=args),
        dsh_config_hashes={str(p):q.sha(p) for p in patches},
        component_proof={rel:q.sha(source/rel) for rel in proof_files},
        original_tasks_sha256=q.sha(tasks_source/'candidate/task-plan.json'),
        qualified_model_hashes={str(p.relative_to(source/'runtime')):q.sha(p)
            for name in ['src','csrc','benchmarks'] for p in (source/'runtime'/name).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts},
        execution_order=['baseline','candidate'], no_quality_retries=True,
        scope='Fresh natural DSH eight-task Off/On comparison. Same prompts, original caps and sampling; '
              'independent empty workspaces. Both use existing residency/cache/bridge and span-merge optimizations. '
              'Only restore_handoff_fastpath differs. Natural generation and tool trajectories may differ. '
              'Declared resource-limit failures remain part of each outcome; they are never rerun or hidden.',
        files={str(p.relative_to(output)):q.sha(p) for p in runtime.rglob('*') if p.is_file()})
    for rel in ['protocol.so']+[f'{name}/{f}.json' for name in ['baseline','candidate'] for f in ['profile','task-plan']]:
        plan['files'][rel]=q.sha(output/rel)
    q.save(output/'manifest.json',plan)
    q.save(output/'status.json',dict(status='prepared',at=time.time()))
    validate(output)


def preflight(root,plan):
    dsh=q.live(plan['dsh_process'])
    if dsh is None or dsh.cmdline()!=plan['dsh_process']['argv']:
        raise ValueError('DSH process identity changed; do not replace it')
    q.require_idle_dsh(plan)
    from scripts.run_dsh_daily_suite import rpc
    rows=rpc(plan['dsh_url'],'session.list',{})['items']
    # DSH list uses sessionId; refuse any reused name before creating a session.
    existing={row['sessionId'] for row in rows}
    intended={case['session_id'] for name in ['baseline','candidate']
              for case in q.read(root/name/'task-plan.json')['cases']}
    if len(intended)!=16 or intended & existing:
        raise ValueError('Fresh session IDs required')
    q.require_free_port(18876);q.require_free_port(plan['bridge_port'])
    subprocess.run([plan['browser_python'],str(root/'runtime/scripts/check_dsh_daily_browser.py'),
        '--root',str(root/'baseline'),'--probe'],check=True,timeout=60)
    q.save(root/'preflight.json',dict(status='passed',fresh_sessions=16,
        existing_idle_sessions=len(rows),dsh_identity=plan['dsh_process'],at=time.time()))


def run(root):
    if ROOT.resolve() != (root/'runtime').resolve():
        raise ValueError('Use the frozen controller')
    plan=validate(root)
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if q.read(root/'status.json')['status']!='prepared':
            raise ValueError('No blind restarts')
        q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()),at=time.time(),
                                             manifest_sha256=q.sha(root/'manifest.json')))
        try:
            preflight(root,plan)
            q.wait_dependency(root,plan)
            validate(root)
            outcomes={}
            for name in plan['execution_order']:
                if q.gpu_pids():
                    raise RuntimeError('Foreign GPU owner appeared; preserve it')
                q.save(root/'status.json',dict(status='running',active=name,at=time.time()))
                try:
                    q.arm(root,name,plan)
                except RuntimeError:
                    result=outcome(root/name)  # Only audited original resource caps are accepted outcomes.
                    if result['full_cohort_passed']:
                        raise
                else:
                    result=outcome(root/name)
                outcomes[name]=result
                q.save(root/name/'outcome.json',result)
            from scripts.analyze_agent_fastpath_dsh import analyze
            q.save(root/'comparison.json',analyze(root))
            q.save(root/'status.json',dict(status='completed',at=time.time(),
                full_paired_cohort_passed=all(x['full_cohort_passed'] for x in outcomes.values()),
                outcomes=outcomes))
        except BaseException as error:
            q.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--tasks-source',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.source:
        prepare(args.source.resolve(),args.tasks_source.resolve(),args.output.resolve())
    else:run(args.output.resolve())
