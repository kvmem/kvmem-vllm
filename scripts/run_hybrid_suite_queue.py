"""Finite benchmark queue; no timer, answer retry, service restoration or polling notifications."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_niah import save, sha
from graph_v2_protocol import require


def read(path):
    return json.loads(Path(path).read_text())


def grading_python(plan):
    """The frozen evaluator needs its own installed dependencies, not vLLM's."""
    record = plan.get('grading_python')
    if record is None:
        return sys.executable
    path = Path(record['path'])
    require(path.is_absolute() and path.is_file() and os.access(path,os.X_OK) and
        sha(path) == record['sha256'], 'Frozen grading interpreter changed')
    for name,digest in record.get('environment_files',{}).items():
        require(sha(Path(name)) == digest,'Frozen grading environment changed')
    return str(path)


def completed(root):
    require(read(root/'status.json')['status'] == 'completed','Campaign did not complete: '+str(root))
    require(read(root/'execution-audit.json')['status'] == 'passed','Campaign final audit failed')
    cleanup = read(root/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],'GPU cleanup missing')


def completed_grading(record):
    """Skip a separately completed grade only with immutable full-cohort proof."""
    proof = record['completed_proof']
    require(proof['cases'] == (102 if record['suite'] == 'longmemeval_s' else 1116),
        'Partial grading cannot replace the full benchmark verdict')
    for name,digest in proof['files'].items():
        require(sha(Path(name)) == digest,'Completed grading evidence changed')
    output = Path(record['output']); status = read(output/'status.json'); scores = read(output/'scores.json')
    require(status['status'] == 'completed' and status['judge'] == 'deepseek-flash',
        'Declared prior grading is incomplete or uses a different judge')
    if record['suite'] == 'longmemeval_s':
        require(scores['status'] == 'completed' and not scores.get('partial') and
            scores['arms']['hybrid']['count'] == proof['cases'],'Missing full LME verdict')
    else:
        require(scores['complete'] and scores['count'] == proof['cases'],'Missing full MAB verdict')
    require(read(output/'plan.json')['inference_audit_sha256'] ==
        sha(Path(record['campaign'])/'execution-audit.json'),'Prior grade belongs to another inference audit')


def wait_for_grading_cache(source):
    """Wait for a specific finite snapshot grader before reusing its calls."""
    while True:
        state = read(source/'status.json')['status']
        if state == 'completed':
            return
        require(state in ('prepared','grading'), 'Snapshot grading failed; preserve its calls')
        process = read(source/'grader-process.json')
        path = Path('/proc')/str(process['pid'])
        require(path.exists(), 'Snapshot grader disappeared before completing')
        proc_state = (path/'stat').read_text().rsplit(')',1)[1].split()[0]
        require(proc_state not in ('Z','X'), 'Snapshot grader exited without completion')
        require((path/'cmdline').read_bytes().decode().split('\0')[:-1] == process['command'],
            'Snapshot grader process identity changed')
        time.sleep(5)


def run(root, plan_file=None):
    plan_file = root/'queue-plan.json' if plan_file is None else plan_file
    plan = read(plan_file); proof = sha(plan_file)
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prerequisite = Path(plan['wait_for'])
        label = plan.get('waiting_label','ruler')
        save(root/'status.json',dict(status='waiting',active=label,plan_sha256=proof,at=time.time()))
        while read(prerequisite/'status.json')['status'] not in ('completed','failed'):
            control = read(prerequisite/'controller-process.json')
            try:
                state = (Path('/proc')/str(control['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]
            except FileNotFoundError:
                raise RuntimeError('RULER controller disappeared; preserve original attempts')
            require(state not in ('Z','X'),'RULER controller exited without final status')
            time.sleep(5)
        completed(prerequisite)
        for arm in plan['arms']:
            require(sha(plan_file) == proof,'Frozen queue plan changed')
            campaign = Path(arm['root'])
            require(sha(campaign/'prepared-proof.json') == arm['prepared_proof_sha256'],'Prepared campaign proof changed')
            if arm.get('qualification'):
                qualified = Path(arm['qualification']); completed(qualified)
                require(read(qualified/'runtime-proof.json') == read(campaign/'runtime-proof.json'),
                    'Memory qualification and full campaign source differ')
            require(read(campaign/'status.json')['status'] == 'prepared' and not (campaign/'attempt.json').exists(),
                'Refuse to rerun an attempted campaign')
            command = [sys.executable,'-B',str(campaign/'runtime/scripts'/arm['controller']),
                'run','--root',str(campaign)]
            env = dict(os.environ,PYTHONPATH=str(campaign/'runtime/src'))
            with (campaign/'controller.log').open('x') as log:
                child = subprocess.Popen(command,cwd=campaign/'runtime',env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            save(campaign/'controller-process.json',dict(pid=child.pid,command=command,at=time.time()))
            save(root/'status.json',dict(status='running',active=arm['name'],pid=child.pid,at=time.time()))
            require(child.wait() == 0,'Campaign failed: '+arm['name'])
            completed(campaign)
        for record in plan['grading']:
            require(sha(plan_file) == proof,'Frozen queue plan changed')
            campaign = Path(record['campaign']); completed(campaign)
            if record.get('completed_proof'):
                completed_grading(record)
                continue
            if record.get('wait_for_cache_source'):
                save(root/'status.json',dict(status='waiting_grade',active=record['suite'],at=time.time()))
                wait_for_grading_cache(Path(record['cache_source']))
            grader = campaign/'runtime/scripts/grade_hybrid_memory.py'
            if plan.get('grader'):
                grader = Path(plan['grader']['script'])
                for name,digest in plan['grader']['files'].items():
                    require(sha(grader.parent/name) == digest,'Frozen grading tools changed')
            command = [grading_python(plan),'-B',str(grader),
                '--campaign',str(campaign),'--output',record['output'],'--evaluation',plan['evaluation'],
                '--credential-file',plan['credential_file']]
            if record.get('cache_source'):
                command += ['--cache-source',record['cache_source']]
            if 'judge_workers' in record:
                require(type(record['judge_workers']) is int and 1 <= record['judge_workers'] <= 4,
                    'Invalid bounded judge concurrency')
                command += ['--judge-workers',str(record['judge_workers'])]
            save(root/'status.json',dict(status='grading',active=record['suite'],at=time.time()))
            with (root/('grading-'+record['suite']+'.log')).open('x') as log:
                subprocess.run(command,env=dict(os.environ,PYTHONPATH=str(campaign/'runtime/src')),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        save(root/'status.json',dict(status='completed',plan_sha256=proof,at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan-file',type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    try:
        run(root,args.plan_file.resolve() if args.plan_file else None)
    except BaseException as error:
        save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
