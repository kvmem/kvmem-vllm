"""Four new seeded DeepSWE cohorts, sequential on one GPU; no answer retries."""
import argparse
import array
import copy
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from scripts import run_agent_handoff_queue as q
from scripts import queue_deepswe_hybrid_lru as lru
from scripts import queue_deepswe_fastpath as old

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(1001,1002,1003,1004)
OVERLAYS=('benchmarks/deepswe/claude_bridge.py','scripts/benchmark_deepswe_first16.py',
          'scripts/queue_deepswe_seed_suite.py','scripts/deepswe_seed_dashboard.py',
          'scripts/deepswe_dashboard.py','scripts/deepswe_window_snapshot.py',
          'benchmarks/deepswe/dashboard/index.html')


def validate_manifest(manifest):
    if manifest['seeds']!=list(SEEDS) or manifest['expected_attempts']!=64:
        raise ValueError('Expected four new predeclared seeds and exactly64 attempts')
    if len(set(manifest['task_ids']))!=16 or len(manifest['task_ids'])!=16:
        raise ValueError('Expected sixteen unique fixed tasks')


def validate_unlimited_prefill(profile):
    kv=profile['kvmem']
    if (kv.get('prefill_batch_tokens',0)!=0 or kv.get('prefill_request_tokens',0)!=0
            or kv.get('adaptive_prefill',False)):
        raise ValueError('User requested prefill throttling off')


def verify_files(root, files):
    for rel,digest in files.items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or q.sha(root/path)!=digest:
            raise ValueError('Frozen input changed: '+rel)


def validate_preflight_reuse(directory):
    root=directory.parent.parent
    plan=q.read(root/'manifest.json');validate_manifest(plan)
    manifest=q.read(directory/'manifest.json')
    if directory.parent.name!='seed-'+str(manifest['seed']) or manifest['seed'] not in SEEDS:
        raise ValueError('Seed/directory identity mismatch')
    if manifest['profile']!=plan['profile'] or manifest['tasks']!=plan['tasks']:
        raise ValueError('Formal profile or cohort changed')
    if q.sha(directory/'manifest.json')!=plan['files'][str(directory.relative_to(root)/'manifest.json')]:
        raise ValueError('Formal manifest changed')
    for path,digest in plan['qualification_proof'].items():
        if q.sha(Path(path))!=digest:raise ValueError('Qualification proof changed')
    return plan


def audit_seeds(directory, seed):
    paths=sorted((directory/'bridge').glob('*/*/plan.json'))
    if not paths:raise ValueError('No actual model request plans')
    tasks={t['task_id'] for t in q.read(directory/'manifest.json')['tasks']}
    for path in paths:
        plan=q.read(path)
        if path.parent.parent.name not in tasks or plan.get('seed')!=seed:
            raise ValueError('Actual model request used wrong task or seed: '+str(path))
        expected=dict(temperature=.6,top_p=.95,top_k=20,min_p=0,
                      presence_penalty=0,repetition_penalty=1,thinking_token_budget=4096)
        if any(plan.get(k)!=v for k,v in expected.items()) or not 0<plan['max_tokens']<=16384:
            raise ValueError('Actual sampling/output cap changed')
    return dict(status='passed',seed=seed,model_requests=len(paths),
                scope='Every saved native request plan, including unfinished requests')


def summarize(root):
    plan=q.read(root/'manifest.json');validate_manifest(plan)
    records=[];rounds=[]
    for seed in SEEDS:
        campaign=root/f'seed-{seed}'
        score=campaign/'scores.json'
        if not score.exists():continue
        report=q.read(score)
        if report['status']!='completed' or report['expected']!=16:
            raise ValueError('Incomplete classified cohort')
        ids=[r['task'] for r in report['tasks']]
        if len(ids)!=16 or set(ids)!=set(plan['task_ids']):
            raise ValueError('Missing or duplicated scored task')
        rounds.append(dict(seed=seed,scored=report['scored'],solved=report['solved'],
                           context_ceiling=report['context_ceiling']))
        for row in report['tasks']:records.append(dict(seed=seed,**row))
    tasks=[]
    for task in plan['task_ids']:
        selected=[r for r in records if r['task']==task]
        rewards=[r.get('reward') if r['state']=='completed' else None for r in selected]
        tasks.append(dict(task=task,attempts=len(selected),scored=sum(r is not None for r in rewards),
                          passed=sum(r==1 for r in rewards),results=selected))
    result=dict(status='completed' if len(records)==64 else 'running',expected_attempts=64,
        classified_attempts=len(records),scored=sum(r['scored'] for r in rounds),
        passed=sum(r['solved'] for r in rounds),rounds=rounds,tasks=tasks,
        limitations='Four fresh fixed seeds; every attempt retained. Context ceilings remain unscored. '
                    'No reward-based retries or best-attempt selection.')
    q.save(root/'scores.json',result);return result


def prepare(source, arm, formal, output, client, decision):
    from scripts.queue_agent_interference import completed
    if output.exists():raise ValueError('Use a new root; never overwrite attempts')
    source_plan=q.read(source/'manifest.json')
    proofs=['qualification'] if arm in source_plan.get('correctness_only_candidates',[]) else ['qualification',arm]
    for name in proofs:completed(source,name)
    verify_files(source/'runtime',source_plan['files'])
    from scripts.benchmark_deepswe_interference import CLIENT_SHA
    if q.sha(client)!=CLIENT_SHA:raise ValueError('Unqualified client binary')
    reference=q.read(formal/'first16/manifest.json')
    old.verify_images(reference)
    profile=q.read(source/arm/'profile.json');profile['port']=18876
    validate_unlimited_prefill(profile)
    if (profile.get('verify_session_restore') or profile.get('verify_session_restore_digest')
            or profile['kvmem'].get('host_stage_timing') or profile['kvmem'].get('restore_stage_timing')):
        raise ValueError('Diagnostic instrumentation is forbidden in formal tasks')
    if (profile['kvmem']['select_budget'],profile['kvmem']['gen_budget'],profile['max_model_len'],profile['max_num_seqs'])!=(98304,16384,10000000,8):
        raise ValueError('Formal budgets changed')
    if reference['seed'] in SEEDS:raise ValueError('New seeds must not repeat old attempted seed')
    output.mkdir(parents=True)
    shutil.copytree(source/'runtime',output/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    for rel in OVERLAYS:shutil.copy2(ROOT/rel,output/'runtime'/rel)
    # Exercise imports in the actual frozen tree, including failure-only paths,
    # before creating any model process or issuing any benchmark request.
    subprocess.run([source_plan['python'],'-B','-c',
        'from scripts import queue_deepswe_seed_suite, benchmark_deepswe_first16, '
        'deepswe_seed_dashboard, deepswe_window_snapshot, analyze_deepswe'],
        cwd=output/'runtime',env=q.environment(output/'runtime'),check=True)
    shutil.copy2(client,output/'claude-exact-json')
    shutil.copy2(decision,output/'configuration-decision.json')
    fixture=q.read(source/'fixture.json')
    tokens=array.array('i',fixture[0]['tokens'][:100352])
    if sys.byteorder!='little':tokens.byteswap()
    (output/'warmup-tokens.bin').write_bytes(tokens.tobytes())
    qualification={str(source/name/f):q.sha(source/name/f) for name in proofs
                   for f in ['execution-audit.json','cleanup.json','evidence-hashes.json']}
    # Keep the existing full-G and atomic-query evidence, without relaunching
    # the user-stopped synthetic ten-million-token trial.
    prior=Path(q.read(formal/'manifest.json')['source'])
    for name in ['atomic-query-preflight.json','short-tail-preflight.json','generation-preflight.json','cleanup-preflight.json']:
        path=prior/'first16'/name
        if q.read(path).get('passed') is not True:raise ValueError('Missing original execution qualification')
        qualification[str(path)]=q.sha(path)
    for seed in SEEDS:
        campaign=output/f'seed-{seed}';directory=campaign/'first16';directory.mkdir(parents=True)
        q.save(directory/'profile.json',profile)
        shutil.copytree(formal/'first16/setup',directory/'setup')
        script=directory/'setup/run_deepswe_cell.sh'
        text=script.read_text();old_mount='/home/chaidi/.local/share/claude/versions/2.1.148:/usr/local/bin/claude:ro'
        if text.count(old_mount)!=1:raise ValueError('Unexpected client mount')
        script.write_text(text.replace(old_mount,str(output/'claude-exact-json')+':/usr/local/bin/claude:ro'))
        shutil.copytree(formal/'first16/protocol',directory/'protocol')
        shutil.copy2(formal/'first16/tasks.txt',directory/'tasks.txt')
        shutil.copy2(formal/'previous-vllm-outcomes.json',campaign/'previous-vllm-outcomes.json')
        task=copy.deepcopy(reference)
        for name in ['preflight_reuse','long_gc_qualification']:task.pop(name,None)
        task.update(seed=seed,profile=profile,profile_sha256=q.sha(directory/'profile.json'),
            claude_sha256=CLIENT_SHA,carried_results=[],duration_seconds=0,
            seed_suite_qualification='../../manifest.json',
            setup_sha256={p.name:q.sha(p) for p in (directory/'setup').iterdir() if p.is_file()})
        task['differences']=[x for x in task['differences'] if not x.startswith('vLLM native per-request seed=')]
        task['differences'].append(f'Fresh requested replicate, seed={seed}; qualified JSON client; fixed optimized runtime.')
        q.save(directory/'manifest.json',task)
        q.save(campaign/'status.json',dict(status='prepared',seed=seed))
    external={str(Path(t['instruction'])):t['instruction_sha256'] for t in reference['tasks']}
    plan=dict(python=source_plan['python'],source=str(source),source_arm=arm,seeds=list(SEEDS),
        expected_attempts=64,task_ids=[t['task_id'] for t in reference['tasks']],tasks=reference['tasks'],
        profile=profile,qualification_proof=qualification,external_files=external,bridge_port=18490,
        files=lru.inventory(output),original_results='Preserved separately; none count toward the new64',
        execution='Four sequential sixteen-task cohorts; at most eight independent agents, no duration cap, no retries')
    validate_manifest(plan);q.save(output/'manifest.json',plan)
    q.save(output/'status.json',dict(status='prepared',active_seed=None,expected_attempts=64,at=time.time()))


def run(root):
    if ROOT.resolve()!=(root/'runtime').resolve():raise ValueError('Use the frozen controller')
    plan=q.read(root/'manifest.json');validate_manifest(plan);verify_files(root,plan['files'])
    validate_unlimited_prefill(plan['profile'])
    if (root/'started.json').exists():raise ValueError('Attempted suite requires audited recovery, never automatic retry')
    for path,digest in plan['external_files'].items():
        if q.sha(Path(path))!=digest:raise ValueError('Task changed')
    for seed in SEEDS:validate_preflight_reuse(root/f'seed-{seed}/first16')
    q.save(root/'started.json',dict(**q.identity(psutil.Process()),at=time.time()))
    q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()),at=time.time(),
        manifest_sha256=q.sha(root/'manifest.json')))
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        lru.acquire_coordination(root,lock,5)
        for seed in SEEDS:
            campaign=root/f'seed-{seed}';directory=campaign/'first16';owned=q.Owned()
            idle=None
            while True:
                lru.stop_requested(root);pids=q.gpu_pids()
                idle=(idle if idle is not None else time.monotonic()) if not pids else None
                q.save(root/'status.json',dict(status='waiting_for_gpu',active_seed=seed,expected_attempts=64,at=time.time()))
                if idle is not None and time.monotonic()-idle>=60:break
                time.sleep(5)
            if psutil.virtual_memory().available<90*2**30:raise ValueError('Insufficient host memory')
            for port in [18876,plan['bridge_port']]:q.require_free_port(port)
            if (directory/'run.json').exists():raise ValueError('Seed cohort was already attempted')
            process=None
            try:
                q.save(root/'status.json',dict(status='running',active_seed=seed,expected_attempts=64,at=time.time()))
                q.save(campaign/'status.json',dict(status='running',seed=seed))
                command=[plan['python'],'-B',str(ROOT/'scripts/benchmark_deepswe_first16.py'),
                    '--config',str(directory/'profile.json'),'--output',str(directory),'--workers','8',
                    '--seed',str(seed),'--backend-kind','kvmem','--duration-seconds','0','--use-prepared',
                    '--optimized-handoff','--bash-repeat-limit','32','--bridge-port',str(plan['bridge_port']),
                    '--warmup-tokens',str(root/'warmup-tokens.bin'),'--startup-timeout-seconds','3600']
                process=owned.start(command,ROOT,q.environment(ROOT),directory/'controller.log',directory/'controller-process.json')
                while process.poll() is None:
                    lru.stop_requested(root);owned.track();lru.sample(campaign)
                    if set(q.gpu_pids())-set(owned.rows):raise RuntimeError('Foreign GPU owner appeared')
                    time.sleep(5)
                lru.sample(campaign)
                if process.returncode:raise RuntimeError('Formal cohort failed; all attempts preserved')
                proof=audit_seeds(directory,seed);q.save(campaign/'seed-audit.json',proof)
                report=old.final_review(campaign);gc=lru.lru_review(campaign)
                if gc['status']=='failed':raise RuntimeError('Historical GC policy failure')
                q.save(campaign/'status.json',dict(status='completed',seed=seed,scored=report['scored'],passed=report['solved']))
            finally:
                # Let the controller snapshot unfinished work before a broad
                # process-tree cleanup can trigger container EXIT traps.
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:process.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        q.save(campaign/'cleanup.json',dict(status='needs_workspace_recovery',
                            reason='Controller preservation still pending; owned processes retained'))
                        raise RuntimeError('Workspace preservation pending; do not destroy owned work')
                state=q.read(directory/'run.json') if (directory/'run.json').exists() else {}
                if state.get('window_snapshot_error'):
                    q.save(campaign/'cleanup.json',dict(status='needs_workspace_recovery',
                        error=state['window_snapshot_error']))
                    raise RuntimeError('Unpreserved workspace retained for audited recovery')
                cleanup=owned.stop();q.save(campaign/'cleanup.json',cleanup)
            if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
                raise RuntimeError('Owned cleanup incomplete')
            summarize(root)
        result=summarize(root)
        if result['classified_attempts']!=64:raise RuntimeError('Incomplete64-attempt cohort')
        q.save(root/'cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[],at=time.time()))
        q.save(root/'execution-audit.json',dict(status='passed',classified_attempts=64,
            rounds={str(seed):q.sha(root/f'seed-{seed}/execution-audit.json') for seed in SEEDS}))
        q.save(root/'status.json',dict(status='completed',active_seed=None,expected_attempts=64,
            scored=result['scored'],passed=result['passed'],at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--arm');p.add_argument('--formal',type=Path)
    p.add_argument('--client',type=Path);p.add_argument('--decision',type=Path)
    a=p.parse_args()
    if a.source:prepare(a.source.resolve(),a.arm,a.formal.resolve(),a.output.resolve(),a.client.resolve(),a.decision.resolve())
    else:
        try:run(a.output.resolve())
        except BaseException as error:
            q.save(a.output/'failure.json',dict(error=repr(error),at=time.time()))
            q.save(a.output/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
