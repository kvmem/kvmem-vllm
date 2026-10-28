"""User-approved prompt-assisted continuation of five interrupted seed1004 tasks.

Original64 outcomes remain immutable. Recovery scores are a separate cohort.
The model runtime and inference profile are copied from the failed frozen suite.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROMPT='继续完成原任务'
TASKS=('python-statemachine-state-data-scoping','expr-try-catch-errors',
       'scc-bounded-memory-spilling','mnamer-daemon-watch-lifecycle',
       'boa-hierarchical-evaluation-cancellation')
CLIENT_SHA='0bc3800ba6bcf8b9df2daa47509f2159a4c2b74b67922424b4e94e1019c89433'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Frozen source anchor mismatch: '+old[:100])
    return text.replace(old,new)

def patch_controller(text):
    text=replace_once(text,'from scripts.queue_deepswe_seed_suite import validate_preflight_reuse',
        'from scripts.recover_deepswe_seed_snapshot import validate_preflight_reuse')
    text=replace_once(text,'if tid not in adopted_cells:cell.mkdir(parents=True)',
        'if tid not in adopted_cells:\n                from scripts.recover_deepswe_seed_snapshot import claim_cell\n                claim_cell(cell, m)')
    text=replace_once(text,'cursor=0;signalled=None\n            while child.poll() is None:',
        "from scripts.recover_deepswe_seed_snapshot import prime_guards\n            prime_guards(cell, guard, bash_guard)\n            cursor=0;signalled=None\n            while child.poll() is None:")
    return replace_once(text,"len(state['tasks'])==16 and not stop.is_set()",
                        "len(state['tasks'])==len(m['tasks']) and not stop.is_set()")

def patch_agent(text):
    text=replace_once(text,'git checkout -B "$CLAUDE_BRANCH_NAME" "$BASE_COMMIT"',
        '# Keep snapshot HEAD, index and working files; never reset the original task.\n'
        'test -f "$artifact_dir/recovery.json"\n'
        'recovery_session=$(python3 -c \'import json; print(json.load(open("/artifacts/recovery.json"))["session_id"])\')')
    text=replace_once(text,'prompt=$(< "$artifact_dir/prompt.md")',
        'prompt=$(< "$artifact_dir/recovery-prompt.txt")')
    return replace_once(text,'  -p\n','  -p\n  --resume "$recovery_session"\n')

def patch_cell(text):
    return replace_once(text,'image=${metadata[0]}',
        'image=$(python3 - "$cell_dir/recovery.json" <<\'RECOVERY_IMAGE\'\n'
        'import json,sys\nprint(json.load(open(sys.argv[1]))["image"])\nRECOVERY_IMAGE\n)')

def validate_source(source):
    from scripts.run_agent_handoff_queue import live
    state=read(source/'status.json');manifest=read(source/'manifest.json')
    if state['status']!='failed' or live(read(source/'queue-process.json')):
        raise ValueError('Original failed queue must have exited')
    for seed in (1001,1002,1003):
        arm=source/f'seed-{seed}'
        if read(arm/'status.json')['status']!='completed' or read(arm/'execution-audit.json')['status']!='passed':
            raise ValueError('Original completed seed not audited')
    run=read(source/'seed-1004/first16/run.json')
    failed={t for t,r in run['tasks'].items() if r['state']=='infrastructure_error'}
    if failed!=set(TASKS) or sum(r['state']=='completed' for r in run['tasks'].values())!=11:
        raise ValueError('Only the five explicitly authorized interrupted tasks can continue')
    if run.get('cleanup_errors') or not run.get('cleanup_finished_at'):
        raise ValueError('Original processes not cleaned')
    for rel,digest in manifest['files'].items():
        if rel in [f'seed-{s}/status.json' for s in (1001,1002,1003,1004)]:continue
        if sha(source/rel)!=digest:raise ValueError('Original frozen file changed: '+rel)
    if sha(source/'claude-exact-json')!=CLIENT_SHA:raise ValueError('Client changed')
    return manifest

COPY_PRIVATE=r'''
import hashlib,json,os,shutil,sys
from pathlib import Path
source=Path('/source');target=Path('/target');tid=sys.argv[1];uid=int(sys.argv[2]);gid=int(sys.argv[3])
a=source/'cases'/tid;b=target/'cases'/tid;b.mkdir(parents=True)
files={};links={}
for name in ('home','claude-config'):
 shutil.copytree(a/name,b/name,symlinks=True)
for name in ('home','claude-config'):
 for base,dirs,names in os.walk(b/name,followlinks=False):
  for leaf in dirs+names:
   p=Path(base)/leaf;rel=p.relative_to(b);old=a/rel
   if p.is_symlink():
    assert os.readlink(p)==os.readlink(old);links[str(rel)]=os.readlink(p)
   elif p.is_file():
    def digest(f):
     h=hashlib.sha256()
     with f.open('rb') as s:
      for x in iter(lambda:s.read(1048576),b''):h.update(x)
     return h.hexdigest()
    assert digest(old)==digest(p);files[str(rel)]=digest(p)
stream=a/'claude.stream.jsonl';shutil.copy2(stream,b/'prior-stream.jsonl')
sessions=set()
for line in stream.read_text().splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if r.get('session_id'):sessions.add(r['session_id'])
assert len(sessions)==1,sessions
sid=sessions.pop();transcript=b/'claude-config/projects/-app'/(sid+'.jsonl');assert transcript.is_file()
for base,dirs,names in os.walk(b,followlinks=False):
 os.chown(base,uid,gid)
 for leaf in dirs+names:os.chown(Path(base)/leaf,uid,gid,follow_symlinks=False)
print(json.dumps(dict(session_id=sid,files=files,symlink_targets=links)))
'''

def prepare(source,output,approval):
    from scripts.deepswe_tool_cpu_limits import freeze_recovery_setup
    original=validate_source(source)
    if read(approval).get('approved') is not True or read(approval).get('prompt')!=PROMPT:
        raise ValueError('Explicit prompt-assisted recovery approval required')
    if output.exists():raise ValueError('Never overwrite or retry an attempted recovery root')
    output.mkdir(parents=True)
    shutil.copytree(source/'runtime',output/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),output/'runtime/scripts'/Path(__file__).name)
    shutil.copy2(ROOT/'scripts/deepswe_recovery_view.py',output/'runtime/scripts/deepswe_recovery_view.py')
    dashboard=output/'runtime/scripts/deepswe_seed_dashboard.py'
    dashboard.write_text(replace_once(dashboard.read_text(), "if __name__=='__main__':",
        "from scripts.deepswe_recovery_view import install_dashboard\nimport sys\n"
        "install_dashboard(sys.modules[__name__])\n\nif __name__=='__main__':"))
    shutil.copy2(source/'warmup-tokens.bin',output/'warmup-tokens.bin')
    shutil.copy2(approval,output/'approval.json')
    for seed in original['seeds']:(output/f'seed-{seed}').symlink_to(source/f'seed-{seed}',target_is_directory=True)
    campaign=output/'recovery-1004';directory=campaign/'first16';directory.mkdir(parents=True)
    old=source/'seed-1004/first16'
    for name in ('profile.json','baseline-comparison.json'):
        if (old/name).exists():shutil.copy2(old/name,directory/name)
    shutil.copytree(old/'protocol',directory/'protocol')
    freeze_recovery_setup(old/'setup',directory/'setup')
    for name,patch in [('container_claude_agent.sh',patch_agent),('run_deepswe_cell.sh',patch_cell)]:
        path=directory/'setup'/name;path.write_text(patch(path.read_text()))
    task=read(old/'manifest.json');task['tasks']=[t for t in task['tasks'] if t['task_id'] in TASKS]
    task['recovery']=dict(prompt=PROMPT,source=str(old),kind='snapshot_prompt_assisted_cold_start')
    task['setup_sha256']={p.name:sha(p) for p in (directory/'setup').iterdir() if p.is_file()}
    (directory/'tasks.txt').write_text('\n'.join(t['task_id'] for t in task['tasks'])+'\n')
    images={r['task']:r['image'] for r in read(old/'window-workspaces.json')['containers'] if r['role']=='agent'}
    copied={}
    for tid in TASKS:
        image=images[tid]
        actual=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]['Id']
        if actual!=image:raise ValueError('Snapshot image identity changed')
        result=subprocess.run(['docker','run','--rm','--network','none','--read-only','--memory','1g',
            '--cpus','2','--pids-limit','32','--mount',f'type=bind,source={old},target=/source,readonly',
            '--mount',f'type=bind,source={directory},target=/target','--entrypoint','python3',image,
            '-c',COPY_PRIVATE,tid,str(os.getuid()),str(os.getgid())],capture_output=True,text=True,timeout=300)
        if result.returncode:raise RuntimeError('Private state copy failed: '+result.stderr)
        proof=json.loads(result.stdout);copied[tid]=proof
        cell=directory/'cases'/tid
        save(cell/'recovery.json',dict(task_id=tid,session_id=proof['session_id'],image=image,
            source_cell=str(old/'cases'/tid),prompt=PROMPT,seed=1004,
            prior_stream_sha256=sha(cell/'prior-stream.jsonl')))
        (cell/'recovery-prompt.txt').write_text(PROMPT+'\n')
    save(output/'copied-state-audit.json',copied)
    save(directory/'manifest.json',task)
    controller=output/'runtime/scripts/benchmark_deepswe_first16.py'
    controller.write_text(patch_controller(controller.read_text()))
    # The existing scoring implementation is identical, except this separate
    # authorized recovery cohort contains five tasks rather than sixteen.
    reviewer=output/'runtime/scripts/queue_deepswe_fastpath.py'
    text=reviewer.read_text();start=text.index('def final_review(root):');end=text.index('\ndef run(root):',start)
    part=text[start:end].replace('len(expected)!=16','len(expected)!=5').replace('expected=16','expected=5').replace("/16,","/5,").replace('classified_tasks=16','classified_tasks=5')
    reviewer.write_text(text[:start]+part+text[end:])
    save(campaign/'previous-vllm-outcomes.json',dict(source=str(source),original_scored=59,
        original_passed=27,original_infrastructure_errors=5,recovery_reported_separately=True))
    save(campaign/'status.json',dict(status='prepared',seed=1004,recovery=True))
    frozen={}
    for base in (output/'runtime',directory/'setup',directory/'protocol'):
        frozen.update({str(p.relative_to(output)):sha(p) for p in base.rglob('*') if p.is_file()})
    for name in ('profile.json','manifest.json','tasks.txt'):
        frozen[str((directory/name).relative_to(output))]=sha(directory/name)
    for name in ('approval.json','copied-state-audit.json','warmup-tokens.bin'):frozen[name]=sha(output/name)
    for tid in TASKS:
        for name in ('recovery.json','recovery-prompt.txt','prior-stream.jsonl'):
            path=directory/'cases'/tid/name;frozen[str(path.relative_to(output))]=sha(path)
    model={str(p.relative_to(output/'runtime')):sha(p) for base in ('src','csrc')
           for p in (output/'runtime'/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if any(sha(source/'runtime'/p)!=d for p,d in model.items()):raise ValueError('Model code changed')
    manifest=dict(original,seeds=[1001,1002,1003,1004],expected_attempts=64,files=frozen,
        source=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        recovery=dict(task_ids=list(TASKS),expected=5,seed=1004,prompt=PROMPT,
                      cold_start=True,original_scored=59,original_passed=27,original_interrupted=5),
        model_files=model,scope='Five approved prompt-assisted continuations, separate from original64; no lossless process/KV resume claim.')
    save(output/'manifest.json',manifest);save(output/'status.json',dict(status='prepared',active_seed=None,recovery=True))
    print(json.dumps(dict(prepared=str(output),manifest_sha256=sha(output/'manifest.json'),tasks=list(TASKS))))

def validate_preflight_reuse(directory):
    root=directory.parent.parent;m=read(root/'manifest.json');task=read(directory/'manifest.json')
    if m['recovery'].get('mode') == 'fresh_restart':
        from scripts.restart_deepswe_five import validate_fresh
        return validate_fresh(directory)
    if set(m['recovery']['task_ids'])!=set(TASKS) or {t['task_id'] for t in task['tasks']}!=set(TASKS):
        raise ValueError('Recovery allowlist changed')
    if task['profile']!=m['profile'] or task['seed']!=1004 or task['recovery']['prompt']!=PROMPT:
        raise ValueError('Recovery parameters changed')
    if sha(directory/'manifest.json')!=m['files'][str(directory.relative_to(root)/'manifest.json')]:
        raise ValueError('Recovery task manifest changed')
    for p,d in m['qualification_proof'].items():
        if sha(p)!=d:raise ValueError('Qualification evidence changed')
    return m

def claim_cell(cell,manifest):
    proof=read(cell/'recovery.json')
    if cell.name not in TASKS or cell.name not in {t['task_id'] for t in manifest['tasks']}:
        raise ValueError('Completed or foreign task cannot resume')
    if proof['task_id']!=cell.name or proof['seed']!=1004 or proof['prompt']!=PROMPT:
        raise ValueError('Wrong recovery identity')
    if (cell/'recovery-prompt.txt').read_text()!=PROMPT+'\n' or sha(cell/'prior-stream.jsonl')!=proof['prior_stream_sha256']:
        raise ValueError('Recovery input changed')
    with (cell/'recovery-started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)

def prime_guards(cell,guard,bash_guard):
    for line in (cell/'prior-stream.jsonl').open():
        guard.line(line)
        if bash_guard is not None:bash_guard.line(line)
    if guard.first_trigger or (bash_guard and bash_guard.first_trigger):
        raise ValueError('Original task already reached its nonconvergence limit')

def run(root):
    import psutil
    from scripts import run_agent_handoff_queue as q
    from scripts import queue_deepswe_hybrid_lru as lru
    from scripts import queue_deepswe_seed_suite as seeds
    from scripts.queue_deepswe_fastpath import final_review
    if ROOT!=(root/'runtime').resolve():raise ValueError('Use frozen recovery runtime')
    m=read(root/'manifest.json');directory=root/m['recovery'].get('campaign','recovery-1004')/'first16';campaign=directory.parent
    seeds.verify_files(root,m['files']);validate_preflight_reuse(directory)
    validate_source(Path(m['source']))
    with (root/'started.json').open('x') as f:json.dump(q.identity(psutil.Process()),f)
    save(root/'queue-process.json',dict(**q.identity(psutil.Process()),manifest_sha256=sha(root/'manifest.json')))
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        lru.acquire_coordination(root,lock,5);idle=None
        while True:
            lru.stop_requested(root);pids=q.gpu_pids()
            idle=(idle if idle is not None else time.monotonic()) if not pids else None
            save(root/'status.json',dict(status='waiting_for_gpu',active_seed=1004,recovery=True,at=time.time()))
            if idle is not None and time.monotonic()-idle>=60:break
            time.sleep(5)
        if psutil.virtual_memory().available<90*2**30:raise ValueError('Insufficient host memory')
        for port in (18876,m['bridge_port']):q.require_free_port(port)
        owned=q.Owned();process=None
        try:
            save(root/'status.json',dict(status='running',active_seed=1004,recovery=True,at=time.time()))
            save(campaign/'status.json',dict(status='running',seed=1004,recovery=True))
            command=[m['python'],'-B',str(ROOT/'scripts/benchmark_deepswe_first16.py'),
                '--config',str(directory/'profile.json'),'--output',str(directory),'--workers','8',
                '--seed','1004','--backend-kind','kvmem','--duration-seconds','0','--use-prepared',
                '--optimized-handoff','--bash-repeat-limit','32','--bridge-port',str(m['bridge_port']),
                '--warmup-tokens',str(root/'warmup-tokens.bin'),'--startup-timeout-seconds','3600']
            process=owned.start(command,ROOT,q.environment(ROOT),directory/'controller.log',directory/'controller-process.json')
            while process.poll() is None:
                lru.stop_requested(root);owned.track();lru.sample(campaign)
                if set(q.gpu_pids())-set(owned.rows):raise RuntimeError('Foreign GPU owner appeared')
                time.sleep(5)
            lru.sample(campaign)
            if process.returncode:raise RuntimeError('Recovery failed; preserve this attempt without retry')
            save(campaign/'seed-audit.json',seeds.audit_seeds(directory,1004))
            report=final_review(campaign)
            report.update(recovery=True,original_outcomes_unchanged=True,
                mode=m['recovery'].get('mode','snapshot_continuation'),
                fixed_added_prompt=m['recovery'].get('prompt',PROMPT))
            save(campaign/'scores.json',report)
            if lru.lru_review(campaign)['status']=='failed':raise RuntimeError('LRU failure')
            save(campaign/'status.json',dict(status='completed',seed=1004,recovery=True,scored=report['scored'],passed=report['solved']))
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:process.wait(timeout=600)
                except subprocess.TimeoutExpired:raise RuntimeError('Preservation pending; keep owned processes')
            state=read(directory/'run.json') if (directory/'run.json').exists() else {}
            if state.get('window_snapshot_error'):raise RuntimeError('Workspace preservation failed; retain processes')
            cleanup=owned.stop();save(campaign/'cleanup.json',cleanup);save(root/'cleanup.json',cleanup)
        if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:raise RuntimeError('Cleanup incomplete')
    save(root/'status.json',dict(status='needs_final_review',active_seed=None,recovery=True,at=time.time(),
        original_scored=59,original_interrupted=5,recovery_scored=report['scored']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-from',type=Path)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--approval',type=Path);a=p.parse_args()
    if a.prepare_from:prepare(a.prepare_from.resolve(),a.root.resolve(),a.approval)
    else:
        try:run(a.root.resolve())
        except BaseException as e:
            save(a.root/'failure.json',dict(error=repr(e),at=time.time()))
            save(a.root/'status.json',dict(status='failed',error=repr(e),active_seed=None,recovery=True,at=time.time()))
            raise
