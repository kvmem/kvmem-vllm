"""One-shot completion stage for the finite 89-task job; no wakeup or timer."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib

import psutil


def read(p):return json.loads(p.read_text())
def save(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024**2):h.update(chunk)
    return h.hexdigest()
def rows(p):
    if not p.exists():return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def require(ok,message):
    if not ok:raise ValueError(message)


def review(root):
    manifest=read(root/'manifest.json');summary=read(root/'formal-events/summary.json')
    names={t['name'] for t in read(root/'tasks.json')}
    launched=set(manifest.get('launched_task_names',names))
    expected=len(launched)
    require(len(summary['outcomes'])==expected,'Missing tasks')
    require({r['task'].removeprefix('terminal-bench/') for r in summary['outcomes']}==launched,'Task set mismatch')
    require(len({r['trial'] for r in summary['outcomes']})==expected,'Duplicate trials')
    checked=0
    for name,entries in manifest['frozen'].items():
        for relative,digest in entries.items():
            require(sha(root/name/relative)==digest,'Frozen source mismatch: '+relative);checked+=1
    for name,digest in manifest['files'].items():
        require(sha(root/name)==digest,'Config mismatch: '+name);checked+=1
    timing_mode=manifest.get('agent_time_limit','official')
    if timing_mode=='none':
        policy=read(root/'agent-time-policy.json')
        require(policy['agent_time_limit']=='none' and set(policy['tasks'])==names,'Missing Agent timing audit')
        for name in names:
            task_path=root/'dataset/tasks'/name/'task.toml'
            require(sha(task_path)==policy['tasks'][name]['effective_sha256'],'Task timing audit mismatch')
            require(tomllib.loads(task_path.read_text()).get('agent',{}).get('timeout_sec') is None,'Agent deadline still enabled')
        for agent in read(root/'job.json')['agents']:
            require(agent.get('override_timeout_sec') is None and agent.get('max_timeout_sec') is None,'Agent deadline override enabled')
    cleanup=read(root/'cleanup.json')
    require(not cleanup['remaining_processes'] and not cleanup['remaining_owned_gpu_pids'],'Owned processes not cleaned')
    lifecycle=rows(root/'formal-events/events.jsonl')
    trial_names={r['trial'] for r in summary['outcomes']}
    released={r['trial'] for r in lifecycle if r['event']=='kv_released'}
    require(released==trial_names,'Missing KV release acknowledgement')
    owner_session={};requests=[]
    for row in rows(root/'bridge/requests.jsonl'):
        if row.get('event')=='session_opened':owner_session[row['owner']]=row['session_id']
        if 'output_tokens' in row:requests.append(row)
    owners={owner for owner,session in owner_session.items() if session.removesuffix('__agent') in trial_names}
    requests=[r for r in requests if r['owner'] in owners]
    accepted=0;reuse=0;finishes=Counter();request_count=0;token_failures=[]
    for owner in owners:
        for p in sorted((root/'bridge'/owner).glob('*/plan.json')):
            plan=read(p);require(plan['seed']==1000,'Actual seed changed')
            require(plan['max_tokens']==131072,'Actual output cap changed')
            require(plan['thinking_token_budget']==4096,'Thinking cap changed')
            for name,value in [('temperature',.6),('top_p',.95),('top_k',20),('min_p',0),('repetition_penalty',1),('presence_penalty',0)]:
                require(plan[name]==value,'Actual sampling changed: '+name)
            request_count+=1
            reuse+=int(plan.get('exact_output_prefix_reused',False))
            if not (p.parent/'accepted.json').exists():
                token_failures.append(str(p.parent.relative_to(root)));continue
            ledger=read(p.parent/'accepted.json');answer=read(p.parent/'response.json');native=read(p.parent/'native.json')
            require(ledger['token_ids']==native['token_ids'],'Native/ledger byte mismatch')
            require(answer['usage']['completion_tokens']==len(ledger['token_ids']),'Accepted usage mismatch')
            accepted+=len(ledger['token_ids']);finishes[native['finish_reason']]+=1
    require(accepted==sum(r['output_tokens'] for r in requests),'Completed request accounting mismatch')
    starts=[e['at'] for e in lifecycle if e['event']=='agent-start']
    ends=[e['at'] for e in lifecycle if e['event']=='agent-end']
    wall=max(ends)-min(starts) if starts and ends else None
    durations=[]
    for r in summary['outcomes']:
        result=read(root/'formal-events'/(r['trial']+'.json'))
        duration=None;phase=result.get('agent_execution') or {}
        if phase.get('started_at') and phase.get('finished_at'):
            duration=(datetime.fromisoformat(phase['finished_at'])-datetime.fromisoformat(phase['started_at'])).total_seconds()
        durations.append(dict(**r,agent_seconds=duration,usage=result.get('agent_result')))
    resources=rows(root/'resources.jsonl');archive=read(root/'model/server/capture/archive.json')
    # Stopped task containers retain each original filesystem for diagnosis.
    all_ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
    containers=[]
    projects={n.lower()+'__env' for n in trial_names}
    for offset in range(0,len(all_ids),100):
        items=json.loads(subprocess.check_output(['docker','inspect',*all_ids[offset:offset+100]]))
        for c in items:
            project=(c['Config'].get('Labels') or {}).get('com.docker.compose.project')
            if project in projects:
                containers.append(dict(id=c['Id'],name=c['Name'],project=project,image=c['Image'],status=c['State']['Status']))
    require(not any(c['status']=='running' for c in containers),'Owned task container still running')
    audit=dict(passed=True,frozen_files_verified=checked,tasks=expected,attempts=expected,retries=0,
        agent_time_limit=timing_mode,official_timing_comparable=(timing_mode=='official'),
        actual_requests=request_count,accepted_completed_response_tokens=accepted,
        actual_seed=1000,output_cap=131072,thinking_budget=4096,
        exact_prefix_reused_requests=reuse,finish_reasons=dict(finishes),
        incomplete_native_requests=token_failures,cleanup=cleanup,
        task_containers_retained=containers,gc=archive.get('historical_gc'),
        cpu_memory_available_min=min((r['memory_available'] for r in resources),default=None),
        gpu_memory_mib_max=max((r.get('gpu_memory_mib',0) for r in resources),default=None),
        agent_window_seconds=wall,accepted_completed_response_tokens_per_wall_second=accepted/wall if wall else None)
    save(root/'execution-audit.json',audit)
    save(root/'results.json',dict(**summary,tasks=durations,measurement=audit))
    evidence={}
    for folder in ('jobs','formal-events','smoke-events','bridge','model','images'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and not p.is_symlink():evidence[str(p.relative_to(root))]=sha(p)
    save(root/'evidence-hashes.json',evidence)
    text=[f'Terminal-Bench 2.1 / Terminus 2: {summary["passed"]}/{expected} passed in this run.',
          f'Scored: {summary["scored"]}; task timeout without score: {summary["task_timeouts"]}; other unscored errors: {summary["errors"]}.',
          'Exactly one attempt per official task; no best-of selection. All raw errors and timeouts retained.',
          f'Completed responses: {accepted:,} accepted output tokens; agent execution window: {wall:.3f} seconds.' if wall else 'No complete agent timing window.',
          'This rate excludes tokens from incomplete responses; interrupted request counts are listed in the audit.',
          f'Frozen sources/configurations verified: {checked}; evidence files hashed: {len(evidence)}.',
          'K98304/G16384; rolling native generation; 131072 output cap; 10M logical ceiling; MTP4; FP8 KV; NVFP4; LRU; no prefill throttling.',
          'Terminus automatic summaries disabled; reasoning retained across turns; official resources/verifiers unchanged.',
          'Agent execution deadline disabled: custom local evaluation, not directly comparable with official timed scores. Verifier/setup/tool timeouts retained.' if timing_mode=='none' else 'Official task Agent deadlines retained.',
          'Maximum eight concurrent tasks; container memory admission capped at 24 GiB without reducing any official task allocation.',
          'Synthetic smoke, fixed warmup and session release requests are excluded from the formal results.',
          '', 'Task | Reward | Exception | Agent seconds']
    for r in sorted(durations,key=lambda r:r['task']):
        text.append(f'{r["task"]} | {r["rewards"]} | {r["exception"]} | {r["agent_seconds"]}')
    (root/'report.txt').write_text('\n'.join(text)+'\n')
    save(root/'final-review.json',dict(status='completed',at=time.time(),report=str(root/'report.txt'),
        report_sha256=sha(root/'report.txt'),evidence_sha256=sha(root/'evidence-hashes.json'),
        passed=summary['passed'],scored=summary['scored'],expected=expected,
        task_timeouts=summary['task_timeouts'],infrastructure_or_unscored_errors=summary['errors']))
    save(root/'status.json',dict(status='completed',at=time.time(),expected=expected,completed=expected,
        passed=summary['passed'],report=str(root/'report.txt'),errors=summary['errors']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.wait:
        owner=read(root/'queue-launch.json')
        while True:
            try:
                process=psutil.Process(owner['pid'])
                if abs(process.create_time()-owner['created'])>.01 or process.status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(10)
    if read(root/'status.json')['status']=='needs_final_review':
        try:review(root)
        except Exception as error:
            save(root/'review-failure.json',dict(at=time.time(),error=repr(error)))
            raise
