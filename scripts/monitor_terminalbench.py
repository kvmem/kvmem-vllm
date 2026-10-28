"""Read-only 20-minute Terminal-Bench checks; repairs wake the authorized chat."""
import argparse
from collections import Counter
from functools import partial
import hashlib
from pathlib import Path
import time

from scripts import monitor_ruler_pooling_queue as base
from scripts.monitor_deepswe_delivery import wake
from scripts.monitor_agent_handoff import process, clean
from scripts.monitor_hybrid_suite_queue import read, resources

PREFIX='【Terminal-Bench 实验定时巡检】'


def inspect(root, *, previous=None, completion=None, now=None, probe=process, resource_probe=resources):
    previous=previous or {};completion=completion or {};now=time.time() if now is None else now
    state=read(root/'status.json',{});manifest=read(root/'manifest.json',{})
    status=state.get('status','missing');issues=[]
    current=read(root/'formal-events/status.json',{});rows=current.get('outcomes',[])
    preserved=read(root/'preserved-outcomes.json',[])
    counts=dict(Counter(r.get('status') for r in rows+preserved))
    errors=[dict(task=r['task'],exception=r.get('exception')) for r in rows if r.get('status')=='error']
    processes={}
    for label,filename in [('queue','queue-process.json'),('formal','formal-process.json'),
                           ('model','model/server-process.json'),('bridge','bridge-process.json')]:
        record=read(root/filename,{})
        if record:processes[label]=probe(record)
    if status=='failed':issues.append('Queue failed: '+str(state.get('error','inspect failure.json')))
    terminal=status in ('completed','failed','needs_final_review')
    if not terminal and status!='prepared' and processes.get('queue',{}).get('state')!='alive':
        issues.append('Queue not visible/alive; verify host process identity before action')
    if status=='formal':
        for label in ('model','bridge','formal'):
            if processes.get(label,{}).get('state')!='alive':
                issues.append(label+': verify service identity/visibility')
    for error in errors:issues.append('Task execution error: '+error['task']+' / '+str(error['exception']))
    if not terminal:
        tail=base.log_tail(root/'model/server.log')
        for fatal in ('EngineCore encountered a fatal error', 'CUDA error: an illegal memory access',
                      'CUDA out of memory', 'Fatal Python error'):
            if fatal in tail:issues.append('Model service fatal log: '+fatal)
    work=read(root/'model/server/capture/public_work.json',{})
    archive=read(root/'model/server/capture/archive.json',{})
    gc=archive.get('historical_gc',{})
    activity_paths=[root/'formal-events/events.jsonl',root/'formal-harness.jsonl',root/'bridge/requests.jsonl']
    activity_paths.extend((root/'jobs').glob('**/trial.log'))
    activity=max((p.stat().st_mtime for p in activity_paths if p.exists()),default=0)
    counters=work.get('completed',{})
    progress=dict(counts=counts,work={k:counters.get(k,0) for k in
                  ('public_batches','public_prefill_tokens','public_non_prefill_tokens')},counter_id=work.get('counter_id'),
                  activity_at=activity,gc={k:gc[k] for k in ('retired','cpu_released','slots_released','policy_failures') if k in gc})
    changed=previous.get('root')!=str(root) or previous.get('progress')!=progress
    last_progress=now if changed else previous.get('last_progress_at',now)
    if status=='formal' and now-last_progress>=1200:
        issues.append('No observed model/tool/verifier progress for20min; inspect before diagnosing stall, never kill on elapsed time alone')
    if gc.get('policy_failures',0):issues.append('Historical GC policy failures')
    res=resource_probe(root)
    for key,limit in [('memory_available_bytes',8*2**30),('disk_free_bytes',30*2**30),('system_disk_free_bytes',2*2**30)]:
        if res.get(key) is not None and res[key]<limit:issues.append('Resource pressure: '+key)
    if res.get('gpu_inspection_error'):issues.append('GPU inspection failed; do not assume release')
    engine=work.get('pid')
    for row in res.get('gpu',[]):row['matches_active_engine']=row['pid']==engine
    review=read(root/'selected-final-review.json',{})
    report=Path(completion.get('report','/nonexistent'))
    complete=(status=='completed' and review.get('scored')==89 and clean(root)
        and completion.get('status')=='completed' and completion.get('root')==str(root)
        and completion.get('final_review_complete') is True and report.is_file()
        and hashlib.sha256(report.read_bytes()).hexdigest()==completion.get('report_sha256'))
    return dict(at=now,root=str(root),queue_status=status,active_arm='formal' if status=='formal' else None,
        arms={},counts=counts,expected=89,scored=counts.get('scored',0),passed=sum(bool(r.get('passed')) for r in rows+preserved),
        current_errors=errors,preserved=len(preserved),launched=len(manifest.get('launched_task_names',[])),
        active=current.get('active',[]),processes=processes,progress=progress,last_progress_at=last_progress,
        issues=issues,complete=complete,**res)


def findings(snapshot,previous):
    events=[v for v in snapshot['issues'] if v not in previous.get('issues',[])]
    if previous.get('issues') and not snapshot['issues']:events.append('Previous issues cleared; verify repair and preserved results')
    if snapshot['queue_status'] in ('needs_final_review','completed') and previous.get('queue_status') not in ('needs_final_review','completed'):
        events.append('Finite queue ended; verify all89 results, original evidence, frozen sources, seeds and GPU cleanup before completion')
    if snapshot['queue_status']=='failed' and previous.get('queue_status')=='failed' and not events:
        events.append('Unresolved failed queue; continue authorized diagnosis while preserving all scored outcomes')
    if snapshot['complete'] and not previous.get('complete'):events.append('Final report verified; monitor exits after notification delivery')
    return events


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','state-dir','instructions','socket'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--thread',required=True);p.add_argument('--interval',type=int,default=1200)
    p.add_argument('--once',action='store_true');p.add_argument('--inspect-only',action='store_true');a=p.parse_args()
    if a.interval!=1200:p.error('Authorized interval is1200 seconds')
    def current(_):
        target=read(a.state_dir/'target.json',{})
        return inspect(Path(target.get('root',str(a.root))),previous=read(a.state_dir/'last-check.json',{}),
                       completion=read(a.state_dir/'completion.json',{}))
    if a.inspect_only:
        import json
        print(json.dumps(current(a.root),ensure_ascii=False,indent=2))
    else:
        base.wake=partial(wake,message_prefix=PREFIX)
        base.run(a,inspector=current,event_finder=findings)
