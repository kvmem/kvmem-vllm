"""Read-only half-hour checks of the authorized64-task suite; wake this chat for repairs."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import time

from scripts import monitor_ruler_pooling_queue as base
from scripts.monitor_agent_handoff import process, clean
from scripts.monitor_hybrid_suite_queue import read, resources
from scripts.monitor_deepswe_delivery import wake

PREFIX='【DeepSWE 64 次实验定时巡检】'
SEEDS=(1001,1002,1003,1004)


def recent_activity(directory):
    # Tools, verification and filesystem preservation can advance without tokens.
    paths=[directory/'controller.log',directory/'run.json',directory/'window-workspaces.json',
           directory/'bridge/requests.jsonl']
    for pattern in ['cases/*/status.json','cases/*/claude.stream.jsonl',
                    'cases/*/verifier/*.log','cases/*/cell_summary.json']:
        paths.extend(directory.glob(pattern))
    return max((p.stat().st_mtime for p in paths if p.exists()),default=0)


def unresolved_log_tail(path, root, counter_id, resolutions):
    """Only skip a manually reviewed, byte-identical prefix of this engine's log."""
    review=resolutions.get(str(path.resolve()),{})
    size=review.get('prefix_bytes') if isinstance(review,dict) else None
    if (not counter_id or not isinstance(size,int) or isinstance(size,bool) or size<=0
            or review.get('status')!='reviewed'
            or review.get('root')!=str(root.resolve())
            or review.get('counter_id')!=counter_id or not path.exists()):
        return base.log_tail(path),0
    with path.open('rb') as stream:
        stream.seek(0,2);length=stream.tell()
        if length<size:return base.log_tail(path),0
        stream.seek(0);digest=hashlib.sha256();remaining=size
        while remaining:
            chunk=stream.read(min(remaining,1024*1024))
            if not chunk:return base.log_tail(path),0
            digest.update(chunk);remaining-=len(chunk)
        if digest.hexdigest()!=review.get('prefix_sha256'):
            return base.log_tail(path),0
        stream.seek(max(size,length-65536))
        return stream.read().decode('utf-8',errors='replace'),size


def inspect(root, *, previous=None, completion=None, log_resolutions=None, now=None, probe=process, resource_probe=resources):
    previous=previous or {};completion=completion or {};now=time.time() if now is None else now
    plan=read(root/'manifest.json',{});state=read(root/'status.json',{})
    status=state.get('status','missing');active=state.get('active_seed')
    issues=[];arms={};processes={};res=resource_probe(root)
    if plan.get('seeds')!=list(SEEDS) or plan.get('expected_attempts')!=64:
        issues.append('Manifest does not match sixteen tasks by four fixed seeds')
    kv=plan.get('profile',{}).get('kvmem',{})
    if kv.get('prefill_batch_tokens',0) or kv.get('prefill_request_tokens',0) or kv.get('adaptive_prefill',False):
        issues.append('Prefill throttling unexpectedly enabled; retain evidence and inspect frozen configuration')
    processes['queue']=probe(read(root/'queue-process.json',read(root/'started.json',{})))
    if status not in ('completed','failed','prepared') and processes['queue']['state']!='alive':
        issues.append('Queue not visible/alive; verify host identity before recovery')
    if status in ('failed','missing'):issues.append('Queue '+status+': '+str(state.get('error','inspect queue.log'))[:1000])
    for seed in SEEDS:
        name=str(seed);campaign=root/f'seed-{seed}';directory=campaign/'first16'
        run=read(directory/'run.json',{});tasks=run.get('tasks',{})
        counts={s:sum(t.get('state')==s for t in tasks.values()) for s in
                ('completed','context_ceiling','infrastructure_error','running')}
        row=dict(status=read(campaign/'status.json',{}).get('status','prepared'),counts=counts,
                 classified=counts['completed']+counts['context_ceiling'],expected=16)
        score=read(campaign/'scores.json',{});audit=read(campaign/'execution-audit.json',{})
        seed_audit=read(campaign/'seed-audit.json',{})
        row['audit_and_cleanup_passed']=(row['status']=='completed' and row['classified']==16
            and audit.get('status')=='passed' and audit.get('classified_tasks')==16
            and score.get('expected')==16 and len(score.get('tasks',[]))==16
            and seed_audit.get('status')=='passed' and seed_audit.get('seed')==seed
            and (campaign/'evidence-hashes.json').is_file() and clean(campaign))
        if row['status']=='completed':
            row['scored']=score.get('scored');row['passed']=score.get('solved')
            if not row['audit_and_cleanup_passed']:issues.append(name+': completion lacks full cohort/seed/evidence/audit/cleanup')
        if seed==active and status=='running':
            row['phase']=run.get('status','starting')
            for label in ['controller','server','bridge']:
                record=read(directory/(label+'-process.json'),{})
                if record:
                    processes[name+'/'+label]=probe(record)
                    if processes[name+'/'+label]['state']!='alive' and not run.get('ended_at'):
                        issues.append(name+'/'+label+': process visibility/identity needs checking')
            work=read(directory/'server/capture/public_work.json',{})
            row['work']={k:work.get('completed',{}).get(k,0) for k in
                         ['public_batches','public_prefill_tokens','public_non_prefill_tokens']}
            row['counter_id']=work.get('counter_id');row['engine_pid']=work.get('pid')
            row['activity_at']=recent_activity(directory)
            archive=read(directory/'server/capture/archive.json',{})
            gc=archive.get('historical_gc',{}) or {}
            row['gc']={k:gc[k] for k in ['mode','policy','retired','cpu_released','slots_released','policy_failures'] if k in gc}
            if gc.get('policy_failures',0):issues.append(name+': GC policy failure')
            old=previous.get('arms',{}).get(name,{}) if previous.get('root')==str(root) else {}
            changed=any(old.get(k)!=row.get(k) for k in ['work','counter_id','counts','phase','activity_at','gc'])
            row['last_progress_at']=now if changed else old.get('last_progress_at',now)
            if now-row['last_progress_at']>=1800:
                issues.append(name+': no observed progress for30min; inspect restore, GC, tools, verification and startup before acting')
            if counts['infrastructure_error'] or run.get('error') or run.get('resource_error') or run.get('history_contract_error'):
                issues.append(name+': infrastructure failure; preserve completed answers and unfinished workspaces')
            for log in ['server.log','controller.log','bridge.log']:
                path=directory/log
                # Health probes can log connection errors before the model
                # starts listening. A log untouched since readiness is history.
                if path.exists() and path.stat().st_mtime < run.get('model_ready_at',0):
                    continue
                tail,reviewed_bytes=unresolved_log_tail(path,root,row['counter_id'],log_resolutions or {})
                if reviewed_bytes:
                    row.setdefault('reviewed_log_prefixes',{})[log]=reviewed_bytes
                if base.FATAL.search(tail):
                    issues.append(name+': inspect possible fatal error in '+log)
        arms[name]=row
    for key,limit in [('disk_free_bytes',10*2**30),('system_disk_free_bytes',2*2**30),('memory_available_bytes',8*2**30)]:
        if res.get(key) is not None and res[key]<limit:issues.append('Resource pressure: '+key)
    if res.get('gpu_inspection_error'):issues.append('GPU visibility unavailable; never assume released')
    elif active and arms.get(str(active),{}).get('engine_pid'):
        engine=arms[str(active)]['engine_pid']
        for row in res['gpu']:row['matches_active_engine']=row['pid']==engine
        if any(not row['matches_active_engine'] for row in res['gpu']):
            issues.append('Another GPU owner visible; inspect and never terminate unrelated work')
    full=(status=='completed' and all(a['audit_and_cleanup_passed'] for a in arms.values())
          and read(root/'scores.json',{}).get('classified_attempts')==64
          and read(root/'execution-audit.json',{}).get('status')=='passed' and clean(root))
    if status=='completed' and not full:issues.append('Full suite completion lacks final scores/audit/cleanup')
    # The chat must verify all evidence and write its actual final report first.
    report=Path(completion.get('report','/nonexistent'))
    reviewed=(completion.get('status')=='completed' and completion.get('root')==str(root)
        and completion.get('final_review_complete') is True and report.is_file()
        and hashlib.sha256(report.read_bytes()).hexdigest()==completion.get('report_sha256'))
    return dict(at=now,root=str(root),queue_status=status,active_arm=active,arms=arms,
                processes=processes,issues=issues,all_classified_and_audited=full,
                complete=full and reviewed,**res)


def findings(snapshot,previous):
    events=[x for x in snapshot['issues'] if x not in previous.get('issues',[])]
    if previous.get('issues') and not snapshot['issues']:events.append('Previous issue cleared; verify recovery audit')
    for name,row in snapshot['arms'].items():
        if row['audit_and_cleanup_passed'] and not previous.get('arms',{}).get(name,{}).get('audit_and_cleanup_passed'):
            events.append('Seed '+name+' completed; verify all16 results and next cohort')
    if snapshot['all_classified_and_audited'] and not previous.get('all_classified_and_audited'):
        events.append('All64 attempts classified and audited; verify final evidence, report every seed and task, then complete monitor')
    if snapshot['complete'] and not previous.get('complete'):
        events.append('Final report verified; this monitor will exit after successful delivery')
    return events


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','state-dir','instructions','socket']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--thread',required=True);p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--once',action='store_true');p.add_argument('--inspect-only',action='store_true');a=p.parse_args()
    if a.interval!=1800:p.error('Authorized interval is1800 seconds')
    def current(_):
        target=read(a.state_dir/'target.json',{})
        return inspect(Path(target.get('root',str(a.root))),previous=read(a.state_dir/'last-check.json',{}),
                       completion=read(a.state_dir/'completion.json',{}),
                       log_resolutions=read(a.state_dir/'log-resolutions.json',{}))
    if a.inspect_only:
        import json
        print(json.dumps(current(a.root),ensure_ascii=False,indent=2))
    else:
        base.wake=partial(wake,message_prefix=PREFIX)
        # Source-only monitor upgrades keep the existing half-hour cadence.
        last=read(a.state_dir/'last-check.json',{})
        due=min(time.time()+a.interval,last.get('at',0)+a.interval)
        while not a.once and time.time()<due and not (a.state_dir/'STOP').exists():
            time.sleep(min(5,max(0,due-time.time())))
        base.run(a,inspector=current,event_finder=findings)
