"""Read-only 20-minute checks; repairs/optimization stay in the existing chat.

The GPU queue owns execution. This process never launches or stops an engine.
An audited A/B completion wakes optimization review rather than ending it.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import psutil

import monitor_ruler_pooling_queue as base
from monitor_hybrid_suite_queue import read, resources

PREFIX = '【KVMem Agent 吞吐定时巡检】'


def process(record):
    try:
        p = psutil.Process(record['pid'])
        if p.create_time() != record['created']:
            return dict(state='identity_changed', pid=p.pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return dict(state='exited', pid=p.pid)
        return dict(state='alive', pid=p.pid)
    except (KeyError, psutil.NoSuchProcess):
        return dict(state='missing_or_exited', pid=record.get('pid'))
    except psutil.AccessDenied:
        return dict(state='visibility_denied', pid=record.get('pid'))


def clean(directory):
    value = read(directory/'cleanup.json', {})
    return (value.get('remaining_processes') == [] and
            value.get('remaining_owned_gpu_pids') == [])


def preserved_limit(root, manifest):
    """Acknowledge a hash-bound task failure, never count it as a passed arm."""
    binding = manifest.get('preserved_baseline')
    if not binding:
        return None
    path = root/binding['proof']
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
        raise ValueError('Preserved baseline proof changed')
    proof = read(path, {})
    if (proof.get('status') != 'incomplete_predeclared_limit' or
            proof.get('failed_cases') != ['r01-fps'] or proof.get('retry_allowed') is not False):
        raise ValueError('Unexpected baseline outcome')
    source = Path(proof['source'])
    if (root/'baseline').resolve() != (source/'baseline').resolve():
        raise ValueError('Preserved baseline reference changed')
    for rel, expected in proof['hashes'].items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Preserved baseline audit changed: '+rel)
    if not clean(source/'baseline'):
        raise ValueError('Preserved baseline cleanup incomplete')
    return proof


def inspect(root, *, previous=None, completion=None, now=None, probe=process, resource_probe=resources):
    now = time.time() if now is None else now
    previous = previous or {}
    state = read(root/'status.json', {})
    manifest = read(root/'manifest.json', {})
    status, active = state.get('status', 'missing'), state.get('active')
    issues, processes, arms = [], {}, {}
    try:
        preserved = preserved_limit(root, manifest)
    except (OSError, ValueError, KeyError) as error:
        preserved = None
        issues.append('Invalid baseline outcome proof: '+str(error))
    res = resource_probe(root)
    queue = probe(read(root/'queue-process.json', {})); processes['queue'] = queue
    if status not in ('completed', 'failed', 'needs_optimization_review') and queue['state'] != 'alive':
        issues.append('Queue process '+queue['state']+'; verify host identity/visibility before recovery')
    if status in ('failed', 'missing'):
        issues.append('Queue '+status+': '+str(state.get('error', 'inspect queue.log'))[:1200])
    dependency = read(Path(manifest.get('dependency_state', '/nonexistent')), {})
    processes['dependency'] = probe(manifest.get('dependency_controller', {}))
    dependency_done = (dependency.get('phase') == 'complete' and
        dependency.get('summary', {}).get('all', {}).get('count') == manifest.get('dependency_expected') and
        processes['dependency']['state'] in ('missing_or_exited','exited','identity_changed'))
    gpu_released = (dependency_done and res.get('gpu') == [] and not res.get('gpu_inspection_error'))
    if status == 'waiting_for_benchmark':
        if dependency.get('phase') in ('failed', 'partial', 'error'):
            issues.append('Predecessor did not complete; inspect, never stop/restart its GPU process')
        if now-state.get('at', now) > 180:
            issues.append('Queue heartbeat is stale; inspect its dependency wait')
    qualification = root/'qualification'
    qa = read(qualification/'execution-audit.json', {})
    qualification_passed = qa.get('status') == 'passed' and clean(qualification)
    for name in ('baseline', 'candidate'):
        directory = root/name
        s = read(directory/'status.json', {})
        plan = read(directory/'task-plan.json', {})
        cases = []
        for task in plan.get('cases', []):
            case = directory/'cases'/task['id']
            c = read(case/'status.json', {})
            if (case/'failure.json').exists():
                acknowledged = (name == 'baseline' and preserved and
                                task['id'] in preserved['failed_cases'])
                if not acknowledged:
                    issues.append(name+'/'+task['id']+': failed attempt; preserve all outputs')
            else:
                acknowledged = False
            cases.append(dict(id=task['id'],status='resource_limit' if acknowledged else c.get('status','waiting'),
                turns=c.get('turns_completed',0),context=c.get('max_logical_input_tokens',0),
                model_steps=c.get('statistics',{}).get('model_steps',0),
                tool_calls=c.get('statistics',{}).get('tool_calls',0)))
        done = sum(c['status']=='completed' for c in cases)
        audit = read(directory/'execution-audit.json', {})
        perf = read(directory/'performance.json', {})
        audited = (len(cases)==8 and done==8 and all(c['turns']>=4 and c['context']>36864 for c in cases)
            and audit.get('status')=='passed' and clean(directory) and
            perf.get('counts',{}).get('completed_tasks')==8 and
            perf.get('accepted_output_tokens',0)>0 and perf.get('wall_seconds',0)>0 and
            (directory/'evidence-hashes.json').is_file() and (directory/'browser-summary.json').is_file())
        value = dict(status=s.get('status','waiting'),completed=done,expected=8,
            cases=cases,audit_and_cleanup_passed=audited)
        if name == 'baseline' and preserved:
            value.update(status='incomplete_predeclared_limit', failure_acknowledged=True)
        arms[name] = value
        if s.get('status')=='completed' and not audited:
            # Controller marks completion immediately before cleanup; allow its
            # bounded shutdown to finish before classifying it as an error.
            if now-s.get('at',now)>120:
                issues.append(name+': completion missing full cohort/audit/performance/evidence/GPU cleanup')
        if active!=name or status!='running':continue
        for which in ('server','bridge'):
            record=read(directory/(which+'-process.json'),{})
            if record:
                processes[name+'/'+which]=probe(record)
                if processes[name+'/'+which]['state']!='alive' and not audited:
                    issues.append(name+'/'+which+': process '+processes[name+'/'+which]['state']+'; verify before recovery')
        work=read(directory/'server/capture/public_work.json', {})
        value['work']={k:work.get('completed',{}).get(k,0) for k in (
            'public_batches','public_prefill_tokens','public_non_prefill_tokens')}
        value['counter_id']=work.get('counter_id')
        value['engine_pid']=work.get('pid')
        old=previous.get('arms',{}).get(name,{}) if previous.get('root')==str(root) else {}
        changed=any(old.get(k)!=value.get(k) for k in ('status','cases','work','counter_id'))
        last=now if changed else old.get('last_progress_at',now)
        value['last_progress_at']=last
        if now-last>=1200:
            issues.append(name+': no observed work advance for 20 minutes; inspect tools, recall, archive and audit before recovery; never kill on elapsed time alone')
        for log in ('server.log','bridge.log'):
            if base.FATAL.search(base.log_tail(directory/log)):
                issues.append(name+': possible fatal error in '+log+'; inspect actual request outcome')
    if status=='qualification':
        for log in (root/'gpu-component.log',qualification/'server.log'):
            if base.FATAL.search(base.log_tail(log)):
                issues.append('Qualification: inspect '+log.name)
    for key,limit in (('disk_free_bytes',10*1024**3),('system_disk_free_bytes',2*1024**3),('memory_available_bytes',2*1024**3)):
        if res.get(key) is not None and res[key]<limit:issues.append('Resource pressure: '+key)
    if res.get('gpu_inspection_error'):issues.append('GPU visibility unavailable; never treat this as idle')
    optimization = {}
    if manifest.get('optimization_kind') == 'graph_span_merge':
        for name in manifest.get('optimization_arms', ('before', 'after')):
            directory = root/'optimization'/name
            state = read(directory/'status.json', {})
            audit = read(directory/'execution-audit.json', {})
            answers = list((directory/'answers').glob('*.json'))
            audited = (state.get('status') == 'completed' and len(answers) == 64 and
                audit.get('status') == 'passed' and audit.get('responses') == 64 and
                audit.get('restores') == 56 and audit.get('exact_output_equal') is True and
                clean(directory) and (directory/'performance.json').is_file() and
                (directory/'evidence-hashes.json').is_file())
            if name == 'profile':
                stages = read(directory/'stage-audit.json', {})
                audited = (audited and stages.get('status') == 'passed' and
                           stages.get('restores') == 56 and stages.get('timed_rows', 0) > 0)
            value = dict(status=state.get('status', 'prepared'), completed=len(answers), expected=64,
                         audit_and_cleanup_passed=audited, component_only=True)
            optimization[name] = value
            if status == 'optimization_running' and active == name:
                record = read(directory/'run/server-process.json', {})
                if record:
                    processes['optimization/'+name+'/server'] = probe(record)
                    if processes['optimization/'+name+'/server']['state'] != 'alive' and not audited:
                        issues.append('Optimization '+name+': server identity not live; verify before recovery')
                work = read(directory/'run/server/capture/public_work.json', {})
                value['work'] = {k:work.get('completed', {}).get(k, 0) for k in (
                    'public_batches', 'public_prefill_tokens', 'public_non_prefill_tokens')}
                value['engine_pid'] = work.get('pid')
                value['counter_id'] = work.get('counter_id')
                old = previous.get('optimization', {}).get(name, {}) if previous.get('root') == str(root) else {}
                changed = any(old.get(k) != value.get(k) for k in ('status', 'completed', 'work', 'counter_id'))
                value['last_progress_at'] = now if changed else old.get('last_progress_at', now)
                if now-value['last_progress_at'] >= 1200:
                    issues.append('Optimization '+name+': no observed work advance; inspect startup/restore/prefill before recovery')
                if base.FATAL.search(base.log_tail(directory/'run/server.log')):
                    issues.append('Optimization '+name+': possible fatal server error; preserve evidence')
            if status == 'needs_optimization_review' and not audited:
                issues.append('Optimization '+name+': queue finished without complete component audit/cleanup')
    engine = (optimization.get(active, {}) if status == 'optimization_running' else arms.get(active, {})).get('engine_pid')
    if status in ('running', 'optimization_running') and engine:
        for owner in res.get('gpu',[]):
            owner['matches_active_engine']=owner['pid']==engine
            if not owner['matches_active_engine']:
                issues.append('Foreign GPU workload overlaps this test; preserve it and assess comparison validity')
    tests_complete = (status=='completed' and qualification_passed and
        all(v['audit_and_cleanup_passed'] for v in arms.values()) and (root/'comparison.json').is_file())
    if status=='completed' and not tests_complete:issues.append('Queue claims completion without all qualification/audit/cleanup evidence')
    # This file is written only by the repairing conversation after reviewing
    # all tests and subsequent optimization experiments, never by this monitor.
    decision=completion or {}
    complete = (tests_complete and not issues and decision.get('status')=='completed' and
        decision.get('root')==str(root) and decision.get('optimization_review_complete') is True and
        isinstance(decision.get('report'),str) and Path(decision['report']).is_file())
    return dict(at=now,root=str(root),queue_status=status,active_arm=active,
        qualification_passed=qualification_passed,tests_complete=tests_complete,complete=complete,
        dependency_phase=dependency.get('phase'),gpu_released=gpu_released,
        processes=processes,arms=arms,optimization=optimization,issues=issues,**res)


def findings(snapshot, previous):
    events=list(snapshot['issues'])
    if snapshot['gpu_released'] and not previous.get('gpu_released'):
        events.append('Predecessor complete and GPU released; verify existing queue has started, do not duplicate it')
    if snapshot['queue_status'] in ('qualification','running') and previous.get('queue_status')=='waiting_for_benchmark':
        events.append('GPU queue started after dependency release; verify qualification and progress')
    if snapshot['qualification_passed'] and not previous.get('qualification_passed'):
        events.append('GPU qualification passed; verify exact-byte audit and continue paired agent tests')
    for name,value in snapshot['arms'].items():
        if value['audit_and_cleanup_passed'] and not previous.get('arms',{}).get(name,{}).get('audit_and_cleanup_passed'):
            events.append(name+': audited stage completed; compare performance and error rates')
    for name, value in snapshot.get('optimization', {}).items():
        if value['audit_and_cleanup_passed'] and not previous.get('optimization', {}).get(name, {}).get('audit_and_cleanup_passed'):
            events.append('Optimization '+name+': fixed-history component completed; verify paired timing and original evidence')
    if snapshot['tests_complete'] and not snapshot['complete']:
        events.append('Paired tests complete; analyze measured bottlenecks and continue authorized optimization, preserve completed results')
    if snapshot['queue_status'] == 'needs_optimization_review' and previous.get('queue_status') != 'needs_optimization_review':
        events.append('Candidate finished; baseline retains its resource-limit failure. Review performance and continue optimization; this is not a fully passed A/B cohort.')
    if snapshot['complete'] and not previous.get('complete'):
        events.append('Tests and optimization review complete; verify final report and stop this monitor')
    if previous.get('issues') and not snapshot['issues']:events.append('Previous issues cleared; verify recovery audit')
    return events


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('root','state-dir','instructions','socket'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--thread',required=True)
    parser.add_argument('--interval',type=int,default=1200)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    if args.interval<60:parser.error('Interval must be at least 60 seconds')
    def current(_):
        target=read(args.state_dir/'target.json',{})
        root=Path(target.get('root',str(args.root)))
        return inspect(root,previous=read(args.state_dir/'last-check.json',{}),
            completion=read(args.state_dir/'completion.json',{}))
    if args.inspect_only:print(json.dumps(current(args.root),ensure_ascii=False,indent=2))
    else:
        base.wake=partial(base.wake,message_prefix=PREFIX)
        base.run(args,inspector=current,event_finder=findings)
