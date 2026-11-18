"""One-shot completion stage for the finite 89-task job; no wakeup or timer."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import tomllib

import psutil

try:
    from .gpu_efficiency_monitor import summarize as summarize_gpu
except ImportError:
    from gpu_efficiency_monitor import summarize as summarize_gpu


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


def event_window(lifecycle, begin, finish):
    starts=[row.get('at') for row in lifecycle if row.get('event')==begin]
    ends=[row.get('at') for row in lifecycle if row.get('event')==finish]
    if not starts or not ends:
        return dict(status='unavailable',reason=f'Missing {begin}/{finish} events')
    if not all(type(value) in (int,float) and math.isfinite(value) for value in starts+ends):
        return dict(status='unavailable',reason='Invalid lifecycle timestamp')
    start,end=min(starts),max(ends)
    if end<=start:
        return dict(status='unavailable',reason='Nonpositive lifecycle window')
    return dict(status='measured',start_epoch=start,end_epoch=end,seconds=end-start)


def agent_intervals(lifecycle):
    """Use actual Agent lifecycle events; task admission is not Agent execution."""
    trials={}
    for row in lifecycle:
        kind=row.get('event')
        if kind not in ('agent-start','agent-end'):
            continue
        trial,at=row.get('trial'),row.get('at')
        if not trial or type(at) not in (int,float) or not math.isfinite(at):
            return None,'Agent events lack a trial identity or valid timestamp'
        events=trials.setdefault(trial,{})
        if kind in events:
            return None,'Duplicate Agent lifecycle event: '+trial
        events[kind]=at
    intervals=[]
    for trial,events in trials.items():
        if set(events)!= {'agent-start','agent-end'} or events['agent-end']<events['agent-start']:
            return None,'Incomplete or reversed Agent lifecycle: '+trial
        intervals.append((events['agent-start'],events['agent-end']))
    if not intervals:
        return None,'No complete Agent lifecycle intervals'
    return intervals,None


def concurrency_summary(intervals,window):
    start,end=window['start_epoch'],window['end_epoch']
    changes=Counter({start:0,end:0})
    for begin,finish in intervals:
        begin,finish=max(start,begin),min(end,finish)
        if finish>begin:
            changes[begin]+=1
            changes[finish]-=1
    active=0;previous=start;seconds=Counter()
    # Aggregate simultaneous boundaries before measuring the following span:
    # adjacent Agents do not briefly count as two concurrent Agents.
    for at,delta in sorted(changes.items()):
        if at>previous:
            seconds[active]+=at-previous
        active+=delta
        previous=at
    occupied=sum(count*duration for count,duration in seconds.items())
    return dict(status='measured',peak=max(seconds,default=0),
        time_weighted_mean=occupied/(end-start),summed_agent_seconds=occupied,
        seconds_by_active_agents={str(count):duration for count,duration in sorted(seconds.items())},
        scope='Concurrent Agent lifetimes include model calls and tool waits; not simultaneous GPU batches.')


def efficiency_metrics(root,lifecycle):
    """Keep setup, Agent work, and GPU sample coverage separate in the report."""
    windows=dict(formal=event_window(lifecycle,'start','end'),
                 agent=event_window(lifecycle,'agent-start','agent-end'))
    intervals,interval_error=agent_intervals(lifecycle)
    telemetry=root/'gpu-utilization.jsonl'
    gpu_rows=[];telemetry_error=None
    if not telemetry.exists():
        telemetry_error='Dedicated GPU telemetry file is unavailable'
    else:
        try:
            gpu_rows=rows(telemetry)
            if not gpu_rows:
                telemetry_error='Dedicated GPU telemetry file is empty'
        except (OSError,ValueError) as error:
            telemetry_error=f'Cannot read dedicated GPU telemetry: {type(error).__name__}: {error}'
    concurrency={};gpu={}
    for name,window in windows.items():
        if window['status']!='measured':
            concurrency[name]=dict(status='unavailable',reason=window['reason'])
            gpu[name]=dict(status='unavailable',reason=window['reason'])
            continue
        concurrency[name]=(dict(status='unavailable',reason=interval_error) if interval_error
                           else concurrency_summary(intervals,window))
        if telemetry_error:
            gpu[name]=dict(status='unavailable',reason=telemetry_error)
            continue
        try:
            sample=summarize_gpu(gpu_rows,window['start_epoch'],window['end_epoch'])
            # A sample is evidence only for its bounded coverage. Preserve
            # errors, external PIDs and uncovered time instead of filling zeroes.
            available=any('gpu_utilization_percent' in device['metrics']
                          for device in sample['devices'].values())
            gpu[name]=(dict(status='sampled',**sample) if available else
                       dict(status='unavailable',reason='No device utilization samples cover this window',**sample))
        except (KeyError,TypeError,ValueError) as error:
            gpu[name]=dict(status='unavailable',reason=f'Cannot summarize GPU telemetry: {type(error).__name__}: {error}')
    return dict(windows=windows,active_agent_concurrency=concurrency,gpu=gpu,
        gpu_source='gpu-utilization.jsonl',
        scope='Formal window includes task setup and verification; Agent window includes tool waits. '
              'Both exclude model startup, synthetic preflight and smoke. GPU utilization is whole-device busy time.')


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
    require(timing_mode in ('official','none','relaxed'),'Unknown Agent timing policy')
    if timing_mode in ('none','relaxed'):
        policy=read(root/'agent-time-policy.json')
        require(policy['agent_time_limit']==timing_mode and set(policy['tasks'])==names,'Missing Agent timing audit')
        if timing_mode=='relaxed':
            require(policy.get('agent_multiplier')==2.0,'Invalid relaxed Agent multiplier')
            hook=read(root/'formal-deadline-hook.json')
            require(hook.get('agent_time_limit')=='relaxed' and hook.get('original_phase_sha256'),
                    'Missing pause-aware Agent deadline hook')
        for name in names:
            task_path=root/'dataset/tasks'/name/'task.toml'
            require(sha(task_path)==policy['tasks'][name]['effective_sha256'],'Task timing audit mismatch')
            expected_timeout=None
            if timing_mode=='relaxed':
                try:
                    from .terminalbench_policy import relaxed_agent_timeout
                except ImportError:
                    from terminalbench_policy import relaxed_agent_timeout
                expected_timeout=relaxed_agent_timeout(policy['tasks'][name]['original_agent_timeout_sec'])
            actual=tomllib.loads(task_path.read_text()).get('agent',{}).get('timeout_sec')
            require(actual==expected_timeout and policy['tasks'][name]['effective_agent_timeout_sec']==expected_timeout,
                    'Agent deadline differs from timing policy')
        job=read(root/'job.json')
        require(job.get('timeout_multiplier',1)==1 and job.get('agent_timeout_multiplier') in (None,1),
                'Agent deadline multiplied outside task timing audit')
        for agent in job['agents']:
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
            require(plan['max_tokens']==manifest.get('output_cap',131072),'Actual output cap changed')
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
    efficiency=efficiency_metrics(root,lifecycle)
    wall=efficiency['windows']['agent'].get('seconds')
    durations=[]
    for r in summary['outcomes']:
        result=read(root/'formal-events'/(r['trial']+'.json'))
        duration=None;phase=result.get('agent_execution') or {}
        if phase.get('started_at') and phase.get('finished_at'):
            duration=(datetime.fromisoformat(phase['finished_at'])-datetime.fromisoformat(phase['started_at'])).total_seconds()
        durations.append(dict(**r,agent_seconds=duration,usage=result.get('agent_result')))
    resources=rows(root/'resources.jsonl');archive=read(root/'model/server/capture/archive.json')
    boundary=read(root/'formal-boundary.json') if (root/'formal-boundary.json').exists() else {}
    before_gc=boundary.get('capture',{}).get('archive',{}).get('historical_gc')
    final_gc=archive.get('historical_gc') or {}
    gc_counters=('retired','redundancy_retired','lru_fallback_retired','lru_representative_retired',
                 'lossy_retired','cpu_released','slots_released','thinking_classified','thinking_pure',
                 'policy_failures','decisions','examined','ticks',
                 'thinking_age_retired','thinking_age_ineligible','thinking_age_passes')
    formal_gc=({key:final_gc.get(key,0)-before_gc.get(key,0) for key in gc_counters}
               if before_gc is not None else None)
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
        subset=expected!=89,benchmark_total_tasks=89,
        agent_time_limit=timing_mode,official_timing_comparable=(timing_mode=='official'),
        actual_requests=request_count,accepted_completed_response_tokens=accepted,
        actual_seed=1000,output_cap=manifest.get('output_cap',131072),thinking_budget=4096,
        exact_prefix_reused_requests=reuse,finish_reasons=dict(finishes),
        incomplete_native_requests=token_failures,cleanup=cleanup,
        task_containers_retained=containers,gc=archive.get('historical_gc'),
        gc_formal_delta=formal_gc,gc_totals_include_warmup_and_preflight=True,
        gc_release_scope='Formal cpu_released/slots_released include normal task-end cleanup; '
                         'retired and its policy breakdown measure strategy deletion.',
        cpu_memory_available_min=min((r['memory_available'] for r in resources),default=None),
        gpu_memory_mib_max=max((r.get('gpu_memory_mib',0) for r in resources),default=None),
        agent_window_seconds=wall,accepted_completed_response_tokens_per_wall_second=accepted/wall if wall else None,
        formal_window_seconds=efficiency['windows']['formal'].get('seconds'),efficiency=efficiency)
    save(root/'execution-audit.json',audit)
    save(root/'results.json',dict(**summary,tasks=durations,measurement=audit))
    evidence={}
    for folder in ('jobs','formal-events','smoke-events','bridge','model','images'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and not p.is_symlink():evidence[str(p.relative_to(root))]=sha(p)
    for name in ('gpu-utilization.jsonl','resources.jsonl','formal-boundary.json'):
        if (root/name).is_file():evidence[name]=sha(root/name)
    save(root/'evidence-hashes.json',evidence)
    profile=read(root/'profile.json') if (root/'profile.json').exists() else {}
    kv=profile.get('kvmem',{})
    gc=kv.get('historical_gc') or {}
    geometry=(f'K{kv.get("select_budget", "unspecified")}/G{kv.get("gen_budget", "unspecified")}; '
              f'logical limit {profile.get("max_model_len", "unspecified")}; '
              f'history GC {gc.get("policy", "disabled")} / {gc.get("mode", "off")}.')
    text=[f'Terminal-Bench 2.1 / Terminus 2: {summary["passed"]}/{expected} passed in this run.',
          f'Explicit {expected}-task subset; not a full 89-task benchmark score.' if expected!=89 else 'Full official 89-task set.',
          f'Scored: {summary["scored"]}; task timeout without score: {summary["task_timeouts"]}; other unscored errors: {summary["errors"]}.',
          'This campaign contains one attempt per selected official task; no best-of selection. All raw errors and timeouts retained; earlier campaigns, if any, are reported separately.',
          f'Completed responses: {accepted:,} accepted output tokens; agent execution window: {wall:.3f} seconds.' if wall else 'No complete agent timing window.',
          'This rate excludes tokens from incomplete responses; interrupted request counts are listed in the audit.',
          f'Frozen sources/configurations verified: {checked}; evidence files hashed: {len(evidence)}.',
          geometry,
          'Terminus automatic summaries disabled; reasoning retained across turns; official resources/verifiers unchanged.',
          {'none': 'Agent execution deadline disabled: custom local evaluation, not directly comparable with official timed scores. Verifier/setup/tool timeouts retained.',
           'relaxed': 'Agent deadline 2x official, recorded pauses excluded: custom local evaluation, not directly comparable with official timed scores. Verifier/setup/tool timeouts retained.',
           'official': 'Official task Agent deadlines retained.'}[timing_mode],
          f'Maximum {manifest.get("max_concurrency",8)} concurrent tasks; container memory admission capped at 24 GiB without reducing any official task allocation.',
          'Synthetic smoke, thinking preflight, fixed warmup and session release requests are excluded from task scores and formal throughput.',
          ('Formal GC deletions: '+str(formal_gc['retired'])+'; redundancy '+str(formal_gc['redundancy_retired'])+
           '; LRU fallback '+str(formal_gc['lru_fallback_retired'])+' (including '+
           str(formal_gc['lru_representative_retired'])+' older representatives).'
           if formal_gc is not None else 'Formal GC baseline unavailable; cumulative GC totals include setup work.'),
          'CPU/slot release counters also include ordinary task-end cleanup; they are not policy deletion totals.']
    for name,window in efficiency['windows'].items():
        if window['status']=='measured':
            text.append(f'{name.capitalize()} measured window: {window["seconds"]:.3f} seconds.')
        concurrency=efficiency['active_agent_concurrency'][name]
        if concurrency['status']=='measured':
            text.append(f'{name.capitalize()} window active Agents: peak {concurrency["peak"]}; '
                        f'time-weighted mean {concurrency["time_weighted_mean"]:.3f} (includes tool waits).')
        else:
            text.append(f'{name.capitalize()} window Agent concurrency unavailable: {concurrency["reason"]}.')
        gpu=efficiency['gpu'][name]
        if gpu['status']=='unavailable':
            text.append(f'{name.capitalize()} window GPU telemetry unavailable: {gpu["reason"]}.')
        else:
            for uuid,device in gpu['devices'].items():
                metric=device['metrics'].get('gpu_utilization_percent')
                if metric:
                    text.append(f'{name.capitalize()} window GPU {uuid}: mean utilization {metric["mean"]:.2f}%; '
                                f'coverage {metric["coverage_fraction"]:.2%}; whole-device busy time.')
            text.append(f'{name.capitalize()} window GPU sample errors: {len(gpu["errors"])}; '
                        f'observed external GPU PIDs: {gpu["external_pids"]}.')
    text.extend(['', 'Task | Reward | Exception | Agent seconds'])
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
