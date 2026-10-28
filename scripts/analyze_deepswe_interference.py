"""Accepted-output and tool accounting for preserved natural observation windows."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.analyze_deepswe_capacity import records, metrics, delta, distribution, segment
from scripts.profile_deepswe_capacity import step_groups, restore_groups


def prefix_rows(path, count):
    data=path.read_bytes()[:count]
    end=data.rfind(b'\n')
    return [json.loads(s) for s in data[:end+1].splitlines()] if end>=0 else []


def tool_counts(rows):
    calls={};results={}
    for row in rows:
        if row.get('type') not in ('assistant','user'):continue
        content=row.get('message',{}).get('content',[])
        if not isinstance(content,list):continue
        for item in content:
            if not isinstance(item,dict):continue
            if item.get('type')=='tool_use':calls.setdefault(item['id'],item.get('name','unknown'))
            elif item.get('type')=='tool_result':results.setdefault(item['tool_use_id'],bool(item.get('is_error')))
    counts=Counter(calls.get(key,'unknown') for key,value in results.items() if value)
    return dict(calls=len(calls),results=len(results),errors=sum(results.values()),errors_by_tool=dict(counts),
                pending=len(set(calls)-set(results)))


def analyze(directory):
    state=json.loads((directory/'run.json').read_text())
    boundary=json.loads((directory/'window-end-metrics.json').read_text())
    if state['status']!='window_complete' or state.get('cleanup_errors'):
        raise ValueError('Natural observation window/cleanup did not pass')
    begin=state['model_ready_at'];end=(boundary['started_at']+boundary['finished_at'])/2
    before=metrics((directory/'metrics-before-formal.txt').read_text());after=metrics(boundary['raw_metrics'])
    accepted=delta(before,after,'vllm:generation_tokens_total')
    if accepted is None or end<=begin:raise ValueError('Missing accepted-output counter or elapsed time')
    samples=[dict(at=begin,raw_metrics=(directory/'metrics-before-formal.txt').read_text())]
    samples += [r for r in records(directory/'resources.jsonl') if begin<=r['at']<end]
    samples.append(dict(at=end,raw_metrics=boundary['raw_metrics']))
    native=segment(samples,begin,end)
    phases=[dict(elapsed_begin=a,elapsed_end=b,metrics=segment(samples,begin+a,min(end,begin+b)))
            for a,b in ((0,300),(300,600),(600,900))]
    bridge=[r for r in prefix_rows(directory/'bridge/requests.jsonl',boundary['bridge_bytes']) if begin<=r.get('at',0)<=end]
    event_counts=dict(Counter(r.get('event') for r in bridge))
    finishes=[r for r in bridge if r.get('event')=='finish']
    starts=[r for r in bridge if r.get('event')=='start']
    tasks={rel.split('/')[1]:tool_counts(prefix_rows(directory/rel,count)) for rel,count in boundary['stream_bytes'].items()}
    offset=state['model_ready_at']-state['model_ready_monotonic']
    steps=[r for r in records(directory/'server/capture/model_step_latency.jsonl') if begin<=r['started_at']+offset<=end]
    restores=[r for r in records(directory/'server/capture/async_restore.jsonl') if begin<=r['finished']+offset<=end]
    eight=[r for r in steps if r['decode_requests']==8 and not r['prefill_tokens']]
    mixed=[r for r in steps if r['decode_requests'] and r['prefill_tokens']]
    pure=[r for r in steps if r['decode_requests'] and not r['prefill_tokens']]
    ttl=sum(t['results'] for t in tasks.values());errors=sum(t['errors'] for t in tasks.values())
    gpu=[]
    for row in samples:
        if row.get('gpu'):
            parts=row['gpu'].split(',')
            if len(parts)==2:
                try:gpu.append(float(parts[1]))
                except ValueError:pass
    return dict(begin=begin,end=end,seconds=end-begin,metric_capture_uncertainty_seconds=boundary['finished_at']-boundary['started_at'],
        accepted_output_tokens=accepted,agent_tokens_per_second=accepted/(end-begin),native=native,
        elapsed_phase_metrics=phases,
        tool_calls=sum(t['calls'] for t in tasks.values()),tool_results=ttl,tool_errors=errors,
        tool_error_rate=errors/ttl if ttl else None,tasks=tasks,bridge_events=event_counts,
        completed_model_requests=len(finishes),
        completed_request_output_tokens=sum(r.get('usage',{}).get('completion_tokens',0) for r in finishes),
        request_metrics={key:distribution([(r.get('metrics') or {}).get(key) for r in finishes])
            for key in ('time_to_first_token_ms','generation_time_ms','queue_time_ms')},
        context_tokens=distribution([r.get('prompt_tokens') for r in starts]),
        continuation_prefix_misses=sum(r.get('turn',0)>1 and not r.get('exact_output_prefix_reused') for r in starts),
        continuation_requests=sum(r.get('turn',0)>1 for r in starts),
        prepare_ms=distribution([1000*r['prepare_seconds'] for r in starts if r.get('prepare_seconds') is not None]),
        gpu_utilization_sampled=distribution(gpu),
        minimum_host_available_bytes=min((r['available_bytes'] for r in samples if 'available_bytes' in r),default=None),
        step_groups=step_groups(steps),restore_groups=restore_groups(restores),
        pure_eight_decode_steps=len(eight),pure_eight_model_ms=distribution([r['model_ms'] for r in eight]),
        mixed_model_ms=distribution([r['model_ms'] for r in mixed]),
        pure_decode_model_ms=distribution([r['model_ms'] for r in pure]),
        foreground_ms=distribution([r['maintenance_host_ms'] for r in steps]),
        foreground_steps_over_100ms=sum(r['maintenance_host_ms']>=100 for r in steps),
        scope='Native accepted output / wall including tools. Sampled GPU utilization is not exact idle time. '
              'Accepted generation includes partial requests still in flight at the boundary; completed-request output is reported separately. '
              'Completed-request timings exclude requests still in flight. Tool counts use complete records within boundary file offsets. '
              'Snapshot/cleanup and cancelled requests after the boundary are excluded. Unfinished tasks are unscored.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arms={name:analyze(a.root/name) for name in ('control','candidate')}
    result=dict(arms=arms,observed_throughput_change_percent=100*(arms['candidate']['agent_tokens_per_second']/arms['control']['agent_tokens_per_second']-1),
        limitation='Single sequential 15-minute natural windows; task/output/tool trajectories differ. Not a complete DeepSWE score or a precise causal speedup estimate.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
