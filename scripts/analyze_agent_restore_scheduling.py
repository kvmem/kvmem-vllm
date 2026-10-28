"""Join immutable restore/scheduler/public-forward traces without GPU-idle claims.

Admission arrival is after session lookup, not HTTP arrival. CUDA event gaps
exclude public model intervals but can contain transfers and private forwards.
Concurrent per-request latencies are never treated as exclusive wall time.
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from types import SimpleNamespace


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(.95), maximum=values[-1])


def summarize_requests(rows):
    return dict(count=len(rows), deferred=sum(row['deferred'] for row in rows),
                post_restore_private_refresh=sum(row['post_restore_private_refresh'] for row in rows),
                phase_ms={name:distribution([r['phase_ms'][name] for r in rows])
                          for name in rows[0]['phase_ms']} if rows else {},
                peer_batches_submitted_during_restore=distribution([r['peer_batches'] for r in rows]),
                restores_with_peer_batches=sum(r['peer_batches'] > 0 for r in rows))


def analyze(capture, *, common_end_wall=None):
    names = ['restore_admission.jsonl', 'async_restore.jsonl', 'async_refresh.jsonl',
             'model_step_latency.jsonl', 'public_work.json', 'checkpoint_timing.jsonl']
    if (capture/'restore_host_stages.jsonl').exists():
        names.append('restore_host_stages.jsonl')
    hashes = {name:digest(capture/name) for name in names}
    work = json.loads((capture/'public_work.json').read_text())
    clock_offset = work['created_wall_time'] - work['created_at']
    end = common_end_wall - clock_offset if common_end_wall is not None else float('inf')
    admissions, events, host = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    private_refresh = defaultdict(list)
    for row in records(capture/'restore_admission.jsonl'):
        # Deferred notices can repeat if a reason changes. First arrival/admission
        # and ready must remain unique for each native request identity.
        own = admissions[row['request_id']]
        if row['stage'] != 'deferred' and row['stage'] in own:
            raise ValueError('Repeated admission milestone')
        own[row['stage']] = row['at']
    for row in records(capture/'async_refresh.jsonl'):
        if row['stage'] == 'suspended':
            private_refresh[row['request_id']].append(row['at'])
        if row['stage'].startswith('restore_'):
            own = events[row['request_id']]
            if row['stage'] in own:
                raise ValueError('Repeated restore milestone')
            own[row['stage']] = row['at']
    if 'restore_host_stages.jsonl' in names:
        for row in records(capture/'restore_host_stages.jsonl'):
            total = [s for s in row['timing']['stages'] if s['name']=='restore_main_thread_total']
            if len(total) != 1 or row['phase'] in host[row['request_id']]:
                raise ValueError('Ambiguous host stage interval')
            host[row['request_id']][row['phase']] = total[0]
    first, steps, previous = {}, [], None
    for row in records(capture/'model_step_latency.jsonl'):
        if previous is not None and (row['call_index'] != previous['call_index'] + 1 or
                                     row['previous_call_index'] != previous['call_index']):
            raise ValueError('CUDA gaps cannot partition subsampled public steps')
        for owner in row['batch_geometry']:
            first.setdefault(owner['request_id'], dict(started_at=row['started_at'],
                model_submitted_at=row['model_submitted_at'], sequence=row['sequence']))
        steps.append({k:row[k] for k in ('sequence','started_at','model_submitted_at',
            'decode_requests','prefill_requests','prefill_tokens','model_ms',
            'previous_end_to_model_ms','maintenance_host_ms')})
        previous = row
    model_submits = [s['model_submitted_at'] for s in steps]
    if model_submits != sorted(model_submits):
        raise ValueError('Public submission order changed')
    joined = []
    for row in records(capture/'async_restore.jsonl'):
        rid = row['request_id']
        if row['cancelled']:
            raise ValueError('Cancelled restore requires separate analysis')
        a, e, f = admissions[rid], events[rid], first[rid]
        milestones = [('admission_queue', a['arrival'], a['admitted']),
            ('admitted_to_worker', a['admitted'], row['started']),
            ('worker_to_background_submit', row['started'], e['restore_submitted']),
            ('background_submit_to_worker_ready', e['restore_submitted'], e['restore_ready']),
            ('worker_ready_to_scheduler_ack', e['restore_ready'], a['ready']),
            ('scheduler_ack_to_public_step', a['ready'], f['started_at']),
            ('public_step_to_model_submit', f['started_at'], f['model_submitted_at'])]
        if any(b < a for _,a,b in milestones):
            raise ValueError('Cross-process milestones out of order')
        phases = {name:1000*(b-a) for name,a,b in milestones}
        total = 1000*(f['model_submitted_at'] - a['arrival'])
        if abs(sum(phases.values()) - total) > .001:
            raise ValueError('Timeline does not partition request wait')
        phases['total_admission_arrival_to_model_submit'] = total
        item = dict(request_id=rid, resident=bool(row.get('resident_active_tokens')),
            deferred='deferred' in a, common_eight_period=f['model_submitted_at'] < end,
            post_restore_private_refresh=any(a['ready'] <= at < f['started_at']
                                            for at in private_refresh[rid]),
            history_tokens=row['history_tokens'], phase_ms=phases,
            milestones={name:dict(begin=a,end=b) for name,a,b in milestones},
            first_public_sequence=f['sequence'], uploaded_bytes=row['uploaded_bytes'],
            model_uploaded_bytes=row.get('model_uploaded_bytes'),
            peer_batches=bisect_left(model_submits,e['restore_ready']) - bisect_left(model_submits,row['started']))
        if host:
            h = host[rid]
            p, m, c = [h[name] for name in ('prepare','restore_model_prepare','restore_copy')]
            item['host_detail_ms'] = dict(
                suspended_to_prepare=1000*(p['started_at']-row['started']),
                prepare=1000*p['wall_seconds'],
                prepare_to_model_prepare=1000*(m['started_at']-p['finished_at']),
                model_prepare=1000*m['wall_seconds'],
                model_prepare_end_to_commit=1000*(c['started_at']-m['finished_at']),
                commit=1000*c['wall_seconds'],
                outer_host_total=1000*sum(t['wall_seconds'] for t in h.values()))
        joined.append(item)
    def step_summary(selected):
        return dict(count=len(selected), model_interval_seconds=sum(s['model_ms'] for s in selected)/1000,
            preceding_nonpublic_model_interval_seconds=sum(s['previous_end_to_model_ms'] or 0 for s in selected)/1000,
            preceding_gap_ms=distribution([s['previous_end_to_model_ms'] for s in selected
                                          if s['previous_end_to_model_ms'] is not None]),
            maintenance_host_ms=distribution([s['maintenance_host_ms'] for s in selected]))
    resident = [r for r in joined if r['resident']]
    summary = dict(all=summarize_requests(joined), resident=summarize_requests(resident))
    if common_end_wall is not None:
        summary['resident_common_eight_period'] = summarize_requests([r for r in resident if r['common_eight_period']])
        summary['resident_common_eight_without_post_restore_refresh'] = summarize_requests([
            r for r in resident if r['common_eight_period'] and not r['post_restore_private_refresh']])
    if host:
        summary['host_detail_ms'] = {name:distribution([r['host_detail_ms'][name] for r in joined])
                                    for name in joined[0]['host_detail_ms']}
    # Local aggregate files must not change underneath this offline analysis.
    if hashes != {name:digest(capture/name) for name in names}:
        raise ValueError('Input evidence changed during analysis')
    return dict(capture=str(capture.resolve()), input_hashes=hashes, clock_offset=clock_offset,
        common_eight_cutoff_wall=common_end_wall, summaries=summary, requests=joined,
        public_steps=dict(all=step_summary(steps),
            common_eight_period=step_summary([s for s in steps if s['started_at']<end]),
            exactly_eight_decode=step_summary([s for s in steps if s['decode_requests']==8 and not s['prefill_tokens']])),
        scope='Read-only joins. Admission arrival is AFTER session lookup, not HTTP arrival. '
              'Per-request waits overlap and cannot be summed as GPU idle. Peer batches count host public-model '
              'submissions during the request restore; no inference about continuous GPU occupancy. '
              'CUDA gaps contain maintenance, transfers, private model work, scheduler and client waits; '
              'model intervals include launch gaps. Neither is an SM utilization measurement. '
              'The fixed-history barrier workload differs from naturally staggered Agent requests.')


def policy_probe(component):
    """Run the frozen priority function on explicit states; no model/GPU work."""
    source = component/'runtime/src/vllm_kvmem/maintenance.py'
    spec = importlib.util.spec_from_file_location('frozen_restore_maintenance', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.MaintenancePolicy.__new__(module.MaintenancePolicy)
    policy.clock = lambda:10.
    policy.aging_seconds, policy.offline = .25, False
    policy.readiness_costs, policy.upload_ratios = {}, {}
    def job(name, stage, size, last):
        return SimpleNamespace(command=SimpleNamespace(resident=True, request_id=name),
            model_bytes=size, stage=stage, done=False, cancelled=False,
            last_progress=last, started=9.9)
    prepared = job('prepared_model_submit', 'restore_model_prepare',157876224,9.96)
    unprepared = job('unprepared', 'prepare',0,9.97)
    host = defaultdict(dict)
    for row in records(component/'run/server/capture/restore_host_stages.jsonl'):
        host[row['request_id']][row['phase']] = next(
            s for s in row['timing']['stages'] if s['name']=='restore_main_thread_total')
    delayed = []
    for rid, stages in host.items():
        begin, end = stages['prepare']['finished_at'], stages['restore_model_prepare']['started_at']
        other = [key for key,value in host.items() if key!=rid and begin<=value['prepare']['started_at']<end]
        if other:
            delayed.append(dict(request_id=rid, wait_ms=1000*(end-begin), other_prepares=other))
    return dict(source=str(source.resolve()),source_sha256=digest(source),
        state=dict(prepared_model_bytes=prepared.model_bytes,unprepared_model_bytes=unprepared.model_bytes,
                   below_aging_threshold=True),
        priority_order=[j.command.request_id for j in policy.order([prepared,unprepared])],
        estimated_costs={j.command.request_id:policy.estimate(j) for j in (prepared,unprepared)},
        delayed_with_other_prepare=delayed, total_restores=len(host),
        scope='Exact frozen priority-function reproduction. Trace ordering is consistent with this rule. '
              'The waiting interval also includes peer work and engine iterations; not an exclusive cost '
              'of the priority rule or a measured speedup from changing it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    states = [json.loads(p.read_text()) for p in (args.agent/'cases').glob('*/status.json')]
    if len(states) != 8 or any(s['status']!='completed' for s in states):
        raise ValueError('Require the complete eight-task candidate')
    cutoff = min(s['completed_at'] for s in states)
    report = dict(agent=analyze(args.agent/'server/capture', common_end_wall=cutoff),
                  component=analyze(args.component/'run/server/capture'),
                  priority_probe=policy_probe(args.component))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({name:data['summaries'] for name,data in report.items() if 'summaries' in data}, indent=2))


if __name__ == '__main__':
    main()
