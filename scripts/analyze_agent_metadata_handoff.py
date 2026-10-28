"""Audit and compare fixed-history restore components without claiming Agent speedup."""
import argparse
import json
from pathlib import Path

from scripts import benchmark_agent_metadata_handoff as benchmark
from scripts import run_agent_handoff_queue as q


def compare(before, after):
    return dict(before=before, after=after, saved=before-after,
                reduction_percent=100*(before-after)/before if before else None)


def analyze(root, *, validator=None):
    (validator or benchmark.validate)(root)
    fixture = q.read(root/'fixture.json')
    metrics, paired = {}, {}
    for name in ('before', 'after'):
        arm = root/'optimization'/name
        audit, cleanup = q.read(arm/'execution-audit.json'), q.read(arm/'cleanup.json')
        if (q.read(arm/'status.json')['status'] != 'completed' or audit['status'] != 'passed' or
                audit['responses'] != 64 or audit['restores'] != 56 or
                cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']):
            raise ValueError('Component has not completed audit and cleanup')
        evidence = q.read(arm/'evidence-hashes.json')
        if len(evidence) != 64 or any(q.sha(arm/rel) != digest for rel,digest in evidence.items()):
            raise ValueError('Answer evidence changed')
        answers = {(i,t):q.read(arm/'answers'/f'{i}-{t}.json') for i in range(8) for t in range(8)}
        restores = [json.loads(line) for line in (arm/'run/server/capture/async_restore.jsonl').read_text().splitlines()]
        if len(restores) != 56 or not all(r['resident_active_tokens'] and r['model_cache_hit'] for r in restores):
            raise ValueError('Resident restore workload differs')
        requests = {row['response']['id']:key for key,row in answers.items()}
        by_pair = {}
        for row in restores:
            matches = [key for rid,key in requests.items() if row['request_id'].startswith(rid+'-')]
            if len(matches) != 1 or matches[0][1] == 0 or matches[0] in by_pair:
                raise ValueError('Restore cannot be uniquely paired with a warm request')
            by_pair[matches[0]] = row
        for key, row in answers.items():
            i,t = key
            expected = benchmark.fixed_input(fixture[i], t)
            digest = benchmark.hashlib.sha256(json.dumps(expected).encode()).hexdigest()
            if (row['input_sha256'] != digest or row['response']['choices'][0]['token_ids'] != [fixture[i]['output_token']]*64):
                raise ValueError('Fixed input/output mismatch')
        turn_wall = [max(answers[i,t]['started']+answers[i,t]['seconds'] for i in range(8))-
                     min(answers[i,t]['started'] for i in range(8)) for t in range(1,8)]
        steps = [json.loads(line) for line in (arm/'run/server/capture/model_step_latency.jsonl').read_text().splitlines()]
        warm_ids = set(r['request_id'] for r in restores)
        warm_steps = [s for s in steps if s['batch_geometry'] and
                      all(x['request_id'] in warm_ids for x in s['batch_geometry'])]
        metrics[name] = dict(
            warm_request_seconds=q.distribution(a['seconds'] for (i,t),a in answers.items() if t),
            restore_seconds=q.distribution(r['finished']-r['started'] for r in restores),
            warm_round_wall_seconds=q.distribution(turn_wall), warm_round_wall_sum_seconds=sum(turn_wall),
            warm_model_ms=q.distribution(s['model_ms'] for s in warm_steps),
            warm_maintenance_host_ms=q.distribution(s['maintenance_host_ms'] for s in warm_steps),
            cold_request_seconds=q.distribution(a['seconds'] for (i,t),a in answers.items() if not t),
            uploaded_kv_bytes=sum(r['uploaded_bytes'] for r in restores),
            uploaded_model_bytes=sum(r['model_uploaded_bytes'] for r in restores),
            evidence_sha256=q.sha(arm/'evidence-hashes.json'))
        paired[name] = {f'{i}-{t}':dict(case=fixture[i]['case'], turn=t, input_sha256=answers[i,t]['input_sha256'],
            request_seconds=answers[i,t]['seconds'], restore_seconds=r['finished']-r['started'])
            for (i,t),r in by_pair.items()}
    changes = {key:{stat:compare(metrics['before'][key][stat],metrics['after'][key][stat])
                    for stat in ('mean','median','p95')}
               for key in ('warm_request_seconds','restore_seconds','warm_round_wall_seconds',
                           'warm_model_ms','warm_maintenance_host_ms','cold_request_seconds')}
    changes['warm_round_wall_sum_seconds'] = compare(metrics['before']['warm_round_wall_sum_seconds'],
                                                    metrics['after']['warm_round_wall_sum_seconds'])
    return dict(component_only=True, inputs_equal=True, paired_requests=56, arms=metrics, changes=changes,
        pairs={key:dict(case=paired['before'][key]['case'], turn=paired['before'][key]['turn'],
            request_seconds=compare(paired['before'][key]['request_seconds'],paired['after'][key]['request_seconds']),
            restore_seconds=compare(paired['before'][key]['restore_seconds'],paired['after'][key]['restore_seconds']))
            for key in sorted(paired['before'])},
        limitations=['Single run per arm with identical inputs and constrained outputs; no confidence interval from independent repeats.',
            'No tools execute in this component. Its decode speed or latency ratio is not end-to-end Agent throughput or quality.',
            'Restore intervals overlap and include scheduling; sums are not GPU-blocked time.',
            'The original Agent baseline remains interrupted and incomplete at its predeclared cap. No full-wall causal A/B claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    q.save(args.output, analyze(args.root.resolve()))
