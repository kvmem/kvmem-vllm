"""Compare uninstrumented tool-return components, never label them Agent scores."""
import argparse
import json
from pathlib import Path
import statistics

from scripts.analyze_host_interference import distribution, read_rows


def arm_result(directory):
    rows = [json.loads(p.read_text()) for p in sorted((directory/'answers').glob('*.json'))]
    if not rows or any(row['error'] for row in rows):
        raise ValueError('Missing or failed responses')
    turns = sorted(set(r['turn'] for r in rows if r['turn']))
    rounds = []
    for turn in turns:
        current = [r for r in rows if r['turn'] == turn]
        if len(current) != 8 or len({r['owner'] for r in current}) != 8:
            raise ValueError('Incomplete owner cohort')
        returning = next(r for r in current if r['owner'] == 7)
        peers = [r for r in current if r['owner'] != 7]
        at = returning['started']
        active = [r for r in peers if r['events'][0][0] < at < r['events'][-1][0]]
        gaps = [(b[0]-a[0])*1000 for r in peers for a,b in zip(r['events'], r['events'][1:])
                if b[0] >= at and a[0] <= returning['events'][0][0]]
        all_gaps = [(b[0]-a[0])*1000 for r in peers for a,b in zip(r['events'], r['events'][1:])]
        elapsed = max(r['finished'] for r in current)-min(r['started'] for r in current)
        rounds.append(dict(turn=turn, suffix=(512 if len(turns)==1 else [32,512,2048,8192][(turn-1)%4]),
            peers_decoding_at_return=len(active), wall_seconds=elapsed,
            accepted_tokens=sum(len(r['output_ids']) for r in current),
            return_ttft_seconds=returning['events'][0][0]-at,
            return_latency_seconds=returning['finished']-at,
            peer_gap_during_return_ms=distribution(gaps), peer_gap_all_ms=distribution(all_gaps),
            peer_gaps_over_100ms=sum(g>100 for g in gaps),
            started=min(r['started'] for r in current),finished=max(r['finished'] for r in current)))
    capture=directory/'run/server/capture'
    steps=read_rows(capture/'model_step_latency.jsonl')
    start,end=min(r['started'] for r in rounds),max(r['finished'] for r in rounds)
    warm=[r for r in steps if start<=r['started_at']<=end]
    step_groups={}
    for key,choose in [('pure_decode',lambda r:not r['prefill_tokens'] and r['decode_requests']),
                       ('mixed',lambda r:r['prefill_tokens'] and r['decode_requests'])]:
        group=[r for r in warm if choose(r)]
        step_groups[key]={field:distribution(r[field] for r in group if r.get(field) is not None) for field in
                         ['model_ms','maintenance_host_ms','previous_end_to_model_ms']}
        step_groups[key]['decode_requests']=distribution(r['decode_requests'] for r in group)
        step_groups[key]['prefill_tokens']=distribution(r['prefill_tokens'] for r in group)
    restores=read_rows(capture/'async_restore.jsonl')
    checkpoint_path=capture/'checkpoint_timing.jsonl'
    checkpoints=read_rows(checkpoint_path) if checkpoint_path.exists() else []
    valid=[r for r in rounds if r['peers_decoding_at_return']==7]
    return dict(rounds=rounds, full_seven_peer_rounds=len(valid), total_rounds=len(rounds),
        warm_round_wall_seconds=sum(r['wall_seconds'] for r in rounds),
        accepted_output_tokens=sum(r['accepted_tokens'] for r in rounds),
        component_output_tokens_per_second=sum(r['accepted_tokens'] for r in rounds)/sum(r['wall_seconds'] for r in rounds),
        step_groups=step_groups,
        foreground_steps_over_100ms=sum(r['maintenance_host_ms']>=100 for r in warm),
        foreground_ms=distribution(r['maintenance_host_ms'] for r in warm),
        checkpoint_timing_available=checkpoint_path.exists(),
        checkpoints_ms=distribution(1000*r['host_total_seconds'] for r in checkpoints if start<=r['at']<=end),
        restores=dict(count=len(restores),resident=sum(r.get('lane')=='resident' for r in restores),
            uploaded_bytes=sum(r.get('uploaded_bytes',0) for r in restores),
            seconds=distribution(r['finished']-r['started'] for r in restores)),
        input_hashes={f'{r["turn"]}-{r["owner"]}':r['input_hash'] for r in rows},
        output_hash_inputs={f'{r["turn"]}-{r["owner"]}':r['output_ids'] for r in rows})


def compare(root, names):
    arms={name:arm_result(root/name) for name in names}
    reference=arms[names[0]]
    expected_output=reference['output_hash_inputs']
    for value in arms.values():
        if value['input_hashes'] != reference['input_hashes'] or value['output_hash_inputs'] != expected_output:
            raise ValueError('Paired input/output trajectory changed')
        del value['output_hash_inputs']
    for name,value in arms.items():
        value['wall_change_percent']=100*(value['warm_round_wall_seconds']/reference['warm_round_wall_seconds']-1)
    return dict(arms=arms,scope='Constrained fixed-history component. SSE inter-event gaps include delivery and may contain multiple accepted MTP tokens. '
        'CUDA model intervals include target/draft/capture stream work. Not a real Agent quality or throughput score. '
        'Three workload repetitions within each arm; arm order was not randomized/interleaved. '
        'Rounds lacking seven simultaneously decoding peers are reported, never silently filtered or rerun.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arms',nargs='+',default=['control','capture','cap512','cap256'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.root,args.arms),indent=2)+'\n')
