"""Freeze and grade completed layerwise MAB answers while inference continues."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import regrade_flash as f
from grade_mab_snapshot import EVALUATORS, verify
from graph_v2_protocol import require
from run_layerwise_memory import memory_pressure_audit
from benchmark_client_io import result_events

REPO = Path(__file__).resolve().parents[1]


def freeze(campaign, output, evaluation):
    require(not output.exists(), 'Use a fresh grading directory')
    fixture = f.read(campaign/'fixture.json')
    profile = f.read(campaign/'profile.json')
    require(fixture['suite'] == 'memoryagentbench' and fixture['cases'] == 1116
            and not fixture.get('qualification'), 'Expected the formal MAB cohort')
    require(fixture['execution_flags']['layerwise_prefill_pressure'],
            'Expected independent cumulative-attention pressure')
    paths = {p.parent.name: p for p in (campaign/'run/answers/cases').glob('*/result.json')}
    require(set(paths) <= set(fixture['case_ids']), 'Unexpected answer identity')
    chosen = []
    for row in fixture['rows']:
        ident = row['id']
        if ident not in paths:
            continue
        path = paths[ident]
        raw = path.read_bytes()
        result = json.loads(raw)
        require(result['id'] == ident and result['suite'] == 'memoryagentbench'
                and result['status'] == 'completed' and result['original_input_intact']
                and result['prompt_tokens'] == row['input_tokens'], 'Invalid completed answer')
        for key in ('token_sha256', 'prompt_sha256', 'query_sha256'):
            require(result[key] == row[key], 'Original input binding changed: '+key)
        require(result['answer'] == result['raw_completion'].split('</think>', 1)[-1].strip(),
                'Answer extraction changed')
        layers = result['layerwise']['layers']
        require([x['layer'] for x in layers] == list(range(16)), 'Incomplete Query layers')
        require(all(x['mode'] == 'sequential' and x['query_source'] == 'same_forward_current_layer'
                    and x['predecessor_layers_completed'] == x['layer']
                    and x['charged_tokens'] <= profile['kvmem']['select_budget']
                    for x in layers), 'Query execution or Active budget changed')
        event_path = path.parent/'events.json'
        pressures = memory_pressure_audit(result_events(result, path.parent), f.read(event_path), True,
            fixture['execution_flags'].get('layerwise_buffered_history', False),
            profile['kvmem']['select_budget'],
            fixture['execution_flags'].get('layerwise_rolling_decode', False))
        require(len(pressures) == result['pressure_refreshes'], 'Pressure audit mismatch')
        require(result['hit_output_cap'] == (result['finish_reason'] == 'length'),
                'Truncation audit mismatch')
        # Length-limited and incorrect answers are included without any resampling.
        chosen.append((ident, path, raw, event_path))
    require(0 < len(chosen) <= 1116, 'No completed answers')
    output.mkdir(parents=True)
    snapshot = output/'snapshot'
    bindings = {}
    event_bindings = {}
    for ident, path, raw, event_path in chosen:
        destination = snapshot/'cases'/ident/'result.json'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(raw)
        require(path.read_bytes() == raw, 'Answer changed during snapshot')
        bindings[ident] = dict(path=str(path), sha256=f.sha(path), snapshot_sha256=f.sha(destination))
        event_bindings[ident] = dict(path=str(event_path), sha256=f.sha(event_path))
    manifest = campaign/'run/answers/manifest.json'
    shutil.copyfile(manifest, snapshot/'manifest.json')
    reference = Path(f.read(manifest)['reference'])/'results/attention/memoryagentbench/special_judgments.jsonl'
    runtime = output/'runtime'
    shutil.copytree(REPO/'scripts', runtime/'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    f.save(output/'runtime-proof.json', dict(files={str(p.relative_to(runtime)): f.sha(p)
        for p in runtime.rglob('*') if p.is_file()}))
    f.save(output/'bindings.json', bindings)
    f.save(output/'event-bindings.json', event_bindings)
    f.save(output/'plan.json', dict(suite='memoryagentbench', campaign=str(campaign),
        evaluation=str(evaluation), cases=len(chosen), full_cohort_cases=1116,
        partial=len(chosen) < 1116, case_ids=[x[0] for x in chosen], at=time.time(),
        selection='Every completed case present at listing, including length-limited answers',
        limitation='Completion-order subset; full inference execution audit is pending.',
        campaign_proofs={name: f.sha(campaign/name) for name in
            ('fixture.json', 'profile.json', 'runtime-proof.json', 'run/answers/manifest.json')},
        evaluator_proofs={name: f.sha(evaluation/name) for name in EVALUATORS},
        reference_binding=dict(path=str(reference), sha256=f.sha(reference)),
        snapshot_manifest_sha256=f.sha(snapshot/'manifest.json')))
    f.save(output/'prepared-proof.json', {name: f.sha(output/name) for name in
        ('plan.json', 'bindings.json', 'event-bindings.json', 'runtime-proof.json')})
    verify_all(output)
    f.save(output/'status.json', dict(status='prepared', cases=len(chosen), at=time.time()))
    return len(chosen)


def verify_all(output):
    plan = verify(output)
    for binding in f.read(output/'event-bindings.json').values():
        f.unchanged(binding)
    return plan


def run(output, credential_file):
    require(Path(__file__).resolve() == output/'runtime/scripts'/Path(__file__).name,
            'Run the frozen grading script')
    plan = verify_all(output)
    judge = f.load(Path(plan['evaluation'])/'evaluators/historical_lme_score.py', 'mab_snapshot_transport')
    require(judge.API_BASE == 'https://api.deepseek.com', 'Unexpected scoring service')
    flash = f.Flash(output, judge, credential_file.read_text().strip())
    models, _ = flash.client.request('/models')
    require(f.MODEL in {x['id'] for x in models['data']}, 'Original judge unavailable')
    f.save(output/'status.json', dict(status='grading', cases=plan['cases'], at=time.time()))
    os.environ['DEEPSEEK_API_KEY'] = 'transport-managed'
    report = f.mab_arm(output, plan, 'layerwise_pressure', output/'snapshot', flash,
                       case_count=plan['cases'], judge_workers=4)
    verify_all(output)
    f.save(output/'scores.json', report)
    f.save(output/'status.json', dict(status='completed', cases=plan['cases'],
                                    response_model=flash.returned_model, at=time.time()))
    print(json.dumps({k: v for k, v in report.items() if k != 'values'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    for name in ('campaign', 'evaluation', 'output', 'credential-file'):
        parser.add_argument('--'+name, type=Path, required=name == 'output')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        print(json.dumps(dict(cases=freeze(args.campaign.resolve(), output, args.evaluation.resolve()))))
    else:
        run(output, args.credential_file)
