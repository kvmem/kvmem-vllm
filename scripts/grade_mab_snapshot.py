"""Grade an immutable completion-order MAB snapshot without stopping inference."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import regrade_flash as f
from grade_lme_snapshot import verify as verify_common
from graph_v2_protocol import require

REPO = Path(__file__).resolve().parents[1]
EVALUATORS = ('evaluators/historical_lme_score.py',
    'evaluators/score_memoryagentbench_official_local.py',
    'evaluators/judge_memoryagentbench_special_flash.py',
    'official/utils/eval_other_utils.py',
    'official/llm_based_eval/summarization_evaluate.py')


def freeze(campaign, output, evaluation):
    require(not output.exists(), 'Use a fresh grading snapshot')
    fixture = f.read(campaign/'fixture.json'); profile = f.read(campaign/'profile.json')
    require(fixture['suite'] == 'memoryagentbench' and fixture['cases'] == 1116 and
            not fixture.get('qualification'), 'Expected the original full MAB campaign')
    paths = {p.parent.name:p for p in (campaign/'run/answers/cases').glob('*/result.json')}
    require(set(paths) <= {r['id'] for r in fixture['rows']}, 'Unexpected completed case')
    chosen = []
    for row in fixture['rows']:
        ident = row['id']
        if ident not in paths:
            continue
        path = paths[ident]; raw = path.read_bytes(); result = json.loads(raw)
        require(result['id'] == ident and result['suite'] == 'memoryagentbench' and
            result['status'] == 'completed' and result['original_input_intact'] and
            result['token_sha256'] == row['token_sha256'] and result['prompt_tokens'] == row['input_tokens'],
            'Completed MAB answer differs from the fixed original input')
        evidence = result['graph_v2']
        require(evidence['quota']['mean_fraction'] == .5 and evidence['cpu_fps_audit'] and
            not result.get('hit_output_cap', False) and
            bool(evidence['native_replay_audit']) == bool(profile['kvmem']['graph_trace']),
            'Missing declared per-case execution audit')
        require(result['answer'] == result['raw_completion'].split('</think>',1)[-1].strip(),
            'Answer extraction differs from raw completion')
        chosen.append((ident, path, raw))
    require(0 < len(chosen) <= 1116, 'No completed MAB answers')
    output.mkdir(parents=True)
    bindings = {}; snapshot = output/'snapshot'
    for ident, path, raw in chosen:
        destination = snapshot/'cases'/ident/'result.json'; destination.parent.mkdir(parents=True)
        destination.write_bytes(raw)
        require(path.read_bytes() == raw, 'Completed answer changed during snapshot')
        bindings[ident] = dict(path=str(path.resolve()), sha256=f.sha(path), snapshot_sha256=f.sha(destination))
    manifest = campaign/'run/answers/manifest.json'
    shutil.copyfile(manifest, snapshot/'manifest.json')
    reference = Path(f.read(manifest)['reference'])/'results/attention/memoryagentbench/special_judgments.jsonl'
    runtime = output/'runtime'
    shutil.copytree(REPO/'scripts', runtime/'scripts', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    f.save(output/'runtime-proof.json', dict(files={str(p.relative_to(runtime)):f.sha(p)
        for p in runtime.rglob('*') if p.is_file()}))
    f.save(output/'bindings.json', bindings)
    f.save(output/'plan.json', dict(suite='memoryagentbench', campaign=str(campaign.resolve()),
        evaluation=str(evaluation.resolve()), cases=len(chosen), full_cohort_cases=1116,
        partial=len(chosen)<1116, case_ids=[i for i,_,_ in chosen], at=time.time(),
        selection='All completed original cases at snapshot listing; no answer/score filtering',
        limitation='Completion-order subset; full campaign execution audit is still pending.',
        campaign_proofs={name:f.sha(campaign/name) for name in
            ('fixture.json','profile.json','prepared-proof.json','run/answers/manifest.json')},
        evaluator_proofs={name:f.sha(evaluation/name) for name in EVALUATORS},
        reference_binding=dict(path=str(reference.resolve()),sha256=f.sha(reference)),
        snapshot_manifest_sha256=f.sha(snapshot/'manifest.json')))
    f.save(output/'prepared-proof.json', {name:f.sha(output/name) for name in
        ('plan.json','bindings.json','runtime-proof.json')})
    f.save(output/'status.json', dict(status='prepared', cases=len(chosen), partial=len(chosen)<1116, at=time.time()))
    return len(chosen)


def verify(output):
    plan = verify_common(output)
    require(plan['suite'] == 'memoryagentbench', 'Not a MAB snapshot')
    f.unchanged(plan['reference_binding'])
    require(f.sha(output/'snapshot/manifest.json') == plan['snapshot_manifest_sha256'], 'Snapshot manifest changed')
    return plan


def run(output, credential_file):
    require(Path(__file__).resolve() == output/'runtime/scripts'/Path(__file__).name, 'Run the frozen grader only')
    plan = verify(output)
    judge = f.load(Path(plan['evaluation'])/'evaluators/historical_lme_score.py','mab_snapshot_transport')
    require(judge.API_BASE == 'https://api.deepseek.com', 'Unexpected judge endpoint')
    flash = f.Flash(output, judge, credential_file.read_text().strip())
    models,_ = flash.client.request('/models')
    require(f.MODEL in {r['id'] for r in models['data']}, 'Configured Flash judge unavailable')
    f.save(output/'model-availability.json',dict(model=f.MODEL,at=time.time()))
    f.save(output/'status.json', dict(status='grading',cases=plan['cases'],partial=plan['partial'],at=time.time()))
    os.environ['DEEPSEEK_API_KEY'] = 'transport-managed'
    report = f.mab_arm(output, plan, 'hybrid', output/'snapshot', flash,
        case_count=plan['cases'], judge_workers=4)
    verify(output)
    f.save(output/'scores.json', report)
    f.save(output/'status.json',dict(status='completed',cases=plan['cases'],partial=plan['partial'],
        judge=f.MODEL,response_model=flash.returned_model,at=time.time()))
    print(json.dumps({k:v for k,v in report.items() if k!='values'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run'))
    for name in ('campaign','evaluation','output','credential-file'):
        parser.add_argument('--'+name,type=Path,required=name=='output')
    args = parser.parse_args(); output = args.output.resolve()
    try:
        if args.action == 'prepare':
            print(json.dumps(dict(frozen_cases=freeze(args.campaign.resolve(),output,args.evaluation.resolve()))))
        else:
            run(output,args.credential_file)
    except BaseException as error:
        if output.exists(): f.save(output/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
