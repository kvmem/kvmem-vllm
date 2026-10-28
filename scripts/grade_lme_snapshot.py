"""Freeze all completed LME answers at one instant, then grade with Flash."""
import argparse
import json
from pathlib import Path
import shutil
import time

import regrade_flash as f
from graph_v2_protocol import require

REPO = Path(__file__).resolve().parents[1]
EVALUATORS = ('evaluators/historical_lme_score.py','evaluators/historical_lme_evaluate_qa.py',
    'evaluators/historical_lme_calibration.json')


def freeze(campaign,output,evaluation):
    require(not output.exists(),'Use a fresh grading snapshot; never replace existing judgments')
    fixture = f.read(campaign/'fixture.json')
    require(fixture['suite'] == 'longmemeval_s' and fixture['cases'] == 102,'Expected the original 102-case LME cohort')
    rows = {r['id']:r for r in fixture['rows']}; chosen = []
    for ident,row in rows.items():
        directory = campaign/'run/answers/cases'/ident; path = directory/'result.json'
        if not path.exists(): continue
        raw = path.read_bytes(); result = json.loads(raw)
        require(result['status'] == 'completed','An attempted LME case failed; preserve and inspect it')
        require(result['id'] == ident and result['suite'] == 'longmemeval_s' and
            result['original_input_intact'] and result['token_sha256'] == row['token_sha256'] and
            result['prompt_tokens'] == row['input_tokens'] and not result.get('hit_output_cap') and
            result['graph_v2']['native_replay_audit']['native_picker_replay_equal'],
            'Completed LME answer lacks original input or native execution audit')
        require(result['answer'] == result['raw_completion'].split('</think>',1)[-1].strip(),
            'Saved answer extraction differs from raw completion')
        chosen.append((ident,path,raw))
    require(0 < len(chosen) <= 102,'No completed LME answers available')
    output.mkdir(parents=True)
    bindings = {}; snapshot = output/'snapshot/cases'
    for ident,path,raw in chosen:
        destination = snapshot/ident/'result.json'; destination.parent.mkdir(parents=True)
        destination.write_bytes(raw)
        require(path.read_bytes() == raw,'Completed answer changed while freezing')
        bindings[ident] = dict(path=str(path.resolve()),sha256=f.sha(path),snapshot_sha256=f.sha(destination))
    runtime = output/'runtime'
    shutil.copytree(REPO/'scripts',runtime/'scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    f.save(output/'runtime-proof.json',dict(files={str(p.relative_to(runtime)):f.sha(p)
        for p in runtime.rglob('*') if p.is_file()}))
    f.save(output/'bindings.json',bindings)
    f.save(output/'plan.json',dict(campaign=str(campaign.resolve()),evaluation=str(evaluation.resolve()),
        lme={'hybrid':str(snapshot.resolve())},cases=len(chosen),full_cohort_cases=102,partial=len(chosen)<102,
        case_ids=[ident for ident,_,_ in chosen],at=time.time(),
        selection='All completed original cases at snapshot creation; no gold/answer/score filtering',
        limitation='Completion-order subset; category proportions may differ from the full 102 cases.',
        campaign_proofs={name:f.sha(campaign/name) for name in ('fixture.json','profile.json','prepared-proof.json')},
        evaluator_proofs={name:f.sha(evaluation/name) for name in EVALUATORS}))
    f.save(output/'prepared-proof.json',{name:f.sha(output/name) for name in
        ('plan.json','bindings.json','runtime-proof.json')})
    f.save(output/'status.json',dict(status='prepared',cases=len(chosen),partial=len(chosen)<102,at=time.time()))
    return len(chosen)


def verify(output):
    for name,digest in f.read(output/'prepared-proof.json').items():
        require(f.sha(output/name) == digest,'Frozen grading metadata changed')
    for name,digest in f.read(output/'runtime-proof.json')['files'].items():
        require(f.sha(output/'runtime'/name) == digest,'Frozen grading source changed')
    plan = f.read(output/'plan.json')
    for name,digest in plan['campaign_proofs'].items():
        require(f.sha(Path(plan['campaign'])/name) == digest,'Inference fixture/profile changed')
    for name,digest in plan['evaluator_proofs'].items():
        require(f.sha(Path(plan['evaluation'])/name) == digest,'Frozen official judge/rubric changed')
    bindings = f.read(output/'bindings.json')
    require(set(bindings) == set(plan['case_ids']) and len(bindings) == plan['cases'],'Grading subset changed')
    for ident,binding in bindings.items():
        f.unchanged(binding)
        require(f.sha(output/'snapshot/cases'/ident/'result.json') == binding['snapshot_sha256'],
            'Frozen answer snapshot changed')
    return plan


def run(output,credential_file):
    require(Path(__file__).resolve() == output/'runtime/scripts'/Path(__file__).name,'Run the frozen grader only')
    plan = verify(output)
    judge = f.load(Path(plan['evaluation'])/'evaluators/historical_lme_score.py','snapshot_flash_transport')
    judge.OFFICIAL = Path(plan['evaluation'])/'evaluators/historical_lme_evaluate_qa.py'
    require(judge.API_BASE == 'https://api.deepseek.com','Unexpected judge endpoint')
    flash = f.Flash(output,judge,credential_file.read_text().strip())
    models,_ = flash.client.request('/models')
    require(f.MODEL in {r['id'] for r in models['data']},'Configured Flash judge unavailable')
    f.save(output/'status.json',dict(status='grading',cases=plan['cases'],partial=plan['partial'],at=time.time()))
    report = f.lme(output,plan,flash,case_count=plan['cases'])
    verify(output)
    f.save(output/'scores.json',report)
    f.save(output/'status.json',dict(status='completed',cases=plan['cases'],partial=plan['partial'],
        judge=f.MODEL,response_model=flash.returned_model,at=time.time()))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--credential-file',type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    try:
        if args.action == 'prepare':
            print(json.dumps(dict(frozen_cases=freeze(args.campaign.resolve(),output,args.evaluation.resolve()))))
        else:
            run(output,args.credential_file)
    except BaseException as error:
        if output.exists(): f.save(output/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
