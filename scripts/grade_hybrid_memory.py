"""Use the existing frozen Flash rubric on newly completed hybrid answers."""
import argparse
import os
from pathlib import Path
import time

import regrade_flash as flash_adapter
from graph_v2_protocol import require


def reuse_flash_calls(source, output):
    """Reuse only exact request-bound responses from a completed snapshot."""
    f = flash_adapter
    state = f.read(source/'status.json')
    require(state['status'] == 'completed' and state['judge'] == f.MODEL,'Partial Flash grading has not completed')
    if (source/'plan.json').exists() and f.read(source/'plan.json').get('suite') == 'memoryagentbench':
        from grade_mab_snapshot import verify
    else:
        from grade_lme_snapshot import verify
    verify(source)
    reused = {}
    for response in sorted((source/'calls').glob('*.response.json')):
        ident = response.name.removesuffix('.response.json')
        request = response.with_name(ident+'.request.json')
        body = f.read(request); raw = f.read(response)['response']
        require(f.digest(body) == ident and body['model'] == f.MODEL,'Cached request content differs')
        f.check_flash(raw['model'].lower())
        for path in (request,response):
            target = output/'calls'/path.name; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(path.read_bytes())
            require(f.sha(target) == f.sha(path),'Cached Flash response changed during copy')
            reused[str(path.resolve())] = f.sha(path)
    f.save(output/'reused-calls.json',dict(source=str(source.resolve()),request_count=len(reused)//2,files=reused))
    return len(reused)//2


def run(args):
    f = flash_adapter
    audit = f.read(args.campaign/'execution-audit.json')
    require(audit['status'] == 'passed' and not audit.get('qualification'),'Grade only fully audited cohorts')
    args.output.mkdir(parents=True,exist_ok=False)
    if getattr(args, 'cache_source', None):
        snapshot = f.read(args.cache_source/'plan.json')
        require(snapshot.get('suite','longmemeval_s') == audit['suite'] and
            Path(snapshot['campaign']).resolve() == args.campaign.resolve(),
            'Partial verdicts must belong to the same suite and inference campaign')
        reuse_flash_calls(args.cache_source,args.output)
    judge = f.load(args.evaluation/'evaluators/historical_lme_score.py','hybrid_flash_transport')
    judge.OFFICIAL = args.evaluation/'evaluators/historical_lme_evaluate_qa.py'
    require(judge.API_BASE == 'https://api.deepseek.com','Unexpected judge transport')
    flash = f.Flash(args.output,judge,args.credential_file.read_text().strip())
    models,_ = flash.client.request('/models')
    require(f.MODEL in {row['id'] for row in models['data']},'Configured Flash judge is unavailable')
    f.save(args.output/'model-availability.json',dict(model=f.MODEL,at=time.time()))
    answers = args.campaign/'run/answers'
    plan = dict(evaluation=str(args.evaluation),lme={'hybrid':str(answers/'cases')})
    f.save(args.output/'plan.json',dict(plan,campaign=str(args.campaign),suite=audit['suite'],
        judge_workers=getattr(args,'judge_workers',1),
        inference_audit_sha256=f.sha(args.campaign/'execution-audit.json')))
    os.environ['DEEPSEEK_API_KEY'] = 'transport-managed'
    if audit['suite'] == 'longmemeval_s':
        report = f.lme(args.output,plan,flash)
    else:
        report = f.mab_arm(args.output,plan,'hybrid',answers,flash,
            judge_workers=getattr(args,'judge_workers',1))
    f.save(args.output/'scores.json',report)
    f.save(args.output/'status.json',dict(status='completed',judge=f.MODEL,response_model=flash.returned_model,at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output','evaluation','credential-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache-source',type=Path)
    parser.add_argument('--judge-workers',type=int,choices=range(1,5),default=1)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.output.exists():
            flash_adapter.save(args.output/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
