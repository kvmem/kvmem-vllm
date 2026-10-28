"""Preserve completed pilot responses after a post-inference audit failure.

No answer is regenerated. Six original cold responses are re-audited offline;
the unattempted next two original questions form a separate matched cold/warm
pair. The full benchmark cohort and its sampling remain unchanged.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

from benchmark_niah import save,sha
from benchmark_memory import source as load_input
from graph_v2_protocol import require
from recover_hybrid_lme import preserve_case
from recover_hybrid_mab import engine
from run_ruler_hybrid_campaign import verify_files

REPO=Path(__file__).resolve().parents[1]
PATCHES=('scripts/ruler_hybrid_protocol.py','scripts/run_hybrid_memory_campaign.py')
CAPTURES=('bounded_refresh.jsonl','block_graph.jsonl','block_graph_inputs.jsonl','archive.json')


def read(path):return json.loads(Path(path).read_text())


def split_cases(fixture, answers):
    """Select by completed-response existence, never by content or correctness."""
    planned=fixture['case_ids']
    saved=[i for i in planned if (answers/'cases'/i/'response.json').is_file()]
    missing=[i for i in planned if i not in saved]
    require(len(missing)==1 and saved,'Expected one unattempted prefix continuation')
    require(not (answers/'cases'/missing[0]).exists(),'Continuation was attempted; preserve it separately')
    group=missing[0].rsplit('_q',1)[0]
    rows=[r['id'] for r in fixture['rows'] if r['id'].rsplit('_q',1)[0]==group]
    pos=rows.index(missing[0]);require(pos+1<len(rows),'Missing next original warm question')
    warm=rows[pos:pos+2]
    require(not set(warm)&set(saved),'Never regenerate a saved answer')
    require(len({i.rsplit('_q',1)[0] for i in saved})==len(saved),
        'Saved responses must be the original cold prefix requests')
    return saved,warm


def clone(source,target,ids,*,alpha,warm):
    require(not target.exists(),'Never overwrite a pilot or frozen runtime')
    import run_hybrid_memory_campaign as campaign
    fixture,profile=campaign.verify(source)
    require(read(source/'status.json')['status']=='prepared' and not (source/'attempt.json').exists(),
        'Clone only an unstarted prepared qualification')
    shutil.copytree(source,target)
    old=read(target/'runtime-proof.json')['files']
    for name in PATCHES:shutil.copy2(REPO/name,target/'runtime'/name)
    files={str(p.relative_to(target/'runtime')):sha(p) for p in (target/'runtime').rglob('*') if p.is_file()}
    require(engine(old)==engine(files),'Audit-only pilot must preserve all inference/scoring sources')
    changes={n:dict(before=old.get(n),after=files.get(n)) for n in set(old)|set(files) if old.get(n)!=files.get(n)}
    require(set(changes)<=set(PATCHES),'Unexpected runtime changes')
    save(target/'runtime-proof.json',dict(files=files))
    profile['kvmem']['graph_alpha']=alpha;save(target/'profile.json',profile)
    fixture.update(case_ids=ids,cases=len(ids),efficiency_pilot=True,efficiency_expected_warm=warm,
        qualification_scope='Explicit audit-recovery efficiency subset; original inputs/seeds/sampling unchanged',
        profile_sha256=sha(target/'profile.json'))
    save(target/'fixture.json',fixture)
    save(target/'audit-update.json',dict(source=str(source),source_proof_sha256=sha(source/'prepared-proof.json'),
        engine_sources_identical=True,changes=changes))
    save(target/'prepared-proof.json',{n:sha(target/n) for n in
        ('profile.json','fixture.json','runtime-proof.json','audit-update.json')})
    save(target/'status.json',dict(status='prepared',cases=len(ids),at=time.time()))
    campaign.verify(target)


def recover_cold(previous,target,ids):
    import hybrid_memory_protocol as policy
    old_fixture=read(previous/'fixture.json');profile=read(target/'profile.json')
    require(read(previous/'status.json')['status']=='failed','Source must preserve the failed pilot')
    cleanup=read(previous/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],'Prior GPU cleanup incomplete')
    require(profile==read(previous/'profile.json') and
        engine(read(previous/'runtime-proof.json')['files'])==engine(read(target/'runtime-proof.json')['files']),
        'Offline audit recovery cannot change the original model/scoring profile')
    capture=previous/'run/server/capture';answers=previous/'run/answers';out=target/'run/answers'
    require(not out.exists(),'Never overwrite recovered answers')
    logs={n:[json.loads(s) for s in (capture/n).read_text().splitlines()] for n in CAPTURES if n.endswith('.jsonl')}
    terminals=read(capture/'archive.json')['completed']
    rows={r['id']:r for r in old_fixture['rows']};originals={};repaired=[]
    for ident in ids:
        directory=answers/'cases'/ident;row=rows[ident];src=load_input(row)
        response=read(directory/'response.json');request=read(directory/'request.json');old=read(directory/'result.json')
        original=Path(old_fixture['reference'])/'results/attention/cases/memoryagentbench'/ident/'request_config.json'
        sampling=read(original);choice=response['choices'][0]
        require(all(request[k]==sampling[k] for k in
            ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed','max_tokens')) and
            request['sampling_sha256']==sha(original) and request['thinking_token_budget']==0 and
            request['token_sha256']==row['token_sha256'] and request['prompt_tokens']==row['input_tokens'] and
            request['kvmem_query_span']==[row['query_begin'],row['query_end']], 'Saved sampling/input identity changed')
        require(choice['prompt_token_ids']==src['tokens'] and response['usage']['prompt_tokens']==len(src['tokens']) and
            choice['finish_reason'] in ('stop','length') and len(choice['token_ids'])==response['usage']['completion_tokens']>0,
            'Saved response is incomplete or belongs to different input')
        match=lambda e:e['request_id'].startswith(response['id']+'-')
        terminal=[t for t in terminals if match(t)]
        require(len(terminal)==1 and terminal[0]['checkpoint_kind']=='P' and
            terminal[0]['checkpoint_tokens']==request['prefix_tokens'] and terminal[0].get('retirement_ack_ready_at'),
            'Original frozen-prefix retirement lacks evidence')
        evidence=policy.audit_case(capture,response['id'],profile,request['kvmem_query_span'],require_pressure=True,
            event_reader=lambda p:logs[p.name],prompt_tokens=len(src['tokens']),output_tokens=request['max_tokens'],
            replay_span=request['kvmem_replay_span'])
        require(evidence['pressure'],'Original cold request unexpectedly reused a prefix')
        raw=choice['text']
        record=dict(src['info'],suite='memoryagentbench',id=ident,status='completed',
            answer=raw.split('</think>',1)[-1].strip(),raw_completion=raw,usage=response['usage'],
            token_sha256=row['token_sha256'],source_token_sha256=row['token_sha256'],
            prompt_sha256=row['prompt_sha256'],query_sha256=row['query_sha256'],prompt_tokens=len(src['tokens']),
            source_prompt_tokens=len(src['tokens']),original_input_intact=True,seconds=old['seconds'],
            metrics=response['metrics'],prefix_tokens=request['prefix_tokens'],prefix_sha256=request['prefix_sha256'],
            prefix_reused=False,attention=evidence['query'],finish_reason=choice['finish_reason'],graph_v2=evidence,
            recovery_timing_scope='Original duration preserved; failed-case duration includes post-response audit. '
                'Server metrics are the original inference timings, not a new measurement.')
        before=preserve_case(directory,out/'cases'/ident,record,dict(graph_v2=evidence,source=str(directory)))
        for n,v in [('terminal.json',terminal[0]),('events.json',[e for e in logs['bounded_refresh.jsonl'] if match(e)])]:
            if not (out/'cases'/ident/n).exists():save(out/'cases'/ident/n,v)
        originals.update({str((directory/n).relative_to(previous)):h for n,h in before.items()})
        if old['status']!='completed':repaired.append(ident)
    for n in CAPTURES:originals['run/server/capture/'+n]=sha(capture/n)
    for n in ('fixture.json','profile.json','runtime-proof.json','prepared-proof.json','run/cleanup.json'):
        originals[n]=sha(previous/n)
    verify_files(previous,originals)
    proof=dict(source=str(previous),source_files=originals,recovered_ids=ids,repaired_audit_ids=repaired,
        original_successes_unchanged=len(ids)-len(repaired),regenerated_answers=0,engine_sources_identical=True,
        owner_cleanup='Failed source controller terminated its owned server; verified no owned groups/GPU processes. '
            'This is an offline cold efficiency subset, not a claim of successful full campaign or warm lifecycle.')
    save(target/'recovery-proof.json',proof)
    save(out/'summary.json',dict(completed=len(ids),expected=len(ids),errors=0))
    save(target/'run/cleanup.json',cleanup)
    save(target/'execution-audit.json',dict(status='passed',cases=len(ids),scope='offline recovered cold pilot only',
        original_inputs_unchanged=True,regenerated_answers=0,source_cleanup_sha256=sha(previous/'run/cleanup.json'),
        recovery_proof_sha256=sha(target/'recovery-proof.json'),case_hashes={str(p.relative_to(target)):sha(p)
            for p in (out/'cases').rglob('*') if p.is_file()}))
    save(target/'status.json',dict(status='completed',cases=len(ids),offline_reaudit=True,at=time.time()))
    return proof


def prepare(previous,template,root):
    require(not root.exists(),'Use a new efficiency version')
    fixture=read(previous/'fixture.json');saved,warm=split_cases(fixture,previous/'run/answers')
    root.mkdir(parents=True)
    for name,ids,alpha,expected in [('cold/graph-off',saved,0.,0),('cold/graph-on',saved,.2,0),
            ('warm/graph-off',warm,0.,1),('warm/graph-on',warm,.2,1)]:
        clone(template,root/name,ids,alpha=alpha,warm=expected)
    proof=recover_cold(previous,root/'cold/graph-off',saved)
    save(root/'selection.json',dict(original_pilot=str(previous),original_case_ids=fixture['case_ids'],
        cold_case_ids=saved,warm_case_ids=warm,new_case_ids=[i for i in warm if i not in fixture['case_ids']],
        reason='Post-inference normalization audit failed. Preserve all six complete responses. '
            'The unattempted next question starts a separate cold request, followed by the next original warm question, '
            'with the same lifecycle in both graph arms. No saved answer is regenerated or omitted by score.',
        no_accuracy_gate=True,full_campaigns_blocked_until_efficiency_review=True))
    arms=[]
    for name in ('cold/graph-on','warm/graph-off','warm/graph-on'):
        p=root/name
        arms.append(dict(name=name,root=str(p),controller='run_hybrid_memory_campaign.py',
            prepared_proof_sha256=sha(p/'prepared-proof.json')))
    save(root/'queue-plan.json',dict(wait_for=str(root/'cold/graph-off'),arms=arms,grading=[]))
    return dict(root=str(root),inherited=len(saved),repaired=len(proof['repaired_audit_ids']),
        regenerated_answers=0,new_inference_cases=len(saved)+2*len(warm),full_campaigns_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','template','root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.previous.resolve(),a.template.resolve(),a.root.resolve()),indent=2))
