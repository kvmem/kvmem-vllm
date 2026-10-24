"""Resume unanswered MAB cases after an audited technical failure, same engine.

The old attempt remains immutable. Completed responses are copied byte-for-byte,
including incorrect answers. A saved response without completed execution evidence
is refused rather than regenerated. CUDA blocking can locate the failing launch;
it never changes the fixed profile, seeds, concurrency or generation limit.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from benchmark_niah import save, sha
from graph_v2_protocol import require
from recover_hybrid_mab import engine, inherit_cases, read

KIND = 'same_engine_technical_failure'
TOOLS = ('scripts/run_hybrid_memory_campaign.py','scripts/recover_hybrid_mab_failure.py')
REPO = Path(__file__).resolve().parents[1]
SAMPLING = ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed','max_tokens')


def validate_request(request, row, sampling, prefix):
    require(all(request.get(k) == sampling[k] for k in SAMPLING),
        'Inherited request seed/sampling changed')
    require(request.get('token_sha256') == row['token_sha256'] and
        request.get('prompt_tokens') == row['input_tokens'] and
        request.get('kvmem_query_span') == [row['query_begin'],row['query_end']] and
        request.get('kvmem_replay_span') == [prefix,row['query_end']] and
        request.get('thinking_token_budget') == 0 and request.get('prefix_tokens') == prefix,
        'Inherited input, Query or replay boundary changed')


def verify_recovery(root, fixture, profile, proof):
    from run_ruler_hybrid_campaign import verify_files
    previous = Path(proof['source'])
    old_fixture = read(previous/'fixture.json')
    require(fixture['suite'] == 'memoryagentbench' and not fixture['qualification'] and
        fixture.get('recovery_kind') == proof['kind'] == KIND and
        proof['engine_sources_identical'] and proof['regenerated_answers'] == 0,
        'Unsupported failure recovery')
    require(profile == read(previous/'profile.json') and
        all(fixture[k] == old_fixture[k] for k in ('rows','case_ids','cases','source_cohort_sha256',
            'reference','source_fixture','source_fixture_sha256','sampling','workers','active_groups','cpu_headroom_gib')),
        'Recovery changed original fixture/profile/sampling/concurrency')
    old = read(previous/'runtime-proof.json')['files']; new = read(root/'runtime-proof.json')['files']
    require(engine(old) == engine(new),'Recovery changed inference/scoring engine sources')
    changed = {k for k in set(old)|set(new) if old.get(k) != new.get(k)}
    require(changed <= set(TOOLS),'Unexpected recovery tool changes')
    require(proof['pending_ids'] == [i for i in fixture['case_ids'] if i not in set(proof['recovered_ids'])] and
        len(set(proof['recovered_ids'])) == len(proof['recovered_ids']) and
        set(proof['recovered_ids']) <= set(fixture['case_ids']), 'Recovery case partition changed')
    require(fixture.get('diagnostic_environment',{}) in ({},{'CUDA_LAUNCH_BLOCKING':'1'}),
        'Unsupported diagnostic environment')
    require(read(previous/'status.json')['status'] == 'failed','Source is not a failed attempt')
    cleanup=read(previous/'run/cleanup.json')
    require(isinstance(cleanup.get('remaining_owned_groups'),(dict,list)) and
        not cleanup['remaining_owned_groups'] and cleanup.get('gpu_processes') == [],
        'Source GPU cleanup missing')
    verify_files(previous,proof['source_files'])


def prepare(root, previous, *, cuda_blocking=False):
    import run_hybrid_memory_campaign as campaign
    require(not root.exists(),'Use a fresh recovery root; never overwrite an attempt')
    old_fixture,profile = campaign.verify(previous)
    require(old_fixture['suite'] == 'memoryagentbench' and not old_fixture['qualification'] and
        read(previous/'status.json')['status'] == 'failed','Expected failed formal MAB attempt')
    require(not old_fixture.get('optimization_source'),'Optimization recovery needs its own explicit chain audit')
    cleanup=read(previous/'run/cleanup.json')
    require(isinstance(cleanup.get('remaining_owned_groups'),(dict,list)) and
        not cleanup['remaining_owned_groups'] and cleanup.get('gpu_processes') == [],'Previous GPU resources remain')
    rows={r['id']:r for r in old_fixture['rows']}
    manifest=read(previous/'run/answers/manifest.json')
    require(manifest['profile_sha256'] == sha(previous/'profile.json') and manifest['cases'] == old_fixture['cases'] and
        manifest['rows'] == old_fixture['rows'],'Original client manifest changed')
    # Verify fixed sampling independently of result correctness.
    for directory in (previous/'run/answers/cases').iterdir():
        result=read(directory/'result.json') if (directory/'result.json').exists() else {}
        if result.get('status') != 'completed':continue
        row=rows[directory.name]; group=directory.name.rsplit('_q',1)[0]
        sampling=read(Path(old_fixture['reference'])/'results/attention/cases/memoryagentbench'/directory.name/'request_config.json')
        validate_request(read(directory/'request.json'),row,sampling,manifest['prefix_boundaries'][group])
    root.mkdir(parents=True)
    shutil.copytree(previous/'runtime',root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in TOOLS:shutil.copy2(REPO/name,root/'runtime'/name)
    save(root/'runtime-proof.json',dict(files={str(p.relative_to(root/'runtime')):sha(p)
        for p in (root/'runtime').rglob('*') if p.is_file()}))
    shutil.copy2(previous/'profile.json',root/'profile.json')
    fixture=deepcopy(old_fixture)
    fixture.update(recovery_source=str(previous),recovery_kind=KIND,
        diagnostic_environment={'CUDA_LAUNCH_BLOCKING':'1'} if cuda_blocking else {})
    proof=inherit_cases(previous,root/'recovered-answers',fixture)
    proof.update(kind=KIND,engine_sources_identical=True,
        pending_ids=[i for i in fixture['case_ids'] if i not in set(proof['recovered_ids'])],
        diagnostic_environment=fixture['diagnostic_environment'],
        scope='No engine/profile change; successful responses reused verbatim; cold prefix re-ingest only for unfinished groups.')
    for name in ('fixture.json','profile.json','runtime-proof.json','prepared-proof.json','status.json',
                 'failure.json','run/cleanup.json','controller.log','run/client.log','run/server.log'):
        if (previous/name).is_file():proof['source_files'][name]=sha(previous/name)
    fixture['recovery_cases']=proof['recovered_ids']
    save(root/'recovery-proof.json',proof);save(root/'fixture.json',fixture)
    save(root/'prepared-proof.json',{n:sha(root/n) for n in
        ('runtime-proof.json','profile.json','fixture.json','recovery-proof.json')})
    save(root/'status.json',dict(status='prepared',cases=fixture['cases'],suite=fixture['suite'],
        inherited=len(proof['recovered_ids']),pending=len(proof['pending_ids'])))
    campaign.verify(root)
    return dict(root=str(root),inherited=len(proof['recovered_ids']),pending=len(proof['pending_ids']),
        source_files=len(proof['source_files']),diagnostic_environment=fixture['diagnostic_environment'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--cuda-blocking',action='store_true')
    a=p.parse_args();print(prepare(a.root.resolve(),a.previous.resolve(),cuda_blocking=a.cuda_blocking))
