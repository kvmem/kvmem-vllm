"""Frozen LME102 comparison of K16K/G4K rolling output to completed K16K/G16K.

Reuse the GPU-qualified engine and the original eight-owner Memory driver.
Never restart MAB, resample a saved answer, or enable a scheduled monitor.
"""
import argparse
from copy import deepcopy
import csv
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p
from benchmark_client_io import result_events

REPO=Path(__file__).resolve().parents[1]
DRIVERS={'scripts/run_layerwise_memory.py','scripts/run_layerwise_rolling_memory.py'}


def rolling_profile(original):
    if (original['kvmem']['select_budget'],original['kvmem']['gen_budget'])!=(16384,16384):
        raise ValueError('Require the completed K16K/G16K reference')
    result=deepcopy(original)
    result['kvmem'].update(gen_budget=4096,rolling_active_window=True)
    result['query_replay_budget']=4096
    result['request_planning_batch_tokens']=original['query_replay_budget']+original['max_num_seqs']*(original['draft_tokens']+1)
    return result


def rolling_fixture(original):
    result=deepcopy(original)
    result['execution_flags']['layerwise_rolling_decode']=True
    return result


def matched_native_arguments(reference, current):
    def stable(argv):
        result=list(argv)
        index=result.index('--kv-transfer-config')
        # This contains the authorized KV budgets and run-local output paths;
        # the full experiment profile is verified separately before startup.
        result[index+1]='<verified-kvmem-controls>'
        return result
    if stable(reference)!=stable(current):
        raise ValueError('Actual native model or batching arguments differ from the reference')


def verify_effective_controls(root,launch):
    source=Path(p.read(root/'rolling-variant.json')['source'])
    reference=p.read(source/'run/server/launch.json')
    matched_native_arguments(reference['vllm_argv'],launch['vllm_argv'])
    p.save(root/'effective-control-parity.json',dict(status='passed',
        reference_launch_sha256=p.sha(source/'run/server/launch.json'),
        actual_launch_sha256=p.sha(root/'run/server/launch.json'),
        native_arguments_equal_except_verified_kvmem=True,
        native_batch_tokens=launch['profile']['max_num_batched_tokens'],
        max_num_seqs=launch['profile']['max_num_seqs']))


def verify_variant(root,fixture,profile):
    import run_layerwise_memory as memory
    import run_agent_handoff_queue as q
    variant=p.read(root/'rolling-variant.json')
    source=Path(variant['source']);qualified=Path(variant['qualification'])
    if source==root or (source/'rolling-variant.json').exists():
        raise ValueError('Rolling reference must be the completed nonrolling cohort')
    for name,digest in variant['source_files'].items():
        if p.sha(Path(name))!=digest:raise ValueError('Frozen reference changed: '+name)
    old_fixture,old_profile=memory.verify(source)
    if fixture!=rolling_fixture(old_fixture) or profile!=rolling_profile(old_profile):
        raise ValueError('Rolling changed an unrelated input, sampling or model control')
    manifest=q.verify_frozen(qualified)
    if Path(manifest['source'])!=source or p.read(qualified/'final-review.json')['status']!='passed':
        raise ValueError('Rolling engine does not match its passed GPU qualification')
    qualified_profile=p.read(qualified/'profile.json')
    for key in ('kvmem','max_num_seqs','max_num_batched_tokens','query_replay_budget','request_planning_batch_tokens'):
        if qualified_profile.get(key)!=profile.get(key):
            raise ValueError('Formal GPU capacity differs from its qualification: '+key)
    proof=p.read(root/'runtime-proof.json')
    original={name.removeprefix('runtime/'):digest for name,digest in manifest['files'].items()
              if name.startswith('runtime/')}
    if (set(proof)-set(original)!= {'scripts/run_layerwise_rolling_memory.py'} or
            set(original)-set(proof) or any(proof[n]!=original[n] for n in proof if n not in DRIVERS)):
        raise ValueError('Formal inference engine differs from the GPU-qualified version')
    if profile.get('verify_session_restore') or fixture.get('restore_digest_qualification'):
        raise ValueError('Expensive diagnostic byte oracle must remain off in performance runs')


def prepare(root,source,qualified):
    import run_layerwise_memory as memory
    import run_agent_handoff_queue as q
    fixture,profile=memory.verify(source)
    q.verify_frozen(qualified)
    if root.exists():raise ValueError('Use a fresh immutable campaign')
    cleanup=p.read(source/'run/cleanup.json')
    if (p.read(source/'status.json')['status']!='completed' or fixture['cases']!=102 or
            fixture['suite']!='longmemeval_s' or fixture['qualification'] or
            cleanup['gpu_processes'] or cleanup['remaining_owned_groups']):
        raise ValueError('Reference cohort must be complete and released')
    reviewed=p.read(qualified/'final-review.json')
    if reviewed['status']!='passed' or not reviewed['all_processes_released']:
        raise ValueError('Rolling qualification has not passed and released its GPU')
    root.mkdir(parents=True)
    campaign=root/'gen4k';campaign.mkdir()
    runtime=campaign/'runtime-v1'
    shutil.copytree(qualified/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in DRIVERS:shutil.copy2(REPO/name,runtime/name)
    shutil.copy2(__file__,root/'launch.py')
    previous=source.parent
    shutil.copy2(previous/'memory_scores_audit.py',root/'memory_scores_audit.py')
    p.save(campaign/'profile.json',rolling_profile(profile))
    p.save(campaign/'fixture.json',rolling_fixture(fixture))
    paths=[source/name for name in ('fixture.json','profile.json','prepared-proof.json','runtime-proof.json',
        'status.json','execution-audit.json','run/cleanup.json','inference-performance.json','confirmed-work-performance.json')]
    paths.extend(path for path in (source/'run/answers/cases').glob('*/*.json')
                 if path.name in ('request.json','response.json','result.json'))
    paths.extend(qualified/name for name in ('manifest.json','final-review.json','status.json','cleanup.json'))
    paths.extend(previous/name for name in ('active16k-score-audit.json','plan.json'))
    p.save(campaign/'rolling-variant.json',dict(source=str(source),qualification=str(qualified),
        source_files={str(path):p.sha(path) for path in paths},fresh_answers=True,
        authorization='User requested Active 16K / Gen 4K on the original LME102; compare accuracy, truncation and actual throughput, all other controls unchanged.',
        control_differences=['kvmem.gen_budget:16384->4096','kvmem.rolling_active_window:false->true',
                             'query_replay_budget:16384->4096','layerwise_rolling_decode:false->true'],
        native_batch_limit_preserved=rolling_profile(profile)['request_planning_batch_tokens']))
    p.save(campaign/'runtime-proof.json',{str(path.relative_to(runtime)):p.sha(path)
        for path in runtime.rglob('*') if path.is_file() and '__pycache__' not in path.parts})
    p.save(campaign/'prepared-proof.json',{name:p.sha(campaign/name) for name in
        ('fixture.json','profile.json','runtime-proof.json','rolling-variant.json')})
    memory.verify(campaign)
    p.save(campaign/'status.json',dict(status='prepared',suite='longmemeval_s',expected=102,at=time.time()))
    previous_plan=p.read(previous/'plan.json')
    policy_files={name:digest for name,digest in previous_plan['source_files'].items()
                  if '/evaluation/' in name or 'grading-policy' in name}
    for name,digest in policy_files.items():
        if p.sha(Path(name))!=digest:raise ValueError('Original official grading policy changed')
    command=list(previous_plan['grading_command'])
    command[2]=str(runtime/'scripts/grade_hybrid_memory.py')
    if p.sha(Path(command[2]))!=previous_plan['grader_sha256']:raise ValueError('Original judge adapter changed')
    for flag,value in (('--campaign',campaign),('--output',root/'grading-gen4k')):
        command[command.index(flag)+1]=str(value)
    plan=dict(campaign=str(campaign),source=str(source),qualified=str(qualified),grading_command=command,
        reference_scores=str(previous/'active16k-score-audit.json'),
        active_pointer=str(REPO/'artifacts/value-recall-active.json'),
        previous_owner=p.read(REPO/'artifacts/value-recall-active.json')['queue_root'],
        mab_pause=previous_plan['mab_pause'],scheduled_monitor=False,automatic_retry=False,
        grading_authorization='This new Gen4K batch requires explicit authorization before external DeepSeek Flash scoring.',
        files={str(path):p.sha(path) for path in (root/'launch.py',root/'memory_scores_audit.py',Path(command[2]))} | policy_files)
    p.save(root/'plan.json',plan)
    p.save(root/'status.json',dict(status='prepared',expected=102,at=time.time()))


def performance(campaign):
    perf=p.read(campaign/'inference-performance.json')
    work=p.read(campaign/'confirmed-work-performance.json')
    if perf['fresh_cases']!=102 or perf['inherited_cases'] or not work['all_submitted_completed']:
        raise ValueError('Performance requires 102 fresh, fence-confirmed responses')
    paths=sorted((campaign/'run/answers/cases').glob('*/result.json'))
    results=[p.read(path) for path in paths]
    events=[result_events(r,path.parent) for r,path in zip(results,paths)]
    wall=perf['inference_wall_seconds']
    return dict(inference_wall_seconds=wall,cases_per_hour=102*3600/wall,
        executed_prefill_tokens=work['completed']['public_prefill_tokens'],
        executed_prefill_tokens_per_wall_second=work['completed']['public_prefill_tokens']/wall,
        output_tokens=sum(r['usage']['completion_tokens'] for r in results),
        output_tokens_per_wall_second=sum(r['usage']['completion_tokens'] for r in results)/wall,
        mean_request_seconds=sum(r['seconds'] for r in results)/102,
        decode_refreshes=sum(e.get('event_type')=='decode_pressure' for es in events for e in es),
        cases_with_decode_refresh=sum(any(e.get('event_type')=='decode_pressure' for e in es) for es in events),
        concurrency_limit=perf['concurrency_limit'],
        scope='Whole inference wall time excluding model load and judge calls; token/s includes prefill, Query, maintenance and decode, not isolated kernel throughput.')


def compare(root,plan):
    old=p.read(Path(plan['reference_scores']));new=p.read(root/'gen4k-score-audit.json')
    if old['protocol']!=new['protocol'] or set(old['values'])!=set(new['values']) or len(new['values'])!=102:
        raise ValueError('Comparison inputs, sampling, question types or cohort differ')
    arms={name:dict(official_score=score['overall_score'],
                   correct=sum(score['values'].values()),truncated=score['truncated'],
                   truncation_rate=score['truncation_rate'],per_type=score['per_type'],performance=performance(campaign))
          for name,score,campaign in [('gen16k',old,Path(plan['source'])),('gen4k',new,Path(plan['campaign']))]}
    wins=[i for i in new['values'] if new['values'][i]>old['values'][i]]
    losses=[i for i in new['values'] if new['values'][i]<old['values'][i]]
    report=dict(status='completed',cases=102,active_budget=16384,arms=arms,
        score_delta_points=new['overall_score']-old['overall_score'],
        wall_speed_ratio=arms['gen16k']['performance']['inference_wall_seconds']/arms['gen4k']['performance']['inference_wall_seconds'],
        paired=dict(improved=wins,worsened=losses,tied=102-len(wins)-len(losses)),
        control_differences=p.read(Path(plan['campaign'])/'rolling-variant.json')['control_differences'],
        interpretation='Historical Gen16K versus fresh Gen4K rolling run; smaller physical window can also change prefill pressure cadence and final Query retained extent. Single run per arm.',at=time.time())
    p.save(root/'comparison.json',report)
    with (root/'per-case.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['id','type','gen16k_score','gen4k_score','delta'])
        for ident in sorted(new['values']):
            writer.writerow([ident,new['protocol'][ident]['type'],old['values'][ident],new['values'][ident],new['values'][ident]-old['values'][ident]])
    lines=['| Gen KV | Official score | Truncated | Inference minutes | Cases/hour |',
           '|---|---:|---:|---:|---:|']
    for name,arm in arms.items():
        lines.append(f"| {name} | {arm['official_score']:.2f} | {arm['truncated']}/102 | {arm['performance']['inference_wall_seconds']/60:.2f} | {arm['performance']['cases_per_hour']:.2f} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def run(root, *, inference_only=False, grade_only=False):
    import run_layerwise_memory as memory
    from run_recall_comparison import termination_guard,stop_owned,gpu_lock
    import benchmark_block_graph as lifecycle
    plan=p.read(root/'plan.json');campaign=Path(plan['campaign']);runtime=campaign/'runtime-v1'
    for name,digest in plan['files'].items():
        if p.sha(Path(name))!=digest:raise ValueError('Frozen queue changed')
    memory.verify(campaign)
    pointer=Path(plan['active_pointer'])
    with termination_guard(),(root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not inference_only:
            authorization=p.read(root/'grading-authorization.json')
            if (authorization.get('authorized') is not True or authorization.get('campaign')!=str(campaign) or
                    authorization.get('service')!='https://api.deepseek.com'):
                raise ValueError('This new batch lacks explicit external grading authorization')
        process_path=root/('grading-process.json' if grade_only else 'process.json')
        if process_path.exists():raise ValueError('Preserve prior attempt; no implicit retry')
        if grade_only:
            if p.read(root/'status.json')['status']!='awaiting_grading_authorization':
                raise ValueError('Only completed local inference may continue to grading')
        else:
            if p.read(Path(plan['mab_pause']))['status']!='paused':raise ValueError('MAB pause changed')
            with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
                if lifecycle.gpu_processes():raise ValueError('GPU has another owner')
                if p.read(pointer)['queue_root']!=plan['previous_owner']:raise ValueError('GPU scheduling owner changed')
                p.save(pointer,dict(queue_root=str(root),root=str(campaign),status=str(campaign/'status.json'),
                    process=str(campaign/'process.json'),queue_process=str(root/'process.json'),
                    experiment_type='layerwise_lme_gen4k_rolling',run_state='running',at=time.time()))
        p.save(process_path,dict(pid=os.getpid(),at=time.time()))
        env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime/'src')+os.pathsep+str(runtime/'scripts'))
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
        def launch(stage,command):
            if p.read(pointer)['queue_root']!=str(root):raise RuntimeError('Another owner acquired scheduling')
            p.save(root/'status.json',dict(status='running',stage=stage,at=time.time()))
            with (root/(stage+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            p.save(root/'child.json',dict(pid=child.pid,stage=stage,at=time.time()))
            try:code=child.wait()
            except BaseException:stop_owned(child);raise
            if code:raise RuntimeError('Preserve failed technical attempt: '+stage)
        try:
            if not grade_only:
                launch('inference',[sys.executable,'-B',str(runtime/'scripts/run_layerwise_memory.py'),'run','--root',str(campaign)])
            audit=p.read(campaign/'execution-audit.json');cleanup=p.read(campaign/'run/cleanup.json')
            if (audit['status']!='passed' or audit['cases']!=102 or audit['regenerated_completed_answers'] or
                    cleanup['gpu_processes'] or cleanup['remaining_owned_groups']):raise ValueError('Formal run incomplete')
            if inference_only:
                p.save(root/'local-performance-comparison.json',dict(gen16k=performance(Path(plan['source'])),
                    gen4k=performance(campaign),accuracy_pending=True))
                p.save(root/'status.json',dict(status='awaiting_grading_authorization',cases=102,at=time.time()))
                return
            launch('grading',plan['grading_command'])
            launch('score-audit',[plan['grading_command'][0],'-B',str(root/'memory_scores_audit.py'),
                '--grading',str(root/'grading-gen4k'),'--suite','longmemeval_s','--campaign',str(campaign),
                '--output',str(root/'gen4k-score-audit.json')])
            compare(root,plan)
            p.save(root/'status.json',dict(status='completed',cases=102,at=time.time()))
        except BaseException as error:
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
        finally:
            active=p.read(pointer)
            if active.get('queue_root')==str(root):
                active.update(run_state=p.read(root/'status.json')['status'],at=time.time());p.save(pointer,active)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','infer','grade','compare'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--qualified',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root,args.source.resolve(),args.qualified.resolve())
    elif args.action in ('run','infer','grade'):
        run(root,inference_only=args.action=='infer',grade_only=args.action=='grade')
    else:compare(root,p.read(root/'plan.json'))
