"""Explicit-user restart of interrupted/unstarted tasks; retain every scored answer."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import run_terminalbench_suite as suite
import run_agent_handoff_queue as q

REPO=Path(__file__).resolve().parents[1]

def partition(tasks,outcomes,approved_errors=None):
    names={t['name'] for t in tasks}
    seen=set();saved=[];failed=[]
    for row in outcomes:
        name=row['task'].removeprefix('terminal-bench/')
        if name not in names or name in seen:raise ValueError('Invalid original task set')
        seen.add(name)
        if row['status']=='scored' and row.get('rewards') is not None:saved.append(row)
        elif row['status']=='error' and (row.get('exception')=='CancelledError' or
                (approved_errors or {}).get(name)==row.get('exception')):failed.append(name)
        else:raise ValueError('Unapproved retry category: '+name)
    keep={r['task'].removeprefix('terminal-bench/') for r in saved}
    return saved,sorted(names-keep),sorted(failed)


def source_outcomes(source, incomplete_trials=None):
    """Refuse to silently treat a started trial missing its result as unstarted."""
    status=q.read(source/'formal-events/status.json');rows=list(status['outcomes'])
    seen={r['trial'] for r in rows};proofs=incomplete_trials or {}
    trials={p.name:p for p in (source/'jobs/formal').iterdir()
            if p.is_dir() and (p/'config.json').exists()}
    missing=(set(trials)|set(status.get('active',[])))-seen
    if set(proofs)!=missing:raise ValueError('Missing exact incomplete-trial classification')
    events=[json.loads(s) for s in (source/'formal-events/events.jsonl').read_text().splitlines()]
    for trial in sorted(missing):
        folder=trials[trial];proof=proofs[trial];config=q.read(folder/'config.json')
        name=Path(config['task']['path']).name
        if proof['task']!='terminal-bench/'+name:raise ValueError('Incomplete trial task mismatch')
        if not proof.get('reason') or not proof.get('evidence'):raise ValueError('Missing interruption evidence')
        if any(q.sha(Path(path))!=digest for path,digest in proof['evidence'].items()):
            raise ValueError('Incomplete trial evidence changed')
        # A preserved answer/reward must be recovered, never regenerated.
        if ((folder/'result.json').exists() or (source/'formal-events'/(trial+'.json')).exists() or
                any((folder/'verifier').glob('reward*'))):
            raise ValueError('Unindexed result requires scoring review before restart')
        if proof['status']=='not_started':
            if (any(e.get('trial')==trial and e.get('event') in ('agent-start','environment-start') for e in events)
                    or any(p.is_file() for p in (folder/'agent').rglob('*'))):
                raise ValueError('Executed trial cannot be classified as unstarted')
        elif proof['status']=='error' and proof.get('exception')=='ControllerInterrupted':
            rows.append(dict(task=proof['task'],trial=trial,status='error',exception='ControllerInterrupted',
                rewards=None,passed=False,classification_source='preserved controller interruption audit'))
        else:raise ValueError('Unapproved incomplete trial classification')
    return rows


def verifier_recovered_outcomes(source, rows, recoveries):
    """Substitute only a reviewed verifier result, retaining the original answer."""
    result=[];used=set()
    for row in rows:
        task=row['task'].removeprefix('terminal-bench/')
        if task not in recoveries:
            result.append(row);continue
        recovery=Path(recoveries[task]);manifest=q.read(recovery/'manifest.json')
        audit=q.read(recovery/'execution-audit.json');cleanup=q.read(recovery/'cleanup.json')
        original=source/'jobs/formal'/row['trial']/'result.json'
        if (row.get('exception')!='VerifierTimeoutError' or row.get('rewards') is not None or
                manifest.get('source')!=str(source) or manifest.get('trial')!=row['trial'] or
                manifest.get('task')!=task or manifest.get('agent_launch_allowed') is not False or
                not audit.get('passed') or not audit.get('answer_unchanged') or not audit.get('no_model_requests') or
                audit.get('source_result_sha256')!=q.sha(original) or
                cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']):
            raise ValueError('Verifier recovery identity/evidence mismatch')
        for name,digest in q.read(recovery/'evidence-hashes.json').items():
            if q.sha(recovery/name)!=digest:raise ValueError('Verifier recovery evidence changed')
        value=q.read(recovery/'result.json');before=q.read(original)
        allowed={'verifier_result','exception_info','verifier_recovery'}
        if ({k:v for k,v in value.items() if k not in allowed}!={k:v for k,v in before.items() if k not in allowed} or
                value.get('trial_name')!=row['trial'] or value.get('exception_info') is not None):
            raise ValueError('Verifier recovery changed original Agent outcome')
        rewards=(value.get('verifier_result') or {}).get('rewards')
        if rewards is None or rewards.get('reward')!=audit.get('reward'):
            raise ValueError('Missing/mismatched verifier-only reward')
        result.append(dict(**{k:v for k,v in row.items() if k not in ('status','exception','rewards','passed')},
                           status='scored',exception=None,rewards=rewards,passed=rewards.get('reward')==1,
                           verifier_recovery=str(recovery),verifier_recovered_result=str(recovery/'result.json'),
                           original_result=str(original),original_result_sha256=q.sha(original)))
        used.add(task)
    if used!=set(recoveries):raise ValueError('Unknown verifier recovery task')
    return result


def prepare(root,source,qualification,superseded_preflight=None,retry_approval=None,cuda_launch_blocking=False,
            terminal_tool_bundles=None,verifier_recoveries=None,recorder_transport=False):
    if root.exists():raise ValueError('New root required')
    original=q.read(source/'manifest.json')
    for folder,expected in original['frozen'].items():
        if suite.hashes(source/folder)!=expected:raise ValueError('Original frozen files changed: '+folder)
    proof=q.read(qualification/'execution-audit.json');cleanup=q.read(qualification/'cleanup.json')
    if not proof['passed'] or cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
        raise ValueError('Service qualification/cleanup missing')
    if suite.hashes(qualification/'runtime')!=q.read(qualification/'manifest.json')['runtime']:
        raise ValueError('Qualification runtime changed')
    priorcleanup=q.read(source/'cleanup.json')
    if priorcleanup['remaining_processes'] or priorcleanup['remaining_owned_gpu_pids']:raise ValueError('Original cleanup incomplete')
    tasks=q.read(source/'tasks.json')
    inherited=q.read(source/'preserved-outcomes.json') if (source/'preserved-outcomes.json').exists() else []
    for row in inherited:
        if q.sha(Path(row['result']))!=row['result_sha256']:
            raise ValueError('Inherited result changed: '+row['task'])
    approvals={};incomplete={}
    if retry_approval is not None:
        approval=q.read(retry_approval)
        if approval['source']!=str(source) or approval['source_manifest_sha256']!=q.sha(source/'manifest.json'):
            raise ValueError('Retry approval belongs to a different attempt')
        approvals=approval['errors']
        incomplete=approval.get('incomplete_trials',{})
    current=verifier_recovered_outcomes(source,source_outcomes(source,incomplete),verifier_recoveries or {})
    # A finished answer must be preserved for verifier recovery, never sent
    # through the ordinary fresh-Agent restart path.
    for row in current:
        if row.get('status')!='error':continue
        path=source/'jobs/formal'/row['trial']/'result.json'
        if (str(row.get('exception','')).startswith('Verifier') and path.exists() and
                (q.read(path).get('agent_execution') or {}).get('finished_at')):
            raise ValueError('Finished answer requires verifier-only review: '+row['task'])
    preserved,launched,interrupted=partition(tasks,inherited+current,approvals)
    root.mkdir(parents=True)
    for folder in ('runtime','dataset','smoke-task','download-cache'):
        shutil.copytree((qualification if folder=='runtime' else source)/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run_terminalbench_job.py','terminalbench_harness.py','terminalbench_pty.py','terminalbench_tool_bundle.py','terminalbench_view.py','finalize_terminalbench.py','restart_terminalbench_remaining.py'):
        shutil.copy2(REPO/'scripts'/name,root/'runtime/scripts'/name)
    model_delta=[rel for rel,h in suite.hashes(root/'runtime').items()
                 if rel.startswith(('src/','csrc/')) and original['frozen']['runtime'].get(rel)!=h]
    expected_delta=q.read(qualification/'manifest.json').get('model_changes',
        ['csrc/qw3_mean_k.cu','src/vllm_kvmem/cuda_sources/qw3_mean_k.cu','src/vllm_kvmem/async_refresh.py'])
    if set(model_delta)!=set(expected_delta):
        raise ValueError('Unexpected model source changes: '+str(model_delta))
    # Suite controller has no new model behavior. The prior frozen version is used.
    for name in original['files']:
        if name=='reused-images.json':continue
        shutil.copy2(source/name,root/name)
    for name in ('job.json','smoke-job.json','download-cache-compose.json'):
        (root/name).write_text((root/name).read_text().replace(str(source),str(root)))
    config=q.read(root/'job.json');config['datasets'][0]['task_names']=launched
    q.save(root/'job.json',config)
    shutil.copy2(source/'connection.json',root/'connection.json');(root/'connection.json').chmod(0o600)
    proofs={p.stem:dict(proof=q.read(p),sha256=q.sha(p)) for p in (source/'images').glob('*.json')}
    q.save(root/'reused-images.json',dict(source=str(source),images=proofs,scope='Exact local immutable image reuse, never reuse task workspaces'))
    preserved_hashes={}
    for folder in ('runtime','jobs','formal-events','bridge','model'):
        for p in sorted((source/folder).rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
                preserved_hashes[str(p.relative_to(source))]=q.sha(p)
    for path in source.iterdir():
        # The historical read-only view can still append its own access log.
        # All stopped experiment logs, resources and control evidence are bound.
        if path.is_file() and not path.is_symlink() and path.name!='view.log':
            preserved_hashes[path.name]=q.sha(path)
    q.save(root/'original-evidence-hashes.json',dict(source=str(source),hashes=preserved_hashes))
    preserved_records=[]
    for row in preserved:
        if row in inherited:
            preserved_records.append(row.copy())
            continue
        p=Path(row['verifier_recovered_result']) if row.get('verifier_recovered_result') else source/'formal-events'/(row['trial']+'.json')
        value=q.read(p)
        if value['verifier_result']['rewards']!=row['rewards']:raise ValueError('Original reward mismatch')
        execution=value.get('agent_execution') or {};seconds=None
        if execution.get('started_at') and execution.get('finished_at'):
            seconds=(datetime.fromisoformat(execution['finished_at'])-datetime.fromisoformat(execution['started_at'])).total_seconds()
        preserved_records.append(dict(**row,source_root=str(source),result=str(p),result_sha256=q.sha(p),agent_seconds=seconds))
    q.save(root/'preserved-outcomes.json',preserved_records)
    bundle_paths={}
    if terminal_tool_bundles:
        from terminalbench_tool_bundle import validate_bundle
        if not set(terminal_tool_bundles).issubset(launched):raise ValueError('Bundle task not launched')
        for task,path in terminal_tool_bundles.items():
            image=next(t['environment']['docker_image'] for t in tasks if t['name']==task)
            validate_bundle(path,task,image)
            target=root/'terminal-tool-bundles'/task;target.mkdir(parents=True)
            for name in ('manifest.json','packages.tar'):shutil.copy2(Path(path)/name,target/name)
            bundle_paths[task]=str(target.relative_to(root))
    (root/'original-attempt').symlink_to(source,target_is_directory=True)
    upstream=Path('/tmp/kvmem-terminus-20261026-env/lib/python3.12/site-packages/harbor/agents/terminus_2')
    upstream_sources={n:q.sha(upstream/n) for n in ('tmux_session.py','terminus_2.py','terminus_json_plain_parser.py')}
    q.save(root/'restart-audit.json',dict(at=time.time(),source=str(source),source_manifest_sha256=q.sha(source/'manifest.json'),
        preserved=len(preserved),launched=len(launched),interrupted_restarted=interrupted,
        never_started=sorted(set(launched)-set(interrupted)),prior_failures_preserved=True,
        source_evidence_sha256=q.sha(root/'original-evidence-hashes.json'),
        original_prompts_verifiers_resources_unchanged=suite.hashes(root/'dataset')==original['frozen']['dataset'],
        service_qualification=str(qualification),service_audit_sha256=q.sha(qualification/'execution-audit.json'),
        enhanced_harness=True,mixed_harness_cohort=True,no_best_of=True,no_overall_agent_deadline=True,
        changes=['Current repair is bound by the service qualification and frozen model/bridge delta records',
                 'Require explicit hash-bound classification for trials missing their final result'],
        inherited_repairs=['New Query import capacity and native lookahead checked before fenced pressure restore in r9',
                 'Readonly observation timeouts, literal NUL parsing and Bash hook acknowledgement repaired in r8',
                 'Recursive preserved-result evidence validation and accurate numeric final task count',
                 'MeanK shared tail padding and Query concatenation event fence already present in r7',
                 'Bash completion, standalone control keys, interrupt confirmation and bounded download stalls'],
        model_delta=[rel for rel,h in suite.hashes(root/'runtime').items() if rel.startswith(('src/','csrc/')) and original['frozen']['runtime'].get(rel)!=h],
        bridge_delta=[rel for rel,h in suite.hashes(root/'runtime').items() if rel.startswith('benchmarks/') and original['frozen']['runtime'].get(rel)!=h]))
    fixed=list(original['files'])+['preserved-outcomes.json','original-evidence-hashes.json','restart-audit.json']
    q.save(root/'execution-mode.json',dict(cuda_launch_blocking=bool(cuda_launch_blocking),
        scope='Synchronous CUDA error attribution; model changes bound separately; sampling and budgets unchanged',
        throughput_comparable_to_async=False if cuda_launch_blocking else None,
        unresolved='Historical r7 CUDA illegal access has not been reproduced; component passes do not prove its cause repaired'))
    fixed.append('execution-mode.json')
    if retry_approval is not None:
        shutil.copy2(retry_approval,root/'retry-approval.json');fixed.append('retry-approval.json')
    if superseded_preflight is not None:
        prior=superseded_preflight.resolve()
        pc=q.read(prior/'cleanup.json')
        if (prior/'formal-process.json').exists() or (prior/'formal-events').exists():
            raise ValueError('Preflight replacement must not have launched formal tasks')
        if pc['remaining_processes'] or pc['remaining_owned_gpu_pids']:
            raise ValueError('Preflight cleanup incomplete')
        q.save(root/'superseded-preflight.json',dict(root=str(prior),status=q.read(prior/'status.json'),
            error='Recorder process name misclassified nested Bash despite acknowledged prompt hook',
            fix='Completion uses the inner Bash hook and busy marker; process diagnostics follow its actual PID',
            no_formal_attempts=True,hashes=suite.hashes(prior)))
        (root/'superseded-preflight').symlink_to(prior,target_is_directory=True)
        fixed.append('superseded-preflight.json')
    manifest={**original,'created':time.time(),'launched_task_names':launched,'preserved_scored':len(preserved),
        'harness_repair':dict(upstream_sources=upstream_sources,policy='completion-control-network-v1',
                              recorder_transport=bool(recorder_transport)),
        'terminal_tool_bundles':bundle_paths,
        'frozen':{name:suite.hashes(root/name) for name in ('runtime','dataset','smoke-task','download-cache',*(['terminal-tool-bundles'] if bundle_paths else []))},
        'files':{name:q.sha(root/name) for name in set(fixed)},
        'notes':[f'Custom enhanced Terminus harness; {len(preserved)} prior scored outcomes retained including failures; {len(launched)} new attempts',
                 'Mixed-source selected table is not a uniform stock-Terminus or uninterrupted evaluation.',
                 'Task instructions, seed, budgets, verifiers and resource allocations unchanged; no timer created.']}
    q.save(root/'manifest.json',manifest);q.save(root/'status.json',dict(status='prepared',expected=89,preserved=len(preserved),remaining=len(launched),at=time.time()))


def verify_history(root,seen=None):
    seen=set() if seen is None else seen
    if root.resolve() in seen:raise ValueError('Cyclic experiment evidence chain')
    seen.add(root.resolve())
    audit=q.read(root/'original-evidence-hashes.json');source=Path(audit['source'])
    for name,digest in audit['hashes'].items():
        if q.sha(source/name)!=digest:raise ValueError('Original evidence changed: '+name)
    count=len(audit['hashes'])
    if (source/'original-evidence-hashes.json').exists():count+=verify_history(source,seen)
    return count


def review_selected(root):
    verified=verify_history(root)
    if (root/'superseded-preflight.json').exists():
        prior=q.read(root/'superseded-preflight.json')
        for name,digest in prior['hashes'].items():
            if q.sha(Path(prior['root'])/name)!=digest:raise ValueError('Preflight evidence changed: '+name)
    preserved=q.read(root/'preserved-outcomes.json');current=q.read(root/'results.json')
    for row in preserved:
        if row.get('result') and q.sha(Path(row['result']))!=row['result_sha256']:
            raise ValueError('Inherited result changed: '+row['task'])
    selected=preserved+[{**row,'source_root':str(root),'enhanced_harness':True} for row in current['tasks']]
    names={t['name'] for t in q.read(root/'tasks.json')}
    if len(selected)!=89 or {r['task'].removeprefix('terminal-bench/') for r in selected}!=names:
        raise ValueError('Selected cohort incomplete or duplicate')
    final=dict(expected=89,scored=sum(r['status']=='scored' for r in selected),passed=sum(r['passed'] for r in selected),
        outcomes=selected,preserved_original=len(preserved),new_attempts=len(current['tasks']),
        mixed_harness_cohort=True,original_failures_retained=True,throughput_scope='New attempts only; previous failed window not merged')
    q.save(root/'selected-results.json',final)
    report=root/'selected-report.txt'
    mode=q.read(root/'execution-mode.json') if (root/'execution-mode.json').exists() else {}
    report.write_text(f'Terminal-Bench 2.1 selected cohort: {final["passed"]}/89 passed; {final["scored"]} scored.\n'
        f'{len(preserved)} original scores retained, {len(current["tasks"])} fresh attempts with enhanced Terminus.\n'
        'All prior failed attempts remain preserved through original-attempt. No low-score retries or best-of selection.\n'
        'Mixed harness/model-repair cohort; not stock official or uninterrupted performance. Agent deadline removed; original verifier/resources retained.\n'
        +('CUDA synchronous diagnostic mode enabled: timings are not comparable with normal asynchronous serving.\n'
          if mode.get('cuda_launch_blocking') else '')+
        'New-attempt timing and throughput: report.txt / execution-audit.json.\n\n'+
        '\n'.join(f'{r["task"]} | reward={r["rewards"]} | {r["status"]} | {r["source_root"]}' for r in sorted(selected,key=lambda x:x['task']))+'\n')
    review=dict(status='completed',at=time.time(),report=str(report),report_sha256=q.sha(report),
        original_evidence_verified=verified,**{k:v for k,v in final.items() if k!='outcomes'})
    q.save(root/'selected-final-review.json',review);q.save(root/'status.json',review)


def run(root):
    manifest=q.read(root/'manifest.json')
    if 'execution-mode.json' in manifest['files']:
        if q.sha(root/'execution-mode.json')!=manifest['files']['execution-mode.json']:
            raise ValueError('Execution mode changed')
        mode=q.read(root/'execution-mode.json')
        os.environ['CUDA_LAUNCH_BLOCKING']='1' if mode['cuda_launch_blocking'] else '0'
    suite.run(root)
    try:review_selected(root)
    except Exception as e:
        q.save(root/'status.json',dict(status='needs_final_review',error=repr(e),at=time.time()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--qualification',type=Path);p.add_argument('--superseded-preflight',type=Path);p.add_argument('--retry-approval',type=Path);p.add_argument('--cuda-launch-blocking',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.root.resolve(),a.source.resolve(),a.qualification.resolve(),a.superseded_preflight,a.retry_approval,a.cuda_launch_blocking)
    else:run(a.root.resolve())
