"""Audited recovery of a transport-failed baseline, preserving successful work."""
import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.run_agent_handoff_queue import read, save, sha, live, verify_frozen

SUCCESS={'completed','complete','done','stop','end-turn'}


def case_resume(case):
    events=[json.loads(l) for l in (case/'events.jsonl').read_text().splitlines() if l]
    results=sorted(case.glob('turn-*/result.json'))
    successful=0
    for path in results:
        result=read(path);reason=result['terminal']['data']['reason']
        if reason['kind'] in SUCCESS:
            successful+=1;continue
        if path!=results[-1] or reason.get('error',{}).get('code')!='TRANSPORT':
            raise ValueError('Only a final transport failure can be resumed')
        pending=set()
        for e in events:
            if e['type']=='tool/call':pending.add(e['data']['callId'])
            if e['type']=='tool/result':
                for part in e['data'].get('message',{}).get('content',[]):
                    pending.discard(part.get('toolCallId'))
        if pending:raise ValueError('Outstanding tools need a side-effect audit')
        return dict(successful_turns=successful,terminal_seq=result['terminal']['seq'],
            prior_steps=result['statistics']['model_steps'],prior_seconds=result['elapsed_seconds'],
            failed_turn_directory=path.parent.name)
    raise ValueError('No failed turn to recover')


def validate_source(source,probe=live):
    manifest=verify_frozen(source)
    if read(source/'status.json').get('status')!='failed':raise ValueError('Original queue not failed')
    if probe(read(source/'queue-process.json')) is not None:raise ValueError('Original queue still alive')
    for name in ('baseline','qualification'):
        cleanup=read(source/name/'cleanup.json')
        if cleanup.get('remaining_processes')!=[] or cleanup.get('remaining_owned_gpu_pids')!=[]:
            raise ValueError('Original cleanup incomplete')
        if any(probe(p) is not None for p in cleanup.get('owned',[])):raise ValueError('Original process alive')
    if read(source/'qualification/execution-audit.json').get('status')!='passed':raise ValueError('Qualification not passed')
    if read(source/'candidate/status.json').get('status')!='prepared' or (source/'candidate/started.json').exists():
        raise ValueError('Candidate already attempted')
    cases={}
    for row in read(source/'baseline/task-plan.json')['cases']:
        case=source/'baseline/cases'/row['id'];status=read(case/'status.json')
        if status['status']=='completed':
            if (case/'failure.json').exists():raise ValueError('Contradictory completed case')
        else:cases[row['id']]=case_resume(case)
    if set(cases)!={'r01-fps','r01-calendar'}:raise ValueError('Unexpected failed cohort; review separately')
    return manifest,cases


def prepare(source,destination,repository,probe=live):
    source=source.resolve();destination=destination.resolve();repository=repository.resolve()
    plan,cases=validate_source(source,probe)
    if destination.exists():raise ValueError('Destination already exists')
    destination.mkdir()
    for relative in plan['files']:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/relative,target)
    changes=[]
    for arm in ('baseline','candidate'):
        target=destination/arm/'runtime/src/vllm_kvmem/block_graph.py'
        old=(source/arm/'runtime/src/vllm_kvmem/block_graph.py').read_text()
        new=(repository/'src/vllm_kvmem/block_graph.py').read_text()
        # Only the tested accounting change may be overlaid onto the baseline.
        a=old.index('            upper = self.live_bytes * 2 + scratch')
        b=new.index('            # The existing live revision already owns its reservation.')
        end=new.index('            charge = self.budget.reserve(self._account, upper, self.max_bytes_per_session)',b)
        expected=old[:a]+new[b:end]+old[a+len('            upper = self.live_bytes * 2 + scratch\n'):]
        if new!=expected:raise ValueError('Unexpected graph semantic changes')
        target.write_text(new);changes.append(str(target.relative_to(destination)))
        save(destination/arm/'status.json',dict(status='prepared'))
    controller_files=['scripts/run_agent_handoff_queue.py','scripts/run_dsh_daily_suite.py',
        'scripts/summarize_dsh_daily_suite.py',
        'scripts/check_dsh_daily_browser.py','scripts/dsh_resume_failed_turn.mjs',
        'scripts/recover_agent_handoff_partial.py','tests/test_block_graph.py',
        'tests/test_dsh_resume_failed_turn.mjs','tests/test_agent_handoff_partial_recovery.py']
    for relative in controller_files:
        target=destination/'candidate/runtime'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repository/relative,target);changes.append(str(target.relative_to(destination)))
    # Completed responses/results and all failed attempts remain at source.
    # Seed the new journal with exact bytes; new writes append only after them.
    for name in ('cases','bridge'):
        shutil.copytree(source/'baseline'/name,destination/'baseline'/name)
    shutil.copytree(source/'baseline/workspaces',destination/'original-workspaces')
    (destination/'baseline/workspaces').mkdir()
    tasks=read(source/'baseline/task-plan.json');tasks['resume_cases']=cases
    tasks['prior_capture_roots']=[str(source/'baseline/server/capture')]
    save(destination/'baseline/recovery-history.json',dict(source=str(source/'baseline')))
    for row in tasks['cases']:
        # Absolute tool paths and the session cwd remain unchanged. Immutable
        # pre-recovery workspace contents are preserved in original-workspaces.
        (destination/'baseline/workspaces'/row['id']).symlink_to(Path(row['workspace']),target_is_directory=True)
        if row['id'] in cases:
            case=destination/'baseline/cases'/row['id'];prior=case/'prior-attempt';prior.mkdir()
            for filename in ('failure.json','status.json',cases[row['id']]['failed_turn_directory']):
                shutil.move(str(case/filename),str(prior/filename))
            save(case/'status.json',read(prior/'status.json'))
    save(destination/'baseline/task-plan.json',tasks);changes.append('baseline/task-plan.json')
    candidate_tasks=read(source/'candidate/task-plan.json')
    for row in candidate_tasks['cases']:
        row['workspace']=str(destination/Path(row['workspace']).relative_to(source))
    save(destination/'candidate/task-plan.json',candidate_tasks);changes.append('candidate/task-plan.json')
    (destination/'qualification').symlink_to(source/'qualification',target_is_directory=True)
    shutil.copy2(source/'gpu-component.log',destination/'gpu-component.log')
    plan['protocol']=str(destination/'protocol.so')
    plan['browser_python']='/tmp/kvmem-agent-browser-20261025/bin/python'
    plan['reuse_qualification']=dict(root=str(source),hashes={str(p.relative_to(source)):sha(p)
        for p in (source/'qualification').rglob('*') if p.is_file()},
        reason='Only CPU graph reservation bound and controller recovery changed; GPU restore/model/bridge code identical.')
    plan['resume_arms']={'baseline':dict(source=str(source/'baseline'),
        original_started=read(source/'baseline/status.json')['at'],
        original_finished=read(source/'baseline/summary.json')['at'],
        recovered_cases=list(cases),preserved_completed_cases=6)}
    allow=[]
    for row in tasks['cases']:
        if row['id'] in cases:
            allow.append(dict(session_id=row['session_id'],terminal_seq=cases[row['id']]['terminal_seq'],
                receipt=str(destination/'baseline/cases'/row['id']/'resume-receipt.json')))
    loop=Path('/home/chaidi/.local/share/deepseek-harness/dsh/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-agent-loop/lib/index.js')
    save(destination/'dsh-recovery-plan.json',dict(cases=allow,loop_file=str(loop),loop_sha256=sha(loop)))
    plugin=destination/'candidate/runtime/scripts/dsh_resume_failed_turn.mjs'
    (destination/'dsh-recovery.patch.yml').write_text('- insert:\n    - id: kvmem-failed-turn-recovery\n      name: '+str(plugin)+'\n      inject: [webServer, agents]\n')
    source_files={str(p.relative_to(source)):sha(p) for directory in ('baseline/cases','baseline/bridge')
        for p in (source/directory).rglob('*') if p.is_file()}
    workspace_proof={str(p.relative_to(destination/'original-workspaces')):sha(p)
        for p in (destination/'original-workspaces').rglob('*') if p.is_file()}
    save(destination/'repair-audit.json',dict(status='passed',source=str(source),at=time.time(),
        source_manifest_sha256=sha(source/'manifest.json'),preserved_source_evidence=source_files,
        preserved_workspace_snapshot=workspace_proof,overlay_files=changes,
        resumed_cases=cases,completed_cases_not_regenerated=6,
        inference_changes='None: one redundant temporary graph reservation removed; all scoring/selection/model parameters unchanged.',
        limitations='Engine RAM checkpoints were lost. Resume reconstructs accepted token ledger exactly, then cold-prefills that history. Baseline is interrupted; whole-run wall is not an uninterrupted 8-way throughput comparison. Use pre-failure common-8 intervals and fixed-trajectory tests for optimization attribution.'))
    # Freeze code and input files, not append-only recovery outputs/workspaces.
    frozen=set(plan['files'])|set(changes)|{'repair-audit.json','dsh-recovery-plan.json','dsh-recovery.patch.yml','baseline/recovery-history.json'}
    plan['files']={p:sha(destination/p) for p in sorted(frozen)}
    save(destination/'manifest.json',plan);save(destination/'status.json',dict(status='prepared',at=time.time()))
    verify_frozen(destination)
    return dict(root=str(destination),manifest_sha256=sha(destination/'manifest.json'),cases=cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','destination','repository'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.destination,args.repository)))
