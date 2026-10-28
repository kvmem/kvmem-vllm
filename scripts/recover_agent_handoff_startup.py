"""Recover only a failed qualification startup, before any answer was produced.

Never resumes a partially answered campaign. The old root remains read-only;
only frozen inputs are copied, and every old file is bound in the recovery audit.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.run_agent_handoff_queue import live, read, save, sha, verify_frozen


def validate_startup_failure(source, process_probe=live):
    plan=verify_frozen(source)
    status=read(source/'status.json')
    if status.get('status')!='failed' or status.get('error')!="TimeoutError('Owned service readiness timed out')":
        raise ValueError('Only the known readiness timeout is eligible')
    if process_probe(read(source/'queue-process.json')) is not None:
        raise ValueError('Original queue is still alive')
    cleanup=read(source/'qualification/cleanup.json')
    if cleanup.get('remaining_processes')!=[] or cleanup.get('remaining_owned_gpu_pids')!=[]:
        raise ValueError('Original qualification cleanup is incomplete')
    for owner in cleanup.get('owned',[]):
        if process_probe(owner) is not None:raise ValueError('Original owned process is still alive')
    qualification=source/'qualification'
    if list(qualification.glob('[0-9]*-[0-9]*.json')) or (qualification/'execution-audit.json').exists():
        raise ValueError('Qualification has attempted answers; requires partial recovery')
    capture=qualification/'server/capture'
    work=capture/'public_work.json'
    if work.exists() and any(read(work).get('completed',{}).values()):
        raise ValueError('Public model work occurred; requires partial recovery')
    allowed={'server-process.json','server.log','cleanup.json','startup.json','server/launch.json'}
    if any(str(p.relative_to(qualification)) not in allowed for p in qualification.rglob('*') if p.is_file()):
        raise ValueError('Unexpected qualification evidence; requires manual partial-recovery audit')
    for name in ('baseline','candidate'):
        d=source/name
        if read(d/'status.json').get('status')!='prepared':raise ValueError('Arm no longer prepared')
        if any((d/x).exists() for x in ('started.json','cases','bridge','server','workspaces')):
            raise ValueError('Arm already attempted; requires partial recovery')
    return plan


def prepare(source, destination, repository, process_probe=live):
    source=source.resolve();destination=destination.resolve();repository=repository.resolve()
    plan=validate_startup_failure(source,process_probe)
    if destination.exists():raise ValueError('Recovery destination already exists')
    evidence={str(p.relative_to(source)):sha(p) for p in source.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name!='queue.lock'}
    destination.mkdir()
    for relative in plan['files']:
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
        target=destination/p;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/p,target)
    changed=[]
    # These paths identify new output locations, never model input or policy.
    for name in ('baseline','candidate'):
        task_path=destination/name/'task-plan.json';tasks=read(task_path)
        for case in tasks['cases']:
            relative=Path(case['workspace']).relative_to(source)
            case['workspace']=str(destination/relative)
        save(task_path,tasks);changed.append(str(task_path.relative_to(destination)))
        preview=destination/name/'launch-preview.json'
        if preview.exists():
            preview.write_text(preview.read_text().replace(str(source),str(destination)))
            changed.append(str(preview.relative_to(destination)))
        save(destination/name/'status.json',dict(status='prepared'))
    for relative in ('scripts/run_agent_handoff_queue.py','scripts/recover_agent_handoff_startup.py',
                     'tests/test_agent_handoff_startup_recovery.py'):
        target=destination/'candidate/runtime'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repository/relative,target)
        changed.append(str(target.relative_to(destination)))
    plan['protocol']=str(destination/Path(plan['protocol']).relative_to(source))
    plan['model_startup_timeout_seconds']=3600
    plan['recovery_source']=str(source)
    plan['recovery_source_manifest_sha256']=sha(source/'manifest.json')
    differences=[]
    for relative, expected in plan['files'].items():
        actual=sha(destination/relative)
        if actual!=expected:
            if relative not in changed:raise ValueError('Unapproved frozen input difference: '+relative)
            differences.append(dict(path=relative,old=expected,new=actual))
    # All settings, exact qualification tokens and existing transcript audits
    # remain byte-identical. No old response/result is placed in a writable run.
    for relative in ('qualification-profile.json','qualification-fixture.json',
                     'baseline/profile.json','candidate/profile.json'):
        if sha(source/relative)!=sha(destination/relative):raise ValueError('Inference inputs changed')
    save(destination/'recovery-audit.json',dict(status='passed',source=str(source),at=time.time(),
        source_manifest_sha256=sha(source/'manifest.json'),source_evidence_sha256=evidence,
        frozen_input_differences=differences,added_files=[p for p in changed if p not in plan['files']],
        no_answers_or_public_forward=True,model_and_bridge_source_unchanged=True,
        reason='FlashInfer gemm_sm120 cold compilation was progressing when the 900s readiness deadline expired.',
        scope='Startup orchestration only: 3600s bounded deadline and startup process/CPU heartbeat.'))
    plan['files']={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()
        and str(p.relative_to(destination)) not in ('baseline/status.json','candidate/status.json')}
    save(destination/'manifest.json',plan)
    save(destination/'status.json',dict(status='prepared',at=time.time()))
    verify_frozen(destination)
    # Detect unexpected writes to the original attempt during preparation.
    for relative, expected in evidence.items():
        if sha(source/relative)!=expected:raise ValueError('Original evidence changed while preparing recovery')
    return dict(root=str(destination),manifest_sha256=sha(destination/'manifest.json'),
                audit_sha256=sha(destination/'recovery-audit.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.destination,args.repository)))
