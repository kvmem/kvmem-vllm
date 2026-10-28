"""Finite natural DeepSWE observation windows after component qualification.

New isolated attempts measure actual tool/LLM execution. Window truncation is
not a task score. Original paused experiments and completed answers are reused
only as provenance; they are never resumed or overwritten here.
"""
import argparse
import array
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from scripts import run_agent_handoff_queue as q
from scripts.queue_agent_interference import completed

ROOT=Path(__file__).resolve().parents[1]
CLIENT_SHA='0bc3800ba6bcf8b9df2daa47509f2159a4c2b74b67922424b4e94e1019c89433'
OVERLAYS=['scripts/benchmark_deepswe_first16.py','scripts/deepswe_window_snapshot.py',
          'scripts/benchmark_deepswe_interference.py','scripts/queue_agent_interference.py',
          'scripts/deepswe_evidence.py']


def check_frozen_entrypoints(runtime, python):
    """Resolve controller dependencies from the frozen tree before GPU use."""
    env=os.environ.copy()
    env['PYTHONPATH']=':'.join(str(runtime/p) for p in ('','src','scripts'))
    rows=[]
    for name in ('benchmark_deepswe_interference.py','benchmark_deepswe_first16.py'):
        result=subprocess.run([python,str(runtime/'scripts'/name),'--help'],
            cwd=runtime,env=env,capture_output=True,text=True,timeout=30)
        if result.returncode:
            raise RuntimeError('Frozen entrypoint import failed: '+name+'\n'+result.stderr)
        rows.append(dict(entrypoint=name,returncode=result.returncode))
    return dict(status='passed',entrypoints=rows)


def validate_profiles(control,candidate):
    profiles=copy.deepcopy([control,candidate])
    for profile in profiles:
        if profile.get('verify_session_restore') or profile.get('verify_session_restore_digest'):
            raise ValueError('Diagnostic verification cannot enter performance windows')
        if profile['kvmem'].get('host_stage_timing') or profile['kvmem'].get('restore_stage_timing'):
            raise ValueError('Diagnostic timing cannot enter performance windows')
        for key in ('native_checkpoint_capture','checkpoint_pinned_prewarm_count','prefill_batch_tokens','prefill_request_tokens'):
            profile['kvmem'].pop(key,None)
    if profiles[0]!=profiles[1]:
        raise ValueError('Unexpected model, recall, memory budget or sampling difference')


def validate_preserved_control(control, reference, profile, seconds):
    state=q.read(control/'run.json');cleanup=q.read(control/'cleanup.json');manifest=q.read(control/'manifest.json')
    if (state['status']!='window_complete' or state.get('cleanup_errors')
            or cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']
            or q.read(control/'execution-audit.json')['status']!='passed'):
        raise ValueError('Preserved control must have complete observation and cleanup')
    expected=copy.deepcopy(profile);expected['port']=18878
    if (q.read(control/'profile.json')!=expected or manifest['duration_seconds']!=seconds
            or manifest['tasks']!=reference['tasks'] or manifest['claude_sha256']!=CLIENT_SHA):
        raise ValueError('Preserved control workload/profile differs')


def qualified_proof_arms(source,candidate):
    plan=q.read(source/'manifest.json')
    proof=['qualification','control']
    if candidate in plan.get('correctness_only_candidates',[]):
        # This new scheduling candidate has a GPU correctness run, but no
        # component performance claim. Natural measurement is its next test.
        qualified=q.read(source/'qualification/profile.json');profile=q.read(source/candidate/'profile.json')
        if not qualified.get('verify_session_restore') or not qualified.get('verify_session_restore_digest'):
            raise ValueError('Correctness-only candidate requires bounded GPU restore verification')
        qualified['verify_session_restore']=qualified['verify_session_restore_digest']=False
        if qualified!=profile:
            raise ValueError('Natural candidate differs from GPU-qualified profile')
    else:proof.append(candidate)
    for arm in proof:completed(source,arm)
    return proof


def prepare(source, formal, output, candidate, client, seconds, reuse_control=None):
    if output.exists() or candidate not in ('capture','prewarm','cap512','cap256','per_request256'):
        raise ValueError('New root and tested candidate required')
    if seconds < 900 or not seconds < 3600:
        raise ValueError('Predeclared observation window must be 15-59 minutes')
    proof_arms=qualified_proof_arms(source,candidate)
    if q.sha(client)!=CLIENT_SHA:raise ValueError('Client must match real Write/Read qualification')
    source_plan=q.read(source/'manifest.json')
    for rel,h in source_plan['files'].items():
        if q.sha(source/'runtime'/rel)!=h:raise ValueError('Component runtime changed')
    for arm in ('control',candidate):
        if q.sha(source/arm/'profile.json')!=source_plan['arms'][arm]['profile_sha256']:
            raise ValueError('Component profile changed')
    validate_profiles(q.read(source/'control/profile.json'),q.read(source/candidate/'profile.json'))
    reference=q.read(formal/'first16/manifest.json')
    if len(reference['tasks'])!=16 or reference['workers']!=8:raise ValueError('Original cohort changed')
    for task in reference['tasks']:
        if q.sha(Path(task['instruction']))!=task['instruction_sha256']:raise ValueError('Task changed')
        for image,h in task['images'].items():
            actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
            if actual!=h:raise ValueError('Task image changed')
    if reuse_control is not None:
        validate_preserved_control(reuse_control,reference,q.read(source/'control/profile.json'),seconds)
        from scripts.deepswe_evidence import verify_evidence
        verify_evidence(reuse_control)
    output.mkdir(parents=True);runtime=output/'runtime'
    shutil.copytree(source/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in OVERLAYS:shutil.copy2(ROOT/rel,runtime/rel)
    q.save(output/'controller-import-preflight.json',check_frozen_entrypoints(runtime,source_plan['python']))
    proof=ROOT/'artifacts/agent-interference-optimization-20261025-r1/client-patched-real.json'
    qualified=q.read(proof)
    if qualified['client_sha256']!=CLIENT_SHA or qualified['exit'] or not qualified['exact_file'] or not qualified['exact_history']:
        raise ValueError('Missing actual-client qualification')
    shutil.copy2(proof,output/'client-qualification.json')
    edit=proof.with_name('client-edit-real.json')
    if not q.read(edit).get('exact_edit_history') or not q.read(edit).get('exact_file'):
        raise ValueError('Missing actual Edit qualification')
    shutil.copy2(edit,output/'client-edit-qualification.json')
    snapshot_proof=proof.with_name('workspace-snapshot-real.json')
    snapshot_source=q.read(proof.with_name('workspace-snapshot-source.json'))
    if (q.read(snapshot_proof)['status']!='passed' or q.sha(snapshot_proof)!=snapshot_source['qualification_sha256']
            or q.sha(ROOT/'scripts/deepswe_window_snapshot.py')!=snapshot_source['sha256']):
        raise ValueError('Workspace snapshot qualification changed')
    shutil.copy2(snapshot_proof,output/'workspace-snapshot-qualification.json')
    shutil.copy2(client,output/'claude-exact-json')
    shutil.copy2(Path(str(client)+'.audit.json'),output/'client-patch-audit.json')
    fixture=q.read(source/'fixture.json')
    count=max(67584,reference['profile']['kvmem']['select_budget']+2048)
    tokens=array.array('i',fixture[0]['tokens'][:count])
    if sys.byteorder!='little':tokens.byteswap()
    (output/'warmup-tokens.bin').write_bytes(tokens.tobytes())
    for arm,origin in [('control','control'),('candidate',candidate)]:
        directory=output/arm
        if arm=='control' and reuse_control is not None:
            directory.symlink_to(reuse_control.resolve(),target_is_directory=True)
            continue
        directory.mkdir()
        profile=q.read(source/origin/'profile.json')
        if profile['verify_session_restore'] or profile.get('verify_session_restore_digest') or profile['kvmem']['host_stage_timing']:
            raise ValueError('Performance windows must disable qualification instrumentation')
        profile['port']=18878
        q.save(directory/'profile.json',profile)
        shutil.copytree(formal/'first16/setup',directory/'setup')
        script=directory/'setup/run_deepswe_cell.sh'
        old='/home/chaidi/.local/share/claude/versions/2.1.148:/usr/local/bin/claude:ro'
        text=script.read_text()
        if text.count(old)!=1:raise ValueError('Unknown client bind mount')
        script.write_text(text.replace(old,str(output/'claude-exact-json')+':/usr/local/bin/claude:ro'))
        (directory/'protocol').mkdir()
        for name in ('protocol.so','provenance.json'):
            shutil.copy2(formal/'first16/protocol'/name,directory/'protocol'/name)
        shutil.copy2(formal/'first16/tasks.txt',directory/'tasks.txt')
        manifest=copy.deepcopy(reference)
        manifest.pop('preflight_reuse',None);manifest['carried_results']=[]
        manifest.update(profile=profile,profile_sha256=q.sha(directory/'profile.json'),
            duration_seconds=seconds,claude_sha256=CLIENT_SHA,
            setup_sha256={p.name:q.sha(p) for p in (directory/'setup').iterdir() if p.is_file()},
            original_formal_root=str(formal),original_manifest_sha256=q.sha(formal/'first16/manifest.json'),
            client_patch='Both arms preserve parsed JSON strings, qualified using actual client Write/Read and next-request history.',
            observation_only=True,snapshot_window_workspaces=True)
        q.save(directory/'manifest.json',manifest)
    q.save(output/'manifest.json',dict(source=str(source),formal=str(formal),candidate=candidate,seconds=seconds,
        python=source_plan['python'],execution_order=['candidate'] if reuse_control is not None else ['control','candidate'],
        preserved_control=dict(root=str(reuse_control.resolve()),evidence_sha256=q.sha(reuse_control/'evidence-hashes.json'),
            scope_sha256=q.sha(reuse_control/'evidence-hash-scope.json')) if reuse_control is not None else None,
        source_manifest_sha256=q.sha(source/'manifest.json'),
        component_proof={f'{a}/{f}':q.sha(source/a/f) for a in proof_arms
                         for f in ('execution-audit.json','cleanup.json','evidence-hashes.json')},
        files={str(p.relative_to(output)):q.sha(p) for p in output.rglob('*') if p.is_file()},
        scope='Fresh natural DeepSWE tasks, same prompts/sampling/limits. Sequential single observation windows; trajectories may differ. '
              'Both use qualified client JSON fix. Interrupted original experiment is preserved. '
              'Accepted output/measurement wall, not MTP verification rows. Unfinished tasks remain unscored; filesystem snapshots retained.'))
    q.save(output/'status.json',dict(status='prepared'))


def run(root):
    if ROOT.resolve()!=(root/'runtime').resolve():raise ValueError('Frozen controller required')
    plan=q.read(root/'manifest.json')
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'started.json').exists():raise ValueError('Never retry attempted natural tasks')
        for rel,h in plan['files'].items():
            if q.sha(root/rel)!=h:raise ValueError('Frozen input changed: '+rel)
        for rel,h in plan['component_proof'].items():
            if q.sha(Path(plan['source'])/rel)!=h:raise ValueError('Component proof changed')
        if plan.get('preserved_control'):
            proof=plan['preserved_control'];control=Path(proof['root'])
            if (q.sha(control/'evidence-hashes.json')!=proof['evidence_sha256']
                    or q.sha(control/'evidence-hash-scope.json')!=proof['scope_sha256']):
                raise ValueError('Preserved control proof changed')
            from scripts.deepswe_evidence import verify_evidence
            verify_evidence(control)
        q.save(root/'started.json',dict(**q.identity(psutil.Process()),at=time.time()))
        for arm in plan['execution_order']:
            directory=root/arm
            if (directory/'run.json').exists():raise ValueError('Attempt already exists')
            if q.gpu_pids():raise RuntimeError('GPU occupied; leave owner untouched')
            q.require_free_port(18878)
            owned=q.Owned()
            try:
                q.save(root/'status.json',dict(status='running',active=arm,at=time.time()))
                env=q.environment(ROOT)
                child=owned.start([plan['python'],str(ROOT/'scripts/benchmark_deepswe_first16.py'),
                        '--config',str(directory/'profile.json'),'--output',str(directory),'--workers','8',
                        '--duration-seconds',str(plan['seconds']),'--warmup-tokens',str(root/'warmup-tokens.bin'),
                        '--use-prepared','--optimized-handoff','--bash-repeat-limit','32',
                        '--snapshot-window-workspaces','--bridge-port','18485','--startup-timeout-seconds','3600'],
                        cwd=ROOT,env=env,log=directory/'controller.log',record=directory/'controller-process.json')
                while child.poll() is None:
                    owned.track();time.sleep(1)
                if child.returncode:raise RuntimeError('Natural observation window failed: '+arm)
                state=q.read(directory/'run.json')
                if state['status']!='window_complete' or state.get('cleanup_errors'):
                    raise RuntimeError('Observation or cleanup incomplete: '+arm)
                if q.read(directory/'window-workspaces.json')['status']!='preserved':raise RuntimeError('Workspaces missing')
                q.save(directory/'execution-audit.json',dict(status='passed',scope='Finite observation, not complete DeepSWE task evaluation',
                    tasks=len(state['tasks']),snapshots=len(q.read(directory/'window-workspaces.json')['containers'])))
            finally:
                # On snapshot failure the benchmark deliberately retains cell
                # wrappers. Do not override that protection with a tree kill.
                state=q.read(directory/'run.json') if (directory/'run.json').exists() else {}
                if state.get('window_snapshot_error'):
                    q.save(directory/'cleanup.json',dict(status='needs_workspace_recovery',error=state['window_snapshot_error']))
                else:q.save(directory/'cleanup.json',owned.stop())
            cleanup=q.read(directory/'cleanup.json')
            if cleanup.get('remaining_processes') or cleanup.get('remaining_owned_gpu_pids'):
                raise RuntimeError('Owned cleanup incomplete')
            from scripts.deepswe_evidence import write_evidence
            write_evidence(directory)
        q.save(root/'status.json',dict(status='needs_analysis',at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path)
    parser.add_argument('--formal',type=Path);parser.add_argument('--candidate');parser.add_argument('--client',type=Path)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--reuse-control',type=Path)
    args=parser.parse_args()
    if args.source:prepare(args.source.resolve(),args.formal.resolve(),args.output.resolve(),args.candidate,args.client.resolve(),args.seconds,args.reuse_control)
    else:
        try:run(args.output.resolve())
        except BaseException as error:
            q.save(args.output/'failure.json',dict(error=repr(error),at=time.time()));raise
