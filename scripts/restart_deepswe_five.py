"""Fresh attempts of five explicitly authorized interrupted seed1004 tasks.

No saved workspace, client history, continuation prompt, or KV is reused.
Original59 scored answers and all interrupted/superseded attempts remain intact.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from scripts import recover_deepswe_seed_snapshot as common
from scripts.recover_deepswe_seed_snapshot import TASKS,read,sha,save,replace_once

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN='restart-1004'


def patch_controller(text):
    text=replace_once(text,'from scripts.queue_deepswe_seed_suite import validate_preflight_reuse',
        'from scripts.recover_deepswe_seed_snapshot import validate_preflight_reuse')
    return replace_once(text,"len(state['tasks'])==16 and not stop.is_set()",
                        "len(state['tasks'])==len(m['tasks']) and not stop.is_set()")


def preserved_predecessor(root):
    from scripts.run_agent_handoff_queue import live
    if live(read(root/'queue-process.json')):raise ValueError('Superseded continuation still running')
    cleanup=read(root/'cleanup.json')
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
        raise ValueError('Superseded continuation has not cleaned up')
    run=read(root/'recovery-1004/first16/run.json')
    if run.get('cleanup_errors') or not run.get('cleanup_finished_at') or run.get('window_snapshot_error'):
        raise ValueError('Superseded continuation preservation incomplete')
    return {str(p):sha(p) for p in (root/'manifest.json',root/'status.json',root/'cleanup.json',
        root/'recovery-1004/first16/run.json',root/'recovery-1004/first16/window-workspaces.json',root/'user-restart-request.json') if p.is_file()}


def validate_task_subset(task,original):
    expected=[t for t in original['tasks'] if t['task_id'] in TASKS]
    if task['tasks']!=expected or len(expected)!=5:raise ValueError('Fresh task list changed')
    for key in ('sampling','seed','profile','workers','optimized_handoff','termination_policies'):
        if task[key]!=original[key]:raise ValueError('Original task settings changed: '+key)
    if task['seed']!=1004 or task.get('carried_results'):raise ValueError('Fresh attempts require original seed and no carried results')


def validate_fresh(directory):
    root=directory.parent.parent;m=read(root/'manifest.json');task=read(directory/'manifest.json')
    if m['recovery'].get('mode')!='fresh_restart' or directory!=root/CAMPAIGN/'first16':
        raise ValueError('Wrong fresh experiment path/mode')
    if m['recovery'].get('prompt') is not None:raise ValueError('Fresh attempt must not append a continuation prompt')
    validate_task_subset(task,read(Path(m['source'])/'seed-1004/first16/manifest.json'))
    if sha(directory/'manifest.json')!=m['files'][str(directory.relative_to(root)/'manifest.json')]:
        raise ValueError('Fresh task manifest changed')
    for p,digest in m['qualification_proof'].items():
        if sha(p)!=digest:raise ValueError('Qualification proof changed')
    for p,digest in m['external_files'].items():
        if sha(p)!=digest:raise ValueError('Original external input changed')
    return m


def prepare(source,output,predecessor,approval):
    from scripts.deepswe_tool_cpu_limits import freeze_recovery_setup
    from scripts.queue_deepswe_fastpath import verify_images
    original=common.validate_source(source)
    authorized=read(approval)
    if not authorized.get('approved') or set(authorized.get('task_ids',[]))!=set(TASKS) or authorized.get('seed')!=1004:
        raise ValueError('Explicit approval required for precisely these five fresh attempts')
    before=preserved_predecessor(predecessor)
    if output.exists():raise ValueError('Use a new experiment root; no overwrite/retry')
    old=source/'seed-1004/first16';original_task=read(old/'manifest.json')
    output.mkdir(parents=True);runtime=output/'runtime'
    shutil.copytree(source/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('restart_deepswe_five.py','recover_deepswe_seed_snapshot.py','deepswe_recovery_view.py'):
        shutil.copy2(ROOT/'scripts'/name,runtime/'scripts'/name)
    shutil.copy2(source/'warmup-tokens.bin',output/'warmup-tokens.bin')
    shutil.copy2(approval,output/'approval.json')
    for seed in original['seeds']:(output/f'seed-{seed}').symlink_to(source/f'seed-{seed}',target_is_directory=True)
    (output/'superseded-continuation').symlink_to(predecessor,target_is_directory=True)
    campaign=output/CAMPAIGN;directory=campaign/'first16';directory.mkdir(parents=True)
    for name in ('profile.json','baseline-comparison.json'):
        if (old/name).is_file():shutil.copy2(old/name,directory/name)
    shutil.copytree(old/'protocol',directory/'protocol')
    freeze_recovery_setup(old/'setup',directory/'setup')
    # Deliberately retain original checkout and fresh CLI initialization.
    agent=(directory/'setup/container_claude_agent.sh').read_text()
    if '--resume' in agent or 'recovery-prompt' in agent or 'git checkout -B "$CLAUDE_BRANCH_NAME" "$BASE_COMMIT"' not in agent:
        raise ValueError('Fresh agent would use a previous session/workspace')
    if sha(directory/'setup/run_deepswe_cell.sh')!=sha(old/'setup/run_deepswe_cell.sh'):
        raise ValueError('Agent/verifier image selection changed')
    task=read(old/'manifest.json');task['tasks']=[t for t in task['tasks'] if t['task_id'] in TASKS]
    task['fresh_restart']=dict(user_authorized=True,source=str(old),reuse_history=False,reuse_workspace=False,added_prompt=None)
    task['setup_sha256']={p.name:sha(p) for p in (directory/'setup').iterdir() if p.is_file()}
    validate_task_subset(task,original_task);verify_images(task)
    save(directory/'manifest.json',task)
    (directory/'tasks.txt').write_text('\n'.join(t['task_id'] for t in task['tasks'])+'\n')
    controller=runtime/'scripts/benchmark_deepswe_first16.py';controller.write_text(patch_controller(controller.read_text()))
    reviewer=runtime/'scripts/queue_deepswe_fastpath.py';text=reviewer.read_text();start=text.index('def final_review(root):');end=text.index('\ndef run(root):',start)
    part=text[start:end].replace('len(expected)!=16','len(expected)!=5').replace('expected=16','expected=5').replace('/16,','/5,').replace('classified_tasks=16','classified_tasks=5')
    reviewer.write_text(text[:start]+part+text[end:])
    dashboard=runtime/'scripts/deepswe_seed_dashboard.py'
    dashboard.write_text(replace_once(dashboard.read_text(),"if __name__=='__main__':",
        "from scripts.deepswe_recovery_view import install_dashboard\nimport sys\ninstall_dashboard(sys.modules[__name__])\n\nif __name__=='__main__':"))
    save(campaign/'previous-vllm-outcomes.json',dict(source=str(source),superseded=str(predecessor),original_scored=59,original_passed=27,
        note='Five original infrastructure failures are replaced only by these explicitly authorized fresh attempts; prior outcomes remain in history.'))
    save(campaign/'status.json',dict(status='prepared',seed=1004,mode='fresh_restart'))
    model={str(p.relative_to(runtime)):sha(p) for name in ('src','csrc') for p in (runtime/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if any(sha(source/'runtime'/p)!=h for p,h in model.items()):raise ValueError('Model runtime changed')
    save(output/'restart-audit.json',dict(at=time.time(),source=str(source),predecessor=str(predecessor),predecessor_files=before,
        reuse_session=False,reuse_workspace=False,added_prompt=None,original_task_files=True,seed=1004,
        model_files_unchanged=True,only_tool_change='pytest auto workers honor existing2CPU quota',original59_not_reexecuted=True,
        prior_attempts_disposition='Preserved; excluded from selected fresh-attempt score and throughput. No best-of selection.'))
    files={str(p.relative_to(output)):sha(p) for base in (runtime,directory) for p in base.rglob('*') if p.is_file()}
    for name in ('approval.json','restart-audit.json','warmup-tokens.bin'):files[name]=sha(output/name)
    m=dict(original,source=str(source),files=files,model_files=model,source_manifest_sha256=sha(source/'manifest.json'),
        recovery=dict(mode='fresh_restart',campaign=CAMPAIGN,task_ids=list(TASKS),expected=5,seed=1004,prompt=None,
            original_scored=59,original_passed=27,original_interrupted=5),superseded_continuation=str(predecessor),
        scope='Explicitly requested fresh restart of five interrupted tasks, paired with the59 preserved scores; all prior attempts retained.')
    save(output/'manifest.json',m);save(output/'status.json',dict(status='prepared',active_seed=None,mode='fresh_restart',at=time.time()))
    validate_fresh(directory)
    print(json.dumps(dict(root=str(output),manifest_sha256=sha(output/'manifest.json'),tasks=list(TASKS))))


def run(root):
    if ROOT!=(root/'runtime').resolve():raise ValueError('Use frozen queue')
    m=validate_fresh(root/CAMPAIGN/'first16')
    for path,digest in read(root/'restart-audit.json')['predecessor_files'].items():
        if sha(path)!=digest:raise ValueError('Superseded evidence changed')
    preserved_predecessor(Path(m['superseded_continuation']))
    if (root/CAMPAIGN/'first16/cases').exists():raise ValueError('Fresh run must begin without any case files/history')
    return common.run(root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-from',type=Path);p.add_argument('--predecessor',type=Path);p.add_argument('--approval',type=Path);a=p.parse_args()
    if a.prepare_from:prepare(a.prepare_from.resolve(),a.root.resolve(),a.predecessor.resolve(),a.approval)
    else:
        try:run(a.root.resolve())
        except BaseException as e:
            save(a.root/'failure.json',dict(error=repr(e),at=time.time()))
            save(a.root/'status.json',dict(status='failed',active_seed=None,error=repr(e),mode='fresh_restart',at=time.time()))
            raise
