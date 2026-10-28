"""One fresh first16 attempt on the qualified hybrid runtime, after DSH A/B.

This is a finite GPU queue, not a recurring monitor. Keep the original task
images, client, sampling and hidden verifier; never resume or retry old answers.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import psutil

from scripts import run_agent_handoff_queue as q
from scripts import benchmark_deepswe_first16 as runner

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path('/home/chaidi/qw3/benchmark/action_pressure_first16_warmfix_20261010')
PREVIOUS = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/deepswe-first16-nvfp4-20261017-r4')
OVERLAYS = ('scripts/queue_deepswe_fastpath.py', 'scripts/benchmark_deepswe_first16.py',
            'benchmarks/deepswe/claude_bridge.py', 'tests/test_deepswe_fastpath_queue.py',
            'tests/test_deepswe_native_transport.py', 'tests/test_deepswe_window_lifecycle.py')


def profile_from(source):
    result = copy.deepcopy(source)
    result.update(port=18880, served_model_name='kvmem-deepswe', max_model_len=524288)
    # Provision names for 16 tasks plus synthetic qualification; not GPU slots
    # or extra historical bytes. Finished tasks explicitly release their owners.
    result['kvmem']['session_capacity'] = 32
    if not result['kvmem'].get('restore_handoff_fastpath'):
        raise ValueError('Qualified fastpath candidate required')
    if (result['kvmem']['select_budget'], result['kvmem']['gen_budget'],
            result['kvmem']['query_mean_quota_fraction'],result['kvmem']['graph_alpha']) != (36864,16384,.5,0):
        raise ValueError('Unexpected recall policy')
    return result


def task_files(manifest):
    paths = {Path(manifest['claude_binary']): manifest['claude_sha256']}
    for task in manifest['tasks']:
        instruction = Path(task['instruction'])
        paths[instruction] = task['instruction_sha256']
        metadata = instruction.with_name('task.toml')
        text = metadata.read_text()
        for key,expected in [('docker_image',task['agent_image']),('base_commit_hash',task['base_commit'])]:
            found = re.search(r'^\s*'+key+r'\s*=\s*"([^"]*)"\s*$',text,re.M)
            if not found or found[1] != expected:
                raise ValueError('Task repository/image changed: '+task['task_id'])
        paths[metadata] = q.sha(metadata)
    return {str(path):digest for path,digest in paths.items()}


def verify_images(manifest):
    images = {name:digest for task in manifest['tasks'] for name,digest in task['images'].items()}
    for name,digest in images.items():
        actual = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',name],text=True).strip()
        if actual != digest:
            raise ValueError('Frozen image changed: '+name)
    return len(images)


def validate(root):
    m = q.verify_frozen(root)
    source = Path(m['source'])
    if q.sha(source/'manifest.json') != m['source_manifest_sha256']:
        raise ValueError('Qualified source identity changed')
    if q.read(root/'first16/profile.json') != profile_from(q.read(source/'candidate/profile.json')):
        raise ValueError('Model/budgets/recall settings changed')
    for rel,digest in m['model_files'].items():
        if q.sha(root/'runtime'/rel) != digest:
            raise ValueError('Model differs from qualified code')
    for path,digest in m['external_files'].items():
        if q.sha(Path(path)) != digest:
            raise ValueError('Frozen task/client changed: '+path)
    for path,digest in m['qualification_proof'].items():
        if q.sha(Path(path)) != digest:
            raise ValueError('Qualification evidence changed')
    task = q.read(root/'first16/manifest.json')
    reference = q.read(REFERENCE/'manifest.json')
    if task['tasks'] != reference['tasks'] or task['sampling'] != reference['sampling']:
        raise ValueError('Tasks or sampling differ from original')
    if task['workers'] != 8 or task['duration_seconds'] != 0 or task['attempts_per_task'] != 1:
        raise ValueError('Not the full original single-attempt cohort')
    if not task['optimized_handoff'] or task['termination_policies'] != {
            'cached_read_streak':32,'identical_successful_bash_streak':32}:
        raise ValueError('Tool handoff or original stopping policy changed')
    return m


def failed_predecessor_proof(source, parent):
    """Record an explicitly acknowledged, fully stopped failed predecessor.

    It remains failed; this only permits the independent requested workload to
    acquire released resources. No predecessor answers or sessions are resumed.
    """
    if q.read(source/'status.json')['status'] != 'failed' or q.live(q.read(source/'queue-process.json')):
        raise ValueError('Failed predecessor must have exited')
    paths = [source/p for p in ('status.json','queue-process.json','queue.log')]
    for name in ('baseline','candidate'):
        directory = source/name
        paths.append(directory/'status.json')
        if (directory/'cleanup.json').exists():
            cleanup = q.read(directory/'cleanup.json')
            if any(cleanup[k] for k in ('remaining_processes','remaining_owned_gpu_pids')):
                raise ValueError('Failed predecessor cleanup incomplete')
            if any(q.live(owner) for owner in cleanup['owned']):
                raise ValueError('Failed predecessor process still alive')
            hashes = q.read(directory/'evidence-hashes.json')
            if not hashes or any(q.sha(directory/rel) != digest for rel,digest in hashes.items()):
                raise ValueError('Failed predecessor answer evidence changed')
            paths += [directory/p for p in ('cleanup.json','execution-audit.json','evidence-hashes.json','summary.json')]
            paths += list((directory/'cases').glob('*/failure.json'))
        elif q.read(directory/'status.json')['status'] != 'prepared' or (directory/'server-process.json').exists():
            raise ValueError('Unaudited predecessor arm')
    q.require_idle_dsh(parent)
    return dict(status='failed_preserved', files={str(p):q.sha(p) for p in paths},
        explanation='Prior DSH pair failed with client timeouts; its cleanup and all answer evidence were verified. '
                    'Candidate DSH arm remains unattempted. This independent DeepSWE run is not a recovery '
                    'or a passed DSH comparison; no old sessions are resumed.')


def unstarted_recovery_proof(old):
    q.verify_frozen(old)
    state=q.read(old/'first16/run.json')
    if state['status']!='needs_attention' or state['tasks']:
        raise ValueError('Only an unstarted formal cohort may use this recovery')
    work=q.read(old/'first16/server/capture/public_work.json')
    if any(work['completed'][k] or work.get('submitted',{}).get(k,0)
           for k in ('public_batches','public_prefill_tokens','public_non_prefill_tokens')):
        raise ValueError('Public inference already occurred; require a different audited recovery')
    cleanup=q.read(old/'cleanup.json')
    if (any(cleanup[k] for k in ('remaining_processes','remaining_owned_gpu_pids')) or
            any(q.live(owner) for owner in cleanup['owned']) or q.live(q.read(old/'queue-process.json'))):
        raise ValueError('Unstarted attempt is not cleaned up')
    return dict(root=str(old),scope='No public forward or formal answer occurred. Fix only synthetic preflight ownership; '
        'all task inputs, runtime model, profile, sampling and verifier remain unchanged.',
        files={str(p.relative_to(old)):q.sha(p) for p in old.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts})


def prepare(source, output, acknowledge_failed_predecessor=False, recover_unstarted=None):
    from scripts.benchmark_agent_fastpath_dsh import validate as validate_source
    parent = validate_source(source)
    recovery = unstarted_recovery_proof(recover_unstarted) if recover_unstarted else None
    failed_proof = failed_predecessor_proof(source,parent) if acknowledge_failed_predecessor else None
    if output.exists():
        raise ValueError('Use a new output directory; no overwrite/retry')
    output.mkdir(parents=True)
    runtime = output/'runtime'
    shutil.copytree(source/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in OVERLAYS:
        shutil.copy2(ROOT/rel,runtime/rel)
    directory = output/'first16'; directory.mkdir()
    profile = profile_from(q.read(source/'candidate/profile.json'))
    q.save(directory/'profile.json',profile)
    shutil.copytree(PREVIOUS/'kvmem-c8/protocol',directory/'protocol')
    args = SimpleNamespace(reference=REFERENCE,config=directory/'profile.json',output=directory,
        workers=8,bash_repeat_limit=32,carry_result=[],backend_kind='kvmem',duration_seconds=0,
        optimized_handoff=True)
    task = runner.prepare(args)
    task['differences'] = [
        'Current qualified NVFP4/FP8/MTP4 hybrid 50:50, K36864/G16384, graph0, attention pressure.',
        'Latest previous vLLM run used K65536/G32768 and a196608 logical limit; current limit524288 was explicitly chosen by the user.',
        'Resident KV, model cache, split restore, native restore frame and handoff fastpath enabled.',
        'Optimized bridge reuses exact identities; original text protocol, thinking4096 and sampling unchanged.',
        'Historical quality comparison is not a controlled optimization-only speedup.']
    q.save(directory/'manifest.json',task)
    original = q.read(REFERENCE/'manifest.json')
    external = task_files(original)
    external[str(REFERENCE/'manifest.json')] = q.sha(REFERENCE/'manifest.json')
    for path,digest in external.items():
        if q.sha(Path(path)) != digest:raise ValueError('Task/client hash mismatch')
    shutil.copy2(ROOT/'artifacts/deepswe-first16-20261015-r12/baseline-comparison.json',
                 directory/'baseline-comparison.json')
    previous_run = q.read(PREVIOUS/'kvmem-c8/run.json')
    q.save(output/'previous-vllm-outcomes.json',dict(
        path=str(PREVIOUS),run_sha256=q.sha(PREVIOUS/'kvmem-c8/run.json'),
        profile=q.read(PREVIOUS/'kvmem-c8/profile.json'),
        tasks={name:{'state':value['state'],'reward':(value.get('result',{}).get('reward') or {}).get('reward')}
               for name,value in previous_run['tasks'].items()}))
    dependency = q.read(source/'queue-process.json')
    proof_root = Path(parent['source'])
    m = dict(source=str(source),source_manifest_sha256=q.sha(source/'manifest.json'),
        dependency=dependency,failed_predecessor=failed_proof,
        python=parent['python'],idle_seconds=60,poll_seconds=15,
        model_startup_timeout_seconds=3600,bridge_port=18490,
        qualification_proof={str(proof_root/p):h for p,h in parent['component_proof'].items()},
        external_files=external,
        model_files={str(p.relative_to(runtime)):q.sha(p) for parent_dir in ('src','csrc')
                     for p in (runtime/parent_dir).rglob('*') if p.is_file()},
        protocol_sha256=q.sha(directory/'protocol/protocol.so'),
        scope='Fresh complete first16; original tasks/verifiers/sampling. 512K context explicitly requested. '
              'Wait for existing DSH pair and cleanup; no old task retry and no recurring monitor.',
        files={str(p.relative_to(output)):q.sha(p) for parent_dir in ('runtime','first16')
               for p in (output/parent_dir).rglob('*') if p.is_file()})
    m['files']['previous-vllm-outcomes.json']=q.sha(output/'previous-vllm-outcomes.json')
    if recovery:
        old=q.read(recover_unstarted/'manifest.json')
        if m['model_files']!=old['model_files'] or q.read(directory/'profile.json')!=q.read(recover_unstarted/'first16/profile.json'):
            raise ValueError('Recovery changed model or configuration')
        original_task=q.read(recover_unstarted/'first16/manifest.json')
        if any(task[k]!=original_task[k] for k in ('tasks','sampling','seed','attempts_per_task','termination_policies','optimized_handoff')):
            raise ValueError('Recovery changed workload')
        q.save(output/'preflight-recovery-audit.json',recovery)
        m['files']['preflight-recovery-audit.json']=q.sha(output/'preflight-recovery-audit.json')
    q.save(output/'manifest.json',m)
    q.save(output/'status.json',dict(status='prepared',at=time.time()))
    validate(output)


def predecessor_ready(status, alive):
    if alive:return False
    if status.get('status') != 'completed':
        raise ValueError('Predecessor stopped without an audited final outcome; preserve both experiments')
    return True


def wait_predecessor(root, manifest):
    source = Path(manifest['source'])
    failed_proof = manifest.get('failed_predecessor')
    if failed_proof is not None:
        if failed_proof['status'] != 'failed_preserved' or q.live(manifest['dependency']):
            raise ValueError('Acknowledged predecessor is not stopped')
        for path,digest in failed_proof['files'].items():
            if q.sha(Path(path)) != digest:
                raise ValueError('Failed predecessor evidence changed')
        q.require_idle_dsh(q.read(source/'manifest.json'))
        q.save(root/'dependency-failed-preserved.json',dict(**failed_proof,at=time.time()))
        return
    while True:
        if (root/'STOP').exists():raise InterruptedError('Queued test stopped by user')
        state = q.read(source/'status.json')
        alive = q.live(manifest['dependency']) is not None
        q.save(root/'status.json',dict(status='waiting_for_dsh_pair',dependency=str(source),
            dependency_status=state['status'],active=state.get('active'),at=time.time()))
        if predecessor_ready(state,alive):break
        time.sleep(manifest['poll_seconds'])
    from scripts.benchmark_agent_fastpath_dsh import outcome
    proof = {name:outcome(source/name) for name in ('baseline','candidate')}
    if not (source/'comparison.json').is_file():raise ValueError('Missing DSH comparison')
    q.save(root/'dependency-completed.json',dict(outcomes=proof,
        comparison_sha256=q.sha(source/'comparison.json'),at=time.time()))


def final_review(root):
    from scripts.analyze_deepswe import analyze
    directory=root/'first16';run=q.read(directory/'run.json')
    if run['status']!='completed' or run.get('cleanup_errors') or not run.get('cleanup_finished_at'):
        raise ValueError('DeepSWE did not finish cleanly; retain all attempted cases')
    expected=[t['task_id'] for t in q.read(directory/'manifest.json')['tasks']]
    if set(run['tasks'])!=set(expected) or len(expected)!=16:
        raise ValueError('Incomplete task cohort')
    if any(t['state'] not in ('completed','context_ceiling') for t in run['tasks'].values()):
        raise ValueError('Infrastructure/unclassified outcome')
    summary=analyze(directory)
    for item in summary['tasks']:
        item['hidden_test_result']=run['tasks'][item['task']].get('result',{}).get('reward')
    counts={'tool_calls':0,'tool_results':0,'tool_errors':0}
    for case in expected:
        calls={};results={}
        for line in (directory/'cases'/case/'claude.stream.jsonl').open():
            try:record=json.loads(line)
            except ValueError:continue
            if record.get('type') not in ('assistant','user'):continue
            content=record.get('message',{}).get('content',[])
            if not isinstance(content,list):continue
            for item in content:
                if not isinstance(item,dict):continue
                if item.get('type')=='tool_use':calls[item['id']]=item
                elif item.get('type')=='tool_result':results[item['tool_use_id']]=item
        counts['tool_calls']+=len(calls);counts['tool_results']+=len(results)
        counts['tool_errors']+=sum(bool(x.get('is_error')) for x in results.values())
    counts['tool_result_error_rate']=counts['tool_errors']/counts['tool_results'] if counts['tool_results'] else None
    evidence={str(p.relative_to(root)):q.sha(p) for p in (directory/'bridge').rglob('*.json')}
    for case in expected:
        d=directory/'cases'/case
        for pattern in ('*.json','*.jsonl','*.patch','verifier/*.json','verifier/*.log'):
            for p in d.glob(pattern):
                if p.is_file():evidence[str(p.relative_to(root))]=q.sha(p)
    q.save(root/'evidence-hashes.json',evidence)
    previous=q.read(root/'previous-vllm-outcomes.json')
    report=dict(status='completed',expected=16,scored=summary['completed_tasks'],
        solved=summary['passed_tasks'],solved_fraction_of_full_cohort=summary['passed_tasks']/16,
        context_ceiling=sum(t['state']=='context_ceiling' for t in run['tasks'].values()),
        infrastructure_errors=summary['infrastructure_errors'],tool_counts=counts,
        aggregate_agent_tokens_per_second=summary['mixed_end_to_end_output_tokens_per_second'],
        accepted_output_tokens=summary['output_tokens'],wall_seconds=summary['elapsed_seconds'],
        tasks=summary['tasks'],previous_vllm=previous,
        caveats=['Context-limited cases are unscored, not invented verifier zeroes.',
                 'Historical budgets, context limits, code and trajectories differ; no causal speedup claim.',
                 'Tool is_error results are distinct from hidden verifier failures and shell exit codes.',
                 'Output/wall includes tool execution and verification; MTP validation rows are excluded.'])
    q.save(root/'scores.json',report)
    q.save(root/'execution-audit.json',dict(status='passed',classified_tasks=16,
        scored=summary['completed_tasks'],context_ceiling=report['context_ceiling'],
        evidence_sha256=q.sha(root/'evidence-hashes.json')))
    return report


def run(root):
    if ROOT.resolve()!=(root/'runtime').resolve():raise ValueError('Use frozen controller')
    m=validate(root)
    if q.read(root/'status.json')['status']!='prepared':raise ValueError('No automatic restart/retry')
    q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()),at=time.time(),
        manifest_sha256=q.sha(root/'manifest.json')))
    owned=q.Owned()
    try:
        wait_predecessor(root,m)
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            idle_since=None
            while True:
                if (root/'STOP').exists():raise InterruptedError('Queued test stopped by user')
                pids=q.gpu_pids();idle_since=(idle_since or time.monotonic()) if not pids else None
                q.save(root/'status.json',dict(status='waiting_for_gpu',gpu_pids=pids,at=time.time()))
                if idle_since and time.monotonic()-idle_since>=m['idle_seconds']:break
                time.sleep(m['poll_seconds'])
            validate(root);verify_images(q.read(root/'first16/manifest.json'))
            for port in (18880,m['bridge_port']):q.require_free_port(port)
            if psutil.virtual_memory().available<90*2**30:
                raise ValueError('Insufficient free host memory before eight-container qualification')
            if q.gpu_pids():raise RuntimeError('GPU acquired by another owner during prelaunch checks')
            directory=root/'first16'
            env=q.environment(root/'runtime')
            env.update(MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            argv=[m['python'],'-B',str(root/'runtime/scripts/benchmark_deepswe_first16.py'),
                '--config',str(directory/'profile.json'),'--output',str(directory),
                '--workers','8','--backend-kind','kvmem','--duration-seconds','0',
                '--bridge-port',str(m['bridge_port']),'--bash-repeat-limit','32',
                '--use-prepared','--optimized-handoff','--startup-timeout-seconds','3600']
            process=owned.start(argv,root/'runtime',env,directory/'controller.log',directory/'controller-process.json')
            q.save(root/'status.json',dict(status='qualification_then_testing',at=time.time()))
            while process.poll() is None:
                owned.track()
                if set(q.gpu_pids())-set(owned.rows):raise RuntimeError('Foreign GPU owner appeared')
                time.sleep(5)
            if process.returncode:raise RuntimeError('DeepSWE controller failed; preserve attempts')
            report=final_review(root)
        cleanup=owned.stop();q.save(root/'cleanup.json',cleanup)
        if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
            raise ValueError('Owned GPU/process cleanup incomplete')
        q.save(root/'status.json',dict(status='completed',solved=report['solved'],
            scored=report['scored'],context_ceiling=report['context_ceiling'],at=time.time()))
    except BaseException as error:
        q.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        q.save(root/'cleanup.json',owned.stop())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acknowledge-failed-predecessor',action='store_true')
    p.add_argument('--recover-unstarted',type=Path)
    args=p.parse_args()
    if args.source:prepare(args.source.resolve(),args.output.resolve(),args.acknowledge_failed_predecessor,
                           args.recover_unstarted.resolve() if args.recover_unstarted else None)
    else:run(args.output.resolve())
