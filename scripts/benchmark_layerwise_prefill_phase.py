"""Finite ABBA of isolated eight-owner historical prefill, preserving responses.

Both arms delay final Query until all eight original histories have completed.
This is a phased diagnostic, not continuous-serving or accuracy acceptance.
No profiler, external scoring, resampling, automatic retry or recurring monitor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
OLD = REPO/'artifacts/g4k-pressure-hotpath-20261104-r1'
PRIOR = REPO/'artifacts/g4k-pressure-implementation-20261104-r1'
BASE = REPO/'artifacts/g4k-gilopt-prefill-diagnostic-20261104-r1/cohort-1'
FULL = REPO/'artifacts/g4k-gilopt-20261104-r1'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p): return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*'))
                     if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'}
def save(p, value, *, new=False):
    with p.open('x' if new else 'w') as f:
        json.dump(value, f, indent=2);f.write('\n')
def verify_files(files):
    for p, digest in files.items():
        if sha(Path(p)) != digest: raise ValueError('Frozen source changed: '+p)


def prepare(root, baseline_bundle=None, candidate_bundle=None, semantic_qualification=None,
            candidate_cpu_affinity=None):
    if root.exists(): raise ValueError('Use a new immutable experiment directory')
    qualification=semantic_qualification or PRIOR/'post-forward-reap-serial/serial-equivalence.json'
    if not read(qualification)['passed']:
        raise ValueError('Candidate requires passed complete-input qualification')
    if candidate_cpu_affinity is not None:
        candidate_cpu_affinity=sorted(set(candidate_cpu_affinity))
        if not candidate_cpu_affinity or not set(candidate_cpu_affinity)<=os.sched_getaffinity(0):
            raise ValueError('CPU affinity must be a nonempty subset of allowed CPUs')
        if shutil.which('taskset') is None:raise ValueError('Missing process-local affinity launcher')
    root.mkdir(parents=True)
    shutil.copy2(__file__, root/'launch.py')
    shutil.copy2(REPO/'src/vllm_kvmem/prefill_phase_gate.py', root/'prefill_phase_gate.py')
    bundles = {'baseline': baseline_bundle or OLD/'inline-prefill-candidate.json',
               'candidate': candidate_bundle or PRIOR/'post-forward-reap-candidate.json'}
    files = {str(root/n):sha(root/n) for n in ('launch.py','prefill_phase_gate.py')}
    files[str(qualification)]=sha(qualification)
    for path in bundles.values():
        b=read(path);assert sha(Path(b['patch']))==b['patch_sha256']
        files.update({str(path):sha(path),b['patch']:b['patch_sha256']})
    files.update({str(FULL/n):sha(FULL/n) for n in ('fixture.json','profile.json')})
    base_tree=tree(BASE/'runtime-v1')
    save(root/'base-runtime.json',base_tree,new=True)
    files[str(root/'base-runtime.json')]=sha(root/'base-runtime.json')
    save(root/'plan.json',dict(order=[['01-A','baseline'],['02-B','candidate'],
        ['03-B','candidate'],['04-A','baseline']],bundles={k:str(v) for k,v in bundles.items()},
        ids=read(FULL/'fixture.json')['case_ids'][:8],workers=8,files=files,
        scope=__doc__,semantic_qualification=str(qualification),
        candidate_cpu_affinity=candidate_cpu_affinity,
        external_grading=False,scheduled_monitor=False,automatic_retry=False),new=True)
    save(root/'prepared-proof.json',{'plan_sha256':sha(root/'plan.json')},new=True)
    save(root/'status.json',dict(status='prepared'),new=True)


def verify(root):
    if sha(root/'plan.json')!=read(root/'prepared-proof.json')['plan_sha256']:
        raise ValueError('Frozen experiment plan changed')
    p=read(root/'plan.json');verify_files(p['files']);return p


def prepare_arm(queue, name, kind):
    plan=verify(queue);root=queue/name
    if root.exists():raise ValueError('Never replace an attempted arm')
    root.mkdir();runtime=root/'runtime-v1'
    shutil.copytree(BASE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assert tree(runtime)==read(queue/'base-runtime.json')
    b=read(Path(plan['bundles'][kind]))
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_CEILING_DIRECTORIES']=str(runtime.parent)
    for check in (True,False):
        subprocess.run(['git','apply',*(['--check'] if check else []),b['patch']],
                       cwd=runtime,env=env,check=True,capture_output=True)
    assert all(sha(runtime/n)==v for n,v in b['expected'].items())
    assert {n for n,v in tree(runtime).items() if read(queue/'base-runtime.json').get(n)!=v}==set(b['expected'])
    shutil.copy2(queue/'prefill_phase_gate.py',runtime/'src/vllm_kvmem/prefill_phase_gate.py')
    with (runtime/'src/vllm_kvmem/layerwise_query.py').open('a') as f:
        f.write("\n# Identical finite measurement gate in both arms.\n"
                "os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '8'\n"
                "from .prefill_phase_gate import install as _install_measurement_gate\n"
                "_install_measurement_gate()\n")
    fixture=read(FULL/'fixture.json');ids=plan['ids']
    fixture.update(rows=[r for r in fixture['rows'] if r['id'] in ids],
        bindings=[b for b in fixture['bindings'] if b['id'] in ids],case_ids=ids,cases=8,
        performance_observation=False)
    assert [r['id'] for r in fixture['rows']]==ids
    save(root/'fixture.json',fixture,new=True);shutil.copy2(FULL/'profile.json',root/'profile.json')
    profile=read(root/'profile.json')
    assert profile['max_num_seqs']==8 and profile['max_num_batched_tokens']==16424
    assert [profile['kvmem'][k] for k in ['select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks']]==[16384,4096,32,41,0]
    save(root/'runtime-proof.json',tree(runtime),new=True)
    pointer=REPO/'artifacts/value-recall-active.json'
    save(root/'plan.json',dict(queue=str(queue),kind=kind,workers=8,
        cpu_affinity=plan.get('candidate_cpu_affinity') if kind=='candidate' else None,
        files={str(root/n):sha(root/n) for n in ['fixture.json','profile.json','runtime-proof.json']},
        pointer=str(pointer),previous_pointer_sha256=sha(pointer)),new=True)
    save(root/'prepared-proof.json',{'plan_sha256':sha(root/'plan.json')},new=True)
    save(root/'status.json',dict(status='prepared'),new=True)


def run_arm(root):
    plan=verify(root);runtime=root/'runtime-v1'
    if plan.get('cpu_affinity') is not None:
        assert os.sched_getaffinity(0)==set(plan['cpu_affinity'])
    assert tree(runtime)==read(root/'runtime-proof.json')
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts')]
    from run_recall_comparison import gpu_lock
    import benchmark_block_graph as lifecycle
    with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        assert not lifecycle.gpu_processes(), 'GPU has another owner'
        pointer=Path(plan['pointer']);assert sha(pointer)==plan['previous_pointer_sha256']
        save(pointer,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),
            process=str(root/'process.json'),experiment_type='isolated_prefill_phase',at=time.time()))
    import run_layerwise_memory as memory
    def strict(path):
        assert path==root
        verify(root);assert tree(runtime)==read(root/'runtime-proof.json')
        return read(root/'fixture.json'),read(root/'profile.json')
    memory.verify=strict
    memory.run(root,workers=8)


def analyze_arm(root):
    def lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
    cap=root/'run/server/capture';gate=lines(cap/'prefill_phase_gate.jsonl')
    release,=[r for r in gate if r['event']=='history_prefill_released']
    arrivals=[r for r in gate if r['event']=='history_prefill_arrived']
    assert len(arrivals)==8 and len(release['owners'])==8
    clocks=lines(cap/'public_work.jsonl');offsets=[r['wall_time']-r['at'] for r in clocks]
    assert max(offsets)-min(offsets)<.01
    clock=min(clocks,key=lambda r:abs(r['at']-release['at']))
    start=min(read(p)['at'] for p in (root/'run/answers/cases').glob('*/attempt.json'))
    end=release['at']+clock['wall_time']-clock['at']
    from benchmark_client_io import result_events
    paths=sorted((root/'run/answers/cases').glob('*/result.json'))
    records=[read(p) for p in paths]
    record_events=[result_events(r,p.parent) for r,p in zip(records,paths)]
    assert len(records)==8 and all(r['original_input_intact'] and r['status']=='completed' for r in records)
    query=[]
    for p in (root/'run/answers/cases').glob('*/events.json'):
        query.extend(e for e in read(p) if e['selection']=='query')
    assert len(query)==8 and sum(e['history_begin'] for e in query)==release['history_tokens']
    stages=lines(cap/'async_refresh.jsonl')
    query_keys={(e['request_id'],e['command_id']) for e in query}
    assert not any((e['request_id'],e['command_id']) in query_keys and
        e['stage']!='suspended' and e['at']<release['at'] for e in stages)
    steps=lines(cap/'model_step_latency.jsonl')
    assert not any(e['decode_requests'] for e in steps if e['started_at']<release['at'])
    perf=read(root/'inference-performance.json')
    pressure=[e for es in record_events for e in es if e.get('event_type')=='prefill_pressure']
    assert all(e.get('fallback_reason') is None for e in pressure)
    report=dict(history_tokens=release['history_tokens'],seconds=end-start,
        tokens_per_second=release['history_tokens']/(end-start),owners=8,
        no_query_or_decode_before_release=True,prefill_pressure=len(pressure),
        post_forward_reaped=sum(e['stage']=='pressure_completion_reaped' for e in stages),
        accepted_output_tokens=sum(r['usage']['completion_tokens'] for r in records),
        full_inference_seconds=perf['inference_wall_seconds'],
        scope='Eight-owner phased historical prefill, client dispatch through last history completion, '
              'including pressure/assembly/HTTP waits, excluding final Query and all generation. '
              'This gate changes service scheduling and is diagnostic-only; not continuous-serving TPS.')
    save(root/'phase-measurement.json',report,new=True);return report


def run(root):
    plan=verify(root)
    if (root/'process.json').exists():raise ValueError('No automatic rerun')
    sys.path.insert(0,str(BASE/'runtime-v1/scripts'))
    from run_recall_comparison import stop_owned,termination_guard
    import benchmark_block_graph as lifecycle
    save(root/'process.json',dict(pid=os.getpid(),at=time.time()),new=True)
    reports=[]
    expected_owner=read(REPO/'artifacts/value-recall-active.json')['queue_root']
    with termination_guard():
        try:
            for name,kind in plan['order']:
                verify(root);assert not lifecycle.gpu_processes()
                assert read(REPO/'artifacts/value-recall-active.json')['queue_root']==expected_owner
                prepare_arm(root,name,kind);arm=root/name
                save(root/'status.json',dict(status='running',arm=name,at=time.time()))
                command=[sys.executable,'-B',str(root/'launch.py'),'arm','--root',str(arm)]
                if kind=='candidate' and plan.get('candidate_cpu_affinity') is not None:
                    command=['taskset','-c',','.join(map(str,plan['candidate_cpu_affinity'])),*command]
                env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
                env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'child.json',dict(pid=child.pid,command=command,at=time.time()))
                try:code=child.wait(timeout=1800)
                except BaseException:stop_owned(child);raise
                if code:raise RuntimeError('Preserve failed technical attempt: '+name)
                assert not lifecycle.gpu_processes()
                report=analyze_arm(arm);reports.append(dict(name=name,kind=kind,**report))
                expected_owner=str(arm)
                print(json.dumps(reports[-1]),flush=True)
            rates={kind:sum(r['history_tokens'] for r in reports if r['kind']==kind)/
                   sum(r['seconds'] for r in reports if r['kind']==kind) for kind in ('baseline','candidate')}
            save(root/'comparison.json',dict(reports=reports,pooled_rates=rates,
                observed_change=rates['candidate']/rates['baseline']-1,
                serial_semantic_qualification=plan['semantic_qualification'],
                eight_owner_exact_equivalence=False,continuous_serving_acceptance=False,
                scope=__doc__),new=True)
            save(root/'status.json',dict(status='completed_measurement',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','arm']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-bundle',type=Path);p.add_argument('--candidate-bundle',type=Path)
    p.add_argument('--semantic-qualification',type=Path)
    p.add_argument('--candidate-cpu-affinity',type=int,nargs='+')
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.root.resolve(),*(v.resolve() if v else None for v in
                (a.baseline_bundle,a.candidate_bundle,a.semantic_qualification)),
                candidate_cpu_affinity=a.candidate_cpu_affinity)
    else:{'run':run,'arm':run_arm}[a.action](a.root.resolve())
