"""Finite, dependency-gated GPU qualification and two-arm DSH agent comparison.

No quality retries, no old-session resume, no foreign GPU cancellation. Every
arm owns a frozen runtime, fresh session IDs and independent project folders.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

import psutil


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):h.update(block)
    return h.hexdigest()


def identity(process):
    return dict(pid=process.pid, created=process.create_time())


def live(owner):
    try:
        p = psutil.Process(owner['pid'])
        return p if p.create_time() == owner['created'] and p.status() != psutil.STATUS_ZOMBIE else None
    except psutil.NoSuchProcess:
        return None


def gpu_pids():
    # A query failure is never evidence of an idle GPU.
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [int(row.strip()) for row in output.splitlines() if row.strip()]


def require_free_port(port):
    # Fail before launching a competing listener; never stop its current owner.
    # Match the HTTP server's reuse semantics: a closed previous arm can leave
    # TIME_WAIT connections even after its listener and all processes exit.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1',port))
        probe.listen(1)


def distribution(values):
    values=sorted(float(v) for v in values if v is not None)
    if not values:return dict(count=0)
    return dict(count=len(values),mean=statistics.mean(values),
        median=statistics.median(values),p95=values[min(len(values)-1,int(.95*len(values)))])


def handoff_metrics(directory, report):
    def rows(relative):
        paths=[directory/relative]
        history=directory/'recovery-history.json'
        if history.exists() and not relative.startswith('bridge/'):
            paths=[Path(read(history)['source'])/relative]+paths
        return [json.loads(line) for path in paths if path.exists() for line in path.read_text().splitlines() if line]
    owners={owner for case in report['cases'].values() for owner in case['backend']['agent_owners']}
    requests=[r for r in rows('bridge/requests.jsonl') if r.get('owner') in owners and 'error' not in r and 'metrics' in r]
    specs=[seg.get('metrics',{}).get('speculative_decoding',{}) for r in requests for seg in r.get('segment_reports',[])]
    steps=sum(s.get('num_spec_steps',0) for s in specs)
    drafts=sum(s.get('num_draft_tokens',0) for s in specs)
    accepted=sum(s.get('num_accepted_draft_tokens',0) for s in specs)
    restore=rows('server/capture/async_restore.jsonl')
    model_steps=rows('server/capture/model_step_latency.jsonl')
    resources=rows('resources.jsonl')
    peak_memory=max([float(r['gpu'].split(',')[0]) for r in resources if 'gpu' in r] or [0])
    cpu_available=[r['available_cpu_bytes'] for r in resources if 'available_cpu_bytes' in r]
    return dict(
        request_ttft_seconds=distribution(r.get('first_native_token_seconds') for r in requests),
        cold_request_ttft_seconds=distribution(r.get('first_native_token_seconds') for r in requests if not r.get('exact_output_prefix_reused')),
        warm_request_ttft_seconds=distribution(r.get('first_native_token_seconds') for r in requests if r.get('exact_output_prefix_reused')),
        mtp_spec_steps=steps,mtp_accepted_draft_tokens=accepted,mtp_draft_tokens=drafts,
        mtp_mean_acceptance_length=1+accepted/steps if steps else None,
        mtp_draft_acceptance_rate=accepted/drafts if drafts else None,
        restore_seconds=distribution(r['finished']-r['started'] for r in restore),
        restore_count=len(restore),resident_restores=sum(bool(r.get('resident_active_tokens')) for r in restore),
        model_cache_hits=sum(bool(r.get('model_cache_hit')) for r in restore),
        restored_kv_bytes=sum(r['uploaded_bytes'] for r in restore),
        restored_model_h2d_bytes=sum(r.get('model_uploaded_bytes',r['model_bytes']) for r in restore),
        maintenance_host_ms=distribution(r.get('maintenance_host_ms') for r in model_steps),
        model_step_ms=distribution(r.get('model_ms') for r in model_steps),
        common_eight_decode_step_ms=distribution(r.get('model_ms') for r in model_steps
            if r.get('decode_requests')==8 and not r.get('prefill_requests')),
        gpu_peak_mib=peak_memory,min_cpu_available_bytes=min(cpu_available) if cpu_available else None,
        caveat='Restore durations overlap across requests; their sum is not GPU blocking time. MTP validation rows are not accepted output tokens.')


def dependency_ready(plan, state, pids, controller_live):
    summary=state.get('summary',{})
    completed=summary.get('completed',summary.get('all',{}).get('count'))
    return (state.get('phase') == 'complete' and
        completed == plan['dependency_expected'] and
        not controller_live and not pids)


def verify_frozen(root):
    manifest = read(root/'manifest.json')
    for relative, expected in manifest['files'].items():
        if sha(root/relative) != expected:
            raise ValueError('Frozen test source/profile changed: '+relative)
    return manifest


def wait_dependency(root, plan):
    stable = None
    while True:
        if (root/'STOP').exists():raise InterruptedError('User stopped queued tests')
        try:
            state = read(plan['dependency_state'])
            pids = gpu_pids()
            alive = live(plan['dependency_controller']) is not None
            ready = dependency_ready(plan, state, pids, alive)
            detail = dict(dependency_phase=state.get('phase'),
                dependency_completed=state.get('completed',state.get('summary',{}).get('all',{}).get('count')),
                dependency_controller_alive=alive, gpu_pids=pids)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            ready = False; detail = dict(read_error=repr(error))
        if ready:
            stable = stable or time.monotonic()
        else:stable = None
        save(root/'status.json',dict(status='waiting_for_benchmark',at=time.time(),**detail))
        if stable is not None and time.monotonic()-stable >= plan['idle_seconds']:return
        time.sleep(plan['poll_seconds'])


class Owned:
    def __init__(self):self.rows = {}; self.processes = []

    def start(self, argv, cwd, env, log, record):
        with Path(log).open('xb') as output:
            p = subprocess.Popen(argv,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        owner = identity(psutil.Process(p.pid)); self.rows[p.pid] = owner; self.processes.append(p)
        save(record,dict(**owner,argv=argv,cwd=str(cwd),at=time.time()))
        return p

    def track(self):
        for row in list(self.rows.values()):
            p = live(row)
            if p is None:continue
            try:
                for child in p.children(recursive=True):self.rows[child.pid] = identity(child)
            except psutil.NoSuchProcess:pass

    def stop(self):
        self.track()
        for p in self.processes:
            owner = self.rows[p.pid]; current = live(owner)
            if current is not None:current.send_signal(signal.SIGTERM)
        deadline=time.monotonic()+40
        while time.monotonic()<deadline and any(live(x) for x in self.rows.values()):
            self.track();time.sleep(.5)
        for row in self.rows.values():
            p=live(row)
            if p is not None:p.send_signal(signal.SIGTERM)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and any(live(x) for x in self.rows.values()):time.sleep(.5)
        for row in self.rows.values():
            p=live(row)
            if p is not None:p.kill()
        for p in self.processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:pass
        return dict(remaining_processes=[r for r in self.rows.values() if live(r)],
                    remaining_owned_gpu_pids=[pid for pid in gpu_pids() if pid in self.rows],
                    owned=list(self.rows.values()),at=time.time())


def environment(runtime):
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join(map(str,(runtime/'src',runtime/'scripts',runtime/'tests',runtime)))
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    return env


def ready_http(url, process, owned, timeout=900, progress_path=None):
    import urllib.request
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started=time.monotonic();deadline=started+timeout;last_report=None
    while time.monotonic()<deadline:
        owned.track()
        if process.poll() is not None:raise RuntimeError('Owned service exited before readiness')
        if progress_path is not None and (last_report is None or time.monotonic()-last_report>=30):
            children=[]
            for owner in list(owned.rows.values()):
                child=live(owner)
                if child is None:continue
                try:
                    cpu=child.cpu_times()
                    children.append(dict(**owner,name=child.name(),cpu_seconds=cpu.user+cpu.system))
                except psutil.NoSuchProcess:pass
            save(progress_path,dict(status='starting',elapsed_seconds=time.monotonic()-started,
                timeout_seconds=timeout,processes=children,at=time.time()))
            last_report=time.monotonic()
        try:
            with opener.open(url,timeout=3) as response:
                if response.status==200:
                    if progress_path is not None:
                        save(progress_path,dict(status='ready',elapsed_seconds=time.monotonic()-started,at=time.time()))
                    return
        except OSError:pass
        time.sleep(2)
    raise TimeoutError('Owned service readiness timed out')


async def qualify_requests(profile, fixture, output):
    import httpx
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}',trust_env=False,timeout=900) as client:
        async def one(index, item):
            tokens=list(item['tokens']); results=[]
            owner=f'handoff-qualification-{index}'
            for turn in range(3):
                payload=dict(model=profile['served_model_name'],prompt=tokens,
                    session_id=owner,cache_salt=owner,max_tokens=64,temperature=0,seed=1000,
                    ignore_eos=True,return_token_ids=True,kvmem_reselect='off')
                before=time.time()
                response=await client.post('/v1/completions',json=payload);response.raise_for_status()
                body=response.json();ids=body['choices'][0].get('token_ids')
                if not ids or len(ids)!=body['usage']['completion_tokens']:
                    raise ValueError('Qualification lacks exact accepted output IDs')
                save(output/f'{index}-{turn}.json',dict(response=body,seconds=time.time()-before,
                    input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
                results.append(len(ids));tokens+=ids+item['suffix']
            return results
        return await asyncio.gather(*(one(i,x) for i,x in enumerate(fixture)))


def start_model(root, profile_path, runtime, python, owned):
    require_free_port(read(profile_path)['port'])
    root.mkdir(exist_ok=False)
    return owned.start([python,'-B','-m','vllm_kvmem','serve','--config',str(profile_path),
        '--run-dir',str(root/'server')],runtime,environment(runtime),root/'server.log',root/'server-process.json')


def qualification(root, plan):
    runtime=root/'candidate/runtime';env=environment(runtime)
    env['KVMEM_RUN_GPU_TESTS']='1'
    command=[plan['python'],'-m','pytest','-q','tests/test_model_snapshot_cache.py',
             'tests/test_session_state_gpu.py','tests/test_session_state.py']
    with (root/'gpu-component.log').open('xb') as log:
        subprocess.run(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    if gpu_pids():raise RuntimeError('GPU component process did not release its context')
    target=root/'qualification';owned=Owned()
    profile_path=root/'qualification-profile.json';profile=read(profile_path)
    try:
        model=start_model(target,profile_path,runtime,plan['python'],owned)
        ready_http(f'http://127.0.0.1:{profile["port"]}/health',model,owned,
            timeout=plan.get('model_startup_timeout_seconds',900),progress_path=target/'startup.json')
        results=asyncio.run(qualify_requests(profile,read(root/'qualification-fixture.json'),target))
        rows=[json.loads(s) for s in (target/'server/capture/async_restore.jsonl').read_text().splitlines()]
        if not any(r.get('model_cache_hit') for r in rows):raise RuntimeError('GPU snapshot cache was not exercised')
        if not any(r.get('resident_active_tokens',0)>0 for r in rows):raise RuntimeError('Active residency was not exercised')
        oracles=target/'server/capture/verified_restores.jsonl'
        if not oracles.exists() or not oracles.stat().st_size:raise RuntimeError('Missing live restore byte oracle')
        verified=[json.loads(s) for s in oracles.read_text().splitlines()]
        if len(verified)!=len(rows) or not all(r.get('byte_equal') for r in verified):
            raise RuntimeError('Incomplete live restore byte equality audit')
        save(target/'execution-audit.json',dict(status='passed',results=results,
            restores=len(rows),model_cache_hits=sum(bool(r.get('model_cache_hit')) for r in rows),
            resident_restores=sum(r.get('resident_active_tokens',0)>0 for r in rows)))
    finally:
        cleanup=owned.stop();save(target/'cleanup.json',cleanup)
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:raise RuntimeError('Qualification cleanup incomplete')


def retire_previous_bridge(plan):
    require_idle_dsh(plan)
    old=plan['previous_bridge'];p=live(old)
    if p is None:return
    if p.cmdline()!=old['argv']:raise RuntimeError('Old bridge identity changed')
    if read(plan['previous_test_status']).get('status')!='paused':
        raise RuntimeError('Original DSH test is no longer paused')
    p.terminate();p.wait(timeout=30)  # Exact previously paused bridge only.


def require_idle_dsh(plan):
    from scripts.run_dsh_daily_suite import rpc
    sessions=rpc(plan['dsh_url'],'session.list',{}).get('items')
    if not isinstance(sessions,list) or any(row.get('running') is not False for row in sessions):
        raise RuntimeError('DSH has an active or unverified session; do not replace its model bridge')


def optimized_bridge(plan, name):
    arms = plan.get('optimized_bridge_arms', ['candidate'])
    if not isinstance(arms, list) or len(set(arms)) != len(arms) or any(
            arm not in ('baseline', 'candidate') for arm in arms):
        raise ValueError('Invalid optimized bridge arm selection')
    return name in arms


def arm(root, name, plan):
    # Both arms use identical controller/metrics tooling; only their separate
    # model and bridge subprocesses load the respective frozen implementations.
    controller=root/'candidate/runtime'
    sys.path[:0]=[str(controller),str(controller/'scripts'),str(controller/'src')]
    from scripts.run_dsh_daily_suite import run_case, rpc, bridge_observations
    from scripts.summarize_dsh_daily_suite import summarize
    from scripts.graph_v2_protocol import release_owner
    directory=root/name;runtime=directory/'runtime';profile=read(directory/'profile.json')
    tasks=read(directory/'task-plan.json');owned=Owned();done=threading.Event();results={};foreign_gpu=set()
    recovering=name in plan.get('resume_arms',{})
    directory.joinpath('cases').mkdir(exist_ok=recovering);directory.joinpath('workspaces').mkdir(exist_ok=recovering)
    if (directory/'started.json').exists():raise RuntimeError('Attempted arm cannot be restarted')
    save(directory/'started.json',dict(at=time.time()))
    try:
        require_idle_dsh(plan)
        require_free_port(profile['port']);require_free_port(plan['bridge_port'])
        server=owned.start([plan['python'],'-B','-m','vllm_kvmem','serve','--config',str(directory/'profile.json'),
            '--run-dir',str(directory/'server')],runtime,environment(runtime),directory/'server.log',directory/'server-process.json')
        ready_http(f'http://127.0.0.1:{profile["port"]}/health',server,owned,
            timeout=plan.get('model_startup_timeout_seconds',900),progress_path=directory/'startup.json')
        argv=[plan['python'],'-B','-m','scripts.dsh_bridge','--protocol',plan['protocol'],
            '--tokenizer',profile['tokenizer'],'--backend',f'http://127.0.0.1:{profile["port"]}',
            '--model',profile['served_model_name'],'--output',str(directory/'bridge'),
            '--port',str(plan['bridge_port']),'--gen-budget','16384','--max-model-len','524288',
            '--max-sessions','64','--owner-namespace','daily','--max-output-segments','4','--require-query-provenance']
        if optimized_bridge(plan, name):argv+=['--optimized-handoff']
        bridge=owned.start(argv,runtime,environment(runtime),directory/'bridge.log',directory/'bridge-process.json')
        ready_http(f'http://127.0.0.1:{plan["bridge_port"]}/health',bridge,owned,timeout=120)
        def observe():
            with (directory/'resources.jsonl').open('x') as stream:
                while not done.wait(5):
                    owned.track()
                    try:
                        stats=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw',
                            '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                        foreign_gpu.update(pid for pid in gpu_pids() if pid not in owned.rows)
                        stream.write(json.dumps(dict(at=time.time(),gpu=stats,
                            foreign_gpu_pids=sorted(foreign_gpu),
                            available_cpu_bytes=psutil.virtual_memory().available))+'\n');stream.flush()
                    except Exception as error:stream.write(json.dumps(dict(error=repr(error)))+'\n')
        watcher=threading.Thread(target=observe);watcher.start()
        began=time.time()
        pending=[]
        for row in tasks['cases']:
            saved=directory/'cases'/row['id']/'status.json'
            if recovering and saved.exists() and read(saved).get('status')=='completed':
                results[row['id']]=read(saved)
            else:pending.append(row)
        barrier=threading.Barrier(len(pending))
        save(directory/'status.json',dict(status='testing',at=began,expected=8))
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs={pool.submit(run_case,directory,plan['dsh_url'],row,barrier,tasks):row for row in pending}
            for future in as_completed(jobs):
                row=jobs[future]
                try:
                    result=future.result();results[row['id']]=result
                    for owner in result['bridge']['owners']:
                        accepted=sorted((directory/'bridge'/owner).glob('*/accepted.json'))[-1]
                        token=read(accepted)['token_ids'][-1]
                        release_owner(profile,directory/'server/capture',owner,token)
                except Exception as error:
                    results[row['id']]=dict(status='failed',error=repr(error))
                    save(directory/'cases'/row['id']/'failure.json',results[row['id']])
                    try:rpc(plan['dsh_url'],'session.cancel',dict(sessionId=row['session_id']))
                    except Exception as cancellation_error:
                        save(directory/'cases'/row['id']/'cancel-error.json',dict(error=repr(cancellation_error)))
                save(directory/'summary.json',dict(results=results,completed=len(results),expected=8,at=time.time()))
        ended=time.time();report=summarize(directory)
        done.set();watcher.join(20)
        report.update(expected_tasks=len(tasks['cases']),minimum_human_turns=tasks['minimum_turns']*len(tasks['cases']))
        save(directory/'observations.json',report)
        errors=[]
        if foreign_gpu:errors.append('Concurrent foreign GPU workload; throughput comparison invalid')
        if report['counts']['completed_tasks']!=8 or report['counts']['tasks_over_36k']!=8:errors.append('Incomplete/context cohort')
        if any(not x['minimum_followups_met'] for x in report['cases'].values()):errors.append('Missing followups')
        if report['unexpected_policy']:errors.append('Recall policy mismatch')
        if name=='candidate':
            restores=[json.loads(s) for s in (directory/'server/capture/async_restore.jsonl').read_text().splitlines()]
            if not any(r.get('model_cache_hit') for r in restores):errors.append('Snapshot cache unused')
            if not any(r.get('resident_active_tokens',0) for r in restores):errors.append('Active residency unused')
        browser_python=plan.get('browser_python',plan['python'])
        subprocess.run([browser_python,str(controller/'scripts/check_dsh_daily_browser.py'),
            '--root',str(directory)],check=True,timeout=600)
        evidence={str(p.relative_to(directory)):sha(p) for parent in ('bridge','cases')
            for p in (directory/parent).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')}
        save(directory/'evidence-hashes.json',evidence)
        save(directory/'execution-audit.json',dict(status='passed' if not errors else 'failed',errors=errors,counts=report['counts']))
        accepted=sum(bridge_observations(directory,row['session_id'])['accepted_output_tokens']
                     for row in tasks['cases'])
        original=plan.get('resume_arms',{}).get(name,{})
        wall=ended-original.get('original_started',began)
        save(directory/'performance.json',dict(started=original.get('original_started',began),finished=ended,wall_seconds=wall,
            recovery_execution_seconds=ended-began if recovering else None,
            interrupted=recovering,uninterrupted_eight_task_comparison=not recovering,
            accepted_output_tokens=accepted,aggregate_agent_tokens_per_second=accepted/wall,
            counts=report['counts'],handoff=handoff_metrics(directory,report),
            scope='Eight natural agent tasks including tools and followups; generations may differ.'))
        if errors:raise RuntimeError(str(errors))
        save(directory/'status.json',dict(status='completed',completed=8,at=time.time()))
    finally:
        done.set()
        if 'watcher' in locals():watcher.join(20)
        # An unexpected orchestration failure must not leave this arm's DSH
        # agents running against a model about to be released. Never cancel
        # pre-existing sessions or sessions owned by other experiments.
        for row in tasks['cases']:
            state=directory/'cases'/row['id']/'status.json'
            if state.exists() and read(state).get('status')!='completed':
                try:rpc(plan['dsh_url'],'session.cancel',dict(sessionId=row['session_id']))
                except Exception as error:
                    save(state.parent/'cleanup-cancel-error.json',dict(error=repr(error)))
        cleanup=owned.stop();save(directory/'cleanup.json',cleanup)
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:raise RuntimeError('Arm cleanup incomplete')


def main(root):
    plan=verify_frozen(root)
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'gpu-started.json').exists():raise RuntimeError('No blind restart of attempted GPU tests')
        save(root/'queue-process.json',dict(**identity(psutil.Process()),at=time.time(),manifest_sha256=sha(root/'manifest.json')))
        try:
            subprocess.run([plan.get('browser_python',plan['python']),
                str(root/'candidate/runtime/scripts/check_dsh_daily_browser.py'),
                '--root',str(root/'baseline'),'--probe'],check=True,timeout=60)
            wait_dependency(root,plan);verify_frozen(root)
            if gpu_pids():raise RuntimeError('GPU ownership changed before qualification')
            save(root/'gpu-started.json',dict(at=time.time()))
            save(root/'status.json',dict(status='qualification',at=time.time()))
            if plan.get('reuse_qualification'):
                proof=plan['reuse_qualification'];source=Path(proof['root'])
                if any(sha(source/p)!=value for p,value in proof['hashes'].items()):
                    raise ValueError('Qualification reuse proof changed')
                if read(root/'qualification/execution-audit.json').get('status')!='passed':
                    raise ValueError('Reused qualification is not passed')
            else:qualification(root,plan)
            retire_previous_bridge(plan)
            for name in ('baseline','candidate'):
                if gpu_pids():raise RuntimeError('Foreign GPU process appeared between stages')
                verify_frozen(root)
                save(root/'status.json',dict(status='running',active=name,at=time.time()))
                arm(root,name,plan)
            save(root/'comparison.json',{name:read(root/name/'performance.json') for name in ('baseline','candidate')})
            save(root/'status.json',dict(status='completed',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root.resolve())
