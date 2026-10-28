"""Frozen finite Terminal-Bench 2.1 x Terminus 2, 89 tasks x one attempt."""
import argparse
import array
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid

import httpx
import psutil
import run_agent_handoff_queue as q
try:
    from .terminalbench_policy import configure_task_deadlines
    from .terminalbench_network import direct_environment, DIRECT_CONTAINER_ENV, assert_direct_docker
    from .terminalbench_network import copy_download_cache, cache_compose_override, CACHE_ENV
except ImportError:
    from terminalbench_policy import configure_task_deadlines
    from terminalbench_network import direct_environment, DIRECT_CONTAINER_ENV, assert_direct_docker
    from terminalbench_network import copy_download_cache, cache_compose_override, CACHE_ENV

REPO = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm')
PYTHON = '/home/chaidi/kvmem-vllm/.venv/bin/python'
HARBOR_PYTHON = '/tmp/kvmem-terminus-20261026-env/bin/python'


def hashes(root):
    return {str(p.relative_to(root)): q.sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def reusable_image(row, proof, detail):
    return (proof.get('task') == row['name'] and
            proof.get('image') == row['environment'].get('docker_image') and
            proof.get('id') == detail.get('Id'))


def prepare(root, dataset, reuse_images_from=None, agent_time_limit='none', download_cache=None):
    if agent_time_limit not in ('none', 'official'):
        raise ValueError('Unknown Agent time limit policy')
    if root.exists():raise ValueError('New root required')
    source = REPO/'artifacts/rolling-long-output-20261026-r1'
    manifest = q.read(source/'manifest.json')
    if hashes(source/'runtime') != manifest['runtime']:raise ValueError('Qualified runtime changed')
    if not q.read(source/'final-review.json')['passed']:raise ValueError('Missing rolling correctness proof')
    root.mkdir(parents=True)
    if download_cache is None:
        download_cache=REPO/'artifacts/terminalbench21-network-repair-20261026-r1/verified-download-cache'
    copy_download_cache(download_cache,root/'download-cache')
    q.save(root/'download-cache-compose.json',cache_compose_override(root/'download-cache'))
    runtime=root/'runtime'
    shutil.copytree(source/'runtime',runtime)
    for folder in ('scripts','benchmarks','benchmarks/deepswe'):
        (runtime/folder).mkdir(exist_ok=True)
        (runtime/folder/'__init__.py').touch()
    for name in ('dsh_bridge.py','dsh_query_epoch.py','run_terminalbench_job.py',
                 'run_terminalbench_suite.py','run_agent_handoff_queue.py','terminalbench_view.py',
                 'finalize_terminalbench.py','terminalbench_policy.py','terminalbench_network.py'):
        shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
    shutil.copy2(REPO/'benchmarks/deepswe/claude_bridge.py',runtime/'benchmarks/deepswe/claude_bridge.py')
    shutil.copy2(REPO/'artifacts/deepswe-first16-20261015-r16/protocol/protocol.so',root/'protocol.so')
    shutil.copy2(source/'input.bin',root/'warmup-tokens.bin')
    shutil.copytree(dataset/'tasks',root/'dataset/tasks')
    timing = configure_task_deadlines(root/'dataset/tasks', agent_time_limit)
    q.save(root/'agent-time-policy.json',timing)
    rows=[]
    for task in sorted((root/'dataset/tasks').iterdir()):
        if not (task/'task.toml').exists():continue
        config=tomllib.loads((task/'task.toml').read_text())
        rows.append(dict(name=task.name,environment=config['environment'],agent=config['agent'],verifier=config['verifier']))
    if len(rows)!=89:raise ValueError('Expected official 89 tasks')
    q.save(root/'tasks.json',rows)
    profile=q.read(source/'profile.json')
    profile.update(port=18880,max_num_seqs=8,verify_session_restore=False)
    profile['served_model_name']='kvmem-qwen38-27b-96k'
    q.save(root/'profile.json',profile)
    q.save(root/'connection.json',dict(base='http://127.0.0.1:18388',control_token=uuid.uuid4().hex))
    os.chmod(root/'connection.json',0o600)
    common=dict(n_attempts=1,n_concurrent_trials=8,quiet=True,timeout_multiplier=1.0,
        retry=dict(max_retries=0),environment=dict(type='docker',delete=False,
            env={**DIRECT_CONTAINER_ENV,**CACHE_ENV},
            extra_docker_compose=[str(root/'download-cache-compose.json')],
            kwargs=dict(keep_containers=True),
            cpu_enforcement_policy='limit',memory_enforcement_policy='limit'),
        agents=[dict(name='terminus-2',model_name='openai/'+profile['served_model_name'],kwargs=dict(
            api_base='http://127.0.0.1:18388/v1',temperature=.6,
            enable_summarize=False,interleaved_thinking=True,session_id_headers=['X-Session-ID'],
            model_info=dict(max_input_tokens=10000000,max_output_tokens=131072,
                input_cost_per_token=0,output_cost_per_token=0,litellm_provider='openai',mode='chat'),
            llm_kwargs=dict(api_key='local-kvmem',timeout=14400,num_retries=0),
            llm_call_kwargs=dict(max_tokens=131072,top_p=.95,seed=1000,
                extra_body=dict(top_k=20,min_p=0,repetition_penalty=1,presence_penalty=0))))])
    formal=dict(**common,job_name='formal',jobs_dir=str(root/'jobs'),datasets=[dict(path=str(root/'dataset/tasks'))])
    q.save(root/'job.json',formal)
    smoke=root/'smoke-task';(smoke/'environment').mkdir(parents=True);(smoke/'tests').mkdir()
    (smoke/'instruction.md').write_text('Write the exact text KVMEM_TERMINUS_OK followed by a newline to /tmp/kvmem-terminus-ok.txt, verify its content, then finish.\n')
    (smoke/'environment/Dockerfile').write_text('FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y bash tmux asciinema python3 && rm -rf /var/lib/apt/lists/*\n')
    (smoke/'tests/test.sh').write_text('#!/bin/bash\nmkdir -p /logs/verifier\nif [ "$(cat /tmp/kvmem-terminus-ok.txt)" = "KVMEM_TERMINUS_OK" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n')
    (smoke/'task.toml').write_text('version = "1.0"\n[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 30\n[environment]\nbuild_timeout_sec = 600\ncpus = 1\nmemory_mb = 2048\nstorage_mb = 10240\n')
    smoke_config=dict(**common,job_name='smoke',jobs_dir=str(root/'jobs'),tasks=[dict(path=str(smoke))])
    q.save(root/'smoke-job.json',smoke_config)
    versions=json.loads(subprocess.check_output([HARBOR_PYTHON,'-c',
        'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))']))
    q.save(root/'dependencies.json',versions)
    dataset_commit=subprocess.check_output(['git','-C',str(dataset),'rev-parse','HEAD'],text=True).strip()
    harbor_commit=subprocess.check_output(['git','-C','/tmp/qw3-harbor-inspect','rev-parse','HEAD'],text=True).strip()
    # Archive the exact upstream source independently of its temporary checkout.
    with (root/'harbor-source.tar').open('xb') as out:
        subprocess.run(['git','-C','/tmp/qw3-harbor-inspect','archive',harbor_commit],stdout=out,check=True)
    frozen={name:hashes(root/name) for name in ('runtime','dataset','smoke-task','download-cache')}
    fixed=['profile.json','tasks.json','job.json','smoke-job.json','dependencies.json','protocol.so','warmup-tokens.bin','harbor-source.tar','agent-time-policy.json','download-cache-compose.json']
    if reuse_images_from:
        prior_path=reuse_images_from/'pre-inference-handoff.json'
        if not prior_path.exists():prior_path=reuse_images_from/'proxy-switch-audit.json'
        prior=q.read(prior_path)
        if not prior['no_model_or_task_started'] or (reuse_images_from/'model/server-process.json').exists():
            raise ValueError('Image preparation reuse requires an unlaunched model/task run')
        proofs={p.stem:dict(proof=q.read(p),sha256=q.sha(p))
                for p in sorted((reuse_images_from/'images').glob('*.json'))}
        for name,relative in prior.get('additional_image_proofs',{}).items():
            p=(reuse_images_from/relative).resolve()
            if not p.is_relative_to(reuse_images_from):raise ValueError('Image proof outside prior root')
            value=q.read(p)
            if value['task']!=name:raise ValueError('Image proof task mismatch')
            proofs[name]=dict(proof=value,sha256=q.sha(p))
        q.save(root/'reused-images.json',dict(source=str(reuse_images_from),
            source_manifest_sha256=q.sha(reuse_images_from/'manifest.json'),
            handoff_audit=str(prior_path),switch_audit_sha256=q.sha(prior_path),images=proofs))
        fixed.append('reused-images.json')
    q.save(root/'manifest.json',dict(created=time.time(),dataset='Terminal-Bench 2.1',dataset_commit=dataset_commit,
        harbor_commit=harbor_commit,harbor_version=versions['harbor'],tasks=89,attempts=1,
        max_concurrency=8,container_memory_admission_bytes=24*1024**3,output_cap=131072,
        agent_time_limit=agent_time_limit,official_timing_comparable=(agent_time_limit=='official'),
        thinking_budget=4096,seed=1000,qualification=str(source/'final-review.json'),
        frozen=frozen,files={f:q.sha(root/f) for f in fixed},
        notes=['No task retries or selection; official task resources/verifiers unchanged',
               'Agent execution has no wall-clock limit; custom local evaluation' if agent_time_limit=='none' else 'Official Agent deadlines retained',
               'Verifier/setup/tool timeouts retained; smoke Agent deadline remains 300 seconds',
               'No automatic summarization; retain reasoning across turns for exact token history',
               'Stock Terminus transport retries remain enabled; no completed task is regenerated',
               'Task containers retained stopped, logs and accepted token journals preserved',
               'Proxy-free host downloads and container exec; completed trials reclaim only owned empty networks',
               'Official uv/Python release archives are SHA256-verified and mounted read-only; verifier source unchanged',
               'Synthetic smoke is excluded from all 89 task scores and throughput']))
    q.save(root/'status.json',dict(status='prepared',at=time.time(),expected=89))


def resources(root,owned):
    owned.track()
    row=dict(at=time.time(),memory_available=psutil.virtual_memory().available,
        disk_free=shutil.disk_usage(root).free,system_free=shutil.disk_usage('/').free,
        processes=[dict(pid=p.pid,rss=p.memory_info().rss) for r in owned.rows.values()
                   if (p:=q.live(r)) is not None],gpu_pids=q.gpu_pids())
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=10,check=True)
    row['gpu_memory_mib']=sum(int(line.split(',')[1]) for line in gpu.stdout.splitlines()
        if line.strip() and int(line.split(',')[0]) in owned.rows)
    with (root/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return row


def fetch_images(root):
    rows=q.read(root/'tasks.json');directory=root/'images';directory.mkdir()
    env=direct_environment()
    reused=q.read(root/'reused-images.json')['images'] if (root/'reused-images.json').exists() else {}
    q.save(root/'download-environment.json',assert_direct_docker())
    def pull(row):
        image=row['environment'].get('docker_image')
        if not image:return dict(task=row['name'],build_from_source=True)
        old=reused.get(row['name'])
        if old:
            check=subprocess.run(['docker','image','inspect',image],env=env,capture_output=True,text=True)
            if check.returncode==0 and reusable_image(row,old['proof'],json.loads(check.stdout)[0]):
                result=dict(old['proof'],reused_verified_local_image=True,original_proof_sha256=old['sha256'])
                q.save(directory/(row['name']+'.json'),result)
                return result
        if shutil.disk_usage('/data').free<60*1024**3:raise RuntimeError('Insufficient Docker disk space')
        with (directory/(row['name']+'.log')).open('xb') as out:
            subprocess.run(['docker','pull',image],env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=1800)
        detail=json.loads(subprocess.check_output(['docker','image','inspect',image],env=env))[0]
        result=dict(task=row['name'],image=image,id=detail['Id'],digests=detail.get('RepoDigests'),size=detail['Size'])
        q.save(directory/(row['name']+'.json'),result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(pull,row):row['name'] for row in rows};completed=[]
        for future in concurrent.futures.as_completed(pending):
            completed.append(future.result())
            q.save(root/'status.json',dict(status='preparing_images',at=time.time(),ready=len(completed),expected=89))
    q.save(root/'image-audit.json',dict(passed=True,images=completed))


def run(root):
    plan=q.read(root/'manifest.json')
    for name,expected in plan['frozen'].items():
        if hashes(root/name)!=expected:raise ValueError('Frozen source changed: '+name)
    for name,expected in plan['files'].items():
        if q.sha(root/name)!=expected:raise ValueError('Frozen file changed: '+name)
    with (root/'started').open('x') as f:f.write(str(time.time()))
    q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()),at=time.time()))
    owned=q.Owned();runtime=root/'runtime';profile=q.read(root/'profile.json');conn=q.read(root/'connection.json')
    lock=open('/tmp/kvmem-coordination-qualification.lock','a')
    try:
        fetch_images(root)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle=None
        while True:
            gpu=q.gpu_pids()
            idle=None if gpu else (idle or time.monotonic())
            q.save(root/'status.json',dict(status='waiting_for_gpu',at=time.time(),gpu=gpu))
            if idle and time.monotonic()-idle>=60:break
            time.sleep(5)
        q.save(root/'status.json',dict(status='starting_model',at=time.time()))
        server=q.start_model(root/'model',root/'profile.json',runtime,PYTHON,owned)
        q.ready_http(f'http://127.0.0.1:{profile["port"]}/health',server,owned,timeout=3600,
                     progress_path=root/'model/startup.json')
        tokens=array.array('i');tokens.frombytes((root/'warmup-tokens.bin').read_bytes())
        with httpx.Client(trust_env=False,timeout=3600) as client:
            payload=dict(model=profile['served_model_name'],prompt=tokens.tolist(),max_tokens=1,
                temperature=0,seed=1000,session_id=root.name+'-warmup',cache_salt=root.name+'-warmup',
                kvmem_reselect='off',return_token_ids=True)
            result=client.post(f'http://127.0.0.1:{profile["port"]}/v1/completions',json=payload)
            result.raise_for_status();q.save(root/'warmup.json',result.json())
            payload.update(prompt=[151645],kv_transfer_params=dict(kvmem_release_session=True))
            result=client.post(f'http://127.0.0.1:{profile["port"]}/v1/completions',json=payload)
            result.raise_for_status();q.save(root/'warmup-release.json',result.json())
        q.require_free_port(18388)
        bridge=owned.start([PYTHON,'-B','-m','scripts.dsh_bridge','--protocol',str(root/'protocol.so'),
            '--tokenizer',profile['tokenizer'],'--backend',f'http://127.0.0.1:{profile["port"]}',
            '--model',profile['served_model_name'],'--output',str(root/'bridge'),'--port','18388',
            '--owner-namespace','tb21','--max-model-len','10000000','--gen-budget','16384',
            '--max-output-tokens','131072','--max-sessions','16','--optimized-handoff',
            '--abort-on-disconnect','--progress-interval','2','--release-control-token',conn['control_token']],
            runtime,direct_environment(q.environment(runtime)),root/'bridge.log',root/'bridge-process.json')
        q.ready_http(conn['base']+'/health',bridge,owned,timeout=120)
        for phase,config in [('smoke','smoke-job.json'),('formal','job.json')]:
            q.save(root/'status.json',dict(status=phase,at=time.time(),expected=89))
            env=direct_environment(q.environment(runtime));env.update(HARBOR_TELEMETRY='0',OPENAI_API_KEY='local-kvmem',
                LITELLM_LOCAL_MODEL_COST_MAP='True')
            job=owned.start([HARBOR_PYTHON,'-B',str(runtime/'scripts/run_terminalbench_job.py'),str(root),str(root/config)],
                root,env,root/(phase+'.log'),root/(phase+'-process.json'))
            while job.poll() is None:
                row=resources(root,owned)
                if row['memory_available']<8*1024**3 or row['disk_free']<30*1024**3:
                    job.send_signal(signal.SIGINT)
                    try:job.wait(timeout=120)
                    except subprocess.TimeoutExpired:pass
                    raise RuntimeError('Host resource preservation guard; all attempt files retained')
                if server.poll() is not None or bridge.poll() is not None:
                    job.send_signal(signal.SIGINT)
                    try:job.wait(timeout=120)
                    except subprocess.TimeoutExpired:pass
                    raise RuntimeError('Owned model/bridge exited')
                time.sleep(5)
            if job.returncode:raise RuntimeError(f'{phase} exited {job.returncode}; preserve logs, no retry')
            if phase=='smoke':
                q.save(root/'formal-boundary.json',dict(at=time.time(),smoke_excluded=True,
                    resources=resources(root,owned)))
        q.save(root/'status.json',dict(status='needs_final_review',at=time.time(),expected=89))
    except BaseException as error:
        q.save(root/'failure.json',dict(at=time.time(),error=repr(error)))
        q.save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error)))
        raise
    finally:
        q.save(root/'cleanup.json',owned.stop())
        lock.close()
    # Finite, local completion stage; it does not schedule any future runs.
    from finalize_terminalbench import review
    try:review(root)
    except Exception as error:
        q.save(root/'review-failure.json',dict(at=time.time(),error=repr(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('root',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--reuse-images-from',type=Path)
    p.add_argument('--download-cache',type=Path,help='Verified public uv/Python archive cache')
    p.add_argument('--agent-time-limit',choices=['none','official'],default='none',
                   help='Agent execution deadline policy for newly prepared runs; default: none')
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root,a.dataset.resolve(),a.reuse_images_from.resolve() if a.reuse_images_from else None,a.agent_time_limit,a.download_cache)
    else:run(root)
