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
try:
    from . import run_agent_handoff_queue as q
except ImportError:
    import run_agent_handoff_queue as q
try:
    from .terminalbench_policy import configure_task_deadlines
    from .terminalbench_network import direct_environment, DIRECT_CONTAINER_ENV, assert_direct_docker
    from .terminalbench_network import copy_download_cache, cache_compose_override, CACHE_ENV
    from .run_terminalbench_job import DEFAULT_CONTAINER_MEMORY_BYTES, HOST_MEMORY_RESERVE_BYTES, validate_memory_budget
except ImportError:
    from terminalbench_policy import configure_task_deadlines
    from terminalbench_network import direct_environment, DIRECT_CONTAINER_ENV, assert_direct_docker
    from terminalbench_network import copy_download_cache, cache_compose_override, CACHE_ENV
    from run_terminalbench_job import DEFAULT_CONTAINER_MEMORY_BYTES, HOST_MEMORY_RESERVE_BYTES, validate_memory_budget

REPO = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm')
PYTHON = sys.executable
HARBOR_PYTHON = '/tmp/kvmem-terminus-20261026-env/bin/python'
BRIDGE_MAX_SESSIONS = 16


def concurrency_limits(profile, task_count, trial_concurrency=None):
    """Task slots include tool waits; model slots only bound native execution."""
    model = profile.get('max_num_seqs', 8)
    if type(model) is not int or model < 1:
        raise ValueError('Terminal-Bench requires a positive integer model concurrency')
    if type(task_count) is not int or task_count < 1:
        raise ValueError('Terminal-Bench requires at least one selected task')
    if trial_concurrency is not None and (type(trial_concurrency) is not int or trial_concurrency < 1):
        raise ValueError('Terminal-Bench trial concurrency must be a positive integer')
    # Preserve legacy defaults for profiles with fewer than eight native slots.
    requested = min(8, model) if trial_concurrency is None else trial_concurrency
    tasks = min(requested, task_count)
    if tasks > BRIDGE_MAX_SESSIONS:
        raise ValueError('Terminal-Bench trial concurrency exceeds bridge owner capacity')
    sessions = profile.get('kvmem', {}).get('session_capacity', 0)
    if type(sessions) is not int or sessions < tasks:
        raise ValueError('Terminal-Bench session_capacity must cover every concurrent task owner; '
                         'tool-waiting sessions retain their history')
    return dict(task_concurrency=tasks, model_concurrency=model,
                session_capacity=sessions, bridge_max_sessions=BRIDGE_MAX_SESSIONS)


class RemoteModel:
    """The model, its GC mechanism checks and GPU telemetry run on another host over SSH.

    Harbor, task containers and the bridge stay local; the model port is tunnelled.
    `spec` (manifest remote_model): ssh options and host, the remote base directory
    holding serve.sh, the run label, thinking age steps, ports and remote paths.
    Remote files that local code reads are copied back under root/model.
    """
    CAPTURE = ('archive.json', 'progress.json', 'scheduler_resources.json')

    def __init__(self, root, spec, owned, runtime):
        self.root, self.spec, self.owned, self.runtime = root, spec, owned, runtime
        self.run_dir = f"{spec['base']}/runs/{spec['label']}"
        self.pid = self.forward = None
        self.processes = []

    def argv(self, command=None, *options):
        return ['ssh', *self.spec['ssh_options'], *options, self.spec['host'], *([command] if command else [])]

    def ssh(self, command, **kwargs):
        return subprocess.run(self.argv(command), capture_output=True, text=True, **kwargs)

    def read(self, path):
        result = self.ssh(f'cat {path}', timeout=60)
        return json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else None

    def start(self):
        busy = self.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader', timeout=60, check=True).stdout.split()
        if busy:
            raise RuntimeError(f'Remote GPUs are busy: {busy}')
        spec = self.spec
        self.ssh(f"cd {spec['base']} && (setsid nohup ./serve.sh {spec['label']} {spec['age_steps']} "
                 f"> logs/{spec['label']}.log 2>&1 < /dev/null &)", timeout=60, check=True)
        deadline = time.monotonic() + spec.get('startup_seconds', 3600)
        while True:
            status = self.read(f'{self.run_dir}/status.json') or {}
            if status.get('stage') == 'serving':
                break
            if status.get('stage') in ('failed', 'server_exited', 'stopped') or time.monotonic() > deadline:
                raise RuntimeError(f'Remote model did not start: {status}')
            time.sleep(10)
        self.pid = int(self.ssh(f"cat {spec['base']}/logs/{spec['label']}.pid", timeout=60, check=True).stdout)
        model = self.root/'model'
        model.mkdir(exist_ok=False)
        self.local = q.read(self.root/'profile.json')['port']
        # Through the shared SSH master a -N client hands the forward to the
        # master and exits, so the forward is owned by the master itself.
        self.forward = f"{self.local}:127.0.0.1:{spec['remote_port']}"
        subprocess.run(self.argv(None, '-O', 'cancel', '-L', self.forward), capture_output=True, timeout=60)
        subprocess.run(self.argv(None, '-O', 'forward', '-L', self.forward), capture_output=True, timeout=60, check=True)
        # -tt: the remote loop gets SIGHUP when its local ssh client stops.
        self.processes = [self.owned.start(self.argv(f'while kill -0 {self.pid} 2>/dev/null; do sleep 10; done', '-tt'),
            self.root, dict(os.environ), model/'watch.log', model/'watch.json')]
        return self

    def poll(self):
        """None while the remote harness is alive and the local forward accepts connections."""
        import socket
        codes = [p.poll() for p in self.processes]
        code = next((c for c in codes if c is not None), None)
        if code is not None:
            return code
        try:
            with socket.create_connection(('127.0.0.1', self.local), timeout=5):
                return None
        except OSError:
            return 255

    def monitor(self):
        """The dedicated 1-second GPU telemetry, sampled on the serving host."""
        spec = self.spec
        return self.owned.start(self.argv(f"cd {spec['runtime']} && exec {spec['python']} -B "
            f"scripts/gpu_efficiency_monitor.py sample --output {self.run_dir}/gpu-utilization.jsonl "
            f"--owned-root-pid {self.pid} --interval 1", '-tt'), self.root, dict(os.environ),
            self.root/'gpu-monitor.log', self.root/'gpu-monitor-process.json')

    def check(self, script, output, *arguments):
        """Run a mechanism check on the serving host; its evidence is copied back."""
        spec = self.spec
        return self.owned.start(self.argv(f"cd {spec['runtime']} && PYTHONPATH=src:scripts:. exec {spec['python']} -B "
            f"scripts/{script} --base http://127.0.0.1:{spec['remote_port']} --profile {self.run_dir}/profile.json "
            f"--protocol {spec['protocol']} --capture {self.run_dir}/server/capture "
            f"--output {self.run_dir}/{output} " + ' '.join(arguments)), self.root, dict(os.environ),
            self.root/f'{output}.log', self.root/f'{output}-process.json')

    def copy(self, paths, target):
        """Copy remote paths (relative to the run directory) under target."""
        target.mkdir(parents=True, exist_ok=True)
        with subprocess.Popen(self.argv(f"cd {self.run_dir} && tar -cf - --ignore-failed-read {' '.join(paths)}"),
                              stdout=subprocess.PIPE) as source:
            subprocess.run(['tar', '-xf', '-', '-C', str(target)], stdin=source.stdout, check=True, timeout=1800)
            source.wait(timeout=60)

    def sync_capture(self):
        self.copy([f'server/capture/{name}' for name in self.CAPTURE], self.root/'model')

    def stop(self):
        """Stop the remote harness (it stops its server group) and copy back its run."""
        if self.pid is None:
            return dict(remote='not_started')
        self.ssh(f'kill -TERM {self.pid}', timeout=60)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and self.ssh(f'kill -0 {self.pid}', timeout=60).returncode == 0:
            time.sleep(5)
        self.ssh(f"pkill -f '[g]pu_efficiency_monitor.py sample --output {self.run_dir}/'", timeout=60)
        self.ssh(f'cd {self.run_dir} && gzip -kf server.log', timeout=600)
        self.copy(['server/capture/*.json', 'server/launch.json', 'profile.json', 'launch.json', 'status.json',
                   'dry-run.stdout', 'server.log.gz'], self.root/'model')
        self.copy(['gpu-utilization.jsonl'], self.root)
        if self.forward:
            subprocess.run(self.argv(None, '-O', 'cancel', '-L', self.forward), capture_output=True, timeout=60)
        return dict(remote='stopped', harness_pid=self.pid,
                    alive=self.ssh(f'kill -0 {self.pid}', timeout=60).returncode == 0)


def hashes(root):
    return {str(p.relative_to(root)): q.sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def reusable_image(row, proof, detail):
    return (proof.get('task') == row['name'] and
            proof.get('image') == row['environment'].get('docker_image') and
            proof.get('id') == detail.get('Id'))


def freeze_runtime(repo, runtime):
    """Freeze the current checkout and built native module, never a prior run."""
    source = repo/'src/vllm_kvmem'
    native = list(source.glob('_native*.so'))
    if not native:
        raise ValueError('Build the current native extension before freezing Terminal-Bench')
    native_inputs = [p for folder in ('csrc', 'third_party/qw3') for p in (repo/folder).rglob('*')
                     if p.is_file() and p.suffix in ('.cpp', '.hpp', '.h') and
                     'tests' not in p.relative_to(repo/folder).parts]
    if (repo/'setup.py').exists():
        native_inputs.append(repo/'setup.py')
    if native_inputs and min(p.stat().st_mtime_ns for p in native) < max(p.stat().st_mtime_ns for p in native_inputs):
        raise ValueError('Native extension predates current sources; rebuild before freezing Terminal-Bench')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.git')
    for folder in ('src/vllm_kvmem', 'csrc', 'third_party/qw3'):
        shutil.copytree(repo/folder, runtime/folder, ignore=ignore)
    for folder in ('scripts', 'benchmarks', 'benchmarks/deepswe'):
        (runtime/folder).mkdir(parents=True, exist_ok=True)
        (runtime/folder/'__init__.py').touch()
    scripts = ('dsh_bridge.py', 'dsh_query_epoch.py', 'run_terminalbench_job.py',
               'run_terminalbench_suite.py', 'run_agent_handoff_queue.py', 'terminalbench_view.py',
               'finalize_terminalbench.py', 'terminalbench_policy.py', 'terminalbench_deadline.py', 'terminalbench_network.py',
               'terminalbench_harness.py', 'terminalbench_pty.py', 'terminalbench_tool_bundle.py',
               'terminalbench_verifier_cache.py',
               'terminalbench_query.py', 'terminalbench_replay.py', 'terminalbench_gpu_probe.py',
               'qualify_thinking_gc_preflight.py', 'gpu_efficiency_monitor.py',
               'terminalbench_proxy_relay.py')
    for name in scripts:
        shutil.copy2(repo/'scripts'/name, runtime/'scripts'/name)
    shutil.copy2(repo/'benchmarks/deepswe/claude_bridge.py', runtime/'benchmarks/deepswe/claude_bridge.py')
    copied = hashes(runtime)
    for name, value in copied.items():
        original = repo/name
        if original.exists() and q.sha(original) != value:
            raise ValueError('Checkout changed while freezing: '+name)
    return dict(origin='current-working-tree', source=str(repo),
                native={str(p.relative_to(repo)): q.sha(p) for p in native},
                runtime_sha256=hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest())


def task_query_manifest(root, harbor_python=HARBOR_PYTHON):
    upstream = json.loads(subprocess.check_output([harbor_python, '-c',
        'import hashlib,inspect,json; from pathlib import Path; '
        'from harbor.agents.terminus_2.terminus_2 import Terminus2; '
        'from harbor.llms.chat import Chat; '
        'from harbor.agents.terminus_2.tmux_session import TmuxSession; '
        'from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser; '
        'print(json.dumps({Path(inspect.getfile(c)).name:hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() '
        'for c in (Terminus2,Chat,TmuxSession,TerminusJSONPlainParser)}))'], text=True))
    tasks = {str(p.parent.relative_to(root)): dict(task_id=p.parent.name, instruction_sha256=q.sha(p))
             for p in [*sorted((root/'dataset/tasks').glob('*/instruction.md')), root/'smoke-task/instruction.md']}
    query = dict(policy='verified-task-query-v1', tasks=tasks,
                 upstream_sources={k: upstream[k] for k in ('terminus_2.py', 'chat.py')})
    harness = dict(policy='completion-control-network-v1', recorder_transport=True,
                   upstream_sources={k: upstream[k] for k in ('terminus_2.py', 'tmux_session.py', 'terminus_json_plain_parser.py')})
    return query, harness


def prepare(root, dataset, reuse_images_from=None, agent_time_limit='relaxed', download_cache=None,
            profile_path=None, task_names=None, query_mode='task', trial_concurrency=None,
            container_memory_admission_bytes=DEFAULT_CONTAINER_MEMORY_BYTES):
    validate_memory_budget(container_memory_admission_bytes,total=psutil.virtual_memory().total)
    if agent_time_limit not in ('none', 'official', 'relaxed', 'fixed'):
        raise ValueError('Unknown Agent time limit policy')
    if root.exists():raise ValueError('New root required')
    source = REPO/'artifacts/rolling-long-output-20261026-r1'
    if query_mode not in ('task', 'legacy'):
        raise ValueError('Unknown Terminal Query mode')
    root.mkdir(parents=True)
    if download_cache is None:
        download_cache=REPO/'artifacts/terminalbench21-network-repair-20261026-r1/verified-download-cache'
    copy_download_cache(download_cache,root/'download-cache')
    q.save(root/'download-cache-compose.json',cache_compose_override(root/'download-cache'))
    runtime=root/'runtime'
    runtime_proof = freeze_runtime(REPO, runtime)
    q.save(root/'runtime-proof.json', runtime_proof)
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
    names = {row['name'] for row in rows}
    selected = sorted(names if task_names is None else task_names)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(names):
        raise ValueError('Terminal-Bench subset must contain distinct official task names')
    q.save(root/'tasks.json',rows)
    profile=q.read(profile_path or source/'profile.json')
    profile.setdefault('port',18880)
    profile.setdefault('max_num_seqs',8)
    profile.setdefault('served_model_name','kvmem-qwen38-27b')
    limits=concurrency_limits(profile,len(selected),trial_concurrency)
    concurrency=limits['task_concurrency']
    # Warm only a small fixed prefix, compatible with explicit smaller K/G
    # profiles. It supplies no task answer and is excluded from measurements.
    warm=array.array('i');warm.frombytes((root/'warmup-tokens.bin').read_bytes())
    warm_limit=min(4096,max(32,profile['kvmem']['select_budget']//2))
    (root/'warmup-tokens.bin').write_bytes(warm[:warm_limit].tobytes())
    q.save(root/'profile.json',profile)
    q.save(root/'connection.json',dict(base='http://127.0.0.1:18388',control_token=uuid.uuid4().hex))
    os.chmod(root/'connection.json',0o600)
    common=dict(n_attempts=1,n_concurrent_trials=concurrency,quiet=True,timeout_multiplier=1.0,
        retry=dict(max_retries=0),environment=dict(type='docker',delete=False,
            env={**DIRECT_CONTAINER_ENV,**CACHE_ENV},
            extra_docker_compose=[str(root/'download-cache-compose.json')],
            kwargs=dict(keep_containers=True),
            cpu_enforcement_policy='limit',memory_enforcement_policy='limit'),
        agents=[dict(name='terminus-2',model_name='openai/'+profile['served_model_name'],kwargs=dict(
            api_base='http://127.0.0.1:18388/v1',temperature=.6,
            enable_summarize=False,interleaved_thinking=True,session_id_headers=['X-Session-ID'],
            model_info=dict(max_input_tokens=profile['max_model_len'],max_output_tokens=131072,
                input_cost_per_token=0,output_cost_per_token=0,litellm_provider='openai',mode='chat'),
            llm_kwargs=dict(api_key='local-kvmem',timeout=14400,num_retries=0),
            llm_call_kwargs=dict(max_tokens=131072,top_p=.95,seed=1000,
                extra_body=dict(top_k=20,min_p=0,repetition_penalty=1,presence_penalty=0))))])
    formal=dict(**common,job_name='formal',jobs_dir=str(root/'jobs'),
                tasks=[dict(path=str(root/'dataset/tasks'/name)) for name in selected])
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
    query, harness = task_query_manifest(root)
    historical = profile['kvmem'].get('historical_gc') or {}
    if historical.get('policy_params', {}).get('thinking_only'):
        try:
            from .qualify_thinking_gc_preflight import preflight_plan
        except ImportError:
            from qualify_thinking_gc_preflight import preflight_plan
        preflight = preflight_plan(profile)
    else:
        preflight = None
    frozen={name:hashes(root/name) for name in ('runtime','dataset','smoke-task','download-cache')}
    fixed=['runtime-proof.json','profile.json','tasks.json','job.json','smoke-job.json','dependencies.json','protocol.so','warmup-tokens.bin','harbor-source.tar','agent-time-policy.json','download-cache-compose.json']
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
        launched_task_names=selected,subset=len(selected)!=89,query_mode=query_mode,
        query_provenance=query if query_mode=='task' else None,harness_repair=harness,
        thinking_gc_preflight=preflight,
        max_concurrency=concurrency,**limits,
        container_memory_admission_bytes=container_memory_admission_bytes,
        container_memory_host_reserve_bytes=HOST_MEMORY_RESERVE_BYTES,output_cap=131072,
        agent_time_limit=agent_time_limit,official_timing_comparable=(agent_time_limit=='official'),
        thinking_budget=4096,seed=1000,runtime_proof=runtime_proof,
        frozen=frozen,files={f:q.sha(root/f) for f in fixed},
        notes=['Explicit task set, one fresh attempt per task; official task resources/verifiers unchanged',
               'Current checkout and built native module frozen; no qualification inherited from an old runtime',
               'Verified task Query metadata and bounded duplex recorder; complete messages unchanged',
               {'none': 'Agent execution has no wall-clock limit; custom local evaluation',
                'relaxed': 'Agent deadlines are 2x official; recorded pauses excluded; custom local evaluation',
                'fixed': 'Every task has a 1-hour Agent wall-clock deadline; custom local evaluation',
                'official': 'Official Agent deadlines retained'}[agent_time_limit],
               'Verifier/setup/tool timeouts retained; smoke Agent deadline remains 300 seconds',
               'No automatic summarization; retain reasoning across turns for exact token history',
               'Task concurrency includes tool waits and verification; native model concurrency is independent',
               'Stock Terminus transport retries remain enabled; no completed task is regenerated',
               'Task containers retained stopped, logs and accepted token journals preserved',
               'Proxy-free host downloads and container exec; completed trials reclaim only owned empty networks',
               'Official uv/Python release archives are SHA256-verified and mounted read-only; verifier source unchanged',
               'Synthetic smoke is excluded from all 89 task scores and throughput']))
    q.save(root/'status.json',dict(status='prepared',at=time.time(),expected=len(selected)))


def resources(root,owned):
    try:
        from .gpu_efficiency_monitor import snapshot
    except ImportError:
        from gpu_efficiency_monitor import snapshot
    owned.track()
    row=dict(at=time.time(),memory_available=psutil.virtual_memory().available,
        disk_free=shutil.disk_usage(root).free,system_free=shutil.disk_usage('/').free,
        processes=[dict(pid=p.pid,rss=p.memory_info().rss) for r in owned.rows.values()
                   if (p:=q.live(r)) is not None],gpu_pids=q.gpu_pids())
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=10,check=True)
    row['gpu_memory_mib']=sum(int(line.split(',')[1]) for line in gpu.stdout.splitlines()
        if line.strip() and int(line.split(',')[0]) in owned.rows)
    # This coarse suite sample complements the dedicated 1-second monitor.
    # Keep device utilization separate from owned-process VRAM attribution.
    row['gpu_telemetry']=snapshot(owned.rows)
    with (root/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return row


def fetch_images(root):
    selected=set(q.read(root/'manifest.json').get('launched_task_names',[]))
    rows=[row for row in q.read(root/'tasks.json') if not selected or row['name'] in selected]
    directory=root/'images';directory.mkdir()
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
            q.save(root/'status.json',dict(status='preparing_images',at=time.time(),ready=len(completed),expected=len(rows)))
    q.save(root/'image-audit.json',dict(passed=True,images=completed))


def run(root):
    plan=q.read(root/'manifest.json')
    expected_tasks=len(plan.get('launched_task_names',range(89)))
    for name,expected in plan['frozen'].items():
        if hashes(root/name)!=expected:raise ValueError('Frozen source changed: '+name)
    for name,expected in plan['files'].items():
        if q.sha(root/name)!=expected:raise ValueError('Frozen file changed: '+name)
    with (root/'started').open('x') as f:f.write(str(time.time()))
    q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()),at=time.time()))
    owned=q.Owned();runtime=root/'runtime';profile=q.read(root/'profile.json');conn=q.read(root/'connection.json')
    lock=open('/tmp/kvmem-coordination-qualification.lock','a')
    remote=RemoteModel(root,plan['remote_model'],owned,runtime) if plan.get('remote_model') else None
    try:
        fetch_images(root)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle=None
        while remote is None:
            gpu=q.gpu_pids()
            idle=None if gpu else (idle or time.monotonic())
            q.save(root/'status.json',dict(status='waiting_for_gpu',at=time.time(),gpu=gpu))
            if idle and time.monotonic()-idle>=60:break
            time.sleep(5)
        q.save(root/'status.json',dict(status='starting_model',at=time.time()))
        if remote is not None:
            server=remote.start()
            gpu_monitor=remote.monitor()
        else:
            server=q.start_model(root/'model',root/'profile.json',runtime,PYTHON,owned)
            gpu_monitor=owned.start([PYTHON,'-B',str(runtime/'scripts/gpu_efficiency_monitor.py'),
                'sample','--output',str(root/'gpu-utilization.jsonl'),
                '--owned-root-pid',str(server.pid),'--interval','1'],
                runtime,q.environment(runtime),root/'gpu-monitor.log',root/'gpu-monitor-process.json')
        q.ready_http(f'http://127.0.0.1:{profile["port"]}/health',server,owned,timeout=3600,
                     progress_path=None if remote else root/'model/startup.json')
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
        if plan.get('thinking_gc_preflight'):
            q.save(root/'status.json',dict(status='thinking_gc_preflight',at=time.time(),expected=expected_tasks))
            if remote is not None:
                check=remote.check('qualify_thinking_gc_preflight.py','thinking-gc-preflight',
                    '--target-tokens',str(plan['thinking_gc_preflight']['target_tokens']))
            else:
                check=owned.start([PYTHON,'-B',str(runtime/'scripts/qualify_thinking_gc_preflight.py'),
                    '--base',f'http://127.0.0.1:{profile["port"]}','--profile',str(root/'profile.json'),
                    '--protocol',str(root/'protocol.so'),'--capture',str(root/'model/server/capture'),
                    '--output',str(root/'thinking-gc-preflight'),
                    '--target-tokens',str(plan['thinking_gc_preflight']['target_tokens'])],
                    runtime,direct_environment(q.environment(runtime)),root/'thinking-gc-preflight.log',
                    root/'thinking-gc-preflight-process.json')
            while check.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError('Owned model exited during thinking GC preflight')
                time.sleep(1)
            if remote is not None:
                remote.copy(['thinking-gc-preflight'],root)
            if check.returncode:
                raise RuntimeError('Thinking GC preflight failed; official tasks were not started')
        if remote is not None and remote.spec.get('age_steps'):
            # The thinking age rule must retire old thinking on the real model before any task.
            q.save(root/'status.json',dict(status='thinking_age_check',at=time.time(),expected=expected_tasks))
            check=remote.check('qualify_thinking_age_gc.py','thinking-age-check')
            while check.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError('Owned model exited during the thinking age check')
                time.sleep(1)
            remote.copy(['thinking-age-check'],root)
            if check.returncode:
                raise RuntimeError('Thinking age check failed; official tasks were not started')
        q.require_free_port(18388)
        bridge=owned.start([PYTHON,'-B','-m','scripts.dsh_bridge','--protocol',str(root/'protocol.so'),
            '--tokenizer',profile['tokenizer'],'--backend',f'http://127.0.0.1:{profile["port"]}',
            '--model',profile['served_model_name'],'--output',str(root/'bridge'),'--port','18388',
            '--owner-namespace','tb21','--max-model-len',str(profile['max_model_len']),
            '--gen-budget',str(profile['kvmem']['gen_budget']),
            '--max-output-tokens',str(plan.get('output_cap',131072)),'--max-sessions',str(plan.get('bridge_max_sessions',BRIDGE_MAX_SESSIONS)),
            '--optimized-handoff',
            *(['--observation-tool-spans'] if plan.get('observation_tool_spans') else []),
            '--abort-on-disconnect','--progress-interval','2','--release-control-token',conn['control_token'],
            *(['--require-query-provenance'] if plan.get('query_provenance') else [])],
            runtime,direct_environment(q.environment(runtime)),root/'bridge.log',root/'bridge-process.json')
        q.ready_http(conn['base']+'/health',bridge,owned,timeout=120)
        for phase,config in [('smoke','smoke-job.json'),('formal','job.json')]:
            q.save(root/'status.json',dict(status=phase,at=time.time(),expected=expected_tasks))
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
                if server.poll() is not None or bridge.poll() is not None or gpu_monitor.poll() is not None:
                    job.send_signal(signal.SIGINT)
                    try:job.wait(timeout=120)
                    except subprocess.TimeoutExpired:pass
                    raise RuntimeError('Owned model/bridge/GPU monitor exited')
                time.sleep(5)
            if job.returncode:raise RuntimeError(f'{phase} exited {job.returncode}; preserve logs, no retry')
            if phase=='smoke':
                capture=root/'model/server/capture'
                if remote is not None:
                    remote.sync_capture()
                q.save(root/'formal-boundary.json',dict(at=time.time(),smoke_excluded=True,
                    resources=resources(root,owned),
                    capture={name:q.read(capture/(name+'.json'))
                             for name in ('archive','progress','scheduler_resources')}))
        q.save(root/'status.json',dict(status='needs_final_review',at=time.time(),expected=expected_tasks))
    except BaseException as error:
        q.save(root/'failure.json',dict(at=time.time(),error=repr(error)))
        q.save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error)))
        raise
    finally:
        remote_cleanup=None
        if remote is not None:
            try:remote_cleanup=remote.stop()
            except Exception as error:remote_cleanup=dict(error=repr(error))
        q.save(root/'cleanup.json',dict(owned.stop(),remote=remote_cleanup))
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
    p.add_argument('--agent-time-limit',choices=['none','official','relaxed','fixed'],default='relaxed',
                   help='Default: relaxed (2x official Agent limits, recorded pauses excluded); verifier/setup unchanged')
    p.add_argument('--profile',type=Path,help='Model and GC configuration to freeze')
    p.add_argument('--tasks',nargs='+',help='Explicit official task subset; omitted runs all 89')
    p.add_argument('--trial-concurrency',type=int,
                   help='Concurrent task lifecycles, including tools; default min(8, model slots). '
                        'May exceed model slots when session capacity covers the owners')
    p.add_argument('--query-mode',choices=['task','legacy'],default='task',help='Legacy only for an explicit baseline comparison')
    p.add_argument('--container-memory-gib',type=int,default=48,
                   help='Aggregate declared container-memory admission budget; default 48 GiB, with a separate 12-GiB host reserve')
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root,a.dataset.resolve(),a.reuse_images_from.resolve() if a.reuse_images_from else None,
                                  a.agent_time_limit,a.download_cache,a.profile,a.tasks,a.query_mode,a.trial_concurrency,
                                  a.container_memory_gib*1024**3)
    else:run(root)
