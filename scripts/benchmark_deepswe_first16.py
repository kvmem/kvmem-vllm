"""Frozen qw3 first16 Claude Code tasks, one attempt each, concurrent local backend.

The unchanged cell runner owns fresh agent/verifier containers and committed
patch scoring. This coordinator owns only its dedicated internal network,
model, protocol adapter and output tree. Benchmark failures are never retried.
"""
import argparse
import array
import asyncio
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import secrets
import types

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_CONCURRENCY = 8


def preflight_completion(http, cfg, payload, *, namespace, release_token, output, timeout):
    """Give cold synthetic probes temporary owners required by graph runtimes.

    Explicit continuation owners are retained for the caller. Temporary owners
    are released after their response; this does not change any formal request.
    """
    data={'model':cfg['served_model_name'],**payload}
    temporary=not data.get('session_id')
    if temporary:
        sid=namespace+'-probe-'+secrets.token_hex(6)
        data.update(session_id=sid,cache_salt=sid)
    records=output/'preflight-requests';records.mkdir(exist_ok=True)
    def post(body,phase):
        path=records/(secrets.token_hex(12)+'.json')
        row={'phase':phase,'request':body,'at':time.time()}
        request=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with http.open(request,timeout=timeout) as response:result=json.load(response)
        except urllib.error.HTTPError as error:
            row.update(error_status=error.code,error_body=error.read().decode(errors='replace'))
            save(path,row)
            raise RuntimeError(f"Preflight HTTP {error.code}: {row['error_body']}") from error
        row['response']=result;save(path,row)
        return result
    result=post(data,'probe')
    if temporary:
        post(dict(model=cfg['served_model_name'],prompt=[release_token],session_id=sid,cache_salt=sid,
            max_tokens=1,temperature=0,kvmem_reselect='off',kvmem_mandatory_spans=[],
            kv_transfer_params={'kvmem_release_session':True}),'temporary-owner-release')
    return result


def wait_backend_ready(http, endpoint, server, bridge, timeout_seconds=1800):
    """Allow cold kernel compilation equally for full-task and timed runs."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.poll() is not None or bridge.poll() is not None:
            raise RuntimeError('Backend startup failed')
        try:
            with http.open(endpoint+'/health', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(min(1, max(0, deadline-time.monotonic())))
    raise TimeoutError('Backend readiness timeout')


def preserve_incomplete_suite(out, children, manifest, state):
    if not manifest.get('seed_suite_qualification') or state.get('status')=='completed':
        return
    try:
        from scripts.deepswe_window_snapshot import snapshot, release_supervisors
        preserved=snapshot(out,children)
        save(out/'failure-workspace-preservation.json',dict(status='preserved',
            source='window-workspaces.json',scoring_unchanged=True))
        release_supervisors(preserved)
    except BaseException as snapshot_error:
        state['window_snapshot_error']=repr(snapshot_error)


def bounded_warmup(a, cfg, out, http):
    """Initialize the model and MeanK without including JIT in the window."""
    raw = a.warmup_tokens.read_bytes()
    values = array.array('i')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    tokens = values.tolist()
    expected_tokens=max(67584,cfg['kvmem']['select_budget']+2048)
    assert len(tokens) == expected_tokens and all(t >= 0 for t in tokens)
    native = a.backend_kind == 'native'
    sid = 'deepswe-initialization-' + out.name
    def complete(prompt, *, cleanup=False):
        data = dict(model=cfg['served_model_name'], prompt=prompt, max_tokens=1,
                    temperature=0, return_token_ids=True, cache_salt=sid)
        if not native:
            data.update(session_id=sid, kvmem_reselect='off' if cleanup else 'force',
                        kvmem_mandatory_spans=[])
            if not cleanup:
                data['kvmem_query_span'] = [len(tokens)-128, len(tokens)]
            else:
                data['kv_transfer_params'] = {'kvmem_release_session':True}
        request = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",
            data=json.dumps(data).encode(), headers={'Content-Type':'application/json'})
        with http.open(request, timeout=900) as response:
            return json.load(response)
    started = time.monotonic()
    result = complete(tokens)
    assert result['usage']['prompt_tokens'] == len(tokens) and result['usage']['completion_tokens'] == 1
    proof = {'required':not native, 'passed':True}
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    if not native:
        timed = load('deepswe_warmup_timed', 'timed_agent_workload.py')
        work = load('deepswe_warmup_capacity', 'capacity_scaling_workload.py')
        capture = out/'server/capture'
        asyncio.run(timed.wait_retirement(capture, not_before=started, expected_views={sid:len(tokens)}))
        proof = work.warmup_selection_evidence({'response_id':result['id']}, cfg,
            work.read_records(capture/'bounded_refresh.jsonl'), [len(tokens)-128,len(tokens)])
        assert proof['passed'], 'MeanK scoring initialization was not exercised'
    model_config = json.loads((Path(cfg['hf_config_path'])/'config.json').read_text())
    eos = model_config.get('text_config', model_config)['eos_token_id']
    cleaned_at = time.monotonic()
    cleanup = complete([eos], cleanup=True)
    if not native:
        retired=asyncio.run(timed.wait_retirement(capture, not_before=cleaned_at))
        assert all(sid not in v['namespace'] for v in retired.get('session_views',[])), 'Warmup session was not released'
        wait_preflight_release(capture)
    save_idle_metrics(cfg,out,http)
    save(out/'bounded-warmup.json', dict(passed=True, input_tokens=len(tokens), output_tokens=1,
        seconds=time.monotonic()-started, input_sha256=hashlib.sha256(raw).hexdigest(),
        response_id=result['id'], cleanup_id=cleanup['id'], kernel_load_proof=proof,
        scope='Initialization only; excluded from formal time/counters; isolated prefix-cache salt'))


def save_idle_metrics(cfg,out,http):
    """Exclude qualification outputs from formal counters after native drain."""
    deadline = time.monotonic()+60
    while True:
        with http.open(f"http://127.0.0.1:{cfg['port']}/metrics", timeout=5) as response:
            metrics = response.read().decode()
        gauges = [float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                  if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
        if len(gauges) == 2 and all(v == 0 for v in gauges):
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Warmup did not drain')
        time.sleep(.25)
    (out/'metrics-before-formal.txt').write_text(metrics)


def preflight_shape(profile):
    concurrency = min(PREFLIGHT_CONCURRENCY, profile.get('max_num_seqs', PREFLIGHT_CONCURRENCY))
    # Preserve the former K=36K stress geometry relative to the active budget.
    # A 128K profile must exercise reselection/recovery at 128K, not just 48K.
    select = profile['kvmem'].get('select_budget', 36864)
    initial = max(47662, select + 10798)
    return concurrency, initial, initial + 3972


def validate_session_capacity(profile, task_count):
    # Keep the original conservative slot provisioning for comparability.
    # Explicit release now removes completed/preflight namespaces entirely;
    # this check is not a bound on the live checkpoint payload bytes.
    preflight_sessions = preflight_shape(profile)[0] + 2  # generation, short tail, cleanup
    required = task_count + preflight_sessions
    capacity = profile['kvmem'].get('session_capacity', 0)
    if profile['kvmem'].get('cpu_history_coordination') and capacity < required:
        raise ValueError(f'DeepSWE requires at least {required} session slots '
                         f'({task_count} tasks + {preflight_sessions} preflight namespaces); '
                         f'configured {capacity}. Keep the original conservative slot budget.')
    return {'configured':capacity, 'required':required, 'preflight_namespaces':preflight_sessions,
            'cleanup_retains_namespace':False}


def validate_checkpoint_history(archive):
    """Capacity fallback to P must not silently pass a no-recompute run."""
    deferred = archive.get('session_checkpoint_deferrals')
    if type(deferred) is not int or deferred < 0:
        raise RuntimeError('Missing or invalid checkpoint deferral accounting')
    if deferred:
        raise RuntimeError('Checkpoint capacity deferred capture; complete-history/no-recompute contract failed')
    for row in archive.get('completed', []):
        status = (row.get('terminal_prefix') or {}).get('status')
        if (row.get('checkpoint_kind') in ('P','M') and
                status in ('FINISHED_STOPPED','FINISHED_LENGTH_CAPPED') and
                (row.get('checkpoint_kind') != 'M' or
                 row.get('history_end') != row.get('checkpoint_tokens'))):
            raise RuntimeError('Successful session retained an incomplete terminal history')


def wait_preflight_release(capture, timeout_seconds=60):
    """A cleanup HTTP response alone does not prove all retained state is gone."""
    deadline = time.monotonic()+timeout_seconds
    while True:
        archive=json.loads((capture/'archive.json').read_text())
        validate_checkpoint_history(archive)
        if (archive.get('session_checkpoints') == 0 and
                archive.get('session_checkpoint_payload_bytes') == 0 and
                archive.get('cpu_authority_bytes') == 0 and
                not any(archive.get(k) for k in ('retiring_requests','session_retiring_requests','pending_downloads'))):
            return archive
        if time.monotonic() >= deadline:
            raise TimeoutError('Preflight cleanup retained history or model checkpoints')
        time.sleep(.05)


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_context_limits(path, cursor, limited):
    """Consume complete records only, including a context error before CLI exit."""
    if not path.exists():
        return cursor
    with path.open() as stream:
        stream.seek(cursor)
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line.endswith('\n'):
                return position
            event = json.loads(line)
            if event.get('event') == 'context_limit':
                limited.add(event['task'])


def read_client_context_limit(path, model, context_tokens):
    """Recognize the pinned CLI's local ceiling after it exits, without an API call.

    Tool/assistant text is not termination evidence. Require the final complete
    record to be the structured blocking-limit result for this model/profile;
    malformed or incomplete logs retain the infrastructure-failure policy.
    """
    if not path.exists():
        return None
    final = None
    with path.open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                return None
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                return None
            if not isinstance(row, dict):
                return None
            final = (line_number, line, row)
    if final is None:
        return None
    line_number, line, row = final
    if (row.get('type') != 'result' or row.get('is_error') is not True
            or row.get('terminal_reason') != 'blocking_limit'
            or row.get('result') != 'Prompt is too long'
            or row.get('api_error_status') is not None):
        return None
    usage = row.get('modelUsage')
    usage = usage.get(model) if isinstance(usage, dict) else None
    if (not isinstance(usage, dict)
            or type(usage.get('contextWindow')) is not int
            or usage['contextWindow'] != context_tokens):
        return None
    return dict(source='client_local_guard', path=str(path), line=line_number,
                record_sha256=hashlib.sha256(line).hexdigest(),
                terminal_reason=row['terminal_reason'], session_id=row.get('session_id'),
                model=model, context_tokens=context_tokens,
                reported_max_output_tokens=usage.get('maxOutputTokens'))


def prepare(a):
    m=json.loads((a.reference/'manifest.json').read_text())
    assert len(m['tasks'])==16
    profile=json.loads(a.config.read_text())
    capacity=validate_session_capacity(profile,len(m['tasks']))
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    setup=output/'setup';setup.mkdir(exist_ok=True)
    for name in ('container_claude_agent.sh','run_deepswe_cell.sh'):
        shutil.copyfile(a.reference/'setup'/name,setup/name)
    shutil.copyfile(a.reference/'tasks.txt',output/'tasks.txt')
    policy=Path('/home/chaidi/qw3/benchmark/attention_score_fusion_deepswe_20261012/deepswe/nonconvergence_frozen.py').read_text()
    guard='import json\n'+policy[policy.index('REFERENCES ='):policy.index('\ndef atomic(')]
    (setup/'read_guard.py').write_text(guard)
    assert digest(m['claude_binary'])==m['claude_sha256']
    for task in m['tasks']:
        assert digest(task['instruction'])==task['instruction_sha256']
        for name,expected in task['images'].items():
            actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',name],text=True).strip()
            assert actual==expected,('image mismatch',name)
    manifest={'reference':str(a.reference),'reference_sha256':digest(a.reference/'manifest.json'),
        'tasks':m['tasks'],'claude_sha256':m['claude_sha256'],'workers':a.workers,
        'sampling':m['sampling'],'seed':getattr(a, 'seed', 1000),'attempts_per_task':1,
        'setup_sha256':{p.name:digest(p) for p in setup.iterdir()},
        'profile':profile,'profile_sha256':digest(a.config), 'session_capacity':capacity,
        'protocol_provenance':json.loads((output/'protocol/provenance.json').read_text()),
        'backend_kind':a.backend_kind,
        'duration_seconds':a.duration_seconds,
        'inference':f"NVFP4 + FP8 KV + {profile.get('draft_method','dflash')}({profile['draft_tokens']}), "
            f"execution={profile.get('execution_mode','eager')}, backend={a.backend_kind}; original qw3 text protocol helpers.",
        'differences':[f"vLLM native per-request seed={getattr(a, 'seed', 1000)} (qw3 server RNG is sequential)",
            'Current attention pressure + Query history; guided/private growth/evidence triggers remain disabled as current implementation profile',
            f"Native generation max_tokens is capped at G={profile['kvmem']['gen_budget']}; logical position limit {profile['max_model_len']}",
            'Anthropic SSE buffers a completed model response with keepalive pings; agent receives identical completed tool content'],
        'failure_policy':'Reward zero is final. Infrastructure failure pauses new admission and preserves all attempts.',
        'scoring':'Original frozen hidden verifier, committed model.patch only, no auto-commit of working tree',
        'network':'Dedicated internal Docker bridge, no WAN; protocol bridge binds its gateway only',
        'agent_timeout':'Original unlimited','verifier_timeout':'Original 30m'}
    manifest['optimized_handoff'] = bool(getattr(a, 'optimized_handoff', False))
    manifest['cpu_admission']={'budget_gib':72,'active_agent_clients':a.workers,
        'short_trajectory_reserve_tokens':49152,'long_trajectory_reserve_tokens':1048576,
        'long_trajectory_slots':1,'policy':'CPU-only; preserve parked histories, no eviction/recompute. Up to eight clients while short, one long trajectory at a time. Release finished histories before admitting further growth.'}
    if a.backend_kind == 'native':
        manifest['differences']=[v for v in manifest['differences'] if not v.startswith('Current attention')]
        manifest['cpu_admission']={'policy':'native_vllm_gpu_scheduler', 'active_agent_clients':a.workers,
            'cpu_kvmem_history':False}
    elif manifest['profile']['kvmem'].get('cpu_history_coordination'):
        manifest['cpu_admission'] = {'budget_bytes':manifest['profile']['kvmem']['cpu_read_cache_bytes'],
            'active_agent_clients':a.workers, 'policy':'engine_shared_history_credits_with_progress_reserve',
            'max_history_tokens':manifest['profile']['max_model_len']}
    manifest['termination_policies']={'cached_read_streak':32,'identical_successful_bash_streak':a.bash_repeat_limit or None}
    manifest['carried_results']=[{'path':str(p.resolve()),'sha256':digest(p)} for p in a.carry_result]
    save(output/'manifest.json',manifest)
    return manifest


def run(a):
    if not math.isfinite(a.duration_seconds) or a.duration_seconds < 0:
        raise ValueError('Observation duration must be finite and nonnegative')
    if a.duration_seconds and (a.adopt_controller or a.duration_seconds < 60 or a.warmup_tokens is None):
        raise ValueError('Bounded windows require >=60 seconds, warmup tokens and a new controller')
    if getattr(a, 'snapshot_window_workspaces', False) and not a.duration_seconds:
        raise ValueError('Workspace snapshots require a finite observation window')
    if a.bash_repeat_limit and a.bash_repeat_limit<2:raise ValueError('Bash repetition limit must be zero or at least two')
    m=json.loads((a.output/'manifest.json').read_text()) if a.adopt_controller or a.use_prepared else prepare(a)
    if type(getattr(a, 'seed', 1000)) is not int or not 0 <= getattr(a, 'seed', 1000) < 2**32:
        raise ValueError('Benchmark seed must be an unsigned 32-bit integer')
    if getattr(a, 'seed', 1000) != m.get('seed', 1000):
        raise ValueError('Requested seed differs from the frozen task manifest')
    if a.use_prepared:
        assert json.loads(a.config.read_text())==m['profile']
        assert (a.workers,a.backend_kind,a.duration_seconds)==(m['workers'],m['backend_kind'],m['duration_seconds'])
        assert bool(getattr(a, 'optimized_handoff', False)) == m.get('optimized_handoff', False)
    if a.adopt_controller:
        assert json.loads(a.config.read_text())==m['profile'], 'Adoption must preserve the inference profile'
        assert a.workers==m['workers'], 'Adoption must preserve client concurrency'
    if a.prepare_only:
        print(json.dumps({'prepared':16,'workers':a.workers,'images':32}));return
    out=a.output.resolve();cfg=json.loads(a.config.read_text())
    preflight_concurrency, initial_tokens, continuation_tokens = preflight_shape(cfg)
    generation_tokens = cfg['kvmem']['gen_budget']
    assert a.adopt_controller or not (out/'run.json').exists(), 'Preserve existing attempts; no implicit rerun'
    run_id='kvds-'+str(int(time.time()))
    environment={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')
                 and not k.endswith(('API_KEY','AUTH_TOKEN'))}
    environment['PYTHONPATH']=str(ROOT/'src')
    environment['TOKENIZERS_PARALLELISM']='false'
    environment['PYTHONDONTWRITEBYTECODE']='1'
    control_token=secrets.token_hex(24)
    network=run_id
    children=[];logs=[];stop=threading.Event();window_done=threading.Event();state_lock=threading.Lock()
    window_closing=threading.Event()
    context_limited=set()
    state={'id':run_id,'started_at':time.time(),'workers':a.workers,'status':'starting','tasks':{},'network':network}
    for path in a.carry_result:
        carried=json.loads(path.read_text());tid=carried['task_id']
        assert tid in {t['task_id'] for t in m['tasks']} and tid not in state['tasks']
        summary=Path(carried['artifact'])/'cell_summary.json'
        assert digest(summary)==carried['cell_summary_sha256']
        assert json.loads(summary.read_text())==carried['result']
        assert carried['result']['verifier_exit']==0 and carried['result']['reward']['reward'] in (0,1)
        state['tasks'][tid]={**carried,'state':'completed','carried':True}
    adopted_cells={}
    if a.adopt_controller:
        state=json.loads((out/'run.json').read_text());run_id=state['id'];network=state['network']
    sys.path.insert(0,str(ROOT/'benchmarks/deepswe'))
    from repetition import BashRepeatGuard, expected_stop_cancellation
    guard_module=types.ModuleType('frozen_read_guard')
    exec((out/'setup/read_guard.py').read_text(),guard_module.__dict__)
    def update():save(out/'run.json',state)
    def spawn(command, log, env=None):
        if len(command)>3 and command[0]=='bash' and command[2] in adopted_cells:
            return adopted_cells.pop(command[2])
        f=log.open('wb');logs.append(f)
        process=subprocess.Popen(command,cwd=ROOT,env=env or environment,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(process);return process
    def interrupt(sig,frame):
        stop.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    network_created=False
    owns_run=not bool(a.adopt_controller)
    try:
        if a.adopt_controller:
            from supervision import adopt
            existing,adopted_cells,control_token,transferred=adopt(a.adopt_controller,ROOT,out)
            owns_run=True
            state=json.loads((out/'run.json').read_text())
            server=existing['server'];bridge=existing['bridge'];children.extend([server,bridge,*adopted_cells.values()])
            network_created=True
            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            endpoint=f"http://{state['gateway']}:{a.bridge_port}"
            revision={'at':time.time(),'previous_controller':transferred,
                'inference_restarted':False,'client_restarted':False,
                'bash_repeat_limit':a.bash_repeat_limit,
                'policy':'Original cached Read guard plus exact successful Bash input/output repetition',
                'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (
                    Path(__file__),ROOT/'benchmarks/deepswe/repetition.py',ROOT/'benchmarks/deepswe/supervision.py')}}
            save(out/'supervision-revision.json',revision)
            state['supervision_revision']=revision;update()
        else:
            subprocess.run(['docker','network','create','--internal','--label','com.kvmem.deepswe.owner='+run_id,network],check=True,capture_output=True)
            network_created=True
            data=json.loads(subprocess.check_output(['docker','network','inspect',network],text=True))[0]
            gateway=data['IPAM']['Config'][0]['Gateway'];state['gateway']=gateway;update()
            if a.backend_kind == 'native':
                sys.path.insert(0, str(ROOT/'scripts'))
                from native_agent_launch import build_launch
                launch=build_launch(cfg,port=cfg['port'],async_scheduling=True)
                save(out/'native-launch.json',{k:v for k,v in launch.items() if k!='environment'})
                server=spawn(launch['argv'],out/'server.log',launch['environment'])
            else:
                server=spawn([sys.executable,'-m','vllm_kvmem','serve','--config',str(a.config.resolve()),'--run-dir',str(out/'server')],out/'server.log')
            bridge=spawn([sys.executable,str(ROOT/'benchmarks/deepswe/claude_bridge.py'),
                '--protocol',str(out/'protocol/protocol.so'),'--tasks',str(out/'tasks.txt'),
                '--output',str(out/'bridge'),'--tokenizer',cfg['tokenizer'],
                '--backend',f"http://127.0.0.1:{cfg['port']}",'--model',cfg['served_model_name'],
                '--run-id',run_id,'--host',gateway,'--port',str(a.bridge_port),
                '--backend-kind',a.backend_kind,
                '--max-model-len',str(cfg['max_model_len']),
                '--gen-budget',str(generation_tokens),
                '--seed',str(m.get('seed', 1000)),
                *(['--optimized-handoff'] if getattr(a, 'optimized_handoff', False) else []),
                *(['--cpu-admission-policy','engine','--cache-geometry',str(out/'server/capture/cache-geometry.json'),
                   '--cpu-admission-bytes',str(cfg['kvmem']['cpu_read_cache_bytes'])]
                  if a.backend_kind == 'kvmem' and cfg['kvmem'].get('cpu_history_coordination') else [])],out/'bridge.log',
                {**environment,'DEEPSWE_CONTROL_TOKEN':control_token})
            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            endpoint=f'http://{gateway}:{a.bridge_port}'
            wait_backend_ready(http,endpoint,server,bridge,
                timeout_seconds=getattr(a,'startup_timeout_seconds',1800))
            if m.get('seed_suite_qualification'):
                from scripts.queue_deepswe_seed_suite import validate_preflight_reuse
                validate_preflight_reuse(out)
                bounded_warmup(a,cfg,out,http)
            elif m.get('preflight_reuse'):
                from scripts.continue_deepswe_formal import validate_reuse, fresh_engine_archive
                proof = validate_reuse(out.parent)
                archive = fresh_engine_archive(out/'server/capture')
                save(out/'preflight-reuse.json', dict(status='reused', source=proof['source'],
                    long_gc_status='user_stopped_incomplete', at=time.time()))
                state['preflight_native_counters'] = {k:archive.get(k,0) for k in (
                    'session_restored_tokens','session_checkpoint_deferrals','disk_read_bytes','disk_write_bytes')}
                save_idle_metrics(cfg,out,http)
            elif a.duration_seconds:
                bounded_warmup(a,cfg,out,http)
            else:
                # Reproduce atomic Query batches larger than the normal 1024-token
                # ingestion chunk. Synthetic content only; no benchmark answer sampled.
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(cfg['tokenizer'],local_files_only=True)
                base=(tokenizer.encode('An ordinary synthetic context. ',add_special_tokens=False)*5000)[:16384]
                query=(tokenizer.encode('A synthetic Query for attention qualification. ',add_special_tokens=False)*1000)[:4096]
                assert len(base)==16384 and len(query)==4096
                def complete(payload):
                    return preflight_completion(http,cfg,payload,namespace=run_id,
                        release_token=tokenizer.convert_tokens_to_ids('<|im_end|>'),output=out,
                        timeout=max(1200,generation_tokens/10))
                def qualify(index):
                    payload={'model':cfg['served_model_name'],'prompt':base+query,'max_tokens':1,'temperature':0,
                             'kvmem_query_span':[len(base),len(base)+len(query)]}
                    return complete(payload)
                with ThreadPoolExecutor(max_workers=preflight_concurrency) as pool:
                    qualification=list(pool.map(qualify,range(preflight_concurrency)))
                save(out/'atomic-query-preflight.json',{'passed':True,'concurrency':preflight_concurrency,'query_tokens':4096,'sampled_rows_per_query':128,
                    'requests':[{'id':r['id'],'usage':r['usage']} for r in qualification]})
                # Prove that clearing a finished namespace also clears its mandatory
                # spans before the short cleanup prompt is admitted.
                # Exercise cold and restored one-token tails inside a live decode
                # batch before paying for the full-G qualification.
                one=[tokenizer.convert_tokens_to_ids('<|im_end|>')]
                uninterrupted=complete({'prompt':one,'max_tokens':2,'return_token_ids':True,
                    'temperature':0,'kvmem_reselect':'off'})
                with ThreadPoolExecutor(max_workers=1) as pool:
                    peer=pool.submit(complete,{'prompt':base[:2048],'max_tokens':1024,
                        'ignore_eos':True,'temperature':0,'kvmem_reselect':'off'})
                    deadline=time.monotonic()+60
                    while True:
                        progress=json.loads((out/'server/capture/progress.json').read_text())
                        if any(2048<h<3072 for h in progress.get('accepted_history',{}).values()):break
                        if peer.done() or time.monotonic()>deadline:
                            raise RuntimeError('Short-tail regression did not overlap live decode')
                        time.sleep(.1)
                    sid=run_id+'-short-tail-preflight'
                    short=complete({'prompt':one,'session_id':sid,'cache_salt':sid,'max_tokens':1,
                        'return_token_ids':True,'temperature':0,'kvmem_reselect':'off','kvmem_mandatory_spans':[]})
                    assert not peer.done(),'Short tail must join a live decode batch'
                    warm=complete({'prompt':one+short['choices'][0]['token_ids'],'session_id':sid,'cache_salt':sid,
                        'max_tokens':1,'return_token_ids':True,'temperature':0,'kvmem_reselect':'off','kvmem_mandatory_spans':[]})
                    reference=complete({'prompt':one+short['choices'][0]['token_ids'],'max_tokens':1,
                        'return_token_ids':True,'temperature':0,'kvmem_reselect':'off'})
                    save(out/'short-tail-preflight-evidence.json',{'uninterrupted':uninterrupted,
                        'cold':short,'warm':warm,'mixed_cold_reference':reference,
                        'warm_uninterrupted_token_equal':short['choices'][0]['token_ids']+warm['choices'][0]['token_ids']==uninterrupted['choices'][0]['token_ids']})
                    assert short['choices'][0]['token_ids']+warm['choices'][0]['token_ids']==uninterrupted['choices'][0]['token_ids']
                    peer_result=peer.result()
                save(out/'short-tail-preflight.json',{'passed':True,'cold':short['id'],'warm':warm['id'],
                    'peer':peer_result['id'],'reference':reference['id'],'uninterrupted':uninterrupted['id'],
                    'warm_uninterrupted_token_equal':True,
                    'mixed_cold_reference_token_equal':warm['choices'][0]['token_ids']==reference['choices'][0]['token_ids']})
                released=complete({'prompt':one,'session_id':sid,'cache_salt':sid,'max_tokens':1,
                    'temperature':0,'kvmem_reselect':'off','kvmem_mandatory_spans':[],
                    'kv_transfer_params':{'kvmem_release_session':True}})
                save(out/'short-tail-release.json',{'request_id':released['id']})
                # Full-G warm continuations reproduce the capacity error that short
                # completions cannot expose. Use synthetic data and fresh namespaces.
                context=(base*((initial_tokens+len(base)-1)//len(base)))[:initial_tokens]
                def qualify_generation(index):
                    sid=run_id+'-generation-preflight-'+str(index)
                    initial=complete({'prompt':context,'session_id':sid,'cache_salt':sid,
                        'max_tokens':1,'temperature':0,'return_token_ids':True,
                        'kvmem_query_span':[18981,20287]})
                    prefix=context+initial['choices'][0]['token_ids']
                    prompt=prefix+base[:continuation_tokens-len(prefix)]
                    started=time.monotonic()
                    result=complete({'prompt':prompt,'session_id':sid,'cache_salt':sid,
                        'max_tokens':generation_tokens,'ignore_eos':True,'temperature':0,
                        'kvmem_query_span':[initial_tokens+50,continuation_tokens]
                            if index==preflight_concurrency-1 else [18981,20287]})
                    assert result['usage']['completion_tokens']==generation_tokens
                    row={'initial_id':initial['id'],'request_id':result['id'],
                         'usage':result['usage'],'seconds':time.monotonic()-started,'cleanup_passed':False}
                    save(out/f'generation-preflight-{index}.json',row)
                    complete({'prompt':[tokenizer.convert_tokens_to_ids('<|im_end|>')],
                        'session_id':sid,'cache_salt':sid,'max_tokens':1,'temperature':0,
                        'kvmem_reselect':'off','kvmem_mandatory_spans':[],
                        'kv_transfer_params':{'kvmem_release_session':True}})
                    row['cleanup_passed']=True
                    save(out/f'generation-preflight-{index}.json',row)
                    return row
                with ThreadPoolExecutor(max_workers=preflight_concurrency) as pool:
                    generation=list(pool.map(qualify_generation,range(preflight_concurrency)))
                save(out/'generation-preflight.json',{'passed':True,'concurrency':preflight_concurrency,
                    'generation_tokens':generation_tokens,
                    'initial_prompt_tokens':initial_tokens,'continuation_prompt_tokens':continuation_tokens,
                    'results':generation})
                namespace=run_id+'-cleanup-preflight'
                initial=complete({'prompt':base[:256],'session_id':namespace,'cache_salt':namespace,
                    'max_tokens':1,'temperature':0,'kvmem_query_span':[128,256],
                    'kvmem_mandatory_spans':[{'begin':0,'end':128,'kind':'system'}]})
                cleared=complete({'prompt':[tokenizer.convert_tokens_to_ids('<|im_end|>')],
                    'session_id':namespace,'cache_salt':namespace,'max_tokens':1,'temperature':0,
                    'kvmem_reselect':'off','kvmem_mandatory_spans':[],
                    'kv_transfer_params':{'kvmem_release_session':True}})
                save(out/'cleanup-preflight.json',{'passed':True,'initial_request':initial['id'],'cleanup_request':cleared['id']})
                archive=wait_preflight_release(out/'server/capture')
                if m.get('long_gc_qualification'):
                    from scripts.deepswe_long_gc_preflight import run as qualify_long_gc
                    qualify_long_gc(cfg, out, m['long_gc_qualification'], tokenizer,
                                    namespace=run_id, server=server, bridge=bridge)
                    archive=wait_preflight_release(out/'server/capture')
                state['preflight_native_counters']={k:archive.get(k,0) for k in (
                    'session_restored_tokens','session_checkpoint_deferrals','disk_read_bytes','disk_write_bytes')}
                save_idle_metrics(cfg,out,http)
            state['status']='running';state['model_ready_at']=time.time();
            state['model_ready_monotonic']=time.monotonic()
            state['window_deadline']=state['model_ready_at']+a.duration_seconds if a.duration_seconds else None
            update()
        monitor_done=threading.Event()
        def monitor():
            cursor=0
            while not monitor_done.is_set():
                if a.duration_seconds and time.time() >= state['window_deadline']:
                    window_closing.set()
                    preserved=None
                    try:
                        if getattr(a, 'snapshot_window_workspaces', False):
                            measurement={'started_at':time.time(), 'started_monotonic':time.monotonic()}
                            with http.open(f"http://127.0.0.1:{cfg['port']}/metrics",timeout=5) as response:
                                measurement['raw_metrics']=response.read().decode()
                            measurement.update(finished_at=time.time(),finished_monotonic=time.monotonic())
                            measurement['stream_bytes']={str(p.relative_to(out)):p.stat().st_size
                                for p in (out/'cases').glob('*/claude.stream.jsonl')}
                            requests=out/'bridge/requests.jsonl'
                            measurement['bridge_bytes']=requests.stat().st_size if requests.exists() else 0
                            save(out/'window-end-metrics.json',measurement)
                            from scripts.deepswe_window_snapshot import snapshot
                            preserved=snapshot(out,children)
                    except BaseException as error:
                        state['window_snapshot_error']=repr(error)
                    finally:
                        window_done.set();stop.set()
                        if preserved:
                            from scripts.deepswe_window_snapshot import release_supervisors
                            release_supervisors(preserved)
                    return
                if server.poll() is not None or bridge.poll() is not None:
                    stop.set()
                sample={'at':time.time()}
                try:
                    memory={k:int(v.split()[0])*1024 for k,v in
                            (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
                    sample={'at':time.time(),'available_bytes':memory['MemAvailable'],
                            'swap_used_bytes':memory['SwapTotal']-memory['SwapFree'],
                            'disk_free_bytes':shutil.disk_usage(out).free}
                    if sample['available_bytes'] < 8*1024**3:
                        state['resource_error']='Host memory headroom below 8GiB'
                        stop.set()
                    cursor = read_context_limits(out/'bridge/requests.jsonl', cursor, context_limited)
                    for name in ('archive','scheduler_resources'):
                        source=out/'server/capture'/f'{name}.json'
                        if source.exists():
                            value=json.loads(source.read_text())
                            if name=='archive':
                                if a.backend_kind == 'kvmem':
                                    try:
                                        validate_checkpoint_history(value)
                                    except RuntimeError as error:
                                        state['history_contract_error']=str(error)
                                        stop.set()
                                value={k:v for k,v in value.items() if k in (
                                    'at','archive_backend','cpu_authority_bytes','disk_read_bytes','disk_write_bytes',
                                    'session_restored_tokens','session_checkpoint_payload_bytes','pending_downloads',
                                    'session_retiring_requests','session_checkpoint_deferrals')}
                            sample[name]=value
                    # History correctness must still be checked when the
                    # telemetry endpoint stalls or the GPU sampler fails.
                    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=5)
                    sample['gpu']=r.stdout.strip()
                    with http.open(f"http://127.0.0.1:{cfg['port']}/metrics",timeout=5) as response:
                        raw=response.read().decode()
                    sample['raw_metrics']=raw
                except Exception as error:sample['telemetry_error']=repr(error)
                with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
                monitor_done.wait(1)
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        def task_attempt(row):
            tid=row['task_id']
            if stop.is_set() or window_closing.is_set() or state['tasks'].get(tid,{}).get('state')=='completed':return
            cell=out/'cases'/tid
            if tid not in adopted_cells:cell.mkdir(parents=True)
            task_env={**environment,'CLAUDE_AGENT_RUNNER':str(out/'setup/container_claude_agent.sh'),
                'QWEN38_CLAUDE_API_BASE':endpoint+'/task/'+tid,'QWEN38_CLAUDE_MODEL_ID':cfg['served_model_name'],
                'QWEN38_PRECISION_LABEL':'NVFP4','QWEN38_CLAUDE_AGENT_TIMEOUT':'0',
                'QWEN38_CLAUDE_AGENT_MEMORY':'16g','QWEN38_CLAUDE_CONTEXT_TOKENS':str(cfg['max_model_len']),
                'QWEN38_CLAUDE_DISABLE_COMPACT':'1','QWEN38_CLAUDE_VERIFIER_TIMEOUT':'30m',
                'QWEN38_SHARED_VLLM_LOG':str(out/'server.log'),'QWEN38_CLAUDE_CONDITION':'vllm_'+a.backend_kind,
                'QWEN38_AGENT_NETWORK':network,'QWEN38_AGENT_NETWORK_POLICY':'docker_internal_bridge_no_wan',
                'QWEN38_VERIFIER_IMAGE':row['verifier_image']}
            with state_lock:
                state['tasks'].setdefault(tid,{'state':'running','started_at':time.time()});update()
            child=spawn(['bash',str(out/'setup/run_deepswe_cell.sh'),tid,str(cell)],cell/'supervisor.log',task_env)
            released=False
            def release_history():
                request=urllib.request.Request(endpoint+'/control/'+tid+'/finish',data=b'{}',
                    headers={'Content-Type':'application/json','x-control-token':control_token})
                with http.open(request,timeout=180) as response:
                    assert json.load(response)['released']
            guard=guard_module.ReadGuard();bash_guard=BashRepeatGuard(a.bash_repeat_limit) if a.bash_repeat_limit else None
            cursor=0;signalled=None
            while child.poll() is None:
                if window_done.is_set():
                    with state_lock:
                        state['tasks'][tid].update(state='window_truncated',ended_at=time.time(),scored=False)
                        update()
                    return
                if tid in context_limited:
                    # The configured context ceiling is a declared workload boundary.
                    # Stop this owned attempt and replenish its worker; it has
                    # no valid verifier score and is not an infrastructure fault.
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                    release_history()
                    with state_lock:
                        state['tasks'][tid].update(state='context_ceiling',ended_at=time.time(),scored=False)
                        update()
                    return
                if stop.is_set() or server.poll() is not None or bridge.poll() is not None:
                    raise RuntimeError('Infrastructure stopped; preserve this attempt without scoring')
                status=json.loads((cell/'status.json').read_text()) if (cell/'status.json').exists() else {}
                trajectory=cell/'claude.stream.jsonl'
                if trajectory.exists():
                    with trajectory.open() as stream:
                        stream.seek(cursor)
                        while True:
                            before=stream.tell();line=stream.readline()
                            if not line.endswith('\n'):
                                cursor=before;break
                            guard.line(line)
                            if bash_guard is not None:bash_guard.line(line)
                trigger=guard.first_trigger or (bash_guard.first_trigger if bash_guard else None)
                if trigger is not None and signalled is None and (cell/'agent.container.cid').exists():
                    cid=(cell/'agent.container.cid').read_text().strip()
                    inspected=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
                    labels=inspected['Config'].get('Labels') or {}
                    assert labels.get('com.qw3.deepswe.run-owner')==hashlib.sha256(str(cell).encode()).hexdigest()[:16]
                    assert labels.get('com.qw3.deepswe.role')=='agent'
                    code="""import os,signal,json
from pathlib import Path
pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 if args and args[0]==b'/usr/local/bin/claude' and b'-p' in args and b'stream-json' in args:pids.append(int(p.name))
if len(pids)!=1:raise RuntimeError('Expected exactly one owned CLI')
for pid in pids:os.kill(pid,signal.SIGTERM)
print(json.dumps({'cli_pids':pids}))
"""
                    stop_started=time.time()
                    signal_result=json.loads(subprocess.check_output(['docker','exec',cid,'python3','-c',code]))
                    signalled={'at':stop_started,'policy':'reference-read-streak32-v1' if guard.first_trigger else 'exact-bash-input-output-streak-v1','first_trigger':trigger,**signal_result}
                    save(cell/'nonconvergence-stop.json',signalled)
                if not released and status.get('state') in ('verifying','finished'):
                    release_history()
                    released=True
                time.sleep(1)
            code=child.returncode
            read_context_limits(out/'bridge/requests.jsonl', 0, context_limited)
            if not released:release_history()
            if tid in context_limited:
                with state_lock:
                    state['tasks'][tid].update(state='context_ceiling',ended_at=time.time(),scored=False,
                        runner_exit=code)
                    update()
                return
            result=json.loads((cell/'cell_summary.json').read_text()) if (cell/'cell_summary.json').exists() else {}
            bridge_errors=[]
            if (out/'bridge/requests.jsonl').exists():
                bridge_errors=[r for line in (out/'bridge/requests.jsonl').read_text().splitlines()
                               if (r:=json.loads(line)).get('task')==tid and r.get('event') in ('error','cancelled')]
            # The owned CLI stop can cancel its in-flight streaming request.
            # Preserve that evidence separately; other API errors still invalidate.
            stop_cancellations=[r for r in bridge_errors if expected_stop_cancellation(r,signalled)]
            bridge_errors=[r for r in bridge_errors if r not in stop_cancellations]
            client_limit = read_client_context_limit(
                cell/'claude.stream.jsonl', cfg['served_model_name'], cfg['max_model_len'])
            if (client_limit is not None and result.get('agent_exit') == 1
                    and not signalled and not bridge_errors):
                # A local CLI guard never reaches the bridge's context_limit log.
                # Keep any verifier artifacts, but do not score this bounded attempt
                # or stop the other workers. Real transport/service failures still
                # follow the existing fail-closed path.
                save(cell/'client-context-limit.json', client_limit)
                with state_lock:
                    state['tasks'][tid].update(state='context_ceiling', ended_at=time.time(),
                        scored=False, runner_exit=code, context_limit=client_limit)
                    update()
                return
            valid=((code==0 or getattr(child,'adopted',False)) and (result.get('agent_exit')==0 or (signalled and result.get('agent_exit') in (130,137,143)))
                   and not bridge_errors and result.get('verifier_exit')==0
                   and isinstance(result.get('reward'),dict) and result['reward'].get('reward') in (0,1))
            with state_lock:
                state['tasks'][tid].update(state='completed' if valid else 'infrastructure_error',
                    ended_at=time.time(),runner_exit=code,runner_exit_observed=not getattr(child,'adopted',False),result=result,bridge_errors=bridge_errors,
                    expected_stop_cancellations=stop_cancellations,
                    termination='nonconvergent' if signalled else 'natural')
                if not valid:stop.set()
                update()
            print(json.dumps({'task':tid,'valid':valid,'reward':result.get('reward')}),flush=True)
        def task(row):
            try:
                return task_attempt(row)
            except BaseException as error:
                stop.set()
                with state_lock:
                    state['tasks'].setdefault(row['task_id'],{}).update(
                        state='infrastructure_error',ended_at=time.time(),error=repr(error))
                    update()
                raise
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(task,row) for row in m['tasks']]
            for future in as_completed(futures):future.result()
        state['status']='needs_attention' if state.get('window_snapshot_error') else 'window_complete' if window_done.is_set() else 'completed' if len(state['tasks'])==16 and not stop.is_set() else 'needs_attention'
    except BaseException as error:
        if owns_run:state['status']='needs_attention';state['error']=repr(error)
        raise
    finally:
        if owns_run:
            # A failed or explicitly stopped formal cohort must retain the
            # unfinished files before wrapper EXIT traps remove containers.
            preserve_incomplete_suite(out,children,m,state)
        # Publish termination before potentially slow process/container cleanup.
        # Otherwise monitors keep counting an already stopped run as inference.
        if owns_run:state['ended_at']=time.time();update()
        if 'monitor_done' in locals():monitor_done.set();monitor_thread.join(timeout=10)
        cleanup_errors=[]
        retained=set()
        if state.get('window_snapshot_error'):
            # A failed filesystem snapshot must not destroy the only remaining
            # workspace through a cell wrapper's EXIT trap. Leave these exact
            # children/container network for an audited manual recovery.
            for child in children:
                args=getattr(child,'args',[])
                if len(args)>1 and args[:2]==['bash',str(out/'setup/run_deepswe_cell.sh')]:
                    retained.add(child.pid)
            cleanup_errors.append({'snapshot':state['window_snapshot_error'],
                                   'retained_cell_processes':sorted(retained)})
        for child in reversed(children):
            if child.pid in retained:continue
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cleanup_deadline=time.monotonic()+15
        for child in reversed(children):
            if child.pid in retained:continue
            try:child.wait(timeout=max(0,cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:cleanup_errors.append({'pid':child.pid,'error':'Process did not exit after kill'})
        for log in logs:log.close()
        if network_created and not state.get('window_snapshot_error'):
            # Killing a docker exec client can leave its detached agent alive.
            # Remove only containers whose role and per-cell owner both match.
            for cid_path in (out/'cases').glob('*/*.container.cid'):
                role=cid_path.name.split('.',1)[0]
                if role not in ('agent','verifier'):continue
                cid=cid_path.read_text().strip()
                inspected=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15)
                if inspected.returncode:continue
                labels=json.loads(inspected.stdout)[0]['Config'].get('Labels') or {}
                owner=hashlib.sha256(str(cid_path.parent).encode()).hexdigest()[:16]
                if labels.get('com.qw3.deepswe.run-owner')!=owner or labels.get('com.qw3.deepswe.role')!=role:
                    cleanup_errors.append({'container':cid,'error':'Ownership mismatch; left untouched'});continue
                removed=subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=30)
                if removed.returncode:cleanup_errors.append({'container':cid,'error':removed.stderr})
            removed=subprocess.run(['docker','network','rm',network],capture_output=True,text=True,timeout=15)
            if removed.returncode:cleanup_errors.append({'network':network,'error':removed.stderr})
        if owns_run:
            state['cleanup_finished_at']=time.time();state['cleanup_errors']=cleanup_errors;update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('/home/chaidi/qw3/benchmark/action_pressure_first16_warmfix_20261010'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2,4,8),default=8)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--adopt-controller',type=int,help='Transfer the owned active run without restarting model or clients')
    p.add_argument('--bash-repeat-limit',type=int,default=0,help='Opt-in exact successful Bash input/output repetition policy; zero preserves reference')
    p.add_argument('--carry-result',type=Path,action='append',default=[],help='Preserve an independently audited completed result without retrying it')
    p.add_argument('--backend-kind',choices=('native','kvmem'),default='kvmem')
    p.add_argument('--duration-seconds',type=float,default=0,help='Formal fixed observation window; zero runs all tasks')
    p.add_argument('--warmup-tokens',type=Path)
    p.add_argument('--snapshot-window-workspaces',action='store_true',help='Preserve unfinished owned container files at a finite window boundary')
    p.add_argument('--use-prepared',action='store_true',help='Use the immutable prevalidated task manifest')
    p.add_argument('--bridge-port',type=int,default=18483);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--optimized-handoff',action='store_true')
    p.add_argument('--startup-timeout-seconds',type=int,default=1800)
    run(p.parse_args())
