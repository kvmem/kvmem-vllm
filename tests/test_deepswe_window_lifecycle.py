"""Exercise bounded-run termination without models, containers, or GPU calls."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('deepswe_window_lifecycle',
    Path(__file__).parents[1]/'scripts/benchmark_deepswe_first16.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('archive', [
    {}, {'session_checkpoint_deferrals':None}, {'session_checkpoint_deferrals':True},
    {'session_checkpoint_deferrals':1},
    {'session_checkpoint_deferrals':0,'completed':[dict(checkpoint_kind='P',history_end=54677,
        checkpoint_tokens=19008,terminal_prefix={'status':'FINISHED_STOPPED'})]},
    {'session_checkpoint_deferrals':0,'completed':[dict(checkpoint_kind='M',history_end=20,
        checkpoint_tokens=19,terminal_prefix={'status':'FINISHED_LENGTH_CAPPED'})]},
])
def test_checkpoint_capacity_fallback_and_missing_evidence_cannot_pass(archive):
    with pytest.raises(RuntimeError):
        runner.validate_checkpoint_history(archive)


def test_full_terminal_history_and_ordinary_cleanup_are_accepted():
    runner.validate_checkpoint_history(dict(session_checkpoint_deferrals=0,completed=[
        dict(checkpoint_kind='M',history_end=101,checkpoint_tokens=101,
            terminal_prefix={'status':'FINISHED_STOPPED'}),
        dict(terminal_prefix={'status':'FINISHED_LENGTH_CAPPED'}),
        dict(checkpoint_kind='P',history_end=50,checkpoint_tokens=32,
            terminal_prefix={'status':'FINISHED_ABORTED'})]))


def test_preflight_release_waits_for_checkpoint_and_archive_fences(tmp_path,monkeypatch):
    now=[0.0]
    path=tmp_path/'archive.json'
    released=dict(session_checkpoint_deferrals=0,session_checkpoints=0,
        session_checkpoint_payload_bytes=0,cpu_authority_bytes=0)
    path.write_text(json.dumps(dict(released,session_checkpoint_payload_bytes=157876224)))
    def sleep(seconds):
        now[0]+=seconds
        path.write_text(json.dumps(released if now[0]>=.1 else dict(released,pending_downloads=1)))
    monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic=lambda:now[0],sleep=sleep))
    assert runner.wait_preflight_release(tmp_path)==released and now[0]==.1
    path.write_text(json.dumps(dict(released,session_checkpoints=1)))
    with pytest.raises(TimeoutError,match='retained history'):
        runner.wait_preflight_release(tmp_path,timeout_seconds=0)


def test_cold_start_can_exceed_ten_minutes_without_busy_polling(monkeypatch):
    clock=[0.0]; polls=[]
    def sleep(seconds):clock[0]+=seconds
    monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=sleep))
    class Response(io.BytesIO):
        @property
        def status(self):return 200 if clock[0]>=900 else 503
    class HTTP:
        def open(self,address,**kwargs):
            polls.append(clock[0]);return Response()
    process=SimpleNamespace(poll=lambda:None)
    runner.wait_backend_ready(HTTP(),'http://local',process,process)
    assert clock[0]==900 and len(polls)==901


def test_startup_is_bounded_and_dead_process_fails_immediately(monkeypatch):
    clock=[0.0]
    def sleep(seconds):clock[0]+=seconds
    monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=sleep))
    class HTTP:
        def open(self,*args,**kwargs):raise ConnectionError('starting')
    process=SimpleNamespace(poll=lambda:None)
    with pytest.raises(TimeoutError,match='readiness'):
        runner.wait_backend_ready(HTTP(),'http://local',process,process,3)
    assert clock[0]==3
    with pytest.raises(RuntimeError,match='startup failed'):
        runner.wait_backend_ready(HTTP(),'http://local',SimpleNamespace(poll=lambda:1),process)
    assert clock[0]==3


@pytest.mark.parametrize('backend,context_ceiling', [('native',False),('kvmem',False),('kvmem',True)])
@pytest.mark.parametrize('snapshot_mode', ['off','success','failure'])
def test_window_stops_live_workers_preserves_unscored_attempts_and_cleans_owned_processes(
        tmp_path,monkeypatch,backend,context_ceiling,snapshot_mode):
    root=Path(__file__).parents[1]
    # Only the original repetition helper is loaded; all external effects below
    # are controlled fakes. Neither model modules nor Docker are imported.
    monkeypatch.syspath_prepend(str(root/'benchmarks/deepswe'))
    output=tmp_path/'run';output.mkdir();(output/'setup').mkdir()
    (output/'setup/read_guard.py').write_text('class ReadGuard:\n first_trigger=None\n def line(self,x):pass\n')
    config=dict(port=18520,tokenizer='/local/tokenizer',served_model_name='model',
        max_model_len=262144,max_num_seqs=8,kvmem=dict(select_budget=65536,gen_budget=32768,
            cpu_history_coordination=True,cpu_read_cache_bytes=100_000_000_000))
    path=output/'profile.json';path.write_text(json.dumps(config))
    tasks=[dict(task_id='task-'+str(i),verifier_image='owned-test-verifier') for i in range(16)]
    (output/'manifest.json').write_text(json.dumps(dict(profile=config,workers=4,
        backend_kind=backend,duration_seconds=60,tasks=tasks)))
    if context_ceiling:
        (output/'bridge').mkdir()
        (output/'bridge/requests.jsonl').write_text(json.dumps(dict(event='context_limit',task='task-0'))+'\n')
    processes=[];terminated=[];external=[]
    class Process:
        def __init__(self,args,**kwargs):
            self.pid=900000+len(processes);self.returncode=None;self.args=args
            processes.append(self)
        def poll(self):return self.returncode
        def wait(self,timeout=None):
            self.returncode=0;return 0
    def killpg(pid,sig):
        terminated.append(pid)
        next(p for p in processes if p.pid==pid).returncode=-int(sig)
    def check_output(args,**kwargs):
        external.append(args)
        assert args[:3]==['docker','network','inspect']
        return json.dumps([dict(IPAM=dict(Config=[dict(Gateway='172.31.0.1')]))])
    def command(args,**kwargs):
        external.append(args)
        assert args[0] in ('docker','nvidia-smi')
        return SimpleNamespace(returncode=0,stdout='0, 0',stderr='')
    class Response(io.BytesIO):
        status=200
    class HTTP:
        def open(self,request,**kwargs):
            address=getattr(request,'full_url',request)
            if address.endswith('/metrics'):
                return Response(b'vllm:generation_tokens_total 0\n')
            return Response(b'{"released":true}')
    started=time.monotonic()
    clock=SimpleNamespace(time=lambda:1000+(time.monotonic()-started)*100,
        monotonic=time.monotonic,sleep=lambda seconds:time.sleep(.002))
    monkeypatch.setattr(runner,'time',clock)
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.subprocess,'check_output',check_output)
    monkeypatch.setattr(runner.subprocess,'run',command)
    monkeypatch.setattr(runner.os,'killpg',killpg)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    monkeypatch.setattr(runner.urllib.request,'build_opener',lambda *args:HTTP())
    monkeypatch.setattr(runner,'bounded_warmup',lambda *args:None)
    from scripts import deepswe_window_snapshot
    snapshot_calls=[]
    def snapshot(*args):
        snapshot_calls.append('snapshot')
        if snapshot_mode=='failure':raise RuntimeError('Snapshot storage failure')
        return dict(status='preserved',supervisors=[])
    monkeypatch.setattr(deepswe_window_snapshot,'snapshot',snapshot)
    monkeypatch.setattr(deepswe_window_snapshot,'release_supervisors',lambda row:snapshot_calls.append('release'))
    # This lifecycle test uses fake services; host memory must be fake too.
    # A concurrent real benchmark must not turn it into a low-memory test.
    read_text=Path.read_text
    def host_memory(path,*args,**kwargs):
        if path==Path('/proc/meminfo'):
            return 'MemAvailable: 67108864 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n'
        return read_text(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',host_memory)
    if backend=='native':
        monkeypatch.setitem(sys.modules,'native_agent_launch',SimpleNamespace(
            build_launch=lambda *args,**kw:dict(argv=['owned-native'],environment={})))
    args=SimpleNamespace(duration_seconds=60,adopt_controller=None,use_prepared=True,
        snapshot_window_workspaces=snapshot_mode!='off',
        warmup_tokens=tmp_path/'init.i32',bash_repeat_limit=0,workers=4,backend_kind=backend,
        output=output,config=path,carry_result=[],prepare_only=False,bridge_port=18521)
    runner.run(args)
    state=json.loads((output/'run.json').read_text())
    if snapshot_mode=='failure':
        assert state['status']=='needs_attention' and state['cleanup_errors']
        assert snapshot_calls==['snapshot']
        assert not any(args[:3]==['docker','network','rm'] for args in external)
        retained={p.pid for p in processes if p.args[:2]==['bash',str(output/'setup/run_deepswe_cell.sh')] and p.poll() is None}
        assert retained and not retained & set(terminated)
        assert all(p.poll() is not None for p in processes if p.pid not in retained)
        return
    assert state['status']=='window_complete' and state['cleanup_errors']==[]
    assert snapshot_calls==(['snapshot','release'] if snapshot_mode=='success' else [])
    assert state['cleanup_finished_at']>=state['ended_at']>=state['window_deadline']
    assert state['tasks'] and len(state['tasks'])<16
    assert all(t['scored'] is False and 'result' not in t for t in state['tasks'].values())
    assert all(t['state'] in ('window_truncated','context_ceiling') for t in state['tasks'].values())
    if context_ceiling:assert state['tasks']['task-0']['state']=='context_ceiling'
    assert set(terminated)=={p.pid for p in processes}
    assert all(p.poll() is not None for p in processes)
    assert any(args[:3]==['docker','network','rm'] for args in external)


def test_context_log_preserves_partial_record(tmp_path):
    path=tmp_path/'requests.jsonl'; limited=set()
    assert runner.read_context_limits(path,0,limited)==0
    first=json.dumps(dict(event='context_limit',task='a'))+'\n'
    second=json.dumps(dict(event='context_limit',task='b'))
    path.write_text(first+second[:20])
    cursor=runner.read_context_limits(path,0,limited)
    assert limited=={'a'} and cursor==len(first)
    with path.open('a') as stream:stream.write(second[20:]+'\n')
    assert runner.read_context_limits(path,cursor,limited)==path.stat().st_size
    assert limited=={'a','b'}


def local_limit_record(model='model', context=196608):
    return dict(type='result', subtype='success', is_error=True,
        terminal_reason='blocking_limit', result='Prompt is too long', api_error_status=None,
        session_id='owned-session', modelUsage={model:dict(contextWindow=context,maxOutputTokens=32000)})


def test_cli_local_limit_requires_terminal_structured_profile_bound_evidence(tmp_path):
    path=tmp_path/'claude.stream.jsonl'
    assert runner.read_client_context_limit(path,'model',196608) is None
    row=local_limit_record()
    terminal=json.dumps(row)+'\n'
    path.write_text(json.dumps(dict(type='assistant',message='ordinary tool call'))+'\n'+terminal)
    proof=runner.read_client_context_limit(path,'model',196608)
    assert proof['source']=='client_local_guard' and proof['line']==2
    assert proof['record_sha256']==runner.hashlib.sha256(terminal.encode()).hexdigest()
    assert proof['context_tokens']==196608 and proof['reported_max_output_tokens']==32000
    assert runner.read_client_context_limit(path,'other-model',196608) is None
    assert runner.read_client_context_limit(path,'model',307200) is None


@pytest.mark.parametrize('changes',[
    dict(type='assistant'),dict(type='user'),dict(is_error=False),dict(is_error=1),
    dict(terminal_reason='api_error'),dict(terminal_reason=None),
    dict(result='Connection failed'),dict(result='Tool said: Prompt is too long'),
    dict(api_error_status=500),dict(modelUsage=None),dict(modelUsage={'model':None}),
    dict(modelUsage={'model':dict(contextWindow='196608')}),
])
def test_cli_local_limit_does_not_hide_other_errors(tmp_path,changes):
    path=tmp_path/'claude.stream.jsonl'
    path.write_text(json.dumps(dict(local_limit_record(),**changes))+'\n')
    assert runner.read_client_context_limit(path,'model',196608) is None


@pytest.mark.parametrize('before,after',[
    ('','partial'),('broken\n',''),('','{}\n'),('','[]\n'),('','{"type":"assistant"}\n'),
])
def test_cli_local_limit_rejects_corrupt_or_nonterminal_result(tmp_path,before,after):
    path=tmp_path/'claude.stream.jsonl'
    path.write_text(before+json.dumps(local_limit_record())+'\n'+after)
    assert runner.read_client_context_limit(path,'model',196608) is None


def test_formal_baseline_waits_for_native_drain(tmp_path,monkeypatch):
    bodies=[b'vllm:generation_tokens_total 1\n',
        b'vllm:num_requests_running{engine="0"} 1\nvllm:num_requests_waiting{engine="0"} 0\n',
        b'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
        b'vllm:generation_tokens_total 262144\n']
    final=bodies[-1]; waits=[]
    class HTTP:
        def open(self,address,**kwargs):return io.BytesIO(bodies.pop(0))
    monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic=lambda:0,sleep=waits.append))
    runner.save_idle_metrics(dict(port=18520),tmp_path,HTTP())
    assert waits==[.25,.25] and (tmp_path/'metrics-before-formal.txt').read_bytes()==final


@pytest.mark.parametrize('metrics_failure',[False,True])
@pytest.mark.parametrize('low_memory',[False,True])
@pytest.mark.parametrize('history_failure',[False,True])
@pytest.mark.parametrize('limit_source',['bridge','client'])
@pytest.mark.parametrize('transport_failure',[False,True])
def test_full_task_context_ceiling_after_cli_exit_is_not_infrastructure_failure(
        tmp_path,monkeypatch,metrics_failure,low_memory,history_failure,limit_source,transport_failure):
    """Full runs continue to later tasks even if a CLI exits before monitor reads its limit."""
    root=Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root/'benchmarks/deepswe'))
    output=tmp_path/'run';output.mkdir();(output/'setup').mkdir();(output/'bridge').mkdir()
    if history_failure:
        capture=output/'server/capture';capture.mkdir(parents=True)
        (capture/'archive.json').write_text(json.dumps(dict(session_checkpoint_deferrals=1)))
    (output/'setup/read_guard.py').write_text('class ReadGuard:\n first_trigger=None\n def line(self,x):pass\n')
    config=dict(port=18520,tokenizer='/local/tokenizer',served_model_name='model',
        max_model_len=196608,max_num_seqs=8,kvmem=dict(select_budget=65536,gen_budget=32768,
            cpu_history_coordination=True,cpu_read_cache_bytes=100_000_000_000))
    path=output/'profile.json';path.write_text(json.dumps(config))
    tasks=[dict(task_id='task-'+str(i),verifier_image='owned-test-verifier') for i in range(16)]
    (output/'manifest.json').write_text(json.dumps(dict(profile=config,workers=8,
        backend_kind='kvmem',duration_seconds=0,tasks=tasks)))
    (output/'run.json').write_text(json.dumps(dict(id='owned-run',network='owned-network',
        gateway='172.31.0.1',tasks={},status='running')))
    processes=[];signals=[];release_requests=[]
    class Process:
        def __init__(self,args,**kwargs):
            self.pid=900000+len(processes); self.returncode=None
            processes.append(self)
            if args[0]=='bash':
                # Completion races the periodic monitor; context record remains
                # the authoritative reason, regardless of the CLI exit code.
                if limit_source=='bridge':
                    with (output/'bridge/requests.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(event='context_limit',task=args[2]))+'\n')
                else:
                    cell=Path(args[3])
                    (cell/'claude.stream.jsonl').write_text(json.dumps(local_limit_record())+'\n')
                    (cell/'cell_summary.json').write_text(json.dumps(dict(agent_exit=1,
                        verifier_exit=0,reward=dict(reward=0))))
                    if transport_failure:
                        with (output/'bridge/requests.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(event='error',task=args[2]))+'\n')
                self.returncode=1
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=0;return 0
    server=Process(['server']);bridge=Process(['bridge'])
    def killpg(pid,sig):
        signals.append(pid)
        next(p for p in processes if p.pid==pid).returncode=-int(sig)
    class HTTP:
        def open(self,request,**kwargs):
            address=getattr(request,'full_url',request)
            if address.endswith('/metrics'):
                if metrics_failure:raise TimeoutError('Metrics unavailable')
                return io.BytesIO(b'vllm:generation_tokens_total 0\n')
            release_requests.append(address)
            return io.BytesIO(b'{"released":true}')
    monkeypatch.setitem(sys.modules,'supervision',SimpleNamespace(adopt=lambda *args:
        (dict(server=server,bridge=bridge),{},'test-token',dict(pid=123))))
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:
        SimpleNamespace(returncode=0,stdout='0, 0',stderr=''))
    monkeypatch.setattr(runner.os,'killpg',killpg)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    monkeypatch.setattr(runner.urllib.request,'build_opener',lambda *args:HTTP())
    read_text=Path.read_text
    def host_memory(path,*args,**kwargs):
        if path==Path('/proc/meminfo'):
            available=7*1024**2 if low_memory else 64*1024**2
            return f'MemAvailable: {available} kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n'
        return read_text(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',host_memory)
    args=SimpleNamespace(duration_seconds=0,adopt_controller=123,use_prepared=True,
        warmup_tokens=None,bash_repeat_limit=0,workers=8,backend_kind='kvmem',
        output=output,config=path,carry_result=[],prepare_only=False,bridge_port=18521)
    runner.run(args)
    state=json.loads((output/'run.json').read_text())
    if low_memory or history_failure:
        assert state['status']=='needs_attention'
        if low_memory:assert state['resource_error']=='Host memory headroom below 8GiB'
        if history_failure:assert 'Checkpoint capacity' in state['history_contract_error']
        assert state['cleanup_errors']==[] and all(p.poll() is not None for p in processes)
        return
    if limit_source=='client' and transport_failure:
        assert state['status']=='needs_attention'
        assert all(t['state']=='infrastructure_error' for t in state['tasks'].values())
        assert all(t['bridge_errors'] for t in state['tasks'].values())
        assert not list((output/'cases').glob('*/client-context-limit.json'))
        return
    assert state['status']=='completed' and state['cleanup_errors']==[]
    assert len(state['tasks'])==len(release_requests)==16
    assert all(t['state']=='context_ceiling' and not t['scored'] for t in state['tasks'].values())
    assert all('result' not in t for t in state['tasks'].values())
    if limit_source=='client':
        assert len(list((output/'cases').glob('*/client-context-limit.json')))==16
        assert all(t['context_limit']['source']=='client_local_guard' for t in state['tasks'].values())
    assert set(signals)=={server.pid,bridge.pid}
