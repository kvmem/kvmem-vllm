import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import monitor_agent_handoff as monitor
import monitor_ruler_pooling_queue as base


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root,status='waiting_for_benchmark'):
    write(root/'manifest.json',dict(dependency_state=str(root/'dependency.json'),
        dependency_expected=60,dependency_controller=dict(pid=1,created=1)))
    write(root/'dependency.json',dict(phase='running'))
    write(root/'queue-process.json',dict(pid=2,created=1))
    write(root/'status.json',dict(status=status,active='baseline',at=1000))
    for name in ('baseline','candidate'):
        write(root/name/'status.json',dict(status='prepared'))
        write(root/name/'task-plan.json',dict(cases=[dict(id=str(i)) for i in range(8)]))


def check(root,**kwargs):
    return monitor.inspect(root,now=kwargs.pop('now',1001),
        probe=kwargs.pop('probe',lambda r:dict(state='alive',pid=r.get('pid'))),
        resource_probe=kwargs.pop('resource_probe',lambda _:dict(gpu=[dict(pid=1)],disk_free_bytes=100*1024**3)),**kwargs)


def finish(root):
    write(root/'status.json',dict(status='completed',at=1000))
    write(root/'qualification/execution-audit.json',dict(status='passed'))
    write(root/'qualification/cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))
    for name in ('baseline','candidate'):
        d=root/name
        write(d/'status.json',dict(status='completed',at=1000))
        for i in range(8):write(d/'cases'/str(i)/'status.json',dict(status='completed',turns_completed=4,max_logical_input_tokens=45000))
        write(d/'execution-audit.json',dict(status='passed'))
        write(d/'cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))
        write(d/'performance.json',dict(counts=dict(completed_tasks=8),accepted_output_tokens=10000,wall_seconds=100))
        write(d/'evidence-hashes.json',{})
        write(d/'browser-summary.json',dict(checked=8))
    write(root/'comparison.json',{})


def test_waiting_is_quiet_and_gpu_gap_does_not_start_anything(tmp_path):
    fixture(tmp_path)
    snap=check(tmp_path,resource_probe=lambda _:dict(gpu=[]))
    assert not snap['issues'] and not snap['gpu_released']
    assert monitor.findings(snap,{})==[]
    assert json.loads((tmp_path/'status.json').read_text())['status']=='waiting_for_benchmark'


def test_component_progress_uses_its_own_engine_and_tokens(tmp_path):
    fixture(tmp_path, 'optimization_running')
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['optimization_kind']='graph_span_merge'
    write(tmp_path/'manifest.json',manifest)
    write(tmp_path/'status.json',dict(status='optimization_running',active='before',at=1000))
    p=tmp_path/'optimization/before'
    write(p/'status.json',dict(status='testing'))
    write(p/'run/server-process.json',dict(pid=3,created=1))
    write(p/'run/server/capture/public_work.json',dict(pid=4,counter_id='new',
        completed=dict(public_batches=5,public_prefill_tokens=70000,public_non_prefill_tokens=10)))
    snap=check(tmp_path,resource_probe=lambda _:dict(gpu=[dict(pid=4)]))
    assert not snap['issues']
    assert snap['optimization']['before']['work']['public_prefill_tokens']==70000
    assert snap['gpu'][0]['matches_active_engine'] is True
    assert snap['complete'] is False


def test_component_queue_cannot_claim_missing_audit_as_success(tmp_path):
    fixture(tmp_path,'needs_optimization_review')
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['optimization_kind']='graph_span_merge'
    write(tmp_path/'manifest.json',manifest)
    snap=check(tmp_path)
    assert any('component audit' in issue for issue in snap['issues'])
    assert not snap['complete']


def test_complete_predecessor_and_released_gpu_notifies_once(tmp_path):
    fixture(tmp_path)
    write(tmp_path/'dependency.json',dict(phase='complete',summary=dict(all=dict(count=60))))
    snap=check(tmp_path,resource_probe=lambda _:dict(gpu=[]),
        probe=lambda r:dict(state='missing_or_exited' if r.get('pid')==1 else 'alive'))
    assert snap['gpu_released']
    assert len(monitor.findings(snap,{}))==1
    assert monitor.findings(snap,snap)==[]


def test_visibility_denied_cannot_be_mistaken_for_gpu_release(tmp_path):
    fixture(tmp_path)
    write(tmp_path/'dependency.json',dict(phase='complete',summary=dict(all=dict(count=60))))
    snap=check(tmp_path,resource_probe=lambda _:dict(gpu=[],gpu_inspection_error='AccessDenied'))
    assert not snap['gpu_released'] and snap['issues']


def test_forward_progress_without_completed_task_is_healthy(tmp_path):
    fixture(tmp_path,'running')
    work=tmp_path/'baseline/server/capture/public_work.json'
    write(work,dict(counter_id='x',completed=dict(public_batches=1)))
    old=check(tmp_path,now=1000)
    write(work,dict(counter_id='x',completed=dict(public_batches=20)))
    snap=check(tmp_path,now=2500,previous=old)
    assert not snap['issues']
    stalled=check(tmp_path,now=3700,previous=snap)
    assert len(stalled['issues'])==1 and 'never kill' in stalled['issues'][0]


def test_ab_completion_continues_optimization_until_review_and_cleanup(tmp_path):
    fixture(tmp_path);finish(tmp_path)
    snap=check(tmp_path)
    assert snap['tests_complete'] and not snap['complete']
    assert any('continue authorized optimization' in x for x in monitor.findings(snap,snap))
    report=tmp_path/'final-report.json';write(report,dict(status='reviewed'))
    decision=dict(status='completed',root=str(tmp_path),optimization_review_complete=True,report=str(report))
    assert check(tmp_path,completion=decision)['complete']
    write(tmp_path/'candidate/cleanup.json',dict(remaining_processes=[99],remaining_owned_gpu_pids=[]))
    assert not check(tmp_path,completion=decision)['complete']


@pytest.mark.parametrize('damage',['context','turns','count','audit','browser','performance'])
def test_completion_requires_all_task_and_evidence_checks(tmp_path,damage):
    fixture(tmp_path);finish(tmp_path)
    d=tmp_path/'candidate'
    if damage in ('context','turns'):
        write(d/'cases/0/status.json',dict(status='completed',turns_completed=3 if damage=='turns' else 4,
            max_logical_input_tokens=36864 if damage=='context' else 45000))
    if damage=='count':write(d/'cases/0/status.json',dict(status='failed'))
    if damage=='audit':write(d/'execution-audit.json',dict(status='failed'))
    if damage=='browser':(d/'browser-summary.json').unlink()
    if damage=='performance':write(d/'performance.json',{})
    assert not check(tmp_path)['tests_complete']


def test_failed_arm_preserves_attempts_and_wakes_same_conversation(tmp_path):
    fixture(tmp_path,'failed')
    write(tmp_path/'baseline/cases/0/failure.json',dict(error='CUDA'))
    snap=check(tmp_path)
    assert len(snap['issues'])==2
    assert (tmp_path/'baseline/cases/0/failure.json').exists()


def acknowledged_limit(root):
    fixture(root); finish(root)
    write(root/'status.json', dict(status='running', active='candidate', at=1000))
    path=root/'baseline/task-plan.json';tasks=json.loads(path.read_text())
    tasks['cases'][0]['id']='r01-fps';write(path,tasks)
    write(root/'baseline/status.json',dict(status='testing'))
    write(root/'baseline/cases/r01-fps/status.json',dict(status='running',turns_completed=3,max_logical_input_tokens=400000))
    write(root/'baseline/cases/r01-fps/failure.json',dict(error='Predeclared turn resource limit'))
    write(root/'baseline/execution-audit.json',dict(status='failed'))
    from scripts.run_agent_handoff_queue import sha
    path=root/'baseline-outcome.json'
    write(path,dict(status='incomplete_predeclared_limit',source=str(root),failed_cases=['r01-fps'],
        retry_allowed=False,hashes={rel:sha(root/rel) for rel in ['baseline/cleanup.json','baseline/execution-audit.json']}))
    m=json.loads((root/'manifest.json').read_text());m['preserved_baseline']=dict(proof=path.name,sha256=sha(path))
    write(root/'manifest.json',m)


def test_acknowledged_limit_is_visible_but_not_a_recovery_loop_or_success(tmp_path):
    acknowledged_limit(tmp_path)
    snap=check(tmp_path)
    assert not snap['issues'] and not snap['tests_complete']
    assert snap['arms']['baseline']['completed']==7
    assert snap['arms']['baseline']['cases'][0]['status']=='resource_limit'
    assert not snap['arms']['baseline']['audit_and_cleanup_passed']
    write(tmp_path/'status.json',dict(status='needs_optimization_review',at=1000))
    ended=check(tmp_path,probe=lambda _:dict(state='missing_or_exited'))
    assert not ended['issues'] and not ended['complete']
    assert any('not a fully passed' in x for x in monitor.findings(ended,snap))


def test_changed_limit_proof_or_new_failure_still_wakes_monitor(tmp_path):
    acknowledged_limit(tmp_path)
    write(tmp_path/'baseline/cases/1/failure.json',dict(error='new failure'))
    assert check(tmp_path)['issues']
    write(tmp_path/'baseline/execution-audit.json',dict(status='passed'))
    issues=check(tmp_path)['issues']
    assert any('Invalid baseline outcome proof' in x for x in issues)


@pytest.mark.parametrize('existing_prefix,should_add',[(monitor.PREFIX,False),('【KVMem RULER 定时巡检】',True)])
def test_agent_monitor_deduplicates_its_own_messages_only(monkeypatch,existing_prefix,should_add):
    class Socket:
        def __init__(self):self.sent=[]
        async def __aenter__(self):return self
        async def __aexit__(self,*_):pass
        async def send(self,s):self.sent.append(json.loads(s))
        async def recv(self):
            last=self.sent[-1];method=last['method'];result={}
            if method=='thread/queue/list':result=dict(data=[dict(id='old',input=[dict(text=existing_prefix+'existing')])])
            if method=='thread/queue/add':result=dict(queuedSubmission=dict(id='new'))
            if method=='thread/read':result=dict(thread=dict(status=dict(type='active')))
            return json.dumps(dict(id=last['id'],result=result))
    socket=Socket()
    monkeypatch.setitem(sys.modules,'websockets',SimpleNamespace(unix_connect=lambda *a,**kw:socket))
    receipt=asyncio.run(base.wake('same-thread','/unused',monitor.PREFIX+'new','stable',message_prefix=monitor.PREFIX))
    assert not receipt['started']
    assert ('thread/queue/add' in [s['method'] for s in socket.sent])==should_add
    assert all(s['params']['threadId']=='same-thread' for s in socket.sent if s['method'].startswith('thread/'))
