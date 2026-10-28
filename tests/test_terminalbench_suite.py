import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from scripts.run_terminalbench_job import classify
from scripts.terminalbench_view import State


def test_image_downloads_strip_proxy_settings_without_changing_other_environment():
    from scripts.run_terminalbench_suite import direct_environment
    original={'HTTP_PROXY':'secret', 'https_proxy':'secret', 'ALL_PROXY':'secret',
              'no_proxy':'localhost', 'npm_config_proxy':'secret', 'SOCKS_PROXY':'secret',
              'PATH':'/bin', 'DOCKER_HOST':'unix:///var/run/docker.sock'}
    assert direct_environment(original)=={'PATH':'/bin','DOCKER_HOST':'unix:///var/run/docker.sock'}
    assert len(original)==8


def test_reusing_download_requires_exact_image_identity():
    from scripts.run_terminalbench_suite import reusable_image
    row={'name':'task','environment':{'docker_image':'image:tag'}}
    proof={'task':'task','image':'image:tag','id':'sha256:original'}
    assert reusable_image(row,proof,{'Id':'sha256:original'})
    assert not reusable_image(row,proof,{'Id':'sha256:different'})
    assert not reusable_image(row,{**proof,'task':'other'},{'Id':'sha256:original'})
    assert not reusable_image(row,{**proof,'image':'other:tag'},{'Id':'sha256:original'})


@pytest.mark.parametrize('bad_seed',[False,True])
@pytest.mark.parametrize('namespace',['','terminal-bench/'])
@pytest.mark.parametrize('agent_time_limit',['official','none'])
def test_final_review_checks_actual_seed_and_preserves_failed_task_results(tmp_path,monkeypatch,bad_seed,namespace,agent_time_limit):
    from scripts import finalize_terminalbench as final
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def lines(name,values):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(v)+'\n' for v in values))
    names=['task-'+str(i) for i in range(89)]
    outcomes=[dict(task=namespace+n,trial=n+'-trial',status='scored',passed=False,
                   rewards={'reward':0},exception=None) for n in names]
    save('config-proof.json',dict(seed=1000))
    save('runtime/source-proof.json',dict(source='frozen'))
    save('manifest.json',dict(frozen={'runtime':{'source-proof.json':final.sha(tmp_path/'runtime/source-proof.json')}},
        files={'config-proof.json':final.sha(tmp_path/'config-proof.json')},agent_time_limit=agent_time_limit))
    save('tasks.json',[dict(name=n) for n in names])
    if agent_time_limit=='none':
        from scripts.terminalbench_policy import configure_task_deadlines
        for name in names:
            p=tmp_path/'dataset/tasks'/name/'task.toml';p.parent.mkdir(parents=True)
            p.write_text('[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 30\n')
        save('agent-time-policy.json',configure_task_deadlines(tmp_path/'dataset/tasks','none'))
        save('job.json',dict(agents=[dict(name='terminus-2')]))
    save('formal-events/summary.json',dict(outcomes=outcomes,passed=0,scored=89,task_timeouts=0,errors=0))
    save('cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))
    for o in outcomes:save('formal-events/'+o['trial']+'.json',dict(agent_execution={},agent_result={}))
    lines('formal-events/events.jsonl',[
        *[dict(trial=o['trial'],event='kv_released',at=20) for o in outcomes],
        dict(event='agent-start',at=10),dict(event='agent-end',at=20)])
    lines('bridge/requests.jsonl',[dict(event='session_opened',owner='owner',session_id=outcomes[0]['trial']+'__agent'),
                                   dict(owner='owner',output_tokens=2)])
    plan=dict(seed=1001 if bad_seed else 1000,max_tokens=131072,thinking_token_budget=4096,
        temperature=.6,top_p=.95,top_k=20,min_p=0,repetition_penalty=1,presence_penalty=0)
    save('bridge/owner/00001/plan.json',plan)
    save('bridge/owner/00001/accepted.json',dict(token_ids=[11,12]))
    save('bridge/owner/00001/native.json',dict(token_ids=[11,12],finish_reason='stop'))
    save('bridge/owner/00001/response.json',dict(usage=dict(completion_tokens=2)))
    # Failed/incomplete requests remain visible without inventing accepted output.
    save('bridge/owner/00002/plan.json',plan)
    save('model/server/capture/archive.json',dict(historical_gc=dict(policy_failures=0)))
    lines('resources.jsonl',[dict(memory_available=100,gpu_memory_mib=80)])
    monkeypatch.setattr(final.subprocess,'check_output',lambda *args,**kwargs:'')
    if bad_seed:
        with pytest.raises(ValueError,match='Actual seed changed'):final.review(tmp_path)
        assert not (tmp_path/'final-review.json').exists()
    else:
        final.review(tmp_path)
        audit=json.loads((tmp_path/'execution-audit.json').read_text())
        assert audit['tasks']==89 and audit['attempts']==89
        assert json.loads((tmp_path/'final-review.json').read_text())['expected']==89
        assert audit['accepted_completed_response_tokens']==2
        assert audit['accepted_completed_response_tokens_per_wall_second']==.2
        assert audit['incomplete_native_requests']==['bridge/owner/00002']
        assert audit['agent_time_limit']==agent_time_limit
        assert audit['official_timing_comparable']==(agent_time_limit=='official')
        assert ('custom local evaluation' in (tmp_path/'report.txt').read_text())==(agent_time_limit=='none')
        assert json.loads((tmp_path/'final-review.json').read_text())['passed']==0


def request(path, body, control=None):
    data=json.dumps(body).encode()
    async def receive():return dict(type='http.request',body=data,more_body=False)
    headers=[] if control is None else [(b'x-control-token',control.encode())]
    return Request(dict(type='http',method='POST',path=path,headers=headers),receive)


@pytest.mark.parametrize('release_failure',[False,True])
def test_release_fences_native_work_and_frees_capacity(tmp_path,monkeypatch,release_failure):
    import scripts.dsh_bridge as bridge
    from benchmarks.deepswe.claude_bridge import Planner
    # Release ownership does not depend on the optional compiled Qwen parser.
    def protocol(**payload):
        assert payload==dict(op='parse',text='</think>ok',tools=[])
        return dict(reasoning='',content='ok',tool_calls=[])
    async def check():
        entered=asyncio.Event();gate=asyncio.Event();calls=[]
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set();await gate.wait()
                for value in [dict(id='original',choices=[dict(index=0,text='</think>ok<|im_end|>',
                    token_ids=[10,11],finish_reason='stop')]),dict(id='original',choices=[],
                    usage=dict(prompt_tokens=2,completion_tokens=2,total_tokens=4))]:
                    yield ('data: '+json.dumps(value)+'\n\n').encode()
                yield b'data: [DONE]\n\n'
        async def upstream(req):
            body=json.loads(req.content)
            if body.get('kv_transfer_params',{}).get('kvmem_release_session'):
                calls.append(body)
                return httpx.Response(500 if release_failure else 200,json=dict(id='cleanup'))
            return httpx.Response(200,stream=Stream())
        client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        class Stub(Planner):
            def __init__(self,*args):self.protocol=protocol;self.end=42
            def prepare(self,payload,state):return ([1,2],{},dict(tools=[]),[],None,False,{})
            def commit(self,state,*args):state.turns+=1
        monkeypatch.setattr(bridge,'Planner',Stub)
        monkeypatch.setattr(bridge,'Protocol',lambda _:protocol)
        monkeypatch.setattr(bridge.httpx,'AsyncClient',lambda **_:client)
        args=SimpleNamespace(protocol=None,gen_budget=16384,output=tmp_path,backend='http://mock',
            model='local',max_model_len=10000000,max_sessions=1,release_control_token='test',progress_interval=.01)
        app=bridge.create_app(args,None)
        endpoints={r.path:r.endpoint for r in app.routes}
        chat=endpoints['/v1/chat/completions'];release=endpoints['/v1/sessions/release']
        payload=dict(model='local',prompt_cache_key='old',messages=[dict(role='user',content='hi')])
        task=asyncio.create_task(chat(request('/v1/chat/completions',payload)))
        await entered.wait()
        unauthorized=await release(request('/v1/sessions/release',{'session_id':'old'}))
        assert unauthorized.status_code==401
        cleanup=asyncio.create_task(release(request('/v1/sessions/release',{'session_id':'old'},'test')))
        await asyncio.sleep(.01);assert not cleanup.done() and not calls
        gate.set();answer=await task
        assert answer['usage']['completion_tokens']==2
        if release_failure:
            with pytest.raises(httpx.HTTPStatusError):await cleanup
        else:
            assert (await cleanup)['released']
            new=await chat(request('/v1/chat/completions',{**payload,'prompt_cache_key':'new'}))
            assert new['usage']['completion_tokens']==2
        denied=await chat(request('/v1/chat/completions',payload))
        assert denied.status_code==400 and b'released' in denied.body
        assert calls[0]['prompt']==[42] and calls[0]['max_tokens']==1
        evidence=list(tmp_path.glob('*/00001/accepted.json'))
        assert evidence and all(json.loads(p.read_text())['token_ids']==[10,11] for p in evidence)
        await client.aclose()
    asyncio.run(asyncio.wait_for(check(),5))


@pytest.mark.parametrize('error,reward,status',[
    (None,{'reward':0},'scored'),(None,{'reward':1},'scored'),
    ('AgentTimeoutError',None,'agent_timeout'),('EnvironmentBuildError',None,'error'),
    ('VerifierTimeoutError',None,'error')])
def test_outcomes_do_not_turn_infrastructure_into_bad_scores(error,reward,status):
    result=classify(dict(task_name='a',trial_name='trial',verifier_result={'rewards':reward},
        exception_info={'exception_type':error} if error else None))
    assert result['status']==status
    assert result['passed']==(reward=={'reward':1})


@pytest.mark.parametrize('task_name',['one','terminal-bench/one'])
def test_live_tokens_exclude_smoke_and_do_not_double_count_commits(tmp_path,task_name):
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    save(tmp_path/'tasks.json',[{'name':'one'}]);save(tmp_path/'formal-boundary.json',{'at':1})
    events=tmp_path/'formal-events/events.jsonl';events.parent.mkdir()
    events.write_text(json.dumps(dict(at=2,event='agent-start',trial='one-x',task=task_name))+'\n')
    bridge=tmp_path/'bridge';bridge.mkdir()
    rows=[dict(event='session_opened',owner='a',session_id='one-x__agent'),
          dict(event='session_opened',owner='s',session_id='smoke-x__agent'),
          dict(owner='s',output_tokens=999,turn=1,prompt_tokens=2,at=3),
          dict(owner='a',output_tokens=20,turn=1,prompt_tokens=10,at=3)]
    (bridge/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    save(bridge/'a/progress.json',dict(turn=2,accepted=7,prompt_tokens=50))
    state=State(tmp_path);state.update();assert state.value['total_tokens']==27
    with (bridge/'requests.jsonl').open('a') as f:
        f.write(json.dumps(dict(owner='a',output_tokens=7,turn=2,prompt_tokens=50,at=4))+'\n')
    state.update();assert state.value['total_tokens']==27
    assert state.value['tasks'][0]['requests']==2
    save(tmp_path/'formal-events/status.json',dict(outcomes=[dict(task=task_name,passed=True)]))
    state.update();assert state.value['tasks'][0]['outcome']['passed']


def test_preserved_context_uses_scored_trial_without_inflating_live_throughput(tmp_path):
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def lines(path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    old=tmp_path/'old';current=tmp_path/'current'
    save(current/'tasks.json',[dict(name=n) for n in ('saved','live','queued')])
    save(current/'formal-boundary.json',dict(at=1))
    save(current/'preserved-outcomes.json',[dict(task='terminal-bench/saved',
        trial='saved-scored',source_root=str(old),status='scored',passed=False,agent_seconds=30)])
    lines(old/'bridge/requests.jsonl',[
        dict(event='session_opened',owner='correct',session_id='saved-scored__agent'),
        dict(event='session_opened',owner='wrong',session_id='saved-interrupted__agent'),
        dict(owner='correct',turn=1,prompt_tokens=100001,output_tokens=90,at=2),
        dict(owner='correct',turn=2,prompt_tokens=100123,output_tokens=30,at=3),
        dict(owner='wrong',turn=9,prompt_tokens=999999,output_tokens=999,at=4)])
    # A committed progress sample must not count its output twice.
    save(old/'bridge/correct/progress.json',dict(turn=2,prompt_tokens=100123,accepted=30))
    lines(current/'formal-events/events.jsonl',[
        dict(event='agent-start',trial='live-new',task='terminal-bench/live',at=4)])
    lines(current/'bridge/requests.jsonl',[
        dict(event='session_opened',owner='live',session_id='live-new__agent'),
        dict(owner='live',turn=1,prompt_tokens=2001,output_tokens=20,at=5)])
    save(current/'bridge/live/progress.json',dict(turn=2,prompt_tokens=5007,accepted=7))
    state=State(current)
    for _ in range(2):
        state.update();saved,live,queued=state.value['tasks']
        assert saved['context']==100123 and saved['requests']==2 and saved['tokens']==120
        assert live['context']==5007 and live['requests']==2 and live['tokens']==27
        assert 'context' not in queued
        assert state.value['total_tokens']==27 and state.value['complete']==1
