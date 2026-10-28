"""The native server must receive the same tokens, with no KVMem controls."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('deepswe_native_transport',
    Path(__file__).parents[1]/'benchmarks/deepswe/claude_bridge.py')
bridge=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bridge
spec.loader.exec_module(bridge)


def test_native_retains_exact_tokens_salt_sampling_and_output_budget():
    tokens=[100,200,300]
    plan={'kvmem_query_span':[1,2],'kvmem_reselect':'force','kvmem_mandatory_spans':[]}
    frozen=copy.deepcopy(plan)
    common=dict(model='same',tokens=tokens,session='user-1',max_tokens=32000,plan=plan)
    native=bridge.backend_request(**common,backend_kind='native')
    kv=bridge.backend_request(**common,backend_kind='kvmem')
    assert not any(k.startswith('kvmem') or k=='session_id' for k in native)
    assert kv==dict(native,session_id='user-1',**plan)
    assert native['prompt'] is tokens and native['cache_salt']=='user-1'
    assert native['max_tokens']==32000 and native['seed']==1000
    assert native['temperature']==.6 and native['top_p']==.95
    assert plan==frozen
    another=bridge.backend_request(**dict(common,session='user-2'),backend_kind='native')
    assert another['cache_salt']!=native['cache_salt']


def test_invalid_backend_fails_closed():
    with pytest.raises(ValueError):
        bridge.backend_request(model='m',tokens=[1],session='s',max_tokens=1,plan={},backend_kind='typo')


@pytest.mark.parametrize('configured,requested,expected',[
    (None,49152,16384),(65536,49152,49152),(65536,8192,8192),(65536,100000,65536)])
def test_independent_output_cap_uses_one_native_request(tmp_path,monkeypatch,configured,requested,expected):
    import httpx
    from fastapi.testclient import TestClient
    sent=[];commits=[]
    class Planner:
        def __init__(self,*args,**kwargs):assert args[2]==16384
        def prepare(self,payload,state):return ([1,2,3],{}, {'tools':[]},None,None,False,None)
        def response(self,raw,tools,model,usage,finish):
            assert finish=='stop' and usage['completion_tokens']==20000
            return {'content':[]}
        def commit(self,state,prepared,payload,response,output):
            commits.append(output);state.turns+=1
    class Backend:
        async def post(self,url,*,json):
            sent.append(json)
            ids=[4]*20000 if expected>=20000 else [4]*expected
            return httpx.Response(200,request=httpx.Request('POST',url),json={
                'id':'single-request','choices':[{'text':'ok','token_ids':ids,'finish_reason':'stop'}],
                'usage':{'prompt_tokens':3,'completion_tokens':len(ids)}})
    # The capped legacy/sub-G cases still exercise the route, without claiming
    # a backend may return more accepted IDs than the supplied request limit.
    if expected<20000:
        Planner.response=lambda self,*_: {'content':[]}
    monkeypatch.setattr(bridge,'Protocol',lambda path:None)
    monkeypatch.setattr(bridge,'Planner',Planner)
    monkeypatch.setattr(bridge.httpx,'AsyncClient',lambda **kw:Backend())
    tasks=tmp_path/'tasks.txt';tasks.write_text('task\n')
    args=SimpleNamespace(gen_budget=16384,max_output_tokens=configured,protocol=None,
        tasks=tasks,output=tmp_path/'bridge',cpu_admission_gib=72,backend_kind='native',
        backend='http://backend',model='model',run_id='run',max_model_len=262144)
    with TestClient(bridge.create_app(args,None)) as client:
        response=client.post('/task/task/v1/messages',json={'model':'model','max_tokens':requested})
        assert response.status_code==200,response.text
    assert len(sent)==len(commits)==1 and sent[0]['max_tokens']==expected
    assert len(commits[0])==min(20000,expected)


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('seed', [1000,1001,1004])
def test_native_messages_and_finish_never_access_cpu_archive_or_send_session_controls(tmp_path,monkeypatch,optimized,seed):
    import httpx
    from fastapi.testclient import TestClient
    sent=[];planner_options=[]
    class Planner:
        end=9
        def __init__(self,*args,**kwargs):planner_options.append(kwargs)
        def prepare(self,payload,state):
            return ([1,2,3],{'kvmem_query_span':[1,2],'kvmem_reselect':'force'},
                    {'tools':[]},None,None,False,None)
        def response(self,*args):return {'content':[]}
        def commit(self,state,*args):state.turns+=1
    class Backend:
        async def post(self,url,*,json):
            sent.append(json)
            return httpx.Response(200,request=httpx.Request('POST',url),json={
                'id':'cmpl-owned','choices':[{'text':'ok','token_ids':[4], 'finish_reason':'stop'}],
                'usage':{'prompt_tokens':3,'completion_tokens':1}})
    monkeypatch.setattr(bridge,'Protocol',lambda path:None)
    monkeypatch.setattr(bridge,'Planner',Planner)
    monkeypatch.setattr(bridge.httpx,'AsyncClient',lambda **kw:Backend())
    monkeypatch.setenv('DEEPSWE_CONTROL_TOKEN','test-owned-control')
    tasks=tmp_path/'tasks.txt';tasks.write_text('task-1\n')
    args=SimpleNamespace(backend_kind='native',gen_budget=32768,protocol=tmp_path/'missing.so',
        tasks=tasks,output=tmp_path/'bridge',cpu_admission_gib=72,cpu_admission_policy='engine',
        cache_geometry=tmp_path/'must-not-be-read.json',backend='http://backend',
        model='model',run_id='run',max_model_len=262144,optimized_handoff=optimized,seed=seed)
    with TestClient(bridge.create_app(args,None)) as client:
        response=client.post('/task/task-1/v1/messages',json={'model':'model','max_tokens':32})
        assert response.status_code==200,response.text
        result=client.post('/control/task-1/finish',headers={'x-control-token':'test-owned-control'},json={})
        assert result.status_code==200 and result.json()['released'] is True
    assert len(sent)==1
    assert planner_options==[{'optimized_handoff':True} if optimized else {}]
    assert sent[0]['prompt']==[1,2,3]
    assert sent[0]['cache_salt']=='run-task-1'
    assert sent[0]['seed']==seed
    assert not any(k.startswith('kvmem') or k=='session_id' for k in sent[0])
