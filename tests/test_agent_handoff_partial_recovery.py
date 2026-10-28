import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.recover_agent_handoff_partial import case_resume
from scripts import run_dsh_daily_suite as suite


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);suite.save(path,value)


def events():
    return [dict(seq=1,type='tool/call',data=dict(callId='one',name='write',arguments={})),
        dict(seq=2,type='tool/result',data=dict(message=dict(content=[dict(toolCallId='one',content=[])]))),
        dict(seq=3,type='turn/end',data=dict(reason=dict(kind='completed'))),
        dict(seq=4,type='turn/end',data=dict(reason=dict(kind='error',error=dict(code='TRANSPORT'))))]


def fixture(case):
    e=events()
    for i,index in enumerate((2,3)):
        write(case/f'turn-{i+1:02d}/result.json',dict(terminal=e[index],
            statistics=dict(model_steps=10),elapsed_seconds=30))
    (case/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in e))


def test_partial_recovery_identifies_only_last_failed_turn(tmp_path):
    fixture(tmp_path)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    d=case_resume(tmp_path)
    assert d['successful_turns']==1 and d['terminal_seq']==4
    assert d['prior_steps']==10 and d['prior_seconds']==30
    assert all(Path(p).read_bytes()==v for p,v in before.items())


@pytest.mark.parametrize('kind',['unsettled_tool','non_transport','success'])
def test_unsafe_or_completed_round_cannot_be_retried(tmp_path,kind):
    fixture(tmp_path)
    if kind=='unsettled_tool':
        with (tmp_path/'events.jsonl').open('a') as f:f.write(json.dumps(dict(type='tool/call',seq=5,data=dict(callId='ambiguous')))+'\n')
    else:
        p=tmp_path/'turn-02/result.json';d=json.loads(p.read_text())
        d['terminal']['data']['reason']=dict(kind='completed') if kind=='success' else dict(kind='error',error=dict(code='QUALITY'))
        write(p,d)
    with pytest.raises(ValueError):case_resume(tmp_path)


def test_resume_wakes_failed_step_without_resending_user_or_recreating_session(tmp_path,monkeypatch):
    case=tmp_path/'cases/test';workspace=tmp_path/'project';workspace.mkdir();(workspace/'index.html').write_text('preserved')
    write(case/'status.json',dict(status='running',turns_completed=4,session_id='same'))
    write(case/'selection.json',dict(current=dict(provider=suite.PROVIDER)))
    (case/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events()))
    calls=[]
    def rpc(url,method,body):
        calls.append(method);assert method=='session.models'
        return dict(routable=True,current=dict(provider=suite.PROVIDER))
    monkeypatch.setattr(suite,'rpc',rpc)
    histories=[([],{}),([dict(seq=5,type='turn/end',data=dict(reason=dict(kind='completed')))],{})]
    monkeypatch.setattr(suite,'history_since',lambda *args:histories.pop(0))
    monkeypatch.setattr(suite,'bridge_observations',lambda *args:dict(max_logical_input_tokens=420000))
    posted=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b'{"accepted":true,"appended_user_messages":0}'
    def open(req,timeout):
        posted.append(json.loads(req.data));assert req.full_url.endswith('/kvmem-resume-failed');return Response()
    monkeypatch.setattr(suite.urllib.request,'build_opener',lambda *args:SimpleNamespace(open=open))
    row=dict(id='test',workspace=str(workspace),session_id='same',task='fps',prompts=['a','b','c','fixed original prompt'])
    plan=dict(max_turns=4,minimum_turns=4,turn_timeout_seconds=3600,max_steps_per_turn=240,
        resume_cases=dict(test=dict(successful_turns=3,terminal_seq=4,prior_steps=10,prior_seconds=30)))
    result=suite.run_case(tmp_path,'http://local',row,SimpleNamespace(wait=lambda **kw:None),plan)
    assert result['status']=='completed' and result['turns_completed']==4
    assert calls==['session.models']
    assert posted==[dict(session_id='same',terminal_seq=4)]
    assert (workspace/'index.html').read_text()=='preserved'
    assert json.loads((case/'turn-04/request.json').read_text())['content'][0]['text']=='fixed original prompt'
