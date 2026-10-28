"""DSH wire conversion preserves tools and isolates auxiliary model requests."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import httpx
from starlette.requests import Request

from benchmarks.deepswe.claude_bridge import Planner, Protocol
from scripts.dsh_bridge import AcceptedTextDecoder, QwenStream, create_app, restore_trajectory, session_owner, to_anthropic, to_openai


def test_tool_call_and_result_keep_identity_and_unicode_arguments():
    request=dict(model='local',messages=[
        dict(role='system',content='system'),dict(role='user',content='read'),
        dict(role='assistant',content=None,reasoning_content='reason',tool_calls=[
            dict(id='call_1',type='function',function=dict(name='read',arguments=json.dumps({'path':'中文.txt'})))]),
        dict(role='tool',tool_call_id='call_1',content='KVMEM-random-marker')],
        tools=[dict(type='function',function=dict(name='read',parameters=dict(type='object')))])
    payload=to_anthropic(request)
    assert payload['system']=='system'
    assert payload['messages'][1]['content'][-1]==dict(type='tool_use',id='call_1',name='read',input={'path':'中文.txt'})
    assert payload['messages'][2]['content']==[dict(type='tool_result',tool_use_id='call_1',content='KVMEM-random-marker')]
    result=to_openai(dict(model='local',content=payload['messages'][1]['content'],
        stop_reason='tool_use',usage=dict(input_tokens=100,output_tokens=10)))
    message=result['choices'][0]['message']
    assert message['tool_calls'][0]['id']=='call_1'
    assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'path':'中文.txt'}
    assert message['reasoning_content']=='reason'
    assert result['choices'][0]['finish_reason']=='tool_calls'
    assert result['usage']==dict(prompt_tokens=100,completion_tokens=10,total_tokens=110)


def test_title_calls_cannot_replace_agent_history_or_graph_namespace():
    agent=dict(system='agent',tools=[dict(name='read')],messages=['first'])
    title=dict(system='title-generator',tools=[],messages=['title request'])
    assert session_owner('same-dsh-session',agent)!=session_owner('same-dsh-session',title)
    assert session_owner('same-dsh-session',agent)==session_owner('same-dsh-session',{**agent,'messages':['next turn']})
    assert session_owner('same-dsh-session',agent)!=session_owner('other-dsh-session',agent)


def test_unsupported_image_is_rejected_instead_of_dropped():
    with pytest.raises(ValueError,match='text only'):
        to_anthropic(dict(model='local',messages=[dict(role='user',content=[dict(type='image_url',image_url='x')])]))


@pytest.fixture
def protocol():
    library=Path(__file__).resolve().parents[1]/'artifacts/deepswe-first16-20261015-r16/protocol/protocol.so'
    if not library.exists():pytest.skip('Frozen Qwen protocol fixture is not installed')
    return Protocol(library)


def parsed_response(protocol,raw,tools,finish='stop'):
    planner=object.__new__(Planner);planner.protocol=protocol
    return planner.response(raw,tools,'local',dict(prompt_tokens=2,completion_tokens=3),finish)


def assemble(deltas):
    message=dict(content='',reasoning_content='',tool_calls=[])
    for delta in deltas:
        for field in ('content','reasoning_content'):message[field]+=delta.get(field,'')
        for call in delta.get('tool_calls',[]):
            i=call['index']
            while len(message['tool_calls'])<=i:
                message['tool_calls'].append(dict(function=dict(arguments='')))
            item=message['tool_calls'][i]
            for field in ('id','type'):
                if field in call:item[field]=call[field]
            function=call['function']
            if 'name' in function:item['function']['name']=function['name']
            item['function']['arguments']+=function.get('arguments','')
    return message


@pytest.mark.parametrize('width',[1,2,5,17,1000])
@pytest.mark.parametrize('raw',[
    '先分析一下\n换行\n</think>\n\r\n正文中文\n<|im_end|>',
    'unclosed reasoning\n\n',
    'reason\n\n</think>\n\nbody\n<tool_call>literal invalid call<|im_end|>',
    '</think>\n\nno reasoning<|endoftext|>',
])
def test_split_tags_unicode_and_trailing_newlines_match_frozen_parser(protocol,width,raw):
    stream=QwenStream(protocol,[]);deltas=[]
    for offset in range(0,len(raw),width):deltas.append(stream.feed(raw[offset:offset+width]))
    output,tail=stream.finish(parsed_response(protocol,raw,[]));deltas.append(tail)
    got=assemble(deltas);expected=output['choices'][0]['message']
    assert got['content']==(expected['content'] or '')
    assert got['reasoning_content']==expected.get('reasoning_content','')
    assert got['tool_calls']==[]


@pytest.mark.parametrize('width',[1,3,19,1000])
def test_tool_header_streams_before_arguments_and_final_ids_match(protocol,width):
    tools=[dict(type='function',function=dict(name='write',parameters=dict(type='object',properties={
        'path':dict(type='string'),'content':dict(type='string'),'count':dict(type='integer')})))]
    raw=('</think>\n<tool_call>\n<function=write>\n<parameter=path>中文.txt</parameter>\n'
         '<parameter=content>line1\nline2 "quote" \\slash</parameter>\n'
         '<parameter=count>3</parameter>\n</function>\n</tool_call><|im_end|>')
    stream=QwenStream(protocol,tools);deltas=[];announced_at=None
    for offset in range(0,len(raw),width):
        delta=stream.feed(raw[offset:offset+width]);deltas.append(delta)
        if delta.get('tool_calls'):announced_at=offset+width
    if width<len(raw):assert announced_at<raw.index('<parameter=content>')
    response=parsed_response(protocol,raw,tools)
    output,tail=stream.finish(response);deltas.append(tail)
    assert assemble(deltas)['tool_calls']==output['choices'][0]['message']['tool_calls']
    assert response['content'][0]['id']==stream.tool['id']
    assert json.loads(assemble(deltas)['tool_calls'][0]['function']['arguments'])['count']==3


def test_incomplete_announced_tool_fails_closed(protocol):
    tools=[dict(type='function',function=dict(name='read',parameters=dict(type='object')))]
    raw='</think><tool_call><function=read><parameter=path>unfinished'
    stream=QwenStream(protocol,tools)
    assert stream.feed(raw)['tool_calls'][0]['function']['name']=='read'
    with pytest.raises(ValueError,match='Output limit.*no tool was executed'):
        stream.finish(parsed_response(protocol,raw,tools,'length'))


def test_truncated_write_after_reasoning_preserves_length_finish_without_announcing_a_tool(protocol):
    tools=[dict(type='function',function=dict(name='write',parameters=dict(type='object')))]
    raw='I will write the file.\n</think>\n<tool_call><function=write><parameter=content>unfinished code'
    stream=QwenStream(protocol,tools);deltas=[stream.feed(c) for c in raw]
    assert stream.tool is None and not any(d.get('tool_calls') for d in deltas)
    response=parsed_response(protocol,raw,tools,'length');output,tail=stream.finish(response)
    assert output['choices'][0]['finish_reason']=='length'
    assert not output['choices'][0]['message'].get('tool_calls')
    assert assemble([*deltas,tail])['content']==output['choices'][0]['message']['content']


@pytest.mark.parametrize('failure',[None,'missing_done','bad_usage','disconnect'])
@pytest.mark.parametrize('optimized',[False,True])
def test_transport_forwards_before_completion_and_never_commits_partial_stream(protocol,tmp_path,monkeypatch,failure,optimized):
    import scripts.dsh_bridge as bridge
    async def check():
        gate=asyncio.Event();commits=[];closed=[]
        class Upstream(httpx.AsyncByteStream):
            async def __aiter__(self):
                def event(value):return ('data: '+json.dumps(value)+'\n\n').encode()
                first=dict(id='native-id',choices=[dict(index=0,text='first actual token',token_ids=[10],finish_reason=None)])
                yield b': heartbeat\n\n'+event(first)
                await gate.wait()
                last=dict(id='native-id',choices=[dict(index=0,text='</think>\nanswer<|im_end|>',token_ids=[11,12],finish_reason='stop')])
                # Split SSE lines across arbitrary network chunks.
                record=event(last)
                yield record[:13];yield record[13:]
                yield event(dict(id='native-id',choices=[],usage=dict(prompt_tokens=2,completion_tokens=4 if failure=='bad_usage' else 3,total_tokens=5),metrics=dict(time_to_first_token_ms=12)))
                if failure!='missing_done':yield b'data: [DONE]\n\n'
            async def aclose(self):closed.append(True)
        async def handle(request):
            data=json.loads(request.content)
            assert data['stream'] and data['stream_options']['include_usage']
            assert data['return_token_ids'] and data['kvmem_mandatory_parts']=='none'
            return httpx.Response(200,stream=Upstream())
        client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        class TestPlanner(Planner):
            def __init__(self,*args):self.protocol=protocol
            def prepare(self,payload,state):return ([1,2],{},dict(tools=[]),[],None,False,{})
            def commit(self,state,prepared,payload,response,output):
                commits.append(output);state.turns+=1
        monkeypatch.setattr(bridge,'Planner',TestPlanner)
        monkeypatch.setattr(bridge,'Protocol',lambda _:protocol)
        monkeypatch.setattr(bridge.httpx,'AsyncClient',lambda **_:client)
        args=SimpleNamespace(protocol=None,gen_budget=16384,output=tmp_path,backend='http://test',
            model='local',max_model_len=262144,max_sessions=4,optimized_handoff=optimized)
        app=create_app(args,None)
        endpoint=next(r.endpoint for r in app.routes if r.path=='/v1/chat/completions')
        body=json.dumps(dict(model='local',prompt_cache_key='session',stream=True,
            messages=[dict(role='user',content='test')])).encode()
        async def receive():return dict(type='http.request',body=body,more_body=False)
        request=Request(dict(type='http',method='POST',path='/v1/chat/completions',headers=[]),receive)
        response=await endpoint(request);iterator=response.body_iterator
        assert 'assistant' in await anext(iterator)
        early=await asyncio.wait_for(anext(iterator),1)
        assert 'first actual token' in early
        assert not gate.is_set() and commits==[]
        if failure=='disconnect':
            await iterator.aclose()
            assert commits==[] and closed
        else:
            gate.set();remaining=''.join([item async for item in iterator])
            if failure:
                assert 'error' in remaining and 'data: [DONE]' not in remaining
                assert commits==[] and not list(tmp_path.glob('**/accepted.json'))
            else:
                assert '[DONE]' in remaining and 'answer' in remaining
                assert commits==[[10,11,12]]
                accepted=json.loads(next(tmp_path.glob('**/accepted.json')).read_text())
                assert accepted['token_ids']==[10,11,12]
                row=json.loads((tmp_path/'requests.jsonl').read_text())
                assert row['native_chunks']==2 and row['transport']=='native_sse'
        await client.aclose()
    asyncio.run(asyncio.wait_for(check(),5))


@pytest.mark.parametrize('corrupt',[None,'missing_turn','prompt','usage'])
def test_restore_uses_actual_ids_preserves_tools_and_excludes_incomplete_attempts(tmp_path,corrupt):
    owner='owned';source=tmp_path/'old';destination=tmp_path/'new'
    class Recorder:
        def prepare(self,payload,state):
            assert payload['messages'][0]['content'][0]['text']=='read'
            return ([101,*state.output],{}, {},[],None,bool(state.output),{})
        def commit(self,state,prepared,payload,response,ids):
            state.output=ids;state.tokens=prepared[0];state.assistant=response['content'];state.turns+=1
    tool=dict(id='actual-call-id',type='function',function=dict(name='read',arguments='{"path":"中文.txt"}'))
    for turn,parent in [(1,source),(2,destination)]:
        p=parent/owner/f'{turn:05d}';p.mkdir(parents=True)
        (p/'request.json').write_text(json.dumps(dict(model='local',messages=[dict(role='user',content='read')])))
        (p/'plan.json').write_text(json.dumps(dict(prompt_tokens=1 if turn==1 else 3,exact_output_prefix_reused=turn==2)))
        response=dict(choices=[dict(message=dict(role='assistant',content=None,reasoning_content='reason',tool_calls=[tool]))],usage=dict(completion_tokens=2))
        (p/'response.json').write_text(json.dumps(response))
        (p/'accepted.json').write_text(json.dumps(dict(token_ids=[900+turn,777])))
    incomplete=destination/owner/'00003';incomplete.mkdir();(incomplete/'request.json').write_text('{}')
    if corrupt=='missing_turn':(source/owner/'00001'/'accepted.json').unlink()
    if corrupt=='prompt':(destination/owner/'00002'/'plan.json').write_text(json.dumps(dict(prompt_tokens=999,exact_output_prefix_reused=True)))
    if corrupt=='usage':(destination/owner/'00002'/'accepted.json').write_text(json.dumps(dict(token_ids=[999])))
    if corrupt:
        with pytest.raises(ValueError):restore_trajectory(Recorder(),owner,[source,destination])
    else:
        state=restore_trajectory(Recorder(),owner,[source,destination])
        assert state.turns==2 and state.output==[902,777] and state.tokens==[101,901,777]
        assert state.assistant[-1]==dict(type='tool_use',id='actual-call-id',name='read',input={'path':'中文.txt'})


class ByteTokenizer:
    backend_tokenizer=SimpleNamespace(decoder=SimpleNamespace(__getstate__=lambda:b'{"type":"ByteLevel"}'))
    def get_added_vocab(self):return {}
    def convert_ids_to_tokens(self,token):
        # Fixed ByteLevel byte-vocabulary fixture, with byte values as IDs.
        if token<=32:return chr(256+token)
        if 127<=token<=160:return chr(289+token-127)
        if token==173:return chr(323)
        return chr(token)


def test_utf8_decoder_keeps_split_characters_between_native_segments():
    decoder=AcceptedTextDecoder(ByteTokenizer())
    encoded='中文 "quoted" \\slash\n'.encode()
    assert ''.join(decoder.feed([b]) for b in encoded)+decoder.finish()=='中文 "quoted" \\slash\n'


@pytest.mark.parametrize('optimized',[False,True])
@pytest.mark.parametrize('truncated',[False,True])
def test_rolling_output_keeps_one_sampler_and_complete_tool_frame(protocol,tmp_path,monkeypatch,optimized,truncated):
    import scripts.dsh_bridge as bridge
    async def check():
        content='x'*33000
        raw='reason</think>\n<tool_call><function=write><parameter=content>'+content+'</parameter></function></tool_call><|im_end|>'
        if truncated:raw=raw[:20000]
        ids=list(raw.encode());sent=[];commits=[]
        tools=[dict(type='function',function=dict(name='write',parameters=dict(type='object',properties={'content':dict(type='string')})))]
        class Source(httpx.AsyncByteStream):
            async def __aiter__(self):
                # One stream crosses G=16K twice inside an unfinished tool argument.
                for begin in range(0,len(ids),8192):
                    part=ids[begin:begin+8192];last=begin+len(part)==len(ids)
                    item=dict(id='one-native-id',choices=[dict(index=0,text=bytes(part).decode(),token_ids=part,
                        finish_reason=('length' if truncated else 'stop') if last else None)])
                    yield ('data: '+json.dumps(item)+'\n\n').encode()
                yield ('data: '+json.dumps(dict(choices=[],usage=dict(prompt_tokens=2,completion_tokens=len(ids))))+'\n\n').encode()
                yield b'data: [DONE]\n\n'
        async def handle(request):
            sent.append(json.loads(request.content));return httpx.Response(200,stream=Source())
        native=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        class TestPlanner(Planner):
            def __init__(self,*args):self.protocol=protocol
            def prepare(self,payload,state):return ([1,2],dict(kvmem_reselect='off'),dict(tools=tools),[],None,False,{})
            def commit(self,state,prepared,payload,response,output):commits.append(output);state.turns+=1
        monkeypatch.setattr(bridge,'Planner',TestPlanner);monkeypatch.setattr(bridge,'Protocol',lambda _:protocol)
        monkeypatch.setattr(bridge.httpx,'AsyncClient',lambda **_:native)
        args=SimpleNamespace(protocol=None,gen_budget=16384,max_output_tokens=65536,
            output=tmp_path,backend='http://test',model='local',max_model_len=262144,max_sessions=4,optimized_handoff=optimized)
        app=create_app(args,ByteTokenizer());endpoint=next(r.endpoint for r in app.routes if r.path=='/v1/chat/completions')
        body=json.dumps(dict(model='local',prompt_cache_key='rolling',stream=True,max_tokens=20000 if truncated else 49152,
            seed=1004,temperature=.6,messages=[dict(role='user',content='write')])).encode()
        async def receive():return dict(type='http.request',body=body,more_body=False)
        response=await endpoint(Request(dict(type='http',method='POST',path='/v1/chat/completions',headers=[]),receive))
        wire=''.join([piece async for piece in response.body_iterator])
        assert 'data: [DONE]' in wire and len(sent)==1 and commits==[ids]
        assert sent[0]['max_tokens']==(20000 if truncated else 49152)
        assert sent[0]['prompt']==[1,2] and sent[0]['seed']==1004 and sent[0]['thinking_token_budget']==4096
        chunks=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        message=assemble([c['choices'][0]['delta'] for c in chunks])
        if truncated:
            assert not message['tool_calls'] and chunks[-1]['choices'][0]['finish_reason']=='length'
        else:
            assert len(message['tool_calls'])==1
            assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'content':content}
        assert chunks[-1]['usage']['completion_tokens']==len(ids)
        await native.aclose()
    asyncio.run(asyncio.wait_for(check(),10))


@pytest.mark.parametrize('failed_segment',[False,True,'context_cap'])
@pytest.mark.parametrize('optimized',[False,True])
def test_continuation_reuses_exact_ids_and_executes_only_one_complete_tool(protocol,tmp_path,monkeypatch,failed_segment,optimized):
    import scripts.dsh_bridge as bridge
    async def check():
        tools=[dict(type='function',function=dict(name='write',parameters=dict(type='object',properties={'content':dict(type='string')})))]
        raw='reason</think>\n<tool_call><function=write><parameter=content>中文</parameter></function></tool_call><|im_end|>'
        accepted=list(raw.encode());quota=accepted.index('中'.encode()[0])+1
        prompts=[];commits=[]
        class Source(httpx.AsyncByteStream):
            def __init__(self,ids,finish):self.ids=ids;self.finish=finish
            async def __aiter__(self):
                event=dict(id='segment-'+str(len(prompts)),choices=[dict(index=0,text='not authoritative',token_ids=self.ids,finish_reason=self.finish)])
                usage=dict(prompt_tokens=len(prompts[-1]),completion_tokens=len(self.ids),total_tokens=len(prompts[-1])+len(self.ids))
                for data in [event,dict(choices=[],usage=usage)]:yield ('data: '+json.dumps(data)+'\n\n').encode()
                yield b'data: [DONE]\n\n'
        async def handle(request):
            body=json.loads(request.content)
            used=len(body['prompt'])-2
            assert body['prompt']==[1,2]+accepted[:used]
            assert body['max_tokens']<=quota
            assert body['kvmem_query_span']==[0,1] and body['kvmem_reselect']=='auto'
            if prompts:assert 'thinking_token_budget' not in body
            prompts.append(body['prompt'])
            if failed_segment is True and len(prompts)==2:return httpx.Response(503,text='injected failure')
            ids=accepted[used:used+body['max_tokens']]
            return httpx.Response(200,stream=Source(ids,'length' if used+len(ids)<len(accepted) else 'stop'))
        client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        class TestPlanner(Planner):
            def __init__(self,*args):self.protocol=protocol
            def prepare(self,payload,state):return ([1,2],dict(kvmem_query_span=[0,1],kvmem_reselect='auto'),dict(tools=tools),[],None,False,{})
            def commit(self,state,prepared,payload,response,output):commits.append(output);state.turns+=1
        monkeypatch.setattr(bridge,'Planner',TestPlanner);monkeypatch.setattr(bridge,'Protocol',lambda _:protocol)
        monkeypatch.setattr(bridge.httpx,'AsyncClient',lambda **_:client)
        context=2+quota+1 if failed_segment=='context_cap' else 262144
        args=SimpleNamespace(protocol=None,gen_budget=quota,max_output_segments=4,output=tmp_path,backend='http://test',model='local',max_model_len=context,max_sessions=4,optimized_handoff=optimized)
        app=create_app(args,ByteTokenizer());endpoint=next(r.endpoint for r in app.routes if r.path=='/v1/chat/completions')
        body=json.dumps(dict(model='local',prompt_cache_key='continuation',stream=True,max_tokens=quota*4,messages=[dict(role='user',content='write')])).encode()
        async def receive():return dict(type='http.request',body=body,more_body=False)
        response=await endpoint(Request(dict(type='http',method='POST',path='/v1/chat/completions',headers=[]),receive))
        wire=''.join([piece async for piece in response.body_iterator])
        assert len(prompts)>=2
        if failed_segment is True:
            assert 'injected failure' in wire and not commits and 'data: [DONE]' not in wire
            assert len(list(tmp_path.glob('**/segment-*.json')))==1
        elif failed_segment=='context_cap':
            assert commits==[accepted[:quota+1]] and 'data: [DONE]' in wire
            chunks=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
            assert not assemble([c['choices'][0]['delta'] for c in chunks])['tool_calls']
            assert chunks[-1]['choices'][0]['finish_reason']=='length'
            assert chunks[-1]['usage']['completion_tokens']==quota+1
        else:
            assert commits==[accepted] and 'data: [DONE]' in wire
            chunks=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
            message=assemble([c['choices'][0]['delta'] for c in chunks])
            assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'content':'中文'}
            assert chunks[-1]['usage']['prompt_tokens']==2 and chunks[-1]['usage']['completion_tokens']==len(accepted)
            assert len(list(tmp_path.glob('**/segment-*.json')))==len(prompts)
        await client.aclose()
    asyncio.run(asyncio.wait_for(check(),5))
