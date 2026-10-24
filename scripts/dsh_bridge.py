"""Local DSH chat transport with exact accepted-token history and no extra pins.

The DSH OpenAI-compatible provider sends its stable session as prompt_cache_key
using cacheRetention=long and supportsLongCacheRetention=true. Native prefix
caching remains disabled. The key only identifies an owned KVMem session.
"""
import argparse
import asyncio
import codecs
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory, backend_request, canonical
from scripts.dsh_query_epoch import apply_query_provenance


def to_anthropic(body):
    """Preserve message/tool content while adapting the two wire vocabularies."""
    messages=[];systems=[]
    for message in body['messages']:
        role=message['role'];content=message.get('content') or ''
        if role in ('system','developer'):
            if not isinstance(content,str):
                content='\n'.join(p['text'] for p in content if p['type']=='text')
            systems.append(content);continue
        if role=='tool':
            messages.append(dict(role='user',content=[dict(type='tool_result',
                tool_use_id=message['tool_call_id'],content=content)]));continue
        blocks=[dict(type='text',text=content)] if isinstance(content,str) and content else []
        if isinstance(content,list):
            for p in content:
                if p['type']!='text':raise ValueError('This local model route accepts text only')
                blocks.append(dict(type='text',text=p['text']))
        if role=='assistant':
            reasoning=message.get('reasoning_content') or message.get('reasoning')
            if reasoning:blocks.insert(0,dict(type='thinking',thinking=reasoning,signature='local'))
            for call in message.get('tool_calls',[]):
                f=call['function'];blocks.append(dict(type='tool_use',id=call['id'],
                    name=f['name'],input=json.loads(f['arguments'])))
        if role not in ('assistant','user'):raise ValueError('Unsupported chat role')
        messages.append(dict(role=role,content=blocks))
    tools=[]
    for tool in body.get('tools',[]):
        f=tool['function'];tools.append(dict(name=f['name'],description=f.get('description',''),
            input_schema=f.get('parameters',dict(type='object',properties={}))))
    return dict(model=body['model'],system='\n\n'.join(systems),messages=messages,tools=tools)


def to_openai(response):
    message=dict(role='assistant',content=''.join(b['text'] for b in response['content'] if b['type']=='text') or None)
    thinking=''.join(b['thinking'] for b in response['content'] if b['type']=='thinking')
    if thinking:message['reasoning_content']=thinking
    calls=[dict(id=b['id'],type='function',function=dict(name=b['name'],arguments=canonical(b['input'])))
           for b in response['content'] if b['type']=='tool_use']
    if calls:message['tool_calls']=calls
    stop=response['stop_reason']
    usage=response['usage'];prompt=usage['input_tokens'];output=usage['output_tokens']
    return dict(id='chatcmpl-'+uuid.uuid4().hex,object='chat.completion',created=int(time.time()),
        model=response['model'],choices=[dict(index=0,message=message,
            finish_reason='tool_calls' if calls else 'length' if stop=='max_tokens' else 'stop')],
        usage=dict(prompt_tokens=prompt,completion_tokens=output,total_tokens=prompt+output))


def session_owner(key,payload,namespace='dsh'):
    # DSH title generation uses the conversation's session key too. Distinct
    # system/tool frames need distinct owners so an auxiliary call cannot
    # replace the agent's accepted-token ledger or graph checkpoints.
    frame=canonical(dict(system=payload.get('system',''),tools=payload.get('tools',[])))
    return namespace+'-'+hashlib.sha256((key+'\0'+frame).encode()).hexdigest()[:40]


class AcceptedTextDecoder:
    """Qwen ByteLevel decoding, retaining UTF-8 bytes across native requests."""
    def __init__(self,tokenizer):
        decoder=json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
        if decoder.get('type')!='ByteLevel':raise ValueError('Continuation requires a verified ByteLevel tokenizer')
        values=list(range(33,127))+list(range(161,173))+list(range(174,256))
        original=list(values);extra=0
        for byte in range(256):
            if byte not in values:original.append(byte);values.append(256+extra);extra+=1
        inverse=dict(zip(map(chr,values),original));added=set(tokenizer.get_added_vocab().values())
        @lru_cache(maxsize=8192)
        def token_bytes(token):
            piece=tokenizer.convert_ids_to_tokens(token)
            return piece.encode() if token in added else bytes(inverse[c] for c in piece)
        self.token_bytes=token_bytes;self.decoder=codecs.getincrementaldecoder('utf-8')('replace')

    def feed(self,ids):
        return self.decoder.decode(b''.join(self.token_bytes(i) for i in ids),final=False)

    def finish(self):return self.decoder.decode(b'',final=True)


class QwenStream:
    """Emit only irrevocable prefixes of the frozen full-response parser.

    The renderer already opened <think>. Hold split control tags and the one
    trailing reasoning newline that the parser removes at </think>. Tool
    arguments stay buffered for the authoritative schema-aware parser, but a
    canonical function header can report real model activity before its body.
    A malformed announced call fails closed instead of executing partial JSON.
    """
    markers=('</think>','<tool_call>','<|im_end|>','<|endoftext|>')

    def __init__(self,protocol,tools, *, cache_parse=False):
        self.protocol=protocol;self.tools=tools;self.raw=''
        self.sent={'reasoning_content':'','content':''};self.tool=None
        self.cache_parse=cache_parse;self._parse_input=None;self._parsed=None

    def _delta(self,message):
        delta={}
        for field,previous in self.sent.items():
            current=message.get(field) or ''
            if not current.startswith(previous):
                raise ValueError('Streamed content differs from the authoritative Qwen parser')
            if len(current)>len(previous):delta[field]=current[len(previous):]
            self.sent[field]=current
        return delta

    def feed(self,text):
        self.raw+=text
        safe=self.raw
        # Do not leak a tag which can become a complete control token next.
        held=max((n for marker in self.markers for n in range(1,len(marker))
                  if safe.endswith(marker[:n])),default=0)
        if held:safe=safe[:-held]
        close=safe.find('</think>')
        tool_start=safe.find('<tool_call>',close+len('</think>')) if close>=0 else -1
        parse_text=safe[:tool_start] if tool_start>=0 else safe
        for ending in ('<|endoftext|>','<|im_end|>'):
            parse_text=parse_text.removesuffix(ending)
        if self.cache_parse and parse_text == self._parse_input:
            parsed=self._parsed
        else:
            parsed=self.protocol(op='parse',text=parse_text,tools=self.tools)
            if self.cache_parse:self._parse_input,self._parsed=parse_text,parsed
        reasoning=parsed['reasoning']
        if close<0:reasoning=reasoning.removesuffix('\n')
        delta=self._delta(dict(reasoning_content=reasoning,content=parsed['content']))
        # Once visible reasoning/text has established TTFT, keep tool frames
        # provisional until full parsing. Otherwise a length-truncated write
        # would leave an announced call which cannot be retracted from SSE.
        if tool_start>=0 and self.tool is None and not any(self.sent.values()):
            header=re.match(r'<tool_call>\s*<function=([^>\n]+)>',safe[tool_start:])
            if header and any(t.get('function',t).get('name')==header[1] for t in self.tools):
                self.tool=dict(id='toolu_'+uuid.uuid4().hex,name=header[1])
                delta['tool_calls']=[dict(index=0,id=self.tool['id'],type='function',
                    function=dict(name=self.tool['name'],arguments=''))]
        return delta

    def finish(self,response):
        # Use the same IDs in the wire response and the accepted-token ledger.
        calls=[b for b in response['content'] if b['type']=='tool_use']
        if self.tool is not None:
            if not calls or calls[0]['name']!=self.tool['name']:
                if response['stop_reason']=='max_tokens':
                    raise ValueError(f"Output limit ({response['usage']['output_tokens']} tokens) reached before completing tool {self.tool['name']}; no tool was executed. Split large writes into smaller calls.")
                raise ValueError('Incomplete or invalid streamed Qwen tool call')
            calls[0]['id']=self.tool['id']
        output=to_openai(response);message=output['choices'][0]['message']
        delta=self._delta(message)
        if message.get('tool_calls'):
            delta['tool_calls']=[]
            for i,call in enumerate(message['tool_calls']):
                if i==0 and self.tool is not None:
                    delta['tool_calls'].append(dict(index=i,function=dict(arguments=call['function']['arguments'])))
                else:delta['tool_calls'].append(dict(index=i,**call))
        return output,delta


async def native_events(result):
    """Read SSE records without depending on HTTP/TCP chunk boundaries."""
    fields=[]
    async for line in result.aiter_lines():
        if line=='':
            if fields:
                data='\n'.join(fields);fields=[]
                if data=='[DONE]':return
                event=json.loads(data)
                if event.get('error'):raise ValueError('Model stream failed: '+str(event['error'])[:2000])
                yield event
        elif line.startswith('data:'):fields.append(line[5:].lstrip(' '))
    raise ValueError('Model stream ended without [DONE]')


def restore_trajectory(planner,owner,directories):
    """Rebuild only completed turns using their actual accepted IDs.

    This permits a bridge update without discarding active conversations or
    re-encoding past generated code. A newer directory wins for a given turn;
    an incomplete attempt has no accepted.json and is never restored.
    """
    turns={}
    for directory in directories:
        for path in (directory/owner).glob('[0-9][0-9][0-9][0-9][0-9]'):
            if (path/'accepted.json').exists() and (path/'response.json').exists():
                turns[int(path.name)]=path
    state=Trajectory()
    for turn,path in sorted(turns.items()):
        if turn!=state.turns+1:raise ValueError('Accepted history has a missing turn; refusing an inexact restore')
        body=json.loads((path/'request.json').read_text());payload=to_anthropic(body)
        prepared=planner.prepare(payload,state);saved=json.loads((path/'plan.json').read_text())
        apply_query_provenance(body, prepared)
        if len(prepared[0])!=saved['prompt_tokens'] or prepared[5]!=saved['exact_output_prefix_reused']:
            raise ValueError('Accepted history does not match its recorded prompt prefix')
        # Extra mandatory pins were removed by this deployment before submit.
        prepared[1].update(kvmem_mandatory_parts='none',kvmem_mandatory_spans=[])
        wire=json.loads((path/'response.json').read_text())
        content=to_anthropic(dict(model=body['model'],messages=[wire['choices'][0]['message']]))['messages'][0]['content']
        ids=json.loads((path/'accepted.json').read_text())['token_ids']
        if len(ids)!=wire['usage']['completion_tokens']:raise ValueError('Accepted history has inconsistent token usage')
        planner.commit(state,prepared,payload,dict(content=content),ids)
    return state


async def bounded_host_work(slots,function,*values,**kwargs):
    """Keep the caller's owner lock until even a cancelled producer is done."""
    async with slots:
        job=asyncio.create_task(asyncio.to_thread(function,*values,**kwargs))
        cancelled=False
        while True:
            try:
                result=await asyncio.shield(job)
                break
            except asyncio.CancelledError:
                cancelled=True
                if job.done():break
            except BaseException:
                if cancelled:raise asyncio.CancelledError()
                raise
        if cancelled:
            # Retrieve any producer exception while propagating cancellation.
            if not job.cancelled():job.exception()
            raise asyncio.CancelledError()
        return result


def create_app(args,tokenizer):
    app=FastAPI();planner=Planner(Protocol(args.protocol),tokenizer,args.gen_budget)
    optimized=getattr(args,'optimized_handoff',False)
    planner.optimized_handoff=optimized
    host_slots=asyncio.Semaphore(8)
    async def host_work(function,*values,**kwargs):
        # Bounded host work with a cancellation fence: never release a session
        # lock while its producer can still commit state or write a journal.
        return await bounded_host_work(host_slots,function,*values,**kwargs)
    async def record(path,value):
        def write():path.write_text(json.dumps(value,ensure_ascii=False))
        if optimized:await host_work(write)
        else:write()
    client=httpx.AsyncClient(trust_env=False,timeout=3600)
    owners={};restored=set();args.output.mkdir(parents=True,exist_ok=True)
    def log(value):
        with (args.output/'requests.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    @app.get('/health')
    async def health():
        result=await client.get(args.backend+'/health')
        return JSONResponse(dict(ok=result.status_code==200),status_code=result.status_code)
    @app.get('/v1/models')
    async def models():
        return dict(object='list',data=[dict(id=args.model,object='model',owned_by='local')])
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        body=await request.json()
        try:
            key=body.get('prompt_cache_key') or request.headers.get('x-session-id')
            if not isinstance(key,str) or not key or len(key)>1024:
                raise ValueError('A stable DSH prompt_cache_key or X-Session-ID is required')
            if body.get('model')!=args.model:raise ValueError('Unknown local model')
            limit=body.get('max_tokens',body.get('max_completion_tokens',args.gen_budget))
            segments=getattr(args,'max_output_segments',1)
            if type(limit) is not int or not 1<=limit<=args.gen_budget*segments:
                raise ValueError(f'Requested output must be between 1 and {args.gen_budget*segments} tokens')
            payload=await host_work(to_anthropic,body) if optimized else to_anthropic(body)
            owner=session_owner(key,payload,getattr(args,'owner_namespace','dsh'))
            if owner not in owners:
                if len(owners)>=args.max_sessions:raise ValueError('Local DSH session capacity exhausted; restart/release before adding owners')
                owners[owner]=(asyncio.Lock(),Trajectory())
        except (ValueError,KeyError,TypeError) as error:
            return JSONResponse(dict(error=dict(message=str(error),type='invalid_request_error')),status_code=400)
        async def generate(emit=None):
            lock,state=owners[owner]
            async with lock:
                if owner not in restored:
                    sources=[*getattr(args,'restore_from',[]),args.output]
                    state=await asyncio.to_thread(restore_trajectory,planner,owner,sources)
                    owners[owner]=(lock,state);restored.add(owner)
                    if state.turns:log(dict(at=time.time(),owner=owner,restored_turns=state.turns))
                started=time.time();prepared=await host_work(planner.prepare,payload,state)
                handoff_timings={'prepare_seconds':time.time()-started}
                tokens,plan,rendered,*_=prepared
                query_source = apply_query_provenance(body, prepared,
                    required=getattr(args, 'require_query_provenance', False))
                # Locality/query coordinates remain available for graph recall;
                # transport policy explicitly disables all semantic extra pins.
                plan['kvmem_mandatory_parts']='none';plan['kvmem_mandatory_spans']=[]
                if len(tokens)+min(limit,args.gen_budget)>args.max_model_len:
                    raise ValueError('Complete history plus output exceeds the logical context limit')
                # Reserve the same first native segment as the original route.
                # A larger logical reply cap must not discard input or reduce
                # the formerly supported input size. Later segments stop at
                # the remaining context boundary and return finish=length.
                output_limit=min(limit,args.max_model_len-len(tokens))
                data=backend_request(model=args.model,tokens=tokens,session=owner,
                    max_tokens=min(output_limit,args.gen_budget),plan=plan,backend_kind='kvmem')
                for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed'):
                    if k in body:data[k]=body[k]
                directory=args.output/owner/f'{state.turns+1:05d}'
                if directory.exists() and (directory/'request.json').exists():
                    directory.rename(directory.with_name(directory.name+'-attempt-'+uuid.uuid4().hex[:8]))
                directory.mkdir(parents=True,exist_ok=True)
                await record(directory/'request.json',body)
                await record(directory/'plan.json',{k:v for k,v in data.items() if k!='prompt'}|
                    dict(prompt_tokens=len(tokens),exact_output_prefix_reused=prepared[5],
                         query_source=query_source))
                data.update(stream=True,stream_options=dict(include_usage=True))
                wire=QwenStream(planner.protocol,rendered['tools'],cache_parse=optimized);output_ids=[];usage=None;finish=None
                decoder=AcceptedTextDecoder(tokenizer) if segments>1 else None
                metrics=None;request_id=None;first_token=None;first_emitted=None;chunks=0
                segment_reports=[]
                for segment in range(segments):
                    begin=len(output_ids);segment_usage=None;segment_metrics=None;finish=None
                    segment_data=dict(data,prompt=tokens+output_ids,max_tokens=min(args.gen_budget,output_limit-begin))
                    if segment:
                        segment_data.pop('thinking_token_budget',None)
                        # The unchanged auto Query is suppressed only after the
                        # native ledger proves this exact accepted prefix.
                    if optimized:
                        encoded=await host_work(json.dumps,segment_data,ensure_ascii=False,
                            separators=(',',':'),allow_nan=False)
                        request_kwargs=dict(content=encoded.encode('utf-8'),headers={'Content-Type':'application/json'})
                    else:request_kwargs=dict(json=segment_data)
                    if segment==0:handoff_timings['before_native_submit_seconds']=time.time()-started
                    async with client.stream('POST',args.backend+'/v1/completions',**request_kwargs) as result:
                        if result.status_code!=200:
                            raise ValueError('Model request failed: '+(await result.aread()).decode()[:2000])
                        async for native in native_events(result):
                            request_id=native.get('id',request_id)
                            if native.get('usage') is not None:segment_usage=native['usage']
                            if native.get('metrics') is not None:segment_metrics=native['metrics']
                            for choice in native.get('choices',[]):
                                if choice.get('index',0)!=0:raise ValueError('Unexpected model choice')
                                ids=choice.get('token_ids')
                                if not isinstance(ids,list):raise ValueError('Model stream omitted actual accepted token IDs')
                                if ids and first_token is None:first_token=time.time()
                                output_ids.extend(ids);chunks+=1
                                text=decoder.feed(ids) if decoder is not None else choice.get('text','')
                                if emit is None:wire.raw+=text
                                else:
                                    delta=wire.feed(text)
                                    if delta:
                                        if first_emitted is None:first_emitted=time.time()
                                        await emit(delta)
                                if choice.get('finish_reason') is not None:finish=choice['finish_reason']
                    if (segment_usage is None or finish is None or
                            segment_usage['completion_tokens']!=len(output_ids)-begin or
                            segment_usage['completion_tokens']>segment_data['max_tokens']):
                        raise ValueError('Incomplete model stream or accepted-token/usage mismatch')
                    report=dict(index=segment,id=request_id,prompt_tokens=len(segment_data['prompt']),
                        output_tokens=len(output_ids)-begin,finish_reason=finish,metrics=segment_metrics)
                    segment_reports.append(report)
                    await record(directory/f'segment-{segment:03d}.json',dict(**report,token_ids=output_ids[begin:]))
                    if finish!='length' or len(output_ids)>=output_limit:break
                    if len(output_ids)==begin:raise ValueError('Continuation made no progress')
                if decoder is not None:
                    tail_text=decoder.finish()
                    if emit is None:wire.raw+=tail_text
                    else:
                        delta=wire.feed(tail_text)
                        if delta:await emit(delta)
                usage=dict(prompt_tokens=len(tokens),completion_tokens=len(output_ids),total_tokens=len(tokens)+len(output_ids))
                metrics=segment_reports[0]['metrics'] if len(segment_reports)==1 else dict(segments=segment_reports)
                # Preserve a failed full-parser input too, so a long coding
                # response can be diagnosed without repeating GPU generation.
                finishing=time.time()
                await record(directory/'native.json',dict(id=request_id,text=wire.raw,
                    token_ids=output_ids,usage=usage,finish_reason=finish,metrics=metrics))
                response=(await host_work(planner.response,wire.raw,rendered['tools'],body['model'],usage,finish)
                    if optimized else planner.response(wire.raw,rendered['tools'],body['model'],usage,finish))
                if emit is not None:output,tail=wire.finish(response)
                else:output=to_openai(response)
                await host_work(planner.commit,state,prepared,payload,response,output_ids)
                await record(directory/'response.json',output)
                # Keep exact IDs for restart/recovery and independent streaming audits.
                await record(directory/'accepted.json',dict(token_ids=output_ids,metrics=metrics))
                handoff_timings['finalize_seconds']=time.time()-finishing
                log(dict(at=time.time(),owner=owner,turn=state.turns,request_id=request_id,
                    prompt_tokens=len(tokens),output_tokens=len(output_ids),
                    query_span=plan.get('kvmem_query_span'),mandatory_spans=[],mandatory_parts='none',
                    exact_output_prefix_reused=prepared[5],seconds=time.time()-started,
                    tool_calls=len(output['choices'][0]['message'].get('tool_calls',[])),
                    transport='native_sse',native_chunks=chunks,metrics=metrics,
                    native_segments=len(segment_reports),segment_reports=segment_reports,
                    optimized_handoff=optimized,handoff_timings=handoff_timings,
                    requested_output_cap=limit,effective_output_cap=output_limit,
                    first_native_token_seconds=None if first_token is None else first_token-started,
                    first_visible_delta_seconds=None if first_emitted is None else first_emitted-started))
                if emit is not None and tail:await emit(tail)
                return output
        if not body.get('stream'):
            try:return await generate()
            except (ValueError,httpx.HTTPError) as error:
                log(dict(at=time.time(),owner=owner,error=str(error)))
                return JSONResponse(dict(error=dict(message=str(error),type='api_error')),status_code=502)
        async def stream():
            queue=asyncio.Queue(maxsize=32)
            job=asyncio.create_task(generate(queue.put))
            base=dict(id='chatcmpl-'+uuid.uuid4().hex,created=int(time.time()),model=body['model'],object='chat.completion.chunk')
            def event(delta,finish=None,**extra):
                return 'data: '+json.dumps(dict(**base,choices=[dict(index=0,delta=delta,finish_reason=finish)],**extra))+'\n\n'
            pending=None
            try:
                yield event(dict(role='assistant'))
                while not job.done() or not queue.empty():
                    pending=asyncio.create_task(queue.get())
                    done,_=await asyncio.wait({pending,job},timeout=2,return_when=asyncio.FIRST_COMPLETED)
                    if pending.done():yield event(pending.result())
                    else:
                        pending.cancel();await asyncio.gather(pending,return_exceptions=True)
                        if not pending.cancelled():yield event(pending.result())
                        if not done:yield ': waiting for model\n\n'
                    pending=None
                result=job.result()
                yield event({},result['choices'][0]['finish_reason'],usage=result['usage'])
                yield 'data: [DONE]\n\n'
            except (ValueError,httpx.HTTPError) as error:
                log(dict(at=time.time(),owner=owner,error=str(error)))
                yield 'data: '+json.dumps(dict(error=dict(message=str(error),type='api_error')))+'\n\n'
            finally:
                if pending is not None:pending.cancel();await asyncio.gather(pending,return_exceptions=True)
                if not job.done():job.cancel()
                await asyncio.gather(job,return_exceptions=True)
        return StreamingResponse(stream(),media_type='text/event-stream')
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--backend',default='http://127.0.0.1:18846');p.add_argument('--model',default='kvmem-dsh-dflash2')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18381)
    p.add_argument('--gen-budget',type=int,default=16384);p.add_argument('--max-model-len',type=int,default=262144)
    p.add_argument('--max-sessions',type=int,default=32)
    p.add_argument('--owner-namespace',default='dsh')
    p.add_argument('--max-output-segments',type=int,choices=range(1,5),default=1)
    p.add_argument('--require-query-provenance',action='store_true')
    p.add_argument('--optimized-handoff',action='store_true')
    p.add_argument('--restore-from',type=Path,action='append',default=[]);args=p.parse_args()
    from transformers import AutoTokenizer
    import uvicorn
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    uvicorn.run(create_app(args,tokenizer),host='127.0.0.1',port=args.port)
