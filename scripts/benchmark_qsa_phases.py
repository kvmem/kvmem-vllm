"""Cold-prompt phase measurements for the explicit QSA/KVMem serving profile.

Decode counts accepted streamed token IDs, never speculative verification rows.
Prefill-only trials generate one token; sustained trials report the common
decode interval with a two-second margin at each end. No overall output rate.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import statistics
import time


def phase_summary(rows, margin=2.0):
    if not rows or any(not r['token_ids_verified'] for r in rows):
        raise ValueError('Phase metrics require verified accepted token IDs')
    first_start = min(r['started'] for r in rows)
    last_first = max(r['first'] for r in rows)
    result = dict(concurrency=len(rows),
        input_tokens_per_request=[r['input_tokens'] for r in rows],
        output_tokens_per_request=[r['output_tokens'] for r in rows],
        prefill_wall_seconds=last_first-first_start,
        effective_prefill_tokens_per_second=sum(r['input_tokens'] for r in rows)/(last_first-first_start),
        ttft_seconds=[r['first']-r['started'] for r in rows])
    if all(r['output_tokens']==1 for r in rows):
        result['scope'] = 'Prefill-only: common wall interval through the last first token; includes queue, KVMem maintenance and first-token sampling.'
        return result
    begin, end = last_first+margin, min(r['last'] for r in rows)-margin
    if end <= begin:
        result.update(decode_steady_seconds=0, decode_steady_aggregate_tokens_per_second=None)
        return result
    counts = [sum(c['tokens'] for c in r['chunks'] if begin < c['at'] <= end) for r in rows]
    result.update(decode_steady_seconds=end-begin,
        decode_steady_tokens_by_request=counts,
        decode_steady_tokens_per_second_by_request=[n/(end-begin) for n in counts],
        decode_steady_aggregate_tokens_per_second=sum(counts)/(end-begin),
        decode_interval_begin=begin, decode_interval_end=end,
        scope='Decode: accepted tokens inside the common streaming interval, trimming two seconds at both ends. Input/TTFT metrics are separate and include KVMem maintenance.')
    return result


async def benchmark(args):
    import httpx
    from transformers import AutoTokenizer
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out/'arguments.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    marker = 'QSA_BENCHMARK_BODY_6b7c'
    template = tok.apply_chat_template([dict(role='user',content=marker)],
        tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
    assert template.count(marker)==1
    prefix, suffix = [tok.encode(p, add_special_tokens=False) for p in template.split(marker)]
    body = tok.encode('Repository review notes. The service accepts independent requests, '
        'keeps an ordered event log, restores a checkpoint after a worker restart, '
        'and publishes a result after all pending writes finish. Compare the '
        'latency, memory use, scheduling, and correctness tradeoffs. '
        'Consider a bounded queue, several clients, and a long conversation. ', add_special_tokens=False)
    words = tok.encode(' alpha beta gamma delta orange violet green silver north south east west '
        'river mountain cloud window table pencil chair garden paper clock ocean '
        'cache process request token memory model system kernel compute buffer', add_special_tokens=False)
    ending = tok.encode('\nWrite a long numbered engineering checklist explaining the design '
        'choices and examples in detail. Continue for at least 1000 words.\n', add_special_tokens=False)
    payloads=[]
    # Real text for calibration traces: one JSON object per line with "context" and
    # "question". The context keeps its newest tokens within --input-tokens; the
    # question is the KVMem query span.
    documents=[json.loads(line) for line in Path(args.prompt_file).read_text().splitlines()
               if line.strip()] if args.prompt_file else None
    for owner in range(args.concurrency):
        if documents:
            row=documents[(args.prompt_offset+owner)%len(documents)]
            question=tok.encode('\n\n'+row['question'].strip()+'\n',add_special_tokens=False)
            room=args.input_tokens-len(prefix)-len(question)-len(suffix)
            assert room>0
            context=tok.encode(row['context'],add_special_tokens=False)[-room:]
            ids=prefix+context+question+suffix
            query=[len(prefix)+len(context),len(ids)]
            payloads.append((ids,json.dumps(dict(model=args.served_model,prompt=ids,
                temperature=0,max_tokens=args.output_tokens,ignore_eos=True,stream=True,
                stream_options={'include_usage':True},return_token_ids=True,
                kvmem_query_span=query,kvmem_reselect='force')).encode()))
            continue
        rng=random.Random(args.seed+owner)
        unique=[rng.choice(words) for _ in range(128)]
        remaining=args.input_tokens-len(prefix)-len(unique)-len(ending)-len(suffix)
        assert remaining>0
        ids=prefix+unique+(body*((remaining+len(body)-1)//len(body)))[:remaining]+ending+suffix
        query=[len(ids)-len(ending)-len(suffix),len(ids)]
        payloads.append((ids,json.dumps(dict(model=args.served_model,prompt=ids,
            temperature=0,max_tokens=args.output_tokens,ignore_eos=True,stream=True,
            stream_options={'include_usage':True},return_token_ids=True,
            kvmem_query_span=query,kvmem_reselect='force')).encode()))
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800,connect=10),trust_env=False) as client:
        deadline=time.monotonic()+args.wait_ready_seconds
        while True:
            try:
                ready=await client.get(args.url+'/health',timeout=3)
                if ready.status_code==200:break
            except httpx.HTTPError:pass
            if time.monotonic()>deadline:raise RuntimeError('Server readiness timeout')
            await asyncio.sleep(5)
        for trial in range(args.trials):
            (out/f'metrics-before-{trial}.prom').write_text((await client.get(args.url+'/metrics')).text)
            async def one(owner):
                ids,payload=payloads[owner]
                started=time.perf_counter()
                chunks=[];tokens=[];parts=[];usage=None;metrics=None;finish=None
                async with client.stream('POST',args.url+'/v1/completions',content=payload,
                        headers={'Content-Type':'application/json'}) as response:
                    if response.status_code!=200:raise RuntimeError((await response.aread()).decode())
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):continue
                        data=line[5:].strip()
                        if data=='[DONE]':break
                        value=json.loads(data)
                        if 'error' in value:raise RuntimeError(value['error'])
                        if value.get('usage'):usage=value['usage']
                        if value.get('metrics'):metrics=value['metrics']
                        for choice in value.get('choices',[]):
                            new=choice.get('token_ids') or []
                            if new:
                                chunks.append(dict(at=time.perf_counter(),tokens=len(new)))
                                tokens.extend(new)
                            parts.append(choice.get('text') or '')
                            if choice.get('finish_reason'):finish=choice['finish_reason']
                assert usage and chunks and usage['prompt_tokens']==len(ids)
                assert usage['completion_tokens']==args.output_tokens==len(tokens)
                row=dict(owner=owner,trial=trial,started=started,first=chunks[0]['at'],last=chunks[-1]['at'],
                    input_tokens=len(ids),output_tokens=len(tokens),token_ids_verified=True,
                    chunks=chunks,usage=usage,metrics=metrics,finish_reason=finish,
                    prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    output_excerpt=''.join(parts)[:600])
                (out/f'request-{trial}-{owner}.json').write_text(json.dumps(row,indent=2)+'\n')
                print(json.dumps(dict(owner=owner,trial=trial,status='complete',output_tokens=len(tokens))),flush=True)
                return row
            rows=await asyncio.gather(*(one(i) for i in range(args.concurrency)))
            result=phase_summary(rows)
            (out/f'trial-{trial}.json').write_text(json.dumps(result,indent=2)+'\n')
            (out/f'metrics-after-{trial}.prom').write_text((await client.get(args.url+'/metrics')).text)
            print(json.dumps(result),flush=True)
            # Allow immutable-history verification and retirement to finish.
            await asyncio.sleep(args.between_trials_seconds)
    trials=[json.loads((out/f'trial-{i}.json').read_text()) for i in range(args.trials)]
    summary=dict(trials=trials,median_effective_prefill_tokens_per_second=statistics.median(
        r['effective_prefill_tokens_per_second'] for r in trials))
    rates=[r.get('decode_steady_aggregate_tokens_per_second') for r in trials]
    if all(x is not None for x in rates):
        summary['median_decode_steady_aggregate_tokens_per_second']=statistics.median(rates)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--served-model',default='kvmem-flash-next')
    p.add_argument('--url',default='http://127.0.0.1:18938')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--input-tokens',type=int,default=258048)
    p.add_argument('--output-tokens',type=int,default=3072)
    p.add_argument('--concurrency',type=int,default=3)
    p.add_argument('--trials',type=int,default=1)
    p.add_argument('--seed',type=int,default=11317)
    p.add_argument('--wait-ready-seconds',type=int,default=1800)
    p.add_argument('--between-trials-seconds',type=float,default=0)
    p.add_argument('--prompt-file',help='JSONL rows with "context" and "question" (real text)')
    p.add_argument('--prompt-offset',type=int,default=0)
    p.add_argument('--enable-thinking',action='store_true')
    args=p.parse_args()
    assert args.input_tokens>0 and args.output_tokens>0 and args.concurrency>0 and args.trials>0
    assert args.between_trials_seconds>=0
    asyncio.run(benchmark(args))
