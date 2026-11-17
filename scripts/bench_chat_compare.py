"""Engine-neutral throughput client over /v1/chat/completions (KVMem/vLLM and Strata alike).

Every engine gets the same user text: the synthetic review prompt of
benchmark_qsa_phases.py, sized so the chat-templated prompt (thinking off) is
about --input-tokens. End-of-turn tokens are banned with logit_bias so every
answer runs to --output-tokens (Strata has no ignore_eos). Tokens are counted
with the model's tokenizer from the streamed text; the totals are checked against
the server's usage when it reports one.

Metrics match benchmark_qsa_phases.phase_summary: effective prefill = prompt
tokens / (last first token - first start); steady decode = tokens streamed by all
requests inside the common interval [last first token + 2 s, first finish - 2 s].
"""
import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from pathlib import Path


def build_prompt(tok, owner, seed, input_tokens):
    rng = random.Random(seed + owner)
    words = ('alpha beta gamma delta orange violet green silver north south east west river mountain cloud window '
             'table pencil chair garden paper clock ocean cache process request token memory model system kernel '
             'compute buffer').split()
    unique = ' '.join(rng.choice(words) for _ in range(128))
    body = ('Repository review notes. The service accepts independent requests, keeps an ordered event log, '
            'restores a checkpoint after a worker restart, and publishes a result after all pending writes finish. '
            'Compare the latency, memory use, scheduling, and correctness tradeoffs. Consider a bounded queue, '
            'several clients, and a long conversation. ')
    ending = ('\nWrite a long numbered engineering checklist explaining the design choices and examples in detail. '
              'Continue for at least 1000 words.\n')

    def ids_for(text):
        templated = tok.apply_chat_template([dict(role='user', content=text)], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return tok.encode(templated, add_special_tokens=False)

    base = len(ids_for(unique + ending))
    per = len(tok.encode(body, add_special_tokens=False))
    copies = max(0, (input_tokens - base) // per)
    text = unique + ' ' + body * copies + ending
    ids = ids_for(text)
    while len(ids) > input_tokens and copies:          # never exceed the target length
        copies -= 1
        text = unique + ' ' + body * copies + ending
        ids = ids_for(text)
    tail = len(tok.encode(ending, add_special_tokens=False))
    suffix = len(ids_for(text)) - len(ids_for(text[:-len(ending)] + ending))   # 0: the ending closes the content
    templ_suffix = len(tok.encode(tok.apply_chat_template([dict(role='user', content='')], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                                  .split('<|im_end|>')[-1], add_special_tokens=False)) + 1
    query = [len(ids) - templ_suffix - tail - suffix, len(ids)]
    return text, ids, query


def phase_summary(rows, margin=2.0):
    first_start = min(r['started'] for r in rows)
    last_first = max(r['first'] for r in rows)
    result = dict(concurrency=len(rows), input_tokens_per_request=[r['input_tokens'] for r in rows],
                  output_tokens_per_request=[r['output_tokens'] for r in rows],
                  prefill_wall_seconds=last_first - first_start,
                  effective_prefill_tokens_per_second=sum(r['input_tokens'] for r in rows) / (last_first - first_start),
                  ttft_seconds=[r['first'] - r['started'] for r in rows],
                  wall_seconds=max(r['last'] for r in rows) - first_start)
    result['end_to_end_output_tokens_per_second'] = sum(r['output_tokens'] for r in rows) / result['wall_seconds']
    begin, end = last_first + margin, min(r['last'] for r in rows) - margin
    if end <= begin:
        result.update(decode_steady_seconds=0, decode_steady_aggregate_tokens_per_second=None)
        return result
    counts = [sum(c['tokens'] for c in r['chunks'] if begin < c['at'] <= end) for r in rows]
    result.update(decode_steady_seconds=end - begin, decode_steady_tokens_by_request=counts,
                  decode_steady_aggregate_tokens_per_second=sum(counts) / (end - begin))
    return result


async def run(args):
    import httpx
    from transformers import AutoTokenizer
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'arguments.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    banned = sorted({tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')} |
                    ({tok.eos_token_id} if tok.eos_token_id is not None else set()))
    prompts = [build_prompt(tok, owner, args.seed, args.input_tokens) for owner in range(args.concurrency)]
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=10), trust_env=False) as client:
        deadline = time.monotonic() + args.wait_ready_seconds
        while True:
            try:
                if (await client.get(args.url + args.health_path, timeout=3)).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Server readiness timeout')
            await asyncio.sleep(5)

        async def one(owner):
            text, ids, query = prompts[owner]
            body = dict(model=args.model, messages=[dict(role='user', content=text)], max_tokens=args.output_tokens,
                        temperature=0, stream=True, stream_options=dict(include_usage=True),
                        logit_bias={str(t): -100 for t in banned},
                        chat_template_kwargs=dict(enable_thinking=False))
            path = '/v1/chat/completions'
            if args.engine in ('kvmem', 'vllm'):
                # KVMem takes its explicit Query span only with token-ID prompts: send the chat-templated IDs
                # (the same prompt the chat route builds) to /v1/completions. Plain vLLM gets the same IDs.
                body = dict(model=args.model, prompt=ids, max_tokens=args.output_tokens, temperature=0, stream=True,
                            stream_options=dict(include_usage=True), ignore_eos=True, return_token_ids=True)
                if args.engine == 'kvmem':
                    body.update(kvmem_query_span=query, kvmem_reselect='force')
                path = '/v1/completions'
            else:
                body.update(reasoning_effort='none')
            started = time.perf_counter()
            chunks, parts, usage = [], [], None
            async with client.stream('POST', args.url + path, json=body) as response:
                if response.status_code != 200:
                    raise RuntimeError((await response.aread()).decode()[:2000])
                async for line in response.aiter_lines():
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    value = json.loads(data)
                    if 'error' in value:
                        raise RuntimeError(value['error'])
                    if value.get('usage'):
                        usage = value['usage']
                    for choice in value.get('choices', []):
                        delta = (choice.get('delta') or {})
                        piece = ((choice.get('text') or '') + (delta.get('content') or '') +
                                 (delta.get('reasoning_content') or delta.get('reasoning') or ''))
                        if piece:
                            chunks.append(dict(at=time.perf_counter(), text=piece))
                            parts.append(piece)
            output = ''.join(parts)
            total = len(tok.encode(output, add_special_tokens=False))
            if usage and usage.get('completion_tokens'):
                total = usage['completion_tokens']
            # Per-chunk token counts, scaled so they add up to the total.
            sizes = [max(1, len(tok.encode(c['text'], add_special_tokens=False))) for c in chunks]
            scale = total / sum(sizes)
            for c, n in zip(chunks, sizes):
                c['tokens'] = n * scale
                del c['text']
            row = dict(owner=owner, started=started, first=chunks[0]['at'], last=chunks[-1]['at'],
                       input_tokens=(usage or {}).get('prompt_tokens') or len(ids), client_prompt_tokens=len(ids),
                       output_tokens=total, chunks=chunks, usage=usage,
                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       output_sha256=hashlib.sha256(output.encode()).hexdigest(), output_excerpt=output[:600])
            (out / f'request-{owner}.json').write_text(json.dumps(row, indent=2) + '\n')
            print(json.dumps(dict(owner=owner, status='complete', output_tokens=total)), flush=True)
            return row

        rows = await asyncio.gather(*(one(i) for i in range(args.concurrency)))
    result = phase_summary(rows)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('kvmem', 'vllm', 'strata'), required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--model', default='model')
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--input-tokens', type=int, default=16384)
    p.add_argument('--output-tokens', type=int, default=8192)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--seed', type=int, default=11317)
    p.add_argument('--health-path', default='/health')
    p.add_argument('--wait-ready-seconds', type=int, default=3600)
    asyncio.run(run(p.parse_args()))
