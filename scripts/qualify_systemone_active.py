"""Compare deterministic parent continuation with/without active SystemOne reads.

Requires an independently owned local test server. This client never manages
server processes. Enable verify_session_restore in the serving profile for
byte-level rollback proof.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx


async def qualify(base_url, output, context_repetitions=0):
    from qualify_systemone import origin
    base_url = origin(base_url)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'arms': []}
    def save():
        (output / 'results.json').write_text(json.dumps(report, indent=2))
    save()
    async with httpx.AsyncClient(base_url=base_url, timeout=180, trust_env=False) as client:
        models = await client.get('/v1/models')
        models.raise_for_status()
        model = models.json()['data'][0]['id']
        response = await client.post('/tokenize', json={'model': model,
            'messages': [{'role': 'user', 'content':
                'Record: the parcel was damaged during shipping.\n' * context_repetitions +
                'The parcel arrived broken. I want a refund. '
                'Write a detailed numbered list of practical steps for handling this return. '
                'Continue adding useful steps.'}], 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}})
        response.raise_for_status()
        tokens = response.json()['tokens']
        report['prompt_tokens'] = len(tokens)
        for active in (False, True):
            session = 'active-qualification-' + uuid4().hex
            arm = {'active': active, 'session': session, 'token_ids': [], 'decisions': []}
            report['arms'].append(arm)
            ready = asyncio.Event()
            async def consume():
                async with client.stream('POST', '/v1/completions', json={
                        'model': model, 'prompt': tokens, 'session_id': session, 'cache_salt': session,
                        'temperature': 0, 'max_tokens': 512, 'ignore_eos': True, 'seed': 42,
                        'return_token_ids': True, 'stream': True}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: '): continue
                        if line[6:] == '[DONE]':
                            arm['done'] = True
                            break
                        chunk = json.loads(line[6:])
                        assert 'error' not in chunk, chunk
                        for choice in chunk.get('choices', []):
                            arm['token_ids'].extend(choice.get('token_ids') or [])
                        if len(arm['token_ids']) >= 32:
                            ready.set()
                assert len(arm['token_ids']) == 512, arm
            task = asyncio.create_task(consume())
            try:
                if active:
                    await asyncio.wait_for(ready.wait(), 60)
                    question = {'type': 'noul', 'instructions': 'Was the parcel damaged?'}
                    payload = {'model': model, 'state': '', 'cache_salt': session,
                        'kvmem_session': {'session_id': session}, 'questions': {
                            'first': question, 'middle': {'type': 'choice', 'instructions': 'What does the customer want?',
                                'criteria': {'refund': 'A refund', 'replacement': 'A replacement'}}, 'repeat': question}}
                    for _ in range(2):
                        started = time.perf_counter()
                        response = await client.post('/v1/systemone', json=payload)
                        result = {'status': response.status_code, 'seconds': time.perf_counter() - started,
                                  'body': response.json()}
                        arm['decisions'].append(result)
                        save()
                        assert response.status_code == 200, result
                        body = result['body']
                        assert body['answers']['first'] == body['answers']['repeat'], body
                        assert body['diagnostics']['first'] == body['diagnostics']['repeat'], body
                        assert body['kvmem']['mode'] == 'active_branch', body
                        assert body['kvmem']['history_end'] >= len(tokens), body
                await task
            finally:
                # Let our finite parent finish even when a decision assertion
                # fails; release only our own namespace after its native fence.
                await task
                response = await client.post('/v1/completions', json={
                    'model': model, 'prompt': [tokens[0]], 'session_id': session, 'cache_salt': session,
                    'max_tokens': 1, 'temperature': 0, 'kv_transfer_params': {'kvmem_release_session': True}})
                arm['release_status'] = response.status_code
                save()
                response.raise_for_status()
        report['parent_token_equal'] = report['arms'][0]['token_ids'] == report['arms'][1]['token_ids']
        assert report['parent_token_equal'], 'Active read changed deterministic parent continuation'
        report['rollback_verified'] = all(d['body']['kvmem']['rollback_verified']
            for d in report['arms'][1]['decisions'])
        report['compacted_prefix'] = all(d['body']['kvmem']['active_tokens'] <
            d['body']['kvmem']['history_end'] for d in report['arms'][1]['decisions'])
        report['status'] = 'passed'
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:18890')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--context-repetitions', type=int, default=0)
    args = parser.parse_args()
    result = asyncio.run(qualify(args.base_url, args.output, args.context_repetitions))
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))
