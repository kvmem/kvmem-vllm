"""Mechanism check for the thinking age rule against an already owned idle server.

One structured agent conversation (the real bridge renderer) holds `turns`
assistant messages, each with its own long reasoning, separated by tool
observations. With thinking_age_steps=N, the thinking of turns 1..turns-N is
old. The check passes when the server retires old thinking without capacity
pressure. Excluded from benchmark scores and timings.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import httpx

try:
    from .qualify_thinking_gc_preflight import clean, read, save, snapshot
except ImportError:
    from qualify_thinking_gc_preflight import clean, read, save, snapshot


def conversation(turns, thinking_tokens, planner):
    """Messages whose assistant reasoning is distinct per turn (no cross-turn redundancy)."""
    question = 'Find the bug in the parser and fix it. Explain each step.'
    messages = [dict(role='user', content=question)]
    for turn in range(1, turns + 1):
        unit = (f'Step {turn}: I inspect file module_{turn}.py, line {turn * 7}, and compare the '
                f'token stream against case {turn * 13}. Hypothesis {turn} remains open.\n')
        repeats = max(1, thinking_tokens // len(planner.encode(unit)))
        messages.append(dict(role='assistant', reasoning_content=unit * repeats,
                             content=f'Running check {turn}.'))
        messages.append(dict(role='user', content=f'Tool observation {turn}: exit code 0, '
                                                  f'{turn * 3} tests passed.'))
    return question, messages


def render(profile, protocol):
    from transformers import AutoTokenizer
    from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory
    from scripts.dsh_bridge import to_anthropic
    from scripts.dsh_query_epoch import prepare_with_query_provenance
    from scripts.terminalbench_query import TaskQuery
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    planner = Planner(Protocol(protocol), tokenizer, profile['kvmem']['gen_budget'])
    return planner, (lambda body: prepare_with_query_provenance(
        planner, to_anthropic(body), Trajectory(), body, required=True)), TaskQuery


async def exercise(base, profile, protocol, capture, output, turns, thinking_tokens, wait_seconds):
    steps = ((profile['kvmem'].get('historical_gc') or {}).get('policy_params') or {}).get('thinking_age_steps', 0)
    if not steps or turns <= steps:
        raise ValueError('Needs a thinking_age_steps profile and more assistant turns than the age')
    output.mkdir(parents=True, exist_ok=False)
    planner, prepare, TaskQuery = render(profile, protocol)
    question, messages = conversation(turns, thinking_tokens, planner)
    session = 'thinking-age-check-' + uuid.uuid4().hex
    source = TaskQuery(session, 'fixed-invariant', question, question)
    # The Query tracker accepts the conversation one user message at a time.
    for end in range(1, len(messages) + 1):
        if messages[end - 1]['role'] == 'user':
            source.proof(messages[:end])
    body = dict(model=profile['served_model_name'], prompt_cache_key=session, messages=messages)
    body['kvmem_query_provenance'] = source.proof(messages)
    prepared, _ = prepare(body)
    tokens, plan = list(prepared[0]), prepared[1]
    plan.update(kvmem_mandatory_parts='none', kvmem_mandatory_spans=[])
    assistants = sum(1 for span in plan.get('kvmem_locality_spans') or [] if span['role'] == 'assistant')
    save(output/'fixture.json', dict(turns=turns, thinking_tokens_per_turn=thinking_tokens, steps=steps,
                                     prompt_tokens=len(tokens), assistant_spans=assistants))
    deadline = time.monotonic() + 90
    while not clean(snapshot(capture)) and time.monotonic() < deadline:
        await asyncio.sleep(.1)
    before = snapshot(capture)
    if not clean(before):
        raise ValueError('Needs a drained server with no other history owners')
    save(output/'before.json', before)
    started = time.perf_counter()
    gc_rows = []
    async with httpx.AsyncClient(base_url=base, trust_env=False, timeout=1800) as client:
        request = dict(model=profile['served_model_name'], session_id=session, cache_salt=session,
                       prompt=tokens, **plan, max_tokens=64, temperature=0, seed=1000, return_token_ids=True)
        job = asyncio.create_task(client.post('/v1/completions', json=request))
        while not job.done():
            gc_rows.append(dict(at=time.perf_counter() - started,
                                gc=snapshot(capture)['archive'].get('historical_gc') or {}))
            await asyncio.wait({job}, timeout=.5)
        response = await job
        save(output/'response.json', dict(status=response.status_code, body=response.text[:4000]))
        response.raise_for_status()
        # Retirement is asynchronous: give the native worker a short window.
        until = time.monotonic() + wait_seconds
        while time.monotonic() < until:
            row = snapshot(capture)['archive'].get('historical_gc') or {}
            gc_rows.append(dict(at=time.perf_counter() - started, gc=row))
            if row.get('thinking_age_retired', 0) > 0:
                break
            await asyncio.sleep(.5)
        after = snapshot(capture)
        save(output/'after.json', after)
        release = await client.post('/v1/completions', json=dict(model=profile['served_model_name'],
            session_id=session, cache_salt=session, prompt=[planner.end], max_tokens=1,
            kv_transfer_params=dict(kvmem_release_session=True)))
        save(output/'release.json', dict(status=release.status_code, body=release.text[:2000]))
    gc = after['archive'].get('historical_gc') or {}
    base_gc = before['archive'].get('historical_gc') or {}
    delta = {k: gc.get(k, 0) - base_gc.get(k, 0) for k in
             ('thinking_pure', 'thinking_age_retired', 'thinking_age_ineligible', 'retired',
              'redundancy_retired', 'lru_fallback_retired', 'policy_failures')}
    verdict = dict(passed=bool(delta['thinking_pure'] > 0 and delta['thinking_age_retired'] > 0 and
                               delta['policy_failures'] == 0),
                   delta=delta, thinking_age_protected=gc.get('thinking_age_protected'),
                   prompt_tokens=len(tokens), turns=turns, steps=steps,
                   seconds=time.perf_counter() - started)
    save(output/'gc-trace.json', gc_rows)
    save(output/'result.json', verdict)
    print(json.dumps(verdict, indent=2))
    if not verdict['passed']:
        raise RuntimeError('Thinking age check failed; inspect result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--turns', type=int, default=13)
    parser.add_argument('--thinking-tokens', type=int, default=6000)
    parser.add_argument('--wait-seconds', type=float, default=30)
    args = parser.parse_args()
    asyncio.run(exercise(args.base, read(args.profile), args.protocol, args.capture, args.output,
                         args.turns, args.thinking_tokens, args.wait_seconds))
