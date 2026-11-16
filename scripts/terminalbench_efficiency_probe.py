"""Fixed-token long-history component probe, never a Terminal-Bench score.

The controller owns the server and GPU. This client preserves real task prefixes
and observations, repeats recorded reasoning to exercise a specified capacity,
and accepts only actual model-returned fixed tokens (including terminal EOS).
Run the same frozen fixture against each compared server configuration.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import httpx
from vllm_kvmem.core.request_records import load_request


CAPTURE_FILES = ('async_restore.jsonl', 'scheduler_refresh.jsonl', 'async_refresh.jsonl',
                 'model_step_latency.jsonl', 'host_stages.jsonl', 'restore_host_stages.jsonl',
                 'checkpoint_timing.jsonl', 'scheduler_finished.jsonl')
COUNTERS = ('retired', 'redundancy_retired', 'lru_fallback_retired', 'lru_representative_retired',
            'policy_failures', 'thinking_classified', 'thinking_pure', 'cpu_released', 'slots_released')


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def repeat_tokens(unit, count):
    if not unit or count < 0:
        raise ValueError('Need nonempty reasoning and nonnegative padding')
    return (unit * ((count + len(unit) - 1) // len(unit)))[:count]


def exception_chain(error):
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        result.append(dict(type=type(error).__name__, message=str(error)))
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return result


def prepare(short_fixture, tokenizer_path, output, owners=4, turns=5,
            initial_tokens=65536, history_step=65536, output_tokens=512):
    from transformers import AutoTokenizer
    from scripts.terminalbench_replay import replay
    if output.exists():
        raise ValueError('Use a new fixture path')
    short = read(short_fixture)
    chosen = short['owners'][:owners]
    if len(chosen) != owners or min(owners, turns, initial_tokens, history_step, output_tokens) < 1:
        raise ValueError('Invalid workload size')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    source = Path(short['source'])
    traces = replay(source, max_owners=owners, max_turns=2, task_names=[v['task'] for v in chosen])
    directories = {r['task']: source/'bridge'/r['owner'] for r in traces['traces']}
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if not isinstance(end, int) or end < 0:
        raise ValueError('Actual tokenizer has no message EOS')
    values = []
    for owner in chosen:
        request_path = directories[owner['task']]/'00002/request.json'
        body = load_request(request_path)
        reasoning = next((m.get('reasoning_content') for m in body['messages']
                          if m['role'] == 'assistant' and m.get('reasoning_content')), None)
        if not isinstance(reasoning, str) or not reasoning:
            raise ValueError('Need genuine recorded assistant reasoning: '+owner['task'])
        unit = tokenizer.encode(reasoning, add_special_tokens=False)
        # These are plain reasoning words; a special message boundary would
        # make the artificial repeated history structurally ambiguous.
        if set(unit).intersection(tokenizer.all_special_ids):
            raise ValueError('Recorded reasoning contains message-control tokens')
        initial_plan = read(directories[owner['task']]/'00001/plan.json')
        initial_spans = initial_plan.get('kvmem_locality_spans') or initial_plan.get('kvmem_message_spans')
        assistant_header = tokenizer.encode('<|im_start|>assistant\n<think>\n', add_special_tokens=False)
        if (initial_plan.get('prompt_tokens') != len(owner['prefix']) or not initial_spans or
                owner['prefix'][initial_spans[-1]['end']:] != assistant_header):
            raise ValueError('Original task prefix does not match its exact outer role boundaries')
        initial_spans = [dict(span) for span in initial_spans]
        close = tokenizer.encode('</think><|im_end|>', add_special_tokens=False)
        separator = tokenizer.encode('\n', add_special_tokens=False)
        users = []
        for index in range(max(1, turns)):
            original = owner['additions'][index % len(owner['additions'])]
            begin, finish = original['query_offset']
            observation = original['tokens'][begin:finish]
            tokens = tokenizer.encode('<|im_start|>user\n', add_special_tokens=False) + observation
            tokens += tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
            users.append(tokens)
        # Only COMPLETE messages get explicit spans. The final generation
        # header remains outside them, just as the real Planner renders it.
        # Extending an explicitly closed assistant span on the EOS request
        # contradicts the saved checkpoint's validated message boundary.
        initial_tail = close + separator + users[0] + assistant_header
        if len(owner['prefix']) + len(initial_tail) > initial_tokens:
            raise ValueError('Initial history cannot fit its complete messages')
        initial_closed = initial_tokens-len(initial_tail)+len(close)
        generation_begin = initial_tokens-len(assistant_header)
        initial_spans.extend([dict(begin=initial_spans[-1]['end'], end=initial_closed, role='assistant'),
            dict(begin=initial_closed+len(separator), end=generation_begin, role='user')])
        additions = []
        for index in range(turns-1):
            user = users[index+1]
            head = user + assistant_header
            tail = close + separator + user + assistant_header
            if len(head)+len(tail) > history_step:
                raise ValueError('History increment cannot fit the real observation')
            closed = history_step-len(tail)+len(close)
            generated = history_step-len(assistant_header)
            additions.append(dict(head=head, tail=tail, generation_begin=generated,
                spans=[dict(begin=0, end=len(user), role='user'),
                       dict(begin=len(user), end=closed, role='assistant'),
                       dict(begin=closed+len(separator), end=generated, role='user')]))
        values.append(dict(task=owner['task'], prefix=owner['prefix'], query=owner['query'],
                           reasoning_tokens=unit, additions=additions,
                           initial_tail=initial_tail, initial_spans=initial_spans,
                           generation_begin=generation_begin,
                           reasoning_source=str(request_path),
                           reasoning_source_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()))
    result = dict(schema=3, scope='Synthetic repeated real TB reasoning; component efficiency only, no task score',
                  short_fixture=str(short_fixture), short_fixture_sha256=hashlib.sha256(short_fixture.read_bytes()).hexdigest(),
                  tokenizer=str(tokenizer_path), owners=values, turns=turns, initial_tokens=initial_tokens,
                  history_step=history_step, output_tokens=output_tokens, output_token=short['output_token'],
                  eos_token=end, expected_final_history=initial_tokens+(turns-1)*history_step+turns*(output_tokens+1))
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    print(json.dumps(dict(fixture=str(output), owners=owners, turns=turns,
                         expected_final_history=result['expected_final_history'], sha256=digest(result))))


def snapshot(capture):
    if capture is None:
        return {}
    return {name: read(capture/(name+'.json')) for name in ('progress', 'archive', 'scheduler_resources')}


def native_summary(capture_dir, records):
    def rows(name):
        path = capture_dir/name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = {row['request_id'] for row in records}
    restores = [row for row in rows('async_restore.jsonl')
                if any(row.get('request_id', '').startswith(rid+'-') for rid in ids)]
    refreshes = [row for row in rows('scheduler_refresh.jsonl')
                 if any(row.get('request_id', '').startswith(rid+'-') for rid in ids)]
    return dict(restores=len(restores), restore_lanes=dict(Counter(r.get('lane') for r in restores)),
                restore_seconds=sum(r['finished']-r['started'] for r in restores),
                uploaded_bytes=sum(r.get('uploaded_bytes', 0) for r in restores),
                model_uploaded_bytes=sum(r.get('model_uploaded_bytes', 0) for r in restores),
                pressure_refreshes=len(refreshes), replay_tokens=sum(r.get('replay_tokens', 0) for r in refreshes))


async def exercise(base, model, fixture, output, capture=None, label='baseline', timeout=3600):
    if output.exists():
        raise ValueError('Use a new probe evidence directory')
    output.mkdir(parents=True)
    config = read(fixture)
    save(output/'fixture.json', config)
    offsets = {name: (capture/name).stat().st_size if capture and (capture/name).exists() else 0
               for name in CAPTURE_FILES}
    before = snapshot(capture)
    save(output/'before.json', before)
    records, errors, releases, observations = [], [], [], [before]
    sessions = {owner['task']: 'efficiency-'+uuid.uuid4().hex for owner in config['owners']}
    started, started_monotonic = time.time(), time.perf_counter()
    save(output/'status.json', dict(status='running', label=label, started_at=started))
    failed = None
    # Each request gets a fresh local TCP connection. Long concurrent response
    # processing can outlast the server's idle keep-alive deadline; reuse is
    # unnecessary here and an ambiguous disconnect must never trigger a second
    # generation. HTTPX's default transport has zero connection retries.
    transport_policy = dict(max_connections=max(16, len(config['owners'])*2),
                            max_keepalive_connections=0, request_header_connection='close',
                            automatic_retries=0)
    save(output/'transport-policy.json', transport_policy)
    async with httpx.AsyncClient(base_url=base, trust_env=False, timeout=timeout,
            limits=httpx.Limits(max_connections=transport_policy['max_connections'], max_keepalive_connections=0),
            headers={'Connection': 'close'}) as client:
        async def call(tokens, spans, owner, turn, part, token, count):
            payload = dict(model=model, session_id=sessions[owner['task']], cache_salt=sessions[owner['task']],
                           prompt=tokens, max_tokens=count, temperature=0, seed=1000,
                           allowed_token_ids=[token], ignore_eos=(part != 'eos'), return_token_ids=True,
                           kvmem_reselect='auto', kvmem_query_span=owner['query'], kvmem_query_mode='agent',
                           kvmem_query_id='fixed-task-'+owner['task'],
                           kvmem_mandatory_parts='none', kvmem_mandatory_spans=[],
                           kvmem_message_spans=[dict(span) for span in spans])
            input_hash = digest(tokens)
            at, began = time.time(), time.perf_counter()
            event = dict(task=owner['task'], turn=turn, part=part, prompt_tokens=len(tokens),
                         input_sha256=input_hash, message_spans_sha256=digest(spans), at=at)
            with (output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='request_start', **event))+'\n')
            async def transport_trace(name, info):
                trace = dict(task=owner['task'], turn=turn, part=part, event=name,
                             at=time.time(), elapsed_seconds=time.perf_counter()-began)
                if info.get('exception') is not None:
                    trace['exception_chain'] = exception_chain(info['exception'])
                # Do not record request bodies or opaque transport objects.
                with (output/'http-transport.jsonl').open('a') as stream:
                    stream.write(json.dumps(trace)+'\n')
            try:
                response = await client.post('/v1/completions', json=payload,
                                             extensions={'trace': transport_trace})
            except BaseException as error:
                failure = dict(**event, ended_at=time.time(), seconds=time.perf_counter()-began,
                               cancelled=isinstance(error, asyncio.CancelledError),
                               exception_chain=exception_chain(error),
                               traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)),
                               retried=False)
                save(output/f"error-{owner['task']}-{turn}-{part}.json", failure)
                with (output/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(event='request_cancelled' if failure['cancelled'] else 'request_error',
                                                  **{k: v for k, v in failure.items() if k != 'traceback'}))+'\n')
                raise
            elapsed, finished = time.perf_counter()-began, time.time()
            if response.is_error:
                save(output/f"error-{owner['task']}-{turn}-{part}.json",
                     dict(**event, status=response.status_code, response=response.text))
            response.raise_for_status()
            data = response.json()
            accepted = data['choices'][0]['token_ids']
            if accepted != [token]*count:
                raise ValueError('Model returned different tokens than the fixed-output contract')
            row = dict(**event, ended_at=finished, seconds=elapsed, request_id=data['id'],
                       output_sha256=digest(accepted), accepted_tokens=len(accepted), response=data)
            records.append(row)
            save(output/'requests.json', records)
            with (output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='request_end', **{k: v for k, v in row.items() if k != 'response'}))+'\n')
            observations.append(snapshot(capture))
            save(output/'status.json', dict(status='running', label=label, started_at=started,
                 completed_requests=len(records), last=dict(task=owner['task'], turn=turn, part=part), at=time.time()))
            return accepted

        async def owner_task(owner):
            tokens = (list(owner['prefix']) + repeat_tokens(owner['reasoning_tokens'],
                config['initial_tokens']-len(owner['prefix'])-len(owner['initial_tail'])) + owner['initial_tail'])
            spans = [dict(span) for span in owner['initial_spans']]
            generation_begin = owner['generation_begin']
            for turn in range(config['turns']):
                if turn:
                    begin = len(tokens)
                    added = owner['additions'][turn-1]
                    tokens.extend(added['head'])
                    tokens.extend(repeat_tokens(owner['reasoning_tokens'],
                        config['history_step']-len(added['head'])-len(added['tail'])))
                    tokens.extend(added['tail'])
                    spans.extend(dict(begin=begin+s['begin'], end=begin+s['end'], role=s['role'])
                                 for s in added['spans'])
                    generation_begin = begin+added['generation_begin']
                tokens.extend(await call(tokens, spans, owner, turn+1, 'thinking', config['output_token'], config['output_tokens']))
                tokens.extend(await call(tokens, spans, owner, turn+1, 'eos', config['eos_token'], 1))
                spans.append(dict(begin=generation_begin, end=len(tokens), role='assistant'))
            if len(tokens) != config['expected_final_history']:
                raise ValueError('Unexpected accumulated history length')

        jobs = [asyncio.create_task(owner_task(owner)) for owner in config['owners']]
        try:
            await asyncio.gather(*jobs)
        except BaseException as error:
            failed = error
            errors.append(repr(error))
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        finally:
            finished, elapsed = time.time(), time.perf_counter()-started_monotonic
            after = snapshot(capture)
            observations.append(after)
            save(output/'after-work.json', after)
            for task, session in sessions.items():
                try:
                    response = await client.post('/v1/completions', json=dict(model=model, session_id=session,
                        cache_salt=session, prompt=[config['eos_token']], max_tokens=1,
                        kv_transfer_params=dict(kvmem_release_session=True)))
                    response.raise_for_status()
                    releases.append(dict(task=task, passed=True, response=response.json()))
                except Exception as error:
                    releases.append(dict(task=task, passed=False, error=repr(error)))
            save(output/'releases.json', releases)
    # Cleanup and capture export are outside the measured workload window.
    if capture:
        from scripts.qualify_thinking_gc_preflight import clean
        deadline = time.monotonic()+30
        while not clean(snapshot(capture)) and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        released = snapshot(capture)
        cleanup_passed = clean(released)
        save(output/'after-release.json', released)
    else:
        cleanup_passed = all(r['passed'] for r in releases)
    captured = output/'capture'
    captured.mkdir()
    if capture:
        for name, offset in offsets.items():
            path = capture/name
            if path.exists():
                with path.open('rb') as source, (captured/name).open('wb') as target:
                    source.seek(offset)
                    for line in source:
                        if line.endswith(b'\n'):
                            target.write(line)
    old_gc = before.get('archive', {}).get('historical_gc') or {}
    counters = [(r.get('archive', {}).get('historical_gc') or {}) for r in observations]
    gc_delta = {name: max([old_gc.get(name, 0), *(r.get(name, 0) for r in counters)])-old_gc.get(name, 0)
                for name in COUNTERS}
    accepted = sum(r['accepted_tokens'] for r in records)
    summary = dict(passed=not errors and cleanup_passed and all(r['passed'] for r in releases), label=label,
                   scope=config['scope'], fixture_sha256=digest(config), started_at=started, finished_at=finished,
                   started_monotonic=started_monotonic, finished_monotonic=started_monotonic+elapsed,
                   wall_seconds=elapsed, owners=len(config['owners']), turns=config['turns'],
                   completed_requests=len(records), expected_requests=len(config['owners'])*config['turns']*2,
                   accepted_output_tokens=accepted, completed_tokens_per_second=accepted/elapsed,
                   expected_final_history_per_owner=config['expected_final_history'],
                   transport_policy=transport_policy,
                   input_fingerprints={f"{r['task']}/{r['turn']}/{r['part']}": r['input_sha256'] for r in records},
                   gc_delta=gc_delta, native=native_summary(captured, records), cleanup_passed=cleanup_passed,
                   errors=errors, note='Wall includes prefill and HTTP/token serialization; forced-token decoding is diagnostic, not natural-generation throughput.')
    save(output/'summary.json', summary)
    save(output/'status.json', dict(status='completed' if summary['passed'] else 'failed', at=time.time(), label=label))
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_fingerprints'}, indent=2))
    if failed:
        raise failed
    if not summary['passed']:
        raise RuntimeError('Efficiency probe cleanup or fixed-token contract failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--short-fixture', type=Path, required=True)
    prep.add_argument('--tokenizer', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    for name, default in (('owners', 4), ('turns', 5), ('initial-tokens', 65536), ('history-step', 65536), ('output-tokens', 512)):
        prep.add_argument('--'+name, type=int, default=default)
    run = sub.add_parser('run')
    run.add_argument('--base', required=True)
    run.add_argument('--model', required=True)
    run.add_argument('--fixture', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--capture', type=Path)
    run.add_argument('--label', default='baseline')
    run.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.short_fixture, args.tokenizer, args.output, args.owners, args.turns,
                args.initial_tokens, args.history_step, args.output_tokens)
    else:
        asyncio.run(exercise(args.base, args.model, args.fixture, args.output, args.capture, args.label, args.timeout))
