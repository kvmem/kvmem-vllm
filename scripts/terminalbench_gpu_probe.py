"""Fixed-token Query continuation probe against an already owned model server.

The initial prompts and appended observations come from preserved TB traces;
outputs are forced to token 830. This is a component test, never task scoring.
The parent controller owns GPU scheduling and the server lifecycle.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx
from vllm_kvmem.core.request_records import load_request


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def prepare(source, tokenizer_path, protocol_path, output, tasks, turns=4):
    from transformers import AutoTokenizer
    from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory
    from scripts.dsh_bridge import to_anthropic
    from scripts.terminalbench_replay import replay
    if output.exists():
        raise ValueError('Use a new probe fixture path')
    evidence = replay(source, max_owners=len(tasks), max_turns=turns, task_names=tasks)
    if {row['task'] for row in evidence['traces']} != set(tasks):
        raise ValueError('Every selected probe task needs a complete source trace')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    planner = Planner(Protocol(protocol_path), tokenizer, 16384)
    values = []
    for row in evidence['traces']:
        directory = source/'bridge'/row['owner']
        body = load_request(directory/'00001/request.json')
        prepared = planner.prepare(to_anthropic(body), Trajectory())
        prefix = list(prepared[0])
        query = prepared[1]['kvmem_query_span']
        additions = []
        for turn in range(2, turns+1):
            request = load_request(directory/f'{turn:05d}'/'request.json')
            users = [message['content'] for message in request['messages'] if message['role']=='user']
            if not users or not isinstance(users[-1], str):
                raise ValueError('Probe requires an exact textual terminal observation')
            # This explicit token concatenation is identical for both arms.
            # Limit diagnostic input only, never any official task request.
            head = tokenizer.encode('</think><|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
            text = tokenizer.encode(users[-1], add_special_tokens=False)[:1024]
            tail = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n', add_special_tokens=False)
            if not text:
                text = [830]
            additions.append(dict(tokens=head+text+tail, query_offset=[len(head), len(head)+len(text)]))
        values.append(dict(task=row['task'], prefix=prefix, query=query, additions=additions))
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, dict(scope='Fixed-token component probe; not official task execution',
                      source=str(source), source_evidence=evidence['evidence_sha256'], owners=values,
                      output_token=830, output_tokens=64,
                      tokenizer=str(tokenizer_path), protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest()))
    print(json.dumps(dict(owners=len(values), requests_per_arm=len(values)*turns,
                          initial_tokens=[len(value['prefix']) for value in values])))


async def exercise(base, model, fixture, output, capture=None, order=('legacy','task')):
    if output.exists():
        raise ValueError('Use a new GPU probe evidence directory')
    output.mkdir(parents=True)
    config = read(fixture)
    save(output/'fixture.json', config)
    records, release_errors, arm_times = [], [], {}
    capture_files = ('async_restore.jsonl','scheduler_refresh.jsonl','async_refresh.jsonl','scheduler_finished.jsonl')
    offsets = {name: (capture/name).stat().st_size if capture and (capture/name).exists() else 0
               for name in capture_files}
    async with httpx.AsyncClient(base_url=base, trust_env=False, timeout=1800) as client:
        for arm in order:
            arm_started=time.perf_counter()
            sessions = {value['task']: 'tb-probe-'+uuid.uuid4().hex for value in config['owners']}
            async def owner(value):
                tokens = list(value['prefix'])
                initial_query = list(value['query'])
                session = sessions[value['task']]
                for turn in range(len(value['additions'])+1):
                    query = initial_query
                    if turn:
                        addition = value['additions'][turn-1]
                        if arm=='legacy':
                            query = [len(tokens)+offset for offset in addition['query_offset']]
                        tokens.extend(addition['tokens'])
                    payload = dict(model=model, session_id=session, cache_salt=session, prompt=tokens,
                                   max_tokens=config['output_tokens'], temperature=0, seed=1000,
                                   allowed_token_ids=[config['output_token']], ignore_eos=True, return_token_ids=True,
                                   kvmem_reselect='auto', kvmem_query_span=query,
                                   kvmem_mandatory_parts='none', kvmem_mandatory_spans=[])
                    if arm=='task':
                        payload.update(kvmem_query_mode='agent', kvmem_query_id='fixed-task-'+value['task'])
                    start = time.perf_counter()
                    result = await client.post('/v1/completions', json=payload)
                    result.raise_for_status()
                    data = result.json()
                    accepted = data['choices'][0]['token_ids']
                    if accepted != [config['output_token']]*config['output_tokens']:
                        raise ValueError('Probe fixed output was not accepted exactly')
                    record = dict(arm=arm, task=value['task'], turn=turn+1, request_id=data['id'],
                                  prompt_tokens=len(tokens), input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                  query=query, seconds=time.perf_counter()-start,
                                  accepted=len(accepted), response=data)
                    records.append(record)
                    save(output/'requests.json', records)
                    tokens.extend(accepted)
            jobs=[asyncio.create_task(owner(value)) for value in config['owners']]
            try:
                await asyncio.gather(*jobs)
                arm_times[arm]=time.perf_counter()-arm_started
            except BaseException:
                for job in jobs:
                    job.cancel()
                await asyncio.gather(*jobs,return_exceptions=True)
                raise
            finally:
                for task, session in sessions.items():
                    try:
                        result = await client.post('/v1/completions', json=dict(model=model, session_id=session,
                            cache_salt=session,prompt=[151645],max_tokens=1,kv_transfer_params=dict(kvmem_release_session=True)))
                        result.raise_for_status()
                        save(output/f'{arm}-{task}-release.json',result.json())
                    except Exception as error:
                        release_errors.append(dict(arm=arm,task=task,error=repr(error)))
                save(output/'release-errors.json',release_errors)
    pairs = {}
    for row in records:
        pairs.setdefault((row['task'],row['turn']),set()).add(row['input_sha256'])
    if any(len(values)!=1 for values in pairs.values()):
        raise ValueError('Compared probe arms received different input tokens')
    captured={name:[] for name in capture_files}
    if capture:
        for name in capture_files:
            path=capture/name
            if not path.exists():
                continue
            with path.open('rb') as stream:
                stream.seek(offsets[name])
                for line in stream:
                    if line.strip():
                        value=json.loads(line)
                        matching=[row for row in records if value.get('request_id','').startswith(row['request_id']+'-')]
                        if matching:
                            captured[name].append(dict(arm=matching[0]['arm'],task=matching[0]['task'],turn=matching[0]['turn'],**value))
    restores=captured['async_restore.jsonl']
    save(output/'native-evidence.json',captured)
    summary={}
    for arm in order:
        rows=[row for row in records if row['arm']==arm and row['turn']>1]
        restored=[row for row in restores if row['arm']==arm]
        refresh=[row for row in captured['scheduler_refresh.jsonl'] if row['arm']==arm]
        rewind=[]
        for row in restored:
            before=next((request for request in records if request['arm']==arm and request['task']==row['task'] and
                         request['turn']==row['turn']-1),None)
            if before:
                # Native checkpoint may exclude the final uncomputed accepted
                # token. Report distances without assuming every distance is a bug.
                rewind.append(max(0,before['prompt_tokens']+before['accepted']-1-row['history_tokens']))
        summary[arm]=dict(continuations=len(rows),wall_seconds=arm_times[arm],
                          continuation_request_seconds=sum(row['seconds'] for row in rows),
                          restore_lanes=dict(Counter(row.get('lane') for row in restored)),
                          uploaded_bytes=sum(row.get('uploaded_bytes',0) for row in restored),
                          model_uploaded_bytes=sum(row.get('model_uploaded_bytes',0) for row in restored),
                          query_replays=len(refresh),query_replay_tokens=sum(row.get('replay_tokens',0) for row in refresh),
                          checkpoint_rewind_tokens=sum(rewind),checkpoint_rewind_events=sum(x>0 for x in rewind))
    save(output/'summary.json',dict(passed=not release_errors, identical_input_tokens=True, fixed_outputs=True,
        comparison=summary, scope='Component request latency sums overlap across owners; not end-to-end throughput or task quality'))
    if release_errors:
        raise RuntimeError('Probe session cleanup failed')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source',type=Path,required=True)
    prep.add_argument('--tokenizer',type=Path,required=True);prep.add_argument('--protocol',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--tasks',nargs='+',required=True)
    prep.add_argument('--turns',type=int,default=4)
    run=sub.add_parser('run');run.add_argument('--base',required=True);run.add_argument('--model',required=True)
    run.add_argument('--fixture',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--capture',type=Path);run.add_argument('--task-first',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.source,args.tokenizer,args.protocol,args.output,args.tasks,args.turns)
    else:
        asyncio.run(exercise(args.base,args.model,args.fixture,args.output,args.capture,
                             ('task','legacy') if args.task_first else ('legacy','task')))
