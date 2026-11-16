"""Freeze an offline performance trace from complete DeepSWE request records.

Selection depends only on manifest order, logical input length and availability
of consecutive complete turns. Source rewards and timing never select samples.
The initial input is a NEW canonical rendering, not the original run's exact
accepted-output tokenization. Future outputs must come from the tested model;
only external tool/message additions, generation lengths and waits are frozen.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from vllm_kvmem.core.request_records import load_request


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja',
                   'config.json', 'tokenizer.model', 'spiece.model')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def journal_index(path):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('event') not in ('start', 'finish', 'error', 'cancelled') or 'turn' not in row:
            continue
        key = row['task'], row['turn'], row['event']
        if key in result:
            raise ValueError(f'Duplicate source journal event: {key}')
        result[key] = row
    return result


def _complete_paths(directory, start, turns, task_id, events):
    paths = [directory/f'{n:05d}' for n in range(start,start+turns)]
    complete = all(all((path/name).is_file() for name in
                   ('request.json','response.json','native.json','plan.json'))
                   and (task_id,int(path.name),'start') in events
                   and (task_id,int(path.name),'finish') in events
                   and not any((task_id,int(path.name),event) in events for event in ('error','cancelled'))
                   for path in paths)
    return paths if complete else None


def choose_sources(run, users, threshold, turns, *, allow_shorter_history_fallback=False,
                   max_history=None):
    if any(type(v) is not int or v <= 0 for v in (users, threshold, turns)):
        raise ValueError('users, history threshold and turns must be positive integers')
    if type(allow_shorter_history_fallback) is not bool:
        raise ValueError('Shorter-history fallback must be explicitly boolean')
    if max_history is not None and (type(max_history) is not int or max_history < threshold):
        raise ValueError('Maximum initial history must be an integer at least the preferred threshold')
    manifest = read(run/'manifest.json')
    state = read(run/'run.json')
    tasks = manifest['tasks'][:users]
    if len(tasks) != users:
        raise ValueError('Manifest contains fewer tasks than requested users')
    events = journal_index(run/'bridge/requests.jsonl')
    chosen = []
    for task in tasks:
        task_id = task['task_id']
        status = state['tasks'].get(task_id, {})
        if status.get('state') != 'completed' or status.get('termination') != 'natural':
            raise ValueError(f'{task_id}: original manifest task did not complete naturally')
        directory = run/'bridge'/task_id
        candidates = sorted((p for p in directory.iterdir() if p.is_dir() and p.name.isdigit()),
                            key=lambda p: int(p.name))
        selected = None
        skipped = []
        selection = None
        for first in candidates:
            plan_path = first/'plan.json'
            if not plan_path.exists():
                continue
            length = read(plan_path).get('prompt_tokens',0)
            if length < threshold:continue
            start = int(first.name)
            if max_history is not None and length > max_history:
                skipped.append(dict(source_turn=start,reason='logical initial history exceeds configured maximum'))
                continue
            paths = _complete_paths(directory,start,turns,task_id,events)
            if paths is None:
                skipped.append(dict(source_turn=start, reason='not enough consecutive complete source turns'))
                continue
            selected = paths
            selection = dict(kind='preferred',source_turn=start,source_logical_initial_tokens=length,
                             fallback_reason=None)
            break
        if selected is None and allow_shorter_history_fallback:
            # Only input length and complete consecutive records rank fallback
            # candidates. Rewards, generation lengths and timing cannot rank them.
            fallback = []
            for first in candidates:
                if not (first/'plan.json').is_file():continue
                length = read(first/'plan.json').get('prompt_tokens',0)
                if type(length) is not int or not 0 < length < threshold:continue
                start = int(first.name)
                paths = _complete_paths(directory,start,turns,task_id,events)
                if paths is None:
                    skipped.append(dict(source_turn=start,reason='not enough consecutive complete source turns'))
                    continue
                fallback.append((length,start,paths))
            if fallback:
                length,start,selected = min(fallback,key=lambda item:(-item[0],item[1]))
                selection = dict(kind='shorter_history_fallback',source_turn=start,
                    source_logical_initial_tokens=length,
                    fallback_reason='No preferred input at or above the threshold within the initial-history '
                        'ceiling has the required consecutive complete turns; select the largest shorter '
                        'logical input with complete turns, earliest turn on ties.')
        if selected is None:
            extra = ' and no permitted shorter-history fallback' if allow_shorter_history_fallback else ''
            raise ValueError(f'{task_id}: no input >= {threshold} with {turns} consecutive complete turns{extra}')
        rows = []
        for path in selected:
            turn = int(path.name)
            # request.json may hold only the turn's changes; the frozen trace keeps the whole request.
            row = {name: load_request(path/'request.json') if name == 'request' else read(path/(name+'.json'))
                   for name in ('request','response','native','plan')}
            row.update(source_turn=turn,path=path,
                       start=events[task_id,turn,'start'],finish=events[task_id,turn,'finish'])
            output = row['native']['usage']['completion_tokens']
            if type(output) is not int or output <= 0:
                raise ValueError(f'{task_id} turn {turn}: invalid source output length')
            if (len(row['native']['choices']) != 1 or
                    len(row['native']['choices'][0]['token_ids']) != output or
                    row['finish']['usage']['completion_tokens'] != output or
                    row['finish']['request_id'] != row['native']['id'] or
                    row['plan']['prompt_tokens'] != row['start']['prompt_tokens'] or
                    row['plan']['prompt_tokens'] != row['native']['usage']['prompt_tokens']):
                raise ValueError(f'{task_id} turn {turn}: source completion/identity records disagree')
            rows.append(row)
        source = dict(task=task_id,rows=rows,skipped_incomplete_candidates=skipped)
        if allow_shorter_history_fallback or max_history is not None:
            source['selection'] = dict(selection,original_criterion=dict(
                history_threshold=threshold,max_initial_history=max_history,
                consecutive_complete_turns=turns,start='earliest qualifying logical prompt'))
        chosen.append(source)
    return chosen


def external_suffix(previous, current, protocol, message_identity):
    """Extract new non-assistant message bytes using the frozen protocol spans."""
    before, after = previous['rendered'], current['rendered']
    old_messages, new_messages = before['messages'], after['messages']
    count = len(old_messages)
    if len(new_messages) <= count+1:
        raise ValueError('Source transition is not append-only after the prior assistant')
    before_bytes, after_bytes = before['prompt'].encode(), after['prompt'].encode()
    if (before_bytes[:before['control_end']] != after_bytes[:after['control_end']] or
            before['tools'] != after['tools']):
        raise ValueError('Source control prefix or tool definitions changed')
    if list(map(message_identity, new_messages[:count])) != list(map(message_identity, old_messages)):
        raise ValueError('Source historical message identity changed')
    expected = protocol(request={**previous['request'], 'messages': previous['request']['messages'] +
                        [{'role': 'assistant', 'content': previous['response']['content']}]})
    if message_identity(new_messages[count]) != message_identity(expected['messages'][-1]):
        raise ValueError('Source returned assistant identity differs from the completed response')
    if any(message.get('role') == 'assistant' for message in new_messages[count+1:]):
        raise ValueError('External addition contains another source assistant answer')
    if before['query_index'] != after['query_index']:
        raise ValueError('Source Query changed; a fixed initial Query plan is insufficient')
    spans = [span for span in after['spans'] if span['index'] >= count+1]
    if [span['index'] for span in spans] != list(range(count+1, len(new_messages))):
        raise ValueError('Protocol spans do not cover every new external message')
    boundary = spans[0]['begin']
    end = boundary
    for span in spans:
        if span['begin'] != end or not span['begin'] < span['end'] <= len(after_bytes):
            raise ValueError('New protocol message spans are not contiguous valid byte ranges')
        end = span['end']
    for pin in after['pins']:
        if pin['end'] > boundary and pin['reason'] not in (2, 5, 6):
            raise ValueError('New mandatory content requires a changing request plan')
    # Match Planner's exact-output continuation delimiter. This includes new
    # message framing and the next assistant-generation header, never the prior
    # assistant's generated bytes. It is tokenized independently for this NEW
    # workload and is not claimed to reconstruct the old accepted token prefix.
    suffix = '\n' + after_bytes[boundary:].decode('utf-8')
    proof = dict(source_byte_begin=boundary, source_byte_end=len(after_bytes),
                 prepended_newline=True, message_indices=[s['index'] for s in spans],
                 message_roles=[m['role'] for m in new_messages[count+1:]],
                 utf8_bytes=len(suffix.encode()), utf8_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                 identity_contract='Frozen claude_bridge.message_identity; no unrecognized changes accepted')
    return suffix, proof


def locality_mapper():
    # The source bridge/request planner may intentionally be frozen V1 code.
    # Load this new metadata-only helper independently, preserving that source
    # renderer and all of its original token/Query/mandatory behavior.
    name = 'timed_trace_locality_mapper'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/'src/vllm_kvmem/serving/prompt_spans.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def rendered_locality(planner, text, spans, tokens, *, source):
    mapper = locality_mapper()
    intervals = [dict(begin=s['begin'], end=s['end'], role=s['role']) for s in spans]
    kwargs = dict(tokenizer=getattr(planner, 'tokenizer', None), source=source,
                  template=getattr(planner.protocol, 'template', 'frozen_qw3_claude_code'))
    mapped = mapper.map_prompt_spans(text, intervals, tokens, **kwargs)
    header = max((s['end'] for s in spans), default=0)
    assistant_begin = None
    if text.encode()[header:].startswith(b'<|im_start|>assistant\n'):
        pending = mapper.map_prompt_spans(text,
            [dict(begin=header, end=len(text.encode()), role='assistant')], tokens, **kwargs)
        if pending.spans:
            assistant_begin = pending.spans[0][0]
    return mapped, assistant_begin


def make_agent_plan(source, planner, bridge):
    first = source['rows'][0]
    prepared = planner.prepare(first['request'], bridge.Trajectory())
    initial, query_plan, rendered = prepared[:3]
    if prepared[5]:
        raise ValueError('Initial seed must use an empty-state canonical rendering')
    first['rendered'] = rendered
    mapped, assistant_begin = rendered_locality(planner, rendered['prompt'], rendered['spans'],
        initial, source='frozen_deepswe_initial')
    query_plan = {**query_plan, **mapped.payload()}
    output = []
    previous = None
    for row in source['rows']:
        if previous is None:
            tool_tokens, wait, proof = [], 0., None
            tool_metadata, next_assistant_begin = None, None
        else:
            row['rendered'] = planner.protocol(request=row['request'])
            try:
                suffix, proof = external_suffix(previous, row, planner.protocol, bridge.message_identity)
            except ValueError as error:
                raise ValueError(f"{source['task']} turn {row['source_turn']}: {error}") from error
            tool_tokens = planner.encode(suffix)
            relative_spans = [dict(begin=s['begin'] - proof['source_byte_begin'] + 1,
                                   end=s['end'] - proof['source_byte_begin'] + 1, role=s['role'])
                              for s in row['rendered']['spans']
                              if s['index'] in proof['message_indices']]
            tool_metadata, next_assistant_begin = rendered_locality(planner, suffix, relative_spans,
                tool_tokens, source='frozen_deepswe_external_append')
            wait = row['start']['request_started_at'] - previous['finish']['at']
            if not math.isfinite(wait) or wait < 0:
                raise ValueError(f"{source['task']} turn {row['source_turn']}: negative/nonfinite tool wait")
        output.append(dict(source_turn=row['source_turn'],
                           output_tokens=row['native']['usage']['completion_tokens'],
                           tool_tokens=tool_tokens, tool_wait_seconds=wait,
                           source_finish_reason=row['native']['choices'][0]['finish_reason'],
                           source_logical_prompt_tokens=row['plan']['prompt_tokens'],
                           tool_tokens_sha256=ids_sha(tool_tokens), external_message_proof=proof,
                           tool_locality_spans=[] if tool_metadata is None else
                               locality_mapper().span_payload(tool_metadata.spans),
                           tool_locality_provenance={} if tool_metadata is None else dict(tool_metadata.provenance),
                           assistant_begin=next_assistant_begin))
        previous = row
    return dict(source_task=source['task'], source_turn=first['source_turn'],
                initial_tokens=initial, initial_tokens_sha256=ids_sha(initial),
                canonical_initial_tokens=len(initial),
                source_logical_initial_tokens=first['plan']['prompt_tokens'],
                query_plan=query_plan, turns=output, initial_assistant_begin=assistant_begin,
                skipped_incomplete_candidates=source['skipped_incomplete_candidates'])


def distribution(values):
    return dict(count=len(values), minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), median=statistics.median(values))


def describe(agents):
    return dict(users=len(agents),
                initial_tokens=[len(a['initial_tokens']) for a in agents],
                source_starts=[dict(task=a['source_task'], turn=a['source_turn']) for a in agents],
                output_tokens=distribution([t['output_tokens'] for a in agents for t in a['turns']]),
                external_tool_tokens=distribution([len(t['tool_tokens']) for a in agents for t in a['turns'][1:]]),
                tool_wait_seconds=distribution([t['tool_wait_seconds'] for a in agents for t in a['turns'][1:]]))


def validate_initial_history(agents,max_history):
    if max_history is not None:
        for agent in agents:
            if len(agent['initial_tokens']) > max_history:
                raise ValueError(f"{agent['source_task']}: canonical initial history {len(agent['initial_tokens'])} "
                    f'exceeds maximum {max_history}; no clipping or replacement candidate applied')


def freeze(args):
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Refusing to replace an already frozen trace')
    if args.turns < 2:
        raise ValueError('At least two turns are needed for a cross-round trace')
    fallback = getattr(args,'allow_shorter_history_fallback',False)
    max_history = getattr(args,'max_history',None)
    sources = choose_sources(run,args.users,args.history_threshold,args.turns,
        allow_shorter_history_fallback=fallback,max_history=max_history)
    bridge_path = (args.bridge or run.parent/'frozen/benchmarks/deepswe/claude_bridge.py').resolve()
    if not bridge_path.is_file():
        raise FileNotFoundError('Provide the source run frozen bridge using --bridge')
    frozen_src = run.parent/'frozen/src'
    if not frozen_src.is_dir():
        raise FileNotFoundError('Source run frozen request planner is missing')
    sys.path.insert(0, str(frozen_src))
    spec = importlib.util.spec_from_file_location('timed_trace_source_bridge', bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    from transformers import AutoTokenizer
    import transformers
    import tokenizers
    profile = read(run/'profile.json')
    tokenizer_path = Path(args.tokenizer or profile['tokenizer']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    library = run/'protocol/protocol.so'
    planner = bridge.Planner(bridge.Protocol(library), tokenizer, gen_budget=131072)
    agents = [make_agent_plan(source, planner, bridge) for source in sources]
    validate_initial_history(agents,max_history)
    request_plan = sys.modules['vllm_kvmem.serving.request_plan']
    if not Path(request_plan.__file__).resolve().is_relative_to(frozen_src.resolve()):
        raise ValueError('RequestPlan was imported from outside the source frozen tree')
    paths = {run/name for name in ('manifest.json', 'run.json', 'profile.json', 'bridge/requests.jsonl')}
    paths.update((bridge_path, Path(request_plan.__file__).resolve(), Path(__file__).resolve(),
                  ROOT/'src/vllm_kvmem/serving/prompt_spans.py'))
    paths.update(path for path in (run/'protocol').rglob('*') if path.is_file())
    for source in sources:
        if fallback or max_history is not None:
            # Preserve every input-length record consulted by the explicit
            # fallback policy, not just the eventual winning trace.
            paths.update(path for path in (run/'bridge'/source['task']).glob('*/plan.json')
                         if path.parent.name.isdigit())
        for row in source['rows']:
            paths.update(row['path']/(name+'.json') for name in ('request', 'response', 'native', 'plan'))
    tokenizer_paths = [tokenizer_path/name for name in TOKENIZER_FILES if (tokenizer_path/name).is_file()]
    if not (tokenizer_path/'tokenizer.json').is_file():
        raise ValueError('This provenance format requires the frozen local tokenizer.json')
    provenance = dict(source_run=str(run), source_file_sha256={str(p): sha(p) for p in sorted(paths)},
                      tokenizer_directory=str(tokenizer_path),
                      tokenizer_file_sha256={str(p): sha(p) for p in tokenizer_paths},
                      tokenizer_class=type(tokenizer).__name__,
                      library_versions=dict(transformers=transformers.__version__, tokenizers=tokenizers.__version__))
    artifact = dict(schema_version=2,
        scope='Fixed-length external-message performance trace; not original DeepSWE task quality',
        selection=dict(users=args.users, manifest_order='first users without reward/timing selection',
                       history_threshold=args.history_threshold, turns=args.turns,
                       start='earliest qualifying logical prompt with consecutive complete turns'),
        semantics=dict(initial='New empty-state canonical Protocol/Planner rendering; not source accepted-history tokenization',
                       continuation='Append each new model response own accepted IDs and frozen external message tokens',
                       outputs='Source completion lengths only; no source answer substituted for new model output',
                       tool_wait='Next request_started_at minus previous finish.at; no clipping',
                       tools='Recorded external results independent of newly generated tool choices',
                       eos='Fixed output lengths are performance quotas; downstream harness must declare stop/EOS policy',
                       quality='Does not measure original task correctness or an on-policy tool agent',
                       locality='Graph-only full-encoding offsets; tool spans are relative to tool_tokens only. '
                           'Add actual accepted history length once; assistant intervals close only at actual EOS. '
                           'Missing/ambiguous offsets stay unknown; V1 frozen inputs are never rewritten.'),
        agent_plans=agents, summary=describe(agents), provenance=provenance)
    if fallback or max_history is not None:
        artifact['selection'].update(allow_shorter_history_fallback=fallback,max_initial_history=max_history,
            fallback_policy='Largest logical initial prompt below the preferred threshold with required '
                'complete consecutive turns; earliest turn on ties. Original first users are never replaced.',
            canonical_capacity_policy='Fail if selected canonical initial tokens exceed max_initial_history; '
                'never truncate, pad, skip or replace a selected user.',
            decisions=[dict(task=source['task'],**source['selection'],
                canonical_initial_tokens=len(agent['initial_tokens'])) for source,agent in zip(sources,agents)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(artifact, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--history-threshold', type=int, default=131072)
    parser.add_argument('--turns', type=int, default=8)
    parser.add_argument('--allow-shorter-history-fallback',action='store_true',
        help='Keep the same first users; if no preferred complete trace exists, use their longest shorter input')
    parser.add_argument('--max-history',type=int,
        help='Maximum logical and canonical initial length; an oversized canonical input fails without clipping')
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    result = freeze(parser.parse_args())
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
