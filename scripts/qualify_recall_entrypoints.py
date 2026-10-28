"""Real-model functional qualification for all three fixed-recall entrances.

One original long RULER prompt forces G+256 output tokens, then the same logical
session receives an appended user Query. These responses are never task scores.
The common non-retaining profile cold-replays the follow-up's complete prefix;
this script does not claim checkpoint reuse or change checkpoint settings.
"""
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import uuid

import recall_comparison_protocol as p
import run_recall_comparison as campaign


FOLLOWUP_QUESTION = 'For this new user request, reply with exactly the word READY.'


def functional_sampling(fixture, profile, *, followup=False):
    """Only functional output controls differ; the model/selection profile does not."""
    k = profile['kvmem']
    p.require(k.get('rolling_active_window', False),
              'Decode-pressure qualification requires the common rolling_active_window profile')
    count = 64 if followup else k['gen_budget'] + 256
    sampling = dict(fixture['sampling'])
    sampling.update(seed=1000, max_tokens=count, min_tokens=count, ignore_eos=True)
    # min_tokens controls native EOS/stop-token behavior, but stop strings may
    # otherwise terminate a functional request before the pressure boundary.
    sampling.pop('stop', None)
    sampling.pop('stop_token_ids', None)
    return sampling


def append_user_query(initial, generated, tokenizer):
    """Append a genuine Qwen user-message span after the complete first output."""
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    prefix = encode('\n</think>\n<|im_end|>\n<|im_start|>user\n')
    question = encode(FOLLOWUP_QUESTION)
    suffix = encode('<|im_end|>\n<|im_start|>assistant\n<think>\n')
    p.require(question and tokenizer.decode(question, skip_special_tokens=False) == FOLLOWUP_QUESTION,
              'Functional follow-up Query cannot be located exactly')
    tokens = list(initial) + list(generated) + prefix
    span = [len(tokens), len(tokens) + len(question)]
    tokens.extend(question)
    tokens.extend(suffix)
    return tokens, span


def input_bytes(tokens):
    values = array('i', tokens)
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def audit_response(response, tokens, sampling, terminal):
    ids = response['output_ids']
    p.require(response.get('finish_reason') == 'length' and len(ids) == sampling['max_tokens'],
              'Functional output stopped before its forced accepted-token count')
    usage = response.get('usage') or {}
    p.require(usage.get('prompt_tokens') == len(tokens) and usage.get('completion_tokens') == len(ids),
              'Functional response token accounting differs')
    p.require(terminal['history_end'] == len(tokens) + len(ids) - 1,
              'Functional accepted history includes missing or unprocessed token rows')
    p.require(not terminal.get('checkpoint_tokens'),
              'Functional qualification expects the unchanged non-retaining campaign profile')


def audit_entrypoints(long_events, followup_events, method, profile, initial_query,
                      followup_query, initial_tokens):
    first = p.audit_events(long_events, method, profile, query_span=initial_query,
        required_types=p.EVENT_TYPES)
    second = p.audit_events(followup_events, method, profile, query_span=followup_query,
        required_types=('query',))
    p.require(first['event_counts']['query'] == second['event_counts']['query'] == 1,
              'Each functional request requires exactly one declared user Query reselection')
    decode = [event for event in long_events if event['event_type'] == 'decode_pressure']
    p.require(all(event['accepted_through'] >= initial_tokens for event in decode),
              'Decode pressure was labelled before the original public prompt finished')
    merged = p.audit_events([*long_events, *followup_events], method, profile,
        required_types=p.EVENT_TYPES)
    return dict(merged, long_request=first, followup_request=second,
                functional_only=True, scored=False, followup_mode='same-session-cold-full-prefix-replay')


def run_client(root, method, capture, output, *, tokenizer=None, stream_fn=None,
               wait_retired_fn=None, wait_released_fn=None, import_proof=True):
    """Run against an already-owned server; injectable I/O supports CPU tests."""
    from benchmark_client_io import CaptureSlice
    import qualify_graph_niah as client
    root, capture, output = Path(root).resolve(), Path(capture).resolve(), Path(output).resolve()
    fixture, _ = campaign.verify(root)
    profile = p.read(root / 'profiles' / (method + '.json'))
    sampling = functional_sampling(fixture, profile)
    p.require(profile['kvmem'].get('session_checkpoint_bytes') == 1 and
              not profile['kvmem'].get('session_capture_decode', False),
              'Use the unchanged non-retaining formal profile for functional qualification')
    if tokenizer is None:
        from tokenizers import Tokenizer
        path = Path(profile['tokenizer']) / 'tokenizer.json'
        p.require(p.sha(path) == fixture['tokenizer_sha256'], 'Functional tokenizer differs from frozen input')
        tokenizer = Tokenizer.from_file(str(path))
    stream_fn = client.stream if stream_fn is None else stream_fn
    wait_retired_fn = client.wait_retired if wait_retired_fn is None else wait_retired_fn
    wait_released_fn = client.wait_owner_released if wait_released_fn is None else wait_released_fn
    source = p.execution_rows(fixture['rows'])[0]
    initial = campaign.load_tokens(root, fixture, source, tokenizer)
    p.require(len(initial) > profile['kvmem']['select_budget'] + profile['kvmem']['gen_budget'],
              'Functional source must cross Prefill pressure before its original Query')
    output.mkdir(parents=True, exist_ok=False)
    launch = campaign.launch_audit(profile, capture, method)
    p.save(output / 'launch-audit.json', launch)
    session = 'recall-functional-' + method + '-' + uuid.uuid4().hex
    p.save(output / 'manifest.json', dict(method=method, session_id=session, functional_only=True,
        scored=False, source_sample_id=source['id'], source_input_sha256=source['served_sha256'],
        fixture_sha256=p.sha(root / 'fixture.json'),
        profile_sha256=p.sha(root / 'profiles' / (method + '.json')),
        pressure_query_window=128, output_tokens=sampling['max_tokens'],
        followup_mode='same-session-cold-full-prefix-replay'))
    events_by_phase, responses = {}, {}

    def call(phase, tokens, query, parameters):
        p.require(len(tokens) + parameters['max_tokens'] <= profile['max_model_len'],
                  'Functional request exceeds common logical model length')
        directory = output / phase
        directory.mkdir(exist_ok=False)
        payload = input_bytes(tokens)
        (directory / 'input.i32').write_bytes(payload)
        request = dict(session_id=session, capture=str(capture), method=method,
            input_file='input.i32', input_sha256=hashlib.sha256(payload).hexdigest(),
            input_tokens=len(tokens), query_span=query, sampling=parameters, functional_only=True)
        p.save(directory / 'request.json', request)
        cursor = CaptureSlice([capture / 'recall_events.jsonl'])
        response = stream_fn(profile, session, tokens, query, parameters,
                             fixture['request_timeout_seconds'])
        p.save(directory / 'response.json', response)
        terminal = wait_retired_fn(capture, response['response_id'])
        audit_response(response, tokens, parameters, terminal)
        released = wait_released_fn(capture, session, response, terminal)
        events = p.events_for_response(cursor.read(capture / 'recall_events.jsonl'), response, session)
        p.save(directory / 'evidence.json', dict(events=events, terminal=terminal, release=released))
        responses[phase], events_by_phase[phase] = response, events
        p.save(output / 'progress.json', dict(method=method, completed_phase=phase,
            response_id=response['response_id'], accepted_output=len(response['output_ids']), at=time.time()))
        return response

    try:
        first = call('long_generation', initial, source['served_query'], sampling)
        p.audit_events(events_by_phase['long_generation'], method, profile,
            query_span=source['served_query'], required_types=p.EVENT_TYPES)
        followup, query = append_user_query(initial, first['output_ids'], tokenizer)
        call('followup_query', followup, query, functional_sampling(fixture, profile, followup=True))
        audit = audit_entrypoints(events_by_phase['long_generation'], events_by_phase['followup_query'],
            method, profile, source['served_query'], query, len(initial))
        combined = [*events_by_phase['long_generation'], *events_by_phase['followup_query']]
        event_path = output / 'functional-events.jsonl'
        event_path.write_text(''.join(json.dumps(event, separators=(',', ':')) + '\n' for event in combined))
        p.save(output / 'audit.json', audit)
        if import_proof:
            campaign.qualify_events(NS(root=root, method=method, events=event_path))
        p.save(output / 'completed.json', dict(status='passed', **audit,
            events_sha256=p.sha(event_path), fixture_sha256=p.sha(root / 'fixture.json'), at=time.time()))
        return audit
    except BaseException as error:
        p.save(output / 'failure.json', dict(status='failed', error=repr(error), at=time.time(),
            preserved_completed_phases=list(responses), functional_only=True))
        raise


def run(args):
    root = args.root.resolve()
    fixture, _ = campaign.verify(root)
    runtime = (args.runtime or Path(fixture['runtime_source'])).resolve()
    campaign.verify(root, runtime)
    methods = (args.method,) if args.method else p.METHODS
    environment = {key: value for key, value in os.environ.items() if not key.startswith('KVMEM_')}
    environment.update(PYTHONPATH=str(runtime / 'src'), PYTHONDONTWRITEBYTECODE='1',
                       TOKENIZERS_PARALLELISM='false')
    with campaign.termination_guard(), campaign.gpu_lock(args.gpu_lock):
        for method in methods:
            campaign.verify(root, runtime)
            profile = p.read(root / 'profiles' / (method + '.json'))
            functional_sampling(fixture, profile)
            attempt = root / 'functional-server' / method
            attempt.mkdir(parents=True, exist_ok=False)
            command = [str(args.python), '-B', '-m', 'vllm_kvmem', 'serve', '--config',
                str(root / 'profiles' / (method + '.json')), '--run-dir', str(attempt / 'server')]
            environment.update(p.method_environment(method))
            server = None
            try:
                with (attempt / 'server.log').open('x') as log:
                    server = subprocess.Popen(command, cwd=runtime, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                p.save(attempt / 'process.json', dict(pid=server.pid, command=command, method=method,
                    environment=p.method_environment(method), functional_only=True, at=time.time()))
                campaign.wait_health(server, profile)
                run_client(root, method, attempt / 'server/capture', root / 'functional' / method)
            finally:
                campaign.stop_owned(server)
                p.save(attempt / 'cleanup.json', dict(owned_pid=server.pid if server else None,
                    returncode=server.returncode if server else None, at=time.time()))
            campaign.verify(root, runtime)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest='action', required=True)
    client = sub.add_parser('client')
    client.add_argument('--root', type=Path, required=True)
    client.add_argument('--method', choices=p.METHODS, required=True)
    client.add_argument('--capture', type=Path, required=True)
    client.add_argument('--output', type=Path, required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--root', type=Path, required=True)
    runner.add_argument('--method', choices=p.METHODS)
    runner.add_argument('--runtime', type=Path)
    runner.add_argument('--python', type=Path, default=Path(sys.executable))
    runner.add_argument('--gpu-lock', type=Path, default=Path('/tmp/kvmem-recall-comparison.lock'))
    return command


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.action == 'client':
        run_client(arguments.root, arguments.method, arguments.capture, arguments.output)
    else:
        run(arguments)
