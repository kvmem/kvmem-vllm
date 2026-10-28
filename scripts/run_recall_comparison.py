"""Prepare and run fresh, paired RULER recall experiments on one owned GPU.

The controller never retries a scored answer. A completed response is immutable:
an interrupted audit is resumed from that response, while retrying an unfinished
technical request requires --retry-technical and keeps its original attempt.
"""
import argparse
from array import array
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

import recall_comparison_protocol as p
import ruler_protocol as ruler


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def prepare(args):
    root = args.root.resolve()
    p.require(not root.exists(), 'Prepare a new campaign directory')
    source_path = args.source.resolve()
    if source_path.is_dir():
        source_path /= 'fixture.json'
    source = p.validate_fixture(p.read(source_path))
    profile = p.validate_profile(p.read(args.config))
    inputs = (args.inputs or Path(source['input_directory'])).resolve()
    upstream = (args.upstream or Path(source['upstream_path'])).resolve()
    ruler.load_scorer(upstream)
    tokenizer = (args.tokenizer_source or Path(profile['tokenizer'])).resolve() / 'tokenizer.json'
    p.require(p.sha(tokenizer) == source['tokenizer_sha256'], 'Tokenizer differs from original frozen token IDs')
    root.mkdir(parents=True)
    (root / 'dataset/inputs').mkdir(parents=True)
    for row in source['rows']:
        path = inputs / (row['id'] + '.i32')
        p.require(path.stat().st_size == row['served_input_tokens'] * 4 and
                  p.sha(path) == row['served_sha256'], 'Original input changed: ' + row['id'])
        shutil.copy2(path, root / 'dataset/inputs' / path.name)
    for relative in ruler.SOURCE_HASHES:
        target = root / 'upstream' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream / relative, target)
    p.save(root / 'profile.json', profile)
    for method in p.METHODS:
        p.save(root / 'profiles' / (method + '.json'), p.method_profile(profile, method))
    fixture = deepcopy(source)
    fixture.update(execution_profile=p.VERSION, input_directory='dataset/inputs', upstream_path='upstream',
        profile_sha256=p.sha(root / 'profile.json'), source_fixture_sha256=p.sha(source_path),
        source_fixture_path=str(source_path), methods=list(p.METHODS), pressure_query_window=p.PRESSURE_WINDOW,
        qualification_ids=[row['id'] for row in p.execution_rows(source['rows'])[:10]],
        workers=1, request_timeout_seconds=args.request_timeout,
        runtime_source=str(args.runtime.resolve()),
        experiment_scope='All three entrances use the named method, fixed accepted tail Q128 for pressure; '
            'original 260 inputs and sampling; no old answers inherited.')
    p.save(root / 'fixture.json', fixture)
    p.save(root / 'source-proof.json', p.source_manifest(args.runtime))
    p.save(root / 'prepared.json', dict(version=p.VERSION, cases=260, methods=list(p.METHODS),
        formal_runs=1300, pressure_query_window=p.PRESSURE_WINDOW,
        fixture_sha256=p.sha(root / 'fixture.json'), profile_sha256=p.sha(root / 'profile.json'),
        method_profile_sha256={method: p.sha(root / 'profiles' / (method + '.json')) for method in p.METHODS},
        source_proof_sha256=p.sha(root / 'source-proof.json'), prepared_at=time.time()))
    print(json.dumps(dict(status='prepared', root=str(root), formal_runs=1300)), flush=True)


def verify(root, runtime=None):
    fixture = p.validate_fixture(p.read(root / 'fixture.json'))
    profile = p.validate_profile(p.read(root / 'profile.json'))
    proof = p.read(root / 'prepared.json')
    for name, key in (('fixture.json', 'fixture_sha256'), ('profile.json', 'profile_sha256'),
                      ('source-proof.json', 'source_proof_sha256')):
        p.require(p.sha(root / name) == proof[key], 'Frozen campaign file changed: ' + name)
    p.require(fixture['methods'] == list(p.METHODS) and fixture['pressure_query_window'] == 128,
              'Frozen method/window matrix differs')
    for method in p.METHODS:
        arm_path = root / 'profiles' / (method + '.json')
        p.require(p.sha(arm_path) == proof['method_profile_sha256'][method] and
                  p.read(arm_path) == p.method_profile(profile, method), 'Method profiles differ beyond recall method')
    if runtime is not None:
        p.verify_sources(runtime, p.read(root / 'source-proof.json'))
    binary_path = root / 'binary-proof.json'
    if binary_path.exists():
        binary = p.read(binary_path)
        p.require(binary['source_proof_sha256'] == proof['source_proof_sha256'] and
                  binary['fixture_sha256'] == proof['fixture_sha256'], 'Native build is bound to other sources/cohort')
        p.verify_binaries(runtime or binary['runtime'], binary['files'])
    return fixture, profile


def bind_binaries(args):
    root, runtime = args.root.resolve(), args.runtime.resolve()
    p.require(not (root / 'binary-proof.json').exists(), 'Existing binary proof is immutable; do not rebind a running campaign')
    verify(root, runtime)
    files = p.binary_manifest(runtime, args.binary or ())
    p.require(files, 'No compiled binaries found; pass their explicit paths with --binary')
    proof = dict(version=1, runtime=str(runtime), files=files,
        source_proof_sha256=p.sha(root / 'source-proof.json'), fixture_sha256=p.sha(root / 'fixture.json'),
        bound_at=time.time(), scope='Target-host binaries built from the frozen implementation; unchanged for all methods.')
    p.save(root / 'binary-proof.json', proof)
    print(json.dumps(dict(status='binary-bound', files=len(files))), flush=True)


def phase_rows(fixture, phase):
    rows = p.execution_rows(fixture['rows'])
    if phase == 'qualification':
        wanted = set(fixture['qualification_ids'])
        rows = [row for row in rows if row['id'] in wanted]
    return rows


def read_events(path):
    if not Path(path).exists():
        return []
    data = Path(path).read_text()
    p.require(not data or data.endswith('\n'), 'Incomplete recall event log')
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def qualify_events(args):
    """Import real functional-test events; no task answers enter formal scores."""
    root = args.root.resolve()
    fixture, profile = verify(root)
    profile = p.read(root / 'profiles' / (args.method + '.json'))
    events = read_events(args.events)
    audit = p.audit_events(events, args.method, profile, required_types=p.EVENT_TYPES)
    target = root / 'qualification' / args.method
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.events, target / 'functional-events.jsonl')
    p.save(target / 'functional-proof.json', dict(audit, functional_only=True,
        source_events_sha256=p.sha(target / 'functional-events.jsonl'),
        fixture_sha256=p.sha(root / 'fixture.json'), imported_at=time.time()))
    print(json.dumps(audit), flush=True)


def functional_qualified(root, method, profile):
    directory = root / 'qualification' / method
    if not (directory / 'functional-proof.json').exists():
        return False
    proof = p.read(directory / 'functional-proof.json')
    p.require(proof['fixture_sha256'] == p.sha(root / 'fixture.json') and
              proof['source_events_sha256'] == p.sha(directory / 'functional-events.jsonl'),
              'Functional qualification evidence changed')
    p.audit_events(read_events(directory / 'functional-events.jsonl'), method, profile,
                   required_types=p.EVENT_TYPES)
    return True


def qualification_gate(root, fixture, profile):
    for method in p.METHODS:
        p.require(functional_qualified(root, method, profile), 'Missing three-entrance functional qualification')
        for ident in fixture['qualification_ids']:
            result = p.read(root / 'arms' / method / 'qualification/cases' / ident / 'result.json')
            p.require(result['status'] == 'completed' and result['method'] == method and
                      result['fixture_sha256'] == p.sha(root / 'fixture.json'),
                      'Missing immutable ten-case qualification result')


def launch_audit(profile, capture, method):
    launch = p.read(capture.parent / 'launch.json')
    argv, env = launch['vllm_argv'], launch['environment']
    for key, value in p.method_environment(method).items():
        p.require(env.get(key) == value, 'Actual worker recall setting differs: ' + key)
    p.require(env.get('KVMEM_PREFILL_CHUNK_TOKENS') == str(profile['prefill_chunk_tokens']),
              'Actual prefill chunk differs')
    for key, value in (('--max-num-seqs', profile['max_num_seqs']),
                       ('--max-num-batched-tokens', profile['max_num_batched_tokens'])):
        p.require(argv.count(key) == 1 and argv[argv.index(key) + 1] == str(value),
                  'Actual engine limit differs: ' + key)
    return dict(passed=True, environment=p.method_environment(method),
                launch_sha256=p.sha(capture.parent / 'launch.json'))


def load_tokens(root, fixture, row, tokenizer):
    path = resolve(root, fixture['input_directory']) / (row['id'] + '.i32')
    p.require(p.sha(path) == row['served_sha256'], 'Frozen original input changed')
    tokens = array('i')
    tokens.frombytes(path.read_bytes())
    if sys.byteorder != 'little':
        tokens.byteswap()
    p.require(len(tokens) == row['served_input_tokens'] and
              tokenizer.decode(tokens[slice(*row['served_query'])], skip_special_tokens=False) == row['question'],
              'Original Query coordinate/tokenizer mismatch')
    return tokens.tolist()


def saved_result(case, row, method, root):
    path = case / 'result.json'
    if not path.exists():
        return None
    result = p.read(path)
    p.require(result['status'] == 'completed' and result['id'] == row['id'] and result['method'] == method and
              result['input_sha256'] == row['served_sha256'] and
              result['fixture_sha256'] == p.sha(root / 'fixture.json'), 'Existing result has another identity')
    response_path = case / result['attempt'] / 'response.json'
    p.require(p.sha(response_path) == result['response_sha256'], 'Saved answer changed')
    for name in ('request', 'evidence'):
        p.require(p.sha(case / result['attempt'] / (name + '.json')) == result[name + '_sha256'],
                  'Saved request or evidence changed')
    return result


def choose_attempt(case, *, retry_technical):
    attempts = sorted((case / 'attempts').glob('[0-9][0-9][0-9][0-9]'))
    responses = [path for path in attempts if (path / 'response.json').exists()]
    p.require(len(responses) <= 1, 'Multiple completed responses; do not choose based on score')
    if responses:
        return responses[0], True
    p.require(not attempts or retry_technical, 'Unfinished technical attempt; use --retry-technical explicitly')
    target = case / 'attempts' / f'{len(attempts) + 1:04d}'
    target.mkdir(parents=True, exist_ok=False)
    return target, False


def run_client(root, method, phase, capture, *, retry_technical=False):
    runtime_src = str(Path(__file__).resolve().parents[1] / 'src')
    if runtime_src not in sys.path:
        sys.path.insert(0, runtime_src)
    from tokenizers import Tokenizer
    from benchmark_client_io import CaptureSlice
    import qualify_graph_niah as client
    fixture, profile = verify(root)
    profile = p.read(root / 'profiles' / (method + '.json'))
    tokenizer_path = Path(profile['tokenizer']) / 'tokenizer.json'
    p.require(p.sha(tokenizer_path) == fixture['tokenizer_sha256'], 'Runtime tokenizer differs')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    scorer = ruler.load_scorer(resolve(root, fixture['upstream_path']))
    output = root / 'arms' / method / phase
    output.mkdir(parents=True, exist_ok=True)
    p.save(output / 'launch-audit.json', launch_audit(profile, capture, method))
    results = []
    for row in phase_rows(fixture, phase):
        case = output / 'cases' / row['id']
        result = saved_result(case, row, method, root)
        if result is not None:
            results.append(result)
            continue
        attempt, retained = choose_attempt(case, retry_technical=retry_technical)
        try:
            cursor = None
            if retained:
                request, response = p.read(attempt / 'request.json'), p.read(attempt / 'response.json')
                request_capture = Path(request['capture'])
            else:
                tokens = load_tokens(root, fixture, row, tokenizer)
                session = 'recall-' + method + '-' + uuid.uuid4().hex
                request = dict(session_id=session, capture=str(capture), method=method, sample_id=row['id'],
                    input_sha256=row['served_sha256'], input_tokens=len(tokens), query_span=row['served_query'],
                    sampling=dict(fixture['sampling'], seed=row['seed']),
                    fixture_sha256=p.sha(root / 'fixture.json'),
                    profile_sha256=p.sha(root / 'profiles' / (method + '.json')))
                p.save(attempt / 'request.json', request)
                cursor = CaptureSlice([capture / 'recall_events.jsonl'])
                response = client.stream(profile, session, tokens, row['served_query'], request['sampling'],
                                         fixture['request_timeout_seconds'])
                p.save(attempt / 'response.json', response)
                request_capture = capture
            p.require(request['fixture_sha256'] == p.sha(root / 'fixture.json') and
                      request['profile_sha256'] == p.sha(root / 'profiles' / (method + '.json')) and
                      request['sampling'] == dict(fixture['sampling'], seed=row['seed']) and
                      request['input_sha256'] == row['served_sha256'], 'Retained request settings changed')
            terminal = client.wait_retired(request_capture, response['response_id'])
            p.require(terminal['history_end'] == row['served_input_tokens'] + len(response['output_ids']) - 1 and
                      not terminal.get('checkpoint_tokens'), 'Accepted history or isolated-session retirement differs')
            release = client.wait_owner_released(request_capture, request['session_id'], response, terminal)
            event_path = request_capture / 'recall_events.jsonl'
            events = p.events_for_response(cursor.read(event_path) if cursor is not None else read_events(event_path),
                                           response, request['session_id'])
            audit = p.audit_events(events, method, profile, query_span=row['served_query'],
                                   required_types=('prefill_pressure', 'query'))
            p.require(audit['event_counts']['query'] == 1, 'RULER needs exactly one public Query reselection')
            p.save(attempt / 'evidence.json', dict(audit=audit, events=events, terminal=terminal, release=release))
            result = dict(id=row['id'], task=row['task'], index=row['index'], method=method, phase=phase,
                status='completed', input_sha256=row['served_sha256'], text=response['text'],
                input_tokens=row['served_input_tokens'], completion_tokens=len(response['output_ids']),
                seconds=response['seconds'], ttft_seconds=response['ttft_seconds'],
                finish_reason=response['finish_reason'], session=request['session_id'],
                response_id=response['response_id'], fixture_sha256=p.sha(root / 'fixture.json'),
                response_sha256=p.sha(attempt / 'response.json'), attempt=str(attempt.relative_to(case)),
                request_sha256=p.sha(attempt / 'request.json'), evidence_sha256=p.sha(attempt / 'evidence.json'),
                event_counts=audit['event_counts'],
                **p.score_case(row, response, scorer, starts_in_thinking=fixture['starts_in_thinking']))
            p.save(case / 'result.json', result)
            results.append(result)
            p.save(output / 'progress.json', dict(method=method, phase=phase, completed=len(results),
                expected=len(phase_rows(fixture, phase)), at=time.time()))
            print(json.dumps(dict(method=method, phase=phase, completed=len(results), id=row['id'],
                                  score=result['official_case_score'])), flush=True)
        except BaseException as error:
            p.save(attempt / 'failure.json', dict(error=repr(error), at=time.time(),
                completed_response_preserved=(attempt / 'response.json').exists()))
            raise
    scores = None
    if phase == 'formal':
        scores = ruler.score_suite(fixture['rows'], {(row['task'], row['index']): row['text'] for row in results},
            scorer, samples_per_task=20, starts_in_thinking=fixture['starts_in_thinking'])
    p.save(output / 'completed.json', dict(status='completed', method=method, phase=phase, cases=len(results),
        qualification_only=phase == 'qualification', scores=scores, results=results,
        fixture_sha256=p.sha(root / 'fixture.json'), at=time.time()))
    return results


def wait_health(process, profile, timeout=1800):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        p.require(process.poll() is None, 'Owned server exited before readiness')
        try:
            with opener.open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError('Owned server readiness timeout')


def stop_owned(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    else:
        # The front-end can exit while a spawned engine still owns the GPU.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def termination_guard():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def gpu_lock(path):
    with Path(path).open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                         text=True).strip()
        p.require(not active, 'GPU already has an owner; do not stop unrelated work')
        yield


def run(args):
    qualify_entrypoints = getattr(args, 'qualify_entrypoints', False)
    p.require(not qualify_entrypoints or args.phase == 'qualification',
              '--qualify-entrypoints is only allowed in the qualification phase')
    root = args.root.resolve()
    fixture, profile = verify(root)
    runtime = (args.runtime or Path(fixture['runtime_source'])).resolve()
    sys.path.insert(0, str(runtime / 'src'))
    verify(root, runtime)
    if args.phase == 'formal':
        qualification_gate(root, fixture, profile)
    methods = (args.method,) if args.method else p.METHODS
    environment = {key: value for key, value in os.environ.items() if not key.startswith('KVMEM_')}
    environment.update(PYTHONPATH=str(runtime / 'src'), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    with termination_guard(), gpu_lock(args.gpu_lock):
        for method in methods:
            profile = p.read(root / 'profiles' / (method + '.json'))
            output = root / 'arms' / method / args.phase
            if (output / 'completed.json').exists():
                # Validate every answer, rather than trusting a stale completion marker.
                for row in phase_rows(fixture, args.phase):
                    p.require(saved_result(output / 'cases' / row['id'], row, method, root), 'Missing completed answer')
                if not qualify_entrypoints or functional_qualified(root, method, profile):
                    continue
            verify(root, runtime)
            attempts = output / 'server-attempts'
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = attempts / f'{len(list(attempts.iterdir())) + 1:04d}'
            attempt.mkdir(exist_ok=False)
            command = [str(args.python), '-B', '-m', 'vllm_kvmem', 'serve', '--config',
                str(root / 'profiles' / (method + '.json')), '--run-dir', str(attempt / 'server')]
            environment.update(p.method_environment(method))
            server = None
            try:
                with (attempt / 'server.log').open('x') as log:
                    server = subprocess.Popen(command, cwd=runtime, env=environment, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                p.save(attempt / 'process.json', dict(pid=server.pid, command=command, method=method,
                    phase=args.phase, at=time.time(), environment=p.method_environment(method)))
                p.save(root / 'status.json', dict(status='loading', method=method, phase=args.phase, at=time.time()))
                wait_health(server, profile)
                p.save(root / 'status.json', dict(status='running', method=method, phase=args.phase, at=time.time()))
                run_client(root, method, args.phase, attempt / 'server/capture', retry_technical=args.retry_technical)
                if qualify_entrypoints and not functional_qualified(root, method, profile):
                    # Import after the controller is initialized: the functional
                    # client reuses helpers here, but must not start another model.
                    from qualify_recall_entrypoints import run_client as qualify_client
                    p.save(root / 'status.json', dict(status='functional-qualification', method=method,
                        phase=args.phase, at=time.time()))
                    qualify_client(root, method, attempt / 'server/capture',
                                   root / 'functional' / method / attempt.name)
                    p.require(functional_qualified(root, method, profile), 'Functional client did not publish a valid proof')
            except BaseException as error:
                p.save(root / 'status.json', dict(status='failed', method=method, phase=args.phase,
                    error=repr(error), at=time.time()))
                raise
            finally:
                stop_owned(server)
                p.save(attempt / 'cleanup.json', dict(owned_pid=server.pid if server else None,
                    returncode=server.returncode if server else None, at=time.time()))
            verify(root, runtime)
    p.save(root / 'status.json', dict(status='completed', phase=args.phase, methods=list(methods), at=time.time()))


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'source', 'config', 'runtime'):
        prep.add_argument('--' + name, type=Path, required=True)
    for name in ('inputs', 'upstream', 'tokenizer-source'):
        prep.add_argument('--' + name, type=Path)
    prep.add_argument('--request-timeout', type=int, default=7200)
    runner = sub.add_parser('run')
    runner.add_argument('--root', type=Path, required=True)
    runner.add_argument('--runtime', type=Path)
    runner.add_argument('--python', type=Path, default=Path(sys.executable))
    runner.add_argument('--method', choices=p.METHODS)
    runner.add_argument('--phase', choices=('qualification', 'formal'), required=True)
    runner.add_argument('--retry-technical', action='store_true')
    runner.add_argument('--qualify-entrypoints', action='store_true',
                        help='Run the three-entrance functional client after the ten cases on the same model process')
    runner.add_argument('--gpu-lock', type=Path, default=Path('/tmp/kvmem-recall-comparison.lock'))
    client = sub.add_parser('client')
    client.add_argument('--root', type=Path, required=True)
    client.add_argument('--method', choices=p.METHODS, required=True)
    client.add_argument('--phase', choices=('qualification', 'formal'), required=True)
    client.add_argument('--capture', type=Path, required=True)
    client.add_argument('--retry-technical', action='store_true')
    functional = sub.add_parser('qualify-events')
    functional.add_argument('--root', type=Path, required=True)
    functional.add_argument('--method', choices=p.METHODS, required=True)
    functional.add_argument('--events', type=Path, required=True)
    binaries = sub.add_parser('bind-binaries')
    binaries.add_argument('--root', type=Path, required=True)
    binaries.add_argument('--runtime', type=Path, required=True)
    binaries.add_argument('--binary', type=Path, action='append')
    return command


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'qualify-events':
        qualify_events(args)
    elif args.action == 'bind-binaries':
        bind_binaries(args)
    elif args.action == 'client':
        run_client(args.root.resolve(), args.method, args.phase, args.capture.resolve(),
                   retry_technical=args.retry_technical)
    else:
        run(args)
