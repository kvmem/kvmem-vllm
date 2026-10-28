"""Paired, frozen full-input FPS prebuild measurements; no answer-based retries.

Single-request AB/BA uses the same two original cases each time. Eight-request
AB uses the first eight original fixture cases, with a synchronized client
start. Both arms reserve the same 12 GiB FPS cache from the 100 GB CPU budget.
Only query_fps_prebuild differs within a pair; CPU/CUDA backend is explicit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Barrier
import time
import uuid

import numpy as np

_spec = importlib.util.spec_from_file_location('_fps_prebuild_halo_controller',
    Path(__file__).with_name('test_ruler_fps_halo_answers.py'))
halo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(halo)

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO/'artifacts/ruler-key-fps-halo-20261023-r1'
ROOT = Path('/data/chaidi/kvmem-vllm-experiments/query-fps-prebuild-live-20261023-r1')
SINGLE_CASES = halo.CASE_IDS[:2]
EIGHT_CASES = halo.CASE_IDS[:8]
CACHE_BYTES = 12*1024**3
PORT = 18868
read, save, sha, require = halo.read, halo.save, halo.sha, halo.require
SCOPE = ('Fixed original full-256K inputs, seeds and generation settings; FPS4 + budgeted halo1, '
    'no Query graph propagation or guidance. Same current runtime and 12 GiB cache in both arms. '
    'Report prefill/Query/decode separately; moving work into prefill is not itself a total speedup. '
    'A generated trajectory or score difference is preserved, never retried based on quality.')


def schedule(suite):
    require(suite in ('single', 'eight', 'both'), 'Unknown measurement suite')
    rows = []
    if suite in ('single', 'both'):
        for pair, order in enumerate(((False, True), (True, False))):
            rows.extend(dict(id=f'single-{pair+1}-{int(enabled)}', suite='single', pair=pair,
                concurrency=1, prebuild=enabled, cases=list(SINGLE_CASES)) for enabled in order)
    if suite in ('eight', 'both'):
        rows.extend(dict(id=f'eight-1-{int(enabled)}', suite='eight', pair=0,
            concurrency=8, prebuild=enabled, cases=list(EIGHT_CASES)) for enabled in (False, True))
    return rows


def profile_for_trial(original, trial, *, device):
    require(device in ('cpu', 'cuda'), 'Explicit prebuild backend must be cpu or cuda')
    k = original['kvmem']
    require((k['query_score_mode'], k['query_fps_halo_blocks'], k['query_fps_optimized']) ==
        ('key_fps_4', 1, True), 'Reference must use optimized FPS4 + atomic halo1')
    require((k['select_budget'], k['gen_budget'], k['block_tokens'], k['sink_blocks'], k['recent_blocks']) ==
        (36864, 16384, 32, 41, 0), 'Original KV/fixed-keep budget differs')
    require(k['cpu_read_cache_bytes'] == 100000000000 and k['archive_backend'] == 'cpu',
        'Preserve the original 100 GB CPU history budget; do not add an unbudgeted cache')
    require(k['graph_alpha'] == 0 and k['pressure_policy'] == 'attention' and
        k['guided_query_tokens'] == 0 and not k['query_history'] and not k['prefill_physical_coalescing'],
        'Original pressure/Query policy changed')
    require(original['max_num_seqs'] == 1 and original['max_num_batched_tokens'] == 16424 and
        original['prefill_chunk_tokens'] == 2048 and original['draft_method'] == 'mtp' and
        original['draft_tokens'] == 4 and original['execution_mode'] == 'compiled_graph',
        'Original execution settings changed')
    require(trial['concurrency'] in (1, 8) and type(trial['prebuild']) is bool, 'Invalid trial')
    profile = deepcopy(original)
    profile.update(port=PORT, max_num_seqs=trial['concurrency'])
    profile['kvmem'].update(query_window_cache_bytes=CACHE_BYTES,
        query_fps_prebuild=trial['prebuild'], query_fps_prebuild_device=device)
    return profile


def assert_paired_profiles(left, right):
    left, right = deepcopy(left), deepcopy(right)
    flags = (left['kvmem'].pop('query_fps_prebuild'), right['kvmem'].pop('query_fps_prebuild'))
    require(all(type(flag) is bool for flag in flags) and set(flags) == {False, True} and left == right,
        'Paired execution profiles differ beyond query_fps_prebuild')


def profile_diff(before, after):
    result = {}
    for key in sorted(before.keys() | after.keys()):
        if key == 'kvmem':
            result.update({'kvmem.'+name: dict(before=before[key].get(name), after=after[key].get(name))
                for name in sorted(before[key].keys() | after[key].keys()) if before[key].get(name) != after[key].get(name)})
        elif before.get(key) != after.get(key):
            result[key] = dict(before=before.get(key), after=after.get(key))
    return result


def verify_trace_capacity(reference, trials, limit):
    """Fail before launch if the bounded shared log cannot hold this cohort."""
    largest = max(sum((reference/halo.ARM/'cases'/case/'query-inputs.jsonl').stat().st_size
        for case in trial['cases']) for trial in trials)
    # Each input/geometry is fixed, but score string lengths may vary slightly.
    require(largest*1.1 <= limit,
        'Bounded shared Query trace capacity is too small for the fixed cohort; '
        'increase its audited diagnostic limit before live launch, never truncate or rename an active shared log')
    return dict(previous_largest_trial_bytes=largest, safety_factor=1.1, configured_limit=limit)


def prepare(root, reference, *, suite, device):
    require(not root.exists(), 'Use a new exclusive measurement root')
    fixture = read(reference/'fixture.json')
    require(tuple(row['id'] for row in fixture['rows']) == halo.CASE_IDS and
        fixture['sampling']['max_tokens'] == 8192 and fixture['sampling']['thinking_token_budget'] == 4096,
        'Original fixed fixture or generation quota changed')
    proof, snapshots = halo.previous.reference_proof(reference, fixture, (halo.ARM,))
    proof['files']['target-spans.json'] = sha(reference/'target-spans.json')
    original = read(reference/halo.ARM/'profile.json')
    trials = schedule(suite)
    from vllm_kvmem.graph_integration import _MAX_LOG_BYTES
    trace_capacity = verify_trace_capacity(reference, trials, _MAX_LOG_BYTES)
    profiles = {trial['id']: profile_for_trial(original, trial, device=device) for trial in trials}
    for trial in trials:
        partner = next(t for t in trials if t['suite'] == trial['suite'] and t['pair'] == trial['pair'] and
            t['prebuild'] != trial['prebuild'])
        assert_paired_profiles(profiles[trial['id']], profiles[partner['id']])
    from vllm_kvmem.serve import build_launch
    for trial in trials:
        build_launch(profiles[trial['id']], REPO, root/'trials'/trial['id']/'preflight-only')
    for row in fixture['rows']:
        path = Path(fixture['input_directory'])/(row['id']+'.i32')
        require(sha(path) == row['served_sha256'] and path.stat().st_size == 4*row['served_input_tokens'],
            'Original input bytes changed')
    root.mkdir(parents=True)
    runtime = root/'runtime'
    for name in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party/qw3'):
        shutil.copytree(REPO/name, runtime/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'setup.py'): shutil.copy2(REPO/name, runtime/name)
    save(root/'runtime-proof.json', dict(files={str(path.relative_to(runtime)): sha(path)
        for path in runtime.rglob('*') if path.is_file()}))
    for name, value in (('fixture.json', fixture), ('target-spans.json', read(reference/'target-spans.json')),
            ('reference-proof.json', proof), ('reference-answers.json', snapshots)):
        save(root/name, value)
    plan = dict(scope=SCOPE, suite=suite, device=device, trials=trials, cache_bytes=CACHE_BYTES,
        reference=str(reference), profile_differences={name: profile_diff(original, profile) for name, profile in profiles.items()},
        trace_capacity=trace_capacity,
        policy='No overwrite, resume or quality-dependent retry. Same frozen runtime for both arms.')
    save(root/'measurement-plan.json', plan)
    names = ['fixture.json', 'target-spans.json', 'reference-proof.json', 'reference-answers.json',
        'runtime-proof.json', 'measurement-plan.json']
    for name, profile in profiles.items():
        path = f'trials/{name}/profile.json'; save(root/path, profile); names.append(path)
    save(root/'prepared-proof.json', {name: sha(root/name) for name in names})
    save(root/'status.json', dict(status='prepared', expected_trials=len(trials), at=time.time()))


def verify(root):
    halo.verify_files(root, read(root/'prepared-proof.json'))
    halo.verify_files(root/'runtime', read(root/'runtime-proof.json')['files'])
    proof = read(root/'reference-proof.json'); halo.verify_files(Path(proof['root']), proof['files'])
    plan, fixture = read(root/'measurement-plan.json'), read(root/'fixture.json')
    require(plan['trials'] == schedule(plan['suite']) and plan['cache_bytes'] == CACHE_BYTES,
        'Frozen trial ordering or cache budget changed')
    require(tuple(row['id'] for row in fixture['rows']) == halo.CASE_IDS, 'Frozen case identity changed')
    for row in fixture['rows']:
        require(sha(Path(fixture['input_directory'])/(row['id']+'.i32')) == row['served_sha256'], 'Input bytes changed')
    return plan, fixture


def save_request_trace(capture, response, destination):
    """Read only after all public responses finish; never move a shared log."""
    lines = []
    with (capture/'block_graph_inputs.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if halo.client.q.native_matches(response, event): lines.append((line, event))
    require(len(lines) == 1 and lines[0][1]['phase'] == 'query', 'Missing or duplicate request-scoped Query evidence')
    with destination.open('x') as stream: stream.write(lines[0][0])
    return lines[0][1]


def phase_metrics(response, evidence, *, client_started_at, steps=()):
    query = next(row for row in evidence['bounded'] if row['selection'] == 'query')
    start, finish = query['refresh_started_at'], query['refresh_finished_at']
    require(finish >= start >= client_started_at, 'Invalid request/Query timing boundaries')
    chunks = response['chunks']; first, last = chunks[0][0], chunks[-1][0]
    after_first = sum(row[1] for row in chunks[1:])
    geometry = [row for row in steps if any(halo.client.q.native_matches(response, entry)
        and entry['prefill'] for entry in row.get('batch_geometry', []))]
    before_query = [row for row in geometry if row['model_submitted_at'] <= start]
    native_start = min((row['model_submitted_at'] for row in before_query), default=None)
    return dict(request_seconds=response['seconds'], ttft_seconds=response['ttft_seconds'],
        client_started_at=client_started_at, first_token_at=client_started_at+first, last_token_at=client_started_at+last,
        native_prefill_before_query_wall_seconds=None if native_start is None else start-native_start,
        client_to_query_seconds=start-client_started_at, query_refresh_seconds=finish-start,
        query_to_first_token_seconds=client_started_at+first-finish,
        decode_stream_seconds=last-first, accepted_tokens_after_first_chunk=after_first,
        decode_stream_tokens_per_second=after_first/(last-first) if last>first else None,
        completion_tokens=len(response['output_ids']), fps=query['query_key_fps_scoring'],
        phase_limit='Native prefill interval includes maintenance before final Query; client-to-Query also includes HTTP/admission. '
            'Query refresh includes replay, scoring, picker and assembly. Decode is accepted streamed tokens after the first chunk; '
            'MTP verification rows are not accepted output. Shared batch CUDA times are not charged to each request.')


def audit_prebuild(query, profile):
    """Observe failures/fallbacks explicitly without treating low accuracy as an error."""
    cfg = profile['kvmem']; stats = query['query_key_fps_scoring']
    actual = stats.get('prebuild', {'enabled': False})
    require(actual.get('enabled') is cfg['query_fps_prebuild'], 'Actual prebuild enablement differs from frozen profile')
    if cfg['query_fps_prebuild']:
        device = str(actual.get('device', ''))
        expected = cfg['query_fps_prebuild_device']
        require(device == expected or expected == 'cuda' and device.startswith('cuda:'), 'Actual prebuild backend differs')
        for name in ('queued_records', 'published_records', 'skipped_records', 'failed_records',
                'deferred_records', 'cancelled_records', 'stale_records', 'query_wait_seconds', 'build_seconds'):
            # The worker's Counter omits events that have never happened.
            value = actual.get(name, 0)
            require(type(value) in (int, float) and np.isfinite(value) and value >= 0, 'Missing/invalid prebuild accounting: '+name)
    return dict(actual, cache_prebuilt_hit_blocks=stats.get('cache_prebuilt_hit_blocks', 0))


def compare_pair(left, right):
    a, b = left['evidence']['query_inputs'][0], right['evidence']['query_inputs'][0]
    domain = ('starts', 'counts', 'score_mask', 'transport_mask', 'mandatory_starts', 'budget')
    same_domain = all(a[key] == b[key] for key in domain)
    x, y = np.asarray(a['raw_scores']), np.asarray(b['raw_scores'])
    query = lambda item: next(row for row in item['evidence']['bounded'] if row['selection'] == 'query')
    same_seeds = same_domain and np.array_equal(x, y)
    same_picker = query(left)['selected_history_ranges'] == query(right)['selected_history_ranges']
    return dict(id=left['result']['id'], domain_equal=same_domain, seeds_exactly_equal=same_seeds,
        maximum_absolute_seed_difference=float(np.abs(x-y).max(initial=0)) if x.shape == y.shape else None,
        picker_equal=same_picker, generated_tokens_equal=left['response']['output_ids'] == right['response']['output_ids'],
        before=dict(score=left['result']['official_case_score'], value_covered=left['result']['value_covered'], metrics=left['metrics']),
        after=dict(score=right['result']['official_case_score'], value_covered=right['result']['value_covered'], metrics=right['metrics']))


def server_environment(runtime):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='5')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').exists():
            env.update(CUDA_HOME=toolkit, PATH=toolkit+'/bin'+os.pathsep+env.get('PATH', '')); break
    return env


def launch_and_wait(root, directory, profile, env):
    command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config', str(directory/'profile.json'),
        '--run-dir', str(directory/'server')]
    started = time.monotonic()
    with (directory/'server.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root/'runtime', env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    save(directory/'server-process.json', dict(pid=process.pid, command=command))
    return process, started


def wait_health(process, profile):
    opener = halo.client.urllib.request.build_opener(halo.client.urllib.request.ProxyHandler({}))
    deadline = time.monotonic()+1200
    while True:
        require(process.poll() is None, 'Paired server exited before health check')
        try:
            with opener.open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                if response.status == 200: return
        except OSError: pass
        require(time.monotonic()<deadline, 'Paired server startup timeout')
        time.sleep(1)


def submit_case(root, directory, fixture, row, profile, barrier):
    case = directory/'cases'/row['id']; case.mkdir(parents=True, exist_ok=False)
    source = Path(fixture['input_directory'])/(row['id']+'.i32')
    require(sha(source) == row['served_sha256'], 'Original prompt changed before submission')
    tokens = np.fromfile(source, dtype='<i4').tolist()
    session = 'fps-prebuild-'+uuid.uuid4().hex
    sampling = dict(fixture['sampling'], seed=row['seed'])
    save(case/'request.json', dict(sampling=sampling, session_id=session, input_sha256=row['served_sha256'],
        query_span=row['served_query'], profile_sha256=sha(directory/'profile.json')))
    if barrier is not None: barrier.wait(timeout=120)
    began = time.monotonic()
    try:
        response = halo.client.stream(profile, session, tokens, row['served_query'], sampling, 7200)
        finished = time.monotonic()
        # Existing client chunk times share its internal monotonic origin.
        # Reconstruct it from the measured duration; preserve a conservative
        # outer bound for JSON setup and post-stream accounting overhead.
        aligned_start = finished-response['seconds']
        save(case/'response.json', response)
        save(case/'client-timing.json', dict(started_at=aligned_start, invoked_at=began,
            finished_at=finished, origin_uncertainty_upper_bound_seconds=max(0., aligned_start-began)))
        return row, tokens, session, response, aligned_start
    except BaseException as error:
        save(case/'failure.json', dict(error=repr(error), started_at=began, at=time.monotonic()))
        raise


def run_trial(root, fixture, trial):
    directory = root/'trials'/trial['id']; profile = read(directory/'profile.json')
    with (directory/'attempt.json').open('x') as stream: json.dump(dict(pid=os.getpid(), at=time.time()), stream)
    rows = {row['id']: row for row in fixture['rows']}; server = None; completed = []
    try:
        server, started = launch_and_wait(root, directory, profile, server_environment(root/'runtime'))
        wait_health(server, profile)
        save(directory/'startup.json', dict(seconds=time.monotonic()-started, scope='Server load/compile/health; excluded from request timings'))
        capture = directory/'server/capture'
        barrier = Barrier(8) if trial['concurrency'] == 8 else None
        if barrier is None:
            # Finish/audit after both requests to avoid audit CPU work between
            # simultaneous requests; each original owner retires independently.
            responses = [submit_case(root, directory, fixture, rows[case], profile, None) for case in trial['cases']]
        else:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(submit_case, root, directory, fixture, rows[case], profile, barrier)
                    for case in trial['cases']]
                responses, failures = [], []
                for future in futures:
                    try: responses.append(future.result())
                    except BaseException as error: failures.append(repr(error))
                require(not failures, 'Concurrent request failures: '+repr(failures))
        require(len(responses) == len(trial['cases']) and
            len({response['response_id'] for _, _, _, response, _ in responses}) == len(responses),
            'A fixed case is missing or native response identities overlap')
        # Do not read shared JSONL while another request may still append its
        # final record. Preserve each original owner; no global log rename.
        for _, _, session, response, _ in responses:
            terminal = halo.client.wait_retired(capture, response['response_id'])
            halo.client.wait_owner_released(capture, session, response, terminal)
        steps_path = capture/'model_step_latency.jsonl'
        steps = [json.loads(line) for line in steps_path.read_text().splitlines()] if steps_path.exists() else []
        for row, tokens, session, response, began in responses:
            case = directory/'cases'/row['id']
            result, evidence = halo.finish_case(root, fixture, row, profile, tokens, session, response, capture)
            selected_trace = save_request_trace(capture, response, case/'query-inputs.jsonl')
            require(selected_trace == evidence['query_inputs'][0], 'Request-scoped export differs from execution audit')
            metrics = phase_metrics(response, evidence, client_started_at=began, steps=steps)
            query = next(event for event in evidence['bounded'] if event['selection'] == 'query')
            metrics['prebuild_audit'] = audit_prebuild(query, profile)
            for name, value in (('result.json', result), ('evidence.json', evidence), ('metrics.json', metrics)): save(case/name, value)
            names = (*halo.CASE_FILES, 'query-inputs.jsonl', 'client-timing.json', 'metrics.json')
            manifest = {name: sha(case/name) for name in names}; save(case/'hashes.json', manifest)
            halo.verify_files(case, manifest)
            completed.append(dict(id=row['id'], metrics=metrics, score=result['official_case_score'], value_covered=result['value_covered']))
            save(directory/'progress.json', dict(completed=len(completed), expected=len(trial['cases']), cases=completed))
        started = min(row['metrics']['client_started_at'] for row in completed)
        finished = max(row['metrics']['client_started_at']+row['metrics']['request_seconds'] for row in completed)
        save(directory/'summary.json', dict(trial=trial, cases=completed, request_wall_seconds=finished-started,
            total_input_tokens=sum(rows[row['id']]['served_input_tokens'] for row in completed),
            total_completion_tokens=sum(row['metrics']['completion_tokens'] for row in completed),
            timing_scope='Request wall span excludes server startup and offline auditing; mixed prefill/decode, not pure decode throughput.'))
    except BaseException as error:
        save(directory/'failure.json', dict(error=repr(error), completed=len(completed), at=time.time()))
        raise
    finally:
        remaining = halo.lifecycle.stop_owned_groups((server,)); gpu = sorted(halo.lifecycle.gpu_processes())
        save(directory/'cleanup.json', dict(remaining_owned_groups=remaining, gpu_processes=gpu))
        require(not remaining and not gpu, 'Owned trial GPU cleanup incomplete')


def audit_trial(root, fixture, trial):
    directory = root/'trials'/trial['id']; profile = read(directory/'profile.json')
    rows = {row['id']: row for row in fixture['rows']}; result = {}
    for case_id in trial['cases']:
        row = rows[case_id]; case = directory/'cases'/case_id
        halo.verify_files(case, read(case/'hashes.json'))
        request, response = read(case/'request.json'), read(case/'response.json')
        require(request['sampling'] == dict(fixture['sampling'], seed=row['seed']) and
            request['input_sha256'] == row['served_sha256'] and request['query_span'] == row['served_query'] and
            request['profile_sha256'] == sha(directory/'profile.json'), 'Recorded request changed fixed inputs or parameters')
        evidence, answer = read(case/'evidence.json'), read(case/'result.json')
        query = next(row for row in evidence['bounded'] if row['selection'] == 'query')
        audit_prebuild(query, profile)
        graph = next(row for row in evidence['graph'] if row['phase'] == 'query')
        replay = halo.audit_selection(query, graph, evidence['query_inputs'][0], profile)
        recorded = deepcopy(answer['selection_audit'])
        for audit in (replay, recorded): audit['propagation'].pop('scoring_seconds', None)
        require(replay == recorded, 'Independent native group picker replay changed')
        scorer = halo.protocol.load_scorer(Path(fixture['upstream_path']))
        final = halo.protocol.final_answer(response['text'], starts_in_thinking=fixture['starts_in_thinking'])
        scoring = scorer[0][halo.protocol.TASK_FAMILIES[row['task']]]
        expected = scoring['metric_fn']([scorer[1](final['text'], scoring)], [row['outputs']])
        spans = read(root/'target-spans.json')[case_id]
        require(answer['official_case_score'] == expected and answer['answer'] == final['text'] and
            answer['value_covered'] == halo.coverage(query['selected_history_ranges'], spans['token_span']),
            'Answer score or full-target coverage differs from original saved response/evidence')
        result[case_id] = dict(result=answer, evidence=evidence, response=response, metrics=read(case/'metrics.json'))
    cleanup = read(directory/'cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'], 'Trial cleanup audit missing')
    capture_files = {str(path.relative_to(directory)): sha(path)
        for path in (directory/'server/capture').rglob('*') if path.is_file()}
    capture_files['server.log'] = sha(directory/'server.log')
    save(directory/'capture-proof.json', capture_files)
    save(directory/'execution-audit.json', dict(status='passed', cases=len(result),
        original_input_seed_sampling=True, request_scoped_query_evidence=True,
        native_picker_replay=True, original_pressure_policy=True, gpu_cleanup=True,
        original_shared_capture_sha256=sha(directory/'capture-proof.json')))
    return result


def run(root):
    require(Path(__file__).resolve() == root/'runtime/scripts'/Path(__file__).name, 'Run the frozen measurement controller')
    plan, fixture = verify(root)
    with (root/'attempt.json').open('x') as stream: json.dump(dict(pid=os.getpid(), at=time.time()), stream)
    completed = {}; comparisons = []
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, halo.lifecycle.cleanup_on_termination():
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            for trial in plan['trials']:
                require(not halo.lifecycle.gpu_processes(), 'Foreign GPU owner; it will not be stopped')
                save(root/'status.json', dict(status='running', active_trial=trial['id'], completed_trials=len(completed), at=time.time()))
                run_trial(root, fixture, trial)
                completed[trial['id']] = audit_trial(root, fixture, trial)
                save(root/'progress.json', dict(completed_trials=list(completed), expected_trials=len(plan['trials'])))
        verify(root)
        for trial in plan['trials']:
            if trial['prebuild']: continue
            partner = next(row for row in plan['trials'] if row['suite'] == trial['suite'] and row['pair'] == trial['pair'] and row['prebuild'])
            assert_paired_profiles(read(root/'trials'/trial['id']/'profile.json'), read(root/'trials'/partner['id']/'profile.json'))
            comparisons.append(dict(suite=trial['suite'], pair=trial['pair'], cases=[compare_pair(
                completed[trial['id']][case], completed[partner['id']][case]) for case in trial['cases']]))
        exact = all(case['seeds_exactly_equal'] and case['picker_equal'] for pair in comparisons for case in pair['cases'])
        save(root/'summary.json', dict(scope=SCOPE, comparisons=comparisons, paired_seed_and_picker_exact=exact,
            qualification='passed' if exact else 'needs_review',
            interpretation='Different live operands/batching can change seeds or trajectories; a difference is preserved for investigation, not automatically an FPS implementation bug.'))
        save(root/'execution-audit.json', dict(status='passed', trials=len(completed),
            fixed_inputs_and_parameters=True, frozen_source=True, native_group_picker_replay=True,
            prior_answers_unchanged=True, per_trial_gpu_cleanup=True, paired_seed_and_picker_exact=exact))
        save(root/'status.json', dict(status='completed', trials=len(completed), qualification='passed' if exact else 'needs_review', at=time.time()))
    except BaseException as error:
        save(root/'failure.json', dict(error=repr(error), completed_trials=list(completed), at=time.time()))
        save(root/'status.json', dict(status='failed', completed_trials=list(completed), error=repr(error), at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--suite', choices=('single', 'eight', 'both'), default='single')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.root.resolve(), args.reference.resolve(), suite=args.suite, device=args.device)
    else: run(args.root.resolve())
