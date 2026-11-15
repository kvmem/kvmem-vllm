"""The layering gate: reference fixtures run on a frozen runtime, compared with a reference run.

Every refactoring step must leave the built-in strategy token-identical at one request on
both fixtures, with an identical decision trace, and inside the run-to-run spread for
throughput (docs/layering.md, "The gate"). This command runs the fixtures and the checks.

    python benchmarks/gate/gate.py freeze DIR [REV]
        Frozen runtime: the tracked tree at REV (default: the working tree, including
        untracked files that are not ignored) plus the built native extensions.
    python benchmarks/gate/gate.py run {mab,agent} RUNTIME OUT [options]
        One arm of a fixture on a frozen runtime. Waits until the GPU is free.
    python benchmarks/gate/gate.py compare REF RUN
        Tokens and decision trace of two arms of one fixture; exit 1 if they differ.
    python benchmarks/gate/gate.py measure RUN [RUN ...]
        Throughput of finished arms.
    python benchmarks/gate/gate.py gate RUNTIME OUT --reference REF [--autotune-dir DIR]
        The whole gate: both fixtures on RUNTIME, tokens and decision trace against REF/mab and
        REF/agent, throughput against REF/mab-repeat and REF/agent-repeat (within --spread).
        Writes OUT/gate-result.json; exit 0 only if everything passes.

Fixtures (both single-request for exactness):
    mab    16 memory-benchmark cases (64-128K contexts), offline prefill, G = 4,096
           (mab16.json; the questions live under --data).
    agent  Agent simulator, one session by default: 48K history, then 12 tool-wait turns
           of ~300 appended and 2,048 generated tokens, G = 4,096 (agent.json). Exercises
           sessions, restore, historical GC, the agent Query and rolling-decode pressure.
           --sessions 4 is the multi-owner smoke run; its outputs are timing-dependent,
           so it checks completion and released resources, not tokens.

Run the arms from a frozen copy of this directory and of the runtime, never from a
checkout that is being edited. Record a new reference whenever GPU kernels are rebuilt
or re-tuned: a stale reference differs from the first decision on, whatever the code.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT_DATA = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/mab-full-paired-20261105-r1/data/questions')
STRATEGY_OPTIONS = ('strategy_query_scorer', 'strategy_pressure_scorer', 'strategy_picker', 'strategy_trace_scores',
                    'strategy_path')


def log(*parts):
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1) + '\n')


def gpu_processes():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def wait_for_gpu(timeout):
    deadline = time.time() + timeout
    while gpu_processes():
        if time.time() > deadline:
            raise SystemExit('GPU still in use by %s' % gpu_processes())
        time.sleep(10)


def mem_available_gib():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return float('inf')


def trace_module():
    """vllm_kvmem.strategy.trace from this tree, loaded alone (it only needs the standard library)."""
    path = HERE.parent.parent / 'src/vllm_kvmem/strategy/trace.py'
    spec = importlib.util.spec_from_file_location('gate_trace', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- freeze

def freeze(target, rev=None):
    repo = HERE.parent.parent
    target = Path(target).resolve()
    if target.exists():
        raise SystemExit('exists: %s' % target)
    target.mkdir(parents=True)
    if rev:
        archive = subprocess.run(['git', 'archive', rev], cwd=repo, capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(target)], input=archive, check=True)
    else:
        files = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=repo,
                               capture_output=True, check=True).stdout.split(b'\0')
        for name in filter(None, files):
            source = repo / name.decode()
            if source.is_file():
                (target / name.decode()).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / name.decode())
    for so in (repo / 'src/vllm_kvmem').glob('*.so'):
        shutil.copy2(so, target / 'src/vllm_kvmem' / so.name)
    if (repo / 'src/vllm_kvmem/cuda_sources').is_dir():
        shutil.copytree(repo / 'src/vllm_kvmem/cuda_sources', target / 'src/vllm_kvmem/cuda_sources',
                        dirs_exist_ok=True)
    head = subprocess.run(['git', 'rev-parse', '--short', rev or 'HEAD'], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = '' if rev else subprocess.run(['git', 'status', '--short'], cwd=repo, capture_output=True,
                                          text=True, check=True).stdout
    save_json(target / 'FROZEN.json', dict(rev=head, working_tree=not rev, dirty=dirty.splitlines(), at=time.time()))
    log('frozen', target, 'rev', head, '(working tree, %d changed)' % len(dirty.splitlines()) if not rev else '')


# ---------------------------------------------------------------- server

def server_environment(runtime, execution, extra):
    """The environment the mab harness gives its server (same toolkit spelling, so no JIT rebuild)."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime / 'src'), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='5')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit) / 'bin/nvcc').exists():
            preferred = Path('/usr/local/cuda')
            if preferred.exists() and Path(toolkit).samefile(preferred):
                toolkit = str(preferred)
            env.update(CUDA_HOME=toolkit, PATH=toolkit + '/bin' + os.pathsep + env.get('PATH', ''))
            break
    env['QUERY_STAGE_EXECUTION_JSON'] = json.dumps(execution)
    env.update(extra)
    return env


def healthy(process, port, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            if opener.open('http://127.0.0.1:%d/health' % port, timeout=5).status == 200:
                return True
        except Exception:  # noqa: BLE001 - still starting
            pass
        time.sleep(5)
    return False


def stop_group(process):
    if process.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# ---------------------------------------------------------------- agent fixture

def run_agent(args):
    fixture = load_json(HERE / 'agent.json')
    runtime, out = Path(args.runtime).resolve(), Path(args.out).resolve()
    if out.exists():
        raise SystemExit('exists: %s' % out)
    profile = json.loads(json.dumps(fixture['profile']))
    opts = json.loads(args.opts)
    for key, value in opts.items():   # the server validates every option; null removes one
        if value is None:
            profile['kvmem'].pop(key, None)
        else:
            profile['kvmem'][key] = value
    profile['port'] = args.port
    execution = dict(fixture['execution'], **json.loads(args.env))
    # The global path: one selection for every layer (no --layerwise-* flag).
    flags = fixture['server_flags'] if args.path == 'layer' else []
    client = dict(fixture['client'], **json.loads(args.client))
    if args.turns is not None:
        client['turns'] = args.turns
    out.mkdir(parents=True)
    save_json(out / 'profile.json', profile)
    save_json(out / 'execution.json', execution)
    save_json(out / 'gate.json', dict(fixture='agent', path=args.path, runtime=str(runtime), tools=str(HERE),
                                      sessions=args.sessions,
                                      client=client, opts=opts, env=json.loads(args.env),
                                      runtime_frozen=load_json(runtime / 'FROZEN.json')
                                      if (runtime / 'FROZEN.json').exists() else None))
    wait_for_gpu(args.gpu_wait)
    log('agent arm', out.name, 'runtime', runtime.name, 'sessions', args.sessions)
    extra = {'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR': str(Path(args.autotune_dir).resolve())} if args.autotune_dir else {}
    env = server_environment(runtime, execution, extra)
    command = [PY, '-B', str(HERE / 'serve_with_env.py'), '--config', str(out / 'profile.json'),
               '--run-dir', str(out / 'server'), *flags]
    with (out / 'server.log').open('x') as stream:
        server = subprocess.Popen(command, cwd=runtime, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    save_json(out / 'process.json', dict(server_pid=server.pid, command=command, at=time.time()))
    status = 'failed'
    try:
        if not healthy(server, args.port, 1800):
            raise SystemExit('server never became healthy; see %s' % (out / 'server.log'))
        log('server up')
        stop = threading.Event()
        sampler = threading.Thread(target=sample_gpu, args=(out / 'gpu.csv', stop), daemon=True)
        sampler.start()
        sim = [PY, '-I', '-B', str(HERE / 'agent_sim.py'), '--port', str(args.port),
               '--model', profile['served_model_name'], '--tokenizer', profile['tokenizer'], '--backend', 'kvmem',
               '--sessions', str(args.sessions), '--history', str(client['history']), '--duration', '7200',
               '--output', str(client['output']), '--tool-median', str(client['tool_median']),
               '--append-median', str(client['append_median']), '--seed', str(client['seed']),
               '--max-turns', str(client['turns']), '--save-tokens', '--tag', 'gate', '--out', str(out)]
        if client.get('shared_prefix'):
            # docs/prefix-sharing.md: one shared system prompt; arms cold, warm or import.
            sim += ['--shared-prefix', str(client['shared_prefix']), '--prefix-mode', client.get('prefix_mode', 'cold'),
                    '--block', str(profile['kvmem']['block_tokens'])]
        started = time.time()
        with (out / 'client.log').open('x') as stream:
            proc = subprocess.Popen(sim, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                if mem_available_gib() < 8:
                    os.killpg(proc.pid, signal.SIGKILL)
                    raise SystemExit('MemAvailable below 8 GiB; arm aborted')
                if server.poll() is not None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    raise SystemExit('server exited during the arm; see %s' % (out / 'server.log'))
                time.sleep(2)
        stop.set()
        sampler.join()
        save_json(out / 'wall.json', dict(client_started=started, client_ended=time.time()))
        time.sleep(20)   # releases and archive retirement settle before the resource snapshot is read
        status = 'done' if proc.returncode == 0 else 'client exit %d' % proc.returncode
    finally:
        stop_group(server)
        save_json(out / 'status.json', dict(status=status, at=time.time()))
    print(json.dumps(measure_agent(out)))
    return 0 if status == 'done' else 1


def sample_gpu(path, stop):
    with open(path, 'w') as stream:
        while not stop.is_set():
            row = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
            if row:
                stream.write('%.2f,%s\n' % (time.time(), row.splitlines()[0]))
                stream.flush()
            stop.wait(0.5)


def agent_rows(run):
    rows = load_json(Path(run) / 'client.json')['rows']
    by = {}
    for row in sorted(rows, key=lambda r: r['started']):
        by.setdefault(row['session'], []).append(row)
    return by


def measure_agent(run):
    run = Path(run)
    data = load_json(run / 'client.json')
    rows = data['rows']
    turns = [r for r in rows if r['kind'] == 'turn']
    grows = [r for r in rows if r['kind'] == 'grow']
    busy = sum(r['ended'] - r['started'] for r in turns)
    trace = trace_module()
    decisions = trace.read(run)
    stages = {}
    layers = trace.locate(run)
    for line in open(layers) if layers.is_file() else ():
        event = json.loads(line)
        if event.get('event') == 'pressure_selected':
            stages[event.get('event_type', '?')] = stages.get(event.get('event_type', '?'), 0) + 1
    latencies = sorted(r['ended'] - r['started'] for r in turns)
    return dict(run=run_name(run), requests=len(rows), turns=len(turns), errors=data['summary']['errors'],
                output_tokens=sum(r['output_tokens'] for r in turns),
                turn_tok_s=round(sum(r['output_tokens'] for r in turns) / busy, 2) if busy else None,
                turn_latency_p50=round(latencies[len(latencies) // 2], 3) if latencies else None,
                grow_seconds=round(sum(r['ended'] - r['started'] for r in grows), 2),
                decisions=len(decisions), query=sum(d.kind == 'query' for d in decisions),
                pressure=sum(d.kind == 'pressure' for d in decisions),
                global_decisions=sum(d.layer == trace.GLOBAL_LAYER for d in decisions), pressure_events=stages,
                methods=sorted({d.method for d in decisions}), released=released(run),
                tracebacks=(run / 'server.log').read_text(errors='replace').count('Traceback'))


def run_name(run):
    return run.name if run.name != 'n01' else run.parent.name


def released(run):
    """What the server still held after the last session was released (all zero when nothing leaked)."""
    capture = run / 'server/capture'
    scheduler = load_json(capture / 'scheduler_resources.json')
    archive = load_json(capture / 'archive_resources.json')
    held = dict(requests=scheduler.get('requests'), refresh_pending=scheduler.get('refresh_pending'),
                connector_pending=scheduler.get('connector_pending'),
                reservations=len(scheduler.get('reservations') or ()), owned_blocks=scheduler.get('owned_blocks'),
                allocated_pages=(scheduler.get('gpu_capacity') or {}).get('physical_allocated_pages'),
                idle_views=(scheduler.get('active_residency') or {}).get('idle_views'),
                archive_active=archive.get('active_requests'), archive_retiring=archive.get('retiring_requests'),
                archive_io_pending=archive.get('archive_io_pending_bytes'))
    held['clean'] = not any(held.values())
    return held


def compare_agent(ref, run):
    a, b = agent_rows(ref), agent_rows(run)
    total = same = 0
    diffs = []
    for session in sorted(set(a) | set(b)):
        ra, rb = a.get(session, []), b.get(session, [])
        if len(ra) != len(rb):
            diffs.append(dict(session=session, requests=(len(ra), len(rb))))
        for index, (x, y) in enumerate(zip(ra, rb)):
            total += 1
            if x.get('tokens') is not None and x['tokens'] == y.get('tokens') and x['prompt_tokens'] == y['prompt_tokens']:
                same += 1
            else:
                first = next((j for j, (p, q) in enumerate(zip(x.get('tokens') or (), y.get('tokens') or ()))
                              if p != q), None)
                diffs.append(dict(session=session, request=index, kind=x['kind'], first_differing_token=first))
    return dict(requests=total, token_identical=same, diffs=diffs[:10],
                same=total > 0 and same == total and not diffs)


# ---------------------------------------------------------------- mab fixture

def run_mab(args):
    """The 16-case memory-benchmark arm: the reference runner of the runtime under test, patched as the
    11-07 concurrency sweep patched it (frozen launch shim, declared execution environment)."""
    fixture_spec = load_json(HERE / 'mab16.json')
    runtime, out = Path(args.runtime).resolve(), Path(args.out).resolve()
    if out.exists():
        raise SystemExit('exists: %s' % out)
    if args.sessions != 1:
        raise SystemExit('The mab fixture is single-request')
    data = Path(args.data)
    fixture = json.loads(json.dumps(fixture_spec['fixture']))
    for row in fixture['rows']:
        row['source'] = str(data / Path(row['source']).name)
        if not Path(row['source']).is_file():
            raise SystemExit('missing question file %s (set --data)' % row['source'])
    profile = json.loads(json.dumps(fixture_spec['profile']))
    opts = json.loads(args.opts)
    unknown = set(opts) - set(profile['kvmem']) - set(STRATEGY_OPTIONS)
    if unknown:
        raise SystemExit('unknown kvmem options: %s' % sorted(unknown))
    profile['kvmem'].update(opts)
    execution = dict(fixture_spec['execution'], **json.loads(args.env))
    out.mkdir(parents=True)
    save_json(out / 'fixture.json', fixture)
    save_json(out / 'profile.json', profile)
    save_json(out / 'execution.json', dict(execution, workers=1, gen_budget=profile['kvmem']['gen_budget']))
    save_json(out / 'gate.json', dict(fixture='mab', runtime=str(runtime), tools=str(HERE), opts=opts,
                                      env=json.loads(args.env),
                                      runtime_frozen=load_json(runtime / 'FROZEN.json')
                                      if (runtime / 'FROZEN.json').exists() else None))
    os.symlink(runtime, out / 'runtime-v1')
    wait_for_gpu(args.gpu_wait)
    log('mab arm', out.name, 'runtime', runtime.name)
    sys.path[:0] = [str(runtime / 'src'), str(runtime / 'scripts')]
    import run_layerwise_memory as memory
    import benchmark_query_fps_prebuild_live as envmod
    original_command = memory.server_command

    def command(root, fx):
        rest = original_command(root, fx)[5:]
        if rest[0] != '--config':
            raise RuntimeError('run_layerwise_memory.server_command changed')
        return [sys.executable, '-B', str(HERE / 'serve_with_env.py'), *rest]
    memory.server_command = command
    memory.verify = lambda root: (fixture, profile)
    spec = importlib.util.spec_from_file_location('gate_mab_loader', runtime / 'scripts/run_full_mab_comparison.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    cache = {}
    memory.source = lambda row: loader.load_source(row, cache)
    original_environment = envmod.server_environment

    def environment(rt):
        env = original_environment(rt)
        env['QUERY_STAGE_EXECUTION_JSON'] = json.dumps(execution)
        return env
    envmod.server_environment = environment
    stop = threading.Event()

    def watchdog():
        while not stop.is_set():
            if mem_available_gib() < 8:
                save_json(out / 'memory-watchdog.json', dict(at=time.time(), available_gib=mem_available_gib()))
                if (out / 'process.json').exists():
                    os.killpg(load_json(out / 'process.json')['server_pid'], signal.SIGTERM)
                return
            stop.wait(2)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        memory.run(out, workers=1)
    except Exception as error:  # noqa: BLE001 - the harness audits fail after every case completes
        log('runner raised', repr(error)[:300])
    finally:
        stop.set()
        for result in (out / 'run/answers/cases').glob('*/result.json'):
            result.unlink()        # ~1.7 GB per run; tokens and decisions are kept elsewhere
    print(json.dumps(measure_mab(out)))
    complete = len(list((out / 'run/answers/cases').glob('*/response.json')))
    return 0 if complete == len(fixture['rows']) else 1


def jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def measure_mab(run):
    run = Path(run)
    cap = run / 'run/server/capture'
    perf = load_json(run / 'inference-performance.json')
    work = load_json(cap / 'public_work.json')
    tokens = work['completed']['public_prefill_tokens']
    t0, t1 = perf['started_at'], perf['completed_at']
    query_start = {}
    for event in jsonl(cap / 'async_refresh.jsonl'):
        if event['stage'] == 'query':
            rid = event['request_id']
            query_start[rid] = min(query_start.get(rid, 1e18), event['at'] - event.get('host_seconds', 0))
    steps = sorted(jsonl(cap / 'model_step_latency.jsonl'), key=lambda s: s['started_at'])
    first = {}
    for step in steps:
        for geometry in step['batch_geometry']:
            rid = geometry['request_id']
            if rid in query_start and step['started_at'] < query_start[rid]:
                first.setdefault(rid, step['started_at'])
    span = sum(query_start[r] - first[r] for r in first)
    phase = sum(s['prefill_tokens'] for s in steps
                if any(first.get(g['request_id'], 1e18) <= s['started_at'] < query_start.get(g['request_id'], 0)
                       for g in s['batch_geometry']))
    trace = trace_module()
    decisions = trace.read(run)
    return dict(run=run_name(run), cases=len(list((run / 'run/answers/cases').glob('*/response.json'))),
                prefill_tokens=tokens, wall=round(t1 - t0, 1), prefill_tok_s=round(tokens / (t1 - t0), 1),
                prefill_phase_tok_s=round(phase / span) if span else None,
                decisions=len(decisions), query=sum(d.kind == 'query' for d in decisions),
                pressure=sum(d.kind == 'pressure' for d in decisions),
                global_decisions=sum(d.layer == trace.GLOBAL_LAYER for d in decisions))


def compare_mab(ref, run):
    def answers(arm):
        cases = {}
        for case in sorted((Path(arm) / 'run/answers/cases').iterdir()):
            choice = load_json(case / 'response.json')['choices'][0]
            cases[case.name] = (choice.get('text'), choice.get('token_ids'))
        return cases
    a, b = answers(ref), answers(run)
    same = sum(a[name] == b.get(name) for name in a)
    differing = [name for name in a if a[name] != b.get(name)]
    return dict(cases=len(a), token_identical=same, diffs=differing[:10],
                same=len(a) > 0 and sorted(a) == sorted(b) and same == len(a))


# ---------------------------------------------------------------- compare / measure

def kind_of(run):
    run = Path(run)
    if (run / 'client.json').exists():
        return 'agent'
    if (run / 'run/answers/cases').is_dir():
        return 'mab'
    raise SystemExit('not a finished gate arm: %s' % run)


def compare(ref, run):
    kind = kind_of(ref)
    if kind_of(run) != kind:
        raise SystemExit('the two arms ran different fixtures')
    tokens = (compare_agent if kind == 'agent' else compare_mab)(ref, run)
    trace = trace_module()
    decisions = trace.diff(trace.read(ref), trace.read(run))
    result = dict(fixture=kind, tokens=tokens, trace=decisions, same=tokens['same'] and decisions['same'])
    print(json.dumps(result, indent=1))
    return 0 if result['same'] else 1


def measure(runs):
    for run in runs:
        print(json.dumps((measure_agent if kind_of(run) == 'agent' else measure_mab)(run)))
    return 0


def throughput(result):
    return result.get('turn_tok_s') if 'turn_tok_s' in result else result.get('prefill_phase_tok_s')


def gate(args):
    """Both fixtures on RUNTIME, each checked against REFERENCE/<fixture> (tokens, decision trace) and
    against REFERENCE/<fixture>-repeat (throughput; the first reference run may be cold)."""
    reference, out = Path(args.reference).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    verdict = {}
    for fixture in args.fixtures.split(','):
        arm = out / fixture
        if not arm.exists():
            argv = ['run', fixture, args.runtime, str(arm), '--port', str(args.port), '--gpu-wait', str(args.gpu_wait)]
            if fixture == 'agent' and args.autotune_dir:
                argv += ['--autotune-dir', args.autotune_dir]
            if fixture == 'mab':
                argv += ['--data', args.data]
            subprocess.run([PY, '-B', str(Path(__file__).resolve()), *argv], check=False)
        ref = reference / fixture
        repeat = reference / (fixture + '-repeat')
        compared = subprocess.run([PY, '-B', str(Path(__file__).resolve()), 'compare', str(ref), str(arm)],
                                  capture_output=True, text=True)
        exact = json.loads(compared.stdout) if compared.stdout.strip() else dict(same=False, error=compared.stderr[-2000:])
        measure_of = measure_agent if fixture == 'agent' else measure_mab
        now = throughput(measure_of(arm))
        base = throughput(measure_of(repeat if repeat.exists() else ref))
        change = (now - base) / base if now and base else None
        verdict[fixture] = dict(exact=exact.get('same', False), tokens=exact.get('tokens'),
                                trace=dict((k, v) for k, v in (exact.get('trace') or {}).items() if k != 'different'),
                                throughput=now, reference_throughput=base,
                                change=round(change, 4) if change is not None else None,
                                within_spread=change is not None and change >= -args.spread)
    verdict['passed'] = all(v['exact'] and v['within_spread'] for v in verdict.values())
    save_json(out / 'gate-result.json', verdict)
    print(json.dumps(verdict, indent=1))
    return 0 if verdict['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('dir')
    p.add_argument('rev', nargs='?')
    p = sub.add_parser('run')
    p.add_argument('fixture', choices=('mab', 'agent'))
    p.add_argument('runtime')
    p.add_argument('out')
    p.add_argument('--sessions', type=int, default=1)
    p.add_argument('--turns', type=int, default=None)
    p.add_argument('--path', choices=('layer', 'global'), default='layer',
                   help='agent fixture: per-layer selection (the fixture flags) or the global path')
    p.add_argument('--opts', default='{}',
                   help='kvmem profile options, JSON (e.g. the strategy_* options); null removes an option')
    p.add_argument('--env', default='{}', help='extra KVMEM_* execution flags, JSON')
    p.add_argument('--client', default='{}',
                   help='agent fixture: client overrides, JSON (e.g. shared_prefix, prefix_mode cold|warm|import)')
    p.add_argument('--port', type=int, default=19641)
    p.add_argument('--autotune-dir', default=None, help='FlashInfer autotune cache root (agent fixture)')
    p.add_argument('--data', default=str(DEFAULT_DATA), help='directory of the memory-benchmark question files')
    p.add_argument('--gpu-wait', type=float, default=6 * 3600, help='seconds to wait for a free GPU')
    p = sub.add_parser('gate', help='both fixtures on RUNTIME against a reference set')
    p.add_argument('runtime')
    p.add_argument('out')
    p.add_argument('--reference', required=True,
                   help='directory with mab, mab-repeat, agent, agent-repeat (arm directories or links)')
    p.add_argument('--fixtures', default='mab,agent')
    p.add_argument('--spread', type=float, default=0.02, help='allowed throughput loss against the repeat')
    p.add_argument('--port', type=int, default=19641)
    p.add_argument('--autotune-dir', default=None)
    p.add_argument('--data', default=str(DEFAULT_DATA))
    p.add_argument('--gpu-wait', type=float, default=6 * 3600)
    p = sub.add_parser('compare')
    p.add_argument('ref')
    p.add_argument('run')
    p = sub.add_parser('measure')
    p.add_argument('runs', nargs='+')
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        return freeze(args.dir, args.rev) or 0
    if args.command == 'run':
        return run_agent(args) if args.fixture == 'agent' else run_mab(args)
    if args.command == 'gate':
        return gate(args)
    if args.command == 'compare':
        return compare(args.ref, args.run)
    return measure(args.runs)


if __name__ == '__main__':
    sys.exit(main())
