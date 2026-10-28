"""Current-source, paired real DeepSWE serving-capacity measurements.

Freeze before running. Every cell uses fresh real task containers, identical
logical context/output limits and unmodified native admission in the control.
Never infer a maximum sustainable arrival rate from closed-loop agent QPS.
"""
import argparse
import array
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path('/home/chaidi/qw3/benchmark/action_pressure_first16_warmfix_20261010')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def profile_for(source):
    p = copy.deepcopy(source)
    p.update(port=18960, served_model_name='deepswe-qps-256k', max_model_len=262144,
             max_num_seqs=8, verify_session_restore=False, verify_session_restore_digest=False)
    # Keep every historical KV page below the common ceiling. The current LRU
    # deployment deletes history at ~256K; it must not confound this comparison.
    p['kvmem'].update(historical_gc=None, repository_gc=False, graph_mode='multi',
                      graph_alpha=0., session_capacity=32)
    assert (p['kvmem']['select_budget'], p['kvmem']['gen_budget']) == (98304, 16384)
    assert p['draft_method'] == 'mtp' and p['draft_tokens'] == 4
    assert not p['kvmem'].get('host_stage_timing')
    return p


def cell_plan(concurrencies, repeats, seconds):
    if (not concurrencies or len(set(concurrencies)) != len(concurrencies)
            or any(c not in (1, 2, 4, 8) for c in concurrencies)
            or repeats < 1 or seconds < 60):
        raise ValueError('Use distinct 1/2/4/8 concurrency levels and finite observation windows')
    cells = []
    for repeat in range(repeats):
        for i, concurrency in enumerate(concurrencies):
            order = ('native', 'kvmem') if (repeat + i) % 2 == 0 else ('kvmem', 'native')
            for backend in order:
                cells.append(dict(name=f'r{repeat+1}-{backend}-c{concurrency}', backend=backend,
                    concurrency=concurrency, seed=1000+repeat, seconds=seconds, repeat=repeat+1))
    return cells


def validate_carry(source, cell, profile, runtime):
    source=source.resolve();old=read(source/'manifest.json');state=read(source/'run.json')
    if (state['status']!='window_complete' or state.get('cleanup_errors')
            or read(source/'summary.json')['errors'] or read(source/'profile.json')!=profile
            or (old['workers'],old['seed'],old['backend_kind'],old['duration_seconds']) !=
               (cell['concurrency'],cell['seed'],cell['backend'],cell['seconds'])):
        raise ValueError('Carried measurement differs or did not complete cleanly')
    original=read(source.parent/'manifest.json')
    prefixes=('src/','csrc/','third_party/','benchmarks/deepswe/')
    required=('scripts/benchmark_deepswe_first16.py','scripts/native_agent_launch.py',
              'scripts/deepswe_window_snapshot.py')
    for name,digest in original['runtime_sha256'].items():
        if name.startswith(prefixes) or name in required:
            if sha(runtime/name)!=digest or sha(source.parent/'runtime'/name)!=digest:
                raise ValueError('Carried inference/task implementation differs: '+name)
    files=('manifest.json','profile.json','run.json','summary.json','window-end-metrics.json',
           'metrics-before-formal.txt','bounded-warmup.json','window-workspaces.json')
    return dict(path=str(source),files={str(source/name):sha(source/name) for name in files},
        reason='Reuse completed measurement with identical runtime/task/profile. Only common CUDA path alias normalized; same resolved nvcc binary.')


def prepare(args):
    from scripts import benchmark_deepswe_first16 as runner
    from scripts.patch_deepswe_client_unicode import patch
    from vllm_kvmem.serve import build_launch
    from scripts.native_agent_launch import build_launch as native_launch
    root = args.output.resolve()
    if (root/'manifest.json').exists():
        raise ValueError('Never overwrite an existing campaign')
    root.mkdir(parents=True, exist_ok=True)
    runtime = root/'runtime'
    runtime.mkdir()
    for name in ('src', 'scripts', 'benchmarks/deepswe', 'csrc', 'third_party'):
        shutil.copytree(ROOT/name, runtime/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('pyproject.toml', 'setup.py', 'CMakeLists.txt'):
        shutil.copy2(ROOT/name, runtime/name)
    profile = profile_for(read(args.profile))
    build_launch(profile, root, root/'validation-server')
    native = native_launch(profile, port=profile['port'], async_scheduling=True, environ={})
    save(root/'effective-native-launch.json', {k:v for k,v in native.items() if k != 'environment'})
    reference = read(REFERENCE/'manifest.json')
    binary, audit = patch(Path(reference['claude_binary']).read_bytes(), reference['claude_sha256'])
    if audit['patched_sha256'] != '0bc3800ba6bcf8b9df2daa47509f2159a4c2b74b67922424b4e94e1019c89433':
        raise ValueError('Client differs from previously qualified exact-JSON client')
    client = root/'claude-exact-json'
    client.write_bytes(binary); client.chmod(0o755)
    save(root/'client-audit.json', audit)
    exported = read(args.history/'manifest.json')
    sample = exported['samples'][0]
    data = (args.history/sample['file']).read_bytes()
    if hashlib.sha256(data).hexdigest() != sample['sha256']:
        raise ValueError('Real DeepSWE history changed')
    warm_tokens = max(67584, profile['kvmem']['select_budget']+2048)
    (root/'warmup.i32').write_bytes(data[:warm_tokens*4])
    if len(data) < warm_tokens*4:
        raise ValueError('Real history is too short for initialization')
    cells = cell_plan(args.concurrencies, args.repeats, args.seconds)
    carry={p.name:p for p in args.carry_cell};carried={}
    if set(carry)-{c['name'] for c in cells}:raise ValueError('Unknown carried cell')
    for cell in cells:
        directory = root/cell['name']
        if cell['name'] in carry:
            carried[cell['name']]=validate_carry(carry[cell['name']],cell,profile,runtime)
            directory.symlink_to(carry[cell['name']].resolve(),target_is_directory=True)
            continue
        directory.mkdir()
        save(directory/'profile.json', profile)
        shutil.copytree(ROOT/'artifacts/deepswe-first16-20261015-r16/protocol', directory/'protocol')
        manifest = runner.prepare(SimpleNamespace(reference=REFERENCE, config=directory/'profile.json',
            output=directory, workers=cell['concurrency'], seed=cell['seed'], bash_repeat_limit=32,
            carry_result=[], backend_kind=cell['backend'], duration_seconds=cell['seconds'],
            optimized_handoff=True))
        script = directory/'setup/run_deepswe_cell.sh'
        text = script.read_text()
        if text.count(reference['claude_binary']+':/usr/local/bin/claude:ro') != 1:
            raise ValueError('Unknown client mount')
        text = text.replace(reference['claude_binary']+':/usr/local/bin/claude:ro',
                            str(client)+':/usr/local/bin/claude:ro')
        script.write_text(text)
        manifest.update(claude_sha256=audit['patched_sha256'],
            client_patch='Same qualified exact JSON string preservation in both arms.',
            setup_sha256={p.name:sha(p) for p in (directory/'setup').iterdir() if p.is_file()},
            snapshot_window_workspaces=True)
        save(directory/'manifest.json', manifest)
    inputs = [root/'warmup.i32', client, root/'client-audit.json', root/'effective-native-launch.json']
    for cell in cells:
        if cell['name'] not in carried:
            inputs.extend(p for p in (root/cell['name']).rglob('*') if p.is_file())
    save(root/'manifest.json', dict(kind='deepswe_qps_256k_v1', created_at=time.time(),
        python=sys.executable, profile_source=str(args.profile.resolve()),
        profile_source_sha256=sha(args.profile), reference=str(REFERENCE),
        reference_sha256=sha(REFERENCE/'manifest.json'), cells=cells, profile=profile,
        runtime_sha256=hashes(runtime), input_sha256={str(p.relative_to(root)):sha(p) for p in inputs},
        carried_windows=carried,build_environment={'CUDA_HOME':'/usr/local/cuda'},
        compiler=dict(path='/usr/local/cuda/bin/nvcc',resolved=str(Path('/usr/local/cuda/bin/nvcc').resolve()),
                      sha256=sha('/usr/local/cuda/bin/nvcc')),
        versions={p:importlib.metadata.version(p) for p in ('vllm','torch','flashinfer-python','transformers')},
        warmup_source=dict(manifest=str(args.history/'manifest.json'), sample=sample,
                           tokens=warm_tokens, formal=False),
        measurement=dict(qps='Completed successful model turns / fixed observation wall seconds',
            output_tps='Accepted generation counter delta / observation wall seconds, includes partial requests',
            delivered_tps='Tokens in completed successful model turns / observation wall seconds',
            quality='Only official scores of naturally completed tasks; censored attempts unscored',
            maximum='Highest observed closed-loop QPS in the tested concurrency grid; no open-loop SLO claim',
            context='Both support 262144 logical tokens; actual prompt lengths reported, no padding/compaction',
            scope='Current K96K/G16K hybrid selective CPU-history deployment versus full native attention; not lossless offload-only speedup'),
        memory_guard_bytes=8*1024**3, disk_guard_bytes=30*1024**3))
    save(root/'status.json', dict(status='prepared', cells=len(cells), at=time.time()))
    print(json.dumps(dict(prepared=str(root), cells=len(cells))))


def summarize_cell(directory):
    from scripts.analyze_deepswe_interference import analyze
    from scripts.analyze_deepswe_capacity import records, distribution
    result = analyze(directory)
    state = read(directory/'run.json')
    events = [r for r in records(directory/'bridge/requests.jsonl')
              if result['begin'] <= r.get('at', 0) <= result['end']]
    starts = {(r['task'],r['turn']):r for r in events if r.get('event') == 'start'}
    finishes = [r for r in events if r.get('event') == 'finish']
    result['qps'] = len(finishes)/result['seconds']
    result['delivered_output_tps'] = result['completed_request_output_tokens']/result['seconds']
    result['backend_roundtrip_seconds'] = distribution([r['at']-starts[(r['task'],r['turn'])]['at']
        for r in finishes if (r['task'],r['turn']) in starts])
    result['request_wall_seconds'] = distribution([
        r.get('seconds',r['at']-starts[(r['task'],r['turn'])].get('request_started_at',
                                                              starts[(r['task'],r['turn'])]['at']))
        for r in finishes if (r['task'],r['turn']) in starts])
    result['outstanding_at_boundary'] = len(starts)-len(finishes)
    result['context_ceiling_events'] = [r for r in events if r.get('event') == 'context_limit']
    result['task_states'] = {k:sum(t['state']==k for t in state['tasks'].values())
                            for k in {t['state'] for t in state['tasks'].values()}}
    result['official_results'] = {k:v.get('result') for k,v in state['tasks'].items()
                                 if v['state']=='completed'}
    result['errors'] = [r for r in events if r.get('event')=='error']
    save(directory/'summary.json', result)
    return result


def summarize(root):
    plan = read(root/'manifest.json'); cells = {}
    for cell in plan['cells']:
        path = root/cell['name']/'summary.json'
        if path.exists():cells[cell['name']] = dict(cell, **read(path))
    pairs = []
    for repeat, concurrency in sorted({(c['repeat'],c['concurrency']) for c in plan['cells']}):
        a,b = (cells.get(f'r{repeat}-{backend}-c{concurrency}') for backend in ('native','kvmem'))
        if a and b:
            pairs.append(dict(repeat=repeat, concurrency=concurrency,
                qps_gain_percent=100*(b['qps']/a['qps']-1) if a['qps'] else None,
                output_tps_gain_percent=100*(b['agent_tokens_per_second']/a['agent_tokens_per_second']-1)
                    if a['agent_tokens_per_second'] else None))
    result=dict(cells=cells, pairs=pairs, expected=len(plan['cells']),
                complete=len(cells)==len(plan['cells']), definitions=plan['measurement'])
    save(root/'summary.json',result)
    return result


def run(root):
    import psutil
    from scripts import run_agent_handoff_queue as q
    plan=read(root/'manifest.json')
    if ROOT != root/'runtime' or hashes(ROOT) != plan['runtime_sha256']:
        raise ValueError('Run only the unchanged frozen source')
    if read(root/'status.json')['status'] != 'prepared':
        raise ValueError('No implicit retry or reuse')
    for rel,digest in plan['input_sha256'].items():
        if sha(root/rel) != digest:raise ValueError('Input changed: '+rel)
    for proof in plan.get('carried_windows',{}).values():
        for name,digest in proof['files'].items():
            if sha(name)!=digest:raise ValueError('Carried evidence changed: '+name)
    if {p:importlib.metadata.version(p) for p in plan['versions']} != plan['versions']:
        raise ValueError('Installed dependency versions changed')
    active=None
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(root/'owner.json',dict(pid=os.getpid(),created=psutil.Process().create_time()))
        try:
            for cell in plan['cells']:
                if cell['name'] in plan.get('carried_windows',{}):
                    summarize(root)
                    continue
                if q.gpu_pids():raise RuntimeError('GPU occupied; no foreign process will be stopped')
                if shutil.disk_usage(root).free < plan['disk_guard_bytes']:
                    raise RuntimeError('Insufficient disk headroom')
                directory=root/cell['name']
                command=[plan['python'],'-B',str(ROOT/'scripts/benchmark_deepswe_first16.py'),
                    '--config',str(directory/'profile.json'),'--output',str(directory),
                    '--workers',str(cell['concurrency']),'--seed',str(cell['seed']),
                    '--backend-kind',cell['backend'],'--duration-seconds',str(cell['seconds']),
                    '--warmup-tokens',str(root/'warmup.i32'),'--bridge-port','18961',
                    '--bash-repeat-limit','32','--use-prepared','--optimized-handoff',
                    '--snapshot-window-workspaces','--startup-timeout-seconds','1800']
                env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_'))}
                env.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT,ROOT/'src',ROOT/'scripts'))),
                    PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',CUDA_LAUNCH_BLOCKING='0')
                env.update(plan.get('build_environment',{}))
                with (directory/'controller.log').open('xb') as log:
                    active=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(directory/'owner.json',dict(pid=active.pid,created=psutil.Process(active.pid).create_time(),argv=command))
                save(root/'status.json',dict(status='running',cell=cell['name'],pid=active.pid,at=time.time()))
                owned={active.pid}
                while active.poll() is None:
                    try:owned.update(p.pid for p in psutil.Process(active.pid).children(recursive=True))
                    except psutil.NoSuchProcess:pass
                    if set(q.gpu_pids())-owned:raise RuntimeError('Foreign GPU process appeared')
                    if psutil.virtual_memory().available < plan['memory_guard_bytes']:
                        raise RuntimeError('Host memory headroom below 8GiB')
                    if shutil.disk_usage(root).free < plan['disk_guard_bytes']:
                        raise RuntimeError('Disk headroom below 30GiB')
                    time.sleep(2)
                if active.returncode:raise RuntimeError(f"Cell failed: {cell['name']} ({active.returncode})")
                summarize_cell(directory);summarize(root)
                active=None
                # Native workers can take a moment to exit after the API parent.
                deadline=time.monotonic()+60
                while q.gpu_pids() and time.monotonic()<deadline:time.sleep(1)
            save(root/'status.json',dict(status='completed',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                error=repr(error),at=time.time()))
            raise
        finally:
            if active is not None and active.poll() is None:
                active.send_signal(signal.SIGTERM)
                try:active.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    # Preserve descendants/containers when snapshot cleanup stalls.
                    save(root/'cleanup-needs-attention.json',dict(pid=active.pid,at=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','summarize'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path)
    p.add_argument('--history',type=Path,default=ROOT/'artifacts/deepswe-real-history-20261017-inputs-r1')
    p.add_argument('--concurrencies',type=int,nargs='+',default=[4,8,1,2])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--carry-cell',type=Path,action='append',default=[])
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='run':run(args.output.resolve())
    else:print(json.dumps(summarize(args.output.resolve()),ensure_ascii=False))


if __name__=='__main__':main()
