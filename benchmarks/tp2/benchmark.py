"""Owned, cold-prefix serving measurements; fixed outputs are diagnostic only."""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid


BASE = Path(__file__).resolve().parent
MODEL = str(BASE / 'models')
TOKENIZER = MODEL
# CUDA 13.0 toolkit root (nvcc, ptxas, include/cccl) and the host compilers for JIT
# builds. Defaults are the A800 host's; override for another machine.
CUDA = os.environ.get('BENCH_CUDA_HOME', '/root/blockdata/cuda13.0/nvidia/cu13')
HOST_CC = os.environ.get('BENCH_CC', '/usr/bin/gcc-11')
HOST_CXX = os.environ.get('BENCH_CXX', '/usr/bin/g++-11')


def cuda_arch():
    """TORCH_CUDA_ARCH_LIST for JIT builds: BENCH_CUDA_ARCH, else the GPUs' compute capability
    (8.0 on the A800, 8.9 on the RTX 4090), else 8.0."""
    value = os.environ.get('BENCH_CUDA_ARCH')
    if value:
        return value
    try:
        caps = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        caps = []
    return caps[0] if caps and len(set(caps)) == 1 else '8.0'
PORT = 18977


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def http(path, data=None, timeout=15):
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def metrics():
    with http('/metrics') as response:
        raw = response.read().decode()
    result = {}
    for line in raw.splitlines():
        if not line.startswith('vllm:') or line.startswith('#'):
            continue
        key, value = line.rsplit(' ', 1)
        if ('_sum' in key or '_count' in key or '_total' in key) and '_bucket' not in key:
            try:
                result[key] = float(value)
            except ValueError:
                pass
    return result


def prompt_builder():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    marker = 'KVMEM_BENCHMARK_HISTORY_MARKER'
    query = ('\nQuestion: Explain how a production inference service should manage CPU and GPU '
        'memory, batch independent requests, and measure input processing separately from '
        'generation. Give detailed practical reasoning in complete paragraphs.')
    template = tok.apply_chat_template([
        {'role': 'system', 'content': 'You are a helpful technical assistant.'},
        {'role': 'user', 'content': marker + query}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    left, right = template.split(marker)
    prefix = tok.encode(left, add_special_tokens=False)
    suffix = tok.encode(right, add_special_tokens=False)
    rng = random.Random(20261116)
    words = ['cache', 'request', 'scheduler', 'memory', 'latency', 'kernel', 'archive',
        'service', 'worker', 'capacity', 'queue', 'session', 'prompt', 'output', 'batch']
    records = []
    for index in range(6000):
        chosen = rng.sample(words, 5)
        records.append(f'Record {index}: The {chosen[0]} subsystem received {rng.randrange(1,999)} '
            f'items and reserved {rng.randrange(16,4096)} units. The {chosen[1]} controller '
            f'recorded the {chosen[2]} condition, updated {chosen[3]}, and checked {chosen[4]}.\n')
    history = tok.encode(''.join(records), add_special_tokens=False)
    def build(length):
        count = length - len(prefix) - len(suffix)
        assert 0 <= count <= len(history)
        ids = prefix + history[:count] + suffix
        assert len(ids) == length
        return ids, [length - len(suffix), length]
    return build


def request_one(directory, ids, query_span, output_tokens, barrier=None, use_kvmem=True, session=None):
    directory.mkdir(parents=True, exist_ok=False)
    continuation = session is not None
    session = session or 'throughput-' + uuid.uuid4().hex
    body = dict(model='kvmem-throughput', prompt=ids, max_tokens=output_tokens,
        temperature=0, ignore_eos=True, return_token_ids=True, stream=True,
        stream_options={'include_usage': True}, cache_salt=session)
    if use_kvmem:
        body.update(session_id=session, kvmem_query_span=query_span,
            kvmem_reselect='auto', kvmem_mandatory_parts='none')
    save(directory/'request.json', body)
    if barrier:
        barrier.wait()
    start = time.perf_counter()
    started_at = time.time()
    chunks, output_ids, text, usage, finish, extra = [], [], '', None, None, {}
    done = False
    try:
        with http('/v1/completions', body, timeout=2400) as response:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                payload = raw[5:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                event = json.loads(payload)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    usage = event['usage']
                for key, value in event.items():
                    if key not in ('choices', 'usage', 'id', 'object', 'created', 'model'):
                        extra[key] = value
                for choice in event.get('choices', []):
                    token_ids = choice.get('token_ids') or []
                    if token_ids:
                        output_ids.extend(token_ids)
                        chunks.append([time.perf_counter()-start, len(token_ids)])
                    text += choice.get('text') or ''
                    finish = choice.get('finish_reason') or finish
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    elapsed = time.perf_counter() - start
    assert done and usage and chunks, ('Incomplete stream', done, usage, len(chunks))
    assert usage['prompt_tokens'] == len(ids), usage
    assert usage['completion_tokens'] == len(output_ids) == output_tokens, (usage, len(output_ids))
    assert finish == 'length', finish
    ttft = chunks[0][0]
    decode_seconds = chunks[-1][0] - ttft
    decode_tokens = len(output_ids) - chunks[0][1]
    row = dict(status='completed', session_id=session, started_at=started_at, input_tokens=len(ids),
        output_tokens=len(output_ids), input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        ttft_seconds=ttft, elapsed_seconds=elapsed,
        cold_input_tokens_per_second=None if continuation else len(ids)/ttft,
        request_mode='warm_continuation' if continuation else 'cold_input',
        decode_seconds=decode_seconds,
        decode_tokens_per_second=(decode_tokens/decode_seconds if decode_seconds > 0 else None),
        e2e_output_tokens_per_second=len(output_ids)/elapsed,
        first_chunk_tokens=chunks[0][1], finish_reason=finish, usage=usage,
        chunks=chunks, output_token_ids=output_ids, text=text, extra=extra,
        token_accounting_passed=True)
    row['start_perf_counter'] = start
    save(directory/'result.json', row)
    return row


def release_session(row, directory):
    session=row['session_id']
    body=dict(model='kvmem-throughput',session_id=session,cache_salt=session,
        prompt=[row['output_token_ids'][-1]],max_tokens=1,temperature=0,
        kv_transfer_params={'kvmem_release_session':True})
    with http('/v1/completions',body,timeout=120) as response:
        result=json.loads(response.read())
    assert not result.get('error'),result
    save(directory/'release.json',result)


def case(root, name, build, length, output, concurrency, repetitions, use_kvmem):
    directory = root/'cases'/name
    directory.mkdir(parents=True, exist_ok=False)
    ids, span = build(length)
    barrier = threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_one, directory/f'warmup-c{i+1}',
            ids, span, output, barrier, use_kvmem) for i in range(concurrency)]
        warmups = [f.result() for f in futures]
    if use_kvmem:
        for i,row in enumerate(warmups):
            release_session(row,directory/f'warmup-c{i+1}')
    all_rounds = []
    for repetition in range(repetitions):
        before = metrics()
        barrier = threading.Barrier(concurrency)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(request_one, directory/f'r{repetition+1}-c{i+1}',
                ids, span, output, barrier, use_kvmem) for i in range(concurrency)]
            rows = [f.result() for f in futures]
        seconds = time.perf_counter() - start
        after = metrics()
        delta = {key:value-before.get(key,0) for key,value in after.items()
            if value != before.get(key,0)}
        save(directory/f'metrics-r{repetition+1}.json', {'before':before,'after':after,'delta':delta})
        if use_kvmem:
            for i,row in enumerate(rows):
                release_session(row,directory/f'r{repetition+1}-c{i+1}')
        elapsed = max(r['elapsed_seconds'] for r in rows)
        tpot = [r['decode_seconds']/(output-r['first_chunk_tokens']) for r in rows
            if output > r['first_chunk_tokens'] and r['decode_seconds'] > 0]
        prefill_sums = [v for k,v in delta.items() if k.startswith('vllm:request_prefill_time_seconds_sum')]
        round_result = dict(repetition=repetition+1, input_tokens=length, output_tokens=output,
            concurrency=concurrency, wall_seconds=seconds, max_request_seconds=elapsed,
            aggregate_input_tokens_per_second=concurrency*length/seconds,
            aggregate_e2e_output_tokens_per_second=concurrency*output/seconds,
            aggregate_decode_tokens_per_second=common_decode(rows),
            mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in rows),
            cold_prefill_input_tokens_per_second=(length/rows[0]['ttft_seconds'] if concurrency==1 else None),
            server_prefill_seconds_sum=sum(prefill_sums) if prefill_sums else None,
            token_accounting_passed=all(r['token_accounting_passed'] for r in rows))
        all_rounds.append(round_result)
        save(directory/'summary.json', {'rounds':all_rounds})
        print(json.dumps({'case':name,**round_result}), flush=True)
    summary = {'case':name,'input_tokens':length,'output_tokens':output,'concurrency':concurrency,
        'repetitions':repetitions,'rounds':all_rounds}
    for key in ('aggregate_e2e_output_tokens_per_second','aggregate_decode_tokens_per_second',
            'mean_ttft_seconds','cold_prefill_input_tokens_per_second','server_prefill_seconds_sum'):
        values=[r[key] for r in all_rounds if r[key] is not None]
        summary['median_'+key]=statistics.median(values) if values else None
    save(directory/'summary.json',summary)
    return summary



def common_decode(rows):
    # Actual aggregate arrival count over the interval in which all requests decode.
    lo = max(r['start_perf_counter'] + r['chunks'][0][0] for r in rows)
    hi = min(r['start_perf_counter'] + r['chunks'][-1][0] for r in rows)
    if hi <= lo:
        return None
    tokens = sum(n for r in rows for t,n in r['chunks']
                 if lo < r['start_perf_counter'] + t <= hi)
    return tokens / (hi-lo)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--native', action='store_true')
    p.add_argument('--embedding-uva', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--compiled-dflash', action='store_true')
    p.add_argument('--capture-sizes', default='2,4,8')
    p.add_argument('--expandable-segments', action='store_true')
    p.add_argument('--allocator-split-mb', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-input', type=int, default=1024)
    p.add_argument('--attention-backend', choices=['FLASHINFER','TRITON_ATTN'], default='FLASHINFER')
    p.add_argument('--qualify', action='store_true')
    p.add_argument('--verify-restores', action='store_true')
    p.add_argument('--qualify-concurrency', type=int, default=2)
    p.add_argument('--qualify-padding', type=int, default=0)
    p.add_argument('--admission-input', type=int, default=0)
    p.add_argument('--bounded-logical', action='store_true')
    p.add_argument('--cpu-query-checkpoint', action='store_true')
    p.add_argument('--sweep-cases', help='Prepared-context decode cases, e.g. 4096:1,4096:2,32768:2')
    p.add_argument('--prefill-chunk', type=int, default=1024)
    p.add_argument('--decode-tokens', type=int, default=512)
    p.add_argument('--decode-warmup', type=int, default=128)
    p.add_argument('--checkpoint-gib', type=float, default=4)
    p.add_argument('--cpu-history-gib', type=float, default=16)
    p.add_argument('--cpu-per-request-gib', type=float, default=2)
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--seqs', type=int, default=4)
    p.add_argument('--max-submit', type=int, default=4)
    p.add_argument('--select', type=int, default=8192)
    p.add_argument('--gen', type=int, default=4096)
    p.add_argument('--max-len', type=int, default=65568)
    p.add_argument('--physical-chunks', type=int, default=2)
    p.add_argument('--draft-tokens', '--mtp', dest='mtp', type=int, default=7)
    p.add_argument('--draft-method', choices=['dflash','mtp'], default='dflash')
    p.add_argument('--strategy', choices=['legacy','layerwise'], default='legacy')
    p.add_argument('--util', type=float, default=.92)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--kv-gib', type=float)
    p.add_argument('--gc', choices=['none','lru','redundancy'], default='none',
        help='historical GC (delete mode, automatic watermarks); redundancy = agent fixture policy, thinking_only off')
    p.add_argument('--repository-blocks', type=int, help='per-request retained-history capacity in 32-token blocks')
    p.add_argument('--lifecycle-trace', action='store_true', help='KVMEM_LIFECYCLE_TRACE=1 (GC eviction trace per rank)')
    p.add_argument('--step-marks', action='store_true', help='per-step host phase marks (model_step_latency.jsonl per rank)')
    p.add_argument('--native-max-mib', type=int, help='action_native_max_bytes in MiB (default 512)')
    p.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
        help='extra server environment (e.g. KVMEM_* A/B flags); repeatable')
    p.add_argument('--dry-run-only', action='store_true', help='validate and save the launch plan, start no server')
    p.add_argument('--copy-check', metavar='CONCURRENCIES', help='greedy verbatim-copy check, e.g. 1,3,5,8 (copy_check_workload.py)')
    p.add_argument('--copy-lines', type=int, default=40, help='passage lines per copy request (~38 tokens each)')
    p.add_argument('--dflash-async', action='store_true',
        help='DFlash2 on the asynchronous refresh lifecycle (experimental_dflash_async_refresh): keeps layerwise selection and sessions; no idle residency')
    p.add_argument('--pure-decode', type=int, default=0, metavar='TOKENS',
        help='with --sweep-cases: one cold submission per owner with TOKENS output; report decode while all owners decode')
    p.add_argument('--native-default-backend', action='store_true',
        help="with --native: drop the harness's --attention-config so vLLM picks its default backend")
    p.add_argument('--native-vllm-defaults', action='store_true',
        help='with --native: keep vLLM defaults (prefix caching and async scheduling on)')
    p.add_argument('--portable-scores', action='store_true',
        help='KVMEM_NATIVE_ATTENTION_FORCE_UNSUPPORTED=1: portable reference scorer (the 4090/sm89 path)')
    args=p.parse_args()
    root=BASE/'runs'/args.label
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__),root/'benchmark.py')
    fixture=json.loads((BASE/'runtime/benchmarks/gate/agent.json').read_text())
    profile=fixture['profile']
    profile.pop('linear_backend_per_quant', None)
    profile.pop('draft_model', None)
    profile.update(model=MODEL,tokenizer=TOKENIZER,hf_config_path=MODEL,
        served_model_name='kvmem-throughput',port=PORT,max_model_len=args.max_len,
        max_num_seqs=args.seqs,tensor_parallel_size=args.tp,draft_tokens=args.mtp,
        execution_mode='eager' if args.eager else 'compiled_graph',
        gpu_memory_utilization=args.util,query_replay_budget=args.prefill_chunk,prefill_chunk_tokens=args.prefill_chunk,
        max_num_batched_tokens=args.prefill_chunk*args.physical_chunks+args.seqs*(args.mtp+1),
        request_planning_batch_tokens=args.prefill_chunk*args.physical_chunks+args.seqs*(args.mtp+1),
        kv_cache_memory_bytes=None if args.kv_gib is None else int(args.kv_gib*2**30),
        reasoning_end_str=None)
    kv=profile['kvmem']
    kv.update(select_budget=args.select,gen_budget=args.gen,session_capacity=max(8,args.seqs),
        cpu_read_cache_bytes=int(args.cpu_history_gib*2**30),cpu_history_cache_bytes_per_request=int(args.cpu_per_request_gib*2**30),
        session_checkpoint_bytes=int(args.checkpoint_gib*2**30),session_gpu_checkpoint_bytes=0,
        query_window_cache_bytes=256*2**20,action_native_max_bytes=512*2**20,
        action_download_max_bytes=256*2**20,
        cpu_seal_hash_workers=4,query_fps_cpu_workers=4,cpu_gather_workers=4,
        historical_gc=None,step_latency_interval=0,resident_restore_slots=1,
        action_native_prefill_static_aligned=False,prefill_physical_max_chunks=max(2,args.physical_chunks),
        strategy_pressure_scorer='lift',prefill_attention_half_life=32768)
    if args.gc != 'none':
        if args.gc == 'lru':
            gc=dict(mode='delete',policy='lru',policy_params={'keep_mandatory':True},poll_ms=10,scan_blocks=256,delete_blocks=32)
        else:
            # kv.update above already set the fixture's own historical_gc to None: reread it.
            gc=json.loads((BASE/'runtime/benchmarks/gate/agent.json').read_text())['profile']['kvmem']['historical_gc']
            gc['policy_params'].update(thinking_only=False)
            gc['policy_params'].pop('query_capture_bytes',None)
        gc.pop('ready_low',None); gc.pop('ready_high',None)
        kv['historical_gc']=gc
    if args.repository_blocks:
        kv['repository_blocks']=args.repository_blocks
    if args.step_marks:
        kv['step_latency_interval']=1
    if args.native_max_mib:
        kv['action_native_max_bytes']=args.native_max_mib*2**20
    if args.eager:
        kv.update(capture_opaque_projection=False,query_replay_backend='eager',query_replay_cobatch=False)
    profile.update(draft_method=args.draft_method,
        draft_model=str(BASE/'draft-w4a16') if args.draft_method=='dflash' else None)
    profile['verify_session_restore'] = args.qualify or args.verify_restores
    if args.draft_method=='dflash':
        profile.update(execution_mode='compiled_graph' if args.compiled_dflash else 'full_graph',
            experimental_dflash_query_planning=True)
        if args.compiled_dflash:
            profile['cudagraph_capture_sizes'] = [int(n) for n in args.capture_sizes.split(',')]
        if args.dflash_async:
            # Keep the MTP layerwise lifecycle (async refresh/restore, native scores); drop what is
            # MTP-only or not yet qualified with DFlash2 (coalescing, opaque capture, sessions, residency).
            profile['experimental_dflash_async_refresh']=True
            kv.update(recent_blocks=64,capture_opaque_projection=False,prefill_physical_coalescing=False,
                prefill_native_plan_arm=False,prefill_action_before_mtp=False,
                prefill_physical_graphless=False,prefill_physical_multi_owner=False,
                active_resident_pressure_only=False,active_resident_ttl_ms=0)
        else:
          kv.update(async_refresh=False,async_restore=False,recent_blocks=64,
            capture_opaque_projection=False,prefill_physical_coalescing=False,
            action_native_scores='off',query_replay_backend='eager',query_replay_cobatch=False,
            prefill_native_plan_arm=False,prefill_action_before_mtp=False,
            prefill_publication_pipeline=False,active_resident_pressure_only=False,
            prefill_physical_graphless=False,prefill_physical_multi_owner=False,
            restore_prepare_split=False,restore_handoff_fastpath=False,maintenance_max_steps=0)
    if args.strategy=='legacy':
        fixture['server_flags']=[]
        kv.update(strategy_pressure_scorer='attention')
        profile['settings']={'strategy':{}}
    else:
        profile['settings']={'strategy':{'KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS':4}}
        if args.draft_method=='dflash' and not args.dflash_async:
            fixture['server_flags'].remove('--layerwise-rolling-decode')
    if args.gc != 'none' and args.strategy=='layerwise':
        # Layerwise readers hold native read rights so historical GC cannot retire a live layer window.
        profile['settings']['strategy']['KVMEM_LAYERWISE_HISTORICAL_GC']=1
    if args.draft_method=='dflash':
        profile['settings']['system']={'KVMEM_CHECKPOINT_PINNED_PREWARM':0}
    save(root/'profile.json',profile)
    env={k:v for k,v in os.environ.items() if not k.startswith('KVMEM_')}
    env.update(PYTHONPATH=str(BASE/'runtime/src'),CUDA_HOME=CUDA,
        CC=HOST_CC,CXX=HOST_CXX,CUDAHOSTCXX=HOST_CXX,
        NVCC_CCBIN=HOST_CXX,
        PATH=str(BASE/'venv/bin')+':'+CUDA+'/bin:'+env.get('PATH',''),
        CPATH=CUDA+'/include/cccl',LD_LIBRARY_PATH=CUDA+'/lib64:'+env.get('LD_LIBRARY_PATH',''),
        TORCH_CUDA_ARCH_LIST=cuda_arch(),TRITON_PTXAS_PATH=CUDA+'/bin/ptxas',
        VLLM_CACHE_ROOT=str(BASE/'cache/vllm'),FLASHINFER_WORKSPACE_BASE=str(BASE/'cache/flashinfer'),
        TORCH_EXTENSIONS_DIR=str(BASE/'cache/extensions'),TRITON_CACHE_DIR=str(BASE/'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(BASE/'cache/inductor'),MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=[sys.executable,'-m','vllm_kvmem','serve','--config',str(root/'profile.json'),
        '--run-dir',str(root/'server'),*fixture['server_flags'],'--dry-run']
    dry=subprocess.run(command,env=env,capture_output=True,text=True)
    (root/'dry-run.stdout').write_text(dry.stdout)
    (root/'dry-run.stderr').write_text(dry.stderr)
    dry.check_returncode()
    plan=json.loads(dry.stdout)
    if args.dry_run_only:
        save(root/'plan.json',plan)
        print('DRY RUN OK', root)
        return
    if args.bounded_logical:
        plan['environment']['KVMEM_LOGICAL_KV_CAPACITY']='1'
    if args.cpu_query_checkpoint:
        plan['environment']['KVMEM_QUERY_CHECKPOINT_CPU']='1'
    for item in args.env:
        key,sep,value=item.partition('=')
        if not sep or not key:
            raise SystemExit(f'--env expects KEY=VALUE, got {item!r}')
        plan['environment'][key]=value
    if args.lifecycle_trace:
        plan['environment']['KVMEM_LIFECYCLE_TRACE']='1'
    if args.step_marks:
        plan['environment']['KVMEM_STEP_MARKS']='1'
    if args.portable_scores:
        plan['environment']['KVMEM_NATIVE_ATTENTION_FORCE_UNSUPPORTED']='1'
    argv=plan['vllm_argv']
    argv += ['--quantization','gptq_marlin','--attention-config',json.dumps({'backend':args.attention_backend})]
    if args.native:
        for key in ('--scheduler-cls','--kv-transfer-config','--worker-extension-cls','--middleware'):
            i=argv.index(key)
            del argv[i:i+2]
        env.update(VLLM_PLUGINS='',VLLM_USE_V2_MODEL_RUNNER='1')
        if args.native_default_backend:
            i=argv.index('--attention-config')
            del argv[i:i+2]
        if args.native_vllm_defaults:
            for key in ('--no-enable-prefix-caching','--no-async-scheduling'):
                if key in argv:
                    argv.remove(key)
    else:
        env.update(plan['environment'])
    env.update(BENCH_GRAPH_TRACE='1', BENCH_GRAPH_DIR=str(root))
    shutil.copy2(BASE/'entry.py',root/'entry.py')
    if args.allocator_split_mb:
        policy=f'max_split_size_mb:{args.allocator_split_mb}'
        env.update(PYTORCH_ALLOC_CONF=policy,PYTORCH_CUDA_ALLOC_CONF=policy)
    if args.expandable_segments:
        env.update(PYTORCH_ALLOC_CONF='expandable_segments:True',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if args.embedding_uva:
        env.update(BENCH_EMBEDDING_UVA='1',BENCH_EMBEDDING_EVIDENCE=str(root/'embedding-uva.json'))
        shutil.copy2(BASE/'entry.py',root/'entry.py')
    (root/'server/capture').mkdir(parents=True)
    save(root/'server/launch.json',plan)
    save(root/'launch.json',dict(argv=argv,environment={k:v for k,v in env.items()
        if k in plan['environment'] or k.startswith(('VLLM_','TORCH_','TRITON_','FLASHINFER_','BENCH_','PYTORCH_'))}))
    save(root/'versions.json',{pkg:importlib.metadata.version(pkg) for pkg in
        ('torch','vllm','triton','flashinfer-python','transformers')})
    save(root/'status.json',dict(stage='starting',at=time.time()))
    log=(root/'server.log').open('w')
    entry=[str(BASE/'entry.py')]
    proc=subprocess.Popen([sys.executable,'-u',*entry,*argv],
        env=env,cwd=BASE/'runtime',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'process.json',dict(pid=proc.pid,started_at=time.time()))
    monitor_log=(root/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,name,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm',
        '--format=csv','-l','1'],stdout=monitor_log,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+1800
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited {proc.returncode}')
            try:
                with http('/health',timeout=2) as response:
                    if response.status == 200: break
            except Exception: pass
            if time.monotonic()>deadline: raise TimeoutError('Server startup')
            time.sleep(2)
        print('SERVER READY',flush=True)
        save(root/'status.json',dict(stage='measuring',at=time.time()))
        if args.qualify:
            subprocess.run([sys.executable, str(BASE/'runtime/scripts/qualify_dflash_rolling.py'),
                '--config', str(root/'profile.json'), '--capture', str(root/'server/capture'),
                '--output', str(root/'qualification'), '--concurrency',str(args.qualify_concurrency),
                '--tool-padding-tokens',str(args.qualify_padding)], env=env,check=True)
            save(root/'status.json',dict(stage='completed',at=time.time()))
            return
        if args.copy_check:
            from copy_check_workload import run as run_copy
            run_copy(root,args,None)
            save(root/'status.json',dict(stage='completed',at=time.time()))
            return
        build=prompt_builder()
        if args.admission_input:
            from admission_workload import run
            run(root,args,build)
            save(root/'status.json',dict(stage='completed',at=time.time()))
            return
        if args.pure_decode:
            from pure_decode_workload import run as run_pure
            run_pure(root,args,build)
            save(root/'status.json',dict(stage='completed',at=time.time()))
            return
        if args.sweep_cases:
            # Prime public request, archive and session kernels before measuring
            # cold 200K histories. Every measured owner still has a fresh salt.
            case(root,'kernel-warmup',build,4096,32,args.seqs,0,not args.native)
            from concurrency_workload import run_sweep
            run_sweep(root,args,build)
            save(root/'status.json',dict(stage='completed',at=time.time()))
            return
        schedule=([(args.smoke_input,32,1)] if args.smoke else
            [(4096,1,1),(32768,1,1),
             (4096,512,1),(4096,512,2),(4096,512,4),
             (32768,512,1),(32768,512,2),(32768,512,4)])
        summaries=[]
        for length,output,conc in schedule:
            if conc > min(args.seqs,args.max_submit): continue
            name=f'in{length}-out{output}-c{conc}'
            summaries.append(case(root,name,build,length,output,conc,args.reps,not args.native))
            save(root/'summary.json',dict(cases=summaries))
        save(root/'status.json',dict(stage='completed',at=time.time()))
    except BaseException as e:
        save(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        save(root/'status.json',dict(stage='failed',at=time.time()))
        raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=20)
        monitor.terminate()
        monitor.wait(timeout=10)
        log.close();monitor_log.close()

if __name__ == '__main__':
    main()
