"""Artifact-independent launcher for the explicitly experimental serving path.

The profile supplies model locations and public connector options. Native vLLM
still owns the API, workers and shutdown; this launcher never edits its installed
files or borrows a benchmark's successful result as qualification evidence.
"""
import argparse
import json
import math
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

from .options import RuntimeOptions


DEFAULTS = {
    'host': '127.0.0.1', 'port': 8000, 'served_model_name': 'kvmem',
    'model_format': 'gguf',
    'execution_mode': 'eager',
    'verify_session_restore': False,
    'verify_session_restore_digest': False,
    'experimental_dflash_query_planning': False,
    'draft_tokens': 7,
    'draft_method': 'dflash',
    'draft_quantization': None,
    'profiler_config': None,
    'max_num_seqs': 4, 'max_num_batched_tokens': 4224,
    'long_prefill_token_threshold': 1024, 'gpu_memory_utilization': .72,
    'max_model_len': 262144, 'q8_tensorcore': False,
    'prefill_chunk_tokens': 0,
    'query_replay_budget': None,
    'reasoning_end_str': None,
    'cuda_allocator_conf': None,
    'recall_experiment': None,
}
PATHS = ('model', 'tokenizer', 'hf_config_path', 'draft_model', 'mm_proj')
# Explicit experiment controls for NVFP4 and FP8 checkpoints.
# Exclude weight-only/emulation backends: they change the activation precision.
LINEAR_BACKENDS = {
    'nvfp4_w4a4': frozenset(('auto', 'cutlass', 'flashinfer_cutlass', 'flashinfer_b12x', 'b12x')),
    'fp8_w8a8': frozenset(('auto', 'cutlass', 'flashinfer_cutlass', 'b12x', 'torch')),
}


def _validate_cuda_allocator_conf(value):
    if value is None:
        return None
    names = ('garbage_collection_threshold', 'per_process_memory_fraction')
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('cuda_allocator_conf requires native GC threshold and process fraction')
    if any(type(value[name]) not in (int, float) or not math.isfinite(value[name])
           or not 0 < value[name] < 1 for name in names):
        raise ValueError('cuda_allocator_conf fractions must be finite numbers strictly between zero and one')
    # In the qualified Torch allocator, GC is gated by an explicitly bounded
    # process fraction. A fraction of 1 clears that bound and silently disables
    # proactive GC, even if garbage_collection_threshold was configured.
    return {name: float(value[name]) for name in names}


def _validate_profiler_config(value, base):
    """Resolve opt-in local tracing without importing the engine in dry runs.

    Native ProfilerConfig remains responsible for validating its detailed
    options; the launcher only constrains the profiler kind and output path.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('profiler') != 'torch':
        raise ValueError('profiler_config must be a torch profiler object or null')
    directory = value.get('torch_profiler_dir')
    if (not isinstance(directory, str) or not directory.strip()
            or '\x00' in directory or '://' in directory):
        raise ValueError('profiler_config requires a local torch_profiler_dir')
    try:
        # Copy the JSON payload so callers cannot alter the launch plan later.
        config = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('profiler_config must contain JSON values') from exc
    path = Path(directory).expanduser()
    config['torch_profiler_dir'] = str((base / path if not path.is_absolute() else path).resolve())
    return config


def _validate_safetensors_config(model):
    """Check local config readability; native vLLM owns weight-format support.

    KVMem operates on model-produced state, independently of weight encoding.
    Do not maintain a second quantization whitelist or rewrite native metadata.
    Architecture, hardware and quantization-specific validation stays in vLLM;
    KV layout/scale and recurrent-state checks stay in the KVMem adapters.
    """
    try:
        config = json.loads((Path(model) / 'config.json').read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Safetensors model requires a readable config.json') from exc
    if not isinstance(config, dict):
        raise ValueError('Safetensors config.json must contain an object')


def build_launch(profile, base, run_dir, *, check_paths=True):
    if not isinstance(profile, dict) or set(profile) - set(DEFAULTS) - set(PATHS) - {
            'kvmem', 'multi_user', 'linear_backend_per_quant'}:
        raise ValueError('Serving profile contains unknown fields or is not an object')
    cfg = {**DEFAULTS, **profile}
    cfg['cuda_allocator_conf'] = _validate_cuda_allocator_conf(cfg['cuda_allocator_conf'])
    cfg['profiler_config'] = _validate_profiler_config(cfg['profiler_config'], base)
    if cfg['execution_mode'] not in ('eager', 'full_graph', 'compiled_graph'):
        raise ValueError('execution_mode must be eager, full_graph or compiled_graph')
    if type(cfg['verify_session_restore']) is not bool:
        raise ValueError('verify_session_restore must be boolean')
    if (type(cfg['verify_session_restore_digest']) is not bool or
            (cfg['verify_session_restore_digest'] and not cfg['verify_session_restore'])):
        raise ValueError('Digest verification requires enabled session restore verification')
    if type(cfg['experimental_dflash_query_planning']) is not bool:
        raise ValueError('experimental_dflash_query_planning must be boolean')
    if cfg['draft_method'] not in ('dflash', 'mtp'):
        raise ValueError('draft_method must be dflash or mtp')
    allowed_depths = (1, 3, 7) if cfg['draft_method'] == 'dflash' else (1, 2, 3, 4, 7)
    if type(cfg['draft_tokens']) is not int or cfg['draft_tokens'] not in allowed_depths:
        raise ValueError(f'draft_tokens must be one of {allowed_depths} for {cfg["draft_method"]}')
    if cfg['draft_quantization'] not in (None, 'fp8'):
        raise ValueError('draft_quantization must be null or fp8')
    if cfg['model_format'] not in ('gguf', 'safetensors'):
        raise ValueError('model_format must be gguf or safetensors')
    is_gguf = cfg['model_format'] == 'gguf'
    if cfg['execution_mode'] == 'compiled_graph' and (is_gguf or cfg['draft_method'] != 'mtp'):
        raise ValueError('Compiled capture currently requires native safetensors Qwen MTP')
    if cfg['draft_method'] == 'mtp' and (is_gguf or cfg['draft_quantization'] is not None):
        raise ValueError('Bounded MTP requires native safetensors quantization metadata')
    if not is_gguf and 'mm_proj' in cfg:
        raise ValueError('mm_proj is only supported for a GGUF target')
    if 'linear_backend_per_quant' in cfg:
        if is_gguf:
            raise ValueError('linear_backend_per_quant requires a safetensors target')
        backends = cfg['linear_backend_per_quant']
        if not isinstance(backends, dict) or not backends:
            raise ValueError('linear_backend_per_quant must be a non-empty object')
        for scheme, backend in backends.items():
            if scheme not in LINEAR_BACKENDS:
                raise ValueError(f'Unsupported linear_backend_per_quant scheme: {scheme}')
            if not isinstance(backend, str) or backend not in LINEAR_BACKENDS[scheme]:
                raise ValueError(f'Unsupported linear backend for {scheme}: {backend!r}')
        cfg['linear_backend_per_quant'] = dict(backends)
    for name in PATHS:
        if name == 'draft_model' and cfg['draft_method'] == 'mtp':
            continue
        if name == 'mm_proj' and not is_gguf:
            continue
        value = cfg.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Serving profile requires local {name}')
        path = Path(value).expanduser()
        path = (base / path if not path.is_absolute() else path).resolve()
        is_file = name == 'mm_proj' or (name == 'model' and is_gguf)
        if check_paths and not (path.is_file() if is_file else path.is_dir()):
            raise ValueError(f'Missing local {name}: {path}')
        cfg[name] = str(path)
    if is_gguf and Path(cfg['model']).suffix.lower() != '.gguf':
        raise ValueError('This serving profile currently qualifies the local GGUF target path')
    if not is_gguf:
        if cfg['hf_config_path'] != cfg['model']:
            raise ValueError('Safetensors hf_config_path must match model to preserve checkpoint quantization metadata')
        if check_paths:
            _validate_safetensors_config(cfg['model'])
    for name in ('max_num_seqs', 'max_num_batched_tokens', 'max_model_len', 'port'):
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    # Explicit Query planning resolves this native threshold to zero. Saved
    # resolved profiles must remain valid inputs to the launcher.
    if type(cfg['long_prefill_token_threshold']) is not int or cfg['long_prefill_token_threshold'] < 0:
        raise ValueError('long_prefill_token_threshold must be a nonnegative integer')
    if (cfg['port'] > 65535 or cfg['max_model_len'] > (10000000 if cfg['draft_method']=='mtp' else 262144)
            or cfg['max_num_batched_tokens'] < 8):
        raise ValueError('Port, draft logical-position range or verifier capacity is unsupported')
    utilization = cfg['gpu_memory_utilization']
    if type(utilization) not in (int, float) or not math.isfinite(utilization) or not 0 < utilization < 1:
        raise ValueError('gpu_memory_utilization must be strictly between zero and one')
    if type(cfg['q8_tensorcore']) is not bool:
        raise ValueError('q8_tensorcore must be boolean')
    if cfg['q8_tensorcore'] and not is_gguf:
        raise ValueError('q8_tensorcore is only supported for a GGUF target')
    for name in ('host', 'served_model_name'):
        if not isinstance(cfg[name], str) or not cfg[name] or '\x00' in cfg[name]:
            raise ValueError(f'Invalid {name}')
    payload = profile.get('kvmem', {})
    if not isinstance(payload, dict):
        raise ValueError('kvmem must contain public connector options')
    transfer = {'kv_connector': 'KVMemConnector', 'kv_role': 'kv_both',
                'kv_connector_module_path': 'vllm_kvmem.connector',
                'kv_connector_extra_config': {'kvmem': payload}}
    # This direct launcher is intrinsically the Mean-K backend. RuntimeOptions
    # validates graph modes here; the unchanged connector payload also carries
    # every graph setting to the worker (including explicit off/zero ablations).
    options = RuntimeOptions.from_vllm(SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=transfer['kv_connector_extra_config']),
        cache_config=SimpleNamespace(cache_dtype='fp8_e4m3')))
    if cfg['max_model_len'] > 1048576:
        gc = options.historical_gc
        if (cfg['draft_method'] != 'mtp' or gc is None or gc.mode != 'delete' or
                not options.request_planning or not options.rolling_active_window or
                not options.cpu_history_coordination or not options.cpu_progress_reserve or
                not options.session_capacity):
            raise ValueError('Logical histories above 1M require bounded MTP, owned rolling sessions, '
                             'deleting historical GC and physical CPU progress reservations')
    if (options.query_score_mode in ('overlap_16_8', 'key_fps_4') and
            os.environ.get('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES')):
        raise ValueError('Indexed Query scoring cannot be combined with experimental Query token scores')
    if cfg['experimental_dflash_query_planning'] and (
            cfg['draft_method'] != 'dflash' or not options.request_planning):
        raise ValueError('experimental_dflash_query_planning requires DFlash2 and request planning')
    from .options import canonical_policy_payload
    payload = canonical_policy_payload(payload, options)
    cfg['kvmem'] = payload
    transfer['kv_connector_extra_config']['kvmem'] = payload
    if options.graph_mode != 'off' and 'multi_user' in profile:
        raise ValueError('Block graph does not support the legacy direct multi_user middleware; '
                         'use elastic-serve or owned sessions on direct serve')
    if not options.memory.speculation:
        raise ValueError('The mainline serving profile requires the DFlash2 transaction contract')
    if options.prefill_batch_tokens and options.prefill_batch_tokens < options.memory.block_tokens * cfg['max_num_seqs']:
        raise ValueError('Shared prefill budget must fit one block per active request')
    if (options.offline_phase_policy and
            options.phase_prefill_wave_tokens < options.memory.block_tokens * cfg['max_num_seqs']):
        raise ValueError('Phase prefill wave must fit one block per active request')
    if options.active_residency_enabled and cfg['draft_method'] != 'mtp':
        raise ValueError('Idle active residency currently qualifies native Qwen MTP')
    if options.rolling_active_window and not (
            cfg['draft_method'] == 'mtp' and options.async_refresh and options.async_restore or
            cfg['draft_method'] == 'dflash' and cfg['experimental_dflash_query_planning'] and
            not options.async_refresh and not options.async_restore):
        raise ValueError('Rolling active window requires asynchronous native MTP or '
                         'experimental synchronous DFlash2')
    if cfg['max_model_len']>262144 and (not options.request_planning or
            options.memory.select_budget+options.memory.gen_budget>262144):
        raise ValueError('Long logical histories require explicit bounded MTP with RoPE window <=256K')
    if (type(cfg['prefill_chunk_tokens']) is not int or cfg['prefill_chunk_tokens']<0 or
            cfg['prefill_chunk_tokens']%options.memory.block_tokens or
            cfg['prefill_chunk_tokens']>options.memory.gen_budget):
        raise ValueError('Prefill chunk must be block aligned and fit G')
    if options.prefill_attention:
        chunk=cfg['prefill_chunk_tokens']
        if not chunk or (options.prefill_attention_stride and
                         (chunk+options.prefill_attention_stride-1)//options.prefill_attention_stride>64):
            raise ValueError('Prefill attention requires an explicit chunk with at most 64 sampled rows')
    if cfg['reasoning_end_str'] is not None and (not isinstance(cfg['reasoning_end_str'],str)
            or not cfg['reasoning_end_str'] or '\x00' in cfg['reasoning_end_str']):
        raise ValueError('Reasoning end string must be nonempty text')
    if options.request_planning:
        if cfg['draft_method'] != 'mtp':
            if not cfg['experimental_dflash_query_planning']:
                raise ValueError('Explicit Query planning currently requires native Qwen MTP; '
                                 'DFlash2 needs explicit experimental opt-in')
            if (cfg['draft_method'] != 'dflash' or cfg['draft_tokens'] != 7 or cfg['execution_mode'] != 'full_graph' or
                    options.async_refresh or options.async_restore or
                    options.memory.recent_blocks * options.memory.block_tokens < 2047):
                raise ValueError('Experimental DFlash2 planning requires k=7, FULL graphs, synchronous '
                                 'refresh/restore and a 2048-token recent band')
        if (options.session_capacity and not options.session_freeze_prompt and options.session_checkpoint_policy!='pm'
                and (options.session_capture_terminal or not options.session_capture_decode)):
            raise ValueError('Explicit Query sessions require sealed decode checkpoints, not terminal checkpoints')
        cfg['long_prefill_token_threshold'] = 0
        replay = cfg['query_replay_budget']
        if replay is None:
            replay = options.memory.gen_budget
        if type(replay) is not int or not options.memory.block_tokens <= replay <= options.memory.gen_budget:
            raise ValueError('Query replay budget must fit one block through G')
        cfg['query_replay_budget'] = replay
        # Generation is incremental. A large G does not require a single
        # equally large prefill batch; enforce this separate atomic Query cap
        # at the HTTP boundary before admitting work to the shared scheduler.
        cfg['max_num_batched_tokens'] = replay + cfg['max_num_seqs'] * (cfg['draft_tokens'] + 1)
    if options.offline_prefill_coalescing:
        baseline = cfg['prefill_chunk_tokens']
        maximum = options.offline_prefill_max_tokens
        if (not baseline or baseline % options.memory.block_tokens or
                not baseline < maximum <= 2 * baseline or maximum > cfg['max_num_batched_tokens']):
            raise ValueError('Offline prefill coalescing requires an aligned baseline and at most a twofold native-fit maximum')
    if options.prefill_physical_coalescing:
        physical = 2048*options.prefill_physical_max_chunks
        sample_rows = 64*options.prefill_physical_max_chunks
        if (cfg['draft_method'] != 'mtp' or cfg['prefill_chunk_tokens'] != 2048 or
                cfg['max_num_batched_tokens'] < physical or options.memory.gen_budget < physical):
            raise ValueError('Physical prefill coalescing requires native MTP, an unchanged '
                             '2048-token logical chunk and capacity for its maximum physical forward')
        # Match the existing native score-bank upper bound. Two original
        # physical leases can coexist, each retaining coordinates and masses.
        context = options.memory.select_budget+options.memory.gen_budget+cfg['max_num_batched_tokens']
        segments = ((context+options.memory.block_tokens-1)//options.memory.block_tokens +
                    2*cfg['max_num_seqs']+136)
        bucket = lambda n: max(256, 1 << (max(1, n)-1).bit_length())
        required = 2*(bucket(24*physical)+bucket(4*sample_rows*segments))
        if options.action_download_max_bytes < required:
            raise ValueError('Physical prefill coalescing exceeds the configured Action download byte budget')
    if (cfg['max_model_len'] < options.memory.select_budget + options.memory.gen_budget or
            cfg['long_prefill_token_threshold'] % options.memory.block_tokens):
        raise ValueError('Model context must cover K+G and prefill chunks must align to authority records')
    loader_args = (['--model-loader-extra-config', json.dumps({'mm_proj': cfg['mm_proj']})]
                   if is_gguf else [])
    # MTP inherits the target FP8 CacheConfig, including its engine-resolved
    # layout. An explicit same-dtype override makes a pre-resolution copy in
    # this pinned vLLM and loses the subsequent layout resolution.
    speculation = ({'method': 'mtp', 'num_speculative_tokens': cfg['draft_tokens']}
                   if cfg['draft_method'] == 'mtp' else
                   {'method': 'dflash', 'model': cfg['draft_model'],
                    'num_speculative_tokens': cfg['draft_tokens'], 'draft_load_config': {'load_format': 'safetensors'},
                    'kv_cache_dtype': 'bfloat16', **({'quantization': cfg['draft_quantization']}
                        if cfg['draft_quantization'] else {})})
    argv = ['serve', cfg['model'], '--served-model-name', cfg['served_model_name'],
        '--host', cfg['host'], '--port', str(cfg['port']), '--tokenizer', cfg['tokenizer'],
        '--hf-config-path', cfg['hf_config_path'], '--config-format', 'hf', '--dtype', 'bfloat16',
        '--language-model-only', *loader_args,
        '--kv-cache-dtype', 'fp8_e4m3', '--max-model-len', str(cfg['max_model_len']),
        '--max-num-seqs', str(cfg['max_num_seqs']), '--max-num-batched-tokens', str(cfg['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(utilization), '--no-enable-prefix-caching',
        '--generation-config', 'vllm',
        '--long-prefill-token-threshold', str(cfg['long_prefill_token_threshold']), '--no-async-scheduling',
        '--scheduler-cls', 'vllm_kvmem.scheduler.KVMemScheduler', '--kv-transfer-config', json.dumps(transfer),
        '--worker-extension-cls', 'vllm_kvmem.systemone_worker.SystemOneWorker',
        '--speculative-config', json.dumps(speculation),
        '--enable-per-request-metrics', '--per-request-spec-decode-metrics', 'summary',
        '--middleware', 'vllm_kvmem.http_sessions.validate_prompt_session',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3']
    env = {'KVMEM_SERVING': '1', 'KVMEM_RUNNER_PROBE_DIR': str(run_dir / 'capture'),
        'KVMEM_ARCHIVE_PROBE': '1', 'KVMEM_LIFECYCLE_TRACE': '1',
        'KVMEM_BOUNDED_TARGET': '1', 'KVMEM_WINDOW_METADATA': '1',
        'KVMEM_DRAFT_LOAD_COMPAT': '1', 'VLLM_USE_V2_MODEL_RUNNER': '1',
        'VLLM_PLUGINS': 'gguf,kvmem' if is_gguf else 'kvmem'}
    if cfg['recall_experiment'] is not None:
        from .recall_experiment import launch_environment
        env.update(launch_environment(cfg['recall_experiment'], options))
        cfg['recall_experiment'] = dict(cfg['recall_experiment'])
    if cfg['cuda_allocator_conf'] is not None:
        settings = 'backend:native,' + ','.join(
            f'{name}:{value}' for name, value in cfg['cuda_allocator_conf'].items())
        # An explicit profile overrides both modern and legacy environment
        # spellings consistently, before native workers import/init CUDA.
        env['PYTORCH_ALLOC_CONF'] = env['PYTORCH_CUDA_ALLOC_CONF'] = settings
    env['KVMEM_BOUNDED_MTP' if cfg['draft_method'] == 'mtp' else 'KVMEM_BOUNDED_DFLASH'] = '1'
    if options.graph_mode != 'off':
        # Collection also requires durable namespace ownership even when it
        # accompanies the legacy (unplanned) selector.
        env['KVMEM_REQUIRE_OWNED_SESSION'] = '1'
    # Persist explicit ablations in launch.json; ambient KVMEM variables are
    # deliberately cleared by main, so a saved profile is authoritative.
    for name in ('split_capture_dma', 'native_prefill_commit', 'async_checkpoint_capture'):
        env['KVMEM_' + name.upper()] = '1' if getattr(options, name) else '0'
    if cfg['max_model_len']>262144:
        # Target AND MTP consume the same dense K+G frame (mtp_runtime.py).
        # Only scheduler/session logical coordinates grow beyond the HF limit;
        # do not alter the checkpoint's rotary frequencies or quantization.
        env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    if cfg['max_model_len'] > 1048576:
        env['KVMEM_LOGICAL_KV_CAPACITY'] = '1'
    if cfg['prefill_chunk_tokens']:
        env['KVMEM_PREFILL_CHUNK_TOKENS'] = str(cfg['prefill_chunk_tokens'])
    if cfg['reasoning_end_str'] is not None:
        argv += ['--reasoning-config',json.dumps({'reasoning_end_str':cfg['reasoning_end_str']})]
    if cfg['execution_mode'] in ('full_graph', 'compiled_graph'):
        if options.query_replay_backend == 'compiled' and cfg['execution_mode'] != 'compiled_graph':
            raise ValueError('Compiled Query replay requires compiled_graph execution')
        compilation = ({'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'}
                       if cfg['execution_mode'] == 'compiled_graph' else
                       {'mode': 0, 'cudagraph_mode': 'FULL'})
        argv += ['--compilation-config', json.dumps(compilation)]
        env['KVMEM_FULL_GRAPH'] = '1'
    else:
        if options.query_replay_backend == 'compiled':
            raise ValueError('Compiled Query replay requires compiled_graph execution')
        argv += ['--enforce-eager']
    if getattr(options, 'capture_opaque_projection', False):
        if cfg['execution_mode'] != 'compiled_graph':
            raise ValueError('Opaque capture projection currently requires compiled_graph')
        # KVTransferConfig does not include its connector payload in vLLM's
        # compilation hash. This projection changes the compiled graph, so bind
        # its ABI and source to the supported additional-config hash instead of
        # reusing the old mutation/copy graph or clearing shared caches.
        from .capture_compile import CAPTURE_COMPILE_KEY, capture_compile_fingerprint
        argv += ['--additional-config', json.dumps({
            CAPTURE_COMPILE_KEY: capture_compile_fingerprint()})]
    if cfg['verify_session_restore']:
        env['KVMEM_VERIFY_SESSION_RESTORE'] = '1'
        if cfg['verify_session_restore_digest']:
            env['KVMEM_VERIFY_SESSION_RESTORE_DIGEST'] = '1'
    if not is_gguf:
        # Native vLLM resolves weight precision and kernels from the checkpoint.
        # Preserve all quantization metadata, including an unquantized config.
        argv += ['--load-format', 'safetensors']
    if 'linear_backend_per_quant' in cfg:
        argv += ['--kernel-config', json.dumps({
            'linear_backend_per_quant': cfg['linear_backend_per_quant']})]
    if cfg['q8_tensorcore']:
        env['KVMEM_Q8_TENSORCORE'] = '1'
    if cfg['profiler_config'] is not None:
        argv += ['--profiler-config', json.dumps(cfg['profiler_config'])]
    if 'multi_user' in profile:
        from .multi_user import validate_config
        settings = profile['multi_user']
        if not isinstance(settings, dict):
            raise ValueError('multi_user must contain admission settings and a users_file')
        settings = dict(settings)
        users_file = settings.get('users_file')
        if not isinstance(users_file, str) or not users_file:
            raise ValueError('multi_user requires a users_file')
        users_path = Path(users_file).expanduser()
        settings['users_file'] = str((base / users_path).resolve())
        settings = validate_config(settings, check_users=check_paths)
        if settings['max_inflight'] > cfg['max_num_seqs']:
            raise ValueError('multi_user.max_inflight must not exceed max_num_seqs')
        cfg['multi_user'] = settings
        argv[argv.index('--middleware') + 1] = 'vllm_kvmem.multi_user.MultiUserMiddleware'
        # Admission counters and tenant queues have a single API process owner.
        argv += ['--api-server-count', '1']
        env['KVMEM_MULTI_USER_CONFIG'] = json.dumps(settings)
    if options.request_planning:
        env['KVMEM_REQUEST_PLAN_CONFIG'] = json.dumps({'block_tokens': options.memory.block_tokens,
            'gen_budget': options.memory.gen_budget, 'select_budget': options.memory.select_budget,
            'sink_blocks': options.memory.sink_blocks, 'recent_blocks': options.memory.recent_blocks,
            'pressure_policy': options.pressure_policy,
            'guided_reserve':options.guided_reserve,'guided_reselect':options.guided_reselect,
            'max_model_len':cfg['max_model_len'],
            'query_replay_budget':cfg['query_replay_budget'],
            'rolling_active_window':options.rolling_active_window})
    return {'scope': 'experimental mainline serving; see support matrix for qualification limits',
            'profile': cfg, 'vllm_argv': argv, 'environment': env}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
        help='Explicit JSON profile; omit for the bundled DFlash2 k7 / GPU .80 agent profile')
    parser.add_argument('--run-dir', type=Path, required=True, help='New owned telemetry/archive directory')
    parser.add_argument('--dry-run', action='store_true', help='Validate local inputs and print the exact launch without importing vLLM')
    parser.add_argument('--verify-guided-pressure', action='store_true',
        help='Diagnostic: verify exact native state restoration after every private pressure Query')
    parser.add_argument('--retrieval-diagnostics', action='store_true',
        help='Diagnostic: save original Query, archived K and native-cache attention; synchronous and expensive')
    parser.add_argument('--experimental-query-token-scores', choices=('sum',),
        help='Answer experiment only: read archived raw K and sum global token attention by page')
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if args.config is None:
        from .default_profile import load_default_profile
        profile, base = load_default_profile(), Path.cwd()
    else:
        source = args.config.expanduser().resolve()
        profile, base = json.loads(source.read_text()), source.parent
    plan = build_launch(profile, base, run_dir)
    if args.retrieval_diagnostics:
        if profile.get('kvmem', {}).get('guided_query_tokens', 0):
            raise ValueError('Retrieval diagnostics require guided generation off')
        plan['environment']['KVMEM_RETRIEVAL_DIAGNOSTICS'] = '1'
    if args.experimental_query_token_scores:
        cfg = plan['profile'].get('kvmem', {})
        if cfg.get('query_score_mode', 'mean_k') != 'mean_k':
            raise ValueError('Indexed Query scoring cannot be combined with experimental Query token scores')
        if (cfg.get('guided_query_tokens', 0) or cfg.get('graph_alpha', .2) != 0 or
                cfg.get('query_history', False) or cfg.get('graph_mode') != 'multi' or
                cfg.get('block_tokens', 32) != 32):
            raise ValueError('Token answer experiment requires Query-only multi, 32-token blocks, no guidance/history')
        plan['environment']['KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES'] = args.experimental_query_token_scores
    if args.verify_guided_pressure:
        if profile.get('kvmem', {}).get('guided_reselect') != 'prefill':
            raise ValueError('Guided pressure verification requires guided_reselect=prefill')
        plan['environment']['KVMEM_VERIFY_GUIDED_PRESSURE'] = '1'
    if 'multi_user' in plan['profile'] and os.environ.get('VLLM_RUST_FRONTEND_PATH'):
        raise ValueError('Multi-user admission requires the Python API frontend')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    # A new directory prevents stale armed flags, cross-server trace replacement
    # and accidental reuse of another live worker's authoritative arena.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'capture').mkdir()
    (run_dir / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    for name in list(os.environ):
        if name.startswith('KVMEM_'):
            del os.environ[name]
    os.environ.update(plan['environment'])
    os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', '')
    if shutil.which('nvcc') is None:
        for cuda_dir in (os.environ.get('CUDA_HOME'), '/usr/local/cuda', '/usr/local/cuda-13'):
            if cuda_dir and (Path(cuda_dir) / 'bin/nvcc').is_file():
                os.environ['CUDA_HOME'] = cuda_dir
                os.environ['PATH'] = str(Path(cuda_dir) / 'bin') + ':' + os.environ['PATH']
                break
    if 'multi_user' in plan['profile']:
        # The tenant middleware owns authentication; an inherited shared key
        # would reject valid tenant keys again inside the native API stack.
        os.environ.pop('VLLM_API_KEY', None)
    from .compat import enable_draft_load_config_fix
    enable_draft_load_config_fix()
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ['vllm', *plan['vllm_argv']]
    vllm_main()
