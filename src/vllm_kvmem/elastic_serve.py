"""Launch a native graph/MTP offload backend and authenticated turn gateway."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import secrets
import signal

from .multi_user import _load_users
from .serve import _validate_safetensors_config, _validate_profiler_config
from .options import (GRAPH_DEFAULTS, canonical_policy_payload, resolve_graph_policy,
                      validate_graph_options, validate_query_score_mode, validate_query_fps_experiments,
                      validate_graph_query_trace_bytes)


def build_launch(profile, base, run_dir):
    if isinstance(profile, dict):
        # These fields are emitted in the resolved launch record, then rebuilt
        # from its canonical settings when that record is used as a profile.
        derived = {'geometry_file', 'backend_url', 'resolved_sink_blocks',
                   'resolved_recent_blocks', 'session_views_file', 'mean_k_backend_profile'}
        profile = {key: value for key, value in profile.items() if key not in derived}
    required = {'model', 'tokenizer', 'users_file'}
    defaults = {'host': '127.0.0.1', 'port': 18382, 'backend_port': 18381,
        'served_model_name': 'kvmem-elastic', 'max_model_len': 262144,
        'max_num_seqs': 32, 'max_num_batched_tokens': 8192,
        'gpu_memory_utilization': .85, 'mtp_k': 3, 'cpu_kv_bytes': 16 * 1024**3,
        'active_bytes': 32 * 1024**3, 'token_history_bytes': 64 * 1024**2,
        'max_sessions': 256, 'max_inflight': 8, 'max_queue': 128,
        'session_ttl_seconds': 3600, 'queue_timeout_seconds': 120,
        'turn_timeout_seconds': 600, 'max_body_bytes': 8 * 1024**2, 'max_output_tokens': 8192,
        'active_context_tokens': 0, 'restore_bytes': 8 * 1024**3, 'max_restore_inflight': 4,
        'token_history_bytes_per_user': 4 * 1024**2, 'max_sessions_per_user': 32,
        'context_mode': 'full', 'draft_model': None, 'gen_budget_tokens': 16384,
        'session_checkpoint_bytes': 2 * 1024**3, 'verify_session_restore': False, 'draft_tokens': 7,
        'linear_backend_per_quant': None, 'draft_quantization': None, 'profiler_config': None,
        'cpu_gather_workers': 8, 'async_seal': False, 'cpu_direct_record_seal': False,
        'cpu_seal_hash_workers': 1, 'cpu_seal_pack_workers': 1, 'cpu_seal_serial_allocate': False, 'cpu_mean_index_simd': False,
        'assembly_read_prefetch': False, 'assembly_fused_layers': False, 'draft_method': 'dflash',
        'gpu_raw_k_cache_bytes': 0, 'reuse_resident_v': False,'async_cpu_scatter':False,
        'cpu_scatter_min_tokens':128, 'execution_mode': 'full_graph', 'batched_cpu_append': False,
        'request_planning': False, 'mean_k_backend': 'torch_reference', 'session_checkpoint_policy':None,
        'query_score_mode': 'mean_k', 'query_window_cache_bytes': 2147483648,
        'query_fps_halo_blocks': 0, 'query_fps_optimized': False,
        'query_fps_prebuild': False,
        'query_fps_prebuild_device': 'cpu',
        'query_fps_cpu_backend': 'torch', 'query_fps_cpu_workers': 8,
        'query_mean_quota_fraction': 0.0,
        'graph_query_trace_max_bytes': 16777216,
        'query_history':False,'query_history_decay':0.5,'guided_query_tokens':0,
        'action_attention':False,'action_scope':'tool-call','action_layers':'spread3',
        'action_interval':16,'action_fraction':.25,'action_decay':.5,'action_combine':'quota',
        'action_native_decode_commit':True,'action_commit_max_bytes':16777216,
        'action_download_pool':True,'action_download_max_bytes':67108864,'action_download_max_slots':128,
        'action_fused_scoring':True,
        'action_native_scores':'off','action_native_max_rows':256,'action_native_max_bytes':2147483648,
        'archive_backend':'nvme','async_refresh':False,'refresh_quantum_tokens':256,
        'query_replay_backend':'eager','query_replay_cobatch':False,'query_replay_max_bytes':67108864,
        'prefill_attention':False,'action_pressure':False,'pressure_policy':None,'prefill_attention_interval':1024,
        'prefill_attention_stride':32,'prefill_attention_half_life':8192,'prefill_chunk_tokens':0,
        'block_tokens':128,'sink_blocks':None,'recent_blocks':None,
        'cpu_history_coordination':False,'cpu_progress_reserve':True,
        'maintenance_max_steps':0,'maintenance_busy_ms':2.0,'maintenance_idle_ms':8.0,
        'maintenance_aging_ms':250.0,'maintenance_inflight_bytes':0,'async_restore':False,
        'adaptive_prefill':False,'step_latency_interval':0,'decode_step_target_ms':50.0,
        'prefill_min_tokens':128,'prefill_max_tokens':2048,'prefill_feedback_timeout_ms':5000.0,'prefill_batch_tokens':0,'prefill_request_tokens':0,'active_resident_ttl_ms':0.0,'active_resident_max_fraction':0.5,
        'active_resident_pressure_only':False,
        'maintenance_transfer_bytes_per_second':0,'maintenance_transfer_burst_bytes':67108864,'offline_agent_policy':False,
        'repository_gc':False, 'historical_gc':None, **GRAPH_DEFAULTS}
    if not isinstance(profile, dict) or required - set(profile) or set(profile) - required - set(defaults):
        raise ValueError('Invalid elastic serving profile')
    cfg = {**defaults, **profile}
    cfg = resolve_graph_policy(cfg, explicit_fields=profile)
    validate_query_score_mode(cfg['query_score_mode'], cfg['query_window_cache_bytes'],
        block_tokens=cfg['block_tokens'], graph_mode=cfg['graph_mode'],
        graph_scoring_version=cfg['graph_scoring_version'],
        graph_aggregation=cfg['graph_aggregation'], graph_alpha=cfg['graph_alpha'],
        query_history=cfg['query_history'], guided_query_tokens=cfg['guided_query_tokens'],
        context_mode=cfg['context_mode'], mean_quota_fraction=cfg['query_mean_quota_fraction'],
        historical_gc=cfg['historical_gc'])
    validate_query_fps_experiments(cfg['query_score_mode'],
        cfg['query_fps_halo_blocks'], cfg['query_fps_optimized'], cfg['query_fps_prebuild'],
        cfg['query_fps_prebuild_device'], cfg['query_fps_cpu_backend'], cfg['query_fps_cpu_workers'],
        mean_quota_fraction=cfg['query_mean_quota_fraction'])
    validate_graph_query_trace_bytes(cfg['graph_query_trace_max_bytes'])
    if type(cfg['repository_gc']) is not bool:
        raise ValueError('repository_gc must be boolean')
    if cfg['historical_gc'] is not None:
        from .historical_gc import HistoricalGcSettings
        HistoricalGcSettings.parse(cfg['historical_gc'])
        if cfg['context_mode'] != 'meank':
            raise ValueError('historical_gc requires MeanK context management')
    validate_graph_options({name: cfg[name] for name in GRAPH_DEFAULTS},
        context_mode=cfg['context_mode'], request_planning=cfg['request_planning'],
        session_capacity=cfg['max_sessions'], repository_gc=cfg['repository_gc'],
        query_history=cfg['query_history'], action_pressure=cfg['action_pressure'],
        action_attention=cfg['action_attention'], prefill_attention=cfg['prefill_attention'],
        action_scope=cfg['action_scope'], action_fraction=cfg['action_fraction'],
        pressure_policy=cfg['pressure_policy'])
    if cfg['repository_gc'] and cfg['context_mode'] != 'meank':
        raise ValueError('repository_gc requires MeanK context management')
    if (cfg['cpu_history_coordination'] or cfg['maintenance_max_steps'] or cfg['async_restore'] or cfg['adaptive_prefill'] or cfg['step_latency_interval'] or cfg['prefill_batch_tokens'] or cfg['prefill_request_tokens'] or cfg['active_resident_ttl_ms'] or cfg['active_resident_pressure_only'] or cfg['maintenance_transfer_bytes_per_second'] or cfg['offline_agent_policy']) and cfg['context_mode'] != 'meank':
        raise ValueError('History/maintenance coordination requires MeanK context management')
    if cfg['archive_backend'] not in ('nvme','cpu'):
        raise ValueError('Archive backend must be nvme or cpu')
    if cfg['action_attention'] and cfg['context_mode']!='meank':
        raise ValueError('Action attention requires MeanK historical context management')
    if (cfg['prefill_attention'] or cfg['action_pressure'] or cfg['prefill_chunk_tokens']) and cfg['context_mode']!='meank':
        raise ValueError('Prefill/pressure attention requires MeanK historical context management')
    if type(cfg['prefill_chunk_tokens']) is not int or cfg['prefill_chunk_tokens'] < 0:
        raise ValueError('prefill_chunk_tokens must be a nonnegative integer')
    if type(cfg['async_refresh']) is not bool or (cfg['async_refresh'] and cfg['context_mode']!='meank'):
        raise ValueError('Asynchronous refresh requires MeanK context management')
    if (cfg['query_history'] or cfg['guided_query_tokens']) and not cfg['request_planning']:
        raise ValueError('Guided refresh/history require explicit request planning')
    cfg['session_checkpoint_policy'] = cfg['session_checkpoint_policy'] or ('pm' if cfg['request_planning'] else 'sealed')
    if 'session_checkpoint_bytes' not in profile and cfg['session_checkpoint_policy']=='pm':
        # P and M each own a recurrent-model snapshot. Keep both for the
        # qualified eight-user Qwen 27B profile; explicit quotas stay authoritative.
        cfg['session_checkpoint_bytes'] = 4 * 1024**3
    if cfg['session_checkpoint_policy'] not in ('sealed','pm'):
        raise ValueError('Invalid checkpoint policy')
    if type(cfg['request_planning']) is not bool or cfg['mean_k_backend'] not in ('torch_reference','qw3_cuda'):
        raise ValueError('Invalid Query planning or MeanK backend')
    if cfg['request_planning'] and (cfg['context_mode'] != 'meank' or cfg['draft_method'] != 'mtp'):
        raise ValueError('Explicit Query planning requires MeanK/native MTP')
    cfg['profiler_config'] = _validate_profiler_config(cfg['profiler_config'], base)
    if type(cfg['cpu_gather_workers']) is not int or not 1 <= cfg['cpu_gather_workers'] <= 16:
        raise ValueError('cpu_gather_workers must be an integer in [1, 16]')
    if cfg['execution_mode'] not in ('full_graph', 'compiled_graph'):
        raise ValueError('execution_mode must be full_graph or compiled_graph')
    if cfg['context_mode'] not in ('full', 'meank'):
        raise ValueError('context_mode must be full or meank')
    if type(cfg['verify_session_restore']) is not bool:
        raise ValueError('verify_session_restore must be boolean')
    if type(cfg['async_seal']) is not bool:
        raise ValueError('async_seal must be boolean')
    if type(cfg['cpu_direct_record_seal']) is not bool:
        raise ValueError('cpu_direct_record_seal must be boolean')
    if cfg['cpu_direct_record_seal'] and (cfg['archive_backend'] != 'cpu' or not cfg['async_seal']):
        raise ValueError('cpu_direct_record_seal requires archive_backend=cpu and async_seal=true')
    if cfg['cpu_direct_record_seal'] and cfg['context_mode'] != 'meank':
        raise ValueError('cpu_direct_record_seal requires MeanK context management')
    if type(cfg['cpu_seal_hash_workers']) is not int or not 1 <= cfg['cpu_seal_hash_workers'] <= 8:
        raise ValueError('cpu_seal_hash_workers must be an integer in [1, 8]')
    if type(cfg['cpu_seal_pack_workers']) is not int or not 1 <= cfg['cpu_seal_pack_workers'] <= 8:
        raise ValueError('cpu_seal_pack_workers must be an integer in [1, 8]')
    if cfg['cpu_seal_pack_workers'] > 1 and (not cfg['cpu_direct_record_seal'] or cfg['historical_gc'] is not None):
        raise ValueError('cpu_seal_pack_workers > 1 requires direct CPU record sealing without historical_gc')
    if type(cfg['cpu_seal_serial_allocate']) is not bool:
        raise ValueError('cpu_seal_serial_allocate must be boolean')
    if cfg['cpu_seal_serial_allocate'] and (not cfg['cpu_direct_record_seal'] or
            cfg['historical_gc'] is not None or cfg['cpu_seal_pack_workers'] != 1):
        raise ValueError('cpu_seal_serial_allocate requires direct CPU records without historical_gc and pack_workers=1')
    if cfg['query_replay_backend'] not in ('eager', 'compiled') or type(cfg['query_replay_cobatch']) is not bool:
        raise ValueError('Invalid Query replay backend or cobatch setting')
    if type(cfg['query_replay_max_bytes']) is not int or cfg['query_replay_max_bytes'] <= 0:
        raise ValueError('Query replay memory budget must be positive')
    if cfg['query_replay_backend'] != 'eager' or cfg['query_replay_cobatch']:
        if cfg['context_mode'] != 'meank' or not cfg['async_refresh'] or cfg['draft_method'] != 'mtp':
            raise ValueError('Optimized Query replay requires asynchronous MeanK MTP')
        if cfg['query_replay_backend'] == 'compiled' and cfg['execution_mode'] != 'compiled_graph':
            raise ValueError('Compiled Query replay requires compiled_graph execution')
    if type(cfg['cpu_mean_index_simd']) is not bool:
        raise ValueError('cpu_mean_index_simd must be boolean')
    if cfg['cpu_mean_index_simd']:
        if cfg['context_mode'] != 'meank':
            raise ValueError('cpu_mean_index_simd requires MeanK context management')
        if cfg['mean_k_backend'] != 'qw3_cuda':
            raise ValueError('cpu_mean_index_simd requires mean_k_backend=qw3_cuda')
    if type(cfg['assembly_read_prefetch']) is not bool:
        raise ValueError('assembly_read_prefetch must be boolean')
    if cfg['assembly_read_prefetch'] and (cfg['archive_backend'] != 'cpu' or cfg['context_mode'] != 'meank'):
        raise ValueError('assembly_read_prefetch requires CPU archive and MeanK context management')
    if type(cfg['assembly_fused_layers']) is not bool:
        raise ValueError('assembly_fused_layers must be boolean')
    if cfg['assembly_fused_layers'] and (cfg['archive_backend'] != 'cpu' or cfg['context_mode'] != 'meank'):
        raise ValueError('assembly_fused_layers requires CPU archive and MeanK context management')
    if cfg['cpu_seal_hash_workers'] > 1:
        if cfg['archive_backend'] != 'cpu' or not cfg['async_seal']:
            raise ValueError('cpu_seal_hash_workers > 1 requires archive_backend=cpu and async_seal=true')
        if cfg['context_mode'] != 'meank':
            raise ValueError('cpu_seal_hash_workers > 1 requires MeanK context management')
    if type(cfg['reuse_resident_v']) is not bool:
        raise ValueError('reuse_resident_v must be boolean')
    if type(cfg['batched_cpu_append']) is not bool:
        raise ValueError('batched_cpu_append must be boolean')
    if type(cfg['async_cpu_scatter']) is not bool:
        raise ValueError('async_cpu_scatter must be boolean')
    if type(cfg['gpu_raw_k_cache_bytes']) is not int or cfg['gpu_raw_k_cache_bytes'] < 0:
        raise ValueError('gpu_raw_k_cache_bytes must be a nonnegative byte count')
    for name in ('host', 'served_model_name'):
        if not isinstance(cfg[name], str) or not cfg[name] or '\x00' in cfg[name]:
            raise ValueError(f'{name} must be a nonempty string')
    for name in required:
        path = Path(cfg[name]).expanduser()
        cfg[name] = str((base / path if not path.is_absolute() else path).resolve())
    _validate_safetensors_config(cfg['model'])
    if not Path(cfg['tokenizer']).is_dir():
        raise ValueError('Missing tokenizer directory')
    _load_users(cfg['users_file'])
    for name in set(defaults) - set(GRAPH_DEFAULTS) - {'host', 'served_model_name', 'gpu_memory_utilization', 'active_context_tokens',
                               'context_mode', 'draft_model', 'verify_session_restore', 'linear_backend_per_quant',
                               'draft_quantization', 'profiler_config', 'async_seal', 'cpu_direct_record_seal', 'cpu_seal_serial_allocate', 'cpu_mean_index_simd', 'assembly_read_prefetch', 'assembly_fused_layers', 'draft_method',
                               'reuse_resident_v', 'gpu_raw_k_cache_bytes','async_cpu_scatter', 'execution_mode', 'batched_cpu_append',
                               'request_planning', 'mean_k_backend', 'query_score_mode', 'session_checkpoint_policy',
                               'query_fps_halo_blocks', 'query_fps_optimized', 'query_fps_prebuild', 'query_fps_prebuild_device',
                               'query_fps_cpu_backend', 'query_fps_cpu_workers',
                               'query_mean_quota_fraction',
                               'query_history','query_history_decay','guided_query_tokens','repository_gc','historical_gc',
                               'query_replay_backend','query_replay_cobatch',
                               'action_attention','action_scope','action_layers','action_interval',
                               'action_native_scores','action_native_max_rows','action_native_max_bytes',
                               'action_native_decode_commit','action_commit_max_bytes',
                               'action_download_pool','action_download_max_bytes','action_download_max_slots',
                               'action_fused_scoring',
                               'action_fraction','action_decay','action_combine','archive_backend','async_refresh',
                               'prefill_attention','action_pressure','pressure_policy','prefill_attention_stride','prefill_attention_half_life','prefill_chunk_tokens',
                               'sink_blocks','recent_blocks','cpu_history_coordination','cpu_progress_reserve',
                               'maintenance_max_steps','maintenance_busy_ms','maintenance_idle_ms',
                               'maintenance_aging_ms','maintenance_inflight_bytes','async_restore',
                               'adaptive_prefill','step_latency_interval','decode_step_target_ms',
                               'prefill_min_tokens','prefill_max_tokens','prefill_feedback_timeout_ms','prefill_batch_tokens','prefill_request_tokens','active_resident_ttl_ms','active_resident_pressure_only','active_resident_max_fraction','maintenance_transfer_bytes_per_second','maintenance_transfer_burst_bytes','offline_agent_policy'}:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if type(cfg['active_context_tokens']) is not int or not 0 <= cfg['active_context_tokens'] <= cfg['max_model_len']:
        raise ValueError('active_context_tokens must be zero or a bounded positive token count')
    utilization = cfg['gpu_memory_utilization']
    if ((cfg['context_mode'] != 'meank' and cfg['mtp_k'] not in (2, 3)) or
            type(utilization) not in (int, float) or not 0 < utilization < 1):
        raise ValueError('Unsupported MTP depth or GPU allocation')
    if cfg['port'] == cfg['backend_port'] or max(cfg['port'], cfg['backend_port']) > 65535:
        raise ValueError('Gateway and backend require distinct valid ports')
    if not 1 <= cfg['max_inflight'] <= cfg['max_num_seqs']:
        raise ValueError('Active count must fit native maximum requests')
    if cfg['restore_bytes'] > cfg['active_bytes'] or cfg['max_restore_inflight'] > cfg['max_inflight']:
        raise ValueError('Restore reservations must fit the active budget')
    if cfg['token_history_bytes_per_user'] > cfg['token_history_bytes'] or cfg['max_sessions_per_user'] > cfg['max_sessions']:
        raise ValueError('User history limits must fit the global budget')
    run_dir = run_dir.resolve()
    cfg['geometry_file'] = str(run_dir / 'cache-geometry.json')
    cfg['backend_url'] = f"http://127.0.0.1:{cfg['backend_port']}"
    if cfg['context_mode'] == 'meank':
        from .serve import build_launch as build_mean_k_launch
        if cfg['draft_method'] not in ('dflash', 'mtp'):
            raise ValueError('MeanK draft_method must be dflash or mtp')
        if ((cfg['draft_method'] == 'dflash' and
                (not isinstance(cfg['draft_model'], str) or not cfg['draft_model'])) or
                not cfg['active_context_tokens'] or cfg['max_output_tokens'] > cfg['gen_budget_tokens']):
            raise ValueError('MeanK requires a local DFlash2 draft, active context and output within G')
        if cfg['draft_method'] == 'dflash':
            draft = Path(cfg['draft_model']).expanduser()
            cfg['draft_model'] = str((base / draft if not draft.is_absolute() else draft).resolve())
        backend = {'model_format': 'safetensors', 'model': cfg['model'], 'tokenizer': cfg['tokenizer'],
            'hf_config_path': cfg['model'], 'draft_model': cfg['draft_model'], 'execution_mode': cfg['execution_mode'],
            'verify_session_restore': cfg['verify_session_restore'],
            'draft_tokens': cfg['draft_tokens'],
            'draft_method': cfg['draft_method'],
            'draft_quantization': cfg['draft_quantization'],
            'profiler_config': cfg['profiler_config'],
            'host': '127.0.0.1', 'port': cfg['backend_port'], 'served_model_name': cfg['served_model_name'],
            'max_model_len': cfg['max_model_len'], 'max_num_seqs': cfg['max_num_seqs'],
            'max_num_batched_tokens': cfg['max_num_batched_tokens'], 'long_prefill_token_threshold': 1024,
            'prefill_chunk_tokens':cfg['prefill_chunk_tokens'],
            'gpu_memory_utilization': cfg['gpu_memory_utilization'],
            'kvmem': {'select_budget': cfg['active_context_tokens'], 'gen_budget': cfg['gen_budget_tokens'],
                'block_tokens':cfg['block_tokens'],'sink_blocks':cfg['sink_blocks'],'recent_blocks':cfg['recent_blocks'],
                'repository_blocks': ((cfg['max_model_len'] + cfg['block_tokens'] - 1) // cfg['block_tokens'] +
                    (2+2*((cfg['draft_tokens']+1+cfg['block_tokens']-1)//cfg['block_tokens'])
                     if cfg['historical_gc'] is not None else 0)),
                'cpu_read_cache_bytes': cfg['cpu_kv_bytes'],
                'archive_backend': cfg['archive_backend'],
                'repository_gc': cfg['repository_gc'],
                'historical_gc': cfg['historical_gc'],
                **{key: cfg[key] for key in GRAPH_DEFAULTS},
                'cpu_gather_workers': cfg['cpu_gather_workers'],
                'async_seal': cfg['async_seal'],
                'cpu_direct_record_seal': cfg['cpu_direct_record_seal'],
                'cpu_seal_hash_workers': cfg['cpu_seal_hash_workers'],
                'cpu_seal_pack_workers': cfg['cpu_seal_pack_workers'],
                'cpu_seal_serial_allocate': cfg['cpu_seal_serial_allocate'],
                'cpu_mean_index_simd': cfg['cpu_mean_index_simd'],
                'assembly_read_prefetch': cfg['assembly_read_prefetch'],
                'assembly_fused_layers': cfg['assembly_fused_layers'],
                'gpu_raw_k_cache_bytes': cfg['gpu_raw_k_cache_bytes'],
                'reuse_resident_v': cfg['reuse_resident_v'],
                'async_cpu_scatter':cfg['async_cpu_scatter'],
                'batched_cpu_append':cfg['batched_cpu_append'],
                'cpu_scatter_min_tokens':cfg['cpu_scatter_min_tokens'],
                'async_refresh':cfg['async_refresh'],
                'refresh_quantum_tokens':cfg['refresh_quantum_tokens'],
                'query_replay_backend':cfg['query_replay_backend'],
                'query_replay_cobatch':cfg['query_replay_cobatch'],
                'query_replay_max_bytes':cfg['query_replay_max_bytes'],
                **{key:cfg[key] for key in ('cpu_history_coordination','cpu_progress_reserve',
                    'maintenance_max_steps','maintenance_busy_ms','maintenance_idle_ms',
                    'maintenance_aging_ms','maintenance_inflight_bytes','async_restore',
                    'adaptive_prefill','step_latency_interval','decode_step_target_ms',
                    'prefill_min_tokens','prefill_max_tokens','prefill_feedback_timeout_ms','prefill_batch_tokens','prefill_request_tokens','active_resident_ttl_ms','active_resident_pressure_only','active_resident_max_fraction','maintenance_transfer_bytes_per_second','maintenance_transfer_burst_bytes','offline_agent_policy')},
                # The archive clips this to its actual historical record
                # geometry. MTP adds a layer; do not retain the target-only
                # 12 GiB constant and leave global CPU capacity unused.
                'cpu_history_cache_bytes_per_request': cfg['cpu_kv_bytes'],
                'packed_assembly': True, 'direct_pinned_reads': True, 'prefill_cadence': 1,
                'async_archive_retirement': True, 'verify_archive_on_finish': False,
                'session_capacity': cfg['max_sessions'], 'session_checkpoint_bytes': cfg['session_checkpoint_bytes'],
                'session_capture_terminal': not cfg['request_planning'] or cfg['session_checkpoint_policy']=='pm',
                'session_capture_decode': cfg['request_planning'] and cfg['session_checkpoint_policy']=='sealed',
                'session_checkpoint_policy':cfg['session_checkpoint_policy'],
                'request_planning': cfg['request_planning'], 'mean_k_backend': cfg['mean_k_backend'],
                'query_score_mode': cfg['query_score_mode'],
                'query_window_cache_bytes': cfg['query_window_cache_bytes'],
                'query_fps_halo_blocks': cfg['query_fps_halo_blocks'],
                'query_fps_optimized': cfg['query_fps_optimized'],
                'query_fps_prebuild': cfg['query_fps_prebuild'],
                'query_fps_prebuild_device': cfg['query_fps_prebuild_device'],
                'query_fps_cpu_backend': cfg['query_fps_cpu_backend'],
                'query_fps_cpu_workers': cfg['query_fps_cpu_workers'],
                'query_mean_quota_fraction': cfg['query_mean_quota_fraction'],
                'graph_query_trace_max_bytes': cfg['graph_query_trace_max_bytes'],
                'query_history':cfg['query_history'],'query_history_decay':cfg['query_history_decay'],
                'guided_query_tokens':cfg['guided_query_tokens'],
                **{key:cfg[key] for key in ('action_attention','action_scope','action_layers',
                    'action_interval','action_fraction','action_decay','action_combine',
                    'action_native_scores','action_native_max_rows','action_native_max_bytes',
                    'action_native_decode_commit','action_commit_max_bytes',
                    'action_download_pool','action_download_max_bytes','action_download_max_slots',
                    'action_fused_scoring',
                    'prefill_attention','action_pressure','pressure_policy','prefill_attention_interval',
                    'prefill_attention_stride','prefill_attention_half_life')}}}
        backend['kvmem'] = canonical_policy_payload(backend['kvmem'], cfg)
        if cfg['linear_backend_per_quant'] is not None:
            backend['linear_backend_per_quant'] = cfg['linear_backend_per_quant']
        # Validate the concrete native arguments before creating any resources.
        concrete = build_mean_k_launch(backend, run_dir, run_dir / 'meank')
        if cfg['request_planning']:
            plan_config = json.loads(concrete['environment']['KVMEM_REQUEST_PLAN_CONFIG'])
            cfg['resolved_sink_blocks'] = plan_config['sink_blocks']
            cfg['resolved_recent_blocks'] = plan_config['recent_blocks']
        cfg['geometry_file'] = str(run_dir / 'meank/capture/cache-geometry.json')
        cfg['session_views_file'] = str(run_dir / 'meank/capture/scheduler_resources.json')
        cfg['mean_k_backend_profile'] = backend
        cfg['mtp_k'] = cfg['draft_tokens'] if cfg['draft_method'] == 'mtp' else None
        cfg = canonical_policy_payload(cfg, cfg)
        return cfg, [sys.executable, '-m', 'vllm_kvmem', 'serve',
            '--config', str(run_dir / 'backend-profile.json'), '--run-dir', str(run_dir / 'meank')]
    if cfg['linear_backend_per_quant'] is not None:
        raise ValueError('Explicit linear backends currently require the MeanK profile')
    cfg['draft_method'] = 'mtp'
    transfer = {'kv_connector': 'KVMemElasticConnector', 'kv_role': 'kv_both',
        'kv_connector_module_path': 'vllm_kvmem.elastic_connector',
        'kv_connector_extra_config': {'cpu_bytes_to_use': cfg['cpu_kv_bytes'],
            'blocks_per_chunk': 1, 'offload_prompt_only': False, 'kvmem_report_dir': str(run_dir)}}
    argv = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', cfg['model'],
        '--tokenizer', cfg['tokenizer'], '--served-model-name', cfg['served_model_name'],
        '--host', '127.0.0.1', '--port', str(cfg['backend_port']), '--dtype', 'bfloat16',
        '--load-format', 'safetensors', '--language-model-only', '--kv-cache-dtype', 'fp8_e4m3',
        '--max-model-len', str(cfg['max_model_len']), '--max-num-seqs', str(cfg['max_num_seqs']),
        '--max-num-batched-tokens', str(cfg['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(cfg['gpu_memory_utilization']), '--generation-config', 'vllm',
        '--enable-prefix-caching', '--mamba-cache-mode', 'align', '--async-scheduling',
        '--long-prefill-token-threshold', '0',
        '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': cfg['mtp_k']}),
        '--compilation-config', json.dumps({'cudagraph_mode': 'FULL_AND_PIECEWISE'}),
        '--kv-transfer-config', json.dumps(transfer), '--enable-per-request-metrics',
        '--per-request-spec-decode-metrics', 'summary',
        '--middleware', 'vllm_kvmem.elastic_control.ElasticControlMiddleware']
    if cfg['profiler_config'] is not None:
        argv += ['--profiler-config', json.dumps(cfg['profiler_config'])]
    return canonical_policy_payload(cfg, cfg), argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    path, run_dir = Path(args.config).resolve(), Path(args.run_dir).resolve()
    cfg, command = build_launch(json.loads(path.read_text()), path.parent, run_dir)
    if args.dry_run:
        print(json.dumps({'config': cfg, 'argv': command}, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=False)
    if cfg['context_mode'] == 'meank':
        (run_dir / 'backend-profile.json').write_text(json.dumps(cfg['mean_k_backend_profile'], indent=2) + '\n')
    (run_dir / 'launch.json').write_text(json.dumps({'config': cfg, 'argv': command}, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('KVMEM_', 'VLLM_'))}
    env.update(VLLM_PLUGINS='', VLLM_USE_V2_MODEL_RUNNER='1', VLLM_LOG_STATS_INTERVAL='1')
    admin_file = run_dir / 'admin.key'
    fd = os.open(admin_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as secret:
        secret.write(secrets.token_urlsafe(32))
    env['KVMEM_ELASTIC_ADMIN_FILE'] = str(admin_file)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    env['PATH'] = str(Path(sys.executable).parent) + ':' + env.get('PATH', '')
    if shutil.which('nvcc', path=env['PATH']) is None:
        for cuda_dir in (env.get('CUDA_HOME'), '/usr/local/cuda', '/usr/local/cuda-13'):
            if cuda_dir and (Path(cuda_dir) / 'bin/nvcc').is_file():
                env['CUDA_HOME'] = cuda_dir
                env['PATH'] = str(Path(cuda_dir) / 'bin') + ':' + env['PATH']
                break
    with (run_dir / 'backend.log').open('w') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run_dir / 'backend-api-pid').write_text(str(server.pid))
    # uvicorn restores and re-raises SIGTERM after graceful gateway shutdown.
    # A Python handler lets finally release the owned GPU backend as well.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    old_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        import httpx
        deadline = time.monotonic() + 900
        with httpx.Client(trust_env=False, timeout=2) as client:
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Native backend failed; inspect backend.log')
                try:
                    if (client.get(cfg['backend_url'] + '/health').status_code == 200
                            and Path(cfg['geometry_file']).is_file()):
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Native backend startup timed out')
                time.sleep(1)
        from .elastic_gateway import create_app
        import uvicorn
        uvicorn.run(create_app(cfg), host=cfg['host'], port=cfg['port'], workers=1)
    finally:
        server.terminate()
        try:
            server.wait(timeout=90)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        signal.signal(signal.SIGTERM, old_sigterm)
    return 0
