"""Opt-in controls preserve policy and reject unqualified runtime combinations."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.core.options import RuntimeOptions
from vllm_kvmem.serving.logical_capacity import physical_config, wrap_planner
from test_tiered_options import options


def incremental(**changes):
    values = dict(gpu_memory_admission='incremental', request_checkpoint_bytes=2**30)
    values.update(changes)
    return options(**values)


@pytest.mark.parametrize('changes', [
    {'gpu_memory_admission': 'auto'}, {'request_checkpoint_bytes': 0},
    {'request_checkpoint_bytes': True}, {'request_swap_min_resident_steps': 0},
    {'request_checkpoint_ssd_bytes': -1}, {'request_checkpoint_ssd_bytes': True},
    {'request_swap_min_resident_steps': True}, {'async_refresh': True},
    {'prefill_deferred_validation': True}, {'gpu_memory_admission': 'static'},
])
def test_invalid_or_unqualified_admission_is_rejected(changes):
    with pytest.raises(ValueError):
        incremental(**changes)


def test_admission_changes_neither_memory_policy_nor_refresh_cadence():
    original = options()
    changed = incremental()
    assert changed.memory == original.memory
    for field in ('pressure_policy', 'prefill_cadence', 'async_refresh', 'query_score_mode',
                  'rolling_active_window', 'refresh_quantum_tokens'):
        assert getattr(changed, field) == getattr(original, field)


def test_checkpoint_ssd_is_opt_in_and_preserves_policy():
    original = incremental()
    tiered = incremental(request_checkpoint_ssd_bytes=4*2**30)
    assert tiered.memory == original.memory
    assert tiered.request_checkpoint_bytes == original.request_checkpoint_bytes
    assert original.request_checkpoint_ssd_bytes == 0
    with pytest.raises(ValueError):
        options(request_checkpoint_ssd_bytes=4096)


def native_config():
    from dataclasses import asdict
    opts = incremental()
    payload = {**asdict(opts.memory), **{key: getattr(opts, key) for key in (
        'archive_backend', 'cpu_read_cache_bytes', 'nvme_max_bytes', 'archive_io_max_bytes',
        'cpu_history_coordination', 'session_capacity', 'session_checkpoint_bytes',
        'async_archive_retirement', 'request_planning', 'gpu_memory_admission',
        'request_checkpoint_bytes', 'request_checkpoint_ssd_bytes', 'request_swap_min_resident_steps')}}
    return NS(model_config=NS(max_model_len=1024, original_max_model_len=1024),
        kv_transfer_config=NS(kv_connector='KVMemConnector',
            kv_connector_module_path='vllm_kvmem.serving.connector', kv_connector_extra_config={'kvmem': payload}),
        scheduler_config=NS(scheduler_cls='vllm_kvmem.serving.scheduler.KVMemScheduler',
            async_scheduling=False, max_num_seqs=8),
        cache_config=NS(cache_dtype='fp8_e4m3', enable_prefix_caching=False, mamba_cache_mode='none'),
        speculative_config=None, parallel_config=NS())


def test_planner_requires_one_complete_window_and_preserves_logical_length(monkeypatch):
    config = native_config()
    opts = RuntimeOptions.from_vllm(config)
    assert opts.gpu_memory_admission == 'incremental'
    bounded = physical_config(config)
    assert bounded.model_config.max_model_len == 96
    assert config.model_config.max_model_len == 1024
    monkeypatch.setattr('vllm_kvmem.serving.logical_capacity.check_specs', lambda *_, **__: None)
    group = NS(kv_cache_spec=NS(max_num_blocks_per_req=lambda cfg, length: 6))
    planner = wrap_planner(lambda cfg, specs, available: [NS(num_blocks=available[0], kv_cache_groups=[group])])
    assert planner(config, [{}], [7])[0].num_blocks == 7
    with pytest.raises(ValueError, match='cannot fit 1'):
        planner(config, [{}], [6])


@pytest.mark.parametrize('field', ['async', 'apc', 'tp', 'draft'])
def test_incremental_native_geometry_rejects_unsupported_runtime(field):
    config = native_config()
    if field == 'async': config.scheduler_config.async_scheduling = True
    if field == 'apc': config.cache_config.enable_prefix_caching = True
    if field == 'tp': config.parallel_config.tensor_parallel_size = 2
    if field == 'draft': config.speculative_config = NS(method='dflash')
    with pytest.raises(ValueError):
        physical_config(config)


def test_explicit_native_pool_bytes_survive_launcher_roundtrip(tmp_path):
    from test_serving_launcher import profile
    from vllm_kvmem.serving.serve import build_launch
    cfg = profile(tmp_path)
    cfg['kv_cache_memory_bytes'] = 2**30
    plan = build_launch(cfg, tmp_path, tmp_path/'run')
    argv = plan['vllm_argv']
    assert argv[argv.index('--kv-cache-memory-bytes')+1] == str(2**30)
    assert build_launch(plan['profile'], tmp_path, tmp_path/'again')['vllm_argv'] == argv
    for invalid in (0, True, -1, '1024'):
        cfg['kv_cache_memory_bytes'] = invalid
        with pytest.raises(ValueError, match='kv_cache_memory_bytes'):
            build_launch(cfg, tmp_path, tmp_path/'bad')


@pytest.mark.parametrize('example', ['serve-qwen-mtp-dynamic.json', 'serve-qwen-mtp-dynamic-ssd.json'])
def test_incremental_launcher_preserves_k_g_and_rejects_graph_execution(tmp_path, example):
    import json
    from pathlib import Path
    from vllm_kvmem.serving.serve import build_launch
    cfg = json.loads((Path(__file__).parents[1]/'examples'/example).read_text())
    model = tmp_path/'model'
    tokenizer = tmp_path/'tokenizer'
    model.mkdir(); tokenizer.mkdir()
    (model/'config.json').write_text('{"model_type": "qwen3_5"}')
    cfg.update(model=str(model), hf_config_path=str(model), tokenizer=str(tokenizer))
    plan = build_launch(cfg, tmp_path, tmp_path/'run')
    assert plan['profile']['kvmem']['select_budget'] == 2048
    assert plan['profile']['kvmem']['gen_budget'] == 2048
    assert plan['profile']['kvmem']['gpu_memory_admission'] == 'incremental'
    assert '--enforce-eager' in plan['vllm_argv']
    assert plan['environment']['KVMEM_LOGICAL_KV_CAPACITY'] == '1'
    assert 'KVMEM_VERIFY_REQUEST_SWAP' not in plan['environment']
    cfg['verify_request_swap'] = True
    verified = build_launch(cfg, tmp_path, tmp_path/'verified')
    assert verified['environment']['KVMEM_VERIFY_REQUEST_SWAP'] == '1'
    assert verified['profile']['verify_request_swap'] is True
    cfg['execution_mode'] = 'full_graph'
    with pytest.raises(ValueError, match='Incremental serving'):
        build_launch(cfg, tmp_path, tmp_path/'bad')


def test_swap_verification_requires_explicit_boolean_and_incremental_mode(tmp_path):
    from test_serving_launcher import profile
    from vllm_kvmem.serving.serve import build_launch
    cfg = profile(tmp_path)
    for value in (True, 1, '1'):
        cfg['verify_request_swap'] = value
        with pytest.raises(ValueError, match='verify_request_swap'):
            build_launch(cfg, tmp_path, tmp_path/'bad')


def test_scheduler_preflights_final_mtp_shape_without_clipping_query(monkeypatch):
    from test_scheduler_transactions import setup_scheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.serving.speculative_admission import SpeculativeQueryPolicy
    scheduler, request, _ = setup_scheduler()
    scheduler._bounded_target = True
    scheduler.num_spec_tokens = scheduler.num_lookahead_tokens = 4
    scheduler.query_policy = SpeculativeQueryPolicy(16, 32, 48, 5)
    scheduler.swap_controller = NS(is_suspended=lambda rid: False)
    scheduler.running = [request]
    request.num_prompt_tokens = 49
    request.num_computed_tokens = 48
    request.max_tokens = 1
    request.kv_transfer_params = {}
    calls = []
    fits = False
    scheduler.kv_cache_manager.can_schedule = lambda req, count, **kw: calls.append((count, kw)) or fits
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self,r,h,n:n)
    assert scheduler._reserve_prefill_lookahead(request, 48, 5) == 0
    assert calls[-1][0] == 5
    assert calls[-1][1]['num_lookahead_tokens'] == 4
    fits = True
    assert scheduler._reserve_prefill_lookahead(request, 48, 5) == 5

