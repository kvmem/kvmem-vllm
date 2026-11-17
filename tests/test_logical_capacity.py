from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.logical_capacity import physical_config, wrap_planner, check_specs
from test_deepswe_lru_10m import config as profile


@pytest.fixture(autouse=True)
def isolated_recall_environment(monkeypatch):
    monkeypatch.delenv('KVMEM_RECALL_METHOD', raising=False)
    monkeypatch.delenv('KVMEM_PRESSURE_QUERY_WINDOW', raising=False)
    monkeypatch.delenv('KVMEM_RECALL_CONCURRENT_OWNERS', raising=False)


def config(tmp_path):
    from vllm.config import CacheConfig
    p = profile(tmp_path)
    cache=CacheConfig(cache_dtype='fp8_e4m3',enable_prefix_caching=False,mamba_cache_mode='none')
    cache.kv_cache_layout='LBNHC'
    return NS(model_config=NS(max_model_len=10000000,original_max_model_len=10000000),
        kv_transfer_config=NS(kv_connector='KVMemConnector',kv_connector_module_path='vllm_kvmem.serving.connector',
                             kv_connector_extra_config={'kvmem':p['kvmem']}),
        scheduler_config=NS(scheduler_cls='vllm_kvmem.serving.scheduler.KVMemScheduler',
                            async_scheduling=False,max_num_seqs=8,disable_hybrid_kv_cache_manager=False),
        cache_config=cache,
        speculative_config=NS(method='mtp',num_speculative_tokens=4,
            use_multi_module_mtp=lambda:False,use_eagle=lambda:False,use_eagle_block_drop=lambda:False),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
            decode_context_parallel_size=1,prefill_context_parallel_size=1),
        attention_config=NS(hisparse_config=None))


def recall_config(tmp_path):
    cfg = config(tmp_path)
    cfg.model_config.max_model_len = cfg.model_config.original_max_model_len = 524288
    cfg.scheduler_config.max_num_seqs = 1
    cfg.kv_transfer_config.kv_connector_extra_config['kvmem'].update(
        select_budget=36864, gen_budget=16384, block_tokens=32,
        historical_gc=None, session_capacity=4)
    return cfg


def specs():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    # Actual native block geometry: 17 attention layers and 48 GDN layers.
    page=1616*2*4*256
    attention=FullAttentionSpec(block_size=1616,num_kv_heads=4,head_size=256,dtype=torch.uint8)
    state=MambaSpec(block_size=10000000,shapes=((page,),),dtypes=(torch.uint8,),num_speculative_blocks=4)
    return [{**{f'attn.{i}':attention for i in range(17)},
             **{f'gdn.{i}':state for i in range(48)}}]


def test_native_planner_keeps_pool_layout_and_full_logical_config(tmp_path):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    cfg=config(tmp_path); before=deepcopy(cfg); original_specs=specs()
    native=wrap_planner(get_kv_cache_configs)
    expected=get_kv_cache_configs(physical_config(cfg),specs(),[40*2**30])
    result=native(cfg,original_specs,[40*2**30])
    assert result==expected
    assert cfg.model_config==before.model_config
    assert cfg.kv_transfer_config==before.kv_transfer_config
    assert original_specs==specs()
    assert result[0].num_blocks>=689
    assert sum(g.kv_cache_spec.max_num_blocks_per_req(physical_config(cfg),114688)
               for g in result[0].kv_cache_groups)==86


def test_native_full_history_check_fails_but_window_check_preserves_memory_rejection(tmp_path):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    cfg=config(tmp_path)
    with pytest.raises(ValueError,match='KV cache is needed'):
        get_kv_cache_configs(cfg,specs(),[40*2**30])
    with pytest.raises(ValueError,match='KV cache is needed'):
        wrap_planner(get_kv_cache_configs)(cfg,specs(),[2**30])
    # Native one-owner fit is insufficient to admit the declared eight owners.
    with pytest.raises(ValueError,match='cannot fit 8'):
        wrap_planner(get_kv_cache_configs)(cfg,specs(),[10*2**30])
    assert cfg.model_config.max_model_len==10000000


@pytest.mark.parametrize('change', ['scheduler','connector','gc','rolling','apc','async','tp','override'])
def test_rejects_unqualified_runtime(tmp_path,change):
    cfg=config(tmp_path)
    if change=='scheduler':cfg.scheduler_config.scheduler_cls='native'
    if change=='connector':cfg.kv_transfer_config.kv_connector='foreign'
    if change=='gc':cfg.kv_transfer_config.kv_connector_extra_config['kvmem']['historical_gc']['mode']='shadow'
    if change=='rolling':cfg.kv_transfer_config.kv_connector_extra_config['kvmem']['rolling_active_window']=False
    if change=='apc':cfg.cache_config.enable_prefix_caching=True
    if change=='async':cfg.scheduler_config.async_scheduling=True
    if change=='tp':cfg.parallel_config.tensor_parallel_size=4
    if change=='override':cfg.cache_config.num_gpu_blocks_override=100000
    with pytest.raises(ValueError):physical_config(cfg)


def test_tp2_bounded_planning_uses_the_frame_on_identical_ranks(tmp_path):
    cfg=config(tmp_path)
    single=physical_config(cfg)
    cfg.parallel_config.tensor_parallel_size=2
    bounded=physical_config(cfg)
    assert bounded is not cfg and cfg.model_config.max_model_len==10000000
    assert bounded.model_config.max_model_len==single.model_config.max_model_len<cfg.model_config.max_model_len


def test_prior_lengths_use_native_planner_without_changes():
    cfg=NS(model_config=NS(max_model_len=1048576));calls=[]
    planner=wrap_planner(lambda *args:calls.append(args) or 'native')
    assert planner(cfg,[],[1])=='native' and calls==[(cfg,[],[1])]
    assert wrap_planner(planner) is planner


@pytest.mark.parametrize('method', ['mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass'])
def test_recall_retains_logical_length_and_uses_only_frozen_physical_frame(tmp_path, monkeypatch, method):
    cfg = recall_config(tmp_path)
    before = deepcopy(cfg)
    # A no-GC 524K request remains native unless this experiment is explicit.
    assert physical_config(cfg) is cfg
    monkeypatch.setenv('KVMEM_RECALL_METHOD', method)
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', '128')
    bounded = physical_config(cfg)
    assert bounded is not cfg and bounded.model_config is not cfg.model_config
    assert bounded.model_config.max_model_len == 53248
    assert cfg.model_config == before.model_config
    assert cfg.kv_transfer_config == before.kv_transfer_config
    assert bounded.kv_transfer_config is cfg.kv_transfer_config
    assert bounded.model_config.original_max_model_len == 524288


def test_recall_actual_native_geometry_fits_4090_pool_and_still_checks_capacity(tmp_path, monkeypatch):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    cfg = recall_config(tmp_path)
    original_specs = specs()
    # The real 4090 startup has 2.97 GiB available after model allocations.
    available = [int(2.97 * 2**30)]
    with pytest.raises(ValueError, match='KV cache is needed'):
        get_kv_cache_configs(cfg, specs(), available)
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'exact_mass')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path / 'capture'))
    bounded = physical_config(cfg)
    expected = get_kv_cache_configs(bounded, specs(), available)
    planner = wrap_planner(get_kv_cache_configs)
    result = planner(cfg, original_specs, available)
    assert result == expected
    assert original_specs == specs()
    assert cfg.model_config.max_model_len == 524288
    per_owner = sum(g.kv_cache_spec.max_num_blocks_per_req(bounded, 53248)
                    for g in result[0].kv_cache_groups)
    assert result[0].num_blocks >= per_owner + 1
    import json
    audit = json.loads((tmp_path / 'capture/logical-kv-capacity.json').read_text())
    assert audit['logical_limit'] == 524288 and audit['physical_frame'] == 53248
    assert audit['workers'][0]['concurrent_owners'] == 1
    assert audit['workers'][0]['required_pages'] == per_owner + 1
    with pytest.raises(ValueError, match='KV cache is needed'):
        planner(cfg, specs(), [128 * 2**20])


@pytest.mark.parametrize('owners', [2, 4, 8])
def test_explicit_parallel_recall_retains_per_request_frame_and_all_owner_capacity_check(tmp_path, monkeypatch, owners):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    cfg = recall_config(tmp_path)
    cfg.scheduler_config.max_num_seqs = owners
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'block_value_norm')
    # Merely raising max_num_seqs still fails unless the launch declares it.
    with pytest.raises(ValueError, match='Unqualified recall'):
        physical_config(cfg)
    monkeypatch.setenv('KVMEM_RECALL_CONCURRENT_OWNERS', str(owners))
    bounded = physical_config(cfg)
    assert bounded.model_config.max_model_len == 36864+16384
    assert cfg.model_config.max_model_len == 524288
    per_owner = None
    result = wrap_planner(get_kv_cache_configs)(cfg, specs(), [40*2**30])
    for cache in result:
        per_owner = sum(group.kv_cache_spec.max_num_blocks_per_req(bounded, 53248)
                        for group in cache.kv_cache_groups)
        assert cache.num_blocks >= owners*per_owner+1
    with pytest.raises(ValueError, match='cannot fit '+str(owners)):
        wrap_planner(get_kv_cache_configs)(cfg, specs(), [int(2.97*2**30)])


@pytest.mark.parametrize('change', [
    'logical', 'owners', 'gc', 'block', 'select', 'gen', 'async_refresh',
    'scheduler', 'connector', 'planning', 'rolling', 'cpu_history', 'cpu_reserve',
    'sessions', 'mtp', 'async', 'apc', 'mamba', 'override', 'unbounded', 'tp',
])
def test_recall_exception_keeps_strict_runtime_guards(tmp_path, monkeypatch, change):
    cfg = recall_config(tmp_path)
    opts = cfg.kv_transfer_config.kv_connector_extra_config['kvmem']
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'mean_k')
    if change == 'logical': cfg.model_config.max_model_len = 1048576
    if change == 'owners': cfg.scheduler_config.max_num_seqs = 2
    if change == 'gc':
        from scripts.queue_deepswe_lru_10m import lru
        opts['historical_gc'] = deepcopy(lru.LRU)
    if change == 'block': opts['block_tokens'] = 64
    if change == 'select': opts['select_budget'] = 18432
    if change == 'gen': opts['gen_budget'] = 8192
    if change == 'async_refresh': opts['async_refresh'] = False
    if change == 'scheduler': cfg.scheduler_config.scheduler_cls = 'native'
    if change == 'connector': cfg.kv_transfer_config.kv_connector = 'foreign'
    if change == 'planning': opts['request_planning'] = False
    if change == 'rolling': opts['rolling_active_window'] = False
    if change == 'cpu_history': opts['cpu_history_coordination'] = False
    if change == 'cpu_reserve': opts['cpu_progress_reserve'] = False
    if change == 'sessions': opts['session_capacity'] = 0
    if change == 'mtp': cfg.speculative_config = None
    if change == 'async': cfg.scheduler_config.async_scheduling = True
    if change == 'apc': cfg.cache_config.enable_prefix_caching = True
    if change == 'mamba': cfg.cache_config.mamba_cache_mode = 'all'
    if change == 'override': cfg.cache_config.num_gpu_blocks_override = 100000
    if change == 'unbounded': cfg.model_config.original_max_model_len = -1
    if change == 'tp': cfg.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError):
        physical_config(cfg)


@pytest.mark.parametrize('method,window', [('unknown', '128'), ('mean_k', '32'), ('exact_mass', '129')])
def test_recall_opt_in_validates_method_and_fixed_query_window(tmp_path, monkeypatch, method, window):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', method)
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', window)
    with pytest.raises(ValueError, match='KVMEM_RECALL_METHOD|WINDOW=128'):
        physical_config(recall_config(tmp_path))


def test_long_ordinary_no_gc_configuration_remains_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg.kv_transfer_config.kv_connector_extra_config['kvmem']['historical_gc'] = None
    with pytest.raises(ValueError, match='Unqualified bounded logical KV'):
        physical_config(cfg)


def test_invalid_specs_rejected():
    with pytest.raises(ValueError):check_specs([])
    with pytest.raises(ValueError):check_specs([{'unknown':object()}])


def test_already_imported_core_alias_is_updated_and_idempotent(monkeypatch):
    import sys
    import vllm.v1.core.kv_cache_utils as module
    from vllm_kvmem.serving import logical_capacity as adapter
    original=module.get_kv_cache_configs
    core=NS(get_kv_cache_configs=original)
    monkeypatch.setitem(sys.modules,'vllm.v1.engine.core',core)
    monkeypatch.setattr(module,'get_kv_cache_configs',original)
    adapter.enable()
    assert core.get_kv_cache_configs is module.get_kv_cache_configs
    first=core.get_kv_cache_configs
    assert first.__wrapped__ is original
    adapter.enable()
    assert core.get_kv_cache_configs is first


def dflash_config(tmp_path):
    cfg=config(tmp_path)
    cfg.model_config.max_model_len=cfg.model_config.original_max_model_len=221184
    cfg.scheduler_config.max_num_seqs=4
    cfg.max_concurrent_batches=1
    cfg.num_lookahead_tokens=2
    cfg.kv_transfer_config.kv_connector_extra_config["kvmem"].update(
        select_budget=32768,gen_budget=16384,recent_blocks=64,
        historical_gc=None,async_refresh=False,async_restore=False,maintenance_max_steps=0)
    cfg.speculative_config.method="dflash"
    cfg.speculative_config.num_speculative_tokens=1
    cfg.speculative_config.draft_model_config=NS(dtype=torch.bfloat16,hf_config=NS(
        architectures=["DFlash2DraftModel"],dflash_config={"block_size":8},
        sliding_window=2048,num_hidden_layers=5,layer_types=["sliding_attention"]*5,
        use_sliding_window=True,max_position_embeddings=262144))
    return cfg


def test_dflash_window_planning_is_explicit_and_preserves_full_logical_limit(tmp_path,monkeypatch):
    cfg=dflash_config(tmp_path)
    assert physical_config(cfg) is cfg
    monkeypatch.setenv("KVMEM_LOGICAL_KV_CAPACITY","1")
    bounded=physical_config(cfg)
    assert bounded.model_config.max_model_len==49152
    assert bounded.model_config.original_max_model_len==221184
    assert cfg.model_config.max_model_len==221184
    assert bounded.kv_transfer_config is cfg.kv_transfer_config


def test_dflash_all_owner_admission_still_rejects_insufficient_pages(tmp_path,monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec,SlidingWindowSpec
    cfg=dflash_config(tmp_path);monkeypatch.setenv("KVMEM_LOGICAL_KV_CAPACITY","1")
    full=FullAttentionSpec(block_size=32,num_kv_heads=4,head_size=128,dtype=torch.float8_e4m3fn)
    draft=SlidingWindowSpec(block_size=32,num_kv_heads=4,head_size=128,dtype=torch.bfloat16,sliding_window=2048)
    caches=[{"full":full,"draft":draft}]
    group=NS(kv_cache_spec=NS(max_num_blocks_per_req=lambda config,length:87))
    def native(config,specs,memory):
        assert config.model_config.max_model_len==49152
        assert specs==caches and specs[0] is not caches[0]
        return [NS(kv_cache_groups=[group],num_blocks=132)]
    with pytest.raises(ValueError,match="cannot fit 4.*need 349.*have 132"):
        wrap_planner(native)(cfg,caches,[4*2**30])
    cfg.scheduler_config.max_num_seqs=1
    assert wrap_planner(native)(cfg,caches,[4*2**30])[0].num_blocks==132
    with pytest.raises(ValueError,match="Unsupported cache"):
        check_specs(caches)


@pytest.mark.parametrize("change",["async","apc","rolling","cpu","tp","positions","override"])
def test_dflash_logical_planning_keeps_contract_guards(tmp_path,monkeypatch,change):
    cfg=dflash_config(tmp_path);monkeypatch.setenv("KVMEM_LOGICAL_KV_CAPACITY","1")
    opts=cfg.kv_transfer_config.kv_connector_extra_config["kvmem"]
    if change=="async":opts["async_refresh"]=True
    if change=="apc":cfg.cache_config.enable_prefix_caching=True
    if change=="rolling":opts["rolling_active_window"]=False
    if change=="cpu":opts["cpu_progress_reserve"]=False
    if change=="tp":cfg.parallel_config.tensor_parallel_size=4
    if change=="positions":cfg.model_config.max_model_len=262145
    if change=="override":cfg.cache_config.num_gpu_blocks_override=100000
    with pytest.raises(ValueError):physical_config(cfg)


def test_dflash_sliding_page_budget_matches_runtime_admission(tmp_path):
    from vllm.v1.kv_cache_interface import SlidingWindowSpec,FullAttentionSpec
    from vllm_kvmem.serving.logical_capacity import window_owner_pages
    cfg=dflash_config(tmp_path);cfg.max_in_flight_tokens=520
    sliding=SlidingWindowSpec(block_size=1584,num_kv_heads=8,head_size=128,
        dtype=torch.bfloat16,sliding_window=2048)
    full=FullAttentionSpec(block_size=3168,num_kv_heads=4,head_size=256,
        dtype=torch.float8_e4m3fn)
    assert window_owner_pages(NS(kv_cache_spec=sliding),cfg,49152)==3
    assert window_owner_pages(NS(kv_cache_spec=full),cfg,49152)==16
