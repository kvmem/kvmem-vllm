from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.logical_capacity import physical_config, wrap_planner, check_specs
from test_deepswe_lru_10m import config as profile


def config(tmp_path):
    from vllm.config import CacheConfig
    p = profile(tmp_path)
    cache=CacheConfig(cache_dtype='fp8_e4m3',enable_prefix_caching=False,mamba_cache_mode='none')
    cache.kv_cache_layout='LBNHC'
    return NS(model_config=NS(max_model_len=10000000,original_max_model_len=10000000),
        kv_transfer_config=NS(kv_connector='KVMemConnector',kv_connector_module_path='vllm_kvmem.connector',
                             kv_connector_extra_config={'kvmem':p['kvmem']}),
        scheduler_config=NS(scheduler_cls='vllm_kvmem.scheduler.KVMemScheduler',
                            async_scheduling=False,max_num_seqs=8,disable_hybrid_kv_cache_manager=False),
        cache_config=cache, num_prefill_lookahead_tokens=0,
        speculative_config=NS(method='mtp',num_speculative_tokens=4,
            use_multi_module_mtp=lambda:False,use_eagle=lambda:False,use_eagle_block_drop=lambda:False),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
            decode_context_parallel_size=1,prefill_context_parallel_size=1),
        attention_config=NS(hisparse_config=None))


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
    if change=='tp':cfg.parallel_config.tensor_parallel_size=2
    if change=='override':cfg.cache_config.num_gpu_blocks_override=100000
    with pytest.raises(ValueError):physical_config(cfg)


def test_prior_lengths_use_native_planner_without_changes():
    cfg=NS(model_config=NS(max_model_len=1048576));calls=[]
    planner=wrap_planner(lambda *args:calls.append(args) or 'native')
    assert planner(cfg,[],[1])=='native' and calls==[(cfg,[],[1])]
    assert wrap_planner(planner) is planner


def test_invalid_specs_rejected():
    with pytest.raises(ValueError):check_specs([])
    with pytest.raises(ValueError):check_specs([{'unknown':object()}])


def test_already_imported_core_alias_is_updated_and_idempotent(monkeypatch):
    import sys
    import vllm.v1.core.kv_cache_utils as module
    from vllm_kvmem import logical_capacity as adapter
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
