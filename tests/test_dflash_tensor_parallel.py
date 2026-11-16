"""TP startup must preserve logical limits and reject the smallest invalid shard."""
from copy import deepcopy
import json
import pytest
import torch
from vllm_kvmem.serving.logical_capacity import physical_config, wrap_planner, check_specs
from test_logical_capacity import dflash_config
from test_dflash_planned_launch import configured
from vllm_kvmem.serving.serve import build_launch

def shard_specs():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    return dict(full=FullAttentionSpec(block_size=3168, num_kv_heads=2,
        head_size=256, dtype=torch.float8_e4m3fn),
        draft=SlidingWindowSpec(block_size=1584, num_kv_heads=4,
        head_size=128, dtype=torch.bfloat16, sliding_window=2048))

def test_tp2_native_planner_keeps_logical_limit_and_each_rank_budget(tmp_path, monkeypatch):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    cfg=dflash_config(tmp_path)
    cfg.parallel_config.tensor_parallel_size=2
    cfg.max_in_flight_tokens=520
    monkeypatch.setenv('KVMEM_LOGICAL_KV_CAPACITY','1')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path/'capture'))
    specs=[shard_specs(),shard_specs()]; before=deepcopy(specs)
    budget=[2*2**30,2**30]
    expected=get_kv_cache_configs(physical_config(cfg),deepcopy(specs),budget)
    actual=wrap_planner(get_kv_cache_configs)(cfg,specs,budget)
    assert actual==expected and specs==before
    assert cfg.model_config.max_model_len==221184
    audit=json.loads((tmp_path/'capture/logical-kv-capacity.json').read_text())
    assert len(audit['workers'])==2 and audit['physical_frame']==49152
    for row in audit['workers']:
        assert row['pool_pages'] >= row['required_pages']==4*row['per_owner_pages']+1
    with pytest.raises(ValueError):
        wrap_planner(get_kv_cache_configs)(cfg,specs,[2*2**30,32*2**20])

def test_missing_or_inconsistent_tp_worker_rejected():
    specs=[shard_specs(),shard_specs()]
    check_specs(specs,allow_dflash=True,expected_workers=2)
    with pytest.raises(ValueError):
        check_specs(specs[:1],allow_dflash=True,expected_workers=2)
    specs[1].pop('draft')
    with pytest.raises(ValueError,match='geometry'):
        check_specs(specs,allow_dflash=True,expected_workers=2)

def test_compiled_tp2_launch_preserves_native_speculation_and_sync(tmp_path):
    cfg=configured(tmp_path)
    cfg.update(tensor_parallel_size=2,draft_tokens=1,execution_mode='compiled_graph')
    plan=build_launch(cfg,tmp_path,tmp_path/'run')
    argv=plan['vllm_argv']
    assert argv[argv.index('--tensor-parallel-size')+1]=='2'
    assert json.loads(argv[argv.index('--speculative-config')+1])['method']=='dflash'
    assert '--no-async-scheduling' in argv
    cfg['tensor_parallel_size']=4
    with pytest.raises(ValueError):build_launch(cfg,tmp_path,tmp_path/'bad')
