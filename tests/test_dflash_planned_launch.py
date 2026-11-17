"""Experimental DFlash planning never silently changes supported deployment modes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_serving_launcher import nvfp4_profile
from vllm_kvmem.serving.serve import build_launch


def test_default_deployment_matches_editable_profile_and_keeps_bounded_dflash(tmp_path):
    from vllm_kvmem.serving.default_profile import load_default_profile
    cfg=load_default_profile()
    assert cfg==json.loads((Path(__file__).parents[1]/'examples/serve-default.json').read_text())
    plan=build_launch(cfg,tmp_path,tmp_path/'run',check_paths=False)
    argv=plan['vllm_argv'];kv=plan['profile']['kvmem']
    assert (cfg['draft_method'],cfg['draft_tokens'],cfg['execution_mode'])==('dflash',7,'full_graph')
    assert argv[argv.index('--gpu-memory-utilization')+1]=='0.8'
    assert argv[argv.index('--max-num-batched-tokens')+1]=='16448'
    assert '--no-enable-prefix-caching' in argv and '--no-async-scheduling' in argv
    assert (kv['select_budget'],kv['gen_budget'],kv['recent_blocks'])==(36864,16384,64)
    assert kv['cpu_read_cache_bytes']==100000000000 and kv['archive_backend']=='cpu'
    assert all(kv[k] for k in ('action_native_decode_commit','action_download_pool',
        'action_fused_scoring','graph_native_propagation'))
    assert kv['action_native_scores']=='off'
    assert cfg['experimental_dflash_query_planning'] and not kv['async_refresh'] and not kv['async_restore']
    assert build_launch(plan['profile'],tmp_path,tmp_path/'replay',check_paths=False)['vllm_argv']==argv
    cfg['kvmem']['gen_budget']=32
    assert load_default_profile()['kvmem']['gen_budget']==16384


def test_default_cli_works_outside_checkout_without_engine_or_side_effects(tmp_path):
    model=nvfp4_profile(tmp_path)
    paths={k:str(tmp_path/model[k]) for k in ('model','tokenizer','hf_config_path','draft_model')}
    code='''import json,os,sys
from vllm_kvmem.serving import default_profile
from vllm_kvmem.serving.serve import main
default_profile._PROFILE.update(json.loads(sys.argv[1]))
before=dict(os.environ)
main(['--run-dir','owned-run','--dry-run'])
assert dict(os.environ)==before
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',code,json.dumps(paths)],cwd=tmp_path,
        env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[1]/'src')},
        capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['profile']['draft_method']=='dflash' and plan['profile']['gpu_memory_utilization']==.8
    assert json.loads(plan['vllm_argv'][plan['vllm_argv'].index('--speculative-config')+1])['kv_cache_dtype']=='bfloat16'
    assert not (tmp_path/'owned-run').exists()


def test_explicit_mtp_profile_bypasses_default_dflash_selection(tmp_path,monkeypatch,capsys):
    from vllm_kvmem.serving import default_profile
    from vllm_kvmem.serving.serve import main
    def refuse_default():
        raise AssertionError('An explicit profile must not read or merge the default deployment')
    monkeypatch.setattr(default_profile,'load_default_profile',refuse_default)
    cfg=nvfp4_profile(tmp_path)
    cfg.update(draft_method='mtp',draft_tokens=4,execution_mode='compiled_graph',gpu_memory_utilization=.61)
    source=tmp_path/'mtp.json';source.write_text(json.dumps(cfg))
    main(['--config',str(source),'--run-dir',str(tmp_path/'owned-run'),'--dry-run'])
    plan=json.loads(capsys.readouterr().out)
    assert (plan['profile']['draft_method'],plan['profile']['gpu_memory_utilization'])==('mtp',.61)
    assert not plan['profile']['experimental_dflash_query_planning']
    assert not (tmp_path/'owned-run').exists()


def configured(tmp_path):
    cfg=nvfp4_profile(tmp_path)
    cfg.update(draft_method='dflash',draft_tokens=7,execution_mode='full_graph',
        experimental_dflash_query_planning=True,prefill_chunk_tokens=1024)
    cfg['kvmem'].update(block_tokens=32,select_budget=4096,gen_budget=4096,
        repository_blocks=2048,recent_blocks=64,request_planning=True,
        session_checkpoint_policy='pm',session_capture_terminal=True,
        action_attention=True,prefill_attention=True,action_fraction=0.,
        action_scope='non-thinking',graph_mode='multi',graph_scoring_version='v2',graph_alpha=.2,
        pressure_policy='graph')
    return cfg


@pytest.mark.parametrize('draft_tokens', [1, 3, 7])
def test_experimental_query_plan_retains_native_draft_cache_and_full_graphs(tmp_path,draft_tokens):
    cfg=configured(tmp_path);cfg['draft_tokens']=draft_tokens
    plan=build_launch(cfg,tmp_path,tmp_path/'run')
    argv=plan['vllm_argv'];spec=json.loads(argv[argv.index('--speculative-config')+1])
    assert spec['method']=='dflash' and spec['num_speculative_tokens']==draft_tokens
    assert spec['kv_cache_dtype']=='bfloat16'
    assert json.loads(argv[argv.index('--compilation-config')+1])=={'mode':0,'cudagraph_mode':'FULL'}
    assert plan['profile']['kvmem']['pressure_policy']=='graph'
    assert 'KVMEM_REQUEST_PLAN_CONFIG' in plan['environment']
    assert build_launch(plan['profile'],tmp_path,tmp_path/'replay')['vllm_argv']==argv


@pytest.mark.parametrize('change,match',[
    ({'experimental_dflash_query_planning':False},'explicit experimental'),
    ({'experimental_dflash_query_planning':1},'must be boolean'),
    ({'execution_mode':'compiled_graph'},'experimental DFlash2 k=1'),
    ({'execution_mode':'eager'},'FULL graphs'),
    ({'draft_tokens':2},'draft_tokens must be'),
    ({'async_refresh':True},'synchronous'),
    ({'recent_blocks':63},'recent band'),
])
def test_unqualified_modes_remain_rejected(tmp_path,change,match):
    cfg=configured(tmp_path)
    for key,value in change.items():
        (cfg['kvmem'] if key in cfg['kvmem'] or key in ('async_refresh','guided_query_tokens') else cfg)[key]=value
    with pytest.raises(ValueError,match=match):build_launch(cfg,tmp_path,tmp_path/'bad')


@pytest.mark.parametrize('tokens', [32, 512])
def test_guided_dflash_reserves_private_scratch_without_increasing_k_or_g(tmp_path,tokens):
    cfg=configured(tmp_path)
    cfg['kvmem']['guided_query_tokens']=tokens
    plan=build_launch(cfg,tmp_path,tmp_path/'guided')
    memory=plan['profile']['kvmem']
    request=json.loads(plan['environment']['KVMEM_REQUEST_PLAN_CONFIG'])
    assert memory['select_budget']==memory['gen_budget']==4096
    assert memory['guided_query_tokens']==tokens
    assert request['guided_reserve']==((256+tokens+31)//32)*32
    assert plan['profile']['draft_method']=='dflash'
    assert not memory.get('async_refresh',False) and not memory.get('async_restore',False)
    assert build_launch(plan['profile'],tmp_path,tmp_path/'again')['vllm_argv']==plan['vllm_argv']


@pytest.mark.parametrize('draft_tokens', [1, 3, 7])
def test_rolling_sync_dflash_keeps_full_output_quota_and_native_graphs(tmp_path,draft_tokens):
    cfg=configured(tmp_path)
    cfg['draft_tokens']=draft_tokens
    cfg['kvmem'].update(rolling_active_window=True,cpu_history_coordination=True,
        async_archive_retirement=True,session_capacity=4,archive_backend='cpu',repository_gc=False,
        cpu_read_cache_bytes=8*1024**3,nvme_preallocate=False,nvme_max_bytes=0)
    plan=build_launch(cfg,tmp_path,tmp_path/'rolling')
    assert json.loads(plan['environment']['KVMEM_REQUEST_PLAN_CONFIG'])['rolling_active_window']
    assert plan['profile']['kvmem']['gen_budget']==4096
    assert plan['profile']['draft_method']=='dflash' and plan['profile']['draft_tokens']==draft_tokens
    assert not plan['profile']['kvmem'].get('async_refresh',False)
    assert not plan['profile']['kvmem'].get('async_restore',False)
    assert build_launch(plan['profile'],tmp_path,tmp_path/'again')['vllm_argv']==plan['vllm_argv']


def test_compiled_dflash_k1_preserves_synchronous_window_and_cache(tmp_path):
    cfg=configured(tmp_path)
    cfg.update(draft_tokens=1,execution_mode='compiled_graph',cudagraph_capture_sizes=[2,4,8])
    plan=build_launch(cfg,tmp_path,tmp_path/'compiled')
    argv=plan['vllm_argv']
    compilation=json.loads(argv[argv.index('--compilation-config')+1])
    assert compilation==dict(mode=3,cudagraph_mode='FULL_AND_PIECEWISE',cudagraph_capture_sizes=[2,4,8])
    assert plan['profile']['kvmem']['select_budget']==4096
    assert plan['profile']['kvmem']['gen_budget']==4096
    assert not plan['profile']['kvmem'].get('async_refresh',False)
    assert not plan['profile']['kvmem'].get('async_restore',False)
    assert plan['environment']['KVMEM_FULL_GRAPH']=='1'
    assert json.loads(argv[argv.index('--speculative-config')+1])['kv_cache_dtype']=='bfloat16'
    assert build_launch(plan['profile'],tmp_path,tmp_path/'again')['vllm_argv']==argv


@pytest.mark.parametrize('change,match',[
    ({'experimental_dflash_query_planning':False},'experimental DFlash2 k=1'),
    ({'draft_tokens':3},'experimental DFlash2 k=1'),
    ({'async_restore':True},'synchronous'),
    ({'async_refresh':True},'synchronous'),
    ({'recent_blocks':63},'recent band'),
])
def test_compiled_dflash_keeps_transaction_guards(tmp_path,change,match):
    cfg=configured(tmp_path)
    cfg.update(draft_tokens=1,execution_mode='compiled_graph')
    for key,value in change.items():
        (cfg['kvmem'] if key in ('async_restore','async_refresh','recent_blocks') else cfg)[key]=value
    with pytest.raises(ValueError,match=match):
        build_launch(cfg,tmp_path,tmp_path/'bad')


def _dflash_async(tmp_path):
    cfg=configured(tmp_path)
    cfg.update(draft_tokens=1,execution_mode='compiled_graph',experimental_dflash_async_refresh=True)
    cfg['kvmem'].update(async_refresh=True,async_restore=True,active_resident_pressure_only=False,
                        active_resident_ttl_ms=0,maintenance_max_steps=8,session_capacity=4,
                        session_checkpoint_bytes=2**30,rolling_active_window=True,cpu_history_coordination=True,
                        async_archive_retirement=True,archive_backend='cpu',repository_gc=False,
                        cpu_read_cache_bytes=8*1024**3,nvme_preallocate=False,nvme_max_bytes=0)
    return cfg


def test_dflash_async_refresh_opt_in_launches_on_the_asynchronous_lifecycle(tmp_path):
    plan=build_launch(_dflash_async(tmp_path),tmp_path,tmp_path/'async')
    assert plan['profile']['kvmem']['async_refresh'] and plan['profile']['kvmem']['async_restore']
    assert plan['environment']['KVMEM_DFLASH_ASYNC_REFRESH']=='1'
    assert plan['environment']['KVMEM_BOUNDED_DFLASH']=='1'
    assert build_launch(plan['profile'],tmp_path,tmp_path/'again')['vllm_argv']==plan['vllm_argv']
    sync=tmp_path/'sync';sync.mkdir()
    assert 'KVMEM_DFLASH_ASYNC_REFRESH' not in build_launch(configured(sync),sync,sync/'plan')['environment']


@pytest.mark.parametrize('change,match',[
    ({'experimental_dflash_async_refresh':1},'must be boolean'),
    ({'experimental_dflash_query_planning':False},'query planning'),
    ({'active_resident_pressure_only':True},'idle active residency'),
    ({'async_restore':False},'refresh/restore'),
])
def test_dflash_async_refresh_opt_in_keeps_its_guards(tmp_path,change,match):
    cfg=_dflash_async(tmp_path)
    for key,value in change.items():
        (cfg['kvmem'] if key in cfg['kvmem'] else cfg)[key]=value
    with pytest.raises(ValueError,match=match):
        build_launch(cfg,tmp_path,tmp_path/'bad')


@pytest.mark.parametrize('draft_tokens', [3, 7])
def test_compiled_dflash_multi_token_needs_its_opt_in(tmp_path, draft_tokens):
    cfg=configured(tmp_path)
    cfg.update(draft_tokens=draft_tokens,execution_mode='compiled_graph',cudagraph_capture_sizes=[2,4,8,16,32,64])
    with pytest.raises(ValueError,match='experimental_dflash_compiled_multi'):
        build_launch(cfg,tmp_path,tmp_path/'refused')
    cfg['experimental_dflash_compiled_multi']=True
    plan=build_launch(cfg,tmp_path,tmp_path/'compiled')
    argv=plan['vllm_argv']
    assert json.loads(argv[argv.index('--speculative-config')+1])['num_speculative_tokens']==draft_tokens
    assert plan['environment']['KVMEM_DFLASH_COMPILED_MULTI']=='1'
    assert plan['environment']['KVMEM_FULL_GRAPH']=='1'
    assert build_launch(plan['profile'],tmp_path,tmp_path/'again')['vllm_argv']==argv


def test_compiled_dflash_multi_token_opt_in_is_validated_and_dflash_only(tmp_path):
    cfg=configured(tmp_path)
    cfg.update(draft_tokens=1,execution_mode='compiled_graph',experimental_dflash_compiled_multi=1)
    with pytest.raises(ValueError,match='must be boolean'):
        build_launch(cfg,tmp_path,tmp_path/'bad')
    k1=tmp_path/'k1';k1.mkdir()
    cfg=configured(k1)
    cfg.update(draft_tokens=1,execution_mode='compiled_graph')
    assert 'KVMEM_DFLASH_COMPILED_MULTI' not in build_launch(cfg,k1,k1/'plan')['environment']
