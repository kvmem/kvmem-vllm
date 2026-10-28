"""Offline bounds, preservation and physical-credit checks; no model requests."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import queue_deepswe_lru_10m as queue
from scripts import deepswe_long_gc_preflight as long
from vllm_kvmem.serve import build_launch
from vllm_kvmem.historical_admission import HistoricalAdmission, HistoricalCapacitySnapshot
from test_serving_launcher import nvfp4_profile


def config(tmp_path):
    cfg=nvfp4_profile(tmp_path)
    cfg.update(draft_method='mtp',draft_tokens=4,max_model_len=10_000_000,max_num_seqs=8,
        prefill_chunk_tokens=2048, max_num_batched_tokens=16424)
    cfg['kvmem'].update(select_budget=98304,gen_budget=16384,block_tokens=32,
        repository_blocks=16384,request_planning=True,session_capacity=32,
        session_checkpoint_policy='pm',session_capture_terminal=True,
        async_refresh=True,async_restore=True,rolling_active_window=True,maintenance_max_steps=8,
        archive_backend='cpu',async_seal=True,async_archive_retirement=True,
        cpu_read_cache_bytes=100000000000,nvme_preallocate=False,nvme_max_bytes=0,
        cpu_history_coordination=True,cpu_progress_reserve=True,
        historical_gc=deepcopy(queue.lru.LRU),graph_mode='off')
    return cfg


def test_decimal_10m_preserves_bounded_kv_and_rotary_profile(tmp_path):
    cfg=config(tmp_path)
    plan=build_launch(cfg,tmp_path,tmp_path/'launch')
    assert plan['profile']['max_model_len']==10_000_000
    assert plan['profile']['kvmem']['select_budget']==98304
    assert plan['profile']['kvmem']['gen_budget']==16384
    assert plan['profile']['kvmem']['repository_blocks']==16384
    assert plan['environment']['VLLM_ALLOW_LONG_MAX_MODEL_LEN']=='1'
    assert plan['environment']['KVMEM_LOGICAL_KV_CAPACITY']=='1'
    assert '--rope-scaling' not in plan['vllm_argv']
    replay=build_launch(plan['profile'],tmp_path,tmp_path/'replay')
    assert replay['profile']==plan['profile']


@pytest.mark.parametrize('key,value',[('historical_gc',None),('cpu_history_coordination',False),
    ('cpu_progress_reserve',False),('rolling_active_window',False),('request_planning',False)])
def test_large_logical_histories_require_physical_budget_and_deleting_gc(tmp_path,key,value):
    cfg=config(tmp_path); cfg['kvmem'][key]=value
    with pytest.raises(ValueError):build_launch(cfg,tmp_path,tmp_path/'bad')


@pytest.mark.parametrize('mode',['off','shadow'])
def test_shadow_gc_does_not_qualify_for_ten_million(tmp_path,mode):
    cfg=config(tmp_path);cfg['kvmem']['historical_gc']['mode']=mode
    with pytest.raises(ValueError):build_launch(cfg,tmp_path,tmp_path/'bad')


def test_above_validated_logical_range_rejected(tmp_path):
    cfg=config(tmp_path);cfg['max_model_len']=10_000_001
    with pytest.raises(ValueError,match='range'):build_launch(cfg,tmp_path,tmp_path/'bad')


@pytest.mark.parametrize('history',[262144,1048576,2_000_000,9_999_936])
def test_logical_length_does_not_charge_deleted_cpu_history(history):
    ledger=HistoricalAdmission(71490,16384,32,lookahead_tokens=5)
    state=HistoricalCapacitySnapshot(1,history,9216,7168,3072)
    result=ledger.preview('owner',state,'request',base_history=history,history_limit=history+64)
    assert result.incoming_records==2
    assert result.retained_records==7168
    assert result.reserved_records==7168+2+ledger.extra_records


def test_incremental_plan_reaches_exact_target_without_skipping_tokens():
    end=0;turns=0
    while end<long.SPEC['target_tokens']:
        count=long.next_input_count(end,long.SPEC)
        assert 0<count<=long.SPEC['append_tokens']
        end+=count+long.SPEC['output_tokens'];turns+=1
        assert end<=long.SPEC['target_tokens']
    assert end==9_999_968 and turns>300
    assert long.next_input_count(end,long.SPEC)==0


def test_append_only_journal_hash_preserves_actual_input_and_output():
    digest=hashlib.sha256();journal=b'';tokens=[]
    for chunk in ([2,300,99999],[7]*64,[21,22,23],[7]*64):
        raw=long.token_bytes(chunk);journal+=raw;digest.update(raw);tokens+=chunk
    assert journal==long.token_bytes(tokens)
    assert len(journal)==4*len(tokens)
    assert digest.hexdigest()==hashlib.sha256(journal).hexdigest()


def test_snapshot_refuses_oversubscription_or_prefix_loss():
    cfg={'kvmem':{'cpu_read_cache_bytes':1000}}
    long.check_snapshot({'historical_gc':{'policy_failures':0}},
                        {'cpu_history_admission':{'reserved_bytes':99,'capacity_bytes':100}},cfg)
    for archive in ({'cpu_authority_bytes':1001},{'session_checkpoint_deferrals':1},
                    {'disk_write_bytes':1},{'historical_gc':{'policy_failures':1}}):
        with pytest.raises(RuntimeError):long.check_snapshot(archive,{},cfg)
    with pytest.raises(RuntimeError):long.check_snapshot({},
        {'cpu_history_admission':{'reserved_bytes':101,'capacity_bytes':100}},cfg)


def workload():
    before={k:{'fixed':k} for k in queue.lru.WORKLOAD_FIELDS}
    before.update(tasks=[{'task_id':str(i)} for i in range(16)],workers=8,attempts_per_task=1,
        duration_seconds=0,carried_results=[],profile={'max_model_len':307200,
        'served_model_name':'old','kvmem':{'select_budget':36864,'gen_budget':16384,
        'historical_gc':deepcopy(queue.lru.LRU)}})
    before['cpu_admission']={'budget_bytes':10**11,'max_history_tokens':307200}
    after=deepcopy(before);after['profile']=queue.profile_from(before['profile'])
    after['cpu_admission']['max_history_tokens']=10_000_000
    after.update(long_gc_qualification=deepcopy(long.SPEC),client_max_output_tokens=16384)
    return before,after


def test_only_authorized_workload_changes_are_allowed():
    before,after=workload(); queue.validate_workload(before,after)
    assert before['profile']['kvmem']['select_budget']==36864
    for key in set(queue.lru.WORKLOAD_FIELDS)-{'setup_sha256','cpu_admission'}:
        invalid=deepcopy(after);invalid[key]='changed'
        with pytest.raises(ValueError):queue.validate_workload(before,invalid)
    for key,value in [('carried_results',['old']),('long_gc_qualification',{}),
                      ('client_max_output_tokens',32000)]:
        invalid=deepcopy(after);invalid[key]=value
        with pytest.raises(ValueError):queue.validate_workload(before,invalid)
    after['profile']['kvmem']['gen_budget']=32768
    with pytest.raises(ValueError):queue.validate_workload(before,after)


def test_pinned_client_change_only_aligns_existing_output_cap():
    source='prefix\nexport MAX_THINKING_TOKENS=4096\nsuffix\n'
    result=queue.client_script(source)
    assert result.replace('export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384\n','')==source
    with pytest.raises(ValueError):queue.client_script(result)
    with pytest.raises(ValueError):queue.client_script('unrecognized')


def test_existing_root_cannot_be_retried_or_overwritten(tmp_path):
    marker=tmp_path/'response';marker.write_text('preserve')
    with pytest.raises(ValueError,match='prior attempts'):
        queue.prepare(tmp_path/'old',tmp_path,tmp_path/'proof')
    assert marker.read_text()=='preserve'
