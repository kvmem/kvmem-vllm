"""Real native FCFS admission and error completion; no model/GPU required."""
import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_kvmem.config import KVMemConfig
from test_native_readiness_admission import native_fixture, enqueue, request


def fixture(monkeypatch,tmp_path,*,async_refresh=False):
    s,window,sessions,_,gate,_=native_fixture(monkeypatch,tmp_path)
    c=KVMemConfig(block_tokens=16,select_budget=32,gen_budget=64,repository_blocks=32,
                  sink_blocks=1,recent_blocks=0)
    sessions.select_budget,sessions.gen_budget,sessions.memory_config=32,64,c
    s.connector.request_plan_memory=c
    s.async_refresh=async_refresh
    s.finished_req_ids_dict=None
    s.vllm_config.kv_transfer_config=None
    s.encoder_cache_manager.free=lambda _:None
    # This test runs native queue traversal, native allocation/free and real
    # typed output serialization. The worker has no GPU and produces no tokens.
    monkeypatch.setattr(Scheduler,'update_from_output',lambda *args,**kwargs:{})
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps(dict(
        block_tokens=16,select_budget=32,gen_budget=64)))
    bad=Request('bad',list(range(207)),SamplingParams(max_tokens=64),None,
        session_id='bad-session',cache_salt='tenant')
    bad.kv_transfer_params={'kvmem_request_plan':{'kvmem_query_span':[160,207]}}
    return s,window,sessions,gate,bad


@pytest.mark.parametrize('async_refresh',[False,True])
@pytest.mark.parametrize('with_peer',[False,True])
def test_impossible_waiter_returns_native_error_without_blocking_peers(monkeypatch,tmp_path,
                                                                    async_refresh,with_peer):
    s,window,sessions,gate,bad=fixture(monkeypatch,tmp_path,async_refresh=async_refresh)
    peer=request('valid',prompt=1)
    enqueue(s,bad,*([peer] if with_peer else []))
    output=s.schedule()
    assert set(output.num_scheduled_tokens)==({'valid'} if with_peer else set())
    assert bad.num_computed_tokens==0 and not window.req_to_blocks.get('bad')
    assert not sessions.entries and not sessions.requests and not gate.permits
    assert 'bad' not in s.kv_cache_manager.reservations
    result=s.update_from_output(output,EMPTY_MODEL_RUNNER_OUTPUT)
    errors=[x for batch in result.values() for x in batch.outputs]
    assert len(errors)==1 and errors[0].request_id=='bad'
    assert errors[0].finish_reason==bad.get_finished_reason()
    assert 'active capacity' in errors[0].stop_reason
    assert bad.status is RequestStatus.FINISHED_ERROR
    assert not s.connector.capacity_rejections
    assert 'bad' not in s.requests and not s.skipped_waiting
    if with_peer:
        assert s.requests['valid'] is peer and peer.status is RequestStatus.RUNNING
        assert window.req_to_blocks.get('valid')
    else:
        assert not s.requests and not s.waiting and not s.running
        assert not s.kv_cache_manager.reservations


def test_all_invalid_step_emits_error_with_zero_model_work(monkeypatch,tmp_path):
    s,window,sessions,gate,bad=fixture(monkeypatch,tmp_path)
    enqueue(s,bad)
    calls=[]
    def execute(output,non_block):
        calls.append(output.total_num_scheduled_tokens)
        assert output.total_num_scheduled_tokens==0 and non_block
        return NS(result=lambda:EMPTY_MODEL_RUNNER_OUTPUT)
    s.get_grammar_bitmask=lambda _:None
    core=NS(scheduler=s,model_executor=NS(execute_model=execute),
        _should_throttle_prefills=lambda:False,
        capture_iteration_details=lambda _:nullcontext(None),
        log_error_detail=lambda _:nullcontext(None),
        _process_aborts_queue=lambda:None,_attach_iteration_details=lambda *_:None,
        _attach_weight_version=lambda *_:None)
    result,model_executed=EngineCore.step(core)
    assert calls==[0] and not model_executed
    assert result[0].outputs[0].request_id=='bad'
    assert bad.status is RequestStatus.FINISHED_ERROR and not s.requests


def test_internal_errors_are_not_reclassified_as_capacity_rejection(monkeypatch,tmp_path):
    s,window,sessions,gate,bad=fixture(monkeypatch,tmp_path)
    def broken(*_):raise RuntimeError('invariant failure')
    monkeypatch.setattr(sessions,'lookup',broken)
    with pytest.raises(RuntimeError,match='invariant failure'):
        s.connector.get_num_new_matched_tokens(bad,0)
    assert not s.connector.capacity_rejections
