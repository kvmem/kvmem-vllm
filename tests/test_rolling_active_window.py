"""Accepted-history rolling refresh preserves native peers and owned leases."""
import json
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.options import RuntimeOptions
from test_phase_options import profile, options


def test_rolling_is_opt_in_and_preserves_server_plan_and_connector(tmp_path):
    from vllm_kvmem.serve import build_launch
    cfg = profile()
    assert not options(cfg['kvmem']).rolling_active_window
    cfg['kvmem']['rolling_active_window'] = True
    plan = build_launch(cfg, tmp_path, tmp_path/'run', check_paths=False)
    assert options(plan['profile']['kvmem']).rolling_active_window
    assert json.loads(plan['environment']['KVMEM_REQUEST_PLAN_CONFIG'])['rolling_active_window'] is True
    second = build_launch(plan['profile'], tmp_path, tmp_path/'again', check_paths=False)
    assert second['environment']['KVMEM_REQUEST_PLAN_CONFIG'] == plan['environment']['KVMEM_REQUEST_PLAN_CONFIG']


@pytest.mark.parametrize('missing', ['request_planning', 'async_refresh', 'async_restore',
                                    'cpu_history_coordination', 'session_capture_terminal'])
def test_rolling_rejects_incomplete_runtime_contract(missing):
    p = profile()['kvmem']
    p.update(rolling_active_window=True)
    p[missing] = False
    with pytest.raises(ValueError, match='Rolling active window requires'):
        options(p)


@pytest.mark.parametrize('value', [1, None, 'true'])
def test_rolling_switch_is_strictly_boolean(value):
    with pytest.raises(ValueError, match='must be boolean'):
        RuntimeOptions(rolling_active_window=value)


def rolling_scheduler(monkeypatch, tmp_path):
    from test_scheduler_transactions import setup_scheduler
    from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy
    scheduler, request, pools = setup_scheduler()
    scheduler._bounded_target = scheduler.async_refresh = scheduler.rolling_active_window = True
    scheduler.prepared_refreshes = {}
    scheduler.query_policy = SpeculativeQueryPolicy(16, 32, 32, 5)
    request.num_computed_tokens = 48
    request.num_prompt_tokens = 33
    request.max_tokens = 32
    request.all_token_ids = list(range(49))
    request.kv_transfer_params = {'kvmem_request_plan': {'kvmem_reselect': 'off'}}
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(block_tokens=16,
        gen_budget=32, select_budget=32, rolling_active_window=True)))
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    return scheduler, request, pools


def test_rolling_pressure_removes_only_owner_without_replay_or_counter_advance(monkeypatch, tmp_path):
    from vllm.v1.core.sched.output import SchedulerOutput
    scheduler, request, _ = rolling_scheduler(monkeypatch, tmp_path)
    peer = NS(request_id='b', num_computed_tokens=20, num_prompt_tokens=10)
    scheduler.requests['b'] = peer
    out = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'a': 5, 'b': 3},
        total_num_scheduled_tokens=8, scheduled_spec_decode_tokens={'a': [111]*4, 'b': [201, 202]})
    metadata = scheduler._build_kv_connector_meta(scheduler.connector, out)
    cmd, = metadata.commands
    assert cmd.selection == 'pressure' and not cmd.input_tokens and cmd.query_span is None
    assert all(ticket.replay_tokens == 0 for _, ticket in cmd.groups)
    assert out.num_scheduled_tokens == {'b': 3} and out.total_num_scheduled_tokens == 3
    assert out.scheduled_spec_decode_tokens == {'b': [201, 202]}
    assert request.num_computed_tokens == 48 and peer.num_computed_tokens == 20
    assert request.max_tokens == 32 and request.next_decode_eligible_step == 2**62
    scheduler._complete_refresh('a', cmd.command_id, selected_tokens=32)
    assert not scheduler.prepared_refreshes and not scheduler.transactions
    assert all(m.offsets['a'] == 16 for m in scheduler.window_managers.values())
    resumed = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'a': 1},
        total_num_scheduled_tokens=1)
    assert not scheduler._build_kv_connector_meta(scheduler.connector, resumed).commands
    assert resumed.num_scheduled_tokens == {'a': 1}


def test_rolling_cancellation_never_releases_pages_before_ack(monkeypatch, tmp_path):
    from vllm.v1.core.sched.output import SchedulerOutput
    scheduler, request, pools = rolling_scheduler(monkeypatch, tmp_path)
    free_before = [p.get_num_free_blocks() for p in pools]
    out = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'a': 5}, total_num_scheduled_tokens=5)
    cmd, = scheduler._build_kv_connector_meta(scheduler.connector, out).commands
    request.is_finished = lambda: True
    assert not scheduler._request_blocks_can_be_freed(request)
    assert [p.get_num_free_blocks() for p in pools] == free_before
    scheduler._complete_refresh('a', cmd.command_id, selected_tokens=32)
    assert not scheduler.table_updates and not scheduler.prepared_refreshes
    assert [p.get_num_free_blocks() for p in pools] == free_before
    for manager in scheduler.window_managers.values():
        manager.free('a')
    assert [p.get_num_free_blocks() for p in pools] == [n+4 for n in free_before]


@pytest.mark.parametrize('selection,enabled', [('query', True), ('chunk', True), ('pressure', False)])
def test_zero_replay_requires_explicit_pressure_contract(selection, enabled):
    from test_async_refresh import manager, command
    obj = manager()
    obj.options.rolling_active_window = enabled
    cmd = replace(command(), selection=selection, input_tokens=(), groups=((0, NS(history_end=64, replay_tokens=0)),))
    with pytest.raises(ValueError, match='Empty refresh'):
        obj.enqueue(cmd)


def test_zero_replay_pressure_consumes_owner_checkpoint_without_private_forward(monkeypatch):
    import vllm_kvmem.async_refresh as ar
    from test_async_refresh import manager, command
    obj = manager()
    obj.options.rolling_active_window = True
    obj.options.memory = NS(select_budget=32,gen_budget=32)
    ticket = NS(history_end=64, replay_tokens=0, original_block_ids=(3, 4), window_tokens=32)
    cmd = replace(command(), selection='pressure', input_tokens=(), groups=((0, ticket),))
    job = obj.enqueue(cmd)
    authority = NS(history_end=64, query_history=None)
    obj.runner._kvmem_capture_runtime = NS(authorities={'a': authority})
    obj.runner.req_states = NS(req_id_to_index={'a': 0})
    pending = {'a': 'exact-owner-checkpoint', 'b': 'peer-checkpoint'}
    captured = []
    def before_attention(batch):
        assert batch.req_ids == ['a'] and not hasattr(batch, 'input_ids')
        captured.append(pending.pop('a'))
    obj.runner._kvmem_sessions = NS(before_attention=before_attention)
    import numpy as np
    import torch
    obj.runner.block_tables = NS(blocks_per_kv_block=[1], num_blocks=NS(np=np.array([[2]])),
        block_tables={0: NS(gpu=torch.tensor([[3, 4]]))})
    monkeypatch.setattr(ar, 'maintenance_batch', lambda *_: pytest.fail('Empty private forward'))
    submitted = []
    obj._submit = lambda j: submitted.append(j)
    assert obj.step(job) and submitted == [job] and job.authority is authority
    assert job.state is None and job.query_rows == []
    assert captured == ['exact-owner-checkpoint'] and pending == {'b': 'peer-checkpoint'}
