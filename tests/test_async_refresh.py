"""Request isolation, replay admission and cancellation fence contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.connector import ControlMetadata, RefreshCommand, RestoreCommand
from test_scheduler_transactions import setup_scheduler


def command(rid='a'):
    ticket=SimpleNamespace(history_end=64,replay_tokens=2)
    return RefreshCommand(rid,1,((0,ticket),),input_tokens=(10,20),asynchronous=True)


def manager():
    obj=AsyncRefresh.__new__(AsyncRefresh)
    obj.jobs={};obj.deferred={};obj.cursor=0
    obj.options=SimpleNamespace(maintenance_max_steps=0)
    obj.runner=SimpleNamespace(_kvmem_pending_capture=None)
    obj.emit=lambda *a,**kw:None
    return obj


@pytest.mark.parametrize('boundary,mode', [(False, 'boundary'), (False, 'both'), (None, 'off')])
def test_agent_public_query_does_not_start_async_private_generation(boundary, mode):
    import torch
    obj = manager()
    obj.options.refresh_quantum_tokens = 256
    obj.options.guided_query_tokens = 512
    obj.options.guided_reselect = mode
    job = RefreshJob(replace(command(), query_span=(64, 66), guided_boundary=boundary))
    job.stage = 'query'
    obj._forward = lambda *args, **kwargs: (None, torch.ones(1, 2, 1, 2))
    submitted = []
    obj._submit = lambda value: submitted.append(value)
    assert obj.step(job)
    assert submitted == [job] and job.offset == 2
    assert len(job.query_rows) == 1
    assert not job.generated and job.guided_policy is None


def test_first_eager_refresh_initializes_window_mapping_and_keeps_existing_neighbours():
    obj = manager()
    obj.options.memory = SimpleNamespace(select_budget=32, gen_budget=32)
    obj.runner._kvmem_window_adapter = None
    obj._ensure_window_adapter(command())
    adapter = obj.runner._kvmem_window_adapter
    adapter.commit('peer', 80, 32)
    obj._ensure_window_adapter(command('new'))
    assert obj.runner._kvmem_window_adapter is adapter
    assert adapter.offsets == {'peer':48}


def test_refresh_removes_only_owner_and_public_replay_does_not_refresh_twice(monkeypatch,tmp_path):
    from vllm_kvmem.admission import QueryPolicy
    scheduler,request,_=setup_scheduler()
    scheduler._bounded_target=True
    scheduler.query_policy=QueryPolicy(16,32,32)
    scheduler.async_refresh=True;scheduler.prepared_refreshes={}
    request.num_computed_tokens=48;request.num_prompt_tokens=80
    request.all_token_ids=list(range(80))
    peer=SimpleNamespace(request_id='b',num_computed_tokens=0,num_prompt_tokens=80)
    scheduler.requests['b']=peer
    out=replace(SchedulerOutput.make_empty(),num_scheduled_tokens={'a':16,'b':8},
                total_num_scheduled_tokens=24,scheduled_spec_decode_tokens={'a':[1]})
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR',str(tmp_path))
    meta=scheduler._build_kv_connector_meta(scheduler.connector,out)
    cmd,=meta.commands
    assert out.num_scheduled_tokens=={'b':8} and out.total_num_scheduled_tokens==8
    assert not out.scheduled_spec_decode_tokens and not out.has_sync_kv_loads
    assert request.num_computed_tokens==48 and request.next_decode_eligible_step==2**62
    assert cmd.input_tokens==tuple(range(48,64)) and cmd.asynchronous
    scheduler._complete_refresh('a',cmd.command_id,selected_tokens=16)
    assert scheduler.prepared_refreshes=={'a':(48,16)}
    resumed=replace(SchedulerOutput.make_empty(),num_scheduled_tokens={'a':16},total_num_scheduled_tokens=16)
    assert not scheduler._build_kv_connector_meta(scheduler.connector,resumed).commands
    assert resumed.num_scheduled_tokens=={'a':16} and not scheduler.prepared_refreshes


def test_cancelled_async_refresh_keeps_old_pages_until_native_free():
    scheduler,request,pools=setup_scheduler()
    scheduler.begin_refresh(request,32,asynchronous=True)
    cmd=scheduler.connector.commands[0]
    request.is_finished=lambda:True
    scheduler._complete_refresh('a',cmd.command_id)
    assert [p.get_num_free_blocks() for p in pools]==[10,10]
    assert not scheduler.table_updates and not scheduler.transactions
    for m in scheduler.window_managers.values():m.free('a')
    assert [p.get_num_free_blocks() for p in pools]==[14,14]


def test_padded_final_query_refresh_never_scores_placeholder_tokens(monkeypatch,tmp_path):
    from vllm_kvmem.admission import QueryPolicy
    scheduler,request,_=setup_scheduler()
    scheduler._bounded_target=True;scheduler.async_refresh=True;scheduler.prepared_refreshes={}
    scheduler.query_policy=QueryPolicy(16,32,32)
    request.num_computed_tokens=48;request.num_prompt_tokens=49;request.max_tokens=1
    request.all_token_ids=list(range(49))
    request.kv_transfer_params={'kvmem_request_plan':{'kvmem_query_span':[48,49]}}
    out=replace(SchedulerOutput.make_empty(),num_scheduled_tokens={'a':5},total_num_scheduled_tokens=5,
                scheduled_spec_decode_tokens={'a':[-1]*4})
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR',str(tmp_path))
    meta=scheduler._build_kv_connector_meta(scheduler.connector,out)
    cmd,=meta.commands
    assert cmd.input_tokens==(48,) and cmd.speculative_padding==4
    assert cmd.groups[0][1].replay_tokens==5
    assert manager().enqueue(cmd).history==48
    assert not out.num_scheduled_tokens and not out.scheduled_spec_decode_tokens


@pytest.mark.parametrize('before_dispatch',[False,True])
def test_cancel_defers_terminal_and_request_index_until_private_work_finishes(before_dispatch):
    obj=manager();cmd=command()
    if not before_dispatch:obj.enqueue(cmd)
    out=replace(SchedulerOutput.make_empty(),finished_req_ids={'a','b'},
        kv_connector_metadata=ControlMetadata(commands=[cmd] if before_dispatch else [],
                                              cancelled={'a'},terminal_prefixes={'a':'prefix','b':'other'}))
    held=obj.defer_terminals(out)
    assert held.finished_req_ids=={'b'}
    assert held.kv_connector_metadata.terminal_prefixes=={'b':'other'}
    if before_dispatch:obj.enqueue(cmd)
    assert obj.jobs['a'].cancelled and obj.jobs['a'].dispatched
    with pytest.raises(ValueError,match='Duplicate'):obj.enqueue(cmd)
    obj.jobs['a'].done=True
    released=obj.defer_terminals(replace(SchedulerOutput.make_empty(),kv_connector_metadata=ControlMetadata()))
    assert released.finished_req_ids=={'a'} and released.kv_connector_metadata.terminal_prefixes=={'a':'prefix'}
    assert not obj.deferred


def test_unready_assembly_does_not_block_other_refresh_or_public_neighbours():
    obj=manager();a=obj.enqueue(command('a'));b=obj.enqueue(command('b'))
    a.stage='assembly'
    a.future=SimpleNamespace(done=lambda:False,result=lambda:pytest.fail('Blocking future wait'))
    real_step=obj.step;advanced=[]
    obj.step=lambda j:real_step(j) if j is a else advanced.append(j.command.request_id) or True
    obj.pump(('public',))
    assert advanced==['b'] and a.peers==b.peers=={'public'}
    with pytest.raises(RuntimeError,match='Suspended'):
        obj.pump(('a',))


@pytest.mark.parametrize('maintenance', [False, True])
def test_unpublished_history_skips_only_its_owner_before_authority_read(maintenance):
    obj=manager();a=obj.enqueue(command('a'));b=obj.enqueue(command('b'))
    obj.runner._kvmem_capture_runtime=SimpleNamespace(publication_ready=lambda ids:'a' not in ids)
    advanced=[]
    obj.step=lambda job:advanced.append(job.command.request_id) or True
    if maintenance:
        obj.maintenance=SimpleNamespace(pump=lambda jobs,step,**kwargs:[step(job) for job in jobs])
    obj.pump(('public',))
    assert advanced==['b']
    obj.runner._kvmem_capture_runtime.publication_ready=lambda ids:True
    obj.pump(('public',))
    assert 'a' in advanced


def test_pending_public_capture_cannot_be_overwritten_by_private_model_work():
    obj=manager();obj.enqueue(command())
    obj.runner._kvmem_pending_capture=object()
    with pytest.raises(RuntimeError,match='unpublished'):obj.pump(('public',))


def test_next_assembly_cannot_reuse_scratch_before_previous_cuda_commit():
    obj=manager();first=obj.enqueue(command('a'));second=obj.enqueue(command('b'))
    first.stage='assembly';first.future=SimpleNamespace(done=lambda:True)
    second.stage='assembly_wait';second.assembly_inputs=('queries','event')
    second.authority=SimpleNamespace()
    calls=[]
    obj.executor=SimpleNamespace(submit=lambda *args:calls.append(args) or 'future')
    assert not obj.step(second) and not calls
    first.done=True;first.future=None
    assert obj.step(second) and calls
    assert second.stage=='assembly' and second.authority._refresh_background
    assert second.assembly_inputs is None


def test_resident_restore_fence_does_not_hold_unrelated_refresh_scratch():
    obj = manager()
    hot = RefreshJob(RestoreCommand('hot', -1, 1, 64, 32, ((1,),), resident=True),
        stage='restore_copy', future=SimpleNamespace(done=lambda: True))
    obj.jobs['hot'] = hot  # DMA still pending, so this lease must remain live.
    refresh = obj.enqueue(command('refresh'))
    refresh.stage = 'assembly_wait'
    refresh.assembly_inputs = ('queries', 'event')
    refresh.authority = SimpleNamespace()
    calls = []
    obj.executor = SimpleNamespace(submit=lambda *args: calls.append(args) or 'future')
    assert obj.step(refresh) and len(calls) == 1
    assert refresh.stage == 'assembly' and not hot.done and hot.future is not None


def test_assembly_fence_covers_query_concatenation(monkeypatch):
    import torch
    obj=manager();obj.stream=object();obj._restore_state=lambda job: events.append('restore')
    events=[]
    job=SimpleNamespace(guided_rows=[],query_rows=[object()])
    query=object()
    def cat(rows,dim):
        events.append('cat');return query
    monkeypatch.setattr(torch,'cat',cat)
    monkeypatch.setattr(torch.cuda,'Event',lambda:SimpleNamespace(record=lambda:events.append('fence')))
    monkeypatch.delenv('KVMEM_RETRIEVAL_DIAGNOSTICS',raising=False)
    obj._submit(job)
    assert events==['restore','cat','fence']
    assert job.assembly_inputs[0] is query
