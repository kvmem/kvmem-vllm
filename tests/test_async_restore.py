"""CPU contracts for request-local checkpoint imports and DMA ownership.

The tests exercise real checkpoint/window/session state with controlled future
and completion fences; they never initialize a CUDA context or run a model.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('vllm')
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.async_restore import RestoreContext, step_restore, validate_pages
from vllm_kvmem.connector import ControlCompletions, ControlMetadata, RestoreCommand
from vllm_kvmem.prompt_checkpoints import CheckpointPhase
from vllm_kvmem.session_control import SessionLedger, SessionNotice
from test_core import Event
from test_prompt_checkpoints import ready, saved
from test_scheduler_transactions import setup_scheduler
from test_session_control import request as session_request


def restored_scheduler(monkeypatch, tmp_path):
    from vllm_kvmem.admission import QueryPolicy
    scheduler, _, pools = setup_scheduler()
    for manager in scheduler.window_managers.values():
        manager.free('a')
    events = []

    def prepare(rid, history, window):
        for manager in scheduler.window_managers.values():
            manager.prepare_import(rid, history, window)

    def commit(rid, history):
        events.append(('import', rid, history))
        for manager in scheduler.window_managers.values():
            assert manager.commit_import(rid, history, Event(True))

    sessions = SessionLedger(2, 4, prepare, lambda _: None, commit)
    scheduler.connector.sessions = sessions
    old = session_request('old', tokens=list(range(65)))
    old.num_computed_tokens, old.num_prompt_tokens = 48, 65
    sessions.lookup(old, 0)
    sessions.allocated(old, 0)
    sessions.capture_before_final_prefill(old, 17)
    binding = sessions.take_commands().bindings[0]
    sessions.finished(old)
    sessions.consume(SessionNotice('retired', binding.namespace, binding.generation, 'old', 48, 32))
    new = session_request('new', tokens=list(range(65)))
    new.num_computed_tokens, new.num_prompt_tokens = 48, 65
    new.num_in_flight_tokens, new.next_decode_eligible_step = 0, 7
    new.is_finished = lambda: False
    peer = SimpleNamespace(request_id='b', num_computed_tokens=32, num_prompt_tokens=32)
    scheduler.requests = {'new': new, 'b': peer}
    assert sessions.lookup(new, 0) == 48
    for manager in scheduler.window_managers.values():
        manager.add_local_computed_blocks('new', (), 0, 48)
        manager.allocate_external_computed_blocks('new', 0, 48)
        manager.allocate_new_blocks('new', 64, 64)
    sessions.allocated(new, 48)
    scheduler._bounded_target = scheduler.async_restore = True
    scheduler._restore_sequence = 0
    scheduler.query_policy = QueryPolicy(16, 32, 32)
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    return scheduler, new, pools, binding, events


@pytest.mark.parametrize('cancelled', [False, True])
def test_native_restore_suspends_only_import_owner_and_fences_all_pages(monkeypatch, tmp_path, cancelled):
    scheduler, request, pools, binding, events = restored_scheduler(monkeypatch, tmp_path)
    new_requests = [SimpleNamespace(req_id='new')]
    output = replace(SchedulerOutput.make_empty(), scheduled_new_reqs=new_requests,
        num_scheduled_tokens={'new': 16, 'b': 1}, total_num_scheduled_tokens=17,
        scheduled_spec_decode_tokens={'new': [99], 'b': [100]})
    before_free = [pool.get_num_free_blocks() for pool in pools]
    metadata = scheduler._build_kv_connector_meta(scheduler.connector, output)
    command, = metadata.commands
    assert isinstance(command, RestoreCommand) and command.command_id < 0
    assert (command.history_end, command.window_tokens) == (48, 32)
    assert output.num_scheduled_tokens == {'b': 1} and output.total_num_scheduled_tokens == 1
    assert output.scheduled_spec_decode_tokens == {'b': [100]}
    assert output.scheduled_new_reqs is new_requests
    assert [row.request_id for row in metadata.sessions.bindings] == ['new']
    assert request.num_computed_tokens == 48 and request.next_decode_eligible_step == 2**62
    assert not scheduler._request_blocks_can_be_freed(request)
    with pytest.raises(RuntimeError, match='before import commit'):
        scheduler._complete_refresh('new', command.command_id)
    assert request.next_decode_eligible_step == 2**62 and scheduler.transactions
    if cancelled:
        request.is_finished = lambda: True
    scheduler.connector.update_connector_output(KVConnectorOutput(
        kv_connector_worker_meta=ControlCompletions(completed=[('new', command.command_id)],
            sessions=[SessionNotice('restored', binding.namespace, binding.generation,
                                    'new', 48, 32, retained=True)])))
    assert events == [('import', 'new', 48)]
    assert request.next_decode_eligible_step == 7 and not scheduler.transactions
    assert not scheduler.table_updates
    assert [pool.get_num_free_blocks() for pool in pools] == before_free
    assert all(not manager.imports and not manager.pending
               for manager in scheduler.window_managers.values())
    for manager in scheduler.window_managers.values():
        manager.free('new')
        manager.free('b')
    assert [pool.get_num_free_blocks() for pool in pools] == [16, 16]


def test_restore_cannot_ack_reallocated_destination_pages(monkeypatch, tmp_path):
    scheduler, request, pools, binding, _ = restored_scheduler(monkeypatch, tmp_path)
    output = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'new': 16},
                     total_num_scheduled_tokens=16)
    command, = scheduler._build_kv_connector_meta(scheduler.connector, output).commands
    scheduler.connector.sessions.consume(SessionNotice('restored', binding.namespace,
        binding.generation, 'new', 48, 32, retained=True))
    scheduler.window_managers[1].native.allocate_new_blocks('new', 96, 96)
    before = [pool.get_num_free_blocks() for pool in pools]
    with pytest.raises(RuntimeError, match='page ownership changed'):
        scheduler._complete_refresh('new', command.command_id)
    assert [pool.get_num_free_blocks() for pool in pools] == before
    assert 'new' in scheduler.transactions and request.next_decode_eligible_step == 2**62


def test_suspended_native_tail_retains_placeholder_shape_and_waits_for_budget(monkeypatch, tmp_path):
    from vllm.v1.core.sched.scheduler import Scheduler
    scheduler, request, _, _, _ = restored_scheduler(monkeypatch, tmp_path)
    request.num_prompt_tokens = 49
    scheduler.num_spec_tokens = 4
    output = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'new':5},
        total_num_scheduled_tokens=5, scheduled_spec_decode_tokens={'new':[-1]*4})
    scheduler._build_kv_connector_meta(scheduler.connector, output)
    assert output.num_scheduled_tokens == {} and output.scheduled_spec_decode_tokens == {}
    assert request.spec_token_ids == [-1]*4
    assert scheduler._deferred_native_padding == {'new':(48,5)}
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead',lambda self,req,history,n:n)
    assert scheduler._reserve_prefill_lookahead(request,48,4) == 0
    assert request.spec_token_ids == [-1]*4 and request.num_computed_tokens == 48
    assert scheduler._reserve_prefill_lookahead(request,48,5) == 5
    with pytest.raises(RuntimeError,match='exact unexecuted tail'):
        scheduler._reserve_prefill_lookahead(request,49,5)
    request.spec_token_ids=[]
    with pytest.raises(RuntimeError,match='exact unexecuted tail'):
        scheduler._reserve_prefill_lookahead(request,48,5)


def test_deferred_padding_never_invents_or_preserves_real_drafts(monkeypatch,tmp_path):
    scheduler,request,_,_,_=restored_scheduler(monkeypatch,tmp_path)
    request.spec_token_ids=[]
    output=replace(SchedulerOutput.make_empty(),scheduled_spec_decode_tokens={'new':[9,10]})
    scheduler._preserve_deferred_padding(request,output,3)
    assert request.spec_token_ids==[] and not getattr(scheduler,'_deferred_native_padding',{})
    output.scheduled_spec_decode_tokens['new']=[-1]*4
    with pytest.raises(RuntimeError,match='single public tail'):
        scheduler._preserve_deferred_padding(request,output,5)


def test_deferred_padding_is_consumed_by_public_execution_or_cancel(monkeypatch,tmp_path):
    from vllm.v1.core.sched.scheduler import Scheduler
    scheduler,request,_,_,_=restored_scheduler(monkeypatch,tmp_path)
    scheduler._bounded_target=False
    scheduler._deferred_native_padding={'new':(48,5),'b':(32,5)}
    monkeypatch.setattr(Scheduler,'_build_kv_connector_meta',lambda self,c,o:o)
    output=replace(SchedulerOutput.make_empty(),num_scheduled_tokens={'new':5},
        scheduled_spec_decode_tokens={'new':[-1]*4},total_num_scheduled_tokens=5)
    assert scheduler._build_kv_connector_meta(scheduler.connector,output) is output
    assert scheduler._deferred_native_padding=={'b':(32,5)}
    scheduler._deferred_native_padding['new']=(48,5)
    monkeypatch.setattr(Scheduler,'_free_request',lambda self,request,delay_free_blocks=False:True)
    assert scheduler._free_request(request)
    assert scheduler._deferred_native_padding=={'b':(32,5)}


def test_restore_validates_cpu_allocation_witness_without_reading_gpu_tables():
    from vllm_kvmem.restore_lease import FreshRestoreLease
    runner = SimpleNamespace(req_states=SimpleNamespace(req_id_to_index={'new': 0}),
        _kvmem_request_generations={'new': 3}, _kvmem_restore_allocations={},
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None, None]),
        block_tables=SimpleNamespace(blocks_per_kv_block=[2, 1],
            num_blocks=SimpleNamespace(np=torch.tensor([[4], [1]]).numpy()),
            # Shape is the only permitted GPU-table metadata. An accidental
            # index/readback would fail because these stand-ins hold no data.
            block_tables=[SimpleNamespace(gpu=SimpleNamespace(shape=(1, 4))),
                          SimpleNamespace(gpu=SimpleNamespace(shape=(1, 4)))]))
    manager = SimpleNamespace(runner=runner)
    command = RestoreCommand('new', -1, 3, 64, 32, ((3, 4), (2,)))
    with pytest.raises(ValueError, match='no fresh native allocation witness'):
        validate_pages(manager, command)
    lease = FreshRestoreLease.capture(runner, 'new', command.block_ids)
    runner._kvmem_restore_allocations['new'] = lease
    assert validate_pages(manager, command) is lease
    assert lease.rows == ((6, 7, 8, 9), (2,))
    with pytest.raises(ValueError, match='native allocation witness'):
        validate_pages(manager, replace(command, block_ids=((3, 4),)))
    with pytest.raises(ValueError, match='native allocation witness'):
        validate_pages(manager, replace(command, block_ids=((3, 4), (7,))))
    runner.block_tables.num_blocks.np[1, 0] = 2
    with pytest.raises(ValueError, match='page count changed'):
        validate_pages(manager, command)


def restore_inflight(tmp_path):
    pool, checkpoint, authority, archive, adapter, _ = saved(tmp_path)
    pool.retire(checkpoint, Event(True))
    ready(pool, checkpoint)
    assert pool.claim(checkpoint.namespace, 'new', list(range(20))) is checkpoint
    plan = pool.prepare_claim_window(checkpoint, wait=False)
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    model = adapter.prepare_restore(checkpoint.model, 'new')
    adapter.submit_restore(model)  # CPU copy stands in for submitted model DMA.
    fence = Event()
    assembly = SimpleNamespace(uploaded_bytes=128, host_prepare_seconds=.002, query=fence.query)
    trace = []
    future = SimpleNamespace(ready=False, reads=0)
    future.done = lambda: future.ready

    def result():
        assert future.ready, 'A producer future must never be waited on'
        future.reads += 1
        return assembly, None, fence

    future.result = result
    assembler = SimpleNamespace(close=lambda: trace.append('close') or True)
    worker = SimpleNamespace(pool=pool, model_adapter=adapter, imports={'new'},
        _verify_restored=lambda *args: trace.append('verify'),
        publish_restore=lambda rid, value, **kwargs: trace.append(('publish', rid, value, kwargs)))
    manager = AsyncRefresh.__new__(AsyncRefresh)
    manager.runner = adapter.runner
    manager.runner._kvmem_sessions = worker
    manager.runner._kvmem_probe_output = tmp_path
    manager.runner._kvmem_restore_allocations = {'new': object()}
    manager.options = SimpleNamespace(async_restore=True)
    manager.deferred, manager.cursor = {}, 0
    manager.emit = lambda _job, stage, **fields: trace.append(stage)
    manager.maintenance = SimpleNamespace(
        observe=lambda job, size: trace.append(('observe', size)),
        release=lambda job: trace.append('release'))
    command = RestoreCommand('new', -1, 3, 8, 8, ((1, 2),))
    job = RefreshJob(command, stage='restore_copy', dispatched=True, future=future,
        authority=authority, checkpoint=checkpoint, model_bytes=checkpoint.model.bytes,
        restore_context=RestoreContext(checkpoint, plan, assembler, model, None))
    manager.jobs = {'new': job}
    authority._refresh_background = True
    return manager, job, future, fence, trace, checkpoint, authority, archive


@pytest.mark.parametrize('cancelled', [False, True])
def test_restore_completion_keeps_native_lease_and_history_until_full_copy_fence(tmp_path, cancelled):
    manager, job, future, fence, trace, checkpoint, authority, archive = restore_inflight(tmp_path)
    if cancelled:
        output = replace(SchedulerOutput.make_empty(), finished_req_ids={'new', 'peer'},
            kv_connector_metadata=ControlMetadata(cancelled={'new'},
                terminal_prefixes={'new': 'owned-prefix', 'peer': 'peer-prefix'}))
        deferred = manager.defer_terminals(output)
        assert deferred.finished_req_ids == {'peer'}
        assert deferred.kv_connector_metadata.terminal_prefixes == {'peer': 'peer-prefix'}
        assert job.cancelled
    assert not manager.step(job) and future.reads == 0
    future.ready = True
    assert not manager.step(job) and future.reads == 1
    assert not manager.step(job) and future.reads == 1
    assert not trace and not job.done and checkpoint.phase is CheckpointPhase.CLAIMED
    assert 'new' in manager.runner._kvmem_sessions.imports
    assert authority._refresh_background and job.restore_context is not None
    fence.done = True
    assert manager.step(job)
    assert job.done and job.future is job.checkpoint is job.restore_context is None
    assert checkpoint.phase is CheckpointPhase.LIVE and checkpoint.request_id == 'new'
    assert checkpoint.authority is authority and checkpoint.model is not None
    assert not manager.runner._kvmem_sessions.imports and not authority._refresh_background
    assert not manager.runner._kvmem_restore_allocations
    assert trace == ['verify', ('publish', 'new', authority, {'retained': True}), 'close',
                     ('observe', 128 + job.model_bytes), 'release', 'restore_ready']
    metrics = json.loads((tmp_path / 'async_restore.jsonl').read_text())
    assert metrics['cancelled'] is cancelled and metrics['active_tokens'] == 8
    if cancelled:
        released = manager.defer_terminals(replace(SchedulerOutput.make_empty(),
                                                   kv_connector_metadata=ControlMetadata()))
        assert released.finished_req_ids == {'new'}
        assert released.kv_connector_metadata.terminal_prefixes == {'new': 'owned-prefix'}
        assert not manager.deferred
    # Completion retains the original CPU checkpoint even when the cancelled
    # request never executes a single public token after its import.
    pool = manager.runner._kvmem_sessions.pool
    pool.retire(checkpoint, Event(True))
    ready(pool, checkpoint)
    assert pool.claim(checkpoint.namespace, 'again', list(range(20))) is checkpoint
    assert pool.evict(checkpoint, Event(True))
    assert archive.close()


def test_restore_waiting_for_native_initialization_does_no_work():
    command = RestoreCommand('new', -1, 1, 64, 32, ())
    manager = SimpleNamespace(runner=SimpleNamespace(_kvmem_sessions=object(),
        req_states=SimpleNamespace(req_id_to_index={})))
    job = RefreshJob(command)
    assert not step_restore(manager, job)
    assert job.stage == 'prepare' and job.checkpoint is None


def test_cancel_arriving_with_first_restore_dispatch_keeps_the_native_index():
    command = RestoreCommand('new', -1, 1, 64, 32, ())
    manager = AsyncRefresh.__new__(AsyncRefresh)
    manager.jobs, manager.deferred = {}, {}
    manager.options = SimpleNamespace(async_restore=True)
    manager.emit = lambda *args, **kwargs: None
    output = replace(SchedulerOutput.make_empty(), finished_req_ids={'new', 'peer'},
        kv_connector_metadata=ControlMetadata(commands=[command], cancelled={'new'},
                                              terminal_prefixes={'new': 'prefix'}))
    held = manager.defer_terminals(output)
    assert held.finished_req_ids == {'peer'} and not held.kv_connector_metadata.terminal_prefixes
    job = manager.enqueue(command)
    assert job.cancelled and job.dispatched and not job.done and job.history == 64
    assert job.stage == 'prepare' and manager.deferred == {'new': 'prefix'}
    with pytest.raises(ValueError, match='Duplicate'):
        manager.enqueue(command)


def test_recycled_state_row_fails_restore_before_committing_checkpoint_or_releasing_scratch(tmp_path):
    manager, job, future, fence, trace, checkpoint, authority, archive = restore_inflight(tmp_path)
    future.ready = fence.done = True
    manager.runner._kvmem_request_generations['new'] = 99
    with pytest.raises(ValueError, match='recycled'):
        manager.step(job)
    assert not trace and not job.done
    assert checkpoint.phase is CheckpointPhase.CLAIMED
    assert job.restore_context is not None and 'new' in manager.runner._kvmem_sessions.imports
    assert authority._refresh_background
    # Restore the fixture's original lease only to clean up this deliberately
    # invalid native state; production treats this invariant violation as fatal.
    manager.runner._kvmem_request_generations['new'] = 3
    assert manager.step(job)
    pool = manager.runner._kvmem_sessions.pool
    pool.drop_live(checkpoint)
    assert authority.close() and archive.close()


def test_second_restore_cannot_claim_shared_scratch_before_first_gpu_fence():
    command = RestoreCommand('new', -1, 3, 64, 32, ())
    binding = SimpleNamespace(generation=3, history=64, window=32)
    first = RefreshJob(RestoreCommand('first', -2, 4, 64, 32, ()),
                       stage='restore_copy', future=SimpleNamespace(done=lambda: True))
    job = RefreshJob(command)
    manager = SimpleNamespace(runner=SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={'new': 0}),
        _kvmem_sessions=SimpleNamespace(bindings={'new': binding}, imports={'new'})),
        jobs={'first': first, 'new': job})
    assert not step_restore(manager, job)
    assert job.checkpoint is None and job.restore_context is None


@pytest.mark.parametrize('mutation', ['scheduled_tokens', 'block_table', 'cached_state', 'finished'])
def test_runner_holds_restore_state_and_allocation_until_completion_but_allows_peers(mutation):
    from vllm_kvmem.runner import make_probe_runner
    calls = []

    class NativeBase:
        def update_requests(self, output):
            calls.append(('update', set(output.num_scheduled_tokens)))
            return 'updated'

        def finish_requests(self, output):
            calls.append(('finish', set(output.finished_req_ids)))
            return 'finished'

    runner = make_probe_runner(NativeBase)()
    runner._kvmem_restore_allocations = {'restore': object()}
    runner._kvmem_request_generations = {'restore': 1, 'peer': 2}
    peer = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'peer': 1},
        total_num_scheduled_tokens=1, scheduled_cached_reqs=SimpleNamespace(req_ids=['peer']))
    assert runner.update_requests(peer) == 'updated'
    assert calls == [('update', {'peer'})]
    calls.clear()
    output = SchedulerOutput.make_empty()
    if mutation == 'scheduled_tokens':
        output = replace(output, num_scheduled_tokens={'restore': 1}, total_num_scheduled_tokens=1)
    elif mutation == 'block_table':
        output.block_table_updates = {'restore': ([1, 2],)}
    elif mutation == 'cached_state':
        output = replace(output, scheduled_cached_reqs=SimpleNamespace(req_ids=['restore']))
    else:
        output = replace(output, finished_req_ids={'restore'})
    method = runner.finish_requests if mutation == 'finished' else runner.update_requests
    with pytest.raises(RuntimeError, match='before its restore fence|public page/state update'):
        method(output)
    assert not calls and runner._kvmem_request_generations['restore'] == 1
    runner._kvmem_restore_allocations.pop('restore')  # Completion has committed and closed its scratch.
    assert method(output) == ('finished' if mutation == 'finished' else 'updated')
    assert len(calls) == 1 and runner._kvmem_request_generations['peer'] == 2
    if mutation == 'finished':
        assert 'restore' not in runner._kvmem_request_generations


@pytest.mark.parametrize('enabled', [False, True])
def test_runner_registers_fresh_allocation_only_for_enabled_checkpoint_imports(enabled):
    from vllm_kvmem.restore_lease import FreshRestoreLease
    from vllm_kvmem.runner import make_probe_runner

    class NativeBase:
        def add_requests(self, output):
            # Native allocation precedes the plugin witness and generation.
            for index, request in enumerate(output.scheduled_new_reqs):
                self.req_states.req_id_to_index[request.req_id] = index
            return 'allocated'

    runner = make_probe_runner(NativeBase)()
    runner._kvmem_next_generation = 0
    runner._kvmem_request_generations, runner._kvmem_restore_allocations = {}, {}
    runner.req_states = SimpleNamespace(req_id_to_index={})
    runner._kvmem_sessions = SimpleNamespace(options=SimpleNamespace(async_restore=enabled),
                                            imports={'restore'})
    runner.block_tables = SimpleNamespace(blocks_per_kv_block=[2],
        num_blocks=SimpleNamespace(np=torch.tensor([[4, 4]]).numpy()),
        block_tables=[SimpleNamespace(gpu=SimpleNamespace(shape=(2, 4)))])
    output = replace(SchedulerOutput.make_empty(), scheduled_new_reqs=[
        SimpleNamespace(req_id='restore', block_ids=((2, 3),)),
        SimpleNamespace(req_id='peer', block_ids=((4, 5),))])
    assert runner.add_requests(output) == 'allocated'
    assert runner._kvmem_request_generations == {'restore': 1, 'peer': 2}
    assert set(runner._kvmem_restore_allocations) == ({'restore'} if enabled else set())
    if enabled:
        lease = runner._kvmem_restore_allocations['restore']
        assert isinstance(lease, FreshRestoreLease)
        assert lease.index == 0 and lease.generation == 1 and lease.rows == ((4, 5, 6, 7),)
        lease.validate(runner, expected_block_ids=((2, 3),))


class LaneEvent(Event):
    def record(self):
        pass


def two_lane_manager(monkeypatch, tmp_path, *, resident_slots=1):
    """Real producer threads, fake device fences, and separate leased rows."""
    from vllm_kvmem.maintenance import MaintenancePolicy
    import vllm_kvmem.async_restore as restore_module
    from vllm_kvmem.window_assembly import NativeWindowAssembler
    options = SimpleNamespace(async_restore=True, maintenance_max_steps=8,
        maintenance_busy_ms=2., maintenance_idle_ms=8., maintenance_aging_ms=250.,
        maintenance_inflight_bytes=0, session_checkpoint_bytes=256,
        memory=SimpleNamespace(select_budget=32, gen_budget=16, block_tokens=4),
        packed_assembly=True, direct_pinned_reads=True, resident_restore_slots=resident_slots)
    runner = SimpleNamespace(model=SimpleNamespace(forward=lambda: None), device='cpu',
        req_states=SimpleNamespace(req_id_to_index={'cold': 0, 'hot': 1, 'next': 2}),
        _kvmem_probe_output=tmp_path, _kvmem_restore_allocations={},
        _kvmem_capture_runtime=SimpleNamespace(names=()))
    manager = AsyncRefresh(runner, options)
    manager.maintenance = MaintenancePolicy(options, 3)
    checkpoints, bindings, claims, publications = {}, {}, [], []
    for index, rid in enumerate(('cold', 'hot', 'next')):
        memory = SimpleNamespace(attach_completion=lambda _: None, commit_selection=lambda: True)
        authority = SimpleNamespace(memory=memory)
        checkpoint = SimpleNamespace(kind='M', position=48, window=SimpleNamespace(window_tokens=48),
            authority=authority, model=SimpleNamespace(bytes=200), phase='idle')
        checkpoints[rid] = checkpoint
        bindings[rid] = SimpleNamespace(namespace=rid, generation=index+1, history=48,
                                         window=48, tokens=tuple(range(49)))
        runner._kvmem_restore_allocations[rid] = SimpleNamespace(
            validate=lambda _runner, **kwargs: None)

    def claim(namespace, rid, tokens, *, position):
        checkpoint = checkpoints[namespace]
        assert checkpoint.phase == 'idle' and position == checkpoint.position
        checkpoint.phase = 'claimed'
        claims.append(rid)
        return checkpoint

    def finish(checkpoint, completion, *, retain):
        assert completion.query() and retain and checkpoint.phase == 'claimed'
        checkpoint.phase = 'live'
        return checkpoint.authority

    def prepare(snapshot, rid, *, lease):
        return SimpleNamespace(request_id=rid, lease=lease)

    def validate(prepared):
        assert runner._kvmem_restore_allocations[prepared.request_id] is prepared.lease

    worker = SimpleNamespace(bindings=bindings, imports=set(bindings),
        pool=SimpleNamespace(claim=claim, prepare_claim_window=lambda *args, **kwargs: object(),
                             finish_claim=finish),
        model_adapter=SimpleNamespace(prepare_restore=prepare, validate_restore=validate),
        _verify_restored=lambda *args: None,
        publish_restore=lambda rid, authority, **kwargs: publications.append(rid))
    runner._kvmem_sessions = worker
    monkeypatch.setattr(torch.cuda, 'Event', LaneEvent)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner',
        lambda *args, **kwargs: SimpleNamespace(close=lambda: True))
    cold_release, hot_started = threading.Event(), threading.Event()
    fences = {rid: Event() for rid in bindings}

    def copy_restore(_manager, job):
        rid = job.command.request_id
        if rid == 'cold':
            assert job.restore_context.assembler is not None
            assert job.restore_context.stream is manager.stream
            assert cold_release.wait(5), 'Test did not release its delayed cold producer'
        else:
            assert job.restore_context.assembler is None
            assert job.restore_context.stream is manager.resident_stream
            hot_started.set()
        return SimpleNamespace(uploaded_bytes=0, host_prepare_seconds=0.,
                               query=fences[rid].query), None, fences[rid]

    monkeypatch.setattr(restore_module, 'copy_restore', copy_restore)
    for rid, binding in bindings.items():
        manager.enqueue(RestoreCommand(rid, -binding.generation, binding.generation, 48, 48,
                                       ((binding.generation,),), resident=rid != 'cold'))
    return manager, checkpoints, claims, publications, fences, cold_release, hot_started


def test_resident_pipeline_can_enqueue_peer_before_first_fence(monkeypatch, tmp_path):
    manager, checkpoints, claims, published, fences, release, started = two_lane_manager(
        monkeypatch, tmp_path, resident_slots=2)
    try:
        hot, peer = manager.jobs['hot'], manager.jobs['next']
        assert manager.step(hot) and started.wait(2)
        hot.future.result(timeout=2)
        assert not manager.step(hot)
        assert manager.step(peer)
        peer.future.result(timeout=2)
        assert manager.maintenance.resident_inflight_bytes == 400
        peer.cancelled = True
        assert not manager.step(peer) and not published
        fences['next'].done = True
        assert manager.step(peer) and published == ['next']
        assert not hot.done and manager.maintenance.resident_inflight_bytes == 200
        assert 'hot' in manager.runner._kvmem_restore_allocations
        fences['hot'].done = True
        assert manager.step(hot) and published == ['next', 'hot']
        assert manager.maintenance.resident_inflight_bytes == 0
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.parametrize('cancelled', [False, True])
def test_hot_restore_finishes_while_cold_producer_is_blocked_and_keeps_its_own_fence(
        monkeypatch, tmp_path, cancelled):
    manager, checkpoints, claims, published, fences, cold_release, hot_started = two_lane_manager(monkeypatch, tmp_path)
    cold, hot, second = (manager.jobs[rid] for rid in ('cold', 'hot', 'next'))
    try:
        assert manager.step(cold)
        assert not cold.future.done()
        assert manager.step(hot) and hot_started.wait(2)
        hot.future.result(timeout=2)  # Only the test waits, never the engine.
        assert manager.stream is not manager.resident_stream
        assert manager.maintenance.owner == 'cold' and manager.maintenance.resident_owner == 'hot'
        assert claims == ['cold', 'hot']
        assert not manager.step(second) and second.checkpoint is None
        assert not manager.step(hot) and not hot.done  # Enqueued != CUDA-complete.
        if cancelled:
            output = replace(SchedulerOutput.make_empty(), finished_req_ids={'cold', 'hot', 'peer'},
                kv_connector_metadata=ControlMetadata(cancelled={'cold', 'hot'},
                    terminal_prefixes={'cold': 'cold-prefix', 'hot': 'hot-prefix', 'peer': 'peer-prefix'}))
            assert manager.defer_terminals(output).finished_req_ids == {'peer'}
        assert set(manager.runner._kvmem_restore_allocations) == {'cold', 'hot', 'next'}
        fences['hot'].done = True
        assert manager.step(hot) and hot.done and not cold.done
        assert published == ['hot'] and checkpoints['hot'].phase == 'live'
        assert 'hot' not in manager.runner._kvmem_restore_allocations
        assert manager.maintenance.owner == 'cold' and manager.maintenance.resident_owner is None
        if cancelled:
            released = manager.defer_terminals(replace(SchedulerOutput.make_empty(),
                                                       kv_connector_metadata=ControlMetadata()))
            assert released.finished_req_ids == {'hot'} and manager.deferred == {'cold': 'cold-prefix'}
        assert manager.step(second)
        second.future.result(timeout=2)
        fences['next'].done = True
        assert manager.step(second)
        cold_release.set()
        cold.future.result(timeout=2)
        assert not manager.step(cold)
        assert 'cold' in manager.runner._kvmem_restore_allocations
        fences['cold'].done = True
        assert manager.step(cold)
        assert published == ['hot', 'next', 'cold']
        assert not manager.runner._kvmem_restore_allocations
        assert manager.maintenance.inflight_bytes == manager.maintenance.resident_inflight_bytes == 0
    finally:
        cold_release.set()
        manager.shutdown()


def test_partial_terminal_pressure_restore_uses_fresh_assembly_and_preserves_model_prefix(
        monkeypatch, tmp_path):
    manager, checkpoints, claims, published, fences, release, _ = two_lane_manager(monkeypatch, tmp_path)
    job = manager.jobs['cold']
    binding = manager.runner._kvmem_sessions.bindings['cold']
    binding.pressure_restore = True
    binding.window = 32
    binding.mandatory_spans = ((0,4,'root'),)
    job.command = replace(job.command, window_tokens=32)
    calls = []
    def prepare(checkpoint, **kwargs):
        calls.append((checkpoint.position, kwargs))
        return SimpleNamespace(total_window_tokens=32)
    manager.runner._kvmem_sessions.pool.prepare_claim_window = prepare
    try:
        assert manager.step(job)
        assert calls == [(48, dict(wait=False, pressure=True, pressure_budget=32,
                                  mandatory_spans=((0,4,'root'),), metadata_binding=binding))]
        assert checkpoints['cold'].position == 48
        assert job.restore_context.assembler is not None
        release.set()
        job.future.result(timeout=2)
        assert not manager.step(job)  # Restore still waits for its own GPU fence.
        fences['cold'].done = True
        assert manager.step(job) and published == ['cold']
    finally:
        release.set()
        manager.shutdown()


def test_shutdown_joins_both_producers_before_fencing_both_streams(tmp_path):
    runner = SimpleNamespace(model=SimpleNamespace(forward=lambda: None))
    manager = AsyncRefresh(runner, SimpleNamespace())
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    completed, fenced = [], []

    def work(index):
        started[index].set()
        assert release.wait(5)
        completed.append(index)

    def sync(index):
        assert sorted(completed) == [0, 1]
        fenced.append(index)

    manager.stream = SimpleNamespace(synchronize=lambda: sync(0))
    manager.resident_stream = SimpleNamespace(synchronize=lambda: sync(1))
    manager.executor.submit(work, 0)
    manager.resident_executor.submit(work, 1)
    try:
        assert all(event.wait(2) for event in started)
        with ThreadPoolExecutor(max_workers=1) as shutdown:
            future = shutdown.submit(manager.shutdown)
            assert not future.done() and not fenced
            release.set()
            future.result(timeout=2)
        assert sorted(fenced) == [0, 1] and manager.closed
        with pytest.raises(RuntimeError, match='closed'):
            manager.enqueue(RestoreCommand('new', -1, 1, 48, 48, ((1,),), resident=True))
    finally:
        release.set()
        manager.shutdown()
