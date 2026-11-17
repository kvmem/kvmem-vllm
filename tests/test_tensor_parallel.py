from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')

from vllm_kvmem.core.archive_admission import ArchiveCapacity
from vllm_kvmem.serving.connector import ControlCompletions
from vllm_kvmem.serving.session_control import SessionNotice
from vllm_kvmem.serving.tensor_parallel import validate_parallelism
from vllm_kvmem.serving.tp_feedback import TensorParallelFeedback


def test_async_ack_different_steps_and_interleaved_requests():
    ledger = TensorParallelFeedback(2)
    pending = {'A': 1, 'B': 7}
    first = ControlCompletions(shards={0: ControlCompletions([('A', 1)], {'A': 384})})
    second = ControlCompletions(shards={1: ControlCompletions([('B', 7)])})
    assert not ledger.collect(second.aggregate(first).shards, pending).completed
    rest = {1: ControlCompletions([('A', 1)], {'A': 384}), 0: ControlCompletions([('B', 7)])}
    joined = ledger.collect(rest, pending)
    assert set(joined.completed) == set(pending.items())
    assert joined.selected_windows == {'A': 384}
    with pytest.raises(ValueError, match='duplicate'):
        first.aggregate(first)
    with pytest.raises(ValueError, match='Stale'):
        ledger.collect(rest, {})


def test_vllm_aggregator_holds_scheduler_commit_until_last_rank(monkeypatch):
    import vllm.distributed
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm_kvmem.serving.connector import KVMemConnector, RefreshCommand
    cfg = NS(kv_transfer_config=KVTransferConfig(kv_connector='KVMemConnector', kv_role='kv_both',
                                                  kv_connector_extra_config={'kvmem': {}}),
             parallel_config=NS(tensor_parallel_size=2))
    scheduler = KVMemConnector(cfg, KVConnectorRole.SCHEDULER, None)
    workers = []
    for rank in range(2):
        monkeypatch.setattr(vllm.distributed, 'get_tp_group', lambda r=rank: NS(rank_in_group=r))
        workers.append(KVMemConnector(cfg, KVConnectorRole.WORKER, None))
    aggregator = KVOutputAggregator.from_connector(scheduler, 2)
    def step():
        outputs = [ModelRunnerOutput.with_kv_conn_output_only(KVConnectorOutput(
            kv_connector_worker_meta=w.build_connector_worker_meta())) for w in workers]
        scheduler.update_connector_output(aggregator.aggregate(outputs).kv_connector_output)
    committed = []
    scheduler.scheduler_completion = lambda rid, cid, **kw: committed.append((rid, cid, kw))
    scheduler.queue(RefreshCommand('r', 7, ()))
    workers[1].worker_completions = [('r', 7)]
    workers[1].worker_selected_windows = {'r': 96}
    step()
    assert not committed and scheduler.pending == {'r': 7}
    workers[0].worker_completions = [('r', 7)]
    workers[0].worker_selected_windows = {'r': 96}
    step()
    assert committed == [('r', 7, {'selected_tokens': 96})] and not scheduler.pending


def test_disagreeing_window_cannot_ack():
    ledger = TensorParallelFeedback(2)
    with pytest.raises(ValueError, match='selected window'):
        ledger.collect({r: ControlCompletions([('A', 4)], {'A': n})
                        for r, n in enumerate((64, 96))}, {'A': 4})
    with pytest.raises(ValueError, match='history'):
        ledger.collect({r: ControlCompletions(sessions=[SessionNotice(
            'retired', ('s', 'x'), 3, 'B', history=n, window=n)])
            for r, n in enumerate((64, 96))}, {})


def test_partially_admitted_checkpoint_is_fenced_until_both_shards_evict():
    from test_session_control import ledger, request
    from vllm_kvmem.serving.session_worker import SessionWorker
    control, _ = ledger()
    req = request()
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)
    cmd = control.take_commands()
    owner = cmd.bindings[0]
    scope = (owner.namespace, owner.generation, req.request_id)
    control.finished(req)
    ranks = TensorParallelFeedback(2)
    first = [SessionNotice('captured', *scope, cmd.captures[0].position),
             SessionNotice('retired', *scope, 16, 12)]
    second = [SessionNotice('capture_deferred', *scope, cmd.captures[0].position),
              SessionNotice('retired', *scope)]
    assert not ranks.collect({0: ControlCompletions(sessions=first)}, {}).sessions
    reports = ranks.collect({1: ControlCompletions(sessions=second)}, {}).sessions
    assert [v.kind for v in reports] == ['capture_deferred', 'retired']
    assert reports[1].discard_shards
    for report in reports:
        control.consume(report)
    eviction, = control.take_commands().evictions
    assert eviction.allow_missing and control.has_pending_work()
    assert control.lookup(request('next'), 0) is None
    # One worker never acquired the snapshot. It may ACK only this explicit
    # partial eviction, while the other worker must run normal cleanup.
    worker = SessionWorker.__new__(SessionWorker)
    worker.bindings, worker.identities, worker.evictions = {}, {}, {}
    worker.pool, worker.completed = NS(entries={}), []
    worker.evict([eviction])
    gone = SessionNotice('evicted', owner.namespace, owner.generation, '')
    assert worker.completed == [gone]
    assert not ranks.collect({1: ControlCompletions(sessions=[gone])}, {}).sessions
    notice, = ranks.collect({0: ControlCompletions(sessions=[gone])}, {}).sessions
    control.consume(notice)
    assert not control.has_pending_work() and control.lookup(request('next'), 0) == 0


def test_archive_credits_readiness_and_session_lifecycle():
    ledger = TensorParallelFeedback(2)
    def report(n, blocked, history, credits):
        return ControlCompletions(archive_sequence=n, archive_capacity={'A': ArchiveCapacity(history, credits)},
                                  publication_sequence=n, publication_blocked=set(blocked))
    a = report(1, ['A'], 64, 8)
    b = report(1, ['B'], 48, 3)
    notice = SessionNotice('retired', ('s', 'x'), 2, 'A', history=48, window=32)
    a.sessions = [notice]
    assert not ledger.collect({0: a}, {}).sessions
    b.sessions = [notice]
    joined = ledger.collect({1: b}, {})
    assert joined.sessions == [notice]
    assert joined.archive_capacity == {'A': ArchiveCapacity(48, 3)}
    assert joined.publication_blocked == {'A', 'B'}
    assert ledger.collect({1: report(2, [], 64, 1)}, {}).publication_blocked == {'A', 'B'}
    assert ledger.collect({0: report(2, ['A'], 96, 3)}, {}).publication_blocked == {'A'}
    with pytest.raises(ValueError, match='Stale'):
        ledger.collect({0: a}, {})


def test_pure_tp_geometry():
    assert validate_parallelism(NS(tensor_parallel_size=2)) == 2
    assert validate_parallelism(NS(tensor_parallel_size=4)) == 4
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match='positive'):
            validate_parallelism(NS(tensor_parallel_size=invalid))
    with pytest.raises(ValueError, match='complete token history'):
        validate_parallelism(NS(tensor_parallel_size=2, decode_context_parallel_size=2))


def test_worker_uses_the_group_gate_for_publication(monkeypatch):
    from vllm_kvmem.serving import gil_gate, tensor_parallel
    monkeypatch.setenv('KVMEM_GIL_GATE_DRAIN_OPEN', '1')
    monkeypatch.setattr(tensor_parallel, '_group', object())
    assert not gil_gate.drain_open_enabled()
    monkeypatch.setattr(tensor_parallel, '_group', None)
    assert gil_gate.drain_open_enabled()


def _exercise_two_ranks(rank, rendezvous):
    from concurrent.futures import Future, ThreadPoolExecutor
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from vllm_kvmem.serving import tensor_parallel as tp
    from vllm_kvmem import _native
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    tp._group = NS(cpu_group=dist.group.WORLD, rank_in_group=rank, world_size=2, ranks=[0, 1])
    tp._private_cpu_group = dist.new_group(ranks=[0, 1], backend='gloo')
    try:
        offset = 20 * rank
        records = {i: NS(handle=NS(slot=offset+i), history_begin=i*32, count=32) for i in range(3)}
        assert tp.selected_slots(records, [0, 2] if rank == 0 else [21]) == (offset, offset+2)
        remaps = [NS(block_id=offset+i, to_base=j*32, n_tokens=32) for j, i in enumerate([2, 0])]
        tp.verify_window(NS(memory=NS(request_id='r'), history_end=96),
                         remaps if rank else remaps[::-1], {v.handle.slot: v for v in records.values()})
        with pytest.raises(RuntimeError, match='missing'):
            tp.selected_slots(records if rank == 0 else {1: records[1]}, [0, 2])
        duplicate = NS(handle=NS(slot=99), history_begin=0, count=32)
        with pytest.raises(RuntimeError, match='ambiguous'):
            tp.selected_slots(records if rank == 0 else {**records, 99: duplicate}, [0])
        def retry():
            raise _native.HistoryAdmissionRetry('retired')
        assert tp.admit_window(lambda: 37) == 37
        with pytest.raises(_native.HistoryAdmissionRetry):
            tp.admit_window(retry)
        with pytest.raises(RuntimeError, match='jointly acquire'):
            tp.admit_window(retry if rank else lambda: 37)
        local = torch.tensor([.5, .01, .49] if rank == 0 else [.01, .5, .49])
        torch.testing.assert_close(tp.mean_head_scores(local), torch.tensor([.255, .255, .49]))
        with ThreadPoolExecutor(max_workers=1) as executor:
            if rank:
                private = executor.submit(tp.private_transfer, tp.mean_head_scores, local + 3)
            public = tp.mean_head_scores(local)
            if not rank:
                private = executor.submit(tp.private_transfer, tp.mean_head_scores, local + 3)
            torch.testing.assert_close(private.result(), public + 3)
        ready = rank == 0
        capture = NS(publication_ready=lambda _: ready, authorities={'r': NS(history_end=128)})
        job = NS(dispatched=True, done=False, command=NS(command_id=2, request_id='r'),
                 cancelled=False, history=128, peers=set(), future=None, restore_event=None)
        steps = []
        def step(current):
            steps.append(current.command.command_id)
            current.done = True
            return True
        manager = NS(runtime=capture, jobs={'r': job}, step=step,
                     runner=NS(req_states=NS(req_id_to_index={'r': 0})))
        tp.pump_refresh(manager, ['peer'])
        assert not steps
        ready = True
        tp.pump_refresh(manager, ['peer'])
        assert steps == [2] and job.peers == {'peer'}
        job.done = False
        future = Future()
        def transfer(current):
            if current.future is None:
                current.future = future
            else:
                current.done = True
            return True
        manager.step = transfer
        tp.pump_refresh(manager, ['peer'])
        assert manager.tp_transfer is job and not job.done
        finished = (None, NS(event=NS(query=lambda: True)))
        if rank == 0:
            future.set_result(finished)
        tp.pump_refresh(manager, ['next-peer'])
        assert not job.done
        if rank == 1:
            future.set_result(finished)
        tp.pump_refresh(manager, ['next-peer'])
        assert job.done and manager.tp_transfer is None and job.peers == {'peer', 'next-peer'}
    finally:
        tp.close_worker()
        dist.destroy_process_group()


def test_two_real_gloo_ranks_use_logical_slots_and_a_common_replay_boundary(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_exercise_two_ranks, args=((tmp_path/'rendezvous').as_uri(),), nprocs=2)


def _four_head_shards(rank, rendezvous):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from vllm_kvmem.serving import tensor_parallel as tp
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=40))
    tp._group = NS(cpu_group=dist.group.WORLD, rank_in_group=rank, world_size=4, ranks=list(range(4)))
    try:
        torch.testing.assert_close(tp.mean_head_scores(torch.tensor([float(rank), 4.-rank])),
                                   torch.tensor([1.5, 2.5]))
        records = {i: NS(handle=NS(slot=20*rank+i), history_begin=i*32, count=32) for i in range(3)}
        assert tp.selected_slots(records, [2, 0] if rank == 0 else []) == (20*rank+2, 20*rank)
    finally:
        tp.close_worker()
        dist.destroy_process_group()


def test_four_head_shards_on_gloo(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_four_head_shards, args=((tmp_path/'rendezvous').as_uri(),), nprocs=4)


def _layerwise_two_ranks(rank, rendezvous):
    """Layerwise TP: per-layer agreement, flags, and the full-stack score reduction."""
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from vllm_kvmem.serving import tensor_parallel as tp
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    tp._group = NS(cpu_group=dist.group.WORLD, rank_in_group=rank, world_size=2, ranks=[0, 1])
    try:
        # Rank-local slots differ; the logical (history_begin, count) picks agree.
        picked = [NS(handle=NS(slot=50*rank+i), history_begin=32*i, count=32) for i in (3, 0, 7)]
        tp.agree_layer_selection('r', 4, picked if rank else picked[::-1], 'pressure')
        with pytest.raises(RuntimeError, match='different pressure blocks for r layer 5'):
            tp.agree_layer_selection('r', 5, picked if rank else picked[:2], 'pressure')
        assert tp.agree_flag(True, 'realign r') is True
        with pytest.raises(RuntimeError, match='disagree: realign q'):
            tp.agree_flag(bool(rank), 'realign q')
        # A layerwise packet: rank-local [layers, rows, cols] stack; layer 0 is the shared view.
        stack = torch.full((3, 2, 4), float(rank)) + torch.arange(3.).view(3, 1, 1)
        packet = NS(masses=stack[0], history_begin=64, rows=[0, 5], logical_group=None, event=None,
                    layerwise_stack=stack)
        packet.layerwise_pressure = ('controller', 'r', ('h',) * 3, stack)
        packet.layerwise_decode = ('controller', 'r', ('l',) * 3, stack)
        plain = NS(masses=torch.full((1, 2), 2.*rank), history_begin=0, rows=[1], logical_group=None, event=None)

        class Packets(dict):
            pass
        result = tp.publish_action_scores(Packets(a=packet, b=plain), None)
        expected = torch.full((3, 2, 4), .5) + torch.arange(3.).view(3, 1, 1)
        reduced = result['a'].layerwise_pressure[-1]
        torch.testing.assert_close(reduced, expected)
        assert result['a'].layerwise_decode[-1] is reduced and not hasattr(result['a'], 'layerwise_stack')
        assert result['a'].masses.data_ptr() == reduced[0].data_ptr() and not result['a'].masses.is_cuda
        torch.testing.assert_close(result['b'].masses, torch.ones(1, 2))
    finally:
        tp.close_worker()
        dist.destroy_process_group()


def test_layerwise_agreement_and_full_stack_reduction_on_gloo(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_layerwise_two_ranks, args=((tmp_path/'rendezvous').as_uri(),), nprocs=2)


def test_layerwise_tp_settings_are_enforced(monkeypatch):
    from vllm_kvmem.serving import tensor_parallel as tp
    options = NS(query_replay_cobatch=False, guided_query_tokens=0, historical_gc=None,
                 repository_gc=False, gpu_memory_admission='static')
    for name, value in tp.LAYERWISE_TP_SETTINGS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE', 'sequential')
    tp.validate_worker_options(options)
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PROMISE', '1')
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_PRESSURE_PROMISE=1'):
        tp.validate_worker_options(options)
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PROMISE', '0')
    for mode, no_probe in (('parallel', '0'), ('sequential', '1')):
        monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE', mode)
        monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NO_PROBE', no_probe)
        with pytest.raises(ValueError, match='sequential Query with the private probe'):
            tp.validate_worker_options(options)
