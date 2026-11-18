"""Historical GC under tensor parallelism: rank 0 proposes, every rank commits the agreed set.

The native worker of rank 0 runs its policy in PROPOSE mode (intents only),
followers run in FOLLOW mode (release jobs only), and serving.tensor_parallel
.gc_sync retires exactly the blocks all ranks accepted, at the same step.
"""
from types import SimpleNamespace as NS
import time

import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.core.historical_gc import HistoricalAuthorityGc, HistoricalGcSettings
from vllm_kvmem.core.memory_storage import MemoryArchive

PROPOSE, FOLLOW, LOCAL = n.HistoricalGcCommit.PROPOSE, n.HistoricalGcCommit.FOLLOW, n.HistoricalGcCommit.LOCAL


def eventually(predicate, seconds=5):
    end = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < end
        time.sleep(.001)


def configuration(commit=LOCAL, *, mode=n.HistoricalGcMode.DELETE, delete_blocks=2,
                  ready_low=2, ready_high=4, policy='lru', policy_params=None):
    config = n.HistoricalGcConfig()
    config.mode = mode
    config.policy = policy
    if policy_params is not None:
        config.policy_params = policy_params
    config.commit = commit
    config.background = False
    config.max_scopes = 1
    config.poll_ms = 1
    config.scan_blocks = 16
    config.delete_blocks = delete_blocks
    config.ready_low, config.ready_high = ready_low, ready_high
    return config


def history(path, commit=LOCAL, *, capacity=8, placeholders=0, configure=None, **config):
    """Eight sealed 4-token records 'r' of one archive; placeholders take the first slots."""
    path.mkdir(parents=True, exist_ok=True)
    archive = MemoryArchive(str(path), 8, 16, native_payloads=True)
    worker = archive.enable_history_gc(configuration(commit, **config))
    if configure is not None:
        configure(archive)
    repo = n.Repository(capacity, 4)
    memory = NS(request_id='r', repository=repo, history_end=32,
                _store=NS(attach_history=lambda scope: None, working_records=lambda: ()))
    gc = HistoricalAuthorityGc(memory, archive)
    held = [repo.reserve(1000 + 4*i, 4) for i in range(placeholders)]
    leases = [repo.reserve(4*i, 4).commit() for i in range(8)]
    archive.write('r', leases, b''.join(bytes([i+1])*8 for i in range(8))).result(5)
    for lease in leases:
        assert gc.scope.update(lease, n.HISTORY_SEALED | n.HISTORY_GPU)
    handles = [n.Handle(lease.handle.slot, lease.handle.generation) for lease in leases]
    return NS(archive=archive, worker=worker, repo=repo, memory=memory, gc=gc, scope=gc.scope,
              handles=handles, held=held)


def pairs(handles):
    return [(handle.slot, handle.generation) for handle in handles]


def drain(h):
    for completion in h.scope.completions(16):
        if completion.cpu and not h.archive.retire_cpu_record('r', completion.handle):
            continue
        assert h.scope.acknowledge(completion)


def close(h):
    for reservation in h.held:
        reservation.cancel()
    h.held.clear()

    def closed():
        h.worker.advance()
        drain(h)
        return h.archive.close()
    eventually(closed)


def advance(h, count=16):
    for _ in range(count):
        h.worker.advance()


def test_leader_proposes_without_retiring(tmp_path):
    h = history(tmp_path, PROPOSE)
    try:
        assert h.worker.commit == PROPOSE
        advance(h)
        assert h.scope.external_proposal_ready
        proposals = pairs(h.scope.external_proposals())
        assert proposals == pairs(h.handles[:2])           # LRU: the two oldest
        assert h.scope.stats['decisions'] == 2 and h.scope.stats['retired'] == 0
        assert all(h.repo.pin(handle) for handle in h.handles)
        advance(h)
        # The batch waits for the synchronized commit: no retirement, no new round.
        assert pairs(h.scope.external_proposals()) == proposals
        assert h.scope.stats['decisions'] == 2 and h.scope.stats['retired'] == 0
        assert not h.scope.completions(16)
    finally:
        close(h)


def test_expired_tools_revalidate_at_prepare_and_retry_follower_veto(tmp_path):
    leader = history(tmp_path/'leader', PROPOSE, policy='tool_result_age')
    follower = history(tmp_path/'follower', FOLLOW, policy='tool_result_age')
    try:
        leader.scope.set_expired_tool_ranges([(4, 8)])
        advance(leader)
        stale = leader.scope.external_proposals()
        assert pairs(stale) == pairs(leader.handles[1:2])
        leader.scope.set_expired_tool_ranges([])  # The accepted prefix was rewound.
        assert leader.scope.external_prepare(stale) == [False]
        assert leader.scope.external_commit([]) == []
        leader.scope.external_release()
        assert leader.repo.pin(stale[0])

        leader.scope.set_expired_tool_ranges([(4, 8)])
        advance(leader)
        choices = leader.scope.external_proposals()
        assert leader.scope.external_prepare(choices) == [True]
        assert follower.scope.external_prepare(follower.handles[1:2]) == [False]
        assert leader.scope.external_commit([]) == []
        leader.scope.external_release()
        follower.scope.external_release()

        follower.scope.set_expired_tool_ranges([(4, 8)])
        advance(leader)
        choices = leader.scope.external_proposals()
        assert pairs(choices) == pairs(leader.handles[1:2])
        assert leader.scope.external_prepare(choices) == follower.scope.external_prepare(follower.handles[1:2]) == [True]
        assert leader.scope.external_commit(choices) == follower.scope.external_commit(follower.handles[1:2]) == [True]
        leader.scope.external_release()
        follower.scope.external_release()
        assert leader.scope.stats['tool_result_age_retired'] == follower.scope.stats['retired'] == 1
        assert not leader.repo.pin(leader.handles[1]) and not follower.repo.pin(follower.handles[1])
    finally:
        close(leader)
        close(follower)


FALLBACK = {'keep_mandatory': True, 'tool_result_age_steps': 12, 'lru_fallback': True}


def test_tool_age_lru_fallback_proposes_under_pressure_and_followers_accept(tmp_path):
    strict = history(tmp_path/'strict', PROPOSE, policy='tool_result_age')
    leader = history(tmp_path/'leader', PROPOSE, policy='tool_result_age', policy_params=FALLBACK)
    follower = history(tmp_path/'follower', FOLLOW, policy='tool_result_age', policy_params=FALLBACK)
    strict_follower = history(tmp_path/'strict-follower', FOLLOW, policy='tool_result_age')
    try:
        # Full repository, no expired tool reply: only the fallback relieves the pressure.
        advance(strict)
        assert not strict.scope.external_proposals()
        advance(leader)
        choices = leader.scope.external_proposals()
        assert pairs(choices) == pairs(leader.handles[:2])          # LRU: the two oldest
        # A strict follower still vetoes records that are not expired tool pages.
        assert strict_follower.scope.external_prepare(strict_follower.handles[:2]) == [False, False]
        strict_follower.scope.external_release()
        assert leader.scope.external_prepare(choices) == follower.scope.external_prepare(follower.handles[:2]) == [True, True]
        assert leader.scope.external_commit(choices) == follower.scope.external_commit(follower.handles[:2]) == [True, True]
        leader.scope.external_release()
        follower.scope.external_release()
        assert leader.scope.stats['retired'] == follower.scope.stats['retired'] == 2
        assert leader.scope.stats['tool_result_age_retired'] == 0
        assert not leader.repo.pin(leader.handles[0]) and not follower.repo.pin(follower.handles[0])
    finally:
        for h in (strict, leader, follower, strict_follower):
            close(h)


def test_external_commit_retires_exactly_the_agreed_set(tmp_path):
    h = history(tmp_path, PROPOSE)
    try:
        advance(h)
        proposals = h.scope.external_proposals()
        assert h.scope.external_prepare(proposals) == [True, True]
        # A prepared record keeps its policy gate: no new hold until release.
        assert h.scope.protect([proposals[1]]) is None
        assert h.scope.external_commit([proposals[0]]) == [True]
        assert not h.repo.pin(proposals[0]) and h.repo.pin(proposals[1])
        advance(h, 4)
        # Its CPU release starts only after release (after the caller's index update),
        # and the leader resolves its batch then too.
        assert not h.scope.completions(16)
        assert h.scope.external_proposal_ready
        h.scope.external_release()
        stats = h.scope.stats
        assert stats['retired'] == 1 and stats['rejected_ineligible'] == 1
        assert not h.scope.external_proposal_ready
        hold = h.scope.protect([proposals[1]])
        assert hold is not None
        del hold
        eventually(lambda: (h.worker.advance(), any(c.cpu for c in h.scope.completions(16)))[1])
        # The policy round continues with the next oldest blocks.
        advance(h, 4)
        assert pairs(h.scope.external_proposals()) == pairs(h.handles[2:4])
    finally:
        close(h)


def test_follower_never_runs_the_policy_but_releases_agreed_blocks(tmp_path):
    h = history(tmp_path, FOLLOW)
    try:
        h.scope.request_capacity(8)
        advance(h, 32)
        stats = h.scope.stats
        assert stats['decisions'] == stats['examined'] == stats['retired'] == 0
        assert not h.scope.external_proposal_ready and h.scope.external_proposals() == []
        assert h.scope.external_prepare(h.handles[:2]) == [True, True]
        assert h.scope.external_commit(h.handles[:2]) == [True, True]
        h.scope.external_release()
        released = set()

        def both_released():
            h.worker.advance()
            released.update((c.handle.slot, c.handle.generation) for c in h.scope.completions(16) if c.cpu)
            return released == set(pairs(h.handles[:2]))
        eventually(both_released)
        assert h.scope.stats['retired'] == 2 and h.scope.stats['decisions'] == 0
    finally:
        close(h)


def test_prepare_refuses_what_this_rank_must_keep_and_defers_the_pause(tmp_path):
    h = history(tmp_path, FOLLOW)
    try:
        hold = h.scope.protect([h.handles[0]])
        lease = h.repo.pin(h.handles[1])
        assert h.scope.update(lease, n.HISTORY_SEALED | n.HISTORY_MANDATORY)
        del lease
        assert h.scope.external_prepare(h.handles[:3]) == [False, False, True]
        h.scope.external_release()
        stale = n.Handle(h.handles[2].slot, h.handles[2].generation + 100)
        assert h.scope.external_prepare([stale]) == [False]
        h.scope.external_release()
        # A pause requested during a prepared round is acknowledged after release.
        assert h.scope.external_prepare([h.handles[2]]) == [True]
        h.scope.pause_policy(True)
        advance(h, 2)
        assert not h.scope.policy_paused
        h.scope.external_release()
        advance(h, 2)
        assert h.scope.policy_paused
        assert h.scope.external_prepare([h.handles[2]]) == [False]
        h.scope.external_release()
        h.scope.pause_policy(False)
        del hold
        assert h.scope.stats['retired'] == 0 and all(h.repo.pin(handle) for handle in h.handles)
    finally:
        close(h)


def test_commit_mode_is_chosen_before_registration(tmp_path):
    archive = MemoryArchive(str(tmp_path/'a'), 8, 16, native_payloads=True)
    worker = archive.enable_history_gc(configuration())
    assert worker.commit == LOCAL
    worker.set_commit(FOLLOW)
    scope = archive.register_history_scope('r', n.Repository(8, 4))
    worker.set_commit(FOLLOW)                                   # Same mode: no-op.
    with pytest.raises(RuntimeError, match='before any scope registers'):
        worker.set_commit(PROPOSE)
    assert scope.external_prepare([]) == []
    eventually(archive.close)
    shadow = MemoryArchive(str(tmp_path/'b'), 8, 16, native_payloads=True)
    with pytest.raises(ValueError, match='shadow'):
        shadow.enable_history_gc(configuration(PROPOSE, mode=n.HistoricalGcMode.SHADOW))
    eventually(shadow.close)
    # TP=1 keeps its own native commits and refuses the external path.
    local = history(tmp_path/'c')
    try:
        with pytest.raises(RuntimeError, match='tensor-parallel commit mode'):
            local.scope.external_prepare(local.handles[:1])
        advance(local)
        assert local.scope.stats['retired'] == 4 and not local.scope.external_proposal_ready
    finally:
        close(local)


def test_synchronized_drain_only_finishes_released_records(tmp_path):
    h = history(tmp_path, FOLLOW)
    records = {(handle.slot, handle.generation): NS(handle=handle, history_begin=4*i, count=4,
                                                    archived=True, index_owner=None)
               for i, handle in enumerate(h.handles)}
    try:
        keys = pairs(h.handles[:2])
        assert h.scope.external_prepare(h.handles[:2]) == [True, True]
        assert h.scope.external_commit(h.handles[:2]) == [True, True]
        retired = []
        h.gc.retire_synchronized(records, keys[:1], retired)
        h.scope.external_release()
        assert retired == [(0, *keys[0])] and h.gc.reclaimed_records == 1
        # The second release reaches drain while its record is still listed: a
        # rank-local removal would diverge the ranks, so it fails loudly.
        with pytest.raises(RuntimeError, match='outside the synchronized commit'):
            eventually(lambda: (h.worker.advance(), h.gc.drain(records, synchronized=True))[1] is not None)
        records.pop(keys[1])
        eventually(lambda: (h.worker.advance(), h.gc.drain(records, synchronized=True),
                            not h.gc.retired_index)[-1])
        assert len(records) == 6
    finally:
        close(h)


def test_tensor_parallel_accepts_only_leader_decided_historical_gc():
    pytest.importorskip('vllm')
    from vllm_kvmem.serving import tensor_parallel as tp
    base = dict(query_replay_cobatch=False, guided_query_tokens=0, repository_gc=False,
                gpu_memory_admission='static')
    tp.validate_worker_options(NS(**base, historical_gc=HistoricalGcSettings(mode='delete', policy='redundancy')))
    tp.validate_worker_options(NS(**base, historical_gc=HistoricalGcSettings(mode='delete')))
    with pytest.raises(ValueError, match='mode=delete'):
        tp.validate_worker_options(NS(**base, historical_gc=HistoricalGcSettings(mode='shadow')))
    with pytest.raises(ValueError, match='mode=delete'):
        tp.validate_worker_options(NS(**base, historical_gc=HistoricalGcSettings(
            mode='delete', policy_params=dict(keep_mandatory=False))))
    with pytest.raises(ValueError, match='retained history'):
        tp.validate_worker_options(NS(**{**base, 'repository_gc': True}, historical_gc=None))


def test_local_vetoes_and_votes():
    pytest.importorskip('vllm')
    from vllm_kvmem.serving import tensor_parallel as tp
    from vllm_kvmem.serving.connector import RestoreCommand
    a, b, c, d, parked = (NS(name=name) for name in ('a', 'b', 'c', 'd', 'parked'))
    authorities = dict(a=a, b=b, c=c, d=d, parked=parked)
    runtime = NS(authorities={'ra': a, 'rb': b})
    jobs = {'ra': NS(done=False, command=NS(request_id='ra')), 'rb': NS(done=True, command=NS(request_id='rb'))}
    sessions = NS(pool=NS(entries={'x': NS(authority=c, phase=NS(name='IDLE')),
                                   'y': NS(authority=d, phase=NS(name='CLAIMED'))}, prompts={}))
    # A refreshing owner and a checkpoint being claimed are busy; idle and done ones are not.
    assert tp._gc_busy(runtime, NS(jobs=jobs, tp_transfer=None), sessions, authorities) == {id(a), id(d)}
    # A restore in flight may import any parked authority.
    restore = RestoreCommand('rb', 1, 1, 0, 0, ())
    busy = tp._gc_busy(runtime, NS(jobs={'rb': NS(done=False, command=restore)}), None, authorities)
    assert busy == {id(b), id(c), id(d), id(parked)}
    assert tp._gc_busy(None, None, None, authorities) == set()
    # Votes: lagging publication is "not now", a hole or a second record is a divergence.
    record = lambda begin, count, archived=True, slot=5: NS(history_begin=begin, count=count, archived=archived,
                                                            handle=n.Handle(slot, 1))
    index = {0: record(0, 4), 4: record(4, 2, archived=False), 8: False}
    assert tp._gc_vote(index, 6, 0, 4)[0] == 1
    assert tp._gc_vote(index, 6, 0, 4, expected=(6, 1)) == (-1, None)
    assert tp._gc_vote(index, 6, 4, 4) == (0, None)          # Tail still growing here.
    assert tp._gc_vote(index, 8, 4, 4) == (-1, None)         # Published, but a different record.
    assert tp._gc_vote(index, 6, 12, 4) == (0, None)         # Not published here yet.
    assert tp._gc_vote(index, 16, 12, 4) == (-1, None)       # Published but missing.
    assert tp._gc_vote(index, 16, 8, 4) == (-1, None)        # Ambiguous.


def _authority(h):
    """A RequestAuthority shell over a real native scope (records keyed by handle)."""
    from vllm_kvmem.serving.authority import RequestAuthority, VersionedRecords
    authority = RequestAuthority.__new__(RequestAuthority)
    authority.memory, authority.archive, authority.gc = h.memory, h.archive, h.gc
    authority.records = VersionedRecords()
    for i, handle in enumerate(h.handles):
        authority.records[(handle.slot, handle.generation)] = NS(
            handle=handle, history_begin=4*i, count=4, archived=True, index_owner=None, staging_slot=None)
    authority.closing = False
    authority._rewind_position = None
    authority.discarded_records = {}
    authority._pending_cache_retire = []
    authority._index_snapshot_depth = 0
    authority.query_window_cache = None
    authority._tp_history_gc = True
    return authority


def _two_ranks(rank, rendezvous, root):
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from vllm_kvmem.serving import tensor_parallel as tp
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    tp._group = NS(cpu_group=dist.group.WORLD, rank_in_group=rank, world_size=2, ranks=[0, 1])
    # Rank 1's history lands on other physical slots: two writes hold its first slots.
    h = history(Path(root)/f'rank{rank}', capacity=10, placeholders=2*rank, configure=tp.configure_history_gc,
                delete_blocks=4, ready_low=3, ready_high=5)
    authority = _authority(h)
    tp.register_gc_authority(authority)
    calls, real = [], (dist.broadcast, dist.all_reduce)

    def counted(name, function):
        def call(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return call
    dist.broadcast, dist.all_reduce = counted('broadcast', real[0]), counted('all_reduce', real[1])

    def live():
        return sorted((r.history_begin, r.count) for r in authority.records.values())

    def everywhere(value):
        gathered = [None, None]
        dist.all_gather_object(gathered, value)
        assert gathered[0] == gathered[1], gathered
        return value

    def sync(*args):
        del calls[:]
        return tp.gc_sync(*args)

    hold = None
    try:
        assert h.worker.commit == (PROPOSE if rank == 0 else FOLLOW)
        assert [handle.slot for handle in h.handles] == list(range(2*rank, 2*rank + 8))
        # 1. Nothing proposed yet: one small broadcast.
        assert sync() == [] and calls == ['broadcast']
        # 2. Rank 0 proposes the three oldest blocks; followers never run a policy.
        advance(h)
        if rank == 0:
            assert h.scope.external_proposal_ready and h.scope.stats['decisions'] == 3
        else:
            assert not h.scope.external_proposal_ready and h.scope.stats['decisions'] == 0
        assert h.scope.stats['retired'] == 0
        # Block 0 is held on rank 1 only: skipped on both. The others retire on both.
        if rank == 1:
            hold = h.scope.protect([h.handles[0]])
            assert hold is not None
        result = sync()
        assert calls == ['broadcast', 'broadcast', 'all_reduce', 'all_reduce']
        # Same logical blocks, each rank's own physical handles.
        assert result == [('r', [(4, *pairs(h.handles)[1]), (8, *pairs(h.handles)[2])])]
        assert everywhere([(rid, [row[0] for row in rows]) for rid, rows in result]) == [('r', [4, 8])]
        expected = [(4*i, 4) for i in range(8) if i not in (1, 2)]
        assert everywhere(live()) == expected
        assert not h.repo.pin(h.handles[1]) and not h.repo.pin(h.handles[2]) and h.repo.pin(h.handles[0])
        assert h.scope.stats['retired'] == 2
        if rank == 0:
            assert h.scope.stats['rejected_ineligible'] == 1 and not h.scope.external_proposal_ready
        # The resolved batch leaves nothing pending: one collective again.
        assert sync() == [] and calls == ['broadcast']
        # 3. CPU release completes later, per rank; drain only finishes it.
        eventually(lambda: (h.worker.advance(), authority._drain_gc(), not authority.gc.retired_index)[-1])
        assert everywhere(live()) == expected
        del hold
        hold = None
        # 4. A follower whose publication lags refuses; nothing retires anywhere.
        if rank == 0:
            h.scope.request_capacity(10)
        advance(h)
        if rank == 0:
            assert h.scope.external_proposal_ready
        for record in authority.records.values():
            record.archived = rank == 0
        assert sync() == [] and calls == ['broadcast', 'broadcast', 'all_reduce']
        for record in authority.records.values():
            record.archived = True
        assert everywhere(live()) == expected
        # 5. An owner busy with a refresh on one rank vetoes it on every rank.
        advance(h)
        if rank == 0:
            assert h.scope.external_proposal_ready
        busy =(NS(authorities={'r': authority}), NS(jobs={'r': NS(done=False, command=NS(request_id='r'))}))
        assert sync(*(busy if rank == 1 else ())) == []
        assert calls == ['broadcast', 'broadcast', 'all_reduce']
        assert everywhere(live()) == expected
        # 6. A logical block missing on a follower (inside its published history) fails every rank.
        advance(h)
        if rank == 0:
            assert h.scope.external_proposal_ready
        if rank == 1:
            authority.records.clear()
        with pytest.raises(RuntimeError, match='missing'):
            sync()
        assert h.scope.stats['retired'] == 2
    finally:
        dist.broadcast, dist.all_reduce = real
        tp.close_worker()
        dist.destroy_process_group()
        del hold
        authority.records.clear()
        close(h)


def test_two_gloo_ranks_retire_the_same_logical_blocks_at_the_same_step(tmp_path):
    pytest.importorskip('vllm')
    import torch.multiprocessing as mp
    mp.spawn(_two_ranks, args=((tmp_path/'rendezvous').as_uri(), str(tmp_path)), nprocs=2)
