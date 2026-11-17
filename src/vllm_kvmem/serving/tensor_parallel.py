"""Tensor-parallel workers keep their KV shards and agree on logical history."""
import hashlib
import os
import struct
import threading
import weakref
from contextvars import ContextVar
from ..core import settings as _settings


_group = None
_private_cpu_group = None
_private = ContextVar('kvmem_private_tp_transfer', default=False)


def active():
    return _group is not None


def validate_parallelism(parallel):
    size = getattr(parallel, 'tensor_parallel_size', 1)
    if any(getattr(parallel, name, 1) != 1 for name in (
            'pipeline_parallel_size', 'prefill_context_parallel_size', 'decode_context_parallel_size')):
        raise ValueError('KVMem requires complete token history on each worker (PP=PCP=DCP=1)')
    if type(size) is not int or size < 1:
        raise ValueError('KVMem requires a positive tensor_parallel_size')
    return size


def configure_worker(config, modules):
    global _group, _private_cpu_group
    size = validate_parallelism(getattr(config, 'parallel_config', None))
    if size == 1:
        close_worker()
        return
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    text = config.model_config.hf_text_config
    if (group.world_size != size or any(
            m.num_heads * size != text.num_attention_heads or
            m.num_kv_heads * size != text.num_key_value_heads for m in modules)):
        raise ValueError('KVMem TP requires even, disjoint Q and KV head shards on all ranks')
    _group = group
    import torch.distributed as dist
    _private_cpu_group = dist.new_group(ranks=group.ranks, backend='gloo', use_local_synchronization=True)


def close_worker():
    global _group, _private_cpu_group
    if _private_cpu_group is not None:
        import torch.distributed as dist
        dist.destroy_process_group(_private_cpu_group)
    _group = _private_cpu_group = None
    with _gc_lock:
        _gc_authorities.clear()


def private_transfer(fn, *args):
    if _private_cpu_group is None:
        return fn(*args)
    token = _private.set(True)
    try:
        return fn(*args)
    finally:
        _private.reset(token)


def submit_transfer(executor, fn, *args):
    if _private_cpu_group is None:
        return executor.submit(fn, *args)
    return executor.submit(private_transfer, fn, *args)


def _cpu_group():
    return _private_cpu_group if _private.get() else _group.cpu_group


# Layerwise selection under TP (sequential Query + prefill/rolling pressure).
# Selections are made on every rank from rank-identical (head-reduced) scores
# and checked to agree before any rank writes KV. These exact optimizations
# start work, collectives or graph-mode decisions on rank-local timing
# (post-forward priming, promises, early leases, deferred work resolved on
# first use, extra lanes, restore-first queues, layer-0 fast path), so TP
# launches turn them off; selections are unchanged by them.
LAYERWISE_TP_SETTINGS = {
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE': '0',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE_HANDOFF': '0',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE_PROGRESS': '0',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE_PREFORWARD': '0',
    'KVMEM_LAYERWISE_PRESSURE_PROMISE': '0',
    'KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK': '0',
    'KVMEM_LAYERWISE_PRESSURE_EARLY_LEASE': '0',
    'KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST': '0',
    'KVMEM_LAYERWISE_PRESSURE_LANES': '1',
    'KVMEM_LAYERWISE_PRESSURE_POST_FORWARD_REAP': '0',
    'KVMEM_LAYERWISE_PRESSURE_DEFERRED_MAPS': '0',
    'KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD': '0',
    'KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE': '0',
    'KVMEM_LAYERWISE_PIPELINE_SWAPS': '0',
    'KVMEM_PIPELINE_DISPATCH_BATCH_MEMBERS': '0',
    'KVMEM_PROMISE_PM_SESSIONS': '0',
    'KVMEM_PROMISE_ASYNC_PAGE_CHECK': '0',
    'KVMEM_RESTORE_FIRST': '0',
    'KVMEM_LAYER0_FAST': '0',
}


def layerwise_tp_violations(get=None):
    """Settings that a TP layerwise worker may not run with (name, value, required)."""
    get = get or _settings.get
    return [(name, get(name), value) for name, value in LAYERWISE_TP_SETTINGS.items()
            if get(name) is not None and get(name) != value]


def validate_worker_options(options):
    mode = _settings.get('KVMEM_LAYERWISE_QUERY_MODE')
    if options.query_replay_cobatch or _settings.get('KVMEM_SYSTEMONE_FORK') == '1':
        raise ValueError('KVMem TP uses global or layerwise selection with private Query replay on all ranks')
    if mode:
        if mode != 'sequential' or _settings.get('KVMEM_LAYERWISE_QUERY_NO_PROBE') == '1':
            raise ValueError('Tensor-parallel layerwise selection requires sequential Query with the private probe')
        bad = layerwise_tp_violations()
        if bad:
            raise ValueError('Tensor-parallel layerwise selection requires rank-synchronous settings: '
                             + ', '.join('%s=%s (needs %s)' % row for row in bad))
    if options.guided_query_tokens or _settings.get('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES') == 'sum':
        raise ValueError('Tensor-parallel Query scoring requires the public Query')
    if options.repository_gc:
        raise ValueError('Tensor-parallel selection requires retained history on every worker')
    gc = options.historical_gc
    if gc is not None:
        # Leader-decided historical GC (gc_sync): rank 0's policy proposes and
        # every rank retires the agreed logical blocks at the same step.
        params = getattr(gc, 'policy_params', None) or {}
        if getattr(gc, 'mode', None) not in ('off', 'delete') or not params.get('keep_mandatory', True):
            raise ValueError('Tensor-parallel historical GC requires mode=delete (or off) with keep_mandatory '
                             '(leader-decided, rank-synchronous retirement)')
    if options.gpu_memory_admission != 'static':
        raise ValueError('Tensor-parallel serving requires static GPU admission')


def pump_refresh(manager, ready_requests):
    """Start private model collectives in the same order on every worker.

    A rank-local timer or DMA event cannot decide when one worker enters
    another TP model forward. Agree on a ready owner at this engine boundary
    and return to public work as soon as its private transfer is enqueued.
    """
    import time
    import torch
    import torch.distributed as dist
    from .connector import RestoreCommand
    capture = manager.runtime
    current = getattr(manager, 'tp_transfer', None)
    if current is not None:
        if current.command.request_id in ready_requests:
            raise RuntimeError('A tensor-parallel transfer owner entered a public batch')
        current.peers.update(ready_requests)
        ready = 0
        if current.future.done():
            if current.future.exception() is not None:
                ready = -1
            else:
                current.result = current.future.result()
                fence = (current.result[2] if isinstance(current.command, RestoreCommand)
                         else current.result[1].event)
                ready = int(fence.query())
        state = torch.tensor([ready], dtype=torch.int32)
        dist.all_reduce(state, op=dist.ReduceOp.MIN, group=_group.cpu_group)
        if state.item() < 0:
            if ready < 0:
                current.future.result()
            raise RuntimeError('A tensor-parallel worker failed to prepare the requested window')
        if not state.item():
            return
        if not manager.step(current) or not current.done:
            raise RuntimeError('Ready tensor-parallel transfer did not commit')
        manager.tp_transfer = None
    jobs = sorted((job for job in manager.jobs.values() if job.dispatched and not job.done),
                  key=lambda job: (job.command.command_id, job.command.request_id))
    if not jobs:
        return
    eligible = []
    for job in jobs:
        rid = job.command.request_id
        publication = capture.publication_ready((rid,))
        exists = rid in manager.runner.req_states.req_id_to_index
        authority = capture.authorities.get(rid)
        accepted = (isinstance(job.command, RestoreCommand) or
                    (authority is not None and authority.history_end >= job.history))
        eligible.append((job.cancelled and not isinstance(job.command, RestoreCommand)) or
                        (publication and exists and accepted))
    mask = torch.tensor(eligible, dtype=torch.int32)
    dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=_group.cpu_group)
    for job, allowed in zip(jobs, mask.tolist()):
        if not allowed:
            continue
        rid = job.command.request_id
        if rid in ready_requests:
            raise RuntimeError('Suspended TP refresh request entered a public batch')
        job.peers.update(ready_requests)
        while not job.done:
            progressed = manager.step(job)
            if job.future is not None:
                # This lane uses its own CPU process group. Neither the GPU
                # model collectives nor foreground score fences wait for it.
                manager.tp_transfer = job
                break
            if progressed:
                continue
            if job.restore_event is not None:
                job.restore_event.synchronize()
            else:
                # In-flight archive/checkpoint transfers advance independently.
                # Their ready fence belongs to this owner on this rank.
                time.sleep(.001)
        break


def mean_head_scores(scores):
    """Each shard already reduced its local heads after per-head softmax."""
    if _group is None or not scores.numel():
        return scores
    import torch.distributed as dist
    # Kernel outputs may be persistent buffers or non-contiguous views.
    # Private Query scoring produces one float per candidate page. The CPU
    # picker consumes it immediately. Keep those reductions independent of
    # the public model's NCCL ordering, even while the next batch executes.
    result = (scores.cpu() if _private.get() else scores).float().contiguous().clone()
    group = _group.device_group if result.is_cuda else _cpu_group()
    dist.all_reduce(result, group=group)
    return result.mul_(1 / _group.world_size)


def _agree(digest, error):
    import torch
    import torch.distributed as dist
    signature = torch.tensor(struct.unpack('<qq', digest.digest()), dtype=torch.int64)
    gathered = [torch.empty_like(signature) for _ in range(_group.world_size)]
    dist.all_gather(gathered, signature, group=_cpu_group())
    if any(not torch.equal(signature, witness) for witness in gathered):
        raise RuntimeError(error)


def verify_query_domain(request_id, records, mask, rows):
    if _group is None:
        return
    digest = hashlib.blake2b(digest_size=16)
    digest.update(request_id.encode())
    digest.update(struct.pack('<QQ', rows, len(records)))
    for record, eligible in zip(records, mask, strict=True):
        digest.update(struct.pack('<QQ?', record.history_begin, record.count, eligible))
    _agree(digest, f'Tensor-parallel workers have different Query candidates for {request_id}')


def selected_slots(records, selected):
    """Translate the coordinator's ordered logical selection to local handles."""
    import torch
    import torch.distributed as dist
    leader = _group.rank_in_group == 0
    src = _group.ranks[0]
    coordinates = []
    if leader:
        by_slot = {r.handle.slot: r for r in records.values()}
        coordinates = [(by_slot[s].history_begin, by_slot[s].count) for s in selected if s in by_slot]
    length = len(coordinates)
    if leader and length != len(selected):
        length = -1
    count = torch.tensor([length], dtype=torch.int64)
    dist.broadcast(count, src, group=_cpu_group())
    count = count.item()
    if count < 0:
        raise RuntimeError('Tensor-parallel coordinator selected a retired record')
    if leader:
        positions = torch.tensor(coordinates, dtype=torch.int64).reshape(-1, 2)
    else:
        positions = torch.empty((count, 2), dtype=torch.int64)
    if positions.numel():
        dist.broadcast(positions, src, group=_cpu_group())
    by_position = {}
    for record in records.values():
        position = (record.history_begin, record.count)
        by_position[position] = None if position in by_position else record.handle.slot
    slots = tuple(by_position.get(tuple(row)) for row in positions.tolist())
    complete = torch.tensor([None not in slots], dtype=torch.int32)
    dist.all_reduce(complete, op=dist.ReduceOp.MIN, group=_cpu_group())
    if not complete.item():
        raise RuntimeError('A selected logical record is missing or ambiguous on a tensor-parallel worker')
    return slots


def admit_window(admit):
    """A retry is safe only while no worker has acquired the new window."""
    import torch
    import torch.distributed as dist
    from .. import _native
    error = plan = None
    status = 0
    try:
        plan = admit()
    except _native.HistoryAdmissionRetry as exc:
        error, status = exc, 1
    except Exception as exc:
        error, status = exc, 2
    limits = torch.tensor([status, -status], dtype=torch.int32)
    dist.all_reduce(limits, op=dist.ReduceOp.MIN, group=_cpu_group())
    low, negative_high = limits.tolist()
    if low == negative_high == 0:
        return plan
    if low == 1 and negative_high == -1:
        raise error
    if status == 2:
        raise error
    raise RuntimeError('Tensor-parallel workers could not jointly acquire the selected window')


def publish_action_scores(packets, pool):
    """Reduce once, after local MTP staging and any native-validation retry."""
    if _group is None:
        return packets
    import torch
    from .logical_action import project_logical_scores
    digest = hashlib.blake2b(digest_size=16)
    scored = []
    for rid, packet in sorted(packets.items()):
        if packet.masses is None or not packet.masses.numel():
            continue
        rows, cols = packet.masses.shape
        # Layerwise packets reduce every layer's masses (layer 0 is its first slice).
        stack = getattr(packet, 'layerwise_stack', None)
        layers = stack.shape[0] if stack is not None else 0
        digest.update(rid.encode() + b'\0')
        digest.update(struct.pack('<QQQQ', packet.history_begin, rows, cols, layers))
        digest.update(struct.pack(f'<{rows}Q', *packet.rows))
        scored.append((rid, packet, stack))
    _agree(digest, 'Tensor-parallel workers sampled different public rows')
    if not scored:
        return packets
    # One collective and one pinned transfer per public batch, also at C32.
    values = [(stack if stack is not None else p.masses).reshape(-1) for _, p, stack in scored]
    merged = mean_head_scores(values[0] if len(values) == 1 else torch.cat(values))
    if pool is not None:
        host, event = pool.copy(merged)
    else:
        host = torch.empty_like(merged, device='cpu', pin_memory=True)
        host.copy_(merged, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
    offset = 0
    for rid, packet, stack in scored:
        source = stack if stack is not None else packet.masses
        reduced = host.narrow(0, offset, source.numel()).view(source.shape)
        offset += source.numel()
        packet.event = event
        layerwise = None
        if stack is None:
            packet.masses = reduced
        else:
            packet.masses = reduced[0]
            del packet.layerwise_stack
            for name in ('layerwise_pressure', 'layerwise_decode'):
                marker = getattr(packet, name, None)
                if marker is not None:
                    if marker[-1] is not stack:
                        raise RuntimeError('Layerwise TP marker does not hold this packet\'s score stack')
                    setattr(packet, name, marker[:-1] + (reduced,))
            layerwise = getattr(packet, 'layerwise_pressure', None)
        if packet.logical_group is not None:
            packets.logical_slices[rid] = project_logical_scores(
                packet.logical_group, packet.tp_logical_plans, packet.masses,
                physical_rows=tuple(packet.rows), event=event, layerwise_pressure=layerwise)
            del packet.tp_logical_plans
    return packets


def agree_flag(value, what):
    """A rank-local decision that gates later collectives must be the same on every rank."""
    if _group is None:
        return value
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f'{what}:{bool(value)}'.encode())
    _agree(digest, f'Tensor-parallel workers disagree: {what}')
    return value


def agree_layer_selection(request_id, layer, records, kind):
    """Layerwise TP: every rank picked the same logical blocks for this layer.

    Picks are made from rank-identical (reduced) scores with position tie
    breaks, so they agree by construction; this fails loudly before any rank
    writes a diverged window instead of letting the ranks drift apart.
    """
    if _group is None:
        return
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f'{kind}:{request_id}:{layer}'.encode())
    for record in sorted(records, key=lambda r: r.history_begin):
        digest.update(struct.pack('<QQ', record.history_begin, record.count))
    _agree(digest, f'Tensor-parallel workers chose different {kind} blocks for {request_id} layer {layer}')


def verify_window(authority, remaps, by_slot):
    """Compare logical token intervals before either rank overwrites its KV."""
    if _group is None:
        return
    request_id, history = authority.memory.request_id, authority.history_end
    digest = hashlib.blake2b(digest_size=16)
    digest.update(request_id.encode())
    digest.update(struct.pack('<QQ', history, len(remaps)))
    for remap in sorted(remaps, key=lambda r: r.to_base):
        digest.update(struct.pack('<QQQ', by_slot[remap.block_id].history_begin,
                                  remap.n_tokens, remap.to_base))
    _agree(digest, f'Tensor-parallel workers selected different history for {request_id}')


# Historical GC under TP: the leader decides, every rank commits at one step.
#
# Each rank archives its own KV-head shard under rank-local record handles,
# while every cross-rank selection check (verify_query_domain, selected_slots,
# agree_layer_selection, tp_feedback notices) needs the live logical
# (history_begin, count) record set to be identical on all ranks. GC running
# independently per rank would retire different blocks at different times. So:
#   * rank 0's native worker runs its policy in PROPOSE mode: it builds the
#     batch of intents but never commits it on its own;
#   * ranks >= 1 run in FOLLOW mode: release jobs only, never a policy round;
#   * once per engine step gc_sync broadcasts rank 0's pending intents as
#     logical blocks, every rank checks them locally without committing (live
#     exactly once, sealed, not current window/recent/held/mandatory, no busy
#     reader), a MIN all-reduce decides, and every rank retires exactly the
#     agreed blocks, in the same order, and drops them from its live index
#     and per-layer metadata in that same call. CPU payload/slot release then
#     completes asynchronously per rank (HistoricalAuthorityGc.drain).
# Rank 0's policy sees only rank 0's half of the heads (Mean-K features, the
# captured Query, importance and cosine groups are per local KV head): an
# approximation of the TP=1 policy, applied identically to every rank.
_gc_authorities = weakref.WeakValueDictionary()
_gc_lock = threading.Lock()    # Authorities may be created off the main thread.


def register_gc_authority(authority):
    """Every authority with native historical GC takes part in gc_sync."""
    if _group is not None:
        with _gc_lock:
            _gc_authorities[authority.memory.request_id] = authority


def configure_history_gc(archive):
    """Choose this rank's GC role (rank 0 proposes, others follow) before any scope registers."""
    worker = getattr(archive, '_history_worker', None)
    if _group is None or worker is None:
        return None
    from .. import _native
    mode = (_native.HistoricalGcCommit.PROPOSE if _group.rank_in_group == 0
            else _native.HistoricalGcCommit.FOLLOW)
    worker.set_commit(mode)
    return mode


def _gc_open(authority):
    gc = getattr(authority, 'gc', None)
    return (authority is not None and getattr(gc, 'native_history', False) and not gc.stopped and
            not getattr(authority, 'closing', True) and getattr(authority, '_rewind_position', None) is None)


def _gc_busy(runtime, refresh, sessions, authorities):
    """ids of authorities whose records another thread may read for a selection now.

    Refresh jobs select and assemble windows on the private TP lane; a session
    restore may import any parked authority; a checkpoint that is not live or
    idle is being retired, claimed or closed. A veto on any rank skips the
    intent on every rank (MIN), so these rank-local views need not agree.
    """
    active = dict(getattr(runtime, 'authorities', None) or {})
    jobs = [job for job in (getattr(refresh, 'jobs', None) or {}).values() if not getattr(job, 'done', False)]
    current = getattr(refresh, 'tp_transfer', None)
    if current is not None and not getattr(current, 'done', False):
        jobs.append(current)
    busy, restoring = set(), False
    if jobs:
        from .connector import RestoreCommand
        for job in jobs:
            authority = active.get(job.command.request_id)
            if authority is not None:
                busy.add(id(authority))
            restoring = restoring or isinstance(job.command, RestoreCommand)
    if restoring:
        live = {id(authority) for authority in active.values()}
        busy.update(id(authority) for authority in authorities.values() if id(authority) not in live)
    pool = getattr(sessions, 'pool', None)
    for frames in (getattr(pool, 'entries', None), getattr(pool, 'prompts', None)):
        for checkpoint in tuple((frames or {}).values()):
            authority = getattr(checkpoint, 'authority', None)
            phase = getattr(getattr(checkpoint, 'phase', None), 'name', 'IDLE')
            if authority is not None and phase not in ('LIVE', 'IDLE'):
                busy.add(id(authority))
    return busy


def _gc_index(authority):
    """(published history end, {history_begin: record, or False if two records start there})."""
    end = authority.history_end    # Read first: a record is listed before history covers it.
    index = {}
    for record in tuple(authority.records.values()):
        index[record.history_begin] = False if record.history_begin in index else record
    return end, index


def _gc_vote(index, end, begin, count, expected=None):
    """1: retirable here; 0: not now (lagging publication); -1: the ranks diverged."""
    record = index.get(begin)
    if record is None:
        return (-1 if begin + count <= end else 0), None
    if record is False:
        return -1, None
    if record.count != count:
        lagging = record.count < count and not record.archived and begin + count > end
        return (0 if lagging else -1), None
    if not record.archived:
        return 0, None
    handle = record.handle
    if expected is not None and (handle.slot, handle.generation) != expected:
        return -1, None
    return 1, handle


def gc_sync(runtime=None, refresh=None, sessions=None):
    """Retire rank 0's pending historical-GC intents on every rank, at once.

    Called once per engine step on every rank at the same point of the main
    thread (runner._kvmem_execute_model, right after advance_publication and
    its pump_refresh), so all ranks reach it in the same collective order. An
    empty step costs one small broadcast. Returns this rank's
    [(request_id, [(history_begin, slot, generation), ...])].
    """
    if _group is None:
        return []
    import json
    import torch
    import torch.distributed as dist
    group = _cpu_group()
    leader = _group.rank_in_group == 0
    src = _group.ranks[0]
    with _gc_lock:
        authorities = dict(_gc_authorities)
    busy = _gc_busy(runtime, refresh, sessions, authorities)
    taken, payload = {}, b''
    if leader:
        rows = []
        for rid in sorted(authorities):
            authority = authorities[rid]
            if not _gc_open(authority) or id(authority) in busy:
                continue    # The batch stays pending and is proposed once the owner is free.
            scope = authority.gc.scope
            if not scope.external_proposal_ready:
                continue
            handles = scope.external_proposals()
            if not handles:
                continue
            mine = []
            for handle in handles:
                record = authority.records.get((handle.slot, handle.generation))
                if record is not None:    # An unlisted intent resolves as refused.
                    mine.append((record.history_begin, record.count, (handle.slot, handle.generation)))
            taken[rid] = mine
            rows.append([rid, [[begin, count] for begin, count, _ in mine]])
        if rows:
            payload = json.dumps(rows, separators=(',', ':')).encode()
    size = torch.tensor([len(payload)], dtype=torch.int64)
    dist.broadcast(size, src, group=group)
    if not size.item():
        return []
    data = (torch.frombuffer(bytearray(payload), dtype=torch.uint8) if leader
            else torch.empty(int(size.item()), dtype=torch.uint8))
    dist.broadcast(data, src, group=group)
    proposals = json.loads((payload if leader else data.numpy().tobytes()).decode())

    # Local checks, without committing. Prepared records keep their policy
    # gates (no new holds) and their scope acknowledges no pause until release.
    plans, votes, error = [], [0] * sum(len(rows) for _, rows in proposals), None
    try:
        position = 0
        for rid, rows in proposals:
            authority = authorities.get(rid)
            usable = _gc_open(authority) and id(authority) not in busy
            mine = taken.get(rid)
            local = [[0, None] for _ in rows]
            if usable:
                end, index = _gc_index(authority)
                for i, (begin, count) in enumerate(rows):
                    local[i] = list(_gc_vote(index, end, begin, count,
                                             mine[i][2] if mine is not None else None))
            handles = [handle for vote, handle in local if vote == 1]
            prepared = bool(handles)
            # Registered before the native call so a failure still releases it.
            plans.append((rid, authority if prepared or rid in taken else None, local))
            if prepared:
                ready = iter(authority.gc.scope.external_prepare(handles))
                for row in local:
                    if row[0] == 1 and not next(ready):
                        row[0] = 0
            votes[position:position + len(local)] = [vote for vote, _ in local]
            position += len(local)
    except Exception as exc:    # Fail every rank together, not only this one.
        error = exc
    decision = torch.tensor([0 if error is not None else 1] + votes, dtype=torch.int32)
    dist.all_reduce(decision, op=dist.ReduceOp.MIN, group=group)
    status, agreed = decision[0].item(), decision[1:].tolist()

    aborted = status < 1 or any(vote < 0 for vote in agreed)
    committed, failure = [], None
    try:
        position = 0
        for rid, owner, local in () if aborted else plans:
            decided = agreed[position:position + len(local)]
            position += len(local)
            allowed = [handle for (vote, handle), ok in zip(local, decided) if ok == 1]
            if owner is None:
                continue
            # The leader also resolves a batch nobody agreed on.
            results = owner.gc.scope.external_commit(allowed)
            keys = [(handle.slot, handle.generation) for handle, ok in zip(allowed, results) if ok]
            if len(keys) != len(allowed):
                failure = failure or RuntimeError(f'Historical GC could not retire agreed blocks of {rid}')
            if keys:
                # Before release: no CPU completion of these slots exists yet.
                committed.append((rid, owner, owner.retire_synchronized(keys)))
    except Exception as exc:
        failure = exc
    finally:
        for _, owner, _ in plans:
            if owner is not None:
                owner.gc.scope.external_release()
    if aborted:
        raise RuntimeError('A historical GC block is missing, ambiguous or different on a tensor-parallel worker'
                           if status == 1 else
                           'A tensor-parallel worker failed to check the historical GC intents') from error
    if any(vote == 1 for vote in agreed):
        done = torch.tensor([0 if failure is not None else 1], dtype=torch.int32)
        dist.all_reduce(done, op=dist.ReduceOp.MIN, group=group)
        if not done.item():
            raise RuntimeError('Tensor-parallel workers could not retire the agreed historical GC blocks') from failure
    elif failure is not None:
        raise failure
    for _, owner, retired in committed:
        owner.finish_synchronized_retirements(retired)
    return [(rid, retired) for rid, _, retired in committed]
