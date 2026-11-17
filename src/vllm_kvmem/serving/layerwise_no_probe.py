"""Public-forward-only Query admission and final layer-zero/MTP publication.

Admission fixes geometry, not a scored global KV set. No target state is borrowed
and no historical GPU bytes are touched until the consuming layer produces Q.
"""
import json
import time

import torch


def window_geometry(records, fixed, budget, block_tokens=32):
    """Derive native lengths without Q or a speculative selection.

    An optional partial block makes the old global selection's token extent
    score-dependent. Reject that case rather than silently changing RoPE or the
    baseline's partial-block policy. Fixed partial blocks and fitting histories
    have an unambiguous extent and are supported.
    """
    slots = {r.handle.slot for r in records}
    fixed = set(fixed)
    if budget <= 0 or budget % block_tokens or not fixed <= slots:
        raise ValueError('Invalid no-probe Query window budget or fixed identities')
    if any(not 0 < r.count <= block_tokens for r in records):
        raise ValueError('Invalid no-probe Query record extent')
    blocks = min(len(records), budget // block_tokens)
    if len(fixed) > blocks or not blocks:
        raise ValueError('No-probe Query cannot fit its mandatory blocks')
    if len(records) <= blocks:
        fixed = slots
    optional_partial = [r for r in records if r.count != block_tokens and r.handle.slot not in fixed]
    if optional_partial:
        raise ValueError('No-probe Query requires partial blocks to be fixed; '
                         'optional partial selection would change native positions')
    tokens = blocks * block_tokens - sum(block_tokens-r.count for r in records if r.handle.slot in fixed)
    return tuple(sorted(fixed)), tokens


def validate_options(options):
    if not options.async_refresh:
        raise ValueError('No-probe Query requires asynchronous request admission')
    unsupported = ('guided_query_tokens', 'query_history', 'repository_gc', 'historical_gc')
    if any(getattr(options, key, False) for key in unsupported):
        raise ValueError('No-probe Query requires retained history without guidance or Query-history fusion')
    if getattr(options, 'session_capacity', 0) and not getattr(options, 'session_freeze_prompt', False):
        raise ValueError('No-probe Query sessions require frozen P checkpoints')


def arm(controller, refresh, job):
    """Acknowledge geometry to the scheduler; actual KV is built during Query."""
    validate_options(refresh.options)
    command = job.command
    if command.query_span is None or job.guided_pressure or command.guided_private:
        raise ValueError('No-probe Query requires an explicit public Query span')
    if job.state is not None or job.steps or job.query_rows or job.guided_rows:
        raise RuntimeError('No-probe Query was reached after private execution')
    with torch.inference_mode():
        controller.prepare(refresh, job, None)
    state = controller.pending.pop(command.request_id)
    state['fixed'], state['selected_tokens'] = window_geometry(
        state['records'], state['fixed'], state['budget'])
    state['partial_excluded'] = set()
    state['global_slots'] = None
    state['no_probe'] = True
    state['refresh_job'] = job
    state['refresh'] = refresh
    state['request_generation'] = refresh.runner._kvmem_request_generations[command.request_id]
    job.selected_tokens = state['selected_tokens']
    # Query rows keep the existing exclusion from cumulative prefill pressure.
    from .prefill_attention import exclude_query_replay
    exclude_query_replay(refresh.runner, job.authority, command.request_id, job.history,
                         job.history + len(command.input_tokens), True)
    refresh.runner._kvmem_window_adapter.commit(command.request_id, job.history, job.selected_tokens)
    if controller.native_batches:
        controller.active_owners[command.request_id] = state
    else:
        controller.active = state
    controller.emit('armed', request_id=command.request_id, layers=len(controller.names),
                    budget=state['budget'], selected_tokens=job.selected_tokens,
                    query_span=state['span'], no_probe=True,
                    actual_method='mean_k_sequential_layerwise', mtp_selection=controller.mtp_selection)
    job.done = True
    job.stage = 'public_query'
    refresh.emit(job, 'no_probe_armed', selected_tokens=job.selected_tokens,
                 private_forward_steps=0, global_assembly_bytes=0)


def commit_layer_zero(controller, state):
    """Publish shared bookkeeping only after all target/MTP writes have fenced.

    The immutable archive retains every layer's sources (GC is disallowed).
    Layer-zero identities provide the canonical append frame; other layers keep
    their own explicit maps. No second GPU assembly accompanies this commit.
    """
    from ..core.memory import Phase
    memory = state['authority'].memory
    chosen = state['layer_zero_records']
    memory._require(Phase.READY)
    def admit():
        return memory._store.select([r.handle for r in chosen])
    if getattr(memory, 'tp_records', None) is not None:
        # TP: the same logical frame on every rank, admitted jointly (a retry
        # or failure on one rank is seen by all before any KV is advertised).
        from .tensor_parallel import agree_layer_selection
        agree_layer_selection(state.get('request_id', memory.request_id), 0, chosen, 'layer-zero frame')
        plan = memory.tp_admit_window(admit)
    else:
        plan = admit()
    memory._plan = plan
    memory.phase = Phase.ASSEMBLING
    memory._submitted = False
    try:
        expected = [(r.handle.slot, r.count) for r in chosen]
        if ([(r.block_id, r.n_tokens) for r in plan.remaps] != expected or
                plan.total_window_tokens != state['selected_tokens']):
            raise RuntimeError('No-probe canonical frame differs from the assembled layer zero')
        event = torch.cuda.Event()
        event.record()
        memory.attach_completion(event)
        event.synchronize()
        if not memory.commit_selection():
            raise RuntimeError('No-probe canonical frame did not commit after its fence')
    except BaseException:
        # The worker must terminate, retaining this frame; partially written KV
        # must never be advertised as the old ready window.
        memory.phase = Phase.FAILED
        raise


def report_completed(controller, state, mtp_stats):
    refresh, job = state['refresh'], state['refresh_job']
    sessions = getattr(controller.runner, '_kvmem_sessions', None)
    if sessions is not None:
        sessions.selection_committed(state['request_id'], state['history'])
    row = dict(request_id=state['request_id'], command_id=job.command.command_id,
        selection='query', no_probe=True, actual_method='mean_k_sequential_layerwise',
        history_begin=state['history'], query_span=state['span'],
        query_tokens=state['query_tokens'], selected_tokens=state['selected_tokens'],
        selected_history_ranges=[[r.history_begin,r.count] for r in state['layer_zero_records']],
        shared_frame='layer0', mtp_selection=controller.mtp_selection,
        private_forward_steps=0, query_execution_counts={}, state_restore_count=0,
        global_assembly_bytes=0, mtp_assembly=mtp_stats,
        refresh_started_at=job.started, refresh_finished_at=time.perf_counter(),
        timing_scope='Admission through final public Query layer selection, not whole public forward')
    with (controller.runner._kvmem_probe_output/'bounded_refresh.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    refresh.emit(job, 'no_probe_completed', private_forward_steps=0, global_assembly_bytes=0)
