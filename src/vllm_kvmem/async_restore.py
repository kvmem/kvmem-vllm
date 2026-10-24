"""Exact checkpoint restore with separate bounded resident and assembly lanes.

Native pages and request indices stay leased until completion, even on cancel.
The engine alone mutates checkpoint/session/authority maps; the producer only
uses borrowed buffers, immutable CPU history and its private CUDA stream.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import json
import time

import torch


@dataclass
class RestoreContext:
    checkpoint: object
    plan: object
    assembler: object
    model: object
    ready: object
    stream: object = None


def resident_lane(job):
    """Resident target pages need only this request's recurrent state upload."""
    return bool(getattr(job.command, 'resident', False))


def lane_busy(manager, *, resident=False):
    # A completed producer future may still own GPU destinations. Keep each
    # lane occupied through the main-thread commit, not just kernel enqueue.
    limit = getattr(manager.options, 'resident_restore_slots', 1) if resident else 1
    return sum(other.future is not None and not other.done and
               resident_lane(other) == resident for other in manager.jobs.values()) >= limit


def copy_restore(manager, job):
    context = job.restore_context
    worker = manager.runner._kvmem_sessions
    stream = context.stream
    with torch.inference_mode(), torch.cuda.device(manager.runner.device), torch.cuda.stream(stream):
        stream.wait_event(context.ready)
        assembly = context.assembler.assemble(job.authority, context.plan) if context.assembler is not None else None
        # Pinned native state code writes only this request's leased state rows
        # and acceptance counter; public batches do not contain this request.
        model = worker.model_adapter.submit_restore(context.model)
        complete = torch.cuda.Event(); complete.record()
        if assembly is None:
            from .window_assembly import AssemblyResult
            assembly = AssemblyResult(complete, (job.authority, context.plan), 0,
                                      job.command.window_tokens, 0)
        return assembly, model, complete


def validate_pages(manager, command):
    runner = manager.runner
    lease = runner._kvmem_restore_allocations.get(command.request_id)
    if lease is None:
        raise ValueError('Restore has no fresh native allocation witness')
    lease.validate(runner, expected_block_ids=command.block_ids)
    return lease


def _stage(timing, name):
    return timing.stage(name) if timing is not None else nullcontext()


def step_restore(manager, job):
    if not getattr(getattr(manager, 'options', None), 'restore_stage_timing', False):
        return _step_restore(manager, job)
    from .restore_timing import RestoreTiming
    timing = RestoreTiming(enabled=True)
    phase, outcome = job.stage, 'error'
    try:
        with timing.stage('restore_main_thread_total'):
            result = _step_restore(manager, job, timing=timing)
        outcome = 'progress' if result else 'pending'
        return result
    finally:
        summary = timing.summary()
        # Polling an unready lane/fence is request-local and nonblocking. Do not
        # turn every such poll into synchronous diagnostic file IO.
        if outcome == 'error' or any(stage['name'] != 'restore_main_thread_total' for stage in summary['stages']):
            row = dict(request_id=job.command.request_id, phase=phase,
                outcome=outcome, lane='resident' if resident_lane(job) else 'assembly',
                history_tokens=job.command.history_end, active_tokens=job.command.window_tokens,
                timing=summary,
                scope='Host intervals on the engine thread; nested stages overlap. '
                      'Wall minus thread CPU time includes descheduling, GIL and native waits, '
                      'not an attribution to GPU idle. Background transfer and diagnostic write time are excluded.')
            try:
                with (manager.runner._kvmem_probe_output/'restore_host_stages.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            except OSError:
                if outcome != 'error':
                    raise
                # Diagnostic storage failure must not hide an existing restore
                # invariant violation (its original traceback still propagates).


def _step_restore(manager, job, *, timing=None):
    worker = manager.runner._kvmem_sessions
    rid, command = job.command.request_id, job.command
    if job.stage in ('prepare', 'restore_model_prepare'):
        if rid not in manager.runner.req_states.req_id_to_index:
            return False  # First control packet precedes native row initialization.
        binding = worker.bindings[rid]
        pressure = getattr(binding, 'pressure_restore', False)
        if (binding.generation, binding.history, binding.window) != (
                command.generation, command.history_end, command.window_tokens) or rid not in worker.imports:
            raise ValueError('Restore command differs from exact checkpoint binding')
        resident = resident_lane(job)
        if lane_busy(manager, resident=resident):
            return False
        with _stage(timing, 'allocation_lease_validate'):
            lease = validate_pages(manager, command)
        if job.checkpoint is None:
            with _stage(timing, 'checkpoint_claim'):
                job.checkpoint = worker.pool.claim(binding.namespace, rid, binding.tokens,
                    position=binding.history, **({'timing': timing} if timing is not None else {}))
            if (job.checkpoint is None or
                    (job.checkpoint.window.window_tokens != binding.window and not pressure)):
                raise ValueError('Checkpoint restore lost its original active view')
            if command.resident and pressure:
                raise ValueError('Pressure restore cannot reuse a resident page frame')
            if command.resident and (job.checkpoint.kind != 'M' or job.checkpoint.position != command.history_end):
                raise ValueError('Resident import requires the exact terminal checkpoint')
            job.authority = job.checkpoint.authority
            job.model_bytes = job.checkpoint.model.bytes
        if job.stage == 'restore_model_prepare':
            plan = job.restore_plan
            if plan is None or job.checkpoint is None:
                raise RuntimeError('Split restore lost its prepared frame')
        else:
            with _stage(timing, 'prepare_claim_window'):
                plan = _prepare_window(manager, worker, job, binding, pressure, timing)
            if plan is None:
                return False  # P rewind waits only this request, never the public batch.
            if pressure and plan.total_window_tokens != binding.window:
                raise ValueError('Pressure restore differs from its reserved window')
            if getattr(manager.options, 'restore_prepare_split', False):
                job.restore_plan = plan
                job.stage = 'restore_model_prepare'
                return True  # Let the cooperative time budget yield to public work.
        from .session_state import PromptStateAdapter
        from .window_assembly import NativeWindowAssembler
        with _stage(timing, 'model_state_prepare'):
            if worker.model_adapter is None:
                worker.model_adapter = PromptStateAdapter(manager.runner)
            model = worker.model_adapter.prepare_restore(job.checkpoint.model, rid, lease=lease)
        with _stage(timing, 'assembly_adapter_prepare'):
            assembler = None if command.resident else NativeWindowAssembler.from_runner(manager.runner, rid, manager.runtime.names,
                manager.options.memory.block_tokens, packed=manager.options.packed_assembly,
                direct_pinned_reads=manager.options.direct_pinned_reads, arena='refresh', lease=lease)
        if assembler is not None:
            assembler.bandwidth_limiter = manager.maintenance.transfer_budget
        with _stage(timing, 'prepare_stream_dependencies'):
            ready = torch.cuda.Event(); ready.record()
            stream_name = 'resident_stream' if resident else 'stream'
            if getattr(manager, stream_name, None) is None:
                setattr(manager, stream_name, torch.cuda.Stream(device=manager.runner.device, priority=0))
            stream = getattr(manager, stream_name)
        with _stage(timing, 'transfer_credit_acquire'):
            if not manager.maintenance.acquire(job):
                raise RuntimeError('Restore lane lost its transfer reservation')
        job.restore_context = RestoreContext(job.checkpoint, plan, assembler, model, ready, stream)
        job.restore_plan = None
        job.stage = 'restore_copy'
        job.authority._refresh_background = True
        job.service_started = time.perf_counter()
        executor = manager.resident_executor if resident else manager.executor
        with _stage(timing, 'background_submit'):
            job.future = executor.submit(copy_restore, manager, job)
        manager.emit(job, 'restore_submitted', model_bytes=job.model_bytes,
                     lane='resident' if resident else 'assembly')
        return True
    return _finish_restore(manager, job, worker, rid, command, timing)


def _prepare_window(manager, worker, job, binding, pressure, timing):
    pressure_kwargs = {}
    if pressure:
        block = manager.options.memory.block_tokens
        pressure_kwargs = dict(pressure=True,
            pressure_budget=(binding.window + block - 1) // block * block,
            mandatory_spans=binding.mandatory_spans)
    return worker.pool.prepare_claim_window(job.checkpoint, wait=False,
        metadata_binding=binding, **pressure_kwargs,
        **({'timing': timing} if timing is not None else {}))


def _finish_restore(manager, job, worker, rid, command, timing):
    if job.stage != 'restore_copy':
        raise RuntimeError('Unknown checkpoint restore phase')
    if not job.future.done():
        return False
    if job.result is None:
        job.result = job.future.result()
    assembly, model, complete = job.result
    if not complete.query():
        return False
    context = job.restore_context
    with _stage(timing, 'model_state_validate'):
        worker.model_adapter.validate_restore(context.model)
    with _stage(timing, 'commit_selection'):
        job.authority.memory.attach_completion(assembly)
        if not job.authority.memory.commit_selection():
            raise RuntimeError('Completed restore failed to commit its exact active view')
    with _stage(timing, 'restore_byte_oracle'):
        worker._verify_restored(rid, job.checkpoint, command.resident)
    # Preserve the imported checkpoint until a newer capture replaces it. A
    # cancelled import can then retire the same CPU history without recompute.
    with _stage(timing, 'finish_claim'):
        authority = worker.pool.finish_claim(job.checkpoint, complete, retain=True)
    if authority is not job.authority:
        raise RuntimeError('Restored authority identity changed')
    with _stage(timing, 'publish_restore'):
        worker.publish_restore(rid, authority, retained=True)
    worker.imports.remove(rid)
    authority._refresh_background = False
    if context.assembler is not None and not context.assembler.close():
        raise RuntimeError('Completed restore still owns assembly scratch')
    manager.runner._kvmem_restore_allocations.pop(rid)
    cache_hit = getattr(context.model, 'cached_payload', None) is not None
    model_uploaded = 0 if cache_hit else job.model_bytes
    manager.maintenance.observe(job, assembly.uploaded_bytes + model_uploaded)
    manager.maintenance.release(job)
    if getattr(manager.maintenance, 'offline', False):
        manager.maintenance.restore_timings.append(dict(request_id=rid,
            active_tokens=command.window_tokens, resident=command.resident,
            uploaded_bytes=assembly.uploaded_bytes, model_bytes=job.model_bytes,
            service_seconds=time.perf_counter() - job.service_started))
    row = dict(request_id=rid, history_tokens=command.history_end, active_tokens=command.window_tokens,
        started=job.started, finished=time.perf_counter(), uploaded_bytes=assembly.uploaded_bytes,
        model_bytes=job.model_bytes, host_prepare_seconds=assembly.host_prepare_seconds,
        model_uploaded_bytes=model_uploaded, model_cache_hit=cache_hit,
        model_snapshot_cache=(worker.model_adapter.snapshot_cache.stats()
            if getattr(worker.model_adapter, 'snapshot_cache', None) is not None else None),
        resident_active_tokens=command.window_tokens if command.resident else 0,
        split_tokens=getattr(assembly, 'split_tokens', 0),
        partial_tokens=getattr(assembly, 'partial_tokens', 0),
        raw_cache_hit_tokens=getattr(assembly, 'raw_cache_hit_tokens', 0),
        resident_v_reused_tokens=getattr(assembly, 'resident_v_reused_tokens', 0),
        public_peer_count=len(job.peers), cancelled=job.cancelled,
        lane='resident' if resident_lane(job) else 'assembly')
    with (manager.runner._kvmem_probe_output/'async_restore.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    job.restore_context = job.checkpoint = job.future = job.result = None
    job.done = True
    manager.emit(job, 'restore_ready', cancelled=job.cancelled)
    return True
