"""Request-local refresh: cooperative private forwards and background assembly.

The scheduler leases the original pages and removes only this request from public
batches. Private model work runs on the engine thread in bounded quanta; CPU
history/MeanK/assembly runs on one separate producer and CUDA stream. Public Q,
GDN state and graph capture buffers are never used concurrently by two forwards.
Resident imports have one independent producer/stream and touch only their own
leased recurrent state, so a cold assembly cannot block their DMA submission.
"""
from ..core.native_pages import stored_expansions
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import os
import threading
import time

import numpy as np
import torch
from ..core import settings as _settings


def maintenance_batch(runner, request_id, tokens, begin):
    from vllm.v1.worker.gpu.input_batch import InputBatch
    device=runner.device; n=len(tokens)
    index=runner.req_states.req_id_to_index[request_id]
    idx=torch.tensor([index],device=device,dtype=torch.int32)
    starts=np.array([0,n],dtype=np.int32)
    return InputBatch(req_ids=[request_id],num_reqs=1,num_reqs_after_padding=1,
        idx_mapping=idx,idx_mapping_np=np.array([index],dtype=np.int32),
        expanded_idx_mapping=idx,expanded_local_pos=torch.zeros(1,device=device,dtype=torch.int32),
        num_scheduled_tokens=np.array([n],dtype=np.int32),num_tokens=n,num_tokens_after_padding=n,
        num_draft_tokens=0,num_draft_tokens_per_req=None,
        query_start_loc=torch.tensor(starts,device=device),query_start_loc_np=starts,
        seq_lens=torch.tensor([begin+n],device=device,dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([begin+n],dtype=torch.int32),dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([begin],dtype=np.int32),prefill_len_np=np.array([begin+n],dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([begin],dtype=np.int32),
        is_prefilling_np=np.ones(1,dtype=np.bool_),has_prefill=True,
        input_ids=torch.tensor(tokens,device=device,dtype=torch.int32),
        positions=torch.arange(begin,begin+n,device=device,dtype=torch.int64),
        is_padding=torch.zeros(n,device=device,dtype=torch.bool),
        logits_indices=torch.tensor([n-1],device=device,dtype=torch.int64),
        cu_num_logits=torch.tensor([0,1],device=device,dtype=torch.int32),
        cu_num_logits_np=np.array([0,1],dtype=np.int32),has_structured_output_reqs=False,
        prompt_lens=None,max_query_len=n)


@dataclass
class RefreshJob:
    command: object
    stage: str = 'prepare'
    done: bool = False
    cancelled: bool = False
    dispatched: bool = False
    selected_tokens: int | None = None
    future: object = None
    assembly_inputs: object = None
    result: object = None
    authority: object = None
    state: object = None
    checkpoint: object = None
    restore_event: object = None
    offset: int = 0
    query_rows: list = field(default_factory=list)
    guided_rows: list = field(default_factory=list)
    generated: list = field(default_factory=list)
    guided_offset: int = 0
    next_token: int | None = None
    guided_fallback: str | None = None
    guided_policy: object = None
    guided_diagnostics: dict = field(default_factory=dict)
    guided_seconds: float = 0.0
    started: float = field(default_factory=time.perf_counter)
    peers: set = field(default_factory=set)
    steps: int = 0
    last_progress: float | None = None
    restore_context: object = None
    restore_plan: object = None
    model_bytes: int = 0
    service_started: float = 0.0
    pressure_oracle: object = None
    graph_binding: object = None
    query_execution_counts: dict = field(default_factory=dict)
    guided_witness: tuple | None = None
    guided_state_verified: bool = False

    @property
    def guided_pressure(self):
        return self.command.selection == 'pressure' and self.command.guided_reason == 'prefill-pressure'

    @property
    def guided_base(self):
        return self.history if self.guided_pressure else self.history + len(self.command.input_tokens)

    def query(self):return self.done

    @property
    def history(self):
        from .connector import RestoreCommand
        return self.command.history_end if isinstance(self.command, RestoreCommand) else self.command.groups[0][1].history_end


class AsyncRefresh:
    def _ensure_window_adapter(self, command):
        if getattr(self.runner, '_kvmem_window_adapter', None) is None:
            from .window_batch import WindowBatchAdapter
            memory = self.options.memory
            self.runner._kvmem_window_adapter = WindowBatchAdapter(
                tuple(group for group, _ in command.groups), memory.select_budget+memory.gen_budget)

    # Opt-in prefill-pressure lanes (KVMEM_LAYERWISE_PRESSURE_LANES). Lane 0
    # is the original executor/stream/staging; other jobs are unchanged. A
    # worker thread selects its lane's stream through this thread-local index.
    @property
    def stream(self):
        streams = self.__dict__.setdefault('_lane_streams', {})
        return streams.get(self._current_lane())

    @stream.setter
    def stream(self, value):
        self.__dict__.setdefault('_lane_streams', {})[self._current_lane()] = value

    def _current_lane(self):
        local = self.__dict__.get('_lane_local')
        return getattr(local, 'index', 0) if local is not None else 0

    def _pressure_lanes(self):
        return getattr(getattr(self.runner, '_kvmem_layerwise_pressure', None), 'lanes', 1)

    def _multi_lane(self, job):
        from .connector import RestoreCommand
        return (self._pressure_lanes() > 1 and not isinstance(job.command, RestoreCommand) and
                job.command.selection == 'pressure' and
                getattr(job.command, 'pressure_stage', None) == 'prefill' and
                not job.guided_pressure)

    def _producers(self):
        """Jobs whose producer may still own scratch, including dispatched pipelines."""
        from .async_restore import resident_lane
        from .pressure_pipeline import pipeline_for
        busy = {id(job): job for job in self.jobs.values()
                if job.future is not None and not job.done and not resident_lane(job)}
        pipeline = pipeline_for(self)
        if pipeline is not None:
            busy.update((id(lease.job), lease.job) for lease in pipeline.active.values()
                        if lease.job is not None)
        return list(busy.values())

    def _claim_lane(self, job):
        """Free lane index for a multi-lane pressure job, or None."""
        lanes = self._pressure_lanes()
        producers = self._producers()
        if len(producers) >= lanes or not all(self._multi_lane(other) for other in producers):
            return None
        used = {getattr(other, 'lane', 0) for other in producers}
        return next(i for i in range(lanes) if i not in used)

    def _executor_for(self, lane):
        if lane == 0:
            return self.executor
        executors = self.__dict__.setdefault('_lane_executors', {})
        if lane not in executors:
            executors[lane] = ThreadPoolExecutor(max_workers=1,
                thread_name_prefix=f'kvmem-refresh-lane{lane}')
        return executors[lane]

    def _assemble_in_lane(self, lane, job, queries, ready):
        local = self.__dict__.setdefault('_lane_local', threading.local())
        local.index = lane
        try:
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=self.runner.device, priority=0)
            return self._assemble(job, queries, ready)
        finally:
            local.index = 0

    def __init__(self, runner, options):
        self.runner,self.options=runner,options
        self.original=runner.model.forward
        self.jobs={};self.cursor=0;self.deferred={}
        from .priority_executor import PriorityExecutor, restore_first
        # Opt-in KVMEM_RESTORE_FIRST: same single producer, queued swap-ins first.
        self.executor=(PriorityExecutor('kvmem-refresh') if restore_first() else
                       ThreadPoolExecutor(max_workers=1,thread_name_prefix='kvmem-refresh'))
        # Resident imports never use the shared assembly scratch. One separate
        # producer and fence-owned credit prevent a long cold KV upload from
        # stopping a ready hot session, without an unbounded DMA queue.
        self.resident_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='kvmem-resident')
        self.resident_stream=None;self.closed=False
        self.stream=None;self.adapter=None;self.tokenizer=None
        self.trace_lock=threading.Lock()
        # Qualification-only switch: retain the old handoff until matched MAB
        # measurements establish a benefit with identical selected KV/output.
        self.query_post_forward_reap = False
        self.maintenance = None
        if getattr(options, 'query_replay_backend', 'eager') == 'compiled' and not getattr(
                runner, '_kvmem_native_compilation', False):
            raise ValueError('Compiled Query replay requires the qualified native compiled model')
        self.cobatch = None
        if getattr(options, 'query_replay_cobatch', False):
            from .query_replay_batch import QueryReplayCoBatch
            self.cobatch = QueryReplayCoBatch(self)

    def initialize_cache(self):
        # load_model installs this adapter before native KV geometry exists.
        # Materialize byte budgets only after the native cache has been created.
        if self.options.maintenance_max_steps and self.maintenance is None:
            from .maintenance import MaintenancePolicy
            from ..adapters.cache_manifest import archive_bytes_per_token
            self.maintenance = MaintenancePolicy(self.options, archive_bytes_per_token(self.runner.kv_cache_config))
        from .pressure_pipeline import pipeline_for
        pipeline = pipeline_for(self)
        if pipeline is not None:
            if getattr(self.options, 'gpu_memory_admission', 'static') != 'static':
                # Opt-in KVMEM_LAYERWISE_PIPELINE_SWAPS: incremental pages with
                # exact layerwise request swaps. layerwise_swap.ready() refuses
                # a swap until the pipeline has reaped the owner's lease (the
                # promise keeps _refresh_background set until then).
                from .pressure_pipeline import swaps_enabled
                if not (swaps_enabled() and getattr(self.options, 'request_swap_layerwise', False)):
                    raise ValueError('Pressure pipeline first qualifies unchanged static page allocation')
            pipeline.install()

    @property
    def runtime(self):return self.runner._kvmem_capture_runtime

    def emit(self, job, stage, **fields):
        row=dict(request_id=job.command.request_id,command_id=job.command.command_id,
                 stage=stage,at=time.perf_counter(),**fields)
        with self.trace_lock:
            with (self.runner._kvmem_probe_output/'async_refresh.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')

    def enqueue(self, command):
        if getattr(self, 'closed', False):
            raise RuntimeError('Maintenance producer is closed')
        from .connector import RestoreCommand
        if isinstance(command, RestoreCommand):
            if not self.options.async_restore or not command.asynchronous or not 0 < command.window_tokens <= command.history_end:
                raise ValueError('Invalid asynchronous checkpoint restore')
            job = self.jobs.get(command.request_id)
            if job is not None and (job.dispatched or job.command != command):
                raise ValueError('Duplicate request restore')
            job = self.jobs.setdefault(command.request_id, RefreshJob(command))
            job.dispatched = True
            self.emit(job, 'restore_suspended')
            return job
        padding=command.speculative_padding
        if (not command.asynchronous or type(padding) is not int or padding<0 or
                (padding and len(command.input_tokens)!=1) or
                len(command.input_tokens)+padding!=command.groups[0][1].replay_tokens):
            raise ValueError('Invalid asynchronous refresh command')
        no_replay_prefill = (_settings.get('KVMEM_PREFILL_PRESSURE_NO_REPLAY', '0') == '1' and
                             getattr(command, 'pressure_stage', None) == 'prefill')
        if not command.input_tokens and (command.selection != 'pressure' or command.query_span is not None or
                not (getattr(self.options, 'rolling_active_window', False) or no_replay_prefill)):
            raise ValueError('Empty refresh input requires rolling accepted-history pressure selection')
        if command.query_span is not None:
            history=command.groups[0][1].history_end
            qb,qe=command.query_span
            if not history<=qb<qe<=history+len(command.input_tokens):
                raise ValueError('Complete explicit Query is missing from its replay reservation')
        job=self.jobs.get(command.request_id)
        if job is not None:
            if job.dispatched or job.command != command:
                raise ValueError('Duplicate request refresh')
        else:
            job=RefreshJob(command);self.jobs[command.request_id]=job
        job.dispatched=True
        self.emit(job,'suspended')
        return job

    def defer_terminals(self, output):
        """Do not recycle request indices, authority or page leases during DMA."""
        from .connector import ControlMetadata, RefreshCommand, RestoreCommand
        control=output.kv_connector_metadata
        if not isinstance(control,ControlMetadata):return output
        if getattr(getattr(self, 'maintenance', None), 'offline', False):
            self.maintenance.readiness_costs = control.maintenance_costs
        finished=set(output.finished_req_ids)
        from .pressure_pipeline import pipeline_for
        pipeline = pipeline_for(self)
        prefixes=dict(control.terminal_prefixes)
        # Cancellation can accompany the first dispatch, before start_load_kv
        # has enqueued the job. Keep that request's native state until its ack.
        for command in control.commands:
            if (isinstance(command, (RefreshCommand, RestoreCommand)) and
                    command.asynchronous and command.request_id in (finished | control.cancelled)):
                self.jobs.setdefault(command.request_id, RefreshJob(command))
        for rid in control.cancelled:
            if rid in self.jobs:
                self.jobs[rid].cancelled=True
            if pipeline is not None and rid in pipeline.active:
                pipeline.active[rid].job.cancelled = True
        for rid in list(finished):
            job=self.jobs.get(rid)
            if (job is not None and not job.done) or (pipeline is not None and rid in pipeline.active):
                if job is not None:
                    job.cancelled=True
                self.deferred[rid]=prefixes.pop(rid,None)
                finished.remove(rid)
        for rid,prefix in list(self.deferred.items()):
            job=self.jobs.get(rid)
            if (job is None or job.done) and not (pipeline is not None and rid in pipeline.active):
                finished.add(rid)
                if prefix is not None:prefixes[rid]=prefix
                del self.deferred[rid]
        return replace(output,finished_req_ids=finished,
                       kv_connector_metadata=replace(control,terminal_prefixes=prefixes))

    def pump(self, ready_requests=()):
        from .pressure_pipeline import pipeline_for
        pipeline = pipeline_for(self)
        if pipeline is not None:
            pipeline.reap(self)
        if self.options.maintenance_max_steps and self.maintenance is None:
            raise RuntimeError('Maintenance budgets were not initialized after native KV allocation')
        for rid,job in list(self.jobs.items()):
            if job.done and rid not in self.deferred:
                del self.jobs[rid]
        jobs=list(self.jobs.values())
        if getattr(getattr(getattr(self.runner, 'vllm_config', None), 'parallel_config', None),
                   'tensor_parallel_size', 1) > 1:
            from .tensor_parallel import pump_refresh
            pump_refresh(self, ready_requests)
            return
        if getattr(self, 'cobatch', None) is not None:
            self.cobatch.begin_turn(ready_requests)
        if not jobs:return
        if getattr(self.runner,'_kvmem_pending_capture',None) is not None:
            raise RuntimeError('Private refresh would overwrite unpublished public capture')
        capture = getattr(self.runner, '_kvmem_capture_runtime', None)
        if capture is not None:
            jobs = [job for job in jobs if capture.publication_ready((job.command.request_id,))]
            if not jobs:
                return
        if getattr(self, 'maintenance', None) is not None:
            for job in jobs:
                job.peers.update(r for r in ready_requests if r != job.command.request_id)
                if job.command.request_id in ready_requests:
                    raise RuntimeError('Suspended refresh request entered a public batch')
            self.maintenance.pump(jobs, self.step, busy=bool(ready_requests))
            return
        # One runnable quantum per engine iteration, round-robin across owners.
        # Polls never wait on futures or CUDA events while neighbours can run.
        for i in range(len(jobs)):
            at=(self.cursor+i)%len(jobs);job=jobs[at]
            job.peers.update(r for r in ready_requests if r!=job.command.request_id)
            if job.command.request_id in ready_requests:
                raise RuntimeError('Suspended refresh request entered a public batch')
            if self.step(job):
                self.cursor=(at+1)%len(jobs)
                break

    def poll_restore_completions(self):
        """Reap completed imports at native post-forward, without new work.

        Only the engine thread commits. No preparation, private forward,
        future wait or CUDA synchronization is allowed here. Native metadata
        then carries both the restored session notice and transaction ACK in
        this batch, instead of waiting for the next pre-forward pump.
        """
        if not getattr(self.options, 'restore_handoff_fastpath', False):
            return
        capture = getattr(self.runner, '_kvmem_capture_runtime', None)
        for job in tuple(self.jobs.values()):
            if job.done or job.stage != 'restore_copy':
                continue
            if capture is not None and not capture.publication_ready((job.command.request_id,)):
                continue
            job.restore_completion_poll = 'post_forward'
            try:
                self.step(job)
            finally:
                job.restore_completion_poll = None

    def poll_pressure_completions(self):
        """Reap ready prefill-only single assemblies in this batch's feedback.

        Ordinarily this only checks completion. The opt-in pipeline handoff
        also primes one eligible job through the original preparation path.
        The producer retains every lane/page/transaction lease until both its
        future and final CUDA fence complete; early ACK permits guarded use.
        """
        pressure = getattr(self.runner, '_kvmem_layerwise_pressure', None)
        pipeline = getattr(pressure, 'pipeline', None)
        if pipeline is not None:
            pipeline.reap(self)
        if (pressure is None or not getattr(pressure, 'single_assembly', False) or
                not getattr(pressure, 'post_forward_reap', False)):
            return
        if pipeline is not None:
            pipeline.prime_post_forward(self)
        capture = getattr(self.runner, '_kvmem_capture_runtime', None)
        if capture is None:
            return
        for job in tuple(self.jobs.values()):
            command = job.command
            if (job.done or job.cancelled or job.stage != 'assembly' or job.future is None or
                    getattr(command, 'selection', None) != 'pressure' or
                    getattr(command, 'pressure_stage', None) != 'prefill' or
                    (pipeline is None and (not job.future.done() or
                     not getattr(job, 'layerwise_single_assembly_done', False)))):
                continue
            rid = command.request_id
            if (capture.authorities.get(rid) is not job.authority or
                    not capture.publication_ready((rid,))):
                continue
            self.step(job)
            if job.done:
                self.emit(job, 'pressure_completion_reaped', completion_poll='post_forward')

    def poll_query_completions(self):
        """Publish an already finished Query assembly at post-forward feedback.

        This only commits the original transaction. It cannot run a private
        Query, acquire another assembly lane, or bypass the CUDA completion
        fence. Final causal layerwise Query execution still happens normally.
        """
        if not getattr(self, 'query_post_forward_reap', False):
            return
        capture = getattr(self.runner, '_kvmem_capture_runtime', None)
        if capture is None:
            return
        for job in tuple(self.jobs.values()):
            command = job.command
            if (job.done or job.cancelled or job.stage != 'assembly' or
                    job.future is None or getattr(command, 'selection', None) != 'query' or
                    not job.future.done()):
                continue
            rid = command.request_id
            if (capture.authorities.get(rid) is not job.authority or
                    not capture.publication_ready((rid,))):
                continue
            self.step(job)
            if job.done:
                self.emit(job, 'query_completion_reaped', completion_poll='post_forward')

    def _forward(self, job, tokens, begin, *, next_token=False):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from ..adapters.capture import capture_to
        runner=self.runner;batch=maintenance_batch(runner,job.command.request_id,tokens,begin)
        saved=(getattr(runner,'_kvmem_last_batch',None),runner._kvmem_window_batch)
        started=time.perf_counter()
        compiled = (job.stage == 'query' and
                    getattr(self.options, 'query_replay_backend', 'eager') == 'compiled')
        try:
            tables,slots=runner.prepare_attn(batch)
            metadata=runner.model_state.prepare_attn(batch,CUDAGraphMode.NONE,tables,slots,
                                                      runner.attn_groups,runner.kv_cache_config)
            input_ids, inputs_embeds = batch.input_ids, None
            if compiled and getattr(runner, 'uses_inputs_embeds', False):
                # AOT warmup specializes the native embedding-input signature.
                # Token-ID-only private forwards cannot reuse that artifact.
                # Use the same text/MM preparation and raw-token policy as the
                # native runner, before its model-specific input overrides.
                from vllm.model_executor.models.interfaces import requires_raw_input_tokens
                inputs_embeds = runner.model_state.prepare_inputs_embeds({}, batch, runner.req_states)
                if inputs_embeds is not None and not requires_raw_input_tokens(runner.model):
                    input_ids = None
            inputs={'input_ids':input_ids,'positions':batch.positions,'inputs_embeds':inputs_embeds,
                    'intermediate_tensors':None,**runner.model_state.prepare_inputs(batch,runner.req_states)}
            captures=[]
            with set_forward_context(metadata,runner.vllm_config,num_tokens=len(tokens),
                    slot_mapping=build_slot_mappings_by_layer(slots,runner.kv_cache_config),
                    skip_compiled=not compiled,is_padding=batch.is_padding),capture_to(captures.append):
                hidden=self.original(**inputs)
            if getattr(runner,'_kvmem_graph',False):captures=runner._kvmem_graph_captures
            if tuple(c.name for c in captures)!=tuple(runner._kvmem_layers):
                raise RuntimeError('Private refresh lost target Q captures')
            q=torch.stack([c.query[:len(tokens)] for c in captures]).clone()
            token=None
            if next_token:
                hidden=hidden[0] if isinstance(hidden,tuple) else hidden
                token=int(runner.model.compute_logits(hidden[-1:]).argmax(-1).item())
            job.steps+=1
            mode = 'compiled' if compiled else 'eager'
            job.query_execution_counts[mode] = job.query_execution_counts.get(mode, 0) + 1
            self.emit(job,job.stage,tokens=len(tokens),host_seconds=time.perf_counter()-started,
                      public_peers=len(job.peers),execution_mode=mode)
            return token,q
        finally:
            runner._kvmem_last_batch,runner._kvmem_window_batch=saved

    def _restore_state(self,job):
        if job.state is not None and job.restore_event is None:
            bundle,checkpoints,before=job.state
            self.adapter.validate(bundle)
            job.restore_event=checkpoints.restore(before,bundle.tensors)
            if job.guided_pressure:
                if job.guided_witness != self.pressure_witness(job.authority):
                    raise RuntimeError('Private guided pressure changed accepted history/statistics')
                if _settings.get('KVMEM_VERIFY_GUIDED_PRESSURE') == '1':
                    from .guided_query import verify_guided_rollback
                    audit = verify_guided_rollback(before, bundle.tensors, [])
                    with (self.runner._kvmem_probe_output/'verified_guided_pressure.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(request_id=job.command.request_id,
                            history=job.history, **audit))+'\n')
                    job.guided_state_verified = True

    @staticmethod
    def pressure_witness(authority):
        action = getattr(authority, 'action_history', None)
        prefill = getattr(action, 'prefill', None)
        graph = getattr(authority, 'block_graph', None)
        return (authority.history_end, id(getattr(action, 'scores', None)),
                id(getattr(action, 'mass', None)), getattr(action, 'samples', None),
                getattr(prefill, 'end', None), getattr(prefill, 'samples', None),
                getattr(prefill, 'chunks', None), getattr(graph, 'binding', None))

    def _submit(self, job):
        self._restore_state(job)
        from .query_capture_buffer import query_for_submit
        query = query_for_submit(job)
        # The assembly stream consumes the concatenated tensor, so its fence
        # must cover cat as well as the query forwards and state restoration.
        ready=torch.cuda.Event();ready.record()
        if _settings.get('KVMEM_RETRIEVAL_DIAGNOSTICS') == '1':
            from .retrieval_diagnostics import capture_native
            capture_native(self, job, query)
        if self.stream is None:self.stream=torch.cuda.Stream(device=self.runner.device,priority=0)
        job.stage='assembly_wait'
        job.assembly_inputs=(query,ready)
        self.emit(job,'assembly_queued')
        pressure = getattr(self.runner, '_kvmem_layerwise_pressure', None)
        if (pressure is not None and getattr(pressure, 'submit_on_prepare', False) and
                getattr(pressure, 'single_assembly', False) and not job.cancelled and
                job.command.selection == 'pressure' and
                getattr(job.command, 'pressure_stage', None) == 'prefill' and
                not job.guided_pressure):
            # Submit on this prepare quantum instead of waiting for another
            # public model batch. The existing step still owns lane/byte
            # admission; CPU selection/copies stay on the original executor.
            if self.step(job):
                self.emit(job, 'pressure_assembly_submitted_inline')

    def _assemble(self,job,queries,ready):
        _entered = time.perf_counter()
        if getattr(job, 'promise_page_checks', None) and getattr(job, 'promised_lease', None) is None:
            # KVMEM_REFRESH_ASYNC_PAGE_CHECK: the deferred reservation read, before any page write.
            from .pressure_promise import verify_page_checks
            verify_page_checks(job)
        from .recall_experiment import (ACCUMULATED_ATTENTION, configured_event_method,
                                       configured_method, select_for_event)
        from .request_plan import RequestPlan, selection_config
        from .window_assembly import NativeWindowAssembler
        pressure_policy = getattr(self.options, 'pressure_policy', None) or (
            'attention' if getattr(self.options, 'action_pressure', False) else 'recency')
        cfg=selection_config(self.options.memory,job.command.groups[0][1].window_tokens,
            selection=job.command.selection,pressure_policy=pressure_policy)
        authority=job.authority
        if configured_method():
            authority.recall_selection_report = None
        from .prompt_spans import selection_metadata
        graph = selection_metadata(authority)
        if graph is not None and job.graph_binding is not None:
            graph.validate_binding(job.graph_binding)
        with torch.inference_mode(),torch.cuda.device(self.runner.device),torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            self.emit(job,'assembly_started')
            if getattr(job, 'promised_lease', None) is not None:
                from ..core.publication_trace import emit as _trace
                _trace('assemble_entry', rid=job.command.request_id, seconds=time.perf_counter()-_entered)
            selection_started = time.perf_counter()
            if getattr(job, 'promised_lease', None) is None:
                authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
            # A promised refresh builds its plan from residency only (no CPU
            # payload sources in this mode); the single-assembly producer runs
            # this read barrier right before reader preparation and any read.
            published = time.perf_counter()
            if getattr(job, 'promised_lease', None) is not None:
                from ..core.publication_trace import emit as _trace
                _trace('assemble_prelude_start', rid=job.command.request_id)
            if _settings.get('KVMEM_RETRIEVAL_DIAGNOSTICS') == '1':
                from .retrieval_diagnostics import capture_archive
                capture_archive(self, job, queries)
            fit=RequestPlan(None,block_tokens=cfg.block_tokens,gen_budget=cfg.gen_budget,
                mandatory=job.command.mandatory_spans).mandatory_fit(job.history,cfg)
            required=set(fit['hard_blocks'])
            mandatory=[r.handle for r in authority.records.values() if r.history_begin//cfg.block_tokens in required]
            if {authority.records[(h.slot,h.generation)].history_begin//cfg.block_tokens for h in mandatory}!=required:
                raise ValueError('Missing required historical evidence')
            if hasattr(job, 'layerwise_decode_selection'):
                slots = job.layerwise_decode_selection
                plan = authority.memory.select({}, tuple(h.slot for h in mandatory), action=False,
                    pressure_scores=[(slot, 1.) for slot in slots], max_tokens=cfg.select_budget)
                if {r.block_id for r in plan.remaps} != set(slots):
                    raise RuntimeError('Rolling layer-zero admission differs from its frozen selection')
            elif configured_method() and job.history > cfg.select_budget:
                if job.guided_pressure or job.guided_rows:
                    raise ValueError('Recall comparison prohibits guided selection')
                if job.command.selection == 'pressure':
                    stage = getattr(job.command, 'pressure_stage', '')
                    if stage not in ('prefill', 'decode'):
                        raise ValueError('Recall pressure event lacks its original execution stage')
                    event_type = stage + '_pressure'
                    if configured_event_method(event_type) == ACCUMULATED_ATTENTION:
                        recall_q, metadata = None, {}
                    else:
                        window = getattr(authority, 'pressure_query_window', None)
                        if window is None:
                            raise ValueError('Recall pressure requires captured accepted Q')
                        recall_q, metadata = window.read(job.history)
                elif job.command.selection == 'query' and job.command.query_span is not None:
                    recall_q = queries
                    begin, end = job.command.query_span
                    metadata = dict(query_source='public_query', query_begin=begin, query_end=end,
                        query_processed_through=end, public_input_end=job.history+len(job.command.input_tokens))
                    event_type = 'query'
                else:
                    raise ValueError('Recall experiment encountered an unqualified selection entry')
                if (_settings.get('KVMEM_VALUE_RECALL_REFERENCE') == '1' and
                        configured_event_method(event_type) != ACCUMULATED_ATTENTION):
                    authority.recall_value_scales = self.runtime.scales[:authority.scoring_layers].detach().cpu()
                plan = select_for_event(authority, recall_q, mandatory, max_tokens=cfg.select_budget,
                    event_type=event_type, query_metadata=metadata, request_id=job.command.request_id,
                    output=self.runner._kvmem_probe_output)
            elif configured_method():
                # A fitting history needs no ranking. In particular, do not
                # send a fitting pressure event through the legacy scorer.
                plan = authority.memory.select({}, action=False, max_tokens=cfg.select_budget)
            elif job.command.selection=='pressure':
                if job.guided_pressure and queries is not None:
                    plan=authority.select_for_queries(queries,mandatory,max_tokens=cfg.select_budget,
                                                     guided_pressure=True)
                else:
                    from .prefill_attention import select_pressure
                    plan=select_pressure(authority,mandatory,max_tokens=cfg.select_budget)
            elif job.history<=cfg.select_budget:plan=authority.memory.select({})
            elif _settings.get('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES') == 'sum':
                if job.guided_rows or job.command.query_span is None:
                    raise ValueError('Token answer experiment requires the original public Query')
                from .query_token_experiment import select_token_queries
                plan=select_token_queries(authority,queries,mandatory,max_tokens=cfg.select_budget)
            else:plan=authority.select_for_queries(queries,mandatory,max_tokens=cfg.select_budget)
            if (getattr(self.options, 'action_profiling', False) or
                    getattr(self.runner, '_kvmem_step_timing', None) is not None):
                self.emit(job, 'selection_prepared',
                    publication_seconds=published-selection_started,
                    selection_seconds=time.perf_counter()-published)
            started=time.perf_counter()
            _a0 = started
            assembler=NativeWindowAssembler.from_runner(self.runner,job.command.request_id,self.runtime.names,
                cfg.block_tokens,packed=self.options.packed_assembly,
                direct_pinned_reads=self.options.direct_pinned_reads,arena='refresh',
                **({'staging_lane':self._current_lane()} if self._current_lane() else {}))
            pressure = getattr(self.runner, '_kvmem_layerwise_pressure', None)
            if (pressure is not None and pressure.single_assembly and
                    job.command.selection == 'pressure'):
                from .layerwise_pressure_efficiency import install_single_assembly
                install_single_assembly(assembler, pressure, job, self.runtime.names)
            if self.maintenance is not None:
                assembler.bandwidth_limiter = self.maintenance.transfer_budget
            if getattr(job, 'promised_lease', None) is not None:
                from ..core.publication_trace import emit as _trace
                _trace('assembler_prelude', rid=job.command.request_id, build_install=time.perf_counter()-_a0)
            try:
                assembly=assembler.assemble(authority,plan)
            except BaseException as error:
                lease = getattr(job, 'layerwise_pipeline', None)
                if lease is not None:
                    # The layer assembler drains its own failed reads/copies.
                    # Wake any foreground consumer; never continue with a
                    # partially rebuilt layer or silently fall back.
                    fence = torch.cuda.Event()
                    fence.record(torch.cuda.current_stream(self.runner.device))
                    lease.fail(error, fence)
                raise
            self.emit(job,'assembly_submitted',host_seconds=time.perf_counter()-started)
            return assembler,assembly,plan,fit,time.perf_counter()-started

    def step(self,job):
        if job.done:return False
        from .connector import RestoreCommand
        if isinstance(job.command, RestoreCommand):
            from .async_restore import step_restore
            return step_restore(self, job)
        if job.cancelled and job.future is None:
            from .pressure_key_reuse import cancelled as _reuse_cancelled
            _reuse_cancelled(self.runner, job)
            self._restore_state(job)
            if job.restore_event is None:
                job.restore_event=torch.cuda.Event();job.restore_event.record()
            if not job.restore_event.query():return False
            if job.pressure_oracle is not None:
                job.pressure_oracle.verify(job);job.pressure_oracle=None
            self._release(job);job.done=True
            self.emit(job,'cancelled');return True
        if job.stage=='assembly_wait':
            # A future can finish enqueueing kernels before its CUDA fence is
            # ready. Keep the sole scratch arena leased until main-thread
            # commit/close, not merely until the producer function returns.
            from .decode_priority import should_yield
            if should_yield(self.jobs.values(), job):
                return False  # Opt-in: a waiting decode-stage pressure takes the lane first.
            from .async_restore import lane_busy
            multi = self._multi_lane(job)
            if multi:
                lane = self._claim_lane(job)
                if lane is None:
                    return False
            else:
                lane = 0
                if lane_busy(self) or (self._pressure_lanes() > 1 and self._producers()):
                    return False
            policy = getattr(self, 'maintenance', None)
            if policy is not None and not (policy.acquire(job, lanes=self._pressure_lanes()) if multi
                                           else policy.acquire(job)):
                return False
            job.authority._refresh_background=True
            job.stage='assembly'
            job.lane=lane
            if lane == 0:
                from .tensor_parallel import submit_transfer
                job.future=submit_transfer(self.executor,self._assemble,job,*job.assembly_inputs)
            else:
                job.future=self._executor_for(lane).submit(self._assemble_in_lane,lane,job,*job.assembly_inputs)
            job.assembly_inputs=None
            return True
        if job.stage=='assembly':
            from .pressure_pipeline import pipeline_for
            pipeline = pipeline_for(self)
            if pipeline is not None and pipeline.dispatch(self, job):
                return True
            if not job.future.done():return False
            if job.result is None:job.result=job.future.result()
            assembler,assembly,plan,fit,seconds=job.result
            if not assembly.event.query():return False
            authority=job.authority;authority._refresh_background=False
            if job.graph_binding is not None:
                if self.runtime.authorities.get(job.command.request_id) is not authority:
                    raise RuntimeError('Asynchronous graph selection lost authority ownership')
                from .prompt_spans import selection_metadata
                selection_metadata(authority).validate_binding(job.graph_binding)
            authority.memory.attach_completion(assembly)
            if not authority.memory.commit_selection():raise RuntimeError('Ready refresh did not commit')
            job.selected_tokens=plan.total_window_tokens
            if not job.cancelled:
                if job.command.selection!='pressure':
                    from .prefill_attention import exclude_query_replay
                    exclude_query_replay(self.runner,authority,job.command.request_id,job.history,
                        job.history+len(job.command.input_tokens),job.command.query_span is not None)
                    from .layerwise_checkpoints import query_frame_committed
                    query_frame_committed(self.runner,job.command.request_id)
                if job.command.selection=='pressure' and not getattr(job,'layerwise_pipeline_published',False):
                    # Install the new per-layer maps before a session snapshot at
                    # this position reads them, as the pressure pipeline does.
                    diagnostic=getattr(self.runner,'_kvmem_layerwise_query',None)
                    if diagnostic is not None:
                        diagnostic.committed(job);job.layerwise_maps_committed=True
                self.runner._kvmem_window_adapter.commit(job.command.request_id,job.history,job.selected_tokens)
                if self.runner._kvmem_sessions is not None:
                    self.runner._kvmem_sessions.selection_committed(job.command.request_id,job.history)
            if job.pressure_oracle is not None:
                job.pressure_oracle.verify(job);job.pressure_oracle=None
            if not assembler.close():raise RuntimeError('Completed assembly retained scratch')
            if getattr(self, 'maintenance', None) is not None:
                self.maintenance.observe(job, assembly.uploaded_bytes)
                self.maintenance.release(job)
            self.report(job,assembly,fit,seconds)
            self._release(job);job.future=job.result=None;job.done=True
            self.emit(job,'cancelled' if job.cancelled else 'ready',public_peers=len(job.peers))
            return True
        if job.stage=='prepare':
            rid=job.command.request_id
            if rid not in self.runner.req_states.req_id_to_index:return False
            self._ensure_window_adapter(job.command)
            if self.runner._kvmem_sessions is not None:
                # A zero-replay pressure ticket still consumes any checkpoint
                # queued at this exact accepted prefix. Session capture only
                # needs owner IDs; never manufacture an empty model batch.
                if job.command.input_tokens:
                    batch = maintenance_batch(self.runner,rid,job.command.input_tokens[:1],job.history)
                else:
                    from types import SimpleNamespace
                    batch = SimpleNamespace(req_ids=[rid])
                self.runner._kvmem_sessions.before_attention(batch)
            authority=self.runtime.authorities.get(rid)
            if authority is None or authority.history_end<job.history:return False
            if authority.history_end!=job.history:raise ValueError('Private Query crossed accepted history')
            job.authority=authority
            from .prompt_spans import selection_metadata, install_selection_metadata
            install_selection_metadata(authority, job.command, validated_prefix=job.history)
            graph = selection_metadata(authority)
            if graph is not None:
                job.graph_binding = graph.binding
            if getattr(authority,'query_history',None) is not None:
                from .query_history import begin_round
                begin_round(self.runner,authority,rid)
            from .pressure_promise import refresh_async_page_check, defer_page_check
            deferred_checks=[] if refresh_async_page_check() else None
            for group,ticket in job.command.groups:
                expansion=stored_expansions(self.runner.block_tables)[group]
                expected=[b*expansion+p for b in ticket.original_block_ids for p in range(expansion)]
                index=self.runner.req_states.req_id_to_index[rid]
                count=int(self.runner.block_tables.num_blocks.np[group,index])
                row=self.runner.block_tables.block_tables[group].gpu[index,:count]
                if deferred_checks is not None:
                    # Opt-in KVMEM_REFRESH_ASYNC_PAGE_CHECK: verified by _assemble
                    # before any page write, without blocking this engine thread.
                    deferred_checks.append(defer_page_check(row,expected))
                    continue
                actual=row.cpu().tolist()
                if actual!=expected:raise ValueError('Private refresh page reservation mismatch')
            if deferred_checks:
                job.promise_page_checks=tuple(getattr(job,'promise_page_checks',()) or ())+tuple(deferred_checks)
            if not job.command.input_tokens and _settings.get('KVMEM_VERIFY_SESSION_RESTORE') == '1':
                from .rolling_verification import RollingPressureOracle
                job.pressure_oracle = RollingPressureOracle(self.runner, job)
            layerwise = getattr(self.runner, '_kvmem_layerwise_query', None)
            if (getattr(layerwise, 'no_probe', False) and job.command.selection == 'query'):
                layerwise.arm_no_probe(self, job)
                return True
            if ((job.command.selection=='pressure' and not job.guided_pressure) or
                    job.history<=job.command.groups[0][1].window_tokens):
                self._submit(job);return True
            from .native_state import NativeRecurrentState
            from ..core.state import StateCheckpoints
            if self.adapter is None:self.adapter=NativeRecurrentState(self.runner)
            bundle=self.adapter.borrow([rid]);checkpoints=StateCheckpoints(rid,bundle.bytes,1)
            job.state=(bundle,checkpoints,checkpoints.capture(job.history,bundle.tensors))
            if job.guided_pressure:
                job.guided_witness = self.pressure_witness(authority)
                self.start_guided(job)
                return True
            job.stage='query';return True
        if job.stage=='query':
            if getattr(self, 'cobatch', None) is not None and self.cobatch.defer(job):
                return False
            end=min(len(job.command.input_tokens),job.offset+self.options.refresh_quantum_tokens)
            _,q=self._forward(job,job.command.input_tokens[job.offset:end],job.history+job.offset)
            self.accept_query(job, q, end)
            return True
        if job.stage=='guided_prompt':
            end=min(len(self.guided_prompt),job.guided_offset+self.options.refresh_quantum_tokens)
            started=time.perf_counter()
            token,_=self._forward(job,self.guided_prompt[job.guided_offset:end],
                job.guided_base+job.guided_offset,next_token=end==len(self.guided_prompt))
            job.guided_seconds+=time.perf_counter()-started;job.guided_offset=end
            if token is not None:job.next_token=token;job.stage='guided_decode'
            return True
        if job.stage=='guided_decode':
            policy=job.guided_policy
            if policy is None:raise RuntimeError('Private refresh lost its guided acceptance policy')
            capture=policy.prepare(job.next_token)
            if capture is None:
                job.guided_diagnostics=policy.report()
                job.guided_fallback=policy.fallback
                if policy.fallback is not None:job.guided_rows.clear()
                self._submit(job);return True
            started=time.perf_counter()
            token,q=self._forward(job,[job.next_token],job.guided_base+len(self.guided_prompt)+len(job.generated),next_token=True)
            policy.commit();job.generated.append(job.next_token)
            job.guided_seconds+=time.perf_counter()-started
            if capture:job.guided_rows.append(q)
            job.next_token=token
            return True
        raise RuntimeError('Unknown refresh stage')

    def accept_query(self, job, q, end):
        """Retain only the logical Query span for either execution path."""
        qb,qe=job.command.query_span or (job.history,job.history+len(job.command.input_tokens))
        lo=max(qb,job.history+job.offset)-job.history-job.offset
        hi=min(qe,job.history+end)-job.history-job.offset
        if hi>lo:
            from .query_capture_buffer import append_query_chunk
            total = min(qe, job.history+len(job.command.input_tokens)) - max(qb, job.history)
            append_query_chunk(job, q, lo, hi, total)
        job.offset=end
        if end==len(job.command.input_tokens):
            boundary_guided = (job.command.guided_boundary if job.command.guided_boundary is not None
                else self.options.guided_reselect in ('boundary', 'both'))
            if self.options.guided_query_tokens and job.command.query_span is not None and boundary_guided:
                self.start_guided(job)
            else:self._submit(job)

    def start_guided(self, job):
        """Probe only the accepted prefix for pressure, before public replay."""
        local_end=job.guided_base-self.runner._kvmem_window_adapter.offsets.get(job.command.request_id,0)
        reserve=self.options.guided_reserve
        available=local_end+reserve<=self.options.memory.select_budget+self.options.memory.gen_budget
        for group,ticket in job.command.groups:
            spec=self.runner.kv_cache_config.kv_cache_groups[group].kv_cache_spec
            available &= local_end+reserve<=len(ticket.original_block_ids)*spec.block_size
        if not available:
            job.guided_fallback='capacity';self._submit(job);return
        if self.tokenizer is None:
            from vllm.tokenizers import get_tokenizer
            c=self.runner.vllm_config.model_config
            self.tokenizer=get_tokenizer(c.tokenizer,tokenizer_mode=c.tokenizer_mode,trust_remote_code=c.trust_remote_code)
            from .guided_query import PRIVATE_PROMPT
            self.guided_prompt=self.tokenizer.encode(PRIVATE_PROMPT,add_special_tokens=False)
        if not self.guided_prompt or len(self.guided_prompt)+self.options.guided_query_tokens>reserve:
            raise ValueError('Private Query exceeds its scheduler scratch reservation')
        from .guided_query import GuidedQueryPolicy
        job.guided_policy=GuidedQueryPolicy(self.tokenizer,self.options.guided_query_tokens)
        job.stage='guided_prompt'

    def _release(self,job):
        if job.state is not None:
            if not job.state[1].release(job.history):raise RuntimeError('Recurrent restore still in flight')
            job.state=None
        job.query_rows.clear();job.guided_rows.clear()
        from .query_capture_buffer import release_query_buffer
        release_query_buffer(job)
        job.assembly_inputs=None

    def report(self,job,assembly,fit,seconds):
        from .prefill_attention import selection_report
        authority=job.authority;cfg=self.options.memory
        row=dict(request_id=job.command.request_id,command_id=job.command.command_id,
            history_begin=job.history,selected_tokens=job.selected_tokens,query_tokens=len(job.command.input_tokens),
            selection=job.command.selection,query_span=job.command.query_span,mandatory_fit=fit,
            replay_span=[job.history,job.history+len(job.command.input_tokens)],
            selected_history_ranges=[[r[2],r[3]] for r in (
                job.pipeline_report_records if hasattr(job, 'pipeline_report_records') else
                authority.memory._store.working_records())],
            query_score_tokens=sum(q.shape[1] for q in (job.guided_rows or job.query_rows)),
            guided_query_tokens=len(job.generated),guided_seconds=job.guided_seconds,guided_fallback=job.guided_fallback,
            query_history_enabled=getattr(authority,'query_history',None) is not None,
            action_attention_enabled=cfg.action_attention,action_fraction=cfg.action_fraction if cfg.action_attention else 0.,
            action_combine=cfg.action_combine,action_history_pages=sum(v>0 for v in getattr(getattr(authority,'action_history',None),'scores',{}).values()),
            query_history_pages=len(authority.query_history.policy.scores) if getattr(authority,'query_history',None) is not None else 0,
            mean_k_backend=self.options.mean_k_backend,assembly_seconds=seconds,
            assembly_host_prepare_seconds=assembly.host_prepare_seconds,assembly_uploaded_bytes=assembly.uploaded_bytes,
            raw_cache_hit_tokens=assembly.raw_cache_hit_tokens,resident_v_reused_tokens=assembly.resident_v_reused_tokens,
            assembly_read_prefetch=getattr(self.options, 'assembly_read_prefetch', False),
            assembly_fused_layers=getattr(self.options, 'assembly_fused_layers', False),
            refresh_started_at=job.started,refresh_finished_at=time.perf_counter(),asynchronous=True,
            public_peer_count=len(job.peers),private_forward_steps=job.steps,cancelled=job.cancelled,
            query_execution_counts=dict(job.query_execution_counts),
            query_replay_cobatch=(self.cobatch.snapshot() if getattr(self, 'cobatch', None) is not None else None),
            maintenance=self.maintenance.snapshot() if getattr(self, 'maintenance', None) is not None else None,
            **selection_report(authority, phase=job.command.selection))
        row.update(job.guided_diagnostics)
        from .recall_experiment import configured_method
        if configured_method():
            report = getattr(authority, 'recall_selection_report', None)
            row['configured_method'] = configured_method()
            row['reselection_required'] = report is not None
            row['recall_experiment'] = dict(report) if report is not None else None
            row['actual_method'] = report['actual_method'] if report is not None else None
            row['query_scoring_method'] = report['actual_method'] if report is not None else 'no_eviction_required'
        elif job.command.selection == 'query':
            row['query_scoring_method'] = ('content_token_softmax_page_sum' if
                _settings.get('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES') == 'sum' else
                getattr(authority, 'query_strategy_name', None) or getattr(authority, 'query_score_mode', 'mean_k'))
            if getattr(authority, 'query_score_mode', 'mean_k') == 'overlap_16_8':
                row['query_window_scoring'] = dict(authority.query_window_stats)
            if getattr(authority, 'query_score_mode', 'mean_k') == 'key_fps_4':
                row['query_key_fps_scoring'] = dict(authority.query_key_fps_stats)
                if getattr(self.options, 'query_fps_halo_blocks', 0):
                    row['query_group_selection'] = dict(authority.query_group_selection)
                if getattr(self.options, 'query_mean_quota_fraction', 0):
                    row['query_quota_selection'] = dict(authority.query_quota_selection)
                    if getattr(authority, 'query_quota_propagation', None) is not None:
                        row['query_quota_propagation'] = authority.query_quota_propagation
        if job.guided_pressure:
            row.update(guided_scope='prefill-pressure', guided_context_end=job.guided_base,
                public_input_replayed_for_guidance=0, guided_state_verified=job.guided_state_verified)
        if (getattr(self.options, 'action_profiling', False) or
                getattr(self.runner, '_kvmem_step_timing', None) is not None):
            row.update(assembly_read_stats=assembly.read_stats, assembly_chunks=assembly.chunks,
                assembly_rebuilt_tokens=assembly.rebuilt_tokens,
                assembly_skipped_tokens=assembly.skipped_tokens)
        with (self.runner._kvmem_probe_output/'bounded_refresh.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def shutdown(self):
        self.closed=True
        self.executor.shutdown(wait=True)
        for executor in self.__dict__.get('_lane_executors', {}).values():
            executor.shutdown(wait=True)
        self.resident_executor.shutdown(wait=True)
        for stream in self.__dict__.get('_lane_streams', {}).values():
            if stream is not None:stream.synchronize()
        if self.resident_stream is not None:self.resident_stream.synchronize()
        from .pressure_pipeline import pipeline_for
        pipeline = pipeline_for(self)
        if pipeline is not None:
            pipeline.reap(self)

    def retire_namespace(self, namespace):
        # The refresh arena index has the same sole CPU producer as assembly.
        # Do not mutate its eviction map from the engine during another read.
        def drop():
            cache=getattr(self.runner,'_kvmem_refresh_gpu_raw_cache',None)
            if cache is not None:cache.drop_namespace(namespace)
        self.executor.submit(drop)
