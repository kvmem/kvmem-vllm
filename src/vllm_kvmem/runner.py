"""Version-isolated runner extension. Probe mode does not enable KVMem serving.

The base runner continues all scheduling, projections, attention and sampling.
This stage verifies actual target-layer capture on a real native batch.
"""
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from .cache_groups import full_attention_groups
from .capture import capture_to, install_qwen_capture


def make_probe_runner(base):
    class KVMemProbeRunner(base):
        def load_model(self, *args, **kwargs):
            from .options import RuntimeOptions
            options = RuntimeOptions.from_vllm(self.vllm_config)
            from .capture_compile import validate_capture_compile_config
            opaque_fingerprint = validate_capture_compile_config(
                self.vllm_config, options.capture_opaque_projection)
            if options.capture_opaque_projection:
                from vllm.config import CUDAGraphMode, CompilationMode
                compilation = self.vllm_config.compilation_config
                if (os.getenv('KVMEM_FULL_GRAPH') != '1' or self.model_config.enforce_eager or
                        compilation.mode != CompilationMode.VLLM_COMPILE or
                        compilation.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE or
                        self.vllm_config.max_concurrent_batches != 1 or
                        getattr(self.speculative_config, 'method', None) != 'mtp'):
                    raise ValueError('Opaque capture projection requires compiled native MTP '
                                     'and synchronous single-batch graphs')
                # Register the ABI before native AOT unpickling resolves ops.
                from .capture_opaque import install_opaque_capture
            super().load_model(*args, **kwargs)
            from .draft_quantization import install_quantized_context
            install_quantized_context(self)
            self._kvmem_graph = os.getenv('KVMEM_FULL_GRAPH') == '1'
            if not self.model_config.enforce_eager and not self._kvmem_graph:
                raise ValueError("Runner capture probe currently requires enforce_eager")
            if self._kvmem_graph:
                from vllm.config import CUDAGraphMode, CompilationMode
                compilation = self.vllm_config.compilation_config
                self._kvmem_native_compilation = compilation.mode == CompilationMode.VLLM_COMPILE
                allowed = ((CompilationMode.NONE, CUDAGraphMode.FULL),
                           (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE))
                if ((compilation.mode, compilation.cudagraph_mode) not in allowed or
                        self.vllm_config.max_concurrent_batches != 1):
                    raise ValueError('KVMem capture requires qualified native compilation/graphs and synchronous batches')
                if self._kvmem_native_compilation and getattr(self.speculative_config, 'method', None) != 'mtp':
                    raise ValueError('Compiled KVMem capture currently qualifies native Qwen MTP only')
            if self.vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Probe currently qualifies single-rank execution only")
            self._kvmem_layers = install_qwen_capture(self.model)
            if self._kvmem_graph:
                # Native memory profiling compiles before KV initialization.
                # Allocate once now; later graph setup must retain these addresses.
                from .capture import install_persistent_capture
                self._kvmem_graph_captures = install_persistent_capture(
                    self.model, self._kvmem_layers, self.max_num_tokens,
                    self.input_buffers.positions)
            from .mtp_runtime import install_mtp_runtime
            install_mtp_runtime(self)
            self._kvmem_capture_opaque = (install_opaque_capture(self, opaque_fingerprint)
                if options.capture_opaque_projection else None)
            self._kvmem_serving = os.getenv('KVMEM_SERVING') == '1'
            self._kvmem_probe_done = self._kvmem_serving
            self._kvmem_probe_output = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"])
            self._kvmem_scheduled = os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") == "1"
            self._kvmem_bounded_target = os.getenv("KVMEM_BOUNDED_TARGET") == "1"
            # Install before profiling/compilation/CUDA graph capture. Graphs
            # retain these addresses, and read fresh sampling metadata each run.
            self._kvmem_native_attention = None
            if getattr(options, 'action_native_scores', 'off') != 'off':
                from .native_attention import install_native_attention
                self._kvmem_native_attention = install_native_attention(self, options, self._kvmem_probe_output)
            if self._kvmem_serving and options.mean_k_backend == 'qw3_cuda':
                # Load before the engine becomes ready. A first Query must not
                # compile the extension while other owners wait for the GPU.
                from .cuda_scoring import kernel
                kernel()
            self._kvmem_step_timing = None
            self._kvmem_public_work = None
            if options.step_latency_interval:
                from .step_timing import StepTiming
                from .work_counters import PublicWorkCounters
                self._kvmem_step_timing = StepTiming(options.step_latency_interval,
                    self._kvmem_probe_output / 'model_step_latency.jsonl')
                self._kvmem_public_work = PublicWorkCounters(
                    self._kvmem_probe_output / 'public_work.json')
            self._kvmem_cpu_history_coordination = options.cpu_history_coordination
            self._kvmem_repository_gc = options.repository_gc
            self._kvmem_prepared_archive = None
            if options.nvme_preallocate or options.nvme_max_bytes or options.archive_io_max_bytes or options.historical_gc is not None:
                from .storage import make_shared_archive
                modules = dict(self.model.named_modules())
                modules.update(getattr(self, '_kvmem_mtp_modules', {}))
                layers = [modules[name] for name in getattr(self, '_kvmem_archive_layers', self._kvmem_layers)]
                geometry = {(layer.num_kv_heads, layer.head_dim) for layer in layers}
                if len(geometry) != 1:
                    raise ValueError('NVMe startup reservation requires uniform target KV geometry')
                heads, dim = geometry.pop()
                record_bytes = len(layers) * heads * dim * options.memory.block_tokens * 2
                self._kvmem_prepared_archive = make_shared_archive(options,
                    self._kvmem_probe_output / 'continuous_archive', record_bytes,
                    self.vllm_config.scheduler_config.max_num_seqs if self._kvmem_bounded_target else 1,
                    max_query_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens)
            self._kvmem_saved_capture = None
            self._kvmem_window_adapter = None
            self._kvmem_window_batch = None
            self._kvmem_continuous = os.getenv("KVMEM_ARCHIVE_PROBE") == "1"
            self._kvmem_capture_runtime = None
            self._kvmem_pending_capture = None
            self._kvmem_sessions = None
            if options.session_capacity:
                if not self._kvmem_bounded_target or not self._kvmem_continuous:
                    raise ValueError('Prompt sessions require bounded continuous authority capture')
                from .session_worker import SessionWorker
                self._kvmem_sessions = SessionWorker(self, options)
            self._kvmem_request_generations = {}
            self._kvmem_next_generation = 0
            self._kvmem_restore_allocations = {}
            if self._kvmem_bounded_target:
                if any(os.getenv(flag) == "1" for flag in (
                        "KVMEM_REPLAY_PROBE", "KVMEM_COMPRESSION_PROBE", "KVMEM_REBUILD_PROBE", "KVMEM_SCHEDULED_REBUILD_PROBE")):
                    raise ValueError("Bounded target execution requires its own scheduled refresh lifecycle")
                from .bounded_runtime import install_bounded_target
                install_bounded_target(self)
            if os.getenv("KVMEM_REPLAY_PROBE") == "1":
                from .replay_probe import install_replay_probe
                install_replay_probe(self)
            if os.getenv("KVMEM_COMPRESSION_PROBE") == "1":
                if os.getenv("KVMEM_REPLAY_PROBE") == "1" or not self._kvmem_continuous:
                    raise ValueError("Compression probe requires its own replay and continuous authority")
                from .compression_probe import install_compression_probe
                install_compression_probe(self)
            if os.getenv("KVMEM_WINDOW_METADATA") == "1":
                from .window_bridge import install_window_bridge
                def current_window(batch):
                    if batch is getattr(self, '_kvmem_last_batch', None):
                        return self._kvmem_window_batch
                    if self._kvmem_graph and self._kvmem_window_adapter is not None:
                        return self._kvmem_window_adapter.capture_window(batch)
                    return None
                install_window_bridge(self.model_state, current_window)
            if os.getenv("KVMEM_REBUILD_PROBE") == "1" and self.speculative_config is not None:
                raise ValueError("Rebuild probe needs target-only qualification before draft lifecycle integration")
            from .profile_scopes import install_profile_scopes
            install_profile_scopes(self)
            if options.host_stage_timing:
                from .host_trace import install_host_trace
                install_host_trace(self, self._kvmem_probe_output)

        def add_requests(self, scheduler_output):
            result = super().add_requests(scheduler_output)
            for request in scheduler_output.scheduled_new_reqs:
                self._kvmem_next_generation += 1
                self._kvmem_request_generations[request.req_id] = self._kvmem_next_generation
                sessions = getattr(self, '_kvmem_sessions', None)
                if (sessions is not None and sessions.options.async_restore and
                        request.req_id in sessions.imports):
                    from .restore_lease import FreshRestoreLease
                    self._kvmem_restore_allocations[request.req_id] = FreshRestoreLease.capture(
                        self, request.req_id, request.block_ids)
            return result

        def update_requests(self, scheduler_output):
            leased = getattr(self, '_kvmem_restore_allocations', {})
            control = getattr(scheduler_output, 'kv_connector_metadata', None)
            table_updates = getattr(control, 'block_table_updates', {})
            updated = (set(scheduler_output.num_scheduled_tokens) |
                       set(getattr(scheduler_output, 'block_table_updates', None) or {}) |
                       set(table_updates) |
                       set(scheduler_output.scheduled_cached_reqs.req_ids))
            if leased.keys() & updated:
                raise RuntimeError('Suspended restore allocation received a public page/state update')
            if table_updates:
                for request_id, block_ids in table_updates.items():
                    index = self.req_states.req_id_to_index.get(request_id)
                    if index is not None:
                        self.block_tables.append_block_ids(index, block_ids, overwrite=True)
                # Full rows already include allocations made in this native step.
                # Let native code update counters/zero pages, without appending twice.
                cached = scheduler_output.scheduled_cached_reqs
                scheduler_output = replace(scheduler_output, scheduled_cached_reqs=replace(
                    cached, new_block_ids=[None if rid in table_updates else blocks
                        for rid, blocks in zip(cached.req_ids, cached.new_block_ids, strict=True)]))
            return super().update_requests(scheduler_output)

        def initialize_kv_cache(self, *args, **kwargs):
            result = super().initialize_kv_cache(*args, **kwargs)
            from .recurrent_table_journal import install_recurrent_table_journal
            is_profiling = kwargs.get('is_profiling', args[1] if len(args) > 1 else False)
            install_recurrent_table_journal(self, is_profiling=is_profiling)
            refresh = getattr(self, '_kvmem_async_refresh', None)
            if refresh is not None:
                refresh.initialize_cache()
            if getattr(self, '_kvmem_graph', False):
                from .graph_runtime import initialize_graph_window
                initialize_graph_window(self)
            from .profile_scopes import install_profile_scopes
            install_profile_scopes(self)
            if not is_profiling:
                from .options import RuntimeOptions
                from .checkpoint_prewarm import prewarm_checkpoint_payloads
                prewarm_checkpoint_payloads(self, RuntimeOptions.from_vllm(self.vllm_config))
            return result

        def finish_requests(self, scheduler_output):
            if getattr(self, '_kvmem_restore_allocations', {}).keys() & scheduler_output.finished_req_ids:
                raise RuntimeError('Native request was freed before its restore fence')
            result = super().finish_requests(scheduler_output)
            for request_id in scheduler_output.finished_req_ids:
                self._kvmem_request_generations.pop(request_id, None)
            return result

        def shutdown(self):
            try:
                refresh = getattr(self, '_kvmem_async_refresh', None)
                if refresh is not None:
                    refresh.shutdown()
                runtime = getattr(self, '_kvmem_capture_runtime', None)
                if runtime is not None:
                    runtime.shutdown_retirement()
                else:
                    archive = getattr(self, '_kvmem_prepared_archive', None)
                    if archive is not None and not archive.close():
                        raise RuntimeError('Unused startup archive retained I/O ownership')
                native = getattr(self, '_kvmem_native_attention', None)
                if native is not None:
                    native.report(force=True)
            finally:
                try:
                    work = getattr(self, '_kvmem_public_work', None)
                    if work is not None:
                        work.poll(force=True, flush=True)
                finally:
                    super().shutdown()

        def _kvmem_refresh(self, command):
            from .refresh_probe import rebuild_native_batch
            captured, request_id = self._kvmem_saved_capture
            if (command.request_id != request_id or
                    self._kvmem_last_batch.req_ids != [request_id]):
                raise ValueError("Scheduled probe refers to a different request")
            if any(ticket.history_end != self._kvmem_last_batch.num_tokens
                   or ticket.window_tokens != ticket.history_end
                   for _, ticket in command.groups):
                raise ValueError("Scheduled probe only qualifies initial identity refresh")
            report = rebuild_native_batch(self, captured, self._kvmem_probe_output)
            if not report["all_byte_equal"]:
                raise RuntimeError("Scheduled refresh changed native KV bytes")
            from .options import RuntimeOptions
            from .window_batch import WindowBatchAdapter
            if os.getenv("KVMEM_WINDOW_METADATA") != "1":
                raise RuntimeError("Scheduled refresh requires the explicit window metadata hook")
            if self._kvmem_window_adapter is None:
                config = RuntimeOptions.from_vllm(self.vllm_config).memory
                self._kvmem_window_adapter = WindowBatchAdapter(
                    tuple(group for group, _ in command.groups), config.select_budget + config.gen_budget)
            extents = {(ticket.history_end, ticket.window_tokens) for _, ticket in command.groups}
            if len(extents) != 1:
                raise ValueError("All full-attention groups must commit the same dense window")
            history_end, window_tokens = extents.pop()
            self._kvmem_window_adapter.commit(command.request_id, history_end, window_tokens)
            report["command_id"] = command.command_id
            report["request_id"] = command.request_id
            path = self._kvmem_probe_output / "capture.json"
            capture_report = json.loads(path.read_text())
            capture_report["scheduled_rebuild"] = report
            path.write_text(json.dumps(capture_report, indent=2))
            self._kvmem_saved_capture = None
            event = torch.cuda.Event()
            event.record()
            return event

        def prepare_inputs(self, *args, **kwargs):
            batch = super().prepare_inputs(*args, **kwargs)
            refresh = getattr(self, '_kvmem_async_refresh', None)
            cobatch = getattr(refresh, 'cobatch', None)
            return cobatch.prepare(batch) if cobatch is not None else batch

        def _kvmem_rebuild_public_metadata(self, batch):
            from vllm.config.compilation import CUDAGraphMode
            from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
            self._kvmem_projecting_query = True
            try:
                tables, slots = self.prepare_attn(batch)
                metadata = self.model_state.prepare_attn(batch, CUDAGraphMode.NONE,
                    tables, slots, self.attn_groups, self.kv_cache_config)
                return metadata, build_slot_mappings_by_layer(slots, self.kv_cache_config)
            finally:
                self._kvmem_projecting_query = False

        def prepare_attn(self, input_batch):
            projecting = getattr(self, '_kvmem_projecting_query', False)
            refresh = getattr(self, '_kvmem_async_refresh', None)
            cobatch = getattr(refresh, 'cobatch', None)
            public = cobatch.public_batch(input_batch) if cobatch is not None else input_batch
            sessions = getattr(self, '_kvmem_sessions', None)
            if sessions is not None and not projecting:
                sessions.before_attention(public)
            if getattr(self, '_kvmem_bounded_target', False) and not projecting:
                from .native_state import canonicalize_before_non_speculative_batch, initialize_cold_single_token_state
                initialize_cold_single_token_state(self, input_batch)
                canonicalize_before_non_speculative_batch(self, input_batch)
            decisions = getattr(self, '_kvmem_active_decisions', None)
            if decisions is not None and not projecting:
                decisions.before_attention(public)
            self._kvmem_last_batch = input_batch
            self._kvmem_window_batch = None
            if self._kvmem_window_adapter is not None:
                window = self._kvmem_window_adapter.prepare(input_batch)
                self._kvmem_window_batch = window
                if window is not None:
                    active_groups = set(full_attention_groups(self.kv_cache_config))
                    if active_groups != set(window.groups):
                        raise ValueError("Window mapping must cover exactly the full-attention cache groups")
                    result = super().prepare_attn(replace(input_batch, positions=window.positions))
                    if not projecting:
                        self._kvmem_prepare_action_metadata(public, window.positions[:public.num_tokens_after_padding])
                    timing = getattr(self, '_kvmem_step_timing', None)
                    if timing is not None and not projecting:
                        timing.model_begin(public, self._kvmem_window_adapter.offsets)
                    return result
            result = super().prepare_attn(input_batch)
            if not projecting:
                self._kvmem_prepare_action_metadata(public, input_batch.positions[:public.num_tokens_after_padding])
            timing = getattr(self, '_kvmem_step_timing', None)
            if timing is not None and not projecting:
                timing.model_begin(public)
            return result

        def _kvmem_prepare_action_metadata(self, batch, positions):
            runtime = getattr(self, '_kvmem_capture_runtime', None)
            action = getattr(runtime, 'action', None)
            if action is not None and getattr(self, '_kvmem_public_action_forward', False):
                action.prepare(batch, positions)
                if getattr(runtime, '_prefill_scatter_deferred', None) is not None:
                    runtime.recheck_prefill_scatter(batch)
                runtime.note_prefill_pipeline_forward(batch)
            else:
                native = getattr(self, '_kvmem_native_attention', None)
                if native is not None:
                    native.disarm()
                if runtime is not None and getattr(runtime, '_prefill_scatter_deferred', None) is not None:
                    runtime.recheck_prefill_scatter(batch)

        def sample(self, hidden_states, input_batch, grammar_output):
            timing = getattr(self, '_kvmem_step_timing', None)
            if timing is not None:
                timing.classify(input_batch)
            result = super().sample(hidden_states, input_batch, grammar_output)
            pending = getattr(self, "_kvmem_pending_capture", None)
            if pending is not None:
                captured, original_batch = pending
                if input_batch is not original_batch:
                    raise ValueError("Continuous authority requires the native unsharded target batch")
                if getattr(self.speculative_config, 'method', None) == 'mtp':
                    # MTP's authoritative shifted-token prefix is produced by
                    # propose after sampling; publish both layers together.
                    self._kvmem_mtp_rejected = result[2]
                else:
                    self._kvmem_capture_runtime.submit(captured, input_batch, result[2])
                    self._kvmem_pending_capture = None
            return result

        def sample_tokens(self, grammar_output):
            result = super().sample_tokens(grammar_output)
            timing = getattr(self, '_kvmem_step_timing', None)
            if timing is not None:
                timing.end()
            work = getattr(self, '_kvmem_public_work', None)
            if work is not None:
                work.poll()
            return result

        def _kvmem_execute_public_forward(self, scheduler_output, *, observe_work, **kwargs):
            # Observe the fresh native execution state, after a successful
            # target forward. Private maintenance uses prepare_attn directly,
            # and must never count here. This also counts prefill chunks that
            # produce no output tokens or do not reach sample_tokens.
            before = getattr(self, 'execute_model_state', None)
            runtime = getattr(self, '_kvmem_capture_runtime', None)
            refresh = getattr(self, '_kvmem_async_refresh', None)
            cobatch = getattr(refresh, 'cobatch', None)
            if (cobatch is not None and observe_work and not kwargs.get('dummy_run') and
                    not kwargs.get('is_profile') and cobatch.arm(scheduler_output)):
                # Native dispatch then uses an unpadded NONE descriptor. The
                # model's existing torch.compile path remains enabled; only
                # replaying a public-only FULL graph is excluded for this turn.
                kwargs['is_profile'] = True
            self._kvmem_public_action_forward = observe_work
            try:
                try:
                    result = super().execute_model(scheduler_output, **kwargs)
                finally:
                    self._kvmem_public_action_forward = False
                state = getattr(self, 'execute_model_state', None)
                if cobatch is not None and cobatch.frame is not None:
                    state = cobatch.complete(state)
                    self.execute_model_state = state
                fresh = observe_work and state is not None and state is not before
                work = getattr(self, '_kvmem_public_work', None)
                if fresh and work is not None:
                    adapter = getattr(self, '_kvmem_window_adapter', None)
                    work.submit(state.input_batch, getattr(adapter, 'offsets', None))
                if runtime is not None and getattr(runtime, '_prefill_scatter_deferred', None) is not None:
                    # Native execute_model has queued the complete target;
                    # sampling and MTP still follow. Starting the old confirmed
                    # packet here cannot compete with its first kernel launches.
                    runtime.resume_prefill_scatter(after_target=True,
                        batch=state.input_batch if fresh else None)
                return result
            except BaseException as error:
                if cobatch is not None:
                    cobatch.abort(error)
                if runtime is not None and getattr(runtime, '_prefill_scatter_deferred', None) is not None:
                    runtime.abort_prefill_scatter(error)
                raise

        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False, context_len=0, valid_dummy_state_slots=False, **kwargs):
            try:
                return self._kvmem_execute_model(scheduler_output, intermediate_tensors,
                    dummy_run, skip_attn_for_dummy_run, is_profile, context_len,
                    valid_dummy_state_slots, **kwargs)
            except BaseException as error:
                runtime = getattr(self, '_kvmem_capture_runtime', None)
                if runtime is not None and getattr(runtime, '_prefill_scatter_deferred', None) is not None:
                    runtime.abort_prefill_scatter(error)
                raise

        def _kvmem_execute_model(self, scheduler_output, intermediate_tensors=None,
                                dummy_run=False, skip_attn_for_dummy_run=False,
                                is_profile=False, context_len=0, valid_dummy_state_slots=False, **native_kwargs):
            kwargs = dict(intermediate_tensors=intermediate_tensors, dummy_run=dummy_run,
                          skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          is_profile=is_profile, context_len=context_len,
                          valid_dummy_state_slots=valid_dummy_state_slots, **native_kwargs)
            timing = getattr(self, '_kvmem_step_timing', None)
            if (timing is not None and not dummy_run and not is_profile and
                    scheduler_output.total_num_scheduled_tokens and
                    (self._kvmem_probe_output / 'armed').is_file()):
                timing.begin(getattr(scheduler_output.kv_connector_metadata, 'cost_classification', None))
            runtime = getattr(self, "_kvmem_capture_runtime", None)
            refresh = getattr(self, '_kvmem_async_refresh', None)
            # Every forward replaces this receipt, including ordinary 2K and
            # idle steps. Worker planning binds it to its own authority/window
            # generations before any grouped target computation is submitted.
            self._kvmem_logical_prefill_groups = (dict(getattr(
                scheduler_output.kv_connector_metadata, 'logical_prefill_groups', {}))
                if not dummy_run and not is_profile else {})
            self._kvmem_logical_prefill_step = (getattr(
                scheduler_output.kv_connector_metadata, 'logical_prefill_step', None)
                if not dummy_run and not is_profile else None)
            if self._kvmem_logical_prefill_groups and (runtime is None or runtime.action is None):
                raise RuntimeError('Physical prefill cannot precede Action runtime initialization')
            terminal = None
            if not dummy_run and runtime is not None:
                if getattr(self, '_kvmem_bounded_target', False):
                    from .connector import ControlMetadata
                    control = scheduler_output.kv_connector_metadata
                    if not isinstance(control, ControlMetadata):
                        raise RuntimeError('Bounded capture requires scheduler control metadata')
                    terminal = control.terminal_prefixes
                # Only the next scheduler decision can authorize the previous
                # sample. Install EOS/cancel clipping before publication, while
                # leaving unrelated request writers free to continue.
                runtime.begin_step(scheduler_output.finished_req_ids, terminal)
            if not dummy_run and refresh is not None:
                scheduler_output = refresh.defer_terminals(scheduler_output)
            if not dummy_run and runtime is not None:
                # A retiring owner may still be publishing a large prefill.
                # Keep its native index until its own history is ready, while
                # allowing the rest of this public batch to proceed.
                scheduler_output = runtime.defer_terminals(scheduler_output)
                if getattr(self, '_kvmem_bounded_target', False):
                    terminal = scheduler_output.kv_connector_metadata.terminal_prefixes
            sessions = getattr(self, '_kvmem_sessions', None)
            if not dummy_run and getattr(self, '_kvmem_serving', False):
                from .connector import ControlMetadata
                if isinstance(scheduler_output.kv_connector_metadata, ControlMetadata):
                    # Native warmup disables the connector and has no metadata.
                    # Arm at the first real scheduler message, before any client
                    # forward; API readiness polling must not create a capture gap.
                    if not (self._kvmem_probe_output / 'armed').is_file():
                        (self._kvmem_probe_output / 'armed').touch()
            work = getattr(self, '_kvmem_public_work', None)
            work_armed = not dummy_run and not is_profile and (self._kvmem_probe_output / 'armed').is_file()
            if work is not None and work_armed:
                from .work_counters import public_work_idle
                # A zero-token publication/refresh turn still has live owners.
                # Only true retirement/idle flushes the final short interval;
                # all turns continue polling completion without synchronizing.
                request_ids = getattr(getattr(self, 'req_states', None), 'req_id_to_index', ())
                work.poll(flush=public_work_idle(scheduler_output, request_ids))
            observe_work = work_armed and bool(scheduler_output.total_num_scheduled_tokens)
            # V2 kernel warmup invokes ordinary execute_model with synthetic
            # requests and no connector metadata. Session control starts at the
            # same explicit post-warmup boundary as accepted authority capture.
            if (not dummy_run and sessions is not None
                    and (self._kvmem_probe_output / "armed").is_file()):
                from .connector import ControlMetadata
                control = scheduler_output.kv_connector_metadata
                if not isinstance(control, ControlMetadata):
                    raise RuntimeError('Prompt sessions require native connector metadata')
                sessions.receive(control.sessions)
            if not dummy_run and runtime is not None:
                # Apply EOS/cancellation truncation before draining the latest
                # accepted sample packet into the immutable authority.
                runtime.finish(scheduler_output.finished_req_ids, terminal)
                defer_scatter = (not is_profile and work_armed and
                                 getattr(getattr(runtime, 'options', None), 'prefill_scatter_after_target', False) and
                                 runtime.defer_prefill_scatter(scheduler_output, refresh))
                if defer_scatter:
                    runtime.advance_publication(scheduler_output.num_scheduled_tokens, refresh,
                                                defer_scatter=True)
                else:
                    runtime.advance_publication(scheduler_output.num_scheduled_tokens, refresh)
            if not dummy_run and getattr(self, "_kvmem_window_adapter", None) is not None:
                for request_id in scheduler_output.finished_req_ids:
                    self._kvmem_window_adapter.free(request_id)
            if (getattr(self, "_kvmem_scheduled", False) and not dummy_run
                    and (self._kvmem_probe_output / "armed").is_file()):
                from vllm.distributed.kv_transfer import get_kv_transfer_group
                get_kv_transfer_group().runtime = self._kvmem_refresh
            if (getattr(self, "_kvmem_bounded_target", False) and not dummy_run
                    and (self._kvmem_probe_output / "armed").is_file()):
                from vllm.distributed.kv_transfer import get_kv_transfer_group
                connector = get_kv_transfer_group()
                connector.session_worker = sessions
                connector.restore_completion_poll = (
                    refresh.poll_restore_completions if refresh is not None and
                    getattr(refresh.options, 'restore_handoff_fastpath', False) else None)
                if timing is not None:
                    connector.step_timings = timing.poll
                if refresh is not None and refresh.maintenance is not None:
                    connector.restore_timings = refresh.maintenance.pop_restore_timings
                connector.runtime = self._kvmem_query_refresh
                connector.retirement_done = (runtime.retirement_done if runtime is not None else lambda _: True)
                # The first real forward constructs the capture runtime only
                # after this setup. Resolve it at feedback time so its first
                # large packet is also excluded from premature rescheduling.
                connector.publication_snapshot = lambda: (
                    self._kvmem_capture_runtime.publication_blocked_requests()
                    if self._kvmem_capture_runtime is not None else set())
                if getattr(self, '_kvmem_cpu_history_coordination', False):
                    connector.history_usage = lambda: (
                        self._kvmem_capture_runtime.archive.history_used_bytes()
                        if self._kvmem_capture_runtime is not None else 0)
                if getattr(self, '_kvmem_repository_gc', False):
                    connector.archive_snapshot = lambda: (
                        self._kvmem_capture_runtime.archive_capacity()
                        if self._kvmem_capture_runtime is not None else {})
            if (dummy_run or
                    not scheduler_output.total_num_scheduled_tokens or
                    (getattr(self, "_kvmem_probe_done", True) and not self._kvmem_continuous) or
                    not (self._kvmem_probe_output / "armed").is_file()):
                return self._kvmem_execute_public_forward(scheduler_output, observe_work=observe_work, **kwargs)
            captured = []
            if getattr(self, '_kvmem_graph', False):
                from .graph_runtime import needs_refresh_batch
                if needs_refresh_batch(scheduler_output):
                    # Native dispatch uses need_eager for this flag. The real
                    # batch still executes all GDN groups and normal sampling.
                    kwargs['is_profile'] = True
            with capture_to(captured.append):
                result = self._kvmem_execute_public_forward(scheduler_output, observe_work=observe_work, **kwargs)
            if getattr(self, '_kvmem_graph', False):
                captured = self._kvmem_graph_captures
            if not captured:
                raise RuntimeError("Real native batch executed without any KVMem captures")
            if self._kvmem_continuous:
                if self._kvmem_pending_capture is not None:
                    raise RuntimeError("Previous target capture never reached native sampling")
                if self._kvmem_capture_runtime is None:
                    from .capture_runtime import CaptureRuntime
                    self._kvmem_capture_runtime = CaptureRuntime(self, self._kvmem_probe_output)
                self._kvmem_pending_capture = (captured, self._kvmem_last_batch)
            if self._kvmem_probe_done:
                return result
            self._kvmem_probe_done = True
            if self._kvmem_scheduled:
                if self._kvmem_last_batch.num_reqs != 1:
                    raise ValueError("First scheduled probe must be a single request")
                self._kvmem_saved_capture = (captured, self._kvmem_last_batch.req_ids[0])
            output = self._kvmem_probe_output
            output.mkdir(parents=True, exist_ok=True)
            rebuild = None
            if os.getenv("KVMEM_REBUILD_PROBE") == "1":
                from .refresh_probe import rebuild_native_batch
                rebuild = rebuild_native_batch(self, captured, output)
            # Diagnostic-only synchronized export, once after the first real batch.
            payload = {c.name: {"positions": c.positions.cpu(), "q": c.query.cpu(),
                                "k": c.key.cpu(), "v": c.value.cpu()}
                       for c in captured}
            torch.save(payload, output / "target_capture.pt")
            report = {"scope": "native target capture; refresh qualification is recorded separately",
                      "layers_expected": self._kvmem_layers,
                      "layers_captured": list(payload),
                      "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                      "requests": scheduler_output.num_scheduled_tokens,
                      "rebuild": rebuild,
                      "finite": all(torch.isfinite(t).all().item() for c in payload.values() for t in c.values())}
            (output / "capture.json").write_text(json.dumps(report, indent=2))
            if report["layers_captured"] != self._kvmem_layers or not report["finite"]:
                raise RuntimeError("Incomplete or non-finite native target capture")
            if rebuild is not None and not rebuild["all_byte_equal"]:
                raise RuntimeError("Native-page archive/rebuild differs from original cache bytes")
            return result

    return KVMemProbeRunner
