"""Native scheduler subclass for transaction suspension and atomic commit.

Experimental bounded target mode clips admission and schedules semantic refresh.
Identity-refresh qualification remains separately available.
"""
import os
import json
import time
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from .connector import KVMemConnector, RefreshCommand, RestoreCommand, ControlMetadata
from .options import RuntimeOptions
from .window_manager import WorkingSetManager
from .admission import QueryPolicy


class _Complete:
    def query(self):
        return True


class KVMemScheduler(Scheduler):
    def _waiting_queues(self):
        # Newer native schedulers keep KV holders in a separate priority queue.
        return tuple(getattr(self, name) for name in
                     ('kv_holding_waiting', 'waiting', 'skipped_waiting')
                     if hasattr(self, name))

    def _waiting_requests(self):
        return [request for queue in self._waiting_queues() for request in queue]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.connector, KVMemConnector):
            raise ValueError("KVMemScheduler requires KVMemConnector")
        if self.scheduler_config.async_scheduling:
            raise ValueError("Scheduler refresh transactions require synchronous execution")
        self.draft_contract = None
        if self.vllm_config.speculative_config is not None:
            if (os.getenv("KVMEM_BOUNDED_DFLASH") != "1" and os.getenv("KVMEM_BOUNDED_MTP") != "1") or os.getenv("KVMEM_BOUNDED_TARGET") != "1":
                raise ValueError("Bounded speculation requires its explicit experimental opt-in")
            from .draft_contract import DraftContract
            self.draft_contract = DraftContract.from_vllm(
                self.vllm_config, RuntimeOptions.from_vllm(self.vllm_config).memory)
        self._bounded_target = os.getenv("KVMEM_BOUNDED_TARGET") == "1"
        if not self._bounded_target and os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") != "1":
            raise ValueError("Compressed runtime not yet enabled; scheduler currently exposes the explicit identity probe")
        if self._bounded_target:
            from .engine_api import require_contract
            require_contract('scheduler')
        options = RuntimeOptions.from_vllm(self.vllm_config)
        self._deferred_prefill_validation = options.prefill_deferred_validation
        self.connector.request_plan_memory = options.memory if options.request_planning else None
        self.connector.request_plan_pressure_policy = options.pressure_policy
        if options.rolling_active_window and not (self.draft_contract is not None and (
                self.draft_contract.method == 'mtp' and options.async_refresh and options.async_restore or
                self.draft_contract.method == 'dflash' and not options.async_refresh and not options.async_restore)):
            raise ValueError('Rolling active window requires asynchronous MTP or '
                             'synchronous DFlash2 accepted-prefix admission')
        if options.async_archive_retirement and not self._bounded_target:
            raise ValueError('Asynchronous archive retirement requires bounded execution')
        config = options.memory
        self.archive_admission = None
        if options.repository_gc:
            if not self._bounded_target:
                raise ValueError('Serving repository GC requires bounded execution')
            from .repository_gc import serving_gc_config
            serving_gc_config(config, self.scheduler_config.max_num_batched_tokens,
                              checkpoint_frames=int(bool(options.session_capacity)))
            from .archive_admission import ArchiveAdmission
            self.archive_admission = ArchiveAdmission(config.block_tokens, config.repository_blocks)
            self.connector.scheduler_archive_update = lambda sequence, capacity: self.archive_admission.update(
                sequence, capacity, {k for k, r in self.requests.items() if not r.is_finished()})
        self.prefill_cadence = options.prefill_cadence
        self.request_planning = options.request_planning
        self.async_refresh = options.async_refresh
        self.async_restore = options.async_restore
        self.rolling_active_window = options.rolling_active_window
        self.guided_options = options
        self._guided_tokenizer = None
        self.prefill_controller = None
        self.prefill_batch = None
        self.prefill_physical_coalescing = options.prefill_physical_coalescing
        self.prefill_physical_max_chunks = options.prefill_physical_max_chunks
        self._logical_prefill_groups = {}
        if options.prefill_batch_tokens:
            from .prefill_control import BatchPrefillBudget
            if options.prefill_batch_tokens < config.block_tokens * self.scheduler_config.max_num_seqs:
                raise ValueError('Shared prefill budget must fit one block per active request')
            self.prefill_batch = BatchPrefillBudget(options.prefill_batch_tokens, config.block_tokens,
                per_request_tokens=getattr(options, 'prefill_request_tokens', 0))
        if options.adaptive_prefill:
            from .prefill_control import PrefillController
            self.prefill_controller = PrefillController(options)
            self.connector.scheduler_step_feedback = self.prefill_controller.observe
        self._restore_sequence = 0
        self._deferred_native_padding = {}
        self.prepared_refreshes = {}
        self.session_capture_decode = options.session_capture_decode
        self._mixed_steps = 0
        self._scheduled_queries = {"prefill": {}, "decode": {}}
        self._cadence_requested = 0
        self.query_policy = QueryPolicy(config.block_tokens, config.select_budget, config.gen_budget)
        if self.draft_contract is not None:
            from .speculative_admission import SpeculativeQueryPolicy
            self.query_policy = SpeculativeQueryPolicy(config.block_tokens, config.select_budget,
                                                       config.gen_budget, self.draft_contract.query_tokens)
        if self._bounded_target and not config.block_tokens <= self.scheduler_config.max_num_batched_tokens <= (
                config.gen_budget + self.scheduler_config.max_num_seqs * (self.draft_contract.query_tokens if self.draft_contract else 1)):
            raise ValueError("Bounded target batch budget must fit an aligned block and G headroom")
        coordinator = self.kv_cache_manager.coordinator
        managers = list(coordinator.single_type_managers)
        self.window_managers = {}
        for group_id, manager in enumerate(managers):
            if type(manager) is FullAttentionManager:
                managers[group_id] = WorkingSetManager(manager, config.select_budget, config.gen_budget)
                self.window_managers[group_id] = managers[group_id]
        if not self.window_managers:
            raise ValueError("No full-attention cache group found")
        coordinator.single_type_managers = tuple(managers)
        if self._bounded_target:
            from .capacity import ReservedKVCacheManager
            self.kv_cache_manager = ReservedKVCacheManager(self.kv_cache_manager, self.vllm_config,
                draft_contract=self.draft_contract,
                max_authorities=self.scheduler_config.max_num_seqs if options.async_archive_retirement else None)
        self.connector.scheduler_completion = self._complete_refresh
        if self._bounded_target:
            self.kv_cache_manager.guided_reserve = options.guided_reserve
            self.kv_cache_manager.guided_block_tokens = config.block_tokens
            self.kv_cache_manager.guided_refresh_control = self._guided_refresh
        self.transactions = {}
        if self._bounded_target:
            self.kv_cache_manager.prefill_refresh_control = self._prefill_refresh_control
        self.table_updates = {}
        self._probe_submitted = False
        if options.session_capacity:
            from .session_control import SessionLedger
            def prepare_import(request_id, history, window):
                prepared = []
                try:
                    for manager in self.window_managers.values():
                        manager.prepare_import(request_id, history, window)
                        prepared.append(manager)
                    if self.archive_admission is not None:
                        from .archive_admission import ArchiveCapacity
                        ready = self.connector.sessions.requests[request_id].ready_records
                        if not 0 <= ready <= config.repository_blocks:
                            raise ValueError('Imported checkpoint exceeds repository capacity')
                        self.archive_admission.snapshots[request_id] = ArchiveCapacity(history, ready)
                except Exception:
                    for manager in prepared:
                        manager.abort_import_preparation(request_id)
                    raise
            def abort_import(request_id):
                for manager in self.window_managers.values():
                    manager.abort_import_preparation(request_id)
                if self.archive_admission is not None:
                    self.archive_admission.forget(request_id)
            def commit_import(request_id, history):
                for manager in self.window_managers.values():
                    manager.validate_import(request_id, history)
                for manager in self.window_managers.values():
                    if not manager.commit_import(request_id, history, _Complete()):
                        raise RuntimeError('Acknowledged prompt import did not commit')
            self.connector.sessions = SessionLedger(options.session_capacity, config.block_tokens,
                                                    prepare_import, abort_import, commit_import,
                                                    terminal_capture=options.session_capture_terminal,
                                                    select_budget=config.select_budget if options.request_planning else None,
                                                    gen_budget=config.gen_budget if options.request_planning else None,
                                                    checkpoint_policy=options.session_checkpoint_policy,memory_config=config,
                                                    preserve_history=options.cpu_history_coordination,
                                                    require_owned=options.graph_mode != 'off',
                                                    pressure_policy=options.pressure_policy,
                                                    freeze_prompt=options.session_freeze_prompt,
                                                    guided_options=options)
        if options.cpu_history_coordination:
            from .coordination import HistoryAdmission
            from .cache_manifest import archive_bytes_per_token
            from .query_window_budget import split_cpu_history_budget
            from .historical_feature_budget import split_historical_feature_budget
            history_bytes, _ = split_cpu_history_budget(options)
            history_bytes, _ = split_historical_feature_budget(options, history_bytes,
                self.scheduler_config.max_num_seqs + options.session_capacity)
            row_bytes = archive_bytes_per_token(self.kv_cache_config)
            record_bytes = row_bytes * config.block_tokens
            capacity = min(history_bytes,
                record_bytes * config.repository_blocks * (self.scheduler_config.max_num_seqs + options.session_capacity))
            if options.historical_gc is not None:
                from .historical_admission import HistoricalHistoryAdmission
                from .historical_gc import historical_record_charge
                record_bytes = historical_record_charge(record_bytes, config.block_tokens)
                capacity = min(history_bytes,
                    record_bytes*config.repository_blocks*(self.scheduler_config.max_num_seqs+options.session_capacity))
                self.connector.history_admission = HistoricalHistoryAdmission(capacity, record_bytes,
                    config.repository_blocks, config.block_tokens, self.connector.sessions,
                    lookahead_tokens=self.draft_contract.query_tokens if self.draft_contract else 1)
            else:
                self.connector.history_admission = HistoryAdmission(capacity, config.block_tokens,
                    row_bytes, self.vllm_config.model_config.max_model_len,
                    reserve_progress=options.cpu_progress_reserve)
        if options.active_residency_enabled:
            if self.draft_contract is not None and self.draft_contract.method != 'mtp':
                raise ValueError('Idle active residency currently qualifies target-only or native MTP')
            from .active_residency import ActiveResidency
            residency = ActiveResidency(self.window_managers, self.kv_cache_manager.block_pool,
                options.active_resident_ttl_ms, options.active_resident_max_fraction,
                pressure_only=options.active_resident_pressure_only)
            self.connector.active_residency = self.kv_cache_manager.active_residency = residency
            self.kv_cache_manager.resident_entry = self.connector.sessions.requests.get
        if options.async_restore:
            from .restore_admission import RestoreAdmission
            self.connector.restore_admission = self.kv_cache_manager.restore_admission = RestoreAdmission(
                residency=self.connector.active_residency,
                resident_slots=getattr(options, 'resident_restore_slots', 1),
                cold_busy=lambda: any(isinstance(command, RefreshCommand)
                                      for command, _ in self.transactions.values()),
                output=Path(os.environ['KVMEM_RUNNER_PROBE_DIR']) / 'restore_admission.jsonl')
        self.connector.publication_live = lambda: {
            rid for rid, request in self.requests.items() if request.status is RequestStatus.RUNNING}
        self.offline_policy = None
        self.prefill_coalescer = None
        if options.offline_agent_policy:
            if not hasattr(Scheduler, '_select_waiting_queue_for_scheduling'):
                raise ValueError('Offline agent ordering requires a native waiting-queue selection hook; '
                                 'disable offline_agent_policy with this engine')
            from vllm.v1.core.sched.request_queue import SchedulingPolicy, FCFSRequestQueue
            from .offline_policy import OfflineAgentPolicy
            if self.policy != SchedulingPolicy.FCFS or any(type(q) is not FCFSRequestQueue for q in self._waiting_queues()):
                raise ValueError('Offline agent ordering requires the qualified native FCFS queues')
            self.offline_policy = OfflineAgentPolicy(self, options,
                Path(os.environ['KVMEM_RUNNER_PROBE_DIR']) / 'offline_decisions.jsonl')
            self.connector.scheduler_step_feedback = self._observe_offline_steps
            self.connector.scheduler_restore_feedback = self.offline_policy.costs.observe_restores
        if options.offline_prefill_coalescing:
            from .prefill_coalescing import OfflinePrefillCoalescer
            baseline = int(os.environ['KVMEM_PREFILL_CHUNK_TOKENS'])
            self.prefill_coalescer = OfflinePrefillCoalescer(baseline,
                options.offline_prefill_max_tokens, config.block_tokens)
            self._coalescing_max_tokens = options.offline_prefill_max_tokens
        self.phase_policy = None
        self._phase_decision = None
        if options.offline_phase_policy:
            from .phase_policy import PhasePolicy
            self.phase_policy = PhasePolicy(self.scheduler_config.max_num_seqs,
                options.phase_prefill_wave_tokens, options.phase_max_wait_ms)

    def begin_refresh(self, request, window_tokens, *, replay_tokens=0, query_span=None, selection='chunk',mandatory_spans=(), input_tokens=(), asynchronous=False, speculative_padding=0, message_spans=(), locality_spans=(), locality_provenance=(), guided_reason='', guided_private=False, guided_boundary=None):
        options = getattr(self, 'guided_options', None)
        if (selection == 'pressure' and request.num_computed_tokens < request.num_prompt_tokens
                and getattr(options, 'guided_reselect', None) == 'prefill'
                and getattr(options, 'guided_query_tokens', 0)):
            if not asynchronous or not input_tokens:
                raise ValueError('Guided prefill pressure requires a replayable asynchronous transaction')
            guided_reason, guided_private, guided_boundary = 'prefill-pressure', True, False
        if getattr(request, "num_in_flight_tokens", 0):
            raise RuntimeError("Wait for all target/speculation work before refreshing")
        if request.request_id in self.transactions or request.request_id in self.connector.pending:
            raise RuntimeError("Request already has a refresh transaction")
        prepared = []
        try:
            for group_id, manager in self.window_managers.items():
                prepared.append((group_id, manager.prepare_refresh(
                    request.request_id, request.num_computed_tokens, window_tokens,
                    replay_tokens=replay_tokens, allow_import=bool(self.connector.sessions and
                        self.connector.sessions.import_scheduled(request.request_id)))))
        except Exception:
            for group_id, ticket in prepared:
                self.window_managers[group_id].abort_preparation(ticket)
            raise
        groups = tuple(prepared)
        command = RefreshCommand(request.request_id, groups[0][1].ticket_id, groups,query_span,selection,mandatory_spans,
                                 tuple(input_tokens), asynchronous, speculative_padding, tuple(message_spans),
                                 tuple(locality_spans), tuple(locality_provenance),
                                 guided_reason, guided_private, guided_boundary,
                                 pressure_stage=('prefill' if request.num_computed_tokens < request.num_prompt_tokens
                                                 else 'decode') if selection == 'pressure' else '')
        self.transactions[request.request_id] = (command, request.next_decode_eligible_step)
        request.next_decode_eligible_step = 2**62
        self.connector.queue(command)

    def _guided_refresh(self, request, history, offset):
        epoch = getattr(request, '_kvmem_guided_epoch', None)
        if epoch is None:
            return None
        from .request_plan import request_plan
        from .guided_refresh import scheduled_refresh
        plan = request_plan(request, self.query_policy.block_tokens, self.query_policy.gen_budget)
        if (epoch.decode_enabled and history >= request.num_prompt_tokens and
                history >= epoch.deadline and history % self.query_policy.block_tokens == 0 and
                self._guided_tokenizer is None):
            from vllm.tokenizers import get_tokenizer
            cfg = self.vllm_config.model_config
            self._guided_tokenizer = get_tokenizer(cfg.tokenizer, tokenizer_mode=cfg.tokenizer_mode,
                                                   trust_remote_code=cfg.trust_remote_code)
        return scheduled_refresh(request, plan, history, offset, self.query_policy.capacity,
                                 self._guided_tokenizer)

    def _snapshot_cadence(self, name):
        from .snapshot_diagnostics import SnapshotCadence
        cadences = getattr(self, '_snapshot_cadences', None)
        if cadences is None:
            self._snapshot_cadences = cadences = {}
        if name not in cadences:
            cadences[name] = SnapshotCadence()
        return cadences[name]

    def _lifecycle_snapshot_identity(self):
        gate = self.kv_cache_manager
        sessions = self.connector.sessions
        entries = getattr(sessions, 'entries', {})
        # No block-table walk or historical-token hash on the steady path.
        # Request/session transitions must publish immediately so cancellation
        # and exact-retirement gates never wait for another model step.
        session_states = tuple((key, entry.generation, entry.request_id, entry.phase,
            entry.history, entry.window, id(entry.tokens), id(entry.active_records))
            for key, entry in entries.items())
        quiescent = not self.requests and not self.transactions and not self.connector.pending
        return (tuple(self.requests), tuple(self.transactions), tuple(self.connector.pending),
                tuple(gate.reservations), session_states,
                gate.block_pool.get_num_free_blocks() if quiescent else None,
                getattr(self, '_lifecycle_finish_generation', 0))

    def _write_lifecycle_snapshot(self, *, force=False):
        if not getattr(self, "_bounded_target", False) or os.getenv("KVMEM_LIFECYCLE_TRACE") != "1":
            return False
        if not self._snapshot_cadence('resources').due(self._lifecycle_snapshot_identity(), force=force):
            return False
        gate = self.kv_cache_manager
        owned = sum(not block.is_null for manager in gate.coordinator.single_type_managers
                    for blocks in manager.req_to_blocks.values() for block in blocks)
        report = {"at": time.perf_counter(), "requests": len(self.requests),
            "refresh_pending": len(self.transactions), "connector_pending": len(self.connector.pending),
            "reservations": list(gate.reservations), "reservation_rejections": gate.rejections,
            "authority_limit": gate.max_authorities, "authority_rejections": gate.authority_rejections,
            "owned_blocks": owned,
            "archive_capacity_deferrals": self.archive_admission.deferred if getattr(self, 'archive_admission', None) else 0,
            "free_blocks": gate.block_pool.get_num_free_blocks(),
            "total_blocks": gate.block_pool.num_gpu_blocks}
        if getattr(self, 'prefill_controller', None) is not None:
            report['adaptive_prefill'] = self.prefill_controller.snapshot()
        if getattr(self, 'prefill_batch', None) is not None:
            report['shared_prefill'] = self.prefill_batch.snapshot()
        if getattr(self.connector, 'active_residency', None) is not None:
            report['active_residency'] = self.connector.active_residency.snapshot()
        if getattr(self.connector, 'restore_admission', None) is not None:
            report['restore_admission'] = self.connector.restore_admission.snapshot()
        report['publication_blocked'] = sorted(getattr(self.connector, 'publication_blocked', ()))
        if getattr(self, 'offline_policy', None) is not None:
            report['offline_policy'] = self.offline_policy.snapshot()
        if getattr(self, 'phase_policy', None) is not None:
            report['phase_policy'] = self.phase_policy.snapshot()
        if getattr(self, 'prefill_coalescer', None) is not None:
            report['prefill_coalescing'] = self.prefill_coalescer.snapshot()
        sessions = self.connector.sessions
        if getattr(self.connector, 'history_admission', None) is not None:
            report['cpu_history_admission'] = self.connector.history_admission.snapshot()
        if sessions is not None:
            report['sessions'] = dict(entries=len(sessions.entries), hits=sessions.hits,
                                      misses=sessions.misses, waits=sessions.waits)
            report['session_views'] = sessions.snapshot()
            report['session_owners'] = sessions.owner_snapshot()
        path = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"])
        temporary = path / "scheduler_resources.tmp"
        temporary.write_text(json.dumps(report))
        temporary.replace(path / "scheduler_resources.json")
        return True

    def _write_scheduled_queries(self, *, force=False):
        if not hasattr(self, '_scheduled_queries'):
            return
        # New query shapes publish immediately; all cumulative counts remain
        # exact in memory and are flushed at lifecycle changes and shutdown.
        identity = tuple((stage, tuple(histogram)) for stage, histogram in self._scheduled_queries.items())
        if not self._snapshot_cadence('queries').due(identity, force=force):
            return
        path = Path(os.environ['KVMEM_RUNNER_PROBE_DIR'])
        temporary = path / 'scheduled_queries.tmp'
        temporary.write_text(json.dumps({"queries": self._scheduled_queries,
            "prefill_cadence": self.prefill_cadence,
            "archive_capacity_deferrals": self.archive_admission.deferred if getattr(self, 'archive_admission', None) else 0,
            "throttle_requests": self._cadence_requested}))
        temporary.replace(path / 'scheduled_queries.json')

    def update_from_output(self, *args, **kwargs):
        rejected = self._finish_capacity_rejections()
        result = super().update_from_output(*args, **kwargs)
        if rejected:
            from vllm.v1.engine import EngineCoreOutputs
            for client, outputs in rejected.items():
                if client not in result:
                    result[client] = EngineCoreOutputs(outputs=[])
                result[client].outputs.extend(outputs)
        # Include native deferred frees and connector completions from this
        # output. A snapshot at schedule() alone can precede the final free.
        changed = self._write_lifecycle_snapshot()
        self._write_scheduled_queries(force=changed or not self.requests)
        return result

    def _finish_capacity_rejections(self):
        """Use native error completion after its waiting traversal has ended.

        The pinned synchronous EngineCore.step always calls update_from_output
        for an empty schedule too. Invalid waiters therefore receive a terminal
        error without any model forward and without waiting for a valid peer.
        """
        from vllm.v1.engine import EngineCoreOutput
        pending = getattr(self.connector, 'capacity_rejections', {})
        errors = dict(pending)
        pending.clear()
        if not errors:
            return {}
        outputs = {}
        for request in self.finish_requests(errors, RequestStatus.FINISHED_ERROR):
            outputs.setdefault(request.client_index, []).append(EngineCoreOutput(
                request_id=request.request_id, new_token_ids=[],
                finish_reason=request.get_finished_reason(), stop_reason=errors[request.request_id],
                events=request.take_events(), trace_headers=request.trace_headers))
        return outputs

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            self._write_lifecycle_snapshot(force=True)
            self._write_scheduled_queries(force=True)

    def _free_request(self, request, delay_free_blocks=False):
        getattr(self.connector, 'publication_blocked', set()).discard(request.request_id)
        restore_admission = getattr(self.connector, 'restore_admission', None)
        if restore_admission is not None:
            restore_admission.cancel(request)
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        self._lifecycle_finish_generation = getattr(self, '_lifecycle_finish_generation', 0) + 1
        getattr(self, 'prepared_refreshes', {}).pop(request.request_id, None)
        getattr(self, '_deferred_native_padding', {}).pop(request.request_id, None)
        if getattr(self, 'archive_admission', None) is not None:
            self.archive_admission.forget(request.request_id)
        if getattr(self, "_bounded_target", False) and os.getenv("KVMEM_LIFECYCLE_TRACE") == "1":
            report = {"request_id": request.request_id, "status": request.status.name,
                      "at": time.perf_counter(), "history_end": request.num_computed_tokens,
                      "in_flight_tokens": request.num_in_flight_tokens,
                      "refresh_pending": request.request_id in self.transactions,
                      "lifetime_reservation_retained": request.request_id in self.kv_cache_manager.reservations}
            with (Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "scheduler_finished.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
        return result

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        if getattr(self, '_deferred_prefill_validation', False):
            # Ordinary retirement retains pages until the worker ACK. Native
            # preemption bypasses that ACK, and a scheduler step fence does not
            # cover reference fallback reading the old prefix on another stream.
            raise RuntimeError('Deferred prefill validation forbids native request preemption')
        return super()._preempt_request(request, timestamp, drop_stale_output)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if (getattr(self, '_deferred_prefill_validation', False) and
                reset_running_requests and self.requests):
            # Reject before native reset pops running requests or frees any
            # blocks. Finished-but-retiring requests still hold worker leases.
            return False
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def _request_blocks_can_be_freed(self, request):
        if request.request_id in self.transactions:
            return False
        if getattr(self, "_bounded_target", False) and not request.is_finished():
            # No silent recompute/preemption of authoritative GDN history. New
            # requests wait until capacity is released by an existing request.
            return False
        return super()._request_blocks_can_be_freed(request)

    def _complete_refresh(self, request_id, command_id, selected_tokens=None):
        command, previous_step = self.transactions[request_id]
        if command.command_id != command_id:
            raise ValueError("Refresh command identity mismatch")
        request = self.requests[request_id]
        if isinstance(command, RestoreCommand):
            current = tuple(tuple(b.block_id for b in manager.req_to_blocks.get(request_id, ()))
                for manager in self.kv_cache_manager.coordinator.single_type_managers)
            if current != command.block_ids or any(request_id in m.imports for m in self.window_managers.values()):
                raise RuntimeError('Restore acknowledged before import commit or after page ownership changed')
            if getattr(self.connector, 'restore_admission', None) is not None:
                self.connector.restore_admission.complete(request_id, command_id)
            request.next_decode_eligible_step = previous_step
            del self.transactions[request_id]
            return
        if command.asynchronous and request.is_finished():
            # The worker has fenced every private forward and transfer. No
            # replay will follow; preserve the original allocation until free.
            for group_id, ticket in command.groups:
                self.window_managers[group_id].validate_refresh(ticket)
            for group_id, ticket in command.groups:
                self.window_managers[group_id].abort_preparation(ticket)
            request.next_decode_eligible_step = previous_step
            del self.transactions[request_id]
            return
        # Validate every group before freeing any pages; no partial commit on a
        # stale ticket or unexpected native allocation change.
        for group_id, ticket in command.groups:
            self.window_managers[group_id].validate_refresh(ticket, selected_tokens)
        for group_id, ticket in command.groups:
            if not self.window_managers[group_id].commit_refresh(ticket, _Complete(), selected_tokens):
                raise RuntimeError("Acknowledged refresh could not commit")
        if command.selection == 'query':
            epoch = getattr(request, '_kvmem_guided_epoch', None)
            if epoch is not None:
                history = command.groups[0][1].history_end
                # Initial explicit Query may precede the first dense K window.
                epoch.anchor = min(epoch.anchor, history)
                epoch.committed(history, automatic=bool(command.guided_reason))
                sessions = self.connector.sessions
                entry = sessions.requests.get(request_id) if sessions is not None else None
                if entry is not None:
                    entry.guided_anchor = history
                    if entry.candidate_position == history and not sessions.freeze_prompt:
                        entry.prompt_guided_anchor = history
        request = self.requests[request_id]
        request.next_decode_eligible_step = previous_step
        if not request.is_finished():
            self.table_updates[request_id] = tuple(
                [b.block_id for b in manager.req_to_blocks.get(request_id, [])]
                for manager in self.kv_cache_manager.coordinator.single_type_managers)
        del self.transactions[request_id]
        if command.asynchronous and command.groups[0][1].replay_tokens:
            self.prepared_refreshes[request_id] = (command.groups[0][1].history_end,
                                                   command.groups[0][1].replay_tokens)
        if getattr(self, "_bounded_target", False):
            report = {"request_id": request_id, "command_id": command_id,
                "history_begin": command.groups[0][1].history_end,
                "selected_tokens": selected_tokens,
                "replay_tokens": command.groups[0][1].replay_tokens,
                "groups": [{"group_id": group_id, "blocks": len(manager.req_to_blocks[request_id]),
                            "block_tokens": manager.block_size, "offset": manager.offsets[request_id]}
                           for group_id, manager in self.window_managers.items()]}
            with (Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "scheduler_refresh.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")

    def _prefill_refresh_control(self, request, tokens):
        """Prove that this allocation becomes an asynchronous pressure command."""
        rid = request.request_id
        history = request.num_computed_tokens
        if (not self.async_refresh or request.status is not RequestStatus.RUNNING or
                request.num_in_flight_tokens or not 0 < tokens <= request.num_prompt_tokens-history or
                rid in self.transactions or rid in self.connector.pending or
                rid in self.prepared_refreshes):
            return False
        offsets = {m.offsets.get(rid, 0) for m in self.window_managers.values()}
        if len(offsets) != 1:
            return False
        from .request_plan import request_plan
        plan = request_plan(request, self.query_policy.block_tokens, self.query_policy.gen_budget)
        return (plan is not None and
                plan.action(history, request.num_prompt_tokens, offsets.pop(), self.query_policy.capacity) == 'pressure' and
                history >= plan.selection_budget(history, request.num_prompt_tokens, self.query_policy.capacity))

    def _restore_control_reservation(self, request, history):
        """Recognize an exact import admitted before its native allocation.

        This allocation will be stripped from public forwards by connector
        metadata construction. A policy preview is not sufficient evidence:
        require both the exact lookup lease and its live restore permit.
        """
        if (not getattr(self, 'async_restore', False) or
                getattr(request, 'status', None) not in (RequestStatus.WAITING, RequestStatus.PREEMPTED)):
            return False
        sessions = self.connector.sessions
        gate = getattr(self.connector, 'restore_admission', None)
        if sessions is None or gate is None:
            return False
        entry = sessions.requests.get(request.request_id)
        permit = gate.permits.get(request.request_id)
        return (entry is not None and entry.request_id == request.request_id and
                entry.phase == 'lookup' and not entry.allocated and entry.history == history and history > 0 and
                permit is not None and permit.request is request and not permit.allocated and
                not permit.cancelled and permit.command_id is None and
                (permit.namespace, permit.generation, permit.history, permit.window) ==
                (entry.namespace, entry.generation, entry.history, entry.window))

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        if getattr(self, 'prefill_physical_coalescing', False):
            self._logical_prefill_groups.pop(request.request_id, None)
        logical_group = None
        if (request.request_id in getattr(self.connector, 'publication_blocked', ()) and
                request.status is RequestStatus.RUNNING):
            # Captured large prefill belongs to an already-running request.
            # The pinned native running loop skips only this row when zero is
            # returned; a zero-token control step still advances publication.
            return 0
        phase = getattr(self, '_phase_decision', None)
        phase_maintenance = request.request_id in getattr(self, '_phase_maintenance', ())
        if phase is not None:
            ready = (self._phase_prefills if num_computed_tokens < request.num_prompt_tokens
                     else self._phase_decodes)
            if request.request_id not in ready and not phase_maintenance:
                return 0
            prefill = num_computed_tokens < request.num_prompt_tokens
            if (phase.phase == 'decode' and prefill and not phase_maintenance or
                    phase.phase == 'prefill' and not prefill):
                return 0
        deferred = getattr(self, '_deferred_native_padding', {}).get(request.request_id)
        if deferred is not None:
            history, full_tokens = deferred
            if num_computed_tokens != history or request.spec_token_ids != [-1] * (full_tokens - 1):
                raise RuntimeError('Deferred native padding lost its exact unexecuted tail')
            if num_new_tokens < full_tokens:
                return 0  # Wait only this request until the original shape fits.
        tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        if not getattr(self, "_bounded_target", False):
            return tokens
        native_tokens = tokens
        padding = (getattr(self, 'num_spec_tokens', 0) > 0 and
                   request.num_prompt_tokens - num_computed_tokens == 1 and
                   tokens == 1 + self.num_spec_tokens)
        prepared = getattr(self, 'prepared_refreshes', {}).get(request.request_id)
        if prepared is not None:
            if prepared[0] != num_computed_tokens:
                raise RuntimeError('Prepared refresh public replay moved its history boundary')
            tokens = min(tokens, prepared[1])
        offsets = {m.offsets.get(request.request_id, 0) for m in self.window_managers.values()}
        if len(offsets) != 1:
            raise RuntimeError("Full-attention groups disagree on committed history mapping")
        offset = offsets.pop()
        from .request_plan import request_plan
        plan = request_plan(request, self.query_policy.block_tokens, self.query_policy.gen_budget)
        # Only the Query replay itself is atomic. Its following tool/history
        # suffix must return to the ordinary prefill cap, even if the original
        # Query remains attached after a P-checkpoint fallback.
        atomic_replay = (plan is not None and plan.query is not None and
                         plan.replay_begin <= num_computed_tokens < plan.replay_end)
        phase_budget = None
        if phase is not None and num_computed_tokens < request.num_prompt_tokens and not phase_maintenance:
            phase_budget = max(0, phase.prefill_budget - sum(self._phase_charges.values()) +
                               self._phase_charges.get(request.request_id, 0))
            if atomic_replay:
                if self._phase_atomic_overshoot and request.request_id not in self._phase_charges:
                    return 0
            elif padding:
                # The wave counts public input, while native owns all MTP
                # placeholder rows. Never clip the qualified 1+k tail shape.
                if phase_budget < 1:
                    self._phase_budget_blocked = True
                    return 0
            else:
                capped = min(tokens, phase_budget)
                if capped < tokens:
                    self._phase_budget_blocked = True
                tokens = capped
        batch_budget = getattr(self, 'prefill_batch', None)
        ordinary = (num_computed_tokens < request.num_prompt_tokens and not atomic_replay and
                    not (batch_budget is not None and self._restore_control_reservation(request, num_computed_tokens)))
        if batch_budget is not None and ordinary:
            available = batch_budget.available(request.request_id)
            if available is not None:
                if padding and available < native_tokens:
                    batch_budget.charge(request.request_id, 0)
                    return 0
                tokens = min(tokens, available)
        if (getattr(self, 'prefill_controller', None) is not None and
                num_computed_tokens < request.num_prompt_tokens and
                not atomic_replay):
            tokens = min(tokens, self.prefill_controller.cap(decode_ready=self._decode_ready()))
        if plan is not None and num_computed_tokens < request.num_prompt_tokens:
            chunk = int(os.getenv('KVMEM_PREFILL_CHUNK_TOKENS','0'))
            if chunk and not atomic_replay:
                # Ordinary ingestion follows the requested chunk size. The
                # complete Query remains atomic even when it exceeds this cap.
                if prepared is None and not padding:
                    chunk = self._coalescing_cap(request, num_computed_tokens, chunk)
                    logical_group = self._logical_prefill_group(
                        request, plan, num_computed_tokens, offset, tokens, chunk)
                    if logical_group is not None:
                        chunk = logical_group.count
                tokens = min(tokens,chunk)
            if (batch_budget is not None and ordinary and not padding and
                    getattr(self, 'num_prefill_lookahead', 1) > 1):
                # Reapply native MTP lookahead after introducing the new cap,
                # before the plan's mandatory checkpoint/Query boundaries.
                # Clipping a mandatory boundary afterwards can strand a row
                # one token before it and turn every future admission into 0.
                # The cap must not change a boundary already reached by the
                # original admission, including when cap == boundary distance.
                baseline = plan.admit(num_computed_tokens, native_tokens, request.num_prompt_tokens,
                                      offset, self.query_policy.capacity)
                if tokens < baseline:
                    tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, tokens)
            tokens = plan.admit(num_computed_tokens, tokens, request.num_prompt_tokens, offset, self.query_policy.capacity)
        else:
            if (plan is None and getattr(self, 'request_planning', False)
                    and num_computed_tokens < request.num_prompt_tokens):
                # Legacy chat/text entries share the larger atomic-Query batch
                # budget. Their chunk still needs room for native MTP lookahead.
                tokens = min(tokens, self.query_policy.gen_budget - self.query_policy.block_tokens,
                             8 * self.query_policy.block_tokens)
            if (batch_budget is not None and ordinary and not padding and
                    getattr(self, 'num_prefill_lookahead', 1) > 1):
                baseline = self.query_policy.admit(num_computed_tokens, native_tokens,
                                                   request.num_prompt_tokens, offset)
                if tokens < baseline:
                    tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, tokens)
            tokens = self.query_policy.admit(num_computed_tokens, tokens, request.num_prompt_tokens, offset)
        if plan is not None and getattr(request, '_kvmem_guided_epoch', None) is not None:
            from .guided_refresh import admit_epoch
            tokens = admit_epoch(request, plan, num_computed_tokens, tokens, offset,
                                 self.query_policy.capacity)
        if logical_group is not None and tokens != logical_group.count:
            # Native clipping must never create a new partial logical chunk
            # from a group of original 2K transactions. Fall back before allocation.
            tokens = plan.admit(num_computed_tokens, min(tokens, 2048),
                               request.num_prompt_tokens, offset, self.query_policy.capacity)
            logical_group = None
        if padding and tokens:
            # Native waiting-queue admission pads a one-token public tail to
            # the full verification shape when joining an existing decode batch.
            # These rows are placeholders, not additional historical input.
            if tokens != 1 or num_computed_tokens - offset + native_tokens > self.query_policy.capacity:
                raise RuntimeError('Padded public tail exceeds its physical reservation')
            tokens = native_tokens
        if (self.session_capture_decode and self.connector.sessions is not None and
                request.request_id in self.connector.sessions.requests and
                num_computed_tokens >= request.num_prompt_tokens):
            # Any speculative prefix can be accepted. Leave a guaranteed sealed
            # boundary for the next CPU snapshot rather than jumping over it.
            tokens = min(tokens, self.query_policy.block_tokens -
                         num_computed_tokens % self.query_policy.block_tokens)
        gate = getattr(self, 'archive_admission', None)
        if gate is not None and tokens and not gate.allows(request.request_id, num_computed_tokens, tokens):
            return 0  # Native running-queue iteration continues with neighbours.
        if batch_budget is not None and ordinary:
            # Count the final admitted shape, including complete MTP padding.
            # Failed native allocations conservatively keep their reservation
            # until the next step; exact restore controls never spend it.
            batch_budget.charge(request.request_id, tokens)
        if phase is not None:
            if phase_budget is not None and not padding and not atomic_replay:
                tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, tokens)
            if tokens:
                public = min(tokens, max(0, request.num_prompt_tokens - num_computed_tokens))
                self._phase_admitted[request.request_id] = (public, bool(atomic_replay), phase_maintenance)
                if phase_budget is not None:
                    self._phase_charges[request.request_id] = public
                    self._phase_atomic_overshoot |= bool(atomic_replay and public > phase_budget)
        if getattr(self, '_cost_admitted', None) is not None and tokens:
            self._cost_admitted[request.request_id] = dict(history=num_computed_tokens,
                prompt_tokens=request.num_prompt_tokens,
                atomic_range=(plan.replay_begin, plan.replay_end) if plan is not None and plan.query is not None else None,
                prepared_replay=prepared is not None,
                authorization_id=getattr(self, '_coalescing_authorizations', {}).get(request.request_id))
        if logical_group is not None and tokens:
            if tokens != logical_group.count:
                raise RuntimeError('Logical prefill changed shape after final admission')
            self._logical_prefill_groups[request.request_id] = logical_group
        return tokens

    def _prepare_logical_prefill(self):
        self._logical_prefill_groups = {}
        self._logical_prefill_owner = None
        if not getattr(self, 'prefill_physical_coalescing', False):
            return
        # The first qualification admits exactly one live request. Counting
        # blocked/waiting neighbours as well avoids guessing native composition.
        requests = list(self.running) + list(self._waiting_requests())
        if len(requests) == 1:
            self._logical_prefill_owner = requests[0]

    def _logical_prefill_group(self, request, plan, history, offset, budget, chunk):
        if (not getattr(self, 'prefill_physical_coalescing', False) or chunk != 2048 or history == 0 or
                getattr(self, '_logical_prefill_owner', None) is not request or budget < 4096 or
                request.request_id in self.transactions or request.request_id in self.connector.pending or
                request.request_id in self.prepared_refreshes or
                getattr(request, '_kvmem_guided_epoch', None) is not None or
                getattr(self, 'prefill_controller', None) is not None or
                getattr(self, 'prefill_batch', None) is not None or
                getattr(self, '_phase_decision', None) is not None or
                getattr(self, 'prefill_coalescer', None) is not None or
                getattr(request, 'spec_token_ids', ()) or
                any(request.request_id in getattr(manager, name, {})
                    for manager in self.window_managers.values()
                    for name in ('pending', 'imports', 'resident_imports'))):
            return None
        from .logical_prefill import LogicalPrefillEligibility, plan_logical_prefill
        gate = getattr(self, 'archive_admission', None)
        if gate is not None and callable(getattr(gate, 'available_tokens', None)):
            budget = min(budget, gate.available_tokens(request.request_id, history))
        maximum = min(getattr(self, 'prefill_physical_max_chunks', 2), budget//2048,
                      (request.num_prompt_tokens-history-1)//2048,
                      (self.query_policy.capacity-history+offset)//2048)
        if maximum < 2:
            return None
        limits = tuple(super(KVMemScheduler, self)._reserve_prefill_lookahead(
            request, history + i * 2048, 2048) for i in range(maximum))
        gates = LogicalPrefillEligibility((request.request_id,), limits, budget,
            native_prefill_lookahead=getattr(self, 'num_prefill_lookahead', 1))
        group, _ = plan_logical_prefill(plan, request_id=request.request_id,
            # An admission receipt, not a worker authority/window generation.
            # Action rebinds to its local generation after validating geometry.
            owner_epoch=('scheduler', self.current_step, request.request_id, offset),
            scheduler_step=self.current_step, history_begin=history, offset=offset,
            prompt_tokens=request.num_prompt_tokens, capacity=self.query_policy.capacity,
            eligibility=gates)
        return group

    def _preserve_deferred_padding(self, request, output, query_tokens):
        # Native waiting admission synthesizes these placeholders, whereas the
        # running queue uses request.spec_token_ids. A suspended first forward
        # must retain them or resuming silently changes the GDN/MTP kernel path.
        padding = output.scheduled_spec_decode_tokens.get(request.request_id)
        if not padding or any(token != -1 for token in padding):
            return
        if (query_tokens != 1 + len(padding) or
                request.num_prompt_tokens - request.num_computed_tokens != 1):
            raise RuntimeError('Native placeholder rows do not describe a single public tail')
        if not hasattr(self, '_deferred_native_padding'):
            self._deferred_native_padding = {}
        self._deferred_native_padding[request.request_id] = (request.num_computed_tokens, query_tokens)
        request.spec_token_ids = list(padding)

    def _build_kv_connector_meta(self, connector, output):
        if getattr(self, "_bounded_target", False):
            for request_id, query_tokens in list(output.num_scheduled_tokens.items()):
                request = self.requests[request_id]
                if (getattr(self, 'async_restore', False) and connector.sessions is not None
                        and connector.sessions.import_scheduled(request_id)):
                    binding = next(b for b in connector.sessions.commands.bindings if b.request_id == request_id)
                    if request.num_in_flight_tokens or request_id in self.transactions:
                        raise RuntimeError('Restore requires a quiescent fresh native allocation')
                    self._restore_sequence += 1
                    command = RestoreCommand(request_id, -self._restore_sequence, binding.generation,
                        binding.history, binding.window, tuple(tuple(b.block_id for b in m.req_to_blocks.get(request_id, ()))
                            for m in self.kv_cache_manager.coordinator.single_type_managers),
                        resident=request_id in getattr(self.kv_cache_manager, 'resident_restores', ()))
                    if getattr(connector, 'restore_admission', None) is not None:
                        connector.restore_admission.dispatch(command)
                    getattr(self.kv_cache_manager, 'resident_restores', set()).discard(request_id)
                    self.transactions[request_id] = (command, request.next_decode_eligible_step)
                    request.next_decode_eligible_step = 2**62
                    connector.queue(command)
                    self._preserve_deferred_padding(request, output, query_tokens)
                    output.num_scheduled_tokens.pop(request_id)
                    output.total_num_scheduled_tokens -= query_tokens
                    output.scheduled_spec_decode_tokens.pop(request_id, None)
                    continue
                if connector.sessions is not None:
                    connector.sessions.capture_before_final_prefill(request, query_tokens)
                    if self.session_capture_decode:
                        connector.sessions.capture_decode_boundary(request)
                if hasattr(self, '_scheduled_queries'):
                    stage = 'prefill' if request.num_computed_tokens < request.num_prompt_tokens else 'decode'
                    histogram = self._scheduled_queries[stage]
                    histogram[str(query_tokens)] = histogram.get(str(query_tokens), 0) + 1
                offset = next(iter(self.window_managers.values())).offsets.get(request_id, 0)
                imported_query = (connector.sessions is not None and connector.sessions.import_scheduled(request_id)
                                  and request.num_computed_tokens > self.query_policy.select_budget)
                from .request_plan import request_plan
                plan = request_plan(request, self.query_policy.block_tokens, self.query_policy.gen_budget)
                action = (plan.action(request.num_computed_tokens, request.num_prompt_tokens, offset,
                    self.query_policy.capacity) if plan is not None else None)
                guided = self._guided_refresh(request, request.num_computed_tokens, offset)
                if guided is not None:
                    action = 'guided'
                if request_id in getattr(self, 'prepared_refreshes', {}):
                    history, reserved = self.prepared_refreshes.pop(request_id)
                    if history != request.num_computed_tokens or query_tokens > reserved:
                        raise RuntimeError('Prepared refresh replay exceeds its reservation')
                    continue
                selected_budget = (plan.selection_budget(request.num_computed_tokens, request.num_prompt_tokens,
                    self.query_policy.capacity) if plan is not None else self.query_policy.select_budget)
                if action == 'exhausted':
                    raise RuntimeError('keep_selected append headroom was exhausted')
                if ((plan is None and (imported_query or self.query_policy.refresh(
                        request.num_computed_tokens, request.num_prompt_tokens, offset))) or
                        (action is not None and request.num_computed_tokens >= selected_budget)):
                    asynchronous = getattr(self, 'async_refresh', False)
                    tokens = ()
                    if guided is not None and request.num_computed_tokens < request.num_prompt_tokens:
                        tokens = tuple(request.prompt_token_ids[request.num_computed_tokens:
                                                                request.num_prompt_tokens])
                        if len(tokens) > query_tokens:
                            raise RuntimeError('Guided boundary split the final public input tail')
                    padding = 0
                    replay_tokens = query_tokens
                    rolling_pressure = (asynchronous and action == 'pressure' and
                        getattr(self, 'rolling_active_window', False) and
                        getattr(plan, 'rolling_active_window', False) and
                        request.num_computed_tokens >= request.num_prompt_tokens)
                    if asynchronous:
                        begin = request.num_computed_tokens
                        if rolling_pressure:
                            # Select from published, accepted history only. The
                            # sampled next token and all draft proposals stay
                            # outside the selection; no private model forward
                            # or public replay advances recurrent state.
                            if (request.num_in_flight_tokens or
                                    begin - offset != self.query_policy.capacity - self.query_policy.block_tokens):
                                raise RuntimeError('Rolling pressure requires a quiescent accepted-prefix boundary')
                            replay_tokens = 0
                        else:
                            tokens = tuple(request.all_token_ids[begin:begin+query_tokens])
                            spec = output.scheduled_spec_decode_tokens.get(request_id, [])
                            if (request.num_prompt_tokens - begin == len(tokens) == 1 and
                                    query_tokens == 1 + len(spec) and spec and all(t == -1 for t in spec)):
                                padding = len(spec)
                            if begin >= request.num_prompt_tokens or len(tokens) + padding != query_tokens:
                                raise ValueError('Async refresh requires a complete public prompt/Query prefix')
                    self.begin_refresh(request, selected_budget, replay_tokens=replay_tokens,
                        query_span=plan.query if action == 'query' else None,
                        selection='pressure' if action == 'pressure' else 'query' if action in ('query', 'guided') else 'chunk',
                        mandatory_spans=plan.mandatory if plan is not None else (),
                        message_spans=plan.message_spans if plan is not None else (),
                        locality_spans=plan.locality_spans if plan is not None else (),
                        locality_provenance=plan.locality_provenance if plan is not None else (),
                        input_tokens=tokens, asynchronous=asynchronous, speculative_padding=padding,
                        guided_reason=guided[1] if guided else '',
                        guided_private=guided[2] if guided else False,
                        guided_boundary=(getattr(request, '_kvmem_guided_epoch').raw_private
                                         if getattr(request, '_kvmem_guided_epoch', None) is not None else None))
                    if asynchronous:
                        # Allocation/new-request/table updates remain delivered
                        # to the worker. Native counters and sampling advance
                        # ONLY the other requests; this Query is replayed later.
                        if not rolling_pressure:
                            self._preserve_deferred_padding(request, output, query_tokens)
                        output.num_scheduled_tokens.pop(request_id)
                        output.total_num_scheduled_tokens -= query_tokens
                        output.scheduled_spec_decode_tokens.pop(request_id, None)
                        continue
                    # Native V2 otherwise defers start_load_kv until after model
                    # forward. The query command must exist before this forward.
                    output.has_sync_kv_loads = True
        for request_id, query_tokens in output.num_scheduled_tokens.items():
            deferred = getattr(self, '_deferred_native_padding', {}).pop(request_id, None)
            if deferred is not None and (query_tokens != deferred[1] or
                    output.scheduled_spec_decode_tokens.get(request_id) != [-1] * (deferred[1] - 1)):
                raise RuntimeError('Deferred native tail resumed without its complete placeholder shape')
        if getattr(self.kv_cache_manager, 'refresh_control_allocations', None):
            controls = [(rid, history, count) for rid, (_, history, count) in
                        self.kv_cache_manager.refresh_control_allocations.items()]
            self.kv_cache_manager.validate_refresh_controls(output.num_scheduled_tokens, self.transactions)
            with (Path(os.environ['KVMEM_RUNNER_PROBE_DIR'])/'refresh_control_allocations.jsonl').open('a') as stream:
                for rid, history, count in controls:
                    stream.write(json.dumps(dict(at=time.monotonic(),request_id=rid,
                        history=history,replay_tokens=count,public_peer_count=len(output.num_scheduled_tokens),
                        private_pressure_verified=True))+'\n')
        metadata = super()._build_kv_connector_meta(connector, output)
        if getattr(self, 'prefill_physical_coalescing', False):
            groups = {rid: group for rid, group in self._logical_prefill_groups.items()
                      if rid in output.num_scheduled_tokens}
            if groups and (len(groups) != 1 or len(output.num_scheduled_tokens) != 1 or
                    any(output.num_scheduled_tokens[rid] != group.count or
                        output.scheduled_spec_decode_tokens.get(rid)
                        for rid, group in groups.items())):
                raise RuntimeError('Physical prefill no longer matches its single-owner admission')
            metadata.logical_prefill_groups = groups
            metadata.logical_prefill_step = self.current_step
        if getattr(self, 'offline_policy', None) is not None:
            metadata.maintenance_costs = self.offline_policy.maintenance_costs()
            metadata.cost_classification = {
                rid: {**self._cost_admitted[rid], 'scheduled_tokens': tokens}
                for rid, tokens in output.num_scheduled_tokens.items()
                if rid in (getattr(self, '_cost_admitted', None) or {})}
        return metadata

    def _observe_offline_steps(self, rows):
        # Both offline arms exclude Query/unknown public work from ordinary
        # ingestion cost fits. Pure decode keeps its existing model-cost fit.
        self.offline_policy.costs.observe_steps([row for row in rows
            if not row.get('prefill_tokens') or row.get('ordinary_prefill_only') is True])
        if getattr(self, 'prefill_coalescer', None) is not None:
            self.prefill_coalescer.observe(rows)

    def _prepare_coalescing(self):
        # Both offline comparison arms publish the same cost labels. This is
        # metadata only; disabled coalescing changes no native token limit.
        self._cost_admitted = {} if getattr(self, 'offline_policy', None) is not None else None
        self._coalescing_candidate = None
        self._coalescing_authorizations = {}
        self._coalescing_caps = {}
        self._coalescing_reason = 'baseline: no single known running prefill'
        self._coalescing_eligible_decode = []
        if getattr(self, 'prefill_coalescer', None) is None:
            return
        self._coalescing_eligible_decode = [r.request_id for r in self.running
            if self._phase_eligible(r) and not r.is_prefill_chunk and
            r.num_computed_tokens >= r.num_prompt_tokens and
            r.num_tokens_with_spec + getattr(r, 'num_output_placeholders', 0) > r.num_computed_tokens]
        # Only a fully known existing running composition is calibrated.
        # Fresh waiting admissions and maintenance fences retain baseline caps.
        if any(self._waiting_queues()):
            self._coalescing_reason = 'baseline: waiting composition unknown'
            return
        decode, prefills = [], []
        for request in self.running:
            if not self._phase_eligible(request) or request.request_id in self.prepared_refreshes:
                return
            if request.num_computed_tokens < request.num_prompt_tokens:
                if self._phase_refresh_only(request):
                    return
                prefills.append(request)
            elif (not request.is_prefill_chunk and
                  request.num_tokens_with_spec + getattr(request, 'num_output_placeholders', 0) >
                  request.num_computed_tokens):
                decode.append(request)
            else:
                return
        if len(prefills) != 1:
            return
        from .offline_policy import context_bucket
        offsets = next(iter(self.window_managers.values())).offsets
        extents = [r.num_tokens_with_spec + getattr(r, 'num_output_placeholders', 0) -
                   offsets.get(r.request_id, 0) for r in decode]
        self._coalescing_candidate = (prefills[0].request_id, len(decode),
                                     context_bucket(max(extents)) if extents else None)

    def _coalescing_cap(self, request, history, baseline):
        candidate = getattr(self, '_coalescing_candidate', None)
        if candidate is None or candidate[0] != request.request_id:
            return baseline
        if request.request_id in self._coalescing_caps:
            return self._coalescing_caps[request.request_id]
        from .offline_policy import context_bucket
        coalescer = self.prefill_coalescer
        remaining = request.num_prompt_tokens - history
        offset = next(iter(self.window_managers.values())).offsets.get(request.request_id, 0)
        base_bucket = context_bucket(history - offset + min(baseline, remaining))
        max_bucket = context_bucket(history - offset + min(self._coalescing_max_tokens, remaining))
        if base_bucket != max_bucket:
            self._coalescing_reason = 'baseline: context bucket edge'
            return baseline  # Never reuse a fit across a context bucket edge.
        cap = coalescer.cap(decode_requests=candidate[1], prefill_requests=1,
            decode_context_bucket=candidate[2], prefill_context_bucket=base_bucket,
            remaining_tokens=remaining)
        if type(cap) is not int or not baseline <= cap <= self._coalescing_max_tokens:
            raise RuntimeError('Coalescing must preserve the baseline chunk and maximum bound')
        self._coalescing_caps[request.request_id] = cap
        self._coalescing_reason = coalescer.last_decision
        authorization = coalescer.last_authorization_id
        if authorization is not None:
            self._coalescing_authorizations[request.request_id] = authorization
        return cap

    def _reconcile_coalescing(self, output=None):
        if getattr(self, 'prefill_coalescer', None) is None:
            return
        for rid, authorization in self._coalescing_authorizations.items():
            entry = (self._cost_admitted or {}).get(rid)
            scheduled = output.num_scheduled_tokens.get(rid, 0) if output is not None else 0
            public = min(scheduled, max(0, entry['prompt_tokens'] - entry['history'])) if entry else 0
            self.prefill_coalescer.reconcile(authorization, actual_tokens=public)

    def _record_coalescing(self, output):
        coalescer = getattr(self, 'prefill_coalescer', None)
        if coalescer is None:
            return
        prefills, decodes, atomics = {}, {}, {}
        for rid, n in output.num_scheduled_tokens.items():
            entry = (self._cost_admitted or {}).get(rid)
            if entry is None:
                continue
            public = min(n, max(0, entry['prompt_tokens'] - entry['history']))
            if not public:
                decodes[rid] = n
                continue
            prefills[rid] = public
            span = entry['atomic_range']
            if entry['prepared_replay'] or (span is not None and
                    entry['history'] < span[1] and entry['history'] + public > span[0]):
                atomics[rid] = public
        candidate = self._coalescing_candidate
        if candidate is None and not self._coalescing_authorizations and not any(
                n > coalescer.baseline for n in prefills.values()):
            return  # No per-step file open on steady decode.
        row = dict(at=time.monotonic(), step=self.current_step, candidate=candidate,
            reason=self._coalescing_reason, caps=self._coalescing_caps,
            authorizations=self._coalescing_authorizations,
            eligible_decode_ids=self._coalescing_eligible_decode,
            scheduled_decode_tokens=decodes, public_prefill_tokens=prefills,
            atomic_prefill_tokens=atomics, baseline_tokens=coalescer.baseline)
        with (Path(os.environ['KVMEM_RUNNER_PROBE_DIR']) / 'coalescing_decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    def _phase_eligible(self, request):
        """Ready at the next native step; no fence is repurposed for phases."""
        return (not request.is_finished() and not request.num_in_flight_tokens and
                request.request_id not in self.transactions and
                request.request_id not in self.connector.pending and
                not (request.request_id in getattr(self.connector, 'publication_blocked', ()) and
                     request.status is RequestStatus.RUNNING) and
                request.next_decode_eligible_step <= self.current_step + 1 and
                not getattr(request, 'num_stale_output_tokens', 0))

    def _phase_refresh_only(self, request):
        # A ready public replay must execute in PREFILL. Before that point the
        # exact same admission merely enqueues private refresh work and is
        # removed from the public batch by _build_kv_connector_meta.
        if not self.async_refresh or request.request_id in self.prepared_refreshes:
            return False
        history = request.num_computed_tokens
        if history >= request.num_prompt_tokens:
            return False
        from .request_plan import request_plan
        plan = request_plan(request, self.query_policy.block_tokens, self.query_policy.gen_budget)
        offset = next(iter(self.window_managers.values())).offsets.get(request.request_id, 0)
        if plan is None:
            return self.query_policy.refresh(history, request.num_prompt_tokens, offset)
        action = plan.action(history, request.num_prompt_tokens, offset, self.query_policy.capacity)
        return (action not in (None, 'exhausted') and
                history >= plan.selection_budget(history, request.num_prompt_tokens, self.query_policy.capacity))

    def _prepare_phase(self):
        policy = getattr(self, 'phase_policy', None)
        if policy is None:
            return
        prefills, decodes, maintenance = {}, set(), set()
        for request in self.running:
            if not self._phase_eligible(request):
                continue
            rid = request.request_id
            if request.num_computed_tokens < request.num_prompt_tokens:
                if self._phase_refresh_only(request):
                    maintenance.add(rid)
                else:
                    prefills[rid] = request.num_prompt_tokens - request.num_computed_tokens
            elif (not request.is_prefill_chunk and rid not in self.prepared_refreshes and
                  request.num_tokens_with_spec + getattr(request, 'num_output_placeholders', 0) >
                  request.num_computed_tokens):
                decodes.add(rid)
        # Pausing a running request retains its native GDN/MTP slot and its
        # full lifetime reservation. A full batch cannot admit another wave.
        slots = max(0, self.scheduler_config.max_num_seqs - len(self.running) -
                    getattr(self, 'num_waiting_for_streaming_input', 0))
        waiting = []
        sessions = self.connector.sessions
        for request in sorted(self._waiting_requests(),
                              key=lambda r: self.offline_policy.ranks.get(r.request_id, 0)):
            if not slots:
                break
            if (not self._phase_eligible(request) or self._is_blocked_waiting_status(request.status) or
                    not self.connector.history_admission.can_admit(request) or
                    self.kv_cache_manager.admission_preview(request) is not None):
                continue
            namespace = sessions.namespace(request)
            entry = sessions.entries.get(namespace)
            if namespace is not None and (
                    namespace in sessions.uncached_sessions and
                    sessions.uncached_sessions[namespace] != request.request_id or
                    entry is not None and entry.phase != 'idle' and
                    not (entry.phase == 'lookup' and entry.request_id == request.request_id) or
                    entry is None and sessions.preserve_history and len(sessions.entries) >= sessions.capacity):
                continue
            history, _, _, reason = self.offline_policy._hint(request)
            if reason is not None:
                continue
            if self.async_restore and history:
                maintenance.add(request.request_id)
            elif len(waiting) < slots:
                waiting.append(request.request_id)
        # Allocation/archive state can change after a read-only preview. An
        # empty proposed wave must yield a real decode opportunity, including
        # when no prefill batch has yet run (the wave timer has no progress).
        if getattr(self, '_phase_yield_after_empty', False) and decodes:
            prefills, waiting = {}, []
        self._phase_yield_after_empty = False
        self._phase_decision = policy.decide(decodes, prefills, waiting)
        self._phase_prefills = set(prefills) | set(waiting)
        self._phase_decodes, self._phase_maintenance = decodes, maintenance
        self._phase_charges, self._phase_admitted = {}, {}
        self._phase_atomic_overshoot = self._phase_budget_blocked = False

    def _record_phase(self, output):
        if getattr(self, 'phase_policy', None) is None:
            return
        prefills, decodes, atomic = {}, {}, 0
        for rid, tokens in output.num_scheduled_tokens.items():
            if rid not in self._phase_admitted:
                raise RuntimeError('Phase isolation violated: missing token admission identity')
            public, replay, maintenance = self._phase_admitted.get(rid, (0, False, False))
            if maintenance or public and self._phase_decision.phase != 'prefill':
                raise RuntimeError('Phase isolation violated: maintenance admitted public prefill')
            if public:
                prefills[rid] = min(public, tokens)
                atomic += prefills[rid] if replay else 0
            else:
                if self._phase_decision.phase != 'decode':
                    raise RuntimeError('Phase isolation violated: prefill wave admitted decode')
                decodes[rid] = tokens
        self._phase_yield_after_empty = (self._phase_decision.phase == 'prefill' and
                                         not prefills and bool(self._phase_decodes))
        observation = self.phase_policy.record(prefills, atomic_query_tokens=atomic,
            decode_tokens_by_request=decodes, blocked_by_budget=self._phase_budget_blocked)
        # Native counters have advanced here, but GPU execution has not. These
        # are admitted public tokens, not completed CUDA work or output tokens.
        now = time.monotonic()
        if (self._phase_decision.changed or prefills or self._phase_maintenance or
                now - getattr(self, '_phase_last_log', float('-inf')) >= 1.):
            self._phase_last_log = now
            path = Path(os.environ['KVMEM_RUNNER_PROBE_DIR']) / 'phase_decisions.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(dict(at=now, step=self.current_step,
                    decision=vars(self._phase_decision), public_prefill_tokens=prefills,
                    decode_query_tokens=decodes, maintenance_candidates=sorted(self._phase_maintenance),
                    observation=observation, cumulative=self.phase_policy.snapshot())) + '\n')

    def _decode_ready(self, *, step=None):
        step = self.current_step if step is None else step
        return any(not r.is_prefill_chunk and r.num_computed_tokens >= r.num_prompt_tokens
            and not r.is_finished() and not r.num_in_flight_tokens
            and r.request_id not in self.transactions
            and r.request_id not in getattr(self.connector, 'publication_blocked', ())
            and r.next_decode_eligible_step <= step
            for r in self.running)

    def _prepare_prefill_budget(self):
        budget = getattr(self, 'prefill_batch', None)
        if budget is None:
            return
        # Native schedule increments current_step before checking readiness.
        # Use that same step here; cadence/adaptive callers retain their own
        # existing point-in-time interpretation of _decode_ready().
        step = self.current_step + 1
        blocked = getattr(self.connector, 'publication_blocked', ())
        def ready(request):
            # Match native admission: stale publication feedback cannot block
            # WAITING rows or remove their share ahead of a ready restore.
            return (request.request_id not in self.transactions and
                    not (getattr(request, 'status', None) is RequestStatus.RUNNING and
                         request.request_id in blocked) and
                    not request.is_finished() and
                    not request.num_in_flight_tokens and request.next_decode_eligible_step <= step)

        policy = getattr(self, 'offline_policy', None)
        if policy is None:
            prefills = sum(ready(r) and r.num_computed_tokens < r.num_prompt_tokens for r in self.running)
            waiting = sum(len(q) for q in self._waiting_queues())
        else:
            from .request_plan import request_plan
            def ordinary(request, *, running):
                if not ready(request):
                    return False
                row = policy.rows.get(request.request_id)
                if row is None or row.reason is not None or row.remaining_tokens <= 0:
                    return False
                if not running and self.async_restore and row.restore_tokens:
                    # Its first native allocation produces a restore command,
                    # not a public forward. Do not dilute ready prefill shares
                    # with work still requiring a worker completion fence.
                    return False
                history = request.num_prompt_tokens - row.remaining_tokens
                plan = request_plan(request, self.query_policy.block_tokens, self.query_policy.gen_budget)
                return not (plan is not None and plan.query is not None and
                            plan.replay_begin <= history < plan.replay_end)

            # Reuse read-only readiness hints prepared by the ordering policy.
            # Never acquire history/GPU/restore credits or reorder either queue
            # to compute a denominator. Real admission can still reject work;
            # those token reservations conservatively expire at the next step.
            prefills = sum(ordinary(r, running=True) for r in self.running)
            waiting = sum(ordinary(r, running=False) for r in self._waiting_requests())
        slots = max(0, self.scheduler_config.max_num_seqs - len(self.running))
        budget.begin(decode_ready=self._decode_ready(step=step),
                     contenders=prefills + min(waiting, slots))

    def _throttle_mixed_prefill(self):
        """Ask the native scheduler for decode-only steps, preserving its gates.

        Every cadence-th step releases prefills. Capacity-bound native scheduling
        can override the request; transactions and unavailable decode work never
        justify delaying prefill. No query sizes or request ordering are changed.
        """
        if not getattr(self, '_bounded_target', False) or getattr(self, 'prefill_cadence', 1) == 1:
            return False
        mixed = (any(r.is_prefill_chunk for r in self.running) or any(self._waiting_queues())) and self._decode_ready()
        if not mixed or self.prefill_capacity_bound:
            self._mixed_steps = 0
            return False
        self._mixed_steps += 1
        throttle = self._mixed_steps % self.prefill_cadence != 0
        self._cadence_requested += int(throttle)
        return throttle

    def schedule(self, throttle_prefills=False):
        if getattr(self.connector, 'active_residency', None) is not None:
            self.connector.active_residency.expire()
        if getattr(self, 'offline_policy', None) is not None:
            self.offline_policy.prepare()
        self._prepare_coalescing()
        self._prepare_logical_prefill()
        self._prepare_phase()
        self._prepare_prefill_budget()
        armed = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "armed"
        if not getattr(self, "_bounded_target", False) and not self._probe_submitted and armed.is_file():
            for request in self.running:
                if request.num_computed_tokens == request.num_prompt_tokens and request.num_output_tokens:
                    self.begin_refresh(request, request.num_computed_tokens)
                    self._probe_submitted = True
                    break
        if getattr(self, '_bounded_target', False):
            self.kv_cache_manager.begin_step()
        try:
            output = super().schedule(throttle_prefills or self._throttle_mixed_prefill())
        except BaseException:
            self._reconcile_coalescing()
            raise
        finally:
            if getattr(self.connector, 'restore_admission', None) is not None:
                self.connector.restore_admission.end_step()
            if getattr(self, '_bounded_target', False):
                self.kv_cache_manager.end_step()
        self._reconcile_coalescing(output)
        self._record_coalescing(output)
        self._record_phase(output)
        if hasattr(self, '_scheduled_queries') and output.total_num_scheduled_tokens:
            # Actual admitted sizes, including native draft reservations and
            # alignment, are the evidence for a matched query-policy comparison.
            self._write_scheduled_queries()
        if self.table_updates:
            # Scheduling can allocate another block after the refresh commit.
            # Publish the complete current row, including those new allocations.
            updates = {
                key: tuple([b.block_id for b in manager.req_to_blocks.get(key, [])]
                           for manager in self.kv_cache_manager.coordinator.single_type_managers)
                for key in self.table_updates
                if key in self.requests and not self.requests[key].is_finished()}
            if hasattr(output, 'block_table_updates'):
                output.block_table_updates = {**(output.block_table_updates or {}), **updates}
            else:
                control = output.kv_connector_metadata
                if not isinstance(control, ControlMetadata):
                    raise RuntimeError('Page publication requires KVMem connector metadata')
                control.block_table_updates.update(updates)
            self.table_updates = {}
        if getattr(self, 'offline_policy', None) is not None:
            self.offline_policy.record(output)
        return output

    def _select_waiting_queue_for_scheduling(self):
        if getattr(self, 'offline_policy', None) is not None:
            queue = self.offline_policy.waiting_queue()
            if getattr(self, '_phase_decision', None) is None:
                return queue
            # Native breaks when a waiting head receives zero tokens. Let
            # maintenance commands pass a phase-deferred public prefill; keep
            # offline ordering within each class and retain every queued ID.
            def rank(request):
                rid = request.request_id
                allowed = (self._phase_decision.phase == 'prefill' and rid in self._phase_prefills)
                return (0 if rid in self._phase_maintenance else 1 if allowed else 2,
                        self.offline_policy.ranks.get(rid, 0))
            queues = [q for q in self._waiting_queues() if q]
            for pending in queues:
                ordered = sorted(pending, key=rank)
                pending.clear()
                pending.extend(ordered)
            return min(queues, key=lambda q: rank(q.peek_request())) if queues else None
        return super()._select_waiting_queue_for_scheduling()
