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
from ..core.options import RuntimeOptions
from .window_manager import WorkingSetManager
from .admission import QueryPolicy
from ..core import settings as _settings

# Placeholder for KVMEM_LIFECYCLE_SNAPSHOT_CACHE (spliced after json.dumps).
_SESSION_VIEWS_SLOT = '\x00kvmem-session-views\x00'


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
            if (_settings.get("KVMEM_BOUNDED_DFLASH") != "1" and _settings.get("KVMEM_BOUNDED_MTP") != "1") or _settings.get("KVMEM_BOUNDED_TARGET") != "1":
                raise ValueError("Bounded speculation requires its explicit experimental opt-in")
            from ..adapters.draft_contract import DraftContract
            self.draft_contract = DraftContract.from_vllm(
                self.vllm_config, RuntimeOptions.from_vllm(self.vllm_config).memory)
        self._bounded_target = _settings.get("KVMEM_BOUNDED_TARGET") == "1"
        if not self._bounded_target and _settings.get("KVMEM_SCHEDULED_REBUILD_PROBE") != "1":
            raise ValueError("Compressed runtime not yet enabled; scheduler currently exposes the explicit identity probe")
        if self._bounded_target:
            from ..adapters.engine_api import require_contract
            require_contract('scheduler')
        options = RuntimeOptions.from_vllm(self.vllm_config)
        self._deferred_prefill_validation = options.prefill_deferred_validation
        self.connector.request_plan_memory = options.memory if options.request_planning else None
        self.connector.request_plan_pressure_policy = options.pressure_policy
        from ..core import settings as _dflash_settings
        dflash_async = _dflash_settings.get('KVMEM_DFLASH_ASYNC_REFRESH') == '1'
        if options.rolling_active_window and not (self.draft_contract is not None and (
                (self.draft_contract.method == 'mtp' or self.draft_contract.method == 'dflash' and dflash_async)
                and options.async_refresh and options.async_restore or
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
            from ..core.repository_gc import serving_gc_config
            serving_gc_config(config, self.scheduler_config.max_num_batched_tokens,
                              checkpoint_frames=int(bool(options.session_capacity)))
            from ..core.archive_admission import ArchiveAdmission
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
        self.prefill_physical_multi_owner = getattr(options, 'prefill_physical_multi_owner', False)
        self.prefill_physical_fill_rows = getattr(options, 'prefill_physical_fill_rows', False)
        self._physical_native_rows = options.action_native_max_rows
        from ..core.defaults import legacy
        from .multiowner_prefill import coalesce_with_decode
        self.prefill_coalesce_with_decode = coalesce_with_decode(
            self.prefill_physical_coalescing and self.prefill_physical_multi_owner)
        self.prefill_physical_max_chunks = options.prefill_physical_max_chunks
        partial_tail = _settings.get('KVMEM_PREFILL_PHYSICAL_PARTIAL_TAIL',
                                 '1' if self.prefill_physical_coalescing and not legacy() else '0')
        if partial_tail not in ('0', '1'):
            raise ValueError('KVMEM_PREFILL_PHYSICAL_PARTIAL_TAIL must be 0 or 1')
        if partial_tail == '1' and not self.prefill_physical_coalescing:
            raise ValueError('Partial physical tails require physical prefill coalescing')
        self.prefill_physical_partial_tail = partial_tail == '1'
        no_replay = _settings.get('KVMEM_PREFILL_PRESSURE_NO_REPLAY', '0')
        if no_replay not in ('0', '1'):
            raise ValueError('KVMEM_PREFILL_PRESSURE_NO_REPLAY must be 0 or 1')
        self.prefill_pressure_no_replay = no_replay == '1'
        min_decoders = _settings.get('KVMEM_PREFILL_CADENCE_MIN_DECODERS', '1')
        if not min_decoders.isdigit() or not 1 <= int(min_decoders) <= 64:
            raise ValueError('KVMEM_PREFILL_CADENCE_MIN_DECODERS must be an integer in [1, 64]')
        self.prefill_cadence_min_decoders = int(min_decoders)
        # Opt-in KVMEM_PREFILL_DECODE_SHARE: decode-only turns sized by time, not
        # by a step count, so one setting fits short (G4K) and long (G16K) prefill
        # steps. Replaces prefill_cadence; the minimum-decoders gate still applies.
        share = _settings.get('KVMEM_PREFILL_DECODE_SHARE', '')
        self.prefill_decode_share = None
        if share:
            try:
                value = float(share)
            except ValueError:
                value = float('nan')
            if not 0 < value <= .5:
                raise ValueError('KVMEM_PREFILL_DECODE_SHARE must be a number in (0, 0.5]')
            if options.prefill_cadence != 1:
                raise ValueError('KVMEM_PREFILL_DECODE_SHARE replaces prefill_cadence; leave it at 1')
            self.prefill_decode_share = value
        self._share_credit = 0.
        self._share_previous = None
        # Opt-in KVMEM_PREFILL_DECODE_BURST: decoders never ride prefill steps;
        # their time share is spent in consecutive decode-only steps instead.
        burst = _settings.get('KVMEM_PREFILL_DECODE_BURST', '0')
        if burst not in ('0', '1'):
            raise ValueError('KVMEM_PREFILL_DECODE_BURST must be 0 or 1')
        if burst == '1' and self.prefill_decode_share is None:
            raise ValueError('KVMEM_PREFILL_DECODE_BURST needs KVMEM_PREFILL_DECODE_SHARE')
        self.prefill_decode_burst = burst == '1'
        # Opt-in KVMEM_PREFILL_DECODE_BURST_MIN_TOKENS=N (size-aware burst): a
        # hold is only worth it for large prefill work. When every schedulable
        # prefill is an admitted request with fewer than N prompt tokens left in
        # total (agent tool results, short user turns) and no arrival waits for
        # admission, decoders ride the prefill step instead of being held out.
        minimum = _settings.get('KVMEM_PREFILL_DECODE_BURST_MIN_TOKENS', '0')
        if not minimum.isdigit() or int(minimum) > 1 << 20:
            raise ValueError('KVMEM_PREFILL_DECODE_BURST_MIN_TOKENS must be an integer in [0, 1048576]')
        self.prefill_burst_min_tokens = int(minimum)
        if self.prefill_burst_min_tokens and not self.prefill_decode_burst:
            raise ValueError('KVMEM_PREFILL_DECODE_BURST_MIN_TOKENS needs KVMEM_PREFILL_DECODE_BURST')
        self._burst_active = False
        self._burst_missed = None
        self._burst_stats = dict(bursts=0, decode_steps=0, held_steps=0, held_decoders=0, misses=0,
                                 ride_along=0)
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
        if self._bounded_target:
            from .admission import validate_target_batch_budget
            validate_target_batch_budget(config.block_tokens, config.gen_budget,
                self.scheduler_config.max_num_batched_tokens, self.scheduler_config.max_num_seqs,
                self.draft_contract.query_tokens if self.draft_contract else 1,
                request_planning=self.request_planning)
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
        self.swap_controller = None
        if self._bounded_target:
            from .capacity import ReservedKVCacheManager
            manager_type, extra = ReservedKVCacheManager, {}
            if options.qsa_split_selector:
                from ..adapters.qsa_streaming import plan_qsa_carves
                if options.gpu_memory_admission != 'static':
                    raise ValueError('qsa_split_selector requires static GPU admission')
                draft = self.draft_contract.query_tokens - 1 if self.draft_contract is not None else 0
                extra['selector_carve'], extra['streaming_carve'], _ = plan_qsa_carves(
                    self.vllm_config, self.kv_cache_config, options, draft)
                if options.qsa_streaming:
                    carve = extra['streaming_carve']
                    for manager in self.window_managers.values():
                        if manager.block_size != carve.page_tokens:
                            raise ValueError('QSA streaming requires one attention page size')
                        manager.set_streaming(carve.pages_per_request * carve.page_tokens)
                    # Prefill attends from a shared frame-sized staging run, so
                    # at most staging_requests streaming requests prefill per step.
                    self._streaming_prefill_limit = 1
            if options.gpu_memory_admission == 'incremental':
                from .incremental_capacity import IncrementalKVCacheManager
                from .request_swap_control import RequestSwapController
                if (_settings.get('KVMEM_SYSTEMONE_FORK') == '1' or
                        (not self.vllm_config.model_config.enforce_eager and not options.request_swap_layerwise) or
                        any(getattr(self.vllm_config.parallel_config, field, 1) != 1 for field in
                            ('tensor_parallel_size', 'pipeline_parallel_size',
                             'decode_context_parallel_size', 'prefill_context_parallel_size'))):
                    raise ValueError('Incremental admission requires eager single-rank execution without forks')
                sizes = {tensor.size for tensor in self.kv_cache_config.kv_cache_tensors}
                if len(sizes) != 1:
                    raise ValueError('Exact request swap requires one native V2 backing allocation')
                page_bytes, remainder = divmod(sizes.pop(), self.kv_cache_config.num_blocks)
                if remainder or not page_bytes:
                    raise ValueError('Exact request swap requires integral native backing pages')
                self.swap_controller = RequestSwapController(self,
                    checkpoint_bytes=options.request_checkpoint_ssd_bytes or options.request_checkpoint_bytes,
                    checkpoint_cpu_bytes=options.request_checkpoint_bytes, page_bytes=page_bytes,
                    finish_callback=self._finish_swapped_request,
                    min_resident_steps=options.request_swap_min_resident_steps)
                manager_type = IncrementalKVCacheManager
                extra['pause_controller'] = self.swap_controller
            if (getattr(self, '_streaming_prefill_limit', None) is None and
                    _settings.get('KVMEM_QSA_SINGLE_PREFILL') == '1'):
                # Exactness / throughput reference: the resident path with streaming's step shapes.
                self._streaming_prefill_limit = 1
            self.kv_cache_manager = manager_type(self.kv_cache_manager, self.vllm_config,
                draft_contract=self.draft_contract,
                max_authorities=(self.scheduler_config.max_num_seqs
                    if options.async_archive_retirement or self.swap_controller is not None else None),
                **extra)
            if self.swap_controller is not None:
                self.swap_controller.bind(self.kv_cache_manager)
                self.connector.request_swap_pending = self.swap_controller.has_pending_work
        if _settings.get('KVMEM_SYSTEMONE_FORK') == '1':
            from .systemone_fork import ForkPagePool
            self.connector.systemone_pool = ForkPagePool(self, config)
        self.connector.scheduler_completion = self._complete_refresh
        schedule_ack = _settings.get('KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK', '0')
        if schedule_ack not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK must be 0 or 1')
        self._schedule_ack = schedule_ack == '1' and self.vllm_config.parallel_config.tensor_parallel_size == 1
        if self._schedule_ack and (self.scheduler_config.async_scheduling or
                                   self.vllm_config.parallel_config.tensor_parallel_size != 1 or
                                   self.vllm_config.parallel_config.distributed_executor_backend
                                   not in (None, 'uni')):
            raise ValueError('Schedule-time pressure ACK requires the synchronous single-process engine')
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
            from .prefix_images import scope_error as prefix_image_scope
            def prepare_import(request_id, history, window):
                prepared = []
                try:
                    for manager in self.window_managers.values():
                        manager.prepare_import(request_id, history, window)
                        prepared.append(manager)
                    if self.archive_admission is not None:
                        from ..core.archive_admission import ArchiveCapacity
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
                                                    guided_options=options,
                                                    prefix_image_scope=prefix_image_scope(
                                                        options, tensor_parallel_size=self.vllm_config
                                                        .parallel_config.tensor_parallel_size),
                                                    prefix_image_model=self.vllm_config.model_config.model)
        if options.cpu_history_coordination:
            from .coordination import HistoryAdmission
            from ..adapters.cache_manifest import archive_bytes_per_token
            from ..core.query_window_budget import split_cpu_history_budget
            from ..core.historical_feature_budget import split_historical_feature_budget
            history_bytes, _ = split_cpu_history_budget(options)
            history_bytes, _ = split_historical_feature_budget(options, history_bytes,
                self.scheduler_config.max_num_seqs + options.session_capacity)
            row_bytes = archive_bytes_per_token(self.kv_cache_config, self.vllm_config.model_config)
            record_bytes = row_bytes * config.block_tokens
            capacity = min(history_bytes,
                record_bytes * config.repository_blocks * (self.scheduler_config.max_num_seqs + options.session_capacity))
            if options.archive_backend == 'tiered':
                from ..core.archive_limits import tiered_capacity_records
                capacity = record_bytes * tiered_capacity_records(options, record_bytes,
                    self.scheduler_config.max_num_seqs)
            if options.historical_gc is not None:
                from .historical_admission import HistoricalHistoryAdmission
                from ..core.historical_gc import historical_record_charge
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
                self.connector.history_admission.storage_tier = (
                    'ssd_logical' if options.archive_backend == 'tiered' else 'cpu')
        if options.active_residency_enabled:
            if self.draft_contract is not None and self.draft_contract.method != 'mtp':
                raise ValueError('Idle active residency currently qualifies target-only or native MTP')
            from .active_residency import ActiveResidency
            tool_grace = None
            if options.active_tool_grace_ms:
                from .tool_grace import ToolGrace
                tool_grace = ToolGrace(options.active_tool_grace_ms, options.active_tool_max_wait_ms,
                    output=Path(_settings.require('KVMEM_RUNNER_PROBE_DIR')) / 'tool_grace.jsonl')
            wait_model = None
            if options.active_resident_victim == 'expected_wait':
                from .residency_policy import ReturnGapModel
                wait_model = ReturnGapModel()
            residency = ActiveResidency(self.window_managers, self.kv_cache_manager.block_pool,
                options.active_resident_ttl_ms, options.active_resident_max_fraction,
                pressure_only=options.active_resident_pressure_only, tool_grace=tool_grace,
                victim=options.active_resident_victim, wait_model=wait_model)
            self.connector.active_residency = self.kv_cache_manager.active_residency = residency
            self.kv_cache_manager.resident_entry = self.connector.sessions.requests.get
            if tool_grace is not None:
                self.connector.tool_grace_allows = self.kv_cache_manager.tool_grace_allows
        if options.async_restore:
            from .restore_admission import RestoreAdmission
            self.connector.restore_admission = self.kv_cache_manager.restore_admission = RestoreAdmission(
                residency=self.connector.active_residency,
                resident_slots=getattr(options, 'resident_restore_slots', 1),
                cold_busy=lambda: any(isinstance(command, RefreshCommand)
                                      for command, _ in self.transactions.values()),
                output=Path(_settings.require('KVMEM_RUNNER_PROBE_DIR')) / 'restore_admission.jsonl')
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
                Path(_settings.require('KVMEM_RUNNER_PROBE_DIR')) / 'offline_decisions.jsonl')
            self.connector.scheduler_step_feedback = self._observe_offline_steps
            self.connector.scheduler_restore_feedback = self.offline_policy.costs.observe_restores
        if options.offline_prefill_coalescing:
            from .prefill_coalescing import OfflinePrefillCoalescer
            baseline = int(_settings.require('KVMEM_PREFILL_CHUNK_TOKENS'))
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
        if not getattr(self, "_bounded_target", False) or _settings.get("KVMEM_LIFECYCLE_TRACE") != "1":
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
        if callable(getattr(gate, 'snapshot', None)):
            report['gpu_capacity'] = gate.snapshot()
        if getattr(self, 'swap_controller', None) is not None:
            report['request_swaps'] = self.swap_controller.snapshot()
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
            cached_views = (_settings.get('KVMEM_LIFECYCLE_SNAPSHOT_CACHE', '0') == '1' and
                            callable(getattr(sessions, 'snapshot_json', None)))
            # Opt-in: the same bytes, with unchanged idle views not re-encoded.
            report['session_views'] = _SESSION_VIEWS_SLOT if cached_views else sessions.snapshot()
            report['session_owners'] = sessions.owner_snapshot()
        path = Path(_settings.require("KVMEM_RUNNER_PROBE_DIR"))
        temporary = path / "scheduler_resources.tmp"
        text = json.dumps(report)
        if report.get('session_views') is _SESSION_VIEWS_SLOT:
            slot = json.dumps(_SESSION_VIEWS_SLOT)
            if text.count(slot) != 1:
                raise RuntimeError('Lifecycle snapshot session-view slot is not unique')
            text = text.replace(slot, sessions.snapshot_json(), 1)
        temporary.write_text(text)
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
        path = Path(_settings.require('KVMEM_RUNNER_PROBE_DIR'))
        temporary = path / 'scheduled_queries.tmp'
        temporary.write_text(json.dumps({"queries": self._scheduled_queries,
            "prefill_cadence": self.prefill_cadence,
            "archive_capacity_deferrals": self.archive_admission.deferred if getattr(self, 'archive_admission', None) else 0,
            "throttle_requests": self._cadence_requested,
            **({"decode_burst": self._burst_stats} if getattr(self, 'prefill_decode_burst', False) else {})}))
        temporary.replace(path / 'scheduled_queries.json')

    def update_from_output(self, *args, **kwargs):
        pool = getattr(self.connector, 'systemone_pool', None)
        if pool is not None:
            output = args[1] if len(args) > 1 else kwargs['model_runner_output']
            meta = getattr(getattr(output, 'kv_connector_output', None), 'kv_connector_worker_meta', None)
            pool.consume(getattr(meta, 'systemone_notices', ()))
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
        controller = getattr(self, 'swap_controller', None)
        if controller is not None and controller.defer_free(request, delay_free_blocks):
            return None, None
        return self._finish_swapped_request(request, delay_free_blocks)

    def _finish_swapped_request(self, request, delay_free_blocks=False):
        getattr(self.connector, 'publication_blocked', set()).discard(request.request_id)
        getattr(self, '_held_capture_positions', {}).pop(request.request_id, None)
        restore_admission = getattr(self.connector, 'restore_admission', None)
        if restore_admission is not None:
            restore_admission.cancel(request)
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        self._lifecycle_finish_generation = getattr(self, '_lifecycle_finish_generation', 0) + 1
        getattr(self, 'prepared_refreshes', {}).pop(request.request_id, None)
        getattr(self, '_deferred_native_padding', {}).pop(request.request_id, None)
        if getattr(self, 'archive_admission', None) is not None:
            self.archive_admission.forget(request.request_id)
        if getattr(self, "_bounded_target", False) and _settings.get("KVMEM_LIFECYCLE_TRACE") == "1":
            report = {"request_id": request.request_id, "status": request.status.name,
                      "at": time.perf_counter(), "history_end": request.num_computed_tokens,
                      "in_flight_tokens": request.num_in_flight_tokens,
                      "refresh_pending": request.request_id in self.transactions,
                      "lifetime_reservation_retained": request.request_id in self.kv_cache_manager.reservations}
            with (Path(_settings.require("KVMEM_RUNNER_PROBE_DIR")) / "scheduler_finished.jsonl").open("a") as stream:
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
        controller = getattr(self, 'swap_controller', None)
        if controller is not None and controller.complete(request_id, command_id):
            return
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
            with (Path(_settings.require("KVMEM_RUNNER_PROBE_DIR")) / "scheduler_refresh.jsonl").open("a") as stream:
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
        controller = getattr(self, 'swap_controller', None)
        if controller is not None and controller.is_suspended(request.request_id):
            return 0
        limit = getattr(self, '_streaming_prefill_limit', None)
        prefill = num_computed_tokens < request.num_prompt_tokens
        if (limit is not None and prefill and request.request_id not in self._streaming_prefills and
                len(self._streaming_prefills) >= limit):
            return 0  # qsa_streaming: the staging run is taken this step.
        tokens = self._reserve_kvmem_lookahead(request, num_computed_tokens, num_new_tokens)
        if limit is not None and prefill and tokens:
            self._streaming_prefills.add(request.request_id)
        if tokens and controller is not None:
            if not self.kv_cache_manager.can_schedule(request, tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    num_external_computed_tokens=max(0, num_computed_tokens-request.num_computed_tokens),
                    has_scheduled_reqs=bool(self.running)):
                return 0
        return tokens

    def _reserve_kvmem_lookahead(self, request, num_computed_tokens, num_new_tokens):
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
            chunk = int(_settings.get('KVMEM_PREFILL_CHUNK_TOKENS','0'))
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
        self._logical_prefill_owners = {}
        self._logical_owner_chunks = {}
        self._logical_owner_max_chunks = getattr(self, 'prefill_physical_max_chunks', 2)
        if not getattr(self, 'prefill_physical_coalescing', False):
            return
        # The first qualification admits exactly one live request. Counting
        # blocked/waiting neighbours as well avoids guessing native composition.
        requests = list(self.running) + list(self._waiting_requests())
        if getattr(self, 'prefill_physical_multi_owner', False) and requests:
            from .multiowner_prefill import owner_chunk_limit
            # Reuse the exact native running-row gates. A pressure transaction
            # or an unpublished running owner cannot enter this forward; its
            # idle score-row share may be used by the remaining owners. Waiting
            # requests remain counted because native may admit them this step.
            step = self.current_step + 1
            blocked = getattr(self.connector, 'publication_blocked', ())
            requests = [item for item in requests if not (
                getattr(item, 'status', None) is RequestStatus.RUNNING and (
                    item.request_id in self.transactions or item.request_id in blocked or
                    item.next_decode_eligible_step > step))]
            if not requests:
                return
            # Keep decode, Query/replay and unknown policy compositions on the
            # original path. All owners share the existing native score bank.
            # Opt-in KVMEM_PREFILL_COALESCE_WITH_DECODE: decoding requests stay
            # ordinary rows of the same step instead; their largest possible
            # sampled-row use (1 + speculative tokens) is reserved from the bank.
            from .request_plan import request_plan
            mixed = getattr(self, 'prefill_coalesce_with_decode', False)
            native_rows, owners = self._physical_native_rows, []
            for item in requests:
                plan = request_plan(item, self.query_policy.block_tokens, self.query_policy.gen_budget)
                history = item.num_computed_tokens
                if (mixed and plan is not None and history >= item.num_prompt_tokens and
                        getattr(item, 'status', None) is RequestStatus.RUNNING):
                    native_rows -= 1 + max(len(getattr(item, 'spec_token_ids', ()) or ()),
                                           getattr(self, 'num_spec_tokens', 0) or 0)
                    continue
                if (plan is None or history >= item.num_prompt_tokens or
                        plan.query is not None and history >= plan.replay_begin or
                        getattr(item, 'spec_token_ids', ())):
                    return
                owners.append(item)
            if not owners or native_rows <= 0:
                return
            requests = owners
            chunk = int(_settings.get('KVMEM_PREFILL_CHUNK_TOKENS', '0'))
            limit = owner_chunk_limit(len(requests), native_rows,
                chunk, self.prefill_physical_max_chunks)
            if getattr(self, 'prefill_physical_fill_rows', False):
                from .multiowner_prefill import owner_chunk_allocations
                limits = owner_chunk_allocations(len(requests), native_rows,
                    chunk, self.prefill_physical_max_chunks, rotation=step)
                limit = max(limits)
                self._logical_owner_chunks = {r.request_id:n for r,n in zip(requests,limits)}
            if limit >= 2:
                self._logical_owner_max_chunks = limit
                self._logical_prefill_owners = {item.request_id: item for item in requests}
            return
        if len(requests) == 1:
            self._logical_prefill_owner = requests[0]

    def _logical_prefill_group(self, request, plan, history, offset, budget, chunk):
        if (not getattr(self, 'prefill_physical_coalescing', False) or chunk not in (1024,2048) or history == 0 or
                (getattr(self, '_logical_prefill_owner', None) is not request and
                 getattr(self, '_logical_prefill_owners', {}).get(request.request_id) is not request) or budget < 2*chunk or
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
        owner_max = getattr(self, '_logical_owner_chunks', {}).get(request.request_id,
            getattr(self, '_logical_owner_max_chunks', getattr(self, 'prefill_physical_max_chunks', 2)))
        room = self.query_policy.capacity-history+offset
        partial = getattr(self, 'prefill_physical_partial_tail', False)
        # Opt-in: one extra span slot for a window-clipped final chunk. The
        # planner admits it only when it is exactly the original clipped step.
        maximum = min(owner_max, budget//chunk,
                      (request.num_prompt_tokens-history-1)//chunk,
                      room//chunk + (1 if partial and room % chunk else 0))
        if maximum < 2:
            return None
        limits = tuple(super(KVMemScheduler, self)._reserve_prefill_lookahead(
            request, history + i * chunk, chunk) for i in range(maximum))
        gates = LogicalPrefillEligibility((request.request_id,), limits, budget,
            native_prefill_lookahead=getattr(self, 'num_prefill_lookahead', 1))
        group, _ = plan_logical_prefill(plan, request_id=request.request_id,
            # An admission receipt, not a worker authority/window generation.
            # Action rebinds to its local generation after validating geometry.
            owner_epoch=('scheduler', self.current_step, request.request_id, offset),
            scheduler_step=self.current_step, history_begin=history, offset=offset,
            prompt_tokens=request.num_prompt_tokens, capacity=self.query_policy.capacity,
            eligibility=gates,logical_chunk_tokens=chunk,
            **({'partial_tail': True} if partial else {}))
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

    def _held_capture_taken(self, request):
        """Whether a released held piece is back at its already-captured position.

        Its 'captured' notice cleared the session candidate, so issuing the P
        capture again would hold the same piece again, forever.
        """
        positions = getattr(self, '_held_capture_positions', {})
        held_at = positions.get(request.request_id)
        if held_at is not None and held_at != request.num_computed_tokens:
            del positions[request.request_id]           # its held piece has run
            return False
        return held_at is not None

    def _hold_for_capture(self, connector, output, request_id, query_tokens, index, *, alone=False):
        """Opt-in KVMEM_HELD_P_CAPTURE: move a final prefill piece behind its P capture.

        Only a piece that is still in this batch beside peers, outside any
        refresh/restore transaction or coalesced prefill group, is held. Like a
        suspended Query, its allocation reaches the worker but not its tokens;
        publication_blocked keeps it out of later batches until the worker
        reports the capture (connector.held_captures).
        """
        if (request_id not in output.num_scheduled_tokens or request_id in self.transactions or
                request_id in getattr(self, '_logical_prefill_groups', {}) or
                len(output.num_scheduled_tokens) < (1 if alone else 2)):
            return
        command = connector.sessions.commands.captures[index]
        if command.request_id != request_id or command.held:
            raise RuntimeError('Held P capture command does not match its request')
        from dataclasses import replace
        connector.sessions.commands.captures[index] = replace(command, held=True)
        self._preserve_deferred_padding(self.requests[request_id], output, query_tokens)
        output.num_scheduled_tokens.pop(request_id)
        output.total_num_scheduled_tokens -= query_tokens
        output.scheduled_spec_decode_tokens.pop(request_id, None)
        connector.held_captures.add(request_id)
        connector.publication_blocked.add(request_id)
        self.__dict__.setdefault('_held_capture_positions', {})[request_id] = \
            self.requests[request_id].num_computed_tokens
        self.held_p_captures = getattr(self, 'held_p_captures', 0) + 1

    def _build_kv_connector_meta(self, connector, output):
        if getattr(self, "_bounded_target", False):
            from .session_control import held_p_capture
            hold_captures = (held_p_capture() if connector.sessions is not None and not self.session_capture_decode
                             and getattr(connector, 'tp_feedback', None) is None else 0)
            held = []
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
                if connector.sessions is not None and not self._held_capture_taken(request):
                    issued = len(connector.sessions.commands.captures)
                    connector.sessions.capture_before_final_prefill(request, query_tokens)
                    if (hold_captures and len(connector.sessions.commands.captures) > issued and
                            request_id not in getattr(self, 'prepared_refreshes', {})):
                        # Never a prepared public Query replay: it must run now.
                        held.append((request_id, query_tokens, issued))
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
                fork_pool = getattr(connector, 'systemone_pool', None)
                if (fork_pool is not None and action is None and
                        request.num_computed_tokens >= request.num_prompt_tokens and
                        fork_pool.take_refresh_probe(request_id)):
                    action = 'pressure'  # Explicit byte-audit injection; ordinary admission is unchanged.
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
                            if (getattr(self, 'prefill_pressure_no_replay', False) and action == 'pressure' and
                                    guided is None and not padding):
                                # Opt-in: the window-boundary block is not replayed as its
                                # own step after the refresh; it starts the next ordinary
                                # (coalesced) group. Selection uses accepted history only.
                                replay_tokens, tokens = 0, ()
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
            for request_id, query_tokens, index in held:
                self._hold_for_capture(connector, output, request_id, query_tokens, index,
                                       alone=hold_captures == 2)
        for request_id, query_tokens in output.num_scheduled_tokens.items():
            deferred = getattr(self, '_deferred_native_padding', {}).pop(request_id, None)
            if deferred is not None and (query_tokens != deferred[1] or
                    output.scheduled_spec_decode_tokens.get(request_id) != [-1] * (deferred[1] - 1)):
                raise RuntimeError('Deferred native tail resumed without its complete placeholder shape')
        if getattr(self.kv_cache_manager, 'refresh_control_allocations', None):
            controls = [(rid, history, count) for rid, (_, history, count) in
                        self.kv_cache_manager.refresh_control_allocations.items()]
            self.kv_cache_manager.validate_refresh_controls(output.num_scheduled_tokens, self.transactions)
            with (Path(_settings.require('KVMEM_RUNNER_PROBE_DIR'))/'refresh_control_allocations.jsonl').open('a') as stream:
                for rid, history, count in controls:
                    stream.write(json.dumps(dict(at=time.monotonic(),request_id=rid,
                        history=history,replay_tokens=count,public_peer_count=len(output.num_scheduled_tokens),
                        private_pressure_verified=True))+'\n')
        metadata = super()._build_kv_connector_meta(connector, output)
        if getattr(self, 'prefill_physical_coalescing', False):
            groups = {rid: group for rid, group in self._logical_prefill_groups.items()
                      if rid in output.num_scheduled_tokens}
            if groups and ((not getattr(self, 'prefill_physical_multi_owner', False) and
                    (len(groups) != 1 or len(output.num_scheduled_tokens) != 1)) or
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
        with (Path(_settings.require('KVMEM_RUNNER_PROBE_DIR')) / 'coalescing_decisions.jsonl').open('a') as stream:
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
                    entry is None and sessions.preserve_history and len(sessions.entries) >= sessions.capacity and
                    not (getattr(sessions, 'idle_lru_eviction', False) and
                         any(e.phase == 'idle' for e in sessions.entries.values()))):
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
            path = Path(_settings.require('KVMEM_RUNNER_PROBE_DIR')) / 'phase_decisions.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(dict(at=now, step=self.current_step,
                    decision=vars(self._phase_decision), public_prefill_tokens=prefills,
                    decode_query_tokens=decodes, maintenance_candidates=sorted(self._phase_maintenance),
                    observation=observation, cumulative=self.phase_policy.snapshot())) + '\n')

    def _decode_ready(self, *, step=None, minimum=1):
        """Whether at least ``minimum`` running requests can take a decode step now."""
        step = self.current_step if step is None else step
        blocked = getattr(self.connector, 'publication_blocked', ())
        ready = (not r.is_prefill_chunk and r.num_computed_tokens >= r.num_prompt_tokens
            and not r.is_finished() and not r.num_in_flight_tokens
            and r.request_id not in self.transactions
            and r.request_id not in blocked
            and r.next_decode_eligible_step <= step
            for r in self.running)
        if minimum <= 1:
            return any(ready)
        return sum(ready) >= minimum

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
        share = getattr(self, 'prefill_decode_share', None)
        if not getattr(self, '_bounded_target', False) or (
                getattr(self, 'prefill_cadence', 1) == 1 and share is None):
            return False
        if getattr(self, 'prefill_decode_burst', False):
            return False    # decided by _burst_begin, before the step is prepared
        # Opt-in KVMEM_PREFILL_CADENCE_MIN_DECODERS: a decode-only turn serves
        # every ready decoder at about one decode step's cost, so it is only
        # requested when at least that many requests are ready to decode.
        minimum = getattr(self, 'prefill_cadence_min_decoders', 1)
        mixed = ((any(r.is_prefill_chunk for r in self.running) or any(self._waiting_queues())) and
                 self._decode_ready(minimum=minimum))
        if not mixed or self.prefill_capacity_bound:
            self._mixed_steps = 0
            self._share_credit, self._share_previous = 0., None
            return False
        if share is not None:
            throttle = self._share_turn(share)
        else:
            self._mixed_steps += 1
            throttle = self._mixed_steps % self.prefill_cadence != 0
        self._cadence_requested += int(throttle)
        return throttle

    SHARE_CREDIT_LIMIT = 1.

    def _share_turn(self, share, now=None):
        """Time-share decode turns: after a step that released prefill and took
        P seconds (wall time to this schedule call), decoders earn
        P * share / (1 - share) seconds of decode-only steps; each decode-only
        step spends its own duration. A turn is requested while credit is left.
        """
        now = time.perf_counter() if now is None else now
        previous = self._share_previous
        if previous is not None:
            started, decode_turn = previous
            elapsed = max(0., now - started)
            if decode_turn:
                self._share_credit -= elapsed
            else:
                self._share_credit = min(self.SHARE_CREDIT_LIMIT,
                                         self._share_credit + elapsed * share / (1 - share))
        throttle = self._share_credit > 0
        self._share_previous = (now, throttle)
        return throttle

    SHARE_BURST_SECONDS = .25

    def _burst_begin(self, now=None):
        """Opt-in KVMEM_PREFILL_DECODE_BURST: keep decode rows out of prefill steps.

        While a prefill can be scheduled, ready decoders no longer ride it:
        they are held out of that step (it stays a pure prefill step) and earn
        the usual time-share credit. Once SHARE_BURST_SECONDS is owed they take
        consecutive decode-only steps until the credit is spent. With nothing
        to arbitrate (no ready decoder, no schedulable prefill, capacity-bound
        admission) scheduling is unchanged and decode steps are not charged, so
        the share is the GPU fraction decoders are guaranteed, not a ceiling.
        No query size, request order or selection input changes.
        Returns (throttle, holds); the holds must be passed to _burst_end.
        """
        share = self.prefill_decode_share
        step = self.current_step + 1    # native schedule increments before its checks
        blocked = getattr(self.connector, 'publication_blocked', ())

        def ready(request):
            return (request.request_id not in self.transactions and request.request_id not in blocked
                    and not request.is_finished() and not request.num_in_flight_tokens
                    and request.next_decode_eligible_step <= step)

        decoding = [r for r in self.running if not r.is_prefill_chunk
                    and r.num_computed_tokens >= r.num_prompt_tokens and not r.is_finished()]
        decoders = [r for r in decoding if ready(r)]
        pending = [r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens]
        prefills = [r.request_id for r in pending if ready(r)]
        admitted = len(prefills)
        waiting = self._waiting_requests()
        if len(self.running) < self.scheduler_config.max_num_seqs:
            prefills += [r.request_id for r in waiting]
        minimum = getattr(self, 'prefill_burst_min_tokens', 0)
        small = bool(minimum and prefills and admitted == len(prefills) and sum(
            r.num_prompt_tokens - r.num_computed_tokens for r in pending if ready(r)) < minimum)
        now = time.perf_counter() if now is None else now
        previous, self._share_previous = self._share_previous, None
        if previous is not None:
            started, decode_turn = previous
            elapsed = max(0., now - started)
            if decode_turn:
                self._share_credit -= elapsed
            else:
                self._share_credit = min(self.SHARE_CREDIT_LIMIT,
                                         self._share_credit + elapsed * share / (1 - share))
        if not decoding or not (pending or waiting):
            # One side is gone: nothing is owed. A decoder that is only busy
            # (its own refresh, an in-flight step) keeps the credit it earned.
            self._share_credit, self._burst_active = 0., False
        signature = frozenset(prefills)
        if signature != self._burst_missed:
            self._burst_missed = None
        if small and decoders and not self._burst_active:
            # Size-aware burst: decoders ride this small prefill step; nothing
            # is held, so nothing is earned (an active burst still finishes).
            self._burst_stats['ride_along'] = self._burst_stats.get('ride_along', 0) + 1
            return False, None
        if (not decoders or not prefills or self.prefill_capacity_bound or
                self._burst_missed is not None):
            return False, None
        if self._burst_active:
            self._burst_active = self._share_credit > 0
        elif self._share_credit >= self.SHARE_BURST_SECONDS:
            self._burst_active = True
            self._burst_stats['bursts'] += 1
        throttle = self._burst_active
        self._share_previous = (now, throttle)
        self._cadence_requested += int(throttle)
        if throttle:
            self._burst_stats['decode_steps'] += 1
            return True, None
        # The hold expires by itself at the next step; transactions that begin
        # meanwhile save and restore it like any other eligibility step.
        hold = step + 1
        held = [(r, r.next_decode_eligible_step) for r in decoders]
        for request in decoders:
            request.next_decode_eligible_step = hold
        self._burst_stats['held_steps'] += 1
        self._burst_stats['held_decoders'] += len(held)
        return False, (hold, signature, held)

    def _burst_end(self, holds, output=None):
        if holds is None:
            return
        hold, signature, held = holds
        for request, previous in held:
            if request.next_decode_eligible_step == hold:
                request.next_decode_eligible_step = previous
        if output is None:
            return
        ids = {request.request_id for request, _ in held}
        if all(rid in ids for rid in output.num_scheduled_tokens):
            # The prefill that justified the hold was not admitted (capacity,
            # restore credit, a policy gate). Do not hold for this same set
            # again, and neither earn nor spend for a step that ran nothing.
            self._burst_missed = signature
            self._share_previous = None
            self._burst_stats['misses'] += 1

    def _schedule_time_acks(self):
        """Apply pressure handoffs that became ready after the last feedback.

        The synchronous single-process engine runs the worker on this thread
        and has no batch in flight here, so the worker state equals the end of
        the previous post-forward. Completions travel the native scheduler
        completion path (update_connector_output -> _complete_refresh).
        """
        if not self.transactions:
            return
        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
        if not has_kv_transfer_group():
            return
        worker = get_kv_transfer_group()
        if worker is self.connector:
            raise RuntimeError('Schedule-time ACK found the scheduler connector, not the worker')
        wait = 0.
        idle_ms = float(_settings.get('KVMEM_LAYERWISE_PRESSURE_IDLE_ACK_WAIT_MS', '0'))
        promise = _settings.get('KVMEM_LAYERWISE_PRESSURE_PROMISE', '0') == '1'
        idle = bool((idle_ms or promise) and all(r.request_id in self.transactions for r in self.running)
                    and not any(self._waiting_queues()))
        if idle_ms and idle:
            # Nothing else is schedulable this step: waiting for a plan that is
            # already being prepared cannot delay any other request's compute.
            wait = min(idle_ms, 200.) / 1000
        if promise and idle:
            # Opt-in promised ACK (pressure_promise): only when the GPU would idle.
            meta = worker.take_schedule_completions(wait, idle=True)
        else:
            meta = worker.take_schedule_completions(wait) if wait else worker.take_schedule_completions()
        if meta is None:
            return
        from types import SimpleNamespace
        self.connector.update_connector_output(SimpleNamespace(
            kv_connector_worker_meta=meta, finished_sending=None, finished_recving=None))

    def schedule(self, throttle_prefills=False):
        if getattr(self, '_schedule_ack', False):
            self._schedule_time_acks()
        if getattr(self, 'swap_controller', None) is not None:
            self.swap_controller.before_schedule()
        if getattr(self.connector, 'active_residency', None) is not None:
            self.connector.active_residency.expire()
        if getattr(self, 'offline_policy', None) is not None:
            self.offline_policy.prepare()
        holds = output = None
        if getattr(self, 'prefill_decode_burst', False) and getattr(self, '_bounded_target', False):
            # Before any step preparation: held decoders must reserve no rows.
            burst, holds = self._burst_begin()
            throttle_prefills = throttle_prefills or burst
        try:
            self._prepare_coalescing()
            self._prepare_logical_prefill()
            self._prepare_phase()
            self._prepare_prefill_budget()
            armed = Path(_settings.require("KVMEM_RUNNER_PROBE_DIR")) / "armed"
            if not getattr(self, "_bounded_target", False) and not self._probe_submitted and armed.is_file():
                for request in self.running:
                    if request.num_computed_tokens == request.num_prompt_tokens and request.num_output_tokens:
                        self.begin_refresh(request, request.num_computed_tokens)
                        self._probe_submitted = True
                        break
            if getattr(self, '_bounded_target', False):
                self.kv_cache_manager.begin_step()
            self._streaming_prefills = set()
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
        finally:
            self._burst_end(holds, output)
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
