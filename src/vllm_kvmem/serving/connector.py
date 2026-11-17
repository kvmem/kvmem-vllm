"""vLLM's supported connector metadata channel for working-set transactions.

No prefix hits are invented. The worker runtime owns capture/storage/assembly;
the connector carries commands and completion acknowledgments to the scheduler.
"""
from dataclasses import dataclass, field
import os

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorWorkerMetadata, SupportsHMA,
    KVConnectorRole,
)

from .window_manager import RefreshTicket
from .terminal import TerminalPrefix
from ..core.archive_admission import ArchiveCapacity
from .session_control import SessionCommands
from .request_swap_worker import SwapCommand
from ..core import settings as _settings


@dataclass(frozen=True)
class RestoreCommand:
    request_id: str
    command_id: int
    generation: int
    history_end: int
    window_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    asynchronous: bool = True
    resident: bool = False


@dataclass(frozen=True)
class RefreshCommand:
    request_id: str
    command_id: int
    groups: tuple[tuple[int, RefreshTicket], ...]
    query_span: tuple[int, int] | None = None
    selection: str = 'chunk'
    mandatory_spans: tuple = ()
    input_tokens: tuple[int, ...] = ()
    asynchronous: bool = False
    speculative_padding: int = 0
    message_spans: tuple = ()
    locality_spans: tuple = ()
    locality_provenance: tuple = ()
    guided_reason: str = ''
    guided_private: bool = False
    guided_boundary: bool | None = None
    pressure_stage: str = ''


@dataclass
class ControlMetadata(KVConnectorMetadata):
    commands: list[RefreshCommand | RestoreCommand | SwapCommand] = field(default_factory=list)
    cancelled: set[str] = field(default_factory=set)
    terminal_prefixes: dict[str, TerminalPrefix] = field(default_factory=dict)
    sessions: SessionCommands = field(default_factory=SessionCommands)
    maintenance_costs: dict[str, float | None] = field(default_factory=dict)
    cost_classification: dict[str, dict] = field(default_factory=dict)
    logical_prefill_groups: dict = field(default_factory=dict)
    logical_prefill_step: int | None = None
    systemone_pool: dict | None = None
    # Full rows after a fenced refresh, on engines without a native row-update field.
    block_table_updates: dict[str, tuple[list[int], ...]] = field(default_factory=dict)


@dataclass
class ControlCompletions(KVConnectorWorkerMetadata):
    completed: list[tuple[str, int]] = field(default_factory=list)
    selected_windows: dict[str, int] = field(default_factory=dict)
    archive_sequence: int | None = None
    archive_capacity: dict[str, ArchiveCapacity] = field(default_factory=dict)
    sessions: list = field(default_factory=list)
    cpu_archive_bytes: int | None = None
    step_timings: list = field(default_factory=list)
    restore_timings: list = field(default_factory=list)
    publication_sequence: int | None = None
    publication_blocked: set[str] = field(default_factory=set)
    # Opt-in session tiering: ('out'|'in', namespace) CPU-residency transitions.
    tiering: list = field(default_factory=list)
    # Opt-in KVMEM_HELD_P_CAPTURE: held final pieces whose P capture finished.
    held_captures_done: list = field(default_factory=list)

    systemone_notices: list = field(default_factory=list)
    shards: dict[int, 'ControlCompletions'] | None = None

    def aggregate(self, other):
        if (not isinstance(other, ControlCompletions) or not self.shards or not other.shards
                or self.shards.keys() & other.shards.keys()):
            raise ValueError('Invalid or duplicate tensor-parallel connector feedback')
        return ControlCompletions(shards=self.shards | other.shards)


class KVMemConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        from .tensor_parallel import validate_parallelism
        size = validate_parallelism(vllm_config.parallel_config)
        self.tp_rank = self.tp_feedback = None
        if size > 1:
            if role == KVConnectorRole.SCHEDULER:
                from .tp_feedback import TensorParallelFeedback
                self.tp_feedback = TensorParallelFeedback(size)
            else:
                from vllm.distributed import get_tp_group
                self.tp_rank = get_tp_group().rank_in_group
        self.systemone_pool = None
        self.systemone_worker = None
        self.commands = []
        self.pending = {}
        self.cancelled = set()
        self.cancel_updates = set()
        self.terminal_prefixes = {}
        self.bounded_target = _settings.get('KVMEM_BOUNDED_TARGET') == '1'
        self.worker_jobs = {}
        self.worker_completions = []
        self.worker_selected_windows = {}
        self.runtime = None
        self.swap_worker = None
        self.request_swap_pending = None
        self.scheduler_completion = None
        options = vllm_config.kv_transfer_config.kv_connector_extra_config.get('kvmem', {})
        self.gpu_memory_admission = options.get('gpu_memory_admission', 'static')
        from .session_tiering import tiering_prefetch, tiering_queue
        self.tiering_prefetch = bool(options.get('cpu_session_ttl_ms')) and tiering_prefetch()
        self.tiering_queue = bool(options.get('cpu_session_ttl_ms')) and tiering_queue()
        self.tiering_out = set()        # sessions the worker reports out of CPU RAM
        self.tiering_requested = set()  # out-of-RAM sessions whose promotion was requested
        self.tiering_queue_waits = 0
        self.async_retirement = options.get('async_archive_retirement', False)
        self.physical_retirement = options.get('archive_backend') == 'tiered'
        repository_gc = options.get('repository_gc', False)
        if type(self.async_retirement) is not bool or type(repository_gc) is not bool:
            raise ValueError('Retirement and repository GC options must be boolean')
        if (self.async_retirement or repository_gc) and _settings.get('KVMEM_BOUNDED_TARGET') != '1':
            raise ValueError('Retirement/GC requires bounded authority admission')
        if self.async_retirement or repository_gc or self.physical_retirement:
            from ..adapters.engine_api import require_contract
            require_contract('retirement')
        self.retirement_done = None
        self.archive_snapshot = None
        self.scheduler_archive_update = None
        self.archive_sequence = 0
        self.sessions = None  # Scheduler ledger; worker lifecycle is separate.
        self.request_plan_memory = None
        self.request_plan_pressure_policy = 'recency'
        self.capacity_rejections = {}
        self.history_admission = None
        self.history_usage = None
        self.session_worker = None
        self.step_timings = None
        self.scheduler_step_feedback = None
        self.restore_timings = None
        self.restore_completion_poll = None
        self.pressure_completion_poll = None
        self.pressure_schedule_poll = None
        self.query_completion_poll = None
        self.scheduler_restore_feedback = None
        self.active_residency = None
        self.tool_grace_allows = None
        self.restore_admission = None
        self.publication_snapshot = None
        self.publication_sequence = 0
        self.publication_blocked = set()
        self.held_captures = set()      # opt-in KVMEM_HELD_P_CAPTURE (scheduler side)
        self.publication_live = None
        if (self.bounded_target and role == KVConnectorRole.SCHEDULER
                and kv_cache_config is not None and _settings.get('KVMEM_RUNNER_PROBE_DIR')):
            from ..adapters.cache_manifest import write_mean_k_geometry
            write_mean_k_geometry(vllm_config, kv_cache_config, _settings.require('KVMEM_RUNNER_PROBE_DIR'))

    def queue(self, command):
        if command.request_id in self.pending:
            raise RuntimeError("Duplicate active working-set transaction")
        self.pending[command.request_id] = command.command_id
        self.commands.append(command)

    def build_connector_meta(self, scheduler_output):
        result = ControlMetadata(self.commands, self.cancel_updates, self.terminal_prefixes)
        if self.systemone_pool is not None:
            result.systemone_pool = self.systemone_pool.metadata()
        self.commands, self.cancel_updates = [], set()
        self.terminal_prefixes = {}
        if self.sessions is not None:
            result.sessions = self.sessions.take_commands()
        return result

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ControlMetadata):
            raise TypeError("Unexpected KVMem control metadata")
        self.cancelled.update(metadata.cancelled)
        if metadata.sessions.evictions:
            if self.session_worker is None:
                raise RuntimeError('Session eviction received without its worker')
            self.session_worker.evict(metadata.sessions.evictions)
        if metadata.sessions.returning and self.session_worker is not None:
            self.session_worker.returning(metadata.sessions.returning)
        for command in metadata.commands:
            if command.request_id in self.worker_jobs:
                raise RuntimeError("Duplicate worker transaction")
            if isinstance(command, SwapCommand):
                if self.swap_worker is None:
                    raise RuntimeError('Request swap command received without its worker')
                job = self.swap_worker.execute(command)
            else:
                if self.runtime is None:
                    raise RuntimeError("Working-set command received without a worker runtime")
                job = self.runtime(command)
            self.worker_jobs[command.request_id] = (command, job)

    def get_finished(self, finished_req_ids):
        if self.restore_completion_poll is not None:
            self.restore_completion_poll()
        if self.pressure_completion_poll is not None:
            self.pressure_completion_poll()
        if self.query_completion_poll is not None:
            self.query_completion_poll()
        for request_id, (command, job) in list(self.worker_jobs.items()):
            if job.query():
                self.worker_completions.append((request_id, command.command_id))
                if getattr(job, "selected_tokens", None) is not None:
                    self.worker_selected_windows[request_id] = job.selected_tokens
                del self.worker_jobs[request_id]
        done = set()
        for request_id in self.cancelled:
            if request_id in self.worker_jobs:
                continue
            if self.swap_worker is not None and request_id in self.swap_worker.checkpoints:
                continue  # Snapshot ownership ends only after an explicit drop.
            if self.retirement_done is not None:
                if not self.retirement_done(request_id):
                    continue
            elif self.async_retirement or self.physical_retirement:
                raise RuntimeError('Missing authority retirement completion handler')
            done.add(request_id)
        self.cancelled.difference_update(done)
        return done or None, None

    def take_schedule_completions(self, wait_seconds=0., idle=False):
        """Worker half of the opt-in schedule-time pressure ACK.

        Only pressure handoffs dispatched by this call are reported, exactly as
        get_finished would report them in the next batch feedback; every other
        completion, cancellation and lifecycle notice keeps its native path.
        """
        poll = self.pressure_schedule_poll
        if poll is None:
            return None
        completed, windows = [], {}
        for request_id in (poll(wait_seconds, idle=True) if idle else
                           poll(wait_seconds) if wait_seconds else poll()):
            entry = self.worker_jobs.get(request_id)
            if entry is None:
                continue
            command, job = entry
            if not job.query():
                raise RuntimeError('Schedule-time handoff did not complete its worker job')
            completed.append((request_id, command.command_id))
            if getattr(job, "selected_tokens", None) is not None:
                windows[request_id] = job.selected_tokens
            del self.worker_jobs[request_id]
        if not completed:
            return None
        return ControlCompletions(completed, windows)

    def build_connector_worker_meta(self):
        capacity = self.archive_snapshot() if self.archive_snapshot is not None else None
        notices = self.session_worker.notices() if self.session_worker is not None else []
        timings = self.step_timings() if self.step_timings is not None else []
        restores = self.restore_timings() if self.restore_timings is not None else []
        publication = self.publication_snapshot() if self.publication_snapshot is not None else None
        if publication is not None:
            self.publication_sequence += 1
            self.publication_blocked = set(publication)
        fork_notices = self.systemone_worker.take_notices() if self.systemone_worker is not None else []
        tiering = getattr(self.session_worker, 'tiering', None) if self.session_worker is not None else None
        tier_events = tiering.take_events() if tiering is not None else []
        take_held = getattr(self.session_worker, 'take_held_done', None) if self.session_worker is not None else None
        held_done = take_held() if take_held is not None else []
        if not fork_notices and not self.worker_completions and capacity is None and not notices and not timings and not restores and publication is None and not tier_events and not held_done:
            return None
        result = ControlCompletions(self.worker_completions, self.worker_selected_windows)
        result.systemone_notices = fork_notices
        result.sessions = notices
        result.tiering = tier_events
        result.held_captures_done = held_done
        result.step_timings = timings
        result.restore_timings = restores
        if publication is not None:
            result.publication_sequence = self.publication_sequence
            result.publication_blocked = set(self.publication_blocked)
        if notices and self.history_usage is not None:
            result.cpu_archive_bytes = self.history_usage()
        if capacity is not None:
            self.archive_sequence += 1
            result.archive_sequence = self.archive_sequence
            result.archive_capacity = capacity
        self.worker_completions = []
        self.worker_selected_windows = {}
        return result if self.tp_rank is None else ControlCompletions(shards={self.tp_rank: result})

    def update_connector_output(self, connector_output):
        metadata = connector_output.kv_connector_worker_meta
        if metadata is not None:
            if not isinstance(metadata, ControlCompletions):
                raise TypeError("Unexpected worker completion metadata")
            if self.tp_feedback is not None:
                if not metadata.shards:
                    raise ValueError('Missing tensor-parallel worker identity')
                metadata = self.tp_feedback.collect(metadata.shards, self.pending)
            if metadata.publication_sequence is not None:
                if metadata.publication_sequence <= self.publication_sequence:
                    raise ValueError('Stale publication readiness feedback')
                self.publication_sequence = metadata.publication_sequence
                live = self.publication_live() if self.publication_live is not None else metadata.publication_blocked
                self.publication_blocked = set(metadata.publication_blocked).intersection(live)
            held = getattr(self, 'held_captures', None)
            if held:
                # A held final piece stays out of batches until its worker
                # capture is reported, whatever the publication snapshot says.
                held.difference_update(getattr(metadata, 'held_captures_done', ()) or ())
                self.publication_blocked |= held
            if metadata.step_timings and self.scheduler_step_feedback is not None:
                self.scheduler_step_feedback(metadata.step_timings)
            if self.tiering_queue:
                for kind, namespace in getattr(metadata, 'tiering', ()) or ():
                    if kind == 'out':
                        self.tiering_out.add(tuple(namespace))
                    else:
                        self.tiering_out.discard(tuple(namespace))
                        self.tiering_requested.discard(tuple(namespace))
            if metadata.restore_timings and self.scheduler_restore_feedback is not None:
                self.scheduler_restore_feedback(metadata.restore_timings)
            for notice in metadata.sessions:
                if self.sessions is None:
                    raise RuntimeError('Unexpected session worker completion')
                if self.sessions.consume(notice) is False:
                    continue
                if self.history_admission is not None:
                    self.history_admission.consume(notice)
                if self.active_residency is not None:
                    self.active_residency.consume(notice)
            if metadata.archive_sequence is not None:
                if self.scheduler_archive_update is None:
                    raise RuntimeError('Unexpected archive capacity feedback')
                self.scheduler_archive_update(metadata.archive_sequence, metadata.archive_capacity)
            for request_id, command_id in metadata.completed:
                if self.pending.get(request_id) != command_id:
                    raise ValueError("Stale or foreign worker completion")
                if self.scheduler_completion is None:
                    raise RuntimeError("Missing scheduler completion handler")
                if request_id in metadata.selected_windows:
                    self.scheduler_completion(request_id, command_id,
                                              selected_tokens=metadata.selected_windows[request_id])
                else:
                    self.scheduler_completion(request_id, command_id)
                del self.pending[request_id]
        if self.history_admission is not None:
            for rid in connector_output.finished_sending or ():
                self.history_admission.released(rid)
            if metadata is not None and metadata.cpu_archive_bytes is not None:
                self.history_admission.observe(metadata.cpu_archive_bytes)
        self.cancelled.difference_update(connector_output.finished_sending or ())

    def on_new_request(self, request):
        super().on_new_request(request)
        if self.tiering_prefetch and self.sessions is not None:
            namespace = self.sessions.namespace(request)
            if namespace is not None and not (getattr(request, 'kv_transfer_params', None) or {}).get(
                    'kvmem_release_session'):
                self.sessions.commands.returning.append(namespace)
        residency = self.active_residency
        model = getattr(residency, 'wait_model', None)
        if residency is not None and (residency.tool_grace is not None or model is not None):
            namespace = self.sessions.namespace(request) if self.sessions is not None else None
            if (getattr(request, 'kv_transfer_params', None) or {}).get('kvmem_release_session'):
                namespace = None
            if residency.tool_grace is not None:
                residency.tool_grace.arrival(request.request_id, namespace)
            if model is not None:
                model.arrival(namespace)      # active_resident_victim: expected_wait

    def request_finished_all_groups(self, request, block_ids):
        residency = self.active_residency
        model = getattr(residency, 'wait_model', None)
        if residency is not None and (residency.tool_grace is not None or model is not None):
            successful = (getattr(request, 'num_output_tokens', 0) > 0 and
                          getattr(getattr(request, 'status', None), 'name', '') != 'FINISHED_ABORTED')
            if residency.tool_grace is not None:
                residency.tool_grace.finished(request.request_id, successful=successful)
            if model is not None and successful and self.sessions is not None:
                model.finished(self.sessions.namespace(request))
        self.capacity_rejections.pop(request.request_id, None)
        self.publication_blocked.discard(request.request_id)
        getattr(self, 'held_captures', set()).discard(request.request_id)
        if self.restore_admission is not None:
            # Return a provisional hot view before SessionLedger aborts import
            # preparation. Allocated permits survive cancellation until ACK.
            self.restore_admission.cancel(request)
        if self.history_admission is not None:
            self.history_admission.finished(request)
        if self.sessions is not None:
            self.sessions.finished(request)
        if self.bounded_target:
            if request.request_id in self.terminal_prefixes:
                raise RuntimeError('Duplicate terminal request boundary')
            self.terminal_prefixes[request.request_id] = TerminalPrefix.from_request(request)
        if request.request_id not in self.pending and not (self.async_retirement or self.physical_retirement):
            return False, None
        self.cancelled.add(request.request_id)
        self.cancel_updates.add(request.request_id)
        return True, None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        from .request_plan import RequestCapacityError
        if request.request_id in self.capacity_rejections:
            return None, False
        try:
            return self._get_matched_tokens(request, num_computed_tokens)
        except RequestCapacityError as error:
            # Leave the native waiting queues intact during iteration. The
            # scheduler publishes a request-local error through native finish
            # handling, including when this step has zero model tokens.
            self.capacity_rejections[request.request_id] = str(error)
            # The client sees only a generic internal error; keep the reason.
            from vllm.logger import init_logger
            init_logger('vllm.kvmem.connector').warning(
                'KVMem rejected request %s (%d prompt tokens): %s', request.request_id,
                getattr(request, 'num_prompt_tokens', -1), error)
            return None, False

    def _get_matched_tokens(self, request, num_computed_tokens):
        if self.tiering_out and self.sessions is not None:
            namespace = self.sessions.namespace(request)
            if namespace is not None and tuple(namespace) in self.tiering_out:
                # Opt-in session tiering: wait in vLLM's queue, holding no GPU
                # slot, until the worker reports this session back in RAM. The
                # promotion is requested once per out-of-RAM period (idempotent
                # on the worker), also when arrival prefetch is off.
                if tuple(namespace) not in self.tiering_requested:
                    self.tiering_requested.add(tuple(namespace))
                    self.sessions.commands.returning.append(namespace)
                self.tiering_queue_waits += 1
                return None, False
        incremental_session = (self.gpu_memory_admission == 'incremental' and
                               bool(getattr(request, 'session_id', None)))
        if incremental_session:
            from .request_plan import RequestCapacityError
            from .session_control import SessionLedger
            from .session_swap import owned_session_admission
            if not isinstance(self.sessions, SessionLedger):
                raise RequestCapacityError('Incremental session requests require an owned session ledger')
            if not owned_session_admission(self.sessions, request):
                return None, False
        if self.tool_grace_allows is not None and not self.tool_grace_allows(request):
            return None, False
        matched = getattr(self.history_admission, 'matched_admission', False)
        if self.history_admission is not None and not matched and not self.history_admission.allows(request):
            return None, False  # Native waiting loop skips only this request.
        if self.sessions is not None:
            tokens = self.sessions.lookup(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if matched and not self.history_admission.allows(request):
                return None, False
            if tokens and self.restore_admission is not None and not self.restore_admission.lookup(
                    request, self.sessions.requests[request.request_id]):
                return None, False
            return tokens, False
        if matched and not self.history_admission.allows(request):
            return None, False
        if self.request_plan_memory is not None:
            from .request_plan import request_plan
            config = self.request_plan_memory
            plan = request_plan(request, config.block_tokens, config.gen_budget)
            if plan is not None:
                plan.validate_capacity(request.num_prompt_tokens, config.select_budget,
                    sink_blocks=config.sink_blocks, recent_blocks=config.recent_blocks,
                    pressure_policy=self.request_plan_pressure_policy)
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self.history_admission is not None:
            entry = self.sessions.requests.get(request.request_id) if self.sessions is not None else None
            self.history_admission.allocated(request, entry.generation if entry else 0)
        if self.sessions is not None:
            self.sessions.allocated(request, num_external_tokens)
            return
        if num_external_tokens:
            raise ValueError("KVMem reselection is not an external contiguous prefix hit")

    def wait_for_layer_load(self, layer_name):
        pass  # Pending requests are suspended; only ready requests run forward.

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass  # The raw capture adapter saves pre-RoPE authorities, not rotated KV.

    def wait_for_save(self):
        pass

    def has_pending_push_work(self):
        return bool(self.pending or self.cancelled or self.publication_blocked or
                    (self.request_swap_pending is not None and self.request_swap_pending()) or
                    (self.sessions is not None and self.sessions.has_pending_work()))
