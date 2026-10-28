"""Bounded worker ownership for exact, sealed prompt checkpoints.

The serving coordinator must supply an authenticated session namespace, serialize
turns within it, and detach the authority before retire(). This pool owns no GPU
pages. Claims transfer an idle authority only after native reconstruction has
completed; the scheduler must not advertise a hit before idle() reports readiness.
"""
from array import array
from contextlib import nullcontext
from .prompt_spans import selection_metadata

from dataclasses import dataclass, replace
from enum import Enum, auto
import time


class CheckpointPhase(Enum):
    LIVE = auto()       # Capture runtime still owns the executing authority.
    RETIRING = auto()   # Detached; rewind/publication/readers must drain.
    IDLE = auto()       # Exact saved authority prefix is available to import.
    CLAIMED = auto()    # New native allocation/restore owns the checkpoint.
    CLOSING = auto()    # Evicted; no new lookup may see it, readers still drain.


def token_bytes(tokens, count, *, native=False):
    if count <= 0 or len(tokens) <= count:
        raise ValueError('Prompt checkpoint must leave at least one token for native prefill')
    if native and type(tokens) in (tuple, list):
        from ._native import checkpoint_token_bytes
        return checkpoint_token_bytes(tokens, count)
    prefix = tokens[:count]
    if any(type(token) is not int or not 0 <= token < 2**32 for token in prefix):
        raise ValueError('Prompt identity requires exact unsigned token IDs')
    result = array('I', prefix)
    if result.itemsize != 4:
        raise RuntimeError('Unqualified token identity storage width')
    return result.tobytes()


@dataclass(eq=False)
class PromptCheckpoint:
    namespace: object
    request_id: str
    authority: object
    window: object
    model: object
    tokens: bytes
    payload_bytes: int
    phase: CheckpointPhase = CheckpointPhase.LIVE
    fence: object = None
    rewind_started: bool = False
    claimant: str | None = None
    restore_started: bool = False
    restore_records: tuple | None = None
    kind: str = 'P'
    action_history: object = None
    claim_action_history: object = None
    claim_action_end: int = 0
    restore_deadline: float | None = None
    tail_rollback: object = None
    # Includes boundary-only snapshots when historical LRU disables graph edges.
    graph_history: object = None
    claim_graph_history: object = None
    claim_graph_end: int = 0
    guided_seed: object = None

    @property
    def position(self):
        return self.window.position


@dataclass
class _TailRollback:
    records: tuple
    raw: object
    values: object
    pins: tuple
    anchor: object
    reclaimed: bool = False


class PromptCheckpointPool:
    def __init__(self, capacity, max_payload_bytes, *, retain_prompt=False, native_restore_frame=False,
                 native_token_identity=False, native_checkpoint_capture=False):
        if any(type(v) is not int or v <= 0 for v in (capacity, max_payload_bytes)):
            raise ValueError('Checkpoint pool needs positive entry and payload limits')
        if type(native_restore_frame) is not bool:
            raise ValueError('native_restore_frame must be boolean')
        if type(native_token_identity) is not bool:
            raise ValueError('native_token_identity must be boolean')
        if type(native_checkpoint_capture) is not bool:
            raise ValueError('native_checkpoint_capture must be boolean')
        self.capacity = capacity
        self.max_payload_bytes = max_payload_bytes
        self.entries = {}
        self.prompts = {}
        self.retain_prompt = retain_prompt
        self.native_restore_frame = native_restore_frame
        self.native_token_identity = native_token_identity
        self.native_checkpoint_capture = native_checkpoint_capture
        self.payload_bytes = 0
        self._discarded_captures = []

    @staticmethod
    def model_ready(checkpoint):
        query = getattr(checkpoint.model, 'query', None)
        return query is None or query()

    def poll_captures(self):
        """Release completed snapshot sources without waiting on other sessions."""
        for checkpoint in (*self.entries.values(), *self.prompts.values()):
            self.model_ready(checkpoint)
        pending = []
        for checkpoint in self._discarded_captures:
            if self.model_ready(checkpoint):
                self._release_payload(checkpoint)
            else:
                pending.append(checkpoint)
        self._discarded_captures = pending
        return not pending

    def _owned(self, checkpoint):
        if (self.entries.get(checkpoint.namespace) is not checkpoint and
                self.prompts.get(checkpoint.namespace) is not checkpoint):
            raise ValueError('Stale or foreign prompt checkpoint lease')

    def _clear_claim_graph(self, checkpoint):
        snapshot = checkpoint.claim_graph_history
        if snapshot is not None:
            self.payload_bytes -= snapshot.bytes
            checkpoint.payload_bytes -= snapshot.bytes
            snapshot.close()
        checkpoint.claim_graph_history = None
        checkpoint.claim_graph_end = 0

    def capture(self, namespace, request_id, authority, tokens, model_adapter, *, replace=False, terminal=False):
        self.poll_captures()
        previous = self.entries.get(namespace)
        if previous is not None and not replace:
            raise ValueError('Session already owns a checkpoint; serialize its turns')
        if previous is not None and (previous.phase is not CheckpointPhase.LIVE or
                previous.request_id != request_id or previous.authority is not authority):
            raise ValueError('Replacement requires the same live session authority')
        if previous is None and len(self.entries) >= self.capacity:
            raise BufferError('Prompt checkpoint entry capacity is exhausted')
        position = authority.history_end
        # Validate token IDs before acquiring native prefix leases or GPU copies.
        if (type(position) is not int or position <= 0 or
                (position % authority.memory.config.block_tokens and not terminal)):
            raise ValueError('Prompt checkpoint requires a sealed authority boundary')
        token_size = position * 4
        preserve = self.retain_prompt and terminal and previous is not None and previous.kind=='P'
        # Pending D2H still owns both buffers even if this namespace replaces
        # its snapshot. Never reuse that credit until its own event completes.
        reusable = previous and not preserve and self.model_ready(previous)
        retained_bytes = self.payload_bytes - (previous.payload_bytes if reusable else 0)
        if token_size >= self.max_payload_bytes - retained_bytes:
            raise BufferError('Prompt checkpoint payload capacity is exhausted')
        identity = token_bytes(tokens, position, native=self.native_checkpoint_capture)
        window = authority.memory.checkpoint_window(allow_partial=terminal,
            **({'native_validation': True} if self.native_checkpoint_capture else {}))
        # Account for the serialized frame geometry as well as token/model data.
        # Python object overhead and authority/archive storage are separate quotas.
        fixed = len(identity) + len(window.records) * 5 * 8
        seed = getattr(authority, 'guided_seed', None)
        if seed is not None and seed.end > position:
            seed = None
        if seed is not None:
            # Immutable CPU Q can be shared with the live authority. Charge
            # every retained version conservatively to the checkpoint quota.
            fixed += seed.bytes
        action = getattr(authority,'action_history',None)
        action_snapshot = action.snapshot(position) if action is not None else None
        if action_snapshot is not None:
            fixed += len(action_snapshot[0])*16+8
            if len(action_snapshot)>2:
                fixed += len(action_snapshot[2][0])*24+24
        graph = selection_metadata(authority)
        graph_snapshot = graph.snapshot(position) if graph is not None else None
        try:
            if graph_snapshot is not None:
                fixed += graph_snapshot.bytes
            available = self.max_payload_bytes - retained_bytes - fixed
            if available <= 0:
                raise BufferError('Prompt checkpoint payload capacity is exhausted')
            model = model_adapter.capture(request_id, position, max_bytes=available,
                                          **({'terminal': True} if terminal else {}))
        except BaseException:
            if graph_snapshot is not None:
                graph_snapshot.close()
            raise
        checkpoint = PromptCheckpoint(namespace, request_id, authority, window,
                                      model, identity, fixed + model.bytes)
        checkpoint.kind = 'M' if terminal else 'P'
        checkpoint.action_history = action_snapshot
        checkpoint.graph_history = graph_snapshot
        checkpoint.guided_seed = seed
        # Keep the previous snapshot available if any validation/copy/admission
        # above fails. In-flight replaced snapshots retain their buffers and
        # payload credit until their own D2H fence completes.
        if previous is not None:
            if preserve:
                if namespace in self.prompts:
                    self._remove(self.prompts[namespace])
                self.prompts[namespace] = previous
            else:
                self._remove(previous)
                if not terminal and namespace in self.prompts:
                    self._remove(self.prompts[namespace])
        self.entries[namespace] = checkpoint
        self.payload_bytes += checkpoint.payload_bytes
        return checkpoint

    def drop_live(self, checkpoint):
        """Discard a snapshot while capture runtime keeps its live authority."""
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.LIVE:
            raise ValueError('Only a live borrowed snapshot may be dropped')
        prompt = self.prompts.get(checkpoint.namespace)
        if prompt is not None and prompt is not checkpoint:
            self._remove(prompt)
        self._remove(checkpoint)

    def update_live_window(self, checkpoint):
        """Keep the pre-Query model state with the committed replay KV frame.

        A short turn may end before the next sealed decode checkpoint. Its
        replay must use the selected frame used by the original durable Query,
        without rerunning selection from the older pressure frame.
        """
        self._owned(checkpoint)
        if (checkpoint.phase is not CheckpointPhase.LIVE or
                checkpoint.authority.history_end != checkpoint.position):
            raise ValueError('Window update requires the exact live checkpoint boundary')
        window = checkpoint.authority.memory.checkpoint_window(
            **({'native_validation': True} if self.native_checkpoint_capture else {}))
        delta = (len(window.records) - len(checkpoint.window.records)) * 5 * 8
        if self.payload_bytes + delta > self.max_payload_bytes:
            raise BufferError('Updated checkpoint window exceeds its payload budget')
        checkpoint.window = window
        checkpoint.payload_bytes += delta
        self.payload_bytes += delta

    def retire(self, checkpoint, completion):
        """Caller has detached capture and all further suffix execution."""
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.LIVE:
            raise ValueError('Prompt authority already detached or claimed')
        checkpoint.fence = completion
        checkpoint.phase = CheckpointPhase.RETIRING

    def _begin_rewind(self, checkpoint):
        authority = checkpoint.authority
        memory = authority.memory
        prefix = checkpoint.window.prefix
        if checkpoint.position % memory.config.block_tokens:
            # A cancelled successor may have sealed the previous M tail. Never
            # overwrite that immutable generation. Save its unchanged prefix,
            # retire the whole tail through the ordinary fenced transaction,
            # then append those CPU bytes into a fresh repository generation.
            memory._validate_window_checkpoint(checkpoint.window)
            if not authority.flush() or authority.writes:
                return False
            from . import _native
            records = checkpoint.window.records
            slot, generation, begin, count, _ = records[-1]
            if begin + count != checkpoint.position or count == memory.config.block_tokens:
                raise ValueError('Partial checkpoint lost its exact terminal frame')
            pins = tuple(memory.repository.pin(_native.Handle(s, g)) for s, g, *_ in records[:-1])
            if not all(pins):
                raise ValueError('Partial rollback lost a saved frame generation')
            raw, values = authority.read_record(_native.Handle(slot, generation))
            # Keep at most one record, independent of total history length.
            raw, values = raw[:, :count].clone(), values[:, :count].clone()
            prefix = memory._store.checkpoint(begin, False)
            pending = _TailRollback(records, raw, values, pins, prefix)
            try:
                policy_hold = memory._protect_checkpoint(records[:-1])
            except BufferError:
                return False
        else:
            pending = None
        spans = authority.begin_rewind(prefix)
        if spans is None:
            return False
        checkpoint.tail_rollback = pending
        if pending is not None:
            # A prior cancellation may have rebased this frame already. Its
            # explicit tail pin must retire too; pending.pins protects only
            # the unchanged complete records throughout this transaction.
            checkpoint.window = replace(checkpoint.window, leases=(), policy_hold=policy_hold)
        return True

    def _poll_rewind(self, checkpoint):
        authority = checkpoint.authority
        pending = checkpoint.tail_rollback
        if pending is None:
            return authority.poll_rewind()
        if not pending.reclaimed:
            if not authority.poll_rewind():
                return False
            pending.reclaimed = True
        _, _, begin, count, dense = pending.records[-1]
        if not authority.append(pending.raw, pending.values, history_begin=begin, local_rope_begin=dense):
            return False
        tail = next(r for r in authority.records.values() if r.history_begin == begin)
        records = pending.records[:-1] + ((tail.handle.slot, tail.handle.generation, begin, count, dense),)
        checkpoint.window = authority.memory.checkpoint_frame(records)
        checkpoint.tail_rollback = None
        return True

    def idle(self, checkpoint):
        self._owned(checkpoint)
        prompt = self.prompts.get(checkpoint.namespace)
        if not self.model_ready(checkpoint) or (prompt is not None and not self.model_ready(prompt)):
            return False
        if checkpoint.phase is CheckpointPhase.IDLE:
            return True
        if checkpoint.phase is not CheckpointPhase.RETIRING:
            return False
        if checkpoint.fence is not None and not checkpoint.fence.query():
            return False
        authority = checkpoint.authority
        if authority.history_end == checkpoint.position and not checkpoint.rewind_started:
            # An exact terminal prefix needs no rewind, but packed restore
            # still requires every sealed page to be published. Keep only this
            # namespace retiring while shared archive admission is busy.
            if not authority.flush() or authority.writes:
                return False
            # At a terminal checkpoint every accepted raw K/V row is retained,
            # including the mutable partial tail. No branching or rewrite of an
            # immutable archive generation is necessary before transfer.
            checkpoint.phase = CheckpointPhase.IDLE
            checkpoint.fence = None
            prompt = self.prompts.get(checkpoint.namespace)
            if prompt is not None:
                prompt.phase = CheckpointPhase.IDLE
            return True
        if not checkpoint.rewind_started:
            if not self._begin_rewind(checkpoint):
                return False
            authority.attach_rewind_completion(checkpoint.fence)
            checkpoint.rewind_started = True
        if not self._poll_rewind(checkpoint):
            return False
        checkpoint.phase = CheckpointPhase.IDLE
        checkpoint.fence = None
        return True

    def claim(self, namespace, request_id, tokens, *, position=None, timing=None):
        checkpoint = self.entries.get(namespace)
        if checkpoint is None or checkpoint.phase is not CheckpointPhase.IDLE:
            return None
        if not self.model_ready(checkpoint):
            return None
        if not request_id or request_id == checkpoint.request_id:
            raise ValueError('Checkpoint import requires a fresh native request identity')
        # Evidence copied from an M->P fallback belongs to that exact claim.
        # A cancelled retained P may next be reused with a different suffix.
        checkpoint.claim_action_history = None
        checkpoint.claim_action_end = 0
        self._clear_claim_graph(checkpoint)
        if position is not None and checkpoint.position != position:
            prompt = self.prompts.get(namespace)
            if prompt is None or prompt.position!=position or prompt.phase is not CheckpointPhase.IDLE:
                return None
            if not self.model_ready(prompt):
                return None
            with timing.stage('token_prefix_compare') if timing is not None else nullcontext():
                if len(tokens)<=position or token_bytes(tokens,position,native=self.native_token_identity)!=prompt.tokens:
                    return None
            # A page-aligned new Query can require P model state even though
            # the complete previous M token prefix is unchanged. Preserve its
            # output evidence only after verifying that entire prefix, never
            # merely because a session name or shorter P prefix matches.
            prompt.claim_action_history = None
            prompt.claim_action_end = 0
            self._clear_claim_graph(prompt)
            with timing.stage('token_prefix_compare') if timing is not None else nullcontext():
                full_prefix = (len(tokens)>checkpoint.position and
                    token_bytes(tokens,checkpoint.position,native=self.native_token_identity)==checkpoint.tokens)
            if full_prefix and checkpoint.graph_history is not None:
                if self.payload_bytes + checkpoint.graph_history.bytes > self.max_payload_bytes:
                    raise BufferError('M graph preservation exceeds checkpoint payload capacity')
                prompt.claim_graph_history = checkpoint.graph_history.clone()
                prompt.claim_graph_end = checkpoint.position
                prompt.payload_bytes += prompt.claim_graph_history.bytes
                self.payload_bytes += prompt.claim_graph_history.bytes
            if full_prefix and checkpoint.action_history is not None:
                from copy import deepcopy
                prompt.claim_action_history=deepcopy(checkpoint.action_history)
                prompt.claim_action_end=checkpoint.position
            self._remove(checkpoint)  # Release later-prefix pins before a P rewind.
            del self.prompts[namespace]
            self.entries[namespace] = checkpoint = prompt
        # No hash-only equality and no reuse of reserialized generated suffixes.
        with timing.stage('token_prefix_compare') if timing is not None else nullcontext():
            if len(tokens) <= checkpoint.position or token_bytes(
                    tokens, checkpoint.position, native=self.native_token_identity) != checkpoint.tokens:
                return None
        if not request_id or request_id == checkpoint.request_id:
            raise ValueError('Checkpoint import requires a fresh native request identity')
        checkpoint.phase = CheckpointPhase.CLAIMED
        checkpoint.claimant = request_id
        checkpoint.rewind_started = False
        checkpoint.restore_deadline = time.monotonic() + 180
        return checkpoint

    def finish_claim(self, checkpoint, completion, *, retain=False):
        """After exact target/GDN/draft restoration, return authority ownership.

        The caller owns fresh native pages through this fence and must already
        have committed RequestMemory.restore_window. A failed import instead calls
        evict with its cleanup fence and must never resume native execution.
        """
        from .memory import Phase
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.CLAIMED:
            raise ValueError('Checkpoint is not claimed by a native import')
        if checkpoint.fence is not None and checkpoint.fence is not completion:
            raise ValueError('Checkpoint import completion fence changed')
        checkpoint.fence = completion
        if not completion.query():
            return None
        authority = checkpoint.authority
        if (not checkpoint.restore_started or authority.memory.phase is not Phase.READY or
                authority.history_end != checkpoint.position or
                tuple(authority.memory._store.working_records()) != checkpoint.restore_records):
            raise ValueError('Checkpoint import did not commit its exact target prefix')
        if retain:
            checkpoint.phase = CheckpointPhase.LIVE
            checkpoint.request_id = checkpoint.claimant
            checkpoint.fence = None
            checkpoint.restore_started = checkpoint.rewind_started = False
            prompt = self.prompts.get(checkpoint.namespace)
            if prompt is not None:
                prompt.phase, prompt.request_id = CheckpointPhase.LIVE, checkpoint.claimant
            return authority
        self._remove(checkpoint)
        prompt = self.prompts.pop(checkpoint.namespace,None)
        if prompt is not None:
            prompt.phase = CheckpointPhase.LIVE
            prompt.request_id = checkpoint.claimant
            self.entries[prompt.namespace] = prompt
        return authority

    def prepare_claim_window(self, checkpoint, *, pressure=False, pressure_budget=None,
                             mandatory_spans=(), wait=True, timing=None, metadata_binding=None):
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.CLAIMED or checkpoint.restore_started:
            raise ValueError('No unsubmitted checkpoint import')
        if not self.model_ready(checkpoint):
            raise RuntimeError('Claimed model checkpoint payload is not ready')
        if metadata_binding is not None and selection_metadata(checkpoint.authority) is not None:
            from .prompt_spans import graph_metadata
            from .block_graph import _merge_spans
            spans, _ = graph_metadata(metadata_binding)
            saved = checkpoint.claim_graph_history or checkpoint.graph_history
            if saved is None:
                raise ValueError('Graph session checkpoint has no matching graph snapshot')
            # Validate against the state we will actually restore, before
            # rewind/graph restore mutate authority or its snapshot ownership.
            _merge_spans(saved.message_spans, saved.message_end_known, spans,
                max(checkpoint.position, checkpoint.claim_graph_end))
        memory = checkpoint.authority.memory
        if memory.history_end > checkpoint.position or checkpoint.rewind_started:
            deadline = checkpoint.restore_deadline
            if deadline is None:
                raise ValueError('Checkpoint claim has no restore deadline')
            authority = checkpoint.authority
            while True:
                remaining = deadline-time.monotonic()
                if remaining<=0:
                    raise TimeoutError('P checkpoint rewind did not drain')
                if not checkpoint.rewind_started:
                    if self._begin_rewind(checkpoint):
                        class Ready:
                            def query(self): return True
                        authority.attach_rewind_completion(Ready())
                        checkpoint.rewind_started = True
                if checkpoint.rewind_started and self._poll_rewind(checkpoint):
                    checkpoint.rewind_started = False
                    break
                if not wait:
                    return None
                if not authority.archive.wait_for_idle(remaining):
                    raise TimeoutError('P checkpoint archive did not drain')
        graph = selection_metadata(checkpoint.authority)
        if graph is not None:
            if checkpoint.graph_history is None:
                raise ValueError('Graph session checkpoint has no matching graph snapshot')
            with timing.stage('graph_restore') if timing is not None else nullcontext():
                if checkpoint.claim_graph_history is not None:
                    graph.restore(checkpoint.claim_graph_history, preserve_future=True,
                        preserved_prefix=checkpoint.graph_history, until=checkpoint.claim_graph_end)
                else:
                    graph.restore(checkpoint.graph_history)
            checkpoint.authority._graph_validated_prefix = max(checkpoint.position, checkpoint.claim_graph_end)
            if getattr(graph, 'boundary_only', False) or getattr(checkpoint.authority, 'query_boundaries', None) is graph:
                checkpoint.authority._query_boundary_validated_prefix = max(checkpoint.position, checkpoint.claim_graph_end)
        action = getattr(checkpoint.authority,'action_history',None)
        checkpoint.authority.guided_seed = checkpoint.guided_seed
        if action is not None:
            with timing.stage('action_restore') if timing is not None else nullcontext():
                original=checkpoint.action_history or ({},0)
                if checkpoint.claim_action_history is not None:
                    action.restore(checkpoint.claim_action_history,checkpoint.authority,
                        preserve_future=True,validated_end=checkpoint.claim_action_end)
                    action.replay_snapshot=(checkpoint.claim_action_end,original)
                else:
                    action.restore(original,checkpoint.authority)
        if graph is not None and metadata_binding is not None:
            from .prompt_spans import install_selection_metadata
            install_selection_metadata(checkpoint.authority, metadata_binding,
                validated_prefix=checkpoint.authority._graph_validated_prefix)
        if pressure:
            if timing is None:
                memory._validate_window_checkpoint(checkpoint.window)
            else:
                memory._validate_window_checkpoint(checkpoint.window, timing=timing)
            if memory.history_end != checkpoint.position:
                raise ValueError('Pressure restore requires the exact saved history prefix')
            # Native allocation is fresh: neither same-position K nor resident V
            # from the retired request is still resident in these physical pages.
            for slot, generation, *_ in memory._store.working_records():
                memory._store.working_resident(memory._store.handle(slot), False)
            from .request_plan import RequestPlan, selection_config
            policy_options = getattr(checkpoint.authority, 'graph_options', None) or getattr(action, 'options', None)
            policy = getattr(policy_options, 'pressure_policy', None)
            if policy is None:
                policy = 'attention' if action is not None and getattr(action.options, 'action_pressure', False) else 'recency'
            cfg = selection_config(memory.config,
                memory.config.select_budget if pressure_budget is None else pressure_budget,
                selection='pressure', pressure_policy=policy)
            fit = RequestPlan(None, block_tokens=cfg.block_tokens,
                mandatory=mandatory_spans).mandatory_fit(checkpoint.position, cfg, preserve_tail=True)
            required = set(fit['hard_blocks'])
            mandatory = [r.handle for r in checkpoint.authority.records.values()
                         if r.history_begin // cfg.block_tokens in required]
            if len(mandatory) != len(required):
                raise ValueError('Pressure restore lost a mandatory historical page')
            from .prefill_attention import select_pressure
            plan = select_pressure(checkpoint.authority, mandatory, max_tokens=pressure_budget,
                candidate_records=checkpoint.window.records)
        else:
            with timing.stage('window_restore') if timing is not None else nullcontext():
                kwargs = {'native_restore_frame': True} if self.native_restore_frame else {}
                if timing is not None:
                    kwargs['timing'] = timing
                plan = memory.restore_window(checkpoint.window, **kwargs)
        with timing.stage('restore_records') if timing is not None else nullcontext():
            checkpoint.restore_records = tuple(memory._store.working_records())
        checkpoint.restore_started = True
        return plan

    def evict(self, checkpoint, completion=None):
        self._owned(checkpoint)
        if checkpoint.phase is CheckpointPhase.LIVE:
            raise ValueError('Live capture still owns authority; use drop_live')
        if checkpoint.phase is CheckpointPhase.CLAIMED:
            if completion is None:
                raise ValueError('Allocated import eviction requires its cleanup fence')
            checkpoint.fence = completion
        elif completion is not None:
            raise ValueError('Unexpected replacement of authority cleanup fence')
        checkpoint.phase = CheckpointPhase.CLOSING
        return self.poll_eviction(checkpoint)

    def poll_eviction(self, checkpoint):
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.CLOSING:
            raise ValueError('Checkpoint has not been evicted')
        if checkpoint.fence is not None and not checkpoint.fence.query():
            return False
        prompt = self.prompts.get(checkpoint.namespace)
        if not self.model_ready(checkpoint) or (prompt is not None and not self.model_ready(prompt)):
            return False
        # Release our prefix pins before closing. External readers still keep
        if prompt is not None and prompt is not checkpoint:
            self._remove(prompt)
        # the entry and its credits unavailable until authority.close succeeds.
        checkpoint.window = checkpoint.model = None
        checkpoint.tail_rollback = None
        checkpoint.tokens = b''
        if not checkpoint.authority.close():
            return False
        self._remove(checkpoint)
        return True

    def _remove(self, checkpoint):
        if self.entries.get(checkpoint.namespace) is checkpoint:
            del self.entries[checkpoint.namespace]
        elif self.prompts.get(checkpoint.namespace) is checkpoint:
            del self.prompts[checkpoint.namespace]
        else:
            raise ValueError('Checkpoint is no longer owned')
        if not self.model_ready(checkpoint):
            self._discarded_captures.append(checkpoint)
            return
        self._release_payload(checkpoint)

    def _release_payload(self, checkpoint):
        checkpoint.guided_seed = None
        self.payload_bytes -= checkpoint.payload_bytes
        for name in ('graph_history', 'claim_graph_history'):
            snapshot = getattr(checkpoint, name, None)
            if snapshot is not None:
                snapshot.close()
                setattr(checkpoint, name, None)
        # Do not let an obsolete Python lease retain native prefix/storage pins.
        checkpoint.window = checkpoint.model = checkpoint.authority = None
        checkpoint.tail_rollback = None
        checkpoint.tokens = b''
