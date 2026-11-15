"""Scheduler-owned exact-prefix session leases, carried by the KV connector.

An idle hit exists only after a worker retirement acknowledgment. All other
lookups either wait for that particular session or use ordinary native prefill.
Generation-qualified messages cannot release a newer incarnation of a session.
"""
from dataclasses import dataclass, field
import hashlib
import os

from .prompt_checkpoints import token_bytes
from ..core import settings as _settings


def session_namespace(session, salt):
    if not session:
        return None
    salt = salt or ''
    if (not isinstance(session, str) or not isinstance(salt, str)
            or len(session) > 1024 or len(salt) > 1024):
        raise ValueError('Session ID and cache salt must be strings of at most 1024 characters')
    return salt, session


def idle_lru_eviction_enabled():
    """KVMEM_SESSION_IDLE_LRU_EVICTION=1: a full preserve_history ledger evicts
    its least recently used idle session instead of blocking new sessions."""
    value = _settings.get('KVMEM_SESSION_IDLE_LRU_EVICTION', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_SESSION_IDLE_LRU_EVICTION must be 0 or 1')
    return value == '1'


def session_release_intent(params, session, prompt, max_tokens):
    """Explicit close via a tiny ordinary request after fenced eviction.

    This request must never import or retain a model checkpoint. Absence of
    the flag preserves normal one-token session requests and their history.
    """
    if params is None:
        return False
    if not isinstance(params, dict):
        raise ValueError('Session transfer parameters must be an object')
    release = params.get('kvmem_release_session', False)
    if type(release) is not bool:
        raise ValueError('kvmem_release_session must be boolean')
    if not release:
        return False
    if (not isinstance(session, str) or not session or
            not isinstance(prompt, (list, tuple)) or len(prompt) != 1 or
            type(prompt[0]) is not int or prompt[0] < 0 or
            type(max_tokens) is not int or max_tokens != 1):
        raise ValueError('Session release requires a namespace, one input token and max_tokens=1')
    plan = params.get('kvmem_request_plan') or {}
    if not isinstance(plan, dict) or plan.get('kvmem_query_span') is not None:
        raise ValueError('Session release cannot execute a Query')
    return True


@dataclass(frozen=True)
class SessionBinding:
    namespace: tuple[str, str]
    generation: int
    request_id: str
    tokens: tuple[int, ...]
    history: int = 0
    window: int = 0
    pressure_restore: bool = False
    query_span: tuple | None = None
    mandatory_spans: tuple = ()
    message_spans: tuple = ()
    locality_spans: tuple = ()
    locality_provenance: tuple = ()
    # Historical features retain the explicit Question even when exact-prefix
    # continuation removes its already-completed selection/replay operation.
    question_span: tuple | None = None
    question_id: str | None = None
    # kv_transfer_params.kvmem_share_prefix: register the retained frozen P as a
    # prefix image once it is idle (serving/prefix_images.py). 0: never.
    share_prefix: int = 0


@dataclass(frozen=True)
class SessionCapture:
    request_id: str
    generation: int
    position: int
    tokens: tuple[int, ...] = ()
    # Opt-in KVMEM_HELD_P_CAPTURE: the scheduler holds this request's final
    # prefill piece; the worker captures after another batch's launch.
    held: bool = False


def held_p_capture():
    """KVMEM_HELD_P_CAPTURE=1: a turn's P checkpoint no longer stalls the batch.

    The P capture before a turn's final prefill piece (publication drain,
    model-state copy, frame/action/layerwise snapshots) ran on the engine
    thread before that piece's forward, while every other request's decode
    rows in the same batch waited with the GPU idle. Enabled, when the piece
    has batch peers, the scheduler holds it for one step: the worker takes the
    same capture of the same, untouched state after the peers' forward is
    enqueued, reports it, and the piece runs in a following step.

    Value 2 (exactness testing only) also holds a piece without peers, so the
    capture runs in a zero-token step and N=1 exercises the same path.
    """
    value = _settings.get('KVMEM_HELD_P_CAPTURE', '0')
    if value not in ('0', '1', '2'):
        raise ValueError('KVMEM_HELD_P_CAPTURE must be 0, 1 or 2')
    return int(value)


# Worker notices of prefix images (serving/prefix_images.py): registration of an
# image from an idle frozen P, and the materialization of a new session from one.
_PREFIX_IMAGE_NOTICES = frozenset(('image_registered', 'image_refused', 'image_released', 'imported',
                                   'import_failed', 'history_pool'))


@dataclass(frozen=True)
class SessionEviction:
    namespace: tuple[str, str]
    generation: int
    allow_missing: bool = False


@dataclass(frozen=True)
class SessionNotice:
    kind: str  # retired, restored, evicted
    namespace: tuple[str, str]
    generation: int
    request_id: str
    history: int = 0
    window: int = 0
    ready_records: int = 0
    prompt_history: int = 0
    prompt_window: int = 0
    prompt_tokens: bytes = b''
    active_records: tuple = ()
    retained: bool = False
    resident_ready: bool = False
    allocated_records: int = 0
    protected_records: int = 0
    discard_shards: bool = False


@dataclass(frozen=True)
class HistoryCapacityRequest:
    namespace: tuple[str, str]
    generation: int
    ready_records: int = 0
    retained_records: int | None = None


@dataclass
class SessionCommands:
    bindings: list[SessionBinding] = field(default_factory=list)
    captures: list[SessionCapture] = field(default_factory=list)
    evictions: list[SessionEviction] = field(default_factory=list)
    terminals: list[SessionCapture] = field(default_factory=list)
    # Explicit close requests have already evicted their old owned session and
    # intentionally create no replacement binding/checkpoint. Carry a narrow
    # allowlist so the worker does not confuse them with unowned graph traffic.
    release_requests: set[str] = field(default_factory=set)
    history_capacity: list[HistoryCapacityRequest] = field(default_factory=list)
    # Opt-in session tiering: sessions whose next turn just arrived, so the
    # worker can start bringing them back from SSD while they queue.
    returning: list = field(default_factory=list)
    # Prefix images (serving/prefix_images.py): new sessions to materialize from
    # an image (SessionImport) and image keys the worker may drop.
    image_imports: list = field(default_factory=list)
    image_releases: list = field(default_factory=list)


@dataclass
class _Entry:
    namespace: tuple[str, str]
    generation: int
    request_id: str | None
    phase: str = 'lookup'
    allocated: bool = False
    history: int = 0
    window: int = 0
    tokens: bytes = b''
    candidate_position: int = 0
    candidate_tokens: bytes = b''
    terminal_position: int = 0
    terminal_tokens: bytes = b''
    ready_records: int = 0
    allocated_records: int = 0
    protected_records: int = 0
    scope_history: int = 0
    retained_position: int = 0
    retained_tokens: bytes = b''
    pressure_restore: bool = False
    restore_window: int = 0
    prompt_history: int = 0
    prompt_window: int = 0
    prompt_tokens: bytes = b''
    query: tuple | None = None
    mandatory: tuple = ()
    active_records: tuple = ()
    query_id: str | None = None
    question_span: tuple | None = None
    question_id: str | None = None
    guided_anchor: int = 0
    prompt_guided_anchor: int = 0
    # Prefix images: the validated kvmem_share_prefix of the active request, and
    # for a session being materialized from an image (phase 'importing') the
    # requesting native request, the image key and whether that request ended.
    share_prefix: int = 0
    import_request: str | None = None
    import_key: object = None
    import_orphan: bool = False
    _fingerprint_tokens: bytes | None = field(default=None, repr=False, compare=False)
    _fingerprint: str = field(default='', repr=False, compare=False)

    def prompt_tokens_for(self, position):
        """Token identity of this idle session's first ``position`` tokens (a P
        below a retained M shares the M identity's prefix)."""
        return self.tokens[:position * 4] if 0 < position <= self.history else b''

    def prefix_fingerprint(self):
        # Tokens are immutable bytes. Retaining the object avoids both repeated
        # hashing and id reuse; a replaced prefix is never mistaken for a hit.
        if self._fingerprint_tokens is not self.tokens:
            self._fingerprint = hashlib.sha256(self.tokens).hexdigest()
            self._fingerprint_tokens = self.tokens
        return self._fingerprint


class SessionLedger:
    def __init__(self, capacity, block_tokens, prepare_import, abort_import, commit_import, *, terminal_capture=False,
                 select_budget=None, gen_budget=None, checkpoint_policy='sealed',memory_config=None,
                 preserve_history=False, require_owned=False, pressure_policy='recency', freeze_prompt=False,
                 guided_options=None, prefix_image_scope='prefix images were not configured by the scheduler',
                 prefix_image_model=''):
        if any(type(v) is not int or v <= 0 for v in (capacity, block_tokens)):
            raise ValueError('Session capacity and checkpoint block size must be positive')
        self.capacity, self.block_tokens = capacity, block_tokens
        self.prepare_import, self.abort_import, self.commit_import = prepare_import, abort_import, commit_import
        self.entries = {}  # Insertion order is the idle LRU order.
        self.requests = {}
        # Checkpoint capacity does not limit independent cold requests. Keep a
        # lightweight lease for those overflow sessions nevertheless: otherwise
        # concurrent turns can bypass per-session serialization when slots fill.
        self.uncached_sessions = {}
        self.uncached_requests = {}
        self.generation = 0
        self.commands = SessionCommands()
        self.terminal_capture = terminal_capture
        self.checkpoint_policy = checkpoint_policy
        if type(freeze_prompt) is not bool or (freeze_prompt and
                (checkpoint_policy != 'sealed' or terminal_capture)):
            raise ValueError('Frozen prompt ledger requires sealed P-only checkpoints')
        self.freeze_prompt = freeze_prompt
        self.guided_options = guided_options
        self.memory_config = memory_config
        self.pressure_policy = pressure_policy
        self.select_budget, self.gen_budget = select_budget, gen_budget
        self.hits = self.misses = self.waits = 0
        self.preserve_history = preserve_history
        self.idle_lru_eviction = idle_lru_eviction_enabled()
        if type(require_owned) is not bool:
            raise ValueError('Owned-session requirement must be boolean')
        self.require_owned = require_owned
        # Prefix images (serving/prefix_images.py): None when this serving
        # configuration is inside their scope (scope_error), else why it is not.
        self.prefix_image_scope = prefix_image_scope
        self.prefix_image_model = '' if prefix_image_model is None else str(prefix_image_model)
        self.prefix_images = None       # scheduler image index, created at the first registration
        self.importing = {}             # native request id -> namespace being imported for it
        self.import_declined = set()    # requests whose import failed: they prefill cold

    @staticmethod
    def namespace(request):
        session = getattr(request, 'session_id', None)
        if not session:
            return None
        if (not isinstance(session, str) or getattr(request, 'resumable', False) or
                getattr(request, 'prompt_embeds', None) is not None or
                getattr(request, 'mm_features', None) or getattr(request, 'lora_request', None)):
            raise ValueError('Prompt sessions require non-streaming, text-only native requests')
        return session_namespace(session, getattr(request, 'cache_salt', None))

    def lookup(self, request, local_tokens):
        namespace = self.namespace(request)
        release = session_release_intent(getattr(request, 'kv_transfer_params', None),
            getattr(request, 'session_id', None), request.prompt_token_ids, self.output_tokens(request))
        share = self._share_prefix_intent(request, namespace, release)
        if namespace is None:
            if self.require_owned:
                raise ValueError('Block graph requests require an owned session namespace')
            self.cold_plan(request)
            return 0
        if local_tokens:
            raise ValueError('Prompt session import requires native prefix caching disabled')
        owner = self.uncached_sessions.get(namespace)
        if owner is not None:
            if owner == request.request_id:
                return 0
            self.waits += 1
            return None
        entry = self.entries.get(namespace)
        if entry is not None and entry.request_id == request.request_id:
            if entry.phase != 'lookup':
                raise ValueError('Native request repeated session lookup after allocation')
            return entry.history
        if entry is not None and entry.phase != 'idle':
            self.waits += 1
            return None
        if release:
            if self.prefix_images is not None:
                # Images registered by this session go with an explicit close.
                self.commands.image_releases.extend(self.prefix_images.release_owner(namespace))
            if entry is not None:
                # Wait for the existing generation's history, model snapshot
                # and resident GPU leases to retire through the normal path.
                self._evict(entry)
                self.waits += 1
                return None
            # Keep namespace serialization through the tiny close request,
            # but create no binding/capture and no replacement cache entry.
            # This also works when all checkpoint slots are already occupied.
            self.cold_plan(request)
            self.uncached_sessions[namespace] = request.request_id
            self.uncached_requests[request.request_id] = namespace
            return 0
        if entry is not None:
            # Finalize against a private ledger copy. A failed combined budget
            # check must retain the original M/P lineage, pins and idle owner.
            from dataclasses import replace
            saved_entry = entry
            entry = replace(entry)
            prompt = request.prompt_token_ids
            from .request_plan import request_plan
            plan = request_plan(request,self.block_tokens,self.gen_budget) if self.gen_budget else None
            matches = len(prompt)>entry.history and token_bytes(prompt,entry.history)==entry.tokens
            same_identity = plan is None or plan.query_id is None or plan.query_id == entry.query_id
            output = self.output_tokens(request)
            effective, changed = (plan.continuation(len(prompt),output,entry.history,entry.window,
                self.select_budget+self.gen_budget,entry.query if matches and same_identity else None,
                entry.mandatory if matches else ()) if plan is not None else (None,False))
            needs_p = not matches or (effective is not None and effective.query is not None and entry.history>effective.replay_begin)
            if needs_p:
                ceiling = effective.replay_begin if effective is not None and effective.query is not None else len(prompt)-1
                if (entry.prompt_history and entry.prompt_history<=ceiling and len(prompt)>entry.prompt_history
                        and token_bytes(prompt,entry.prompt_history)==entry.prompt_tokens):
                    entry.history,entry.window,entry.tokens = entry.prompt_history,entry.prompt_window,entry.prompt_tokens
                    if not matches:
                        # Only the verified P prefix survives a changed branch.
                        # Old Query/pin coordinates may point into replaced or
                        # truncated text and must not enter the new request.
                        # A pin can straddle P. Its verified historical part
                        # still belongs to P; only the edited future is dropped.
                        entry.mandatory = tuple((b, min(e, entry.history), kind)
                            for b, e, kind in entry.mandatory if b < entry.history)
                        if entry.query is not None and entry.query[1] > entry.history:
                            entry.query = None
                    effective,changed = (plan.continuation(len(prompt),output,entry.history,entry.window,
                        self.select_budget+self.gen_budget,entry.query if same_identity else None,entry.mandatory) if plan is not None else (None,False))
                else:
                    if effective is not None:
                        self.validate_plan_capacity(request, effective)
                        self.set_plan(request,effective)
                    self._evict(saved_entry)
                    self.waits += 1
                    return None
            if changed:
                if self.memory_config is not None and not needs_p:
                    required=set(effective.mandatory_fit(entry.history,self.memory_config)['hard_blocks'])
                    available={b//self.block_tokens for b,n in entry.active_records}
                    changed=not required.issubset(available)
            if changed:
                # A new historical hard pin must be applied before using a
                # cached view which may omit it. Rebuild from durable tokens.
                self.validate_plan_capacity(request, effective)
                self.set_plan(request,effective)
                self._evict(saved_entry)
                self.waits += 1
                return None
            if self.guided_options is not None:
                from .guided_refresh import prepare_request
                effective = prepare_request(request, effective, entry, self.guided_options, imported=True)
            self.validate_plan_capacity(request, effective, history=entry.history, window=entry.window)
            for name in ('history', 'window', 'tokens', 'query', 'mandatory'):
                setattr(saved_entry, name, getattr(entry, name))
            entry = saved_entry
            if effective is not None:
                self.set_plan(request,effective)
                entry.query = plan.query or entry.query
                entry.query_id = plan.query_id
                entry.mandatory = effective.mandatory
            # Only this request's explicit, validated source is eligible. An
            # omitted Query or explicit off must not inherit an old Question.
            entry.question_span = plan.query if plan is not None else None
            entry.question_id = plan.query_id if entry.question_span is not None else None
            entry.pressure_restore = False  # Storage alone never triggers selection.
            entry.restore_window = 0
            capacity = self.select_budget + self.gen_budget if effective is not None else None
            terminal_reserve_overflow = (effective is not None and effective.query is None and
                entry.history > effective.generation_checkpoint(len(prompt)) and
                entry.window + len(prompt) - entry.history + effective.output_reserve >
                capacity - self.block_tokens)
            prefill_probe_overflow = (effective is not None and
                effective.prefill_window == 'pressure' and
                (effective.query is None or entry.history < effective.replay_begin) and
                entry.window > effective.prefill_boundary(entry.history, capacity))
            query_replay_overflow = (effective is not None and effective.query is not None and
                entry.history == effective.replay_begin and
                entry.window + effective.replay_end - entry.history > capacity - self.block_tokens)
            if terminal_reserve_overflow or prefill_probe_overflow or query_replay_overflow:
                # A partial terminal M prefix can already lie past the last
                # safe prefill pressure point. Native allocation reserves draft
                # lookahead BEFORE connector metadata can compact the window;
                # sealing this tail and probing a whole block can exceed K+G.
                # Select during the fenced import, preserving the accepted tail
                # and model state. A long tool suffix may use later ordinary
                # pressure points; it does not need to fit in one reservation.
                # A new atomic Query also needs its entire replay plus native
                # lookahead in the OLD view, before semantic selection runs.
                # A warm M import can have skipped that earlier pressure point.
                # Compact the import first; preserve the Query and its later
                # semantic selection instead of splitting or clipping it.
                budget = self.select_budget
                if terminal_reserve_overflow:
                    budget = min(budget,
                        (capacity - effective.output_reserve -
                         (len(prompt) - entry.history) - self.block_tokens)
                        // self.block_tokens * self.block_tokens)
                if budget <= 0:
                    raise ValueError('Continuation output reserve exceeds active capacity')
                entry.pressure_restore = True
                entry.restore_window = min(entry.history, budget -
                    (-entry.history % self.block_tokens))
            entry.share_prefix = share
            entry.phase, entry.request_id = 'lookup', request.request_id
            self.requests[request.request_id] = entry
            try:
                self.prepare_import(request.request_id, entry.history, entry.restore_window or entry.window)
            except Exception:
                entry.phase, entry.request_id = 'idle', None
                entry.pressure_restore, entry.restore_window = False, 0
                entry.share_prefix = 0
                del self.requests[request.request_id]
                raise
            return entry.history
        if self.prefix_images is not None and self._start_prefix_import(request, namespace):
            self.waits += 1
            return None
        self.cold_plan(request)
        if len(self.entries) >= self.capacity:
            if self.preserve_history and not self.idle_lru_eviction:
                # The caller must explicitly finish/release a session. Neither
                # LRU eviction nor uncached execution preserves its history.
                self.waits += 1
                return None
            # Entries are reinserted when they turn idle: the first idle entry
            # is the least recently used one. Opt-in KVMEM_SESSION_IDLE_LRU_EVICTION
            # applies this to preserved histories too (clients such as dsh never
            # release a session); the evicted session's next turn is a cold prefill.
            idle = next((e for e in self.entries.values() if e.phase == 'idle'), None)
            if idle is not None:
                # Under the opt-in, one eviction at a time: the native waiting
                # loop retries this lookup every step, and each retry must not
                # evict another idle session before the first slot is freed.
                if not (self.idle_lru_eviction and
                        any(e.phase == 'evicting' for e in self.entries.values())):
                    self._evict(idle)
                self.waits += 1
                return None
            if self.idle_lru_eviction and any(e.phase == 'evicting' for e in self.entries.values()):
                self.waits += 1
                return None
            if self.require_owned or self.preserve_history:
                # Graph identities cannot degrade to temporary request IDs
                # when all owned checkpoint slots are currently leased.
                self.waits += 1
                return None
            # All cache slots are in use. A new session may still run normally;
            # it cannot create a checkpoint or touch any existing namespace,
            # but must exclude another turn until this request finishes.
            if share:
                from .request_plan import RequestCapacityError
                raise RequestCapacityError('kvmem_share_prefix: no session slot is free to retain the prefix')
            self.uncached_sessions[namespace] = request.request_id
            self.uncached_requests[request.request_id] = namespace
            return 0
        self.generation += 1
        entry = _Entry(namespace, self.generation, request.request_id)
        entry.share_prefix = share
        from .request_plan import request_plan
        plan = request_plan(request,self.block_tokens,self.gen_budget) if self.gen_budget else None
        if plan is not None:
            from dataclasses import replace
            if (plan.prefill_window=='keep_selected' and
                    request.num_prompt_tokens+plan.output_reserve>self.select_budget+self.gen_budget-self.block_tokens):
                plan = replace(plan,prefill_window='pressure')  # Original cold checkpoint-hole fallback.
            if self.guided_options is not None:
                from .guided_refresh import prepare_request
                plan = prepare_request(request, plan, entry, self.guided_options)
            self.set_plan(request,plan)
            entry.query,entry.mandatory = plan.query,plan.mandatory
            entry.query_id = plan.query_id
            entry.question_span = plan.query
            entry.question_id = plan.query_id if plan.query is not None else None
        self.entries[namespace] = self.requests[request.request_id] = entry
        return 0

    @staticmethod
    def output_tokens(request):
        params = getattr(request,'sampling_params',None)
        return getattr(params,'max_tokens',None) or getattr(request,'max_tokens',0) or 0

    def cold_plan(self,request):
        if not self.gen_budget:
            return
        from .request_plan import request_plan
        from dataclasses import replace
        plan=request_plan(request,self.block_tokens,self.gen_budget)
        self.validate_plan_capacity(request, plan)
        if (plan is not None and plan.prefill_window=='keep_selected' and
                request.num_prompt_tokens+plan.output_reserve>self.select_budget+self.gen_budget-self.block_tokens):
            self.set_plan(request,replace(plan,prefill_window='pressure'))

    def validate_plan_capacity(self, request, plan, *, history=0, window=0):
        if plan is None:
            return
        config = self.memory_config
        plan.validate_capacity(request.num_prompt_tokens, self.select_budget,
            sink_blocks=getattr(config, 'sink_blocks', 0),
            recent_blocks=getattr(config, 'recent_blocks', 0),
            resume_history=history, resume_window=window, pressure_policy=self.pressure_policy)

    @staticmethod
    def set_plan(request, plan):
        request.kv_transfer_params = {**(getattr(request,'kv_transfer_params',None) or {}),
                                      'kvmem_request_plan':plan.payload()}

    def snapshot(self):
        return [{'namespace':list(e.namespace),'history':e.history,'window':e.window,
                 'prefix_sha256':e.prefix_fingerprint(),'query':e.query,
                 'mandatory':e.mandatory,'generation':e.generation,'active_records':e.active_records,
                 'allocated_records':e.allocated_records,'protected_records':e.protected_records,
                 'ready_records':e.ready_records}
                for e in self.entries.values() if e.phase=='idle']

    def snapshot_json(self):
        """json.dumps(self.snapshot()), byte for byte, re-encoding only idle
        entries whose fields changed (opt-in KVMEM_LIFECYCLE_SNAPSHOT_CACHE).

        Every field is replaced, never mutated in place (active_records is a
        tuple held by identity; query/mandatory are tuples compared by value).
        """
        import json
        cache = self.__dict__.setdefault('_view_json', {})
        parts, live = [], set()
        for e in self.entries.values():
            if e.phase != 'idle':
                continue
            key = (e.history, e.window, e.prefix_fingerprint(), e.query, e.mandatory, e.generation,
                   e.allocated_records, e.protected_records, e.ready_records)
            hit = cache.get(e.namespace)
            if hit is None or hit[0] != key or hit[1] is not e.active_records:
                text = json.dumps({'namespace':list(e.namespace),'history':e.history,'window':e.window,
                    'prefix_sha256':e.prefix_fingerprint(),'query':e.query,
                    'mandatory':e.mandatory,'generation':e.generation,'active_records':e.active_records,
                    'allocated_records':e.allocated_records,'protected_records':e.protected_records,
                    'ready_records':e.ready_records})
                hit = cache[e.namespace] = (key, e.active_records, text)
            live.add(e.namespace)
            parts.append(hit[2])
        for namespace in [n for n in cache if n not in live]:
            del cache[namespace]
        return '[' + ', '.join(parts) + ']'

    def owner_snapshot(self):
        """All live namespace leases, including retirement before idle publish."""
        return [dict(namespace=list(e.namespace), generation=e.generation,
                     request_id=e.request_id, phase=e.phase)
                for e in self.entries.values()] + [
                    dict(namespace=list(namespace), generation=None,
                         request_id=request_id, phase='uncached')
                    for namespace, request_id in self.uncached_sessions.items()]

    def allocated(self, request, external_tokens):
        entry = self.requests.get(request.request_id)
        if entry is None:
            if external_tokens:
                raise ValueError('Unowned external prompt checkpoint')
            if (request.request_id in self.uncached_requests and session_release_intent(
                    getattr(request, 'kv_transfer_params', None),
                    getattr(request, 'session_id', None), request.prompt_token_ids,
                    self.output_tokens(request))):
                self.commands.release_requests.add(request.request_id)
            return
        if entry.phase != 'lookup' or entry.allocated or external_tokens != entry.history:
            raise ValueError('Session allocation differs from its exact prefix lease')
        entry.phase, entry.allocated = 'active', True
        from .request_plan import request_plan
        plan = request_plan(request,self.block_tokens,self.gen_budget) if self.gen_budget else None
        self.commands.bindings.append(SessionBinding(entry.namespace, entry.generation,
            request.request_id, tuple(request.prompt_token_ids), entry.history,
            entry.restore_window or entry.window,entry.pressure_restore,
            plan.query if plan is not None else None,plan.mandatory if plan is not None else (),
            plan.message_spans if plan is not None else (),
            plan.locality_spans if plan is not None else (),
            plan.locality_provenance if plan is not None else (),
            question_span=entry.question_span, question_id=entry.question_id,
            share_prefix=entry.share_prefix))
        self.hits += int(bool(external_tokens))
        self.misses += int(not external_tokens)

    def capture_before_final_prefill(self, request, query_tokens):
        entry = self.requests.get(request.request_id)
        position = request.num_computed_tokens
        from .request_plan import request_plan
        plan = request_plan(request,self.block_tokens,self.gen_budget) if self.gen_budget else None
        at_query = plan is not None and plan.replay_begin == position
        at_point = plan is not None and plan.prefill_checkpoint(request.num_prompt_tokens)==position
        if self.checkpoint_policy=='pm' and not at_point:
            return
        if (entry is None or entry.phase != 'active' or entry.candidate_position or
                not 0 < position < request.num_prompt_tokens or position % self.block_tokens or
                (position + query_tokens < request.num_prompt_tokens and not (at_query or at_point))):
            return
        if self.freeze_prompt and entry.retained_position == position:
            # An asynchronous import already owns the exact immutable P model,
            # window and input statistics. Recapturing it would duplicate DMA
            # and replace the snapshot needed if this request is cancelled.
            if token_bytes(request.prompt_token_ids, position) != entry.retained_tokens:
                raise ValueError('Retained frozen prompt differs from the current token prefix')
            return
        entry.candidate_position = position
        entry.prompt_guided_anchor = entry.guided_anchor
        entry.candidate_tokens = token_bytes(request.prompt_token_ids, position)
        self.commands.captures.append(SessionCapture(request.request_id, entry.generation, position))

    def capture_decode_boundary(self, request):
        """Advance the persistent accepted prefix, never a speculative suffix."""
        entry = self.requests.get(request.request_id)
        position = request.num_computed_tokens
        if (entry is None or entry.phase != 'active' or entry.candidate_position or
                position < request.num_prompt_tokens or position <= entry.retained_position or
                not position or position % self.block_tokens):
            return
        tokens = tuple(request.all_token_ids)
        entry.candidate_position = position
        entry.candidate_tokens = token_bytes(tokens, position)
        self.commands.captures.append(SessionCapture(request.request_id, entry.generation, position, tokens))

    def finished(self, request):
        importing = self.importing.pop(request.request_id, None)
        if importing is not None:
            # Its session is still being materialized from a prefix image. The
            # import completes and the orphan session is then evicted.
            entry = self.entries.get(importing)
            if entry is not None and entry.phase == 'importing':
                entry.import_orphan = True
            return
        self.import_declined.discard(request.request_id)
        namespace = self.uncached_requests.pop(request.request_id, None)
        if namespace is not None:
            del self.uncached_sessions[namespace]
            return
        entry = self.requests.get(request.request_id)
        if entry is None:
            return
        if not entry.allocated:
            del self.requests[request.request_id]
            if entry.history:
                self.abort_import(request.request_id)
                entry.phase, entry.request_id = 'idle', None
                entry.pressure_restore, entry.restore_window = False, 0
                entry.share_prefix = 0
            else:
                del self.entries[entry.namespace]
            return
        if entry.phase != 'active':
            raise ValueError('Duplicate session terminal transition')
        entry.phase = 'retiring'
        if self.terminal_capture and request.status.name in ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED'):
            from .terminal import TerminalPrefix
            position = TerminalPrefix.from_request(request).history_end
            if position:
                # The final prefill batch can capture P and finish the request
                # before its connector acknowledgment is consumed. M has its
                # own pending identity; an unacknowledged P is not retained yet.
                entry.terminal_position = position
                tokens = tuple(request.all_token_ids)
                entry.terminal_tokens = token_bytes(tokens, position)
                self.commands.terminals.append(SessionCapture(request.request_id, entry.generation, position, tokens))

    def _evict(self, entry):
        if entry.phase != 'idle':
            raise ValueError('Cannot evict a leased or retiring session')
        entry.phase = 'evicting'
        self.commands.evictions.append(SessionEviction(entry.namespace, entry.generation))

    def consume(self, notice):
        if notice.kind in _PREFIX_IMAGE_NOTICES:
            return self._consume_prefix_image(notice)
        entry = self.entries.get(notice.namespace)
        # Idle observations can already be in a worker result when the next
        # scheduler iteration leases/evicts the scope. They are advisory and
        # must never replace a running turn's fenced reservation.
        if notice.kind == 'history_capacity' and (entry is None or
                entry.generation != notice.generation or entry.phase not in ('idle', 'lookup')):
            return False
        if entry is None or entry.generation != notice.generation:
            raise ValueError('Stale or foreign session completion')
        if notice.kind == 'history_capacity':
            if (entry.phase not in ('idle', 'lookup') or notice.request_id or
                    notice.history != entry.scope_history or
                    any(type(v) is not int or v < 0 for v in
                        (notice.ready_records, notice.allocated_records, notice.protected_records)) or
                    notice.protected_records > notice.allocated_records or
                    notice.ready_records < entry.ready_records or
                    notice.allocated_records > entry.allocated_records):
                raise ValueError('Stale or consumed idle historical capacity')
            entry.ready_records, entry.allocated_records = notice.ready_records, notice.allocated_records
            entry.protected_records = notice.protected_records
            return
        if notice.kind == 'evicted':
            if entry.phase != 'evicting' or notice.request_id or notice.history or notice.window:
                raise ValueError('Unexpected session eviction completion')
            del self.entries[entry.namespace]
            if self.prefix_images is not None:
                self.prefix_images.forget_namespace(entry.namespace)
            return
        if entry.request_id != notice.request_id:
            raise ValueError('Session completion belongs to another native request')
        if notice.kind in ('captured', 'capture_deferred'):
            terminal = notice.history == entry.terminal_position and bool(entry.terminal_position)
            position = entry.terminal_position if terminal else entry.candidate_position
            identity = entry.terminal_tokens if terminal else entry.candidate_tokens
            if (entry.phase not in ('active', 'retiring') or
                    notice.history != position or not notice.history):
                raise ValueError('Checkpoint acknowledgment differs from the requested accepted prefix')
            if notice.kind == 'captured' and notice.history >= entry.retained_position:
                entry.retained_position = notice.history
                entry.retained_tokens = identity
            if terminal:
                entry.terminal_position, entry.terminal_tokens = 0, b''
            else:
                entry.candidate_position, entry.candidate_tokens = 0, b''
            return
        if notice.kind == 'restored':
            if (entry.phase not in ('active', 'retiring') or not entry.history or
                    (notice.history, notice.window) != (entry.history, entry.restore_window or entry.window)):
                raise ValueError('Unexpected prompt import completion')
            self.commit_import(notice.request_id, notice.history)
            if notice.retained:
                entry.retained_position, entry.retained_tokens = entry.history, entry.tokens
            entry.history = entry.window = 0
            entry.pressure_restore, entry.restore_window = False, 0
            entry.tokens = b''
            return
        if notice.kind != 'retired' or entry.phase != 'retiring':
            raise ValueError('Unexpected session retirement completion')
        if entry.history:
            raise ValueError('Retirement arrived before prompt import restoration')
        if getattr(notice, 'discard_shards', False):
            # One worker retained a different P/M checkpoint under CPU
            # pressure. Keep the namespace fenced until every shard is gone.
            entry.request_id, entry.phase = None, 'evicting'
            self.commands.evictions.append(SessionEviction(entry.namespace, entry.generation, allow_missing=True))
            del self.requests[notice.request_id]
            return
        if notice.history:
            expected = entry.retained_position or entry.candidate_position
            identity = entry.retained_tokens or entry.candidate_tokens
            if (notice.history != expected or not 0 < notice.window <= notice.history or
                    type(notice.ready_records) is not int or notice.ready_records < 0):
                raise ValueError('Worker retained a different prompt checkpoint')
            if (type(notice.allocated_records) is not int or notice.allocated_records < 0 or
                    type(notice.protected_records) is not int or not 0 <= notice.protected_records <= notice.allocated_records):
                raise ValueError('Worker retained invalid historical CPU capacity')
            if (notice.prompt_history and (not 0<notice.prompt_history<notice.history or
                    not 0<notice.prompt_window<=notice.prompt_history or
                    notice.prompt_tokens!=identity[:notice.prompt_history*4])):
                raise ValueError('P checkpoint differs from the retained token lineage')
            entry.history, entry.window = notice.history, notice.window
            entry.prompt_history,entry.prompt_window,entry.prompt_tokens = notice.prompt_history,notice.prompt_window,notice.prompt_tokens
            entry.active_records = notice.active_records
            entry.ready_records = notice.ready_records
            entry.allocated_records = notice.allocated_records
            entry.protected_records = notice.protected_records
            entry.scope_history = notice.history
            entry.tokens = identity
            entry.candidate_position, entry.candidate_tokens = 0, b''
            entry.terminal_position, entry.terminal_tokens = 0, b''
            entry.retained_position, entry.retained_tokens = 0, b''
            entry.share_prefix = 0
            entry.request_id, entry.phase, entry.allocated = None, 'idle', False
            # Most recently retired session is the last idle eviction candidate.
            del self.entries[entry.namespace]
            self.entries[entry.namespace] = entry
        else:
            if notice.window:
                raise ValueError('Empty checkpoint has a nonempty target frame')
            del self.entries[entry.namespace]
            if self.prefix_images is not None:
                self.prefix_images.forget_namespace(entry.namespace)
        del self.requests[notice.request_id]

    def take_commands(self):
        result, self.commands = self.commands, SessionCommands()
        return result

    def has_pending_work(self):
        return any(e.phase in ('retiring', 'evicting', 'importing') for e in self.entries.values())

    def import_scheduled(self, request_id):
        return any(b.request_id == request_id and b.history for b in self.commands.bindings)

    # Prefix images (docs/prefix-sharing.md, serving/prefix_images.py) ---------

    def _share_prefix_intent(self, request, namespace, release):
        """Validated kv_transfer_params.kvmem_share_prefix of this request, or 0.

        P must be the frozen prompt position this request captures; the worker
        registers the idle P checkpoint as an image after the request retires.
        """
        params = getattr(request, 'kv_transfer_params', None)
        if not isinstance(params, dict) or params.get('kvmem_share_prefix') is None:
            return 0
        from .request_plan import RequestCapacityError, request_plan
        value = params['kvmem_share_prefix']

        def refuse(reason):
            raise RequestCapacityError('kvmem_share_prefix: ' + reason)
        if type(value) is not int or value <= 0:
            refuse('must be a positive integer token position')
        if namespace is None or release:
            refuse('requires an owned session (session_id) and cannot close it')
        if self.prefix_image_scope is not None:
            refuse('outside the prefix image scope: ' + self.prefix_image_scope)
        if not self.gen_budget:
            refuse('requires request planning')
        if value % self.block_tokens or value >= request.num_prompt_tokens:
            refuse('%d must be a block boundary before the end of the prompt' % value)
        plan = request_plan(request, self.block_tokens, self.gen_budget)
        point = plan.prefill_checkpoint(request.num_prompt_tokens) if plan is not None else None
        if point != value:
            refuse('%d is not the frozen prompt position of this request (%s)' % (value, point))
        return value

    def _start_prefix_import(self, request, namespace):
        """A new session whose prompt starts with a registered image: materialize it
        from the image (worker) instead of prefilling; the request waits until then.

        A full ledger keeps today's rules (wait or evict) and retries later.
        """
        index = self.prefix_images
        if (not index.images or request.request_id in self.import_declined or
                len(self.entries) >= self.capacity):
            return False
        image = index.match(namespace[0], request.prompt_token_ids)
        if image is None:
            return False
        from .prefix_images import SessionImport
        self.generation += 1
        entry = _Entry(namespace, self.generation, None, phase='importing')
        entry.import_request, entry.import_key = request.request_id, image.key
        self.entries[namespace] = entry
        self.importing[request.request_id] = namespace
        index.acquire(image.key)
        self.commands.image_imports.append(SessionImport(namespace, entry.generation, image.key))
        return True

    def _consume_prefix_image(self, notice):
        """Image registration and import notices. False: no other consumer
        (CPU history credits, residency) reads the notice. Registration, image
        release and pool notices also reach the historical CPU admission (pool
        charge), which ignores the rest."""
        if notice.kind == 'image_refused':
            return False
        if notice.kind in ('history_pool', 'image_released'):
            return True
        entry = self.entries.get(notice.namespace)
        if notice.kind == 'image_registered':
            if (entry is None or entry.generation != notice.generation or entry.phase != 'idle' or
                    entry.prompt_tokens_for(notice.history) != notice.tokens):
                # Only the idle parent holds the continuation state an import
                # must reproduce (Query, pins, guided anchors); drop the image.
                self.commands.image_releases.append(notice.key)
                return True
            if self.prefix_images is None:
                from .prefix_images import PrefixImageIndex, max_image_bytes
                from ..core.storage import prefix_pool_bytes
                self.prefix_images = PrefixImageIndex(self.prefix_image_model,
                                                      prefix_pool_bytes() or max_image_bytes())
                self.prefix_images.entries = self.entries
            self.commands.image_releases.extend(self.prefix_images.register(notice, entry))
            return True
        if (entry is None or entry.generation != notice.generation or entry.phase != 'importing' or
                notice.request_id):
            raise ValueError('Unexpected prefix image import completion')
        index = self.prefix_images
        key, request_id, orphan = entry.import_key, entry.import_request, entry.import_orphan
        entry.import_key = entry.import_request = None
        entry.import_orphan = False
        if self.importing.get(request_id) == notice.namespace:
            del self.importing[request_id]
        image = index.images[key]       # an in-flight import holds a reference
        if notice.kind == 'import_failed':
            del self.entries[notice.namespace]
            if not orphan:
                self.import_declined.add(request_id)      # it prefills cold instead
            self.commands.image_releases.extend(index.release_ref(key))
            return False
        if notice.history != key.position or not 0 < notice.window <= notice.history:
            raise ValueError('Imported session differs from its prefix image')
        # Exactly the idle state the parent session reached at P (retired notice).
        entry.history, entry.window, entry.tokens = notice.history, notice.window, image.tokens
        entry.active_records = notice.active_records
        entry.ready_records = notice.ready_records
        entry.allocated_records = notice.allocated_records
        entry.protected_records = notice.protected_records
        entry.scope_history = notice.history
        for name, value in image.ledger.items():
            setattr(entry, name, value)
        entry.request_id, entry.phase, entry.allocated = None, 'idle', False
        del self.entries[notice.namespace]
        self.entries[notice.namespace] = entry
        index.imported(key, notice.namespace)
        self.commands.image_releases.extend(index.release_ref(key))
        if orphan:
            self._evict(entry)
        return True
