"""Scheduler-thread CPU history credits; no GPU allocator or model policy.

Credits cover a complete accepted turn, persist across tool waits, and are only
released after the worker's generation-qualified retirement/eviction fence.
An optional growth reserve lets one resident trajectory reach the configured
history ceiling without evicting or recomputing another session's history.
"""
from dataclasses import dataclass
import time


@dataclass
class HistoryLease:
    generation: int
    bytes: int
    request_id: str | None
    phase: str = 'active'


class HistoryAdmission:
    def __init__(self, capacity_bytes, block_tokens, bytes_per_token, max_tokens,
                 *, reserve_progress=True, clock=time.monotonic):
        if any(type(v) is not int or v <= 0 for v in
               (capacity_bytes, block_tokens, bytes_per_token, max_tokens)):
            raise ValueError('Invalid CPU history credit geometry')
        self.block_tokens, self.bytes_per_token = block_tokens, bytes_per_token
        self.record_bytes = block_tokens * bytes_per_token
        self.capacity_bytes = capacity_bytes // self.record_bytes * self.record_bytes
        self.max_tokens, self.reserve_progress = max_tokens, reserve_progress
        self.clock = clock
        self.leases, self.requests, self.waiting = {}, {}, {}
        self.deferrals = self.admissions = self.total_wait_seconds = 0
        self.observed_bytes = 0
        # Two partial record generations may coexist at P/M boundaries. Model
        # state, gather/DMA buffers and token identities have separate budgets.
        self.maximum_bytes = self.extent_bytes(max_tokens)
        if self.maximum_bytes > self.capacity_bytes:
            raise ValueError('CPU history budget cannot fit one full trajectory')

    def extent_bytes(self, tokens):
        if type(tokens) is not int or not 0 <= tokens <= self.max_tokens:
            raise ValueError('History extent exceeds its declared ceiling')
        return ((tokens + self.block_tokens - 1) // self.block_tokens + 2) * self.record_bytes

    @staticmethod
    def key(request):
        from .session_control import SessionLedger, session_release_intent
        namespace = SessionLedger.namespace(request)
        release = session_release_intent(getattr(request,'kv_transfer_params',None),
            getattr(request,'session_id',None),getattr(request,'prompt_token_ids',None),
            SessionLedger.output_tokens(request))
        return ('session', *namespace) if namespace is not None and not release else ('request', request.request_id)

    @staticmethod
    def closing_namespace(request, key):
        # A namespace-bearing request has an ephemeral key only after the
        # explicit close intent was validated by key(). Ordinary requests keep
        # their original accounting and do not enter this branch.
        if key[0] == 'request' and getattr(request,'session_id',None):
            from .session_control import SessionLedger
            return ('session', *SessionLedger.namespace(request))
        return None

    def demand(self, request):
        from .session_control import SessionLedger
        end = min(self.max_tokens, request.num_prompt_tokens + SessionLedger.output_tokens(request))
        return self.extent_bytes(end)

    def _usage(self, amounts):
        used = sum(amounts.values())
        resident = [n for key, n in amounts.items() if key[0] == 'session']
        reserve = max(0, self.maximum_bytes - max(resident)) if resident and self.reserve_progress else 0
        return used, reserve

    def can_admit(self, request):
        """Read-only preview; ordering must not create leases or wait counters."""
        key = self.key(request)
        closing = self.closing_namespace(request,key)
        if closing in self.leases:
            # Reach SessionLedger's eviction even when every byte is reserved.
            # This is not allocation: the close cannot run until the old
            # generation's worker ACK releases these credits, checked below.
            return True
        lease = self.leases.get(key)
        if lease is not None and (lease.phase == 'evicting' or lease.request_id not in (None, request.request_id)):
            allowed = False
        else:
            amounts = {k: v.bytes for k, v in self.leases.items()}
            amounts[key] = max(amounts.get(key, 0), self.demand(request))
            used, reserve = self._usage(amounts)
            allowed = used + reserve <= self.capacity_bytes
        return allowed

    def allows(self, request):
        allowed = self.can_admit(request)
        if not allowed:
            self.deferrals += 1
            self.waiting.setdefault(request.request_id, self.clock())
        return allowed

    def allocated(self, request, generation=0):
        rid, key = request.request_id, self.key(request)
        closing = self.closing_namespace(request,key)
        if closing is not None and (generation or closing in self.leases):
            raise ValueError('Session close allocation precedes fenced eviction or has a retained generation')
        if rid in self.requests or not self.allows(request):
            raise RuntimeError('CPU history allocation without available credits')
        previous = self.leases.get(key)
        if previous is not None and previous.generation != generation:
            raise ValueError('CPU history lease generation changed without eviction')
        amount = max(previous.bytes if previous else 0, self.demand(request))
        self.leases[key] = HistoryLease(generation, amount, rid)
        self.requests[rid] = key
        self.admissions += 1
        began = self.waiting.pop(rid, None)
        if began is not None:
            self.total_wait_seconds += self.clock() - began

    def finished(self, request):
        self.waiting.pop(request.request_id, None)
        key = self.requests.get(request.request_id)
        if key is not None:
            lease = self.leases[key]
            if lease.phase != 'active':
                raise ValueError('Duplicate CPU history retirement')
            lease.phase = 'retiring'

    def consume(self, notice):
        """Called only AFTER the session ledger has validated the worker notice."""
        if notice.kind == 'imported':
            # A new session materialized from a prefix image holds a retained
            # history of P tokens, exactly as after a retired turn that kept P.
            key = ('session', *notice.namespace)
            if key in self.leases:
                raise ValueError('Prefix image import reused a live CPU history lease')
            self.leases[key] = HistoryLease(notice.generation, self.extent_bytes(notice.history), None, 'idle')
            return
        if notice.kind not in ('retired', 'evicted'):
            return
        key = ('session', *notice.namespace)
        lease = self.leases.get(key)
        if lease is None or lease.generation != notice.generation:
            raise ValueError('Stale CPU history completion')
        if notice.kind == 'evicted':
            if lease.request_id is not None:
                raise ValueError('Cannot release a leased CPU history')
            del self.leases[key]
            return
        if lease.request_id != notice.request_id or lease.phase != 'retiring':
            raise ValueError('CPU history retirement belongs to another request')
        if notice.history or getattr(notice, 'discard_shards', False):
            # Keep the high-water reservation: a P fallback or an in-flight
            # immutable version must never be mistaken for reclaimed storage.
            if self.extent_bytes(notice.history) > lease.bytes:
                raise ValueError('Worker history exceeded its admitted extent')
            lease.request_id, lease.phase = None, 'idle'
        else:
            del self.leases[key]
        del self.requests[notice.request_id]

    def released(self, request_id):
        """Native finished_sending fence for requests without retained sessions."""
        key = self.requests.get(request_id)
        if key is None:
            return  # Session notice already settled this request.
        lease = self.leases[key]
        if key[0] != 'request' or lease.phase != 'retiring':
            raise ValueError('Missing session retirement before CPU credit release')
        del self.requests[request_id], self.leases[key]

    def snapshot(self):
        used, reserve = self._usage({k: v.bytes for k, v in self.leases.items()})
        return dict(capacity_bytes=self.capacity_bytes, reserved_bytes=used,
                    storage_tier=getattr(self, 'storage_tier', 'cpu'),
                    unmaterialized_reserved_bytes=max(0, used-self.observed_bytes),
                    progress_reserve_bytes=reserve, available_bytes=self.capacity_bytes-used-reserve,
                    leases=len(self.leases), active=sum(v.phase == 'active' for v in self.leases.values()),
                    lease_views=[dict(key=list(key), generation=lease.generation,
                                      request_id=lease.request_id, phase=lease.phase, bytes=lease.bytes)
                                 for key, lease in self.leases.items()],
                    retiring=sum(v.phase == 'retiring' for v in self.leases.values()),
                    waiting=len(self.waiting), deferrals=self.deferrals, admissions=self.admissions,
                    last_observed_archive_bytes=self.observed_bytes,
                    total_wait_seconds=self.total_wait_seconds,
                    oldest_wait_seconds=max((self.clock()-t for t in self.waiting.values()), default=0))

    def observe(self, amount):
        if type(amount) is not int or not 0 <= amount <= sum(v.bytes for v in self.leases.values()):
            raise ValueError('CPU archive storage exceeded its history credits')
        self.observed_bytes = amount
