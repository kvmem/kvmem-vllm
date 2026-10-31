"""Reserve maintenance capacity before allocating a native request slot.

There is one resident producer and one shared cold/Query assembler. Waiting
imports keep their CPU session lease but do not allocate native pages/indices.
An admitted hot view is borrowed during the synchronous scheduler transaction:
TTL/pressure cannot change its lane between prefix lookup and allocation.
"""
from dataclasses import dataclass
import json
import time


@dataclass
class RestorePermit:
    request: object
    namespace: tuple
    generation: int
    history: int
    window: int
    view: object = None
    allocated: bool = False
    command_id: int | None = None
    dispatched_at: float | None = None
    cancelled: bool = False

    @property
    def lane(self):
        return 'resident' if self.view is not None else 'assembly'


class RestoreAdmission:
    def __init__(self, *, residency=None, cold_busy=lambda: False, output=None, clock=time.perf_counter,
                 resident_slots=1):
        if type(resident_slots) is not int or not 1 <= resident_slots <= 32:
            raise ValueError('Resident restore slots must be between 1 and 32')
        self.resident_slots = resident_slots
        self.residency = residency
        self.cold_busy = cold_busy
        self.output, self.clock = output, clock
        self.permits = {}
        self.waiting = {}
        self.deferred = {'resident': 0, 'assembly': 0}
        self.admitted = {'resident': 0, 'assembly': 0}
        self.rolled_back = 0
        self.peak_occupied = {'resident': 0, 'assembly': 0}

    def _record(self, request_id, stage, lane, **fields):
        now = self.clock()
        record = self.waiting.setdefault(request_id, dict(first_seen=now, first_deferred_at=None, state=None))
        if stage == 'deferred' and record['first_deferred_at'] is None:
            record['first_deferred_at'] = now
        state = (stage, lane, fields.get('reason'))
        if record['state'] == state:
            return
        record['state'] = state
        if self.output is not None:
            with self.output.open('a') as stream:
                stream.write(json.dumps(dict(request_id=request_id, stage=stage, lane=lane, at=now,
                    first_seen_at=record['first_seen'], first_deferred_at=record['first_deferred_at'],
                    **fields)) + '\n')

    def lookup(self, request, entry):
        """Called after exact prefix/Query planning, before native allocation.

        False is translated to the connector's (None, False), which makes the
        pinned native waiting loop CONTINUE with other requests. No artificial
        prefix hit or native asynchronous-KV status is introduced.
        """
        if not entry.history:
            return True
        window = getattr(entry, 'restore_window', 0) or entry.window
        rid = request.request_id
        existing = self.permits.get(rid)
        if existing is not None:
            if (existing.request is not request or existing.allocated or
                    (existing.namespace, existing.generation, existing.history, existing.window) !=
                    (entry.namespace, entry.generation, entry.history, window)):
                raise ValueError('Restore admission identity changed before allocation')
            return True
        view = self.residency.match(entry) if self.residency is not None else None
        lane = 'resident' if view is not None else 'assembly'
        if rid not in self.waiting:
            self._record(rid, 'arrival', lane)
        occupied = sum(p.lane == lane for p in self.permits.values()) >= (
            self.resident_slots if lane == 'resident' else 1)
        query_busy = view is None and self.cold_busy()
        if occupied or query_busy:
            self.deferred[lane] += 1
            self._record(rid, 'deferred', lane, reason='lane_busy' if occupied else 'query_assembly')
            return False
        permit = RestorePermit(request, entry.namespace, entry.generation, entry.history, window, view)
        if view is not None:
            # Stages only existing attention pages, not a runner index or GDN
            # allocation. A failed/unused allocation returns them unchanged.
            self.residency.claim(view, rid)
        self.permits[rid] = permit
        self.peak_occupied[lane] = max(self.peak_occupied[lane],
            sum(p.lane == lane for p in self.permits.values()))
        return True

    def before_allocate(self, request, external_tokens):
        permit = self.permits.get(request.request_id)
        if (permit is None or permit.request is not request or permit.allocated or
                external_tokens != permit.history):
            raise ValueError('Native restore allocation has no exact admission permit')
        if permit.view is not None:
            for group, blocks in permit.view.groups:
                manager = self.residency.managers[group]
                if (manager.req_to_blocks.get(request.request_id) or
                        manager.resident_imports.get(request.request_id) != blocks):
                    raise ValueError('Admitted resident pages changed before native allocation')
        return permit

    def allocated(self, request):
        permit = self.permits[request.request_id]
        if permit.request is not request or permit.allocated:
            raise ValueError('Duplicate or foreign restore allocation')
        permit.allocated = True
        self.admitted[permit.lane] += 1
        self._record(request.request_id, 'admitted', permit.lane, admitted_at=self.clock(),
                     admission_wait_seconds=self.clock()-self.waiting[request.request_id]['first_seen'])

    def rollback(self, request):
        permit = self.permits.get(request.request_id)
        if permit is None:
            return
        if permit.request is not request or permit.allocated or permit.command_id is not None:
            raise ValueError('Allocated restore permit must remain leased until its completion')
        if permit.view is not None:
            self.residency.rollback(permit.view, request.request_id)
        del self.permits[request.request_id]
        self.rolled_back += 1
        self._record(request.request_id, 'deferred', permit.lane, reason='native_admission')

    def retain_failed_allocation(self, request):
        # Fatal partial native mutation: keep ownership but never report a
        # successful admission or return possibly modified pages to idle reuse.
        permit = self.permits[request.request_id]
        permit.allocated = True
        self._record(request.request_id, 'allocation_failed', permit.lane)

    def end_step(self):
        # Native token/phase or memory admission can stop after prefix lookup.
        # Such a lookup must not monopolize a lane across scheduler iterations.
        for permit in list(self.permits.values()):
            if not permit.allocated:
                self.rollback(permit.request)

    def dispatch(self, command):
        permit = self.permits.get(command.request_id)
        if (permit is None or not permit.allocated or permit.command_id is not None or
                (permit.generation, permit.history, permit.window, permit.view is not None) !=
                (command.generation, command.history_end, command.window_tokens, command.resident)):
            raise ValueError('Restore command differs from its admitted native allocation')
        permit.command_id = command.command_id
        permit.dispatched_at = self.clock()

    def complete(self, request_id, command_id):
        permit = self.permits.get(request_id)
        if permit is None or not permit.allocated or permit.command_id != command_id:
            raise ValueError('Stale or foreign restore admission completion')
        # The worker ACK follows the CUDA fence and scheduler import commit.
        # Cancellation is deliberately not a completion signal.
        if (permit.view is None and not permit.cancelled and permit.dispatched_at is not None and
                self.residency is not None and self.residency.tool_grace is not None):
            self.residency.tool_grace.observe_restore(self.clock() - permit.dispatched_at)
        self._record(request_id, 'ready', permit.lane, command_id=command_id,
                     total_wait_seconds=self.clock()-self.waiting[request_id]['first_seen'],
                     cancelled=permit.cancelled)
        del self.permits[request_id]
        self.waiting.pop(request_id, None)

    def cancel(self, request):
        permit = self.permits.get(request.request_id)
        if permit is not None and permit.request is not request:
            raise ValueError('Restore cancellation belongs to another request incarnation')
        if permit is not None and permit.allocated:
            permit.cancelled = True
            self._record(request.request_id, 'cancelled_inflight', permit.lane, command_id=permit.command_id)
        if permit is not None and not permit.allocated:
            self.rollback(request)
        if permit is None or not permit.allocated:
            if request.request_id in self.waiting:
                lane = self.waiting[request.request_id]['state'][1]
                self._record(request.request_id, 'cancelled', lane)
                del self.waiting[request.request_id]

    def snapshot(self):
        return dict(capacity={'resident': self.resident_slots, 'assembly': 1},
                    occupied={lane: sum(p.lane == lane for p in self.permits.values())
                              for lane in ('resident', 'assembly')},
                    requests={rid: dict(lane=p.lane, allocated=p.allocated, command_id=p.command_id, cancelled=p.cancelled)
                              for rid, p in self.permits.items()},
                    waiting=sum(rid not in self.permits or not self.permits[rid].allocated for rid in self.waiting),
                    deferred=dict(self.deferred), admitted=dict(self.admitted), rolled_back=self.rolled_back,
                    peak_occupied=dict(self.peak_occupied))
