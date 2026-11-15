"""Scheduler-owned whole-turn KV reservations, independent of GC progress.

Snapshots describe allocated CPU descriptors, including retired generations
whose readers or cleanup still own capacity. Pending GC bytes are never credit.
Only a fenced worker settlement can replace a running turn's reservation.
"""
from dataclasses import dataclass
import os
from ..core import settings as _settings


def history_lru_eviction_enabled():
    """KVMEM_SESSION_HISTORY_LRU_EVICTION=1: a CPU history deficit that idle
    sessions' unprotected records cannot cover evicts the least recently used
    idle session (fenced, one at a time) instead of deferring every turn
    forever. Without it a host budget filled by parked sessions whose whole
    history is protected (short histories: the per-layer checkpoint windows
    cover almost all of it) blocks every returning turn. The evicted session's
    next turn is a cold prefill."""
    value = _settings.get('KVMEM_SESSION_HISTORY_LRU_EVICTION', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_SESSION_HISTORY_LRU_EVICTION must be 0 or 1')
    return value == '1'


def _extent(*values):
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('Historical capacity requires nonnegative integer extents')


@dataclass(frozen=True)
class HistoricalCapacitySnapshot:
    generation: int
    history_end: int
    ready_records: int
    allocated_records: int
    protected_records: int = 0

    def __post_init__(self):
        _extent(self.generation, self.history_end, self.ready_records, self.allocated_records, self.protected_records)
        if self.protected_records > self.allocated_records:
            raise ValueError('Protected historical data exceeds allocated CPU records')
        if not self.generation:
            raise ValueError('Historical capacity requires a scope generation')


@dataclass(frozen=True)
class HistoricalTurnReservation:
    namespace: object
    generation: int
    request_id: str
    base_history: int
    history_limit: int
    incoming_records: int
    retained_records: int
    extra_records: int

    @property
    def reserved_records(self):
        return self.retained_records + self.incoming_records + self.extra_records


class HistoricalAdmission:
    def __init__(self, capacity_records, repository_records, block_tokens, *,
                 lookahead_tokens=1, pipeline_packets=2, checkpoint_versions=2):
        _extent(capacity_records, repository_records, block_tokens, lookahead_tokens,
                pipeline_packets, checkpoint_versions)
        if not min(capacity_records, repository_records, block_tokens, lookahead_tokens, pipeline_packets):
            raise ValueError('Historical reservation geometry must be positive')
        self.capacity_records, self.repository_records, self.block_tokens = capacity_records, repository_records, block_tokens
        # Rejected native rows and P/M tail versions may still be draining.
        # Each in-flight packet is charged a complete partial-record envelope.
        self.extra_records = checkpoint_versions + pipeline_packets * (
            (lookahead_tokens + block_tokens - 1) // block_tokens)
        if self.extra_records >= min(capacity_records, repository_records):
            raise ValueError('Historical capacity cannot fit the in-flight envelope')
        self.snapshots, self.reservations, self.retiring = {}, {}, set()
        self.admissions = self.deferrals = self.settlements = 0
        # Shared prefixes (docs/prefix-sharing.md, phase 2): distinct payloads charged
        # once to the shared-prefix pool, as the worker last reported them. The
        # snapshot reports the pool once it has been used.
        self.pool_records = 0
        self.pool_used = False

    def _validate_snapshot(self, snapshot):
        if (not isinstance(snapshot, HistoricalCapacitySnapshot) or
                snapshot.ready_records > self.repository_records or
                snapshot.allocated_records > self.capacity_records or
                snapshot.ready_records + snapshot.allocated_records > self.repository_records):
            raise ValueError('Invalid historical scope capacity')

    def observe_idle(self, namespace, snapshot):
        """An idle scope can return only already-reclaimed, observed capacity."""
        self._validate_snapshot(snapshot)
        if namespace in self.reservations:
            raise ValueError('Running/retiring reservations require a fenced settlement')
        previous = self.snapshots.get(namespace)
        if previous is not None and (snapshot.generation != previous.generation or
                snapshot.history_end != previous.history_end or
                snapshot.allocated_records > previous.allocated_records or
                snapshot.ready_records < previous.ready_records):
            raise ValueError('Idle capacity observation changed its identity or consumed capacity')
        if self.reserved_records - (previous.allocated_records if previous else 0) + snapshot.allocated_records > self.capacity_records:
            raise ValueError('Worker history exceeds admitted CPU capacity')
        self.snapshots[namespace] = snapshot

    @property
    def reserved_records(self):
        return sum(reservation.reserved_records for reservation in self.reservations.values()) + sum(
            snapshot.allocated_records for namespace, snapshot in self.snapshots.items()
            if namespace not in self.reservations) + self.pool_records

    def observe_pool(self, records):
        _extent(records)
        self.pool_records = records
        self.pool_used |= bool(records)

    def transfer_to_pool(self, namespace, generation, records, pool_records):
        """A registration moved ``records`` of an idle session to the pool charge,
        atomically with the pool's new size (never a transient double count)."""
        _extent(records, pool_records)
        snapshot = self.snapshots.get(namespace)
        if (records and snapshot is not None and snapshot.generation == generation and
                namespace not in self.reservations):
            allocated = max(0, snapshot.allocated_records - records)
            self.snapshots[namespace] = HistoricalCapacitySnapshot(snapshot.generation, snapshot.history_end,
                snapshot.ready_records, allocated, min(snapshot.protected_records, allocated))
        self.pool_records = pool_records
        self.pool_used |= bool(pool_records)

    def import_idle(self, namespace, snapshot):
        """An idle session materialized from a prefix image. Its private charge (the
        index copies of its aliased records) exists already; it is recorded even
        when it overcommits, and later admissions wait for reclamation."""
        self._validate_snapshot(snapshot)
        if namespace in self.reservations or namespace in self.snapshots:
            raise ValueError('A prefix image import reused a live historical scope')
        self.snapshots[namespace] = snapshot

    def preview(self, namespace, snapshot, request_id, *, base_history, history_limit):
        self._validate_snapshot(snapshot)
        _extent(base_history, history_limit)
        if not request_id or history_limit < base_history or base_history > snapshot.history_end:
            raise ValueError('Historical reservation has an invalid accepted-prefix interval')
        if namespace in self.reservations:
            raise ValueError('Historical scope already owns an execution reservation')
        previous = self.snapshots.get(namespace)
        if previous is not None and previous != snapshot:
            raise ValueError('Historical admission requires its latest idle capacity snapshot')
        tail = base_history % self.block_tokens
        incoming = (tail + history_limit - base_history + self.block_tokens - 1) // self.block_tokens - bool(tail)
        reservation = HistoricalTurnReservation(namespace, snapshot.generation, request_id,
            base_history, history_limit, incoming, snapshot.allocated_records, self.extra_records)
        occupied = self.reserved_records - (previous.allocated_records if previous else 0)
        if (incoming + self.extra_records > snapshot.ready_records or
                occupied + reservation.reserved_records > self.capacity_records):
            return None
        return reservation

    def admit(self, namespace, snapshot, request_id, *, base_history, history_limit):
        reservation = self.preview(namespace, snapshot, request_id,
                                   base_history=base_history, history_limit=history_limit)
        if reservation is None:
            self.deferrals += 1
            return None
        self.snapshots[namespace] = snapshot
        self.reservations[namespace] = reservation
        self.admissions += 1
        return reservation

    def finish(self, namespace, generation, request_id):
        reservation = self.reservations.get(namespace)
        if (reservation is None or (reservation.generation, reservation.request_id) != (generation, request_id) or
                namespace in self.retiring):
            raise ValueError('Stale or duplicate historical finish')
        self.retiring.add(namespace)

    def settle(self, namespace, generation, request_id, snapshot=None):
        """Worker ACK after publication, GPU/CPU readers and rollback drain.

        A retained snapshot keeps every still-allocated record charged. None
        releases a destroyed scope, never merely an EOS or cancellation intent.
        """
        reservation = self.reservations.get(namespace)
        if (reservation is None or namespace not in self.retiring or
                (reservation.generation, reservation.request_id) != (generation, request_id)):
            raise ValueError('Stale or premature historical settlement')
        if snapshot is not None:
            self._validate_snapshot(snapshot)
            if (snapshot.generation != generation or snapshot.history_end > reservation.history_limit or
                    snapshot.allocated_records > reservation.reserved_records):
                raise ValueError('Worker history exceeded its complete-turn reservation')
        del self.reservations[namespace]
        self.retiring.remove(namespace)
        if snapshot is None:
            del self.snapshots[namespace]
        else:
            self.snapshots.pop(namespace, None)
            self.snapshots[namespace] = snapshot
        self.settlements += 1

    def snapshot(self):
        incoming = sum(value.incoming_records + value.extra_records for value in self.reservations.values())
        idle = [value for key,value in self.snapshots.items() if key not in self.reservations]
        return dict(capacity_records=self.capacity_records, reserved_records=self.reserved_records,
            available_records=self.capacity_records-self.reserved_records,
            incoming_reserved_records=incoming, retained_reserved_records=self.reserved_records-incoming,
            active=len(self.reservations)-len(self.retiring), retiring=len(self.retiring),
            idle_protected_records=sum(value.protected_records for value in idle),
            idle_reclaimable_records=sum(value.allocated_records-value.protected_records for value in idle),
            admissions=self.admissions, deferrals=self.deferrals, settlements=self.settlements,
            **(dict(pool_records=self.pool_records) if self.pool_used else {}))

    def evicted(self, namespace, generation):
        """A generation-qualified worker ACK for an idle scope's destruction."""
        previous = self.snapshots.get(namespace)
        if namespace in self.reservations or previous is None or previous.generation != generation:
            raise ValueError('Stale or leased historical eviction')
        del self.snapshots[namespace]


class HistoricalHistoryAdmission:
    """Connector adapter; resolve the exact P/M prefix before reserving bytes."""
    matched_admission = True

    def __init__(self, capacity_bytes, record_bytes, repository_records, block_tokens,
                 sessions=None, *, lookahead_tokens=1):
        _extent(capacity_bytes, record_bytes)
        if not record_bytes:
            raise ValueError('Historical record size must be positive')
        self.record_bytes, self.sessions = record_bytes, sessions
        self.ledger = HistoricalAdmission(capacity_bytes//record_bytes, repository_records,
            block_tokens, lookahead_tokens=lookahead_tokens)
        self.requests, self.waiting = {}, set()
        self.observed_bytes = 0
        self._pressure_requests, self._sent_pressure = {}, {}
        self.history_evictions = 0
        # Shared prefixes: images released for a deficit and not yet acknowledged,
        # and pool payloads still draining. Neither counts as free capacity.
        self._released_images, self._pool_draining = set(), 0
        self.image_releases = 0

    def _proposal(self, request):
        from .coordination import HistoryAdmission
        from .session_control import SessionLedger
        from .request_plan import RequestCapacityError
        key = HistoryAdmission.key(request)
        entry = self.sessions.requests.get(request.request_id) if self.sessions is not None else None
        base = entry.history if entry is not None else 0
        snapshot = self.ledger.snapshots.get(key)
        if snapshot is None:
            snapshot = HistoricalCapacitySnapshot(entry.generation if entry else 1,
                entry.scope_history if entry else 0,
                entry.ready_records if entry and entry.scope_history else self.ledger.repository_records,
                entry.allocated_records if entry else 0, entry.protected_records if entry else 0)
        limit = request.num_prompt_tokens + SessionLedger.output_tokens(request)
        tail = base % self.ledger.block_tokens
        incoming = (tail+limit-base+self.ledger.block_tokens-1)//self.ledger.block_tokens-bool(tail)
        if incoming+self.ledger.extra_records > self.ledger.repository_records:
            raise RequestCapacityError('Complete incoming turn and in-flight KV reserve exceed repository capacity')
        if incoming+self.ledger.extra_records > self.ledger.capacity_records:
            raise RequestCapacityError('Complete incoming turn and in-flight KV reserve exceed CPU history capacity')
        return key, snapshot, base, limit

    def can_admit(self, request):
        if self.sessions is not None:
            from .coordination import HistoryAdmission
            key = HistoryAdmission.key(request)
            if key[0] == 'session' and request.request_id not in self.sessions.requests:
                # This read-only scheduling hint must allow the ledger to
                # resolve M/P fallback or schedule a necessary fenced eviction.
                entry = self.sessions.entries.get(key[1:])
                return entry is None or entry.phase == 'idle'
        key, snapshot, base, limit = self._proposal(request)
        if key in self.ledger.reservations:
            return False
        return self.ledger.preview(key, snapshot, request.request_id,
            base_history=base, history_limit=limit) is not None

    def allows(self, request):
        allowed = self.can_admit(request)
        if not allowed:
            self.ledger.deferrals += 1
            self.waiting.add(request.request_id)
            self._request_capacity(request)
        else:
            self._pressure_requests.pop(request.request_id, None)
            self._publish_pressure()
        return allowed

    def _request_capacity(self, request):
        if self.sessions is None:
            return
        key, snapshot, base, limit = self._proposal(request)
        tail = base % self.ledger.block_tokens
        incoming = (tail+limit-base+self.ledger.block_tokens-1)//self.ledger.block_tokens-bool(tail)
        required = incoming+self.ledger.extra_records
        demands = {}
        if key[0] == 'session' and snapshot.history_end and snapshot.ready_records < required:
            demands[key] = (snapshot.generation, required, None)
        unobserved = snapshot.allocated_records if key not in self.ledger.snapshots else 0
        deficit = max(0, self.ledger.reserved_records + unobserved + required - self.ledger.capacity_records)
        if deficit and self._release_unreferenced_images():
            # Shared prefixes: first release images no session references. Their
            # bytes are not counted free until the worker's pool notice shows them
            # gone; meanwhile nothing lossier (history reclaim, eviction) runs.
            self._pressure_requests[request.request_id] = demands
            self._publish_pressure()
            return
        # Scope snapshots retain retirement order: reclaim older idle histories
        # first. Active reservations never become supply for another turn.
        for victim, state in self.ledger.snapshots.items():
            if not deficit:
                break
            if victim[0] != 'session' or victim in self.ledger.reservations:
                continue
            reclaim = min(deficit, state.allocated_records-state.protected_records)
            if reclaim:
                previous = demands.get(victim, (state.generation, 0, None))
                demands[victim] = (state.generation, previous[1], state.allocated_records-reclaim)
                deficit -= reclaim
        if deficit and history_lru_eviction_enabled():
            self._evict_idle(key)
        self._pressure_requests[request.request_id] = demands
        self._publish_pressure()

    def _release_unreferenced_images(self):
        """True while an image release for a CPU deficit is still settling."""
        index = getattr(self.sessions, 'prefix_images', None) if self.sessions is not None else None
        if index is not None:
            released = index.unreferenced(self.sessions.entries)
            if released:
                self.sessions.commands.image_releases.extend(released)
                self._released_images.update(released)
                self.image_releases += len(released)
        return bool(self._released_images) or self._pool_draining > 0

    def _evict_idle(self, requester):
        """Opt-in KVMEM_SESSION_HISTORY_LRU_EVICTION: evict the least recently
        used idle session holding CPU history (never the requester's own). The
        ledger's insertion order is its idle LRU order; one fenced eviction at a
        time, since the waiting loop retries every step and the first eviction's
        records return only with its worker ACK."""
        entries = self.sessions.entries
        if any(entry.phase == 'evicting' for entry in entries.values()):
            return
        index = getattr(self.sessions, 'prefix_images', None)
        for namespace, entry in list(entries.items()):
            key = ('session', *namespace)
            state = self.ledger.snapshots.get(key)
            # A session holding no private record may still keep a shared-prefix
            # image referenced; evicting it can let that image's bytes go.
            if (key == requester or entry.phase != 'idle' or key in self.ledger.reservations or
                    state is None or not (state.allocated_records or
                                          (index is not None and index.references_entry(entry)))):
                continue
            self.sessions._evict(entry)
            self.history_evictions += 1
            return

    def _publish_pressure(self):
        if self.sessions is None:
            return
        from .session_control import HistoryCapacityRequest
        combined = {}
        for requests in self._pressure_requests.values():
            for key, (generation, ready, retained) in requests.items():
                if key in self.ledger.reservations:
                    continue
                previous = combined.get(key, (generation, 0, None))
                targets = [value for value in (previous[2], retained) if value is not None]
                combined[key] = (generation, max(previous[1], ready), min(targets) if targets else None)
        for key in self._sent_pressure.keys() | combined.keys():
            old = self._sent_pressure.get(key)
            target = combined.get(key)
            if old == target:
                continue
            generation, ready, retained = target or (old[0], 0, None)
            entry = self.sessions.entries.get(key[1:])
            if entry is not None and entry.generation == generation:
                self.sessions.commands.history_capacity.append(HistoryCapacityRequest(
                    key[1:], generation, ready, retained))
        self._sent_pressure = combined

    def allocated(self, request, generation=0):
        key, snapshot, base, limit = self._proposal(request)
        if request.request_id in self.requests or (key[0] == 'session' and snapshot.generation != generation):
            raise ValueError('Historical allocation has a stale scope identity')
        reservation = self.ledger.admit(key, snapshot, request.request_id,
            base_history=base, history_limit=limit)
        if reservation is None:
            raise RuntimeError('Native allocation exceeded its historical admission preview')
        self.requests[request.request_id] = key
        self.waiting.discard(request.request_id)
        self._pressure_requests.pop(request.request_id, None)
        self._publish_pressure()

    def finished(self, request):
        self.waiting.discard(request.request_id)
        self._pressure_requests.pop(request.request_id, None)
        self._publish_pressure()
        key = self.requests.get(request.request_id)
        if key is not None:
            reservation = self.ledger.reservations[key]
            self.ledger.finish(key, reservation.generation, request.request_id)

    def consume(self, notice):
        if notice.kind == 'history_pool':
            self.ledger.observe_pool(notice.records)
            self._pool_draining = notice.draining
            return
        if notice.kind == 'image_released':
            self._released_images.discard(notice.key)
            return
        key = ('session', *notice.namespace)
        if notice.kind == 'image_registered':
            self.ledger.transfer_to_pool(key, notice.generation, notice.transferred, notice.pool_records)
            return
        if notice.kind == 'imported':
            self.ledger.import_idle(key, HistoricalCapacitySnapshot(notice.generation, notice.history,
                notice.ready_records, notice.allocated_records, notice.protected_records))
            return
        if notice.kind == 'evicted':
            self.ledger.evicted(key, notice.generation)
        elif notice.kind == 'history_capacity':
            self.ledger.observe_idle(key, HistoricalCapacitySnapshot(notice.generation,
                notice.history, notice.ready_records, notice.allocated_records, notice.protected_records))
        elif notice.kind == 'retired':
            snapshot = (HistoricalCapacitySnapshot(notice.generation, notice.history,
                notice.ready_records, notice.allocated_records, notice.protected_records) if notice.history else None)
            self.ledger.settle(key, notice.generation, notice.request_id, snapshot)
            del self.requests[notice.request_id]

    def released(self, request_id):
        key = self.requests.get(request_id)
        if key is None:
            return
        if key[0] != 'request':
            raise ValueError('Missing session retirement before historical credit release')
        reservation = self.ledger.reservations[key]
        self.ledger.settle(key, reservation.generation, request_id)
        del self.requests[request_id]

    def observe(self, amount):
        _extent(amount)
        if amount > self.ledger.reserved_records*self.record_bytes:
            raise ValueError('CPU archive exceeded reserved historical capacity')
        self.observed_bytes = amount

    def snapshot(self):
        counters = self.ledger.snapshot()
        # Same per-owner view as HistoryAdmission.snapshot: clients use it as
        # the per-owner release ACK (an owner is released once it is absent).
        views = [dict(key=list(key), generation=value.generation, request_id=value.request_id,
                      phase='retiring' if key in self.ledger.retiring else 'active',
                      bytes=value.reserved_records*self.record_bytes)
                 for key, value in self.ledger.reservations.items()]
        shared = (dict(pool_draining_records=self._pool_draining, image_releases=self.image_releases,
                       images_releasing=len(self._released_images))
                  if self.ledger.pool_used or self.image_releases else {})
        return {**counters, 'capacity_bytes':self.ledger.capacity_records*self.record_bytes,
            'reserved_bytes':self.ledger.reserved_records*self.record_bytes,
            'available_bytes':counters['available_records']*self.record_bytes,
            'last_observed_archive_bytes':self.observed_bytes, 'waiting':len(self.waiting),
            'history_evictions':self.history_evictions, 'lease_views':views, **shared}
