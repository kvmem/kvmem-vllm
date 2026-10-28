"""Accepted-position block dependencies, with atomic, bounded state publication.

This graph has no physical page identities.  Immutable Python containers are
charged at their in-memory size, conservatively charging shared payload again
for every live/staging/checkpoint/scoring owner.  Charges are consequently an
upper bound on retained graph memory, rather than serialized-size estimates.
"""
from dataclasses import dataclass, fields
import math
import operator
import sys
import threading
import weakref


MASS_TOLERANCE = .005
SNAPSHOT_VERSION = 1


def _size(value, seen=None):
    """Actual Python container footprint, including referenced numeric objects."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    result = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        result += sum(_size(v, seen) for v in value)
    elif isinstance(value, dict):
        result += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    elif hasattr(value, '__dataclass_fields__'):
        result += sum(_size(getattr(value, f.name), seen) for f in fields(value))
    return result


class GraphBudget:
    """Shared cap for all graph owners, including retained immutable snapshots."""
    def __init__(self, max_total_bytes=268435456):
        if type(max_total_bytes) is not int or max_total_bytes <= 0:
            raise ValueError('Graph global capacity must be a positive integer')
        self.maximum = max_total_bytes
        self.used = 0
        self._owners = {}
        self._lock = threading.RLock()

    def reserve(self, owner, amount, limit):
        if type(amount) is not int or amount < 0:
            raise ValueError('Invalid graph byte reservation')
        with self._lock:
            current = self._owners.get(owner, 0)
            if current + amount > limit or self.used + amount > self.maximum:
                raise BufferError('Block graph memory capacity exceeded before acceptance '
                    f'(owner_used={current}, requested={amount}, owner_limit={limit}, '
                    f'global_used={self.used}, global_limit={self.maximum})')
            self.used += amount
            self._owners[owner] = current + amount
        return _Charge(self, owner, amount)

    def owner_bytes(self, owner):
        with self._lock:
            return self._owners.get(owner, 0)

    def _release(self, owner, amount):
        with self._lock:
            self.used -= amount
            left = self._owners[owner] - amount
            if left:
                self._owners[owner] = left
            else:
                del self._owners[owner]


class _Charge:
    def __init__(self, budget, owner, amount):
        self.budget, self.owner, self.amount = budget, owner, amount

    def close(self):
        with self.budget._lock:
            amount, self.amount = self.amount, 0
            if amount:
                self.budget._release(self.owner, amount)

    def __del__(self):
        self.close()


@dataclass(frozen=True, slots=True)
class GraphSample:
    position: int
    targets: tuple
    masses: tuple
    kind: str = 'output'


@dataclass(frozen=True, slots=True)
class _DenseSamples:
    positions: tuple
    targets: tuple
    masses: object
    kind: str
    native: object
    indices: tuple | None = None
    output_state: tuple | None = None

    def __len__(self):
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class GraphRow:
    source: int
    targets: tuple
    weights: tuple
    samples: int
    prefill_samples: int = 0
    output_samples: int = 0

    @property
    def retained_mass(self):
        return math.fsum(self.weights)


@dataclass(frozen=True, slots=True)
class GraphPartial:
    source: int
    targets: tuple
    sums: tuple
    samples: int
    prefill_samples: int = 0
    output_samples: int = 0


@dataclass(frozen=True, slots=True)
class _Data:
    rows: tuple = ()
    partial: GraphPartial | None = None
    accepted_through: int = 0
    completed_through: int = 0
    message_spans: tuple = ()
    message_end_known: tuple = ()
    rows_bytes: int = 0
    metadata_bytes: int = 0
    locality_provenance: tuple = ()


def _graph_size(data):
    # Row payloads are immutable and their conservative retained size is
    # accumulated only when a source seals; ordinary decode is O(partial).
    return (sys.getsizeof(data) + sys.getsizeof(data.rows) + data.rows_bytes +
            _size(data.partial) + data.metadata_bytes + 1024)


class GraphSnapshot:
    """An independently charged lease over one immutable graph revision."""
    version = SNAPSHOT_VERSION

    def __init__(self, graph, data=None):
        self._data = graph._data if data is None else data
        self._graph_owner = weakref.ref(graph)
        self.owner = graph.owner
        self._account = graph._account
        self.block_tokens = graph.block_tokens
        self.branch_id = graph.branch_id
        self.epoch = graph.runtime_epoch
        self.revision = graph.revision
        self._budget = graph.budget
        self._limit = graph.max_bytes_per_session
        self.bytes = _graph_size(self._data)
        self._charge = self._budget.reserve(self._account, self.bytes, self._limit)

    @property
    def runtime_epoch(self):
        return self.epoch

    @property
    def rows(self):
        return self._data.rows

    @property
    def partial(self):
        return self._data.partial

    @property
    def accepted_through(self):
        return self._data.accepted_through

    @property
    def completed_through(self):
        return self._data.completed_through

    @property
    def message_spans(self):
        return self._data.message_spans

    @property
    def message_end_known(self):
        return self._data.message_end_known

    @property
    def locality_provenance(self):
        return self._data.locality_provenance

    @property
    def binding(self):
        return (self.owner, self.epoch, self.branch_id, self.revision,
                self.accepted_through)

    @property
    def closed(self):
        return self._charge.amount == 0

    def reserve_scoring_scratch(self, amount):
        """Charge temporary maps/vectors and their returned score/trace owner.

        The caller transfers this lease into GraphScores. It is independent of
        the snapshot lease so a retained result remains charged after snapshot
        close, until the result is explicitly closed or garbage collected.
        """
        if self.closed:
            raise RuntimeError('Cannot score a released graph snapshot')
        return self._budget.reserve(self._account, amount, self._limit)

    def clone(self):
        if self.closed:
            raise RuntimeError('Cannot retain a released graph snapshot')
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result._charge = self._budget.reserve(self._account, self.bytes, self._limit)
        return result

    def close(self):
        cache=getattr(self,'_native_scoring_cache',None)
        if cache is not None:
            cache.close()
        self._charge.close()
        self._data = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class GraphUpdate:
    """A fully reserved candidate; detachable across CPU publication workers.

    prepare holds the owner lock. detach releases only the lock, retaining its
    charge and one pending-owner marker. resume validates the immutable binding
    before any parser/history commit, including after a source-copy handoff.
    """
    def __init__(self, graph, data=None, charge=None, final_bytes=0):
        self.graph, self.data, self.charge = graph, data, charge
        self.final_bytes = final_bytes
        self.done = False
        self._locked = True
        self._thread = threading.get_ident()
        self._binding = graph.binding
        graph._pending_update = self

    def detach(self):
        if self.done:
            raise RuntimeError('Cannot detach a finished graph reservation')
        if self._locked:
            if self._thread != threading.get_ident():
                raise RuntimeError('Graph reservation lock belongs to another thread')
            self._locked = False
            self.graph._lock.release()
        return self

    def resume(self):
        if self.done:
            raise RuntimeError('Graph reservation has already finished')
        if self._locked:
            if self._thread != threading.get_ident():
                raise RuntimeError('Graph reservation was not detached before thread handoff')
        else:
            self.graph._lock.acquire()
            self._locked = True
            self._thread = threading.get_ident()
        try:
            self.graph.validate_binding(self._binding)
            if self.graph._pending_update is not self:
                raise RuntimeError('Graph reservation no longer owns this acceptance')
        except BaseException:
            self.detach()
            raise
        return self

    def commit(self):
        if self.done:
            return
        self.resume()
        graph = self.graph
        try:
            if self.data is not None:
                old = graph._charge
                graph._data, graph._charge = self.data, self.charge
                graph.revision += 1
                old.close()
                graph._charge.budget._release(graph._charge.owner, graph._charge.amount - self.final_bytes)
                graph._charge.amount = self.final_bytes
                self.charge = None
                self.data = None
                if (graph._preserved is not None and
                        graph.accepted_through > graph._preserved_until):
                    graph._preserved.close()
                    graph._preserved = None
        finally:
            self.__dict__.pop('output_state',None)
            graph._pending_update = None
            self.done = True
            self._locked = False
            graph._lock.release()

    def abort(self):
        if self.done:
            return
        if not self._locked:
            self.graph._lock.acquire()
            self._locked = True
            self._thread = threading.get_ident()
        elif self._thread != threading.get_ident():
            raise RuntimeError('Graph reservation abort crossed a held thread lock')
        try:
            self.__dict__.pop('output_state',None)
            if self.charge is not None:
                self.charge.close()
            self.data = None
            self.charge = None
            if self.graph._pending_update is self:
                self.graph._pending_update = None
            self.done = True
        finally:
            self._locked = False
            self.graph._lock.release()

    close = abort

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.abort()


def _spans(value):
    result = []
    previous = 0
    for item in value:
        if isinstance(item, dict):
            begin, end, role = item['begin'], item['end'], item['role']
        else:
            begin, end, role = item
        if (type(begin) is not int or type(end) is not int or
                not 0 <= begin < end or begin < previous or not isinstance(role, str)):
            raise ValueError('Invalid graph message spans')
        result.append((begin, end, role))
        previous = end
    return tuple(result)


def _merge_spans(old, old_known, supplied, prefix):
    incoming = _spans(supplied)
    if type(prefix) is not int or prefix < 0:
        raise ValueError('Invalid validated message prefix')
    retained = [(begin, min(end, prefix), role, known and end <= prefix)
                for (begin, end, role), known in zip(old, old_known) if begin < prefix]
    # Both inputs are ordered, nonoverlapping message spans. Only intersecting
    # spans can contradict the proven prefix; scanning all pairs made every
    # long Agent handoff quadratic in its number of messages.
    cursor = 0
    for begin, end, role, known in retained:
        while cursor < len(incoming) and incoming[cursor][1] <= begin:
            cursor += 1
        index = cursor
        while index < len(incoming) and incoming[index][0] < end:
            new_begin, new_end, new_role = incoming[index]
            if max(begin, new_begin) < min(end, new_end, prefix):
                if (begin != new_begin or role != new_role or
                        new_end < end or (known and new_end != end)):
                    raise ValueError('Conflicting message boundaries in validated historical prefix')
            index += 1
    # A newly supplied complete span can extend an inherited unknown end.
    # A valid nonoverlapping list has at most one span for each (begin, role).
    # Index that identity directly instead of repeatedly scanning the result.
    result = {(begin, role): ((begin, end, role), known) for begin, end, role, known in retained}
    for span in incoming:
        result[(span[0], span[2])] = (span, True)
    merged = sorted(result.values())
    ordered = tuple(span for span, _ in merged)
    _spans(ordered)
    return ordered, tuple(known for _, known in merged)


class BlockGraph:
    def __init__(self, owner, block_tokens, *, branch_id=None, topk=4, min_mass=0.,
                 max_bytes_per_session=33554432, budget=None, initial_position=0):
        if (type(block_tokens) is not int or block_tokens <= 0 or type(topk) is not int or
                topk <= 0 or not math.isfinite(min_mass) or min_mass < 0 or
                type(initial_position) is not int or initial_position < 0 or
                type(max_bytes_per_session) is not int or max_bytes_per_session <= 0):
            raise ValueError('Invalid block graph geometry or capacity')
        self.owner, self.block_tokens = owner, block_tokens
        self.topk, self.min_mass = topk, min_mass
        self.max_bytes_per_session = max_bytes_per_session
        self.budget = budget if budget is not None else GraphBudget()
        self._account = object()  # Equal namespace strings never share mutable accounting.
        self._lock = threading.RLock()
        self.branch_id = 0 if branch_id is None else branch_id
        self.runtime_epoch, self.revision = 0, 0
        self._data = _Data(accepted_through=initial_position,
                           completed_through=initial_position // block_tokens * block_tokens)
        self._charge = self.budget.reserve(self._account, _graph_size(self._data),
                                           max_bytes_per_session)
        self._preserved = None
        self._preserved_until = 0
        self.closed = False
        self._pending_update = None

    @property
    def accepted_through(self):
        return self._data.accepted_through

    @property
    def completed_through(self):
        return self._data.completed_through

    @property
    def binding(self):
        return (self.owner, self.runtime_epoch, self.branch_id, self.revision,
                self.accepted_through)

    @property
    def live_bytes(self):
        return self._charge.amount

    @property
    def bytes(self):
        return self.budget.owner_bytes(self._account)

    @property
    def rows(self):
        return self._data.rows

    @property
    def partial(self):
        return self._data.partial

    @property
    def message_spans(self):
        return self._data.message_spans

    @property
    def message_end_known(self):
        return self._data.message_end_known

    @property
    def locality_provenance(self):
        return self._data.locality_provenance

    def _open(self):
        if self.closed:
            raise RuntimeError('Block graph is released')

    def validate_binding(self, binding):
        with self._lock:
            self._open()
            if self.binding != binding:
                raise RuntimeError('Stale block graph selection epoch or accepted revision')

    def invalidate(self):
        with self._lock:
            self._open()
            self.runtime_epoch += 1

    def prepare_accept(self, begin, end, source_positions=(), target_starts=(),
                       mass_rows=(), kind='output', *, row_indices=None, output_state=None):
        if len(source_positions) != (len(mass_rows) if row_indices is None else len(row_indices)):
            raise ValueError('Graph source/mass row shape mismatch')
        targets = tuple(target_starts)
        # Captured CPU matrices already own this buffer. Keep it dense through
        # validation/aggregation, allocating graph output only after admission.
        # Sequence callers remain the scalar reference implementation.
        try:
            view = memoryview(mass_rows)
        except TypeError:
            view = None
        if view is not None and view.ndim == 2 and view.format in ('f', 'd'):
            from . import _native
            native = getattr(_native, 'prepare_block_graph', None)
            if native is not None:
                if view.shape[1] != len(targets) or (row_indices is None and view.shape[0] != len(source_positions)):
                    raise ValueError('Graph attention matrix shape mismatch')
                return self.prepare(begin, end, _DenseSamples(
                    tuple(operator.index(p) for p in source_positions), targets, view, kind, native,
                    None if row_indices is None else tuple(row_indices), output_state))
        if row_indices is not None or output_state is not None:
            raise ValueError('Indexed output acceptance requires a dense float matrix')
        return self.prepare(begin, end, tuple(GraphSample(operator.index(position), targets, values, kind)
                            for position, values in zip(source_positions, mass_rows)))

    def prepare(self, accepted_begin, accepted_end, rows=(), *, message_spans=None,
                verified_prefix=None, locality_provenance=None):
        self._lock.acquire()
        charge = None
        try:
            self._open()
            if self._pending_update is not None:
                raise BlockingIOError('Graph owner already has a reserved acceptance')
            if (type(accepted_begin) is not int or type(accepted_end) is not int or
                    not 0 <= accepted_begin <= accepted_end or accepted_begin > self.accepted_through):
                raise ValueError('Graph accepted progress has a gap or invalid interval')
            # Pre-reserve temporary containers and a conservative new-state bound.
            # Includes normalized row dicts and a full-repository partial scratch.
            dense = isinstance(rows, _DenseSamples)
            count = len(rows) * len(rows.targets) if dense else sum(len(row.targets) for row in rows)
            maximum_row = len(rows.targets) if dense else max((len(row.targets) for row in rows), default=0)
            target_bound = min(count, (accepted_end + self.block_tokens - 1) // self.block_tokens)
            scratch = 2048 + (maximum_row + target_bound) * 320 + len(rows) * 512
            if self.partial is not None:
                scratch += len(self.partial.targets) * 320
            if dense and rows.output_state is not None:
                scratch += 4096 + (len(rows.output_state[0]) + len(rows.targets)) * 320
            if message_spans is not None:
                scratch += (len(message_spans) + len(self.message_spans)) * 512
            if locality_provenance is not None:
                from .prompt_spans import normalize_provenance
                locality_provenance = normalize_provenance(locality_provenance)
                scratch += _size(locality_provenance) * 2
            metadata = None
            if not len(rows) and accepted_end <= self.accepted_through:
                # A refresh reinstalls the same metadata on every boundary.
                # Validate/merge it under its own small reservation before
                # charging full graph staging. Frozen-P owners retain graph
                # snapshots too; charging two graph copies for an unchanged
                # empty span list could exhaust their capacity unnecessarily.
                metadata_scratch = 2048
                if message_spans is not None:
                    metadata_scratch += (len(message_spans) + len(self.message_spans)) * 512
                if locality_provenance is not None:
                    metadata_scratch += _size(locality_provenance) * 2
                charge = self.budget.reserve(self._account, metadata_scratch, self.max_bytes_per_session)
                spans, ends_known = (self.message_spans, self.message_end_known) if message_spans is None else _merge_spans(
                    self.message_spans, self.message_end_known, message_spans,
                    self.accepted_through if verified_prefix is None else verified_prefix)
                provenance = self.locality_provenance if locality_provenance is None else locality_provenance
                metadata = spans, ends_known, provenance
                charge.close()
                charge = None
                if (spans == self.message_spans and ends_known == self.message_end_known and
                        provenance == self.locality_provenance):
                    return GraphUpdate(self)
            # The existing live revision already owns its reservation. Staging
            # retains one new immutable graph (including a conservative charge
            # for shared rows); scratch covers merge/row/output temporaries.
            # Reserving two *additional* full graphs here charges a nonexistent
            # copy and can reject long sessions with valid retained snapshots.
            upper = self.live_bytes + scratch
            charge = self.budget.reserve(self._account, upper, self.max_bytes_per_session)
            seen = set()
            for row in (() if dense else rows):
                if (type(row.position) is not int or not accepted_begin <= row.position < accepted_end or
                        row.kind not in ('prefill', 'output') or len(row.targets) != len(row.masses) or
                        not row.targets or row.position in seen):
                    raise ValueError('Invalid or duplicate graph sampled source row')
                seen.add(row.position)
                if any(type(p) is not int or p < 0 or p % self.block_tokens for p in row.targets):
                    raise ValueError('Graph targets must be logical block starts')
                if any(not math.isfinite(v) or v < 0 for v in row.masses):
                    raise ValueError('Graph attention mass must be finite and nonnegative')
                total = math.fsum(row.masses)
                if abs(total - 1) > MASS_TOLERANCE:
                    raise ValueError('Graph attention row does not sum to one')
                if row.position < self.accepted_through:
                    continue
                if any(target > row.position and mass > 0 for target, mass in zip(row.targets, row.masses)):
                    raise ValueError('Graph sampled mass points to a future token block')
            dense_result = None
            if dense:
                if (rows.kind not in ('prefill', 'output') or
                        len(set(rows.positions)) != len(rows.positions) or
                        any(not accepted_begin <= p < accepted_end for p in rows.positions) or
                        any(type(p) is not int or p < 0 or p % self.block_tokens for p in rows.targets)):
                    raise ValueError('Invalid dense graph sampled source/target rows')
                dense_result = rows.native(rows.positions, rows.targets, rows.masses,
                    self.accepted_through, max(self.accepted_through, accepted_end),
                    self.block_tokens, self.topk, self.min_mass, rows.kind == 'prefill', self.partial,
                    *(() if rows.indices is None else (rows.indices,)))
                if rows.output_state is not None:
                    from . import _native
                    old, samples, maximum = rows.output_state
                    staged_output = _native.prepare_decode_attention(old, samples, rows.indices,
                        rows.targets, rows.masses, maximum)
            if metadata is None:
                spans, ends_known = (self.message_spans, self.message_end_known) if message_spans is None else _merge_spans(
                    self.message_spans, self.message_end_known, message_spans,
                    self.accepted_through if verified_prefix is None else verified_prefix)
                provenance = self.locality_provenance if locality_provenance is None else locality_provenance
            else:
                spans, ends_known, provenance = metadata
            end = max(self.accepted_through, accepted_end)
            if (end == self.accepted_through and spans == self.message_spans and
                    ends_known == self.message_end_known and provenance == self.locality_provenance):
                charge.close()
                return GraphUpdate(self)
            completed = []
            added_bytes = 0
            partial = None if dense else self.partial
            source = partial.source if partial is not None else None
            totals = dict(zip(partial.targets, partial.sums)) if partial is not None else {}
            samples = partial.samples if partial is not None else 0
            prefill = partial.prefill_samples if partial is not None else 0
            output = partial.output_samples if partial is not None else 0

            def finish():
                nonlocal source, totals, samples, prefill, output, added_bytes
                if samples:
                    candidates = [(target, mass / samples) for target, mass in totals.items()
                                  if target < source and mass > 0 and mass / samples >= self.min_mass]
                    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
                    selected = candidates[:self.topk]
                    sealed = GraphRow(source, tuple(v[0] for v in selected),
                                      tuple(v[1] for v in selected), samples, prefill, output)
                    completed.append(sealed)
                    added_bytes += _size(sealed)
                source, totals, samples, prefill, output = None, {}, 0, 0, 0

            for row in (() if dense else sorted(rows, key=lambda row: row.position)):
                position, kind = row.position, row.kind
                if position < self.accepted_through:
                    continue
                total = math.fsum(row.masses)
                values = {}
                for target, mass in zip(row.targets, row.masses):
                    if mass:
                        values[target] = values.get(target, 0.) + float(mass) / total
                current = position // self.block_tokens * self.block_tokens
                if source is not None and source != current:
                    finish()
                if source is None:
                    source = current
                for target, mass in values.items():
                    totals[target] = totals.get(target, 0.) + mass
                samples += 1
                prefill += kind == 'prefill'
                output += kind == 'output'
            if source is not None and source + self.block_tokens <= end:
                finish()
            pending = None
            if source is not None:
                entries = sorted(totals.items())
                pending = GraphPartial(source, tuple(v[0] for v in entries), tuple(v[1] for v in entries),
                                       samples, prefill, output)
            if dense:
                completed = [GraphRow(*row) for row in dense_result[0]]
                pending = GraphPartial(*dense_result[1]) if dense_result[1] is not None else None
                added_bytes = sum(_size(row) for row in completed)
            stored = self.rows + tuple(completed) if completed else self.rows
            metadata_bytes = self._data.metadata_bytes if (spans is self.message_spans and ends_known is self.message_end_known and provenance is self.locality_provenance) else _size((spans, ends_known, provenance))
            data = _Data(stored, pending, end, end // self.block_tokens * self.block_tokens, spans, ends_known,
                         self._data.rows_bytes + added_bytes, metadata_bytes, provenance)
            size = _graph_size(data)
            if size > upper:
                raise BufferError('Graph staging estimate exceeded before acceptance')
            # Keep scratch charged until commit, after this frame and its row
            # temporaries have gone; publication compacts the live reservation.
            update = GraphUpdate(self, data, charge, size)
            if dense and rows.output_state is not None:
                update.output_state = staged_output
            return update
        except BaseException:
            if charge is not None:
                charge.close()
            self._lock.release()
            raise

    def update_message_spans(self, spans, validated_prefix=0, *, locality_provenance=None):
        with self.prepare(self.accepted_through, self.accepted_through,
                          message_spans=spans, verified_prefix=validated_prefix,
                          locality_provenance=locality_provenance) as update:
            update.commit()

    def snapshot(self, position=None):
        with self._lock:
            self._open()
            if position is not None and position < self.accepted_through:
                if self._preserved is not None and position == self._preserved.accepted_through:
                    return self._preserved.clone()
                raise BufferError('No exact graph snapshot for an earlier accepted boundary')
            if position is not None and position != self.accepted_through:
                raise BufferError('Graph accepted events have not reached checkpoint boundary')
            return GraphSnapshot(self)

    def set_preserved_prefix(self, snapshot, until):
        with self._lock:
            self._open()
            if snapshot._account is not self._account or snapshot.owner != self.owner or not snapshot.accepted_through <= until <= self.accepted_through:
                raise ValueError('Invalid preserved graph prefix')
            retained = snapshot.clone()
            if self._preserved is not None:
                self._preserved.close()
            self._preserved, self._preserved_until = retained, until

    def restore(self, snapshot, *, preserve_future=False, preserved_prefix=None, until=None):
        with self._lock:
            self._open()
            if self._pending_update is not None:
                raise BlockingIOError('Graph restore must wait for its reserved acceptance')
            if (snapshot.closed or snapshot._account is not self._account or
                    snapshot.owner != self.owner or snapshot.block_tokens != self.block_tokens):
                raise ValueError('Graph snapshot is released, foreign, or has incompatible geometry')
            retained = None
            if preserved_prefix is not None:
                if (not preserve_future or preserved_prefix.closed or
                        preserved_prefix._account is not self._account or
                        until is None or not preserved_prefix.accepted_through <= until <= snapshot.accepted_through):
                    raise ValueError('Invalid preserved graph prefix restore')
                retained = preserved_prefix.clone()
            data = snapshot._data
            try:
                charge = self.budget.reserve(self._account, _graph_size(data), self.max_bytes_per_session)
            except BaseException:
                if retained is not None:
                    retained.close()
                raise
            self._charge.close()
            self._charge, self._data = charge, data
            self.runtime_epoch += 1
            # A restored branch is a new runtime identity even if its historical
            # prefix is shared. Epoch monotonically invalidates all old readers.
            self.branch_id = snapshot.branch_id if preserve_future else self.runtime_epoch
            self.revision += 1
            if self._preserved is not None:
                self._preserved.close()
            self._preserved = retained
            self._preserved_until = 0 if until is None else until

    def stats(self):
        return {'revision': self.revision, 'runtime_epoch': self.runtime_epoch,
                'locality_provenance': dict(self.locality_provenance),
                'accepted_through': self.accepted_through, 'completed_through': self.completed_through,
                'observed_sources': len(self.rows), 'edges': sum(len(row.targets) for row in self.rows),
                'samples': sum(row.samples for row in self.rows) + (self.partial.samples if self.partial else 0),
                'prefill_samples': sum(row.prefill_samples for row in self.rows) + (self.partial.prefill_samples if self.partial else 0),
                'output_samples': sum(row.output_samples for row in self.rows) + (self.partial.output_samples if self.partial else 0),
                'retained_mass': sum(row.retained_mass for row in self.rows),
                'no_sample_sources': self.completed_through // self.block_tokens - len(self.rows),
                'partial_source': None if self.partial is None else self.partial.source,
                'partial_samples': 0 if self.partial is None else self.partial.samples,
                'live_bytes': self.live_bytes, 'snapshot_staging_bytes': self.bytes - self.live_bytes,
                'total_bytes': self.bytes, 'global_bytes': self.budget.used}

    def close(self):
        with self._lock:
            if not self.closed:
                if self._pending_update is not None:
                    self._pending_update.abort()
                self.runtime_epoch += 1
                self.closed = True
                cache=getattr(self,'_native_scoring_cache',None)
                if cache is not None:
                    cache.close()
                self._charge.close()
                if self._preserved is not None:
                    self._preserved.close()
                    self._preserved = None
                self._data = _Data()
