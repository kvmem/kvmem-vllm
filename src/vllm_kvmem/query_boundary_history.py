"""Bounded message boundaries for graph-free retrieval over sparse history.

This stores no attention edges or scores. Removing a KV block may remove its
otherwise unused message metadata, but never changes a surviving message's
original endpoints. Checkpoint snapshots remain independent immutable owners.
"""
from dataclasses import dataclass
from threading import RLock

from .block_graph import _merge_spans, _size
from .prompt_spans import normalize_provenance


@dataclass(frozen=True)
class _BoundaryData:
    message_spans: tuple = ()
    message_end_known: tuple = ()
    locality_provenance: tuple = ()
    has_metadata: bool = False
    accepted_through: int = 0


def _ranges(values):
    result = []
    for item in values:
        if (not isinstance(item, (tuple, list)) or len(item) != 2 or
                any(type(v) is not int for v in item) or not 0 <= item[0] < item[1]):
            raise ValueError('Live boundary ranges require original half-open positions')
        begin, end = item
        if result and begin < result[-1][0]:
            raise ValueError('Live boundary ranges must be ordered')
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def _pruned(spans, known, live_ranges, future_begin):
    ranges = _ranges(live_ranges)
    if future_begin is not None and (type(future_begin) is not int or future_begin < 0):
        raise ValueError('Invalid future message boundary')
    kept, ends, cursor = [], [], 0
    for span, end_known in zip(spans, known):
        begin, end, _ = span
        while cursor < len(ranges) and ranges[cursor][1] <= begin:
            cursor += 1
        intersects = cursor < len(ranges) and ranges[cursor][0] < end
        if intersects or future_begin is not None and end > future_begin:
            kept.append(span)
            ends.append(end_known)
    return tuple(kept), tuple(ends)


class QueryBoundarySnapshot:
    """Checkpoint metadata, deliberately distinct from graph proof/snapshots."""
    boundary_only = True

    def __init__(self, history, data=None):
        self._data = history._data if data is None else data
        self.owner, self._account = history.owner, history._account
        self.epoch, self.branch_id, self.revision = history.runtime_epoch, history.branch_id, history.revision
        self.block_tokens = history.block_tokens
        self.closed = False

    @property
    def bytes(self):
        return 0 if self.closed else _size(self._data) + 256

    @property
    def message_spans(self): return self._data.message_spans

    @property
    def message_end_known(self): return self._data.message_end_known

    @property
    def locality_provenance(self): return self._data.locality_provenance

    @property
    def has_metadata(self): return self._data.has_metadata

    @property
    def accepted_through(self): return self._data.accepted_through

    @property
    def binding(self):
        return self.owner, self.epoch, self.branch_id, self.revision, self.accepted_through

    def clone(self):
        if self.closed:
            raise ValueError('Boundary snapshot is closed')
        other = object.__new__(type(self))
        other.__dict__ = dict(self.__dict__)
        return other

    def close(self):
        self.closed = True
        self._data = _BoundaryData()

    def __enter__(self):
        if self.closed:
            raise ValueError('Boundary snapshot is closed')
        return self

    def __exit__(self, *_): self.close()


class QueryBoundaryHistory:
    def __init__(self, owner, block_tokens=32, *, max_spans=65536, initial_position=0):
        if (type(block_tokens) is not int or block_tokens <= 0 or
                type(max_spans) is not int or max_spans <= 0 or
                type(initial_position) is not int or initial_position < 0):
            raise ValueError('Invalid graph-free boundary capacity')
        self.owner, self.block_tokens, self.max_spans = owner, block_tokens, max_spans
        self._account = object()
        self._lock = RLock()
        self._data = _BoundaryData(accepted_through=initial_position)
        self.runtime_epoch = self.branch_id = self.revision = 0
        self._preserved = None
        self._preserved_until = self._validated_future_end = 0
        self.closed = False

    @property
    def message_spans(self): return self._data.message_spans

    @property
    def message_end_known(self): return self._data.message_end_known

    @property
    def locality_provenance(self): return self._data.locality_provenance

    @property
    def has_metadata(self): return self._data.has_metadata

    @property
    def accepted_through(self): return self._data.accepted_through

    @property
    def binding(self):
        return self.owner, self.runtime_epoch, self.branch_id, self.revision, self.accepted_through

    @property
    def bytes(self):
        return 0 if self.closed else _size(self._data) + 256 + (self._preserved.bytes if self._preserved else 0)

    def _open(self):
        if self.closed:
            raise ValueError('Boundary history is closed')

    def validate_binding(self, binding):
        with self._lock:
            self._open()
            if binding != self.binding:
                raise RuntimeError('Stale Query boundary revision')

    def _publish(self, data):
        if len(data.message_spans) > self.max_spans:
            raise BufferError('Graph-free message boundary capacity exceeded')
        if data != self._data:
            self._data = data
            self.revision += 1

    def update_message_spans(self, spans, validated_prefix=0, *, locality_provenance=None,
                             live_ranges=None, future_begin=None):
        with self._lock:
            self._open()
            merged, known = _merge_spans(self.message_spans, self.message_end_known, spans, validated_prefix)
            present = bool(merged) or self.has_metadata and validated_prefix > 0
            if live_ranges is not None:
                merged, known = _pruned(merged, known, live_ranges, future_begin)
            provenance = (self.locality_provenance if locality_provenance is None else
                          normalize_provenance(locality_provenance))
            self._publish(_BoundaryData(merged, known, provenance, present, self.accepted_through))

    def prune(self, live_ranges, *, future_begin=None):
        with self._lock:
            self._open()
            spans, known = _pruned(self.message_spans, self.message_end_known, live_ranges, future_begin)
            # Pruning only discards metadata for unavailable KV. It must not
            # invalidate an assembly already owning its selected live pages.
            # Exact endpoints for every surviving page remain unchanged.
            self._data = _BoundaryData(spans, known, self.locality_provenance,
                                      self.has_metadata, self.accepted_through)

    def advance(self, begin, end):
        with self._lock:
            self._open()
            if (type(begin) is not int or type(end) is not int or not 0 <= begin <= end or
                    begin > self.accepted_through or
                    begin < self.accepted_through and self._validated_future_end < self.accepted_through):
                raise ValueError('Boundary acceptance is not a continuous verified prefix')
            if end <= self.accepted_through:
                return
            self._publish(_BoundaryData(self.message_spans, self.message_end_known,
                self.locality_provenance, self.has_metadata, end))
            if self._preserved is not None and end > self._preserved_until:
                self._preserved.close()
                self._preserved = None
            self._validated_future_end = 0

    def snapshot(self, position=None):
        with self._lock:
            self._open()
            if position is not None and position != self.accepted_through:
                if self._preserved is not None and position == self._preserved.accepted_through:
                    return self._preserved.clone()
                raise BufferError('No exact boundary snapshot at requested accepted position')
            return QueryBoundarySnapshot(self)

    def _validate_snapshot(self, snapshot):
        if (not isinstance(snapshot, QueryBoundarySnapshot) or snapshot.closed or
                snapshot._account is not self._account or snapshot.owner != self.owner or
                snapshot.block_tokens != self.block_tokens):
            raise ValueError('Boundary snapshot is closed, foreign, or incompatible')

    def set_preserved_prefix(self, snapshot, until):
        with self._lock:
            self._open()
            self._validate_snapshot(snapshot)
            if type(until) is not int or not snapshot.accepted_through <= until <= self.accepted_through:
                raise ValueError('Invalid preserved boundary prefix')
            retained = snapshot.clone()
            if self._preserved is not None:
                self._preserved.close()
            self._preserved, self._preserved_until = retained, until

    def restore(self, snapshot, *, preserve_future=False, preserved_prefix=None, until=None):
        with self._lock:
            self._open()
            self._validate_snapshot(snapshot)
            if type(preserve_future) is not bool:
                raise ValueError('Boundary future preservation must be boolean')
            retained = None
            if preserved_prefix is not None:
                self._validate_snapshot(preserved_prefix)
                if (not preserve_future or type(until) is not int or
                        not preserved_prefix.accepted_through <= until <= snapshot.accepted_through):
                    raise ValueError('Invalid preserved boundary prefix restore')
                retained = preserved_prefix.clone()
            if len(snapshot.message_spans) > self.max_spans:
                raise BufferError('Restored boundary metadata exceeds capacity')
            if self._preserved is not None:
                self._preserved.close()
            self._data = snapshot._data
            self.runtime_epoch += 1
            self.branch_id = snapshot.branch_id if preserve_future else self.runtime_epoch
            self.revision += 1
            self._preserved, self._preserved_until = retained, until or 0
            self._validated_future_end = snapshot.accepted_through if preserve_future else 0

    def rewind(self, position):
        with self._lock:
            self._open()
            if type(position) is not int or not 0 <= position <= self.accepted_through:
                raise ValueError('Invalid boundary rewind position')
            spans, known = _merge_spans(self.message_spans, self.message_end_known, (), position)
            self._data = _BoundaryData(spans, known, self.locality_provenance, self.has_metadata, position)
            self.runtime_epoch += 1
            self.branch_id = self.runtime_epoch
            self.revision += 1
            self._validated_future_end = 0
            if self._preserved is not None:
                self._preserved.close()
                self._preserved = None

    def invalidate(self):
        with self._lock:
            self._open()
            self.runtime_epoch += 1
            self.revision += 1

    def close(self):
        with self._lock:
            if self._preserved is not None:
                self._preserved.close()
                self._preserved = None
            self.closed = True
            self._data = _BoundaryData()
