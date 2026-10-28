"""Bounded reuse of immutable packed model snapshots, never native state rows.

The CPU checkpoint is still authoritative. A cache entry is usable only with
the very same live snapshot object; replacement, eviction and model reload
cannot turn a position match into a cache hit. Prepared restores retain their
own tensor reference until their CUDA completion, independently of eviction.
"""
from collections import OrderedDict
import threading
import weakref


class ModelSnapshotCache:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Model snapshot cache requires a nonnegative byte capacity')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = 0
        self.lock = threading.RLock()

    def _drop(self, key):
        _, _, size = self.entries.pop(key)
        self.bytes -= size

    def put(self, snapshot, payload):
        size = payload.numel() * payload.element_size()
        if not self.capacity or size > self.capacity:
            return False
        if size != snapshot.bytes or not payload.is_contiguous():
            raise ValueError('Cached model payload differs from checkpoint schema')
        key = id(snapshot)
        owner = weakref.ref(self)

        def expired(reference):
            cache = owner()
            if cache is not None:
                with cache.lock:
                    entry = cache.entries.get(key)
                    if entry is not None and entry[0] is reference:
                        cache._drop(key)

        with self.lock:
            if key in self.entries:
                self._drop(key)
            while self.entries and self.bytes + size > self.capacity:
                self._drop(next(iter(self.entries)))
                self.evictions += 1
            self.entries[key] = (weakref.ref(snapshot, expired), payload, size)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True

    def get(self, snapshot):
        with self.lock:
            entry = self.entries.get(id(snapshot))
            if entry is None or entry[0]() is not snapshot:
                self.misses += 1
                return None
            self.entries.move_to_end(id(snapshot))
            self.hits += 1
            return entry[1]

    def stats(self):
        with self.lock:
            return dict(capacity_bytes=self.capacity, bytes=self.bytes,
                        peak_bytes=self.peak_bytes, entries=len(self.entries),
                        hits=self.hits, misses=self.misses, evictions=self.evictions)
