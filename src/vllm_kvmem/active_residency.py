"""Revocable native attention-page leases after durable CPU retirement.

Only exact terminal M views qualify. The pool holds native block references,
not another KV allocation; on a hit their ownership moves to a new request.
CPU checkpoints remain authoritative, including recurrent and MTP model state.
TTL is checked on scheduler activity. Memory pressure can revoke idle leases
immediately, but never a lease already transferred to a live import.
"""
from collections import OrderedDict
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ActiveView:
    namespace: tuple
    generation: int
    history: int
    window: int
    groups: tuple
    expires: float
    protected_until: float = 0.

    @property
    def blocks(self):
        return tuple(block for _, blocks in self.groups for block in blocks)


class ActiveResidency:
    def __init__(self, managers, pool, ttl_ms, max_fraction, *, pressure_only=False,
                 clock=time.monotonic, tool_grace=None):
        self.managers, self.pool, self.clock = managers, pool, clock
        self.ttl = None if pressure_only else ttl_ms / 1000
        self.max_blocks = int((pool.num_gpu_blocks - 1) * max_fraction)
        self.entries = OrderedDict()
        self.hits = self.retained = self.expired = self.pressure_evictions = self.mismatches = 0
        self.reused_tokens = 0
        self.tool_grace = tool_grace

    def _drop(self, namespace, reason):
        view = self.entries.pop(namespace)
        self.pool.free_blocks(reversed(view.blocks))
        if reason == 'expired': self.expired += 1
        elif reason == 'pressure': self.pressure_evictions += 1
        elif reason == 'mismatch': self.mismatches += 1

    def expire(self):
        if self.ttl is None:
            return
        now = self.clock()
        for key, view in list(self.entries.items()):
            if view.expires <= now:
                self._drop(key, 'expired')

    def consume(self, notice):
        if notice.kind == 'evicted':
            view = self.entries.get(notice.namespace)
            if view is not None and view.generation == notice.generation:
                self._drop(notice.namespace, 'mismatch')
            return
        if notice.kind != 'retired':
            return
        self.expire()
        if notice.namespace in self.entries:
            self._drop(notice.namespace, 'mismatch')
        if not notice.resident_ready:
            return
        groups = []
        for group, manager in self.managers.items():
            count = (notice.window + manager.block_size - 1) // manager.block_size
            blocks = tuple(manager.req_to_blocks.get(notice.request_id, ())[:count])
            if (not count or len(blocks) != count or
                    manager.offsets.get(notice.request_id, 0) != notice.history - notice.window or
                    any(b.is_null or b.ref_cnt != 1 for b in blocks)):
                raise ValueError('Durable active view differs from terminal native allocation')
            groups.append((group, blocks))
        view = ActiveView(notice.namespace, notice.generation, notice.history, notice.window,
                          tuple(groups), float('inf') if self.ttl is None else self.clock() + self.ttl,
                          self.tool_grace.retired(notice.namespace, notice.generation) if self.tool_grace else 0.)
        if len(view.blocks) > self.max_blocks:
            return
        while self.entries and sum(len(v.blocks) for v in self.entries.values()) + len(view.blocks) > self.max_blocks:
            self._drop(next(iter(self.entries)), 'pressure')
        # The native terminal free drops its original reference after this ack.
        self.pool.touch(view.blocks)
        self.entries[view.namespace] = view
        self.retained += 1

    def match(self, entry):
        self.expire()
        if entry is None:
            return None
        view = self.entries.get(entry.namespace)
        if view is not None and getattr(entry, 'pressure_restore', False):
            # A pressure compaction changes the page set and compact positions.
            # Restore from CPU into a fresh allocation, never reuse old pages.
            self._drop(entry.namespace, 'mismatch')
            return None
        if view is not None and (view.generation, view.history, view.window) != (
                entry.generation, entry.history, entry.window):
            self._drop(entry.namespace, 'mismatch')
            return None
        return view

    def make_room(self, required_free, *, protect=None):
        entries = list(self.entries.items())
        if self.tool_grace is not None:
            # Cold admission was checked before the native allocation loop.
            # Running growth and fairness overrides may still revoke idle KV.
            entries.sort(key=lambda item: self.tool_grace.protected(item[1]))
        for key, view in entries:
            if self.pool.get_num_free_blocks() >= required_free:
                break
            if view is not protect:
                self._drop(key, 'pressure')

    def claim(self, view, request_id):
        if self.entries.get(view.namespace) is not view:
            raise ValueError('Stale resident active view')
        staged = []
        try:
            for group, blocks in view.groups:
                manager = self.managers[group]
                manager.stage_resident_import(request_id, view.history, view.window, blocks)
                staged.append(manager)
        except Exception:
            for manager in staged:
                manager.resident_imports.pop(request_id)
            raise
        del self.entries[view.namespace]

    def rollback(self, view, request_id):
        for group, blocks in view.groups:
            manager = self.managers[group]
            if manager.req_to_blocks.get(request_id) or manager.resident_imports.get(request_id) != blocks:
                raise RuntimeError('Cannot return an allocated resident import to the idle pool')
        for group, _ in view.groups:
            self.managers[group].resident_imports.pop(request_id)
        self.entries[view.namespace] = view

    def committed(self, view, request_id):
        for group, blocks in view.groups:
            manager = self.managers[group]
            if (request_id in manager.resident_imports or
                    tuple(manager.req_to_blocks.get(request_id, ())[:len(blocks)]) != blocks):
                raise RuntimeError('Resident native allocation did not adopt its held pages')
        self.hits += 1
        self.reused_tokens += view.window

    def snapshot(self):
        return dict(retention_policy='pressure' if self.ttl is None else 'ttl',
            idle_views=len(self.entries), idle_blocks=sum(len(v.blocks) for v in self.entries.values()),
            max_blocks=self.max_blocks, retained=self.retained, hits=self.hits, reused_tokens=self.reused_tokens,
            expired=self.expired, pressure_evictions=self.pressure_evictions, mismatches=self.mismatches,
            tool_grace=self.tool_grace.snapshot() if self.tool_grace else None)
