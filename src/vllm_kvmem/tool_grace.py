"""Bounded idle-KV grace learned from agent return gaps, not queue time.

The server observes completion -> arrival, which includes tool and client work.
This is a scheduling hint only: it never owns pages or changes session identity.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import time


@dataclass
class _Arrival:
    namespace: tuple | None
    at: float
    deferred: bool = False
    yielded: bool = False


class ToolGrace:
    def __init__(self, max_grace_ms, max_wait_ms, *, clock=time.monotonic, output=None):
        if not (math.isfinite(max_grace_ms) and math.isfinite(max_wait_ms) and
                0 < max_grace_ms <= max_wait_ms):
            raise ValueError('Tool grace must be positive, finite and no longer than maximum wait')
        self.max_grace = max_grace_ms / 1000
        self.max_wait = max_wait_ms / 1000
        self.clock, self.output = clock, output
        self.arrivals = {}
        self.sessions = OrderedDict()
        self.restore_cost = 1.0
        self.samples = self.deferrals = self.yields = self.restores = 0

    def _emit(self, event, **fields):
        if self.output is not None:
            with self.output.open('a') as f:
                f.write(json.dumps(dict(event=event, at=self.clock(), **fields)) + '\n')

    def _session(self, namespace):
        if namespace not in self.sessions:
            self.sessions[namespace] = dict(finished=None, mean=None, deviation=0.)
        self.sessions.move_to_end(namespace)
        while len(self.sessions) > 1024:
            self.sessions.popitem(last=False)
        return self.sessions[namespace]

    def arrival(self, request_id, namespace):
        if request_id in self.arrivals:
            return
        now = self.clock()
        self.arrivals[request_id] = _Arrival(namespace, now)
        if namespace is None:
            return
        state = self._session(namespace)
        finished = state['finished']
        state['finished'] = None  # Sample once, before native queue admission.
        if finished is not None:
            gap = max(0., now - finished)
            mean = state['mean']
            state['mean'] = gap if mean is None else .75 * mean + .25 * gap
            state['deviation'] = 0. if mean is None else .75 * state['deviation'] + .25 * abs(gap - mean)
            self.samples += 1
            self._emit('return', request_id=request_id, namespace=namespace,
                       return_gap_seconds=gap, predicted_seconds=state['mean'])

    def finished(self, request_id, *, successful):
        arrival = self.arrivals.pop(request_id, None)
        if arrival is not None and arrival.namespace is not None and successful:
            self._session(arrival.namespace)['finished'] = self.clock()

    def grace(self, namespace):
        state = self._session(namespace)
        predicted = (.25 if state['mean'] is None else
                     state['mean'] + 2 * state['deviation'] + .05)
        # Waiting longer than a cold restore cannot amortize the next swap.
        limit = min(self.max_grace, self.restore_cost)
        return min(limit, max(.05, predicted)) if predicted <= limit else 0.

    def retired(self, namespace, generation):
        seconds = self.grace(namespace)
        self._emit('retain', namespace=namespace, generation=generation,
                   grace_seconds=seconds, cold_restore_seconds=self.restore_cost)
        return self.clock() + seconds

    def protected(self, view):
        return self.clock() < view.protected_until

    def defer(self, request_id):
        arrival = self.arrivals.get(request_id)
        # Missing arrival evidence must not create an unbounded wait.
        if arrival is None:
            return False
        waited = self.clock() - arrival.at
        if waited >= self.max_wait:
            if not arrival.yielded:
                arrival.yielded = True
                self.yields += 1
                self._emit('queue_yield', request_id=request_id, waited_seconds=waited)
            return False
        if not arrival.deferred:
            arrival.deferred = True
            self.deferrals += 1
            self._emit('defer', request_id=request_id, waited_seconds=waited)
        return True

    def observe_restore(self, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.restore_cost = .8 * self.restore_cost + .2 * min(10., seconds)
            self.restores += 1

    def snapshot(self):
        return dict(return_samples=self.samples, deferred_requests=self.deferrals,
                    queue_yields=self.yields, cold_restore_samples=self.restores,
                    cold_restore_seconds=self.restore_cost, max_grace_seconds=self.max_grace,
                    max_wait_seconds=self.max_wait, tracked_sessions=len(self.sessions),
                    tracked_requests=len(self.arrivals))
