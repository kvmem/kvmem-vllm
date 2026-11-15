"""Opt-in KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD: fold layer 0 now, layers 1.. on first use.

Publishing a prefill span folds every layer's sampled attention masses into
that layer's cumulative history (one native ``committed_batch`` over all
layers). That is on the refresh critical path, but only layer 0 is read
before the next refresh's layer-0 selection; layer i is first read when
layer i is selected (lazily during the next forward with
KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST), or by a checkpoint, swap or restore.

``DeferredFoldHistory`` queues the exact fold arguments and applies them, in
order, before any read or write of its state, through the same native kernel
with the same per-layer decay. A history is therefore bit-identical to the
eager one whenever it is observed. A staged copy carries its own queue, so a
failed publication still cannot change committed state. Grouped forcing
(``force_all``) applies one queued span to a contiguous run of layers in one
batched native call, as the eager publication did.
"""
import math
import os
import threading

import numpy as np

from .buffered_prefill_attention import BufferedPrefillHistory, require_batch_accept
from ..core import settings as _settings

# Bounds the pinned score downloads a long run of unselected spans can retain.
MAX_PENDING = 64
_LOCK = threading.RLock()
_FIELDS = ('_buffered_state', 'end', 'samples', 'chunks', 'successful_buffered_commits',
           'successful_buffered_batch_commits', 'last_commit_backend')


def enabled():
    value = _settings.get('KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD must be 0 or 1')
    return value == '1'


class _Span:
    """One accepted span's fold arguments, shared by every deferred layer.

    ``buffer`` is a private C-contiguous copy of layers ``first``.. of the
    masses (same dtype, same values), so the span never retains the pinned
    action-download lease: that pool is bounded and never waits, and with
    several owners the queued spans of owners between refreshes exhausted it.
    The eager fold released the lease at publication; so does this.
    """
    __slots__ = ('end', 'rows', 'starts', 'local_begin', 'logical', 'buffer', 'first', 'shift')

    def __init__(self, end, rows, starts, local_begin, logical, masses, first=1, shift=0):
        self.end, self.rows, self.starts, self.local_begin = end, rows, starts, local_begin
        self.shift = shift
        self.logical, self.first = logical, first
        source = masses.numpy() if hasattr(masses, 'numpy') else np.asarray(masses)
        self.buffer = np.array(source[first:], copy=True, order='C')

    def layers(self, first, count):
        """Masses of layers first..first+count-1 (indices into the full span)."""
        if first < self.first:
            raise RuntimeError('Deferred span does not retain the requested layer')
        return self.buffer[first - self.first:first - self.first + count]


def _apply(histories, span, first):
    """Fold ``span`` into histories[j] as layer first+j (one native batch)."""
    states = [h.__dict__['_buffered_state'] for h in histories]
    decays = [math.exp2(-(span.end - h.__dict__['end']) / h.options.prefill_attention_half_life)
              for h in histories]
    count = len(histories)
    updated = require_batch_accept()(states, decays, span.rows, span.starts,
                                     list(span.logical[first:first+count]), span.local_begin,
                                     span.layers(first, count), span.shift)
    for history, state in zip(histories, updated):
        d = history.__dict__
        d['_buffered_state'] = state
        d['end'] = span.end
        d['samples'] += len(span.rows)
        d['chunks'] += 1
        d['successful_buffered_commits'] += 1
        d['successful_buffered_batch_commits'] += 1
        d['last_commit_backend'] = 'native_buffered'


def _field(name):
    def get(self):
        self.force_pending()
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def put(self, value):
        self.force_pending()
        self.__dict__[name] = value
    return property(get, put)


class DeferredFoldHistory(BufferedPrefillHistory):
    deferred_fold = True

    def __init__(self, options):
        self.__dict__['_pending'] = []
        super().__init__(options)

    def pending_end(self):
        """The end after every queued fold, without applying them."""
        with _LOCK:
            pending = self.__dict__['_pending']
            return pending[-1][0].end if pending else self.__dict__['end']

    def pending_count(self):
        return len(self.__dict__['_pending'])

    def defer(self, span, layer):
        with _LOCK:
            self.__dict__['_pending'].append((span, layer))

    def force_pending(self):
        pending = self.__dict__.get('_pending')
        if not pending:
            return
        with _LOCK:
            while pending:
                span, layer = pending[0]
                _apply([self], span, layer)
                pending.pop(0)

    def __copy__(self):
        with _LOCK:
            new = object.__new__(type(self))
            new.__dict__.update(self.__dict__)
            new.__dict__['_pending'] = list(self.__dict__['_pending'])
        return new

    def __getstate__(self):
        self.force_pending()
        state = dict(self.__dict__)
        state['_pending'] = []
        return state


for _name in _FIELDS:
    setattr(DeferredFoldHistory, _name, _field(_name))
del _name


def deferrable(histories, end, count, accepted, rows, logical, masses):
    """The eager batch path's own preconditions, evaluated without forcing."""
    return (len(histories) > 1 and accepted == count and bool(rows) and
            all(type(h) is DeferredFoldHistory for h in histories) and
            all(end > h.pending_end() for h in histories) and
            all(h.pending_count() < MAX_PENDING for h in histories) and
            len(logical) == len(histories) and getattr(masses, 'ndim', None) == 3 and
            masses.shape[0] == len(histories))


def accept_deferred(histories, end, count, accepted, rows, starts, logical, local_begin, masses):
    """Eagerly fold layer 0 (exactly as the batch did); queue every other layer."""
    from .buffered_prefill_attention import accept_buffered_batch
    span = _Span(end, tuple(rows), tuple(starts), local_begin, tuple(logical), masses,
                 shift=end - count - local_begin)
    if accept_buffered_batch(histories[:1], end, count, accepted, rows, starts,
                             logical[:1], local_begin, masses[:1]) != [True]:
        raise RuntimeError('Layer-zero attention fold was not accepted')
    # The eager batch validated every layer before any commit; keep that: the
    # same native checks (and errors) for the layers whose fold is deferred.
    from .. import _prefill_state_native
    validate = getattr(_prefill_state_native, 'validate_batch', None)
    if validate is None:
        raise RuntimeError('Deferred layer folds require the rebuilt prefill state extension')
    validate(span.rows, span.logical[1:], local_begin, span.layers(1, len(histories) - 1))
    for layer in range(1, len(histories)):
        histories[layer].defer(span, layer)
    return [True] * len(histories)


def force_all(histories):
    """Apply every queued fold, batching contiguous layers of the same span."""
    with _LOCK:
        lazy = [h for h in histories if type(h) is DeferredFoldHistory and h.__dict__['_pending']]
        while lazy:
            span = lazy[0].__dict__['_pending'][0][0]
            group = [h for h in lazy if h.__dict__['_pending'][0][0] is span]
            group.sort(key=lambda h: h.__dict__['_pending'][0][1])
            run = []
            for history in group + [None]:
                layer = None if history is None else history.__dict__['_pending'][0][1]
                if run and (history is None or layer != run[-1][1] + 1):
                    _apply([h for h, _ in run], span, run[0][1])
                    run = []
                if history is not None:
                    run.append((history, layer))
            for history in group:
                history.__dict__['_pending'].pop(0)
            lazy = [h for h in lazy if h.__dict__['_pending']]
