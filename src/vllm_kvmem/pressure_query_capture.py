"""Accepted public attention-Q tail for the fixed-method recall experiment.

Q is copied from the ordinary model forward in its original content frame.
Only scheduler-confirmed rows are admitted. The retained interval is explicit:
missing rows, stale branch epochs, replayed rows and future reads fail closed.
"""
import os
from threading import RLock

import torch


METHODS = ('mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass')
PRESSURE_QUERY_WINDOW = 128


def configured_method():
    method = os.getenv('KVMEM_RECALL_METHOD', '')
    if not method:
        return None
    if method not in METHODS:
        raise ValueError('KVMEM_RECALL_METHOD must be ' + ', '.join(METHODS))
    if os.getenv('KVMEM_PRESSURE_QUERY_WINDOW', '128') != '128':
        raise ValueError('This recall experiment fixes KVMEM_PRESSURE_QUERY_WINDOW=128')
    return method


class PressureQueryWindow:
    """A bounded CPU copy of the latest contiguous accepted Q interval."""

    def __init__(self, window=PRESSURE_QUERY_WINDOW):
        if type(window) is not int or window <= 0:
            raise ValueError('Pressure Query window must be positive')
        self.window = window
        self.begin = self.end = 0
        self.rows = None
        self.epoch = 0
        self.closed = False
        self.last_stage = None
        self._lock = RLock()

    def validate_epoch(self, epoch):
        with self._lock:
            if self.closed or epoch != self.epoch:
                raise RuntimeError('Stale or closed accepted pressure Query capture')

    def accept(self, history_begin, accepted, rows, *, row_offset=0,
               epoch=None, prefill=False):
        """Commit the retained public prefix, never rejected speculative rows.

        A long prefill transports its scheduled suffix only. If terminal
        trimming removes that suffix, the interval stays explicitly incomplete;
        a later read cannot substitute an older or shorter observation window.
        """
        with self._lock:
            self.validate_epoch(self.epoch if epoch is None else epoch)
            if (type(history_begin) is not int or history_begin < 0 or
                    type(accepted) is not int or accepted < 0 or
                    type(row_offset) is not int or row_offset < 0):
                raise ValueError('Invalid accepted pressure Query interval')
            if (rows is None or rows.ndim != 4 or rows.device.type != 'cpu' or
                    not rows.is_floating_point() or any(n <= 0 for n in rows.shape)):
                raise ValueError('Pressure Query rows require a nonempty CPU [L,Q,H,D] tensor')
            if history_begin < self.end:
                raise RuntimeError('Public pressure Query capture overlaps accepted history')
            if history_begin > self.end:
                # A restored prefix can predate this small cache. Do not imply
                # coverage across a gap; a full new tail must be observed.
                self.rows = None
                self.begin = self.end = history_begin
            if accepted == 0:
                return
            if self.rows is not None and (self.rows.shape[0], *self.rows.shape[2:]) != (
                    rows.shape[0], *rows.shape[2:]):
                raise ValueError('Pressure Query geometry changed within one authority')
            new_end = history_begin + accepted
            available = max(0, min(rows.shape[1], accepted - row_offset))
            row_begin = history_begin + row_offset
            if not available or row_begin + available != new_end:
                self.rows = None
                self.begin = self.end = new_end
                self.last_stage = 'prefill' if prefill else 'decode'
                return
            current = rows[:, :available]
            if self.rows is not None and self.end == row_begin:
                current = torch.cat((self.rows, current), dim=1)
                row_begin = self.begin
            # Clone even when already CPU BF16: packet DMA storage is recycled
            # after publication, and must never back the accepted Q window.
            self.rows = current[:, -self.window:].to(torch.bfloat16).contiguous().clone()
            self.end = new_end
            self.begin = new_end - self.rows.shape[1]
            self.last_stage = 'prefill' if prefill else 'decode'

    def read(self, end):
        with self._lock:
            if self.closed or type(end) is not int or end <= 0 or end != self.end:
                raise RuntimeError('Pressure Query read differs from the accepted boundary')
            begin = max(0, end - self.window)
            if self.rows is None or self.begin != begin or self.rows.shape[1] != end-begin:
                raise RuntimeError('Pressure Query window lacks the complete accepted tail')
            if not bool(torch.isfinite(self.rows).all()):
                raise ValueError('Nonfinite accepted pressure Query rows')
            return self.rows.clone(), dict(query_source='accepted_public_tail',
                query_begin=begin, query_end=end, query_rows=end-begin,
                accepted_through=end, query_window=self.window,
                query_capture_stage=self.last_stage, query_capture_epoch=self.epoch,
                query_coordinate_frame='content_pre_rope')

    def rewind(self, position):
        with self._lock:
            if self.closed or type(position) is not int or not 0 <= position <= self.end:
                raise ValueError('Invalid pressure Query rewind boundary')
            self.epoch += 1
            if self.rows is not None and self.begin < position <= self.end:
                self.rows = self.rows[:, :position-self.begin].clone()
            else:
                self.rows = None
                self.begin = position
            self.end = position

    def close(self):
        with self._lock:
            self.epoch += 1
            self.rows = None
            self.closed = True
