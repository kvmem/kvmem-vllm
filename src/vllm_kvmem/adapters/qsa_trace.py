"""Opt-in QSA selection trace for the hot-buffer calibration.

``KVMEM_QSA_TOPK_TRACE=<directory>`` registers a hook on the select/attend
split (adapters/qsa_split.py) that copies each ordinary run's selected 4-token
groups to the host before attention reads them. Draft reuse, cache-only, DCP
and overlapped calls are not traced.

The host copy synchronizes, so the trace accepts eager execution only.

Each flush writes ``<directory>/qsa-topk-<pid>-<seq>.npz``:

- decode calls (rows <= the layer's decode capacity), one entry per row:
  ``layer``, ``call``, ``batch_row``, ``state_slot``, ``query_position`` and
  ``groups[rows, 513]``. ``groups[:, k]`` is the k-th selected group id
  (token position // 4) in selection order, -1 for none; a row's open tail
  group, when present, follows its completed groups.
- prefill calls, one entry per request in the call: ``p_layer``, ``p_call``,
  ``p_state_slot``, ``p_rows``, ``p_union`` (distinct groups selected by any
  of the request's rows) and ``p_last_position``.

``trace.json`` names the layers in index order.
"""
import atexit
import json
import os
import threading
from pathlib import Path

import numpy as np
import torch

from ..core import settings as _settings

GROUP_TOKENS = 4
FLUSH_ROWS = 1 << 15


class QSATopKTrace:
    def __init__(self, directory, layers, flush_rows=FLUSH_ROWS):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.names = list(layers)
        self._by_buffer = {}
        for index, (name, module) in enumerate(layers.items()):
            buffer = getattr(module, '_selected_positions', None)
            if buffer is None:
                raise ValueError(f'QSA layer {name} has no selection buffer')
            self._by_buffer[buffer.data_ptr()] = (index, module)
        self.calls = [0] * len(self.names)
        self.flush_rows = flush_rows
        self._decode, self._prefill, self._rows, self._seq = [], [], 0, 0
        self._lock = threading.Lock()
        (self.directory / 'trace.json').write_text(json.dumps(dict(
            layers=self.names, group_tokens=GROUP_TOKENS, pid=os.getpid()), indent=1) + '\n')
        atexit.register(self.flush)

    def hook(self, binding, inputs, query_positions):
        """adapters/qsa_split.py hook: record the selection before attention reads it."""
        if self.layer(binding) is not None:
            self.record(binding, inputs)

    def layer(self, binding):
        selected = getattr(binding, 'selected_positions', None)
        return None if selected is None else self._by_buffer.get(selected.data_ptr())

    def record(self, binding, inputs):
        index, module = self.layer(binding)
        call = self.calls[index]
        self.calls[index] = call + 1
        rows = int(inputs['query'].shape[0])
        positions = binding.selected_positions[:rows, ::GROUP_TOKENS]
        groups = torch.where(positions >= 0, positions // GROUP_TOKENS,
                             torch.full_like(positions, -1)).to(torch.int32)
        request_ids = inputs['request_ids'][:rows].long()
        valid = request_ids >= 0
        slots = binding.raw_state_slot_ids.long()[request_ids.clamp(min=0)]
        slots = torch.where(valid, slots, torch.full_like(slots, -1))
        query_positions = inputs['query_positions'][:rows]
        if rows <= int(getattr(module, 'max_decode_rows', 0)):
            entry = dict(batch_row=request_ids.to(torch.int32).cpu().numpy(),
                         state_slot=slots.to(torch.int32).cpu().numpy(),
                         query_position=query_positions.cpu().numpy(),
                         groups=groups.cpu().numpy())
            entry['layer'] = np.full(rows, index, np.int16)
            entry['call'] = np.full(rows, call, np.int64)
            with self._lock:
                self._decode.append(entry)
                self._rows += rows
        else:
            summaries = []
            for request in torch.unique(request_ids[valid]).tolist():
                mask = request_ids == request
                chosen = groups[mask]
                union = int(torch.unique(chosen[chosen >= 0]).numel())
                summaries.append((index, call, int(slots[mask][0]), int(mask.sum()), union,
                                  int(query_positions[mask].max())))
            with self._lock:
                self._prefill.extend(summaries)
                self._rows += len(summaries)
        if self._rows >= self.flush_rows:
            self.flush()

    def flush(self):
        with self._lock:
            decode, prefill = self._decode, self._prefill
            self._decode, self._prefill, self._rows = [], [], 0
            if not decode and not prefill:
                return None
            seq = self._seq
            self._seq += 1
        arrays = {}
        if decode:
            for key in ('layer', 'call', 'batch_row', 'state_slot', 'query_position', 'groups'):
                arrays[key] = np.concatenate([entry[key] for entry in decode])
        if prefill:
            table = np.asarray(prefill, dtype=np.int64)
            for column, key in enumerate(('p_layer', 'p_call', 'p_state_slot', 'p_rows', 'p_union',
                                          'p_last_position')):
                arrays[key] = table[:, column]
        path = self.directory / f'qsa-topk-{os.getpid()}-{seq:05d}.npz'
        np.savez_compressed(path, **arrays)
        return path


def install_qsa_trace(runner):
    directory = _settings.get('KVMEM_QSA_TOPK_TRACE')
    if not directory:
        return None
    layers = getattr(runner, '_kvmem_qsa_layers', None)
    if not layers:
        raise ValueError('KVMEM_QSA_TOPK_TRACE requires the Flash Next QSA adapter')
    if not runner.model_config.enforce_eager:
        raise ValueError('KVMEM_QSA_TOPK_TRACE copies selections to the host and requires eager execution')
    from .qsa_split import install_split
    trace = QSATopKTrace(directory, layers)
    install_split(runner).append(trace.hook)
    runner._kvmem_qsa_trace = trace
    return trace
