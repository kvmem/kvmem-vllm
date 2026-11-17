"""Host checks of the select/attend split (adapters/qsa_split.py) and the selection trace on it."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_kvmem.adapters.qsa_split import SplitQSA
from vllm_kvmem.adapters.qsa_trace import QSATopKTrace

WIDTH = 2051


def _selection(rows, groups_per_row, tail=None):
    """Selected positions as the B12X expand step writes them: gid*4+j, then a tail, then -1."""
    out = torch.full((rows, WIDTH), -1, dtype=torch.int32)
    for row, groups in enumerate(groups_per_row):
        cursor = 0
        for gid in groups:
            out[row, cursor:cursor + 4] = torch.arange(gid * 4, gid * 4 + 4, dtype=torch.int32)
            cursor += 4
        if tail is not None:
            start, stop = tail[row]
            out[row, cursor:cursor + stop - start + 1] = torch.arange(start, stop + 1, dtype=torch.int32)
    return out


@pytest.fixture
def fake_api(monkeypatch):
    calls = []
    module = types.ModuleType('fake_b12x_contract')

    def run(binding, **kwargs):
        calls.append(('run', kwargs.get('selection_only', False), kwargs.get('reuse')))
        if kwargs.get('selection_only') or kwargs.get('reuse') is None:
            binding.selected_positions.copy_(binding.next_selection)
        return None

    def _run_attention(binding, *, query, request_ids, query_positions):
        calls.append(('attend', int(query.shape[0])))
        return binding.output[:query.shape[0]]

    run.__module__ = module.__name__
    module.run = run
    module._run_attention = _run_attention
    module.Binding = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module, calls


def _layer(max_rows=8, max_decode_rows=4):
    return SimpleNamespace(_selected_positions=torch.full((max_rows, WIDTH), -1, dtype=torch.int32),
                           max_decode_rows=max_decode_rows)


def _binding(layer, selection, slots):
    return SimpleNamespace(selected_positions=layer._selected_positions, next_selection=selection,
                           raw_state_slot_ids=torch.tensor(slots, dtype=torch.int32),
                           output=torch.zeros(8, 1), selection_stream=None)


def _inputs(rows, request_ids, positions):
    return dict(query=torch.zeros(rows, 2, 4), request_ids=torch.tensor(request_ids, dtype=torch.int32),
                query_positions=torch.tensor(positions, dtype=torch.int64))


def test_split_run_traces_decode_groups_and_attends_once(tmp_path, fake_api):
    module, calls = fake_api
    layer = _layer()
    trace = QSATopKTrace(tmp_path, {'l0': layer})
    api = SplitQSA(module, [trace.hook])
    selection = torch.full((8, WIDTH), -1, dtype=torch.int32)
    selection[:2] = _selection(2, [[3, 7], [7, 9]], tail=[(40, 41), (40, 42)])
    binding = _binding(layer, selection, slots=[5, 6, -1, -1])
    api.run(binding, **_inputs(2, [0, 0], [41, 42]))
    # Selection-only first, then the vendor attention on the same binding.
    assert calls == [('run', True, None), ('attend', 2)]
    path = trace.flush()
    data = np.load(path)
    assert data['layer'].tolist() == [0, 0] and data['call'].tolist() == [0, 0]
    assert data['state_slot'].tolist() == [5, 5]
    assert data['groups'].shape == (2, 513)
    assert data['groups'][0, :3].tolist() == [3, 7, 10] and data['groups'][0, 3] == -1
    assert data['groups'][1, :3].tolist() == [7, 9, 10]


def test_prefill_calls_keep_only_per_request_unions(tmp_path, fake_api):
    module, _ = fake_api
    layer = _layer(max_rows=8, max_decode_rows=2)
    trace = QSATopKTrace(tmp_path, {'l0': layer})
    api = SplitQSA(module, [trace.hook])
    selection = torch.full((8, WIDTH), -1, dtype=torch.int32)
    selection[:5] = _selection(5, [[1, 2], [2, 3], [1], [4], [4, 5]])
    binding = _binding(layer, selection, slots=[2, 9])
    api.run(binding, **_inputs(5, [0, 0, 0, 1, 1], [10, 11, 12, 30, 31]))
    data = np.load(trace.flush())
    assert 'groups' not in data.files
    assert data['p_state_slot'].tolist() == [2, 9]
    assert data['p_union'].tolist() == [3, 2]
    assert data['p_rows'].tolist() == [3, 2]
    assert data['p_last_position'].tolist() == [12, 31]


def test_reuse_and_cache_only_pass_through_and_unknown_layers_are_not_traced(tmp_path, fake_api):
    module, calls = fake_api
    layer = _layer()
    trace = QSATopKTrace(tmp_path, {'l0': layer})
    api = SplitQSA(module, [trace.hook])
    selection = torch.full((8, WIDTH), -1, dtype=torch.int32)
    binding = _binding(layer, selection, slots=[0])
    api.run(binding, **_inputs(1, [0], [3]), reuse='anchors')
    api.run(binding, **_inputs(1, [0], [3]), cache_only=True)
    assert calls == [('run', False, 'anchors'), ('run', False, None)]
    other = _binding(_layer(), selection, slots=[0])
    api.run(other, **_inputs(1, [0], [3]))
    assert calls[2:] == [('run', True, None), ('attend', 1)]   # split, but not recorded
    assert trace.flush() is None
    assert api.Binding is object


def test_hooks_rewrite_positions_and_spoof_query_positions(fake_api):
    module, calls = fake_api
    layer = _layer()
    selection = torch.full((8, WIDTH), -1, dtype=torch.int32)
    selection[:1] = _selection(1, [[3]])
    binding = _binding(layer, selection, slots=[0])
    seen = []

    def rewrite(binding, inputs, positions):
        binding.selected_positions[:1, :4] += 100
        return torch.full_like(positions, 1 << 40)

    def observe(binding, inputs, positions):
        seen.append((binding.selected_positions[0, :4].tolist(), int(positions[0])))

    module._run_attention = lambda binding, *, query, request_ids, query_positions: calls.append(
        ('attend', int(query_positions[0])))
    api = SplitQSA(module, [rewrite, observe])
    api.run(binding, **_inputs(1, [0], [13]))
    assert seen == [([112, 113, 114, 115], 1 << 40)]
    assert calls[-1] == ('attend', 1 << 40)
    # No hooks: the combined vendor run, never the split.
    calls.clear()
    SplitQSA(module, []).run(binding, **_inputs(1, [0], [13]))
    assert calls == [('run', False, None)]
