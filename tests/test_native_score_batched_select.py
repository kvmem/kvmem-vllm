"""KVMEM_NATIVE_SCORE_BATCHED_SELECT: per-request selected-row masses are identical whether each
(request, layer) gathers its own rows or reads a view of one per-layer gather."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.serving import native_attention as na


def stats(masses, selected, selected_slots, slots):
    s = SimpleNamespace(disabled=None, batch_reason=None, request_reasons={}, frame_invalid=False,
                        reasons={(n, 'decode'): None for n in masses},
                        reduced=set(masses), masses=masses, selected=selected, selected_slots=selected_slots,
                        slots=slots, reduction_rows=sum(c for _, c in selected_slots.values()), counts=Counter(),
                        frame_success=Counter(), mode='native', _gathered={}, _validation_ticket=None)
    s._check_detached_validation = lambda: None
    s.validate = lambda: True
    s._reduce = lambda name: na.NativeAttentionStats._reduce(s, name)
    return s


@pytest.fixture
def frame():
    g = torch.Generator().manual_seed(3)
    masses = {f'l{i}': torch.rand(12, 9, generator=g) for i in range(3)}
    selected_slots = {'a': (0, 2), 'b': (2, 3), 'c': (5, 1)}
    selected = torch.tensor([1, 4, 5, 6, 9, 11, 0, 0], dtype=torch.int32)
    slots = {'a': (0, 4, 7), 'b': (4, 6, 9), 'c': (10, 2, 3), 'd': (2, 2, 5)}
    return masses, selected, selected_slots, slots


@pytest.mark.parametrize('flag', ['0', '1'])
def test_same_values_with_and_without_batching(frame, monkeypatch, flag):
    masses, selected, selected_slots, slots = frame
    monkeypatch.setattr(na, '_BATCHED_SELECT', None)
    monkeypatch.setenv('KVMEM_NATIVE_SCORE_BATCHED_SELECT', '0')
    base = stats(masses, selected, selected_slots, slots)
    expected = {(n, r): na.NativeAttentionStats.score(base, n, r, False, defer_validation=True)
                for n in masses for r in slots}
    monkeypatch.setattr(na, '_BATCHED_SELECT', None)
    monkeypatch.setenv('KVMEM_NATIVE_SCORE_BATCHED_SELECT', flag)
    s = stats(masses, selected, selected_slots, slots)
    for (n, r), want in expected.items():
        got = na.NativeAttentionStats.score(s, n, r, False, defer_validation=True)
        assert want is not None
        assert got.shape == want.shape and got.dtype == want.dtype and torch.equal(got, want)
    assert (len(s._gathered) == len(masses)) == (flag == '1')


def test_flag_is_validated(monkeypatch):
    monkeypatch.setattr(na, '_BATCHED_SELECT', None)
    monkeypatch.setenv('KVMEM_NATIVE_SCORE_BATCHED_SELECT', 'yes')
    with pytest.raises(ValueError):
        na._batched_select()
