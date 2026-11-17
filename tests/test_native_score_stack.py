"""KVMEM_NATIVE_SCORE_STACK: one request's masses for every layer from one stacked gather are
identical to the per-layer score() results stacked, with the same counters; any per-layer
fallback makes score_stack decline without counting, so the caller keeps the per-layer path."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.serving import native_attention as na

NAMES = ('l0', 'l1', 'l2', 'l3')


def stats(stack, selected, selected_slots, slots):
    masses = {name: stack[i] for i, name in enumerate(NAMES)}
    s = SimpleNamespace(disabled=None, batch_reason=None, request_reasons={}, frame_invalid=False,
                        reasons={(n, 'decode'): None for n in NAMES}, names=list(NAMES),
                        reduced=set(NAMES), masses=masses, _mass_stack=stack, selected=selected,
                        selected_slots=selected_slots, slots=slots,
                        reduction_rows=sum(c for _, c in selected_slots.values()), counts=Counter(),
                        frame_success=Counter(), mode='reuse', _gathered={}, _gathered_stack=None,
                        _validation_ticket=None)
    s._check_detached_validation = lambda: None
    s.validate = lambda: True
    s._reduce = lambda name: na.NativeAttentionStats._reduce(s, name)
    s._reduce_layers = lambda names: na.NativeAttentionStats._reduce_layers(s, names)
    return s


@pytest.fixture
def frame():
    g = torch.Generator().manual_seed(7)
    stack = torch.rand(len(NAMES), 12, 9, generator=g)
    selected_slots = {'a': (0, 2), 'b': (2, 3), 'c': (5, 1)}
    selected = torch.tensor([1, 4, 5, 6, 9, 11, 0, 0], dtype=torch.int32)
    slots = {'a': (0, 4, 7), 'b': (4, 6, 9), 'c': (10, 2, 3), 'd': (2, 2, 5)}
    return stack, selected, selected_slots, slots


@pytest.mark.parametrize('batched_select', ['0', '1'])
def test_stack_equals_per_layer_scores(frame, monkeypatch, batched_select):
    stack, selected, selected_slots, slots = frame
    monkeypatch.setattr(na, '_BATCHED_SELECT', None)
    monkeypatch.setenv('KVMEM_NATIVE_SCORE_BATCHED_SELECT', batched_select)
    base = stats(stack, selected, selected_slots, slots)
    s = stats(stack, selected, selected_slots, slots)
    for request in slots:
        want = torch.stack([na.NativeAttentionStats.score(base, n, request, False, defer_validation=True)
                            for n in NAMES])
        got = na.NativeAttentionStats.score_stack(s, NAMES, request, False)
        assert got.shape == want.shape and got.dtype == want.dtype
        assert torch.equal(got.contiguous(), want)
    assert s.counts == base.counts and s.frame_success == base.frame_success
    assert s._gathered_stack is not None and s._gathered_stack.shape == (len(NAMES), 6, 9)


@pytest.mark.parametrize('spoil', ['layer_reason', 'request_reason', 'unprepared', 'frame_invalid',
                                   'disabled', 'order'])
def test_any_fallback_declines_without_counting(frame, spoil):
    stack, selected, selected_slots, slots = frame
    s = stats(stack, selected, selected_slots, slots)
    names, request = NAMES, 'a'
    if spoil == 'layer_reason':
        s.reasons[('l2', 'decode')] = 'backend_not_observed'
    elif spoil == 'request_reason':
        s.request_reasons['a'] = 'candidate_not_prepared'
    elif spoil == 'unprepared':
        request = 'zz'
    elif spoil == 'frame_invalid':
        s.frame_invalid = True
    elif spoil == 'disabled':
        s.disabled = 'startup_buffer_capacity'
    else:
        names = NAMES[::-1]
    assert na.NativeAttentionStats.score_stack(s, names, request, False) is None
    assert not s.counts and not s.frame_success and s._gathered_stack is None


def test_flag_is_validated(monkeypatch):
    monkeypatch.setattr(na, '_SCORE_STACK', None)
    monkeypatch.setenv('KVMEM_NATIVE_SCORE_STACK', 'yes')
    with pytest.raises(ValueError):
        na.score_stack_enabled()
    monkeypatch.setattr(na, '_SCORE_STACK', None)
