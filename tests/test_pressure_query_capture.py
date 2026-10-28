from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.pressure_query_capture import (
    PressureQueryWindow, configured_method, PRESSURE_QUERY_WINDOW)
from vllm_kvmem.capture_runtime import CaptureRuntime, CapturedPacket


def queries(count, *, first=0, layers=2, heads=2, dim=3):
    return torch.stack([torch.arange(first, first+count)[:, None, None].expand(count, heads, dim)
        + layer*1000 for layer in range(layers)]).to(torch.bfloat16)


def captures(count, *, first=0):
    # The final layer and any suffix beyond actual public starts are private.
    return [NS(query=row) for row in queries(count, first=first, layers=3)]


def runtime(ids=('r',)):
    obj = CaptureRuntime.__new__(CaptureRuntime)
    obj.pressure_query_enabled = True
    obj.pressure_query_window = 128
    obj.pressure_query_layers = (0, 1)
    obj.pressure_query_heads = 2
    obj.dim = 3
    obj.pressure_query_stats = dict(packets=0, bytes=0, accepted_rows=0)
    obj.authorities = {rid: NS(pressure_query_window=PressureQueryWindow()) for rid in ids}
    return obj


def batch(ids, histories, counts, prefill, gpu_starts=None):
    starts = np.cumsum([0, *counts])
    return NS(req_ids=ids, num_computed_tokens_np=np.array(histories),
        query_start_loc_np=starts, is_prefilling_np=np.array(prefill),
        query_start_loc=torch.tensor(starts if gpu_starts is None else gpu_starts),
        num_tokens=int(starts[-1]))


def packet(obj, public, values, indices=None):
    indices = tuple(range(len(public.req_ids))) if indices is None else indices
    prefix = torch.arange(48, dtype=torch.uint8)
    payload, descriptors = obj._capture_pressure_queries(prefix, values, public, indices)
    assert torch.equal(payload[:48], prefix)
    return CapturedPacket(tuple(public.req_ids[i] for i in indices), public.num_tokens,
        NS(payload=payload), pressure_queries=descriptors)


def test_experiment_environment_validates_method_and_frozen_window(monkeypatch):
    monkeypatch.delenv('KVMEM_RECALL_METHOD', raising=False)
    assert configured_method() is None
    for method in ('mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass'):
        monkeypatch.setenv('KVMEM_RECALL_METHOD', method)
        assert configured_method() == method
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'attention')
    with pytest.raises(ValueError, match='KVMEM_RECALL_METHOD'):
        configured_method()
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'mean_k')
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', '32')
    with pytest.raises(ValueError, match='fixes'):
        configured_method()
    assert PRESSURE_QUERY_WINDOW == 128


def test_long_prefill_retains_original_last_128_rows_and_owns_bytes():
    obj = runtime()
    values = captures(1024)
    item = packet(obj, batch(['r'], [0], [1024], [True]), values)
    row = item.pressure_queries['r']
    assert row['count'] == 128 and row['row_offset'] == 896
    obj._accept_pressure_queries(item, 'r', 0, 1024)
    actual, metadata = obj.authorities['r'].pressure_query_window.read(1024)
    expected = torch.stack([v.query[896:1024] for v in values[:2]])
    assert torch.equal(actual, expected)
    item.download.payload.zero_()
    assert torch.equal(obj.authorities['r'].pressure_query_window.read(1024)[0], expected)
    assert metadata == dict(query_source='accepted_public_tail', query_begin=896,
        query_end=1024, query_rows=128, accepted_through=1024, query_window=128,
        query_capture_stage='prefill', query_capture_epoch=0,
        query_coordinate_frame='content_pre_rope')


def test_short_history_uses_all_rows_then_decode_appends_only_accepted_prefix():
    obj = runtime()
    initial = packet(obj, batch(['r'], [0], [3], [True]), captures(3))
    obj._accept_pressure_queries(initial, 'r', 0, 3)
    # CPU scheduler count can be an upper bound; accepted header owns position.
    generated = packet(obj, batch(['r'], [6], [5], [False]), captures(9, first=3))
    obj._accept_pressure_queries(generated, 'r', 3, 2)
    actual, metadata = obj.authorities['r'].pressure_query_window.read(5)
    assert torch.equal(actual, queries(5))
    assert metadata['query_rows'] == 5 and metadata['query_capture_stage'] == 'decode'
    assert obj.pressure_query_stats['accepted_rows'] == 5


def test_adaptive_mtp_device_starts_exclude_other_owner_and_private_suffix():
    obj = runtime(('a', 'b'))
    public = batch(['a', 'b'], [0, 0], [5, 5], [False, False], gpu_starts=[0, 2, 5])
    values = captures(12)
    item = packet(obj, public, values, indices=(1,))
    obj._accept_pressure_queries(item, 'b', 0, 3)
    actual, _ = obj.authorities['b'].pressure_query_window.read(3)
    assert torch.equal(actual, queries(3, first=2))
    assert obj.authorities['a'].pressure_query_window.rows is None


def test_rolling_window_spans_multiple_accepted_decode_chunks():
    window = PressureQueryWindow()
    window.accept(0, 128, queries(128), prefill=True)
    window.accept(128, 3, queries(7, first=128), prefill=False)
    window.accept(131, 2, queries(5, first=131), prefill=False)
    actual, metadata = window.read(133)
    assert torch.equal(actual, queries(128, first=5))
    assert metadata['query_begin'] == 5


def test_terminal_trim_cannot_silently_reuse_wrong_prefill_suffix():
    obj = runtime()
    item = packet(obj, batch(['r'], [0], [1024], [True]), captures(1024))
    obj._accept_pressure_queries(item, 'r', 0, 900)
    window = obj.authorities['r'].pressure_query_window
    with pytest.raises(RuntimeError, match='complete accepted tail'):
        window.read(900)
    window.accept(900, 128, queries(128, first=900), prefill=True)
    assert torch.equal(window.read(1028)[0], queries(128, first=900))


def test_future_reads_and_duplicate_public_replays_fail_closed():
    window = PressureQueryWindow()
    window.accept(0, 4, queries(4), prefill=True)
    for end in (0, 3, 5):
        with pytest.raises(RuntimeError, match='accepted boundary'):
            window.read(end)
    with pytest.raises(RuntimeError, match='overlaps'):
        window.accept(0, 4, queries(4), prefill=True)
    assert window.end == 4


def test_rewind_rejects_stale_packets_and_requires_complete_new_tail():
    obj = runtime()
    window = obj.authorities['r'].pressure_query_window
    window.accept(0, 256, queries(128, first=128), row_offset=128, prefill=True)
    stale = packet(obj, batch(['r'], [256], [5], [False]), captures(5, first=256))
    window.rewind(224)
    with pytest.raises(RuntimeError, match='Stale'):
        obj._accept_pressure_queries(stale, 'r', 224, 2)
    with pytest.raises(RuntimeError, match='complete accepted tail'):
        window.read(224)
    window.accept(224, 32, queries(32, first=224), epoch=1, prefill=True)
    assert torch.equal(window.read(256)[0], queries(128, first=128))


def test_restored_gap_requires_new_complete_tail():
    window = PressureQueryWindow()
    window.accept(1000, 5, queries(5, first=1000), prefill=True)
    with pytest.raises(RuntimeError, match='complete accepted tail'):
        window.read(1005)
    window.accept(1005, 123, queries(123, first=1005), prefill=True)
    assert torch.equal(window.read(1128)[0], queries(128, first=1000))


def test_zero_accepted_rows_do_not_advance_or_admit_drafts():
    window = PressureQueryWindow()
    window.accept(0, 2, queries(2), prefill=True)
    window.accept(2, 0, queries(5, first=2))
    assert torch.equal(window.read(2)[0], queries(2))


def test_close_drops_rows_and_rejects_acceptance():
    window = PressureQueryWindow()
    window.accept(0, 2, queries(2), prefill=True)
    window.close()
    assert window.rows is None
    with pytest.raises(RuntimeError, match='accepted boundary'):
        window.read(2)
    with pytest.raises(RuntimeError, match='closed'):
        window.accept(2, 1, queries(1))


def test_public_prefill_boundary_mismatch_and_missing_descriptor_are_errors():
    obj = runtime()
    item = packet(obj, batch(['r'], [0], [5], [True]), captures(5))
    with pytest.raises(RuntimeError, match='accepted prefix'):
        obj._accept_pressure_queries(item, 'r', 1, 5)
    with pytest.raises(RuntimeError, match='descriptor'):
        obj._accept_pressure_queries(item, 'r', 0, 5)


def test_capture_disabled_does_not_copy_or_require_new_fields():
    obj = CaptureRuntime.__new__(CaptureRuntime)
    original = torch.zeros(48, dtype=torch.uint8)
    result, descriptors = obj._capture_pressure_queries(original, (), None, ())
    assert result is original and descriptors == {}
    obj._accept_pressure_queries(None, 'r', 0, 1)


@pytest.mark.parametrize('valid,drained', [(True, True), (False, True), (True, False)])
def test_authority_rewind_mutates_q_epoch_only_after_valid_drained_request(valid, drained):
    from vllm_kvmem.authority import RequestAuthority
    window = PressureQueryWindow()
    window.accept(0, 64, queries(64), prefill=True)
    checkpoint = NS(position=32)
    authority = NS(_require_open=lambda: None,
        memory=NS(checkpoint_valid=lambda _: valid, begin_rewind=lambda _: ()),
        shape=(2, 32), flush=lambda: drained, writes=[], query_score_mode='mean_k',
        query_window_cache=None, pressure_query_window=window, _recall_feature_cache={'old': object()})
    if not valid:
        with pytest.raises(ValueError, match='Stale or foreign'):
            RequestAuthority.begin_rewind(authority, checkpoint)
    else:
        assert RequestAuthority.begin_rewind(authority, checkpoint) == (() if drained else None)
    assert window.epoch == int(valid and drained)
    assert window.end == (32 if valid and drained else 64)
    assert bool(authority._recall_feature_cache) != (valid and drained)
