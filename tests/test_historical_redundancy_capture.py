from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.capture_runtime import CaptureRuntime, CapturedPacket
from vllm_kvmem.historical_gc import (
    HistoricalGcSettings, RedundancyHistory, redundancy_layers,
    REDUNDANCY_Q_SLOT_BYTES, REDUNDANCY_Q_MAX_SLOTS)
from vllm_kvmem.session_control import SessionBinding


class Scope:
    def __init__(self):
        self.epoch = 0
        self.questions = []
        self.observations = []
        self.features = []
        self.invalidations = 0

    def configure_features(self, *geometry):
        self.geometry = geometry
        return True

    def begin_question(self, expected):
        self.epoch += 1
        self.expected = expected
        return self.epoch

    def append_question(self, epoch, offset, rows, count):
        assert rows.dtype == np.float32 and rows.flags.c_contiguous
        self.questions.append((epoch, offset, rows.copy(), count))
        return True

    def observe_query(self, epoch, rows, count):
        assert rows.dtype == np.float32 and rows.flags.c_contiguous
        self.observations.append((epoch, rows.copy(), count))
        return True

    def invalidate_question(self):
        self.invalidations += 1

    def publish_features(self, lease, values):
        self.features.append((lease, values.copy()))
        return True


def binding(rid='r', *, query=(100, 105), prompt=105, history=0, namespace=('salt', 'session')):
    return SessionBinding(namespace, 1, rid, tuple(range(prompt)), history=history, query_span=query)


def runtime(bindings, *, target_layers=5, heads=2, dim=3):
    obj = CaptureRuntime.__new__(CaptureRuntime)
    obj.redundancy_enabled = True
    obj.redundancy_layers = redundancy_layers(target_layers)
    obj.redundancy_query_heads = heads
    obj.dim = dim
    obj.runner = NS(_kvmem_sessions=NS(bindings={b.request_id: b for b in bindings}, release_requests=set()))
    obj.redundancy_q_stats = dict(packets=0, bytes=0, overflow_packets=0)
    obj.authorities = {}
    for b in bindings:
        scope = Scope()
        obj.authorities[b.request_id] = NS(redundancy_history=RedundancyHistory(scope, target_layers, 1, heads, dim))
    return obj


def batch(ids, histories, counts, prefill, *, gpu_starts=None):
    starts = np.cumsum([0, *counts])
    return NS(req_ids=ids, num_computed_tokens_np=np.array(histories),
        query_start_loc_np=starts, is_prefilling_np=np.array(prefill),
        query_start_loc=torch.tensor(starts if gpu_starts is None else gpu_starts), num_tokens=int(starts[-1]))


def captures(tokens, *, layers=6, heads=2, dim=3):
    # Final layer is draft, final rows can be a private cobatch suffix.
    return [NS(query=(torch.arange(tokens*heads*dim).reshape(tokens, heads, dim)+100*layer).to(torch.bfloat16))
        for layer in range(layers)]


def packet_roundtrip(obj, b, captured, *, retained, history, prefix=None, ids=None):
    prefix = torch.arange(48, dtype=torch.uint8) if prefix is None else prefix
    ids = tuple(range(len(b.req_ids))) if ids is None else ids
    payload, descriptor = obj._capture_redundancy_queries(prefix, captured, b, ids)
    assert torch.equal(payload[:prefix.numel()], prefix)
    packet = CapturedPacket(tuple(b.req_ids[i] for i in ids), b.num_tokens,
        NS(payload=payload), redundancy_queries=descriptor)
    for i in ids:
        obj._accept_redundancy_queries(packet, b.req_ids[i], history[i], retained[i])
    return payload, descriptor


def test_schema_defaults_independent_and_strict():
    params = {'representative_similarity': -0.2}
    config = HistoricalGcSettings(policy='redundancy', policy_params=params)
    assert config.policy_params['feature_cache_bytes'] == 128*1024**2
    assert config.policy_params['query_cache_bytes'] == 8*1024**2
    assert params == {'representative_similarity': -0.2}
    assert HistoricalGcSettings().policy_params == {'keep_mandatory': True}
    assert REDUNDANCY_Q_SLOT_BYTES == 64*1024**2 and REDUNDANCY_Q_MAX_SLOTS == 4
    for key, value in [('keep_mandatory', False), ('reference_blocks', True),
                       ('reference_blocks', 0), ('query_cache_bytes', 2**40+1),
                       ('importance_weight', float('nan')), ('importance_weight', True),
                       ('importance_weight', 10**1000),
                       ('representative_similarity', -1.01), ('unknown', 1)]:
        with pytest.raises(ValueError):
            HistoricalGcSettings(policy='redundancy', policy_params={key: value})


def test_selected_layers_exclude_draft_and_no_extra_history_q_bytes():
    obj = runtime([binding()])
    assert obj.redundancy_layers == (0, 1, 2, 4)
    prefix = torch.arange(48, dtype=torch.uint8)
    payload, rows = obj._capture_redundancy_queries(prefix, captures(80),
        batch(['r'], [0], [80], [True]), (0,))
    assert payload is prefix and rows['r']['count'] == 0
    assert obj.redundancy_q_stats['bytes'] == 0


def test_real_sidecar_roundtrip_question_spans_chunks_and_short_final():
    obj = runtime([binding()])
    first, second = captures(103), captures(20)
    packet_roundtrip(obj, batch(['r'], [0], [103], [True]), first,
        retained=[103], history=[0])
    history = obj.authorities['r'].redundancy_history
    assert history.offset == 3 and not history.complete
    packet_roundtrip(obj, batch(['r'], [103], [9], [True]), second,
        retained=[2], history=[103])
    assert history.complete and history.offset == 5
    assert [r[1] for r in history.scope.questions] == [0, 3]
    assert [r[3] for r in history.scope.questions] == [3, 2]
    expected = np.stack([first[i].query[100:103].float().numpy() for i in obj.redundancy_layers])
    np.testing.assert_array_equal(history.scope.questions[0][2], expected)


def test_mtp_and_terminal_trim_only_consumed_target_rows():
    b = binding(query=(0, 1), prompt=1)
    obj = runtime([b])
    packet_roundtrip(obj, batch(['r'], [0], [1], [True]), captures(1), retained=[1], history=[0])
    generated = captures(12)  # private rows 5..12 are outside the public batch
    packet_roundtrip(obj, batch(['r'], [1], [5], [False]), generated, retained=[2], history=[1])
    observed = obj.authorities['r'].redundancy_history.scope.observations[0]
    assert observed[2] == 2
    expected = np.stack([generated[i].query[:2].float().numpy() for i in obj.redundancy_layers])
    np.testing.assert_array_equal(observed[1], expected)


def test_one_token_prefill_is_question_not_observation():
    obj = runtime([binding(query=(0, 1), prompt=1)])
    packet_roundtrip(obj, batch(['r'], [0], [1], [True]), captures(1), retained=[1], history=[0])
    history = obj.authorities['r'].redundancy_history
    assert history.complete and len(history.scope.questions) == 1
    assert not history.scope.observations


def test_adaptive_gpu_starts_not_cpu_upper_bound_and_split_owner():
    a, b = binding('a', query=(0, 1), prompt=1), binding('b', query=(0, 1), prompt=1, namespace=('s', 'b'))
    obj = runtime([a, b])
    for owner in (a, b):
        obj.authorities[owner.request_id].redundancy_history.accept(owner, 0, 1, prefill=True,
            rows=torch.ones(4, 1, 2, 3, dtype=torch.bfloat16))
    values = captures(12)
    public = batch(['a', 'b'], [1, 1], [5, 5], [False, False], gpu_starts=[0, 2, 5])
    packet_roundtrip(obj, public, values, retained=[2, 3], history=[1, 1], ids=(1,))
    observation = obj.authorities['b'].redundancy_history.scope.observations[0]
    expected = np.stack([values[i].query[2:5].float().numpy() for i in obj.redundancy_layers])
    np.testing.assert_array_equal(observation[1], expected)
    assert not obj.authorities['a'].redundancy_history.scope.observations


def test_overflow_is_explicit_fail_closed_without_partial_question(monkeypatch):
    import vllm_kvmem.historical_gc as gc
    monkeypatch.setattr(gc, 'REDUNDANCY_Q_SLOT_BYTES', 4)
    obj = runtime([binding(query=(0, 3), prompt=3)])
    payload, _ = packet_roundtrip(obj, batch(['r'], [0], [3], [True]), captures(3), retained=[3], history=[0])
    history = obj.authorities['r'].redundancy_history
    assert payload.numel() == 48 and history.invalid
    assert history.stats['transport_overflow'] == 1 and not history.scope.questions


def test_exact_prefix_tool_reuse_and_rewind_invalidate_same_request():
    b = binding('first', query=(0, 2), prompt=2)
    history = RedundancyHistory(Scope(), 2, 1, 2, 3)
    history.accept(b, 0, 2, prefill=True, rows=torch.ones(2, 2, 2, 3))
    epoch = history.epoch
    next_b = binding('tool', query=(0, 2), prompt=4, history=2)
    history.accept(next_b, 2, 2, prefill=True, rows=None)
    assert history.complete and history.epoch == epoch and history.stats['question_reuses'] == 1
    history.invalidate()
    history.accept(next_b, 0, 2, prefill=True, rows=torch.ones(2, 2, 2, 3))
    assert history.complete and history.epoch != epoch


def test_changed_or_missing_cached_question_cannot_reuse():
    b = binding(query=(0, 2), prompt=2)
    history = RedundancyHistory(Scope(), 1, 1, 2, 3)
    history.accept(b, 0, 2, prefill=True, rows=torch.ones(1, 2, 2, 3))
    changed = SessionBinding(b.namespace, 1, 'changed', (9, 9, 3), history=2, query_span=(0, 2))
    history.accept(changed, 2, 1, prefill=False, rows=torch.ones(1, 1, 2, 3))
    assert history.invalid and history.stats['missing_cached_question'] == 1
    assert not history.scope.observations


@pytest.mark.parametrize('valid,drained', [(True, True), (False, True), (True, False)])
def test_authority_invalidates_gc_before_retiring_rewind_suffix(valid, drained):
    from vllm_kvmem.authority import RequestAuthority

    events = []
    checkpoint = NS(position=32)

    def rewind(value):
        assert value is checkpoint and events == ['invalidate']
        events.append('retire_suffix')
        return ()

    authority = NS(_require_open=lambda: None,
        memory=NS(checkpoint_valid=lambda _: valid, begin_rewind=rewind),
        shape=(1, 32), flush=lambda: drained, writes=[], query_score_mode='mean_k',
        redundancy_history=NS(invalidate=lambda: events.append('invalidate')),
        query_window_cache=None)
    if not valid:
        with pytest.raises(ValueError, match='Stale or foreign'):
            RequestAuthority.begin_rewind(authority, checkpoint)
        assert not events
    elif not drained:
        assert RequestAuthority.begin_rewind(authority, checkpoint) is None
        assert not events
    else:
        assert RequestAuthority.begin_rewind(authority, checkpoint) == ()
        assert events == ['invalidate', 'retire_suffix']


def test_missing_required_rows_invalidates_instead_of_silent_omission():
    history = RedundancyHistory(Scope(), 1, 1, 2, 3)
    history.accept(binding(query=(0, 3), prompt=3), 0, 3, prefill=True, rows=None)
    assert history.invalid and not history.complete


@pytest.mark.parametrize('token', [-1, 2**32, True, 1.5])
def test_question_digest_validates_uint32_tokens(token):
    invalid = NS(tokens=(token,))
    with pytest.raises(ValueError, match='uint32'):
        RedundancyHistory._digest(invalid, (0, 1))


def test_decode_uses_accepted_history_when_cpu_mtp_upper_bound_lags():
    b = binding(query=(0, 1), prompt=1)
    obj = runtime([b])
    packet_roundtrip(obj, batch(['r'], [0], [1], [True]), captures(1), retained=[1], history=[0])
    packet_roundtrip(obj, batch(['r'], [9], [5], [False]), captures(9), retained=[2], history=[6])
    history = obj.authorities['r'].redundancy_history
    assert history.complete and not history.invalid
    assert history.stats['observation_tokens'] == 2


def test_sealed_meank_publish_does_not_need_sum_or_raw_keys():
    class Lease:
        released = False
        def reset(self):
            self.released = True
    lease = Lease()
    scope = Scope()
    history = RedundancyHistory(scope, 5, 1, 2, 3)
    means = torch.arange(15, dtype=torch.float16).reshape(5, 1, 3)
    record = NS(mean_key=means, archived=True, handle=object(), key_sum=None, raw=None)
    assert history.publish_key(record, NS(pin=lambda _: lease))
    np.testing.assert_array_equal(scope.features[0][1], means[[0, 1, 2, 4]].float().numpy())
    assert lease.released


def test_unrelated_unowned_public_request_rejected_but_release_is_allowed():
    obj = runtime([])
    b = batch(['release'], [0], [1], [True])
    with pytest.raises(ValueError, match='owned session'):
        obj._capture_redundancy_queries(torch.zeros(48, dtype=torch.uint8), captures(1), b, (0,))
    obj.runner._kvmem_sessions.release_requests.add('release')
    packet_roundtrip(obj, b, captures(1), retained=[1], history=[0])


def test_native_header_retained_prefix_and_kv_payload_ignore_sidecar():
    # Native input/history is logical even after a working-window reduction;
    # captured attention/RoPE coordinates carry the independent local position.
    b = binding(query=(100000, 100005), prompt=100005)
    obj = runtime([b], target_layers=1, heads=1, dim=2)
    history = obj.authorities['r'].redundancy_history
    seen = []
    authority = NS(redundancy_history=history,
        append=lambda raw, values, **kw: seen.append((raw.clone(), values.clone(), kw)) or True)
    obj.ensure_authority = lambda _: authority
    obj.authorities['r'] = authority
    obj._prepare_graph_acceptance = lambda *a: None
    obj._abort_graph_updates = lambda *a: None
    obj._publication_metric = lambda *a, **kw: None
    obj.options = NS(batched_cpu_append=False)
    obj.terminal_prefixes = {'r': NS(history_end=100002)}
    obj.layers, obj.heads, obj.dim, obj.row_elements, obj.raw_dtype = [None], 1, 2, 2, torch.bfloat16
    header = torch.tensor([0, 5, 100000, 32768, 2], dtype=torch.int64).view(torch.uint8)
    raw = torch.arange(10).to(torch.bfloat16).view(torch.uint8).flatten()
    vals = torch.arange(10, dtype=torch.uint8)
    public = batch(['r'], [100000], [5], [True])
    payload, descriptors = obj._capture_redundancy_queries(torch.cat((header, raw, vals)),
        captures(5, layers=1, heads=1, dim=2), public, (0,))
    packet = CapturedPacket(('r',), 5, NS(payload=payload, query=lambda: True),
        redundancy_queries=descriptors)
    metadata = obj._copy_source_owners(packet, [0], False)
    assert metadata == [(0, 2, 1, 5, 100000, 2)]
    assert seen[0][2]['accepted'] == 2
    assert seen[0][2]['history_begin'] == 100000
    assert seen[0][2]['local_rope_begin'] == 32768
    torch.testing.assert_close(seen[0][0].flatten(), torch.arange(10).to(torch.bfloat16))
    torch.testing.assert_close(seen[0][1].flatten(), vals)
    obj._accept_redundancy_queries(packet, 'r', 100000, metadata[0][1])
    assert history.scope.questions[0][3] == 2 and not history.complete


@pytest.mark.parametrize('backend', ['torch_reference', 'qw3_cuda'])
def test_real_native_archive_mean_features_and_question_publication(tmp_path, backend):
    import time
    from vllm_kvmem import _native as n
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive

    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 2*32*8*2, 64, native_payloads=True)
    settings = n.HistoricalGcConfig()
    settings.policy = 'redundancy'
    settings.mode, settings.background = n.HistoricalGcMode.SHADOW, False
    settings.ready_low, settings.ready_high = 2, 4
    worker = archive.enable_history_gc(settings)
    authority = RequestAuthority('features', config, archive, 2, 1, 8,
        raw_dtype=torch.float8_e4m3fn, mean_k_backend=backend,
        cpu_release_sealed_sums=backend == 'qw3_cuda')
    scope = authority.gc.scope
    authority.redundancy_history = source = RedundancyHistory(scope, 2, 1, 2, 8)
    raw = (torch.rand(2, 65, 1, 8, generator=torch.Generator().manual_seed(12))+.25).to(torch.float8_e4m3fn)
    values = torch.zeros(raw.shape, dtype=torch.uint8)
    try:
        assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
        authority.publish_sealed(lambda: None)
        authority.poll()
        assert source.configured and source.stats['features_published'] == 2
        assert source.stats['features_missing'] == 0 and scope.feature_stats['keys'] == 2
        for record in authority.records.values():
            if record.archived:
                assert (record.key_sum is None) == (backend == 'qw3_cuda')
                expected = raw[:, record.history_begin:record.history_begin+32].float().mean(1)
                if backend == 'qw3_cuda':
                    expected = expected.half().float()
                lease = authority.memory.repository.pin(record.handle)
                try:
                    # A same-generation exact value republishes idempotently.
                    assert scope.publish_features(lease, expected.numpy())
                finally:
                    lease.reset()
        del record
        b = binding(query=(0, 35), prompt=35)
        source.accept(b, 0, 20, prefill=True, rows=torch.ones(2, 20, 2, 8))
        assert not scope.feature_stats['question_ready']
        source.accept(b, 20, 15, prefill=True, rows=torch.ones(2, 15, 2, 8))
        assert scope.feature_stats['question_ready']
        assert scope.feature_stats['question_tokens'] == 35
        assert scope.feature_stats['question_segments'] == 2
        source.accept(b, 35, 8, prefill=False, rows=torch.ones(2, 11, 2, 8))
        assert scope.feature_stats['observed_tokens'] == 8
        source.invalidate()
        assert not scope.feature_stats['question_ready']
        source.accept(b, 0, 35, prefill=True, rows=torch.ones(2, 35, 2, 8))
        assert scope.feature_stats['question_ready']
        assert scope.feature_stats['observed_tokens'] == 0
    finally:
        deadline = time.monotonic()+5
        while not authority.close():
            worker.advance()
            assert time.monotonic() < deadline
            time.sleep(.001)
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
    assert archive.io_usage()['cpu_index_bytes'] == archive.io_usage()['cpu_authority_bytes'] == 0
