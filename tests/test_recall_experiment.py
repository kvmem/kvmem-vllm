"""Candidate-domain, score normalization and native-picker recall contracts."""
from contextlib import contextmanager
import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import Phase, RequestMemory
from vllm_kvmem import recall_experiment as experiment
from vllm_kvmem.recall_scoring import build_block_features, block_logits
from test_core import append, Event


class ScoringAuthority:
    def __init__(self, keys, counts, *, backend='torch', memory=None):
        self.shape = (keys.shape[0], *keys.shape[2:])
        self.scoring_layers = keys.shape[0]
        self.raw_dtype = keys.dtype
        self.mean_k_backend = backend
        self.history_end = sum(counts)
        self.graph_options = NS(graph_mode='off', graph_alpha=0., query_score_mode='mean_k')
        self.reads = []
        self.records = {}
        self.memory = memory
        handles = ([row.handle for row in memory._test_spans] if memory else
                   [_native.Handle(i, 1) for i in range(len(counts))])
        begin = 0
        for i, (handle, count) in enumerate(zip(handles, counts)):
            raw = keys[:, i, :count].clone()
            total = raw.float().sum(1)
            mean = total / count
            if backend == 'qw3_cuda':
                mean = mean.half()
            record = NS(handle=handle, history_begin=begin, count=count, raw=raw,
                        key_sum=total, mean_key=mean)
            self.records[(handle.slot, handle.generation)] = record
            begin += count

    def _require_open(self):
        pass

    def read_records(self, handles):
        self.reads.append(tuple((h.slot, h.generation) for h in handles))
        return [(self.records[(h.slot, h.generation)].raw,
                 torch.zeros_like(self.records[(h.slot, h.generation)].raw, dtype=torch.uint8))
                for h in handles]

    @contextmanager
    def index_snapshot(self):
        rows = sorted(self.records.values(), key=lambda r: r.history_begin)
        yield rows, torch.stack([r.mean_key for r in rows], dim=1)


@pytest.fixture(autouse=True)
def frozen_environment(monkeypatch):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'mean_k')
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', '128')
    monkeypatch.setenv('KVMEM_RECALL_DEVICE', 'cpu')
    monkeypatch.setenv('KVMEM_RECALL_QUERY_BATCH', '4')
    monkeypatch.setenv('KVMEM_RECALL_BLOCK_BATCH', '2')
    monkeypatch.delenv('KVMEM_RECALL_SAMPLE_ID', raising=False)


@pytest.mark.parametrize('method', experiment.METHODS)
@pytest.mark.parametrize('query_batch,block_batch', [(1, 1), (3, 2), (16, 128)])
def test_score_records_full_domain_matches_unbatched_reference(method, query_batch, block_batch, monkeypatch):
    generator = torch.Generator().manual_seed(431)
    keys = torch.randn(2, 5, 32, 2, 5, generator=generator)
    counts = [32, 32, 32, 32, 9]
    queries = torch.randn(2, 9, 4, 5, generator=generator)
    mask = [True, False, True, True, True]
    authority = ScoringAuthority(keys, counts)
    rows = list(authority.records.values())
    monkeypatch.setenv('KVMEM_RECALL_QUERY_BATCH', str(query_batch))
    monkeypatch.setenv('KVMEM_RECALL_BLOCK_BATCH', str(block_batch))
    actual, stats = experiment.score_records(authority, rows, queries, mask, method)
    logits = block_logits(queries, build_block_features(keys, counts, method))
    expected = logits.masked_fill(~torch.tensor(mask), -torch.inf).softmax(-1).sum((0, 1, 2)) / 8
    torch.testing.assert_close(torch.tensor(actual), expected, atol=1e-6, rtol=2e-6)
    assert actual[1] == 0
    assert stats['query_rows'] == 9 and stats['score_sum'] == pytest.approx(9)
    assert stats['query_reduction'] == 'sum'
    assert stats['head_layer_reduction'] == 'mean'


def test_mean_k_uses_published_mean_instead_of_recomputing_from_archived_k():
    keys = torch.zeros(1, 2, 32, 1, 2)
    keys[:, 0, :, :, 0] = 100.
    authority = ScoringAuthority(keys, [32, 32])
    rows = list(authority.records.values())
    rows[0].mean_key = torch.tensor([[[-5., 0.]]], dtype=torch.float16)
    rows[1].mean_key = torch.tensor([[[5., 0.]]], dtype=torch.float16)
    rows[0].key_sum = rows[1].key_sum = None
    actual, _ = experiment.score_records(authority, rows, torch.tensor([[[[1., 0.]]]]),
                                          [True, True], 'mean_k')
    expected = torch.tensor([-5., 5.]).div(2 ** .5).softmax(0)
    torch.testing.assert_close(torch.tensor(actual), expected)
    assert not authority.reads


@pytest.mark.parametrize('method', experiment.METHODS)
def test_all_masked_domain_is_zero_mass_without_reading_index_or_falling_back(method):
    authority = ScoringAuthority(torch.zeros(1, 2, 32, 1, 2), [32, 32])
    scores, stats = experiment.score_records(authority, list(authority.records.values()),
        torch.zeros(1, 128, 1, 2), [False, False], method)
    assert scores == [0., 0.]
    assert stats['no_eligible_candidates'] and stats['score_sum'] == 0
    assert not authority.reads


@pytest.mark.parametrize('method', experiment.METHODS)
def test_qw3_query_half_precision_is_identical_for_all_methods(method):
    # This cancellation makes missing Q half-rounding visible in final scores.
    keys = torch.zeros(1, 2, 32, 1, 2)
    keys[:, 0] = 1000.
    authority = ScoringAuthority(keys, [32, 32], backend='qw3_cuda')
    queries = torch.tensor([[[[1.0003, -1.]]]])
    scores, _ = experiment.score_records(authority, list(authority.records.values()),
                                          queries, [True, True], method)
    torch.testing.assert_close(torch.tensor(scores), torch.tensor([.5, .5]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('method', ['fixed4', 'fps4', 'min_max', 'exact_mass'])
def test_prototype_cache_reuses_immutable_records_and_invalidates_changed_identity_count_or_method(method):
    generator = torch.Generator().manual_seed(873)
    keys = torch.randn(1, 3, 32, 1, 3, generator=generator)
    authority = ScoringAuthority(keys, [32, 32, 7])
    rows = list(authority.records.values())
    query = torch.randn(1, 2, 1, 3, generator=generator)
    first, _ = experiment.score_records(authority, rows, query, [True] * 3, method)
    assert len(authority.reads) == 1
    again, _ = experiment.score_records(authority, rows, query, [True] * 3, method)
    assert first == again and len(authority.reads) == 1

    tail = rows[-1]
    tail.count = 8
    tail.raw = keys[:, 2, :8].clone()
    authority.history_end += 1
    experiment.score_records(authority, rows, query, [True] * 3, method)
    assert len(authority.reads) == 2 and len(authority.reads[-1]) == 1
    assert len(authority._recall_feature_cache['entries']) == 3

    old = rows[1]
    new = NS(**vars(old))
    new.raw = -old.raw
    authority.records[(new.handle.slot, new.handle.generation)] = new
    rows[1] = new
    experiment.score_records(authority, rows, query, [True] * 3, method)
    assert len(authority.reads) == 3 and len(authority.reads[-1]) == 1

    retired = rows.pop(0)
    del authority.records[(retired.handle.slot, retired.handle.generation)]
    experiment.score_records(authority, rows, query, [True] * 2, method)
    assert len(authority._recall_feature_cache['entries']) == 2
    other = 'min_max' if method != 'min_max' else 'fps4'
    experiment.score_records(authority, rows, query, [True] * 2, other)
    assert len(authority.reads) == 4 and len(authority.reads[-1]) == 2
    assert authority._recall_feature_cache['method'] == other


def native_authority(*, count=233, budget=128, sink=1, recent=0):
    memory = RequestMemory('recall-test', KVMemConfig(block_tokens=32, select_budget=budget,
        gen_budget=256, repository_blocks=24, sink_blocks=sink, recent_blocks=recent, action_fraction=0))
    memory._test_spans = append(memory, count)
    blocks = (count + 31) // 32
    keys = torch.zeros(1, blocks, 32, 1, 2)
    for block in range(blocks):
        keys[:, block, :, :, 0] = block - 3
    return ScoringAuthority(keys, [min(32, count - i * 32) for i in range(blocks)], memory=memory)


def finish(memory):
    memory.attach_completion(Event(True))
    assert memory.commit_selection()


def metadata(authority, *, rows=128):
    return dict(query_source='accepted_public_tail', query_begin=authority.history_end-rows,
                query_end=authority.history_end, query_rows=rows,
                accepted_through=authority.history_end, query_window=128,
                query_coordinate_frame='content_pre_rope')


@pytest.mark.parametrize('event_type', ['prefill_pressure', 'decode_pressure'])
@pytest.mark.parametrize('method', experiment.METHODS)
def test_pressure_native_picker_stays_in_working_domain_and_charges_budget(
        tmp_path, monkeypatch, event_type, method):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', method)
    authority = native_authority()
    memory = authority.memory
    records = list(authority.records.values())
    initial = memory.select({r.handle.slot: float(-i) for i, r in enumerate(records)}, action=False)
    finish(memory)
    del initial
    working = tuple(memory._store.working_records())
    assert len(working) == 4 and len(records) == 8
    candidate_ids = {row[0] for row in working}
    query = torch.tensor([1., 0.]).reshape(1, 1, 1, 2).expand(1, 128, 1, 2)
    try:
        plan = experiment.select_for_event(authority, query, (), max_tokens=96,
            event_type=event_type, query_metadata=metadata(authority),
            request_id='sample-0', output=tmp_path)
        selected = {row.block_id for row in plan.remaps}
        report = authority.recall_selection_report
        assert selected <= candidate_ids
        assert report['candidate_block_count'] == 4
        assert report['candidate_domain'] == 'working_records'
        assert report['charged_tokens'] <= report['budget'] == 96
        assert selected == set(report['selected_block_ids'])
        assert report['actual_method'] == report['configured_method'] == method
        assert report['fallback_reason'] is None
        assert report['query_rows'] == 128
        assert json.loads((tmp_path/'recall_events.jsonl').read_text()) == report
        finish(memory)
        del plan
    finally:
        assert memory.close()


@pytest.mark.parametrize('method', experiment.METHODS)
def test_query_preserves_native_frame_mask_pins_runtime_budget_and_can_recall_archived_block(
        tmp_path, monkeypatch, method):
    from vllm_kvmem.authority import query_score_frame
    monkeypatch.setenv('KVMEM_RECALL_METHOD', method)
    authority = native_authority(count=256, budget=160, sink=1, recent=1)
    memory = authority.memory
    rows = list(authority.records.values())
    initial = memory.select({r.handle.slot: float(-i) for i, r in enumerate(rows)}, action=False, max_tokens=96)
    finish(memory)
    del initial
    working_before = {r[0] for r in memory._store.working_records()}
    mandatory = [rows[2].handle]
    query = torch.tensor([[[[1., 0.]]]])
    with query_score_frame(memory, rows, (mandatory[0].slot,), 128) as frame:
        score_mask = tuple(frame.score_mask)
    scores, _ = experiment.score_records(authority, rows, query, score_mask, method)
    expected = set(memory.preview_selection(dict(zip([r.handle.slot for r in rows], scores)),
        (mandatory[0].slot,), action=False, prefill_scores=None, max_tokens=128))
    assert memory._store.runtime_select_budget == 0
    try:
        plan = experiment.select_for_event(authority, query, mandatory, max_tokens=128,
            event_type='query', query_metadata=dict(query_source='public_query',
                query_begin=256, query_end=257), request_id='sample-1', output=tmp_path)
        actual = {r.block_id for r in plan.remaps}
        assert actual == expected and mandatory[0].slot in actual
        assert actual - working_before
        report = authority.recall_selection_report
        assert report['score_mask_count'] == sum(score_mask)
        assert report['candidate_block_count'] == len(rows)
        assert report['charged_tokens'] <= 128
        assert memory._store.runtime_select_budget == 0
        finish(memory)
        del plan
    finally:
        assert memory.close()


def test_bad_pressure_query_or_missing_required_block_fails_without_fallback(tmp_path):
    authority = native_authority()
    memory = authority.memory
    try:
        with pytest.raises(ValueError, match='128-token tail'):
            experiment.select_for_event(authority, torch.zeros(1, 127, 1, 2), (), max_tokens=96,
                event_type='decode_pressure', query_metadata=metadata(authority, rows=127),
                request_id='bad', output=tmp_path)
        stale = _native.Handle(100, 8)
        with pytest.raises(ValueError, match='Required block outside'):
            experiment.select_for_event(authority, torch.zeros(1, 128, 1, 2), (stale,), max_tokens=96,
                event_type='decode_pressure', query_metadata=metadata(authority),
                request_id='bad', output=tmp_path)
        assert memory.phase is Phase.READY and memory._plan is None
        assert not (tmp_path/'recall_events.jsonl').exists()
    finally:
        assert memory.close()


@pytest.mark.parametrize('changed_field', ['query_begin', 'query_end', 'query_rows'])
def test_pressure_query_metadata_must_match_the_actual_accepted_window(tmp_path, changed_field):
    authority = native_authority()
    memory = authority.memory
    evidence = metadata(authority)
    evidence[changed_field] += 1
    try:
        with pytest.raises(ValueError, match='128-token tail'):
            experiment.select_for_event(authority, torch.zeros(1, 128, 1, 2), (), max_tokens=96,
                event_type='decode_pressure', query_metadata=evidence,
                request_id='bad-metadata', output=tmp_path)
        assert memory.phase is Phase.READY and memory._plan is None
        assert not (tmp_path/'recall_events.jsonl').exists()
    finally:
        assert memory.close()


def test_trace_failure_does_not_open_selection_transaction(tmp_path, monkeypatch):
    authority = native_authority()
    memory = authority.memory
    def fail(*args):
        raise OSError('trace disk unavailable')
    monkeypatch.setattr(experiment, '_write_event', fail)
    try:
        with pytest.raises(OSError, match='trace disk'):
            experiment.select_for_event(authority, torch.zeros(1, 1, 1, 2), (), max_tokens=96,
                event_type='query', query_metadata=dict(query_source='public_query',
                    query_begin=authority.history_end, query_end=authority.history_end+1),
                request_id='bad', output=tmp_path)
        assert memory.phase is Phase.READY and memory._plan is None
    finally:
        assert memory.close()


def test_runtime_budget_below_fixed_native_keep_bands_is_rejected(tmp_path):
    authority = native_authority(count=64, budget=64, sink=1, recent=1)
    memory = authority.memory
    try:
        with pytest.raises(ValueError, match='smaller than.*sink.*recent'):
            experiment.select_for_event(authority, torch.zeros(1, 1, 1, 2), (), max_tokens=32,
                event_type='query', query_metadata=dict(query_source='public_query',
                    query_begin=authority.history_end, query_end=authority.history_end+1),
                request_id='fixed-only', output=tmp_path)
        assert memory.phase is Phase.READY and memory._plan is None
        assert not (tmp_path/'recall_events.jsonl').exists()
    finally:
        assert memory.close()


def test_raw_key_read_failure_cannot_fall_back_to_mean_or_open_transaction(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'fps4')
    authority = native_authority()
    memory = authority.memory
    def unavailable(handles):
        raise BlockingIOError('Accepted Key read temporarily unavailable')
    authority.read_records = unavailable
    try:
        with pytest.raises(BlockingIOError, match='Key read'):
            experiment.select_for_event(authority, torch.zeros(1, 1, 1, 2), (), max_tokens=96,
                event_type='query', query_metadata=dict(query_source='public_query',
                    query_begin=authority.history_end, query_end=authority.history_end+1),
                request_id='read-error', output=tmp_path)
        assert memory.phase is Phase.READY and memory._plan is None
        assert not (tmp_path/'recall_events.jsonl').exists()
    finally:
        assert memory.close()


@pytest.mark.parametrize('key,value', [('graph_mode', 'multi'), ('graph_alpha', .2),
    ('guided_query_tokens', 1), ('query_history', True), ('query_fps_halo_blocks', 1),
    ('query_mean_quota_fraction', .5), ('query_fps_prebuild', True), ('query_score_mode', 'key_fps_4')])
def test_enhancement_or_legacy_mixed_method_configuration_fails(key, value):
    options = NS(graph_mode='off', graph_alpha=0., query_score_mode='mean_k')
    setattr(options, key, value)
    with pytest.raises(ValueError):
        experiment.validate_options(options)


def test_invalid_method_window_and_scratch_configuration_fail(monkeypatch):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'attention')
    with pytest.raises(ValueError, match='METHOD'):
        experiment.configured_method()
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'mean_k')
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', '32')
    with pytest.raises(ValueError, match='128-token'):
        experiment.configured_method()
    options = NS(request_planning=True, async_refresh=True,
                 memory=NS(block_tokens=32), graph_mode='off', graph_alpha=0., query_score_mode='mean_k')
    config = dict(method='fps4', pressure_query_window=128)
    for change in ({'query_batch': 17}, {'block_batch': 129}, {'device': 'mps'},
                   {'pressure_query_window': 32}, {'method': 'fallback'}):
        with pytest.raises(ValueError):
            experiment.launch_environment(dict(config, **change), options)
