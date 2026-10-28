"""Recall comparison rejects mixed methods, leaks and answer-based selection."""
from copy import deepcopy
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('recall_comparison_protocol')


@pytest.fixture
def controller(protocol):
    return importlib.import_module('run_recall_comparison')


def profile():
    return dict(max_num_seqs=1, kvmem=dict(block_tokens=32, select_budget=1024,
        gen_budget=512, sink_blocks=1, recent_blocks=1, graph_alpha=0,
        guided_query_tokens=0, query_history=False, repository_gc=False))


def event(method='fps4', kind='prefill_pressure'):
    return dict(sample_id='sample-1', request_id='response-1-worker', event_type=kind,
        configured_method=method, actual_method=method, query_source='accepted_tail',
        candidate_block_count=64, selected_block_ids=[1, 5, 9], fallback_reason=None,
        query_begin=128, query_end=256, accepted_through=256, selected_tokens=96, budget=1024)


def test_three_entrance_coverage_and_common_window(protocol):
    events = [event(kind=kind) for kind in protocol.EVENT_TYPES]
    events[-1].update(query_begin=240, query_end=256, query_source='public_query',
                      query_processed_through=256, public_input_end=260)
    result = protocol.audit_events(events, 'fps4', profile(), query_span=[240, 256],
                                   required_types=protocol.EVENT_TYPES)
    assert result['event_counts'] == dict(prefill_pressure=1, decode_pressure=1, query=1)
    with pytest.raises(ValueError, match='entrances'):
        protocol.audit_events(events[:1], 'fps4', profile(), required_types=protocol.EVENT_TYPES)


def test_public_query_processed_boundary_is_distinct_from_accepted_kv(protocol):
    query = event(kind='query')
    query.update(query_begin=250, query_end=260, accepted_through=250,
                 query_processed_through=260, public_input_end=265)
    protocol.audit_events([query], 'fps4', profile(), query_span=[250, 260])
    query['query_processed_through'] = 259
    with pytest.raises(ValueError, match='processed public-input'):
        protocol.audit_events([query], 'fps4', profile())


@pytest.mark.parametrize('change,message', [
    (dict(actual_method='mean_k'), 'Mixed-method'),
    (dict(fallback_reason='missing_query'), 'fallback'),
    (dict(query_end=257), 'unaccepted'),
    (dict(query_begin=224), '128 accepted'),
    (dict(selected_tokens=1025), 'budget'),
    (dict(selected_block_ids=[1, 1]), 'duplicate'),
    (dict(propagation_applied=True), 'enhancement'),
])
def test_event_contract_rejects_invalid_runs(protocol, change, message):
    row = event(); row.update(change)
    with pytest.raises(ValueError, match=message):
        protocol.audit_events([row], 'fps4', profile())


def test_method_profiles_only_change_method(protocol):
    baseline = profile()
    baseline['recall_experiment'] = dict(device='cpu', query_batch=4, block_batch=32)
    before = deepcopy(baseline)
    profiles = [protocol.method_profile(baseline, method) for method in protocol.METHODS]
    assert baseline == before
    for method, actual in zip(protocol.METHODS, profiles):
        assert actual['recall_experiment'].pop('method') == method
    assert all(actual == profiles[0] for actual in profiles)
    assert profiles[0]['recall_experiment']['pressure_query_window'] == 128


def comparison_rows():
    return [dict(id='a', task='niah_single_1', index=0, served_sha256='a'),
            dict(id='b', task='niah_single_1', index=1, served_sha256='b'),
            dict(id='c', task='qa_1', index=0, served_sha256='c')]


def make_results(rows, method, scores):
    return [dict(row, status='completed', method=method, input_sha256=row['served_sha256'],
                 official_case_score=score, case_score_unrounded=score, full_reference_recall=score == 100)
            for row, score in zip(rows, scores)]


def test_oracle_uses_task_weights_and_excludes_exact_mass(protocol):
    rows = comparison_rows()
    # Raw sample averaging would favor mean_k; task macro favors fixed4.
    values = dict(mean_k=[100, 100, 0], fixed4=[0, 0, 100], fps4=[0, 0, 40],
                  min_max=[20, 20, 0], exact_mass=[100, 100, 100])
    arms = {method: make_results(rows, method, scores) for method, scores in values.items()}
    result = protocol.compare_results(rows, arms)
    assert result['potential']['best_fixed_methods'] == ['mean_k', 'fixed4']
    assert result['potential']['best_fixed_score'] == 50
    assert result['potential']['per_request_oracle_score'] == 100
    assert result['potential']['gain_points'] == 50
    assert result['unique_successes']['mean_k']['sample_ids'] == ['a', 'b']
    assert result['unique_successes']['fixed4']['sample_ids'] == ['c']
    pair = result['pairwise']['mean_k__vs__fixed4']
    assert (pair['wins'], pair['losses'], pair['ties']) == (2, 1, 0)
    assert 'exact_mass' not in result['potential']['candidate_methods']


def test_incomplete_or_changed_pair_cannot_produce_oracle(protocol):
    rows = comparison_rows()
    arms = {method: make_results(rows, method, [0, 0, 0]) for method in protocol.METHODS}
    arms['fps4'].pop()
    with pytest.raises(ValueError, match='Missing'):
        protocol.compare_results(rows, arms)
    arms['fps4'] = make_results(rows, 'fps4', [0, 0, 0])
    arms['fps4'][0]['input_sha256'] = 'different'
    with pytest.raises(ValueError, match='input mismatch'):
        protocol.compare_results(rows, arms)


def test_resume_never_regenerates_complete_answer(controller, tmp_path):
    case = tmp_path / 'case'
    attempt, retained = controller.choose_attempt(case, retry_technical=False)
    assert not retained
    (attempt / 'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='retry-technical'):
        controller.choose_attempt(case, retry_technical=False)
    second, retained = controller.choose_attempt(case, retry_technical=True)
    assert second != attempt and not retained
    (second / 'response.json').write_text(json.dumps(dict(text='incorrect but complete')))
    resumed, retained = controller.choose_attempt(case, retry_technical=True)
    assert resumed == second and retained
    assert len(list((case / 'attempts').iterdir())) == 2


def test_multiple_completed_answers_fail_instead_of_choosing_best(controller, tmp_path):
    case = tmp_path / 'case'
    for name in ('0001', '0002'):
        target = case / 'attempts' / name
        target.mkdir(parents=True)
        (target / 'response.json').write_text('{}')
    with pytest.raises(ValueError, match='Multiple completed'):
        controller.choose_attempt(case, retry_technical=True)


def test_final_answer_only_and_unrounded_official_mass(protocol):
    row = dict(task='niah_multivalue', outputs=['A', 'B', 'C'])
    metric = lambda preds, refs: round(sum(word in preds[0] for word in refs[0]) / len(refs[0]) * 100, 2)
    scorer = ({'niah': dict(metric_fn=metric)}, lambda text, config: text.strip())
    result = protocol.score_case(row, dict(text='A B C</think>A'), scorer, starts_in_thinking=True)
    assert result['official_case_score'] == 33.33
    assert result['case_score_unrounded'] == pytest.approx(100 / 3)
    assert not result['full_reference_recall']


def test_client_resumes_saved_response_after_audit_failure_without_model_retry(controller, protocol,
        monkeypatch, tmp_path):
    root, capture = tmp_path, tmp_path / 'server/capture'
    capture.mkdir(parents=True)
    token_directory = root / 'tokenizer'
    token_directory.mkdir()
    (token_directory / 'tokenizer.json').write_text('{}')
    settings = profile()
    settings.update(tokenizer=str(token_directory), port=8000, served_model_name='test', prefill_chunk_tokens=32)
    protocol.save(root / 'profile.json', settings)
    protocol.save(root / 'profiles/fps4.json', settings)
    row = dict(id='case', task='qa_1', index=0, outputs=['Paris'], input='context Question: city?',
        served_query=[100, 110], served_input_tokens=128, question='city?', served_sha256='input', seed=42)
    fixture = dict(rows=[row], sampling=dict(max_tokens=8192), qualification_ids=['case'],
        tokenizer_sha256=protocol.sha(token_directory / 'tokenizer.json'), upstream_path='upstream',
        starts_in_thinking=False, request_timeout_seconds=10)
    protocol.save(root / 'fixture.json', fixture)
    monkeypatch.setattr(controller, 'verify', lambda *args: (fixture, settings))
    monkeypatch.setattr(controller, 'launch_audit', lambda *args: dict(passed=True))
    monkeypatch.setattr(controller, 'load_tokens', lambda *args: [1] * 128)
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(
        Tokenizer=SimpleNamespace(from_file=lambda path: object())))
    metric = lambda predictions, refs: 100. if 'Paris' in predictions[0] else 0.
    monkeypatch.setattr(controller.ruler, 'load_scorer', lambda path:
        ({'qa': dict(metric_fn=metric)}, lambda text, config: text))
    client = SimpleNamespace(stream=None, wait_retired=None, wait_owner_released=None)
    monkeypatch.setitem(sys.modules, 'qualify_graph_niah', client)
    calls = []
    response = dict(response_id='response-1', text='Paris', output_ids=[1, 2],
        seconds=1., ttft_seconds=.5, finish_reason='stop')
    pressure = event()
    pressure.update(query_begin=0, query_end=96, accepted_through=96, selected_tokens=64)
    query = event(kind='query')
    query.update(query_begin=100, query_end=110, accepted_through=100,
                 query_processed_through=110, public_input_end=128)
    log = capture / 'recall_events.jsonl'
    def stream(*args):
        calls.append(args)
        log.write_text(json.dumps(pressure) + '\n')
        return response
    monkeypatch.setattr(client, 'stream', stream)
    monkeypatch.setattr(client, 'wait_retired', lambda *args: dict(history_end=129))
    monkeypatch.setattr(client, 'wait_owner_released', lambda *args: dict(released=True))
    with pytest.raises(ValueError, match='entrances'):
        controller.run_client(root, 'fps4', 'qualification', capture)
    assert len(calls) == 1
    with log.open('a') as output:
        output.write(json.dumps(query) + '\n')
    result = controller.run_client(root, 'fps4', 'qualification', capture)
    assert len(calls) == 1 and result[0]['official_case_score'] == 100
    result_again = controller.run_client(root, 'fps4', 'qualification', capture)
    assert result_again == result and len(calls) == 1
    case = root / 'arms/fps4/qualification/cases/case'
    (case / result[0]['attempt'] / 'response.json').write_text('{}')
    with pytest.raises(ValueError, match='answer changed'):
        controller.saved_result(case, row, 'fps4', root)


def test_source_freeze_allows_target_build_but_binary_binding_detects_changes(protocol, tmp_path):
    source = tmp_path / 'src/vllm_kvmem'
    source.mkdir(parents=True)
    (source / 'recall.py').write_text('value = 1\n')
    binary = source / '_native.cpython-test.so'
    binary.write_bytes(b'local-build')
    sources = protocol.source_manifest(tmp_path)
    assert all(not name.endswith('.so') for name in sources)
    binary.write_bytes(b'target-gpu-build')
    protocol.verify_sources(tmp_path, sources)
    binaries = protocol.binary_manifest(tmp_path)
    protocol.verify_binaries(tmp_path, binaries)
    binary.write_bytes(b'changed-after-freeze')
    with pytest.raises(ValueError, match='binaries changed'):
        protocol.verify_binaries(tmp_path, binaries)


def test_extra_installed_binary_is_bound(protocol, tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    binary = tmp_path / 'installed-native.so'
    binary.write_bytes(b'installed')
    manifest = protocol.binary_manifest(runtime, [binary])
    assert list(manifest) == [str(binary)]
    protocol.verify_binaries(runtime, manifest)
    binary.unlink()
    with pytest.raises(ValueError, match='Missing compiled'):
        protocol.verify_binaries(runtime, manifest)


def test_functional_option_is_for_qualification_only(controller):
    args = controller.parser().parse_args(['run', '--root', '/tmp/unused', '--phase', 'formal',
                                          '--qualify-entrypoints'])
    with pytest.raises(ValueError, match='only allowed'):
        controller.run(args)


def test_functional_qualification_reuses_the_ten_case_model_process(controller, protocol, monkeypatch, tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    settings = profile()
    protocol.save(tmp_path / 'profiles/fps4.json', settings)
    fixture = dict(runtime_source=str(runtime))
    monkeypatch.setattr(controller, 'verify', lambda *args: (fixture, settings))
    monkeypatch.setattr(controller, 'gpu_lock', lambda *args: nullcontext())
    monkeypatch.setattr(controller, 'termination_guard', lambda: nullcontext())
    state = dict(qualified=False)
    calls = []
    process = SimpleNamespace(pid=12345, returncode=0)
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: calls.append('launch') or process)
    monkeypatch.setattr(controller, 'wait_health', lambda *args: calls.append('healthy'))
    monkeypatch.setattr(controller, 'stop_owned', lambda *args: calls.append('stop'))
    monkeypatch.setattr(controller, 'functional_qualified', lambda *args: state['qualified'])
    captures = []
    def main_client(root, method, phase, capture, **kwargs):
        calls.append('ten-cases'); captures.append(capture)
    def functional_client(root, method, capture, output):
        calls.append('functional'); captures.append(capture); state['qualified'] = True
    monkeypatch.setattr(controller, 'run_client', main_client)
    monkeypatch.setitem(sys.modules, 'qualify_recall_entrypoints', SimpleNamespace(run_client=functional_client))
    args = controller.parser().parse_args(['run', '--root', str(tmp_path), '--runtime', str(runtime),
        '--phase', 'qualification', '--method', 'fps4', '--qualify-entrypoints'])
    controller.run(args)
    assert calls == ['launch', 'healthy', 'ten-cases', 'functional', 'stop']
    assert captures[0] == captures[1]
