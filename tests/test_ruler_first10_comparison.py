"""The matched subset is fixed by original case identity, never by answers."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('compare_ruler_first10')


def fixture(module, count=20):
    rows = []
    for task in module.protocol.TASK_FAMILIES:
        for index in range(count):
            rows.append(dict(id=f'{task}-{index:04}', task=task, index=index,
                input='The answer is Paris. Question: where?', outputs=['Paris'],
                upstream_index=index+90, served_input_tokens=262000, served_query=[261990, 261999],
                question='where?', served_sha256='frozen-input', rendered_sha256='frozen-render',
                upstream_row_sha256='frozen-row', seed=42+index))
    return dict(target_length=262144, seed=42, tokenizer_sha256='tokenizer',
        upstream=module.protocol.UPSTREAM_COMMIT, starts_in_thinking=True,
        sampling=dict(max_tokens=8192, temperature=1.), prompt_protocol='original',
        corpus_manifest_sha256='corpus', rows=rows, samples_per_task=count,
        upstream_path='frozen-upstream', workers=8)


def scorer(module):
    def metric(predictions, references):
        return round(100*sum(sum(r.lower() in p.lower() for r in refs)/len(refs)
            for p, refs in zip(predictions, references))/len(predictions), 2)
    return ({family: dict(metric_fn=metric) for family in module.protocol.TASK_FAMILIES.values()},
        lambda text, config: text.strip())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_campaign(module, root, *, count=20):
    plan = fixture(module, count)
    write(root/'fixture.json', plan)
    results = []
    for row in plan['rows']:
        # Earlier-index failures cannot be replaced by later successful cases.
        text = 'Paris is only in thought' if row['index'] == 0 else 'thought</think>Paris'
        extracted = module.protocol.final_answer(text, starts_in_thinking=True)
        response = dict(text=text, output_ids=[1, 2], response_id=row['id'])
        result = dict(id=row['id'], task=row['task'], index=row['index'], status='completed',
            input_sha256=row['served_sha256'], input_tokens=row['served_input_tokens'],
            text=text, response_id=row['id'], completion_tokens=2, answer=extracted['text'],
            thinking_complete=extracted['thinking_complete'], official_case_score=0 if row['index']==0 else 100)
        results.append(result)
        if row['index'] < 10:
            directory = root/'run/answers/cases'/row['id']
            request = dict(plan['sampling'], seed=row['seed'], prompt_sha256=row['served_sha256'],
                prompt_tokens=row['served_input_tokens'], kvmem_query_span=row['served_query'])
            for name, data in [('request.json', request), ('response.json', response), ('result.json', result)]:
                write(directory/name, data)
    full = module.protocol.score_suite(plan['rows'],
        {(r['task'], r['index']):r['text'] for r in results}, scorer(module),
        samples_per_task=count, starts_in_thinking=True)
    write(root/'run/answers/completed.json', dict(status='completed', protocol_passed=True,
        cases=len(results), results=results, score=full['macro_score'], fixture_sha256=module.sha(root/'fixture.json')))
    write(root/'run/answers/scores.json', full)
    write(root/'status.json', dict(status='completed', score=full['macro_score']))
    write(root/'profile.json', dict(kvmem=dict(graph_alpha=0)))
    return plan


def test_first10_uses_original_index_not_input_or_completion_order(comparison):
    plan = fixture(comparison)
    plan['rows'].reverse()
    rows = comparison.first10(plan)
    assert len(rows) == 130
    assert [r['index'] for r in rows[:10]] == list(range(10))
    assert all(r['index'] < 10 for r in rows)


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'substitute_later'])
def test_missing_or_duplicate_fixed_cases_are_rejected(comparison, fault):
    plan = fixture(comparison)
    if fault == 'duplicate':
        plan['rows'].append(deepcopy(plan['rows'][0]))
    elif fault == 'missing':
        plan['rows'].pop(0)
    else:
        plan['rows'][0]['index'] = 20
    with pytest.raises(ValueError):
        comparison.first10(plan)


def test_signature_allows_execution_differences_but_not_seed_or_input_changes(comparison):
    plan = fixture(comparison)
    subset = deepcopy(plan)
    subset.update(rows=comparison.first10(subset), samples_per_task=10, workers=1)
    assert comparison.fixture_signature(subset) == comparison.fixture_signature(plan)
    subset['rows'][0]['seed'] += 1
    assert comparison.fixture_signature(subset) != comparison.fixture_signature(plan)
    subset['rows'][0]['seed'] -= 1
    subset['rows'][0]['input'] += ' changed'
    assert comparison.fixture_signature(subset) != comparison.fixture_signature(plan)


def test_recompute_fixed_130_from_260_and_match_new_130_campaign(comparison, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison.protocol, 'load_scorer', lambda _: scorer(comparison))
    old, new = tmp_path/'historical', tmp_path/'candidate'
    make_campaign(comparison, old)
    make_campaign(comparison, new, count=10)
    actual = comparison.compare({'old': old, 'candidate': new}, fixture_path=new/'fixture.json')
    assert actual['methods']['old']['full_cohort_macro_score'] == 95
    assert actual['methods']['old']['scores']['macro_score'] == 90
    assert actual['methods']['candidate']['scores']['macro_score'] == 90
    assert len(actual['methods']['old']['cases']) == 130
    assert actual['methods']['old']['cases'][0]['official_case_score'] == 0
    assert '90.0000' in comparison.report(actual)


@pytest.mark.parametrize('fault', ['response', 'sampling', 'fixture', 'evidence'])
def test_modified_historical_evidence_fails_closed(comparison, tmp_path, monkeypatch, fault):
    monkeypatch.setattr(comparison.protocol, 'load_scorer', lambda _: scorer(comparison))
    root = tmp_path/'campaign'
    make_campaign(comparison, root)
    case = root/'run/answers/cases/niah_single_1-0000'
    if fault == 'response':
        response = comparison.read(case/'response.json'); response['text'] = '</think>Paris'
        write(case/'response.json', response)
    elif fault == 'sampling':
        request = comparison.read(case/'request.json'); request['seed'] += 1
        write(case/'request.json', request)
    elif fault == 'fixture':
        plan = comparison.read(root/'fixture.json'); plan['workers'] = 1
        write(root/'fixture.json', plan)
    else:
        write(case/'evidence.json', {'audit': 'ok'})
        write(case/'hashes.json', {name: comparison.sha(case/name) for name in
            ('request.json', 'response.json', 'result.json', 'evidence.json')})
        write(case/'evidence.json', {'audit': 'changed'})
    with pytest.raises(ValueError):
        comparison.extract_reference(root)


def test_native_request_uses_same_sampling_and_input_proof(comparison):
    plan = fixture(comparison)
    row = plan['rows'][0]
    request = dict(sampling=dict(plan['sampling'], seed=row['seed']),
        input_sha256=row['served_sha256'], input_tokens=row['served_input_tokens'])
    comparison.validate_request(request, row, plan)
    request['input_tokens'] -= 1
    with pytest.raises(ValueError, match='served input'):
        comparison.validate_request(request, row, plan)


def test_cpu_fps_nested_request_keeps_original_public_query(comparison):
    plan = fixture(comparison)
    row = plan['rows'][0]
    request = dict(sampling=dict(plan['sampling'], seed=row['seed']),
        prompt_sha256=row['served_sha256'], prompt_tokens=row['served_input_tokens'],
        query_span=row['served_query'], profile_sha256='frozen-profile')
    comparison.validate_request(request, row, plan)
    request['query_span'] = [261989, 261999]
    with pytest.raises(ValueError, match='Public Query changed'):
        comparison.validate_request(request, row, plan)
    request['query_span'] = row['served_query']
    request['input_sha256'] = row['served_sha256']
    with pytest.raises(ValueError, match='Ambiguous'):
        comparison.validate_request(request, row, plan)


def test_candidate_nested_schema_is_accepted_end_to_end(comparison, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison.protocol, 'load_scorer', lambda _: scorer(comparison))
    root = tmp_path/'candidate'
    plan = make_campaign(comparison, root, count=10)
    for row in plan['rows']:
        path = root/'run/answers/cases'/row['id']/'request.json'
        write(path, dict(sampling=dict(plan['sampling'], seed=row['seed']),
            prompt_sha256=row['served_sha256'], prompt_tokens=row['served_input_tokens'],
            query_span=row['served_query'], profile_sha256='frozen-profile'))
    actual, _ = comparison.extract_reference(root)
    assert actual['scores']['macro_score'] == 90
