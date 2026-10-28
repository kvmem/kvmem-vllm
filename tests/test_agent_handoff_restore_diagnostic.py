import json

import pytest

from scripts.diagnose_agent_handoff_restore import fixed_requests, token_hash


def cohort(tmp_path):
    fixture = [dict(case=str(i), tokens=[i, 10, 11], suffix=[12]) for i in range(8)]
    for i, item in enumerate(fixture):
        tokens = list(item['tokens'])
        for turn in range(3):
            ids = [100 + turn] * 64
            (tmp_path/f'{i}-{turn}.json').write_text(json.dumps(dict(input_sha256=token_hash(tokens),
                response=dict(choices=[dict(token_ids=ids)], usage=dict(completion_tokens=64)))))
            tokens += ids + item['suffix']
    return fixture


def test_fixed_prompts_include_exact_prior_accepted_ids(tmp_path):
    fixture = cohort(tmp_path)
    result = fixed_requests(fixture, tmp_path)
    assert len(result) == 8
    assert result[3]['turns'][2]['tokens'] == [3, 10, 11] + [100]*64 + [12] + [101]*64 + [12]
    result[3]['turns'][0]['tokens'].append(999)
    assert fixture[3]['tokens'] == [3, 10, 11]
    assert 999 not in result[3]['turns'][1]['tokens']


def test_changed_original_input_rejected(tmp_path):
    fixture = cohort(tmp_path)
    fixture[3]['suffix'] = [999]
    with pytest.raises(ValueError, match='prompt binding'):
        fixed_requests(fixture, tmp_path)


def test_incomplete_output_rejected(tmp_path):
    fixture = cohort(tmp_path)
    path = tmp_path/'3-1.json'
    data = json.loads(path.read_text())
    data['response']['choices'][0]['token_ids'].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='64-token'):
        fixed_requests(fixture, tmp_path)


def test_missing_session_rejected(tmp_path):
    fixture = cohort(tmp_path)
    with pytest.raises(ValueError, match='eight'):
        fixed_requests(fixture[:-1], tmp_path)
