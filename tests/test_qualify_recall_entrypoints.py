"""Functional acceptance cannot turn short output or mixed methods into proof."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def qualifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('qualify_recall_entrypoints')


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return NS(ids=list(map(ord, text)))

    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return ''.join(map(chr, ids))


def profile():
    return dict(max_model_len=4096, served_model_name='test', port=12345,
        kvmem=dict(select_budget=256, gen_budget=128, block_tokens=32,
            rolling_active_window=True, session_checkpoint_bytes=1, session_capture_decode=False))


def event(kind, end, *, method='fps4', query=None, prompt=None, request='response-1'):
    value = dict(sample_id='sample', request_id=request+'-native', event_type=kind,
        configured_method=method, actual_method=method, query_source='accepted_public_tail',
        candidate_block_count=16, selected_block_ids=[1, 2], fallback_reason=None,
        query_begin=max(0, end-128), query_end=end, accepted_through=end,
        selected_tokens=64, budget=256)
    if kind == 'query':
        value.update(query_begin=query[0], query_end=query[1], accepted_through=query[0],
            query_processed_through=query[1], public_input_end=prompt, query_source='public_query')
    return value


def test_forced_output_crosses_g_and_preserves_other_sampling(qualifier):
    configured = profile()
    configured['kvmem']['gen_budget'] = 16384
    fixture = dict(sampling=dict(max_tokens=8192, seed=42, temperature=1,
        top_p=.95, top_k=20, stop=['END'], stop_token_ids=[2]))
    original = deepcopy(fixture)
    parameters = qualifier.functional_sampling(fixture, configured)
    assert parameters['max_tokens'] == parameters['min_tokens'] == 16640
    assert parameters['ignore_eos'] and parameters['temperature'] == 1
    assert parameters['top_p'] == .95 and parameters['top_k'] == 20
    assert 'stop' not in parameters and 'stop_token_ids' not in parameters
    assert fixture == original
    assert qualifier.functional_sampling(fixture, configured, followup=True)['max_tokens'] == 64
    configured['kvmem']['rolling_active_window'] = False
    with pytest.raises(ValueError, match='rolling_active_window'):
        qualifier.functional_sampling(fixture, configured)


def test_followup_keeps_entire_accepted_response_and_exact_new_user_span(qualifier):
    original, output = [1, 2, 3], [4, 5, 6, 7]
    tokens, span = qualifier.append_user_query(original, output, Tokenizer())
    assert tokens[:7] == original + output
    assert Tokenizer().decode(tokens[slice(*span)]) == qualifier.FOLLOWUP_QUESTION
    assert '<|im_start|>user\n' in Tokenizer().decode(tokens[7:span[0]])
    assert '<|im_start|>assistant\n<think>\n' in Tokenizer().decode(tokens[span[1]:])


@pytest.mark.parametrize('change,terminal_change,message', [
    ({'finish_reason': 'stop'}, {}, 'stopped'),
    ({'output_ids': [1]}, {}, 'stopped'),
    ({'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}, {}, 'accounting'),
    ({}, {'history_end': 4}, 'accepted history'),
    ({}, {'checkpoint_tokens': 1}, 'non-retaining'),
])
def test_response_audit_requires_full_forced_output_and_accepted_prefix(qualifier, change, terminal_change, message):
    response = dict(finish_reason='length', output_ids=[9, 9],
        usage=dict(prompt_tokens=2, completion_tokens=2))
    terminal = dict(history_end=3, checkpoint_tokens=0)
    response.update(change)
    terminal.update(terminal_change)
    with pytest.raises(ValueError, match=message):
        qualifier.audit_response(response, [1, 2], dict(max_tokens=2), terminal)


def test_all_entrances_require_real_decode_and_distinct_followup_query(qualifier):
    initial, followup = [780, 790], [1200, 1210]
    first = [event('prefill_pressure', 512),
        event('query', 780, query=initial, prompt=800), event('decode_pressure', 1000)]
    second = [event('query', 1200, query=followup, prompt=1220, request='response-2')]
    audit = qualifier.audit_entrypoints(first, second, 'fps4', profile(), initial, followup, 800)
    assert audit['passed'] and audit['functional_only'] and not audit['scored']
    assert audit['event_counts'] == dict(prefill_pressure=1, decode_pressure=1, query=2)
    invalid = deepcopy(first)
    invalid[-1].update(query_begin=472, query_end=600, accepted_through=600)
    with pytest.raises(ValueError, match='before the original public prompt'):
        qualifier.audit_entrypoints(invalid, second, 'fps4', profile(), initial, followup, 800)
    invalid = deepcopy(first)
    invalid[-1]['actual_method'] = 'mean_k'
    with pytest.raises(ValueError, match='Mixed-method'):
        qualifier.audit_entrypoints(invalid, second, 'fps4', profile(), initial, followup, 800)


def setup_client(qualifier, monkeypatch, tmp_path, *, missing_decode=False):
    root, capture, output = tmp_path/'campaign', tmp_path/'capture', tmp_path/'functional'
    (root/'profiles').mkdir(parents=True)
    capture.mkdir()
    configured = profile()
    (root/'profiles/fps4.json').write_text(json.dumps(configured))
    (root/'fixture.json').write_text('{}')
    source = dict(id='original-case', task='niah_single_1', index=0,
        served_query=[780, 790], served_sha256='source-hash')
    fixture = dict(rows=[source], sampling=dict(max_tokens=8192, temperature=1),
        request_timeout_seconds=7200)
    monkeypatch.setattr(qualifier.campaign, 'verify', lambda *args: (fixture, configured))
    monkeypatch.setattr(qualifier.campaign, 'load_tokens', lambda *args: [10]*800)
    monkeypatch.setattr(qualifier.campaign, 'launch_audit', lambda *args: dict(passed=True))
    calls, imports, terminals = [], [], {}
    monkeypatch.setattr(qualifier.campaign, 'qualify_events', lambda args: imports.append(args))

    def stream(actual_profile, session, tokens, span, sampling, timeout):
        assert actual_profile == configured and timeout == 7200
        count = sampling['max_tokens']
        response_id = 'response-' + str(len(calls)+1)
        values = [event('prefill_pressure', 512, request=response_id),
            event('query', span[0], query=span, prompt=len(tokens), request=response_id)]
        if not calls and not missing_decode:
            values.append(event('decode_pressure', len(tokens)+200, request=response_id))
        with (capture/'recall_events.jsonl').open('a') as file:
            for value in values:
                value['sample_id'] = session
                file.write(json.dumps(value)+'\n')
        terminals[response_id] = dict(history_end=len(tokens)+count-1, checkpoint_tokens=0)
        calls.append(dict(session=session, tokens=list(tokens), query=list(span), sampling=dict(sampling)))
        return dict(response_id=response_id, output_ids=[11]*count, text='functional text',
            finish_reason='length', usage=dict(prompt_tokens=len(tokens), completion_tokens=count),
            seconds=1., ttft_seconds=.1)

    kwargs = dict(tokenizer=Tokenizer(), stream_fn=stream,
        wait_retired_fn=lambda _, rid: terminals[rid],
        wait_released_fn=lambda *args: dict(released=True))
    return root, capture, output, calls, imports, kwargs


def test_mock_workload_uses_same_session_saves_unscored_evidence_and_imports(qualifier, monkeypatch, tmp_path):
    root, capture, output, calls, imports, kwargs = setup_client(qualifier, monkeypatch, tmp_path)
    result = qualifier.run_client(root, 'fps4', capture, output, **kwargs)
    assert result['passed'] and len(calls) == 2 and calls[0]['session'] == calls[1]['session']
    assert calls[1]['tokens'][:1184] == calls[0]['tokens']+[11]*384
    assert calls[0]['sampling']['max_tokens'] == 384 and calls[1]['sampling']['max_tokens'] == 64
    completed = json.loads((output/'completed.json').read_text())
    assert completed['functional_only'] and not completed['scored']
    assert len(imports) == 1 and imports[0].events == output/'functional-events.jsonl'
    assert not (root/'arms').exists()
    assert (output/'long_generation/response.json').exists()
    assert (output/'followup_query/input.i32').stat().st_size == len(calls[1]['tokens'])*4
    with pytest.raises(FileExistsError):
        qualifier.run_client(root, 'fps4', capture, output, **kwargs)
    assert len(calls) == 2


def test_missing_decode_preserves_response_and_stops_before_followup_or_proof(qualifier, monkeypatch, tmp_path):
    root, capture, output, calls, imports, kwargs = setup_client(
        qualifier, monkeypatch, tmp_path, missing_decode=True)
    with pytest.raises(ValueError, match='entrances'):
        qualifier.run_client(root, 'fps4', capture, output, **kwargs)
    assert len(calls) == 1 and not imports
    assert (output/'long_generation/response.json').exists()
    failure = json.loads((output/'failure.json').read_text())
    assert failure['preserved_completed_phases'] == ['long_generation']
    assert not (output/'completed.json').exists()
