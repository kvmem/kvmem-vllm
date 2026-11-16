"""Saved benchmark records keep what analyses use without copying prompts or server logs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_client_io as io  # noqa: E402


def _response(prompt):
    return dict(id='cmpl-1', usage=dict(prompt_tokens=len(prompt), completion_tokens=2),
                choices=[dict(index=0, text='ok', token_ids=[7, 8], finish_reason='stop', prompt_token_ids=prompt)])


def test_compact_response_drops_only_the_echoed_prompt():
    prompt = list(range(1000))
    full = _response(prompt)
    saved = io.compact_response(full, prompt)
    choice = saved['choices'][0]
    assert 'prompt_token_ids' not in choice
    assert choice['prompt_token_count'] == 1000
    assert choice['prompt_token_sha256'] == io.token_digest(prompt)
    assert {k: v for k, v in choice.items() if not k.startswith('prompt_token')} == \
        {k: v for k, v in full['choices'][0].items() if k != 'prompt_token_ids'}
    assert saved['usage'] == full['usage'] and saved['id'] == full['id']
    assert 'prompt_token_ids' in full['choices'][0]            # the caller's object is untouched
    assert len(json.dumps(saved)) < len(json.dumps(full)) / 10
    assert io.prompt_matches(choice, prompt) and io.prompt_matches(full['choices'][0], prompt)
    assert not io.prompt_matches(choice, prompt[:-1]) and not io.prompt_matches(choice, prompt[::-1])


def test_mismatched_echo_and_streamed_responses_are_kept_as_is():
    prompt = [1, 2, 3]
    wrong = _response([1, 2, 4])
    assert io.compact_response(wrong, prompt) is wrong        # evidence stays for the caller's check
    streamed = dict(output_ids=[5], text='x')
    assert io.compact_response(streamed, prompt) is streamed


def test_digest_matches_benchmark_memory():
    # Saved digests must agree with the runners' existing token_sha256 values.
    import hashlib
    tokens = [3, 1, 4, 1, 5]
    assert io.token_digest(tokens) == hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def _log(path, rows):
    with path.open('a') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def test_capture_reference_reads_back_the_requests_events(tmp_path):
    capture = tmp_path / 'run' / 'server' / 'capture'
    capture.mkdir(parents=True)
    log = capture / 'layerwise_query.jsonl'
    _log(log, [dict(event='old', request_id='r0-1')])
    with log.open('a') as stream:
        stream.write('{"partial": ')                            # earlier traffic's unfinished line
    slice_ = io.CaptureSlice([log])
    with log.open('a') as stream:
        stream.write('1}\n')
    _log(log, [dict(event='layer_selected', request_id='r1-0', layer=0),
               dict(event='pressure_selected', request_id='other-0'),
               dict(event='pressure_selected', request_id='r1-0', layers=[dict(layer=0, selected_starts=[0, 32])])])
    events = [e for e in slice_.read(log) if e['request_id'].startswith('r1-')]
    reference = slice_.reference(log, 'r1-', events)
    assert reference['count'] == 2 and reference['capture_file'] == 'layerwise_query.jsonl'
    assert io.capture_events(reference) == events
    # Later traffic does not change what the reference names.
    _log(log, [dict(event='pressure_selected', request_id='r1-9')])
    assert io.capture_events(reference) == events
    case = tmp_path / 'run' / 'answers' / 'cases' / 'c0'
    case.mkdir(parents=True)
    result = dict(layerwise=dict(layers=[], events=reference))
    assert io.result_events(result, case) == events
    # A moved run resolves the log from the case directory.
    moved = dict(reference, capture_dir=str(tmp_path / 'gone'))
    assert io.result_events(dict(layerwise=dict(events=moved)), case) == events
    # Older results kept the copy inline.
    assert io.result_events(dict(layerwise=dict(events=events))) == events
    log.write_text('')                                          # truncated log: refuse, never guess
    with pytest.raises(ValueError):
        io.capture_events(reference)
