"""Test the HTTP qualification client without opening a socket or using a GPU."""
import asyncio
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
import pytest


spec = importlib.util.spec_from_file_location('qualify_systemone',
    Path(__file__).resolve().parents[1] / 'scripts/qualify_systemone.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def response_for(payload):
    answers, diagnostics = {}, {}
    for qid, question in payload['questions'].items():
        labels = ['yes', 'no'] if question['type'] == 'noul' else list(question['criteria'])
        probs = dict.fromkeys(labels, 1 / len(labels))
        answers[qid] = {'type': question['type'], 'probabilities': probs,
                        'confidence': .8 / len(labels)}
        answers[qid].update({'noul': probs['yes']} if question['type'] == 'noul'
                            else {'choice': labels[-1]})
        diagnostics[qid] = {'label_mass': .8, 'argmax_is_label': True}
    return {'id': 'decision-repeated', 'model': payload['model'], 'object': 'structured_decision',
            'usage': {'input_tokens': 42, 'output_tokens': len(answers)},
            'answers': answers, 'diagnostics': diagnostics}


def test_run_saves_raw_responses_and_exercises_reused_ids(tmp_path):
    observed = []

    async def handle(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'model'}]})
        payload = json.loads(request.content)
        observed.append((payload, request.headers['X-Request-Id']))
        if 'state' not in payload:
            return httpx.Response(400, json={'error': 'state required'})
        for question in payload['questions'].values():
            if question['type'] == 'unknown' or question.get('criteria') == []:
                return httpx.Response(400, json={'error': 'invalid question'})
        return httpx.Response(200, json=response_for(payload))

    output = tmp_path / 'evidence'
    args = SimpleNamespace(base_url='http://127.0.0.1:8888', output=output,
                           model=None, capture=None, timeout=5)
    report = asyncio.run(client.run(args, transport=httpx.MockTransport(handle)))
    assert report['status'] == 'passed'
    assert len(report['cases']) == len(observed) == 11
    assert len(list(output.glob('*.response.body'))) == 12
    assert len({header for _, header in observed}) == 1
    assert len({p['request_id'] for p, _ in observed if 'request_id' in p}) == 1
    assert report['release']['passed'] is None


@pytest.mark.parametrize('mutation', [
    lambda answer: answer['probabilities'].update(yes=float('nan')),
    lambda answer: answer['probabilities'].update(yes=.2),
    lambda answer: answer.update(confidence=.99),
    lambda answer: answer.update(noul=.1),
])
def test_invalid_model_probabilities_are_failures(mutation):
    payload = {'model': 'model', 'questions': {'q': {'type': 'noul'}}}
    body = response_for(payload)
    mutation(body['answers']['q'])
    with pytest.raises(AssertionError):
        client.validate_answer(body, payload)


def test_capture_requires_fresh_idle_state_and_matching_release():
    identity = 'kvmem-decision-' + 'a' * 32
    state = dict(requests=0, refresh_pending=0, connector_pending=0, owned_blocks=0,
                 reservations=[], session_views=[], at=10, total_blocks=5, free_blocks=4)
    archive = dict(active_requests=0, retiring_requests=0, session_retiring_requests=0,
                   pending_downloads=0, session_checkpoints=0, session_checkpoint_payload_bytes=0,
                   completed=[{'request_id': identity + '-0'}, {'request_id': identity + '-release-0'}])
    snapshot = {'scheduler_resources': state, 'archive': archive}
    previous = {'archive': {'completed': []}}
    assert client.release_evidence(snapshot, previous, 1, 9)['passed']
    assert not client.release_evidence(snapshot, previous, 1, 11)['passed']
    archive['completed'].pop()
    assert not client.release_evidence(snapshot, previous, 1, 9)['passed']
    archive['completed'].append({'request_id': identity + '-release-0'})
    state['owned_blocks'] = 1
    assert not client.release_evidence(snapshot, previous, 1, 9)['passed']


@pytest.mark.parametrize('url', ['https://example.com', 'http://example.com',
                               'http://localhost/path', 'http://user@localhost'])
def test_external_destinations_rejected(url):
    with pytest.raises(ValueError):
        client.origin(url)


def test_client_disconnect_alone_is_not_cancellation_proof():
    owner = {'request_id': 'native-read'}
    released = {'passed': True}
    assert client.cancellation_verdict(owner, [], released)[0] == 'failed'
    normal = [{'request_id': 'native-read', 'status': 'FINISHED_LENGTH_CAPPED'}]
    assert client.cancellation_verdict(owner, normal, released)[0] == 'not_exercised'
    aborted = [{'request_id': 'native-read', 'status': 'FINISHED_ABORTED'}]
    assert client.cancellation_verdict(owner, aborted, released)[0] == 'passed'
    assert client.cancellation_verdict(owner, aborted, {'passed': False})[0] == 'failed'
    assert client.cancellation_verdict(None, aborted, released)[0] == 'not_exercised'


def test_active_target_requires_fresh_matching_namespace():
    identity = 'kvmem-decision-' + 'a' * 32
    owner = {'namespace': ['salt', identity], 'request_id': identity + '-0', 'phase': 'active'}
    snapshot = {'scheduler_resources': {'at': 20, 'requests': 1, 'session_owners': [owner]}}
    assert client.active_read(snapshot, 'salt', 19) == owner
    assert client.active_read(snapshot, 'wrong-salt', 19) is None
    assert client.active_read(snapshot, 'salt', 21) is None
    owner['request_id'] = identity + '-release-0'
    assert client.active_read(snapshot, 'salt', 19) is None


@pytest.mark.parametrize('too_fast', [False, True])
def test_cancel_waits_for_native_abort_and_release_or_reports_not_exercised(tmp_path, too_fast):
    output, capture = tmp_path / 'output', tmp_path / 'capture'
    output.mkdir()
    capture.mkdir()
    identity = 'kvmem-decision-' + 'b' * 32
    state = dict(at=time.perf_counter(), requests=0, refresh_pending=0, connector_pending=0,
                 owned_blocks=0, reservations=[], session_views=[], total_blocks=10, free_blocks=9)
    archive = dict(active_requests=0, retiring_requests=0, session_retiring_requests=0,
                   pending_downloads=0, session_checkpoints=0, session_checkpoint_payload_bytes=0,
                   completed=[])

    def publish():
        state['at'] = time.perf_counter()
        client.save(capture / 'scheduler_resources.json', state)
        client.save(capture / 'archive.json', archive)

    publish()

    async def handle(request):
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 1600})
        payload = json.loads(request.content)
        state.update(requests=1, session_owners=[{'namespace': [payload['cache_salt'], identity],
                                                'request_id': identity + '-0', 'phase': 'active'}])
        publish()
        try:
            if not too_fast:
                await asyncio.sleep(60)
            return httpx.Response(200, json=response_for(payload))
        finally:
            status = 'FINISHED_LENGTH_CAPPED' if too_fast else 'FINISHED_ABORTED'
            state.update(requests=0, session_owners=[])
            archive['completed'] = [{'request_id': identity + '-0'},
                                    {'request_id': identity + '-release-0'}]
            (capture / 'scheduler_finished.jsonl').write_text(json.dumps({
                'request_id': identity + '-0', 'status': status, 'at': time.perf_counter()}) + '\n')
            publish()

    async def exercise():
        args = SimpleNamespace(output=output, capture=capture, timeout=1)
        async with httpx.AsyncClient(base_url='http://localhost', transport=httpx.MockTransport(handle)) as http:
            return await client.cancel_request(http, args, 'model')

    report = asyncio.run(exercise())
    assert report['status'] == ('not_exercised' if too_fast else 'passed')
    assert report['client_cancelled'] is (not too_fast)
    assert report['rendered_prompt_tokens'] == 1600
    assert report['release']['passed']
