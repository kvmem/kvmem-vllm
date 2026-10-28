"""Exercise an already running, owned loopback SystemOne service.

This client never starts or stops a service. Responses are model-backed only
when the supplied service is; no mock engine is installed by this script.
"""
import argparse
import asyncio
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx


def origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1', 'localhost')
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment):
        raise ValueError('Use an owned HTTP loopback origin without a path or credentials')
    return value.rstrip('/')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_answer(body, payload):
    questions = payload['questions']
    require(body.get('object') == 'structured_decision', 'Unexpected response object')
    require(isinstance(body.get('id'), str) and body['id'], 'Missing decision ID')
    require(body.get('model') == payload['model'], 'Unexpected served model')
    require(set(body.get('answers', {})) == set(questions), 'Missing or extra answers')
    require(set(body.get('diagnostics', {})) == set(questions), 'Missing diagnostics')
    usage = body.get('usage', {})
    require(type(usage.get('input_tokens')) is int and usage['input_tokens'] > 0,
            'No model input token usage')
    require(type(usage.get('output_tokens')) is int and usage['output_tokens'] == len(questions),
            'Expected one native label read per question')
    observed = {}
    for qid, question in questions.items():
        answer, diagnostic = body['answers'][qid], body['diagnostics'][qid]
        labels = {'yes', 'no'} if question['type'] == 'noul' else set(question['criteria'])
        probs = answer.get('probabilities', {})
        require(answer.get('type') == question['type'], f'{qid}: incorrect question type')
        require(set(probs) == labels and all(probability(p) for p in probs.values()),
                f'{qid}: invalid probability labels or values')
        require(math.isclose(sum(probs.values()), 1, abs_tol=1e-6), f'{qid}: unnormalized probabilities')
        mass, confidence = diagnostic.get('label_mass'), answer.get('confidence')
        require(probability(mass) and probability(confidence), f'{qid}: invalid mass or confidence')
        require(type(diagnostic.get('argmax_is_label')) is bool, f'{qid}: missing argmax diagnostic')
        require(math.isclose(confidence, max(probs.values()) * mass, abs_tol=1e-6),
                f'{qid}: confidence differs from upstream vocabulary probability semantics')
        if question['type'] == 'noul':
            require(probability(answer.get('noul')) and
                    math.isclose(answer['noul'], probs['yes'], abs_tol=1e-6), f'{qid}: invalid noul answer')
            observed[qid] = {'yes_probability': answer['noul'], 'confidence': confidence}
        else:
            choice = answer.get('choice')
            require(choice in probs and math.isclose(probs[choice], max(probs.values()), abs_tol=1e-6),
                    f'{qid}: choice is not a maximum-probability option')
            observed[qid] = {'choice': choice, 'confidence': confidence}
    return observed


def capture_snapshot(directory):
    result = {}
    for name in ('scheduler_resources', 'archive'):
        path = directory / (name + '.json')
        try:
            result[name] = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            result[name] = {'read_error': repr(error)}
    return result


def release_evidence(snapshot, previous, expected_reads, not_before):
    state, archive = snapshot['scheduler_resources'], snapshot['archive']
    failures = []
    for key in ('requests', 'refresh_pending', 'connector_pending', 'owned_blocks'):
        if type(state.get(key)) is not int or state[key] != 0:
            failures.append('scheduler.' + key)
    for key in ('reservations', 'session_views'):
        if state.get(key) != []:
            failures.append('scheduler.' + key)
    if 'session_owners' in state and state['session_owners'] != []:
        failures.append('scheduler.session_owners')
    if not isinstance(state.get('at'), (int, float)) or state['at'] < not_before:
        failures.append('scheduler.freshness')
    if not isinstance(state.get('total_blocks'), int) or state.get('free_blocks') != state['total_blocks'] - 1:
        failures.append('scheduler.free_blocks')
    for key in ('active_requests', 'retiring_requests', 'session_retiring_requests',
                'pending_downloads', 'session_checkpoints', 'session_checkpoint_payload_bytes'):
        if type(archive.get(key)) is not int or archive[key] != 0:
            failures.append('archive.' + key)
    for key in ('cpu_authority_bytes', 'cpu_payload_readers', 'cpu_index_owners', 'cpu_index_bytes'):
        if key in archive and (type(archive[key]) is not int or archive[key] != 0):
            failures.append('archive.' + key)
    admission = state.get('cpu_history_admission')
    if admission is not None and (admission.get('leases') != 0 or admission.get('reserved_bytes') != 0):
        failures.append('scheduler.cpu_history_admission')
    before_ids = {row.get('request_id') for row in previous['archive'].get('completed', [])}
    reads, releases = set(), set()
    for row in archive.get('completed', []):
        rid = row.get('request_id', '')
        if rid in before_ids:
            continue
        match = re.search(r'kvmem-decision-([0-9a-f]{32})(-release)?(?:-|$)', rid)
        if match:
            (releases if match[2] else reads).add(match[1])
    if len(reads) != expected_reads or reads != releases:
        failures.append('one_fenced_release_per_unique_server_session')
    return {'passed': not failures, 'unsettled': failures, 'read_sessions': sorted(reads),
            'released_sessions': sorted(releases), 'expected_reads': expected_reads}


async def wait_release(args, previous, expected_reads, not_before, name):
    deadline = time.monotonic() + args.timeout
    while True:
        snapshot = capture_snapshot(args.capture)
        evidence = release_evidence(snapshot, previous, expected_reads, not_before)
        if evidence['passed'] or time.monotonic() >= deadline:
            save(args.output / (name + '.json'), snapshot)
            return evidence
        await asyncio.sleep(.02)


def finished_records(directory, not_before):
    try:
        lines = (directory / 'scheduler_finished.jsonl').read_text().splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # A concurrently appended last line can be incomplete.
        if row.get('at', 0) >= not_before:
            records.append(row)
    return records


def active_read(snapshot, salt, not_before):
    state = snapshot['scheduler_resources']
    if state.get('at', 0) < not_before or state.get('requests', 0) <= 0:
        return None
    for owner in state.get('session_owners', []):
        namespace, rid = owner.get('namespace', []), owner.get('request_id', '')
        if (len(namespace) == 2 and namespace[0] == salt and owner.get('phase') != 'idle'
                and re.search(r'kvmem-decision-[0-9a-f]{32}(?:-|$)', rid) and '-release' not in rid):
            return owner
    return None


def cancellation_verdict(owner, records, released):
    if not released['passed']:
        return 'failed', 'Cancelled/completed request did not release all resources'
    if owner is None:
        return 'not_exercised', 'HTTP request finished before a fresh active scheduler owner was observed'
    terminal = [row for row in records if row.get('request_id') == owner['request_id']]
    if any(row.get('status') == 'FINISHED_ABORTED' for row in terminal):
        return 'passed', 'Native scheduler acknowledged ABORTED and the fenced release completed'
    if any(row.get('status') in ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED') for row in terminal):
        return 'not_exercised', 'Native generation completed before the disconnect could abort it'
    return 'failed', 'No native ABORTED or normal terminal record for the observed request'


async def cancel_request(client, args, model):
    """Disconnect only after observing this request's live scheduler lease."""
    salt = 'systemone-cancel-' + uuid4().hex
    question = {'type': 'noul', 'instructions': 'Is the parcel damaged?'}
    repeats = 140
    for attempt in range(8):
        state = 'The parcel is damaged and the customer requests a refund. ' * repeats
        # This matches upstream NextTokenStrategy._content and NoulQuestion.prompt.
        content = state + '\n\nQuestion: Is the parcel damaged?\nyes\nno\nAnswer with yes or no only.'
        token_payload = {'model': model, 'messages': [{'role': 'user', 'content': content}],
                         'add_generation_prompt': True, 'add_special_tokens': False,
                         'chat_template_kwargs': {'enable_thinking': False}}
        response = await client.post('/tokenize', json=token_payload)
        save(args.output / f'cancel-tokenize-{attempt}.request.json', token_payload)
        (args.output / f'cancel-tokenize-{attempt}.response.body').write_bytes(response.content)
        response.raise_for_status()
        count = response.json()['count']
        if type(count) is int and 256 <= count < 1900:
            break
        require(type(count) is int and count >= 1900, 'Cancellation prompt unexpectedly short or invalid')
        repeats = max(1, int(repeats * 1700 / count))
    else:
        raise AssertionError('Unable to keep the rendered cancellation prompt below 2048 tokens')
    payload = {'model': model, 'state': state, 'questions': {'damaged': question},
               'cache_salt': salt, 'request_id': salt, 'chat_template_kwargs': {'enable_thinking': False}}
    save(args.output / 'cancel.request.json', payload)
    previous = capture_snapshot(args.capture)
    begin = time.perf_counter()
    task = asyncio.create_task(client.post('/v1/systemone', json=payload,
                                           headers={'X-Request-Id': salt}))
    owner, active = None, None
    report = {'status': 'running', 'begin': begin, 'rendered_prompt_tokens': count,
              'client_cancelled': False, 'cache_salt': salt}
    try:
        while not task.done() and time.perf_counter() - begin < args.timeout:
            active = capture_snapshot(args.capture)
            candidate = active_read(active, salt, begin)
            if candidate and not any(row.get('request_id') == candidate['request_id']
                                     for row in finished_records(args.capture, begin)):
                owner = candidate
                break
            await asyncio.sleep(.002)
        if owner is not None and not task.done():
            save(args.output / 'cancel-active.json', active)
            report.update(active_owner=owner, disconnect_at=time.perf_counter())
            task.cancel()  # httpx closes the in-flight HTTP connection on cancellation.
            try:
                await task
            except asyncio.CancelledError:
                report['client_cancelled'] = True
        else:
            require(task.done(), 'Timed out before observing a live cancellation target')
            response = await task
            (args.output / 'cancel.response.body').write_bytes(response.content)
            report['http_status'] = response.status_code
            require(response.status_code == 200, 'Cancellation target failed before disconnect')
            report['observed'] = validate_answer(response.json(), payload)
        report['release'] = await wait_release(args, previous, 1, begin, 'capture-after-cancel')
        report['native_terminal_records'] = finished_records(args.capture, begin)
        report['status'], report['reason'] = cancellation_verdict(
            owner, report['native_terminal_records'], report['release'])
        require(report['status'] != 'passed' or report['client_cancelled'],
                'ABORTED record without a client disconnect is not cancellation proof')
    except Exception as error:
        report.update(status='failed', error=repr(error))
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        save(args.output / 'cancel.result.json', report)
    return report


async def run(args, *, transport=None):
    base = origin(args.base_url)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'base_url': base, 'started_at': time.time(), 'cases': [],
              'scope': 'Native HTTP/schema/probability and disposable-session qualification; '
                       'no accuracy threshold, long-context or throughput qualification.',
              'cancellation': {'status': 'not_requested'}}
    save(args.output / 'results.json', report)
    previous = capture_snapshot(args.capture) if args.capture else None
    if previous:
        save(args.output / 'capture-before.json', previous)
    try:
        async with httpx.AsyncClient(base_url=base, timeout=args.timeout, trust_env=False,
                                     follow_redirects=False, transport=transport) as client:
            models = await client.get('/v1/models')
            (args.output / 'models.response.body').write_bytes(models.content)
            models.raise_for_status()
            ids = [row['id'] for row in models.json()['data']]
            model = args.model or ids[0]
            require(model in ids, 'Requested model is not advertised by this server')
            report['model'] = model
            shared_id = 'systemone-qualification-repeated-id'
            questions = {
                'damaged': {'type': 'noul', 'instructions': 'Is the delivered package damaged?',
                            'criteria': {'true': 'The package is damaged.', 'false': 'It is intact.'}},
                'action': {'type': 'choice', 'instructions': 'Choose the action requested by the customer.',
                           'criteria': {'refund': 'Return the payment.', 'replace': 'Send a replacement.',
                                        'keep': 'Keep the item without further action.'}},
            }
            template = {'model': model, 'state': 'The package arrived broken. Please refund my payment.',
                        'questions': questions, 'request_id': shared_id,
                        'cache_salt': 'systemone-local-qualification'}
            first_request = time.perf_counter()

            async def case(name, payload, expected=200):
                begin = time.perf_counter()
                record = {'name': name, 'request': payload, 'expected_status': expected,
                          'headers': {'X-Request-Id': shared_id}, 'begin': begin, 'status': 'running'}
                save(args.output / (name + '.request.json'), record)
                try:
                    response = await client.post('/v1/systemone', json=payload, headers=record['headers'])
                    (args.output / (name + '.response.body')).write_bytes(response.content)
                    record.update(http_status=response.status_code, response_headers=dict(response.headers),
                                  seconds=time.perf_counter() - begin)
                    require(response.status_code == expected,
                            f'{name}: HTTP {response.status_code}, expected {expected}: {response.text[:500]}')
                    body = response.json()
                    require(isinstance(body, dict) and body, 'Missing JSON object response')
                    if expected == 200:
                        record['observed'] = validate_answer(body, payload)
                    record['status'] = 'passed'
                except Exception as error:
                    record.update(status='failed', error=repr(error), seconds=time.perf_counter() - begin)
                report['cases'].append(record)
                save(args.output / (name + '.result.json'), record)
                save(args.output / 'results.json', report)

            await case('short-noul', {**template, 'questions': {'damaged': questions['damaged']}})
            await case('short-choice', {**template, 'questions': {'action': questions['action']}})
            # The full native app maps request validation errors to HTTP 400.
            await case('missing-state', {'model': model, 'questions': questions}, 400)
            await case('unknown-type', {**template, 'questions': {'q': {'type': 'unknown'}}}, 400)
            await case('invalid-choice', {**template, 'questions': {'q': {'type': 'choice', 'criteria': []}}}, 400)
            for i in range(2):
                await case(f'repeated-id-{i}', template)
            await asyncio.gather(*(case(f'concurrent-{i}', {**template,
                'state': f'Order {i}: the item arrived broken; the customer requests a refund.'}) for i in range(4)))
            expected_reads = sum(len(row['request']['questions']) for row in report['cases']
                                 if row['expected_status'] == 200 and row['status'] == 'passed')
            if args.capture:
                report['release'] = await wait_release(args, previous, expected_reads,
                                                       first_request, 'capture-after')
            else:
                report['release'] = {'passed': None, 'reason': 'No capture directory; resource release unverified'}
            require(all(row['status'] == 'passed' for row in report['cases']), 'HTTP cases failed')
            require(report['release']['passed'] is not False, 'Session release evidence did not settle')
            report['status'] = 'passed'
            if getattr(args, 'cancel', False):
                require(args.capture is not None, '--cancel requires an owned capture directory')
                report['cancellation'] = await cancel_request(client, args, model)
                probe_previous = capture_snapshot(args.capture)
                probe_begin = time.perf_counter()
                await case('after-cancel-probe', {**template, 'questions': {'damaged': questions['damaged']}})
                report['probe_release'] = await wait_release(args, probe_previous, 1,
                                                             probe_begin, 'capture-after-probe')
                require(report['cases'][-1]['status'] == 'passed' and report['probe_release']['passed'],
                        'Normal request or resource release failed after cancellation experiment')
                require(report['cancellation']['status'] != 'failed', 'Cancellation experiment failed')
                if report['cancellation']['status'] == 'not_exercised':
                    report['status'] = 'incomplete'
    except Exception as error:
        report.update(status='failed', error=repr(error))
    finally:
        report['finished_at'] = time.time()
        save(args.output / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', required=True, type=Path, help='New evidence directory')
    parser.add_argument('--model', help='Defaults to the first advertised model')
    parser.add_argument('--capture', type=Path, help='Owned service capture directory, read only')
    parser.add_argument('--timeout', type=float, default=180, help='HTTP and retirement timeout in seconds')
    parser.add_argument('--cancel', action='store_true', help='Also test HTTP disconnect against an active '
                        'native request; requires --capture and may report not_exercised')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if args.cancel and args.capture is None:
        parser.error('--cancel requires --capture')
    result = asyncio.run(run(args))
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']),
                      'results': str(args.output.resolve() / 'results.json')}, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 2 if result['status'] == 'incomplete' else 1)


if __name__ == '__main__':
    main()
