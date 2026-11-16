"""Agent bridge request records saved as deltas (core/request_records.py)."""
import copy
import json

import pytest

from vllm_kvmem.core import request_records as rr


def conversation(turns):
    """Anthropic-style requests that grow by one exchange per turn, moving the cache marker like Claude Code."""
    system = [dict(type='text', text='system prompt ' * 200)]
    messages, out = [dict(role='user', content=[dict(type='text', text='task')])], []
    for turn in range(turns):
        payload = dict(model='m', system=system, tools=[dict(name='Bash')], max_tokens=1000 + turn % 2,
                       messages=copy.deepcopy(messages))
        payload['messages'][-1] = dict(payload['messages'][-1], cache_control=dict(type='ephemeral'))
        out.append(payload)
        messages += [dict(role='assistant', content=[dict(type='text', text=f'answer {turn}')]),
                     dict(role='user', content=[dict(type='tool_result', content='x' * 50)])]
    return out


def save(root, records, payloads):
    for turn, payload in enumerate(payloads, 1):
        directory = root / f'{turn:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        record = records.record(directory.name, payload, base=f'{turn - 1:05d}' if turn > 1 else None)
        (directory / 'request.json').write_text(json.dumps(record))
    return root


def test_turns_after_the_first_store_only_their_changes(tmp_path):
    payloads = conversation(40)
    root = save(tmp_path / 'owner', rr.RequestRecords(), payloads)
    first = json.loads((root / '00001' / 'request.json').read_text())
    assert first == payloads[0]                                     # the first turn is the original format
    later = json.loads((root / '00040' / 'request.json').read_text())
    assert later['format'] == rr.FORMAT and later['base'] == '00039'
    assert set(later['changed']) == {'max_tokens'} and later['removed'] == []
    full = sum(len(json.dumps(p)) for p in payloads)
    saved = sum((root / f'{t:05d}' / 'request.json').stat().st_size for t in range(1, 41))
    assert saved * 5 < full
    for turn, payload in enumerate(payloads, 1):                    # sequential and random-order reads
        assert rr.load_request(root / f'{turn:05d}' / 'request.json') == payload
    assert rr.load_request(root / '00007' / 'request.json') == payloads[6]


def test_retries_and_unknown_bases_stay_consistent(tmp_path):
    payloads = conversation(4)
    records, root = rr.RequestRecords(), tmp_path / 'owner'
    save(root, records, payloads[:2])
    # A failed attempt of turn 3 is renamed aside and the turn is sent again (dsh bridge).
    failed = dict(payloads[2], max_tokens=7)
    (root / '00003').mkdir()
    (root / '00003' / 'request.json').write_text(json.dumps(records.record('00003', failed, base='00002')))
    (root / '00003').rename(root / '00003-attempt-1')
    (root / '00003').mkdir()
    (root / '00003' / 'request.json').write_text(json.dumps(records.record('00003', payloads[2], base='00002')))
    assert rr.load_request(root / '00003-attempt-1' / 'request.json') == failed
    assert rr.load_request(root / '00003' / 'request.json') == payloads[2]
    # A bridge restarted mid-conversation has no base in memory: it saves the whole request.
    restarted = rr.RequestRecords().record('00004', payloads[3], base='00003')
    assert restarted == payloads[3]
    # The caller mutating its payload afterwards does not change what the next delta is based on.
    payload = copy.deepcopy(payloads[3])
    records.record('00004', payload, base='00003')
    payload['messages'].clear()
    assert records.record('00005', payloads[3], base='00004')['keep_messages'] == len(payloads[3]['messages'])


def test_corrupt_or_missing_bases_are_refused(tmp_path):
    payloads = conversation(3)
    root = save(tmp_path / 'owner', rr.RequestRecords(), payloads)
    rr._cache.clear()
    record = json.loads((root / '00003' / 'request.json').read_text())
    (root / '00003' / 'request.json').write_text(json.dumps(dict(record, messages=record['messages'][:-1])))
    with pytest.raises(ValueError):
        rr.load_request(root / '00003' / 'request.json')
    rr._cache.clear()
    (root / '00001' / 'request.json').unlink()
    with pytest.raises(FileNotFoundError):
        rr.load_request(root / '00002' / 'request.json')


def test_requests_without_messages_are_saved_whole():
    records = rr.RequestRecords()
    records.record('00001', dict(prompt=[1, 2]))
    assert records.record('00002', dict(prompt=[1, 2, 3]), base='00001') == dict(prompt=[1, 2, 3])
