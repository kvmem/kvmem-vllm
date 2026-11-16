"""Per-turn agent request records saved as deltas.

The agent bridges save each turn's request under ``OWNER/NNNNN/request.json``.
An agent resends the whole conversation every turn, so full copies grow with
the square of the conversation length. A turn whose previous turn was recorded
in this process saves only what changed:

    {"format": "kvmem-request-delta-v1", "base": "00041",
     "keep_messages": 80, "messages": [...messages from index 80...],
     "changed": {...other fields that differ...}, "removed": [...],
     "sha256": "<digest of the whole request>"}

``base`` names the previous turn's directory beside this one. The first turn,
and any turn whose base is not known, is saved in full (the original format),
so every chain ends at a complete record. :func:`load_request` rebuilds and
verifies the whole request and reads full records unchanged.
"""
import hashlib
import json
from pathlib import Path

FORMAT = 'kvmem-request-delta-v1'


def digest(payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def _delta(base_name, previous, payload):
    old, new = previous.get('messages'), payload.get('messages')
    if not isinstance(old, list) or not isinstance(new, list):
        return None
    keep = 0
    for a, b in zip(old, new):
        if a != b:
            break
        keep += 1
    changed = {k: v for k, v in payload.items() if k != 'messages' and (k not in previous or previous[k] != v)}
    removed = sorted(k for k in previous if k not in payload)
    return dict(format=FORMAT, base=base_name, keep_messages=keep, messages=new[keep:],
                changed=changed, removed=removed, sha256=digest(payload))


class RequestRecords:
    """One conversation's saved requests, keeping the last few in memory to diff against."""

    def __init__(self, keep=4):
        self.keep, self.recent = keep, {}

    def record(self, name, payload, base=None):
        """The record to save for turn directory NAME: a delta against BASE when known, else the full request."""
        record = None
        if base is not None and base in self.recent and base != name:
            record = _delta(base, self.recent[base], payload)
        self.recent.pop(name, None)
        self.recent[name] = json.loads(json.dumps(payload))   # detached copy: callers may mutate theirs
        while len(self.recent) > self.keep:
            self.recent.pop(next(iter(self.recent)))
        return payload if record is None else record


_cache = {}


def load_request(path):
    """The whole request saved at PATH (``.../NNNNN/request.json``), full or delta."""
    path = Path(path).resolve()
    chain, current = [], path
    while True:
        if str(current) in _cache:
            payload = json.loads(_cache[str(current)])
            break
        record = json.loads(current.read_text())
        if not (isinstance(record, dict) and record.get('format') == FORMAT):
            payload = record
            break
        chain.append((current, record))
        if len(chain) > 100000:
            raise ValueError('Request delta chain does not end at a full record')
        current = current.parent.parent / record['base'] / current.name
    for current, record in reversed(chain):
        messages = payload.get('messages')
        if not isinstance(messages, list) or len(messages) < record['keep_messages']:
            raise ValueError('Request delta does not fit its base: ' + str(current))
        payload = {k: v for k, v in payload.items() if k not in record['removed']}
        payload.update(record['changed'])
        payload['messages'] = messages[:record['keep_messages']] + record['messages']
        if digest(payload) != record['sha256']:
            raise ValueError('Rebuilt request does not match its recorded digest: ' + str(current))
    _cache.clear()   # sequential readers ask for the next turn; one entry keeps that O(1)
    _cache[str(path)] = json.dumps(payload, ensure_ascii=False)
    return payload
