"""Opt-in thinking age rule: turn ordinals, completed turns, config and native retirement."""
from types import SimpleNamespace as NS
import time

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.core.historical_gc import HistoricalGcSettings
from vllm_kvmem.serving.session_control import SessionBinding
from vllm_kvmem.serving.thinking_history import ThinkingHistory

from test_historical_thinking_capture import LAYOUT, Record, Repository, Scope, binding


START, END, THINK, UNTHINK, NEWLINE = 100, 101, 102, 103, 20
HEADER = {'user': [12, 20], 'assistant': [13, 20], 'tool': [14, 20]}


def message(role, body):
    return [START, *HEADER[role], *body, END, NEWLINE]


def assistant(thinking):
    return message('assistant', [THINK, *[7] * thinking, UNTHINK, 8, 9])


def conversation(turns, thinking=12):
    """user, then (assistant, tool) x turns; spans exclude the separator newline."""
    tokens, spans = [], []
    parts = [('user', message('user', [5, 5, 5]))]
    for _ in range(turns):
        parts += [('assistant', assistant(thinking)), ('tool', message('tool', [6, 6]))]
    for role, part in parts:
        spans.append((len(tokens), len(tokens) + len(part) - 1, role))
        tokens += part
    return tokens, spans


def accept(history, repo, records, b, begin, tokens):
    rows, offset = [], 0
    while offset < len(tokens):
        position = begin + offset
        start = position // history.block_tokens * history.block_tokens
        record = records.setdefault(start, Record(NS(slot=start // history.block_tokens, generation=1), start))
        at = position - start
        count = min(history.block_tokens - at, len(tokens) - offset)
        record.count = at + count
        record.archived = record.count == history.block_tokens
        rows.append(((record.handle.slot, 1), record, at, offset, count))
        offset += count
    history.note_append(begin, len(tokens), rows)
    history.accept(b, begin, len(tokens), tokens, list(range(begin, begin + len(tokens))), repo)


def pure_turns(scope):
    """Turn of every published pure-thinking record, by slot."""
    return {slot: scope.turn_of[slot, generation] for slot, generation, thinking, known in scope.calls
            if thinking == known == 4}


def expected_turn(spans, position):
    ordinal = 0
    for begin, end, role in spans:
        if role == 'assistant':
            ordinal += 1
            if begin <= position < end:
                return ordinal
    return None


def test_records_carry_their_assistant_turn_across_requests():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 4096)
    prompt, spans = conversation(3)
    first = prompt + [START, *HEADER['assistant'], THINK]    # generation header of turn 4
    b = SessionBinding(('salt', 'session'), 1, 'r', tuple(first), message_spans=tuple(spans))
    accept(history, repo, records, b, 0, first)
    accept(history, repo, records, b, len(first), [7] * 12)
    assert scope.completed == [3]
    turns = pure_turns(scope)
    assert set(turns.values()) == {1, 2, 3, 4}
    for slot, turn in turns.items():
        position = slot * 4
        assert turn == (expected_turn(spans, position) if position < len(first) else 4)

    # Next request: turn 4 is complete, a tool result follows, turn 5 starts.
    turn4 = (len(prompt), len(first) + 12 + 3, 'assistant')    # through END, not the newline
    second = first + [7] * 12 + [UNTHINK, 8, END, NEWLINE]
    spans2 = list(spans) + [turn4]
    tool = message('tool', [6, 6])
    spans2.append((len(second), len(second) + len(tool) - 1, 'tool'))
    second += tool + [START, *HEADER['assistant'], THINK]
    b2 = SessionBinding(('salt', 'session'), 1, 'r2', tuple(second), message_spans=tuple(spans2), history=len(first) + 12)
    accept(history, repo, records, b2, len(first) + 12, second[len(first) + 12:])
    accept(history, repo, records, b2, len(second), [7] * 12)
    assert scope.completed == [3, 4]
    later = pure_turns(scope)
    assert {slot: turn for slot, turn in later.items() if slot in turns} == turns    # never renumbered
    assert max(later.values()) == 5


def test_continued_assistant_message_is_not_complete():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 4096)
    prompt, spans = conversation(2)
    tail = assistant(8)[:-2]    # an open assistant message the prompt continues
    spans = list(spans) + [(len(prompt), len(prompt) + len(tail), 'assistant')]
    tokens = prompt + tail
    b = SessionBinding(('salt', 'session'), 1, 'r', tuple(tokens), message_spans=tuple(spans))
    accept(history, repo, records, b, 0, tokens)
    assert history.generation_allowed and history.generation_span is None
    assert scope.completed == [2]
    assert history._turn_at(b, len(tokens)) == 3


def test_unknown_boundaries_publish_no_turns():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 4096)
    tokens = [START, 13, 20, THINK, *[7] * 16]
    accept(history, repo, records, SessionBinding(('salt', 'session'), 1, 'r', tuple(tokens)), 0, tokens)
    assert scope.completed == [0]
    assert set(pure_turns(scope).values()) <= {0}


def test_settings_accept_age_only_with_thinking_capture():
    settings = HistoricalGcSettings(mode='delete', policy='redundancy',
                                    policy_params={'thinking_only': True, 'thinking_age_steps': 10})
    assert settings.policy_params['thinking_age_steps'] == 10
    # Absent keeps every existing configuration's expanded parameters unchanged.
    plain = HistoricalGcSettings(policy='redundancy', policy_params={'thinking_only': True})
    assert 'thinking_age_steps' not in plain.policy_params
    for params, policy in (({'thinking_age_steps': 10}, 'lru'),
                           ({'thinking_age_steps': 10}, 'redundancy'),
                           ({'thinking_only': True, 'thinking_age_steps': -1}, 'redundancy'),
                           ({'thinking_only': True, 'thinking_age_steps': True}, 'redundancy')):
        with pytest.raises(ValueError):
            HistoricalGcSettings(policy=policy, policy_params=params)


def test_native_scope_retires_old_thinking_without_pressure(tmp_path):
    from vllm_kvmem.serving.authority import RequestAuthority
    from vllm_kvmem.core.config import KVMemConfig
    from vllm_kvmem.core.memory_storage import MemoryArchive

    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 2*32*8*2, 64, native_payloads=True)
    settings = _native.HistoricalGcConfig()
    settings.policy = 'redundancy'
    settings.policy_params = {'thinking_only': True, 'thinking_age_steps': 1}
    settings.mode, settings.background = _native.HistoricalGcMode.DELETE, False
    settings.ready_low, settings.ready_high = 2, 4
    worker = archive.enable_history_gc(settings)
    authority = RequestAuthority('age', config, archive, 2, 1, 8, raw_dtype=torch.float8_e4m3fn)
    scope = authority.gc.scope
    authority.thinking_history = history = ThinkingHistory(scope, LAYOUT, 32, 512)
    # Turn 1 holds one pure 32-token record (tokens 32..63); turn 2 completes.
    turn1 = [START, 13, 20, THINK, *[7] * 60, UNTHINK, 8, END, NEWLINE]
    user = message('user', [5, 5, 5])
    turn2 = [START, 13, 20, THINK, *[7] * 28, UNTHINK, 8, END, NEWLINE]
    tokens = turn1 + user + turn2 + [START, 13, 20, THINK]
    spans, at = [], 0
    for role, part in (('assistant', turn1), ('user', user), ('assistant', turn2)):
        spans.append((at, at + len(part) - 1, role))
        at += len(part)
    b = SessionBinding(('salt', 'session'), 1, 'r', tuple(tokens), message_spans=tuple(spans))
    raw = torch.ones(2, len(tokens), 1, 8).to(torch.float8_e4m3fn)
    values = torch.zeros(raw.shape, dtype=torch.uint8)
    try:
        assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
        authority.publish_sealed(lambda: None)
        history.accept(b, 0, len(tokens), tokens, list(range(len(tokens))), authority.memory.repository)
        authority.poll()
        assert scope.stats['thinking_pure'] == 1
        deadline = time.monotonic() + 5
        while scope.stats['thinking_age_retired'] < 1:
            worker.advance()
            assert time.monotonic() < deadline
        stats = scope.stats
        assert stats['retired'] == stats['thinking_age_retired'] == 1 and stats['lossy_retired'] == 0
    finally:
        deadline = time.monotonic() + 5
        while not authority.close():
            worker.advance()
            assert time.monotonic() < deadline
            time.sleep(.001)
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
