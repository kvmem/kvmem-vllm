from dataclasses import dataclass
from types import SimpleNamespace as NS
import gc
import weakref

import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.capture_runtime import CaptureRuntime, CapturedPacket
from vllm_kvmem.serving.session_control import SessionBinding
from vllm_kvmem.serving.thinking_history import ThinkingHistory, thinking_token_layout


MARKERS = [100, 101, 102, 103]
HEADERS = [[10, 20], [11, 20], [12, 20], [13, 20], [14, 20]]
PREFIX = [100, 13, 20, 102]
LAYOUT = MARKERS, HEADERS


class Tokenizer:
    markers = dict(zip(('<|im_start|>', '<|im_end|>', '<think>', '</think>'), MARKERS))
    roles = dict(zip(('system', 'developer', 'user', 'assistant', 'tool'), HEADERS))

    def get_chat_template(self):
        return ' '.join(self.markers)

    def convert_tokens_to_ids(self, value):
        return self.markers[value]

    def encode(self, value, add_special_tokens):
        assert add_special_tokens is False
        if value in self.markers:
            return [self.markers[value]]
        role = value.removeprefix('<|im_start|>').removesuffix('\n')
        return [100, *self.roles[role]]


class Scope:
    def __init__(self):
        self.calls = []
        self.failures = 0
        self.turn_of = {}
        self.completed = []

    def publish_turns(self, completed):
        self.completed.append(completed)

    def publish_thinking(self, lease, thinking, known, turn=0):
        self.calls.append((lease.handle.slot, lease.handle.generation, thinking, known))
        self.turn_of[lease.handle.slot, lease.handle.generation] = turn
        if self.failures:
            self.failures -= 1
            return False
        return True


class Lease:
    def __init__(self, handle):
        self.handle = handle
        self.released = False

    def reset(self):
        self.released = True


class Repository:
    def __init__(self):
        self.retired = set()
        self.leases = []

    def pin(self, handle):
        if (handle.slot, handle.generation) in self.retired:
            return None
        lease = Lease(handle)
        self.leases.append(lease)
        return lease


@dataclass
class Record:
    handle: object
    history_begin: int
    count: int = 0
    archived: bool = False


def binding(tokens=PREFIX, *, history=0, rid='r', generation=1, spans=None):
    return SessionBinding(('salt', 'session'), generation, rid, tuple(tokens), history=history,
        message_spans=((0, len(tokens), 'assistant'),) if spans is None else tuple(spans))


def append(history, repo, records, b, begin, tokens, *, positions=None, generation=1):
    rows, offset = [], 0
    while offset < len(tokens):
        position = begin + offset
        start = position // history.block_tokens * history.block_tokens
        record = records.setdefault(start, Record(NS(slot=start // history.block_tokens,
            generation=generation), start))
        at = position - start
        count = min(history.block_tokens-at, len(tokens)-offset)
        record.count = at + count
        record.archived = record.count == history.block_tokens
        rows.append(((record.handle.slot, generation), record, at, offset, count))
        offset += count
    history.note_append(begin, len(tokens), rows)
    history.accept(b, begin, len(tokens), tokens,
        positions if positions is not None else list(range(begin, begin+len(tokens))), repo)


def test_token_layout_uses_real_encodings_and_fails_closed():
    assert thinking_token_layout(Tokenizer()) == LAYOUT
    unsupported = Tokenizer()
    unsupported.get_chat_template = lambda: 'another template'
    assert thinking_token_layout(unsupported) is None
    bad = Tokenizer()
    bad.convert_tokens_to_ids = lambda _: 42
    assert thinking_token_layout(bad) is None


def test_token_roles_prefix_partial_headers_and_mixed_blocks():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 64)
    b = binding()
    append(history, repo, records, b, 0, PREFIX[:2])
    append(history, repo, records, b, 2, PREFIX[2:])
    append(history, repo, records, b, 4, [1, 2, 3, 4])
    append(history, repo, records, b, 8, [5, 103, 6, 7])
    assert [call[2:] for call in scope.calls] == [(0, 4), (4, 4), (1, 4)]
    assert history.stats['pure_blocks'] == 1
    assert history.pending_tokens == 0 and not history.pending
    assert all(lease.released for lease in repo.leases)


def test_user_quoted_think_markers_and_unknown_template_are_not_thinking():
    for layout in (LAYOUT, None):
        scope, repo, records = Scope(), Repository(), {}
        history = ThinkingHistory(scope, layout, 4, 64)
        tokens = [100, 12, 20, 102, 8, 9, 103, 101]
        append(history, repo, records, binding(tokens, spans=((0, 8, 'user'),)), 0, tokens)
        assert all(call[2] == 0 for call in scope.calls)
        assert history.stats['unknown_tokens'] == (8 if layout is None else 0)


def test_template_disabled_thinking_and_tool_continuation():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 128)
    sequence = [*PREFIX, 103, 1, 2, 3, 101, 100, 14, 20,
                102, 4, 5, 103, 101, *PREFIX, 6, 7, 8]
    append(history, repo, records, binding(sequence), 0, sequence)
    # Markers inside tool output did not create a thinking body; only the last
    # fresh assistant prefix enables the final three generated rows.
    assert history.stats['thinking_tokens'] == 3


@pytest.mark.parametrize('role', ['user', 'tool', 'system', 'developer'])
def test_outer_role_blocks_nested_assistant_headers_and_cross_packet_contamination(role):
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 128)
    outer = [100, 12, 20, 7, *PREFIX, 8, 9, 10, 101, 20]
    tokens = outer + PREFIX + [20]
    # Use locality_spans as the real benchmark bridge does. The last canonical
    # assistant prefix is outside the exact outer user/tool range.
    b = SessionBinding(('salt', 'session'), 1, 'r', tuple(tokens),
        locality_spans=((0, len(outer), role),))
    append(history, repo, records, b, 0, tokens[:7])
    append(history, repo, records, b, 7, tokens[7:len(outer)])
    assert history.stats['thinking_tokens'] == 0 and not history.parser.thinking
    append(history, repo, records, b, len(outer), tokens[len(outer):])
    assert history.generation_allowed and history.parser.thinking
    before = history.stats['thinking_tokens']
    append(history, repo, records, b, len(tokens), [7] * 64)
    assert history.stats['thinking_tokens'] - before == 64


def test_raw_prompt_without_outer_roles_never_seeds_thinking():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 128)
    tokens = PREFIX + [7] * 4
    b = binding(tokens, spans=())
    append(history, repo, records, b, 0, tokens)
    append(history, repo, records, b, len(tokens), [7] * 8)
    assert not history.generation_allowed
    assert history.stats['thinking_tokens'] == 0
    assert history.stats['unknown_tokens'] == 16
    assert all(call[2:] == (0, 0) for call in scope.calls)


def test_verified_prefix_seed_obeys_outer_role_instead_of_nested_fake_header():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 64)
    tokens = [100, 12, 20, 7, *PREFIX]
    b = binding(tokens, history=len(tokens), spans=((0, len(tokens), 'user'),))
    append(history, repo, records, b, len(tokens), [7] * 4)
    assert not history.parser.thinking and not history.generation_allowed
    assert scope.calls[-1][2:] == (0, 0)


def test_published_generation_retries_without_owning_dead_record():
    scope, repo, records = Scope(), Repository(), {}
    scope.failures = 2
    history = ThinkingHistory(scope, LAYOUT, 4, 64)
    append(history, repo, records, binding(history=4), 4, [1, 2, 3, 4])
    assert len(history.publications) == 1
    history.retry(repo, limit=1)
    assert len(history.publications) == 1
    history.retry(repo, limit=1)
    assert not history.publications and scope.calls[-1][2:] == (4, 4)
    scope.failures = 1
    append(history, repo, records, binding(history=4), 8, [5, 6, 7, 8])
    ref = weakref.ref(records.pop(8))
    gc.collect()
    assert ref() is None
    history.retry(repo)
    assert not history.publications


def test_rewind_reseeds_only_verified_prefix_and_rejects_stale_session():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 64)
    append(history, repo, records, binding(), 0, PREFIX + [1, 2, 3, 4])
    history.rewind(4)
    records.pop(4)
    append(history, repo, records, binding(history=4, rid='next'), 4, [103, 7, 8, 9], generation=2)
    assert scope.calls[-1][1:] == (2, 0, 4)
    history.rewind(4)
    records.pop(4)
    append(history, repo, records, binding(history=0, rid='no-proof'), 4, [1, 2, 3, 4], generation=3)
    assert scope.calls[-1][1:] == (3, 0, 0)
    with pytest.raises(RuntimeError, match='session generation'):
        append(history, repo, records, binding(generation=2), 8, [1, 2, 3, 4])


def test_position_mismatch_is_unknown_and_pending_is_bounded():
    scope, repo, records = Scope(), Repository(), {}
    history = ThinkingHistory(scope, LAYOUT, 4, 8)
    append(history, repo, records, binding(history=4), 4, [1, 2, 3, 4], positions=[4, 5, 7, 8])
    assert scope.calls[-1][2:] == (0, 0)
    record = Record(NS(slot=0, generation=1), 0, 4, True)
    history.note_append(8, 4, [(None, record, 0, 0, 4)])
    history.note_append(12, 4, [(None, record, 0, 0, 4)])
    with pytest.raises(RuntimeError, match='bounded'):
        history.note_append(16, 4, [(None, record, 0, 0, 4)])


def test_sidecar_uses_device_starts_and_does_not_commit_rejected_close_marker():
    b = binding(history=4)
    scope, repo = Scope(), Repository()
    history = ThinkingHistory(scope, LAYOUT, 4, 64)
    record = Record(NS(slot=1, generation=1), 4, 1, False)
    history.note_append(4, 1, [(None, record, 0, 0, 1)])
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.thinking_enabled = True
    runtime.thinking_layout = LAYOUT
    runtime.thinking_capture_stats = dict(packets=0, bytes=0, accepted_rows=0)
    runtime.runner = NS(_kvmem_sessions=NS(bindings={'r': b}, release_requests=set()))
    runtime.authorities = {'r': NS(thinking_history=history, memory=NS(repository=repo))}
    batch = NS(req_ids=['other', 'r'], query_start_loc_np=np.array([0, 2, 5]),
        query_start_loc=torch.tensor([0, 1, 4]), num_tokens=5,
        input_ids=torch.tensor([999, 7, 103, 8, 999]), positions=torch.tensor([-1, 4, 5, 6, -1]))
    payload, descriptors = runtime._capture_thinking_tokens(torch.zeros(48, dtype=torch.uint8), batch, (1,))
    packet = CapturedPacket(('r',), 3, NS(payload=payload), thinking_tokens=descriptors)
    runtime._accept_thinking_tokens(packet, 'r', 4, 1)
    assert history.parser.thinking and history.stats['thinking_tokens'] == 1
    assert runtime.thinking_capture_stats['accepted_rows'] == 1
    assert record._thinking_counts == (1, 1, 1)
    # The rejected </think> is absent from parser state. The next actual input
    # continues thinking rather than being labelled from sampled output IDs.
    record.count = 4
    record.archived = True
    history.note_append(5, 3, [(None, record, 1, 0, 3)])
    history.accept(b, 5, 3, [9, 10, 11], [5, 6, 7], repo)
    assert scope.calls[-1][2:] == (4, 4)
    report = runtime._thinking_progress()
    assert report['layout_supported'] and report['active_owners'] == 1
    assert report['owner_stats']['accepted_tokens'] == report['owner_stats']['thinking_tokens'] == 4
    assert report['owner_stats']['pure_blocks'] == 1


def test_real_authority_seal_before_token_publication_and_native_counts(tmp_path):
    import time
    from vllm_kvmem.serving.authority import RequestAuthority
    from vllm_kvmem.core.config import KVMemConfig
    from vllm_kvmem.core.memory_storage import MemoryArchive

    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 2*32*8*2, 64, native_payloads=True)
    settings = _native.HistoricalGcConfig()
    settings.policy, settings.policy_params = 'redundancy', {'thinking_only': True}
    settings.mode, settings.background = _native.HistoricalGcMode.SHADOW, False
    settings.ready_low, settings.ready_high = 2, 4
    worker = archive.enable_history_gc(settings)
    authority = RequestAuthority('thinking', config, archive, 2, 1, 8,
        raw_dtype=torch.float8_e4m3fn)
    scope = authority.gc.scope
    authority.thinking_history = history = ThinkingHistory(scope, LAYOUT, 32, 512)
    tokens = PREFIX + [7] * 60 + [103]
    raw = torch.ones(2, len(tokens), 1, 8).to(torch.float8_e4m3fn)
    values = torch.zeros(raw.shape, dtype=torch.uint8)
    try:
        assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
        authority.publish_sealed(lambda: None)
        assert scope.stats['thinking_classified'] == 0
        assert history.pending_tokens == 65
        history.accept(binding(tokens), 0, 65, tokens, list(range(65)), authority.memory.repository)
        authority.poll()
        assert history.pending_tokens == 0
        assert scope.stats['thinking_classified'] == 2
        assert scope.stats['thinking_pure'] == 1
        assert history.stats['thinking_tokens'] == 60
    finally:
        deadline = time.monotonic() + 5
        while not authority.close():
            worker.advance()
            assert time.monotonic() < deadline
            time.sleep(.001)
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
    assert archive.io_usage()['cpu_index_bytes'] == archive.io_usage()['cpu_authority_bytes'] == 0
