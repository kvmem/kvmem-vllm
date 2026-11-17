"""Historical GC side data (redundancy Q, thinking inputs) on logical publication.

A physically coalesced owner commits one logical span at a time. Its thinking
labels follow each committed span from the session binding's prompt tokens; its
Question rows follow the completed owner from the shared DMA's trailing bytes.
"""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.serving.capture_runtime import CapturedPacket
from vllm_kvmem.core.historical_gc import RedundancyHistory
from vllm_kvmem.serving.multiowner_prefill import attach_owner_views
from vllm_kvmem.serving.session_control import SessionBinding
from vllm_kvmem.serving.terminal import TerminalPrefix
from vllm_kvmem.serving.thinking_history import ThinkingHistory
from test_historical_thinking_capture import LAYOUT, PREFIX, Repository, append
from test_logical_capture_publication import runtime_fixture
from test_logical_publication import fixture

HEADS = 3
THINKING = 3000


class Scope:
    def __init__(self):
        self.epoch, self.question, self.invalidated, self.thinking = 0, [], 0, []

    def configure_features(self, *shape):
        return True

    def begin_question(self, count):
        self.epoch += 1
        self.question = []
        return self.epoch

    def append_question(self, epoch, offset, rows, count):
        assert epoch == self.epoch and rows.shape[1] == count
        self.question.append((offset, torch.from_numpy(rows.copy())))
        return True

    def observe_query(self, epoch, rows, count):
        return True

    def invalidate_question(self):
        self.invalidated += 1

    def publish_features(self, lease, value):
        return True

    def publish_thinking(self, lease, thinking, known, turn=0):
        self.thinking.append((thinking, known))
        return True

    def publish_turns(self, completed):
        pass


def prompt(rid, length, *, question=None):
    tokens = PREFIX + [7]*THINKING + [103] + [8]*(length-len(PREFIX)-THINKING-1)
    return SessionBinding(('salt', rid), 1, rid, tuple(tokens),
        message_spans=((0, length, 'assistant'),), question_span=question)


def enable(runtime, authorities, bindings, *, verify=False):
    runtime.redundancy_enabled = runtime.thinking_enabled = True
    runtime.redundancy_layers, runtime.redundancy_query_heads = (0,), HEADS
    runtime.redundancy_q_stats = dict(packets=0, bytes=0, overflow_packets=0)
    runtime.thinking_capture_stats = dict(packets=0, bytes=0, accepted_rows=0,
        synthesized_rows=0, synthesized_verified_rows=0)
    runtime.thinking_synth_verify = verify
    runtime.runner._kvmem_sessions = NS(bindings=bindings, release_requests=set())
    scopes = {}
    for rid, authority in authorities.items():
        scope = scopes[rid] = Scope()
        authority.thinking_history = ThinkingHistory(scope, LAYOUT, 32, 1 << 20)
        authority.redundancy_history = RedundancyHistory(scope, 1, 1, HEADS, runtime.dim)
    return scopes


def capture(runtime, payload, bindings, counts, logical):
    """The submit-side captures, on a CPU batch with the forwarded prompt rows."""
    rids = list(bindings)
    starts = np.concatenate(([0], np.cumsum(counts)))
    total = int(starts[-1])
    batch = NS(req_ids=rids, num_computed_tokens_np=np.zeros(len(rids), dtype=np.int64),
        query_start_loc_np=starts, query_start_loc=torch.from_numpy(starts),
        is_prefilling_np=np.ones(len(rids), dtype=bool), num_tokens=total,
        input_ids=torch.tensor([t for rid, n in zip(rids, counts) for t in bindings[rid].tokens[:n]]),
        positions=torch.cat([torch.arange(n) for n in counts]))
    query = (torch.arange(total*HEADS*runtime.dim, dtype=torch.float32) % 251 / 8).reshape(
        total, HEADS, runtime.dim).to(torch.bfloat16)
    base = payload.numel()
    indices = tuple(range(len(rids)))
    payload, redundancy = runtime._capture_redundancy_queries(payload, {0: NS(query=query)}, batch, indices)
    payload, thinking = runtime._capture_thinking_tokens(payload, batch, indices, frozenset(logical))
    return payload, redundancy, thinking, payload.numel()-base, query


def reference(binding, retained):
    history = ThinkingHistory(Scope(), LAYOUT, 32, 1 << 20)
    if retained:
        append(history, Repository(), {}, binding, 0, list(binding.tokens[:retained]))
    return {name: history.stats[name] for name in ('accepted_tokens', 'thinking_tokens', 'unknown_tokens')}


def token_stats(authority):
    stats = authority.thinking_history.stats
    return {name: stats[name] for name in ('accepted_tokens', 'thinking_tokens', 'unknown_tokens')}


@contextmanager
def single(tmp_path, *, question=(2000, 2100), verify=False, slot_bytes=None):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        binding = prompt('r', 4200, question=question)
        env.scope = enable(runtime, {'r': env.authority}, {'r': binding}, verify=verify)['r']
        if slot_bytes is not None:
            runtime.redundancy_q_slot_bytes = slot_bytes
        download = env.packet.download
        download.payload, env.packet.redundancy_queries, env.packet.thinking_tokens, env.packet.aux_bytes, env.query = \
            capture(runtime, download.payload, {'r': binding}, [4096], {'r'})
        env.binding = binding
        yield env


@pytest.mark.parametrize('verify', [False, True])
@pytest.mark.parametrize('retained', [0, 1, 2048, 2049, 4096])
def test_logical_owner_accepts_span_thinking_and_owner_question(tmp_path, retained, verify):
    with single(tmp_path, verify=verify) as env:
        runtime, packet = env.runtime, env.packet
        row = packet.thinking_tokens['r']
        assert row['synthesized'] and ('byte_offset' in row) == verify
        assert packet.aux_bytes == 100*HEADS*2*2 + (4096*16 if verify else 0)
        if retained == 4096:
            runtime.begin_step()
        else:
            runtime.begin_step({'r'}, {'r': TerminalPrefix(retained, 4096, 0, 'FINISHED_ABORTED')})
        assert runtime.drain_requests({'r'}, wait=True)
        assert env.authority.history_end == retained
        # Per-span accepts continue one parser across the 2K child boundary.
        assert token_stats(env.authority) == reference(env.binding, retained)
        assert not env.authority.thinking_history.pending
        assert env.authority.thinking_history.stats['invalidations'] == 0
        stats = runtime.thinking_capture_stats
        assert stats['accepted_rows'] == stats['synthesized_rows'] == retained
        assert stats['synthesized_verified_rows'] == (retained if verify else 0)
        history = env.authority.redundancy_history
        kept = max(0, min(retained, 2100)-2000)
        assert history.stats['question_tokens'] == kept and history.complete == (kept == 100)
        assert not history.invalid and not env.scope.invalidated
        if kept:
            [(offset, rows)] = env.scope.question
            assert offset == 0
            assert torch.equal(rows, env.query[2000:2000+kept].float().unsqueeze(0))
        assert not packet.redundancy_queries and not packet.thinking_tokens
        assert runtime.released == [packet.download]


def test_capacity_retry_accepts_each_span_once(tmp_path, monkeypatch):
    with single(tmp_path) as env:
        runtime = env.runtime
        original, blocked = env.authority._prepare_append, [True]
        def prepare(*args, **kwargs):
            return None if kwargs['history_begin'] == 2048 and blocked[0] else original(*args, **kwargs)
        monkeypatch.setattr(env.authority, '_prepare_append', prepare)
        runtime.begin_step()
        assert not runtime.drain_requests({'r'}, wait=False)
        assert runtime.thinking_capture_stats['accepted_rows'] == 2048
        assert env.authority.redundancy_history.stats['question_tokens'] == 0
        assert set(env.packet.redundancy_queries) == set(env.packet.thinking_tokens) == {'r'}
        blocked[0] = False
        assert runtime.drain_requests({'r'}, wait=False)
        assert token_stats(env.authority) == reference(env.binding, 4096)
        assert env.authority.redundancy_history.complete


def test_side_bytes_are_part_of_the_validated_extent(tmp_path):
    with single(tmp_path) as env:
        env.packet.aux_bytes -= 12
        env.runtime.begin_step()
        with pytest.raises(ValueError, match='invalid byte extent'):
            env.runtime.drain_requests({'r'}, wait=True)
        assert env.authority.history_end == 0 and not env.runtime.released


@pytest.mark.parametrize('name,match', [('redundancy_queries', 'redundancy Q metadata'),
                                        ('thinking_tokens', 'thinking token metadata')])
def test_missing_descriptor_fails_before_any_append(tmp_path, name, match):
    with single(tmp_path) as env:
        getattr(env.packet, name).clear()
        env.runtime.begin_step()
        with pytest.raises(RuntimeError, match=match):
            env.runtime.drain_requests({'r'}, wait=True)
        assert env.authority.history_end == 0 and not env.runtime.released


def test_verify_mode_rejects_a_forwarded_row_that_differs(tmp_path):
    with single(tmp_path, verify=True) as env:
        row = env.packet.thinking_tokens['r']
        at = row['byte_offset']
        env.packet.download.payload[at:at+4096*16].view(torch.int64)[2050] += 1
        env.runtime.begin_step()
        with pytest.raises(RuntimeError, match='differ from the forwarded device rows'):
            env.runtime.drain_requests({'r'}, wait=True)
        # The first child was exact and stays committed; nothing is replayed.
        assert env.runtime.thinking_capture_stats['synthesized_verified_rows'] == 2048
        assert not env.runtime.released


def test_prompt_shorter_than_the_forward_fails_closed(tmp_path):
    with single(tmp_path) as env:
        short = prompt('r', 4000, question=(2000, 2100))
        env.packet.thinking_tokens['r']['binding'] = short
        env.runtime.begin_step()
        with pytest.raises(RuntimeError, match='outside the admitted prompt span'):
            env.runtime.drain_requests({'r'}, wait=True)


def test_question_transport_overflow_invalidates_only_the_question(tmp_path):
    with single(tmp_path, slot_bytes=8) as env:
        runtime = env.runtime
        assert env.packet.aux_bytes == 0 and runtime.redundancy_q_stats['overflow_packets'] == 1
        runtime.begin_step()
        assert runtime.drain_requests({'r'}, wait=True)
        assert env.authority.history_end == 4096
        history = env.authority.redundancy_history
        assert history.invalid and history.stats['transport_overflow'] == 1
        assert token_stats(env.authority) == reference(env.binding, 4096)


def test_synthesized_descriptor_cannot_reach_ordinary_publication(tmp_path):
    with single(tmp_path) as env:
        with pytest.raises(RuntimeError, match='ordinary publication'):
            env.runtime._accept_thinking_tokens(env.packet, 'r', 0, 16)


@contextmanager
def shared(tmp_path, *, ordinary_peer=False, verify=False):
    with runtime_fixture(tmp_path/'first') as a, fixture(tmp_path/'second', request_id='s') as b:
        r = a.runtime
        r.options.prefill_physical_multi_owner = True
        r.authorities['s'] = b.authority
        r.accepted['s'] = r.rejected['s'] = 0
        r.last_steps['s'] = []
        r._snapshot_capacity('s', b.authority)
        b.raw.fill_(3)
        b.values.fill_(27)
        counts = [4096, 2048 if ordinary_peer else 4096]
        total = sum(counts)
        bindings = {'r': prompt('r', 4200, question=(4000, 4096)), 's': prompt('s', 4300, question=(10, 74))}
        scopes = enable(r, {'r': a.authority, 's': b.authority}, bindings, verify=verify)
        header = torch.tensor([0, 4096, total, 0, 0, 0, 0, 0, 0], dtype=torch.int64)
        raw = torch.cat((a.raw, b.raw[:, :counts[1]]), dim=1)
        values = torch.cat((a.values, b.values[:, :counts[1]]), dim=1)
        data = torch.cat((header.view(torch.uint8), raw.view(torch.uint8).flatten(), values.flatten()))
        plans = {'r': (a.group, a.parts)} if ordinary_peer else {'r': (a.group, a.parts), 's': (b.group, b.parts)}
        data, redundancy, thinking, aux, query = capture(r, data, bindings, counts, plans)
        download = NS(payload=data, slot=0, query=lambda: True, event=NS(synchronize=lambda: None))
        parent = CapturedPacket(('r', 's'), total, download, confirmed=False, request_tokens=tuple(counts),
            source_generation=17, redundancy_queries=redundancy, thinking_tokens=thinking, aux_bytes=aux,
            action_inputs={'r': object(), 's': b.parts[0].action_packet(b.state) if ordinary_peer else object()})
        attach_owner_views(parent, plans)
        r.queue, r.downloads.pending, r.released = [parent], {0: download}, []
        def release(d):
            r.released.append(d)
            del r.downloads.pending[d.slot]
            return True
        r.downloads.release = release
        yield NS(runtime=r, parent=parent, a=a, b=b, bindings=bindings, scopes=scopes, query=query, counts=counts)


@pytest.mark.parametrize('verify', [False, True])
@pytest.mark.parametrize('order', [('r', 's'), ('s', 'r')])
def test_shared_source_hands_each_logical_owner_its_own_side_data(tmp_path, order, verify):
    with shared(tmp_path, verify=verify) as env:
        r, parent = env.runtime, env.parent
        # Descriptors move to the owner views; none stays on the parent.
        assert not parent.redundancy_queries and not parent.thinking_tokens
        for index, rid in enumerate(('r', 's')):
            child = parent.logical_owners[index]
            assert set(child.redundancy_queries) == set(child.thinking_tokens) == {rid}
        r.begin_step()
        for rid in order:
            assert r.drain_requests({rid}, wait=True)
        assert r.accepted == {'r': 4096, 's': 4096} and len(r.released) == 1
        for rid, env_owner, begin, span in (('r', env.a, 0, (4000, 4096)), ('s', env.b, 4096, (10, 74))):
            authority = env_owner.authority
            assert token_stats(authority) == reference(env.bindings[rid], 4096)
            history = authority.redundancy_history
            assert history.complete and history.stats['question_tokens'] == span[1]-span[0]
            [(offset, rows)] = env.scopes[rid].question
            assert offset == 0
            assert torch.equal(rows, env.query[begin+span[0]:begin+span[1]].float().unsqueeze(0))
        stats = r.thinking_capture_stats
        assert stats['accepted_rows'] == stats['synthesized_rows'] == 8192
        assert stats['synthesized_verified_rows'] == (8192 if verify else 0)


@pytest.mark.parametrize('order', [('r', 's'), ('s', 'r')])
def test_logical_owner_and_ordinary_peer_use_their_own_paths(tmp_path, order):
    with shared(tmp_path, ordinary_peer=True) as env:
        r, parent = env.runtime, env.parent
        # The ordinary peer keeps device-row descriptors on the shared packet.
        assert set(parent.redundancy_queries) == set(parent.thinking_tokens) == {'s'}
        assert 'byte_offset' in parent.thinking_tokens['s'] and not parent.thinking_tokens['s']['synthesized']
        assert 'byte_offset' not in parent.logical_owners[0].thinking_tokens['r']
        r.begin_step()
        for rid in order:
            assert r.drain_requests({rid}, wait=True)
        assert r.accepted == {'r': 4096, 's': 2048} and len(r.released) == 1
        assert token_stats(env.a.authority) == reference(env.bindings['r'], 4096)
        assert token_stats(env.b.authority) == reference(env.bindings['s'], 2048)
        stats = r.thinking_capture_stats
        assert stats['accepted_rows'] == 6144 and stats['synthesized_rows'] == 4096
        for rid, authority, begin, span in (('r', env.a.authority, 0, (4000, 4096)), ('s', env.b.authority, 4096, (10, 74))):
            assert authority.redundancy_history.complete
            [(offset, rows)] = env.scopes[rid].question
            assert torch.equal(rows, env.query[begin+span[0]:begin+span[1]].float().unsqueeze(0))
        assert not parent.redundancy_queries and not parent.thinking_tokens


def test_shared_extent_includes_side_bytes(tmp_path):
    with shared(tmp_path) as env:
        env.parent.aux_bytes += 16
        env.runtime.begin_step()
        with pytest.raises(ValueError, match='invalid byte extent'):
            env.runtime.drain_requests({'r'}, wait=True)
        assert env.a.authority.history_end == env.b.authority.history_end == 0
