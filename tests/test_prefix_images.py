"""Shared prefix images on the CPU bytes archive (docs/prefix-sharing.md, serving/prefix_images.py). CPU only."""
from types import SimpleNamespace as NS
import time

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.serving.authority import RequestAuthority
from vllm_kvmem.serving.capture_runtime import CaptureRuntime
from vllm_kvmem.serving.coordination import HistoryAdmission
from vllm_kvmem.serving.prefix_images import (ImageKey, ImageNotice, PrefixImageIndex, SessionImport,
    _with_handles, _without_handles, images_of, scope_error)
from vllm_kvmem.serving.prompt_checkpoints import CheckpointPhase, token_bytes
from vllm_kvmem.serving.request_plan import RequestCapacityError
from vllm_kvmem.serving.session_control import (SessionBinding, SessionCapture, SessionCommands,
    SessionEviction, SessionLedger, SessionNotice)
from vllm_kvmem.serving.session_worker import SessionWorker
from vllm_kvmem.serving.window_batch import WindowBatchAdapter
from test_authority import setup, payload, drain
from test_session_state import fixture

RECORD_BYTES = 2 * 4 * 1 * 2 * 3        # test_authority geometry: 2 layers, 4 tokens, 1 head, dim 2


def memory_archive(tmp_path, capacity=16):
    return MemoryArchive(str(tmp_path), RECORD_BYTES, capacity)


def records_by_begin(authority):
    return {r.history_begin: r for r in authority.records.values()}


# --- archive aliasing ----------------------------------------------------------

def test_alias_publishes_the_identical_bytes_and_reads_them_back(tmp_path):
    parent, archive = setup(tmp_path, 'parent', memory_archive(tmp_path))
    assert parent.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    drain(parent)
    records = sorted(parent.records.values(), key=lambda r: r.history_begin)
    leases = [parent.memory.repository.pin(r.handle) for r in records]
    sources = archive.record_payloads('parent', leases)
    assert all(type(s) is bytes and len(s) == RECORD_BYTES for s in sources)
    child, _ = setup(tmp_path, 'child', archive)
    repository = child.memory.repository
    handles = []
    for record in records:
        lease = repository.reserve(record.history_begin, record.count).commit()
        handles.append(_native.Handle(lease.handle.slot, lease.handle.generation))
        child.memory._store.register_block(lease, record.history_begin)
        assert not lease          # moved into the store
    assert child.history_end == 8
    pins = [repository.pin(h) for h in handles]
    used = archive.tier.used_slots
    assert archive.alias_records('child', pins, sources)
    for pin, source in zip(pins, sources):
        entry = archive._entries[archive.key('child', pin)]
        assert entry.ready and archive.tier.records[entry.allocation.slot] is source
    assert archive.tier.used_slots == used + 2 and archive.namespace_allocated_records('child') == 2
    # Readers see the parent's exact data through the child's own keys.
    for record, pin, handle in zip(records, pins, handles):
        assert archive.read('child', [pin]).result(timeout=5)[0] == archive.read('parent', [
            parent.memory.repository.pin(record.handle)]).result(timeout=5)[0]
    with pytest.raises(ValueError, match='overwrite'):
        archive.alias_records('child', pins[:1], sources[:1])
    # Too few free records: nothing is published.
    small = memory_archive(tmp_path / 'small', capacity=1)
    assert small.alias_records('child', pins, sources) is False and not small._entries
    for pin in pins + leases:
        pin.reset()


# --- worker: registration and import ---------------------------------------------

SCOPE = dict(session_freeze_prompt=True, request_planning=True, archive_backend='cpu')


def worker_fixture(tmp_path, *, archive=None, share=8, capacity=3, commands=None):
    authority, archive = setup(tmp_path, 'a', memory_archive(tmp_path) if archive is None else archive)
    assert authority.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    drain(authority)
    adapter, conv, ssm, cache, _ = fixture()
    runner = adapter.runner
    runner.device = torch.device('cpu')
    runner._kvmem_window_adapter = WindowBatchAdapter((0,), 16)
    runner._kvmem_probe_output = tmp_path
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    built = []

    def build_authority(request_id, binding, *, graph_metadata=True):
        child = RequestAuthority(request_id, authority.memory.config, archive, 2, 1, 2, resident_records=4)
        # Shift the child's repository so that its handles differ from the parent's.
        child.memory.repository.reserve(0, 4).cancel()
        built.append((request_id, binding, graph_metadata, child))
        return child
    runtime.__dict__.update(authorities={'a': authority}, names=('l0', 'l1'),
        accepted={'a': 8}, rejected={'a': 0}, last_steps={'a': []}, completed=[],
        runner=runner, queue=[], terminal_prefixes={}, terminal_discarded={}, archive=archive,
        build_authority=build_authority, _write_archive_report=lambda: None)
    runner._kvmem_capture_runtime = runtime
    options = NS(memory=authority.memory.config, session_capacity=capacity, session_checkpoint_bytes=8192,
                 verify_archive_on_finish=True, packed_assembly=True, direct_pinned_reads=True, **SCOPE)
    worker = SessionWorker(runner, options)
    worker.model_adapter = adapter
    if commands is None:
        commands = SessionCommands([SessionBinding(('scope', 'session'), 1, 'a', tuple(range(11)),
                                                   share_prefix=share)], [SessionCapture('a', 1, 8)])
    binding, = commands.bindings
    worker.receive(commands)
    worker.before_attention(NS(req_ids=['a']))
    return worker, binding, authority, archive, (conv, ssm, cache), built


def retire(worker, authority, request_id='a'):
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    assert worker.detach(request_id, authority, {'request_id': request_id})
    del worker.runtime.authorities[request_id]
    deadline = time.monotonic() + 5
    while not worker.retirement_done(request_id):
        assert time.monotonic() < deadline, 'retirement stalled'
        time.sleep(.001)
    return worker.notices()


def events(tmp_path):
    import json
    path = tmp_path / 'prefix_images.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_idle_frozen_p_registers_an_image_and_import_builds_an_equal_idle_session(tmp_path, monkeypatch):
    worker, binding, parent, archive, (conv, ssm, cache), built = worker_fixture(tmp_path)
    parent_checkpoint = worker.pool.entries[binding.namespace]
    snapshot = parent_checkpoint.model
    retired, registered = retire(worker, parent)
    assert retired.kind == 'retired' and retired.history == 8
    assert registered.kind == 'image_registered' and registered.history == 8
    key = registered.key
    assert (key.salt, key.position) == ('scope', 8) and registered.tokens == token_bytes(list(range(11)), 8)
    image = images_of(worker)[key]
    assert parent_checkpoint.phase is CheckpointPhase.IDLE and image.model is snapshot
    parent_records = records_by_begin(parent)
    namespace = ('scope', 'other')
    worker.receive(SessionCommands(image_imports=[SessionImport(namespace, 7, key)]))
    imported, = worker.notices()
    assert (imported.kind, imported.namespace, imported.generation, imported.request_id) == (
        'imported', namespace, 7, '')
    assert (imported.history, imported.window, imported.active_records) == (8, 8, ((0, 4), (4, 4)))
    (authority_id, synthetic, graph_metadata, child), = built
    assert synthetic.namespace == namespace and synthetic.generation == 7 and graph_metadata is False
    checkpoint = worker.pool.entries[namespace]
    assert checkpoint.phase is CheckpointPhase.IDLE and checkpoint.kind == 'P'
    assert checkpoint.authority is child and checkpoint.request_id == authority_id == child.memory.request_id
    assert checkpoint.model is snapshot and checkpoint.tokens == parent_checkpoint.tokens
    assert worker.identities[namespace] == 7
    assert worker.pool.payload_bytes == parent_checkpoint.payload_bytes + checkpoint.payload_bytes
    # The store holds the same history as the parent at P under its own handles.
    assert child.history_end == 8
    child_records = records_by_begin(child)
    handles = {begin: record.handle for begin, record in child_records.items()}
    assert sorted(child_records) == sorted(parent_records) == [0, 4]
    assert all(handles[b] != parent_records[b].handle for b in handles)
    translated = tuple((handles[b].slot, handles[b].generation, b, c, d)
                       for _, _, b, c, d in parent_checkpoint.window.records)
    assert checkpoint.window.records == translated == tuple(child.memory._store.working_records())
    child.memory._validate_window_checkpoint(checkpoint.window)
    for begin, record in child_records.items():
        source = parent_records[begin]
        assert record.archived and record.submitted and record.count == source.count
        assert record.key_sum is source.key_sum and record.mean_key is source.mean_key
        assert record.digest == source.digest and record.raw is None
        entry = archive._entries[(authority_id, record.handle.slot, record.handle.generation)]
        assert archive.tier.records[entry.allocation.slot] is image.blocks[begin // 4].payload
        expected = parent.read_record(source.handle)
        actual = child.read_record(record.handle)
        assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    rows = [row for row in events(tmp_path) if row['event'] in ('registered', 'imported')]
    assert [row['event'] for row in rows] == ['registered', 'imported']
    assert rows[1]['position'] == 8 and 'total' in rows[1]['seconds']

    # The unchanged claim path restores the imported session for its first request.
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    calls = []

    class Assembler:
        def assemble(self, owner, plan):
            assert owner is child and all(not r.skip for r in plan.remaps)
            calls.append('assemble')
            return NS(event=NS(synchronize=lambda: None), query=lambda: True)

        def close(self):
            return True
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *a, **k: Assembler())
    runner = worker.runner
    del runner.req_states.req_id_to_index['a']
    del runner._kvmem_request_generations['a']
    runner.req_states.req_id_to_index['new'] = 2
    runner._kvmem_request_generations['new'] = 3
    runner.req_states.prefill_len.np[2] = 14
    conv[5:7].fill_(-3); ssm[5:7].fill_(-3)
    worker.receive(SessionCommands([SessionBinding(namespace, 7, 'new', tuple(range(14)), 8, 8)],
                                   [SessionCapture('new', 7, 8)]))
    worker.before_attention(NS(req_ids=['new']))
    assert calls == ['assemble']
    assert worker.runtime.authorities['new'] is child and worker.notices()[0].kind == 'restored'
    restored = worker.model_adapter._recurrent_views('new')
    assert restored.keys() == snapshot.recurrent.keys()
    assert all(torch.equal(restored[name].cpu(), saved) for name, saved in snapshot.recurrent.items())
    # The parent's frozen P is untouched and still claimable.
    assert worker.pool.entries[binding.namespace] is parent_checkpoint
    assert parent_checkpoint.phase is CheckpointPhase.IDLE
    # Evicting the parent keeps the image's (and the child's) payload readable.
    worker.evict([SessionEviction(binding.namespace, 1)])
    assert [n.kind for n in worker.notices()] == ['evicted']
    raw, values = child.read_record(child_records[0].handle)
    assert torch.equal(raw, payload(0, 4)[0])
    worker.receive(SessionCommands(image_releases=[key]))
    assert key not in images_of(worker) and events(tmp_path)[-1]['event'] == 'evicted'


def test_import_failure_installs_nothing_and_reports(tmp_path):
    worker, binding, parent, archive, _, built = worker_fixture(tmp_path)
    _, registered = retire(worker, parent)
    used = archive.tier.used_slots
    # Too few free archive records for the image: the partial session is closed.
    free, archive._free_ids = archive._free_ids, archive._free_ids[:1]
    worker.receive(SessionCommands(image_imports=[SessionImport(('scope', 'x'), 3, registered.key)]))
    archive._free_ids = free + [i for i in archive._free_ids if i not in free]
    failed, = worker.notices()
    assert (failed.kind, failed.namespace) == ('import_failed', ('scope', 'x'))
    assert ('scope', 'x') not in worker.pool.entries and ('scope', 'x') not in worker.identities
    assert archive.tier.used_slots == used
    # A released image cannot be imported.
    worker.receive(SessionCommands(image_releases=[registered.key],
                                   image_imports=[SessionImport(('scope', 'y'), 4, registered.key)]))
    assert [n.kind for n in worker.notices()] == ['image_released', 'import_failed']
    assert 'no longer registered' in events(tmp_path)[-1]['reason']


def test_capture_outside_the_cpu_bytes_archive_is_refused(tmp_path):
    from vllm_kvmem.core.storage import AsyncArchive
    archive = AsyncArchive(str(tmp_path / 'nvme'), RECORD_BYTES, 16)
    worker, binding, parent, _, _, _ = worker_fixture(tmp_path, archive=archive)
    retired, refused = retire(worker, parent)
    assert refused.kind == 'image_refused' and 'MemoryArchive' in refused.reason
    assert not images_of(worker)
    assert events(tmp_path)[-1]['event'] == 'refused'


def test_layer_maps_lose_parent_handles_and_take_the_imported_ones():
    from vllm_kvmem.serving.layerwise_checkpoints import LayerwisePrefix
    rows = ((3, 9, 0, 4, 0), (5, 9, 4, 4, 4))
    prefix = LayerwisePrefix(8, ('l0', 'l1'), (rows, rows[1:]), (((), 8, 1, 1), ((), 8, 1, 1)))
    object.__setattr__(prefix, '_map_pairs', [(3, 9), (5, 9)])
    object.__setattr__(prefix, '_decode_sorted', True)
    bare = _without_handles(prefix)
    assert bare.maps == (((-1, -1, 0, 4, 0), (-1, -1, 4, 4, 4)), ((-1, -1, 4, 4, 4),))
    assert '_map_pairs' not in bare.__dict__ and bare.histories is prefix.histories
    assert bare.__dict__['_decode_sorted'] and bare.evidence() == prefix.evidence()
    handles = {0: _native.Handle(1, 2), 4: _native.Handle(2, 2)}
    bound = _with_handles(bare, handles)
    assert bound.maps == (((1, 2, 0, 4, 0), (2, 2, 4, 4, 4)), ((2, 2, 4, 4, 4),))
    with pytest.raises(ValueError, match='outside the image'):
        _with_handles(bare, {0: handles[0]})


def test_scheduler_and_worker_round_trip_admits_the_import_as_a_session_hit(tmp_path, monkeypatch):
    def plan_request(name, session, tokens, share=None):
        params = {'kvmem_request_plan': {'kvmem_query_span': [9, 11], 'kvmem_replay_span': [8, 11],
                                         'kvmem_reselect': 'force'}}
        if share is not None:
            params['kvmem_share_prefix'] = share
        return NS(request_id=name, session_id=session, cache_salt='tenant', prompt_token_ids=tokens,
                  num_computed_tokens=0, num_prompt_tokens=len(tokens), max_tokens=2, kv_transfer_params=params)
    prepared = []
    control = SessionLedger(3, 4, lambda *a: prepared.append(a), lambda *a: None, lambda *a: None,
        select_budget=8, gen_budget=8, freeze_prompt=True, prefix_image_scope=None, prefix_image_model='')
    parent = plan_request('a', 's', list(range(11)), share=8)
    assert control.lookup(parent, 0) == 0
    control.allocated(parent, 0)
    parent.num_computed_tokens = 8
    control.capture_before_final_prefill(parent, 3)
    worker, binding, authority, archive, _, built = worker_fixture(tmp_path, commands=control.take_commands())
    assert binding.share_prefix == 8
    control.finished(parent)
    for notice in retire(worker, authority):
        control.consume(notice)
    assert len(control.prefix_images.images) == 1
    child = plan_request('b', 's2', list(range(8)) + [50, 51, 52])
    assert control.lookup(child, 0) is None
    worker.receive(control.take_commands())
    imported, = worker.notices()
    assert control.consume(imported) is True
    assert control.lookup(child, 0) == 8 and prepared[-1] == ('b', 8, 8)
    control.allocated(child, 8)
    child.num_computed_tokens = 8
    control.capture_before_final_prefill(child, 3)
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *a, **k: NS(
        assemble=lambda owner, plan: NS(event=NS(synchronize=lambda: None), query=lambda: True),
        close=lambda: True))
    runner = worker.runner
    runner.req_states.req_id_to_index['b'] = 2
    runner._kvmem_request_generations['b'] = 3
    runner.req_states.prefill_len.np[2] = 11
    worker.receive(control.take_commands())
    worker.before_attention(NS(req_ids=['b']))
    (_, _, _, child_authority), = built
    assert worker.runtime.authorities['b'] is child_authority
    restored, = worker.notices()
    assert restored.kind == 'restored' and control.consume(restored) is None
    # The child captured its own frozen P again, as a sequential follow-up would.
    checkpoint = worker.pool.entries[('tenant', 's2')]
    assert checkpoint.request_id == 'b' and checkpoint.phase is CheckpointPhase.LIVE
    assert checkpoint.position == 8 and checkpoint.authority is child_authority


# --- scheduler: ledger, index, credits ------------------------------------------------

def request(name, session, tokens, *, salt='tenant', share=None, query=(17, 20)):
    params = {'kvmem_request_plan': {'kvmem_query_span': list(query), 'kvmem_replay_span': [16, query[1]],
                                     'kvmem_reselect': 'force'}}
    if share is not None:
        params['kvmem_share_prefix'] = share
    return NS(request_id=name, session_id=session, cache_salt=salt, prompt_token_ids=tokens,
              num_computed_tokens=0, num_prompt_tokens=len(tokens), max_tokens=4, kv_transfer_params=params)


def ledger(scope=None, capacity=4, preserve=False):
    events = []
    control = SessionLedger(capacity, 4, lambda *a: events.append(('prepare', a)),
        lambda *a: events.append(('abort', a)), lambda *a: events.append(('commit', a)),
        select_budget=32, gen_budget=16, freeze_prompt=True, prefix_image_scope=scope,
        prefix_image_model='model', preserve_history=preserve)
    return control, events


def registered_parent(control, *, session='parent', tokens=None, share=16, bytes=100):
    tokens = list(range(25)) if tokens is None else tokens
    req = request('p-' + session, session, tokens, share=share)
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    req.num_computed_tokens = 16
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    binding, = commands.bindings
    assert binding.share_prefix == share and commands.captures[0].position == 16
    control.finished(req)
    control.consume(SessionNotice('retired', binding.namespace, binding.generation, req.request_id, 16, 12,
                                  active_records=((0, 4), (4, 4), (8, 4))))
    key = ImageKey('model', 'tenant', 16, 'sha-' + session)
    notice = ImageNotice('image_registered', binding.namespace, binding.generation, req.request_id, 16, key,
                         token_bytes(tokens, 16), bytes)
    assert control.consume(notice) is True        # the historical admission reads its pool transfer
    return binding, key


def test_new_session_imports_the_image_and_is_admitted_as_a_session_hit():
    control, events = ledger()
    parent, key = registered_parent(control)
    parent_entry = control.entries[parent.namespace]
    assert control.prefix_images.images[key].ledger['query'] == parent_entry.query
    req = request('b', 'child', list(range(16)) + [90] * 9)
    assert control.lookup(req, 0) is None
    command, = control.take_commands().image_imports
    assert command.key == key and command.namespace == ('tenant', 'child')
    assert control.entries[command.namespace].phase == 'importing'
    assert control.lookup(req, 0) is None and not control.take_commands().image_imports
    assert control.has_pending_work()
    notice = SessionNotice('imported', command.namespace, command.generation, '', 16, 12, 7,
                           active_records=((0, 4), (4, 4), (8, 4)))
    assert control.consume(notice) is True
    entry = control.entries[command.namespace]
    assert entry.phase == 'idle' and entry.history == 16 and entry.window == 12
    assert entry.tokens == parent_entry.tokens and entry.query == parent_entry.query
    assert entry.mandatory == parent_entry.mandatory and entry.ready_records == 7
    assert control.prefix_images.images[key].refs == 0
    assert control.lookup(req, 0) == 16
    assert events[-1] == ('prepare', ('b', 16, 12))
    control.allocated(req, 16)
    binding, = control.take_commands().bindings
    assert (binding.namespace, binding.history, binding.window, binding.share_prefix) == (
        command.namespace, 16, 12, 0)
    # CPU history credits: the imported session holds P like a retired turn.
    credits = HistoryAdmission(1 << 20, 4, 8, 64)
    credits.consume(notice)
    lease = credits.leases[('session', 'tenant', 'child')]
    assert (lease.generation, lease.bytes, lease.request_id, lease.phase) == (
        command.generation, credits.extent_bytes(16), None, 'idle')


def test_matching_requires_the_same_salt_and_an_exact_prefix():
    control, _ = ledger()
    registered_parent(control)
    for req in (request('salt', 'x', list(range(25)), salt='other'),
                request('edit', 'y', [0] + list(range(1, 25))[::-1])):
        assert control.lookup(req, 0) == 0
        assert not control.take_commands().image_imports


def test_index_picks_the_longest_prefix_and_respects_references_and_budget():
    index = PrefixImageIndex('model', 250)
    tokens = list(range(40))

    def notice(position, size, session='p'):
        return ImageNotice('image_registered', ('tenant', session), 1, 'r', position,
                           ImageKey('model', 'tenant', position, str(position)), token_bytes(tokens, position), size)
    entry = NS(query=None, mandatory=(), query_id=None, guided_anchor=0, prompt_guided_anchor=0)
    assert index.register(notice(8, 100), entry) == []
    assert index.register(notice(16, 100, 'q'), entry) == []
    assert index.match('tenant', tokens).key.position == 16
    assert index.match('tenant', tokens[:16]).key.position == 8       # P must leave a token
    assert index.match('tenant', [99] + tokens[1:]) is None
    assert index.match('other', tokens) is None
    assert PrefixImageIndex('another-model', 250).register(notice(8, 1), entry) == [notice(8, 1).key]
    eight, sixteen = (ImageKey('model', 'tenant', p, str(p)) for p in (8, 16))
    index.acquire(eight)
    # Over budget: the least recently used unreferenced image goes, never a referenced one.
    assert index.register(notice(24, 100, 'r'), entry) == [sixteen]
    assert eight in index.images and index.used_bytes == 200
    # An explicit close releases its images once their imports finish.
    assert index.release_owner(('tenant', 'p')) == []
    assert index.match('tenant', tokens[:20]) is None                  # released: no new matches
    assert index.release_ref(eight) == [eight] and eight not in index.images
    with pytest.raises(KeyError):
        index.release_ref(eight)


def test_cancelled_import_evicts_the_orphan_and_failed_import_prefills_cold():
    control, events = ledger()
    _, key = registered_parent(control)
    req = request('b', 'child', list(range(16)) + [90] * 9)
    assert control.lookup(req, 0) is None
    command, = control.take_commands().image_imports
    control.finished(req)
    assert control.entries[command.namespace].import_orphan
    assert control.consume(SessionNotice('imported', command.namespace, command.generation, '', 16, 12)) is True
    eviction, = control.take_commands().evictions
    assert (eviction.namespace, control.entries[command.namespace].phase) == (command.namespace, 'evicting')
    control.consume(SessionNotice('evicted', command.namespace, command.generation, ''))
    assert command.namespace not in control.entries
    # A failed import: the request prefills cold, without another import attempt.
    other = request('c', 'third', list(range(16)) + [91] * 9)
    assert control.lookup(other, 0) is None
    command, = control.take_commands().image_imports
    assert control.consume(SessionNotice('import_failed', command.namespace, command.generation, '')) is False
    assert command.namespace not in control.entries
    assert control.lookup(other, 0) == 0 and not control.take_commands().image_imports
    assert control.prefix_images.images[key].refs == 0


def test_full_ledger_waits_and_explicit_close_releases_the_image():
    control, _ = ledger(capacity=1, preserve=True)
    parent, key = registered_parent(control)
    req = request('b', 'child', list(range(16)) + [90] * 9)
    assert control.lookup(req, 0) is None          # full preserved ledger: wait, as today
    commands = control.take_commands()
    assert not commands.image_imports and not commands.evictions and req.request_id not in control.importing
    close = NS(request_id='close', session_id='parent', cache_salt='tenant', prompt_token_ids=[7],
               num_computed_tokens=0, num_prompt_tokens=1, max_tokens=1,
               kv_transfer_params={'kvmem_release_session': True})
    assert control.lookup(close, 0) is None
    commands = control.take_commands()
    assert commands.image_releases == [key] and commands.evictions[0].namespace == parent.namespace
    assert key not in control.prefix_images.images
    # Room again after the parent's eviction, but no image left: an ordinary cold session.
    control.consume(SessionNotice('evicted', parent.namespace, parent.generation, ''))
    assert control.lookup(req, 0) == 0 and not control.take_commands().image_imports


def test_share_prefix_is_validated_and_refused_outside_the_scope():
    control, _ = ledger()
    for share, message in ((12, 'not the frozen prompt position'), (18, 'block boundary'),
                           (True, 'positive integer'), (25, 'block boundary')):
        with pytest.raises(RequestCapacityError, match=message):
            control.lookup(request('x', 's', list(range(25)), share=share), 0)
    assert not control.entries
    refused, _ = ledger(scope='tensor parallel serving')
    with pytest.raises(RequestCapacityError, match='prefix image scope: tensor parallel serving'):
        refused.lookup(request('x', 's', list(range(25)), share=16), 0)
    unconfigured = SessionLedger(2, 4, *(lambda *a: None,) * 3, select_budget=32, gen_budget=16,
                                 freeze_prompt=True)
    with pytest.raises(RequestCapacityError, match='not configured'):
        unconfigured.lookup(request('x', 's', list(range(25)), share=16), 0)


@pytest.mark.parametrize('change,reason', [
    (dict(historical_gc=object()), 'KVMEM_PREFIX_POOL_BYTES'), (dict(archive_backend='nvme'), 'CPU archive'),
    (dict(repository_gc=True), 'repository GC'), (dict(query_history=True), 'query_history'),
    (dict(query_score_mode='key_fps_4'), 'FPS'), (dict(query_fps_prebuild=True), 'query_fps_prebuild'),
    (dict(request_planning=False), 'request planning')])
def test_scope_refuses_what_a_checkpoint_does_not_carry(change, reason):
    options = NS(session_capacity=8, request_planning=True, archive_backend='cpu', graph_mode='multi')
    assert scope_error(options) is None
    assert 'tensor parallel' in scope_error(options, tensor_parallel_size=2)
    for name, value in change.items():
        setattr(options, name, value)
    assert reason in scope_error(options)


def test_phase2_scope_accepts_the_agent_stack(monkeypatch):
    options = NS(session_capacity=40, request_planning=True, archive_backend='cpu', graph_mode='off',
                 session_freeze_prompt=False, session_checkpoint_policy='pm', session_capture_terminal=True,
                 async_restore=True, historical_gc=object(), cpu_release_sealed_sums=True,
                 cpu_session_ttl_ms=5.0, gpu_memory_admission='static')
    monkeypatch.setenv('KVMEM_PREFIX_POOL_BYTES', str(1 << 30))
    assert scope_error(options) is None
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'meank')
    assert 'KVMEM_RECALL_METHOD' in scope_error(options)


def test_default_path_carries_no_image_state():
    control, _ = ledger()
    req = NS(request_id='a', session_id='s', cache_salt='tenant', prompt_token_ids=list(range(25)),
             num_computed_tokens=0, num_prompt_tokens=25, kv_transfer_params=None)
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    commands = control.take_commands()
    assert commands.bindings[0].share_prefix == 0
    assert commands.image_imports == [] and commands.image_releases == []
    assert control.prefix_images is None and control.importing == {} and control.import_declined == set()
    assert SessionBinding(('a', 'b'), 1, 'r', ()) == SessionBinding(('a', 'b'), 1, 'r', (), share_prefix=0)


def test_evicting_an_imported_idle_session_releases_only_its_own_allocations(tmp_path):
    worker, binding, parent, archive, _, built = worker_fixture(tmp_path)
    _, registered = retire(worker, parent)
    image = images_of(worker)[registered.key]
    used = archive.tier.used_slots
    worker.receive(SessionCommands(image_imports=[SessionImport(('scope', 'other'), 5, registered.key)]))
    assert worker.notices()[0].kind == 'imported'
    (authority_id, _, _, child), = built
    assert archive.namespace_allocated_records(authority_id) == 2 and archive.tier.used_slots == used + 2
    worker.evict([SessionEviction(('scope', 'other'), 5)])
    assert [n.kind for n in worker.notices()] == ['evicted']
    assert archive.namespace_allocated_records(authority_id) == 0 and archive.tier.used_slots == used
    assert ('scope', 'other') not in worker.pool.entries and ('scope', 'other') not in worker.identities
    # The parent and the image still hold the very same payload objects.
    for begin, record in records_by_begin(parent).items():
        lease = parent.memory.repository.pin(record.handle)
        assert archive.record_payloads('a', [lease])[0] is image.blocks[begin // 4].payload
        lease.reset()


def test_http_validation_of_share_prefix():
    from vllm_kvmem.serving.http_sessions import validate_session_payload
    validate_session_payload({'session_id': 's', 'kv_transfer_params': {'kvmem_share_prefix': 1024}}, None)
    validate_session_payload({'kv_transfer_params': {}}, None)
    for payload in ({'kv_transfer_params': {'kvmem_share_prefix': 1024}},
                    {'session_id': 's', 'kv_transfer_params': {'kvmem_share_prefix': '1024'}},
                    {'session_id': 's', 'kv_transfer_params': {'kvmem_share_prefix': 0}}):
        with pytest.raises(ValueError, match='kvmem_share_prefix'):
            validate_session_payload(payload, None)


def test_imported_graph_restores_the_parents_immutable_graph_data():
    from vllm_kvmem.serving.block_graph import BlockGraph, GraphSnapshot
    from test_block_graph import accept, row
    parent = BlockGraph(('tenant', 'parent'), 4)
    accept(parent, 0, 9, [row(8, kind='prefill')])
    accept(parent, 9, 10, [row(9)])
    accept(parent, 10, 12)
    saved = parent.snapshot(12)
    child = BlockGraph(('tenant', 'child'), 4, budget=parent.budget)
    with pytest.raises(ValueError, match='foreign'):
        child.restore(saved)                      # why the import rebinds the data to the child
    adopted = GraphSnapshot(child, data=saved._data)
    child.restore(adopted)
    assert child.rows == parent.rows and child.accepted_through == child.completed_through == 12
    assert child.rows[0].source == 8 and child._data is saved._data
    for item in (adopted, saved, child, parent):
        item.close()
    assert parent.budget.used == 0
