"""Shared prefixes, phase 2 steps 4-7 (docs/prefix-sharing.md): the shared-prefix
pool charge, agent-stack registration and import with historical GC, asynchronous
restore of an imported session, and the scheduler's deficit order."""
from types import SimpleNamespace as NS
import json
import time

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.core.historical_gc import historical_record_charge
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.serving.historical_admission import (HistoricalAdmission, HistoricalCapacitySnapshot,
    HistoricalHistoryAdmission)
from vllm_kvmem.serving.prefix_images import (HistoryPoolNotice, ImageKey, ImageNotice, SessionImport,
    images_of, index_records, registration_checkpoint)
from vllm_kvmem.serving.prompt_checkpoints import CheckpointPhase, token_bytes
from vllm_kvmem.serving.session_control import (SessionBinding, SessionCapture, SessionCommands,
    SessionEviction, SessionLedger, SessionNotice)
from test_authority import payload
from test_prefix_images import retire, worker_fixture

RECORD = 2 * 4 * 1 * 2 * 3


def eventually(predicate):
    end = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < end
        time.sleep(.001)


def of(notices, kind):
    return [notice for notice in notices if getattr(notice, 'kind', None) == kind]


def gc_archive(tmp_path):
    archive = MemoryArchive(str(tmp_path / 'arena'), RECORD, 64, native_payloads=True)
    config = n.HistoricalGcConfig()
    config.mode, config.background, config.max_scopes, config.poll_ms = n.HistoricalGcMode.DELETE, True, 4, 1
    config.ready_low, config.ready_high, config.scan_blocks, config.delete_blocks = 1, 2, 8, 8
    worker = archive.enable_history_gc(config)
    archive.history_record_charge = historical_record_charge(RECORD, 4)
    return archive, worker


# --- step 5 (with step 4 accounting): agent stack with historical GC ---------------

def test_registration_and_import_on_the_native_arena_with_historical_gc(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_PREFIX_POOL_BYTES', str(1 << 20))
    archive, gc_worker = gc_archive(tmp_path)
    worker, binding, parent, _, (conv, ssm, cache), built = worker_fixture(tmp_path, archive=archive)
    arena = archive.tier.payloads
    notices = retire(worker, parent)
    registered, = of(notices, 'image_registered')
    index = index_records(archive, 2)          # the parent's own index of the two shared records
    assert 0 < index < 2 and registered.pool_records == 2 and registered.transferred == 2 - index
    image = images_of(worker)[registered.key]
    assert image.native and all(type(b.payload) is n.HistoricalPayloadHandle for b in image.blocks)
    # The parent's two prefix records moved to the pool: its private charge keeps only their index.
    parent_checkpoint = worker.pool.entries[binding.namespace]
    allocated, protected = worker._history_rows(parent_checkpoint)
    assert allocated == parent._resident_count + len(parent.gc.pending_index) + index and protected <= allocated
    assert parent.gc.scope.stats['shared_marked'] == 2
    pool_notice = of(notices, 'history_pool')[-1]
    assert isinstance(pool_notice, HistoryPoolNotice) and pool_notice.records == 2
    used = arena.stats['used_bytes']
    worker.receive(SessionCommands(image_imports=[SessionImport(('scope', 'other'), 7, registered.key)]))
    imported, = of(worker.notices(), 'imported')
    assert imported.history == 8
    (authority_id, _, _, child), = built
    # Aliases, not copies: arena bytes unchanged; the child's GC scope knows them as shared.
    assert arena.stats['used_bytes'] == used
    assert child.gc.scope.stats['shared_marked'] == 2 and child.gc.scope.stats['published'] == 2
    for block in image.blocks:
        assert arena.owners(block.payload) == (3, 1)          # parent, image, child
    for record in child.records.values():
        source = next(r for r in parent.records.values() if r.history_begin == record.history_begin)
        assert record.index_owner is not None and record.index_owner is not source.index_owner
        # A copy in the child's own index owner (the reference MeanK backend keeps sums).
        assert record.key_sum is not source.key_sum and torch.equal(record.key_sum, source.key_sum)
        assert record.key_sum.untyped_storage().data_ptr() != source.key_sum.untyped_storage().data_ptr()
        assert torch.equal(*(r.read_record(x.handle)[0] for r, x in ((child, record), (parent, source))))
    assert imported.allocated_records == index
    # Claiming the imported session runs the unchanged restore path.
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *a, **k: NS(
        assemble=lambda owner, plan: NS(event=NS(synchronize=lambda: None), query=lambda: True),
        close=lambda: True))
    runner = worker.runner
    runner.req_states.req_id_to_index['new'] = 2
    runner._kvmem_request_generations['new'] = 3
    runner.req_states.prefill_len.np[2] = 14
    worker.receive(SessionCommands([SessionBinding(('scope', 'other'), 7, 'new', tuple(range(14)), 8, 8)]))
    worker.before_attention(NS(req_ids=['new']))
    assert worker.runtime.authorities['new'] is child
    # The parent leaves: its descriptors drop their references; the child still reads.
    del worker.runtime.authorities['new']
    del record, source, parent_checkpoint      # no test reference may hold the parent's records
    worker.evict([SessionEviction(binding.namespace, 1)])

    def evicted():
        gc_worker.advance()
        worker.notices()
        return binding.namespace not in worker.pool.entries
    eventually(evicted)
    assert all(arena.owners(block.payload) == (2, 1) for block in image.blocks)
    assert archive.pool_records() == 2            # orphans stay charged to the pool
    worker.receive(SessionCommands(image_releases=[registered.key]))
    released = [notice for notice in worker.notices() if getattr(notice, 'kind', '') == 'image_released']
    assert released and released[0].key == registered.key
    assert all(arena.owners(block.payload) == (1, 0) for block in image.blocks)
    assert archive.pool_records() == 2 and arena.stats['used_bytes'] == used


def test_terminal_sessions_register_their_retained_prompt_checkpoint():
    p = NS(kind='P', position=8)
    m = NS(kind='M', position=10)
    worker = NS(pool=NS(prompts={('t', 's'): p}, entries={('t', 's'): m}))
    binding = NS(namespace=('t', 's'), share_prefix=8)
    assert registration_checkpoint(worker, binding, m) is p
    assert registration_checkpoint(NS(pool=NS(prompts={})), binding, m) is m      # refused later as not P
    frozen = NS(kind='P', position=8)
    assert registration_checkpoint(NS(pool=NS(prompts={})), binding, frozen) is frozen


def test_pool_budget_refuses_registration(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_PREFIX_POOL_BYTES', str(historical_record_charge(RECORD, 4)))   # one record
    archive, _ = gc_archive(tmp_path)
    worker, binding, parent, *_ = worker_fixture(tmp_path, archive=archive)
    refused, = of(retire(worker, parent), 'image_refused')
    assert 'pool budget' in refused.reason
    assert archive.pool_records() == 0 and not images_of(worker)


# --- asynchronous restore of an imported session -------------------------------------

def test_async_restore_claims_an_imported_checkpoint_and_retains_it(tmp_path):
    pytest.importorskip('vllm')
    from vllm_kvmem.serving.async_refresh import AsyncRefresh, RefreshJob
    from vllm_kvmem.serving.async_restore import RestoreContext
    from vllm_kvmem.serving.connector import RestoreCommand
    from test_core import Event
    worker, binding, parent, archive, _, built = worker_fixture(tmp_path)
    registered, = of(retire(worker, parent), 'image_registered')
    namespace = ('scope', 'other')
    worker.receive(SessionCommands(image_imports=[SessionImport(namespace, 7, registered.key)]))
    assert of(worker.notices(), 'imported')
    (_, _, _, child), = built
    pool, adapter = worker.pool, worker.model_adapter
    checkpoint = pool.entries[namespace]
    # The first half of async_restore._step_restore: claim and prepare without waiting.
    assert pool.claim(namespace, 'new', list(range(14)), position=8) is checkpoint
    plan = pool.prepare_claim_window(checkpoint, wait=False)
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    model = adapter.prepare_restore(checkpoint.model, 'new')
    adapter.submit_restore(model)
    fence = Event(True)
    trace = []
    assembly = NS(uploaded_bytes=0, host_prepare_seconds=0., query=lambda: True)
    future = NS(done=lambda: True, result=lambda: (assembly, None, fence))
    session = NS(pool=pool, model_adapter=adapter, imports={'new'},
                 _verify_restored=lambda *args: trace.append('verify'),
                 publish_restore=lambda rid, value, **kwargs: trace.append(('publish', rid, value, kwargs)))
    manager = AsyncRefresh.__new__(AsyncRefresh)
    manager.runner = adapter.runner
    manager.runner._kvmem_sessions = session
    manager.runner._kvmem_probe_output = tmp_path
    manager.runner._kvmem_restore_allocations = {'new': object()}
    manager.options = NS(async_restore=True)
    manager.deferred, manager.cursor = {}, 0
    manager.emit = lambda _job, stage, **fields: trace.append(stage)
    manager.maintenance = NS(observe=lambda job, size: None, release=lambda job: None)
    job = RefreshJob(RestoreCommand('new', -1, 3, 8, 8, ((1, 2),)), stage='restore_copy', dispatched=True,
                     future=future, authority=child, checkpoint=checkpoint, model_bytes=checkpoint.model.bytes,
                     restore_context=RestoreContext(checkpoint, plan, NS(close=lambda: True), model, None))
    manager.jobs = {'new': job}
    child._refresh_background = True
    assert manager.step(job) and job.done
    assert checkpoint.phase is CheckpointPhase.LIVE and checkpoint.request_id == 'new'
    assert checkpoint.authority is child and pool.entries[namespace] is checkpoint
    assert ('publish', 'new', child, {'retained': True}) in trace


# --- step 4: scheduler pool charge --------------------------------------------------

def test_registration_moves_parent_records_to_the_pool_atomically():
    ledger = HistoricalAdmission(100, 32, 4)
    key = ('session', 't', 'parent')
    ledger.snapshots[key] = HistoricalCapacitySnapshot(1, 64, 16, 16, 4)
    assert ledger.reserved_records == 16
    ledger.transfer_to_pool(key, 1, 16, 16)
    assert ledger.snapshots[key].allocated_records == 0 and ledger.snapshots[key].protected_records == 0
    assert ledger.pool_records == 16 and ledger.reserved_records == 16      # never counted twice
    ledger.observe_idle(key, HistoricalCapacitySnapshot(1, 64, 16, 0, 0))   # the worker agrees
    other = ('session', 't', 'child')
    ledger.import_idle(other, HistoricalCapacitySnapshot(7, 64, 16, 1, 0))
    assert ledger.reserved_records == 17
    with pytest.raises(ValueError, match='reused'):
        ledger.import_idle(other, HistoricalCapacitySnapshot(7, 64, 16, 1, 0))
    ledger.transfer_to_pool(('session', 't', 'gone'), 3, 5, 20)            # unknown scope: pool only
    assert ledger.pool_records == 20 and ledger.snapshot()['pool_records'] == 20


def test_observe_is_a_true_arena_bytes_check_including_the_pool():
    admission = HistoricalHistoryAdmission(10 * 8, 8, 32, 4)
    # Nothing shared: the snapshot is the same as before shared prefixes existed.
    assert not {'pool_records', 'image_releases', 'pool_draining_records'} & admission.snapshot().keys()
    admission.consume(HistoryPoolNotice(records=3))
    assert admission.snapshot()['pool_records'] == 3 and admission.snapshot()['pool_draining_records'] == 0
    admission.observe(3 * 8)
    with pytest.raises(ValueError, match='exceeded'):
        admission.observe(4 * 8)


# --- step 6: global deficit order -------------------------------------------------

def simulation(capacity_records, *, lru=False, monkeypatch=None):
    if lru:
        monkeypatch.setenv('KVMEM_SESSION_HISTORY_LRU_EVICTION', '1')
    control = SessionLedger(8, 4, lambda *a: None, lambda *a: None, lambda *a: None, select_budget=32,
                            gen_budget=16, freeze_prompt=True, prefix_image_scope=None, prefix_image_model='m',
                            preserve_history=True)
    admission = HistoricalHistoryAdmission(capacity_records * 8, 8, 64, 4, control)
    return control, admission


def idle_session(control, admission, session, *, history=16, allocated=0, protected=0):
    from vllm_kvmem.serving.session_control import _Entry
    control.generation += 1
    entry = _Entry(('t', session), control.generation, None, phase='idle', history=history,
                   tokens=token_bytes(list(range(history + 1)), history))
    control.entries[entry.namespace] = entry
    admission.ledger.snapshots[('session', 't', session)] = HistoricalCapacitySnapshot(
        entry.generation, history, 32, allocated, protected)
    return entry


def register_image(control, admission, parent, *, records):
    key = ImageKey('m', 't', 16, 'sha-' + parent.namespace[1])
    notice = ImageNotice('image_registered', parent.namespace, parent.generation, 'r', 16, key,
                         parent.tokens, 8 * records, transferred=0,
                         pool_records=admission.ledger.pool_records + records)
    assert control.consume(notice) is True
    admission.consume(notice)
    return key


def admit(control, admission, req):
    """The connector's order for a matched-admission request: ledger lookup, then CPU history."""
    return control.lookup(req, 0) is not None and admission.allows(req)


def request(name, session, tokens=20):
    # Tokens unlike any image prefix: an ordinary cold session of 6 records + output.
    return NS(request_id=name, session_id=session, cache_salt='t', prompt_token_ids=[1000 + i for i in range(tokens)],
              num_computed_tokens=0, num_prompt_tokens=tokens, max_tokens=4, kv_transfer_params=None)


def test_deficit_releases_unreferenced_images_first_and_waits_for_their_bytes():
    control, admission = simulation(16)
    parent = idle_session(control, admission, 'parent')
    key = register_image(control, admission, parent, records=4)
    idle_session(control, admission, 'idle', allocated=6, protected=0)
    assert admission.ledger.reserved_records == 10
    # The parent session is gone: nobody references the image any more.
    del control.entries[parent.namespace]
    del admission.ledger.snapshots[('session', 't', parent.namespace[1])]
    req = request('new', 'fresh')
    assert not admit(control, admission, req)
    commands = control.take_commands()
    assert commands.image_releases == [key] and not commands.history_capacity     # no lossy reclaim yet
    assert not admit(control, admission, req) and not control.take_commands().history_capacity
    admission.consume(ImageNotice('image_released', (), 0, '', 16, key))
    admission.consume(HistoryPoolNotice(records=4, draining=4))                    # bytes still draining
    assert not admit(control, admission, req) and not control.take_commands().history_capacity
    admission.consume(HistoryPoolNotice(records=0, draining=0))
    assert admission.ledger.reserved_records == 6
    assert admit(control, admission, req)                                          # freed bytes admit it


def test_referenced_images_are_never_released_for_a_deficit_and_reclaim_follows():
    control, admission = simulation(16)
    parent = idle_session(control, admission, 'parent')
    key = register_image(control, admission, parent, records=4)
    idle_session(control, admission, 'idle', allocated=6, protected=1)
    req = request('new', 'fresh')
    assert not admit(control, admission, req)
    commands = control.take_commands()
    assert not commands.image_releases                 # the parent still owns its payloads
    assert [(r.namespace, r.retained_records) for r in commands.history_capacity] == [(('t', 'idle'), 2)]
    assert key in control.prefix_images.images


def test_idle_session_eviction_unreferences_images_and_makes_progress(monkeypatch):
    control, admission = simulation(20, lru=True, monkeypatch=monkeypatch)
    parent = idle_session(control, admission, 'parent')
    key = register_image(control, admission, parent, records=4)
    child = idle_session(control, admission, 'child', allocated=1, protected=1)
    control.prefix_images.imported(key, child.namespace)
    idle_session(control, admission, 'idle', allocated=4, protected=4)
    req = request('new', 'fresh', tokens=44)      # 12 records + 1 in flight: needs the image's 4 too
    rounds = []
    for _ in range(12):
        if admit(control, admission, req):
            break
        commands = control.take_commands()
        rounds.append((tuple(commands.image_releases), tuple(e.namespace for e in commands.evictions)))
        for eviction in commands.evictions:          # the worker acknowledges each eviction
            notice = SessionNotice('evicted', eviction.namespace, eviction.generation, '')
            control.consume(notice)
            admission.consume(notice)
        for image_key in commands.image_releases:
            admission.consume(ImageNotice('image_released', (), 0, '', 16, image_key))
            admission.consume(HistoryPoolNotice(records=0, draining=0))
    else:
        pytest.fail('the deficit never resolved: %r' % rounds)
    evicted = [namespace for _, evictions in rounds for namespace in evictions]
    released = [k for releases, _ in rounds for k in releases]
    # One fenced eviction at a time; once neither the parent nor the child references
    # the image, it is released; the session with only protected history stays.
    assert evicted == [parent.namespace, child.namespace] and released == [key]


# --- the agent profile's authorities (CaptureRuntime.build_authority) ------------------

THINKING_LAYOUT = ([100, 101, 102, 103], [[10, 20], [11, 20], [12, 20], [13, 20], [14, 20]])
# user 0..8; assistant from 8: <|im_start|>assistant\\n<think>, thinking tokens, then the
# Query/continuation from 16 on (thinking continues past P until </think>).
AGENT_TOKENS = (1,) * 8 + (100, 13, 20, 102) + (7,) * 4
AGENT_SPANS = ((0, 8, 'user'), (8, 22, 'assistant'))
CONTINUATION = (7, 7, 103, 5, 5, 5)


def agent_stack(tmp_path, monkeypatch):
    """A CPU worker whose authorities are built by the real CaptureRuntime.build_authority
    with the agent profile's history settings (benchmarks/gate/agent.json): native arena via
    make_shared_archive, background historical GC with the redundancy policy (thinking_only,
    lru_fallback, ...), MeanK qw3 backend with released sealed sums, redundancy features,
    thinking classification and current-window protection on owned sessions."""
    from threading import Event
    from vllm_kvmem.core.config import KVMemConfig
    from vllm_kvmem.core.historical_gc import HistoricalGcSettings
    from vllm_kvmem.core.storage import make_shared_archive
    from vllm_kvmem.serving.capture_runtime import CaptureRuntime
    from vllm_kvmem.serving.session_worker import SessionWorker
    from vllm_kvmem.serving.window_batch import WindowBatchAdapter
    from test_session_state import fixture
    monkeypatch.setenv('KVMEM_PREFIX_POOL_BYTES', str(1 << 20))
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8, repository_blocks=8, sink_blocks=1,
                         recent_blocks=0)
    params = dict(keep_mandatory=True, thinking_only=True, pool_pairwise=True, all_history_references=False,
                  refresh_before_rank=True, epoch_peak=True, lru_fallback=True, importance_weight=0.1,
                  representative_similarity=0.9, candidate_blocks=128, reference_blocks=32,
                  pair_comparisons=512, query_segments_per_step=32, query_capture_bytes=1 << 16,
                  feature_cache_bytes=1 << 20, query_cache_bytes=1 << 20)
    gc = HistoricalGcSettings(mode='delete', policy='redundancy', policy_params=params, poll_ms=1,
                              scan_blocks=8, delete_blocks=8, ready_low=1, ready_high=2)
    archive = make_shared_archive(NS(memory=config, session_capacity=4, archive_backend='cpu', historical_gc=gc,
                                     cpu_read_cache_bytes=1 << 30, query_score_mode='mean_k',
                                     archive_io_max_bytes=0, cpu_gather_workers=1, async_seal=False),
                                  tmp_path / 'arena', RECORD, 2)
    adapter, *_ = fixture(draft=False)
    runner = adapter.runner
    runner.device = torch.device('cpu')
    runner._kvmem_window_adapter = WindowBatchAdapter((0,), 16)
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_layers = ('l0', 'l1')
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.__dict__.update(
        options=NS(graph_mode='off', historical_gc=gc, query_score_mode='mean_k', mean_k_backend='qw3_cuda',
                   cpu_release_sealed_sums=True, query_history=False, query_fps_prebuild=False,
                   strategy_picker='native', prefill_attention=False),
        config=config, archive=archive, layers=('l0', 'l1'), heads=1, dim=2, raw_dtype=torch.bfloat16,
        resident_records=4, gc_config=None, gc_progress=Event(), redundancy_enabled=True,
        redundancy_query_heads=1, thinking_enabled=True, thinking_layout=THINKING_LAYOUT,
        thinking_max_pending_tokens=64, action=NS(ids=tuple(THINKING_LAYOUT[0])),
        authorities={}, names=('l0', 'l1'), accepted={'a': 16}, rejected={'a': 0}, last_steps={'a': []},
        completed=[], runner=runner, queue=[], terminal_prefixes={}, terminal_discarded={},
        _write_archive_report=lambda: None)
    runner._kvmem_capture_runtime = runtime
    worker = SessionWorker(runner, NS(memory=config, session_capacity=4, session_checkpoint_bytes=8192,
        verify_archive_on_finish=True, packed_assembly=True, direct_pinned_reads=True,
        session_freeze_prompt=True, request_planning=True, archive_backend='cpu', historical_gc=gc))
    worker.model_adapter = adapter
    return NS(worker=worker, runtime=runtime, runner=runner, archive=archive)


def accept(authority, binding, begin, tokens, local=None):
    """The capture runtime's accepted-token step: thinking labels, then archive publication."""
    from test_authority import drain
    assert authority.append(*payload(begin, len(tokens)), history_begin=begin,
                            local_rope_begin=begin if local is None else local)
    authority.thinking_history.accept(binding, begin, len(tokens), tokens, list(range(begin, begin + len(tokens))),
                                      authority.memory.repository)
    drain(authority)


def frame_state(authority):
    """Per history block: (in the protected current frame, in the working window)."""
    store = authority.memory._store
    current = {begin for _, _, begin in store.current_frame_records()}
    working = {begin for _, _, begin, _, _ in store.working_records()}
    return {r.history_begin: (r.history_begin in current, r.history_begin in working)
            for r in authority.records.values()}


def register_agent_prefix(stack, *, layerwise=False):
    """An agent-profile parent session prefills 16 tokens, pressure keeps blocks 0 and 8 in
    its window, the frozen P is captured at 16 and the turn retires: the image registers.
    ``layerwise``: the P checkpoint also carries per-layer windows (layer 1 keeps block 12
    instead of 8), as a layerwise pressure capture stores them."""
    from vllm_kvmem.serving.authority import _Ready
    worker, runtime, runner = stack.worker, stack.runtime, stack.runner
    binding = SessionBinding(('scope', 'session'), 1, 'a', AGENT_TOKENS + CONTINUATION[:4], share_prefix=16,
                             message_spans=AGENT_SPANS)
    parent = runtime.build_authority('a', binding)
    runtime.authorities['a'] = parent
    assert parent.memory._store.protect_current_window          # as serving configures it
    for begin in (0, 8):
        accept(parent, binding, begin, AGENT_TOKENS[begin:begin + 8])
    parent.action_history.scores = {0: 1.0, 8: 3.0}
    records = {r.history_begin: r for r in parent.records.values()}
    plan = parent.memory.select({r.handle.slot: float(begin == 8) for begin, r in records.items()})
    parent.memory.attach_completion(_Ready())
    assert parent.memory.commit_selection()
    del plan
    runner.req_states.num_computed_tokens_np[0] = 16
    runner.req_states.prefill_len.np[0] = 20
    worker.receive(SessionCommands([binding], [SessionCapture('a', 1, 16)]))
    worker.before_attention(NS(req_ids=['a']))
    at_p = frame_state(parent)
    assert at_p == {0: (True, True), 4: (False, False), 8: (True, True), 12: (False, False)}
    if layerwise:
        from vllm_kvmem.serving.layerwise_checkpoints import LayerwisePrefix
        handles = {r.history_begin: (r.handle.slot, r.handle.generation) for r in parent.records.values()}
        maps = tuple(tuple((*handles[begin], begin, 4, dense) for begin, dense in rows)
                     for rows in (((0, 0), (8, 4)), ((0, 0), (12, 4))))
        worker.pool.entries[binding.namespace].layerwise_pressure = LayerwisePrefix(16, ('l0', 'l1'), maps,
                                                                                  (((),), ((),)))
    accept(parent, binding, 16, CONTINUATION[:4], local=8)
    assert worker.detach('a', parent, {'request_id': 'a'})
    del runtime.authorities['a']
    deadline = time.monotonic() + 5
    while not worker.retirement_done('a'):
        assert time.monotonic() < deadline, 'retirement stalled'
        time.sleep(.001)
    stack.registration_notices = worker.notices()
    registered, = of(stack.registration_notices, 'image_registered')
    return parent, binding, at_p, registered


def claim(stack, namespace, generation, request_id, index):
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    runner, worker = stack.runner, stack.worker
    runner.req_states.req_id_to_index[request_id] = index
    runner._kvmem_request_generations[request_id] = 10 + index
    runner.req_states.num_computed_tokens_np[index] = 16
    runner.req_states.prefill_len.np[index] = 22
    tokens = AGENT_TOKENS + CONTINUATION
    worker.receive(SessionCommands([SessionBinding(namespace, generation, request_id, tokens, 16, 8,
                                                   message_spans=AGENT_SPANS)]))
    worker.before_attention(NS(req_ids=[request_id]))
    assert len(of(worker.notices(), 'restored')) == 1     # idle-history notices may come along
    authority = worker.runtime.authorities[request_id]
    accept(authority, SessionBinding(namespace, generation, request_id, tokens, 16, 8, message_spans=AGENT_SPANS),
           16, tokens[16:20], local=8)
    return authority


def test_agent_profile_import_keeps_current_window_protection_and_equals_the_parent_at_p(tmp_path, monkeypatch):
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    stack = agent_stack(tmp_path, monkeypatch)
    worker = stack.worker
    parent, binding, at_p, registered = register_agent_prefix(stack)
    image = images_of(worker)[registered.key]
    assert image.native and registered.pool_records == 4
    worker.receive(SessionCommands(image_imports=[SessionImport(('scope', 'other'), 7, registered.key)]))
    imported, = of(worker.notices(), 'imported')
    assert not any(e['event'] == 'import_failed' for e in map(json.loads, open(tmp_path / 'prefix_images.jsonl')))
    checkpoint = worker.pool.entries[('scope', 'other')]
    child = checkpoint.authority
    store = child.memory._store
    # The real build: protection configured before the first install and never toggled.
    assert store.protect_current_window and child.redundancy_history and child.thinking_history
    with pytest.raises(RuntimeError, match='before the first append'):
        store.set_protect_current_window(False)      # the native rule the first version broke
    # Only the frozen frame is current, exactly as in the parent at its frozen P.
    assert frame_state(child) == at_p
    parent_window = tuple(r[2:] for r in worker.pool.entries[binding.namespace].window.records)
    assert tuple(r[2:] for r in checkpoint.window.records) == parent_window == image.window == ((0, 4, 0), (8, 4, 4))
    # Agent-profile features of every imported record: redundancy keys, thinking counts,
    # shared marking in the child's own GC scope, MeanK index in its own owners.
    parent_records = {r.history_begin: r for r in parent.records.values()}
    assert child.redundancy_history.stats['features_published'] == 4
    assert child.gc.scope.stats['shared_marked'] == 4 and child.gc.scope.stats['published'] == 4
    for record in child.records.values():
        source = parent_records[record.history_begin]
        assert record._thinking_published == source._thinking_published
        assert record.key_sum is None and torch.equal(record.mean_key, source.mean_key)
        assert record.index_owner is not source.index_owner
    assert parent_records[12]._thinking_published == (4, 4)
    # Claiming either session restores the same frame and continues the same way.
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *a, **k: NS(
        assemble=lambda owner, plan: NS(event=NS(synchronize=lambda: None), query=lambda: True),
        close=lambda: True))
    claimed_child = claim(stack, ('scope', 'other'), 7, 'new', 2)
    claimed_parent = claim(stack, binding.namespace, 1, 'p2', 1)
    assert claimed_child is child and claimed_parent is parent
    assert frame_state(child) == frame_state(parent)
    assert child.action_history.scores == parent.action_history.scores == {0: 1.0, 8: 3.0}
    published = {r.history_begin: getattr(r, '_thinking_published', None) for r in child.records.values()}
    assert published == {r.history_begin: getattr(r, '_thinking_published', None) for r in parent.records.values()}
    assert published[16][0] > 0           # thinking continues across P: the parser was seeded


def test_agent_profile_import_recreates_the_layerwise_gc_hold(tmp_path, monkeypatch):
    stack = agent_stack(tmp_path, monkeypatch)
    _, _, at_p, registered = register_agent_prefix(stack, layerwise=True)
    image = images_of(stack.worker)[registered.key]
    assert image.layerwise is not None and image.layerwise.gc_hold is None     # handle-free, no parent hold
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')                  # the agent profile's execution env
    stack.runner._kvmem_layerwise_pressure = object()
    stack.worker.receive(SessionCommands(image_imports=[SessionImport(('scope', 'other'), 7, registered.key)]))
    imported, = of(stack.worker.notices(), 'imported')
    # The scheduler's view of the idle import, kept current by idle history reports.
    ledger, key = HistoricalAdmission(100, 8, 4), ('session', 'scope', 'other')
    ledger.import_idle(key, HistoricalCapacitySnapshot(imported.generation, imported.history,
        imported.ready_records, imported.allocated_records, imported.protected_records))

    def observe():
        for notice in of(stack.worker.notices(), 'history_capacity'):
            if notice.namespace == ('scope', 'other'):
                ledger.observe_idle(key, HistoricalCapacitySnapshot(notice.generation, notice.history,
                    notice.ready_records, notice.allocated_records, notice.protected_records))
        return ledger.snapshots[key]
    checkpoint = stack.worker.pool.entries[('scope', 'other')]
    child = checkpoint.authority
    assert frame_state(child) == at_p
    # Handles only: a test reference to a Record would keep its index owner alive.
    handles = {r.history_begin: (r.handle.slot, r.handle.generation) for r in child.records.values()}
    assert checkpoint.layerwise_pressure.maps == tuple(
        tuple((*handles[begin], begin, 4, dense) for begin, dense in rows) for rows in (((0, 0), (8, 4)), ((0, 0), (12, 4))))
    hold = checkpoint.layerwise_pressure.gc_hold
    assert hold is not None

    def live(begin):
        lease = child.memory.repository.pin(n.Handle(*handles[begin]))
        alive = bool(lease)
        if lease:
            lease.reset()
        return alive
    # Slot pressure on the child's own scope: the record outside every window goes
    # (logically: its payload is shared); the frame stays current and held, and block
    # 12, in layer 1's window only, stays under the recreated per-layer hold ...
    child.gc.scope.request_capacity(8)
    eventually(lambda: not live(4))
    assert live(0) and live(8) and live(12)
    assert child.gc.scope.stats['shared_released'] >= 1
    # ... until that hold is dropped.
    hold.release()
    eventually(lambda: not live(12))
    assert live(0) and live(8)
    # Two shared records left: the child's index charge drops from 2 records to 1, and
    # the scheduler learns it (as for any idle session; it never grows).
    assert imported.allocated_records == index_records(stack.archive, 4) == 2
    assert index_records(stack.archive, 2) == 1
    eventually(lambda: (observe().allocated_records, observe().ready_records) == (1, imported.ready_records + 2))


def test_agent_profile_gc_of_shared_records_never_grows_an_idle_charge(tmp_path, monkeypatch):
    """Historical GC retiring shared prefix records of an idle parent and an idle import:
    each session's reported history charge only shrinks (a retired shared record's index
    owner stays at its fraction until reaped), so the scheduler's idle observation never
    sees capacity consumed, and its view converges to the worker's."""
    stack = agent_stack(tmp_path, monkeypatch)
    worker = stack.worker
    parent, binding, _, registered = register_agent_prefix(stack)
    retired, = of(stack.registration_notices, 'retired')
    ledger = HistoricalAdmission(100, 8, 4)
    parent_key, child_key = ('session', *binding.namespace), ('session', 'scope', 'other')
    ledger.import_idle(parent_key, HistoricalCapacitySnapshot(retired.generation, retired.history,
        retired.ready_records, retired.allocated_records, retired.protected_records))
    ledger.transfer_to_pool(parent_key, registered.generation, registered.transferred, registered.pool_records)
    worker.receive(SessionCommands(image_imports=[SessionImport(('scope', 'other'), 7, registered.key)]))
    imported, = of(worker.notices(), 'imported')
    ledger.import_idle(child_key, HistoricalCapacitySnapshot(imported.generation, imported.history,
        imported.ready_records, imported.allocated_records, imported.protected_records))
    keys = {binding.namespace: parent_key, ('scope', 'other'): child_key}

    def observe():
        for notice in of(worker.notices(), 'history_capacity'):
            ledger.observe_idle(keys[notice.namespace], HistoricalCapacitySnapshot(notice.generation,
                notice.history, notice.ready_records, notice.allocated_records, notice.protected_records))
        return ledger.snapshots[parent_key], ledger.snapshots[child_key]
    child = worker.pool.entries[('scope', 'other')].authority
    # A reader still holding the indexes of blocks 4 and 12 (as a scoring snapshot may)
    # keeps their index owners pending after GC retires the records.
    readers = [r for authority in (parent, child) for r in authority.records.values() if r.history_begin in (4, 12)]
    pending = [(r.handle.slot, r.handle.generation) for r in readers]
    for authority in (parent, child):
        authority.gc.scope.request_capacity(8)
    eventually(lambda: observe() and all(key in authority.gc.pending_index for key, authority in
                                         zip(pending, (parent, parent, child, child))))
    for _ in range(20):
        observe()                     # never raises: the charge does not grow meanwhile
        time.sleep(.001)
    del readers

    def settled():
        views = observe()
        rows = [worker._history_rows(worker.pool.entries[namespace]) for namespace in keys]
        return all(view.ready_records == 6 and view.allocated_records == row[0]
                   for view, row in zip(views, rows))
    eventually(settled)
    # Both kept the frozen frame (blocks 0 and 8) and lost the shared blocks 4 and 12.
    for authority in (parent, child):
        assert sorted(r.history_begin for r in authority.records.values()) == [0, 8]
        assert authority.gc.scope.stats['shared_released'] == 2
    assert ledger.snapshots[child_key].allocated_records == index_records(stack.archive, 2) == 1
    assert stack.archive.pool_records() == 4          # the image still owns all four payloads
