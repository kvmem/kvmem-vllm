"""Request-local publication fences with real CPU archive ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.action_attention import ActionHistory, ActionMetadata, ActionPacket, ActionRuntime
from vllm_kvmem.capture_runtime import CaptureRuntime, CapturedPacket
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.terminal import TerminalPrefix
from test_authority import setup, payload, drain
from test_terminal_prefix import request


def fixture(tmp_path, *, threshold=4, batched=True):
    a, archive = setup(tmp_path/'a')
    b, _ = setup(tmp_path/'b', 'b', archive)
    for owner in (a, b):
        assert owner.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
        drain(owner)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.raw_dtype = torch.bfloat16
    runtime.options = RuntimeOptions(memory=a.memory.config, async_cpu_scatter=True,
        cpu_scatter_min_tokens=threshold, batched_cpu_append=batched,
        action_scope='non-thinking', action_interval=1)
    runtime.config, runtime.archive = a.memory.config, archive
    runtime.runner = NS(vllm_config=NS(max_concurrent_batches=1))
    runtime.output = tmp_path
    runtime.layers, runtime.heads, runtime.dim = [None, None], 1, 2
    runtime.row_elements, runtime.resident_records = 4, 4
    runtime.authorities = {'a': a, 'b': b}
    runtime.accepted, runtime.rejected = {'a': 4, 'b': 4}, {'a': 0, 'b': 0}
    runtime.last_steps = {'a': [], 'b': []}
    runtime.terminal_prefixes, runtime.terminal_discarded = {}, {}
    runtime.batches, runtime._progress_batch, runtime.completed = 2, 0, []
    runtime.retiring = {}
    runtime._write_archive_report = lambda: None
    runtime.queue = []
    runtime.downloads = NS(buffers=[object(), object()], pending={})
    runtime.released = []

    def release(download):
        assert runtime.downloads.pending[download.slot] is download
        assert download.query()
        del runtime.downloads.pending[download.slot]
        runtime.released.append(download)
        download.owners = ()
        return True
    runtime.downloads.release = release
    runtime.archive_capacity()  # Freeze the committed prefix before any append.
    return runtime, a, b, archive


def packet(runtime, specs, *, rejected=None, confirmed=True, ready=True):
    """specs = (request id, logical begin, candidate count)."""
    ids, histories, sizes = zip(*specs)
    starts = [0]
    for size in sizes:
        starts.append(starts[-1]+size)
    raw = torch.cat([payload(begin, count)[0] for _, begin, count in specs], dim=1)
    header = torch.tensor([*starts, *histories, *histories, *(rejected or (0,)*len(ids))], dtype=torch.int64)
    data = torch.cat((header.view(torch.uint8), raw.contiguous().view(torch.uint8).flatten(),
                      raw.to(torch.uint8).flatten()))
    slot = next(i for i in range(2) if i not in runtime.downloads.pending)
    download = NS(payload=data, slot=slot, query=lambda: ready, owners=(object(),),
                  event=NS(synchronize=lambda: pytest.fail('Unrelated DMA was synchronized')))
    runtime.downloads.pending[slot] = download
    result = CapturedPacket(tuple(ids), starts[-1], download, confirmed=confirmed,
                            request_tokens=tuple(sizes))
    runtime.queue.append(result)
    return result


def close(runtime, owners, archive):
    runtime.join_cpu_scatter()
    executor = getattr(runtime, '_cpu_scatter_executor', None)
    if executor is not None:
        executor.shutdown(wait=True)
    for owner in owners:
        drain(owner)
        assert owner.close()
    assert archive.close()


@pytest.mark.parametrize('batched', [False, True])
def test_decode_publishes_while_shared_packet_prefill_is_copying(tmp_path, monkeypatch, batched):
    runtime, a, b, archive = fixture(tmp_path, batched=batched)
    shared = packet(runtime, [('b', 4, 4), ('a', 4, 1)])
    entered, release = Event(), Event()
    original = b.memory.begin_append

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(b.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert runtime.publication_blocked_requests() == {'b'}
        assert runtime.drain_requests({'a'}, wait=True)
        assert a.history_end == 5 and b.history_end == 4
        assert runtime.publication_ready({'a'}) and not runtime.publication_ready({'b'})
        assert shared.published == {1} and shared.consuming == {0}
        assert shared.download.owners and not runtime.released
        # Conservative credit never observes b's reserved-but-uncommitted rows.
        assert runtime.archive_capacity()['b'].history_end == 4
        assert not runtime._cpu_scatter_future.done()
    finally:
        release.set()
        close(runtime, (a, b), archive)
    assert runtime.released == [shared.download]


def test_small_owner_reuses_second_slot_for_three_steps_while_prefill_waits(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path)
    shared = packet(runtime, [('b', 4, 4), ('a', 4, 1)])
    entered, release = Event(), Event()
    original = b.memory.begin_append

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(b.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert runtime.drain_requests({'a'}, wait=True)
        for step in range(3):
            latest = packet(runtime, [('a', 5+step, 1)], confirmed=False)
            assert not runtime.publication_ready({'a'})
            assert not runtime.drain_requests({'a'}, wait=True)
            assert a.history_end == 5+step and latest.download.owners
            runtime.begin_step()
            runtime.ensure_download_slot({'a'})
            assert a.history_end == 6+step
            assert any(item is latest.download for item in runtime.released)
            assert all(item is not shared.download for item in runtime.released)
            assert len(runtime.downloads.pending) == 1
            assert not runtime._cpu_scatter_future.done()
    finally:
        release.set()
        close(runtime, (a, b), archive)


def test_unrelated_unready_dma_does_not_block_or_release_decode(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    slow = packet(runtime, [('b', 4, 4)], ready=False)
    fast = packet(runtime, [('a', 4, 1)])
    assert runtime.drain_requests({'a'}, wait=True)
    assert a.history_end == 5 and b.history_end == 4
    assert runtime.released == [fast.download] and slow.download.owners
    slow.download.query = lambda: True
    assert runtime.drain_requests({'b'}, wait=True)
    close(runtime, (a, b), archive)


def test_confirmed_terminal_prefix_clips_mtp_before_any_publication(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    shared = packet(runtime, [('a', 4, 8), ('b', 4, 4)], rejected=(2, 0), confirmed=False)
    assert not runtime.start_cpu_scatter()
    assert not runtime.drain_requests({'a'}, wait=True)
    assert a.history_end == 4 and not shared.published
    terminal = {'a': TerminalPrefix.from_request(request(10, 0, 8, 4))}
    runtime.begin_step({'a'}, terminal)
    runtime.finish({'a'}, terminal)
    report = runtime.completed[0]
    assert report['history_end'] == 7
    assert report['rejected_tokens'] == 2 and report['terminal_discarded_tokens'] == 3
    assert runtime.publication_ready({'a'}) and b.history_end == 4
    assert not runtime.released and shared.download.owners
    assert runtime.drain_requests({'b'}, wait=True)
    close(runtime, (b,), archive)


def test_running_worker_cannot_consume_new_unconfirmed_sample(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path)
    old = packet(runtime, [('b', 4, 4)])
    entered, release = Event(), Event()
    original = b.memory.begin_append
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(b.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        latest = packet(runtime, [('a', 4, 8)], rejected=(2,), confirmed=False)
        release.set()
        runtime.join_cpu_scatter()
        assert a.history_end == 4 and b.history_end == 8
        assert runtime.released == [old.download] and latest.download.owners
        assert not latest.published and not latest.consuming
        terminal = {'a': TerminalPrefix.from_request(request(10, 0, 8, 4))}
        runtime.begin_step({'a'}, terminal)
        runtime.finish({'a'}, terminal)
        assert runtime.completed[0]['history_end'] == 7
        assert runtime.completed[0]['terminal_discarded_tokens'] == 3
        assert runtime.released == [old.download, latest.download]
    finally:
        release.set()
        close(runtime, (b,), archive)


def test_action_parser_and_page_mapping_wait_only_for_its_owner(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path, threshold=8)
    shared = packet(runtime, [('b', 4, 8), ('a', 4, 4)], rejected=(0, 1))
    ids = [10, 11, 12, 13, 14]
    action = ActionRuntime.__new__(ActionRuntime)
    action.runtime, action.runner, action.options, action.ids = runtime, runtime.runner, runtime.options, ids
    action.metadata_early_batches = action.candidate_samples = action.packets = 0
    action.capture_seconds = 0.0
    action.groups = {'attn': 0}
    action.q_dtypes = {'attn': None}
    module = NS(head_dim=2, num_kv_heads=1, attn=NS(layer_name='attn', kv_cache=None,
                _k_scale=1., _q_scale=1., impl=NS(scale=1.)),
                rotary_emb=NS(cos_sin_cache=None))
    action.layers = [(0, module)]
    runtime.runner.req_states = NS(req_id_to_index={'a': 0})
    runtime.runner.block_tables = NS(block_tables=[NS(gpu=torch.zeros((1, 4), dtype=torch.int64))],
        num_blocks=NS(np=torch.tensor([[4]])), kernel_block_sizes=(4,))
    runtime.action = action
    history = a.action_history = ActionHistory(runtime.options, ids)
    history.begin('a', [14, 12])
    mask = history.parser.prepare([20, 13, 21, 22])
    rows = [i for i in range(4) if mask >> i & 1]
    shared.action_inputs['a'] = ActionPacket(history, rows, [0, 4], torch.tensor([[.25, .75], [1., 0.]]), None)
    entered, release = Event(), Event()
    original = b.memory.begin_append

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(b.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        # Reject the old packet-wide API if Action tries to use it.
        monkeypatch.setattr(runtime, 'drain', lambda **_: pytest.fail('Global Action publication barrier'))
        import vllm_kvmem.action_attention as capture
        import vllm_kvmem.action_scoring as scoring
        seen = []
        def attention(*args, **kwargs):
            starts, counts = args[6:8]
            seen.append((starts.tolist(), counts.tolist()))
            return torch.tensor([[.1, .4, .5]])
        monkeypatch.setattr(scoring, 'page_attention', attention)
        monkeypatch.setattr(capture, 'native_vllm_fp8_views', lambda *args: (None, None))
        empty = torch.empty
        monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: empty(*args,
            **{key:value for key,value in kwargs.items() if key!='pin_memory'}))
        monkeypatch.setattr(torch.cuda, 'Event', lambda: NS(record=lambda: None, synchronize=lambda: None))
        batch = NS(num_reqs=1, req_ids=('a',), is_prefilling_np=[False], input_ids=torch.tensor([20]))
        action.metadata = ActionMetadata(batch, (0,), ((0, 1),), torch.tensor([[20], [7], [7]]), None)
        prepared = action.capture([NS(query=torch.zeros((1, 1, 2)))], batch)
        assert prepared['a'].rows == [0]
        assert seen == [([0, 4, 7], [4, 3, 1])]
        assert prepared['a'].history == [0, 4, 4]  # Rejected row is not in the dense frame.
        assert a.history_end == 7 and b.history_end == 4
        assert history.samples == 1 and history.mass == {0: .25, 4: .75}
        assert history.parser.state == (False, False, 1)  # New candidate not yet accepted.
        assert not runtime._cpu_scatter_future.done() and not runtime.released
    finally:
        release.set()
        close(runtime, (a, b), archive)


def test_failure_keeps_shared_dma_owned_and_reaches_every_waiter(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path)
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 4)])
    error = OSError('owner publication failed')
    monkeypatch.setattr(a.memory, 'begin_append', lambda *args, **kwargs: (_ for _ in ()).throw(error))
    for wanted in ({'a'}, {'b'}):
        with pytest.raises(OSError, match='owner publication failed') as caught:
            runtime.drain_requests(wanted, wait=True)
        assert caught.value is error
    assert shared.download.owners and not runtime.released
    with pytest.raises(OSError, match='owner publication failed'):
        runtime.publication_blocked_requests()
    # A fatal transaction retains its source; test-owned teardown only.
    runtime.queue.clear()
    runtime.downloads.pending.clear()
    close(runtime, (a, b), archive)


def test_same_owner_cannot_overtake_inflight_publication(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path, threshold=1)
    first = packet(runtime, [('a', 4, 1)])
    second = packet(runtime, [('a', 5, 1)])
    entered, release = Event(), Event()
    original = a.memory.begin_append

    def delayed(*args, **kwargs):
        if not entered.is_set():
            result = original(*args, **kwargs)
            entered.set()
            assert release.wait(10)
            return result
        return original(*args, **kwargs)
    monkeypatch.setattr(a.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert not runtime.drain_requests({'a'}, wait=False)
        assert not second.consuming and a.history_end == 4
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiter = executor.submit(runtime.drain_requests, {'a'}, True)
            assert not waiter.done()
            release.set()
            assert waiter.result(5)
        assert a.history_end == 6 and runtime.released == [first.download, second.download]
    finally:
        release.set()
        close(runtime, (a, b), archive)


@pytest.mark.parametrize('explicit_finish', [False, True])
def test_finished_owner_is_deferred_once_until_clipped_cpu_prefix_is_ready(tmp_path, explicit_finish):
    from dataclasses import replace
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm_kvmem.connector import ControlMetadata
    runtime, a, b, archive = fixture(tmp_path)
    shared = packet(runtime, [('b', 4, 8), ('a', 4, 1)], rejected=(2, 0), confirmed=False)
    prefix = TerminalPrefix.from_request(request(10, 0, 8, 4))
    original = replace(SchedulerOutput.make_empty(), finished_req_ids={'b'},
        num_scheduled_tokens={'a': 1}, total_num_scheduled_tokens=1,
        kv_connector_metadata=ControlMetadata(terminal_prefixes={'b': prefix}))
    # The final scheduler decision clips b before the worker can claim it.
    runtime.begin_step(original.finished_req_ids, original.kv_connector_metadata.terminal_prefixes)
    held = runtime.defer_terminals(original)
    assert held.finished_req_ids == set() and held.num_scheduled_tokens == {'a': 1}
    assert not held.kv_connector_metadata.terminal_prefixes
    assert runtime.terminal_prefixes['b'] is prefix and not runtime.retirement_done('b')
    assert runtime.drain_requests({'a'}, wait=True) and a.history_end == 5
    # An intervening scheduler step does not repeat finished_req_ids.
    empty = replace(SchedulerOutput.make_empty(), kv_connector_metadata=ControlMetadata())
    assert not runtime.defer_terminals(empty).finished_req_ids
    assert runtime.drain_requests({'b'}, wait=True) and b.history_end == 7
    assert not runtime.retirement_done('b')  # Native finish has not been replayed yet.
    if explicit_finish:
        # Graceful shutdown may explicitly finalize a published owner without
        # another scheduler iteration replaying its deferred finished notice.
        runtime.finish({'b'})
    else:
        resumed = runtime.defer_terminals(empty)
        assert resumed.finished_req_ids == {'b'}
        assert resumed.kv_connector_metadata.terminal_prefixes['b'] is prefix
        runtime.finish(resumed.finished_req_ids, resumed.kv_connector_metadata.terminal_prefixes)
    report = runtime.completed[0]
    assert report['history_end'] == 7 and report['rejected_tokens'] == 2
    assert report['terminal_discarded_tokens'] == 3
    assert runtime.retirement_done('b')
    assert not runtime.defer_terminals(empty).finished_req_ids
    assert runtime.released == [shared.download]
    close(runtime, (a,), archive)
