"""P/M state ownership and cancelled partial-tail rebasing with real native LRU."""
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool
from vllm_kvmem.session_worker import verify_preserved_authority
from test_authority import payload
from test_core import Event
from test_session_state import fixture


def eventually(worker, predicate):
    end = time.monotonic()+5
    while True:
        worker.advance()
        if predicate():
            return
        assert time.monotonic() < end
        time.sleep(.001)


@pytest.mark.parametrize("boundaries", [False, True])
def test_pm_frames_survive_lru_and_cancelled_successor_rebases_only_its_partial_tail(tmp_path, boundaries):
    config = KVMemConfig(block_tokens=2, select_budget=4, gen_budget=4,
                        repository_blocks=16, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 24, 32, native_payloads=True)
    settings = n.HistoricalGcConfig()
    settings.mode, settings.background = n.HistoricalGcMode.DELETE, False
    settings.ready_low, settings.ready_high, settings.scan_blocks = 13, 14, 16
    worker = archive.enable_history_gc(settings)
    authority = RequestAuthority('a', config, archive, 2, 1, 2)
    if boundaries:
        from vllm_kvmem.query_boundary_history import QueryBoundaryHistory
        authority.query_boundaries = QueryBoundaryHistory(('tenant', 's'), block_tokens=2)
        authority.query_boundaries.update_message_spans([(0, 8, 'user'), (8, 20, 'assistant')])
    pool = PromptCheckpointPool(2, 8192, retain_prompt=True)
    adapter, conv, ssm, draft, _ = fixture()

    def publish():
        for job, _ in authority.writes:
            job.result(5)
        assert authority.flush()
        for job, _ in authority.writes:
            job.result(5)
        authority.poll()

    def restore(checkpoint, name):
        assert pool.claim(checkpoint.namespace, name, list(range(20))) is checkpoint
        plan = pool.prepare_claim_window(checkpoint)
        authority.memory.attach_completion(Event(True))
        assert authority.memory.commit_selection()
        assert pool.finish_claim(checkpoint, Event(True), retain=True) is authority
        plan.release_cpu_sources()

    prompt = terminal = None
    try:
        assert authority.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
        publish()
        plan = authority.memory.select({r.handle.slot: float(r.history_begin) for r in authority.records.values()})
        authority.memory.attach_completion(Event(True))
        assert authority.memory.commit_selection()
        plan.release_cpu_sources()
        del plan
        prompt = pool.capture(('tenant', 's'), 'a', authority, list(range(20)), adapter)
        if boundaries:
            assert prompt.graph_history.boundary_only
            assert prompt.graph_history.accepted_through == 8
        for _ in range(16):
            worker.advance()
            authority.poll()
        assert authority.gc_reclaimed_records == 2
        assert authority.memory.checkpoint_valid(prompt.window.prefix)
        assert [r.history_begin for r in authority.records.values()] == [0, 6]
        assert verify_preserved_authority(authority) == 2

        assert authority.append(*payload(8, 3), history_begin=8, local_rope_begin=4)
        publish()
        adapter.runner.req_states.num_computed_tokens_np[0] = 11
        adapter.runner.req_states.prefill_len.np[0] = 20
        adapter.runner._kvmem_capture_runtime = SimpleNamespace(
            last_steps={'a':[dict(history_begin=8, accepted=3, scheduled=3)]}, authorities={'a':authority})
        adapter.recurrent.conv_dim_first = False
        adapter.recurrent.conv_widths = {'gdn':4}
        terminal = pool.capture(prompt.namespace, 'a', authority, list(range(20)), adapter,
                                replace=True, terminal=True)
        pool.retire(terminal, Event(True))
        eventually(worker, lambda: pool.idle(terminal))
        model = terminal.model
        saved_state = tuple(tensor.clone() for tensor in (*model.recurrent.values(), *model.draft.values()))
        for branch in range(3):
            restore(terminal, f'branch-{branch}')
            assert authority.append(*payload(50+branch, 1), history_begin=11, local_rope_begin=7)
            publish()
            old_tail = next(r for r in authority.records.values() if r.history_begin == 10).handle
            pool.retire(terminal, Event(True))
            eventually(worker, lambda: pool.idle(terminal))
            assert authority.history_end == terminal.position == 11
            new_tail = terminal.window.records[-1]
            assert (new_tail[0], new_tail[1]) != (old_tail.slot, old_tail.generation)
            assert terminal.model is model and pool.prompts[prompt.namespace] is prompt
            assert authority.memory.checkpoint_valid(prompt.window.prefix)
            if boundaries:
                assert authority.query_boundaries.accepted_through == 11
                assert terminal.graph_history.accepted_through == 11
                assert prompt.graph_history.accepted_through == 8
            actual_state = (*model.recurrent.values(), *model.draft.values())
            assert all(torch.equal(actual, expected) for actual, expected in zip(actual_state, saved_state))
            assert verify_preserved_authority(authority) == 3
        pool.evict(terminal)
        eventually(worker, lambda: pool.poll_eviction(terminal))
        assert not pool.entries and not pool.prompts and not authority.records
    finally:
        if terminal is not None and pool.entries:
            pool.evict(terminal)
            eventually(worker, lambda: pool.poll_eviction(terminal))
        prompt = terminal = None
        eventually(worker, authority.close)
        eventually(worker, archive.close)
