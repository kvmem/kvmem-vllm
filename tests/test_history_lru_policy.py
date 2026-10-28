"""LRU semantics over real accepted history, independent of retrieval scores.

The policy may retire a resident block's CPU copy; its old GPU lease remains
valid until normal window replacement. Explicit frame holds, sink and recent
blocks are retention rules. Read leases fence byte destruction, not retirement.
"""
import time

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive


class Ready:
    def query(self):
        return True


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


@pytest.fixture
def history(tmp_path):
    config = KVMemConfig(block_tokens=4, select_budget=16, gen_budget=16,
        repository_blocks=16, sink_blocks=1, recent_blocks=1)
    archive = MemoryArchive(str(tmp_path), 48, 32, native_payloads=True)
    settings = n.HistoricalGcConfig()
    settings.mode = n.HistoricalGcMode.DELETE
    settings.background = False
    settings.ready_low, settings.ready_high = 2, 4
    settings.scan_blocks, settings.delete_blocks = 16, 4
    worker = archive.enable_history_gc(settings)
    authority = RequestAuthority('lru', config, archive, 2, 1, 2,
                                 resident_records=16)
    try:
        yield authority, archive, worker
    finally:
        eventually(authority.close)
        eventually(archive.close)


def append(authority, count, local):
    start = authority.history_end
    raw = torch.arange(4*start, 4*(start+count), dtype=torch.float32)
    raw = raw.reshape(2, count, 1, 2).to(torch.bfloat16)
    assert authority.append(raw, raw.to(torch.uint8), history_begin=start,
                            local_rope_begin=local)
    for job, _ in tuple(authority.writes):
        job.result(5)
    authority.poll()


def records_by_start(authority):
    return {record.history_begin: record for record in authority.records.values()}


def select(authority, preferred):
    scores = {record.handle.slot: float(record.history_begin in preferred)
              for record in authority.records.values()}
    plan = authority.memory.select(scores)
    authority.memory.attach_completion(Ready())
    assert authority.memory.commit_selection()
    del plan


def advance(authority, worker, count=32):
    for _ in range(count):
        worker.advance()
        authority.poll()


def ordered_history(authority):
    append(authority, 32, 0)
    select(authority, {4, 24})
    # Public acceptance stamps only the selected window and new tail. Blocks
    # [8,24) were also scanned for selection but must remain cold.
    append(authority, 4, 16)
    return {start: record.handle for start, record in records_by_start(authority).items()}


def test_accepted_window_use_preserves_older_hot_blocks_over_newer_cold_blocks(history):
    authority, _, worker = history
    handles = ordered_history(authority)
    authority.gc.scope.request_capacity(10)  # 7 ready slots: delete 3 cold blocks.
    advance(authority, worker)
    assert authority.history_end == 36
    assert authority.gc_reclaimed_records == 3
    assert sorted(records_by_start(authority)) == [0, 4, 20, 24, 28, 32]
    assert all(not authority.memory.repository.pin(handles[start]) for start in (8, 12, 16))
    assert authority.memory.repository.pin(handles[4])  # Older, actually used.
    assert authority.memory.repository.pin(handles[32])  # Recent tail retained.


def test_full_scoring_and_raw_key_reads_do_not_heat_cold_history(history):
    authority, _, worker = history
    handles = ordered_history(authority)
    for _ in range(5):
        with authority.index_snapshot() as (records, means):
            assert len(records) == means.shape[1] == 9
        keys, stats = authority.read_score_keys([handles[8]])
        assert keys.shape[0] == 1
        assert stats['value_read_bytes'] == 0
        del keys, records, means
    authority.gc.scope.request_capacity(10)
    advance(authority, worker)
    assert sorted(records_by_start(authority)) == [0, 4, 20, 24, 28, 32]
    assert not authority.memory.repository.pin(handles[8])


def test_policy_hold_protects_a_cold_restore_frame_until_release(history):
    authority, _, worker = history
    handles = ordered_history(authority)
    hold = authority.gc.scope.protect([handles[8]])
    try:
        authority.gc.scope.request_capacity(10)
        advance(authority, worker)
        assert sorted(records_by_start(authority)) == [0, 4, 8, 24, 28, 32]
        assert authority.memory.repository.pin(handles[8])
        assert not authority.memory.repository.pin(handles[12])
    finally:
        del hold
    authority.gc.scope.request_capacity(11)
    advance(authority, worker)
    assert not authority.memory.repository.pin(handles[8])
    assert authority.memory.repository.pin(handles[0])  # Sink is still mandatory.
    assert authority.memory.repository.pin(handles[32])  # Recent is still mandatory.


def test_inflight_cpu_read_survives_retirement_but_does_not_resurrect_generation(history):
    authority, archive, worker = history
    handles = ordered_history(authority)
    reader = authority.gc.scope.admit_read(handles[8])
    assert reader is not None and reader[1] is not None
    old_bytes = bytes(memoryview(reader[1]))
    authority.gc.scope.request_capacity(10)
    advance(authority, worker)
    assert authority.gc.scope.stats['retired'] == 3
    assert authority.gc.scope.stats['cpu_released'] == 2
    assert authority.gc.scope.admit_read(handles[8]) is None
    assert not authority.memory.repository.pin(handles[8])
    assert bytes(memoryview(reader[1])) == old_bytes
    assert archive.io_usage()['cpu_authority_bytes'] == 7*48
    del reader
    advance(authority, worker)
    assert authority.gc.scope.stats['cpu_released'] == 3
    assert authority.gc_reclaimed_records == 3
    assert archive.io_usage()['cpu_authority_bytes'] == 6*48


def test_many_small_repository_turns_keep_logical_holes_and_reject_stale_handles(history):
    authority, archive, worker = history
    append(authority, 32, 0)
    old_handles = {}
    reused_slots = set()
    for _ in range(40):
        by_start = records_by_start(authority)
        # Choosing recent pages does not renumber the retained logical history.
        preferred = set(sorted(by_start)[-3:])
        select(authority, preferred)
        authority.gc.scope.request_capacity(10)
        advance(authority, worker)
        for record in authority.records.values():
            assert record.count == 4
            key = record.handle.slot
            previous = old_handles.get(key)
            if previous is not None and previous.generation != record.handle.generation:
                assert not authority.memory.repository.pin(previous)
                reused_slots.add(key)
            old_handles[key] = record.handle
        authority.gc.scope.request_capacity(0)
        append(authority, 8, 16)
        assert len(authority.records) <= 8
        assert archive.tier.used_slots <= 8
        assert 0 in records_by_start(authority)
    assert authority.history_end == 352
    assert authority.gc_reclaimed_records >= 70
    assert len(reused_slots) >= 8
    starts = sorted(records_by_start(authority))
    assert any(b-a > 4 for a, b in zip(starts, starts[1:]))
    with authority.index_snapshot() as (records, means):
        assert [record.history_begin for record in records] == starts
        assert means.shape[1] == len(starts)
