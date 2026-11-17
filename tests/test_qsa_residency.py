"""Host checks of the residency plan reference (adapters/qsa_residency.py)."""
import pytest

from vllm_kvmem.adapters.qsa_residency import reference_plan


def test_rewrites_hot_ring_and_invalid_entries():
    group_slot, slot_group, refs = {}, [-1] * 4, [0] * 4
    rows = [[40, 41, 8, 9, -1], [8, 10, 100, -1, -1]]
    # ring holds positions >= 96; groups 10 (40..43) and 2 (8..11) need hot slots
    out, misses, victims = reference_plan(rows, 96, group_slot, slot_group, refs, hot_tokens=16, ring_tokens=32)
    assert misses == [2, 10] and victims == [0, 1]
    assert out == [[4, 5, 0, 1, -1], [0, 2, 16 + 100 % 32, -1, -1]]
    assert group_slot == {2: 0, 10: 1} and refs == [1, 1, 0, 0]


def test_eviction_order_and_sweep():
    group_slot, slot_group, refs = {}, [-1] * 3, [0] * 3
    reference_plan([[0, 4, 8]], 1000, group_slot, slot_group, refs, 12, 32)      # groups 0,1,2 fill
    # group 0 hit (pinned); misses 3 and 4 need two victims among slots 1,2 (both referenced):
    out, misses, victims = reference_plan([[0, 12, 16]], 1000, group_slot, slot_group, refs, 12, 32)
    assert misses == [3, 4] and victims == [1, 2]
    assert refs == [1, 1, 1] and group_slot == {0: 0, 3: 1, 4: 2}
    assert out == [[0, 4, 8]]
    with pytest.raises(OverflowError):
        reference_plan([[0, 20, 24, 28]], 1000, group_slot, slot_group, refs, 12, 32)


def test_clear_slots_are_taken_before_marked_ones():
    group_slot, slot_group, refs = {}, [-1] * 3, [0] * 3
    reference_plan([[0, 4, 8]], 1000, group_slot, slot_group, refs, 12, 32)
    refs[1] = 0                                   # slot 1 not recently used
    _, misses, victims = reference_plan([[20]], 1000, group_slot, slot_group, refs, 12, 32)
    assert misses == [5] and victims == [1]
    assert refs == [1, 1, 1]                      # no sweep: a clear slot sufficed


def test_entries_after_the_row_position_are_dropped():
    group_slot, slot_group, refs = {}, [-1] * 4, [0] * 4
    # row 0 is at position 101, row 1 at 102: QSA selected 102 and 103 for both (same verify interval)
    rows = [[8, 102, 103, -1], [8, 102, 103, -1]]
    out, misses, _ = reference_plan(rows, 96, group_slot, slot_group, refs, 16, 32, limits=[101, 102])
    assert misses == [2]
    assert out == [[0, -1, -1, -1], [0, 16 + 102 % 32, -1, -1]]
