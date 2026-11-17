"""Host checks of the hot-buffer replay (scripts/qsa_hot_buffer_sim.py) on synthetic traces."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def sim():
    spec = importlib.util.spec_from_file_location('qsa_hot_buffer_sim', ROOT / 'scripts/qsa_hot_buffer_sim.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, steps):
    """steps: [(call, slot, layer, position, [groups...]), ...] one verify row each."""
    rows = len(steps)
    groups = np.full((rows, 513), -1, np.int32)
    for i, (_, _, _, _, chosen) in enumerate(steps):
        groups[i, :len(chosen)] = chosen
    np.savez_compressed(tmp_path / 'qsa-topk-1-00000.npz',
                        layer=np.array([s[2] for s in steps], np.int16),
                        call=np.array([s[0] for s in steps], np.int64),
                        batch_row=np.zeros(rows, np.int32),
                        state_slot=np.array([s[1] for s in steps], np.int32),
                        query_position=np.array([s[3] for s in steps], np.int64),
                        groups=groups)
    (tmp_path / 'trace.json').write_text('{"layers": ["a"], "group_tokens": 4}')


def test_union_of_verify_rows_and_lru_misses(sim, tmp_path):
    # One request, one layer. Positions far beyond the ring so nothing is ring-resident.
    base = 100000
    _write(tmp_path, [
        (0, 3, 0, base, [1, 2]), (0, 3, 0, base + 1, [2, 3]),   # union {1,2,3}: 3 cold misses
        (1, 3, 0, base + 2, [1, 4]), (1, 3, 0, base + 3, [4]),  # union {1,4}: miss 4
        (2, 3, 0, base + 4, [1, 2, 3]),                          # hits 1,3; 2 was evicted? capacity 3 groups
    ])
    trace = sim.load([tmp_path])
    stream = sim.streams(trace['decode'])[(3, 0)]
    assert [len(step[1]) for step in stream] == [3, 2, 3]
    rows = sim.replay(stream, capacity_groups=3, ring_tokens=16, policy='lru')
    # Step 1 evicts LRU group 2 (3 was used in the same step, 1 is needed), so step 2 misses 2 only.
    assert [(r[3], r[4], r[5]) for r in rows] == [(3, True, True), (1, True, False), (1, True, False)]


def test_ring_groups_and_overflow(sim, tmp_path):
    position = 1000
    ring = 64
    # Groups at or after (position - ring + 1) // 4 live in the ring and never need hot slots.
    newest = (position - ring + 1 + 3) // 4
    _write(tmp_path, [
        (0, 1, 0, position, [newest, newest + 1, 5]),
        (1, 1, 0, position + 1, [5, 6, 7, 8]),       # 4 groups outside the ring, capacity 3: unfit
    ])
    trace = sim.load([tmp_path])
    rows = sim.replay(sim.streams(trace['decode'])[(1, 0)], capacity_groups=3, ring_tokens=ring, policy='clock')
    assert rows[0][2] == 1 and rows[0][3] == 1 and rows[0][4]
    assert rows[1][2] == 4 and not rows[1][4]


def test_refresh_restarts_the_frame(sim, tmp_path):
    _write(tmp_path, [
        (0, 2, 0, 50000, [10]), (1, 2, 0, 50001, [10]),
        (2, 2, 0, 20000, [10]),   # positions moved back: a refresh, the cache starts cold
    ])
    trace = sim.load([tmp_path])
    rows = sim.replay(sim.streams(trace['decode'])[(2, 0)], capacity_groups=4, ring_tokens=16, policy='lru')
    assert [(r[3], r[5]) for r in rows] == [(1, True), (0, False), (1, True)]


def test_simulate_reports_bytes_per_step(sim, tmp_path):
    base = 100000
    _write(tmp_path, [(c, 0, 0, base + c, [c, c + 1]) for c in range(4)])
    report = sim.simulate(sim.load([tmp_path]), hot_pages=(1,), policies=('lru',), ring_tokens=16)
    row = report['decode'][0]
    assert row['request_layer_steps'] == 4 and row['unfit_steps'] == 0
    # After the cold step each call brings one new group: 4 tokens x 1 KiB.
    assert row['bytes_per_decode_step']['mean'] == 4 * 1024
    assert row['cold_fill_bytes'] == 2 * 4 * 1024


def test_nru_prefers_unreferenced_slots_and_sweeps(sim):
    cache = sim.NRU(3)
    assert cache.access({1, 2, 3}) == 3            # fills free slots
    assert cache.access({1}) == 0                   # all three still referenced
    # Miss 4: no clear slot among unpinned {2, 3}, so the sweep clears them and 4 takes one.
    assert cache.access({1, 4}) == 1
    assert len(cache.where) == 3 and 4 in cache.where and 1 in cache.where
    with pytest.raises(OverflowError):
        cache.access({5, 6, 7, 8})
