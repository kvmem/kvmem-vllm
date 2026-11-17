"""KVMEM_NATIVE_WINDOW_FRAME: the native window/frame matcher answers exactly what the original
path concludes (canonical_segments(window_segments(...)) equal to the frozen frame's prefix, as
NativeAttentionStats.resolve compares), and the rows taken from the frame equal the canonical rows."""
import numpy as np
import pytest

from vllm_kvmem.serving.native_attention_planning import canonical_segments, make_decode_frame
from test_native_restore_frame import fixture

BT = 4


def python_rows(memory, history, local, count):
    starts, counts, keys = memory._store.action_window_segments(history, local, count, BT)
    try:
        return canonical_segments(starts, counts, keys, local + count, BT)
    except ValueError:
        return None


def prefix(frame, extent):
    end = int(np.searchsorted(frame['starts'], extent - 1, side='right'))
    return np.stack((frame['starts'][:end], np.minimum(frame['counts'][:end], extent - frame['starts'][:end]),
                     frame['history'][:end]), axis=1)


def frame_at(memory, history, local, count, context):
    rows = python_rows(memory, history, local, count)
    return make_decode_frame(rows[:, 0], rows[:, 1], rows[:, 2].tolist(), history, local, count, context, BT)


@pytest.mark.parametrize('kind,history,local', [('dense', 24, 24), ('sparse', 24, 12), ('sparse', 20, 8),
                                                ('partial', 22, 10), ('partial', 21, 9)])
def test_matcher_agrees_with_the_original_comparison(kind, history, local):
    memory, _ = fixture(kind, current='same')
    try:
        context = local + 40
        frame = frame_at(memory, history, local, 1, context)
        checked = matched = 0
        for advance in range(0, 30):
            for count in (0, 1, 3, 5):
                hb, lb = history + advance, local + advance
                if lb + count > context or lb + count == 0:
                    continue
                rows = python_rows(memory, hb, lb, count)
                want = rows is not None and np.array_equal(prefix(frame, lb + count), rows)
                got = memory._store.action_window_matches_frame(hb, lb, count, BT, frame['starts'],
                                                                frame['counts'], frame['history'])
                assert got == want, (advance, count)
                checked += 1
                matched += got
                if got:
                    assert np.array_equal(prefix(frame, lb + count), rows)
        assert checked and matched == checked    # one window: its frame covers every decode step
        # A frame differing inside the compared prefix is never confirmed (nor by the original check).
        rows = python_rows(memory, history, local, 1)
        for field in ('starts', 'counts', 'history'):
            spoiled = {key: value.copy() for key, value in frame.items() if key != 'layout'}
            spoiled[field][1] += 1
            assert not np.array_equal(prefix(spoiled, local + 1), rows)
            assert not memory._store.action_window_matches_frame(history, local, 1, BT, spoiled['starts'],
                                                                 spoiled['counts'], spoiled['history'])
    finally:
        assert memory.close()


@pytest.mark.parametrize('kind,history,local', [('sparse', 24, 12), ('partial', 22, 10)])
def test_a_changed_selection_is_never_confirmed(kind, history, local):
    old, _ = fixture(kind, current='same')
    new, _ = fixture(kind, current='different')
    try:
        frame = frame_at(old, history, local, 1, local + 40)
        for advance in range(0, 12):
            for count in (1, 4):
                hb, lb = history + advance, local + advance
                rows = python_rows(new, hb, lb, count)
                want = rows is not None and np.array_equal(prefix(frame, lb + count), rows)
                got = new._store.action_window_matches_frame(hb, lb, count, BT, frame['starts'],
                                                             frame['counts'], frame['history'])
                assert got == want and not got
    finally:
        assert old.close() and new.close()


def test_invalid_arguments_answer_false():
    memory, _ = fixture('dense', current='same')
    try:
        frame = frame_at(memory, 24, 24, 1, 40)
        assert not memory._store.action_window_matches_frame(-1, 24, 1, BT, frame['starts'], frame['counts'],
                                                             frame['history'])
        assert not memory._store.action_window_matches_frame(24, 24, 1, BT, frame['starts'][:-1],
                                                             frame['counts'], frame['history'])
        empty = np.empty(0, dtype=np.int64)
        assert not memory._store.action_window_matches_frame(24, 24, 1, BT, empty, empty, empty)
    finally:
        assert memory.close()


def test_flag_is_validated(monkeypatch):
    from vllm_kvmem.serving import native_attention as na
    monkeypatch.setattr(na, '_WINDOW_FRAME', None)
    monkeypatch.setenv('KVMEM_NATIVE_WINDOW_FRAME', 'yes')
    with pytest.raises(ValueError):
        na.window_frame_enabled()
    monkeypatch.setattr(na, '_WINDOW_FRAME', None)
