"""Bounded graph-free metadata, including sparse history and P/M replay."""
import pytest

from vllm_kvmem.query_boundary_history import QueryBoundaryHistory
from vllm_kvmem.query_group_recall import build_halo_groups
from types import SimpleNamespace as NS


def test_gc_prunes_without_changing_live_endpoints_or_selected_plan_binding():
    history = QueryBoundaryHistory(('tenant', 's'), max_spans=4)
    spans = [(0, 64, 'user'), (64, 160, 'assistant'), (192, 256, 'tool')]
    history.update_message_spans(spans)
    history.advance(0, 256)
    binding = history.binding
    saved = history.snapshot()
    history.prune([(96, 128), (224, 256)])
    assert history.message_spans == tuple(spans[1:])
    assert saved.message_spans == tuple(spans)
    history.validate_binding(binding)
    assert saved.bytes > history.bytes
    saved.close()
    assert saved.bytes == 0


def test_future_input_is_kept_until_accepted_then_can_be_pruned():
    history = QueryBoundaryHistory('s')
    history.update_message_spans([(0, 64, 'user'), (64, 128, 'tool'), (128, 256, 'user')])
    history.advance(0, 96)
    history.prune([(0, 32)], future_begin=96)
    assert len(history.message_spans) == 3
    history.advance(96, 256)
    history.prune([(0, 32)], future_begin=256)
    assert history.message_spans == ((0, 64, 'user'),)


def test_empty_after_gc_does_not_turn_unknown_blocks_into_an_unconstrained_message():
    history = QueryBoundaryHistory('s')
    history.update_message_spans([(0, 32, 'user')])
    history.advance(0, 96)
    history.prune([(32, 96)])
    assert not history.message_spans and history.has_metadata
    records = [NS(history_begin=32, count=32), NS(history_begin=64, count=32)]
    groups, _, _ = build_halo_groups(records, [1., 2.], message_spans=history.message_spans,
                                    message_metadata_present=history.has_metadata)
    assert groups == [(32, 64), (64, 96)]
    # Unchanged legacy callers still allow grouping when no metadata existed.
    assert build_halo_groups(records, [1., 2.])[0] == [(32, 96), (32, 96)]


def test_bounded_install_filters_full_transcript_before_capacity_check():
    history = QueryBoundaryHistory('s', max_spans=2)
    full = [(i*32, (i+1)*32, 'tool') for i in range(1000)]
    history.update_message_spans(full, live_ranges=[(0, 32), (31968, 32000)])
    assert history.message_spans == (full[0], full[-1])
    saved = history.snapshot()
    with pytest.raises(BufferError, match='capacity'):
        history.update_message_spans(full)
    assert history.message_spans == saved.message_spans


def test_rewind_crops_unknown_end_then_exact_extension_and_conflict_are_checked():
    history = QueryBoundaryHistory('s')
    history.update_message_spans([(0, 96, 'user')])
    history.advance(0, 96)
    history.rewind(64)
    assert history.message_spans == ((0, 64, 'user'),)
    assert history.message_end_known == (False,)
    history.update_message_spans([(0, 128, 'user')], validated_prefix=64)
    assert history.message_spans == ((0, 128, 'user'),)
    before = history.binding
    with pytest.raises(ValueError, match='Conflicting'):
        history.update_message_spans([(0, 128, 'tool')], validated_prefix=64)
    assert history.binding == before


def test_preserve_verified_m_future_while_replaying_original_p_checkpoint():
    history = QueryBoundaryHistory('s')
    history.update_message_spans([(0, 64, 'user'), (64, 128, 'assistant')])
    history.advance(0, 64)
    prompt = history.snapshot(64)
    history.advance(64, 128)
    terminal = history.snapshot(128)
    binding = history.binding
    history.restore(terminal, preserve_future=True, preserved_prefix=prompt, until=128)
    with pytest.raises(RuntimeError, match='Stale'):
        history.validate_binding(binding)
    assert history.snapshot(64).accepted_through == 64
    history.advance(64, 96)
    history.advance(96, 128)
    assert history.accepted_through == 128
    history.advance(128, 160)
    with pytest.raises(BufferError, match='No exact'):
        history.snapshot(64)
    assert history.accepted_through == 160
    prompt.close(); terminal.close(); history.close()


def test_same_namespace_different_authority_cannot_import_a_snapshot():
    first, second = QueryBoundaryHistory('s'), QueryBoundaryHistory('s')
    snap = first.snapshot()
    with pytest.raises(ValueError, match='foreign'):
        second.restore(snap)
    clone = snap.clone()
    snap.close()
    first.restore(clone)
    assert clone.bytes > 0
    clone.close()
    with pytest.raises(ValueError, match='closed'):
        first.restore(clone)


@pytest.mark.parametrize('ranges', [[(32, 64), (0, 16)], [(0, 0)], [(False, 32)], [(0, 32, 64)]])
def test_invalid_prune_is_transactional(ranges):
    history = QueryBoundaryHistory('s')
    history.update_message_spans([(0, 64, 'user')])
    before = history.binding, history.message_spans
    with pytest.raises(ValueError):
        history.prune(ranges)
    assert (history.binding, history.message_spans) == before


def test_accepted_positions_must_be_continuous_without_verified_future():
    history = QueryBoundaryHistory('s')
    with pytest.raises(ValueError, match='continuous'):
        history.advance(1, 32)
    history.advance(0, 32)
    with pytest.raises(ValueError, match='continuous'):
        history.advance(0, 16)


def test_sparse_pruning_preserves_every_halo_group_against_full_message_metadata():
    import random
    randomizer = random.Random(114)
    for _ in range(200):
        spans, position = [], 0
        while position < 4096:
            position += randomizer.randint(0, 31)
            end = min(4096, position + randomizer.randint(1, 256))
            if end > position:
                spans.append((position, end, 'tool'))
            position = end
        starts = sorted(randomizer.sample(range(0, 4096, 32), randomizer.randint(1, 20)))
        records = [NS(history_begin=begin, count=32) for begin in starts]
        values = [randomizer.random() for _ in records]
        expected = build_halo_groups(records, values, message_spans=spans)
        history = QueryBoundaryHistory('s')
        history.update_message_spans(spans)
        history.advance(0, 4096)
        history.prune([(begin, begin + 32) for begin in starts])
        actual = build_halo_groups(records, values, message_spans=history.message_spans,
                                  message_metadata_present=history.has_metadata)
        assert actual == expected


def test_continuous_turns_keep_only_live_boundaries_and_checkpoint_ownership():
    history = QueryBoundaryHistory('s', max_spans=2)
    old_checkpoint = None
    for turn in range(100):
        begin, end = turn * 32, (turn + 1) * 32
        full = [(i*32, (i+1)*32, 'user') for i in range(turn + 1)]
        history.update_message_spans(full, validated_prefix=begin,
            live_ranges=[(0, 32), (begin, end)] if begin else [(0, end)])
        history.advance(begin, end)
        checkpoint = history.snapshot()
        if old_checkpoint is not None:
            assert old_checkpoint.accepted_through == begin
            old_checkpoint.close()
        old_checkpoint = checkpoint
        assert len(history.message_spans) <= 2
        assert history.accepted_through == end
    old_checkpoint.close()
    history.close()
