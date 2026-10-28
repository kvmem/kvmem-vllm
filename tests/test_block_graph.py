import math
import pytest

from vllm_kvmem.block_graph import BlockGraph, GraphBudget, GraphSample


def accept(graph, begin, end, rows=()):
    with graph.prepare(begin, end, rows) as update:
        update.commit()


def row(position, targets=(0, 4, 8), masses=(.2, .3, .5), kind='output'):
    return GraphSample(position, targets, masses, kind)


def test_partial_survives_callbacks_and_nonsampled_completion():
    graph = BlockGraph('s', 4)
    accept(graph, 0, 9, [row(8, kind='prefill')])
    assert graph.partial.samples == 1 and graph.rows == ()
    accept(graph, 9, 10, [row(9)])
    assert graph.partial.prefill_samples == graph.partial.output_samples == 1
    accept(graph, 10, 12)
    stored, = graph.rows
    assert stored.source == 8 and stored.targets == (4, 0)
    assert stored.weights == pytest.approx((.3, .2))
    assert stored.samples == 2 and graph.partial is None
    assert graph.completed_through == graph.accepted_through == 12
    graph.close()
    assert graph.budget.used == 0


def test_split_and_combined_multi_block_prefill_are_equal():
    inputs = [row(p, (0, 4, 8), (.5, .5, 0.) if p < 8 else (.5, .3, .2), 'prefill')
              for p in (5, 6, 8, 10)]
    combined, split = BlockGraph('a', 4), BlockGraph('b', 4)
    accept(combined, 0, 12, inputs)
    accept(split, 0, 6, inputs[:1])
    accept(split, 6, 9, inputs[1:3])
    accept(split, 9, 12, inputs[3:])
    assert combined.rows == split.rows
    assert [r.source for r in combined.rows] == [4, 8]
    assert [r.samples for r in combined.rows] == [2, 2]


def test_complete_distribution_topk_not_per_sample_topk():
    graph = BlockGraph('a', 4)
    # Target 0 is fifth in each row, but its consistent .1 outranks any
    # alternating target's .0875 after aggregation. Four row-local maxima
    # must therefore not be discarded before averaging the full source.
    targets = tuple(range(0, 40, 4))
    first = (.1, .175, .175, .175, .175, 0, 0, 0, 0, .2)
    second = (.1, 0, 0, 0, 0, .175, .175, .175, .175, .2)
    accept(graph, 0, 44, [row(40, targets, first), row(41, targets, second)])
    stored, = graph.rows
    assert stored.targets == (36, 0, 4, 8)
    assert stored.weights == pytest.approx((.2, .1, .0875, .0875))


def test_duplicate_target_segments_merged_once_and_counted_once():
    graph = BlockGraph('a', 4)
    accept(graph, 0, 12, [row(9, (0, 4, 4, 8), (.1, .2, .3, .4))])
    stored, = graph.rows
    assert stored.targets == (4, 0)
    assert stored.weights == pytest.approx((.5, .1))
    assert stored.samples == 1


def test_completed_sources_are_causal_topk_with_deterministic_ties():
    graph = BlockGraph('a', 4, topk=2)
    accept(graph, 0, 20, [row(19, (16, 12, 8, 4, 0), (.2,) * 5)])
    assert graph.rows[0].targets == (0, 4)
    assert graph.rows[0].retained_mass == pytest.approx(.4)


def test_unsampled_sources_identity_and_watermark():
    graph = BlockGraph('a', 4)
    accept(graph, 0, 128)
    assert not graph.rows and graph.partial is None
    assert graph.stats()['no_sample_sources'] == 32
    assert graph.completed_through == 128


def test_replay_deduplicates_by_accepted_positions_not_request_id():
    graph = BlockGraph('s', 4)
    accept(graph, 0, 10, [row(8)])
    binding = graph.binding
    accept(graph, 0, 10, [row(8)])
    assert graph.binding == binding
    accept(graph, 8, 12, [row(8), row(10)])
    assert graph.rows[0].samples == 2


@pytest.mark.parametrize('bad', [row(12), row(8, (0,), (float('nan'),)),
    row(8, (0,), (-1.,)), row(8, (0,), (.9,)), row(8, (1,), (1.,)),
    row(8, (0, 4), (1.,)), row(8, (0,), (1.,), 'thinking')])
def test_row_validation_is_atomic(bad):
    graph = BlockGraph('a', 4)
    baseline = graph.bytes, graph.binding
    with pytest.raises(ValueError):
        graph.prepare(0, 12, [bad])
    assert (graph.bytes, graph.binding) == baseline
    assert not graph.rows and graph.partial is None


def test_complete_row_roundoff_normalized_once():
    graph = BlockGraph('a', 4)
    accept(graph, 0, 12, [row(8, (0, 4), (.502, .501))])
    assert math.fsum(graph.rows[0].weights) == pytest.approx(1.)


def test_duplicate_source_rejected_before_mutation_and_gap_rejected():
    graph = BlockGraph('a', 4)
    with pytest.raises(ValueError):
        graph.prepare(0, 12, [row(8), row(8)])
    with pytest.raises(ValueError):
        graph.prepare(1, 12)
    assert graph.accepted_through == 0


def test_partial_snapshot_roundtrip_excludes_later_source_statistics():
    graph = BlockGraph('a', 4)
    accept(graph, 0, 9, [row(8)])
    snap = graph.snapshot()
    accept(graph, 9, 12, [row(9, masses=(.8, .1, .1))])
    future = graph.rows
    graph.restore(snap)
    assert graph.partial.samples == 1 and not graph.rows
    accept(graph, 9, 12, [row(9)])
    assert graph.rows != future
    assert graph.rows[0].weights == pytest.approx((.3, .2))
    snap.close()


def test_preserved_m_prefix_replay_and_intermediate_p_snapshot():
    graph = BlockGraph('a', 4)
    accept(graph, 0, 9, [row(8)])
    p = graph.snapshot()
    accept(graph, 9, 12, [row(9)])
    m = graph.snapshot()
    graph.restore(m, preserve_future=True)
    graph.set_preserved_prefix(p, until=12)
    accept(graph, 0, 9, [row(8)])
    captured_p = graph.snapshot(position=9)
    assert captured_p.partial.samples == 1 and captured_p.rows == ()
    assert graph.rows[0].samples == 2
    with pytest.raises(BufferError):
        graph.snapshot(position=10)
    accept(graph, 9, 12, [row(9)])
    assert graph.rows[0].samples == 2
    accept(graph, 12, 13)
    with pytest.raises(BufferError):
        graph.snapshot(position=9)
    for snapshot in (p, m, captured_p):
        snapshot.close()
    graph.close()
    assert graph.bytes == graph.budget.used == 0


def test_restore_and_invalidate_monotonic_epoch_cannot_aba():
    graph = BlockGraph('a', 4)
    snapshot = graph.snapshot()
    binding = graph.binding
    graph.invalidate()
    graph.restore(snapshot)
    with pytest.raises(RuntimeError):
        graph.validate_binding(binding)
    epoch = graph.runtime_epoch
    graph.restore(snapshot)
    assert graph.runtime_epoch > epoch
    with pytest.raises(RuntimeError):
        graph.validate_binding(binding)
    snapshot.close()


def test_snapshot_accounting_clone_close_and_owner_isolation():
    pool = GraphBudget(1_000_000)
    one = BlockGraph('same', 4, budget=pool)
    two = BlockGraph('same', 4, budget=pool)
    accept(one, 0, 12, [row(8)])
    snap = one.snapshot()
    clone = snap.clone()
    assert one.bytes == one.live_bytes + snap.bytes + clone.bytes
    assert pool.used == one.bytes + two.bytes
    snap.close()
    assert clone.rows and not two.rows
    clone.close()
    one.close(); two.close()
    assert pool.used == 0


def test_global_and_staging_capacity_failure_no_partial_commit():
    pool = GraphBudget(12000)
    graph = BlockGraph('a', 4, budget=pool, max_bytes_per_session=12000)
    initial = graph.binding, graph.bytes
    with pytest.raises(BufferError):
        graph.prepare(0, 1000, [row(999, tuple(range(0, 1000, 4)), (1 / 250,) * 250)])
    assert (graph.binding, graph.bytes) == initial
    transaction = graph.prepare(0, 12, [row(8)])
    staged = graph.bytes
    assert staged > initial[1] and graph.accepted_through == 0
    transaction.abort()
    assert (graph.binding, graph.bytes) == initial
    graph.close()
    assert pool.used == 0


def test_snapshot_capacity_rejects_without_leaking_live_state():
    pool = GraphBudget(12000)
    graph = BlockGraph('a', 4, budget=pool)
    snapshots = []
    with pytest.raises(BufferError):
        for _ in range(100):
            snapshots.append(graph.snapshot())
    assert graph.accepted_through == 0
    for snapshot in snapshots:
        snapshot.close()
    assert pool.used == graph.live_bytes
    graph.close()
    assert pool.used == 0


def test_message_spans_inherit_verified_prefix_and_reject_conflict():
    graph = BlockGraph('a', 4)
    graph.update_message_spans(((0, 8, 'user'), (8, 12, 'assistant')))
    accept(graph, 0, 12)
    graph.update_message_spans(((12, 20, 'tool'),), validated_prefix=12)
    assert graph.message_spans == ((0, 8, 'user'), (8, 12, 'assistant'), (12, 20, 'tool'))
    before = graph.binding, graph.bytes
    with pytest.raises(ValueError, match='Conflicting'):
        graph.update_message_spans(((0, 10, 'user'),), validated_prefix=12)
    assert (graph.binding, graph.bytes) == before
    snapshot = graph.snapshot()
    graph.update_message_spans((), validated_prefix=12)
    assert graph.message_spans == ((0, 8, 'user'), (8, 12, 'assistant'))
    graph.restore(snapshot)
    assert graph.message_spans[-1] == (12, 20, 'tool')
    snapshot.close()


def test_foreign_snapshot_rejected():
    first, second = BlockGraph('a', 4), BlockGraph('b', 4)
    snapshot = first.snapshot()
    with pytest.raises(ValueError):
        second.restore(snapshot)
    snapshot.close()


def test_numpy_row_views_convenience_does_not_require_python_float_lists():
    import numpy as np
    graph = BlockGraph('a', 4)
    with graph.prepare_accept(0, 12, [8, 9], [0, 4, 8],
                              np.array([[.2, .3, .5], [.2, .3, .5]]), 'prefill') as update:
        update.commit()
    assert graph.rows[0].prefill_samples == 2


def test_unknown_message_end_does_not_invent_boundary_or_accept_conflicting_split():
    graph = BlockGraph('a', 4)
    graph.update_message_spans(((0, 100, 'user'),))
    with pytest.raises(ValueError, match='Conflicting'):
        graph.update_message_spans(((0, 40, 'user'), (40, 100, 'assistant')), validated_prefix=50)
    graph.update_message_spans((), validated_prefix=50)
    assert graph.message_spans == ((0, 50, 'user'),)
    assert graph.message_end_known == (False,)
    from vllm_kvmem.graph_scoring import locality_boundary
    assert locality_boundary(48, 52, 4, graph.message_spans, .25, graph.message_end_known) == (1., True)
    assert locality_boundary(40, 44, 4, graph.message_spans, .25, graph.message_end_known) == (1., False)
    graph.update_message_spans(((0, 100, 'user'),), validated_prefix=50)
    assert graph.message_spans == ((0, 100, 'user'),) and graph.message_end_known == (True,)


def test_snapshot_close_releases_payload_reference_and_same_owner_reuse_is_foreign():
    first, second = BlockGraph('same', 4), BlockGraph('same', 4)
    snap = first.snapshot()
    with pytest.raises(ValueError, match='foreign'):
        second.restore(snap)
    snap.close()
    assert snap._data is None
    with pytest.raises(RuntimeError):
        snap.clone()


def test_preserved_prefix_restore_budget_failure_is_atomic():
    pool = GraphBudget(100_000)
    graph = BlockGraph('s', 4, budget=pool)
    accept(graph, 0, 9, [row(8)])
    p = graph.snapshot()
    accept(graph, 9, 12, [row(9)])
    m = graph.snapshot()
    before = graph.binding, graph.bytes, graph.rows
    pool.maximum = pool.used + p.bytes + graph.live_bytes - 1
    with pytest.raises(BufferError):
        graph.restore(m, preserve_future=True, preserved_prefix=p, until=12)
    assert (graph.binding, graph.bytes, graph.rows) == before
    pool.maximum = 100_000
    graph.restore(m, preserve_future=True, preserved_prefix=p, until=12)
    saved = graph.snapshot(position=9)
    assert saved.partial.samples == 1
    saved.close(); p.close(); m.close(); graph.close()
    assert pool.used == 0


def test_unsampled_progress_reuses_completed_row_payload():
    graph = BlockGraph('s', 4)
    accept(graph, 0, 12, [row(8)])
    rows = graph.rows
    accept(graph, 12, 1000)
    assert graph.rows is rows


def test_future_positive_mass_rejected_but_causal_mask_zeros_allowed():
    graph = BlockGraph('s', 4)
    with pytest.raises(ValueError, match='future'):
        graph.prepare(0, 4, [row(1, (0, 4), (.5, .5))])
    accept(graph, 0, 4, [row(1, (0, 4), (1., 0.))])
    assert graph.rows[0].samples == 1 and not graph.rows[0].targets


def test_message_metadata_staging_admits_large_valid_span_list():
    graph = BlockGraph('s', 4)
    spans = tuple((i * 4, i * 4 + 4, 'tool') for i in range(2000))
    graph.update_message_spans(spans)
    assert graph.message_spans == spans
    assert graph.bytes < 33554432


def test_staging_remains_charged_until_publication():
    graph = BlockGraph('s', 4)
    live = graph.live_bytes
    update = graph.prepare(0, 12, [row(8)])
    assert graph.bytes == live + update.charge.amount
    assert update.charge.amount > update.final_bytes
    update.commit()
    assert graph.bytes == graph.live_bytes == update.final_bytes
    assert update.data is None and update.charge is None


@pytest.mark.parametrize('known', [False, True])
def test_unchanged_metadata_with_retained_prefix_does_not_reserve_graph_copies(known):
    graph = BlockGraph('s', 4)
    accept(graph, 0, 400, [row(p) for p in range(8, 400, 4)])
    spans = ((0, 400, 'tool'),) if known else ()
    graph.update_message_spans(spans, validated_prefix=400)
    prefix = graph.snapshot()
    before = graph.binding, graph.bytes, graph.rows, prefix.binding
    # Enough for metadata validation, deliberately not two more graph copies.
    graph.max_bytes_per_session = graph.bytes + 4096
    graph.budget.maximum = graph.budget.used + 4096
    assert graph.live_bytes * 2 > 4096
    for _ in range(3):
        graph.update_message_spans(spans, validated_prefix=400)
        assert (graph.binding, graph.bytes, graph.rows, prefix.binding) == before
    prefix.close()
    graph.close()
    assert graph.budget.used == 0


def test_metadata_noop_still_validates_conflicts_prefix_and_pending_transaction():
    graph = BlockGraph('s', 4)
    graph.update_message_spans(((0, 16, 'tool'),))
    before = graph.binding, graph.bytes
    with pytest.raises(ValueError, match='validated message prefix'):
        graph.update_message_spans((), validated_prefix=-1)
    with pytest.raises(ValueError, match='Conflicting'):
        graph.update_message_spans(((0, 8, 'user'),), validated_prefix=16)
    assert (graph.binding, graph.bytes) == before
    update = graph.prepare(0, 0)
    update.detach()
    with pytest.raises(BlockingIOError, match='reserved acceptance'):
        graph.update_message_spans(((0, 16, 'tool'),), validated_prefix=16)
    update.close()
    assert (graph.binding, graph.bytes) == before
    graph.close()


def test_long_session_metadata_and_acceptance_fit_with_two_retained_snapshots():
    """Regression: ~420K history previously reserved a nonexistent graph copy."""
    graph = BlockGraph('long-agent', 32)
    end = 430080
    for begin in range(0, end, 2048):
        stop = min(end, begin + 2048)
        accept(graph, begin, stop, [row(p, (0, 32, 64, 96), (.25,) * 4)
            for p in range(max(128, begin), stop, 32)])
    snapshots = [graph.snapshot(), graph.snapshot()]
    before = graph.binding, graph.bytes, graph.rows
    spans = tuple((p, p + 560, 'tool') for p in range(0, end, 560))
    spans = tuple((b, min(e, end), r) for b, e, r in spans)
    update = graph.prepare(end, end, message_spans=spans, verified_prefix=end)
    # Prove old accounting would reject this same candidate at unchanged caps.
    assert graph.bytes <= graph.max_bytes_per_session
    assert graph.bytes + graph.live_bytes > graph.max_bytes_per_session
    update.abort()
    assert (graph.binding, graph.bytes, graph.rows) == before
    graph.update_message_spans(spans, validated_prefix=end)
    assert graph.rows is before[2]
    assert all(s.rows is graph.rows and s.message_spans == () for s in snapshots)
    accept(graph, end, end + 32, [row(end, (0, 32, 64, 96), (.25,) * 4)])
    assert len(graph.rows) == len(before[2]) + 1
    for snapshot in snapshots:
        snapshot.close()
    graph.close()
    assert graph.budget.used == 0
