"""Differential coverage for graph metadata without changing graph semantics."""
import random

import pytest

from vllm_kvmem.block_graph import _merge_spans, _spans, BlockGraph


def reference_merge(old, old_known, supplied, prefix):
    # Frozen pre-optimization implementation: independent oracle for conflict
    # handling, prefix cropping, unknown ends and replacement rules.
    incoming = _spans(supplied)
    if type(prefix) is not int or prefix < 0:
        raise ValueError('Invalid validated message prefix')
    retained = [(begin, min(end, prefix), role, known and end <= prefix)
                for (begin, end, role), known in zip(old, old_known) if begin < prefix]
    for begin, end, role, known in retained:
        for new_begin, new_end, new_role in incoming:
            if max(begin, new_begin) < min(end, new_end, prefix):
                if (begin != new_begin or role != new_role or
                        new_end < end or (known and new_end != end)):
                    raise ValueError('Conflicting message boundaries in validated historical prefix')
    result = {(begin, end, role): known for begin, end, role, known in retained}
    for span in incoming:
        begin, end, role = span
        for previous in tuple(result):
            if previous[0] == begin and previous[2] == role:
                del result[previous]
        result[span] = True
    ordered = tuple(sorted(result))
    _spans(ordered)
    return ordered, tuple(result[item] for item in ordered)


def outcome(function, *args):
    try:
        return ('ok', function(*args))
    except ValueError as error:
        return ('error', str(error))


@pytest.mark.parametrize('seed', range(12))
def test_differential_prefix_and_conflict_cases(seed):
    rng = random.Random(seed)
    def spans():
        result = []
        end = 0
        for _ in range(rng.randrange(30)):
            begin = end + rng.randrange(4)
            end = begin + rng.randrange(1, 12)
            result.append((begin, end, rng.choice(['user', 'assistant', 'tool'])))
        return tuple(result)
    for _ in range(500):
        old = spans()
        incoming = rng.choice([old, old[::2], spans(), ()])
        known = tuple(rng.choice([False, True]) for _ in old)
        prefix = rng.randrange(320)
        assert outcome(_merge_spans, old, known, incoming, prefix) == outcome(
            reference_merge, old, known, incoming, prefix)


def test_unknown_end_can_extend_but_not_split_or_change_role():
    old = ((0, 8, 'tool'), (12, 20, 'assistant'))
    assert _merge_spans(old, (True, False), ((12, 28, 'assistant'),), 20) == (
        ((0, 8, 'tool'), (12, 28, 'assistant')), (True, True))
    for incoming in [((12, 28, 'tool'),), ((12, 16, 'assistant'), (16, 28, 'assistant'))]:
        with pytest.raises(ValueError, match='Conflicting'):
            _merge_spans(old, (True, False), incoming, 20)


def test_conflict_keeps_graph_snapshot_binding_and_budget():
    graph = BlockGraph('owner', 32)
    graph.update_message_spans(((0, 100, 'tool'),))
    snapshot = graph.snapshot()
    before = (graph.binding, graph.message_spans, graph.bytes, graph.budget.used)
    with pytest.raises(ValueError, match='Conflicting'):
        graph.update_message_spans(((0, 50, 'tool'), (50, 100, 'assistant')), validated_prefix=100)
    assert (graph.binding, graph.message_spans, graph.bytes, graph.budget.used) == before
    assert snapshot.message_spans == before[1]
    snapshot.close()
    graph.close()


@pytest.mark.parametrize('incoming,prefix', [(((4, 3, 'tool'),), 8), (((4, 9, 'tool'), (7, 12, 'tool')), 8), ((), -1), ((), True)])
def test_invalid_input_rejected_identically(incoming, prefix):
    assert outcome(_merge_spans, (), (), incoming, prefix) == outcome(reference_merge, (), (), incoming, prefix)
