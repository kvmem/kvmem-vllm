"""Age-adjusted per-layer prefill pressure (strategy_pressure_scorer=lift) and the shared
shallow Query selection (KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS); docs/age-adjusted-pressure.md."""
from copy import copy
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem import _prefill_state_native as native
from vllm_kvmem.serving import capabilities, layerwise_checkpoints
from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory, accept_buffered_batch
from vllm_kvmem.serving.deferred_fold import DeferredFoldHistory, accept_deferred, force_all
from vllm_kvmem.serving.layerwise_query import query_global_layers
from vllm_kvmem.serving.layerwise_pressure import LIFT, LayerwisePrefillPressure

State = native.State
BUCKETS = 14


@pytest.fixture(autouse=True)
def lift_native():
    if not hasattr(State, 'lift_empty') or not hasattr(State, 'with_values'):
        pytest.skip('Requires the rebuilt prefill-state extension (lift states)')


def lift_bucket(distance):
    """[0,32), [32,64), [64,128), ..., [65536,131072), >=131072 tokens."""
    return 0 if distance < 32 else min(BUCKETS - 1, int(distance).bit_length() - 5)


class Reference:
    """The design's definition, in plain Python and in a different summation order."""

    def __init__(self):
        self.mass, self.exposure, self.expected = {}, {}, {}
        self.bucket_mass, self.bucket_pairs = [0.] * BUCKETS, [0] * BUCKETS

    def commit(self, decay, rows, starts, history, local_begin, masses, shift):
        self.mass = {k: decay * self.mass.get(k, 0.) for k in history}
        self.exposure = {k: decay * self.exposure.get(k, 0.) for k in history}
        self.expected = {k: decay * self.expected.get(k, 0.) for k in history}
        m = np.asarray(masses, dtype=np.float64)
        inv, pages = 1. / len(rows), min(len(starts), len(history))
        visible = []
        for r, row in enumerate(rows):
            position, at = local_begin + row, local_begin + row + shift - 32
            seen = [(p, lift_bucket(at - history[p])) for p in range(pages) if starts[p] <= position]
            for p, b in seen:
                self.bucket_mass[b] += float(m[r, p])
                self.bucket_pairs[b] += 1
            visible.append(seen)
        mean = [bm / bp if bp else 0. for bm, bp in zip(self.bucket_mass, self.bucket_pairs)]
        for r, seen in enumerate(visible):
            norm = sum(mean[b] for _, b in seen)
            exposed = set()
            for p, b in seen:
                key = history[p]
                self.mass[key] += inv * float(m[r, p])
                if key not in exposed:
                    self.exposure[key] += inv
                    exposed.add(key)
                self.expected[key] += inv * (mean[b] / norm if norm > 0 else 1. / max(len(seen), 1))

    def scores(self):
        top = max([self.mass[k] / e for k, e in self.expected.items() if e > 0], default=0.) + 1.
        return {k: self.mass[k] / e if e > 0 else top for k, e in self.expected.items()}

    def drop(self, keep):
        for table in (self.mass, self.exposure, self.expected):
            for key in [k for k in table if not keep(k)]:
                del table[key]


def packets(seed=3, steps=5, count=1024, dtype=np.float64):
    """Chunks read into a window that keeps a random subset of older blocks.

    Yields (end, count, rows, starts, history, local_begin, masses, shift): history holds the
    window's history positions (32-token blocks), starts their window positions.
    """
    rng = np.random.default_rng(seed)
    kept = []
    for step in range(1, steps + 1):
        end = 4096 + step * count
        older = sorted(k for k in kept if rng.random() < .7)
        new = list(range(end - count, end, 32))
        history = older + new
        starts = list(range(0, 32 * len(history), 32))
        local_begin = 32 * len(older)
        rows = list(range(31, count, 64))
        masses = np.zeros((len(rows), len(history)), dtype=dtype)
        for r, row in enumerate(rows):
            visible = [p for p in range(len(history)) if starts[p] <= local_begin + row]
            weights = rng.random(len(visible)) ** 3
            masses[r, visible] = weights / weights.sum()
        yield end, count, rows, starts, history, local_begin, masses, end - count - local_begin
        kept = history


def bits(state):
    return sorted((key, struct.pack('<dd', *pair)) for key, pair in state.values().items())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_lift_state_matches_the_definition_and_keeps_mass_and_exposure(dtype):
    lift, plain, reference, last = State.lift_empty(), State.from_values({}), Reference(), 0
    for end, count, rows, starts, history, local_begin, masses, shift in packets(dtype=dtype):
        decay = 2. ** (-(end - last) / 32768.)
        lift = lift.committed(decay, rows, starts, history, local_begin, masses, shift)
        plain = plain.committed(decay, rows, starts, history, local_begin, masses)
        reference.commit(decay, rows, starts, history, local_begin, masses.astype(np.float64), shift)
        last = end
        assert lift.lift and not plain.lift
        # The extra profile never touches the accumulated attention: every double is equal.
        assert bits(lift) == bits(plain)
        scores = lift.scores()
        expected = reference.scores()
        assert scores.keys() == expected.keys()
        assert np.allclose([scores[k] for k in expected], [expected[k] for k in expected], rtol=1e-11, atol=0)
        assert lift.scores_for(list(history)) == [scores[k] for k in history]
    # Lift scores rank by distance-normalised attention, not by time visible.
    assert scores != plain.scores()


def test_unscored_keys_rank_first():
    # A packet whose rows see only the first two blocks: the third has no expected attention yet.
    state = State.lift_empty().committed(1., [31, 63], [0, 32, 1000], [0, 32, 64], 0,
                                         np.array([[.6, .4, 0.], [.3, .7, 0.]]), 0)
    scores = state.scores()
    assert scores[64] == max(scores[0], scores[32]) + 1.
    assert state.scores_for([64, 99]) == [scores[64]] * 2


def test_lift_batch_and_deferred_folds_equal_scalar_commits():
    layers = 16
    opts = [NS(native_prefill_commit=True, prefill_attention_half_life=32768. + i, pressure_policy='attention',
               guided_query_tokens=0, query_history=False, prefill_attention=True) for i in range(layers)]

    def lifted(cls):
        histories = [cls(o) for o in opts]
        for h in histories:
            h._buffered_state = State.lift_empty()
        return histories

    scalar, batch, deferred = (lifted(BufferedPrefillHistory), lifted(BufferedPrefillHistory),
                               lifted(DeferredFoldHistory))
    rng = np.random.default_rng(11)
    for end, count, rows, starts, history, local_begin, masses, _ in packets(seed=7, steps=4):
        logical = [list(history) for _ in range(layers)]
        stack = np.stack([masses[:, rng.permutation(masses.shape[1])] for _ in range(layers)])
        stack = stack / stack.sum(axis=2, keepdims=True)
        for layer, h in enumerate(scalar):
            assert h.accept(end, count, count, rows, starts, logical[layer], local_begin, stack[layer])
        staged = [copy(h) for h in batch]
        assert accept_buffered_batch(staged, end, count, count, rows, starts, logical, local_begin, stack) == \
            [True] * layers
        batch = staged
        staged = [copy(h) for h in deferred]
        assert accept_deferred(staged, end, count, count, rows, starts, logical, local_begin, stack) == \
            [True] * layers
        deferred = staged
    force_all(deferred)
    for a, b, c in zip(scalar, batch, deferred):
        assert a._buffered_state.lift and b._buffered_state.lift and c._buffered_state.lift
        assert bits(a._buffered_state) == bits(b._buffered_state) == bits(c._buffered_state)
        assert a.scores() == b.scores() == c.scores()


def test_gc_prune_keeps_the_lift_state_and_refuses_new_keys():
    h = BufferedPrefillHistory(NS(native_prefill_commit=True, prefill_attention_half_life=32768.))
    h.native_score_export = True
    h._buffered_state = State.lift_empty()
    reference, last = Reference(), 0
    stream = list(packets(seed=9, steps=4))
    for step, (end, count, rows, starts, history, local_begin, masses, shift) in enumerate(stream):
        if step == 2:
            # Historical GC prune (layerwise_pressure.prune_layer_histories): drop every third key.
            keep = lambda key: key // 32 % 3 != 0
            values = h.values
            h.values = {k: v for k, v in values.items() if keep(k)}
            reference.drop(keep)
            assert h._buffered_state.lift
            assert np.allclose(sorted(h.scores().values()), sorted(reference.scores().values()), rtol=1e-11)
        decay = 2. ** (-(end - last) / 32768.)
        assert h.accept(end, count, count, rows, starts, history, local_begin, masses)
        reference.commit(decay, rows, starts, history, local_begin, masses, shift)
        last = end
    scores, expected = h.scores(), reference.scores()
    assert np.allclose([scores[k] for k in expected], [expected[k] for k in expected], rtol=1e-11)
    values = h.values
    key, pair = next(iter(values.items()))
    with pytest.raises(ValueError, match='can only drop keys'):
        h.values = {**values, 10 ** 9: pair}
    with pytest.raises(ValueError, match='can only drop keys'):
        h.values = {key: [pair[0] * 2, pair[1]]}
    # An ordinary state still takes any values.
    plain = BufferedPrefillHistory(NS(native_prefill_commit=True, prefill_attention_half_life=8192.))
    plain.values = {10 ** 9: [1., 1.]}
    assert not plain._buffered_state.lift


def lift_controller(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '1')
    query = NS(native_batches=True, optimized=True, names=tuple(range(4)),
               runner=NS(model_config=NS(enforce_eager=True)))
    return LayerwisePrefillPressure(query)


def lift_options(scorer=LIFT):
    return NS(native_prefill_commit=True, prefill_attention_half_life=32768., pressure_policy='attention',
              guided_query_tokens=0, query_history=False, prefill_attention=True, strategy_pressure_scorer=scorer)


def test_layer_histories_start_as_lift_states_only_for_the_lift_scorer(monkeypatch):
    c = lift_controller(monkeypatch)
    assert all(h._buffered_state.lift for h in c._history('owner', lift_options()))
    assert not any(h._buffered_state.lift for h in c._history('other', lift_options('attention')))
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '0')
    with pytest.raises(ValueError, match='native score export'):
        c.new_history(lift_options())
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY', '0')
    with pytest.raises(ValueError, match='native buffered histories'):
        LayerwisePrefillPressure(NS(native_batches=True, optimized=True, names=(0,),
                                    runner=NS(model_config=NS(enforce_eager=True)))).new_history(lift_options())


def test_session_checkpoints_keep_lift_states(monkeypatch):
    c = lift_controller(monkeypatch)
    h = c.new_history(lift_options())
    end = 0
    for end, count, rows, starts, history, local_begin, masses, _ in packets(seed=2, steps=3):
        assert h.accept(end, count, count, rows, starts, history, local_begin, masses)
    state = h._buffered_state
    rows = layerwise_checkpoints.StateRows(state.sorted())
    assert rows.state.lift and rows.state.scores() == state.scores()
    monkeypatch.setenv('KVMEM_CHECKPOINT_FAST_STATE', '1')
    restored = layerwise_checkpoints._restored_history(c, lift_options(), (rows, end, 3, 3), end)
    assert restored._buffered_state is rows.state and restored.scores() == h.scores()
    # The dictionary restore path cannot carry the expected attention: it refuses.
    monkeypatch.setenv('KVMEM_CHECKPOINT_FAST_STATE', '0')
    monkeypatch.setattr(layerwise_checkpoints, '_NATIVE', None, raising=False)
    plain_rows = tuple((k, m, e) for k, (m, e) in sorted(state.values().items()))
    with pytest.raises(ValueError, match='can only drop keys'):
        layerwise_checkpoints._restored_history(c, lift_options(), (plain_rows, end, 3, 3), end)


def test_global_layers_setting_and_launch_checks(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS', raising=False)
    assert query_global_layers() == 0
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS', '4')
    assert query_global_layers() == 4
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS', 'x')
    with pytest.raises(ValueError, match='non-negative integer'):
        query_global_layers()
    kvmem = dict(action_attention=True, pressure_policy='attention', prefill_attention=True,
                 strategy_pressure_scorer=LIFT)
    report = capabilities.check(kvmem, query_mode='sequential', pressure=True,
                                environment={'KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS': '4'})
    assert report['pressure_scorer'] == LIFT and report['query_global_layers'] == 4
    assert 'query_global_layers' not in capabilities.check(kvmem, query_mode='sequential', pressure=True)
    with pytest.raises(ValueError, match='sequential per-layer Query'):
        capabilities.check(kvmem, query_mode='parallel', pressure=True,
                           environment={'KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS': '4'})
    with pytest.raises(ValueError, match='sequential per-layer Query'):
        capabilities.check(kvmem, query_mode='sequential', pressure=True,
                           environment={'KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS': '4',
                                        'KVMEM_LAYERWISE_QUERY_NO_PROBE': '1'})
    with pytest.raises(ValueError, match='NATIVE_SCORE_EXPORT=1'):
        capabilities.check(kvmem, query_mode='sequential', pressure=True,
                           environment={'KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT': '0'})
    with pytest.raises(ValueError, match='does not run on the global path'):
        capabilities.check(kvmem, query_mode=None, pressure=False)
