"""KVMEM_NATIVE_DECODE_LAYER_BATCH: one native call for every layer's decode
identities and an accepted-mass update that reads the identity array directly.

Both must equal the per-layer paths they replace bit for bit: the same
identities, the same mass keys, insertion order and float values, and the same
errors (the batch call answers None and the per-layer code raises its own).
"""
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem.serving import layerwise_decode
from test_native_decode_segments import (_as_items, _controller, _mass_case, _python_identities,
                                         columns, outcome)

CAP = 16777216


def _batch():
    from vllm_kvmem import _native
    if not (hasattr(_native, 'decode_identities_from_maps') and
            hasattr(_native, 'prepare_decode_attention_map_targets')):
        pytest.skip('native extension lacks the decode layer batch')
    return _native


def _flags(monkeypatch, batch, layer_map='1', mass_map='1'):
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_LAYER_MAP', layer_map)
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_MASS_MAP', mass_map)
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_LAYER_BATCH', batch)


def _window(rng, c, origin, total, rid='r', mode='merge', ahead=4000):
    begin = origin + rng.randrange(0, ahead)
    local = total + (begin - origin)
    count = rng.randrange(1, 6)
    starts, counts, _ = columns(rng, c.records(rid, None, begin, local, 0), begin, local, count, mode)
    return begin, local, count, starts, counts


def _shapes(starts, counts):
    """The same score columns in every container the callers may pass."""
    wide = np.zeros((len(starts), 3), np.int64)
    wide[:, 0], wide[:, 2] = starts, counts
    return [(starts, counts),
            (np.asarray(starts, np.int64), np.asarray(counts, np.int64)),
            (wide[:, 0], wide[:, 2]),                                   # strided views
            (np.asarray(starts, np.int32), np.asarray(counts, np.int32)),
            (tuple(starts), np.asarray(counts, np.int64))]


def _check(fused, want):
    assert fused is not None
    history, array = fused
    assert type(history) is list and all(type(v) is int for v in history) and history == want[0]
    assert type(array) is np.ndarray and array.dtype == np.int64 and array.shape == (len(want), len(want[0]))
    assert array.tolist() == want and array.flags.c_contiguous and not array.flags.writeable


@pytest.mark.parametrize('seed', range(40))
def test_batch_identities_equal_per_layer(seed):
    _batch()
    rng = random.Random(seed)
    c, origin, total = _controller(rng, layers=(16, 4, 1)[seed % 3], blocks=rng.choice((40, 300)),
                                   partial=seed % 4 != 0)
    begin, local, count, starts, counts = _window(rng, c, origin, total, mode=('merge', 'split')[seed % 2])
    want = outcome(_python_identities, c, begin, local, count, starts, counts)
    per_layer = layerwise_decode._fused_identities(c, 'r', begin, local, count, starts, counts)
    for s, n in _shapes(starts, counts):
        fused = layerwise_decode._fused_identities(c, 'r', begin, local, count, s, n, batch=True)
        if want[0] == 'ok':
            assert per_layer == want[1]
            _check(fused, want[1])
        else:
            assert per_layer is None and fused is None


@pytest.mark.parametrize('case', ['ok', 'length', 'backwards', 'numpy_ints', 'gap_column', 'list_records',
                                  'empty_column', 'short_columns', 'extra_column', 'float_column',
                                  'cross_block', 'bool_count'])
def test_batch_falls_back_with_identical_errors(case, monkeypatch):
    _batch()
    rng = random.Random(3)
    c, origin, total = _controller(rng, layers=4, blocks=60)
    begin = origin + 100
    local, count = total + 100, 3
    starts, counts, _ = columns(rng, c.records('r', None, begin, local, 0), begin, local, count, 'merge')
    starts, counts = list(starts), list(counts)
    if case == 'length': counts[-1] += 1    # the native window is one token longer than the frame
    elif case == 'backwards': begin = origin - 1
    elif case == 'numpy_ints':
        o, layers = c.maps['r']; r = layers[1][0]
        c.maps['r'] = (o, layers[:1] + ((r[:2] + (np.int64(r[2]),) + r[3:],) + layers[1][1:],) + layers[2:])
    elif case == 'gap_column': starts[2] += 1
    elif case == 'list_records':
        o, layers = c.maps['r']; c.maps['r'] = (o, tuple(list(l) for l in layers))
    elif case == 'empty_column': starts.insert(1, starts[1]); counts.insert(1, 0)
    elif case == 'short_columns': starts.pop(); counts.pop()
    elif case == 'extra_column': starts.append(starts[-1] + counts[-1]); counts.append(1)
    elif case == 'float_column': counts[0] = float(counts[0])
    elif case == 'cross_block':            # one score column spanning two logical blocks
        starts[1:3] = [starts[1]]; counts[1:3] = [counts[1] + counts[2]]
    elif case == 'bool_count': count = True
    pend = NS(history_begin=begin, candidate_count=count)
    masses = np.zeros((4, 1, 1), np.float32)
    results = {}
    for flag in ('0', '1'):
        _flags(monkeypatch, flag)
        c.decode_histories = {}
        c.__dict__.pop('_decode_layer_maps', None)
        got = outcome(layerwise_decode.attach_packet, c, pend, 'r', None, starts,
                      counts if case == 'float_column' else np.asarray(counts), masses)
        if got[0] == 'ok':
            logical = pend.layerwise_decode[2]
            # Inputs outside the batch call's domain run the per-layer or Python path.
            batched = flag == '1' and case in ('ok', 'list_records')
            assert type(logical) is (np.ndarray if batched else list)
            assert type(pend.history) is list and all(type(v) is int for v in pend.history)
            got = 'ok', np.asarray(logical).tolist(), pend.history
        results[flag] = got
    assert results['0'] == results['1']
    assert (results['0'][0] == 'ok') == (case in ('ok', 'numpy_ints', 'list_records', 'float_column'))


def test_batch_native_rejects_inputs_outside_its_domain():
    native = _batch()
    rng = random.Random(8)
    c, origin, total = _controller(rng, layers=3, blocks=50)
    begin, local, count, starts, counts = _window(rng, c, origin, total)
    parsed = [native.parse_layer_map(records) for records in c.maps['r'][1]]
    args = origin, begin, local, count, starts, counts
    one = native.decode_identities_from_map(parsed[0], *args)
    ok = native.decode_identities_from_maps(parsed, *args)
    assert ok is not None and ok[0] == one and ok[1][0].tolist() == one
    assert native.decode_identities_from_maps(tuple(parsed), *args)[1].tolist() == ok[1].tolist()
    bad_maps = [[], iter(parsed), parsed[0], None, [parsed[0], None], [parsed[0].astype(np.int32)],
                [parsed[0].ravel()], [np.asfortranarray(parsed[0])], [parsed[0][::2]],
                [parsed[0][:, :2]], [parsed[0].tolist()]]
    for maps in bad_maps:
        assert native.decode_identities_from_maps(maps, *args) is None
    for position in range(4):                       # scalars outside the exact int domain
        for value in (np.int64(args[position]), float(args[position]), 1 << 70, None, 'x'):
            changed = list(args); changed[position] = value
            assert native.decode_identities_from_maps(parsed, *changed) is None
            assert native.decode_identities_from_map(parsed[0], *changed) is None
    assert ok[1].tolist() == native.decode_identities_from_maps(parsed, *args)[1].tolist()
    with pytest.raises(ValueError):
        ok[1][0, 0] = 1                             # shared by every layer's staged history


def test_batch_cache_follows_frames_layers_and_requests(monkeypatch):
    _batch()
    monkeypatch.setattr(layerwise_decode, '_LAYER_MAP_CACHE_LIMIT', 6)
    rng = random.Random(21)
    c, origin, total = _controller(rng, layers=3, blocks=50)
    frames = {'r': (origin, total)}
    for k in range(5):                              # several requests share one controller
        other, o, t = _controller(rng, layers=3, blocks=50)
        c.maps[f'r{k}'] = other.maps['r']
        frames[f'r{k}'] = o, t

    def verify(rid):
        o, t = frames[rid]
        begin, local, count, starts, counts = _window(rng, c, o, t, rid=rid, ahead=300)
        want = [layerwise_decode._segment_identities_python(c.records(rid, None, begin, local, layer),
                                                            begin, local, count, starts, counts)
                for layer in range(3)]
        if rng.random() < .5:                       # per-layer entries live in the same cache
            assert layerwise_decode._fused_identities(c, rid, begin, local, count, starts, counts) == want
        _check(layerwise_decode._fused_identities(c, rid, begin, local, count, starts, counts, batch=True), want)

    for _ in range(4):
        for rid in rng.sample(sorted(frames), len(frames)):
            verify(rid)
        # A refresh installs a whole new frame for one request ...
        rid = rng.choice(sorted(frames))
        other, o, t = _controller(rng, layers=3, blocks=50)
        c.maps[rid] = other.maps['r']; frames[rid] = o, t
        verify(rid)
        # ... and a lazy layer commit replaces a single layer's map object.
        rid = rng.choice(sorted(frames))
        o, layers = c.maps[rid]
        moved = tuple((s + 1, g, b, n, p) for s, g, b, n, p in layers[1])
        blocks = [r[2] for r in layers[1]]
        blocks[0] = blocks[0] + 32 if blocks[0] + 32 < blocks[1] else blocks[0] - 32
        moved = tuple((s, g, b, n, p) for (s, g, _, n, p), b in zip(moved, blocks))
        c.maps[rid] = (o, (layers[0], moved, layers[2]))
        verify(rid)
        # The same tuple wrapped in a new frame object must not be parsed again.
        cache = c._decode_layer_maps
        before = cache.get((rid, None))
        c.maps[rid] = (o, tuple(c.maps[rid][1]))
        verify(rid)
        if before is not None and (rid, None) in cache:
            assert cache[(rid, None)][1][0] is before[1][0] or len(cache) <= 6


def _target_views(targets):
    """One identity row as the forms an int64 array row can take."""
    base = np.asarray(targets, np.int64)
    wide = np.zeros((2, len(targets) * 2), np.int64)
    wide[1, ::2] = base
    stack = np.stack([base + 7, base, base - 3])
    return [base, wide[1, ::2], stack[1]]


def _hex(mass):
    return [(k, float(v).hex()) for k, v in mass.to_dict().items()]


@pytest.mark.parametrize('seed', range(40))
def test_mass_targets_chain_equals_list_chain(seed):
    native = _batch()
    rng = random.Random(seed)
    old, _, _, _ = _mass_case(rng, np.float32)
    kind = ('random', 'sorted', 'increasing', 'few')[seed % 4]
    a_mass, a_n = dict(old), 0
    b_mass, b_n = dict(old), 0
    for step in range(6):
        cols = rng.choice((1, 7, 600))
        _, indices, targets, m = _mass_case(rng, np.float32 if (seed + step) % 2 else np.float64, cols=cols,
                                            zero_frac=rng.choice((0., .3, .9)))
        if kind == 'sorted': targets = sorted(targets)                      # non-decreasing, duplicates
        elif kind == 'increasing': targets = sorted(rng.sample(range(-64, 1 << 40, 32), cols))
        elif kind == 'few': targets = [rng.choice((0, 32, 64)) for _ in range(cols)]
        if step == 3:
            targets = [t - 2048 for t in targets]                          # negative and unseen keys
        if step % 2:
            wide = np.zeros((m.shape[0], cols * 2), m.dtype); wide[:, ::2] = m; m = wide[:, ::2]
        view = _target_views(targets)[(seed + step) % 3]
        keep = view.copy()
        a_mass, a_n = native.prepare_decode_attention_map(a_mass, a_n, indices, targets, m, CAP)
        b_mass, b_n = native.prepare_decode_attention_map_targets(b_mass, b_n, indices, view, m, CAP)
        assert type(b_mass) is native.DecodeMassMap and a_n == b_n and len(a_mass) == len(b_mass)
        assert _hex(a_mass) == _hex(b_mass) and _as_items(a_mass) == _as_items(b_mass)
        assert (view == keep).all()
        if step == 2:                       # GC pruning / a restore hands back a plain dict
            a_mass = {k: v for k, v in a_mass.to_dict().items() if k % 64}
            b_mass = dict(a_mass)


@pytest.mark.parametrize('case', ['sum', 'negative', 'nan', 'inf', 'order', 'duplicate', 'row_range', 'shape',
                                  'dtype', 'capacity', 'source'])
@pytest.mark.parametrize('ordered', [False, True])
def test_mass_targets_failures_equal_list_entry(case, ordered):
    native = _batch()
    rng = random.Random(13)
    old, _, targets, m = _mass_case(rng, np.float32)
    if ordered: targets = sorted(targets)
    indices, cap = [0, 2, 4], CAP
    if case == 'sum': m[2] *= 1.5
    elif case == 'negative': m[0, 3] = -1e-3
    elif case == 'nan': m[4, 7] = np.nan
    elif case == 'inf': m[2, 1] = np.inf
    elif case == 'order': indices = [2, 0]
    elif case == 'duplicate': indices = [1, 1]
    elif case == 'row_range': indices = [0, 9]
    elif case == 'shape': targets = targets[:-1]
    elif case == 'dtype': m = m.astype(np.float16)
    elif case == 'capacity': cap = 4096
    base = native.DecodeMassMap.from_dict(old)
    for source in (dict(old), base) if case != 'source' else ([1, 2],):
        a = outcome(native.prepare_decode_attention_map, source, 0, indices, targets, m, cap)
        b = outcome(native.prepare_decode_attention_map_targets, source, 0, indices,
                    np.asarray(targets, np.int64), m, cap)
        assert a[0] != 'ok' and a == b
    assert base.to_dict() == old            # a failed prepare never mutates its input


def test_mass_targets_requires_an_int64_row():
    native = _batch()
    rng = random.Random(2)
    old, indices, targets, m = _mass_case(rng, np.float32)
    for bad in (targets, np.asarray(targets, np.int32), np.asarray(targets, np.float64),
                np.asarray([targets], np.int64), np.asarray(targets, np.uint64), None):
        with pytest.raises(TypeError):
            native.prepare_decode_attention_map_targets(dict(old), 0, [0], bad, m, CAP)


def _pend(begin, count, interval):
    return NS(candidate_count=count, history_begin=begin, rows=list(range(count)),
              state=NS(options=NS(action_interval=interval, action_commit_max_bytes=CAP)))


def test_live_window_shared_identities_keep_separate_layer_scores(monkeypatch):
    rng = random.Random(12)
    layers = 4
    controller, begin, local = _controller(rng, layers=layers, blocks=40)
    records = controller.maps.pop('r')[1][0]
    reads = []

    def working_records():
        reads.append(1)
        return records

    authority = NS(memory=NS(_store=NS(working_records=working_records)))
    count = 5
    starts, counts, _ = columns(rng, records, begin, local, count, 'split')
    shape = (layers, count, len(starts))
    masses = np.random.default_rng(4).random(shape)
    masses = (masses / masses.sum(axis=2, keepdims=True)).astype(np.float32)

    def evaluate(batch):
        _flags(monkeypatch, batch)
        monkeypatch.setenv('KVMEM_NATIVE_DECODE_SEGMENTS', '1')
        monkeypatch.setenv('KVMEM_NATIVE_DECODE_PREPARE_BATCH', '1')
        controller.decode_histories['r'] = [
            layerwise_decode.NativeDecodeHistory(_batch().DecodeMassMap.from_dict({}), 0, 0, 0)
            for _ in range(layers)]
        reads.clear()
        pending = _pend(begin, count, 1)
        layerwise_decode.attach_packet(controller, pending, 'r', authority, starts, counts, masses)
        _, staged, _ = layerwise_decode.prepare_accept(controller, pending, 3)
        return pending, staged, len(reads)

    old, reference, old_reads = evaluate('0')
    new, staged, new_reads = evaluate('1')
    assert (old_reads, new_reads) == (layers, 1)
    assert old.history == new.history
    targets = new.layerwise_decode[2]
    np.testing.assert_array_equal(targets, old.layerwise_decode[2])
    assert not targets.flags.writeable and targets.strides[0] == 0
    assert [list(h.mass.items()) for h in staged] == [list(h.mass.items()) for h in reference]
    assert staged[0].mass != staged[1].mass


def _run_chain(seed, interval, steps=30, requests=4, layers=16):
    """Interleaved decode steps of several requests: attach, accept, commit."""
    from vllm_kvmem.serving.layerwise_decode import DecodeHistory
    rng = random.Random(seed)
    c, origin, total = _controller(rng, layers=layers, blocks=80)
    state = {'r': [origin, total, origin]}
    for k in range(requests - 1):
        other, o, t = _controller(rng, layers=layers, blocks=80)
        c.maps[f'r{k}'] = other.maps['r']
        state[f'r{k}'] = [o, t, o]
    trace = []
    for step in range(steps):
        for rid in sorted(state):
            o, t, begin = state[rid]
            local, count = t + (begin - o), rng.randrange(1, 6)
            starts, counts, _ = columns(rng, c.records(rid, None, begin, local, 0), begin, local, count, 'merge')
            cols = len(starts)
            masses = np.asarray([[[0. if rng.random() < .2 else rng.random() for _ in range(cols)]
                                  for _ in range(count)] for _ in range(layers)], np.float64)
            masses[:, :, rng.randrange(cols)] += 1e-3
            masses = (masses / masses.sum(2, keepdims=True)).astype(np.float32)
            pend = _pend(begin, count, interval)
            layerwise_decode.attach_packet(c, pend, rid, None, starts, np.asarray(counts), masses)
            accepted = rng.randrange(0, count + 1)
            _, staged, kind = layerwise_decode.prepare_accept(c, pend, accepted)
            assert kind == 'decode'
            c.decode_histories[rid] = staged
            state[rid][2] = begin + accepted
            trace.append((rid, list(pend.history), [type(v) for v in pend.history],
                          np.asarray(pend.layerwise_decode[2]).tolist()))
            if step == steps // 2 and rid == 'r':   # a restore rebuilds plain dict histories
                c.decode_histories[rid] = [DecodeHistory({k: v for k, v in h.mass.items() if k % 64},
                                                         h.samples, h.end, h.chunks) for h in staged]
    final = {rid: [([(k, v.hex()) for k, v in h.mass.items()], h.samples, h.end, h.chunks,
                    type(h).__name__) for h in c.decode_histories[rid]] for rid in state}
    return trace, final


@pytest.mark.parametrize('seed,interval', [(1, 16), (2, 2), (3, 1), (4, 3)])
def test_attach_and_accept_chains_equal_per_layer_path(seed, interval, monkeypatch):
    _batch()
    _flags(monkeypatch, '0')
    want = _run_chain(seed, interval)
    _flags(monkeypatch, '1')
    got = _run_chain(seed, interval)
    assert got[0] == want[0] and got[1] == want[1]
    assert any(samples for rid in want[1] for _, samples, _, _, _ in want[1][rid])
    _flags(monkeypatch, '0', mass_map='0')          # and both equal the dict path
    plain = _run_chain(seed, interval)
    assert plain[0] == want[0]
    assert {rid: [h[:4] for h in hs] for rid, hs in plain[1].items()} == \
        {rid: [h[:4] for h in hs] for rid, hs in want[1].items()}


def test_batch_flag_is_validated_and_needs_both_base_flags(monkeypatch):
    _batch()
    _flags(monkeypatch, 'yes')
    with pytest.raises(ValueError):
        layerwise_decode.layer_batch_enabled()
    _flags(monkeypatch, '1')
    assert layerwise_decode.layer_batch_enabled()
    for layer_map, mass_map in (('0', '1'), ('1', '0'), ('0', '0')):
        _flags(monkeypatch, '1', layer_map, mass_map)
        assert not layerwise_decode.layer_batch_enabled()
        rng = random.Random(4)
        c, origin, total = _controller(rng, layers=2, blocks=40)
        begin, local, count, starts, counts = _window(rng, c, origin, total)
        pend = NS(history_begin=begin, candidate_count=count)
        layerwise_decode.attach_packet(c, pend, 'r', None, starts, np.asarray(counts),
                                       np.zeros((2, 1, 1), np.float32))
        assert type(pend.layerwise_decode[2]) is list and pend.history is pend.layerwise_decode[2][0]
    _flags(monkeypatch, '0')
    assert not layerwise_decode.layer_batch_enabled()


@pytest.mark.parametrize('seed', range(300))
def test_batch_native_equals_per_layer_native_on_damaged_frames(seed):
    """The single-pass batch call and the per-layer call agree on None as well:
    random damage to the frame rows, the coordinates and the score columns."""
    native = _batch()
    rng = random.Random(1000 + seed)
    layers = rng.choice((1, 2, 5))
    c, origin, total = _controller(rng, layers=layers, blocks=rng.choice((2, 3, 12, 40)), partial=seed % 3 != 0)
    begin, local, count, starts, counts = _window(rng, c, origin, total, mode=('merge', 'split')[seed % 2],
                                                  ahead=rng.choice((1, 40, 700)))
    parsed = [native.parse_layer_map(records).copy() for records in c.maps['r'][1]]
    starts, counts = list(starts), list(counts)
    for _ in range(rng.randrange(0, 3)):
        kind = rng.randrange(12)
        rows = parsed[rng.randrange(layers)]
        if not len(rows) or not starts:
            break
        row = rng.randrange(len(rows))
        if kind == 0: rows[row, rng.randrange(3)] += rng.choice((-64, -32, -1, 1, 32, 64))
        elif kind == 1: rows[row, 1] = rng.choice((0, -1, 33, 1 << 62))
        elif kind == 2: rows[row, 2] = rng.choice((-1, local, local + 5, 1 << 62))
        elif kind == 3: parsed[rng.randrange(layers)] = rows[:rng.randrange(len(rows))].copy()
        elif kind == 4: parsed[rng.randrange(layers)] = np.zeros((0, 3), np.int64)
        elif kind == 5: begin += rng.choice((-40, -1, 1, 31, 32, 33))
        elif kind == 6: local += rng.choice((-33, -1, 1, 32))
        elif kind == 7: count += rng.choice((-count - 3, -count, -1, 1, 40))
        elif kind == 8: origin += rng.choice((-32, -1, 1, 32))
        elif kind == 9:
            k = rng.randrange(len(starts)); starts[k] += rng.choice((-1, 1)); counts[k] += rng.choice((-1, 0, 1))
        elif kind == 10:
            k = rng.randrange(len(starts)); del starts[k]; del counts[k]
        elif kind == 11 and len(starts) > 1:
            k = rng.randrange(len(starts) - 1)
            counts[k] += counts[k + 1]; del starts[k + 1]; del counts[k + 1]
    want = [native.decode_identities_from_map(rows, origin, begin, local, count, starts, counts)
            for rows in parsed]
    got = native.decode_identities_from_maps(parsed, origin, begin, local, count, starts, counts)
    if any(one is None for one in want):
        assert got is None
    else:
        assert got is not None and got[1].tolist() == want and got[0] == want[0]
    _DAMAGED[seed] = got is not None


_DAMAGED = {}


def test_damaged_frames_cover_both_outcomes():
    if len(_DAMAGED) < 300:
        pytest.skip('needs the whole damaged-frame sweep')
    valid = sum(_DAMAGED.values())
    assert 30 <= valid <= 270


# KVMEM_NATIVE_DECODE_PREPARE_BATCH and the threaded layer loops
# (KVMEM_NATIVE_PUBLISH_THREADS): every layer's result equals its per-layer
# call bit for bit, for any thread count, errors included.

def _prepare_batch_native():
    native = _batch()
    if not hasattr(native, 'prepare_decode_attention_maps_targets'):
        pytest.skip('native extension lacks the decode prepare batch')
    return native


def _layers_case(rng, layers, dtype, rows=5, cols=600):
    """Per-layer old maps, one shared row set, an int64 [layers, cols] identity
    array and one [layers, rows, cols] mass buffer, as attach_packet stages them."""
    olds, targets, masses = [], [], []
    _, indices, _, _ = _mass_case(rng, dtype, rows=rows, cols=cols)
    for _ in range(layers):
        old, _, t, m = _mass_case(rng, dtype, rows=rows, cols=cols, zero_frac=rng.choice((0., .3, .9)))
        if rng.random() < .3:
            t = sorted(t)                                       # the merge-walk branch
        olds.append(native_map(old))
        targets.append(t)
        masses.append(m)
    return olds, indices, np.asarray(targets, np.int64), np.stack(masses)


def native_map(old):
    from vllm_kvmem import _native
    return _native.DecodeMassMap.from_dict(old)


def _per_layer(native, olds, indices, targets, masses, cap=CAP):
    """The per-layer loop: the first failing layer raises, in layer order."""
    out = []
    for layer, old in enumerate(olds):
        mass, n = native.prepare_decode_attention_map_targets(old, 7, indices, targets[layer], masses[layer], cap)
        assert n == 7 + len(indices)
        out.append(mass)
    return out


@pytest.mark.parametrize('threads', ['1', '8'])
@pytest.mark.parametrize('seed', range(24))
def test_prepare_batch_equals_per_layer_calls(seed, threads, monkeypatch):
    native = _prepare_batch_native()
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', threads)
    rng = random.Random(500 + seed)
    layers = (1, 4, 16)[seed % 3]
    olds, indices, targets, masses = _layers_case(rng, layers, np.float32 if seed % 2 else np.float64,
                                                  cols=rng.choice((1, 40, 600)))
    if seed % 4 == 1:                                           # strided mass views
        wide = np.zeros(masses.shape[:2] + (masses.shape[2] * 2,), masses.dtype)
        wide[:, :, ::2] = masses
        masses = wide[:, :, ::2]
    if seed % 4 == 2:                                           # a read-only, strided identity array
        wide = np.zeros((targets.shape[0], targets.shape[1] * 2), np.int64)
        wide[:, ::2] = targets
        targets = wide[:, ::2]
        targets.flags.writeable = False
    before = [_hex(m) for m in olds]
    want = _per_layer(native, olds, indices, targets, masses)
    got = native.prepare_decode_attention_maps_targets(olds, indices, targets, masses, CAP)
    assert type(got) is list and len(got) == layers
    assert all(type(m) is native.DecodeMassMap for m in got)
    assert [_hex(m) for m in got] == [_hex(m) for m in want]
    assert [_hex(m) for m in olds] == before                    # inputs are never mutated


@pytest.mark.parametrize('threads', ['1', '8'])
@pytest.mark.parametrize('case', ['sum', 'negative', 'nan', 'order', 'duplicate', 'row_range', 'capacity',
                                  'two_layers', 'capacity_after_sum'])
def test_prepare_batch_raises_what_the_first_failing_layer_raises(case, threads, monkeypatch):
    native = _prepare_batch_native()
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', threads)
    rng = random.Random(77)
    olds, _, targets, masses = _layers_case(rng, 12, np.float32, cols=300)
    indices, cap = [0, 2, 4], CAP
    masses = masses.copy()
    if case == 'sum': masses[5, 2] *= 1.5
    elif case == 'negative': masses[3, 0, 3] = -1e-3
    elif case == 'nan': masses[11, 4, 7] = np.nan
    elif case == 'order': indices = [2, 0]
    elif case == 'duplicate': indices = [1, 1]
    elif case == 'row_range': indices = [0, 9]
    elif case == 'capacity': cap = 4096 + 320 * (300 + len(olds[6])) - 1         # only layers this big fail
    elif case == 'two_layers': masses[9, 0, 1] = -1.; masses[4, 2] *= 2.
    elif case == 'capacity_after_sum':
        masses[2, 4] *= 3.
        big = max(range(3, 12), key=lambda layer: len(olds[layer]))
        cap = 4096 + 320 * (300 + len(olds[big])) - 1
    want = outcome(_per_layer, native, olds, indices, targets, masses, cap)
    got = outcome(native.prepare_decode_attention_maps_targets, olds, indices, targets, masses, cap)
    assert want[0] != 'ok'
    assert got == want


def test_prepare_batch_declines_inputs_outside_its_domain():
    native = _prepare_batch_native()
    rng = random.Random(9)
    olds, indices, targets, masses = _layers_case(rng, 3, np.float32, cols=50)
    call = native.prepare_decode_attention_maps_targets
    assert call(olds, indices, targets, masses, CAP) is not None
    for bad in (dict(olds=[{1: 1.}] + olds[1:]), dict(olds=olds[:2]), dict(olds=[]),
                dict(targets=targets[0]), dict(targets=targets.astype(np.int32)), dict(targets=targets.tolist()),
                dict(targets=targets[:, :-1]), dict(masses=masses[0]), dict(masses=masses[:2]),
                dict(masses=masses.astype(np.float16)), dict(masses=masses[:, :, :-1])):
        args = {**dict(olds=olds, targets=targets, masses=masses), **bad}
        assert call(args['olds'], indices, args['targets'], args['masses'], CAP) is None


@pytest.mark.parametrize('seed,interval', [(1, 16), (2, 2), (3, 1), (4, 3)])
@pytest.mark.parametrize('threads', ['1', '8'])
def test_prepare_batch_chains_equal_per_layer_path(seed, interval, threads, monkeypatch):
    _prepare_batch_native()
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', '1')
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_PREPARE_BATCH', '0')
    _flags(monkeypatch, '0')
    want = _run_chain(seed, interval)
    calls = []
    original = layerwise_decode._native.prepare_decode_attention_maps_targets

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(layerwise_decode._native, 'prepare_decode_attention_maps_targets', counted)
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', threads)
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_PREPARE_BATCH', '1')
    _flags(monkeypatch, '1')
    got = _run_chain(seed, interval)
    assert got[0] == want[0] and got[1] == want[1]
    assert calls                                                # the batch call did run


def test_prepare_batch_flag_is_validated(monkeypatch):
    _prepare_batch_native()
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_PREPARE_BATCH', 'yes')
    with pytest.raises(ValueError):
        layerwise_decode.prepare_batch_enabled()
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_PREPARE_BATCH', '1')
    assert layerwise_decode.prepare_batch_enabled()
    monkeypatch.delenv('KVMEM_NATIVE_DECODE_PREPARE_BATCH')
    assert not layerwise_decode.prepare_batch_enabled()


@pytest.mark.parametrize('seed', range(120))
def test_threaded_batch_identities_equal_serial(seed, monkeypatch):
    """decode_identities_from_maps splits its layers across the publication pool;
    the array (and None) is the same for any thread count."""
    native = _batch()
    rng = random.Random(4000 + seed)
    layers = rng.choice((2, 5, 16))
    c, origin, total = _controller(rng, layers=layers, blocks=rng.choice((12, 40, 300)), partial=seed % 3 != 0)
    begin, local, count, starts, counts = _window(rng, c, origin, total, mode=('merge', 'split')[seed % 2],
                                                  ahead=rng.choice((1, 40, 700)))
    parsed = [native.parse_layer_map(records).copy() for records in c.maps['r'][1]]
    if seed % 4 == 3:                                           # one damaged layer: None either way
        rows = parsed[rng.randrange(layers)]
        if len(rows):
            rows[rng.randrange(len(rows)), 1] = 33
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', '1')
    serial = native.decode_identities_from_maps(parsed, origin, begin, local, count, starts, counts)
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', '8')
    for _ in range(3):
        threaded = native.decode_identities_from_maps(parsed, origin, begin, local, count, starts, counts)
        if serial is None:
            assert threaded is None
        else:
            assert threaded[0] == serial[0] and threaded[1].tolist() == serial[1].tolist()
            assert not threaded[1].flags.writeable
