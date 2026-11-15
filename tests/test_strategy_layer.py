"""The strategy layer: contract, registry, the serving seams and the decision trace."""
import ast
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import strategy
from vllm_kvmem.strategy import (ContractError, PickView, PressureLayerView, QueryLayerView, Strategy,
                                 check_pick, check_pressure_scores, check_query_scores, trace)
from vllm_kvmem.strategy import registry as _registry
from test_layerwise_pressure import controller, native_authority, options

PACKAGE = Path(strategy.__file__).parent


# --- layering -----------------------------------------------------------------

def test_strategy_package_imports_only_itself_and_the_standard_library():
    allowed = {'argparse', 'dataclasses', 'hashlib', 'json', 'math', 'pathlib', 'sys', 'typing'}
    for path in sorted(PACKAGE.glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert {alias.name.split('.')[0] for alias in node.names} <= allowed, path.name
            elif isinstance(node, ast.ImportFrom):
                # level 1 is this package; anything above it is serving or core.
                assert node.level == 1 or (node.level == 0 and node.module.split('.')[0] in allowed), (
                    path.name, node.module, node.level)


# --- registry -----------------------------------------------------------------

def test_defaults_name_the_built_in_paths_and_resolve_to_none():
    assert strategy.DEFAULTS == {strategy.QUERY_SCORER: 'meank', strategy.PRESSURE_SCORER: 'attention',
                                 strategy.PICKER: 'native'}
    for role, name in strategy.DEFAULTS.items():
        assert strategy.get(role, name).native
        assert strategy.resolve(role, name) is None
    assert strategy.resolve(strategy.QUERY_SCORER, 'recency').name == 'recency'
    assert strategy.resolve(strategy.PRESSURE_SCORER, 'recency').name == 'recency'
    assert {'lru', 'redundancy'} <= set(strategy.names(strategy.EVICTOR))


def test_unknown_and_duplicate_strategies_are_refused():
    with pytest.raises(ValueError, match='Unknown picker'):
        strategy.get(strategy.PICKER, 'missing')
    with pytest.raises(ValueError, match='already registered'):
        strategy.register(Strategy(strategy.PICKER, 'native', lambda: None))
    with pytest.raises(ValueError, match='known role'):
        strategy.register(Strategy('other', 'x', lambda: None))


def test_runtime_options_validate_strategy_names():
    from vllm_kvmem.core.options import RuntimeOptions
    base = RuntimeOptions()
    assert (base.strategy_query_scorer, base.strategy_pressure_scorer, base.strategy_picker,
            base.strategy_trace_scores) == ('meank', 'attention', 'native', False)
    assert RuntimeOptions(strategy_pressure_scorer='recency').strategy_pressure_scorer == 'recency'
    for field in ('strategy_query_scorer', 'strategy_pressure_scorer', 'strategy_picker'):
        with pytest.raises(ValueError, match='Unknown'):
            RuntimeOptions(**{field: 'missing'})
    with pytest.raises(ValueError):
        RuntimeOptions(strategy_trace_scores=1)


def test_launcher_refuses_a_scorer_the_launched_path_would_ignore():
    from vllm_kvmem.serving.serve import validate_strategy_scope
    validate_strategy_scope({}, query_mode=None, pressure=False)
    validate_strategy_scope(dict(strategy_query_scorer='recency', strategy_pressure_scorer='recency'),
                            query_mode='sequential', pressure=True)
    with pytest.raises(ValueError, match='strategy_query_scorer'):
        validate_strategy_scope(dict(strategy_query_scorer='recency'), query_mode='global', pressure=True)
    with pytest.raises(ValueError, match='strategy_pressure_scorer'):
        validate_strategy_scope(dict(strategy_pressure_scorer='recency'), query_mode='sequential', pressure=False)


# --- contract -----------------------------------------------------------------

def record(begin, count=32, slot=None):
    return NS(history_begin=begin, count=count, handle=NS(slot=begin // 32 if slot is None else slot))


def query_view(eligible=(False, True, True, True)):
    records = [record(32 * i) for i in range(len(eligible))]
    return QueryLayerView(3, records, eligible, None, None, 32 * len(eligible), {})


def test_query_scores_are_checked():
    view = query_view()
    assert check_query_scores(view, [0., .2, .1, .7]) == [0., .2, .1, .7]
    for bad, match in (([0., .2, .1], 'scores for'), ([0., float('nan'), .1, .7], 'finite'),
                       ([.5, .2, .1, .7], 'ineligible')):
        with pytest.raises(ContractError, match=match):
            check_query_scores(view, bad)


def test_pressure_scores_are_checked():
    view = PressureLayerView(0, [(0, 1, 0, 32, 0), (1, 1, 32, 32, 32)], None, None, False)
    assert check_pressure_scores(view, [0., 1.]) == [0., 1.]
    for bad, match in (([1.], 'window blocks'), ([-1., 1.], 'non-negative'),
                       ([float('inf'), 1.], 'finite'), ([0., 0.], 'positive')):
        with pytest.raises(ContractError, match=match):
            check_pressure_scores(view, bad)


def test_picks_are_checked():
    view = PickView('pressure', {}, ((1, .5), (2, .4), (3, .3)), (0,), 96, None)
    assert check_pick(view, [0, 1, 2]) == (0, 1, 2)
    for bad, match in (([1, 2], 'mandatory'), ([0, 1, 9], 'outside the window'),
                       ([0, 1, 2, 3], 'budget'), ([0, 1, 1], 'twice')):
        with pytest.raises(ContractError, match=match):
            check_pick(view, bad)
    # A Query pick may take any scored block.
    assert check_pick(PickView('query', {7: 1.}, (), (), 64, None), [7, 9]) == (7, 9)


def test_recency_scorers_rank_by_position_and_keep_the_contract():
    view = query_view()
    scores = strategy.resolve(strategy.QUERY_SCORER, 'recency').scores(view)
    assert check_query_scores(view, scores) == [0., .5, .75, 1.]
    window = PressureLayerView(0, [(4, 1, 96, 32, 0), (2, 1, 0, 32, 32), (9, 1, 480, 16, 64)], None, None, False)
    assert check_pressure_scores(window, strategy.resolve(strategy.PRESSURE_SCORER, 'recency').scores(window)) == [
        128., 32., 496.]


# --- the pressure seam, on the native store -------------------------------------

def pressure_plan(seed=3, *, fast=False, layers=4, fused=False, monkeypatch=None, **settings):
    if fused:
        monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LAYER_PICK', '1')
        monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', '1')
    c = controller(); c.names = tuple(f'layer{i}' for i in range(layers)); c.fast_prepare = fast or fused
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config)
    for key, value in settings.items():
        setattr(o, key, value)
    hs = c._history('r', o)
    rows = list(a.records.values()); rng = random.Random(seed)
    for history in hs:
        history.values = {r.history_begin: [rng.choice([.01, .2, .2, 1.]), 1.] for r in rows if rng.random() > .1}
        history.values[rows[0].history_begin] = [1., 1.]
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    try:
        c.prepare(NS(options=o), job)
        chosen, evidence = job.layerwise_pressure_plan
        return [[r.history_begin for r in group] for group in chosen], evidence
    finally:
        assert a.memory.close()


@pytest.mark.parametrize('fast', [False, True])
def test_naming_the_built_in_pressure_scorer_changes_nothing(fast):
    assert pressure_plan(fast=fast) == pressure_plan(fast=fast, strategy_pressure_scorer='attention')


@pytest.mark.parametrize('fast', [False, True])
def test_recency_pressure_keeps_the_newest_window_blocks_in_every_layer(fast):
    attention, _ = pressure_plan(fast=fast)
    chosen, evidence = pressure_plan(fast=fast, strategy_pressure_scorer='recency')
    assert len(set(map(tuple, chosen))) == 1 and chosen != attention
    candidates = evidence[0]['candidate_starts']
    free = len(chosen[0]) - 1
    # The sink block stays; the rest of the budget goes to the newest candidates.
    assert chosen[0] == [0] + sorted(candidates)[-free:]
    assert {layer['actual_method'] for layer in evidence} == {'recency_per_layer'}
    assert all('candidate_scores' not in layer for layer in evidence)


def test_a_strategy_scorer_runs_through_the_fused_native_pick(monkeypatch):
    plain = pressure_plan(strategy_pressure_scorer='recency')
    fused = pressure_plan(fused=True, monkeypatch=monkeypatch, strategy_pressure_scorer='recency')
    assert fused == plain


@pytest.mark.parametrize('scorer', ['attention', 'recency'])
@pytest.mark.parametrize('fast', [False, True])
def test_trace_scores_record_one_score_per_candidate(scorer, fast):
    base, plain = pressure_plan(fast=fast, strategy_pressure_scorer=scorer)
    chosen, evidence = pressure_plan(fast=fast, strategy_pressure_scorer=scorer, strategy_trace_scores=True)
    assert chosen == base
    for layer, reference in zip(evidence, plain):
        assert len(layer['candidate_scores']) == len(layer['candidate_starts'])
        replay_fields = {'candidate_scores', 'mandatory_starts', 'budget_tokens', 'sink_tokens'}
        assert {k: v for k, v in layer.items() if k not in replay_fields} == reference
        assert layer['budget_tokens'] > 0 and layer['sink_tokens'] >= 0


def test_a_scorer_that_breaks_the_contract_stops_the_refresh():
    class Negative:
        name = 'test-negative'

        def scores(self, view):
            return [-1.] * len(view.mapping)

    _registry.register(Strategy(strategy.PRESSURE_SCORER, 'test-negative', Negative))
    try:
        with pytest.raises(ContractError, match='non-negative'):
            pressure_plan(strategy_pressure_scorer='test-negative')
    finally:
        del _registry._REGISTRY[strategy.PRESSURE_SCORER]['test-negative']


# --- the picker seam, on the native store ---------------------------------------

def memory_with_blocks():
    authority = native_authority(count=256, budget=96, sink=1)
    return authority.memory, [r.handle.slot for r in authority.records.values()]


def test_a_picker_replaces_only_the_scores_to_blocks_step():
    memory, slots = memory_with_blocks()
    scores = {slot: float(slot + 1) for slot in slots}
    native = memory.preview_selection(scores, max_tokens=64)
    seen = []

    class Through:
        name = 'through'

        def pick(self, view):
            seen.append(view)
            return native

    memory.picker = Through()
    assert memory.preview_selection(scores, max_tokens=64) == tuple(native)
    assert (seen[0].kind, dict(seen[0].scores), seen[0].budget_tokens) == ('query', scores, 64)
    memory.preview_selection({}, pressure_scores=[(s, 1.) for s in native], max_tokens=64)
    assert seen[1].kind == 'pressure' and [s for s, _ in seen[1].candidates] == list(native)
    with pytest.raises(ValueError, match='native picker only'):
        memory.preview_group_selection(((0, 32),), (1.,))
    with pytest.raises(ValueError, match='native picker only'):
        memory.preview_selection(scores, prefill_scores={0: 1.})
    memory.picker = None
    assert memory.preview_selection(scores, max_tokens=64) == tuple(native)
    assert memory.close()


def test_a_picker_that_breaks_the_contract_is_refused():
    memory, slots = memory_with_blocks()

    class Greedy:
        name = 'greedy'

        def pick(self, view):
            return tuple(slots)

    memory.picker = Greedy()
    with pytest.raises(ContractError, match='budget'):
        memory.preview_selection({slot: 1. for slot in slots}, max_tokens=64)
    with pytest.raises(ContractError, match='outside the window'):
        memory.preview_selection({}, pressure_scores=[(slots[0], 1.)], max_tokens=4096)
    assert memory.close()


# --- decision trace -------------------------------------------------------------

def write_trace(path, *, last=96, scores=False):
    events = [
        dict(event='armed', request_id='x'),
        dict(event='pressure_selected', request_id='x', accepted_history=128, layers=[
            dict(layer=0, actual_method='accumulated_attention_per_layer', candidate_starts=[0, 32, 64],
                 selected_starts=[0, 64], **({'candidate_scores': [1., .1, .5]} if scores else {})),
            dict(layer=1, actual_method='accumulated_attention_per_layer', candidate_starts=[0, 32, 64],
                 selected_starts=[0, 32])]),
        dict(event='query_candidates', request_id='x', history=160, candidate_starts=[0, 32, 64, 96]),
        dict(event='layer_selected', request_id='x', layer=0, selected_starts=[0, last],
             **({'candidate_scores': [0., .1, .2, .7]} if scores else {})),
        dict(event='pressure_selected', request_id='y', accepted_history=64, selected_tokens=64),
    ]
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    return path


def test_trace_reads_per_layer_decisions_and_compares_runs(tmp_path):
    capture = tmp_path / 'run' / 'server' / 'capture'
    capture.mkdir(parents=True)
    a = trace.read(write_trace(capture / trace.FILE, scores=True).parents[3])
    assert [(d.kind, d.request, d.layer, d.history, d.selected) for d in a] == [
        ('pressure', 0, 0, 128, (0, 64)), ('pressure', 0, 1, 128, (0, 32)), ('query', 0, 0, 160, (0, 96))]
    assert a[0].scores == (1., .1, .5) and a[1].scores == () and a[2].candidates == (0, 32, 64, 96)
    summary = trace.summary(a)
    assert (summary['decisions'], summary['pressure']['with_scores'], summary['query']['methods']) == (3, 1, ['meank'])
    same = trace.read(write_trace(tmp_path / 'same.jsonl'))
    assert trace.diff(a, same)['same'] and trace.digest(a) == trace.digest(same)
    other = trace.read(write_trace(tmp_path / 'other.jsonl', last=64))
    result = trace.diff(a, other)
    assert not result['same'] and result['equal'] == 2
    assert result['different'] == [dict(index=2, kind='query', request=0, layer=0, history=(160, 160),
                                        selected=(2, 2), common=1)]
    assert trace.main(['diff', str(tmp_path / 'same.jsonl'), str(tmp_path / 'other.jsonl')]) == 1
    assert trace.main(['diff', str(capture / trace.FILE), str(tmp_path / 'same.jsonl')]) == 0
    with pytest.raises(FileNotFoundError):
        trace.locate(tmp_path / 'missing')
    assert not trace.diff([], [])['same']


# --- paths: one registry for the global and the per-layer path ----------------

def test_strategies_declare_the_paths_they_decide_on():
    G, L = strategy.GLOBAL, strategy.LAYER
    q, p = strategy.QUERY_SCORER, strategy.PRESSURE_SCORER
    assert strategy.names(q, G) == ('key_fps', 'meank', 'overlap', 'recency')
    assert strategy.names(q, L) == ('meank', 'recency')
    assert strategy.names(p, G) == ('attention', 'graph', 'recency')
    assert strategy.names(p, L) == ('attention', 'lift', 'recency')
    assert strategy.get(q, 'overlap').legacy and not strategy.get(q, 'key_fps').legacy
    # lift (age-adjusted pressure): built in, per layer only.
    assert strategy.resolve(p, 'lift', L) is None
    with pytest.raises(ValueError, match='does not run on the global path'):
        strategy.resolve(p, 'lift', G)
    # recency pressure: built in on the global path, a strategy object per layer.
    assert strategy.resolve(p, 'recency', G) is None
    assert strategy.resolve(p, 'recency', L).name == 'recency'
    assert strategy.resolve(q, 'recency', G).name == 'recency'
    with pytest.raises(ValueError, match='does not run on the layer path'):
        strategy.resolve(q, 'key_fps', L)
    with pytest.raises(ValueError, match='nonempty subset'):
        Strategy(q, 'x', lambda: None, paths=('elsewhere',))
    with pytest.raises(ValueError, match='only on its own paths'):
        Strategy(q, 'x', lambda: None, native=(G,), paths=(L,))


def test_serving_binds_every_global_built_in():
    from vllm_kvmem.serving import authority, prefill_attention
    G = strategy.GLOBAL
    assert strategy.native(strategy.QUERY_SCORER, 'meank', G) is authority._global_meank
    assert strategy.native(strategy.QUERY_SCORER, 'key_fps', G) is authority._global_key_fps
    assert strategy.native(strategy.QUERY_SCORER, 'overlap', G) is authority._global_overlap
    for name in ('recency', 'attention', 'graph'):
        assert strategy.native(strategy.PRESSURE_SCORER, name, G) is getattr(prefill_attention, '_pressure_' + name)
    with pytest.raises(ValueError, match='not native'):
        strategy.bind(strategy.QUERY_SCORER, 'recency', G, lambda: None)
    with pytest.raises(ValueError, match='does not run'):
        strategy.native(strategy.QUERY_SCORER, 'key_fps', strategy.LAYER)


def test_on_the_global_path_the_strategy_options_are_the_older_switches():
    from vllm_kvmem.core.options import RuntimeOptions
    recency = RuntimeOptions(strategy_pressure_scorer='recency')
    assert (recency.strategy_path, recency.pressure_policy, recency.action_pressure) == ('global', 'recency', False)
    assert RuntimeOptions(strategy_pressure_scorer='recency', pressure_policy='recency').pressure_policy == 'recency'
    with pytest.raises(ValueError, match='conflicts with pressure_policy'):
        RuntimeOptions(strategy_pressure_scorer='recency', pressure_policy='attention')
    # Per layer the same name is the per-layer scorer; the global policy is untouched.
    layer = RuntimeOptions(strategy_pressure_scorer='recency', strategy_path='layer')
    assert layer.pressure_policy == RuntimeOptions().pressure_policy
    assert RuntimeOptions(strategy_query_scorer='recency').query_score_mode == 'mean_k'
    with pytest.raises(ValueError, match='replaces query_score_mode'):
        RuntimeOptions(strategy_query_scorer='recency', query_score_mode='key_fps_4')
    with pytest.raises(ValueError, match='does not run on the layer path'):
        RuntimeOptions(strategy_query_scorer='key_fps', strategy_path='layer')
    with pytest.raises(ValueError, match='strategy_path'):
        RuntimeOptions(strategy_path='both')


def test_key_fps_by_either_name_is_the_same_option():
    from vllm_kvmem.core.options import RuntimeOptions
    # Without the graph prerequisites both spellings fail the same validation.
    with pytest.raises(ValueError, match='query_score_mode=key_fps_4 requires'):
        RuntimeOptions(query_score_mode='key_fps_4')
    with pytest.raises(ValueError, match='query_score_mode=key_fps_4 requires'):
        RuntimeOptions(strategy_query_scorer='key_fps')
    with pytest.raises(ValueError, match='conflicts with query_score_mode'):
        RuntimeOptions(strategy_query_scorer='key_fps', query_score_mode='overlap_16_8')


def test_the_global_path_runs_a_strategy_object_behind_the_contract():
    from vllm_kvmem.serving.authority import global_query_scorer, _global_meank
    records = [record(32 * i) for i in range(4)]
    auth = NS(query_score_mode='mean_k', graph_options=NS(strategy_path='global', strategy_query_scorer='recency'))
    scorer = global_query_scorer(auth)
    assert auth.query_strategy_name == 'recency'
    scores, mean_scores = scorer(auth, records, None, None, (False, True, True, True), None, False, None)
    assert mean_scores is None and scores[0] == 0 and scores[1] < scores[2] < scores[3]
    with pytest.raises(ValueError, match='public Query only'):
        scorer(auth, records, None, None, (False, True, True, True), None, True, None)
    # The per-layer path keeps its shared frame on the built-in scorer.
    auth.graph_options = NS(strategy_path='layer', strategy_query_scorer='recency')
    assert global_query_scorer(auth) is _global_meank


def test_global_pressure_dispatches_by_policy():
    from vllm_kvmem.serving.prefill_attention import select_pressure
    calls = []
    memory = NS(select=lambda scores, pins, **kw: calls.append((scores, pins, kw)) or 'plan')
    auth = NS(graph_options=NS(pressure_policy='recency'), memory=memory,
              records={1: NS(handle=NS(slot=7), history_begin=64)})
    assert select_pressure(auth, (NS(slot=3),), max_tokens=96) == 'plan'
    assert calls == [({7: 64.0}, (3,), dict(action=False, max_tokens=96))]
    auth.graph_options = NS(pressure_policy='other')
    with pytest.raises(ValueError, match='Unknown pressure policy'):
        select_pressure(auth)


def test_trace_includes_global_decisions(tmp_path):
    capture = tmp_path / 'server' / 'capture'
    capture.mkdir(parents=True)
    write_trace(capture / trace.FILE)
    rows = [dict(request_id='x', selection='pressure', history_begin=128, selected_history_ranges=[[0, 32], [64, 32]]),
            dict(request_id='z', selection='query', history_begin=160, query_scoring_method='key_fps_4',
                 selected_history_ranges=[[0, 32]]),
            dict(request_id='z', selection='chunk', history_begin=0, selected_history_ranges=[])]
    (capture / trace.GLOBAL_FILE).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    decisions = trace.read(tmp_path)
    globals_ = [d for d in decisions if d.layer == trace.GLOBAL_LAYER]
    assert [(d.kind, d.request, d.history, d.selected, d.method) for d in globals_] == [
        ('pressure', 0, 128, (0, 64), ''), ('query', 2, 160, (0,), 'key_fps_4')]
    assert trace.summary(decisions)['global_decisions'] == 2
    only_global = tmp_path / 'global-only'
    only_global.mkdir()
    (only_global / trace.GLOBAL_FILE).write_text((capture / trace.GLOBAL_FILE).read_text())
    assert len(trace.read(only_global)) == 2 and len(trace.read(only_global / trace.GLOBAL_FILE)) == 2
