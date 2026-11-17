"""Strategy capability checks: declared needs against what a launched configuration provides.

One check at launch replaces the strategy rejections that were spread over the
launcher and the refresh paths (layering step 4):

- every chosen strategy must run on the launched path (the registry's paths);
- its declared needs must be signals the configuration produces;
- per-layer selection refuses, at launch, the combinations it used to refuse
  only at the first refresh;
- strategies chosen through the older options that decide only on the global
  path are reported when the per-layer path is launched: there they shape the
  shared frame (and a global MTP selection) but no attention layer's window.

    python -m vllm_kvmem.serving.capabilities table     # strategy x path table (Markdown)
"""
import sys

# Signals a strategy can declare in Strategy.needs, and when serving produces them.
SIGNALS = {
    'positions': 'block positions in the history (always)',
    'layer_query': 'the Query rows of each attention layer (always, at a Query)',
    'mean_index': 'the per-block mean-key index (always)',
    'key_fps_index': 'the key FPS index (built for the key_fps scorer)',
    'window_cache': 'the overlapping-window cache (built for the overlap scorer)',
    'layer_attention': 'accumulated attention (action_attention)',
    'block_graph': 'the block graph (graph_mode locality, attention or multi)',
}
GRAPH_SCORING = ('locality', 'attention', 'multi')
# Signals that cost GPU or host time to produce.
COSTLY = frozenset(('layer_attention', 'key_fps_index', 'window_cache', 'block_graph'))


def provided_signals(kvmem):
    """The signals a profile's kvmem options make serving produce."""
    get = kvmem.get
    signals = {'positions', 'layer_query', 'mean_index'}
    if get('action_attention', False):
        signals.add('layer_attention')
    if get('graph_mode', 'off') in GRAPH_SCORING:
        signals.add('block_graph')
    mode = get('query_score_mode', 'mean_k')
    if mode == 'key_fps_4' or get('strategy_query_scorer') == 'key_fps':
        signals.add('key_fps_index')
    if mode == 'overlap_16_8' or get('strategy_query_scorer') == 'overlap':
        signals.add('window_cache')
    return signals


def chosen(kvmem, path):
    """(role, name) of every strategy the options choose on this path."""
    from .. import strategy
    query = kvmem.get('strategy_query_scorer', 'meank')
    pressure = kvmem.get('strategy_pressure_scorer', 'attention')
    if path == strategy.GLOBAL:
        if query == 'meank':
            query = strategy.QUERY_SCORE_MODES.get(kvmem.get('query_score_mode', 'mean_k'), 'meank')
        if pressure == 'attention':
            policy = kvmem.get('pressure_policy')
            if policy is None:
                policy = 'attention' if kvmem.get('action_pressure', False) else 'recency'
            pressure = policy
    return ((strategy.QUERY_SCORER, query), (strategy.PRESSURE_SCORER, pressure),
            (strategy.PICKER, kvmem.get('strategy_picker', 'native')))


def global_only(kvmem, environment):
    """Strategies chosen through older options that decide only on the global path."""
    found = []
    mode = kvmem.get('query_score_mode', 'mean_k')
    if mode != 'mean_k':
        found.append('query_score_mode=' + mode)
    if kvmem.get('query_mean_quota_fraction', 0):
        found.append('query_mean_quota_fraction')
    if kvmem.get('graph_mode', 'off') in GRAPH_SCORING:
        found.append('graph_mode=' + kvmem['graph_mode'])
    if kvmem.get('guided_query_tokens', 0):
        found.append('guided_query_tokens')
    if kvmem.get('query_history', False):
        found.append('query_history')
    if environment.get('KVMEM_RECALL_METHOD'):
        found.append('KVMEM_RECALL_METHOD=' + environment['KVMEM_RECALL_METHOD'])
    if environment.get('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES'):
        found.append('--experimental-query-token-scores')
    return found


def check_qsa(kvmem, environment):
    """Flash Next QSA, strategy_path=layer: per-layer MeanK at the final Query (the
    first KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS layers shared) and per-layer pressure
    from QSA's own sparse-attention mass (serving/qsa_pressure.py) for the attention
    and lift scorers; recency keeps the global pressure selection."""
    from .. import strategy
    choice = chosen(kvmem, strategy.LAYER)
    for role, name in choice:
        strategy.check_path(role, name, strategy.LAYER)
    query, pressure_name, picker = (name for _, name in choice)
    if query != 'meank' or picker != 'native':
        raise ValueError('The QSA per-layer path runs the built-in MeanK Query scorer and native picker')
    if pressure_name not in ('attention', 'lift', 'recency'):
        raise ValueError('The QSA per-layer path supports the attention, lift and recency pressure scorers')
    shared = str(environment.get('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS', '0') or '0')
    if not shared.isdigit():
        raise ValueError('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS must be a non-negative integer')
    return dict(path='layer', backend='qsa', query_scorer=query, pressure_scorer=pressure_name,
                picker=picker, shared_query_layers=int(shared),
                per_layer_pressure=pressure_name in ('attention', 'lift'))


def check(kvmem, *, query_mode, pressure, batch_selection=False, environment=None):
    """Refuse what the launched configuration cannot run; return a report of what it will run.

    query_mode: the --layerwise-query-diagnostic value (None on the global path).
    pressure: --layerwise-prefill-pressure. batch_selection: pressure batch selection is on.
    """
    from .. import strategy
    environment = environment or {}
    if query_mode == 'qsa':
        return check_qsa(kvmem, environment)
    path = strategy.LAYER if query_mode else strategy.GLOBAL
    signals = provided_signals(kvmem)
    choice = chosen(kvmem, path)
    for role, name in choice:
        entry = strategy.check_path(role, name, path)
        if path == strategy.LAYER and role == strategy.PRESSURE_SCORER and not pressure:
            continue    # no per-layer pressure is launched; its scorer never runs
        missing = [need for need in entry.needs if need not in signals]
        if missing:
            raise ValueError('%s %r needs %s, which this configuration does not produce' % (
                role, name, '; '.join('%s: %s' % (need, SIGNALS.get(need, 'unknown signal')) for need in missing)))
    query, pressure_name, picker = (name for _, name in choice)
    if path == strategy.LAYER:
        if query_mode != 'sequential' and query != 'meank':
            raise ValueError('strategy_query_scorer applies to sequential per-layer Query selection '
                             '(--layerwise-query-diagnostic sequential)')
        if pressure_name != 'attention' and not pressure:
            raise ValueError('strategy_pressure_scorer applies to per-layer pressure selection '
                             '(--layerwise-prefill-pressure)')
        policy = kvmem.get('pressure_policy')
        if policy is None:
            policy = 'attention' if kvmem.get('action_pressure', False) else 'recency'
        if pressure and policy != 'attention':
            raise ValueError('Per-layer pressure requires pressure_policy=attention; choose another pressure '
                             'strategy with strategy_pressure_scorer')
        if pressure and (kvmem.get('guided_query_tokens', 0) or kvmem.get('query_history', False) or
                         not kvmem.get('prefill_attention', False)):
            raise ValueError('Per-layer pressure requires prefill_attention and no guidance or Query history')
        if batch_selection and (not strategy.get(strategy.PRESSURE_SCORER, pressure_name).native_on(path) or
                                not strategy.get(strategy.PICKER, picker).native_on(path)):
            raise ValueError('Pressure batch selection implements the attention scorer and native picker only')
    if path == strategy.LAYER and pressure and pressure_name == 'lift' and \
            environment.get('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT') == '0':
        raise ValueError('strategy_pressure_scorer=lift is kept in the native layer-history state; it needs '
                         'KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT=1')
    shared = str(environment.get('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS', '0') or '0')
    if not shared.isdigit():
        raise ValueError('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS must be a non-negative integer')
    if int(shared) and (query_mode != 'sequential' or environment.get('KVMEM_LAYERWISE_QUERY_NO_PROBE') == '1'):
        raise ValueError('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS applies to the sequential per-layer Query with its '
                         'probe (--layerwise-query-diagnostic sequential), which makes the shared selection')
    report = dict(path=path, query_scorer=query, pressure_scorer=pressure_name, picker=picker,
                  signals=sorted(signals))
    if int(shared):
        report['query_global_layers'] = int(shared)
    # Signals by need is not wired yet: report the costly signals no chosen strategy reads.
    needed = set()
    for role, name in choice:
        if not (path == strategy.LAYER and role == strategy.PRESSURE_SCORER and not pressure):
            needed.update(strategy.get(role, name).needs)
    if kvmem.get('graph_mode', 'off') in GRAPH_SCORING:
        needed.add('block_graph')    # graph propagation of the (shared-frame) Query reads it
    report['unused_signals'] = sorted(signals & COSTLY - needed)
    gc = kvmem.get('historical_gc')
    if isinstance(gc, dict) and gc.get('mode', 'off') != 'off':
        report['evictor'] = strategy.check_path(strategy.EVICTOR, gc.get('policy', 'lru'), path).name
    if path == strategy.LAYER:
        report['shared_frame_only'] = global_only(kvmem, environment)
    return report


def table():
    """Every registered strategy on every path, as Markdown."""
    from .. import strategy
    lines = ['| Role | Name | Global path | Per-layer path | Needs |', '| --- | --- | --- | --- | --- |']
    for entry in strategy.table():
        cells = []
        for path in (strategy.GLOBAL, strategy.LAYER):
            if path not in entry.paths:
                cells.append('refused at launch')
            else:
                cells.append('runs (built in)' if entry.native_on(path) else 'runs (strategy object)')
        lines.append('| %s | `%s`%s | %s | %s | %s |' % (
            entry.role.replace('_', ' '), entry.name, ' (legacy)' if entry.legacy else '', cells[0], cells[1],
            ', '.join(entry.needs) or '-'))
    return '\n'.join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['table']:
        print(table())
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
