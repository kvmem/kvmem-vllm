"""The strategies shipped with the package.

Entries marked native are serving's existing code paths: the per-layer MeanK
Query scores, the accumulated-attention pressure scores and their age-adjusted
variant (lift, kept in the native layer-history state), the native store
picker and the native historical-GC policies. They are listed so that a
profile can name them and the strategy table is complete; serving runs them
unchanged. The recency scorers are the lower reference for comparisons: they
keep what is most recent and ignore content.
"""
from .contract import EVICTOR, GLOBAL, LAYER, PICKER, PRESSURE_SCORER, QUERY_SCORER, Strategy
from .registry import register


class RecencyQueryScorer:
    """Later history scores higher; the same choice in every layer, and globally."""
    name = 'recency'

    def scores(self, view):
        total = max(1, view.history_tokens)
        return [(record.history_begin + record.count) / total if eligible else 0.0
                for record, eligible in zip(view.records, view.eligible)]


class RecencyPressureScorer:
    """A window block scores its end position, so the newest blocks stay."""
    name = 'recency'

    def scores(self, view):
        return [float(row[2] + row[3]) for row in view.mapping]


def _native():
    raise RuntimeError('A native strategy is run by serving and is never built')


BOTH = (GLOBAL, LAYER)

register(Strategy(QUERY_SCORER, 'meank', _native, needs=('layer_query', 'mean_index'), native=True, paths=BOTH,
                  summary='Softmax of the Query against per-block mean keys (per layer, or summed over layers)'))
register(Strategy(QUERY_SCORER, 'key_fps', _native, needs=('layer_query', 'key_fps_index'), native=True,
                  paths=(GLOBAL,), summary='Max over 4 farthest-point-sampled keys per block (query_score_mode=key_fps_4)'))
register(Strategy(QUERY_SCORER, 'overlap', _native, needs=('layer_query', 'window_cache'), native=True,
                  paths=(GLOBAL,), legacy=True,
                  summary='Overlapping 16/8-token window keys (query_score_mode=overlap_16_8)'))
register(Strategy(QUERY_SCORER, 'recency', RecencyQueryScorer, needs=('positions',), paths=BOTH,
                  summary='Most recent history first; lower reference'))
register(Strategy(PRESSURE_SCORER, 'attention', _native, needs=('layer_attention',), native=True, paths=BOTH,
                  summary='Accumulated attention mass (per layer, or summed over layers)'))
register(Strategy(PRESSURE_SCORER, 'recency', RecencyPressureScorer, needs=('positions',), native=(GLOBAL,),
                  paths=BOTH, summary='Most recent window blocks stay; lower reference'))
register(Strategy(PRESSURE_SCORER, 'graph', _native, needs=('layer_attention', 'block_graph'), native=True,
                  paths=(GLOBAL,), summary='Attention seeds propagated over the block graph (pressure_policy=graph)'))
register(Strategy(PRESSURE_SCORER, 'lift', _native, needs=('layer_attention',), native=True, paths=(LAYER,),
                  summary='Age-adjusted: accumulated attention divided by what a block at the same distance '
                          'typically receives (per layer)'))
register(Strategy(PICKER, 'native', _native, native=True, paths=BOTH,
                  summary='Native store picker: sink and recent bands, mandatory blocks, top scores to the budget'))
register(Strategy(EVICTOR, 'lru', _native, native=True, paths=BOTH,
                  summary='Historical GC, least recently used (historical_gc.policy)'))
register(Strategy(EVICTOR, 'redundancy', _native, native=True, paths=BOTH,
                  summary='Historical GC by redundancy, optional thinking-only scope and LRU fallback'))

# The older profile options name the same global strategies.
QUERY_SCORE_MODES = {'mean_k': 'meank', 'key_fps_4': 'key_fps', 'overlap_16_8': 'overlap'}
