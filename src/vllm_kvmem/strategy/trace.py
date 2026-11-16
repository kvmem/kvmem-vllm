"""Decision traces: what every selection chose in a serving run.

Serving already writes its per-layer decisions to capture/layerwise_query.jsonl
and every refresh's global selection (one for all layers, layer -1) to
capture/bounded_refresh.jsonl. This module reads both into one shape and
compares two runs. It is the test for both layers:

- a serving change keeps the strategy, so two runs must give equal traces;
- a strategy change is studied by replaying a trace recorded with
  strategy_trace_scores (candidate scores included) on the CPU.

    python -m vllm_kvmem.strategy.trace summary RUN
    python -m vllm_kvmem.strategy.trace diff RUN_A RUN_B      (exit 1 if they differ)
    python -m vllm_kvmem.strategy.trace table                 (registered strategies)

RUN is the jsonl file, a capture directory, or any directory above one.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

FILE = 'layerwise_query.jsonl'
GLOBAL_FILE = 'bounded_refresh.jsonl'
EVICTION_FILE = 'historical_gc_evictions.jsonl'
GLOBAL_LAYER = -1


@dataclass(frozen=True)
class Decision:
    kind: str              # 'query' or 'pressure'
    request: int           # order of the request's first decision in the run
    layer: int
    history: int           # accepted history length the decision was made at
    method: str
    selected: tuple        # history_begin of every selected block, in order
    candidates: tuple      # history_begin of every candidate, or () when not recorded
    scores: tuple          # one score per candidate, or () when not recorded
    # Recorded with strategy_trace_scores, for an offline replay (replay.py):
    fixed: tuple = ()      # blocks placed before scoring (sink, mandatory, recent, partial)
    excluded: tuple = ()   # candidates that may not be chosen (Query: partial blocks outside the frame)
    eligible: tuple = ()   # per candidate, whether it competes (Query)
    counts: tuple = ()     # per candidate, its token count, when recorded
    budget: int = -1       # selected-token budget
    sink: int = -1         # sink tokens (pressure)
    stage: str = ''        # 'prefill' or 'decode' (pressure)


def locate(run):
    """The per-layer decision file of a run (absent on the global path)."""
    path = Path(run)
    if path.is_file():
        return path
    return capture_dir(path) / FILE


def capture_dir(run):
    path = Path(run)
    if path.is_file():
        return path.parent
    for candidate in (path, path / 'capture', path / 'server' / 'capture', path / 'run' / 'server' / 'capture'):
        if (candidate / FILE).is_file() or (candidate / GLOBAL_FILE).is_file():
            return candidate
    raise FileNotFoundError('No %s or %s under %s' % (FILE, GLOBAL_FILE, run))


def read(run):
    """Every decision of a run: per-layer ones in serving order, then global ones in serving order.

    RUN is a run or capture directory, a per-layer jsonl file, or a bounded_refresh.jsonl file.
    """
    decisions, order = [], {}
    path = Path(run)
    if path.is_file():
        if path.name == GLOBAL_FILE:
            return _read_global(path, decisions, order)
        _read_layers(path, decisions, order)
        return _read_global(path.parent / GLOBAL_FILE, decisions, order) if path.name == FILE else decisions
    directory = capture_dir(path)
    if (directory / FILE).is_file():
        _read_layers(directory / FILE, decisions, order)
    _read_global(directory / GLOBAL_FILE, decisions, order)
    return _read_evictions(directory / EVICTION_FILE, decisions, order)


def _read_evictions(path, decisions, order):
    """One decision per request: every block the historical-GC evictor retired for it."""
    if not path.is_file():
        return decisions
    evicted, policy = {}, {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            request = order.setdefault(row['request_id'], len(order))
            evicted.setdefault(request, set()).update(int(b) for b in row.get('history_begins', ()))
            policy[request] = row.get('policy') or ''
    for request in sorted(evicted):
        decisions.append(Decision('evict', request, GLOBAL_LAYER, -1, policy[request],
                                  tuple(sorted(evicted[request])), (), ()))
    return decisions


def candidate_starts(row, block=32):
    """A trace row's candidate block starts: newer traces store ``candidate_runs``
    (``[first_start, count]`` runs of consecutive blocks), older ones the full list.

    Same decoding as vllm_kvmem.core.block_runs; kept here because this package
    imports only itself and the standard library (the gate loads it by path).
    """
    if 'candidate_runs' in row:
        return [start for first, count in row['candidate_runs'] for start in range(first, first + count * block, block)]
    return list(row.get('candidate_starts', ()))


def _read_layers(path, decisions, order):
    pool = {}
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            name = event.get('event')
            if name not in ('pressure_selected', 'layer_selected', 'query_candidates'):
                continue
            request = order.setdefault(event['request_id'], len(order))
            if name == 'query_candidates':
                pool[request] = event
            elif name == 'layer_selected':
                frame = pool.get(request, {})
                scores = tuple(event.get('candidate_scores') or ())
                decisions.append(Decision(
                    'query', request, int(event['layer']), int(frame.get('history', -1)),
                    event.get('query_scorer', 'meank'),
                    tuple(event.get('selected_starts') or event.get('selected_blocks') or ()),
                    tuple(candidate_starts(frame)) if scores else (), scores,
                    fixed=tuple(frame.get('fixed_starts', ())), excluded=tuple(frame.get('excluded_starts', ())),
                    eligible=tuple(frame.get('eligible', ())), counts=tuple(frame.get('candidate_counts', ())),
                    budget=int(frame.get('budget', -1))))
            else:
                stage = str(event.get('event_type', '')).removesuffix('_pressure')
                for layer in event.get('layers', ()):
                    scores = tuple(layer.get('candidate_scores') or ())
                    decisions.append(Decision(
                        'pressure', request, int(layer['layer']), int(event.get('accepted_history', -1)),
                        layer.get('actual_method', ''), tuple(layer.get('selected_starts', ())),
                        tuple(candidate_starts(layer)), scores,
                        fixed=tuple(layer.get('mandatory_starts', ())), budget=int(layer.get('budget_tokens', -1)),
                        sink=int(layer.get('sink_tokens', -1)), stage=stage))
    return decisions


def _read_global(path, decisions, order):
    """One decision per refresh: the selection made for every layer at once."""
    if not path.is_file():
        return decisions
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            ranges = row.get('selected_history_ranges')
            if ranges is None or row.get('selection') not in ('query', 'pressure'):
                continue
            request = order.setdefault(row['request_id'], len(order))
            method = row.get('query_scoring_method') or row.get('pressure_policy') or row.get('attention_source', '')
            decisions.append(Decision(row['selection'], request, GLOBAL_LAYER, int(row.get('history_begin', -1)),
                                      method, tuple(int(begin) for begin, _ in ranges), (), ()))
    return decisions


def digest(decisions):
    """One hash over what was chosen; scores and candidates are not part of it."""
    state = hashlib.sha256()
    for d in decisions:
        state.update(json.dumps([d.kind, d.request, d.layer, d.history, d.selected]).encode())
    return state.hexdigest()


def summary(decisions):
    out = dict(decisions=len(decisions), requests=len({d.request for d in decisions}), digest=digest(decisions),
               global_decisions=sum(d.layer == GLOBAL_LAYER for d in decisions))
    for kind in ('query', 'pressure', 'evict'):
        rows = [d for d in decisions if d.kind == kind]
        out[kind] = dict(decisions=len(rows), methods=sorted({d.method for d in rows}),
                         selected_blocks=sum(len(d.selected) for d in rows),
                         with_scores=sum(bool(d.scores) for d in rows))
    return out


def diff(a, b, *, limit=10):
    """Compare what two runs chose, decision by decision in serving order."""
    out = dict(a=len(a), b=len(b), equal=0, different=[])
    for index, (x, y) in enumerate(zip(a, b)):
        # A Query decision's history is recorded only with strategy_trace_scores (-1 otherwise).
        same_history = x.history == y.history or x.history < 0 or y.history < 0
        if (x.kind, x.request, x.layer, x.selected) == (y.kind, y.request, y.layer, y.selected) and same_history:
            out['equal'] += 1
        elif len(out['different']) < limit:
            common = len(set(x.selected) & set(y.selected))
            out['different'].append(dict(index=index, kind=x.kind, request=x.request, layer=x.layer,
                                         history=(x.history, y.history), selected=(len(x.selected), len(y.selected)),
                                         common=common))
    out['same'] = len(a) == len(b) == out['equal'] and len(a) > 0
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['summary'] and len(argv) == 2:
        print(json.dumps(summary(read(argv[1])), indent=1))
        return 0
    if argv[:1] == ['diff'] and len(argv) == 3:
        result = diff(read(argv[1]), read(argv[2]))
        print(json.dumps(result, indent=1))
        return 0 if result['same'] else 1
    if argv == ['table']:
        from . import table
        print('%-16s %-11s %-13s %-13s %s' % ('role', 'name', 'paths', 'built-in on', 'summary'))
        for s in table():
            print('%-16s %-11s %-13s %-13s %s%s' % (s.role, s.name, ','.join(s.paths), ','.join(s.native) or '-',
                                                    s.summary, ' (legacy)' if s.legacy else ''))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
