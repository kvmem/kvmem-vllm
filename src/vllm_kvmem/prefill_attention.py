"""qw3 prefill mass/exposure policy, keyed by logical history, never GPU slots."""
import math
import os


class PrefillHistory:
    def __init__(self, options):
        self.options = options
        self.values = {}
        self.end = self.samples = self.chunks = 0
        flag = os.getenv('KVMEM_NATIVE_PREFILL_COMMIT',
                         '1' if getattr(options, 'native_prefill_commit', True) else '0')
        if flag not in ('0', '1'):
            raise ValueError('KVMEM_NATIVE_PREFILL_COMMIT must be 0 or 1')
        self.native_commit = flag == '1'

    def rows(self, begin, count, local_begin, *, replay=False):
        if replay or begin + count <= self.end or not count:
            return []
        capacity = self.options.memory.select_budget + self.options.memory.gen_budget
        near_capacity = capacity - local_begin - count <= count + self.options.memory.block_tokens
        if self.end and begin + count - self.end < self.options.prefill_attention_interval and not near_capacity:
            return []
        stride = self.options.prefill_attention_stride or count
        rows = list(reversed(range(count - 1, -1, -stride)))
        # Explicit Query probes are atomic and may exceed the ordinary prefill
        # chunk. Preserve the original stride; page_attention bounds scratch
        # by evaluating these independent query rows in groups of at most 64.
        return rows

    def accept(self, end, count, accepted, rows, starts, history, local_begin, masses):
        self.last_commit_backend = None
        # qw3 commits a prefill transaction only when its whole chunk is accepted.
        if accepted != count or end <= self.end or not rows:
            return False
        decay = math.exp2(-(end - self.end) / self.options.prefill_attention_half_life)
        if self.native_commit:
            from . import _native
            # CPU tensor.numpy() shares its storage; the production path never
            # materializes hundreds of thousands of Python scalar objects.
            if hasattr(masses, 'numpy'):
                buffer = masses.numpy()
            else:
                import numpy as np
                try:
                    buffer = np.asarray(masses)
                except (TypeError, ValueError):
                    # Retain legacy malformed-sequence validation/rollback.
                    return self._accept_python(end, rows, starts, history, local_begin, masses, decay)
                if buffer.dtype.kind != 'f':
                    buffer = buffer.astype('float64')
            if buffer.ndim != 2:
                return self._accept_python(end, rows, starts, history, local_begin, masses, decay)
            self.last_commit_backend = 'native'
            updated = _native.commit_prefill_attention(self.values, decay, rows,
                starts, history, local_begin, buffer)
            self._publish(end, rows, updated)
            return True
        if hasattr(masses, 'tolist'):
            masses = masses.tolist()
        return self._accept_python(end, rows, starts, history, local_begin, masses, decay)

    def _accept_python(self, end, rows, starts, history, local_begin, masses, decay):
        self.last_commit_backend = 'python'
        updated = {key: [decay * v for v in self.values.get(key, (0., 0.))] for key in history}
        inv = 1. / len(rows)
        for row, values in zip(rows, masses):
            if len(values) != len(history) or any(not math.isfinite(v) or v < 0 for v in values) or abs(sum(values)-1) > .005:
                raise RuntimeError('Invalid prefill attention mass')
            exposed = set()
            for start, key, value in zip(starts, history, values):
                if start > local_begin + row:
                    continue
                updated[key][0] += inv * value
                if key not in exposed:
                    updated[key][1] += inv
                    exposed.add(key)
        if len(rows) != len(masses):
            raise RuntimeError('Missing prefill attention rows')
        self._publish(end, rows, updated)
        return True

    def _publish(self, end, rows, updated):
        self.values = updated
        self.end = end
        self.samples += len(rows)
        self.chunks += 1

    def scores(self):
        return {key: mass / exposure for key, (mass, exposure) in self.values.items() if exposure > 0}

    def snapshot(self):
        return {key: tuple(v) for key, v in self.values.items()}, self.end, self.samples, self.chunks

    def restore(self, snapshot, position):
        values, end, samples, chunks = snapshot
        if end > position:
            self.values = {}; self.end = self.samples = self.chunks = 0
        else:
            self.values = {key: list(v) for key, v in values.items()}
            self.end, self.samples, self.chunks = end, samples, chunks


def selection_report(authority, *, phase=None):
    state = getattr(authority, 'action_history', None)
    prefill = getattr(state, 'prefill', None)
    scores = prefill.scores() if prefill else {}
    live = {r.history_begin for r in authority.records.values()}
    output = state is not None and any(v > 0 and key in live for key, v in state.scores.items())
    source = 'output' if output else 'prefill' if any(v > 0 and key in live for key,v in scores.items()) else 'none'
    result = {'attention_source': source, 'prefill_attention_samples': prefill.samples if prefill else 0,
            'prefill_attention_chunks': prefill.chunks if prefill else 0,
            'prefill_attention_pages': sum(v > 0 and key in live for key,v in scores.items())}
    selection = (getattr(authority, 'query_selection_report', None) if phase == 'query' else None) or getattr(authority, 'graph_selection_report', {})
    if phase is not None and selection.get('phase') == phase:
        result.update({key: selection[key] for key in ('phase','seed_source','scoring_version',
            'score_domain_version','pressure_policy','alpha','score_mask_count','transport_mask_count')
            if key in selection})
        if phase == 'pressure':
            result['attention_source'] = selection['seed_source']
    return result


def exclude_query_replay(runner, authority, request_id, begin, end, explicit):
    state = getattr(authority, 'action_history', None)
    if state is None:
        return
    if explicit:
        # qw3 replays the public Query AND its assistant/template suffix. The
        # latter may be a separate native prefill batch; it is not new input.
        states = runner.req_states
        end = int(states.prompt_len.np[states.req_id_to_index[request_id]])
    state.prefill_replay = (begin, end)


def pressure_seed(authority, records):
    """Read the old complete-window output/prefill seed without committing it.

    Fallback is decided BEFORE applying the transport mask, including fixed
    sink/mandatory rows. Selection never finishes/decays the Action state.
    """
    state = getattr(authority, 'action_history', None)
    if state is None:
        raise ValueError('Attention pressure requires accepted attention state')
    scores = []
    for slot, generation, begin, _, _ in records:
        handle = authority.memory._store.handle(slot)
        if handle.generation != generation:
            raise RuntimeError('Stale resident identity in pressure attention')
        score = state.scores.get(begin, 0.)
        if state.samples:
            score = state.options.action_decay * score + state.mass.get(begin, 0.) / state.samples
        scores.append((slot, score))
    if any(not math.isfinite(score) or score < 0 for _, score in scores):
        raise ValueError('Pressure seed must be finite and nonnegative')
    source = 'output' if any(score > 0 for _, score in scores) else 'none'
    if state.prefill is not None and not any(score > 0 for _, score in scores):
        prefill = state.prefill.scores()
        scores = [(slot, prefill.get(begin, 0.)) for slot, _, begin, _, _ in records]
        source = 'prefill' if any(score > 0 for _, score in scores) else 'none'
    if any(not math.isfinite(score) or score < 0 for _, score in scores):
        raise ValueError('Pressure seed must be finite and nonnegative')
    if not scores and authority.history_end:
        raise RuntimeError('Attention pressure has no resident history')
    return scores, source


def select_pressure(authority, mandatory=(), *, max_tokens=None, candidate_records=None):
    state = getattr(authority, 'action_history', None)
    options = getattr(authority, 'graph_options', None) or getattr(state, 'options', None)
    policy = getattr(options, 'pressure_policy', None)
    if policy is None:
        policy = 'attention' if state is not None and getattr(state.options, 'action_pressure', False) else 'recency'
    if policy == 'recency':
        return authority.memory.select({r.handle.slot: float(r.history_begin) for r in authority.records.values()},
                                       tuple(h.slot for h in mandatory), action=False, max_tokens=max_tokens)
    # Checkpoint records remain authoritative after physical residency clears.
    records = tuple(authority.memory._store.working_records() if candidate_records is None else candidate_records)
    if policy == 'graph':
        from .graph_integration import pressure_score_and_select
        return pressure_score_and_select(authority, records, mandatory, max_tokens=max_tokens)
    if policy != 'attention':
        raise ValueError('Unknown pressure policy')
    scores, source = pressure_seed(authority, records)
    if getattr(authority,'block_graph',None) is not None:
        from .authority import SCORE_DOMAIN_VERSION, record_key
        from .graph_integration import emit_graph_report
        graph=authority.block_graph
        picked = authority.memory.preview_selection({}, tuple(h.slot for h in mandatory),
            action=False,pressure_scores=scores,max_tokens=max_tokens)
        report=dict(phase='pressure',seed_source=source,pressure_policy='attention',
            mode=options.graph_mode,scoring_version=getattr(options,'graph_scoring_version','v1'),
            alpha=getattr(options,'graph_alpha',.2),propagation_applied=False,
            score_domain_version=SCORE_DOMAIN_VERSION,candidate_count=len(records),
            score_mask_count=len(records),transport_mask_count=0,
            revision=graph.revision,runtime_epoch=graph.runtime_epoch,
            accepted_through=graph.accepted_through,
            selected_tokens=sum(authority.records[record_key(authority.memory._store.handle(s))].count for s in picked),
            budget=max_tokens or authority.memory.config.select_budget,
            graph_bytes=graph.bytes,graph_live_bytes=graph.live_bytes,
            locality_provenance=dict(getattr(graph,'locality_provenance',())))
        authority.graph_selection_report=report
        emit_graph_report(authority,report)
    plan = authority.memory.select({}, tuple(h.slot for h in mandatory), action=False, pressure_scores=scores, max_tokens=max_tokens)
    return plan
