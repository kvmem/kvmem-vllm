"""Shared graph transport with separate native Query and pressure pickers."""
from contextlib import ExitStack, contextmanager
import json
import threading
import time


_LOG_LOCK = threading.Lock()
_MAX_LOG_BYTES = 16 * 1024 * 1024
_MAX_EVENTS = 256
_MAX_EVENT_BYTES = 65536
# Opt-in snapshots cover the complete 131K structured qualification inputs.
# The existing per-authority count and global log quota remain unchanged.
_MAX_REPLAY_NODES = 8192
_MAX_REPLAY_RECORD_BYTES = 2 * 1024 * 1024


def propagation_options(options):
    result = dict(mode=options.graph_mode,
        scoring_version=getattr(options, 'graph_scoring_version', 'v1'),
        aggregation=getattr(options, 'graph_aggregation', 'sum'),
        alpha=getattr(options, 'graph_alpha', .2),
        locality_weight=options.graph_locality_weight,
        attention_weight=options.graph_attention_weight,
        cross_message_weight=options.graph_cross_message_weight)
    if getattr(options,'graph_native_propagation',False):
        result['native']=True
    return result


@contextmanager
def query_snapshot(authority):
    graph = getattr(authority, 'block_graph', None)
    if graph is None:
        yield None
        return
    snapshot = graph.snapshot()
    try:
        if snapshot.accepted_through < authority.history_end:
            raise RuntimeError('Graph accepted events have not caught up with the Query index')
        if (snapshot.accepted_through > authority.history_end and
                getattr(authority, '_graph_validated_prefix', 0) < snapshot.accepted_through):
            raise RuntimeError('Future graph statistics lack a verified complete prefix')
        yield snapshot
    finally:
        snapshot.close()


def score_and_select(authority, records, raw_scores, eligible, snapshot,
                     mandatory_slots, max_tokens, *, mean_scores=None):
    """Called only for scoring modes, while the current index is pinned."""
    with ExitStack() as results:
        return _score_and_select(authority, records, raw_scores, eligible, snapshot,
            mandatory_slots, max_tokens, results, mean_scores=mean_scores)


def _score_and_select(authority, records, raw_scores, eligible, snapshot,
                      mandatory_slots, max_tokens, results, *, mean_scores=None):
    from .graph_scoring import score_graph
    graph = authority.block_graph
    options = authority.graph_options
    binding = graph.binding
    if (snapshot.owner, snapshot.epoch, snapshot.branch_id, snapshot.revision,
            snapshot.accepted_through) != binding:
        raise RuntimeError('Graph and Query index snapshots belong to different revisions')
    started = time.perf_counter()
    starts = [record.history_begin for record in records]
    result = results.enter_context(score_graph(raw_scores, starts, eligible, snapshot,
        **propagation_options(options),
        block_tokens=authority.memory.config.block_tokens, trace=False))
    graph.validate_binding(binding)
    # Generation-qualified handles were pinned by index_snapshot. Verify the
    # mapping again before converting to the native selector's slot interface.
    for record in records:
        if authority.memory._store.handle(record.handle.slot) != record.handle:
            raise RuntimeError('Graph Query index contains a recycled generation')
    raw_by_slot = {record.handle.slot: score for record, score in zip(records, raw_scores)}
    baseline = authority.memory.preview_selection(raw_by_slot, mandatory_slots,
        action=False, prefill_scores=None, max_tokens=max_tokens)
    propagated = {record.handle.slot: score for record, score in zip(records, result.scores)}
    old = set(baseline)
    positions = {record.handle.slot: record.history_begin for record in records}
    halo = getattr(options, 'query_fps_halo_blocks', 0)
    group_evidence = None
    quota_evidence = None
    quota_propagation = None
    picker_pins = mandatory_slots
    fraction = getattr(options, 'query_mean_quota_fraction', 0)
    if fraction:
        from .query_hybrid_recall import preview_hybrid_selection, quota_summary
        from .query_group_recall import halo_summary
        if (not halo or getattr(authority, 'query_score_mode', 'mean_k') != 'key_fps_4'
                or mean_scores is None):
            raise ValueError('Hybrid quota requires independent MeanK and FPS seeds')
        # Both seeds use the same immutable owner/revision and original
        # transport mask, before the MeanK quota turns its choices into pins.
        # Do not blend seeds or propagate again after reserving that quota.
        mean_result = results.enter_context(score_graph(mean_scores, starts, eligible, snapshot,
            **propagation_options(options),
            block_tokens=authority.memory.config.block_tokens, trace=False))
        graph.validate_binding(binding)
        if getattr(options, 'graph_alpha', 0):
            quota_propagation = dict(order='independent_graph_then_quota',
                mean_k=dict(mean_result.diagnostics), key_fps_4=dict(result.diagnostics))
        preview = preview_hybrid_selection(authority.memory, records, result.scores, mean_result.scores,
            mandatory_slots, max_tokens=max_tokens, mean_fraction=fraction,
            message_spans=snapshot.message_spans)
        groups, group_scores = preview['groups'], preview['group_scores']
        picker_pins = preview['mandatory_slots']
        selected = set(preview['selected_slots'])
        group_evidence, quota_evidence = preview['group_selection'], preview['quota_selection']
        record_query_inputs(authority, records, raw_scores, eligible, snapshot,
            mandatory_slots, max_tokens, group_selection=group_evidence, mean_scores=mean_scores,
            quota_propagated_scores=(dict(mean_k=mean_result.scores, key_fps_4=result.scores)
                if quota_propagation is not None else None))
    elif halo:
        from .query_group_recall import build_halo_groups, describe_halo_selection, halo_summary
        if (getattr(authority, 'query_score_mode', 'mean_k') != 'key_fps_4'
                or not result.diagnostics['identity'] or list(result.scores) != list(raw_scores)):
            raise ValueError('FPS halo requires unchanged alpha-zero FPS seeds')
        groups, group_scores, metadata = build_halo_groups(records, raw_scores,
            radius=halo, message_spans=snapshot.message_spans)
        selected = set(authority.memory.preview_group_selection(groups, group_scores,
            mandatory_slots, max_tokens=max_tokens))
        budget = max_tokens if max_tokens is not None else authority.memory.config.select_budget
        fixed = authority.memory._store.fixed_selection_blocks(list(mandatory_slots),
            pressure=False, budget=budget)
        group_evidence = describe_halo_selection(metadata, records, selected, old, fixed, budget)
        record_query_inputs(authority, records, raw_scores, eligible, snapshot,
            mandatory_slots, max_tokens, group_selection=group_evidence)
    else:
        selected = set(authority.memory.preview_selection(propagated, mandatory_slots,
            action=False, prefill_scores=None, max_tokens=max_tokens))
    trace = ()
    if options.graph_trace:
        # Detailed diagnostics are opt-in and bounded. Explain newly selected
        # blocks first, rather than whichever old targets had largest inflow.
        traced = results.enter_context(score_graph(raw_scores, starts, eligible, snapshot,
            **propagation_options(options),
            block_tokens=authority.memory.config.block_tokens, trace=True,
            trace_targets=[positions[slot] for slot in selected-old]))
        trace = traced.trace
    # A diagnostic scratch-capacity failure must precede the native assembly
    # transaction. Both score results keep their accounted leases until return.
    graph.validate_binding(binding)
    report = dict(result.diagnostics)
    domain = getattr(authority, 'query_score_domain', {})
    report.update(phase='query', seed_source=getattr(authority, 'query_score_mode', 'mean_k'),
        propagation_applied=not result.diagnostics['identity'],
        pressure_policy=getattr(options,'pressure_policy','recency'),
        score_domain_version=domain.get('version', 'unknown'),
        score_mask_count=sum(domain.get('score_mask', ())),
        transport_mask_count=sum(bool(v) for v in eligible),
        locality_provenance=dict(getattr(snapshot, 'locality_provenance', ())))
    report.update(mode=options.graph_mode, budget=max_tokens or authority.memory.config.select_budget,
        revision=snapshot.revision, runtime_epoch=snapshot.epoch,
        branch_id=str(snapshot.branch_id), accepted_through=snapshot.accepted_through,
        eligible_blocks=sum(bool(value) for value in eligible),
        graph_bytes=graph.bytes, graph_live_bytes=graph.live_bytes,
        snapshot_bytes=snapshot.bytes, graph_global_bytes=graph.budget.used,
        selected_tokens=sum(r.count for r in records if r.handle.slot in selected), added_blocks=len(selected-old),
        displaced_blocks=len(old-selected), graph_selection_seconds=time.perf_counter()-started)
    if group_evidence is not None:
        report['query_group_selection'] = halo_summary(group_evidence)
        if quota_evidence is not None:
            report['query_group_selection']['quota_selection'] = quota_summary(quota_evidence)
            if not options.graph_trace:
                report['query_group_selection'] = {key:value for key,value in report['query_group_selection'].items()
                    if not key.endswith('_starts')}
            report['query_quota_selection'] = quota_summary(quota_evidence)
    if quota_propagation is not None:
        report['query_quota_propagation'] = quota_propagation
    if options.graph_trace:
        report['selected_starts'] = sorted(positions[slot] for slot in selected)[:256]
        report['baseline_starts'] = sorted(positions[slot] for slot in old)[:256]
        report['added_starts'] = sorted(positions[slot] for slot in selected-old)[:64]
        report['displaced_starts'] = sorted(positions[slot] for slot in old-selected)[:64]
        report['trace'] = [dict(item, selected=item['target'] in {positions[s] for s in selected},
            newly_selected=item['target'] in {positions[s] for s in selected-old}) for item in trace]
    # Serialization, trace quota and filesystem failures must happen before
    # beginning an assembly transaction that the caller does not yet own.
    emit_graph_report(authority, report)
    graph.validate_binding(binding)
    if group_evidence is not None:
        plan = authority.memory.select_groups(groups, group_scores, picker_pins,
            max_tokens=max_tokens, expected_slots=selected)
        authority.query_group_selection = report['query_group_selection']
        if quota_evidence is not None:
            authority.query_quota_selection = quota_summary(quota_evidence)
            authority.query_quota_propagation = quota_propagation
    else:
        plan = authority.memory.select(propagated, mandatory_slots,
            action=False, prefill_scores=None, max_tokens=max_tokens)
    authority.graph_selection_report = report
    authority._graph_selection_binding = binding
    # Retain only an epoch/revision token, never an unaccounted snapshot closure.
    # This guard covers both synchronous and asynchronous assembly publication.
    authority.memory._selection_validator = lambda: graph.validate_binding(binding)
    return plan


def emit_graph_report(authority, report):
    """Bound both the event count per owner and shared diagnostic file size."""
    count = getattr(authority, '_graph_report_count', 0)
    if count >= _MAX_EVENTS:
        return
    path = getattr(authority, 'graph_report_path', None)
    if path is None:
        return
    row = dict(report, authority_id=authority.memory.request_id,
        request_id=getattr(authority, 'graph_request_id', authority.memory.request_id))
    encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > _MAX_EVENT_BYTES:
        row.pop('trace', None)
        row['trace_omitted_byte_limit'] = True
        encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
    with _LOG_LOCK:
        if path.exists() and path.stat().st_size + len(encoded.encode()) + 1 > _MAX_LOG_BYTES:
            return
        with path.open('a') as stream:
            stream.write(encoded + '\n')
        authority._graph_report_count = count + 1


def pressure_score_and_select(authority, candidates, mandatory=(), *, max_tokens=None):
    """Propagate read-only accepted attention over the original working window."""
    from .authority import record_key, SCORE_DOMAIN_VERSION
    from .prefill_attention import pressure_seed
    from .graph_scoring import score_graph
    memory, graph = authority.memory, getattr(authority, 'block_graph', None)
    options = authority.graph_options
    if graph is None or getattr(options, 'graph_scoring_version', 'v1') != 'v2':
        raise ValueError('Graph pressure requires V2 owned graph state')
    if options.graph_mode not in ('locality', 'attention', 'multi'):
        raise ValueError('Graph pressure requires a scoring mode')
    started = time.perf_counter()
    with ExitStack() as scope:
        seen, starts = set(), []
        for slot, generation, begin, count, _ in candidates:
            if (slot in seen or type(count) is not int or count <= 0 or
                    begin < 0 or begin % memory.config.block_tokens or
                    begin + count > authority.history_end):
                raise ValueError('Invalid graph pressure candidate extent')
            handle = memory._store.handle(slot)
            record = authority.records.get((slot, generation))
            if (handle.generation != generation or record is None or
                    record.history_begin != begin or record.count != count):
                raise ValueError('Stale graph pressure candidate generation/count')
            lease = memory.repository.pin(handle)
            if not lease:
                raise ValueError('Graph pressure candidate is no longer live')
            scope.callback(lease.reset)
            seen.add(slot); starts.append(begin)
        if len(set(starts)) != len(starts):
            raise ValueError('Duplicate logical graph pressure candidate')
        for handle in mandatory:
            record = authority.records.get(record_key(handle))
            if record is None or memory._store.handle(handle.slot) != handle:
                raise ValueError('Stale graph pressure mandatory generation')
        raw_pairs, source = pressure_seed(authority, candidates)
        raw = [score for _, score in raw_pairs]
        slots = tuple(h.slot for h in mandatory)
        fixed = set(memory._store.fixed_selection_blocks(slots, pressure=True, budget=max_tokens or 0))
        mask = [slot not in fixed for slot, *_ in candidates]
        snapshot = scope.enter_context(query_snapshot(authority))
        binding = graph.binding
        if (snapshot.owner, snapshot.epoch, snapshot.branch_id, snapshot.revision,
                snapshot.accepted_through) != binding:
            raise RuntimeError('Pressure graph snapshot belongs to another revision')
        result = scope.enter_context(score_graph(raw, starts, mask, snapshot,
            **propagation_options(options), block_tokens=memory.config.block_tokens))
        # Keep every zero-score candidate: this vector constrains native
        # chronological fallback and must never reopen cold repository history.
        scores = [(row[0], score) for row, score in zip(candidates, result.scores)]
        baseline = set(memory.preview_selection({}, slots, action=False,
            pressure_scores=raw_pairs, max_tokens=max_tokens))
        expected = set(memory.preview_selection({}, slots, action=False,
            pressure_scores=scores, max_tokens=max_tokens))
        trace = ()
        positions = {row[0]:row[2] for row in candidates}
        if options.graph_trace:
            traced = scope.enter_context(score_graph(raw, starts, mask, snapshot,
                **propagation_options(options), block_tokens=memory.config.block_tokens,
                trace=True, trace_targets=[positions[s] for s in expected-baseline if s in positions]))
            trace = traced.trace
        graph.validate_binding(binding)
        record_pressure_inputs(authority, candidates, raw, mask, snapshot, slots,
            max_tokens, source, baseline)
        selected = expected
        report = dict(result.diagnostics, phase='pressure', seed_source=source,
            score_domain_version=SCORE_DOMAIN_VERSION, pressure_policy='graph',
            budget=max_tokens or memory.config.select_budget, candidate_count=len(candidates),
            score_mask_count=len(candidates), transport_mask_count=sum(mask),
            locality_provenance=dict(getattr(snapshot, 'locality_provenance', ())),
            selected_tokens=sum(authority.records[record_key(memory._store.handle(s))].count for s in selected),
            added_blocks=len(selected-baseline),
            displaced_blocks=len(baseline-selected), graph_selection_seconds=time.perf_counter()-started,
            graph_bytes=graph.bytes, graph_live_bytes=graph.live_bytes,
            graph_global_bytes=graph.budget.used, snapshot_bytes=snapshot.bytes,
            authority_accepted_boundary=authority.history_end,
            branch_id=str(snapshot.branch_id))
        if options.graph_trace:
            report.update(trace=trace, selected_starts=sorted(positions[s] for s in selected if s in positions),
                baseline_starts=sorted(positions[s] for s in baseline if s in positions))
        emit_graph_report(authority, report)
        graph.validate_binding(binding)
        plan = memory.select({}, slots, action=False, pressure_scores=scores, max_tokens=max_tokens)
        authority.graph_selection_report = report
        authority._graph_selection_binding = binding
        memory._selection_validator = lambda: graph.validate_binding(binding)
        return plan


def record_pressure_inputs(authority, candidates, raw, mask, snapshot, mandatory, budget, source, baseline):
    options = authority.graph_options
    path = getattr(authority, 'graph_report_path', None)
    if (not options.graph_trace or path is None or len(candidates)>_MAX_REPLAY_NODES or len(authority.records)>_MAX_REPLAY_NODES or
            getattr(authority,'_graph_input_count',0)>=8):
        return
    from dataclasses import asdict
    from .authority import SCORE_DOMAIN_VERSION
    cfg = authority.memory.config
    row = dict(version=2, phase='pressure', seed_source=source,
        scoring_version=options.graph_scoring_version, alpha=options.graph_alpha,
        aggregation=getattr(options, 'graph_aggregation', 'sum'),
        locality_weight=options.graph_locality_weight, attention_weight=options.graph_attention_weight,
        cross_message_weight=options.graph_cross_message_weight,
        score_domain_version=SCORE_DOMAIN_VERSION, pressure_policy='graph', mode=options.graph_mode,
        authority_id=authority.memory.request_id,
        request_id=getattr(authority,'graph_request_id',authority.memory.request_id),
        starts=[r[2] for r in candidates], counts=[r[3] for r in candidates],
        handles=[(r[0],r[1]) for r in candidates], candidate_records=candidates,
        raw_scores=raw, eligible=mask, transport_mask=mask, score_mask=[True]*len(raw),
        mandatory_starts=[r.history_begin for r in authority.records.values() if r.handle.slot in mandatory],
        baseline_selected_starts=[r.history_begin for r in authority.records.values() if r.handle.slot in baseline],
        accepted_boundary=authority.history_end, budget=budget or cfg.select_budget,
        config={name:getattr(cfg,name) for name in ('block_tokens','select_budget','gen_budget',
            'repository_blocks','sink_blocks','recent_blocks','action_attention','action_fraction','action_combine')},
        graph=dict(rows=[asdict(r) for r in snapshot.rows],
            partial=asdict(snapshot.partial) if snapshot.partial is not None else None,
            message_spans=snapshot.message_spans, message_end_known=snapshot.message_end_known,
            locality_provenance=dict(getattr(snapshot,'locality_provenance',())),
            accepted_through=snapshot.accepted_through, completed_through=snapshot.completed_through,
            revision=snapshot.revision),
        repository_records=[(r.handle.slot,r.handle.generation,r.history_begin,r.count)
            for r in authority.records.values()])
    encoded=json.dumps(row,separators=(',',':'),allow_nan=False)
    if len(encoded.encode())>_MAX_REPLAY_RECORD_BYTES:
        return
    path=path.with_name('block_graph_inputs.jsonl')
    with _LOG_LOCK:
        if path.exists() and path.stat().st_size+len(encoded.encode())+1>_MAX_LOG_BYTES:
            return
        with path.open('a') as stream:
            stream.write(encoded+'\n')
        authority._graph_input_count=getattr(authority,'_graph_input_count',0)+1


def record_query_inputs(authority, records, scores, eligible, snapshot, mandatory_slots, max_tokens,
                        *, group_selection=None, mean_scores=None, quota_propagated_scores=None):
    """Opt-in bounded replay evidence, never a per-decode graph dump."""
    options = getattr(authority, 'graph_options', None)
    if getattr(options, 'query_fps_halo_blocks', 0) and group_selection is None:
        # The halo branch exports once after native preview, before assembly.
        return
    path = getattr(authority, 'graph_report_path', None)
    overlap = getattr(authority, 'query_score_mode', 'mean_k') in ('overlap_16_8', 'key_fps_4')
    counter = '_window_query_input_count' if overlap else '_graph_input_count'
    if (snapshot is None or options is None or not options.graph_trace or path is None or
            getattr(authority, counter, 0) >= 8):
        return
    hybrid = bool(getattr(options, 'query_mean_quota_fraction', 0))
    node_limit = 32768 if hybrid else _MAX_REPLAY_NODES
    if len(records) > node_limit:
        if overlap:
            raise BufferError('Overlap Query trace exceeds its bounded node capacity')
        return
    from dataclasses import asdict
    cfg = authority.memory.config
    action = getattr(authority, 'action_history', None)
    prefill = getattr(action, 'prefill', None)
    row = dict(version=2, phase='query', seed_source=getattr(authority, 'query_score_mode', 'mean_k'),
        scoring_version=getattr(options, 'graph_scoring_version', 'v1'),
        alpha=getattr(options, 'graph_alpha', .2),
        aggregation=getattr(options, 'graph_aggregation', 'sum'),
        locality_weight=options.graph_locality_weight, attention_weight=options.graph_attention_weight,
        cross_message_weight=options.graph_cross_message_weight,
        accepted_boundary=authority.history_end,
        score_domain_version=getattr(authority, 'query_score_domain', {}).get('version', 'unknown'),
        score_mask=getattr(authority, 'query_score_domain', {}).get('score_mask', ()),
        transport_mask=eligible,
        handles=[(r.handle.slot,r.handle.generation) for r in records],
        authority_id=authority.memory.request_id,
        request_id=getattr(authority, 'graph_request_id', authority.memory.request_id),
        mode=options.graph_mode, starts=[r.history_begin for r in records],
        counts=[r.count for r in records], raw_scores=scores, eligible=eligible,
        mandatory_starts=[r.history_begin for r in records if r.handle.slot in mandatory_slots],
        budget=max_tokens or cfg.select_budget,
        config={name: getattr(cfg, name) for name in ('block_tokens', 'select_budget',
            'gen_budget', 'repository_blocks', 'sink_blocks', 'recent_blocks',
            'action_attention', 'action_fraction', 'action_combine')},
        graph=dict(rows=[asdict(item) for item in snapshot.rows],
            partial=asdict(snapshot.partial) if snapshot.partial is not None else None,
            message_spans=snapshot.message_spans, message_end_known=snapshot.message_end_known,
            locality_provenance=dict(getattr(snapshot,'locality_provenance',())),
            accepted_through=snapshot.accepted_through, completed_through=snapshot.completed_through,
            revision=snapshot.revision),
        legacy_action_scores=getattr(action, 'scores', {}),
        prefill_scores=prefill.scores() if prefill is not None and not action.samples else None)
    if mean_scores is not None:
        row['mean_scores'] = mean_scores
        row['query_mean_quota_fraction'] = options.query_mean_quota_fraction
    if quota_propagated_scores is not None:
        row['quota_propagated_scores'] = quota_propagated_scores
    if group_selection is not None:
        row['query_group_selection'] = group_selection
    if options.graph_mode in ('locality','attention','multi'):
        baseline=authority.memory.preview_selection({r.handle.slot:s for r,s in zip(records,scores)},
            mandatory_slots,action=False,max_tokens=max_tokens)
        row['baseline_selected_starts']=[r.history_begin for r in records if r.handle.slot in baseline]
    encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
    # The full 256K overlap qualification exports 8K nodes with four edges
    # each, plus original seeds and masks. Its public Query has a separate
    # counter so pressure diagnostics cannot consume this evidence quota.
    limit = (3 if group_selection is not None else 2) * _MAX_REPLAY_RECORD_BYTES if overlap else _MAX_REPLAY_RECORD_BYTES
    if hybrid:
        limit = 24*1024**2
    if len(encoded.encode()) > limit:
        if overlap:
            raise BufferError('Overlap Query trace exceeds its bounded record byte capacity')
        return
    path = path.with_name('block_graph_inputs.jsonl')
    log_limit = getattr(options, 'graph_query_trace_max_bytes', _MAX_LOG_BYTES)
    with _LOG_LOCK:
        if (path.stat().st_size if path.exists() else 0) + len(encoded.encode()) + 1 > log_limit:
            if overlap:
                raise BufferError('Overlap Query trace exceeds its bounded log byte capacity')
            return
        with path.open('a') as stream:
            stream.write(encoded + '\n')
        setattr(authority, counter, getattr(authority, counter, 0) + 1)
