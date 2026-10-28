"""Original FPS4/MeanK pickers with bounded message metadata and no graph.

Scores are already block-softmax/head/layer/token aggregates. This path neither
normalizes them again nor applies the transport mask to the seed vectors.
"""
import json
import threading
import time

from .query_group_recall import build_halo_groups, describe_halo_selection, halo_summary
from .query_hybrid_recall import preview_hybrid_selection, quota_summary


_LOG_LOCK = threading.Lock()


def _write_evidence(authority, report, trace):
    path = getattr(authority, 'query_report_path', getattr(authority, 'graph_report_path', None))
    if path is None:
        return
    report = dict(report, authority_id=authority.memory.request_id,
                  request_id=getattr(authority, 'graph_request_id', authority.memory.request_id))
    count = getattr(authority, '_no_graph_report_count', 0)
    encoded = json.dumps(report, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > 65536:
        raise BufferError('Graph-free Query report exceeds its bounded event capacity')
    report_path = path.with_name('query_recall.jsonl')
    input_path = path.with_name('query_recall_inputs.jsonl')
    inputs = None
    trace_count = getattr(authority, '_no_graph_input_count', 0)
    if trace is not None and trace_count < 8:
        if len(trace['starts']) > 32768:
            raise BufferError('Graph-free Query trace exceeds bounded node capacity')
        inputs = json.dumps(trace, separators=(',', ':'), allow_nan=False)
        if len(inputs.encode()) > 24 * 1024**2:
            raise BufferError('Graph-free Query trace exceeds bounded record capacity')
    with _LOG_LOCK:
        if inputs is not None:
            cap = getattr(authority.graph_options, 'graph_query_trace_max_bytes', 16 * 1024**2)
            if (input_path.stat().st_size if input_path.exists() else 0) + len(inputs.encode()) + 1 > cap:
                raise BufferError('Graph-free Query trace exceeds bounded file capacity')
            with input_path.open('a') as stream:
                stream.write(inputs + '\n')
            authority._no_graph_input_count = trace_count + 1
        if count < 256 and (not report_path.exists() or report_path.stat().st_size + len(encoded.encode()) + 1 <= 16 * 1024**2):
            with report_path.open('a') as stream:
                stream.write(encoded + '\n')
            authority._no_graph_report_count = count + 1


def select_without_graph(authority, records, raw_scores, transport_mask, mandatory_slots=(),
                         max_tokens=None, *, mean_scores=None):
    """Preview the unchanged native picker, publish evidence, then start assembly."""
    options = authority.graph_options
    if (getattr(authority, 'block_graph', None) is not None or
            getattr(options, 'graph_mode', None) != 'off' or getattr(options, 'graph_alpha', None) != 0 or
            getattr(authority, 'query_score_mode', None) != 'key_fps_4'):
        raise ValueError('Graph-free Query picker requires FPS4 with graph off and alpha zero')
    if len(raw_scores) != len(records) or len(transport_mask) != len(records):
        raise ValueError('Graph-free Query seeds and masks do not match the pinned index')
    boundaries = authority.query_boundaries
    started = time.perf_counter()
    with boundaries.snapshot() as snapshot:
        if snapshot.accepted_through < authority.history_end:
            raise RuntimeError('Query message boundaries have not reached accepted history')
        if (snapshot.accepted_through > authority.history_end and
                getattr(authority, '_query_boundary_validated_prefix', 0) < snapshot.accepted_through):
            raise RuntimeError('Future Query boundaries lack a verified accepted prefix')
        binding = snapshot.binding
        handles = tuple(record.handle for record in records)

        def validate():
            boundaries.validate_binding(binding)
            if any(current is None or current != handle
                   for handle in handles for current in (authority.memory._store.handle(handle.slot),)):
                raise RuntimeError('Graph-free Query index contains a recycled generation')

        validate()
        memory = authority.memory
        budget = memory.config.select_budget if max_tokens is None else max_tokens
        raw_by_slot = {record.handle.slot: score for record, score in zip(records, raw_scores)}
        baseline = set(memory.preview_selection(raw_by_slot, mandatory_slots,
            action=False, prefill_scores=None, max_tokens=max_tokens))
        fraction = getattr(options, 'query_mean_quota_fraction', 0)
        halo = getattr(options, 'query_fps_halo_blocks', 0)
        quota = groups = group_scores = group_evidence = None
        pins = mandatory_slots
        if fraction:
            if halo != 1 or mean_scores is None:
                raise ValueError('Graph-free hybrid quota requires MeanK seeds and a one-block halo')
            preview = preview_hybrid_selection(memory, records, raw_scores, mean_scores, mandatory_slots,
                max_tokens=max_tokens, mean_fraction=fraction, message_spans=snapshot.message_spans,
                message_metadata_present=snapshot.has_metadata)
            groups, group_scores, pins = preview['groups'], preview['group_scores'], preview['mandatory_slots']
            selected = set(preview['selected_slots'])
            group_evidence, quota = preview['group_selection'], preview['quota_selection']
        elif halo:
            groups, group_scores, metadata = build_halo_groups(records, raw_scores,
                radius=halo, message_spans=snapshot.message_spans,
                message_metadata_present=snapshot.has_metadata)
            selected = set(memory.preview_group_selection(groups, group_scores, mandatory_slots,
                max_tokens=max_tokens))
            fixed = memory._store.fixed_selection_blocks(list(mandatory_slots), pressure=False, budget=budget)
            group_evidence = describe_halo_selection(metadata, records, selected, baseline, fixed, budget)
        else:
            selected = baseline
        domain = getattr(authority, 'query_score_domain', {})
        report = dict(phase='query', mode='off', selection_backend='native_query_no_graph',
            metadata_kind='message_boundaries_only', propagation_applied=False,
            seed_source='key_fps_4', score_domain_version=domain.get('version', 'unknown'),
            score_mask_count=sum(domain.get('score_mask', ())),
            transport_mask_count=sum(bool(x) for x in transport_mask), budget=budget,
            boundary_revision=snapshot.revision, boundary_epoch=snapshot.epoch,
            accepted_through=snapshot.accepted_through, boundary_bytes=snapshot.bytes,
            boundary_spans=len(snapshot.message_spans), message_metadata_present=snapshot.has_metadata,
            selected_tokens=sum(r.count for r in records if r.handle.slot in selected),
            selected_blocks=len(selected), added_blocks=len(selected-baseline),
            displaced_blocks=len(baseline-selected),
            query_selection_seconds=time.perf_counter()-started)
        if group_evidence is not None:
            report['query_group_selection'] = halo_summary(group_evidence)
            # Keep high-cardinality evidence in the separately bounded trace.
            if not getattr(options, 'graph_trace', False):
                report['query_group_selection'] = {k: v for k, v in report['query_group_selection'].items()
                                                    if not k.endswith('_starts')}
            if quota is not None:
                report['query_quota_selection'] = quota_summary(quota)
                report['query_group_selection']['quota_selection'] = quota_summary(quota)
        trace = None
        if getattr(options, 'graph_trace', False):
            trace = dict(version=1, metadata_kind='message_boundaries_only', propagation_applied=False,
                mode='off', starts=[r.history_begin for r in records], counts=[r.count for r in records],
                handles=[(h.slot, h.generation) for h in handles], raw_scores=list(raw_scores),
                mean_scores=None if mean_scores is None else list(mean_scores),
                score_mask=domain.get('score_mask', ()), transport_mask=list(transport_mask),
                message_spans=snapshot.message_spans, message_end_known=snapshot.message_end_known,
                message_metadata_present=snapshot.has_metadata,
                mandatory_starts=[r.history_begin for r in records if r.handle.slot in mandatory_slots],
                selected_starts=[r.history_begin for r in records if r.handle.slot in selected],
                budget=budget, boundary_revision=snapshot.revision,
                query_mean_quota_fraction=fraction, query_group_selection=group_evidence)
        _write_evidence(authority, report, trace)
        validate()
        if groups is not None:
            plan = memory.select_groups(groups, group_scores, pins, max_tokens=max_tokens, expected_slots=selected)
            authority.query_group_selection = report['query_group_selection']
        else:
            plan = memory.select(raw_by_slot, mandatory_slots, action=False,
                                 prefill_scores=None, max_tokens=max_tokens)
        if quota is not None:
            authority.query_quota_selection = quota_summary(quota)
        authority.query_quota_propagation = None
        authority.query_selection_report = report
        selected_handles = tuple(handle for handle in handles if handle.slot in selected)

        def validate_selected():
            boundaries.validate_binding(binding)
            # The score snapshot may release unselected generations after
            # preview. Their subsequent GC/reuse does not invalidate a plan
            # which already owns every selected generation independently.
            if any(current is None or current != handle
                   for handle in selected_handles for current in (memory._store.handle(handle.slot),)):
                raise RuntimeError('Graph-free selected Query page has a recycled generation')

        memory._selection_validator = validate_selected
        return plan
