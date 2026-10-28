"""Opt-in, uniform block scoring for the fixed-policy recall experiment.

This module changes scores only. Native Query/pressure candidate domains,
mandatory blocks, budget accounting and assembly remain authoritative.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
import time

import torch


METHODS = ('mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass')
_LOG_LOCK = threading.Lock()


def configured_method():
    method = os.getenv('KVMEM_RECALL_METHOD', '')
    if method and method not in METHODS:
        raise ValueError('Invalid KVMEM_RECALL_METHOD')
    if method and os.getenv('KVMEM_PRESSURE_QUERY_WINDOW', '128') != '128':
        raise ValueError('Recall comparison requires a fixed 128-token pressure Q window')
    return method


def validate_options(options, method=None):
    if not (configured_method() if method is None else method):
        return
    forbidden = ('guided_query_tokens', 'query_history', 'query_fps_halo_blocks',
                 'query_mean_quota_fraction', 'query_fps_prebuild')
    if any(getattr(options, name, 0) for name in forbidden):
        raise ValueError('Recall experiment requires guided/history/halo/hybrid/prebuild disabled')
    if (getattr(options, 'graph_mode', 'off') != 'off' or
            getattr(options, 'graph_alpha', 0) != 0):
        raise ValueError('Recall experiment requires graph disabled')
    if getattr(options, 'query_score_mode', 'mean_k') != 'mean_k':
        raise ValueError('Recall experiment owns scoring; legacy query_score_mode must be mean_k')


def launch_environment(config, options):
    if config is None:
        return {}
    allowed = {'method', 'pressure_query_window', 'device', 'query_batch', 'block_batch'}
    if not isinstance(config, dict) or set(config) - allowed or config.get('method') not in METHODS:
        raise ValueError('Invalid recall_experiment profile')
    if type(config.get('pressure_query_window')) is not int or config['pressure_query_window'] != 128:
        raise ValueError('recall_experiment requires pressure_query_window=128')
    device, queries, blocks = config.get('device', 'cpu'), config.get('query_batch', 4), config.get('block_batch', 32)
    if (device not in ('cpu', 'cuda') or type(queries) is not int or not 1 <= queries <= 16 or
            type(blocks) is not int or not 1 <= blocks <= 128):
        raise ValueError('Invalid recall_experiment scratch geometry/device')
    validate_options(options, method=config['method'])
    if not options.request_planning or not options.async_refresh:
        raise ValueError('Recall experiment requires planned asynchronous reselection')
    if options.memory.block_tokens != 32:
        raise ValueError('This recall campaign freezes one 32-token physical block')
    if getattr(options, 'historical_gc', None) is not None:
        raise ValueError('Recall comparison requires complete history without historical GC')
    return {'KVMEM_RECALL_METHOD': config['method'], 'KVMEM_PRESSURE_QUERY_WINDOW': '128',
            'KVMEM_RECALL_DEVICE': device, 'KVMEM_RECALL_QUERY_BATCH': str(queries),
            'KVMEM_RECALL_BLOCK_BATCH': str(blocks)}


def _signature(record):
    return (record.handle.slot, record.handle.generation, record.history_begin,
            record.count, id(record))


def _features(authority, records, method):
    from .recall_scoring import build_block_features
    cache = getattr(authority, '_recall_feature_cache', None)
    if not cache or cache.get('method') != method:
        cache = dict(method=method, entries={})
        authority._recall_feature_cache = cache
    entries = cache['entries']
    live = {_signature(record) for record in authority.records.values()}
    for identity in tuple(entries):
        if identity not in live:
            del entries[identity]
    missing = [record for record in records if _signature(record) not in entries]
    layers = authority.scoring_layers
    for begin in range(0, len(missing), 16):
        group = missing[begin:begin + 16]
        archived = authority.read_records([record.handle for record in group])
        keys = torch.zeros((layers, len(group), *authority.shape[1:]), dtype=authority.raw_dtype)
        for index, (record, (key, _)) in enumerate(zip(group, archived)):
            if key.shape != (authority.shape[0], record.count, *authority.shape[2:]):
                raise ValueError('Recall index raw K geometry differs from accepted record')
            keys[:, index, :record.count] = key[:layers]
        features = build_block_features(keys, [r.count for r in group], method)
        for index, record in enumerate(group):
            entries[_signature(record)] = features.select_blocks([index])
    return [entries[_signature(record)] for record in records]


def score_records(authority, records, queries, score_mask, method):
    """Normalize each Q over the complete candidate domain, then aggregate.

Q batches bound scratch without changing the softmax domain. Raw token logits
are reduced to logsumexp per block for ExactMass before global normalization.
"""
    from .recall_scoring import BlockFeatures, block_logits, concat_block_features
    if (queries is None or queries.ndim != 4 or not records or
            len(score_mask) != len(records)):
        raise ValueError('Recall scoring requires Q and a nonempty valid candidate domain')
    if queries.shape[0] != authority.scoring_layers or queries.shape[1] < 1:
        raise ValueError('Recall Q layers/rows do not match the frozen scoring geometry')
    if not any(score_mask):
        return [0.] * len(records), dict(query_rows=queries.shape[1], score_sum=0.,
            normalization='empty_eligible_domain', no_eligible_candidates=True,
            scoring_seconds=0., feature_seconds=0.)
    device_name = os.getenv('KVMEM_RECALL_DEVICE', 'cpu')
    if device_name not in ('cpu', 'cuda'):
        raise ValueError('KVMEM_RECALL_DEVICE must be cpu or cuda')
    device = torch.device(device_name)
    # The original CUDA MeanK/FPS contract first rounds captured Q to FP16.
    # Apply the same conversion to every arm, including the exact reference.
    queries = queries.to(dtype=torch.float16) if authority.mean_k_backend == 'qw3_cuda' else queries
    query_batch = int(os.getenv('KVMEM_RECALL_QUERY_BATCH', '4'))
    block_batch = int(os.getenv('KVMEM_RECALL_BLOCK_BATCH', '32'))
    if not 1 <= query_batch <= 16 or not 1 <= block_batch <= 128:
        raise ValueError('Recall scratch tiles exceed the qualified bounds')
    began = time.perf_counter()
    if method == 'mean_k':
        # Preserve the original published MeanK index, including its precision.
        values = torch.stack([r.mean_key if r.mean_key is not None else
            (r.key_sum / r.count).to(torch.float16 if authority.mean_k_backend == 'qw3_cuda'
                                    else torch.float32) for r in records], dim=1)
        values = values[:authority.scoring_layers].float().unsqueeze(2)
        features = BlockFeatures(method=method, values=values,
            valid=torch.ones(values.shape[:-1], dtype=torch.bool),
            counts=torch.tensor([r.count for r in records], dtype=torch.int64))
        feature_parts = [features.select_blocks(list(range(i, min(i + block_batch, len(records)))))
                         for i in range(0, len(records), block_batch)]
    else:
        per_block = _features(authority, records, method)
        feature_parts = [concat_block_features(per_block[i:i + block_batch])
                         for i in range(0, len(records), block_batch)]
    built = time.perf_counter()
    mask = torch.tensor(score_mask, dtype=torch.bool, device=device)
    totals = torch.zeros(len(records), dtype=torch.float64, device=device)
    for begin in range(0, queries.shape[1], query_batch):
        query = queries[:, begin:begin + query_batch].to(device=device, dtype=torch.float32)
        logits = torch.empty((query.shape[0], query.shape[1], query.shape[2], len(records)),
                             device=device, dtype=torch.float32)
        offset = 0
        for part in feature_parts:
            local = part.to(device)
            count = local.values.shape[1]
            logits[..., offset:offset + count] = block_logits(query, local)
            offset += count
            del local
        logits.masked_fill_(~mask, -torch.inf)
        probabilities = logits.softmax(-1)
        totals += probabilities.sum((0, 1, 2), dtype=torch.float64)
        del logits, probabilities, query
    totals /= queries.shape[0] * queries.shape[2]
    if not bool(torch.isfinite(totals).all()) or abs(float(totals.sum()) - queries.shape[1]) > 1e-3:
        raise ValueError('Recall score normalization failed')
    scores = totals.cpu().tolist()
    return scores, dict(scoring_device=device_name, query_batch=query_batch, block_batch=block_batch,
        feature_seconds=built-began, scoring_seconds=time.perf_counter()-built,
        query_rows=queries.shape[1], score_sum=sum(scores),
        query_reduction='sum', head_layer_reduction='mean', key_frame='content',
        query_precision='fp16_to_fp32' if authority.mean_k_backend == 'qw3_cuda' else 'fp32',
        normalization='global_candidate_token_softmax_block_sum' if method == 'exact_mass'
                      else 'global_candidate_block_softmax')


def _write_event(output, event):
    path = Path(output) / 'recall_events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(',', ':'), allow_nan=False)
    with _LOG_LOCK, path.open('a') as stream:
        stream.write(line + '\n')


def select_for_event(authority, queries, mandatory, *, max_tokens, event_type,
                     query_metadata, request_id, output):
    from .authority import record_key, query_score_frame
    method = configured_method()
    if not method or event_type not in ('query', 'prefill_pressure', 'decode_pressure'):
        raise ValueError('Invalid recall experiment event')
    authority._require_open()
    validate_options(authority.graph_options)
    pressure = event_type != 'query'
    with authority.index_snapshot() as (all_records, _):
        pins = tuple(handle.slot for handle in mandatory)
        if pressure:
            working = tuple(authority.memory._store.working_records())
            by_identity = {record_key(record.handle): record for record in all_records}
            records = []
            for slot, generation, begin, count, _ in working:
                record = by_identity.get((slot, generation))
                if record is None or (record.history_begin, record.count) != (begin, count):
                    raise ValueError('Pressure candidates do not match accepted working records')
                records.append(record)
            frame_scope = nullcontext(None)
        else:
            records = all_records
            frame_scope = query_score_frame(authority.memory, records, pins, max_tokens)
        if not set(record_key(handle) for handle in mandatory).issubset(
                record_key(record.handle) for record in records):
            raise ValueError('Required block outside original recall candidate domain')
        if any(record.history_begin + record.count > authority.history_end for record in records):
            raise ValueError('Recall candidate extends beyond the accepted history')
        if (pressure and (query_metadata['query_end'] != authority.history_end or
                query_metadata['query_begin'] != max(0, authority.history_end - 128) or
                queries.shape[1] != min(128, authority.history_end) or
                query_metadata.get('query_rows', queries.shape[1]) != queries.shape[1])):
            raise ValueError('Pressure Q is not the complete accepted 128-token tail')
        if not pressure and (queries is None or queries.shape[1] !=
                query_metadata['query_end'] - query_metadata['query_begin']):
            raise ValueError('Public Query Q rows differ from its declared input span')
        with frame_scope as frame:
            mask = [True] * len(records) if pressure else tuple(frame.score_mask)
            scores, stats = score_records(authority, records, queries, mask, method)
        by_slot = {record.handle.slot: value for record, value in zip(records, scores)}
        kwargs = dict(action=False, prefill_scores=None, max_tokens=max_tokens)
        if pressure:
            kwargs['pressure_scores'] = list(by_slot.items())
            selection_scores = {}
        else:
            selection_scores = by_slot
        selected = set(authority.memory.preview_selection(selection_scores, pins, **kwargs))
        if not selected.issubset(by_slot):
            raise ValueError('Native picker escaped the frozen candidate set')
        selected_records = [r for r in records if r.handle.slot in selected]
        event = dict(version=1, sample_id=os.getenv('KVMEM_RECALL_SAMPLE_ID', request_id),
            request_id=request_id, authority_id=authority.memory.request_id,
            event_type=event_type, configured_method=method, actual_method=method,
            fallback_reason=None, accepted_through=authority.history_end,
            candidate_block_count=len(records), score_mask_count=sum(mask),
            candidate_domain='working_records' if pressure else 'accepted_history',
            selected_block_ids=[r.handle.slot for r in selected_records],
            selected_block_generations=[r.handle.generation for r in selected_records],
            selected_block_starts=[r.history_begin for r in selected_records],
            selected_tokens=sum(r.count for r in selected_records),
            charged_tokens=len(selected_records) * authority.shape[1],
            budget=max_tokens, block_tokens=authority.shape[1], pressure_query_window=128)
        event.update(query_metadata)
        event.update(stats)
        _write_event(output, event)
        expected = tuple(record.handle for record in selected_records)
        plan = authority.memory.select(selection_scores, pins, **kwargs)

        def validate_selected():
            if any(authority.memory._store.handle(h.slot) != h for h in expected):
                raise RuntimeError('Recall selection contains a recycled generation')

        validate_selected()
        authority.memory._selection_validator = validate_selected
        authority.recall_selection_report = event
        return plan
