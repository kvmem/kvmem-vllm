"""Explicit, synchronous retrieval experiments; never enabled by a profile.

The observer receives no answer, target offset, or benchmark annotation. Native
attention uses the actual FP8 working cache, its scale and dense RoPE positions.
Archived content-frame Q/K are saved separately for operand-controlled replays.
This deliberately expensive path is unsuitable for throughput measurements.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def enabled():
    return os.environ.get('KVMEM_RETRIEVAL_DIAGNOSTICS') == '1'


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def capture_native(refresh, job, query):
    if job.command.selection != 'query' or query is None:
        return
    if job.guided_rows or job.command.query_span is None:
        raise ValueError('Retrieval diagnostics require the original explicit Query')
    from .action_scoring import page_attention
    from .codec import native_vllm_fp8_views
    runner = refresh.runner
    root = runner._kvmem_probe_output / 'retrieval_diagnostics'
    root.mkdir(exist_ok=True)
    directory = root / job.command.request_id
    directory.mkdir(exist_ok=False)
    job.retrieval_diagnostic_path = directory
    authority = job.authority
    records = list(authority.memory._store.working_records())
    records.sort(key=lambda r:r[4])
    offset = runner._kvmem_window_adapter.offsets.get(job.command.request_id, 0)
    local_history = job.history - offset
    cursor = 0
    segments = []
    for slot, generation, begin, count, local in records:
        if local != cursor:
            raise ValueError('Diagnostic working segments do not form a complete partition')
        segments.append([begin, count, local])
        cursor += count
    if cursor != local_history:
        raise ValueError('Diagnostic history differs from actual native window')
    for begin in range(0, len(job.command.input_tokens), 32):
        count = min(32, len(job.command.input_tokens)-begin)
        segments.append([job.history+begin, count, local_history+begin])
    length = local_history + len(job.command.input_tokens)
    qb, qe = job.command.query_span
    if query.shape[1] != qe-qb:
        raise ValueError('Diagnostic lost an original Query row')
    positions = torch.arange(qb-offset, qe-offset, device=query.device, dtype=torch.int64)
    starts = torch.tensor([s[2] for s in segments], device=query.device, dtype=torch.int32)
    counts = torch.tensor([s[1] for s in segments], device=query.device, dtype=torch.int32)
    modules = dict(runner.model.named_modules())
    groups = {name:i for i,g in enumerate(runner.kv_cache_config.kv_cache_groups) for name in g.layer_names}
    index = runner.req_states.req_id_to_index[job.command.request_id]
    tables = {}
    masses, scales, codecs = [], [], []
    began = time.perf_counter()
    for layer_index, name in enumerate(runner._kvmem_layers):
        module = modules[name]
        attn = module.attn
        group = groups[attn.layer_name]
        if group not in tables:
            from .native_pages import kernel_table
            tables[group] = kernel_table(runner, group, index).clone()
        key, _ = native_vllm_fp8_views(attn.kv_cache, module.num_kv_heads, module.head_dim)
        q_dtype = refresh.runtime.action.prefill_q_dtypes[attn.layer_name]
        masses.append(page_attention(query[layer_index], positions, key, tables[group],
            module.rotary_emb.cos_sin_cache, attn._k_scale, starts, counts, length,
            scale=attn.impl.scale, q_dtype=q_dtype, q_scale=attn._q_scale,
            block_tokens=32, fused=True, complete_partition=True).cpu().numpy())
        scales.append(float(attn._k_scale.item()))
        codecs.append(str(q_dtype))
    np.save(directory/'native_attention.npy', np.stack(masses), allow_pickle=False)
    torch.save(query.detach().cpu(), directory/'queries.pt')
    _save(directory/'native.json', dict(request_id=job.command.request_id,
        history=job.history, query_span=[qb,qe], dense_query_positions=positions.cpu().tolist(),
        native_length=length, segments=segments, layers=list(runner._kvmem_layers),
        key_scales=scales, query_codecs=codecs, query_shape=list(query.shape),
        seconds=time.perf_counter()-began,
        scope='Reconstructed sampled attention over actual native FP8 cache; full causal softmax, Q-head mean. Not content-frame retrieval scores.',
        limits=['The diagnostic reduction need not be bitwise identical to FlashInfer.',
                'Pages absent from the active window have no native attention observation.']))


def capture_archive(refresh, job, query):
    directory = getattr(job, 'retrieval_diagnostic_path', None)
    if directory is None:
        return
    from .authority import query_score_frame
    from .cuda_scoring import mean_key_scores
    a = job.authority
    began = time.perf_counter()
    with a.index_snapshot() as (records, means):
        means = means[:a.scoring_layers]
        # A fixed 36K picker is evaluated even in the full-prefill arm. The
        # full arm itself keeps the whole prefix; this preview never commits.
        budget = 36864
        fit = refresh.options.memory.sink_blocks
        mandatory = tuple(r.handle.slot for r in records if r.history_begin < fit*32)
        with query_score_frame(a.memory, records, mandatory, budget) as frame:
            score_mask = list(frame.score_mask)
            transport_mask = list(frame.transport_mask)
            scores = mean_key_scores(query, means.to(query.device),
                torch.tensor(score_mask, device=query.device, dtype=torch.bool)).cpu().tolist()
        slots = a.memory.preview_selection({r.handle.slot:s for r,s in zip(records,scores)},
            mandatory, action=False, prefill_scores=None, max_tokens=budget)
        selected = set(slots)
        torch.save(means.detach().cpu(), directory/'means.pt')
        layers,_,heads,dim = means.shape
        raw = np.lib.format.open_memmap(directory/'raw_keys.npy', mode='w+', dtype=np.uint8,
                                        shape=(layers,job.history,heads,dim))
        cursor = 0
        for begin in range(0,len(records),128):
            group = records[begin:begin+128]
            tensors = a.read_records([r.handle for r in group])
            for record, (key, _) in zip(group,tensors):
                if record.history_begin != cursor or key.dtype != torch.float8_e4m3fn:
                    raise ValueError('Diagnostic archive requires complete original FP8 history')
                raw[:,cursor:cursor+record.count] = key[:layers].view(torch.uint8).numpy()
                cursor += record.count
        if cursor != job.history:
            raise ValueError('Diagnostic archive has a missing accepted record')
        raw.flush()
        del raw
        _save(directory/'archive.json', dict(history=job.history, dtype='float8_e4m3fn', raw_k_scale=1.0,
            layers=list(refresh.runner._kvmem_layers), starts=[r.history_begin for r in records],
            counts=[r.count for r in records], score_mask=score_mask, transport_mask=transport_mask,
            mean_scores=scores, budget=budget, selected_starts=[r.history_begin for r in records if r.handle.slot in selected],
            key_shape=[layers,job.history,heads,dim], seconds=time.perf_counter()-began,
            scope='All original accepted pre-RoPE archived K; no target positions or gold were provided to the observer.'))
