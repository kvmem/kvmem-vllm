"""Offline profiling of frozen real-agent windows; never submits model work."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

if __package__:
    from .analyze_deepswe_capacity import distribution, metrics, records
else:
    from analyze_deepswe_capacity import distribution, metrics, records


def native_geometry(body):
    lines = [line for line in body.splitlines() if line.startswith('vllm:cache_config_info{')]
    if len(lines) != 1:
        return None
    labels = dict(re.findall(r'(\w+)="([^"\\]*)"', lines[0]))
    fields = ('block_size', 'num_gpu_blocks', 'kv_cache_size_tokens')
    if not all(labels.get(f, '').isdigit() for f in fields):
        return None
    result = {f: int(labels[f]) for f in fields}
    if min(result.values()) <= 0:
        return None
    result['mamba_cache_mode'] = labels.get('mamba_cache_mode')
    result['cache_dtype'] = labels.get('cache_dtype')
    return result


def capacity_overlap(starts, finishes, samples, steps, offset, workers, geometry, end):
    """Match pure public KVMem cohorts to an outer HTTP request envelope."""
    if geometry is None:
        return dict(available=False, reason='Native physical pool geometry not recorded')
    completed = {(r['task'], r['turn']): r['at'] for r in finishes}
    edges = []
    seen = set()
    for r in starts:
        key = r['task'], r['turn']
        if key in seen:
            raise ValueError('Duplicate HTTP start in capacity proof')
        seen.add(key)
        # This must be a superset of possible live HTTP requests. Shrinking
        # envelopes could replace a real near-boundary request with an old
        # unfinished one and falsely identify a full cohort by cardinality.
        left, right = r['at']-.02, completed.get(key, end+1)+.02
        if left >= right:
            continue
        edges.extend([(left, 1, key, r), (right, -1, key, r)])
    edges.sort(key=lambda e: e[0])
    points = [(s['at'], None) for s in samples]
    points += [(r['started_at']-offset, r) for r in steps
               if r.get('decode_requests') == workers and not r.get('prefill_tokens')
               and not r.get('prefill_requests')]
    points.sort(key=lambda p: p[0])
    active, index, peak, best = {}, 0, None, None
    matches = nominal = hard = 0
    for at, step in points:
        while index < len(edges) and edges[index][0] <= at:
            _, kind, key, row = edges[index]
            if kind == 1:
                active[key] = row
            else:
                active.pop(key, None)
            index += 1
        values = list(active.values())
        total = sum(r['prompt_tokens'] for r in values)
        # At least one full-attention group exists in this frozen hybrid model.
        # Ignore all recurrent/MTP extra blocks: exceeding the entire native
        # shared pool on attention pages alone is a conservative hard bound.
        block = geometry['block_size']
        pages = sum((r['prompt_tokens']+block-1)//block for r in values)
        proof = dict(at=at, users=len(values), full_prompt_tokens=total,
            native_attention_pages_lower_bound=pages,
            views=[dict(task=r['task'], turn=r['turn'], prompt_tokens=r['prompt_tokens']) for r in values])
        if peak is None or total > peak['full_prompt_tokens']:
            peak = proof
        if step is None or len(values) != workers or len({r['task'] for r in values}) != workers:
            continue
        matches += 1
        nominal += total > geometry['kv_cache_size_tokens']
        hard += pages > geometry['num_gpu_blocks']
        proof = dict(proof, model_step_sequence=step['sequence'],
                     maximum_active_context_bucket=step.get('decode_context_bucket'))
        if best is None or total > best['full_prompt_tokens']:
            best = proof
    return dict(available=True, native_geometry=geometry,
        peak_http_demand=peak, matched_target_pure_decode_steps=matches,
        steps_above_reported_fullcontext_equivalent=nominal,
        steps_attention_alone_exceeds_native_physical_pool=hard, best_matched_cohort=best,
        scope='Outstanding HTTP demand alone is not GPU concurrency. A matched KVMem pure public step has exactly the configured number of decode rows and the same number of distinct possible agent HTTP envelopes; expand each envelope by20ms, rejecting extra unfinished candidates. Assumes only the frozen bridge plus one-token cleanup requests use this owned server; cleanup requests have no decode step. Each agent has an independent cache_salt, so full-context pages cannot be shared between users. Full history lengths exclude current generated tokens. Reported native token capacity is a full-context equivalent, not a hard mixed-length limit; attention pages exceeding the entire shared pool is the stricter test, ignoring recurrent/MTP overhead. This is not an individual256K-per-user peak or CPU byte oracle.')


def gpu_demand(samples, begins, finishes, end):
    """Sampled low utilization vs outstanding HTTP demand, not CUDA idle time."""
    edges = sorted([(r['at'], 1) for r in begins] + [(r['at'], -1) for r in finishes])
    groups = defaultdict(list)
    index = outstanding = 0
    for sample in samples:
        at = sample['at']
        if at > end or not sample.get('gpu'):
            continue
        while index < len(edges) and edges[index][0] <= at:
            outstanding += edges[index][1]
            index += 1
        if outstanding < 0:
            raise ValueError('HTTP completion without a corresponding start')
        _, util = map(float, sample['gpu'].split(','))
        gauges = metrics(sample.get('raw_metrics', ''))
        key = ('outstanding_model_http' if outstanding else 'no_outstanding_model_http',
               'below10percent' if util < 10 else 'at_least10percent')
        groups[key].append(dict(util=util, outstanding=outstanding,
            running=gauges.get('vllm:num_requests_running'),
            waiting=gauges.get('vllm:num_requests_waiting')))
    return dict(groups=[dict(http_state=k[0], gpu_utilization_band=k[1],
        samples=len(v), utilization=distribution([x['util'] for x in v]),
        outstanding_http=distribution([x['outstanding'] for x in v]),
        native_running=distribution([x['running'] for x in v]),
        native_waiting=distribution([x['waiting'] for x in v]))
        for k, v in sorted(groups.items())],
        scope='One-second sampled utilization and bridge HTTP demand. No outstanding HTTP may include tool execution, verification, setup or client preparation. Outstanding demand includes queues and CPU preparation; this does not prove GPU-ready work or exact CUDA idle seconds.')


def step_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        prefill, decode = r['prefill_tokens'], r['decode_requests']
        kind = 'mixed' if prefill and decode else 'prefill' if prefill else 'decode'
        key = kind, decode, r.get('decode_context_bucket'), r.get('prefill_context_bucket')
        groups[key].append(r)
    fields = ('step_ms', 'model_ms', 'maintenance_ms', 'maintenance_host_ms', 'host_submit_ms')
    return [dict(kind=k[0], decode_requests=k[1], maximum_decode_context_bucket=k[2],
        maximum_prefill_context_bucket=k[3], steps=len(v),
        prefill_tokens=sum(r['prefill_tokens'] for r in v),
        durations={field: distribution([r.get(field) for r in v]) for field in fields})
        for k, v in sorted(groups.items(), key=lambda pair: repr(pair[0]))]


def restore_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get('cancelled'):
            continue
        groups[r.get('lane', 'unknown')].append(r)
    return [dict(lane=lane, count=len(v),
        seconds=distribution([r['finished']-r['started'] for r in v]),
        host_prepare_seconds=distribution([r.get('host_prepare_seconds') for r in v]),
        uploaded_bytes=sum(r.get('uploaded_bytes', 0) for r in v),
        recurrent_model_bytes=sum(r.get('model_bytes', 0) for r in v),
        resident_active_tokens=sum(r.get('resident_active_tokens', 0) for r in v),
        resident_v_reused_tokens=sum(r.get('resident_v_reused_tokens', 0) for r in v))
        for lane, v in sorted(groups.items())]


def checkpoint_profile(rows):
    stages = sorted({key for row in rows for key in row.get('host_stage_seconds', {})})
    return dict(events=len(rows), status_counts={
        status: sum(r.get('status') == status for r in rows)
        for status in sorted({r['status'] for r in rows})}, durations={
        f: distribution([r.get(f) for r in rows])
        for f in ('host_sync_wait_seconds', 'host_total_seconds')},
        host_stages={f: distribution([r.get('host_stage_seconds', {}).get(f) for r in rows])
                     for f in stages})


def history_matched_requests(rows):
    groups = defaultdict(list)
    for r in rows:
        n = r['prompt_tokens']
        bucket = ('<=64K' if n <= 65536 else '64K-96K' if n <= 98304 else
                  '96K-128K' if n <= 131072 else '128K-192K' if n <= 196608 else '192K-256K')
        groups[bucket].append(r)
    result = []
    for bucket, values in sorted(groups.items()):
        durations = [(r.get('metrics') or {}).get('generation_time_ms') for r in values]
        measured = [(r, d) for r, d in zip(values, durations) if d is not None and d > 0]
        seconds = sum(d for _, d in measured)/1000
        generation = sum(max(0, r['usage']['completion_tokens']-1) for r, _ in measured)
        result.append(dict(prompt_length_bucket=bucket, completed_model_turns=len(values),
            prompt_tokens=distribution([r['prompt_tokens'] for r in values]),
            new_input_tokens=distribution([r.get('new_input_tokens') for r in values]),
            completion_tokens=distribution([r['usage']['completion_tokens'] for r in values]),
            native_ttft_ms=distribution([(r.get('metrics') or {}).get('time_to_first_token_ms') for r in values]),
            native_queue_ms=distribution([(r.get('metrics') or {}).get('queue_time_ms') for r in values]),
            summed_decoder_seconds=seconds, tokens_after_first=generation,
            weighted_per_decoder_tokens_per_second=generation/seconds if seconds else None))
    return dict(buckets=result, scope='Completed requests grouped by full prompt history length. Decode durations can overlap; rate is per decoder, not aggregate throughput. Matching history bins does not match tool text, generated lengths or trajectories, and incomplete requests are omitted.')


def confirmed_public_work(rows, begin, end):
    before = [r for r in rows if r['wall_time'] <= begin]
    during = [r for r in rows if begin < r['wall_time'] <= end]
    if not before or not during:
        return dict(available=False)
    first, last = before[-1], during[-1]
    if first['counter_id'] != last['counter_id']:
        raise ValueError('Public work counter instance changed')
    fields = ('public_batches', 'public_prefill_tokens', 'public_non_prefill_tokens',
              'prefill_request_batches', 'decode_request_batches')
    # A completion fence may still lag at the boundaries. Keep its uncertainty
    # rather than attributing warmup work or unfinished forwards to the window.
    bounds = {}
    for field in fields:
        lower = max(0, last['completed'][field]-first['submitted'][field])
        upper = last['submitted'][field]-first['completed'][field]
        if upper < lower:
            raise ValueError('Public work counters moved backwards')
        bounds[field] = dict(lower=lower, upper=upper)
    return dict(available=True, first_snapshot_at=first['wall_time'],
        last_snapshot_at=last['wall_time'], bounds=bounds,
        scope='Nonblocking completion brackets between these snapshot times, including intermediate prefill chunks and any repeated public prefill. Non-prefill work includes MTP verification, not accepted output. Private Query forwards and restored KV are excluded; the final unsampled tail is excluded.')


def output_reservations(root, starts):
    counts = defaultdict(int)
    missing = 0
    for r in starts:
        path = root/'bridge'/r['task']/f"{r['turn']:05d}"/'plan.json'
        try:
            plan = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            missing += 1
            continue
        counts[plan['max_tokens']] += 1
    profile = json.loads((root/'profile.json').read_text())
    kv = profile['kvmem']
    capacity = kv['select_budget']+kv['gen_budget']
    block = kv['block_tokens']
    return dict(started_requests=len(starts), missing_plan_files=missing,
        requested_output_tokens=[dict(tokens=n, requests=c,
            available_above_select_budget=capacity-kv['select_budget']-n-block)
            for n, c in sorted(counts.items())],
        scope='KVMem headroom above the selected history budget after reserving the requested maximum output and one block. Actual tool input and previous generated text consume this headroom; native vLLM uses the same request maximum without this active-window limit.')


def profile_stage(root, native_pool=None):
    run_path = root/'run.json'
    if not run_path.exists():
        return dict(status='not_started')
    run = json.loads(run_path.read_text())
    begin = run.get('model_ready_at')
    if begin is None:
        return dict(status=run['status'], formal_started=False)
    samples = records(root/'resources.jsonl')
    end = min(run.get('window_deadline') or float('inf'), run.get('ended_at') or
              max((s['at'] for s in samples), default=begin))
    samples = [s for s in samples if begin <= s['at'] <= end]
    events = records(root/'bridge/requests.jsonl')
    starts = [r for r in events if r.get('event') == 'start' and begin <= r['at'] <= end]
    finishes = [r for r in events if r.get('event') == 'finish' and begin <= r['at'] <= end]
    start_keys = {(r['task'], r['turn']) for r in starts}
    closed_http = [r for r in events if r.get('event') in ('finish','cancelled','error')
        and begin <= r['at'] <= end and (r['task'],r['turn']) in start_keys]
    result = dict(status=run['status'], begin=begin, end=end,
        gpu_demand=gpu_demand(samples, starts, closed_http, end),
        closed_unsuccessful_http=sum(r['event'] != 'finish' for r in closed_http),
        history_matched_requests=history_matched_requests(finishes),
        output_reservations=output_reservations(root, starts))
    # A cancelled HTTP stream need not mean its native request has retired yet.
    # Keep its outer envelope open for the conservative GPU cohort proof below.
    capture = root/'server/capture'
    if not (capture/'model_step_latency.jsonl').exists():
        result['capacity_overlap'] = capacity_overlap(starts, finishes, samples, [], 0,
            run['workers'], native_pool, end)
        return result
    offset = run['model_ready_monotonic']-begin
    lo, hi = begin+offset+2, end+offset-2
    rows = [r for r in records(capture/'model_step_latency.jsonl') if lo <= r['started_at'] <= hi]
    result['model_steps'] = step_groups(rows)
    result['capacity_overlap'] = capacity_overlap(starts, finishes, samples, rows, offset,
        run['workers'], native_pool, end)
    restores = [r for r in records(capture/'async_restore.jsonl') if lo <= r['started'] <= r['finished'] <= hi]
    result['restores'] = restore_groups(restores)
    checkpoints = [r for r in records(capture/'checkpoint_timing.jsonl') if lo <= r['at'] <= hi]
    result['checkpoints'] = checkpoint_profile(checkpoints)
    result['public_work'] = confirmed_public_work(records(capture/'public_work.jsonl'), begin, end)
    result['duration_scope'] = 'Per-step and per-request spans can overlap. Their sums are not disjoint fractions of GPU wall time; host spans can include asynchronous GPU execution.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.root/'manifest.json').read_text())
    carried = manifest.get('carried_windows', {})
    stages = manifest.get('comparison_stages', manifest['stages'])
    results = {}
    for stage in stages:
        native = 'native-'+stage.split('-')[-1]
        native_root = Path(carried[native]['path']) if native in carried else a.root/native
        baseline = native_root/'metrics-before-formal.txt'
        pool = native_geometry(baseline.read_text()) if baseline.exists() else None
        results[stage] = profile_stage(Path(carried[stage]['path']) if stage in carried else a.root/stage, pool)
    result = dict(kind='deepswe_capacity_profile_v1', stages=results)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
