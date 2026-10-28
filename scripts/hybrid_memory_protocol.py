"""Explicit MeanK/FPS4 quota benchmark; never relabel it as historical C."""
from copy import deepcopy
from pathlib import Path

import graph_v2_protocol as original
import ruler_hybrid_protocol as ruler

SCHEMA = 'kvmem-mean-fps-half-memory-1'
SCORE_DOMAIN = original.SCORE_DOMAIN
require = original.require
release_owner = original.release_owner


def infer_arm(profile):
    require(profile['kvmem'].get('query_mean_quota_fraction') == .5,'Expected fixed 50:50 quota')
    alpha = profile['kvmem'].get('graph_alpha')
    require(type(alpha) in (int,float) and alpha in (0,.2),'Unsupported hybrid graph alpha')
    return 'mean_fps_half_graph' if alpha else 'mean_fps_half'


def validate_profile(profile, arm, suite):
    require(arm == infer_arm(profile), 'Unknown hybrid memory arm')
    k = profile['kvmem']
    for key,value in dict(query_score_mode='key_fps_4', query_fps_halo_blocks=1,
            query_fps_optimized=True,query_fps_prebuild=True,query_fps_prebuild_device='cpu',
            query_fps_cpu_backend='native',query_fps_cpu_workers=16,
            query_window_cache_bytes=12*1024**3,graph_alpha=(.2 if arm == 'mean_fps_half_graph' else 0.),graph_mode='multi',
            graph_aggregation='sum',guided_query_tokens=0,query_history=False,
            prefill_physical_coalescing=False).items():
        require(k.get(key) == value, 'Hybrid memory setting differs: '+key)
    # Reuse the historical geometry/ownership contract after explicitly
    # removing only the two scoring/diagnostic differences it cannot express.
    checked = deepcopy(profile)
    checked['kvmem'].update(graph_alpha=.2,graph_trace=False)
    original.validate_profile(checked,'C',suite)
    require(k['query_window_cache_bytes'] <= 16*1024**3,'FPS cache exceeds existing client headroom')


def expected_query_budget(profile, selected, *, prompt_tokens, output_tokens, replay_span=None):
    from vllm_kvmem.request_plan import RequestPlan
    k = profile['kvmem']; block = k['block_tokens']
    query = selected['query_span']; history = selected['history_begin']
    expected_replay = [query[0]//block*block,query[1]] if replay_span is None else list(replay_span)
    require(len(expected_replay) == 2 and all(type(v) is int for v in expected_replay) and
        0 <= expected_replay[0] <= query[0]//block*block and expected_replay[0]%block == 0 and
        expected_replay[1] == query[1], 'Invalid original Query replay bounds')
    require(type(prompt_tokens) is int and type(output_tokens) is int and
        0 < query[0] < query[1] <= prompt_tokens and 0 < output_tokens <= k['gen_budget'] and
        history == expected_replay[0] and selected['replay_span'] == expected_replay,
        'Missing original request bounds or unexpected Query replay coordinates')
    plan = RequestPlan(tuple(query),block_tokens=block,gen_budget=k['gen_budget'],output_tokens=output_tokens,
        replay=tuple(expected_replay))
    return plan.selection_budget(history,prompt_tokens,k['select_budget']+k['gen_budget'])


def audit_case(capture,response_id,profile,query,*,require_pressure=False,event_reader=None,
               prompt_tokens=None,output_tokens=None,replay_span=None):
    import qualify_block_graph as q
    reader = event_reader or q.records
    match = lambda e:q.native_matches({'response_id':response_id},e)
    events = [e for e in reader(Path(capture)/'bounded_refresh.jsonl') if match(e)]
    graphs = [e for e in reader(Path(capture)/'block_graph.jsonl') if match(e)]
    selected = [e for e in events if e['selection'] == 'query']
    pressure = [e for e in events if e['selection'] == 'pressure']
    require(len(selected) == 1,'Missing unique public Query')
    selected = selected[0]; k = profile['kvmem']
    require(selected['query_span'] == list(query) and selected['mean_k_backend'] == 'qw3_cuda' and
        selected['query_scoring_method'] == 'key_fps_4' and not selected['query_history_enabled'] and
        selected['action_history_pages'] == 0 and selected['prefill_attention_samples'] > 0,
        'Original Query/history/precision isolation changed')
    if require_pressure:
        require(pressure,'Original long input did not execute pressure selection')
    for event in events:
        require(event['guided_query_tokens'] == 0 and event['score_domain_version'] == SCORE_DOMAIN and
            event['scoring_version'] == 'v2' and event['pressure_policy'] == 'attention' and
            0 < event['selected_tokens'] <= k['select_budget'],'Selection policy or budget differs')
    pg = [e for e in graphs if e['phase'] == 'pressure']
    qg = [e for e in graphs if e['phase'] == 'query']
    require(len(qg) == 1 and len(pg) == len(pressure),'Missing Query/pressure graph accounting')
    for p,g in zip(pressure,pg):
        require(p['seed_source'] == g['seed_source'] == 'prefill' and not g['propagation_applied'] and
            g['transport_mask_count'] == 0 and all(p[key] == g[key] for key in
            ('selected_tokens','score_mask_count','transport_mask_count')),
            'Pressure must retain cumulative input attention without propagation')
    graph = qg[0]
    ruler.audit_quota_propagation(selected,graph,profile)
    budget = expected_query_budget(profile,selected,prompt_tokens=prompt_tokens,output_tokens=output_tokens,
        replay_span=replay_span)
    quota = selected.get('query_quota_selection',{})
    require(quota == graph.get('query_quota_selection') and quota.get('mean_fraction') == .5 and
        quota.get('mode') == 'mean_fps_quota' and quota.get('budget') == budget,
        'Missing actual hybrid quota evidence')
    ranges = selected['selected_history_ranges']
    require(len({s for s,_ in ranges}) == len(ranges) == quota['selected_blocks'] and
        all(s%32 == 0 and 0 < n <= 32 for s,n in ranges) and
        sum(n for _,n in ranges) == selected['selected_tokens'] and
        len(ranges)*32 == quota['charged_tokens'] <= budget and
        quota['mean_selected_blocks'] == quota['mean_quota_blocks'] == quota['optional_blocks']//2 and
        quota['fixed_blocks']+quota['mean_selected_blocks']+quota['fps_added_blocks']+
            quota['mean_fill_blocks'] == quota['selected_blocks'],
        'Original native block budget or separate quota accounting differs')
    require(set(range(0,k['sink_blocks']*32,32)) <= {s for s,_ in ranges},'Fixed sink lost')
    cpu = ruler.audit_cpu_fps(selected['query_key_fps_scoring'],workers=16)
    full = None
    if k['graph_trace']:
        inputs = [e for e in reader(Path(capture)/'block_graph_inputs.jsonl') if match(e)]
        require(len(inputs) == 1,'Missing independent full-score replay evidence')
        full = ruler.audit_selection(selected,graph,inputs[0],profile,expected_budget=budget)
    return dict(query=selected,pressure=pressure,graph=graphs,quota=quota,cpu_fps_audit=cpu,
        native_replay_audit=full,score_domain_version=SCORE_DOMAIN,
        phases=dict(query=1,pressure=len(pressure)),
        audit_scope=('Full original vectors plus independent native picker replay' if full else
            'Per-case stage counts, selected ranges, budgets, CPU-only construction and pressure isolation; '
            'full-vector native replay is covered by the separate longest-prefix qualification'))
