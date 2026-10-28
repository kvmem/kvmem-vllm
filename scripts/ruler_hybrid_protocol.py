"""Fixed 13 x 10 RULER protocol for 50:50 MeanK / CPU-built FPS4 with budgeted halo1."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from threading import Lock

import ruler_protocol as ruler
from graph_v2_protocol import SCORE_DOMAIN, require

_spec = importlib.util.spec_from_file_location('_ruler_fps_cpu_halo', Path(__file__).with_name('test_ruler_fps_halo_answers.py'))
halo = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(halo)
PROFILE = 'ruler-mean-fps-quota-half-256k-20261023'
QUALIFICATION_PROFILE = PROFILE+'-qualification-eight'
GRAPH_PROFILE = 'ruler-mean-fps-quota-half-graph-256k-20261024'
GRAPH_QUALIFICATION_PROFILE = GRAPH_PROFILE+'-qualification-eight'
CACHE_BYTES = 12*1024**3
TRACE_BYTES = 1024**3
SAMPLING = dict(max_tokens=8192, temperature=1., top_p=.95, top_k=20, min_p=0,
    presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096)


def is_qualification(fixture):
    return fixture['execution_profile'] in (QUALIFICATION_PROFILE, GRAPH_QUALIFICATION_PROFILE)


def expected_profile(reference, *, qualification=False, graph_alpha=0.):
    require(type(graph_alpha) in (int,float) and graph_alpha in (0,.2), 'Unsupported hybrid graph alpha')
    p = deepcopy(reference); p['port'] = 18870 if qualification else 18869
    p['kvmem'].update(query_score_mode='key_fps_4', query_fps_halo_blocks=1,
        query_mean_quota_fraction=.5, query_fps_optimized=True, query_fps_prebuild=True, query_fps_prebuild_device='cpu',
        query_fps_cpu_backend='native', query_fps_cpu_workers=16,
        query_window_cache_bytes=CACHE_BYTES, graph_query_trace_max_bytes=TRACE_BYTES,
        graph_mode='multi', graph_scoring_version='v2', graph_aggregation='sum', graph_alpha=graph_alpha,graph_trace=True,
        guided_query_tokens=0, query_history=False, prefill_physical_coalescing=False)
    return p


def select_rows(source):
    require(source['cases']==260 and source['samples_per_task']==20 and source['target_length']==262144,
        'Expected the original complete 260-case 256K fixture')
    ruler.validate_cohort(source['rows'],20)
    for task in ruler.TASK_FAMILIES:
        rows = [row for row in source['rows'] if row['task']==task]
        require({row['index'] for row in rows}==set(range(20)), 'Original task ordinals differ')
    rows = [deepcopy(row) for row in source['rows'] if row['index']<10]
    ruler.validate_cohort(rows,10)
    return rows


def execution_rows(rows):
    return sorted(rows,key=lambda row:(row['index'],list(ruler.TASK_FAMILIES).index(row['task'])))


def qualification_rows(source):
    """The first original admission wave; independent of previous answers."""
    return execution_rows(select_rows(source))[:8]


def budget_audit(profile, fixture):
    from qualify_graph_niah import history_bytes
    total = profile['kvmem']['cpu_read_cache_bytes']; cache = profile['kvmem']['query_window_cache_bytes']
    credit = fixture['cpu_payload_credits_bytes']; charges = sorted((history_bytes(row,fixture,profile) for row in fixture['rows']),reverse=True)
    require(total==100000000000 and cache==CACHE_BYTES and credit==82820130816 and credit<=total-cache,
        'History admission/cache allocations exceed the original shared CPU budget')
    require(sum(charges[:8])<=credit,'Eight largest full original histories exceed existing conservative credits')
    return dict(total_cpu_budget_bytes=total, shared_fps_cache_bytes=cache,
        native_archive_limit_bytes=total-cache, client_history_credits_bytes=credit,
        eight_largest_charged_histories_bytes=sum(charges[:8]), other_fixed_reserve_bytes=total-cache-credit,
        charge_scope='Each history includes original input + max output 8192 + generation reserve 16384; '
            '12 GiB cache is carved from the original 16 GiB non-history reserve, not subtracted twice.',
        maximum_inflight=8)


def validate_execution(profile,fixture):
    qualification=is_qualification(fixture)
    alpha = .2 if fixture['execution_profile'] in (GRAPH_PROFILE,GRAPH_QUALIFICATION_PROFILE) else 0.
    require(fixture['execution_profile'] in (PROFILE,QUALIFICATION_PROFILE,GRAPH_PROFILE,GRAPH_QUALIFICATION_PROFILE) and
        profile==expected_profile(fixture['reference_profile'],qualification=qualification,graph_alpha=alpha),
        'Execution profile differs from explicit frozen changes')
    require(fixture['cases']==(8 if qualification else 130) and fixture['samples_per_task']==(None if qualification else 10) and fixture['target_length']==262144 and
        fixture['workers']==8 and fixture['sampling']==SAMPLING and fixture['starts_in_thinking'] is True and
        fixture['request_timeout_seconds']==3600, 'Original cohort/sampling/timeout changed')
    if qualification:
        require([(row['task'],row['index']) for row in fixture['rows']]==
            [(task,0) for task in list(ruler.TASK_FAMILIES)[:8]],
            'Qualification must use the fixed original first eight queued cases')
    else:
        ruler.validate_cohort(fixture['rows'],10)
        require(all({r['index'] for r in fixture['rows'] if r['task']==task}==set(range(10))
            for task in ruler.TASK_FAMILIES),'Use each original task first ten ordinals')
    k=profile['kvmem']
    require((k['select_budget'],k['gen_budget'],k['block_tokens'],k['sink_blocks'],k['recent_blocks'])==
        (36864,16384,32,41,0),'Original KV/fixed-keep budget differs')
    require(profile['execution_mode']=='compiled_graph' and profile['draft_method']=='mtp' and profile['draft_tokens']==4 and
        profile['max_num_seqs']==8 and profile['max_num_batched_tokens']==16424 and profile['prefill_chunk_tokens']==2048,
        'Keep native compiled MTP4, at most eight owners, batch16424/chunk2048')
    require(k['pressure_policy']=='attention' and k['prefill_attention'] and k['action_attention'] and
        k['action_fraction']==0 and k['action_scope']=='non-thinking' and k['mean_k_backend']=='qw3_cuda' and
        k['archive_backend']=='cpu' and k['graph_trace'], 'Pressure, Query precision or CPU archive policy changed')
    return budget_audit(profile,fixture)


def audit_native_cpu_counters(stats, *, workers, background=False):
    prefix='cpu_' if background else 'fps_cpu_'
    require(stats.get(prefix+'backend')=='native' and stats.get(prefix+'workers')==workers,
        'Actual native CPU backend or shared worker count differs')
    names=('fps_cpu_groups','fps_cpu_native_groups','fps_cpu_reference_groups','fps_cpu_ambiguity_groups',
        'fps_cpu_pool_busy_groups','fps_cpu_unsupported_fallback_groups')
    counts={name:stats.get(name) for name in names}
    require(all(type(value) is int and value>=0 for value in counts.values()),'Missing or invalid native CPU fallback counters')
    require(counts['fps_cpu_native_groups']+counts['fps_cpu_pool_busy_groups']+
        counts['fps_cpu_unsupported_fallback_groups']==counts['fps_cpu_groups'] and
        counts['fps_cpu_reference_groups']==counts['fps_cpu_ambiguity_groups']+
        counts['fps_cpu_pool_busy_groups']+counts['fps_cpu_unsupported_fallback_groups'] and
        counts['fps_cpu_ambiguity_groups']<=counts['fps_cpu_native_groups'],
        'Native CPU groups/reference ambiguity accounting differs')
    require(counts['fps_cpu_pool_busy_groups']==counts['fps_cpu_unsupported_fallback_groups']==0,
        'Unexpected native CPU pool-busy or unsupported fallback; preserve this attempt')
    return counts


def audit_cpu_fps(stats, *, workers=8):
    require(stats.get('construction_device')=='cpu' and stats.get('fps_gpu_groups')==0,
        'Every representative Key build, including Query misses, must use CPU')
    prebuild=stats.get('prebuild',{})
    require(prebuild.get('enabled') is True and prebuild.get('device')=='cpu','CPU prebuild did not execute')
    require(prebuild.get('fps_gpu_groups')==0,'Background representative construction used GPU or lacks counters')
    query_native=audit_native_cpu_counters(stats,workers=workers)
    background_native=audit_native_cpu_counters(prebuild,workers=workers,background=True)
    require(not prebuild.get('last_error') and prebuild.get('failed_records',0)==0,
        'CPU prebuild encountered a real error; preserve this attempt')
    counters=('queued_records','published_records','skipped_records','failed_records','deferred_records',
        'cancelled_records','stale_records','query_wait_seconds','build_seconds','fps_gpu_groups')
    for name in counters:
        value=prebuild.get(name,0)
        require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid CPU prebuild accounting: '+name)
    return dict(construction_device='cpu',cpu_backend='native',cpu_workers=workers,
        query_native=query_native,background_native=background_native,
        query_gpu_groups=0,background_gpu_groups=0,
        cache_hit_blocks=stats.get('cache_hit_blocks',0), cache_prebuilt_hit_blocks=stats.get('cache_prebuilt_hit_blocks',0),
        cache_rebuilt_blocks=stats.get('cache_rebuilt_blocks',0), cache_bytes=stats.get('cache_bytes'),
        cache_build_seconds=stats.get('cache_build_seconds'),scoring_seconds=stats.get('scoring_seconds'),prebuild=prebuild)


def audit_quota_propagation(query, graph, profile):
    """Compact per-case proof that BOTH original seeds used one graph version."""
    alpha = profile['kvmem']['graph_alpha']
    require(graph.get('seed_source') == 'key_fps_4' and graph.get('mode') == 'multi' and
        graph.get('scoring_version') == 'v2' and graph.get('aggregation') == 'sum' and
        graph.get('alpha') == alpha, 'Hybrid Query graph policy differs')
    if alpha == 0:
        require(graph.get('identity') is True and graph.get('propagation_applied') is False,
            'Alpha-zero hybrid unexpectedly propagated seeds')
        return None
    require(alpha == .2, 'Unsupported hybrid graph alpha')
    proof = graph.get('query_quota_propagation', {})
    require(proof == query.get('query_quota_propagation') and
        proof.get('order') == 'independent_graph_then_quota' and
        set(proof) == {'order','mean_k','key_fps_4'}, 'Missing independent two-seed propagation proof')
    for name in ('mean_k','key_fps_4'):
        d = proof[name]
        require(d.get('mode') == 'multi' and d.get('scoring_version') == 'v2' and
            d.get('aggregation') == 'sum' and d.get('alpha') == alpha and
            d.get('locality_weight') == d.get('attention_weight') == .1 and
            d.get('score_sum_conserved') is True and d.get('eligible') == graph['transport_mask_count'],
            'Independent branch graph operator or mask differs: '+name)
        require(all(d.get(key) == graph.get(key) for key in
            ('revision','runtime_epoch','accepted_through')), 'Hybrid graph snapshot binding differs: '+name)
        require(math.isclose(d['raw_score_sum'], d['propagated_score_sum'],rel_tol=1e-10,abs_tol=1e-10),
            'Hybrid graph did not preserve seed mass: '+name)
        require(d['locality_weight']+d['attention_weight'] == alpha and
            type(d['identity']) is bool, 'Invalid independent graph diagnostics')
    require(graph['identity'] == proof['key_fps_4']['identity'] and
        graph['propagation_applied'] is (not graph['identity']), 'FPS propagation summary differs')
    return proof


def mean_key_mass_bound(tokens, pages, layers, heads):
    """Roundoff bound for the unchanged qw3 FP32 softmax/score reductions.

    FP16 operands do not explain mass drift: normalization uses FP32. The
    small-index kernel atomically adds L*T*H positive contributions per page;
    its error cannot be checked with a Query-length-independent 1e-3 bound.
    The tiled path reduces at most `batch` distributions, then sums batches.
    gamma(n)=n*u/(1-n*u) bounds positive FP32 sequential sums. Include the
    row reduction, division/weight rounding and fast-exp allowance as well.
    This only checks normalization. Exact exported-score graph/picker replay
    and Query row/lifecycle checks remain separate and mandatory.
    """
    require(all(type(x) is int and x > 0 for x in (tokens,pages,layers,heads)) and
        pages <= 32768, 'Invalid MeanK normalization geometry')
    distributions = tokens*layers*heads
    if pages <= 8192:
        route, steps = 'atomic_fp32', distributions+pages+32
    else:
        batch = min(distributions,(64*1024**2//4)//pages)
        route = 'tiled_fp32'
        steps = batch+math.ceil(distributions/batch)+math.ceil(pages/128)+7+32
    nu = steps*2.**-24
    require(nu < .2, 'MeanK roundoff bound too loose; requires independent numerical review')
    return dict(route=route,distributions=distributions,rounding_steps_bound=steps,
        relative_bound=nu/(1-nu),absolute_bound=tokens*nu/(1-nu))


def audit_seed_mass(mean, fps, query, inputs, profile):
    tokens = query['query_score_tokens']
    require(type(tokens) is int and tokens > 0,'Invalid Query token count')
    masses = dict(mean_k=math.fsum(mean),key_fps_4=math.fsum(fps))
    require(abs(masses['key_fps_4']-tokens) < 1e-3,'Full-domain FPS Query seed mass differs')
    proof = dict(expected_mass=tokens,actual_mass=masses,mean_k_absolute_error=abs(masses['mean_k']-tokens),
        fps_absolute_error=abs(masses['key_fps_4']-tokens),score_vectors_changed=False)
    if proof['mean_k_absolute_error'] < 1e-3:
        return dict(proof,check='strict_absolute_1e-3')
    require(query.get('mean_k_backend') == profile['kvmem'].get('mean_k_backend') == 'qw3_cuda',
        'Unexpected MeanK backend for FP32 roundoff audit')
    config = Path(profile.get('hf_config_path') or profile.get('model',''))
    if config.is_dir(): config = config/'config.json'
    require(config.is_file(),'MeanK roundoff audit requires original model geometry')
    data = config.read_bytes(); model = json.loads(data); model = model.get('text_config',model)
    kinds = model.get('layer_types',[])
    require(len(kinds) == model.get('num_hidden_layers') and kinds.count('full_attention') > 0,
        'Missing original full-attention layer geometry')
    bound = mean_key_mass_bound(tokens,sum(inputs['score_mask']),kinds.count('full_attention'),model['num_attention_heads'])
    require(proof['mean_k_absolute_error'] <= bound['absolute_bound'],
        'Full-domain MeanK Query seed mass exceeds FP32 roundoff bound')
    return dict(proof,check='native_fp32_positive_sum_bound',model_config_sha256=hashlib.sha256(data).hexdigest(),
        layers=kinds.count('full_attention'),query_heads=model['num_attention_heads'],**bound)


def audit_selection(query, graph, inputs, profile, *, expected_budget=None):
    """Replay the two stages independently of the production quota helper."""
    import analyze_block_graph as offline
    from vllm_kvmem.graph_scoring import score_graph
    k = profile['kvmem']; stats = query.get('query_key_fps_scoring', {})
    require(k['query_mean_quota_fraction'] == inputs.get('query_mean_quota_fraction') == .5 and
        k['graph_alpha'] in (0,.2) and k['query_fps_halo_blocks'] == 1 and
        k['query_score_mode'] == query.get('query_scoring_method') == 'key_fps_4',
        'Actual hybrid quota/scorer differs')
    require(stats.get('method') == 'key_fps_4' and stats.get('representatives_per_block') == 4 and
        stats.get('block_tokens') == 32 and stats.get('selection') == 'raw_key_farthest_first' and
        stats.get('normalization') == 'block softmax after per-layer/per-Query/per-head representative max' and
        stats.get('optimized') is k['query_fps_optimized'] and
        stats.get('mean_quota_scoring_seconds', -1) >= 0,
        'Independent FPS/MeanK scoring did not execute as requested')
    for event in (graph, inputs):
        require(event.get('seed_source') == 'key_fps_4' and event.get('mode') == 'multi' and
            event.get('scoring_version') == 'v2' and event.get('aggregation') == 'sum' and event.get('alpha') == k['graph_alpha'],
            'Hybrid graph configuration differs')
    branch_proof = audit_quota_propagation(query,graph,profile)
    mean = inputs.get('mean_scores', [])
    require(len(mean) == len(inputs['raw_scores']) and
        all(type(s) in (int,float) and math.isfinite(s) and s >= 0 for s in mean) and
        all(keep or s == 0 for keep,s in zip(inputs['score_mask'],mean)),
        'MeanK scores must cover the original full native score domain')
    mass_audit = audit_seed_mass(mean,inputs['raw_scores'],query,inputs,profile)
    cfg, snapshot = offline._validate_record(inputs,max_nodes=min(32768,k.get('repository_blocks',inputs['config']['repository_blocks'])))
    # The native production operator is checked against the independent
    # Python reference on each original full vector. Replay exact exported
    # doubles afterward so a near-tie cannot change native picker tie order.
    seeds = dict(mean_k=mean,key_fps_4=inputs['raw_scores'])
    picked_scores, diagnostics_by_branch = {}, {}
    for name,seed in seeds.items():
        with score_graph(seed,inputs['starts'],inputs['transport_mask'],snapshot,
                mode='multi',scoring_version='v2',aggregation='sum',alpha=k['graph_alpha'],
                cross_message_weight=inputs['cross_message_weight'],block_tokens=32,native=False) as output:
            actual = inputs.get('quota_propagated_scores',{}).get(name) if k['graph_alpha'] else seed
            require(actual is not None and len(actual) == len(seed) and
                all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in actual) and
                all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(actual,output.scores)),
                'Independent graph vector replay differs: '+name)
            require(all(keep or a == b for keep,a,b in zip(inputs['transport_mask'],actual,seed)),
                'Graph modified a transport-excluded seed: '+name)
            if branch_proof is not None:
                d = branch_proof[name]
                require(d['identity'] == output.diagnostics['identity'] and
                    math.isclose(d['raw_score_sum'],math.fsum(seed),rel_tol=1e-10,abs_tol=1e-10) and
                    math.isclose(d['propagated_score_sum'],math.fsum(actual),rel_tol=1e-10,abs_tol=1e-10),
                    'Compact branch proof differs from full original vectors: '+name)
            picked_scores[name] = actual
            diagnostics_by_branch[name] = {k:v for k,v in output.diagnostics.items() if k != 'scoring_seconds'}
    mean = picked_scores['mean_k']
    expected = halo.expected_groups(dict(inputs,raw_scores=picked_scores['key_fps_4']))
    metadata = inputs['query_group_selection']
    for key in halo.GROUP_KEYS:
        if key != 'mode':
            require(metadata.get(key) == expected[key], 'FPS group geometry changed: '+key)
    require(metadata.get('mode') == 'mean_fps_quota', 'Missing hybrid group provenance')
    memory, handles = offline._native_history(cfg, inputs['starts'], inputs['counts'])
    to_starts = lambda slots: {start for start,h in handles.items() if h.slot in slots}
    try:
        offline._validate_native_masks(inputs,memory,handles)
        pins = tuple(handles[s].slot for s in inputs['mandatory_starts'])
        budget = inputs['budget']
        expected_budget = k['select_budget'] if expected_budget is None else expected_budget
        require(0 < budget == expected_budget <= k['select_budget'], 'Native scoped selection budget differs')
        fixed_slots = set(memory._store.fixed_selection_blocks(list(pins), pressure=False, budget=budget))
        fixed = to_starts(fixed_slots)
        optional = max(0,min(len(handles),budget//32)-len(fixed))
        mean_quota = optional//2
        ordered = sorted(zip(inputs['starts'],mean),key=lambda pair:(pair[1],pair[0]),reverse=True)
        mean_chosen = [s for s,_ in ordered if s not in fixed][:mean_quota]
        protected = fixed | set(mean_chosen)
        native_mean = memory.preview_selection({handles[s].slot:v for s,v in zip(inputs['starts'],mean)},
            tuple(fixed_slots),action=False,prefill_scores=None,max_tokens=(len(fixed)+mean_quota)*32)
        require(to_starts(native_mean) == protected, 'MeanK top quota differs from independent ranking')
        protected_slots = tuple(handles[s].slot for s in protected)
        groups, scores = list(expected['groups']), list(expected['group_scores'])
        fps_selected = to_starts(memory.preview_group_selection(groups,scores,protected_slots,max_tokens=budget))
        priority = {s:-float(i+1) for i,(s,_) in enumerate(ordered)}
        groups += [[s,s+n] for s,n in zip(inputs['starts'],inputs['counts'])]
        scores += [priority[s] for s in inputs['starts']]
        actual = to_starts(memory.preview_group_selection(groups,scores,protected_slots,max_tokens=budget))
    finally:
        require(memory.close(),'Hybrid replay retained native metadata')
    ranges = query['selected_history_ranges']; counts = dict(zip(inputs['starts'],inputs['counts']))
    require(actual == {s for s,_ in ranges} and len(actual) == len(ranges) and
        all(counts.get(s) == n for s,n in ranges), 'Observed native hybrid selection differs from replay')
    require(fixed <= protected <= fps_selected <= actual and len(actual)*32 <= budget,
        'Quota/fixed content or physical block budget violated')
    quota = dict(mode='mean_fps_quota',mean_fraction=.5,budget=budget,
        fixed_blocks=len(fixed),optional_blocks=optional,mean_quota_blocks=mean_quota,
        mean_selected_blocks=len(protected-fixed),fps_added_blocks=len(fps_selected-protected),
        mean_fill_blocks=len(actual-fps_selected),selected_blocks=len(actual),charged_tokens=32*len(actual),
        score_combination='separate_full_domain_scores; budget_quota; no_score_blend',
        fixed_starts=sorted(fixed),mean_selected_starts=sorted(protected-fixed),
        fps_added_starts=sorted(fps_selected-protected),mean_fill_starts=sorted(actual-fps_selected),
        selected_starts=sorted(actual))
    require(metadata['quota_selection'] == quota,'Exported quota accounting differs')
    brief = {key:value for key,value in quota.items() if not key.endswith('_starts')}
    require(query.get('query_quota_selection') == graph.get('query_quota_selection') == brief,
        'Worker/graph hybrid summaries differ')
    summary = {key:value for key,value in metadata.items() if key not in
        ('groups','group_scores','group_members','anchor_starts')}
    summary['quota_selection'] = brief
    require(query.get('query_group_selection') == graph.get('query_group_selection') == summary,
        'Group summaries differ from complete hybrid export')
    require(metadata['selected_starts'] == sorted(actual) and
        metadata['selected_tokens'] == sum(n for _,n in ranges) and metadata['charged_tokens'] == len(actual)*32,
        'Hybrid selected extents/charges differ')
    return dict(native_group_picker_replay_equal=True,native_picker_replay_equal=True,
        independent_group_geometry_verified=True,independent_mean_quota_verified=True,
        selected_tokens=sum(n for _,n in ranges),charged_block_tokens=32*len(actual),
        propagation=diagnostics_by_branch['key_fps_4'],
        mean_propagation=diagnostics_by_branch['mean_k'],
        independent_graph_vectors_verified=True,
        propagation_applied=not diagnostics_by_branch['key_fps_4']['identity'],graph_alpha=k['graph_alpha'],
        seed_mass_audit=mass_audit,quota=quota)


def audit_case(capture,response,profile,query_span,*,event_reader=None):
    import qualify_block_graph as q
    reader=event_reader or q.records; match=lambda e:q.native_matches(response,e)
    bounded=[e for e in reader(Path(capture)/'bounded_refresh.jsonl') if match(e)]
    graph=[e for e in reader(Path(capture)/'block_graph.jsonl') if match(e)]
    inputs=[e for e in reader(Path(capture)/'block_graph_inputs.jsonl') if match(e)]
    queries=[e for e in bounded if e['selection']=='query']; pressure=[e for e in bounded if e['selection']=='pressure']
    require(len(queries)==len(inputs)==1 and pressure,'Missing unique public Query or full-input pressure evidence')
    selected=queries[0]; k=profile['kvmem']
    require(selected['query_span']==list(query_span) and selected['mean_k_backend']=='qw3_cuda' and
        not selected['query_history_enabled'] and selected['action_history_pages']==0 and
        selected.get('prefill_attention_samples',0)>0,'Query coordinates/precision or history isolation changed')
    for event in bounded:
        require(event['guided_query_tokens']==0 and 0<event['selected_tokens']<=k['select_budget'] and
            event['score_domain_version']==SCORE_DOMAIN and event['scoring_version']=='v2' and
            event['pressure_policy']=='attention','Actual guided/pressure/budget semantics differ')
    pressure_graph=[e for e in graph if e['phase']=='pressure']
    require(len(pressure_graph)==len(pressure),'Missing pressure graph summary')
    for p,g in zip(pressure,pressure_graph):
        require(p['seed_source']=='prefill' and g['seed_source']=='prefill' and not g['propagation_applied'] and
            g['transport_mask_count']==0 and all(p[key]==g[key] for key in
            ('seed_source','selected_tokens','score_mask_count','transport_mask_count')),
            'Prefill no longer uses cumulative attention without graph propagation')
    query_graph=[e for e in graph if e['phase']=='query'];require(len(query_graph)==1,'Missing final Query graph identity')
    selection=audit_selection(selected,query_graph[0],inputs[0],profile)
    cpu=audit_cpu_fps(selected['query_key_fps_scoring'],workers=k['query_fps_cpu_workers'])
    return dict(query=selected,bounded=bounded,pressure=pressure,graph=graph,query_inputs=inputs,
        selection_audit=selection,cpu_fps_audit=cpu,phases=dict(query=1,pressure=len(pressure)))


class QueryTraceIndex:
    """Stream the append-only shared log once into bounded immutable request files.

    Only request-id -> file metadata stays in RAM. No active log is renamed,
    truncated or rewritten; a partial final line waits for the next read.
    """
    def __init__(self,path,directory,*,max_records=130,max_bytes=TRACE_BYTES,max_record_bytes=6*1024**2):
        self.path,self.directory=Path(path),Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.max_records,self.max_bytes,self.max_record_bytes=max_records,max_bytes,max_record_bytes
        self.lock=Lock();self.offset=0;self.identity=None;self.files={};self.used=set()

    def _poll(self):
        try: stream=self.path.open('rb')
        except FileNotFoundError:return
        with stream:
            stat=os.fstat(stream.fileno());identity=(stat.st_dev,stat.st_ino)
            require(self.identity in (None,identity) and stat.st_size>=self.offset,'Shared Query trace rotated or truncated')
            require(stat.st_size<=self.max_bytes,'Shared Query trace exceeds frozen byte limit')
            self.identity=identity;stream.seek(self.offset)
            while True:
                line=stream.readline(self.max_record_bytes+2)
                if not line:break
                require(len(line)<=self.max_record_bytes+1,'Individual Query trace exceeds bounded record limit')
                if not line.endswith(b'\n'):break
                event=json.loads(line);rid=event.get('request_id')
                require(event.get('phase')=='query' and isinstance(rid,str) and rid and rid not in self.files,
                    'Duplicate or malformed request-scoped Query trace')
                require(len(self.files)<self.max_records,'More Query traces than frozen original cases')
                name=hashlib.sha256(rid.encode()).hexdigest()+'.jsonl';path=self.directory/name
                with path.open('xb') as target:target.write(line)
                self.files[rid]=dict(path=path,sha256=hashlib.sha256(line).hexdigest(),bytes=len(line))
                self.offset+=len(line)

    def take(self,response,destination):
        import qualify_block_graph as q
        with self.lock:
            self._poll()
            matches=[rid for rid in self.files if q.native_matches(response,{'request_id':rid})]
            require(len(matches)==1,'Missing or ambiguous request-scoped Query trace')
            rid=matches[0];require(rid not in self.used,'A Query trace was already consumed')
            record=self.files[rid];data=record['path'].read_bytes()
            require(hashlib.sha256(data).hexdigest()==record['sha256'],'Saved request trace changed')
            with Path(destination).open('xb') as target:target.write(data)
            self.used.add(rid)
            return json.loads(data)

    def audit(self):
        with self.lock:
            self._poll()
            require(len(self.files)==self.max_records and self.used==set(self.files), 'Not all original Query traces were consumed exactly once')
            require(self.path.stat().st_size==self.offset,'Unfinished trailing Query trace')
            return dict(records=len(self.files),bytes=self.offset,maximum_bytes=self.max_bytes,
                per_request_files=True,shared_original_preserved=True)
