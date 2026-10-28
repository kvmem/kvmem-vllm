"""Fixed 13 x 10 RULER protocol for CPU-built FPS4 and budgeted halo1."""
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
PROFILE = 'ruler-fps-native-cpu-halo-256k-20261023'
QUALIFICATION_PROFILE = PROFILE+'-qualification-eight'
CACHE_BYTES = 12*1024**3
TRACE_BYTES = 1024**3
SAMPLING = dict(max_tokens=8192, temperature=1., top_p=.95, top_k=20, min_p=0,
    presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096)


def expected_profile(reference, *, qualification=False):
    p = deepcopy(reference); p['port'] = 18870 if qualification else 18869
    p['kvmem'].update(query_score_mode='key_fps_4', query_fps_halo_blocks=1,
        query_fps_optimized=True, query_fps_prebuild=True, query_fps_prebuild_device='cpu',
        query_fps_cpu_backend='native', query_fps_cpu_workers=16,
        query_window_cache_bytes=CACHE_BYTES, graph_query_trace_max_bytes=TRACE_BYTES,
        graph_mode='multi', graph_scoring_version='v2', graph_aggregation='sum', graph_alpha=0.,graph_trace=True,
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
    qualification=fixture['execution_profile']==QUALIFICATION_PROFILE
    require(fixture['execution_profile'] in (PROFILE,QUALIFICATION_PROFILE) and
        profile==expected_profile(fixture['reference_profile'],qualification=qualification),
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
    selection=halo.audit_selection(selected,query_graph[0],inputs[0],profile)
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
