"""Frozen qw3 MemoryAgentBench cohort; exact CPU prefix reuse, no KV recomputation.

Original full token inputs and per-case sampling are verified before sending.
Answers/gold metadata are only read for saved results, never used for selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
from threading import Semaphore, Lock
import time
import urllib.error
import urllib.request
import uuid

from benchmark_niah import CpuCredits, save, sha
from benchmark_client_io import (CaptureSlice, ClientTiming, CompactResults, PreparedSequence,
                                 VERSION as CLIENT_IO_VERSION)


def digest_tokens(tokens):
    return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()


def source(row):
    path=Path(row['source'])
    assert sha(path)==row['source_cache_sha256']
    with gzip.open(path,'rt') as stream: data=json.load(stream)
    assert len(data['tokens'])==row['input_tokens']
    assert digest_tokens(data['tokens'])==row['token_sha256']
    assert hashlib.sha256(data['full'].encode()).hexdigest()==row['prompt_sha256']
    assert data['query_begin']==row['query_begin'] and data['query_end']==row['query_end']
    return data


def wait_terminal(capture,rid,timeout=180):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        path=capture/'archive.json'
        records=json.loads(path.read_text()).get('completed',[]) if path.exists() else []
        found=[r for r in records if r['request_id'].startswith(rid+'-') and
               (r.get('retirement_ack_ready_at') or r.get('retirement_finished_at'))]
        if found:return found[0]
        time.sleep(.1)
    raise TimeoutError('Memory benchmark terminal checkpoint not acknowledged')


def run(args):
    cfg=json.loads(args.config.read_text());kv=cfg['kvmem']
    graph_v2 = kv.get('graph_scoring_version', 'v1') == 'v2'
    if graph_v2:
        if kv.get('query_mean_quota_fraction', 0):
            import hybrid_memory_protocol as graph_protocol
        elif getattr(args, 'guided_comparison', False):
            import guided_benchmark_protocol as graph_protocol
        else:
            import graph_v2_protocol as graph_protocol
        graph_protocol.validate_profile(cfg, graph_protocol.infer_arm(cfg), 'memoryagentbench')
    else:
        assert kv['archive_backend']=='cpu' and kv['prefill_attention'] and kv['action_pressure']
    assert kv['session_freeze_prompt'] and not kv['query_history']
    assert not kv['session_capture_terminal'] and not kv['session_capture_decode']
    reference=json.loads((args.reference/'cohort.json').read_text())
    rows=[r for r in reference['cases'] if r['suite']=='memoryagentbench']
    assert len(rows)==1116
    args.output.mkdir(parents=True,exist_ok=True)
    groups={}
    for row in rows:groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
    prefixes={g:min(r['query_begin'] for r in group)//1024*1024 for g,group in groups.items()}
    manifest={'reference':str(args.reference),'cohort_sha256':sha(args.reference/'cohort.json'),
        'profile_sha256':sha(args.config),'cases':1116,'groups':len(groups),'rows':rows,
        'workers':args.workers,'active_groups':args.active_groups,'cpu_headroom_gib':args.cpu_headroom_gib,
        'client_bookkeeping_version':CLIENT_IO_VERSION,'capture_read_scope':'request-start-offsets',
        'prefix_boundaries':prefixes,'gold_used_for_selection':False,
        'protocol':'Original full token IDs, no truncation; independent questions restore frozen P before Query selection. Input attention only; output/Query-history scores never transfer across questions.',
        'reference_protocol_difference':'qw3 uses named frozen-prefix ingest; vLLM captures the same 1024-aligned prefix while serving the first full question. No zero-output warmup request.'}
    if graph_v2:
        manifest.update(graph_protocol=graph_protocol.SCHEMA, graph_arm=graph_protocol.infer_arm(cfg),
            score_domain_version=graph_protocol.SCORE_DOMAIN, pressure_policy=kv['pressure_policy'],
            graph_scoring_version='v2', graph_alpha=kv['graph_alpha'])
    path=args.output/'manifest.json'
    if path.exists():assert args.resume and json.loads(path.read_text())==manifest
    else:save(path,manifest)
    if args.prepare_only:
        print(json.dumps({'prepared':1116,'groups':len(groups),'max_input':max(r['input_tokens'] for r in rows)}));return
    for row in rows:
        previous=args.output/'cases'/row['id']/'result.json'
        if previous.exists() and json.loads(previous.read_text())['status']!='completed':
            raise RuntimeError('Archive and document the technical failure before resuming: '+row['id'])
    assert 1<=args.active_groups<=kv['session_capacity']
    credits=CpuCredits(kv['cpu_read_cache_bytes']-args.cpu_headroom_gib*1024**3)
    gate=Semaphore(args.active_groups);lock=Lock()
    progress=CompactResults(rows,args.output/'cases',
        ('id','status','source','prompt_tokens','usage','seconds','prefix_reused','error'))
    def summarize():
        records=progress.snapshot()
        report={'expected':1116,'finished':len(records),
            'completed':sum(r['status']=='completed' for r in records),
            'errors':sum(r['status']!='completed' for r in records),
            'input_tokens':sum(r.get('prompt_tokens',0) for r in records),
            'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in records),
            'results':records}
        save(args.output/'summary.json',report)
        save(args.output/'cpu_admission.json',{'budget':credits.budget,'peak':credits.peak,'events':credits.events})
        return report
    def group_run(group,items):
        pending=[r for r in items if not (args.output/'cases'/r['id']/'result.json').exists()]
        if args.case_id:pending=[r for r in pending if r['id'] in args.case_id]
        if args.per_group_limit:pending=pending[:args.per_group_limit]
        if not pending:return
        extent=max(r['input_tokens'] for r in items)+2*kv['gen_budget']
        amount=((extent+31)//32)*32*17*4*256*2
        session='mab-'+uuid.uuid4().hex
        prefix=prefixes[group];prefix_sha=None;completed=0
        def prepare(row):
            timing=ClientTiming()
            with timing.stage('source_verify_and_prefix_hash'):
                src=source(row)
                current=digest_tokens(src['tokens'][:prefix])
                old=args.reference/'results/attention/cases/memoryagentbench'/row['id']/'request_config.json'
                sampling=json.loads(old.read_text())
                assert sampling['max_tokens']==src['max_tokens'] and not src['thinking']
            return src,current,sampling,timing
        with gate:
            credits.acquire(group,amount)
            try:
                with PreparedSequence(pending,prepare) as prepared:
                    for row,(src,current,sampling,timing) in prepared:
                        if credits.cancelled:raise RuntimeError('Another group failed')
                        directory=args.output/'cases'/row['id'];directory.mkdir(parents=True,exist_ok=True)
                        assert not (directory/'attempt.json').exists(), 'Archive an interrupted attempt before resuming'
                        tokens=src['tokens']
                        if prefix_sha is None:prefix_sha=current
                        assert current==prefix_sha,'Shared prefix changed'
                        old=args.reference/'results/attention/cases/memoryagentbench'/row['id']/'request_config.json'
                        payload={k:sampling[k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed','max_tokens')}
                        assert sampling['max_tokens']==src['max_tokens'] and not src['thinking']
                        assert len(tokens)+payload['max_tokens']<=cfg['max_model_len']
                        payload.update(model=cfg['served_model_name'],prompt=tokens,session_id=session,cache_salt=session,
                            return_token_ids=True,thinking_token_budget=0,
                            kvmem_query_span=[row['query_begin'],row['query_end']],
                            kvmem_replay_span=[prefix,row['query_end']],kvmem_reselect='force')
                        save(directory/'request.json',{k:v for k,v in payload.items() if k!='prompt'}|
                            {'prompt_tokens':len(tokens),'token_sha256':row['token_sha256'],
                             'prefix_tokens':prefix,'prefix_sha256':prefix_sha,'sampling_sha256':sha(old)})
                        save(directory/'attempt.json',{'started_at':time.time(),'session':session,'group':group})
                        capture_slice=CaptureSlice(args.capture/name for name in ('bounded_refresh.jsonl','block_graph.jsonl','block_graph_inputs.jsonl'))
                        started=time.monotonic()
                        try:
                            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                            req=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",
                                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                            with timing.stage('http'):
                                with http.open(req,timeout=args.timeout) as stream:response=json.load(stream)
                            seconds=time.monotonic()-started;save(directory/'response.json',response)
                            assert response['usage']['prompt_tokens']==len(tokens)
                            assert response['choices'][0]['prompt_token_ids']==tokens
                            rid=response['id']
                            with timing.stage('terminal_wait'):
                                terminal=wait_terminal(args.capture,rid)
                            assert terminal['checkpoint_kind']=='P' and terminal['checkpoint_tokens']==prefix
                            with timing.stage('capture_and_validation'):
                                refresh=[e for e in capture_slice.read(args.capture/'bounded_refresh.jsonl') if e['request_id'].startswith(rid+'-')]
                                queries=[e for e in refresh if e['selection']=='query'];assert len(queries)==1
                                q=queries[0]
                                assert q['query_span']==[row['query_begin'],row['query_end']]
                                assert q['attention_source']=='prefill' and q['action_history_pages']==0
                                assert q['prefill_attention_samples']>0 and q['selected_tokens']<=kv['select_budget']
                                # Subsequent questions must restore their prefix. A full
                                # cold prefill would emit pressure events again.
                                pressure=[e for e in refresh if e['selection']=='pressure']
                                if completed:assert not pressure,'Frozen prefix was not reused'
                                text=response['choices'][0]['text']
                                answer=text.split('</think>',1)[-1].strip()
                                record={**src['info'],'suite':'memoryagentbench','id':row['id'],'status':'completed',
                                    'answer':answer,'raw_completion':text,'usage':response['usage'],
                                    'token_sha256':row['token_sha256'],'source_token_sha256':row['token_sha256'],
                                    'prompt_sha256':row['prompt_sha256'],'query_sha256':row['query_sha256'],
                                    'prompt_tokens':len(tokens),'source_prompt_tokens':len(tokens),'original_input_intact':True,
                                    'seconds':seconds,'metrics':response.get('metrics'),
                                    'prefix_tokens':prefix,'prefix_sha256':prefix_sha,'prefix_reused':bool(completed),
                                    'attention':q,'finish_reason':response['choices'][0]['finish_reason']}
                                if graph_v2:
                                    record['graph_v2'] = graph_protocol.audit_case(args.capture, rid, cfg,
                                        [row['query_begin'], row['query_end']],
                                        require_pressure=not completed and len(tokens) > kv['select_budget'] + kv['gen_budget'],
                                        event_reader=capture_slice.read,
                                        **(dict(prompt_tokens=len(tokens),output_tokens=payload['max_tokens'],
                                                replay_span=payload['kvmem_replay_span'])
                                           if kv.get('query_mean_quota_fraction',0) else {}))
                            save(directory/'events.json',refresh);save(directory/'terminal.json',terminal)
                            save(directory/'result.json',record);completed+=1
                            with timing.stage('progress_publish'):
                                with lock:
                                    progress.put(record)
                                    summary=summarize()
                                    print(json.dumps({'completed':summary['completed'],'id':row['id'],'seconds':seconds,'reused':completed>1}),flush=True)
                            save(directory/'client_timing.json',timing.summary())
                        except BaseException as error:
                            credits.cancel()
                            detail = error.read().decode(errors='replace') if isinstance(error,urllib.error.HTTPError) else str(error)
                            failed={'id':row['id'],'status':'error','error':detail,'seconds':time.monotonic()-started}
                            save(directory/'result.json',failed)
                            with lock:
                                progress.put(failed)
                                summarize()
                            raise
                if graph_v2:
                    # Frozen P sessions retain their own namespace after the
                    # last question. Release it before returning its CPU credit
                    # or admitting the next group; release is never a retry.
                    release = graph_protocol.release_owner(cfg, args.capture, session, tokens[-1])
                    save(args.output/'releases'/(group+'.json'), release)
            except BaseException:
                credits.cancel()
                raise
            finally:credits.release(group,amount)
    selected=list(groups.items())[:args.group_limit or None]
    assert not args.case_id or set(args.case_id)<=set(r['id'] for r in rows)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(group_run,g,items) for g,items in selected]
        for f in as_completed(futures):f.result()
    with lock:summary=summarize()
    assert summary['errors']==0
    if not (args.group_limit or args.per_group_limit or args.case_id):
        assert summary['completed']==1116,'Incomplete frozen benchmark cohort'
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('/home/chaidi/qw3/benchmark/attention_multilayer_prefix_reuse_20261012_r2'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--active-groups',type=int,default=2);p.add_argument('--cpu-headroom-gib',type=int,default=28)
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--group-limit',type=int,default=0)
    p.add_argument('--per-group-limit',type=int,default=0);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--case-id',action='append',default=[])
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
