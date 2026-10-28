"""Frozen qw3 LongMemEval-S 102 inputs, independent CPU histories, eight clients."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from benchmark_memory import source, wait_terminal
from benchmark_niah import CpuCredits, save, sha
from benchmark_client_io import CaptureSlice, CompactResults, VERSION as CLIENT_IO_VERSION

SAMPLING = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty', 'seed')


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
        graph_protocol.validate_profile(cfg, graph_protocol.infer_arm(cfg), 'longmemeval_s')
    else:
        assert kv['archive_backend']=='cpu' and kv['prefill_attention'] and kv['action_pressure']
        assert kv['session_checkpoint_bytes']==1 and kv['query_history']
    rows=[r for r in json.loads((args.reference/'cohort.json').read_text())['cases'] if r['suite']=='longmemeval_s']
    assert len(rows)==102 and len({r['id'] for r in rows})==102
    configs={};bindings=[]
    for row in rows:
        src=source(row)
        assert hashlib.sha256(src['query'].encode()).hexdigest()==row['query_sha256']
        old=args.reference/'results/attention/cases/longmemeval_s'/row['id']/'request_config.json'
        sampling=json.loads(old.read_text());configs[row['id']]=sampling
        assert src['thinking'] and sampling['enable_thinking'] and sampling['thinking_budget']==4096
        assert src['max_tokens']==sampling['max_tokens']==32768
        assert sampling['token_sha256']==row['token_sha256']
        assert sampling['kvmem_query_token_span']==dict(begin=row['query_begin'],end=row['query_end'])
        assert row['input_tokens']+kv['gen_budget']<=cfg['max_model_len']
        bindings.append({'id':row['id'],'source_sha256':sha(Path(row['source'])),
            'reference_request_sha256':sha(old),'sampling':{k:sampling[k] for k in SAMPLING}})
    manifest={'reference':str(args.reference),'source_cohort_sha256':sha(args.reference/'cohort.json'),
        'server_profile_sha256':sha(args.config),'workers':args.workers,'cases':102,'rows':rows,
        'bindings':bindings,'cpu_headroom_gib':args.cpu_headroom_gib,'gold_used_for_selection':False,
        'client_bookkeeping_version':CLIENT_IO_VERSION,'capture_read_scope':'request-start-offsets',
        'original_max_tokens':32768,'served_max_tokens':kv['gen_budget'],
        'output_cap_difference':'Native vLLM requires single-turn max_tokens <= G. Retain original K=36864/G=16384 attention pressure geometry, cap output at 16384; audit that no answer reaches cap.',
        'thinking_token_budget':4096,'protocol':'Cold independent full original token IDs, no truncation, no recompute. Prefill attention and pressure then explicit public Query selection; no cross-case feedback.'}
    if graph_v2:
        manifest.update(graph_protocol=graph_protocol.SCHEMA, graph_arm=graph_protocol.infer_arm(cfg),
            score_domain_version=graph_protocol.SCORE_DOMAIN, pressure_policy=kv['pressure_policy'],
            graph_scoring_version='v2', graph_alpha=kv['graph_alpha'])
    path=args.output/'manifest.json'
    if path.exists():assert args.resume and json.loads(path.read_text())==manifest
    else:save(path,manifest)
    if args.prepare_only:
        print(json.dumps({'prepared':len(rows),'input_range':[min(r['input_tokens'] for r in rows),max(r['input_tokens'] for r in rows)]}));return
    for row in rows:
        d=args.output/'cases'/row['id']
        if (d/'attempt.json').exists():
            assert (d/'result.json').exists() and json.loads((d/'result.json').read_text())['status']=='completed', 'Preserve and investigate interrupted/failed attempts before resume'
    credits=CpuCredits(kv['cpu_read_cache_bytes']-args.cpu_headroom_gib*1024**3)
    progress=CompactResults(rows,args.output/'cases',
        ('id','status','prompt_tokens','usage','hit_output_cap','error'))
    def execute(row):
        extent=row['input_tokens']+2*kv['gen_budget'];amount=((extent+31)//32)*32*17*4*256*2
        enqueued=time.monotonic();credits.acquire(row['id'],amount)
        directory=args.output/'cases'/row['id'];directory.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();session='lme-'+uuid.uuid4().hex
        save(directory/'attempt.json',{'started_at':time.time(),'session':session})
        try:
            src=source(row);tokens=src['tokens'];old=configs[row['id']]
            payload={k:old[k] for k in SAMPLING}
            payload.update(model=cfg['served_model_name'],prompt=tokens,session_id=session,cache_salt=session,
                max_tokens=kv['gen_budget'],thinking_token_budget=4096,return_token_ids=True,
                kvmem_query_span=[row['query_begin'],row['query_end']],kvmem_reselect='force')
            save(directory/'request.json',{k:v for k,v in payload.items() if k!='prompt'}|
                 {'prompt_tokens':len(tokens),'token_sha256':row['token_sha256'],'original_max_tokens':old['max_tokens']})
            print(json.dumps({'event':'start','id':row['id'],'input_tokens':len(tokens)}),flush=True)
            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            capture_slice=CaptureSlice(args.capture/name for name in ('bounded_refresh.jsonl','block_graph.jsonl','action_attention.jsonl','block_graph_inputs.jsonl'))
            sent=time.monotonic()
            with http.open(req,timeout=args.timeout) as stream:response=json.load(stream)
            seconds=time.monotonic()-sent;save(directory/'response.json',response)
            assert response['usage']['prompt_tokens']==len(tokens)
            assert response['choices'][0]['prompt_token_ids']==tokens
            rid=response['id'];terminal=wait_terminal(args.capture,rid)
            assert terminal.get('retirement_finished_at'), 'Independent history has not retired'
            events=[e for e in capture_slice.read(args.capture/'bounded_refresh.jsonl') if e['request_id'].startswith(rid+'-')]
            queries=[e for e in events if e['selection']=='query'];assert len(queries)==1
            q=queries[0];assert q['query_span']==[row['query_begin'],row['query_end']]
            assert q['attention_source']=='prefill' and q['action_history_pages']==0
            assert q['prefill_attention_samples']>0 and q['selected_tokens']<=kv['select_budget']
            assert q['mean_k_backend']=='qw3_cuda' and q['query_history_enabled'] is (not graph_v2)
            pressure=[e for e in events if e['selection']=='pressure']
            assert pressure and all(e['attention_source']=='prefill' for e in pressure)
            action=[e for e in capture_slice.read(args.capture/'action_attention.jsonl') if e['request_id'].startswith(rid+'-')]
            assert len(action)==1
            raw=response['choices'][0]['text'];answer=raw.split('</think>',1)[-1].strip()
            record={**src['info'],'id':row['id'],'suite':'longmemeval_s','status':'completed',
                'answer':answer,'raw_completion':raw,'thinking_closed':'</think>' in raw,
                'usage':response['usage'],'metrics':response.get('metrics'),'seconds':seconds,
                'cpu_queue_seconds':started-enqueued,'cpu_reserved_bytes':amount,
                'prompt_tokens':len(tokens),'token_sha256':row['token_sha256'],'prompt_sha256':row['prompt_sha256'],
                'query_sha256':row['query_sha256'],'original_input_intact':True,
                'attention':q,'pressure_refreshes':len(pressure),'finish_reason':response['choices'][0]['finish_reason'],
                'hit_output_cap':response['usage']['completion_tokens']>=payload['max_tokens']}
            if graph_v2:
                import qualify_graph_niah as graph_client
                record['graph_v2'] = graph_protocol.audit_case(args.capture, rid, cfg,
                    [row['query_begin'], row['query_end']], require_pressure=True,event_reader=capture_slice.read,
                    **(dict(prompt_tokens=len(tokens),output_tokens=payload['max_tokens'])
                       if kv.get('query_mean_quota_fraction',0) else {}))
                record['release'] = graph_client.wait_owner_released(args.capture, session,
                    {'response_id': rid}, terminal)
            save(directory/'events.json',events);save(directory/'terminal.json',terminal)
            save(directory/'action_attention.json',action[0]);save(directory/'result.json',record)
            progress.put(record)
            print(json.dumps({'event':'finish','id':row['id'],'seconds':seconds,'output_tokens':response['usage']['completion_tokens']}),flush=True)
            return record
        except BaseException as error:
            credits.cancel()
            detail=error.read().decode(errors='replace') if isinstance(error,urllib.error.HTTPError) else repr(error)
            failed={'id':row['id'],'status':'error','error':detail,'seconds':time.monotonic()-started}
            save(directory/'result.json',failed);progress.put(failed)
            raise
        finally:credits.release(row['id'],amount)
    def summarize():
        records=progress.snapshot()
        summary={'expected':102,'completed':sum(r['status']=='completed' for r in records),
            'errors':sum(r['status']!='completed' for r in records),
            'hit_output_cap':sum(r.get('hit_output_cap',False) for r in records),
            'input_tokens':sum(r.get('prompt_tokens',0) for r in records),
            'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in records)}
        save(args.output/'summary.json',summary)
        save(args.output/'cpu_admission.json',{'budget':credits.budget,'peak':credits.peak,'events':list(credits.events)})
        return summary
    pending=[r for r in rows if not (args.output/'cases'/r['id']/'result.json').exists()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,r) for r in pending]
        for f in as_completed(futures):
            try:f.result();print(json.dumps(summarize()),flush=True)
            except BaseException:
                credits.cancel()
                for other in futures:other.cancel()
                summarize();raise
    summary=summarize();assert summary['completed']==102 and not summary['errors']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('/home/chaidi/qw3/benchmark/attention_multilayer_prefix_reuse_20261012_r2'))
    for name in ('config','capture','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--cpu-headroom-gib',type=int,default=28)
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    run(p.parse_args())
