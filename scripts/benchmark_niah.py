#!/usr/bin/env python3
"""Frozen qw3 NIAH inputs/scoring against the current native KVMem service.

Gold spans are used exclusively for offline validation and retrieval auditing;
only original token IDs, the public Query span and sampling settings are sent.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import statistics
from threading import Condition, Lock
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from tokenizers import Tokenizer

from benchmark_client_io import compact_response

TASKS = ['niah_single_2', 'niah_multikey_2', 'niah_multivalue', 'niah_multiquery']


def normalized_config(raw):
    if 'kvmem' not in raw:
        return raw
    kv=raw['kvmem']
    return {**raw,'backend_port':raw['port'],'query_history':kv['query_history'],
        'guided_query_tokens':kv['guided_query_tokens'],'active_context_tokens':kv['select_budget'],
        'gen_budget_tokens':kv['gen_budget'],'block_tokens':kv['block_tokens']}


class CpuCredits:
    """Benchmark ingress gate: reserve entire input+output, never truncate it.

Eight clients remain the maximum. Long histories wait outside the engine when
their complete immutable records plus staging credit exceed the CPU budget.
Fresh sessions use a zero-retention checkpoint quota in this campaign.
"""
    def __init__(self, budget):
        self.budget=budget
        self.used=self.active=self.peak=0
        self.condition=Condition()
        self.events=[]
        self.cancelled=False

    def acquire(self, case, amount):
        if amount>self.budget:
            raise ValueError('One complete history exceeds CPU benchmark credits')
        with self.condition:
            self.condition.wait_for(lambda:self.cancelled or self.used+amount<=self.budget)
            if self.cancelled:
                raise RuntimeError('Campaign stopped after a technical failure')
            self.used+=amount;self.active+=1;self.peak=max(self.peak,self.active)
            self.events.append({'at':time.time(),'case':case,'event':'admit','active':self.active,'reserved_bytes':self.used})

    def release(self, case, amount):
        with self.condition:
            self.used-=amount;self.active-=1
            self.events.append({'at':time.time(),'case':case,'event':'release','active':self.active,'reserved_bytes':self.used})
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled=True
            self.condition.notify_all()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scorer(path):
    # Execute only the frozen score function, never the external campaign runner.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    scope = {'re': re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['score']


def covered(begin, end, records):
    cursor = begin
    for start, count in sorted(records):
        if start > cursor:
            break
        cursor = max(cursor, start + count)
        if cursor >= end:
            return True
    return False


def prepare(args, cfg):
    source = json.loads((args.reference / 'cohort.json').read_text())
    chosen = [r for r in source if r['nominal_input_tokens'] in args.lengths
              and r['position_percent'] in args.depths and r['seed'] in args.seeds]
    chosen.sort(key=lambda r: (r['nominal_input_tokens'], r['position_percent'], TASKS.index(r['task']), r['seed']))
    assert len(chosen) == len(TASKS)*len(args.lengths)*len(args.depths)*len(args.seeds)
    tok = Tokenizer.from_file(str(Path(cfg['tokenizer']) / 'tokenizer.json'))
    rows = []
    for original in chosen:
        source_path = Path(original['token_path'])
        assert sha(source_path) == original['token_sha256']
        tokens = np.fromfile(source_path, dtype='<i4').tolist()
        assert len(tokens) == original['input_tokens']
        row = dict(original)
        qb, qe = row['query_begin'], row['query_end']
        assert tok.decode(tokens[qb:qe], skip_special_tokens=False) == row['question']
        assert not any(value in row['question'] for value in row['expected'])
        removal = None
        assert len(tokens)+args.max_output<=cfg['max_model_len'], 'Full frozen input must fit; truncation is forbidden'
        block=cfg.get('block_tokens',128)
        reserve=((256+cfg['guided_query_tokens']+block-1)//block)*block
        assert qe + reserve <= cfg['max_model_len']
        for needle in row['needles']:
            text = tok.decode(tokens[needle['start']:needle['end']], skip_special_tokens=False)
            assert needle['key'] in text and needle['value'] in text
        destination = args.output / 'inputs' / (row['id'] + '.i32')
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(tokens, dtype='<i4').tofile(destination)
        row.update(served_input_tokens=len(tokens), served_query=[qb, qe],
                   served_sha256=sha(destination), removed_filler=removal,
                   actual_depth_percent=[100*n['start']/len(tokens) for n in row['needles']])
        rows.append(row)
    identity = {'source_cohort_sha256': sha(args.reference/'cohort.json'),
                'source_scorer_sha256': sha(args.reference/'run.py'),
                'tokenizer_sha256': sha(Path(cfg['tokenizer'])/'tokenizer.json'),
                'server_profile_sha256': sha(args.config), 'max_output': args.max_output,
                'temperature': 1.0, 'top_p': .95, 'top_k': 20,
                'thinking_token_budget':4096,
                'thinking': 'original open think prefix; original forced transition plus close at 4096',
                'query_history': cfg['query_history'], 'guided_query_tokens': cfg['guided_query_tokens'],
                'active_budget': cfg['active_context_tokens'], 'gen_budget': cfg['gen_budget_tokens'],
                'draft_method': cfg['draft_method'], 'draft_tokens': cfg['draft_tokens'],
                'workers': args.workers, 'cases': len(rows),
                'cpu_headroom_gib':args.cpu_headroom_gib,
                'protocol': 'single request; chronological pressure prefill then explicit Query replay',
                'selection_uses_gold': False,
                'rows': rows}
    if cfg.get('kvmem', {}).get('action_attention'):
        identity['attention_settings'] = {k: v for k, v in cfg['kvmem'].items()
            if k.startswith(('action_', 'prefill_attention')) or k in ('async_refresh', 'refresh_quantum_tokens')}
        identity['attention_protocol'] = ('Cold independent sessions: output Action scores are '
            'collected after first Query selection; no historical Action feedback at that selection.')
        if cfg['kvmem'].get('prefill_attention'):
            identity['protocol']='single request; attention pressure prefill then explicit Query replay'
            identity['attention_protocol']='Cold prefill attention must contribute to the first semantic selection; output feedback remains isolated.'
    path = args.output/'manifest.json'
    if path.exists():
        assert args.resume, 'Existing campaign requires --resume'
        assert json.loads(path.read_text()) == identity, 'Frozen campaign changed'
    else:
        save(path, identity)
    return rows


def read_events(path, offset=0):
    if not path.exists():
        return []
    with path.open() as stream:
        stream.seek(offset)
        return [json.loads(line) for line in stream if line.strip()]


def run_case(args, cfg, row, scorer):
    directory = args.output/'cases'/row['id']
    directory.mkdir(parents=True, exist_ok=True)
    source = args.output/'inputs'/(row['id']+'.i32')
    assert sha(source) == row['served_sha256']
    tokens = np.fromfile(source, dtype='<i4').tolist()
    session = 'niah-' + uuid.uuid4().hex
    payload = {'model': cfg['served_model_name'], 'prompt': tokens, 'session_id': session,
               'cache_salt': session, 'max_tokens': args.max_output, 'temperature': 1.,
               'top_p': .95, 'top_k': 20, 'min_p':0, 'presence_penalty':0,'repetition_penalty':1,
               'seed': row['seed'], 'return_token_ids': True,'thinking_token_budget':4096,
               'kvmem_query_span': row['served_query'], 'kvmem_reselect': 'force'}
    save(directory/'request.json', {k:v for k,v in payload.items() if k != 'prompt'})
    files = {name:args.capture/name for name in ('bounded_refresh.jsonl','query_history.jsonl')}
    offsets = {name: path.stat().st_size if path.exists() else 0 for name,path in files.items()}
    started = time.monotonic()
    print(json.dumps({'event':'start', 'id':row['id'], 'input_tokens':len(tokens)}), flush=True)
    result = {k:row[k] for k in ('id','task','nominal_input_tokens','served_input_tokens',
                                'position_percent','seed','served_sha256','removed_filler')}
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{cfg['backend_port']}/v1/completions",
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with http.open(request, timeout=args.timeout) as response:
            response = json.load(response)
        save(directory/'response.json', compact_response(response, tokens))
        assert response['usage']['prompt_tokens'] == len(tokens)
        raw = response['choices'][0]['text']
        evaluation = scorer(raw, row['expected'])
        events = {name:[e for e in read_events(path, offsets[name])
                        if e['request_id'].startswith(response['id']+'-')]
                  for name,path in files.items()}
        replays = [e for e in events['bounded_refresh.jsonl'] if e['selection']=='query']
        assert len(replays)==1, 'Missing/duplicate explicit Query replay'
        replay = replays[0]
        assert replay['query_span']==row['served_query']
        assert replay['query_history_enabled'] and replay['mean_k_backend']=='qw3_cuda'
        assert 0 <= replay['guided_query_tokens'] <= cfg['guided_query_tokens']
        assert replay['selected_tokens'] <= cfg['active_context_tokens']
        if cfg.get('kvmem', {}).get('action_attention'):
            assert replay['action_attention_enabled']
            assert replay['action_history_pages'] == 0, 'Cold NIAH must not inherit another session feedback'
            if cfg['kvmem'].get('prefill_attention'):
                assert replay['attention_source']=='prefill' and replay['prefill_attention_samples']>0
                if cfg['kvmem'].get('action_pressure'):
                    pressure=[e for e in events['bounded_refresh.jsonl'] if e['selection']=='pressure']
                    assert pressure and all(e['attention_source']=='prefill' for e in pressure)
        assert len(events['query_history.jsonl']) == 1 and events['query_history.jsonl'][0]['observed']
        assert events['query_history.jsonl'][0]['input_end']==len(tokens)
        needles = [{**n, 'selected':covered(n['start'], n['end'], replay['selected_history_ranges'])}
                   for n in row['needles']]
        save(directory/'events.json', events)
        result.update(status='completed', evaluation=evaluation,
                      completion_tokens=response['usage']['completion_tokens'],
                      finish_reason=response['choices'][0]['finish_reason'],
                      needle_retrieval=needles, all_needles_selected=all(n['selected'] for n in needles),
                      guided_tokens=replay['guided_query_tokens'], guided_fallback=replay.get('guided_fallback'),
                      selected_tokens=replay['selected_tokens'], guided_seconds=replay['guided_seconds'],
                      pressure_refreshes=sum(e['selection']=='pressure' for e in events['bounded_refresh.jsonl']))
    except Exception as error:
        detail = error.read().decode() if isinstance(error, urllib.error.HTTPError) else str(error)
        result.update(status='error', error=type(error).__name__+': '+detail)
    result['seconds'] = time.monotonic()-started
    save(directory/'result.json', result)
    print(json.dumps({'event':'finish', 'id':row['id'], 'status':result['status'],
        'coverage':result.get('evaluation',{}).get('normalized_coverage'),
        'all_needles_selected':result.get('all_needles_selected'), 'seconds':result['seconds']}), flush=True)
    return result


def admitted_case(args,cfg,row,scorer,credits):
    # 17 archived full-attention layers, 4 KV heads, head_dim=256,
    # FP8 raw K + FP8 V. Keep G extra credit for sealed/staging overlap.
    block=cfg.get('block_tokens',128)
    extent=row['served_input_tokens']+args.max_output+cfg['gen_budget_tokens']
    amount=((extent+block-1)//block)*block*17*4*256*2
    started=time.monotonic();credits.acquire(row['id'],amount)
    try:
        result=run_case(args,cfg,row,scorer)
        if result['status'] != 'completed':
            # Stop ingress before releasing these credits. The main collector
            # may still be writing reports while queued clients wake up.
            credits.cancel()
        result['cpu_queue_seconds']=time.monotonic()-started-result['seconds']
        result['cpu_reserved_bytes']=amount
        save(args.output/'cases'/row['id']/'result.json',result)
        # Request HTTP completion may precede fenced authority retirement.
        # With no retained checkpoints, wait for its native free acknowledgment
        # before admitting a replacement history against the same credits.
        response_path=args.output/'cases'/row['id']/'response.json'
        if response_path.exists():
            rid=json.loads(response_path.read_text())['id']
            deadline=time.monotonic()+120
            while True:
                archive_path=args.capture/'archive.json'
                archived=json.loads(archive_path.read_text()) if archive_path.exists() else {}
                done=[e for e in archived.get('completed',[])
                      if e['request_id'].startswith(rid+'-') and e.get('retirement_finished_at')]
                if done:break
                if time.monotonic()>deadline:raise TimeoutError('CPU history retirement acknowledgment')
                time.sleep(.1)
            if cfg.get('kvmem', {}).get('action_attention'):
                action = [e for e in read_events(args.capture/'action_attention.jsonl')
                          if e['request_id'].startswith(rid+'-')]
                assert len(action) == 1, 'Missing/duplicate terminal Action attention evidence'
                result['action_attention'] = action[0]
                save(args.output/'cases'/row['id']/'action_attention.json', action[0])
                save(args.output/'cases'/row['id']/'result.json', result)
        return result
    finally:
        credits.release(row['id'],amount)


def report(args, rows):
    results = [json.loads(p.read_text()) for r in rows
               if (p:=args.output/'cases'/r['id']/'result.json').exists()]
    good = [r for r in results if r['status']=='completed']
    groups = []
    for length in args.lengths:
        for task in ['all']+TASKS:
            selected = [r for r in good if r['nominal_input_tokens']==length and (task=='all' or r['task']==task)]
            if selected:
                groups.append({'nominal_input_tokens':length, 'task':task, 'cases':len(selected),
                    'actual_input_tokens': sorted(set(r['served_input_tokens'] for r in selected)),
                    'coverage': statistics.fmean(r['evaluation']['normalized_coverage'] for r in selected),
                    'exact':sum(r['evaluation']['normalized_exact_numeric_set'] for r in selected),
                    'all_needles_selected':sum(r['all_needles_selected'] for r in selected),
                    'thinking_closed':sum(r['evaluation']['thinking_closed'] for r in selected)})
    summary={'expected':len(rows),'finished':len(results),'completed':len(good),
             'errors':len(results)-len(good),'groups':groups,'results':results}
    save(args.output/'summary.json', summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',default=[65536,131072,262144])
    parser.add_argument('--depths',type=int,nargs='+',default=[10,30,50,70,90])
    parser.add_argument('--seeds',type=int,nargs='+',default=[1000,2000,3000])
    parser.add_argument('--max-output',type=int,default=8192)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--cpu-headroom-gib',type=int,default=8)
    parser.add_argument('--timeout',type=int,default=3600)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--case-id',action='append',default=[])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=normalized_config(json.loads(args.config.read_text()))
    assert cfg['query_history'] and cfg['draft_method']=='mtp'
    assert 0 < args.max_output <= cfg['gen_budget_tokens']
    rows=prepare(args,cfg)
    if args.prepare_only:
        print(json.dumps({'prepared':len(rows)}));return
    scorer=load_scorer(args.reference/'run.py')
    assert cfg['kvmem']['archive_backend']=='cpu'
    assert cfg['kvmem']['session_checkpoint_bytes']==1, 'Campaign must release completed independent histories'
    credits=CpuCredits(cfg['kvmem']['cpu_read_cache_bytes']-args.cpu_headroom_gib*1024**3)
    ledger=args.output/'cpu_admission.json'
    if ledger.exists():
        prior=json.loads(ledger.read_text())
        assert prior['budget']==credits.budget
        credits.events=prior['events']
        credits.peak=prior['peak']
    pending=[r for r in rows if not (args.output/'cases'/r['id']/'result.json').exists()]
    if args.case_id:
        assert set(args.case_id)<=set(r['id'] for r in rows)
        pending=[r for r in pending if r['id'] in args.case_id]
    if args.limit:pending=pending[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(admitted_case,args,cfg,r,scorer,credits) for r in pending]
        for future in as_completed(futures):
            try:
                result=future.result(); report(args,rows)
                save(ledger,{'peak':credits.peak,'budget':credits.budget,'events':credits.events})
                if result['status']!='completed':
                    raise RuntimeError('Technical failure; preserve evidence and stop campaign: '+result['id'])
            except Exception:
                credits.cancel()
                for pending_future in futures:pending_future.cancel()
                raise
    summary=report(args,rows)
    print(json.dumps({k:summary[k] for k in ('expected','finished','completed','errors','groups')}),flush=True)


if __name__=='__main__':
    main()
