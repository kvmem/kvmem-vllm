"""Audit LongMemEval-S frozen outputs and reuse only exact original judge requests.

External scoring is opt-in, uses only DEEPSEEK_API_KEY from the environment,
and retains qw3's original calibrated judge and frozen official prompt.
Missing judgments never contribute an incorrect answer or partial accuracy.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
from tokenizers import Tokenizer
from vllm_kvmem.serving.action_attention import ActionHistory

from benchmark_memory import source, digest_tokens
from benchmark_client_io import prompt_matches
from benchmark_niah import save, sha

FROZEN=Path('/home/chaidi/qw3/benchmark/qw3_memory_comparison_20261004/frozen')


def load_judge():
    path=FROZEN/'historical_lme_score.py'
    spec=importlib.util.spec_from_file_location('frozen_lme_judge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OFFICIAL=FROZEN/'historical_lme_evaluate_qa.py'
    return module


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def analyze(args):
    manifest=json.loads((args.campaign/'manifest.json').read_text());ref=Path(manifest['reference'])
    assert sha(ref/'cohort.json')==manifest['source_cohort_sha256']
    lme=load_judge();check=lme.prompt_function();out=args.campaign/'scoring'
    expected_identity=json.loads((ref/'results/attention/judge_deepseek_v41_flash/identity.json').read_text())
    assert expected_identity['parameters']==lme.PARAMETERS and expected_identity['api_base']==lme.API_BASE
    assert expected_identity['scorer_sha256']==sha(FROZEN/'historical_lme_score.py')
    assert expected_identity['prompt_sha256']==sha(lme.OFFICIAL)
    config = getattr(args, 'config', None) or args.campaign.parent/'profile.json'
    cfg=json.loads(config.read_text())
    assert sha(config)==manifest['server_profile_sha256']
    options=SimpleNamespace(**cfg['kvmem']);tok=Tokenizer.from_file(str(Path(cfg['tokenizer'])/'tokenizer.json'))
    markers=[tok.token_to_id(t) for t in ('<tool_call>','</tool_call>','<think>','</think>','<|im_start|>')]
    records={};requests={};bindings={};action_samples=0
    for row in manifest['rows']:
        d=args.campaign/'cases'/row['id'];p=d/'result.json'
        if not p.exists():continue
        r=json.loads(p.read_text())
        if r['status']!='completed':continue
        src=source(row);response=json.loads((d/'response.json').read_text());request=json.loads((d/'request.json').read_text())
        assert prompt_matches(response['choices'][0],src['tokens'])
        assert r['token_sha256']==row['token_sha256']==digest_tokens(src['tokens'])
        assert r['prompt_sha256']==row['prompt_sha256'] and r['query_sha256']==row['query_sha256']
        assert r['raw_completion']==response['choices'][0]['text'] and r['usage']==response['usage']
        assert r['answer']==r['raw_completion'].split('</think>',1)[-1].strip()
        assert len(response['choices'][0]['token_ids'])==r['usage']['completion_tokens']
        assert all(r[k]==v for k,v in src['info'].items())
        original=json.loads((ref/'results/attention/cases/longmemeval_s'/row['id']/'request_config.json').read_text())
        for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed'):
            assert request[k]==original[k]
        assert request['max_tokens']==manifest['served_max_tokens'] and request['thinking_token_budget']==4096
        assert request['kvmem_query_span']==[row['query_begin'],row['query_end']]
        q=r['attention'];action=json.loads((d/'action_attention.json').read_text())
        assert q['attention_source']=='prefill' and q['action_history_pages']==0 and q['selected_tokens']<=36864
        assert action['prefill_samples']==q['prefill_attention_samples']
        state=ActionHistory(options,markers);state.begin('offline',src['tokens']);expected=0
        consumed=response['choices'][0]['token_ids'][:-1]
        for start in range(0,len(consumed),32):
            chunk=consumed[start:start+32];state.parser.prepare(chunk);expected+=state.parser.commit(len(chunk)).bit_count()
        assert action['samples']==expected and action['rounds']==int(expected>0)
        if expected:assert abs(action['score_sum']-1)<.005
        action_samples+=expected
        assert r['metrics']['speculative_decoding']['num_spec_tokens']==4
        assert json.loads((d/'terminal.json').read_text())['retirement_finished_at']
        body={**lme.PARAMETERS,'messages':[{'role':'user','content':check(r['question_type'],r['question'],r['gold_answer'],r['answer'],abstention=r['question_id'].endswith('_abs'))}]}
        bindings[row['id']]={'result_sha256':sha(p),'source_sha256':row['source_cache_sha256'],
            'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),'judge_request_sha256':lme.digest(body)}
        records[row['id']]=r;requests[row['id']]=body
    assert records
    previous=json.loads((out/'bindings.json').read_text()) if (out/'bindings.json').exists() else {}
    assert all(bindings.get(k)==v for k,v in previous.items()),'A frozen answer/input binding changed'
    save(out/'bindings.json',bindings);save(out/'judge-identity.json',expected_identity)
    judgments={}
    for ident,binding in bindings.items():
        path=out/'scores'/f'{ident}.json'
        if path.exists():
            judgment=json.loads(path.read_text())
            assert judgment['request_sha256']==binding['judge_request_sha256']
            if type(judgment.get('correct')) is bool:judgments[ident]=judgment
    reused={}
    for arm in ('attention','query'):
        old=ref/'results'/arm;jroot=old/'judge_deepseek_v41_flash'
        assert json.loads((jroot/'identity.json').read_text())==expected_identity
        assert json.loads((jroot/'calibration.json').read_text())['passed']
        for judgment in lines(old/'longmemeval_judgments.jsonl'):
            ident=judgment['question_id']
            if ident in judgments or ident not in records or type(judgment.get('correct')) is not bool:continue
            if judgment['request_sha256']!=bindings[ident]['judge_request_sha256']:continue
            prior=json.loads((old/'cases/longmemeval_s'/ident/'result.json').read_text())
            current=records[ident]
            if any(prior.get(k)!=current.get(k) for k in ('token_sha256','prompt_sha256','question','question_type','gold_answer','answer')):continue
            assert judgment['answer_sha256']==bindings[ident]['answer_sha256']
            request=jroot/'requests'/f'{ident}.request.json'
            assert json.loads(request.read_text())==requests[ident]
            value={**judgment,'reused_from':str(old/'longmemeval_judgments.jsonl'),'reuse_file_sha256':sha(old/'longmemeval_judgments.jsonl')}
            save(out/'scores'/f'{ident}.json',value);judgments[ident]=value
    if args.external:
        key=os.environ.get('DEEPSEEK_API_KEY')
        if not key:raise RuntimeError('DEEPSEEK_API_KEY is required; model answers are preserved')
        calibration_path=out/'calibration.json'
        if not calibration_path.exists():
            fixtures=json.loads((FROZEN/'historical_lme_calibration.json').read_text())['cases'];checks=[]
            for i,fixture in enumerate(fixtures):
                label,raw,_=lme.assess(lme.Client(key),check,fixture['fixture'],fixture['hypothesis'],out/'calibration'/f'case_{i:02d}')
                checks.append({'expected':fixture['expected'],'observed':label,'response_model':raw['model']})
            save(calibration_path,{'passed':all(c['expected']==c['observed'] for c in checks),'cases':checks})
        assert json.loads(calibration_path.read_text())['passed'],'Original judge calibration failed'
        def grade(ident):
            r=records[ident];base={'question_id':ident,'question_type':r['question_type'],
                'answer_sha256':bindings[ident]['answer_sha256'],'request_sha256':bindings[ident]['judge_request_sha256'],'judge_model':lme.PARAMETERS['model']}
            try:
                label,raw,digest=lme.assess(lme.Client(key),check,{'question_id':ident,'question_type':r['question_type'],'question':r['question'],'answer':r['gold_answer']},r['answer'],out/'requests'/ident)
                assert digest==base['request_sha256'];base.update(correct=label,error=None,raw=raw['choices'][0]['message']['content'],response_model=raw['model'])
            except Exception as e:base.update(correct=None,error=str(e))
            save(out/'scores'/f'{ident}.json',base);return ident,base
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ident,value in pool.map(grade,[k for k in records if k not in judgments]):
                if type(value.get('correct')) is bool:judgments[ident]=value
    complete=len(records)==len(judgments)==102
    types=Counter(r['question_type'] for r in records.values())
    by_type={}
    for kind,count in types.items():
        values=[j for j in judgments.values() if j['question_type']==kind]
        by_type[kind]={'completed':count,'scored':len(values),'expected':17,
            'accuracy':sum(j['correct'] for j in values)/17 if count==len(values)==17 else None}
    report={'expected':102,'completed':len(records),'scored':len(judgments),'pending':102-len(judgments),
        'pending_ids':[r['id'] for r in manifest['rows'] if r['id'] not in judgments],
        'exact_request_judgments_reused':sum('reused_from' in j for j in judgments.values()),
        'complete':complete,'accuracy':sum(j['correct'] for j in judgments.values())/102 if complete else None,
        'by_type':by_type,'correct':sum(j['correct'] for j in judgments.values()) if complete else None,
        'reference_by_type':{arm:{kind:{'correct':sum(j['correct'] for j in lines(ref/f'results/{arm}/longmemeval_judgments.jsonl') if j['question_type']==kind),'expected':17} for kind in types} for arm in ('attention','query')},
        'output_cap_hits':sum(r['hit_output_cap'] for r in records.values()),
        'action_samples_verified':action_samples,'prefill_query_samples':sum(r['attention']['prefill_attention_samples'] for r in records.values()),
        'thinking_not_closed':sum(not r['thinking_closed'] for r in records.values()),
        'output_token_range':[min(r['usage']['completion_tokens'] for r in records.values()),max(r['usage']['completion_tokens'] for r in records.values())],
        'reference':{arm:json.loads((ref/'comparison.json').read_text())['summary'][arm]['longmemeval_s'] for arm in ('attention','query')},
        'note':'Accuracy is withheld until all 102 have official judgments. Exact-answer reused subset is selected by answer identity and cannot estimate full accuracy. Original output cap 32768 becomes 16384 to preserve original G=16384. K/G, inputs, sampling otherwise matched; quantization and execution differ.'}
    ledger=json.loads((args.campaign/'cpu_admission.json').read_text());times=[e['at'] for e in ledger['events']]
    elapsed=max(times)-min(times);inputs=sum(r['prompt_tokens'] for r in records.values());outputs=sum(r['usage']['completion_tokens'] for r in records.values())
    report['performance']={'wall_seconds':elapsed,'client_workers':manifest['workers'],'peak_admitted':ledger['peak'],
        'input_tokens':inputs,'output_tokens':outputs,'mixed_input_tokens_per_second':inputs/elapsed,
        'mixed_output_tokens_per_second':outputs/elapsed,'mean_http_seconds':statistics.fmean(r['seconds'] for r in records.values()),
        'mean_ttft_seconds':statistics.fmean(r['metrics']['time_to_first_token_ms']/1000 for r in records.values()),
        'mean_generation_seconds':statistics.fmean(r['metrics']['generation_time_ms']/1000 for r in records.values()),
        'mean_cpu_queue_seconds':statistics.fmean(r['cpu_queue_seconds'] for r in records.values()),
        'not_isolated_prefill_or_steady_decode':True}
    resources=lines(args.campaign/'resources.jsonl')
    if resources:
        archives=[r['archive'] for r in resources if r.get('archive')]
        metric_peak=lambda prefix:max([float(line.rsplit(' ',1)[1]) for r in resources for line in r.get('metrics',[]) if line.startswith(prefix)] or [0])
        checks={'samples':len(resources),'min_available_bytes':min(r['meminfo']['MemAvailable'] for r in resources),
            'max_cpu_authority_bytes':max([r.get('cpu_authority_bytes') or 0 for r in archives] or [0]),
            'max_disk_read_bytes':max([r.get('disk_read_bytes') or 0 for r in archives] or [0]),
            'max_disk_write_bytes':max([r.get('disk_write_bytes') or 0 for r in archives] or [0]),
            'max_nvme_capacity_bytes':max([r.get('nvme_capacity_bytes') or 0 for r in archives] or [0]),
            'max_swap_used_bytes':max(r['meminfo']['SwapTotal']-r['meminfo']['SwapFree'] for r in resources),
            'preemptions':metric_peak('vllm:num_preemptions_total{'),'peak_native_running':metric_peak('vllm:num_requests_running{')}
        assert all(checks[k]==0 for k in ('max_disk_read_bytes','max_disk_write_bytes','max_nvme_capacity_bytes','max_swap_used_bytes','preemptions'))
        report['resources']=checks
    save(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('reference','pending_ids','note','by_type')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--external',action='store_true')
    p.add_argument('--config',type=Path,help='Explicit profile for nested qualification/dataset outputs')
    analyze(p.parse_args())
