"""Audit paired first answers and actual per-layer pressure selection evidence."""
import argparse
import csv
from pathlib import Path
import statistics

import recall_comparison_protocol as p
import run_layerwise_query_diagnostic as driver
from vllm_kvmem.core.block_runs import candidate_starts


def analyze(root):
    fixtures={arm:p.read(root/arm/'fixture.json') for arm in ['shared','independent']}
    if fixtures['shared']['rows']!=fixtures['independent']['rows']:
        raise ValueError('Unpaired original sample list')
    for arm in fixtures:
        driver.verify(root/arm)
        if p.read(root/arm/'status.json')['status']!='completed':raise ValueError('Wait for both complete arms')
    cases=[];pressure_overlaps=[];pressure_events=0;selections=0
    ids=set()
    for row in fixtures['shared']['rows']:
        arms={}
        for arm in fixtures:
            case=root/arm/'formal/sequential/cases'/row['id']
            result=p.read(case/'result.json');response=p.read(case/'response.json');request=p.read(case/'request.json')
            evidence=p.read(case/'evidence.json')
            if p.sha(case/'response.json')!=result['response_sha256'] or p.sha(case/'evidence.json')!=result['evidence_sha256']:
                raise ValueError('Original answer/evidence changed')
            if response['response_id'] in ids:raise ValueError('Response reused')
            ids.add(response['response_id'])
            if request['sampling']!=p.case_sampling(fixtures[arm],row) or request['query_span']!=row['served_query']:
                raise ValueError('Generation/Query controls changed')
            if arm=='independent':
                events=[e for e in evidence['events'] if e['event']=='pressure_selected']
                if not events:raise ValueError('No actual pressure event')
                for event in events:
                    pressure_events+=1
                    if event['actual_method']!='accumulated_attention_per_layer' or event['fallback_reason'] is not None:
                        raise ValueError('Pressure method changed or fell back')
                    layers=event['layers'];selections+=len(layers)
                    if [e['layer'] for e in layers]!=list(range(16)):raise ValueError('Incomplete pressure layers')
                    sets=[]
                    for layer in layers:
                        chosen=set(layer['selected_starts']);candidate=set(candidate_starts(layer))
                        if (not chosen<=candidate or layer['charged_tokens']>36864 or
                            layer['selected_tokens']!=event['selected_tokens'] or
                            layer['accepted_score_end']>event['accepted_history']):
                            raise ValueError('Pressure candidate, time or budget violation')
                        sets.append({s for s in chosen if s>=41*32})
                    pressure_overlaps.append(statistics.fmean(len(a&b)/max(1,min(len(a),len(b)))
                        for i,a in enumerate(sets) for b in sets[i+1:]))
            arms[arm]=dict(score=result['official_case_score'],seconds=result['seconds'],
                truncated=result['finish_reason']=='length',response_sha256=result['response_sha256'])
        cases.append(dict(id=row['id'],task=row['task'],arms=arms))
    per_task={task:{arm:statistics.fmean(c['arms'][arm]['score'] for c in cases if c['task']==task)
        for arm in fixtures} for task in dict.fromkeys(c['task'] for c in cases)}
    macros={arm:p.read(root/arm/'analysis/comparison.json')['macro_score'] for arm in fixtures}
    ds=[c['arms']['independent']['score']-c['arms']['shared']['score'] for c in cases]
    report=dict(cases=len(cases),independent_first_answers=len(ids),macro_scores=macros,
        macro_delta=macros['independent']-macros['shared'],per_task=per_task,
        paired=dict(wins=sum(d>0 for d in ds),losses=sum(d<0 for d in ds),ties=sum(d==0 for d in ds)),
        truncated={arm:sum(c['arms'][arm]['truncated'] for c in cases) for arm in fixtures},
        median_seconds={arm:statistics.median(c['arms'][arm]['seconds'] for c in cases) for arm in fixtures},
        pressure_events=pressure_events,layer_pressure_selections=selections,
        mean_optional_block_overlap=statistics.fmean(pressure_overlaps),rows=cases,
        inputs_query_sampling_budgets_equal=True,formal_answers_resampled=0,
        limitations=['13x2 small batch; scores are not evidence of a stable full-benchmark gain.',
            'Independent scoring necessarily observes all 16 complete-attention layers; original shared control retains spread3.',
            'Physical layer slots stay fixed; the original-text union may differ.'])
    (root/'analysis').mkdir(exist_ok=True);p.save(root/'analysis/comparison.json',report)
    with (root/'analysis/per-task.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.writer(stream);w.writerow(['task','shared_pressure','independent_pressure','delta'])
        for task,v in per_task.items():w.writerow([task,v['shared'],v['independent'],v['independent']-v['shared']])
    print({k:v for k,v in report.items() if k not in ['rows','per_task','limitations']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root.resolve())
