"""Measure simultaneous accepted decode after each owner's context is prepared."""
import concurrent.futures
import json
import statistics
import threading
import time

from benchmark import request_one, release_session, save, metrics


def window(rows):
    starts=[r['start_perf_counter']+r['chunks'][0][0] for r in rows]
    ends=[r['start_perf_counter']+r['chunks'][-1][0] for r in rows]
    lo,hi=max(starts),min(ends)
    tokens=sum(n for r in rows for t,n in r['chunks']
        if lo<r['start_perf_counter']+t<=hi) if hi>lo else 0
    active=peak=0
    for _,delta in sorted([(t,1) for t in starts]+[(t,-1) for t in ends]):
        active+=delta;peak=max(peak,active)
    return dict(common_decode_seconds=max(0,hi-lo),common_decode_tokens=tokens,
        common_decode_tokens_per_second=tokens/(hi-lo) if hi>lo else None,
        peak_overlapping_decode_requests=peak)


def capacity_case(root,length,concurrency,args,build):
    directory=root/'cases'/f'prepared{length}-out{args.decode_tokens}-c{concurrency}'
    directory.mkdir(parents=True,exist_ok=False)
    ids,span=build(length)
    states=[]
    def batch(phase,count):
        barrier=threading.Barrier(concurrency)
        stopped=threading.Event();samples=[]
        def observe():
            previous=None
            path=root/'server/capture/scheduler_resources.json'
            with (directory/f'{phase}-capacity.jsonl').open('w') as output:
                while not stopped.is_set():
                    try:
                        snapshot=json.loads(path.read_text())
                    except (FileNotFoundError,json.JSONDecodeError):
                        stopped.wait(.25);continue
                    if snapshot['at']!=previous:
                        previous=snapshot['at'];capacity=snapshot['gpu_capacity']
                        sample=dict(at=snapshot['at'],observed_at=time.perf_counter(),
                            active_reservations=len(capacity['requests']),
                            request_owned_pages=capacity['request_owned_pages'],
                            physical_free_pages=capacity['physical_free_pages'],
                            quotas=capacity['requests'],rejections=capacity['rejections'])
                        samples.append(sample);output.write(json.dumps(sample)+'\n');output.flush()
                        if not (root/'capacity-layout.json').exists():save(root/'capacity-layout.json',capacity)
                    stopped.wait(.25)
        observer=threading.Thread(target=observe,daemon=True);observer.start()
        def graph_counts():
            result={}
            for path in root.glob('graph-replays-rank*.json'):
                try:result[path.stem]=json.loads(path.read_text())
                except (OSError,json.JSONDecodeError):pass
            return result
        metrics_before=metrics()
        graph_before=graph_counts()
        begin=time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                if phase=='prepare':
                    futures=[pool.submit(request_one,directory/f'prepare-c{i+1}',ids,span,count,barrier)
                        for i in range(concurrency)]
                else:
                    futures=[pool.submit(request_one,directory/f'{phase}-c{i+1}',state['ids'],span,
                        count,barrier,True,state['row']['session_id']) for i,state in enumerate(states)]
                rows=[f.result() for f in futures]
        finally:
            stopped.set();observer.join()
        seconds=time.perf_counter()-begin
        graph_after=graph_counts()
        metrics_after=metrics()
        save(directory/f'{phase}-metrics.json',dict(before=metrics_before,after=metrics_after,
            delta={key:value-metrics_before.get(key,0) for key,value in metrics_after.items()
                   if value!=metrics_before.get(key,0)}))
        save(directory/f'{phase}-graphs.json',dict(before=graph_before,after=graph_after,
            delta={rank:{key:counts[key]-graph_before.get(rank,{}).get(key,0)
                for key in ('target_replays','draft_replays')}
                for rank,counts in graph_after.items()}))
        if phase=='prepare':
            states.extend(dict(ids=ids+row['output_token_ids'],row=row) for row in rows)
        else:
            for state,row in zip(states,rows):
                state['ids']+=row['output_token_ids'];state['row']=row
        return rows,seconds,max((s['active_reservations'] for s in samples),default=0)
    try:
        prepared,seconds,_=batch('prepare',1)
        preparation=dict(requests=concurrency,input_tokens=length*concurrency,wall_seconds=seconds,
            aggregate_prefill_input_tokens_per_second=length*concurrency/seconds,
            per_request_server_prefill_seconds=[r['extra']['metrics']['time_to_first_token_ms']/1000 for r in prepared],
            includes='request queue, prefill, first token and session publication; separate from decode rounds')
        save(directory/'preparation.json',preparation)
        batch('warmup',args.decode_warmup)
        rounds=[]
        for repetition in range(1,args.reps+1):
            rows,seconds,native_peak=batch(f'r{repetition}',args.decode_tokens)
            for row in rows:
                spec=row['extra']['metrics']['speculative_decoding']
                assert spec['num_spec_tokens']==args.mtp and spec['num_spec_steps']>0
            result=dict(repetition=repetition,initial_input_tokens=length,submitted_requests=concurrency,
                input_tokens_min=min(r['input_tokens'] for r in rows),input_tokens_max=max(r['input_tokens'] for r in rows),
                output_tokens=sum(r['output_tokens'] for r in rows),wall_seconds=seconds,
                aggregate_turn_output_tokens_per_second=sum(r['output_tokens'] for r in rows)/seconds,
                mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in rows),
                mean_queue_seconds=statistics.mean(r['extra']['metrics']['queue_time_ms']/1000 for r in rows),
                sampled_peak_native_reservations=native_peak,
                token_accounting_passed=all(r['token_accounting_passed'] for r in rows),**window(rows))
            rounds.append(result);save(directory/'summary.json',dict(rounds=rounds))
            print(json.dumps(result),flush=True)
        values=[r['common_decode_tokens_per_second'] for r in rounds if r['common_decode_tokens_per_second'] is not None]
        result=dict(case=directory.name,initial_input_tokens=length,submitted_requests=concurrency,
            preparation=preparation,rounds=rounds,
            median_common_decode_tokens_per_second=statistics.median(values) if values else None,
            all_rounds_have_common_decode=len(values)==args.reps,
            max_overlapping_decode_requests=max(r['peak_overlapping_decode_requests'] for r in rounds),
            median_turn_output_tokens_per_second=statistics.median(r['aggregate_turn_output_tokens_per_second'] for r in rounds),
            scope='Warm continuation, independent sessions with fixed outputs; histories grow by returned tokens between rounds.')
        save(directory/'summary.json',result)
        return result
    finally:
        for i,state in enumerate(states):
            destination=directory/f'release-c{i+1}';destination.mkdir(exist_ok=True)
            release_session(state['row'],destination)


def run_sweep(root,args,build):
    schedule=[tuple(map(int,item.split(':'))) for item in args.sweep_cases.split(',')]
    assert all(0<n<=args.seqs and length+1+args.decode_warmup+args.reps*args.decode_tokens<=args.max_len
        for length,n in schedule)
    summaries=[]
    for length,concurrency in schedule:
        summaries.append(capacity_case(root,length,concurrency,args,build))
        save(root/'summary.json',dict(cases=summaries))
