"""Finite native multi-owner RULER execution with independent per-case evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np

import recall_comparison_protocol as p
import ruler_protocol
import run_layerwise_query_diagnostic as serial
from benchmark_client_io import CaptureSlice
from benchmark_niah import CpuCredits
from vllm_kvmem.core.block_runs import candidate_starts


def query_mode(fixture):
    mode = fixture.get('performance_query_mode', 'sequential')
    if mode not in ('sequential', 'shared_global'):
        raise ValueError('Unknown Query execution mode')
    if mode == 'shared_global' and (not fixture.get('qualification_only') or
            not fixture.get('performance_comparison') or fixture['layerwise_prefill_pressure'] or
            fixture.get('layerwise_pressure_single_assembly') or
            fixture.get('layerwise_pressure_fast_metadata') or fixture.get('layerwise_packed_assembly') or
            fixture.get('layerwise_prepared_index') or fixture.get('layerwise_batched_reads')):
        raise ValueError('Shared Query is only an explicit shared-pressure performance control')
    return mode


def server_command(root, attempt, fixture):
    command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config', str(root/'profile.json'),
               '--run-dir', str(attempt/'server')]
    if query_mode(fixture) == 'sequential':
        command += ['--layerwise-query-diagnostic', 'sequential',
                    '--layerwise-query-optimized', '--layerwise-query-native-batches']
    if fixture['layerwise_prefill_pressure']:
        command += ['--layerwise-prefill-pressure']
    if fixture.get('layerwise_pressure_single_assembly'):
        command += ['--layerwise-pressure-single-assembly']
    if fixture.get('layerwise_pressure_fast_metadata'):
        command += ['--layerwise-pressure-fast-metadata']
    if fixture.get('layerwise_packed_assembly'):
        command += ['--layerwise-packed-assembly']
    if fixture.get('layerwise_prepared_index'):
        command += ['--layerwise-prepared-index']
    if fixture.get('layerwise_batched_reads'):
        command += ['--layerwise-batched-reads']
    if fixture.get('layerwise_paired_reads'):
        command += ['--layerwise-paired-reads']
    if fixture.get('layerwise_buffered_history'):
        if not fixture['layerwise_prefill_pressure']:
            raise ValueError('Buffered histories require independent prefill pressure')
        command += ['--layerwise-buffered-history']
    if fixture.get('rope_position_policy') == 'original':
        command += ['--layerwise-original-rope']
    return command


def audit_shared_events(events, bounds, response, row, profile):
    """Audit the actual native shared policy, with no per-layer diagnostic hook."""
    rid = response['response_id']
    if events or any(not e.get('request_id', '').startswith(rid+'-') for e in bounds):
        raise ValueError('Shared control contains layerwise work or foreign-owner evidence')
    kv = profile['kvmem']
    if (kv['pressure_policy'] != 'attention' or kv.get('query_score_mode', 'mean_k') != 'mean_k' or
            not kv.get('prefill_attention') or kv.get('graph_alpha', 0) or
            kv.get('query_mean_quota_fraction', 0) or kv.get('guided_query_tokens', 0)):
        raise ValueError('Shared control changed its declared recall policy')
    queries = [e for e in bounds if e['selection'] == 'query']
    pressures = [e for e in bounds if e['selection'] == 'pressure']
    if (len(queries) != 1 or queries[0]['query_span'] != row['served_query'] or
            queries[0].get('query_scoring_method') != 'mean_k' or not pressures):
        raise ValueError('Missing original Query or shared Mean-K/pressure evidence')
    for event in bounds:
        if event.get('fallback_reason') or event.get('guided_fallback') or event.get('cancelled'):
            raise ValueError('Shared control has a fallback or cancelled refresh')
        ranges = event['selected_history_ranges']
        begins = [begin for begin, _ in ranges]
        block = kv['block_tokens']
        if (len(begins) != len(set(begins)) or
                any(begin < 0 or begin % block or count <= 0 or count > block or
                    begin+count > event['history_begin'] for begin, count in ranges) or
                len(ranges)*block > kv['select_budget'] or
                sum(count for _, count in ranges) != event['selected_tokens'] or
                not set(range(0, kv['sink_blocks']*block, block)) <= set(begins)):
            raise ValueError('Shared control changed its history, sink or Active budget')
    if any(e.get('attention_source') != 'prefill' or e.get('prefill_attention_samples', 0) <= 0 or
           e.get('private_forward_steps') != 0 for e in pressures):
        raise ValueError('Shared pressure did not use accepted accumulated prefill attention')
    return []


def audit_events(events, bounds, response, row, independent):
    rid=response['response_id']
    if any(not e.get('request_id','').startswith(rid+'-') for e in events+bounds):
        raise ValueError('Foreign request leaked into per-case evidence')
    layers=[e for e in events if e['event']=='layer_selected']
    if [e['layer'] for e in layers]!=list(range(16)):
        raise ValueError('Missing ordered per-layer Query selections')
    if any(e['predecessor_layers_completed']!=e['layer'] or e['query_source']!='same_forward_current_layer' or
           e['charged_tokens']>36864 or e['selected_tokens']>36864 for e in layers):
        raise ValueError('Query order/source or per-layer budget violation')
    query=[e for e in bounds if e['selection']=='query']
    if len(query)!=1 or query[0]['query_span']!=row['served_query']:
        raise ValueError('Original explicit Query changed')
    pressures=[e for e in bounds if e['selection']=='pressure']
    actual=[e for e in events if e['event']=='pressure_selected']
    method='accumulated_attention_per_layer' if independent else 'accumulated_attention'
    if not pressures or len(pressures)!=len(actual) or any(e['actual_method']!=method for e in actual):
        raise ValueError('Actual pressure method differs or is missing')
    if independent:
        for event in actual:
            if event['event_type']!='prefill_pressure' or event['fallback_reason'] is not None:
                raise ValueError('Unexpected pressure entry or semantic fallback')
            if [e['layer'] for e in event['layers']]!=list(range(16)):
                raise ValueError('Missing independent pressure layer')
            for layer in event['layers']:
                if (layer['charged_tokens']>36864 or layer['selected_tokens']!=event['selected_tokens'] or
                    layer['accepted_score_end']>event['accepted_history'] or
                    not set(layer['selected_starts'])<=set(candidate_starts(layer)) or
                    not set(range(0,1312,32))<=set(layer['selected_starts'])):
                    raise ValueError('Pressure scope, accepted time or per-layer budget violation')
    return layers


def infer(root, capture):
    import qualify_graph_niah as client
    serial.verify(root)
    fixture,profile=p.read(root/'fixture.json'),p.read(root/'profile.json')
    workers=fixture['layerwise_workers'];rows=fixture['rows']
    if workers!=profile['max_num_seqs'] or not 1<=workers<=8:
        raise ValueError('Frozen concurrency does not match engine capacity')
    scorer=ruler_protocol.load_scorer(root/'upstream')
    out=root/'formal/sequential';(out/'cases').mkdir(parents=True,exist_ok=True)
    credits=CpuCredits(profile['kvmem']['cpu_read_cache_bytes']-16*1024**3)
    lock=Lock();results={};active=set()
    started=time.perf_counter();started_wall=time.time()
    def progress():
        p.save(root/'status.json',dict(status='running',phase='formal',arm='sequential',
            workers=workers,completed=len(results),expected=len(rows),active_samples=sorted(active),
            elapsed_seconds=time.perf_counter()-started,at=time.time()))
    def execute(row):
        # Full original history plus the original Gen reserve and staging.
        amount=(row['served_input_tokens']+2*16384+31)//32*32*17*4*256*2
        credits.acquire(row['id'],amount)
        try:
            case=out/'cases'/row['id'];case.mkdir(parents=True,exist_ok=False)
            with lock:active.add(row['id']);progress()
            input_file=root/'inputs'/(row['id']+'.i32')
            if p.sha(input_file)!=row['served_sha256']:raise ValueError('Original input changed')
            tokens=np.fromfile(input_file,dtype='<i4').tolist()
            sampling=p.case_sampling(fixture,row);session='layerwise-parallel-'+uuid.uuid4().hex
            p.save(case/'request.json',dict(sample_id=row['id'],session=session,input_sha256=row['served_sha256'],
                sampling=sampling,query_span=row['served_query'],fixture_sha256=p.sha(root/'fixture.json')))
            cursor=CaptureSlice(capture/name for name in ['layerwise_query.jsonl','bounded_refresh.jsonl'])
            response=client.stream(profile,session,tokens,row['served_query'],sampling,3600)
            p.save(case/'response.json',response)
            terminal=client.wait_retired(capture,response['response_id'])
            if terminal['history_end']!=len(tokens)+len(response['output_ids'])-1:
                raise ValueError('Original input/output acceptance accounting changed')
            released=client.wait_owner_released(capture,session,response,terminal)
            rid=response['response_id']
            events=[e for e in cursor.read(capture/'layerwise_query.jsonl') if e.get('request_id','').startswith(rid+'-')]
            bounds=[e for e in cursor.read(capture/'bounded_refresh.jsonl') if e.get('request_id','').startswith(rid+'-')]
            if query_mode(fixture) == 'shared_global':
                layers=audit_shared_events(events,bounds,response,row,profile)
            else:
                layers=audit_events(events,bounds,response,row,fixture['layerwise_prefill_pressure'])
            if fixture.get('rope_position_policy') == 'original':
                if any(l.get('rope_position_policy') != 'original' or
                       l.get('query_rope_span') != row['served_query'] or
                       l.get('query_rope_verified') is not True for l in layers):
                    raise ValueError('Original Query RoPE evidence is missing')
                rope_events=[e for e in events if e['event']=='pressure_rope']
                pressure_events=[e for e in events if e['event']=='pressure_selected']
                if ([e['accepted_history'] for e in rope_events] !=
                        [e['accepted_history'] for e in pressure_events] or
                        any(e['rope_position_policy']!='original' for e in rope_events)):
                    raise ValueError('Original pressure RoPE evidence is missing')
            p.save(case/'evidence.json',dict(layers=layers,events=events,refresh=bounds,
                terminal=terminal,release=released,mode='sequential',workers=workers))
            score=p.score_case(row,response,scorer,
                starts_in_thinking=fixture.get('starts_in_thinking', False))
            result=dict(id=row['id'],task=row['task'],index=row['index'],arm='sequential',phase='formal',
                input_sha256=row['served_sha256'],response_sha256=p.sha(case/'response.json'),
                evidence_sha256=p.sha(case/'evidence.json'),text=response['text'],
                finish_reason=response['finish_reason'],seconds=response['seconds'],ttft_seconds=response['ttft_seconds'],
                completion_tokens=len(response['output_ids']),max_tokens=sampling['max_tokens'],**score)
            p.save(case/'result.json',result)
            with lock:
                results[row['id']]=result;active.remove(row['id']);progress()
                print(json.dumps(dict(id=row['id'],completed=len(results),expected=len(rows),
                    seconds=result['seconds'],official_case_score=result['official_case_score'])),flush=True)
        except BaseException:
            credits.cancel();raise
        finally:credits.release(row['id'],amount)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,row) for row in rows]
        for future in as_completed(futures):future.result()
    elapsed=time.perf_counter()-started
    p.save(root/'concurrency-performance.json',dict(workers=workers,cases=len(results),
        started_at=started_wall,completed_at=time.time(),inference_wall_seconds=elapsed,
        started_monotonic=started,
        original_input_tokens=sum(r['served_input_tokens'] for r in rows),
        input_tokens_per_second=sum(r['served_input_tokens'] for r in rows)/elapsed,
        cases_per_hour=3600*len(results)/elapsed,cpu_admission_peak=credits.peak,
        cpu_admission_budget=credits.budget,cpu_admission_events=list(credits.events),
        scope='Whole concurrent inference, including preparation, pressure, Query, output and evidence; excludes model load.'))
    if not fixture.get('qualification_only', False) and len(rows)==13*fixture.get('samples_per_task', 2):
        serial.analyze_balanced26(root)
    else:
        (root/'analysis').mkdir(exist_ok=True)
        p.save(root/'analysis/qualification.json',dict(cases=len(rows),independent_first_answers=len(results),
            layer_pressure=fixture['layerwise_prefill_pressure'],workers=workers,
            scores={k:v['official_case_score'] for k,v in results.items()},
            methods_inputs_query_and_budgets_verified=True))


def run(root):
    sys.path[:0]=[str(root/'runtime-v1/src'),str(root/'runtime-v1/scripts')]
    from run_recall_comparison import wait_health,stop_owned,gpu_lock,termination_guard
    import benchmark_block_graph as lifecycle
    serial.verify(root)
    fixture,profile=p.read(root/'fixture.json'),p.read(root/'profile.json')
    if any((root/'formal/sequential/cases').glob('*/response.json')):
        raise ValueError('Preserve first answers; recovery requires a separate explicit version')
    env={k:v for k,v in os.environ.items() if not k.startswith(('KVMEM_','VLLM_'))}
    env.update(PYTHONPATH=str(root/'runtime-v1/src'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda',
        PATH='/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+os.environ['PATH'])
    with termination_guard(),gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        if lifecycle.gpu_processes():raise RuntimeError('GPU has another owner')
        attempts=root/'formal/sequential/server-attempts';attempts.mkdir(parents=True,exist_ok=True)
        attempt=attempts/f'{len(list(attempts.iterdir()))+1:04d}';attempt.mkdir()
        capture=attempt/'server/capture'
        command=server_command(root,attempt,fixture)
        server=observer=None
        try:
            p.save(root/'status.json',dict(status='loading',workers=fixture['layerwise_workers'],at=time.time()))
            with (attempt/'server.log').open('x') as log:
                server=subprocess.Popen(command,cwd=root/'runtime-v1',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            p.save(root/'process.json',dict(pid=os.getpid(),server_pid=server.pid,command=command,at=time.time()))
            wait_health(server,profile)
            if fixture.get('performance_comparison'):
                observer=subprocess.Popen([sys.executable,'-B',
                    str(root/'runtime-v1/scripts/gpu_efficiency_monitor.py'),'sample',
                    '--output',str(attempt/'gpu.jsonl'),'--interval','1',
                    '--owned-root-pid',str(server.pid)],cwd=root/'runtime-v1',env=env,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            infer(root,capture);serial.verify(root)
            p.save(root/'status.json',dict(status='completed',workers=fixture['layerwise_workers'],cases=len(fixture['rows']),at=time.time()))
        except BaseException as error:
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
        finally:
            if observer is not None:
                observer.terminate()
                try:observer.wait(timeout=15)
                except subprocess.TimeoutExpired:observer.kill();observer.wait()
            stop_owned(server)
            p.save(attempt/'cleanup.json',dict(pid=server.pid if server else None,returncode=server.returncode if server else None,at=time.time()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root.resolve())
