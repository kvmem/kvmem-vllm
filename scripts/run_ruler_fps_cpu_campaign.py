"""Own a frozen 130-case RULER CPU-FPS4/halo1 campaign; reject unsafe resume."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from threading import Barrier
import time
from types import SimpleNamespace as NS
import uuid

import benchmark_block_graph as lifecycle
from benchmark_ruler import save, sha, summarize_result
import qualify_block_graph as q
import qualify_graph_niah as client
import ruler_fps_cpu_protocol as policy
import ruler_protocol as ruler

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'artifacts/ruler-v1-256k-query-replay-20261022-r1'
ROOT=Path('/data/chaidi/kvmem-vllm-experiments/ruler-fps-native-cpu-halo-256k-20261023-r1')
require=policy.require
CASE_FILES=('request.json','response.json','result.json','evidence.json','release.json','query-inputs.jsonl','native-stats.json',
    'client-timing.json','metrics.json')


def read(path):return json.loads(Path(path).read_text())


def verify_files(root,files):
    for name,digest in files.items():require(sha(Path(root)/name)==digest,'Frozen file changed: '+str(Path(root)/name))


def prepare(root,source,*,qualification=False):
    require(not root.exists(),'Use a fresh campaign root; successful answers and failed attempts cannot be overwritten')
    original=read(source/'fixture.json');reference=read(source/'profile.json')
    require(sha(source/'fixture.json')==read(source/'dataset-summary.json')['fixture_sha256'],'Original 260 fixture proof differs')
    require(sha(source/'profile.json')==original['profile_sha256'],'Original profile proof differs')
    profile=policy.expected_profile(reference,qualification=qualification)
    selected=policy.qualification_rows(original) if qualification else policy.select_rows(original)
    cohort=(dict(cases=8,samples_per_task=None,tasks=8,official_default_sample_count=False) if qualification
        else ruler.validate_cohort(selected,10))
    fixture={**original,**cohort,
        'rows':selected,'execution_profile':policy.QUALIFICATION_PROFILE if qualification else policy.PROFILE,
        'reference_profile':reference,'source_fixture_sha256':sha(source/'fixture.json'),
        'source_root':str(source),'source_selection':('First eight original cases in the full campaign admission order; '
            'one qualification wave, no answer-based selection or full-RULER score.' if qualification else
            'Original index 0..9 of each of the 13 RULER v1 tasks; no answer-based selection.')}
    budget=policy.validate_execution(profile,fixture)
    from vllm_kvmem.serve import build_launch
    build_launch(profile,REPO,root/'preflight-only')
    ruler.load_scorer(Path(fixture['upstream_path']))
    require(sha(Path(profile['tokenizer'])/'tokenizer.json')==fixture['tokenizer_sha256'],'Original tokenizer changed')
    inputs={}
    for row in fixture['rows']:
        path=Path(fixture['input_directory'])/(row['id']+'.i32')
        require(path.stat().st_size==row['served_input_tokens']*4 and sha(path)==row['served_sha256'],'Original full input changed')
        inputs[row['id']]=dict(path=str(path),sha256=row['served_sha256'],tokens=row['served_input_tokens'],seed=row['seed'])
    source_proof=dict(root=str(source),files={name:sha(source/name) for name in ('fixture.json','profile.json','dataset-summary.json')})
    root.mkdir(parents=True);runtime=root/'runtime'
    for name in ('src/vllm_kvmem','scripts','csrc','third_party/qw3'):
        shutil.copytree(REPO/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','setup.py'):shutil.copy2(REPO/name,runtime/name)
    save(root/'runtime-proof.json',dict(files={str(path.relative_to(runtime)):sha(path)
        for path in runtime.rglob('*') if path.is_file()}))
    save(root/'profile.json',profile);fixture['profile_sha256']=sha(root/'profile.json')
    save(root/'fixture.json',fixture);save(root/'source-proof.json',source_proof)
    save(root/'input-proof.json',inputs);save(root/'budget-audit.json',budget)
    save(root/'dataset-summary.json',dict(**cohort,target_length=262144,
        fixture_sha256=sha(root/'fixture.json'),input_count=len(inputs),selection=fixture['source_selection']))
    names=('profile.json','fixture.json','source-proof.json','input-proof.json','budget-audit.json','dataset-summary.json','runtime-proof.json')
    save(root/'prepared-proof.json',{name:sha(root/name) for name in names})
    save(root/'status.json',dict(status='prepared',cases=fixture['cases'],at=time.time()))


def verify(root):
    verify_files(root,read(root/'prepared-proof.json'))
    verify_files(root/'runtime',read(root/'runtime-proof.json')['files'])
    source=read(root/'source-proof.json');verify_files(Path(source['root']),source['files'])
    fixture,profile=read(root/'fixture.json'),read(root/'profile.json')
    policy.validate_execution(profile,fixture)
    require(sha(root/'profile.json')==fixture['profile_sha256'],'Frozen profile differs')
    require(sha(Path(profile['tokenizer'])/'tokenizer.json')==fixture['tokenizer_sha256'],'Tokenizer bytes changed')
    ruler.load_scorer(Path(fixture['upstream_path']))
    for record in read(root/'input-proof.json').values():
        path=Path(record['path']);require(path.stat().st_size==record['tokens']*4 and sha(path)==record['sha256'],'Fixed input changed')
    return fixture,profile


def assert_frozen_controller(root):
    require(Path(__file__).resolve()==root/'runtime/scripts'/Path(__file__).name,'Run the frozen controller only')


def actual_launch(profile,capture):
    audit=client.audit_launch(profile,capture)
    launch=read(capture.parent/'launch.json');argv=launch['vllm_argv']
    payload=json.loads(argv[argv.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
    names=('query_score_mode','query_fps_halo_blocks','query_fps_optimized','query_fps_prebuild',
        'query_fps_prebuild_device','query_fps_cpu_backend','query_fps_cpu_workers',
        'query_window_cache_bytes','graph_query_trace_max_bytes')
    require(all(payload[name]==profile['kvmem'][name] for name in names),'Actual worker FPS configuration differs')
    return dict(audit,actual_fps={name:payload[name] for name in names})


def score_results(fixture,results,scorer):
    require({row['id'] for row in results}=={row['id'] for row in fixture['rows']} and
        len(results)==fixture['cases'],'Score only the immutable full requested cohort')
    if fixture['execution_profile']!=policy.QUALIFICATION_PROFILE:
        predictions={(row['task'],row['index']):row['text'] for row in results}
        return ruler.score_suite(fixture['rows'],predictions,scorer,samples_per_task=10,starts_in_thinking=True)
    # Each final answer is still graded by the same official task function.
    # This subset does not cover 13 tasks and is never labelled a RULER macro.
    return dict(cases=8,qualification_only=True,full_ruler_score=False,
        scoring_scope='Eight fixed performance qualification cases only; original official per-case scoring, no accuracy gate.',
        macro_score=statistics.fmean(row['official_case_score'] for row in results),
        per_task={row['task']:dict(cases=1,score=row['official_case_score']) for row in results})


def performance_summary(metrics,completed,*,qualification):
    require(len(metrics)==completed['cases'],'Missing per-case phase measurements')
    if qualification:require(completed['queue_peak_inflight']==8,'Qualification did not admit eight original requests')
    prebuilt=sum(row['cpu_fps_audit']['cache_prebuilt_hit_blocks'] for row in metrics)
    native=sum(row['cpu_fps_audit']['query_native']['fps_cpu_native_groups']+
        row['cpu_fps_audit']['background_native']['fps_cpu_native_groups'] for row in metrics)
    require(prebuilt>0 and native>0,'Native CPU preparation or prebuilt-cache use did not actually execute')
    begin=min(row['client_started_at'] for row in metrics)
    end=max(row['client_started_at']+row['request_seconds'] for row in metrics)
    fields=('request_seconds','ttft_seconds','client_to_query_seconds','query_refresh_seconds',
        'query_to_first_token_seconds','decode_stream_seconds')
    phases={name:dict(median=statistics.median(row[name] for row in metrics),
        minimum=min(row[name] for row in metrics),maximum=max(row[name] for row in metrics)) for name in fields}
    return dict(cases=len(metrics),concurrent_admission_peak=completed['queue_peak_inflight'],
        request_wall_seconds=end-begin,input_tokens=completed['input_tokens'],
        completion_tokens=completed['completion_tokens'],phases_seconds=phases,
        native_cpu_groups=native,prebuilt_hit_blocks=prebuilt,
        query_miss_blocks=sum(row['cpu_fps_audit']['cache_rebuilt_blocks'] for row in metrics),
        ambiguity_reference_groups=sum(row['cpu_fps_audit'][where]['fps_cpu_ambiguity_groups']
            for row in metrics for where in ('query_native','background_native')),
        timing_scope='Client-to-Query includes admission, prefill and maintenance; Query includes replay, scoring, '
            'picker and assembly. Request wall span includes mixed prefill/decode; no pure prefill throughput or '
            'causal speedup is inferred. Original generated trajectories/output token counts are retained.',
        no_accuracy_gate=True)


def run_client(root):
    import numpy as np
    from tokenizers import Tokenizer
    from benchmark_client_io import CaptureSlice
    from vllm_kvmem.offline_queue import OfflineQueue,OfflineTask
    assert_frozen_controller(root);fixture,profile=verify(root)
    output=root/'run/answers';output.mkdir(parents=True,exist_ok=False)
    capture=root/'run/server/capture';save(output/'launch-audit.json',actual_launch(profile,capture))
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    scorer=ruler.load_scorer(Path(fixture['upstream_path']))
    rows=policy.execution_rows(fixture['rows']);expected_cases=fixture['cases']
    barrier=Barrier(8) if fixture['execution_profile']==policy.QUALIFICATION_PROFILE else None
    queue=OfflineQueue([OfflineTask(row['id'],row,client.history_bytes(row,fixture,profile)) for row in rows],
        max_inflight=8,history_capacity_bytes=fixture['cpu_payload_credits_bytes'])
    traces=policy.QueryTraceIndex(capture/'block_graph_inputs.jsonl',output/'query-traces',
        max_records=expected_cases,max_bytes=profile['kvmem']['graph_query_trace_max_bytes'])

    def infer(task):
        row=task.payload;directory=output/'cases'/row['id'];directory.mkdir(parents=True,exist_ok=False)
        try:
            source=Path(fixture['input_directory'])/(row['id']+'.i32')
            require(sha(source)==row['served_sha256'],'Original prompt bytes changed before dispatch')
            tokens=np.fromfile(source,dtype='<i4').tolist()
            require(len(tokens)==row['served_input_tokens'] and
                tokenizer.decode(tokens[slice(*row['served_query'])],skip_special_tokens=False)==row['question'],
                'Public Query coordinates differ from original input')
            session='ruler-fps-cpu-'+uuid.uuid4().hex;sampling=dict(fixture['sampling'],seed=row['seed'])
            save(directory/'request.json',dict(sampling=sampling,session_id=session,prompt_sha256=row['served_sha256'],
                prompt_tokens=len(tokens),query_span=row['served_query'],profile_sha256=sha(root/'profile.json')))
            cursor=CaptureSlice(capture/name for name in ('bounded_refresh.jsonl','block_graph.jsonl','action_attention.jsonl'))
            if barrier is not None:barrier.wait(timeout=120)
            began=time.monotonic()
            response=client.stream(profile,session,tokens,row['served_query'],sampling,fixture['request_timeout_seconds'])
            finished=time.monotonic();aligned_start=finished-response['seconds']
            save(directory/'response.json',response)
            save(directory/'client-timing.json',dict(started_at=aligned_start,invoked_at=began,finished_at=finished,
                origin_uncertainty_upper_bound_seconds=max(0.,aligned_start-began)))
            return session,tokens,response,cursor
        except BaseException as error:
            save(directory/'failure.json',dict(error=repr(error),stage='inference',at=time.time()));raise

    def finish(task,pending):
        row=task.payload;directory=output/'cases'/row['id'];session,tokens,response,cursor=pending
        try:
            terminal=client.wait_retired(capture,response['response_id'])
            require(terminal['history_end']==len(tokens)+len(response['output_ids'])-1 and not terminal.get('checkpoint_tokens'),
                'Accepted original history/retirement differs')
            release=client.wait_owner_released(capture,session,response,terminal)
            query=traces.take(response,directory/'query-inputs.jsonl')
            reader=lambda path:(query,) if Path(path).name=='block_graph_inputs.jsonl' else cursor.read(path)
            evidence=policy.audit_case(capture,response,profile,row['served_query'],event_reader=reader)
            action=[event for event in cursor.read(capture/'action_attention.jsonl') if q.native_matches(response,event)]
            expected=q.expected_output_samples(NS(convert_tokens_to_ids=tokenizer.token_to_id),profile,tokens,response['output_ids'])
            require(len(action)==1 and action[0]['samples']==expected,'Rejected/private/thinking rows contaminated accepted attention')
            save(directory/'evidence.json',dict(evidence,terminal=terminal,action=action,
                client_capture_io=dict(bytes_read=cursor.bytes_read,records_read=cursor.records_read)))
            save(directory/'release.json',release)
            from benchmark_matched_prefill import audit_native_stats
            native=read(capture/'native_attention_stats.json');audit_native_stats(native);save(directory/'native-stats.json',native)
            result=summarize_result(row,session,len(tokens),response,evidence,scorer,starts_in_thinking=fixture['starts_in_thinking'])
            from benchmark_query_fps_prebuild_live import phase_metrics
            metrics=phase_metrics(response,evidence,client_started_at=read(directory/'client-timing.json')['started_at'])
            metrics['cpu_fps_audit']=evidence['cpu_fps_audit'];save(directory/'metrics.json',metrics)
            result['cpu_fps_audit']=evidence['cpu_fps_audit'];save(directory/'result.json',result)
            return result
        except BaseException as error:
            save(directory/'failure.json',dict(error=repr(error),stage='retirement/audit',at=time.time()));raise

    observed=[]
    def on_result(result):
        state=queue.states[result['id']];result.update(cpu_queue_seconds=state['admitted_at']-state['queued_at'],
            retirement_seconds=state['completed_at']-state['inference_done_at'])
        directory=output/'cases'/result['id'];save(directory/'result.json',result)
        save(directory/'hashes.json',{name:sha(directory/name) for name in CASE_FILES})
        observed.append(result)
        save(output/'progress.json',dict(completed=len(observed),expected=expected_cases,at=time.time(),
            per_task={task:dict(completed=sum(r['task']==task for r in observed),
                score=sum(r['official_case_score'] for r in observed if r['task']==task)/max(1,sum(r['task']==task for r in observed)))
                for task in ruler.TASK_FAMILIES}))
        print(json.dumps(dict(completed=len(observed),id=result['id'],score=result['official_case_score'])),flush=True)
    def failed(error):save(output/'failure.json',dict(error=repr(error),completed=len(observed),at=time.time()))
    def state(value):save(output/'queue.json',value)
    started=time.monotonic()
    results=queue.run(infer,finish,on_result=on_result,on_failure=failed,on_state=state)
    require(len(results)==expected_cases and queue.reserved_bytes==queue.inflight==0,'Incomplete cases or leaked client admission credits')
    released=q.wait_released(capture,[row['session'] for row in results],timeout=180)
    require(released['archive'].get('disk_read_bytes',0)==released['archive'].get('disk_write_bytes',0)==0,'CPU history used SSD')
    trace_audit=traces.audit();save(output/'trace-audit.json',trace_audit)
    scores=score_results(fixture,results,scorer)
    save(output/'scores.json',scores)
    save(output/'completed.json',dict(status='completed',cases=expected_cases,seconds=time.monotonic()-started,
        score=scores['macro_score'],queue_peak_inflight=queue.peak_inflight,queue_peak_bytes=queue.peak_bytes,
        results=results,fixture_sha256=sha(root/'fixture.json'),protocol_passed=True,
        original_inputs_seeds_sampling=True,all_owned_sessions_released=True,
        input_tokens=sum(row['input_tokens'] for row in results),completion_tokens=sum(row['completion_tokens'] for row in results)))


def audit_execution(root):
    import numpy as np
    from tokenizers import Tokenizer
    fixture,profile=verify(root);capture=root/'run/server/capture';output=root/'run/answers'
    scorer=ruler.load_scorer(Path(fixture['upstream_path']));hashes={};cpu=[];results={};metrics=[]
    expected_cases=fixture['cases'];qualification=fixture['execution_profile']==policy.QUALIFICATION_PROFILE
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    for row in fixture['rows']:
        directory=output/'cases'/row['id'];manifest=read(directory/'hashes.json')
        require(set(CASE_FILES)<=set(manifest),'Case evidence hashes incomplete');verify_files(directory,manifest)
        request,response,result,evidence=(read(directory/name) for name in ('request.json','response.json','result.json','evidence.json'))
        require(request['sampling']==dict(fixture['sampling'],seed=row['seed']) and request['prompt_sha256']==row['served_sha256'] and
            request['query_span']==row['served_query'] and request['profile_sha256']==fixture['profile_sha256'],
            'Final request differs from original seed/input/sampling')
        by_name={'bounded_refresh.jsonl':evidence['bounded'],'block_graph.jsonl':evidence['graph'],
            'block_graph_inputs.jsonl':evidence['query_inputs']}
        replay=policy.audit_case(capture,response,profile,row['served_query'],event_reader=lambda path:by_name[Path(path).name])
        recorded=deepcopy(evidence['selection_audit']);actual=deepcopy(replay['selection_audit'])
        for item in (recorded,actual):item['propagation'].pop('scoring_seconds',None)
        require(recorded==actual and replay['cpu_fps_audit']==evidence['cpu_fps_audit'],'Final native picker/CPU construction audit changed')
        rerun=summarize_result(row,request['session_id'],row['served_input_tokens'],response,evidence,scorer,starts_in_thinking=True)
        require(all(result[key]==value for key,value in rerun.items()),'Saved official result differs from original response')
        require(json.loads((directory/'query-inputs.jsonl').read_text())==evidence['query_inputs'][0], 'Request-scoped trace differs')
        require(read(directory/'release.json')['released'] is True,'Original owner release missing')
        require(evidence['terminal']['history_end']==row['served_input_tokens']+len(response['output_ids'])-1 and
            not evidence['terminal'].get('checkpoint_tokens'),'Final accepted history/terminal differs')
        tokens=np.fromfile(Path(fixture['input_directory'])/(row['id']+'.i32'),dtype='<i4').tolist()
        accepted=q.expected_output_samples(NS(convert_tokens_to_ids=tokenizer.token_to_id),profile,tokens,response['output_ids'])
        require(len(evidence['action'])==1 and evidence['action'][0]['samples']==accepted,'Final accepted-row attention audit differs')
        from benchmark_query_fps_prebuild_live import phase_metrics
        timing=read(directory/'client-timing.json')
        metric=phase_metrics(response,evidence,client_started_at=timing['started_at'])
        metric['cpu_fps_audit']=evidence['cpu_fps_audit']
        require(metric==read(directory/'metrics.json'),'Final phase metrics differ from saved response/evidence')
        metrics.append(metric);cpu.append(evidence['cpu_fps_audit'])
        results[row['id']]=result
        hashes.update({str((directory/name).relative_to(root)):digest for name,digest in manifest.items()})
    scores=score_results(fixture,list(results.values()),scorer)
    require(scores==read(output/'scores.json'),'Final official task scores changed')
    completed=read(output/'completed.json')
    require(completed['status']=='completed' and completed['cases']==expected_cases and completed['protocol_passed'] is True and
        completed['fixture_sha256']==sha(root/'fixture.json') and completed['score']==scores['macro_score'] and
        len(completed['results'])==expected_cases and {row['id']:row for row in completed['results']}==results,
        'Completed campaign summary differs from immutable per-case results')
    trace=read(output/'trace-audit.json')
    require(trace['records']==expected_cases and trace['maximum_bytes']==policy.TRACE_BYTES and
        trace['bytes']==(capture/'block_graph_inputs.jsonl').stat().st_size<=policy.TRACE_BYTES,
        'Final shared Query trace count/capacity differs')
    from ruler_pooling_protocol import audit_optimized
    execution=audit_optimized(capture,profile)
    actions=q.records(capture/'action_attention.jsonl');require(len(actions)==expected_cases,'Missing accepted-attention reports')
    require(all(row.get('decode_commit_python_calls',0)==0 and row.get('download_pool_total') and
        not row['download_pool_total']['exhaustions'] and not row['download_pool_total']['quarantined_slots'] for row in actions),
        'Unexpected native acceptance/download resource failure')
    cleanup=read(root/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],'Owned GPU cleanup incomplete')
    performance=performance_summary(metrics,completed,qualification=qualification)
    save(root/'performance.json',performance)
    audit=dict(status='passed',cases=expected_cases,case_hashes=hashes,source_and_original_inputs_unchanged=True,
        official_scores_recomputed=True,native_group_picker_replay=True,pressure_cumulative_attention=True,
        construction_device='cpu',cpu_backend='native',cpu_workers=profile['kvmem']['query_fps_cpu_workers'],
        query_gpu_groups=0,background_gpu_groups=0,native_cpu_groups=performance['native_cpu_groups'],
        ambiguity_reference_groups=performance['ambiguity_reference_groups'],pool_busy_groups=0,unsupported_fallback_groups=0,
        prebuilt_hit_blocks=sum(row['cache_prebuilt_hit_blocks'] for row in cpu),
        query_miss_blocks=sum(row['cache_rebuilt_blocks'] for row in cpu),execution=execution,
        budget=read(root/'budget-audit.json'),gpu_cleanup=True)
    audit['query_trace_sha256']=sha(capture/'block_graph_inputs.jsonl')
    save(root/'execution-audit.json',audit);save(root/'scores.json',scores)
    lines=['# RULER v1 256K：CPU FPS4＋预算内左右邻块分组召回','',
        (f"固定前 8 题性能资格完成；逐题官方分数平均 **{scores['macro_score']:.2f}%**，不是完整 RULER 分数。" if qualification else
        f"13 类各原前 10 题，共 130 题；官方任务等权平均 **{scores['macro_score']:.2f}%**。"),'',
        f"代表 Key 全部由原生 CPU 构建池处理（含 Query 缺失补建，{profile['kvmem']['query_fps_cpu_workers']} workers；有歧义的组明确回退到参考实现）；Query 评分使用 FPS4、α=0、halo1 原生组 picker。",
        'Prefill pressure 仍为累计 attention，guided 关闭。K=36,864，G=16,384，sink=1,312，recent=0。',
        ('使用原始完整输入与种子、8192 输出及4096思考上限；这是一次固定8路性能资格，不作完整质量结论。' if qualification else
        '使用原始完整输入与种子、8192 输出及4096思考上限；每类10题是小样本筛查。'),
        '评分为官方最终答案子串规则，不能将回答正确等同于 KV 完整召回。','',
        '| 任务 | 题数 | 官方分数 |','|---|---:|---:|']
    lines += [f"| {task} | {score['cases']} | {score['score']:.2f}% |" for task,score in scores['per_task'].items()]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return scores


def run(root):
    assert_frozen_controller(root);fixture,profile=verify(root)
    expected_cases=fixture['cases']
    with (root/'attempt.json').open('x') as stream:json.dump(dict(pid=os.getpid(),at=time.time(),resume_supported=False),stream)
    target=root/'run';target.mkdir(exist_ok=False);server=worker=None
    from benchmark_query_fps_prebuild_live import server_environment,wait_health
    env=server_environment(root/'runtime');started=time.monotonic()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock,lifecycle.cleanup_on_termination():
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not lifecycle.gpu_processes(),'Foreign GPU owner; do not stop or replace it')
            try:
                command=[sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(root/'profile.json'),'--run-dir',str(target/'server')]
                with (target/'server.log').open('x') as log:
                    server=subprocess.Popen(command,cwd=root/'runtime',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(target/'server-process.json',dict(pid=server.pid,command=command));save(root/'status.json',dict(status='loading',cases=expected_cases,at=time.time()))
                wait_health(server,profile);ready=time.monotonic()
                command=[sys.executable,'-B',str(root/'runtime/scripts'/Path(__file__).name),'client','--root',str(root)]
                with (target/'client.log').open('x') as log:
                    worker=subprocess.Popen(command,cwd=root/'runtime',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(target/'client-process.json',dict(pid=worker.pid,command=command))
                save(root/'status.json',dict(status='testing',cases=expected_cases,server_pid=server.pid,client_pid=worker.pid,
                    cold_prepare_seconds=ready-started,at=time.time()))
                with (target/'resources.jsonl').open('x') as telemetry:
                    while worker.poll() is None:
                        require(server.poll() is None,'Model exited during RULER inference')
                        require(not (target/'answers/failure.json').exists(),'Client failed; inspect preserved failure/evidence')
                        memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                            if line.startswith(('MemAvailable:','MemTotal:'))}
                        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'],text=True).strip()
                        telemetry.write(json.dumps(dict(at=time.time(),cpu=memory,gpu=gpu))+'\n');telemetry.flush();time.sleep(5)
                require(worker.returncode==0,'Client exited with failure; no automatic resume or regeneration')
                done=read(target/'answers/completed.json');require(done['status']=='completed' and done['cases']==expected_cases,'Incomplete original cohort')
            finally:
                remaining=lifecycle.stop_owned_groups((worker,server));gpu=sorted(lifecycle.gpu_processes())
                save(target/'cleanup.json',dict(remaining_owned_groups=remaining,gpu_processes=gpu))
                require(not remaining and not gpu,'Owned campaign GPU cleanup incomplete')
        scores=audit_execution(root)
        save(root/'status.json',dict(status='completed',cases=expected_cases,score=scores['macro_score'],
            cold_prepare_seconds=ready-started,total_seconds=time.monotonic()-started,at=time.time()))
    except BaseException as error:
        save(root/'failure.json',dict(error=repr(error),at=time.time(),resume_supported=False))
        save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','client'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root,args.source.resolve())
    elif args.action=='client':run_client(root)
    else:run(root)
