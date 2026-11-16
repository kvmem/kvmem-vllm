"""Independent full MAB Mean-K/layerwise runs and blinded assistant-judge packets.

No network judge, no answer-based retries, no scheduled monitor. Model serving
and cleanup reuse the last completed qualified MAB runtime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
import time
import urllib.request
import uuid

from prepare_full_mab_comparison import read, save, sha, digest, token_sha
from benchmark_client_io import compact_response
REPO=next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
BASE=REPO/'artifacts/mab-pressure-multiowner-20261105-r10/formal'
ARMS=('mean_k','layerwise')
STAGES=('qualification_mean_k','qualification_layerwise','formal_mean_k','formal_layerwise')


def explicit_execution(fixture):
    """Optional frozen execution flags; model, sampling and budgets stay in profile."""
    env=dict(fixture.get('execution_environment',{}))
    boolean={'KVMEM_LAYERWISE_PRESSURE_'+name for name in (
        'PIPELINE','PIPELINE_HANDOFF','PIPELINE_PREFORWARD','PIPELINE_PROGRESS','PREPARED_COMMIT',
        'PARENT_STAGING','BATCH_SELECTION','PAIRED_GATHER','RESIDENT_VALUES','PREPARED_READS',
        'REUSE_LAYER_VIEWS','FAST_METADATA','POST_FORWARD_REAP','SUBMIT_ON_PREPARE',
        'SINGLE_ASSEMBLY','SELECTION_YIELD','BATCH_ACCEPT')}
    boolean.update(('KVMEM_WORKER_STATIC_GC','KVMEM_QUERY_STAGE_PROFILE',
                    'KVMEM_LAYERWISE_QUERY_RECORD_INDEX','KVMEM_LAYERWISE_QUERY_PAIRED_GATHER',
                    'KVMEM_LAYERWISE_QUERY_LAZY_READER','KVMEM_LAYERWISE_QUERY_RESIDENT_V'))
    allowed={key:('0','1') for key in boolean}
    allowed.update(KVMEM_LAYERWISE_QUERY_SCORE_BACKEND=('legacy','tiled'),
                   KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS=('1','2'))
    if any(key not in allowed or value not in allowed[key] for key,value in env.items()):
        raise ValueError('Undeclared full-MAB execution control')
    if fixture['arm']=='mean_k' and any(env.get(k)=='1' for k in boolean if 'PRESSURE_' in k):
        raise ValueError('Pure Mean-K cannot enable layerwise pressure execution')
    return env


def tree(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def method_profile(original,arm):
    profile=deepcopy(original);kv=profile['kvmem']
    profile.update(port=19036,served_model_name='mab-full-paired',max_model_len=2097152,full_history_benchmark=True)
    kv['repository_blocks']=67584  # Complete 2M logical input plus 2*Gen, no historical deletion.
    if arm=='mean_k':
        kv.update(action_attention=False,prefill_attention=False,pressure_policy='recency',
            graph_mode='off',graph_alpha=0.,action_native_scores='off',
            prefill_publication_pipeline=False,prefill_native_plan_arm=False,
            prefill_action_before_mtp=False,prefill_physical_coalescing=False,
            prefill_physical_multi_owner=False,action_native_prefill_static_aligned=False)
    elif arm!='layerwise':raise ValueError(arm)
    return profile


def prepare(root):
    data=read(root/'data/cohort.json');assert data['cases']==2071 and data['contexts']==130
    assert tree(BASE/'runtime-v1')==read(BASE/'runtime-proof.json')
    assert not (root/'plan.json').exists(),'Do not overwrite a frozen comparison'
    runtime=root/'runtime-v1';shutil.copytree(BASE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Minimal opt-in launch guard extension; no inference/retrieval/kernel files change.
    path=runtime/'src/vllm_kvmem/serve.py';code=path.read_text()
    marker="    'model_adapter': None,\n";assert code.count(marker)==1
    code=code.replace(marker,marker+"    'full_history_benchmark': False,\n")
    marker="    if cfg['max_model_len'] > 1048576:\n";assert code.count(marker)==1
    code=code.replace(marker,"    if type(cfg['full_history_benchmark']) is not bool:\n        raise ValueError('full_history_benchmark must be boolean')\n    if cfg['full_history_benchmark']:\n        from .full_history_benchmark import validate_full_history\n        validate_full_history(cfg, options)\n    if cfg['max_model_len'] > 1048576 and not cfg['full_history_benchmark']:\n")
    marker="        env['KVMEM_LOGICAL_KV_CAPACITY'] = '1'\n";assert code.count(marker)==1
    code=code.replace(marker,marker+"    if cfg['full_history_benchmark']:\n        env['KVMEM_FULL_HISTORY_BENCHMARK'] = '1'\n")
    path.write_text(code)
    path=runtime/'src/vllm_kvmem/logical_capacity.py';code=path.read_text()
    replacements={
        "    if not recall and config.model_config.max_model_len <= 1048576:\n":
        "    finite_history = os.getenv('KVMEM_FULL_HISTORY_BENCHMARK') == '1'\n    if not recall and not finite_history and config.model_config.max_model_len <= 1048576:\n",
        "    elif gc is None or gc.mode != 'delete':\n":
        "    elif finite_history:\n        from .full_history_benchmark import validate_runtime_capacity\n        validate_runtime_capacity(config, options)\n    elif gc is None or gc.mode != 'delete':\n",
        "            not options.rolling_active_window or not options.cpu_history_coordination or\n":
        "            (not options.rolling_active_window and not finite_history) or not options.cpu_history_coordination or\n"}
    for before,after in replacements.items():
        assert code.count(before)==1;code=code.replace(before,after)
    path.write_text(code)
    shutil.copy2(REPO/'src/vllm_kvmem/full_history_benchmark.py',runtime/'src/vllm_kvmem/full_history_benchmark.py')
    for name in ('run_full_mab_comparison.py','prepare_full_mab_comparison.py'):
        shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
    proof=tree(runtime);save(root/'runtime-proof.json',proof)
    original=read(BASE/'profile.json');basefixture=read(BASE/'fixture.json')
    groups=data['groups'];byid={r['id']:r for r in data['rows']};bindings={b['id']:b for b in data['bindings']}
    # Fast cold/warm check first; the new 1.69M prefix and its next question are
    # required before formal inference. New answer formats are included too.
    qualified=[]
    for source,count in [('icl_banking77_5900shot_balance',2),('recsys_redial_full',2),
                         ('detective_qa',1),('infbench_sum_eng_shots2',1),('longmemeval_s*',1)]:
        qualified+=next(g for g in groups if g['source']==source)['cases'][:count]
    for stage in STAGES:
        dest=root/stage;dest.mkdir();(dest/'runtime-v1').symlink_to(runtime,target_is_directory=True)
        arm='mean_k' if stage.endswith('mean_k') else 'layerwise';qualification=stage.startswith('qualification')
        chosen=qualified if qualification else [r['id'] for r in data['rows']]
        fixture=dict(suite='memoryagentbench',arm=arm,cases=len(chosen),case_ids=chosen,
            rows=[byid[x] for x in chosen],bindings=[bindings[x] for x in chosen],
            reference=str(root/'data'),source_cohort_sha256=sha(root/'data/cohort.json'),
            qualification=qualification,restore_digest_qualification=qualification,
            performance_observation=True,execution_flags=basefixture['execution_flags'] if arm=='layerwise' else {})
        profile=method_profile(original,arm)
        save(dest/'fixture.json',fixture);save(dest/'profile.json',profile)
        if qualification:save(dest/'verification-profile.json',dict(profile,verify_session_restore=True,verify_session_restore_digest=True))
        save(dest/'status.json',dict(status='prepared',cases=len(chosen)))
    evaluation=root/'evaluation';evaluation.mkdir()
    official=Path('/home/chaidi/MemoryAgentBench-official')
    for sub in ('utils','llm_based_eval','configs/data_conf'):
        shutil.copytree(official/sub,evaluation/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    entity=evaluation/'processed_data/Recsys_Redial/entity2id.json';entity.parent.mkdir(parents=True)
    shutil.copy2('/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/MemoryAgentBench/entity2id.json',entity)
    save(root/'judge-protocol.json',dict(version='assistant_blind_v1',external_api=False,
        official_commit='455306dcabc3842526eb83cd4e225e5d486c5c5d',
        deterministic='Pinned official post_process: substring EM / EM / Recall@5 by source.',
        semantic='Current conversation assistant, method identity blinded; official LME equivalence rubric and summary precision/recall/fluency prompts.',
        summaries='Extract claims then judge claim precision and reference keypoint recall; use official F1 times fluency. Never substitute ROUGE or binary pass rate.',
        expected_semantic_answers=800,expected_deterministic_answers=3342,
        aggregation='Average per source within each of the four categories, then equal category macro. Also report per-source, paired wins/losses and truncation.',
        missing='Pending judgments remain pending, never assigned zero. Preserve all first answers including output-cap hits.',
        label='Official tasks and deterministic metrics; assistant semantic judge, NOT original GPT-4o judge.'))
    files={str(p.relative_to(root)):sha(p) for p in [root/'runtime-proof.json',root/'data/proof.json',root/'judge-protocol.json']}
    for stage in STAGES:
        for name in ('fixture.json','profile.json','verification-profile.json'):
            p=root/stage/name
            if p.exists():files[str(p.relative_to(root))]=sha(p)
    files.update({str(p.relative_to(root)):sha(p) for p in evaluation.rglob('*') if p.is_file()})
    save(root/'plan.json',dict(cases_per_arm=2071,arms=list(ARMS),order=list(STAGES),workers=8,
        files=files,source=str(BASE),source_runtime_proof_sha256=sha(BASE/'runtime-proof.json'),
        changed_runtime_files=[n for n,h in proof.items() if read(BASE/'runtime-proof.json').get(n)!=h],
        budgets=dict(active=36864,gen=16384,block=32,sink=41,recent=0),
        method_definitions=dict(mean_k='Global Mean-K Query; recency pressure; cumulative attention and graph disabled.',
            layerwise='Per-layer cumulative attention prefill pressure; causal per-layer Mean-K Query; existing validated r10 implementation.'),
        execution_difference='Pure Mean-K disables attention-dependent native-statistics/coalescing optimizations as required by current engine; this compares complete schemes, not an isolated layerwise ablation.',
        whole_history_admission=True,external_grading=False,automatic_retry=False,scheduled_monitor=False,
        first_answers_independent=True,qualification_answers_excluded_from_formal=True))
    save(root/'prepared.json',dict(plan_sha256=sha(root/'plan.json')))
    save(root/'status.json',dict(status='prepared',at=time.time()))
    verify(root)


def verify(root):
    assert sha(root/'plan.json')==read(root/'prepared.json')['plan_sha256']
    plan=read(root/'plan.json')
    for name,h in plan['files'].items():assert sha(root/name)==h,name
    assert tree(root/'runtime-v1')==read(root/'runtime-proof.json')
    for name,h in read(root/'data/proof.json').items():assert sha(root/'data'/name)==h,name
    return plan


def load_source(row,cache):
    path=Path(row['source']);assert sha(path)==row['source_cache_sha256']
    item=read(path);prefix=item['prefix']
    if prefix not in cache:
        assert sha(prefix)==item['prefix_sha256']
        with gzip.open(prefix,'rt') as stream:cache[prefix]=json.load(stream)
    base=cache[prefix];tokens=base['tokens']+item['suffix_tokens'];full=base['full']+item['suffix']
    assert len(tokens)==row['input_tokens'] and token_sha(tokens)==row['token_sha256']
    assert digest(full)==row['prompt_sha256'] and digest(item['query'])==row['query_sha256']
    assert [item['query_begin'],item['query_end']]==[row['query_begin'],row['query_end']]
    return dict(item,tokens=tokens,full=full)


def audit_global(query,refresh,events,kv):
    if events or query['query_scoring_method']!='mean_k':raise ValueError('Global Mean-K changed method')
    for e in refresh:
        if e['action_attention_enabled'] or e['attention_source']!='none' or e['prefill_attention_samples']:
            raise ValueError('Pure Mean-K unexpectedly accumulated attention')
        if e.get('guided_fallback') or e.get('guided_query_tokens') or e.get('cancelled'):
            raise ValueError('Unexpected guidance/fallback/cancellation')
        if e['selected_tokens']>kv['select_budget']:raise ValueError('Active budget exceeded')
    return [e for e in refresh if e['selection']=='pressure']


def infer(memory,root,capture,workers=8):
    from benchmark_memory import wait_terminal
    from benchmark_niah import CpuCredits
    from benchmark_client_io import CaptureSlice
    from graph_v2_protocol import release_owner
    fixture,profile=memory.verify(root);kv=profile['kvmem'];layerwise=fixture['arm']=='layerwise'
    rows=fixture['rows'];bindings={r['id']:r for r in fixture['bindings']};groups={}
    for row in rows:groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
    out=root/'run/answers';out.mkdir(parents=True,exist_ok=True)
    save(out/'manifest.json',dict(fixture=str(root/'fixture.json'),cases=len(rows),rows=rows,workers=workers,
        reference=fixture['reference'],profile_sha256=sha(root/'profile.json'),arm=fixture['arm']))
    credits=CpuCredits(kv['cpu_read_cache_bytes']-16*1024**3);lock=Lock();results={}
    def group_run(group,items):
        prefix=min(r['query_begin'] for r in items)//1024*1024
        extent=max(r['input_tokens'] for r in items)+2*kv['gen_budget'];amount=((extent+31)//32)*32*17*4*256*2
        credits.acquire(group,amount);session='mab-full-'+uuid.uuid4().hex;prefix_state=None;cache={}
        try:
            for done,row in enumerate(items):
                directory=out/'cases'/row['id'];directory.mkdir(parents=True,exist_ok=True)
                if (directory/'attempt.json').exists():raise ValueError('Existing technical attempt must be preserved')
                src=load_source(row,cache);tokens=src['tokens'];binding=bindings[row['id']]
                payload=dict(binding['sampling'],max_tokens=binding['max_tokens'],thinking_token_budget=0,
                    model=profile['served_model_name'],prompt=tokens,session_id=session,cache_salt=session,
                    return_token_ids=True,kvmem_query_span=[row['query_begin'],row['query_end']],
                    kvmem_replay_span=[prefix,row['query_end']],kvmem_reselect='force')
                save(directory/'request.json',dict({k:v for k,v in payload.items() if k!='prompt'},
                    token_sha256=row['token_sha256'],prompt_tokens=len(tokens),sampling_sha256=binding['sha256']))
                save(directory/'attempt.json',dict(session=session,group=group,at=time.time()))
                offset=CaptureSlice(capture/name for name in ('bounded_refresh.jsonl','layerwise_query.jsonl','verified_restores_digest.jsonl'))
                started=time.perf_counter();http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request=urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with http.open(request,timeout=14400) as stream:response=json.load(stream)
                elapsed=time.perf_counter()-started;save(directory/'response.json',compact_response(response,tokens))
                assert response['usage']['prompt_tokens']==len(tokens) and response['choices'][0]['prompt_token_ids']==tokens
                rid=response['id'];terminal=wait_terminal(capture,rid)
                refresh=[e for e in offset.read(capture/'bounded_refresh.jsonl') if e['request_id'].startswith(rid+'-')]
                queries=[e for e in refresh if e['selection']=='query'];assert len(queries)==1
                query=queries[0];assert query['query_span']==payload['kvmem_query_span']
                events=[e for e in offset.read(capture/'layerwise_query.jsonl') if e.get('request_id','').startswith(rid+'-')]
                layers=[e for e in events if e['event']=='layer_selected']
                if layerwise:
                    assert [e['layer'] for e in layers]==list(range(16))
                    assert all(e['mode']=='sequential' and e['query_source']=='same_forward_current_layer'
                        and e['predecessor_layers_completed']==e['layer'] and e['charged_tokens']<=kv['select_budget'] for e in layers)
                    pressures=memory.memory_pressure_audit(events,refresh,True,True,kv['select_budget'])
                    prefix_state=memory.memory_prefix_audit(events,prefix,prefix_state)
                else:pressures=audit_global(query,refresh,events,kv)
                capacity=memory.memory_generation_capacity_audit(query,layers if layerwise else [dict(charged_tokens=query['selected_tokens'])],len(tokens),binding['max_tokens'],kv)
                assert terminal.get('checkpoint_kind')=='P' and terminal.get('checkpoint_tokens')==prefix
                if done and pressures:raise ValueError('Warm request repeated the cold prefix')
                restore=None
                if done and fixture['qualification']:
                    restores=[e for e in offset.read(capture/'verified_restores_digest.jsonl') if e['request_id'].startswith(rid+'-')]
                    assert len(restores)==1;restore=restores[0]
                    assert restore['history_tokens']==prefix and restore['sha256_equal'] and restore['recurrent_byte_equal']
                    assert restore['verification']=='sha256_all_target_bytes_and_exact_model_tensors' and restore['working_attention_layers']==17
                text=response['choices'][0]['text'];finish=response['choices'][0]['finish_reason']
                record=dict(src['info'],id=row['id'],suite='memoryagentbench',status='completed',
                    answer=text.split('</think>',1)[-1].strip(),raw_completion=text,usage=response['usage'],
                    seconds=elapsed,prompt_tokens=len(tokens),token_sha256=row['token_sha256'],prompt_sha256=row['prompt_sha256'],
                    query_sha256=row['query_sha256'],original_input_intact=True,attention=query,
                    # The server's capture log keeps every event; the result points at the
                    # request's slice (benchmark_client_io.result_events reads it back).
                    layerwise=dict(events=offset.reference(capture/'layerwise_query.jsonl',rid+'-',events),layers=layers),
                    actual_query_method='mean_k_sequential_layerwise' if layerwise else 'mean_k',
                    actual_pressure_method='accumulated_attention_per_layer' if layerwise else 'recency',
                    pressure_refreshes=len(pressures),finish_reason=finish,hit_output_cap=finish=='length',
                    prefix_reused=bool(done),prefix_tokens=prefix,prefix_sha256=token_sha(tokens[:prefix]),
                    generation_capacity=capacity,prefix_restore_digest=restore)
                if prefix_state is not None:record['layerwise_prefix']=prefix_state
                save(directory/'events.json',refresh);save(directory/'terminal.json',terminal);save(directory/'result.json',record)
                with lock:
                    results[row['id']]={k:record[k] for k in ('id','seconds','usage','hit_output_cap','prefix_reused')}
                    save(out/'summary.json',dict(expected=len(rows),completed=len(results),results=list(results.values())))
                    save(root/'status.json',dict(status='running',completed=len(results),expected=len(rows),at=time.time()))
                    print(json.dumps(dict(id=row['id'],completed=len(results),expected=len(rows),seconds=elapsed)),flush=True)
            save(out/'releases'/(group+'.json'),release_owner(profile,capture,session,tokens[-1]))
        except BaseException:
            credits.cancel();raise
        finally:credits.release(group,amount)
    start=time.perf_counter();started_at=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(group_run,g,items) for g,items in groups.items()]
        for f in as_completed(futures):f.result()
    elapsed=time.perf_counter()-start
    save(root/'inference-performance.json',dict(inference_wall_seconds=elapsed,started_at=started_at,completed_at=time.time(),
        concurrency_limit=workers,completed_cases=len(results),fresh_cases=len(results),fresh_cases_per_hour=3600*len(results)/elapsed,
        scope='Inference dispatch wall time, includes prefix reuse, Query, maintenance and decode; not pure prefill.'))
    save(out/'cpu_admission.json',dict(budget=credits.budget,peak=credits.peak,events=credits.events))
    assert set(results)==set(fixture['case_ids'])
    save(root/'execution-audit.json',dict(status='passed',cases=len(results),method=fixture['arm'],
        full_inputs_preserved=True,first_answers_only=True,output_caps_and_sampling_unchanged=True))


def arm(root,stage):
    verify(root);dest=root/stage;runtime=root/'runtime-v1'
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts')]
    spec=importlib.util.spec_from_file_location('frozen_mab_memory',runtime/'scripts/run_layerwise_memory.py')
    memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    memory.verify=lambda path:(read(path/'fixture.json'),read(path/'profile.json')) if path==dest else (_ for _ in ()).throw(ValueError('Wrong stage'))
    original_command=memory.server_command
    fixture=read(dest/'fixture.json');execution=explicit_execution(fixture)
    def command(path,fixture):
        if fixture['arm']=='layerwise':result=original_command(path,fixture)
        else:
            config='verification-profile.json' if fixture['qualification'] else 'profile.json'
            result=[sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(path/config),'--run-dir',str(path/'run/server')]
        if execution:
            assert result[1:5]==['-B','-m','vllm_kvmem','serve']
            return [result[0],'-B',str(runtime/'scripts/_query_stage_server.py'),*result[5:]]
        return result
    memory.server_command=command
    def checked_infer(path,capture,**kwargs):
        launch=read(path/'run/server/launch.json')
        if any(launch['environment'].get(k)!=v for k,v in execution.items()):
            raise ValueError('Actual full-MAB execution differs from frozen controls')
        save(path/'execution-environment-audit.json',dict(passed=True,expected=execution,
            source_sha256=sha(path/'run/server/launch.json')))
        return infer(memory,path,capture,1 if stage.startswith('qualification') else 8)
    memory.infer=checked_infer
    import benchmark_query_fps_prebuild_live as env_module
    original_environment=env_module.server_environment
    def environment(path):
        env=original_environment(path)
        if execution:
            env['QUERY_STAGE_EXECUTION_JSON']=json.dumps(execution)
            native,=list((runtime/'src/vllm_kvmem').glob('_native*.so'))
            env['QUERY_STAGE_NATIVE_BINARY_JSON']=json.dumps(dict(path=str(native.resolve()),sha256=sha(native)))
            env['QUERY_STAGE_NATIVE_AUDIT_OUTPUT']=str(dest/'run/native-runtime-identity.json')
        return env
    env_module.server_environment=environment
    try:memory.run(dest)
    finally:env_module.server_environment=original_environment
    verify(root)


def prepare_judgments(root):
    """Only packet creation, no external inference or invented judgments."""
    semantic=[];mapping={};scoring=root/'scoring';scoring.mkdir(exist_ok=True)
    salt=read(root/'prepared.json')['plan_sha256']
    sys.path.insert(0,str(REPO/'artifacts/mab-pressure-partial-score-20261105-r1/scoring-deps'))
    spec=importlib.util.spec_from_file_location('mab_full_official_metrics',root/'evaluation/utils/eval_other_utils.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    def builtin(v):
        if isinstance(v,dict):return {str(k):builtin(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [builtin(x) for x in v]
        return v.item() if hasattr(v,'item') else v
    for method in ARMS:
        records=[read(p) for p in sorted((root/('formal_'+method)/'run/answers/cases').glob('*/result.json'))]
        assert len(records)==2071
        cwd=Path.cwd()
        try:
            os.chdir(root/'evaluation')
            for r in records:
                values,extra=metrics.post_process({'output':r['answer']},r['gold_answer'],{'sub_dataset':r['source']})
                r['official_local_metrics']=builtin(values)
                r['official_parsed_output']=builtin(extra.get('parsed_output'))
        finally:os.chdir(cwd)
        dest=scoring/method/'rows/all/results.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        for r in records:
            if r['family'] not in ('longmemeval','infbench'):continue
            ident=digest(salt+method+r['id'])[:24]
            packet={k:r[k] for k in ('raw_question','gold_answer','question_type','question_id','keypoints','answer')}
            packet.update(blind_id=ident,family=r['family'])
            semantic.append(packet)
            mapping[ident]=dict(method=method,case_id=r['id'],answer_sha256=digest(r['raw_completion']))
    assert len(semantic)==800
    semantic.sort(key=lambda r:r['blind_id'])
    (scoring/'blind-packets.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in semantic))
    save(scoring/'private-method-mapping.json',mapping)
    save(scoring/'status.json',dict(status='awaiting_assistant_judgments',pending_semantic=800,
        deterministic_answers_scored=3342,external_api_used=False))


def run(root):
    plan=verify(root)
    sys.path[:0]=[str(root/'runtime-v1/scripts'),str(root/'runtime-v1/src')]
    from run_recall_comparison import termination_guard,stop_owned,gpu_lock
    from benchmark_block_graph import gpu_processes
    child=None
    # Reserve the queue separately; each owned server retains the established GPU lock.
    with termination_guard(),gpu_lock(root/'queue.lock'):
        assert not gpu_processes(),'GPU is owned by another task'
        if (root/'process.json').exists():raise ValueError('Explicit recovery required; do not overwrite attempts')
        save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        pointer=REPO/'artifacts/value-recall-active.json'
        if pointer.exists():save(root/'previous-pointer.json',read(pointer))
        save(pointer,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),process=str(root/'process.json'),at=time.time()))
        try:
            for stage in plan['order']:
                if stage.startswith('formal'):
                    for q in STAGES[:2]:
                        assert read(root/q/'status.json')['status']=='completed'
                        assert read(root/q/'execution-audit.json')['status']=='passed'
                        assert not read(root/q/'run/cleanup.json')['gpu_processes']
                save(root/'status.json',dict(status='running',stage=stage,at=time.time()))
                with (root/(stage+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-B',str(root/'runtime-v1/scripts/run_full_mab_comparison.py'),
                        'arm','--root',str(root),'--stage',stage],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'child.json',dict(pid=child.pid,stage=stage,at=time.time()))
                if child.wait():raise RuntimeError('Preserved technical failure: '+stage)
            prepare_judgments(root)
            save(root/'status.json',dict(status='inference_completed',cases_per_arm=2071,grading='assistant_pending',at=time.time()))
        except BaseException as e:
            save(root/'status.json',dict(status='failed',error=repr(e),at=time.time()));raise
        finally:stop_owned(child)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','arm','judge-packets'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=STAGES)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root)
    elif a.action=='run':run(root)
    elif a.action=='arm':arm(root,a.stage)
    else:prepare_judgments(root)
