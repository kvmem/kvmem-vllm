"""Original LME/MAB cohorts and request parameters with layerwise Query recall."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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

import recall_comparison_protocol as p
from benchmark_memory import source, wait_terminal, digest_tokens
from benchmark_niah import CpuCredits
from benchmark_client_io import CaptureSlice, compact_response

REPO=Path(__file__).resolve().parents[1]
REFERENCE=REPO/'artifacts/fused-stable-quality-20261021-r1'
BASE=REPO/'artifacts/layerwise-meank-balanced26-20261102-r2'
HISTORICAL_MEMORY={'longmemeval_s':'layerwise-longmem-102-20261102-r5',
                   'memoryagentbench':'layerwise-mab-1116-20261102-r6'}
SAMPLING=('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed')
EXECUTION_KV_KEYS={'action_native_scores', 'action_native_max_rows', 'action_native_max_bytes',
                   'cpu_seal_hash_workers','cpu_direct_record_seal','cpu_seal_serial_allocate'}
EXECUTION_FLAGS=('layerwise_prefill_pressure','layerwise_pressure_single_assembly',
    'layerwise_pressure_fast_metadata','layerwise_packed_assembly',
    'layerwise_prepared_index','layerwise_batched_reads','layerwise_paired_reads',
    'layerwise_buffered_history','layerwise_rolling_decode')
EXPERIMENTAL_EXECUTION_FLAGS=('layerwise_pressure_prepared_reads',
    'layerwise_pressure_post_forward_reap','layerwise_pressure_submit_on_prepare','layerwise_pressure_native_score_export',
    'layerwise_pressure_reuse_layer_views','layerwise_pressure_resident_values')


def inherited_execution(origin):
    result={key:origin.get(key,False) for key in EXECUTION_FLAGS}
    # Do not add absent False keys to historical frozen fixtures/gates.
    result.update({key:origin[key] for key in EXPERIMENTAL_EXECUTION_FLAGS if key in origin})
    return result


def server_command(root,fixture):
    flags=fixture.get('execution_flags',{})
    if flags.get('layerwise_buffered_history') and not flags.get('layerwise_prefill_pressure'):
        raise ValueError('Buffered histories require independent prefill pressure')
    config='verification-profile.json' if fixture.get('restore_digest_qualification') else 'profile.json'
    command=[sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(root/config),
        '--run-dir',str(root/'run/server'),'--layerwise-query-diagnostic','sequential',
        '--layerwise-query-optimized','--layerwise-query-native-batches']
    for key in (*EXECUTION_FLAGS,*EXPERIMENTAL_EXECUTION_FLAGS):
        if fixture.get('execution_flags',{}).get(key):command.append('--'+key.replace('_','-'))
    return command


def restore_verification_profile(profile):
    """Explicit launch controls survive serve's intentional ambient-env reset."""
    return dict(deepcopy(profile),verify_session_restore=True,verify_session_restore_digest=True)


def supplemental_qualification(previous, rows):
    """Preserve first responses; qualify a new cold/warm pair after telemetry failure."""
    fixture,profile=verify(previous)
    if (fixture['suite']!='memoryagentbench' or not fixture['qualification'] or
            p.read(previous/'status.json')['status']!='failed'):
        raise ValueError('Supplement requires a failed MAB qualification')
    cleanup=p.read(previous/'run/cleanup.json')
    if cleanup['gpu_processes'] or cleanup['remaining_owned_groups']:
        raise ValueError('Previous qualification still owns GPU resources')
    saved={f.parent.name for f in (previous/'run/answers/cases').glob('*/response.json')}
    groups={}
    for row in rows:groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
    longest=max((g for g in groups.values() if len(g)>=2),key=lambda g:max(r['input_tokens'] for r in g))
    pending=[r for r in longest if r['id'] not in saved][:2]
    if len(pending)!=2:raise ValueError('No two unanswered questions for supplemental warm restore')
    largest=max(rows,key=lambda r:r['input_tokens'])
    record=p.read(previous/'run/answers/cases'/largest['id']/'result.json')
    if (record['status']!='completed' or record['token_sha256']!=largest['token_sha256'] or
            not record['original_input_intact'] or record['prompt_tokens']!=largest['input_tokens']):
        raise ValueError('Largest input lacks a complete, input-bound first answer')
    memory_pressure_audit(record['layerwise']['events'],p.read(
        previous/'run/answers/cases'/largest['id']/'events.json'),True,
        fixture['execution_flags'].get('layerwise_buffered_history',False))
    if [r['layer'] for r in record['layerwise']['layers']]!=list(range(16)):
        raise ValueError('Largest input lacks causal Query layer evidence')
    files={str(f):p.sha(f) for f in (previous/'run/answers').rglob('*') if f.is_file()}
    for name in ('profile.json','fixture.json','runtime-proof.json','prepared-proof.json','status.json',
                 'run/cleanup.json','run/server/launch.json'):
        files[str(previous/name)]=p.sha(previous/name)
    return pending,dict(source=str(previous),files=files,preserved_response_ids=sorted(saved),
        new_case_ids=[r['id'] for r in pending],largest_verified_id=largest['id'],
        resampled_previous_answers=0,scope='Additional diagnostic pair only; previous answers remain immutable.')


def memory_pressure_audit(events, refresh, independent, buffered=False, select_budget=36864, rolling=False):
    if buffered and not independent:raise ValueError('Buffered state requires independent pressure')
    pressures=[e for e in refresh if e['selection']=='pressure']
    actual=[e for e in events if e['event']=='pressure_selected']
    method='accumulated_attention_per_layer' if independent else 'accumulated_attention'
    if len(pressures)!=len(actual) or any(e['actual_method']!=method for e in actual):
        raise ValueError('Pressure scoring silently changed')
    if independent:
        for event in actual:
            decode=rolling and event['event_type']=='decode_pressure'
            if event.get('fallback_reason') is not None or (event['event_type']!='prefill_pressure' and not decode):
                raise ValueError('Unexpected pressure entry or fallback')
            if [e['layer'] for e in event['layers']]!=list(range(16)):
                raise ValueError('Missing independent pressure layer')
            for layer in event['layers']:
                if decode and (layer.get('attention_state_backend')!='native_decode' or
                               layer.get('score_scope')!='all_accepted_including_thinking' or
                               layer.get('query_samples',0)<=0):
                    raise ValueError('Decode pressure lacks accepted per-layer output attention')
                if buffered and not decode and (layer.get('attention_state_backend')!='native_buffered' or
                                 layer.get('buffered_commits',0)+layer.get('buffered_restores',0)<=0):
                    raise ValueError('Pressure did not use the configured buffered history')
                if (layer['charged_tokens']>select_budget or layer['selected_tokens']!=event['selected_tokens'] or
                    layer['accepted_score_end']>event['accepted_history'] or
                    not set(layer['selected_starts'])<=set(layer['candidate_starts']) or
                    not set(range(0,1312,32))<=set(layer['selected_starts'])):
                    raise ValueError('Pressure scope, accepted evidence or Active budget changed')
    return pressures


def memory_prefix_audit(events, prefix, previous):
    captured=[e for e in events if e['event']=='pressure_prefix_captured']
    restored=[e for e in events if e['event']=='pressure_prefix_restored']
    if len(captured)!=1 or len(restored)!=(1 if previous is not None else 0):
        raise ValueError('Missing per-layer frozen-prefix capture/restore evidence')
    for event in captured+restored:
        if (event['accepted_history']!=prefix or len(event['layer_names'])!=16 or
                len(event['layer_tokens'])!=16 or len(set(event['layer_tokens']))!=1 or
                event['actual_method']!='accumulated_attention_per_layer' or event['fallback_reason'] is not None):
            raise ValueError('Frozen-prefix layer identity or method changed')
    keys=('accepted_history','layer_names','layer_tokens','map_sha256','attention_sha256')
    current={key:captured[0][key] for key in keys}
    if previous is not None and (current!=previous or any(restored[0][k]!=previous[k] for k in keys)):
        raise ValueError('P-prefix restore changed per-layer blocks or accumulated scores')
    return current


def memory_restore_digest_audit(rows, response_id, prefix, working_tokens):
    matching=[r for r in rows if r['request_id'].startswith(response_id+'-')]
    if len(matching)!=1:
        raise ValueError('Missing unique request-bound frozen-prefix byte verification')
    row=matching[0]
    if (row['history_tokens']!=prefix or row['working_tokens']!=working_tokens or
            not row['sha256_equal'] or not row['recurrent_byte_equal'] or
            row['working_attention_layers']!=17 or row['mtp_attention_layers']!=1 or
            # This model's one MTP layer is included in the 17 KV layers.
            # Its adapter correctly has no separate persistent draft tensors.
            row['recurrent_tensors']<=0 or row['draft_layers']!=0 or
            row['target_kv_bytes']!=working_tokens*17*4*256*2 or len(row['sha256'])!=34 or
            row['verification']!='sha256_all_target_bytes_and_exact_model_tensors' or
            row['oracle_kind']!='fp8_immutable_rebuild'):
        raise ValueError('Frozen-prefix KV, GDN or draft byte verification is incomplete')
    return row


def memory_generation_capacity_audit(query, layers, prompt_tokens, max_output, kv):
    """Prove the unchanged output cap fits the actual per-layer final Query view."""
    selected=max(layer['charged_tokens'] for layer in layers)
    replay=prompt_tokens-query['history_begin']
    capacity=kv['select_budget']+kv['gen_budget']
    if kv.get('rolling_active_window'):
        headroom=capacity-selected-replay
        if selected>kv['select_budget'] or replay<0 or headroom<kv['block_tokens']:
            raise ValueError('Rolling final Query leaves no valid generation window')
        return dict(selected_history_tokens=selected,replay_tokens=replay,max_output_tokens=max_output,
            physical_capacity_tokens=capacity,initial_generation_headroom_tokens=headroom,
            no_post_query_pressure_needed=False,rolling_active_window=True)
    headroom=capacity-selected-replay-max_output
    if (selected>kv['select_budget'] or replay<0 or headroom<kv['block_tokens']):
        raise ValueError('Actual final Query did not reserve the complete original output limit')
    return dict(selected_history_tokens=selected,replay_tokens=replay,max_output_tokens=max_output,
        physical_capacity_tokens=capacity,speculation_headroom_tokens=headroom,
        no_post_query_pressure_needed=True)


def completed_memory_qualification(root, suite):
    fixture,profile=verify(root)
    audit=p.read(root/'execution-audit.json');cleanup=p.read(root/'run/cleanup.json')
    if (p.read(root/'status.json')['status']!='completed' or audit['status']!='passed' or
            not fixture['qualification'] or fixture['suite']!=suite or
            audit['cases']!=fixture['cases'] or cleanup['gpu_processes'] or cleanup['remaining_owned_groups']):
        raise ValueError('Memory model qualification is incomplete or has not released its GPU')
    files={str(root/name):p.sha(root/name) for name in
        ('prepared-proof.json','runtime-proof.json','fixture.json','profile.json','execution-audit.json','status.json','run/cleanup.json')}
    for name in ('verification-profile.json','supplemental-qualification.json'):
        if (root/name).exists():files[str(root/name)]=p.sha(root/name)
    if (root/'supplemental-qualification.json').exists():
        files.update(p.read(root/'supplemental-qualification.json')['files'])
    if (profile['kvmem'].get('cpu_seal_hash_workers',1)>1 or profile['kvmem'].get('cpu_direct_record_seal',False)
            or fixture.get('cpu_region_gather_min_bytes') is not None):
        from run_layerwise_matched_performance import archive_backend_audit
        archive=root/'run/server/capture/archive.json'
        archive_backend_audit(profile,p.read(archive),fixture.get('cpu_region_gather_min_bytes'))
        files[str(archive)]=p.sha(archive)
    warm=0
    for row in fixture['rows']:
        case=root/'run/answers/cases'/row['id'];result=p.read(case/'result.json')
        if result['status']!='completed' or result['token_sha256']!=row['token_sha256']:
            raise ValueError('Qualification first answer does not match its original input')
        for name in ('request.json','response.json','result.json','terminal.json','events.json'):
            files[str(case/name)]=p.sha(case/name)
        if suite=='memoryagentbench' and result['prefix_reused']:
            response=p.read(case/'response.json')
            memory_restore_digest_audit([result['prefix_restore_digest']],response['id'],
                result['prefix_tokens'],result['layerwise_prefix']['layer_tokens'][0])
            warm+=1
    if suite=='memoryagentbench' and not warm:
        raise ValueError('MAB qualification did not verify a real warm prefix restore')
    return fixture,profile,dict(source=str(root),files=files,verified_warm_restores=warm,
        scope='Same frozen engine and controls; complete first answers and request-bound byte verification.')


def normalize_cuda_alias(environment,preferred=Path('/usr/local/cuda')):
    """Use the RULER spelling only when both paths are the same installation.

    Ninja hashes the literal compiler/include paths. Alternating cuda and
    cuda-13 aliases otherwise rebuilds identical kernels between suites.
    Different CUDA installations must retain their original compiler.
    """
    result=dict(environment)
    selected=Path(result['CUDA_HOME']) if result.get('CUDA_HOME') else None
    if selected is not None and selected.exists() and preferred.exists() and selected.samefile(preferred):
        result['CUDA_HOME']=str(preferred)
        result['PATH']=str(preferred/'bin')+os.pathsep+result.get('PATH','')
    return result


def prepare(root,suite,qualification=False,recovery=None,pressure_runtime=None,qualification_source=None,
            efficiency_source=None,efficiency_authorization=None,qualification_after=None):
    if root.exists():raise ValueError('New memory runs require a fresh directory')
    if qualification_after is not None and (not qualification or suite!='memoryagentbench' or recovery is not None):
        raise ValueError('Supplemental qualification is only for a new MAB diagnostic pair')
    if qualification_source is not None and (qualification or pressure_runtime is None):
        raise ValueError('A completed qualification source applies only to formal per-layer Memory')
    fixture=p.read(REFERENCE/(suite+'-fixture.json'))
    original=p.read(REFERENCE/(suite+'-off.json'))
    profile=deepcopy(original)
    profile.update(port=19025 if suite=='longmemeval_s' else 19026,
                   served_model_name='layerwise-'+suite)
    execution={}
    region_min_bytes=None
    memory_gate=None
    efficiency_gate=None
    if pressure_runtime is not None:
        pressure_runtime=pressure_runtime.resolve()
        from run_layerwise_query_diagnostic import verify as verify_pressure
        verify_pressure(pressure_runtime)
        origin_fixture=p.read(pressure_runtime/'fixture.json')
        origin_profile=p.read(pressure_runtime/'profile.json')
        region_min_bytes=origin_fixture.get('cpu_region_gather_min_bytes')
        if origin_profile['kvmem'].get('cpu_seal_hash_workers',1)>1:
            profile['kvmem']['cpu_seal_hash_workers']=origin_profile['kvmem']['cpu_seal_hash_workers']
        if origin_profile['kvmem'].get('cpu_direct_record_seal',False):
            profile['kvmem']['cpu_direct_record_seal']=True
        if origin_profile['kvmem'].get('cpu_seal_serial_allocate',False):
            profile['kvmem']['cpu_seal_serial_allocate']=True
        if qualification_source is not None:
            prior_gate=p.read(qualification_source/'efficiency-gate.json')
            if efficiency_source is None:efficiency_source=Path(prior_gate['source'])
            if efficiency_authorization is None:efficiency_authorization=prior_gate.get('authorization_file')
        if efficiency_source is None:
            raise ValueError('Per-layer Memory requires accepted matched efficiency evidence')
        from layerwise_efficiency_gate import create as efficiency_acceptance
        efficiency_gate=efficiency_acceptance(pressure_runtime,efficiency_source,efficiency_authorization)
        if not origin_fixture['layerwise_prefill_pressure']:
            raise ValueError('Memory pressure runtime must use independent cumulative attention')
        if not qualification and not p.read(pressure_runtime/'equivalence.json')['passed']:
            raise ValueError('Formal memory requires passed full-input execution equivalence')
        execution=inherited_execution(origin_fixture)
        if suite=='memoryagentbench' and (not profile['kvmem']['session_freeze_prompt'] or
                                         profile['kvmem']['async_restore']):
            raise ValueError('Original MAB protocol must use frozen synchronous P restore')
        if origin_profile['kvmem'].get('action_native_scores','off')!='off':
            # Preserve the original 1,024-token memory chunk/stride. Only
            # execution scratch changes: eight owners plus speculative rows.
            import math
            rows=profile['max_num_seqs']*math.ceil(profile['prefill_chunk_tokens']/profile['kvmem']['prefill_attention_stride'])+64
            profile['kvmem'].update(action_native_scores='reuse',action_native_max_rows=rows,
                                    action_native_max_bytes=4*1024**3)
        if not qualification:
            if qualification_source is None:
                raise ValueError('Formal per-layer Memory requires its completed model qualification')
            qualified,qualified_profile,memory_gate=completed_memory_qualification(qualification_source,suite)
            if (qualified_profile!=profile or qualified['execution_flags']!=execution or
                    qualified['pressure_runtime_source']!=str(pressure_runtime)):
                raise ValueError('Formal Memory differs from the qualified runtime or controls')
            from run_layerwise_query_efficiency import source_audit
            source_audit(pressure_runtime)
    rows=fixture['rows'];selected=rows
    recovered_ids=set()
    if recovery is not None:
        previous,previous_profile=verify(recovery)
        if previous['suite']!=suite or previous_profile!=profile:
            raise ValueError('Recovery cannot change the suite or frozen controls')
        if p.read(recovery/'status.json')['status']!='failed':
            raise ValueError('Recovery requires a failed, preserved attempt')
        cleanup=p.read(recovery/'run/cleanup.json')
        if cleanup['gpu_processes'] or cleanup['remaining_owned_groups']:
            raise ValueError('Previous GPU owner has not released its resources')
        recovered_ids={f.parent.name for f in (recovery/'run/answers/cases').glob('*/result.json')
                       if p.read(f).get('status')=='completed'}
    if qualification:
        if suite=='longmemeval_s':selected=rows[:2]
        else:
            groups={}
            for row in rows:groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
            repeatable=[g for g in groups.values() if len(g)>=2]
            longest=max(repeatable,key=lambda g:max(r['input_tokens'] for r in g))
            # After a restart, the first unanswered question must ingest P
            # again. Keep completed answers verbatim and include the next
            # original question to exercise a real warm restore after it.
            if recovered_ids:
                pending=[r for r in longest if r['id'] not in recovered_ids][:2]
                if len(pending)!=2:raise ValueError('No two unanswered questions for warm-restore qualification')
                ids=recovered_ids|{r['id'] for r in pending}
                selected=[r for r in longest if r['id'] in ids]
            else:selected=longest[:2]
            largest=max(rows,key=lambda r:r['input_tokens'])
            if largest['id'] not in {r['id'] for r in selected}:selected=selected+[largest]
    supplement=None
    if qualification_after is not None:
        selected,supplement=supplemental_qualification(qualification_after,rows)
    if (profile['kvmem']['select_budget'],profile['kvmem']['gen_budget'])!=(36864,16384):
        raise ValueError('Memory KV budget differs from the original')
    reference=Path(fixture['reference'])
    if p.sha(reference/'cohort.json')!=fixture['source_cohort_sha256']:
        raise ValueError('Original memory cohort changed')
    bindings=[]
    for row in selected:
        config=reference/'results/attention/cases'/suite/row['id']/'request_config.json'
        old=p.read(config)
        if old['token_sha256']!=row['token_sha256']:raise ValueError('Source request tokens differ')
        bindings.append(dict(id=row['id'],request=str(config),sha256=p.sha(config),
            sampling={k:old[k] for k in SAMPLING},max_tokens=16384 if suite=='longmemeval_s' else old['max_tokens'],
            thinking_token_budget=4096 if suite=='longmemeval_s' else 0))
    root.mkdir(parents=True)
    runtime=root/'runtime-v1'
    runtime_source=qualification_source if memory_gate is not None else (pressure_runtime or BASE)
    shutil.copytree(runtime_source/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    modules=(('layerwise_checkpoints.py','prompt_checkpoints.py','session_worker.py','async_restore.py','restore_digest.py',
              'layerwise_pressure.py','original_rope.py')
             if pressure_runtime is not None else ('layerwise_query.py', 'request_plan.py', 'window_assembly.py'))
    if memory_gate is None:
        for name in modules:shutil.copy2(REPO/'src/vllm_kvmem'/name,runtime/'src/vllm_kvmem'/name)
        shutil.copy2(__file__,runtime/'scripts'/Path(__file__).name)
        for name in ('layerwise_efficiency_gate.py','run_layerwise_matched_performance.py',
                     'run_layerwise_query_efficiency.py','run_layerwise_pressure_efficiency.py'):
            shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
    # RULER's shorter output cap never needed the original Memory-specific
    # final-Query output reservation. Reuse the exact previously run module,
    # including its complete template-suffix reservation, for both Memory suites.
    memory_request_plan=REPO/'artifacts'/HISTORICAL_MEMORY[suite]/'runtime-v1/src/vllm_kvmem/request_plan.py'
    target_request_plan=runtime/'src/vllm_kvmem/request_plan.py'
    if memory_gate is None:
        shutil.copy2(memory_request_plan,target_request_plan)
    if p.sha(target_request_plan)!=p.sha(memory_request_plan):
        raise ValueError('Original Memory Query/output reservation changed')
    p.save(root/'memory-runtime-reuse.json',dict(source=str(memory_request_plan),source_sha256=p.sha(memory_request_plan),
        target='src/vllm_kvmem/request_plan.py',
        scope='Exact original layerwise Memory input/output reservation; model, input, sampling and configured Active/Gen budgets unchanged.'))
    if pressure_runtime is None:
        serve=runtime/'src/vllm_kvmem/serve.py';text=serve.read_text()
        anchor='    args = parser.parse_args(argv)\n'
        text=text.replace(anchor,"    parser.add_argument('--layerwise-query-optimized', action='store_true')\n"
            "    parser.add_argument('--layerwise-query-native-batches', action='store_true')\n"+anchor)
        text=text.replace("        if profile.get('execution_mode') != 'eager' or profile.get('max_num_seqs') != 1:\n",
            "        if not args.layerwise_query_native_batches and (profile.get('execution_mode') != 'eager' or profile.get('max_num_seqs') != 1):\n")
        anchor="        plan['environment']['KVMEM_LAYERWISE_QUERY_MODE'] = args.layerwise_query_diagnostic\n"
        if text.count(anchor)!=1:raise ValueError('Layerwise native launch anchor changed')
        text=text.replace(anchor,anchor+"        if args.layerwise_query_optimized:\n"
            "            plan['environment']['KVMEM_LAYERWISE_QUERY_OPTIMIZED'] = '1'\n"
            "        if args.layerwise_query_native_batches:\n"
            "            plan['environment']['KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES'] = '1'\n")
        serve.write_text(text)
    p.save(root/'profile.json',profile)
    digest_qualification=bool(qualification and pressure_runtime is not None and suite=='memoryagentbench')
    if digest_qualification:p.save(root/'verification-profile.json',restore_verification_profile(profile))
    p.save(root/'fixture.json',dict(fixture,suite=suite,rows=selected,cases=len(selected),
        qualification=qualification,case_ids=[r['id'] for r in selected],bindings=bindings,
        restore_digest_qualification=digest_qualification,
        performance_observation=True,
        full_cohort_count=len(rows),source_fixture=str(REFERENCE/(suite+'-fixture.json')),
        source_fixture_sha256=p.sha(REFERENCE/(suite+'-fixture.json')),
        source_profile=str(REFERENCE/(suite+'-off.json')),source_profile_sha256=p.sha(REFERENCE/(suite+'-off.json')),
        execution_flags=execution,execution_kvmem_overrides={k:profile['kvmem'][k] for k in EXECUTION_KV_KEYS
            if profile['kvmem'].get(k)!=original['kvmem'].get(k)},
        cpu_region_gather_min_bytes=region_min_bytes,
        pressure_runtime_source=str(pressure_runtime) if pressure_runtime is not None else None))
    if recovery is not None:
        from recover_hybrid_mab import inherit_cases
        current=p.read(root/'fixture.json')
        old_rows={r['id']:r for r in previous['rows']}
        if any(r['id'] not in old_rows or r!=old_rows[r['id']]
               for r in current['rows'] if r['id'] in recovered_ids):
            raise ValueError('Inherited answer inputs differ from the original rows')
        proof=inherit_cases(recovery,root/'recovered-answers',current)
        proof.update(kind='layerwise_staging_repair',newly_inferred_only_unanswered=True)
        proof['source_files'].update({n:p.sha(recovery/n) for n in
            ('profile.json','fixture.json','runtime-proof.json','prepared-proof.json','status.json','run/cleanup.json')})
        p.save(root/'recovery-proof.json',proof)
    p.save(root/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
                                  if f.is_file() and '__pycache__' not in f.parts})
    proof_names=['profile.json','fixture.json','runtime-proof.json','memory-runtime-reuse.json']
    if digest_qualification:proof_names.append('verification-profile.json')
    if supplement is not None:
        from recover_hybrid_mab import engine
        if engine(p.read(root/'runtime-proof.json'))!=engine(p.read(qualification_after/'runtime-proof.json')):
            raise ValueError('Supplemental qualification cannot change the previously exercised model engine')
        p.save(root/'supplemental-qualification.json',supplement)
        proof_names.append('supplemental-qualification.json')
    if efficiency_gate is not None:
        p.save(root/'efficiency-gate.json',efficiency_gate);proof_names.append('efficiency-gate.json')
    if memory_gate is not None:
        if p.read(root/'runtime-proof.json')!=p.read(qualification_source/'runtime-proof.json'):
            raise ValueError('Formal Memory changed the GPU-qualified engine')
        p.save(root/'qualification-gate.json',memory_gate);proof_names.append('qualification-gate.json')
    if recovery is not None:proof_names.append('recovery-proof.json')
    p.save(root/'prepared-proof.json',{name:p.sha(root/name) for name in proof_names})
    p.save(root/'control-parity.json',dict(original_profile=str(REFERENCE/(suite+'-off.json')),
        profile_differences={k:dict(before=original.get(k),after=profile.get(k))
            for k in profile if profile[k]!=original.get(k)},
        original_sample_ids=True,all_kvmem_parameters_equal=profile['kvmem']==original['kvmem'],
        original_kvmem_parameters_except_execution_equal=all(
            profile['kvmem'].get(k)==original['kvmem'].get(k)
            for k in set(profile['kvmem'])|set(original['kvmem']) if k not in EXECUTION_KV_KEYS),
        execution_flags=execution,
        generation_sampling_and_limits_equal_to_previous_vllm=True,
        change=('Per-layer cumulative prefill pressure and sequential Mean-K Query; preserve frozen P per-layer maps/scores.'
                if pressure_runtime is not None else
                'Final Query uses optimized sequential layerwise Mean-K. Native batching/compiled prefill/decode unchanged.')))
    p.save(root/'status.json',dict(status='prepared',suite=suite,expected=len(selected),at=time.time()))
    print(json.dumps(dict(root=str(root),suite=suite,cases=len(selected),qualification=qualification)),flush=True)


def budget_variant_profile(original, select_budget):
    """The budget sweep changes exactly one control of a completed LME run."""
    if type(select_budget) is not int or select_budget not in (24576,16384):
        raise ValueError('This sweep only authorizes 24K and 16K Active KV')
    if (original['kvmem']['select_budget'],original['kvmem']['gen_budget'])!=(36864,16384):
        raise ValueError('Budget reference must use the completed 36K/16K controls')
    result=deepcopy(original)
    result['kvmem']['select_budget']=select_budget
    return result


def prepare_budget_variant(root, source_root, select_budget):
    if root.exists():raise ValueError('Budget variants require a fresh directory')
    if (source_root/'budget-variant.json').exists():raise ValueError('Use the original completed 36K reference')
    fixture,original=verify(source_root)
    audit=p.read(source_root/'execution-audit.json');cleanup=p.read(source_root/'run/cleanup.json')
    if (fixture['suite']!='longmemeval_s' or fixture['qualification'] or fixture['cases']!=102 or
            p.read(source_root/'status.json')['status']!='completed' or audit['status']!='passed' or
            audit['cases']!=102 or cleanup['gpu_processes'] or cleanup['remaining_owned_groups']):
        raise ValueError('Budget sweep requires the completed, released LME102 reference')
    profile=budget_variant_profile(original,select_budget)
    root.mkdir(parents=True)
    shutil.copytree(source_root/'runtime-v1',root/'runtime-v1',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Only the experiment driver changes; the entire inference engine remains
    # byte-identical to the completed reference, including its checkpoint code.
    driver='scripts/run_layerwise_memory.py'
    shutil.copy2(__file__,root/'runtime-v1'/driver)
    old_proof=p.read(source_root/'prepared-proof.json')
    for name in old_proof:
        if name not in ('profile.json','runtime-proof.json'):
            shutil.copy2(source_root/name,root/name)
    p.save(root/'profile.json',profile)
    runtime=root/'runtime-v1'
    proof={str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
           if f.is_file() and '__pycache__' not in f.parts}
    source_proof=p.read(source_root/'runtime-proof.json')
    if set(proof)!=set(source_proof) or any(proof[n]!=source_proof[n] for n in proof if n!=driver):
        raise ValueError('Budget sweep changed the reference inference engine')
    p.save(root/'runtime-proof.json',proof)
    source_names=set(old_proof)|{'prepared-proof.json','status.json','execution-audit.json','run/cleanup.json'}
    p.save(root/'budget-variant.json',dict(source=str(source_root),select_budget=select_budget,
        gen_budget=16384,reference_select_budget=36864,
        source_files={name:p.sha(source_root/name) for name in source_names},
        changed_runtime_files=[driver],fresh_answers=True,
        authorization='User requested Active KV 24K and 16K, Gen KV 16K, all other controls unchanged.'))
    p.save(root/'prepared-proof.json',{name:p.sha(root/name) for name in set(old_proof)|{'budget-variant.json'}})
    p.save(root/'control-parity.json',dict(reference=str(source_root),
        differences={'kvmem.select_budget':dict(before=36864,after=select_budget)},
        original_samples_and_sampling_equal=True,inference_engine_byte_identical=True,
        all_other_profile_parameters_equal=True,gen_budget=16384))
    p.save(root/'status.json',dict(status='prepared',suite='longmemeval_s',expected=102,at=time.time()))
    verify(root)


def verify(root):
    for name,digest in p.read(root/'prepared-proof.json').items():
        if p.sha(root/name)!=digest:raise ValueError('Frozen campaign controls changed: '+name)
    for name,digest in p.read(root/'runtime-proof.json').items():
        if p.sha(root/'runtime-v1'/name)!=digest:raise ValueError('Frozen memory runtime changed: '+name)
    fixture=p.read(root/'fixture.json');profile=p.read(root/'profile.json')
    if (root/'rolling-variant.json').exists():
        from run_layerwise_rolling_memory import verify_variant
        verify_variant(root,fixture,profile)
        return fixture,profile
    if fixture.get('restore_digest_qualification'):
        if (not fixture['qualification'] or fixture['suite']!='memoryagentbench' or
                p.read(root/'verification-profile.json')!=restore_verification_profile(profile)):
            raise ValueError('Restore verification must only add explicit qualification diagnostics')
    if (root/'supplemental-qualification.json').exists():
        supplement=p.read(root/'supplemental-qualification.json')
        if (supplement['resampled_previous_answers'] or
                set(fixture['case_ids']) & set(supplement['preserved_response_ids']) or
                fixture['case_ids']!=supplement['new_case_ids']):
            raise ValueError('Supplemental qualification would resample an existing answer')
        for name,digest in supplement['files'].items():
            if p.sha(Path(name))!=digest:raise ValueError('Preserved qualification evidence changed')
    if fixture.get('pressure_runtime_source') is not None:
        from layerwise_efficiency_gate import verify as verify_efficiency
        if not (root/'efficiency-gate.json').exists():
            raise ValueError('Prepared Memory root lacks efficiency acceptance; prepare a new version after the gate passes')
        verify_efficiency(p.read(root/'efficiency-gate.json'),fixture['pressure_runtime_source'])
    original=p.read(Path(fixture['source_profile']))
    expected=deepcopy(original['kvmem']);overrides=fixture.get('execution_kvmem_overrides',{})
    if not set(overrides)<=EXECUTION_KV_KEYS:raise ValueError('Unknown memory control override')
    expected.update(overrides)
    if (root/'budget-variant.json').exists():
        variant=p.read(root/'budget-variant.json');source_root=Path(variant['source'])
        if source_root.resolve()==root.resolve() or (source_root/'budget-variant.json').exists():
            raise ValueError('Budget reference must be the original completed campaign')
        for name,digest in variant['source_files'].items():
            if p.sha(source_root/name)!=digest:raise ValueError('Frozen budget reference changed: '+name)
        source_fixture,source_profile=verify(source_root)
        if (fixture!=source_fixture or profile!=budget_variant_profile(source_profile,variant['select_budget']) or
                variant['gen_budget']!=16384 or not variant['fresh_answers']):
            raise ValueError('Budget sweep changed more than Active KV')
        source_proof=p.read(source_root/'runtime-proof.json');proof=p.read(root/'runtime-proof.json')
        if (set(proof)!=set(source_proof) or
                any(proof[n]!=source_proof[n] for n in proof if n!='scripts/run_layerwise_memory.py')):
            raise ValueError('Budget sweep inference engine differs from the reference')
        expected['select_budget']=variant['select_budget']
    if profile['kvmem']!=expected:raise ValueError('Original KVMem controls changed')
    for name in ('source_profile','source_fixture'):
        if p.sha(Path(fixture[name]))!=fixture[name+'_sha256']:raise ValueError('Original controls changed')
    for binding in fixture['bindings']:
        if p.sha(Path(binding['request']))!=binding['sha256']:raise ValueError('Original sampling changed')
    if (root/'memory-runtime-reuse.json').exists():
        reuse=p.read(root/'memory-runtime-reuse.json')
        if (p.sha(Path(reuse['source']))!=reuse['source_sha256'] or
                p.sha(root/'runtime-v1'/reuse['target'])!=reuse['source_sha256']):
            raise ValueError('Original Memory execution dependency changed')
    if (root/'qualification-gate.json').exists():
        for name,digest in p.read(root/'qualification-gate.json')['files'].items():
            if p.sha(Path(name))!=digest:raise ValueError('Memory qualification evidence changed')
    if (root/'recovery-proof.json').exists():
        from run_ruler_hybrid_campaign import verify_files
        recovery=p.read(root/'recovery-proof.json')
        if recovery['regenerated_answers']!=0:raise ValueError('Completed answers must not be regenerated')
        verify_files(Path(recovery['source']),recovery['source_files'])
        verify_files(root/'recovered-answers',recovery['recovered_files'])
    return fixture,profile


def infer(root,capture,*,workers=8):
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError('Memory client workers must be an integer in [1, 8]')
    fixture,profile=verify(root);suite=fixture['suite'];kv=profile['kvmem']
    rows=fixture['rows'];bindings={r['id']:r for r in fixture['bindings']}
    reference=Path(fixture['reference']);out=root/'run/answers';out.mkdir(parents=True,exist_ok=True)
    p.save(out/'manifest.json',dict(reference=str(reference),suite=suite,cases=len(rows),rows=rows,
        cohort_sha256=fixture['source_cohort_sha256'],source_cohort_sha256=fixture['source_cohort_sha256'],
        profile_sha256=p.sha(root/'profile.json'),layerwise=True,workers=workers))
    groups={}
    for row in rows:
        group=row['id'] if suite=='longmemeval_s' else row['id'].rsplit('_q',1)[0]
        groups.setdefault(group,[]).append(row)
    credits=CpuCredits(kv['cpu_read_cache_bytes']-16*1024**3)
    lock=Lock();results={}
    compact_fields=('id','status','usage','seconds','finish_reason','hit_output_cap','prefix_reused')
    recovered=root/'recovered-answers/cases'
    if recovered.exists():
        shutil.copytree(recovered,out/'cases')
        for directory in recovered.iterdir():
            record=p.read(directory/'result.json')
            results[record['id']]={k:record[k] for k in compact_fields}
    def progress():
        p.save(out/'summary.json',dict(expected=len(rows),completed=len(results),errors=0,results=list(results.values())))
        p.save(root/'status.json',dict(status='running',suite=suite,completed=len(results),expected=len(rows),at=time.time()))
    def execute_group(group,items):
        pending=[row for row in items if row['id'] not in results]
        if not pending:return
        prefix=min(r['query_begin'] for r in items)//1024*1024 if suite=='memoryagentbench' else 0
        # Rolling changes GPU window capacity, not the possible archived output.
        # Preserve the reference CPU reservation and eight-owner admission.
        reserve=max(kv['gen_budget'],max(bindings[r['id']]['max_tokens'] for r in items)) if kv.get('rolling_active_window') else kv['gen_budget']
        extent=max(r['input_tokens'] for r in items)+2*reserve
        amount=((extent+31)//32)*32*17*4*256*2
        credits.acquire(group,amount);session='layerwise-memory-'+uuid.uuid4().hex;done=0;prefix_state=None
        try:
            for row in pending:
                directory=out/'cases'/row['id'];directory.mkdir(parents=True,exist_ok=True)
                if (directory/'response.json').exists() or (directory/'attempt.json').exists():
                    raise ValueError('Preserve prior attempted answers; explicit technical recovery is required')
                src=source(row);tokens=src['tokens'];binding=bindings[row['id']]
                payload=dict(binding['sampling'],max_tokens=binding['max_tokens'],
                    thinking_token_budget=binding['thinking_token_budget'],model=profile['served_model_name'],
                    prompt=tokens,session_id=session,cache_salt=session,return_token_ids=True,
                    kvmem_query_span=[row['query_begin'],row['query_end']],kvmem_reselect='force')
                if suite=='memoryagentbench':payload['kvmem_replay_span']=[prefix,row['query_end']]
                p.save(directory/'request.json',dict({k:v for k,v in payload.items() if k!='prompt'},
                    token_sha256=row['token_sha256'],prompt_tokens=len(tokens),sampling_sha256=binding['sha256']))
                p.save(directory/'attempt.json',dict(session=session,group=group,at=time.time()))
                offset=CaptureSlice(capture/name for name in ('bounded_refresh.jsonl','layerwise_query.jsonl','action_attention.jsonl',
                                                            'verified_restores_digest.jsonl'))
                started=time.monotonic();http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                req=urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
                    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with http.open(req,timeout=3600) as stream:response=json.load(stream)
                seconds=time.monotonic()-started;p.save(directory/'response.json',compact_response(response,tokens))
                if response['usage']['prompt_tokens']!=len(tokens) or response['choices'][0]['prompt_token_ids']!=tokens:
                    raise ValueError('Original full input identity changed')
                rid=response['id'];terminal=wait_terminal(capture,rid)
                released=None
                if suite=='longmemeval_s':
                    from qualify_graph_niah import wait_owner_released
                    released=wait_owner_released(capture,session,{'response_id':rid},terminal)
                refresh=[e for e in offset.read(capture/'bounded_refresh.jsonl') if e['request_id'].startswith(rid+'-')]
                query=[e for e in refresh if e['selection']=='query']
                if len(query)!=1 or query[0]['query_span']!=payload['kvmem_query_span']:
                    raise ValueError('Original Query range changed')
                events=[e for e in offset.read(capture/'layerwise_query.jsonl') if e.get('request_id','').startswith(rid+'-')]
                layers=[e for e in events if e['event']=='layer_selected']
                if [e['layer'] for e in layers]!=list(range(16)) or any(e['charged_tokens']>kv['select_budget'] for e in layers):
                    raise ValueError('Actual sequential layerwise selection or budgets not proven')
                capacity_audit=memory_generation_capacity_audit(query[0],layers,len(tokens),payload['max_tokens'],kv)
                independent=fixture.get('execution_flags',{}).get('layerwise_prefill_pressure',False)
                pressures=memory_pressure_audit(events,refresh,independent,
                    fixture.get('execution_flags',{}).get('layerwise_buffered_history',False),kv['select_budget'],
                    rolling=fixture.get('execution_flags',{}).get('layerwise_rolling_decode',False))
                if suite=='memoryagentbench':
                    if terminal.get('checkpoint_kind')!='P' or terminal.get('checkpoint_tokens')!=prefix:
                        raise ValueError('Original frozen prefix was not retained')
                    if done and pressures:raise ValueError('Frozen-prefix continuation became a cold rerun')
                    if independent:prefix_state=memory_prefix_audit(events,prefix,prefix_state)
                text=response['choices'][0]['text'];answer=text.split('</think>',1)[-1].strip()
                record=dict(src['info'],id=row['id'],suite=suite,status='completed',answer=answer,
                    raw_completion=text,thinking_closed='</think>' in text,usage=response['usage'],
                    seconds=seconds,prompt_tokens=len(tokens),token_sha256=row['token_sha256'],
                    prompt_sha256=row['prompt_sha256'],query_sha256=row['query_sha256'],
                    # The capture log keeps every event; the result points at this request's
                    # slice (benchmark_client_io.result_events reads it back).
                    original_input_intact=True,attention=query[0],
                    layerwise=dict(layers=layers,events=offset.reference(capture/'layerwise_query.jsonl',rid+'-',events)),
                    pressure_refreshes=len(pressures),finish_reason=response['choices'][0]['finish_reason'],
                    hit_output_cap=response['choices'][0]['finish_reason']=='length',
                    prefix_reused=bool(done),prefix_tokens=prefix,prefix_sha256=digest_tokens(tokens[:prefix]))
                record['generation_capacity']=capacity_audit
                if prefix_state is not None:record['layerwise_prefix']=prefix_state
                if suite=='memoryagentbench' and independent and done and fixture['qualification']:
                    record['prefix_restore_digest']=memory_restore_digest_audit(
                        offset.read(capture/'verified_restores_digest.jsonl'),rid,prefix,prefix_state['layer_tokens'][0])
                if released is not None:record['release']=released
                p.save(directory/'events.json',refresh);p.save(directory/'terminal.json',terminal)
                p.save(directory/'result.json',record)
                done+=1
                with lock:
                    # Full per-layer evidence stays in its immutable case
                    # file. Rewriting it for every later answer would turn
                    # the 1,116-case progress file into quadratic I/O.
                    results[row['id']]={k:record[k] for k in compact_fields}
                    progress()
                    print(json.dumps(dict(id=row['id'],completed=len(results),expected=len(rows),seconds=seconds)),flush=True)
            if suite=='memoryagentbench':
                from graph_v2_protocol import release_owner
                p.save(out/'releases'/(group+'.json'),release_owner(profile,capture,session,tokens[-1]))
        except BaseException:
            credits.cancel();raise
        finally:credits.release(group,amount)
    inherited=set(results)
    started_wall=time.time();started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute_group,g,items) for g,items in groups.items()]
        for future in as_completed(futures):future.result()
    elapsed=time.perf_counter()-started
    fresh=[row for row in rows if row['id'] not in inherited]
    p.save(root/'inference-performance.json',dict(started_at=started_wall,completed_at=time.time(),
        inference_wall_seconds=elapsed,concurrency_limit=workers,completed_cases=len(results),
        fresh_cases=len(fresh),inherited_cases=len(inherited),
        fresh_cases_per_hour=3600*len(fresh)/elapsed,
        fresh_logical_request_tokens=sum(row['input_tokens'] for row in fresh),
        fresh_prefix_reused_cases=sum(bool(results[row['id']]['prefix_reused']) for row in fresh),
        scope='Whole inference dispatch, including CPU admission, prefix import, pressure, Query and decode; '
              'excludes model load. Logical input includes reused P prefixes and is NOT executed GPU prefill. '
              'Inherited first answers are excluded from throughput numerator.'))
    p.save(out/'cpu_admission.json',dict(budget=credits.budget,peak=credits.peak,events=list(credits.events)))
    if set(results)!=set(fixture['case_ids']):raise ValueError('Original memory cohort incomplete')
    if recovered.exists():
        from run_ruler_hybrid_campaign import verify_files
        proof=p.read(root/'recovery-proof.json')
        verify_files(out,{k:v for k,v in proof['recovered_files'].items() if k.startswith('cases/')})
    p.save(root/'execution-audit.json',dict(status='passed',suite=suite,cases=len(results),
        qualification=fixture['qualification'],actual_method='mean_k_sequential_layerwise',
        original_inputs_and_sampling_unchanged=True,
        original_kvmem_budget_and_score_sampling_unchanged=not any((root/name).exists() for name in ('budget-variant.json','rolling-variant.json')),
        authorized_active_budget_variant=p.read(root/'budget-variant.json') if (root/'budget-variant.json').exists() else None,
        authorized_rolling_variant=p.read(root/'rolling-variant.json') if (root/'rolling-variant.json').exists() else None,
        execution_kvmem_overrides=fixture.get('execution_kvmem_overrides',{}),
        pressure_method='accumulated_attention_per_layer' if fixture.get('execution_flags',{}).get('layerwise_prefill_pressure') else 'accumulated_attention',
        inherited_answers=len(list(recovered.iterdir())) if recovered.exists() else 0,
        regenerated_completed_answers=0))


def run(root,*,workers=8):
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError('Memory client workers must be an integer in [1, 8]')
    sys.path[:0]=[str(root/'runtime-v1/src'),str(root/'runtime-v1/scripts')]
    fixture,profile=verify(root)
    if (root/'process.json').exists() or (root/'run/server.log').exists():
        raise ValueError('Preserve the first Memory attempt; use explicit recovery into a fresh root')
    from run_recall_comparison import wait_health,stop_owned,gpu_lock,termination_guard
    import benchmark_block_graph as lifecycle
    from benchmark_query_fps_prebuild_live import server_environment
    runtime=root/'runtime-v1';server=observer=None
    original_environment=server_environment(runtime)
    environment=normalize_cuda_alias(original_environment)
    before=Path(original_environment['CUDA_HOME']);after=Path(environment['CUDA_HOME'])
    p.save(root/'toolchain-alias-audit.json',dict(before=str(before),after=str(after),
        same_installation=before.samefile(after),compiler_sha256=p.sha(after/'bin/nvcc'),
        change='Literal compiler/include path only; same CUDA installation and all model/sampling/KV controls unchanged.'))
    with termination_guard(),gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        if lifecycle.gpu_processes():raise ValueError('GPU has another owner')
        command=server_command(root,fixture)
        (root/'run').mkdir(exist_ok=True)
        try:
            p.save(root/'status.json',dict(status='loading',suite=fixture['suite'],at=time.time()))
            with (root/'run/server.log').open('x') as log:
                server=subprocess.Popen(command,cwd=runtime,env=environment,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            p.save(root/'process.json',dict(pid=os.getpid(),server_pid=server.pid,command=command,at=time.time()))
            wait_health(server,profile)
            if (root/'rolling-variant.json').exists():
                from run_layerwise_rolling_memory import verify_effective_controls
                verify_effective_controls(root,p.read(root/'run/server/launch.json'))
            if fixture.get('restore_digest_qualification'):
                launch=p.read(root/'run/server/launch.json')
                if any(launch['environment'].get(k)!='1' for k in (
                        'KVMEM_VERIFY_SESSION_RESTORE','KVMEM_VERIFY_SESSION_RESTORE_DIGEST')):
                    raise ValueError('Actual server did not enable explicit restore byte verification')
            if fixture.get('performance_observation'):
                observer=subprocess.Popen([sys.executable,'-B',str(runtime/'scripts/gpu_efficiency_monitor.py'),
                    'sample','--output',str(root/'run/gpu.jsonl'),'--interval','1',
                    '--owned-root-pid',str(server.pid)],cwd=runtime,env=environment,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            infer(root,root/'run/server/capture',workers=workers);verify(root)
        except BaseException as error:
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
        finally:
            if observer is not None:
                observer.terminate()
                try:observer.wait(timeout=15)
                except subprocess.TimeoutExpired:observer.kill();observer.wait()
            stop_owned(server)
            p.save(root/'run/cleanup.json',dict(gpu_processes=sorted(lifecycle.gpu_processes()),remaining_owned_groups=[],at=time.time()))
        work=root/'run/server/capture/public_work.json'
        if (profile['kvmem'].get('cpu_seal_hash_workers',1)>1 or profile['kvmem'].get('cpu_direct_record_seal',False)
                or fixture.get('cpu_region_gather_min_bytes') is not None):
            from run_layerwise_matched_performance import archive_backend_audit
            archive=root/'run/server/capture/archive.json'
            p.save(root/'archive-execution-audit.json',dict(
                archive_backend_audit(profile,p.read(archive),fixture.get('cpu_region_gather_min_bytes')),
                source_sha256=p.sha(archive)))
        if work.exists():
            confirmed=p.read(work)
            performance=p.read(root/'inference-performance.json')
            p.save(root/'confirmed-work-performance.json',dict(counter_id=confirmed['counter_id'],
                source_sha256=p.sha(work),completed=confirmed['completed'],
                all_submitted_completed=confirmed['all_submitted_completed'],
                public_prefill_tokens_per_inference_second=confirmed['completed']['public_prefill_tokens']/performance['inference_wall_seconds'],
                scope='CUDA-fence-confirmed public token counts divided by full inference wall time. '
                      'Includes decode/maintenance time; not a pure-prefill kernel rate or logical reused-prefix throughput.'))
        p.save(root/'status.json',dict(status='completed',suite=fixture['suite'],completed=fixture['cases'],at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','prepare-budget','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--suite',choices=('longmemeval_s','memoryagentbench'))
    parser.add_argument('--qualification',action='store_true')
    parser.add_argument('--recover-from',type=Path)
    parser.add_argument('--pressure-runtime',type=Path)
    parser.add_argument('--qualification-source',type=Path)
    parser.add_argument('--qualification-after',type=Path)
    parser.add_argument('--efficiency-source',type=Path)
    parser.add_argument('--efficiency-authorization',type=Path)
    parser.add_argument('--budget-source',type=Path)
    parser.add_argument('--select-budget',type=int,choices=(24576,16384))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root.resolve(),args.suite,args.qualification,args.recover_from,args.pressure_runtime,
                                     args.qualification_source.resolve() if args.qualification_source else None,
                                     args.efficiency_source.resolve() if args.efficiency_source else None,
                                     args.efficiency_authorization.resolve() if args.efficiency_authorization else None,
                                     args.qualification_after.resolve() if args.qualification_after else None)
    elif args.action=='prepare-budget':
        if args.budget_source is None or args.select_budget is None:parser.error('Budget source and select budget are required')
        prepare_budget_variant(args.root.resolve(),args.budget_source.resolve(),args.select_budget)
    else:run(args.root.resolve())
