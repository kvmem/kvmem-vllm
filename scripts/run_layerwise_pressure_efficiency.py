"""Frozen full-input execution ablations; preserve the running formal campaign."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p
from run_layerwise_query_diagnostic import verify
from vllm_kvmem.core.block_runs import candidate_starts

REPO = Path(__file__).resolve().parents[1]
STAGES = ('baseline', 'native_scores', 'single_assembly', 'fast_metadata', 'compiled')


def replace_once(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError('Frozen patch anchor changed: '+str(path))
    path.write_text(text.replace(before, after))


def prepare(root, reference):
    verify(reference)
    if root.exists():
        raise ValueError('Use a new version; never overwrite a frozen comparison')
    root.mkdir(parents=True)
    source_fixture, source_profile = p.read(reference/'fixture.json'), p.read(reference/'profile.json')
    if (source_profile['kvmem']['select_budget'], source_profile['kvmem']['gen_budget'],
            source_fixture['layerwise_workers']) != (36864,16384,8):
        raise ValueError('Reference budgets or workers changed')
    # The first full eight-owner wave contains single-key, multikey, multivalue
    # and multiquery tasks and already has complete original-run evidence.
    rows = source_fixture['rows'][:8]
    for row in rows:
        if not (reference/'formal/sequential/cases'/row['id']/'result.json').exists():
            raise ValueError('Reference first answer is incomplete')
    plans=[]
    for number, name in enumerate(STAGES):
        arm=root/name;runtime=arm/'runtime-v1';arm.mkdir()
        shutil.copytree(reference/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(reference/'upstream',arm/'upstream')
        (arm/'inputs').mkdir()
        for row in rows:
            src=reference/'inputs'/(row['id']+'.i32')
            if p.sha(src)!=row['served_sha256']:raise ValueError('Original input changed')
            shutil.copy2(src,arm/'inputs'/src.name)
        fixture=deepcopy(source_fixture)
        fixture.update(rows=deepcopy(rows),cases=8,cohort='pressure-efficiency-full-input8',
            qualification_only=True,qualification_ids=[r['id'] for r in rows],
            diagnostic_scope='Fresh original 256K inputs; exact pressure/Query block and output-token equivalence.',
            layerwise_pressure_single_assembly=number>=2,layerwise_pressure_fast_metadata=number>=3)
        profile=deepcopy(source_profile)
        profile.update(port=19060+number,served_model_name='pressure-efficiency-'+name)
        if number>=1:
            profile['kvmem'].update(action_native_scores='reuse',action_native_max_rows=192)
            shutil.copy2(REPO/'src/vllm_kvmem/native_attention.py',runtime/'src/vllm_kvmem/native_attention.py')
        if number>=2:
            for module in ['layerwise_query.py','layerwise_pressure.py','layerwise_pressure_efficiency.py','original_rope.py']:
                shutil.copy2(REPO/'src/vllm_kvmem'/module,runtime/'src/vllm_kvmem'/module)
            replace_once(runtime/'src/vllm_kvmem/window_assembly.py',
                "if arena not in ('main', 'refresh', 'layerwise'):",
                "if arena not in ('main', 'refresh', 'layerwise', 'pressure'):")
            anchor="            if self.maintenance is not None:\n                assembler.bandwidth_limiter = self.maintenance.transfer_budget\n"
            replace_once(runtime/'src/vllm_kvmem/async_refresh.py',anchor,
                "            pressure = getattr(self.runner, '_kvmem_layerwise_pressure', None)\n"
                "            if (pressure is not None and pressure.single_assembly and\n"
                "                    job.command.selection == 'pressure'):\n"
                "                from .layerwise_pressure_efficiency import install_single_assembly\n"
                "                install_single_assembly(assembler, pressure, job, self.runtime.names)\n"+anchor)
            serve=runtime/'src/vllm_kvmem/serve.py'
            replace_once(serve,'    args = parser.parse_args(argv)\n',
                "    parser.add_argument('--layerwise-pressure-single-assembly', action='store_true')\n"
                "    parser.add_argument('--layerwise-pressure-fast-metadata', action='store_true')\n"
                '    args = parser.parse_args(argv)\n')
            anchor="        plan['environment']['KVMEM_LAYERWISE_PREFILL_PRESSURE'] = '1'\n"
            replace_once(serve,anchor,anchor+
                "    for enabled, name in ((args.layerwise_pressure_single_assembly, 'SINGLE_ASSEMBLY'),\n"
                "                          (args.layerwise_pressure_fast_metadata, 'FAST_METADATA')):\n"
                "        if enabled:\n"
                "            if not args.layerwise_prefill_pressure:\n"
                "                raise ValueError('Pressure optimization requires per-layer pressure')\n"
                "            plan['environment']['KVMEM_LAYERWISE_PRESSURE_' + name] = '1'\n")
        if name=='compiled':profile['execution_mode']='compiled_graph'
        # Driver changes only select the explicitly frozen execution flags.
        shutil.copy2(REPO/'scripts/run_concurrent_layerwise_ruler.py',runtime/'scripts/run_concurrent_layerwise_ruler.py')
        p.save(arm/'fixture.json',fixture);p.save(arm/'profile.json',profile)
        proof={str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
               if f.is_file() and '__pycache__' not in f.parts}
        p.save(arm/'runtime-proof.json',proof)
        p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=8,
            profile_sha256=p.sha(arm/'profile.json'),fixture_sha256=p.sha(arm/'fixture.json'),
            runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
        p.save(arm/'status.json',dict(status='prepared',completed=0,expected=8))
        verify(arm)
        source_proof=p.read(reference/'runtime-proof.json')
        changes={k:v for k,v in proof.items() if source_proof.get(k)!=v}
        plans.append(dict(name=name,root=str(arm),prepared_sha256=p.sha(arm/'prepared.json'),
                          changed_files=changes))
    shutil.copy2(Path(__file__),root/'launch.py')
    source_files={str(reference/n):p.sha(reference/n) for n in ['fixture.json','profile.json','runtime-proof.json','prepared.json']}
    for row in rows:
        case=reference/'formal/sequential/cases'/row['id']
        for n in ['request.json','response.json','result.json','evidence.json']:
            source_files[str(case/n)]=p.sha(case/n)
    p.save(root/'plan.json',dict(reference=str(reference),stages=plans,source_files=source_files,
        sample_ids=[r['id'] for r in rows],launch_sha256=p.sha(root/'launch.py'),
        no_formal_interruption=True,no_automatic_retry=True,scheduled_monitor=False,
        controls=dict(active_kv=36864,gen_kv=16384,workers=8,block_tokens=32,sink_blocks=41,
            prefill_chunk_tokens=512,thinking=True,max_tokens=8192),
        acceptance='Every pressure event and final Query: identical per-layer selected blocks; identical full output token IDs.'))
    p.save(root/'status.json',dict(status='prepared',stage=None,at=time.time()))


def query_scope(evidence):
    armed=[e for e in evidence['events'] if e['event']=='armed']
    entries=[e for e in evidence['refresh'] if e['selection']=='query']
    if len(armed)!=1 or len(entries)!=1:
        raise ValueError('Missing unique final Query scope')
    a,b=armed[0],entries[0]
    return dict(armed={k:a[k] for k in ('layers','budget','gen_budget','selected_tokens','query_span',
                                      'fixed_blocks','candidate_blocks','actual_method')},
                entry={k:b[k] for k in ('history_begin','selected_tokens','query_tokens','query_span',
                    'query_score_tokens','guided_query_tokens','guided_fallback','query_history_enabled',
                    'mean_k_backend','replay_span')},mandatory_blocks=b['mandatory_fit']['hard_blocks'])


def full_prefix_candidate_proof(evidence, profile):
    """Prove the Query domain from retained prefix geometry, without new inference.

    These qualifications have repository GC disabled, unique accepted records
    in [0, history_begin), and one record per physical block. A complete prefix
    count therefore identifies the entire set. This proof is invalid for GC or
    sparse archives and explicitly rejects those configurations.
    """
    scope=query_scope(evidence);a,b=scope['armed'],scope['entry'];kv=profile['kvmem']
    if kv.get('repository_gc',False) or kv.get('historical_gc'):
        raise ValueError('Full-prefix proof does not support historical eviction')
    block=kv['block_tokens'];end=b['history_begin'];expected=list(range(0,end,block))
    if a['candidate_blocks']!=len(expected):
        raise ValueError('Final Query is missing accepted prefix candidates')
    if a['budget']!=kv['select_budget'] or a['gen_budget']!=kv['gen_budget']:
        raise ValueError('Final Query changed frozen budgets')
    required={i*block for i in scope['mandatory_blocks']}
    domain=set(expected)
    for layer in evidence['layers']:
        selected=layer['selected_starts']
        if (len(selected)!=len(set(selected)) or not set(selected)<=domain or
                not required<=set(selected) or layer['charged_tokens']!=len(selected)*block or
                layer['charged_tokens']>kv['select_budget']):
            raise ValueError('Final Query selection violated its prefix/mandatory/budget scope')
    return dict(verified=True,scope=scope,candidate_starts_sha256=hashlib.sha256(
        json.dumps(expected).encode()).hexdigest(),candidate_count=len(expected),
        evidence_basis='Complete accepted-prefix count with frozen no-GC unique block archive; per-layer selected IDs checked.')


def signature(evidence):
    pressures=[]
    for event in evidence['events']:
        if event['event']=='pressure_selected':
            pressures.append(dict(accepted_history=event['accepted_history'],method=event['actual_method'],
                fallback=event.get('fallback_reason'),layers=[{**{k:l[k] for k in (
                    'layer','selected_starts','charged_tokens','accepted_score_end',
                    'query_samples','sampled_chunks')},'candidate_starts':candidate_starts(l)} for l in event['layers']]))
    queries=[{k:l[k] for k in ('layer','selected_starts','charged_tokens','query_source',
                             'predecessor_layers_completed')} for l in evidence['layers']]
    return pressures,dict(scope=query_scope(evidence),layers=queries)


def compare(arm, baseline):
    verify(arm);verify(baseline)
    fixture=p.read(arm/'fixture.json'); checks=[]
    for row in fixture['rows']:
        cases=[r/'formal/sequential/cases'/row['id'] for r in [baseline,arm]]
        requests=[p.read(c/'request.json') for c in cases]
        for k in ['sampling','input_sha256','query_span']:
            if requests[0][k]!=requests[1][k]:raise ValueError('Frozen input/generation mismatch: '+k)
        evidence=[p.read(c/'evidence.json') for c in cases]
        responses=[p.read(c/'response.json') for c in cases]
        for c,response in zip(cases,responses):
            result=p.read(c/'result.json')
            if result['response_sha256']!=p.sha(c/'response.json'):raise ValueError('First answer changed')
            if result['evidence_sha256']!=p.sha(c/'evidence.json'):raise ValueError('First evidence changed')
        scope_proofs=[full_prefix_candidate_proof(e,p.read(r/'profile.json'))
                      for e,r in zip(evidence,[baseline,arm])]
        before,after=map(signature,evidence)
        differences=[]
        for i in range(max(len(before[0]),len(after[0]))):
            x=before[0][i] if i<len(before[0]) else None;y=after[0][i] if i<len(after[0]) else None
            if x!=y:
                differences.append(dict(event=i,before_history=x and x['accepted_history'],
                    after_history=y and y['accepted_history'],layers=[l for l in range(16)
                        if x is None or y is None or x['layers'][l]!=y['layers'][l]]))
        checks.append(dict(id=row['id'],same_pressure=before[0]==after[0],same_query=before[1]==after[1],
            same_output_tokens=responses[0]['output_ids']==responses[1]['output_ids'],
            same_text=responses[0]['text']==responses[1]['text'],pressure_differences=differences,
            query_candidate_scopes=scope_proofs,
            pressure_events=len(after[0]),source_hashes={str(c/n):p.sha(c/n) for c in cases
                for n in ['request.json','response.json','evidence.json','result.json']}))
    passed=all(all(x[k] for k in ['same_pressure','same_query','same_output_tokens','same_text']) for x in checks)
    report=dict(passed=passed,baseline=str(baseline),arm=str(arm),checks=checks,cases=len(checks),
        performance=p.read(arm/'concurrency-performance.json'),
        baseline_performance=p.read(baseline/'concurrency-performance.json') if (baseline/'concurrency-performance.json').exists() else None,
        at=time.time())
    p.save(arm/'equivalence.json',report)
    return report


def alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')


def run(root):
    import pynvml
    plan=p.read(root/'plan.json');reference=Path(plan['reference'])
    if p.sha(root/'launch.py')!=plan['launch_sha256']:raise ValueError('Frozen queue changed')
    for path,digest in plan['source_files'].items():
        if p.sha(Path(path))!=digest:raise ValueError('Original configuration/answer changed')
    active=root.parent/'value-recall-active.json'
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        while True:
            pointer=p.read(active)
            if pointer.get('queue_root') not in (str(reference),str(root)):
                raise RuntimeError('Another campaign has acquired GPU ownership; leave it untouched')
            state=p.read(reference/'status.json')
            if state['status']=='failed':raise RuntimeError('Formal campaign failed; no automatic restart')
            terminal=p.read(reference/'queue-status.json') if (reference/'queue-status.json').exists() else {}
            if state['status']=='completed' and terminal.get('status')=='completed':
                process=p.read(reference/'process.json')
                pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
                owners=pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if not owners and not any(alive(process[k]) for k in ['pid','server_pid']):break
            p.save(root/'status.json',dict(status='waiting_for_formal',formal_completed=state.get('completed'),at=time.time()))
            time.sleep(30)
        p.save(active,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),
            process=str(root/'process.json'),experiment_type='layerwise-pressure-efficiency',at=time.time()))
        reports=[]
        for stage in plan['stages']:
            arm=Path(stage['root']);verify(arm)
            if p.sha(arm/'prepared.json')!=stage['prepared_sha256']:raise ValueError('Stage proof changed')
            p.save(root/'status.json',dict(status='running',stage=stage['name'],active_root=str(arm),at=time.time()))
            command=[sys.executable,'-B',str(arm/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
            with (arm/'driver.log').open('x') as log:
                child=subprocess.Popen(command,cwd=arm/'runtime-v1',stdout=log,stderr=subprocess.STDOUT)
            p.save(root/'child.json',dict(pid=child.pid,stage=stage['name'],command=command,at=time.time()))
            if child.wait()!=0:raise RuntimeError('Technical failure; retain first answers: '+stage['name'])
            if stage['name']=='baseline':
                # Capture reproducibility against the original run, but compare
                # optimization against this fresh, identically-sized wave.
                historical=compare(arm,reference)
                p.save(root/'historical-reproducibility.json',historical)
            else:
                report=compare(arm,root/'baseline');reports.append(report)
                native=p.read(arm/'formal/sequential/server-attempts/0001/server/capture/native_attention_stats.json')
                counts=native['layer_phase_counts']
                native_ok=(len(native['producer_layers'])==16 and not native['disabled_reason'] and
                    not native['validation_failures'] and counts.get('reuse:prefill',0)>0 and
                    not any(v for k,v in counts.items() if k.startswith('fallback:prefill')))
                p.save(arm/'native-reuse-audit.json',dict(passed=native_ok,statistics=native))
                if not report['passed'] or not native_ok:
                    p.save(root/'comparison.json',dict(passed=False,reports=reports))
                    raise RuntimeError('Equivalence/native-reuse acceptance failed; do not continue: '+stage['name'])
        p.save(root/'comparison.json',dict(passed=True,reports=reports))
        p.save(root/'status.json',dict(status='completed',stages=len(plan['stages']),at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.reference.resolve())
        else:run(root)
    except BaseException as error:
        if root.exists() and args.action=='run':
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
