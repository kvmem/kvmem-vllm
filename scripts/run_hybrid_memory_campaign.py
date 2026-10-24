"""Freeze and run one explicitly scoped hybrid LME/MAB campaign."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import benchmark_block_graph as lifecycle
from benchmark_niah import save, sha
from benchmark_query_fps_prebuild_live import server_environment, wait_health
import hybrid_memory_protocol as policy
from run_ruler_hybrid_campaign import verify_files

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO/'artifacts/fused-stable-quality-20261021-r1'
require = policy.require


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, suite, *, qualification=False, graph_alpha=0.):
    require(not root.exists(),'Use a fresh memory campaign; never overwrite attempted answers')
    require(suite in ('longmemeval_s','memoryagentbench') and
        (not qualification or suite == 'memoryagentbench'),'Unknown memory campaign scope')
    require(type(graph_alpha) in (int,float) and graph_alpha in (0.,.2),'Unsupported hybrid graph alpha')
    name = suite+'-off.json'; manifest = read(REFERENCE/'manifest.json')
    require(sha(REFERENCE/name) == manifest['profile_hashes'][name],'Reference profile bytes changed')
    fixture_name = suite+'-fixture.json'
    require(sha(REFERENCE/fixture_name) == manifest['fixture_hashes'][fixture_name],'Reference cohort bytes changed')
    original = read(REFERENCE/name); profile = deepcopy(original); k = profile['kvmem']
    profile['port'] = 18872 if qualification else (18871 if suite == 'longmemeval_s' else 18873)
    k.update(query_mean_quota_fraction=.5,query_score_mode='key_fps_4',query_fps_halo_blocks=1,
        query_fps_optimized=True,query_fps_prebuild=True,query_fps_prebuild_device='cpu',
        query_fps_cpu_backend='native',query_fps_cpu_workers=16,query_window_cache_bytes=12*1024**3,
        graph_alpha=graph_alpha,graph_aggregation='sum',graph_query_trace_max_bytes=1024**3,
        graph_trace=suite == 'longmemeval_s' or qualification,prefill_physical_coalescing=False)
    policy.validate_profile(profile,policy.infer_arm(profile),suite)
    fixture = read(REFERENCE/fixture_name)
    source = Path(fixture['reference']); cohort = read(source/'cohort.json')
    require(sha(source/'cohort.json') == fixture['source_cohort_sha256'],'Original memory source cohort changed')
    rows = [r for r in cohort['cases'] if r['suite'] == suite]
    require(rows == fixture['rows'],'Frozen source memory rows differ')
    ids = [r['id'] for r in rows]
    if qualification:
        groups = {}
        for row in rows: groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
        repeatable = {g:items for g,items in groups.items() if len(items) >= 2}
        require(repeatable,'No original multi-question prefix exists')
        group = max(repeatable,key=lambda g:max(r['input_tokens'] for r in repeatable[g]))
        ids = [r['id'] for r in repeatable[group][:2]]
        longest = max(rows,key=lambda r:r['input_tokens'])['id']
        if longest not in ids:
            ids.append(longest)
    from vllm_kvmem.serve import build_launch
    build_launch(profile,REPO,root/'preflight-only')
    root.mkdir(parents=True)
    runtime = root/'runtime'
    for name in ('src/vllm_kvmem','scripts','csrc','third_party/qw3'):
        shutil.copytree(REPO/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','setup.py'): shutil.copy2(REPO/name,runtime/name)
    save(root/'runtime-proof.json',dict(files={str(p.relative_to(runtime)):sha(p)
        for p in runtime.rglob('*') if p.is_file()}))
    save(root/'profile.json',profile)
    save(root/'fixture.json',dict(suite=suite,qualification=qualification,case_ids=ids,cases=len(ids),
        reference=str(source),source_cohort_sha256=fixture['source_cohort_sha256'],rows=rows,
        profile_sha256=sha(root/'profile.json'),reference_profile=original,
        source_fixture=str(REFERENCE/fixture_name),source_fixture_sha256=sha(REFERENCE/fixture_name),
        cpu_headroom_gib=16,workers=8,active_groups=8,
        qualification_scope='First two questions of longest reusable prefix plus absolute longest original question; independent of gold/answers' if qualification else None,
        sampling='Original source request per-question seeds and sampling; unchanged MAB output caps; existing LME G cap16384/thinking4096',
        audit_scope='Full-vector native replay' if k['graph_trace'] else
            'Compact per-case quota/CPU/pressure/selected-range checks; full-vector replay in longest-prefix qualification'))
    save(root/'prepared-proof.json',{n:sha(root/n) for n in ('profile.json','fixture.json','runtime-proof.json')})
    save(root/'status.json',dict(status='prepared',cases=len(ids),suite=suite,at=time.time()))


def verify(root):
    verify_files(root,read(root/'prepared-proof.json'))
    verify_files(root/'runtime',read(root/'runtime-proof.json')['files'])
    fixture,profile = read(root/'fixture.json'),read(root/'profile.json')
    require(sha(Path(fixture['reference'])/'cohort.json') == fixture['source_cohort_sha256'] and
        sha(Path(fixture['source_fixture'])) == fixture['source_fixture_sha256'],'Reference cohort changed')
    policy.validate_profile(profile,policy.infer_arm(profile),fixture['suite'])
    if fixture.get('optimization_source'):
        from recover_hybrid_mab import verify_optimization
        verify_optimization(root,fixture)
    if fixture.get('recovery_source'):
        proof = read(root/'recovery-proof.json')
        if proof.get('kind') == 'same_engine_technical_failure':
            from recover_hybrid_mab_failure import verify_recovery
            verify_recovery(root,fixture,profile,proof)
        optimized_mab = (fixture['suite'] == 'memoryagentbench' and
            fixture.get('optimization_source') == fixture['recovery_source'] and
            proof.get('kind') == 'fps_layer_cache')
        require(proof['source'] == fixture['recovery_source'] and proof['regenerated_answers'] == 0 and
            (proof['engine_sources_identical'] or optimized_mab),'Invalid answer recovery proof')
        verify_files(Path(proof['source']),proof['source_files'])
        verify_files(root/'recovered-answers',proof['recovered_files'])
    return fixture,profile


def client_command(root, fixture):
    name = 'benchmark_longmemeval.py' if fixture['suite'] == 'longmemeval_s' else 'benchmark_memory.py'
    command = [sys.executable,'-B',str(root/'runtime/scripts'/name),'--config',str(root/'profile.json'),
        '--reference',fixture['reference'],'--capture',str(root/'run/server/capture'),
        '--output',str(root/'run/answers'),'--workers','8','--cpu-headroom-gib','16','--timeout','3600']
    if fixture['suite'] == 'memoryagentbench':
        command += ['--active-groups','8']
        if fixture['qualification']:
            for ident in fixture['case_ids']: command += ['--case-id',ident]
    if fixture.get('recovery_source'):
        require(fixture['suite'] == 'longmemeval_s' or
            (fixture['suite'] == 'memoryagentbench' and
             (fixture.get('optimization_source') == fixture['recovery_source'] or
              fixture.get('recovery_kind') == 'same_engine_technical_failure')),
            'Unsupported recovery suite')
        command += ['--resume']
    return command


def audit(root):
    fixture, profile = verify(root)
    output = root/'run/answers'; summary = read(output/'summary.json')
    require(summary['completed'] == fixture['cases'] and not summary['errors'],'Incomplete fixed memory cohort')
    directories = {p.name:p for p in (output/'cases').iterdir() if p.is_dir()}
    require(set(directories) == set(fixture['case_ids']),'Actual memory question set changed')
    rows = {r['id']:r for r in fixture['rows']}; hashes = {}; native = 0; warm = []
    for ident in fixture['case_ids']:
        directory = directories[ident]; result = read(directory/'result.json'); row = rows[ident]
        require(result['status'] == 'completed' and result['original_input_intact'] and
            result['token_sha256'] == row['token_sha256'] and result['prompt_tokens'] == row['input_tokens'],
            'Original full memory input identity differs')
        evidence = result['graph_v2']; cpu = evidence['cpu_fps_audit']
        if profile['kvmem']['graph_alpha']:
            query_graph = [g for g in evidence['graph'] if g['phase'] == 'query']
            require(len(query_graph) == 1,'Final independent graph proof missing')
            policy.ruler.audit_quota_propagation(evidence['query'],query_graph[0],profile)
        require(evidence['quota']['mean_fraction'] == .5 and not result.get('hit_output_cap',False),
            'Hybrid ratio changed or LME hit output cap')
        require(bool(evidence['native_replay_audit']) is bool(profile['kvmem']['graph_trace']),
            'Declared full/compact audit scope differs')
        native += cpu['query_native']['fps_cpu_native_groups']+cpu['background_native']['fps_cpu_native_groups']
        if result.get('prefix_reused'):
            warm.append(dict(id=ident,hit_blocks=cpu['cache_hit_blocks'],
                rebuilt_blocks=cpu['cache_rebuilt_blocks'],prefix_tokens=result['prefix_tokens']))
        for path in directory.iterdir():
            if path.is_file(): hashes[str(path.relative_to(root))] = sha(path)
    if fixture['suite'] == 'memoryagentbench':
        releases = list((output/'releases').glob('*.json'))
        require(releases and all(read(p)['released'] for p in releases),'Frozen P owners not fully released')
        if fixture['qualification']:
            expected_warm = fixture.get('efficiency_expected_warm',1) if fixture.get('efficiency_pilot') else 1
            require(type(expected_warm) is int and expected_warm in (0,1) and len(warm) == expected_warm and
                all(w['hit_blocks'] >= w['prefix_tokens']//32 for w in warm),
                'Longest-prefix continuation did not reuse its immutable representative cache')
    cleanup = read(root/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],'Memory GPU cleanup incomplete')
    report = dict(status='passed',cases=fixture['cases'],suite=fixture['suite'],qualification=fixture['qualification'],
        case_hashes=hashes,original_inputs_unchanged=True,mean_fraction=.5,native_cpu_groups=native,
        query_graph_alpha=profile['kvmem']['graph_alpha'],query_graph_aggregation='sum',
        query_graph_order='independent_graph_then_quota',
        warm_prefixes=warm,cpu_fps_construction=True,pressure_cumulative_attention=True,
        audit_scope=fixture['audit_scope'],gpu_cleanup=True)
    if fixture.get('recovery_source'):
        proof = read(root/'recovery-proof.json')
        verify_files(output,{name:digest for name,digest in proof['recovered_files'].items()
            if name.startswith('cases/')})
        report['recovery'] = dict(source=proof['source'],inherited_cases=len(proof['recovered_ids']),
            original_successes_unchanged=proof['original_successes_unchanged'],
            repaired_audit_cases=len(proof['repaired_audit_ids']),regenerated_inherited_answers=0,
            new_inference_cases=fixture['cases']-len(proof['recovered_ids']),
            scope=('Bounded per-layer FPS Key residency; exact scoring component qualification; '
                'original completed responses unchanged; only unfinished cases inferred.'
                if proof.get('kind') == 'fps_layer_cache' else
                'Same inference/scoring sources and profile; preserved technical failure, only unfinished cases inferred.'
                if proof.get('kind') == 'same_engine_technical_failure' else
                'Same inference/scoring sources and profile; only post-inference audit budget checks corrected.'))
        if fixture.get('diagnostic_environment'):
            report['recovery']['diagnostic_environment'] = fixture['diagnostic_environment']
            report['recovery']['timing_scope'] = 'Synchronous CUDA error localization; not a throughput measurement.'
    if fixture.get('optimization_source'):
        report['optimization'] = read(root/'optimization-proof.json')
    save(root/'execution-audit.json',report)
    return report


def run(root):
    require(Path(__file__).resolve() == root/'runtime/scripts'/Path(__file__).name,'Use frozen controller only')
    fixture,profile = verify(root)
    with (root/'attempt.json').open('x') as stream: json.dump(dict(pid=os.getpid(),at=time.time()),stream)
    target = root/'run'; target.mkdir(exist_ok=False); server = worker = None
    if fixture.get('recovery_source'):
        shutil.copytree(root/'recovered-answers',target/'answers')
    env = server_environment(root/'runtime'); began = time.monotonic()
    if fixture.get('diagnostic_environment'):
        require(fixture.get('recovery_kind') == 'same_engine_technical_failure' and
            fixture['diagnostic_environment'] == {'CUDA_LAUNCH_BLOCKING':'1'},
            'Unsupported recovery diagnostic environment')
        env.update(fixture['diagnostic_environment'])
        save(target/'diagnostic-environment.json',fixture['diagnostic_environment'])
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock,lifecycle.cleanup_on_termination():
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not lifecycle.gpu_processes(),'Foreign GPU owner; preserve it')
            try:
                command = [sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(root/'profile.json'),
                    '--run-dir',str(target/'server')]
                with (target/'server.log').open('x') as log:
                    server = subprocess.Popen(command,cwd=root/'runtime',env=env,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                save(target/'server-process.json',dict(pid=server.pid,command=command))
                save(root/'status.json',dict(status='loading',cases=fixture['cases'],at=time.time()))
                wait_health(server,profile)
                command = client_command(root,fixture)
                with (target/'client.log').open('x') as log:
                    worker = subprocess.Popen(command,cwd=root/'runtime',env=env,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                save(target/'client-process.json',dict(pid=worker.pid,command=command))
                save(root/'status.json',dict(status='testing',cases=fixture['cases'],at=time.time()))
                while worker.poll() is None:
                    require(server.poll() is None,'Model exited during memory inference')
                    time.sleep(5)
                require(worker.returncode == 0,'Memory client failed; preserve attempt and evidence')
            finally:
                remaining = lifecycle.stop_owned_groups((worker,server)); gpu = sorted(lifecycle.gpu_processes())
                save(target/'cleanup.json',dict(remaining_owned_groups=remaining,gpu_processes=gpu))
                require(not remaining and not gpu,'Owned memory GPU cleanup failed')
        audit(root)
        save(root/'status.json',dict(status='completed',cases=fixture['cases'],seconds=time.monotonic()-began,at=time.time()))
    except BaseException as error:
        save(root/'failure.json',dict(error=repr(error),at=time.time()))
        save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--suite',choices=('longmemeval_s','memoryagentbench'))
    parser.add_argument('--qualification',action='store_true')
    parser.add_argument('--graph-alpha',type=float,choices=(0.,.2),default=0.)
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'prepare': prepare(root,args.suite,qualification=args.qualification,graph_alpha=args.graph_alpha)
    else: run(root)
