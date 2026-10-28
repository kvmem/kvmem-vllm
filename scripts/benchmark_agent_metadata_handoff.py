"""Fixed-history GPU component comparison for graph message-boundary merging.

Preserves the completed DSH cohort. Original accepted IDs are replayed on CPU
to recover eight real long prefixes. Each GPU arm restores these same prefixes
and emits a constrained 64-token component suffix, so natural generation cannot
change the next request's input. This is not an Agent quality/throughput test.
"""
import argparse
import ast
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import psutil

from scripts import run_agent_handoff_queue as q

ROOT = Path(__file__).resolve().parents[1]


def verify_change(before, after):
    trees = [ast.parse(p.read_text()) for p in (before, after)]
    for tree in trees:
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_merge_spans']
        if len(matches) != 1:
            raise ValueError('Missing unique span merge function')
        tree.body.remove(matches[0])
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError('Candidate changes model code outside message-boundary merging')


def restore_fixture(source, output, plan):
    from transformers import AutoTokenizer
    from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory
    from scripts.dsh_bridge import to_anthropic
    from scripts.dsh_query_epoch import apply_query_provenance
    profile = q.read(source/'candidate/profile.json')
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    planner = Planner(Protocol(source/'protocol.so'), tokenizer, 16384, optimized_handoff=True)
    evidence = q.read(source/'candidate/evidence-hashes.json')
    states = {case['id']:q.read(source/'candidate/cases'/case['id']/'status.json')
              for case in q.read(source/'candidate/task-plan.json')['cases']}
    cutoff = min(state['completed_at'] for state in states.values())
    logs = [json.loads(line) for line in (source/'candidate/bridge/requests.jsonl').read_text().splitlines()]
    fixtures, used = [], {}
    # The renderer/planner files must match the completed experiment exactly.
    for rel in ('benchmarks/deepswe/claude_bridge.py', 'scripts/dsh_bridge.py',
                'scripts/dsh_query_epoch.py', 'src/vllm_kvmem/prompt_spans.py'):
        if q.sha(ROOT/rel) != q.sha(source/'candidate/runtime'/rel):
            raise ValueError('Trace reconstruction implementation changed: '+rel)
    for case in q.read(source/'candidate/task-plan.json')['cases']:
        state = states[case['id']]
        owners = state['bridge']['agent_owners']
        if len(owners) != 1:
            raise ValueError('Require one original Agent owner per task')
        owner = owners[0]
        directory = source/'candidate/bridge'/owner
        paths = sorted(p for p in directory.iterdir() if p.name.isdigit() and (p/'accepted.json').exists())
        # One real common-eight boundary, rather than combining all individual
        # maxima into an artificial workload exceeding the CPU history budget.
        eligible = {row['turn'] for row in logs if row.get('owner') == owner and
                    row['at'] <= cutoff and 'metrics' in row and 'error' not in row}
        selected = max((p for p in paths if int(p.name) in eligible), key=lambda p:int(p.name))
        trajectory = Trajectory()
        for path in paths:
            if int(path.name) != trajectory.turns+1:
                raise ValueError('Original accepted journal has a gap')
            for name in ('request.json', 'plan.json', 'response.json', 'accepted.json'):
                rel = str((path/name).relative_to(source/'candidate'))
                digest = q.sha(path/name)
                if evidence.get(rel) != digest:
                    raise ValueError('Original answer/journal changed: '+rel)
                used[rel] = digest
            body = q.read(path/'request.json')
            payload = to_anthropic(body)
            prepared = planner.prepare(payload, trajectory)
            saved = q.read(path/'plan.json')
            apply_query_provenance(body, prepared)
            if (len(prepared[0]) != saved['prompt_tokens'] or prepared[5] != saved['exact_output_prefix_reused'] or
                    prepared[1]['kvmem_locality_spans'] != saved['kvmem_locality_spans']):
                raise ValueError('Reconstructed prefix/metadata differs from the original submission')
            if path == selected:
                token = q.read(path/'accepted.json')['token_ids'][0]
                if token in tokenizer.all_special_ids:
                    raise ValueError('Component suffix must use a regular original token')
                fixtures.append(dict(case=case['id'], source=str(path), owner=owner,
                    tokens=list(prepared[0]), output_token=token,
                    locality_spans=saved['kvmem_locality_spans'],
                    locality_provenance=saved['kvmem_locality_provenance']))
                break
            wire = q.read(path/'response.json')
            content = to_anthropic(dict(model=body['model'], messages=[wire['choices'][0]['message']]))['messages'][0]['content']
            ids = q.read(path/'accepted.json')['token_ids']
            if len(ids) != wire['usage']['completion_tokens']:
                raise ValueError('Accepted output count changed')
            planner.commit(trajectory, prepared, payload, dict(content=content), ids)
            if trajectory.turns % 20 == 0:
                q.save(output/'fixture-progress.json', dict(case=case['id'], reconstructed_turns=trajectory.turns,
                    target_turn=int(selected.name), completed_cases=len(fixtures), at=time.time()))
        q.save(output/'fixture-progress.json', dict(completed_cases=len(fixtures), at=time.time()))
    q.save(output/'fixture.json', fixtures)
    q.save(output/'fixture-source-hashes.json', used)
    return fixtures


def fixed_input(item, turn):
    # These are component continuations, not new DSH user questions. Every
    # input is predeclared, independent of the output produced by either arm.
    return item['tokens'] + ([item['output_token']]*64 + [item['output_token']]) * turn


def prepare(source, output):
    source = source.resolve()
    plan = q.verify_frozen(source)
    if output.exists():
        raise ValueError('Never reuse an existing experiment root')
    if q.read(source/'candidate/execution-audit.json')['status'] != 'passed':
        raise ValueError('Completed candidate required')
    cleanup = q.read(source/'candidate/cleanup.json')
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
        raise ValueError('Completed candidate was not cleaned up')
    verify_change(source/'candidate/runtime/src/vllm_kvmem/block_graph.py', ROOT/'src/vllm_kvmem/block_graph.py')
    output.mkdir(parents=True)
    q.save(output/'status.json', dict(status='preparing', at=time.time()))
    for name in ('baseline', 'candidate', 'qualification', 'baseline-outcome.json'):
        (output/name).symlink_to(source/name)
    fixtures = restore_fixture(source, output, plan)
    if len(fixtures) != 8 or any(len(f['tokens']) <= 36864 for f in fixtures):
        raise ValueError('Require eight long original prefixes')
    for name in ('before', 'after'):
        arm = output/'optimization'/name
        runtime = arm/'runtime'
        shutil.copytree(source/'candidate/runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for rel in ('scripts/benchmark_agent_metadata_handoff.py', 'tests/test_agent_metadata_handoff.py'):
            shutil.copy2(ROOT/rel, runtime/rel)
        if name == 'after':
            shutil.copy2(ROOT/'src/vllm_kvmem/block_graph.py', runtime/'src/vllm_kvmem/block_graph.py')
        profile = q.read(source/'candidate/profile.json')
        profile['port'] = 18878
        q.save(arm/'profile.json', profile)
        q.save(arm/'status.json', dict(status='prepared'))
    manifest = {key:plan[key] for key in ('python', 'dependency_state', 'dependency_expected',
        'dependency_controller', 'idle_seconds', 'poll_seconds', 'model_startup_timeout_seconds', 'preserved_baseline')}
    manifest.update(source=str(source), source_manifest_sha256=q.sha(source/'manifest.json'),
        optimization_kind='graph_span_merge', component_only=True,
        scope='Fixed accepted-token prefixes from all eight completed DSH tasks, last completed native request '
              'before the first task finished (the common-eight boundary). '
              'Same model/recall/budgets; 8 component turns of 64 output tokens constrained to one original token. '
              'No DSH tools or new task answers. Not an end-to-end Agent speedup or quality comparison.',
        workload=dict(concurrency=8, turns=8, output_tokens=64, temperature=0, seed=1000,
            allowed_token_ids='one original regular token per case', reselect='off'),
        preserved_source_hashes={name:q.sha(source/name) for name in (
            'candidate/evidence-hashes.json', 'candidate/execution-audit.json', 'candidate/cleanup.json',
            'candidate/performance.json', 'baseline-outcome.json')},
        files={str(p.relative_to(output)):q.sha(p) for p in sorted((output/'optimization').rglob('*'))
            if p.is_file() and p.name != 'status.json'})
    manifest['files'].update({name:q.sha(output/name) for name in ('fixture.json', 'fixture-source-hashes.json')})
    q.save(output/'manifest.json', manifest)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate(output)


def validate(root):
    plan = q.verify_frozen(root)
    source = Path(plan['source'])
    if q.sha(source/'manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    for rel, digest in plan['preserved_source_hashes'].items():
        if q.sha(source/rel) != digest:
            raise ValueError('Completed parent evidence changed')
    before, after = [root/'optimization'/name for name in ('before', 'after')]
    verify_change(before/'runtime/src/vllm_kvmem/block_graph.py', after/'runtime/src/vllm_kvmem/block_graph.py')
    if q.read(before/'profile.json') != q.read(after/'profile.json'):
        raise ValueError('Paired profiles differ')
    for rel in ('baseline', 'candidate', 'qualification'):
        if (root/rel).resolve() != (source/rel).resolve():
            raise ValueError('Completed experiment link changed')
    if 'port_recovery' in plan:
        from scripts.recover_agent_metadata_port import verify_recovery
        verify_recovery(root, plan)
    return plan


async def client(root, fixture, profile):
    import httpx
    barrier = asyncio.Barrier(8)
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}', timeout=3600, trust_env=False) as http:
        async def one(index, item):
            for turn in range(8):
                await barrier.wait()
                tokens = fixed_input(item, turn)
                owner = f'fixed-metadata-{index}'
                data = dict(model=profile['served_model_name'], prompt=tokens, session_id=owner, cache_salt=owner,
                    max_tokens=64, temperature=0, seed=1000, ignore_eos=True, return_token_ids=True,
                    allowed_token_ids=[item['output_token']], kvmem_reselect='off',
                    kvmem_locality_spans=item['locality_spans'], kvmem_locality_provenance=item['locality_provenance'])
                started = time.time()
                response = await http.post('/v1/completions', json=data)
                response.raise_for_status()
                body = response.json()
                equal = body['choices'][0].get('token_ids') == [item['output_token']]*64
                q.save(root/'answers'/f'{index}-{turn}.json', dict(case=item['case'], turn=turn,
                    started=started, seconds=time.time()-started, response=body, exact_output_equal=equal,
                    input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
                if not equal:
                    raise ValueError('Fixed component suffix mismatch; preserve attempt, never retry')
        async with asyncio.TaskGroup() as group:
            for index, item in enumerate(fixture):
                group.create_task(one(index, item))


def run_arm(root, name, plan):
    arm = root/'optimization'/name
    if (arm/'started.json').exists():
        raise ValueError('Never restart an attempted component arm')
    q.save(arm/'started.json', dict(at=time.time()))
    owned = q.Owned()
    try:
        profile = q.read(arm/'profile.json')
        server = q.start_model(arm/'run', arm/'profile.json', arm/'runtime', plan['python'], owned)
        q.ready_http(f'http://127.0.0.1:{profile["port"]}/health', server, owned,
            timeout=plan['model_startup_timeout_seconds'], progress_path=arm/'startup.json')
        (arm/'answers').mkdir()
        q.save(arm/'status.json', dict(status='testing', at=time.time()))
        asyncio.run(client(arm, q.read(root/'fixture.json'), profile))
        rows = [json.loads(x) for x in (arm/'run/server/capture/async_restore.jsonl').read_text().splitlines()]
        if len(rows) != 56 or not all(r.get('resident_active_tokens') and r.get('model_cache_hit') for r in rows):
            raise ValueError('Fixed workload did not exercise all 56 resident restores')
        answers = [q.read(p) for p in sorted((arm/'answers').glob('*.json'))]
        if len(answers) != 64 or not all(r['exact_output_equal'] for r in answers):
            raise ValueError('Incomplete fixed trace')
        q.save(arm/'execution-audit.json', dict(status='passed', responses=64, restores=56,
            exact_output_equal=True, component_only=True))
        q.save(arm/'performance.json', dict(
            restore_seconds=q.distribution(r['finished']-r['started'] for r in rows),
            warm_request_seconds=q.distribution(r['seconds'] for r in answers if r['turn']>0),
            cold_request_seconds=q.distribution(r['seconds'] for r in answers if r['turn']==0),
            scope=plan['scope']))
        q.save(arm/'evidence-hashes.json', {str(p.relative_to(arm)):q.sha(p) for p in (arm/'answers').glob('*.json')})
        q.save(arm/'status.json', dict(status='completed', at=time.time()))
    finally:
        cleanup = owned.stop()
        q.save(arm/'cleanup.json', cleanup)
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
        raise RuntimeError('Owned component cleanup incomplete')


def run(root):
    if ROOT.resolve() != (root/'optimization/after/runtime').resolve():
        raise ValueError('Execute frozen component controller')
    plan = validate(root)
    with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if q.read(root/'status.json')['status'] != 'prepared':
            raise ValueError('No component queue restarts')
        q.save(root/'queue-process.json', dict(**q.identity(psutil.Process()), at=time.time(),
            manifest_sha256=q.sha(root/'manifest.json')))
        try:
            q.wait_dependency(root, plan)
            validate(root)
            execution_arms = plan.get('execution_arms', ['before', 'after'])
            if execution_arms != ['before', 'after'] and not (
                    execution_arms == ['after'] and 'port_recovery' in plan):
                raise ValueError('Unsupported component execution arms')
            for name in execution_arms:
                if q.gpu_pids():
                    raise RuntimeError('Foreign GPU owner appeared; preserve it')
                q.save(root/'status.json', dict(status='optimization_running', active=name, at=time.time()))
                run_arm(root, name, plan)
            # Input equality, not just a matching seed, is required across arms.
            for p in (root/'optimization/before/answers').glob('*.json'):
                before, after = q.read(p), q.read(root/'optimization/after/answers'/p.name)
                if before['input_sha256'] != after['input_sha256']:
                    raise ValueError('Paired component inputs differ')
            q.save(root/'optimization-comparison.json', {name:q.read(root/'optimization'/name/'performance.json')
                for name in ('before', 'after')})
            q.save(root/'status.json', dict(status='needs_optimization_review', at=time.time(),
                component_completed=True, full_paired_test_passed=False))
        except BaseException as error:
            q.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source:
        prepare(args.source, args.output.resolve())
    else:
        run(args.output.resolve())
