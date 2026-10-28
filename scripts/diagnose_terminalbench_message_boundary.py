"""Fixed-history metadata fallback qualification, without regenerating task answers."""
import argparse
import asyncio
from pathlib import Path
import shutil
import time

import httpx
import run_agent_handoff_queue as q
from run_terminalbench_suite import hashes
import diagnose_terminalbench_query_import as driver

REPO = Path(__file__).resolve().parents[1]


def prepare(root, source, fixture):
    if root.exists():
        raise ValueError('New qualification root required')
    original = q.read(source/'manifest.json')['frozen']['runtime']
    if hashes(source/'runtime') != original:
        raise ValueError('Frozen source changed')
    root.mkdir(parents=True)
    shutil.copytree(source/'runtime', root/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    change = 'benchmarks/deepswe/claude_bridge.py'
    shutil.copy2(REPO/change, root/'runtime'/change)
    shutil.copy2(__file__, root/'runtime/scripts'/Path(__file__).name)
    shutil.copy2(fixture, root/'fixture.json')
    if q.read(root/'fixture.json')['bridge_sha256'] != q.sha(root/'runtime'/change):
        raise ValueError('Fixture was not produced by the frozen bridge')
    profile = q.read(source/'profile.json')
    profile.update(port=18881, verify_session_restore=True, verify_session_restore_digest=True)
    q.save(root/'profile.json', profile)
    q.save(root/'manifest.json', dict(source=str(source), runtime=hashes(root/'runtime'),
        model_changes=[], bridge_changes=[change], cuda_launch_blocking=True,
        files={name: q.sha(root/name) for name in ('profile.json', 'fixture.json')},
        scope='Eight fixed-history owners: canonical fallback to P then appended tool restore to M. '
              'Warm setup omits its Query to capture a branch-safe predecode P boundary; '
              'canonical fallback keeps its original Query and all input tokens/metadata. '
              'Only metadata producer changes. Fixed output token830; component correctness, not Agent performance.'))


async def workload(root, profile):
    fixture = q.read(root/'fixture.json')
    answers = []
    histories = {}
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}',
                                 trust_env=False, timeout=3600) as client:
        async def call(owner, phase, tokens, plan):
            session = root.name + '-' + str(owner)
            payload = dict(model=profile['served_model_name'], session_id=session, cache_salt=session,
                prompt=tokens, max_tokens=64, seed=1000, temperature=0,
                allowed_token_ids=[830], ignore_eos=True, return_token_ids=True, **plan)
            start = time.time()
            response = await client.post('/v1/completions', json=payload)
            if response.status_code != 200:
                raise RuntimeError(response.text)
            data = response.json()
            q.save(root/f'{phase}-{owner}.json', data)
            ids = data['choices'][0]['token_ids']
            if ids != [830]*64 or data['usage']['completion_tokens'] != 64:
                raise ValueError('Incomplete fixed output')
            histories[owner] = tokens + ids
            answers.append(dict(owner=owner, phase=phase, id=data['id'], prompt_tokens=len(tokens),
                                accepted=len(ids), start=start, end=time.time()))
            q.save(root/'answers.json', answers)
        for phase, name in [('warm', '00004'), ('canonical', '00005')]:
            case = fixture[name]
            plan = dict(case['plan'])
            if phase == 'warm':
                # Include all three historical assistant ends in the P view,
                # without replaying prior Agent answers. The actual boundary
                # comes from branch_safe_checkpoint, not floor(prompt/block).
                plan.pop('kvmem_query_span', None)
                plan['kvmem_reselect'] = 'off'
            await asyncio.gather(*(call(i, phase, list(case['tokens']), plan) for i in range(8)))
        for i in range(8):
            prefix = histories[i]
            plan = dict(fixture['00005']['plan'])
            plan.pop('kvmem_query_span', None)
            plan['kvmem_reselect'] = 'off'
            plan['kvmem_locality_spans'] = plan['kvmem_locality_spans'] + [
                dict(begin=len(prefix), end=len(prefix)+32, role='tool')]
            histories[i] = (prefix + [830]*32, plan)
        await asyncio.gather(*(call(i, 'continued', *histories[i]) for i in range(8)))
        for i in range(8):
            session = root.name + '-' + str(i)
            response = await client.post('/v1/completions', json=dict(model=profile['served_model_name'],
                session_id=session, cache_salt=session, prompt=[151645], max_tokens=1,
                kv_transfer_params=dict(kvmem_release_session=True)))
            response.raise_for_status()
            q.save(root/f'release-{i}.json', response.json())
    import json
    capture = root/'run/server/capture'
    proofs = [json.loads(line) for line in (capture/'verified_restores_digest.jsonl').read_text().splitlines()]
    if len(proofs) != 16 or not all(row['sha256_equal'] and row['recurrent_byte_equal'] for row in proofs):
        raise ValueError('Missing exact KV/state restoration proof')
    for answer in answers:
        if answer['phase'] == 'warm':
            continue
        matched = [row for row in proofs if row['request_id'].startswith(answer['id']+'-')]
        from vllm_kvmem.request_plan import RequestPlan
        warm_plan = dict(fixture['00004']['plan'])
        warm_plan.pop('kvmem_query_span', None)
        warm_plan['kvmem_reselect'] = 'off'
        expected = (RequestPlan.parse(warm_plan, len(fixture['00004']['tokens']), 32, 16384)
                    .prefill_checkpoint(len(fixture['00004']['tokens']))
                    if answer['phase'] == 'canonical' else len(fixture['00005']['tokens'])+63)
        if len(matched) != 1 or matched[0]['history_tokens'] != expected:
            raise ValueError('Expected P fallback or M continuation not exercised')
    q.save(root/'execution-audit.json', dict(passed=True, replies=len(answers), accepted=1536,
        canonical_fallbacks=8, continued_restores=8, verified_restores=len(proofs), session_releases=8,
        complete_fixed_outputs=True, model_sources_unchanged=True,
        bridge_sha256=q.read(root/'fixture.json')['bridge_sha256'],
        limitation='Component-only constrained outputs; prior CUDA cause remains unconfirmed.'))
    q.save(root/'evidence-hashes.json', {str(path.relative_to(root)): q.sha(path)
        for path in root.rglob('*.json') if path.is_file() and 'runtime' not in path.parts
        and path.name not in ('status.json', 'process.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root.resolve(), args.source.resolve(), args.fixture.resolve())
    else:
        # Reuse the qualified global GPU lock, 60-second idle gate, startup
        # heartbeat and exact-owned-process cleanup. No scheduler is created.
        driver.workload = workload
        driver.run(args.root.resolve())
