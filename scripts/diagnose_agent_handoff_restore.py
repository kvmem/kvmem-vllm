"""Frozen eight-session restore diagnostic from the completed handoff qualification.

Replays exact fixed prompts; never submits a DSH task or replaces an answer.
Stage timing is diagnostic only and does not establish an Agent speedup.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import psutil

from scripts import run_agent_handoff_queue as queue

ROOT = Path(__file__).resolve().parents[1]


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def fixed_requests(fixture, qualification):
    """Pin every prompt to prior accepted IDs, not this diagnostic's generation."""
    if len(fixture) != 8:
        raise ValueError('Require the original eight sessions')
    result = []
    for index, item in enumerate(fixture):
        tokens = list(item['tokens'])
        turns = []
        for turn in range(3):
            original = queue.read(qualification / f'{index}-{turn}.json')
            if original['input_sha256'] != token_hash(tokens):
                raise ValueError('Original qualification prompt binding changed')
            ids = original['response']['choices'][0]['token_ids']
            if len(ids) != 64 or original['response']['usage']['completion_tokens'] != 64:
                raise ValueError('Require original 64-token component outputs')
            turns.append(dict(tokens=list(tokens), expected_output=list(ids)))
            tokens += ids + item['suffix']
        result.append(dict(case=item['case'], turns=turns))
    return result


def prepare(source, output):
    source = source.resolve()
    if output.exists():
        raise ValueError('Never reuse an attempted diagnostic directory')
    plan = queue.verify_frozen(source)
    if queue.read(source/'candidate/execution-audit.json')['status'] != 'passed':
        raise ValueError('Candidate execution audit missing')
    cleanup = queue.read(source/'candidate/cleanup.json')
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:
        raise ValueError('Candidate not cleaned up')
    requests = fixed_requests(queue.read(source/'qualification-fixture.json'), source/'qualification')
    profile = queue.read(source/'candidate/profile.json')
    original_profile = json.loads(json.dumps(profile))
    profile['port'] = 18878
    profile['kvmem']['restore_stage_timing'] = True
    output.mkdir(parents=True)
    shutil.copytree(source/'candidate/runtime', output/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Only diagnostic controller/tests change; the model runtime remains frozen.
    for rel in ('scripts/diagnose_agent_handoff_restore.py', 'tests/test_agent_handoff_restore_diagnostic.py'):
        shutil.copy2(ROOT/rel, output/'runtime'/rel)
    queue.save(output/'profile.json', profile)
    queue.save(output/'reference-profile.json', original_profile)
    queue.save(output/'requests.json', requests)
    manifest = {k:plan[k] for k in ('python', 'dependency_state', 'dependency_expected',
        'dependency_controller', 'idle_seconds', 'poll_seconds', 'model_startup_timeout_seconds')}
    manifest.update(source=str(source), source_manifest_sha256=queue.sha(source/'manifest.json'),
        diagnostic_only=True, performance_claim_allowed=False,
        scope='Eight real 45K-52K qualification inputs, three fixed turns of 64 accepted tokens. '
              'Restoration timing only; excludes bridge/tool runtime and does not measure full Agent throughput.',
        source_proof={str(p.relative_to(source)):queue.sha(p) for p in (
            source/'candidate/execution-audit.json', source/'candidate/cleanup.json',
            source/'candidate/evidence-hashes.json', source/'qualification-fixture.json')},
        files={str(p.relative_to(output)):queue.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    queue.save(output/'manifest.json', manifest)
    queue.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate(output)


def validate(root):
    plan = queue.verify_frozen(root)
    if plan.get('diagnostic_only') is not True or plan.get('performance_claim_allowed') is not False:
        raise ValueError('Diagnostic cannot be presented as an Agent performance comparison')
    source = Path(plan['source'])
    if queue.sha(source/'manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Source experiment manifest changed')
    for rel, digest in plan['source_proof'].items():
        if queue.sha(source/rel) != digest:
            raise ValueError('Source proof changed: '+rel)
    reference = queue.read(root/'reference-profile.json')
    expected = json.loads(json.dumps(reference))
    expected['port'] = 18878
    expected['kvmem']['restore_stage_timing'] = True
    if queue.read(root/'profile.json') != expected:
        raise ValueError('Diagnostic changed model/recall settings')
    return plan


async def requests(root, profile, fixture):
    import httpx
    barrier = asyncio.Barrier(8)
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}', timeout=900, trust_env=False) as client:
        async def one(index, item):
            for turn, fixed in enumerate(item['turns']):
                await barrier.wait()
                owner = f'restore-stage-{index}'
                payload = dict(model=profile['served_model_name'], prompt=fixed['tokens'],
                    session_id=owner, cache_salt=owner, max_tokens=64, temperature=0, seed=1000,
                    ignore_eos=True, return_token_ids=True, kvmem_reselect='off')
                started = time.time()
                response = await client.post('/v1/completions', json=payload)
                response.raise_for_status()
                body = response.json()
                ids = body['choices'][0].get('token_ids')
                record = dict(case=item['case'], turn=turn, input_sha256=token_hash(fixed['tokens']),
                    started=started, seconds=time.time()-started, response=body,
                    exact_output_equal=ids == fixed['expected_output'])
                queue.save(root/'answers'/f'{index}-{turn}.json', record)
                # Do not change later inputs or retry when a fixed trace diverges.
                if not record['exact_output_equal']:
                    raise ValueError('Fixed-trace output diverged; preserve diagnostic and do not retry')
        async with asyncio.TaskGroup() as group:
            for index, item in enumerate(fixture):
                group.create_task(one(index, item))


def run(root):
    if ROOT.resolve() != (root/'runtime').resolve():
        raise ValueError('Execute the frozen diagnostic controller')
    plan = validate(root)
    with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if queue.read(root/'status.json')['status'] != 'prepared' or (root/'started.json').exists():
            raise ValueError('No automatic diagnostic restarts')
        queue.save(root/'queue-process.json', dict(**queue.identity(psutil.Process()),
            at=time.time(), manifest_sha256=queue.sha(root/'manifest.json')))
        owned = queue.Owned()
        try:
            queue.wait_dependency(root, plan)
            validate(root)
            if queue.gpu_pids():
                raise RuntimeError('GPU ownership changed')
            queue.save(root/'started.json', dict(at=time.time()))
            queue.save(root/'status.json', dict(status='starting', at=time.time()))
            profile = queue.read(root/'profile.json')
            server = queue.start_model(root/'run', root/'profile.json', root/'runtime', plan['python'], owned)
            queue.ready_http(f'http://127.0.0.1:{profile["port"]}/health', server, owned,
                timeout=plan['model_startup_timeout_seconds'], progress_path=root/'run/startup.json')
            (root/'answers').mkdir()
            queue.save(root/'status.json', dict(status='testing', at=time.time()))
            asyncio.run(requests(root, profile, queue.read(root/'requests.json')))
            captures = root/'run/server/capture'
            restores = [json.loads(x) for x in (captures/'async_restore.jsonl').read_text().splitlines()]
            stages = [json.loads(x) for x in (captures/'restore_host_stages.jsonl').read_text().splitlines()]
            if len(restores) != 16 or {r['request_id'] for r in restores} != {r['request_id'] for r in stages}:
                raise ValueError('Missing restore stage coverage')
            queue.save(root/'execution-audit.json', dict(status='passed', responses=24,
                exact_output_equal=True, restores=16, stage_records=len(stages),
                resident_restores=sum(bool(r.get('resident_active_tokens')) for r in restores)))
            queue.save(root/'status.json', dict(status='diagnostic_completed', at=time.time()))
        except BaseException as error:
            queue.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            queue.save(root/'cleanup.json', owned.stop())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source:
        prepare(args.source, args.output.resolve())
    else:
        run(args.output.resolve())
