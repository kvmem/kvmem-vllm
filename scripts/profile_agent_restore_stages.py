"""Profile residual restore host stages on the completed fixed-history fixture.

Reuses all prior answers. Only the existing timing switch changes; this new
component run is for attribution, never another natural Agent answer attempt.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import psutil

from scripts import benchmark_agent_metadata_handoff as benchmark
from scripts import run_agent_handoff_queue as q

ROOT = Path(__file__).resolve().parents[1]


def stage_audit(arm):
    restores = [json.loads(x) for x in (arm/'run/server/capture/async_restore.jsonl').read_text().splitlines()]
    rows = [json.loads(x) for x in (arm/'run/server/capture/restore_host_stages.jsonl').read_text().splitlines()]
    expected = {r['request_id'] for r in restores}
    if len(expected) != 56 or {r['request_id'] for r in rows} != expected:
        raise ValueError('Stage profile does not cover all 56 restores')
    for rid in expected:
        own = [r for r in rows if r['request_id'] == rid]
        names = {s['name'] for r in own for s in r['timing']['stages']}
        if any(r['outcome'] == 'error' for r in own) or not {
                'prepare_claim_window', 'model_state_prepare', 'background_submit', 'publish_restore'}.issubset(names):
            raise ValueError('Incomplete restore stage profile')
    return dict(status='passed', restores=56, timed_rows=len(rows), component_only=True,
                performance_comparison_allowed=False)


def validate(root):
    plan = benchmark.validate(root)
    binding = plan['restore_profile']
    source = Path(binding['source'])
    if q.sha(source/'manifest.json') != binding['source_manifest_sha256']:
        raise ValueError('Completed component manifest changed')
    for rel,digest in binding['preserved_hashes'].items():
        if q.sha(source/rel) != digest:
            raise ValueError('Completed component evidence changed')
    for name in ('before','after'):
        if (root/'optimization'/name).resolve() != (source/'optimization'/name).resolve():
            raise ValueError('Completed component link changed')
    expected = q.read(source/'optimization/after/profile.json')
    expected['kvmem']['restore_stage_timing'] = True
    if q.read(root/'optimization/profile/profile.json') != expected:
        raise ValueError('Diagnostic changed model/budget/recall profile')
    if q.sha(root/'fixture.json') != q.sha(source/'fixture.json'):
        raise ValueError('Diagnostic fixture differs')
    for rel,digest in binding['model_runtime_hashes'].items():
        if q.sha(root/'optimization/profile/runtime'/rel) != digest:
            raise ValueError('Diagnostic model runtime differs')
    return plan


def prepare(source, output):
    plan = benchmark.validate(source)
    if q.live(q.read(source/'queue-process.json')) is not None:
        raise ValueError('Component queue still running')
    if q.read(source/'status.json')['status'] != 'needs_optimization_review':
        raise ValueError('Completed component comparison required')
    for name in ('before','after'):
        arm = source/'optimization'/name
        audit, cleanup = q.read(arm/'execution-audit.json'), q.read(arm/'cleanup.json')
        if audit['status'] != 'passed' or audit['responses'] != 64 or audit['restores'] != 56 or any(
                cleanup[key] for key in ('remaining_processes','remaining_owned_gpu_pids')):
            raise ValueError('Prior component audit/cleanup incomplete')
        evidence = q.read(arm/'evidence-hashes.json')
        if len(evidence) != 64 or any(q.sha(arm/rel) != digest for rel,digest in evidence.items()):
            raise ValueError('Prior answers changed')
    if output.exists():
        raise ValueError('Never reuse a diagnostic root')
    output.mkdir(parents=True)
    for rel in ('baseline','candidate','qualification','baseline-outcome.json'):
        (output/rel).symlink_to(source/rel)
    (output/'optimization').mkdir()
    for name in ('before','after'):
        (output/'optimization'/name).symlink_to(source/'optimization'/name, target_is_directory=True)
    arm = output/'optimization/profile'
    runtime = arm/'runtime'
    shutil.copytree(source/'optimization/after/runtime', runtime, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in ('scripts/profile_agent_restore_stages.py','tests/test_agent_restore_stage_profile.py'):
        shutil.copy2(ROOT/rel, runtime/rel)
    for rel in ('fixture.json','fixture-source-hashes.json','port-recovery-audit.json'):
        shutil.copy2(source/rel, output/rel)
    profile = q.read(source/'optimization/after/profile.json')
    profile['kvmem']['restore_stage_timing'] = True
    q.save(arm/'profile.json', profile)
    q.save(arm/'status.json', dict(status='prepared'))
    preserved = {}
    for name in ('before','after'):
        original = source/'optimization'/name
        for rel in ('status.json','execution-audit.json','evidence-hashes.json','performance.json','cleanup.json'):
            preserved[f'optimization/{name}/{rel}'] = q.sha(original/rel)
        preserved.update({f'optimization/{name}/{rel}':digest for rel,digest in q.read(original/'evidence-hashes.json').items()})
    model_hashes = {str(p.relative_to(source/'optimization/after/runtime')):q.sha(p)
                    for p in (source/'optimization/after/runtime').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    plan['restore_profile'] = dict(source=str(source), source_manifest_sha256=q.sha(source/'manifest.json'),
        preserved_hashes=preserved, model_runtime_hashes=model_hashes)
    plan['optimization_arms'] = ['before','after','profile']
    plan['scope'] = ('Restore stage profiling only: original fixed eight long histories, eight 64-token constrained '
                     'continuations each. Only restore_stage_timing is enabled. No natural Agent task retries, '
                     'no model/recall/budget changes. Diagnostic overhead prevents performance A/B claims.')
    plan['files'].update({str(p.relative_to(output)):q.sha(p) for p in arm.rglob('*') if p.is_file() and p.name != 'status.json'})
    q.save(output/'manifest.json', plan)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate(output)


def run(root):
    if ROOT.resolve() != (root/'optimization/profile/runtime').resolve():
        raise ValueError('Execute frozen diagnostic controller')
    plan = validate(root)
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if q.read(root/'status.json')['status'] != 'prepared':
            raise ValueError('Never restart attempted diagnostic')
        q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()), at=time.time(),
            manifest_sha256=q.sha(root/'manifest.json')))
        try:
            q.wait_dependency(root,plan)
            validate(root)
            if q.gpu_pids():
                raise RuntimeError('Foreign GPU owner appeared; preserve it')
            q.save(root/'status.json',dict(status='optimization_running',active='profile',at=time.time()))
            benchmark.run_arm(root,'profile',plan)
            audit = stage_audit(root/'optimization/profile')
            q.save(root/'optimization/profile/stage-audit.json', audit)
            q.save(root/'status.json',dict(status='needs_optimization_review',component_completed=True,
                restore_stage_profile_completed=True,full_paired_test_passed=False,at=time.time()))
        except BaseException as error:
            q.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.source:
        prepare(args.source.resolve(),args.output.resolve())
    else:
        run(args.output.resolve())
