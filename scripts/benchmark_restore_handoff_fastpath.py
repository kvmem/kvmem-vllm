"""Qualify and compare restore scheduling with identical fixed eight-way inputs.

Finite controller, no DSH session continuation and no recurring monitor. A fresh
control uses the same compiled library as the candidate, with the opt-in off.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import time

import psutil

from scripts import benchmark_agent_metadata_handoff as b
from scripts import run_agent_handoff_queue as q

ROOT = Path(__file__).resolve().parents[1]
CHANGES = (
    'csrc/bindings.cpp', 'src/vllm_kvmem/_native.cpython-312-x86_64-linux-gnu.so',
    'src/vllm_kvmem/async_refresh.py', 'src/vllm_kvmem/async_restore.py',
    'src/vllm_kvmem/connector.py', 'src/vllm_kvmem/maintenance.py',
    'src/vllm_kvmem/options.py', 'src/vllm_kvmem/prompt_checkpoints.py',
    'src/vllm_kvmem/runner.py', 'src/vllm_kvmem/session_worker.py',
)
SUPPORT = ('scripts/benchmark_restore_handoff_fastpath.py',
           'scripts/analyze_agent_metadata_handoff.py',
           'tests/test_restore_handoff_fastpath.py',
           'tests/test_restore_handoff_benchmark.py')


def profiles(control):
    control = copy.deepcopy(control)
    if control['kvmem'].get('restore_handoff_fastpath', False):
        raise ValueError('Control must not already enable the fastpath')
    candidate = copy.deepcopy(control)
    candidate['kvmem']['restore_handoff_fastpath'] = True
    qualification = copy.deepcopy(candidate)
    qualification.update(port=18877, verify_session_restore=True)
    return control, candidate, qualification


def validate(root):
    plan = q.verify_frozen(root)
    source = Path(plan['source'])
    if q.sha(source/'manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Parent evidence binding changed')
    for rel, digest in plan['preserved_hashes'].items():
        if q.sha(source/rel) != digest:
            raise ValueError('Preserved evidence changed: '+rel)
    control, candidate, qualification = profiles(q.read(source/'optimization/after/profile.json'))
    for path, expected in [('optimization/before/profile.json', control),
                           ('optimization/after/profile.json', candidate),
                           ('qualification-profile.json', qualification)]:
        if q.read(root/path) != expected:
            raise ValueError('Unexpected profile/sampling change: '+path)
    for path in ['candidate/runtime', 'optimization/before/runtime', 'optimization/after/runtime']:
        if (root/path).resolve() != (root/'runtime').resolve():
            raise ValueError('Runtime identity changed')
    if q.sha(root/'fixture.json') != q.sha(source/'fixture.json'):
        raise ValueError('Fixed fixture changed')
    original = Path(plan['qualification_fixture'])
    if q.sha(root/'qualification-fixture.json') != q.sha(original):
        raise ValueError('Qualification fixture changed')
    return plan


def prepare(source, output, qualification_fixture):
    original = b.validate(source)
    prior = source/'optimization/after'
    audit, cleanup = q.read(prior/'execution-audit.json'), q.read(prior/'cleanup.json')
    if audit.get('status') != 'passed' or audit.get('responses') != 64 or any(
            cleanup[k] for k in ('remaining_processes', 'remaining_owned_gpu_pids')):
        raise ValueError('Completed, cleaned parent required')
    if output.exists():
        raise ValueError('Never reuse an attempted experiment directory')
    output.mkdir(parents=True)
    runtime = output/'runtime'
    shutil.copytree(prior/'runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    before_hashes = {}
    for rel in CHANGES + SUPPORT:
        if (runtime/rel).exists():
            before_hashes[rel] = q.sha(runtime/rel)
        shutil.copy2(ROOT/rel, runtime/rel)
    # Freeze every native input as build provenance, even if unchanged. No
    # unrelated native source changes may be silently included in the library.
    for path in (ROOT/'csrc').rglob('*'):
        if path.is_file():
            rel = path.relative_to(ROOT)
            if str(rel) != 'csrc/bindings.cpp' and q.sha(path) != q.sha(prior/'runtime'/rel):
                raise ValueError('Unrelated native source changed: '+str(rel))
    for rel in ('fixture.json', 'fixture-source-hashes.json'):
        shutil.copy2(source/rel, output/rel)
    shutil.copy2(qualification_fixture, output/'qualification-fixture.json')
    (output/'candidate').mkdir()
    (output/'candidate/runtime').symlink_to(runtime, target_is_directory=True)
    control, candidate, qualification = profiles(q.read(prior/'profile.json'))
    for name, profile in [('before', control), ('after', candidate)]:
        arm = output/'optimization'/name
        arm.mkdir(parents=True)
        (arm/'runtime').symlink_to(runtime, target_is_directory=True)
        q.save(arm/'profile.json', profile)
        q.save(arm/'status.json', dict(status='prepared'))
    q.save(output/'qualification-profile.json', qualification)
    preserved = {f'optimization/after/{rel}':q.sha(prior/rel) for rel in
                 ('execution-audit.json','cleanup.json','evidence-hashes.json','performance.json')}
    for rel, digest in q.read(prior/'evidence-hashes.json').items():
        if q.sha(prior/rel) != digest:
            raise ValueError('Completed original answer changed')
        preserved[f'optimization/after/{rel}'] = digest
    manifest = {k:original[k] for k in ('python','dependency_state','dependency_expected',
                'dependency_controller','idle_seconds','poll_seconds','model_startup_timeout_seconds')}
    q.save(output/'source-change-audit.json', dict(
        scope='Prepared resident priority, native post-forward fenced reap, native exact token identity',
        replaced_before_sha256=before_hashes,
        replaced_after_sha256={rel:q.sha(runtime/rel) for rel in CHANGES + SUPPORT},
        source_native_inputs={str(p.relative_to(runtime)):q.sha(p) for p in (runtime/'csrc').rglob('*') if p.is_file()},
        shared_library=True, default_enabled=False, no_recall_or_budget_changes=True))
    manifest.update(source=str(source), source_manifest_sha256=q.sha(source/'manifest.json'),
        preserved_hashes=preserved, qualification_fixture=str(qualification_fixture),
        component_only=True, scope='Eight fixed original long histories, eight constrained 64-token continuations; '
        'fresh opt-out control versus opt-in candidate on the same binary. No DSH answer retries. '
        'This is not real Agent wall throughput; original baseline resource-limit outcome remains preserved.',
        files={str(p.relative_to(output)):q.sha(p) for p in runtime.rglob('*') if p.is_file()})
    for rel in ['fixture.json','fixture-source-hashes.json','qualification-fixture.json',
                'qualification-profile.json','source-change-audit.json',
                'optimization/before/profile.json','optimization/after/profile.json']:
        manifest['files'][rel] = q.sha(output/rel)
    q.save(output/'manifest.json', manifest)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    validate(output)


def completion_audit(arm, enabled):
    rows = [json.loads(x) for x in (arm/'run/server/capture/async_restore.jsonl').read_text().splitlines()]
    post = sum(r.get('completion_poll') == 'post_forward' for r in rows)
    if len(rows) != 56 or (not enabled and post):
        raise ValueError('Invalid completion path coverage')
    return dict(status='passed', restores=len(rows), post_forward=post,
                pre_forward=len(rows)-post, enabled=enabled,
                caveat='A post-forward completion still requires the normal CUDA fence and session ownership checks.')


def run(root):
    if ROOT.resolve() != (root/'runtime').resolve():
        raise ValueError('Execute the frozen controller')
    plan = validate(root)
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if q.read(root/'status.json')['status'] != 'prepared':
            raise ValueError('No automatic retries')
        q.save(root/'queue-process.json',dict(**q.identity(psutil.Process()),at=time.time(),
                                             manifest_sha256=q.sha(root/'manifest.json')))
        try:
            q.wait_dependency(root,plan)
            validate(root)
            if q.gpu_pids():
                raise RuntimeError('Foreign GPU owner; preserve it')
            q.save(root/'status.json',dict(status='qualification',at=time.time()))
            q.qualification(root,plan)
            for name in ['before','after']:
                if q.gpu_pids():
                    raise RuntimeError('Foreign GPU owner; preserve it')
                q.save(root/'status.json',dict(status='optimization_running',active=name,at=time.time()))
                b.run_arm(root,name,plan)
                q.save(root/'optimization'/name/'completion-audit.json',
                       completion_audit(root/'optimization'/name,name=='after'))
            from scripts.analyze_agent_metadata_handoff import analyze
            q.save(root/'comparison.json',analyze(root,validator=validate))
            q.save(root/'status.json',dict(status='completed',component_only=True,at=time.time(),
                                          full_agent_ab_passed=False))
        except BaseException as error:
            q.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--qualification-fixture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.source:
        prepare(args.source.resolve(),args.output.resolve(),args.qualification_fixture.resolve())
    else:
        run(args.output.resolve())
