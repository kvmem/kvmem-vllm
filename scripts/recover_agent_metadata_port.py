"""Continue only an unlaunched after arm following the audited port preflight failure.

The completed before arm and all failed startup evidence stay in their original
directories. Model code, fixture, profiles, seeds and output constraints do not
change. This is deliberately not a general partial-answer resume facility.
"""
import argparse
import ast
from pathlib import Path
import shutil
import time

from scripts import benchmark_agent_metadata_handoff as benchmark
from scripts import run_agent_handoff_queue as q


def completed_before(source, probe=q.live):
    arm = source/'optimization/before'
    audit = q.read(arm/'execution-audit.json')
    if (q.read(arm/'status.json').get('status') != 'completed' or
            audit.get('status') != 'passed' or audit.get('responses') != 64 or
            audit.get('restores') != 56 or audit.get('exact_output_equal') is not True):
        raise ValueError('Before component is not fully audited')
    cleanup = q.read(arm/'cleanup.json')
    if (cleanup.get('remaining_processes') != [] or cleanup.get('remaining_owned_gpu_pids') != [] or
            any(probe(row) is not None for row in cleanup.get('owned', []))):
        raise ValueError('Before component cleanup incomplete')
    expected = {f'answers/{index}-{turn}.json' for index in range(8) for turn in range(8)}
    evidence = q.read(arm/'evidence-hashes.json')
    if set(evidence) != expected or set(str(p.relative_to(arm)) for p in (arm/'answers').glob('*.json')) != expected:
        raise ValueError('Before answer cohort changed')
    fixture = q.read(source/'fixture.json')
    if len(fixture) != 8:
        raise ValueError('Original fixture cohort changed')
    for index, item in enumerate(fixture):
        for turn in range(8):
            rel = f'answers/{index}-{turn}.json'
            row = q.read(arm/rel)
            tokens = benchmark.fixed_input(item, turn)
            digest = benchmark.hashlib.sha256(benchmark.json.dumps(tokens).encode()).hexdigest()
            if (q.sha(arm/rel) != evidence[rel] or row['input_sha256'] != digest or
                    row['case'] != item['case'] or row['turn'] != turn or
                    row['response']['choices'][0]['token_ids'] != [item['output_token']]*64 or
                    row.get('exact_output_equal') is not True):
                raise ValueError('Before answer/input evidence changed')
    return q.read(arm/'performance.json')


def validate_source(source, probe=q.live):
    plan = benchmark.validate(source)
    if 'port_recovery' in plan:
        raise ValueError('Nested or repeated recovery is unsupported')
    state = q.read(source/'status.json')
    if state.get('status') != 'failed' or state.get('error') != "OSError(98, 'Address already in use')":
        raise ValueError('Source did not fail at the supported port preflight')
    if probe(q.read(source/'queue-process.json')) is not None:
        raise ValueError('Source queue is still alive')
    completed_before(source, probe)
    arm = source/'optimization/after'
    cleanup = q.read(arm/'cleanup.json')
    if (q.read(arm/'status.json').get('status') != 'prepared' or not (arm/'started.json').is_file() or
            cleanup.get('owned') != [] or cleanup.get('remaining_processes') != [] or
            cleanup.get('remaining_owned_gpu_pids') != [] or
            any((arm/p).exists() for p in ('run', 'answers', 'startup.json', 'execution-audit.json', 'performance.json'))):
        raise ValueError('After arm may have executed; cannot use startup-only recovery')
    log = (source/'queue.log').read_text()
    if 'require_free_port' not in log or "Address already in use" not in log:
        raise ValueError('Missing port preflight traceback')
    return plan


def replace_port_check(original, fixed):
    """Import only the repaired port helper, never unrelated working-tree edits."""
    def node(text):
        return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'require_free_port')
    old, new = node(original), node(fixed)
    lines = original.splitlines(keepends=True)
    return ''.join(lines[:old.lineno-1] + fixed.splitlines(keepends=True)[new.lineno-1:new.end_lineno] + lines[old.end_lineno:])


def verify_recovery(root, plan, probe=q.live):
    binding = plan['port_recovery']
    source = Path(binding['source'])
    proof = root/binding['proof']
    if q.sha(proof) != binding['sha256'] or plan.get('execution_arms') != ['after']:
        raise ValueError('Recovery binding changed')
    audit = q.read(proof)
    if audit.get('source') != str(source) or audit.get('reused_completed_arms') != ['before']:
        raise ValueError('Recovery source changed')
    for rel, digest in audit['preserved_hashes'].items():
        if q.sha(source/rel) != digest:
            raise ValueError('Preserved component evidence changed: '+rel)
    if (root/'optimization/before').resolve() != (source/'optimization/before').resolve():
        raise ValueError('Completed before arm reference changed')
    for rel in ('fixture.json', 'fixture-source-hashes.json', 'optimization/after/profile.json'):
        if q.sha(root/rel) != q.sha(source/rel):
            raise ValueError('Fixed inputs/profile changed')
    for rel, digest in audit['model_runtime_hashes'].items():
        if q.sha(root/'optimization/after/runtime'/rel) != digest:
            raise ValueError('After model runtime changed')
    completed_before(source, probe)


def prepare(source, output, repository, probe=q.live):
    source, output = source.resolve(), output.resolve()
    plan = validate_source(source, probe)
    if output.exists():
        raise ValueError('Never reuse an existing recovery root')
    output.mkdir(parents=True)
    for rel in ('baseline', 'candidate', 'qualification', 'baseline-outcome.json'):
        (output/rel).symlink_to(source/rel)
    (output/'optimization').mkdir()
    (output/'optimization/before').symlink_to(source/'optimization/before', target_is_directory=True)
    after = output/'optimization/after'
    runtime = after/'runtime'
    shutil.copytree(source/'optimization/after/runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    changed = ['scripts/benchmark_agent_metadata_handoff.py', 'scripts/recover_agent_metadata_port.py',
               'tests/test_agent_metadata_port_recovery.py', 'tests/test_agent_port_probe.py']
    for rel in changed:
        shutil.copy2(repository/rel, runtime/rel)
    port_script = runtime/'scripts/run_agent_handoff_queue.py'
    port_script.write_text(replace_port_check(port_script.read_text(), (repository/'scripts/run_agent_handoff_queue.py').read_text()))
    changed.append('scripts/run_agent_handoff_queue.py')
    for rel in ('fixture.json', 'fixture-source-hashes.json', 'optimization/after/profile.json'):
        shutil.copy2(source/rel, output/rel)
    q.save(after/'status.json', dict(status='prepared'))
    # Freeze completed responses, diagnostics and the failed attempt, as well as
    # the original immutable source/profile list. Nothing is removed or rewritten.
    preserved = dict(plan['files'])
    for p in source.rglob('*'):
        if p.is_file() and 'runtime' not in p.relative_to(source).parts:
            preserved[str(p.relative_to(source))] = q.sha(p)
    model_hashes = {str(p.relative_to(runtime)):q.sha(p) for p in runtime.rglob('*') if p.is_file()
                    and str(p.relative_to(runtime)) not in changed}
    q.save(output/'port-recovery-audit.json', dict(source=str(source), at=time.time(),
        reused_completed_arms=['before'], execution_arms=['after'], preserved_hashes=preserved,
        model_runtime_hashes=model_hashes, controller_only_changes=changed,
        reason='Port preflight rejected a closed listener. TIME_WAIT false positive reproduced; '
               'SO_REUSEADDR and listen preflight accept reusable closed ports and reject live listeners.',
        source_historical_socket_state_observed=False,
        original_after_public_requests=0, original_agent_answers_reexecuted=0))
    plan['execution_arms'] = ['after']
    plan['port_recovery'] = dict(source=str(source), proof='port-recovery-audit.json',
                                sha256=q.sha(output/'port-recovery-audit.json'))
    plan['files'].update({str(p.relative_to(output)):q.sha(p) for p in runtime.rglob('*') if p.is_file()})
    plan['files']['port-recovery-audit.json'] = q.sha(output/'port-recovery-audit.json')
    q.save(output/'manifest.json', plan)
    q.save(output/'status.json', dict(status='prepared', at=time.time()))
    benchmark.validate(output)
    return dict(root=str(output), manifest_sha256=q.sha(output/'manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.output, Path(__file__).resolve().parents[1]))
