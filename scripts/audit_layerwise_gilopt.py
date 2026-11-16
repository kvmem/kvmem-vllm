"""Read-only runtime audit and first-divergence analysis for the G4K experiment.

Write evidence outside the observed campaign. An after-the-fact hash audit does
not replace launch-time verification or establish model/output equivalence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from benchmark_client_io import result_events
from run_layerwise_pressure_efficiency import signature


CHANGED = ('src/vllm_kvmem/layerwise_pressure.py', 'src/vllm_kvmem/window_assembly.py')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def tree(root):
    return {str(f.relative_to(root)): sha(f) for f in sorted(root.rglob('*'))
            if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'}


def verify_files(root, files):
    for name, digest in files.items():
        if sha(root/name) != digest:
            raise ValueError('Frozen file changed: '+str(root/name))


def apply_patch(source, patch, destination):
    """Apply only the two approved changes, never copy workspace inference files."""
    paths = [line[6:] for line in patch.read_text().splitlines() if line.startswith('+++ b/')]
    if sorted(paths) != sorted(CHANGED):
        raise ValueError('Patch must change exactly the two approved inference modules')
    for name in CHANGED:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    # A frozen runtime lives inside the working repository. Otherwise Git
    # discovers the outer .git and silently skips paths outside this subdir.
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    environment['GIT_CEILING_DIRECTORIES'] = str(destination.resolve().parent)
    subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=destination, check=True,
                   env=environment, capture_output=True, text=True)
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=destination, check=True,
                   env=environment, capture_output=True, text=True)
    if any(sha(source/name) == sha(destination/name) for name in CHANGED):
        raise ValueError('Patch did not change both approved inference modules')


def audit(baseline, optimized, patch):
    verify_files(baseline, read(baseline/'prepared-proof.json'))
    old = tree(baseline/'runtime-v1')
    if old != read(baseline/'runtime-proof.json'):
        raise ValueError('Baseline runtime tree differs from its frozen manifest')
    new = tree(optimized/'runtime-v1')
    # The original shell patch left this non-importable backup in the running
    # tree. Preserve and hash it rather than silently ignoring extra files.
    backups = {'src/vllm_kvmem/window_assembly.py.orig': old['src/vllm_kvmem/window_assembly.py']}
    extras = {name: new[name] for name in new.keys() - old.keys()}
    if old.keys() - new.keys() or (extras and extras != backups):
        raise ValueError('Runtime file set changed')
    changed = sorted(k for k in old if old[k] != new[k])
    if changed != sorted(CHANGED):
        raise ValueError('Unexpected inference/runtime changes: '+repr(changed))
    with tempfile.TemporaryDirectory(prefix='gilopt-', dir='/tmp/opencode') as tmp:
        expected = Path(tmp)
        apply_patch(baseline/'runtime-v1', patch, expected)
        if any(sha(expected/n) != new[n] for n in CHANGED):
            raise ValueError('Candidate is not the baseline plus exactly the approved patch')
    controls = ('fixture.json', 'profile.json')
    if any(sha(baseline/n) != sha(optimized/n) for n in controls):
        raise ValueError('Fixture/profile bytes differ')
    files = {str(root/n): sha(root/n) for root in (baseline, optimized) for n in controls}
    files.update({str(baseline/n): sha(baseline/n)
                  for n in ('runtime-proof.json', 'prepared-proof.json')})
    files[str(patch)] = sha(patch)
    launch_parity = None
    if (optimized/'run/server/launch.json').exists():
        launches = [read(root/'run/server/launch.json') for root in (baseline, optimized)]
        for launch in launches:
            launch['environment'].pop('KVMEM_RUNNER_PROBE_DIR', None)
        if launches[0] != launches[1]:
            raise ValueError('Effective launch differs beyond capture output directory')
        files[str(baseline/'run/server/launch.json')] = sha(baseline/'run/server/launch.json')
        launch_parity = True
    for name in ('run_g4k.py', 'toolchain-alias-audit.json', 'run/server/launch.json', 'process.json'):
        if (optimized/name).exists():
            files[str(optimized/name)] = sha(optimized/name)
    for binding in read(baseline/'fixture.json')['bindings']:
        if sha(Path(binding['request'])) != binding['sha256']:
            raise ValueError('Original request binding changed')
        files[binding['request']] = binding['sha256']
    return dict(baseline=str(baseline), optimized=str(optimized), at=time.time(),
        passed=True, changed_runtime_files=changed, runtime_file_count=len(old),
        optimized_file_count=len(new), preserved_nonimportable_backups=extras,
        baseline_runtime=old, optimized_runtime=new, files=files,
        patch_reapplied_to_frozen_baseline=True, fixture_profile_byte_equal=True,
        effective_launch_equal_except_capture_directory=launch_parity,
        launch_time_verification=False, output_equivalence=False,
        scope='Read-only post-launch audit, not a retroactive launch proof or correctness acceptance.')


def verify_audit(report):
    verify_files(Path('/'), report['files'])
    for key in ('baseline', 'optimized'):
        if tree(Path(report[key])/'runtime-v1') != report[key+'_runtime']:
            raise ValueError('Observed runtime tree changed: '+key)
    return report


def evidence(result, refresh, case_dir=None):
    return dict(events=result_events(result, case_dir), layers=result['layerwise']['layers'], refresh=refresh)


def first_difference(before, after, path=''):
    """Report the earliest differing field, including missing list elements."""
    if type(before) is not type(after):
        return dict(path=path, before=before, after=after)
    if isinstance(before, dict):
        for key in dict.fromkeys((*before, *after)):
            child = path+'.'+str(key) if path else str(key)
            if key not in before or key not in after:
                return dict(path=child, before=before.get(key), after=after.get(key), missing=True)
            difference = first_difference(before[key], after[key], child)
            if difference:
                return difference
    elif isinstance(before, (list, tuple)):
        for i in range(max(len(before), len(after))):
            child = f'{path}[{i}]'
            if i >= len(before) or i >= len(after):
                return dict(path=child, before=before[i] if i < len(before) else None,
                            after=after[i] if i < len(after) else None, missing=True)
            difference = first_difference(before[i], after[i], child)
            if difference:
                return difference
    elif before != after:
        return dict(path=path, before=before, after=after)
    return None


def canonical_signature(result, refresh, case_dir=None):
    # Reuse the strict accepted pressure/Query signature, adding rolling stage
    # identity so prefill and decode events cannot accidentally compare equal.
    pressures, query = signature(evidence(result, refresh, case_dir))
    events = [e for e in result_events(result, case_dir) if e['event'] == 'pressure_selected']
    for pressure, event in zip(pressures, events):
        pressure.update(event_type=event['event_type'], query_source=event['query_source'],
                        selected_tokens=event['selected_tokens'])
    return dict(pressure=pressures, query=query)


def compare_cases(before_dir, after_dir):
    roots = (before_dir, after_dir)
    results = [read(p/'result.json') for p in roots]
    requests = [read(p/'request.json') for p in roots]
    # Session/cache UUIDs are ownership identities, not sampling controls.
    for request in requests:
        for key in ('session_id', 'cache_salt'):
            request.pop(key, None)
    controls = first_difference(requests[0], requests[1])
    if controls:
        raise ValueError('Input/sampling mismatch: '+repr(controls))
    signs = [canonical_signature(r, read(p/'events.json'), p) for r, p in zip(results, roots)]
    pressure = first_difference(signs[0]['pressure'], signs[1]['pressure'], 'pressure')
    query = first_difference(signs[0]['query'], signs[1]['query'], 'query')
    choices = [read(p/'response.json')['choices'][0] for p in roots]
    if any(not isinstance(c.get('token_ids'), list) or not c['token_ids'] for c in choices):
        raise ValueError('Missing complete output token IDs')
    output = first_difference(choices[0]['token_ids'], choices[1]['token_ids'], 'token_ids')
    context = None
    for i in range(max(len(signs[0]['pressure']), len(signs[1]['pressure']))):
        pair = [s['pressure'][i] if i < len(s['pressure']) else None for s in signs]
        if pair[0] != pair[1]:
            context = dict(event=i, accepted_history=[p and p['accepted_history'] for p in pair],
                event_type=[p and p['event_type'] for p in pair], layers=[])
            if all(p is not None for p in pair):
                for a, b in zip(pair[0]['layers'], pair[1]['layers']):
                    if a != b:
                        context['layers'].append(dict(layer=a['layer'],
                            same_candidates=a['candidate_starts'] == b['candidate_starts'],
                            same_score_end=a['accepted_score_end'] == b['accepted_score_end'],
                            selected_symmetric_difference=sorted(set(a['selected_starts']) ^ set(b['selected_starts']))))
            break
    same_text = choices[0]['text'] == choices[1]['text']
    return dict(id=results[0]['id'], same_pressure=pressure is None, same_query=query is None,
        same_output_tokens=output is None, same_text=same_text,
        same_answer=results[0]['answer'] == results[1]['answer'],
        passed=not (pressure or query or output) and same_text,
        first_pressure_difference=pressure, first_query_difference=query,
        first_output_difference=output, first_pressure_context=context,
        pressure_events=[len(s['pressure']) for s in signs],
        request_seconds=[r['seconds'] for r in results],
        source_files={str(p/n): sha(p/n) for p in roots for n in
                      ('request.json', 'response.json', 'result.json', 'events.json')})


def compare(before, after, case_ids=None):
    a, b = (root/'run/answers/cases' for root in (before, after))
    if case_ids is None:
        case_ids = sorted(f.parent.name for f in b.glob('*/result.json') if (a/f.parent.name/'result.json').exists())
    checks = [compare_cases(a/cid, b/cid) for cid in case_ids]
    complete = []
    for root in (before, after):
        path = root/'inference-performance.json'
        complete.append(read(path) if path.exists() and read(root/'status.json')['status'] == 'completed' else None)
    return dict(baseline=str(before), optimized=str(after), cases=len(checks), checks=checks,
        passed=bool(checks) and all(c['passed'] for c in checks),
        equal_counts={key: sum(c[key] for c in checks) for key in
                      ('same_pressure', 'same_query', 'same_output_tokens', 'same_text', 'same_answer')},
        complete_inference_performance=complete, at=time.time(),
        scope='Exact accepted pressure/Query scope/blocks and full output IDs/text. '
              'Partial matched request latency is not campaign throughput; no accuracy grading.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'verify', 'compare'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--optimized', type=Path)
    parser.add_argument('--patch', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        verify_audit(read(args.report))
        print('Runtime audit verified (not model equivalence).')
        return
    if args.action == 'audit':
        report = audit(args.baseline.resolve(), args.optimized.resolve(), args.patch.resolve())
    else:
        report = compare(args.baseline.resolve(), args.optimized.resolve())
    save_new(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k in
                      ('passed', 'cases', 'equal_counts', 'runtime_file_count', 'changed_runtime_files')}))


if __name__ == '__main__':
    main()
