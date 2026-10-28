"""Qualify an explicit FPS implementation change, then inherit completed MAB answers.

Old runtime, attempts, responses and evidence are immutable. The only permitted
engine change is query_key_fps.py, bound to an exact-score performance report.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from benchmark_niah import save, sha
from graph_v2_protocol import require

REPO = Path(__file__).resolve().parents[1]
SCORER = 'src/vllm_kvmem/query_key_fps.py'


def read(path):
    return json.loads(Path(path).read_text())


def engine(files):
    return {k:v for k,v in files.items() if k.startswith(('src/','csrc/','third_party/'))
        or k in ('setup.py','pyproject.toml')}


def validate_engine_change(old, new, component):
    changes = {k:dict(before=old.get(k),after=new.get(k))
        for k in sorted(set(old)|set(new)) if old.get(k) != new.get(k)}
    require(set(changes) == {SCORER},'Unexpected inference engine changes')
    require(component['exact_scores'] and component['source_sha256'] == new[SCORER] and
        component['reference_sha256'] == old[SCORER], 'Scorer is not bound to exact-score qualification')
    require(component['results'] and all(r['speedup'] > 1 for r in component['results']),
        'Scoring optimization has no measured component benefit')
    for result in component['results']:
        for m in result['measurements']:
            require(m['exact_scores'],'Component score equality failed')
            if m['arm'] == 'new':
                s = m['stats']
                require(s['scoring_scratch_bound_bytes'] <= s['scoring_scratch_limit_bytes'] and
                    m['peak_scratch_bytes'] <= s['scoring_scratch_limit_bytes'],
                    'Optimized scorer exceeded scratch capacity')
    return changes


def inherit_cases(previous, output, fixture):
    """Copy successful cases exactly; preserve interrupted attempts at source."""
    require(not output.exists(),'Never overwrite recovery output')
    source = previous/'run/answers'; rows = {r['id']:r for r in fixture['rows']}
    output.mkdir(parents=True)
    shutil.copy2(source/'manifest.json',output/'manifest.json')
    originals = {'run/answers/manifest.json':sha(source/'manifest.json')}
    recovered, interrupted = [], []
    for directory in sorted((source/'cases').iterdir()):
        require(directory.name in rows and directory.is_dir(),'Unexpected source case')
        paths = [p for p in directory.iterdir() if p.is_file()]
        originals.update({str(p.relative_to(previous)):sha(p) for p in paths})
        result = read(directory/'result.json') if (directory/'result.json').exists() else None
        if result is None or result.get('status') != 'completed':
            require(not (directory/'response.json').exists(),
                'Complete saved response needs explicit audit recovery, not regeneration')
            interrupted.append(directory.name)
            continue
        row = rows[directory.name]
        require(result['id'] == row['id'] and result['original_input_intact'] and
            result['token_sha256'] == row['token_sha256'] and result['prompt_tokens'] == row['input_tokens'],
            'Inherited answer input identity differs')
        require(all((directory/n).is_file() for n in ('response.json','request.json','terminal.json','events.json')),
            'Inherited answer lacks original response or execution evidence')
        # Do not examine gold, answer correctness or scores when choosing reuse.
        shutil.copytree(directory,output/'cases'/directory.name)
        recovered.append(directory.name)
    if (source/'releases').exists():
        shutil.copytree(source/'releases',output/'releases')
        originals.update({str(p.relative_to(previous)):sha(p) for p in (source/'releases').glob('*.json')})
    save(output/'summary.json',dict(expected=fixture['cases'],completed=len(recovered),errors=0,
        inherited_answers=len(recovered),regenerated_answers=0))
    recovered_files = {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    from run_ruler_hybrid_campaign import verify_files
    verify_files(previous,originals)
    return dict(source=str(previous),kind='fps_layer_cache',source_files=originals,
        recovered_ids=recovered,interrupted_ids=interrupted,repaired_audit_ids=[],
        original_successes_unchanged=len(recovered),regenerated_answers=0,
        engine_sources_identical=False,recovered_files=recovered_files)


def verify_optimization(root, fixture):
    from run_ruler_hybrid_campaign import verify_files
    proof = read(root/'optimization-proof.json'); previous = Path(proof['source'])
    require(proof['source'] == fixture['optimization_source'],'Optimization source changed')
    verify_files(previous,proof['source_files'])
    old = engine(read(previous/'runtime-proof.json')['files'])
    new = engine(read(root/'runtime-proof.json')['files'])
    require(validate_engine_change(old,new,read(root/'component.json')) == proof['engine_changes'],
        'Qualified engine change differs')
    old_profile,profile = read(previous/'profile.json'),read(root/'profile.json')
    expected = deepcopy(old_profile)
    if fixture['qualification']:
        expected['port'] = 18872
        expected['kvmem']['graph_trace'] = True
    require(profile == expected,'Original model, sampling or recall profile changed')
    old_fixture = read(previous/'fixture.json')
    require(fixture['rows'] == old_fixture['rows'] and
        fixture['source_cohort_sha256'] == old_fixture['source_cohort_sha256'],
        'Original memory cohort changed')
    if proof.get('qualification'):
        q = Path(proof['qualification'])
        require(sha(q/'execution-audit.json') == proof['qualification_audit_sha256'] and
            read(q/'execution-audit.json')['status'] == 'passed' and
            read(q/'runtime-proof.json') == read(root/'runtime-proof.json'),
            'Completed model qualification does not match recovery runtime')


def prepare(root, previous, component, *, qualification=None):
    import run_hybrid_memory_campaign as campaign
    require(not root.exists(),'Use a fresh recovery or qualification root')
    old_fixture, old_profile = campaign.verify(previous)
    require(old_fixture['suite'] == 'memoryagentbench' and not old_fixture['qualification'],
        'Expected original formal MAB campaign')
    cleanup = read(previous/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],
        'Previous model resources remain active')
    require((previous/'optimization-stop.json').exists(),'Missing explicit optimization interruption record')
    report = read(component)
    root.mkdir(parents=True)
    if qualification is None:
        shutil.copytree(previous/'runtime',root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in (SCORER,'scripts/run_hybrid_memory_campaign.py','scripts/recover_hybrid_mab.py'):
            shutil.copy2(REPO/name,root/'runtime'/name)
    else:
        from run_hybrid_suite_queue import completed
        completed(qualification)
        shutil.copytree(qualification/'runtime',root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    runtime_files = {str(p.relative_to(root/'runtime')):sha(p)
        for p in (root/'runtime').rglob('*') if p.is_file()}
    save(root/'runtime-proof.json',dict(files=runtime_files))
    shutil.copy2(component,root/'component.json')
    changes = validate_engine_change(engine(read(previous/'runtime-proof.json')['files']),engine(runtime_files),report)
    fixture,profile = deepcopy(old_fixture),deepcopy(old_profile)
    fixture['qualification'] = qualification is None
    fixture['optimization_source'] = str(previous)
    fixture['profile_sha256'] = None
    if qualification is None:
        groups = {}
        for row in fixture['rows']:groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
        group = max((items for items in groups.values() if len(items)>=2),
            key=lambda items:max(r['input_tokens'] for r in items))
        ids = [r['id'] for r in group[:2]]
        longest = max(fixture['rows'],key=lambda r:r['input_tokens'])['id']
        if longest not in ids:ids.append(longest)
        fixture.update(case_ids=ids,cases=len(ids),audit_scope='Full-vector native replay; separate qualification, never replaces formal answers')
        profile['port'] = 18872;profile['kvmem']['graph_trace'] = True
    else:
        fixture['recovery_source'] = str(previous)
    save(root/'profile.json',profile); fixture['profile_sha256'] = sha(root/'profile.json')
    source_files = {n:sha(previous/n) for n in
        ('fixture.json','profile.json','runtime-proof.json','prepared-proof.json','optimization-stop.json','run/cleanup.json')}
    opt = dict(source=str(previous),source_files=source_files,engine_changes=changes,
        qualification=str(qualification) if qualification else None,
        qualification_audit_sha256=sha(qualification/'execution-audit.json') if qualification else None,
        numerical_scope='Same tile GEMM, pooling, masks, softmax, accumulation and picker; exact fixed-operand scores',
        timing_scope='Old and new attempts reported separately; restarted prefixes incur cold preparation again')
    save(root/'optimization-proof.json',opt)
    files = ['profile.json','fixture.json','runtime-proof.json','component.json','optimization-proof.json']
    if qualification is not None:
        recovery = inherit_cases(previous,root/'recovered-answers',fixture)
        save(root/'recovery-proof.json',recovery); files.append('recovery-proof.json')
        fixture['recovery_cases'] = recovery['recovered_ids']
    save(root/'fixture.json',fixture)
    save(root/'prepared-proof.json',{n:sha(root/n) for n in files})
    save(root/'status.json',dict(status='prepared',cases=fixture['cases'],suite='memoryagentbench'))
    campaign.verify(root)
    return dict(root=str(root),cases=fixture['cases'],inherited=len(fixture.get('recovery_cases',[])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--component',type=Path,required=True)
    p.add_argument('--qualification',type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.previous.resolve(),a.component.resolve(),
        qualification=a.qualification.resolve() if a.qualification else None)))
