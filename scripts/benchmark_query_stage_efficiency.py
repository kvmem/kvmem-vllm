"""Finite serial full-input Query profiling and controlled candidate measurements.

Preserves the latest eight-way runtime, model/budgets/recall/sampling and original
MAB P prefixes. Only the measurement gate is removed. New opt-in instrumentation
or execution code is applied as a recorded patch, identically in every arm.
This driver never resumes the formal experiment or grades answers.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time


REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
REFERENCE = REPO/'artifacts/layerwise-dynamic8-prefill-20261106-r1'
FORMAL = REPO/'artifacts/mab-pressure-multiowner-20261105-r10/formal'
DEFAULT_IDS = ('005_Accurate_Retrieval_r005_eventqa_full_q0000',
               '005_Accurate_Retrieval_r005_eventqa_full_q0001',
               '025_Long_Range_Understanding_r078_infbench_sum_eng_shots2_q0000')
OVERLAYS = ('src/vllm_kvmem/layerwise_query.py', 'src/vllm_kvmem/transfer.py',
            'src/vllm_kvmem/query_stage_profile.py')
PREFILL_STATE_SOURCES = frozenset(('csrc/buffered_prefill_state.cpp',
                                 'csrc/prefill_attention_commit.hpp'))
GATE = ("\n# Identical finite historical-prefill measurement in both allocation modes.\n"
        "os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '8'\n"
        "os.environ['KVMEM_DIAGNOSTIC_PREFILL_GPU_COMPLETION'] = '1'\n"
        "from .prefill_phase_gate import install as _install_dynamic8_measurement\n"
        "_install_dynamic8_measurement()\n")
SERVER_SHIM = '''"""Frozen experiment-only explicit launch environment."""
import hashlib
import json
import os
from pathlib import Path
try:
    from vllm_kvmem import serve           # a runtime from before the layer directories
except ImportError:
    from vllm_kvmem.serving import serve

def main():
    if os.environ.get('QUERY_STAGE_NATIVE_BINARY_JSON'):
        from vllm_kvmem import _native
        expected = json.loads(os.environ['QUERY_STAGE_NATIVE_BINARY_JSON'])
        actual = Path(_native.__file__).resolve()
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if actual != Path(expected['path']).resolve() or digest != expected['sha256']:
            raise RuntimeError('Frozen native extension was shadowed or changed')
        with Path(os.environ['QUERY_STAGE_NATIVE_AUDIT_OUTPUT']).open('x') as stream:
            json.dump(dict(path=str(actual), sha256=digest, passed=True,
                scope='Actual server-process import before model initialization'), stream)
    if os.environ.get('QUERY_STAGE_PREFILL_STATE_BINARY_JSON'):
        from vllm_kvmem import _prefill_state_native
        expected = json.loads(os.environ['QUERY_STAGE_PREFILL_STATE_BINARY_JSON'])
        actual = Path(_prefill_state_native.__file__).resolve()
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if actual != Path(expected['path']).resolve() or digest != expected['sha256']:
            raise RuntimeError('Frozen prefill state extension was shadowed or changed')
        with Path(os.environ['QUERY_STAGE_PREFILL_STATE_AUDIT_OUTPUT']).open('x') as stream:
            json.dump(dict(path=str(actual), sha256=digest, passed=True,
                scope='Actual server-process import before model initialization'), stream)
    original = serve.build_launch
    def build(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan['environment'].update(json.loads(os.environ['QUERY_STAGE_EXECUTION_JSON']))
        return plan
    serve.build_launch = build
    serve.main()

if __name__ == '__main__':
    main()
'''


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def cache(root):
    return {str(p.relative_to(root)): hashlib.sha256(json.dumps(read(p), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest() for p in sorted(root.rglob('*.json'))}


def save(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')


def ungated(text):
    if GATE in text:
        if text.count(GATE) != 1 or not text.endswith(GATE):
            raise ValueError('Unexpected measurement gate placement')
        text = text[:-len(GATE)]
    if 'KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS' in text or 'KVMEM_DIAGNOSTIC_PREFILL_COHORT_LENGTHS' in text:
        raise ValueError('Serial Query diagnostic cannot retain a finite-owner gate')
    return text


def validate_controls(fixture, profile):
    kv = profile['kvmem']
    if [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] != [36864,16384,32,41,0]:
        raise ValueError('Original Active/Gen or recall geometry changed')
    if [profile[k] for k in ('execution_mode','draft_tokens','max_num_seqs','max_num_batched_tokens')] != ['compiled_graph',4,8,16424]:
        raise ValueError('Original compiled MTP4 and scheduler controls changed')
    if kv.get('gpu_memory_admission') != 'static' or kv.get('request_swap_layerwise'):
        raise ValueError('Use the completed static allocation control')
    flags = fixture['execution_flags']
    if not flags.get('layerwise_prefill_pressure') or not flags.get('layerwise_paired_reads'):
        raise ValueError('Require the qualified layerwise pressure and Query paths')
    if flags.get('layerwise_query_no_probe') or flags.get('layerwise_query_full_diagnostics'):
        raise ValueError('Do not change global MTP or enable old Q diagnostics')


def workload(fixture, case_ids):
    ids = tuple(case_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Require distinct original sample IDs')
    by_id = {r['id']:r for r in fixture['rows']}
    bindings = {r['id']:r for r in fixture['bindings']}
    rows = [deepcopy(by_id[cid]) for cid in ids]
    groups = defaultdict(list)
    all_groups = defaultdict(list)
    for r in fixture['rows']: all_groups[r['id'].rsplit('_q',1)[0]].append(r)
    for r in rows: groups[r['id'].rsplit('_q',1)[0]].append(r)
    for name, selected in groups.items():
        if min(r['query_begin'] for r in selected)//1024 != min(r['query_begin'] for r in all_groups[name])//1024:
            raise ValueError('Subset would change the original frozen P prefix')
    return rows, [deepcopy(bindings[cid]) for cid in ids]


def variant_order(variants, order):
    forbidden = {'KVMEM_LAYERWISE_QUERY_MODE', 'KVMEM_LAYERWISE_QUERY_NO_PROBE',
                 'KVMEM_LAYERWISE_QUERY_MTP_SELECTION', 'KVMEM_LAYERWISE_QUERY_DIAGNOSTICS'}
    if not order or any(mode not in variants for mode in order):
        raise ValueError('Every planned arm must have explicit execution flags')
    for flags in variants.values():
        if any(not isinstance(k,str) or not isinstance(v,str) or k in forbidden or
               not k.startswith(('KVMEM_QUERY_', 'KVMEM_LAYERWISE_QUERY_')) for k,v in flags.items()):
            raise ValueError('Only explicit Query execution switches may differ')
    return [dict(name=f'{i+1:02d}-{mode}', variant=mode, environment=variants[mode])
            for i,mode in enumerate(order)]


JIT_STATS_SOURCES = frozenset(('src/vllm_kvmem/native_stats.cuh',
                               'src/vllm_kvmem/native_prefill_stats.cuh'))


def patch_runtime(runtime, overlay_source, overlays, patch_path):
    """Produce and apply a reviewable patch; never overlay entire source trees."""
    shared_header = 'csrc/prefill_attention_commit.hpp'
    if (shared_header in overlays and
            (runtime/shared_header).read_bytes() != (overlay_source/shared_header).read_bytes()):
        raise ValueError('Shared prefill arithmetic header must remain unchanged in this candidate')
    text = (runtime/'src/vllm_kvmem/layerwise_query.py').read_text()
    (runtime/'src/vllm_kvmem/layerwise_query.py').write_text(ungated(text))
    before = tree(runtime); patch = []
    cuda_pair = {'csrc/qw3_mean_k.cu', 'src/vllm_kvmem/cuda_sources/qw3_mean_k.cu'}
    if set(overlays) & cuda_pair and not cuda_pair <= set(overlays):
        raise ValueError('Declare both packaged and repository CUDA scorer sources')
    for name in overlays:
        relative = Path(name)
        python_module=(relative.parts[:2] == ('src','vllm_kvmem') and relative.suffix=='.py')
        # FlashInfer statistics headers are JIT-compiled from the runtime tree.
        declared_native_source=(name=='csrc/bindings.cpp' or name in cuda_pair or
                                name in PREFILL_STATE_SOURCES or name in JIT_STATS_SOURCES)
        if relative.is_absolute() or '..' in relative.parts or not (python_module or declared_native_source):
            raise ValueError('Overlay must name an explicit inference module')
        target, origin = runtime/name, overlay_source/name
        original = target.read_text() if target.exists() else ''
        updated = origin.read_text()
        if name.endswith('/layerwise_query.py'): updated = ungated(updated)
        patch.extend(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile='a/'+name if target.exists() else '/dev/null', tofile='b/'+name))
    if cuda_pair <= set(overlays):
        first, second = (overlay_source/name for name in sorted(cuda_pair))
        if first.read_bytes() != second.read_bytes():
            raise ValueError('Packaged CUDA scorer differs from repository source')
    patch_path.write_text(''.join(patch))
    if patch:
        env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        env['GIT_CEILING_DIRECTORIES'] = str(runtime.resolve().parent)
        for options in (['--check'], []):
            subprocess.run(['git','apply',*options,str(patch_path.resolve())], cwd=runtime,
                           env=env, check=True, capture_output=True, text=True)
    after = tree(runtime)
    changed = sorted(k for k in set(before)|set(after) if before.get(k) != after.get(k))
    if not set(changed) <= set(overlays): raise ValueError('Patch changed an undeclared source file')
    return changed


def freeze_native(runtime, native_binary, changed):
    """Copy an explicitly supplied extension, never build or mutate an install."""
    if native_binary is None:
        if 'csrc/bindings.cpp' in changed:
            raise ValueError('Changed native source requires its explicitly supplied binary')
        return None
    binary=Path(native_binary).resolve()
    expected_name='_native'+sysconfig.get_config_var('EXT_SUFFIX')
    originals=list((runtime/'src/vllm_kvmem').glob('_native*.so'))
    if len(originals)!=1 or originals[0].name!=expected_name or binary.name!=expected_name:
        raise ValueError('Native replacement must match the original module and Python ABI')
    if not binary.is_file() or binary.read_bytes()[:4]!=b'\x7fELF':
        raise ValueError('Native replacement must be an existing ELF extension')
    if 'csrc/bindings.cpp' not in changed:
        raise ValueError('Native replacement requires a declared C++ source patch')
    target=originals[0];old_sha256=sha(target);new_sha256=sha(binary)
    if old_sha256==new_sha256:
        raise ValueError('Supplied native binary is unchanged despite the source patch')
    shutil.copy2(binary,target)
    return dict(relative_path=str(target.relative_to(runtime)),original_sha256=old_sha256,
        source_binary=str(binary),sha256=new_sha256,
        declared_source_sha256={'csrc/bindings.cpp':sha(runtime/'csrc/bindings.cpp')},
        scope='Explicit prebuilt extension copied identically into all new arms; no build or installed-library mutation. '
              'Source hashes identify the declared patch, not a claim of reproducible build provenance.')


def freeze_prefill_state(runtime, binary, changed):
    """Freeze the separate immutable-history extension without touching _native."""
    if 'csrc/prefill_attention_commit.hpp' in changed:
        raise ValueError('Shared prefill arithmetic header must remain unchanged in this candidate')
    changed_sources = set(changed) & PREFILL_STATE_SOURCES
    if binary is None:
        if changed_sources:
            raise ValueError('Changed prefill state source requires its explicitly supplied binary')
        return None
    if not changed_sources:
        raise ValueError('Prefill state replacement requires a declared C++ source patch')
    binary = Path(binary).resolve()
    expected_name = '_prefill_state_native' + sysconfig.get_config_var('EXT_SUFFIX')
    originals = list((runtime/'src/vllm_kvmem').glob('_prefill_state_native*.so'))
    if len(originals) != 1 or originals[0].name != expected_name or binary.name != expected_name:
        raise ValueError('Prefill state replacement must match the original module and Python ABI')
    if not binary.is_file() or binary.read_bytes()[:4] != b'\x7fELF':
        raise ValueError('Prefill state replacement must be an existing ELF extension')
    if not all((runtime/name).is_file() for name in PREFILL_STATE_SOURCES):
        raise ValueError('Prefill state replacement requires its complete source and header')
    target = originals[0]
    original_hash, new_hash = sha(target), sha(binary)
    if original_hash == new_hash:
        raise ValueError('Supplied prefill state binary is unchanged despite the source patch')
    shutil.copy2(binary, target)
    return dict(relative_path=str(target.relative_to(runtime)), original_sha256=original_hash,
        source_binary=str(binary), sha256=new_hash,
        declared_source_sha256={name:sha(runtime/name) for name in sorted(PREFILL_STATE_SOURCES)},
        scope='Explicit prebuilt prefill state extension copied identically into all new arms; '
              'no build or installed-library mutation. Source hashes identify the declared patch, '
              'not a claim of reproducible build provenance.')


def prepare(root, *, source_arm=REFERENCE/'static1', fixture_path=FORMAL/'fixture.json',
            overlay_source=REPO, overlays=OVERLAYS, case_ids=DEFAULT_IDS,
            autotune_source=REFERENCE/'autotune-seed', variants=None, order=('baseline','probe'), port=19450,
            native_binary=None, prefill_state_binary=None):
    root, source_arm = Path(root).resolve(), Path(source_arm).resolve()
    fixture_path, overlay_source, autotune_source = map(lambda p:Path(p).resolve(),
        (fixture_path,overlay_source,autotune_source))
    variants = variants or {'baseline':{'KVMEM_QUERY_STAGE_PROFILE':'0'}, 'probe':{'KVMEM_QUERY_STAGE_PROFILE':'1'}}
    trials = variant_order(variants, order)
    if root.exists(): raise ValueError('Preserve prior first answers; choose a new root')
    if tree(source_arm/'runtime-v1') != read(source_arm/'runtime-proof.json'):
        raise ValueError('Source runtime changed')
    fixture = read(fixture_path); profile = read(source_arm/'profile.json')
    # Inherit the latest executed optimization flags and CPU admission override.
    control_fixture = read(source_arm/'fixture.json')
    fixture['execution_flags'] = deepcopy(control_fixture['execution_flags'])
    fixture['execution_kvmem_overrides'] = deepcopy(control_fixture.get('execution_kvmem_overrides',{}))
    validate_controls(fixture,profile)
    rows,bindings = workload(fixture,case_ids)
    if not cache(autotune_source): raise ValueError('Missing common autotuning seed')
    for row,binding in zip(rows,bindings):
        if sha(row['source']) != row['source_cache_sha256'] or sha(binding['request']) != binding['sha256']:
            raise ValueError('Original input/sampling changed')
    root.mkdir(parents=True)
    for row,binding in zip(rows,bindings):
        for record,field,folder,suffix in ((row,'source','inputs','.json.gz'),(binding,'request','bindings','.json')):
            destination = root/folder/(row['id']+suffix); destination.parent.mkdir(exist_ok=True)
            shutil.copy2(record[field],destination); record[field] = str(destination)
    fixture.update(rows=rows,bindings=bindings,cases=len(rows),case_ids=list(case_ids),
        qualification=False,restore_digest_qualification=False,diagnostic_only=True,
        performance_observation=True,external_grading=False)
    profile['port'] = port
    runtime = root/'source-runtime'
    shutil.copytree(source_arm/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    changed = patch_runtime(runtime,overlay_source,overlays,root/'implementation.patch')
    native=freeze_native(runtime,native_binary,changed)
    if native is not None:
        save(root/'native-extension.json',native)
    prefill_state=freeze_prefill_state(runtime,prefill_state_binary,changed)
    if prefill_state is not None:
        save(root/'prefill-state-extension.json',prefill_state)
    (runtime/'scripts/_query_stage_server.py').write_text(SERVER_SHIM)
    proof = tree(runtime)
    shutil.copytree(autotune_source,root/'autotune-seed')
    save(root/'autotune-seed-proof.json',tree(root/'autotune-seed'))
    files = {str(p.relative_to(root)):sha(p) for folder in ('inputs','bindings') for p in (root/folder).iterdir()}
    files['implementation.patch'] = sha(root/'implementation.patch')
    files['autotune-seed-proof.json'] = sha(root/'autotune-seed-proof.json')
    if native is not None: files['native-extension.json']=sha(root/'native-extension.json')
    if prefill_state is not None: files['prefill-state-extension.json']=sha(root/'prefill-state-extension.json')
    for trial in trials:
        arm = root/trial['name']; shutil.copytree(runtime,arm/'runtime-v1')
        shutil.copytree(root/'autotune-seed',arm/'autotune')
        for name,value in [('fixture.json',fixture),('profile.json',profile),('runtime-proof.json',proof)]:
            save(arm/name,value);files[str((arm/name).relative_to(root))] = sha(arm/name)
        save(arm/'status.json',dict(status='prepared'))
    shutil.copy2(__file__,root/'launch.py');files['launch.py'] = sha(root/'launch.py')
    save(root/'plan.json',dict(trials=trials,files=files,workers=1,case_ids=list(case_ids),
        source_runtime_sha256=proof,changed_runtime_files=changed,
        native_extension=native,
        prefill_state_extension=prefill_state,
        source_controls={str(source_arm/'runtime-proof.json'):sha(source_arm/'runtime-proof.json'),
                         str(source_arm/'profile.json'):sha(source_arm/'profile.json'),
                         str(fixture_path):sha(fixture_path)},
        automatic_retry=False,formal_resume=False,external_grading=False,scheduled_monitor=False,
        scope='Full original cold/warm inputs; serial clients with original eight-slot server. '
              'Probe timers are diagnostic only. Performance comparisons require probe off.'))
    save(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save(root/'status.json',dict(status='prepared'))


def verify(root):
    if sha(root/'plan.json') != read(root/'prepared-proof.json')['plan_sha256']:
        raise ValueError('Frozen plan changed')
    plan = read(root/'plan.json')
    for name,digest in plan['files'].items():
        if sha(root/name) != digest: raise ValueError('Frozen input/control changed: '+name)
    if tree(root/'source-runtime') != plan['source_runtime_sha256']:
        raise ValueError('Frozen source runtime changed')
    if tree(root/'autotune-seed') != read(root/'autotune-seed-proof.json'):
        raise ValueError('Frozen autotuning seed changed')
    for trial in plan['trials']:
        arm = root/trial['name']
        if tree(arm/'runtime-v1') != read(arm/'runtime-proof.json'):
            raise ValueError('Frozen arm runtime changed')
        validate_controls(read(arm/'fixture.json'),read(arm/'profile.json'))
    return plan


def arm(root, name):
    plan = verify(root)
    trial, = [r for r in plan['trials'] if r['name']==name]
    dest = root/name;runtime=dest/'runtime-v1'
    seed = cache(root/'autotune-seed')
    if cache(dest/'autotune') != seed: raise ValueError('Trial must start from common autotuning seed')
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts')]
    spec = importlib.util.spec_from_file_location('query_stage_memory',runtime/'scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    def frozen(path):
        if path != dest: raise ValueError('Unexpected diagnostic path')
        verify(root)
        return read(dest/'fixture.json'),read(dest/'profile.json')
    memory.verify = frozen
    original_command=memory.server_command
    def server_command(path,fixture):
        command=original_command(path,fixture)
        if command[1:5]!=['-B','-m','vllm_kvmem','serve']:
            raise ValueError('Unexpected frozen server command')
        return [command[0],'-B',str(runtime/'scripts/_query_stage_server.py'),*command[5:]]
    memory.server_command=server_command
    import benchmark_query_fps_prebuild_live as env_module
    original = env_module.server_environment
    def environment(path):
        env=original(path)
        for key in {k for r in plan['trials'] for k in r['environment']}: env.pop(key,None)
        # serve intentionally clears ambient KVMEM_* values. The identical
        # frozen shim adds only declared switches to the explicit launch plan.
        env['QUERY_STAGE_EXECUTION_JSON']=json.dumps(trial['environment'])
        native=plan.get('native_extension')
        if native is not None:
            env['QUERY_STAGE_NATIVE_BINARY_JSON']=json.dumps(dict(
                path=str(runtime/native['relative_path']),sha256=native['sha256']))
            env['QUERY_STAGE_NATIVE_AUDIT_OUTPUT']=str(dest/'run/native-runtime-identity.json')
        else:
            env.pop('QUERY_STAGE_NATIVE_BINARY_JSON',None)
            env.pop('QUERY_STAGE_NATIVE_AUDIT_OUTPUT',None)
        prefill_state=plan.get('prefill_state_extension')
        if prefill_state is not None:
            env['QUERY_STAGE_PREFILL_STATE_BINARY_JSON']=json.dumps(dict(
                path=str(runtime/prefill_state['relative_path']),sha256=prefill_state['sha256']))
            env['QUERY_STAGE_PREFILL_STATE_AUDIT_OUTPUT']=str(dest/'run/prefill-state-runtime-identity.json')
        else:
            env.pop('QUERY_STAGE_PREFILL_STATE_BINARY_JSON',None)
            env.pop('QUERY_STAGE_PREFILL_STATE_AUDIT_OUTPUT',None)
        env['VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR'] = str(dest/'autotune')
        return env
    env_module.server_environment=environment
    try:
        memory.run(dest,workers=plan['workers'])
        launch=read(dest/'run/server/launch.json')
        if any(launch['environment'].get(k)!=v for k,v in trial['environment'].items()):
            raise ValueError('Declared Query execution switch did not reach the actual launch')
        if plan.get('native_extension') is not None:
            native_audit=read(dest/'run/native-runtime-identity.json')
            if (not native_audit['passed'] or native_audit['sha256']!=plan['native_extension']['sha256'] or
                    Path(native_audit['path']).resolve()!=(runtime/plan['native_extension']['relative_path']).resolve()):
                raise ValueError('Expected native binary was not imported')
        if plan.get('prefill_state_extension') is not None:
            expected=plan['prefill_state_extension']
            audit=read(dest/'run/prefill-state-runtime-identity.json')
            if (not audit['passed'] or audit['sha256']!=expected['sha256'] or
                    Path(audit['path']).resolve()!=(runtime/expected['relative_path']).resolve()):
                raise ValueError('Expected prefill state binary was not imported')
    finally:
        env_module.server_environment=original
        final=cache(dest/'autotune')
        save(dest/'autotune-audit.json',dict(initial=seed,final=final,unchanged=seed==final))


def probe_output_audit(destination, *, enabled, case_ids):
    """Require actual complete per-layer records, not only a launch switch."""
    path=destination/'run/server/capture/query_stage_profile.jsonl'
    if not enabled:
        if path.exists() and path.read_text().strip():
            raise ValueError('Disabled probe unexpectedly recorded Query stages')
        return dict(enabled=False,records=0,passed=True)
    if not path.exists(): raise ValueError('Enabled Query probe did not record stages')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    expected=set()
    for cid in case_ids:
        result=read(destination/'run/answers/cases'/cid/'result.json')
        layers=result['layerwise']['layers']
        ids={row['request_id'] for row in layers}
        if len(ids)!=1 or [row['layer'] for row in layers]!=list(range(16)):
            raise ValueError('Case lacks one complete Query owner')
        expected.update(ids)
    if len(rows)!=len(case_ids) or {row['request_id'] for row in rows}!=expected:
        raise ValueError('Query profile is missing, duplicated, or belongs to another request')
    required=('score_gpu_dispatch','score_download_wait','selection_cpu',
              'assembler_views_prepare','archive_report','h2d_copy','layout_and_rebuild')
    for row in rows:
        if any(row[key] for key in ('dropped_rows','dropped_gpu_events','unresolved_gpu_events')):
            raise ValueError('Incomplete Query probe observation')
        for stage in required:
            if {s.get('layer') for s in row['stages'] if s['stage']==stage}!=set(range(16)):
                raise ValueError('Missing per-layer Query probe stage: '+stage)
    return dict(enabled=True,records=len(rows),request_ids=sorted(expected),passed=True,
                source_sha256=sha(path))


def summarize(root):
    plan=verify(root);sys.path.insert(0,str(root/'source-runtime/scripts'))
    from audit_layerwise_gilopt import compare
    comparisons=[];reports={}
    first=root/plan['trials'][0]['name']
    for trial in plan['trials']:
        dest=root/trial['name']
        if read(dest/'status.json')['status']!='completed': raise ValueError('Incomplete trial')
        if read(dest/'run/cleanup.json')['gpu_processes']: raise ValueError('GPU remains owned')
        if dest!=first: comparisons.append(compare(first,dest,plan['case_ids']))
        from benchmark_client_io import result_events
        cases=[read(dest/'run/answers/cases'/cid/'result.json') for cid in plan['case_ids']]
        case_events=[result_events(case,dest/'run/answers/cases'/cid) for case,cid in zip(cases,plan['case_ids'])]
        durations=[];short=[];scoring=[];assembly=[]
        for case,es in zip(cases,case_events):
            ls=case['layerwise']['layers']
            a=next(r for r in es if r['event']=='armed');b=next(r for r in es if r['event']=='completed')
            value=b['monotonic_at']-a['monotonic_at'];durations.append(value)
            if case['attention']['query_score_tokens']<=256: short.append(value)
            scoring.append(sum(r['scoring_seconds'] for r in ls))
            assembly.append(sum(r['assembly_host_seconds'] for r in ls))
        reports[trial['name']]=dict(variant=trial['variant'],environment=trial['environment'],
            inference=read(dest/'inference-performance.json'),query_chain_seconds=durations,
            query_chain_mean_seconds=statistics.mean(durations),
            short_query_mean_seconds=statistics.mean(short) if short else None,
            mean_scoring_host_seconds=statistics.mean(scoring),mean_assembly_host_seconds=statistics.mean(assembly),
            warm_cases=sum(r['prefix_reused'] for r in cases),
            autotune_unchanged=read(dest/'autotune-audit.json')['unchanged'],
            probe_audit=probe_output_audit(dest,
                enabled=trial['environment'].get('KVMEM_QUERY_STAGE_PROFILE','0')=='1',
                case_ids=plan['case_ids']))
    report=dict(reports=reports,comparisons=comparisons,
        full_input_equivalent=all(c['passed'] for c in comparisons),
        scope='Query chain ends after final layer installation, before its attention; host scoring includes '
              'previous GPU work. Inference wall includes cold prefill and warm restore. Diagnostic arms '
              'cannot establish speedup. No automatic retry, formal resume, or external grading.')
    save(root/'comparison.json',report)
    return report


def run(root):
    plan=verify(root);save(root/'queue-process.json',dict(pid=os.getpid(),at=time.time()))
    sys.path.insert(0,str(root/'source-runtime/scripts'))
    from run_recall_comparison import stop_owned,termination_guard
    child=None
    with termination_guard():
        try:
            for trial in plan['trials']:
                name=trial['name'];save(root/'status.json',dict(status='running',active_arm=name,at=time.time()),replace=True)
                command=[sys.executable,'-B',str(root/'launch.py'),'arm','--root',str(root),'--arm',name]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'child.json',dict(pid=child.pid,command=command),replace=True)
                child.wait()
                if child.returncode: raise RuntimeError(name+' failed; preserve evidence, no automatic retry')
                cleanup=read(root/name/'run/cleanup.json')
                if cleanup['gpu_processes'] or cleanup['remaining_owned_groups']:
                    raise RuntimeError('Completed arm did not release owned GPU resources')
            report=summarize(root)
            save(root/'status.json',dict(status='completed',equivalent=report['full_input_equivalent'],at=time.time()),replace=True)
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()),replace=True);raise
        finally: stop_owned(child)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','arm','verify','analyze'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm')
    parser.add_argument('--source-arm',type=Path,default=REFERENCE/'static1')
    parser.add_argument('--fixture',type=Path,default=FORMAL/'fixture.json')
    parser.add_argument('--overlay-source',type=Path,default=REPO)
    parser.add_argument('--overlays-json',type=Path)
    parser.add_argument('--native-binary',type=Path)
    parser.add_argument('--prefill-state-binary',type=Path)
    parser.add_argument('--case-ids-json',type=Path)
    parser.add_argument('--autotune-source',type=Path,default=REFERENCE/'autotune-seed')
    parser.add_argument('--variants-json',type=Path)
    parser.add_argument('--order',default='baseline,probe')
    parser.add_argument('--port',type=int,default=19450)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':
        prepare(root,source_arm=args.source_arm,fixture_path=args.fixture,overlay_source=args.overlay_source,
            overlays=read(args.overlays_json) if args.overlays_json else OVERLAYS,
            case_ids=read(args.case_ids_json) if args.case_ids_json else DEFAULT_IDS,
            autotune_source=args.autotune_source,variants=read(args.variants_json) if args.variants_json else None,
            order=tuple(args.order.split(',')),port=args.port,native_binary=args.native_binary,
            prefill_state_binary=args.prefill_state_binary)
    elif args.action=='run':run(root)
    elif args.action=='arm':arm(root,args.arm)
    elif args.action=='analyze':summarize(root)
    else:verify(root)


if __name__=='__main__':main()
