"""Offline checks only: no model, Docker, GPU, client or external requests."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import queue_deepswe_hybrid_lru as queue


def workload():
    result = {name: {'unchanged': name} for name in queue.WORKLOAD_FIELDS}
    result.update(tasks=[{'task_id': f'fixed-{i:02d}'} for i in range(16)], workers=8,
        attempts_per_task=1, duration_seconds=0, seed=1000, optimized_handoff=True,
        carried_results=[], profile=dict(max_model_len=307200, max_num_seqs=8,
            served_model_name='kvmem-deepswe', port=18880, kvmem=dict(graph_mode='multi',
                graph_alpha=0., cpu_seal_serial_allocate=True, historical_gc=None,
                select_budget=36864, gen_budget=16384, query_mean_quota_fraction=.5,
                query_fps_cache_fp8=True, cpu_release_sealed_sums=True)))
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    queue.q.save(path, value)


def test_profile_changes_only_lru_compatibility_and_preserves_original():
    original = workload()['profile']; before = deepcopy(original)
    after = queue.profile_from(original)
    assert original == before
    assert after['kvmem'].pop('historical_gc') == queue.LRU
    before['kvmem'].pop('historical_gc')
    after['kvmem']['graph_mode'] = before['kvmem']['graph_mode']
    after['kvmem']['cpu_seal_serial_allocate'] = before['kvmem']['cpu_seal_serial_allocate']
    after['served_model_name'] = before['served_model_name']
    assert after == before


@pytest.mark.parametrize('field', queue.WORKLOAD_FIELDS+('carried_results',))
def test_original_workload_cannot_change(field):
    before = workload(); after = deepcopy(before)
    after['profile'] = queue.profile_from(before['profile'])
    queue.validate_workload(before, after)
    after[field] = ['changed']
    with pytest.raises(ValueError):
        queue.validate_workload(before, after)


def test_sampling_recall_and_logical_cap_stay_fixed():
    before = workload(); after = deepcopy(before)
    after['profile'] = queue.profile_from(before['profile'])
    after['profile']['kvmem']['graph_alpha'] = .2
    with pytest.raises(ValueError, match='Only historical LRU'):
        queue.validate_workload(before, after)
    after['profile'] = queue.profile_from(before['profile'])
    after['profile']['max_model_len'] = 2**21
    with pytest.raises(ValueError, match='300K'):
        queue.validate_workload(before, after)


def test_existing_root_rejected_without_touching_evidence(tmp_path):
    (tmp_path/'evidence').write_text('old')
    with pytest.raises(ValueError, match='no automatic retry'):
        queue.prepare(tmp_path/'source', tmp_path, tmp_path/'validation', tmp_path/'profile')
    assert (tmp_path/'evidence').read_text() == 'old'


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    workspace = tmp_path/'workspace'; workspace.mkdir()
    monkeypatch.setattr(queue, 'ROOT', workspace)
    source = tmp_path/'source'; source.mkdir()
    for directory in queue.RUNTIME_DIRECTORIES:
        (workspace/directory).mkdir()
        (workspace/directory/'current.py').write_text(directory+' current')
    (workspace/'pyproject.toml').write_text('[project]\nname="frozen"\n')
    required = ('src/vllm_kvmem/authority.py', 'src/vllm_kvmem/historical_gc.py',
        'src/vllm_kvmem/query_key_fps.py', 'tests/test_historical_hybrid_gpu.py')
    for relative in required:
        path = workspace/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('qualified '+relative)
    for forbidden in ('artifacts', 'data', '.git', '.venv', 'venv'):
        path = workspace/'tests'/forbidden; path.mkdir()
        (path/'secret').write_text('must not freeze')
    (source/'runtime/src').mkdir(parents=True)
    (source/'runtime/src/old.py').write_text('old runtime')
    (source/'first16/protocol').mkdir(parents=True)
    (source/'first16/protocol/protocol.so').write_bytes(b'frozen protocol')
    (source/'first16/setup').mkdir()
    (source/'first16/setup/runner.sh').write_text('fixed task setup')
    before = workload()
    save(source/'first16/manifest.json', before)
    save(source/'first16/profile.json', before['profile'])
    save(source/'first16/baseline-comparison.json', {'old': 'comparison'})
    save(source/'previous-vllm-outcomes.json', {'old': 'outcome'})
    save(source/'status.json', {'status': 'failed'})
    save(source/'queue-process.json', {'pid': 100, 'created': 1})
    save(source/'cleanup.json', dict(owned=[], remaining_processes=[], remaining_owned_gpu_pids=[]))
    external = tmp_path/'external'; external.write_text('fixed external client')
    save(source/'manifest.json', dict(files={}, python='/fixed/python', bridge_port=18490,
                                      external_files={str(external): queue.q.sha(external)}))
    monkeypatch.setattr(queue.q, 'live', lambda _: None)
    profile = workspace/'artifacts/lru/profile.json'
    save(profile, queue.profile_from(before['profile']))
    validation = profile.with_name('validation.json')
    hashes = {relative:queue.q.sha(workspace/relative) for relative in required}
    hashes[str(profile.relative_to(workspace))] = queue.q.sha(profile)
    save(validation, dict(status='component_validation_passed', cpu={'passed':1},
        cuda_hybrid={'passed':1}, native_index={'ctest_passed':1}, sha256=hashes))
    def runner(args):
        result = deepcopy(before); result['profile'] = queue.q.read(args.config)
        assert args.workers == 8 and args.bash_repeat_limit == 32
        assert args.carry_result == [] and args.duration_seconds == 0
        assert args.backend_kind == 'kvmem' and args.optimized_handoff
        return result
    monkeypatch.setattr(queue.runner, 'prepare', runner)
    return SimpleNamespace(workspace=workspace, source=source, output=tmp_path/'new',
        validation=validation, profile=profile, external=external)


def test_prepare_freezes_current_explicit_directories_protocol_and_hashes_every_file(prepared):
    p = prepared
    before = queue.compact.preserved_hashes(p.source)
    queue.prepare(p.source, p.output, p.validation, p.profile)
    manifest = queue.validate(p.output)
    assert before == queue.compact.preserved_hashes(p.source)
    assert set(path.name for path in (p.output/'runtime').iterdir()) == set(queue.RUNTIME_DIRECTORIES+queue.RUNTIME_FILES)
    assert (p.output/'first16/protocol/protocol.so').read_bytes() == b'frozen protocol'
    assert (p.output/'first16/setup/runner.sh').read_text() == 'fixed task setup'
    assert not any('secret' in path for path in manifest['runtime_files'])
    assert 'src/current.py' in manifest['runtime_files'] and 'src/old.py' not in manifest['runtime_files']
    delta = queue.q.read(p.output/'runtime-diff.json')
    assert 'src/current.py' in delta['added'] and 'src/old.py' in delta['removed']
    assert 'component-evidence/artifacts/lru/profile.json' in manifest['files']
    (p.output/'runtime/src/extra.py').write_text('unqualified')
    with pytest.raises(ValueError, match='inventory changed'):
        queue.validate(p.output)


def test_changed_qualified_source_rejected_before_new_root_is_created(prepared):
    p = prepared
    (p.workspace/'src/vllm_kvmem/authority.py').write_text('untested change')
    with pytest.raises(ValueError, match='Source changed after component'):
        queue.prepare(p.source, p.output, p.validation, p.profile)
    assert not p.output.exists()


def test_pending_gpu_proof_requires_explicit_defer_and_records_frozen_gate(prepared):
    p = prepared
    proof = queue.q.read(p.validation)
    proof.update(status='cpu_component_validation_passed',
                 cuda_hybrid={'status': 'pending_frozen_queue'})
    save(p.validation, proof)
    with pytest.raises(ValueError, match='explicitly queued CUDA'):
        queue.prepare(p.source, p.output, p.validation, p.profile)
    assert not p.output.exists()
    queue.prepare(p.source, p.output, p.validation, p.profile, defer_gpu=True)
    manifest = queue.validate(p.output)
    assert manifest['component_gpu_deferred']
    assert manifest['gpu_qualification_before_inference']
    manifest['gpu_qualification_before_inference'] = False
    save(p.output/'manifest.json', manifest)
    with pytest.raises(ValueError, match='pre-inference gate'):
        queue.validate(p.output)


@pytest.mark.parametrize('field,value', [('cpu', {'passed': 0}),
    ('native_index', {'ctest_passed': 0}), ('cuda_hybrid', {'status': 'failed'})])
def test_defer_does_not_waive_failed_or_missing_components(prepared, field, value):
    p = prepared
    proof = queue.q.read(p.validation)
    proof.update(status='cpu_component_validation_passed',
                 cuda_hybrid={'status': 'pending_frozen_queue'})
    proof[field] = value
    save(p.validation, proof)
    with pytest.raises(ValueError, match='validation required'):
        queue.prepare(p.source, p.output, p.validation, p.profile, defer_gpu=True)
    assert not p.output.exists()


def test_failed_gpu_gate_never_launches_model_or_tasks(prepared, monkeypatch):
    p = prepared
    queue.prepare(p.source, p.output, p.validation, p.profile)
    monkeypatch.setattr(queue, 'ROOT', p.output/'runtime')
    monkeypatch.setattr(queue.q, 'gpu_pids', lambda: [])
    monkeypatch.setattr(queue.q, 'identity', lambda _: {'pid': 123, 'created': 1})
    monkeypatch.setattr(queue.q, 'require_free_port', lambda _: None)
    monkeypatch.setattr(queue.old, 'verify_images', lambda _: None)
    monkeypatch.setattr(queue.psutil, 'virtual_memory', lambda: SimpleNamespace(available=128*2**30))
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    clock = iter([0, 0, 61])
    monkeypatch.setattr(queue.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(queue.fcntl, 'flock', lambda *args: None)
    commands = []

    class Owner:
        def start(self, args, *rest):
            commands.append(args)
            return SimpleNamespace(poll=lambda: 1, returncode=1)

        def stop(self):
            return {'remaining_processes': [], 'remaining_owned_gpu_pids': []}

    monkeypatch.setattr(queue.q, 'Owned', Owner)
    with pytest.raises(RuntimeError, match='GPU qualification failed'):
        queue.run(p.output)
    assert len(commands) == 1 and commands[0][3:5] == ['pytest', '-q']
    assert not (p.output/'first16/controller-process.json').exists()
    assert queue.q.read(p.output/'status.json')['status'] == 'failed'


def test_external_client_is_still_bound_after_freeze(prepared):
    p = prepared; queue.prepare(p.source, p.output, p.validation, p.profile)
    p.external.write_text('different client')
    with pytest.raises(ValueError, match='Original task/client changed'):
        queue.validate(p.output)


def test_copied_artifact_proof_is_bound_after_freeze(prepared):
    p = prepared; queue.prepare(p.source, p.output, p.validation, p.profile)
    (p.output/'component-evidence/artifacts/lru/profile.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen test source/profile changed|Component evidence copy changed'):
        queue.validate(p.output)


def test_old_runtime_difference_evidence_cannot_silently_change(prepared):
    p = prepared; queue.prepare(p.source, p.output, p.validation, p.profile)
    (p.source/'runtime/src/old.py').write_text('changed historical code')
    with pytest.raises(ValueError, match='Prior runtime inventory changed'):
        queue.validate(p.output)


def freeze_source_file(source, relative, content):
    path = source/relative; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    manifest = queue.q.read(source/'manifest.json')
    manifest['files'][relative] = queue.q.sha(path)
    save(source/'manifest.json', manifest)
    return path


def test_only_observed_nodeids_mismatch_is_recorded_and_cannot_change_again(prepared):
    p = prepared
    cache = freeze_source_file(p.source, queue.SOURCE_MUTABLE_CACHE, '["old::test"]')
    original_manifest = (p.source/'manifest.json').read_bytes()
    original_digest = queue.q.sha(cache)
    cache.write_text('["old::test", "gpu::test"]')
    current_digest = queue.q.sha(cache)
    queue.prepare(p.source, p.output, p.validation, p.profile)
    audit = queue.q.read(p.output/'source-cache-audit.json')
    assert audit['checked_files'] == 1
    assert len(audit['changes']) == 1
    change = audit['changes'][0]
    assert change['path'] == queue.SOURCE_MUTABLE_CACHE
    assert change['original_sha256'] == original_digest
    assert change['current_sha256'] == current_digest
    assert (p.source/'manifest.json').read_bytes() == original_manifest
    queue.validate(p.output)
    cache.write_text('["another::test"]')
    with pytest.raises(ValueError, match='cache changed again'):
        queue.validate(p.output)


@pytest.mark.parametrize('relative', ['runtime/src/model.py', 'first16/profile.json',
    'runtime/.pytest_cache/v/cache/lastfailed', 'runtime/src/__pycache__/model.pyc'])
def test_generated_cache_exception_does_not_waive_other_frozen_files(prepared, relative):
    p = prepared
    path = freeze_source_file(p.source, relative, 'original')
    path.write_text('changed')
    with pytest.raises(ValueError, match='Frozen predecessor source/profile changed'):
        queue.prepare(p.source, p.output, p.validation, p.profile)
    assert not p.output.exists()


def test_cache_exception_rejects_non_nodeid_content(prepared):
    p = prepared
    path = freeze_source_file(p.source, queue.SOURCE_MUTABLE_CACHE, '[]')
    path.write_text('{"not": "test nodeids"}')
    with pytest.raises(ValueError, match='cache format'):
        queue.verify_source(p.source)


def test_live_prior_or_incomplete_cleanup_cannot_be_overridden(prepared, monkeypatch):
    p = prepared
    monkeypatch.setattr(queue.q, 'live', lambda _: object())
    with pytest.raises(ValueError, match='controller is still running'):
        queue.stopped_prior(p.source)
    monkeypatch.setattr(queue.q, 'live', lambda _: None)
    save(p.source/'cleanup.json', dict(owned=[], remaining_processes=[1], remaining_owned_gpu_pids=[]))
    with pytest.raises(ValueError, match='cleanup was incomplete'):
        queue.stopped_prior(p.source)


def test_runtime_symlink_escape_rejected(tmp_path):
    (tmp_path/'escape').symlink_to('/tmp')
    with pytest.raises(ValueError, match='symlinks'):
        queue.inventory(tmp_path)


def test_gpu_qualification_explicitly_enables_both_gpu_gates():
    env = {'original': 'kept'}
    args, qualified = queue.qualification_command({'python':'/fixed/python'}, env)
    assert env == {'original':'kept'}
    assert args == ['/fixed/python', '-B', '-m', 'pytest', '-q', *queue.GPU_TESTS]
    assert qualified['KVMEM_TEST_CUDA'] == qualified['KVMEM_RUN_GPU_TESTS'] == '1'


def test_busy_coordination_lock_waits_without_starting_gpu(tmp_path, monkeypatch):
    attempts, waits = [], []

    def acquire(lock, flags):
        attempts.append(flags)
        if len(attempts) < 3:
            raise BlockingIOError(11, 'owned by another test')

    monkeypatch.setattr(queue.fcntl, 'flock', acquire)
    monkeypatch.setattr(queue.time, 'sleep', waits.append)
    queue.acquire_coordination(tmp_path, object(), 15)
    assert len(attempts) == 3 and waits == [15, 15]
    assert queue.q.read(tmp_path/'status.json')['reason'] == 'another_qualification_holds_lock'
    assert not (tmp_path/'gpu-component-process.json').exists()


def test_waiting_for_coordination_still_obeys_stop(tmp_path, monkeypatch):
    def acquire(*args):
        raise BlockingIOError(11, 'busy')

    monkeypatch.setattr(queue.fcntl, 'flock', acquire)
    monkeypatch.setattr(queue.time, 'sleep', lambda _: (tmp_path/'STOP').write_text('user'))
    with pytest.raises(InterruptedError, match='User stopped'):
        queue.acquire_coordination(tmp_path, object(), 15)


def test_coordination_other_errors_are_not_hidden(tmp_path, monkeypatch):
    def acquire(*args):
        raise OSError(5, 'broken lock filesystem')

    monkeypatch.setattr(queue.fcntl, 'flock', acquire)
    with pytest.raises(OSError, match='broken lock'):
        queue.acquire_coordination(tmp_path, object(), 15)


def test_existing_attempt_cannot_restart(prepared, monkeypatch):
    p = prepared; queue.prepare(p.source, p.output, p.validation, p.profile)
    monkeypatch.setattr(queue, 'ROOT', p.output/'runtime')
    save(p.output/'queue-process.json', {'pid':100, 'created':1})
    with pytest.raises(ValueError, match='No automatic retry'):
        queue.run(p.output)


def test_samples_keep_numeric_memory_and_lru_without_private_content(tmp_path, monkeypatch):
    capture = tmp_path/'first16/server/capture'
    save(capture/'archive.json', dict(cpu_authority_bytes=100, completed=[{'tokens':['SECRET']}],
        historical_gc=dict(mode='delete', policy='lru', retired=7, cpu_released=6, secret='SECRET')))
    save(capture/'progress.json', dict(accepted_history={'SECRET':300000}, batches_submitted=20))
    save(capture/'public_work.json', {'counter_id':'a'*32, 'pid':123})
    save(capture/'scheduler_resources.json', {'cpu_history_admission': dict(capacity_bytes=200,
        available_bytes=100, reserved_bytes=80, progress_reserve_bytes=20, leases=8,
        lease_views=[{'key':'SECRET'}], total_wait_seconds=1.5)})
    save(tmp_path/'first16/run.json', {'status':'running', 'tasks':{'SECRET':{'state':'running'}}})
    monkeypatch.setattr(queue.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1000))
    result = queue.sample(tmp_path)
    assert result['phase'] == 'formal'
    assert result['historical_gc']['archive']['retired'] == 7
    assert result['cpu_admission']['scheduler_resources']['capacity_bytes'] == 200
    assert result['cpu_admission']['scheduler_resources']['total_wait_seconds'] == 1.5
    assert result['accepted_history'] == dict(owners=1, total=300000, maximum=300000)
    assert 'SECRET' not in (tmp_path/'lru-samples.jsonl').read_text()


def row(retired=0, cpu=0, slots=0, *, phase='formal', instance='a'*32, failures=0):
    return dict(phase=phase, engine_instance=instance, historical_gc={'archive':dict(
        mode='delete', policy='lru', retired=retired, cpu_released=cpu, slots_released=slots,
        policy_failures=failures)})


@pytest.mark.parametrize('rows,status', [
    ([row(),row(5,5,5)], 'passed'),
    ([row(),row(0,50,50)], 'not_exercised'),
    ([row(),row(5,5,5,phase='cleanup')], 'not_exercised'),
    ([row(),row(5,5,5,phase='model_qualification')], 'not_exercised'),
    ([row(),row(5,5,5,instance='b'*32)], 'not_exercised'),
    ([row(),row(5,5,0)], 'not_exercised'),
    ([row(),row(5,5,5,failures=1)], 'failed'),
])
def test_lru_audit_requires_formal_policy_and_release_within_same_instance(tmp_path, rows, status):
    (tmp_path/'lru-samples.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in rows))
    assert queue.lru_review(tmp_path)['status'] == status
