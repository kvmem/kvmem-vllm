import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import benchmark_agent_metadata_handoff as benchmark
from scripts import recover_agent_metadata_port as recovery
from scripts import run_agent_handoff_queue as q

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path, monkeypatch):
    source = tmp_path/'old'
    before, after = [source/'optimization'/name for name in ('before', 'after')]
    (before/'answers').mkdir(parents=True)
    (after/'runtime/scripts').mkdir(parents=True)
    (after/'runtime/tests').mkdir()
    (after/'runtime/src').mkdir()
    (after/'runtime/src/model.py').write_text('MODEL = 42\n')
    shutil.copy2(REPO/'scripts/run_agent_handoff_queue.py', after/'runtime/scripts/run_agent_handoff_queue.py')
    fixture = [dict(case=str(i), tokens=[1, 2, i], output_token=100+i) for i in range(8)]
    q.save(source/'fixture.json', fixture)
    q.save(source/'fixture-source-hashes.json', {})
    for i, item in enumerate(fixture):
        for turn in range(8):
            q.save(before/'answers'/f'{i}-{turn}.json', dict(case=str(i), turn=turn,
                input_sha256=hashlib.sha256(json.dumps(benchmark.fixed_input(item, turn)).encode()).hexdigest(),
                exact_output_equal=True, response=dict(choices=[dict(token_ids=[100+i]*64)])))
    q.save(before/'evidence-hashes.json', {str(p.relative_to(before)):q.sha(p) for p in (before/'answers').glob('*.json')})
    q.save(before/'status.json', dict(status='completed'))
    q.save(before/'execution-audit.json', dict(status='passed', responses=64, restores=56, exact_output_equal=True))
    for arm in (before, after):
        q.save(arm/'cleanup.json', dict(owned=[], remaining_processes=[], remaining_owned_gpu_pids=[]))
        q.save(arm/'profile.json', dict(port=18878, K=36864))
    q.save(before/'performance.json', dict(restores=56))
    q.save(after/'status.json', dict(status='prepared'))
    q.save(after/'started.json', dict(at=123))
    q.save(source/'status.json', dict(status='failed', error="OSError(98, 'Address already in use')"))
    q.save(source/'queue-process.json', dict(pid=123, created=1))
    (source/'queue.log').write_text('require_free_port: Address already in use\n')
    q.save(source/'manifest.json', dict(files={}))
    for name in ('baseline', 'candidate', 'qualification'):
        (source/name).mkdir()
    q.save(source/'baseline-outcome.json', dict(status='incomplete_predeclared_limit'))
    monkeypatch.setattr(benchmark, 'validate', q.verify_frozen)
    return source


def test_recovery_reuses_all_before_answers_and_preserves_failed_start(source, tmp_path):
    output = tmp_path/'new'
    original = {str(p.relative_to(source)):q.sha(p) for p in source.rglob('*') if p.is_file()}
    recovery.prepare(source, output, REPO, probe=lambda _:None)
    plan = q.verify_frozen(output)
    recovery.verify_recovery(output, plan)
    assert plan['execution_arms'] == ['after']
    assert (output/'optimization/before').resolve() == source/'optimization/before'
    assert not (output/'optimization/after/started.json').exists()
    assert not (output/'optimization/after/answers').exists()
    assert all(q.sha(source/rel) == digest for rel, digest in original.items())
    assert q.sha(output/'optimization/after/runtime/src/model.py') == q.sha(source/'optimization/after/runtime/src/model.py')
    assert q.sha(output/'fixture.json') == q.sha(source/'fixture.json')


@pytest.mark.parametrize('problem', ['run', 'answers', 'startup.json', 'execution-audit.json', 'performance.json'])
def test_any_after_execution_prevents_startup_recovery(source, problem):
    (source/'optimization/after'/problem).touch()
    with pytest.raises(ValueError, match='may have executed'):
        recovery.validate_source(source, probe=lambda _:None)


def test_live_old_queue_prevents_recovery(source):
    with pytest.raises(ValueError, match='still alive'):
        recovery.validate_source(source, probe=lambda _:object())


@pytest.mark.parametrize('problem', ['answer', 'fixture', 'audit', 'cleanup', 'error'])
def test_invalid_source_evidence_prevents_recovery(source, problem):
    before = source/'optimization/before'
    if problem == 'answer':
        (before/'answers/0-0.json').write_text('{}')
    elif problem == 'fixture':
        fixture = q.read(source/'fixture.json'); fixture[0]['tokens'] = [999]
        q.save(source/'fixture.json', fixture)
    elif problem == 'audit':
        q.save(before/'execution-audit.json', dict(status='failed'))
    elif problem == 'cleanup':
        q.save(before/'cleanup.json', dict(remaining_processes=[1], remaining_owned_gpu_pids=[]))
    else:
        q.save(source/'status.json', dict(status='failed', error='model error'))
    with pytest.raises((ValueError, KeyError)):
        recovery.validate_source(source, probe=lambda _:None)


def test_preserved_answer_changes_detected_after_preparation(source, tmp_path):
    output = tmp_path/'new'
    recovery.prepare(source, output, REPO, probe=lambda _:None)
    with (source/'optimization/before/answers/0-0.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='evidence changed'):
        recovery.verify_recovery(output, q.read(output/'manifest.json'))


def test_port_patch_does_not_copy_other_queue_changes():
    old = 'A=1\ndef require_free_port(port):\n return port\ndef other():\n return 7\n'
    new = 'A=999\ndef require_free_port(port):\n return port+1\ndef other():\n return 999\n'
    assert recovery.replace_port_check(old, new) == old.replace('return port\n', 'return port+1\n')


def test_runner_never_calls_completed_before_arm(source, tmp_path, monkeypatch):
    output = tmp_path/'new'
    recovery.prepare(source, output, REPO, probe=lambda _:None)
    monkeypatch.setattr(benchmark, 'ROOT', output/'optimization/after/runtime')
    monkeypatch.setattr(q, 'wait_dependency', lambda *_:None)
    monkeypatch.setattr(q, 'gpu_pids', lambda:[])
    calls = []
    def run_arm(root, name, plan):
        calls.append(name)
        shutil.copytree(root/'optimization/before/answers', root/'optimization/after/answers')
        q.save(root/'optimization/after/performance.json', dict(restores=56))
    monkeypatch.setattr(benchmark, 'run_arm', run_arm)
    benchmark.run(output)
    assert calls == ['after']
    assert q.read(output/'status.json')['full_paired_test_passed'] is False
