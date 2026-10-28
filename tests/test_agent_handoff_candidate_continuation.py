import json
from pathlib import Path

import pytest

from scripts import continue_agent_handoff_candidate as continuation
from scripts.run_agent_handoff_queue import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    write(root/'status.json', dict(status='failed'))
    write(root/'queue-process.json', dict(pid=123, created=1))
    write(root/'baseline/cleanup.json', dict(remaining_processes=[], remaining_owned_gpu_pids=[], owned=[]))
    write(root/'baseline/execution-audit.json', dict(status='failed', errors=['Incomplete/context cohort', 'Missing followups']))
    tasks = dict(max_steps_per_turn=240, turn_timeout_seconds=3600, minimum_turns=4,
                 cases=[dict(id='r01-fps' if i == 0 else f'task-{i}',
                    session_id=f'unused-{i}', workspace=str(root/f'candidate/workspaces/{i}'),
                    prompts=['original']*4) for i in range(8)])
    for arm in ('baseline', 'candidate'):
        write(root/arm/'task-plan.json', tasks)
        write(root/arm/'profile.json', dict(temperature=.6, seed=1000, graph_alpha=0))
    for row in tasks['cases']:
        write(root/'baseline/cases'/row['id']/'status.json', dict(
            status='running' if row['id'] == 'r01-fps' else 'completed',
            turns_completed=3 if row['id'] == 'r01-fps' else 4))
    write(root/'baseline/cases/r01-fps/failure.json', dict(
        error="RuntimeError('Predeclared turn resource limit; original attempt retained')"))
    write(root/'baseline/cases/r01-fps/turn-04/limit.json', dict(steps=240, elapsed=1337))
    write(root/'baseline/browser-summary.json', dict(checked=7))
    write(root/'baseline/performance.json', dict(interrupted=True))
    evidence = {str(p.relative_to(root/'baseline')):sha(p)
                for p in (root/'baseline/cases').rglob('*.json')}
    write(root/'baseline/evidence-hashes.json', evidence)
    write(root/'candidate/status.json', dict(status='prepared'))
    write(root/'qualification/execution-audit.json', dict(status='passed'))
    (root/'gpu-component.log').write_text('28 passed')
    paths = [f'{arm}/{p}' for arm in ('baseline', 'candidate') for p in ('task-plan.json', 'profile.json')]
    write(root/'manifest.json', dict(files={p:sha(root/p) for p in paths},
        browser_python='unused', reuse_qualification=dict(root=str(root), hashes={
            'qualification/execution-audit.json':sha(root/'qualification/execution-audit.json')})))


def prepared(tmp_path):
    source = tmp_path/'old'; fixture(source)
    dest = tmp_path/'new'
    repo = Path(__file__).resolve().parents[1]
    continuation.prepare(source, dest, repo, probe=lambda _:None)
    return source, dest


def test_preserves_failed_baseline_and_candidate_sampling(tmp_path):
    source, dest = prepared(tmp_path)
    outcome = continuation.verify_preserved_baseline(dest)
    assert outcome['retry_allowed'] is False and outcome['completed_tasks'] == 7
    assert (dest/'baseline').resolve() == source/'baseline'
    assert (dest/'candidate/profile.json').read_bytes() == (source/'candidate/profile.json').read_bytes()
    before = json.loads((source/'candidate/task-plan.json').read_text())
    after = json.loads((dest/'candidate/task-plan.json').read_text())
    for a, b in zip(before['cases'], after['cases']):
        workspace = b.pop('workspace'); expected = a.pop('workspace')
        assert workspace == str(dest/Path(expected).relative_to(source))
        assert a == b
    assert (dest/'baseline/cases/r01-fps/failure.json').exists()
    assert json.loads((dest/'manifest.json').read_text())['execution_arms'] == ['candidate']


@pytest.mark.parametrize('damage', ['live', 'limit', 'transport', 'audit', 'candidate_attempt', 'answer', 'cleanup'])
def test_does_not_hide_unexpected_failure_or_rerun_attempt(tmp_path, damage):
    fixture(tmp_path)
    probe = lambda _:None
    if damage == 'live': probe = lambda _:object()
    if damage == 'limit': write(tmp_path/'baseline/cases/r01-fps/turn-04/limit.json', dict(steps=120, elapsed=100))
    if damage == 'transport': write(tmp_path/'baseline/cases/r01-fps/failure.json', dict(error='TRANSPORT'))
    if damage == 'audit': write(tmp_path/'baseline/execution-audit.json', dict(status='failed', errors=['Recall policy mismatch']))
    if damage == 'candidate_attempt': write(tmp_path/'candidate/started.json', dict(at=1))
    if damage == 'answer': write(tmp_path/'baseline/cases/task-1/status.json', dict(status='completed', turns_completed=5))
    if damage == 'cleanup': write(tmp_path/'baseline/cleanup.json', dict(remaining_processes=[1], remaining_owned_gpu_pids=[]))
    with pytest.raises(ValueError): continuation.validate_baseline(tmp_path, probe)


def test_evidence_change_after_freeze_is_rejected(tmp_path):
    source, dest = prepared(tmp_path)
    write(source/'baseline/cases/task-1/status.json', dict(status='completed', turns_completed=9))
    with pytest.raises(ValueError, match='answers changed'):
        continuation.verify_preserved_baseline(dest)


def test_only_candidate_runs_and_completion_remains_incomplete(tmp_path, monkeypatch):
    source, dest = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(continuation, 'wait_dependency', lambda *_:None)
    monkeypatch.setattr(continuation, 'require_idle_dsh', lambda *_:None)
    monkeypatch.setattr(continuation, 'gpu_pids', lambda:[])
    monkeypatch.setattr(continuation.subprocess, 'run', lambda *a, **k:None)
    def arm(root, name, plan):
        calls.append(name)
        write(root/name/'performance.json', dict(accepted_output_tokens=42))
    monkeypatch.setattr(continuation, 'arm', arm)
    continuation.run(dest)
    assert calls == ['candidate']
    state = json.loads((dest/'status.json').read_text())
    assert state['status'] == 'needs_optimization_review'
    assert state['full_paired_test_passed'] is False
    with pytest.raises(RuntimeError, match='Never restart'):
        continuation.run(dest)
