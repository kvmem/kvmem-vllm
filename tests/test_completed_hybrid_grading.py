from pathlib import Path

import pytest

from benchmark_niah import save,sha
from run_hybrid_suite_queue import completed_grading


def fixture(root):
    campaign = root/'inference'; output = root/'grade'
    save(campaign/'execution-audit.json',dict(status='passed',cases=102))
    save(output/'plan.json',dict(inference_audit_sha256=sha(campaign/'execution-audit.json')))
    save(output/'status.json',dict(status='completed',judge='deepseek-flash'))
    save(output/'scores.json',dict(status='completed',arms=dict(hybrid=dict(count=102,correct=95))))
    proof = dict(cases=102,files={str(p):sha(p) for p in output.glob('*.json')})
    return dict(campaign=str(campaign),output=str(output),suite='longmemeval_s',completed_proof=proof)


def test_complete_grade_is_reused_without_another_api_run(tmp_path):
    record = fixture(tmp_path)
    completed_grading(record)
    assert len(list((tmp_path/'grade').iterdir())) == 3


@pytest.mark.parametrize('fault',['partial','tampered','other_inference'])
def test_partial_changed_or_unrelated_grades_are_never_accepted(tmp_path,fault):
    record = fixture(tmp_path)
    if fault == 'partial': record['completed_proof']['cases'] = 61
    elif fault == 'tampered': save(Path(record['output'])/'scores.json',dict(status='completed',arms={}))
    else: save(Path(record['campaign'])/'execution-audit.json',dict(status='passed',cases=102,changed=True))
    with pytest.raises(ValueError): completed_grading(record)


def test_grading_cache_wait_accepts_completion_and_rejects_failure(tmp_path):
    from run_hybrid_suite_queue import wait_for_grading_cache
    save(tmp_path/'status.json',dict(status='completed'))
    wait_for_grading_cache(tmp_path)
    save(tmp_path/'status.json',dict(status='failed'))
    with pytest.raises(ValueError,match='Snapshot grading failed'):
        wait_for_grading_cache(tmp_path)


def test_queue_binds_separate_grading_environment_without_changing_inference_python(tmp_path):
    import sys
    from run_hybrid_suite_queue import grading_python
    env = tmp_path/'pyvenv.cfg'; env.write_text('bound grading environment')
    plan = dict(grading_python=dict(path=sys.executable,sha256=sha(Path(sys.executable)),
        environment_files={str(env):sha(env)}))
    assert grading_python({}) == sys.executable
    assert grading_python(plan) == sys.executable
    env.write_text('changed')
    with pytest.raises(ValueError,match='environment changed'): grading_python(plan)
