from pathlib import Path

import pytest

from scripts.benchmark_agent_fastpath_dsh import task_plan,outcome
from scripts.run_agent_handoff_queue import optimized_bridge,save,sha


def test_bridge_control_can_keep_existing_optimization_without_changing_legacy():
    assert not optimized_bridge({},'baseline') and optimized_bridge({},'candidate')
    plan={'optimized_bridge_arms':['baseline','candidate']}
    assert all(optimized_bridge(plan,x) for x in ['baseline','candidate'])
    with pytest.raises(ValueError):
        optimized_bridge({'optimized_bridge_arms':['alien']},'candidate')


def test_fresh_tasks_preserve_prompts_limits_and_sampling():
    original=dict(minimum_turns=4,max_steps_per_turn=240,temperature=.6,
        cases=[dict(id=f'c{i}',task=f't{i}',prompts=['first','followup']*2,session_id='old',
                    workspace='/old',preview_port=1) for i in range(8)])
    root=Path('/tmp/new-agent-test')
    a,b=[task_plan(original,root,name) for name in ['baseline','candidate']]
    assert original['cases'][0]['session_id']=='old'
    assert a['max_steps_per_turn']==b['max_steps_per_turn']==240
    assert len({x['session_id'] for p in [a,b] for x in p['cases']})==16
    for old,left,right in zip(original['cases'],a['cases'],b['cases']):
        assert left['prompts']==right['prompts']==old['prompts']
        assert left['workspace']!=right['workspace'] and left['preview_port']!=right['preview_port']


def setup_outcome(root,error,steps=240):
    case=root/'cases/one';(case/'turn-04').mkdir(parents=True)
    save(root/'execution-audit.json',dict(status='failed',errors=['Incomplete/context cohort','Missing followups'],
        counts={'completed_tasks':0,'bridge_errors':0}))
    save(root/'cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))
    save(root/'task-plan.json',dict(cases=[dict(id='one')],max_steps_per_turn=240,turn_timeout_seconds=3600))
    save(case/'failure.json',dict(error=error))
    save(case/'turn-04/limit.json',dict(steps=steps,elapsed=100))
    save(root/'evidence-hashes.json',{'cases/one/failure.json':sha(case/'failure.json')})


def test_resource_limit_is_an_explicit_outcome_not_a_success_or_retry(tmp_path):
    setup_outcome(tmp_path,"RuntimeError('Predeclared turn resource limit; original attempt retained')")
    result=outcome(tmp_path)
    assert result['status']=='completed_with_resource_limits'
    assert not result['full_cohort_passed'] and not result['retry_allowed']


@pytest.mark.parametrize('error,steps',[("RuntimeError('engine crash')",240),
    ("RuntimeError('Predeclared turn resource limit; original attempt retained')",23)])
def test_infrastructure_or_unproven_limit_is_never_accepted(tmp_path,error,steps):
    setup_outcome(tmp_path,error,steps)
    with pytest.raises(ValueError):outcome(tmp_path)
