import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from scripts.queue_deepswe_seed_suite import SEEDS, validate_manifest, audit_seeds, summarize

spec=importlib.util.spec_from_file_location('seed_suite_bridge',Path(__file__).parents[1]/'benchmarks/deepswe/claude_bridge.py')
bridge=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bridge;spec.loader.exec_module(bridge)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def manifest():
    return dict(seeds=list(SEEDS),expected_attempts=64,task_ids=[f'task-{i}' for i in range(16)])


@pytest.mark.parametrize('backend',['native','kvmem'])
def test_distinct_seeds_reach_model_without_changing_sampling_tokens_or_plan(backend):
    values=dict(model='m',tokens=[1,2,3],session='session',max_tokens=16384,
                plan={'kvmem_reselect':'force'},backend_kind=backend)
    before=copy.deepcopy(values)
    original=bridge.backend_request(**values)
    assert original['seed']==1000
    for seed in SEEDS:
        actual=bridge.backend_request(**values,seed=seed)
        assert actual==dict(original,seed=seed)
    assert values==before


@pytest.mark.parametrize('seed',[True,-1,2**32,1001.0,'1001'])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError,match='seed'):
        bridge.backend_request(model='m',tokens=[1],session='s',max_tokens=1,plan={},backend_kind='kvmem',seed=seed)


@pytest.mark.parametrize('bad',['seed','duplicate','count'])
def test_fixed_suite_rejects_changed_cohort(bad):
    value=manifest()
    if bad=='seed':value['seeds'][0]=1000
    elif bad=='duplicate':value['task_ids'][0]=value['task_ids'][1]
    else:value['expected_attempts']=63
    with pytest.raises(ValueError):validate_manifest(value)


@pytest.mark.parametrize('change',['none','seed','sampling','output'])
def test_actual_request_plan_audit_checks_seed_and_unchanged_sampling(tmp_path,change):
    save(tmp_path/'manifest.json',dict(tasks=[dict(task_id='fixed')]))
    plan=bridge.backend_request(model='m',tokens=[1],session='s',max_tokens=16384,
                                plan={},backend_kind='kvmem',seed=1002)
    if change=='seed':plan['seed']=1000
    elif change=='sampling':plan['temperature']=0
    elif change=='output':plan['max_tokens']=16385
    save(tmp_path/'bridge/fixed/00001/plan.json',plan)
    if change=='none':assert audit_seeds(tmp_path,1002)['model_requests']==1
    else:
        with pytest.raises(ValueError):audit_seeds(tmp_path,1002)


def test_all_four_attempts_remain_in_denominator_with_failures_and_unscored_limits(tmp_path):
    plan=manifest();save(tmp_path/'manifest.json',plan)
    for seed in SEEDS:
        tasks=[dict(task=task,state='completed',reward=int(i%2==0)) for i,task in enumerate(plan['task_ids'])]
        tasks[-1].update(state='context_ceiling',reward=None)
        save(tmp_path/f'seed-{seed}/scores.json',dict(status='completed',expected=16,
            scored=15,solved=8,context_ceiling=1,tasks=tasks))
    result=summarize(tmp_path)
    assert result['classified_attempts']==64 and result['scored']==60 and result['passed']==32
    assert all(t['attempts']==4 for t in result['tasks'])
    assert result['tasks'][1]['passed']==0 and result['tasks'][-1]['scored']==0


def test_partial_scores_do_not_claim_complete_suite(tmp_path):
    save(tmp_path/'manifest.json',manifest())
    result=summarize(tmp_path)
    assert result['status']=='running' and result['classified_attempts']==0


def test_duplicate_result_is_rejected(tmp_path):
    plan=manifest();save(tmp_path/'manifest.json',plan)
    tasks=[dict(task=plan['task_ids'][0],state='completed',reward=1)]*16
    save(tmp_path/'seed-1001/scores.json',dict(status='completed',expected=16,scored=16,solved=16,
         context_ceiling=0,tasks=tasks))
    with pytest.raises(ValueError,match='duplicated'):summarize(tmp_path)


@pytest.mark.parametrize('key',['prefill_batch_tokens','prefill_request_tokens','adaptive_prefill'])
def test_user_requested_prefill_throttling_off(key):
    from scripts.queue_deepswe_seed_suite import validate_unlimited_prefill
    validate_unlimited_prefill({'kvmem':{}})
    validate_unlimited_prefill({'kvmem':{key:0}})
    with pytest.raises(ValueError,match='throttling off'):
        validate_unlimited_prefill({'kvmem':{key:256}})


@pytest.mark.parametrize('failure',[False,True])
def test_formal_failure_preserves_work_before_releasing_supervisors(tmp_path,monkeypatch,failure):
    from scripts import benchmark_deepswe_first16 as run
    from scripts import deepswe_window_snapshot as fs
    calls=[];state={'status':'needs_attention','tasks':{'done':{'reward':1}}}
    original=copy.deepcopy(state['tasks'])
    def snapshot(out,children):
        calls.append('preserve')
        assert out==tmp_path and children==['owned']
        if failure:raise RuntimeError('snapshot incomplete')
        return {'status':'preserved'}
    monkeypatch.setattr(fs,'snapshot',snapshot)
    monkeypatch.setattr(fs,'release_supervisors',lambda r:calls.append('release'))
    run.preserve_incomplete_suite(tmp_path,['owned'],{'seed_suite_qualification':True},state)
    assert calls==(['preserve'] if failure else ['preserve','release'])
    assert bool(state.get('window_snapshot_error'))==failure and state['tasks']==original
    if not failure:assert json.loads((tmp_path/'failure-workspace-preservation.json').read_text())['scoring_unchanged']


def test_completed_suite_is_never_resnapshotted(tmp_path,monkeypatch):
    from scripts import benchmark_deepswe_first16 as run
    from scripts import deepswe_window_snapshot as fs
    monkeypatch.setattr(fs,'snapshot',lambda *a:pytest.fail('completed evidence touched'))
    run.preserve_incomplete_suite(tmp_path,[],{'seed_suite_qualification':True},{'status':'completed'})


def test_dashboard_has_all64_tasks_and_only_active_round_metrics(tmp_path):
    from scripts.deepswe_seed_dashboard import SeedDashboard, page
    save(tmp_path/'manifest.json',manifest())
    save(tmp_path/'status.json',dict(status='running',active_seed=1002))
    for seed in SEEDS:
        d=tmp_path/f'seed-{seed}/first16'
        save(d/'manifest.json',dict(tasks=[dict(task_id=x) for x in manifest()['task_ids']]))
    value=SeedDashboard(tmp_path).snapshot()
    assert value['planned_tasks']==64 and len(value['tasks'])==64
    assert len({x['id'] for x in value['tasks']})==64
    assert value['seed_suite']['active_seed']==1002 and value['completed_tasks']==0
    assert '64 次测试'.encode() in page()
