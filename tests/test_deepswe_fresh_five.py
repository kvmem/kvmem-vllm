import copy
import json
from pathlib import Path
import pytest
from scripts import restart_deepswe_five as f
from scripts.recover_deepswe_seed_snapshot import TASKS,read,save
from test_deepswe_snapshot_recovery import SOURCE,recovery_fixture,fresh_module


def test_fresh_controller_keeps_empty_cases_original_prompt_and_no_resume():
    p=SOURCE/'runtime/scripts/benchmark_deepswe_first16.py'
    if not p.exists():pytest.skip('Frozen fixture unavailable')
    text=f.patch_controller(p.read_text());compile(text,'fresh','exec')
    assert 'if tid not in adopted_cells:cell.mkdir(parents=True)' in text
    assert 'prime_guards' not in text and 'claim_cell' not in text
    assert "len(state['tasks'])==len(m['tasks'])" in text
    with pytest.raises(ValueError):f.patch_controller(text)


def test_task_subset_preserves_every_original_parameter():
    p=SOURCE/'seed-1004/first16/manifest.json'
    if not p.exists():pytest.skip('Frozen fixture unavailable')
    original=read(p);task=copy.deepcopy(original);task['tasks']=[t for t in task['tasks'] if t['task_id'] in TASKS]
    f.validate_task_subset(task,original)
    for key in ('sampling','seed','profile','workers','optimized_handoff','termination_policies','tasks','carried_results'):
        changed=copy.deepcopy(task);changed[key]=[] if key!='carried_results' else ['old']
        with pytest.raises(ValueError):f.validate_task_subset(changed,original)


def test_fresh_dashboard_uses_only_five_authorized_replacements(tmp_path):
    from scripts.deepswe_recovery_view import install_dashboard,install_monitor
    source,root=recovery_fixture(tmp_path)
    run=read(source/'seed-1004/first16/run.json')
    # Replace the fixture's five failed task IDs with the actual allowlist.
    rows=list(run['tasks'].items())
    run['tasks']={str(i):v for i,(_,v) in enumerate(rows[:11])}|{t:dict(state='infrastructure_error') for t in TASKS}
    save(source/'seed-1004/first16/run.json',run)
    save(source/'seed-1004/first16/manifest.json',dict(tasks=[dict(task_id=t,title=t) for t in run['tasks']]))
    (root/'recovery-1004').rename(root/'restart-1004')
    m=read(root/'manifest.json');m['recovery']=dict(mode='fresh_restart',campaign='restart-1004',task_ids=list(TASKS),prompt=None);save(root/'manifest.json',m)
    view=fresh_module('fresh_dashboard_test',Path('scripts/deepswe_seed_dashboard.py'));install_dashboard(view)
    data=view.SeedDashboard(root).snapshot()
    assert len(data['tasks'])==64 and data['planned_tasks']==64
    assert len(data['prior_attempts'])==5 and data['infrastructure_errors']==0
    assert {r['task_id'] for r in data['tasks'] if r.get('fresh_restart')}==set(TASKS)
    assert '从原始题目重新测试'.encode() in view.page()
    monitor=fresh_module('fresh_monitor_test',Path('scripts/monitor_deepswe_seed_suite.py'));install_monitor(monitor)
    x=monitor.inspect(root,probe=lambda _:dict(state='alive'),resource_probe=lambda _:dict(gpu=[]))
    assert x['active_arm']=='restart-1004' and x['recovery']['counts']['running']==5
    assert x['selected_outcomes']['preserved_original']==59 and not x['complete']
