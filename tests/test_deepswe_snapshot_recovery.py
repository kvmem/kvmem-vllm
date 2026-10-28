import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import recover_deepswe_seed_snapshot as r

SOURCE=Path('/data/chaidi/kvmem-vllm-experiments/deepswe-seed4-96k-10m-20261025-r3')

def cell(tmp_path):
    p=tmp_path/r.TASKS[0];p.mkdir();(p/'prior-stream.jsonl').write_text('old\n')
    (p/'recovery-prompt.txt').write_text(r.PROMPT+'\n')
    r.save(p/'recovery.json',dict(task_id=p.name,seed=1004,prompt=r.PROMPT,prior_stream_sha256=r.sha(p/'prior-stream.jsonl')))
    return p,dict(tasks=[dict(task_id=p.name)])

def test_claim_is_exclusive_and_preserves_original_stream(tmp_path):
    p,m=cell(tmp_path);r.claim_cell(p,m)
    with pytest.raises(FileExistsError):r.claim_cell(p,m)
    assert (p/'prior-stream.jsonl').read_text()=='old\n'

@pytest.mark.parametrize('change',['seed','prompt','task_id','stream','prompt_file','allowlist'])
def test_recovery_rejects_changed_input(tmp_path,change):
    p,m=cell(tmp_path);proof=r.read(p/'recovery.json')
    if change in proof:proof[change]='wrong';r.save(p/'recovery.json',proof)
    elif change=='stream':(p/'prior-stream.jsonl').write_text('different')
    elif change=='prompt_file':(p/'recovery-prompt.txt').write_text('better answer')
    else:m['tasks']=[]
    with pytest.raises(ValueError):r.claim_cell(p,m)
    assert not (p/'recovery-started.json').exists()

def test_guards_keep_prior_history_and_reject_prior_termination(tmp_path):
    p,_=cell(tmp_path)
    class Guard:
        def __init__(self):self.lines=[];self.first_trigger=None
        def line(self,v):self.lines.append(v)
    a,b=Guard(),Guard();r.prime_guards(p,a,b)
    assert a.lines==b.lines==['old\n']
    a.first_trigger={'streak':32}
    with pytest.raises(ValueError):r.prime_guards(p,a,b)

def test_frozen_transforms_do_not_change_verifier_or_model(tmp_path):
    root=SOURCE/'seed-1004/first16/setup'
    if not root.exists():pytest.skip('Frozen experiment fixture unavailable')
    agent=r.patch_agent((root/'container_claude_agent.sh').read_text())
    assert 'git checkout -B' not in agent and '--resume "$recovery_session"' in agent
    assert 'git diff --binary "$BASE_COMMIT" HEAD' in agent
    cell=(root/'run_deepswe_cell.sh').read_text();patched=r.patch_cell(cell)
    assert patched[patched.index('base_commit=${metadata[1]}'):]==cell[cell.index('base_commit=${metadata[1]}'):]
    controller=r.patch_controller((SOURCE/'runtime/scripts/benchmark_deepswe_first16.py').read_text())
    compile(controller,'controller','exec')
    assert "len(state['tasks'])==len(m['tasks'])" in controller
    assert "--seed',str" in controller
    for patch,source in [(r.patch_agent,agent),(r.patch_cell,patched),(r.patch_controller,controller)]:
        with pytest.raises(ValueError):patch(source)

def recovery_fixture(tmp_path):
    source=tmp_path/'original';root=tmp_path/'new'
    for p in (source,root):
        r.save(p/'manifest.json',dict(seeds=[1001,1002,1003,1004],expected_attempts=64,profile={'kvmem':{}},source=str(source),recovery=p==root))
    r.save(source/'status.json',dict(status='failed',error='old memory guard'))
    r.save(root/'status.json',dict(status='running',active_seed=1004))
    for seed in (1001,1002,1003,1004):
        c=source/f'seed-{seed}'
        r.save(c/'first16/manifest.json',dict(tasks=[dict(task_id=str(i),title=str(i)) for i in range(16)]))
        r.save(c/'first16/run.json',dict(status='completed' if seed!=1004 else 'needs_attention',
            started_at=1,model_ready_at=1,ended_at=2,tasks={str(i):dict(state='completed' if seed!=1004 or i<11 else 'infrastructure_error') for i in range(16)}))
        (root/f'seed-{seed}').symlink_to(c,target_is_directory=True)
    d=root/'recovery-1004/first16'
    r.save(d/'manifest.json',dict(tasks=[dict(task_id=t,title=t) for t in r.TASKS]))
    r.save(d/'run.json',dict(status='running',started_at=10,model_ready_at=10,tasks={t:dict(state='running',started_at=10) for t in r.TASKS}))
    return source,root

def fresh_module(name,path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_monitor_recovery_does_not_hide_new_failure_or_claim_original64_passed(tmp_path):
    from scripts.deepswe_recovery_view import install_monitor
    _,root=recovery_fixture(tmp_path)
    m=fresh_module('recovery_monitor_test',Path('scripts/monitor_deepswe_seed_suite.py'));install_monitor(m)
    kwargs=dict(probe=lambda _:dict(state='alive'),resource_probe=lambda _:dict(gpu=[]),now=20)
    value=m.inspect(root,**kwargs)
    assert value['recovery']['counts']['running']==5
    assert value['original_outcomes']['scored']==59 and not value['complete']
    assert not value['issues'] and not value['all_classified_and_audited']
    r.save(root/'status.json',dict(status='failed',error='new failure'))
    value=m.inspect(root,**kwargs)
    assert any('new failure' in x for x in value['issues'])
    r.save(root/'status.json',dict(status='needs_final_review'))
    assert not m.inspect(root,**kwargs)['complete']

def test_dashboard_keeps_64_original_rows_and_five_separate_recoveries(tmp_path):
    from scripts.deepswe_recovery_view import install_dashboard
    _,root=recovery_fixture(tmp_path)
    m=fresh_module('recovery_dashboard_test',Path('scripts/deepswe_seed_dashboard.py'));install_dashboard(m)
    value=m.SeedDashboard(root).snapshot()
    assert value['planned_tasks']==64 and len(value['tasks'])==69
    assert sum(bool(t.get('recovery')) for t in value['tasks'])==5
    assert value['infrastructure_errors']==5
    assert value['recovery']['expected']==5
    assert '恢复阶段'.encode() in m.page()
