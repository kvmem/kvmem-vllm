import json
from pathlib import Path
import pytest
from scripts.restart_terminalbench_remaining import partition, review_selected, verify_history
from scripts.finalize_terminalbench import sha


def test_restart_only_interrupted_and_unstarted():
    tasks=[{'name':x} for x in ('pass','fail','crash','new')]
    rows=[dict(task='terminal-bench/'+x,status='scored',passed=x=='pass',rewards={'reward':int(x=='pass')}) for x in ('pass','fail')]
    rows.append(dict(task='terminal-bench/crash',status='error',exception='CancelledError'))
    old,new,failed=partition(tasks,rows)
    assert len(old)==2 and new==['crash','new'] and failed==['crash']
    assert old[1]['passed'] is False
    with pytest.raises(ValueError):partition(tasks,rows+[rows[0]])
    rows[-1]['exception']='AgentTimeoutError'
    with pytest.raises(ValueError):partition(tasks,rows)


def test_merge_requires_intact_original_evidence(tmp_path):
    prior=tmp_path/'prior';prior.mkdir();(prior/'proof').write_text('original')
    def save(n,v):(tmp_path/n).write_text(json.dumps(v))
    save('original-evidence-hashes.json',dict(source=str(prior),hashes={'proof':sha(prior/'proof')}))
    names=[f'task-{i}' for i in range(89)]
    rows=[dict(task=n,status='scored',passed=False,rewards={'reward':0},source_root=str(prior)) for n in names]
    save('preserved-outcomes.json',rows[:10]);save('results.json',{'tasks':rows[10:]})
    save('tasks.json',[{'name':n} for n in names])
    review_selected(tmp_path)
    result=json.loads((tmp_path/'selected-results.json').read_text())
    assert result['scored']==89 and result['passed']==0 and result['new_attempts']==79
    (prior/'proof').write_text('changed')
    with pytest.raises(ValueError,match='Original evidence changed'):review_selected(tmp_path)


def test_explicit_repair_allowlist_keeps_all_scores_and_refuses_other_errors():
    tasks=[{'name':x} for x in ('old','new-score','probe','nul','waiting')]
    rows=[dict(task=x,status='scored',passed=False,rewards={'reward':0}) for x in ('old','new-score')]
    rows += [dict(task='probe',status='error',exception='RuntimeError'),
             dict(task='nul',status='error',exception='ValueError')]
    approved={'probe':'RuntimeError','nul':'ValueError'}
    saved,launched,interrupted=partition(tasks,rows,approved)
    assert [r['task'] for r in saved]==['old','new-score']
    assert launched==['nul','probe','waiting'] and interrupted==['nul','probe']
    with pytest.raises(ValueError):partition(tasks,rows,{'probe':'RuntimeError'})
    rows[-1]['exception']='VerifierTimeoutError'
    with pytest.raises(ValueError):partition(tasks,rows,approved)


def test_recursive_history_rejects_modified_ancestor_and_cycle(tmp_path):
    oldest=tmp_path/'oldest';oldest.mkdir();(oldest/'answer').write_text('unchanged')
    middle=tmp_path/'middle';middle.mkdir()
    proof=middle/'original-evidence-hashes.json'
    proof.write_text(json.dumps(dict(source=str(oldest),hashes={'answer':sha(oldest/'answer')})))
    latest=tmp_path/'latest';latest.mkdir()
    (latest/'original-evidence-hashes.json').write_text(json.dumps(dict(source=str(middle),
        hashes={'original-evidence-hashes.json':sha(proof)})))
    assert verify_history(latest)==2
    (oldest/'answer').write_text('changed')
    with pytest.raises(ValueError,match='Original evidence changed'):verify_history(latest)
    (latest/'original-evidence-hashes.json').write_text(json.dumps(dict(source=str(latest),hashes={})))
    with pytest.raises(ValueError,match='Cyclic'):verify_history(latest)


def test_diagnostic_execution_mode_is_frozen_before_start(tmp_path,monkeypatch):
    from scripts import restart_terminalbench_remaining as restart
    mode=tmp_path/'execution-mode.json';mode.write_text(json.dumps({'cuda_launch_blocking':True}))
    (tmp_path/'manifest.json').write_text(json.dumps({'files':{'execution-mode.json':sha(mode)}}))
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING','0')
    called=[]
    monkeypatch.setattr(restart.suite,'run',lambda root:called.append(restart.os.environ['CUDA_LAUNCH_BLOCKING']))
    monkeypatch.setattr(restart,'review_selected',lambda root:None)
    restart.run(tmp_path)
    assert called==['1']
    mode.write_text(json.dumps({'cuda_launch_blocking':False}))
    with pytest.raises(ValueError,match='Execution mode changed'):restart.run(tmp_path)
    assert called==['1']


def test_missing_trial_requires_audited_interruption_and_preserves_unindexed_score(tmp_path):
    from scripts.restart_terminalbench_remaining import source_outcomes
    events=tmp_path/'formal-events';events.mkdir()
    trial=tmp_path/'jobs/formal/task__id';(trial/'agent').mkdir(parents=True)
    (trial/'verifier').mkdir()
    (trial/'config.json').write_text(json.dumps({'task':{'path':'/tasks/task'}}))
    (events/'status.json').write_text(json.dumps(dict(outcomes=[],active=['task__id'])))
    (events/'events.jsonl').write_text(json.dumps(dict(trial='task__id',event='agent-start'))+'\n')
    evidence=tmp_path/'incident.json';evidence.write_text('backend failed; snapshot retained')
    proof={'task__id':dict(task='terminal-bench/task',status='error',exception='ControllerInterrupted',
        reason='Bound backend exited',evidence={str(evidence):sha(evidence)})}
    with pytest.raises(ValueError,match='classification'):source_outcomes(tmp_path)
    rows=source_outcomes(tmp_path,proof)
    assert rows[0]['exception']=='ControllerInterrupted'
    proof['task__id']['status']='not_started'
    with pytest.raises(ValueError,match='Executed'):source_outcomes(tmp_path,proof)
    (events/'events.jsonl').write_text(json.dumps(dict(trial='task__id',event='waiting_for_memory'))+'\n')
    assert source_outcomes(tmp_path,proof)==[]
    (trial/'verifier/reward.txt').write_text('0')
    with pytest.raises(ValueError,match='Unindexed result'):source_outcomes(tmp_path,proof)
    (trial/'verifier/reward.txt').unlink();evidence.write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):source_outcomes(tmp_path,proof)
