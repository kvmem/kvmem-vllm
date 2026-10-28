import io
import tarfile
import pytest
from scripts.recover_terminalbench_verifier import app_identity, validate_source


def original():
    return dict(agent_execution={'finished_at':'2026-10-26T23:36:24Z'},
                exception_info={'exception_type':'VerifierTimeoutError'},verifier_result=None)


@pytest.mark.parametrize('case', ['pass', 'fail', 'download_failure', 'incomplete', 'mismatch'])
def test_verifier_reward_requires_actual_original_tests(tmp_path, case):
    import json
    from scripts.recover_terminalbench_verifier import verifier_reward, expected_verifier_tests
    expected = expected_verifier_tests('''
def test_pipeline_parallel_exists(): pass
def test_no_hooks_in_pipeline_parallel(): pass
@pytest.mark.parametrize("world_size", [1, 2])
def test_pipeline_parallel(world_size): pass
''')
    assert len(expected) == 4
    reward = 1 if case == 'pass' else 0
    (tmp_path/'reward.txt').write_text(str(reward))
    tests = [{'name':name, 'status':'failed' if case == 'fail' and i == 0 else 'passed'}
             for i, name in enumerate(sorted(expected))]
    if case == 'incomplete': tests.pop()
    if case != 'download_failure':
        (tmp_path/'ctrf.json').write_text(json.dumps({'results':{'summary':{'tests':len(tests)},'tests':tests}}))
    if case in ('pass', 'fail'):
        assert verifier_reward(tmp_path, expected) == reward
    else:
        with pytest.raises((ValueError, FileNotFoundError)): verifier_reward(tmp_path, expected)


@pytest.mark.parametrize('case', ['complete', 'missing_variant', 'duplicate_variant', 'wrong_reward', 'wrong_ctrf'])
def test_ctrf_collapsed_parameters_need_each_original_pytest_outcome(tmp_path, case):
    import json
    from scripts.recover_terminalbench_verifier import verifier_reward
    names=['test_outputs.py::test_static', 'test_outputs.py::test_param[1]', 'test_outputs.py::test_param[2]']
    lines=['PASSED ../tests/'+names[0], 'FAILED ../tests/'+names[1], 'FAILED ../tests/'+names[2]]
    if case=='missing_variant':lines.pop()
    if case=='duplicate_variant':lines[-1]=lines[1]
    (tmp_path/'test-stdout.txt').write_text('\n'.join(lines)+'\n')
    (tmp_path/'reward.txt').write_text('1' if case=='wrong_reward' else '0')
    tests=[{'name':names[0],'status':'passed'}, {'name':'test_outputs.py::test_param','status':'passed' if case=='wrong_ctrf' else 'failed'}]
    (tmp_path/'ctrf.json').write_text(json.dumps({'results':{'summary':{'tests':2},'tests':tests}}))
    if case=='complete':assert verifier_reward(tmp_path,names)==0
    else:
        with pytest.raises(ValueError):verifier_reward(tmp_path,names)


def test_only_finished_unscored_timeout_is_eligible():
    validate_source(original(),['test-stdout.txt'])


@pytest.mark.parametrize('change',['unfinished','scored','other_error','reward.txt','reward.json','ctrf.json'])
def test_do_not_rerun_scored_or_unfinished_answer(change):
    r=original();files=[]
    if change=='unfinished':r['agent_execution']={}
    elif change=='scored':r['verifier_result']={'rewards':{'reward':0}}
    elif change=='other_error':r['exception_info']['exception_type']='CancelledError'
    else:files=[change]
    with pytest.raises(ValueError):validate_source(r,files)


def test_answer_identity_compares_bytes_modes_and_links_not_tar_order(tmp_path):
    def archive(name,reverse=False,data=b'original',mode=0o644,target='answer.py'):
        p=tmp_path/name
        regular=tarfile.TarInfo('app/answer.py');regular.size=len(data);regular.mode=mode
        link=tarfile.TarInfo('app/link');link.type=tarfile.SYMTYPE;link.linkname=target
        members=[(regular,io.BytesIO(data)),(link,None)]
        with tarfile.open(p,'w') as tar:
            for m,b in reversed(members) if reverse else members:tar.addfile(m,b)
        return p
    before=app_identity(archive('old.tar'))
    assert before==app_identity(archive('reordered.tar',reverse=True))
    assert before!=app_identity(archive('changed.tar',data=b'new answer'))
    assert before!=app_identity(archive('mode.tar',mode=0o755))
    assert before!=app_identity(archive('link.tar',target='/other'))


@pytest.mark.parametrize('corruption',[None,'source','trial','agent','cleanup','original','reward'])
def test_select_recovered_verifier_only_when_exact_answer_and_evidence_match(tmp_path,corruption):
    import json,hashlib
    from scripts.restart_terminalbench_remaining import verifier_recovered_outcomes
    source=tmp_path/'source';recovery=tmp_path/'recovery'
    source_result=source/'jobs/formal/trial/result.json';source_result.parent.mkdir(parents=True)
    recovery.mkdir()
    def save(p,d):p.write_text(json.dumps(d))
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    before={**original(),'trial_name':'trial','agent_result':{'answer':'unchanged'},'config':{'seed':1000}}
    save(source_result,before)
    value={**before,'exception_info':None,'verifier_result':{'rewards':{'reward':0}},'verifier_recovery':{}}
    manifest=dict(source=str(source),trial='trial',task='task',agent_launch_allowed=False)
    audit=dict(passed=True,answer_unchanged=True,no_model_requests=True,reward=0,source_result_sha256=digest(source_result))
    cleanup=dict(remaining_processes=[],remaining_owned_gpu_pids=[])
    if corruption=='source':manifest['source']='other'
    if corruption=='trial':manifest['trial']='other'
    if corruption=='agent':value['agent_result']={'answer':'regenerated'}
    if corruption=='cleanup':cleanup['remaining_processes']=[12]
    if corruption=='original':audit['source_result_sha256']='0'*64
    if corruption=='reward':audit['reward']=1
    for n,d in [('result.json',value),('manifest.json',manifest),('execution-audit.json',audit),('cleanup.json',cleanup)]:save(recovery/n,d)
    save(recovery/'evidence-hashes.json',{p.name:digest(p) for p in recovery.iterdir()})
    row=dict(task='terminal-bench/task',trial='trial',status='error',exception='VerifierTimeoutError',rewards=None,passed=False)
    if corruption:
        with pytest.raises(ValueError):verifier_recovered_outcomes(source,[row],{'task':recovery})
    else:
        result=verifier_recovered_outcomes(source,[row],{'task':recovery})[0]
        assert result['status']=='scored' and result['rewards']=={'reward':0} and not result['passed']
        assert result['original_result_sha256']==digest(source_result)
