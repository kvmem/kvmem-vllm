import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import grade_lme_snapshot as snapshot
import regrade_flash as f


def cases(path,answers):
    for i,answer in enumerate(answers):
        f.save(path/str(i)/'result.json',dict(id=str(i),status='completed',answer=answer,
            raw_completion=answer,question_id=str(i),question_type='single-session-user',
            question='original question',gold_answer='gold'))


@pytest.mark.parametrize('count',[3,102])
def test_partial_single_arm_uses_snapshot_denominator_and_no_invented_baseline(tmp_path,count):
    source = tmp_path/'cases'; cases(source,['right','wrong','right']*(count//3))
    official = tmp_path/'official.py'; official.write_text('original official rubric')
    evaluation = tmp_path/'evaluation'
    f.save(evaluation/'evaluators/historical_lme_calibration.json',dict(cases=[dict(
        fixture=dict(question_type='single-session-user',question_id='cal',question='cal',answer='right'),
        hypothesis='right',expected=True)]))
    judge = NS(OFFICIAL=official,PARAMETERS=dict(model=f.MODEL,max_tokens=16,temperature=0),
        prompt_function=lambda:lambda kind,question,gold,answer,abstention:answer,
        verdict=lambda raw:raw['choices'][0]['message']['content']=='yes')
    calls = []
    def request(body):
        calls.append(body)
        return dict(model=f.MODEL,choices=[dict(message=dict(content='yes' if body['messages'][0]['content']=='right' else 'no'),
            finish_reason='stop')]),f.digest(body)
    flash = NS(judge=judge,request=request,returned_model=f.MODEL)
    report = f.lme(tmp_path/'grades',dict(evaluation=str(evaluation),lme={'hybrid':str(source)}),flash,case_count=count)
    if count < 102:
        assert report['partial'] and report['graded_cases'] == count and report['full_cohort_cases'] == 102
        assert report['arms']['hybrid']['by_type_counts'] == {'single-session-user':count}
    else:
        assert report['status'] == 'completed' and not report.get('partial',False)
    assert report['arms']['hybrid']['count'] == count and report['arms']['hybrid']['correct'] == count*2//3
    assert report['arms']['hybrid']['accuracy'] == 2/3 and report['pairs'] == {}
    assert all(call['model'] == f.MODEL and call['max_tokens'] == 16 for call in calls)


@pytest.mark.parametrize('count',[0,-1,True,103])
def test_invalid_snapshot_size_fails_before_judge_calls(tmp_path,count):
    with pytest.raises(ValueError,match='1..102'): f.lme(tmp_path,{},None,case_count=count)


def test_freeze_keeps_all_completed_answers_and_ignores_unfinished_cases(tmp_path,monkeypatch):
    campaign = tmp_path/'campaign'; campaign.mkdir()
    rows = [dict(id=str(i),token_sha256='token'+str(i),input_tokens=100+i) for i in range(102)]
    f.save(campaign/'fixture.json',dict(suite='longmemeval_s',cases=102,rows=rows))
    for name in ('profile.json','prepared-proof.json'): f.save(campaign/name,{})
    for row in rows[:2]:
        result = dict(row,suite='longmemeval_s',status='completed',original_input_intact=True,
            prompt_tokens=row['input_tokens'],answer='saved answer',raw_completion='saved answer',
            graph_v2=dict(native_replay_audit=dict(native_picker_replay_equal=True)))
        f.save(campaign/'run/answers/cases'/row['id']/'result.json',result)
    (campaign/'run/answers/cases/2').mkdir(); f.save(campaign/'run/answers/cases/2/attempt.json',{})
    repository = tmp_path/'repo'; (repository/'scripts').mkdir(parents=True)
    (repository/'scripts/driver.py').write_text('frozen source')
    monkeypatch.setattr(snapshot,'REPO',repository)
    evaluation = tmp_path/'evaluation'
    for name in snapshot.EVALUATORS:
        path = evaluation/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('rubric')
    output = tmp_path/'snapshot'
    assert snapshot.freeze(campaign,output,evaluation) == 2
    plan = snapshot.verify(output)
    assert plan['case_ids'] == ['0','1'] and plan['partial']
    # Completing another case later never changes the frozen denominator.
    f.save(campaign/'run/answers/cases/2/result.json',dict(status='completed'))
    assert snapshot.verify(output)['cases'] == 2
    with pytest.raises(ValueError,match='fresh'): snapshot.freeze(campaign,output,evaluation)
    original = campaign/'run/answers/cases/0/result.json'
    original.write_text(original.read_text()+' ')
    with pytest.raises(AssertionError,match='Source answer changed'): snapshot.verify(output)


def test_full_grader_reuses_only_exact_request_bound_flash_calls(tmp_path,monkeypatch):
    from grade_hybrid_memory import reuse_flash_calls
    source = tmp_path/'partial'; source.mkdir()
    f.save(source/'status.json',dict(status='completed',judge=f.MODEL))
    checked = []
    monkeypatch.setattr(snapshot,'verify',lambda root:checked.append(root))
    body = dict(model=f.MODEL,messages=[dict(role='user',content='original rubric and answer')])
    ident = f.digest(body)
    f.save(source/'calls'/(ident+'.request.json'),body)
    f.save(source/'calls'/(ident+'.response.json'),dict(response=dict(model=f.MODEL,choices=[])))
    output = tmp_path/'full'; output.mkdir()
    assert reuse_flash_calls(source,output) == 1 and checked == [source]
    assert f.sha(output/'calls'/(ident+'.response.json')) == f.sha(source/'calls'/(ident+'.response.json'))
    body['messages'][0]['content'] = 'changed input'
    f.save(source/'calls'/(ident+'.request.json'),body)
    with pytest.raises(ValueError,match='content differs'): reuse_flash_calls(source,tmp_path/'bad')
