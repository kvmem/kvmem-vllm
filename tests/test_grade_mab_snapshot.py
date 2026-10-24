from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import grade_mab_snapshot as snapshot
import regrade_flash as f


def test_snapshot_is_fixed_and_checks_manifest_reference_and_answers(tmp_path, monkeypatch):
    campaign = tmp_path/'campaign'; evaluation = tmp_path/'eval'; reference = tmp_path/'reference'
    rows = [dict(id=str(i),token_sha256='tokens'+str(i),input_tokens=100+i) for i in range(1116)]
    f.save(campaign/'fixture.json',dict(suite='memoryagentbench',cases=1116,rows=rows))
    f.save(campaign/'profile.json',dict(kvmem=dict(graph_trace=False)))
    f.save(campaign/'prepared-proof.json',{})
    f.save(campaign/'run/answers/manifest.json',dict(reference=str(reference)))
    f.write_lines(reference/'results/attention/memoryagentbench/special_judgments.jsonl',[])
    for row in rows[:2]:
        f.save(campaign/'run/answers/cases'/row['id']/'result.json',dict(row,
            suite='memoryagentbench',status='completed',original_input_intact=True,
            prompt_tokens=row['input_tokens'],answer='answer',raw_completion='answer',
            graph_v2=dict(quota=dict(mean_fraction=.5),cpu_fps_audit=dict(passed=True),native_replay_audit=None)))
    repo = tmp_path/'repo'; (repo/'scripts').mkdir(parents=True)
    (repo/'scripts/driver.py').write_text('frozen')
    monkeypatch.setattr(snapshot,'REPO',repo)
    for name in snapshot.EVALUATORS:
        p=evaluation/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('rubric')
    output=tmp_path/'grade'
    assert snapshot.freeze(campaign,output,evaluation) == 2
    assert snapshot.verify(output)['case_ids'] == ['0','1']
    f.save(campaign/'run/answers/cases/2/result.json',dict(status='completed'))
    assert snapshot.verify(output)['cases'] == 2
    with pytest.raises(ValueError,match='fresh'): snapshot.freeze(campaign,output,evaluation)
    original=(output/'snapshot/manifest.json').read_bytes()
    (output/'snapshot/manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest changed'): snapshot.verify(output)
    (output/'snapshot/manifest.json').write_bytes(original)
    source=campaign/'run/answers/cases/0/result.json'
    source.write_text(source.read_text()+' ')
    with pytest.raises(AssertionError,match='Source answer changed'): snapshot.verify(output)


@pytest.mark.parametrize('count',[0,-1,True,1117])
def test_invalid_case_count_makes_no_judge_calls(tmp_path,count):
    with pytest.raises(ValueError,match='1..1116'):
        f.mab_arm(tmp_path,{},'hybrid',tmp_path,None,case_count=count)


@pytest.mark.parametrize('bad_special',[False,True])
def test_partial_mab_preserves_category_macro_and_special_coverage(tmp_path,monkeypatch,bad_special):
    answers=tmp_path/'answers'; output=tmp_path/'grade/mab/hybrid'; reference=tmp_path/'reference'
    sources=['longmemeval_s','longmemeval_s','factconsolidation_x','infbench_sum']
    categories=['Accurate_Retrieval']*2+['Conflict_Resolution','Long_Range_Understanding']
    rows=[dict(id=str(i),answer='a',raw_completion='a',source=sources[i],split=categories[i],
        official_local_metrics=dict(substring_exact_match=1)) for i in range(4)]
    for row in rows: f.save(answers/'cases'/row['id']/'result.json',row)
    f.save(answers/'manifest.json',dict(reference=str(reference)))
    f.write_lines(reference/'results/attention/memoryagentbench/special_judgments.jsonl',[])
    f.save(output/'official_local_summary.json',{})
    f.write_lines(output/'rows/selected/results.jsonl',rows)
    special=[dict(key='0',score=1),dict(key='1',score=0),dict(key='3',score=.25)]
    if bad_special: special.pop()
    f.write_lines(output/'special_judgments.jsonl',special)
    monkeypatch.setattr(f,'load',lambda *args:NS(key=lambda row:row['id'],main=lambda:0))
    monkeypatch.setenv('DEEPSEEK_API_KEY','transport-managed')
    flash=NS(returned_model=f.MODEL)
    if bad_special:
        with pytest.raises(AssertionError):
            f.mab_arm(tmp_path/'grade',dict(evaluation=str(tmp_path)), 'hybrid',answers,flash,case_count=4)
        return
    result=f.mab_arm(tmp_path/'grade',dict(evaluation=str(tmp_path)), 'hybrid',answers,flash,case_count=4)
    assert result['partial'] and not result['complete'] and result['count']==4
    assert result['full_cohort_cases']==1116 and result['three_category_macro']==pytest.approx((.5+1+.25)/3)
    assert result['by_category_counts']==dict(Accurate_Retrieval=2,Conflict_Resolution=1,Long_Range_Understanding=1)


def test_prefetch_preserves_official_prompt_settings_and_skips_local_cases(tmp_path,monkeypatch):
    rows={str(i):dict(source='longmemeval_s' if i<2 else 'ruler_qa2',question_type='knowledge-update',
        raw_question='question',gold_answer='gold',answer=str(i),qa_pair_id='id_abs' if i==1 else 'id')
        for i in range(3)}
    calls=[]
    monkeypatch.setattr(f,'complete_mab_call',lambda flash,body:calls.append(body))
    module=NS(longmem_prompt=lambda *args:repr(args))
    f.prefetch_mab_longmem(tmp_path,rows,module,None,workers=4)
    assert len(calls)==2
    for body in calls:
        assert body['model']==f.MODEL and body['max_tokens']==1024
        assert body['thinking']==dict(type='enabled') and body['temperature']==0 and body['stream'] is False
    assert {b['messages'][0]['content'] for b in calls}=={
        repr(('knowledge-update','question','gold','0',False)),repr(('knowledge-update','question','gold','1',True))}
    assert f.read(tmp_path/'mab-prefetch-progress.json')==dict(completed=2,expected=2)
