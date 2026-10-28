from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmark_niah import save,sha
from recover_hybrid_mab import inherit_cases,validate_engine_change,SCORER
import run_hybrid_memory_campaign as campaign


def test_only_qualified_scorer_can_change():
    old = {SCORER:'old','src/vllm_kvmem/authority.py':'unchanged'}
    new = {**old,SCORER:'new'}
    report = dict(exact_scores=True,source_sha256='new',reference_sha256='old',results=[
        dict(speedup=4.,measurements=[dict(arm='new',exact_scores=True,peak_scratch_bytes=20,
            stats=dict(scoring_scratch_bound_bytes=30,scoring_scratch_limit_bytes=40))])])
    assert set(validate_engine_change(old,new,report)) == {SCORER}
    with pytest.raises(ValueError,match='engine changes'):
        validate_engine_change(old,{**new,'src/vllm_kvmem/authority.py':'changed'},report)
    bad = deepcopy(report);bad['results'][0]['measurements'][0]['peak_scratch_bytes'] = 41
    with pytest.raises(ValueError,match='scratch'):validate_engine_change(old,new,bad)
    bad = deepcopy(report);bad['source_sha256'] = 'unqualified'
    with pytest.raises(ValueError,match='exact-score'):validate_engine_change(old,new,bad)


def make_case(root,ident,completed):
    row = dict(id=ident,token_sha256='original-token-hash',input_tokens=256000)
    directory = root/'run/answers/cases'/ident
    save(directory/'attempt.json',dict(seed=123))
    if completed:
        save(directory/'result.json',dict(id=ident,status='completed',original_input_intact=True,
            token_sha256=row['token_sha256'],prompt_tokens=row['input_tokens'],answer='wrong answer',score=0))
        for name in ('response.json','request.json','terminal.json','events.json'):
            save(directory/name,dict(original=name,seed=123))
    return row


def test_mab_resume_copies_every_success_including_wrong_answers_and_keeps_attempts(tmp_path):
    source = tmp_path/'source'
    save(source/'run/answers/manifest.json',dict(original=True))
    rows = [make_case(source,'g_q0',True),make_case(source,'g_q1',False)]
    proof = inherit_cases(source,tmp_path/'restored',dict(rows=rows,cases=2))
    assert proof['recovered_ids'] == ['g_q0'] and proof['interrupted_ids'] == ['g_q1']
    assert proof['regenerated_answers'] == 0
    assert not (tmp_path/'restored/cases/g_q1').exists()
    assert (source/'run/answers/cases/g_q1/attempt.json').exists()
    for name in ('result.json','response.json','request.json','terminal.json','events.json'):
        assert sha(source/'run/answers/cases/g_q0'/name) == sha(tmp_path/'restored/cases/g_q0'/name)
    with pytest.raises(ValueError,match='overwrite'):
        inherit_cases(source,tmp_path/'restored',dict(rows=rows,cases=2))


def test_complete_unpublished_response_requires_audit_instead_of_regeneration(tmp_path):
    source = tmp_path/'source';save(source/'run/answers/manifest.json',{})
    row = make_case(source,'g_q0',False)
    save(source/'run/answers/cases/g_q0/response.json',dict(answer='already generated'))
    with pytest.raises(ValueError,match='audit recovery'):
        inherit_cases(source,tmp_path/'restored',dict(rows=[row],cases=1))


def test_mab_resume_requires_explicit_qualified_source():
    fixture = dict(suite='memoryagentbench',qualification=False,reference='/tmp/original',
        recovery_source='/tmp/old',optimization_source='/tmp/old')
    command = campaign.client_command(Path('/tmp/recovery'),fixture)
    assert '--resume' in command and '--case-id' not in command
    fixture['optimization_source'] = '/tmp/different'
    with pytest.raises(ValueError,match='Unsupported recovery'):
        campaign.client_command(Path('/tmp/recovery'),fixture)
