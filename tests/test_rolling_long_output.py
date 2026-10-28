"""Qualification must fail closed without >G in-request byte evidence."""
import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def qualification(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('rolling_long_qualification',scripts/'qualify_rolling_long_output.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def evidence():
    rows=[];proofs=[]
    for owner in range(2):
        identity=f'cmpl-owner-{owner}'
        rows.extend([dict(owner=owner,phase='long',id=identity,prompt_tokens=100352,accepted=36864,finish='length'),
            dict(owner=owner,phase='resume',accepted=64),dict(owner=owner,phase='eos',accepted=1,finish='stop'),
            dict(owner=owner,phase='release')])
        for history in (114656,131008):
            proofs.append(dict(request_id=identity+'-0-native',selected_tokens=98304,history_end=history,
                replay_tokens=0,private_forward_steps=0,target_kv_byte_equal=True,recurrent_byte_equal=True,
                cancelled=False,target_kv_bytes=123,mtp_attention_layers=1))
    return rows,proofs


def test_complete_long_output_evidence(qualification):
    assert qualification.audit(*evidence())['passed']


@pytest.mark.parametrize('change',['output','id','replay','kv','state','capacity','early','eos','resume','release'])
def test_incomplete_or_corrupt_evidence_rejected(qualification,change):
    rows,proofs=copy.deepcopy(evidence())
    if change=='output':rows[0]['accepted']=16384
    elif change=='id':proofs[0]['request_id']='foreign-request'
    elif change=='replay':proofs[0]['private_forward_steps']=1
    elif change=='kv':proofs[0]['target_kv_byte_equal']=False
    elif change=='state':proofs[0]['recurrent_byte_equal']=False
    elif change=='capacity':proofs[0]['selected_tokens']=98336
    elif change=='early':proofs[1]['history_end']=114656
    else:rows[:]=[r for r in rows if not(r['owner']==0 and r['phase']==change)]
    with pytest.raises(ValueError):qualification.audit(rows,proofs)
