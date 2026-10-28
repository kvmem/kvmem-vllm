import hashlib
import json
import pytest
from scripts.terminalbench_view import State


def setup(tmp_path):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    save('tasks.json',[{'name':'setup'},{'name':'verifier'}])
    original=[dict(task=n,trial=n+'-old',status='error',exception=e,rewards=None,passed=False)
              for n,e in [('setup','RuntimeError'),('verifier','VerifierTimeoutError')]]
    status=save('formal-events/status.json',{'outcomes':original})
    proof=save('proof.json',{'passed':True})
    notes=dict(root=str(tmp_path.resolve()),evidence={str(proof):hashlib.sha256(proof.read_bytes()).hexdigest()},tasks={
        'setup':dict(trial='setup-old',exception='RuntimeError',kind='pending_setup',detail='old setup error'),
        'verifier':dict(trial='verifier-old',exception='VerifierTimeoutError',kind='verifier_review',rewards={'reward':0},detail='original answer reviewed')})
    return save('notes.json',notes),status,original,proof


def test_review_preserves_raw_error_and_never_counts_pending_setup_as_scored(tmp_path):
    notes,status,original,proof=setup(tmp_path);before=status.read_bytes()
    state=State(tmp_path,notes);state.update();v=state.value
    assert v['scored']==1 and v['passed']==0 and v['pending_repairs']==1
    assert v['tasks'][1]['outcome']['status']=='scored'
    assert v['tasks'][1]['original_outcome']['exception']=='VerifierTimeoutError'
    assert v['tasks'][0]['outcome']['status']=='error'
    assert status.read_bytes()==before


@pytest.mark.parametrize('changed',['trial','exception'])
def test_old_note_never_masks_a_new_error(tmp_path,changed):
    notes,status,rows,proof=setup(tmp_path)
    rows[0][changed]='new';status.write_text(json.dumps({'outcomes':rows}))
    state=State(tmp_path,notes);state.update()
    assert state.value['tasks'][0]['review_note'] is None
    assert state.value['pending_repairs']==0


@pytest.mark.parametrize('changed',['root','evidence'])
def test_reject_unbound_review(tmp_path,changed):
    notes,status,rows,proof=setup(tmp_path)
    if changed=='root':
        v=json.loads(notes.read_text());v['root']='other';notes.write_text(json.dumps(v))
    else:proof.write_text('changed')
    with pytest.raises(ValueError):State(tmp_path,notes)
