import json

import pytest

from scripts.benchmark_restore_handoff_fastpath import completion_audit, profiles


def test_profiles_change_only_fastpath_and_qualification_oracle():
    original = {'port':18878,'verify_session_restore':False,'kvmem':{'select_budget':36864}}
    before,after,qual=profiles(original)
    assert before == original and 'restore_handoff_fastpath' not in original['kvmem']
    assert after['kvmem'].pop('restore_handoff_fastpath') is True
    assert after == before
    assert qual['port']==18877 and qual['verify_session_restore'] is True
    assert qual['kvmem']['restore_handoff_fastpath'] is True
    with pytest.raises(ValueError,match='Control'):
        profiles(qual)


def test_audit_rejects_wrong_count_and_disabled_post_forward(tmp_path):
    path=tmp_path/'run/server/capture/async_restore.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'completion_poll':'post_forward'})+'\n')
    with pytest.raises(ValueError):
        completion_audit(tmp_path,True)
    path.write_text(path.read_text()*56)
    with pytest.raises(ValueError):
        completion_audit(tmp_path,False)
    assert completion_audit(tmp_path,True)['post_forward']==56
