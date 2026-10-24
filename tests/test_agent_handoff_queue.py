"""A GPU gap, stale completion, read error or foreign process is not permission."""
import json
from pathlib import Path

import pytest

from scripts.run_agent_handoff_queue import dependency_ready, verify_frozen, handoff_metrics


@pytest.mark.parametrize('phase,count,pids,alive,expected', [
    ('running',60,[],False,False), ('complete',59,[],False,False),
    ('complete',60,[123],False,False), ('complete',60,[],True,False),
    ('failed',60,[],False,False), ('partial',60,[],False,False),
    ('complete',60,[],False,True),
])
def test_queue_waits_for_full_predecessor_cleanup(phase,count,pids,alive,expected):
    assert dependency_ready({'dependency_expected':60},
        {'phase':phase,'summary':{'all':{'count':count}}},pids,alive) is expected


def test_frozen_profile_tampering_stops_before_any_gpu_start(tmp_path):
    import hashlib
    path=tmp_path/'profile.json';path.write_text('{"K":36864}')
    (tmp_path/'manifest.json').write_text(json.dumps({'files':{'profile.json':hashlib.sha256(path.read_bytes()).hexdigest()}}))
    assert verify_frozen(tmp_path)['files']
    path.write_text('{"K":32768}')
    with pytest.raises(ValueError,match='changed'):verify_frozen(tmp_path)


def test_metrics_exclude_auxiliary_generation_and_weight_speculative_steps(tmp_path):
    (tmp_path/'bridge').mkdir()
    rows=[dict(owner='agent',metrics={},segment_reports=[dict(metrics=dict(speculative_decoding=dict(
        num_spec_steps=2,num_accepted_draft_tokens=6,num_draft_tokens=8)))]),
        dict(owner='agent',metrics={},segment_reports=[dict(metrics=dict(speculative_decoding=dict(
        num_spec_steps=1,num_accepted_draft_tokens=1,num_draft_tokens=4)))]),
        dict(owner='title',metrics={},segment_reports=[dict(metrics=dict(speculative_decoding=dict(
        num_spec_steps=100,num_accepted_draft_tokens=0,num_draft_tokens=400)))])]
    (tmp_path/'bridge/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=dict(cases={'a':dict(backend=dict(agent_owners=['agent']))})
    result=handoff_metrics(tmp_path,report)
    assert result['mtp_spec_steps']==3
    assert result['mtp_mean_acceptance_length']==1+7/3
    assert result['mtp_draft_acceptance_rate']==7/12
