import json

import pytest

from scripts.analyze_agent_interference import arm_result, compare


def arm(path, *, changed=False):
    (path/'answers').mkdir(parents=True)
    capture=path/'run/server/capture';capture.mkdir(parents=True)
    for i in range(8):
        start=10 if i<7 else 10.2
        row=dict(owner=i,turn=1,error=None,started=start,finished=11,
            events=[[10.1,1],[10.5,2],[10.9,1]] if i<7 else [[10.6,4]],
            output_ids=[i]*4,input_hash='changed' if changed else str(i))
        (path/'answers'/f'1-{i}.json').write_text(json.dumps(row))
    step=dict(started_at=10.4,prefill_tokens=512,decode_requests=7,
              model_ms=100,maintenance_host_ms=20,previous_end_to_model_ms=None)
    (capture/'model_step_latency.jsonl').write_text(json.dumps(step)+'\n')
    restore=dict(started=10.2,finished=10.3,lane='resident',uploaded_bytes=0)
    (capture/'async_restore.jsonl').write_text(json.dumps(restore)+'\n')


def test_delivery_gaps_and_missing_optional_observers(tmp_path):
    arm(tmp_path)
    value=arm_result(tmp_path)
    assert value['full_seven_peer_rounds']==1
    assert value['rounds'][0]['peer_gap_during_return_ms']['count']==14
    assert value['accepted_output_tokens']==32
    assert value['component_output_tokens_per_second']==32
    assert not value['checkpoint_timing_available']
    assert value['step_groups']['mixed']['previous_end_to_model_ms']['count']==0


def test_compare_multiple_arms_checks_all_trajectories(tmp_path):
    for name in ('a','b','c'):arm(tmp_path/name)
    result=compare(tmp_path,['a','b','c'])
    assert all(x['wall_change_percent']==0 for x in result['arms'].values())
    p=tmp_path/'c/answers/1-0.json';row=json.loads(p.read_text());row['output_ids']=[2]*4
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='trajectory'):compare(tmp_path,['a','b','c'])


def test_missing_peer_or_failed_answer_never_passes(tmp_path):
    arm(tmp_path)
    (tmp_path/'answers/1-0.json').unlink()
    with pytest.raises(ValueError,match='cohort'):arm_result(tmp_path)
