import json

import pytest

from scripts.analyze_agent_restore_scheduling import analyze


def fixture(tmp_path, *, skip=False, reversed_ack=False, private=False):
    def write(name, rows):
        (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write('public_work.json', [dict(created_wall_time=100, created_at=0)])
    write('checkpoint_timing.jsonl', [])
    write('restore_admission.jsonl', [dict(request_id='r', stage=s, at=t)
        for s,t in [('arrival',1),('admitted',2),('ready',8 if reversed_ack else 6)]])
    write('async_refresh.jsonl', [dict(request_id='r',stage=s,at=t)
        for s,t in [('restore_suspended',3),('restore_submitted',4),('restore_ready',5)]]
        + ([dict(request_id='r',stage='suspended',at=6.5)] if private else []))
    write('async_restore.jsonl', [dict(request_id='r',started=3,finished=5,cancelled=False,
        resident_active_tokens=32,history_tokens=320,uploaded_bytes=0,model_uploaded_bytes=0)])
    steps=[]
    for index,(rid,begin,model) in enumerate([('peer',3.1,3.2),('r',7,7.2)],1):
        steps.append(dict(sequence=index,call_index=index+(skip and index==2),
            previous_call_index=index-1 if index>1 else None,
            started_at=begin,model_submitted_at=model,decode_requests=1,
            prefill_requests=0,prefill_tokens=0,model_ms=10,
            previous_end_to_model_ms=20 if index>1 else None,
            maintenance_host_ms=(model-begin)*1000,batch_geometry=[dict(request_id=rid)]))
    write('model_step_latency.jsonl',steps)
    return tmp_path


def test_milestones_partition_request_and_do_not_claim_idle(tmp_path):
    report=analyze(fixture(tmp_path),common_end_wall=110)
    row=report['requests'][0]
    assert row['peer_batches']==1
    assert row['phase_ms']['total_admission_arrival_to_model_submit']==pytest.approx(6200)
    assert sum(v for k,v in row['phase_ms'].items() if not k.startswith('total_'))==pytest.approx(6200)
    assert 'neither' in report['scope'].lower()
    assert report['public_steps']['all']['preceding_nonpublic_model_interval_seconds']==.02


def test_private_reselection_is_not_called_ready_queue_delay(tmp_path):
    report=analyze(fixture(tmp_path,private=True),common_end_wall=110)
    assert report['requests'][0]['post_restore_private_refresh']
    assert report['summaries']['resident_common_eight_without_post_restore_refresh']['count']==0


@pytest.mark.parametrize('kwargs',[{'skip':True},{'reversed_ack':True}])
def test_reject_incomplete_or_misordered_timeline(tmp_path,kwargs):
    with pytest.raises(ValueError):
        analyze(fixture(tmp_path,**kwargs))
