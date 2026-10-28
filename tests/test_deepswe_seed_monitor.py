import copy
import hashlib
import json
import pytest
from pathlib import Path
from scripts.monitor_deepswe_seed_suite import inspect,findings,SEEDS


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def fixture(root):
    save(root/'manifest.json',dict(seeds=list(SEEDS),expected_attempts=64,profile={'kvmem':{}}))
    save(root/'status.json',dict(status='running',active_seed=1001))
    d=root/'seed-1001/first16'
    save(d/'run.json',dict(status='running',tasks={'a':{'state':'running'}}))
    save(d/'server/capture/public_work.json',dict(counter_id='same',pid=123,
         completed=dict(public_batches=4,public_prefill_tokens=100,public_non_prefill_tokens=200)))
    return d


def check(root,**kwargs):
    return inspect(root,probe=lambda _:dict(state='alive'),resource_probe=lambda _:
        dict(gpu=[],disk_free_bytes=100*2**30,system_disk_free_bytes=100*2**30,memory_available_bytes=100*2**30),**kwargs)


def test_ordinary_progress_is_silent_and_never_kills(tmp_path):
    d=fixture(tmp_path);first=check(tmp_path,now=100)
    assert findings(first,{})==[] and not first['complete']
    work=json.loads((d/'server/capture/public_work.json').read_text());work['completed']['public_batches']+=1
    save(d/'server/capture/public_work.json',work)
    next=check(tmp_path,previous=first,now=1901)
    assert not next['issues'] and findings(next,first)==[]


def test_stalled_progress_requests_inspection_only_once(tmp_path):
    fixture(tmp_path);first=check(tmp_path,now=100)
    next=check(tmp_path,previous=first,now=1901)
    assert any('inspect restore' in x for x in findings(next,first))
    assert findings(check(tmp_path,previous=next,now=3701),next)==[]


def test_task_failure_and_recovery_are_not_silent(tmp_path):
    fixture(tmp_path);first=check(tmp_path)
    save(tmp_path/'status.json',dict(status='failed',error='engine failed'))
    failed=check(tmp_path,previous=first)
    assert any('engine failed' in x for x in findings(failed,first))
    save(tmp_path/'status.json',dict(status='running',active_seed=1001))
    recovered=check(tmp_path,previous=failed)
    assert any('cleared' in x for x in findings(recovered,failed))


def complete(root):
    fixture(root);save(root/'status.json',dict(status='completed'))
    for seed in SEEDS:
        c=root/f'seed-{seed}'
        save(c/'status.json',dict(status='completed'))
        save(c/'first16/run.json',dict(status='completed',tasks={str(i):dict(state='completed') for i in range(16)}))
        save(c/'execution-audit.json',dict(status='passed',classified_tasks=16))
        save(c/'seed-audit.json',dict(status='passed',seed=seed))
        save(c/'scores.json',dict(expected=16,tasks=[{}]*16,scored=16,solved=1))
        save(c/'cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))
        save(c/'evidence-hashes.json',{})
    save(root/'scores.json',dict(classified_attempts=64))
    save(root/'execution-audit.json',dict(status='passed'))
    save(root/'cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[]))


def test_full_results_require_review_before_monitor_auto_exit(tmp_path):
    complete(tmp_path);value=check(tmp_path)
    assert value['all_classified_and_audited'] and not value['complete']
    assert any('All64' in x for x in findings(value,{}))
    report=tmp_path/'report.md';report.write_text('Verified all64 original attempts')
    proof=dict(status='completed',root=str(tmp_path),final_review_complete=True,report=str(report),
               report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    assert check(tmp_path,completion=proof)['complete']
    report.write_text('changed')
    assert not check(tmp_path,completion=proof)['complete']


def test_claimed_complete_missing_seed_audit_rejected(tmp_path):
    complete(tmp_path);(tmp_path/'seed-1004/seed-audit.json').unlink()
    value=check(tmp_path)
    assert value['issues'] and not value['all_classified_and_audited']


def test_gpu_read_failure_never_assumes_idle(tmp_path):
    fixture(tmp_path)
    value=inspect(tmp_path,probe=lambda _:dict(state='alive'),resource_probe=lambda _:dict(gpu=[],gpu_inspection_error='error'))
    assert any('visibility' in x for x in value['issues'])


def test_old_startup_health_traceback_is_not_current_failure(tmp_path):
    import os
    d=fixture(tmp_path);run=json.loads((d/'run.json').read_text())
    run['model_ready_at']=200;save(d/'run.json',run)
    log=d/'bridge.log';log.write_text('Traceback (most recent call last):\nhealth ConnectError\n')
    os.utime(log,(100,100))
    assert not any('fatal' in x for x in check(tmp_path)['issues'])
    os.utime(log,(201,201))
    assert any('fatal' in x for x in check(tmp_path)['issues'])


def reviewed_log(root):
    d=fixture(root);path=d/'bridge.log'
    content=b'Traceback (most recent call last):\nhealth ConnectError\n'
    path.write_bytes(content)
    proof={str(path.resolve()):dict(status='reviewed',root=str(root.resolve()),counter_id='same',
           prefix_bytes=len(content),prefix_sha256=hashlib.sha256(content).hexdigest())}
    return path,proof


def test_benign_append_does_not_reopen_reviewed_startup_trace(tmp_path):
    path,proof=reviewed_log(tmp_path)
    with path.open('ab') as out:out.write(b'tokenizer length warning\n')
    value=check(tmp_path,log_resolutions=proof)
    assert not value['issues']
    assert value['arms']['1001']['reviewed_log_prefixes']['bridge.log']>0
    assert b'Traceback' in path.read_bytes()  # Original evidence is untouched.


def test_new_error_after_review_still_alarms(tmp_path):
    path,proof=reviewed_log(tmp_path)
    first=check(tmp_path,log_resolutions=proof)
    with path.open('ab') as out:out.write(b'Traceback (most recent call last):\nnew failure\n')
    later=check(tmp_path,log_resolutions=proof,previous=first)
    assert any('fatal error in bridge.log' in item for item in findings(later,first))


@pytest.mark.parametrize('change',['root','counter','hash','truncate','rewrite','unreviewed'])
def test_review_cannot_mask_other_or_changed_log(tmp_path,change):
    path,proof=reviewed_log(tmp_path);row=proof[str(path.resolve())]
    if change=='root':row['root']+='-different'
    elif change=='counter':row['counter_id']='new-engine'
    elif change=='hash':row['prefix_sha256']='bad'
    elif change=='truncate':path.write_bytes(b'Traceback (most recent call last):\n')
    elif change=='rewrite':path.write_bytes(path.read_bytes().replace(b'health',b'broken'))
    else:row['status']='unreviewed'
    assert any('fatal error in bridge.log' in item for item in check(tmp_path,log_resolutions=proof)['issues'])
