import json
from pathlib import Path
from scripts.monitor_terminalbench import inspect,findings


def fixture(root,status='formal'):
    def save(n,v):
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    save('status.json',{'status':status})
    save('manifest.json',{'launched_task_names':['remaining']})
    for name in ['queue-process.json','formal-process.json','model/server-process.json','bridge-process.json']:
        save(name,{'pid':123,'created':456})
    save('formal-events/status.json',{'outcomes':[],'active':['running']})
    save('preserved-outcomes.json',[{'task':'done','status':'scored','passed':False}])
    return save


def check(root,**kwargs):
    return inspect(root,probe=lambda r:dict(state='alive',pid=r['pid']),resource_probe=lambda r:dict(gpu=[]),**kwargs)


def test_normal_progress_is_quiet_and_preserved_failure_is_not_retried(tmp_path):
    save=fixture(tmp_path);a=check(tmp_path,now=100)
    assert not findings(a,{}) and a['scored']==1 and a['passed']==0
    save('model/server/capture/public_work.json',{'counter_id':'a','completed':{'public_batches':100},'pid':123})
    b=check(tmp_path,previous=a,now=1400)
    assert not b['issues'] and b['last_progress_at']==1400


def test_new_task_error_wakes_once_even_if_queue_alive(tmp_path):
    save=fixture(tmp_path);a=check(tmp_path,now=1)
    save('formal-events/status.json',{'outcomes':[{'task':'broken','status':'error','exception':'CancelledError'}]})
    b=check(tmp_path,previous=a,now=2)
    assert any('CancelledError' in s for s in findings(b,a))
    assert not findings(check(tmp_path,previous=b,now=3),b)


def test_failed_queue_and_long_silence_are_distinct(tmp_path):
    save=fixture(tmp_path);a=check(tmp_path,now=1)
    b=check(tmp_path,previous=a,now=1202)
    assert any('inspect before' in s for s in b['issues'])
    save('status.json',{'status':'failed','error':'Owned model/bridge exited'})
    c=check(tmp_path,previous=b,now=1300)
    assert any('Queue failed' in s for s in c['issues'])


def test_finite_completion_requires_final_review_report_and_cleanup(tmp_path):
    save=fixture(tmp_path,'completed');a=check(tmp_path)
    assert not a['complete'] and findings(a,{})
    save('selected-final-review.json',{'scored':89})
    save('cleanup.json',{'remaining_processes':[],'remaining_owned_gpu_pids':[]})
    report=tmp_path/'report.txt';report.write_text('All outcomes verified')
    import hashlib
    c=dict(status='completed',root=str(tmp_path),final_review_complete=True,report=str(report),
           report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    assert check(tmp_path,completion=c)['complete']
    report.write_text('changed')
    assert not check(tmp_path,completion=c)['complete']
