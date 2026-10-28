"""Queue safety, fixed workload, and outcome accounting without GPU/Docker."""
import copy
import io
import json
import urllib.error

import pytest

from scripts import queue_deepswe_fastpath as suite
from scripts import analyze_deepswe


def test_cold_preflight_has_owned_session_and_releases_it_without_changing_tokens(tmp_path,monkeypatch):
    from vllm_kvmem.http_sessions import validate_session_payload
    monkeypatch.setenv('KVMEM_REQUIRE_OWNED_SESSION','1')
    seen=[]
    class HTTP:
        def open(self,request,**kwargs):
            payload=json.loads(request.data);validate_session_payload(payload,None);seen.append(payload)
            return io.BytesIO(b'{"id":"synthetic-response"}')
    payload=dict(prompt=[1,2,3],max_tokens=1,temperature=0,kvmem_query_span=[1,3])
    original=copy.deepcopy(payload)
    assert suite.runner.preflight_completion(HTTP(),dict(port=1234,served_model_name='m'),payload,
        namespace='preflight',release_token=9,output=tmp_path,timeout=5)['id']=='synthetic-response'
    assert payload==original and len(seen)==2
    assert seen[0]==dict(model='m',**original,session_id=seen[0]['session_id'],cache_salt=seen[0]['session_id'])
    assert seen[1]['session_id']==seen[0]['session_id'] and seen[1]['prompt']==[9]
    assert seen[1]['kv_transfer_params']==dict(kvmem_release_session=True)
    seen.clear()
    payload.update(session_id='continuation',cache_salt='continuation')
    suite.runner.preflight_completion(HTTP(),dict(port=1234,served_model_name='m'),payload,
        namespace='preflight',release_token=9,output=tmp_path,timeout=5)
    assert seen==[dict(model='m',**payload)]  # caller owns its warm continuation


def test_preflight_http_failure_is_preserved_with_server_reason(tmp_path):
    class HTTP:
        def open(self,request,**kwargs):
            raise urllib.error.HTTPError(request.full_url,400,'Bad Request',{},io.BytesIO(b'owned session required'))
    with pytest.raises(RuntimeError,match='owned session required'):
        suite.runner.preflight_completion(HTTP(),dict(port=1234,served_model_name='m'),dict(prompt=[1]),
            namespace='preflight',release_token=9,output=tmp_path,timeout=5)
    rows=list((tmp_path/'preflight-requests').glob('*.json'))
    assert len(rows)==1 and suite.q.read(rows[0])['error_status']==400


def test_unstarted_recovery_refuses_any_formal_answer_or_public_forward(tmp_path,monkeypatch):
    (tmp_path/'first16/server/capture').mkdir(parents=True)
    suite.q.save(tmp_path/'first16/run.json',dict(status='needs_attention',tasks={}))
    work=dict(completed=dict(public_batches=0,public_prefill_tokens=0,public_non_prefill_tokens=0))
    suite.q.save(tmp_path/'first16/server/capture/public_work.json',work)
    suite.q.save(tmp_path/'cleanup.json',dict(remaining_processes=[],remaining_owned_gpu_pids=[],owned=[]))
    suite.q.save(tmp_path/'queue-process.json',{})
    monkeypatch.setattr(suite.q,'verify_frozen',lambda root:{})
    monkeypatch.setattr(suite.q,'live',lambda row:None)
    proof=suite.unstarted_recovery_proof(tmp_path)
    assert proof['files']['first16/run.json']==suite.q.sha(tmp_path/'first16/run.json')
    work['completed']['public_batches']=1
    suite.q.save(tmp_path/'first16/server/capture/public_work.json',work)
    with pytest.raises(ValueError,match='Public inference'):
        suite.unstarted_recovery_proof(tmp_path)
    suite.q.save(tmp_path/'first16/run.json',dict(status='needs_attention',tasks={'t':{'state':'completed'}}))
    with pytest.raises(ValueError,match='unstarted formal cohort'):
        suite.unstarted_recovery_proof(tmp_path)


def test_gpu_idle_between_arms_or_after_failed_queue_does_not_allow_successor():
    for status in ('running','completed','failed'):
        assert not suite.predecessor_ready({'status':status},alive=True)
    assert suite.predecessor_ready({'status':'completed'},alive=False)
    for status in ('running','failed','prepared'):
        with pytest.raises(ValueError,match='without an audited'):
            suite.predecessor_ready({'status':status},alive=False)


def test_acknowledged_failure_requires_cleanup_immutable_evidence_and_idle_sessions(tmp_path,monkeypatch):
    suite.q.save(tmp_path/'status.json',dict(status='failed'))
    suite.q.save(tmp_path/'queue-process.json',dict(pid=1,created=2))
    (tmp_path/'queue.log').write_text('original timeout')
    base=tmp_path/'baseline';base.mkdir()
    candidate=tmp_path/'candidate';candidate.mkdir()
    suite.q.save(base/'status.json',dict(status='testing'))
    suite.q.save(candidate/'status.json',dict(status='prepared'))
    (base/'answer.json').write_text('unchanged accepted output')
    suite.q.save(base/'evidence-hashes.json',{'answer.json':suite.q.sha(base/'answer.json')})
    suite.q.save(base/'execution-audit.json',dict(status='failed'))
    suite.q.save(base/'summary.json',dict(completed=1))
    cleanup=dict(remaining_processes=[],remaining_owned_gpu_pids=[],owned=[])
    suite.q.save(base/'cleanup.json',cleanup)
    monkeypatch.setattr(suite.q,'live',lambda row:None)
    idle=[];monkeypatch.setattr(suite.q,'require_idle_dsh',lambda plan:idle.append(True))
    proof=suite.failed_predecessor_proof(tmp_path,{})
    assert proof['status']=='failed_preserved' and idle==[True]
    assert suite.q.read(tmp_path/'status.json')['status']=='failed'
    suite.q.save(tmp_path/'manifest.json',{})
    suite.wait_predecessor(tmp_path,dict(source=str(tmp_path),dependency={},failed_predecessor=proof))
    assert suite.q.read(tmp_path/'dependency-failed-preserved.json')['status']=='failed_preserved'
    cleanup['remaining_owned_gpu_pids']=[123]
    suite.q.save(base/'cleanup.json',cleanup)
    with pytest.raises(ValueError,match='cleanup incomplete'):
        suite.failed_predecessor_proof(tmp_path,{})
    with pytest.raises(ValueError,match='evidence changed'):
        suite.wait_predecessor(tmp_path,dict(source=str(tmp_path),dependency={},failed_predecessor=proof))
    cleanup['remaining_owned_gpu_pids']=[];suite.q.save(base/'cleanup.json',cleanup)
    (base/'answer.json').write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):
        suite.failed_predecessor_proof(tmp_path,{})


def test_profile_preserves_all_model_and_recall_values_except_documented_service_fields():
    original=dict(port=18876,served_model_name='old',max_model_len=524288,
        max_num_seqs=8,max_num_batched_tokens=16424,
        kvmem=dict(restore_handoff_fastpath=True,select_budget=36864,gen_budget=16384,
                   query_mean_quota_fraction=.5,graph_alpha=0,session_capacity=8,
                   pressure_policy='attention',query_fps_halo_blocks=1,cpu_read_cache_bytes=10**11))
    saved=copy.deepcopy(original);new=suite.profile_from(original)
    assert original==saved
    expected=copy.deepcopy(original)
    expected.update(port=18880,served_model_name='kvmem-deepswe')
    expected['kvmem']['session_capacity']=32
    assert new==expected
    original['kvmem']['graph_alpha']=.2
    with pytest.raises(ValueError,match='recall'):
        suite.profile_from(original)


def test_task_metadata_must_match_frozen_image_and_repository(tmp_path):
    instruction=tmp_path/'instruction.md';instruction.write_text('Implement feature')
    metadata=tmp_path/'task.toml';metadata.write_text('docker_image="image:v1"\nbase_commit_hash="abc"\n')
    client=tmp_path/'client';client.write_bytes(b'fixed-client')
    manifest=dict(claude_binary=str(client),claude_sha256=suite.q.sha(client),tasks=[
        dict(task_id='t',instruction=str(instruction),instruction_sha256=suite.q.sha(instruction),
             agent_image='image:v1',base_commit='abc')])
    assert suite.task_files(manifest)[str(metadata)]==suite.q.sha(metadata)
    metadata.write_text('docker_image="image:v2"\nbase_commit_hash="abc"\n')
    with pytest.raises(ValueError,match='repository/image'):
        suite.task_files(manifest)


def test_context_limited_cases_stay_unscored_and_duplicate_stream_events_are_deduplicated(tmp_path,monkeypatch):
    d=tmp_path/'first16';d.mkdir();(d/'bridge').mkdir();(d/'cases').mkdir()
    tasks=[{'task_id':f't{i}'} for i in range(16)]
    states={t['task_id']:dict(state='completed' if i<2 else 'context_ceiling') for i,t in enumerate(tasks)}
    suite.q.save(d/'manifest.json',dict(tasks=tasks))
    suite.q.save(d/'run.json',dict(status='completed',cleanup_errors=[],cleanup_finished_at=1,tasks=states))
    suite.q.save(tmp_path/'previous-vllm-outcomes.json',dict(tasks={}))
    response=dict(type='assistant',message=dict(content=[dict(type='tool_use',id='call',name='Read',input={})]))
    result=dict(type='user',message=dict(content=[dict(type='tool_result',tool_use_id='call',is_error=True)]))
    for t in tasks:
        case=d/'cases'/t['task_id'];case.mkdir()
        (case/'claude.stream.jsonl').write_text('\n'.join(json.dumps(x) for x in [response,response,result,result])+'\n')
    monkeypatch.setattr(analyze_deepswe,'analyze',lambda root:dict(completed_tasks=2,passed_tasks=1,
        infrastructure_errors=0,mixed_end_to_end_output_tokens_per_second=10,output_tokens=100,
        elapsed_seconds=10,tasks=[dict(task=k,state=v['state'],reward=(1 if k=='t0' else 0) if v['state']=='completed' else None)
                                  for k,v in states.items()]))
    report=suite.final_review(tmp_path)
    assert report['scored']==2 and report['solved']==1 and report['context_ceiling']==14
    assert report['solved_fraction_of_full_cohort']==1/16
    assert all(t['reward'] is None for t in report['tasks'][2:])
    assert report['tool_counts']==dict(tool_calls=16,tool_results=16,tool_errors=16,tool_result_error_rate=1)
    states['t5']['state']='infrastructure_error'
    suite.q.save(d/'run.json',dict(status='completed',cleanup_errors=[],cleanup_finished_at=1,tasks=states))
    with pytest.raises(ValueError,match='unclassified'):
        suite.final_review(tmp_path)
