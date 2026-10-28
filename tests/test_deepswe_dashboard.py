import importlib.util
import json
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).resolve().parents[1] / 'scripts/deepswe_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def sample(at, prompt, cached, output, phase='running'):
    return {'at': at, 'run_phase': phase, 'counters': {
        dashboard.PROMPT: prompt, dashboard.CACHED: cached, dashboard.OUTPUT: output}}


def test_excludes_preflight_and_cached_history_from_throughput():
    rows = [sample(90, 5000, 1000, 2000, 'starting'), sample(100, 7000, 2000, 2100),
            sample(110, 18000, 12000, 2400)]
    series = dashboard.throughput_series(rows, 99)
    assert len(series) == 2
    assert series[0]['prefill'] is None
    assert series[-1]['prefill'] == 100  # 11K prompt, but 10K restored history.
    assert series[-1]['decode'] == 30


def test_preflight_rate_is_separate_and_cannot_cross_formal_boundary():
    rows = [sample(90, 5000, 1000, 2000, 'starting'),
            sample(100, 7000, 2000, 2100, 'starting'), sample(110, 18000, 12000, 2400)]
    preflight = dashboard.throughput_series(rows, 80, 'starting')
    assert len(preflight) == 2 and preflight[-1]['decode'] == 10
    assert dashboard.throughput_series(rows, 105)[0]['decode'] is None


def test_reset_missing_metrics_and_collection_gaps_are_not_fake_zeroes():
    rows = [sample(100, 100, 0, 50), sample(110, 10, 0, 1)]
    assert dashboard.throughput_series(rows, 99)[-1]['decode'] is None
    rows.append({'at': 120, 'error': 'offline'})
    rows.append(sample(130, 20, 0, 2))
    assert dashboard.throughput_series(rows, 99)[-1]['decode'] is None
    rows.append(sample(200, 30, 0, 3))
    assert dashboard.throughput_series(rows, 99)[-1]['prefill'] is None
    rows.append({'at': 210, 'run_phase': 'running', 'counters': {}})
    assert dashboard.throughput_series(rows, 99)[-1]['decode'] is None


def test_partial_json_line_is_consumed_once_when_complete(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(b'{"at":1}\n{"at":')
    reader = dashboard.JsonLines(path)
    assert reader.update() == [{'at': 1}]
    assert reader.update() == []
    with path.open('ab') as stream:
        stream.write(b'2}\n')
    assert reader.update() == [{'at': 2}]
    assert list(reader.rows) == [{'at': 1}, {'at': 2}]


def test_all_manifest_tasks_visible_and_ended_run_not_reported_as_active(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'tasks': [{'task_id': 'a'}, {'task_id': 'b'}]}))
    (tmp_path / 'run.json').write_text(json.dumps({'status': 'needs_attention', 'tasks': {
        'a': {'state': 'running', 'started_at': 100}}, 'model_ready_at': 100, 'ended_at': 130}))
    state = dashboard.Dashboard(tmp_path).snapshot()
    assert [t['state'] for t in state['tasks']] == ['interrupted', 'queued']
    assert state['elapsed_seconds'] == 30
    assert state['latest'] == {}


def test_history_retirement_is_read_from_native_archive_without_inference(tmp_path):
    capture = tmp_path / 'server/capture'
    capture.mkdir(parents=True)
    stats = dict(mode='delete', policy='lru', retired=64, cpu_released=68)
    (capture / 'archive.json').write_text(json.dumps({'historical_gc': stats}))
    assert dashboard.Dashboard(tmp_path).snapshot()['history_gc'] == stats
    (capture / 'archive.json').unlink()
    assert dashboard.Dashboard(tmp_path).snapshot()['history_gc'] is None


def test_gc_partial_progress_does_not_claim_formal_task_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.time, 'time', lambda: 200)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(workers=8,
        tasks=[dict(task_id=str(i)) for i in range(16)])))
    (tmp_path/'run.json').write_text(json.dumps(dict(status='starting', started_at=90)))
    (tmp_path/'long-gc-preflight').mkdir()
    progress = dict(status='running', at=198, started_at=100, target_tokens=10000000,
        owners=[dict(history=500000, turns=15, done=False) for _ in range(8)],
        retired_blocks=1234)
    (tmp_path/'long-gc-preflight/progress.json').write_text(json.dumps(progress))
    capture = tmp_path/'server/capture'
    capture.mkdir(parents=True)
    (capture/'archive.json').write_text(json.dumps(dict(cpu_authority_bytes=456)))
    original = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    state = dashboard.Dashboard(tmp_path).snapshot()
    view = state['preflight']['progress']
    assert view['percent'] == 5 and view['completed_owners'] == 0
    assert view['history_tokens'] == 4000000 and view['target_total'] == 80000000
    assert view['elapsed_seconds'] == 98 and view['age_seconds'] == 2 and not view['stale']
    assert view['retired_blocks'] == 1234 and len(view['owners']) == 8
    assert state['cpu_used_bytes'] == 456
    assert state['planned_tasks'] == 16 and state['completed_tasks'] == 0
    assert state['requests'] == state['output_tokens'] == 0 and state['latest'] == {}
    assert original == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_gc_missing_owners_and_old_heartbeat_are_not_success():
    progress = dict(status='running', at=100, started_at=50, target_tokens=1000,
        owners=[dict(history=1000, turns=2, done=True)])
    view = dashboard.gc_progress_view(progress, 8, 200)
    assert view['percent'] is None and view['history_tokens'] is None
    assert view['owners'][0]['percent'] == 100 and view['owners'][1]['percent'] is None
    assert view['completed_owners'] == 1 and view['status'] == 'running' and view['stale']


def test_gc_reaching_history_target_still_requires_audit():
    progress = dict(status='running', at=100, target_tokens=1000,
        owners=[dict(history=1200, done=False), dict(history=1000, done=True)])
    view = dashboard.gc_progress_view(progress, 2, 100)
    assert view['percent'] == 100 and view['status'] == 'running'
    assert view['completed_owners'] == 1


def test_paused_gc_shows_only_pre_pause_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.time, 'time', lambda: 200)
    (tmp_path/'run.json').write_text(json.dumps(dict(status='starting', started_at=100)))
    (tmp_path/'user-pause.json').write_text(json.dumps(dict(status='paused',
        reason='user_requested', at=130)))
    (tmp_path/'long-gc-preflight').mkdir()
    before = dict(status='running', at=120, started_at=100, target_tokens=1000,
        owners=[dict(history=100, turns=1) for _ in range(8)])
    after = dict(before, at=140, owners=[dict(history=900, turns=9) for _ in range(8)])
    (tmp_path/'long-gc-preflight/progress.json').write_text(json.dumps(after))
    (tmp_path/'long-gc-preflight/samples.jsonl').write_text(
        '\n'.join(map(json.dumps, [before, after]))+'\n')
    state = dashboard.Dashboard(tmp_path).snapshot()
    assert state['state'] == 'paused' and state['preflight']['progress']['percent'] == 10
    assert state['preflight']['progress']['elapsed_seconds'] == 20


def test_failed_queue_cannot_leave_gc_display_running(tmp_path):
    root = tmp_path/'first16'
    root.mkdir()
    (root/'run.json').write_text(json.dumps(dict(status='starting', started_at=100)))
    (tmp_path/'status.json').write_text(json.dumps(dict(status='failed')))
    assert dashboard.Dashboard(root).snapshot()['state'] == 'failed'


def test_formal_handoff_does_not_invent_passed_long_gc(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(preflight_reuse='preflight-reuse-proof.json',
        tasks=[dict(task_id='a')], workers=8)))
    state = dashboard.Dashboard(tmp_path).snapshot()
    assert state['preflight']['reuse_requested']
    assert state['preflight']['progress'] is None and state['preflight']['long_gc'] == {}
    assert state['completed_tasks'] == 0 and state['tasks'][0]['state'] == 'queued'


@pytest.mark.parametrize('queued,expected', [('waiting_for_gpu', 'waiting_for_gpu'),
    ('gpu_lru_qualification', 'starting'), ('failed', 'failed')])
def test_prepared_first16_shows_actual_queue_state_without_model_metrics(tmp_path, queued, expected):
    root = tmp_path/'first16'
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'tasks': [{'task_id': 'a'}]}))
    (tmp_path/'status.json').write_text(json.dumps({'status': queued}))
    state = dashboard.Dashboard(root).snapshot()
    assert state['state'] == expected and state['queue_state'] == queued
    assert state['e2e_output_rate'] is None and state['latest'] == {}
    assert state['tasks'][0]['state'] == 'queued'
    (root/'run.json').write_text(json.dumps({'status': 'running', 'model_ready_at': 100}))
    assert dashboard.Dashboard(root).snapshot()['state'] == 'running'


def test_live_controller_logs_need_no_extra_sampler_or_summary_writer(tmp_path,monkeypatch):
    monkeypatch.setattr(dashboard.time,'time',lambda:130)
    (tmp_path/'bridge').mkdir();(tmp_path/'cases/a').mkdir(parents=True)
    (tmp_path/'server/capture').mkdir(parents=True)
    def write(path,value):
        (tmp_path/path).write_text(json.dumps(value))
    write('manifest.json',dict(tasks=[dict(task_id='a'),dict(task_id='b')],profile=dict(max_model_len=524288)))
    write('run.json',dict(status='running',model_ready_at=100,tasks={
        'a':dict(state='running',started_at=100),'b':dict(state='context_ceiling',started_at=100,ended_at=120)}))
    write('server/capture/scheduler_resources.json',dict(cpu_history_admission=dict(waiting=0)))
    resources=[]
    for at,prompt,cached,output in [(99,100,20,100),(110,200,60,110),(120,400,160,130)]:
        raw='\n'.join(f'{k}{{engine="0"}} {v}' for k,v in {
            dashboard.PROMPT:prompt,dashboard.CACHED:cached,dashboard.OUTPUT:output,
            'vllm:num_requests_running':8,'vllm:num_requests_waiting':0}.items())
        resources.append(dict(at=at,raw_metrics=raw,gpu='70000, 95',archive=dict(cpu_authority_bytes=123)))
    (tmp_path/'resources.jsonl').write_text('\n'.join(map(json.dumps,resources))+'\n')
    finish=dict(event='finish',task='a',at=112,request_id='request-a',usage=dict(completion_tokens=30))
    events=[dict(finish,at=98,request_id='preflight'),finish,finish,
            dict(event='start',task='a',at=125,prompt_tokens=70000)]
    (tmp_path/'bridge/requests.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    call=dict(type='assistant',message=dict(content=[dict(type='tool_use',id='c',name='Bash')]))
    result=dict(type='user',message=dict(content=[dict(type='tool_result',tool_use_id='c',is_error=True)]))
    (tmp_path/'cases/a/claude.stream.jsonl').write_text('\n'.join(map(json.dumps,[call,call,result,result]))+'\n')
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    board=dashboard.Dashboard(tmp_path);state=board.snapshot()
    assert state['latest']['decode']==2 and state['latest']['prefill']==10
    assert state['output_tokens']==30 and state['requests']==1 and state['e2e_output_rate']==1
    assert state['gpu']['utilization']==95 and state['cpu_used_bytes']==123
    assert state['tasks'][0]['state']=='model' and state['tasks'][0]['context_tokens']==70000
    assert state['tasks'][0]['pending_seconds']==5
    assert state['tasks'][1]['elapsed_seconds']==20 and state['tasks'][1]['reward'] is None
    assert state['context_limited_tasks']==1 and state['completed_tasks']==0
    assert state['tool_counts']==dict(tool_calls=1,tool_results=1,tool_errors=1,error_rate=1)
    assert state['tasks'][0]['last_tool']==dict(name='Bash',phase='completed',error=True)
    assert not state['stale']
    monkeypatch.setattr(dashboard.time,'time',lambda:140)
    assert board.snapshot()['output_tokens']==30
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_resource_sampler_failure_does_not_fabricate_zero_throughput():
    row=dashboard.resource_sample(dict(at=10,gpu='not available',telemetry_error='endpoint timed out'))
    assert row['error']=='endpoint timed out' and row['counters']=={} and row['gpu']=={}


def test_user_pause_freezes_running_work_but_preserves_existing_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.time, 'time', lambda: 150)
    records = {
        'running': dict(state='running', started_at=100),
        'stopped': dict(state='infrastructure_error', started_at=100, ended_at=140),
        'failed_before_pause': dict(state='infrastructure_error', started_at=100, ended_at=120),
        'unknown_error_time': dict(state='infrastructure_error', started_at=100),
        'completed': dict(state='completed', started_at=100, ended_at=110,
                          result=dict(reward=dict(reward=1))),
        'limited': dict(state='context_ceiling', started_at=100, ended_at=125),
        'queued': dict(state='queued'),
        'interrupted_before_pause': dict(state='running', started_at=100, ended_at=115),
    }
    (tmp_path/'manifest.json').write_text(json.dumps(dict(tasks=[dict(task_id=t) for t in records])))
    (tmp_path/'run.json').write_text(json.dumps(dict(status='needs_attention',
        model_ready_at=100, ended_at=140, tasks=records)))
    (tmp_path/'live-summary.json').write_text(json.dumps(dict(tasks=[
        dict(task='stopped', total_seconds=40)])))
    (tmp_path/'user-pause.json').write_text(json.dumps(dict(status='paused', at=130,
        reason='user_requested', audit=str(tmp_path/'pause-audit.json'))))
    (tmp_path/'bridge').mkdir()
    (tmp_path/'bridge/requests.jsonl').write_text('\n'.join(map(json.dumps, [
        dict(event='finish', at=110, task='running', request_id='done', usage=dict(completion_tokens=60)),
        dict(event='start', at=125, task='running', prompt_tokens=1000),
    ]))+'\n')
    (tmp_path/'native-metrics.jsonl').write_text('\n'.join(map(json.dumps, [
        sample(110, 100, 0, 50), sample(120, 200, 0, 60),
    ]))+'\n')
    original = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    board = dashboard.Dashboard(tmp_path)
    state = board.snapshot()
    tasks = {task['id']: task for task in state['tasks']}
    assert state['state'] == 'paused' and state['metrics_historical'] and not state['stale']
    assert state['ended_at'] == 130 and state['elapsed_seconds'] == 30
    assert state['user_pause']['reason'] == 'user_requested'
    assert state['latest']['decode'] == 1  # Preserve the actual historical sample.
    assert state['e2e_output_rate'] == 2
    assert tasks['running']['state'] == tasks['stopped']['state'] == 'paused'
    assert tasks['running']['elapsed_seconds'] == tasks['stopped']['elapsed_seconds'] == 30
    assert tasks['running']['pending_seconds'] is None
    assert tasks['failed_before_pause']['state'] == tasks['unknown_error_time']['state'] == 'infrastructure_error'
    assert tasks['failed_before_pause']['elapsed_seconds'] == 20
    assert tasks['interrupted_before_pause']['state'] == 'interrupted'
    assert tasks['interrupted_before_pause']['elapsed_seconds'] == 15
    assert tasks['completed']['state'] == 'completed' and tasks['completed']['elapsed_seconds'] == 10
    assert tasks['limited']['state'] == 'context_ceiling' and tasks['queued']['state'] == 'queued'
    assert state['completed_tasks'] == state['passed_tasks'] == state['context_limited_tasks'] == 1
    assert state['infrastructure_errors'] == 2
    monkeypatch.setattr(dashboard.time, 'time', lambda: 1000)
    later = board.snapshot()
    assert later['elapsed_seconds'] == 30 and later['e2e_output_rate'] == 2
    assert later['tasks'][0]['elapsed_seconds'] == 30 and not later['stale']
    assert original == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('pause', [
    dict(status='completed', reason='user_requested', at=130),
    dict(status='paused', reason='error', at=130),
    dict(status='paused', reason='user_requested'),
    dict(status='paused', reason='user_requested', at=True),
    dict(status='paused', reason='user_requested', at=-1),
    dict(status='paused', reason='user_requested', at=float('nan')),
    [],
])
def test_invalid_pause_record_cannot_hide_a_run_failure(tmp_path, pause):
    (tmp_path/'run.json').write_text(json.dumps(dict(status='failed')))
    (tmp_path/'user-pause.json').write_text(json.dumps(pause))
    state = dashboard.Dashboard(tmp_path).snapshot()
    assert state['state'] == 'failed' and state['user_pause'] is None


def test_pause_does_not_extend_a_run_that_already_ended(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.time, 'time', lambda: 200)
    (tmp_path/'run.json').write_text(json.dumps(dict(status='completed', model_ready_at=100, ended_at=120)))
    (tmp_path/'user-pause.json').write_text(json.dumps(dict(status='paused', reason='user_requested', at=130)))
    state = dashboard.Dashboard(tmp_path).snapshot()
    assert state['state'] == 'paused' and state['elapsed_seconds'] == 20


def test_pause_cutoff_excludes_already_consumed_late_finish_metrics_and_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard.time, 'time', lambda: 200)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(tasks=[dict(task_id='a')])))
    (tmp_path/'run.json').write_text(json.dumps(dict(status='running', model_ready_at=100,
        tasks={'a': dict(state='running', started_at=100)})))
    (tmp_path/'live-summary.json').write_text(json.dumps(dict(at=140,
        cpu_authority_bytes=99999, tasks=[dict(task='a', requests=999, output_tokens=99999)])))
    (tmp_path/'bridge').mkdir()
    events = [dict(event='start', task='a', at=105, prompt_tokens=1200),
        dict(event='finish', task='a', at=120, request_id='before', usage=dict(completion_tokens=60),
             metrics=dict(generation_time_ms=1000, time_to_first_token_ms=1000)),
        dict(event='start', task='a', at=135, prompt_tokens=99000),
        dict(event='finish', task='a', at=140, request_id='after', usage=dict(completion_tokens=1000),
             metrics=dict(generation_time_ms=1000, time_to_first_token_ms=10000))]
    (tmp_path/'bridge/requests.jsonl').write_text('\n'.join(map(json.dumps, events))+'\n')
    (tmp_path/'native-metrics.jsonl').write_text('\n'.join(map(json.dumps, [
        sample(110, 100, 0, 50), sample(120, 200, 0, 60), sample(140, 99999, 0, 99999)]))+'\n')
    (tmp_path/'resources.jsonl').write_text('\n'.join(map(json.dumps, [
        dict(at=120, gpu='1000, 30', archive=dict(cpu_authority_bytes=123, historical_gc=dict(retired=10)),
             scheduler_resources=dict(cpu_history_admission=dict(waiting=1))),
        dict(at=140, gpu='9000, 99', archive=dict(cpu_authority_bytes=999, historical_gc=dict(retired=90)),
             scheduler_resources=dict(cpu_history_admission=dict(waiting=5))),
    ]))+'\n')
    capture = tmp_path/'server/capture'
    capture.mkdir(parents=True)
    (capture/'archive.json').write_text(json.dumps(dict(historical_gc=dict(retired=999))))
    (capture/'scheduler_resources.json').write_text(json.dumps(dict(cpu_history_admission=dict(waiting=99))))
    board = dashboard.Dashboard(tmp_path)
    live = board.snapshot()
    assert live['output_tokens'] == 1060 and live['requests'] == 2
    (tmp_path/'user-pause.json').write_text(json.dumps(dict(status='paused', reason='user_requested', at=130)))
    original = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(dashboard.time, 'time', lambda: 201)
    paused = board.snapshot()
    assert paused['measurement_cutoff'] == 130 and paused['elapsed_seconds'] == 30
    assert paused['output_tokens'] == 60 and paused['requests'] == 1 and paused['e2e_output_rate'] == 2
    assert paused['ttft_seconds'] == 1 and paused['single_stream_decode_rate'] == 59
    assert paused['metric_at'] == 120 and paused['latest']['decode'] == 1
    assert paused['latest']['prefill'] == 10 and len(paused['series']) == 2
    assert paused['gpu_at'] == 120 and paused['gpu']['utilization'] == 30
    assert paused['cpu_used_bytes'] == 123 and paused['cpu']['waiting'] == 1
    assert paused['history_gc'] == dict(retired=10)
    assert paused['tasks'][0]['requests'] == 1 and paused['tasks'][0]['output_tokens'] == 60
    assert paused['tasks'][0]['context_tokens'] == 1200
    assert original == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
