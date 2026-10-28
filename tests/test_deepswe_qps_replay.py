import array
import asyncio
import json

import httpx
import pytest

from scripts import benchmark_deepswe_qps_replay as replay


def fixture(tmp_path):
    directory=tmp_path/'native';directory.mkdir()
    replay.save(directory/'profile.json',dict(port=1234,served_model_name='test'))
    replay.save(tmp_path/'manifest.json',dict(output_tokens=2,scope='test'))
    samples=[]
    for i in range(8):
        name=f'{i}.i32';(tmp_path/name).write_bytes(array.array('I',[i,99,100]).tobytes())
        samples.append(dict(file=name,prompt_tokens=3,continuation_token_id=42,
                            request_plan=dict(kvmem_query_span=[1,3]),sha256=replay.sha(tmp_path/name)))
    replay.save(tmp_path/'history-manifest.json',dict(samples=samples))


def backend(monkeypatch, wrong_count=False):
    sent=[];count=[0]
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):
            return httpx.Response(200,request=httpx.Request('GET',url),text=(
                f'vllm:generation_tokens_total {count[0]}\n'
                'vllm:num_preemptions_total 0\n'
                'vllm:prompt_tokens_by_source_total{source="local_compute"} 100\n'))
        async def post(self,url,json):
            sent.append(json);n=json['max_tokens'];count[0]+=n
            return httpx.Response(200,request=httpx.Request('POST',url),json=dict(
                choices=[dict(token_ids=[7]*n)],usage=dict(prompt_tokens=len(json['prompt']),
                    completion_tokens=n-1 if wrong_count else n)))
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    return sent


def test_native_uses_recorded_continuation_not_divergent_prime_prediction(tmp_path,monkeypatch):
    fixture(tmp_path);sent=backend(monkeypatch)
    result=asyncio.run(replay.workload(tmp_path,'native'))
    assert len(sent)==16
    for prime,resume in zip(sent[:8],sent[8:]):
        assert resume['prompt']==prime['prompt']+[42]
        assert resume['cache_salt']==prime['cache_salt']
        assert not any(k.startswith('kvmem') or k=='session_id' for k in resume)
    assert result['output_tokens']==16
    assert result['qps']==8/result['resume_seconds']
    assert result['resume_output_tps']==16/result['resume_seconds']
    assert result['preemptions']==0


def test_incomplete_generation_cannot_be_reported_as_success(tmp_path,monkeypatch):
    fixture(tmp_path);backend(monkeypatch,wrong_count=True)
    with pytest.raises(ValueError,match='fixed quota'):
        asyncio.run(replay.workload(tmp_path,'native'))
    assert (tmp_path/'native/prime-0.json').exists()
    assert not (tmp_path/'native/summary.json').exists()


def stream_client(*, done=True, token_ids=(7,8), error=False):
    chunks=[dict(id='actual',choices=[dict(index=0,text='你好\n',token_ids=[token_ids[0]],
        prompt_token_ids=[1,2,3],finish_reason=None)]),
        dict(id='actual',choices=[dict(index=0,text='world',token_ids=list(token_ids[1:]),finish_reason='length')]),
        dict(id='actual',choices=[],usage=dict(prompt_tokens=3,completion_tokens=2),
             metrics=dict(time_to_first_token_ms=12))]
    if error:chunks=[dict(error=dict(message='engine failed'))]
    body=''.join('data: '+json.dumps(c,ensure_ascii=False)+'\n\n' for c in chunks)
    if done:body+='data: [DONE]\n\n'
    class Bytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            # Split UTF-8 and JSON across network chunks.
            data=body.encode()
            for offset in range(0,len(data),7):yield data[offset:offset+7]
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,stream=Bytes())))


def test_capture_real_sse_events_with_usage_only_chunk(tmp_path):
    path=tmp_path/'capture.jsonl'
    async def exercise():
        async with stream_client() as client:
            return await replay.receive_stream(client,'http://test/completions',
                dict(prompt=[1,2,3],max_tokens=2),path,0,0)
    result,info=asyncio.run(exercise())
    assert result['choices'][0]['text']=='你好\nworld'
    assert result['choices'][0]['token_ids']==[7,8]
    assert result['metrics']['time_to_first_token_ms']==12
    events=[json.loads(line) for line in path.read_text().splitlines()]
    assert info['events']==len(events)==4
    assert [e['seq'] for e in events]==list(range(4))
    assert events[-1]['data']=='[DONE]'
    assert all(a['at']<=b['at'] for a,b in zip(events,events[1:]))


@pytest.mark.parametrize('options,message',[
    (dict(done=False),'Incomplete stream'),
    (dict(token_ids=(7,)),'fixed quota'),
    (dict(error=True),'SSE server error'),
])
def test_bad_stream_cannot_succeed(tmp_path,options,message):
    async def exercise():
        async with stream_client(**options) as client:
            return await replay.receive_stream(client,'http://test/completions',
                dict(prompt=[1,2,3],max_tokens=2),tmp_path/'capture.jsonl',0,0)
    with pytest.raises(ValueError,match=message):asyncio.run(exercise())


def test_nvml_records_actual_bytes_and_final_sample_without_http(tmp_path):
    from types import SimpleNamespace
    samples=[]
    def memory(_,version):
        assert version==123
        used=100+len(samples);samples.append(used)
        return SimpleNamespace(used=used,total=1000,free=950-used,reserved=50)
    fake=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,
        nvmlDeviceGetCount=lambda:1,nvmlDeviceGetHandleByIndex=lambda _:0,
        nvmlDeviceGetUUID=lambda _:'GPU-test',nvmlDeviceGetName=lambda _:b'Test GPU',
        nvmlSystemGetDriverVersion=lambda:'test',nvmlDeviceGetMemoryInfo=memory,nvmlMemory_v2=123,
        nvmlDeviceGetUtilizationRates=lambda _:SimpleNamespace(gpu=80))
    sampler=replay.Telemetry(tmp_path,'http://unreachable',nvml=fake)
    sampler.stop_event.set();sampler._gpu();sampler.stop()
    rows=[json.loads(line) for line in (tmp_path/'gpu-telemetry.jsonl').read_text().splitlines()]
    assert [r['memory_used_bytes'] for r in rows]==[100,101]
    assert rows[1]['at']>=rows[0]['at']>=rows[0]['query_begin']
    assert all(r['memory_total_bytes']==r['memory_used_bytes']+r['memory_free_bytes']+r['memory_reserved_bytes'] for r in rows)
    assert replay.read(tmp_path/'telemetry-status.json')['passed']


def test_replay_rejects_different_venvs_sharing_one_python_binary(tmp_path,monkeypatch):
    executable=tmp_path/'python-base';executable.write_bytes(b'python')
    paths=[]
    for name in ('recorded','current'):
        path=tmp_path/name/'bin'/'python';path.parent.mkdir(parents=True)
        path.symlink_to(executable);paths.append(path)
    assert paths[0].resolve()==paths[1].resolve()
    monkeypatch.setattr(replay.sys,'executable',str(paths[1]))
    with pytest.raises(ValueError,match='campaign interpreter'):
        replay.validate_campaign_environment(dict(python=str(paths[0]),versions={'vllm':'0.30.0'}))


@pytest.mark.parametrize('installed,expected_message',[
    ('0.30.0',None),
    ('0.32.0','vllm: expected 0.30.0, found 0.32.0'),
    (None,'vllm: expected 0.30.0, found not installed'),
])
def test_replay_requires_recorded_dependency_versions(monkeypatch,installed,expected_message):
    def version(package):
        assert package=='vllm'
        if installed is None:raise replay.importlib.metadata.PackageNotFoundError(package)
        return installed
    monkeypatch.setattr(replay.importlib.metadata,'version',version)
    campaign=dict(python=replay.sys.executable,versions={'vllm':'0.30.0'})
    if expected_message:
        with pytest.raises(ValueError,match=expected_message):replay.validate_campaign_environment(campaign)
    else:replay.validate_campaign_environment(campaign)


def test_replay_checks_environment_before_loading_frozen_runtime(tmp_path,monkeypatch):
    campaign=tmp_path/'campaign';campaign.mkdir()
    recorded=tmp_path/'different-venv'/'bin'/'python'
    replay.save(campaign/'manifest.json',dict(python=str(recorded),versions={'vllm':'0.30.0'}))
    replay.save(tmp_path/'manifest.json',dict(campaign=str(campaign),runtime=str(tmp_path/'absent-runtime'),
        python=str(recorded),campaign_manifest_sha256=replay.sha(campaign/'manifest.json')))
    with pytest.raises(ValueError,match='campaign interpreter'):replay.run(tmp_path)
