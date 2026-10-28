"""Identical full real DeepSWE histories, separate from live task scoring."""
import argparse
import array
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_campaign_environment(campaign):
    """Keep native and KVMem in the same recorded interpreter and dependency set."""
    def interpreter_identity(executable):
        path=Path(executable).expanduser().absolute()
        # Resolving the binary alone collapses distinct virtual environments
        # whose bin/python symlinks point at the same system interpreter.
        return path.parent.resolve(),path.resolve()
    if interpreter_identity(campaign['python'])!=interpreter_identity(sys.executable):
        raise ValueError('Run the replay with the frozen campaign interpreter: '+campaign['python'])
    expected=campaign.get('versions')
    if not isinstance(expected,dict) or not expected:
        raise ValueError('Frozen campaign must record dependency versions')
    changed=[]
    for package,wanted in expected.items():
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual='not installed'
        if actual!=wanted:changed.append(f'{package}: expected {wanted}, found {actual}')
    if changed:
        raise ValueError('Frozen campaign dependency versions changed: '+'; '.join(changed))


class Telemetry:
    """Independent NVML sampling; HTTP metrics must never stall VRAM samples."""
    def __init__(self,directory,base,interval=1.,nvml=None):
        if nvml is None:import pynvml as nvml
        self.nvml=nvml;self.directory=directory;self.base=base;self.interval=interval
        self.stop_event=threading.Event();self.errors=[];self.threads=[]
        nvml.nvmlInit()
        if nvml.nvmlDeviceGetCount()!=1:raise ValueError('This experiment requires one physical GPU')
        self.handle=nvml.nvmlDeviceGetHandleByIndex(0)
        decode=lambda s:s.decode() if isinstance(s,bytes) else s
        save(directory/'telemetry-device.json',dict(source='NVML v2 device memory.used; excludes driver-reserved bytes, matching nvidia-smi',
            uuid=decode(nvml.nvmlDeviceGetUUID(self.handle)),name=decode(nvml.nvmlDeviceGetName(self.handle)),
            driver=decode(nvml.nvmlSystemGetDriverVersion()),interval_seconds=interval,
            timing='monotonic timestamp after each actual sample; no interpolation'))

    def snapshot(self):
        began=time.monotonic();memory=self.nvml.nvmlDeviceGetMemoryInfo(self.handle,version=self.nvml.nvmlMemory_v2)
        utilization=self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        return dict(at=time.monotonic(),wall_time=time.time(),query_begin=began,
            memory_used_bytes=int(memory.used),memory_total_bytes=int(memory.total),
            memory_free_bytes=int(memory.free),memory_reserved_bytes=int(memory.reserved),
            gpu_utilization_percent=int(utilization.gpu))

    def _gpu(self):
        try:
            with (self.directory/'gpu-telemetry.jsonl').open('w',buffering=1) as out:
                while True:
                    began=time.monotonic();out.write(json.dumps(self.snapshot())+'\n')
                    if self.stop_event.wait(max(0,self.interval-(time.monotonic()-began))):
                        out.write(json.dumps(self.snapshot())+'\n');break
        except BaseException as error:self.errors.append(repr(error))

    def _metrics(self):
        import urllib.request
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with (self.directory/'metrics-telemetry.jsonl').open('w',buffering=1) as out:
            while not self.stop_event.is_set():
                began=time.monotonic()
                try:
                    with opener.open(self.base+'/metrics',timeout=1) as response:body=response.read().decode()
                    row=dict(at=time.monotonic(),query_begin=began,raw_metrics=body)
                except Exception as error:row=dict(at=time.monotonic(),query_begin=began,error=repr(error))
                out.write(json.dumps(row)+'\n')
                self.stop_event.wait(max(0,2-(time.monotonic()-began)))

    def start(self):
        for action in (self._gpu,self._metrics):
            thread=threading.Thread(target=action,daemon=True);thread.start();self.threads.append(thread)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:thread.join(timeout=5)
        if any(t.is_alive() for t in self.threads):self.errors.append('telemetry thread failed to stop')
        self.nvml.nvmlShutdown()
        save(self.directory/'telemetry-status.json',dict(passed=not self.errors,errors=self.errors))


async def receive_stream(client, url, payload, path, began, phase_begin):
    """Persist complete SSE events at client arrival; never infer token times."""
    text_parts=[];token_ids=[];usage=None;metrics=None;finished=None;done=False;events=0
    first_token=None;prompt_ids=None;response_id=None
    with Path(path).open('w',buffering=1) as capture:
        async with client.stream('POST',url,json=payload) as response:
            response.raise_for_status()
            fields=[]
            async for line in response.aiter_lines():
                if line:
                    if line.startswith('data:'):fields.append(line[5:].lstrip(' '))
                    continue
                if not fields:continue
                now=time.monotonic();data='\n'.join(fields);fields=[]
                capture.write(json.dumps(dict(seq=events,at=now,elapsed=now-began,
                    phase_elapsed=now-phase_begin,data=data),ensure_ascii=False)+'\n')
                events+=1
                if data=='[DONE]':done=True;break
                chunk=json.loads(data)
                if 'error' in chunk:raise ValueError('SSE server error: '+str(chunk['error']))
                response_id=chunk.get('id',response_id)
                if chunk.get('usage') is not None:usage=chunk['usage']
                if chunk.get('metrics') is not None:metrics=chunk['metrics']
                for choice in chunk.get('choices',[]):
                    if choice['index']!=0:raise ValueError('Unexpected streamed choice')
                    ids=choice.get('token_ids')
                    if ids is None:raise ValueError('Streamed token IDs missing')
                    if ids and first_token is None:first_token=now
                    token_ids.extend(ids);text_parts.append(choice.get('text',''))
                    if choice.get('prompt_token_ids') is not None:prompt_ids=choice['prompt_token_ids']
                    if choice.get('finish_reason') is not None:finished=choice['finish_reason']
    if not done or finished!='length' or usage is None:
        raise ValueError('Incomplete stream: DONE, quota finish, and final usage are required')
    if len(token_ids)!=payload['max_tokens'] or usage['completion_tokens']!=len(token_ids):
        raise ValueError('Streamed token count differs from fixed quota')
    if usage['prompt_tokens']!=len(payload['prompt']) or prompt_ids!=payload['prompt']:
        raise ValueError('Streamed input differs from frozen prompt')
    return dict(id=response_id,choices=[dict(index=0,text=''.join(text_parts),token_ids=token_ids,
        finish_reason=finished)],usage=usage,metrics=metrics or {}), dict(
        file=Path(path).name,events=events,first_token_at=first_token,client_ttft_seconds=first_token-began,
        timing='monotonic timestamp of complete SSE data event arrival; no token interpolation')


def prepare(args):
    budgets={backend:getattr(args,f'{backend}_gpu_memory_utilization',None) for backend in ('native','kvmem')}
    if any(value is not None and (not math.isfinite(value) or not 0<value<1) for value in budgets.values()):
        raise ValueError('GPU memory utilization must be between zero and one')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    campaign=args.campaign.resolve();plan=read(campaign/'manifest.json')
    source=read(args.history/'manifest.json')
    if len(source['samples']) != 8:raise ValueError('Eight distinct real histories required')
    for path,digest in source['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Original trajectory provenance changed')
    shutil.copy2(Path(__file__),root/'run.py')
    shutil.copy2(campaign/'warmup.i32',root/'warmup.i32')
    shutil.copy2(args.history/'manifest.json',root/'history-manifest.json')
    for row in source['samples']:
        if sha(args.history/row['file'])!=row['sha256']:raise ValueError('History changed')
        if not 212992<=row['prompt_tokens'] or row['prompt_tokens']+1+16384>262144:
            raise ValueError('Full history must fit the common logical limit and output reservation')
        shutil.copy2(args.history/row['file'],root/row['file'])
    if len({r['sha256'] for r in source['samples']})!=8:
        raise ValueError('Do not duplicate a prompt to simulate eight distinct histories')
    for backend in ('native','kvmem'):
        directory=root/backend;directory.mkdir()
        profile=dict(plan['profile'],port=18962)
        if budgets[backend] is not None:profile['gpu_memory_utilization']=budgets[backend]
        budgets[backend]=profile['gpu_memory_utilization']
        save(directory/'profile.json',profile)
    save(root/'manifest.json',dict(campaign=str(campaign),campaign_manifest_sha256=sha(campaign/'manifest.json'),
        runtime=str(campaign/'runtime'),python=plan['python'],output_tokens=4096,concurrency=8,
        stream=bool(getattr(args,'stream',False)),telemetry=bool(getattr(args,'telemetry',False)),created_at=time.time(),
        gpu_memory_utilization=budgets,memory_policy=getattr(args,'memory_policy','inherited'),
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},
        scope='Eight exact recorded prompts from four real tasks, no crop/padding; cold prime then fixed recorded continuation. '
              'Same 4096 output quota with ignore_eos; measures capacity and serving throughput, not task correctness.'))
    save(root/'status.json',dict(status='prepared',at=time.time()))


async def workload(root, backend):
    import httpx
    from scripts.analyze_deepswe_capacity import metrics,delta,prefill_count,distribution
    directory=root/backend;profile=read(directory/'profile.json');manifest=read(root/'manifest.json')
    source=read(root/'history-manifest.json');native=backend=='native'
    base=f'http://127.0.0.1:{profile["port"]}'
    rows=[]
    for sample in source['samples']:
        tokens=array.array('I');tokens.frombytes((root/sample['file']).read_bytes())
        if sys.byteorder!='little':tokens.byteswap()
        if len(tokens)!=sample['prompt_tokens']:raise ValueError('Token count changed')
        rows.append(dict(sample,tokens=tokens.tolist()))
    async with httpx.AsyncClient(trust_env=False,timeout=3600) as client:
        async def counters():
            response=await client.get(base+'/metrics');response.raise_for_status()
            return response.text
        phase_times={}
        async def call(i,phase):
            sample=rows[i];tokens=sample['tokens']+([sample['continuation_token_id']] if phase=='resume' else [])
            count=manifest['output_tokens'] if phase=='resume' else 1
            sid=f'{root.name}-{backend}-{i}'
            payload=dict(model=profile['served_model_name'],prompt=tokens,max_tokens=count,
                temperature=.6,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,repetition_penalty=1.,
                seed=1000,thinking_token_budget=4096,ignore_eos=True,return_token_ids=True,cache_salt=sid)
            if not native:payload.update(session_id=sid,**sample['request_plan'])
            began=time.monotonic()
            stream_info=None
            if phase=='resume' and manifest.get('stream'):
                payload.update(stream=True,stream_options=dict(include_usage=True))
                result,stream_info=await receive_stream(client,base+'/v1/completions',payload,
                    directory/f'{phase}-{i}.sse.jsonl',began,phase_times['resume_begin'])
                status_code=200
            else:
                response=await client.post(base+'/v1/completions',json=payload)
                status_code=response.status_code
                result=response.json() if status_code==200 else None
            ended=time.monotonic()
            record=dict(phase=phase,index=i,session=sid,input_tokens=len(tokens),input_sha256=hashlib.sha256(
                array.array('I',tokens).tobytes()).hexdigest(),begin=began,end=ended,seconds=ended-began,
                http_status=status_code,source_sha256=sample['sha256'],stream=stream_info)
            if status_code!=200:
                save(directory/f'{phase}-{i}.json',dict(record,error=response.text))
                response.raise_for_status()
            record['response']=result
            save(directory/f'{phase}-{i}.json',record)
            if result['usage']['prompt_tokens']!=len(tokens) or result['usage']['completion_tokens']!=count:
                raise ValueError('Delivered work differs from fixed quota')
            if len(result['choices'][0]['token_ids'])!=count:raise ValueError('Accepted token IDs missing')
            return record
        before=await counters();(directory/'metrics-before.txt').write_text(before)
        began=time.monotonic()
        phase_times.update(prime_begin=began,prime_begin_wall=time.time())
        save(directory/'phase-times.json',phase_times)
        prime=await asyncio.gather(*(call(i,'prime') for i in range(8)))
        primed=time.monotonic()
        phase_times['prime_end']=primed
        if not native:
            from scripts.benchmark_deepswe_first16 import wait_preflight_release
            # Read the idle-session publication view without issuing a cleanup.
            capture=directory/'server/capture';deadline=time.monotonic()+60
            while True:
                archive=read(capture/'archive.json');resources=read(capture/'scheduler_resources.json')
                views={tuple(r['namespace']):r for r in resources.get('session_views',[])}
                valid=all((r['session'],r['session']) in views and
                    views[(r['session'],r['session'])]['history']==r['input_tokens'] for r in prime)
                if valid and not archive.get('pending_downloads') and not archive.get('session_retiring_requests'):break
                if time.monotonic()>deadline:raise RuntimeError('Full primed histories did not publish')
                await asyncio.sleep(.1)
            save(directory/'primed-history.json',dict(archive=archive,resources=resources))
            if archive.get('disk_read_bytes') or archive.get('disk_write_bytes'):
                raise ValueError('CPU-only history performed disk KV I/O')
        middle=await counters();(directory/'metrics-primed.txt').write_text(middle)
        resume_begin=time.monotonic()
        phase_times.update(resume_begin=resume_begin,resume_begin_wall=time.time())
        save(directory/'phase-times.json',phase_times)
        outputs=await asyncio.gather(*(call(i,'resume') for i in range(8)))
        end=time.monotonic()
        phase_times['resume_end']=end
        save(directory/'phase-times.json',phase_times)
        expected=8*manifest['output_tokens'];deadline=time.monotonic()+30
        while True:
            after=await counters()
            count=delta(metrics(middle),metrics(after),'vllm:generation_tokens_total')
            if count==expected:break
            if count is not None and count>expected:raise ValueError('Unexpected model output outside this probe')
            if time.monotonic()>deadline:raise ValueError('Accepted counter/delivered output reconciliation failed')
            await asyncio.sleep(.2)
        (directory/'metrics-after.txt').write_text(after)
        a,b,c=map(metrics,(before,middle,after))
        result=dict(status='completed',source_manifest_sha256=sha(root/'history-manifest.json'),
            concurrency=8,output_tokens=expected,cold_prime_seconds=primed-began,
            retirement_seconds=resume_begin-primed,resume_seconds=end-resume_begin,
            total_seconds=end-began,qps=8/(end-resume_begin),
            resume_output_tps=expected/(end-resume_begin),cold_plus_resume_output_tps=(expected+8)/(end-began),
            input_tokens=[r['input_tokens'] for r in prime],
            preemptions=delta(a,c,'vllm:num_preemptions_total'),
            resume_computed_prefill_tokens=prefill_count(c)-prefill_count(b),
            request_seconds=distribution([r['seconds'] for r in outputs]),
            native_ttft_ms=distribution([r['response'].get('metrics',{}).get('time_to_first_token_ms') for r in outputs]),
            client_stream_ttft_seconds=distribution([r['stream']['client_ttft_seconds'] for r in outputs]) if manifest.get('stream') else None,
            scope=manifest['scope'])
        save(directory/'summary.json',result)
        return result


def run(root):
    plan=read(root/'manifest.json');campaign=Path(plan['campaign']);runtime=Path(plan['runtime'])
    if sha(campaign/'manifest.json')!=plan['campaign_manifest_sha256']:raise ValueError('Campaign changed')
    original=read(campaign/'manifest.json')
    if plan['python']!=original['python']:raise ValueError('Replay interpreter differs from frozen campaign')
    validate_campaign_environment(original)
    import psutil
    sys.path[:0]=[str(runtime),str(runtime/'src'),str(runtime/'scripts')]
    from scripts import run_agent_handoff_queue as q
    from scripts.native_agent_launch import build_launch
    from scripts.benchmark_deepswe_first16 import bounded_warmup
    from types import SimpleNamespace
    import urllib.request
    for name,digest in plan['files'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen replay input changed: '+name)
    if read(root/'status.json')['status']!='prepared':raise ValueError('No implicit retry')
    while True:
        state=read(campaign/'status.json')
        if state['status']=='completed':break
        if state['status'] in ('failed','interrupted'):raise RuntimeError('Live campaign did not complete')
        save(root/'status.json',dict(status='waiting_for_live_campaign',live=state,at=time.time()))
        time.sleep(5)
    def interrupt(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for backend in ('native','kvmem'):
                if q.gpu_pids():raise RuntimeError('Foreign GPU process present')
                directory=root/backend;profile=read(directory/'profile.json');owned=q.Owned()
                guard_stop=threading.Event();guard_error=[];telemetry=None
                def guard():
                    while not guard_stop.wait(2):
                        owned.track()
                        error=('GPU telemetry failed' if telemetry is not None and telemetry.errors else
                               'low host memory' if psutil.virtual_memory().available<8*1024**3 else
                               'foreign GPU process' if set(q.gpu_pids())-set(owned.rows) else None)
                        if error:
                            guard_error.append(error);os.kill(os.getpid(),signal.SIGTERM);return
                try:
                    save(root/'status.json',dict(status='running',backend=backend,at=time.time()))
                    env=q.environment(runtime)
                    env.update(MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',CUDA_LAUNCH_BLOCKING='0',PYTHONDONTWRITEBYTECODE='1')
                    env.update(read(campaign/'manifest.json').get('build_environment',{}))
                    if backend=='native':
                        launch=build_launch(profile,port=profile['port'],async_scheduling=True,environ=env)
                        save(directory/'launch.json',{k:v for k,v in launch.items() if k!='environment'})
                        server=owned.start(launch['argv'],runtime,launch['environment'],directory/'server.log',directory/'owner.json')
                    else:
                        server=owned.start([plan['python'],'-B','-m','vllm_kvmem','serve','--config',str(directory/'profile.json'),
                            '--run-dir',str(directory/'server')],runtime,env,directory/'server.log',directory/'owner.json')
                    monitor=threading.Thread(target=guard,daemon=True);monitor.start()
                    q.ready_http(f'http://127.0.0.1:{profile["port"]}/health',server,owned,timeout=1800)
                    if plan.get('telemetry'):
                        telemetry=Telemetry(directory,f'http://127.0.0.1:{profile["port"]}');telemetry.start()
                    bounded_warmup(SimpleNamespace(warmup_tokens=root/'warmup.i32',backend_kind=backend),profile,directory,
                        urllib.request.build_opener(urllib.request.ProxyHandler({})))
                    asyncio.run(workload(root,backend))
                finally:
                    guard_stop.set()
                    try:
                        if telemetry is not None:telemetry.stop()
                    finally:save(directory/'cleanup.json',owned.stop())
                    if guard_error:save(directory/'guard-error.json',guard_error)
                if telemetry is not None and telemetry.errors:raise RuntimeError('GPU telemetry failed')
            arms={b:read(root/b/'summary.json') for b in ('native','kvmem')}
            save(root/'summary.json',dict(arms=arms,
                resume_gain_percent=100*(arms['kvmem']['resume_output_tps']/arms['native']['resume_output_tps']-1),
                cold_plus_resume_gain_percent=100*(arms['kvmem']['cold_plus_resume_output_tps']/arms['native']['cold_plus_resume_output_tps']-1)))
            save(root/'status.json',dict(status='completed',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path);p.add_argument('--history',type=Path)
    p.add_argument('--stream',action='store_true',help='Capture actual resume SSE arrival timestamps')
    p.add_argument('--telemetry',action='store_true',help='Record NVML VRAM at 1 Hz and independent serving metrics')
    p.add_argument('--native-gpu-memory-utilization',type=float)
    p.add_argument('--kvmem-gpu-memory-utilization',type=float)
    p.add_argument('--memory-policy',choices=('inherited','near-full-physical'),default='inherited')
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    else:run(args.output.resolve())
