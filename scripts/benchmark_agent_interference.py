"""Finite fixed-history tool-return interference test; never an Agent score.

Reuse real, hash-bound token fixtures, not original answers. Seven peers decode
while one session returns with an original-token suffix. Each arm has the same
predeclared inputs/outputs. Existing shared prefill admission is tested as-is.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import psutil
from scripts import run_agent_handoff_queue as q

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ['csrc/restore_frame.hpp','csrc/bindings.cpp','src/vllm_kvmem/memory.py',
    'src/vllm_kvmem/prompt_checkpoints.py','src/vllm_kvmem/session_worker.py',
    'src/vllm_kvmem/options.py','src/vllm_kvmem/profile_scopes.py',
    'src/vllm_kvmem/host_trace.py','src/vllm_kvmem/_native.cpython-312-x86_64-linux-gnu.so',
    'scripts/benchmark_agent_interference.py']


def prepare(source, fixture, output):
    if output.exists():
        raise ValueError('New experiment directory required')
    output.mkdir(parents=True)
    runtime = output/'runtime'
    shutil.copytree(source/'runtime', runtime, ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    prior = q.read(ROOT/'artifacts/agent-interference-optimization-20261025-r1/source-before.json')['files']
    for rel in OVERLAYS:
        if rel in prior and q.sha(source/'runtime'/rel) != prior[rel]:
            raise ValueError('Unrelated source change: '+rel)
        (runtime/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, runtime/rel)
    # Apply only this turn's runner hook, preserving other conversations' work.
    runner = runtime/'src/vllm_kvmem/runner.py'
    text = runner.read_text(); anchor = '            install_profile_scopes(self)\n\n        def add_requests('
    if text.count(anchor) != 1:
        raise ValueError('Unknown runner profile hook')
    runner.write_text(text.replace(anchor, '''            install_profile_scopes(self)
            if options.host_stage_timing:
                from .host_trace import install_host_trace
                install_host_trace(self, self._kvmem_probe_output)

        def add_requests('''))
    for p in (ROOT/'csrc').rglob('*'):
        if p.is_file() and p.suffix in ('.cpp','.hpp','.h','.cu','.cuh'):
            rel=p.relative_to(ROOT)
            if str(rel) not in OVERLAYS and q.sha(p)!=q.sha(source/'runtime'/rel):
                raise ValueError('Native binary includes unrelated source: '+str(rel))
    shutil.copy2(fixture, output/'fixture.json')
    values=q.read(fixture)
    if len(values)!=8 or any(len(x['tokens'])<98304 for x in values):
        raise ValueError('Eight long real-history prefixes required')
    control=q.read(source/'first16/profile.json');control['port']=18878
    arms={}
    for name,native,cap,trace,verify in [('profile',False,0,True,False),
          ('qualification',True,512,False,True),('control',False,0,False,False),
          ('capture',True,0,False,False),('cap512',True,512,False,False),('cap256',True,256,False,False)]:
        profile=copy.deepcopy(control)
        profile['verify_session_restore']=verify
        profile['kvmem'].update(native_checkpoint_capture=native,host_stage_timing=trace,
            restore_stage_timing=trace,prefill_batch_tokens=cap)
        arm=output/name;arm.mkdir();q.save(arm/'profile.json',profile)
        q.save(arm/'status.json',dict(status='prepared'))
        arms[name]=dict(profile_sha256=q.sha(arm/'profile.json'),native=native,cap=cap,trace=trace,verify=verify)
    q.save(output/'manifest.json',dict(source=str(source),source_manifest_sha256=q.sha(source/'manifest.json'),
        fixture_source=str(fixture),fixture_sha256=q.sha(fixture),python='/home/chaidi/kvmem-vllm/.venv/bin/python',
        arms=arms,files={str(p.relative_to(runtime)):q.sha(p) for p in runtime.rglob('*') if p.is_file()},
        scope='Fixed real histories, constrained tokens, seven decoding peers plus one tool return. '
              'No original Agent answers rerun. Diagnostic throughput is not real Agent throughput.',
        repetitions=3,suffixes=[32,512,2048,8192],peer_output=512,return_output=64))
    q.save(output/'status.json',dict(status='prepared'))


def payload(profile, tokens, item, owner, output):
    return dict(model=profile['served_model_name'],prompt=tokens,session_id=owner,cache_salt=owner,
        max_tokens=output,temperature=0,seed=1000,ignore_eos=True,return_token_ids=True,
        allowed_token_ids=[item['output_token']],kvmem_reselect='off',
        kvmem_locality_spans=item['locality_spans'], kvmem_locality_provenance=item['locality_provenance'])


async def workload(arm, profile, fixtures, plan, short=False):
    import httpx
    histories=[list(v['tokens']) for v in fixtures]
    owners=[f'interference-{arm.name}-{i}' for i in range(8)]
    rows=[]
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}',trust_env=False,timeout=3600) as http:
        async def request(i, turn, prepared, first=None):
            body, expected, input_hash = prepared
            started=time.perf_counter();events=[];ids=[];usage={};error=None
            try:
                async with http.stream('POST','/v1/completions',content=body,headers={'Content-Type':'application/json'}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: ') or line=='data: [DONE]':continue
                        value=json.loads(line[6:]);now=time.perf_counter()
                        if value.get('error'):raise RuntimeError(value['error'])
                        for choice in value.get('choices',[]):
                            chunk=choice.get('token_ids') or []
                            ids.extend(chunk)
                            if chunk:
                                events.append([now,len(chunk)])
                                if first is not None:first.set()
                        if value.get('usage'):usage=value['usage']
                if ids != [fixtures[i]['output_token']]*expected:
                    raise ValueError('Exact constrained output IDs missing/different')
            except BaseException as exc:
                error=repr(exc)
                raise
            finally:
                row=dict(owner=i,turn=turn,started=started,finished=time.perf_counter(),events=events,
                    output_ids=ids,usage=usage,error=error,input_hash=input_hash,
                    wire_hash=hashlib.sha256(body).hexdigest())
                q.save(arm/'answers'/f'{turn:03}-{i}.json',row);rows.append(row)
            histories[i].extend(ids)
        def wire(i,n):
            data=payload(profile,histories[i],fixtures[i],owners[i],n)
            data.update(stream=True,stream_options={'include_usage':True})
            return (json.dumps(data,separators=(',',':')).encode(),n,
                    hashlib.sha256(json.dumps(histories[i]).encode()).hexdigest())
        # Serialize before decoding starts, to avoid client-side JSON work
        # contaminating peer inter-event gap measurements.
        bodies=[wire(i,64) for i in range(8)]
        async with asyncio.TaskGroup() as group:
            for i in range(8):group.create_task(request(i,0,bodies[i]))
        turn=0
        for repetition in range(1 if short else plan['repetitions']):
            for suffix in ([512] if short else plan['suffixes']):
                turn+=1
                for i in range(8):
                    n=suffix if i==7 else 32
                    # Fixed original token sequences, identical across arms.
                    histories[i].extend(fixtures[i]['tokens'][-n:])
                bodies=[wire(i,plan['return_output'] if i==7 else (128 if short else plan['peer_output'])) for i in range(8)]
                first=[asyncio.Event() for _ in range(7)]
                async def returning():
                    await asyncio.wait_for(asyncio.gather(*(e.wait() for e in first)),1800)
                    await asyncio.sleep(.05)
                    await request(7,turn,bodies[7])
                start=time.perf_counter()
                async with asyncio.TaskGroup() as group:
                    for i in range(7):group.create_task(request(i,turn,bodies[i],first[i]))
                    group.create_task(returning())
                q.save(arm/'progress.json',dict(turn=turn,repetition=repetition,suffix=suffix,
                    round_seconds=time.perf_counter()-start,history_tokens=list(map(len,histories)),at=time.time()))
        q.save(arm/'final-input-hashes.json',[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in histories])
    return rows


def run(root,name):
    plan=q.read(root/'manifest.json');arm=root/name;runtime=root/'runtime'
    if ROOT.resolve()!=runtime.resolve():raise ValueError('Use frozen controller')
    if name not in plan['arms'] or (arm/'started.json').exists():raise ValueError('No implicit retry')
    for rel,h in plan['files'].items():
        if q.sha(runtime/rel)!=h:raise ValueError('Frozen source changed: '+rel)
    if q.sha(arm/'profile.json')!=plan['arms'][name]['profile_sha256']:raise ValueError('Profile changed')
    if q.sha(root/'fixture.json')!=plan['fixture_sha256']:raise ValueError('Fixture changed')
    if q.gpu_pids():raise RuntimeError('GPU occupied; leave existing owner untouched')
    owned=q.Owned();q.save(arm/'started.json',dict(at=time.time(),process=q.identity(psutil.Process())))
    profile=q.read(arm/'profile.json')
    sampler_stop=threading.Event()
    def sample():
        with (arm/'host-memory.jsonl').open('x') as stream:
            while not sampler_stop.wait(1):
                children=psutil.Process().children(recursive=True)
                rows=[]
                for child in children:
                    try:rows.append(dict(pid=child.pid,rss=child.memory_info().rss))
                    except psutil.Error:pass
                stream.write(json.dumps(dict(at=time.time(),available=psutil.virtual_memory().available,processes=rows))+'\n');stream.flush()
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        server=q.start_model(arm/'run',arm/'profile.json',runtime,plan['python'],owned)
        q.ready_http(f'http://127.0.0.1:{profile["port"]}/health',server,owned,timeout=3600,progress_path=arm/'startup.json')
        (arm/'answers').mkdir();q.save(arm/'status.json',dict(status='testing'))
        rows=asyncio.run(workload(arm,profile,q.read(root/'fixture.json'),plan,short=name in ('profile','qualification')))
        capture=arm/'run/server/capture'
        restores=[json.loads(x) for x in (capture/'async_restore.jsonl').read_text().splitlines()]
        if len(restores)!=len(rows)-8:raise ValueError('Missing restores')
        if plan['arms'][name]['verify']:
            if profile.get('verify_session_restore_digest'):
                oracle=[json.loads(x) for x in (capture/'verified_restores_digest.jsonl').read_text().splitlines()]
                if len(oracle)!=len(restores) or not all(x['sha256_equal'] and x['recurrent_byte_equal'] for x in oracle):raise ValueError('Bounded restore oracle mismatch')
            else:
                oracle=[json.loads(x) for x in (capture/'verified_restores.jsonl').read_text().splitlines()]
                if len(oracle)!=len(restores) or not all(x['byte_equal'] for x in oracle):raise ValueError('Byte restore mismatch')
        q.save(arm/'execution-audit.json',dict(status='passed',responses=len(rows),restores=len(restores),component_only=True,
            target_verification='all-byte SHA256' if profile.get('verify_session_restore_digest') else 'direct' if plan['arms'][name]['verify'] else 'not_enabled'))
        q.save(arm/'evidence-hashes.json',{str(p.relative_to(arm)):q.sha(p) for p in (arm/'answers').glob('*')})
        q.save(arm/'status.json',dict(status='completed'))
    except BaseException as exc:
        q.save(arm/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        cleanup=owned.stop();q.save(arm/'cleanup.json',cleanup)
        sampler_stop.set();sampler.join(timeout=5)
    if cleanup['remaining_processes'] or cleanup['remaining_owned_gpu_pids']:raise RuntimeError('Owned cleanup incomplete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--fixture',type=Path);p.add_argument('--arm')
    a=p.parse_args()
    if a.source:prepare(a.source.resolve(),a.fixture.resolve(),a.output.resolve())
    else:run(a.output.resolve(),a.arm)
