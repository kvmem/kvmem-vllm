"""Real-prefix rewind and varied MTP acceptance stress; no benchmark answers."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import shutil
import time
import httpx
import psutil
import run_agent_handoff_queue as q
from run_terminalbench_suite import hashes, PYTHON

REPO=Path(__file__).resolve().parents[1]

def prepare(root,source):
    if root.exists():raise ValueError('New root required')
    m=q.read(source/'manifest.json')
    if hashes(source/'runtime')!=m['frozen']['runtime']:raise ValueError('Source changed')
    root.mkdir(parents=True)
    shutil.copytree(source/'runtime',root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,root/'runtime/scripts'/Path(__file__).name)
    shutil.copy2(REPO/'artifacts/terminalbench-cuda-diagnostic-20261026-r2/fixture.json',root/'fixture.json')
    p=q.read(source/'profile.json');p['port']=18881
    q.save(root/'profile.json',p)
    q.save(root/'manifest.json',dict(source=str(source),runtime=hashes(root/'runtime'),
        model_changes=[],files={n:q.sha(root/n) for n in ('profile.json','fixture.json')},
        scope='Fixed original prefixes, eight owners, sampled outputs stress MTP acceptance; no task scores or throughput claim'))

async def workload(root,p):
    fixture=q.read(root/'fixture.json');answers=[];tokens={};sessions={}
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{p["port"]}',trust_env=False,timeout=3600) as client:
        async def call(i,phase,cap,reselect=False):
            session=sessions.setdefault(i,root.name+'-'+str(i))
            if i not in tokens:tokens[i]=list(fixture['query' if i%2==0 else 'decode']['tokens'])
            prefix=tokens[i]
            payload=dict(model=p['served_model_name'],session_id=session,cache_salt=session,
                prompt=list(prefix),max_tokens=cap,seed=1000,temperature=.6,top_p=.95,top_k=20,ignore_eos=True,
                return_token_ids=True,kvmem_reselect='auto' if reselect else 'off')
            if reselect:payload['kvmem_query_span']=[len(prefix)-698,len(prefix)]
            start=time.time();r=await client.post('/v1/completions',json=payload)
            r.raise_for_status();data=r.json();q.save(root/f'{phase}-{i}.json',data)
            ids=data['choices'][0]['token_ids']
            if len(ids)!=cap or data['usage']['completion_tokens']!=cap:raise ValueError('Incomplete stress output')
            tokens[i].extend(ids)
            answers.append(dict(owner=i,phase=phase,start=start,end=time.time(),accepted=len(ids),id=data['id']))
        await asyncio.gather(*(call(i,'warm',64) for i in range(8)))
        for turn in range(4):
            for i in range(8):
                span=fixture['query']['plan']['kvmem_query_span']
                if i%2==0:
                    # Rewind to the prefill checkpoint: discard synthetic fixed outputs
                    # and vary the original query suffix, just as Terminus rewrites history.
                    tokens[i]=list(fixture['query']['tokens']) + fixture['query']['tokens'][span[0]:span[1]]
                else:tokens[i].extend([830]*32)
            await asyncio.gather(*(call(i,f'refresh-{turn}',256 if i%2==0 else 1024,i%2==0) for i in range(8)))
        for i,session in sessions.items():
            r=await client.post('/v1/completions',json=dict(model=p['served_model_name'],session_id=session,
                cache_salt=session,prompt=[151645],max_tokens=1,kv_transfer_params=dict(kvmem_release_session=True)))
            r.raise_for_status();q.save(root/f'release-{i}.json',r.json())
    q.save(root/'answers.json',answers)
    path=root/'run/server/capture/async_refresh.jsonl'
    events=[json.loads(s) for s in path.read_text().splitlines()]
    counts={k:sum(e.get('stage')==k for e in events) for k in sorted({e.get('stage','') for e in events})}
    if counts.get('assembly_started',0)<8:raise ValueError('Query assembly path not exercised')
    q.save(root/'execution-audit.json',dict(passed=True,replies=len(answers),accepted=sum(x['accepted'] for x in answers),
        stages=counts,model_runtime=hashes(root/'runtime'),all_outputs_fixed=False,acceptance_stress=True,session_releases=8,
        limit='Service stress qualification, not a reproduction of historical GPU state or Agent answers'))

def run(root):
    m=q.read(root/'manifest.json')
    assert hashes(root/'runtime')==m['runtime']
    assert all(q.sha(root/n)==v for n,v in m['files'].items())
    (root/'started').open('x').close()
    q.save(root/'process.json',q.identity(psutil.Process()))
    owned=q.Owned();lock=open('/tmp/kvmem-coordination-qualification.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle=None
        while True:
            ids=q.gpu_pids();idle=None if ids else (idle or time.monotonic())
            q.save(root/'status.json',dict(status='waiting_for_gpu',gpu=ids,at=time.time()))
            if idle and time.monotonic()-idle>=60:break
            time.sleep(5)
        p=q.read(root/'profile.json')
        q.save(root/'status.json',dict(status='starting',at=time.time()))
        server=q.start_model(root/'run',root/'profile.json',root/'runtime',PYTHON,owned)
        q.ready_http(f'http://127.0.0.1:{p["port"]}/health',server,owned,timeout=3600,progress_path=root/'startup.json')
        q.save(root/'status.json',dict(status='testing',at=time.time()))
        asyncio.run(workload(root,p))
        q.save(root/'status.json',dict(status='completed',at=time.time()))
    except BaseException as e:
        q.save(root/'status.json',dict(status='failed',error=repr(e),at=time.time()));raise
    finally:
        q.save(root/'cleanup.json',owned.stop());lock.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('root',type=Path);a.add_argument('--source',type=Path);v=a.parse_args()
    if v.action=='prepare':prepare(v.root.resolve(),v.source.resolve())
    else:run(v.root.resolve())
