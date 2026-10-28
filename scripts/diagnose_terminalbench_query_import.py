"""Fixed-input Query-import boundary qualification; no benchmark answers/retries."""
import argparse, asyncio, fcntl, json, os, shutil, time
from pathlib import Path
import httpx, psutil
import run_agent_handoff_queue as q
from run_terminalbench_suite import hashes, PYTHON
REPO=Path(__file__).resolve().parents[1]

def prepare(root,source,fixture):
    if root.exists(): raise ValueError('New diagnostic root required')
    old=q.read(source/'manifest.json')['frozen']['runtime']
    if hashes(source/'runtime')!=old: raise ValueError('Source changed')
    root.mkdir(parents=True)
    shutil.copytree(source/'runtime',root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    change='src/vllm_kvmem/session_control.py'
    shutil.copy2(REPO/change,root/'runtime'/change)
    shutil.copy2(__file__,root/'runtime/scripts'/Path(__file__).name)
    shutil.copy2(fixture,root/'fixture.json')
    p=q.read(source/'profile.json');p.update(port=18881,verify_session_restore=True,verify_session_restore_digest=True)
    q.save(root/'profile.json',p)
    q.save(root/'manifest.json',dict(source=str(source),runtime=hashes(root/'runtime'),model_changes=[change],
        files={n:q.sha(root/n) for n in ('profile.json','fixture.json')},
        cuda_launch_blocking=True,scope='Eight owners; one exact historical extent and seven equivalent local boundaries. Fixed outputs; no Agent score/throughput claim'))

async def workload(root,p):
    f=q.read(root/'fixture.json');sessions={};tokens={};answers=[]
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{p["port"]}',trust_env=False,timeout=3600) as client:
        async def call(i,phase,cap,query=None):
            session=sessions.setdefault(i,root.name+'-'+str(i));prefix=tokens[i]
            payload=dict(model=p['served_model_name'],session_id=session,cache_salt=session,prompt=list(prefix),
                max_tokens=cap,seed=1000,temperature=0,allowed_token_ids=[830],ignore_eos=True,
                return_token_ids=True,kvmem_reselect='auto' if query else 'off',kvmem_mandatory_spans=[])
            if query:payload['kvmem_query_span']=query
            start=time.time();response=await client.post('/v1/completions',json=payload)
            if response.status_code!=200:raise RuntimeError(response.text)
            data=response.json();q.save(root/f'{phase}-{i}.json',data);ids=data['choices'][0]['token_ids']
            if ids!=[830]*cap or data['usage']['completion_tokens']!=cap:raise ValueError('Incomplete fixed output')
            answers.append(dict(owner=i,phase=phase,id=data['id'],prompt=len(prefix),accepted=cap,
                query=query,start=start,end=time.time()));tokens[i].extend(ids);q.save(root/'answers.json',answers)
        for i in range(8):tokens[i]=list(f['tokens'][:150100 if i==0 else 111361 if i%2==0 else 111356])
        await asyncio.gather(*(call(i,'warm',11368 if i==0 else 64,[148379,150093] if i==0 else None) for i in range(8)))
        queries={}
        for i in range(8):
            history=len(tokens[i])-1;begin=(history+31)//32*32
            # Include the last accepted but uncomputed token when H is sealed.
            if begin==history:queries[i]=[begin,begin+4492];tokens[i].extend(f['tokens'][161473:165971])
            else:
                gap=begin-len(tokens[i]);tokens[i].extend([830]*gap)
                queries[i]=[begin,begin+4492];tokens[i].extend(f['tokens'][161472:165971])
            assert len(tokens[i])==queries[i][1]+7
        await asyncio.gather(*(call(i,'query',64,queries[i]) for i in range(8)))
        # Continued tool append must retain a valid M checkpoint after pressure+Query.
        for i in range(8):tokens[i].extend([830]*32)
        await asyncio.gather(*(call(i,'continued',64) for i in range(8)))
        for i,session in sessions.items():
            response=await client.post('/v1/completions',json=dict(model=p['served_model_name'],session_id=session,
                cache_salt=session,prompt=[151645],max_tokens=1,kv_transfer_params=dict(kvmem_release_session=True)))
            response.raise_for_status();q.save(root/f'release-{i}.json',response.json())
    capture=root/'run/server/capture'
    rows=lambda name:[json.loads(s) for s in (capture/name).read_text().splitlines()]
    restores=rows('verified_restores_digest.jsonl');query_rows=[a for a in answers if a['phase']=='query']
    proof=[]
    for row in query_rows:
        match=[v for v in restores if v['request_id'].startswith(row['id']+'-')]
        if len(match)!=1:raise ValueError('Missing unique Query restore proof')
        v=match[0];expected=98304 if row['owner']%2==0 and row['owner'] else 98299
        if v['working_tokens']!=expected or not v['sha256_equal'] or not v['recurrent_byte_equal']:
            raise ValueError('Incorrect compacted import proof')
        if row['owner']==0 and v['history_tokens']!=161467:raise ValueError('Exact historical boundary not reached')
        proof.append(v)
    if len(restores)!=16 or not all(v['sha256_equal'] and v['recurrent_byte_equal'] for v in restores):
        raise ValueError('Missing continued restore proof')
    refreshes=rows('scheduler_refresh.jsonl')
    for row in query_rows:
        matching=[x for x in refreshes if x['request_id'].startswith(row['id']+'-') and x['replay_tokens']==4492]
        if len(matching)!=1:raise ValueError('New Query was not replayed atomically')
    q.save(root/'execution-audit.json',dict(passed=True,replies=len(answers),accepted=sum(x['accepted'] for x in answers),
        restore_proofs=proof,verified_restores=len(restores),complete_query_replays=8,session_releases=8,
        exact_failure_extent=True,all_outputs_fixed=True,limit='Component correctness only; previous CUDA incident remains unresolved'))
    q.save(root/'evidence-hashes.json',{str(path.relative_to(root)):q.sha(path) for path in root.rglob('*.json')
        if path.is_file() and 'runtime' not in path.parts and path.name not in ('status.json','process.json')})

def run(root):
    m=q.read(root/'manifest.json');assert hashes(root/'runtime')==m['runtime']
    assert all(q.sha(root/n)==v for n,v in m['files'].items())
    (root/'started').open('x').close();q.save(root/'process.json',q.identity(psutil.Process()))
    os.environ['CUDA_LAUNCH_BLOCKING']='1';owned=q.Owned();lock=open('/tmp/kvmem-coordination-qualification.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle=None
        while True:
            ids=q.gpu_pids();idle=None if ids else (idle or time.monotonic())
            q.save(root/'status.json',dict(status='waiting_for_gpu',gpu=ids,at=time.time()))
            if idle and time.monotonic()-idle>=60:break
            time.sleep(5)
        p=q.read(root/'profile.json');q.save(root/'status.json',dict(status='starting',at=time.time()))
        server=q.start_model(root/'run',root/'profile.json',root/'runtime',PYTHON,owned)
        q.ready_http(f'http://127.0.0.1:{p["port"]}/health',server,owned,timeout=3600,progress_path=root/'startup.json')
        q.save(root/'status.json',dict(status='testing',at=time.time()));asyncio.run(workload(root,p))
        q.save(root/'status.json',dict(status='completed',at=time.time()))
    except BaseException as e:
        q.save(root/'status.json',dict(status='failed',error=repr(e),at=time.time()));raise
    finally:q.save(root/'cleanup.json',owned.stop());lock.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('root',type=Path);a.add_argument('--source',type=Path);a.add_argument('--fixture',type=Path);v=a.parse_args()
    if v.action=='prepare':prepare(v.root.resolve(),v.source.resolve(),v.fixture.resolve())
    else:run(v.root.resolve())
