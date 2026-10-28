"""Frozen two-owner >G native-output qualification; never runs Agent tasks.

Reuse the completed DeepSWE model, K96K/G16K, MTP4 and fixed warmup tokens.
Only the HTTP admission gate is patched. Fixed tokens and expensive byte
oracles prove continuity, not output quality or Agent throughput.
"""
import argparse
import array
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import httpx
import psutil
import run_agent_handoff_queue as q


def require(ok, message):
    if not ok:raise ValueError(message)


def hashes(root):
    return {str(p.relative_to(root)):q.sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def prepare(args):
    root=args.output.resolve();source=args.source.resolve()
    require(not root.exists(),'Use a new output directory; no retries')
    profile=q.read(source/'restart-1004/first16/profile.json')
    kv=profile['kvmem']
    require((kv['select_budget'],kv['gen_budget'],kv['block_tokens'])==(98304,16384,32), 'Require K96K/G16K/block32')
    require(kv['rolling_active_window'] and profile['draft_method']=='mtp' and profile['draft_tokens']==4,
            'Require qualified rolling MTP4 runtime')
    original=source/'runtime/src/vllm_kvmem/http_sessions.py'
    changed=Path(__file__).resolve().parents[1]/'src/vllm_kvmem/http_sessions.py'
    # Bind exactly this change, without importing other shared-worktree edits.
    old="    if max_tokens > cfg['gen_budget']:\n"
    replacement=("    # G is physical decode headroom in rolling mode, not a response limit.\n"
        "    # Keep one native request (and its sampler/tool frame) across refreshes.\n"
        "    # Native serving still enforces prompt + output against max_model_len.\n"
        "    if max_tokens > cfg['gen_budget'] and not cfg.get('rolling_active_window', False):\n")
    require(original.read_text().count(old)==1 and original.read_text().replace(old,replacement)==changed.read_text(),
            'Candidate must differ only in the rolling HTTP output gate')
    root.mkdir(parents=True)
    for name in ('src','csrc'):
        shutil.copytree(source/'runtime'/name,root/'runtime'/name,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(changed,root/'runtime/src/vllm_kvmem/http_sessions.py')
    (root/'controller').mkdir()
    for path in (Path(__file__),Path(q.__file__)):
        shutil.copy2(path,root/'controller'/path.name)
    shutil.copy2(source/'warmup-tokens.bin',root/'input.bin')
    # Concurrency two bounds test cost; memory budgets, precision and policies stay fixed.
    profile.update(port=args.port,max_num_seqs=2,verify_session_restore=True)
    q.save(root/'profile.json',profile)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    token=tokenizer.encode(' x',add_special_tokens=False)[-1]
    require(token not in tokenizer.all_special_ids,'Fixed output must be an ordinary token')
    manifest=dict(source=str(source),source_profile_sha256=q.sha(source/'restart-1004/first16/profile.json'),
        output_tokens=36864,concurrency=2,token=token,eos=tokenizer.eos_token_id,idle_seconds=60,
        scope='Correctness component only; fixed output, two owners, no task retries or throughput claim',
        runtime=hashes(root/'runtime'),controller=hashes(root/'controller'),
        profile_sha256=q.sha(root/'profile.json'),input_sha256=q.sha(root/'input.bin'),
        old_http_sha256=q.sha(original),new_http_sha256=q.sha(changed))
    q.save(root/'manifest.json',manifest);q.save(root/'status.json',dict(status='prepared',at=time.time()))


def audit(rows, proofs, k=98304, g=16384):
    long=[r for r in rows if r['phase']=='long']
    require(len(long)==2,'Missing long requests')
    for row in long:
        require(row['accepted']==36864 and row['finish']=='length','Incomplete >2G output')
        owned=[p for p in proofs if p['request_id'].startswith(row['id']+'-')]
        require(len(owned)>=2,'Each original request must cross multiple rolling windows')
        require(all(p['selected_tokens']<=k and p['replay_tokens']==p['private_forward_steps']==0
            and p['target_kv_byte_equal'] and p['recurrent_byte_equal'] and not p['cancelled']
            and p['target_kv_bytes']>0 and p['mtp_attention_layers']>0 for p in owned),
            'Rolling KV/recurrent/MTP byte proof failed')
        require(max(p['history_end'] for p in owned)>row['prompt_tokens']+g,
            'No verified pressure after exceeding G output')
    for owner in range(2):
        require(any(r['owner']==owner and r['phase']=='resume' and r['accepted']==64 for r in rows), 'Missing exact-prefix resume')
        require(any(r['owner']==owner and r['phase']=='eos' and r['finish']=='stop' and r['accepted']<=1 for r in rows), 'EOS did not stop')
        require(any(r['owner']==owner and r['phase']=='release' for r in rows), 'Session release missing')
    return dict(passed=True,long_requests=len(long),rolling_refreshes=len(proofs),same_native_request=True,
        kv_budget=k,gen_headroom=g,output_per_long_request=36864,byte_oracle=True)


async def workload(root, profile, manifest):
    values=array.array('i');values.frombytes((root/'input.bin').read_bytes())
    if sys.byteorder!='little':values.byteswap()
    initial=values.tolist();require(len(initial)==100352,'Unexpected fixed prompt')
    rows=[]
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}',trust_env=False,timeout=3600) as client:
        async def owner(index):
            session=root.name+'-'+str(index);tokens=list(initial)
            async def call(phase,cap,token,*,eos=False,release=False):
                prompt=[manifest['eos']] if release else list(tokens)
                payload=dict(model=profile['served_model_name'],prompt=prompt,max_tokens=cap,session_id=session,
                    cache_salt=session,return_token_ids=True,temperature=0,seed=1000,ignore_eos=not eos,
                    allowed_token_ids=[token],kvmem_reselect='off',stream=True,stream_options=dict(include_usage=True))
                if release:payload['kv_transfer_params']={'kvmem_release_session':True}
                path=root/f'owner-{index}-{phase}';path.mkdir()
                q.save(path/'request.json',payload)
                accepted=[];usage=None;finish=None;identity=None;saw_done=False;started=time.time()
                with (path/'stream.jsonl').open('x') as trace:
                    async with client.stream('POST','/v1/completions',json=payload) as response:
                        if response.status_code!=200:raise RuntimeError((await response.aread()).decode())
                        async for line in response.aiter_lines():
                            if not line.startswith('data:'):continue
                            data=line[5:].strip()
                            if data=='[DONE]':saw_done=True;break
                            item=json.loads(data);trace.write(json.dumps(dict(at=time.time(),event=item))+'\n');trace.flush()
                            require('error' not in item,'Native stream error: '+str(item))
                            if item.get('id'):
                                require(identity in (None,item['id']),'Request ID changed inside continuous output')
                                identity=item['id']
                            if item.get('usage') is not None:usage=item['usage']
                            for choice in item.get('choices',[]):
                                ids=choice.get('token_ids');require(isinstance(ids,list),'Missing accepted IDs')
                                accepted.extend(ids)
                                if choice.get('finish_reason'):finish=choice['finish_reason']
                            q.save(path/'progress.json',dict(at=time.time(),accepted=len(accepted),id=identity))
                require(saw_done and usage and usage['completion_tokens']==len(accepted),'Truncated/mismatched native response')
                require(len(accepted)<=cap and all(t==token for t in accepted),'Fixed output mismatch')
                if not eos:require(len(accepted)==cap,'Premature output termination')
                q.save(path/'accepted.json',accepted)
                row=dict(owner=index,phase=phase,id=identity,prompt_tokens=len(prompt),accepted=len(accepted),
                    finish=finish,seconds=time.time()-started,usage=usage)
                rows.append(row);q.save(path/'result.json',row)
                if not release:tokens.extend(accepted)
            await call('long',manifest['output_tokens'],manifest['token'])
            # An appended external token follows the exact accepted prefix.
            tokens.append(manifest['token'])
            await call('resume',64,manifest['token'])
            await call('eos',128,manifest['eos'],eos=True)
            await call('release',1,manifest['token'],release=True)
        results=await asyncio.gather(*(owner(i) for i in range(2)),return_exceptions=True)
        q.save(root/'responses.json',rows)
        errors=[r for r in results if isinstance(r,BaseException)]
        if errors:raise errors[0]
    proofs=[json.loads(s) for s in (root/'run/server/capture/verified_rolling_pressure.jsonl').read_text().splitlines()]
    q.save(root/'execution-audit.json',audit(rows,proofs))


def run(root):
    root=root.resolve();m=q.read(root/'manifest.json')
    require(Path(__file__).resolve().parent==root/'controller','Run the frozen controller')
    require(q.read(root/'status.json')['status']=='prepared','No automatic retries')
    require(hashes(root/'runtime')==m['runtime'] and hashes(root/'controller')==m['controller'],'Frozen source changed')
    require(q.sha(root/'profile.json')==m['profile_sha256'] and q.sha(root/'input.bin')==m['input_sha256'],'Frozen input/profile changed')
    q.save(root/'process.json',q.identity(psutil.Process())|dict(at=time.time()))
    owned=q.Owned();success=False
    with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            q.save(root/'status.json',dict(status='waiting_for_gpu',at=time.time()))
            stable=None
            while stable is None or time.monotonic()-stable<m['idle_seconds']:
                stable=None if q.gpu_pids() else stable or time.monotonic()
                time.sleep(2)
            require(psutil.virtual_memory().available>40*1024**3,'Insufficient CPU memory headroom')
            profile=q.read(root/'profile.json')
            server=q.start_model(root/'run',root/'profile.json',root/'runtime',sys.executable,owned)
            q.save(root/'status.json',dict(status='loading',at=time.time()))
            q.ready_http(f'http://127.0.0.1:{profile["port"]}/health',server,owned,timeout=3600,progress_path=root/'startup.json')
            q.save(root/'status.json',dict(status='running',at=time.time()))
            asyncio.run(workload(root,profile,m));success=True
        except BaseException as error:
            q.save(root/'failure.json',dict(error=repr(error),at=time.time()));raise
        finally:
            cleanup=owned.stop();q.save(root/'cleanup.json',cleanup)
            success=success and not cleanup['remaining_processes'] and not cleanup['remaining_owned_gpu_pids']
            q.save(root/'status.json',dict(status='passed' if success else 'failed',at=time.time()))
            evidence={str(p.relative_to(root)):q.sha(p) for p in root.rglob('*.json')
                      if 'runtime' not in p.parts and 'controller' not in p.parts and p.name!='evidence-hashes.json'}
            q.save(root/'evidence-hashes.json',evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--port',type=int,default=18879)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run(args.output)
    else:prepare(args)
