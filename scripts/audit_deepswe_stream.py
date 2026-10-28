"""Audit immutable inputs, measured SSE events, counters and eight-way capacity."""
import argparse
import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert abs(a-b)<1e-6,(a,b)
def main(root):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_deepswe_capacity import metrics,delta
    manifest=read(root/'manifest.json');source=read(root/'history-manifest.json')
    summary=read(root/'summary.json');assert read(root/'status.json')['status']=='completed'
    assert manifest['stream'] and manifest['concurrency']==8 and manifest['output_tokens']==4096
    for name,digest in manifest['files'].items():assert sha(root/name)==digest,name
    campaign=Path(manifest['campaign']);assert sha(campaign/'manifest.json')==manifest['campaign_manifest_sha256']
    original=read(campaign/'manifest.json')
    for name,digest in original['runtime_sha256'].items():assert sha(campaign/'runtime'/name)==digest,name
    profiles={b:read(root/b/'profile.json') for b in ('native','kvmem')}
    if manifest.get('gpu_memory_utilization'):
        for b in profiles:
            assert profiles[b].pop('gpu_memory_utilization')==manifest['gpu_memory_utilization'][b]
    assert profiles['native']==profiles['kvmem'],'Only the declared GPU allocation budget may differ'
    assert len(set(r['sha256'] for r in source['samples']))==8
    arms={};data_files=[root/'manifest.json',root/'summary.json',root/'history-manifest.json']
    for backend in ('native','kvmem'):
        d=root/backend;times=read(d/'phase-times.json');r=summary['arms'][backend];requests=[]
        cleanup=read(d/'cleanup.json');assert not cleanup['remaining_processes'] and not cleanup['remaining_owned_gpu_pids']
        assert read(d/'bounded-warmup.json')['passed']
        for i,sample in enumerate(source['samples']):
            prime=read(d/f'prime-{i}.json');rec=read(d/f'resume-{i}.json');path=d/f'resume-{i}.sse.jsonl'
            ids=[];parts=[];arrivals=[];done=False;usage=None;prompt=None;finish=None;last_at=rec['begin'];chunks=0;events=0
            with path.open() as f:
                for seq,line in enumerate(f):
                    event=json.loads(line);events+=1
                    assert seq==event['seq'] and not done
                    assert rec['begin']<=event['at']<=rec['end'] and event['at']>=last_at
                    close(event['phase_elapsed'],event['at']-times['resume_begin'])
                    close(event['elapsed'],event['at']-rec['begin']);last_at=event['at']
                    if event['data']=='[DONE]':done=True;continue
                    payload=json.loads(event['data']);assert 'error' not in payload
                    if payload.get('usage'):usage=payload['usage']
                    for c in payload.get('choices',[]):
                        assert c['index']==0
                        if c.get('prompt_token_ids') is not None:prompt=c['prompt_token_ids']
                        token_ids=c['token_ids'];ids.extend(token_ids);parts.append(c.get('text',''))
                        if token_ids:arrivals.append(dict(t=event['phase_elapsed'],tokens=len(token_ids)));chunks+=1
                        if c.get('finish_reason'):finish=c['finish_reason']
            assert done and finish=='length' and usage['completion_tokens']==len(ids)==4096
            assert len(prompt)==usage['prompt_tokens']==rec['input_tokens']==sample['prompt_tokens']+1
            binary=array.array('I');binary.frombytes((root/sample['file']).read_bytes())
            if sys.byteorder!='little':binary.byteswap()
            assert prompt==list(binary)+[sample['continuation_token_id']]
            assert sha(root/sample['file'])==sample['sha256']==prime['input_sha256']
            assert hashlib.sha256(array.array('I',prompt).tobytes()).hexdigest()==rec['input_sha256']
            assert rec['source_sha256']==sample['sha256']
            assert ids==rec['response']['choices'][0]['token_ids']
            assert ''.join(parts)==rec['response']['choices'][0]['text']
            assert events==rec['stream']['events']
            close(arrivals[0]['t']+times['resume_begin'],rec['stream']['first_token_at'])
            close(rec['stream']['client_ttft_seconds'],rec['stream']['first_token_at']-rec['begin'])
            requests.append(dict(index=i,task=sample['task'],turn=sample['turn'],input_tokens=len(prompt),output_tokens=len(ids),
                first_chunk_seconds=arrivals[0]['t'],end_seconds=rec['end']-times['resume_begin'],
                sse_events=events,output_chunks=chunks,largest_chunk_tokens=max(a['tokens'] for a in arrivals)))
            data_files.extend((path,d/f'resume-{i}.json',d/f'prime-{i}.json'))
        a,b,c=[metrics((d/name).read_text()) for name in ('metrics-before.txt','metrics-primed.txt','metrics-after.txt')]
        assert delta(a,b,'vllm:generation_tokens_total')==8
        assert delta(b,c,'vllm:generation_tokens_total')==32768
        close(times['resume_end']-times['resume_begin'],r['resume_seconds'])
        close(32768/r['resume_seconds'],r['resume_output_tps']);close(8/r['resume_seconds'],r['qps'])
        assert max(q['end_seconds'] for q in requests)<=r['resume_seconds']
        arms[backend]=dict(requests=requests,prime_preemptions=delta(a,b,'vllm:num_preemptions_total'),
            resume_preemptions=delta(b,c,'vllm:num_preemptions_total'),summary=r,cleanup_passed=True)
        if manifest.get('telemetry'):
            assert read(d/'telemetry-status.json')['passed']
            gpu=[json.loads(line) for line in (d/'gpu-telemetry.jsonl').read_text().splitlines()]
            assert len(gpu)>2 and gpu[0]['at']<=times['prime_begin'] and gpu[-1]['at']>=times['resume_end']
            assert all(a['at']<=b['at'] for a,b in zip(gpu,gpu[1:]))
            assert all(s['memory_total_bytes']==s['memory_used_bytes']+s['memory_free_bytes']+s['memory_reserved_bytes']
                and 0<=s['gpu_utilization_percent']<=100 and s['query_begin']<=s['at'] for s in gpu)
            chosen=[s for s in gpu if times['resume_begin']<=s['at']<=times['resume_end']]
            assert len(chosen)>=int(r['resume_seconds']*.8),'Insufficient measured VRAM samples'
            gaps=[b['at']-a['at'] for a,b in zip(gpu,gpu[1:])]
            arms[backend]['gpu_telemetry']=dict(samples=len(gpu),resume_samples=len(chosen),
                maximum_interval_seconds=max(gaps),
                resume_vram_min_bytes=min(s['memory_used_bytes'] for s in chosen),
                resume_vram_max_bytes=max(s['memory_used_bytes'] for s in chosen),
                minimum_free_bytes=min(s['memory_free_bytes'] for s in gpu),
                resume_used_fraction_min=min(s['memory_used_bytes']/(s['memory_total_bytes']-s['memory_reserved_bytes']) for s in chosen),
                device=read(d/'telemetry-device.json'))
            if manifest.get('memory_policy')=='near-full-physical':
                assert arms[backend]['gpu_telemetry']['resume_used_fraction_min']>=.92,'Near-full memory target not achieved'
            data_files.extend(d/name for name in ('gpu-telemetry.jsonl','metrics-telemetry.jsonl',
                'telemetry-device.json','telemetry-status.json'))
        data_files.extend(d/name for name in ('phase-times.json','summary.json','profile.json','cleanup.json',
            'metrics-before.txt','metrics-primed.txt','metrics-after.txt','bounded-warmup.json'))
    times=read(root/'kvmem/phase-times.json')
    with (root/'kvmem/server/capture/model_step_latency.jsonl').open() as f:
        steps=[r for line in f if times['resume_begin']<=(r:=json.loads(line))['started_at']<=times['resume_end']]
    pure=[s for s in steps if s['prefill_tokens']==0 and s['decode_requests']==8]
    geometry=[g for s in pure for g in s['batch_geometry'] if not g['prefill']]
    assert pure,'No observed eight-way decode steps'
    history=read(root/'kvmem/primed-history.json');archive=history['archive']
    views={tuple(v['namespace']):v for v in history['resources']['session_views']}
    for i,sample in enumerate(source['samples']):
        sid=read(root/'kvmem'/f'prime-{i}.json')['session']
        assert views[sid,sid]['history']==sample['prompt_tokens']
    assert not archive.get('disk_read_bytes') and not archive.get('disk_write_bytes')
    capacity=dict(pure_eight_decode_steps=len(pure),decode_batch_histogram=dict(Counter(s['decode_requests'] for s in steps if s['prefill_tokens']==0)),
        minimum_logical_tokens_in_eight_decode=min(g['logical_seq_len'] for g in geometry),
        maximum_active_tokens_in_eight_decode=max(g['active_seq_len'] for g in geometry),
        primed_histories=[dict(namespace=v['namespace'],history=v['history'],window=v['window']) for v in views.values()],
        cpu_archive_bytes={k:v for k,v in archive.items() if 'bytes' in k and isinstance(v,(int,float))},
        startup_capacity={})
    for backend in arms:
        capacity['startup_capacity'][backend]=[re.sub(r'\x1b\[[0-9;]*m','',line) for line in (root/backend/'server.log').read_text().splitlines()
            if any(s in line for s in ('Available KV cache memory:','GPU KV cache size:','Maximum concurrency for'))]
    data_files.append(root/'kvmem/primed-history.json')
    if manifest.get('telemetry'):
        assert arms['native']['gpu_telemetry']['device']['uuid']==arms['kvmem']['gpu_telemetry']['device']['uuid']
    result=dict(passed=True,scope=manifest['scope'],arms=arms,capacity=capacity,
        resume_tps_ratio=arms['kvmem']['summary']['resume_output_tps']/arms['native']['summary']['resume_output_tps'],
        cold_inclusive_tps_ratio=arms['kvmem']['summary']['cold_plus_resume_output_tps']/arms['native']['summary']['cold_plus_resume_output_tps'],
        source_sha256={str(f.relative_to(root)):sha(f) for f in data_files},auditor_sha256=sha(Path(__file__)))
    (root/'stream-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','resume_tps_ratio','cold_inclusive_tps_ratio','capacity')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);main(p.parse_args().root.resolve())
