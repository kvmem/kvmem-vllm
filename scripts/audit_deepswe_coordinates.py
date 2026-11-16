"""CPU-only replay of saved canonical-fallback requests and exact planned spans."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from vllm_kvmem.core.request_records import load_request


def audit(run,cases):
    from transformers import AutoTokenizer
    spec=importlib.util.spec_from_file_location('coordinate_audit_bridge',
        Path(__file__).resolve().parents[1]/'benchmarks/deepswe/claude_bridge.py')
    bridge=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bridge;spec.loader.exec_module(bridge)
    profile=json.loads((run/'profile.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    planner=bridge.Planner(bridge.Protocol(run/'protocol/protocol.so'),tokenizer,profile['kvmem']['gen_budget'])
    starts={}
    for line in (run/'bridge/requests.jsonl').read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row['event']=='start':starts[row['task'],row['turn']]=row
    results=[]
    for name in cases:
        task,number=name.rsplit(':',1);turn=int(number)
        old=starts[task,turn]
        if old['exact_output_prefix_reused']:raise ValueError('Select an original canonical-fallback turn')
        directory=run/'bridge'/task/f'{turn:05d}'
        payload=load_request(directory/'request.json')
        reference=json.loads((directory/'plan.json').read_text())
        begin=time.perf_counter()
        tokens,plan,rendered,_,_,reused,locations=planner.prepare(payload,bridge.Trajectory())
        seconds=time.perf_counter()-begin
        digest=hashlib.sha256(bridge.canonical(tokens).encode()).hexdigest()
        expected={k:v for k,v in reference.items() if k.startswith('kvmem_')}
        checks=dict(token_ids_equal=digest==reference['token_sha256'],
            token_count_equal=len(tokens)==reference['prompt_tokens'],query_and_mandatory_equal=plan==expected,
            canonical_fallback=reused is False)
        # Independently encode early/middle/late prefixes. Exhaustive Unicode,
        # partial-delimiter and BPE boundary checks live in the unit suite.
        spans=[s for s in rendered['spans'] if s['role']!='assistant']
        text=rendered['prompt'].encode();checked=0
        for index in sorted({0,len(spans)//2,len(spans)-1}):
            if index<0:continue
            span=spans[index]
            for key in ('content_begin','content_end'):
                point=span[key]
                location=(-1,point) if point<=rendered['control_end'] else next(
                    (s['index'],point-s['begin']) for s in rendered['spans'] if s['begin']<=point<=s['end'])
                if locations[location]!=len(tokenizer.encode(text[:point].decode(),add_special_tokens=False)):
                    raise AssertionError('Independent original prefix coordinate differs')
                checked+=1
        results.append(dict(task=task,turn=turn,prompt_tokens=len(tokens),locations=len(locations),
            independent_prefix_checks=checked,checks=checks,replay_prepare_seconds=seconds,
            recorded_bridge_prepare_seconds=old['at']-old['request_started_at']-old['cpu_queue_seconds']))
        if not all(checks.values()):raise AssertionError('Original tokens or semantic plan changed: '+name)
    return dict(passed=True,results=results,
        scope='CPU replay against saved original token hashes and complete Query/mandatory plans. '
              'Recorded bridge time and replay time have different host load; not a paired GPU throughput result.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.run,args.cases)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
