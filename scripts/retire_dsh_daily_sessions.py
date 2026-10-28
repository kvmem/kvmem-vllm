"""Finite companion: retire only completed sessions declared by this experiment.

The native preserve-history contract deliberately requires explicit release;
an idle chat must remain reusable through its last follow-up. This process does
not change, restart, retry or cancel any task and never touches foreign owners.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def run(root):
    sys.path[:0]=[str(root/'runtime/scripts'),str(root/'runtime/src')]
    from graph_v2_protocol import release_owner
    plan=json.loads((root/'task-plan.json').read_text())
    profile=json.loads((root/'profile.json').read_text())
    output=root/'session-retirement';output.mkdir(exist_ok=True)
    deadline=time.time()+24*3600
    while time.time()<deadline:
        for row in plan['cases']:
            target=output/(row['id']+'.json')
            state=root/'cases'/row['id']/'status.json'
            if target.exists() or not state.exists():continue
            status=json.loads(state.read_text())
            if status.get('status')!='completed':continue
            owners=status['bridge']['owners'];proof=[]
            for owner in owners:
                d=root/'bridge'/owner
                first=next(d.glob('*/request.json'))
                body=json.loads(first.read_text())
                assert owner.startswith('daily-') and body['prompt_cache_key']==row['session_id']
                accepted=sorted(d.glob('*/accepted.json'))[-1]
                token=json.loads(accepted.read_text())['token_ids'][-1]
                proof.append(dict(owner=owner,result=release_owner(profile,root/'server/capture',owner,token)))
            target.write_text(json.dumps(dict(status='released',at=time.time(),session_id=row['session_id'],owners=proof),indent=2))
            print('released',row['id'],flush=True)
        state=json.loads((root/'status.json').read_text())['status']
        if state=='needs_attention' or all((output/(row['id']+'.json')).exists() for row in plan['cases']):break
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
