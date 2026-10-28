"""Finish this finite suite with evidence hashes; never retry agent answers."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.run_dsh_daily_suite import save
from scripts.summarize_dsh_daily_suite import summarize


def run(root):
    deadline=time.time()+24*3600
    while time.time()<deadline:
        state=json.loads((root/'status.json').read_text())['status']
        if state=='needs_attention':
            summarize(root);return
        if state=='agent_runs_completed_pending_audit':
            reports=list((root/'cases').glob('*/browser/result.json'))
            retired=list((root/'session-retirement').glob('*.json')) if (root/'session-retirement').exists() else []
            if len(reports)==24 and len(retired)==24:break
        time.sleep(15)
    else:
        save(root/'execution-audit.json',dict(status='incomplete',reason='Finite 24-hour controller deadline'))
        return
    report=summarize(root);errors=[];c=report['counts']
    if c['completed_tasks']!=24 or c['tasks_over_36k']!=24:errors.append('Incomplete task/context cohort')
    if not all(row['minimum_followups_met'] for row in report['cases'].values()):errors.append('Missing follow-ups')
    if report['unexpected_policy']:errors.append('Unexpected recall policy')
    proof=json.loads((root/'runtime-proof.json').read_text())['files']
    for name,expected in proof.items():
        p=root/'runtime'/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append('Frozen source changed: '+name)
    hashes={}
    # Preserve raw native outputs and tool traces, not only summary numbers.
    for prefix in ('bridge','cases','session-retirement'):
        for p in (root/prefix).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl'):
                hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(root/'completed-evidence-hashes.json',hashes)
    save(root/'execution-audit.json',dict(status='passed' if not errors else 'failed',errors=errors,
        scope='Fixed configuration, independent attempts, >=4 human turns each, >36864 backend input tokens each, frozen runtime, explicit completed-owner retirement, raw evidence hashes. Browser defects and tool failures are quality outcomes.',
        cases=24,counts=c,at=time.time()))
    save(root/'status.json',dict(status='completed' if not errors else 'needs_attention',completed=c['completed_tasks'],
        expected=24,at=time.time(),counts=c))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
