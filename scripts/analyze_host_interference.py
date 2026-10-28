"""Attribute nested host spans without counting inclusive time twice."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def union(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('Reversed interval')
        total+=max(0.,b-max(a,end if end is not None else a))
        end=max(end if end is not None else b,b)
    return total


def distribution(values):
    values=sorted(values)
    if not values:return {'count':0}
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
        p95=values[min(len(values)-1,int(.95*len(values)))],max=values[-1],sum=sum(values))


def read_rows(path):
    # Only a complete newline-delimited prefix is safe while a writer runs.
    data=path.read_bytes();end=data.rfind(b'\n')
    return [json.loads(line) for line in data[:end+1].splitlines()] if end>=0 else []


def analyze(path):
    rows=read_rows(path)
    children=defaultdict(list);stages=defaultdict(list)
    for r in rows:children[r['parent']].append(r)
    for r in rows:
        own=[c for c in children[r['span']] if c['thread']==r['thread']]
        child_wall=union((max(c['started_at'],r['started_at']),min(c['finished_at'],r['finished_at'])) for c in own)
        r['exclusive_wall_seconds']=max(0.,r['wall_seconds']-child_wall)
        r['exclusive_cpu_seconds']=max(0.,r['thread_cpu_seconds']-sum(c['thread_cpu_seconds'] for c in own))
        stages[r['stage']].append(r)
    summary={k:{f:distribution(1000*r[f] for r in values) for f in
        ('wall_seconds','thread_cpu_seconds','exclusive_wall_seconds','exclusive_cpu_seconds')}
        for k,values in stages.items()}
    return dict(stages_ms=summary,slowest=sorted(rows,key=lambda r:r['exclusive_wall_seconds'],reverse=True)[:30],
        scope='Host-only diagnostic, milliseconds. Direct-child interval union removed for exclusive wall. '
              'Thread CPU excludes other threads. Waiting/descheduling is not necessarily GIL wait or GPU idle.')


def maintenance_overlap(path, steps):
    rows=read_rows(path)
    # These intervals end at the host's public model submission, not at a
    # later CUDA fence wait. Post-forward capture waits must not be mistaken
    # for preparation that prevented the target forward from being submitted.
    output=[]
    for step in read_rows(steps):
        if step.get('maintenance_host_ms',0)<50:continue
        start,end=step['started_at'],step['model_submitted_at']
        matches=[]
        for r in rows:
            overlap=min(end,r['finished_at'])-max(start,r['started_at'])
            if overlap>0:
                matches.append(dict(stage=r['stage'],span=r['span'],parent=r['parent'],
                    thread=r['thread'],overlap_ms=overlap*1000,
                    inclusive_cpu_ms=r['thread_cpu_seconds']*1000))
        output.append(dict(sequence=step['sequence'],maintenance_ms=step['maintenance_host_ms'],
            prefill_tokens=step['prefill_tokens'],decode_requests=step['decode_requests'],
            spans=sorted(matches,key=lambda r:r['overlap_ms'],reverse=True)))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=Path)
    a=p.parse_args();value=analyze(a.trace)
    if a.steps:value['maintenance_overlap']=maintenance_overlap(a.trace,a.steps)
    a.output.write_text(json.dumps(value,indent=2)+'\n')
