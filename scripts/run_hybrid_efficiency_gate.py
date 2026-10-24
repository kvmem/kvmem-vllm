"""One finite efficiency prerequisite, then the authorized full benchmark queue.

This is not a recurring monitor. Any pilot failure/regression stops the chain;
there are no retries, regenerated answers or quality-dependent decisions.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from benchmark_niah import save,sha
from graph_v2_protocol import require
from run_hybrid_suite_queue import completed,read

LIMITS=dict(graph_compute_max_seconds=1.,graph_selection_extra_max_seconds=1.,
    cold_median_ttft_ratio=1.15,warm_ttft_ratio=1.20,warm_ttft_slack_seconds=1.,
    scratch_limit_bytes=512*1024**2,cache_limit_bytes=12*1024**3,
    extra_missing_blocks=64)


def evaluate(cold,warm):
    checks=[]
    def check(name,ok,**values):checks.append(dict(name=name,passed=bool(ok),**values))
    require(cold['matched_inputs_seeds_sampling'] and warm['matched_inputs_seeds_sampling'],
        'Missing paired original-input proof')
    require(cold['no_accuracy_gate'] and warm['no_accuracy_gate'],'Accuracy cannot gate performance sampling')
    for label,report in [('cold',cold),('warm',warm)]:
        require(report['per_case'],'Empty efficiency cohort')
        for row in report['per_case']:
            a,b=row['off'],row['on'];ident=label+'/'+row['id']
            compute=sum(b['graph_branch_seconds'].values())
            check(ident+'/graph_core',compute<=LIMITS['graph_compute_max_seconds'],seconds=compute)
            extra=b['graph_selection_seconds']-a['graph_selection_seconds']
            check(ident+'/graph_and_picker_increment',extra<=LIMITS['graph_selection_extra_max_seconds'],seconds=extra)
            for arm,v in [('off',a),('on',b)]:
                check(ident+'/'+arm+'/bounded_scratch',type(v['scoring_scratch_bound_bytes']) is int and
                    0<v['scoring_scratch_bound_bytes']<=LIMITS['scratch_limit_bytes'],bytes=v['scoring_scratch_bound_bytes'])
                allocation=v.get('scoring_allocation_proof',{})
                cache_ok=(v['scoring_path']=='tiled_layer_cache' or
                    (v['scoring_path']=='dense_layer_cache' and
                     allocation.get('kind')=='derived_from_frozen_source_and_recorded_geometry' and
                     allocation.get('source_sha256')=='451e9fc3048823765471e7f91dd4e4ca45910327c8017305552493f421ef9b68'))
                check(ident+'/'+arm+'/layer_cache',cache_ok,path=v['scoring_path'],allocation_proof=allocation)
                check(ident+'/'+arm+'/bounded_cpu_cache',0<=v['cache_bytes']<=v['cache_capacity_bytes']==LIMITS['cache_limit_bytes'],
                    bytes=v['cache_bytes'],capacity_bytes=v['cache_capacity_bytes'])
            check(ident+'/no_bulk_rebuild',b['cache_rebuilt_blocks']<=a['cache_rebuilt_blocks']+LIMITS['extra_missing_blocks'],
                off_blocks=a['cache_rebuilt_blocks'],on_blocks=b['cache_rebuilt_blocks'])
            if b['prefix_reused']:
                allowed=a['ttft_seconds']*LIMITS['warm_ttft_ratio']+LIMITS['warm_ttft_slack_seconds']
                check(ident+'/warm_ttft',b['ttft_seconds']<=allowed,
                    off_seconds=a['ttft_seconds'],on_seconds=b['ttft_seconds'],allowed_seconds=allowed)
    medians={arm:statistics.median(v[arm]['ttft_seconds'] for v in cold['per_case']) for arm in ('off','on')}
    require(medians['off']>0 and all(math.isfinite(v) for v in medians.values()),'Invalid timing')
    check('cold_median_ttft',medians['on']/medians['off']<=LIMITS['cold_median_ttft_ratio'],
        off_seconds=medians['off'],on_seconds=medians['on'],ratio=medians['on']/medians['off'])
    require(all(not v['on']['prefix_reused'] for v in cold['per_case']) and
        sum(v['on']['prefix_reused'] for v in warm['per_case'])==1,'Cold/warm qualification lifecycle changed')
    return dict(status='passed' if all(c['passed'] for c in checks) else 'blocked_performance_review',
        limits=LIMITS,checks=checks,no_accuracy_gate=True,paired_cases=cold['cases']+warm['cases'],
        interpretation='A bounded single-pass regression screen, not a statistically significant speedup claim. '
            'Decode/output/MTP differences are retained separately. Known long-Query scoring cost remains in both arms.')


def run(root,efficiency):
    manifest=read(root/'efficiency-gate-plan.json')
    require(manifest['limits']==LIMITS and manifest['efficiency_root']==str(efficiency), 'Frozen efficiency gate changed')
    from run_ruler_hybrid_campaign import verify_files
    verify_files(root,manifest['files'])
    require(sha(root/'queue-plan.json')==manifest['full_queue_plan_sha256'],'Full queue changed during preflight')
    require(not (root/'queue-process.json').exists(),'Never duplicate a full queue')
    began=time.monotonic()
    while read(efficiency/'status.json')['status'] not in ('completed','failed'):
        process=read(efficiency/'queue-process.json');p=Path('/proc')/str(process['pid'])
        require(p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X'),
            'Efficiency queue exited without completion')
        require((p/'cmdline').read_bytes().decode().split('\0')[:-1]==process['command'],
            'Efficiency queue process identity changed')
        require(time.monotonic()-began<3*3600,'Efficiency prerequisite timed out; preserve attempts')
        time.sleep(5)
    require(read(efficiency/'status.json')['status']=='completed','Efficiency pilot failed; full suite remains blocked')
    for name in ('cold/graph-off','cold/graph-on','warm/graph-off','warm/graph-on'):completed(efficiency/name)
    from analyze_hybrid_efficiency import compare
    cold,warm=compare(efficiency/'cold'),compare(efficiency/'warm')
    verdict=evaluate(cold,warm)
    verdict.update(cold_report_sha256=sha(efficiency/'cold/comparison.json'),
        warm_report_sha256=sha(efficiency/'warm/comparison.json'),at=time.time())
    save(root/'efficiency-gate.json',verdict)
    require(verdict['status']=='passed','Efficiency regression requires investigation; full suite remains blocked')
    verify_files(root,manifest['files'])
    require(sha(root/'queue-plan.json')==manifest['full_queue_plan_sha256'],'Full queue changed after preflight')
    command=[sys.executable,'-B',str(root/'ruler/runtime/scripts/run_hybrid_suite_queue.py'),'--root',str(root)]
    with (root/'queue.log').open('x') as log:
        child=subprocess.Popen(command,cwd=root/'ruler/runtime',
            env=dict(os.environ,PYTHONPATH=str(root/'ruler/runtime/src'),PYTHONDONTWRITEBYTECODE='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'queue-process.json',dict(pid=child.pid,command=command,at=time.time(),
        efficiency_gate_sha256=sha(root/'efficiency-gate.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--efficiency',type=Path,required=True)
    args=p.parse_args()
    try:run(args.root.resolve(),args.efficiency.resolve())
    except BaseException as error:
        save(args.root/'efficiency-gate-failure.json',dict(error=repr(error),at=time.time()))
        save(args.root/'status.json',dict(status='blocked_efficiency_review',error=repr(error),full_campaigns_started=False,at=time.time()))
        raise
