import json
from scripts.analyze_host_interference import union,analyze


def test_union_does_not_count_nested_or_overlapping_waits_twice():
    assert union([(0,5),(1,3),(4,8),(10,11)])==9
    assert union([])==0


def test_exclusive_time_subtracts_only_direct_same_thread_children(tmp_path):
    def row(span,parent,thread,start,end,cpu):
        return dict(span=span,parent=parent,thread=thread,started_at=start,finished_at=end,
            wall_seconds=end-start,thread_cpu_seconds=cpu,stage=str(span))
    rows=[row(1,None,1,0,10,5),row(2,1,1,1,6,3),row(3,2,1,2,4,2),row(4,None,2,0,10,9)]
    p=tmp_path/'trace';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    r=analyze(p)['stages_ms']
    assert r['1']['exclusive_wall_seconds']['sum']==5000
    assert r['1']['exclusive_cpu_seconds']['sum']==2000
    assert r['2']['exclusive_wall_seconds']['sum']==3000
