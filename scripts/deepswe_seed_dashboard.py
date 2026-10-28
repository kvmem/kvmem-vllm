"""Read-only aggregate view of four independent sixteen-task seed cohorts."""
import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from scripts.deepswe_dashboard import Dashboard, PAGE, read_json


class SeedDashboard:
    def __init__(self, root):
        self.root=root
        self.seeds=read_json(root/'manifest.json')['seeds']
        self.views={seed:Dashboard(root/f'seed-{seed}/first16') for seed in self.seeds}

    def snapshot(self):
        status=read_json(self.root/'status.json')
        active=status.get('active_seed')
        if active not in self.views:
            attempted=[s for s in self.seeds if (self.root/f'seed-{s}/first16/run.json').exists()]
            active=attempted[-1] if attempted else self.seeds[0]
        views={s:v.snapshot() for s,v in self.views.items()}
        result=copy.deepcopy(views[active])
        result['run_name']=self.root.name
        result['tasks']=[dict(row,id=f"seed {seed} · {row['id']}",seed=seed,task_id=row['id'])
                         for seed,view in views.items() for row in view['tasks']]
        for name in ['planned_tasks','completed_tasks','passed_tasks','context_limited_tasks','infrastructure_errors']:
            result[name]=sum(v[name] for v in views.values())
        result['seed_suite']=dict(active_seed=active,rounds=[dict(seed=seed,state=view['state'],
            scored=view['completed_tasks'],passed=view['passed_tasks']) for seed,view in views.items()])
        result['tool_counts']={k:sum(v['tool_counts'][k] for v in views.values())
                              for k in ['tool_calls','tool_results','tool_errors']}
        t=result['tool_counts'];t['error_rate']=t['tool_errors']/t['tool_results'] if t['tool_results'] else None
        result['output_tokens']=sum(v['output_tokens'] for v in views.values())
        result['requests']=sum(v['requests'] for v in views.values())
        result['elapsed_seconds']=sum(v['elapsed_seconds'] for v in views.values())
        result['e2e_output_rate']=result['output_tokens']/result['elapsed_seconds'] if result['elapsed_seconds'] else None
        if status.get('status') in ['completed','failed','waiting_for_gpu','prepared']:
            result['state']=status['status']
        result['at']=time.time()
        return result


def page():
    value=PAGE.read_text()
    value=value.replace('<h2>16 个样本</h2>','<h2>16 个样本 × 4 个 seed · 64 次测试</h2>')
    value=value.replace('题已评分','次已评分')
    marker=" if(d.state==='paused')notice="
    if value.count(marker)!=1:raise ValueError('Unexpected dashboard notice block')
    inserted=" if(d.seed_suite)notice=`固定 seed 1001 / 1002 / 1003 / 1004，共 64 次独立测试。当前吞吐图：seed ${d.seed_suite.active_seed}。`+d.seed_suite.rounds.map(r=>` seed ${r.seed}：已评分 ${r.scored}/16，通过 ${r.passed}。`).join('')+(d.state==='failed'?' 队列异常已停止，原始结果保留。':'');\n else if(d.state==='paused')notice="
    value=value.replace(marker,inserted)
    return value.encode()


def serve(root, port):
    view=SeedDashboard(root);html=page()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':body,kind=html,'text/html; charset=utf-8'
            elif path=='/api/status':body,kind=json.dumps(view.snapshot(),ensure_ascii=False,allow_nan=False).encode(),'application/json'
            elif path=='/health':body,kind=b'{"ok":true}','application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--port',type=int,default=50125);a=p.parse_args();serve(a.root.resolve(),a.port)
