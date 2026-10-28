"""Read-only live view for a single frozen Terminal-Bench job, no scheduler."""
import argparse
import hashlib
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time


def read(path, default=None):
    try:return json.loads(path.read_text())
    except (OSError, ValueError):return {} if default is None else default


def task_key(name):
    return name.removeprefix('terminal-bench/') if isinstance(name,str) else name


class JsonLines:
    def __init__(self, path):self.path=path;self.offset=0
    def new(self):
        try:
            with self.path.open('rb') as f:
                f.seek(self.offset)
                while True:
                    start=f.tell();line=f.readline()
                    if not line.endswith(b'\n'):break
                    self.offset=f.tell()
                    try:yield json.loads(line)
                    except ValueError:pass
        except FileNotFoundError:pass


class RequestStats:
    """Actual bridge token counts, keyed by the exact trial session."""
    def __init__(self, root):
        self.root = root
        self.requests=JsonLines(root/'bridge/requests.jsonl')
        self.owners={};self.counts=defaultdict(lambda:dict(tokens=0,requests=0,turn=0))
    def update(self):
        for e in self.requests.new():
            if e.get('event')=='session_opened':self.owners[e['owner']]=e['session_id']
            if 'output_tokens' in e:
                row=self.counts[e['owner']];row['tokens']+=e['output_tokens'];row['requests']+=1
                row.update(turn=e['turn'],context=e['prompt_tokens'],updated=e['at'])
        by_session={}
        for owner,key in self.owners.items():
            row=dict(self.counts[owner]);progress=read(self.root/'bridge'/owner/'progress.json')
            if progress.get('turn',0)>row['turn']:
                row['tokens']+=progress.get('accepted',0);row['requests']+=1
                row['context']=progress.get('prompt_tokens',0)
            by_session[key]=row
        return by_session


class State:
    def __init__(self,root,review_notes=None):
        self.root=root;self.value={};self.history=deque(maxlen=3600)
        self.review_notes={}
        if review_notes is not None:
            notes=read(review_notes)
            if notes.get('root')!=str(root.resolve()):raise ValueError('View review belongs to another run')
            for path,digest in notes['evidence'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise ValueError('View review evidence changed: '+path)
            self.review_notes=notes['tasks']
        self.agent_time_limit=read(root/'manifest.json').get('agent_time_limit','official')
        self.request_stats=RequestStats(root)
        self.preserved_stats={}
        self.events=JsonLines(root/'formal-events/events.jsonl')
        self.resources=JsonLines(root/'resources.jsonl');self.resource={}
        self.trials={}
    def update(self):
        now=time.time()
        for e in self.events.new():
            row=self.trials.setdefault(e['trial'],dict(task=task_key(e.get('task')),started=e['at']))
            row.update(stage=e['event'],updated=e['at'])
            if e.get('task'):row['task']=task_key(e['task'])
            if e['event']=='end':row['ended']=e['at']
        by_session=self.request_stats.update()
        for e in self.resources.new():self.resource=e
        outcomes=read(self.root/'formal-events/status.json').get('outcomes',[])
        preserved=read(self.root/'preserved-outcomes.json',[])
        outcomes=preserved+outcomes
        original={task_key(r['task']):r for r in outcomes}
        selected=[];annotations={}
        for row in outcomes:
            name=task_key(row['task']);note=self.review_notes.get(name)
            if note and row.get('status')=='error' and row.get('trial')==note['trial'] and row.get('exception')==note['exception']:
                annotations[name]=note
                if note['kind']=='verifier_review':
                    row={**row,'status':'scored','rewards':note['rewards'],
                         'passed':note['rewards']['reward']==1,'exception':None}
            selected.append(row)
        outcomes=selected
        completed={task_key(r['task']):r for r in outcomes}
        tasks=[];total=0
        for task in read(self.root/'tasks.json',[]):
            info=next(((key,value) for key,value in self.trials.items() if value['task']==task['name']),None)
            key,state=info if info else ('',{})
            stats=by_session.get(key+'__agent',{})
            # Only current formal sessions contribute to live throughput.
            total+=stats.get('tokens',0)
            old=next((r for r in preserved if task_key(r['task'])==task['name']),None)
            if old:
                source=old.get('source_root')
                if source and old.get('trial'):
                    if source not in self.preserved_stats:
                        self.preserved_stats[source]=RequestStats(Path(source)).update()
                    # Do not substitute another attempt of the same task.
                    stats=self.preserved_stats[source].get(old['trial']+'__agent',{})
                else:
                    stats={}
            tasks.append(dict(name=task['name'],stage=state.get('stage','waiting'),
                preserved=bool(old),
                elapsed=(state.get('ended',now)-state['started']) if state else (old.get('agent_seconds') or 0) if old else 0,
                **stats,outcome=completed.get(task['name']),
                original_outcome=original.get(task['name']),review_note=annotations.get(task['name'])))
        boundary=read(self.root/'formal-boundary.json').get('at')
        if boundary:
            self.history.append(dict(at=now,tokens=total,memory_available=self.resource.get('memory_available')))
        older=next((p for p in reversed(self.history) if p['at']<=now-30),None)
        rate=(total-older['tokens'])/(now-older['at']) if older else None
        archive=read(self.root/'model/server/capture/archive.json')
        self.value=dict(at=now,root=str(self.root),agent_time_limit=self.agent_time_limit,
            status=read(self.root/'status.json'),tasks=tasks,diagnostic_mode=read(self.root/'execution-mode.json').get('cuda_launch_blocking',False),
            total_tokens=total,rate30=rate,average=total/(now-boundary) if boundary else None,
            complete=len(completed),passed=sum(r['passed'] for r in outcomes),
            scored=sum(r.get('status')=='scored' for r in outcomes),
            pending_repairs=sum(n['kind']=='pending_setup' for n in annotations.values()),
            resource=self.resource,gc=archive.get('historical_gc',{}),history=list(self.history)[::5])
    def loop(self):
        while True:
            try:self.update()
            except Exception as e:self.value={'error':repr(e),'at':time.time()}
            time.sleep(2)


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Terminal-Bench · 实时实验</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1116;color:#dce6eb;font:15px system-ui,sans-serif}main{max-width:1640px;margin:auto;padding:32px}h1{font-size:28px;margin:0 0 12px}.sub{color:#8b9ea9;line-height:1.8}header{display:flex;justify-content:space-between;gap:20px}.tag{color:#67ddbd}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:28px 0}.card,section{background:#121d24;border:1px solid #22313b;border-radius:14px;padding:20px}.value{font-size:27px;margin-top:8px}.label{color:#8b9ea9;font-size:13px}section{margin-top:18px}canvas{width:100%;height:150px}table{width:100%;border-collapse:collapse;margin-top:16px}td,th{text-align:left;padding:15px 10px;border-bottom:1px solid #24323b}th{color:#8b9ea9;font-size:13px}.status{color:#77c8fa}.good{color:#67ddbd}.bad{color:#ff9e91}.muted{color:#899ca9}.context{white-space:nowrap;font-variant-numeric:tabular-nums}td:first-child{max-width:520px;overflow-wrap:anywhere}progress{width:100%;height:8px;accent-color:#61d6b5;margin-top:18px}.scroll{overflow:auto}@media(max-width:900px){main{padding:16px}.cards{grid-template-columns:repeat(2,1fr)}header{display:block}table{min-width:850px}}</style>
<main><header><div><h1>Terminal-Bench 2.1 <span class="tag">× Terminus 2</span></h1><div class="sub">89 个样本 · 最多 8 并发 · 增强执行器（已评分结果保留，故障尝试另存）<br>Qwen3.8-27B · NVFP4 / FP8 KV · 96K / 16K 滚动窗口 · 128K 单次输出 · 10M 逻辑上限 · LRU GC</div></div><div class="sub" id="phase">连接中…</div></header><progress id="bar" max="89" value="0"></progress>
<div class="cards"><div class="card"><div class="label">已评分 / 总任务</div><div class="value" id="done">—</div></div><div class="card"><div class="label">已通过</div><div class="value" id="passed">—</div></div><div class="card"><div class="label">近 30 秒接受输出 · token/s</div><div class="value" id="rate">—</div></div><div class="card"><div class="label">累计平均 · token/s</div><div class="value" id="avg">—</div></div><div class="card"><div class="label">CPU 可用内存 · GiB</div><div class="value" id="mem">—</div></div><div class="card"><div class="label">GPU 已用显存 · GiB</div><div class="value" id="gpu">—</div></div></div>
<section><div class="label">真实接受输出吞吐（30 秒窗口，包含工具执行和等待；排除预热、接口预检及 MTP 验证行）</div><canvas id="chart"></canvas><div class="sub" id="gc">GC 数据等待中</div></section>
<section><b>89 个样本</b><div class="sub" id="repairs"></div><span id="timing" class="sub" style="float:right">约 2 秒更新 · 原任务验证器</span><div class="scroll"><table><thead><tr><th>任务</th><th>当前状态</th><th>模型请求</th><th title="最近一次模型请求的完整输入 token 数，含历史消息与工具返回；不是物理 KV 占用">逻辑上下文 · tokens</th><th>接受输出</th><th>已耗时</th><th>评分 / 异常</th></tr></thead><tbody id="rows"></tbody></table></div></section><p class="sub">逻辑上下文显示最近一次模型请求的完整输入 token 数，包含历史消息与工具返回；已保留样本读取对应原始运行记录，尚未请求的样本不填估计值。该长度与 96K 物理 KV 预算不同。</p><p class="sub">任务未通过、任务超时和基础设施异常分别保留；不会自动重试或挑选结果。吞吐仅统计本轮新运行的任务；旧中断尝试保留。增强执行器与保留结果分别记录，不代表统一官方 harness 成绩。</p></main>
<script>const $=x=>document.getElementById(x),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),num=x=>x==null||!Number.isFinite(x)?'—':x.toLocaleString('en-US',{maximumFractionDigits:1}),tok=x=>x==null?'—':x>=1000?(x/1000).toFixed(1)+'K':x,st={waiting:'等待',start:'准备环境','environment-start':'启动容器','agent-start':'执行中','agent-end':'等待验证','verification-start':'验证中',end:'已结束',kv_released:'等待验证',waiting_for_memory:'等待内存'};
function plot(h){let c=$('chart'),d=devicePixelRatio||1,w=c.clientWidth,v=150;c.width=w*d;c.height=v*d;let x=c.getContext('2d');x.scale(d,d);let a=[];for(let i=1;i<h.length;i++){let j=i-1;while(j>0&&h[i].at-h[j].at<30)j--;let dt=h[i].at-h[j].at;if(dt>=10)a.push(Math.max(0,(h[i].tokens-h[j].tokens)/dt))}if(!a.length)return;let m=Math.max(100,...a);x.strokeStyle='#283943';x.beginPath();x.moveTo(40,125);x.lineTo(w,125);x.stroke();x.fillStyle='#8b9ea9';x.fillText(Math.round(m)+' tok/s',4,15);x.strokeStyle='#67ddbd';x.lineWidth=2;x.beginPath();a.forEach((r,i)=>{let xx=40+i*(w-45)/Math.max(1,a.length-1),yy=125-r/m*105;i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke()}
async function refresh(){try{let r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error(r.status);let d=await r.json();if(d.error)throw Error(d.error);$('timing').textContent='约 2 秒更新 · '+(d.agent_time_limit==='none'?'Agent 无时间上限 · 原验证器时限保留':'原任务时限与验证器');let s=d.status,ph={completed:'全部结束，汇总核验完成',superseded_before_inference:'切换为直连下载，已保留镜像',prepared:'已准备',preparing_images:'准备官方任务镜像',waiting_for_gpu:'等待 GPU 空闲',starting_model:'模型加载中',smoke:'Terminus 接口预检',formal:'正式测试运行中',needs_final_review:'全部结束，等待汇总核验',failed:'实验异常，保留现场'}; $('phase').textContent=(d.diagnostic_mode?'诊断模式 · 吞吐不与旧轮次比较 · ':'')+(ph[s.status]||s.status)+(s.status==='preparing_images'?' '+s.ready+'/89':'')+' · '+new Date(d.at*1000).toLocaleTimeString();$('bar').value=d.scored;$('done').textContent=d.scored+' / 89';$('passed').textContent=d.passed;$('repairs').textContent=d.pending_repairs?'有 '+d.pending_repairs+' 题保留依赖安装超时的旧记录；修复已验证，等待当前任务结束后补测。':'';$('rate').textContent=num(d.rate30);$('avg').textContent=num(d.average);$('mem').textContent=num(d.resource.memory_available/2**30);$('gpu').textContent=num(d.resource.gpu_memory_mib==null?null:d.resource.gpu_memory_mib/1024);let g=d.gc;$('gc').textContent='LRU GC · CPU 释放 '+num(g.cpu_released??g.cpu_blocks_released)+' 块 · 槽位释放 '+num(g.slots_released)+' · 策略错误 '+num(g.policy_failures);$('rows').innerHTML=d.tasks.map((t,i)=>{let o=t.outcome,n=t.review_note,done=o?(o.status==='scored'?'已评分':o.status==='agent_timeout'?'任务超时':'运行异常'):(st[t.stage]||t.stage),res=o?(o.rewards!=null?(o.passed?'通过':'未通过'):(o.exception||'—')):'—';if(n){done=n.kind==='pending_setup'?'修复已验证 · 待补测':'已补验 · 原答案';res=n.kind==='pending_setup'?'待补测':res;}return '<tr><td>'+String(i+1).padStart(2,'0')+' · '+esc(t.name)+'</td><td class="status">'+esc(done+(t.preserved?' · 保留原结果':''))+(n?'<div class="sub">'+esc(n.detail)+'</div>':'')+'</td><td>'+num(t.requests)+'</td><td class="context">'+(t.context==null?(t.preserved?'原记录缺失':t.stage==='waiting'||t.stage==='waiting_for_memory'?'尚未请求':'等待首个请求'):num(t.context))+'</td><td>'+tok(t.tokens)+'</td><td>'+Math.floor(t.elapsed/60)+'分 '+Math.floor(t.elapsed%60)+'秒</td><td class="'+(n?.kind==='pending_setup'?'muted':o?(o.passed?'good':'bad'):'muted')+'">'+esc(res)+'</td></tr>'}).join('');plot(d.history)}catch(e){$('phase').textContent='页面数据暂不可用：'+e.message}finally{setTimeout(refresh,2000)}}refresh();</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=50127)
    p.add_argument('--review-notes',type=Path)
    args=p.parse_args();state=State(args.root,args.review_notes);state.update()
    threading.Thread(target=state.loop,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0]=='/api/state':
                body=json.dumps(state.value,ensure_ascii=False).encode();kind='application/json'
            elif self.path.split('?')[0]=='/':body=HTML.encode();kind='text/html'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
