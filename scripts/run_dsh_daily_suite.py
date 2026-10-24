"""Finite, isolated DSH agent workload; retain first attempts and exact event logs.

This is an experiment controller, not a recurring monitor. Each repetition has a
new directory/session; follow-ups retain that session. No model answer is retried
on quality grounds. GPU work is performed by the separately recorded service.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid

PROVIDER = 'kvmem-hybrid-5050'
MODEL = 'kvmem-dsh-hybrid-mtp4'
THRESHOLD = 36864

TASKS = [
 ('fps', '帮我写一个网页版的 CSGO 风格单人游戏。实现第一人称移动、鼠标瞄准、射击和换弹、机器人敌人、生命值、得分、回合结束与重新开始，并做一个可玩的地图。',
  ['在这个游戏中添加更多功能：至少三种差异明显的武器、购买菜单、护甲和机器人难度选择，保留原来的移动和射击功能。',
   '添加更多地图：至少三个布局不同的地图，支持地图选择；补充掩体、出生点与小地图，检查碰撞和机器人不会卡在墙里。',
   '继续完善游戏：加入拆弹或占点目标、胜负判定、暂停、音量设置和本地最佳成绩。检查多次重开、切换地图和购买武器后的问题并修复。']),
 ('tower', '帮我写一个网页版塔防游戏。需要可视化路线、至少两种防御塔、敌人波次、金币、生命值、暂停和重开，能直接玩。',
  ['增加升级和出售防御塔、三种敌人以及难度选择，说明各塔和敌人的差异。',
   '添加三个不同地图和路线编辑预览，支持选择关卡，检查敌人寻路和塔的攻击范围。',
   '加入关卡解锁、本地存档和音量设置；测试暂停后继续、失败重开和重复加载存档，修复发现的问题。']),
 ('snake', '帮我做一个漂亮的网页版贪吃蛇游戏。要有键盘控制、碰撞、计分、速度变化、暂停和重新开始，手机上也能操作。',
  ['增加特殊食物、障碍物、三种难度和可切换皮肤，保持经典模式可用。',
   '添加至少三个地图模式，包括穿墙、迷宫和动态障碍；增加地图选择和规则说明。',
   '加入本地排行榜、游戏设置持久化、连吃奖励和统计面板。测试快速输入、暂停重开和地图切换，修复边界问题。']),
 ('kanban', '帮我做一个网页版项目看板。支持新增、编辑和删除任务，待办/进行中/完成三列，拖动任务、截止日期和浏览器本地保存。',
  ['增加标签、优先级、负责人、搜索与多条件筛选，添加任务详情和检查清单。',
   '增加多个项目切换、看板与列表视图、JSON 导入导出；导入格式错误时给出提示，不能破坏已有数据。',
   '完善批量操作、撤销删除、逾期提醒和统计图。测试刷新后恢复、空项目、重复导入和筛选后拖动，修复问题。']),
 ('expense', '帮我做一个网页版个人记账工具。支持收入和支出、分类、日期、备注、编辑删除、月度统计和本地保存。',
  ['增加多个账户、账户转账、分类预算和超预算提示，添加日期及分类筛选。',
   '增加 CSV 导入导出、月度和年度图表、重复账单模板；验证非法金额和重复导入不会污染账本。',
   '增加分类管理、批量删除与撤销、备份恢复、移动端布局。检查小数金额、跨月数据和空账本并修复。']),
 ('calendar', '帮我做一个网页版日历和日程管理工具。支持月视图、新增编辑删除日程、日期和时间、颜色分类及本地保存。',
  ['增加周视图和日视图、全天事件、拖动改期以及日程搜索，保留月视图。',
   '增加重复日程、结束日期、冲突提示和导入导出，正确处理跨天事件。',
   '增加待办清单、今日概览、键盘操作和移动端布局；测试月末跨月、闰年、重复事件删除和刷新恢复，修复问题。']),
 ('notes', '帮我做一个网页版笔记应用。支持新建编辑删除笔记、Markdown 编辑和预览、标题搜索、侧边栏和本地保存。',
  ['增加文件夹、标签、收藏和回收站，支持排序与跨文件夹搜索。',
   '增加多笔记导入导出、分屏预览、编辑历史和字数统计，处理导入冲突和非法数据。',
   '增加待办列表、主题切换、自动保存状态和键盘快捷键；检查空笔记、大文档、撤销删除及刷新恢复，修复问题。']),
 ('pomodoro', '帮我做一个网页版番茄钟和学习计划工具。支持工作/休息倒计时、开始暂停重置、任务列表、每日完成次数和本地保存。',
  ['增加自定义时长、长休息、声音开关、自动切换和任务预计番茄数，检查计时状态。',
   '增加周计划、日历热力图、统计图和数据导出，支持多个学习项目。',
   '增加历史记录编辑、设置备份恢复、主题和移动端适配；测试切后台、暂停后刷新、跨日统计和极短倒计时，修复问题。']),
]


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rpc(url, method, payload):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = dict(type='client-request', rpcId=str(uuid.uuid4()), method=method, payload=payload)
    request = urllib.request.Request(url + '/api/' + method, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with op.open(request, timeout=60) as response:
        result = json.load(response)['result']
    if not result['ok']:
        raise RuntimeError(str(result))
    return result['value']


def history_since(url, session, last_seq):
    """Page backwards until the durable boundary; never truncate long tool turns."""
    payload = dict(sessionId=session, maxMessages=80)
    result, projection = {}, None
    while True:
        page = rpc(url, 'session.history', payload)
        if projection is None:
            projection = page.get('projections')
        events = [row['event'] for row in page['events']]
        for event in events:
            if event['seq'] > last_seq:
                result[event['seq']] = event
        if not events or not page.get('hasMore') or min(e['seq'] for e in events) <= last_seq:
            break
        before = min(e['seq'] for e in events)
        if 'beforeSeq' in payload and before >= payload['beforeSeq']:
            raise RuntimeError('History cursor did not advance')
        payload['beforeSeq'] = before
    return sorted(result.values(), key=lambda e: e['seq']), projection


def event_statistics(events):
    calls, results, errors, endings, requests = [], [], [], [], []
    call_names, command_failures = {}, []
    steps = 0
    for event in events:
        kind, data = event['type'], event.get('data') or {}
        if kind == 'tool/call':
            call_names[data.get('callId')] = data.get('name')
            args = data.get('arguments')
            try:
                if isinstance(args, str): args = json.loads(args)
            except ValueError: pass
            calls.append((data.get('name'), json.dumps(args, sort_keys=True, ensure_ascii=False)))
        if kind == 'tool/result':
            parts = data.get('message', {}).get('content', [])
            results.append(event['seq'])
            if any(p.get('isError') for p in parts):
                errors.append(event['seq'])
            for part in parts:
                if call_names.get(part.get('toolCallId')) != 'bash': continue
                body = '\n'.join(c.get('text', '') for c in part.get('content', []))
                codes = [int(c) for c in re.findall(r'\[exit code: (-?\d+)\]', body)]
                if any(code != 0 for code in codes):
                    command_failures.append(event['seq']); break
        if kind == 'turn/end': endings.append(dict(seq=event['seq'], reason=data.get('reason')))
        if kind == 'step/start': steps += 1
        if 'approval' in kind or 'question' in kind: requests.append(event['seq'])
    counts = Counter(calls)
    return dict(model_steps=steps, tool_calls=len(calls), tool_results=len(results),
        tool_error_results=len(errors), tool_error_event_seqs=errors,
        bash_nonzero_exit_results=len(command_failures), bash_nonzero_event_seqs=command_failures,
        tool_or_command_failure_results=len(set(errors) | set(command_failures)),
        repeated_identical_calls=sum(n - 1 for n in counts.values()),
        repeated_call_groups=[dict(name=k[0],count=n) for k,n in counts.items() if n>1],
        tool_result_error_rate=(len(errors)/len(results) if results else None),
        tool_or_command_failure_rate=(len(set(errors) | set(command_failures))/len(results) if results else None),
        turn_endings=endings, approval_or_question_events=requests)


def bridge_observations(root, session):
    """Use backend tokenization, not UI estimates, to check the logical input."""
    owners = []
    for directory in sorted((root/'bridge').glob('daily-*')):
        first = next(directory.glob('[0-9][0-9][0-9][0-9][0-9]/request.json'), None)
        if first is None: continue
        body = json.loads(first.read_text())
        if body.get('prompt_cache_key') == session:
            owners.append(directory.name)
    rows = []
    log = root/'bridge/requests.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            try: row = json.loads(line)
            except ValueError: continue  # An in-flight final write is re-read next poll.
            if row.get('owner') in owners: rows.append(row)
    agent_owners = []
    for owner in owners:
        first = next((root/'bridge'/owner).glob('[0-9][0-9][0-9][0-9][0-9]/request.json'))
        if json.loads(first.read_text()).get('tools'): agent_owners.append(owner)
    agent_rows = [r for r in rows if r.get('owner') in agent_owners]
    return dict(owners=owners, agent_owners=agent_owners,
        completed_model_requests=sum('prompt_tokens' in r and 'error' not in r for r in agent_rows),
        errors=[r for r in rows if 'error' in r],
        max_logical_input_tokens=max([r.get('prompt_tokens', 0) for r in agent_rows] or [0]),
        accepted_output_tokens=sum(r.get('output_tokens', 0) for r in agent_rows),
        segmented_model_requests=sum(r.get('native_segments', 0)>1 for r in agent_rows),
        exact_prefix_reused_requests=sum(bool(r.get('exact_output_prefix_reused')) for r in agent_rows))


def workspace_manifest(workspace):
    return {str(p.relative_to(workspace)): dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in workspace.rglob('*') if p.is_file() and not p.is_symlink()
        and not any(x in ('node_modules','.git') for x in p.relative_to(workspace).parts)}


def make_plan(root):
    suffix = ('\n请直接在当前独立工作目录实现，入口为 index.html，用 HTML/CSS/JavaScript 和浏览器原生能力，'
        '资源全部本地可用，不依赖 CDN、网络服务或安装新依赖。需要的简化请自行决定并说明。'
        '只修改当前目录，不访问其他任务目录、不使用子代理、不提交 Git。'
        '不需要常驻预览服务；完成后给出操作说明和验证结果，保留 README。')
    cases=[]
    for rep in range(1,4):
        for index,(name,initial,followups) in enumerate(TASKS):
            ident=f'r{rep:02d}-{name}'
            cases.append(dict(id=ident, repetition=rep, task=name,
                session_id='kvmem-daily-'+ident+'-20261024-r1',
                workspace=str((root/'workspaces'/ident).resolve()), preview_port=19080+(rep-1)*8+index,
                prompts=[initial+suffix]+[p+' 请直接在当前项目上修改并验证，不要只给建议；保留此前功能。' for p in followups]))
    return dict(cases=cases,parallel_sessions=8,repetitions=3,minimum_turns=4,
        minimum_logical_input_tokens_exclusive=THRESHOLD,max_turns=10,turn_timeout_seconds=3600,
        max_steps_per_turn=240,provider=PROVIDER,model=MODEL,reasoning='medium',
        sampling='DSH defaults; adapter defaults temperature=.6/top_p=.95/top_k=20/seed=1000; exact bodies archived.',
        no_quality_retries=True,independent_workspaces=True,
        additional_followups='Real review/test/feature requirements only; no padding or fabricated tool output')


def extra_prompt(number, task):
    additions=[
        '请为当前项目补充可重复运行的本地功能测试和测试说明，覆盖主要流程与关键边界。先检查现有实现，运行能运行的检查，修复发现的问题，再给出实际结果。',
        '请检查各文件中的状态管理和事件处理，补充键盘可操作性、清晰的错误提示、窄屏布局，并修复重复初始化或重置可能导致的状态错误。',
        '请新增一个适合当前应用的高级功能和相应设置界面，自己选择有实际用途的功能；保持已有数据兼容，更新说明并验证旧功能。',
        '请对当前实现做一次完整回归检查：检查持久化数据、重置与恢复、视图切换以及所有主要按钮，补充缺失测试，修复问题并更新使用说明。',
        '请完善帮助面板和演示模式，加入可重复加载的示例数据或关卡，保证演示不会覆盖已有存档。检查导入导出或存档流程并修复问题。',
        '请阅读当前项目中与主要流程有关的实现并做最后一轮质量检查，改进性能和结构，补充边界测试及可复现步骤。不要重建项目或删除已有功能。',
    ]
    return additions[number-4]+' 所有修改仍限当前独立目录，不启动常驻服务。'


def run_case(root, url, row, barrier, plan):
    case=root/'cases'/row['id']; case.mkdir(exist_ok=False)
    workspace=Path(row['workspace']); workspace.mkdir(exist_ok=False)
    session=row['session_id']; events=[]; last_seq=-1
    status=dict(status='preparing',session_id=session,workspace=str(workspace),turns_completed=0)
    save(case/'status.json',status)
    created=rpc(url,'session.create',dict(sessionId=session,cwd=str(workspace)))
    assert created['sessionId']==session
    rpc(url,'session.selectModel',dict(sessionId=session,provider=PROVIDER,model=MODEL,reasoningEffort='medium'))
    models=rpc(url,'session.models',dict(sessionId=session))
    assert models['routable'] and models['current']['provider']==PROVIDER
    save(case/'selection.json',dict(current=models['current'],routable=models['routable']))
    rpc(url,'session.rename',dict(sessionId=session,title='50:50 无图 · '+row['id']))
    barrier.wait(timeout=180)
    for turn in range(plan['max_turns']):
        prompt=row['prompts'][turn] if turn<4 else extra_prompt(turn,row['task'])
        td=case/f'turn-{turn+1:02d}'; td.mkdir(exist_ok=False)
        request=dict(sessionId=session,mode='queue',clientTimeZone='Asia/Shanghai',content=[dict(type='text',text=prompt)])
        # Persist before mutation; if the response is lost, never blindly resubmit.
        save(td/'request.json',request)
        before,projection=history_since(url,session,last_seq)
        if before:
            events.extend(before);last_seq=max(e['seq'] for e in before)
            with (case/'events.jsonl').open('a') as f:
                for e in before:f.write(json.dumps(e,ensure_ascii=False)+'\n')
        boundary=last_seq; began=time.time();save(td/'started.json',dict(at=began,before_seq=boundary))
        accepted=rpc(url,'session.prompt',request);save(td/'acknowledgement.json',accepted)
        status.update(status='running',turn=turn+1,turn_started_at=began)
        save(case/'status.json',status)
        terminal=None
        while terminal is None:
            new,projection=history_since(url,session,last_seq)
            if new:
                events.extend(new);last_seq=max(e['seq'] for e in new)
                with (case/'events.jsonl').open('a') as f:
                    for e in new:f.write(json.dumps(e,ensure_ascii=False)+'\n')
                save(case/'projections.json',projection)
            current=[e for e in events if e['seq']>boundary]
            endings=[e for e in current if e['type']=='turn/end']
            if endings:terminal=endings[-1];break
            steps=sum(e['type']=='step/start' for e in current)
            if time.time()-began>plan['turn_timeout_seconds'] or steps>=plan['max_steps_per_turn']:
                save(td/'limit.json',dict(at=time.time(),steps=steps,elapsed=time.time()-began))
                rpc(url,'session.cancel',dict(sessionId=session))
                raise RuntimeError('Predeclared turn resource limit; original attempt retained')
            time.sleep(3)
        stats=event_statistics(current);bridge=bridge_observations(root,session)
        save(td/'result.json',dict(elapsed_seconds=time.time()-began,statistics=stats,
            bridge=bridge,terminal=terminal,workspace_files=workspace_manifest(workspace)))
        status.update(turns_completed=turn+1,max_logical_input_tokens=bridge['max_logical_input_tokens'],
            statistics=event_statistics(events),bridge=bridge,at=time.time())
        save(case/'status.json',status)
        reason=terminal.get('data',{}).get('reason',{})
        if reason.get('kind') not in ('completed','complete','done','stop','end-turn'):
            # Acceptable reason vocabulary is verified against the smoke before suite launch.
            raise RuntimeError('Agent ended abnormally: '+str(reason))
        if turn+1>=plan['minimum_turns'] and bridge['max_logical_input_tokens']>THRESHOLD:
            status.update(status='completed',completed_at=time.time(),context_requirement_passed=True)
            save(case/'status.json',status); return status
    raise RuntimeError('Context length requirement unmet after all declared follow-ups')


def run(root,url):
    plan=json.loads((root/'task-plan.json').read_text())
    assert not (root/'suite-started.json').exists(), 'Never blindly restart attempted suite'
    save(root/'suite-started.json',dict(at=time.time(),plan_sha256=digest(root/'task-plan.json')))
    results={}
    for rep in range(1,4):
        rows=[r for r in plan['cases'] if r['repetition']==rep];barrier=threading.Barrier(8)
        save(root/'status.json',dict(status='testing',active_repetition=rep,completed=len(results),expected=24,at=time.time()))
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs={pool.submit(run_case,root,url,row,barrier,plan):row for row in rows}
            for job in as_completed(jobs):
                row=jobs[job]
                try:results[row['id']]=job.result()
                except Exception as error:
                    result=dict(status='failed',error=repr(error),at=time.time())
                    save(root/'cases'/row['id']/'failure.json',result);results[row['id']]=result
                save(root/'summary.json',dict(results=results,completed=sum(v['status']=='completed' for v in results.values()),
                    failed=sum(v['status']=='failed' for v in results.values()),expected=24,at=time.time()))
        if any(results[r['id']]['status']!='completed' for r in rows):
            save(root/'status.json',dict(status='needs_attention',active_repetition=rep,at=time.time()))
            return
    save(root/'status.json',dict(status='agent_runs_completed_pending_audit',completed=24,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:3080')
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        assert not (args.root/'task-plan.json').exists()
        save(args.root/'task-plan.json',make_plan(args.root))
    else:run(args.root,args.url)
