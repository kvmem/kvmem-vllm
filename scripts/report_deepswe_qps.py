"""Render completed paired DeepSWE cells without substituting historical results."""
import argparse
import csv
import json
from pathlib import Path


def latency_from_events(root,cell):
    # Frozen early summaries measured admission-to-finish. Recompute the full
    # bridge span from immutable raw events, without editing carried evidence.
    directory=root/cell['name']
    boundary=read(directory/'window-end-metrics.json')
    data=(directory/'bridge/requests.jsonl').read_bytes()[:boundary['bridge_bytes']]
    rows=[json.loads(line) for line in data[:data.rfind(b'\n')+1].splitlines()]
    finishes=[r for r in rows if r.get('event')=='finish' and cell['begin']<=r['at']<=cell['end']]
    values=sorted(r['seconds'] for r in finishes)
    if not values:return None
    position=.95*(len(values)-1);index=int(position)
    return values[index]+(values[min(index+1,len(values)-1)]-values[index])*(position-index)


def read(path):
    return json.loads(path.read_text())


def report(root, replay=None):
    summary=read(root/'summary.json');groups={}
    for cell in summary['cells'].values():
        groups.setdefault((cell['concurrency'],cell['backend']),[]).append(cell)
    rows=[]
    for (concurrency,backend),cells in sorted(groups.items()):
        seconds=sum(r['seconds'] for r in cells)
        rows.append(dict(concurrency=concurrency,backend=backend,repeats=len(cells),seconds=seconds,
            qps=sum(r['completed_model_requests'] for r in cells)/seconds,
            accepted_output_tps=sum(r['accepted_output_tokens'] for r in cells)/seconds,
            delivered_output_tps=sum(r['completed_request_output_tokens'] for r in cells)/seconds,
            completed_requests=sum(r['completed_model_requests'] for r in cells),
            mean_delivered_tokens_per_completed_request=(
                sum(r['completed_request_output_tokens'] for r in cells)/sum(r['completed_model_requests'] for r in cells)
                if sum(r['completed_model_requests'] for r in cells) else None),
            outstanding_at_boundaries=sum(r['outstanding_at_boundary'] for r in cells),
            ttft_p95_ms_per_repeat=[r['request_metrics']['time_to_first_token_ms']['p95'] for r in cells],
            bridge_end_to_end_p95_seconds_per_repeat=[latency_from_events(root,r) for r in cells],
            preemptions=sum(r['native']['preemptions'] or 0 for r in cells),
            api_errors=sum(len(r['errors']) for r in cells),
            tool_calls=sum(r['tool_calls'] for r in cells),
            tool_errors=sum(r['tool_errors'] for r in cells),
            tasks_observed=sorted({name for r in cells for name in r['tasks']}),
            official_results_per_repeat=[r['official_results'] for r in cells],
            context_mean_per_repeat=[r['context_tokens']['mean'] for r in cells],
            max_prompt_tokens=max(r['context_tokens']['maximum'] or 0 for r in cells),
            qps_per_repeat=[r['qps'] for r in cells],
            accepted_output_tps_per_repeat=[r['agent_tokens_per_second'] for r in cells]))
    (root/'aggregate.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (root/'aggregate.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    indexed={(r['concurrency'],r['backend']):r for r in rows}
    gains=[]
    for concurrency in sorted({r['concurrency'] for r in rows}):
        native=indexed.get((concurrency,'native'));kvmem=indexed.get((concurrency,'kvmem'))
        if native and kvmem and native['repeats']==kvmem['repeats']:
            gains.append(dict(concurrency=concurrency,
                qps_gain_percent=100*(kvmem['qps']/native['qps']-1),
                output_tps_gain_percent=100*(kvmem['accepted_output_tps']/native['accepted_output_tps']-1)))
    (root/'aggregate-gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    has_replay=replay is not None and (replay/'summary.json').exists()
    text=['# 256K 上限：真实 DeepSWE 长历史 8 并发' if has_replay else
          '# 256K 上下文：真实 DeepSWE 性能对照','',
          '单张 RTX PRO 6000 Blackwell 96GB；同一 Qwen3.8-27B NVFP4 权重、FP8 KV、MTP4、',
          'CUDA Graph 和 vLLM 0.30.0。两组 max_model_len=262144、max_num_seqs=8、',
          'gpu_memory_utilization=0.72；KVMem 使用 K98304/G16384、100 GB CPU 历史预算，',
          '关闭历史 GC。详细冻结参数、代码和输入身份见 [manifest.json](manifest.json)。','',
          '## 补充：真实 agent 短窗口','',
          f"已完成 {len(summary['cells'])}/{summary['expected']} 个测量窗口。",'',
          '从原始 16 题池按固定顺序启动并发 1/2/4/8 的真实 DeepSWE agent，',
          '运行真实仓库和工具，每组 300 秒，两次重复使用种子 1000/1001，并反转后端顺序。',
          '加载、编译、共同预热、窗口后的容器保存均不计时。实际展开的题目见 aggregate.json。','',
          'QPS 是模型回合/s；token/s 是包含工具等待的整窗接受输出率。各重复先独立测量，再按实际墙钟汇总。', '',
          '| 并发 | 后端 | 完成重复 | QPS | 接受 token/s | 交付 token/s | 抢占 | 接口错误 |',
          '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        text.append(f"| {r['concurrency']} | {r['backend']} | {r['repeats']} | {r['qps']:.4f} | "
                    f"{r['accepted_output_tps']:.2f} | {r['delivered_output_tps']:.2f} | {r['preemptions']:.0f} | {r['api_errors']} |")
    text+=['','合并两个重复后的同并发变化：','',
           '| 并发 | QPS 变化 | 接受输出吞吐变化 |','| ---: | ---: | ---: |']
    for r in gains:
        text.append(f"| {r['concurrency']} | {r['qps_gain_percent']:+.2f}% | {r['output_tps_gain_percent']:+.2f}% |")
    if summary['complete']:
        text+=['','在本次测试的并发范围内，分别取各后端的最高合并观测值：','',
               '| 指标 | 原生 vLLM | KVMem | 变化 |','| --- | --- | --- | ---: |']
        for key,label in [('qps','QPS'),('accepted_output_tps','输出 token/s')]:
            best={b:max((r for r in rows if r['backend']==b),key=lambda r:r[key]) for b in ('native','kvmem')}
            n,k=best['native'],best['kvmem']
            text.append(f"| {label} | {n[key]:.4f}（并发 {n['concurrency']}） | "
                        f"{k[key]:.4f}（并发 {k['concurrency']}） | {100*(k[key]/n[key]-1):+.2f}% |")
        text+=['','这只是在 1/2/4/8 并发、当前短窗任务组合中的观测峰值，不是平台的全局容量上限。']
    text+=['','各配对重复的变化（正值表示 KVMem 更高）：','',
           '| 并发 | 重复 | QPS 变化 | 接受输出吞吐变化 |','| ---: | ---: | ---: | ---: |']
    def pct(v):return '不可计算' if v is None else f'{v:+.2f}%'
    for r in summary['pairs']:
        text.append(f"| {r['concurrency']} | {r['repeat']} | {pct(r['qps_gain_percent'])} | {pct(r['output_tps_gain_percent'])} |")
    text+=['','上下文和延迟（数值范围是两个重复的观测范围，不是置信区间）：','',
           '| 并发 | 后端 | 平均输入 token | 最长输入 token | 引擎 TTFT p95 秒 | agent 完整回复 p95 秒 | 工具错误 / 调用 |',
           '| ---: | --- | --- | ---: | --- | --- | --- |']
    def span(values,scale=1):
        values=[v/scale for v in values if v is not None]
        return '缺失' if not values else f'{min(values):.2f}–{max(values):.2f}'
    for r in rows:
        text.append(f"| {r['concurrency']} | {r['backend']} | {span(r['context_mean_per_repeat'])} | "
                    f"{r['max_prompt_tokens']} | {span(r['ttft_p95_ms_per_repeat'],1000)} | "
                    f"{span(r['bridge_end_to_end_p95_seconds_per_repeat'])} | {r['tool_errors']} / {r['tool_calls']} |")
    text+=['','两组均支持 262,144 token，但实际任务历史长度在 aggregate.json 中单独报告。',
           'KVMem 使用 K96K/G16K 和完整 CPU 历史，包含选择性 attention 的影响。',
           '上述是闭环真实 agent 工作负载；轨迹和输出长度可能不同，不能据此认定解题质量等价，',
           '也不能把测试范围内的最高 QPS 称为给定延迟 SLO 下的最大可持续 QPS。',
           '窗口末未完成回合不计入 QPS；其已生成 token 包含在接受吞吐内。未完成题目不打分。']
    if not any(r['official_results'] for r in summary['cells'].values()):
        text+=['','本轮没有题目在测量窗口内自然结束，因此没有官方解题评分，不能比较解题成功率。']
    observed={name for r in rows for name in r['tasks_observed']}
    text += ['',f'实际运行了题池中的 {len(observed)} 道不同题目；低并发窗口只展开固定顺序中的前序题目。']
    text+=['','aggregate.json 的 bridge_end_to_end_p95_seconds_per_repeat 从原始 finish.seconds 重新计算，',
           '包含桥接器准备及 CPU 准入等待。原生 TTFT 是引擎首 token 时间；本任务客户端接收缓冲后的完整回复，',
           '因此原生 TTFT 不等于 agent 的可见首字延迟。不同并发在固定时间内展开的实际题目集合可能不同，',
           '跨并发变化不能全部归因于 GPU 容量。']
    text+=['','工具错误来自 agent 执行的命令或工具，不等同于模型接口失败；任务质量需单独评价。',
           '图中的误差线只表示两次观测的最小值和最大值。原始逐组结果在 summary.json，',
           '原始请求、窗口边界、服务器指标和容器快照分别保存在各测量窗口目录中。']
    if has_replay:
        supplement_at=text.index('## 补充：真实 agent 短窗口')
        supplement=text[supplement_at:]
        text=text[:supplement_at]
        matched=read(replay/'summary.json')
        text+=['## 主结果：长历史 8 并发','',
               '目标是让 8 个真实长历史请求同时进入系统，比较开启和关闭 KVMem 的服务性能。',
               '8 并发指客户端同时提交 8 个请求，后端自行调度；排队、抢占、重算、CPU 传输和解码均计入墙钟。',
               '输入为 228,827–230,102 token 的完整真实 DeepSWE 历史，共同配置上限为 262,144 token。',
               '先对这 8 份输入各生成 1 token 完成首次准备，再接上相同的录制 continuation token，',
               '同时请求每份固定生成 4,096 token。QPS = 8 / 整批续接秒；输出吞吐 = 32,768 / 整批续接秒。','',
               '| 后端 | 冷准备秒 | 续接秒 | 续接 QPS | 续接 token/s | 含冷准备 token/s | 续接重算输入 token | 总抢占 |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name,r in matched['arms'].items():
            text.append(f"| {name} | {r['cold_prime_seconds']:.2f} | {r['resume_seconds']:.2f} | {r['qps']:.4f} | "
                        f"{r['resume_output_tps']:.2f} | {r['cold_plus_resume_output_tps']:.2f} | "
                        f"{r['resume_computed_prefill_tokens']:.0f} | {r['preemptions']} |")
        text+=['',f"续接 QPS / 吞吐达到原生的 **{1+matched['resume_gain_percent']/100:.2f} 倍**"
               f"（{pct(matched['resume_gain_percent'])}）；含首次准备的整批吞吐为 "
               f"**{1+matched['cold_plus_resume_gain_percent']/100:.2f} 倍**"
               f"（{pct(matched['cold_plus_resume_gain_percent'])}）。",
               '8 份不同完整输入来自 4 道真实题各 2 个回合，约 229K token；固定 4096 输出和 ignore_eos。',
               '本组对照每个后端运行一次，未由单次结果估计置信区间，也未测定开放到达流的最大可持续 QPS。',
               '这是容量/服务吞吐诊断，不是新的真实工具任务或质量评分。',
               '当前 KVMem 同时使用 CPU 历史和选择性 attention，结果包含整个部署方案的影响。',
               '重算输入取原生 local-compute prefill counter；KVMem 私有查询重放不在该计数内，',
               '但它及 CPU 传输、历史恢复的耗时均包含在续接墙钟中。']
        if (replay/'counter-audit.json').exists():
            counters=read(replay/'counter-audit.json')
            n,k=counters['native'],counters['kvmem']
            text+=['',f"原生首次准备抢占 {n['prime_preemptions']:.0f} 次、续接抢占 {n['resume_preemptions']:.0f} 次；"
                   f"KVMem 分别为 {k['prime_preemptions']:.0f} 次和 {k['resume_preemptions']:.0f} 次。"]
        if (replay/'capacity-audit.json').exists():
            capacity=read(replay/'capacity-audit.json')
            text+=['',f"续接期间实测 KVMem 纯 8 请求解码步数：{capacity['pure_eight_decode_steps']}。",
                   f"这些步中的最小逻辑历史为 {capacity['minimum_logical_tokens_in_eight_decode']} token，",
                   f"最大 GPU 活跃长度为 {capacity['maximum_active_tokens_in_eight_decode']} token。",
                   'CPU 完整历史及原生启动页池估算见长历史实验的 capacity-audit.json。',
                   '原生日志的理论并发估算与实测解码批次不同，不能直接当成测得的用户并发上限。']
        text+=['',f'[长历史原始结果](../{replay.name}/summary.json)；'
               f'[容量证据](../{replay.name}/capacity-audit.json)。']
        text+=['']+supplement
    if (root/'final-audit.json').exists():
        audit=read(root/'final-audit.json')
        text+=['',f"最终输入、代码及输出配额核对：{'通过' if audit['passed'] else '失败'}。"
               '详见 [核对记录](final-audit.json)。']
    if (root/'throughput.png').exists():
        text+=['','![真实任务 QPS 与输出吞吐](throughput.png)']
    (root/'report.md').write_text('\n'.join(text)+'\n')
    return rows


def plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    palette={'native':'#426b9d','kvmem':'#d16a32'}
    for axis,key,label,repkey in (
        (axes[0],'qps','Completed model requests / s','qps_per_repeat'),
        (axes[1],'accepted_output_tps','Accepted output tokens / s','accepted_output_tps_per_repeat')):
        for backend in ('native','kvmem'):
            chosen=sorted((r for r in rows if r['backend']==backend),key=lambda r:r['concurrency'])
            x=[r['concurrency'] for r in chosen];y=[r[key] for r in chosen]
            errors=[[r[key]-min(r[repkey]) for r in chosen],
                    [max(r[repkey])-r[key] for r in chosen]]
            axis.errorbar(x,y,yerr=errors,marker='o',capsize=4,lw=2,color=palette[backend],
                          label='Native vLLM' if backend=='native' else 'KVMem')
        axis.set_xticks([1,2,4,8]);axis.set_xlabel('Concurrent real DeepSWE agents')
        axis.set_ylabel(label);axis.set_ylim(bottom=0);axis.grid(axis='y',alpha=.2)
        axis.spines[['top','right']].set_visible(False);axis.legend(frameon=False)
    figure.suptitle('Real DeepSWE: 262,144-token context limit',fontsize=14,fontweight='bold')
    figure.supxlabel('5-minute windows; bars show observed repeat range. Actual histories and trajectories vary.',fontsize=9)
    figure.savefig(root/'throughput.png',dpi=180)
    figure.savefig(root/'throughput.svg')
    plt.close(figure)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--replay',type=Path);p.add_argument('--plot',action='store_true');args=p.parse_args()
    rows=report(args.root,args.replay)
    if args.plot:plot(args.root,rows)
