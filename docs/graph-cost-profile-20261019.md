# Attention 统计、构图与 Query 图传播开销初测

已完成三组真实模型测量，每组 8 个独立会话、完整 256K 输入（262,101 token）
和固定 16K 接受输出。三组都开启 KVMem；原始 Mean-K、CPU 完整历史和 GPU
active view 均保留。八个会话重复使用同一份原始输入，命名空间独立，原生
prefix caching 关闭。模型为 27B NVFP4，FP8 KV，MTP(k=4)，编译与 CUDA Graph；
K=36,864、G=16,384、batch=16,424、chunk=1,024、GPU memory utilization=.72，
CPU 历史预算 100 GB，不用 SSD。服务核心与上一轮 C8 验收逐文件相同。

为了隔离维护开销，三组统一使用 recency pressure，Query 均不沿图传播：

| 组 | attention 统计 | graph 构建 | Query |
| --- | --- | --- | --- |
| disabled | 关闭 output/prefill 统计 | 关闭 | 原始 Mean-K |
| attention | 开启原有统计 | 关闭 | 原始 Mean-K |
| collect | 开启原有统计 | 开启 collect | 原始 Mean-K，图不改选页 |

这三组不是正式质量测试中的 A/C/D，也不是原生 vLLM 对照。压力选页保持一致，
不能把下表与旧 D 的 752 token/s 直接当作单开关配对，因为 D 的 pressure 及
Query 都使用图传播。完整原输入、采样、K/G、16K 输出和八并发没有缩减。

## 实测

| 指标 | disabled | attention | collect |
| --- | ---: | ---: | ---: |
| 冷准备：至八路均取得首 token | 348.24 s | 371.49 s | 374.44 s |
| 有效冷准备输入率，含排队、选页、装配 | 6,021 token/s | 5,644 token/s | 5,600 token/s |
| Prefill 模型执行率 | 6,260 token/s | 5,949 token/s | 5,911 token/s |
| 共同八路 decode 步骤平均总用时 | 37.58 ms | 41.94 ms | 42.82 ms |
| 同步骤内模型 CUDA 事件区间均值 | 37.47 ms | 40.68 ms | 41.47 ms |
| 共同窗口实际输出吞吐 | 743.26 token/s | 831.59 token/s | 663.50 token/s |
| 整批输出吞吐，包含 prefill/收尾 | 240.65 token/s | 231.58 token/s | 219.22 token/s |
| 整批请求完成时间，不含服务加载 | 544.66 s | 566.00 s | 597.91 s |
| 实际输出／原生 owner decode 验证次数 | 3.395 | 4.378 | 3.611 |

Prefill 模型执行率是公开模型执行 token 数除以相应 CUDA 事件区间；不等于
端到端有效输入率。共同窗口按接受的 SSE token IDs 与最终 usage 核对，每端
至少裁去 64 token；保留窗口内全部停顿，不用单路峰值相加。单步用时为
原生执行事件区间，也可能包含 enqueue/维护等待，不是纯 kernel 时间。

观测增量：attention 相对 disabled 的冷准备时间 +6.68%、八路平均步骤用时
+11.59%；collect 相对 attention 分别 +0.79%、+2.09%。两项开启后，相对
disabled 的平均步骤用时合计增加约 13.92%。这是本次观测，不是已经消除
所有其他因素的纯因果开销。

三组原始输入和 pressure 选页多重集合一致，但 Query 选页和接受输出不完全
一致，实际验证次数也变化。更高的接受输出／验证次数使 attention 组的
token/s 反而上升；collect 的该比值又下降，因此其 token/s 降幅约 20.2%，
不能全算作构图成本。只有一次按顺序执行的 cold cohort，没有交错重复或
相同接受轨迹回放，不能给出稳定的因果百分比。

attention/collect 的有效 output attention 样本分别为 4,966/7,325，prefill
样本均为 65,384。生成轨迹也改变了实际统计次数，进一步限制了净开销归因。

## 分项计时与频率

前一轮完整 D 配置的 8 并发 256K/16K 验收已包含真实一跳传播计时，服务核心
与本次相同；该轮的图传播策略与上述三组不同，分开记录：

| D 中的操作 | 次数 | 累计主机区间 | 每次平均 |
| --- | ---: | ---: | ---: |
| Query Mean-K 分数沿 V2 multi 图传播 | 8 | 0.6146 s | 76.82 ms |
| Query 图选页包装流程，包含上行传播计时 | 8 | 0.9106 s | 113.83 ms |
| Pressure attention seed 沿图传播 | 104 | 4.1761 s | 40.15 ms |
| Pressure 图选页包装流程，包含上行传播计时 | 104 | 9.8328 s | 94.55 ms |

Query 时约有 8,173 个已观测图节点、32,682 条边。上表传播与包装流程是
包含关系，不能相加；包装流程还包括排名、校验和诊断。77 ms 是每次重选
都会产生的代价，不能仅按一次长文总耗时占比推断多轮 agent 的影响。

本次 collect 组的 graph.prepare_accept 累计主机区间为 70.135 s，共
38,482 次调用；graph commit 为 0.991 s。前一轮 D 对应为 60.141 s／35,166
次和 0.901 s。两轮输出轨迹和调用次数不同，不覆盖或混用数字。
这些维护工作可与 GPU、其他线程重叠，不能解释成 GPU 空闲 70 s，也不能
用它除以整批耗时声称净吞吐损失。每 owner 的最终图数据约 4.3～4.5 MB，
这是结束前状态，不是峰值内存。

另从同核心、两 owner 的真实长工具异步验收提取已有 CUDA 事件采样：
active 至少 32K 时，三层 attention 评分区间合计均值为 prefill 7.79 ms
（35 个采样点，每次 28 或 32 个 Q 行），decode 1.89 ms（8 个采样点，每次 1 个
Q 行）；decode 中位数 1.50 ms、p95 4.66 ms。每 8 个 batch 采样一次，
无事件丢失或未完成事件。它包含 enqueue 间隙，采样数有限且只有两 owner，
不能乘以八来推算八并发损失，也不替代上述无 profiler 的服务对照。

## 原始门禁与独立执行审计

三组均为 8/8 HTTP 成功，完整接受输出 131,072 token，终态 CPU 历史逐会话
为 262,101+16,384−1。整个共同窗口所有重叠原生步骤分别为 3,551、3,231、
3,465 步，全部 decode=8/prefill=0、序列连续；真实 public-forward 计数在
窗口内保持完整输入量不变，没有 Query/pressure 工作重叠。

旧 scheduled_queries.json 是周期写盘的调度尝试次数直方图，不是 GPU
prefill token 计数。disabled/attention 的旧快照门禁各观测到迟到的 +2 次
prefill 调度计数，故原客户端及状态仍为 failed，原日志未修改。collect 的
旧门禁正常 passed。三组独立 native 快照覆盖分别 99.35%、99.46%、99.57%，
原 30 s、90% 覆盖及全部 SSE/usage 条件保留。

对两次误报，另用完整原生执行日志核对整个窗口，包括跨边界步骤，而不只
检查完全落在窗口内的行；再核对真实 public-forward 计数、完整终态、源文件
和资源释放。独立审计通过，没有把旧 failed 改写为 passed，也没有因为门禁
重跑任一组模型。协调器预 GPU 的模块导入错误和首次退出同样保留；剩余
两组使用同一份冻结运行时及参数继续首次执行。

全部会话、CPU 历史、图状态、所属进程及 GPU PID 已释放，历史 KV disk
read/write 均为 0。CPU 控制／计数／边界／CLI 回归共 47 项通过。

汇总数据：[graph-cost-20261019.json](benchmarks/graph-cost-20261019.json)。本地证据：

- `artifacts/graph-cost-split-20261019-r1/comparison-r1.json` 与 `.csv`
- `screen-r2/{disabled,attention,collect}/`：原冻结文件、模型回答和门禁结果
- `{disabled,attention,collect}-executed-audit-r1.json`：独立执行证据及校验值
- `existing-D-reading.json`、`existing-attention-three-layers.json`：已有分项计时
- `audit-tools-r1/`、`provenance-addendum-r1.json`：原审计依赖及参考版本含义

## 复现入口

在项目根目录使用项目虚拟环境。以一个已通过的完整 C8 验收目录为 reference：

```sh
PYTHONPATH=src .venv/bin/python -B scripts/benchmark_graph_cost.py prepare --reference artifacts/p-restore-prefix-fix-20261019/native-C8-r3 --output /tmp/kvmem-cost-new
.venv/bin/python -B /tmp/kvmem-cost-new/frozen/benchmark_graph_cost.py run --output /tmp/kvmem-cost-new
.venv/bin/python -B scripts/analyze_graph_cost.py --root /tmp/kvmem-cost-new --output /tmp/kvmem-cost-new/comparison.json
```

run 会等待 GPU 和内存空闲，不停止其他任务。它分别保留原快照门禁及独立
执行审计；HTTP 错误、实际 prefill、缺失输入/输出、错误 owner 或执行日志缺口
均不能通过独立审计。失败尝试不覆盖、不自动重跑。当前这轮旧协调器的独立
审计放在父目录，分析时另传 `--audits artifacts/graph-cost-split-20261019-r1`。
