# 图评分与 Query／累计 Attention 的 256K 对照

生产评分基线为 [25a7bd9](https://github.com/Di-Chai/kvmem-vllm/commit/25a7bd9051b65906943a9b4be1f53298305c905d)。
后续新增离线队列与逐请求租约观测，不修改三种评分算法。
图功能保持默认关闭。本页记录该基线上的评分实验，不代表对全部 agent 任务的质量结论。

## 固定协议

- qw3 `official_template_20260927` 原始完整输入中 256K 的全部 60 题：四种任务、五个深度、三个 seed。
- 逐题完整 int32 token 文件、Query 坐标、原始评分函数均冻结并校验；不截断 filler，不按答案筛题。
- 三组各 60 题，8 个客户端；每条历史按完整输入、最大输出和 generation staging 预留 CPU 信用。
  当前协议使用连续 FIFO waiting queue，任一请求完成并通过其自身退休门禁后立即补位。
  running 与 retiring 合计最多 8 条，不等待整波结束；详见 [离线队列策略](offline-inference-queue.md)。
- 同一 27B NVFP4 权重、FP8 KV、native MTP(k=4)、编译/CUDA Graph。
- B32、K36,864、G16,384、sink1,312、recent0；普通 prefill chunk1,024。
- r5 起显式恢复旧测试的 `query_replay_budget=16384`、`max_num_batched_tokens=16424`，
  启动器解析后仍为 `16384 + 8×5 = 16424`。每请求 prefill chunk 保持 1024。
  不仅检查 profile：每个 arm 在发出第一题前，检查实际 `launch.json` 的 native argv、
  chunk 环境变量、Query/K/G 上限、FP8 KV、MTP(k=4) 和 Graph 执行设置，保存 `launch-audit.json`。
- temperature1、top_p.95、top_k20、min_p0、presence0、repetition1；逐题原 seed；thinking4,096、最大输出8,192。
- 完整历史在 CPU；cache 上限 100,000,000,000 bytes；保留 16 GiB 入口 payload 余量，运行保留 8 GiB CPU 和 4 GiB GPU 可用内存。
- 模型逻辑上限 524,288，容纳完整 256K 原始输入及输出；GPU 实际窗口仍受上述 K/G 限制。

| arm | 评分 | `graph_mode` | `action_fraction` |
|---|---|---|---:|
| baseline | 原 MeanK Query | off | 0 |
| legacy | Query＋累计 prefill/output Attention 配额 | off | 0.5 |
| multi | 当前 Query 加权的一跳 locality＋attention 图 | multi | 0 |

multi 使用固定 `wL=.1`、`wA=.2`、top4、cross-message=.25，未按测试答案调权。
裸 token 输入不捏造消息边界；缺少边界时沿用已实现且有日志的 locality fallback。

三个 arm 共享相同冻结 runtime，profile 只允许 `graph_mode` 和 `action_fraction` 不同。
为隔离评分选择，三组均采集相同的 input/non-thinking output attention。
因此 baseline 是 Query-only **选页**基线，并非关闭所有 attention 采集的最低成本生产部署。

首轮统一关闭 Query history、guided Query、attention pressure 与 repository GC，保持 chronological
pressure 行为相同。图评分现阶段的支持边界不允许它与 attention pressure 同开。
所以本实验比较正常 Query 评分，不把不同 pressure 策略造成的收益归到图上。
过去完整 attention pressure 部署的 97.08%、53/60 只作历史参考，不作同配置对照。

## 验证和结果口径

测试不向模型发送独立的金标、针位置或证据标签，只发送原始输入与公开 Query 坐标。
每题检查原输入字节、返回接受 token 数、终态历史长度、唯一 Query 重放、预算、CPU 退休确认和逐请求释放门禁。
输出 Attention 行数由独立 accepted-token parser 重放核对；Query replay 不得增加 prefill 观测。
图组还要求真实 Query 图评分日志、非空来源与边以及正确接受水位；报告实际改变选页的题数。

主要结果为逐题数值覆盖率宏平均和整题数字集合正确数，附逐题配对、图独有命中／丢失和
McNemar exact 双侧 p 值。固定 seed 的一次配对仍不代表多轮稳定性；不同实际 batch 和生成轨迹也会影响输出。
选页证据覆盖单独报告，不能以“证据已选中”替代最终回答正确。

吞吐为完整长输入、选页、恢复及短答案混合任务的端到端速率；不能称为纯 decode 吞吐。
首次加载、编译和矩阵算子调优不计入客户端答题窗口，但包含在全程资源监控中。
每个 arm 使用独立实例，持有现有 GPU 测试锁，退出验证自有进程组和 GPU 进程已清理。
任何技术失败停止该次尝试并保留证据，不因低分自动重试。

此前波次协议的测试入口与评分门禁通过 133 项 CPU 回归；连续队列及相关路径已通过 409 项 CPU 回归。
完整三组真实模型已完成，每组原始 60 题，所有会话和自有 GPU 进程均已释放。

| 评分 | 数值覆盖率 | 完整数字集合正确 | 整批耗时 | 准入后平均每题 |
|---|---:|---:|---:|---:|
| Query＋图 | 65.00% | 30/60 | 3731.07 s | 483.26 s |
| Query | 66.67% | 30/60 | 2905.62 s | 375.58 s |
| Query＋累计 Attention | 66.25% | 31/60 | 2871.94 s | 371.34 s |

图组未显示质量提升；相对 Query 覆盖率低 1.67 个百分点，整题命中数相同。
相对累计 Attention 覆盖率低 1.25 个百分点，少一题整题正确。
两项整题配对 McNemar exact p 均为 1，不能宣称统计上的优劣。
本配置三组均关闭 pressure attention，不能将这组结果等同于历史完整 Attention 路径的 97.08%。
原始报告与逐题配对见 `artifacts/graph-niah256k-20261018-r5/comparison.json`。

首次尝试 r1 保留为失败证据：客户端错误地用未对齐的 Query 起点检查图接受水位，
而实际公开 Query replay 从所在页的起点开始。修正门禁为页对齐的 replay 起点，并新增回归；
这没有修改生产评分路径、样本、采样参数或预算。r1 的两条回答不计入正式结果。
随后 r2 把显式关闭移到每波结束。用户要求持续补位后，进一步检查确认：非保留检查点的
独立题目会在普通退休后自动删除空会话。新协议不再发送额外 GPU 清理请求，而是检查
本请求的 scheduler 会话、CPU 历史租约与 GPU reservation 已释放，再补入下一题。
波次协议的 r2 保留为部分实验，不与后续连续协议混算。所有尝试的冻结文件与证据独立保留。

连续协议的首次启动 r3 在客户端输入校验阶段发现漏传 tokenizer，未发出模型请求；
失败触发自有进程清理并确认无 GPU 残留。参数传递已修复，新增完整输入提交阶段回归；
重新冻结的新尝试使用 r4 目录，r3 的失败证据保持原样。

r4 沿用旧实验片段的 `query_replay_budget=1024`。启动器据此将 profile 中的 4224
归一化为实际 native 总预算 1064，而历史完整 Attention 运行的实际值为 16424。
因此这次性能对比没有对齐总 batch 预算。用户要求恢复相同参数后，r4 在完成 34/60
时正常停止，保留部分结果、逐秒监控、退出原因和无残留清理证明，不并入新结果。

r5 使用新的独立目录，三组全部从原始 60 题开始。相对 r4，模型／评分／存储／采样／
连续补位参数保持相同，只把 Query 原子上限和 native 总预算恢复为 16384／16424。
这里对齐的是历史运行的批处理参数；前述 pressure、历史 raw-K 精度等差异仍不能被称为
已经与历史 97.08% 高分部署完全对齐。302 项相关回归包含三组 profile 经真实启动器
归一化后的检查，以及有效 batch、Query、chunk、MTP 或执行模式漂移时的拒绝检查。
r5 图组真实服务的 `launch-audit.json` 已通过，实际 forward 已出现 7 路 prefill、
单步 7168 token。启动证据见 r5 的 `startup-parameter-proof.json`；这只证明新预算已生效，
尚不是整轮性能或质量结果。逐秒监控同步记录在 r5 的 `telemetry-r1/`。

## 图维护热点与稠密 CPU 路径

r5 的图组和 Query 组各完成原始 60 题后，客户端整批耗时分别为 3731.07 秒和
2905.62 秒，差 825.45 秒。图组 `graph_prepare_seconds=794.55`、调用 20,977 次；
60 次最终 Query 图选择合计仅 6.47 秒（其中传播评分 3.62 秒），图提交合计 0.94 秒。
两组 attention event 等待均约 1.9 秒，metadata 等待均不足 0.1 秒。
这些是主机 elapsed 计数，可能重叠设备工作，不能相加为独占 GPU idle，
但它们明确将重点指向逐块接受路径，而非最后的一跳传播。

同形状的 CPU profiling 进一步复现：32 行 × 1568 列 FP32 attention，每行按原始
完整分布验证、归一化、聚合和因果 top-k。原路径逐项迭代 NumPy scalar、构建 dict，
再对约 1500 个候选全排序，长 prefill 期间反复执行并占用 GIL。
早期 host benchmark 使用每行约五条稀疏边，未代表这个生产稠密输入成本。

新增 `csrc/block_graph_accept.hpp`：直接读取现有 CPU buffer，在 C++ 中释放 GIL 后
完成校验、按 source 完整聚合和 partial-sort top-k，仅返回封存行及必要的完整 partial。
不提前按单 sample 截断，不改变归一化、重复 target 合并顺序、确定性并列排序或评分权重。
内存准入仍在 native 计算之前；原有 prepare/commit/abort、跨线程 handoff、接受水位、
checkpoint 与回放去重门禁继续生效。普通序列输入保留原标量实现作为对照。

100 次 CPU 对照中平均每次 40.95 ms → 1.16 ms，约 35.1 倍；图行、权重和 partial
与原实现逐项完全相同。此数值只描述维护热点，不能称为整轮模型吞吐提升。
复现：`PYTHONPATH=src python scripts/benchmark_graph_accept.py --output <new-path.json>`。
相关图、接受事务、Action、prefill、检查点和 CPU scatter 回归已通过 396 项，
另验证 50 组包含极小 FP64 权重的矩阵与标量图输出完全相同。
原始证据位于 `artifacts/graph-overhead-20261018/`。

真实模型验证另取固定 8 题：四类任务各取深度 10%、seed 1000/2000，未经答案筛选。
before 使用 r5 的冻结旧 runtime；after 使用新增稠密路径的冻结 runtime。
二者共享逐字节相同的 256K 输入、采样、模型、图权重、预算和 8 路并发。
新对照在 r5 完整三组测试结束后独占 GPU，不中断原测试，目录为
`artifacts/graph-overhead-20261018/gpu-r2/`。r1 在使用 GPU 前取消，保留为准备记录；
r2 冻结了完整 native 输入边界校验后的构建。
该短对照只检验额外开销和功能回归，不能替代完整 60 题的质量对照。

两组均已通过，8 路完整历史保持在 CPU、零磁盘 KV 读写，终态会话、信用与 GPU 进程
全部释放。相同 8 题、相同 profile 的真实结果如下，均不包含服务加载/编译时间：

| 指标 | 原 Python 维护 | 稠密 native 维护 |
|---|---:|---:|
| 图 prepare 累计主机耗时 | 94.445 s | 6.686 s |
| 最终 Query 图选择合计 | 0.703 s | 0.641 s |
| 平均首接受 token 延迟 | 447.194 s | 370.193 s |
| 准入后平均每题耗时 | 453.564 s | 375.836 s |
| 整批客户端窗口 | 465.343 s | 386.378 s |
| 窗口内平均 NVML GPU 利用率 | 74.89% | 89.40% |
| 总输入量 / 整批客户端窗口 | 4505.96 token/s | 5426.85 token/s |

图维护主机 elapsed 减少 92.9%，平均首 token 延迟减少 17.2%，整批时间减少 17.0%，
整批输入处理率增加 20.4%。窗口包含 Query 重放、选页、短答案及退休门禁，
最后一项不是纯 GPU prefill benchmark，更不是 decode 吞吐。
两次输出总量为 3163 / 2914 token；平均首 token 后阶段分别为 6.370 / 5.643 秒。
因此延迟改善主要来自首 token 前，不能解释为少生成一些答案造成的假收益。

短样本数值覆盖率 65.625% / 68.75%，完整数字集合均为 4/8；部分答案不同。
已观察到两组实际 batch 组合不同，浮点执行可能随之变化；不能把此小样本差异归为评分质量提升。
已单独通过同 attention buffer 的标量/native 图状态逐项完全相同比对；
仍需区分数学语义回归与完整模型生成轨迹的逐 token 一致性。
证据见 `gpu-r2/comparison.json`、逐题 request/response/evidence、冻结 runtime hash、
两组 `cleanup.json` 及 `telemetry-r1/`。这次只修复图维护额外成本，图功能继续默认关闭。

## 运行期利用率检查（非最终性能结论）

r2 首波完成时，8 个客户端并发提交的请求已经全部执行；观察到的模型 decode batch 最大为 2。
45 秒只读 GPU 采样的平均利用率为 65.9%，范围 1–100%，其中 8/45 个采样低于 30%。
同窗口 179 个模型 step 中，149 个仅有一路 prefill，7 个两路 prefill，
7 个一路 prefill 加一路 decode，16 个仅有一路 decode。
这一窗口主要处于波次收尾，不能推断整轮平均利用率。

实际总 token 预算 1064，以及一波结束后才补齐下一波的质量测试协议，均限制了持续大 batch。
图状态准备在首波累计约 134 秒主机计时，Attention 捕获累计约 107 秒主机计时；
这些计时可能与设备工作重叠，不能相加或等同于 GPU idle。
该采样属于旧波次协议；r4 后续改为连续补位，但仍保持 1064 的实际总预算。
用户要求纠正批处理参数后的 r5 恢复 16424；这些不同协议的部分结果不合并。
诊断原始采样及结构化说明保留在 r2 的 `gpu-diagnostic-1792310994.json`、
`status-diagnostic-20261018.json`。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  .venv/bin/python -B scripts/benchmark_graph_niah.py prepare \
  --output artifacts/graph-niah256k-repro

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  .venv/bin/python -B \
  artifacts/graph-niah256k-repro/frozen/benchmark_graph_niah.py run \
  --output artifacts/graph-niah256k-repro
```

输入和权重使用现有本机路径；更换模型、预算或采样参数需要独立的新协议与目录。
原始模型和大体积 token／运行文件不随 Git 提交发布。
