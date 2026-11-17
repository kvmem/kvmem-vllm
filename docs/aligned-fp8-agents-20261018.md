# 语义对齐后的 FP8 多并发验证（2026-10-18）

本实验接在 [qw3 上下文语义验收](qw3-alignment-20261018.md)之后，先固定新版本的
FP8 基线，再分析 KVMem 的主机维护成本，最后以相同输入和参数比较候选修改。
此前 FP8 约 134 token/s 的结果仅作历史参考，不能与本轮差值直接归因于某一修改。

后续增加有界封页缓冲池及显式 CUDA 空闲缓存回收，并重新校准主机启动余量；
各次失败记录、资格、保护条件和最终比较另见
[主机启动余量与后续验收](host-admission-calibration-20261018.md)。本文中的原始
基线、组件和 profiler 数据保留各自冻结版本，不与后续结果混成同一实验。

最终有效八用户、300 秒对照已完成：服务输出率 **145.73 → 146.95 token/s**，
实际公共 prefill 完成率 **1975.93–1991.00 → 2044.77–2066.13 token/s**。
两组均零抢占、完整历史和资源释放通过，KV pool 同为 779 blocks。候选最低
CPU 可用内存 26.45 GiB、GPU 空闲显存 5.94 GiB，通过原 8／4 GiB 条件。
输出率只差 0.84%，完成请求为 93／89，不能据此声称稳定服务加速；相同
8 路/context bucket 的纯 decode 模型步也基本不变（57.06／57.16 ms）。
详表、控制器差异和解读见校准报告的“最终完成的同协议 A/B”。

## 固定条件

- 单卡 RTX PRO 6000 Blackwell Server，Qwen3.8-27B-FP8 权重、FP8 KV、
  MTP(k=4)、编译和 CUDA Graph，vLLM 0.30.0。
- 使用原先 16 个 DeepSWE 样本中的同一组 8 条轨迹，并行保留本次模型实际产生的
  完整输出，沿用原工具消息、工具等待和采样设置。它是轨迹回放吞吐实验，
  不代表重新执行工具后的任务解决率。
- 每用户 K=65,536、G=32,768、CPU 历史上限 262,144 token；单次实际生成上限
  16,384。CPU 缓存预算 100 GB，每个逻辑 epoch 内保留完整历史，不用 SSD、
  不因资源压力丢弃历史或重算。
- 两轮预热之后计量固定 300 秒服务窗口。保留工具等待、恢复、排队和公共 prefill。
  图评分关闭；功能验证中的昂贵逐字节恢复比较不带入性能窗口。
- 每份结果固定源码、模型、配置、输入和资格证明的哈希。测量前检查 GPU 归属及
  完整 CPU 历史内存余量；结束后排空、显式释放会话并证明进程退出。

## 合法的上下文边界

一个 epoch 最多使用原始轨迹的 16 轮。每次发送请求前，按当前模型已经接受的
完整输出、完整工具消息计算下一轮 prompt；只有
`prompt_tokens + 16384 <= 262144` 才允许发送。若下一轮不满足此条件，或已完成
16 轮，等待该用户当前请求结束并显式释放其会话，然后从同一份完整初始输入
开始新的独立 epoch。其他用户继续运行，没有全体重启屏障。

因此，达到逻辑上下文上限时允许结束 epoch；CPU 压力不是提前重置历史的理由。
不会裁剪工具内容、缩短生成上限、补入原轨迹的模型输出，或把未发送的尾部轮次
算作已执行。每个用户必须先完成原始的两轮预热，若实际输出使第二轮无法容纳，
该次实验失败，不自动换样本。窗口内的冷启动、清理等待和重新 prefill 均计入
300 秒，清理请求的一 token 输出单列，不计入有效输出。到达窗口末尾后停止
发送新工作，已发出的请求正常排空。最终 `own-history-proof.json` 逐请求验证
这些边界及完整输入/输出哈希。

## FP8 运行时资格

`aligned-fp8-agents-20261018-baseline-r3/qualification` 已通过。它使用正式基线
相同的全部 247 个冻结运行时文件和相同 FP8 权重，另外用缩小的 K=32768、
G=4096 和极低的空闲驻留额度，主动触发冷恢复及 rolling 边界。它是功能资格，
不是正式 K64/G32、八并发吞吐窗口，也不能用 NVFP4 语义资格代替。

| 检查 | 实际结果 |
| --- | --- |
| 完整原始长历史，2 用户各 3 轮 | 6/6 次完整 CPU 历史与终端 M 发布通过 |
| 跨轮 CPU 冷恢复 | 4 次 target/MTP immutable FP8 rebuild、4 次 recurrent 字节比较全部通过 |
| 生成中的 rolling 边界 | 两用户各 1 次，target/MTP 与 recurrent 字节一致，公开重放和 private forward 均为 0 |
| MTP 接受边界 | 两次 rolling 前实际接受 2 和 5 token，均等于已发布历史末尾 |
| 并发覆盖 | 1478 个实际双请求纯 decode 步骤；仅作覆盖证据 |
| 输出记账 | native 与客户端均为 12290 token，含 2 个清理 token；0 次抢占 |
| 停止服务 | 独立 post-stop 证明 GPU 进程列表为空，绑定资格和性能 manifest 哈希 |

这里的旧 rolling 资格使用“一 token 探针请求替换旧会话，然后退出服务”的清理
方式。停止服务前仍有 2 个一 token 会话、2 个 checkpoint 和少量 CPU 准入额度；
不能把 `cleanup_completed=true` 解释成服务存活时全部会话资源为零。进程退出
回收与显式 release 是不同证明：正式性能协议仍必须通过独立的显式释放、终端
完整历史和最终资源归零检查。该资格没有注入取消，不能据此宣称取消路径通过。

只读复验已通过原参考、完整冻结运行时、模型、profile/fixture、语义资格及
FP8 字节证据的全部门禁。基线 manifest SHA256 为
`b09de958df9b331cf636b0a3a645b3bd76ddec14e8816167a20e19007b59f61d`，
FP8 资格 manifest 为
`30c43f110915f3b6fcfc1e5e425372107afbaa017e3ba175991bc40ee00933af`。
证据见基线目录的 `qualification/results.json`、
`qualification/process/qualification-cleanup.json` 和 `qualification/process/post-stop.json`。

完整封页候选也已通过独立资格：
`qw3-alignment-20261018-sealed-copy-natural-async-r1` 的 2 用户 × 3 轮
guided/collect 上下文回归通过，非法请求独立拒绝，服务内资源归零且自有 GPU
进程退出；随后 `aligned-fp8-agents-20261018-candidate-r1/qualification` 使用
候选的完整冻结运行时，6 个工作请求的 CPU 完整历史发布通过，4 次 target/MTP
冷恢复、4 次 recurrent 恢复及 2 次 rolling 字节比较全部通过。native/client
均为 12290 token，0 次抢占，实际双请求纯 decode 覆盖 1526 步，停止服务后的
GPU 进程列表为空。最后两次一 token 探针清理仍具有上文说明的生命周期边界；
此次也未注入取消。这些数据证明候选可进入性能测试，不能作为吞吐或质量提升。

原始协议的正式八用户准入按完整历史预算要求 MemAvailable 至少
113,834,459,136 字节（约 106.02 GiB），连续三次检查均满足后才启动。
预算没有因当前空闲内存接近门槛而减小；只回收控制器自身的临时内存，未清理
其他进程或系统缓存。运行期间仍保留内存余量监测。

## 统计口径

1. **服务窗口输出率**：被接受的输出 token 除以完整窗口时间。包括 prefill、
   恢复和工具等待的影响，不能称为纯 decode 内核吞吐。
2. **公共 prefill**：实际送入模型的 prompt 位置，排除直接恢复的 KV 和私有
   Query；分别报告窗口 token/s 与纯 prefill 步骤中的速度。
3. **decode 协同**：区分纯 decode 和混合步骤，报告执行时间分布、实际 batch
   大小及抢占。MTP 提议数不能替代被接受的输出数。
4. **主机等待**：分开观察历史封存、恢复、维护和检查点。不同请求的等待会重叠，
   不能相加后视为 GPU idle 时间。

性能采样与 profiler 分开运行。诊断使用相同已通过资格的运行时，固定 32 个
profiling 步骤；其结果标记 `performance_eligible=false`，不纳入性能收益。

基线与候选的最终比较按以下顺序解释：

- 先核对相同输入、工具等待、预算和生成上限，候选只允许声明的生产代码差异；
  各自必须通过完整历史、native/client 计数、无抢占和显式 release 门禁。
- 用 `[T0,T1)` 内客户端接受的 workload token 事件计算输出率，包括在 T1 后
  才结束的请求在窗口内已交付的 token，排除预热和清理输出。MTP 提议、验证
  位置、`public_non_prefill_tokens` 都不是接受输出数。
- 公共 prefill 使用实际 forward 的位置计数，报告采样边界对应的完成量上下界，
  并把普通增量 prefill 与原子公开 Query 重放分开。恢复的 KV、私有 Query、
  请求声明的 prompt 长度及 native 命中缓存后的 prompt credit 不能替代此值。
- 纯 decode、混合和纯 prefill 分别比较实际 batch 分布及 model/maintenance/
  step 延迟；尽量匹配 decode 上下文和 prefill 组成。若没有接受输出到 GPU
  step 的严格对应，不计算“所有输出 ÷ 纯 decode 步骤耗时”这种混合口径。
- CPU publication 结合完整历史发布、封页/检查点耗时、后台队列和请求等待
  判断；恢复按 resident/CPU 路径拆开。request-seconds 会跨用户重叠，模型
  事件耗时也可能包含异步工作，均不能直接相加或从 300 秒中相减来推导 GPU
  idle。具体阻塞归因由独立 trace 支持。

## 已确认的优化候选

`qw3_cuda` 完整页最终封存会按原始 token 顺序重新计算 MeanK。当前批量历史
追加在此之前还计算一次增量和，随后被覆盖。只有本次追加补满该页时才可跳过
前一次求和；未满页、其他 MeanK 后端及拒绝 token 的处理都须保持原语义。

组件证据保存在
`artifacts/qw3-alignment-20261018/sealed-mean-append-micro.json`：8 用户 × 2048
token、16 主模型层＋1 MTP 层、B32、4×256 通道、单 CPU Torch 线程，3 次测量。
独立的 FP8 转换及求和中位数为 680.5 ms，现有完整 append 主机耗时中位数
为 1.214 s。两项来自分别执行的计量，不能称为同一轨迹内的 56% 占比，
更不能解释成 GPU 吞吐提升。512 个封页的历史字节与 MeanK 索引均通过比较；
部分页测试确认仍需保留累积和。

候选已完成相同 CPU 组件测试，结果见
[micro-comparison.json](../artifacts/sealed-copy-only-20261018/micro-comparison.json)：

| 组件墙钟中位数，3 次 | 冻结基线 | 候选 | 减少 |
| --- | ---: | ---: | ---: |
| 完整页 append | 1.214 s | 0.246 s | 79.7% |
| append＋强制归档完成屏障 | 2.119 s | 1.058 s | 50.1% |

每轮 512 个完整封页的 raw K、V、MeanK 索引 SHA256 均一致，部分页累积契约
保持。相对于正式基线的实际服务运行时，差异仅为 `authority.py`、
`csrc/bindings.cpp` 及重编译的 `_native.so`，具体哈希和差异保存在同目录的
`candidate-source-identity.json`。此表只覆盖该批量完整封页场景；归档屏障是
组件测试主动加入的等待，不代表每个服务回调都会等待。它不能外推为 decode
或八 agent 吞吐改善，真实 GPU 窗口收益仍待同参数测试。

相关 CPU 回归为 106 项通过，另有 4 项独立生命周期检查通过，覆盖源对象释放、
发布后 staging 释放、取消未发布页及资源回收；重新构建后的现有 C++ 五项检查
通过。测试集合与先前 805 项广泛回归有重叠，不合并为新总数。扩展保留原九字段
scatter 作业协议，只对显式标记且已补满页的兼容作业跳过求和；所有作业先验证
再写入。未满页仍走原增量路径，最终 MeanK 的 FP32 token 顺序归约保持不变。

## 八并发基线结果

`artifacts/aligned-fp8-agents-20261018-baseline-r3` 已完成 300 秒正式窗口和正常
排空，并通过部署与生命周期门禁。计量区间为 monotonic
`[4815038.0682113, 4815338.0682113)`；下列输出和 prefill 均按该窗口计算，
不使用预热累计值。

| 指标 | 本轮基线 |
| --- | ---: |
| 接受的 workload 输出 | 46,814 token，**156.05 token/s** |
| 实际公共 prefill 完成率上下界 | **1994.45–2028.59 token/s** |
| 公共 prefill 提交采样内区间 | 2015.75 token/s，边界口径与完成率不同 |
| 窗口内发送／完成轮次 | 97／89 |
| 窗口结束后正常排空 | 127.20 s，后续输出不扩大窗口分子 |
| NVML 时间加权利用率 | 86.51% |
| NVML 利用率低于 10% 的采样时间比例 | 2.19% |
| 抢占 | 0 |

NVML 数据覆盖完整窗口，272 次采样、最长间隔约 1.13 秒；它不是 SM occupancy，
也不能把 `100% − 86.51%` 当成可消除的主机 idle 比例。四次合法 epoch 重启均
因为已完成 16 轮，冷 prefill 与清理成本保留在窗口中，没有因内存压力提前
重置历史。本轮 156.05 与旧 FP8 约 134 token/s 的差值只作历史观察，不能归因
于单一优化；新的候选必须与本轮基线按相同协议比较。

步骤拆分如下。时间来自公共模型步骤的事件采样，包含 MTP、采样或 capture；
累计事件时长不等于独占 GPU 墙钟时间。

| 步骤 | 数量 | model p50／p95 | step p50／p95 | 累计采样 step 时长 |
| --- | ---: | ---: | ---: | ---: |
| 纯 decode | 1781 | 52.50／58.63 ms | 53.59／109.17 ms | 112.43 s（38.37%） |
| prefill＋decode 混合 | 385 | 490.18／709.70 ms | 496.13／718.63 ms | 179.99 s（61.43%） |
| 纯 prefill | 2 | 样本不足以单独评价 | 样本不足以单独评价 | 0.58 s（0.20%） |

括号内比例仅以这三类采样 step 时长之和为分母，不是 300 秒内的独占时间份额。
纯 decode 的平均实际 batch 为 6.39，按 step 时长加权为 6.28，实际 batch=8
出现 184 步。混合步骤平均 decode batch 为 5.81，并执行 605,828 个公共
prefill 位置；该值按步骤开始时间归窗，与完成边界上下界不是同一统计。
这说明已存在持续多路 decode，但混合步骤持续时间明显更长；还不能仅凭这些
统计把增量耗时归因于某个 KVMem 函数。

主机相关观察也限定于该窗口：

- 纯 decode 的 maintenance p50/p95 为 0.11/55.98 ms，最长 2.32 s；
  具体长尾来源留给独立 trace，不能把全部维护时间都归于 CPU 封页求和。
- 93 次恢复均复用 resident active KV，恢复记录中的 H2D 上传量为 0。
  pre-submit p95 为 0.624 s，service p95 为 0.378 s，总恢复 p95 为 1.000 s；
  各分位数不具有可加性，窗口内也未覆盖 CPU 冷恢复的性能分布。
- completion fence 累计 39.67 request-seconds，候选等待 p95 为 0.666 s。
  不同用户等待会重叠，不能把 39.67 秒解读为 GPU 停顿。
- 完整历史发布证明覆盖 113 个非清理请求，包含 16 个预热请求和正常排空后
  才结束的工作，因此其总量不是窗口内 publication 开销。终端回收时间记录
  也不能替代每一步 CPU 追加、封页及 checkpoint 的耗时；暂不据此报告主机
  publication 总占比。

生命周期复查通过：133 个请求的完整自有输出与合法 epoch 哈希验证通过，
native/client 输出总量一致；最终服务仍存活时的显式 release 证明所有会话、
CPU leases、GPU owned blocks、下载、历史 KV、checkpoint 和图内存均为零，
磁盘 KV 读写为零。停止服务后的 GPU 进程列表也为空。部署最小采样显存余量
7077 MiB，高于要求的 4096 MiB。这里只运行 KVMem 基线单臂，顶层
`comparison_complete=false` 表示没有原生 vLLM 对照臂，不是该基线验收失败。

原始结果见基线目录的 `separated-metrics.json`、
`rolling-candidate-v2/results.json`、`deployment-acceptance.json`、
`own-history-proof.json`、`tools/explicit-release-proof.json` 与 `post-stop.json`。
独立 profiling 已完成，见后文。候选同参数实验尚未取得正式吞吐结果。

## 候选正式窗口的资源阻塞

**后续进展：** 用户要求校准保守门槛后，基于全程峰值将额外启动预留从
32 调整为 28 GiB，保留原完整历史信用与 8 GiB 运行时保护。新的优化前基线已
通过准入并启动，两组将使用同一新条件重新计量；旧失败记录继续保留。
计算、门禁和复现见 [启动余量校准](host-admission-calibration-20261018.md)。
下文描述原 `candidate-r1` 的历史失败，不表示新实验仍被该门槛阻塞。

候选 `aligned-fp8-agents-20261018-candidate-r1` 的上下文和 FP8 资格通过后，
正式入口等待完整主机内存准入 600 秒，仍未达到 113,834,459,136 字节门槛，
以 `Insufficient host headroom after bounded wait; full budget unchanged` 退出。
本次没有启动模型或 300 秒性能窗口；其顶层 `failed` 表示性能尝试的准入失败，
不是之前已通过的模型资格失败。最高采样可用内存距门槛仍约 62 MiB。

原始失败和所有采样保留，摘要及输入哈希见
[candidate-r1-admission-summary.json](../artifacts/aligned-fp8-agents-20261018-candidate-r1-admission-summary.json)。
没有减少历史预算或并发，没有终止其他工作，也没有清理系统缓存。CPU 组件的
79.7% 改善不能代替缺失的整机结果。目前仍不能给出候选相对 156.05 token/s
基线的吞吐收益；总目标中的这项验收尚未完成。

内存满足同一条件后须保留本次失败目录，在新目录完成资格及正式测试，再运行
`artifacts/compare_sealed_copy_agents.py`。该比较器会重新验证源码、输入、完整
自有历史、实际输出、零抢占和清理证据，拒绝对尚未完成的候选生成收益数字。

后续只读交付审计重新核对了 3743 个证据文件，包括 7 组图模型的冻结运行时和
原始结果、4 组上下文资格的源码／结果哈希，以及 FP8 基线与候选资格的完整
字节、历史和释放门禁；当前 248 个运行时文件仍与候选冻结内容一致。审计保留
原图 R3 与后续源码版本的资格范围区别，并明确 `objective_achieved=false`。
见 [goal-completion-audit-20261018-r1.json](../artifacts/goal-completion-audit-20261018-r1.json)。

## 复现候选实验

在仓库根目录使用一个不存在的新目录。下面沿用已完成的上下文语义资格；
若生产源码发生变化，先按 [qw3 对齐记录](qw3-alignment-20261018.md)为新源码
重跑资格并替换对应路径，不能绕过源码绑定。准备过程只冻结代码和证据；
后两步分别启动独立 GPU 服务，并持有共享 GPU 锁、检查空闲状态和清理自有进程。

```bash
FP8_CANDIDATE_OUT=/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/aligned-fp8-agents-repro
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B scripts/benchmark_aligned_fp8_agents.py \
  --prepare-only --output "$FP8_CANDIDATE_OUT" \
  --reference artifacts/fp8-weight-performance-20261017-r1 --runtime "$PWD" \
  --alignment-qualification artifacts/qw3-alignment-20261018-sealed-copy-natural-async-r1 \
  --baseline artifacts/aligned-fp8-agents-20261018-baseline-r3 \
  --allow-production-change src/vllm_kvmem/serving/authority.py \
  --allow-production-change csrc/bindings.cpp \
  --allow-production-change src/vllm_kvmem/_native.cpython-312-x86_64-linux-gnu.so \
  --port 18616

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FP8_CANDIDATE_OUT/runtime/rolling-candidate-v2/src" \
  .venv/bin/python -B \
  "$FP8_CANDIDATE_OUT/frozen/scripts/benchmark_aligned_fp8_agents.py" \
  --run-qualification --output "$FP8_CANDIDATE_OUT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FP8_CANDIDATE_OUT/runtime/rolling-candidate-v2/src" \
  .venv/bin/python -B \
  "$FP8_CANDIDATE_OUT/frozen/scripts/benchmark_aligned_fp8_agents.py" \
  --run-prepared --output "$FP8_CANDIDATE_OUT"
```

固定窗口为默认 300 秒；资格失败、GPU 被占用或完整历史内存余量不足时，
控制器不会静默缩小预算继续。保存失败尝试，修复后创建新实验目录。

## 独立诊断：32 步 trace

`aligned-fp8-agents-20261018-profile-r1` 已完成，状态为
`diagnostic_completed`、`performance_eligible=false`，显式释放与停后 GPU
进程检查通过。它绑定已通过的基线运行时，完整两轮预热后才启动 profiler。
32 个 execute 范围含 4 个零 token 维护步骤、2 个纯 prefill、6 个混合步骤和
20 个纯 decode；最高 decode batch=7，没有 C8 稳态样本。不能把这一段启动
续接过程的时间比例或 60 秒诊断运行率当作基线性能。

完整 trace 范围约 4.497 s。按设备时间线求并集后，kernel 覆盖 2.238 s、
copy 覆盖 32.37 ms，kernel/copy/memset 的联合覆盖为 2.271 s；其余 2.227 s
没有这些被追踪活动。不同类别和阶段可能重叠，不能直接相加；没有追踪到活动
也不等于已经证明硬件 idle 或存在等量可消除开销。特别是 profiler 改变了
执行耗时，这个约 50.5% 的 trace 覆盖率不能与无 profiler 的 NVML 86.5%
直接比较。

线程覆盖存在明确缺口：所有 KVMem CPU scope 和 `cpu_op` 都只来自
`pid=tid=2228743` 的 `VLLM::EngineCor` 主线程。23 次 `authority.append_batch`
合计 58.16 ms、18 次 `append.seal_finalize_fp8` 合计 4.32 ms，也都在主线程。
另外两个未命名 tid 只有 CUDA API 记录，没有后台 scatter 的 CPU 范围。
因此不能用这些较小数字推断后台历史追加成本很低，也不能把全部 23 次追加
标为 decode：其中 22 次并不嵌套在当前 native execute 范围内，待提交的 packet
还可能来自上一批。

终态累计计数则记录了后台 bulk append 173.24 s、CPU scatter 总计 230.47 s，
证明后台路径实际执行过。这些值覆盖长历史冷预热、60 秒诊断窗口及排空，不能
除以 4.497 s 或 60 s，也无法据现有快照精确拆出窗口 publication 占比。
前述完整封页组件 A/B 可以补充该路径的证据，仍不能替代服务窗口 A/B。

对长主机范围与 GPU 活动逐区间求交后，得到以下不同情况：

| 诊断区间 | 主机区间长度 | 同一区间已追踪 GPU 联合覆盖 | 可以确认的范围 |
| --- | ---: | ---: | --- |
| 最长主线程 kernel launch 间隙 | 410.44 ms | 0.0045 ms | 含一段 restore/prepare 维护以及范围外等待，不能全部归到一个函数 |
| 最长 `maintenance.pump` | 117.97 ms | 0 | 内含 `restore.step` 117.85 ms、`prepare_claim_window` 90.82 ms，三者嵌套 |
| 对齐后的最长 host maintenance | 138.20 ms | 3.66 ms，其中 copy 3.14 ms | 由保存的 wall/monotonic 锚近似对齐；是本 trace 中的步骤 |
| 最长 `action.capture` | 141.53 ms | 138.62 ms | 大部分时间设备仍在执行，不能把完整主机范围当作 idle |

这支持继续核查 resident 恢复准备和主线程提交间隙，但尚不足以认定其全部时间
可优化。纯 decode 基线出现过的 2.32 s maintenance 尖峰属于另一个未插桩运行，
本 trace 没有覆盖，不能拿本次 138 ms 样本代替它的根因分析。

派生分析保存在诊断目录的 `trace-full-analysis.json` 和
`trace-thread-overlap-audit.json`；后者绑定原 trace SHA256，并附可重放的
`audit_trace_threads.py`。原始 trace、冻结源码和测试状态未修改。
