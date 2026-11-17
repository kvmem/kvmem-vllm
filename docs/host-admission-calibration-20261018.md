# FP8 八并发的主机启动余量校准（2026-10-18）

此前候选在启动模型之前被固定的 32 GiB 额外主机余量挡住；没有发生实际 OOM。
本次按用户要求校准这一启动条件，并为优化前后两组重新运行正式窗口。
历史容量、模型执行参数及运行中的内存保护保持原协议。

## 依据与计算

校准来源是已通过完整生命周期、输出记账和 FP8 字节资格的
`artifacts/aligned-fp8-agents-20261018-baseline-r3`。使用整个服务生命周期的
`gpu_samples.jsonl`，包含加载、冷 prefill、预热、300 秒窗口和排空；不是只取
decode 阶段，也不是将进程 RSS 求和作为物理内存占用。

| 来源观测或预算 | 数值 |
| --- | ---: |
| 内存采样 | 922 次，覆盖 1012.72 秒，最大间隔 1.223 秒 |
| 准入阶段可用内存的保守高值 | 106.2053 GiB |
| 全程最低可用内存 | 19.5158 GiB |
| 上述两个边界的最大系统内存下降量 | 86.6895 GiB |
| 完整 8 用户 × 256Ki 历史及索引预算 | 74.0166 GiB |
| 原额外启动预留／原门槛 | 32 GiB／106.0166 GiB |
| 新额外启动预留／新门槛 | **28 GiB／102.0166 GiB** |
| 在新门槛启动、重现来源内存下降量时的估计余量 | **15.3271 GiB** |

算法保持运行时 8 GiB `MemAvailable` 保护线，再要求预测余量至少多 4 GiB。
只按整 GiB 调整，并将本次最大下调幅度限制在 4 GiB；不会为了碰到当前空闲
内存而持续降低门槛。完整历史和索引信用不按实际较短输出缩减。

上述下降量是全系统观测，含其他进程变化，不能精确归因给推理进程。1 秒级
采样可能遗漏更短的峰值，预测不是任意负载的内存保证。因此继续保留运行时
保护、监控失败即终止工作及自有进程清理；新结果另存全程内存证明。

校准文件位于
[calibration.json](../artifacts/host-admission-calibration-20261018-r1/calibration.json)，
绑定来源基线资格、原始采样、时序和结果的 SHA256。加载、采样覆盖、保护线
或完整历史证明不合格时拒绝生成校准；改变模型、设备或回放协议时拒绝复用。
旧实验和默认未校准路径仍使用原 32 GiB 条件，原失败记录保留。

## 实现与验证

- `scripts/host_admission_calibration.py` 实现确定性计算及全程内存报告；
  `scripts/benchmark_aligned_fp8_agents.py` 增加显式校准参数与冻结绑定。
- 模型运行时沿用已通过资格的完整冻结源。只有测试控制器的启动余量改变，
  因而可以显式复用完全相同运行时、模型和上下文配置的 FP8 资格；源码、
  profile、字节 oracle、输出记账或受保护的退出证明不匹配时仍拒绝。
- CPU 相关回归 **171 passed**，另有比较器检查 **15 passed**。前者包含已有
  async/cleanup 和基准门禁，集合有重叠，不与先前广泛测试计数相加。
  日志保存在 `artifacts/host-admission-calibration-20261018-r1/`。
- 正常本地环境的 CPU 回归关闭 GPU。第一次沙箱执行停在异步内部唤醒，已只
  中断该测试进程后重跑；未修改测试断言来跳过失败。
- 新比较入口 `scripts/compare_calibrated_fp8_agents.py` 要求两组校准文件、
  控制器、预算和推理参数一致，并重新验证资格、完整历史、输出及资源清理。
  它不会拿新候选直接与旧 32 GiB 门槛下的结果生成正式收益。
- 复用资格的清理收据仍绑定原始受保护的 owner，不会改写旧收据来绑定新性能
  manifest。比较器校验原 owner 的冻结控制器、运行时和清理记录，并要求其
  运行时与新实验完全相同；新增测试覆盖 owner、运行时、收据替换和残留进程。

## 新对照与复现

最初的两个实验目录为 `calibrated-fp8-agents-20261018-baseline-r1` 和
`calibrated-fp8-agents-20261018-candidate-r1`。两组均为原 8 条轨迹、两轮预热、
K64/G32、每用户 256Ki CPU 历史、单轮自然输出上限 16Ki、FP8 权重及 KV、
MTP(k=4)、compiled/CUDA Graph、300 秒计量。先运行基线，再运行各独立候选。
单个 A/B 顺序只能提供观察结果，不足以宣称稳健的吞吐提升。

新基线已完成正式窗口及排空，全部完整历史、native/client 输出及资源释放
门禁通过，零抢占。窗口内接受 43,720 token，即 **145.73 token/s**；实际公共
prefill 完成率为 **1975.93–1991.00 token/s**。全程最低可用内存为
**21.1812 GiB**，高于 8 GiB 运行保护线 13.1812 GiB。GPU pool 仍为 779 blocks，
CPU 历史准入容量仍为 99,999,350,784 字节，与旧基线一致。

校准调整启动条件，不减少实际模型工作。新旧基线的单次差值不能归因给门槛
调整；新候选须与本次新基线比较。

候选 `calibrated-fp8-agents-20261018-candidate-r1` 后来触发了原有 8 GiB 运行时
保护，控制器正常终止并证明 GPU 进程为空，没有实际 OOM。它在预热后进入了
约 54 秒计时，未完成固定 300 秒，不纳入性能比较，也不计算该短窗口的收益。
已发布历史峰值为 58.56 GiB、检查点约 2.36 GiB，与新基线的 59.71／2.36 GiB
接近，但系统可用内存明显更低。后续分配器调查与修复见下一节；没有通过减少
历史或继续下调保护线掩盖问题。

## 候选的物理内存调查与修复

独立的 `host-memory-model-diagnostic-20261018-r1` 使用同一候选、8 条完整轨迹
和原 8 GiB 保护线，加入分配器观测及一次清理探针；其结果明确标记为诊断，
不能计入性能比较。探针前，进程 RSS 约 89.1 GiB，普通分配器报告约 21.7 GiB
空闲堆空间。Python GC 没有明显效果；`malloc_trim` 使 RSS 实际下降
**21.49 GiB**，历史与检查点字节不变；随后释放 pinned 缓存几乎没有额外收益。
这证明其中大量内存是仍驻留的普通分配器空闲页，不能都算作新增的历史 KV。
观测只发生在进度发布点，不能据此重建连续的内存峰值。

该清理调用本身用了 **1.68 秒**，因此生产路径没有加入全局 trim。修复改为在
`RequestAuthority` 内复用有界的临时 raw K/V 封页缓冲区，避免大块临时分配与
长期存活的小块索引反复交错。不可变 archive、MeanK 和检查点继续独立拥有。
归还必须等真实 archive Future 完成，或回退／取消的 generation lease 回收
完成；后台 GC tombstone 同样负责归还。最大数量仍受 `resident_records` 限制，
成功关闭 authority 后缓冲池归零，不改变原始 FP8 历史字节及评分路径。

同一份 CPU 流式实验、4 用户各 64Ki 完整历史的结果如下。它不包含真实模型，
不能从组件结果推断服务吞吐或正式 8 用户内存安全。

| CPU 流式路径 | 发布完毕 RSS | 分配器空闲堆 |
| --- | ---: | ---: |
| 原基线 | 12.98 GiB | 3.15 GiB |
| 仅跳过完整封页重复求和 | 12.97 GiB | 3.11 GiB |
| 加入有界缓冲区复用 | 10.17 GiB | 约 0.05 GiB |

所有路径保留相同的 9,126,805,504 字节 archive，首尾历史 raw K/V 和 MeanK
检查通过。完整诊断和流式结果、冻结源码绑定及 SHA256 见
[causal-summary.json](../artifacts/host-memory-investigation-20261018-r1/causal-summary.json)。
新实现通过 117 项历史生命周期测试及另外 273 项图评分／恢复／上下文回归。
最初回归发现的 checkpoint 测试租约未释放，以及后台回收回退尾页的缓冲区
归还遗漏均已修正，失败日志保留。新的真实模型资格与正式窗口使用独立目录
`qw3-alignment-20261018-pooled-copy-natural-async-r1`、
`calibrated-fp8-agents-20261018-candidate-r2`；旧失败记录不改写。

### 缓冲池候选的实际验收结果

新上下文资格通过两用户各三轮以及非法请求隔离和显式释放；新 FP8 资格通过
6 个完整历史请求、4 次 target/MTP KV 冷恢复、4 次 recurrent 冷恢复、两次
rolling 字节检查。native/client 均为 12,290 token，零抢占，停止后 GPU 为空。
与先前 rolling 资格相同，它没有注入取消，资格末尾一 token 探针会话依赖退出
服务清理；正式服务窗口另有显式释放与资源归零证明。

`candidate-r2` 随后完整跑完 300 秒及 28.59 秒排空，完整历史、输出计数、会话
释放和退出检查通过。采样覆盖全服务阶段 900.85 秒、共 820 次，全程最低
可用 CPU 内存 **26.5678 GiB**，高于原 8 GiB 保护线约 18.57 GiB。这次校准
后的 CPU 准入和运行保护已通过实际八用户验证。

**但该组未通过 GPU 部署验收，不能作为正式性能收益。** GPU 的最低采样空闲
显存只有 1,971 MiB，低于原 4,096 MiB 要求；`deployment-acceptance.json`
保留 `accepted=false`，严格比较器按预期拒绝生成完成的 A/B 收益。窗口原始
输出率 177.64 token/s、公共 prefill 1683.04–1703.52 token/s 仅为这次未通过
部署条件的观察值。对照输出率为 145.73 token/s，但自然输出长度、请求完成数
和 prefill 工作量不同，不能把差值直接视为等量工作加速。

额外约 3.3 GiB GPU 占用在预热期形成，并在正式窗口中持续保留。现有 NVML
记录本身不能判断它是存活张量、可释放缓存还是碎片；后续独立的只读 CUDA
allocator 诊断保存在 `gpu-memory-model-diagnostic-20261018-r1`，不计入性能。
CPU 门槛校准完成不等于 GPU 余量也已通过；两条保护线均不降低。

### GPU 空闲缓存调查与显式回收策略

完整长历史、原 8 用户预热后的独立快照显示，PyTorch 实际张量约 78.49 GiB，
预留约 89.83 GiB。普通池内 **11.04 GiB** 是完全没有活跃分配的缓存段；普通
池内未完全空闲段中的空余约 **0.23 GiB**。这次主要可回收空间是空闲缓存，
不能将它都解释为碎片，也不能将独立诊断的分配顺序等同于原失败运行。

新增可选的 `cuda_allocator_conf` 启动参数。默认 `null`，沿用既有分配器设置；
本轮候选显式选择原生 allocator 的 `garbage_collection_threshold=0.9` 和
`per_process_memory_fraction=0.99`，在原生分配路径按需回收老的空闲缓存段。
0.99 是 Torch 进程级上限，不是 K/G 或用户历史预算。安装的 Torch 2.13 源码
只有在进程 fraction **小于 1** 时才设置 GC 使用的边界；只写 GC 阈值会没有
预期效果。依据见 [实际安装版本源码](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L1656)
和 [PyTorch 分配器文档](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。
启动器在导入 vLLM/CUDA 前一致设置新旧环境变量；显式配置优先，未启用时不
覆盖调用方环境。该策略不在 decode 循环中调用 `empty_cache`。

先用缩小到 4% 的进程边界、20% GC 阈值验证行为：保持 512 MiB 活跃张量，
依次申请／释放 128–640 MiB 临时张量，无 GC 时最终预留 2432 MiB，开启后
为 1152 MiB，4 次底层释放，零 OOM、零分配重试，活跃张量内容不变。保留一张
活跃 CUDA Graph 后重做，两组为 2434／1154 MiB，Graph 回放结果同样正确。
这些是缩小规模的功能测试，不能作为真实模型 90%／99% 策略的性能结果。
冻结诊断、原始快照和功能测试哈希见
[gpu-causal-summary.json](../artifacts/host-memory-investigation-20261018-r1/gpu-causal-summary.json)。

相关 CPU 回归为 230 passed，图配置和弹性入口复验 114 passed，最终比较器
27 passed；集合有重叠，不累加。沙箱 CPU 异步唤醒停滞的两个自有测试进程已
中断，正常本地环境禁用 GPU 后通过，原始日志保留。

新候选 `calibrated-fp8-agents-20261018-candidate-r3` 明确声明新增启动器源文件
和这一项执行配置差异。比较器继续要求原完整历史、生成参数、CPU 预算、
102.02 GiB 启动条件和 8／4 GiB 保护。控制器仅允许准备／校验函数变化，
实际计时、测量、排空和清理函数及其他全局执行代码的 AST 必须一致；所有
共享回放代码继续逐字节相同。最终还要求实际原生 KV pool 与对照相同，不能
靠缩小 KV 池获得显存余量。新源已完成两用户各三轮上下文资格；FP8 精确字节
资格与完整 300 秒窗口独立验收，结果见下一节，不能用组件测试替代。

## 最终完成的同协议 A/B

最终有效对照是 `baseline-r1` 与 `candidate-r3`。后者已通过新源的独立 FP8
资格：6 个完整历史请求、4 次 KV 冷恢复、4 次 recurrent 恢复、两次 rolling
字节检查，native/client 均为 12,290 token、零抢占，实际双路纯 decode 覆盖
1,559 步。与前文资格相同，不宣称已注入验证取消。正式 C8 窗口另外完成显式
release 和全资源归零，停止后 GPU 进程为空。

两组均完成固定 300 秒，完整历史、accepted-output/native 对账、CPU-only
存储、零抢占、显式释放、主机保护和 GPU 部署余量全部通过。实际 GPU KV pool
均为 **779 blocks、block size 1616**，没有通过减小 KV 池获得余量。候选全程
内存采样覆盖 929.33 秒、845 次，包含加载、冷准备、预热、窗口和排空。

| 指标 | 优化前 baseline-r1 | 缓冲池＋显式 CUDA GC candidate-r3 |
| --- | ---: | ---: |
| 窗口 accepted 输出 token | 43,720 | 44,086 |
| 服务窗口输出率 | 145.73 token/s | 146.95 token/s |
| 实际公共 prefill 完成率 | 1975.93–1991.00 token/s | 2044.77–2066.13 token/s |
| 窗口内完成／发起请求 | 93／101 | 89／96 |
| 可观测 TTFT p95 | 3.45 秒 | 2.61 秒 |
| 输出 chunk 间隔 p95 | 0.643 秒 | 0.641 秒 |
| NVML 采样利用率 | 87.31% | 87.70% |
| NVML 低于 10% 的采样时间占比 | 2.40% | 1.47% |
| 全服务阶段最低 CPU 可用内存 | 21.18 GiB | 26.45 GiB |
| 全服务阶段最低 GPU 空闲显存 | 5.16 GiB | 5.94 GiB |
| 窗口后排空 | 77.44 秒 | 68.30 秒 |
| 抢占 | 0 | 0 |

输出率只增加 **0.84%**，完成轮次反而少 **4.30%**；两组自然输出长度不同，
候选也在窗口内执行了更多公共 prefill。**没有证据证明稳定的端到端吞吐提升。**
公共 prefill 这一行是实际完成工作除以整个服务窗口，并非隔离的 prefill kernel
速度；TTFT 只表示本轮观测分位数，不承诺所有长请求满足该延迟。NVML 不是 SM
occupancy，不能把低利用率采样直接等同于全部 GPU idle。

纯 decode 步骤中的平均请求数为 **5.82 → 6.47**，但属于不同自然轨迹下的
观测，不是已隔离的因果收益。相同 batch=8、context bucket=65536 的模型步
中位数为 **57.06 → 57.16 ms**，对应样本 27／157；host maintenance 中位数
1.438／1.424 ms。batch=6 的模型步也为 51.03／51.14 ms，基本一致。新候选
mixed 步骤有 377 次，模型步中位数约 577.55 ms；对照为 355 次、573.16 ms。
混合长 prefill 的成本远大于普通 decode，CPU 完整封页组件的降耗没有在这次
服务窗口中转化为明确的整体加速。MTP 接受长度、实际上下文及恢复重叠没有被
这些粗粒度分桶完全控制，不能将差值都归因给某个模块。

本轮确认的收益是完整封页 CPU 组件减少重复求和、有界临时缓存减少主机堆
滞留、以及显式 CUDA 策略控制未使用缓存，使同样 KV 容量的八用户运行通过
资源余量验收。CPU 组件的约 50% 降耗与 CUDA 小规模回收数值仍只适用于各自
组件测试，不换算成服务加速；原未通过 GPU 余量的 177.64 token/s 也不纳入。

只读比较器复核了 **2738** 份证据，完整结果和相同 decode 分桶数据见
[completed-comparison.json](../artifacts/host-admission-calibration-20261018-r1/completed-comparison.json)。
保留所有失败尝试和原始状态。各实验根目录的通用 `comparison_complete=false`
来自原先“同一根目录两 arm”控制器；这里使用独立两目录比较，不改写旧状态，
以本次独立报告及每组 `deployment-acceptance.json` 为准。

下面复现精确冻结的候选，使用新输出目录和空闲端口。显式复用相同运行时的
已完成 FP8 资格，原清理收据继续绑定原 owner，不改写它。更改生产源码时必须
重新冻结上下文及 FP8 资格，不能沿用这个选项跳过验证。

```bash
CUDA_GC_REFERENCE="$PWD/artifacts/calibrated-fp8-agents-20261018-candidate-r3"
CUDA_GC_REPRO="$PWD/artifacts/calibrated-fp8-agents-repro-gc"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CUDA_GC_REFERENCE/runtime/rolling-candidate-v2/src" \
  .venv/bin/python -B \
  "$CUDA_GC_REFERENCE/frozen/scripts/benchmark_aligned_fp8_agents.py" \
  --prepare-only --output "$CUDA_GC_REPRO" --port 18664 \
  --reference artifacts/fp8-weight-performance-20261017-r1 \
  --runtime "$CUDA_GC_REFERENCE/runtime/rolling-candidate-v2" \
  --alignment-qualification artifacts/qw3-alignment-20261018-pooled-gc-natural-async-r1 \
  --fp8-qualification-reference "$CUDA_GC_REFERENCE/qualification" \
  --baseline artifacts/calibrated-fp8-agents-20261018-baseline-r1 \
  --host-calibration artifacts/host-admission-calibration-20261018-r1/calibration.json \
  --cuda-allocator-policy native_gc_90 \
  --allow-production-change src/vllm_kvmem/serving/authority.py \
  --allow-production-change csrc/bindings.cpp \
  --allow-production-change src/vllm_kvmem/_native.cpython-312-x86_64-linux-gnu.so \
  --allow-production-change src/vllm_kvmem/serving/serve.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CUDA_GC_REPRO/runtime/rolling-candidate-v2/src" \
  .venv/bin/python -B \
  "$CUDA_GC_REPRO/frozen/scripts/benchmark_aligned_fp8_agents.py" \
  --run-prepared --output "$CUDA_GC_REPRO"
```

重新实验使用新目录，保留所有既有尝试。以下是基线示例；新候选必须在这个
新基线全部完成后准备，绑定其路径并声明原有三项生产代码差异。

```bash
HOST_CALIBRATION_OUT=/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/host-calibration-repro.json
FP8_BASELINE_OUT=/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/calibrated-fp8-baseline-repro
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B scripts/benchmark_aligned_fp8_agents.py \
  --calibrate-host --reference artifacts/aligned-fp8-agents-20261018-baseline-r3 \
  --output "$HOST_CALIBRATION_OUT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B scripts/benchmark_aligned_fp8_agents.py \
  --prepare-only --reference artifacts/fp8-weight-performance-20261017-r1 \
  --runtime artifacts/aligned-fp8-agents-20261018-baseline-r3/runtime/rolling-candidate-v2 \
  --alignment-qualification artifacts/qw3-alignment-20261018-natural-async-r2 \
  --fp8-qualification-reference artifacts/aligned-fp8-agents-20261018-baseline-r3/qualification \
  --host-calibration "$HOST_CALIBRATION_OUT" --output "$FP8_BASELINE_OUT" --port 18632

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FP8_BASELINE_OUT/runtime/rolling-candidate-v2/src" \
  .venv/bin/python -B \
  "$FP8_BASELINE_OUT/frozen/scripts/benchmark_aligned_fp8_agents.py" \
  --run-prepared --output "$FP8_BASELINE_OUT"
```

运行后的 `host-memory-envelope.json` 记录真实准入值、全程最低余量与保护线。
最终仍需完整历史、accepted-output/native 计数、零抢占、显式会话 release 和
GPU 进程退出证明，才能把某组结果纳入性能比较。
