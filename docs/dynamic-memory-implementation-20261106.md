# 动态分配与分层存储实现和验证

实现日期：2026 年 11 月 6 日。对应[实现方案](dynamic-memory-tiering-plan-20261105.md)，在现有工作区上增加可选模式，默认仍为原有静态准入。K/G、选块算法、mandatory span、Query replay 和刷新判定沿用现有实现。

本文保存第一阶段的实现边界和实验结果。后续增加的会话换出、暂停快照 SSD 分层和调度优化见[第二阶段报告](dynamic-memory-phase2-20261106.md)；下文“第一版”限制不代表第二阶段的全部能力。

## 已实现的资源路径

| 阶段 | 代码 | 行为 |
| --- | --- | --- |
| 资源账本 | `resource_ledger.py`、`capacity.py` | 区分已分配、已承诺和推进专用余量；带完成确认的释放；GPU 物理页与未来额度分开报告 |
| CPU/SSD 历史存储 | `tiered_storage.py`、`archive_limits.py`、`storage.py` | CPU 按实际 payload 使用；达到高水位后写回冷记录，降到低水位；SSD 文件按写入增长 |
| 请求状态换出 | `request_checkpoint.py`、`request_swap.py`、`request_swap_worker.py` | 原样复制当前 FP8 attention KV、全部 recurrent/MTP 行与 accepted counter；完成后才归还源页面 |
| GPU 增量准入 | `incremental_capacity.py`、`request_swap_control.py` | 按最终不可分割操作申请 native pages；保留 K+G 上界检查；压力下暂停和恢复请求 |
| 运行接入 | `scheduler.py`、`connector.py`、`runner.py`、`logical_capacity.py` | 零 token 控制步骤传输指令；保留原生 request slot；恢复完整页表后继续同一个请求 |

GPU 总 KV backing 仍由原生 vLLM 预先建立共享池。动态模式改变的是**请求在池内占有的页及未来增长额度**，并不把 GPU 变成逐 token 调用 CUDA 分配器。`max_num_seqs` 仍限制执行槽，暂停请求保留槽；本版不会以换出为由绕过槽位上限。

CPU 历史采用独立记录位置目录。CPU 和 SSD 之间移动不会删除逻辑记录，也不会重选块。落盘成功且读者排空后才能移除 CPU 副本；读取校验 generation 和 checksum。短写、截断、checksum 错误和磁盘写失败会显式失败，不能丢弃唯一有效源。文件在创建后立即 unlink，进程退出后由系统关闭；SSD 提供本进程的后备存储，不支持重启恢复。

历史准入的整轮逻辑额度在 `tiered` 模式下移到 SSD 容量，CPU 不再按所有请求最大历史占用锁定。最底层仍有有限保证：`nvme_max_bytes` 和 repository 记录上限决定能接收多少历史；磁盘不足不会通过静默删历史来掩盖。本版不扩容 repository 元数据，也不将索引分页到 SSD。

## 暂停恢复的边界

状态流转为 `capture → paused → restore → drop → runnable`。调度器先停止该请求的新计算，worker 排空其 capture/publication 并同步 GPU，再保存当前 dense KV 字节。保留同一个 request slot，使 sampler、RNG、请求 token、Mean-K/Action 状态和逻辑映射继续属于原请求。恢复只安装新物理页表和原始字节，不进行 raw-K 重建或新的选块。

取消请求时先保留原请求对象和 worker slot，等待在途复制及 checkpoint drop 完成，再执行正常终结和归档回收。已暂停、无会话绑定的请求可直接丢弃 checkpoint；已开始恢复的请求先排空恢复再终结。

checkpoint 使用独立的有界 CPU pinned pool，不占用历史 CPU 缓存额度，当前也不进入 SSD。调度器按真实 native backing page 大小保守计费，worker 按实际 snapshot 字节计费。初始化要求该池能够保留一个最坏窗口，同时留下足够的推进余量以换出恢复目标所需的其他请求。恢复前暂时保证该请求既有 K+G 上界所需的页，但只分配快照实际占用的页；余下页留给恢复后的第一个完整操作，实际计算前进后立即解除保留。同一时刻只有一个恢复请求享有这项首步保护。已恢复请求还必须满足最短驻留步数，才允许再次换出。

第一版以正确性为优先，每次仅一个迁移控制指令在途，capture 前有完整 GPU 同步。大量迁移可能降低吞吐，不能把可接纳请求数提高直接等同于加速。

## 配置和适用范围

从[动态模式示例](../examples/serve-qwen-mtp-dynamic.json)开始，替换模型路径。其 K/G 均为 2,048，与普通 MTP starter 相同；显式 1.4 GB KV 池用于制造资源压力，不是所有模型通用的容量建议。启动时按本机真实页几何检查能否容纳至少一个完整 K+G 窗口及状态。

新增设置均在 `kvmem` 下，最后一项为顶层设置：

| 设置 | 默认值 | 用途 |
| --- | --- | --- |
| `archive_backend` | `nvme`；MTP 示例显式为 `cpu` | 新增 `tiered`，启用 CPU 优先、SSD 后备 |
| `tiered_segment_bytes` | 64 MiB | SSD 段的目标逻辑大小，至少一条完整记录；payload 按需写入 |
| `tiered_cpu_low_watermark` / `tiered_cpu_high_watermark` | 0.70 / 0.85 | CPU payload 写回回滞；不是进程总 RAM 水位 |
| `gpu_memory_admission` | `static` | `incremental` 启用操作额度和请求换出 |
| `request_checkpoint_bytes` | 0 | 动态模式必需的共享 pinned checkpoint 上限 |
| `request_swap_min_resident_steps` | 32 | 恢复后的最短驻留步数；还要求实际计算前进 |
| `kv_cache_memory_bytes` | `null` | 可选的原生 GPU KV backing 字节数，用于明确容量或对照实验 |

`tiered` 要求 CPU、SSD、archive I/O 均有正额度，`cpu_history_coordination=true`，并禁止全盘预分配。现有 CPU 专用 direct seal、prepared reader 等快路径继续由原配置规则拒绝；分层读取使用已有有界 full-record fallback。

动态 GPU 第一阶段支持单 rank、APC 关闭、`mamba_cache_mode=none` 的同步 target/native MTP；公开 launcher 限定 eager MTP。HTTP 请求必须无 `session_id`。会话导入、异步刷新/恢复、rolling window、历史 GC、guided probes、deferred/coalesced prefill、SystemOne fork、图模式等未经组合验证的路径不允许开启动态模式。已有静态模式继续处理原来支持的配置。

各项 CPU 设置是分别计量的资源上限。历史 payload、pinned I/O、请求 checkpoint、会话 checkpoint、索引、runtime、分配器保留内存和 OS page cache 不应混为一个额度。本版不能把 `cpu_read_cache_bytes` 宣称为进程 RSS 上限。

运行中 `archive_resources.json` 最多每秒更新 10 次，独立于退休时的完整 `archive.json`。它报告实际 payload、I/O、SSD、worker RSS 和 cgroup 指标；RSS/文件缓存属于不同且有重叠的统计范围，不能直接相加。阶段采样峰值、进程生命周期峰值和无有效采样分别标注。

## 验证方法

新增单元及组件测试覆盖：原子额度申请与回滚、重复/过期确认、GPU 多组完整分配、G 边界和 MTP 保留行、物理释放前的 fence、CPU/SSD 字节一致性、共享容量跨请求使用、读取中退休、短写和 I/O 故障、checkpoint slot/generation、原生零 token 控制步骤和取消。

`scripts/benchmark_dynamic_memory.py` 冻结源代码、profile 和每个 HTTP 输入，顺序启动独立服务器。默认运行 CPU → tiered → tiered → CPU；每臂先预热，再测 1/2/4 并发。记录完整输出 token IDs、TTFT、完成延迟，以及运行中的 GPU owners/pages、CPU payload 峰值、SSD 读写和 swap 次数。计时从第一条请求发出到最后一条公开响应完成，包含排队、prefill 和 decode，排除模型启动与最终回收等待。每个阶段结束后仍必须验证回收完成。

分析同时检查同模式重复和跨模式输出。缺少重复、资源采样或存在同模式漂移时返回 `needs_review`；稳定重复下的跨模式输出差异返回失败。该合成输入实验用于实现一致性和资源性能，不替代模型任务质量评估。

运行示例：

```bash
PYTHONPATH=src python scripts/benchmark_dynamic_memory.py \
  --profile /absolute/path/to/profile.json --output /absolute/path/to/new-result \
  --arms cpu tiered tiered cpu --levels 1 2 4

PYTHONPATH=src python scripts/benchmark_dynamic_memory.py \
  --profile /absolute/path/to/fixed-gpu-profile.json --output /absolute/path/to/new-result \
  --arms incremental tiered tiered incremental --levels 1 2 4 --cancel-stress
```

第二个 profile 必须给出相同的 `kv_cache_memory_bytes`。只观察到更多请求同时存活不算换出验证，必须观察 `request_swaps.pauses/restores`，并检查恢复后的输出和最终所有权归零。

`--cancel-stress` 在最后一个动态服务器上额外执行真实断连测试，性能计时不包含它。脚本先取得独立参考，再同时提交四条请求；只有在指定请求出现在 `capturing/paused/restoring` 状态时才关闭其 HTTP 流，并要求原生终结记录为 `FINISHED_ABORTED`。其余请求和后续新请求必须与参考逐 token 一致，GPU 页、checkpoint、历史存储和退休队列最终归零。

主机端联合回归已通过 679 项，CPU 存储兼容性另有 82 项通过，另一个 CUDA 项在无 GPU 的命令中跳过。独立 CUDA FP8/GDN 字节恢复测试已通过。确定性混合请求长跑覆盖 48 组原生页池组合、384 请求、1,299 次迁移 ACK 和 88 次取消，全部在有界步数内结束并回收资源。该长跑模拟 worker 完成时机，真实模型性能单独测量。

## 首轮模型对照

环境为 RTX PRO 6000 Blackwell、当前工作区的本地 Qwen3.8-27B NVFP4 模型、FP8 KV、native MTP 1、eager、K/G 2,048/2,048。每请求 8,192 输入 token 和 128 输出 token，CPU 历史缓存 768 MiB，SSD 上限 8 GiB。两组使用相同的量化、选块和刷新配置。

首轮 `serving-abba-r1` 的 28 条测量响应全部完成，同模式重复及所有跨模式比较逐 token 一致。吞吐按两次实验总输出 token / 总耗时计算：

| HTTP 并发 | CPU token/s | tiered token/s | 相对变化 |
| ---: | ---: | ---: | ---: |
| 1 | 7.551 | 7.482 | −0.91% |
| 2 | 10.549 | 10.382 | −1.58% |
| 4 | 10.570 | 19.447 | +83.98% |

这一轮证明的是分层历史准入收益，GPU 仍使用原有生命周期预留。该轮缺少运行中峰值采样，不能据此声称实际同时 GPU 驻留量；其分析状态因此保留 `needs_review`。

## 分层存储的重复验证

`serving-abba-r2` 使用相同负载再次进行 CPU → tiered → tiered → CPU。所有输出和同模式重复依然逐 token 一致。4 并发的两次重复结果为：

| 指标 | CPU | tiered |
| --- | ---: | ---: |
| 合并输出吞吐 token/s | 10.579 | 19.303 |
| 每轮完成全部 4 请求的时间 s | 48.27 / 48.53 | 26.50 / 26.55 |
| 实际同时 GPU 驻留请求峰值 | 2 | 4 |
| 实际请求占用 native pages 峰值 | 18 | 36 |

该条件下输出吞吐提高约 82.5%。tiered 的 768 MiB payload 额度内，记录到的累计真实峰值为 682,950,656 bytes；第一轮 C4 写盘 721,944,576 bytes、读盘 209,453,056 bytes。原 CPU 后端因为历史整轮额度只能同时接纳两个请求，GPU 本身仍有充足空间。因此这项收益来自解除 CPU 历史预留瓶颈，不能归因于 GPU 动态换出。

原 r2 的 C1 运行中 CPU 采样为零，故完整分析保留 `needs_review`；明确筛选原 C4 两次重复的 `analysis-c4.json` 通过输出和资源验证。之后采样入口已改到每步发布反馈，避免仅在退休时记录数据。

本机原始证据位于 `artifacts/dynamic-memory-20261106/`，包含冻结源码清单、输入/输出、profile、逐阶段资源与时序、测试 JUnit 和服务器日志。目录为本机实验产物，不作为可移植模型权重或重启快照。

## 固定 GPU 池的动态准入结果

`gpu-admission-abba-r1` 使用同一个冻结源码和相同的 tiered 历史后端，只切换静态/增量 GPU 准入。顺序为 incremental → tiered static → tiered static → incremental，每臂都有预热及 1/2/4 并发测量。K/G、精度、选块、刷新设置和 1.4 GB KV backing 额度均相同。原生页池实际为 25 页，其中一页为不可分配的 null page，每个完整 K+G 请求上界为 9 页。

最终分析状态为 `passed`：同模式重复稳定，全部 28 条测量响应及跨模式比较逐 token 一致，所有阶段都取得有效运行中资源采样。

| HTTP 并发 | 静态准入 token/s | 动态准入 token/s |
| ---: | ---: | ---: |
| 1 | 7.403 | 7.391 |
| 2 | 10.246 | 10.244 |
| 4 | 10.220 | 10.062 |

4 并发下，实际同时 GPU 驻留请求峰值由 **2 增至 3**，请求页占用峰值由 18 增至全部 24 个可用页。两次动态 C4 各完成 6 次换出和 6 次恢复，结束时均无遗留 checkpoint、请求或页面。

这个负载下动态模式吞吐约低 **1.55%**；TTFT 中位数由约 30.6 秒增至 34.5 秒。因此本版验证了超额准入、精确切换和资源回收，不宣称 GPU 换出加速。worker RSS 观测峰值约为静态 4.22 GiB、动态 6.05 GiB，包含 checkpoint、历史、模型映射及分配器等成本，不是单独的 archive 占用。后续性能优化应优先减少同步和不必要的轮换，并在保持当前 K/G 与刷新判定下重新做对照。

## 真实取消与后续请求

最后一个动态服务器完成性能测量后，执行了独立的 8,192 输入 / 256 输出 token 取消负载。四条请求同时到达，其中一条在历史位置 128、`paused` 状态下被真实关闭 HTTP 流；服务器随后记录 `FINISHED_ABORTED`。其余三条各完整输出 256 token，与独立参考逐 token 相同；随后新提交的一条请求也完全一致。

最终 `requests=0`，所有 24 个可用 GPU 页归还，checkpoint live/promised 均为零，历史额度和退休状态归零。结果保存在 `gpu-admission-abba-r1/04-incremental/cancel-stress/result.json`。其他取消阶段由组件回归和延迟 ACK 长跑覆盖；本次真实 HTTP 取消明确覆盖 paused 阶段。

以上第一阶段的可用交付是上述受限范围的实验模式。当时会话绑定请求的动态 GPU 换出、暂停 checkpoint 的 SSD 分层、跨 rank、异步刷新组合、索引分页、进程总 RAM 压力调节和掉电恢复属于后续阶段；后续实现与验证范围以对应记录为准。

## 后续 checkpoint 关闭异常审查

2026 年 11 月 6 日对第二阶段 `checkpoint_storage.py` 的 SSD 暂停快照路径补充故障审查，确认并修复两个关闭异常问题。此节是存储组件的修复记录，不更新前述整模型验证范围，也不代表逐层召回与动态换出组合已通过整模型验收。

1. 删除磁盘快照时，`close` 可能已经释放文件描述符却仍返回错误。原实现保留旧描述符整数，后续再次删除可能误关已经复用该整数的无关文件。现在记录不可判定的关闭错误，后续操作显式失败，不重试该描述符，也不释放原有资源额度。
2. 后台写盘失败后，清理临时文件的 `close` 也可能失败。原实现可能遗失仍打开的描述符，并在后续删除或关闭时报告成功。现在同样保留失败状态及保守所有权，完整 CPU 源和对应额度保持有效，不能把失败回滚当作成功回收。

`CheckpointStore.snapshot()` 新增两个诊断字段：`close_error` 保存观察到的关闭异常信息，正常为 `null`；`ssd_close_uncertain_bytes` 表示发生关闭不确定性的快照额度，正常为 0。后者是保守所有权标记，可能与 `logical_bytes`、`ssd_used_bytes` 或 `ssd_pending_write_bytes` 重叠，不能相加当作额外实际磁盘占用。异常后的 `drop`、恢复、准入和存储 `close` 均停止推进；不尝试凭描述符整数修复或重新释放资源。

worker 交接也保持失败状态：故障注入确认删除失败时不发 drop ACK、不调用成功释放，checkpoint live 额度仍为 24 bytes；后台回滚关闭失败时，已有 12 bytes live 和待接纳请求的 32 bytes promised 均保留，完整 CPU 源仍存在。`RequestSwapWorker.close()` 在仍有 checkpoint 时返回 `False`，不会宣称已关闭；直接关闭已失败的 store 会报告关闭错误。这些数字来自微型组件测试，非模型运行资源大小。

相关 CPU 回归命令如下，结果为 **97 passed，1 GPU 测试 skipped**。覆盖描述符关闭前失败、关闭后失败及复用、重复删除/关闭不得误关无关文件、后台失败写盘回滚，以及已有精确快照、会话、历史分层和 SSD worker 测试；不包含整模型性能或层级兼容验收。

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  tests/test_checkpoint_storage.py tests/test_request_checkpoint.py \
  tests/test_session_swap.py tests/test_tiered_storage.py \
  tests/test_tiered_reclamation.py tests/test_request_swap_worker_ssd.py
```

## 逐层召回与动态 GPU 页分配接入（2026-11-06）

新增默认关闭的 `kvmem.request_swap_layerwise`。本轮打通 CPU archive、异步逐层累计 attention prefill pressure、顺序逐层 Mean-K Query、物理 prefill 合批及 compiled_graph/MTP4。Active=36864、Gen=16384、block32、sink41、recent0、原始完整输入及采样保持不变。固定 GPU backing 池内部按操作分配页，不表示 CUDA 总显存池会随请求自动缩小，也不消除 KV 组装开销。

交换前等待该请求的私有 Query、重选组装和已接受发布完成；保存 CPU 层映射、累计历史及窗口版本，恢复前后检查不变。原生 checkpoint 仍逐层复制实际 K/V、GDN 与 MTP 状态，不把各层重建成共享块集。修复了一个兼容检查误拒：某请求尚无窗口时，其他请求首次创建全局 window adapter 不应使前者的恢复失败。启动与 worker 入口拒绝关闭异步刷新、original-RoPE、no-probe、rolling decode 或 layer0 MTP 等未验证组合，避免静默失去逐层执行。

本次完整输入来自原 MAB 两个独立冷前缀（411886 / 461542 tokens），仅作为兼容验收，未恢复暂停的正式 MAB、未外发评分。单路对照使用相同48GiB物理页池；双路故意降为4GiB以制造真实交换，语义 Active/Gen 不变。

| 验证 | 结果 |
| --- | --- |
| r2 静态单路 vs 动态单路 | 2/2完整完成；50次pressure、32层Query选块及完整输出全部一致 |
| r2 双路紧张页池 | 2/2完成；在29696处换出，恢复1,858,371,588 bytes逐字节一致；选块0/2严格一致，输出1/2一致 |
| r3 延迟第二请求到达 | 2/2完成；在124704处交换已有16套不同选块的请求，恢复2,364,735,492 bytes逐字节一致；层映射及累计历史检查通过；选块0/2严格一致，输出1/2一致 |
| 最终回收 | 两次真实交换均完成drop；checkpoint live/promised=0；无遗留GPU进程 |
| 回归 | 652项CPU通过、2项GPU专用测试跳过；另行真实CUDA字节恢复测试3项通过 |

r2 单路两题总执行墙钟为187.50秒与186.61秒，包括准入、Query及decode，不是纯prefill吞吐；差别不足以宣称加速。r2/r3强制交换还包含同步字节校验，不能用来报告正式多路吞吐收益。

r3第一题在第二请求进入之前，52992、69120、85248、101376处的四次pressure都与单路参考16/16层一致；并发进入后117504处开始不同，早于124704处的capture。由此确认该例首次差异出现在交换之前的计算阶段。批次形状和调度轨迹确实变化，但没有保存完整中间数值，不能把某个GEMM或浮点归约认定为根因。精确复制通过仅证明恢复字节完整，不等于跨并发选块和输出严格等价。实验报告分别保留交换资格与严格等价状态，不将总体 `passed=false` 改成通过。

证据：

- [完整对照](../artifacts/layerwise-incremental-20261106-r2/qualification.json)、[两路差异审计](../artifacts/layerwise-incremental-20261106-r2/tight-divergence-audit.json)。
- [已有逐层映射后的交换覆盖](../artifacts/layerwise-incremental-after-pressure-20261106-r3/qualification.json)。
- [交换之前的首次差异审计](../artifacts/layerwise-incremental-after-pressure-20261106-r3/post-pressure-divergence-audit.json)：117504处平均块重叠99.68%，其中15层合计换59块，早于124704处capture；本请求此前逻辑分块和attention采样边界一致，两次物理forward已与另一请求合批。
- [最终CPU回归](../artifacts/dynamic-layerwise-compatibility-20261106-r1/final-host-r2.xml)、[真实CUDA验证](../artifacts/dynamic-layerwise-compatibility-20261106-r1/swap-byte-gpu.xml)。
- [冻结之后新增防误用检查的源码哈希](../artifacts/dynamic-layerwise-compatibility-20261106-r1/post-freeze-guards.json)：完整模型验证使用r2冻结版本；之后仅补充未支持配置的拒绝检查，并通过CPU回归，没有修改冻结目录。

r1启动前发现测试驱动降低max_num_seqs后与原16424批量上限冲突，未执行模型请求；失败记录保留。r2保持原max_num_seqs=8，由客户端控制实际1/2路；r3只改变第二请求到达时间，复用r2冻结运行时及只读单路参考，没有因答案差异重采样。该开关继续默认关闭，尚未验收八路严格等价和纯prefill收益。
