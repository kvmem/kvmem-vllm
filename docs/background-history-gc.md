# 后台异步历史 KV 管理

首版将策略与释放机制分开：LRU 只提出删除某个 `(slot, generation)` 的意图；
机制负责关闭新的使用入口，并等待各自已有使用者退出后释放资源。前台 decode
继续使用已经准入的 GPU 窗口，GC 不提交 GPU 工作，也不强制换窗。

首版实现与验收已完成。公开配置、CPU archive、scheduler 整轮容量预约和 P/M 会话
均已接通。Qwen3.6-27B-NVFP4 的 P/M、GDN、原生 MTP、取消、160 MiB 共享
历史配额，以及五组完整功能性能对照均通过。验收范围为 CPU archive + eager。

## 使用

示例配置：`examples/serve-qwen3.6-27b-historical-gc.json`。它使用本机模型路径、
CPU archive、eager 执行、异步 refresh/restore 和 P/M；按机器修改模型路径与显存额度。

```bash
PYTHONPATH=src python -m vllm_kvmem serve \
  --config examples/serve-qwen3.6-27b-historical-gc.json \
  --run-dir artifacts/history-gc-serve
```

`kvmem.historical_gc` 为 null 或省略时保留原有路径；显式对象启用新的 native
ownership、容量预约和共享 worker。对象中的 `mode` 支持：

| 模式 | 行为 |
| --- | --- |
| off | 不运行删除策略；保留新 ownership 和正常退出释放，作为对照 |
| shadow | 计算并统计策略意图，不退休历史块 |
| delete | 提交不可逆退休并在后台安全释放 |

```json
"historical_gc": {
  "mode": "delete",
  "policy": "lru",
  "policy_params": {"keep_mandatory": true},
  "poll_ms": 10,
  "scan_blocks": 256,
  "delete_blocks": 64,
  "ready_low": 0,
  "ready_high": 0
}
```

`ready_low/high` 是每个 repository 可复用 metadata 槽位的水位。`0,0` 自动采用约
5%/10%；显式值必须满足 `0 < low < high <= repository_blocks`。
`scan_blocks` 限制每轮策略观察量，`delete_blocks` 限制每轮意图数量。
示例中的较高水位用于在短历史中实际触发 GC，生产环境应结合保留需求调整。

`policy` 选择删除策略，`policy_params` 只传递该策略的参数；`mode` 独立控制是否
执行删除。`lru` 的 `keep_mandatory` 默认为 `true`，保留
sink/recent 和 P/M frame 等 mandatory 块；设为 `false` 允许这些块参与 LRU 删除，
可能使对应 checkpoint 失效。省略新增字段与原有 LRU 行为一致。未知策略、未知
参数和错误参数类型在启动时拒绝，`off`/`shadow` 也执行配置校验。

新增原生策略实现 `HistoricalKvPolicy`，在 `csrc/historical_policy_factory.hpp`
登记校验和构造入口，并在 `HistoricalGcSettings` 的参数 schema 中登记公开参数。
有状态策略实现 `reset()`，清除轮次状态并保留预分配缓冲区。工厂在服务启动时
为每个 pooled scope 创建独立策略实例，后台复用时调用 `reset()`；decode 不创建
策略对象。已有 C++ 自定义策略注入入口保留，安全退休和资源释放流程由所有策略共用。

当前要求 CPU archive、request planning、async seal/retirement、CPU history
coordination 和 owned sessions。与旧 `repository_gc`、block graph 的组合会在启动时
拒绝。支持异步 refresh/restore 和原生 MTP；首版未验收 CUDA graph serving。
请求须携带稳定的 `session_id` 与租户隔离的 `cache_salt`。

## 主动回收旧工具返回

`tool_result_age` 按会话保留最近 N 条完整工具返回。下面的配置先以 `shadow` 记录回收候选。

```json
"historical_gc": {
  "mode": "shadow",
  "policy": "tool_result_age",
  "policy_params": {"tool_result_age_steps": 12}
}
```

`tool_result_age_steps` 默认 12，计数单位是工具返回消息。第 13 条的最后一个
token 被接受后，第 1 条里的合格页进入后台回收，无需达到容量水位。来源是请求
计划的 `kvmem_message_spans` 或 `kvmem_locality_spans` 中完整的 `tool` 区间。
复用前缀的消息边界沿用会话校验，回退会重新计算完成条数。

只有完整落在同一条到期工具返回里的封存页可以退休。跨消息页、角色不明、
thinking、工具调用以及其它助手内容保留。显式必留区间、sink/recent、当前 GPU
窗口和 P/M 检查点继续受保护。保护解除或归档完成后，后台会再次检查到期页。
已获准的读操作随现有引用退出后释放字节，退休的页永久退出后续历史召回。
在用的 GPU 页随正常换窗或请求结束归还。这条归档回收策略借用了 [AKV 的滚动保留规则](https://arxiv.org/pdf/2610.28577v1#page=8)。

单卡可用 `shadow` 查看 `shadow_decisions`，切换到 `delete` 后用
`tool_result_age_retired` 和 `cpu_released` 分别核对提交和实际释放。
Tensor parallel 走已有的 rank 同步提交。容量吃紧时仍只删除合格的旧工具页，
其余容量缺口由现有准入机制处理。

## 资源与并发契约

1. 策略候选是已封页的历史块。LRU 使用每个 scope 的观察快照，依据最近的接受包
   使用时间排序；共享 CPU 压力跨 scope 记账。这是近似的分 scope LRU，不是全局
   精确 LRU。sink/recent 和精确 P/M frame 的保留属于策略规则。
2. 不可逆退休允许已有引用和 GPU residency。退休后新 pin、CPU read 和新的执行
   准入失败；旧操作仍可完成。旧 read 不能充当新执行许可，旧 lease 不能 rescue。
3. CPU payload、CPU 索引、metadata/GPU 权限分别持有。CPU read/DMA source 权限
   结束后，原生 reaper 释放 CPU 数据；metadata/GPU lease 不阻止这一释放。
4. MeanK 的 FP32 sum 和可选 FP16 mean 直接使用 native 分配的缓冲区，Torch
   视图保留 buffer owner。前台移除 Record 只排队小型 owner，原生 reaper 只释放
   C++ 数组，不持有 Torch/Python 对象。普通逐块回收在索引释放后确认 CPU 回执，
   再归还 metadata；关闭 scope 可以先归还原生 metadata，但整轮 CPU 额度仍需
   等待索引实际释放。没有后台 Python 回调。
5. 已准入的 GPU 窗口可以继续使用退休块，普通换窗移出旧块或请求结束时，由原有
   allocator 归还 GPU 页。GC 不直接 free 正在计算的 GPU 页。
6. slot、CPU allocation、IO publication 和回执都校验 generation；晚到的旧消息不能
   作用于替代 allocation。完成位保持到描述符清理及显式 ACK，不会因队列满丢失。
7. 封页 IO 与 engine 更新分开使用原子发布通道；decode 不等待 IO 发布 guard。
   大块 scope 数组在服务启动时按固定 scope 数预分配，后台清理并复用。
8. 新 turn 在运行前一次预约全部新增输入、最大输出、P/M 两代尾页和两包在途余量。
   CPU 额度包含 KV 和最坏情况下的 MeanK；尾页和待释放索引仍计入占用。只有完成
   worker fence 的 settlement 可替换活跃预约，EOS/取消本身不归还额度。
9. 容量不足时排队在准入边界。scheduler 向空闲 scope 发送 generation 限定的 Ready
   或 CPU 保留目标，LRU 只能提供实际已释放容量；活跃 turn 和 pending GC 不作为
   可用额度。未封页尾块、待释放索引也不能作为 LRU 的回收供给；下发 native
   payload 目标时剔除这些 index-only 预约，避免排队差一个尾块而停滞。下一轮激活后，旧 idle 容量回报失效。
10. P/M checkpoint 的 policy hold 不持有 CPU read。显式退休或自定义策略仍可使
    checkpoint 失效；机制不复活已删 generation。取消半页时仅重建尾页 generation，
    完整前缀与 GDN/MTP 状态保留。

LRU 会减少未来可检索的上下文，不能保证任务语义与保留完整历史完全一致。保护
当前 GPU 权限保证已有计算安全，P/M hold 保证受保留 frame 可恢复。精确保留内容
应通过策略的 mandatory 规则表达。所有历史均受保护且预算不足时，准入仍需等待
会话释放或其他容量变化。

## 张量并行（TP）

TP 下每个 rank 归档自己的 KV head 分片，句柄是 rank 本地的 (slot, generation)；
所有跨 rank 的选择校验都要求同一时刻各 rank 的逻辑记录集 (history_begin, count)
完全一致。因此 TP 只允许 `mode: delete`（或 `off`）且 `keep_mandatory: true`，
由 leader 决策、所有 rank 在同一 engine step 提交：

1. rank 0 的 native worker 以 `PROPOSE` 模式运行策略，只生成 intent，不自行退休；
   rank ≥ 1 以 `FOLLOW` 模式运行，只执行释放任务，从不运行策略。
2. 每个 engine step，runner 在 `advance_publication`（含 `pump_refresh`）之后于
   主线程调用 `tensor_parallel.gc_sync`。rank 0 把待处理 intent 以逻辑块广播；
   空 step 只有一次小广播。
3. 每个 rank 在本地检查但不提交（`external_prepare`）：逻辑块恰好存在一次、已封页
   且已归档、不在当前窗口/recent/hold/mandatory 中、scope 未暂停、所属请求没有
   进行中的 refresh/restore。通过的记录保留 policy gate，scope 在 release 前不确认
   pause。
4. MIN all-reduce 决定后，所有 rank 按相同顺序退休同一集合（`external_commit`），
   并在同一调用中从 `authority.records`、按层历史、action 分数和 query 边界中移除；
   随后 `external_release` 才开始各 rank 异步的 CPU/slot 释放。TP 下 `drain` 只完成
   已移除记录的释放；若活动记录在同步提交之外被释放，直接报错。
5. follower 上逻辑块缺失、重复或几何不同（且在其已发布历史内）时，所有 rank 一同
   抛出 `RuntimeError`，不继续分叉。

rank 0 的策略只看到本 rank 的一半 head（Mean-K 特征、Question、importance 与余弦
分组都是本地 KV head），是 TP=1 策略的近似，并对所有 rank 一致生效。该路径目前只有
CPU 测试（`tests/test_tp_historical_gc.py`，含两进程 Gloo），尚无 GPU 服务验证。

## 指标

`capture/archive.json`、`capture/progress.json` 和 scheduler resources 提供：

- `historical_gc`：published、examined、decisions、retired、cpu_released、
  slots_released、pending_slots、pending_cpu_bytes、retries、stale、policy_failures、
  shadow_decisions、admission_attempts/retries。
- `historical_gc.policy/policy_params`：实际启用的策略名称与参数（公开配置展开默认值）。
- `historical_gc.metadata`：启动分配次数、后台复用次数、预分配 scope 数与容量。
- `cpu_authority_bytes`、`cpu_index_bytes/owners` 及各自 released bytes。
- `cpu_history_admission`：活跃/退休预约、已保留/可用 bytes、deferrals 和 settlements。
- `gc_wait_count/seconds`：旧式 GC 等待指标；新路径不调用该等待接口。须同时检查
  执行准入重试、IO deferrals 和容量预约证据，不能仅凭两个零值推断延迟无变化。

释放表示归还 allocator / GPU 页池，进程 RSS 或显存保留量不一定同步下降。

checkpoint、IO/staging、原生 metadata 池和 GPU cache 有各自的固定预算/指标；
`cpu_read_cache_bytes` 在本模式下约束历史 KV + MeanK 的保守容量，不是进程 RSS 上限。

## 验证与复现

策略名称/参数/工厂入口的扩展由原生与 host 集成回归验证；下方真实模型和性能
数字为首版冻结版本的验收记录，并非这次工厂扩展的重新测量结果。

```bash
cmake -S . -B build/history-gc -DCMAKE_BUILD_TYPE=Debug
cmake --build build/history-gc -j2
ctest --test-dir build/history-gc --output-on-failure
PYTHONPATH=src python -m pytest -q tests/test_historical*.py tests/test_native_memory_storage.py
```

GPU 测试需显式 `KVMEM_RUN_GPU_TESTS=1`。覆盖普通、packed、direct、split 组装、
queued IO 在退休前取得读权限、CPU 先于当前 GPU 工作结束释放、旧 generation 复用，
以及真实模型 K scale 下的逐字节 FP8 重建。后者发现并修复了近似除法跨越 FP8
舍入中点的问题；保留独立 CPU oracle，没有放宽误差标准。

原生 Host CTest、ASan/UBSan/LeakSanitizer 当前为 8/8；TSan 为 4/4。index reaper
新增用例验证最后一个前台 owner 仅排队、C++ deleter 在后台执行、关闭等待旧 owner，
容量拒绝不改变已有存储的所有权；另有 200 次满容量复用，验证容量归还先于
完成回执。原有并发覆盖 1,000 次 CPU 复用、2,000 次旧意图复用、
10,000 次 engine/IO 并行元数据更新、500 次执行准入与退休交错。

相关 host serving 回归为 435 passed、1 skipped、1 个已存在的失败：
`test_v2_requires_a_real_pm_restore_profile` 期待 `P/M sessions`，原实现返回
`CPU P/M or frozen-P sessions`。这个错误文案匹配问题在 HEAD 已存在，本次没有修改。
新增公开配置/索引/准入/async refresh 回归为 49 passed，GPU FP8/历史管理回归为
22 passed。这些数量来自不同范围，不能相加作为不重复的总测试数。

真实 Qwen3.6-27B-NVFP4、RTX PRO 6000 Blackwell 的完整生命周期证据位于
`artifacts/background-history-gc-model/final-model-lifecycle/01-delete/client/results.json`：

- M 在 history=1828 恢复 676-token 窗口，半页取消重试在 history=1878 恢复
  726-token 窗口；P 回退恢复 history=1664 的 512-token 窗口。
- 17 个注意力层（含 1 个原生 MTP 层）的目标 KV 和 96 个 recurrent tensor 与独立
  重建/保存状态逐字节一致。
- 在私有 Query 和后台 assembly 开始后断开请求，均收到取消 fence，GPU owned
  blocks 回到零。测试结束后 CPU KV、索引与 scope 也全部归还。

`scripts/qualify_historical_gc.py` 支持
冻结源码、独占本机 GPU、自动启动/关闭自己创建的服务，以及接入已有 owned server。
输入 JSON 含 `input_ids` 和 `append_ids`。例如：

```bash
CUDA_HOME=/usr/local/cuda-13 PYTHONPATH=src python scripts/qualify_historical_gc.py \
  --config examples/serve-qwen3.6-27b-historical-gc.json \
  --input /path/to/frozen-input.json --output artifacts/history-gc-lifecycle \
  --kind lifecycle
```

性能输入应保留在一个稳定 GPU 窗口内，并使用足够高的 Ready 水位，使删除与 decode
实际重叠。`--kind performance --pairs 5 --decode-tokens 512` 对每种模式先暖机，再
进行独立的 off/delete 服务对照，组间反转顺序，保存 SSE 与 native step 时序。
五组中位数的初始目标：吞吐损失 ≤1%、p95 步间隔增长 ≤2%、p99 ≤5%；每组保留
原始结果，不以 pooled token 数掩盖异常。对照 off 保留 native ownership/预约路径。加上 `--baseline legacy` 则以省略
`historical_gc` 的旧 archive 路径为基线，测量完整功能的新增开销。跨硬件、并发数及完整任务质量仍应
分别验收；这些结果不代表所有负载严格零开销。

最终模型/性能摘要：`docs/benchmarks/background-history-gc-serving-20261020.json`。
所有最终 GPU serving 用例使用相同冻结包摘要
`ef577c5265cd80b8ed2044bd505e1c0bd45889f2bddc1998397c055ab3c4bd51`。

160 MiB 压力用例的整轮容量为 137 个 record envelope，采样峰值恰为 137。已有
turn 正常完成，新 turn 先排队，再通过实际回收的空闲历史获得完整预约。GDN/MTP
恢复字节一致，GC 等待计数为零。回归还覆盖尾页从 native payload 目标中扣除、
尾页及待释放索引不作为 LRU 供给，避免较早的不可删尾页阻塞后面的可回收 scope。

最终性能证据：`artifacts/background-history-gc-model/final-model-performance-5/results.json`。
每组独立启动旧路径与 delete 服务，各自暖机，组间交错顺序。400-token 合成 notes
输入在一个稳定 GPU 窗口内生成 512 token，以隔离回收开销。五组输出全部一致，
每组均在初始性能预算内；配对中位数如下：

| 指标 | delete 相对旧路径 | 初始上限 |
| --- | --- | --- |
| 吞吐损失 | -0.17%（提高 0.17%） | ≤1% |
| p95 native decode 步间隔增长 | -0.14% | ≤2% |
| p99 native decode 步间隔增长 | -2.48% | ≤5% |

每次测量均有 16 个策略退休，实时 progress 采样确认 CPU 回收发生于公开 decode
期间。这里的吞吐差异很小，不能据此宣称普遍提速；结果支持已测配置下未出现
有意义的 decode 性能回退。测试使用合成稳定窗口负载，不代表长任务检索质量评测。

历史阶段摘要保存在 `docs/benchmarks/background-history-gc-{components,manager,admission}-*.json`。
XML 报告可能被后续复测更新，各阶段 JSON 保留当时摘要。完整模型与性能阶段应读取
新的 serving 摘要，不把早期组件结果当作端到端资格证明。
