# qw3 上下文语义对齐（2026-10-18）

本次以 `/home/chaidi/qw3` 的实际工作树为只读参考，核对 KVMem 的上下文管理语义，
没有替换 vLLM 的模型执行器、continuous batching 或 CUDA Graph。此前的
[2026-10-14 审计](context-management-parity-20261014.md)是历史基线，不能用其中
的旧缺口判断当前实现。新图评分是默认关闭的独立实验，见
[图评分实现与验证](graph-retrieval.md)。

## 本次修复

| 问题 | 修复后的行为 | 验证重点 |
| --- | --- | --- |
| private guided Query 在换行或达到长度上限时也被接受 | 同步、异步刷新共用接受策略。跳过前导空白，保留正文换行；仅自然终止且有效的文本替换公开 Query。空、结构化、截断输出回退原 Query。 | 私有 Q 发布行数与消耗行数分别记录；回退不改变公开采样器、历史或输出预算。 |
| 完整 M 前缀已验证，但回到 P 重放时清空 prefill attention 历史 | 用已验证的 M 上界恢复评分水位；真实编辑只恢复 P 快照。中间 P 不能携带未来反馈。 | M→P 保留有效样本；公开重放不重复计分；非法验证上界在修改状态前拒绝。 |
| Query 与输出分别合法，合并后却无法容纳历史必留页 | 联合检查完整重放、输出、native lookahead 和固定 sink/recent 预算。冷请求和 force 请求提前拒绝；auto 会话先验证前缀并判断旧 Query 是否被省略。 | 不截断 Query、不缩短输出；失败不改原 M/P；异步判定的容量错误只结束自身请求。 |
| message-range Query 拒绝 assistant/tool | 接受调用方提供的混合角色连续范围，保留角色和绝对 token 坐标，整个 Query 仍原子重放。 | 范围和角色往返不变，超预算拒绝；不会自动推断聊天模板坐标。 |

`guided_query_tokens` 继续表示实际消耗的私有输出 token；新增
`guided_captured_tokens`、`guided_published_tokens`、`guided_ignored_prefix_tokens`、
`guided_natural_termination` 用于区分暂存、接受和回退。评测不能再假设消耗行数
恒等于最终评分行数。

## 保留的边界

- GPU 有界 active view、CPU historical authority、原版基础 MeanK/Store、
  immutable raw K 与重新施加 RoPE、accepted-prefix、P/M 恢复和原生 MTP
  仍是当前主路径。并不要求将 qw3 的单请求执行循环复制进 vLLM。
- 用户此前明确暂缓的调用方 evidence.changed 和增长阈值重选，本次不启用。
  单轮输出仍受 G 限制；可选 rolling active window 不是本次语义验收的配置。
- guided＋Query relevance history 是用户要求保留的扩展组合，qw3 的组合限制
  不适用于这里。图评分 collect 可以观测它；图传播评分模式仍禁止 legacy
  历史分数和 pressure 混合，以保持实验定义。
- private Query 保留现有无 thinking 的检索 prompt 和 greedy 生成。qw3 的
  direct SOURCE REQUEST 重写、可选 private thinking 没有开放；现有 prompt
  的 close-think 前缀是无条件的。忽略的前导分隔 token 计入总暂存上限，
  保证不额外占用 K/G。不能宣称私有生成轨迹与 qw3 逐 token 等价。
- 对外 Query 仍是完整 prompt 的绝对 token 范围。qw3 的消息内 UTF-8 字节
  偏移、默认最后 user Query、聊天模板自动映射不是同一协议，尚未迁移。
- 独立 prefill budget、prefill-only start/append/finish、多模态/MRoPE、
  其他检索器、任意分支/命名 checkpoint、增量 GPU MeanK 和完整 no-copy
  window 仍不在这次对齐范围。多 GPU 与原生异步多在途 batch 未新增资格。

源码指纹和精确 guided 参考范围保存于
`artifacts/qw3-alignment-20261018/guided-policy-provenance.json`。
完整源码审计见同目录的 `context-parity-audit.json`：23 个对应文件中 18 个
逐字节相同，这个计数包含 LICENSE 和测试，不能称为 18 个核心实现文件。
Store、公共配置头、Action parser、refresh helper 存在适配或未迁移更新，
报告逐项列出。基础组件相同不等于两套模型的逐页选择或生成结果完全等价。
CPU 测试和真实模型证据保存在同一实验目录；性能数据使用独立、关闭昂贵
恢复字节校验的 FP8 实验，避免把功能验收时间当作吞吐。

## 真实模型验收结果

四组 R2 均通过。冻结的生产源码一致，均使用 NVFP4 模型、FP8 KV、compiled MTP(k=4)，
2 个独立会话各完成 3 轮；K=4096、G=4096、逻辑上限 32768、CPU 历史容量
4 GiB。开启 Action、prefill attention、Query history 和恢复字节校验。
`graph_mode=collect` 只记录图，不参与选页；pressure 主动重选和 rolling active
window 均关闭。同步组同时关闭异步恢复/刷新及其维护协调，不能用两组耗时
推导某一个异步开关的性能影响。

| R2 组别 | guided 上限 | 完成轮次 | 发布有效 guided Query | 回退公开 Query | 恢复字节校验 |
| --- | ---: | ---: | ---: | ---: | --- |
| natural-async | 256 | 6/6 | 4 | 0 | 4 次 target/MTP＋4 次 recurrent，全部通过 |
| natural-sync | 256 | 6/6 | 3 | 1 | 4 次 target/MTP＋4 次 recurrent，全部通过 |
| fallback-async | 1 | 6/6 | 0 | 4 | 4 次 target/MTP＋4 次 recurrent，全部通过 |
| fallback-sync | 1 | 6/6 | 0 | 4 | 4 次 target/MTP＋4 次 recurrent，全部通过 |

自然终止组的 7 次接受都满足自然终止和有效文本约束；同步组另有一次模型直接
终止、未生成有效内容，正确报告 `empty-output`，使用原来的 56 个公开 Query
token 评分。小上限组的 8 次尝试都消耗 1 个 token 后触顶，报告
`unterminated-output`，发布私有 Q 为零，仍使用完整 56-token 公开 Query。
这验证的是接受/回退策略；没有要求同步和异步产生相同私有生成文本。

各会话依次经过首次 Query、保留选页的工具续接、force Query 的 M→P 回退重放。
P 坐标均为 12128；首次 Query 范围为 `[12150,12206)`，按页对齐后重放
`[12128,12206)`。每轮显式选页均为 4096 token，工具续接没有额外重选。
独立的超预算请求在四组中均收到 HTTP 400：完整 Query 和原定输出预留后
只有 992 token 可用于必留内容，小于 sink 所需的 1312 token，没有缩短 Query
或输出配额。

恢复和反馈证据如下：

- 每组 2 次恢复终端 M=12471、2 次恢复 P=12128。总共 16 次 target/MTP
  immutable FP8 rebuild 对比和 16 次 recurrent 字节对比通过；每次覆盖
  17 个 attention 层（其中 1 个 MTP 层）和 96 个 recurrent tensor。
- 每个会话的 Query history 都按 round `1→2→3` 推进，输入水位为
  `12215→14520→14829`，`observed` 为 `true→false→true`。工具续接的分数
  哈希与上一轮完全相同，没有重复记录或衰减旧 Query。
- 自然终止组的 prefill 样本数为 `352→416→416`，小上限组为
  `357→421→421`；两组在工具续接后都达到水位 14496。随后回到 P=12128
  仍保留已验证 M 的样本和水位，公开重放没有再次计入样本。
- 24 个终端的图 `accepted_through` 都等于历史末尾和 M checkpoint 坐标。
  私有 guided Query 没有扩张公开历史坐标。所有会话、恢复任务、下载、历史
  KV、检查点和图内存最终释放，磁盘 KV 读写均为零；四次控制器清理都记录
  自有进程组和 GPU 进程列表为空。同步组未启用 CPU 准入器，对应统计记为
  未启用，不伪造零采样值。

合计 24/24 个回答包含固定合成问题的预期代码。这是功能夹具中的内容检查，
不是大海捞针、MemoryAgentBench 或 DeepSWE 准确率。字节 oracle、图 collect
和不同长度的私有生成会影响耗时；本组不提供吞吐提升结论。各组实际出现了
双请求纯 decode batch，具体次数仅作为并发覆盖证据。

汇总与每份证据的 SHA256 见
[model-comparison-r2.json](../artifacts/qw3-alignment-20261018/model-comparison-r2.json)。
该汇总重新校验了四组全部冻结文件、profile/fixture 哈希，并使用各组冻结的
checker 对原始 capture 和 traces 离线重放，结果与已保存的检查完全一致。
完整 CPU 回归记录为 **805 passed、1 skipped**；后续 profiler scope 的 8 项
和资格客户端的 25 项是独立复查集合，与前面的测试有重叠，不能直接累加。

R1 保持原始失败记录：`fallback-async-r1` 已完成 6 个模型回答并清理资源，
但旧 checker 把 initial pressure 选页事件也算作显式 Query，触发
`Public Query requires one bounded selection`。修正事件过滤后，保留证据的
[离线复查](../artifacts/qw3-alignment-20261018/fallback-r1-posthoc-audit.json)
通过；没有修改 R1 的失败状态，也没有用离线复查替代新的四组 R2 真模型验收。

## 复现

从仓库根目录运行以下聚焦 CPU 检查；广泛回归的历史计数以
`artifacts/qw3-alignment-20261018/cpu-final-regression.log` 为准。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B \
  -m pytest -q -p no:cacheprovider \
  tests/test_guided_policy.py tests/test_query_history.py \
  tests/test_context_continuation.py tests/test_prefill_attention.py \
  tests/test_query_capacity_admission.py tests/test_request_plan.py \
  tests/test_context_alignment_qualification.py tests/test_profile_scopes.py
```

重新运行模型时创建新目录，保留原 R2；下例复现 natural-async。准备阶段冻结
当前源码、原始 fixture 和对应的完整 R2 profile；运行阶段只使用新冻结目录，
持有共享 GPU 锁，拒绝占用中的 GPU，并清理自己启动的进程。

```bash
ALIGNMENT_REPRO_OUT=/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/qw3-alignment-natural-async-repro
PYTHONPATH=src .venv/bin/python -B \
  scripts/benchmark_block_graph.py prepare \
  --output "$ALIGNMENT_REPRO_OUT" --arm collect --port 18628 \
  --fixture artifacts/qw3-alignment-20261018/fixture.json \
  --profile artifacts/qw3-alignment-20261018-natural-async-r2/profile.json \
  --qualifier scripts/qualify_context_alignment.py

PYTHONPATH="$ALIGNMENT_REPRO_OUT/runtime/src" \
  .venv/bin/python -B \
  "$ALIGNMENT_REPRO_OUT/runtime/scripts/benchmark_block_graph.py" run \
  --output "$ALIGNMENT_REPRO_OUT" --arm collect \
  --fixture "$ALIGNMENT_REPRO_OUT/fixture.json"
```

其他三组分别替换 profile 为 `natural-sync-r2`、`fallback-async-r2`、
`fallback-sync-r2` 的 `profile.json`，使用各自新的输出目录；同步组在准备
命令追加 `--synchronous`。若需要逐字节重现本次源版本，先对照汇总中的
manifest 和冻结 `runtime_sha256`；当前工作树重跑不能自动称为同一源码版本。

后续内存管理优化另有 `pooled-copy-natural-async-r1` 和
`pooled-gc-natural-async-r1` 两组独立上下文资格，均完成两用户各三轮、非法
请求隔离和服务内资源释放。它们不替代上表四组 R2 的接受／回退矩阵；新增的
精确 FP8 冷恢复与吞吐证据见
[后续内存与执行策略验收](host-admission-calibration-20261018.md)。只读参考的
26 个源文件及 23 个 vendored 文件已再次逐字节核对，与这份审计的指纹一致，
记录见 `artifacts/host-memory-investigation-20261018-r1/current-reference-recheck.json`。
