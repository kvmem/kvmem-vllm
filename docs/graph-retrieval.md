# Query 驱动的 block 图检索实验

更新：2026-10-18。功能默认关闭。当前实现是在既有 Mean-K 分数上做一次关系传播，再交给原生选块器；不改变 KV 格式、Mean-K CUDA kernel、GPU active 预算或历史 KV 装配流程。

V2 的配置、兼容规则见 [V2 配置说明](graph-scoring-v2-configuration.md)，实际验收进展见
[实施记录](graph-v2-implementation-20261018.md)。以下旧双权重及绝对 attention mass
传播公式描述 V1；V2 显式选择 `graph_scoring_version=v2`，使用共享 `graph_alpha`
和有效 attention 边的按来源归一化，并可启用累计 attention seed 的 graph pressure。
Mean-K 评分域修复对所有版本生效，旧实验不能充当修复后的冻结基线。

第一阶段验证的问题是：**固定预算下，由当前 Query 加权的历史依赖和相邻内容，能否改善选页。** CPU 合成结果已经验证算法与接线；真实模型的质量、吞吐及全部运行路径仍以本页末尾的独立验收记录为准。

## 模式与启动约束

| `graph_mode` | 行为 |
|---|---|
| `off`，默认 | 不创建图、不额外采集图数据、不传播分数；保持既有路径。 |
| `collect` | 保存当前已启用的 attention 采样和消息元数据；原 Query history、legacy attention 融合和 pressure 选择策略保持原行为。未开启 attention 采集时，没有 attention 出边。 |
| `locality` | 只做原始位置相邻传播；不要求开启 attention 采样。 |
| `attention` | 只做当前 Query 加权的历史 attention 依赖传播。 |
| `multi` | 两类关系同时传播一次。 |

所有启用模式要求 Mean-K 上下文管理、owned sessions 和 `repository_gc=false`。`locality/attention/multi` 还要求：

- `request_planning=true`、`query_history=false`、`action_pressure=false`。
- legacy attention **有效**融合权重为零：启用 `action_attention` 时必须令 `action_fraction=0`；未启用时，默认的 `action_fraction=0.25` 不会误判为非零融合。
- `attention/multi` 要求 `action_attention=true`、`prefill_attention=true`、`action_scope="non-thinking"`。

不支持的组合在启动校验时抛错。`collect` 不强制关闭 Query history 或 pressure，也不把既有融合权重改为零。正常 Query 是唯一图评分入口；无 Query 的 pressure selection 不做图传播。`observe_only` 仍只更新既有 Query 观测，不进行图选择，也不要求该次尚未提交的图水位已经追上 archive。

普通图请求必须携带 owned session；会话容量不足时排队，不能静默退化为匿名无图请求。仅由 session ledger 明确标记的释放控制请求可以不创建图。direct profile 的旧 `multi_user` 中间件会丢弃所需元数据，因此与图启用组合在启动时拒绝；普通 owned-session direct 入口与 elastic gateway 使用保留该协议的路径。

实现支持 `torch_reference` CPU 路径和 `qw3_cuda` Mean-K 路径。当前服务的模型、FP8 KV 布局、MTP、编译和会话约束仍由既有启动器校验；图功能不会扩大这些底层支持范围。

## 节点、候选集合与一跳公式

节点身份为 owned session、历史分支及原始 token 起点 `floor(p/B)*B`，其中 `B=block_tokens`。dense active 位置、GPU page、repository slot 和临时 request ID 均不作为永久节点身份。选页时通过当前已 pin 的 index snapshot 映射到带 generation 的 handle；换槽或 generation 变化不能把物理位置误当成旧逻辑节点。

Mean-K 的 `score_mask` 直接来自 qw3 原生 `repository_score_frame`，使用当前索引的
generation、counts、accepted boundary 和 scoped runtime budget。额外 mandatory
不因必留身份从 softmax 删除；sink/recent 是否屏蔽也遵循原生条件。图的集合 `E`
另由原生 picker 实际固定保留集合构造，允许原始分数为零的竞争节点接收传播。
令原始分数为 `x`，在 `E` 内归一化得到 `r=x/sum_E(x)`，V1 公式为：

```text
s = (1 - wL - wA) r + wL PLᵀ r + wA PAᵀ r
默认实验初值：wL = 0.1，wA = 0.2
P[source, target] = 来源向目标传递的比例
```

两个算子都读取同一份不可变的 `r`。本轮新传入 B2 的分数不会继续从 B2 传向 B1；B2 自己原有的分数仍可传播。未启用算子的权重归还直接项，因此 locality-only 为 `0.9r+0.1PLᵀr`。

权重必须有限、非负、总和不超过 1；输出检查有限性、非负性和质量守恒。仅容忍明确的浮点误差，最后不重新归一化掩盖丢失。传给 native picker 的分数乘回原始 eligible 总量，日志区分 raw 与 normalized。

关闭传播、有效权重全零、算子全部 identity 时直接返回原始分数；空 eligible 或全零 Mean-K 不制造均匀分布。非 eligible 节点不发送也不接收质量。选择之后不追加邻居、不求依赖闭包，仍遵守当前运行时预算、native tie-breaking、mandatory/sink/recent 优先级和历史顺序。

## Locality：两个固定方向槽位

来源 `p` 只考虑 `p-B` 和 `p+B`，每个方向基础比例为 `0.5`：

| 已知边界情况 | 边界系数 `b` | 单方向传播比例 |
|---|---:|---:|
| 两个完整 block 的联合 token 区间被同一个已知 message span 包含 | 1 | 0.5 |
| 联合区间或 block 内部跨越已知消息边界 | `graph_cross_message_weight`，默认 0.25 | 默认 0.125 |
| 边界信息不足 | 1，记录 fallback | 0.5 |

只有一个邻居时仍最多转出 0.5；跨消息衰减不会再被按行归一化抵消。缺失、不可用或被 mask 的邻居，其质量留在来源自身；不跨洞寻找下一个现存 block，不用压缩窗口相邻关系，也不建立同消息全连接。

新增自动消息边界使用独立 `kvmem_locality_spans`，经 `RequestPlan → scheduler/connector
→ SessionBinding → authority → checkpoint` 传递，仅影响图。原 `kvmem_message_spans`
仍保留其 mandatory 契约；两者已知重叠部分必须一致。真实 DeepSWE renderer producer
基于完整编码 offsets 映射，无法精确映射的输入报告 unknown。已有 dataclass 的位置参数调用保持兼容。

续接省略元数据时，只继承通过 token 前缀验证的已知部分。若一条旧消息 span 被验证边界截断，`message_end_known=false` 表示这个截断位置不是已知消息终点，不能据此施加跨消息衰减；新后缀保持未知。重新提供完整 span 可以扩展未知终点。同一已验证前缀上的起点、已知终点或角色发生冲突时明确拒绝，不静默拼接。

## Attention：保留来源，完整聚合后取 top-k

采集复用 vLLM 现有 `page_attention`：对采样 Q 与实际 working K 额外计算 attention，而非获取已有 FlashInfer logits。其成本不能沿用 qw3 hook 的性能测量。

`ActionPacket.source_positions` 保存实际 consumed Q row 的原始历史位置，**不加 1**。每行 mass 已在所选 heads/layers 上平均；图按实际接受的采样行等权累计 prefill 与 non-thinking output，分别保存两类计数：

```text
a[source,target] = 所有实际接受采样行对该目标的 mass 总和 / 该来源样本数
```

这是采样条件下的观测平均。未采样 thinking 部分视为未知；不使用 response decay，不套用 legacy prefill 的 mass/exposure 比率。目标在某次窗口中不可见，不产生显式“观测到零”的事件。

每行先验证形状、逻辑位置、有限非负和总质量；总质量容限沿用 `0.005`，通过后允许对整行做一次数值归一化。partial block 和追加 suffix 可能对应同一个目标起点：先合并重复 segment，再累计，一行只增加一次样本数。

只有来源 block 的真实接受事件处理完毕后才封存：按跨行平均 mass 降序选择最多 `graph_attention_topk` 条边，默认 4；同分按目标原始起点升序。目标必须严格早于来源，且平均质量为正并达到绝对阈值。**不先对每个采样行截断 top-k，不使用 top-p，也不重新归一化保留边。**

检索阶段才把保存的边过滤到当前 `E`。被 mask、missing 的边和未保留部分的质量全部留在来源自身；不从原第 5 名补位、不放大剩余边。没有样本或尚未完成的来源使用 identity row。边权 0.03 即只传出该 attention 算子质量的 3%，不因唯一存活而变成 100%。

## 接受事务与会话生命周期

每会话保存完整来源的稀疏行、至多一个 partial source 的完整目标累加器、样本计数、`accepted_through`、`completed_through`、revision、runtime epoch 和消息元数据。partial 可跨 forward、响应及工具调用；`ActionHistory.begin()` 不清空它。

接受发布顺序是：

```text
确认 scheduler 接受/终止边界
→ 验证 attention、构造 legacy 暂存状态、预留图更新空间
→ 原始 KV 的 append / archive 接受
→ 使用同一次 native parser commit 的 newly accepted mask
→ 发布 legacy 与图更新
→ 发布本请求的 accepted 水位和容量反馈
```

图启用时的 parser 包装只提供只读 preview，原生 parser 仍决定范围与间隔；legacy 和图共享一次 commit。生产路径在当前 claim 中任何 raw KV append 之前，先验证并预留各 owner 的图更新空间。图容量或行校验失败时，raw 历史、parser、legacy attention 和图均未提前接受。准备好的 reservation 随 packet 保留；源复制与语义发布跨线程时显式 detach/resume，发布前重新验证图版本和所有权。未能推进的 owner、异常及取消会撤销其暂存额度。若 raw 已接受后才发生 epoch 失效，拒绝旧语义发布并失败/回收该 owner；不声称已经复制的 raw 字节会自动回滚。

`_publish_claim` 对每次实际接受回调都推进图水位，包括无 ActionPacket、采样间隔跳过、thinking-only 和 locality-only。完成来源依据的是已经处理的接受事件，而非可能领先的 `authority.history_end` 或 archive 封页状态。一个 prefill 回调可以按原始位置依次封存多个来源。

拒绝的 MTP 行、EOS/取消后丢弃的 suffix、private guided Query、public Query replay 和重复历史前缀不增加图样本。prefill 保留“完整 chunk 接受才提交 attention”的规则；部分接受只推进其真实接受水位，对应观测保持未知。去重按原始接受位置与水位进行，跨轮换 request ID 不会重计。

独立的版本化 `GraphSnapshot` 不附加到旧 action tuple：

- 真正编辑历史或回退分支：恢复相应较早快照，连 partial 累加器也恢复，不能仅过滤边端点。
- 完整 M token 前缀验证通过，但模型状态需要退到 P 重放：保留已接受的 M 图和精确 P 图快照，重放不重复累计。
- 保留 M 期间再次捕获原 P：使用保存的 P 图。其他更早边界没有精确快照时拒绝/推迟该 checkpoint，不能把未来 M 图写进较早 checkpoint，也不伪造截断统计。
- M 恢复与精确 P 保留共同预留容量后原子发布。恢复元数据冲突在释放旧 checkpoint 所有权之前检查。

同步和异步 Query 共用 `RequestAuthority.select_for_queries`。图快照与 authority、runtime epoch、branch、revision、accepted boundary 绑定；选择和最终 `commit_selection` 都验证，restore/rewind 增加 epoch，旧任务不能因 revision 恢复到旧值而误发布。native index/assembly 原有 lease 和 GPU fence 继续生效。

图生命周期随 owned authority 与 checkpoint 清理，TTL、释放和失败路径需归还相关引用及额度；不同 namespace 不共享可变状态。显式会话释放是控制请求，不为清理响应建立新的历史图。

## 容量与计费

| 参数 | 默认值 |
|---|---:|
| `graph_locality_weight` | 0.1 |
| `graph_attention_weight` | 0.2 |
| `graph_attention_topk` | 4 |
| `graph_attention_min_mass` | 0.0 |
| `graph_cross_message_weight` | 0.25 |
| `graph_max_bytes_per_session` | 33,554,432，32 MiB |
| `graph_max_total_bytes` | 268,435,456，256 MiB |
| `graph_trace` | false |

实现使用不可变 dataclass/tuple 行与至多一个累加器，并按 Python 容器和其数值对象的实际大小计费。已有行的大小缓存避免每次无样本接受都重新遍历全图。共享的不可变 payload 在不同 live/staging/snapshot 所有者中保守重复计费，因此计费值不是 RSS，也不是序列化大小。

计费覆盖 live、完整 partial、提交暂存、消息边界、P/M、保留的 P 和在途评分 snapshot；提交前使用保守 scratch 上界预留，成功后收敛到最终不可变状态大小。graph snapshot 字节同时计入既有 checkpoint admission。每会话上限不能大于全局上限。

评分创建的逻辑位置映射、传播向量和返回分数也在分配前预留：32 KiB 固定额度、每个 index 节点 1,024 bytes、每个观测来源 128 bytes、每个消息 span 512 bytes；trace 另加 256 KiB。额度保留到评分结果释放，不能通过提前关闭 snapshot 绕过。trace 的第二次解释传播同样计费，且发生在原生 assembly 事务之前；容量不足可安全重试。默认关闭的详细诊断另受下文的固定条数／文件字节限制，不能将这些日志当作图预算内的持久状态。

超限抛出明确 `BufferError`，不会丢旧边、截短 partial 或把不完整统计封为完整。当前没有独立的图 spill/回收机制，也不保证任意历史长度与采样密度都适合默认额度。正常配置需给接收暂存和 P/M 恢复保留余量。

## 配置与两个入口

完整 elastic 示例：[examples/query-block-graph-elastic.json](../examples/query-block-graph-elastic.json)。示例使用本机模型位置、K64K/G32K、CPU history、native MTP 和编译路径；替换模型、tokenizer、用户文件与服务端口后使用。示例参数是实验初值，不是经过质量调优的推荐值。

elastic profile 的图字段位于顶层；direct `serve` profile 的同名字段位于 `kvmem` 内，随后通过 connector extra config 原样送到 worker。例如在已验证的 direct profile 中合并以下 `kvmem` 片段：

```json
{
  "request_planning": true,
  "session_capacity": 8,
  "session_checkpoint_policy": "pm",
  "mean_k_backend": "qw3_cuda",
  "query_history": false,
  "guided_query_tokens": 0,
  "repository_gc": false,
  "action_attention": true,
  "action_scope": "non-thinking",
  "action_layers": "spread3",
  "action_interval": 16,
  "action_fraction": 0.0,
  "action_pressure": false,
  "prefill_attention": true,
  "prefill_attention_interval": 1024,
  "prefill_attention_stride": 32,
  "graph_mode": "multi",
  "graph_locality_weight": 0.1,
  "graph_attention_weight": 0.2,
  "graph_attention_topk": 4,
  "graph_attention_min_mass": 0.0,
  "graph_cross_message_weight": 0.25,
  "graph_max_bytes_per_session": 33554432,
  "graph_max_total_bytes": 268435456,
  "graph_trace": false
}
```

该片段不替代完整 profile 中的模型、K/G、CPU archive 和 checkpoint 字节配置；direct profile 顶层还需既有 `prefill_chunk_tokens`，保证每块最多 64 个采样行。elastic 侧另设 `context_mode="meank"`，owned sessions 由 `max_sessions` 等原有字段配置。

先校验具体启动参数，再删除 `--dry-run` 启动：

```bash
cd /home/chaidi/kvmem-freetoken/kvmem-vllm
export PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src
.venv/bin/python -m vllm_kvmem elastic-serve \
  --config examples/query-block-graph-elastic.json --run-dir /tmp/kvmem-graph-elastic --dry-run
.venv/bin/python -m vllm_kvmem serve \
  --config /path/to/direct-graph-profile.json --run-dir /tmp/kvmem-graph-direct --dry-run
```

direct 与 elastic 两入口共享验证。locality-only 可同时关闭 `action_attention/prefill_attention`；做受控性能对照时，则应各组保持相同 attention 采样开关，仅改变图和融合策略。

## 诊断与固定轨迹回放

`block_graph.jsonl` 在正常 Query 记录模式、实际权重、预算、eligible 数量、revision/水位、覆盖、partial、自环吸收质量、locality fallback、raw/propagated 总量、评分耗时、live/snapshot 字节以及相对 Mean-K 新增/替换 block 数。

开启 `graph_trace` 后，另行解释优先新增的目标：原 Query 分数、自身保留、locality 外部传入、attention 外部传入、贡献最大的来源、最终分数和是否入选。正式评分先执行无 trace 的传播，开启 trace 才额外执行一次解释传播；两次都读取同一初始 `r`，不是两跳。详细诊断本身会增加 host 成本，不能混入生产成本却不说明。

输出上限：每 authority 最多 256 个普通图事件，每条 64 KiB，文件最多 16 MiB；详细 trace 最多 64 个目标、每目标 4 个来源。`block_graph_inputs.jsonl` 仅在 trace 下导出，最多每 authority 8 条、每条 256 KiB、文件 16 MiB、单条最多 2,048 个 index blocks。超大输入不导出；这些限制不截断 live 图，也不改变选页。

`action_attention.jsonl` 补充图的 prefill/output 接受样本数、图准备/封存时间和提交时间。原 attention 采集成本仍看既有 Action 采样指标，不能全部记为新增图成本。

固定记录的六组离线比较：

```bash
PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src \
  .venv/bin/python -B scripts/analyze_block_graph.py \
  --input /path/to/capture/block_graph_inputs.jsonl \
  --output-dir /tmp/kvmem-graph-offline
```

工具对同一 recorded Mean-K、图和候选 mask 分别执行 Mean-K、collect、legacy cumulative attention、locality、attention、multi，通过 native picker 使用相同 K/G、mandatory/keep bands 与 runtime budget。legacy 对照默认融合系数 0.25，可显式指定。长历史的 native metadata 分块建立，不为了离线测试扩大 active 窗口。当前离线导出/回放用于默认 top-4、连续无 GC 历史。

`analysis.json` 和 `summary.csv` 区分选页变化、保留原 Mean-K 质量、观测边共同入选及有标签的证据覆盖；没有人工/任务标签时，证据命中、无关/旧版/重复块和答案正确率保持 null。离线共同入选不等于真实闭环质量，因为不同选页会改变模型输出及后续 attention。

本机模型 fixture 支持严格的证据标注：同时传入 `--fixture`、`--traces`、`--tokenizer`。标注器校验 frozen fixture、实际请求 token 哈希、请求所属 agent 和 tokenizer，定位完整的来源陈述及其所需账本依据；缺失、歧义或身份不符会失败。新增 `annotated-inputs.jsonl`、`annotation-provenance.json` 和 `arm-summary.csv`。只将 token 完全相同的完整 block 作为重复内容，未知的过期／无关标签仍为 null。

```bash
PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src \
  .venv/bin/python -B scripts/analyze_block_graph.py \
  --input artifacts/block-graph-model-20261018-collect-r3/server/capture/block_graph_inputs.jsonl \
  --fixture artifacts/block-graph-model-fixture-20261018.json \
  --traces artifacts/block-graph-model-20261018-collect-r3/qualification/traces.json \
  --tokenizer /home/chaidi/qw3/models/Qwen3.8-27B-BF16 \
  --output-dir /tmp/kvmem-graph-labelled-offline
```

合成回归可用 `--synthetic` 代替 `--input`。已有结果位于 [synthetic-offline/analysis.json](../artifacts/block-graph-20261018/synthetic-offline/analysis.json)：一个 B=4 的独立微型 fixture 中，预算均为 12 tokens，Mean-K/collect 选择 `[8,12,20]`，attention/multi 选择 `[4,8,20]`，依赖为因果方向 `20→4`。这是标签已知的合成选页验证。

需求中的 `B1→B5`、分数 `[.01,.60,.15,.12,.07,.05]` 示例只作为独立算子方向/预算数学测试：按该编号它指向未来，与生产建图严格 `target<source` 的约束不同。生产采集及 recorded-input 回放不会为重现这个例子而允许未来 attention 边。

## CPU、native 与组件性能证据

最终受影响 CPU 回归 **697 passed、1 skipped**（[日志](../artifacts/block-graph-20261018/cpu-final-regression.log)）；其中跳过的是需要 CUDA 的组件测试，真实模型覆盖另列。重新构建的 C++ 回归 **5/5 passed**（[日志](../artifacts/block-graph-20261018/cpp-regression.log)），检查使用显式 CHECK／异常，不依赖 Release 模式会关闭的 assert。覆盖状态/传播、接受事务、P/M partial、编辑回退、异步失效、预算、消息边界、direct/elastic 参数接线以及原有相关功能。具体新增入口：

- `test_block_graph.py`、`test_graph_scoring.py`：聚合后 top-k、单跳质量守恒、masked/missing 自环、消息边界和 identity。
- `test_graph_capture.py`：实际 Q 位置、共享 parser mask、部分 prefill、无样本、MTP/EOS 裁剪和提交前失败。
- `test_graph_lifecycle.py`、`test_graph_message_spans.py`：owned sessions、P/M、partial、容量、epoch 和元数据。
- `test_graph_options.py`、`test_graph_offline.py`、`test_graph_qualification.py`：启动器、六组固定输入回放与真实模型客户端断言。

目标环境解释器不在当前仓库中，必须显式设置 `PYTHONPATH`。可重新运行核心受影响集合：

```bash
cd /home/chaidi/kvmem-freetoken/kvmem-vllm
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src \
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/kvmem-graph-regression \
  tests/test_core.py tests/test_state_retrieval.py tests/test_action_attention.py \
  tests/test_prefill_attention.py tests/test_query_history.py tests/test_request_plan.py \
  tests/test_block_graph.py tests/test_graph_*.py tests/test_prompt_checkpoints.py \
  tests/test_context_continuation.py tests/test_session_worker.py tests/test_async_refresh.py \
  tests/test_graph_archive_transaction.py tests/test_serving_launcher.py

cmake -S . -B /tmp/kvmem-graph-native -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/kvmem-graph-native -j2
ctest --test-dir /tmp/kvmem-graph-native --output-on-failure
```

CPU host 组件实测：[block-graph-host-20261018-r3/results.json](../artifacts/block-graph-host-20261018-r3/results.json)。两个会话，每个 256K 历史、B32、8,192 来源、每来源两个样本、top-4，50 次重复；计时关闭 allocation tracing。此轮在模型测试之间、GPU 无模型任务时运行，包含最终评分内存预留实现。

| 组件 | 中位数 | p95 |
|---|---:|---:|
| 无样本接受进度 | 6.52 µs | 8.76 µs |
| 接受样本聚合与 top-k | 44.81 µs | 55.33 µs |
| graph snapshot 捕获/释放 | 2.52 µs | 3.04 µs |
| M 恢复并保留精确 P | 4.50 µs | 10.36 µs |
| locality 传播，8,192 blocks | 18.11 ms | 18.34 ms |
| attention 传播，8,192 blocks | 26.02 ms | 27.30 ms |
| multi 传播，8,192 blocks | 39.35 ms | 39.99 ms |

聚合存在一次约 12.81 ms 的长尾，使均值约 301.53 µs；不能只用中位数声称无抖动。每会话 live 计费约 4.11 MiB；两个会话各保留 P/M 时合计约 24.67 MiB，测试过程全局计费峰值约 41.98 MiB（包含评分 scratch，最高单会话约 29.65 MiB），释放后为 0。allocation probe 的实际 retained Python 分配约 8.08 MiB，说明保守重复计费与真实共享对象内存不同。此处不包含 GPU attention 计算、KV 搬运、模型 prefill/decode，也没有测到任务吞吐提升。

组件复现：

```bash
PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src \
  .venv/bin/python -B scripts/benchmark_block_graph_host.py \
  --history-tokens 262144 --block-tokens 32 --repeats 50 \
  --output /tmp/kvmem-block-graph-host.json
```

## 真实模型验证与实验边界

最终 r3 的六组异步实例与额外同步 multi 实例全部通过，共 42 次回答、28 次跨轮 target/MTP KV 与 recurrent state 字节核验。两会话分别执行初始问题、工具续接和修改后缀；编译/CUDA Graph 与 native MTP(k=4) 开启，KV 为 FP8，使用本机 27B NVFP4 权重。每个初始输入约 12K token，B32、K4096/G4096、CPU history 4 GiB，输出下限 129／上限 257，实际均生成 257 token。两会话分别问不同的账本代号，检查会话隔离。

实际模型观测到每组 946–1102 个 rejected MTP token；独立 accepted-token parser oracle 与终态计数一致。每会话第一轮输出采样 16 行，续接累计 32 行，编辑后仍为 32 行，没有带入被丢弃第二轮的样本。collect 两个相同首轮输入的逻辑选页与基线精确一致；locality/attention/multi 均实际改变过选择。所有实例结束后 CPU authority、P/M checkpoint 和图计费均为 0，控制器没有遗留 GPU 进程。

完整结果：[model-comparison-r3.json](../artifacts/block-graph-20261018/model-comparison-r3.json)；选页一致性：[collect-baseline-r3.json](../artifacts/block-graph-20261018/collect-baseline-r3.json)。每组仅一次短运行，诊断 trace、逐步计时和恢复字节核验均开启，不能当作生产吞吐基准。

| 方案 | 代号回答正确 | 六请求墙钟秒 | 整轮输出 token/s | 单请求生成期间池化 token/s | 纯 prefill 模型步 token/s |
|---|---:|---:|---:|---:|---:|
| Mean-K | 6/6 | 26.82 | 57.49 | 77.48 | 5222 |
| collect | 6/6 | 26.90 | 57.31 | 60.49 | 5659 |
| 累计 attention | 6/6 | 26.72 | 57.71 | 78.50 | 5134 |
| locality | 6/6 | 27.35 | 56.38 | 59.81 | 5667 |
| attention | 6/6 | 26.80 | 57.54 | 61.77 | 5487 |
| multi | 6/6 | 26.80 | 57.54 | 61.84 | 5340 |

“单请求生成期间”是各请求首个 token chunk 之后的输出总数除以各自首末 chunk 时间之和，包含其他请求的 prefill、恢复和调度干扰，既不是双路总吞吐，也不是纯 decode kernel 速率。baseline/collect 的这一指标差异明显，但整轮输出率相近；首 token 的出场时机、后续干扰和自然生成轨迹均会影响它，本次未作因果消融，不能归结为单一建图成本。纯 prefill 使用实际执行 token 除以对应 GPU 模型步时间，包含重放；它不是整轮有效输入吞吐。同步 multi 为 25.29 秒、60.98 输出 token/s，仅用于确认同步接线，不作为选择更快调度策略的结论。

固定真实 collect 轨迹的六组离线结果：[offline-labelled/analysis.json](../artifacts/block-graph-model-20261018-collect-r3/offline-labelled/analysis.json)。所有组保持 4096 token 预算；相对 Mean-K，collect、累计 attention、locality、attention、multi 分别新增／替换合计 **0、77、11、11、16** 个块。完整关键证据覆盖和来源—依据共同入选均为 100%，没有挤掉已标注的 Mean-K 证据；六组平均都含 77.67 个可验证的完全重复 block 副本。过期／无关标签未知，保持 null。基线已经完成这组任务，因此**没有观察到图评分提高质量，亦未证明提高性能**。

同一份离线结果的六条 Query 记录中，`observed_source_coverage` 为 **98.46%–98.94%**，口径是已观测完整来源数除以当前索引 block 数，并非全部 token 或 attention heads 的采样覆盖率。`mean_retained_attention_mass` 为 **28.67%–29.11%**，是对已观测来源的 top-4 出边质量取平均；其余质量保留在自身，不能将这组采样条件下的均值解释为全量 attention 的保留比例。

后续 CPU 缓冲池和可选 CUDA 缓存策略修改后，使用当前源码与 native 扩展重做
同一批六条记录的六组标注回放，见
[final-runtime-offline-recheck/analysis.json](../artifacts/block-graph-20261018/final-runtime-offline-recheck/analysis.json)。
除实际评分耗时外，选页、评分诊断、预算和全部标签指标与原离线结果完全一致。
这补充证明后续优化未改变这批固定输入的选页行为；它不替代新源码下七组闭环
模型质量实验，也不改变上述“尚未观察到质量提升”的结论。

小模型验收的原 attention 捕获 host 累计约 2.95–3.19 秒，包含提交／同步，不是隔离的 GPU kernel 时间。collect 的新增图准备（含聚合/top-k）累计 0.222 秒、提交约 0.0034 秒；multi 分别为 0.232 秒、0.0034 秒。multi 一跳传播中位数 4.02 ms，含解释与选择的过程为 13.75 ms；日志采样到全局图计费最高约 2.28 MiB。这些计数可能与模型工作重叠，不能直接相加推导端到端损失；完整模型 checkpoint 中位数约 2.8–4.0 ms，也包含模型状态，图独立快照成本见前面的 CPU 表。

r1 曾因退休记录读取过早及释放控制请求的所有权门禁失败，未列为通过。r2 通过后复查补齐评分临时内存计费，再冻结运行 r3。r3 结束后控制器另补 SIGTERM／完整子进程组回收、incomplete 状态拒绝及 profile/fixture 哈希校验，22 项纯 CPU 控制器测试通过（[日志](../artifacts/block-graph-20261018/controller-regression.log)）；未改变模型执行代码。r3 的模型运行时文件哈希相同，且所有已保存的实际配置与完整 fixture 事后比较一致；旧 manifest 没有启动前 profile/fixture 哈希，因此不能声称此次已使用新控制器的更强冻结协议。


独立实例使用 [benchmark_block_graph.py](../scripts/benchmark_block_graph.py) 冻结 runtime/profile，持有既有 GPU 测试锁，保留每次尝试的 manifest、服务日志、客户端结果和清理记录；客户端 [qualify_block_graph.py](../scripts/qualify_block_graph.py) 使用同一 fixture、至少两个并发会话。示例准备/运行方式：

```bash
PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src \
  .venv/bin/python -B scripts/benchmark_block_graph.py prepare \
  --arm multi --output /tmp/kvmem-graph-multi-r1 --fixture /tmp/kvmem-graph-shared-fixture.json
PYTHONPATH=/home/chaidi/kvmem-freetoken/kvmem-vllm/src \
  .venv/bin/python -B scripts/benchmark_block_graph.py run \
  --arm multi --output /tmp/kvmem-graph-multi-r1 --fixture /tmp/kvmem-graph-shared-fixture.json
```

各 arm 使用独立目录，fixture 不变；同步 Query/restore 另用 `--synchronous` 准备独立实例。客户端逐项检查真实 prefill 与 accepted non-thinking output 建图、MTP rejected suffix 排除、编译/CUDA Graph、P/M 与编辑恢复、跨会话隔离、同步/异步 Query、collect 与基线选页一致、图评分确实改变过排序/选择、全部 owned sessions 释放。

六组初次对照统一 `query_history=false`、`action_pressure=false`、`guided_query_tokens=0`；模型、输入、B、K/G、保留规则、Query 获取方式和输出限制相同。为了隔离新增图开销，客户端六组均保持原 attention 采样开启，baseline 仅融合权重为零；不能把“关闭原 attention 采集”的收益归到图状态实现上。

端到端应另报答案正确率、证据覆盖、依赖共同入选、被挤掉的 Mean-K 命中和无关/旧版/重复内容；性能分开报告原 attention 采集、新聚合/top-k、传播、checkpoint、内存及整轮延迟/吞吐。原生模型步内 prefill token/s 与含恢复/排队的请求输出率分开，不能把后者称作纯 decode 吞吐。少量任务是功能与探索性质量检查，不足以证明统计稳定的质量或吞吐提升。

当前未实现两跳、聚类/K 相似图、GNN、自动句段/AST、整消息闭包、图 pressure 策略、进程重启持久化或 repository GC 联动。`graph_runtime.py` 是已有 CUDA execution graph 模块，与这里的 block 关系图不同。

以上结果对应各目录冻结的 r3 实现，不自动覆盖后续 qw3 对齐修改。已验证的运行边界以本节的 direct 实例为准。elastic 参数贯通、禁用 attention 的 locality-only、collect 保留原 Query history／pressure 行为均有 CPU 回归；本次没有为这些组合另跑真实模型矩阵。`locality/attention/multi` 与非零 guided Query 的组合、其他模型精度、更长真实任务、大并发吞吐与完整大海捞针／DeepSWE 质量仍未在此图评分实验中验收。

后续 [qw3 上下文语义对齐验收](qw3-alignment-20261018.md#真实模型验收结果) 已在四组独立 R2 实例验证 `collect + guided Query + Query history` 的自然接受、回退及跨轮恢复。该组合中的图只收集观测，不参与选页，因此不能作为 guided Query 与图传播评分组合已通过的证据。
