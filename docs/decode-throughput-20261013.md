# 多并发 decode 吞吐实测：2026-10-13

**当前多用户 Tensor Core 路径的 decode 总吞吐随并发提升：
1 路 61.01、2 路 86.27、4 路 160.57、8 路 269.50 输出 token/s。
8 路约为单路的 4.42 倍。** 这是共同稳态 decode 时间窗的端到端 token
到达吞吐，排除了 prefill 和入口排队；它仍包含解码期间的调度、归档、采样、
草稿模型和输出传输开销，并非只计 CUDA kernel 时间。

## 条件与结果

沿用现有多用户服务：单张 RTX PRO 6000 Blackwell、Qwen3.8-27B Q8_0、
目标 FP8 KV、DFlash2 BF16、vLLM 0.30.0、eager／同步调度。
`q8_tensorcore=true`，全局最多准入 8 个请求，各并发请求使用不同 API key。
KVMem 保持 K131072/G65536/B128。

各档使用相同技术写作提示，实际输入 **152 token**，要求持续撰写多用户推理服务
设计评审；每请求固定输出 **1024 token**，temperature=0、seed=104729、
thinking 关闭、ignore_eos=true。先进行了不计入正式成绩的 128-token 烟测。
正式按 C1、C2、C4、C8 顺序各执行两轮，共 **30/30 请求成功、30720 输出 token**。
内容核对未发现生成 EOS token，保留了原始 token IDs 及文本首尾供复核。

每个请求裁掉至少前后各 64 token，取所有请求剩余 decode 区间的交集。
表中合并值为两轮窗口内 token 数之和除以窗口秒数之和。

| 同时生成请求数 | 第 1 轮 tok/s | 第 2 轮 tok/s | 合并 decode 总吞吐 | 相对单路 | 窗口内每请求平均 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.35 | 61.68 | **61.01** | 1.00× | 61.01 |
| 2 | 88.84 | 83.85 | **86.27** | 1.41× | 43.14 |
| 4 | 160.66 | 160.49 | **160.57** | 2.63× | 40.14 |
| 8 | 269.49 | 269.50 | **269.50** | 4.42× | 33.69 |

“每请求平均”是共同窗口的总吞吐除以请求数，不是每个用户完全相同的速度保证。
**总吞吐增加的同时，单个请求的生成速度下降。** 当前配置的并发上限为 8；
16／32 个客户端会涉及排队，不能据此称为 16／32 路同时 decode。

此前的约 130 tok/s 使用 290-token 输入、128-token 输出，并把 prefill、排队
和整轮完成计入时间。它与这里的工作负载、计时区间均不同，不能直接相除宣称
本次代码提速。此次交付新增了独立 decode 基准与统计回归；上述成绩来自此前
已在运行的同一个推理 worker。

## 如何避免误计

- 精确累计 SSE `choices.token_ids`，逐请求校验与最终 usage 完全一致、输出达到
  指定长度且 finish reason 为 length；role-only 事件不计数，多 token burst
  按实际 token 数计数。
- 起点在所有请求均已输出至少 64 token 之后；终点在任意请求进入最后 64 token
  之前。跨边界的完整 burst 会额外裁掉一些 token。窗口内零进展停顿保留在分母。
- 原生 accepted-history、prefill 计数和资源快照独立确认本轮请求身份和并发进展。
  正式八个窗口的有效原生覆盖为 **98.26%–99.74%**，无采样错误；有效窗口中
  owner 数分别为 1／2／4／8，无可见 prefill 或待完成 refresh。
  最终复测脚本默认要求至少 90% 原生覆盖；窗口内部检测到 prefill 增量、刷新、
  owner 错配或读取错误时直接失败。跨窗口边界的非原子读取单列为不确定证据，
  从有效覆盖中扣除。正式八批原始样本用该严格条件重新核对后全部通过。
- 原生 archive-commit 速率单独报告，因快照延迟不替代精确 SSE token 计数。
  SSE 只记录整个 burst 的到达时刻，无法区分 burst 内单个 token 的生成时间。
- 全请求 TPOT 按首末 token 事件的时间差除以“总输出减去首 burst token 数”计算；
  它可能包含其他请求初始 prefill 的干扰，所以主要成绩采用共同稳态窗口。

原生 owner 同时存在且持续前进证明请求并发，不能证明每一个 GPU step 的
实际 batch 都恰好等于 C。样本只有两轮，也未给出重复实验置信区间。

## 提升的解释与适用边界

并发能让模型处理更多请求的 token，提升服务总吞吐。当前实现还存在明确的
内核切换门槛：[q8_adapter.py](../src/vllm_kvmem/adapters/q8_adapter.py) 仅在输入行数
至少 16 时使用 Q8 Tensor Core 路径。常规 DFlash decode 下，target 有 8C 行，
草稿共享 Q8 输出头有 7C 行：C1 通常均用原生路径；C2 target 跨过门槛而草稿
输出头仍未跨过；C4／C8 两者均可使用加速路径。

因此这张表反映整个当前服务配置的并发扩展能力，提升同时受 batching、kernel
选择和 DFlash 接受率影响。不同并发档的输出 token 序列实际不同，即使提示、
seed 和长度相同；没有以本次吞吐证明跨并发的输出质量等价，也没有仅凭曲线将
收益归因给某一个 kernel。

这是**短上下文、长输出、同类技术写作请求**的 decode 测量。它不代表 134K 历史、
压缩刷新、混合长短 prefill、真实 agent 任务或长期满负载下的吞吐。
先前的多用户隔离、长会话恢复与质量限制见
[多用户状态报告](status-20261013.md)。

## 复测与证据

从仓库目录运行，确保服务没有其他负载，使用新的输出目录：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python scripts/qualify_decode.py \
  --users-file artifacts/multi-user-20261013/users.json \
  --capture artifacts/multi-user-20261013/multi-server-tensorcore/capture \
  --output artifacts/decode-scaling-20261013/retest
```

本机原始证据位于 `artifacts/decode-scaling-20261013/`：

- `tensorcore/results.json`：每个请求的完整 token 时间线、usage、窗口和原生采样。
- `summary.json`：上表的合并统计；`content-audit.json`：输出内容核对。
- `environment.json`：实际 profile、版本与运行 worker 来源。
- `qualify_decode_measured.py`：本次正式测量的冻结 harness，SHA256 为
  `9a84ee3fa35403bbc562ac92ad5116d14dcb342e82c393627703c2e180404245`。
- `qualify_decode_strict.py`、`native_strict_recheck.json`：最终严格校验版本及
  对正式八批原始数据的独立复核；原始结果未重写。旧样本仅记录读取结束时刻，
  离线复核无法补回读取开始时刻。
- `strict-live/results.json`：最终脚本追加的 4 路、每路 512-token 实机核验通过，
  使用完整读取起止时刻，原生覆盖 **95.83%**，无干扰或边界异常；该短核验不混入
  上面的 1024-token 吞吐表。
- `harness-tests.xml`／`.log`：**20 项统计与模拟流回归测试通过**，覆盖 burst
  计数、usage 缺失、停顿、错配身份、覆盖不足、prefill／刷新干扰及边界读取。

凭据未写入测量结果。入口文档见 [multi-user-testing.md](multi-user-testing.md)。
