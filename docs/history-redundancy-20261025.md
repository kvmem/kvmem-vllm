# 可配置的后台历史 redundancy 回收

`historical_gc.policy` 支持 `lru` 和 `redundancy`。默认仍是 LRU，整个
`historical_gc` 省略时仍不启用后台历史删除。策略负责提出候选，native manager
统一执行代次校验、保护检查、退休、异步释放与空槽复用。

## 如何判断冗余

1. 从最多四个均匀选取的 target attention 层取得封存块已有的 MeanK，不包含
   draft 层，也不重新保留已释放的 FP32 求和数组。`qw3_cuda` 使用已存的 FP16
   MeanK，`torch_reference` 从已有 FP32 sum/count 取得其原有均值。每个层/KV head 单独归一化，
   块间相似度是这些分组余弦相似度的平均。
2. 重要性使用当前公开问题的 Query：每 32 tokens 一个均值段，保留全部 GQA
   heads；对每个 KV head 取对应 Query heads 的最大点积，除以 `sqrt(d)`，再跨
   层和 KV heads 平均。多个问题段、已接受生成内容的观测窗口取最高值。
   生成内容使用 fast 8-token 窗口和 slow 32-token 窗口（保留四段）。
3. 候选必须找到仍被保留的**直接代表块**，默认相似度至少 0.9。只对满足条件的
   候选排序，分数为 `0.1 × softmax(重要性) - 0.9 × softmax(相似度)`，低分先回收。
   缺少完整特征、问题观测不完整或超出缓存额度时，不将其当作冗余块删除。
4. 成功删除后保留代表关系；A 像 B、B 像 C 不足以推导出 A 像 C，不能据此连续
   删除所有代表。删除前重新检查候选和代表的 slot/generation、特征版本以及问题
   epoch。新问题或回退使旧问题的待执行决策失效。

这是对已有策略思想的本地实现。参考
[上游固定版本](https://github.com/zyf32453/kvmem-llama.cpp/tree/c02612d05e88833a854bb688c65d17e40a58b86a)
时需注意：本地候选是符合保护条件的封存历史块，并未移植其 thinking-only 候选
筛选；执行机制使用本仓库的后台 native manager，并非上游的 decode 间隙提交。

## 配置示例

下面是已有合格 CPU/native GC 配置中的策略替换片段。保留原有水位、KV 预算、
50:50 召回及其他参数即可。`shadow` 只统计意图，`delete` 才永久删除历史 KV。

```json
{
  "historical_gc": {
    "mode": "shadow",
    "policy": "redundancy",
    "policy_params": {
      "keep_mandatory": true,
      "importance_weight": 0.1,
      "representative_similarity": 0.9,
      "reference_blocks": 32,
      "candidate_blocks": 128,
      "pair_comparisons": 512,
      "query_segments_per_step": 8,
      "feature_cache_bytes": 134217728,
      "query_cache_bytes": 8388608,
      "allow_lossy_fallback": false
    }
  }
}
```

以上为全部策略参数及默认值。重要性权重取 `[0,1]`，余弦阈值取 `[-1,1]`。
四项工作量限制均为 `1..65536` 的整数；两个缓存额度以 bytes 为单位。
`keep_mandatory=false` 对本策略无效并会被拒绝。

`allow_lossy_fallback=true` 是显式的有损压力后备：无足够冗余候选时可回收冷块，
不能将这种删除解释为已经证实内容冗余；有紧急 ready-slot 请求时禁止该后备。
默认关闭，因此没有合格冗余候选时可能无法解除内存压力。

## 内存、工作量与安全边界

- 不做全历史两两比较。候选、参考池、pair 比较及 Query 分段计算都有单步上限。
- 特征及 Query 缓存按会话作用域限额；仍被策略快照引用的旧数据也计费。缓存不够
  时保护对应内容，不静默改用 LRU。
- scheduler 与 archive 使用同一预算函数：从配置 CPU 总额度中先扣共享 FPS/窗口
  缓存，再扣所有可能存活 scope 的特征/Query 缓存、策略元数据和共享 Query 暂存。
  暂存保守预留 512 MiB；不会把这些容量重复作为可准入的历史 KV。
- Query 只传输公开问题涉及的 prefill 行和实际接受的 target decode 行。单传输槽
  上限 64 MiB，最多四槽；超限使该问题观测无效，不截断后冒充完整问题。
  MTP 拒绝行、未接受输入、私有重放不会进入观测。
- Query 必须在真实前向中完整采集。工具续接仅在已验证相同问题/前缀时复用；缺少
  已缓存问题的观测时保护历史，不为了 GC 额外重放用户输入。
- 当前明确拒绝 `offline_prefill_coalescing`、`prefill_physical_coalescing`、
  `prefill_publication_pipeline`、`early_capture_slot_release`、`prefill_capture_ahead`
  组合。其接受行发布生命周期需要另行验证。LRU 不增加这些限制。
- 原有 sink/recent、checkpoint、GPU 使用中及在途读等保护仍有效。这里只删除
  满足现有机制条件的历史块；不改变 picker、KV 预算、图传播或模型采样。

删除是永久的，语义相似并不保证信息可互换，仍需任务质量测试。后台 KV 回收也不
修改服务端逻辑上下文上限，不会自动消除 `Prompt is too long`。

## 验证与实验状态

已验证配置、特征/Query 生命周期、代表保护、代次复用及 native 回收机制：
277 项 Python 回归通过、4 项 GPU 测试跳过，9 组 C++ 测试通过；最后补充的
回退先失效与队列门控检查共 83 项通过。这些集合部分重合，不应简单相加。
证据保存在 `artifacts/history-redundancy-20261025-r1`。

按用户最新要求，下一轮 DeepSWE 先测试 LRU，仍为原 16 题、300K、最多 8 路。
旧暂停实验保留，新实验独立冻结；GPU 忙碌时只排队，必须先通过当前冻结代码的
GPU 组件验证，再启动正式任务。尚未实测 redundancy 的完整模型质量或吞吐。
