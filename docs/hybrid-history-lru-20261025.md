# 50:50 无图召回的后台历史 KV LRU

本次将已有 native historical GC 接入当前 MeanK/FPS4 50:50 召回、CPU FPS
预构建和紧凑索引。它永久删除最冷的历史 KV；没有启用上下文压缩，也不向 SSD
溢出。全局默认仍关闭，实验配置已显式启用 `delete`。

## 策略与边界

- 冷热依据最近一次接受执行对块的实际使用。全历史评分、读 Key 和预构建缓存
  不会把扫描到的全部块变热。采用每会话的有界扫描 LRU，并非全局精确排序。
- 可复用槽位低于低水位时开始回收，达到高水位后停止；共享 CPU 准入压力也可
  请求回收空闲会话的冷历史。回收不是无限删除全部旧内容。
- 保留 sink/recent、P/M 检查点所需块；Query 评分至选择提交期间保护读入集合。
  正在执行的 GPU 窗口继续遵守原 lease/fence，后台不释放正在使用的 GPU 页。
- 历史位置保留原始编号，空洞不拼接。每块仍为 32 tokens，KV 预算仍为 36864，
  FPS4 的左右邻块在原预算内成组选择，MeanK 保留一半额度。
- 被永久回收的历史无法再次召回。因此此策略可能改变长期任务的质量；运行安全
  和未删除块的数值一致，不等于答案质量与完整历史相同。

## 本次补全的逻辑

1. 退休完成回执携带原始位置、slot、generation。FPS4 缓存只删除匹配代次的行，
   复用空槽；旧后台构建结果不能把已经删除的行重新发布。活跃 Query 结束后才
   应用延迟回执，避免无关旧块的回收使正在评分的缓存失效。
2. native 索引将 FP32 求和与 FP16 MeanK 的所有权分开；封存成功后可以释放 sum，
   同时保留 mean。FPS4 继续缓存原始 FP8 Key，恢复数值沿用原有规则。
3. 关闭图传播时仅保存消息边界、来源及恢复绑定，不再累积图边。边界随存活历史
   裁剪；未来尚未接受的公开输入保留其边界。检查点恢复、P/M 回退和编辑分支均
   保持原始逻辑位置及验证规则。
4. 原 50:50 picker 与邻块规则复用，种子评分不重新归一化，transport mask 不改变
   种子评分。记录写入独立的 `query_recall.jsonl`，明确标记没有传播。
5. 清理已回收块的 attention/action 元数据。回收后的 payload、索引、缓存槽位及
   metadata 可以重新使用，不随累计写入量无限增长。

## 配置

完整准备配置：`artifacts/hybrid-lru-20261025-r1/profile.json`，相对前一轮 300K
配置保留模型、采样相关设置、K/G、8 并发、CPU/FPS 缓存额度及恢复优化。
必要差异是 `graph_mode=off`、native 历史回收、关闭与 native 分配冲突的
`cpu_seal_serial_allocate`。CPU 紧凑索引开关继续保持开启。

```json
{
  "graph_mode": "off",
  "graph_alpha": 0.0,
  "cpu_release_sealed_sums": true,
  "query_fps_cache_fp8": true,
  "cpu_seal_serial_allocate": false,
  "historical_gc": {
    "mode": "delete",
    "policy": "lru",
    "policy_params": {"keep_mandatory": true},
    "poll_ms": 10,
    "scan_blocks": 256,
    "delete_blocks": 64,
    "ready_low": 8192,
    "ready_high": 9216
  }
}
```

水位是 **16384 个 repository 槽位中的空闲数量**，不是保留块数。上述设置在约
8192 个块占用时开始回收，目标约保留 7168 个块，即约 262144 → 229376 tokens
的物理历史；保护块、尾页、在途读和共享内存压力会影响实际数量及释放时刻。
它有意在 300K 逻辑上限之前触发。

`historical_gc` 省略时保持旧路径；`mode=off` 保留 native 所有权但不删；
`shadow` 仅统计策略意图；`delete` 才实际回收。当前 hybrid 组合要求
`keep_mandatory=true`、CPU archive、无图传播、无 guided/query-history。

## 验证范围

验证记录和源码摘要位于 `artifacts/hybrid-lru-20261025-r1/validation.json`。
八 owner 共享 CPU archive、16 线程 FPS 的有界容量测试，累计写入达到每路
repository 容量的 6.75 倍，反复复用槽位；未删 K/V、MeanK、FPS 代表和原生 picker
结果一致，关闭后各项拥有者计费归零。另覆盖回收与 Query/预构建交错、旧代次、
半页取消、P/M 恢复、消息空洞及 CUDA 评分。

本次不重启失败的 DeepSWE 答案，不把组件测试当作完整 Agent 的质量或吞吐测量。
前一版 MeanK/eager 的模型验收记录见 `docs/background-history-gc.md`，其性能数字
不适用于此次 hybrid 组合。

## 与逻辑上下文上限的区别

LRU 约束物理历史 KV 占用；当前配置的 `max_model_len=307200` 没有改变。
桥接、服务请求长度校验、完整 token 身份和会话检查点仍有各自的容量约束。
因此它不会自动消除 `Prompt is too long`，也不等于已经支持无限长 Agent。
长任务还需要单独确定逻辑上限及其输入/输出预约，不能只调大上限后假定所有
CPU 数据都已成为常量。释放回 allocator 也不保证进程 RSS 立即下降。
