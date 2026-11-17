# 8 用户：32K 输入与 16K 输出吞吐（2026-10-13）

在当前 RTX PRO 6000 Blackwell Server 96GB 上，8 个独立认证用户同时发起完整
上下文请求，每路输入恰好 32768 token、输出恰好 16384 token，最终历史为
49152 token。全部请求成功，无拒绝、无抢占。

| 指标 | 实测 |
| --- | ---: |
| 8 路共同稳态 decode 合计 | 591.92 output token/s |
| 共同窗口每用户平均 | 73.99 output token/s |
| 共同窗口各用户范围 | 69.50–89.91 output token/s |
| 整轮 HTTP 合计输出吞吐 | 505.28 output token/s |
| 全部输出 token | 131072 |
| 整轮完成时间 | 259.40 秒 |
| 内部首 token 延迟 p50／p95 | 16.29／26.05 秒 |
| 请求完成延迟 p95 | 258.89 秒 |

## 条件与计量

沿用当前已安装的 Qwen3.8-27B NVFP4 模型、目标与 MTP FP8 e4m3 Attention KV、
原生 GDN 状态、MTP k=3、async scheduling 和 FULL_AND_PIECEWISE CUDA Graph。
启用 KVMem elastic connector 的 CPU offload 和真实认证／历史／active admission
网关。CPU KV 池48GiB、active40GiB，测试网关最多8路。没有选取或裁剪上下文。

现有服务网关的输出上限是8192。本次在同一模型后端上临时启动18384端口的测试
网关，将测试生成上限设为16384；测试后临时网关退出，原18381／18382／18383
服务健康。该结果不表示原网关的持久配置已支持16K输出。

每用户使用独立认证身份与随机 session，cache salt由网关生成。原生计数确认
本轮输入总数262144、prefix命中0、CPU加载0，因此整轮包含真实冷预填充。
模型权重与编译图已驻留，不包括模型启动时间。生成参数temperature0／seed104729，
ignore_eos=true固定输出长度；这是合成性能测试，不是任务质量或持续混合负载SLO验收。

在测试进程中透明记录网关内部原生SSE的token ID事件，并继续调用原有SSE解析器。
每路流式token数、最终usage、JSON响应和累计历史长度一致。公共入口返回完成时的
JSON，因此这里的首token延迟是网关内部首个已接受token时间；调用方收到结果要等
整次生成结束。

decode取所有8路共同生成窗口，裁掉每路起止至少128 token，保留窗口内所有停顿。
窗口长179.16秒，共106048输出token。原生running=8、waiting=0、无新增prefill的
指标覆盖窗口99.85%，零抢占。指标为约1秒发布的引擎聚合数据，不是逐请求调度快照。
各路速度不同，首路约209秒完成，末路约259秒完成；整轮吞吐包含降至少于8路的收尾。

## 随输出增长的共同窗口

| 每路已生成token范围 | 合计decode token/s | 共同窗口秒 |
| --- | ---: | ---: |
| 0–4096 | 611.19 | 48.07 |
| 4096–8192 | 594.99 | 38.93 |
| 8192–12288 | 574.86 | 24.60 |
| 12288–16384 | 566.59 | 6.29 |

上述是各路均处在指定输出范围内的交集；各用户的上下文长度并不完全相同。
末段共同窗口只有6.29秒，当时实际context约45K至48K，不是所有8路同时处于48K
的单独定长测试。速度也受生成内容和MTP接受率影响，不能把变化仅归因于context长度。

网关峰值同时active8路，预留20106444800字节（约18.73GiB的页payload账目）；
原生KV使用率峰值30.12%。CPU stage-out实际复制14624489472字节。完成后网关
active／restore字节、active／queued请求及原生running／waiting均归零。

## 复现与证据

结果、时间序列和测试网关配置位于
`artifacts/elastic-long-output-20261013/c8-32k-16k/`，核心文件为`results.json`与
`summary.json`。采样脚本为`scripts/qualify_elastic_long_output.py`。
输入沿用此前长文测试的冻结token IDs；本次只有一个批次，不提供多轮误差范围。

```bash
.venv/bin/python \
  scripts/qualify_elastic_long_output.py \
  --launch artifacts/elastic-serving-20261013/final-service/launch.json \
  --input artifacts/elastic-serving-20261013/hot-32k-c8-c32/input.json \
  --output artifacts/elastic-long-output-20261013/c8-32k-16k-repeat \
  --concurrency 8 --output-tokens 16384 --port 18384
```
