# 原生 vLLM + NVFP4 + MTP 并发测试：2026-10-13

本轮直接使用 vLLM 0.30.0，关闭 KVMem connector、scheduler、middleware 和插件，
比较 Qwen3.8-27B NVFP4 的 MTP k=2／k=3。随后按用户要求，将 MTP 注意力 KV
从 BF16 改为继承全局 FP8。四组主测试共328请求、335872输出token完成并通过
逐token计数；最终全FP8的两组均确认了32K输入下的完整32路共同decode窗口。

## eager、KVMem、NVFP4 和 MTP 的关系

- **eager／CUDA Graph** 是执行方式：eager 逐轮由 CPU 提交各项 GPU 运算；
  CUDA Graph 捕获和重放固定形状的运算，减少提交开销。
- **KVMem** 管理各请求的完整历史归档和被选入 GPU 的 active context。当前适配
  依赖每轮 Python capture 收集 Q/K/V，要求 eager 与同步调度。
- **NVFP4** 是权重和激活量化方案。本 checkpoint 的 MLP／lm_head 为 W4A4，
  注意力投影为 FP8 W8A8；不表示所有模块或 KV 均为四位。
- **MTP** 使用 checkpoint 内置预测模块提出候选 token，目标模型验证接受。
  k=2／k=3 表示候选深度；实际输出受接受率影响，单轮可加上一个 bonus token。
  eager 和 MTP 可以组合，CUDA Graph 和 MTP 也可以组合。

此前 KVMem 使用独立 DFlash2 K7。本轮原生服务使用 MTP、编译、CUDA Graph 与
异步调度，历史保留在原生 GPU KV cache 中。二者同时改变多项配置，差距不能被
解释为单独的 KVMem 开销。

官方参考：[MTP](https://docs.vllm.ai/en/v0.30.0/features/speculative_decoding/mtp/)、
[CUDA Graph](https://docs.vllm.ai/en/v0.30.0/design/cuda_graphs/)。

## 固定模型与配置

单张 RTX PRO 6000 Blackwell Server 96GB；vLLM 0.30.0、Torch 2.13.0+cu130、
FlashInfer 0.6.18.post1。使用本地已验证的
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/tree/482ca0f3832238542f8f5295dde86b5f22711d80)，
revision `482ca0f3832238542f8f5295dde86b5f22711d80`；tokenizer 沿用本地 BF16 目录。
量化由 checkpoint 自动识别为 ModelOpt mixed，实际 GEMM 后端为 FlashInfer。

两种深度均使用：TP1、最大上下文262144、max_num_seqs32、batch token8192、
显存利用率0.85、原生 V2 runner、异步调度、`FULL_AND_PIECEWISE` 图模式。
长 prefill 不设单请求1024-token上限。关闭 prefix caching，使用 vLLM generation
默认配置，temperature0／seed104729／thinking关闭；固定输出性能请求忽略 EOS。
API 仅绑定本地18381端口；这里的客户端标签不构成生产身份认证或租户配额测试。

启动脚本逐项清除 KVMem 环境变量和 PYTHONPATH，设置 `VLLM_PLUGINS=''`，
使用原生 vLLM CLI，不修改已安装的 vLLM 文件。startup日志确认 MTP 架构、目标
及预测模块的图捕获与推测接受计数。冷启动编译、图捕获和调优不计入性能成绩。
8192-row大词表投影的调优候选曾触及显存上限，FlashInfer 捕获 OOM 后回退默认
tactic；须与成功请求中的推理错误区分，原始日志保留。

## 测量方法

使用冻结的逐token SSE harness，流式 `token_ids` 必须与最终 usage、摘要和固定
输出长度一致，不能把 SSE chunk 数当作 token 数。默认每请求输出1024token，
共同稳态窗口裁掉每请求前后至少64token，至少持续2秒且每路有足够事件。

每0.25秒读取原生 Prometheus 指标，运行／等待 gauge 约1秒发布一次；共同窗口
还须确认目标运行数、没有排队和新增 prefill、有效覆盖至少90%。该证据用于佐证
原生运行数量，不提供 KVMem 式逐请求 owner 快照。只有一批测试流量访问服务。

分别报告共同稳态 decode、整轮 HTTP 吞吐及 TTFT。32个客户端可能因预填充错开或
GPU缓存容量而排队；共同窗口缺失或证据不足记为 N/A，不报告假定的32路 decode。
MTP 草稿及接受 token 的原生计数用于确认推测实际运行。

短输入为此前152-token写作提示；32K输入直接复用旧冻结的32768个token ID，
不重新套模板。固定长度内容检查只判断计数、明显EOS／工具标记／重复，不能代替
通用任务质量验收。另用4份独立干净32K文档进行正常停止的中部随机值检索。

## 目标 FP8、MTP BF16 KV 的初轮控制

下列均为每请求1024输出token、各档一次观察。decode为通过共同窗口和原生确认的
合计token/s；HTTP包含预填充、排队和响应结束。此表不是最终全FP8 KV配置。

| 输入 | 客户端 | k2 decode | k3 decode | k2 HTTP | k3 HTTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 152 | 1 | 115.26 | 134.06 | 114.10 | 130.33 |
| 152 | 8 | 833.79 | 885.14 | 802.34 | 846.03 |
| 152 | 32 | 2238.17 | 2232.64 | 2114.96 | 2059.52 |
| 32768 | 1 | 98.08 | 98.43 | 74.16 | 75.16 |
| 32768 | 8 | 547.61 | 557.54 | 193.31 | 193.49 |
| 32768 | 32 | N/A | N/A | 223.70 | 222.89 |

短输入32路的原生运行数确认为32；32K长文k2最高观测运行数26，存在容量排队。
长文k2 C32 TTFT p95为127.20秒；此前KVMem+DFlash2同输入整轮吞吐119.47token/s、
TTFT p95为236.07秒，但该比较同时改变推测方法、执行方式、调度和batch设置。

k2另测32K输入、32客户端、每请求8192输出：32请求完成，262144token计数通过，
HTTP584.00token/s、wall448.88秒、TTFTp95319.29秒，最高运行数26、两次抢占。
仍没有完整32路共同decode窗口。这是另一种输出长度，不能混入1024-output表格。
k3的同类可选长输出诊断在用户要求优先全FP8 KV后停止，未完成，不作为成绩。

k2的短41、长41和8192-output32条输出均通过计数检查，没有检出EOS／工具标记或
短周期长重复；正常停止的4个32K检索全部精确匹配、未返回其他客户端的值，
finish_reason均为stop。这是窄范围内容检查。

## 全局 FP8 KV 继承与初始化修正

全FP8首次尝试显式设置 speculative `kv_cache_dtype='fp8_e4m3'`，启动时失败，
没有发送性能请求。定位到原生 `load_eagle_model` 会复制 `CacheConfig`，其中
`init=False` 的已解析布局没有保留；FP8 MTP选择FlashInfer，在图捕获中读取布局
时触发 `KV cache layout has not been resolved yet`。

改用默认继承：全局 `--kv-cache-dtype fp8_e4m3`，不设置 speculative KV override。
目标和MTP共用全局FP8缓存配置及布局，不需要修改第三方源码。失败日志与修正
证据保留在 `fp8-explicit-startup-failure.json`，成功推理与内容检查见下表。
这里的FP8指目标和MTP的注意力KV；GDN recurrent state沿用原生默认精度，
不把混合模型的全部状态宣称为FP8。

## FP8 KV 实测结果

每请求输出1024token，每档一次观察；decode为所有客户端共同稳态的合计token/s。
原生指标确认全部窗口无排队或新增prefill，覆盖92.68%–99.31%，各批次无抢占。

| 输入 | 客户端 | k2 decode | k3 decode | k2 HTTP | k3 HTTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 152 | 1 | 124.53 | 143.21 | 124.01 | 139.67 |
| 152 | 8 | 904.55 | 940.56 | 867.21 | 899.72 |
| 152 | 32 | 2321.28 | 2255.27 | 2157.12 | 2120.00 |
| 32768 | 1 | 102.97 | 108.49 | 78.07 | 81.22 |
| 32768 | 8 | 631.73 | 629.28 | 208.10 | 205.59 |
| 32768 | 32 | 1163.00 | 1178.63 | 246.31 | 245.55 |

32K C32的共同稳态分别持续16.60／14.57秒，原生确认运行数32、等待数0。
整轮各生成32768token，wall133.04／133.45秒，TTFT p95为102.85／103.08秒。
因此约1170token/s仅描述预填充完成后的decode；包含全部预填充和结束响应的
吞吐约246token/s。FP8减少了MTP KV占用，改善容量；此前BF16预测KV的同类
两组最高运行数为26／23，没有完整32路窗口。

相对初轮BF16预测KV，32K C8 decode分别提高约15.4%／12.9%，C32整轮HTTP
提高约10%。k3在短输入C1／C8更快，短C32则k2更快；32K C8／C32二者接近。
每档只有一次观察，不能据此认定某一深度在所有负载上最优，也未与MTP关闭
进行同配置对照。此结论仅比较k2／k3和预测KV类型。

32K C32的KV容量接近上限：k3日志在输出尾段达到99.6%缓存使用率。
本轮确认的是32K输入加1024输出，不能推及32路更长输出或持续服务负载。

两组全FP8各82／82主性能请求完成，计数与流式token摘要一致。短输入各41条
均未检出EOS／工具语法／特殊控制标记／长重复；32K长输入k3的41条也未检出。
k2长输入有1／41条异常：C32的client_index7在第38个输出token出现首个EOS，
全程9个EOS，包含工具调用语法和短控制标记循环。因固定性能请求忽略EOS，
仍继续生成到1024token；这是实际生成token速率，不能等同有效回答文字速率。
其token摘要为`f45bc87b3f78f34000a63a81024333a9af7c9da23ddb8b381632d3f5e94bccdf`。
返回的工具语法没有被执行；该结果来自沿用的旧长文性能fixture，不与正常停止
的新文档检索混合，也不擅自删除后重算完整32路窗口。

两组另各提交4份独立32K文档，正常停止检索共8／8精确匹配本客户端随机值，
未返回其他客户端随机值，没有工具调用，finish_reason均为stop。它们只提供
窄范围检索检查：四个客户端同时提交不证明全程四路decode，也不构成生产认证、
多轮KVMem active/history管理或通用质量验收。

## 复现及当前服务

本机完整启动参数保存在`k3-fp8-inherit/launch.json`；可用已保存启动脚本复现
新的独立运行目录（已有目录会拒绝覆盖）：

```sh
.venv/bin/python \
  artifacts/native-mtp-20261013/launch_native_fp8_inherit.py \
  --k 3 --run k3-fp8-reproduction
```

关键原生参数如下。MTP不单独指定`kv_cache_dtype`，继承全局已解析FP8缓存配置。

```text
--kv-cache-dtype fp8_e4m3
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":false}'
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
--async-scheduling
--max-num-seqs 32
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.85
--long-prefill-token-threshold 0
--no-enable-prefix-caching
```

测试结束保留k3全FP8原生服务：`http://127.0.0.1:18381/v1`，model为`native-mtp`。
最终健康检查200，running／waiting／KV占用均归零，原生抢占计数为0。
显存约83904MiB由模型、图与预分配缓存保留，GPU利用率0%；并非所有预分配显存
在空闲时释放。KVMem未启用；不要把该原生端口当作多租户认证入口。

## 证据

所有原始数据位于 `artifacts/native-mtp-20261013/`。各服务目录包含 `launch.json`、
输入哈希、`results.json`、原生指标、内容审计和资源记录。脚本源码与版本由
`frozen-manifest.json`、`measurement-plan.json`及各启动清单保存。
`summary.json`核验四组主测试的冻结harness、逐token摘要、推测接受计数和
原生窗口；`content-review.json`汇总内容检查，`final-validation.json`保存健康和
资源回收证据。已完成k2的8192-output容量诊断另存，未混入主表；warmup、
中断的k3长输出诊断和失败的显式FP8初始化均排除。
