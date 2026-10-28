# Support and compatibility

KVMem is an experimental integration. The package exposes checked adapters for
specific vLLM interfaces and model-state layouts; the installation requirement
`vllm>=0.30.0` is not a promise that every version, model or hardware combination
has been tested.

## Runtime and hardware

| Area | Recorded coverage | Boundary |
| --- | --- | --- |
| Host core | Linux, Python 3.12, C++17; CPU-only Python and C++ tests | CUDA and vLLM are not required for the host-core build |
| vLLM 0.30.0 | Actual-model serving, MTP/DFlash experiments and the recorded long-history throughput run | Individual profiles and experiments have separate evidence |
| vLLM 0.32.0 | Actual-model eager MTP and native SystemOne checks | These checks do not establish compiled-graph parity for every older benchmark |
| Other vLLM releases / upstream main | Interface and source checks where supported | Source CI is not GPU model qualification; unsupported structures fail explicitly |
| GPU setup | One NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA 13.0 environment | Multi-GPU, tensor-parallel and other GPU families are not established by these runs |
| Model family | Tested Qwen hybrid GDN 27B checkpoints | A safetensors loader accepting a configuration does not qualify another architecture |
| Weights | Tested ModelOpt mixed NVFP4/FP8 safetensors and historical GGUF Q8_0 | The GGUF path has additional pinned-plugin and kernel requirements |
| Attention KV | Native FP8 E4M3 layouts; immutable raw K/V history and model-state restoration | Other KV codecs, scales or recurrent layouts need separate adapters and validation |
| Requests | Text completion/chat and supported native decision requests | Multimodal inputs and arbitrary model extensions are not qualified by text tests |

See [vLLM compatibility](vllm-compatibility.md) for exact checks, source revisions,
SystemOne restrictions and actual-model results. Neither plugin registration nor
a passing interface check establishes that a particular model fits in memory.

## What KVMem changes

vLLM continues to own native execution, scheduling, attention kernels and
sampling. KVMem retains historical state outside the active GPU window and
selects context to restore when needed. Selected-context execution can change
model outputs relative to full-context attention. Storage/restore byte checks,
throughput measurements and end-task quality are different kinds of evidence.

The direct `serve` path and the separate [elastic entry](elastic-serving.md)
have different APIs and configurations. Do not copy an `elastic-serve` profile
into the direct launcher or assume their measurements describe the same path.

The currently bundled DFlash default, the public MTP starter example and the
performance-demonstration profile also differ. Use explicit configurations and
report the configuration actually measured.

## Context and memory budgets

- The logical context limit, selected-context budget `K` and generation workspace
  `G` are separate limits. Setting a 256K context limit does not reserve space for
  eight complete 256K GPU-resident histories.
- Rolling-window profiles can generate across multiple workspace windows when
  their required planning, session and restoration options are enabled. Other
  profiles retain their per-window output limits; see
  [rolling long output](rolling-long-output.md).
- `gpu_memory_utilization` controls an engine budget. Actual device memory also
  reflects model weights, native caches, runtime allocations and KVMem buffers.
  Measure physical usage and leave runtime headroom.
- CPU history, checkpoint, transfer and index pools consume separate memory.
  One configured cache size is not a process-wide RAM cap.
- Deleting historical GC intentionally removes old history under its policy.
  It is an explicit behavior change, not lossless offloading; see
  [historical GC](background-history-gc.md).

The long-history demonstration measures eight fixed recorded continuations with
equal output quotas. Its generation throughput includes waits during that
continuation batch and excludes prior history preparation. It does not establish
maximum sustainable arrival QPS, identical model answers or completed DeepSWE
task success. Use the workload and timing definition attached to each result.

## Sessions and data lifetime

Sessions are scoped to the worker/model load. A session namespace and an exact
prefix match permit reuse; they do not authenticate a caller. Requests using the
same active session serialize while unrelated sessions remain eligible.

CPU-backed history and checkpoints are not a durable conversation database.
Do not assume recovery across process restarts, worker changes or distributed
replicas. The optional NVMe storage machinery has its own ownership and lifecycle
contracts; enabling a disk backend does not add a distributed session service.

The direct API binds to loopback in the starter profile. Shared authenticated
serving uses a gateway or the separately documented
[multi-user entry](multi-user-serving.md), whose supported options differ from
the direct path. Provide the full conversation when continuing a session, and
observe actual hit/miss and cleanup evidence rather than inferring reuse from a
session ID alone.

## Reporting an issue

Include the repository revision, Python/vLLM/CUDA versions, GPU model, the
explicit profile with local paths and credentials removed, the failing command,
and the relevant error or capture records. State whether the failure occurs
during dry run, model startup, first request, continuation, or shutdown. A small
reproducible input is more useful than an unrelated large benchmark directory.

For a performance report, include actual input/output token counts, concurrency,
memory measurements, timing boundaries and the native comparison configuration.
Report correctness or task-quality checks separately from speed.

## Historical corrections

The 2026-09-29 cold-block-residency correction invalidated earlier compressed
numerical-correctness/performance acceptance claims. That correction and the
original support matrix remain in the
[historical support record](archive/support-before-public-guide.md). Dated
reports describe their own source and environment; they do not override the
current compatibility guide or qualify a different configuration.
