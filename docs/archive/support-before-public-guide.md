> Historical guide archived on 2026-10-28. The observations and corrections below
> describe their recorded experiments, not the current installation or support matrix.
> See the [current guide](../support.md).

# Qualification boundaries

The new [original MeanK integration](../meank-integration-20261013.md) adds NVFP4,
native FULL decode graphs and terminal accepted-history checkpoints. It has
separate component, real restoration and throughput evidence; the older eager
measurements below remain historical. MeanK currently uses DFlash2, whereas the
native MTP offload entry keeps complete KV. Cold-prefill token equality, executed
agent-task quality and sustained mixed-tool SLO acceptance remain incomplete.

The native NVFP4/FP8/MTP elastic-turn entry has separate
[2026-10-13 CPU-restore and selected-context evidence](../elastic-throughput-20261013.md).
It uses upstream graph execution and KV offloading; it does not qualify the
older eager/MeanK compressed-runtime boundaries documented below.

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. A corrected long-C4 lifecycle run now passes; matched
performance and executed task-quality acceptance remain pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

This is a development plugin, not a production support declaration. Installation
and discovery alone leave the serving engine unchanged. Engine hooks require
explicit flags and pinned source hashes; no installed vLLM files are rewritten.

| Area | Current evidence | Still required |
| --- | --- | --- |
| Engine | vLLM 0.30.0 V2 runner; native scheduling, batching, attention and sampling | Additional engine versions and graph modes |
| Target weights | Qwen3.8-27B GGUF Q8_0 through pinned vllm-gguf-plugin | Other quantizations and model families |
| Target KV | Native FP8 e4m3 layouts; raw pre-RoPE K authority and FP8 V; byte-checked assembly | Other native KV codecs and scale layouts |
| Hybrid state | Qwen3.5-family GDN native state identity restore/replay; active-prefill HTTP cancellation releases all resources | Compressed-refresh cancellation and persistent-session state continuation |
| DFlash2 | Corrected long compressed C4 lifecycle; short native exact terminal archive clipping and checksums pass | Matched performance, decode-time refresh/cancellation, recurrent terminal rollback in serving and persistent sessions |
| Bounded target | Scheduled semantic prefill/selection, actual-length acknowledgment, page trimming implemented | Corrected real long C1/C4 pass; pressure tests and performance acceptance remain |
| GPU admission | Future growth reservations tested with native full-attention/Mamba allocator | Attached to bounded scheduler; native GPU pressure tests remain |
| Storage | Immutable archive, generation-qualified leases, packed/direct transfers; async retirement passes native long C4; CPU tests cover serving GC, strict NVMe reservation and shared I/O byte admission | GPU GC/quota/backpressure qualification; whole-process host-byte admission and fair I/O scheduling |
| Prompt sessions | Short and frozen 134K exact-token restore; four active sessions, isolated exact continuation, cancellation/recovery, scope isolation and idle reclamation pass actual GPU/HTTP gates | Concurrent numerical quality, growing multi-turn agent histories, expiry and agent performance; see `prompt-sessions.md` |
| Measurements | Exact frozen-token inputs, SSE arrival/usage validation, C1/C4 aggregation | Native SSE C1/short-C4 passes; matched warm repeats and executed agent-task success remain |

BF16 is used for activations and the draft model; it does not change the Q8_0
target weight path. Initial integration requires one GPU/rank, eager execution,
text-only input, prefix caching disabled and Mamba cache mode `none`. Bounded
refresh has corrected target-only and experimental DFlash2 long-C4 lifecycle evidence.
`--bounded-dflash` uses synchronous DFlash2 with native BF16 sliding draft KV;
Matched DFlash performance and decode-time refresh qualification are still pending. Public connector
metadata carries commands/acknowledgments; isolated runner/scheduler adapters
supply the missing historical selection, replay and bounded-allocation semantics.

An optional `--q8-tensorcore` qualification flag experiments with a source-pinned
GGUF batch linear adapter on SM120, for BF16 input batches of at least 16 rows.
It preserves original Q8_0 bytes/FP16 scales and the native fast-math Q8_1
activation quantization, using grouped integer Tensor Core dots and FP32 scaling.
It changes floating-point reduction order. It is disabled by default and has not
yet passed task-quality equivalence or repeated matched performance acceptance;
single short and long native comparisons are recorded in the qualification log. This does not
add BF16 target weights or change target FP8 KV.
