> Historical README preserved on 2026-10-28. For current setup and support, start at the [project README](../../README.md). Dated results and pending-work statements below describe their original experiments.

# vllm-kvmem

**Default agent deployment, 2026-10-20:** `vllm-kvmem serve --run-dir runs/agent-01`
now selects the bundled NVFP4 + FP8 target-KV, DFlash2 k=7 profile with GPU memory
utilization **0.80**, FULL CUDA Graphs, eight active requests, K36,864/G16,384,
CPU-only 100 GB history, V2 multi/alpha .2 graph pressure and all four attention
overhead optimizations. The [editable profile](../../examples/serve-default.json)
uses this host's external model paths. Inspect it with `--dry-run`; use
`--config local-profile.json` to select another deployment.
The [real-tool agent screen](../../docs/dflash-agents-20261020.md) observed
202.64 vs 173.05 generated token/s with 5.33 GiB more peak device memory;
single-window trajectory/mode differences and full-task quality limits remain.

**Offline agent policy, 2026-10-16:** the [scheduling policy and acceptance evidence](../../docs/offline-agent-scheduling-policy.md)
define decode priority, pressure-managed active residency, measured ready-work
ordering and separate CPU/GPU admission. Three paired synthetic runs passed with
a median 19.1% multi-turn throughput gain; output-event p95 increased. Performance
acceptance now uses [fixed 300-second windows](../../docs/fixed-window-agent-benchmark.md)
with continuous ingress. The old first16 run stopped after 15 A0 tasks at the
user's measurement-scope change; it is not a completed A0/P1 task speedup claim.
The [P1 candidate profile](../../examples/qwen-nvfp4-offline-agent-k128-g128.json)
uses K128K/G128K, four native requests, CPU-only history and pressure-only residency;
replace its local model paths before launching.

The experimental [public prefill/decode phase policy](../../docs/phased-prefill-decode-policy.md)
is off by default. It preserves the original RPC and attention-scoring paths;
its real-model qualification and fixed-window comparison are separate from the CPU gather experiment.

The experimental [cost-aware prefill coalescer](../../docs/cost-aware-prefill-coalescing.md)
keeps continuous decode scheduling and compares bounded 2K/4K ordinary-prefill
chunks. It is off by default; correctness and whole-window performance are separate gates.

**Offline batch inference, 2026-10-15:** a JSONL workload can now keep a bounded
request pool continuously replenished, without waiting for whole batches or
enabling the latency controller. See the [offline entry point and measurements](../../docs/coordination-v2-20261015.md#当前优先项离线批量推理)
and [native profile](../../examples/qwen-nvfp4-offline-native.json). Queued requests,
actual model batch occupancy, and sampled GPU utilization are reported separately;
this is not a promise of 100% utilization or an SLO qualification.

**Implementation status, 2026-10-14:** the [implemented / partial / missing checklist](../../docs/implementation-status.md)
records the current KV management, qw3 parity gaps, qualification results and next
steps. Per-turn output stays within G. The explicit token-ID Query profile now
supports request planning, complete Query/partial-tail replay and the original
FP16 MeanK CUDA scorer. Stable continuations preserve selected pages, admission uses
actual active headroom, mandatory spans use the original capacity-fit helper, and
P/M checkpoints retain both the Query origin and accepted terminal state. See the
[current context-policy qualification](../../docs/context-policy-20261014.md)
and [profile](../../examples/qwen-nvfp4-meank-explicit-query-elastic.json); remaining
qw3 protocol and GPU-index gaps are listed separately.

**Current P/M qualification:** final-wheel C8 completed five turns per user,
40 turns and 32 checked restores, including oversized mandatory spans and a 20K
tool continuation that exhausts active headroom. Each history reached 73,447 tokens.
At C8 / 32K input + 4K output, two normal-service runs measured **561.08 / 561.39
steady tok/s** and **337.23 / 334.17 whole-batch tok/s** with KVMem and sessions enabled.
C8 input throughput with one output token measured **6,426 / 7,573 tok/s**.
See [measurement conditions, diagnostic comparison and remaining limits](../../docs/context-policy-20261014.md).

**Earlier explicit Query baseline, 2026-10-14:** real-model C8 cross-turn recovery
and short-turn replay pass with complete Query spans and CPU GDN byte checks.
At 32K input + 4K output each, session-enabled steady decode measured **503.22
tok/s**, including periodic CPU checkpoints, versus 378.21 before pinned
snapshots. Without session checkpoints, KVMem measured 576.48 tok/s. These are
single-run observations with different generated token sequences; see the
[evidence and limits](../../docs/explicit-query-20261014.md).

**Native compilation and batched history, 2026-10-14:** KVMem capture, MeanK
probe/replay and native MTP now support native compilation with CUDA Graphs.
Vectorized CPU copy/MeanK maintenance reduced the isolated component by 59.7%.
Eight users completed 32K input plus 16K output each at 538.75 steady / 449.94
whole-batch tok/s, versus 510.48 / 415.01 previously. The same native control
still runs faster; see the [qualification and remaining-gap report](../../docs/compiled-capture-history-20261014.md)
and [compiled profile](../../examples/qwen-nvfp4-meank-mtp-compiled-elastic.json).

**FP8 immutable raw K, 2026-10-16:** serving now archives pre-RoPE K in unit-scale
FP8 E4M3FN, including the GPU raw-K cache; MeanK indices remain FP16. Raw K plus
V history shrinks from three to two bytes per element (one third less). Restore
uses FP32 re-RoPE before native FP8 encoding. See the
[format and validation report](../../docs/fp8-raw-k-20261016.md).

**Raw-K / resident V caching, 2026-10-14:** opt-in bounded GPU raw-key caching,
shared V staging and adaptive CPU history maintenance now reduce C8/64K payload
H2D by about 58% and assembly path time by 15–21%. Steady decode did not show a
clear gain in this round; complete C8/32K+16K measured 513.79 versus 510.40 tok/s.
See the [measurement and correctness report](../../docs/raw-k-v-reuse-20261014.md)
and [cached profile](../../examples/qwen-nvfp4-meank-mtp-cached-elastic.json).

**MeanK with native Qwen MTP, 2026-10-14:** `draft_method: "mtp"` loads the
checkpoint's MTP head. Target MeanK scores select both target and MTP immutable
KV; both are re-RoPEd into the same bounded working window. At C8 with 64K input
and 4K output, final steady decode measured 486.27 / 492.02 tok/s, about 22% above
the preceding DFlash MeanK result. Eight users also completed full 32K input
plus 16K output at 513.65 steady tok/s. Bounded background sealing and cached
GPU-page credits reduce KVMem CPU overhead. See the [qw3 optimization and MTP
report](../../docs/decode-optimization-20261014.md) and
[MTP profile](../../examples/qwen-nvfp4-meank-mtp-elastic.json). The older full-context
~592 tok/s result remains higher; fixed-length tests do not qualify agent quality.

**Original MeanK integration, 2026-10-13:** `context_mode: "meank"` now uses
model-attention MeanK scoring, the unchanged original KVMem selector, immutable
CPU raw KV and dense re-RoPE assembly with vLLM batched CUDA Graph decode.
Accepted terminal history, partial tails, GDN and DFlash2 state survive GPU-page
release and restore. A single-session 256K history restoration passed real GPU
byte checks; eight independent users completed 32K input plus 16K output each.
With 32K input and 4K output, C1/C4/C8 steady decode measured
63.17 / 232.17 / 378.42 aggregate tok/s. The older ~592 tok/s result has not
been reproduced by the MeanK path.
This path currently uses DFlash2; the older full-context MTP results below apply
to `context_mode: "full"`. See [implementation and limits](../../docs/meank-integration-20261013.md)
and the [MeanK throughput and restoration evidence](../../docs/meank-throughput-20261013.md).
The [C8 profiling report](../../docs/meank-profiling-20261013.md) attributes about 85%
of sampled decode GPU activity to target verification and 11% to the DFlash graph;
KV packing/D2H is small in this workload. Instrumented timing is not a new benchmark.
The [KVMem overhead report](../../docs/kvmem-overhead-20261013.md) compares the same
DFlash configuration against native vLLM and measures CPU history/staging changes
adapted from qw3. At C8 with 64K input and 4K output, cumulative assembly fell
from 55.21 to 21.05 seconds, TTFT p95 from 241.45 to 187.36 seconds, and whole-batch
output throughput rose from 99.68 to 120.71 tok/s. Decode results vary with draft
acceptance; these measurements do not establish a consistent decode speedup.

**Elastic MTP preview, 2026-10-13:** `vllm-kvmem elastic-serve` adds authenticated,
versioned full-history turns with native NVFP4/FP8/MTP CUDA Graph execution,
CPU KV offload, byte admission, bounded restores, TTL and explicit selected contexts.
At 32 clients, full-context CPU recovery measured 546.28 generated tok/s over
the whole batch; an explicitly selected context measured 986.01 tok/s. Native
32K/C32 steady decode with offload enabled measured 1171.41 tok/s. See
[setup and API](../../docs/elastic-serving.md) and
[measurements and qualification limits](../../docs/elastic-throughput-20261013.md).

**Native MTP comparison, 2026-10-13:** with KVMem disabled, NVFP4 and FP8
attention KV for both target and MTP reached 1163.00 / 1178.63 aggregate steady
decode tok/s at 32K input and 32 clients for k=2 / k=3. Whole-batch throughput,
including prefill, was 246.31 / 245.55 tok/s. This is a native vLLM baseline,
not KVMem multi-user qualification. See the [configuration, exact-token evidence
and output limits](../../docs/native-vllm-mtp-throughput-20261013.md).

**NVFP4 preview, 2026-10-13:** the launcher also supports the NVIDIA
Qwen3.8-27B ModelOpt NVFP4/FP8 mixed checkpoint with the same BF16 DFlash2 and
multi-user isolation. All 84 baseline requests completed, including 32K input
with 32 clients and a 134K refresh case. Native CUTLASS backend tuning then
completed 102 requests: short C8 steady decode measured 386.54–392.75 tok/s,
and short C32 whole-batch throughput reached 712.80 tok/s (18 active owners).
Four independent users also passed natural-stop 32K document retrieval.
See the [NVFP4 measurements, output
quality limits and backend tuning](../../docs/nvfp4-throughput-20261013.md).

**Long context and 32 clients, 2026-10-13:** 32K-input steady decode measured
38.17–47.22 / 121.00 / 154.74 aggregate output tok/s at 1 / 4 / 8 clients.
All 32 long-input clients completed, with capacity queueing: 19.63 tok/s over
the whole batch, not 32 simultaneous decode streams. See the
[long-context results and measurement limits](../../docs/long-context-throughput-20261013.md).

**Decode scaling, 2026-10-13:** the existing multi-user Tensor Core server measures
61.01 / 86.27 / 160.57 / 269.50 aggregate output tok/s at 1 / 2 / 4 / 8 concurrent
requests in shared steady decode windows. See the [method, evidence and scope](../../docs/decode-throughput-20261013.md).

**Multi-user preview, 2026-10-13 (`0.1.0.dev1`):** the public launcher now supports
per-user API keys, server-owned session namespaces, bounded fair request admission,
and streaming cancellation cleanup. Session overflow retains serialization, and
serving diagnostics retain only the latest 256 completed requests. See
[multi-user setup](../../docs/multi-user-serving.md),
[concurrency qualification](../../docs/multi-user-testing.md), and the
[2026-10-13 actual-model results](../../docs/status-20261013.md). Both native Q8 and the
opt-in Tensor Core path completed the 32-user short-request sweep and overload,
fairness, cancellation and recovery checks. At 32 users the measured whole-window
output rates were 49.65 and 130.15 tok/s; only 8 requests are admitted at once.
Tensor Core remains disabled by default, and production task-quality and sustained
long-context acceptance remain incomplete. The model/runtime support boundaries
below still apply.

**Current status / 当前进度：** [2026-10-13 multi-user results and remaining
work](../../docs/status-20261013.md), with [2026-09-29 implementation and historical
results](../../docs/status-20260929.md) retained for context. The latest status page
supersedes corresponding pending-test statements below. This is an experimental
development snapshot, not a production release. The historical session agent run
was interrupted after its backend terminated; it is not a completed
quality/performance result.

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

Work in progress: bounded long-context KVMem for native vLLM batched execution.
This repository currently contains the extracted host core, its Python bindings,
request transactions, async NVMe storage, GPU transfer/FP8 codecs, recurrent
checkpoints, MeanK scoring and fused raw-Q/K capture. Versioned vLLM hooks qualify
the draft loader, scheduler/worker identity refresh, and continuous target capture
with native DFlash rejection counts. Raw K and FP8 V use bounded pinned transfers,
mutable partial tails and immutable complete archive records. Actual native GDN
identity restore/replay is qualified. A long compressed replay completed byte
checks, but its recurrent metadata classification required a subsequent correction;
corrected bounded target serving has now passed one real 197K-history C1 lifecycle
check and subsequent short C4. Native long C4 also passes lifecycle checks, with
severe prefill/retirement interference. DFlash2 compressed prefill plus decode
now passes a corrected long C4 lifecycle check. Exact terminal archive clipping
and async retirement pass short C1/C4 with full archive checksums. An opt-in Q8
integer Tensor Core adapter improves a short fixed-output C4 test from 88.20 to
173.11 aggregate arrival tok/s (1.96x); output streams differ, so this is not a
task-quality acceptance. On four frozen long histories, observed whole-window
arrival decode rises from 10.44 to 18.62 tok/s and service time falls from
1862.80 to 761.16 seconds. These runs retain Q8_0/FP8 and K131072/G65536;
mixed prefill/refresh blocking remains severe and changed outputs still need
executed task-quality checks. Short and frozen 134K prompt-session restore now match exact greedy token IDs.
A four-session HTTP lifecycle gate passes isolation, same-session serialization,
cancellation/recovery and idle reclamation. Concurrent output streams differ from
serial controls, so task quality remains under test. Controlled long-context repeats,
decode-time refresh and native capacity pressure remain pending.
**Production acceptance is incomplete. Installing the plugin is passive; enabling
experimental KVMem requires an explicit serving profile or qualification launcher.** See [development milestones](../../docs/development.md) and
[qualification boundaries](../../docs/support.md).

## Host core development

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest tests/test_core.py
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core -j2
ctest --test-dir build/core --output-on-failure
```

Linux and a C++17 compiler are required. The host core does not require torch or
CUDA. Source provenance and the original tests are preserved under `third_party`.

## Engine qualification

KVMem resolves the current vLLM release (`vllm>=0.30.0`) and checks the native
interfaces it uses instead of requiring an exact vLLM version. See
[upstream compatibility and native SystemOne](../../docs/vllm-compatibility.md) for
installation, checks and the distinction between CPU and GPU qualification.

Historical GPU results used vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0 and GGUF plugin revision
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. Q8_0 is supplied by that independent
vLLM quantization plugin. Model weights are external; never included in this repo.

For a GGUF target with a safetensors DFlash draft, older V2 runners need the
explicit draft load configuration fix. Opt in using:

```sh
python -m vllm_kvmem --draft-load-compat vllm serve ...
```

This checks the loader's draft configuration selection and applies the fix only
when needed. New upstream loaders already handle it and are left intact.
Installed vLLM files remain unchanged; the plugin propagates the adapter to workers.
It is not an unmodified upstream reference. See `src/vllm_kvmem/compat.py` for
upstream provenance. No performance or KVMem functionality is implied by this fix.

GPU setup for the current release on this CUDA 13 / SM120 machine:
`bash scripts/setup_gpu.sh --latest`. To reproduce the historical GGUF environment,
use `bash scripts/setup_gpu.sh --locked`.
Explicit GPU component tests: `KVMEM_RUN_GPU_TESTS=1 .venv/bin/python -m pytest tests/test_codec_gpu.py tests/test_capture_gpu.py`.
Run only when the GPU is available; this command is not a model performance test.
See [qualification results and limitations](../../docs/qualification-20260928.md).
The opt-in [exact prompt-session path](../../docs/prompt-sessions.md) has passed short/134K
actual-model restore and C4 lifecycle gates. Official session-enabled agent tests
are running; no final quality or long-task throughput acceptance is claimed.

## Experimental profile launcher

[Installation and API usage](../../docs/installation.md) describe an entry that does not
require prior benchmark execution records. The bundled default selects the
user-approved DFlash2 deployment; inspect it with:

```sh
.venv/bin/python -m vllm_kvmem serve \
  --run-dir runs/session-01 --dry-run
```

Copy `examples/serve-default.json` and pass `--config local-profile.json` to
change model paths or settings. Explicit older profiles retain their settings.
Omit `--dry-run` to start native vLLM. The public profile launcher and first-request
automatic arming passed an actual-model GPU startup check on 2026-10-13. The
multi-user entry also passed short-request load, isolation, overload and recovery
checks; the final component/HTTP suite passed 386 tests. See the
[current results and limits](../../docs/status-20261013.md), including exact 134K session
continuation. Production task-quality and sustained long-context acceptance
remain incomplete; no remote package release has been published.
