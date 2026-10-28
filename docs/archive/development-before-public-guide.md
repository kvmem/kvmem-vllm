> Historical guide archived on 2026-10-28. The observations and corrections below
> describe their recorded experiments, not the current installation or support matrix.
> See the [current guide](../development.md).

# Development goal and acceptance

Develop an out-of-tree KVMem plugin in this repository, with explicit versioned
vLLM hooks where public extension points cannot represent historical working-set
reselection. The qw3 executor is a read-only semantic reference, never the serving
backend. The goal remains active until the full acceptance scope below is proved.

Primary qualification is Qwen3.8-27B Q8_0 target weights, FP8 e4m3 target KV,
DFlash2, and per-request K131072/G65536. BF16 activations and draft weights do not
change the Q8_0 target path. vLLM owns native GPU allocation, scheduling, batching,
attention and sampling. Logical history, dense working positions, physical pages
and generation-qualified repository handles must remain distinct.

## Acceptance scope and current evidence

| Requirement | Current evidence | Remaining acceptance |
| --- | --- | --- |
| Original selection/storage semantics | Unchanged host core with file hashes in `third_party/qw3/PROVENANCE.json`; MeanK score bridge; immutable raw K; dense position assembly; original sink/recent bands | Match old controller's final-query override and record granularity before declaring an equivalent-workload speedup |
| Bounded native target execution | `engine_deepswe_effect_bounded_03`: real 197110-token history, 65 matched refresh commits, bounded native attention allocation, coherent natural-EOS continuation and complete cleanup | Cancellation and GPU/CPU/NVMe capacity-pressure tests; long C4 passes but performance is poor |
| DeltaNet and transaction correctness | Native target/DFlash identity replay, C4 recurrent-state restoration, rejected candidate exclusion, per-generation leases and failure fencing | Compressed target/draft refresh, accepted-boundary checkpoints, terminal-prefix rollback and persistent sessions |
| Efficient transfers and storage | Packed D2H; fenced H2D; moved-block FP8 assembly; coalesced native NVMe I/O; opt-in archive-layout assembly and bounded CPU read cache | Independent performance repeats, overlapping preparation, background policy GC and nonblocking retirement |
| Cross-request resources | Native lifetime GPU-page reservation adapter with hybrid-manager tests; shared archive namespace isolation; bounded queue/cache components | Native pressure/cancellation proof, disk admission, CPU total-budget accounting, long-running reclamation |
| Model/quantization extensibility | Separate state, position, capture and KV-layout adapters; public connector options; explicit source/version checks | Broader qualified model, quantization and KV-format matrix; unsupported cases must remain explicit |
| Performance and task quality | Exact frozen DeepSWE token inputs and provenance; native SSE token/usage agreement; cold/arrival-decode measurements kept separate | Matched warmed C1/C4 throughput and latency, independent repeats, DFlash acceptance/verification cost, memory peaks, executed agent-task success |
| Delivery | Independent repository, reproducible GPU setup and dependency pins, local commits and qualification records | Clean install/release verification and final publication/review artifacts |

The current corrected bounded target diagnostic uses 128-token records and a
1024-token batch budget. The old qw3 frozen reference used 32-token records and
its own final-query override. Keep those differences visible; do not change
precision, K/G, histories or output stopping rules to manufacture an improvement.
There is no hard 130 tok/s or 48 GiB acceptance threshold. Substantial concurrent
throughput improvement must be supported by actual matched measurements.

## Current execution stage (2026-09-29)

The latest source includes the cold-block residency correction, component-level
background GC, exact terminal archive clipping, asynchronous retirement and an
opt-in Q8 integer Tensor Core adapter. Short C1/C4 native runs verify terminal
archive prefixes and complete cleanup. Recurrent accepted-boundary GPU components
pass, but are not yet wired into persistent serving. GC serving activation,
capacity-pressure/cancellation and persistent-session qualification remain open.

The corrected long-C4 native Q8 run and Tensor Core run are both complete.
`engine_long_c4_dflash_direct_cadence_02` (`053383f`) measures 10.443 whole-window
arrival decode tok/s; `engine_long_c4_dflash_tensorcore_01` (`2eae320`) measures
18.619 tok/s, with service time 1862.795 versus 761.158 seconds. They use the same
four inputs, 498 total output tokens, Q8_0/FP8, DFlash2, K131072/G65536, B128,
direct pinned reads, 36 GiB CPU cache, cadence 8 and asynchronous retirement.
The earlier baseline predates terminal clipping; this is an observed comparison,
not a repeated same-commit controlled experiment. Mixed prefill/refresh blocking
and three changed output streams require further performance and quality checks.

Actual official agent qualification is in progress. The first pair of attempts
reached the API but received HTTP 400 before inference due to missing native tool
parser flags. A subsequent short preflight exposed a missing capture-directory
creation in the backend launcher. Both infrastructure failures are retained;
neither is a model score. The corrected launcher enables native Qwen XML tools
and reasoning parsing and prepares telemetry before starting engine processes.
The latest full host suite passes 194 tests, with 41 GPU cases skipped while the
model service is active (`components_nvme_reservation_host_20260929.xml`).

The backend launcher also now arms capture after native warmup. An initial
HTTP-only preflight succeeded but the unarmed engine then failed at retirement;
that result is explicitly withdrawn in `agent_backend_tensorcore_03`'s correction
record. The replacement preflight requires tool parsing, native metrics, exact
terminal archive, zero archive/physical-page ownership and a live API. It passes
on `agent_backend_tensorcore_04` (source `9cf462a`): 284 prompt/31 output tokens,
314 archived rows, 1072/1073 free native blocks (one null), no remaining request,
reservation, download or archive slot. The two official attempts in
`agent_tensorcore_c2_initial_02` subsequently exposed the original proxy's
900-second read timeout and were stopped as infrastructure failures. No score
or clean C2 performance claim is made. Replacement backend `_05` passes tool
and actual GPU cancellation preflights; `agent_tensorcore_c2_initial_03` has now
started through the timeout-aligned proxy and disconnect-aware relay.

All earlier compressed model runs predate the cold-block residency correction.
Their timings and cleanup observations are retained in the qualification record,
but their compressed numerical-correctness/performance acceptance is withdrawn.
The requirement table above is a roadmap of historical evidence, not an override
of this correction. Corrected model runs, matched warmed repeats and executed
agent-task quality remain required.

## Evidence corrections

`engine_compression_k128g64_02` completed deterministic repeated replay but used
an incorrect cache-group classifier: native Mamba specs also advertise slot
mapping. Its recurrent-metadata semantic claim was withdrawn. The corrected
helper selects actual full-attention specs, and the corrected real bounded run
above passes. Keep the old artifact as a historical observation, not a semantic
reference. Detailed measurements and limitations are in
[the qualification record](../qualification-20260928.md), with measurement definitions
in [measurement.md](../measurement.md) and boundaries in [support.md](../support.md).


## Experimental bounded DFlash2 wiring

The new diagnostic `scripts/qualify_engine.py --bounded-dflash` enables a narrow
contract for the existing BF16 DFlash2 draft while retaining Q8_0 target / FP8
KV and K131072/G65536. Native vLLM uses method `dflash` and selects its DFlash2
implementation from the `DFlash2DraftModel` architecture. Five upstream source
hashes pin the draft factory, position/slot handling and model implementation.

The scheduler clips speculative queries at the accepted-history checkpoint and
reserves both speculative GDN rows and recycling-aware draft SWA pages. It leaves
room for verifier and drafter lookahead. Reselection scores use known prompt rows,
or only the first already-sampled input during decode, never unverified draft
candidates. Provisional target forwards restore every batch neighbour's recurrent
state; only the final forward reaches native sampling and drafting. Draft pages
and their logical position frame remain independent of the target dense window.
The actual native draft groups are checked against BF16 SWA geometry before any
compressed refresh. Mandatory target recent context must contain the draft window.
Logical histories beyond the draft RoPE table are rejected pending an explicit
renormalization/session adapter.

Native allocator tests cover combined full attention + speculative Mamba + SWA,
future-growth credit exclusion, recycling and release. These and the admission /
transaction / harness tests pass; they are not compressed DFlash GPU evidence.
The long target C4 run `engine_long_c4_cached_packed_01` was launched from commit
`776bb7a` before these DFlash changes and continues with that loaded runtime and
its saved source manifest. Do not attribute its eventual results to later code.


A separate retirement measurement option now allows omitting diagnostic archive
readback, while preserving publication fences and complete cleanup. Both modes
are covered by real host-archive retirement tests; the publication-only mode has
not yet been measured with the native GPU server. The default still verifies
all checksums. Its implementation does not change the live C4 run, which retains
full readback, and does not satisfy the outstanding background-retirement gate.

The native cancellation diagnostic is now wired behind `--cancel-during-refresh`.
Worker transaction timestamps, native termination status and post-output capacity
snapshots support a client-disconnect test with subsequent C4 recovery. The
validator requires physical page return, not just cleared request IDs or credits.
CPU/socket and scheduler tests pass; the actual native GPU cancellation run remains
pending. Source changes made after `engine_long_c4_cached_packed_01` launched do
not affect its already loaded runtime.


`engine_deepswe_effect_bounded_dflash_01` is now running the same exact 197110-token
Effect input as the corrected target C1, from commit `62726f0`. It retains full
checksum verification and the same K/G and precision. The requested batch budget
was 1024, but native draft-slot reservation followed by KVMem alignment produces
896-token prefill chunks. It is a correctness run, not a matched 1024-query
performance comparison. Future matched C1 uses budget 1152 / request cap 1024;
C4 uses budget 4224 / cap 1024, and must verify the actual scheduled queries.
It has now passed 74 matched refreshes and complete five-request archive cleanup.
C1 generated 77 natural-stop tokens at 85.509 arrival decode tok/s over 0.889 s;
the subsequent short C4 reached 91.956 aggregate tok/s. C1 token IDs match the
cached target control and all short C4 texts match prior native DFlash controls.
This is compressed-prefill / later-decode evidence; decode-time refresh and long
DFlash C4 are still pending. See the qualification record for timing caveats.

## Direct pinned archive reads and reproducible workers

The live DFlash run's host breakdown identified repeated zero-initialization of
the CPU read buffer as a large cost: one 5.44 GB, entirely cache-hit refresh spent
2.645 s allocating buffers, 0.388 s copying cached bytes, and 3.927 s assembling
in total. These are one-round wall timings, not an isolated optimization result.

The new opt-in `--packed-assembly --direct-pinned-reads` reads immutable records
directly into the existing two bounded pinned H2D slots. It removes the separate
zeroed bytearray and second host copy. Namespace/generation checks and archive
leases remain mandatory; CPU preparation and DMA both own their staging slot.
Injected read failure after an earlier GPU chunk must retain the transaction
failure fence. The default path remains unchanged. CPU suite: 104 passed,
28 GPU tests skipped at that stage. The complete CPU/GPU suite subsequently
passed 136 tests at `82e504c`, including direct-path bytes and failed-read fences.
Direct-path native model serving/performance qualification is still pending.
`assembly_host_prepare_seconds` now includes staging copy/submission and waits
for a bounded slot in both paths; earlier reports excluded the upload call, so
use total `assembly_seconds` for comparisons spanning this instrumentation change.

Future engine runs copy the worker Python package and built native extension into
their artifact directory and import that snapshot in all child processes. Both
source and worker manifests are hashed; changing the workspace while a test runs
cannot alter a later worker import. Dependency environments remain pinned and
must not be rebuilt or updated during a run.

An experimental `--prefill-cadence N` now requests native prefill throttling on
N-1 of each N mixed scheduling steps. It does not resize queries, reorder the
running queue or override native admission/capacity gates. Prefill-only work,
unavailable decode and capacity-bound scheduling proceed normally. The default
is 1. This can reduce repeated prefill interruptions but cannot shorten one
blocking refresh or remove retirement pauses; GPU benefits remain unmeasured.
The harness records actual prefill/decode query-size histograms to detect draft
headroom and alignment changes in future comparisons. Targeted scheduler,
option and harness tests: 35 passed. Native cancellation qualification also
counts the disconnected request when waiting for archive retirement.

## Active next GPU run

`engine_long_c4_dflash_direct_cadence_01` launched from `80853e8`, with its own
frozen worker package and native extension. It uses the same four real long
inputs as the earlier target C4, Q8_0/FP8, BF16 DFlash2, K131072/G65536 and B128.
Candidates: direct pinned reads, 36 GiB CPU cache, packed assembly, prefill cadence
8, global batch budget 4224 / request cap 1024, and publication-only retirement.
It must demonstrate actual four-request native compressed batching and cleanup.
This is a combined-path qualification, not a matched speedup against the old
256-query, full-readback target C4. Actual scheduling histograms and complete
HTTP arrival windows will determine follow-up matched controls. Final outcome
is pending; the persistent goal remains active.

## Bounded asynchronous archive retirement

The opt-in `--async-archive-retirement` detaches a finalized authority only after
all native capture DMA has drained. A single bounded CPU worker publishes its
remaining records, optionally verifies every checksum, and reclaims the entire
namespace. Other native requests can keep executing. Publication, verification,
reclamation and acknowledgment timestamps are retained in the archive report.

The native connector deliberately retains the completed request's physical GPU
pages and lifetime reservation until CPU reclamation succeeds. An additional
authority-count gate caps live plus retiring requests at `max_num_seqs`, so
background cleanup cannot silently exceed the configured archive namespace
budget. This first implementation trades delayed capacity reuse for safe overlap;
it does not yet release GPU pages ahead of CPU retirement. Cancellation and
checksum/I/O failures cannot issue a successful free acknowledgment. Idle native
`no_forward` steps deliver completion even when every HTTP request has ended.
The pinned helper source hash rejects incompatible native completion semantics.

Host tests cover concurrent namespace writes during blocked checksum reading,
held generation leases, queue bounds, corrupt archive failure, admission with
spare physical pages, idle native connector completion, and graceful shutdown
ordering. Native GPU retirement/cancellation/pressure qualification remains
pending. The active long-C4 run uses its frozen pre-change package and synchronous
publication-only retirement; it cannot qualify this new option. This feature
does not implement active-history policy GC, persistent sessions or total CPU /
disk byte admission, which remain separate goal requirements.

Latest CPU regression after this integration: **120 passed, 28 GPU tests skipped**
to avoid contention with the active model run. The earlier complete CPU/GPU suite
of 136 tests qualifies the previous code, not asynchronous retirement. Evidence
for this CPU run is `artifacts/components_async_retirement_cpu.xml`.

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. Corrected model qualification is pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

The original repository GC now has Python bindings, including generation leases,
backing flags, policy configuration and GIL-safe background callback shutdown.
Five new host tests cover protection, callback failures/retry and shutdown.
These bindings are not yet activated in the serving authority: durable backing
publication, full retrieval-snapshot leases and bundle cleanup must be integrated
before enabling active-history GC. The qualification harness now stops its owned
server before joining interrupted streaming readers.

## Authority GC integration (CPU qualified, serving activation pending)

An optional `RequestAuthority(gc_config=...)` now attaches the original policy
worker. It publishes RawBacked/ValueBacked/Indexed only after successful durable
write completion. Scoring holds generation-qualified leases over the entire
live index through native selection; returning copied handle values alone does
not protect against concurrent retirement.

Background release reclaims the generation's disk allocation and optional CPU
read-cache entry, then queues a bounded tombstone. The foreground removes old
index generations before installing recycled reservations. The callback owns no
Store/authority/GC bridge reference, touches no GPU data, and never mutates the
foreground dictionary. Finalization stops policy GC before checksum iteration;
reports distinguish previously reclaimed records from live records actually
verified. Request-step aging advances once at construction; ingestion grace can
make progress within that request without inventing per-token request steps.

Seven integration cases cover fixed-capacity reuse across 272 accepted tokens
in an eight-block repository, original sink protection, cache invalidation,
scoring leases and exception cleanup, unobserved durable writes, background
single-request ingestion, and final checksum accounting. Combined CPU regression:
**133 passed, 28 GPU tests skipped** while the model run owns the GPU. Evidence:
`artifacts/components_authority_gc_cpu.xml`. Native serving options, scheduler
backpressure on repository exhaustion, and GPU GC qualification remain pending;
this component is not activated in the current model test.

The corrected long-C4 combined-path run is
`engine_long_c4_dflash_direct_cadence_02`, launched from `053383f` with a frozen
worker snapshot. It additionally opts into asynchronous archive retirement.
Actual decode rates and final lifecycle results remain pending; no speedup is
claimed from the earlier interrupted run or from the component tests.

## Scheduler terminal prefix (CPU qualified, GPU pending)

The bounded connector now carries the finished request's scheduler-owned prefix
before native request fields are discarded. Sampling acceptance is not the final
boundary: native EOS/length checks can trim an accepted speculative burst. The
worker applies that boundary before draining the latest capture packet. Aborted
in-flight rows and rows past the final evaluated output prefix are counted as
`terminal_discarded_tokens`, separately from sampling rejections, and never enter
the raw-K/V archive or MeanK index. Late boundaries after excess publication fail
closed; this path does not silently rewind immutable records.

New bounded archive reports require exact agreement between scheduler prefix,
archive history and HTTP token usage, replacing the historical extra-burst
allowance. Legacy reports retain their old validation semantics. Finalization
also distinguishes GC-deleted records from remaining records checksum-verified
on exit. CPU tests cover EOS overshoot, ordinary decode, prefill/decode abort,
ignored requests, metadata transfer, a real archive with mixed-request capture,
and strict qualification checks. Current CPU regression: **143 passed, 28 GPU
tests skipped** (`artifacts/components_terminal_prefix_cpu.xml`).

This fixes prospective capture publication only. Native recurrent state and
DFlash sliding state may still have advanced past the final retained prefix;
persistent sessions remain disabled until final restore/replay is implemented
and qualified. The running long-C4 uses its earlier frozen source and therefore
does not qualify this new boundary path. Its finished namespaces are discarded,
not reused as sessions; its final report must retain that limitation.

## Canonical accepted GDN state (GPU qualification pending)

`NativeRecurrentState.canonicalize_accepted` can move a nonempty retained native
speculative prefix to the base SSM row and canonical convolution window, then
reset native acceptance to one for a subsequent prefill/decode. It supports the
native SD and DS convolution layouts, validates all groups before mutation and
returns a CUDA completion fence. Source hashes pin the native Qwen GDN layer,
convolution kernel, recurrent kernel and hybrid-state lifecycle. This operation
is not enabled in serving or claimed as session support yet.

Zero retained rows and partial ordinary prefill fail explicitly: their previous
recurrent states require checkpoint restore/replay. The independent DFlash draft
KV/hidden lifecycle and logical-position limits also still require session
integration. GPU cases compare speculative-prefix normalization with independently
executed native convolution and GDN prefixes, including a subsequent ordinary
convolution step and untouched neighbouring pages. These six cases are staged,
not yet executed while the long-C4 model owns the GPU.

Latest host regression: **146 passed, 34 skipped** (28 existing GPU tests plus
6 new native-prefix GPU cases), `artifacts/components_accepted_state_cpu.xml`.

## Next performance evidence and agent gate preparation

The harness now exposes an optional native profiler window for concurrent work,
with a fixed step limit and an explicit instrumented-only timing label. Host
checks validate against the installed vLLM ProfilerConfig and cover overlapping
GPU kernel interval accounting; targeted profiling/qualification tests: 18 pass.
No GPU profile has been run yet and no bottleneck improvement is claimed.

Read-only inspection confirms the existing official DeepSWE environment uses
Pier `datacurve-pier` 0.3.1 and mini-swe-agent 2.4.6. The original requestplan10
runner restarts QW3 for every task and allows a 3M logical context; invoking it
unchanged would violate this plugin's shared-vLLM test scope and current 262K
draft-position boundary. Agent evaluation must therefore get its own vLLM
runner with the same task/agent/verifier provenance, explicit limits, and actual
container/verifier results. Frozen prompts and parseable tool calls remain
insufficient quality evidence. Existing containers/services were only inspected,
not stopped or replaced.

## Latest native result and next gate

The corrected long-C4 run has finished: all four compressed requests and all
resource cleanup checks pass. Whole-group arrival decode is 10.443 tok/s, with
mixed-prefill stalls up to 25.789 s; it is not a performance acceptance. The
short 3.249 s all-stream overlap shows 98.189 arrivals/s only as a diagnostic.
Native sampling draft acceptance is 44.982%, GPU sampled peak is 79.32 GiB,
and asynchronous CPU archive reclamation takes about 28–31 ms per request.
See the qualification record for complete scopes and limits.

All 182 current CPU/GPU component cases pass, including canonical GDN-prefix
state comparisons. The reference fixture correction is recorded with the failed
and corrected artifacts. Next native run will exercise exact terminal-prefix
capture and a bounded GPU profiler window; profiling timings are not benchmarks.
Persistent streaming-session integration and actual agent execution remain open.

The initial terminal-profiler launch (`engine_terminal_dflash_profile_01`) was
stopped before readiness/requests because the harness's default bounded mode
would synthesize a 134144-token C1. Its stop reason and interrupted result remain
in the artifact directory. An explicit `--bounded-short-lifecycle` mode now
separates this terminal/retirement check from the existing mandatory compression
gate. Profiling/qualification host tests: 21 pass; long gates are not weakened.

## Q8 batch linear hotspot and experimental replacement

The short native terminal/profiler gate passed. Its Q8_0 DP4A matrix kernel
accounts for 91.42% of summed GPU kernel time. An optional SM120 Tensor Core
adapter now computes exact INT32 dots per original 32-element Q8 group, scales
in FP32 with unchanged FP16 scales, and reduces groups without unpacking the
whole model into BF16 weights. It only replaces MMQ batches with >=16 rows;
the single-request 8-row DFlash verifier keeps its original path.

The first prototype used round-to-nearest division, which differs at quantizer
ties from the upstream `--use_fast_math` build. That version was never enabled
in serving. The corrected quantizer uses the native constant reciprocal and
`div.approx.ftz`; identity Q8 weights expose native activation quantization and
match every tested value exactly. FP64 grouped-dot references and native CUDA
comparisons pass six GPU cases. The full component suite passes **191 tests**
(`components_q8_tensorcore.xml`). A subsequent short-C4 warmup archive-integrity
test also passes with all 20 qualification host tests.

Isolated real layer-0 QKV, FFN gate and FFN down weights with synthetic inputs
show 1.84–2.02x kernel speed ratios at 32 rows and 1.94–2.09x at 1024 rows.
Relative RMS output differences are 2.6e-5–5.3e-5 in these batch cases, reflecting
FP32 reduction order and BF16 rounding boundaries. Evidence is retained in
`artifacts/q8_tensorcore_real_weights.json`; these are not model throughput or
task-quality results. The switch remains experimental and off by default.

`--warmup-short-concurrent` retains an entire additional short C4 request group
before the measured group, including its streams and archive/terminal checks.
It is only allowed with short lifecycle and streaming metrics, without profiling;
it does not relax the mandatory long compression gate. Native baseline/candidate
runs and real task-quality qualification remain pending.

The first short native comparison passes all lifecycle checks but shows only
88.20 -> 89.09 aggregate arrival tok/s because the candidate retains a 2.799 s
measured pause. Four measured outputs differ. Exact batch-row specialization
has now been removed from the matrix and output reduction kernels; seven
targeted GPU tests pass, including C4/C3/C2 row transitions reusing the same
compiled kernels. This change needs a new native run and is not a retroactive
correction to the recorded throughput.

An installed wheel in a fresh CPU-only environment passes 22 core/storage/GC
tests outside the source package path. Wheel metadata now carries the original
QW3 and vLLM licenses alongside this package's license. Evidence and wheel hash:
`artifacts/wheel-smoke/summary.json`. This verifies host packaging, not a clean
GPU dependency installation or release readiness.

Dynamic-row native follow-up passes and reaches 173.112 aggregate arrival tok/s
in short C4 versus the baseline 88.203 (1.9626x), with longest measured gap
0.1287 s and unchanged ~89 tok/s C1. All nine request lifecycle checks pass.
The same long frozen C4 is the next gate. Output IDs differ under concurrency;
long-history and executed task-quality acceptance remain open, and the adapter
is still opt-in. See the qualification record for full measurement scope.

## Official agent execution entry point

`scripts/qualify_agent.py` snapshots the fixed DeepSWE tasks and invokes pinned
Pier/mini-swe-agent against one shared API, preserving original solver/tool/
verifier behavior and one-attempt outcomes. Companion backend/relay launchers
freeze worker code, collect native per-request metrics, and own only their
new processes/containers/network. Fifteen host tests cover scored failures,
invalid/duplicate evidence, detached snapshots, recipe preservation and exact
relay HTTP payloads. A prepare-only task snapshot validates actual installed
versions and task provenance; it is not a quality score.

A real fresh Docker client reaches the current API through the newly owned
bridge relay. Only `/v1/models` was queried, adding no generation work to the
ongoing GPU benchmark. The relay container/network were subsequently removed;
existing QW3 containers and its occupied port 443 were untouched. Artifacts:
`agent_prepare_02`, `agent_relay_smoke_01`, `agent_harness_complete_cpu.xml`.
See `benchmarks/deepswe/README.md` for execution/measurement scope. Actual agent
attempts and native-metric collection on those attempts remain to be run.

The long-C4 Tensor Core gate has now completed: same frozen inputs/output
lengths, ten matching refreshes and complete resource cleanup. Whole-window
arrival decode is 18.619 tok/s versus 10.443 earlier; service time falls from
1862.795 to 761.158 s and longest gap from 25.789 to 14.846 s. Mixed prefill /
refresh interference remains. Three outputs differ, so executed task-quality
qualification is still mandatory. The first shared agent backend is being
started from this completed recipe with native per-request timing enabled.

The shared Tensor Core agent backend is ready with native timing and speculative
summary metrics. A new owned relay links it to two concurrent official attempts:
`termenv-preserve-ansi-resets` and `tomlkit-toml-table-converters`, under
`artifacts/agent_tensorcore_c2_initial_01`. The attempts are still in environment
build/setup, not scored. Their task/adapter/harness and ready backend/relay
evidence are detached snapshots. No retry or solver-policy change is introduced.

An additional packaging check builds an sdist, extracts it into a fresh directory
with no `.so`, recompiles the native extension, installs that wheel into the
isolated CPU environment and passes 22 core/storage/GC cases. Archive/wheel hashes
and logs are in `artifacts/source-build-smoke`. Full GPU clean installation and
release qualification remain separate gates.

The first real agent launch reached the API but exposed missing tool-serving
flags inherited from the completion-only recipe. All observed model calls
returned HTTP 400 before inference (`auto` tool choice requires an enabled
parser). This is retained as a setup failure, not a model score; the attempts
and their owned backend/relay were stopped. The qualified tokenizer explicitly
uses `<function=...>/<parameter=...>` XML. The agent backend now enables native
`qwen3_xml` tool parsing, auto tool choice, and `qwen3` reasoning parsing. Existing
QW3 services remain untouched. Requalification must verify actual tool calls and
native metrics before accepting any agent measurement.

## Repository GC serving integration (experimental, default off)

`repository_gc: true` in the public connector's KVMem options now attaches one
original CPU policy worker to each captured request authority. Its ready reserve
matches `QwenExecutor::kvmem_start_repository_gc`: low=max(capacity/20,
2*append_blocks), high=max(capacity/10, low+append_blocks), with recent-band and
accepted-ingestion grace preserved. K+G plus the high reserve must fit strictly
below repository capacity; startup rejects configurations that would need to
evict protected working state to make space. Neither native GPU pages nor
speculative/recurrent state is freed by the CPU policy worker.

Reclamation signals the capture runtime after generation-bound archive release.
No-write capacity exhaustion can now wait for this background progress instead
of immediately raising BufferError. This is a measured synchronous fallback,
not nonblocking scheduler admission: `gc_wait_count`/`gc_wait_seconds` are exported,
and absence of append progress for 30 seconds fails explicitly. Scheduler-side
ready-capacity feedback is now wired as described below; CPU/NVMe global quotas
and GPU pressure qualification remain necessary before production activation.

The native CPU integration test drives two namespaces sharing one archive through
336 accepted tokens each, with only 24 repository slots per namespace. Both
reclaim more than 50 generations, preserve sink data and release all slots at
shutdown. Separate checks cover reserve rejection and bounded pressure waiting.
38 targeted tests pass (`serving_gc_cpu_02.xml`); this is not GPU or quality proof.
The GPU harness exposes `--repository-gc`, `--repository-blocks` and
`--require-gc-reclamation`; the latter prevents an enabled-but-idle policy from
passing a pressure gate. Current official agent runs use their frozen prior code
and do not activate this change.

After integration, the complete host suite passes 181 cases with 41 GPU cases
deferred (`components_serving_gc_host_20260929.xml`). At this observation the
two official tasks have completed 17 model calls without HTTP errors and remain
live; accepted-history capture is progressing for both. Neither has a verifier
score. These live counters are progress evidence, not a final quality or
aggregate-throughput result.

## Per-request archive capacity feedback

With experimental repository GC enabled, the existing native connector metadata
channel now reports each live authority's accepted history and ready repository
records. Scheduler admission debits all accepted/computing rows beyond that
snapshot, including worst-case speculative acceptance, while crediting unused
space in a mutable tail. It defers the whole original query if it cannot fit;
it does not shorten candidate blocks or change K/G. The native running queue
continues to other requests. Idle no-forward steps still return updated credits,
so background reclamation can resume a blocked request without a model forward.
Monotonic metadata sequence/history checks reject stale feedback; completed
requests lose their cached capacity state.

The worker does **not** drain the most recent capture packet while constructing
feedback: native EOS truncation arrives in the next scheduler message. This
preserves exact terminal-prefix clipping. Pending DMA is instead conservatively
charged against scheduler history until normal capture processing publishes it.
`archive_capacity_deferrals` is recorded alongside GC fallback wait metrics.

52 targeted host tests pass (`archive_admission_cpu_02.xml`), including the real
connector types, every partial-block tail, delayed capture, rejected candidates,
neighbour isolation, stale messages, terminal clipping and the scheduler hook.
GPU behaviour and reduction of capture-thread waits are unverified. This covers
repository-record credits; it does not yet implement global host/NVMe admission
or asynchronous treatment of every I/O-queue/staging pressure condition. The
live agent test remains on its frozen earlier worker package.

## Strict shared NVMe reservation

Optional `nvme_preallocate` reserves and verifies the physical disk blocks for
the full shared arena during model loading, before API readiness. The binding
uses the original tier's preallocation path and verifies the private inode's
size/allocation; unsupported or failed allocation cannot silently become sparse
success. The private file is unlinked on both success and failure, with the tier
fd retaining ownership until shutdown. Imported qw3 source remains unchanged.
`nvme_max_bytes` rejects an oversized geometry-derived shared arena before any
arena is created. It is a startup capacity quota, not dynamic cross-service
quota management. Defaults remain off/geometry-derived for existing recipes.

The GPU harness exposes `--nvme-preallocate` and `--nvme-max-mib` and verifies
the corresponding runtime report. Native CPU tests verify physical allocation,
I/O, an injected file-size reservation failure, unlink/close lifecycle, and
global quota rejection. 45 targeted tests and the full 194-case host suite pass;
41 GPU cases were deferred. GPU startup reservation and global CPU-buffer byte
admission remain unqualified.

## Agent transport correction and real cancellation

The official generated Squid proxy in `_02` returned `TCP_MISS_TIMEDOUT/504`
after 900642 ms. The client retried while the old relay kept upstream inference
alive, creating three active requests for two serial agents. Both attempts were
stopped, retained as unscored infrastructure failures, and their owned backend,
relay, task containers and networks were removed. `transport_failure.json` and
the saved proxy log preserve the evidence. Host builds/tests also overlapped
this run; it is not an exclusive performance benchmark.

The new process-local `pier_transport_entry.py` verifies the pinned official
network-setup source hash and changes only read/forward timeouts to the existing
21600-second model-call budget. It does not edit the installed Pier package or
alter allowlists, authentication, solver policy, retry count or verifier. The
relay now monitors client disconnects during silent upstream work, shuts down
the native API socket and records delivery/cancellation separately. Undelivered
responses are excluded from completed-request timing. 14 transport/harness host
tests pass; the pinned real Pier CLI entry point also loads successfully.

`agent_backend_tensorcore_05/cancel_preflight.json` proves actual native GPU
abort through the new relay: after 2048 accepted prefill tokens, client disconnect
releases all request, reservation, physical-page, archive and download ownership
in 1.107 seconds. The terminal prefix is `FINISHED_ABORTED` at 2048; another 1024
in-flight rows are discarded. The API stays healthy. This is active-prefill
cancellation, not the still-required compressed-refresh cancellation gate.
The next real attempts are `_03`, with instance-bound tool and cancellation
evidence. No task score is available yet.

## Shared in-flight archive byte admission

`archive_io_max_bytes` (harness: `--archive-io-max-mib`) now bounds the sum of
logical payload bytes across admitted archive reads and writes, in addition to
the existing job-count cap. It applies across request namespaces, cache hits,
direct reads into caller-owned buffers and detached finalization. Write snapshots
are taken only after admission; publication, I/O failure, executor rejection and
preparation rollback release credits and generation leases. A permanently oversized
batch fails explicitly. Model startup checks that the configured cap fits a full
K+G read and the maximal capture flush, rather than discovering an impossible
quota during inference. Default zero preserves prior count-only recipes.

Current/peak bytes and deferrals are exported in archive evidence. Capacity held
by detached finalizers is recognized by capture backpressure instead of falsely
reporting repository exhaustion. This remains a synchronous fallback. The cap
is **in-flight I/O extent**, not a whole-process RSS limit: completed read results,
resident authority tensors, pinned pools, read-cache entries and serialization
scratch still have separate owners/bounds. End-to-end host-byte admission and
fair asynchronous scheduling remain pending. No throughput gain is claimed.

The targeted native CPU suite passes 56 tests, including mixed-size namespace
competition, all-cache/direct reads, refused/oversized writes before snapshot,
read/write errors, executor rejection, lease reclamation and detached-retirement
pressure (`archive_io_budget_cpu_03.xml`). The complete host suite passes
212 tests with 41 GPU cases deferred (`components_archive_io_host_20260929.xml`).
The live `_03` official agent pair uses frozen prior code and does not activate
this quota; at this checkpoint 19 model calls have completed without relay error,
but neither task has a verifier score. These counts are not a quality result or
an exclusive performance measurement.

## Prompt-prefix evidence and fenced repository rewind

`agent_prefix_analysis_02` reconstructs 21 completed historical prompts from
snapshotted trajectories using the installed tokenizer/template and the official
bash schema. All reconstructed token counts match native response usage, and all
19 adjacent reconstructed prompts preserve the entire preceding prompt prefix.
The first diagnostic accidentally counted BatchEncoding keys rather than token
IDs; its correction file withdraws it. The replacement explicitly requests a
token list. No generated raw token IDs were saved in these calls, so this does
not establish equality of the generated suffix or executable state reuse.

The original imported core already implements repository checkpoint and rewind
transactions. New bindings expose these without changing imported source.
`RequestMemory` now pins a current checkpoint, validates request/branch identity,
prepares a sealed-record rewind and publishes it only after the caller's suffix
cleanup/model-state fence completes. Returned discarded spans copy value handles
instead of retaining hidden transaction owners. Rewind permanently invalidates
later checkpoints even if another branch grows back to the same length. Native
transactions retain their Store lifetime; checkpoint ownership retains prefix
leases until explicitly released. Partial-record boundaries are rejected because
the immutable archive does not yet have a partial-tail branch-generation protocol.

23 targeted host tests pass (`repository_rewind_cpu_02.xml`), including fence
delay, failure cleanup, cross-request/partial-boundary rejection, permanent branch
invalidation, suffix recycling and checkpoint/native owner lifetime. Rebuilding
the extension and running the full host suite yields 215 passes and 41 deferred
GPU cases (`components_repository_rewind_host_20260929.xml`). The live agent worker
continues using its frozen extension, verified through `/proc/.../maps`.

These are prerequisites only. The HTTP scheduler/worker turn handshake, authority
index rewind, GDN/draft restoration, exact prompt comparison and idle-session
admission remain to be connected and GPU-tested. `engine-integration.md` now
describes the current execution path and these remaining boundaries instead of
the earlier staged probe-only architecture.

## Authority rewind and exact window reconstruction

`RequestAuthority.begin_rewind` now waits nonblockingly for its existing archive
publication before creating the native transaction. A supplied model/page cleanup
fence gates commit. After commit, discarded generations leave the retrieval index
but remain in a bounded cleanup set until archive readers release them. Append,
selection and new reads cannot resume during this transition. Shared archive
namespaces remain independent and recycled suffix slots receive new generations.
Close can drain an interrupted rewind without bypassing its fence or checkpoint
leases.

Background GC remains running through rewind. Its tombstones distinguish discarded
suffixes (including unarchived partial tails) from retained-history collection.
GC hole accounting is reset against the shorter branch; collected old suffixes
cannot inflate a new branch's publication count. Tests force foreground release
to defer to the real GC thread, verifying at least 40 suffix releases, repeated
branching after more than 50 normal GC collections, and unchanged prefix data.
39 targeted tests pass (`authority_rewind_cpu_02.xml`).

`WindowCheckpoint` also captures the original chronological selected records,
repository generations and dense positions. The original Store supplies value
tuples, without hidden native owner references. Restore validates the whole frame
and requires authority history to have already rewound. It invalidates working
residency and prepares full K **and V** reload into fresh native pages: equal dense
positions cannot prove new allocation contents. Ordinary assembly fences still
gate publication. The sealed frame may include K plus the accepted prefill chunk,
so it is bounded by K+G, not incorrectly forced back to K.

25 focused host tests pass (`checkpoint_window_cpu_01.xml`); the rebuilt extension
and full host suite pass 223 tests with 41 GPU cases deferred
(`components_checkpoint_window_host_20260929.xml`). The existing ordinary/packed/
direct GPU assembly gate now includes restoration into a different poisoned page
table and byte comparison against native FP8 kernels; these added GPU checks have
**not yet run**, because the two official agent trials still own the GPU.
Neither change is activated by the frozen live backend. Persistent HTTP reuse
still requires model/draft state export/import, exact prompt matching, scheduler
admission and completion acknowledgment; these host results do not establish it.

## Portable prompt model state and native checkpoint allocation

The prompt-state adapter captures only exact non-final prefill boundaries with
the canonical native acceptance counter. It copies GDN base convolution/SSM rows
and the DFlash visible accepted context to CPU after validating the native group
and page layout. DFlash's head-major BF16 cache is addressed by logical positions
and its actual physical page table; null entries outside the sliding window are
ignored, while null/aliased visible pages are rejected. No GPU page view remains
in the snapshot. Its payload-byte limit is checked before materializing copies.

Restore accepts a different native request/state index and different page IDs,
validates all tensors before the first write, and returns a CUDA completion event
when applicable. Host fixtures cover an evicted source request, poisoned old pages,
unchanged neighbours/speculative scratch rows, strided draft caches and a sliding
tail spanning physical pages. Target-only and DFlash fixtures pass. This is not
support for restoring a speculative terminal state, nor a global host-memory cap.
The full host suite passes 227 tests with 42 explicit GPU cases skipped
(`components_portable_prompt_host_20260929.xml`). The new CUDA fixture is pending.

Native connector prefix allocation also needs more than a window offset:
`add_local_computed_blocks` and `allocate_external_computed_blocks` otherwise
allocate the full logical history even if the allocation estimate was compressed.
The working-set adapter now translates all three paths only for an explicitly
prepared exact checkpoint import. Fresh local-computed zero remains an ownership
count; target external history becomes the saved dense frame. GDN and draft
managers retain logical history and native sliding eviction. Commit requires the
correct checkpoint and completion fence; allocated cancellation stays under native
fenced cleanup. Thirteen focused tests pass against actual native managers
(`native_checkpoint_import_cpu_01.xml`), including hybrid speculative allocation,
neighbour isolation and future-growth reservations. No connector prefix hit or
HTTP state reuse is enabled by these changes. The frozen agent backend is unchanged.
The combined host regression passes 230 tests, with 42 GPU cases deferred
(`components_prompt_import_host_20260929.xml`).

## Exact prompt checkpoint ownership pool

`PromptCheckpointPool` now combines sealed `WindowCheckpoint`, portable model
state and exact token-prefix bytes. The pool checks retained-entry and payload
limits before copying model state. It never interprets a mere session ID or a
hash collision as a reusable prefix, and it leaves at least one input token for
native prefill. The serving coordinator must provide a properly scoped namespace;
this internal component does not implement authentication or HTTP admission.

Capture initially borrows the live authority. After the caller detaches it,
retirement waits for archive publication, the supplied completion fence, rewind
and all discarded suffix readers. Only then can a matching next turn claim it.
The target frame must undergo full reconstruction before ownership transfers to
the fresh request. Failed imports require a cleanup fence; eviction releases our
checkpoint pins but keeps entry/byte credits charged until external readers and
archive reclamation drain. Dropping a live snapshot leaves its executing authority
intact. Obsolete handles cannot retain internal native prefix leases.

Fifteen focused host checks pass (`prompt_checkpoint_pool_cpu_02.xml`). They compose
real repository/archive rewind with CPU GDN/draft fixtures and new native request
indices, check exact-prefix misses, per-namespace isolation, suffix-reader stalls,
admission failure and failed-import cleanup. The full host suite passes 236 tests,
with 42 GPU cases skipped (`components_prompt_pool_host_20260929.xml`). This pool
is not connected to HTTP serving yet. It also does not replace shared authority
admission, total host-memory accounting, an idle eviction/expiry policy or actual
GPU/model continuation tests.

The ongoing official pair remains on frozen backend `agent_backend_tensorcore_05`.
At 2026-09-29 06:41 CST, 21 termenv and 24 tomlkit model responses have completed
with no relay error or undelivered response. Both official trials remain running
without final scores. These are live observations, not a clean performance result;
host development/checks overlap and cross-turn prompt reuse is still disabled.

## Prompt-session scheduler and worker wiring

The previously separate checkpoint primitives are now connected behind explicit
`session_capacity` / `session_checkpoint_bytes` options. `SessionLedger` carries
generation-qualified binding/capture/eviction messages through the native KV
connector and accepts reusable prefixes only after worker retirement acknowledgment.
Exact token-prefix misses evict/recompute; cancellation before allocation restores
the idle lease. One session's active turn serializes later turns without preventing
unrelated native requests from running. Idle LRU eviction remains fenced through
storage readers and shared archive capacity includes idle authority allowances.

`SessionWorker` receives controls before native execution and imports state in the
pre-attention hook, after native request state/table initialization and block zeroing.
It reconstructs the original target FP8 frame, restores native GDN plus BF16 draft
context, and then publishes the restored authority to capture. A final-prefill
checkpoint command may capture the next turn at that same boundary. Import and
semantic query refresh may share a forward; the connector consumes the import
completion before allowing refresh page trimming. The original repository GC
remains running during optional checksum verification, using snapshot leases for
stable reads, and starts a new policy step on resumed turns. Its reserve now covers
the union of a saved and current working frame.

The worker's new host checks use the real archive/repository and CPU state fixtures,
with a stand-in target assembler; they do not claim GPU assembly/model correctness.
They cover full capture/retire/import, recycled native state indices, prefix misses,
byte/count admission, checksum failure, failed-import cleanup and empty short-turn
retirement. A native-manager scheduler test covers import plus reselection within
one forward and ordered completion. The combined host run passes 252 tests with
42 GPU cases skipped (`components_sessions_host_20260929.xml`). Six backend/HTTP
gate checks also pass (`session_http_gate_cpu_01.xml`), including rejection of
changed token outputs or fictitious reuse counters.

`serve_agent_backend.py` can explicitly launch the changed session configuration;
the new command is labeled experimental and is not substituted for the base recipe's
qualification. `qualify_sessions.py` is the pending actual HTTP gate for unchanged
greedy token IDs after native page recycling and for changed-prefix recomputation.
It can take a frozen long input without changing its tokens. See `prompt-sessions.md`
for the exact commands and limits. The official agent pair continues unchanged on
its frozen non-session backend. No session throughput or quality claim is made.
The final combined host regression passes 254 tests, with 42 GPU cases deferred
(`components_sessions_final_host_20260929.xml`).

A fresh source-release check at commit `b9ec4ae` builds an sdist, extracts it into
an empty directory with no extension binary, recompiles the native wheel and
installs it into a new environment without torch or vLLM. The installed package
passes 59 host core/archive/checkpoint/configuration/session-control tests; one
native-connector test is skipped because that environment has no vLLM. All four
new session modules and all three project/vendor licenses are present in the wheel.
Hashes, logs and JUnit evidence are in `artifacts/source-build-sessions_01`. This
qualifies host packaging of the current source; it is not a clean GPU dependency
installation or an actual session-model result.
