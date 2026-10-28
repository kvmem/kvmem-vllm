# Tool-return interference investigation and opt-in optimizations

Status: GPU correctness and all five fixed-input comparisons passed; natural
DeepSWE observation windows are in progress.
The previous formal DeepSWE run remains paused. No timer was created or resumed.

## Scope and preserved evidence

The workload remains 8 clients, K=98,304, G=16,384, logical ceiling 10,000,000,
NVFP4 weights, FP8 KV, MTP(4), block 32, sink 1,312, recent 0. MeanK/FPS4 quotas
remain 50:50 with FPS4 neighbors inside the budget, graph/guidance off, cumulative
attention for prefill pressure, and the existing CPU-history LRU policy. No SSD
KV spill or physical prefill coalescing is introduced.

The formal run is
`/data/chaidi/kvmem-vllm-experiments/deepswe-hybrid-lru-96k-10m-formal-20261025-r1`.
Its pause audit is `first16/optimization-pause-20261025-r2`: six completed tasks
and 54 completed evidence files were hash checked, and seven unfinished container
filesystems were committed to separately named images. These images preserve
files, not live processes or an in-memory KV checkpoint. Original answers must
not be regenerated to produce a cleaner comparison.

## Findings and changes

1. A single returning request can delay peers through mixed prefill/decode steps.
   The fixed-history control reproduced 0.6–0.8 second peer SSE gaps with 2K–8K
   added input, while ordinary decode model-step median was about 48 ms. SSE gaps
   include delivery and can contain several MTP-accepted tokens; they are not
   per-token GPU kernel timings.
2. Foreground terminal checkpoint preparation includes token packing, window
   validation and pinned payload allocation. Diagnostic traces measured initial
   158 MB pinned allocations around 131–139 ms, with model-state capture taking
   135–146 ms in those examples. This does not mean every checkpoint pays this
   cost. Repeated Python token/frame work also grows with logical history.
3. The pinned Claude Code 2.1.148 client applies a second Unicode unescape to
   parsed tool argument strings. A real-client Write/Read test demonstrated both
   altered file contents and altered next-request history. The isolated patch
   preserves JSON values; documented Edit defaults and strict bridge transcript
   identity remain unchanged. This fixes a demonstrated rare recomputation
   source, not every ordinary tool-return pause.

New runtime options, all default off:

| Option | Behavior |
|---|---|
| `native_checkpoint_capture` | Native token serialization and read-only frame validation, retaining generation, prefix, transaction, ordering, visibility and history checks. |
| `checkpoint_pinned_prewarm_count` | Preallocate and release pinned payload buffers during startup so the existing PyTorch allocator can reuse them. Limited by checkpoint budget, rounded allocation charge and max sessions in flight. |
| `host_stage_timing` | Bounded asynchronous nested host tracing, without foreground file I/O or extra CUDA synchronization. Diagnostic only. |
| `prefill_request_tokens` | Optional per-request ceiling inside an enabled shared prefill budget. Zero retains the old fair-share behavior; positive values cannot exceed or enlarge the shared budget. Query replay remains atomic and idle prefill is unchanged. |

The pre-existing `prefill_batch_tokens` limiter is tested at 512 and 256. It
reduces ordinary prefill credits while decode is runnable, retains aligned
positive progress, and exempts atomic Query work. This is an existing admission
mechanism being evaluated, not a new scheduler implementation. Recall semantics,
budgets and accepted logical counters are unchanged; different chunking may still
change floating-point execution and natural generation trajectories.

`scripts/patch_deepswe_client_unicode.py` produces a new executable only after
checking the pinned input SHA256 and unique equal-length replacement site. It
does not edit the installed client. The patched executable's SHA256 is
`0bc3800ba6bcf8b9df2daa47509f2159a4c2b74b67922424b4e94e1019c89433`.
Both natural performance arms must use the same qualified client, so a server
comparison is not confounded by changing client normalization in just one arm.

## Correctness and measurement protocol

Frozen component root:
`/data/chaidi/kvmem-vllm-experiments/agent-interference-20261025-r4`.

Qualification completed 16 replies and eight restores. Every target KV byte was
covered by chunked SHA256 comparison; 96 recurrent/model tensors per restore
were compared directly. Both resident terminal-window and independently rebuilt
FP8 history paths were exercised. The audit passed and owned GPU/process cleanup
was empty. Hash verification is described as hash verification, not as a direct
byte-by-byte comparison. The older r3 diagnostic failed before a restore verdict;
it remains preserved and is not counted as passing. Excess diagnostic memory was
suspected, but its historical exit cause was not conclusively established.

The bounded verifier avoids retaining eight complete multi-GB target KV copies.
It has independent reference tests against the original FP8 reconstruction,
partial-tail coverage, corruption detection and recurrent-state mismatch tests.
It is enabled only for qualification, never for performance measurements.

Performance arms are control, native checkpoint capture, capture plus pinned
prewarm, and the latter plus prefill caps 512 or 256. Every arm uses the same eight
saved real token histories (157K–213K), three repetitions of suffixes
32/512/2,048/8,192, seven peers generating 512 tokens and one returning request
generating 64. Inputs and constrained output IDs must match across arms. Every
round explicitly reports whether all seven peers were already decoding when the
eighth request returned. All completed attempts are retained, including negative
performance results. The arms are sequential, not a randomized statistical study.

These constrained component results are not real Agent throughput or quality.
The natural follow-up uses independent DeepSWE workspaces and fixed observation
windows, original prompts, temperature .6, top-p .95, top-k 20, seed 1000,
thinking budget 4096, max output 16384, and original termination guards.
It records accepted output divided by elapsed observation time, tool errors,
prefix misses and peer timing. MTP verification rows are never counted as output.
Unfinished tasks are unscored and snapshotted before cleanup; a snapshot failure
retains the owned wrappers and containers for recovery instead of deleting work.
Snapshot and cleanup time are outside the measured observation window.

Natural trajectories may differ. A single sequential pair supports an observed
comparison and bottleneck diagnosis, not an unqualified causal speedup claim.

## Fixed-input results

Each arm completed 104 responses, including 12 warm eight-request rounds. Every
warm round had all seven peers already decoding before the eighth request
returned. The 104 input hashes and constrained output token lists matched across
arms; the execution audits, evidence hashes and owned-resource cleanup passed.

| Configuration | Warm round wall, seconds | Change | Peer gap, 2K returned input, ms | Peer gap, 8K returned input, ms |
|---|---:|---:|---:|---:|
| Control | 120.492 | — | 666.74 | 763.60 |
| Native checkpoint capture | 116.469 | −3.34% | 651.70 | 745.52 |
| Capture + startup pinned prewarm | 120.923 | +0.36% | 656.87 | 748.88 |
| Above + 512 prefill cap | 126.247 | +4.78% | 369.28 | 386.79 |
| Above + 256 prefill cap | 122.779 | +1.90% | 243.14 | 195.91 |

Each gap column is the **mean of three per-round P95 values**, not a pooled P95.
The measurement covers peer SSE gaps while the returning request waits for its
first output. Component output is constrained and cannot be called real Agent
throughput. The CUDA model interval includes target, sampling, draft and capture
stream work; it is not a pure attention-kernel timer.

Observed mixed-step prefill maxima were respectively 2,048, 512 and 256 for
control and the two capped arms. Mixed model-interval P95 fell from 713.96 ms to
344.14/224.53 ms. This confirms that the cap was exercised, rather than relying
only on configured values. Finer chunks reduce individual stalls but add batch
work. Pinned prewarm did not demonstrate a steady-state benefit in this run and
remains optional; its 10 buffers charge 2.5 GiB of the existing checkpoint budget.

The natural candidate selection rule was recorded before the final component
completed: require all 12 seven-peer overlaps, at least 20% lower average large-
input peer P95 and no more than 5% additional warm wall time; among qualifying
caps choose the lower wall time. Both caps qualified and 256 was selected.

The first natural root failed importing a missing frozen controller dependency,
before any model/controller/answer launch. Its files are preserved. The replacement
`deepswe-interference-natural-20261025-r2` includes an actual standalone import
preflight and startup-recovery audit. Its control finished successfully, including
eight filesystem snapshots and GPU cleanup, but the controller's broad evidence
scan could not read container-owned private configuration files.

The complete evidence scan now runs inside the pinned task image with a read-only
mount of this single completed observation, no network, and only the extra read
capability needed for those files. It never prints contents or changes host file
permissions. All regular files are hashed, symlinks are recorded without following
them, and traversal/read failures fail the audit. The control passed an independent
rescan. This is an archive/controller repair, not a model failure or an answer retry.

Current natural root:
`/data/chaidi/kvmem-vllm-experiments/deepswe-interference-natural-20261025-r3`.
It reuses the completed control through a link and launches only the previously
unattempted candidate. Reuse checks the exact profile, task list, client checksum,
duration, execution audit, cleanup and evidence. Its continuation audit preserves
both earlier failures. No attempted answer was regenerated for either recovery.

## Natural control, completed

The control's window contained 340,178 accepted generated tokens over 900.0694
seconds: 377.9464 token/s, including tool waits and initial task setup. Of these,
330,993 tokens belonged to the 444 requests completed before the boundary;
remaining accepted tokens were still in flight. There were 596 tool results,
11 marked errors, no bridge errors and no exact-prefix misses in 443 continuations.
All 443 observed restores used resident KV. Thus full-window CPU-to-GPU reload
was not the cause of this run's mixed-step stalls.

The 4,249 pure eight-request decode model intervals had median 51.63 ms and P95
54.02 ms. Mixed prefill/decode model intervals had median 217.83 ms and P95
635.80 ms. These compare different batch compositions and do not establish GPU
idle time. All eight tasks were deliberately window-truncated and remain unscored.
The first/middle/last five-minute accepted-output rates were approximately
288.5/406.6/438.1 token/s, demonstrating why an early partial average should not
be compared with a complete window.

## Global-cap natural result and correction

The global-256 candidate completed the same 900-second observation protocol:
278,953 accepted tokens, 309.61 token/s, **18.08% lower** than control. Mixed
model-interval P95 improved from 635.80 to 172.29 ms, but the number of mixed
steps rose from 569 to 2,478. Its first/middle/last five-minute rates were
80.0/389.9/458.5 token/s. It had 377 completed model requests, 517 tool results
with 11 marked errors, zero exact-prefix misses, and no bridge errors. All
unfinished files and responses were preserved. This negative result is retained;
the global cap is not promoted as a throughput optimization.

The fixed-input experiment had only one returning prefill alongside seven
decoders. Natural tasks often have several returning prefills. Dividing a shared
256-token pool among those contenders produces 32–64-token slices. That repeats
fixed preparation/model work and delays rebuilding the full decoding batch.
The per-request extension keeps 2,048 total credits while capping each ordinary
prefill at 256, so seven concurrent prefills can receive 256 each. It still never
enlarges the native batch budget or changes logical counters, ordering, selection,
Query atomicity or MTP tail reservations. With no ready decode it is disabled.

The corrected candidate also disables startup pinned prewarm, whose steady-state
benefit was not established. It keeps native checkpoint capture. Frozen source is
`/data/chaidi/kvmem-vllm-experiments/agent-interference-per-request-20261025-r1`.
Its scheduler is derived from the already-qualified source with just the new
constructor argument: unrelated workspace scheduler changes are excluded.
105 tests passed against the frozen implementation, including the new fair-share,
default-off, option validation and atomic-Query cases. A first test invocation had
fixture-only failures (an undersized synthetic repository and examples omitted
from the frozen package); corrected fixtures then passed, without a model change.
The corrected candidate passed GPU qualification: 16 responses, eight restores,
all target KV bytes covered by SHA256 comparison, recurrent/model checks and
empty owned-process/GPU cleanup. Its new natural window completed at
`/data/chaidi/kvmem-vllm-experiments/deepswe-interference-per-request-20261025-r1`.
Old control and global-cap answers are reused and will not be rerun. Qualification
establishes correctness for the exercised cases, not a throughput improvement.

The observed 377.95 token/s control is the throughput reference to preserve.
Further tuning should reduce peer stalls without materially lowering aggregate
accepted output. The global-256 result illustrates why a lower mixed-step P95
alone is insufficient: mixed model intervals became shorter but their cumulative
duration rose from about 155.94 to 366.52 seconds within the observation. These
are measured model intervals, not a sum of request restore waits or an estimate
of GPU idle time. Completed-request TTFT P95 also worsened from 2.82 to 19.47
seconds. Natural execution trajectories differ, so this is workload evidence for
rejecting that configuration, not an isolated causal estimate for each change.

## Local evidence

`artifacts/agent-interference-optimization-20261025-r1` contains the pre-edit
source copies, native build/test logs, CPU measurements, real-client qualification,
host profile analysis and component analyses. The live/frozen runtime is never
edited in place. The finite queue refuses previously attempted arms and gates
performance tests on qualification audit, evidence hashes and GPU cleanup.

## Final selection and fresh64-task suite

The per-request256 cap produced327,568 accepted tokens over900.1517 seconds,
or363.9031 token/s, versus377.9464 for control (-3.72%). Its eight unfinished
workspaces were preserved; execution/evidence and GPU cleanup audits passed.
The shared256 cap produced309.6123 token/s. These are sequential natural
Agent windows with different trajectories, not isolated causal estimates.
The user explicitly requested that prefill throttling be disabled.

The formal suite therefore uses the qualified r4/capture runtime, without
shared/per-request prefill caps or adaptive prefill, and without pinned prewarm.
Native checkpoint capture stays enabled; its isolated real-Agent gain remains
unproven. The qualified isolated client JSON fix is retained. Original16 tasks
run afresh for each seed1001/1002/1003/1004, four sequential cohorts, maximum8
concurrent agents,64 total. All use96K/16K,10M logical limit, LRU,50:50 no-graph
recall and original sampling/verification/termination rules. Prior results are
kept outside this fresh cohort. A requested1800-second local monitor observes
progress without changing experiments and wakes this existing chat only on
actionable issues, recovery or completion. No stopped old timer is re-enabled.
