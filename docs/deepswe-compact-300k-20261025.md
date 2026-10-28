# DeepSWE 300K CPU index compaction and tool handoff analysis

The user requested a new first16 experiment with eight concurrent agents and a
307200-token logical limit, after two storage-only optimizations. The previous
512K attempt stopped at the existing 8 GiB host-memory guard. Its answers,
workspaces, logs and failure remain preserved; the new configuration is a
separate cohort and does not resume or replace those answers.

## Storage changes

- `query_fps_cache_fp8=true` stores the four selected, original FP8 Key values
  in FP8 instead of their lossless FP16 expansion. Selection still uses the
  previous arithmetic. The same FP32 scoring and block pooling remain, with
  the previous scoring tile geometry. This halves the representative numeric
  payload; offsets, metadata and allocation slack are additional.
- `cpu_release_sealed_sums=true` releases a complete block's FP32 accumulator
  only after its immutable MeanK exists and archive submission succeeds.
  Incomplete tails retain their sums. Rewind reconstructs any new partial tail
  from the original archived Key bytes. Original K/V and MeanK are preserved.
- Both settings default to false and require the qualified CPU archive/native
  MeanK path without historical GC. Compact FPS storage also requires original
  FP8 Keys. GPU caches, KV budgets, recall and sampling are unchanged.

At eight full 307200-token histories, the raw FP8 K/V payload is 79.6875 GiB.
MeanK is about 2.34375 GiB and compact FPS4 plus its charged row metadata about
4.85 GiB before allocation slack. These are payload estimates, not process RSS
or a guarantee of capacity. Checkpoints, transfer buffers, Python/runtime,
agent containers and the OS also consume memory. The 8 GiB guard stays enabled.

## Why peers slow down near tool calls

The preserved 512K attempt's formal interval contains 22825 model steps over
1639.71 seconds. Qualification steps were excluded. Full statistics and source
hashes are in `artifacts/deepswe-tool-handoff-analysis-20261025-r1`.

| Measurement | Result |
| --- | ---: |
| Pure decode model step, median | 37.72 ms |
| Mixed prefill/decode model step, median | 175.05 ms |
| Mixed prefill/decode model step, P95 | 420.61 ms |
| Shared foreground maintenance, median / P95 / max | 2.06 / 75.67 / 521.90 ms |
| Restore active-KV resident hits | 1237 / 1239 |
| Restore model checkpoint cache hits | 1238 / 1239 |
| Checkpoint host capture, median / P95 / max | 6.80 / 16.05 / 353.74 ms |

Tool results introduce new input. When this input is prefilling in a shared
batch, the other requests cannot take their next decode step until that batch
finishes. Batch geometry varies, so the observed 4.6x median step ratio is
descriptive, not a controlled causal throughput estimate.

Request completion and resumption also enter shared worker maintenance before
the next public forward: session receipt, archive publication, checkpoint
capture and restore preparation. One slow owner's synchronous portion can
delay the whole worker. A recorded 583.63 ms gap coincided with 521.90 ms of
maintenance and a terminal checkpoint taking 353.74 ms, of which 351.71 ms was
canonicalization/submission. The nested intervals must not be added together.
The precise cause of that unusually slow submission is not yet isolated.

There are also 1.18–1.48 second gaps without a fully explaining captured stage;
these remain unresolved scheduler/host intervals. Gaps between public kernels
are not measurements of GPU idleness: copies and private work can occupy them.
Restore duration is per-request elapsed time and can overlap other requests.

Resident KV is already reused in 99.84% of these restores. Repeated whole-window
K/V uploads therefore do not explain the usual slowdown in this run. All 106
formal refresh selections were pressure events; there is no formal private
Query-stage evidence supporting a full FPS4 scan on every tool return. Bridge
preparation already runs off the async event loop; any CPU contention there
requires separate attribution. The external tool process itself does not
require all other GPU requests to wait.

## Experimental controls and validation

The new queue freezes the prior runtime and changes only authority,
capture-runtime option forwarding, runtime options and FPS cache/scoring
storage handling. Original tasks/images/verifiers, protocol, seed, sampling,
thinking budget, tool guards, output caps, K36864/G16384, 50:50/no graph,
neighbors, pressure attention, MTP4, chunk2048/batch16424 and memory guard are
bound to the prior manifest. Only the logical cap and the two storage flags
change in the model profile. No SSD and no recurring monitor are introduced.

252 regression tests passed, followed by the final compact-index/controller
set (31 passing, overlapping the regression set). GPU component tests gate the
experiment and compare FP8 versus FP16 representatives' scores bitwise,
including actual 256-dimensional heads, dense/tiled scoring and ranking.
The existing multi-request model preflight then gates task admission.

The GPU component suite passed all 18 checks, including the four explicit CUDA
comparisons. The active frozen experiment is
`/data/chaidi/kvmem-vllm-experiments/deepswe-hybrid-compact-300k-20261025-r3`.
Preparation-only r1/r2 directories contain no model requests or answers and
remain preserved with their errors. Their private-file hash issue was resolved
by reading as the original container file owner in a fixed-image, read-only,
networkless helper. The live dashboard retains port50125 and now reads r3.
Atomic Query concurrency8, short-tail token equality, eight full 16384-token
generation continuations and session cleanup preflights all passed. The first
eight formal tasks are running. Formal quality and peak-memory results remain
pending; startup proof is `experiment-start.json` in the analysis artifact.

Future latency work should separately evaluate smaller/adaptive prefill chunks
when peers are decoding and move or bound long shared maintenance work. Those
changes are deliberately not part of this memory-capacity experiment.

## 300K run incident and controller correction

The r3 run stopped on 2026-10-25 at 12:22 China time. At 12:21:18 the wasmi
CLI had emitted a structured local `blocking_limit` / `Prompt is too long`
result. It never sent that rejected request to the bridge. The controller only
recognized bridge-reported context limits, misclassified this exit as an
infrastructure error and stopped the seven other active tasks. Eight tasks
were still unstarted. Minimum sampled host headroom was 26.05 GiB, above the
unchanged 8 GiB guard; this incident is not a repeat of the 512K memory stop.

The controller now recognizes the final complete CLI blocking-limit record,
requiring the current model/window, agent exit 1 and no bridge errors. It
records an unscored context ceiling, releases that owner's history and admits
the next task without stopping peers. Real service/transport and host/history
failures retain their original stopping policy. Ninety lifecycle and queue
tests passed. Replaying all eight existing logs identifies only wasmi.

The frozen r3 runtime and original outputs are unchanged. No model retry was
launched. The seven interrupted agent containers are gone and have no exported
final worktrees or patches; transcripts alone do not prove exact workspace
restoration. The partial verifier output for wasmi is retained but is not a
normal completed-task score. Full incident evidence, classifications and source
hashes are under `artifacts/deepswe-300k-context-incident-20261025-r1`.
