# MAB long-history FPS scoring efficiency

The original hybrid MAB attempt was stopped at the user's request after **399
completed questions**. Four other attempts had no complete saved response. All
original answers, attempts, logs and the frozen runtime remain under
`artifacts/ruler-hybrid-quota-20261024-r2/mab`. Its controller completed GPU
cleanup. The replacement is `artifacts/ruler-hybrid-quota-20261024-r3`.

## Measured bottleneck

Across the 395 completed prefix-reuse questions, HTTP time was 35.27 s median,
TTFT 34.70 s, and generation after the first token 0.074 s. Waiting for the
single selection/assembly lane took 21.88 s median, and selection took 9.57 s,
including FPS scoring at 8.15 s and independent MeanK scoring at 0.21 s.
These are wall intervals, and separate medians must not be added. Every warm
question had zero rebuilt representative blocks. CPU construction was keeping
up. The 100 GB history allowance and existing client headroom admitted at most
four long-prefix owners despite the configured eight-client maximum.

The long-history tiled scorer copied each layer's representative Keys from CPU
to GPU again for every eight Query rows. The selection/assembly lane correctly
serialized its shared scratch, but this amplified all requests' wait times.
Baseline evidence and result hashes:
`artifacts/mab-fps-layer-cache-baseline-20261024.json`.

## Change and component evidence

With the existing `query_fps_optimized=true` setting, keep one layer's compact
FP16 representatives on the GPU while its Query chunks are scored. Admission
includes the layer and the original tile's conservative scratch estimate in
the existing 512 MiB bound. Fall back to the original transfer path if it does
not fit. The original tile shapes, FP32 dot products, representative max,
full-domain block softmax, masks and accumulation order are unchanged.
`query_fps_optimized=false` retains the old path. No persistent GPU cache,
additional worker, precision change, budget change or scoring approximation
was introduced. Per-request statistics now record the tiled path, admitted
layers, logical Key upload bytes and scratch bound.

Fixed synthetic FP8-representable Keys and BF16 Q cover 16 target layers,
24 Query heads, four KV heads and head dimension 256. Two interleaved measured
trials per implementation followed one warmup. All score vectors were
**bitwise equal** to the original frozen runtime.

| History blocks / Query rows | Original median | Layer cache median | Speedup |
|---|---:|---:|---:|
| 12,864 / 264 | 4.465 s | 1.062 s | 4.20× |
| 23,584 / 272 | 8.324 s | 1.966 s | 4.23× |

The largest observed temporary CUDA allocation was 245,654,528 bytes (234.27
MiB); its conservative bound was 317,818,112 bytes, below 512 MiB. These are
scorer components including transfers and synchronization, not model throughput.
Report: `artifacts/mab-fps-layer-cache-20261024-r1.json`.

Validation: 15 CUDA component tests passed (including exact tiled probabilities,
GQA, partial representative slots, masks and 1/17/264 Query rows); the focused
CPU integration/recovery suite passed 127 tests with eight CUDA opt-in skips.

## Frozen qualification and continuation

The new qualification repeats the original three fixed qualification requests:
the longest reusable prefix's first two questions and the longest original
question. Qualification answers are separate and never replace formal answers.
It retains native full-vector selection replay and cleanup checks.

`recover_hybrid_mab.py` allows only `query_key_fps.py` to change among inference
sources, bound to the component report. It copies completed formal cases byte
for byte, including unsuccessful answers; answer scores never determine reuse.
An uncommitted complete response would require explicit audit recovery rather
than regeneration. Only unanswered questions enter a fresh frozen campaign.
The original profile, fixture, input hashes, seeds, sampling, output limits,
50:50 optional quota, neighbor grouping and cumulative-attention pressure
policy are retained. Initial work to restore the remaining groups' prefixes
is reported as cold preparation, separately from subsequent warm questions.

All three real-model qualification requests completed. Full-vector native
selection replay, prefix cache reuse, original input/sampling validation and
GPU cleanup passed. Matched qualification observations:

| Request | Old HTTP | New HTTP | Old FPS | New FPS |
|---|---:|---:|---:|---:|
| Reusable 754,688-token prefix, first question | 382.15 s | 397.95 s | 7.98 s | 1.64 s |
| Same prefix, second question | 14.10 s | 7.39 s | 8.29 s | 1.98 s |
| Longest standalone question | 481.57 s | 403.82 s | 65.02 s | 14.49 s |

The warm request was 47.58% shorter and retained the same 15-token answer.
The first cold request retained its 12-token answer but did not improve in
total time. The standalone generation changed from 1,116 to 1,142 tokens and
its answer text differed, so its complete latency difference is not solely
attributable to scoring. These are single matched before/after qualification
observations, not an interleaved model-throughput benchmark. Fixed-input score
equality is established separately by the component trials. Evidence:
`artifacts/mab-fps-layer-cache-live-20261024.json`.

The replacement formal campaign is frozen at
`ruler-hybrid-quota-20261024-r3/mab`. Its recovery proof binds **399 inherited
answers**, all original per-case files and four interrupted attempts; only
the remaining **717** questions are dispatched. The original full profile is
byte-compatible with the saved client manifest. A finite queue starts this
campaign after qualification and retains the requested Flash grading step.
The earlier completed RULER/LME results are linked for reference and are not
rerun. No periodic monitor, DSH service or A800 job was started.
