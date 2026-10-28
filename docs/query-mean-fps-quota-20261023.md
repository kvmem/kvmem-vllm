# MeanK / FPS4 50:50 budget quota experiment

The optional Query recall mode is `query_mean_quota_fraction=0.5`, with
`query_score_mode=key_fps_4`, `query_fps_halo_blocks=1`, graph alpha zero and
guided Query disabled. The default fraction remains zero.

MeanK and FPS4 each compute their own scores over the original full native
score mask. Scores are not blended. Original fixed content is admitted first;
half of the remaining block slots (rounded down) protect the native MeanK
ranking. FPS4 anchor/left/right logical groups then compete for remaining
slots, charging only newly added original blocks. Groups which cannot fit are
skipped atomically. Negative-priority MeanK singleton groups fill stranded
slots after every FPS group has been considered. One final native group
selection binds assembly to the exact preview. No post-picker expansion occurs.

For K=36864, sink=1312, recent=0, this leaves 555 MeanK and 556 FPS4 block
slots. Actual fixed pins always determine the quota; a partial block still
costs one 32-token slot. Prefill pressure keeps cumulative attention.

## Frozen model experiments

Data directory: `/data/chaidi/kvmem-vllm-experiments/ruler-hybrid-quota-20261023-r1`.
Repository entry: `artifacts/ruler-hybrid-quota-20261023-r1`.

| Stage | Scope | Status at launch |
|---|---|---|
| RULER qualification | Fixed first admission wave, 8 original 256K cases | Completed; 8/8 correct, final execution audit and GPU cleanup passed |
| RULER | v1, all 13 tasks, original indices 0–9, 130 cases | Running |
| LongMemEval-S | Original 102 cases and original sampling | Prepared; queued after RULER |
| MAB qualification | Two consecutive questions of the longest reusable prefix, plus the absolute longest 757K question | Prepared; 3 original cases, full-score native replay |
| MemoryAgentBench | Original 1116 questions / 29 prefix groups | Prepared; gated on successful qualification and identical frozen runtime |
| Grading | Existing frozen official LME/MAB rubric and Flash transport | Queued after inference |

All formal answers use new output directories. Prior answers and scores are
retained. RULER qualification and the 130-case campaign have identical source
proofs, original seeds and sampling. The eight qualification answers are a
startup check, not a complete RULER result. The first-ten cohort has been used
in earlier diagnostics and is not an independent held-out validation set.

RULER retains the previous CPU-FPS campaign geometry: NVFP4, FP8 KV, MTP4,
K36864/G16384, at most eight owners, batch16424, chunk2048, CPU history100GB,
no SSD. Memory suites retain their own historical frozen profiles, including
chunk1024 and MAB frozen-P ownership. Thus memory and RULER wall times must
not be presented as measurements with identical prefill geometry.

Each root has immutable `profile.json`, `fixture.json`, `prepared-proof.json`
and `runtime-proof.json`. The finite queue uses `queue-plan.json`,
`queue-process.json`, `queue.log` and root `status.json`. It waits for each
controller's final audit and GPU cleanup, fails on technical errors, and never
retries answers based on scores. It creates no timer and restores no service.

## Longer memory histories

Histories exceeding the existing 512MiB FPS scratch estimate use Key tiles,
with per-block representative max followed by one softmax over the entire
original block domain. Smaller histories retain their original scoring path.
CPU tests compare tiled/non-tiled probabilities and ranks, GQA, missing
representatives, masks, Query FP16 conversion and scratch exhaustion.

In hybrid mode, rewinding a frozen prefix invalidates queued work and cache
epochs while retaining unchanged prefix representatives. Changed suffix
signatures are invalidated; signature/generation checks remain mandatory.
The rollback test covers a late old-branch worker and a new suffix.

LongMemEval and MAB qualification export full vectors for independent native
picker replay. Full MAB exports compact per-case quota/selected-range/CPU and
pressure checks; it does not claim full-vector replay of all 1116 questions.
This keeps shared diagnostic storage bounded. Full MAB admission is gated on
the longest-prefix test actually retaining its representative cache.

Component checks passed before launch: 158 tests covering options, native
selection, quota replay, group boundaries, rollback, CPU prebuild, bounded
tiling and campaign contracts. Subsequent memory-scope checks passed 17 tests.

## Results and audit-only recovery on 2026-10-24

The complete 130-case RULER campaign finished with **93.1923076923077**,
with independent native selection replay, official rescoring, accepted input
attention isolation and GPU cleanup all passing. Its total controller time was
6143.71 seconds. This is not a pure prefill timing measurement.

The comparison was rescored from immutable answers using exactly the same
original task indices 0–9, input hashes, seeds and sampling:

| Method | 13-task macro | CWE | QA1 | UUID multikey3 |
|---|---:|---:|---:|---:|
| Native full context | 96.85 | 99 | 80 | 100 |
| MeanK / FPS4 quota half, this run | 93.19 | 84 | 70 | 80 |
| CPU FPS4 + halo1 | 90.13 | 60 | 60 | 90 |
| Historical MeanK + sum graph | 87.15 | 93 | 80 | 20 |

The gain over FPS4/halo1 is 3.06 points; performance and all-category monotonic
improvement are not implied. Historical runtime/batching differ, and the sum
reference also used different prefill coalescing. These are matched question
quality results, not controlled throughput comparisons. The 130-case reference
scores must not be confused with previous 260-case scores.

The original LME attempt stopped in its client audit. Its first eight requests
all produced complete saved responses, including two already validated results.
The audit incorrectly required every historical selection budget to equal K.
The unchanged native scheduler reserves unaligned public input and speculative
headroom inside K; six of these requests legitimately selected under 36736 or
36768 tokens. Both worker and graph quota reports agreed. The fixed audit
derives the exact scoped budget from original prompt/output/replay bounds,
and retains independent full-vector picker replay with that budget.

Recovery root:
`artifacts/ruler-hybrid-quota-20261024-r2` →
`/data/chaidi/kvmem-vllm-experiments/ruler-hybrid-quota-20261024-r2`.

The eight saved responses passed corrected offline replay. They were copied
with source hashes into a new frozen recovery campaign; original successful
result bytes, failed results, requests, responses and captures remain unchanged.
Engine/scoring source hashes and the profile are identical. Only the remaining
94 LME inputs are dispatched. The new finite queue then runs the three MAB
qualification questions, the original 1116-question MAB subset, and Flash
grading. MAB qualification now also binds its explicit frozen-prefix replay
span to the original request. The old failed attempt and queue are retained.

`lme/recovery-proof.json` records the eight inherited cases, six corrected
audits, two unchanged successful results and zero regenerated answers. Per-case
recovery evidence and timing provenance distinguish saved inference from new
inference. `ruler-comparison/comparison.json` contains the rescored references
and immutable source evidence bindings. No timer or service restoration was
started.

## Requested partial LME grading

The first grading snapshot froze all **61** cases completed at that instant,
without selecting by answer or score. Flash returned **57/61 correct (93.44%)**:
single-session user 17/17, single-session assistant 17/17, preference 14/17,
multi-session 5/6, temporal reasoning 2/2, knowledge update 2/2. This completion
subset is unbalanced and is not the full 102-case result.

Evidence is in `ruler-hybrid-quota-20261024-r2/grading-lme-partial-r1/`:
the report, source/snapshot hashes, fixed cohort plan, calibration, individual
verdicts and original Flash API responses are retained. The original official
judge prompt and parameters are unchanged. Inference continued throughout.

The grader now supports a declared partial denominator and a single arm
without fabricating a baseline comparison. The finite queue's grading tools
were versioned separately to use that fix and reuse exact request-bound Flash
responses in the later full LME grading. `queue-plan-grading-update.json` and
`queue-grading-transition.json` record this change. Only the queue process was
replaced; the active LME controller, model processes, frozen inference/scoring
sources and sampling were untouched. No inference request was retried.

## Completed full LME grading

All 102 LME cases completed without runtime errors or output-cap hits; the
full-vector native picker audit and GPU cleanup passed. Flash graded **95/102
correct (93.14%)**, with each of the six categories containing 17 questions:
user facts 17/17, assistant facts 17/17, preferences 14/17, multi-session 14/17,
temporal reasoning 16/17, knowledge update 17/17.

The full report and immutable answer/judge bindings are in
`ruler-hybrid-quota-20261024-r2/grading-lme/`. Exact partial-grading API requests
and responses were reused; the earlier 57/61 partial report remains intact.
Full scoring ran on the CPU/API while the MAB qualification model continued.

The finite queue now uses `queue-plan-early-lme-grade.json`. It waits for the
already-running MAB qualification, then runs the original 1116-question subset.
The separately completed LME grade is bound by its source-answer, verdict and
inference-audit hashes and verified instead of regenerated at queue completion.
`queue-early-lme-transition.json` proves that only the queue process changed;
the MAB qualification controller and its frozen model/runtime remained intact.
