# Age-adjusted pressure and the shared shallow selection

Two per-layer selection strategies that together form the **combined** method. Both are
opt-in; the defaults run the built-in per-layer path unchanged.

| Part | Setting | Where it acts |
| --- | --- | --- |
| Age-adjusted prefill pressure | `kvmem.strategy_pressure_scorer = "lift"`, with `kvmem.prefill_attention_half_life = 32768` | which blocks each layer keeps while the prompt is read |
| Shared shallow selection | `KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS=4` (a `strategy` setting) | which blocks the first layers retrieve at the final Query |

Both need the per-layer path: `--layerwise-query-diagnostic sequential` with its probe
(not `KVMEM_LAYERWISE_QUERY_NO_PROBE`), and for the pressure scorer also
`--layerwise-prefill-pressure` with native buffered histories and native score export (both
default-on). The launcher refuses other combinations
(`vllm_kvmem.serving.capabilities.check`).

## 1. Problem

Per-layer prefill pressure keeps each full-attention layer's window at K tokens while the
prompt is read, ranking blocks by

    score(b) = decayed attention received / decayed time visible

Every block gets a burst of attention while it is new and close to the text being read, so
new blocks always win, and a dropped block is never scored again. On LongMemEval-S at
K 8,192 the windows were 87% the same as "keep the most recent tokens" at layer 3; about 97%
of window blocks lay in the last 16K tokens; answer evidence in the window just before the
Query was 10–18% per layer. The half-life barely matters for this score: decay scales its
numerator and denominator alike.

A full-context capture (16 questions, every per-layer prefill attention packet) showed that
later text does look back at the answer evidence: from 4K tokens or more away, evidence blocks
get 1.3–3.2× the attention of a typical block at the same distance. The score above buries
that signal under the local burst.

## 2. Age-adjusted score (`lift`)

For each layer of each request:

1. **Distance profile.** Every accepted packet's sampled rows update 14 log2 distance buckets
   (from the row back to a block's end: [0,32), [32,64), …, [65536,131072), ≥131072): summed
   attention and the number of (row, visible block) pairs. The profile is their mean, the
   attention a block typically gets at that distance. The packet being accepted is included
   before it is used.
2. **Expected attention.** For each sampled row, each visible block's expected attention is the
   profile mean for its distance, scaled so the row's expectations sum to 1 like a row of
   attention. It is accumulated exactly like the real attention (× 1/rows, decayed at every
   packet).
3. **Score:** `lift(b) = decayed attention received / decayed expected attention`. About 1 for
   an ordinary block, including a new block in its local burst; above 1 for a block that later
   text keeps looking back at. Blocks no row has scored yet rank first (max + 1). With no
   positive score the layer falls back to the score of section 1.
4. **Half-life 32,768 tokens** (8,192 by default): once the local burst is normalised away, a
   slower decay keeps long-range references longer.

Sink, mandatory blocks, the picker and the budget are unchanged. Rolling decode pressure keeps
the section-1 score. There are no thresholds or calibration data; the profile comes from the
request's own rows.

### Implementation

- `csrc/buffered_prefill_state.cpp`: `State.lift_empty()` makes a lift state, which also keeps
  one expected-attention double per key and the 14-bucket profile. `committed` /
  `committed_batch` take `history_shift = end − count − local_begin`, which maps a row's window
  position to its history position (the window is compacted after evictions). A lift commit
  reads every value once in a validation pass that also builds the row's bucket counts, then
  accumulates mass, exposure and expected attention; ordinary states run the original loop.
  `scores()` / `scores_for()` return mass / expected for lift states, so the fused native pick,
  the batch selection and the plain path all rank by it without Python per block. Mass and
  exposure are bit-identical to an ordinary state's.
- `State.with_values(values)`: historical GC prunes a layer history by assigning its values;
  a lift state keeps its profile and the remaining keys' expected attention, and refuses new
  keys or changed values.
- `LayerwisePrefillPressure.new_history` creates every layer history of a lift request as a
  lift state. Session checkpoints share the native state (`KVMEM_CHECKPOINT_FAST_STATE`,
  default-on), so restores keep it; the dictionary restore path cannot carry it and refuses.
- Deferred folds carry the shift with each queued span. Pressure decisions are labelled
  `lift_per_layer` in `capture/layerwise_query.jsonl`.

## 3. Shared shallow selection

At the final per-layer Query, the first N full-attention layers (model layers 3, 7, 11, 15 for
N = 4 on Qwen3.8-27B) take the shared global MeanK selection: the native frame chosen from
every layer's Query before the per-layer pass, with its mandatory, sink and partial blocks and
the same token count. They skip their own scoring; the later layers keep their per-layer MeanK
selection. Shallow layers retrieve poorly on their own: per-layer evidence coverage at
K 36,864 was lowest at model layer 3 (94%). Their decisions are labelled `query_scorer: global_selection`.

## 4. Results

LongMemEval-S, 500 questions; DeepSeek judge (`deepseek-flash`, official prompts, temperature 0).
Native full context (256K): 93.6.

| Select budget K / G | per-layer | global | shared shallow only | age-adjusted only | combined |
| --- | --- | --- | --- | --- | --- |
| 8,192 / 4,096 | 81.0 | 77.4 | 82.4 | 83.8 | **85.6** (p = 0.01 vs per-layer) |
| 16,384 / 8,192 | 86.6 | | | | **90.8** (p = 0.005) |
| 32,768 / 16,384 | (92.4 at K 36,864) | (92.0 at K 36,864) | | | **93.2** (vs native p = 0.85) |

The largest gains are on knowledge-update questions (+10 to +15 points), which need an old and
an updated fact far apart; multi-session +5 to +7.5; 2 of 30 preference questions lost. RULER-130
(256K, K 36,864): combined 94.46, per-layer 94.82 on the same runtime; 6 of 130 cases differ
(4 won, 2 lost on qa_1, which samples at temperature 1.0).

Efficiency, same runtime and autotune cache for every arm: no measurable cost.

| Benchmark | per-layer | combined |
| --- | --- | --- |
| MAB fixture, one request: prompt tok/s to first token | 4,849 / 4,893 | 4,889 / 4,902 |
| Agent fixture, one session: turn decode tok/s | 177.6 / 178.3 | 179.2 / 178.7 |
| Agent fixture, four sessions: turn decode tok/s | 142.8 | 143.5 |
| 10 LongMemEval questions at once (K 8,192): decode step at 10 decoders | 32.9 / 32.8 ms | 32.7 / 32.6 ms |
| RULER-130, 8 concurrent: input tok/s | 4,942 | 5,082 |

A lift refresh reads more distinct archive blocks (the layers keep different old blocks:
379 → 539 per refresh at K 8,192), which cost the combined method +1.3% engine step time on
LongMemEval at K 8,192; the shared selection's cheaper final Query (scoring −21 to −27%)
outweighs it at larger budgets (K 16,384: −1.6%; RULER at K 36,864: +2.8% input tok/s).

## 5. Limits

- Per-layer path only; global (single-selection) pressure is unchanged.
- At K 8,192 the age-adjusted score keeps 38–41% of the newest 1K tokens (the section-1 score
  keeps nearly all). No benchmark showed a cost; a recent band would remove it but is untested.
- Session restores of lift states need `KVMEM_CHECKPOINT_FAST_STATE=1` (the default).
