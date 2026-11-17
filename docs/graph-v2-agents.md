# V2 natural fixed8 agent pairs

`scripts/benchmark_graph_v2_agents.py` prepares a separate V2 D experiment. It
does not edit or relax the historical FP8 benchmark gates. Preparation and
validation are CPU-only. GPU runs use the frozen entry point, a fresh service
for each job, and the same global device lock as the existing benchmarks.

`--candidate` accepts `raw-k-cache` (default, 0 versus 4 GiB **total**, split
between two async arenas), `cpu-direct-record-seal` (false versus true), or
`async-action-metadata` (false versus true). Each pair changes exactly that one
setting. The boolean pairs retain raw-K at zero; `native_action_metadata`
always remains the baseline value. All profiles use NVFP4/MTP4/FP8 KV, V2 D
alpha 0.2, K36864/G16384, C8, chunk1024 and batch16424. Four GiB of P/M
checkpoints retain eight sessions; checkpoint deferrals fail the run.

The original fixed8 fixture is copied byte-for-byte, including initial contexts,
the early absolute Query coordinates, mandatory spans, tool tokens and waits.
Each owner appends its own complete accepted output. Natural generation retains
the 16384-token cap and existing sampling. Two complete warmup turns per owner
precede the shared 300-second window. Epochs end normally after 16 turns or when
the next complete input plus output allowance would exceed 262144 tokens. No
pressure-driven truncation, recomputation or forced answer lengths are added.

The original fixture uses `auto` with an early Query inside K. Its initial
prefix fits intact and the observed trajectory can contain only graph pressure,
with no Mean-K Query selection event. This is reported as unexercised Query
propagation, not fabricated coverage. Initial epochs without a scored selection
must have an explicitly early auto Query and actual pressure windows whose hard
mandatory blocks and selected ranges protect that Query. Later turns must still
prove restoration of the original Query-containing history. The independent
structured and NIAH tests qualify scored Query propagation.

From the repository root, prepare one screening pair:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B \
  scripts/benchmark_graph_v2_agents.py prepare \
  --candidate raw-k-cache --pairs 1 \
  --output /absolute/path/to/new-agent-screen
```

The default model profile is
`artifacts/graph-overhead-20261018/gpu-r2/after/profile.json`. The default
qualification evidence is `artifacts/graph-v2-sessions-20261018/sync-r1` and
`async-r1`; explicit replacements use `--sync-qualification` and
`--async-qualification`. The original fixture is
`artifacts/calibrated-fp8-agents-20261018-candidate-r3/fixture.json`.

Validate the frozen experiment without using the GPU:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/absolute/path/to/new-agent-screen/frozen/src \
  .venv/bin/python -B \
  /absolute/path/to/new-agent-screen/frozen/scripts/benchmark_graph_v2_agents.py \
  validate --output /absolute/path/to/new-agent-screen
```

When the GPU window is authorized and idle, use the same frozen command with
`run` in place of `validate` and omit the empty `CUDA_VISIBLE_DEVICES`. If the
screening result warrants repetition, prepare a **new** directory with
`--pairs 3`. Order is fixed AB, BA, AB. Started, failed, interrupted and
completed campaigns cannot be resumed or silently repeated.

The freeze includes source/native extension, launcher helpers, exact profiles,
expected worker payloads, fixture hashes, model/config/tokenizer identities,
and copied sync/async qualification evidence. The checklist calls these prior
two-owner byte-restore/release tests, not qualification of the new eight-owner
profile or enabled candidate. Missing new boolean fields normalize to false.
Only the explicitly recorded default-off sealing/config/telemetry source
extensions may differ from prior lifecycle evidence; other runtime changes
require fresh qualification. Each new run validates actual launch and cache
geometry, complete terminal history, zero checkpoint deferral, no preemption,
budget bounds and final release. Failure preserves evidence and stops later jobs.

`v2-evidence.json` contains per-request input/output/selection hashes, retained
Query coordinates, actual native batch and pure-decode batch counts, accepted
output TPS and offered external input rates, H2D/restore/gather measurements,
seal/metadata counters and NVML samples. Warmup, cleanup and drain outputs are
excluded from fixed-window TPS. Seal/metadata totals are explicitly cumulative
over the fresh worker's entire lifetime; timestamped archive snapshots are also
retained. Missing counters remain unobserved, not zero. Overlapping host work
is never classified as GPU idle, and proposed speculative tokens are not
counted as accepted output or pure decode TPS.

Eight owner loops do not imply every GPU step has batch eight. The report keeps
HTTP overlap, measured native peaks and `actual_c8_compute_observed` separate;
a peak below eight does not trigger answer retries. Comparisons of common
logical turns preserve all input/output/selection drift. Unexercised candidates
or drift produce `inconclusive`, with no usable speedup estimate. Missing actual
C8 computation or complete latency measurements also makes comparison
inconclusive. Request, observed TTFT and accepted-token-event gap p50/p95 are
retained with censoring; an MTP event may contain several accepted tokens.
Three comparable pairs report the median ratio and whether it reaches 1.05,
plus every pair's p95 latency ratios. Consistent material tail regression still
requires review. `merge_gain_qualified` and `performance_gain_established`
remain false; these reports make no statistical-significance or agent
task-quality claim.
