# Cold graph collection overhead pair

`scripts/benchmark_graph_collection.py` defines an independent `off` / `collect`
pair on the original eight 256K NIAH inputs. The profile difference is exactly
`kvmem.graph_mode`; the complete profile comparison does not discard other
`graph_*` fields. Both sides keep aligned native MeanK, output/prefill attention
collection, **attention pressure**, V2 configuration, alpha 0.2, K36864/G16384,
FP8 KV, NVFP4/MTP4, C8, chunk1024 and batch16424. Raw-K cache is zero; direct
record sealing, asynchronous/native Action metadata and Action profiling are
false. Normal asynchronous sealing remains enabled on both sides.

`off` does not disable attention capture. It disables graph state and its
acceptance/parser/snapshot/report work. This pair therefore measures the
additional graph collection path, including ordinary collection diagnostics.
It does not measure attention capture versus no attention capture.

Prepare a fresh artifact during an available CPU window:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B \
  scripts/benchmark_graph_collection.py prepare \
  --output /absolute/path/to/new-collection-pair
```

Use the frozen controller for validation and execution:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B \
  /absolute/path/to/new-collection-pair/frozen/benchmark_graph_collection.py \
  validate --output /absolute/path/to/new-collection-pair
```

During an authorized GPU window, replace `validate` with `run` and omit the
empty `CUDA_VISIBLE_DEVICES`. Order is off then collect, with a fresh service
for each. The controller retains original first answers and failures and
refuses to rerun a started campaign. Existing frozen A timing is not reused.

The freeze binds both profiles, original sample and scorer hashes, the same
Python/native KVMem runtime and helpers, model/tokenizer identities, installed
vLLM version and cache-layout source, expected actual launch payloads and
interpreter. The qualifier checks effective launch/RequestPlan settings,
native cache geometry, observed MeanK backend and original Query coordinates,
actual Query/pressure budgets and selected ranges, independent accepted-token
attention sample oracles, complete history and owner release.

The off audit is separate from A/C/D's graph audit. Off must have no graph
events, terminal graph state or allocated graph budget. Its bounded-refresh
reports currently lack graph-derived phase/version/alpha/exact-seed fields;
their absence is expected and nothing is synthesized to replace them. The
live `attention_source` diagnostic is retained under its own name, not
relabeled as an exact working-candidate pressure seed. Collect must demonstrate
real accepted graph state, bounded resources and `selection_unchanged=true`.

Comparison preserves per-case input, Query, selection, accepted-output and
attention-sample identities. Drift, missing exercised collection or missing
actual native C8 makes the result `inconclusive` without retries. Peak native
batch seven still constitutes a completed protocol run; it is not a claim of
eight simultaneous GPU requests. Comparable results expose descriptive cold
wall cost and observed extra collection seconds. NVML uses the actual client
interval through release. Cumulative engine graph counters are never summed
across owners, and overlapping host work is never labeled GPU idle.

This is a **single cold smoke pair**. It has no request warmup, sustained
replacement, 300-second measured interval or repeated alternating pairs.
The planned separate fixed-256K warmup/300s/repeated performance workload remains
uncovered by this entry. The natural eight-agent 300s entry is a different
workload and cannot substitute for that missing fixed-input arm.

CPU validation: 32 tests passed; the actual pair was prepared and its frozen
manifest validated at `artifacts/graph-v2-collection-20261018/r1`. No GPU run
has been started for this pair. A prepared directory is not evidence of a
passed GPU collection comparison.
