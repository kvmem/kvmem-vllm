# V2 performance pairs

`scripts/benchmark_graph_v2_performance.py` freezes two independent V2 D runs
against the same original eight 256K NIAH cases. Preparation and validation
do not launch a model. Each pair changes exactly one runtime setting:

| `--candidate` | Frozen variants | Changed setting |
| --- | --- | --- |
| `raw-k-cache` (default) | `raw0`, `raw4g` | `gpu_raw_k_cache_bytes`: 0 → 4294967296 total; async arenas each have a 2 GiB cap |
| `raw-k-cache-window` | `raw0`, `raw10g` | `gpu_raw_k_cache_bytes`: 0 → 10737418240 total; each arena has a 5 GiB budget and a 4.78125 GiB effective geometry limit |
| `cpu-direct-record-seal` | `seal-copy`, `seal-direct` | `cpu_direct_record_seal`: false → true |
| `async-action-metadata` | `metadata-sync`, `metadata-async` | `async_action_metadata`: false → true |

Both boolean pairs keep raw-K caching disabled. Other pairs keep direct sealing
and asynchronous Action metadata disabled. Every pair retains the base
`native_action_metadata` setting; it is never enabled along with the metadata
candidate. All runs use graph mode `multi`, V2 alpha 0.2 and graph pressure.

The window-sized raw-K candidate is derived from the current geometry, not a
parameter sweep. FP8 raw K uses `17 layers × 4 heads × 256 dimensions = 17408`
bytes per token. Eight active windows of 36864 tokens require 4.78125 GiB per
arena. The current independent main/refresh arenas split the configured budget
equally, so complete coverage requires 9.5625 GiB total; the next whole-GiB
budget is 10 GiB. The cache also caps each arena at eight windows, hence its
effective maximum remains 4.78125 GiB even with a 5 GiB arena budget. Allocation
is lazy, and an unused main arena does not lend its budget to refresh. The
existing 4 GiB pair remains a separate experiment. Neither pair changes vLLM's
KV pool or `gpu_memory_utilization`; raw-K allocation is additional GPU memory,
subject to the existing allocation check and controller free-memory guard.

From the repository root, prepare a new directory:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B \
  scripts/benchmark_graph_v2_performance.py prepare \
  --candidate cpu-direct-record-seal \
  --profile artifacts/graph-overhead-20261018/gpu-r2/after/profile.json \
  --output /absolute/path/to/new-experiment
```

Use that directory's frozen controller for `validate` or `run`:

```sh
.venv/bin/python -B \
  /absolute/path/to/new-experiment/frozen/benchmark_graph_v2_performance.py \
  validate --output /absolute/path/to/new-experiment
```

Replace `validate` with `run` only when ready to use the GPU, without the empty
`CUDA_VISIBLE_DEVICES` setting. The candidate and variants come from the frozen
manifest. Optional `--variants seal-copy` runs one prepared variant; a later
invocation can run `seal-direct`. Failed, interrupted and completed attempts
cannot be restarted. A new attempt needs a new output directory.

This is one cold eight-case screening run per variant. It has no request
warmup, sustained replacement, fixed 300-second window or repeated pairs.
The controller preserves raw telemetry, per-case selection/output hashes,
quality scores, actual native concurrency, whole-fixture useful-input and
accepted-output rates, and NVML samples over the actual queue execution through
owner release. `comparison.json` treats raw ratios as descriptive. Changed
outputs or selections, an unexercised candidate, missing actual native C8, or
incomplete direct-seal measurements produce `inconclusive`. Otherwise the
result is only `comparable_cold_smoke`; `performance_gain_established` is always
false. Failed or divergent answers are never silently repeated.

Sealing counters come from the final archive snapshot and count successful
publications only. Missing pack/hash/copy/direct-byte measurements are listed
as unobserved with null values, distinct from measured zero. Action metadata
reports retain one latest cumulative upload snapshot rather than summing the
same engine totals across owners. Raw-cache capacity is a gauge: without arena
identity the observed maximum is not total allocated bytes and snapshots are
never summed. Host work and overlapping service intervals
are not estimates of GPU idle time. A single pair is not a repeated statistical
speedup estimate.
