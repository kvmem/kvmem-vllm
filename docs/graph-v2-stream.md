# Fixed 256K continuous request stream

This is a separate sustained-load protocol. The cold eight-case screen and the
natural multi-turn agent workload do not satisfy it.

`benchmark_graph_v2_stream.py` freezes the original eight 256K NIAH token inputs
in a finite queue of 128 independent aliases. Each alias has a unique
`source_request`, independent session/cache salt and request ID. Token bytes,
public Query coordinates, sampling seeds, output ceiling, K=36864, G=16384 and
native MTP(k=4) remain unchanged. Gold answers are only used after inference.
Preparation never starts a GPU service.

The client continuously admits up to eight owners. A request retains its slot
and CPU full-history credits through its own ordinary retirement and scheduler
release acknowledgement. Once that owner releases, its slot can refill while
any other request is still running or retiring. There is no wave barrier.
Post-run graph/answer audits do not delay slot replacement.

The sole queue coordinator fixes T0 when at least eight complete requests have
released, and fixes T1=T0+300 seconds. Existing requests continue across T0.
Every subsequent admission checks that same immutable deadline. At T1 the
coordinator stops admission and drains already admitted requests; drain work
and output are excluded from the measurement. Unused aliases are explicitly
`not_admitted_deadline`, not failed requests. Exhausting the finite queue before
T1 invalidates the sustained-load attempt. All warmup, window and drain events
remain available for audit. Worker token callbacks cannot change the window.

The client observes progress, archive, public-work counters, scheduler credits
and NVML every second. These files retain their publication timestamps and
counter identities, where available, as well as client read-time brackets.
Each sample also retains the opened source file's publication mtime and age;
an age over five seconds is marked stale as a diagnostic. A read timestamp
never substitutes for an update timestamp. In particular, `progress.json` can
remain unchanged for a long period while public-work and model-step counters
advance. Such stale logical-frontier evidence does not imply an idle GPU.
The primary prefill metric is the delta in **completed public prefill work**,
divided by 300 seconds. These are executed public prompt positions, including
re-prefill/replay, not useful unique input. Private Query forwards are excluded
by the existing public-work counter. Completion is observed asynchronously;
the report includes the original boundary snapshots and conservative
submitted/completed bounds. It never treats a fence observation as an exact GPU
completion timestamp.

Accepted output is the actual SSE `token_ids` received in [T0,T1). The complete
per-request event journal must agree with native `usage.completion_tokens` and
the retained response IDs. Zero measured output and zero completed requests
are valid: a 256K input can take longer than the measurement window. No MTP
proposal or verification position becomes an accepted-output count.

Exact unique-input throughput is left null. As auxiliary evidence, native
owners are mapped back to retained response IDs, their published logical
frontiers must be monotonic, and each frontier is capped at the original input
length. A strict unique-input lower bound only includes requests admitted at
or after T0 and observed before T1. Carry-in owners retain observed frontier
deltas, but contribute zero to that strict lower bound. Missing/ambiguous
identity or regressing publication watermarks leave the lower bound null.

Completed model-step timing rows separately report pure-prefill model tokens
per GPU second and pure-decode batch-size frequencies. The latter is never a
decode-token throughput estimate. Steps are selected by host start in the
window and may cross its edge; that scope is explicit. Neither host service
time nor NVML utilization is an inferred GPU-idle measurement.

Prepare one pair using a new directory:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B \
  scripts/benchmark_graph_v2_stream.py prepare \
  --candidate raw-k-cache-window --pairs 1 \
  --output /absolute/path/to/new-stream-attempt
```

The same four candidate contracts as the cold screen are supported:
`raw-k-cache`, `raw-k-cache-window`, `cpu-direct-record-seal` and
`async-action-metadata`. Each fresh baseline/candidate pair changes only its
declared configuration field. `--pairs 3` freezes AB/BA/AB order. Input aliases,
profiles, runtime/native extension, model identity and helpers are hashed.

Validate with the frozen controller:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B \
  /absolute/path/to/new-stream-attempt/frozen/benchmark_graph_v2_stream.py \
  validate --output /absolute/path/to/new-stream-attempt
```

`run` starts fresh owned services sequentially and requires GPU authorization;
omit the empty CUDA visibility setting only for that step. The existing owned
process cleanup, GPU lock, 4 GiB device-headroom guard and CPU-history guards
remain in force. A started, failed or completed campaign cannot be retried.
Failures preserve outstanding credits and evidence, then the owning controller
terminates its own client/service groups.

Comparisons retain common-request input/output/selection identity and raw
rates, including null ratios when the baseline output rate is zero. Drift,
unobserved candidate activity, absent native C8 or unavailable public-work
boundary brackets makes a pair inconclusive. Existing terminal-only metadata
upload totals cannot by themselves prove activity inside the fixed window.
Neither one pair nor three pairs automatically establishes a performance gain;
quality, work identity, boundary uncertainty and paired consistency remain
explicit review requirements.
