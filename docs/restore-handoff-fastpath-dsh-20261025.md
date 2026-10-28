# Real Agent comparison of the restore handoff fastpath

This experiment measures the qualified restore handoff optimization with real
DSH agents, including tool execution and user follow-ups. It is separate from
the fixed-output component comparison in `restore-handoff-fastpath-20261025.md`.

## Execution

- Root: `/data/chaidi/kvmem-vllm-experiments/agent-handoff-fastpath-dsh-20261025-r1`.
- Frozen manifest SHA256:
  `fd6db4a81e59ef045347149413ea5ccb778c8862e307cd0ee5dc61ee93b82844`.
- Finite controller: `runtime/scripts/benchmark_agent_fastpath_dsh.py`.
- Order: baseline (fastpath disabled), then candidate (fastpath enabled).
- Both arms use the same qualified model code, native library, optimized bridge,
  span merge optimization, resident KV and model checkpoint cache. The only
  profile difference is `kvmem.restore_handoff_fastpath`.
- Sixteen fresh sessions and independent workspaces are used. The eight task
  prompts and original limits come from the previous DSH fixture. Each task
  requires at least four human rounds and more than 36,864 logical input tokens.
- No periodic monitor is started. The existing monitor remains stopped.

The model and recall configuration remains NVFP4 weights, FP8 KV, MTP4,
K=36864/G=16384, block32, sink1312/recent0, MeanK/FPS4 50:50, FPS4 neighbours
within the original budget, graph alpha0, guided0, cumulative attention for
pressure, eight requests, batch16424/chunk2048, CPU history100GB, sixteen FPS
workers and 12GiB shared cache. Physical prefill coalescing remains disabled.
Prompts, sampling, tools and output/step limits are preserved in the frozen
task plans and request evidence.

## Measurement and outcome rules

The primary result is accepted output tokens divided by full arm wall time,
including tools and follow-ups. MTP verification rows are excluded. Report it
with task completion, per-task duration, output length, tool calls, tool errors,
shell nonzero exits, bridge failures, basic browser checks, restore timings and
memory usage. Natural generation can change code, token counts and tool paths
despite identical task prompts and sampling settings. This single sequential
pair does not establish repeatability or a confidence interval.

Original per-turn resource limits remain outcomes. A task reaching its limit
is not retried or granted a larger budget; other tasks and the other arm may
continue only after preserving and validating the limit evidence. Infrastructure
or unclassified failures stop the queue. Already completed answers are never
overwritten. The previous interrupted seven-task baseline is not resumed and
is not used as this experiment's denominator.

Each arm writes `performance.json`, `observations.json`, `execution-audit.json`,
`evidence-hashes.json`, `browser-summary.json`, `cleanup.json` and `outcome.json`.
After both arms finish, the controller writes `comparison.json`; its
`full_paired_cohort_passed` field explicitly distinguishes a complete pair from
one containing original resource-limit failures. `status.json` is the current
execution state. The launch checks passed, but no real Agent speedup is claimed
until the measured outcomes and final audits are available.
