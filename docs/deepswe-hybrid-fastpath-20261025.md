# DeepSWE first16 with hybrid recall and restore handoff optimization

The user requested the previously tested 16 tasks, and explicitly selected a
524,288-token logical context limit to observe completion quality. This is a
fresh experiment; it does not resume any previous failed answer.

## Fixed workload and current deployment

- Original fixture: `/home/chaidi/qw3/benchmark/action_pressure_first16_warmfix_20261010`.
- Same 16 instructions, base commits, 16 agent images, 16 verifier images,
  Claude Code 2.1.148 binary, text protocol, tools and committed-patch grading.
- Eight independent agent containers; each free worker takes the next task.
  Every task receives one attempt. Hidden tests remain isolated from the agent.
- Original sampling: temperature0.6, top-p0.95, top-k20, min-p0, presence penalty0,
  repetition penalty1, seed1000, thinking4096, Claude effort xhigh.
- Original no-compaction policy, unlimited agent wall time, 30-minute verifier
  timeout, cached-Read streak32 and identical-successful-Bash streak32 guards.
- Current qualified NVFP4/FP8 KV/MTP4 implementation, K36864/G16384,
  50:50 MeanK/FPS4, FPS neighbours within the same budget, graph alpha0,
  guided0, cumulative attention for pressure, batch16424/chunk2048.
- Current CPU history100GB, FPS CPU16 workers/cache12GiB, resident KV,
  model checkpoint cache2GiB, split restore, native frame and restore handoff
  fastpath. Physical prefill coalescing remains off.
- Session names are provisioned for32 instead of8 to accommodate16 task names
  plus synthetic qualification. GPU concurrency remains8; CPU history and KV
  budgets are not increased. Completed task owners are explicitly released.

The production model/native files come from the frozen qualified runtime in
`agent-handoff-fastpath-dsh-20261025-r1`. The DeepSWE bridge now exposes the
already tested exact-identity/local-span optimization as an opt-in flag. The
original request tokenization, sampling and tool protocol are retained.

## Queue and outputs

Current root: `/data/chaidi/kvmem-vllm-experiments/deepswe-hybrid-fastpath-512k-20261025-r2`.

The finite queue waits for both current DSH arms to finish, validates their
outcomes and resource cleanup, waits for the exact controller to exit and the
GPU to remain idle60 seconds, then runs the DeepSWE preflight and formal cohort.
It does not acquire the GPU in the gap between DSH arms and does not restart the
cancelled periodic monitor. It never stops another experiment's processes.

Before this queue was launched, the DSH baseline controller failed with seven
client RPC timeouts; only one task completed and its candidate arm never
started. The prior failure and all accepted answers remain intact. The new
independent DeepSWE workload uses an explicit failed-predecessor acknowledgement
which binds the failure logs, answer hashes, exited processes and empty cleanup.
DSH sessions must also be idle. This releases the resource dependency without
claiming a completed DSH comparison or resuming any of its failed answers.

The r1 startup reached service readiness but its legacy anonymous synthetic
Query probes were rejected by the current owned-session requirement. Public
submitted/completed counters were all zero, and no formal task was started.
The r2 controller gives cold synthetic probes unique temporary sessions and
releases them after the response; continuation probes keep their original
explicit owners. Request/response and HTTP error evidence are now saved.
`preflight-recovery-audit.json` binds all r1 files and empty cleanup; source
model files, task inputs, profiles, sampling and verifiers must match exactly.

`first16/run.json` records per-task status. Raw responses, accepted token IDs,
Claude tool events, patches, verifier results and failure evidence are retained.
Final `scores.json` separates verified solutions, valid verifier scores,
context-ceiling outcomes and infrastructure failures. Tool errors and hidden
test failures are different measurements. `execution-audit.json`,
`evidence-hashes.json` and `cleanup.json` provide execution and cleanup evidence.

Historical comparisons are descriptive. The latest previous vLLM first16 run
used a196,608-token context ceiling and K65536/G32768;14 tasks hit that ceiling.
The older complete historical reference also used different budgets/thinking
settings. A change in this new cohort's solve rate or throughput cannot be
attributed solely to the restore optimization.
