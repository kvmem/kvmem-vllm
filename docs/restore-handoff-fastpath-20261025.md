# Restore handoff fastpath

`kvmem.restore_handoff_fastpath` is an opt-in, default-off path for split
asynchronous session restoration. It requires `async_restore=true`,
`restore_prepare_split=true`, and `offline_agent_policy=false`.

## Changes

- Prioritize an already prepared resident restore's model-submit phase over a
  new preparation. Fence completion and aged jobs retain their higher priority;
  existing transfer credits and restore lane limits remain unchanged.
- At native connector `get_finished`, perform a nonblocking completion poll for
  `restore_copy` jobs. This can publish a completed restore and its transaction
  acknowledgement with the current batch. It does not start new preparation or
  private inference, wait on a future, or bypass the CUDA completion fence.
  Publication readiness, allocation ownership, cancellation handling and the
  normal session notice/ACK ordering remain in force. This is not a new worker
  to scheduler messaging channel, so it cannot remove all notification delay.
- Pack and validate exact token prefixes in the native extension instead of a
  Python per-token loop. Match the existing unsigned 32-bit, exact builtin-int
  contract, rejecting booleans, coercions and overflow. The complete byte string
  is compared, without replacing equality with a hash. Non-list/tuple inputs
  retain the existing path.

KV scoring, graph propagation, selection, budgets and model sampling do not
change. The earlier `_merge_spans` optimization is present in both arms.

## Validation

222 runtime tests and 17 controller/port checks passed. CPU measurements on the
eight original accepted-token histories (157,678–212,733 compared tokens) found
exactly equal bytes: the previous packing/validation took 10.46–14.28 ms per
call, and native packing took 0.82–1.05 ms. This ratio is not an Agent speedup.

The new GPU qualification completed 24 responses (eight owners, three rounds),
with all 16 resident restores and model-cache hits passing the live byte oracle.
GPU component tests passed 28 checks. Qualification resources were released.

The finite comparison is stored at
`/data/chaidi/kvmem-vllm-experiments/agent-handoff-fastpath-20261025-r1`.
Its manifest binds the frozen source/library, profiles and original fixtures.
The control and candidate use the same binary, differing only in the new flag.
Both use eight original histories, eight rounds and 64 constrained output tokens
per response. These constraints make all subsequent inputs identical; this is a
restore component benchmark, not natural Agent generation or tool execution.

The original Agent baseline failure and every old answer remain preserved.
The recurring monitor remains stopped. This change has not been committed or
pushed.

## GPU result

Both fresh arms completed 64/64 responses and 56/56 resident restores, with
matching declared input hashes and exact constrained outputs. Both had zero KV
and model-state CPU-to-GPU upload bytes on these warm restores. All frozen-file
hashes, prior evidence and cleanup were verified after completion; no owned GPU
process remains. The candidate completed 48/56 restores at the new post-forward
poll, while the control completed all 56 at the original pre-forward poll.

| Metric | Off | On | Reduction |
| --- | ---: | ---: | ---: |
| Seven eight-way warm round wall times, summed | 29.373 s | 20.554 s | 30.0% |
| Warm request median | 3.543 s | 2.555 s | 27.9% |
| Restore median | 420.55 ms | 198.58 ms | 52.8% |
| Restore P95 | 738.08 ms | 325.37 ms | 55.9% |
| Initial long-input request mean | 237.040 s | 236.664 s | 0.16% |

Read-only trace joins, with request identities and clock consistency checked,
show these mean intervals:

| Interval | Off | On |
| --- | ---: | ---: |
| Admission queue | 632.26 ms | 366.38 ms |
| Worker arrival to background submit | 304.03 ms | 147.88 ms |
| Background submit to worker ready | 99.95 ms | 59.17 ms |
| Worker ready to scheduler acknowledgement | 41.55 ms | 11.32 ms |
| Scheduler acknowledgement to public step | 8.49 ms | 6.06 ms |
| Admission arrival to model submission, total | 1275.25 ms | 720.35 ms |

Admission arrival follows session lookup and is not HTTP arrival. Concurrent
request intervals overlap; their sums cannot be called GPU idle time. The
combined change improves preparation/submission and completion notification;
this experiment does not isolate each of the three changes' causal share.

This is a single sequential Off/On component comparison. The previous r7 warm
restore median was about 217 ms, versus 421 ms in this fresh control; absolute
latency varies between runs. The comparison therefore uses the fresh same-binary
control, not historical numbers. The initial long-input durations were nearly
identical between the two fresh arms. Independent repeats and natural Agent
tool workloads have not measured an end-to-end gain for this new fastpath;
the 30.0% component result must not be advertised as Agent throughput uplift.

Machine-readable results:

- `artifacts/agent-handoff-fastpath-20261025-r1/comparison.json`
- `artifacts/agent-handoff-fastpath-20261025-r1/qualification/execution-audit.json`
- `artifacts/agent-handoff-fastpath-20261025-r1/source-change-audit.json`
- `artifacts/restore-handoff-fastpath-20261025-r1/phase-comparison.json`
- `artifacts/restore-handoff-fastpath-20261025-r1/before-scheduling.json`
- `artifacts/restore-handoff-fastpath-20261025-r1/after-scheduling.json`
