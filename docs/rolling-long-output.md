# Single-request rolling output

`gen_budget` is physical decode headroom, not necessarily a response length.
With `request_planning` and `rolling_active_window` enabled, native token-ID
completion requests may set `max_tokens` above G. Non-rolling backends retain
the previous requirement that output fits within G.

For K=98304, G=16384, block=32, the existing scheduler visits an accepted-prefix
pressure boundary before the K+G allocation fills. It suspends that owner,
selects at most K tokens, atomically installs the selected KV view, then resumes
the same native request. The output counter, sampler, reasoning budget and
tool frame continue across this operation. No extra user prompt, synthetic
assistant turn or per-G native request is inserted. Other schedulable owners
remain eligible while the refresh completes.

## Configuration

The backend profile still uses:

```json
{"kvmem": {"select_budget": 98304, "gen_budget": 16384,
           "rolling_active_window": true}}
```

These are fields within the existing qualified profile, not a complete launch
profile. All prerequisite asynchronous refresh/restore and history coordination
settings remain required.

Both `benchmarks/deepswe/claude_bridge.py` and `scripts/dsh_bridge.py` accept
`--max-output-tokens` independently of `--gen-budget`. For example,
`--gen-budget 16384 --max-output-tokens 131072` permits one native response up
to 128K while keeping G=16K. The DSH bridge requires `--max-output-segments 1`
(the default) for this mode. Its older segmented mode remains a separate option.

The caller must also request a larger output limit. A client that sends
`max_tokens=16384` still stops at 16384; the server does not silently increase
an explicitly requested limit. A future DeepSWE launch must configure its
isolated client's output cap along with the bridge and freeze the new values
in that run's manifest. Historical DeepSWE queue profiles and results keep
their original 16384 output cap. Omitting the new bridge option preserves that
previous default.

Generation stops at EOS, the requested stop condition, cancellation, or the
independent output cap. Prompt plus requested output must fit the logical
model limit. KV re-selection does not make logical positions or the full
accepted-token journal infinite. The separate CPU history/GC policy still
governs archived KV. Non-rolling backends reject requests above G before
they enter the scheduler.

Tool parsing uses the entire response. Incomplete tool arguments at an output
limit are not executed. A complete tool frame can therefore span several KV
refreshes without being cut into separate model requests.

## Validation

`scripts/qualify_rolling_long_output.py` prepares an isolated frozen component
run from the completed DeepSWE runtime. It changes only HTTP output admission
in model source, retains K96K/G16K and MTP4, and enables the existing byte oracle.
Two owners each receive 100352 fixed input tokens and generate 36864 fixed
accepted tokens in one native request, followed by an exact-prefix resume,
an EOS check, and session release. The audit requires multiple zero-replay
refreshes per long request, including after G output, bounded selected KV,
exact target/MTP KV and recurrent state, response ID continuity, usage/accepted
ID agreement, and owned-process/GPU cleanup.

This constrained-output run verifies mechanics. It does not measure Agent
quality, unconstrained generation equivalence, eight-way throughput, or a
10M-token continuous output. Offline bridge tests cover a complete tool
argument longer than 32K, truncation without tool execution, exact accepted
IDs, sampling fields and legacy defaults.

The 2026-10-26 run passed: both long requests returned exactly 36864 accepted
tokens; all four rolling refreshes selected 98304 tokens and passed exact
target/MTP KV and recurrent-state checks. Both exact-prefix resumes and EOS
checks passed. All session payloads, active owners and owned GPU processes
were released. The related offline checks passed 217 tests.
The frozen inputs, responses, source hashes and
final audit (local experiment artifact: `artifacts/rolling-long-output-20261026-r1/final-review.json`)
are retained in the experiment directory.
