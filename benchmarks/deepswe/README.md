# Executed agent qualification

This directory preserves the existing official Pier/mini-swe-agent policy and
fixed requestplan10 selection. The historical 8/10 in the selection file is
provenance from another framework/run, **not a vLLM result**. The adapter is the
exact `ReliableMiniSweAgent` prefix extracted from the original source; its
hash and extraction boundary are recorded in `PROVENANCE.json`. Solver prompts,
tools, main loop and the task verifier stay under the official packages.

The new runner uses one existing shared inference service. It copies each task
into its own artifact directory, records every copied file hash, and runs exactly
one attempt with no retries. Official numeric zero and a scored agent timeout
are retained as outcomes. Missing scores, duplicate attempts and other exceptions
cannot qualify as task-quality evidence. The official task agent timeout is
preserved (`--agent-timeout-multiplier 1.0`); installation/verifier infrastructure
uses the existing 4x multiplier. The inherited per-shell and model-call limits
are documented in the adapter. The qualification runner uses full history by default.

## Optional model-generated history summaries

Both the Anthropic Messages bridge in this directory and the OpenAI Chat
Completions bridge in [dsh_bridge.py](../../scripts/dsh_bridge.py) accept
`--compaction framework` (the default) and `--compaction gateway`. Use the
default for agents that already request summaries. For agents that resend their
entire transcript, enable gateway mode on the bridge command. For example, add
the following flags to the existing launch command.

```sh
--compaction gateway --compaction-trigger 0.80 --compaction-target 0.55 \
  --compaction-summary-tokens 2048 --compaction-keep-messages 4
```

Fractions apply to the input budget after reserving the public response.
The bridge sends the same model a private completion with the older history
and a handoff instruction. It can summarize several closed segments when the
incoming raw request already exceeds the model limit. The original task,
current user request or verified Query, tool definitions, recent messages
and outstanding tool transactions remain available verbatim. The bridge
checks both the private and visible input plus output quotas. A large indivisible
tool transaction or protected prompt can still exceed the context limit.

The client sees the ordinary response and its usage. The bridge records private
model time and token counts in `requests.jsonl`. Each owner directory stores
its current `compaction.json` alongside the existing raw turn journal.
The source-prefix digest, model, control messages, template, and tools must
match for reuse. Route each owner to one bridge writer and use the normal
per-task URL or DSH conversation key. Anthropic `count_tokens` measures the
committed projection without generating a summary. A client that rejects an
oversized request locally still needs sufficient headroom to reach the bridge.
The first16 runner also accepts `--compaction gateway` and records the mode
in its run manifest for comparisons with the default full-history run.


## Sequence

1. Start `scripts/serve_agent_backend.py --recipe <completed-results.json>
   --output <new-server-artifact>`. This freezes current worker code, records
   the qualified recipe, enables native per-request timing and speculative
   summary metrics, and owns the GPU lock plus its own API process group.
   It refuses an occupied GPU, failed recipe or profiler recipe. Use matched
   recipes for native Q8 and the optional Tensor Core backend; preserve all
   K/G, precision and scheduler settings.
   It also enables native Qwen XML auto tool calls and the Qwen reasoning parser,
   matching the qualified tokenizer's `<function=...>/<parameter=...>` format.
   A completion-only recipe without these API flags rejects the official
   agent's `tool_choice=auto` before inference; that configuration failure is
   not a model outcome.
   Once ready, run `scripts/preflight_agent_api.py --api-base <host-api-base>
   --engine-evidence <server-artifact>/server_state.json --output <new-preflight.json>`
   to check one native tool call, decode/spec metrics, exact terminal archive and
   complete resource retirement before launching tasks. This executes no shell tool
   and is not a task or throughput result. Failures are saved, never overwritten.
2. Start `scripts/manage_agent_relay.py --upstream-port <server-port>
   --image <existing-task-image-with-python3> --output <new-relay-artifact>`.
   It creates its own Docker bridge address and container, exposing port 443
   only on that private address. It does not occupy/reuse the old QW3 relay.
   `resources.json` records owned IDs and the API base. SIGTERM/SIGINT removes
   only those resources. The relay supports ordinary non-streaming agent calls;
   it passes JSON request/response bytes unchanged, requesting identity content
   encoding, and journals native response metrics without API credentials.
   During a silent upstream call it monitors client disconnects and closes the
   native API socket so timed-out callers do not leave orphaned GPU requests.
   Before starting tasks, run `scripts/preflight_agent_cancel.py --engine-evidence
   <server-artifact>/server_state.json --relay-evidence <relay-artifact>/resources.json
   --output <new-cancel-preflight.json>`. The idle backend must demonstrate actual
   accepted prefill, native abort, exact terminal clipping and full resource
   release through this relay. This is not compressed-refresh cancellation.
3. Run the pinned Pier environment against that address:

   ```sh
   .venv/bin/python scripts/qualify_agent.py \
     --pier-python /path/to/pinned-pier-venv/bin/python \
     --task-root /path/to/deep-swe/tasks \
     --task termenv-preserve-ansi-resets \
     --concurrency 1 \
     --api-base http://BRIDGE_ADDRESS:443/v1 \
     --health-api-base http://127.0.0.1:SERVER_PORT/v1 \
     --engine-evidence /path/to/server-artifact/server_state.json \
     --relay-evidence /path/to/relay-artifact/resources.json \
     --preflight-evidence /path/to/new-preflight.json \
     --cancel-preflight-evidence /path/to/new-cancel-preflight.json \
     --output /path/to/new-agent-artifact
   ```

   Repeat `--task` for fixed tasks and use concurrency 1/2/4 on the same shared
   server. `--prepare-only` validates versions and snapshots inputs but invokes
   neither the agent nor verifier; it is explicitly not a scored run. Python,
   Pier commit/version, mini-swe-agent version and task-manifest hash are checked.
   The wheel SHA in the historical lock is retained as provenance; this runner
   does not claim to have reverified the wheel downloaded inside every container.
   Actual execution checks that the ready backend, managed relay and host health
   URL describe the same endpoint. It retains the per-task native request records
   and token-weighted per-request decode separately from concurrent throughput.
   Both preflight files must match these running backend/relay instances.
   The copied `pier_transport_entry.py` verifies the pinned official network
   setup source and aligns generated Squid read/forward timeouts with the existing
   21600-second model-call limit. The default 900-second proxy timeout previously
   caused HTTP 504 and overlapping retries. Installed Pier files, allowlists,
   solver/retry policy and scoring remain unchanged; this transport adjustment
   is recorded separately from the unchanged agent-adapter extraction.
4. Inspect every task's official result, full trajectory and produced patch,
   plus relay metrics and server archive/resource state. Then stop the owned
   relay and server using their recorded manager PIDs. Do not stop other
   containers or inference processes.

The independent relay smoke test used only `/v1/models` from a fresh container,
then confirmed its own container/network were removed. It did not issue extra
generation requests into the concurrent long-history benchmark.

## Metric interpretation and remaining gates

`generation_time_ms` from pinned vLLM measures first-to-last generated token;
decode rate is `(completion_tokens - 1) * 1000 / generation_time_ms`. Native
`tokens_per_second` includes prefill and must not be relabeled decode throughput.
Report per-request rates and token-weighted decode separately from whole-task
wall time, tool time, aggregate service throughput and verifier reward. Summing
overlapping request durations is not aggregate GPU throughput. Missing native
metrics, one-token outputs, failed requests, context-limit errors and retries
must remain visible.
Only successfully delivered HTTP responses enter completed-request timings;
disconnects, undelivered responses and relay errors remain in the journal.

These scripts are an execution/measurement entry point, not evidence of task
success. Current bounded DFlash serving still has a 262144 logical-position
limit and no qualified persistent-session continuation. Requests beyond that
limit must be recorded as failures; do not silently truncate or summarize them
to make a quality comparison pass. Persistent multi-turn state reuse, the full
fixed task set, matched repeats and task-quality acceptance remain required.
