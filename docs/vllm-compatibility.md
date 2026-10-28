# vLLM compatibility and native SystemOne

> **Documentation preview.** This branch previews the README, GIF and guides. The runtime and benchmark-tooling updates described here are still pending publication. Installation commands and new profiles require that development runtime; they are not a validated installation path for this preview branch.

KVMem follows upstream vLLM through capability checks, a small set of native
adapters, and regression tests. The package requires `vllm>=0.30.0`; it does not
reject a release or development build because its version string changed.
This is not a promise that every future upstream implementation is compatible.
Missing interfaces and unsupported runtime layouts still fail explicitly.

## Install and check

For the current released vLLM and the default safetensors serving profile:

```sh
bash scripts/setup_gpu.sh --latest
.venv/bin/python -m vllm_kvmem.engine_api --systemone
```

The installer resolves vLLM and its dependencies together. The old
`requirements-cu130.lock` is a reproducibility artifact, used only by
`scripts/setup_gpu.sh --locked`; it is not the runtime compatibility policy.
The historical GGUF plugin and specialized FlashInfer source overlays have
their own checks, independent of the vLLM version. A new vLLM release does not
automatically qualify an unsupported CUDA kernel overlay or GGUF plugin.

An upstream source checkout can be checked without installing or importing its
GPU implementation:

```sh
PYTHONPATH=src python -m vllm_kvmem.engine_api --source /path/to/vllm/vllm --systemone
KVMEM_TEST_VLLM_SOURCE=/path/to/vllm/vllm KVMEM_TEST_REQUIRE_SYSTEMONE=1 \
  PYTHONPATH=src python -m pytest tests/test_engine_api.py -q
```

CI checks the historical baseline, latest GitHub release and current upstream
main, including a daily run. These source checks verify required signatures and
patch structure. They are explicitly **not GPU qualification**. Memory ownership,
cache geometry and request generation checks remain in the runtime.
Ownership-sensitive source operations (state counter initialization, staged
page writes and idle connector completion) are checked as syntax trees rather
than whole-file hashes. Packed per-layer page layouts outside the existing KV
reader contract are rejected explicitly.

## Native SystemOne

On a model with upstream decision support, the usual `vllm-kvmem serve` launcher
exposes vLLM's own `POST /v1/systemone`. Reinstall this package after updating it
so the `vllm.endpoint_plugins` entry point is discovered. The launcher already
allowlists the `kvmem` plugin. vLLM 0.30.0 remains usable for existing APIs but
does not gain a backported SystemOne route.

The plugin attaches to the native decision strategy after native app setup. It
adds no route and copies no question schema, prompt renderer or answer formula.
By default, each native label read receives a server-generated, temporary KVMem session.
Concurrent reads are bounded by the engine's sequence limit. Completion,
failure and cancellation use the existing fenced session-release protocol;
a read retains its concurrency slot until release completes. These requests
do not reuse or modify an agent's persistent session.

Current integration is for rendered text prompts and the direct native API.
The separate elastic and legacy multi-user gateways need their own routing and
admission integration before exposing this endpoint. Automatic cross-question prefix
sharing and explicit question-aware retrieval spans are not implemented here;
long inputs use the existing KVMem pressure policy.

Native probability extraction is retained. Where the upstream speculative
sampler omits `logprob_token_ids`, a checked import adapter forwards label IDs
and request mappings to its existing `compute_topk_scores` helper. The adapter
becomes a no-op when native extraction already supplies that argument.

Responses retain upstream vLLM semantics. In particular, TypeSafe's `confidence`
definition and limits are not identical to upstream vLLM's; this implementation
does not claim strict TypeSafe/Jev semantic equivalence.

### Explicit active-session decisions (experimental)

An optional `kvmem_session` extension reads an existing, currently generating
parent from its GPU state. Keep `state` empty and use the parent's `cache_salt`:

```json
{
  "state": "",
  "cache_salt": "tenant-a",
  "kvmem_session": {"session_id": "agent-123"},
  "questions": {
    "ready": {"type": "noul", "instructions": "Is the task complete?"}
  }
}
```

The reference can also include positive `generation` and `history_end` values
as strict preconditions. The worker binds a request to one existing session
incarnation and waits for the next scheduler-confirmed boundary after prefill.
All questions in that HTTP request observe the same accepted prefix. The
response's `kvmem` object reports the selected generation, history boundary,
active token count, question token count, rollback bytes and execution time.
Native `usage.input_tokens` counts newly evaluated question/turn-separator
tokens; the reused active prefix is reported separately.

This is a synchronous temporary branch: the worker pauses public execution,
reads the existing GPU prefix, appends each native-rendered question in scratch
tail slots, and computes label logprobs with the native helper. It restores
overwritten tail KV, every recurrent/speculation row and the acceptance counter
between questions and before public execution resumes. It creates no scheduler
request, publishes no question tokens to history, and leaves APC disabled.
The private branch temporarily closes the current Qwen assistant turn before
adding the question. Its meaning is a judgment of the parent's **active model
context**, including any KVMem compaction, rather than rereading a supplied raw
state string. A reference to a completed/idle session is currently unsupported.

The initial supported path is single-worker Qwen GDN, text-only, FP8 KV, eager
execution, synchronous refresh, and either native MTP or no speculative draft.
CUDA Graph, DFlash, native attention scoring, top-level decision instructions,
multimodal input and LoRA are rejected. There must already be enough exclusive
tail space in the parent's pages and K+G window; this version does not reserve
additional scheduler pages. Requests are bounded to eight questions, 512
rendered tokens per question, a 512 MiB rollback snapshot and a 30-second queue
wait. Unsupported/stale/busy session conditions return HTTP 409; malformed
extensions return HTTP 400. These limits apply only to the extension.

Use `scripts/qualify_systemone_active.py --output /path/to/new/results` against
an independently owned local server to compare deterministic parent outputs
with and without active reads. Setting `verify_session_restore: true` in the
serving profile additionally checks byte equality of the full target
prefix, scratch tail and all recurrent/counter state after rollback. This audit
adds copies and synchronization and is intended for qualification, not timing.

## Upstream changes covered

- Forward additional native runner and allocator keyword arguments.
- Recognize allocator-page tables as well as older expanded kernel-page tables.
  Direct KV readers and restoration witnesses translate to the bound tensor's
  kernel pages; allocator validation uses the stored table's own page count.
- Apply the old DFlash loader correction only when upstream still needs it.
- Keep attention metadata adaptation guarded by its required source structure.
- Publish complete refreshed page rows through KVMem connector metadata when
  the engine has no native `SchedulerOutput.block_table_updates` field. The
  worker uses native `append_block_ids(..., overwrite=True)` and suppresses
  duplicate per-step appends while retaining native counter and zeroing work.

The optional `offline_agent_policy` still requires the native queue-selection
hook. Upstream main removed that hook; enabling this experiment on that engine
fails explicitly. The default serving profile uses native online scheduling.

## Local validation, 2026-10-25

The same 20-file CPU regression selection was exercised against these engines:

| Engine source | Result |
| --- | --- |
| Installed 0.30.0 baseline | 260 passed, 3 skipped |
| Release 0.32.0 (`5df63a7203cd`) | 262 passed, 1 skipped |
| Upstream main (`71a82756040a`) | 262 passed, 1 skipped |

Every engine skips the opt-in CUDA checkpoint test. The 0.30.0 baseline also
skips two native-SystemOne checks because that API is absent. Tests cover native
scheduler admission and allocation, session lifecycle, page translation,
attention metadata and source adapters. On the two newer sources, the real
upstream decision handler, question types, label-read strategy and answer
calculation execute with a fake inference engine. GPU kernels are not exercised.
The source check CLI also passes with `--systemone` on both newer snapshots.

The newer sources were selected with `PYTHONPATH` using existing Python
dependencies; they were not installed as complete, runnable GPU environments.
The existing service environment was not upgraded. The initial test process
could not access the NVIDIA driver; a later host-level check confirmed that the
GPU and driver work. No actual-model GPU checks were run in this selection, so
these results do not establish end-to-end model correctness or performance.
CI runs only the source checks, not this broader local selection.

A separate actual-model attempt installed the released vLLM 0.32.0 with its
resolved dependencies in an isolated environment. The NVFP4 model and native
SystemOne route started on the RTX PRO 6000. The first model request exposed
an old-environment assumption: `SchedulerOutput.block_table_updates` was read
unconditionally although it is absent upstream. The connector publication
adapter above fixes that path; 59 related CPU tests pass on both installed
0.30.0 and 0.32.0, including full-row replacement, peer appends, transport and
restore fencing. The corrected GPU rerun was initially deferred because another
GC experiment acquired the shared GPU lock. The subsequent results are below.
Evidence is under `artifacts/systemone-gpu-20261025-0s6d5a3v/`; use
`scripts/qualify_systemone.py` against an owned service to reproduce the HTTP
and session-release checks.

## Actual-model validation, 2026-10-26

An isolated snapshot of the current source was rebuilt and exercised with
vLLM 0.32.0, the Qwen3.8-27B NVFP4 model, FP8 KV and an RTX PRO 6000. The MTP
profile uses eager execution, one speculative token, two concurrent sequences,
K=2048 and G=2048. Historical GC is disabled in this profile. The original
service environment was not upgraded. The related CPU selection passes all
81 tests; the HTTP client regression selection passes all 14 tests after
correcting its missing-state expectation to the native full-app HTTP 400.

The native endpoint plugin passed 12 HTTP cases, covering noul and choice,
invalid requests, repeated caller IDs, concurrency and a post-cancellation
probe. A separate disconnect was observed by the native scheduler as
`FINISHED_ABORTED`, followed by its matching fenced release. The 14 initial
label reads, the cancelled read and the probe each have matching release
records; the resource checks report no outstanding requests, owners,
checkpoints, archive payloads or allocated request blocks.

Both question types also passed while sharing a model batch with ordinary MTP
decoding. The capture verifies the final SystemOne prefill and a speculative
decode in the same batch. Compared with the same service in isolation, the
maximum conditional-probability changes were 0 for noul and approximately
0.000617 for choice. These are observed differences, not an accuracy guarantee
or a comparison with an independent native reference.

Evidence: `artifacts/systemone-gpu-20261026-q8mu5cc_/mtp-02/` (`client/` and
`mixed/`). The preceding `mtp-01` attempt had stale copied package metadata
that omitted the endpoint plugin; its results are not integration evidence.
Refreshing the snapshot's package metadata loaded the registered plugin for
`mtp-02`, which records the expected `kvmem-decision-*` sessions.

A small warm-service HTTP measurement used ten requests per scenario, after
one excluded warmup. The input counts below include the native rendered prompt;
the four-question request reports the sum across its four independent reads.
Each scenario also passed the resource-release checks.

| Rendered input tokens/request | Questions/request | Client concurrency | Median HTTP latency | P95 HTTP latency |
| --- | --- | --- | --- | --- |
| 384 | 1 | 1 | 0.728 s | 0.736 s |
| 2848 | 1 | 1 | 3.631 s | 3.744 s |
| 1536 (384 per question) | 4 | 1 | 2.522 s | 2.535 s |
| 384 | 1 | 4 | 1.279 s | 2.557 s |

Raw responses and timings are in `mtp-02/efficiency/`. These are measurements
of the diagnostic eager MTP profile, with request/step capture enabled, not a
sustained-throughput benchmark or an optimized production configuration. No
independent native-service timing was measured, so no speedup is claimed.
Repeated inputs still create fresh sessions and do not reuse their prefix KV.

This validates the disposable-session path. It does not establish shared-state
KV reuse, long-context answer quality, the default DFlash/graph profile,
historical-GC interaction or equivalence with TypeSafe's confidence semantics.

Before promoting a new engine environment, run actual-model checks for short
and compressed sessions, cancellation/reclamation, ordinary generation, and
SystemOne reads mixed with DFlash decoding. Compare requested-label probabilities
to a native non-speculative reference, not just the chosen answer. CUDA graph
capture, recurrent-state restore and FlashInfer overlays need GPU execution.

### Active-branch validation, 2026-10-26

The experimental extension was exercised on the same vLLM 0.32.0 / Qwen NVFP4
eager MTP configuration, using independently owned server processes. The final
six-file CPU selection passes 187 tests, including native HTTP success and
disconnect cleanup, stale/wrong owner rejection, unsupported-mode rejection,
all-question snapshot isolation, and rollback after an injected forward error.
No upstream package source was edited.

Two actual-model comparisons each generate 512 deterministic parent tokens
with and without two inserted three-question decisions (noul, choice, repeated
noul). Parent output token IDs match exactly in both comparisons. Within each
snapshot, the repeated question's answer, probabilities and diagnostics match
exactly. Different snapshots may give different answers as the parent advances.

| Parent prompt | Accepted boundary at first decision | Reused active tokens | New tokens for three questions | Observed HTTP latency (two calls) |
| --- | --- | --- | --- | --- |
| 40 tokens | 74 | 74 | 118 | 0.887 s, 0.705 s |
| 5040 tokens | 5073 | 3089 | 118 | 0.730 s, 0.713 s |

The compressed comparison enables byte-level verification: full target prefix,
overwritten tail KV, every recurrent/speculation row and the accepted counter
match their original values after each transaction. The rollback allocation is
311,427,076 bytes (about 297 MiB), mostly recurrent state; the production path
does not clone the reused prefix KV. The verification run additionally clones
that prefix for comparison. It reports roughly 0.39–0.41 seconds inside the
three-question transaction. These are two functional-test samples with audit
overhead, not a throughput benchmark or a speedup over a native reference.

Both long-context parent histories finish at exactly 5551 accepted input
tokens (5040 prompt + 511 consumed output tokens). Only the two parents and
their two explicit release requests appear in the archive: private question
tokens are absent. Final active requests, session checkpoints, archive slots,
scheduler reservations and owned blocks are zero; owned test processes and
GPU allocations are also cleaned up.

Evidence is under `artifacts/systemone-active-20261026-v_60_0d9/active-03/client/`
and `active-04/compressed/` (including `resource-proof.json`). Earlier attempts
are development failures, not passing qualification. The short comparison did
not enable byte auditing; the compressed comparison did. Historical GC remains
disabled, and these runs do not qualify multi-parent concurrency, async
refresh, CUDA Graph, DFlash, idle-session branching or long-context accuracy.
