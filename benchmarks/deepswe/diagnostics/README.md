# Experimental DeepSWE diagnostics

These scripts preserve a specific four-agent performance investigation. They
are **reference-fixture diagnostics**, not a general benchmark or a one-command
DeepSWE setup. The frozen experiment inputs are not distributed in this
directory, so a repository checkout alone cannot run the request replay.

Only syntax and CPU-side input checks have been validated. **Neither the HTTP
inference workload nor the GPU assembly component has been GPU-validated.**
The replay and assembly scripts require an explicit `--execute` to perform GPU
work. By default they only validate local inputs; they do not contact a server,
initialize CUDA, or create the requested output directory. The fixture exporter
is always CPU-only. None of these scripts starts, stops, or configures a server.

| Script | Purpose | Required external inputs |
|---|---|---|
| `prepare_fixture.py` | Reconstruct exact saved prompt token IDs and verify each source turn's token SHA and request plan. | The original frozen experiment bundle, its bridge/protocol helpers, saved responses and local tokenizer. |
| `diagnose.py` | Replay three fixed decode peers while a fourth request performs cold prefill and pressure refresh. | Exported fixture, the matching frozen client runtime, and the actual server profile. Execution additionally requires an isolated localhost server and its local capture files. |
| `benchmark_generic_assembly.py` | Alternate generic assembly implementations over the same fully published CPU archive and native GPU page mapping. | At least two compatible frozen runtime directories, their compiled native extension, and the included selection metadata. |

Python dependencies must match the supplied frozen runtime. The HTTP client
uses `httpx`; fixture export also uses `transformers`, the frozen bridge's
dependencies and its protocol shared library. GPU component execution requires
the runtime's compatible PyTorch, Triton, vLLM and native extension. These
experimental scripts do not install dependencies or fetch model weights.

## Exact-token replay

The exporter targets `kvmem-5090-four-agent-regression-20261031-r2`, with this
external bundle structure:

```text
<frozen-bundle>/
  runtime/src/vllm_kvmem/
  runtime/scripts/export_deepswe_capacity_history.py
  agent-runtime/benchmarks/deepswe/claude_bridge.py
  live-baseline/
    run.json
    profile.json
    bridge/requests.jsonl
    bridge/<task>/<turn>/{request,plan,response,native}.json
    protocol/protocol.so
```

The tokenizer path in that saved profile must resolve locally. These raw
requests, model outputs, token fixtures and task workspaces are intentionally
not included here. Export outside formal measurements, into a new directory:

```bash
python benchmarks/deepswe/diagnostics/prepare_fixture.py \
  --source /path/to/frozen-bundle \
  --output /path/to/new-fixture
```

The fixed source selection contains awilix turn 5 (39,844 input tokens), koota
turn 9 (39,070), textual turn 10 (38,442), and superjson turn 25 (67,491).
Reconstruction replays 49 saved bridge turns and checks their original token
hashes and plans. It uses recorded accepted output IDs without padding,
cropping, or re-encoding assistant text. Exported files contain task content;
keep them outside the published source tree.

Validate the exported fixture and profile without contacting a server:

```bash
python benchmarks/deepswe/diagnostics/diagnose.py \
  --fixture /path/to/new-fixture \
  --runtime /path/to/frozen-client-runtime \
  --profile /path/to/server-profile.json \
  --capture /path/to/server/capture \
  --output /path/to/new-diagnostic-output \
  --tag old-a
```

Only add `--execute` when the isolated server is ready and its GPU work is
intended. Run the client on the same host as the server: capture correlation
uses the same monotonic clock. The client requires an idle server without
retained sessions, and only releases sessions from its own namespace.

The required profile is K=36,864, G=12,288, MTP=4, query budget=1,856, assembly
chunks=4, four GPU request slots and no weight/MLP offload. Other settings must
also match the frozen experiment; passing the checked fields alone does not
establish equivalence. The scripts do not change this configuration.

The workload primes three sessions, then resumes each with its exact source
prefix plus the recorded first continuation token. Each peer generates 8,192
tokens with greedy sampling and `ignore_eos=true`. After observed three-peer
decode steps, the client submits the 67,491-token cold target for one output
token. Query scoring is explicitly disabled while original semantic pins and
locality are retained. This isolates pressure refresh from cold Query replay.

The expected diagnostic shape is two target pressure refreshes, no private
Query forwards, complete output quotas and successful own-session cleanup.
These expectations remain unverified on a GPU and deliberately fail loudly if
the runtime behaves differently. Recorded batch-shape histograms and
three-decode coverage distinguish actual GPU batching from HTTP stream
liveness. Peer output counts at the target's start/end are also retained.

Generated peer contents may differ across runtimes despite identical input IDs
and output budgets. There are no tool calls or task-quality scores. Do not
report this diagnostic as live-agent throughput or as semantic correctness.
Timing profiles keep the byte oracle disabled. A separate correctness run may
enable `verify_session_restore=true`; the client then checks the server's
`verified_rolling_pressure.jsonl` target/MTP, recurrent and draft byte proofs.
That verification adds synchronization and must not be mixed with timing runs.

## Generic KV assembly component

The included `pressure-selection.json` contains selection coordinates and
timing metadata from the first formal pressure event, not prompt text or model
KV. Its provenance uses artifact-relative identifiers and preserves the
original source hashes; the referenced full logs are not distributed here.

Validate runtime compatibility without importing Torch or initializing CUDA:

```bash
python benchmarks/deepswe/diagnostics/benchmark_generic_assembly.py \
  --runtime old=/path/to/old/runtime \
  --runtime current=/path/to/current/runtime \
  --output /path/to/new-component-output
```

Each runtime path must contain `src/vllm_kvmem/` and its frozen native extension.
The first runtime supplies the byte reference. Additional `--runtime label=path`
arguments are accepted. Archive, codec, native extension and kernel helpers
must have identical hashes: the test isolates `window_assembly.py` changes and
rejects incompatible helpers rather than silently combining versions.

Explicit `--execute` enables GPU allocation and assembly. End other GPU work
before executing; the script's advisory lock alone cannot detect every other
GPU owner. It never takes ownership of or stops another workload.

Geometry is fixed to the experiment: 17 FP8 attention layers, 4 KV heads,
dimension 256, 32-token CPU records, native pages of 1,616 tokens, K36K/G12K,
no raw-K GPU cache and no resident-V reuse. The selected history contains
49,088 tokens; 23,136 selected tokens remain unchanged and 13,728 are rebuilt,
transferring 477,954,048 bytes. **KV payloads are deterministic synthetic FP8
bytes, not saved model KV.**

The default matrix is `--chunks 4,7,16 --read-workers 1,8`. Here read workers
means the CPU archive's gather worker count; its I/O executor stays at two.
Chunks 4 and 7 remain below the 8 MiB parallel-gather threshold; chunk 16 crosses
it. Actual gather task counts are checked. One ready archive and destination
arena are shared across arms. Publication, allocation, compilation warmup,
seed resets and byte checks are outside measured samples.

The byte oracle covers the entire physical K/V arena, including unchanged
records, null/unused pages and final-page padding. It checks equality with the
reference generic assembler, and that the reference preserved unwritten
slots. This is an implementation-equivalence check, not a model-quality claim.

Normal timing separates host submission, remaining completion wait,
main-thread CPU, process CPU and per-thread CPU tick deltas. Optional
`--instrument-stages` adds host-stage call counts and wall/current-thread CPU
timing for archive reads, reader waits, upload, per-layer submission, metadata,
staging fences and final completion. It has measurement overhead and must be
reported separately. Nested inclusive spans must not be summed; CUDA event
spans include host gaps, and completion wait is not pure DMA duration.

For two arms, estimated persistent GPU arenas occupy about 6 GiB, the CPU
archive 1.6 GiB, and pinned staging 115 MiB, plus setup and execution scratch.
These are geometry estimates, not measured resource results. Final performance
claims still require matched, completed live DeepSWE runs and a separate
correctness assessment.
