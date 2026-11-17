# Serving and sessions

Start with [installation](installation.md) and an explicit JSON profile. The
`vllm-kvmem serve` entry starts native vLLM with the KVMem connector, scheduler
and worker adapters. Merely installing the package does not enable KV offload.

## Profiles

| Setting | Starter | 256K benchmark-derived profile |
| --- | ---: | ---: |
| File | [`serve-qwen-mtp.json`](../examples/serve-qwen-mtp.json) | [`serve-qwen-mtp-256k.json`](../examples/serve-qwen-mtp-256k.json) |
| Logical context limit | 16,384 | 262,144 |
| Concurrent sequences | 2 | 8 |
| Execution | Eager | Compiled CUDA Graphs |
| Native MTP draft tokens | 1 | 4 |
| Selected-history budget, K | 2,048 | 98,304 |
| Growth reserve, G | 2,048 | 16,384 |
| CPU archive budget | 2 GiB | 100,000,000,000 bytes (about 93.1 GiB) |
| GPU allocation fraction | 0.55 | 0.90 |
| Rolling working window | Off | On |

The starter is derived from the actual-model vLLM 0.32.0 validation described in
[compatibility](vllm-compatibility.md). Its resolved batch budget is 2,052 tokens:
the 2,048-token query replay allowance plus two sequences × two speculative
slots. The raw `max_num_batched_tokens` value in a profile is not necessarily the
effective value when request planning is enabled; inspect `--dry-run`.

The larger profile retains the configuration of the
[generation-throughput comparison](benchmarks.md), apart from portable paths,
the public model name and port. It uses more advanced native attention and
capture paths; successful interface checks on a new vLLM release do not qualify
every FlashInfer source overlay. Use the documented benchmark environment for
reproduction and the starter for initial installation checks.

Both examples require you to set `model`, `tokenizer` and `hf_config_path`.
The latter must equal `model` for safetensors. Relative paths are resolved from
the profile's directory, and `~` is expanded. Shell variables inside JSON strings
are not expanded; use the Python configuration snippet in the README.

The bundled default (`serve` without `--config`) is a separate, historical
DFlash2 profile with machine-specific paths. These new examples do not change
that runtime default.

## Tensor-parallel Qwen MTP

The [dual-A800 serving profile](../examples/serve-qwen-mtp-tp2-a800.json)
configures TP=2 and up to 32 requests for Qwen3.8-27B GPTQ-Int4, BF16
activations, FP8 E4M3 KV and native MTP4. Copy it and set `model`, `tokenizer`
and `hf_config_path` to the local checkpoint. The launcher also accepts
`-tp 2` as an override. Set `kvmem.query_replay_cobatch` to `false` so the two workers
enter private Query forwards together. Use static GPU admission, retained
CPU history and synchronous vLLM scheduling. Historical GC is accepted in
`delete` mode only: rank 0's policy proposes and every worker retires the same
logical blocks at the same step ([historical GC under TP](background-history-gc.md#张量并行tp);
CPU-tested only). Each worker owns 12 query heads,
two KV heads and its target, draft and recurrent-state shards.
TP workers use the ordinary publication gate. The launcher sets
`KVMEM_GIL_GATE_DRAIN_OPEN=0` for TP, so a local engine wait does not admit
its own archive publisher ahead of the other rank. The per-session lifecycle
snapshot cache remains available.
Other TP sizes require even, disjoint partitions of both Q and KV heads. The
GPU serving and throughput measurements here cover TP=2.
[Dual-A800 throughput and correctness results](tensor-parallel-a800.md)

```sh
vllm-kvmem serve --config local-tp-profile.json --run-dir runs/tp2
```

K and G count logical tokens for each request. A K of 36,864 and G of 16,384
have the same meaning at TP=1 and TP=2. The workers combine attention masses
across heads, agree on historical token ranges, then map them to their own KV
slots. The scheduler waits for both workers before publishing a rebuilt window.
CPU archive and session checkpoint byte limits apply per worker. If a full
checkpoint pool lets only one worker retain a newer session checkpoint, the
scheduler evicts that session's shards on both workers. A later turn can
recompute the prompt after both evictions finish.

## Understand the memory budgets

`max_model_len` is the request's logical position limit. It is not a promise that
every request retains that many KV tokens on the GPU simultaneously.

`select_budget` (K) and `gen_budget` (G) bound the GPU working context. KVMem
archives historical KV and restores selected blocks into a dense working window.
Query replay, mandatory spans and speculative rows also need capacity; useful
history does not always occupy all K slots. Selection can change model outputs
relative to native full-context attention.

With rolling disabled, planned single-turn output must fit within G. With rolling
enabled, G is a growth reserve between working-window refreshes, rather than the
maximum response length. Native vLLM still enforces input plus requested output
against `max_model_len`. See [rolling long output](rolling-long-output.md).

`cpu_read_cache_bytes`, session checkpoints and other buffers are separate
budgets. A 2 GiB archive setting is not a 2 GiB process-RAM limit. Host memory is
also needed for model loading, indices, transfer buffers and runtime overhead.

`gpu_memory_utilization` controls an allocation budget, not a precise
`nvidia-smi` reading. Keep operating headroom and measure actual usage under load.
The benchmark's near-full settings were tested on a 96 GB GPU and approximately
125 GiB of host RAM; increasing only this fraction does not make a configuration
portable to another GPU.

## A first HTTP request

For opt-in CPU/SSD history storage and incremental GPU admission, see
[dynamic memory implementation](dynamic-memory-implementation-20261106.md) and
the [dynamic MTP example](../examples/serve-qwen-mtp-dynamic.json). The
default incremental path accepts eager MTP requests with synchronous session imports and retains native request
slots while swapping exact KV/state bytes. It preserves K/G and refresh rules;
the default admission mode and existing profiles remain static. Archive,
checkpoint and I/O byte limits are separate from process RAM usage.
The [SSD checkpoint example](../examples/serve-qwen-mtp-dynamic-ssd.json)
adds a separately bounded CPU staging pool and lazy SSD backing for paused
requests. Session namespace ownership survives a swap; canceled session requests
restore their pages before the existing terminal P/M capture and release path.
See [phase 2 implementation and qualification](dynamic-memory-phase2-20261106.md)
for the storage bounds, supported combinations, and measured results.

The default-off `kvmem.request_swap_layerwise` integration permits incremental
admission with CPU archives, asynchronous refresh, physical prefill coalescing,
and eager or `compiled_graph` MTP execution. It requires the sequential,
optimized, native-batch layerwise Query flags plus per-layer prefill pressure
and single assembly. It rejects inactive controllers, rolling decode,
original-RoPE, and no-probe variants. Keep Active/Gen and recall parameters
unchanged when comparing allocation modes. Before releasing native pages, the
worker drains accepted publications and waits for the request's private refresh
and assembly owners; restore checks retained per-layer maps, histories and
window generations. This does not remove the existing KV assembly costs.
`scripts/qualify_layerwise_incremental.py` freezes complete-input serial and
forced-swap checks; support must be qualified on the chosen execution profile,
and byte restoration alone does not prove equal final answers or throughput.

The README's `/v1/chat/completions` example is a short smoke check for the starter.
That profile has rolling disabled, so native chat remains available as a separate
entry. A chat response does not demonstrate explicit query selection or long
history reuse.

Use token-ID `/v1/completions` requests for explicit query coordinates and for the
rolling profile. The following short example uses the same tokenizer as the
server and prints the streamed completion. It works with either example profile
after startup; `TOKENIZER_DIR` should point to your downloaded tokenizer.

```sh
.venv/bin/python - <<'PY'
import json
import os
import urllib.request
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    os.environ["TOKENIZER_DIR"], local_files_only=True
)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain KV caching in three sentences."}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
payload = {
    "model": "kvmem",
    "prompt": prompt,
    "session_id": "token-demo-1",
    "cache_salt": "local-demo",
    "kvmem_query_span": [0, len(prompt)],
    "max_tokens": 128,
    "temperature": 0.6,
    "stream": True,
    "stream_options": {"include_usage": True},
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=300) as response:
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(event["error"])
        for choice in event.get("choices", []):
            print(choice.get("text", ""), end="", flush=True)
print()
PY
```

Here the entire **short** prompt is the query. For a long history, set
`kvmem_query_span` to the relevant query's half-open `[begin, end)` token interval
inside the complete input. Do not mark a 229K history as one atomic query: query
replay, including its aligned tail, must fit `query_replay_budget` and G.
Token spans use logical positions in the original input, not character offsets
or positions within the selected GPU window. Invalid plans fail explicitly.

## Experimental Query selection without a private replay

The optimized sequential path can select historical KV directly from each
target attention layer's fresh Q during the public Query forward:

```sh
vllm-kvmem serve --config /path/to/profile.json --run-dir /path/to/run \
  --layerwise-query-diagnostic sequential --layerwise-query-optimized \
  --layerwise-query-native-batches --layerwise-query-no-probe
```

This opt-in mode omits the private Query forward, its state checkpoint/restore,
and the preliminary global KV selection and assembly. It fixes the window's
token extent before the public forward, then installs each layer's selection
before that layer consumes KV. Per-layer history maps remain distinct. The first
target full-attention layer supplies the shared append frame and MTP's history
selection; MTP's own archived KV is assembled before drafting begins.

The profile must enable `async_refresh`, retain historical KV (disable
`repository_gc` and `historical_gc`), and disable `guided_query_tokens` and
`query_history`. Sessions require `session_freeze_prompt`: restoring that saved
prefix between requests remains necessary. Explicit Query spans and the existing
replay/growth budgets still apply. Partial blocks must be mandatory, or the
entire history must fit; optional partial blocks are rejected because their
score-dependent token count would change the native window geometry.

Omit `--layerwise-query-no-probe` to retain the existing sequential replay path
and its global MTP selection. For a comparison that isolates replay removal,
also pass `--layerwise-query-mtp-selection layer0` to that replay control. Changing
MTP's history selection can change draft acceptance and sampled outputs, so
equivalence must be checked on the intended workload. The default remains the
existing replay path.

`scripts/benchmark_layerwise_no_probe.py` freezes the local runtime and full
local MAB fixtures, waits for the GPU to become available, and qualifies original
replay, MTP-matched replay, and no-probe runs. It compares Q fingerprints,
per-layer selections, output token IDs and frozen-prefix restoration before
running symmetric performance trials with the diagnostic checks disabled.

## Continue a session

Send the complete accumulated token history on each request, including accepted
assistant output and subsequent tool or user messages. Keep the same
`session_id` and `cache_salt` for that conversation; use different session IDs
for independent conversations. The server checks exact token-prefix identity
before reusing state. An ID alone does not prove a cache hit.

The session namespace is `(cache_salt, session_id)` within one worker/model
lifetime. Same-session work is serialized; different sessions can be batched.
Changing tokenization or an earlier part of the prompt can force recomputation.
Use token IDs returned by the server when exact continuation matters rather than
assuming that re-tokenizing displayed text preserves the sequence.

Session IDs and cache salts are identifiers, not authentication. The direct
examples bind to loopback. Applications exposing the service need their own
authentication and trusted namespace assignment. The older
[multi-user gateway](multi-user-serving.md) and
[elastic turn API](elastic-serving.md) have separate configuration and API
contracts; their examples are not interchangeable with direct `serve` profiles.

Session state is retained in memory, subject to capacity and cleanup policy.
It is not a durable cross-worker or restart checkpoint. See the
[support boundaries](support.md) before relying on persistence or concurrency
behavior outside the tested paths.

### Finish a session

Release a conversation when it is finished, using the same session namespace:

```sh
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kvmem",
    "session_id": "token-demo-1",
    "cache_salt": "local-demo",
    "prompt": [1],
    "max_tokens": 1,
    "kvmem_reselect": "off",
    "kvmem_mandatory_spans": [],
    "kv_transfer_params": {"kvmem_release_session": true}
  }'
```

This is the current close protocol: a one-token input and one-token output limit,
with the release flag and no query span. Cleanup follows the normal ownership
fences, and the close request does not retain a replacement checkpoint. Send it
after completing or cancelling outstanding work for that conversation.

The 256K profile enables coordinated history retention. When all 32 retained
session slots are occupied, new sessions can wait for capacity instead of
silently discarding an older history. A completed generation does not itself
close its retained session. Explicit release prevents a sequence of unrelated
conversations from filling this pool; increasing `session_capacity` also needs
corresponding host-memory and repository capacity.

## Observe and troubleshoot

Use `/health` for readiness, `/v1/models` for the served model ID and `/metrics`
for native vLLM counters. The new run directory records the resolved launch plan
and capture diagnostics. Run directories are generated data, not configuration
templates, and are ignored by Git.

- **Missing local model or tokenizer:** update the three model paths and repeat
  `--dry-run`; it checks the quantized model's `config.json` too.
- **Token-ID prompt required:** use `/v1/completions` with integer IDs for the
  rolling profile, or use the non-rolling starter for a short native chat check.
- **Query exceeds the replay budget:** check token coordinates and the complete
  replayed suffix before changing memory budgets. Inputs are not silently cropped.
- **CUDA, kernel or source-overlay rejection:** check the recorded
  [runtime compatibility](vllm-compatibility.md), CUDA toolkit and model hardware
  requirements. Do not infer GPU compatibility from an interface-only check.
- **Memory pressure:** measure both host and device memory, reduce concurrency
  and size K/G and archive budgets together. The 256K benchmark profile is not
  intended as a universal first-launch configuration.

For a report, include the source commit, dependency versions, model revision,
sanitized profile, input/output lengths and relevant error text. Remove local
paths, credentials and private request content before sharing logs.
