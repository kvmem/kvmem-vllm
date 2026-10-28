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
