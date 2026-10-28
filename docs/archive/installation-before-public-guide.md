> Historical guide archived on 2026-10-28. The observations and corrections below
> describe their recorded experiments, not the current installation or support matrix.
> See the [current guide](../installation.md).

# Installation and experimental serving

The default direct agent deployment, selected by the user on 2026-10-20, is
NVFP4 + FP8 target KV, DFlash2 k=7, FULL CUDA Graphs, GPU memory utilization .80,
eight active requests, K36864/G16384, synchronous Query/pressure restoration and
all four attention-maintenance optimizations. Its CPU-only history budget is
100,000,000,000 bytes; prefix caching is disabled. The editable configuration is
`examples/serve-default.json`; its paths refer to this host's external models.

```sh
PYTHONPATH=src python -m vllm_kvmem serve --run-dir runs/agent-01 --dry-run
```

Omit `--dry-run` to start the default. After installing the current package,
`vllm-kvmem serve --run-dir runs/agent-01` uses the same bundled profile without
depending on a checkout's examples directory or benchmark execution records.
To change paths/settings, copy the example and pass `--config local-profile.json`.
The default explicitly enables the experimental DFlash Query-planning option
within its qualified synchronous/full-graph scope. Existing explicit MTP and
other profiles retain their settings. See the [agent results and limits](../dflash-agents-20261020.md):
the larger pool improves this single window, with extra GPU memory and no
completed full-task quality evaluation.

For the earlier original-MeanK integration, see
[MeanK setup and implementation](../meank-integration-20261013.md).
The [elastic API](../elastic-serving.md) also supports full-context native MTP
offloading as a separate mode. See [measured results](../meank-throughput-20261013.md)
for their different performance and qualification boundaries.

The installed development wheel for this update is retained locally at
`artifacts/meank-20261013/dist/`. Reinstall when replacing another build of the
same `0.1.0.dev1` version. The eager/DFlash2 profiles described below are the
earlier direct-serving entry.

This is a pinned development integration, not a production release. The actual
model gates for the direct eager entry cover Qwen3.8-27B GGUF Q8_0 and NVIDIA ModelOpt NVFP4/FP8
mixed safetensors, target FP8 e4m3, DFlash2 BF16
draft/native sliding KV, TP=1, eager synchronous execution and text-only requests.
The kernel/source adapters require vLLM 0.30.0. They do not rewrite installed vLLM
files. Broader model, quantization, KV codec and engine versions need separate
adapters and qualification; installing the package does not establish support.
The [NVFP4 report](../nvfp4-throughput-20261013.md) records the exact checkpoint,
tested workloads, kernel choices and remaining output-quality limits.

## Install from the independent repository

On the qualified Linux/CUDA 13/RTX PRO 6000 Blackwell setup:

```sh
cd ~
git clone https://github.com/Di-Chai/kvmem-vllm.git
cd kvmem-vllm
bash scripts/setup_gpu.sh
```

The setup script installs the pinned Python/GPU requirements and the independent
GGUF plugin revision recorded in `requirements-cu130.lock` and the script. It uses
`uv` and expects CUDA at `/usr/local/cuda-13`. It builds for SM120. Model files are
external and must be provided separately. A different CUDA/GPU environment needs
its own dependency/build validation; the host-only core can be installed without
torch or vLLM using `uv pip install '.[test]'` in a Python 3.12 environment.

The entry point `vllm.general_plugins:kvmem` is registered by package installation.
Discovery alone is passive. The explicit launcher supplies the pinned runner,
metadata, bounded scheduler and public KV connector settings needed for serving.

## Configure and inspect a launch

Copy `examples/qwen-q8-fp8-dflash.json` to a local profile and replace the model
paths. Relative paths resolve against the profile directory; `~` is expanded.
The `tokenizer` and `hf_config_path` folders supply tokenizer/model configuration,
not BF16 target weights. `mm_proj` is required by the pinned GGUF loader even though
this profile enables language-only execution. The draft uses its BF16 safetensors.

For NVFP4, use `examples/qwen-nvfp4-fp8-dflash-multi-user.json` and the
[NVFP4 setup instructions](../nvfp4-throughput-20261013.md). Set
`model_format: "safetensors"`, point both `model` and `hf_config_path` at the
same ModelOpt NVFP4 checkpoint, omit `mm_proj`, and disable `q8_tensorcore`.
The optional `linear_backend_per_quant` mapping selects compatible native
vLLM kernels; availability depends on the installed GPU libraries. The launcher
preserves checkpoint quantization metadata and does not force a uniform FP4
scheme onto mixed-precision weights.

```sh
cp examples/qwen-q8-fp8-dflash.json local-profile.json
# Edit the five model/config paths in local-profile.json.
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json --run-dir runs/session-01 --dry-run
```

The dry run validates local paths and options, prints the exact native argument
vector and environment, and neither imports torch/vLLM nor creates the run
directory. It does not allocate GPU memory or prove that the model fits. The
example preserves K131072/G65536 and B128, uses a 36-GiB shared read cache and a
four-entry/1-GiB CPU prompt-checkpoint pool. These are separate limits, not a total
host-RAM quota. NVMe arena geometry includes both active and retained authorities;
physical disk reservation and shared I/O-byte controls are opt-in public connector
options. The example leaves repository GC and physical reservation disabled because
their serving GPU qualification is still pending.

`q8_tensorcore` defaults to false. Setting it to true enables the experimental
SM120 batched Q8 integer Tensor Core adapter with the measured speed gains and
remaining quality limitations described in `support.md`. It keeps target Q8_0
weights and FP8 KV; it does change floating-point reduction order.

## Start and use the API

```sh
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json --run-dir runs/session-01
```

After reinstalling the updated package, the console command `vllm-kvmem serve`
is equivalent. vLLM owns the API and process lifecycle. The new run directory must
not already exist, so another server's archive or readiness marker cannot be
reused accidentally. `launch.json` records the resolved profile and argument vector;
`capture/` contains the runtime lifecycle/authority evidence and NVMe arena.

The profile defaults to `127.0.0.1:8000` and model name `kvmem`. A normal OpenAI
completion/chat request works without a session. To request exact-prefix reuse,
provide a stable `session_id` body field or native `X-Session-ID` header:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Session-ID: task-01' \
  -d '{"model":"kvmem","messages":[{"role":"user","content":"Explain this repository."}],"max_tokens":128}'
```

A short request may not create a checkpoint. Reuse requires an exact matching
token prefix and a sealed non-final prefill snapshot; a session header by itself
does not prove a cache hit. Continue sending the full intended conversation.
Do not remove old messages or alter task input to manufacture reuse. The serving
gateway must supply an appropriate `cache_salt`/session scope for its clients;
these fields are namespace identifiers, not authentication.

The profile installs `vllm_kvmem.http_sessions.validate_prompt_session` through
vLLM's public `--middleware` option. Session IDs and salts used for reuse are
limited to 1024 characters and rejected with HTTP 400 before scheduler dispatch.
Body/header/`vllm_xargs` precedence matches the pinned engine. Native ASGI host
tests cover rejection, concurrent valid requests and unchanged streaming bodies;
the current real-model entry checks are recorded in the
[2026-10-13 report](../status-20261013.md). Direct custom launches must include the
middleware explicitly to obtain the same validation.

For per-user authentication, bounded admission and server-owned session namespaces,
use [multi-user serving](../multi-user-serving.md). That entry installs its own
validation middleware in place of this single-user guard.

Serving mode distinguishes native warmup (no connector metadata) from the first
real scheduler message and enables accepted capture before that first forward.
It omits the one-off diagnostic tensor snapshot used by the original probe tools.
There is no API-ready/externally-created-marker race. The public profile entry now
has actual-model single-user startup and authenticated multi-user validation from
2026-10-13, including short and 134K exact session continuation. See the
[current qualification report](../status-20261013.md) for the tested configurations,
concurrent load results, and remaining quality and capacity limits.

The public entry also passes an isolated source-package check recorded in
`artifacts/source-build-serving_01`: rebuild the native wheel from an extracted
sdist with no extension binary, install into a fresh environment without torch or
vLLM, and run 73 host checks (2 skipped). Both console help and a real-path console
dry run pass without creating a runtime directory. The source archive includes
the example profile; the installed wheel contains the console entry and all three
licenses. This verifies host installation, not GPU startup or performance.

For measured support and test commands, see `prompt-sessions.md`, `support.md` and
`qualification-20260928.md`. Current agent evaluation remains incomplete.
