# Installation and first request

KVMem is an experimental vLLM integration for bounded GPU working contexts and
CPU-backed KV history. Install it from source, supply model files separately, and
launch it with an explicit configuration. Installing the package alone does not
enable KVMem for an ordinary vLLM server.

## Requirements

| Component | Requirement or tested configuration |
| --- | --- |
| Host development | Linux, Python 3.12+, a C++17 compiler; CMake 3.20+ for the C++ tests |
| GPU serving | A compatible NVIDIA GPU/driver and CUDA toolkit; the supplied installer resolves CUDA 13.0 packages |
| Tested hardware | One RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM; this is the measured setup, not a minimum-memory guarantee |
| GPU runtime | vLLM with the required native interfaces; see the [compatibility matrix](support.md#runtime-and-hardware) |
| Installer | `uv` available on `PATH` |
| Weights and tokenizer | Local, matching model/config/tokenizer directories; model weights are not included |

Model size, active context, concurrent requests, history pools and temporary
buffers all affect GPU and host memory requirements. The 100 GB CPU history
settings in some research profiles are unsuitable for small-memory machines and
are not a whole-process RAM limit.

## Install the GPU package

Run these commands from your checkout:

```sh
bash scripts/setup_gpu.sh --latest
.venv/bin/python -m vllm_kvmem serve --help
```

The installer creates or updates the checkout's `.venv`, installs this package
in editable mode together with vLLM and test dependencies, and runs the native
interface check. `--latest` is also the script's default. It resolves the vLLM
release available at installation time; successful dependency resolution is not
an end-to-end GPU qualification for every future release.

To run the interface check again:

```sh
.venv/bin/python -m vllm_kvmem.engine_api --systemone
```

The `--systemone` check includes the optional native decision API. It requires a
vLLM version that provides that API; ordinary serving on the historical 0.30.0
baseline does not require SystemOne. The [compatibility guide](vllm-compatibility.md)
distinguishes source checks, runtime adapters and actual-model validation.

### Historical locked environment

To reproduce the older dependency set instead of resolving a new runtime:

```sh
bash scripts/setup_gpu.sh --locked
```

This uses `requirements-cu130.lock` with vLLM 0.30.0, then builds the separately
pinned GGUF plugin. This branch of the script expects CUDA at
`/usr/local/cuda-13` and targets SM120. It is intended for that historical
environment; it is not the portable installation path for other GPU families.
It does not run the SystemOne check because that API is absent in the baseline.

## Download the model and tokenizer

The current actual-model measurements use Qwen hybrid GDN models. The commands
below select the measured
[NVIDIA Qwen3.8-27B NVFP4 revision](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/tree/482ca0f3832238542f8f5295dde86b5f22711d80)
and its matching
[Qwen tokenizer revision](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).
The target uses ModelOpt mixed NVFP4/FP8 weights. Model licenses and hardware
requirements apply separately from this repository's license.

Choose local directories outside your source checkout:

```sh
export MODEL_DIR="$HOME/models/Qwen3.8-27B-NVFP4"
export TOKENIZER_DIR="$HOME/models/Qwen3.8-27B-tokenizer"

.venv/bin/hf download nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir "$MODEL_DIR"

.venv/bin/hf download Qwen/Qwen3.8-27B \
  tokenizer.json tokenizer_config.json chat_template.jinja vocab.json merges.txt config.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "$TOKENIZER_DIR"
```

Only tokenizer/configuration files are downloaded from the second repository;
BF16 target weights are not needed for this NVFP4 setup. If you already have
these files, set `MODEL_DIR` and `TOKENIZER_DIR` to their existing locations.
Accepting a safetensors directory does not imply support for every model
architecture; consult the [support guide](support.md) for other configurations.

## Create an explicit configuration

Start with the [native-MTP example](../examples/serve-qwen-mtp.json). MTP uses the
target checkpoint's own prediction head, so this example does not require a
separately downloaded DFlash model. With the environment variables above set:

```sh
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

profile = json.loads(Path("examples/serve-qwen-mtp.json").read_text())
profile["model"] = str(Path(os.environ["MODEL_DIR"]).expanduser().resolve())
profile["hf_config_path"] = profile["model"]
profile["tokenizer"] = str(Path(os.environ["TOKENIZER_DIR"]).expanduser().resolve())
Path("local-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
PY
```

`local-profile.json` is ignored by Git. For safetensors, `hf_config_path` must
match `model` so the native loader sees the checkpoint's own quantization
metadata. Relative paths in a profile resolve against that profile's directory,
and `~` is expanded.

The starter uses the actual-model-tested eager MTP1 configuration: two concurrent
sequences, a 16K logical context limit, K=2,048 and G=2,048. It is a small-window
setup for verifying the service, not the eight-request, 256K compiled-graph
throughput profile. See [serving configuration](serving.md) for the larger
profile and explicit token-ID request planning.

### Inspect before starting

```sh
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json \
  --run-dir runs/first-server \
  --dry-run
```

This validates local paths and options and prints the resolved native command
and environment. It does not import torch/vLLM, allocate GPU memory or create
the run directory. A successful dry run verifies configuration, not model
loading, available memory or numerical behavior.

`serve` without `--config` still selects the bundled DFlash2 k=7 deployment in
`default_profile.py`. That profile uses the original machine's absolute paths,
GPU budget 0.80 and large CPU history pools. Use an explicit local profile on a
new machine. The [historical DFlash guide](dflash-agents-20261020.md) records why
that deployment was chosen.

## Start the service

If your CUDA toolkit is outside the standard locations, make its compiler
available before starting the service. Replace the path with the toolkit
directory that contains `bin/nvcc`:

```sh
export CUDA_HOME=/absolute/path/to/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

The first native CUDA build can take several minutes. To bound compiler
concurrency and host-memory use, optionally set the limits used in the
installation smoke check:

```sh
export MAX_JOBS=4
export FLASHINFER_NVCC_THREADS=1
```

These settings control compilation; they do not change the profile's GPU
context or output budgets.

```sh
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json \
  --run-dir runs/first-server
```

The installed `.venv/bin/vllm-kvmem serve` command is equivalent. Choose a new run
directory for each launch: an existing directory is rejected to prevent reuse
of stale server state. The directory records `launch.json` and the server's
`capture/` diagnostics. The model and engine may compile GPU kernels during the
first startup.

In another terminal, wait for readiness and send a request:

```sh
curl --fail http://127.0.0.1:8000/health

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kvmem","messages":[{"role":"user","content":"Explain what a KV cache stores."}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

For streaming, add `"stream":true` to the JSON body and `-N` to `curl`. The
profile's `host`, `port` and `served_model_name` determine the URL and model
field. The example binds to loopback. This short request verifies ordinary API
use; it does not demonstrate long-history retrieval or a session cache hit.

## Continue a conversation

When the profile enables sessions, provide a stable `session_id` in the request
body or `X-Session-ID` header, and keep sending the complete intended conversation:

```sh
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: example-conversation' \
  -d '{"model":"kvmem","messages":[{"role":"user","content":"Explain CPU KV offloading."}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

A session selects candidate retained state; exact token-prefix checks determine
reuse. A short request or a changed prefix may have no reusable checkpoint.
Session IDs and `cache_salt` identify namespaces, not authenticated users. See
[current serving behavior](serving.md), [historical session checks](prompt-sessions.md),
and [runtime boundaries](support.md#sessions-and-data-lifetime) before integrating
a shared service.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| `Missing local model`, tokenizer or draft path | Edit the explicit profile; the bundled default has machine-specific paths |
| Safetensors config mismatch | Set `hf_config_path` to the same directory as `model`; retain the original model configuration |
| Run directory already exists | Choose a new run directory for this launch |
| Native interface or layout check fails | Record the vLLM version and read [compatibility](vllm-compatibility.md); this is different from a missing model file |
| GPU OOM or host-memory pressure | Reduce concurrency and the relevant pools; monitor actual GPU and host usage, not just the configured budget fraction |
| First startup appears slow | Check compilation progress and ensure the CUDA toolkit and compiler are available |
| Session has no cache hit | Verify the full token prefix, session namespace and completed checkpoint publication |

For CPU-only installation and contribution checks, use the
[development guide](development.md). The superseded installation record is
preserved in the [archive](archive/installation-before-public-guide.md).
