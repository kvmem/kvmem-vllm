# KVMem for vLLM

**Serve long-context requests with CPU KV history and a bounded GPU working set.**

KVMem for vLLM brings [KVMem](https://github.com/kvmem/kvmem)'s memory management to vLLM's batched inference engine. It archives historical KV on the CPU and restores selected history into GPU memory, leaving room for more concurrent long-context requests. vLLM continues to run the model, attention kernels, scheduling infrastructure, sampling and streaming API.

[Quick start](#quick-start) · [Documentation](docs/README.md) · [Benchmarks](docs/benchmarks.md) · [Contributing](CONTRIBUTING.md)

![Native vLLM and KVMem: eight requests merged into one terminal per side, with measured generation throughput and GPU memory](docs/assets/deepswe-generation-throughput.gif)

Recorded output and GPU memory samples, looping at the same 2.5× playback speed on both sides. **3.70× measured generation throughput.** [Measurement conditions](docs/benchmarks.md).

## Why KVMem?

Long agent histories consume GPU memory even between turns. KVMem separates retained history from the context that is actively computed:

- **CPU history:** retain archived KV and session checkpoints outside GPU memory.
- **Bounded GPU context:** select historical blocks and reserve space for new tokens.
- **Multiple requests:** reuse vLLM's batching with explicit ownership and restoration of each request's state.
- **Session continuation:** reuse an acknowledged prefix when a later turn has the same session and token history.

Selection changes the model's active attention context. This is not a guarantee of identical outputs or task quality relative to native full-context attention. The current integration is experimental and focused on a single NVIDIA GPU, text-only Qwen3.5-family hybrid models and FP8 KV; see the [support matrix](docs/support.md).

## Measured generation throughput

On one RTX PRO 6000 Blackwell 96 GB, with a 256K context limit and eight real DeepSWE histories of approximately 229K input tokens each:

| | Native vLLM | vLLM + KVMem |
| --- | ---: | ---: |
| Generation throughput | 126.2 tokens/s | 466.9 tokens/s |
| GPU memory used during generation | 94.1 GiB | 93.5 GiB |
| Output per request | 4,096 tokens | 4,096 tokens |

**3.70× aggregate generation throughput** in this measured batch. Each side generated **32,768 tokens in total**. The metric includes waiting and recomputation during continuation, and excludes the initial history-preparation phase. This is a single workload comparison, not a maximum-QPS estimate or a completed-task quality result. The benchmark uses a larger configuration than the starter below. See [conditions, configuration and evidence](docs/benchmarks.md).

## Quick start

The starter uses native MTP, eager execution, a 16K logical context limit and two concurrent sequences. It is intended for a first working server. The tested NVFP4 checkpoint requires NVIDIA Blackwell hardware; the recorded setup used a 96 GB GPU. This does not establish a minimum VRAM requirement for other cards.

Prerequisites: Linux, Python 3.12+, a C++17 compiler, a compatible CUDA toolkit with `nvcc`, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). See [installation](docs/installation.md) for model downloads, dependency choices and troubleshooting.

### 1. Install from source

```sh
git clone https://github.com/kvmem/kvmem-vllm.git
cd kvmem-vllm
bash scripts/setup_gpu.sh --latest
```

This resolves the current vLLM release with its dependencies and checks the required interfaces. A passing interface check is not a GPU performance qualification. The starter's recorded GPU validation used vLLM 0.32.0; the throughput comparison above used 0.30.0. See [vLLM compatibility](docs/vllm-compatibility.md).

### 2. Configure your model paths

Download the [NVIDIA NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) and the [Qwen tokenizer](https://huggingface.co/Qwen/Qwen3.8-27B) using the pinned instructions in [installation](docs/installation.md). You do not need the BF16 model weights to obtain the tokenizer.

Set these variables to the downloaded directories:

```sh
export MODEL_DIR=/absolute/path/to/Qwen3.8-27B-NVFP4
export TOKENIZER_DIR=/absolute/path/to/Qwen3.8-27B-tokenizer

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

`hf_config_path` must remain the quantized model directory so vLLM reads its original quantization metadata. `local-profile.json` is ignored by Git. Use this explicit profile: bare `serve` still selects the historical, machine-specific DFlash configuration.

### 3. Inspect and start the server

```sh
.venv/bin/vllm-kvmem serve --config local-profile.json \
  --run-dir runs/quickstart --dry-run

.venv/bin/vllm-kvmem serve --config local-profile.json \
  --run-dir runs/quickstart
```

The dry run validates configuration and local paths without starting the GPU engine or creating the run directory. For each actual launch, use a new run directory. First startup may compile CUDA extensions.

### 4. Send a request

In another terminal, wait for `http://127.0.0.1:8000/health` to return HTTP 200, then run:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kvmem",
    "session_id": "quickstart-1",
    "cache_salt": "local-demo",
    "messages": [{"role": "user", "content": "Explain KV caching in three sentences."}],
    "max_tokens": 128,
    "temperature": 0.6,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": true
  }'
```

This is a short chat smoke check. Explicit query selection and the rolling long-context profile use token-ID `/v1/completions` requests; see [serving and session usage](docs/serving.md).

## Choose a configuration

| Profile | Purpose |
| --- | --- |
| [`serve-qwen-mtp.json`](examples/serve-qwen-mtp.json) | First server: eager MTP1, 16K context, two sequences, 2 GiB CPU archive budget. |
| [`serve-qwen-mtp-256k.json`](examples/serve-qwen-mtp-256k.json) | Benchmark-derived 256K / eight-request configuration with compiled MTP4, 98,304 selected tokens and 16,384 generation reserve. Requires substantially more host/GPU memory and the matching runtime capabilities. |
| [`serve-default.json`](examples/serve-default.json) | Historical bundled DFlash deployment. Contains local paths and requires a separate draft checkpoint. |

GPU allocation fractions are not physical VRAM readings. Size the GPU working set, CPU archive and concurrency together; monitor actual memory under your workload. The 256K measurement used nearly all available GPU memory and is not a universal memory setting.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Installation](docs/installation.md) | Model files, dependencies, dry runs and startup problems. |
| [Serving](docs/serving.md) | Configuration, token-ID requests, session identity and long-context usage. |
| [Support](docs/support.md) | Tested models, hardware, runtime boundaries and current limitations. |
| [Benchmarks](docs/benchmarks.md) | Generation-throughput definition, measured results and reproducibility limits. |
| [Development](docs/development.md) | CPU-only builds, focused checks, native tests and project layout. |
| [Documentation index](docs/README.md) | Advanced features and dated research reports. |

The previous README's experiment log is retained in the [historical archive](docs/archive/readme-before-public-guide.md).

## Related projects and license

- [KVMem](https://github.com/kvmem/kvmem): the underlying long-context memory approach.
- [KVMem-QW3](https://github.com/kvmem/kvmem-qw3): a standalone inference integration; its hardware and API support are separate from this vLLM plugin.
- [vLLM](https://github.com/vllm-project/vllm): the inference and serving engine used here.

This project is licensed under [Apache-2.0](LICENSE). Imported source retains its licenses and provenance under [`third_party/qw3`](third_party/qw3/PROVENANCE.json) and [`third_party/vllm`](third_party/vllm/PROVENANCE.json). Model checkpoints and benchmark datasets have their own licenses and are obtained separately.
