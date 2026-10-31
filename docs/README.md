# Documentation

Start with the [project README](../README.md). The guides below describe the
current public entry points. Dated reports retain the settings and evidence of
their own experiments; they are not default installation instructions.

## Get started

| Guide | Use it to |
| --- | --- |
| [Installation](installation.md) | Install dependencies, obtain the model/tokenizer and validate a profile. |
| [Serving and sessions](serving.md) | Start a server, submit chat or token-ID requests, and size memory budgets. |
| [Support matrix](support.md) | Check model, GPU, API and runtime boundaries. |
| [Benchmarks](benchmarks.md) | Read real-agent observations, historical generation results and their measurement limits. |
| [Development](development.md) | Build the host core and run focused Python/C++ checks. |
| [Contributing](../CONTRIBUTING.md) | Prepare a reproducible bug report or a scoped change. |

## Advanced features

| Topic | Notes |
| --- | --- |
| [vLLM compatibility](vllm-compatibility.md) | Interface checks, supported adapters and actual-model validation of native SystemOne. |
| [Rolling long output](rolling-long-output.md) | Generation across bounded working-window refreshes; requires token-ID requests. |
| [Graph scoring configuration](graph-scoring-v2-configuration.md) | Explicit graph-policy choices and validation rules. |
| [Query replay](query-replay-20261022.md) | Implementation and measured behavior of complete query replay. |
| [Elastic serving](elastic-serving.md) | A separate authenticated turn gateway, with its own profiles and API. |
| [Multi-user serving](multi-user-serving.md) | The legacy direct multi-user middleware and its distinct support boundaries. |

## Reports and history

Files containing dates are research records, benchmarks or design notes. Read
their configuration and limitations before applying a result to another model,
context length or deployment. Referenced `artifacts/` directories are local
experiment outputs and are not part of a source checkout.

- [Latest public generation-throughput summary](benchmarks.md): eight recorded
  long histories with near-full device memory; portable numeric evidence included.
- [Earlier DeepSWE campaign](deepswe-qps-256k-20261027.md): separate live-task and
  replay experiments, with different profiles and memory budgets.
- [Engine integration notes](engine-integration.md) and
  [prompt-session notes](prompt-sessions.md): earlier implementation contracts
  and qualification records; use [serving](serving.md) for current entry points.
- [Historical guide snapshots](archive/README.md): the old README, installation,
  development and support pages, preserved during the public-documentation refresh.

## Project relationships

[KVMem](https://github.com/kvmem/kvmem) describes the memory approach;
[KVMem-QW3](https://github.com/kvmem/kvmem-qw3) integrates it with a standalone
inference runtime. This repository integrates it with native
[vLLM](https://github.com/vllm-project/vllm). Features qualified in one integration
are not automatically supported in another. Imported source provenance is
recorded under [`third_party`](../third_party).
