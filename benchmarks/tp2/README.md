# TP=2 decode throughput harness (2×A800)

The harness behind the 16 × 32K tensor-parallel decode measurement: 718.1 decode tokens/s on
2×A800-80GB at commit aac5998, DFlash2 + combined layerwise selection, K=128K / G=32K.
It is the A800 copy that produced that result, except that the CUDA toolkit and host compiler
paths can be overridden (`BENCH_CUDA_HOME`, `BENCH_CC`, `BENCH_CXX`).

## Base directory

`benchmark.py` resolves everything relative to its own directory. Copy this folder to a base
directory and add:

| Path | Contents |
| --- | --- |
| `runtime/` | a kvmem-vllm checkout with the native extension built in `src/vllm_kvmem` |
| `venv/` | the Python environment (`requirements-cu130.lock`) |
| `models/` | target model: `sokada4/Qwen3.8-27B-GPTQ-Int4` |
| `draft-w4a16/` | DFlash2 draft: `syvai/Qwen3.8-27B-DFlash2-W4A16` |

The harness creates `cache/`, `runs/` and `logs/`.

## Run

```sh
BENCH_CUDA_HOME=/path/to/cuda-13.0 ./run_16x32k.sh a800-16x32k
python summarize.py runs/a800-16x32k
```

`run_16x32k.sh` starts one server (TP=2), prepares 16 sessions of 32,768 prompt tokens,
decodes 64 warmup tokens, then two measured rounds of 2,048 tokens per session, and stops the
server. A run takes about 15 minutes after compile caches are warm.

Files: `benchmark.py` (server launch from `runtime/benchmarks/gate/agent.json` via
`vllm_kvmem serve --dry-run`, request client), `concurrency_workload.py` (prepared-context
rounds), `entry.py` (vLLM entry point with per-rank graph evidence), `pure_decode_workload.py`
and `copy_check_workload.py` (other modes: `--pure-decode`, `--copy-check`), `summarize.py`.
