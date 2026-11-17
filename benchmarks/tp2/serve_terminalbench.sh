#!/bin/bash
# Serve the Terminal-Bench model on the 4090 pair: latest 8-concurrency layerwise MTP4 stack
# (native overlay scores, draft seq-lens, full verify graphs), agent GC with thinking capture.
# usage: tbage-serve.sh LABEL AGE_STEPS(0=off)
cd /root/kvmem-tb-age
export BENCH_CUDA_HOME=/root/data/kvmem-vllm-gc/toolchains/cuda130/nvidia/cu13
label=$1; age=$2
extra=(); [ "$age" -gt 0 ] && extra=(--thinking-age-steps "$age")
echo $$ > "logs/$label.pid"   # exec keeps this PID for the harness
exec venv/bin/python benchmark.py --label "$label" --serve-only --tp 2 --draft-method mtp --draft-tokens 4 \
  --strategy layerwise --select 36864 --gen 16384 --seqs 8 --max-submit 8 --max-len 10000000 --prefill-chunk 1024 \
  --kv-gib 10.15 --cpu-history-gib 64 --cpu-per-request-gib 8 --checkpoint-gib 12 --native-max-mib 4096 \
  --gc redundancy --gc-thinking-only "${extra[@]}" --repository-blocks 8192 --session-capacity 10 \
  --keep-reasoning-end --env KVMEM_FA2_VERIFY_GRAPH=1
