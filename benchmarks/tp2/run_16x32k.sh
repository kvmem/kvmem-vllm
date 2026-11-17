#!/bin/bash
# 16 concurrent 32K-token sessions, 2xA800-80GB, TP=2: DFlash2 (1 draft token, compiled graphs)
# + combined layerwise selection on the async refresh lifecycle, K=128K select / G=32K generation.
# Measured 2026-11-17 at aac5998: 723.1 / 713.1 decode tokens/s (median 718.1), 37.1 ms/step.
#
# usage: run_16x32k.sh LABEL   (run from the base directory that holds benchmark.py; see README.md)
# Writes runs/LABEL/ and logs/LABEL.log. The run directory must not exist yet.
set -uo pipefail
cd "$(dirname "$0")"
L=${1:?usage: run_16x32k.sh LABEL}
mkdir -p logs
venv/bin/python benchmark.py --label "$L" --tp 2 \
  --draft-method dflash --compiled-dflash --dflash-async --draft-tokens 1 --strategy layerwise \
  --capture-sizes 2,4,8,16,32 --seqs 16 --max-submit 16 --max-len 221184 \
  --select 131072 --gen 32768 --kv-gib 56 --native-max-mib 6144 \
  --prefill-chunk 1024 --physical-chunks 1 \
  --cpu-history-gib 64 --cpu-per-request-gib 8 --checkpoint-gib 44 \
  --sweep-cases 32768:16 --decode-warmup 64 --decode-tokens 2048 --reps 2 --step-marks \
  --env KVMEM_NATIVE_SCORE_STACK=1 --env KVMEM_NATIVE_WINDOW_FRAME=1 \
  --env KVMEM_DFLASH_SUBMIT_AFTER_DRAFT=1 --env KVMEM_DFLASH_STAGE_BEFORE_DRAFT=1 \
  --env KVMEM_FA2_VERIFY_GRAPH=1 \
  "${@:2}" > "logs/$L.log" 2>&1
echo "exit $? $(date +%T)"
