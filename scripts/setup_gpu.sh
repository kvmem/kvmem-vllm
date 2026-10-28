#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:---latest}
if [ "$mode" != --latest ] && [ "$mode" != --locked ]; then
  echo 'Usage: scripts/setup_gpu.sh [--latest|--locked]' >&2
  exit 2
fi
uv venv --python 3.12 --allow-existing .venv
if [ "$mode" = --latest ]; then
  # Resolve the current release and its dependencies together. The lock below
  # remains available solely to reproduce historical qualification runs.
  uv pip install --python .venv/bin/python --torch-backend=cu130 --upgrade -e '.[vllm,test]'
  .venv/bin/python -m vllm_kvmem.engine_api --systemone
  exit 0
fi
uv pip install --python .venv/bin/python --torch-backend=cu130 -r requirements-cu130.lock
if [ ! -d .deps/vllm-gguf-plugin/.git ]; then
  git clone https://github.com/vllm-project/vllm-gguf-plugin.git .deps/vllm-gguf-plugin
fi
git -C .deps/vllm-gguf-plugin checkout --detach e2b8ad532b8b5ea175100202c30430c1d2b5e6a8
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda-13 \
  PATH="$PWD/.venv/bin:/usr/local/cuda-13/bin:$PATH" \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps .deps/vllm-gguf-plugin
uv pip install --python .venv/bin/python --no-build-isolation -e .
