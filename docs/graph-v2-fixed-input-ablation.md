# Fixed structured-input graph ablation

This supplementary handoff §9.4 experiment runs four real model profiles on
two frozen turn-1 prompts from the successful structured session qualification.
The prompts retain the original accepted assistant token IDs and tool-use IDs.
Every arm submits each complete prompt to a fresh owner. New generated outputs
never become another arm's input. The previous sync/async qualification remains
the evidence for multi-turn checkpoint restoration.

| Arm | Query propagation | Locality metadata |
| --- | --- | --- |
| v1-attention | Legacy attention weight .2, locality weight 0 | Unknown |
| v2-attention | Alpha .2 | Unknown |
| v2-multi-unknown | Alpha .2, fixed half per operator | Unknown |
| v2-multi-known | Alpha .2, fixed half per operator | Exact frozen Protocol spans |

All profiles use recency pressure, K=36864/G=16384, and identical model,
batch settings, sampling and accepted-attention capture. Removing graph-only
metadata must preserve tokens, Query/replay and mandatory decisions. Thinking
budget 0 and natural EOS are declared smoke settings, not quality tuning.

The opt-in graph exporter accepts at most 8192 nodes and 2 MiB per snapshot.
Its existing eight-record per-owner and 16 MiB total-file limits remain. Export
stays disabled when graph_trace is false. Missing exports fail qualification.

The client reconstructs the native selector through analyze_block_graph and
checks every selected block against bounded_refresh's complete [begin, count]
working-record ranges. It does not rely on the truncated graph summary list.
After all eight real requests pass and release their owners, the analyzer
replays all four variants against the two known-arm snapshots. Seeds, graph
attention rows, masks, candidate counts, budget and mandatory blocks remain
fixed. Original records are hashed and never modified.

The report separates actual own-snapshot selection closure from canonical
counterfactual score/selection changes. Normalization must have a measurable
effect and at least one positive seed must cross a genuine unmasked boundary;
otherwise the result is inconclusive. Selected-page changes are not required.
No gold evidence or answer labels exist: retrieval accuracy and answer quality
stay unknown. Timing includes diagnostic export/replay and is not throughput.

From the repository root, prepare with the intended final absolute location:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -B scripts/benchmark_graph_v2_ablation.py prepare \
  --output /data/chaidi/kvmem-vllm-experiments/graph-v2-fixed-ablation-20261018-r1 \
  --source artifacts/graph-v2-sessions-20261018/sync-r1
```

Run the frozen controller after the GPU is available. The run command starts
four separately owned servers sequentially and never retries a failed arm.

```sh
.venv/bin/python -B \
  /data/chaidi/kvmem-vllm-experiments/graph-v2-fixed-ablation-20261018-r1/frozen/benchmark_graph_v2_ablation.py run \
  --output /data/chaidi/kvmem-vllm-experiments/graph-v2-fixed-ablation-20261018-r1
.venv/bin/python -B \
  /data/chaidi/kvmem-vllm-experiments/graph-v2-fixed-ablation-20261018-r1/frozen/analyze_graph_v2_ablation.py \
  --campaign /data/chaidi/kvmem-vllm-experiments/graph-v2-fixed-ablation-20261018-r1
```

Do not move the campaign after freezing: its source/fixture bindings contain
absolute paths. The freeze retains producer/protocol provenance and hashes all
runtime and input files. No judge API or credential is used.
