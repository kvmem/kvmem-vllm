# Three-tier KV for Qwen3.8 Flash Next (v1)

KVMem bounds each request's attention context to a selected window of K
historical tokens plus G generated tokens. On Flash Next (12 QSA layers plus
the MTP draft layer), QSA then attends to about 2,048 indexer-selected tokens
of that window per row and layer. Version 1 of the three-tier layout keeps each
level of this hierarchy in the memory that suits it:

| Tier | Holds | Moved |
| --- | --- | --- |
| GPU | Per request and QSA layer: `qsa_hot_pages` hot pages caching the groups QSA selected recently, and `qsa_ring_pages` write-ring pages for new tokens. The selector keys of the whole window and the recurrent (GDN) state stay on the GPU. | Every decode step: missing selected 4-token groups are copied in from CPU (residency kernels, clock eviction). |
| CPU | The window itself (K + G tokens of FP8 K/V per layer) in a pinned host pool. | At every reselection the window is rebuilt from the archive; new tokens are written back from the ring every step. |
| SSD | The full history (`archive_backend=nvme`), stored layer-major and unpadded (`archive_layout=layer`). | Read at reselection with O_DIRECT parallel reads into a pinned ring, then DMA to the GPU for assembly. |

Every option is opt-in. Selection, scoring and outputs are the same as with
the window resident on the GPU: attention reads the same bytes.

## Configuration

A profile's `kvmem` block (K128K/G32K example):

```json
"select_budget": 131072, "gen_budget": 32768,
"qsa_split_selector": true, "qsa_streaming": true, "qsa_hot_pages": 3, "qsa_ring_pages": 1,
"archive_backend": "nvme", "cpu_read_cache_bytes": 0, "nvme_max_bytes": 137438953472,
"archive_io_max_bytes": 17179869184, "archive_direct_reads": true, "archive_direct_threads": 8,
"archive_layout": "layer"
```

- `qsa_ring_pages`: one step of one request (a prefill chunk or the atomic
  Query replay) must fit in `qsa_ring_pages * 3008 - 8` tokens. Two ring pages
  allow ~6K-token prefill chunks (`prefill_chunk_tokens` and
  `query_replay_budget` 5952 with `request_planning_batch_tokens` 6019).
- KVMem's per-layer strategies run on the same layout. `strategy_path: "layer"`
  selects per layer at the final Query, the first
  `KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS` layers sharing the global selection
  (a `settings.strategy` entry). `strategy_pressure_scorer: "attention"` or
  `"lift"` (with `prefill_attention_half_life`) keeps K blocks per layer while
  the prompt is read; the attention mass comes from QSA's own sparse softmax
  over its selected tokens, recomputed for the sampled rows
  (serving/qsa_pressure.py). `qsa_layerwise_mtp` chooses whether the draft
  layer follows the global selection or layer 0's.

Implementation: adapters/qsa_streaming.py, adapters/qsa_stream_runtime.py and
csrc/qsa_residency.cu (GPU tier); core/storage.py, core/layer_major.py,
csrc/direct_record_reader.hpp and csrc/layer_major_arena.hpp (SSD tier);
serving/qsa_assembly.py and serving/direct_assembly_reads.py (reselection);
serving/qsa_layerwise.py and serving/qsa_pressure.py (per-layer strategies).

## Results

RTX PRO 6000 Blackwell (96 GB, PCIe Gen5 x16), Samsung 990 EVO Plus NVMe on a
PCIe Gen3 x4 link (3.72 GB/s), Flash Next NVFP4 FP4-Local, MTP3, vendor vLLM
image release-10152026, CUDA graphs. Synthetic review prompts, greedy, end of
turn ignored. Single measurements unless noted.

### Exactness

Single requests produce tokens identical to the same configuration with the
window resident on the GPU and the history in RAM: streaming (K8K/G4K, 1.5K to
24K prompts, eager and graphs), SSD archive with buffered, direct and
layer-major reads, per-layer Query selection, the combined method (lift
pressure with 4 shared layers) and the two-page ring with 5,952-token chunks.
All 2,606 records of a layer-major run passed the whole-record digest check
after being rebuilt from their per-layer groups. With several concurrent
requests the eager reference itself is not reproducible run to run (two runs
of the resident combined-method reference disagree on all 4 concurrent
requests), so concurrent outputs are not compared.

### Reselection from SSD

| Read path | 8K-token reselection | K128K reselection (131K tokens) |
| --- | --- | --- |
| CPU archive (records in RAM, original path) | 246 ms | 5.15 s |
| SSD, original buffered reads | 476 ms | — |
| SSD, O_DIRECT parallel reads, record layout | 75 ms | 0.61–0.87 s (2.62 GB) |
| SSD, O_DIRECT, layer-major unpadded | 50 ms (136 MB) | ~0.7 s |
| SSD, layer-major, per-layer selection | 70 ms | 1.0–1.7 s |

Per-layer 16 KB slices of the record layout read at only 0.96 GB/s (about
60K IOPS on this drive); the layer-major layout turns them into contiguous runs.

### Prefill at K128K/G32K

200K-token prompts, SSD archive, global selection unless noted.

| Configuration | 4 requests | 8 requests |
| --- | --- | --- |
| 2,048-token chunks, 1 ring page | 3,883 tok/s | 3,693 tok/s |
| per-layer Query selection, 4 shared layers | 3,787 tok/s | 3,490 tok/s |
| combined method (lift pressure + 4 shared layers) | 3,283 tok/s | 3,197 tok/s |
| 5,952-token chunks, 2 ring pages | 8,108 tok/s | 7,329 tok/s |

PCIe is not the prefill limit: staging a 160K-token frame for every layer
(about 2 GB per step) adds 16 ms to a ~390 ms step, because the next layer's
copy overlaps the current layer's compute. Small chunks are: a 2,048-token step
runs at about 5,060 tok/s and a 5,952-token step at about 12,400 tok/s.

### Concurrency and decode at K128K/G32K

184K-token prompts (every window reaches K + G), 8K outputs, 2,112-token
batches, SSD archive, global selection. Steady decode counts the tokens of all
requests while every one of them is decoding.

GPU pages (3.27 MB each) on this GPU: 4,361 usable. A resident request needs
13 x 55 window pages + 148 recurrent pages = 863, so at most 5 fit. A three-tier
request needs 13 x (3 hot + 1 ring) + 148 = 200 pages, after 799 pages
reserved for 16 slots' selector keys and staging: at most 17, capped at 16 by
`max_num_seqs`.

| Configuration | Concurrent requests | Steady decode | Decode step | Output tok/s, end to end |
| --- | --- | --- | --- | --- |
| Resident window | 4 | 318 tok/s | 32.4 ms | 125 |
| Resident window (maximum) | 5 | 353 tok/s | 36.6 ms | 129 |
| Three-tier | 4 | 268 tok/s | 33.6 ms | 98 |
| Three-tier | 8 | 464 tok/s | 43.6 ms | 128 |
| Three-tier | 12 | 531 tok/s | 57.6 ms | 131 |
| Three-tier (maximum) | 16 | 652 tok/s | 63.4 ms | 181* |

At the maximum concurrency each layout reaches, the three-tier layout decodes
1.84x faster (652 vs 353 tok/s): 3.2x the requests at a 1.7x longer step.

\* The 16-request run used 12K outputs, so that all 16 requests decode together
after the longer prefill; its end-to-end figure is not comparable with the 8K
rows. It also capped the KV pool at the 4,080 pages 16 requests need
(`kv_cache_memory_bytes`): with the default pool (4,361 pages) only 66 MiB of
GPU memory stayed free, and the final-Query MeanK scoring of several requests
ran out of memory during prefill.

## Limitations and future targets

1. **Per-request decode cost.** At equal concurrency the three-tier layout
   decodes 16% slower (268 vs 318 tok/s at 4 requests). At K128K, 18.6% of the
   groups QSA selects per step miss the 3 hot pages. Targets: an adaptive hot
   set, prefetching the next step's likely groups, a fused plan-and-copy kernel.
2. **Concurrency vs batch size.** Larger prefill batches need more activation
   memory: with the 6,019-token batch the KV pool falls from 4,361 to 2,303
   pages (7 three-tier requests, 2 resident). The 799 reserved pages are sized
   for every slot's full window. Targets: size the selector keys with the
   admitted requests, share or shrink staging, keep the batch for prefill steps
   only. vLLM's memory profile also misses KVMem's transient GPU memory at the
   final Query (MeanK scoring over the whole history): 16 requests need the pool
   capped by hand (`kv_cache_memory_bytes`). Target: reserve that headroom
   automatically or score in bounded chunks.
3. **Recurrent state dominates.** 148 of a three-tier request's 200 GPU pages
   are GDN state (the main copy plus 3 MTP speculative copies). Target: fewer
   speculative copies.
4. **Synchronous reselection.** A K128K reselection stalls the engine for
   0.6–0.9 s (1.0–1.7 s per layer); all requests wait. Target: start the SSD
   reads when the selection is known and assemble over several steps.
5. **Per-layer pressure cost.** Recomputing QSA's attention mass for 64 sampled
   rows per layer costs about 72 ms of GPU and 30 ms of host time per prefill
   step (−15% prefill). Targets: a fused FP8 kernel, an asynchronous commit.
6. **Per-layer selection semantics.** QSA scores every layer with the Query of
   the refresh probe, so all reads can start at once; the full-attention path
   re-scores each layer in a second, layer-by-layer forward. Decode-time
   reselections stay global (no decode attention packets yet). Target: the
   sequential mode, using these selections as prefetch hints.
7. **Sessions.** Multi-turn prompt sessions are not yet supported with
   streaming; the window is rebuilt per request.
8. **Synchronous scheduling.** KVMem's refresh transactions require it; on a
   single request, asynchronous scheduling alone made plain vLLM decode 49%
   faster. Target: asynchronous-safe refresh.
9. **CPU archive path.** Without the SSD tier, reselection still uses the
   original buffered assembly (5.15 s at K128K). Target: the same pinned ring
   and DMA path for records in RAM.
10. **Startup.** A server takes about 6 minutes to start. Reading the 102 GB
    checkpoint takes 35 s (2.9 GB/s; InstantTensor settings do not change it),
    but vLLM spends about 110 s more on loading its 375K tensors and 71 s on
    post-processing (FP4 MoE repacking, the 26.8 GB PLE table), then 80 s on
    engine initialization. Targets: cache post-processed weights; make
    selection and scoring options switchable per request so strategy
    comparisons share one server.
11. **Drive.** The NVMe runs at PCIe Gen3 x4 (3.72 GB/s); a Gen4 slot would
    double the reselection read rate.
