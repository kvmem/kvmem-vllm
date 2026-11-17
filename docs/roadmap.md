# Roadmap

**Draft v2, 2026-11-12.** The order of work is settled; the open decisions at the end affect scope and dates.
The running experiment log stays in `PROGRESS.md`; this file holds only the plan.

## What the roadmap serves

- **A series of papers.** The disaggregated-KV system paper is done. The retrieval paper comes first. The parallel-serving system is built alongside it and becomes the next paper. See [Papers](#8-papers).
- **A public release** that others can install and run on common GPUs and on more than one model.

Two applications lead: **memory benchmarks** (offline prefill) and **long-horizon agents** (DeepSWE, Terminal-Bench).
Online serving through agent harnesses follows them and reuses their work.

## Two lanes, two acceptance rules

Every change belongs to one lane. Keeping them apart is what makes results comparable.

| Lane | What it changes | Accepted when |
| --- | --- | --- |
| **System** | Speed, memory, capacity, robustness. Never which history blocks are selected or evicted. | Opt-in first; token-identical at one request; measured on the scorecard; then default-on. No GPU memory outside the KV pool. Portable code paths only. |
| **Strategy** | Which history is retrieved, kept or evicted, so it can change model output. | Runs as a configuration choice on the same pipeline; compared side by side at equal budgets on the quality scorecard. A default changes only with a comparison table. |

Quality is a research question in the strategy lane, not a regression gate on system work.

## Where we are

| Application | Latest evidence | Gap |
| --- | --- | --- |
| Memory benchmarks | RULER-130 on the current stack: 5,680 prefill tok/s end to end at 8 concurrent, macro score 93.8–94.4 (11-10). | MemoryAgentBench, LongMemEval-S and NIAH were last scored in mid-October on the earlier selection path. No native-vLLM comparison on the current stack. |
| Long-horizon agents | DeepSWE first16 through Claude Code (11-10): 15 of 16 completed, 3 passed, 259 decode tok/s, GPU 48% busy. Agent simulator (11-12): 1.21× native vLLM at 8 agents, 3.1× at 24. | The DeepSWE run predates the 11-11/12 gains. Terminal-Bench was last run on 10-29, with the `terminus-2` agent. Simulator gains are unconfirmed on real agents. No compaction baseline yet. |
| Online serving | DeepSeek-harness 8-agent suite (11-11): 32 of 32 turns, no crashes, 370 output tok/s. | Session ledger never evicts; no restart recovery; bridges are scripts outside the package. |

| GPU | Weights | 1 card | 2 cards | 4 cards |
| --- | --- | --- | --- | --- |
| RTX PRO 6000, 96 GB | NVFP4 | Primary; all current-stack results | not run | not run |
| RTX 5090, 32 GB | NVFP4 | Measured 10-31 on the earlier stack | not run | not run |
| A800, 80 GB | Int4 (GPTQ, AWQ) | Agent simulator 11-12; decode about 2.6× slower than the PRO 6000 | Tensor parallel measured 11-11, uncommitted | not run |
| RTX 4090, 24 GB | Int4 (GPTQ) | Runs with MLP layers on the host (10-29) | Tensor parallel in progress 11-12 | not run |

All of the above is one model family: Qwen 27B hybrid checkpoints, plus a separate adapter for Qwen3.8 Flash Next.

## Tracks

### 1. Scorecard and baselines

A fixed set of runs repeated at every milestone, so progress is measured the same way each time.

- **Memory:** RULER-130, MemoryAgentBench, LongMemEval-S, NIAH at 256K.
- **Agents:** DeepSWE and Terminal-Bench, both through the Claude Code harness, **4 seeds per arm**.
- **Metrics per run:** prefill and decode tok/s, wall time, quality score, peak GPU and host memory, errors, and the run-to-run spread.
- **System baseline:** native vLLM with prefix caching on the same GPU, at its best measured admission setting.
- **Quality baseline for agents:** native vLLM with Claude Code compaction. Proposed protocol:
  - Baseline arm: compaction on, context limit set to the largest that fits the same number of concurrent agents in GPU memory.
  - KVMem arm: same harness, GPU and concurrency; compaction does not trigger below the logical context limit.
- **References for memory benchmarks:** native full attention above, recency-only below. They do not run through a harness, so compaction does not apply.
- **Mechanics:** one command per run, one result schema, FlashInfer autotune cache pinned, results table kept in the repo.

What 4 seeds resolve (rough, treating runs as independent):

| Set | Runs per arm | Smallest pass-rate difference it can show |
| --- | --- | --- |
| DeepSWE first16 | 64 | about 14 points (at a pass rate near 20%) |
| Terminal-Bench, about 90 tasks | about 360 | about 7 points (at a pass rate near 30%) |

One first16 pass took 3.7 hours on 11-10, so 4 seeds cost about 15 GPU-hours per arm. First16 stays the throughput fixture; a pass-rate claim on DeepSWE needs a larger set.

First deliverable: scorecard v0 on the PRO 6000 with the committed default stack. It replaces every stale number above.

### 2. Strategy lab

Keep several retrieval and eviction strategies working on the same fast pipeline and compare them side by side. This track produces the retrieval paper and has first priority.

Strategies in the code today:

| Decision | Options |
| --- | --- |
| Retrieval for a new query | Global MeanK top-K; per-layer selection; block scores `mean_k`, `overlap_16_8`, `key_fps_4`; guided Query; Query replay |
| What stays when the window fills | `recency`, `attention`, `graph` (v2: locality, attention, multi); decode scoring scope (all or non-thinking output) |
| Eviction from CPU history | none, `lru`, `redundancy`, thinking-only with LRU fallback; two unmerged branches (`statistical-gc`, `tbpro-gc-ab`) |
| Outside KVMem | Claude Code compaction on native vLLM (the agent baseline) |

The throughput stack was built on the per-layer path, and the launcher rejects many combinations. A comparison today would mix strategy with code path.

1. **Audit:** a matrix of strategy × pipeline feature, marking what runs, what is rejected and what silently falls back.
2. **Interface:** scorer, picker and evictor contracts, so a strategy plugs into the fast pipeline and inherits its speed. The contract, the per-layer seams and the trace reader exist; see [layers](layering.md) for what remains.
3. **Harness:** same inputs, same budgets (selected tokens, generation reserve, CPU history cap), same seeds.
4. **Output:** per benchmark, a table with strategies as rows: score, whether the gold evidence was in the window (RULER, NIAH), throughput, bytes moved per refresh.
5. **Smoke test** per strategy in CI, so none stops working unnoticed.

### 3. Throughput

Ordered by the application it serves. Estimates come from `PROGRESS.md`. This track and the next two produce the parallel-serving paper.

- **Agents**
  - Native, GIL-free decode publication and action capture. The engine thread is about 70% busy at 16 running requests; this is the limit beyond 12–16.
  - Turn boundaries: request-finish stall (est. +2–3%), medium prefill steps (est. +2–4%), turn-start plans. The held capture alone measured −2.7% and needs rework or removal.
  - Swap-in: layer-pipelined restore; step-boundary waiting is about 160 of 200 ms for a resident session.
- **Memory benchmarks**
  - Eager refresh at 2 concurrent (est. +7–8%, awaiting approval).
  - First-request warm-up at 256K (4.8 s).
  - Final-Query out-of-memory at 16 concurrent.
  - In-kernel statistics cost about 6% of GPU time; removing it changes results, so it is a strategy-lane decision.
- **Closed:** larger CUDA-graph capture sizes (declined, extra GPU memory); layer-0 GPU path (withdrawn); layer-0 wait (parked).

### 4. Capacity and tiering

Host memory is the binding limit: about 34.8 KB per history token, so a 128 GB host holds 12 running requests at 256K.

- Built: historical GC with the fast stack; CPU→SSD session tiering and expected-wait victims (opt-in); the configuration compiler.
- Next: demote-instead-of-evict across CPU and SSD; a RAM-bound capacity test; the compiler as the standard way to produce a profile.

### 5. Hardware matrix

Target: 4090, 5090, A800 and PRO 6000, on 1 to 4 cards, with **tensor parallel** as the multi-card mode.

- Land the tensor-parallel work, then extend it to 4 cards. Attention heads are split evenly across cards, so the card count must divide the head count; 1, 2 and 4 are planned.
- One compiler-generated profile per cell, each validated by a short run.
- Support levels: cells run at every milestone, cells run per release, and cells with a predicted profile only.
- Per-GPU notes in the docs: weight format, what fits, expected throughput.

One engine per card with shared CPU history is not planned.

### 6. Model coverage

Scope: models of similar size to Qwen 27B, and smaller ones. Smaller models also make a single 24 GB card practical without MLP layers on the host.

1. **Adapter contract:** write down what a model must expose: attention KV layout and codec, position re-rotation, recurrent-state capture and restore, draft head, weight backend. Today this is spread over the 27B path and two adapters (`q8_adapter`, `qsa_adapter`).
2. **Same family first:** smaller Qwen hybrid checkpoints reuse the layout and cost least.
3. **One attention-only model family,** to show the contract is not specific to the hybrid layout.
4. **Qualification per model:** one-request exactness, a four-request smoke, a scorecard subset, and a compiler profile.

### 7. Release

- **Configuration:** about 95 default-on environment flags move into typed profile fields and presets. Flags remain for A/B runs.
- **Path:** probe hardware → compile a profile → serve, documented end to end.
- **CI:** CPU-only today (two workflows). Add a GPU job: one-request exactness, a four-request smoke, a short scorecard.
- **Tests and docs:** triage the CPU test failures noted on 11-10; regenerate the status page and README numbers from the scorecard.
- **Bridges:** package the Claude Code and DeepSeek-harness bridges as supported entry points.
- **Versions:** state the supported vLLM releases and check them in CI.

### 8. Papers

| Paper | Order | Fed by | Evidence still needed |
| --- | --- | --- | --- |
| Disaggregated KV system | Done | — | Keep its numbers reproducible from the release path. |
| Retrieval | **First** | Tracks 1, 2 | Side-by-side strategy tables on the memory benchmarks; agent pass rates over 4 seeds against Claude Code compaction. |
| Parallel serving | System built alongside; paper second | Tracks 1, 3, 4, 5 | Throughput against native vLLM as concurrent agents grow; scaling across GPUs and 1, 2, 4 cards; the optimization ladder. |
| Later candidates | — | Future work | MoE models; others as they appear. |

Parallel serving means many concurrent long-context agents on one GPU, plus scaling across cards.

For each paper: fix the claim and the figure list first, then map every figure to a scorecard run, a frozen profile and a commit.

### 9. Online serving

Later, except where an item also blocks agents: session eviction and release, restart recovery, cold prefill after a harness rewrites history.

## Sequence

Two streams run side by side. Stream A leads.

| Horizon | Stream A: retrieval paper | Stream B: parallel-serving system |
| --- | --- | --- |
| **Now** (about two weeks) | Claim and figure list. Strategy audit matrix. Scorecard v0 quality runs: the four memory benchmarks on the current default, with the full-attention and recency-only references. Compaction-baseline protocol; Terminal-Bench through Claude Code. | Land in-flight work and clean the tree. Scorecard v0 throughput runs with the native baseline. Tensor-parallel merge. |
| **Next** (the following month) | Strategy interface. First side-by-side tables on the memory benchmarks. Agent pass-rate runs, 4 seeds per arm. Evidence-in-window diagnostics. | Native decode publication. Turn-boundary stalls. Four-card tensor parallel and per-GPU profiles. Demote-instead-of-evict. |
| **Later** | Evidence freeze and reproduction package. | Claim and figure list for the paper. Scaling evidence across the hardware matrix. |

Shared by both streams and the release, starting in Next: the adapter contract and a first additional model, typed configuration, the GPU CI job. Later: the release candidate, an attention-only model family, online-serving robustness.

GPU time: both streams share the PRO 6000. Quality arms must use the same weights, so they all run there and take priority. Throughput sweeps run between them; tensor-parallel work uses the A800 and 4090 machines.

## Future work

- **MoE (mixture-of-experts) models: essential.** They are the next model class after the dense and hybrid models above, and the main reason to need several cards.
- One engine per card with shared CPU history.
- Online-serving robustness beyond what agents need.
- **Best configuration per hardware, found directly** (2026-11-17). A registry of measured presets, one per hardware × model × workload class, each holding the complete serving profile (draft method, selection strategy, budgets, concurrency, graph sizes, winning optimization flags) and its evidence (commit, harness command, tok/s, exactness checks). `vllm-kvmem serve --preset auto` detects the GPUs (the planner's hardware probe) and loads the matching preset. `vllm-kvmem tune` builds one for hardware without a preset: candidates come from each optimization's declared hardware scope, budgets from the planner, short harness runs decide, and the planner's noise rule keeps the simplest configuration within noise of the best. Seed presets: 2×A800 TP=2 (16 × 32K, 718 tok/s, `benchmarks/tp2/run_16x32k.sh`) and 2×RTX 4090 TP=2 once tuned.

## Improvement loop

1. Run the scorecard.
2. Attribute lost time per workload and rank the losses in tok/s.
3. Fix the top item under the system-lane rule.
4. Promote it to default, recalibrate the compiler, rerun the scorecard.
5. Log in `PROGRESS.md`; revise this file at each milestone.

## Decisions

- 2026-11-12 (first round): serves papers and a public release; memory benchmarks and agents lead; quality is a separate research problem with strategies compared side by side; hardware is 4090, A800, 5090 and PRO 6000 on 1 to 4 cards.
- 2026-11-12 (second round): several papers (disaggregated KV done; retrieval; parallel serving; more later); tensor parallel first, MoE as essential future work; models of similar size to Qwen 27B or smaller are in scope; Claude Code compaction is the quality baseline; 4 seeds.
- 2026-11-12 (third round): the retrieval paper goes first; the parallel-serving system is built in parallel; parallel serving means many concurrent long-context agents on one GPU, plus scaling across cards.

## Open decisions

| # | Decision | Assumed until decided |
| --- | --- | --- |
| 1 | Retrieval paper deadline | No date; horizons stay relative |
| 2 | Which models to add first | A smaller Qwen hybrid checkpoint, then one attention-only family |
| 3 | Terminal-Bench agent | Move from `terminus-2` to Claude Code so the compaction baseline applies |
| 4 | DeepSWE set size for pass-rate claims | First16 for throughput only until a larger set is chosen |
| 5 | GPU time on the PRO 6000 | Retrieval-paper quality runs first; throughput sweeps between them |
