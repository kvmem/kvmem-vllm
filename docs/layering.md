# Layers

The package is being split along one seam. **Serving** moves blocks and schedules work.
**Strategy** decides which blocks are retrieved, kept or evicted. They meet at a small
contract, and every decision is recorded so each side can be tested without the other.

This page describes the layers, what exists today, the steps that built it and what is left.
The plan it belongs to is the [roadmap](roadmap.md).

## Layers

| Layer | Directory | Owns | May import |
| --- | --- | --- | --- |
| Apps | `apps/` | Configuration compiler (`planner`), web UI, elastic gateway | Every layer |
| Strategy | `strategy/` | Scorers, pickers, evictors: contract, registry, trace, replay | Its own package and the standard library |
| Serving | `serving/` | Refresh pipeline, scheduling, sessions, tiering, tensor parallel, signal capture, the built-in strategy implementations | Strategy contract, model adapters, core |
| Model adapters | `adapters/` | Per-model KV layout, state capture, engine compatibility | Strategy contract, core |
| Core | `core/` | KV archive, records, the store, options, settings; the native extensions at the package root | Strategy contract |

Benchmarks and bridges live outside the package (`benchmarks/`, `scripts/`).
`tests/test_layers.py` checks the import rule for every layer on every run, with no
exceptions, and `tests/test_strategy_layer.py` checks that the strategy package imports
nothing else. Where core needs serving's behavior (tensor-parallel collectives, the layer
materialization type), serving registers it on `RequestMemory`; core never imports serving.

Modules are no longer importable at the flat package root, and a test fails if the
repository names a module there. The module paths vLLM receives
(`vllm_kvmem.serving.scheduler.KVMemScheduler`, `vllm_kvmem.serving.connector`) use the layer
names; this leaves vLLM's configuration hash and the agent fixture's tokens and decisions
unchanged. Experiment scripts and their tests still name files in the flat layout where
they handle runtimes frozen before the move.

## The contract

`src/vllm_kvmem/strategy/contract.py` defines three roles. A fourth, eviction, is listed
in the registry and chosen through `historical_gc.policy`; the evictors run in native code.

| Role | Decides | Sees | Returns |
| --- | --- | --- | --- |
| Query scorer | How relevant each history block is to a Query, per attention layer or globally | The candidate records, which are eligible, the Query rows, the mean-key index | One score per record; 0 for ineligible records |
| Pressure scorer | Which blocks of a full window are worth keeping, per layer | The layer's window rows and its accumulated attention | One score ≥ 0 per row, at least one > 0 |
| Picker | Which blocks a set of scores selects | Scores, mandatory blocks, token budget | The selected blocks |

Rules every strategy keeps, because serving pipelines work on the strength of them:

1. **Pure.** The result depends only on the view: no hidden state, no reading of GPU
   residency, no dependence on another layer's or another request's decision.
2. **Deterministic** for equal inputs.
3. **Read-only.** A view is a few references to buffers serving already holds. Nothing
   is copied to build it and nothing reachable from it may be changed.

Serving checks every result (`check_query_scores`, `check_pressure_scores`, `check_pick`)
and stops the refresh with a `ContractError` instead of running a selection it cannot trust.

Decisions are made on one of two **paths**. On the **global** path one selection serves
every layer (`authority.select_for_queries`, `prefill_attention.select_pressure`). On the
**per-layer** path each attention layer picks its own window (`--layerwise-*` flags). Each
registry entry declares the paths it runs on and the paths where serving's built-in code
implements it; serving binds that code to the registry (`registry.bind`), and both paths
look their strategy up by name.

## Choosing a strategy

Three profile options under `kvmem`, all defaulting to serving's built-in code:

| Option | Default | Global path | Per-layer path |
| --- | --- | --- | --- |
| `strategy_query_scorer` | `meank` | `meank`, `key_fps`, `recency`, `overlap` (legacy) | `meank`, `recency` |
| `strategy_pressure_scorer` | `attention` | `attention`, `graph`, `recency` | `attention`, `lift`, `recency` |
| `strategy_picker` | `native` | `native` | `native` |

The launcher records the path in `strategy_path` and refuses a name that does not run on
it. On the global path these options are the same switch as the older ones:
`strategy_query_scorer=key_fps` is `query_score_mode=key_fps_4`, and
`strategy_pressure_scorer=recency` is `pressure_policy=recency`; a conflicting pair is
refused. A default name resolves to "no strategy object", and serving runs its existing
path unchanged, including the fused native pick. The `recency` scorers keep the most
recent history and ignore content; they are the lower reference for comparisons.

`lift` is age-adjusted per-layer pressure: accumulated attention divided by what a block at
the same distance typically receives, kept in the native layer-history state. With the
`strategy` setting `KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS=N` the first N layers take the shared
global selection at the final per-layer Query; together they form the combined method of
[age-adjusted-pressure.md](age-adjusted-pressure.md).

A scorer that is not built in still runs on the fast pipeline. Its scores feed the same
native picker, through the fused native call when that is enabled. Two shortcuts
implement the built-in strategy only and refuse anything else: pressure batch selection
and group selection.

`python -m vllm_kvmem.serving.capabilities table` prints every registered strategy on every path,
with its declared needs. At launch, `vllm_kvmem.serving.capabilities.check` refuses a strategy
whose path or signals the configuration lacks and records the result in `launch.json`
under `strategy`.

## Adding a strategy

```python
from vllm_kvmem.strategy import PRESSURE_SCORER, Strategy, register

class OldestFirst:
    name = 'oldest'

    def scores(self, view):
        newest = max(row[2] + row[3] for row in view.mapping)
        return [float(newest - row[2] + 1) for row in view.mapping]

register(Strategy(PRESSURE_SCORER, 'oldest', OldestFirst, needs=('positions',),
                  summary='Oldest window blocks stay'))
```

Then set `"strategy_pressure_scorer": "oldest"` in the profile. A strategy on the hot
path should be cheap: it runs once per layer per refresh on the refresh thread.

`paths` defaults to the per-layer path. A Query scorer object may also declare the global
path, where it scores all layers at once (`view.layer` is `None`); global pressure runs
built-in scorers only.

## The decision trace

Serving writes each per-layer decision to `capture/layerwise_query.jsonl`, each refresh's
global selection to `capture/bounded_refresh.jsonl` (layer -1) and each historical-GC
eviction to `capture/historical_gc_evictions.jsonl`. `vllm_kvmem.strategy.trace` reads
them into one shape (kinds `query`, `pressure`, `evict`) and compares runs:

```sh
python -m vllm_kvmem.strategy.trace summary RUN
python -m vllm_kvmem.strategy.trace diff RUN_A RUN_B     # exit 1 if they differ
```

This gives each layer its own test:

- **Serving change:** the strategy is unchanged, so the trace and the generated tokens
  must be identical at one request.
- **Strategy change:** set `strategy_trace_scores` to also record every candidate's score
  and what the picker needed (mandatory blocks, budget, sink band), then replay the run on
  the CPU through other strategies:

  ```sh
  python -m vllm_kvmem.strategy.replay RUN [--evidence FILE]
  ```

  It picks every recorded per-layer decision again, first from the recorded scores (which
  must repeat the recorded choice) and then through every registered scorer the trace can
  feed. It reports exact repeats, overlap with the recorded blocks and, with `--evidence`,
  how much of the gold evidence each strategy keeps. Later decisions depend on earlier
  ones, so this screens strategies; finalists still run end to end. The reference pickers
  match the native store picker (`tests/test_strategy_replay.py`).

## What is in place

- The `strategy` package: contract, registry with paths and native bindings, built-in
  entries, `recency` scorers, trace reader, offline replay.
- Query and pressure scorers on both paths, looked up by name; the global path's
  `query_score_mode` and `pressure_policy` are names for the same entries.
- Picker seam at `RequestMemory.preview_selection`, the funnel for global and per-layer picks.
- Launch-time capability checks (`vllm_kvmem.serving.capabilities`) and the generated strategy table.
- The decision trace: per-layer, global and eviction decisions.
- `vllm_kvmem.core.settings`: the one accessor for `KVMEM_*` settings, and profile settings in
  `system` and `strategy` namespaces over the default-on preset ([settings.md](settings.md)).
- One directory per layer, with the import rule checked on every test run.
- The gate as a tracked command, `benchmarks/gate/gate.py`, with both fixtures.

## Steps

Seven steps, each ended at the [gate](#the-gate) below. What is left is listed under
[Open decisions](#open-decisions).

### 1. Prove the seam on the agent path (done 2026-11-12)

The seam had run on offline prefill only. Agents add sessions, historical GC, rolling
decode and several owners, which is where earlier restore bugs appeared.

- Built-in strategy on the agent fixture (one session, 12 turns, saved token IDs): 13 of 13
  requests token-identical to the commit before the strategy layer, and 160 of 160 layer
  decisions identical, 96 of them in rolling decode.
- The `recency` scorers on the same fixture: all 12 turns complete; every Query and pressure
  decision, including the 6 rolling-decode ones, keeps the same newest blocks in all 16
  layers, as the scorer intends.
- Four-session smoke runs, built-in and `recency`: 48 of 48 turns, no error, and no request,
  page, reservation or archive work held after the sessions are released.
- References: `mab` is the 2026-11-12 run of the commit before the strategy layer. `agent`
  is new; the earlier tool-wait runs used 24K of history and made no selection at all.
- The gate is a tracked command ([Running the gate](#running-the-gate)).

### 2. Audit and choose (done 2026-11-12)

- [strategy-audit.md](strategy-audit.md): every strategy against every path, as runs,
  rejected (at launch or late) or silent.
- Main finding: the per-layer path silently ignores every global Query strategy (window
  overlap, key FPS, the MeanK quota, both graph versions, guidance, Query history, the recall
  methods). Each shapes only the shared frame, while every attention layer still picks by
  per-layer MeanK. Five refusals come only at the first refresh.
- Chosen for step 3, the recommendation below: per-layer and global MeanK, key FPS, graph
  attention, attention and recency pressure, and the GC eviction policies. Window overlap,
  graph v1, guidance, Query history and Query token scores stay on the global path, marked
  legacy. The recall-experiment methods are left as they are. Nothing is deleted.

### 3. Port the chosen strategies (done 2026-11-12)

- Registry entries declare the paths they decide on and where serving's built-in code
  implements them; serving binds that code to the registry. New entries: Query scorers
  `key_fps` and `overlap` (legacy), pressure scorer `graph`; `recency` runs on both paths.
- `authority.select_for_queries` and `prefill_attention.select_pressure` look their scorer
  up in the registry instead of branching on `query_score_mode` and `pressure_policy`. A
  Query scorer object (`recency`) runs on the global path behind the contract.
- On the global path `strategy_query_scorer` and `strategy_pressure_scorer` are the same
  switch as the older options; the launcher records the path in `strategy_path`.
- The trace also holds each refresh's global decision, so a global strategy's own trace
  can be compared. Before and after the port, on the agent fixture: global MeanK, key FPS
  (also chosen as `strategy_query_scorer=key_fps`), recency pressure, graph pressure and
  per-layer recency gave the same tokens and decisions.

### 4. Capability checks (done 2026-11-12); signals by need (decided against)

- `vllm_kvmem.serving.capabilities` checks every chosen strategy against the launched path and the
  signals the configuration produces, and refuses at launch what was refused only at the
  first refresh. Strategies set through older options that decide only on the global path
  are reported for a per-layer launch (`launch.json`, `strategy.shared_frame_only`).
  `python -m vllm_kvmem.serving.capabilities table` generates the strategy table.
- **Signals by need: not built, by decision (2026-11-13).** The attention statistics feed
  the prefill publication pipeline, which requires action and prefill attention, and they
  are computed inside the attention kernel. Only the `recency` reference would stop paying
  for them, so its throughput includes their cost (about 3–6% of GPU time); its decisions
  and quality do not depend on it. The launch report lists such signals (`unused_signals`).

### 5. Evictors and offline replay (done 2026-11-12)

- `historical_gc.policy` must name a registered evictor. Evictions are written to
  `capture/historical_gc_evictions.jsonl` and read as `evict` decisions.
- With `strategy_trace_scores`, per-layer decisions also record what the picker needed, and
  `python -m vllm_kvmem.strategy.replay` picks them again on the CPU through other
  strategies. Its reference pickers equal the native store picker. On recorded runs the
  recorded scores repeat every per-layer decision: 160 of 160 on the agent fixture and
  4,400 of 4,400 on the memory benchmark. Recency would keep 89% and 92% of the same
  pressure blocks, and 77% and 41% of the same Query blocks.
- Recording scores changes no decision: both fixtures give the same tokens and decisions
  with and without `strategy_trace_scores`.

### 6. Typed configuration (done 2026-11-12)

- Every `KVMEM_*` read in the package goes through `vllm_kvmem.core.settings` (318 reads in 79
  modules); a test forbids direct reads.
- A serving profile may carry `settings` with a `preset` (`default-on` or `legacy`) and the
  namespaces `system` (never changes output) and `strategy` (may change it). The launcher
  expands them into the server environment; explicit execution flags still override them.
  Every setting has a namespace ([settings.md](settings.md)).
- The reference deployments set no profile settings; their effective settings are
  unchanged.

### 7. Directory moves (done 2026-11-12, finished 2026-11-13)

- 253 files moved into `core/`, `adapters/`, `serving/` and `apps/`; package-internal
  imports rewritten to the new paths.
- The import rule holds for every layer. The seven imports that pointed up after the move
  were inverted: two flag readers moved into `core/archive_flags.py`, the engine version
  check into `adapters/engine_api.py`, `checkpoint_storage` into serving, and serving
  registers its tensor-parallel and layer-materialization hooks on `RequestMemory`.
- The aliases at the old paths were removed, and every reference in the repository uses
  the layer paths, including the module paths vLLM receives.
- `capture.py`, `_fused_capture.py`, `capture_opaque.py` and `capture_compile.py` moved
  together and byte for byte: their source fingerprint is part of vLLM's compile
  configuration.

### Order

Steps 1, 2 and the replay tool in step 5 are independent and can run in parallel.
Steps 3 and 4 follow step 2. Step 6 comes before step 7 so that the move is a pure rename.

## The gate

Every step must leave the built-in strategy:

- token-identical at one request on both fixtures: 16 memory-benchmark cases (offline
  prefill, 256 Query and 4,144 pressure layer decisions) and the agent simulator;
- with an identical decision trace (`python -m vllm_kvmem.strategy.trace diff`);
- inside the run-to-run spread for throughput, measured against a repeat of the
  reference, not against its first, cold run.

When running it:

- Run from a frozen copy of the tree, never from a checkout that is being edited.
- Record a new reference whenever GPU kernels are rebuilt or re-tuned. A stale reference
  differs from the first decision on, by a few blocks, whatever the code.

### Running the gate

`benchmarks/gate/gate.py` runs both fixtures and every check:

```sh
python benchmarks/gate/gate.py freeze RUNTIME          # the working tree plus built extensions
python RUNTIME/benchmarks/gate/gate.py gate RUNTIME OUT --reference REF --autotune-dir TUNE
```

- `REF` holds four arm directories, or links to them. `mab` and `agent` are compared for
  tokens and decisions. `mab-repeat` and `agent-repeat` are repeats of them, used for
  throughput.
- The command writes `OUT/gate-result.json` and exits 0 only if both fixtures are exact
  and neither loses more than `--spread` (2%) of throughput against its repeat.
- Single steps: `gate.py run {mab,agent} RUNTIME OUT`, `gate.py compare REF RUN`,
  `gate.py measure RUN`. A reference is recorded with `gate.py run` on the reference tree.

| Fixture | File | What it runs | Throughput |
| --- | --- | --- | --- |
| `mab` | `mab16.json` | 16 memory-benchmark cases, offline prefill, 4,400 layer decisions | prefill-phase tok/s |
| `agent` | `agent.json` | One session: 48K history, then 12 turns of ~300 appended and 2,048 generated tokens; 160 layer decisions, 6 of 9 pressure events in rolling decode | output tok/s inside turns |

- Both fixtures use G = 4,096. The agent history exceeds the 36,864-token select budget, so
  the agent Query, prefill pressure and rolling-decode pressure all decide. At 24K history,
  as in the earlier tool-wait runs, the agent path makes no selection at all.
- `--sessions 4` on the agent fixture is the multi-owner smoke run. Its outputs depend on
  timing, so it checks that every turn completes, that no traceback appears and that the
  server holds no request, page, reservation or archive work after the sessions are released.
- The agent fixture keeps its own FlashInfer autotune cache (`--autotune-dir`), so tuning by
  other runs on the machine cannot change its GEMM kernels. The `mab` reference was recorded
  with the default cache and keeps it.
- The question files of `mab` are read from `--data`.

Working notes and the current reference runs are in `artifacts/layers-refactor-20261112/`,
which is local and not part of a source checkout.

## Open decisions

| Decision | Recommendation |
| --- | --- |
| Move `main` forward to the strategy-layer commit | Done: `main` follows every step that passed its gate |
| Comparison set for the retrieval paper | Adopted for step 3: per-layer MeanK, global MeanK, key-FPS, graph attention, recency, and the eviction policies. Whether the recall-experiment methods join it is open |
| Signals by need (step 4) | Decided against (2026-11-13): only the recency reference would benefit |
| Legacy experiments | Marked legacy in the audit, not deleted. Deleting needs a list confirmed by the user |
