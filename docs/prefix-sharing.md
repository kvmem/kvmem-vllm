# Shared prefixes

Status 2026-11-15: phase 1 (CPU bytes archive) implemented and verified on the GPU (RULER qa1 100/100
token-identical, imports about 0.08 s). Phase 2 (the agent stack: native payload arena, historical GC,
pm sessions, async restore) implemented and CPU-tested, not yet run on the GPU (see "Phase 2: how to
use it on the agent stack"). Phase 3 (automatic registration) is not implemented.

## Goal

Let requests that start with the same tokens (a system prompt and tool definitions shared by many
agents, the context of several questions, a forked session) reuse one frozen prefix instead of each
prefilling it, without giving up KVMem's bounded GPU window. Questions on one MemoryAgentBench
context, which today run one after another in one session, can then run in parallel.

vLLM's automatic prefix caching cannot do this here: it shares KV blocks that sit in the GPU block
pool, while KVMem keeps only a bounded per-layer window on the GPU (each layer different blocks) and
the full prefix in its CPU archive.

## Design

**Prefix image.** A frozen prefix at token position P, captured from an IDLE frozen prompt
checkpoint (kind P): the archive payload of every block below P, the per-block retrieval metadata,
each layer's window and score history at P, the GDN recurrent and MTP draft state at P, and the
token identity. Everything in it is immutable. Sharing happens at registered boundaries only: the
recurrent state exists only at saved positions (vLLM's hybrid prefix caching has the same limit).

**Key and isolation.** (model, cache_salt, sha256 of tokens[:P]). A request may use an image only
with the same cache_salt and model; tenants never share.

**Registration.** Phase 1: explicit. A request marks `kvmem_share_prefix: P` (for example the end of
the system prompt and tools); after its frozen P checkpoint is IDLE it is registered as an image.
Phase 3: automatic policies (every frozen session prompt, popular prefixes).

**Import.** A new request whose session has no state and whose prompt starts with a registered image
(longest exact match) gets an ordinary new session B whose state is materialized from the image: its
own authority, repository slots, archive allocations and namespace, with the archive payload bytes
aliased (immutable Python `bytes` in the CPU archive, so no copy). B then holds an IDLE frozen P
checkpoint, and the existing claim and restore path runs unchanged: window assembly, recurrent state,
prefill from P on. Because B is an ordinary session afterwards, GC, swaps, tiering and eviction need
no change.

**Lifetime.** Images are reference counted and evicted least recently used under a CPU budget. A
payload is freed when the last image or session referencing it lets go; shared payloads are counted
once in CPU accounting.

**Exactness.** B's state is the frozen state a session restore already uses, so answers must be
token-identical to the sequential path at temperature 0: MAB follow-ups against the sequential runner
(the 11-15 MAB <=256K run has 600 follow-up answers as reference), agents against a cold prefill.

## Phases

1. Image capture and import on the CPU (bytes) archive, sync restore, no historical GC; explicit
   registration; MAB runner sends a context's follow-ups in parallel; exactness and throughput checks.
2. Native payload arena support (historical GC on, the agent stack): shared, reference-counted payload
   buffers; agent qualification (gate agent fixture with a shared system prompt, 8-agent app suite).
3. Automatic registration and eviction policy.

## Phase 1: how to use it

- **Register.** Send `"kv_transfer_params": {"kvmem_share_prefix": P}` with a session request
  (`session_id`, `cache_salt`). P must be the request's frozen prompt position: the start of its
  Query replay (`kvmem_replay_span`), a block boundary before the end of the prompt. When the request
  has retired and its P checkpoint is IDLE, the worker captures the image. A bad P or a configuration
  outside the scope below is a request error (`kvmem_share_prefix: ...`).
- **Import.** Any later request with a new `session_id`, the same `cache_salt` and a prompt that
  starts with the image's P tokens (and is longer) is admitted as a session hit at P: it waits one or
  two engine steps while the worker materializes the session, then restores like a follow-up turn.
  The new session is an ordinary session afterwards: it keeps its own frozen P, holds a session slot
  and must be released (`kvmem_release_session`) like any other.
- **Release.** Closing the registering session (`kvmem_release_session`) releases its images; LRU
  eviction applies under `KVMEM_PREFIX_IMAGE_MAX_BYTES` (strategy setting, default 96 GiB). Imported
  sessions keep their aliased payload after the image is gone.
- **Scope.** Phase 1 is the CPU bytes archive (no native payload arena) with sealed frozen P
  checkpoints and request planning; the full scope, shared with phase 2, is under "Scope and limits".
- **Accounting.** An imported session's archive allocations and CPU history credit count P like a
  session that prefilled P; the aliased bytes are not double counted in RAM, but they are in the
  archive slot pool and the credit ledger (phase 2 counts shared payloads once). Images count their
  full size against the image budget.
- **Events.** `capture/prefix_images.jsonl`: `registered`, `duplicate`, `refused`, `imported` (host
  timings), `import_failed`, `evicted`.
- **Code.** `serving/prefix_images.py` (capture, import, scheduler index), `SessionLedger` (lookup
  hook, placeholder entries, notices), `SessionWorker` (registration at retirement, import commands),
  `MemoryArchive.record_payloads` / `alias_records`, `PromptCheckpointPool.install_imported`,
  `CaptureRuntime.build_authority`, native `Store.register_block`.

## Phase 2: how to use it on the agent stack

Phase 2 brings images to the agent stack: the native payload arena, historical GC, pm sessions
(`session_checkpoint_policy: pm`, terminal capture, no frozen prompt), asynchronous restore, sessions
that live for hours and a system prompt shared by many agents (the gate agent profile,
`benchmarks/gate/agent.json`, is in scope).

**Server.** Set `KVMEM_PREFIX_POOL_BYTES` (strategy setting, bytes): the CPU budget of the
shared-prefix pool. It is required when historical GC is on (without it a registration is refused);
0, the default, keeps the native arena exactly as before. The pool is part of the CPU history
capacity (`cpu_history_coordination`), not added to it: size it for the shared prompts you register,
in records of `history_record_charge` bytes (one archived block plus its MeanK index). With
historical GC and the pool set, the archive keeps its descriptor bound and bounds payload slots by the
byte budget (`make_shared_archive`), so aliases take descriptors but no bytes. When set, it is also
the LRU byte budget of the scheduler's image index in place of `KVMEM_PREFIX_IMAGE_MAX_BYTES` (an
image's size there also counts its recurrent and draft state).

**Register a shared prompt.** Send one dedicated request on a session that you keep open:

```json
{"prompt": [<shared prompt: P tokens, P a multiple of block_tokens>, <q more tokens>],
 "max_tokens": 1, "session_id": "agents-prefix", "cache_salt": "<the agents' salt>",
 "kvmem_query_span": [P, P + q], "kvmem_query_mode": "agent", "kvmem_reselect": "auto",
 "kvmem_mandatory_spans": [],
 "kv_transfer_params": {"kvmem_share_prefix": P}}
```

P must be the request's frozen prompt position (`RequestPlan.prefill_checkpoint`): with a Query that
starts at P, that is P itself. The q tokens are only a Query for this request; nothing after P is
shared (a short marker or the first user message works). After the request retires, the worker
captures the image from the session's retained P (a pm session keeps it in `pool.prompts` beside its
M; no frozen prompt is needed) and reports it to the scheduler; a request can import it from the next
engine step on. `benchmarks/gate/agent_sim.py --shared-prefix` sends exactly this request.

**Agents.** Every agent uses a new `session_id`, the same `cache_salt` and a prompt that starts with
the P shared tokens (and is longer). Its first request is admitted as a session hit at P: the worker
materializes the session from the image (its own authority, slots, index copies, GC scope and
layerwise holds; payloads aliased), and the request restores from P like a follow-up turn, by the
synchronous or the asynchronous restore path. From then on it is an ordinary session.

**Release.** Closing the registration session (`kvmem_release_session`) releases its images, as in
phase 1: keep it open while agents may still start. A released image's payloads stay while any
session aliases them, charged to the pool. The scheduler also releases images for the pool budget
(least referenced, then least recently used) and for a CPU history deficit (below).

**Accounting model.** A CPU payload (one archived record) is charged once. While it has a single
owner it belongs to that session's private charge, exactly as before. When it is shared, by an image
(a pool owner) or by another session's alias, it is charged to the shared-prefix pool at the full
per-record charge, and it stays there until its last reference drops: a payload never moves back to a
private charge, and an orphan (no image left, still aliased) stays pooled. Each session holding a
shared record still keeps its own MeanK index of it, charged to that session as the index's fraction
of the per-record charge. So:

- registration moves the parent's prefix records to the pool: the `image_registered` notice carries
  the new pool size and the records leaving the parent's charge, applied in one scheduler step, so
  nothing is counted twice or freed early;
- an import is recorded as an idle session charged for its index copies only;
- the worker's history rows (`SessionWorker._history_rows`) count private records only, and
  protection counts private records only. A shared record's index is counted while its index owner
  exists, including after GC retired the record and until the owner is reaped, so an idle session's
  reported charge never grows (the scheduler refuses an idle observation that grows);
- the worker reports the pool's size (`history_pool` notices: records, of which draining); payloads
  whose last owner left are still counted until their bytes are reclaimed;
- the scheduler's CPU history reservation is private snapshots + active reservations + the pool,
  and `observe()` still checks the arena's real bytes against it.

Historical GC remains each session's own policy: a session may retire a shared record from its
retrievable history, which drops only its reference and frees no bytes. Byte budgets are met with
private records only; slot pressure (repository slots belong to each session) may retire shared
records. GC never counts bytes as freed that another owner still holds.

**CPU history deficit order.** When an admission does not fit (scheduler,
`HistoricalHistoryAdmission`):

1. release every image no live session references (neither its registering session nor an importer).
   Until the worker acknowledges the release and the freed payloads have drained, nothing lossier
   runs and nothing is counted free;
2. reclaim idle sessions' unprotected private history, oldest first (as before);
3. with `KVMEM_SESSION_HISTORY_LRU_EVICTION=1`, evict the least recently used idle session, one fenced
   eviction at a time (as before). A session that holds no private record but keeps an image
   referenced is also a candidate, since evicting it can let the image go.

Images that sessions still reference are released only for the pool budget, least referenced first.

**Scope and limits** (`serving/prefix_images.scope_error`, checked at registration, capture and
import). In scope: the CPU archive with or without the native payload arena, synchronous and
asynchronous restore, frozen and pm sessions, historical GC with a pool budget, a single rank.
Refused: tensor parallel serving, legacy repository GC, Query window/FPS and query-boundary caches
(`query_score_mode` other than `mean_k`, `query_fps_prebuild`), `query_history`, recall-method Q
windows (`KVMEM_RECALL_METHOD`) and System One forks. Shared payloads never spill to SSD: session
tiering skips pool-charged records before any SSD write, and the arena refuses any other shared
payload (`spill_busy`), so an imported session spills only its private records. A registration needs
request planning and an owned session; it is refused while the P checkpoint is not idle, has
claim-only evidence, or any block below P is unpublished or retired.

**Events and counters.**
- `capture/prefix_images.jsonl`: `registered` (`native`, `transferred`, `pool_records`), `duplicate`,
  `refused`, `imported` (host timings, `allocated_records`), `import_failed`, `evicted`.
- Scheduler `cpu_history_admission` (in `scheduler_resources.json`): once the pool is used,
  `pool_records`, `pool_draining_records`, `image_releases` (for deficits) and `images_releasing`;
  `reserved_records` includes the pool.
- Arena stats: `shared_slots`, `pool_slots`, `unique_bytes`. Historical GC scope stats:
  `shared_marked`, `shared_released`.

**Implementation.**
- *Arena owners* (`csrc/historical_payload.*`). The payload control word holds, besides phase and
  readers, the number of extra owners and of shared-prefix pool owners; a single-owner payload has
  none and keeps its exact former control values. `share(handle, pool)`, `retire()` returning
  `Shared` for a non-last owner, `release_pool()`, `retire_unshared()` (historical GC: retires only a
  sole plain owner, otherwise reports `Shared` and changes nothing) and `owners()`. Spill refuses
  shared and pool-owned payloads.
- *Descriptors and payload slots* (`core/native_memory_storage.py`, `core/memory_storage.py`). An
  archive descriptor owns one reference to its payload and drops it exactly once. A descriptor whose
  payload others still own is freed at once; its arena slot stays occupied ("detached") until the
  last owner leaves. From the first share on, or with `payload_capacity` below the descriptor
  capacity, arena slots are tracked separately from descriptor ids, and writes are admitted only
  while a payload slot is free. `MemoryArchive.alias_records` (new descriptors sharing payloads,
  published to the importing namespace's GC scope and marked shared), `share_to_pool` (registration:
  the image becomes a pool owner, the parent's records are marked shared), `release_pool`,
  `pool_records`, `namespace_shared_records`; the tier tracks which descriptors are pool-charged.
- *Shared-aware historical GC* (`csrc/historical_manager.*`, `historical_kv_policy.hpp`,
  `historical_redundancy.cpp`). `HistoricalGcScope.mark_shared(lease)` marks a live record's payload
  as shared for its generation. A retired shared record is released logically: no arena retirement,
  no pending-byte credit, the slot is reclaimed, and the archive descriptor release drops the
  reference. If the arena reports other owners of an unmarked record, the release turns shared and
  withdraws its credit. Scope byte deficits use private bytes; the LRU and redundancy policies skip
  shared victims unless a slot deficit is open.
- *Capture and import* (`serving/prefix_images.py`). Registration takes the idle P from
  `pool.prompts` (pm) or the retained entry (frozen), checks the pool budget, shares every payload to
  the pool and copies each block's MeanK sums and the thinking classification. Import aliases the
  payloads, copies the index into the new session's own index owners (never shared), republishes
  redundancy keys and thinking counts and recreates the layerwise GC hold. The authority is built by
  `CaptureRuntime.build_authority` exactly as for a request, so current-window protection is set
  before the first block is installed (the store refuses to change it later); `Store.register_block`
  installs each block with its current-frame state, so only the frozen frame is in the protected
  current frame, as in a session that prefilled P. The import then joins the idle-session
  bookkeeping a retirement sets up (idle history reports, session tiering).
- *Scheduler* (`serving/historical_admission.py`, `serving/session_control.py`). Pool term,
  transfer and import records, the deficit order above, and the image index's reference tracking
  (registering session and importers).

## Verification plan

- CPU tests: import produces a store, records and archive reads identical to the parent's at P;
  translated window and layerwise maps; refcounts and eviction; isolation by cache_salt.
- GPU: MAB <=256K follow-ups through images, token-identical to the sequential answers; then MAB long.
- Gate unchanged on the default path (no image registered).
- GPU, agent stack: the gate agent fixture with a shared system prompt (`gate.py run agent ...
  --client '{"shared_prefix": 16384, "prefix_mode": "import"}' --env '{"KVMEM_PREFIX_POOL_BYTES": ...}'`)
  against the `warm` arm (each session reaches the same frozen P itself; must be token-identical) and
  the `cold` arm (throughput baseline); then the 8-agent app suite.
