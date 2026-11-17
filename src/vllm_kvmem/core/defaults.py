"""Default-on optimization stack.

Most kvmem-vllm optimizations were introduced behind opt-in ``KVMEM_*``
environment flags and had to be copied into every run's execution.json; a
missing flag silently lost throughput. Since 2026-11-11 the flags measured to
be effective are on by default (user decision).

``DEFAULTS`` is the stack common to the three validated deployments
(RULER-130 stack r2, DeepSWE layerwise first16, dsh c8 r5; identical values in
all three) plus the flags validated after them:
- KVMEM_REQUEST_PLAN_CACHE: +8% tok/s at 8x120K with 150 locality spans, exact.
- KVMEM_CHECKPOINT_FAST_STATE, KVMEM_FAST_REQUEST_STAGING: per-turn session
  host work, +57-70% on the long-history short-turn repro, exact.
- KVMEM_SESSION_FAST_TURN, KVMEM_LAZY_ARCHIVE_REPORT: a further +7.5%, exact.
- KVMEM_CHECKPOINT_CAPTURE_INDEX: session capture index and lazy decode rows,
  exact.
- Fast session swap-in (user decision 2026-11-12): KVMEM_ASSEMBLY_ISOLATE_PARTIAL,
  KVMEM_RESTORE_DIRECT_LAYERS, KVMEM_RESTORE_PREPARED_READS,
  KVMEM_RESTORE_GATE_EXEMPT. 32 tool-waiting agents: 375.5 -> 492.8 tok/s
  (+31%); an evicted swap-in at N=1 3.7 s -> 0.2 s; byte-identical restored
  pages, N=1 token-identical. Outside layerwise session restores they change
  nothing (each path falls back on its own).
- Rolling decode scoring (user decision 2026-11-11, the default scoring
  policy): KVMEM_LAYERWISE_DECODE_SCOPE=non-thinking counts accepted
  non-thinking output, KVMEM_LAYERWISE_DECODE_INTERVAL=4 samples every 4th such
  token, KVMEM_LAYERWISE_DECODE_PREFILL=1 also folds accepted prefill rows
  (tool results, user input) minus thinking. Candidates stay all history. It
  changes which blocks rolling decode keeps (not exact against "all"); outside
  rolling decode it changes nothing.

- Native score capture and DFlash2 staging (2026-11-17), all exact:
  KVMEM_NATIVE_SCORE_STACK=1 (every layer's native masses from one gather) and
  KVMEM_NATIVE_WINDOW_FRAME=1 (decode windows from the frozen frame), c1
  token- and selection-identical on 2xA800; on 2xRTX 4090 MTP4 layerwise
  8 x 32K they cut score capture 11.8 -> 3.7 ms and the step 60.2 -> 52.9 ms
  (383.1 -> 404.6 decode tok/s). KVMEM_DFLASH_SUBMIT_AFTER_DRAFT=1 with
  KVMEM_DFLASH_STAGE_BEFORE_DRAFT=1 (compiled DFlash2 capture staged before the
  draft, submitted after it): token-identical at c1 on both GPUs, +3.8% at
  16 x 32K on 2xA800, decode step -7% at 8 x 32K on 2xRTX 4090. Each falls
  back on its own where it does not apply (no native overlay, not DFlash2).

- Decode-parallel staging (2026-11-16): KVMEM_NATIVE_DECODE_LAYER_BATCH=1 with
  KVMEM_NATIVE_DECODE_PREPARE_BATCH=1 stage every layer's rolling-decode mass
  and identities in one native call on the publication pool (GIL released);
  bitwise exact. A800, 8 agents x 128K history: decode step 84 -> 70 ms
  (about +20% decode tokens/s), one-session token-identical.

Deliberately NOT defaulted (set them explicitly per run):
- diagnostics: KVMEM_PUBLICATION_TRACE, KVMEM_QUERY_STAGE_PROFILE,
  KVMEM_HD_ALLOW_PTRACE;
- capacity/policy features: KVMEM_LAYERWISE_HISTORICAL_GC,
  KVMEM_SESSION_IDLE_LRU_EVICTION;
- anything that needs extra GPU memory (e.g. larger CUDA-graph capture sizes).

Semantics: at package import every entry is set with ``setdefault``, so an
explicit environment value (including "0") always wins.
``KVMEM_LEGACY_DEFAULTS=1`` skips the table entirely and restores the
pre-2026-11-11 opt-in behaviour (use it for baselines and A/B arms).
"""
import os

DEFAULTS = {
    'KVMEM_ARCHIVE_SKIP_DIGEST': '1',
    'KVMEM_AUTHORITY_BATCHED_MEANS': '1',
    'KVMEM_CHECKPOINT_FAST_STATE': '1',
    'KVMEM_CHECKPOINT_LAZY_EVIDENCE': '1',
    'KVMEM_CHECKPOINT_PINNED_PREWARM': '17',
    'KVMEM_DECODE_ACTION_BEFORE_MTP': '1',
    'KVMEM_DECODE_INLINE_READS': '1',
    'KVMEM_FAST_REQUEST_STAGING': '1',
    'KVMEM_GIL_GATE_MS': '50',
    'KVMEM_GIL_GATE_DRAIN_OPEN': '1',   # 11-12: +18-22% at 16 agents, +17.7% at 24, neutral at 8; N=1 exact
    'KVMEM_GLOBAL_PREFILL_BUFFERED': '1',
    'KVMEM_GRAPH_NATIVE_SIZE': '1',
    'KVMEM_LAYERWISE_BATCHED_READS': '1',
    'KVMEM_LIFECYCLE_SNAPSHOT_CACHE': '1',   # 11-12: byte-identical scheduler_resources.json; +6.1% at 24 agents with round 2
    'KVMEM_LAYERWISE_BUFFERED_HISTORY': '1',
    'KVMEM_LAYERWISE_PACKED_ASSEMBLY': '1',
    'KVMEM_LAYERWISE_PAIRED_READS': '1',
    'KVMEM_LAYERWISE_PREFILL_PRESSURE': '1',
    'KVMEM_LAYERWISE_PREPARED_INDEX': '1',
    'KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT': '1',
    'KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION': '0',
    'KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE': '1',
    'KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD': '1',
    'KVMEM_LAYERWISE_PRESSURE_DEFERRED_MAPS': '1',
    'KVMEM_LAYERWISE_PRESSURE_EARLY_LEASE': '1',
    'KVMEM_LAYERWISE_PRESSURE_EARLY_LEASE_YIELD_MS': '30',
    'KVMEM_LAYERWISE_PRESSURE_FAST_METADATA': '1',
    'KVMEM_LAYERWISE_PRESSURE_FAST_PREPARE': '1',
    'KVMEM_LAYERWISE_PRESSURE_IDLE_ACK_WAIT_MS': '60',
    'KVMEM_LAYERWISE_PRESSURE_LANES': '1',
    'KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST': '1',
    'KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT': '1',
    'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER': '0',
    'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS': '1',
    'KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING': '0',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE': '1',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE_HANDOFF': '1',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE_PREFORWARD': '0',
    'KVMEM_LAYERWISE_PRESSURE_PIPELINE_PROGRESS': '1',
    'KVMEM_LAYERWISE_PRESSURE_POST_FORWARD_REAP': '1',
    'KVMEM_LAYERWISE_PRESSURE_PREPARED_COMMIT': '1',
    'KVMEM_LAYERWISE_PRESSURE_PREPARED_READS': '1',
    'KVMEM_LAYERWISE_PRESSURE_PROMISE': '1',
    'KVMEM_LAYERWISE_PRESSURE_REUSE_LAYER_VIEWS': '1',
    'KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK': '1',
    'KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY': '1',
    'KVMEM_LAYERWISE_PRESSURE_SUBMIT_ON_PREPARE': '1',
    'KVMEM_LAYERWISE_QUERY_LAZY_READER': '1',
    'KVMEM_LAYERWISE_QUERY_PAIRED_GATHER': '0',
    'KVMEM_LAYERWISE_QUERY_RECORD_INDEX': '1',
    'KVMEM_LAYERWISE_QUERY_RESIDENT_V': '0',
    'KVMEM_LAYERWISE_QUERY_SCORE_BACKEND': 'tiled',
    'KVMEM_LAYERWISE_SESSIONS': '1',
    'KVMEM_LAZY_ARCHIVE_REPORT': '1',
    'KVMEM_MIXED_DECODE_CANDIDATES': '1',
    'KVMEM_NATIVE_ARM_BATCHED': '1',
    'KVMEM_NATIVE_DECODE_LAYER_BATCH': '1',
    'KVMEM_NATIVE_DECODE_LAYER_MAP': '1',
    'KVMEM_NATIVE_DECODE_MASS_MAP': '1',
    'KVMEM_NATIVE_DECODE_PREPARE_BATCH': '1',
    'KVMEM_NATIVE_DECODE_SEGMENTS': '1',
    'KVMEM_NATIVE_DECODE_SPARSE_MASS': '1',
    'KVMEM_NATIVE_PREFILL_PLAN': '1',
    'KVMEM_NATIVE_PUBLISH_SELECTION': '1',
    'KVMEM_NATIVE_PUBLISH_THREADS': '8',
    'KVMEM_PIPELINE_DISPATCH_BATCH_MEMBERS': '1',
    'KVMEM_PREFILL_CADENCE_MIN_DECODERS': '2',
    'KVMEM_PREFILL_DECODE_BURST': '1',
    'KVMEM_PREFILL_DECODE_SHARE': '0.08',
    'KVMEM_PREFILL_PRESSURE_NO_REPLAY': '1',
    'KVMEM_PRESSURE_LAYER0_PARALLEL_GATHER': '1',
    'KVMEM_PRESSURE_NATIVE_COMMIT_MAP': '1',
    'KVMEM_PRESSURE_NATIVE_COMMIT_TABLE': '1',
    'KVMEM_PRESSURE_NATIVE_LAYER_PICK': '1',
    'KVMEM_PRESSURE_NATIVE_LOOKUPS': '1',
    'KVMEM_PRESSURE_NATIVE_PLAN_CHECK': '1',
    'KVMEM_PRESSURE_NATIVE_READER': '1',
    'KVMEM_PRESSURE_READER_ADOPT': '1',
    'KVMEM_PRESSURE_READER_BATCH': '1',
    'KVMEM_PRESSURE_READER_EXTEND': '1',
    'KVMEM_PRESSURE_RECORD_INDEX_CACHE': '1',
    'KVMEM_PRESSURE_SINGLE_PICK': '1',
    'KVMEM_PROMISE_ASYNC_PAGE_CHECK': '1',
    'KVMEM_PROMISE_PM_SESSIONS': '1',
    'KVMEM_QUERY_RESERVE_FINAL_WINDOW': '1',
    'KVMEM_REFRESH_ASYNC_PAGE_CHECK': '1',
    'KVMEM_REQUEST_PLAN_CACHE': '1',
    'KVMEM_RUNNER_ARMED_LATCH': '1',
    'KVMEM_SESSION_FAST_TURN': '1',
    'KVMEM_WINDOW_TRANSLATE_DYNAMIC': '1',
    'KVMEM_CHECKPOINT_CAPTURE_INDEX': '1',
    # Fast session swap-in (user decision 2026-11-12; +31% at 32 agents, exact).
    'KVMEM_ASSEMBLY_ISOLATE_PARTIAL': '1',
    'KVMEM_RESTORE_DIRECT_LAYERS': '1',
    'KVMEM_RESTORE_PREPARED_READS': '1',
    'KVMEM_RESTORE_GATE_EXEMPT': '1',
    # Native score capture and DFlash2 staging (2026-11-17; exact).
    'KVMEM_NATIVE_SCORE_STACK': '1',
    'KVMEM_NATIVE_WINDOW_FRAME': '1',
    'KVMEM_DFLASH_SUBMIT_AFTER_DRAFT': '1',
    'KVMEM_DFLASH_STAGE_BEFORE_DRAFT': '1',
    # Default rolling decode scoring policy (user decision 2026-11-11).
    'KVMEM_LAYERWISE_DECODE_SCOPE': 'non-thinking',
    'KVMEM_LAYERWISE_DECODE_INTERVAL': '4',
    'KVMEM_LAYERWISE_DECODE_PREFILL': '1',
}

LEGACY = 'KVMEM_LEGACY_DEFAULTS'


def legacy(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get(LEGACY, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{LEGACY} must be 0 or 1')
    return value == '1'


# Names the package import set in this process (os.environ), for the startup log.
APPLIED = ()


def apply(environ=None):
    """Set every default not already present; returns the names it set."""
    global APPLIED
    process = environ is None
    environ = os.environ if process else environ
    if legacy(environ):
        return ()
    applied = tuple(name for name in DEFAULTS if name not in environ)
    for name in applied:
        environ[name] = DEFAULTS[name]
    if process:
        APPLIED = applied
    return applied


def summary():
    """One startup log line: how much of the stack came from the defaults."""
    if legacy():
        return 'KVMem default-on optimizations: off (KVMEM_LEGACY_DEFAULTS=1)'
    explicit = sorted(k for k in DEFAULTS if k not in APPLIED and os.environ.get(k) != DEFAULTS[k])
    return (f'KVMem default-on optimizations: {len(APPLIED)}/{len(DEFAULTS)} flags from defaults'
            + (f'; explicitly overridden: {", ".join(explicit)}' if explicit else ''))
