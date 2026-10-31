"""Validated plugin settings carried by vLLM's public connector extra config."""
from dataclasses import dataclass, field

from .config import KVMemConfig


GRAPH_DEFAULTS = {
    'graph_mode': 'off',
    'graph_scoring_version': 'v1',
    'graph_alpha': 0.2,
    'graph_aggregation': 'sum',
    'graph_locality_weight': 0.1,
    'graph_attention_weight': 0.2,
    'graph_attention_topk': 4,
    'graph_attention_min_mass': 0.0,
    'graph_cross_message_weight': 0.25,
    'graph_max_bytes_per_session': 33554432,
    'graph_max_total_bytes': 268435456,
    'graph_trace': False,
    'graph_native_propagation': True,
}
GRAPH_SCORING_MODES = frozenset(('locality', 'attention', 'multi'))
_UNSET = object()
_LEGACY_GRAPH_WEIGHTS = frozenset(('graph_locality_weight', 'graph_attention_weight'))
_POLICY_FIELDS = frozenset((*GRAPH_DEFAULTS, 'pressure_policy', 'action_pressure'))


def validate_graph_aggregation(aggregation, version):
    if not isinstance(aggregation, str) or aggregation not in ('sum', 'max'):
        raise ValueError('graph_aggregation must be sum or max')
    if aggregation == 'max' and version != 'v2':
        raise ValueError('graph_aggregation=max requires graph_scoring_version=v2')


def resolve_graph_policy(values, *, explicit_fields=None):
    """Resolve policy aliases before defaults erase which fields were supplied."""
    import math
    explicit = set(values) if explicit_fields is None else set(explicit_fields)
    cfg = {**GRAPH_DEFAULTS, 'action_pressure': False, **values}
    version = cfg['graph_scoring_version']
    if not isinstance(version, str) or version not in ('v1', 'v2'):
        raise ValueError('graph_scoring_version must be v1 or v2')
    validate_graph_aggregation(cfg['graph_aggregation'], version)
    alpha = cfg['graph_alpha']
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('graph_alpha must be finite and in [0, 1]')
    legacy_weights = explicit & _LEGACY_GRAPH_WEIGHTS
    if 'graph_alpha' in explicit and legacy_weights:
        raise ValueError('graph_alpha cannot be combined with legacy graph weights')
    if version == 'v2' and legacy_weights:
        raise ValueError('graph_scoring_version=v2 rejects legacy graph weights')
    if version == 'v1' and 'graph_alpha' in explicit:
        raise ValueError('graph_alpha requires graph_scoring_version=v2')
    if type(cfg['action_pressure']) is not bool:
        raise ValueError('action_pressure must be boolean')
    if 'pressure_policy' in explicit:
        policy = cfg.get('pressure_policy')
        if not isinstance(policy, str) or policy not in ('recency', 'attention', 'graph'):
            raise ValueError('pressure_policy must be recency, attention or graph')
        if 'action_pressure' in explicit and (
                policy == 'graph' or cfg['action_pressure'] != (policy == 'attention')):
            raise ValueError('pressure_policy conflicts with explicit action_pressure')
    else:
        policy = 'attention' if cfg['action_pressure'] else 'recency'
    cfg['pressure_policy'] = policy
    # Compatibility for consumers that only need to know whether pressure uses
    # attention. Dispatch between legacy and graph pressure uses the enum.
    cfg['action_pressure'] = policy != 'recency'
    return cfg


def canonical_policy_payload(values, resolved):
    """Serialize resolved policies without turning inactive defaults into input."""
    payload = dict(values)
    get = resolved.get if isinstance(resolved, dict) else lambda name: getattr(resolved, name)
    version = get('graph_scoring_version')
    payload['graph_scoring_version'] = version
    payload['graph_aggregation'] = get('graph_aggregation')
    payload['pressure_policy'] = get('pressure_policy')
    if version == 'v2':
        payload['graph_alpha'] = get('graph_alpha')
        for name in (*_LEGACY_GRAPH_WEIGHTS, 'action_pressure'):
            payload.pop(name, None)
    else:
        payload.pop('graph_alpha', None)
    return payload


def validate_graph_options(values, *, context_mode='meank', request_planning=False,
                           session_capacity=0, repository_gc=False, query_history=False,
                           action_pressure=False, action_attention=False,
                           prefill_attention=False, action_scope='tool-call',
                           action_fraction=0.25, pressure_policy=None):
    """Shared validation before either launcher creates a backend or resources."""
    import math
    cfg = {**GRAPH_DEFAULTS, **values}
    version = cfg['graph_scoring_version']
    if not isinstance(version, str) or version not in ('v1', 'v2'):
        raise ValueError('graph_scoring_version must be v1 or v2')
    validate_graph_aggregation(cfg['graph_aggregation'], version)
    alpha = cfg['graph_alpha']
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('graph_alpha must be finite and in [0, 1]')
    policy = pressure_policy if pressure_policy is not None else ('attention' if action_pressure else 'recency')
    if not isinstance(policy, str) or policy not in ('recency', 'attention', 'graph'):
        raise ValueError('pressure_policy must be recency, attention or graph')
    mode = cfg['graph_mode']
    if not isinstance(mode, str) or mode not in ('off', 'collect', *GRAPH_SCORING_MODES):
        raise ValueError('graph_mode must be off, collect, locality, attention or multi')
    for name in ('graph_locality_weight', 'graph_attention_weight',
                 'graph_attention_min_mass', 'graph_cross_message_weight'):
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and in [0, 1]')
    if cfg['graph_locality_weight'] + cfg['graph_attention_weight'] > 1:
        raise ValueError('Graph propagation weights must sum to at most one')
    for name in ('graph_attention_topk', 'graph_max_bytes_per_session', 'graph_max_total_bytes'):
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['graph_max_bytes_per_session'] > cfg['graph_max_total_bytes']:
        raise ValueError('Per-session graph capacity must fit the global graph capacity')
    if type(cfg['graph_trace']) is not bool:
        raise ValueError('graph_trace must be boolean')
    if type(cfg['graph_native_propagation']) is not bool:
        raise ValueError('graph_native_propagation must be boolean')
    if policy == 'graph':
        if version != 'v2' or mode not in GRAPH_SCORING_MODES:
            raise ValueError('pressure_policy=graph requires graph_scoring_version=v2 and a scoring graph_mode')
        if not action_attention or not prefill_attention or action_scope != 'non-thinking':
            raise ValueError('Graph pressure requires action_attention, prefill_attention and action_scope=non-thinking')
    if mode == 'off':
        return
    if context_mode != 'meank':
        raise ValueError('Block graph requires context_mode=meank')
    if repository_gc:
        raise ValueError('Block graph requires repository_gc=false')
    if not session_capacity:
        raise ValueError('Block graph requires owned sessions')
    if mode in GRAPH_SCORING_MODES:
        if not request_planning:
            raise ValueError('Block graph scoring requires request planning')
        if query_history:
            raise ValueError('Block graph scoring requires query_history=false')
        if version == 'v1' and policy != 'recency':
            raise ValueError('V1 block graph scoring requires action_pressure=false (pressure_policy=recency)')
        if action_attention and action_fraction != 0:
            raise ValueError('Block graph scoring requires zero effective legacy attention fusion')
        if mode in ('attention', 'multi') and (
                not action_attention or not prefill_attention or action_scope != 'non-thinking'):
            raise ValueError('Attention graph scoring requires action_attention, prefill_attention '
                             'and action_scope=non-thinking')


def validate_query_score_mode(mode, cache_bytes, *, block_tokens, graph_mode,
                              graph_scoring_version, graph_aggregation, graph_alpha,
                              query_history=False, guided_query_tokens=0,
                              context_mode='meank', mean_quota_fraction=0., historical_gc=None):
    """Keep experimental block scorers separate from graph aggregation."""
    if not isinstance(mode, str) or mode not in ('mean_k', 'overlap_16_8', 'key_fps_4'):
        raise ValueError('query_score_mode must be mean_k, overlap_16_8 or key_fps_4')
    if type(cache_bytes) is not int or cache_bytes <= 0:
        raise ValueError('query_window_cache_bytes must be a positive integer')
    if mode == 'mean_k':
        return
    if context_mode != 'meank' or block_tokens != 32:
        raise ValueError(f'query_score_mode={mode} requires MeanK context management and block_tokens=32')
    hybrid = type(mean_quota_fraction) in (int, float) and 0 < mean_quota_fraction <= 1
    alphas = (0,) if mode == 'key_fps_4' and not hybrid else (0, .2)
    alpha_description = '0' if len(alphas) == 1 else '0 or 0.2'
    graphless_history = (historical_gc is not None and mode == 'key_fps_4' and
                         graph_mode == 'off' and graph_alpha == 0)
    if ((graph_mode != 'multi' and not graphless_history) or graph_scoring_version != 'v2' or
            graph_aggregation != 'sum' or type(graph_alpha) not in (int, float) or
            graph_alpha not in alphas):
        raise ValueError(f'query_score_mode={mode} requires graph_mode=multi, '
                         f'graph_scoring_version=v2, graph_aggregation=sum and graph_alpha={alpha_description}')
    if query_history or guided_query_tokens:
        raise ValueError(f'query_score_mode={mode} requires query_history=false and guided_query_tokens=0')


def validate_query_fps_experiments(mode, halo_blocks, optimized, prebuild=False, prebuild_device='cpu',
                                   cpu_backend='torch', cpu_workers=8, mean_quota_fraction=0.0):
    import math

    if type(halo_blocks) is not int or halo_blocks not in (0, 1):
        raise ValueError('query_fps_halo_blocks must be 0 or 1')
    if type(optimized) is not bool:
        raise ValueError('query_fps_optimized must be boolean')
    if type(prebuild) is not bool:
        raise ValueError('query_fps_prebuild must be boolean')
    if not isinstance(prebuild_device, str) or prebuild_device not in ('cpu', 'cuda'):
        raise ValueError('query_fps_prebuild_device must be cpu or cuda')
    if not isinstance(cpu_backend, str) or cpu_backend not in ('torch', 'native'):
        raise ValueError('query_fps_cpu_backend must be torch or native')
    if type(cpu_workers) is not int or not 1 <= cpu_workers <= 32:
        raise ValueError('query_fps_cpu_workers must be an integer in 1..32')
    if (type(mean_quota_fraction) not in (int, float) or
            not math.isfinite(mean_quota_fraction) or not 0 <= mean_quota_fraction <= 1):
        raise ValueError('query_mean_quota_fraction must be a finite number in [0, 1]')
    if mean_quota_fraction > 0 and (mode != 'key_fps_4' or halo_blocks != 1):
        raise ValueError('query_mean_quota_fraction requires query_score_mode=key_fps_4 '
                         'and query_fps_halo_blocks=1')
    if cpu_backend == 'native' and (mode != 'key_fps_4' or not optimized or
                                   not prebuild or prebuild_device != 'cpu'):
        raise ValueError('query_fps_cpu_backend=native requires query_score_mode=key_fps_4, '
                         'query_fps_optimized=true, query_fps_prebuild=true and '
                         'query_fps_prebuild_device=cpu')
    if prebuild and (mode != 'key_fps_4' or not optimized):
        raise ValueError('query_fps_prebuild requires query_score_mode=key_fps_4 '
                         'and query_fps_optimized=true')
    if (halo_blocks or optimized) and mode != 'key_fps_4':
        raise ValueError('FPS halo/optimization require query_score_mode=key_fps_4')


def validate_graph_query_trace_bytes(value):
    if type(value) is not int or not 0 < value <= 1024**3:
        raise ValueError('graph_query_trace_max_bytes must be a positive integer at most 1 GiB')


@dataclass(frozen=True)
class RuntimeOptions:
    memory: KVMemConfig = field(default_factory=KVMemConfig)
    cpu_read_cache_bytes: int = 0
    packed_assembly: bool = False
    verify_archive_on_finish: bool = True
    direct_pinned_reads: bool = False
    prefill_cadence: int = 1
    async_archive_retirement: bool = False
    repository_gc: bool = False
    nvme_preallocate: bool = False
    nvme_max_bytes: int = 0
    archive_io_max_bytes: int = 0
    session_capacity: int = 0
    session_checkpoint_bytes: int = 0
    session_capture_decode: bool = False
    cpu_history_cache_bytes_per_request: int = 0
    session_capture_terminal: bool = False
    cpu_gather_workers: int = 8
    async_seal: bool = False
    gpu_raw_k_cache_bytes: int = 0
    reuse_resident_v: bool = False
    async_cpu_scatter: bool = False
    cpu_scatter_min_tokens: int = 128
    batched_cpu_append: bool = False
    request_planning: bool = False
    mean_k_backend: str = 'torch_reference'
    session_checkpoint_policy: str = 'sealed'
    query_history: bool = False
    query_history_decay: float = 0.5
    guided_query_tokens: int = 0
    archive_backend: str = 'nvme'
    action_scope: str = 'tool-call'
    action_layers: str = 'spread3'
    action_interval: int = 16
    action_decay: float = 0.5
    action_native_decode_commit: bool = True
    action_commit_max_bytes: int = 16777216
    action_download_pool: bool = True
    action_download_max_bytes: int = 67108864
    action_download_max_slots: int = 128
    action_fused_scoring: bool = True
    action_native_scores: str = 'off'
    action_native_prefill_aligned_pieces: bool = False
    action_native_max_rows: int = 256
    action_native_max_bytes: int = 2147483648
    async_refresh: bool = False
    refresh_quantum_tokens: int = 256
    prefill_attention: bool = False
    action_pressure: bool = _UNSET
    prefill_attention_interval: int = 1024
    prefill_attention_stride: int = 32
    prefill_attention_half_life: float = 8192
    session_freeze_prompt: bool = False
    cpu_history_coordination: bool = False
    cpu_progress_reserve: bool = True
    maintenance_max_steps: int = 0
    maintenance_busy_ms: float = 2.0
    maintenance_idle_ms: float = 8.0
    maintenance_aging_ms: float = 250.0
    maintenance_inflight_bytes: int = 0
    async_restore: bool = False
    adaptive_prefill: bool = False
    step_latency_interval: int = 0
    decode_step_target_ms: float = 50.0
    prefill_min_tokens: int = 128
    prefill_max_tokens: int = 2048
    prefill_feedback_timeout_ms: float = 5000.0
    prefill_batch_tokens: int = 0
    prefill_request_tokens: int = 0
    active_resident_ttl_ms: float = 0.0
    active_resident_pressure_only: bool = False
    active_resident_max_fraction: float = 0.5
    active_tool_grace_ms: float = 0.0
    active_tool_max_wait_ms: float = 15000.0
    maintenance_transfer_bytes_per_second: int = 0
    maintenance_transfer_burst_bytes: int = 67108864
    offline_agent_policy: bool = False
    offline_prefill_coalescing: bool = False
    offline_prefill_max_tokens: int = 4096
    offline_phase_policy: bool = False
    phase_prefill_wave_tokens: int = 8192
    phase_max_wait_ms: float = 1000.0
    split_capture_dma: bool = True
    native_prefill_commit: bool = True
    async_checkpoint_capture: bool = True
    restore_stage_timing: bool = False
    host_stage_timing: bool = False
    native_checkpoint_capture: bool = False
    checkpoint_pinned_prewarm_count: int = 0
    native_restore_frame: bool = False
    native_action_metadata: bool = False
    resident_restore_slots: int = 1
    session_gpu_checkpoint_bytes: int = 0
    restore_prepare_split: bool = False
    restore_handoff_fastpath: bool = False
    async_action_metadata: bool = False
    rolling_active_window: bool = False
    early_capture_slot_release: bool = False
    prefill_publication_pipeline: bool = False
    prefill_native_plan_arm: bool = False
    early_capture_feedback_bytes: int = 268435456
    action_profiling: bool = False
    action_profile_interval: int = 1
    action_profile_max_events: int = 256
    graph_mode: str = 'off'
    graph_locality_weight: float = _UNSET
    graph_attention_weight: float = _UNSET
    graph_attention_topk: int = 4
    graph_attention_min_mass: float = 0.0
    graph_cross_message_weight: float = 0.25
    graph_max_bytes_per_session: int = 33554432
    graph_max_total_bytes: int = 268435456
    graph_trace: bool = False
    graph_native_propagation: bool = True
    graph_scoring_version: str = 'v1'
    graph_aggregation: str = 'sum'
    graph_alpha: float = _UNSET
    pressure_policy: str = _UNSET
    _policy_explicit_fields: frozenset | None = field(default=None, repr=False, compare=False)
    _policy_resolved_values: tuple | None = field(default=None, repr=False, compare=False)
    cpu_direct_record_seal: bool = False
    historical_gc: object = None
    guided_query_backend: str = 'eager'
    guided_reselect: str = 'boundary'
    guided_trigger_tokens: int = 0
    guided_max_refreshes: int = 2
    assembly_chunk_blocks: int = 32
    prefill_capture_ahead: bool = False
    cpu_seal_hash_workers: int = 1
    capture_fused_pack: bool = False
    prefill_action_before_mtp: bool = False
    prefill_deferred_validation: bool = False
    prefill_deferred_validation_max_bytes: int = 67108864
    prefill_deferred_validation_scratch_bytes: int = 2147483648
    prefill_scatter_after_target: bool = False
    cpu_mean_index_simd: bool = False
    assembly_read_prefetch: bool = False
    assembly_fused_layers: bool = False
    prefill_physical_coalescing: bool = False
    prefill_physical_max_chunks: int = 2
    action_native_prefill_cta_gate: bool = False
    capture_opaque_projection: bool = False
    action_native_prefill_static_aligned: bool = False
    cpu_seal_pack_workers: int = 1
    cpu_seal_serial_allocate: bool = False
    query_replay_backend: str = 'eager'
    query_replay_cobatch: bool = False
    query_replay_max_bytes: int = 67108864
    query_score_mode: str = 'mean_k'
    query_window_cache_bytes: int = 2147483648
    query_fps_halo_blocks: int = 0
    query_fps_optimized: bool = False
    query_fps_prebuild: bool = False
    query_fps_prebuild_device: str = 'cpu'
    graph_query_trace_max_bytes: int = 16777216
    query_fps_cpu_backend: str = 'torch'
    query_fps_cpu_workers: int = 8
    query_mean_quota_fraction: float = 0.0
    cpu_release_sealed_sums: bool = False
    query_fps_cache_fp8: bool = False

    @property
    def active_residency_enabled(self):
        return self.active_resident_pressure_only or bool(self.active_resident_ttl_ms)

    @property
    def guided_reserve(self):
        if not self.guided_query_tokens:
            return 0
        block = self.memory.block_tokens
        return ((256 + self.guided_query_tokens + block - 1) // block) * block

    @property
    def guided_trigger(self):
        if self.guided_trigger_tokens:
            return self.guided_trigger_tokens
        block, generation = self.memory.block_tokens, self.memory.gen_budget
        guard = max(min(4096, generation // 4), self.guided_reserve + 3*block)
        return max(block, (generation-guard)//block*block)

    def __post_init__(self):
        import math
        from . import _native
        from .historical_gc import HistoricalGcSettings
        historical = HistoricalGcSettings.parse(self.historical_gc)
        object.__setattr__(self, 'historical_gc', historical)
        if historical is not None and (self.archive_backend != 'cpu' or self.repository_gc or
                not self.request_planning or not self.async_archive_retirement or not self.async_seal or
                not self.cpu_history_coordination or not self.session_capacity or self.graph_mode != 'off'):
            raise ValueError('historical_gc requires CPU authority, planning, async seal/retirement, '
                             'CPU history coordination and owned sessions; repository_gc and block graph must be disabled')
        if historical is not None and historical.ready_high > self.memory.repository_blocks:
            raise ValueError('historical_gc watermarks exceed repository capacity')
        if historical is not None and historical.policy == 'redundancy':
            incompatible = ('offline_prefill_coalescing', 'prefill_physical_coalescing',
                            'prefill_publication_pipeline', 'early_capture_slot_release',
                            'prefill_capture_ahead')
            enabled = [name for name in incompatible if getattr(self, name)]
            if enabled:
                raise ValueError('historical_gc redundancy does not yet support accepted Query '
                                 'publication with: '+', '.join(enabled))
        if (historical is not None and self.query_score_mode == 'key_fps_4' and
                not historical.policy_params['keep_mandatory']):
            raise ValueError('Historical FPS retrieval requires keep_mandatory=true for Query and checkpoint holds')
        supplied = {name: getattr(self, name) for name in _POLICY_FIELDS
                    if getattr(self, name) is not _UNSET}
        explicit = (frozenset(supplied) if self._policy_explicit_fields is None
                    else self._policy_explicit_fields)
        if self._policy_resolved_values is not None:
            # dataclasses.replace carries already resolved defaults. Preserve
            # their provenance, while treating actual replacement values as
            # explicit settings instead of silently ignoring a policy change.
            explicit = explicit | frozenset(name for name, previous in self._policy_resolved_values
                if supplied.get(name, _UNSET) != previous)
        resolved = resolve_graph_policy(supplied, explicit_fields=explicit)
        for name in _POLICY_FIELDS:
            object.__setattr__(self, name, resolved[name])
        object.__setattr__(self, '_policy_explicit_fields', explicit)
        object.__setattr__(self, '_policy_resolved_values',
                           tuple((name, resolved[name]) for name in sorted(_POLICY_FIELDS)))
        validate_query_score_mode(self.query_score_mode, self.query_window_cache_bytes,
            block_tokens=self.memory.block_tokens, graph_mode=self.graph_mode,
            graph_scoring_version=self.graph_scoring_version,
            graph_aggregation=self.graph_aggregation, graph_alpha=self.graph_alpha,
            query_history=self.query_history, guided_query_tokens=self.guided_query_tokens,
            mean_quota_fraction=self.query_mean_quota_fraction, historical_gc=historical)
        validate_query_fps_experiments(self.query_score_mode,
            self.query_fps_halo_blocks, self.query_fps_optimized, self.query_fps_prebuild,
            self.query_fps_prebuild_device, self.query_fps_cpu_backend, self.query_fps_cpu_workers,
            mean_quota_fraction=self.query_mean_quota_fraction)
        validate_graph_query_trace_bytes(self.graph_query_trace_max_bytes)
        for name in ('cpu_release_sealed_sums', 'query_fps_cache_fp8'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name+' must be boolean')
        if (self.cpu_release_sealed_sums or self.query_fps_cache_fp8) and (
                self.archive_backend != 'cpu' or self.mean_k_backend != 'qw3_cuda'):
            raise ValueError('Compact CPU indices require CPU native MeanK')
        if self.query_fps_cache_fp8 and self.query_score_mode != 'key_fps_4':
            raise ValueError('query_fps_cache_fp8 requires key_fps_4')
        if self.query_score_mode in ('overlap_16_8', 'key_fps_4'):
            from .query_window_budget import split_cpu_history_budget
            split_cpu_history_budget(self)
        validate_graph_options({name: getattr(self, name) for name in GRAPH_DEFAULTS},
            request_planning=self.request_planning, session_capacity=self.session_capacity,
            repository_gc=self.repository_gc, query_history=self.query_history,
            action_pressure=self.action_pressure, action_attention=self.memory.action_attention,
            prefill_attention=self.prefill_attention, action_scope=self.action_scope,
            action_fraction=self.memory.action_fraction, pressure_policy=self.pressure_policy)
        if type(self.async_action_metadata) is not bool:
            raise ValueError('Asynchronous Action metadata must be boolean')
        if type(self.action_native_decode_commit) is not bool:
            raise ValueError('Native decode commit must be boolean')
        if type(self.action_download_pool) is not bool:
            raise ValueError('Action download pool must be boolean')
        if type(self.action_fused_scoring) is not bool:
            raise ValueError('Fused Action scoring must be boolean')
        if (type(self.action_download_max_bytes) is not int or self.action_download_max_bytes<256 or
                type(self.action_download_max_slots) is not int or self.action_download_max_slots<1):
            raise ValueError('Action download pool requires bounded positive byte/event capacity')
        if type(self.action_commit_max_bytes) is not int or self.action_commit_max_bytes < 4096:
            raise ValueError('Action commit capacity must be at least 4096 bytes')
        if self.action_native_scores not in ('off', 'verify', 'reuse'):
            raise ValueError('Native Action scores must be off, verify or reuse')
        if type(self.action_native_prefill_aligned_pieces) is not bool:
            raise ValueError('Native prefill aligned pieces must be boolean')
        if self.action_native_prefill_aligned_pieces and self.action_native_scores == 'off':
            raise ValueError('Native prefill aligned pieces require native Action scores')
        if type(self.action_native_prefill_cta_gate) is not bool:
            raise ValueError('Native prefill CTA gate must be boolean')
        if type(self.capture_opaque_projection) is not bool:
            raise ValueError('Opaque capture projection must be boolean')
        if self.action_native_prefill_cta_gate and self.action_native_scores == 'off':
            raise ValueError('Native prefill CTA gate requires native Action scores')
        if type(self.action_native_prefill_static_aligned) is not bool:
            raise ValueError('Native prefill static aligned must be boolean')
        if self.action_native_prefill_static_aligned:
            if self.action_native_scores == 'off':
                raise ValueError('Native prefill static aligned requires native Action scores')
            if self.action_native_prefill_aligned_pieces or self.action_native_prefill_cta_gate:
                raise ValueError('Native prefill static aligned cannot combine runtime aligned pieces or CTA gate')
        if (type(self.action_native_max_rows) is not int or self.action_native_max_rows < 1 or
                type(self.action_native_max_bytes) is not int or self.action_native_max_bytes < 1):
            raise ValueError('Native Action buffer limits must be positive integers')
        if self.action_native_scores != 'off' and not self.memory.action_attention:
            raise ValueError('Native Action scores require action attention')
        if self.async_action_metadata and not self.memory.action_attention:
            raise ValueError('Asynchronous Action metadata requires action attention')
        if type(self.early_capture_slot_release) is not bool:
            raise ValueError('Early capture slot release must be boolean')
        if (type(self.early_capture_feedback_bytes) is not int or
                self.early_capture_feedback_bytes <= 0):
            raise ValueError('Early capture feedback bytes must be positive')
        if self.early_capture_slot_release and not self.async_cpu_scatter:
            raise ValueError('Early capture slot release requires async CPU scatter')
        if self.early_capture_slot_release and not self.split_capture_dma:
            raise ValueError('Early capture slot release requires split capture DMA')
        if type(self.prefill_publication_pipeline) is not bool:
            raise ValueError('prefill_publication_pipeline must be boolean')
        if self.prefill_publication_pipeline:
            if not (self.async_cpu_scatter and self.split_capture_dma and
                    self.memory.action_attention and self.prefill_attention):
                raise ValueError('Prefill publication pipeline requires async CPU scatter, '
                                 'split capture DMA and prefill action attention')
            if self.early_capture_slot_release:
                raise ValueError('Prefill publication pipeline requires completion-owned capture slots')
        if type(self.prefill_native_plan_arm) is not bool:
            raise ValueError('prefill_native_plan_arm must be boolean')
        if self.prefill_native_plan_arm and (not self.prefill_publication_pipeline or
                                             self.action_native_scores == 'off'):
            raise ValueError('Prefill native plan arm requires the publication pipeline and native scores')
        if type(self.prefill_capture_ahead) is not bool:
            raise ValueError('prefill_capture_ahead must be boolean')
        if type(self.capture_fused_pack) is not bool:
            raise ValueError('capture_fused_pack must be boolean')
        if type(self.prefill_action_before_mtp) is not bool:
            raise ValueError('prefill_action_before_mtp must be boolean')
        if self.prefill_action_before_mtp and (not self.prefill_publication_pipeline or
                                               self.action_native_scores != 'reuse'):
            raise ValueError('Prefill action before MTP requires the publication pipeline and native reuse')
        if type(self.prefill_deferred_validation) is not bool:
            raise ValueError('prefill_deferred_validation must be boolean')
        if type(self.prefill_scatter_after_target) is not bool:
            raise ValueError('prefill_scatter_after_target must be boolean')
        if self.prefill_scatter_after_target and not (
                self.async_cpu_scatter and self.split_capture_dma and
                self.prefill_publication_pipeline and self.prefill_native_plan_arm):
            raise ValueError('Prefill scatter after target requires asynchronous split capture, '
                             'publication pipeline and native prefill plan arming')
        if self.prefill_deferred_validation and not (
                self.prefill_action_before_mtp and self.prefill_native_plan_arm and
                self.async_archive_retirement):
            raise ValueError('Deferred prefill validation requires early MTP scoring, '
                             'native plan arm and asynchronous archive retirement')
        for name in ('prefill_deferred_validation_max_bytes',
                     'prefill_deferred_validation_scratch_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive integer')
        if self.prefill_capture_ahead and (not self.prefill_publication_pipeline or
                                          self.action_native_scores == 'off'):
            raise ValueError('Prefill capture ahead requires the publication pipeline and native scores')
        if type(self.prefill_physical_coalescing) is not bool:
            raise ValueError('prefill_physical_coalescing must be boolean')
        if (type(self.prefill_physical_max_chunks) is not int or
                not 2 <= self.prefill_physical_max_chunks <= 8):
            raise ValueError('prefill_physical_max_chunks must be an integer in [2, 8]')
        if self.prefill_physical_coalescing:
            if not (self.request_planning and self.prefill_publication_pipeline and
                    self.prefill_native_plan_arm and self.action_native_scores == 'reuse' and
                    self.batched_cpu_append and self.graph_mode != 'off'):
                raise ValueError('Physical prefill coalescing requires request planning, '
                                 'the publication pipeline, native plan arm, native reuse, '
                                 'batched CPU append and an active graph')
            if (self.early_capture_slot_release or self.prefill_deferred_validation or
                    self.prefill_scatter_after_target or self.offline_prefill_coalescing or
                    self.offline_phase_policy or self.prefill_batch_tokens or self.adaptive_prefill or
                    self.guided_query_tokens):
                raise ValueError('Physical prefill coalescing does not qualify early slot release, '
                                 'deferred validation/scatter, other coalescing or prefill budget '
                                 'controllers, or guided Query')
            if self.action_native_max_rows < 64*self.prefill_physical_max_chunks:
                raise ValueError('Physical prefill coalescing requires native sample capacity for every logical chunk')
            if not self.action_download_pool or self.action_download_max_slots < 4:
                raise ValueError('Physical prefill coalescing requires four bounded Action download slots')
        if type(self.rolling_active_window) is not bool:
            raise ValueError('Rolling active window must be boolean')
        if self.rolling_active_window and not (self.request_planning and
                self.async_refresh == self.async_restore and
                self.cpu_history_coordination and self.session_capture_terminal):
            raise ValueError('Rolling active window requires planned CPU history sessions, '
                             'terminal capture and matching refresh/restore modes')
        if type(self.restore_prepare_split) is not bool or (self.restore_prepare_split and not self.async_restore):
            raise ValueError('restore_prepare_split requires asynchronous restore')
        if (type(self.session_gpu_checkpoint_bytes) is not int or
                not 0 <= self.session_gpu_checkpoint_bytes <= 16 * 1024**3):
            raise ValueError('session_gpu_checkpoint_bytes must be an integer in [0, 16 GiB]')
        if self.session_gpu_checkpoint_bytes and (not self.async_restore or
                not self.session_capture_terminal or not self.session_capacity):
            raise ValueError('GPU model checkpoints require asynchronous terminal sessions')
        if (type(self.resident_restore_slots) is not int or
                not 1 <= self.resident_restore_slots <= 32 or
                (self.resident_restore_slots > 1 and not self.async_restore)):
            raise ValueError('Resident restore slots must be 1..32 and require async restore when greater than one')
        for name in ('split_capture_dma', 'native_prefill_commit', 'async_checkpoint_capture',
                     'restore_stage_timing', 'host_stage_timing', 'native_checkpoint_capture',
                     'native_restore_frame', 'native_action_metadata',
                     'action_profiling'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name in ('action_profile_interval', 'action_profile_max_events'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.action_profiling and not self.memory.action_attention:
            raise ValueError('Action profiling requires action_attention')
        if type(self.offline_agent_policy) is not bool:
            raise ValueError('Offline agent policy must be boolean')
        if type(self.offline_prefill_coalescing) is not bool:
            raise ValueError('Offline prefill coalescing must be boolean')
        if self.offline_prefill_coalescing and (not self.offline_agent_policy or self.offline_phase_policy):
            raise ValueError('Offline prefill coalescing requires continuous offline agent policy without phase gating')
        if (type(self.offline_prefill_max_tokens) is not int or self.offline_prefill_max_tokens <= 0 or
                (self.offline_prefill_coalescing and
                 (self.offline_prefill_max_tokens % self.memory.block_tokens or
                  self.offline_prefill_max_tokens > self.memory.gen_budget - self.memory.block_tokens))):
            raise ValueError('Offline prefill maximum must be positive, aligned and fit G headroom')
        if type(self.offline_phase_policy) is not bool:
            raise ValueError('Offline phase policy must be boolean')
        if self.offline_phase_policy and not self.offline_agent_policy:
            raise ValueError('Offline phase policy requires the coordinated offline agent policy')
        if (type(self.phase_prefill_wave_tokens) is not int or self.phase_prefill_wave_tokens <= 0 or
                (self.offline_phase_policy and (self.phase_prefill_wave_tokens % self.memory.block_tokens or
                 self.phase_prefill_wave_tokens > self.memory.gen_budget))):
            raise ValueError('Phase prefill wave must be positive, block aligned and fit G')
        if (type(self.phase_max_wait_ms) not in (int, float) or
                not math.isfinite(self.phase_max_wait_ms) or self.phase_max_wait_ms <= 0):
            raise ValueError('Phase maximum wait must be positive and finite')
        if type(self.active_resident_pressure_only) is not bool:
            raise ValueError('Pressure-only active residency must be boolean')
        if self.offline_agent_policy and (not self.cpu_history_coordination or not self.async_restore or
                not self.active_residency_enabled or not self.step_latency_interval or
                self.adaptive_prefill or self.maintenance_transfer_bytes_per_second or
                self.prefill_cadence != 1):
            raise ValueError('Offline agent policy requires coordinated CPU history, resident async restore and timing; '
                             'independent latency and fixed-rate controllers must be disabled')
        if self.prefill_batch_tokens and (self.offline_phase_policy or self.offline_prefill_coalescing):
            raise ValueError('Shared prefill budget supports continuous offline ordering only; '
                             'independent phase and coalescing controllers must be disabled')
        if (type(self.maintenance_transfer_bytes_per_second) is not int or self.maintenance_transfer_bytes_per_second < 0 or
                type(self.maintenance_transfer_burst_bytes) is not int or self.maintenance_transfer_burst_bytes <= 0):
            raise ValueError('Invalid background KV transfer rate or burst')
        if self.maintenance_transfer_bytes_per_second and not self.maintenance_max_steps:
            raise ValueError('Transfer pacing requires background maintenance coordination')
        if (type(self.active_resident_ttl_ms) not in (int, float) or
                not math.isfinite(self.active_resident_ttl_ms) or self.active_resident_ttl_ms < 0 or
                type(self.active_resident_max_fraction) not in (int, float) or
                not math.isfinite(self.active_resident_max_fraction) or
                not 0 < self.active_resident_max_fraction <= 1):
            raise ValueError('Invalid idle active-view TTL or pool fraction')
        if self.active_resident_pressure_only and self.active_resident_ttl_ms:
            raise ValueError('Pressure-only residency and finite TTL are mutually exclusive')
        if (type(self.active_tool_grace_ms) not in (int, float) or
                not math.isfinite(self.active_tool_grace_ms) or self.active_tool_grace_ms < 0 or
                type(self.active_tool_max_wait_ms) not in (int, float) or
                not math.isfinite(self.active_tool_max_wait_ms) or self.active_tool_max_wait_ms <= 0 or
                self.active_tool_grace_ms > self.active_tool_max_wait_ms):
            raise ValueError('Invalid tool grace or maximum queue wait')
        if self.active_tool_grace_ms and not self.active_resident_pressure_only:
            raise ValueError('Tool grace requires pressure-only active residency')
        if self.active_residency_enabled and (not self.async_restore or not self.session_capture_terminal):
            raise ValueError('Idle active residency requires exact terminal capture and async restore')
        if (type(self.prefill_batch_tokens) is not int or self.prefill_batch_tokens < 0 or
                self.prefill_batch_tokens % self.memory.block_tokens or
                (self.prefill_batch_tokens and not self.request_planning)):
            raise ValueError('Shared prefill budget requires explicit planning and aligned nonnegative tokens')
        if (type(self.prefill_request_tokens) is not int or self.prefill_request_tokens < 0 or
                self.prefill_request_tokens % self.memory.block_tokens or
                self.prefill_request_tokens > self.prefill_batch_tokens):
            raise ValueError('Per-request prefill cap requires aligned tokens within an enabled shared budget')
        if type(self.adaptive_prefill) is not bool or type(self.step_latency_interval) is not int or not 0 <= self.step_latency_interval <= 1024:
            raise ValueError('Invalid adaptive prefill/step timing settings')
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in
               (self.decode_step_target_ms, self.prefill_feedback_timeout_ms)):
            raise ValueError('Prefill feedback time settings must be positive and finite')
        if any(type(v) is not int or v <= 0 for v in (self.prefill_min_tokens, self.prefill_max_tokens)) or self.prefill_min_tokens > self.prefill_max_tokens:
            raise ValueError('Invalid adaptive prefill chunk bounds')
        if self.adaptive_prefill and (not self.step_latency_interval or not self.request_planning or
                self.prefill_min_tokens % self.memory.block_tokens or self.prefill_max_tokens % self.memory.block_tokens or
                self.prefill_max_tokens > self.memory.gen_budget - self.memory.block_tokens):
            raise ValueError('Adaptive prefill requires timing, explicit planning and aligned chunks within G headroom')
        frozen_prompt = (self.session_checkpoint_policy == 'sealed' and self.session_freeze_prompt and
                         not self.session_capture_terminal and not self.session_capture_decode and
                         not self.query_history)
        if type(self.async_restore) is not bool or (self.async_restore and (
                not self.async_refresh or not self.maintenance_max_steps or not self.session_capacity or
                self.repository_gc or self.archive_backend != 'cpu' or
                (self.session_checkpoint_policy != 'pm' and not frozen_prompt))):
            raise ValueError('Asynchronous restore requires CPU P/M or frozen-P sessions, non-GC async refresh and maintenance budgets')
        if type(self.restore_handoff_fastpath) is not bool or (self.restore_handoff_fastpath and (
                not self.async_restore or not self.restore_prepare_split or self.offline_agent_policy)):
            raise ValueError('Restore handoff fastpath requires split asynchronous restore without offline ordering')
        if type(self.cpu_history_coordination) is not bool or type(self.cpu_progress_reserve) is not bool:
            raise ValueError('CPU history coordination settings must be boolean')
        if self.cpu_history_coordination and (self.archive_backend != 'cpu' or self.repository_gc or
                not self.async_archive_retirement or not self.request_planning or not self.session_capacity):
            raise ValueError('CPU history coordination requires CPU authority, planning and fenced non-GC retirement')
        if type(self.maintenance_max_steps) is not int or not 0 <= self.maintenance_max_steps <= 64:
            raise ValueError('Maintenance step limit must be in [0, 64]')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in
               (self.maintenance_busy_ms, self.maintenance_idle_ms, self.maintenance_aging_ms)):
            raise ValueError('Maintenance time budgets must be positive and finite')
        if type(self.maintenance_inflight_bytes) is not int or self.maintenance_inflight_bytes < 0:
            raise ValueError('Maintenance transfer budget must be nonnegative')
        if self.maintenance_max_steps and not self.async_refresh:
            raise ValueError('Maintenance coordination requires asynchronous refresh')
        if self.maintenance_inflight_bytes and not self.maintenance_max_steps:
            raise ValueError('Maintenance transfer budget requires maintenance coordination')
        if type(self.session_freeze_prompt) is not bool or (self.session_freeze_prompt and (
                not self.request_planning or not self.session_capacity or self.session_capture_terminal or
                self.session_capture_decode or self.query_history)):
            raise ValueError('Frozen prompt sessions require explicit planning, P-only checkpoints and no cross-question Query history')
        if (type(self.prefill_attention) is not bool or type(self.action_pressure) is not bool or
                type(self.prefill_attention_interval) is not int or self.prefill_attention_interval < 1 or
                type(self.prefill_attention_stride) is not int or self.prefill_attention_stride < 0 or
                type(self.prefill_attention_half_life) not in (int,float) or
                not math.isfinite(self.prefill_attention_half_life) or self.prefill_attention_half_life <= 0):
            raise ValueError('Invalid prefill attention settings')
        if (self.prefill_attention or self.action_pressure) and not self.memory.action_attention:
            raise ValueError('Prefill/pressure attention requires action attention')
        if type(self.async_refresh) is not bool or type(self.refresh_quantum_tokens) is not int or not 1 <= self.refresh_quantum_tokens <= 2048:
            raise ValueError('Invalid asynchronous refresh settings')
        if self.async_refresh and (not self.request_planning or self.repository_gc):
            raise ValueError('Asynchronous refresh requires request planning and bounded non-GC history')
        if self.query_replay_backend not in ('eager', 'compiled'):
            raise ValueError('Query replay backend must be eager or compiled')
        if type(self.query_replay_cobatch) is not bool:
            raise ValueError('Query replay cobatch must be boolean')
        if type(self.query_replay_max_bytes) is not int or self.query_replay_max_bytes <= 0:
            raise ValueError('Query replay memory budget must be positive')
        if (self.query_replay_backend != 'eager' or self.query_replay_cobatch) and not self.async_refresh:
            raise ValueError('Optimized Query replay requires asynchronous refresh')
        if (type(self.action_interval) is not int or not 1<=self.action_interval<=4096 or
                type(self.action_decay) not in (float,int) or not math.isfinite(self.action_decay) or
                not 0<=self.action_decay<=1 or not isinstance(self.action_scope,str) or
                not isinstance(self.action_layers,str)):
            raise ValueError('Invalid action attention scope/layers/interval/decay')
        _native.validate_action_options(self.action_scope, self.action_layers,
                                       self.action_interval, self.action_decay)
        if self.memory.action_attention and not self.session_capacity:
            raise ValueError('Action attention requires owned prompt sessions')
        if type(self.query_history) is not bool:
            raise ValueError('Query history must be boolean')
        if (type(self.query_history_decay) not in (float, int) or
                not math.isfinite(self.query_history_decay) or not 0 <= self.query_history_decay <= 1):
            raise ValueError('Query history decay must be in [0, 1]')
        if self.guided_query_backend not in ('eager', 'cuda_graph'):
            raise ValueError('Guided Query backend must be eager or cuda_graph')
        if self.guided_reselect not in ('off', 'boundary', 'middecode', 'both', 'prefill'):
            raise ValueError('guided_reselect must be off, boundary, middecode, both or prefill')
        if self.guided_query_tokens and self.guided_reselect == 'prefill':
            if not self.async_refresh or self.pressure_policy != 'attention':
                raise ValueError('Guided prefill pressure requires async_refresh and attention fallback')
            if self.query_history or self.memory.recent_blocks:
                raise ValueError('Guided prefill pressure requires query_history=false and recent_blocks=0')
        if type(self.guided_trigger_tokens) is not int or self.guided_trigger_tokens < 0:
            raise ValueError('guided_trigger_tokens must be a nonnegative integer')
        if type(self.guided_max_refreshes) is not int or not 0 <= self.guided_max_refreshes <= 8:
            raise ValueError('guided_max_refreshes must be in [0, 8]')
        if self.guided_query_tokens and self.guided_reselect in ('middecode', 'both'):
            if not self.rolling_active_window:
                raise ValueError('Guided automatic refresh requires rolling_active_window')
            if self.async_refresh:
                raise ValueError('Guided automatic refresh currently requires synchronous refresh')
            if not 0 < self.guided_trigger < self.memory.gen_budget:
                raise ValueError('Guided refresh trigger must be smaller than G')
            if self.guided_trigger + self.guided_reserve + 2*self.memory.block_tokens > self.memory.gen_budget:
                raise ValueError('Guided refresh trigger leaves insufficient private headroom')
        if type(self.guided_query_tokens) is not int or not 0 <= self.guided_query_tokens <= 4096:
            raise ValueError('Guided query tokens must be in [0, 4096]')
        if self.archive_backend not in ('nvme', 'cpu'):
            raise ValueError('Archive backend must be nvme or cpu')
        if self.archive_backend == 'cpu' and (not self.cpu_read_cache_bytes or self.nvme_preallocate or self.nvme_max_bytes):
            raise ValueError('CPU-only archive requires a memory budget and no NVMe reservation')
        if type(self.cpu_direct_record_seal) is not bool:
            raise ValueError('cpu_direct_record_seal must be boolean')
        if self.cpu_direct_record_seal and (self.archive_backend != 'cpu' or not self.async_seal):
            raise ValueError('cpu_direct_record_seal requires archive_backend=cpu and async_seal=true')
        if type(self.cpu_mean_index_simd) is not bool:
            raise ValueError('cpu_mean_index_simd must be boolean')
        if self.cpu_mean_index_simd and self.mean_k_backend != 'qw3_cuda':
            raise ValueError('cpu_mean_index_simd requires mean_k_backend=qw3_cuda')
        if type(self.assembly_read_prefetch) is not bool:
            raise ValueError('assembly_read_prefetch must be boolean')
        if self.assembly_read_prefetch and not (
                self.archive_backend == 'cpu' and self.packed_assembly and self.direct_pinned_reads):
            raise ValueError('assembly_read_prefetch requires CPU archive and direct packed reads')
        if type(self.assembly_fused_layers) is not bool:
            raise ValueError('assembly_fused_layers must be boolean')
        if self.assembly_fused_layers and not (
                self.archive_backend == 'cpu' and self.packed_assembly and self.direct_pinned_reads):
            raise ValueError('assembly_fused_layers requires CPU archive and direct packed reads')
        if type(self.cpu_seal_hash_workers) is not int or not 1 <= self.cpu_seal_hash_workers <= 8:
            raise ValueError('cpu_seal_hash_workers must be an integer in [1, 8]')
        if self.cpu_seal_hash_workers > 1 and (self.archive_backend != 'cpu' or not self.async_seal):
            raise ValueError('cpu_seal_hash_workers > 1 requires archive_backend=cpu and async_seal=true')
        if type(self.cpu_seal_pack_workers) is not int or not 1 <= self.cpu_seal_pack_workers <= 8:
            raise ValueError('cpu_seal_pack_workers must be an integer in [1, 8]')
        if self.cpu_seal_pack_workers > 1 and (not self.cpu_direct_record_seal or self.historical_gc is not None):
            raise ValueError('cpu_seal_pack_workers > 1 requires direct CPU record sealing without historical_gc')
        if type(self.cpu_seal_serial_allocate) is not bool:
            raise ValueError('cpu_seal_serial_allocate must be boolean')
        if self.cpu_seal_serial_allocate and (not self.cpu_direct_record_seal or
                self.historical_gc is not None or self.cpu_seal_pack_workers != 1):
            raise ValueError('cpu_seal_serial_allocate requires direct CPU record sealing without historical_gc and pack_workers=1')
        if (self.query_history or self.guided_query_tokens) and not self.request_planning:
            raise ValueError('Query history/guided refresh require explicit request planning')
        if self.query_history and not self.session_capacity:
            raise ValueError('Query history requires bounded prompt sessions')
        if self.guided_query_tokens and self.guided_reserve + 3 * self.memory.block_tokens > self.memory.gen_budget:
            raise ValueError('Guided scratch reserve must fit generation headroom')
        if self.session_checkpoint_policy not in ('sealed','pm'):
            raise ValueError('Invalid checkpoint policy')
        if self.session_checkpoint_policy=='pm' and self.session_capacity and not self.session_capture_terminal:
            raise ValueError('P/M checkpoints require terminal capture')
        if type(self.request_planning) is not bool or self.mean_k_backend not in ('torch_reference', 'qw3_cuda'):
            raise ValueError('Invalid request planning or MeanK backend')
        if self.request_planning and self.memory.gen_budget < 3 * self.memory.block_tokens:
            raise ValueError('Explicit planning requires at least three blocks of G replay headroom')
        if type(self.cpu_scatter_min_tokens) is not int or not 1<=self.cpu_scatter_min_tokens<=65536:
            raise ValueError('CPU scatter minimum must be an integer in [1, 65536]')
        if type(self.cpu_gather_workers) is not int or not 1 <= self.cpu_gather_workers <= 16:
            raise ValueError('CPU gather workers must be an integer in [1, 16]')
        if type(self.assembly_chunk_blocks) is not int or not 1 <= self.assembly_chunk_blocks <= 256:
            raise ValueError('Assembly chunk blocks must be an integer in [1, 256]')
        if any(type(value) is not int or value < 0 for value in
               (self.cpu_read_cache_bytes, self.nvme_max_bytes, self.archive_io_max_bytes,
                self.session_capacity, self.session_checkpoint_bytes, self.cpu_history_cache_bytes_per_request,
                self.gpu_raw_k_cache_bytes)):
            raise ValueError("Storage capacity must be a nonnegative byte count")
        if self.cpu_history_cache_bytes_per_request > self.cpu_read_cache_bytes:
            raise ValueError('Per-request historical cache must fit the global CPU cache budget')
        if any(type(value) is not bool for value in
               (self.packed_assembly, self.verify_archive_on_finish, self.direct_pinned_reads,
                self.async_archive_retirement, self.repository_gc, self.nvme_preallocate,
                self.session_capture_decode, self.async_seal, self.reuse_resident_v,self.async_cpu_scatter,self.batched_cpu_append)):
            raise ValueError("Assembly and archive verification settings must be boolean")
        if type(self.session_capture_terminal) is not bool:
            raise ValueError('Terminal checkpoint setting must be boolean')
        if self.direct_pinned_reads and not self.packed_assembly:
            raise ValueError("Direct pinned reads require packed assembly")
        if (self.gpu_raw_k_cache_bytes or self.reuse_resident_v) and not self.direct_pinned_reads:
            raise ValueError('GPU raw cache and V reuse require direct packed assembly')
        if type(self.prefill_cadence) is not int or not 1 <= self.prefill_cadence <= 64:
            raise ValueError("Prefill cadence must be an integer in [1, 64]")
        if bool(self.session_capacity) != bool(self.session_checkpoint_bytes):
            raise ValueError('Prompt sessions require both entry and checkpoint-byte capacity')
        if (type(self.checkpoint_pinned_prewarm_count) is not int or
                not 0 <= self.checkpoint_pinned_prewarm_count <= 64 or
                (self.checkpoint_pinned_prewarm_count and not self.session_capacity)):
            raise ValueError('Pinned checkpoint prewarm needs 0..64 buffers and enabled sessions')
        if self.session_capacity and not self.async_archive_retirement:
            raise ValueError('Prompt sessions require fenced asynchronous authority retirement')
        if self.session_capture_decode and not self.session_capacity:
            raise ValueError('Decode checkpoints require session capacity')
        if self.session_capture_terminal and not self.session_capacity:
            raise ValueError('Terminal checkpoints require session capacity')
        if self.session_capacity and self.memory.repository_blocks <= 2 * (
                self.memory.select_budget + self.memory.gen_budget) // self.memory.block_tokens:
            raise ValueError('Prompt sessions require room for checkpoint and live working frames')

    @classmethod
    def from_vllm(cls, config):
        transfer = config.kv_transfer_config
        extra = transfer.kv_connector_extra_config if transfer is not None else {}
        if not isinstance(extra, dict):
            raise ValueError("Connector extra config must be an object")
        payload = extra.get('kvmem', {})
        if not isinstance(payload, dict):
            raise ValueError("kvmem connector settings must be an object")
        payload = dict(payload)
        cache_bytes = payload.pop('cpu_read_cache_bytes', 0)
        packed = payload.pop('packed_assembly', False)
        verify = payload.pop('verify_archive_on_finish', True)
        direct = payload.pop('direct_pinned_reads', False)
        cadence = payload.pop('prefill_cadence', 1)
        retirement = payload.pop('async_archive_retirement', False)
        gc = payload.pop('repository_gc', False)
        historical_gc = payload.pop('historical_gc', None)
        preallocate = payload.pop('nvme_preallocate', False)
        nvme_bytes = payload.pop('nvme_max_bytes', 0)
        io_bytes = payload.pop('archive_io_max_bytes', 0)
        session_capacity = payload.pop('session_capacity', 0)
        session_bytes = payload.pop('session_checkpoint_bytes', 0)
        capture_decode = payload.pop('session_capture_decode', False)
        history_bytes = payload.pop('cpu_history_cache_bytes_per_request', 0)
        capture_terminal = payload.pop('session_capture_terminal', False)
        gather_workers = payload.pop('cpu_gather_workers', 8)
        async_seal = payload.pop('async_seal', False)
        direct_record_seal = payload.pop('cpu_direct_record_seal', False)
        seal_hash_workers = payload.pop('cpu_seal_hash_workers', 1)
        seal_pack_workers = payload.pop('cpu_seal_pack_workers', 1)
        seal_serial_allocate = payload.pop('cpu_seal_serial_allocate', False)
        raw_cache = payload.pop('gpu_raw_k_cache_bytes', 0)
        reuse_v = payload.pop('reuse_resident_v', False)
        cpu_scatter = payload.pop('async_cpu_scatter', False)
        scatter_min = payload.pop('cpu_scatter_min_tokens',128)
        batched_append = payload.pop('batched_cpu_append', False)
        planning = payload.pop('request_planning', False)
        scorer = payload.pop('mean_k_backend', 'torch_reference')
        checkpoint_policy = payload.pop('session_checkpoint_policy','sealed')
        query_history = payload.pop('query_history', False)
        query_decay = payload.pop('query_history_decay', 0.5)
        guided_tokens = payload.pop('guided_query_tokens', 0)
        archive_backend = payload.pop('archive_backend', 'nvme')
        graph = {name: payload.pop(name) for name in GRAPH_DEFAULTS if name in payload}
        graph['pressure_policy'] = payload.pop('pressure_policy', _UNSET)
        action = {name: payload.pop(name, default) for name, default in (
            ('query_score_mode', 'mean_k'), ('query_window_cache_bytes', 2147483648),
            ('query_fps_halo_blocks', 0), ('query_fps_optimized', False),
            ('query_fps_prebuild', False),
            ('query_fps_prebuild_device', 'cpu'),
            ('graph_query_trace_max_bytes', 16777216),
            ('query_fps_cpu_backend', 'torch'), ('query_fps_cpu_workers', 8),
            ('query_mean_quota_fraction', 0.0),
            ('cpu_release_sealed_sums', False), ('query_fps_cache_fp8', False),
            ('assembly_chunk_blocks', 32),
            ('guided_query_backend', 'eager'),
            ('query_replay_backend', 'eager'), ('query_replay_cobatch', False),
            ('query_replay_max_bytes', 67108864),
            ('guided_reselect', 'boundary'), ('guided_trigger_tokens', 0),
            ('guided_max_refreshes', 2),
            ('action_scope', 'tool-call'), ('action_layers', 'spread3'),
            ('action_interval', 16), ('action_decay', 0.5),
            ('action_native_decode_commit', True), ('action_commit_max_bytes', 16777216),
            ('action_download_pool', True), ('action_download_max_bytes', 67108864),
            ('action_download_max_slots', 128),
            ('action_fused_scoring', True),
            ('action_native_scores', 'off'), ('action_native_max_rows', 256),
            ('action_native_prefill_aligned_pieces', False),
            ('action_native_prefill_cta_gate', False),
            ('capture_opaque_projection', False),
            ('action_native_prefill_static_aligned', False),
            ('action_native_max_bytes', 2147483648),
            ('async_refresh', False), ('refresh_quantum_tokens', 256),
            ('prefill_attention', False), ('action_pressure', _UNSET),
            ('prefill_attention_interval', 1024), ('prefill_attention_stride', 32),
            ('prefill_attention_half_life', 8192), ('session_freeze_prompt', False),
            ('cpu_history_coordination', False), ('cpu_progress_reserve', True),
            ('maintenance_max_steps', 0), ('maintenance_busy_ms', 2.0),
            ('maintenance_idle_ms', 8.0), ('maintenance_aging_ms', 250.0),
            ('maintenance_inflight_bytes', 0), ('async_restore', False),
            ('adaptive_prefill', False), ('step_latency_interval', 0),
            ('decode_step_target_ms', 50.0), ('prefill_min_tokens', 128),
            ('prefill_max_tokens', 2048), ('prefill_feedback_timeout_ms', 5000.0),
            ('prefill_batch_tokens', 0), ('prefill_request_tokens', 0), ('active_resident_ttl_ms', 0.0),
            ('active_resident_pressure_only', False),
            ('active_tool_grace_ms', 0.0), ('active_tool_max_wait_ms', 15000.0),
            ('active_resident_max_fraction', 0.5), ('maintenance_transfer_bytes_per_second', 0),
            ('maintenance_transfer_burst_bytes', 67108864), ('offline_agent_policy', False),
            ('offline_prefill_coalescing', False), ('offline_prefill_max_tokens', 4096),
            ('offline_phase_policy', False), ('phase_prefill_wave_tokens', 8192),
            ('phase_max_wait_ms', 1000.0), ('split_capture_dma', True),
            ('native_prefill_commit', True), ('async_checkpoint_capture', True),
            ('restore_stage_timing', False), ('native_restore_frame', False),
            ('host_stage_timing', False), ('native_checkpoint_capture', False),
            ('checkpoint_pinned_prewarm_count', 0),
            ('native_action_metadata', False), ('resident_restore_slots', 1),
            ('session_gpu_checkpoint_bytes', 0),
            ('restore_prepare_split', False),
            ('restore_handoff_fastpath', False),
            ('async_action_metadata', False),
            ('rolling_active_window', False), ('early_capture_slot_release', False),
            ('prefill_publication_pipeline', False),
            ('prefill_native_plan_arm', False),
            ('prefill_capture_ahead', False),
            ('capture_fused_pack', False),
            ('prefill_action_before_mtp', False),
            ('prefill_deferred_validation', False),
            ('prefill_deferred_validation_max_bytes', 67108864),
            ('prefill_deferred_validation_scratch_bytes', 2147483648),
            ('prefill_scatter_after_target', False),
            ('cpu_mean_index_simd', False),
            ('assembly_read_prefetch', False),
            ('assembly_fused_layers', False),
            ('prefill_physical_coalescing', False),
            ('prefill_physical_max_chunks', 2),
            ('early_capture_feedback_bytes', 268435456), ('action_profiling', False),
            ('action_profile_interval', 1), ('action_profile_max_events', 256))}
        try:
            memory = KVMemConfig(**payload)
        except TypeError as error:
            raise ValueError(f"Unknown KVMem setting: {error}") from error
        if memory.kv_dtype != config.cache_config.cache_dtype:
            raise ValueError("KVMem source codec differs from the engine KV cache dtype")
        return cls(memory, cache_bytes, packed, verify, direct, cadence, retirement, gc, preallocate, nvme_bytes,
                   io_bytes, session_capacity, session_bytes, capture_decode, history_bytes, capture_terminal,
                   gather_workers, async_seal, raw_cache, reuse_v,cpu_scatter,scatter_min,batched_append,planning,scorer,checkpoint_policy,
                   query_history,query_decay,guided_tokens,archive_backend,
                   cpu_direct_record_seal=direct_record_seal,cpu_seal_hash_workers=seal_hash_workers,
                   cpu_seal_pack_workers=seal_pack_workers,
                   cpu_seal_serial_allocate=seal_serial_allocate,
                   historical_gc=historical_gc,**action,**graph)
