"""Narrow initial contract for a native DFlash2 sliding-window draft.

The target uses dense selected positions; the draft retains its own native
logical positions and sliding pages. Only final target hidden states reach the
native proposer. This contract does not itself enable bounded speculation.
"""
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def verify_draft_sources():
    from .engine_api import require_contract
    require_contract('draft')


@dataclass(frozen=True)
class DraftContract:
    query_tokens: int
    sliding_window: int
    max_position: int
    layers: int
    method: str = 'dflash'

    @classmethod
    def from_vllm(cls, config, memory):
        import torch
        spec = config.speculative_config
        if spec is not None and spec.method == 'mtp':
            verify_mtp_sources()
            text = spec.draft_model_config.hf_text_config
            layers = getattr(text, 'mtp_num_hidden_layers', 1)
            query = spec.num_speculative_tokens + 1
            if (getattr(text, 'model_type', None) != 'qwen3_5_text' or layers != 1 or
                    not 1 < query <= 8 or spec.kv_cache_dtype is not None or
                    config.cache_config.cache_dtype != 'fp8_e4m3' or
                    config.max_concurrent_batches != 1 or config.scheduler_config.async_scheduling or
                    config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != 'none' or
                    config.num_lookahead_tokens != query - 1):
                raise ValueError('Unqualified Qwen MTP bounded transaction or FP8 cache')
            return cls(query, 0, config.model_config.max_model_len, layers, 'mtp')
        verify_draft_sources()
        if spec is None or spec.method != 'dflash':
            raise ValueError('Bounded speculation initially requires DFlash2')
        draft = spec.draft_model_config
        hf = draft.hf_config
        block = hf.dflash_config.get('block_size')
        query = spec.num_speculative_tokens + 1
        window = getattr(hf, 'sliding_window', 0)
        layers = hf.num_hidden_layers
        if (hf.architectures != ['DFlash2DraftModel'] or
                type(block) is not int or not 1 < query <= block or query > memory.block_tokens // 2 or
                type(window) is not int or window <= 0 or
                hf.layer_types != ['sliding_attention'] * layers or layers <= 0 or
                not hf.use_sliding_window or draft.dtype != torch.bfloat16):
            raise ValueError('Unqualified DFlash2 query, layer or KV geometry')
        if memory.recent_blocks * memory.block_tokens < window - 1:
            raise ValueError('Target mandatory recent band must contain the draft context window')
        # Persistent logical histories beyond the draft RoPE table require an
        # explicit frame-renormalization adapter; never silently clamp positions.
        if config.model_config.max_model_len > hf.max_position_embeddings:
            raise ValueError('Logical context exceeds the qualified draft position table')
        if config.max_concurrent_batches != 1 or config.scheduler_config.async_scheduling:
            raise ValueError('Initial bounded DFlash2 transactions require synchronous scheduling')
        if config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != 'none':
            raise ValueError('Initial bounded DFlash2 transactions require APC-off GDN none state')
        if config.num_lookahead_tokens != query:
            raise ValueError('Native draft lookahead differs from the qualified query block')
        return cls(query, window, hf.max_position_embeddings, layers)

    def validate_native_groups(self, runner):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
        proposer = runner.speculator
        if self.method == 'mtp':
            names = set(proposer.draft_attn_layer_names)
            groups = tuple(i for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                           if names.intersection(group.layer_names))
            covered = set()
            for i in groups:
                group = runner.kv_cache_config.kv_cache_groups[i]
                spec = group.kv_cache_spec
                specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
                if any(type(s) is not FullAttentionSpec or s.dtype not in (torch.float8_e4m3fn, torch.uint8) or
                       s.sliding_window is not None for s in specs):
                    raise ValueError('MTP requires lockstep full FP8 attention groups')
                covered.update(names.intersection(group.layer_names))
            if len(covered) != self.layers or covered != names or proposer.num_speculative_steps != self.query_tokens - 1:
                raise ValueError('MTP layer/query differs from the bounded contract')
            return groups
        groups = tuple(proposer.draft_kv_cache_group_ids)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError('Missing or duplicate native draft KV groups')
        layers = 0
        for group_id in groups:
            group = runner.kv_cache_config.kv_cache_groups[group_id]
            spec = group.kv_cache_spec
            specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
            if any(type(item) is not SlidingWindowSpec or item.sliding_window != self.sliding_window or
                   item.dtype != torch.bfloat16 for item in specs):
                raise ValueError('Draft cache must retain native BF16 sliding-window pages')
            layers += len(group.layer_names)
        if layers != self.layers or proposer.num_query_per_req != self.query_tokens:
            raise ValueError('Native draft groups/query differ from the model contract')
        return groups


@lru_cache(maxsize=1)
def verify_mtp_sources():
    from .engine_api import require_contract
    require_contract('mtp')
