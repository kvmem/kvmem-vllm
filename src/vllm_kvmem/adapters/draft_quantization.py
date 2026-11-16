"""Use native quantized projections for DFlash's context KV.

The upstream fused BF16 path slices raw qkv weights and calls F.linear, which
does not represent FP8 or packed integer weights. Each native projection applies
its own codec; the target and draft KV layout remain unchanged. Packed DFlash2
also needs its context buffers prepared without reading a nonexistent .weight.
"""
from contextlib import contextmanager
from types import MethodType

import torch


def _context_quantization(runner):
    config = runner.speculative_config
    # MTP inherits the target quantization. Its native projections must not
    # enter the DFlash-only context patch or DFlash source/geometry checks.
    if config is None or config.method != 'dflash':
        return None
    draft = getattr(config, 'draft_model_config', None)
    return getattr(config, 'quantization', None) or getattr(draft, 'quantization', None)


def _project_context(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
    if len(self.layers) != num_layers:
        raise ValueError('Draft context projection differs from its native layer geometry')
    normalized = self.hidden_norm(context_states)
    keys, values = [], []
    for layer in self.layers:
        attn = layer.self_attn
        result = attn.qkv_proj(normalized)
        projected = result[0] if isinstance(result, tuple) else result
        key = projected[:, attn.q_size:attn.q_size + attn.kv_size]
        value = projected[:, attn.q_size + attn.kv_size:]
        keys.append(key.reshape(num_ctx, num_kv_heads, head_dim))
        values.append(value.reshape(num_ctx, num_kv_heads, head_dim))
    return torch.stack(keys), torch.stack(values)


def _packed_context_buffers(self, layers_attn, has_bias):
    # Native loading still sets all RoPE, cache and layer metadata. Only the
    # BF16-only fused weight buffer is replaced by codec-aware projections.
    self._hidden_norm_weight = self.hidden_norm.weight.data
    self._k_norm_weights = torch.stack([a.k_norm.weight.data for a in layers_attn]).contiguous()
    self._project_context_kv = MethodType(_project_context, self)


def _native_dflash2_type():
    from .engine_api import check_source, engine_root
    source = (engine_root()/'model_executor/models/qwen3_dflash.py').read_text()
    check_source(source, 'DFlashQwen3Model._build_context_kv_buffers', ('layers_attn', 'has_bias'))
    check_source(source, 'DFlashQwen3Model._project_context_kv',
                 ('context_states', 'num_ctx', 'num_layers', 'num_kv_heads', 'head_dim'))
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3Model
    return DFlash2Qwen3Model


@contextmanager
def prepare_quantized_context(runner):
    """Scope the packed-weight loader adapter to this DFlash2 model load."""
    if _context_quantization(runner) != 'compressed-tensors':
        yield
        return
    from .draft_contract import verify_draft_sources
    verify_draft_sources()
    model_type = _native_dflash2_type()
    name = '_build_context_kv_buffers'
    owned = name in model_type.__dict__
    original = getattr(model_type, name)
    setattr(model_type, name, _packed_context_buffers)
    try:
        yield
    finally:
        if owned:
            setattr(model_type, name, original)
        else:
            delattr(model_type, name)


def install_quantized_context(runner):
    if _context_quantization(runner) not in ('fp8', 'compressed-tensors'):
        return
    from .draft_contract import verify_draft_sources
    verify_draft_sources()
    model = runner.speculator.model.model
    if not model.layers:
        raise ValueError('Quantized context projection requires the DFlash2 model')
    model._project_context_kv = MethodType(_project_context, model)
