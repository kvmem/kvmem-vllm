"""Validate native DFlash2 TP shards before compiling or allocating KV."""
import json

def validate_draft_parallel(runner):
    cfg = runner.vllm_config
    spec = runner.speculative_config
    size = cfg.parallel_config.tensor_parallel_size
    if spec is None or spec.method != 'dflash' or size == 1:
        return
    from vllm.distributed import get_tp_group
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3Model
    group = get_tp_group()
    model = runner.speculator.model.model
    if size != 2 or group.world_size != size or not isinstance(model, DFlash2Qwen3Model):
        raise ValueError('Only native DFlash2 TP=2 shards are qualified')
    config = model.config
    rows = []
    for layer in model.layers:
        attn = layer.self_attn
        if (attn.num_heads * size != config.num_attention_heads or
                attn.num_kv_heads * size != config.num_key_value_heads or
                attn.q_size != attn.num_heads * attn.head_dim or
                attn.kv_size != attn.num_kv_heads * attn.head_dim):
            raise ValueError('DFlash2 TP requires even disjoint Q/KV shards')
        rows.append(dict(q_heads=attn.num_heads, kv_heads=attn.num_kv_heads,
                         head_dim=attn.head_dim))
    if model._num_kv_heads != rows[0]['kv_heads']:
        raise ValueError('DFlash2 context projection must use the rank-local KV layout')
    target = runner.model
    language = target.get_language_model() if hasattr(target, 'get_language_model') else target
    embed = model.embed_tokens
    if (embed is not language.model.embed_tokens or embed.tp_size != size):
        raise ValueError('DFlash2 must share the native TP-sharded target embedding')
    evidence = dict(tp_rank=group.rank_in_group, tp_size=size, layers=rows,
                    shared_embedding=True, embedding_shape=list(embed.weight.shape))
    # Engine startup logs retain this before the per-rank capture directory exists.
    print('KVMem DFlash2 TP geometry '+json.dumps(evidence), flush=True)
