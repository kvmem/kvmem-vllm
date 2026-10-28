"""Small proposed upstream hook: per-group attention lengths in V2 metadata.

The existing model-specific metadata interface supplies the overrides. No model
or scheduler computations change for callers which do not supply these fields.
Enabled only by the explicit KVMEM_WINDOW_METADATA compatibility option.
"""
def patch_metadata_source(source: bytes) -> str:
    text = source.decode()
    old = """            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,"""
    new = """            seq_lens=common_attn_metadata_extra_kwargs.pop("seq_lens", seq_lens),
            seq_lens_cpu_upper_bound=common_attn_metadata_extra_kwargs.pop(
                "seq_lens_cpu_upper_bound", seq_lens_cpu_upper_bound
            ),
            max_seq_len=common_attn_metadata_extra_kwargs.pop("max_seq_len", max_seq_len),"""
    if text.count(old) != 1:
        raise ImportError("Ambiguous per-group metadata patch")
    text = text.replace(old, new)
    old_positions = "            positions=positions,"
    if text.count(old_positions) != 1:
        raise ImportError("Ambiguous per-group position patch")
    text = text.replace(old_positions,
                        '            positions=common_attn_metadata_extra_kwargs.pop("positions", positions),')
    anchor = "    attn_metadata: dict[str, Any] = {}"
    if text.count(anchor) != 1:
        raise ImportError("Ambiguous attention-window context hook")
    return text.replace(anchor, """    from vllm_kvmem.window_batch import contextual_metadata
    model_specific_attn_metadata = contextual_metadata(model_specific_attn_metadata)
""" + anchor)
