import importlib.metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_kvmem.metadata import WindowLengths, WorkingSetMetadata
from vllm_kvmem.metadata_patch import patch_metadata_source


def test_real_vllm_metadata_separates_working_set_from_recurrent_history():
    from vllm_kvmem.engine_api import engine_root
    path = engine_root() / 'v1/worker/gpu/attn_utils.py'
    source = patch_metadata_source(path.read_bytes())
    module = ModuleType("_kvmem_metadata_qualification")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        class Builder:
            supports_update_block_table = False
            def build(self, common_prefix_len, common_attn_metadata, **kwargs):
                return common_attn_metadata, kwargs
        builder = Builder()
        groups = [[SimpleNamespace(layer_names=[name], get_metadata_builder=lambda _: builder,
                    build_metadata=lambda metadata, ubatch, **kwargs: builder.build(0, metadata, **kwargs))]
                  for name in ("attention", "recurrent")]
        original_lens = torch.tensor([100001, 250001], dtype=torch.int32)
        window_lens = torch.tensor([257, 513], dtype=torch.int32)
        prefill_flags = torch.tensor([False, True])
        class NativeMetadata:
            def get_extra_common_attn_kwargs(self, group, count):
                return {"is_prefilling": prefill_flags}

            def get_extra_attn_kwargs(self, builder, count):
                return {"native_marker": "unchanged"}
        wrapped = WorkingSetMetadata(NativeMetadata(), {
            0: WindowLengths(window_lens, window_lens.clone(), 1024)})
        kwargs = dict(
            attn_groups=groups, num_reqs=2, num_tokens=2,
            query_start_loc_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            max_query_len=1, seq_lens=original_lens, max_seq_len=262144,
            block_tables=[torch.tensor([[1], [2]], dtype=torch.int32)] * 2,
            slot_mappings=torch.tensor([[16, 32], [48, 64]], dtype=torch.int64),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None, None]),
            seq_lens_cpu_upper_bound=original_lens.clone(),
            model_specific_attn_metadata=wrapped,
        )
        result = module.build_attn_metadata(**kwargs)
        attn, attn_extra = result["attention"]
        recurrent, recurrent_extra = result["recurrent"]
        assert torch.equal(attn.seq_lens, window_lens)
        assert torch.equal(recurrent.seq_lens, original_lens)
        assert attn.max_seq_len == 1024 and recurrent.max_seq_len == 262144
        assert torch.equal(attn.is_prefilling, prefill_flags)
        assert attn_extra == recurrent_extra == {"native_marker": "unchanged"}
        from vllm_kvmem.window_batch import attention_windows
        positions = torch.tensor([256, 512])
        kwargs["model_specific_attn_metadata"] = NativeMetadata()
        with attention_windows({0: WindowLengths(window_lens, window_lens.clone(), 1024, positions)}):
            contextual = module.build_attn_metadata(**kwargs)
        assert contextual["attention"][0].positions is positions
        assert contextual["recurrent"][0].positions is None
        assert torch.equal(contextual["attention"][0].seq_lens, window_lens)
        assert torch.equal(contextual["recurrent"][0].seq_lens, original_lens)
        kwargs["model_specific_attn_metadata"] = None
        unmodified = module.build_attn_metadata(**kwargs)
        assert all(torch.equal(meta.seq_lens, original_lens) for meta, _ in unmodified.values())
    finally:
        del sys.modules[module.__name__]
