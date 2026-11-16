from types import SimpleNamespace, MethodType
import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as native
from vllm_kvmem.adapters import draft_contract
from vllm_kvmem.adapters.draft_graphs import install_independent_draft_graphs


@pytest.mark.parametrize("target,support,expected", [
    (CUDAGraphMode.PIECEWISE, AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.FULL_DECODE_ONLY),
    (CUDAGraphMode.PIECEWISE, AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, CUDAGraphMode.NONE),
    (CUDAGraphMode.NONE, AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.NONE),
])
def test_independent_policy_preserves_native_backend_checks(monkeypatch,target,support,expected):
    monkeypatch.setattr(draft_contract,"verify_draft_sources",lambda: None)
    seen=[]
    def manager(config,device,mode,*,decode_query_len):
        seen.append((mode,decode_query_len))
        return SimpleNamespace(cudagraph_mode=mode)
    monkeypatch.setattr(native,"DFlashCudaGraphManager",manager)
    draft=SimpleNamespace(vllm_config=object(),device="cuda",num_query_per_req=2,
        attn_cg_support=SimpleNamespace(min_cg_support=support,min_cg_attn_backend="test"),
        _speculator_name="DFlash2")
    draft.init_cudagraph_manager=MethodType(native.DFlashSpeculator.init_cudagraph_manager,draft)
    runner=SimpleNamespace(speculator=draft)
    install_independent_draft_graphs(runner)
    draft.init_cudagraph_manager(target)
    assert seen==[(expected,2)]
    with pytest.raises(ValueError,match="already installed"):
        install_independent_draft_graphs(runner)
