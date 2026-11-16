"""Keep DFlash query graph policy independent of target attention limitations.

Only the qualified compiled DFlash2 k=1 path installs this instance adapter.
Native DFlash still checks its own backend capability and owns all capture,
metadata, padding and replay. Target PIECEWISE does not imply draft PIECEWISE.
"""
from types import MethodType


def install_independent_draft_graphs(runner):
    from vllm.config import CUDAGraphMode
    from .draft_contract import verify_draft_sources
    verify_draft_sources()
    speculator = runner.speculator
    if getattr(speculator, "_kvmem_independent_graphs", False):
        raise ValueError("Independent DFlash graphs already installed")
    original = speculator.init_cudagraph_manager

    def init(self, target_mode):
        # The launcher requested FULL_AND_PIECEWISE. FlashInfer's FP8 target
        # may resolve to PIECEWISE while the BF16 draft supports full graphs.
        # NONE remains explicit NONE, and native support checks remain intact.
        draft_mode = (CUDAGraphMode.FULL_DECODE_ONLY
                      if target_mode == CUDAGraphMode.PIECEWISE else target_mode)
        original(draft_mode)
        manager = self.query_cudagraph_manager
        print("KVMem DFlash graph policy: target=" + str(target_mode)
              + ", draft=" + str(manager.cudagraph_mode)
              + ", backend=" + str(self.attn_cg_support.min_cg_attn_backend),
              flush=True)

    speculator.init_cudagraph_manager = MethodType(init, speculator)
    speculator._kvmem_independent_graphs = True
