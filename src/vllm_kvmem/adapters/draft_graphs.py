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


def deferred_submit_enabled():
    """Opt-in KVMEM_DFLASH_SUBMIT_AFTER_DRAFT=1 (compiled DFlash2 k=1): see install_deferred_capture_submit."""
    from ..core import settings as _settings
    value = _settings.get('KVMEM_DFLASH_SUBMIT_AFTER_DRAFT', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_DFLASH_SUBMIT_AFTER_DRAFT must be 0 or 1')
    return value == '1'


def stage_before_draft_enabled():
    """Opt-in KVMEM_DFLASH_STAGE_BEFORE_DRAFT=1 (with KVMEM_DFLASH_SUBMIT_AFTER_DRAFT): score
    capture is staged before the draft and resolved after it, as for MTP drafts."""
    from ..core import settings as _settings
    value = _settings.get('KVMEM_DFLASH_STAGE_BEFORE_DRAFT', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_DFLASH_STAGE_BEFORE_DRAFT must be 0 or 1')
    return value == '1'


def flush_deferred_submit(runner, *, after_draft=False):
    """Run the sampled batch's KVMem capture submission, if one is waiting."""
    pending = getattr(runner, '_kvmem_deferred_submit', None)
    if pending is None:
        return False
    runner._kvmem_deferred_submit = None
    counts = runner.__dict__.setdefault('_kvmem_deferred_submit_counts', {'after_draft': 0, 'fallback': 0})
    counts['after_draft' if after_draft else 'fallback'] += 1
    runner._kvmem_capture_runtime.submit(*pending)
    return True


def install_deferred_capture_submit(runner):
    """Submit a sampled DFlash2 batch's KVMem capture right after the draft is enqueued.

    Without it the submission (publication drain, score capture, native
    validation fence) runs between sampling and the draft launch, so the GPU
    idles while the host works. The submission, its inputs and its order
    relative to every other publication are unchanged; only its position
    relative to the draft's GPU work moves. The draft never reads KVMem
    captures (it consumes the target's hidden states). sample_tokens flushes a
    submission whose step ran no proposal.
    """
    speculator = runner.speculator
    if getattr(speculator, '_kvmem_deferred_submit', False):
        raise ValueError('Deferred DFlash2 capture submission already installed')
    original = speculator.propose
    stage = stage_before_draft_enabled()

    def propose(self, *args, **kwargs):
        pending = getattr(runner, '_kvmem_deferred_submit', None)
        runtime = getattr(runner, '_kvmem_capture_runtime', None)
        action = getattr(runtime, 'action', None)
        staged = False
        try:
            if stage and pending is not None and action is not None:
                # Opt-in KVMEM_DFLASH_STAGE_BEFORE_DRAFT: the MTP decode staging
                # path (score capture plus a validation ticket recorded before
                # the draft's kernels); submit() below resolves it after them.
                staged = action.stage_before_mtp(pending[0], pending[1])
            result = original(*args, **kwargs)
            flush_deferred_submit(runner, after_draft=True)
            return result
        except BaseException as error:
            if staged or getattr(action, '_mtp_capture_frame', None) is not None:
                # As the MTP hook: a staged frame may own an unconfirmed packet.
                runtime.quarantine_publication(error)
                try:
                    action.cancel_before_mtp()
                except BaseException as cleanup_error:
                    runtime.quarantine_publication(cleanup_error)
                    raise
                runner._kvmem_deferred_submit = None
            raise

    speculator.propose = MethodType(propose, speculator)
    speculator._kvmem_deferred_submit = True
    runner._kvmem_defer_dflash_submit = True
