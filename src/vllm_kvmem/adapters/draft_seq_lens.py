"""KVMEM_DRAFT_SEQ_LENS_CPU=1: exact CPU sequence lengths for draft steps 2 and later.

vLLM's FlashInfer metadata builder reads ``seq_lens.cpu()`` whenever a batch has
no trtllm-gen decode kernel (GPUs such as the A800 and RTX 4090). The
autoregressive drafter (MTP) rebuilds attention metadata for every draft step,
so each step synchronizes the GPU and the GPU then idles while the CPU builds
and launches the next step.

Between two draft-step builds vLLM changes the lengths in exactly one place,
``_update_draft_inputs_kernel``: each of the first ``num_reqs`` lengths becomes
``min(length + 1, max_model_len)``; padded entries are untouched. After the
first draft step of a proposal has read the real lengths, every later step's
lengths are therefore known on the CPU without a synchronization. The builder
receives exactly the values it would have read, so its plans are unchanged.

Every predicted step is checked: the real lengths are copied to pinned memory
without waiting and compared at the next real read (the following proposal's
first step synchronizes anyway). A difference raises.
"""
import contextvars
import dataclasses

from ..core import settings as _settings

_MODE = contextvars.ContextVar('kvmem_draft_seq_lens', default=None)


def enabled():
    value = _settings.get('KVMEM_DRAFT_SEQ_LENS_CPU', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_DRAFT_SEQ_LENS_CPU must be 0 or 1')
    return value == '1'


def _known_seq_lens(tensor, read):
    """``tensor`` (same storage) whose ``.cpu()`` returns ``read()``."""
    import torch

    class KnownCpuSeqLens(torch.Tensor):
        # Operations on it return plain tensors; only .cpu() is replaced.
        __torch_function__ = torch._C._disabled_torch_function_impl

        def cpu(self, *args, **kwargs):
            return read()

    return torch.Tensor._make_subclass(KnownCpuSeqLens, tensor)


class _Checks:
    """Predicted lengths awaiting comparison with the real ones."""

    def __init__(self):
        self.pending = []
        self.checked = self.predicted_steps = 0

    def add(self, real, predicted):
        import torch
        copy = torch.empty(real.shape, dtype=real.dtype, pin_memory=True)
        copy.copy_(real, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending.append((copy, predicted, event))
        self.predicted_steps += 1

    def verify(self):
        for copy, predicted, event in self.pending:
            event.synchronize()
            self.checked += 1
            if not bool((copy == predicted).all()):
                raise RuntimeError('KVMEM_DRAFT_SEQ_LENS_CPU: predicted draft sequence lengths differ from '
                                   f'the GPU lengths ({predicted.tolist()} vs {copy.tolist()})')
        self.pending.clear()


def install():
    """Patch the autoregressive drafter and the FlashInfer builder (idempotent)."""
    import torch
    from vllm.v1.attention.backends import flashinfer
    from vllm.v1.worker.gpu.spec_decode import speculator as base
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    builder = flashinfer.FlashInferMetadataBuilder
    if getattr(builder, '_kvmem_draft_seq_lens', False):
        return
    original_build = builder.build
    original_uniform = base.DraftModelSpeculator._build_uniform_attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        mode = _MODE.get()
        if mode is None:
            return original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        kind, state = mode
        real = common_attn_metadata.seq_lens
        if kind == 'record':
            def read():
                value = real.cpu()
                if state.get('cpu') is None:
                    state['cpu'] = value.clone()
                return value
        else:
            if tuple(state['cpu'].shape) != tuple(real.shape):
                return original_build(self, common_prefix_len, common_attn_metadata, fast_build)

            def read():
                if not state.get('queued'):
                    state['checks'].add(real, state['cpu'])
                    state['queued'] = True
                return state['cpu'].clone()
        common = dataclasses.replace(common_attn_metadata, seq_lens=_known_seq_lens(real, read))
        return original_build(self, common_prefix_len, common, fast_build)

    def build_uniform(self, batch_desc, num_reqs, num_query_per_req, seq_lens_cpu_upper_bound, step,
                      *args, **kwargs):
        if (not isinstance(self, AutoRegressiveSpeculator) or not self.advance_draft_positions or
                num_query_per_req != 1 or step < 1):
            return original_uniform(self, batch_desc, num_reqs, num_query_per_req, seq_lens_cpu_upper_bound,
                                    step, *args, **kwargs)
        checks = self.__dict__.setdefault('_kvmem_draft_seq_checks', _Checks())
        last = self.__dict__.get('_kvmem_draft_seq_lens')
        if step == 1 or last is None or last['step'] != step - 1 or last['num_reqs'] != num_reqs:
            state = {}
            token = _MODE.set(('record', state))
            try:
                result = original_uniform(self, batch_desc, num_reqs, num_query_per_req,
                                          seq_lens_cpu_upper_bound, step, *args, **kwargs)
            finally:
                _MODE.reset(token)
            # This build read the real lengths: every earlier copy has landed.
            checks.verify()
            self._kvmem_draft_seq_lens = (dict(step=step, num_reqs=num_reqs, cpu=state['cpu'])
                                          if state.get('cpu') is not None else None)
            return result
        predicted = last['cpu'].clone()
        predicted[:num_reqs] = torch.clamp(predicted[:num_reqs] + 1, max=self.max_model_len)
        state = dict(cpu=predicted, checks=checks)
        token = _MODE.set(('use', state))
        try:
            result = original_uniform(self, batch_desc, num_reqs, num_query_per_req,
                                      seq_lens_cpu_upper_bound, step, *args, **kwargs)
        finally:
            _MODE.reset(token)
        self._kvmem_draft_seq_lens = dict(step=step, num_reqs=num_reqs, cpu=predicted)
        return result

    builder.build = build
    base.DraftModelSpeculator._build_uniform_attn_metadata = build_uniform
    builder._kvmem_draft_seq_lens = True
