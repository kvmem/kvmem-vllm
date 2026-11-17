"""Warm the existing pinned allocator before admitting public requests.

No checkpoint bytes are populated and no custom buffer reuse is introduced.
Normal snapshots still own independent tensors and their original DMA fences.
The allocator retains freed blocks for its normal, event-aware reuse path.

Entering torch.cuda.graph() empties the pinned host cache, so the runner warms
after the startup graph capture (at KV cache initialization only when the
engine runs eagerly). A graph captured later at run time drops the blocks again.
"""
import json
import os
import time
from ..core import settings as _settings


def prewarm_count(payload_bytes, requested, max_sequences, capacity_bytes, retained=0):
    """retained: checkpoints that sessions can keep alive at once (0: running requests only)."""
    if any(type(v) is not int or v < 0 for v in
           (payload_bytes, requested, max_sequences, capacity_bytes, retained)):
        raise ValueError('Pinned prewarm geometry must be nonnegative integers')
    if not payload_bytes:
        return 0, 0
    # The pinned caching allocator rounds up to power-of-two blocks. Charge
    # the rounded reservation, not just the tensor's logical byte length.
    charged = 1 << (payload_bytes-1).bit_length()
    return min(requested, max(max_sequences+2, retained), capacity_bytes//charged), charged


def requested_count(options):
    """KVMEM_CHECKPOINT_PINNED_PREWARM overrides checkpoint_pinned_prewarm_count.

    The environment form enables the same startup warming without changing the
    serving profile (and with it the vLLM config hash that keys tuned kernels).
    """
    value = _settings.get('KVMEM_CHECKPOINT_PINNED_PREWARM')
    if value is None:
        return options.checkpoint_pinned_prewarm_count
    if not value.isascii() or not value.isdigit() or int(value) > 64:
        raise ValueError('KVMEM_CHECKPOINT_PINNED_PREWARM must be an integer in [0, 64]')
    return int(value)


def prewarm_checkpoint_payloads(runner, options, stage='kv_cache_init'):
    requested = requested_count(options)
    if not requested or not options.session_capacity:
        # Only prompt-session checkpoints use these pinned blocks. The preset
        # turns warming on for every profile; a session-less profile (Flash
        # Next) captures nothing, so there is nothing to warm.
        return None
    speculation = getattr(runner, 'speculative_config', None)
    if speculation is not None and speculation.method != 'mtp':
        raise ValueError('Pinned checkpoint prewarm only supports target-only or MTP')
    from .native_state import NativeRecurrentState
    import torch
    recurrent = NativeRecurrentState(runner)
    # capture stores the canonical row of every GDN layer, not its temporary
    # speculative rows. Only shape/dtype metadata is read; no GPU data read.
    size = sum(t[0].numel()*t.element_size() for layers in recurrent.groups.values()
               for caches in layers.values() for t in caches)
    # Sessions keep checkpoints after their request ends: one per session,
    # two (prompt and terminal) under the pm policy, plus one being replaced.
    retained = options.session_capacity*(
        2 if getattr(options, 'session_checkpoint_policy', 'sealed') == 'pm' else 1)+1
    count, charged = prewarm_count(size, requested,
        runner.vllm_config.scheduler_config.max_num_seqs, options.session_checkpoint_bytes, retained)
    started = time.perf_counter()
    buffers = [torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=True) for _ in range(count)]
    report = dict(count=count, requested=requested, stage=stage, payload_bytes=size,
        charged_block_bytes=charged, charged_total_bytes=count*charged,
        checkpoint_budget_bytes=options.session_checkpoint_bytes,
        allocation_seconds=time.perf_counter()-started,
        scope='Startup-only allocator warming. No model state captured or reused; normal snapshot lifetime and fences unchanged.')
    del buffers
    output = getattr(runner, '_kvmem_probe_output', None)
    if output is not None:
        (output/'checkpoint-pinned-prewarm.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
