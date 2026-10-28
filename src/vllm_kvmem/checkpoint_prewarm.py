"""Warm the existing pinned allocator before admitting public requests.

No checkpoint bytes are populated and no custom buffer reuse is introduced.
Normal snapshots still own independent tensors and their original DMA fences.
The allocator retains freed blocks for its normal, event-aware reuse path.
"""
import json
import time


def prewarm_count(payload_bytes, requested, max_sequences, capacity_bytes):
    if any(type(v) is not int or v < 0 for v in
           (payload_bytes, requested, max_sequences, capacity_bytes)):
        raise ValueError('Pinned prewarm geometry must be nonnegative integers')
    if not payload_bytes:
        return 0, 0
    # The pinned caching allocator rounds up to power-of-two blocks. Charge
    # the rounded reservation, not just the tensor's logical byte length.
    charged = 1 << (payload_bytes-1).bit_length()
    return min(requested, max_sequences+2, capacity_bytes//charged), charged


def prewarm_checkpoint_payloads(runner, options):
    requested = options.checkpoint_pinned_prewarm_count
    if not requested:
        return None
    if not options.session_capacity:
        raise ValueError('Pinned checkpoint prewarm needs prompt sessions')
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
    count, charged = prewarm_count(size, requested,
        runner.vllm_config.scheduler_config.max_num_seqs, options.session_checkpoint_bytes)
    started = time.perf_counter()
    buffers = [torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=True) for _ in range(count)]
    report = dict(count=count, requested=requested, payload_bytes=size,
        charged_block_bytes=charged, charged_total_bytes=count*charged,
        checkpoint_budget_bytes=options.session_checkpoint_bytes,
        allocation_seconds=time.perf_counter()-started,
        scope='Startup-only allocator warming. No model state captured or reused; normal snapshot lifetime and fences unchanged.')
    del buffers
    output = getattr(runner, '_kvmem_probe_output', None)
    if output is not None:
        (output/'checkpoint-pinned-prewarm.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
