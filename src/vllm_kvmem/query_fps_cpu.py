"""Bounded CPU FPS with original-Key publication and exact tie repair.

The native pool owns its threads. Neither this module nor the pool changes
PyTorch/OpenMP's process-wide settings or initializes CUDA.
"""
import torch

from . import _native


def select_native_cpu_representatives(keys, valid_counts, pool, *, cache_dtype=None):
    from .query_key_fps import REPRESENTATIVES, select_key_representatives

    if keys.device.type != 'cpu' or keys.dtype != torch.float8_e4m3fn or keys.ndim != 5:
        raise ValueError('Native CPU FPS requires original CPU e4m3 Keys')
    if (min(keys.shape) <= 0 or keys.shape[1] > 32 or keys.shape[2] > 32 or
            keys.shape[-1] > 4096 or keys.numel() > 64*1024**2):
        raise ValueError('Native CPU FPS exceeds bounded input geometry')
    counts = torch.as_tensor(valid_counts, device='cpu')
    if (counts.shape != (keys.shape[1],) or counts.dtype not in
            (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError('Invalid native CPU FPS counts')
    groups = keys.shape[0]*keys.shape[1]*keys.shape[3]
    workers = pool.stats()['workers']
    stats = dict(fps_gpu_groups=0, fps_cpu_groups=groups,
        fps_cpu_native_groups=0, fps_cpu_reference_groups=0,
        fps_cpu_ambiguity_groups=0, fps_cpu_pool_busy_groups=0,
        fps_cpu_unsupported_fallback_groups=0, fps_cpu_fallback_groups=0,
        fps_cpu_backend='native', fps_cpu_workers=workers,
        fps_cpu_input_copy_bytes=0,
        fps_device_cache_conversion=False, fps_cpu_exact_cache_conversion=False,
        fps_reduction_guard='4 * dim * FP32 epsilon relative top-two gap')
    try:
        raw_bytes = keys.detach().view(torch.uint8)
        extent, dense = 1, True
        for stride, size in sorted((stride, size) for size, stride in
                zip(raw_bytes.shape, raw_bytes.stride()) if size > 1):
            dense &= stride == extent
            extent *= size
        if not dense:
            # Live archive slabs already use dense L/B permutations. A valid
            # sliced caller is normalized within the same 64 MiB input bound.
            raw_bytes = raw_bytes.contiguous()
            stats['fps_cpu_input_copy_bytes'] = raw_bytes.numel()
        offsets, ambiguous = pool.select(raw_bytes.numpy(), counts.tolist())
    except _native.KeyFPSCPUBusyError:
        # A bounded optional accelerator may decline admission. The original
        # CPU construction remains exact and this event is explicitly audited.
        representatives, indices = select_key_representatives(keys, counts)
        stats.update(fps_cpu_reference_groups=groups, fps_cpu_pool_busy_groups=groups,
            fps_cpu_fallback_groups=groups, fps_fallback_reason='native CPU pool busy')
        return representatives, indices, stats

    indices = torch.from_numpy(offsets).long()
    uncertain = torch.from_numpy(ambiguous)
    expected = (keys.shape[0], keys.shape[1], REPRESENTATIVES, keys.shape[3])
    if indices.shape != expected or uncertain.shape != (keys.shape[0], keys.shape[1], keys.shape[3]):
        raise ValueError('Native CPU FPS returned invalid geometry')
    triples = uncertain.nonzero()
    if len(triples):
        layer, block, head = triples.unbind(1)
        subset = keys.view(torch.uint8)[layer, block, :, head, :].view(keys.dtype)
        _, repaired = select_key_representatives(subset.unsqueeze(0).unsqueeze(3), counts[block])
        indices[layer, block, :, head] = repaired[0, :, :, 0]
    # Gather original bytes only after repairing all uncertain groups. No
    # native arithmetic result is used as a representative Key value.
    selected = keys.view(torch.uint8).gather(2,
        indices.clamp_min(0)[..., None].expand(*indices.shape, keys.shape[-1]))
    selected.masked_fill_(indices[..., None] < 0, 0)
    representatives = selected.view(keys.dtype)
    if cache_dtype is not None:
        # The native pool rejects all nonfinite visible e4m3 inputs; every
        # finite value embeds exactly in these already-qualified cache dtypes.
        if cache_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Invalid exact CPU FP8 cache dtype')
        representatives = representatives.to(cache_dtype)
        stats['fps_cpu_exact_cache_conversion'] = True
    stats.update(fps_cpu_native_groups=groups, fps_cpu_reference_groups=len(triples),
        fps_cpu_ambiguity_groups=len(triples), fps_cpu_fallback_groups=len(triples),
        fps_fallback_reason='near-tie CPU reduction guard' if len(triples) else None)
    return representatives, indices, stats
