"""Loader and host reference for the QSA streaming residency kernels (csrc/qsa_residency.cu).

The kernels are JIT-built like the MeanK kernel (core/cuda_scoring.py): from the
packaged ``cuda_sources`` copy, or ``csrc`` in a source tree. Callers build them
at load time, before any CUDA-graph capture.

``reference_plan`` repeats the plan kernel on the host, in the same order, for
tests: misses and victims in the kernel's thread-strided order (bitmap word w
and hot slot j belong to thread w % 512 and j % 512, each visiting its share in
ascending order), victims by category: free, clear reference bit, marked.
"""
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from ..core import settings as _settings

THREADS = 512
GROUP_TOKENS = 4
SOURCE = 'qsa_residency.cu'


def _source():
    root = Path(__file__).resolve().parents[1]
    for candidate in (root / 'cuda_sources' / SOURCE, root.parents[1] / 'csrc' / SOURCE):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'{SOURCE} is neither packaged nor in a source tree')


@lru_cache(maxsize=1)
def kernels():
    import hashlib
    if shutil.which('ninja') is None:
        import ninja
        os.environ['PATH'] = str(ninja.BIN_DIR) + os.pathsep + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    source = _source()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    directory = Path(_settings.get('KVMEM_CUDA_BUILD_DIR', tempfile.gettempdir())).resolve() / (
        'kvmem_qsa_residency_' + digest)
    directory.mkdir(parents=True, exist_ok=True)
    stable = directory / SOURCE
    if not stable.exists():
        shutil.copyfile(source, stable)
    return load(name='kvmem_qsa_residency_' + digest, sources=[str(stable)], build_directory=str(directory),
                extra_cuda_cflags=['-O3'], verbose=False)


class ResidencyHook:
    """adapters/qsa_split.py hook for streaming decode, one state set per QSA layer.

    For every layer run it (1) copies the K/V this step wrote into the ring back
    to the pinned pool, (2) plans residency and rewrites the selection to
    hot-buffer positions, (3) copies the missing groups in, then lets the vendor
    attend with query positions that mask nothing (selection already emits only
    causal positions). Requests whose state slot is not streaming are untouched.

    ``allocate(shape, dtype)`` provides device buffers; serving passes one that
    carves them from the KV pool.
    """

    SPOOF = 1 << 62

    def __init__(self, layers, *, slots, max_rows, max_frame_tokens, hot_pages, page_tokens, token_bytes, pool_base,
                 pool_layer_bytes, pool_plane_bytes, allocate, copy_blocks=64):
        import torch
        self.kernels = kernels()
        self.slots, self.page_tokens, self.token_bytes = slots, page_tokens, token_bytes
        self.hot_pages = hot_pages
        self.hot_tokens = hot_pages * page_tokens
        self.hot_groups = self.hot_tokens // GROUP_TOKENS
        self.max_groups = -(-max_frame_tokens // GROUP_TOKENS)
        self.pool_base = pool_base                      # [slots] int64 device pointers (UVA)
        self.pool_layer_bytes, self.pool_plane_bytes = pool_layer_bytes, pool_plane_bytes
        self.copy_blocks = copy_blocks
        self.streaming = allocate((slots,), torch.uint8).zero_()
        self.stats = allocate((4,), torch.int64).zero_()
        self.layers = {}
        for index, (name, layer) in enumerate(layers.items()):
            state = dict(
                index=index, layer=layer,
                group_slot=allocate((slots, self.max_groups), torch.int32).fill_(-1),
                slot_group=allocate((slots, self.hot_groups), torch.int32).fill_(-1),
                referenced=allocate((slots, self.hot_groups), torch.uint8).zero_(),
                hot_page_ids=allocate((slots, hot_pages + 1), torch.int32).zero_(),
                copy_list=allocate((slots * self.hot_groups, 3), torch.int32),
                copy_count=allocate((1,), torch.int32).zero_())
            self.layers[layer._selected_positions.data_ptr()] = state
        shared = self.kernels.plan_shared_bytes(self.max_groups, self.hot_groups)
        if shared > 48 * 1024:
            raise ValueError('QSA residency maps exceed the plan kernel shared memory')
        # Built before any graph capture; replays only slice it.
        self._spoof = allocate((max_rows,), torch.int64).fill_(self.SPOOF)

    def reset_slot(self, slot):
        """Forget a slot's residency (refresh, preemption, free); the ring is reseeded by the caller."""
        for state in self.layers.values():
            state['group_slot'][slot].fill_(-1)
            state['slot_group'][slot].fill_(-1)
            state['referenced'][slot].zero_()

    def __call__(self, binding, inputs, query_positions):
        import torch
        state = self.layers.get(binding.selected_positions.data_ptr())
        if state is None:
            return None
        rows = int(inputs['query'].shape[0])
        cache = state['layer'].kv_cache
        main = cache.view(torch.uint8) if cache.dtype != torch.uint8 else cache
        page_stride = int(cache.stride(0)) * cache.element_size()
        plane_stride = self.page_tokens * self.token_bytes
        layer_offset = state['index'] * self.pool_layer_bytes
        slots = binding.raw_state_slot_ids
        self.kernels.residency_writeback(
            inputs['request_ids'][:rows], inputs['query_positions'][:rows], rows, slots, self.streaming,
            self.pool_base, layer_offset, self.pool_plane_bytes, main, page_stride, plane_stride,
            state['hot_page_ids'], self.hot_pages, self.page_tokens, self.token_bytes, self.copy_blocks)
        state['copy_count'].zero_()
        selected = binding.selected_positions[:rows]
        requests = int(inputs['query_start_loc'].shape[0]) - 1
        self.kernels.residency_plan(
            selected, int(selected.shape[1]), inputs['query_start_loc'], inputs['query_positions'], slots,
            self.streaming, state['group_slot'], state['slot_group'], state['referenced'], state['copy_list'],
            state['copy_count'], requests, self.hot_tokens, self.page_tokens, self.stats)
        self.kernels.residency_copy(
            state['copy_list'], state['copy_count'], self.pool_base, layer_offset, self.pool_plane_bytes, main,
            page_stride, plane_stride, state['hot_page_ids'], self.page_tokens, self.token_bytes,
            self.copy_blocks)
        if rows > self._spoof.shape[0]:
            raise ValueError('QSA residency rows exceed the planned capacity')
        return self._spoof[:rows]


def _victim_order(hot_groups):
    return sorted(range(hot_groups), key=lambda j: (j % THREADS, j // THREADS))


def reference_plan(rows, ring_start, group_slot, slot_group, referenced, hot_tokens, ring_tokens, limits=None):
    """One request's plan on the host. Mutates the maps; returns (rewritten rows, misses, victims).

    rows: list of lists of selected positions (-1 = none); group_slot: dict g -> j;
    slot_group: list j -> g or -1; referenced: list of 0/1; limits: each row's query
    position (entries after it become -1: attention runs unmasked afterwards).
    """
    if limits is None:
        limits = [float('inf')] * len(rows)
    rows = [[p if p <= limit else (-1 if p >= 0 else p) for p in row] for row, limit in zip(rows, limits)]
    hot = len(slot_group)
    # Bitmap words go to thread word % 512, which walks its words in ascending order.
    needed = sorted({p // GROUP_TOKENS for row in rows for p in row if 0 <= p < ring_start},
                    key=lambda g: ((g >> 5) % THREADS, (g >> 5) // THREADS, g & 31))
    pinned = set()
    misses = []
    for g in needed:
        j = group_slot.get(g, -1)
        if j >= 0:
            pinned.add(j)
            referenced[j] = 1
        else:
            misses.append(g)
    order = _victim_order(hot)
    categories = [[], [], []]
    for j in order:
        if j in pinned:
            continue
        categories[0 if slot_group[j] < 0 else (2 if referenced[j] else 1)].append(j)
    victims = (categories[0] + categories[1] + categories[2])[:len(misses)]
    if len(victims) < len(misses):
        raise OverflowError('hot buffer cannot hold the selection')
    if len(misses) > len(categories[0]) + len(categories[1]):
        for j in range(hot):
            if j not in pinned:
                referenced[j] = 0
    for g, j in zip(misses, victims):
        old = slot_group[j]
        if old >= 0:
            group_slot.pop(old, None)
        group_slot[g] = j
        slot_group[j] = g
        referenced[j] = 1
    out = []
    for row in rows:
        new = []
        for p in row:
            if p < 0:
                new.append(p)
            elif p >= ring_start:
                new.append(hot_tokens + p % ring_tokens)
            else:
                new.append(group_slot[p // GROUP_TOKENS] * GROUP_TOKENS + p % GROUP_TOKENS)
        out.append(new)
    return out, misses, victims
