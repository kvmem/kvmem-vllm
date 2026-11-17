"""GPU checks for QSA streaming pieces against the real B12X kernels (explicit opt-in)."""
import os
from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='explicit GPU opt-in')

PAGE, HEADS, DIM, N = 32, 2, 256, 64
STRIDE = PAGE * HEADS * DIM * 2 + PAGE // 4 * 128 * 2   # one padded native page


def _rebuild(monkeypatch, *, carved, pages=6, main_pages=(3, 4)):
    """Run QSAWindowAssembler over two nonadjacent records; return (backing, cache, selector view, table)."""
    import torch
    from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSAImpl
    from vllm.models.qwen4_exp.common.b12x_qsa_cache import qsa_compressed_cache_view
    from vllm_kvmem.adapters.qsa_adapter import finish_qsa_refresh
    from vllm_kvmem.adapters.qsa_streaming import SelectorState
    from vllm_kvmem.core.memory import Phase
    from vllm_kvmem.serving.qsa_assembly import QSAWindowAssembler
    torch.manual_seed(113)
    device = 'cuda'
    backing = torch.zeros(pages * STRIDE, device=device, dtype=torch.uint8)
    cache = backing.as_strided((pages, 2, PAGE, HEADS * DIM), (STRIDE, PAGE * HEADS * DIM, HEADS * DIM, 1))
    impl = Qwen4ExpQSAImpl.__new__(Qwen4ExpQSAImpl)
    impl.num_kv_heads, impl.head_size, impl.kv_cache_dtype = HEADS, DIM, 'fp8_e4m3'
    angles = torch.arange(512, device=device)[:, None] * (10000 ** (-torch.arange(32, device=device).float() / 32))
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1)
    norm = torch.randn(128, device=device, dtype=torch.bfloat16) * .1
    caps = NS(index_head_dim=128, index_rotary_dim=64, compress_ratio=4, raw_ring_capacity=8,
              compressed_page_size=PAGE // 4, position_axes=1, mrope_interleaved=False, rms_norm_eps=1e-6)
    layer = NS(device=torch.device(device), num_kv_heads=HEADS, head_dim=DIM, layer_name='main', kv_cache=cache,
               impl=impl, rotary_emb=NS(cos_sin_cache=trig), compress_ratio=4, index_head_dim=128,
               raw_ring_capacity=8, position_axes=1, _qsa_decode_context=NS(caps=caps),
               indexer=NS(k_layernorm=NS(weight=norm)), _k_scale=torch.ones((), device=device),
               _raw_k_ring=torch.zeros((2, 8, 128), device=device, dtype=torch.bfloat16),
               _raw_logical_positions=torch.full((2, 8), -1, device=device, dtype=torch.int64),
               _raw_rope_positions=torch.full((2, 8, 1), -1, device=device, dtype=torch.int64),
               _raw_interval_start_positions=torch.zeros((2,), device=device, dtype=torch.int64),
               set_skip_topk=lambda value: None, max_seqs=2, max_seq_len=6 * PAGE)
    runner = NS(_kvmem_qsa_layers={'main': layer}, _kvmem_qsa_pending_rings=[],
                req_states=NS(req_id_to_index={'r': 1}),
                kv_cache_config=NS(kv_cache_groups=[NS(layer_names=['main'])]),
                model_state=NS(qsa_committed_num_accepted_tokens_gpu=torch.tensor([1, 3], device=device)))
    if carved:
        state = SelectorState(layers=1, frame=N, slots=2)
        layer._compressed_cache = state.carved_view(0, layer, cache, 4, 128)
        runner._kvmem_qsa_selector = state
    else:
        layer._compressed_cache = qsa_compressed_cache_view(cache, compress_ratio=4, index_head_dim=128)
    table = torch.tensor(list(main_pages), device=device, dtype=torch.int32)
    monkeypatch.setattr('vllm_kvmem.serving.qsa_assembly.kernel_table', lambda *args: table)
    monkeypatch.setattr('vllm.forward_context.get_forward_context', lambda: NS(additional_kwargs={}))
    keys = torch.randn(N, HEADS, DIM).to(torch.float8_e4m3fn)
    values = torch.randn(N, HEADS, DIM).to(torch.float8_e4m3fn)
    selector = torch.randn(N, 128, dtype=torch.bfloat16)
    raw_archive = torch.zeros(2, N, HEADS, DIM, dtype=torch.float8_e4m3fn)
    value_archive = torch.zeros(2, N, HEADS, DIM, dtype=torch.uint8)
    raw_archive[0].copy_(keys)
    value_archive[0].copy_(values.view(torch.uint8))
    value_archive[1].reshape(N, -1)[:, :256].copy_(selector.view(torch.uint8))
    records = {7: NS(handle=NS(slot=7), count=32), 13: NS(handle=NS(slot=13), count=32)}
    remaps = [NS(block_id=13, to_base=32, n_tokens=32), NS(block_id=7, to_base=0, n_tokens=32)]
    plan = NS(remaps=remaps, total_window_tokens=N)

    def read(handles):
        return [(raw_archive[:, :32] if h.slot == 7 else raw_archive[:, 32:],
                 value_archive[:, :32] if h.slot == 7 else value_archive[:, 32:]) for h in handles]

    authority = NS(shape=(2, 32, HEADS, DIM), records=records, read_records=read,
                   memory=NS(phase=Phase.ASSEMBLING, _plan=plan))
    assembler = QSAWindowAssembler(runner, 'r', ['main', '__qsa_index__.main'], 32)
    assembler.assemble(authority, plan).event.synchronize()
    finish_qsa_refresh(runner)
    torch.cuda.synchronize()
    return backing, cache, layer, table


def test_carved_selector_rebuild_is_bitwise_equal_to_page_tails(monkeypatch):
    import torch
    from vllm.models.qwen4_exp.common.b12x_qsa_cache import qsa_compressed_cache_view
    tail_backing, tail_cache, tail_layer, table = _rebuild(monkeypatch, carved=False)
    carved_backing, carved_cache, carved_layer, _ = _rebuild(monkeypatch, carved=True)
    tail_keys = qsa_compressed_cache_view(tail_cache, compress_ratio=4, index_head_dim=128)[table.long()]
    # Frame N=64 is two 32-token pages, so slot 1 owns compressed pages 2 and 3.
    carved_keys = carved_layer._compressed_cache[2:4]
    assert torch.equal(carved_keys.view(torch.uint8), tail_keys.view(torch.uint8))
    # Main K/V bytes are identical; the carve touched only native block 1, never the null page.
    k_tail, v_tail = tail_layer.impl._kv_cache_views(tail_cache)
    k_carved, v_carved = carved_layer.impl._kv_cache_views(carved_cache)
    assert torch.equal(k_carved[table.long()].view(torch.uint8), k_tail[table.long()].view(torch.uint8))
    assert torch.equal(v_carved[table.long()].view(torch.uint8), v_tail[table.long()].view(torch.uint8))
    assert not carved_backing[:STRIDE].any()
    assert torch.equal(carved_layer._raw_k_ring, tail_layer._raw_k_ring)
    assert torch.equal(carved_layer._raw_interval_start_positions, tail_layer._raw_interval_start_positions)


def test_residency_plan_kernel_matches_host_reference():
    import random
    import torch
    from vllm_kvmem.adapters.qsa_residency import kernels, reference_plan
    ext = kernels()
    rng = random.Random(7)
    device = 'cuda'
    slots, max_groups, hot_groups, width, ring, hot_tokens = 4, 600, 40, 48, 64, 160
    margin = 8          # the oldest ring positions may be aliased by draft writes
    request_slot = torch.tensor([2, 0, 1, 3], dtype=torch.int32, device=device)
    streaming = torch.tensor([1, 1, 1, 0], dtype=torch.uint8, device=device)
    group_slot = torch.full((slots, max_groups), -1, dtype=torch.int32, device=device)
    slot_group = torch.full((slots, hot_groups), -1, dtype=torch.int32, device=device)
    referenced = torch.zeros((slots, hot_groups), dtype=torch.uint8, device=device)
    copy_list = torch.zeros((slots * hot_groups, 3), dtype=torch.int32, device=device)
    copy_count = torch.zeros(1, dtype=torch.int32, device=device)
    stats = torch.zeros(4, dtype=torch.int64, device=device)
    host = {s: ({}, [-1] * hot_groups, [0] * hot_groups) for s in range(slots)}
    for step in range(8):
        newest = 2000 + 4 * step
        rows, positions, starts = [], [], [0]
        for request in range(4):
            for _ in range(4):
                groups = rng.sample(range(0, 120), 9)
                row = [4 * g + rng.randrange(4) for g in groups] + [newest - rng.randrange(60) for _ in range(3)]
                row += [-1] * (width - len(row))
                rows.append(row)
                positions.append(newest - 3 + len(positions) % 4)
            starts.append(len(rows))
        selected = torch.tensor(rows, dtype=torch.int32, device=device)
        query_start = torch.tensor(starts, dtype=torch.int32, device=device)
        query_positions = torch.tensor(positions, dtype=torch.int64, device=device)
        copy_count.zero_()
        ext.residency_plan(selected, width, query_start, query_positions, request_slot, streaming, group_slot,
                           slot_group, referenced, copy_list, copy_count, 4, hot_tokens, ring, margin, stats)
        torch.cuda.synchronize()
        copies = {tuple(e) for e in copy_list[:int(copy_count)].tolist()}
        expected_copies = set()
        for request in range(4):
            slot = int(request_slot[request])
            block = rows[starts[request]:starts[request + 1]]
            got = selected[starts[request]:starts[request + 1]].tolist()
            if not streaming[slot]:
                assert got == block
                continue
            ring_start = positions[starts[request + 1] - 1] - ring + 1 + margin
            limits = positions[starts[request]:starts[request + 1]]
            expected, misses, victims = reference_plan(block, ring_start, *host[slot], hot_tokens, ring, limits)
            assert got == expected, (step, request)
            expected_copies |= {(slot, g, j) for g, j in zip(misses, victims)}
            group_map, owners, refs = host[slot]
            assert slot_group[slot].tolist() == owners
            assert referenced[slot].tolist() == refs
            assert {g: j for g, j in enumerate(group_slot[slot].tolist()) if j >= 0} == group_map
        assert copies == expected_copies
    assert int(stats[3]) == 0                     # no errors


def test_residency_copy_and_writeback_move_exact_bytes():
    import torch
    from vllm_kvmem.adapters.qsa_residency import kernels
    ext = kernels()
    device = 'cuda'
    layers, frame, token_bytes, page_tokens, pages = 2, 256, 512, 32, 8
    slot_bytes = layers * 2 * frame * token_bytes
    pool = torch.randint(0, 256, (3, slot_bytes), dtype=torch.uint8).pin_memory()
    pool_base = torch.tensor([pool[s].data_ptr() for s in range(3)], dtype=torch.int64, device=device)
    plane = page_tokens * token_bytes
    stride = 2 * plane + 2048
    backing = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
    cache = backing.as_strided((pages, 2, page_tokens, token_bytes), (stride, plane, token_bytes, 1))
    hot_pages = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0], [0, 0, 0, 0]], dtype=torch.int32, device=device)
    layer = 1
    layer_offset = layer * 2 * frame * token_bytes
    entries = [(0, 5, 0), (0, 40, 17), (1, 63, 9)]       # (slot, group, hot slot)
    copy_list = torch.tensor(entries, dtype=torch.int32, device=device)
    count = torch.tensor([len(entries)], dtype=torch.int32, device=device)
    ext.residency_copy(copy_list, count, pool_base, layer_offset, frame * token_bytes, backing, stride, plane,
                       hot_pages, page_tokens, token_bytes, 4)
    torch.cuda.synchronize()
    for slot, g, j in entries:
        page = int(hot_pages[slot, (j * 4) // page_tokens])
        for p in (0, 1):
            src = pool[slot, layer_offset + p * frame * token_bytes + g * 4 * token_bytes:][:4 * token_bytes]
            dst = cache[page, p, (j * 4) % page_tokens:(j * 4) % page_tokens + 4].reshape(-1)
            assert torch.equal(dst.cpu(), src)
    # Write-back: rows of request 0 (slot 1) wrote frame positions 70 and 71 into the ring page.
    ring_page = int(hot_pages[1, 3])
    cache[ring_page, :, 70 % page_tokens:72 % page_tokens or None].random_(0, 256)
    row_request = torch.tensor([0, 0], dtype=torch.int32, device=device)
    query_positions = torch.tensor([70, 71], dtype=torch.int64, device=device)
    request_slot = torch.tensor([1], dtype=torch.int32, device=device)
    streaming = torch.tensor([0, 1, 0], dtype=torch.uint8, device=device)
    ext.residency_writeback(row_request, query_positions, 2, request_slot, streaming, pool_base, layer_offset,
                            frame * token_bytes, backing, stride, plane, hot_pages, 3, page_tokens, token_bytes, 2)
    torch.cuda.synchronize()
    for f in (70, 71):
        for p in (0, 1):
            ring_bytes = cache[ring_page, p, f % page_tokens].cpu()
            pool_bytes = pool[1, layer_offset + p * frame * token_bytes + f * token_bytes:][:token_bytes]
            assert torch.equal(ring_bytes, pool_bytes)
    # A two-page ring (columns 2..3 of the row): frame column c writes to ring page c % 2.
    two = torch.tensor([[1, 2, 5, 6]], dtype=torch.int32, device=device)
    for f, page in ((70, 5), (100, 6), (131, 5)):
        cache[page, :, f % page_tokens].random_(0, 256)
    ext.residency_writeback(torch.zeros(3, dtype=torch.int32, device=device),
                            torch.tensor([70, 100, 131], dtype=torch.int64, device=device), 3,
                            torch.tensor([0], dtype=torch.int32, device=device),
                            torch.tensor([1, 0], dtype=torch.uint8, device=device), pool_base, layer_offset,
                            frame * token_bytes, backing, stride, plane, two, 2, page_tokens, token_bytes, 2)
    torch.cuda.synchronize()
    for f, page in ((70, 5), (100, 6), (131, 5)):
        for p in (0, 1):
            assert torch.equal(cache[page, p, f % page_tokens].cpu(),
                               pool[0, layer_offset + p * frame * token_bytes + f * token_bytes:][:token_bytes])


def _mini_runtime(page_tokens=32, token_bytes=512, frame_pages=4, slots=2, layers=2, hot_pages=2, ring_pages=1):
    """StreamingRuntime fields for the copy helpers over a small native pool and a registered host pool."""
    import mmap
    import torch
    from types import SimpleNamespace as NS
    from vllm_kvmem.adapters.qsa_residency import kernels
    from vllm_kvmem.adapters.qsa_stream_runtime import StreamingRuntime
    ext = kernels()
    rt = StreamingRuntime.__new__(StreamingRuntime)
    rt.ext, rt.page_tokens, rt.token_bytes, rt.hot_pages = ext, page_tokens, token_bytes, hot_pages
    rt.ring_pages = ring_pages
    rt.ring_tokens = ring_pages * page_tokens
    rt.plane_stride = page_tokens * token_bytes
    rt.page_stride = 2 * rt.plane_stride + 1024                  # padded native page
    owned = hot_pages + ring_pages
    blocks = 1 + 2 * frame_pages + slots * layers * owned
    backing = torch.zeros(blocks * rt.page_stride, dtype=torch.uint8, device='cuda')
    rt.carve = NS(staging_buffers=2, frame_pages=frame_pages,
                  staging_row=lambda b: list(range(1 + b * frame_pages, 1 + (b + 1) * frame_pages)))
    rt.plane_bytes = frame_pages * page_tokens * token_bytes
    rt.layer_bytes = 2 * rt.plane_bytes
    rt.slot_bytes = layers * rt.layer_bytes
    rt._mapping = mmap.mmap(-1, slots * rt.slot_bytes)
    host = torch.frombuffer(rt._mapping, dtype=torch.uint8)
    rt._pool_host = host
    rt._pool_dev = ext.host_register(host.data_ptr(), host.numel())
    rt.pool_view = ext.device_view(rt._pool_dev, host.numel(), 0)
    first = 1 + 2 * frame_pages
    rt.hosts = {s: [[first + (s * layers + i) * owned + j for j in range(owned)]
                    for i in range(layers)] for s in range(slots)}
    return rt, backing, host


def _page_token(backing, rt, page, plane, token):
    start = page * rt.page_stride + plane * rt.plane_stride + token * rt.token_bytes
    return backing[start:start + rt.token_bytes]


@pytest.mark.parametrize('ring_pages', (1, 2))
def test_host_pool_helpers_and_prefill_staging_fill(ring_pages):
    import torch
    from vllm_kvmem.adapters.qsa_stream_runtime import RING_MARGIN
    rt, backing, host = _mini_runtime(ring_pages=ring_pages)
    try:
        host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8))
        # device_view aliases the host pool.
        assert torch.equal(rt.pool_view[:4096].cpu(), host[:4096])
        slot, index, first, end = 1, 1, 80, 100           # chunk [80, 100) already written to the ring
        for f in range(end - rt.ring_tokens, end):
            for plane in (0, 1):
                _page_token(backing, rt, rt.ring_page_at(slot, index, f), plane, f % rt.page_tokens).random_(0, 256)
        torch.cuda.synchronize()
        rt._fill_staging(index, backing, index * rt.layer_bytes, slot, first, end)
        torch.cuda.synchronize()
        run = rt.carve.staging_row(index % 2)
        ring_start = max(0, end - rt.ring_tokens + RING_MARGIN)
        pool = slot * rt.slot_bytes + index * rt.layer_bytes
        for f in range(end):
            page, offset = divmod(f, rt.page_tokens)
            for plane in (0, 1):
                got = _page_token(backing, rt, run[page], plane, offset).cpu()
                if f < ring_start:
                    start = pool + plane * rt.plane_bytes + f * rt.token_bytes
                    want = host[start:start + rt.token_bytes]
                else:
                    want = _page_token(backing, rt, rt.ring_page_at(slot, index, f), plane,
                                       f % rt.page_tokens).cpu()
                assert torch.equal(got, want), (f, plane)
        # seed_ring: ring <- newest page_tokens frame positions from the pool.
        rt.seed_ring(slot, index, backing, 90)
        torch.cuda.synchronize()
        for f in range(90 - rt.ring_tokens, 90):
            for plane in (0, 1):
                start = pool + plane * rt.plane_bytes + f * rt.token_bytes
                got = _page_token(backing, rt, rt.ring_page_at(slot, index, f), plane, f % rt.page_tokens).cpu()
                assert torch.equal(got, host[start:start + rt.token_bytes]), (f, plane)
        # pool_layer_views: FP8 page layout over the host pool (written by kernels by device address).
        key, value = rt.pool_layer_views(0, 1, 2, 256)
        key.view(torch.uint8)[2, 5].fill_(7)
        torch.cuda.synchronize()
        start = rt.layer_bytes + (2 * rt.page_tokens + 5) * rt.token_bytes
        assert host[start:start + rt.token_bytes].eq(7).all()
    finally:
        rt.ext.host_unregister(host.data_ptr())


@pytest.mark.parametrize('ring_pages', (1, 2))
def test_device_stage_fill_matches_the_host_reference(ring_pages):
    import torch
    from vllm_kvmem.adapters.qsa_stream_runtime import RING_MARGIN
    rt, backing, host = _mini_runtime(ring_pages=ring_pages)
    try:
        host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8))
        backing.random_(0, 256)
        slot, index, first, end = 1, 1, 80, 100
        # Batch: request 0 decodes (4 rows, slot 0), request 1 stages [80, 100) (slot 1).
        query_start = torch.tensor([0, 4, 24], dtype=torch.int32, device='cuda')
        positions = torch.tensor([50, 51, 52, 53] + list(range(first, end)), dtype=torch.int64, device='cuda')
        request_slot = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
        streaming = torch.tensor([1, 1, 0], dtype=torch.uint8, device='cuda')
        hot = torch.tensor([row[index] for s, row in sorted(rt.hosts.items())], dtype=torch.int32, device='cuda')
        stats = torch.zeros(4, dtype=torch.int64, device='cuda')
        pool_base = torch.tensor([rt._pool_dev + s * rt.slot_bytes for s in range(2)], dtype=torch.int64,
                                 device='cuda')
        run_device = torch.tensor(rt.carve.staging_row(0), dtype=torch.int32, device='cuda')
        rt.ext.stage_fill(query_start, 2, positions, request_slot, streaming, 4, run_device, pool_base,
                          index * rt.layer_bytes, rt.plane_bytes, backing, rt.page_stride, rt.plane_stride, hot,
                          rt.hot_pages, rt.page_tokens, rt.token_bytes, RING_MARGIN, stats, 16, False)
        rt._fill_staging(index, backing, index * rt.layer_bytes, slot, first, end)   # layer 1 -> run 1
        torch.cuda.synchronize()
        assert int(stats[3]) == 0
        for f in range(end):
            page, offset = divmod(f, rt.page_tokens)
            for plane in (0, 1):
                assert torch.equal(_page_token(backing, rt, rt.carve.staging_row(0)[page], plane, offset),
                                   _page_token(backing, rt, rt.carve.staging_row(1)[page], plane, offset)), f
        # Two staging requests in one step: nothing is copied and the error counter says so.
        rt.ext.stage_fill(torch.tensor([0, 10, 20], dtype=torch.int32, device='cuda'), 2, positions[:20],
                          request_slot, streaming, 4, run_device, pool_base, 0, rt.plane_bytes, backing,
                          rt.page_stride, rt.plane_stride, hot, rt.hot_pages, rt.page_tokens, rt.token_bytes,
                          RING_MARGIN, stats, 16, False)
        torch.cuda.synchronize()
        assert int(stats[3]) == 1
    finally:
        rt.ext.host_unregister(host.data_ptr())


@pytest.mark.parametrize('ring_pages', (1, 2))
def test_prefetched_pool_plus_ring_only_fill_rebuilds_the_frame(ring_pages):
    import torch
    from vllm_kvmem.adapters.qsa_stream_runtime import RING_MARGIN
    rt, backing, host = _mini_runtime(ring_pages=ring_pages)
    try:
        host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8))
        backing.random_(0, 256)
        rt.copy_stream = torch.cuda.Stream()
        rt.staging_first = [rt.carve.staging_row(b)[0] for b in range(2)]
        rt.prefetch_layers = 0
        slot, index, first, end = 1, 1, 80, 100
        # As _frame_pages: frame pages plus one spare, never past the staging run.
        rt.step_pages = min(rt.carve.frame_pages, -(-end // rt.page_tokens) + 1)
        query_start = torch.tensor([0, end - first], dtype=torch.int32, device='cuda')
        positions = torch.arange(first, end, dtype=torch.int64, device='cuda')
        request_slot = torch.tensor([slot], dtype=torch.int32, device='cuda')
        streaming = torch.tensor([0, 1, 0], dtype=torch.uint8, device='cuda')
        hot = torch.tensor([row[index] for s, row in sorted(rt.hosts.items())], dtype=torch.int32, device='cuda')
        stats = torch.zeros(4, dtype=torch.int64, device='cuda')
        pool_base = torch.tensor([rt._pool_dev + s * rt.slot_bytes for s in range(2)], dtype=torch.int64,
                                 device='cuda')
        run = rt.carve.staging_row(index % 2)
        done = rt._prefetch(index, slot, backing)
        torch.cuda.current_stream().wait_event(done)
        rt.ext.stage_fill(query_start, 1, positions, request_slot, streaming, 4,
                          torch.tensor(run, dtype=torch.int32, device='cuda'), pool_base, index * rt.layer_bytes,
                          rt.plane_bytes, backing, rt.page_stride, rt.plane_stride, hot, rt.hot_pages,
                          rt.page_tokens, rt.token_bytes, RING_MARGIN, stats, 16, True)
        torch.cuda.synchronize()
        ring_start = max(0, end - rt.ring_tokens + RING_MARGIN)
        pool = slot * rt.slot_bytes + index * rt.layer_bytes
        for f in range(end):
            page, offset = divmod(f, rt.page_tokens)
            for plane in (0, 1):
                got = _page_token(backing, rt, run[page], plane, offset).cpu()
                if f < ring_start:
                    start = pool + plane * rt.plane_bytes + f * rt.token_bytes
                    want = host[start:start + rt.token_bytes]
                else:
                    want = _page_token(backing, rt, rt.ring_page_at(slot, index, f), plane,
                                       f % rt.page_tokens).cpu()
                assert torch.equal(got, want), (f, plane)
        assert int(stats[3]) == 0 and rt.prefetch_layers == 1
    finally:
        rt.ext.host_unregister(host.data_ptr())
