"""Host checks of qsa_streaming: carve layout, page clamp, reservations and the runtime's masks."""
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.core.compressed_pages import (MapAllocator, frame_tokens, plan_carve, plan_streaming_carve,
                                              residency_map_bytes)


def _flash_next(slots=16, select=32768, gen=16384):
    memory = SimpleNamespace(select_budget=select, gen_budget=gen)
    return plan_carve(layers=13, slots=slots, frame_tokens=frame_tokens(memory, 3), page_tokens=3008,
                      native_page_bytes=3272704, compress_ratio=4, index_head_dim=128)


def test_streaming_carve_geometry_k32k_g16k():
    from vllm_kvmem.core.compressed_pages import carve_range, place_carves
    selector = _flash_next()
    carve = plan_streaming_carve(selector, hot_pages=3, max_rows=2048)
    assert selector.pages_per_slot == 17 and selector.blocks == 208
    assert carve.frame_pages == 17 and carve.staging_blocks == 34 and carve.map_blocks == 6
    selector, carve, main_end = place_carves(selector, carve, 3523)
    # [staging 34 | selector 208 | maps 6] ending at the pool end.
    assert carve.first_block == 3523 - 248 == 3275 and selector.first_block == 3309 == main_end
    assert carve.maps_first_block == 3517 and carve.end_block == 3523
    assert carve_range(selector, carve) == (3275, 3523)
    assert carve.max_groups == 17 * 3008 // 4 and carve.hot_groups == 3 * 3008 // 4
    assert carve.pages_per_request == 4
    assert carve.staging_row(1) == list(range(3292, 3309))
    assert carve.frame_capacity_tokens == 51136
    # 13 layers x K|V x 51,136 tokens x 512 bytes per slot.
    assert carve.pool_bytes_per_slot(512) == 13 * 2 * 51136 * 512
    with pytest.raises(ValueError):
        carve.staging_row(2)


def test_map_allocator_views_are_disjoint_and_aligned():
    carve = plan_streaming_carve(_flash_next(slots=3, select=8192, gen=4096), hot_pages=2, max_rows=64)
    raw = torch.zeros(carve.map_bytes, dtype=torch.uint8)
    views = MapAllocator(raw, carve.map_plan()).allocate_all()
    spans = []
    for key, shape, dtype in carve.map_plan():
        view = views[key]
        assert tuple(view.shape) == shape and str(view.dtype) == 'torch.' + dtype
        start = view.data_ptr() - raw.data_ptr()
        assert start % 256 == 0
        spans.append((start, start + view.numel() * view.element_size()))
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert spans[-1][1] <= carve.map_bytes == residency_map_bytes(carve.map_plan())
    # Slot masks carry a dummy entry past the last slot.
    assert views['streaming'].shape == (4,) and views['decode_mask'].shape == (4,)
    with pytest.raises(ValueError):
        MapAllocator(raw[:1024], carve.map_plan()).allocate_all()


def test_options_require_the_selector_split():
    from vllm_kvmem.core.options import RuntimeOptions
    fields = RuntimeOptions.__dataclass_fields__
    assert fields['qsa_streaming'].default is False and fields['qsa_hot_pages'].default == 3


vllm = pytest.importorskip('vllm')


def _manager(blocks=40):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm_kvmem.serving.window_manager import WorkingSetManager
    pool = BlockPool(blocks, False, 16)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8)
    return WorkingSetManager(FullAttentionManager(spec, pool, False, 0, 16), 64, 64), pool


class _Event:
    def query(self):
        return True


def test_streaming_requests_own_exactly_their_physical_pages():
    from vllm_kvmem.serving.window_manager import RefreshRequired
    manager, pool = _manager()
    manager.set_streaming(4 * 16)
    free = pool.get_num_free_blocks()
    # A 10-token prefill and a 120-token frame both own the same four pages.
    assert manager.get_num_blocks_to_allocate('a', 10, (), 0, 0, 10) == 4
    first = manager.allocate_new_blocks('a', 10, 10)
    assert len(first) == 4 and pool.get_num_free_blocks() == free - 4
    assert manager.get_num_blocks_to_allocate('a', 120, (), 10, 10, 120) == 0
    assert manager.allocate_new_blocks('a', 120, 120) == []
    # Logical limits are unchanged: the frame still refreshes at K + G.
    with pytest.raises(RefreshRequired):
        manager.allocate_new_blocks('a', 129, 129)
    ticket = manager.prepare_refresh('a', 128, 48)
    assert len(ticket.retained_block_ids) == 4
    assert manager.commit_refresh(ticket, _Event())
    assert [b.block_id for b in manager.req_to_blocks['a']] == [b.block_id for b in first]
    assert manager.offsets['a'] == 80
    manager.free('a')
    assert pool.get_num_free_blocks() == free


def test_streaming_must_be_configured_before_admission():
    manager, _ = _manager()
    manager.allocate_new_blocks('a', 10, 10)
    with pytest.raises(ValueError):
        manager.set_streaming(64)
    fresh, _ = _manager()
    with pytest.raises(ValueError):
        fresh.set_streaming(65)


def test_reservation_takes_staging_selector_and_maps_at_the_pool_end(monkeypatch):
    from vllm_kvmem.core.compressed_pages import place_carves
    from vllm_kvmem.serving import capacity

    selector = plan_carve(layers=2, slots=3, frame_tokens=12292, page_tokens=3008, native_page_bytes=3272704,
                          compress_ratio=4, index_head_dim=128)
    stream = plan_streaming_carve(selector, hot_pages=3, max_rows=2048)
    selector, stream, main_end = place_carves(selector, stream, 200)
    native = SimpleNamespace(block_pool=FakePool(200), coordinator=SimpleNamespace(single_type_managers=()),
                             max_model_len=1)
    config = SimpleNamespace(speculative_config=None, max_concurrent_batches=1,
                             cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode='none'))
    monkeypatch.setattr(capacity.ReservedKVCacheManager, '_demand', lambda self, tokens: 0)
    import vllm_kvmem.adapters.engine_api as engine_api
    monkeypatch.setattr(engine_api, 'require_contract', lambda name: None)
    manager = capacity.ReservedKVCacheManager(native, config, selector_carve=selector, streaming_carve=stream)
    assert [b.block_id for b in manager.selector_blocks] == list(range(stream.first_block, 200))
    assert stream.staging_row(1)[-1] < main_end <= stream.maps_first_block
    other = plan_streaming_carve(plan_carve(layers=3, slots=3, frame_tokens=12292, page_tokens=3008,
                                            native_page_bytes=3272704, compress_ratio=4, index_head_dim=128),
                                 hot_pages=3, max_rows=2048)
    native.block_pool = FakePool(200)
    with pytest.raises(ValueError):
        capacity.ReservedKVCacheManager(native, config, selector_carve=selector, streaming_carve=other)


class FakePool:
    """BlockPool subset: a fresh free queue 1..n-1 (block 0 is the null page)."""

    def __init__(self, n):
        self.num_gpu_blocks, self.free = n, [SimpleNamespace(block_id=i) for i in range(1, n)]

    def get_num_free_blocks(self):
        return len(self.free)

    def get_new_blocks(self, n):
        taken, self.free = self.free[:n], self.free[n:]
        return taken

    def free_blocks(self, blocks):
        self.free.extend(blocks)


def _runtime(slots=4, hot_pages=2, width=6, ring_pages=1):
    """A StreamingRuntime with only its mask/table state (CPU tensors)."""
    from vllm_kvmem.adapters.qsa_stream_runtime import SPOOF, StreamingRuntime
    runtime = StreamingRuntime.__new__(StreamingRuntime)
    runtime.hot_pages = hot_pages
    runtime.ring_pages = ring_pages
    runtime.verify_rows = 4
    runtime.streaming = torch.zeros(slots + 1, dtype=torch.uint8)
    runtime.decode_mask = torch.zeros(slots + 1, dtype=torch.uint8)
    runtime.spoof = torch.full((16,), SPOOF, dtype=torch.int64)
    runtime.row_positions = torch.zeros(16, dtype=torch.int64)
    runtime.state = [dict(hot_page_ids=torch.tensor([[100 * layer + 10 * s + j for j in range(hot_pages + ring_pages)]
                                                     for s in range(slots)], dtype=torch.int32))
                     for layer in range(2)]
    runtime.carve = SimpleNamespace(staging_buffers=2)
    runtime.staging_tables = [torch.tensor([90, 91, 92] + [-1] * (width - 3), dtype=torch.int32),
                              torch.tensor([95, 96, 97] + [-1] * (width - 3), dtype=torch.int32)]
    return runtime


def test_decode_mask_drops_prefilling_slots_only():
    runtime = _runtime()
    runtime.streaming[:4] = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
    slots = torch.tensor([3, 0, 2, -1], dtype=torch.int32)
    # rows per request 9, 4, 6, 7: requests 0, 2, 3 need staging (> one verify interval)
    staging = runtime._staging_requests(torch.tensor([0, 9, 13, 19, 26], dtype=torch.int32), 4)
    assert staging.tolist() == [True, False, True, True]
    mask = runtime._decode_mask(slots, staging, 4)
    assert mask.tolist() == [1, 1, 0, 0, 0]
    assert runtime._decode_mask(slots, None, 0).tolist() == [1, 1, 0, 1, 0]


def test_spoofed_positions_only_for_streaming_decode_rows():
    from vllm_kvmem.adapters.qsa_stream_runtime import SPOOF
    runtime = _runtime()
    mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.uint8)
    slots = torch.tensor([2, 1, -1], dtype=torch.int32)                 # request -> slot
    request_ids = torch.tensor([0, 0, 1, 2, -1], dtype=torch.int32)      # row -> request
    positions = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int64)
    out = runtime._spoofed(request_ids, positions, slots, mask, 5)
    assert out.tolist() == [SPOOF, SPOOF, 7, 8, 9]


def test_main_table_rows_for_decode_prefill_and_resident_requests():
    runtime = _runtime()
    runtime.streaming[:4] = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
    table = torch.arange(4 * 6, dtype=torch.int32).view(4, 6)
    original = table.clone()
    slots = torch.tensor([1, 3, 2, -1], dtype=torch.int32)
    # rows 4, 2048, 4, 1: request 1 needs staging; the vendor prefill flag is ignored
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 4, 2052, 2056, 2057], dtype=torch.int32),
                               is_prefilling=torch.tensor([1, 0, 0, 0], dtype=torch.uint8))
    runtime.rewrite_main_table(1, metadata, table, slots, 4)
    assert table[0].tolist() == [110, 111, 112, -1, -1, -1]     # decoding: hot pages then ring
    assert table[1].tolist() == [95, 96, 97, -1, -1, -1]        # prefilling: staging run of layer 1
    assert torch.equal(table[2], original[2])                   # resident request untouched
    assert torch.equal(table[3], original[3])                   # padded row untouched


def test_warmup_requests_are_not_streamed():
    from vllm_kvmem.adapters.qsa_stream_runtime import _is_warmup
    assert _is_warmup('_warmup_3_') and _is_warmup('_v2_mixed_warmup_decode_')
    assert not _is_warmup('cmpl-123') and not _is_warmup('warmup-user-request')


def test_short_streaming_requests_reserve_the_full_page_set():
    from vllm_kvmem.serving import capacity
    manager, _ = _manager()
    manager.set_streaming(4 * 16)
    owner = capacity.ReservedKVCacheManager.__new__(capacity.ReservedKVCacheManager)
    owner.groups, owner.static_blocks = (manager,), {}
    assert owner._demand(10) == 4 and owner._demand(1000) == 4 and owner._demand(0) == 0


def test_unbound_runtime_passes_through():
    from vllm_kvmem.adapters.qsa_stream_runtime import StreamingRuntime
    runtime = StreamingRuntime.__new__(StreamingRuntime)
    runtime.carve = None
    binding = SimpleNamespace(selected_positions=torch.zeros(1))
    assert runtime(binding, {}, None) is None
    calls = []
    runtime.reuse_attention(lambda *a, **k: calls.append(k['query_positions']), binding,
                            query=None, request_ids=None, query_positions=7, reuse=True)
    assert calls == [7]


def test_staging_steps_never_replay_a_graph():
    cg = pytest.importorskip('vllm.v1.worker.gpu.cudagraph_utils')
    from vllm.config import CUDAGraphMode
    from vllm_kvmem.adapters.qsa_stream_runtime import StreamingRuntime, install_eager_staging
    runtime = StreamingRuntime.__new__(StreamingRuntime)
    runtime.hosts, runtime.verify_rows, runtime.eager_staging_steps = {0: []}, 4, 0
    full = cg.BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=32, num_reqs=1)
    manager = SimpleNamespace(dispatch=lambda *a, **k: full)
    install_eager_staging(runtime, manager)
    assert manager.dispatch(1, 30, None, 0, max_query_len=30).cg_mode == CUDAGraphMode.NONE   # small prefill
    assert manager.dispatch(4, 16, 4, 0, max_query_len=4) is full                            # verify rows
    assert manager.dispatch(4, 10, None, 0, max_query_len=4) is full                         # varlen decode
    runtime.hosts = {}
    assert manager.dispatch(1, 30, None, 0, max_query_len=30) is full                        # nothing streams
    assert runtime.eager_staging_steps == 1


def test_multi_page_ring_rows_cycle_and_decode_tables_cover_every_ring_page():
    runtime = _runtime(ring_pages=2, width=8)
    runtime.page_tokens = 32
    runtime.hosts = {1: [[11, 12, 13, 14], [21, 22, 23, 24]]}          # H0 H1 R0 R1 per layer
    # Frame column c writes to ring page c % 2.
    assert runtime.override_row(5, None, 1, 0) == [13, 14, 13, 14, 13]
    assert [runtime.ring_page_at(1, 1, f) for f in (0, 31, 32, 64, 100)] == [23, 23, 24, 23, 24]
    runtime.streaming[:4] = 1
    table = torch.full((2, 8), 7, dtype=torch.int32)
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32))
    runtime.rewrite_main_table(0, metadata, table, torch.tensor([1, 2], dtype=torch.int32), 2)
    # Decode rows attend [H0, H1, R0, R1]: the plan's ring token indices reach both ring pages.
    assert table[0].tolist() == [10, 11, 12, 13, -1, -1, -1, -1]
    assert table[1].tolist() == [20, 21, 22, 23, -1, -1, -1, -1]
