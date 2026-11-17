"""Host checks of the carved QSA selector pages (core/compressed_pages.py, adapters/qsa_streaming.py)."""
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.adapters import qsa_streaming
from vllm_kvmem.adapters.qsa_streaming import SelectorState, _patch_layer
from vllm_kvmem.core.compressed_pages import frame_tokens, plan_carve


def test_flash_next_carve_geometry():
    memory = SimpleNamespace(select_budget=8192, gen_budget=4096)
    carve = plan_carve(layers=13, slots=3, frame_tokens=frame_tokens(memory, 3), page_tokens=3008,
                       native_page_bytes=3272704, compress_ratio=4, index_head_dim=128)
    assert carve.compressed_page_bytes == 192512 and carve.per_native_page == 17
    assert carve.pages_per_slot == 5           # ceil((8192 + 4096 + 4) / 3008)
    assert carve.blocks_per_layer == 1 and carve.blocks == 13
    assert list(carve.slot_row(2, 8)) == [10, 11, 12, 13, 14, -1, -1, -1]
    big = plan_carve(layers=13, slots=16, frame_tokens=65536 + 4096 + 4, page_tokens=3008,
                     native_page_bytes=3272704, compress_ratio=4, index_head_dim=128)
    assert big.pages_per_slot == 24 and big.blocks_per_layer == 23 and big.blocks == 299
    with pytest.raises(ValueError):
        plan_carve(layers=1, slots=1, frame_tokens=8, page_tokens=3008, native_page_bytes=3272705,
                   compress_ratio=4, index_head_dim=128)


# A miniature pool with the vendor's padded page: 2 x page x width main bytes,
# then a tail of page/4 x dim BF16 selector bytes. 17 compressed pages per native page.
PAGE, DIM, WIDTH = 8, 4, 16
TAIL = PAGE // 4 * DIM * 2
NATIVE = 2 * PAGE * WIDTH + TAIL


def _pool(blocks):
    storage = torch.zeros(blocks * NATIVE, dtype=torch.uint8)
    return storage, storage.as_strided((blocks, 2, PAGE, WIDTH), (NATIVE, PAGE * WIDTH, WIDTH, 1))


def test_carved_view_covers_only_the_layers_blocks():
    storage, kv_cache = _pool(40)
    state = SelectorState(layers=2, frame=3 * PAGE, slots=4)   # 3 pages per slot, 12 per layer -> 1 block
    layer = SimpleNamespace(max_seqs=4, max_seq_len=10 * PAGE)
    views = [state.carved_view(i, layer, kv_cache, 4, DIM) for i in range(2)]
    # Placed at the end of the 40-block pool; QSA main views stop before it.
    assert state.carve.blocks_per_layer == 1 and state.carve.first_block == 38 and state.carve.end_block == 40
    assert state.main_end == 38 and state.streaming_carve is None
    assert views[0].shape == (17, 2, DIM) and views[0].dtype == torch.bfloat16
    views[1].fill_(1.0)
    # Layer 1 owns native block 39 exactly; everything below it is untouched.
    touched = storage.view(40, NATIVE).ne(0).any(dim=1).nonzero().flatten().tolist()
    assert touched == [39]
    assert state.slot_table.tolist()[3] == [9, 10, 11] + [-1] * 7
    # A throwaway profiling cache smaller than the carve keeps the vendor tail view.
    assert state.carved_view(0, layer, _pool(2)[1], 4, DIM) is None


def test_staging_fills_compressed_rows_by_state_slot():
    _, kv_cache = _pool(40)
    state = SelectorState(layers=1, frame=2 * PAGE, slots=4)
    host = SimpleNamespace(max_seqs=4, max_seq_len=6 * PAGE)
    state.carved_view(0, host, kv_cache, 4, DIM)
    calls = []

    class Layer:
        def bind_kv_cache(self, kv_cache):
            calls.append('bind')

        def _ensure_qsa_context_buffers(self, context):
            context.main_block_table = torch.full((4, 6), -1, dtype=torch.int32)
            context.compressed_block_table = context.main_block_table

        def _stage_runtime_metadata(self, metadata, rows, *, row_capacity, main_block_table=None,
                                    compressed_block_table=None):
            for table in {id(main_block_table): main_block_table,
                          id(compressed_block_table): compressed_block_table}.values():
                table.fill_(-1)
                table[:2].copy_(metadata.block_table[:2, :6])
            return 'staged'

        def _require_qsa_metadata(self, metadata):
            return None, metadata.slots, None, None

        def _bind_qsa_context(self, context, staged=None, output=None):
            return int(self.kv_cache.shape[0])

    layer = Layer()
    layer.kv_cache = kv_cache
    _patch_layer(layer, 0, state)
    context = SimpleNamespace(main_block_table=None, compressed_block_table=None)
    layer._ensure_qsa_context_buffers(context)
    assert context.compressed_block_table is not context.main_block_table
    metadata = SimpleNamespace(block_table=torch.arange(2 * 8, dtype=torch.int32).view(2, 8),
                               seq_lens=torch.tensor([5, 9]), slots=torch.tensor([3, 1], dtype=torch.int32))
    assert layer._stage_runtime_metadata(metadata, 2, row_capacity=4, main_block_table=context.main_block_table,
                                         compressed_block_table=context.compressed_block_table) == 'staged'
    assert context.main_block_table[:2].tolist() == [[0, 1, 2, 3, 4, 5], [8, 9, 10, 11, 12, 13]]
    assert context.compressed_block_table.tolist() == [[6, 7, -1, -1, -1, -1], [2, 3, -1, -1, -1, -1],
                                                       [-1] * 6, [-1] * 6]
    # B12X sees main K/V pages below the carve only, during the binding call.
    assert layer._bind_qsa_context(context) == state.main_end == 39
    assert layer.kv_cache is kv_cache


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


def test_scheduler_reservation_takes_the_last_blocks(monkeypatch):
    from vllm_kvmem.core.compressed_pages import place_carves
    from vllm_kvmem.serving import capacity

    carve = plan_carve(layers=2, slots=3, frame_tokens=12292, page_tokens=3008, native_page_bytes=3272704,
                       compress_ratio=4, index_head_dim=128)

    def build(pool, placed):
        native = SimpleNamespace(block_pool=pool, coordinator=SimpleNamespace(single_type_managers=()),
                                 max_model_len=1)
        config = SimpleNamespace(speculative_config=None, max_concurrent_batches=1,
                                 cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode='none'))
        monkeypatch.setattr(capacity.ReservedKVCacheManager, '_demand', lambda self, tokens: 0)
        import vllm_kvmem.adapters.engine_api as engine_api
        monkeypatch.setattr(engine_api, 'require_contract', lambda name: None)
        return capacity.ReservedKVCacheManager(native, config, selector_carve=placed)

    pytest.importorskip('vllm')
    pool = FakePool(100)
    placed, _, main_end = place_carves(carve, None, 100)
    manager = build(pool, placed)
    assert main_end == 98 and [b.block_id for b in manager.selector_blocks] == [98, 99]
    assert [b.block_id for b in pool.free] == list(range(1, 98))       # free order kept
    with pytest.raises(ValueError):
        build(FakePool(120), placed)            # not the end of this pool
    used = FakePool(100)
    used.get_new_blocks(1)                       # block 1 taken: still fine
    build(used, placed)
    stale = FakePool(100)
    stale.free = stale.free[:-1]                 # block 99 not free
    with pytest.raises(ValueError):
        build(stale, placed)


_VENDOR = '''from ..common.b12x_qsa_cache import qsa_compressed_cache_view
class Qwen4ExpQSAAttention:
    def bind_kv_cache(self, kv_cache):
        compressed_cache = qsa_compressed_cache_view(
            kv_cache, compress_ratio=self.compress_ratio, index_head_dim=self.index_head_dim)
        self._compressed_cache = compressed_cache
    def _ensure_qsa_context_buffers(self, context): pass
    def _stage_runtime_metadata(self, metadata, rows, *, row_capacity, main_block_table=None,
                                compressed_block_table=None): pass
    @staticmethod
    def _require_qsa_metadata(metadata): pass
    def _bind_qsa_context(self, context, staged=None, output=None, *, overlap=False, state=None):
        main_k_cache, main_v_cache = impl._kv_cache_views(self.kv_cache)
'''


def test_split_source_contract(tmp_path):
    from vllm_kvmem.adapters.engine_api import UnsupportedEngine
    from vllm_kvmem.adapters.qsa_contract import check_qsa_split_sources
    root = tmp_path / 'models/qwen4_exp/nvidia'
    root.mkdir(parents=True)
    (root / 'b12x_qsa.py').write_text(_VENDOR)
    check_qsa_split_sources(tmp_path)
    for broken in (_VENDOR.replace('self._compressed_cache = compressed_cache',
                                   'self._compressed_cache = compressed_cache.clone()'),
                   _VENDOR.replace('from ..common.b12x_qsa_cache import qsa_compressed_cache_view\n',
                                   'from ..common import b12x_qsa_cache\n'),
                   _VENDOR.replace('compressed_block_table=None', 'tables=None'),
                   _VENDOR.replace('impl._kv_cache_views(self.kv_cache)', 'impl._kv_cache_views(cache)')):
        (root / 'b12x_qsa.py').write_text(broken)
        with pytest.raises(UnsupportedEngine):
            check_qsa_split_sources(tmp_path)
    assert qsa_streaming.VENDOR_MODULE.endswith('b12x_qsa')


def test_a_later_pool_binding_replaces_the_placement():
    state = SelectorState(layers=1, frame=2 * PAGE, slots=4)
    layer = SimpleNamespace(max_seqs=4, max_seq_len=6 * PAGE)
    state.carved_view(0, layer, _pool(40)[1], 4, DIM)          # e.g. a profiling-size pool
    assert state.main_end == 39
    view = state.carved_view(0, layer, _pool(60)[1], 4, DIM)   # the real pool, bound last
    assert state.main_end == 59 and state.carve.first_block == 59 and state.views[0] is view


def test_narrow_context_tables_take_the_slot_row_prefix():
    state = SelectorState(layers=1, frame=2 * PAGE, slots=4)
    state.carved_view(0, SimpleNamespace(max_seqs=4, max_seq_len=6 * PAGE), _pool(40)[1], 4, DIM)
    full = state.table(6, torch.device('cpu'))
    assert state.table(3, torch.device('cpu')).tolist() == full[:, :3].tolist()
    with pytest.raises(RuntimeError):
        state.table(7, torch.device('cpu'))
