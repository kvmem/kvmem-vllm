"""Structural preflight for the B12X QSA APIs used by the eager adapter."""
import ast
from contextlib import contextmanager
from functools import lru_cache

from .engine_api import UnsupportedEngine, check_source, engine_root, find_symbol


def check_qsa_sources(root):
    base = root / 'models/qwen4_exp/nvidia'
    contracts = {
        'b12x_qsa.py': (
            ('Qwen4ExpQSAAttention._project_qkv_gate', ('qkv', 'positions')),
            ('Qwen4ExpQSAAttention.set_skip_topk', ('skip',)),
            ('Qwen4ExpQSAAttention.get_recurrent_checkpoint_tensors', ()),
            ('Qwen4ExpQSAImpl._kv_cache_views', ('kv_cache',))),
        'b12x_indexer_qsa.py': (('QSAIndexer.project', ('hidden_states',)),),
        'model_state.py': (('Qwen4ExpModelState.get_recurrent_checkpoint_tensors', ()),),
        'ple_layer.py': (('Qwen4ExpPLELayer.get_state_shape', ()),),
    }
    sources = {}
    for filename, methods in contracts.items():
        path = base / filename
        if not path.is_file():
            raise UnsupportedEngine(f'Flash Next B12X capability missing: {path}')
        source = sources[filename] = path.read_text()
        for name, parameters in methods:
            check_source(source, name, parameters)
    # The complete per-slot state, its ownership and the cache head layout
    # matter even if upstream keeps the method signatures unchanged.
    invariants = (
        ('b12x_qsa.py', 'Qwen4ExpQSAAttention.get_recurrent_checkpoint_tensors',
         '(self._raw_k_ring, self._raw_logical_positions, self._raw_rope_positions, '
         'self._raw_interval_start_positions)'),
        ('model_state.py', 'Qwen4ExpModelState.get_recurrent_checkpoint_tensors',
         '(self.qsa_state_is_fresh_gpu, self.qsa_committed_num_accepted_tokens_gpu)'),
        ('b12x_qsa.py', 'Qwen4ExpQSAImpl._kv_cache_views', 'kv_cache.unbind(1)'),
        ('b12x_qsa.py', 'Qwen4ExpQSAImpl._kv_cache_views',
         'key_cache.unflatten(-1, (self.num_kv_heads, self.head_size))'),
    )
    for filename, name, expression in invariants:
        method = find_symbol(sources[filename], name)
        wanted = ast.dump(ast.parse(expression, mode='eval').body)
        if not any(ast.dump(node) == wanted for node in ast.walk(method)):
            raise UnsupportedEngine(f'Flash Next state/cache contract changed: {name}')


def check_qsa_split_sources(root):
    """Vendor names adapters/qsa_streaming.py patches for qsa_split_selector."""
    path = root / 'models/qwen4_exp/nvidia/b12x_qsa.py'
    if not path.is_file():
        raise UnsupportedEngine(f'Flash Next B12X capability missing: {path}')
    source = path.read_text()
    for name, parameters in (
            ('Qwen4ExpQSAAttention.bind_kv_cache', ('kv_cache',)),
            ('Qwen4ExpQSAAttention._ensure_qsa_context_buffers', ('context',)),
            ('Qwen4ExpQSAAttention._stage_runtime_metadata',
             ('metadata', 'rows', 'row_capacity', 'main_block_table', 'compressed_block_table')),
            ('Qwen4ExpQSAAttention._require_qsa_metadata', ('metadata',)),
            ('Qwen4ExpQSAAttention._bind_qsa_context', ('context', 'staged', 'output'))):
        check_source(source, name, parameters)
    # The main K/V views B12X binds must come from self.kv_cache, which the
    # adapter narrows to the pages below the carve during the call.
    binder = find_symbol(source, 'Qwen4ExpQSAAttention._bind_qsa_context')
    wanted = ast.dump(ast.parse('impl._kv_cache_views(self.kv_cache)', mode='eval').body)
    if not any(ast.dump(node) == wanted for node in ast.walk(binder)):
        raise UnsupportedEngine('Flash Next QSA binding no longer derives main K/V views from self.kv_cache')
    # The selector view must come from the module-level helper the adapter
    # replaces, and must be the cache every plan and binding uses.
    tree = ast.parse(source)
    if not any(isinstance(node, ast.ImportFrom) and any(a.name == 'qsa_compressed_cache_view' and a.asname is None
                                                        for a in node.names) for node in tree.body):
        raise UnsupportedEngine('Flash Next selector view is no longer a module-level import')
    method = find_symbol(source, 'Qwen4ExpQSAAttention.bind_kv_cache')
    for expression in ('qsa_compressed_cache_view(kv_cache, compress_ratio=self.compress_ratio, '
                       'index_head_dim=self.index_head_dim)',):
        wanted = ast.dump(ast.parse(expression, mode='eval').body)
        if not any(ast.dump(node) == wanted for node in ast.walk(method)):
            raise UnsupportedEngine('Flash Next selector binding contract changed: ' + expression)
    assigned = [node for node in ast.walk(method) if isinstance(node, ast.Assign) and
                any(ast.unparse(t) == 'self._compressed_cache' for t in node.targets)]
    if [ast.unparse(node.value) for node in assigned] != ['compressed_cache']:
        raise UnsupportedEngine('Flash Next selector cache ownership changed in bind_kv_cache')


@lru_cache(maxsize=1)
def verify_qsa_sources():
    check_qsa_sources(engine_root())
    # B12X is packaged inside FlashInfer by the qualified vendor image.
    import inspect
    from b12x.attention.qsa._kernels import launch_compress_completed_groups
    expected = {'raw_index_key', 'query_positions', 'rope_positions', 'request_ids',
        'query_start_loc', 'raw_state_slot_ids', 'raw_k_ring', 'raw_logical_positions',
        'raw_rope_positions', 'key_norm_weight', 'rope_cos', 'rope_sin',
        'compressed_cache', 'compressed_block_table', 'caps'}
    if not expected <= set(inspect.signature(launch_compress_completed_groups).parameters):
        raise UnsupportedEngine('B12X selector compression API changed')


@contextmanager
def release_target_loader_staging(runner):
    """Give InstantTensor driver-visible free memory before constructing MTP.

    Target quantization leaves unused caching-allocator blocks. InstantTensor
    checks cudaMemGetInfo rather than PyTorch's reusable reserved bytes, so its
    second iterator can reject a model whose live weights fit comfortably.
    This startup-only fence/trim preserves every live model allocation.
    """
    import gc
    import logging
    import torch
    proposer = runner.speculator
    original = proposer.load_model
    def load(*args, **kwargs):
        torch.cuda.synchronize(runner.device)
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(runner.device)
        logging.getLogger(__name__).info('Flash Next MTP loader: %.2f GiB free after releasing target staging',
                                        free / (1 << 30))
        return original(*args, **kwargs)
    proposer.load_model = load
    try:
        yield
    finally:
        proposer.load_model = original
