"""Structural checks for the native APIs KVMem consumes, without importing CUDA.

These checks detect interface drift, not numerical correctness. Layout/ownership
checks stay at their use sites; GPU qualification is separate from this preflight.
Upstream versions, comments and unrelated implementations are not allowlisted.
"""
import ast
from functools import lru_cache
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path


class UnsupportedEngine(ValueError):
    pass


# File, symbol, parameters used by the adapter. No copied upstream implementation.
CONTRACTS = {
    'runner': (
        ('v1/worker/gpu/model_runner.py', 'GPUModelRunner.' + method, params)
        for method, params in (
            ('load_model', ()), ('initialize_kv_cache', ('kv_cache_config',)),
            ('add_requests', ('scheduler_output',)), ('update_requests', ('scheduler_output',)),
            ('finish_requests', ('scheduler_output',)), ('prepare_inputs', ('scheduler_output',)),
            ('prepare_attn', ('input_batch',)),
            ('sample', ('hidden_states', 'input_batch', 'grammar_output')),
            ('sample_tokens', ('grammar_output',)),
            ('execute_model', ('scheduler_output', 'dummy_run', 'is_profile')),
            ('shutdown', ()),
        )),
    'scheduler': (
        ('v1/core/sched/scheduler.py', 'Scheduler.' + method, params)
        for method, params in (
            ('schedule', ('throttle_prefills',)),
            ('update_from_output', ('scheduler_output', 'model_runner_output')),
            ('_free_request', ('request', 'delay_free_blocks')),
        )),
    'capacity': (
        ('v1/core/kv_cache_manager.py', 'KVCacheManager.' + method, params)
        for method, params in (
            ('allocate_slots', ('request', 'num_new_tokens', 'num_lookahead_tokens')),
            ('free', ('request',)), ('get_blocks', ('request_id',)),
            ('get_block_ids', ('request_id',)),
            ('truncate_computed_blocks', ('blocks', 'num_computed_tokens')),
        )),
    'restore': (
        ('v1/worker/gpu/block_table.py', 'BlockTables.append_block_ids',
         ('req_index', 'new_block_ids', 'overwrite')),
        ('v1/worker/gpu/block_table.py', 'BlockTables.apply_staged_writes', ()),
        ('v1/worker/gpu/buffer_utils.py', 'StagedWriteTensor.stage_write', ('index', 'start', 'x')),
    ),
    'state': (
        ('model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py',
         'QwenGatedDeltaNetAttention.get_state_shape', ()),
        ('model_executor/layers/mamba/mamba_utils.py', 'is_conv_state_dim_first', ()),
        ('v1/worker/gpu/model_states/mamba_hybrid.py',
         'MambaHybridModelState.add_request', ('req_index', 'new_req_data')),
        ('v1/worker/gpu/model_states/mamba_hybrid.py',
         'MambaHybridModelState.postprocess_state', ('num_sampled',)),
    ),
    'draft': (
        ('v1/worker/gpu/spec_decode/__init__.py', 'init_speculator', ('vllm_config', 'device')),
        ('v1/worker/gpu/spec_decode/dflash/speculator.py', 'DFlashSpeculator.propose',
         ('input_batch', 'last_hidden_states', 'num_sampled', 'num_rejected')),
        ('v1/worker/gpu/spec_decode/dflash2/speculator.py', 'DFlash2Speculator', ()),
    ),
    'mtp': (
        ('v1/worker/gpu/spec_decode/mtp/speculator.py', 'MTPSpeculator.load_draft_model',
         ('target_model', 'target_attn_layer_names')),
        ('model_executor/models/qwen3_5_mtp.py', 'Qwen3_5MTP', ()),
    ),
    'retirement': (
        ('v1/worker/gpu/kv_connector.py', 'ActiveKVConnector.no_forward', ('scheduler_output',)),
        ('v1/worker/gpu/kv_connector.py', 'ActiveKVConnector.post_forward', ('finished_req_ids',)),
    ),
    'systemone': (
        ('engine/protocol.py', 'EngineClient.generate', ('prompt', 'sampling_params', 'request_id', 'session_id')),
        ('entrypoints/generate/structured_decisions/serving.py',
         'ServingStructuredDecisions.create_decision', ('request',)),
        ('plugins/endpoint_plugins/interface.py', 'EndpointPlugin.init_state', ('engine_client', 'state', 'args')),
    ),
    'systemone_active': (
        ('v1/engine/async_llm.py', 'AsyncLLM.collective_rpc', ('method', 'kwargs')),
        ('v1/worker/gpu/sample/logprob.py', 'compute_token_logprobs', ('logits', 'token_ids')),
    ),
}
CONTRACTS = {name: tuple(items) for name, items in CONTRACTS.items()}

# These ownership-sensitive operations matter even when signatures are stable.
# Compare AST expressions, so formatting, comments and unrelated edits are free.
INVARIANTS = {
    'restore': (
        ('v1/worker/gpu/block_table.py', 'BlockTables.append_block_ids',
         'self.block_tables[i].stage_write(req_index, start, block_ids)'),
    ),
    'state': (
        ('v1/worker/gpu/model_states/mamba_hybrid.py', 'MambaHybridModelState.add_request',
         'self.num_accepted_tokens_gpu[req_index].fill_(1)'),
        ('v1/worker/gpu/model_states/mamba_hybrid.py', '_scatter_num_accepted_kernel',
         'tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))'),
    ),
    'retirement': (
        ('v1/worker/gpu/kv_connector.py', 'ActiveKVConnector.no_forward', 'self.finish_forward()'),
        ('v1/worker/gpu/kv_connector.py', 'ActiveKVConnector.post_forward',
         'self.kv_connector.get_transfer_results(finished_req_ids)'),
        ('v1/worker/gpu/kv_connector.py', 'ActiveKVConnector.post_forward',
         'self.kv_connector.build_connector_worker_meta()'),
    ),
}


def engine_root():
    # Respect source/editable installs instead of inspecting a different wheel
    # that happens to provide the distribution metadata on the same sys.path.
    spec = find_spec('vllm')
    if spec is not None and spec.origin:
        return Path(spec.origin).parent
    return Path(distribution('vllm').locate_file('vllm'))


def find_symbol(source, symbol):
    node = ast.parse(source)
    for part in symbol.split('.'):
        matches = [child for child in getattr(node, 'body', ())
                   if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and child.name == part]
        if len(matches) != 1:
            raise UnsupportedEngine(f'vLLM API missing or ambiguous: {symbol}')
        node = matches[0]
    return node


def check_source(source, symbol, parameters=()):
    node = find_symbol(source, symbol)
    if isinstance(node, ast.ClassDef):
        return
    args = node.args
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    missing = set(parameters) - names
    if missing:
        raise UnsupportedEngine(f'vLLM API {symbol} is missing parameters: {sorted(missing)}')


def check_contract(name, root):
    root = Path(root)
    for relative, symbol, parameters in CONTRACTS[name]:
        path = root / relative
        if not path.is_file():
            raise UnsupportedEngine(f'vLLM {name} capability missing: {relative}')
        try:
            check_source(path.read_text(), symbol, parameters)
        except (UnsupportedEngine, SyntaxError) as error:
            raise UnsupportedEngine(f'vLLM {name} capability incompatible ({relative}): {error}') from error
    for relative, symbol, expression in INVARIANTS.get(name, ()):
        node = find_symbol((root / relative).read_text(), symbol)
        required = ast.dump(ast.parse(expression, mode='eval').body)
        if not any(ast.dump(child) == required for child in ast.walk(node)):
            raise UnsupportedEngine(f'vLLM {name} ownership contract changed: {symbol}: {expression}')


@lru_cache(maxsize=None)
def require_contract(name):
    check_contract(name, engine_root())


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='vLLM package directory in an upstream checkout')
    parser.add_argument('--systemone', action='store_true', help='Also require the native SystemOne API')
    args = parser.parse_args()
    root = args.source or engine_root()
    features = [name for name in CONTRACTS if name != 'systemone' or args.systemone]
    for name in features:
        check_contract(name, root)
    from .compat import patched_source
    from .metadata_patch import patch_metadata_source
    patched_source((root / 'v1/worker/gpu/spec_decode/dflash/utils.py').read_bytes())
    patch_metadata_source((root / 'v1/worker/gpu/attn_utils.py').read_bytes())
    from .logprob_patch import patch_rejection_source
    patch_rejection_source((root / 'v1/worker/gpu/spec_decode/rejection_sampler.py').read_bytes())
    print(json.dumps({'source': str(root), 'structural_checks': features,
                      'gpu_qualified': False}, indent=2))


if __name__ == '__main__':
    main()
