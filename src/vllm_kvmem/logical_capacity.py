"""Use the owned K+G window for native startup allocation, not logical history.

Native grouping, padding, null-page checks and allocation remain authoritative.
Only a private config copy used for KV planning has a bounded model length;
HTTP validation, worker token identities and the scheduler retain the full limit.
"""
from copy import copy
from functools import wraps
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

TARGET = 'vllm.v1.core.kv_cache_utils'


def physical_config(config):
    from .options import RuntimeOptions
    if config.model_config.max_model_len <= 1048576:
        return config
    transfer = config.kv_transfer_config
    if (transfer is None or transfer.kv_connector != 'KVMemConnector' or
            transfer.kv_connector_module_path != 'vllm_kvmem.connector' or
            config.scheduler_config.scheduler_cls != 'vllm_kvmem.scheduler.KVMemScheduler'):
        raise ValueError('Bounded logical KV planning requires the owned KVMem scheduler/connector')
    options = RuntimeOptions.from_vllm(config)
    gc = options.historical_gc
    if (gc is None or gc.mode != 'delete' or not options.request_planning or
            not options.rolling_active_window or not options.cpu_history_coordination or
            not options.cpu_progress_reserve or not options.session_capacity or
            config.speculative_config is None or config.speculative_config.method != 'mtp' or
            config.scheduler_config.async_scheduling or config.cache_config.enable_prefix_caching or
            config.cache_config.mamba_cache_mode != 'none' or
            config.cache_config.num_gpu_blocks_override is not None or
            config.model_config.original_max_model_len == -1 or
            any(getattr(config.parallel_config, field, 1) != 1 for field in
                ('tensor_parallel_size','pipeline_parallel_size','decode_context_parallel_size',
                 'prefill_context_parallel_size'))):
        raise ValueError('Unqualified bounded logical KV planning configuration')
    frame = options.memory.select_budget + options.memory.gen_budget
    if not frame <= 262144 or config.model_config.max_model_len > 10000000:
        raise ValueError('Unqualified logical/physical frame length')
    result = copy(config)
    result.model_config = copy(config.model_config)
    result.model_config.max_model_len = frame
    return result


def check_specs(specs):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    if len(specs) != 1 or not specs[0]:
        raise ValueError('Bounded logical planning requires one populated worker')
    if not any(type(spec) is FullAttentionSpec for spec in specs[0].values()):
        raise ValueError('No bounded full-attention cache')
    for spec in specs[0].values():
        if type(spec) not in (FullAttentionSpec, MambaSpec):
            raise ValueError('Unsupported cache type for bounded logical planning')


def wrap_planner(original):
    if getattr(original, '_kvmem_logical_capacity', False):
        return original

    @wraps(original)
    def plan(vllm_config, kv_cache_specs, available_memory):
        bounded = physical_config(vllm_config)
        if bounded is vllm_config:
            return original(vllm_config, kv_cache_specs, available_memory)
        check_specs(kv_cache_specs)
        # Native code may unify the input spec dictionaries. Isolate its mapping
        # mutations as well; immutable KVCacheSpec objects themselves are shared.
        result = original(bounded, [dict(row) for row in kv_cache_specs], available_memory)
        owners = vllm_config.scheduler_config.max_num_seqs
        rows = []
        for cache in result:
            per_owner = sum(group.kv_cache_spec.max_num_blocks_per_req(
                bounded, bounded.model_config.max_model_len) for group in cache.kv_cache_groups)
            needed = owners*per_owner + 1  # native null page is never available
            if needed > cache.num_blocks:
                raise ValueError(f'GPU pool cannot fit {owners} K+G windows plus recurrent states: '
                                 f'need {needed} native pages, have {cache.num_blocks}')
            rows.append(dict(pool_pages=cache.num_blocks, per_owner_pages=per_owner,
                             concurrent_owners=owners, required_pages=needed))
        if directory := os.getenv('KVMEM_RUNNER_PROBE_DIR'):
            path = Path(directory)/'logical-kv-capacity.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(status='passed',
                logical_limit=vllm_config.model_config.max_model_len,
                physical_frame=bounded.model_config.max_model_len, workers=rows,
                scope='Native allocation and all-owner capacity check; no inference qualification'), indent=2)+'\n')
            temp.replace(path)
        return result
    plan._kvmem_logical_capacity = True
    return plan


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode()
        from .engine_api import check_source
        check_source(source, 'get_kv_cache_configs',
                     ('vllm_config','kv_cache_specs','available_memory'))
        source += ('\nfrom vllm_kvmem.logical_capacity import wrap_planner as _kvmem_wrap_planner\n'
                   'get_kv_cache_configs = _kvmem_wrap_planner(get_kv_cache_configs)\n')
        return compile(source, self.path, 'exec')


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError('Cannot locate native KV planner')
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def enable():
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        return
    if TARGET in sys.modules:
        # Engine workers may import core before loading general plugins. At
        # that point startup has not invoked KV allocation; update its exact
        # imported alias too, and reject any competing adapter.
        module = sys.modules[TARGET]
        original = module.get_kv_cache_configs
        if getattr(original, '_kvmem_logical_capacity', False):
            return
        from .engine_api import check_source
        check_source(Path(module.__file__).read_text(), 'get_kv_cache_configs',
                     ('vllm_config','kv_cache_specs','available_memory'))
        core = sys.modules.get('vllm.v1.engine.core')
        if core is not None and getattr(core, 'get_kv_cache_configs', original) is not original:
            raise RuntimeError('Unexpected native planner alias; refusing competing adaptation')
        module.get_kv_cache_configs = wrap_planner(original)
        if core is not None and hasattr(core, 'get_kv_cache_configs'):
            core.get_kv_cache_configs = module.get_kv_cache_configs
        return
    sys.meta_path.insert(0, _Finder())
