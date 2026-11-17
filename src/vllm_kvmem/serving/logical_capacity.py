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
from ..core import settings as _settings

TARGET = 'vllm.v1.core.kv_cache_utils'


def physical_config(config):
    from ..core.options import RuntimeOptions
    from .pressure_query_capture import configured_method
    recall = configured_method()
    finite_history = _settings.get('KVMEM_FULL_HISTORY_BENCHMARK') == '1'
    transfer = getattr(config, 'kv_transfer_config', None)
    payload = getattr(transfer, 'kv_connector_extra_config', {}) or {}
    incremental = payload.get('kvmem', {}).get('gpu_memory_admission') == 'incremental'
    bounded_dflash = (_settings.get('KVMEM_LOGICAL_KV_CAPACITY') == '1' and
        getattr(getattr(config, 'speculative_config', None), 'method', None) == 'dflash')
    if not recall and not finite_history and not incremental and not bounded_dflash and config.model_config.max_model_len <= 1048576:
        return config
    transfer = config.kv_transfer_config
    if (transfer is None or transfer.kv_connector != 'KVMemConnector' or
            transfer.kv_connector_module_path != 'vllm_kvmem.serving.connector' or
            config.scheduler_config.scheduler_cls != 'vllm_kvmem.serving.scheduler.KVMemScheduler'):
        raise ValueError('Bounded logical KV planning requires the owned KVMem scheduler/connector')
    options = RuntimeOptions.from_vllm(config)
    if bounded_dflash:
        from ..adapters.draft_contract import DraftContract
        contract = DraftContract.from_vllm(config, options.memory)
        frame = options.memory.select_budget + options.memory.gen_budget
        if (contract.query_tokens != 2 or recall or finite_history or incremental or options.historical_gc is not None or
                not options.request_planning or not options.rolling_active_window or
                not options.cpu_history_coordination or not options.cpu_progress_reserve or
                not options.session_capacity or options.async_refresh or options.async_restore or
                config.cache_config.num_gpu_blocks_override is not None or
                config.model_config.original_max_model_len == -1 or
                not frame < config.model_config.max_model_len <= contract.max_position or
                getattr(config.parallel_config, 'tensor_parallel_size', 1) not in (1, 2) or
                any(getattr(config.parallel_config, field, 1) != 1 for field in
                    ('pipeline_parallel_size','decode_context_parallel_size',
                     'prefill_context_parallel_size'))):
            raise ValueError('Unqualified DFlash2 bounded logical KV planning configuration')
        result = copy(config)
        result.model_config = copy(config.model_config)
        result.model_config.max_model_len = frame
        return result
    if incremental:
        if (recall or finite_history or config.scheduler_config.async_scheduling or
                config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != 'none' or
                (config.speculative_config is not None and config.speculative_config.method != 'mtp') or
                any(getattr(config.parallel_config, field, 1) != 1 for field in
                    ('tensor_parallel_size', 'pipeline_parallel_size', 'decode_context_parallel_size',
                     'prefill_context_parallel_size'))):
            raise ValueError('Incremental KV planning requires single-rank synchronous target or native MTP')
        result = copy(config)
        result.model_config = copy(config.model_config)
        result.model_config.max_model_len = options.memory.select_budget + options.memory.gen_budget
        return result
    gc = options.historical_gc
    if recall:
        # This opt-in campaign retains the complete CPU history while native
        # attention owns one frozen K+G frame per explicitly declared owner.
        # The native planner below still checks the complete all-owner pool.
        # Do not extend this exception
        # to ordinary no-GC serving or a different logical/physical geometry.
        if (config.model_config.max_model_len != 524288 or
                getattr(config.parallel_config, 'tensor_parallel_size', 1) != 1 or
                _settings.get('KVMEM_RECALL_CONCURRENT_OWNERS', '1') not in ('1', '2', '4', '8') or
                str(config.scheduler_config.max_num_seqs) != _settings.get('KVMEM_RECALL_CONCURRENT_OWNERS', '1') or
                gc is not None or
                options.memory.block_tokens != 32 or
                options.memory.select_budget != 36864 or
                options.memory.gen_budget != 16384 or not options.async_refresh):
            raise ValueError('Unqualified recall logical/physical KV configuration')
    elif finite_history:
        from .full_history_benchmark import validate_runtime_capacity
        validate_runtime_capacity(config, options)
    elif gc is None or gc.mode != 'delete':
        raise ValueError('Unqualified bounded logical KV planning configuration')
    if (not options.request_planning or
            (not options.rolling_active_window and not finite_history) or not options.cpu_history_coordination or
            not options.cpu_progress_reserve or not options.session_capacity or
            config.speculative_config is None or config.speculative_config.method != 'mtp' or
            config.scheduler_config.async_scheduling or config.cache_config.enable_prefix_caching or
            config.cache_config.mamba_cache_mode != 'none' or
            config.cache_config.num_gpu_blocks_override is not None or
            config.model_config.original_max_model_len == -1 or
            # TP=2 ranks own disjoint heads with identical local geometry (check_specs).
            getattr(config.parallel_config, 'tensor_parallel_size', 1) not in (1, 2) or
            any(getattr(config.parallel_config, field, 1) != 1 for field in
                ('pipeline_parallel_size','decode_context_parallel_size',
                 'prefill_context_parallel_size'))):
        raise ValueError('Unqualified bounded logical KV planning configuration')
    frame = options.memory.select_budget + options.memory.gen_budget
    if not frame <= 262144 or config.model_config.max_model_len > 10000000:
        raise ValueError('Unqualified logical/physical frame length')
    result = copy(config)
    result.model_config = copy(config.model_config)
    result.model_config.max_model_len = frame
    return result


def check_specs(specs, *, allow_dflash=False, expected_workers=1):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
    if (expected_workers not in (1, 2) or
            len(specs) != expected_workers or any(not row for row in specs)):
        raise ValueError('Bounded logical planning requires every declared worker')
    # TP ranks own disjoint heads with identical local cache geometry. Native
    # planning still receives each rank's available bytes independently.
    if any(row != specs[0] for row in specs[1:]):
        raise ValueError('TP workers disagree on local cache geometry')
    for row in specs:
        if not any(type(spec) is FullAttentionSpec for spec in row.values()):
            raise ValueError('No bounded full-attention cache')
        for spec in row.values():
            if allow_dflash and type(spec) is SlidingWindowSpec:
                import torch
                if spec.dtype != torch.bfloat16 or spec.sliding_window != 2048:
                    raise ValueError('Unqualified DFlash2 sliding cache for bounded planning')
            elif type(spec) not in (FullAttentionSpec, MambaSpec):
                raise ValueError('Unsupported cache type for bounded logical planning')



def window_owner_pages(group, config, frame):
    from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
    spec = group.kv_cache_spec
    first = spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
    if (getattr(config.speculative_config, 'method', None) == 'dflash' and
            type(first) is SlidingWindowSpec):
        # Match NativeCapacity: sliding pages recycle before allocation, while
        # one extra DFlash query block can be in flight beyond target tokens.
        lookahead = config.speculative_config.num_speculative_tokens + 1
        return first.max_admission_blocks_per_request(
            config.max_in_flight_tokens + lookahead, frame)
    return spec.max_num_blocks_per_req(config, frame)


def wrap_planner(original):
    if getattr(original, '_kvmem_logical_capacity', False):
        return original

    @wraps(original)
    def plan(vllm_config, kv_cache_specs, available_memory):
        bounded = physical_config(vllm_config)
        if bounded is vllm_config:
            return original(vllm_config, kv_cache_specs, available_memory)
        check_specs(kv_cache_specs,
            allow_dflash=getattr(vllm_config.speculative_config, 'method', None) == 'dflash',
            expected_workers=getattr(vllm_config.parallel_config, 'tensor_parallel_size', 1))
        if len(available_memory) != len(kv_cache_specs):
            raise ValueError('Missing worker memory budget')
        # Native code may unify the input spec dictionaries. Isolate its mapping
        # mutations as well; immutable KVCacheSpec objects themselves are shared.
        result = original(bounded, [dict(row) for row in kv_cache_specs], available_memory)
        owners = vllm_config.scheduler_config.max_num_seqs
        incremental = (vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            'kvmem', {}).get('gpu_memory_admission') == 'incremental')
        rows = []
        for cache in result:
            per_owner = sum(window_owner_pages(group, bounded,
                bounded.model_config.max_model_len) for group in cache.kv_cache_groups)
            required_owners = 1 if incremental else owners
            needed = required_owners*per_owner + 1  # native null page is never available
            if needed > cache.num_blocks:
                raise ValueError(f'GPU pool cannot fit {required_owners} K+G windows plus recurrent states: '
                                 f'need {needed} native pages, have {cache.num_blocks}')
            rows.append(dict(pool_pages=cache.num_blocks, per_owner_pages=per_owner,
                             concurrent_owners=owners, required_pages=needed,
                             admission='incremental' if incremental else 'static'))
        if directory := _settings.get('KVMEM_RUNNER_PROBE_DIR'):
            path = Path(directory)/'logical-kv-capacity.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(status='passed',
                logical_limit=vllm_config.model_config.max_model_len,
                physical_frame=bounded.model_config.max_model_len, workers=rows,
                scope='Native allocation capacity check; no inference qualification'), indent=2)+'\n')
            temp.replace(path)
        return result
    plan._kvmem_logical_capacity = True
    return plan


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode()
        from ..adapters.engine_api import check_source
        check_source(source, 'get_kv_cache_configs',
                     ('vllm_config','kv_cache_specs','available_memory'))
        source += ('\nfrom vllm_kvmem.serving.logical_capacity import wrap_planner as _kvmem_wrap_planner\n'
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
        from ..adapters.engine_api import check_source
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
