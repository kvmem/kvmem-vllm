"""Opt-in CPU scopes for native execution and KVMem's context lifecycle.

These ranges annotate launches and host work; they do not synchronize CUDA or
measure GPU duration themselves. Native objects are wrapped per runner instance.
Project-owned classes are wrapped once in the dedicated profiler worker process.
Ordinary serving never installs these wrappers.
"""
from functools import wraps


def _wrapped_for(method, stage):
    return getattr(method, '_kvmem_profile_stage', None) == stage


def _wrap(owner, method_name, stage):
    if owner is None:
        return
    original = getattr(owner, method_name, None)
    if not callable(original) or _wrapped_for(original, stage):
        return
    from torch.profiler import record_function

    @wraps(original)
    def scoped(*args, **kwargs):
        with record_function('kvmem::' + stage):
            return original(*args, **kwargs)

    scoped._kvmem_profile_stage = stage
    setattr(owner, method_name, scoped)


def _owned_stages():
    # Import only in an explicitly enabled profiler worker, after model loading.
    from .authority import RequestAuthority
    from . import authority as authority_module
    from .capture_runtime import CaptureRuntime
    from .action_attention import ActionRuntime
    from .cached_storage import CachedArchive
    from .native_state import NativeRecurrentState
    from .session_state import PromptStateAdapter
    from .state import StateCheckpoints
    from .storage import AsyncArchive
    from .transfer import PackedDownloads, PackedTransfers
    from .window_assembly import NativeWindowAssembler
    from .window_batch import WindowBatchAdapter
    from .async_refresh import AsyncRefresh
    from . import async_restore
    from .maintenance import MaintenancePolicy
    from .prompt_checkpoints import PromptCheckpointPool
    from . import prompt_checkpoints
    from .memory import RequestMemory
    from .session_worker import SessionWorker
    from .action_attention import ActionHistory
    from .prefill_attention import PrefillHistory
    from . import _native
    return (
        (AsyncRefresh, 'step', 'refresh.step'),
        (async_restore, '_step_restore', 'restore.step'),
        (MaintenancePolicy, 'pump', 'maintenance.pump'),
        (PromptCheckpointPool, 'prepare_claim_window', 'session.prepare_claim_window'),
        (PromptCheckpointPool, 'capture', 'checkpoint.capture'),
        (PromptCheckpointPool, '_remove', 'checkpoint.remove'),
        (prompt_checkpoints, 'token_bytes', 'checkpoint.token_bytes'),
        (RequestMemory, 'checkpoint_window', 'checkpoint.window'),
        (RequestMemory, '_protect_checkpoint', 'checkpoint.protect'),
        (RequestMemory, 'checkpoint', 'checkpoint.prefix'),
        (RequestMemory, '_validate_window_checkpoint', 'checkpoint.validate'),
        (SessionWorker, 'detach', 'session.detach'),
        (SessionWorker, 'before_attention', 'session.before_attention'),
        (ActionHistory, 'snapshot', 'action.snapshot'),
        (ActionHistory, 'restore', 'action.restore'),
        (PrefillHistory, 'restore', 'prefill.restore'),
        (_native, 'finalize_mean_fp8', 'append.seal_finalize_fp8'),
        (_native, 'finalize_mean_bf16', 'append.seal_finalize_bf16'),
        (_native, 'scatter_history_batch', 'append.scatter'),
        (CaptureRuntime, 'submit', 'capture.submit'),
        (CaptureRuntime, '_capture_payload', 'capture.pack_fp8'),
        (CaptureRuntime, 'ensure_download_slot', 'capture.download_slot'),
        (CaptureRuntime, 'advance_publication', 'capture.advance_publication'),
        (CaptureRuntime, 'drain', 'capture.drain'),
        (CaptureRuntime, 'drain_requests', 'capture.drain_requests'),
        (CaptureRuntime, '_publish_claim', 'capture.publish_claim'),
        (CaptureRuntime, '_copy_source_owners', 'capture.copy_source_owners'),
        (CaptureRuntime, '_accept_query_input', 'capture.query_history'),
        (CaptureRuntime, 'finish', 'capture.finish'),
        (CaptureRuntime, 'join_cpu_scatter', 'capture.join_cpu_scatter'),
        (CaptureRuntime, '_write_progress', 'capture.progress'),
        (ActionRuntime, 'prepare', 'action.prepare'),
        (ActionRuntime, 'capture', 'action.capture'),
        (ActionRuntime, 'accept', 'action.accept'),
        (RequestAuthority, 'append', 'authority.append'),
        (authority_module, 'append_authority_batch', 'authority.append_batch'),
        (RequestAuthority, '_prepare_append', 'authority.prepare_append'),
        (RequestAuthority, '_commit_prepared_append', 'authority.commit_append'),
        (RequestAuthority, 'flush', 'authority.flush'),
        (RequestAuthority, 'select_for_queries', 'authority.select'),
        (WindowBatchAdapter, 'prepare', 'window.prepare'),
        (NativeWindowAssembler, 'assemble', 'window.assemble'),
        (NativeRecurrentState, 'borrow', 'recurrent.borrow'),
        (NativeRecurrentState, 'canonicalize_accepted', 'recurrent.canonicalize'),
        (StateCheckpoints, 'capture', 'recurrent.checkpoint'),
        (StateCheckpoints, 'restore', 'recurrent.rollback'),
        (PromptStateAdapter, 'capture', 'session.capture'),
        (PromptStateAdapter, 'restore', 'session.restore'),
        (PackedDownloads, 'download', 'transfer.download'),
        (PackedTransfers, 'upload_from_reader', 'transfer.upload'),
        (AsyncArchive, 'write', 'archive.write'),
        (CachedArchive, '_cache_write', 'archive.cache_write'),
        (CachedArchive, 'read', 'archive.read'),
    )


def install_profile_scopes(runner):
    config = getattr(getattr(runner, 'vllm_config', None), 'profiler_config', None)
    if getattr(config, 'profiler', None) != 'torch':
        return
    for method, stage in (
        ('execute_model', 'runner.execute'),
        ('sample_tokens', 'runner.sample_tokens'),
        ('prepare_attn', 'runner.prepare_attn'),
        ('sample', 'runner.sample'),
    ):
        _wrap(runner, method, stage)
    manager = getattr(runner, 'cudagraph_manager', None)
    # initialize_graph_window preserves the previous replay method underneath
    # its counter wrapper. If that method is already scoped, a second install
    # must not introduce a duplicate nested target.graph range.
    if not _wrapped_for(getattr(manager, '_kvmem_original_fullgraph', None), 'target.graph'):
        _wrap(manager, 'run_fullgraph', 'target.graph')
    speculator = getattr(runner, 'speculator', None)
    _wrap(speculator, 'propose', 'draft.propose')
    _wrap(getattr(speculator, 'model', None), 'precompute_and_store_context_kv', 'draft.context_kv')
    _wrap(getattr(speculator, 'query_cudagraph_manager', None), 'run_fullgraph', 'draft.graph')
    _wrap(getattr(speculator, 'prefill_cudagraph_manager', None), 'run_fullgraph', 'mtp.prefill_graph')
    _wrap(getattr(speculator, 'decode_cudagraph_manager', None), 'run_fullgraph', 'mtp.decode_graph')
    for owner, method, stage in _owned_stages():
        _wrap(owner, method, stage)
