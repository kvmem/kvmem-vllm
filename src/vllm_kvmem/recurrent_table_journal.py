"""Journal the exact native CPU writes submitted to recurrent GPU page tables.

This is not a presumed mirror: only a witnessed complete overwrite can seed a
row, subsequent native appends extend it, and GPU submission commits it. Reads
before submission are forbidden. Unknown rows use the existing GPU read path.
"""
from .native_pages import stored_expansions
import torch


class RecurrentTableJournal:
    def __init__(self, runner, groups):
        self.runner, self.tables, self.groups = runner, runner.block_tables, tuple(groups)
        self.committed, self.pending = {}, {}
        self.failed = False
        self.tensor_ids = {gid: self.tables.block_tables[gid].gpu.data_ptr() for gid in self.groups}
        append = self.tables.append_block_ids
        apply = self.tables.apply_staged_writes
        self.native_append, self.native_apply = append, apply

        def append_ids(req_index, new_block_ids, overwrite):
            proposed = {}
            for gid in self.groups:
                key = (req_index, gid)
                old = self.pending.get(key, self.committed.get(key))
                start = int(self.tables.num_blocks.np[gid, req_index])
                expansion = stored_expansions(self.tables)[gid]
                incoming = tuple(b * expansion + i for b in new_block_ids[gid] for i in range(expansion))
                if overwrite:
                    proposed[key] = (incoming, None)
                elif old is not None and len(old[0]) == start:
                    proposed[key] = (old[0] + incoming, None)
                else:
                    proposed[key] = None
            try:
                result = append(req_index, new_block_ids, overwrite)
            except BaseException:
                self.failed = True
                raise
            self.pending.update(proposed)
            return result

        def apply_writes():
            try:
                result = apply()
            except BaseException:
                self.failed = True
                raise
            if not self.pending:
                return result
            # Native writes and all borrowers execute on this same stream.
            # Recording its identity requires no device synchronization.
            identities = {index: (rid, self.runner._kvmem_request_generations.get(rid))
                          for rid, index in self.runner.req_states.req_id_to_index.items()}
            for key, value in self.pending.items():
                if value is None:
                    self.committed.pop(key, None)
                else:
                    self.committed[key] = (value[0], self._stream(key[1]), identities.get(key[0]))
            self.pending.clear()
            return result

        self.append_ids, self.apply_writes = append_ids, apply_writes
        self.tables.append_block_ids = append_ids
        self.tables.apply_staged_writes = apply_writes

    def _stream(self, group):
        device = self.tables.block_tables[group].gpu.device
        return torch.cuda.current_stream(device).cuda_stream if device.type == 'cuda' else None

    def rows(self, request_id, group):
        if (self.failed or self.runner.block_tables is not self.tables or
                self.tables.append_block_ids is not self.append_ids or
                self.tables.apply_staged_writes is not self.apply_writes):
            raise RuntimeError('Recurrent allocation journal lost its native write witness')
        if group not in self.groups:
            return None
        if self.tables.block_tables[group].gpu.data_ptr() != self.tensor_ids[group]:
            raise RuntimeError('Recurrent page table storage changed outside the native write journal')
        index = self.runner.req_states.req_id_to_index[request_id]
        if request_id not in self.runner._kvmem_request_generations:
            raise ValueError('Recurrent allocation has no live request generation')
        key = (index, group)
        if key in self.pending:
            raise RuntimeError('Recurrent page table has unsubmitted native writes')
        row = self.committed.get(key)
        if row is None:
            return None
        rows, stream, identity = row
        if identity != (request_id, self.runner._kvmem_request_generations[request_id]):
            raise ValueError('Recurrent request incarnation changed without a witnessed allocation')
        if len(rows) != int(self.tables.num_blocks.np[group, index]):
            raise RuntimeError('Recurrent page count changed outside the native write journal')
        if stream != self._stream(group):
            # A CPU witness does not establish execution order across streams.
            # Callers outside the qualified engine stream retain the old read.
            return None
        return rows


def _assert_startup_without_requests(runner):
    """Only native startup may replace the cache, never a drained serving run."""
    if (getattr(runner, '_kvmem_next_generation', 0) or
            runner.req_states.req_id_to_index or
            getattr(runner, '_kvmem_request_generations', {}) or
            getattr(runner, '_kvmem_restore_allocations', {}) or
            any(getattr(runner, name, None) is not None for name in (
                '_kvmem_capture_runtime', '_kvmem_pending_capture', '_kvmem_saved_capture'))):
        raise RuntimeError('Recurrent cache reinitialization requires startup without request state')
    sessions = getattr(runner, '_kvmem_sessions', None)
    if sessions is not None:
        if (sessions.model_adapter is not None or any(getattr(sessions, name) for name in (
                'bindings', 'identities', 'imports', 'captures', 'retiring', 'evictions',
                'completed', 'terminal_captures', 'restore_oracles')) or
                any(getattr(sessions.pool, name) for name in (
                    'entries', 'prompts', '_discarded_captures', 'payload_bytes'))):
            raise RuntimeError('Recurrent cache reinitialization cannot discard session state')
    refresh = getattr(runner, '_kvmem_async_refresh', None)
    if refresh is not None and (refresh.jobs or refresh.deferred or refresh.adapter is not None):
        raise RuntimeError('Recurrent cache reinitialization cannot discard maintenance state')


def install_recurrent_table_journal(runner, *, is_profiling=False):
    from .checkpoint_timing import async_checkpoint_capture
    if not async_checkpoint_capture(runner):
        return
    existing = getattr(runner, '_kvmem_recurrent_table_journal', None)
    if is_profiling:
        # Native graph-memory profiling builds a temporary cache, synchronizes
        # and tears it down, then initialize_kv_cache builds the serving cache.
        # Dummy forwards have no request allocations to witness. In particular,
        # do not retain that temporary GPU table beyond native teardown.
        _assert_startup_without_requests(runner)
    if existing is not None:
        if existing.tables is not runner.block_tables:
            _assert_startup_without_requests(runner)
            if (existing.failed or existing.pending or existing.committed or
                    existing.tables.append_block_ids is not existing.append_ids or
                    existing.tables.apply_staged_writes is not existing.apply_writes):
                raise RuntimeError('Recurrent cache reinitialization cannot discard a used or failed journal')
            # Break the old table -> wrappers -> journal cycle and invalidate
            # external borrowers. No native submission or live row is discarded.
            existing.tables.append_block_ids = existing.native_append
            existing.tables.apply_staged_writes = existing.native_apply
            existing.failed = True
            del runner._kvmem_recurrent_table_journal
        else:
            return
    if is_profiling:
        return
    from .restore_lease import verify_restore_allocation_sources
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
    verify_restore_allocation_sources()
    groups = []
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        spec = spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
        if isinstance(spec, MambaSpec):
            groups.append(gid)
    runner._kvmem_recurrent_table_journal = RecurrentTableJournal(runner, groups)
