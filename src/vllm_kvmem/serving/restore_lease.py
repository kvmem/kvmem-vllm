"""CPU allocation witness for a fresh, suspended checkpoint import only.

Native block tables have no persistent CPU mirror. This witness records the
exact new-request allocation which the native runner stages onto the GPU.
It must be discarded before that request can receive any later table update.
It is not a replacement for generic live-request page-table validation.
"""
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def verify_restore_allocation_sources():
    from ..adapters.engine_api import require_contract
    require_contract('restore')


def _blocks(block_ids, sliding=()):
    """Distinct non-null pages per group; a sliding-window group (``sliding``) may
    hold null pages (0) for positions outside its window, as vLLM allocates them."""
    result = tuple(tuple(group) for group in block_ids)
    for index, group in enumerate(result):
        live = [b for b in group if b != 0] if index in sliding else list(group)
        if (not group or not live or any(type(b) is not int or b < 0 for b in group) or
                any(b <= 0 for b in live) or len(set(live)) != len(live)):
            raise ValueError('Restore allocation requires nonempty distinct non-null native pages')
    if not result:
        raise ValueError('Restore allocation requires nonempty distinct non-null native pages')
    return result


def sliding_groups(runner):
    """Opt-in KVMEM_DFLASH_ASYNC_REFRESH: indices of sliding-window KV groups (DFlash2's
    independent draft pages); empty otherwise, keeping the strict all-non-null rule."""
    from ..core import settings as _settings
    if _settings.get('KVMEM_DFLASH_ASYNC_REFRESH') != '1':
        return ()
    from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
    result = []
    for index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if specs and all(isinstance(s, SlidingWindowSpec) for s in specs):
            result.append(index)
    return tuple(result)


@dataclass(frozen=True)
class FreshRestoreLease:
    runner: object
    request_id: str
    index: int
    generation: int
    block_ids: tuple[tuple[int, ...], ...]
    rows: tuple[tuple[int, ...], ...]
    sliding: tuple[int, ...] = ()

    @classmethod
    def capture(cls, runner, request_id, block_ids):
        """Call after native add_requests and assigning its new generation.

        The caller registers the result in ``_kvmem_restore_allocations`` and
        holds every native page/index until the restore completion fence.
        """
        verify_restore_allocation_sources()
        sliding = sliding_groups(runner)
        blocks = _blocks(block_ids, sliding)
        from ..core.native_pages import kernel_expansions
        expansions = kernel_expansions(runner)
        if len(blocks) != len(expansions) or any(type(n) is not int or n <= 0 for n in expansions):
            raise ValueError('Restore allocation omitted native cache groups')
        rows = tuple(tuple(b * n + i for b in group for i in range(n))
                     for group, n in zip(blocks, expansions))
        result = cls(runner, request_id, runner.req_states.req_id_to_index[request_id],
                     runner._kvmem_request_generations[request_id], blocks, rows, sliding)
        result._validate(runner, registered=False)
        return result

    def _validate(self, runner, *, registered):
        if (runner is not self.runner or
                runner.req_states.req_id_to_index.get(self.request_id) != self.index or
                runner._kvmem_request_generations.get(self.request_id) != self.generation):
            raise ValueError('Fresh restore allocation request/index was recycled')
        if registered and getattr(runner, '_kvmem_restore_allocations', {}).get(self.request_id) is not self:
            raise ValueError('Fresh restore allocation witness is no longer leased')
        tables = runner.block_tables
        if len(tables.block_tables) != len(self.rows):
            raise ValueError('Fresh restore allocation cache groups changed')
        for group, rows in enumerate(self.rows):
            table = tables.block_tables[group].gpu
            from ..core.native_pages import stored_expansions
            stored_count = len(self.block_ids[group]) * stored_expansions(tables)[group]
            if (int(tables.num_blocks.np[group, self.index]) != stored_count or
                    stored_count > table.shape[1]):
                raise ValueError('Fresh restore allocation page count changed')

    def validate(self, runner, *, expected_block_ids=None):
        self._validate(runner, registered=True)
        if expected_block_ids is not None and _blocks(expected_block_ids, self.sliding) != self.block_ids:
            raise ValueError('Restore command differs from its native allocation witness')

    def group_rows(self, group, *, page_count=None):
        rows = self.rows[group]
        if page_count is not None and max(rows) >= page_count:
            raise ValueError('Restore page exceeds the native cache arena')
        return rows
