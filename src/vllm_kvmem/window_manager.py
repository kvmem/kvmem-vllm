"""Bounded full-attention allocation adapter for the native vLLM coordinator.

Composition preserves the native manager/pool and all its allocation bookkeeping.
Only full-attention token extents are translated. Recurrent managers and Request
logical counters are untouched. Scheduler wiring must suspend a pending request
and send complete replacement block-table rows after commit.
"""
from dataclasses import dataclass
from itertools import count


class RefreshPending(RuntimeError):
    pass


class RefreshRequired(RuntimeError):
    pass


_ticket_ids = count(1)


@dataclass(frozen=True)
class RefreshTicket:
    request_id: str
    ticket_id: int
    history_end: int
    window_tokens: int
    original_block_ids: tuple[int, ...]
    retained_block_ids: tuple[int, ...]
    replay_tokens: int = 0


class WorkingSetManager:
    def __init__(self, native, select_budget: int, gen_budget: int):
        from .integration import check_engine_version
        check_engine_version()
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
        if type(native) is not FullAttentionManager:
            raise ValueError("Only native full-attention groups may use this adapter")
        if native.enable_caching or native.dcp_world_size != 1 or native.pcp_world_size != 1:
            raise ValueError("First integration requires prefix caching off and single context-parallel rank")
        if min(select_budget, gen_budget) <= 0:
            raise ValueError("K/G must be positive")
        self.native = native
        self.select_budget = select_budget
        self.capacity = select_budget + gen_budget
        self.offsets: dict[str, int] = {}
        self.pending: dict[str, RefreshTicket] = {}
        self.imports: dict[str, tuple[int, int]] = {}
        self.resident_imports: dict[str, tuple] = {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _local(self, request_id: str, tokens: int, *, admission=False):
        if request_id in self.pending:
            raise RefreshPending(request_id)
        value = tokens - self.offsets.get(request_id, 0)
        if value < 0:
            raise ValueError("Logical position predates committed working set; restore a checkpoint first")
        if admission:
            return min(value, self.capacity)
        if value > self.capacity:
            raise RefreshRequired(request_id)
        return value

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                                   total_computed_tokens, num_local_computed_tokens,
                                   num_tokens_main_model, apply_admission_cap=False, **kwargs):
        if new_computed_blocks:
            raise ValueError("Working-set manager cannot consume ordinary prefix-cache hits")
        external = total_computed_tokens - num_local_computed_tokens
        if external:
            self._external_extent(request_id, num_local_computed_tokens, external)
        # A fresh connector import has no locally computed prefix. Its zero is
        # an ownership count, not a logical position to subtract the offset from.
        local_computed = (0 if external else
                          self._local(request_id, num_local_computed_tokens,
                                      admission=apply_admission_cap))
        required = self.native.get_num_blocks_to_allocate(
            request_id, self._local(request_id, num_tokens, admission=apply_admission_cap),
            new_computed_blocks,
            self._local(request_id, total_computed_tokens, admission=apply_admission_cap),
            local_computed,
            self._local(request_id, num_tokens_main_model, admission=apply_admission_cap),
            apply_admission_cap=apply_admission_cap,
            **kwargs,
        )
        return max(0, required - len(self.resident_imports.get(request_id, ())))

    def stage_resident_import(self, request_id, history, window, blocks):
        if (self.imports.get(request_id) != (history, window) or
                self.req_to_blocks.get(request_id) or request_id in self.resident_imports or
                len(blocks) != (window + self.block_size - 1) // self.block_size or
                any(b.is_null or b.ref_cnt != 1 for b in blocks)):
            raise ValueError('Resident import requires an exact exclusive idle view')
        self.resident_imports[request_id] = tuple(blocks)

    def prepare_import(self, request_id, history_end, window_tokens):
        """Reserve the exact checkpoint frame before connector allocation.

        This installs positions only, never a claim that pages contain KV. The
        caller must restore target/GDN/draft state and acknowledge its fence
        before committing the import or running another query. Repeated lookup
        before allocation may prepare the same immutable checkpoint again.
        """
        if (any(type(v) is not int for v in (history_end, window_tokens)) or
                not 0 < window_tokens <= min(history_end, self.capacity)):
            raise ValueError('Invalid imported checkpoint extent')
        value = history_end, window_tokens
        if request_id in self.imports:
            if self.imports[request_id] != value:
                raise ValueError('Checkpoint changed while import was pending')
            return
        if (request_id in self.pending or request_id in self.offsets or
                self.native.req_to_blocks.get(request_id)):
            raise ValueError('Import requires a fresh native request allocation')
        self.imports[request_id] = value
        self.offsets[request_id] = history_end - window_tokens

    def _external_extent(self, request_id, local, external):
        checkpoint = self.imports.get(request_id)
        if local != 0 or checkpoint is None or external != checkpoint[0]:
            raise ValueError('External KV requires an exact prepared checkpoint import')
        return checkpoint[1]

    def add_local_computed_blocks(self, request_id, new_computed_blocks,
                                  num_local_computed_tokens, num_external_computed_tokens):
        if new_computed_blocks:
            raise ValueError('Working-set manager cannot consume ordinary prefix-cache hits')
        external = (self._external_extent(request_id, num_local_computed_tokens,
                                         num_external_computed_tokens)
                    if num_external_computed_tokens else 0)
        result = self.native.add_local_computed_blocks(request_id, new_computed_blocks,
                                                       num_local_computed_tokens, external)
        resident = self.resident_imports.pop(request_id, ())
        if resident:
            # Transfer the idle lease's reference. These pages are not newly
            # allocated and must not enter the native block-zeroing list.
            self.native.req_to_blocks[request_id].extend(resident)
        return result

    def allocate_external_computed_blocks(self, request_id, num_local_computed_tokens,
                                          num_external_computed_tokens, **kwargs):
        external = (self._external_extent(request_id, num_local_computed_tokens,
                                         num_external_computed_tokens)
                    if num_external_computed_tokens else 0)
        return self.native.allocate_external_computed_blocks(request_id,
                                                             num_local_computed_tokens, external, **kwargs)

    def validate_import(self, request_id, history_end):
        checkpoint = self.imports.get(request_id)
        if checkpoint is None or checkpoint[0] != history_end:
            raise ValueError('Stale or foreign checkpoint import completion')
        if len(self.native.req_to_blocks.get(request_id, ())) * self.native.block_size < checkpoint[1]:
            raise ValueError('Imported checkpoint has no allocated target frame')

    def commit_import(self, request_id, history_end, completion):
        self.validate_import(request_id, history_end)
        if not completion.query():
            return False
        del self.imports[request_id]
        return True

    def abort_import_preparation(self, request_id):
        if self.native.req_to_blocks.get(request_id):
            raise ValueError('Allocated import requires fenced native request cleanup')
        del self.imports[request_id]
        del self.offsets[request_id]

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        return self.native.allocate_new_blocks(request_id, self._local(request_id, num_tokens),
                                               self._local(request_id, num_tokens_main_model))

    def prepare_refresh(self, request_id: str, history_end: int, window_tokens: int, *, replay_tokens=0,
                        allow_import=False):
        if request_id in self.pending or (request_id in self.imports and not allow_import):
            raise RefreshPending(request_id)
        old_window = self._local(request_id, history_end)
        if not 0 < window_tokens <= min(old_window, self.select_budget):
            raise ValueError("Refresh must select a nonempty subset within K")
        if type(replay_tokens) is not int or replay_tokens < 0 or old_window + replay_tokens > self.capacity:
            raise ValueError("Query probe exceeds the admitted old working window")
        blocks = self.native.req_to_blocks.get(request_id)
        if not blocks or old_window + replay_tokens > len(blocks) * self.native.block_size:
            raise ValueError("Refresh references unallocated history")
        # The worker can replay the scheduled query before acknowledging. Keep
        # its destination pages as well as the selected historical prefix.
        count_kept = (window_tokens + replay_tokens + self.native.block_size - 1) // self.native.block_size
        ticket = RefreshTicket(request_id, next(_ticket_ids), history_end, window_tokens,
                               tuple(b.block_id for b in blocks),
                               tuple(b.block_id for b in blocks[:count_kept]), replay_tokens)
        self.pending[request_id] = ticket
        return ticket

    def validate_refresh(self, ticket: RefreshTicket, selected_tokens=None):
        if self.pending.get(ticket.request_id) is not ticket:
            raise ValueError("Stale or foreign refresh completion")
        blocks = self.native.req_to_blocks[ticket.request_id]
        if tuple(b.block_id for b in blocks) != ticket.original_block_ids:
            raise RuntimeError("Native allocation changed while refresh was pending")
        if selected_tokens is not None and (type(selected_tokens) is not int or
                                           not 0 < selected_tokens <= ticket.window_tokens):
            raise ValueError("Worker selected extent exceeds the reserved historical prefix")

    def abort_preparation(self, ticket: RefreshTicket):
        """Undo preparation only before a worker command has been published."""
        self.validate_refresh(ticket)
        del self.pending[ticket.request_id]

    def commit_refresh(self, ticket: RefreshTicket, completion, selected_tokens=None) -> bool:
        self.validate_refresh(ticket, selected_tokens)
        if not completion.query():
            return False
        blocks = self.native.req_to_blocks[ticket.request_id]
        selected_tokens = ticket.window_tokens if selected_tokens is None else selected_tokens
        keep = (selected_tokens + ticket.replay_tokens + self.native.block_size - 1) // self.native.block_size
        removed = blocks[keep:]
        del blocks[keep:]
        self.native.block_pool.free_blocks(reversed(removed))
        self.offsets[ticket.request_id] = ticket.history_end - selected_tokens
        del self.pending[ticket.request_id]
        return True

    def pop_blocks_for_free(self, request_id: str):
        if request_id in self.pending:
            raise RefreshPending("Drain refresh before cancelling/freeing request pages")
        self.offsets.pop(request_id, None)
        self.imports.pop(request_id, None)
        return self.native.pop_blocks_for_free(request_id)

    def free(self, request_id: str):
        blocks = self.pop_blocks_for_free(request_id)
        self.native.block_pool.free_blocks(reversed(blocks))
