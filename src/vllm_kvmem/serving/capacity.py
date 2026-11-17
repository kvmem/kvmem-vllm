"""Reserve each admitted target request's future native GPU-page growth.

The native allocator still owns every physical block. Reservations are credits,
not extra allocations: trimming a window returns physical pages while retaining
that request's right to grow again. The optional qualified draft contract adds
native SWA and speculative GDN credits.
Execution remains synchronous, APC-off, TP=1.
"""
import os
from .window_manager import WorkingSetManager
from ..core import settings as _settings


class ReservedKVCacheManager:
    def __init__(self, native, vllm_config, *, draft_contract=None, max_authorities=None, selector_carve=None,
                 streaming_carve=None):
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SlidingWindowManager
        from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

        from ..adapters.engine_api import require_contract
        require_contract('capacity')
        if ((vllm_config.speculative_config is not None) != (draft_contract is not None) or
                vllm_config.cache_config.enable_prefix_caching or
                vllm_config.cache_config.mamba_cache_mode != "none" or
                vllm_config.max_concurrent_batches != 1):
            raise ValueError("Capacity reservation requires synchronous APC-off execution and an explicit draft contract")
        self.native = native
        if max_authorities is not None and (type(max_authorities) is not int or max_authorities <= 0):
            raise ValueError('Authority admission requires a positive request count')
        self.max_authorities = max_authorities
        self.authority_rejections = 0
        # QSA carves (core/compressed_pages.py): the last blocks of the fresh pool,
        # exactly the range the worker bound views over. Taken once, never freed.
        self.selector_carve, self.streaming_carve = selector_carve, streaming_carve
        self.selector_blocks = ()
        if selector_carve is not None:
            from ..core.compressed_pages import carve_range
            if streaming_carve is not None and streaming_carve.selector != selector_carve:
                raise ValueError('QSA streaming pages require the matching selector carve')
            first, end = carve_range(selector_carve, streaming_carve)
            pool = native.block_pool
            if end != getattr(pool, 'num_gpu_blocks', end):
                raise ValueError('QSA carves must end at the native pool end')
            taken = pool.get_new_blocks(pool.get_num_free_blocks())
            carved = [block for block in taken if first <= block.block_id < end]
            rest = [block for block in taken if not first <= block.block_id < end]
            pool.free_blocks(rest)             # original free order, carves removed
            if sorted(block.block_id for block in carved) != list(range(first, end)):
                pool.free_blocks(carved)
                raise ValueError('QSA carves require the last blocks of a fresh pool')
            self.selector_blocks = tuple(sorted(carved, key=lambda block: block.block_id))
        self.lookahead = draft_contract.query_tokens if draft_contract is not None else 0
        self.guided_reserve = 0
        self.groups = tuple(native.coordinator.single_type_managers)
        self.static_blocks = {}
        for index, manager in enumerate(self.groups):
            if manager.block_pool is not native.block_pool:
                raise ValueError("Capacity reservation requires the native shared page pool")
            if isinstance(manager, WorkingSetManager):
                continue
            spec = manager.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.first_spec
            if type(manager) is SlidingWindowManager and draft_contract is not None:
                if type(spec) is not SlidingWindowSpec or spec.sliding_window != draft_contract.sliding_window:
                    raise ValueError("Unqualified draft sliding-window reservation")
                # Native recycling happens before allocation. Include an extra
                # drafter lookahead beyond the target in-flight token budget.
                cap = spec.max_admission_blocks_per_request(
                    vllm_config.max_in_flight_tokens + self.lookahead, native.max_model_len)
                self.static_blocks[index] = cap
                continue
            if (type(manager) is not MambaManager or type(spec) is not MambaSpec or
                    spec.num_speculative_blocks != (self.lookahead - 1 if self.lookahead else 0) or
                    spec.num_prefill_checkpoint_blocks or spec.mamba_cache_mode != "none" or
                    spec.block_size < native.max_model_len):
                raise ValueError("Unqualified recurrent-state reservation geometry")
            self.static_blocks[index] = spec.max_num_blocks_per_req(vllm_config, native.max_model_len)
        self.reservations = {}
        self._step_remaining = None
        self._step_total = 0
        self.rejections = 0
        self.native_allocation_rejections = 0
        self.active_residency = None
        self.resident_entry = None
        self.resident_restores = set()
        self.restore_admission = None
        self.prefill_refresh_control = None
        self.refresh_control_allocations = {}
        if self._demand(native.max_model_len) > native.block_pool.get_num_free_blocks():
            raise ValueError("GPU page pool cannot fit one worst-case KVMem request")

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _demand(self, tokens):
        return sum(self.static_blocks[index] if index in self.static_blocks else
                   ((min(tokens, manager.capacity) if manager.physical_tokens is None else
                     # qsa_streaming: the full page set is allocated with the first token.
                     (manager.physical_tokens if tokens > 0 else 0))
                    + manager.block_size - 1) // manager.block_size
                   for index, manager in enumerate(self.groups))

    def _owned(self, request_id):
        # APC is disabled: live pages are exclusive and null pages own no credit.
        return sum(not block.is_null for manager in self.groups
                   for block in manager.req_to_blocks.get(request_id, ()))

    def _remaining(self, request_id, demand):
        remaining = demand - self._owned(request_id)
        if remaining < 0:
            raise RuntimeError("Native request allocation exceeds its admitted lifetime bound")
        return remaining

    def snapshot(self):
        """Explain physical ownership separately from future growth credits.

        Call on the scheduler thread between allocator mutations. Counts use
        native pool pages, not archive blocks or inferred KV bytes. The pool
        contains one unavailable null page. Idle/request overlap during fenced
        ownership transfer is counted once; detached frees and other owners
        appear in the residual, which is not advertised as available capacity.
        This observation neither expires idle views nor changes admission.
        """
        requests, group_rows, all_owned, reserved_owned, staged = {}, [], set(), set(), set()
        for index, manager in enumerate(self.groups):
            group_owned = set()
            for request_id, blocks in manager.req_to_blocks.items():
                pages = {block.block_id for block in blocks if not block.is_null}
                group_owned.update(pages)
                all_owned.update(pages)
                if request_id in self.reservations:
                    reserved_owned.update(pages)
            for blocks in getattr(manager, 'resident_imports', {}).values():
                staged.update(block.block_id for block in blocks if not block.is_null)
            group_rows.append(dict(group=index, block_tokens=manager.block_size,
                owned_pages=len(group_owned),
                kind='attention' if isinstance(manager, WorkingSetManager) else type(manager).__name__,
                static_pages_per_request=self.static_blocks.get(index, 0)))
        for request_id, (_, quota) in self.reservations.items():
            owned = self._owned(request_id)
            requests[request_id] = dict(owned_pages=owned, lifetime_quota_pages=quota,
                                       future_reserved_pages=quota-owned)
        residency = getattr(self, 'active_residency', None)
        idle = {block.block_id for view in residency.entries.values()
                for block in view.blocks if not block.is_null} if residency is not None else set()
        idle_only = idle - all_owned
        staged_only = staged - all_owned - idle
        pool = self.native.block_pool
        total = pool.num_gpu_blocks - 1
        free = pool.get_num_free_blocks()
        allocated = total - free
        future = sum(row['future_reserved_pages'] for row in requests.values())
        return dict(mode='lifetime', unit='native_pages', null_pages=1,
            physical_total_pages=total, physical_free_pages=free,
            physical_allocated_pages=allocated, request_owned_pages=len(all_owned),
            reserved_request_owned_pages=len(reserved_owned),
            idle_resident_pages=len(idle_only), idle_request_overlap_pages=len(idle & all_owned),
            staged_import_pages=len(staged_only),
            selector_carve_pages=len(self.selector_blocks),
            deferred_or_other_pages=(allocated-len(all_owned)-len(idle_only)-len(staged_only)
                                     -len(self.selector_blocks)),
            lifetime_quota_pages=sum(row['lifetime_quota_pages'] for row in requests.values()),
            future_reserved_pages=future, free_after_growth_reserve_pages=free-future,
            requests=requests, groups=group_rows,
            rejections=dict(lifetime_budget=self.rejections-self.authority_rejections,
                retirement_or_active_limit=self.authority_rejections,
                native_allocation=self.native_allocation_rejections))

    def begin_step(self):
        """Reconcile trims/imports once, before native synchronous scheduling.

        During this step native allocation changes only its current request;
        free and deferred-free operations below update the same credit ledger.
        Acknowledged window trims happen between scheduling steps. Standalone
        callers retain the conservative scan path outside this explicit scope.
        """
        if self._step_remaining is not None:
            raise RuntimeError('Capacity credit step already open')
        if self.refresh_control_allocations:
            raise RuntimeError('Unverified refresh control allocation')
        remaining = {key:self._remaining(key, quota)
                     for key, (_, quota) in self.reservations.items()}
        self._step_remaining, self._step_total = remaining, sum(remaining.values())

    def end_step(self):
        self._step_remaining, self._step_total = None, 0

    def _update_credit(self, request_id, demand):
        if self._step_remaining is not None:
            remaining = self._remaining(request_id, demand)
            self._step_total += remaining - self._step_remaining.get(request_id, 0)
            self._step_remaining[request_id] = remaining

    def _forget_credit(self, request_id):
        if self._step_remaining is not None:
            self._step_total -= self._step_remaining.pop(request_id, 0)

    def validate_refresh_controls(self, scheduled, transactions):
        command_inputs = lambda command: getattr(command, 'input_tokens', ()) if command is not None else ()
        """A no-lookahead frame must never reach a public model forward."""
        for rid, (request, history, count) in self.refresh_control_allocations.items():
            transaction = transactions.get(rid)
            command = transaction[0] if transaction is not None else None
            # Opt-in zero-replay prefill pressure: the boundary block is not
            # replayed; the refresh window rebuild owns and releases its pages.
            replayed = (0 if _settings.get('KVMEM_PREFILL_PRESSURE_NO_REPLAY', '0') == '1' and
                        not command_inputs(command) and getattr(command, 'pressure_stage', None) == 'prefill'
                        else count)
            if (rid in scheduled or command is None or not command.asynchronous or
                    command.selection != 'pressure' or command.query_span is not None or
                    command.speculative_padding or len(command.input_tokens) != replayed or
                    request.num_computed_tokens != history or request.num_in_flight_tokens or
                    any(ticket.history_end != history or ticket.replay_tokens != replayed
                        for _, ticket in command.groups)):
                raise RuntimeError('Refresh control allocation escaped its private replay transaction')
        self.refresh_control_allocations.clear()

    def admission_preview(self, request):
        """Conservative read-only lifetime preview, including revocable idle pages.

        Actual native allocation and the existing credit gate remain decisive.
        Idle pages can either be adopted or freed, so count them once as capacity.
        """
        rid = request.request_id
        if rid not in self.reservations and self.max_authorities is not None and len(self.reservations) >= self.max_authorities:
            return 'retirement_or_active_limit'
        tokens = min(self.native.max_model_len, request.num_prompt_tokens + request.max_tokens +
                     2 * self.lookahead + getattr(self, 'guided_reserve', 0))
        demand = self._demand(tokens)
        required = self._remaining(rid, demand) + sum(
            self._remaining(key, quota) for key, (_, quota) in self.reservations.items() if key != rid)
        residency = getattr(self, 'active_residency', None)
        revocable = sum(len(v.blocks) for v in residency.entries.values()) if residency is not None else 0
        return 'gpu_lifetime_budget' if required > self.block_pool.get_num_free_blocks() + revocable else None

    def tool_grace_allows(self, request):
        """Skip a cold waiter before native allocation, so hot peers can pass.

        Reserve all admitted requests' future growth. A grace period delays
        only a new admission that would need protected idle attention pages.
        Native allocation remains authoritative and running growth never waits.
        """
        residency = self.active_residency
        policy = residency.tool_grace if residency is not None else None
        rid = request.request_id
        if policy is None or rid in self.reservations:
            return True
        residency.expire()
        arrival = policy.arrivals.get(rid)
        if arrival is None or arrival.namespace in residency.entries:
            return True
        protected = [v for v in residency.entries.values() if policy.protected(v)]
        if not protected:
            return True
        tokens = min(self.native.max_model_len, request.num_prompt_tokens + request.max_tokens +
                     2 * self.lookahead + getattr(self, 'guided_reserve', 0))
        needed = self._remaining(rid, self._demand(tokens)) + sum(
            self._remaining(key, quota) for key, (_, quota) in self.reservations.items() if key != rid)
        available = self.block_pool.get_num_free_blocks() + sum(
            len(v.blocks) for v in residency.entries.values() if not policy.protected(v))
        return needed <= available or not policy.defer(rid)

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        gate = self.restore_admission
        permit = (gate.before_allocate(request, kwargs['num_external_computed_tokens'])
                  if gate is not None and kwargs.get('num_external_computed_tokens', 0) else None)
        try:
            result = self._allocate_slots(request, num_new_tokens, *args, **kwargs)
        except BaseException:
            if permit is not None:
                if any(m.req_to_blocks.get(request.request_id) for m in self.groups):
                    # Partial native mutation is fatal. Retain the permit and
                    # pages; never turn uncertain DMA destinations into idle KV.
                    gate.retain_failed_allocation(request)
                else:
                    gate.rollback(request)
            raise
        if permit is not None:
            if result is None:
                gate.rollback(request)
            else:
                gate.allocated(request)
        return result

    def _guided_allocation_options(self, request, num_new_tokens, kwargs):
        request_id = request.request_id
        if getattr(self,'guided_reserve',0):
            from .request_plan import request_plan
            # The public scheduler still advances only its original token count.
            # Extra lookahead pages belong to this synchronous private probe.
            manager = next(m for m in self.groups if isinstance(m,WorkingSetManager))
            plan = request_plan(request, self.guided_block_tokens,manager.capacity-manager.select_budget)
            at = request.num_computed_tokens + kwargs.get('num_external_computed_tokens',0)
            local_end = at-manager.offsets.get(request_id,0)+num_new_tokens
            control = getattr(self, 'guided_refresh_control', None)
            guided = control(request, at, manager.offsets.get(request_id, 0)) if control else None
            pressure_private = (plan is not None and plan.pressure_probe_tokens and
                at < request.num_prompt_tokens and plan.action(at, request.num_prompt_tokens,
                    manager.offsets.get(request_id, 0), manager.capacity) == 'pressure')
            if (plan is not None and (plan.query is not None and at == plan.replay_begin or
                    guided is not None and guided[2] or pressure_private)
                    and local_end+self.guided_reserve<=manager.capacity):
                kwargs = {**kwargs,'num_lookahead_tokens':max(kwargs.get('num_lookahead_tokens',0),self.guided_reserve)}
        return kwargs

    def _prefill_allocation_options(self, request, num_new_tokens, args, kwargs, existing):
        request_id = request.request_id
        control = None
        lookahead = kwargs.get('num_lookahead_tokens', 0)
        if (existing is not None and lookahead and not args and
                not kwargs.get('num_new_computed_tokens', 0) and
                not kwargs.get('num_external_computed_tokens', 0) and
                not kwargs.get('new_computed_blocks') and
                self.prefill_refresh_control is not None):
            extents = [(request.num_computed_tokens - m.offsets.get(request_id, 0) + num_new_tokens,
                        m.capacity) for m in self.groups if isinstance(m, WorkingSetManager)]
            if (extents and all(0 <= end <= cap for end, cap in extents) and
                    any(end + lookahead > cap for end, cap in extents) and
                    self.prefill_refresh_control(request, num_new_tokens)):
                # Native allocation precedes connector metadata construction.
                # This exact pressure step is removed from the public batch;
                # its complete input is replayed privately after selecting K.
                # Reserve the replay frame, not draft rows that never execute.
                # Lifetime credits remain unchanged; the next public forward
                # allocates ordinary MTP lookahead against the compacted view.
                kwargs = {**kwargs, 'num_lookahead_tokens': 0}
                control = (request, request.num_computed_tokens, num_new_tokens)
        return kwargs, control

    def _native_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        return self.native.allocate_slots(request, num_new_tokens, *args, **kwargs)

    def _allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        request_id = request.request_id
        existing = self.reservations.get(request_id)
        if existing is not None and existing[0] is not request:
            raise ValueError("Request identity reused before releasing capacity")
        if existing is None and self.max_authorities is not None and len(self.reservations) >= self.max_authorities:
            self.rejections += 1
            self.authority_rejections += 1
            return None
        # A terminal speculative step may verify a full block and draft another
        # before the scheduler observes EOS/max_tokens. Retain those credits.
        tokens = min(self.native.max_model_len,
                     request.num_prompt_tokens + request.max_tokens + 2 * self.lookahead + getattr(self,'guided_reserve',0))
        demand = self._demand(tokens)
        if existing is not None and existing[1] != demand:
            raise ValueError("Request lifetime bound changed after admission")
        others = (self._step_total - self._step_remaining.get(request_id, 0)
                  if self._step_remaining is not None else
                  sum(self._remaining(key, quota) for key, (_, quota) in self.reservations.items()
                      if key != request_id))
        extra = kwargs.get("reserved_blocks", 0)
        # Preserve any additional native connector constraint conservatively.
        # This connector normally contributes zero extra credits.
        reserved = others + extra
        residency = getattr(self, 'active_residency', None)
        view = None
        permit = self.restore_admission.permits.get(request_id) if self.restore_admission is not None else None
        if residency is not None:
            residency.expire()
            if existing is None and kwargs.get('num_external_computed_tokens', 0):
                # Prefix admission already transferred this exact hot view to
                # resident_imports. TTL/pressure cannot silently change its lane.
                view = permit.view if permit is not None else residency.match(self.resident_entry(request_id))
            needed = self._remaining(request_id, demand) + reserved - (len(view.blocks) if view else 0)
            residency.make_room(needed, protect=view)
        else:
            needed = self._remaining(request_id, demand) + reserved
        if existing is None and needed > self.block_pool.get_num_free_blocks():
            self.rejections += 1
            return None
        kwargs = self._guided_allocation_options(request, num_new_tokens, kwargs)
        if view is not None and permit is None:
            residency.claim(view, request_id)
        kwargs, control = self._prefill_allocation_options(
            request, num_new_tokens, args, kwargs, existing)
        try:
            result = self._native_allocate_slots(request, num_new_tokens, *args,
                                                 **{**kwargs, "reserved_blocks": reserved})
        except Exception:
            # Unexpected partial native allocation is fatal; never publish a
            # reusable view after an uncertain mutation.
            if view is not None and permit is None and not any(m.req_to_blocks.get(request_id) for m in self.groups):
                residency.rollback(view, request_id)
            raise
        if result is None:
            self.native_allocation_rejections += 1
        if view is not None:
            if result is None:
                if permit is None:
                    residency.rollback(view, request_id)
            else:
                residency.committed(view, request_id)
                self.resident_restores.add(request_id)
        if result is not None:
            self.reservations[request_id] = (request, demand)
            if control is not None:
                self.refresh_control_allocations[request_id] = control
        if result is not None or existing is not None:
            if self._step_remaining is not None:
                self._update_credit(request_id, demand)
            elif result is not None:
                self._remaining(request_id, demand)
        return result

    def free(self, request):
        self.native.free(request)
        getattr(self, 'resident_restores', set()).discard(request.request_id)
        self._forget_credit(request.request_id)
        self.reservations.pop(request.request_id, None)

    def pop_blocks_for_free(self, request):
        blocks = self.native.pop_blocks_for_free(request)
        getattr(self, 'resident_restores', set()).discard(request.request_id)
        self._forget_credit(request.request_id)
        # Deferred physical blocks remain unavailable in the native pool. The
        # completed request no longer needs credits for future growth.
        self.reservations.pop(request.request_id, None)
        return blocks
