"""Reserve each admitted target request's future native GPU-page growth.

The native allocator still owns every physical block. Reservations are credits,
not extra allocations: trimming a window returns physical pages while retaining
that request's right to grow again. The optional qualified draft contract adds
native SWA and speculative GDN credits.
Execution remains synchronous, APC-off, TP=1.
"""
from .window_manager import WorkingSetManager


class ReservedKVCacheManager:
    def __init__(self, native, vllm_config, *, draft_contract=None, max_authorities=None):
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SlidingWindowManager
        from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

        from .engine_api import require_contract
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
                   (min(tokens, manager.capacity) + manager.block_size - 1) // manager.block_size
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
        """A no-lookahead frame must never reach a public model forward."""
        for rid, (request, history, count) in self.refresh_control_allocations.items():
            transaction = transactions.get(rid)
            command = transaction[0] if transaction is not None else None
            if (rid in scheduled or command is None or not command.asynchronous or
                    command.selection != 'pressure' or command.query_span is not None or
                    command.speculative_padding or len(command.input_tokens) != count or
                    request.num_computed_tokens != history or request.num_in_flight_tokens or
                    any(ticket.history_end != history or ticket.replay_tokens != count
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
        if view is not None and permit is None:
            residency.claim(view, request_id)
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
        try:
            result = self.native.allocate_slots(request, num_new_tokens, *args,
                                                 **{**kwargs, "reserved_blocks": reserved})
        except Exception:
            # Unexpected partial native allocation is fatal; never publish a
            # reusable view after an uncertain mutation.
            if view is not None and permit is None and not any(m.req_to_blocks.get(request_id) for m in self.groups):
                residency.rollback(view, request_id)
            raise
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
