"""Bounded cooperative work with fenced assembly and resident restore credits.

Time limits are host launch budgets, not CUDA execution deadlines. A single
kernel/prepare step cannot be preempted; quantum token sizes remain unchanged.
"""
import time
from collections import OrderedDict


class MaintenancePolicy:
    def __init__(self, options, bytes_per_token, *, clock=time.perf_counter):
        self.max_steps = options.maintenance_max_steps
        self.busy_seconds = options.maintenance_busy_ms / 1000
        self.idle_seconds = options.maintenance_idle_ms / 1000
        self.aging_seconds = options.maintenance_aging_ms / 1000
        self.bytes_per_token, self.clock = bytes_per_token, clock
        self.block_tokens = options.memory.block_tokens
        self.restores = getattr(options, 'async_restore', False)
        largest = self._bytes(options.memory.select_budget + (options.memory.gen_budget if self.restores else 0))
        if self.restores:
            largest += options.session_checkpoint_bytes
        self.capacity = options.maintenance_inflight_bytes or largest
        if self.capacity < largest:
            raise ValueError('Maintenance transfer budget must fit one complete active view')
        self.owner = None
        self.inflight_bytes = 0
        # A resident restore only uploads its bounded model checkpoint, never
        # target KV or shared assembly scratch. Keep one independent credit
        # until its CUDA fence is committed, including cancelled requests.
        self.resident_owners = {}
        self.resident_slots = getattr(options, 'resident_restore_slots', 1)
        self.resident_inflight_bytes = 0
        self.resident_item_capacity = options.session_checkpoint_bytes if self.restores else 0
        self.resident_capacity = self.resident_slots * self.resident_item_capacity
        self.resident_peak_inflight = 0
        self.turns = self.advances = self.forward_quanta = self.time_yields = 0
        self.upload_ratios = OrderedDict()
        self.restore_timings = []
        self.offline = getattr(options, 'offline_agent_policy', False)
        self.restore_handoff_fastpath = getattr(options, 'restore_handoff_fastpath', False)
        self.readiness_costs = {}
        self.transfer_budget = None
        if getattr(options, 'maintenance_transfer_bytes_per_second', 0):
            from .transfer_budget import TransferBudget
            self.transfer_budget = TransferBudget(options.maintenance_transfer_bytes_per_second,
                                                   options.maintenance_transfer_burst_bytes)

    def cost(self, job):
        if getattr(job.command, 'resident', False):
            return job.model_bytes
        window = getattr(job.command, 'window_tokens', None)
        if window is None:
            window = job.command.groups[0][1].window_tokens
        return self._bytes(min(job.history, window)) + getattr(job, 'model_bytes', 0)

    def _bytes(self, tokens):
        # Region gather may read complete records for partial window endpoints.
        block = self.block_tokens
        return ((tokens + block - 1) // block + 2) * block * self.bytes_per_token

    @staticmethod
    def namespace(job):
        return getattr(getattr(getattr(job, 'authority', None), 'memory', None), 'request_id', None)

    def estimate(self, job):
        # Reuse observed transfer sizes (including raw-K hits / resident V),
        # without scanning pages or synchronizing CUDA just to sort the queue.
        return self.cost(job) * self.upload_ratios.get(self.namespace(job), 1.)

    def pop_restore_timings(self):
        rows, self.restore_timings = self.restore_timings, []
        return rows

    def observe(self, job, uploaded_bytes):
        maximum = self.cost(job)
        if not 0 <= uploaded_bytes <= maximum:
            raise ValueError('Assembly upload exceeded its transfer reservation')
        if getattr(job.command, 'resident', False):
            # Hot model-state uploads do not predict a later cold KV transfer.
            return
        namespace = self.namespace(job)
        if namespace is not None:
            ratio = uploaded_bytes / maximum
            previous = self.upload_ratios.pop(namespace, ratio)
            self.upload_ratios[namespace] = .5 * previous + .5 * ratio
            if len(self.upload_ratios) > 128:
                self.upload_ratios.popitem(last=False)

    def order(self, jobs):
        now = self.clock()
        pending = [j for j in jobs if not j.done]
        ordinary = [j for j in pending if j.stage not in ('assembly','restore_copy') and not j.cancelled
                    and now - (j.last_progress if j.last_progress is not None else j.started) < self.aging_seconds]
        measured = bool(ordinary) and all(self.readiness_costs.get(j.command.request_id) is not None for j in ordinary)
        def priority(job):
            last = job.last_progress if job.last_progress is not None else job.started
            # Reap ready fences first to release scarce scratch. Polling these
            # jobs never waits for CUDA or for an unfinished producer future.
            if job.stage in ('assembly', 'restore_copy') or job.cancelled:
                return (0, last, 0)
            if now - last >= self.aging_seconds:
                return (1, last, 0)
            # A resident job has model_bytes == 0 until its checkpoint is
            # claimed. That unknown size must not put a new prepare ahead of
            # a frame ready for its small model-submit stage. Keep fence reap
            # and aging ahead of this preference, and never alter byte credits.
            if self.restore_handoff_fastpath:
                if getattr(job.command, 'resident', False) and job.stage == 'restore_model_prepare':
                    return (2, last, 0)
                return (3, self.estimate(job), last)
            if self.offline:
                return (2, self.readiness_costs[job.command.request_id] if measured else job.started, last)
            return (2, self.estimate(job), last)
        return sorted(pending, key=priority)

    def acquire(self, job):
        if getattr(job.command, 'resident', False):
            rid = job.command.request_id
            if rid in self.resident_owners:
                raise ValueError('Duplicate resident transfer reservation')
            if len(self.resident_owners) >= self.resident_slots:
                return False
            amount = self.cost(job)
            if not 0 < amount <= self.resident_item_capacity:
                raise ValueError('Resident model restore exceeds its transfer reservation')
            self.resident_owners[rid] = amount
            self.resident_inflight_bytes += amount
            self.resident_peak_inflight = max(self.resident_peak_inflight, len(self.resident_owners))
            return True
        if self.owner is not None:
            return False
        amount = self.cost(job)
        if amount > self.capacity:
            raise ValueError('Refresh exceeds its transfer reservation')
        self.owner, self.inflight_bytes = job.command.request_id, amount
        return True

    def release(self, job):
        if getattr(job.command, 'resident', False):
            if job.command.request_id not in self.resident_owners:
                raise ValueError('Foreign resident transfer completion')
            self.resident_inflight_bytes -= self.resident_owners.pop(job.command.request_id)
            return
        if self.owner != job.command.request_id:
            raise ValueError('Foreign maintenance transfer completion')
        self.owner, self.inflight_bytes = None, 0

    def pump(self, jobs, step, *, busy):
        if self.transfer_budget is not None:
            self.transfer_budget.set_busy(busy)
        self.turns += 1
        start = self.clock()
        duration = self.busy_seconds if busy else self.idle_seconds
        advances = forwards = 0
        while advances < self.max_steps:
            advanced = False
            for job in self.order(jobs):
                if busy and forwards and job.stage in ('query', 'guided_prompt', 'guided_decode'):
                    continue
                before = job.steps
                if not step(job):
                    continue
                job.last_progress = self.clock()
                forwards += job.steps - before
                advances += 1
                advanced = True
                break  # Reorder after a state change/fence release.
            if not advanced:
                break  # Never spin on an unfinished future or DMA fence.
            if self.clock() - start >= duration:
                self.time_yields += 1
                break
        self.advances += advances
        self.forward_quanta += forwards

    def snapshot(self):
        return dict(turns=self.turns, advances=self.advances, forward_quanta=self.forward_quanta,
                    time_yields=self.time_yields, inflight_bytes=self.inflight_bytes,
                    capacity_bytes=self.capacity,
                    resident_inflight_bytes=self.resident_inflight_bytes,
                    resident_capacity_bytes=self.resident_capacity,
                    resident_slots=self.resident_slots,
                    resident_inflight=len(self.resident_owners),
                    resident_peak_inflight=self.resident_peak_inflight,
                    readiness_costs=self.readiness_costs if self.offline else None,
                    transfer_budget=self.transfer_budget.snapshot() if self.transfer_budget is not None else None)

    @property
    def resident_owner(self):
        # Preserve the single-lane diagnostic used by older integrations.
        return next(iter(self.resident_owners), None)
