"""Authority integration for the native archive-wide history worker.

The engine consumes bounded completion descriptors. Policy, irreversible
retirement and large CPU payload destruction remain on native workers.
"""
import time
import math
import hashlib
from threading import Event
from dataclasses import dataclass, field


# Public schemas mirror the native factory. No Python policy callbacks execute
# on the worker; adding a policy requires registering its native constructor.
_POLICY_DEFAULTS = {'lru': {'keep_mandatory': True}, 'redundancy': {
    'keep_mandatory': True, 'importance_weight': 0.1,
    'representative_similarity': 0.9, 'reference_blocks': 32,
    'candidate_blocks': 128, 'pair_comparisons': 512,
    'query_segments_per_step': 8, 'feature_cache_bytes': 134217728,
    'query_cache_bytes': 8388608, 'allow_lossy_fallback': False}}

# Added capacity in the two bulk and two decode D2H slots. Only relevant
# Question rows and generated rows are transmitted, never the whole history.
REDUNDANCY_Q_SLOT_BYTES = 64 * 1024**2
REDUNDANCY_Q_MAX_SLOTS = 4


def redundancy_layers(target_layers):
    if type(target_layers) is not int or target_layers <= 0:
        raise ValueError('Redundancy requires a positive target layer count')
    count = min(4, target_layers)
    return tuple(0 if count == 1 else i*(target_layers-1)//(count-1) for i in range(count))


class RedundancyHistory:
    """Accepted-public feature publisher; native storage owns every copied byte.

    No tensor or token journal is retained here. A cached complete Question is
    reused only after the session machinery has verified its exact prefix.
    Rewind deliberately fails closed instead of importing future observations.
    """
    def __init__(self, scope, target_layers, kv_heads, query_heads, head_dim):
        self.scope = scope
        self.layers = redundancy_layers(target_layers)
        self.configured = scope.configure_features(len(self.layers), kv_heads, query_heads, head_dim)
        self.request_id = self.namespace = self.span = self.digest = None
        self.epoch = None
        self.offset = 0
        self.complete = False
        self.invalid = False
        self.stats = dict(features_published=0, features_missing=0,
            question_tokens=0, observation_tokens=0, question_reuses=0,
            invalidations=0, missing_cached_question=0, transport_overflow=0)

    def invalidate(self, *, overflow=False):
        self.scope.invalidate_question()
        self.epoch = None
        self.complete = False
        self.invalid = True
        self.digest = None
        self.request_id = None
        self.stats['invalidations'] += 1
        self.stats['transport_overflow'] += int(overflow)

    @staticmethod
    def _digest(binding, span):
        # Called only for an already accepted or verified cached prefix.
        from array import array
        import sys
        tokens = binding.tokens[span[0]:span[1]]
        if any(type(token) is not int or not 0 <= token <= 0xffffffff for token in tokens):
            raise ValueError('Redundancy Question tokens must be uint32 integers')
        packed = array('I', tokens)
        if packed.itemsize != 4:
            raise RuntimeError('Redundancy Question digest requires four-byte unsigned integers')
        if sys.byteorder != 'little':
            packed.byteswap()
        return hashlib.sha256(packed.tobytes()).digest()

    @staticmethod
    def _float_rows(rows):
        import torch
        return rows.to(dtype=torch.float32, memory_format=torch.contiguous_format,
            copy=not rows.is_contiguous()).numpy()

    def begin(self, binding, history_begin):
        if self.configured is False:
            self.invalid = True
            return
        if self.request_id == binding.request_id:
            return
        span = binding.query_span
        namespace = binding.namespace, binding.generation
        reusable = (self.complete and not self.invalid and self.namespace == namespace and
            self.span == span and span is not None and binding.history >= span[1] and
            history_begin >= span[1] and self.digest == self._digest(binding, span))
        self.request_id = binding.request_id
        if reusable:
            self.stats['question_reuses'] += 1
            return
        self.namespace, self.span = namespace, span
        self.offset, self.complete, self.invalid, self.digest = 0, False, False, None
        if span is None or not 0 <= span[0] < span[1] <= len(binding.tokens):
            self.invalidate()
            return
        if history_begin > span[0]:
            # No real forward exists for the missing cached Question prefix.
            self.stats['missing_cached_question'] += 1
            self.invalidate()
            return
        self.epoch = self.scope.begin_question(span[1]-span[0])

    def publish_key(self, record, repository):
        if self.configured is False:
            return False
        if not record.archived or (record.mean_key is None and record.key_sum is None):
            self.stats['features_missing'] += 1
            return False
        lease = repository.pin(record.handle)
        if not lease:
            return False
        try:
            # Advanced indexing makes a small independent selected-layer copy;
            # the native method copies it and never retains Python ownership.
            if record.mean_key is not None:
                value = record.mean_key[list(self.layers)].float().contiguous().numpy()
            else:
                # The reference backend keeps its original FP32 index sums;
                # compute only these tiny selected-layer means, never raw KV.
                value = (record.key_sum[list(self.layers)] / record.count).contiguous().numpy()
            published = self.scope.publish_features(lease, value)
            self.stats['features_published' if published else 'features_missing'] += 1
            return published
        finally:
            lease.reset()

    def accept(self, binding, history_begin, retained, *, prefill, row_offset=0,
               rows=None, overflow=False):
        self.begin(binding, history_begin)
        if overflow:
            self.invalidate(overflow=True)
        if self.invalid or not retained:
            return
        if rows is None:
            relevant = (max(history_begin, self.span[0]) < min(history_begin+retained, self.span[1])
                if prefill else self.complete and history_begin+retained > len(binding.tokens))
            if relevant:
                self.invalidate()
            return
        if prefill:
            begin = max(history_begin+row_offset, self.span[0])
            end = min(history_begin+retained, history_begin+row_offset+rows.shape[1], self.span[1])
            if end <= begin:
                return
            offset = begin-self.span[0]
            if offset != self.offset:
                self.invalidate()
                return
            start = begin-history_begin-row_offset
            selected = self._float_rows(rows[:, start:start+end-begin])
            if not self.scope.append_question(self.epoch, offset, selected, end-begin):
                self.invalidate()
                return
            self.offset += end-begin
            self.stats['question_tokens'] += end-begin
            if self.offset == self.span[1]-self.span[0]:
                self.complete = True
                self.digest = self._digest(binding, self.span)
        elif self.complete:
            start = max(0, len(binding.tokens)-history_begin)
            count = min(retained, rows.shape[1])-start
            if count > 0:
                selected = self._float_rows(rows[:, start:start+count])
                if not self.scope.observe_query(self.epoch, selected, count):
                    self.invalidate()
                else:
                    self.stats['observation_tokens'] += count


def historical_record_charge(record_bytes, block_tokens, raw_element_bytes=1):
    """KV plus worst-case FP32 sum and FP16 MeanK for every target/draft layer."""
    elements, remainder = divmod(record_bytes, block_tokens*(raw_element_bytes+1))
    if remainder or not elements:
        raise ValueError('Historical record geometry cannot hold a complete index')
    return record_bytes + elements*6


@dataclass(frozen=True)
class HistoricalGcSettings:
    mode: str = 'off'
    poll_ms: int = 10
    scan_blocks: int = 256
    delete_blocks: int = 64
    ready_low: int = 0
    ready_high: int = 0
    policy: str = 'lru'
    policy_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ('off', 'shadow', 'delete'):
            raise ValueError('historical_gc.mode must be off, shadow or delete')
        if not isinstance(self.policy, str) or self.policy not in _POLICY_DEFAULTS:
            raise ValueError(f'Unknown historical_gc.policy: {self.policy!r} (supported: lru, redundancy)')
        if not isinstance(self.policy_params, dict):
            raise ValueError('historical_gc.policy_params must be an object')
        defaults = _POLICY_DEFAULTS[self.policy]
        for key, value in self.policy_params.items():
            if key not in defaults:
                raise ValueError(f'Unknown historical_gc.policy_params.{self.policy} parameter: {key!r}')
            default = defaults[key]
            if type(default) is bool:
                if type(value) is not bool:
                    raise ValueError(f'historical_gc.policy_params.{key} must be boolean')
            elif type(default) is float:
                lower = -1 if key == 'representative_similarity' else 0
                if type(value) not in (int, float) or not lower <= value <= 1 or not math.isfinite(value):
                    raise ValueError(f'historical_gc.policy_params.{key} must be finite and in [{lower},1]')
            elif type(value) is not int or not 1 <= value <= (2**40 if key.endswith('_bytes') else 65536):
                raise ValueError(f'historical_gc.policy_params.{key} must be a positive bounded integer')
        # Own the configuration and expand defaults without changing the input.
        object.__setattr__(self, 'policy_params', {**defaults, **self.policy_params})
        if self.policy == 'redundancy' and not self.policy_params['keep_mandatory']:
            raise ValueError('historical_gc redundancy requires keep_mandatory=true')
        for name in ('poll_ms', 'scan_blocks', 'delete_blocks'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65536:
                raise ValueError(f'historical_gc.{name} must be a positive bounded integer')
        if (type(self.ready_low) is not int or type(self.ready_high) is not int or
                (self.ready_low, self.ready_high) != (0, 0) and not 0 < self.ready_low < self.ready_high):
            raise ValueError('historical_gc ready watermarks must be automatic (0,0) or 0 < low < high')

    @classmethod
    def parse(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError('historical_gc must be an object or null')
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f'Unknown historical_gc setting: {error}') from error

    def native(self, memory, max_scopes):
        from . import _native
        config = _native.HistoricalGcConfig()
        config.mode = getattr(_native.HistoricalGcMode, self.mode.upper())
        config.policy, config.policy_params = self.policy, self.policy_params
        config.max_scopes = max_scopes
        config.scope_capacity = memory.repository_blocks
        config.poll_ms, config.scan_blocks, config.delete_blocks = self.poll_ms, self.scan_blocks, self.delete_blocks
        config.ready_low = self.ready_low or max(1, memory.repository_blocks//20)
        config.ready_high = self.ready_high or max(config.ready_low+1, memory.repository_blocks//10)
        if config.ready_high > memory.repository_blocks:
            raise ValueError('historical_gc watermarks exceed repository capacity')
        config.background = True
        return config


class IndexReleaseFence:
    """Join native completions without retaining any of their storage owners.

    Compact indices release the temporary FP32 sums before their persistent
    FP16 means. A retired record cannot grant a reusable index credit until
    both allocations, including any tensor aliases, have actually been reaped.
    """
    __slots__ = ('_fences',)

    def __init__(self, *fences):
        if not fences:
            raise ValueError('Index release requires native completion fences')
        self._fences = tuple(fences)

    def query(self):
        return all(fence.query() for fence in self._fences)


class HistoricalAuthorityGc:
    native_history = True

    def __init__(self, memory, archive):
        self.memory, self.archive = memory, archive
        self.scope = archive.register_history_scope(memory.request_id, memory.repository)
        memory._store.attach_history(self.scope)
        memory._history_scope = self.scope
        self.reclaimed_records = 0
        self.retired_index = {}
        self.pending_index = {}
        self.stopped = False

    def defer_index(self, key, record):
        if record.index_owner is not None:
            # Split owners expose a joined fence. Keep the fence, never the
            # owner itself: a pending completion must not retain its storage.
            self.pending_index[key] = record.index_owner.completion

    def index_released(self):
        for key, fence in tuple(self.pending_index.items()):
            if fence.query():
                del self.pending_index[key]
        return not self.pending_index

    def drain(self, records, discarded=None, retired=None):
        discarded = {} if discarded is None else discarded
        if self.stopped:
            # Finalization freezes the logical live view independently of CPU
            # readers still draining. Keep their small cleanup owners until
            # the actual release event; verification must not reopen them.
            for key, record in tuple(records.items()):
                if record.archived and not self.memory.repository.pin(record.handle):
                    self.retired_index[key] = records.pop(key)
                    self.reclaimed_records += 1
                    if retired is not None:
                        retired.append((record.history_begin, *key))
        for event in self.scope.completions(64):
            key = event.handle.slot, event.handle.generation
            record = records.get(key)
            removed = discarded.get(key)
            # I/O can publish native backing before its Future is observed.
            # Publication bookkeeping and staging must complete before its
            # descriptor/index entry is consumed or its slot is reused.
            if event.cpu:
                if record is not None and not record.archived:
                    continue
                if not self.archive.retire_cpu_record(self.memory.request_id, event.handle):
                    continue
                if removed is not None:
                    self.defer_index(key, removed)
                    del discarded[key]
                elif record is not None:
                    self.defer_index(key, record)
                    del records[key]
                    self.reclaimed_records += 1
                    if retired is not None:
                        retired.append((record.history_begin, *key))
                retired_record = self.retired_index.pop(key, None)
                if retired_record is not None:
                    self.defer_index(key, retired_record)
                # Completion is reliable until ACK. Python snapshots can still
                # own the Record; no reusable metadata credit precedes actual
                # index storage release by the native worker.
                fence = self.pending_index.get(key)
                if fence is not None:
                    if not fence.query():
                        continue
                    del self.pending_index[key]
            if not self.scope.acknowledge(event):
                raise RuntimeError('Historical completion acknowledgement was rejected')
        # Native policy failures pause new decisions and remain visible in
        # scope stats. They do not abort decode or stop already committed work.

    def stop(self):
        """Finalization barrier, never called to supply decode capacity."""
        if self.stopped:
            return
        self.scope.pause_policy(True)
        deadline, retry = time.monotonic() + 30, Event()
        while not self.scope.policy_paused:
            if time.monotonic() >= deadline:
                raise TimeoutError('Historical policy did not acknowledge finalization pause')
            if not self.archive._history_background:
                self.archive._history_worker.advance()
            else:
                retry.wait(.001)
        self.stopped = True

    def close(self):
        if not self.archive._history_background:
            self.archive._history_worker.advance()
        if not self.scope.close():
            return False
        for key, record in tuple(self.retired_index.items()):
            if not self.archive.retire_cpu_record(self.memory.request_id, record.handle):
                return False
            self.defer_index(key, record)
            del self.retired_index[key]
        return self.index_released()
