"""Optional feedback cap for ordinary prefill chunks, never explicit Query replay.

The native scheduler keeps request ordering, decode admission and global batch
limits. This controller only shrinks each ordinary chunk while decode is ready;
a positive floor preserves prefill progress. The target is a measured model-step
latency, including MTP, not a promise about user-visible inter-token latency.
"""
from collections import OrderedDict, deque
import math
import statistics
import time


class BatchPrefillBudget:
    """Shared ordinary-prefill credits while decode is runnable.

    Fair per-request shares prevent the first long prefill from spending every
    step's credits. Reservations are conservative (native allocation can still
    fail) and reset each step; they never advance native logical counters.
    Explicit Query replay is admitted separately and must remain atomic.
    This is an optional token cap, not a latency estimate. Offline ordering and
    aging choose the contenders; without ready decode the cap is disabled.
    """
    def __init__(self, tokens, block_tokens, *, per_request_tokens=0):
        if type(tokens) is not int or tokens <= 0 or tokens % block_tokens:
            raise ValueError('Batch prefill budget must contain aligned blocks')
        if (type(per_request_tokens) is not int or not 0 <= per_request_tokens <= tokens
                or per_request_tokens % block_tokens):
            raise ValueError('Per-request prefill cap must be aligned and within batch credits')
        self.tokens, self.block = tokens, block_tokens
        self.per_request_tokens = per_request_tokens
        self.remaining = self.share = None
        self.contenders = 0
        self.reserved = {}
        self.steps = self.limited_steps = self.deferred = 0

    def begin(self, *, decode_ready, contenders):
        self.steps += 1
        self.reserved.clear()
        self.remaining = self.tokens if decode_ready else None
        self.contenders = contenders
        count = max(1, contenders)
        self.share = max(self.block, self.tokens // count // self.block * self.block)
        if self.per_request_tokens:
            self.share = min(self.share, self.per_request_tokens)
        self.limited_steps += int(decode_ready)

    def available(self, request_id):
        if self.remaining is None:
            return None
        return min(self.share, self.remaining + self.reserved.get(request_id, 0))

    def charge(self, request_id, tokens):
        if self.remaining is None:
            return
        previous = self.reserved.get(request_id, 0)
        if not 0 <= tokens <= self.available(request_id):
            raise ValueError('Ordinary prefill exceeded shared batch credits')
        self.remaining += previous - tokens
        self.reserved[request_id] = tokens
        self.deferred += int(tokens == 0)

    def snapshot(self):
        return dict(tokens=self.tokens, remaining=self.remaining, per_request_share=self.share,
                    per_request_tokens=self.per_request_tokens,
                    contenders=self.contenders, decode_ready=self.remaining is not None,
                    steps=self.steps, limited_steps=self.limited_steps, deferrals=self.deferred)


class PrefillController:
    def __init__(self, options, *, clock=time.monotonic):
        self.clock = clock
        self.minimum = options.prefill_min_tokens
        self.maximum = options.prefill_max_tokens
        self.block = options.memory.block_tokens
        self.target = options.decode_step_target_ms
        self.timeout = options.prefill_feedback_timeout_ms / 1000
        self.limit = self.maximum
        self.sequence = 0
        self.last_feedback = None
        self.low_samples = 0
        self.decreases = self.increases = self.observations = 0
        self.last_ms = None
        self.last_maintenance_ms = None
        self.decode_costs = OrderedDict()
        self.mixed_costs = OrderedDict()
        self.fixed_ms = self.token_ms = None
        self.unattainable = 0

    @staticmethod
    def _bucket(mapping, key):
        bucket = mapping.setdefault(key, deque(maxlen=32))
        mapping.move_to_end(key)
        if len(mapping) > 128:
            mapping.popitem(last=False)
        return bucket

    def _set_limit(self, value):
        value = min(self.maximum, max(self.minimum, int(value) // self.block * self.block))
        self.decreases += int(value < self.limit)
        self.increases += int(value > self.limit)
        self.limit = value

    def observe(self, rows):
        now = self.clock()
        for row in rows:
            sequence = row['sequence']
            if sequence <= self.sequence:
                continue
            self.sequence = sequence
            elapsed = row['step_ms']
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Invalid model-step latency feedback')
            age = now - row['submitted_at']
            # Old total-step-only feedback cannot distinguish a slow CPU
            # maintenance step from extra prefill work. Wait for split timing.
            model_ms = row.get('model_ms')
            maintenance_ms = row.get('maintenance_ms')
            if age < 0 or age > self.timeout or model_ms is None or maintenance_ms is None:
                continue
            if any(not math.isfinite(v) or v < 0 for v in (model_ms, maintenance_ms)):
                raise ValueError('Invalid split model-step latency feedback')
            decode = row['decode_requests']
            tokens = row['prefill_tokens']
            if not decode:
                continue
            if not tokens:
                self._bucket(self.decode_costs, decode).append((now, model_ms))
                continue
            count = row.get('prefill_requests', 1)
            if type(count) is not int or count < 1:
                raise ValueError('Mixed feedback requires a positive prefill request count')
            self.last_feedback, self.last_ms = now, elapsed
            self.last_maintenance_ms = maintenance_ms
            self.observations += 1
            baseline = [ms for t, ms in self.decode_costs.get(decode, ())
                        if now - t <= max(30., self.timeout)]
            base = statistics.median(baseline) if baseline else 0.
            samples = self._bucket(self.mixed_costs, (decode, count))
            while samples and now - samples[0][0] > self.timeout:
                samples.popleft()
            # Keep observed model cost, not residuals against a moving decode
            # baseline. The intercept includes decode and fixed mixed work.
            samples.append((now, tokens, model_ms))
            # Fit marginal token cost from different observed chunk sizes.
            # An intercept retains fixed mixed-batch overhead instead of
            # repeatedly charging it to the shrinking token count.
            slopes = [(y2-y1)/(x2-x1)
                      for i, (_, x1, y1) in enumerate(samples)
                      for _, x2, y2 in list(samples)[i+1:]
                      if x1 != x2 and max(x1, x2) >= 1.5 * min(x1, x2)]
            if not slopes:
                # Changing occupancy or expired feedback must not start another
                # blind halving experiment. Natural tail/global-budget chunks
                # provide different sizes; until then marginal cost is unknown.
                self.fixed_ms = self.token_ms = None
                self._set_limit(self.maximum)
                continue
            self.token_ms = max(0., statistics.median(slopes))
            self.fixed_ms = max(base, statistics.median(
                y-self.token_ms*x for _, x, y in samples))
            if self.fixed_ms + self.token_ms*count*self.minimum > 1.1*self.target:
                # Even the smallest chunk cannot meet the target. Amortize the
                # fixed work instead of trapping every decode step at the floor.
                self.unattainable += 1
                self._set_limit(self.maximum)
            elif self.token_ms == 0:
                self._set_limit(self.maximum)
            else:
                desired = (self.target-self.fixed_ms)/(self.token_ms*count)
                if not .8*self.limit <= desired <= 1.25*self.limit:
                    self._set_limit(min(desired, 2*self.limit))

    def cap(self, *, decode_ready):
        if (not decode_ready or self.last_feedback is None or
                self.clock() - self.last_feedback > self.timeout):
            return self.maximum
        return self.limit

    def snapshot(self):
        return dict(chunk_tokens=self.limit, target_step_ms=self.target,
                    last_mixed_step_ms=self.last_ms, observations=self.observations,
                    decreases=self.decreases, increases=self.increases,
                    maintenance_ms=self.last_maintenance_ms,
                    fixed_model_ms=self.fixed_ms, marginal_prefill_ms_per_token=self.token_ms,
                    unattainable_target_samples=self.unattainable)
