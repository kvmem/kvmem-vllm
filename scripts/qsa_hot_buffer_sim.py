#!/usr/bin/env python3
"""Replay QSA selection traces through a per-layer GPU hot buffer.

Input: directories written with KVMEM_QSA_TOPK_TRACE (adapters/qsa_trace.py).
Each decode call of one request selects up to 512 groups of 4 tokens per
verify row. All of a request's rows are attended in one launch, so the union of
their groups must be resident at once. Groups whose tokens are all inside the
ring (the newest ``ring_tokens`` positions) never need the hot buffer.

For every hot-buffer size and policy the replay reports, per request and layer
and step: the union size, the misses (groups fetched from the CPU pool), steps
whose union does not fit (an exact launch is impossible there), and the bytes
moved per decode step summed over all requests and layers.

    python scripts/qsa_hot_buffer_sim.py TRACE_DIR [TRACE_DIR ...] \
        --hot-pages 2 3 4 --policy lru clock --out sim.json
"""
import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

PAGE_TOKENS = 3008
GROUP_TOKENS = 4
KV_BYTES_PER_TOKEN_LAYER = 1024  # FP8 K + V, 2 heads x 256 dims


def load(paths):
    """Concatenate every trace file under ``paths`` into decode and prefill tables."""
    decode, prefill, layers = defaultdict(list), defaultdict(list), None
    for root in map(Path, paths):
        meta = root / 'trace.json'
        if meta.exists():
            names = json.loads(meta.read_text())['layers']
            if layers is not None and names != layers:
                raise ValueError(f'{root}: traced layers differ from the other directories')
            layers = names
        for path in sorted(root.glob('qsa-topk-*.npz')):
            with np.load(path) as data:
                for key in data.files:
                    (prefill if key.startswith('p_') else decode)[key].append(data[key])
    decode = {k: np.concatenate(v) for k, v in decode.items()}
    prefill = {k: np.concatenate(v) for k, v in prefill.items()}
    return dict(layers=layers or [], decode=decode, prefill=prefill)


def streams(decode):
    """{(state_slot, layer): [(call, groups set, first_position, last_position), ...]} in call order."""
    if not decode:
        return {}
    keys = np.lexsort((decode['call'], decode['layer'], decode['state_slot']))
    out = defaultdict(list)
    current, rows = None, []

    def close():
        slot, layer, call = current
        chosen = np.concatenate([decode['groups'][i] for i in rows])
        positions = decode['query_position'][rows]
        out[(slot, layer)].append((call, set(chosen[chosen >= 0].tolist()),
                                   int(positions.min()), int(positions.max())))

    for i in keys:
        slot = int(decode['state_slot'][i])
        if slot < 0:
            continue
        key = (slot, int(decode['layer'][i]), int(decode['call'][i]))
        if key != current:
            if current is not None:
                close()
            current, rows = key, []
        rows.append(i)
    if current is not None:
        close()
    return out


MAX_DECODE_ADVANCE = 16


def _split_frames(steps):
    """Split a slot's steps where the frame restarts.

    Positions move back after a refresh or when a new request takes the slot,
    and jump forward when a prefill ran in between; a decode step advances by
    at most the verify width.
    """
    frame, last = [], None
    for step in steps:
        if last is not None and (step[3] < last or step[2] > last + MAX_DECODE_ADVANCE):
            yield frame
            frame = []
        frame.append(step)
        last = step[3]
    if frame:
        yield frame


class LRU:
    def __init__(self, capacity):
        self.capacity, self.slots = capacity, OrderedDict()

    def access(self, needed):
        misses = [g for g in needed if g not in self.slots]
        for g in needed:
            if g in self.slots:
                self.slots.move_to_end(g)
        for g in misses:
            if len(self.slots) >= self.capacity:
                victim = next(iter(self.slots))
                if victim in needed:
                    raise OverflowError
                del self.slots[victim]
            self.slots[g] = None
        for g in needed:
            self.slots.move_to_end(g)
        return len(misses)


class Clock:
    def __init__(self, capacity):
        self.capacity = capacity
        self.group = [-1] * capacity
        self.referenced = [False] * capacity
        self.where = {}
        self.hand = 0

    def access(self, needed):
        misses = []
        for g in needed:
            slot = self.where.get(g)
            if slot is None:
                misses.append(g)
            else:
                self.referenced[slot] = True
        for g in misses:
            for _ in range(2 * self.capacity + 1):
                slot = self.hand
                self.hand = (self.hand + 1) % self.capacity
                victim = self.group[slot]
                if victim in needed:
                    continue
                if self.referenced[slot]:
                    self.referenced[slot] = False
                    continue
                break
            else:
                raise OverflowError
            if victim >= 0:
                del self.where[victim]
            self.group[slot] = g
            self.where[g] = slot
            self.referenced[slot] = True
        return len(misses)


class NRU:
    """The parallel policy the residency kernel uses: one pass per step, no per-miss scan.

    Hits get their reference bit set. Misses take unpinned slots whose bit is
    clear first, then any unpinned slot; when clear slots run out, every
    unpinned bit is cleared (one second-chance sweep per exhaustion).
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.group = [-1] * capacity
        self.referenced = [False] * capacity
        self.where = {}

    def access(self, needed):
        misses = [g for g in needed if g not in self.where]
        pinned = {self.where[g] for g in needed if g in self.where}
        for slot in pinned:
            self.referenced[slot] = True
        if not misses:
            return 0
        free = [s for s in range(self.capacity) if s not in pinned and self.group[s] < 0]
        clear = [s for s in range(self.capacity) if s not in pinned and self.group[s] >= 0
                 and not self.referenced[s]]
        marked = [s for s in range(self.capacity) if s not in pinned and self.group[s] >= 0
                  and self.referenced[s]]
        order = free + clear + marked
        if len(order) < len(misses):
            raise OverflowError
        if len(free) + len(clear) < len(misses):
            for s in marked:
                self.referenced[s] = False
        for g, slot in zip(misses, order):
            victim = self.group[slot]
            if victim >= 0:
                del self.where[victim]
            self.group[slot] = g
            self.where[g] = slot
            self.referenced[slot] = True
        return len(misses)


POLICIES = dict(lru=LRU, clock=Clock, nru=NRU)


def replay(steps, capacity_groups, ring_tokens, policy):
    """Per-step (union, misses, fits) for one request-layer stream."""
    rows = []
    for frame in _split_frames(steps):
        cache = POLICIES[policy](capacity_groups)
        cold = True
        for call, groups, first, last in frame:
            ring_start = last - ring_tokens + 1
            needed = {g for g in groups if g * GROUP_TOKENS < ring_start}
            fits = len(needed) <= capacity_groups
            misses = len(needed)
            if fits:
                try:
                    misses = cache.access(needed)
                except OverflowError:
                    fits = False
            rows.append((call, len(groups), len(needed), misses, fits, cold))
            cold = False
    return rows


def _quantiles(values):
    if not values:
        return dict(mean=0.0, p50=0.0, p99=0.0, max=0.0)
    array = np.asarray(values, dtype=np.float64)
    return dict(mean=float(array.mean()), p50=float(np.percentile(array, 50)),
                p99=float(np.percentile(array, 99)), max=float(array.max()))


def simulate(trace, hot_pages=(2, 3, 4), policies=('lru', 'clock', 'nru'), ring_tokens=PAGE_TOKENS,
             page_tokens=PAGE_TOKENS, gbps=40.0):
    request_layer = streams(trace['decode'])
    results = []
    for pages in hot_pages:
        capacity = pages * page_tokens // GROUP_TOKENS
        for policy in policies:
            union, needed, misses, cold_bytes = [], [], [], 0
            unfit = steps = 0
            per_call = defaultdict(int)
            for (slot, layer), steps_list in request_layer.items():
                for call, size, need, miss, fits, cold in replay(steps_list, capacity, ring_tokens, policy):
                    steps += 1
                    union.append(size * GROUP_TOKENS)
                    needed.append(need * GROUP_TOKENS)
                    unfit += not fits
                    moved = miss * GROUP_TOKENS * KV_BYTES_PER_TOKEN_LAYER
                    if cold:
                        cold_bytes += moved
                        continue
                    misses.append(miss * GROUP_TOKENS)
                    per_call[call] += moved
            step_bytes = list(per_call.values())
            mean_bytes = float(np.mean(step_bytes)) if step_bytes else 0.0
            results.append(dict(
                hot_pages=pages, policy=policy, capacity_tokens=capacity * GROUP_TOKENS,
                request_layer_steps=steps, unfit_steps=unfit,
                union_tokens=_quantiles(union), outside_ring_tokens=_quantiles(needed),
                miss_tokens_per_request_layer_step=_quantiles(misses),
                warm_miss_fraction=(float(np.sum(misses)) / max(1.0, float(np.sum(needed))) if misses else 0.0),
                cold_fill_bytes=cold_bytes,
                bytes_per_decode_step=_quantiles(step_bytes),
                ms_per_decode_step_at_gbps=mean_bytes / (gbps * 1e9) * 1e3))
    prefill = trace['prefill']
    prefill_summary = None
    if prefill:
        frame = prefill['p_last_position'] + 1
        prefill_summary = dict(
            chunks=int(len(frame)),
            union_tokens=_quantiles((prefill['p_union'] * GROUP_TOKENS).tolist()),
            union_fraction_of_frame=_quantiles((prefill['p_union'] * GROUP_TOKENS / frame).tolist()),
            rows=_quantiles(prefill['p_rows'].tolist()))
    return dict(layers=trace['layers'], ring_tokens=ring_tokens, page_tokens=page_tokens, gbps=gbps,
                decode=results, prefill=prefill_summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('traces', nargs='+')
    parser.add_argument('--hot-pages', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--policy', nargs='+', choices=sorted(POLICIES), default=['lru', 'clock', 'nru'])
    parser.add_argument('--ring-tokens', type=int, default=PAGE_TOKENS)
    parser.add_argument('--page-tokens', type=int, default=PAGE_TOKENS)
    parser.add_argument('--gbps', type=float, default=40.0, help='effective host-to-device rate for the ms estimate')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    report = simulate(load(args.traces), args.hot_pages, args.policy, args.ring_tokens, args.page_tokens,
                      args.gbps)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    for row in report['decode']:
        print('%d pages %-5s union p99 %6.0f tok  unfit %d/%d  warm miss %.3f  step %.1f MB (%.2f ms)' % (
            row['hot_pages'], row['policy'], row['union_tokens']['p99'], row['unfit_steps'],
            row['request_layer_steps'], row['warm_miss_fraction'],
            row['bytes_per_decode_step']['mean'] / 1e6, row['ms_per_decode_step_at_gbps']))
    if report['prefill']:
        print('prefill union/frame p50 %.3f' % report['prefill']['union_fraction_of_frame']['p50'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
