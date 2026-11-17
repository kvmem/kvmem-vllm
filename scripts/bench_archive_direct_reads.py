#!/usr/bin/env python3
"""SSD tier read benchmark: today's archive read path against the O_DIRECT direct reader.

Writes --records Flash Next records (851,968 B each) into an NVMe arena in
--dir, then reads a K32K/G16K-sized selection (--select records) the way window
assembly does, in chunks of --chunk records:

  buffered   NvmeTier.io (one thread, buffered pread, page cache dropped), what
             AsyncArchive.read does today, into a bytearray
  direct     DirectRecordReader with T threads into page-aligned memory, with
             --ring chunks in flight

Selections are either the newest records in order (contiguous) or a random
sorted subset (scattered), like a MeanK selection over a long history.

    python scripts/bench_archive_direct_reads.py --dir /home/me/ssd-bench --records 6144
"""
import argparse
import ctypes
import json
import mmap
import os
import random
import time

RECORD = 851_968


def address(buffer):
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True)
    p.add_argument('--records', type=int, default=6144)
    p.add_argument('--select', type=int, default=1536)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--ring', type=int, default=4)
    p.add_argument('--threads', type=int, nargs='+', default=[4, 8, 16, 32])
    p.add_argument('--request-mib', type=int, nargs='+', default=[1, 4])
    p.add_argument('--out')
    args = p.parse_args()
    from vllm_kvmem import _native
    os.makedirs(args.dir, exist_ok=True)
    tier = _native.NvmeTier(args.dir, args.records * RECORD, RECORD)
    slots = []
    payload = bytearray(os.urandom(RECORD))
    started = time.perf_counter()
    batch = 32
    for first in range(0, args.records, batch):
        group = [tier.allocate(b).slot for b in range(first, min(first + batch, args.records))]
        buffer = payload * len(group)
        tier.io([_native.IoSpan(s, i * RECORD, RECORD) for i, s in enumerate(group)], buffer, True)
        slots += group
    os.sync()
    write_s = time.perf_counter() - started
    rows = [dict(kind='write', gbps=args.records * RECORD / write_s / 1e9, seconds=write_s)]
    print(json.dumps(rows[-1]), flush=True)
    rng = random.Random(7)
    selections = dict(contiguous=slots[-args.select:],
                      scattered=sorted(rng.sample(slots, args.select)))
    for name, selection in selections.items():
        chunks = [selection[i:i + args.chunk] for i in range(0, len(selection), args.chunk)]
        total = len(selection) * RECORD
        # Today's path: one synchronous buffered batch per chunk.
        out = bytearray(args.chunk * RECORD)
        started = time.perf_counter()
        for chunk in chunks:
            tier.io([_native.IoSpan(s, i * RECORD, RECORD) for i, s in enumerate(chunk)], out, False)
        seconds = time.perf_counter() - started
        rows.append(dict(kind='buffered', selection=name, gbps=total / seconds / 1e9, seconds=seconds))
        print(json.dumps(rows[-1]), flush=True)
        ring = [mmap.mmap(-1, args.chunk * RECORD) for _ in range(args.ring)]
        for threads in args.threads:
            for mib in args.request_mib:
                reader = _native.DirectRecordReader(tier, threads=threads, max_request_bytes=mib << 20)
                started = time.perf_counter()
                tickets = {}
                for i in range(min(args.ring, len(chunks))):
                    tickets[i] = reader.submit(chunks[i], address(ring[i]), len(ring[i]))
                for i in range(len(chunks)):
                    reader.wait(tickets.pop(i))
                    following = i + args.ring
                    if following < len(chunks):
                        slot = ring[following % args.ring]
                        tickets[following] = reader.submit(chunks[following], address(slot), len(slot))
                seconds = time.perf_counter() - started
                reader.close()
                rows.append(dict(kind='direct', selection=name, threads=threads, request_mib=mib,
                                 gbps=total / seconds / 1e9, seconds=seconds))
                print(json.dumps(rows[-1]), flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(record_bytes=RECORD, records=args.records, select=args.select, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
