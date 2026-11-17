#!/usr/bin/env python3
"""Transfer microbenchmarks for the QSA streaming tiers (P0 calibration).

gpu   Device-side gather of scattered groups from pinned host memory through
      UVA (one CTA per segment, 16-byte loads), as the per-step miss fetch
      would do, and bulk host-to-device copies of one layer's candidate range,
      as streaming prefill would do. Run inside the vendor container (needs
      torch with CUDA and nvcc for the inline kernel).
ssd   Random reads of archive-sized records (851,968 B: one 32-token Flash
      Next record) from a file on the device under test, with a thread pool,
      optionally O_DIRECT. The file is created only when --create-gib is given.

    python scripts/bench_qsa_stream_io.py gpu --out gpu.json
    python scripts/bench_qsa_stream_io.py ssd --file /path/on/nvme/kv.bin --create-gib 32 --out ssd.json
"""
import argparse
import json
import mmap
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORD_BYTES = 851968

_KERNEL = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void gather_segments(const int4* __restrict__ host, const long* __restrict__ index,
                                int4* __restrict__ out, int vectors) {
  const int4* src = host + index[blockIdx.x] * (long)vectors;
  int4* dst = out + (long)blockIdx.x * vectors;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x) dst[i] = src[i];
}

void uva_gather(torch::Tensor host, torch::Tensor index, torch::Tensor out, int64_t segment_bytes) {
  TORCH_CHECK(host.is_pinned() && index.is_cuda() && out.is_cuda());
  int vectors = segment_bytes / 16;
  int threads = vectors < 256 ? vectors : 256;
  gather_segments<<<index.numel(), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int4*>(host.data_ptr()), index.data_ptr<long>(),
      reinterpret_cast<int4*>(out.data_ptr()), vectors);
}
'''


def _timed(torch, fn, repeat):
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeat / 1e3


def gpu(args):
    import torch
    from torch.utils.cpp_extension import load_inline
    ext = load_inline('qsa_uva_gather', cuda_sources=_KERNEL, functions=['uva_gather'],
                      cpp_sources='void uva_gather(torch::Tensor host, torch::Tensor index, '
                                  'torch::Tensor out, int64_t segment_bytes);',
                      extra_cuda_cflags=['-O3'], verbose=False)
    pool = torch.empty(args.pool_gib << 30, dtype=torch.uint8, pin_memory=True)
    pool.view(torch.int64)[::4096].copy_(torch.arange(pool.numel() // 32768, dtype=torch.int64))
    rows = []
    for segment in args.segment_bytes:
        segments = pool.numel() // segment
        for count in args.segments:
            index = torch.tensor(random.Random(count).sample(range(segments), count), dtype=torch.long,
                                 device='cuda')
            out = torch.empty(count * segment, dtype=torch.uint8, device='cuda')
            seconds = _timed(torch, lambda: ext.uva_gather(pool, index, out, segment), args.repeat)
            check = index[:4].cpu().tolist()
            expect = torch.cat([pool[i * segment:(i + 1) * segment] for i in check])
            if not torch.equal(out[:4 * segment].cpu(), expect):
                raise RuntimeError('UVA gather returned wrong bytes')
            rows.append(dict(kind='uva_gather', segment_bytes=segment, segments=count,
                             bytes=count * segment, seconds=seconds, gbps=count * segment / seconds / 1e9))
            print(json.dumps(rows[-1]), flush=True)
    for mib in args.bulk_mib:
        source = pool[:mib << 20]
        target = torch.empty(mib << 20, dtype=torch.uint8, device='cuda')
        seconds = _timed(torch, lambda: target.copy_(source, non_blocking=True), args.repeat)
        rows.append(dict(kind='bulk_h2d', bytes=mib << 20, seconds=seconds, gbps=(mib << 20) / seconds / 1e9))
        print(json.dumps(rows[-1]), flush=True)
    return dict(device=torch.cuda.get_device_name(), pool_gib=args.pool_gib, rows=rows)


def ssd(args):
    path = Path(args.file)
    if not path.exists():
        if not args.create_gib:
            raise SystemExit(f'{path} does not exist; pass --create-gib to create it')
        chunk = os.urandom(64 << 20)
        with open(path, 'wb') as f:
            for _ in range((args.create_gib << 30) // len(chunk)):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    size = path.stat().st_size
    records = size // RECORD_BYTES
    flags = os.O_RDONLY | (os.O_DIRECT if args.direct else 0)
    fd = os.open(path, flags)
    try:
        if not args.direct:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        rows = []
        for threads in args.threads:
            picks = random.Random(threads).sample(range(records), min(args.reads, records))
            buffers = [mmap.mmap(-1, RECORD_BYTES) for _ in range(threads)]

            def read(job):
                worker, record = job
                return os.preadv(fd, [buffers[worker]], record * RECORD_BYTES)

            jobs = [(i % threads, r) for i, r in enumerate(picks)]
            started = time.perf_counter()
            with ThreadPoolExecutor(threads) as pool:
                total = sum(pool.map(read, jobs))
            seconds = time.perf_counter() - started
            rows.append(dict(kind='ssd_random_records', threads=threads, records=len(picks), bytes=total,
                             seconds=seconds, gbps=total / seconds / 1e9, direct=args.direct))
            print(json.dumps(rows[-1]), flush=True)
            if not args.direct:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return dict(file=str(path), file_bytes=size, record_bytes=RECORD_BYTES, rows=rows)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    g = sub.add_parser('gpu')
    g.add_argument('--pool-gib', type=int, default=8)
    g.add_argument('--segment-bytes', type=int, nargs='+', default=[2048, 4096])
    g.add_argument('--segments', type=int, nargs='+', default=[1024, 8192, 32768, 131072])
    g.add_argument('--bulk-mib', type=int, nargs='+', default=[4, 16, 64, 256])
    g.add_argument('--repeat', type=int, default=20)
    s = sub.add_parser('ssd')
    s.add_argument('--file', required=True)
    s.add_argument('--create-gib', type=int, default=0)
    s.add_argument('--reads', type=int, default=4096)
    s.add_argument('--threads', type=int, nargs='+', default=[1, 4, 8, 16])
    s.add_argument('--direct', action='store_true')
    for p in (g, s):
        p.add_argument('--out')
    args = parser.parse_args(argv)
    if args.mode == 'gpu':
        if args.pool_gib <= 0:
            raise SystemExit('--pool-gib must be positive')
    elif any(t <= 0 or t > (os.cpu_count() or 1) for t in args.threads):
        raise SystemExit('--threads must be between 1 and the core count')
    result = gpu(args) if args.mode == 'gpu' else ssd(args)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
