"""Bounded, opt-in host tracing. No CUDA synchronization or foreground I/O.

Installed only in an isolated diagnostic worker. Nested spans are inclusive;
wall minus thread CPU is waiting/descheduling, not a measure of GIL/GPU idle.
"""
import atexit
from functools import wraps
import itertools
import json
from pathlib import Path
import queue
import threading
import time


class HostTrace:
    def __init__(self, path, *, capacity=32768):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=capacity)
        self.local = threading.local()
        self.ids = itertools.count(1)
        self.recorded = self.dropped = self.written = 0
        self.export_seconds = 0.
        self.error = None
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._export, name='kvmem-host-trace', daemon=True)
        self.worker.start()

    def wrap(self, owner, name, stage, *, sequence=lambda: None):
        original = getattr(owner, name, None)
        if not callable(original) or getattr(original, '_kvmem_host_trace', None) is self:
            return

        @wraps(original)
        def measured(*args, **kwargs):
            if self.stopping.is_set():
                return original(*args, **kwargs)
            parent = getattr(self.local, 'current', None)
            span = next(self.ids)
            self.local.current = span
            started, cpu = time.perf_counter(), time.thread_time()
            error = None
            try:
                return original(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                ended_cpu, ended = time.thread_time(), time.perf_counter()
                self.local.current = parent
                row = dict(stage=stage, span=span, parent=parent, thread=threading.get_native_id(),
                    started_at=started, finished_at=ended, wall_seconds=ended-started,
                    thread_cpu_seconds=ended_cpu-cpu, sequence=sequence(), error=error)
                # Primitive identities only; retain no request, tensor, lease or token list.
                obj = args[0] if args else None
                rid = getattr(obj, 'request_id', None)
                if isinstance(rid, str):
                    row['request_id'] = rid
                if stage == 'checkpoint.capture' and len(args) >= 3:
                    row['request_id'] = args[2]
                if stage == 'checkpoint.token_bytes' and len(args) >= 2:
                    row['tokens'] = args[1]
                if stage == 'checkpoint.validate' and len(args) >= 2 and hasattr(args[1], 'records'):
                    row['records'] = len(args[1].records)
                try:
                    self.pending.put_nowait(row)
                    self.recorded += 1
                except queue.Full:
                    self.dropped += 1

        measured._kvmem_host_trace = self
        setattr(owner, name, measured)

    def _export(self):
        try:
            with self.path.open('x') as stream:
                while not self.stopping.is_set() or not self.pending.empty():
                    try:
                        batch = [self.pending.get(timeout=.25)]
                    except queue.Empty:
                        continue
                    while len(batch) < 256:
                        try:
                            batch.append(self.pending.get_nowait())
                        except queue.Empty:
                            break
                    start = time.perf_counter()
                    stream.write(''.join(json.dumps(row, separators=(',', ':'))+'\n' for row in batch))
                    stream.flush()
                    self.written += len(batch)
                    self.export_seconds += time.perf_counter()-start
                    self._summary()
        except BaseException as exc:
            self.error = repr(exc)
        finally:
            self._summary()

    def _summary(self):
        target = self.path.with_suffix('.status.json')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(recorded=self.recorded, written=self.written,
            dropped=self.dropped, pending=self.pending.qsize(), export_seconds=self.export_seconds,
            error=self.error, stopping=self.stopping.is_set())))
        temporary.replace(target)

    def close(self):
        self.stopping.set()
        self.worker.join(timeout=5)


def install_host_trace(runner, directory):
    if getattr(runner, '_kvmem_host_trace', None) is not None:
        return
    from .profile_scopes import _owned_stages
    trace = HostTrace(Path(directory)/'host_stages.jsonl')
    runner._kvmem_host_trace = trace
    def sequence():
        return getattr(getattr(runner, '_kvmem_step_timing', None), 'sequence', None)
    for owner, name, stage in _owned_stages():
        trace.wrap(owner, name, stage, sequence=sequence)
    for name in ('prepare_attn', 'add_requests', 'execute_model'):
        trace.wrap(runner, name, 'runner.'+name, sequence=sequence)
    atexit.register(trace.close)
