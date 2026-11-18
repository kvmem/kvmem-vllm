"""Compact in-step archive observations, independent of retirement reports."""
import json
from pathlib import Path
from threading import Lock
import time

from .process_memory import sample_process_memory


class ArchiveResourceReporter:
    """Publish counters at most 10Hz; expensive host observations at most 1Hz.

    Called by worker feedback after its publication lock is released. It does
    not traverse authority records, synchronize CUDA, or include completed
    request histories. With a snapshot writer, disk publication can coalesce
    while the sampled counters and their sequence numbers remain exact.
    """
    def __init__(self, output, archive, *, interval=.1, clock=time.monotonic,
                 memory_sampler=sample_process_memory, checkpoint_sampler=None, submit=None):
        self.output, self.archive = Path(output), archive
        self.interval, self.clock, self.memory_sampler = interval, clock, memory_sampler
        self._lock = Lock()
        self._next = self._next_memory = float('-inf')
        self._memory = None
        self.samples = 0
        self.checkpoint_sampler = checkpoint_sampler
        self.submit = submit

    def publish(self, active_requests, retiring_requests, *, force=False):
        now = self.clock()
        if not force and now < self._next:
            return False
        with self._lock:
            if not force and now < self._next:
                return False
            if now >= self._next_memory:
                self._memory = self.memory_sampler()
                self._next_memory = now + 1.
            report = dict(at=now, resource_sample=self.samples+1,
                          active_requests=active_requests, retiring_requests=retiring_requests,
                          process_memory=self._memory, **self.archive.io_usage())
            if self.checkpoint_sampler is not None:
                report['request_checkpoints'] = self.checkpoint_sampler()
            if self.submit is None:
                temporary = self.output/'archive_resources.tmp'
                temporary.write_text(json.dumps(report, separators=(',', ':'))+'\n')
                temporary.replace(self.output/'archive_resources.json')
            else:
                self.submit('archive_resources', report)
            self.samples += 1
            self._next = now + self.interval
            return True
