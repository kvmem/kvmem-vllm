"""Bounded request-local capture reads and compact incremental progress.

These helpers change client bookkeeping only. Full case results remain the
authoritative artifacts; callers retain their retirement and validation barriers.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
import time

VERSION = 'incremental-client-2'


def token_digest(tokens):
    """sha256 of the compact JSON token list (the same value as benchmark_memory.digest_tokens)."""
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def compact_response(response, tokens=None):
    """The response to save: each echoed prompt replaced by its length and digest.

    With ``return_token_ids`` the server echoes the whole prompt in every
    choice (8.6 MB for a 1M-token case, 99.9% of the file) next to the few
    output tokens the run is about. Pass the prompt that was sent: an echo that
    differs from it is kept in full, so the caller's own check fails with the
    evidence on disk. Output token IDs, text, usage and metrics stay unchanged.
    A response without an echo (streaming clients) is returned as is.
    """
    choices = response.get('choices') or []
    if not any('prompt_token_ids' in choice for choice in choices):
        return response
    if tokens is not None and any(choice.get('prompt_token_ids') is not None and
                                  list(choice['prompt_token_ids']) != list(tokens) for choice in choices):
        return response
    compacted = []
    for choice in choices:
        echoed = choice.get('prompt_token_ids')
        if echoed is None:
            compacted.append(choice)
            continue
        choice = {key: value for key, value in choice.items() if key != 'prompt_token_ids'}
        choice['prompt_token_count'] = len(echoed)
        choice['prompt_token_sha256'] = token_digest(echoed)
        compacted.append(choice)
    return dict(response, choices=compacted)


def prompt_matches(choice, tokens):
    """True when a saved choice, full or compacted, carries exactly ``tokens``."""
    if choice.get('prompt_token_ids') is not None:
        return list(choice['prompt_token_ids']) == list(tokens)
    return (choice.get('prompt_token_count') == len(tokens) and
            choice.get('prompt_token_sha256') == token_digest(tokens))


class PreparedSequence:
    """Prepare at most one next input while the caller serves the current one.

    The caller must acquire its history credit before entering this context.
    Preparation never sends requests or changes their order. Exceptions surface
    at the affected item, so no later input is dispatched after a failed check.
    """
    def __init__(self, items, prepare):
        self._items = iter(items)
        self._prepare = prepare
        self._executor = None
        self._pending = None

    def __enter__(self):
        if self._executor is not None:
            raise RuntimeError('Preparation context already entered')
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='benchmark-input')
        self._advance()
        return self

    def _advance(self):
        try:
            item = next(self._items)
        except StopIteration:
            self._pending = None
        else:
            self._pending = (item, self._executor.submit(self._prepare, item))

    def __iter__(self):
        return self

    def __next__(self):
        if self._executor is None:
            raise RuntimeError('Preparation context is not active')
        if self._pending is None:
            raise StopIteration
        item, future = self._pending
        prepared = future.result()
        self._advance()
        return item, prepared

    def __exit__(self, *exc):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = self._pending = None


class ClientTiming:
    """Request-local host evidence; concurrent stages are not GPU idle time."""
    def __init__(self):
        self.started_at = time.time()
        self.started = time.perf_counter()
        self.stages = []

    @contextmanager
    def stage(self, name):
        start, cpu = time.perf_counter(), time.thread_time()
        try:
            yield
        finally:
            self.stages.append(dict(name=name, started_seconds=start-self.started,
                wall_seconds=time.perf_counter()-start, thread_cpu_seconds=time.thread_time()-cpu))

    def summary(self):
        return dict(started_at=self.started_at, wall_seconds=time.perf_counter()-self.started,
            stages=list(self.stages), scope='Client host intervals; not GPU kernel or idle attribution. '
                'Prepare wait excludes work already overlapped with the previous request.')


class CompactResults:
    """Load prior results once; subsequent summaries never reread case files."""
    def __init__(self, rows, cases, fields):
        self.order = tuple(row['id'] for row in rows)
        if len(set(self.order)) != len(self.order):
            raise ValueError('Duplicate progress case identity')
        self.allowed = frozenset(self.order)
        self.fields = frozenset(fields) | {'id', 'status'}
        self._records = {}
        self._lock = RLock()
        for ident in self.order:
            path = Path(cases)/ident/'result.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record.get('id') != ident:
                    raise ValueError('Persisted result identity differs from its case')
                self.put(record)

    def put(self, record):
        ident = record['id']
        if ident not in self.allowed or record.get('status') not in ('completed', 'error'):
            raise ValueError('Invalid compact result identity/status')
        compact = deepcopy({key:value for key,value in record.items() if key in self.fields})
        with self._lock:
            self._records[ident] = compact

    def snapshot(self):
        with self._lock:
            return deepcopy([self._records[ident] for ident in self.order if ident in self._records])


@dataclass(frozen=True)
class _CaptureMark:
    device: int
    inode: int
    offset: int
    anchor: bytes
    at_line_boundary: bool


class CaptureSlice:
    """Capture a log boundary before HTTP dispatch; parse its new rows once.

    A preexisting partial final line belongs to earlier traffic and is skipped.
    Trailing partial lines remain unparsed, as in the native graph audit reader.
    Rotation/truncation/rewrite is an error, never a whole-file fallback.
    """
    def __init__(self, paths):
        self._marks, self._cache, self._spans = {}, {}, {}
        self.bytes_read = self.records_read = 0
        for value in paths:
            path = Path(value)
            try:
                with path.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    stream.seek(max(0,stat.st_size-64))
                    anchor = stream.read(stat.st_size-stream.tell())
                    self._marks[path] = _CaptureMark(stat.st_dev,stat.st_ino,stat.st_size,
                        anchor,not anchor or anchor.endswith(b'\n'))
            except FileNotFoundError:
                self._marks[path] = None

    def read(self, value):
        path = Path(value)
        if path not in self._marks:
            raise ValueError('Capture path was not declared before dispatch')
        if path in self._cache:
            return self._cache[path]
        mark = self._marks[path]
        try:
            with path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                if mark is not None:
                    if (stat.st_dev,stat.st_ino)!=(mark.device,mark.inode) or stat.st_size<mark.offset:
                        raise RuntimeError('Capture log rotated or truncated during request')
                    stream.seek(mark.offset-len(mark.anchor))
                    if stream.read(len(mark.anchor)) != mark.anchor:
                        raise RuntimeError('Capture prefix changed during request')
                    stream.seek(mark.offset)
                data = stream.read()
        except FileNotFoundError:
            if mark is not None:
                raise RuntimeError('Capture log disappeared during request') from None
            data = b''
        self.bytes_read += len(data)
        begin = mark.offset if mark is not None else 0
        if mark is not None and not mark.at_line_boundary:
            head,separator,data = data.partition(b'\n')
            begin += len(head) + len(separator)
            if not separator:
                data = b''
        data = data[:data.rfind(b'\n')+1]
        records = tuple(json.loads(line) for line in data.splitlines() if line.strip())
        self.records_read += len(records)
        self._cache[path] = records
        self._spans[path] = (begin, begin + len(data))
        return records

    def reference(self, value, request_prefix, events):
        """Where a request's events sit in the capture log, for a result to keep instead of a copy.

        The server's capture log already holds every event; ``capture_events``
        reads them back from this byte range and checks the count.
        """
        path = Path(value)
        self.read(path)
        begin, end = self._spans[path]
        return dict(capture_file=path.name, capture_dir=str(path.parent), begin=begin, end=end,
                    request_prefix=request_prefix, count=len(events))


def capture_events(reference, capture_dir=None):
    """The events a result referenced with CaptureSlice.reference, read back from the capture log."""
    directory = Path(capture_dir or reference['capture_dir'])
    with (directory / reference['capture_file']).open('rb') as stream:
        stream.seek(reference['begin'])
        data = stream.read(reference['end'] - reference['begin'])
    events = [json.loads(line) for line in data.splitlines() if line.strip()]
    events = [e for e in events if str(e.get('request_id', '')).startswith(reference['request_prefix'])]
    if len(events) != reference['count']:
        raise ValueError('Capture log no longer holds the referenced events')
    return events


def result_events(result, case_dir=None):
    """A saved result's layerwise events: copied inline (older runs) or referenced.

    Cases live at RUN/run/answers/cases/ID and the server log at
    RUN/run/server/capture; that location is used when the recorded absolute
    one has moved.
    """
    events = (result.get('layerwise') or {}).get('events')
    if events is None or isinstance(events, list):
        return events or []
    recorded = Path(events['capture_dir'])
    if not (recorded / events['capture_file']).is_file() and case_dir is not None:
        recorded = Path(case_dir).parents[2] / 'server' / 'capture'
    return capture_events(events, recorded)
