"""SSD tier: pipelined record reads for window assembly (archive_direct_reads).

A selection is assembled chunk by chunk. Each chunk's records are read with
O_DIRECT straight into one slot of a process-wide pinned ring (no page cache,
no bounce buffer), copied to the GPU by DMA, and exposed per archive layer in
exactly the layout AuthorityStore.read_records produces, so assembly reads the
same bytes as the buffered path.

The ring has RING slots: RING - 1 chunks are being read while the newest chunk
is in DMA, so the drive keeps a deep queue while the GPU assembles. A slot is
refilled only after the event of its previous DMA completed.
"""
import threading
import time

import torch

ALIGN = 4096
RING = 4

_rings = {}
_rings_lock = threading.Lock()


class _PinnedRing:
    def __init__(self, slot_bytes, slots):
        self.slot_bytes = -(-slot_bytes // ALIGN) * ALIGN
        self._backing = torch.empty(self.slot_bytes * slots + ALIGN, dtype=torch.uint8, pin_memory=True)
        start = (-self._backing.data_ptr()) % ALIGN
        self.slots = [self._backing[start + i * self.slot_bytes:start + (i + 1) * self.slot_bytes]
                      for i in range(slots)]
        self.lock = threading.Lock()


def _ring(slot_bytes):
    with _rings_lock:
        ring = _rings.get(slot_bytes)
        if ring is None:
            ring = _rings[slot_bytes] = _PinnedRing(slot_bytes, RING)
        return ring


def _chunk_views(authority, device_bytes, counts):
    """Per-layer accessors over one chunk of whole archive records on the GPU."""
    shape = authority.shape                      # (archive layers, block tokens, heads, dim)
    records, tokens_per_record = len(counts), shape[1]
    rows = device_bytes.view(records, authority.record_bytes)
    raw = rows[:, :authority.raw_record_bytes].view(authority.raw_dtype).unflatten(1, shape)
    values = rows[:, authority.raw_record_bytes:].unflatten(1, shape)
    valid = None
    if any(count != tokens_per_record for count in counts):
        valid = torch.cat([torch.arange(count, dtype=torch.int64) + j * tokens_per_record
                           for j, count in enumerate(counts)]).to(device_bytes.device)

    def layer(planes, index):
        out = planes[:, index].reshape(records * tokens_per_record, *shape[2:])
        return out if valid is None else out.index_select(0, valid)

    return (lambda i: layer(raw, i)), (lambda i: layer(values, i)), sum(counts)


def direct_chunks(authority, chunks, device, stats):
    """Yield (raw_of, values_of, tokens, bytes) for each chunk (a list of record handles)."""
    record_bytes = authority.record_bytes
    ring = _ring(max(len(chunk) for chunk in chunks) * record_bytes)
    with ring.lock:
        events = [None] * RING
        pending = {}

        def issue(index):
            slot = index % RING
            if events[slot] is not None:
                events[slot].synchronize()       # its previous DMA has left the slot
                events[slot] = None
            pending[index] = authority.start_direct_read(chunks[index], ring.slots[slot])

        try:
            for index in range(min(RING - 1, len(chunks))):
                issue(index)
            for index, chunk in enumerate(chunks):
                held, counts = pending.pop(index)
                waited = time.perf_counter()
                result = held.wait()
                stats['archive_direct_wait_s'] += time.perf_counter() - waited
                stats['archive_direct_bytes'] += result['bytes']
                stats['archive_direct_requests'] += result['requests']
                size = len(chunk) * record_bytes
                device_bytes = torch.empty(size, dtype=torch.uint8, device=device)
                device_bytes.copy_(ring.slots[index % RING][:size], non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                events[index % RING] = event
                if index + RING - 1 < len(chunks):
                    issue(index + RING - 1)
                raw_of, values_of, tokens = _chunk_views(authority, device_bytes, counts)
                yield raw_of, values_of, tokens, size
        finally:
            # Never return the ring while the drive or a DMA may still touch it.
            for held, _ in pending.values():
                try:
                    held.wait()
                except BaseException:
                    pass
            for event in events:
                if event is not None:
                    event.synchronize()
