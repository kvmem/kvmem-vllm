"""SSD tier: O_DIRECT batched record reads (csrc/direct_record_reader.hpp)."""
import mmap
import os

import pytest

from vllm_kvmem import _native

RECORD = 3 * 4096


def _aligned(nbytes):
    return mmap.mmap(-1, nbytes)        # page aligned, like cudaHostAlloc memory


def _address(buffer):
    import ctypes
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def _tier(tmp_path, records=64):
    tier = _native.NvmeTier(str(tmp_path), records * RECORD, RECORD)
    payloads = {}
    for block in range(records):
        slot = tier.allocate(block)
        slot = slot if isinstance(slot, int) else slot.slot
        data = os.urandom(RECORD)
        tier.io([_native.IoSpan(slot, 0, RECORD)], bytearray(data), True)
        payloads[slot] = data
    return tier, payloads


def test_batches_merge_split_and_match_the_buffered_reads(tmp_path):
    if not hasattr(_native, 'DirectRecordReader'):
        pytest.skip('native module predates the direct reader')
    tier, payloads = _tier(tmp_path)
    try:
        reader = _native.DirectRecordReader(tier, threads=4, max_request_bytes=2 * 4096)
    except OSError as error:            # filesystems without O_DIRECT (tmpfs on old kernels)
        pytest.skip(str(error))
    # Runs of consecutive slots (merged, then split into 8 KiB requests) and scattered slots.
    slots = [5, 6, 7, 8, 30, 2, 3, 63, 0]
    destination = _aligned(len(slots) * RECORD)
    ticket = reader.submit(slots, _address(destination), len(destination))
    stats = reader.wait(ticket)
    assert stats['bytes'] == len(slots) * RECORD
    # [5..8] -> 4 records = 6 requests of 8 KiB; 30, 63, 0: 2 each; [2, 3]: 3.
    assert stats['requests'] == 6 + 2 + 3 + 2 + 2
    for i, slot in enumerate(slots):
        assert destination[i * RECORD:(i + 1) * RECORD] == payloads[slot]
    # Several batches in flight at once.
    buffers = [_aligned(4 * RECORD) for _ in range(3)]
    tickets = [reader.submit([s, s + 1, s + 2, s + 3], _address(b), len(b))
               for s, b in zip((10, 20, 40), buffers)]
    for ticket, start, buffer in zip(tickets, (10, 20, 40), buffers):
        reader.wait(ticket)
        assert bytes(buffer) == b''.join(payloads[s] for s in range(start, start + 4))
    # Negative slots are holes the caller fills (records still in memory).
    holes = _aligned(4 * RECORD)
    holes[RECORD:2 * RECORD] = b'x' * RECORD
    stats = reader.wait(reader.submit([9, -1, 11, 12], _address(holes), len(holes)))
    assert stats['bytes'] == 3 * RECORD and stats['requests'] == 1 + 4
    assert holes[:RECORD] == payloads[9] and holes[RECORD:2 * RECORD] == b'x' * RECORD
    assert holes[2 * RECORD:] == payloads[11] + payloads[12]
    assert reader.wait(reader.submit([-1, -1], _address(holes), len(holes)))['bytes'] == 0
    reader.close()


def test_rejects_bad_batches(tmp_path):
    if not hasattr(_native, 'DirectRecordReader'):
        pytest.skip('native module predates the direct reader')
    tier, _ = _tier(tmp_path, records=4)
    try:
        reader = _native.DirectRecordReader(tier, threads=2)
    except OSError as error:
        pytest.skip(str(error))
    buffer = _aligned(2 * 4096 + RECORD)
    with pytest.raises(ValueError):
        reader.submit([0], _address(buffer) + 512, RECORD)          # unaligned destination
    with pytest.raises(ValueError):
        reader.submit([0, 1], _address(buffer), RECORD)             # destination too small
    with pytest.raises(IndexError):
        reader.submit([4], _address(buffer), RECORD)                # outside the arena
    with pytest.raises(ValueError):
        reader.wait(12345)
    reader.close()


def test_chunk_views_match_the_buffered_record_layout():
    # direct_assembly_reads exposes the same per-layer bytes as read_records + torch.cat.
    import torch
    from types import SimpleNamespace
    from vllm_kvmem.serving.direct_assembly_reads import _chunk_views
    for raw_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        shape = (6, 4, 1, 8)                                  # archive layers, tokens, heads, dim
        elements = 6 * 4 * 1 * 8
        raw_bytes = elements * raw_dtype.itemsize
        authority = SimpleNamespace(shape=shape, raw_dtype=raw_dtype, raw_record_bytes=raw_bytes,
                                    record_bytes=raw_bytes + elements)
        counts = [4, 4, 3, 4, 1]                              # partial records inside the chunk too
        data = torch.randint(0, 256, (len(counts) * authority.record_bytes,), dtype=torch.uint8)
        raw_of, values_of, tokens = _chunk_views(authority, data, counts)
        rows = data.view(len(counts), -1)
        raw = torch.cat([rows[j, :raw_bytes].view(raw_dtype).view(shape)[:, :c] for j, c in enumerate(counts)], 1)
        values = torch.cat([rows[j, raw_bytes:].view(shape)[:, :c] for j, c in enumerate(counts)], 1)
        assert tokens == sum(counts)
        for layer in range(shape[0]):
            assert torch.equal(raw_of(layer).view(torch.uint8), raw[layer].contiguous().view(torch.uint8))
            assert torch.equal(values_of(layer), values[layer])
            assert raw_of(layer).is_contiguous() and values_of(layer).is_contiguous()
