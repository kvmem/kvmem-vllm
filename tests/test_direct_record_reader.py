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
    # A hole inside every record (unused bytes) is skipped; the rest merges across records.
    skip = _aligned(3 * RECORD)
    skip[:] = b'z' * len(skip)
    stats = reader.wait(reader.submit([20, 21, 22], _address(skip), len(skip), hole_offset=4096, hole_bytes=4096))
    assert stats['bytes'] == 3 * (RECORD - 4096)
    for k, slot in enumerate((20, 21, 22)):
        got, want = skip[k * RECORD:(k + 1) * RECORD], payloads[slot]
        assert got[:4096] == want[:4096] and got[8192:] == want[8192:] and got[4096:8192] == b'z' * 4096
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
    with pytest.raises(ValueError):
        reader.submit([0], _address(buffer), RECORD, hole_offset=100, hole_bytes=4096)  # unaligned hole
    with pytest.raises(ValueError):
        reader.submit([0], _address(buffer), RECORD, hole_offset=0, hole_bytes=RECORD)  # whole record
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


def _qsa_like_layout(layers=2, tokens=8, width=1024, aux=512):
    """Records shaped like Flash Next's: [raw plane][values plane], 2*layers archive
    layers; selector layers keep zeros in raw and in the tail of each value row."""
    plane = 2 * layers * tokens * width
    slice_bytes = tokens * width
    groups = []
    for layer in range(layers):
        groups.append([(layer * slice_bytes, 1, slice_bytes, 0),
                       (plane + layer * slice_bytes, 1, slice_bytes, 0),
                       (plane + (layers + layer) * slice_bytes, tokens, aux, width)])
    return 2 * plane, groups


def _qsa_like_record(layers=2, tokens=8, width=1024, aux=512):
    record_bytes, _ = _qsa_like_layout(layers, tokens, width, aux)
    record = bytearray(os.urandom(record_bytes))
    plane, slice_bytes = record_bytes // 2, tokens * width
    for layer in range(layers, 2 * layers):                 # selector raw: zeros
        record[layer * slice_bytes:(layer + 1) * slice_bytes] = bytes(slice_bytes)
        for t in range(tokens):                              # selector values: tail zeros
            row = plane + layer * slice_bytes + t * width
            record[row + aux:row + width] = bytes(width - aux)
    return bytes(record)


def test_layer_major_arena_is_lossless_and_reads_packed_groups(tmp_path):
    if not hasattr(_native, 'LayerMajorArena'):
        pytest.skip('native module predates the layer-major arena')
    record_bytes, groups = _qsa_like_layout()
    try:
        arena = _native.LayerMajorArena(str(tmp_path), 16, record_bytes, groups, threads=4)
    except OSError as error:
        pytest.skip(str(error))
    group_bytes = [arena.group_bytes(g) for g in range(arena.group_count)]
    assert group_bytes == [8192 + 8192 + 8 * 512] * 2
    assert arena.stored_bytes_per_record == sum(group_bytes) < record_bytes
    records = {slot: _qsa_like_record() for slot in range(16)}
    order = [3, 4, 5, 9, 0, 15]
    payload = bytearray(b''.join(records[s] for s in order))
    arena.write(order, payload)
    arena.write([s for s in range(16) if s not in order],
                bytearray(b''.join(records[s] for s in range(16) if s not in order)))
    back = bytearray(len(payload))
    arena.read(order, back)
    assert bytes(back) == bytes(payload)
    # Packed group reads, plane-major per requested group, with a hole (-1).
    slots = [4, 5, -1, 6, 12]
    destination = _aligned(len(slots) * sum(group_bytes))
    offsets = [0, len(slots) * group_bytes[0]]
    stats = arena.wait(arena.submit_groups(slots, [1, 0], offsets, _address(destination), len(destination)))
    assert stats['bytes'] == 4 * sum(group_bytes)
    plane, slice_bytes = record_bytes // 2, 8 * 1024
    for k, group in enumerate((1, 0)):
        for i, slot in enumerate(slots):
            if slot < 0:
                continue
            got = destination[offsets[k] + i * group_bytes[group]:offsets[k] + (i + 1) * group_bytes[group]]
            rec = records[slot]
            want = (rec[group * slice_bytes:(group + 1) * slice_bytes] +
                    rec[plane + group * slice_bytes:plane + (group + 1) * slice_bytes] +
                    b''.join(rec[plane + (2 + group) * slice_bytes + t * 1024:][:512] for t in range(8)))
            assert got == want
    # Bytes the layout would drop must be zero.
    bad = bytearray(records[1])
    bad[2 * slice_bytes + 7] = 1                              # selector raw plane
    with pytest.raises(RuntimeError):
        arena.write([1], bad)
    arena.close()


def test_layer_major_tier_round_trips_and_packs_like_the_arena(tmp_path):
    if not hasattr(_native, 'LayerMajorArena'):
        pytest.skip('native module predates the layer-major arena')
    import torch
    from vllm_kvmem.core.layer_major import LayerMajorTier, pack_record_groups, qsa_record_groups
    # 2 QSA layers + 2 selector layers, 8 tokens, 1 head x 1024 dims, FP8 raw, 512 selector bytes.
    groups = qsa_record_groups(2, 8, 1, 1024, 1, 512)
    record_bytes, reference = _qsa_like_layout()
    assert [[tuple(s) for s in g] for g in groups] == [[tuple(s) for s in g] for g in reference]
    try:
        tier = LayerMajorTier(tmp_path / 'not-yet-created', 8, record_bytes, groups, threads=2)
    except OSError as error:
        pytest.skip(str(error))
    records = [_qsa_like_record() for _ in range(3)]
    spans = [tier.span(slot, i * record_bytes, record_bytes) for i, slot in enumerate((5, 6, 2))]
    tier.io(spans, bytearray(b''.join(records)), True)
    back = bytearray(3 * record_bytes)
    tier.io(spans, back, False)
    assert bytes(back) == b''.join(records)
    # pack_record_groups (in-memory records) == what the arena stores for a group.
    destination = _aligned(tier.group_bytes[1])
    tier.arena.wait(tier.arena.submit_groups([6], [1], [0], _address(destination), len(destination)))
    packed = torch.zeros(tier.group_bytes[1], dtype=torch.uint8)
    pack_record_groups(torch.frombuffer(bytearray(records[1]), dtype=torch.uint8), groups[1], packed)
    assert bytes(destination) == bytes(packed.numpy())
    with pytest.raises(ValueError):
        tier.io([tier.span(5, record_bytes, record_bytes)], bytearray(2 * record_bytes), False)
    tier.arena.close()
