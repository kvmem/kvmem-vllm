"""SSD tier: layer-major, unpadded archive layout (archive_layout=layer).

The archive record of one block is [raw plane][values plane] over every archive
layer. For Flash Next the archive layers are the 13 QSA layers followed by their
13 selector ("__qsa_index__") layers, whose raw plane is zero and whose value
rows use only the selector's BF16 bytes. On SSD each record is stored as one
group per QSA layer (csrc/layer_major_arena.hpp):

    [raw K of layer l][values of layer l][selector bytes of layer l, row by row]

so a per-layer selection reads one contiguous run per range of consecutive
slots, and the zero padding is never written (16,640 instead of 26,624 bytes per
token). Writes refuse records whose dropped bytes are not zero.
"""
from types import SimpleNamespace

from .. import _native


def qsa_record_groups(main_layers, block_tokens, heads, dim, raw_element_bytes, selector_bytes):
    """Groups for records of main_layers QSA layers + as many selector layers."""
    width = heads * dim
    raw_slice = block_tokens * width * raw_element_bytes
    value_slice = block_tokens * width
    plane = 2 * main_layers * raw_slice
    if not 0 < selector_bytes <= width:
        raise ValueError('Selector bytes must fit one value row')
    groups = []
    for layer in range(main_layers):
        groups.append([(layer * raw_slice, 1, raw_slice, 0),
                       (plane + layer * value_slice, 1, value_slice, 0),
                       (plane + (main_layers + layer) * value_slice, block_tokens, selector_bytes, width)])
    return groups


class LayerMajorTier:
    """The archive's tier interface (allocate/release/io) over a LayerMajorArena."""
    def __init__(self, directory, capacity, record_bytes, groups, threads=8):
        import os
        os.makedirs(directory, exist_ok=True)        # as the record-layout NVMe tier does
        self.arena = _native.LayerMajorArena(str(directory), capacity, record_bytes, groups, threads=threads)
        self.record_bytes = record_bytes
        self.groups = groups
        self.group_bytes = [self.arena.group_bytes(g) for g in range(self.arena.group_count)]
        self._used = 0

    @staticmethod
    def span(slot, offset, size):
        return slot, offset, size

    def allocate(self, block_id):
        # Block ids are unique below the capacity: the slot is the block id.
        self._used += 1
        return SimpleNamespace(slot=block_id, block_id=block_id)

    def release(self, allocation):
        self._used -= 1
        return True

    @property
    def used_slots(self):
        return self._used

    def io(self, spans, buffer, write):
        slots = []
        for index, (slot, offset, size) in enumerate(spans):
            if size != self.record_bytes or offset != index * self.record_bytes:
                raise ValueError('Layer-major archive I/O moves whole records in buffer order')
            slots.append(slot)
        view = memoryview(buffer).cast('B')
        if write:
            self.arena.write(slots, view)
        else:
            self.arena.read(slots, view)
        return dict(bytes=len(slots) * self.record_bytes, syscalls=0, duration_ns=0)

    def read_from_sources(self, spans, payload, sources):
        raise ValueError('Pre-admitted CPU sources require the CPU archive')


def pack_record_groups(record, segments, out):
    """Pack one group of a whole record (uint8 CPU tensor) into out, as the arena stores it."""
    cursor = 0
    for offset, rows, row_bytes, stride in segments:
        if rows == 1:
            out[cursor:cursor + row_bytes].copy_(record[offset:offset + row_bytes])
        else:
            source = record.as_strided((rows, row_bytes), (stride, 1), offset)
            out[cursor:cursor + rows * row_bytes].view(rows, row_bytes).copy_(source)
        cursor += rows * row_bytes
    if cursor != out.numel():
        raise ValueError('Layer group destination does not match its segments')
