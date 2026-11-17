"""QSA selector pages carved from the native KV pool (Flash Next, qsa_split_selector).

Placement: carved blocks sit at the END of the pool (``place_carves``). B12X
rejects a writable selector cache that overlaps the read-only main K/V view, so
the QSA bindings see a main view that stops at the first selector block; native
block IDs stay absolute and every main page vLLM hands out lies below it.

The vendor layout packs each QSA layer's compressed selector keys into the tail
of every main K/V page, so selector coverage and resident main K/V are the same
pages. With the split, the scheduler takes one contiguous range of native block
IDs at startup and never frees it. Each QSA layer owns an equal contiguous
share, viewed as dense compressed pages: one Flash Next native page (3,272,704
bytes) holds exactly 17 compressed pages of 752 groups x 128 BF16.

A request state slot owns a fixed run of compressed pages in every layer, sized
for the largest frame a request can reach (K + G + draft lookahead). Neither the
scheduler nor the batch order decides where a slot's selector keys live, so the
worker and the scheduler derive the same carve from the same configuration and
need no extra messages: the worker binds views over the range, the scheduler
reserves it and checks that it received exactly that range.
"""
from dataclasses import dataclass, replace

NULL_BLOCKS = 1


@dataclass(frozen=True)
class SelectorCarve:
    layers: int
    slots: int
    page_tokens: int
    pages_per_slot: int
    per_native_page: int
    native_page_bytes: int
    compressed_page_bytes: int
    first_block: int = NULL_BLOCKS

    @property
    def compressed_pages_per_layer(self):
        return self.slots * self.pages_per_slot

    @property
    def blocks_per_layer(self):
        return -(-self.compressed_pages_per_layer // self.per_native_page)

    @property
    def blocks(self):
        return self.layers * self.blocks_per_layer

    @property
    def end_block(self):
        return self.first_block + self.blocks

    def layer_first_block(self, layer):
        if not 0 <= layer < self.layers:
            raise ValueError('QSA selector layer index outside the carve')
        return self.first_block + layer * self.blocks_per_layer

    def slot_row(self, slot, width):
        """Compressed page IDs for logical pages 0..width-1 of a slot's frame; -1 past its run."""
        if not 0 <= slot < self.slots:
            raise ValueError('QSA state slot outside the carve')
        if width < self.pages_per_slot:
            raise ValueError('QSA compressed table is narrower than a slot run')
        base = slot * self.pages_per_slot
        return [base + j for j in range(self.pages_per_slot)] + [-1] * (width - self.pages_per_slot)

    def describe(self):
        return dict(layers=self.layers, slots=self.slots, page_tokens=self.page_tokens,
                    pages_per_slot=self.pages_per_slot, per_native_page=self.per_native_page,
                    blocks_per_layer=self.blocks_per_layer, blocks=self.blocks,
                    first_block=self.first_block, bytes=self.blocks * self.native_page_bytes)


def plan_carve(*, layers, slots, frame_tokens, page_tokens, native_page_bytes, compress_ratio, index_head_dim,
               index_element_bytes=2):
    for name, value in (('layers', layers), ('slots', slots), ('frame_tokens', frame_tokens),
                        ('page_tokens', page_tokens), ('native_page_bytes', native_page_bytes),
                        ('compress_ratio', compress_ratio), ('index_head_dim', index_head_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f'QSA selector carve requires a positive integer {name}')
    if page_tokens % compress_ratio:
        raise ValueError('QSA main page tokens must be a multiple of the compression ratio')
    compressed_page_bytes = page_tokens // compress_ratio * index_head_dim * index_element_bytes
    per_native_page, remainder = divmod(native_page_bytes, compressed_page_bytes)
    if remainder or per_native_page <= 0:
        raise ValueError('A native page must hold a whole number of compressed selector pages')
    return SelectorCarve(layers=layers, slots=slots, page_tokens=page_tokens,
                         pages_per_slot=-(-frame_tokens // page_tokens), per_native_page=per_native_page,
                         native_page_bytes=native_page_bytes, compressed_page_bytes=compressed_page_bytes)


def frame_tokens(memory, draft_tokens):
    """The largest frame a request can address: K + G plus one verify interval."""
    return memory.select_budget + memory.gen_budget + draft_tokens + 1


# ---------------------------------------------------------------------------
# qsa_streaming: native blocks reserved after the selector carve.
#
# Every streaming request owns exactly hot_pages + 1 native pages per QSA layer
# (hot pages, then the ring page every K/V write lands in); its frame K/V lives
# in a pinned host pool. Shared, startup-reserved blocks complete the design:
#
# - prefill staging: ``staging_buffers`` frame-sized runs of native pages. A
#   prefilling streaming request attends from a staging run laid out exactly as
#   resident frame pages (column j = frame tokens [j*page, (j+1)*page)), filled
#   from the host pool and the ring before the layer attends. Runs alternate by
#   QSA layer so one can be filled while the previous layer attends.
# - residency maps: the per-slot, per-layer maps of the residency kernels
#   (adapters/qsa_residency.py), bump-allocated from raw bytes of these blocks.
#
# The scheduler and the worker both derive this layout from the configuration.

MAP_ALIGNMENT = 256
_DTYPE_BYTES = {'uint8': 1, 'int32': 4, 'int64': 8}
GROUP_TOKENS = 4


def residency_map_plan(*, layers, slots, max_groups, hot_groups, hot_pages, max_rows):
    """[(key, shape, dtype)] in allocation order; one source for both sides."""
    # Slot masks carry one extra, always-zero entry: a dummy target for rows
    # without a state slot, so masked updates need no host branch.
    plan = [('streaming', (slots + 1,), 'uint8'), ('decode_mask', (slots + 1,), 'uint8'),
            ('stats', (4,), 'int64'), ('pool_base', (slots,), 'int64'),
            ('spoof', (max_rows,), 'int64'), ('row_positions', (max_rows,), 'int64')]
    for layer in range(layers):
        plan += [((layer, 'group_slot'), (slots, max_groups), 'int32'),
                 ((layer, 'slot_group'), (slots, hot_groups), 'int32'),
                 ((layer, 'referenced'), (slots, hot_groups), 'uint8'),
                 ((layer, 'hot_page_ids'), (slots, hot_pages + 1), 'int32'),
                 ((layer, 'copy_list'), (slots * hot_groups, 3), 'int32'),
                 ((layer, 'copy_count'), (1,), 'int32')]
    return plan


def residency_map_bytes(plan):
    total = 0
    for _, shape, dtype in plan:
        count = 1
        for extent in shape:
            count *= extent
        total = -(-total // MAP_ALIGNMENT) * MAP_ALIGNMENT + count * _DTYPE_BYTES[dtype]
    return -(-total // MAP_ALIGNMENT) * MAP_ALIGNMENT


@dataclass(frozen=True)
class StreamingCarve:
    selector: SelectorCarve
    hot_pages: int
    frame_pages: int
    staging_buffers: int
    map_bytes: int
    max_groups: int
    hot_groups: int
    max_rows: int
    staging_first: int = -1     # set by place_carves
    maps_first: int = -1

    @property
    def native_page_bytes(self):
        return self.selector.native_page_bytes

    @property
    def page_tokens(self):
        return self.selector.page_tokens

    @property
    def layers(self):
        return self.selector.layers

    @property
    def slots(self):
        return self.selector.slots

    @property
    def first_block(self):
        """Lowest streaming block (staging); the selector carve follows, then the maps."""
        return self.staging_first

    @property
    def staging_blocks(self):
        return self.staging_buffers * self.frame_pages

    @property
    def map_blocks(self):
        return -(-self.map_bytes // self.native_page_bytes)

    @property
    def maps_first_block(self):
        return self.maps_first

    @property
    def blocks(self):
        return self.staging_blocks + self.map_blocks

    @property
    def end_block(self):
        return self.maps_first + self.map_blocks

    @property
    def frame_capacity_tokens(self):
        """Host pool tokens per slot and layer: whole pages, so staging copies are page-sized."""
        return self.frame_pages * self.page_tokens

    @property
    def pages_per_request(self):
        """Native pages a streaming request owns in each QSA layer."""
        return self.hot_pages + 1

    def staging_row(self, buffer):
        if not 0 <= buffer < self.staging_buffers:
            raise ValueError('QSA staging buffer outside the carve')
        start = self.staging_first + buffer * self.frame_pages
        return list(range(start, start + self.frame_pages))

    def map_plan(self):
        return residency_map_plan(layers=self.layers, slots=self.slots, max_groups=self.max_groups,
                                  hot_groups=self.hot_groups, hot_pages=self.hot_pages, max_rows=self.max_rows)

    def pool_bytes_per_slot(self, token_bytes):
        return self.layers * 2 * self.frame_capacity_tokens * token_bytes

    def describe(self):
        return dict(hot_pages=self.hot_pages, frame_pages=self.frame_pages, staging_buffers=self.staging_buffers,
                    staging_blocks=self.staging_blocks, map_bytes=self.map_bytes, map_blocks=self.map_blocks,
                    first_block=self.first_block, blocks=self.blocks, max_groups=self.max_groups,
                    hot_groups=self.hot_groups, max_rows=self.max_rows)


def plan_streaming_carve(selector, *, hot_pages, max_rows, staging_buffers=2):
    for name, value in (('hot_pages', hot_pages), ('max_rows', max_rows), ('staging_buffers', staging_buffers)):
        if type(value) is not int or value <= 0:
            raise ValueError(f'QSA streaming carve requires a positive integer {name}')
    if selector.page_tokens % GROUP_TOKENS:
        raise ValueError('QSA page tokens must hold whole compression groups')
    frame_pages = selector.pages_per_slot
    max_groups = frame_pages * selector.page_tokens // GROUP_TOKENS
    hot_groups = hot_pages * selector.page_tokens // GROUP_TOKENS
    plan = residency_map_plan(layers=selector.layers, slots=selector.slots, max_groups=max_groups,
                              hot_groups=hot_groups, hot_pages=hot_pages, max_rows=max_rows)
    return StreamingCarve(selector=selector, hot_pages=hot_pages, frame_pages=frame_pages,
                          staging_buffers=staging_buffers, map_bytes=residency_map_bytes(plan),
                          max_groups=max_groups, hot_groups=hot_groups, max_rows=max_rows)


class MapAllocator:
    """Bump allocator over a raw uint8 CUDA view, following ``residency_map_plan``."""

    def __init__(self, raw, plan):
        self.raw, self.plan, self.cursor, self.index, self.views = raw, plan, 0, 0, {}

    def allocate_all(self):
        import torch
        for key, shape, dtype in self.plan:
            count = 1
            for extent in shape:
                count *= extent
            nbytes = count * _DTYPE_BYTES[dtype]
            self.cursor = -(-self.cursor // MAP_ALIGNMENT) * MAP_ALIGNMENT
            if self.cursor + nbytes > self.raw.numel():
                raise ValueError('QSA residency maps exceed their reserved blocks')
            view = self.raw[self.cursor:self.cursor + nbytes].view(getattr(torch, dtype)).view(shape)
            self.views[key] = view
            self.cursor += nbytes
        return self.views


def place_carves(selector, streaming, num_blocks):
    """Place the carves at the end of a ``num_blocks`` pool: [staging | selector | maps].

    Returns (selector, streaming, main_end): QSA main K/V views must stop at
    ``main_end`` (the first selector block); staging pages lie below it.
    """
    tail = selector.blocks + (streaming.blocks if streaming is not None else 0)
    start = num_blocks - tail
    if type(num_blocks) is not int or start <= NULL_BLOCKS:
        raise ValueError('The native pool cannot hold the QSA carves')
    staging = streaming.staging_blocks if streaming is not None else 0
    placed = replace(selector, first_block=start + staging)
    if streaming is None:
        return placed, None, placed.first_block
    stream = replace(streaming, selector=placed, staging_first=start, maps_first=placed.end_block)
    if stream.end_block != num_blocks:
        raise AssertionError('QSA carve placement must end at the pool end')
    return placed, stream, placed.first_block


def carve_range(selector, streaming=None):
    """[first, end) of every carved block, for the scheduler's one-time reservation."""
    first = streaming.first_block if streaming is not None else selector.first_block
    end = streaming.end_block if streaming is not None else selector.end_block
    return first, end
