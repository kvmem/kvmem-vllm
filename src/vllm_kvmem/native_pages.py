"""Translate native persistent tables to the pages used by bound KV tensors.

Older runners store kernel pages; newer runners store allocator pages and let
attention builders expand them. Keep scheduler witnesses in the stored layout
and direct KV readers in the kernel layout. Never infer layout from a version.
"""


def stored_expansions(tables):
    if getattr(tables, 'redirect_writes_to_null_block', False):
        raise ValueError('KVMem cannot borrow redirected warmup block tables')
    values = getattr(tables, 'blocks_per_kv_block', None)
    if values is None:
        sizes = getattr(tables, 'block_sizes', None)
        if sizes is None or len(sizes) != len(tables.block_tables):
            raise ValueError('Unsupported native persistent block-table layout')
        values = [1] * len(sizes)
    if len(values) != len(tables.block_tables) or any(type(n) is not int or n <= 0 for n in values):
        raise ValueError('Invalid native block-table expansion')
    return tuple(values)


def kernel_block_size(runner, group):
    sizes = getattr(runner.block_tables, 'kernel_block_sizes', None)
    if sizes is None:
        sizes = runner.kernel_block_sizes
    size = sizes[group]
    if type(size) is not int or size <= 0:
        raise ValueError('Native attention group has no kernel block size')
    return size


def kernel_expansions(runner):
    tables = runner.block_tables
    stored = stored_expansions(tables)
    if hasattr(tables, 'blocks_per_kv_block'):
        return stored
    result = []
    for group, size in enumerate(tables.block_sizes):
        kernel = runner.kernel_block_sizes[group]
        # Recurrent groups have state rows, not attention kernel blocks.
        kernel = size if kernel is None else kernel
        if type(kernel) is not int or kernel <= 0 or size % kernel:
            raise ValueError('Native allocator and kernel page sizes are incompatible')
        for attention in getattr(runner, 'attn_groups', [()] * len(tables.block_sizes))[group]:
            stride = getattr(attention, 'kernel_block_stride', 0)
            selected = getattr(attention, 'kernel_block_size', None)
            if stride not in (0, size // kernel) or selected not in (None, kernel):
                raise ValueError('Packed per-layer kernel pages need a separate KVMem layout adapter')
        result.append(size // kernel)
    return tuple(result)


def kernel_table(runner, group, index):
    tables = runner.block_tables
    count = int(tables.num_blocks.np[group, index])
    row = tables.block_tables[group].gpu[index, :count]
    if hasattr(tables, 'kernel_block_sizes'):
        if getattr(tables, 'redirect_writes_to_null_block', False):
            raise ValueError('KVMem cannot borrow redirected warmup block tables')
        return row
    stored = stored_expansions(tables)[group]
    expansion = kernel_expansions(runner)[group]
    if expansion == stored:
        return row
    if stored != 1:
        raise ValueError('Unsupported mixed native page layout')
    import torch
    offsets = torch.arange(expansion, dtype=row.dtype, device=row.device)
    # Match the native mapping, including all subpages of the null allocator page.
    return (row[:, None] * expansion + offsets).reshape(-1)
