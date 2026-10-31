"""Collect private Query quanta without a second full-size concatenation."""


def append_query_chunk(job, query, lo, hi, total_tokens):
    """Own copied Q values; capture buffers may be reused by the next forward.

    The caller submits copies on its current stream. Existing restore/assembly
    fences protect this storage exactly as they protected the old chunk clones.
    Chunk views retain the original reporting and guided-fallback interface.
    """
    if not 0 <= lo < hi <= query.shape[1] or total_tokens <= 0:
        raise ValueError('Invalid private Query capture extent')
    buffer = getattr(job, 'query_buffer', None)
    written = getattr(job, 'query_buffer_tokens', 0)
    count = hi - lo
    shape = (query.shape[0], total_tokens, *query.shape[2:])
    if written + count > total_tokens:
        raise ValueError('Private Query capture exceeds its logical span')
    if buffer is None:
        if written or job.query_rows:
            raise ValueError('Private Query capture lost its storage owner')
        buffer = query.new_empty(shape)
        job.query_buffer = buffer
    elif buffer.shape != shape or buffer.dtype != query.dtype or buffer.device != query.device:
        raise ValueError('Private Query capture geometry changed')
    chunk = buffer[:, written:written+count]
    chunk.copy_(query[:, lo:hi])
    job.query_rows.append(chunk)
    job.query_buffer_tokens = written + count


def query_for_submit(job):
    import torch
    if job.guided_rows:
        return torch.cat(job.guided_rows, dim=1)
    buffer = getattr(job, 'query_buffer', None)
    if buffer is not None:
        if (job.query_buffer_tokens != buffer.shape[1] or
                sum(row.shape[1] for row in job.query_rows) != buffer.shape[1]):
            raise ValueError('Private Query capture is incomplete')
        return buffer
    return torch.cat(job.query_rows, dim=1) if job.query_rows else None


def release_query_buffer(job):
    # Called only by the existing completion/cancellation retirement path.
    job.query_buffer = None
    job.query_buffer_tokens = 0
