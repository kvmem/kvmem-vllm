"""Original bounded qw3 FP16 page-softmax kernel, on the current Torch stream."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
import shutil

import torch


_SOURCE_NAMES = ('qw3_mean_k.cu', 'qw3_mean_k_kernel.cuh', 'qw3_mean_k_tiled.cuh')
QUERY_CONVERSION_BYTES = 64 * 1024 * 1024


def _publish_source(path, content):
    """Publish one complete immutable source without touching an existing copy."""
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        pass
    else:
        if existing != content:
            raise RuntimeError(f'MeanK cached source content differs: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Another process may have published these same bytes meanwhile.
            # Unlike replace(), link-if-absent preserves its inode and mtime.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f'MeanK cached source content differs: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_build_sources(source, build_root):
    # Snapshot all inputs before publishing any of them. Including boundaries
    # prevents different distributions across the three files sharing a key.
    contents = [(name, (source / name).read_bytes()) for name in _SOURCE_NAMES]
    key = hashlib.sha256()
    for name, content in contents:
        key.update(name.encode() + b'\0' + len(content).to_bytes(8, 'big') + content)
    digest = key.hexdigest()
    directory = build_root / ('kvmem_meank_' + digest[:12])
    # Keep the full content key in the input path even if module names share a
    # truncated prefix. All quoted local headers remain adjacent to the .cu.
    stable = directory / 'sources' / digest
    stable.mkdir(parents=True, exist_ok=True)
    for name, content in contents:
        _publish_source(stable / name, content)
    return directory, stable / _SOURCE_NAMES[0], digest[:12]


@lru_cache(maxsize=1)
def kernel():
    if shutil.which('ninja') is None:
        import ninja
        os.environ['PATH'] = str(ninja.BIN_DIR) + os.pathsep + os.environ.get('PATH','')
    os.environ.setdefault('MAX_JOBS','1')
    from torch.utils.cpp_extension import load
    source = Path(__file__).parent / 'cuda_sources'
    if not source.exists():
        source = Path(__file__).resolve().parents[2] / 'csrc'
    directory, stable_source, digest = _prepare_build_sources(source,
        Path(os.getenv('KVMEM_CUDA_BUILD_DIR', tempfile.gettempdir())).resolve())
    return load(name='kvmem_meank_' + digest, sources=[str(stable_source)],
        build_directory=str(directory), extra_cuda_cflags=['-O3'], verbose=False)


def mean_key_scores(queries, means, eligible, *, scale=None):
    if queries.ndim != 4 or means.ndim != 4 or eligible.shape != (means.shape[1],) or eligible.dtype != torch.bool:
        raise ValueError('Invalid MeanK Query/index/mask')
    if queries.device != means.device or eligible.device != queries.device:
        raise ValueError('MeanK tensors must share a device')
    if any(size <= 0 for size in queries.shape):
        raise ValueError('MeanK queries must have nonempty geometry')
    indices = eligible.nonzero().flatten()
    scores = torch.zeros(means.shape[1], device=queries.device, dtype=torch.float32)
    if not indices.numel():
        return scores
    # Compact only eligible pages before softmax, exactly excluding sink/recent
    # and mandatory pages. Scatter scores back without changing history order.
    compact_means = means.index_select(1, indices).to(
        dtype=torch.float16, memory_format=torch.contiguous_format)
    attention_scale = queries.shape[-1] ** -.5 if scale is None else scale
    scorer = kernel()
    if queries.dtype == torch.float16 and queries.is_contiguous():
        values = scorer.score(queries, compact_means, attention_scale)
    else:
        # The kernel sums over Query tokens and averages only layers/heads.
        # Keep its full page softmax domain while bounding dtype/layout copies;
        # a 2304-token BF16 Query otherwise duplicates 432 MiB at handoff.
        bytes_per_token = queries.shape[0] * queries.shape[2] * queries.shape[3] * 2
        chunk_tokens = max(1, QUERY_CONVERSION_BYTES // bytes_per_token)
        values = torch.zeros(indices.numel(), device=queries.device, dtype=torch.float32)
        for begin in range(0, queries.shape[1], chunk_tokens):
            chunk = queries[:, begin:begin+chunk_tokens].to(
                dtype=torch.float16, memory_format=torch.contiguous_format)
            values.add_(scorer.score(chunk, compact_means, attention_scale))
            del chunk
    scores.index_copy_(0, indices, values)
    return scores
