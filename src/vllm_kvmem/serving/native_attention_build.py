"""Isolated, source-checked FlashInfer builds; installed packages are never edited."""
from functools import cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile


class UnsupportedNativeStats(RuntimeError):
    pass


_STATIC_ALIGNMENT_SOURCES = {
    'flashinfer/attention/scheduler.cuh': 'a11a39aea393cdb380ad9439aac9170cd96849d7426ebbb920c5c8cdadfcc575',
    'flashinfer/attention/variants.cuh': '90a228dd71b3ce8030d663e2b16f67f840fe4a3e99912f8b47774d3fd905a080',
}


def _check_static_alignment_sources(include):
    # Removing split-piece CAS depends on page-based split sizes and the
    # default no-window interpretation. Keep this extra contract local to the
    # optional specialization so drift can fall back to generic native stats.
    for relative, expected in _STATIC_ALIGNMENT_SOURCES.items():
        path = include / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedNativeStats('static_source_mismatch:' + relative)


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise UnsupportedNativeStats('FlashInfer source structure changed')
    return source.replace(before, after)


def _write(path, value):
    data = value.encode() if isinstance(value, str) else value
    if path.exists() and path.read_bytes() == data:
        return
    # Another TP worker may already be compiling from this shared cache.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def _publish_tree(source, path):
    """Copy a directory into a shared cache once; concurrent TP workers may race.

    The copy is built privately and published with one rename, so nobody sees a
    partial tree; a worker that loses the rename discards its own copy.
    """
    if path.is_dir():
        return
    staging = Path(tempfile.mkdtemp(dir=path.parent, prefix='.' + path.name + '-'))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        try:
            os.rename(staging, path)
        except OSError:
            if not path.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def patch_xqa(source, name):
    """Extend the native ABI all the way to its existing post-QK accumulator."""
    if name == 'mha.cu':
        source = _replace(source, '#include "mha.h"', '#include "mha.h"\n#include "native_stats.cuh"')
        source = _replace(source, 'void* __restrict__ scratch = nullptr) {',
                          'void* __restrict__ scratch = nullptr, KVMemStats stats = {}) {', 2)
        source = _replace(source, 'semaphores, scratch);\n}\n#else',
                          'semaphores, scratch, stats);\n}\n#else')
        capture = '''
      // Reuse native, scaled QK. Every quad handles four columns of one parity.
      if (stats.rows != nullptr) {
      #pragma unroll
      for (uint32_t m = 0; m < acc.rows; ++m) {
        #pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; ++i) {
          uint32_t mixed = instM * m + laneId() / 4 + 8 * i;
          #if SPEC_DEC
          mixed += idxHeadTokenInGrp;
          uint32_t row = mixed / headGrpSize;
          if (row >= actualQSeqLen) continue;
          uint32_t globalRow = reqSeqOffset + row;
          uint32_t visible = cacheSeqLen - actualQSeqLen + row + 1;
          #else
          if (mixed >= headGrpSize) continue;
          uint32_t globalRow = idxReq;
          uint32_t visible = cacheSeqLen;
          #endif
          int sample = stats.rows[globalRow];
          if (sample < 0) continue;
          #pragma unroll
          for (uint32_t n = 0; n < acc.cols; n += 2) {
            stats.store16_weight(acc(m, n)(i, 0), acc(m, n)(i, 1),
                acc(m, n+1)(i, 0), acc(m, n+1)(i, 1), regRowMax[m * InstAcc::rows + i],
                warpTileTokenBeg + instN * n, visible, sample,
                idxHeadGrp * headGrpSize + mixed % headGrpSize, false);
          }
        }
      }
      }
'''
        marker = '      QuadRegRowMax const regRowMax = warpTileOnlineSoftmax(warp, initRowMaxQuad, acc);'
        source = _replace(source, marker, marker + capture)
        start = source.index('void launchMHAFlashInfer(')
        prefix, tail = source[:start], source[start:]
        prefix = _replace(prefix, '                     semaphores, scratch);',
                          '                     semaphores, scratch, KVMemStats{});')
        tail = _replace(tail, 'cudaStream_t stream) {', 'KVMemStats stats, cudaStream_t stream) {')
        tail = _replace(tail, 'semaphores, scratch);', 'semaphores, scratch, stats);')
        return prefix + tail
    if name == 'mha.h':
        source = '#include "native_stats.cuh"\n' + source
        start = source.index('void launchMHAFlashInfer(')
        end = source.index(');', start) + 2
        return source[:start] + _replace(source[start:end], 'cudaStream_t stream);',
                                          'KVMemStats stats, cudaStream_t stream);') + source[end:]
    if name in ('xqa_wrapper.cu', 'flashinfer_xqa_binding.cu'):
        start = source.index('void xqa_wrapper(')
        prefix, tail = source[:start], source[start:]
        extra = ('bool enable_pdl, TensorView statRows, TensorView statRequests, '
                 'TensorView statPieces, TensorView statLse, int64_t statContext, int64_t statWidth')
        tail = _replace(tail, 'bool enable_pdl)', extra + ')')
        if name == 'xqa_wrapper.cu':
            # Only the standard warp-specialized SM12x path is enabled by the adapter.
            end = tail.index('  launchMHAFlashInfer(')
            head, launch = tail[:end], tail[end:]
            launch = _replace(launch, '                      stream);', '''
                      KVMemStats{static_cast<int32_t const*>(statRows.data_ptr()),
                                  static_cast<int32_t const*>(statRequests.data_ptr()),
                                  static_cast<int32_t const*>(statPieces.data_ptr()),
                                  static_cast<float*>(statLse.data_ptr()),
                                  int(statContext), int(statWidth), int(nbKHeads * HEAD_GRP_SIZE)},
                      stream);''')
            tail = head + launch
        return prefix + tail
    raise ValueError(name)


def patch_prefill(source, *, cta_gate=False):
    marker = '''          update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        }

        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {'''
    anchor = 'void BatchPrefillWithPagedKVCacheDevice('
    if anchor not in source:
        raise UnsupportedNativeStats('FlashInfer paged prefill structure changed')
    start = source.index(anchor)
    paged = _replace(source[start:], marker, '''          update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        }
        variant.template CaptureSoftmax<KTraits>(params, request_idx, qo_packed_idx_base, kv_idx_base,
            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx);

        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {''')
    if cta_gate:
        # All valid CTA threads reach this point, after the PDL wait and before
        # either VO or KV loop. A packed-Q tile can begin/end within one GQA
        # group: include both boundary Q rows even if only some heads are here.
        gate_marker = '''#endif

    // Split-D over the VO dimension'''
        gate = '''#endif

    const uint32_t kvmem_q_begin = (qo_tile_idx * CTA_TILE_Q) / group_size;
    bool kvmem_local_sampled = false;
    for (uint32_t kvmem_q = kvmem_q_begin + block.thread_rank();
         kvmem_q < qo_upper_bound; kvmem_q += KTraits::NUM_THREADS) {
      kvmem_local_sampled |=
          params.stat_rows[params.stat_offset + params.q_indptr[request_idx] + kvmem_q] >= 0;
    }
    const bool kvmem_cta_sampled = __syncthreads_or(kvmem_local_sampled);

    // Split-D over the VO dimension'''
        paged = _replace(paged, gate_marker, gate)
        call = '''        variant.template CaptureSoftmax<KTraits>(params, request_idx, qo_packed_idx_base, kv_idx_base,
            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx);'''
        paged = _replace(paged, call, '''        if (kvmem_cta_sampled) {
''' + call + '''
        }''')
    return source[:start] + paged


def patch_static_readonly(source):
    """Component-only variant: change two immutable loads, no floating math.

    Keep the checked-in header and all historical overlays unchanged. A unique
    class makes profiler/SASS dispatch attribution independent of module names.
    """
    marker = 'struct KVMemStaticAlignedAttention :'
    if source.count(marker) != 1:
        raise UnsupportedNativeStats('Static readonly class structure changed')
    start = source.index(marker)
    prefix, variant = source[:start], source[start:]
    variant = _replace(variant, 'int sample = stats.rows[p.q_indptr[batch] + q];',
        'int sample = __ldg(stats.rows + __ldg(p.q_indptr + batch) + q);')
    variant = variant.replace('KVMemStaticAlignedAttention', 'KVMemStaticReadonlyAttention')
    return prefix + variant


@cache
def source_overlay(directory, *, prefill_cta_gate=False, prefill_static_aligned=False,
                   prefill_static_readonly=False):
    if prefill_static_readonly and (not prefill_static_aligned or prefill_cta_gate):
        raise ValueError('Readonly metadata requires only the static aligned component')
    from flashinfer.jit import env
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'native_attention_sources.json').read_text())
    roots = {'csrc': env.FLASHINFER_CSRC_DIR, 'include': env.FLASHINFER_INCLUDE_DIR,
             'python': env.FLASHINFER_INCLUDE_DIR.parent.parent}
    for filename, expected in manifest.items():
        root, relative = filename.split('/', 1)
        path = roots[root] / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedNativeStats('source_mismatch:' + filename)
    own = Path(__file__).resolve().parents[1]    # the package root holds the .cuh sources
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    digest.update(inspect.getsource(patch_xqa).encode())
    digest.update(inspect.getsource(patch_prefill).encode())
    for name in ('native_stats.cuh', 'native_prefill_stats.cuh'):
        digest.update((own / name).read_bytes())
    if prefill_cta_gate:
        digest.update(b'prefill_cta_gate_v1')
    if prefill_static_aligned:
        _check_static_alignment_sources(roots['include'])
        digest.update(json.dumps(_STATIC_ALIGNMENT_SOURCES, sort_keys=True).encode())
        digest.update(b'prefill_static_aligned_v1')
    if prefill_static_readonly:
        digest.update(b'prefill_static_readonly_v1')
        digest.update(inspect.getsource(patch_static_readonly).encode())
    # A content-addressed canonical path prevents absolute includes in generated
    # instantiations from invalidating the cache for every new serving run.
    target = env.FLASHINFER_GEN_SRC_DIR / 'kvmem_stats' / digest.hexdigest()[:16]
    target.mkdir(parents=True, exist_ok=True)
    xqa = target / 'xqa'
    _publish_tree(roots['csrc'] / 'xqa', xqa)
    for name in ('mha.cu', 'mha.h', 'xqa_wrapper.cu'):
        _write(xqa / name, patch_xqa((roots['csrc'] / 'xqa' / name).read_text(), name))
    name = 'flashinfer_xqa_binding.cu'
    _write(target / name, patch_xqa((roots['csrc'] / name).read_text(), name))
    # Relative includes retain their original package directory. Only prefill.cuh
    # is overlaid, and generated instantiations include it by absolute path.
    header = (roots['include'] / 'flashinfer/attention/prefill.cuh').read_text()
    header = header.replace('#include "', '#include "' + str(roots['include'] / 'flashinfer/attention') + '/')
    _write(target / 'prefill.cuh', patch_prefill(header, cta_gate=prefill_cta_gate))
    _write(target / 'tvm_ffi_utils.h', (roots['csrc'] / 'tvm_ffi_utils.h').read_bytes())
    for name in ('native_stats.cuh', 'native_prefill_stats.cuh'):
        payload = (own / name).read_bytes()
        if prefill_static_readonly and name == 'native_prefill_stats.cuh':
            payload = patch_static_readonly(payload.decode()).encode()
        _write(target / name, payload)
    _write(xqa / 'native_stats.cuh', (own / 'native_stats.cuh').read_bytes())
    return target


def xqa_module(directory, q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged=False):
    return _xqa_module(q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged)


@cache
def _xqa_module(q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged):
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    overlay = source_overlay(None)
    spec = gen_xqa_module(q_dtype, kv_dtype, page, dim, group, False, out_dtype, q_len, ragged)
    flags = [flag for flag in spec.extra_cuda_cflags if not flag.startswith('-DUSE_SM90_MHA=')]
    sources = [overlay / 'xqa/mha.cu', overlay / 'xqa/xqa_wrapper.cu', overlay / 'flashinfer_xqa_binding.cu']
    return gen_jit_spec('kvmem_' + overlay.name + '_' + spec.name, sources,
        extra_cuda_cflags=flags + ['-DUSE_SM90_MHA=0'], extra_ldflags=['-lcuda']).build_and_load()


def prefill_module(directory, q_dtype, kv_dtype, out_dtype, idtype, dim, *, aligned_pieces=False,
                   cta_gate=False, static_aligned=False, static_readonly=False, quad_store=False):
    return _prefill_module(q_dtype, kv_dtype, out_dtype, idtype, dim, aligned_pieces, cta_gate,
                          static_aligned, static_readonly, quad_store)


@cache
def _prefill_module(q_dtype, kv_dtype, out_dtype, idtype, dim, aligned_pieces=False, cta_gate=False,
                    static_aligned=False, static_readonly=False, quad_store=False):
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    if quad_store and (not static_aligned or static_readonly):
        raise ValueError('Quad statistics stores require only the static aligned component')
    if static_readonly and (not static_aligned or aligned_pieces or cta_gate):
        raise ValueError('Readonly metadata requires only the static aligned component')
    if static_aligned and (aligned_pieces or cta_gate):
        raise ValueError('Static aligned FA2 cannot combine runtime aligned pieces or CTA gate')
    overlay = (source_overlay(None, prefill_static_aligned=True, prefill_static_readonly=True) if static_readonly else
               source_overlay(None, prefill_static_aligned=True) if static_aligned else
               source_overlay(None, prefill_cta_gate=True) if cta_gate else source_overlay(None))
    name = f'kvmem_fa2_{overlay.name}_{q_dtype}_{kv_dtype}_{out_dtype}_{idtype}_{dim}'.replace('torch.', '')
    if cta_gate:
        name += '_cta_gate'
    tensors = ['stat_rows', 'stat_requests', 'stat_pieces', 'stat_lse']
    tensor_types = ['int32_t', 'int32_t', 'int32_t', 'float']
    variant = 'KVMemAttention'
    if aligned_pieces:
        name += '_aligned16'
        tensors.append('stat_aligned')
        tensor_types.append('int32_t')
        variant = 'KVMemAlignedAttention'
    if static_aligned:
        name += '_static_aligned16'
        variant = 'KVMemStaticAlignedAttention'
    if quad_store:
        name += '_quad_store'
        variant = 'KVMemStaticQuadAttention'
    if static_readonly:
        name += '_readonly_metadata'
        variant = 'KVMemStaticReadonlyAttention'
    spec = gen_customize_batch_prefill_module('fa2', name, q_dtype, kv_dtype, out_dtype, idtype, dim, dim,
        tensors, tensor_types,
        ['stat_context', 'stat_width', 'stat_offset', 'sm_scale'], ['int32_t', 'int32_t', 'int32_t', 'double'],
        variant, '#include "' + str(overlay / 'native_prefill_stats.cuh') + '"')
    # The kernel header is included before the generated configuration.
    # Its dependent Capture call resolves only for this custom variant.
    generated = overlay / name
    generated.mkdir(exist_ok=True)
    sources = []
    for path in spec.sources:
        original = path.read_text()
        destination = generated / path.name
        _write(destination, original.replace('#include <flashinfer/attention/prefill.cuh>',
                                            '#include "' + str(overlay / 'prefill.cuh') + '"'))
        sources.append(destination)
    _write(generated / 'batch_prefill_config.inc', (spec.sources[0].parent / 'batch_prefill_config.inc').read_bytes())
    spec.sources = sources
    module = spec.build_and_load()
    if static_readonly:
        from .native_prefill_readonly import CheckedReadonlyPrefillModule
        return CheckedReadonlyPrefillModule(module, library_path=spec.get_library_path(), overlay=overlay)
    return module
