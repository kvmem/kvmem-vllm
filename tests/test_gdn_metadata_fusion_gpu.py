"""GPU check: fused GDN group metadata writes exactly what vLLM's update_block_table writes.

Real GDNAttentionMetadataBuilder / B12xGdnMixedMetadata objects (explicit opt-in).
"""
import copy
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='explicit GPU opt-in')

MAX_TOKENS, MAX_SEQS, COLUMNS, GRAPH_BS = 2112, 16, 4, 64


def _random_like(tensor, generator):
    import torch
    if tensor.dtype == torch.bool:
        return torch.randint(0, 2, tensor.shape, generator=generator).bool().to(tensor.device)
    return torch.randint(-5, 900, tensor.shape, generator=generator).to(tensor.dtype).to(tensor.device)


def _builder(gdn, mixed_cls, generator):
    import torch
    device = 'cuda'
    b = gdn.GDNAttentionMetadataBuilder.__new__(gdn.GDNAttentionMetadataBuilder)
    b.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode='none'))
    b.kv_cache_spec = None
    b._b12x_mixed = mixed_cls(max_tokens=MAX_TOKENS, max_seqs=MAX_SEQS, state_columns=COLUMNS, device=device)
    m = b._b12x_mixed
    for name in ('query_start_loc', 'state_indices', 'has_initial_state', 'live_counts', 'token_indices',
                 'request_rows', 'spec_query_start_loc', 'spec_state_indices', 'spec_accepted', 'spec_counts',
                 'spec_token_indices', 'spec_request_rows', 'checkpoint_columns', 'batch_ptr',
                 'token_chunk_offset_ptr'):
        getattr(m, name).copy_(_random_like(getattr(m, name), generator))
    for name in ('state_indices', 'checkpoint_offsets'):
        getattr(m.checkpoint, name).copy_(_random_like(getattr(m.checkpoint, name), generator))
    b._b12x_prefill_live_counts = _random_like(torch.zeros(2, dtype=torch.int32, device=device), generator)
    b.mamba_aligned_state_indices = None
    b._reuse_spec_decode_inputs = True
    b.use_spec_decode, b.num_spec, b.state_index_columns = True, COLUMNS - 1, COLUMNS
    b.use_full_cuda_graph, b.decode_cudagraph_max_bs = True, GRAPH_BS
    b._decode_state_indices_source = b._decode_state_indices_view = None
    for name, shape, dtype in (('spec_state_indices_tensor', (GRAPH_BS, COLUMNS), torch.int32),
                               ('non_spec_state_indices_tensor', (GRAPH_BS,), torch.int32),
                               ('spec_sequence_masks', (GRAPH_BS,), torch.bool),
                               ('spec_token_indx', (GRAPH_BS * COLUMNS,), torch.int32),
                               ('non_spec_token_indx', (GRAPH_BS * COLUMNS,), torch.int32),
                               ('spec_query_start_loc', (GRAPH_BS + 1,), torch.int32),
                               ('non_spec_query_start_loc', (GRAPH_BS + 1,), torch.int32),
                               ('num_accepted_tokens', (GRAPH_BS,), torch.int32)):
        setattr(b, name, _random_like(torch.zeros(shape, dtype=dtype, device=device), generator))
    return b


def _clone(builders):
    out = []
    for b in builders:
        c = copy.copy(b)
        c.__dict__ = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in b.__dict__.items()}
        m = copy.copy(b._b12x_mixed)
        m.__dict__ = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in b._b12x_mixed.__dict__.items()}
        m.checkpoint = SimpleNamespace(state_indices=b._b12x_mixed.checkpoint.state_indices.clone(),
                                       checkpoint_offsets=b._b12x_mixed.checkpoint.checkpoint_offsets.clone())
        m._worklists = tuple(getattr(m, name) if name != 'checkpoint_offsets' else m.checkpoint.checkpoint_offsets
                             for name in ('query_start_loc', 'has_initial_state', 'live_counts', 'token_indices',
                                          'request_rows', 'spec_query_start_loc', 'spec_accepted', 'spec_counts',
                                          'spec_token_indices', 'spec_request_rows', 'checkpoint_columns',
                                          'batch_ptr', 'token_chunk_offset_ptr', 'checkpoint_offsets'))
        c._b12x_mixed = m
        out.append(c)
    return out


def _source_metadata(gdn, first, n, num_non_spec, generator):
    import torch
    m = first._b12x_mixed
    m._num_non_spec, m._num_spec = num_non_spec, n
    m.request_rows.copy_(torch.randint(0, n, m.request_rows.shape, generator=generator).cuda())
    m.spec_request_rows.copy_(torch.randint(0, n, m.spec_request_rows.shape, generator=generator).cuda())
    m.checkpoint_columns.copy_(torch.randint(0, 8, m.checkpoint_columns.shape, generator=generator).cuda())
    m.checkpoint.checkpoint_offsets.copy_(torch.randint(0, 2, m.checkpoint.checkpoint_offsets.shape,
                                                        generator=generator).int().cuda())
    first.spec_sequence_masks[:n] = True
    return gdn.GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0, num_spec_decodes=n,
        num_spec_decode_tokens=COLUMNS * n, num_actual_tokens=COLUMNS * n,
        spec_query_start_loc=first.spec_query_start_loc[:n + 1],
        non_spec_query_start_loc=first.non_spec_query_start_loc[:1],
        spec_state_indices_tensor=first.spec_state_indices_tensor[:n],
        spec_sequence_masks=first.spec_sequence_masks[:n], spec_sequence_masks_cpu=torch.ones(n, dtype=torch.bool),
        spec_token_indx=first.spec_token_indx[:COLUMNS * n], non_spec_token_indx=first.non_spec_token_indx[:0],
        num_accepted_tokens=first.num_accepted_tokens[:n], b12x_mixed=m,
        b12x_prefill_live_counts=first._b12x_prefill_live_counts, num_reqs=n,
        seq_lens=torch.arange(n, device='cuda'), prefill_checkpoint=None, is_uniform_spec_decode=False)


def _tensors(builder):
    m = builder._b12x_mixed
    out = {k: v for k, v in builder.__dict__.items() if hasattr(v, 'data_ptr') and k != '_b12x_mixed'}
    out.update({'mixed.' + k: v for k, v in m.__dict__.items() if hasattr(v, 'data_ptr')})
    out['mixed.checkpoint.state_indices'] = m.checkpoint.state_indices
    out['mixed.checkpoint.checkpoint_offsets'] = m.checkpoint.checkpoint_offsets
    return out


@pytest.mark.parametrize('n,num_non_spec', [(16, 0), (5, 0), (9, 3)])
def test_fused_update_matches_vendor_update_block_table(n, num_non_spec):
    import torch
    import importlib
    gdn = importlib.import_module('vllm.v1.attention.backends.gdn_attn')
    from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata
    from vllm_kvmem.adapters import gdn_metadata_fusion as fusion
    generator = torch.Generator().manual_seed(n * 7 + num_non_spec)
    groups = 6
    builders = [_builder(gdn, B12xGdnMixedMetadata, generator) for _ in range(groups)]
    metadata = _source_metadata(gdn, builders[0], n, num_non_spec, generator)
    tables = torch.randint(1, 4000, (groups, 24, 8), generator=generator).int().cuda()
    vendor_set, fused_set = builders, _clone(builders)
    fused_metadata = gdn.GDNAttentionMetadata(**{**metadata.__dict__, 'b12x_mixed': fused_set[0]._b12x_mixed,
        'spec_query_start_loc': fused_set[0].spec_query_start_loc[:n + 1],
        'spec_sequence_masks': fused_set[0].spec_sequence_masks[:n],
        'spec_token_indx': fused_set[0].spec_token_indx[:COLUMNS * n],
        'non_spec_token_indx': fused_set[0].non_spec_token_indx[:0],
        'num_accepted_tokens': fused_set[0].num_accepted_tokens[:n],
        'b12x_prefill_live_counts': fused_set[0]._b12x_prefill_live_counts})
    update = fusion._INSTALLED.get('update', gdn.GDNAttentionMetadataBuilder.update_block_table)
    want = [update(b, metadata, tables[g, :n], None) for g, b in enumerate(vendor_set) if g]
    fusion._STATE.pending = []
    got = []
    for g, b in enumerate(fused_set):
        if g:
            assert fusion.eligible(b, fused_metadata, tables[g, :n])
            got.append(fusion.defer(b, fused_metadata, tables[g, :n]))
    fusion.flush()
    torch.cuda.synchronize()
    for g in range(1, groups):
        a, b = _tensors(vendor_set[g]), _tensors(fused_set[g])
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), (g, key)
        assert (vendor_set[g]._b12x_mixed._num_non_spec, vendor_set[g]._b12x_mixed._num_spec) == \
            (fused_set[g]._b12x_mixed._num_non_spec, fused_set[g]._b12x_mixed._num_spec)
    for w, f in zip(want, got):
        for field in w.__dataclass_fields__:
            x, y = getattr(w, field), getattr(f, field)
            if hasattr(x, 'data_ptr'):
                assert x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y), field
            elif field not in ('b12x_mixed',):
                assert type(x) is type(y) and (x == y if not hasattr(x, '__dict__') else True), field


def test_ineligible_batches_keep_the_vendor_path():
    import torch
    import importlib
    gdn = importlib.import_module('vllm.v1.attention.backends.gdn_attn')
    from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata
    from vllm_kvmem.adapters import gdn_metadata_fusion as fusion
    generator = torch.Generator().manual_seed(3)
    builders = [_builder(gdn, B12xGdnMixedMetadata, generator) for _ in range(2)]
    metadata = _source_metadata(gdn, builders[0], 8, 0, generator)
    table = torch.randint(1, 4000, (8, 8), generator=generator).int().cuda()
    fusion._STATE.mask_cache = None
    assert fusion.eligible(builders[1], metadata, table)
    for change in (dict(num_prefills=1), dict(num_decodes=1), dict(is_uniform_spec_decode=True),
                   dict(spec_sequence_masks_cpu=torch.tensor([True] * 7 + [False]))):
        fusion._STATE.mask_cache = None
        assert not fusion.eligible(builders[1], gdn.GDNAttentionMetadata(**{**metadata.__dict__, **change}), table)
    builders[1].vllm_config.cache_config.mamba_cache_mode = 'align'
    fusion._STATE.mask_cache = None
    assert not fusion.eligible(builders[1], metadata, table)
