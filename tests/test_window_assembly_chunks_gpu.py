"""Explicit GPU qualification of assembly batching, with unchanged FP8 bytes."""
import math
import os

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


@pytest.mark.parametrize('candidate_chunk', [4, 16, 128, 256])
def test_chunks_match_32_across_resident_moves_partial_tail_and_restore(tmp_path, candidate_chunk):
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.codec import native_vllm_fp8_views
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler

    block, layers, heads, dim, page = 32, 2, 2, 64, 16
    count = 264 * block + 7
    cfg = KVMemConfig(block_tokens=block, select_budget=131*block, gen_budget=135*block,
                      repository_blocks=512, sink_blocks=0, recent_blocks=0)
    record_bytes = layers * block * heads * dim * 2
    archive = MemoryArchive(str(tmp_path), record_bytes, 512)
    owner = RequestAuthority('chunk-equivalence', cfg, archive, layers, heads, dim,
                              raw_dtype=torch.float8_e4m3fn, resident_records=300)
    torch.manual_seed(10971)
    raw = torch.randn(layers, count, heads, dim).to(torch.float8_e4m3fn)
    encoded = torch.randint(0, 127, raw.shape, dtype=torch.uint8)
    assert owner.append(raw, encoded, history_begin=0, local_rope_begin=0)
    owner.publish_sealed(lambda: archive.wait_for_capacity(.05))
    page_count = math.ceil(count / page)
    table = (torch.randperm(page_count, device='cuda') + 1).to(torch.int32)
    angles = torch.randn((page_count+1)*page, 16, device='cuda')
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(count, device='cuda')
    slots = table[positions//page].long()*page + positions%page
    references, candidates = [], []
    for i in range(layers):
        backing = torch.zeros((page_count+1, heads, page, 2*dim), dtype=torch.uint8, device='cuda')
        key, value = native_vllm_fp8_views(backing, heads, dim)
        scale = torch.tensor(.04052734375 + i*.00390625, device='cuda')
        assemble_fp8(raw[i].cuda(), encoded[i].view(torch.float8_e4m3fn).cuda(),
                     positions, slots, trig, key, value, scale)
        other_key, other_value = native_vllm_fp8_views(backing.clone(), heads, dim)
        references.append(LayerWindow(key, value, trig, scale, table))
        candidates.append(LayerWindow(other_key, other_value, trig, scale, table))
    chosen = set(range(1, 260, 2)) | {264}
    scores = {r.handle.slot: float(r.history_begin//block in chosen) for r in owner.records.values()}
    plan = owner.memory.select(scores, action=False)
    selected = tuple((r.block_id, r.n_tokens, r.to_base) for r in plan.remaps)
    assert len(selected) == 131 and selected[-1][1] == 7
    for restoring in (False, True):
        if restoring:
            checkpoint = owner.memory.checkpoint_window(allow_partial=True)
            plan = owner.memory.restore_window(checkpoint)
            # A different native allocation must reload cold V; no old page
            # values or old table row may be reused by either chunk size.
            fresh_table = table.flip(0).contiguous()
            for layer in references + candidates:
                layer.kernel_page_table = fresh_table
                layer.key.view(torch.uint8).fill_(199)
                layer.value.view(torch.uint8).fill_(199)
        assemblers = [NativeWindowAssembler(tuple(frame), block, chunk_blocks=chunk,
            packed=True, direct_pinned_reads=True, reuse_resident_v=True,
            raw_dtype=torch.float8_e4m3fn)
            for frame, chunk in ((references, 32), (candidates, candidate_chunk))]
        baseline = assemblers[0].assemble(owner, plan)
        optimized = assemblers[1].assemble(owner, plan)
        baseline.event.synchronize()
        optimized.event.synchronize()
        assert tuple((r.block_id, r.n_tokens, r.to_base) for r in plan.remaps) == selected
        for left, right in zip(references, candidates):
            assert torch.equal(left.key.view(torch.uint8), right.key.view(torch.uint8))
            assert torch.equal(left.value.view(torch.uint8), right.value.view(torch.uint8))
        assert optimized.rebuilt_tokens == baseline.rebuilt_tokens == 130*block+7
        assert optimized.partial_tokens == baseline.partial_tokens == 7
        assert optimized.uploaded_bytes == baseline.uploaded_bytes
        assert optimized.resident_v_reused_tokens == baseline.resident_v_reused_tokens == (0 if restoring else 130*block)
        assert (optimized.chunks < baseline.chunks) == (candidate_chunk > 32)
        owner.memory.attach_completion(optimized)
        assert owner.memory.commit_selection()
        assert all(assembler.close() for assembler in assemblers)
        for assembler in assemblers:
            assert assembler.staging.close()
        del plan, baseline, optimized
        if restoring:
            del checkpoint
    assert owner.close() and archive.close()
