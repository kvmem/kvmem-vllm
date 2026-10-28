"""Explicit GPU scoring gate with real compact historical-LRU ownership."""
import os

import pytest
import torch

from test_historical_hybrid_integration import harness, append, advance
from vllm_kvmem.authority import query_score_frame, _Ready
from vllm_kvmem.query_key_fps import select_key_representatives, key_fps_scores
from vllm_kvmem.query_hybrid_recall import preview_hybrid_selection


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='explicit GPU opt-in required')
def test_native_cuda_mean_fps_and_picker_over_recycled_compact_history(tmp_path, monkeypatch):
    from vllm_kvmem.cuda_scoring import mean_key_scores as native_scores
    with harness(tmp_path, monkeypatch) as (authority, archive, worker, manager, expected):
        # This gate runs the actual CUDA MeanK kernel, unlike the CPU suite.
        monkeypatch.setattr('vllm_kvmem.cuda_scoring.mean_key_scores', native_scores)
        append(authority, expected, 384, 0)
        q = torch.randn(1, 5, 2, 8, generator=torch.Generator().manual_seed(303)).cuda()
        for _ in range(5):
            assert manager.wait_idle(10)
            with authority.index_snapshot() as (records, _):
                raw = torch.stack([expected[r.history_begin][0][:1] for r in records], dim=1)
                representatives, offsets = select_key_representatives(raw, [32]*len(records))
                means = raw.float().mean(2).half().cuda()
                with query_score_frame(authority.memory, records) as frame:
                    mask = torch.tensor(frame.score_mask, dtype=torch.bool, device='cuda')
                    fps = key_fps_scores(q, representatives, offsets, mask,
                                         query_half=True, optimized=True).cpu().tolist()
                    mean = native_scores(q, means, mask).cpu().tolist()
                reference = preview_hybrid_selection(authority.memory, records, fps, mean,
                    mean_fraction=.5, message_spans=authority.query_boundaries.message_spans,
                    message_metadata_present=authority.query_boundaries.has_metadata)
                plan = authority.select_for_queries(q)
                assert {row.block_id for row in plan.remaps} == set(reference['selected_slots'])
                assert plan.total_window_tokens <= authority.memory.config.select_budget
                assert authority.query_quota_propagation is None
                torch.cuda.synchronize()
                local = plan.total_window_tokens
                authority.memory.attach_completion(_Ready())
                assert authority.memory.commit_selection()
                del records, plan, raw, representatives, offsets, means
            authority.gc.scope.request_capacity(8)
            advance(authority, worker)
            assert len(authority.records) == 8
            authority.gc.scope.request_capacity(0)
            append(authority, expected, 128, local)
        assert authority.gc_reclaimed_records == 20
        torch.cuda.synchronize()
