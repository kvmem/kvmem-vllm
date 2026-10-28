"""A split prepare is fenced across public turns and keeps cancellation exact."""
import pytest

from test_async_restore import two_lane_manager


@pytest.mark.parametrize('cancelled', [False, True])
def test_split_frame_survives_public_turn_and_only_publishes_after_dma(monkeypatch, tmp_path, cancelled):
    manager, checkpoints, claims, published, fences, release, hot_started = two_lane_manager(monkeypatch, tmp_path)
    manager.options.restore_prepare_split = True
    cold, hot = manager.jobs['cold'], manager.jobs['hot']
    try:
        assert manager.step(cold) and cold.stage == 'restore_model_prepare'
        assert cold.future is None and not published
        assert manager.step(hot) and hot.stage == 'restore_model_prepare'
        hot.cancelled = cancelled
        assert manager.step(hot) and hot_started.wait(2)
        hot.future.result(timeout=2)
        assert not manager.step(hot) and not published
        fences['hot'].done = True
        assert manager.step(hot) and hot.done
        assert published == ['hot'] and not cold.done
        # The cancelled import still commits its exact accepted checkpoint for
        # safe retirement; no public generation occurs for the cancelled owner.
        assert hot.restore_plan is None
        assert manager.step(cold)
    finally:
        release.set()
        for fence in fences.values():fence.done = True
        manager.shutdown()
