"""Native feature publication and witnessed GC through the real CPU archive."""
import gc
import time
import weakref

import numpy as np
import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.memory_storage import MemoryArchive


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


@pytest.fixture
def history(tmp_path):
    config = n.HistoricalGcConfig()
    config.policy = 'redundancy'
    config.mode = n.HistoricalGcMode.DELETE
    config.background = False
    config.max_scopes = 1
    config.scan_blocks = 8
    config.delete_blocks = 2
    config.ready_low, config.ready_high = 2, 4
    archive = MemoryArchive(str(tmp_path), 8, 16, native_payloads=True)
    worker = archive.enable_history_gc(config)
    repo = n.Repository(8, 4)
    scope = archive.register_history_scope('a', repo)
    scope.pause_policy(True)
    leases = [repo.reserve(4 * i, 4).commit() for i in range(8)]
    archive.write('a', leases, b'original' * 8).result(5)
    for lease in leases:
        assert scope.update(lease, n.HISTORY_SEALED | n.HISTORY_GPU)
    del lease
    try:
        yield archive, worker, repo, scope, leases
    finally:
        leases.clear()
        eventually(archive.close)


def advance(archive, worker, scope, count=64):
    for _ in range(count):
        worker.advance()
        for event in scope.completions(8):
            if event.cpu and not archive.retire_cpu_record('a', event.handle):
                continue
            assert scope.acknowledge(event)


def complete_features(scope, leases):
    assert scope.configure_features(1, 1, 2, 2)
    key = np.array([[[1., 0.]]], dtype=np.float32)
    for lease in leases:
        assert scope.publish_features(lease, key)
    return key


def test_features_are_native_immutable_copies_not_python_buffer_owners(history):
    _, _, _, scope, leases = history
    key = complete_features(scope, leases)
    original = key.copy()
    reference = weakref.ref(key)
    key[:] = 2
    assert not scope.publish_features(leases[0], key)
    del key
    gc.collect()
    assert reference() is None
    assert scope.publish_features(leases[0], original)
    assert scope.feature_stats['keys'] == 8


@pytest.mark.parametrize('buffer', [
    np.ones((1, 1, 2), dtype=np.float64),
    np.ones((1, 2), dtype=np.float32),
    np.ones((1, 1, 3), dtype=np.float32),
    np.ones((1, 1, 4), dtype=np.float32)[..., ::2],
    np.ones((1, 1, 2), dtype=np.float32)[..., ::-1],
])
def test_feature_binding_rejects_wrong_dtype_shape_and_strides(history, buffer):
    _, _, _, scope, leases = history
    assert scope.configure_features(1, 1, 2, 2)
    with pytest.raises(ValueError, match='float32|C-contiguous'):
        scope.publish_features(leases[0], buffer)
    assert scope.feature_stats['keys'] == 0


def test_feature_publication_rejects_foreign_or_retired_handles(history):
    archive, worker, repo, scope, leases = history
    key = complete_features(scope, leases)
    foreign_repo = n.Repository(8, 4)
    foreign = foreign_repo.reserve(0, 4).commit()
    try:
        with pytest.raises(ValueError, match='foreign repository'):
            scope.publish_features(foreign, key)
        handle = leases[0].handle
        repo.retire_permanently(handle)
        assert not scope.publish_features(leases[0], key)
        advance(archive, worker, scope)
        assert scope.feature_stats['keys'] == 7
    finally:
        foreign.reset()


def test_query_geometry_incomplete_bank_and_stale_epochs_do_not_enable_delete(history):
    archive, worker, repo, scope, leases = history
    complete_features(scope, leases)
    query = np.tile(np.array([1., 0.], dtype=np.float32), (1, 1, 2, 1))
    epoch = scope.begin_question(2)
    assert scope.append_question(epoch, 0, query, 1)
    assert not scope.feature_stats['question_ready']
    scope.pause_policy(False)
    advance(archive, worker, scope)
    assert scope.stats['retired'] == 0
    with pytest.raises(ValueError, match='C-contiguous'):
        scope.append_question(epoch, 1, query.swapaxes(1, 2), 1)
    next_epoch = scope.begin_question(1)
    assert next_epoch != epoch
    assert not scope.append_question(epoch, 1, query, 1)
    assert not scope.feature_stats['question_ready']
    assert scope.append_question(next_epoch, 0, query, 1)
    assert scope.feature_stats['question_ready']
    advance(archive, worker, scope)
    assert scope.stats['redundancy_retired'] == scope.stats['retired'] == 4
    assert scope.stats['lossy_retired'] == 0
    assert scope.stats['cpu_released'] == 4
    assert scope.feature_stats['keys'] == 4
    assert sum(bool(repo.pin(lease.handle)) for lease in leases) == 4
    assert archive.io_usage()['cpu_authority_bytes'] == 32
    assert repo.ready_slots == 0  # Existing GPU metadata remains pinned.
    leases.clear()
    advance(archive, worker, scope)
    assert repo.ready_slots == 4


def test_missing_question_cannot_be_bypassed_with_observations(history):
    archive, worker, _, scope, leases = history
    complete_features(scope, leases)
    query = np.ones((1, 8, 2, 2), dtype=np.float32)
    epoch = scope.begin_question(0)
    assert not scope.observe_query(epoch, query, 8)
    assert not scope.feature_stats['question_ready']
    scope.pause_policy(False)
    advance(archive, worker, scope)
    assert scope.stats['retired'] == 0
    epoch = scope.begin_question(8)
    assert scope.append_question(epoch, 0, query, 8)
    assert scope.observe_query(epoch, query, 8)
    assert scope.feature_stats['observation_segments'] == 1
    scope.invalidate_question()
    assert not scope.feature_stats['question_ready']
    assert not scope.observe_query(epoch, query, 8)
    advance(archive, worker, scope)
    assert scope.stats['retired'] == 0


def test_feature_and_query_allocation_stay_within_declared_bounds(history):
    _, _, _, scope, leases = history
    complete_features(scope, leases)
    stats = scope.feature_stats
    assert 0 < stats['feature_bytes'] <= stats['feature_peak_bytes'] <= stats['feature_capacity_bytes']
    assert 0 < stats['query_bytes'] <= stats['query_peak_bytes'] <= stats['query_capacity_bytes']
    epoch = scope.begin_question(1 << 40)
    assert epoch
    stats = scope.feature_stats
    assert not stats['question_ready']
    assert stats['overflow'] > 0
    assert stats['query_peak_bytes'] <= stats['query_capacity_bytes']
