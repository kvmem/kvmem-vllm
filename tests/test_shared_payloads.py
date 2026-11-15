"""Shared prefixes, phase 2 steps 1-3 (docs/prefix-sharing.md): reference-counted
arena payloads, descriptor/payload decoupling with aliasing, and historical GC
that releases shared records logically and spends byte budgets on private ones."""
import threading
import time

import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.core.memory_storage import MemoryArchive

R = n.HistoricalPayloadRetireResult


def eventually(predicate):
    end = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < end
        time.sleep(.001)


# --- step 1: arena owners --------------------------------------------------

def test_owners_keep_a_payload_until_the_last_one_leaves():
    arena = n.HistoricalPayloadArena(2, 8, background=False)
    h = n.HistoricalPayloadHandle(0, 1)
    arena.publish(h, b'p' * 8)
    assert arena.owners(h) == (1, 0)
    assert arena.share(h) and arena.share(h, pool=True)
    assert arena.owners(h) == (3, 1)
    stats = arena.stats
    assert (stats['shared_slots'], stats['pool_slots'], stats['unique_bytes'], stats['used_bytes']) == (1, 1, 8, 8)
    assert arena.retire_unshared(h) == R.SHARED and arena.owners(h) == (3, 1)
    assert not arena.spill(h)
    assert arena.retire(h) == R.SHARED and arena.retire(h) == R.SHARED
    assert arena.retire(h) == R.STALE and arena.owners(h) == (1, 1)
    assert arena.stats['shared_slots'] == 0
    assert bytes(memoryview(arena.acquire(h))) == b'p' * 8
    assert arena.advance() == 0 and arena.stats['used_bytes'] == 8
    assert arena.release_pool(h) == R.COMMITTED and arena.release_pool(h) == R.ALREADY_RETIRED
    assert not arena.share(h) and arena.owners(h) == (0, 0)
    assert arena.advance() == 1 and arena.reclaimed(h) and arena.stats['pool_slots'] == 0
    eventually(arena.close)


def test_single_owner_payloads_retire_and_spill_exactly_as_before():
    arena = n.HistoricalPayloadArena(2, 8, background=False)
    first, second = n.HistoricalPayloadHandle(0, 1), n.HistoricalPayloadHandle(1, 2)
    arena.publish(first, b'a' * 8)
    arena.publish(second, b'b' * 8)
    assert arena.retire_unshared(first) == R.COMMITTED and arena.retire(first) == R.ALREADY_RETIRED
    assert arena.spill(second) and arena.spilled(second)
    assert arena.retire(second) == R.COMMITTED
    assert arena.advance() == 2 and arena.reclaimed(first) and arena.reclaimed(second)
    stats = arena.stats
    assert (stats['shared_slots'], stats['pool_slots'], stats['used_bytes'], stats['spilled_slots']) == (0, 0, 0, 0)
    eventually(arena.close)


def test_threads_sharing_and_dropping_release_each_payload_once():
    arena = n.HistoricalPayloadArena(4, 16, background=True, poll_ms=1)
    errors = []
    for round_ in range(200):
        h = n.HistoricalPayloadHandle(round_ % 4, round_ + 1)
        if round_ >= 4:
            eventually(lambda: arena.reclaimed(n.HistoricalPayloadHandle(round_ % 4, round_ - 3)))
        arena.publish(h, bytes([round_ % 256]) * 16)
        barrier = threading.Barrier(5)
        results = []

        def owner(pool):
            if not arena.share(h, pool=pool):
                errors.append('share')
            barrier.wait()
            results.append(arena.release_pool(h) if pool else arena.retire(h))
        threads = [threading.Thread(target=owner, args=(i % 2 == 1,)) for i in range(4)]
        for thread in threads:
            thread.start()
        barrier.wait()
        results.append(arena.retire(h))
        for thread in threads:
            thread.join()
        if results.count(R.COMMITTED) != 1 or results.count(R.SHARED) != 4:
            errors.append(results)
    assert not errors
    eventually(lambda: arena.stats['used_slots'] == 0)
    assert arena.stats['released_slots'] == 200 and arena.stats['shared_slots'] == 0
    eventually(arena.close)


# --- step 2: archive aliasing and payload slots ---------------------------------

def native_archive(tmp_path, capacity=16, **kwargs):
    return MemoryArchive(str(tmp_path), 8, capacity, native_payloads=True, **kwargs)


def write(archive, namespace, repo, values, begin=0):
    leases = [repo.reserve(begin + i * 4, 4).commit() for i in range(len(values))]
    archive.write(namespace, leases, b''.join(values)).result(5)
    return leases


def read(archive, namespace, leases):
    return bytes(archive.read(namespace, leases).result(5)[0])


def release(archive, namespace, leases):
    """Drop each descriptor; the last owner of a payload waits for its bytes."""
    for lease in leases:
        handle = lease.handle
        lease.reset()
        eventually(lambda: archive.retire_cpu_record(namespace, handle))


def test_default_tier_keeps_descriptor_and_payload_slots_identical(tmp_path):
    archive = native_archive(tmp_path)
    leases = write(archive, 'a', n.Repository(8, 4), [b'x' * 8, b'y' * 8])
    try:
        tier = archive.tier
        assert tier._arena_free is None and not tier._arena_slot
        for lease in leases:
            entry = archive._entries[archive.key('a', lease)]
            handle = tier.records[entry.allocation.slot]
            assert (handle.slot, handle.generation) == (entry.allocation.slot, entry.allocation.generation)
    finally:
        leases.clear()
        eventually(archive.close)


def test_alias_shares_payloads_without_bytes_and_survives_either_side(tmp_path):
    archive = native_archive(tmp_path)
    parent_repo, child_repo = n.Repository(8, 4), n.Repository(8, 4)
    values = [bytes([i + 1]) * 8 for i in range(3)]
    parent = write(archive, 'parent', parent_repo, values)
    used = archive.tier.payloads.stats['used_bytes']
    handles = archive.record_payload_handles('parent', parent)
    child = [child_repo.reserve(i * 4, 4).commit() for i in range(3)]
    assert archive.alias_records('child', child, handles)
    arena = archive.tier.payloads
    assert arena.stats['used_bytes'] == used == 24 and arena.stats['shared_slots'] == 3
    assert read(archive, 'child', child) == read(archive, 'parent', parent) == b''.join(values)
    assert archive.namespace_allocated_records('child') == 3
    # The parent leaves first: its descriptors are free at once, its payloads stay.
    release(archive, 'parent', parent)
    assert archive.namespace_allocated_records('parent') == 0
    assert read(archive, 'child', child) == b''.join(values)
    assert len(archive.tier._detached) == 3 and arena.stats['shared_slots'] == 0
    # New writes reuse the parent's descriptor ids but not its occupied arena slots.
    other_repo = n.Repository(8, 4)
    other = write(archive, 'other', other_repo, [b'o' * 8] * 3)
    occupied = {h.slot for h in handles}
    assert not occupied & {h.slot for h in archive.record_payload_handles('other', other)}
    assert read(archive, 'child', child) == b''.join(values)
    # The last owner leaves: the bytes go and the arena slots are free again.
    release(archive, 'child', child)
    eventually(lambda: arena.advance() >= 0 and arena.stats['used_bytes'] == 24)
    assert not archive.tier._detached and occupied <= archive.tier._arena_free
    release(archive, 'other', other)
    eventually(lambda: arena.advance() >= 0 and arena.stats['used_bytes'] == 0)
    eventually(archive.close)


def test_releasing_an_alias_frees_its_descriptor_and_keeps_the_parent_readable(tmp_path):
    archive = native_archive(tmp_path)
    parent_repo, child_repo = n.Repository(8, 4), n.Repository(8, 4)
    parent = write(archive, 'parent', parent_repo, [b'a' * 8, b'b' * 8])
    child = [child_repo.reserve(i * 4, 4).commit() for i in range(2)]
    assert archive.alias_records('child', child, archive.record_payload_handles('parent', parent))
    free = len(archive._free_ids)
    release(archive, 'child', child)
    assert len(archive._free_ids) == free + 2 and archive.tier.payloads.stats['used_bytes'] == 16
    assert read(archive, 'parent', parent) == b'a' * 8 + b'b' * 8
    assert archive.tier.payloads.stats['shared_slots'] == 0
    with pytest.raises(ValueError, match='live'):
        archive.alias_records('late', [n.Repository(8, 4).reserve(0, 4).commit()],
                              [n.HistoricalPayloadHandle(9 % 16, 99)])
    release(archive, 'parent', parent)
    eventually(lambda: archive.tier.payloads.advance() >= 0 and archive.tier.payloads.stats['used_bytes'] == 0)
    eventually(archive.close)


def test_fewer_payload_slots_than_descriptors_bound_bytes_not_aliases(tmp_path):
    archive = native_archive(tmp_path, capacity=8, payload_capacity=3)
    repo, child_repo = n.Repository(8, 4), n.Repository(8, 4)
    parent = write(archive, 'a', repo, [b'1' * 8, b'2' * 8, b'3' * 8])
    extra = repo.reserve(12, 4).commit()
    assert archive.write('a', [extra], b'4' * 8) is None          # payload slots full: backpressure
    child = [child_repo.reserve(i * 4, 4).commit() for i in range(3)]
    assert archive.alias_records('b', child, archive.record_payload_handles('a', parent))
    assert len(archive._entries) == 6 and archive.tier.payloads.stats['used_slots'] == 3
    release(archive, 'a', parent)
    release(archive, 'b', child)
    eventually(lambda: archive.tier.payloads.advance() >= 0 and archive.tier.payload_room(1))
    job = archive.write('a', [extra], b'4' * 8)
    assert job is not None and job.result(5)
    extra.reset()
    eventually(archive.close)


# --- step 3: historical GC with shared records -----------------------------------

def gc_config(**overrides):
    config = n.HistoricalGcConfig()
    config.mode = n.HistoricalGcMode.DELETE
    config.background = False
    config.max_scopes = 2
    config.poll_ms = 1
    config.scan_blocks = 8
    config.delete_blocks = 8
    config.ready_low = overrides.get('ready_low', 2)
    config.ready_high = overrides.get('ready_high', 4)
    return config


def gc_archive(tmp_path, **overrides):
    archive = native_archive(tmp_path, capacity=32)
    worker = archive.enable_history_gc(gc_config(**overrides))
    parent_repo, child_repo = n.Repository(8, 4), n.Repository(8, 4)
    parent_scope = archive.register_history_scope('parent', parent_repo)
    child_scope = archive.register_history_scope('child', child_repo)
    return archive, worker, (parent_repo, parent_scope), (child_repo, child_scope)


def sealed_write(archive, namespace, repo, scope, values, begin=0):
    leases = write(archive, namespace, repo, values, begin)
    for lease in leases:
        assert scope.update(lease, n.HISTORY_SEALED)
    return leases


def drain(archive, namespace, scope):
    for completion in scope.completions(16):
        if completion.cpu and not archive.retire_cpu_record(namespace, completion.handle):
            continue
        assert scope.acknowledge(completion)


def steps(archive, worker, scopes, count=24):
    for _ in range(count):
        worker.advance()
        for namespace, scope in scopes:
            drain(archive, namespace, scope)


def import_prefix(archive, parent, child_repo, child_scope, count):
    leases = [child_repo.reserve(i * 4, 4).commit() for i in range(count)]
    assert archive.alias_records('child', leases, archive.record_payload_handles('parent', parent[:count]))
    for lease in leases:
        assert child_scope.update(lease, n.HISTORY_SEALED)
    return leases


def test_slot_pressure_retires_shared_records_logically_and_frees_no_bytes(tmp_path):
    archive, worker, (parent_repo, parent_scope), (child_repo, child_scope) = gc_archive(tmp_path)
    scopes = (('parent', parent_scope), ('child', child_scope))
    values = [bytes([i + 1]) * 8 for i in range(4)]
    parent = sealed_write(archive, 'parent', parent_repo, parent_scope, values)
    arena = archive.tier.payloads
    shared = import_prefix(archive, parent, child_repo, child_scope, 4)
    assert child_scope.stats['shared_marked'] == 4 and child_scope.stats['published'] == 4
    private = sealed_write(archive, 'child', child_repo, child_scope, [b'c' * 8] * 4, begin=16)
    assert child_repo.ready_slots == 0 and arena.stats['used_bytes'] == 64
    handles = [lease.handle for lease in shared]
    shared.clear()
    private.clear()
    steps(archive, worker, scopes)
    stats = child_scope.stats
    # LRU picks the four oldest (the imported prefix): each drops only the child's reference.
    assert stats['retired'] == 4 and stats['shared_released'] == 4 and stats['cpu_released'] == 0
    assert stats['pending_cpu_bytes'] == 0 and arena.stats['used_bytes'] == 64
    assert all(not child_repo.pin(handle) for handle in handles) and child_repo.ready_slots == 4
    assert arena.stats['shared_slots'] == 0
    assert read(archive, 'parent', parent) == b''.join(values)
    parent.clear()
    eventually(archive.close)


def test_byte_budget_spends_private_records_only(tmp_path):
    archive, worker, (parent_repo, parent_scope), (child_repo, child_scope) = gc_archive(tmp_path)
    scopes = (('parent', parent_scope), ('child', child_scope))
    parent = sealed_write(archive, 'parent', parent_repo, parent_scope, [b'p' * 8] * 4)
    shared = import_prefix(archive, parent, child_repo, child_scope, 4)
    private = sealed_write(archive, 'child', child_repo, child_scope, [b'q' * 8, b'r' * 8], begin=16)
    shared_handles = [lease.handle for lease in shared]
    private_handles = [lease.handle for lease in private]
    shared.clear()
    private.clear()
    steps(archive, worker, scopes)
    assert child_scope.stats['retired'] == 0          # no slot pressure (2 of 8 slots ready)
    child_scope.request_capacity(0, 8)                 # keep one private record of CPU bytes
    steps(archive, worker, scopes)
    stats = child_scope.stats
    assert stats['retired'] == 1 and stats['cpu_released'] == 1 and stats['shared_released'] == 0
    assert all(child_repo.pin(handle) for handle in shared_handles)
    assert not child_repo.pin(private_handles[0]) and child_repo.pin(private_handles[1])
    assert archive.tier.payloads.stats['used_bytes'] == 40
    parent.clear()
    eventually(archive.close)


def test_unmarked_payload_shared_by_a_pool_owner_is_released_logically(tmp_path):
    archive, worker, (parent_repo, parent_scope), _ = gc_archive(tmp_path, ready_low=4, ready_high=6)
    scopes = (('parent', parent_scope),)
    parent = sealed_write(archive, 'parent', parent_repo, parent_scope, [bytes([i]) * 8 for i in range(8)])
    handles = archive.record_payload_handles('parent', parent)
    arena = archive.tier.payloads
    assert all(archive.tier.share_pool(handle) for handle in handles[:2])    # an image, not marked
    parent.clear()
    steps(archive, worker, scopes)
    stats = parent_scope.stats
    # Six records go; the two oldest belong to the image as well: no bytes credited for them.
    assert stats['retired'] == 6 and stats['shared_marked'] == stats['shared_released'] == 2
    assert stats['cpu_released'] == 4 and stats['pending_cpu_bytes'] == 0
    eventually(lambda: arena.advance() >= 0 and arena.stats['used_bytes'] == 32)
    assert all(arena.owners(handle) == (1, 1) for handle in handles[:2])
    assert all(archive.tier.release_pool(handle) == R.COMMITTED for handle in handles[:2])
    eventually(lambda: arena.advance() >= 0 and arena.stats['used_bytes'] == 16)
    eventually(archive.close)


def test_closing_a_sharing_scope_drops_each_reference_once(tmp_path):
    archive, worker, (parent_repo, parent_scope), (child_repo, child_scope) = gc_archive(tmp_path)
    scopes = (('parent', parent_scope), ('child', child_scope))
    values = [bytes([i + 7]) * 8 for i in range(3)]
    parent = sealed_write(archive, 'parent', parent_repo, parent_scope, values)
    shared = import_prefix(archive, parent, child_repo, child_scope, 3)
    arena = archive.tier.payloads
    handles = archive.record_payload_handles('parent', parent)
    assert all(arena.owners(handle) == (2, 0) for handle in handles)
    shared.clear()
    assert not child_scope.close()
    for _ in range(64):
        worker.advance()
        drain(archive, 'child', child_scope)
        if child_scope.close():
            break
    assert child_scope.close()
    stats = child_scope.stats
    assert stats['shared_released'] == 3 and stats['cpu_released'] == 0
    assert archive.namespace_allocated_records('child') == 0
    assert all(arena.owners(handle) == (1, 0) for handle in handles)
    assert read(archive, 'parent', parent) == b''.join(values) and arena.stats['used_bytes'] == 24
    parent.clear()
    eventually(archive.close)


def test_session_tiering_never_spills_shared_payloads(tmp_path):
    archive = native_archive(tmp_path / 'arena')
    tier = archive.tier
    tier.enable_spill(str(tmp_path / 'spill'), 1 << 20, segment_bytes=1 << 12)
    parent_repo, child_repo = n.Repository(8, 4), n.Repository(8, 4)
    parent = write(archive, 'parent', parent_repo, [b'a' * 8, b'b' * 8])
    child = [child_repo.reserve(i * 4, 4).commit() for i in range(2)]
    assert archive.alias_records('child', child, archive.record_payload_handles('parent', parent))
    own = write(archive, 'child', child_repo, [b'c' * 8], begin=8)

    def allocations(namespace, leases):
        return [archive._entries[archive.key(namespace, lease)].allocation for lease in leases]
    # The child's aliases are pool-charged: skipped before any SSD write. Its private record spills.
    assert tier.spill(allocations('child', child + own)) == 1 and tier.spill_stats['spill_busy'] == 0
    # A plain descriptor whose payload another owner shares: the arena refuses the spill.
    assert tier.spill(allocations('parent', parent)) == 0 and tier.spill_stats['spill_busy'] == 2
    assert read(archive, 'child', child + own) == b'a' * 8 + b'b' * 8 + b'c' * 8
    for namespace, leases in (('child', child + own), ('parent', parent)):
        release(archive, namespace, leases)
    eventually(lambda: tier.payloads.advance() >= 0 and tier.payloads.stats['used_bytes'] == 0)
    eventually(archive.close)
