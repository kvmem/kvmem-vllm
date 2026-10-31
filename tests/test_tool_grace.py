from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.active_residency import ActiveResidency
from vllm_kvmem.tool_grace import ToolGrace
from test_active_residency import save
from test_capacity import make_gate, request


def policy(now):
    return ToolGrace(1000, 15000, clock=lambda: now[0])


def test_learns_return_before_admission_and_never_counts_queue_wait():
    now = [0.]
    p = policy(now)
    ns = ('tenant', 'session')
    p.arrival('first', ns)
    now[0] = 10.
    p.finished('first', successful=True)
    now[0] = 10.1
    p.arrival('second', ns)
    now[0] = 40.
    p.arrival('second', ns)  # Repeated lookup must not resample or reset age.
    assert p.samples == 1 and p.grace(ns) == pytest.approx(.15)
    assert not p.defer('second') and p.yields == 1
    p.finished('second', successful=False)
    now[0] = 100.
    p.arrival('third', ns)
    assert p.samples == 1


def test_long_tool_gap_and_cheap_restore_disable_grace():
    now = [0.]
    p = policy(now)
    ns = ('a', 'session')
    p.arrival('a', ns)
    p.finished('a', successful=True)
    now[0] = 20.
    p.arrival('b', ns)
    assert p.grace(ns) == 0
    for _ in range(20):
        p.observe_restore(.01)
    assert p.grace(('new', 'session')) == 0
    assert p.grace(('other-tenant', 'session')) == 0


def test_namespace_learning_isolated_and_history_bounded():
    now = [0.]
    p = policy(now)
    p.arrival('a', ('tenant-a', 's'))
    p.finished('a', successful=True)
    now[0] = 10.
    p.arrival('b', ('tenant-b', 's'))
    assert p.samples == 0
    for i in range(1100):
        p.arrival(str(i), ('tenant', str(i)))
        p.finished(str(i), successful=False)
    p.finished('b', successful=False)
    assert len(p.sessions) == 1024 and not p.arrivals


def crowded():
    now = [10.]
    gate, manager, pool = make_gate(9)
    p = policy(now)
    resident = ActiveResidency({0: manager}, pool, 0, 1., pressure_only=True,
                               clock=lambda: now[0], tool_grace=p)
    gate.active_residency = resident
    view = save(resident, manager)
    peer = request('peer')
    assert gate.allocate_slots(peer, 16) is not None
    cold, hot = request('cold'), request('hot')
    p.arrival('cold', ('tenant', 'cold'))
    p.arrival('hot', view.namespace)
    return gate, resident, manager, pool, p, now, view, peer, cold, hot


def test_skip_cold_waiter_without_mutation_and_allow_hot_successor():
    gate, resident, manager, pool, p, now, view, peer, cold, hot = crowded()
    free = pool.get_num_free_blocks()
    assert not gate.tool_grace_allows(cold)
    assert gate.tool_grace_allows(hot)
    assert pool.get_num_free_blocks() == free
    assert resident.entries[view.namespace] is view
    assert not manager.resident_imports
    assert p.deferrals == 1
    now[0] += .3
    assert gate.tool_grace_allows(cold)
    assert gate.allocate_slots(cold, 16) is not None
    assert resident.pressure_evictions == 1
    gate.free(cold)
    gate.free(peer)
    assert pool.get_num_free_blocks() == 8


def test_existing_growth_can_reclaim_protected_idle_pages():
    gate, resident, manager, pool, p, now, view, peer, cold, hot = crowded()
    assert p.protected(view)
    assert gate.tool_grace_allows(peer)
    # A peer's reserved lifetime growth cannot be held hostage by idle KV.
    resident.make_room(pool.num_gpu_blocks - 2)
    assert not resident.entries
    assert all(b.ref_cnt == 0 for b in view.blocks)
    gate.free(peer)
    assert pool.get_num_free_blocks() == 8


def test_free_capacity_is_used_immediately_and_old_waiter_overrides_fresh_grace():
    gate, resident, manager, pool, p, now, view, peer, cold, hot = crowded()
    gate.free(peer)
    assert gate.tool_grace_allows(cold)  # Free pages suffice; no artificial delay.
    assert gate.allocate_slots(peer, 16) is not None
    now[0] += 16
    # Another hot completion must not restart the old cold request's age.
    from dataclasses import replace
    fresh = replace(view, protected_until=now[0] + 1.)
    resident.entries[view.namespace] = fresh
    assert gate.tool_grace_allows(cold) and p.yields == 1
    assert gate.allocate_slots(cold, 16) is not None
    gate.free(cold)
    gate.free(peer)
    assert pool.get_num_free_blocks() == 8


def test_pressure_prefers_expired_grace_but_never_reclaims_claimed_import():
    from test_active_residency import setup
    resident, manager, pool, now = setup(pressure_only=True)
    resident.tool_grace = p = policy(now)
    a = save(resident, manager, 'a', ('tenant', 'a'))
    b = save(resident, manager, 'b', ('tenant', 'b'))
    from dataclasses import replace
    resident.entries[b.namespace] = replace(b, protected_until=0.)
    resident.make_room(13)
    assert list(resident.entries) == [a.namespace]
    manager.prepare_import('next', 64, 48)
    resident.claim(a, 'next')
    now[0] += 20
    resident.make_room(16)
    assert all(b.ref_cnt == 1 for b in a.blocks)
    resident.rollback(a, 'next')
    resident.make_room(16)
    assert pool.get_num_free_blocks() == 16


def test_restore_cost_excludes_queue_and_cancelled_inflight_restore():
    from vllm_kvmem.restore_admission import RestoreAdmission
    from test_restore_admission import entry, command
    from test_active_residency import setup
    resident, _, _, now = setup(pressure_only=True)
    resident.tool_grace = p = policy(now)
    gate = RestoreAdmission(residency=resident, clock=lambda: now[0])
    req = request('cold')
    assert gate.lookup(req, entry('cold'))
    now[0] += 30
    gate.allocated(req)
    gate.dispatch(command(gate.permits['cold']))
    now[0] += .5
    gate.complete('cold', 1)
    assert p.restore_cost == pytest.approx(.9) and p.restores == 1
    assert gate.lookup(req, entry('cold'))
    gate.allocated(req)
    gate.dispatch(command(gate.permits['cold'], 2))
    gate.cancel(req)
    now[0] += 100
    gate.complete('cold', 2)
    assert p.restores == 1


def test_connector_defers_before_session_claim_and_observes_arrival():
    from vllm_kvmem.connector import KVMemConnector
    from vllm_kvmem.session_control import SessionLedger
    from test_active_residency import setup
    resident, _, _, now = setup(pressure_only=True)
    resident.tool_grace = p = policy(now)
    connector = object.__new__(KVMemConnector)
    connector.active_residency = resident
    connector.sessions = NS(namespace=SessionLedger.namespace)
    connector.tool_grace_allows = lambda request: False
    req = NS(request_id='a', session_id='session', cache_salt='tenant')
    connector.on_new_request(req)
    assert p.arrivals['a'].namespace == ('tenant', 'session')
    assert connector._get_matched_tokens(req, 0) == (None, False)


def test_public_options_roundtrip_and_defaults(tmp_path):
    from test_graph_options import runtime, settings, model_profile, worker_payload
    from vllm_kvmem.serve import build_launch
    assert runtime().active_tool_grace_ms == 0
    profile = model_profile(tmp_path)
    from test_agent_handoff_options import configured
    profile['kvmem'] = configured(active_resident_pressure_only=True, session_capture_terminal=True,
        active_tool_grace_ms=1000., active_tool_max_wait_ms=15000.)
    result = runtime(worker_payload(build_launch(profile, tmp_path, tmp_path/'run')))
    assert result.active_tool_grace_ms == 1000 and result.active_tool_max_wait_ms == 15000


@pytest.mark.parametrize('updates', [
    dict(active_tool_grace_ms=True), dict(active_tool_grace_ms=-1),
    dict(active_tool_grace_ms=float('nan')), dict(active_tool_grace_ms=float('inf')),
    dict(active_tool_max_wait_ms=False), dict(active_tool_max_wait_ms=0),
    dict(active_tool_max_wait_ms=float('nan')), dict(active_tool_grace_ms=20000),
    dict(active_tool_grace_ms=1000),  # Residency must explicitly be pressure-only.
])
def test_invalid_options(updates):
    from test_graph_options import runtime
    with pytest.raises(ValueError):
        runtime(updates)
