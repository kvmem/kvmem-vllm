"""Capture optimization must validate without changing selection or leases."""
from dataclasses import replace
import random

import pytest

from test_native_restore_frame import fixture, append, plan_value
from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool


@pytest.mark.parametrize('kind', ['dense', 'sparse', 'partial'])
def test_native_capture_same_records_and_no_residency_mutation(kind):
    m, cp = fixture(kind, 'same')
    old, reference = fixture(kind, 'same')
    before = (tuple(m._store.working_records()), m.repository.stats['leases'])
    for _ in range(10):
        m._validate_window_checkpoint(cp, native=True)
        assert (tuple(m._store.working_records()), m.repository.stats['leases']) == before
    native = m.checkpoint_window(allow_partial=kind=='partial', native_validation=True)
    assert native.records == reference.records and native.position == reference.position
    a = m._store.select([m._store.handle(r[0]) for r in cp.records])
    b = old._store.select([old._store.handle(r[0]) for r in reference.records])
    assert plan_value(a) == plan_value(b)
    assert m.close() and old.close()


def test_differential_corruption_checks_and_lease_rollback():
    rng = random.Random(2001)
    m, cp = fixture('partial', 'same')
    before = (tuple(m._store.working_records()), m.repository.stats['leases'])
    for _ in range(600):
        rows = [list(r) for r in cp.records]
        row, field = rng.randrange(len(rows)), rng.randrange(5)
        rows[row][field] += rng.choice([1, 4, 32])
        damaged = replace(cp, records=tuple(tuple(r) for r in rows))
        outcomes = []
        for native in (False, True):
            try:
                m._validate_window_checkpoint(damaged, native=native)
                outcomes.append(True)
            except (ValueError, RuntimeError):
                outcomes.append(False)
        assert outcomes == [False, False]
        assert (tuple(m._store.working_records()), m.repository.stats['leases']) == before
    assert m.close()


def test_saved_prefix_validation_allows_suffix_but_rejects_foreign_prefix():
    m, cp = fixture('sparse', 'same')
    other, foreign = fixture('sparse', 'same')
    append(m, 4, local=12)
    m._validate_window_checkpoint(cp, native=True)
    with pytest.raises(ValueError, match='foreign'):
        m._validate_window_checkpoint(replace(cp, prefix=foreign.prefix), native=True)
    assert m.close() and other.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_pool_capture_and_restore_preserve_state(tmp_path, enabled):
    from test_prompt_checkpoints import saved, ready, Event
    pool = PromptCheckpointPool(2, 4096, native_checkpoint_capture=enabled)
    pool, cp, a, archive, adapter, _ = saved(tmp_path, pool=pool)
    assert cp.tokens == bytes().join(i.to_bytes(4, 'little') for i in range(8))
    pool.retire(cp, Event(True)); ready(pool, cp)
    assert pool.claim(cp.namespace, 'new', list(range(11))) is cp
    plan = pool.prepare_claim_window(cp)
    a.memory.attach_completion(Event(True)); assert a.memory.commit_selection()
    assert pool.finish_claim(cp, Event(True)) is a
    del plan
    assert a.close() and archive.close()


def test_options_are_explicit_booleans():
    from test_options import config
    from vllm_kvmem.options import RuntimeOptions
    for name in ('native_checkpoint_capture', 'host_stage_timing'):
        assert not getattr(RuntimeOptions.from_vllm(config({})), name)
        assert getattr(RuntimeOptions.from_vllm(config({'kvmem': {name: True}})), name)
        with pytest.raises(ValueError, match='boolean'):
            RuntimeOptions.from_vllm(config({'kvmem': {name: 1}}))
