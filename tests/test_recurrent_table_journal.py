from types import SimpleNamespace

import pytest
import torch

from test_native_state import adapter
from vllm_kvmem.recurrent_table_journal import RecurrentTableJournal, install_recurrent_table_journal


def fixture(*, install_journal=True):
    owner, conv, ssm, table = adapter()
    tables = owner.runner.block_tables
    tables.blocks_per_kv_block = [1]
    writes = []

    def append(index, groups, overwrite):
        start = 0 if overwrite else int(tables.num_blocks.np[0, index])
        writes.append((index, start, list(groups[0])))
        tables.num_blocks.np[0, index] = start + len(groups[0])
        if getattr(tables, 'fail', None) == 'append':
            raise OSError('partial native append failed')

    def apply():
        for index, start, ids in writes:
            table[index, start:start+len(ids)] = torch.tensor(ids)
        writes.clear()
        if getattr(tables, 'fail', None) == 'apply':
            raise OSError('partial native submission failed')

    tables.append_block_ids, tables.apply_staged_writes = append, apply
    journal = RecurrentTableJournal(owner.runner, [0]) if install_journal else None
    if journal is not None:
        owner.runner._kvmem_recurrent_table_journal = journal
    return owner, journal, tables, table


def test_unknown_row_falls_back_but_witnessed_rows_need_no_gpu_read(monkeypatch):
    owner, journal, tables, table = fixture()
    assert journal.rows('a', 0) is None
    assert owner._rows('a', 0) == (0, (1, 2))
    tables.append_block_ids(0, ([1, 2],), True)
    with pytest.raises(RuntimeError, match='unsubmitted'):
        owner.borrow(['a'])
    tables.apply_staged_writes()

    def no_read(*args, **kwargs):
        pytest.fail('Witnessed recurrent rows must not perform a device-to-host copy')

    monkeypatch.setattr(torch.Tensor, 'cpu', no_read)
    bundle = owner.borrow(['a'])
    owner.validate(bundle)
    assert bundle.page_rows['a', 0] == (1, 2)


def test_same_count_overwrite_is_not_mistaken_for_unchanged_allocation():
    owner, journal, tables, _ = fixture()
    tables.append_block_ids(0, ([1, 2],), True)
    tables.apply_staged_writes()
    bundle = owner.borrow(['a'])
    tables.append_block_ids(0, ([2, 5],), True)
    tables.apply_staged_writes()
    with pytest.raises(ValueError, match='page table changed'):
        owner.validate(bundle)


@pytest.mark.parametrize('mutation', ['generation', 'index', 'pointer', 'count', 'hook'])
def test_stale_witness_cannot_validate_recycled_state(mutation):
    owner, journal, tables, _ = fixture()
    tables.append_block_ids(0, ([1, 2],), True)
    tables.apply_staged_writes()
    bundle = owner.borrow(['a'])
    if mutation == 'generation': owner.runner._kvmem_request_generations['a'] += 1
    if mutation == 'index': owner.runner.req_states.req_id_to_index['a'] = 1
    if mutation == 'pointer': tables.block_tables[0].gpu = tables.block_tables[0].gpu.clone()
    if mutation == 'count': tables.num_blocks.np[0, 0] = 1
    if mutation == 'hook': tables.append_block_ids = lambda *a: None
    with pytest.raises((ValueError, RuntimeError)):
        owner.validate(bundle)


def test_new_generation_requires_fresh_overwrite_and_cross_request_alias_is_rejected():
    owner, journal, tables, _ = fixture()
    tables.append_block_ids(0, ([1, 2],), True)
    tables.apply_staged_writes()
    owner.runner._kvmem_request_generations['a'] += 1
    with pytest.raises(ValueError, match='incarnation'):
        journal.rows('a', 0)
    tables.append_block_ids(0, ([1, 2],), True)
    tables.append_block_ids(1, ([1, 4],), True)
    tables.apply_staged_writes()
    with pytest.raises(ValueError, match='share'):
        owner.borrow(['a', 'b'])


def test_append_requires_known_prefix_and_cross_stream_has_no_cpu_witness(monkeypatch):
    owner, journal, tables, _ = fixture()
    tables.append_block_ids(0, ([],), False)
    tables.apply_staged_writes()
    assert journal.rows('a', 0) is None
    tables.append_block_ids(0, ([1],), True)
    tables.append_block_ids(0, ([2],), False)
    tables.apply_staged_writes()
    assert journal.rows('a', 0) == (1, 2)
    monkeypatch.setattr(journal, '_stream', lambda _: 'another stream')
    assert journal.rows('a', 0) is None


def test_explicit_ablation_uses_original_gpu_table_validation(monkeypatch):
    owner, journal, tables, table = fixture()
    tables.append_block_ids(0, ([1, 2],), True)
    tables.apply_staged_writes()
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', '0')
    table[0, 0] = 5
    assert owner._rows('a', 0) == (0, (5, 2))


def test_runtime_option_disables_optimization_without_launcher_env(monkeypatch):
    from vllm_kvmem.checkpoint_timing import async_checkpoint_capture
    monkeypatch.delenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', raising=False)
    runner = SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=SimpleNamespace(
        kv_connector_extra_config={'kvmem': {'async_checkpoint_capture': False}})))
    assert not async_checkpoint_capture(runner)
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', '1')
    assert async_checkpoint_capture(runner)


@pytest.mark.parametrize('operation', ['append', 'apply'])
def test_partial_native_failure_poisoned_witness_never_falls_back(operation):
    owner, journal, tables, _ = fixture()
    tables.append_block_ids(0, ([1, 2],), True)
    tables.apply_staged_writes()
    tables.fail = operation
    with pytest.raises(OSError):
        tables.append_block_ids(0, ([2, 5],), True)
        tables.apply_staged_writes()
    with pytest.raises(RuntimeError, match='lost its native write witness'):
        owner.borrow(['a'])


def test_empty_warmup_submission_does_not_publish_unknown_request_rows():
    owner, journal, tables, _ = fixture()
    tables.apply_staged_writes()
    assert not journal.committed and journal.rows('a', 0) is None


def empty_startup(*, install_journal=False):
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm_kvmem.session_worker import SessionWorker
    runner = fixture(install_journal=install_journal)[0].runner
    runner.req_states.req_id_to_index.clear()
    runner._kvmem_request_generations.clear()
    runner._kvmem_next_generation = 0
    runner._kvmem_restore_allocations = {}
    runner._kvmem_capture_runtime = None
    runner._kvmem_pending_capture = None
    runner._kvmem_saved_capture = None
    runner._kvmem_sessions = SessionWorker(runner, SimpleNamespace(
        session_capacity=8, session_checkpoint_bytes=1048576))
    runner._kvmem_async_refresh = SimpleNamespace(jobs={}, deferred={}, adapter=None,
                                                initialize_cache=lambda: None)
    runner.block_tables.num_blocks.np.fill(0)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(
        kv_cache_spec=MambaSpec(block_size=16, shapes=((2, 2),), dtypes=(torch.float32,)))])
    return runner


@pytest.mark.parametrize('positional', [False, True])
def test_runner_temporary_graph_cache_then_serving_cache_has_fresh_journal(monkeypatch, positional):
    """Exercise the real override in native graph profiling -> teardown -> init order."""
    import gc
    import weakref
    from vllm_kvmem.engine_api import engine_root, find_symbol
    from vllm_kvmem.runner import make_probe_runner
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', '1')
    # Check positional forwarding without importing unrelated GPU model kernels.
    native = find_symbol((engine_root() / 'v1/worker/gpu/model_runner.py').read_text(),
                         'GPUModelRunner.initialize_kv_cache')
    assert [arg.arg for arg in native.args.args][:3] == [
        'self', 'kv_cache_config', 'is_profiling']
    calls = []

    class Base:
        def __init__(self):
            self.__dict__.update(vars(empty_startup()))
            self._kvmem_sessions.runner = self
            self.vllm_config = SimpleNamespace(kv_transfer_config=None,
                cache_config=SimpleNamespace(cache_dtype='fp8_e4m3'))

        def initialize_kv_cache(self, kv_cache_config, is_profiling=False,
                                kv_cache_allocation_context=None):
            self.kv_cache_config = kv_cache_config
            self.block_tables = empty_startup().block_tables
            calls.append(is_profiling)
            return 'initialized'

    runner = make_probe_runner(Base)()
    config = runner.kv_cache_config
    if positional:
        assert runner.initialize_kv_cache(config, True) == 'initialized'
    else:
        assert runner.initialize_kv_cache(config, is_profiling=True) == 'initialized'
    assert not hasattr(runner, '_kvmem_recurrent_table_journal')
    temporary_table = weakref.ref(runner.block_tables.block_tables[0].gpu)
    runner.block_tables.apply_staged_writes()  # no-request dummy graph work
    del runner.kv_cache_config  # native _teardown_profiling_state
    assert runner.initialize_kv_cache(config) == 'initialized'
    gc.collect()
    assert temporary_table() is None, 'No journal may keep the profiling GPU table alive'
    journal = runner._kvmem_recurrent_table_journal
    assert journal.tables is runner.block_tables and journal.groups == (0,)
    assert not journal.pending and not journal.committed
    runner.req_states.req_id_to_index['real'] = 0
    runner._kvmem_request_generations['real'] = 1
    runner._kvmem_next_generation = 1
    runner.block_tables.append_block_ids(0, ([1, 2],), True)
    runner.block_tables.apply_staged_writes()
    assert journal.rows('real', 0) == (1, 2)
    assert calls == [True, False]


def test_unused_startup_journal_can_rebind_but_never_reuses_its_witness(monkeypatch):
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', '1')
    runner = empty_startup(install_journal=True)
    previous = runner._kvmem_recurrent_table_journal
    old_tables = previous.tables
    runner.block_tables = empty_startup().block_tables
    install_recurrent_table_journal(runner)
    current = runner._kvmem_recurrent_table_journal
    assert current is not previous and current.tables is runner.block_tables
    assert previous.failed and old_tables.append_block_ids is previous.native_append
    assert old_tables.apply_staged_writes is previous.native_apply
    assert not current.pending and not current.committed
    install_recurrent_table_journal(runner)
    assert runner._kvmem_recurrent_table_journal is current  # same cache is idempotent
    with pytest.raises(RuntimeError, match='lost its native write witness'):
        previous.rows('real', 0)


@pytest.mark.parametrize('state', [
    'native-request', 'generation', 'served-then-drained', 'restore-lease', 'runtime',
    'capture', 'session-binding', 'session-checkpoint', 'deferred-checkpoint',
    'session-adapter', 'refresh-job', 'pending-writes', 'committed-row', 'failed', 'lost-hook'])
def test_reinitialization_never_discards_requests_or_retained_state(monkeypatch, state):
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', '1')
    runner = empty_startup(install_journal=True)
    old = runner._kvmem_recurrent_table_journal
    if state == 'native-request': runner.req_states.req_id_to_index['live'] = 0
    elif state == 'generation': runner._kvmem_request_generations['live'] = 1
    elif state == 'served-then-drained': runner._kvmem_next_generation = 1
    elif state == 'restore-lease': runner._kvmem_restore_allocations['live'] = object()
    elif state == 'runtime': runner._kvmem_capture_runtime = object()
    elif state == 'capture': runner._kvmem_pending_capture = object()
    elif state == 'session-binding': runner._kvmem_sessions.bindings['live'] = object()
    elif state == 'session-checkpoint': runner._kvmem_sessions.pool.entries['idle'] = object()
    elif state == 'deferred-checkpoint': runner._kvmem_sessions.pool._discarded_captures.append(object())
    elif state == 'session-adapter': runner._kvmem_sessions.model_adapter = object()
    elif state == 'refresh-job': runner._kvmem_async_refresh.jobs['live'] = object()
    elif state == 'pending-writes': old.tables.append_block_ids(0, ([1, 2],), True)
    elif state == 'committed-row':
        old.tables.append_block_ids(0, ([1, 2],), True)
        old.tables.apply_staged_writes()
    elif state == 'failed': old.failed = True
    elif state == 'lost-hook': old.tables.append_block_ids = lambda *a: None
    runner.block_tables = empty_startup().block_tables
    with pytest.raises(RuntimeError, match='reinitialization'):
        install_recurrent_table_journal(runner)
    assert runner._kvmem_recurrent_table_journal is old


def test_profiling_cannot_bypass_active_request_protection(monkeypatch):
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', '1')
    runner = empty_startup()
    runner._kvmem_next_generation = 1
    with pytest.raises(RuntimeError, match='startup without request state'):
        install_recurrent_table_journal(runner, is_profiling=True)
    assert not hasattr(runner, '_kvmem_recurrent_table_journal')
