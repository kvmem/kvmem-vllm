from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch

from vllm_kvmem.session_control import SessionBinding
from vllm_kvmem.systemone_worker import ActiveDecisionQueue, TailSnapshot


def queue():
    binding = SessionBinding(('tenant', 'parent'), 7, 'native-parent', ())
    runner = NS(_kvmem_bounded_target=True,
        _kvmem_sessions=NS(bindings={'native-parent': binding}, retiring={}, imports=set()),
        _kvmem_capture_runtime=NS(drain_requests=lambda *a, **k: True,
            authorities={'native-parent': NS(history_end=50)}))
    return ActiveDecisionQueue(runner), binding


def submit(q, **reference):
    return q.rpc('submit', 'operation', {'namespace': ['tenant', 'parent'], **reference},
                 [{'tokens': [1, 2], 'labels': [3]}])


def test_queue_waits_for_confirmed_prefill_and_keeps_parent_identity(monkeypatch):
    q, binding = queue()
    assert submit(q) == {'status': 'pending'}
    seen = []
    monkeypatch.setattr('vllm_kvmem.systemone_worker.execute_reads',
                        lambda *args: seen.append(args) or {'status': 'done'})
    batch = NS(req_ids=['native-parent'], num_computed_tokens_np=[40], prefill_len_np=[50])
    q.before_attention(batch)
    assert not seen
    batch.num_computed_tokens_np = [50]
    q.runner._kvmem_capture_runtime.drain_requests = lambda *a, **k: False
    q.before_attention(batch)
    assert not seen
    q.runner._kvmem_capture_runtime.drain_requests = lambda *a, **k: True
    q.before_attention(batch)
    assert len(seen) == 1 and seen[0][3] is binding
    q.before_attention(batch)
    assert len(seen) == 1
    assert q.rpc('poll', 'operation') == {'status': 'done'}
    q.rpc('cancel', 'operation')
    assert not q.pending


@pytest.mark.parametrize('change', ['tenant', 'generation', 'incarnation', 'retire', 'history'])
def test_queue_rejects_stale_or_wrong_owner(change, monkeypatch):
    q, binding = queue()
    monkeypatch.setattr('vllm_kvmem.systemone_worker.execute_reads', lambda *a: pytest.fail('must not execute'))
    if change == 'generation':
        assert 'error' in submit(q, generation=6)
        return
    assert submit(q, history_end=49 if change == 'history' else 50)['status'] == 'pending'
    if change == 'tenant':
        q.runner._kvmem_sessions.bindings['native-parent'] = replace(binding, namespace=('other', 'parent'))
    elif change == 'incarnation':
        q.runner._kvmem_sessions.bindings['native-parent'] = replace(binding, request_id='new-native-parent')
    elif change == 'retire':
        q.runner._kvmem_sessions.retiring['native-parent'] = object()
    q.before_attention(NS(req_ids=['native-parent'], num_computed_tokens_np=[50], prefill_len_np=[50]))
    assert 'error' in q.rpc('poll', 'operation')


def test_cancelled_and_expired_reads_never_execute(monkeypatch):
    q, _ = queue()
    monkeypatch.setattr('vllm_kvmem.systemone_worker.execute_reads', lambda *a: pytest.fail('must not execute'))
    submit(q)
    q.rpc('cancel', 'operation')
    q.before_attention(None)
    submit(q)
    q.pending['operation']['expires'] = 0
    q.before_attention(None)
    assert 'error' in q.pending['operation']['result']


@pytest.mark.parametrize('field,value', [('_kvmem_graph', True),
    ('_kvmem_async_refresh', object()), ('_kvmem_native_attention', object()),
    ('speculative_config', NS(method='dflash')),
    ('vllm_config', NS(parallel_config=NS(tensor_parallel_size=2)))])
def test_unqualified_execution_is_rejected_before_queueing(field, value):
    q, _ = queue()
    setattr(q.runner, field, value)
    assert 'error' in submit(q)
    assert not q.pending


def test_tail_rollback_preserves_prefix_and_neighbours_byte_for_byte():
    cache = torch.arange(6*16*2*4, dtype=torch.int32).reshape(6, 16, 2, 4)
    original = cache.clone()
    pages = torch.tensor([2, 2, 4, 4])
    offsets = torch.tensor([14, 15, 0, 1])
    snapshot = TailSnapshot([(cache, pages, offsets)])
    for value in (0, 789, -999):
        cache[pages, offsets] = value
        snapshot.restore()
        assert torch.equal(cache, original)


@pytest.mark.parametrize('fail', [False, True])
def test_transaction_restores_parent_even_when_private_forward_fails(monkeypatch, fail):
    pytest.importorskip('triton')
    from vllm_kvmem.systemone_worker import execute_reads
    from vllm_kvmem.window_batch import WindowBatchAdapter
    q, binding = queue()
    runner = q.runner
    runner.device = torch.device('cpu')
    runner.model_config = NS(max_model_len=1024)
    runner.kv_cache_config = None
    runner._kvmem_window_adapter = WindowBatchAdapter((0,), 128)
    runner._kvmem_last_batch = object()
    runner._kvmem_window_batch = object()
    runner._kvmem_projecting_query = False
    saved = {name: getattr(runner, name) for name in ('_kvmem_window_adapter',
        '_kvmem_last_batch', '_kvmem_window_batch', '_kvmem_projecting_query')}
    recurrent = torch.arange(12, dtype=torch.float32)
    counter = torch.ones(1, dtype=torch.int32)
    runner.model_state = NS(num_accepted_tokens_gpu=counter)
    bundle = NS(tensors={'state': recurrent, 'counter': counter}, bytes=52,
                request_indices={'native-parent': 0})
    state = NS(borrow=lambda _: bundle, validate=lambda _: None)
    monkeypatch.setattr('vllm_kvmem.native_state.NativeRecurrentState', lambda _: state)
    monkeypatch.setattr('vllm_kvmem.cache_groups.full_attention_groups', lambda _: (0,))
    monkeypatch.setattr('vllm_kvmem.publication_step.native_publication_step', lambda step: step)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=lambda: None))
    monkeypatch.delenv('KVMEM_VERIFY_SESSION_RESTORE', raising=False)
    runtime = runner._kvmem_capture_runtime
    runtime.config = NS(select_budget=64, gen_budget=64)
    runtime.last_steps = {'native-parent': [{'history_begin': 49, 'accepted': 1, 'scheduled': 1}]}
    cache = torch.arange(64, dtype=torch.uint8).reshape(2, 16, 1, 2)
    original = cache.clone(), recurrent.clone()
    pages, offsets = torch.tensor([1, 1]), torch.tensor([0, 1])
    monkeypatch.setattr('vllm_kvmem.systemone_worker.tail_views', lambda *args: [(cache, pages, offsets)])
    observed = []
    def read(*args):
        assert torch.equal(recurrent, original[1]) and torch.equal(cache, original[0])
        observed.append(True)
        recurrent.fill_(99)
        counter.fill_(9)
        cache[pages, offsets] = 255
        runner._kvmem_last_batch = 'private'
        if fail: raise RuntimeError('injected forward failure')
        return {'argmax': 3, 'logprobs': [-.5]}
    monkeypatch.setattr('vllm_kvmem.systemone_worker.private_read', read)
    args = (runner, NS(req_ids=['native-parent'], num_computed_tokens_np=[50],
        prefill_len_np=[50]), 0, binding, [{'tokens': [1, 2], 'labels': [3]}] * 3)
    if fail:
        with pytest.raises(RuntimeError, match='injected forward failure'): execute_reads(*args)
    else:
        assert len(execute_reads(*args)['reads']) == 3 and len(observed) == 3
    assert torch.equal(recurrent, original[1]) and torch.equal(cache, original[0])
    assert counter.item() == 1
    assert all(getattr(runner, name) is value for name, value in saved.items())
