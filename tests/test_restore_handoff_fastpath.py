"""Preserve exact prefix identity and fenced ownership while shortening handoff."""
import json
import random
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.prompt_checkpoints import token_bytes


@pytest.mark.parametrize('container', [list, tuple])
def test_native_prefix_matches_reference_bytes_and_ignores_suffix(container):
    rng=random.Random(7)
    for count in [1, 31, 32, 33, 1024, 262144]:
        tokens=container([rng.randrange(2**32) for _ in range(count)]+[object()])
        assert token_bytes(tokens,count,native=True)==token_bytes(tokens,count)


class IntSubclass(int):
    pass


@pytest.mark.parametrize('bad', [True, False, -1, 2**32, 2**80, 1., None, '1', IntSubclass(1)])
def test_native_identity_rejects_coercion_and_overflow(bad):
    for native in (False,True):
        with pytest.raises(ValueError,match='exact unsigned'):
            token_bytes([0,bad,1],2,native=native)


@pytest.mark.parametrize('count', [-1,0,3,4])
def test_native_identity_preserves_prefill_tail_contract(count):
    with pytest.raises(ValueError,match='at least one token'):
        token_bytes([0,1,2],count,native=True)


def test_priority_promotes_prepared_resident_but_preserves_aging_and_fences():
    from test_maintenance import policy,job
    p=policy(restore_handoff_fastpath=True)
    a,b=job('prepared',stage='restore_model_prepare'),job('new')
    for j,size in [(a,157876224),(b,0)]:
        j.command.resident=True;j.model_bytes=size
    old,complete=job('old',age=1.),job('complete',stage='assembly')
    assert p.order([b,a,old,complete])==[complete,old,a,b]
    p.restore_handoff_fastpath=False
    assert p.order([a,b])==[b,a]  # Explicit legacy control remains available.


@pytest.mark.parametrize('cancelled',[False,True])
def test_post_forward_ack_keeps_lease_until_future_and_cuda_complete(tmp_path,cancelled):
    from test_async_restore import restore_inflight
    from test_connector import pair
    manager,job,future,fence,trace,checkpoint,authority,archive=restore_inflight(tmp_path)
    manager.options.restore_handoff_fastpath=True
    job.cancelled=cancelled
    _,connector=pair()
    connector.restore_completion_poll=manager.poll_restore_completions
    connector.worker_jobs={'new':(job.command,job)}
    try:
        connector.get_finished(set())
        assert not job.done and future.reads==0 and not connector.worker_completions
        future.ready=True
        connector.get_finished(set())
        assert not job.done and future.reads==1 and not connector.worker_completions
        assert 'new' in manager.runner._kvmem_restore_allocations
        fence.done=True
        connector.get_finished(set())
        assert job.done and connector.worker_completions==[('new',-1)]
        assert not manager.runner._kvmem_restore_allocations
        connector.get_finished(set())
        assert connector.worker_completions==[('new',-1)]  # No duplicate ACK.
        row=json.loads((tmp_path/'async_restore.jsonl').read_text())
        assert row['completion_poll']=='post_forward' and row['cancelled']==cancelled
    finally:
        manager.runner._kvmem_sessions.pool.drop_live(checkpoint)
        assert authority.close() and archive.close()


def test_post_forward_never_starts_prepare_or_query_and_honors_publication():
    from vllm_kvmem.async_refresh import AsyncRefresh
    manager=AsyncRefresh.__new__(AsyncRefresh)
    manager.options=NS(restore_handoff_fastpath=True)
    manager.runner=NS(_kvmem_capture_runtime=NS(publication_ready=lambda ids:ids!=('blocked',)))
    manager.jobs={name:NS(command=NS(request_id=name),stage=stage,done=False) for name,stage in
                  [('cold','prepare'),('ready','restore_model_prepare'),('query','query'),
                   ('blocked','restore_copy'),('copy','restore_copy')]}
    called=[];manager.step=lambda j:called.append(j.command.request_id)
    manager.poll_restore_completions()
    assert called==['copy']
    manager.options.restore_handoff_fastpath=False
    manager.poll_restore_completions()
    assert called==['copy']


def test_fastpath_option_is_explicit_and_requires_async_split():
    from vllm_kvmem.options import RuntimeOptions
    from test_options import config
    assert RuntimeOptions.from_vllm(config({})).restore_handoff_fastpath is False
    for flag in [True,1,'true']:
        with pytest.raises(ValueError,match='Restore handoff fastpath'):
            RuntimeOptions.from_vllm(config({'kvmem':{'restore_handoff_fastpath':flag}}))
