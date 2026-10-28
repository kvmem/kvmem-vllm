from types import SimpleNamespace
import pytest
import torch

from vllm_kvmem.checkpoint_prewarm import prewarm_checkpoint_payloads, prewarm_count


def test_reservation_accounts_for_allocator_rounding_and_concurrency():
    assert prewarm_count(157876224, 64, 8, 4*1024**3) == (10, 256*1024**2)
    assert prewarm_count(157876224, 10, 8, 512*1024**2) == (2, 256*1024**2)
    assert prewarm_count(157876224, 10, 8, 128*1024**2) == (0, 256*1024**2)
    with pytest.raises(ValueError):
        prewarm_count(1, True, 8, 512)


def test_disabled_does_not_access_runner_or_allocate():
    assert prewarm_checkpoint_payloads(None, SimpleNamespace(checkpoint_pinned_prewarm_count=0)) is None


def test_no_gpu_data_or_snapshot_lifetime_changed(monkeypatch, tmp_path):
    import vllm_kvmem.native_state as native
    a=torch.ones((7,3,5),dtype=torch.float32)
    b=torch.ones((7,2,4),dtype=torch.float32)
    monkeypatch.setattr(native, 'NativeRecurrentState', lambda _:SimpleNamespace(groups={0:{'layer':(a,b)}}))
    calls=[]
    original=torch.empty
    def allocate(size, **kw):
        calls.append((size,kw));return original(size,dtype=kw['dtype'],device='cpu')
    monkeypatch.setattr(torch,'empty',allocate)
    runner=SimpleNamespace(speculative_config=SimpleNamespace(method='mtp'),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=8)),
        _kvmem_probe_output=tmp_path)
    options=SimpleNamespace(checkpoint_pinned_prewarm_count=10,session_capacity=16,
                            session_checkpoint_bytes=512)
    report=prewarm_checkpoint_payloads(runner,options)
    assert report['count']==4 and report['payload_bytes']==92
    assert all(size==92 and kw==dict(dtype=torch.uint8,device='cpu',pin_memory=True) for size,kw in calls)
    assert torch.all(a==1) and torch.all(b==1)
    runner.speculative_config.method='dflash'
    with pytest.raises(ValueError,match='MTP'):
        prewarm_checkpoint_payloads(runner,options)
