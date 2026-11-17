from types import SimpleNamespace
import pytest
import torch

from vllm_kvmem.serving.checkpoint_prewarm import prewarm_checkpoint_payloads, prewarm_count


def test_reservation_accounts_for_allocator_rounding_and_concurrency():
    assert prewarm_count(157876224, 64, 8, 4*1024**3) == (10, 256*1024**2)
    assert prewarm_count(157876224, 10, 8, 512*1024**2) == (2, 256*1024**2)
    assert prewarm_count(157876224, 10, 8, 128*1024**2) == (0, 256*1024**2)
    with pytest.raises(ValueError):
        prewarm_count(1, True, 8, 512)


def test_disabled_does_not_access_runner_or_allocate():
    assert prewarm_checkpoint_payloads(None, SimpleNamespace(checkpoint_pinned_prewarm_count=0)) is None


def test_no_gpu_data_or_snapshot_lifetime_changed(monkeypatch, tmp_path):
    import vllm_kvmem.serving.native_state as native
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


def test_session_retained_checkpoints_raise_the_concurrency_cap():
    # pm sessions keep a prompt and a terminal checkpoint each: 2*8+1 blocks.
    assert prewarm_count(157876224, 64, 8, 6*1024**3, 17) == (17, 256*1024**2)
    assert prewarm_count(157876224, 12, 8, 6*1024**3, 17) == (12, 256*1024**2)
    assert prewarm_count(157876224, 64, 8, 1024**3, 17) == (4, 256*1024**2)
    assert prewarm_count(157876224, 64, 16, 6*1024**3, 9) == (18, 256*1024**2)


def test_environment_overrides_the_profile_count(monkeypatch):
    from vllm_kvmem.serving.checkpoint_prewarm import requested_count
    options = SimpleNamespace(checkpoint_pinned_prewarm_count=3)
    monkeypatch.delenv('KVMEM_CHECKPOINT_PINNED_PREWARM', raising=False)
    assert requested_count(options) == 3
    monkeypatch.setenv('KVMEM_CHECKPOINT_PINNED_PREWARM', '17')
    assert requested_count(options) == 17
    monkeypatch.setenv('KVMEM_CHECKPOINT_PINNED_PREWARM', '0')
    assert requested_count(options) == 0 and prewarm_checkpoint_payloads(None, options) is None
    for bad in ('65', '-1', 'on', ''):
        monkeypatch.setenv('KVMEM_CHECKPOINT_PINNED_PREWARM', bad)
        with pytest.raises(ValueError, match='KVMEM_CHECKPOINT_PINNED_PREWARM'):
            requested_count(options)


@pytest.mark.parametrize('eager', [True, False])
def test_warming_follows_the_startup_graph_capture(monkeypatch, eager):
    # torch.cuda.graph() empties the pinned host cache on entry, so blocks
    # warmed before the startup capture would be gone at the first checkpoint.
    from vllm_kvmem.serving.runner import make_probe_runner
    events = []

    class Base:
        def initialize_kv_cache(self, is_profiling=False):
            events.append('profile' if is_profiling else 'kv')

        def capture_model(self):
            events.append('graphs')
            return 7

    runner = make_probe_runner(Base)()
    runner.vllm_config = object()
    runner.model_config = SimpleNamespace(enforce_eager=eager)
    monkeypatch.setattr('vllm_kvmem.serving.profile_scopes.install_profile_scopes', lambda _: None)
    monkeypatch.setattr('vllm_kvmem.serving.recurrent_table_journal.install_recurrent_table_journal', lambda *a, **kw: None)
    monkeypatch.setattr('vllm_kvmem.core.options.RuntimeOptions.from_vllm', lambda _: 'options')
    monkeypatch.setattr('vllm_kvmem.serving.checkpoint_prewarm.prewarm_checkpoint_payloads',
                        lambda r, options, stage: events.append((options, stage)))
    runner.initialize_kv_cache(is_profiling=True)
    runner.initialize_kv_cache()
    if eager:  # The worker skips capture_model for an eager engine.
        assert events == ['profile', 'kv', ('options', 'kv_cache_init')]
    else:
        assert runner.capture_model() == 7
        assert events == ['profile', 'kv', 'graphs', ('options', 'after_graph_capture')]
        runner.initialize_kv_cache()
        assert events[4:] == ['kv']  # Warmed once.


def test_graph_memory_profile_does_not_consume_checkpoint_prewarm(monkeypatch):
    from vllm_kvmem.serving.runner import make_probe_runner
    events = []

    class Base:
        def capture_model(self, *, profile_only=False):
            events.append(('capture', profile_only))
            return 11 if profile_only else 7

    runner = make_probe_runner(Base)()
    runner.vllm_config = object()
    monkeypatch.setattr('vllm_kvmem.core.options.RuntimeOptions.from_vllm', lambda _: 'options')
    monkeypatch.setattr('vllm_kvmem.serving.checkpoint_prewarm.prewarm_checkpoint_payloads',
                        lambda r, options, stage: events.append((options, stage)))

    assert runner.capture_model(profile_only=True) == 11
    assert events == [('capture', True)]
    assert not getattr(runner, '_kvmem_checkpoint_prewarmed', False)
    assert runner.capture_model(profile_only=False) == 7
    assert events == [('capture', True), ('capture', False), ('options', 'after_graph_capture')]
    assert runner.capture_model() == 7
    assert events[-1] == ('capture', False)
    assert len(events) == 4


def test_session_less_profiles_skip_warming_under_the_preset(monkeypatch):
    # Flash Next runs without prompt sessions; the default-on preset must not stop it.
    monkeypatch.setenv('KVMEM_CHECKPOINT_PINNED_PREWARM', '17')
    options = SimpleNamespace(checkpoint_pinned_prewarm_count=0, session_capacity=0)
    assert prewarm_checkpoint_payloads(None, options) is None
