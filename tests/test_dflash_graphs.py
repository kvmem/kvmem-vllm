from types import SimpleNamespace, MethodType
import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as native
from vllm_kvmem.adapters import draft_contract
from vllm_kvmem.adapters.draft_graphs import install_independent_draft_graphs


@pytest.mark.parametrize("target,support,expected", [
    (CUDAGraphMode.PIECEWISE, AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.FULL_DECODE_ONLY),
    (CUDAGraphMode.PIECEWISE, AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, CUDAGraphMode.NONE),
    (CUDAGraphMode.NONE, AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.NONE),
])
def test_independent_policy_preserves_native_backend_checks(monkeypatch,target,support,expected):
    monkeypatch.setattr(draft_contract,"verify_draft_sources",lambda: None)
    seen=[]
    def manager(config,device,mode,*,decode_query_len):
        seen.append((mode,decode_query_len))
        return SimpleNamespace(cudagraph_mode=mode)
    monkeypatch.setattr(native,"DFlashCudaGraphManager",manager)
    draft=SimpleNamespace(vllm_config=object(),device="cuda",num_query_per_req=2,
        attn_cg_support=SimpleNamespace(min_cg_support=support,min_cg_attn_backend="test"),
        _speculator_name="DFlash2")
    draft.init_cudagraph_manager=MethodType(native.DFlashSpeculator.init_cudagraph_manager,draft)
    runner=SimpleNamespace(speculator=draft)
    install_independent_draft_graphs(runner)
    draft.init_cudagraph_manager(target)
    assert seen==[(expected,2)]
    with pytest.raises(ValueError,match="already installed"):
        install_independent_draft_graphs(runner)


def test_deferred_capture_submit_runs_after_the_draft_and_falls_back(monkeypatch):
    from types import SimpleNamespace
    from vllm_kvmem.adapters import draft_graphs
    order = []
    runtime = SimpleNamespace(submit=lambda *args: order.append(('submit', args)))
    speculator = SimpleNamespace(propose=lambda *args, **kwargs: order.append(('draft', args)) or 'drafts')
    runner = SimpleNamespace(speculator=speculator, _kvmem_capture_runtime=runtime)
    draft_graphs.install_deferred_capture_submit(runner)
    assert runner._kvmem_defer_dflash_submit
    runner._kvmem_deferred_submit = ('captured', 'batch', 'rejected')
    assert speculator.propose('inputs') == 'drafts'
    assert order == [('draft', ('inputs',)), ('submit', ('captured', 'batch', 'rejected'))]
    assert runner._kvmem_deferred_submit is None
    assert not draft_graphs.flush_deferred_submit(runner)        # nothing waiting
    runner._kvmem_deferred_submit = ('c2', 'b2', 'r2')            # a step without a proposal
    assert draft_graphs.flush_deferred_submit(runner)
    assert order[-1] == ('submit', ('c2', 'b2', 'r2'))
    assert runner._kvmem_deferred_submit_counts == {'after_draft': 1, 'fallback': 1}
    with pytest.raises(ValueError):
        draft_graphs.install_deferred_capture_submit(runner)


def test_deferred_capture_submit_flag_is_validated(monkeypatch):
    from vllm_kvmem.adapters.draft_graphs import deferred_submit_enabled
    monkeypatch.setenv('KVMEM_DFLASH_SUBMIT_AFTER_DRAFT', 'yes')
    with pytest.raises(ValueError):
        deferred_submit_enabled()


def test_staged_capture_runs_before_the_draft_and_submits_after(monkeypatch):
    from types import SimpleNamespace
    from vllm_kvmem.adapters import draft_graphs
    monkeypatch.setenv('KVMEM_DFLASH_STAGE_BEFORE_DRAFT', '1')
    order = []
    action = SimpleNamespace(stage_before_mtp=lambda captured, batch: order.append(('stage', captured, batch)) or True,
                             _mtp_capture_frame=None, cancel_before_mtp=lambda: order.append('cancel'))
    runtime = SimpleNamespace(submit=lambda *args: order.append(('submit', args)), action=action,
                              quarantine_publication=lambda error: order.append(('quarantine', str(error))))
    speculator = SimpleNamespace(propose=lambda *a, **k: order.append('draft') or 'drafts')
    runner = SimpleNamespace(speculator=speculator, _kvmem_capture_runtime=runtime)
    draft_graphs.install_deferred_capture_submit(runner)
    runner._kvmem_deferred_submit = ('cap', 'batch', 'rej')
    assert speculator.propose() == 'drafts'
    assert order == [('stage', 'cap', 'batch'), 'draft', ('submit', ('cap', 'batch', 'rej'))]


def test_failed_draft_after_staging_quarantines_publication(monkeypatch):
    from types import SimpleNamespace
    from vllm_kvmem.adapters import draft_graphs
    monkeypatch.setenv('KVMEM_DFLASH_STAGE_BEFORE_DRAFT', '1')
    order = []
    action = SimpleNamespace(stage_before_mtp=lambda *a: True, _mtp_capture_frame=None,
                             cancel_before_mtp=lambda: order.append('cancel'))
    runtime = SimpleNamespace(submit=lambda *a: order.append('submit'), action=action,
                              quarantine_publication=lambda error: order.append(('quarantine', str(error))))
    def broken(*a, **k):
        raise RuntimeError('draft failed')
    runner = SimpleNamespace(speculator=SimpleNamespace(propose=broken), _kvmem_capture_runtime=runtime)
    draft_graphs.install_deferred_capture_submit(runner)
    runner._kvmem_deferred_submit = ('cap', 'batch', 'rej')
    with pytest.raises(RuntimeError, match='draft failed'):
        runner.speculator.propose()
    assert order == [('quarantine', 'draft failed'), 'cancel'] and runner._kvmem_deferred_submit is None
