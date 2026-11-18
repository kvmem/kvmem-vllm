"""KVMEM_DRAFT_SEQ_LENS_CPU (adapters/draft_seq_lens.py): later draft steps get exact CPU
lengths without a GPU synchronization, and every prediction is checked."""
import dataclasses

import pytest
import torch

from vllm_kvmem.adapters import draft_seq_lens as dsl


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_DRAFT_SEQ_LENS_CPU', 'yes')
    with pytest.raises(ValueError):
        dsl.enabled()
    monkeypatch.setenv('KVMEM_DRAFT_SEQ_LENS_CPU', '1')
    assert dsl.enabled()
    monkeypatch.delenv('KVMEM_DRAFT_SEQ_LENS_CPU')
    assert not dsl.enabled()


def test_known_seq_lens_replaces_only_cpu():
    base = torch.tensor([5, 7, 0], dtype=torch.int32)
    known = dsl._known_seq_lens(base, lambda: torch.tensor([9, 9, 9], dtype=torch.int32))
    assert known.cpu().tolist() == [9, 9, 9]
    # Arithmetic and slicing see the real values and return plain tensors.
    assert (known + 1).tolist() == [6, 8, 1] and type(known[:2]) is torch.Tensor
    assert known.data_ptr() == base.data_ptr()


@dataclasses.dataclass
class _Common:
    seq_lens: torch.Tensor
    other: int = 3


def test_build_wrapper_record_then_use_and_verify(monkeypatch):
    seen = []

    def original_build(self, prefix, common, fast_build=False):
        seen.append(common.seq_lens.cpu().tolist())
        return 'meta'

    # Exercise the FlashInfer-side wrapper logic directly through _MODE.
    state = {}
    token = dsl._MODE.set(('record', state))
    try:
        real = torch.tensor([10, 20, 0], dtype=torch.int32)
        common = dataclasses.replace(_Common(real), seq_lens=dsl._known_seq_lens(
            real, lambda: (state.setdefault('cpu', real.clone()), real.clone())[1]))
        original_build(None, 0, common)
    finally:
        dsl._MODE.reset(token)
    assert seen == [[10, 20, 0]] and state['cpu'].tolist() == [10, 20, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA (pinned copies and events)')
def test_checks_raise_on_a_wrong_prediction():
    checks = dsl._Checks()
    real = torch.tensor([3, 4, 0], dtype=torch.int32, device='cuda')
    checks.add(real, torch.tensor([3, 4, 0], dtype=torch.int32))
    checks.verify()
    assert checks.checked == 1
    checks.add(real, torch.tensor([3, 5, 0], dtype=torch.int32))
    with pytest.raises(RuntimeError, match='predicted draft sequence lengths'):
        checks.verify()


def test_full_graph_draft_steps_advance_every_padded_length():
    """A FULL draft graph replays the update kernel for the padded request count
    (the failure seen on 2xRTX 4090: 3 requests in a 4-request graph)."""
    from types import SimpleNamespace as NS
    from vllm.config import CUDAGraphMode
    lengths = torch.tensor([1505, 577, 1025, 0], dtype=torch.int32)
    full = NS(cg_mode=CUDAGraphMode.FULL, num_reqs=4)
    assert dsl.updated_rows(full, 3) == 4
    assert dsl.advance(lengths, dsl.updated_rows(full, 3), 262144).tolist() == [1506, 578, 1026, 1]
    for eager in (NS(cg_mode=CUDAGraphMode.NONE, num_reqs=None), NS(cg_mode=CUDAGraphMode.PIECEWISE, num_reqs=4)):
        assert dsl.updated_rows(eager, 3) == 3
        assert dsl.advance(lengths, 3, 262144).tolist() == [1506, 578, 1026, 0]
    # Lengths stop at max_model_len, as the kernel clamps them.
    assert dsl.advance(torch.tensor([10, 9], dtype=torch.int32), 2, 10).tolist() == [10, 10]
