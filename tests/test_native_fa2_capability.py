"""Which GPUs run the score overlay on FA2 kernels alone (fa2_only)."""
import pytest

from vllm_kvmem.serving.native_attention import fa2_overlay_capability


def test_sm80_always_and_sm89_only_with_its_opt_in(monkeypatch):
    monkeypatch.delenv('KVMEM_NATIVE_FA2_SM89', raising=False)
    assert fa2_overlay_capability((8, 0))
    assert not fa2_overlay_capability((8, 9))
    for other in ((8, 6), (9, 0), (12, 0), (7, 5)):
        assert not fa2_overlay_capability(other)
    monkeypatch.setenv('KVMEM_NATIVE_FA2_SM89', '1')
    assert fa2_overlay_capability((8, 9)) and fa2_overlay_capability((8, 0))
    assert not fa2_overlay_capability((12, 0))
    monkeypatch.setenv('KVMEM_NATIVE_FA2_SM89', 'yes')
    with pytest.raises(ValueError):
        fa2_overlay_capability((8, 9))
