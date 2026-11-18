"""Which GPUs run the score overlay on FA2 kernels alone (fa2_only)."""
import pytest

from vllm_kvmem.serving.native_attention import fa2_overlay_capability


def test_sm80_and_sm89_by_default_sm89_can_opt_out(monkeypatch):
    monkeypatch.delenv('KVMEM_NATIVE_FA2_SM89', raising=False)
    assert fa2_overlay_capability((8, 0)) and fa2_overlay_capability((8, 9))
    for other in ((8, 6), (9, 0), (12, 0), (7, 5)):
        assert not fa2_overlay_capability(other)
    monkeypatch.setenv('KVMEM_NATIVE_FA2_SM89', '0')
    assert not fa2_overlay_capability((8, 9)) and fa2_overlay_capability((8, 0))
    monkeypatch.setenv('KVMEM_NATIVE_FA2_SM89', '1')
    assert fa2_overlay_capability((8, 9))
    monkeypatch.setenv('KVMEM_NATIVE_FA2_SM89', 'yes')
    with pytest.raises(ValueError):
        fa2_overlay_capability((8, 9))
