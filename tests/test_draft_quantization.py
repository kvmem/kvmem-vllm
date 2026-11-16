from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')


def test_context_projection_uses_codec_forward_instead_of_transposed_weight_storage():
    from vllm_kvmem.adapters.draft_quantization import install_quantized_context
    torch.manual_seed(73)
    weights = [torch.randn(8, 5) for _ in range(2)]
    calls = []
    class Projection:
        def __init__(self, index):
            self.index = index
            self.weight = weights[index].t()  # Native packed/transposed storage
        def __call__(self, value):
            calls.append(self.index)
            return torch.nn.functional.linear(value, weights[self.index]), None
    model = NS(hidden_norm=lambda x: x,
        layers=[NS(self_attn=NS(qkv_proj=Projection(i), q_size=4, kv_size=2)) for i in range(2)])
    runner = NS(speculative_config=NS(method='dflash', quantization='fp8'),
                speculator=NS(model=NS(model=model)))
    install_quantized_context(runner)
    inputs = torch.randn(3, 5)
    keys, values = model._project_context_kv(inputs, 3, 2, 1, 2)
    independent = [torch.nn.functional.linear(inputs, w) for w in weights]
    assert calls == [0, 1]
    torch.testing.assert_close(keys, torch.stack([x[:, 4:6].reshape(3, 1, 2) for x in independent]))
    torch.testing.assert_close(values, torch.stack([x[:, 6:8].reshape(3, 1, 2) for x in independent]))
def test_mtp_inherited_fp8_does_not_enter_dflash_patch(monkeypatch):
    from types import SimpleNamespace
    from vllm_kvmem.adapters.draft_quantization import install_quantized_context
    import vllm_kvmem.adapters.draft_contract as contract

    def forbidden():
        raise AssertionError('Native MTP must not run DFlash verification')

    monkeypatch.setattr(contract, 'verify_draft_sources', forbidden)
    # No DFlash model is provided: it must not even be accessed.
    runner = SimpleNamespace(speculative_config=SimpleNamespace(method='mtp', quantization='fp8'))
    install_quantized_context(runner)


@pytest.mark.parametrize('quantization', [None, 'fp8', 'compressed-tensors'])
def test_other_draft_modes_do_not_patch_loader(monkeypatch, quantization):
    import vllm_kvmem.adapters.draft_quantization as adapter

    def forbidden():
        raise AssertionError('Unrelated model load must not import DFlash2')

    monkeypatch.setattr(adapter, '_native_dflash2_type', forbidden)
    runner = NS(speculative_config=NS(method='mtp', quantization=quantization))
    with adapter.prepare_quantized_context(runner):
        pass
    if quantization != 'compressed-tensors':
        runner.speculative_config.method = 'dflash'
        with adapter.prepare_quantized_context(runner):
            pass


@pytest.mark.parametrize('fail_load', [False, True])
def test_packed_draft_load_keeps_native_codec_and_restores_class(monkeypatch, fail_load):
    import vllm_kvmem.adapters.draft_quantization as adapter
    import vllm_kvmem.adapters.draft_contract as contract

    class Parent:
        def _build_context_kv_buffers(self, *args):
            raise AssertionError('BF16 weight buffer must not read packed storage')

    class Model(Parent):
        pass

    monkeypatch.setattr(adapter, '_native_dflash2_type', lambda: Model)
    monkeypatch.setattr(contract, 'verify_draft_sources', lambda: None)
    torch.manual_seed(91)
    weights = [torch.randn(8, 5) for _ in range(2)]
    calls = []

    class PackedProjection:
        # Packed layers expose no ordinary .weight. The native codec owns math.
        def __init__(self, index): self.index = index
        def __call__(self, x):
            calls.append(self.index)
            return torch.nn.functional.linear(x, weights[self.index]), None

    model = Model()
    model.hidden_norm = torch.nn.LayerNorm(5)
    attns = [NS(qkv_proj=PackedProjection(i), q_size=4, kv_size=2,
                k_norm=NS(weight=torch.ones(2))) for i in range(2)]
    model.layers = [NS(self_attn=a) for a in attns]
    runner = NS(speculative_config=NS(method='dflash', quantization=None,
        draft_model_config=NS(quantization='compressed-tensors')))
    try:
        with adapter.prepare_quantized_context(runner):
            model._build_context_kv_buffers(attns, False)
            if fail_load:
                raise RuntimeError('loader failure')
    except RuntimeError as error:
        assert fail_load and str(error) == 'loader failure'
    assert '_build_context_kv_buffers' not in Model.__dict__
    inputs = torch.randn(3, 5)
    keys, values = model._project_context_kv(inputs, 3, 2, 1, 2)
    expected = [torch.nn.functional.linear(model.hidden_norm(inputs), w) for w in weights]
    assert calls == [0, 1]
    torch.testing.assert_close(keys, torch.stack([x[:,4:6].reshape(3,1,2) for x in expected]))
    torch.testing.assert_close(values, torch.stack([x[:,6:8].reshape(3,1,2) for x in expected]))
