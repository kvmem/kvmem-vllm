"""CPU-only ownership and no-readback oracles for fresh import allocations."""
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.serving.restore_lease import FreshRestoreLease, verify_restore_allocation_sources


def allocation():
    runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={'restore': 0, 'peer': 1},
            num_computed_tokens_np=[8, 8], prefill_len=SimpleNamespace(np=[12, 12])),
        _kvmem_request_generations={'restore': 7, 'peer': 8},
        _kvmem_restore_allocations={},
        block_tables=SimpleNamespace(blocks_per_kv_block=[1, 2],
            num_blocks=SimpleNamespace(np=torch.tensor([[2, 2], [4, 4]]).numpy()),
            block_tables=[SimpleNamespace(gpu=torch.tensor([[2, 3], [4, 5]])),
                          SimpleNamespace(gpu=torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]]))]),
        model_state=SimpleNamespace(num_accepted_tokens_gpu=torch.tensor([7, 9])))
    lease = FreshRestoreLease.capture(runner, 'restore', ((2, 3), (4, 5)))
    runner._kvmem_restore_allocations['restore'] = lease
    return runner, lease


def test_installed_native_allocation_lifecycle_matches_pinned_contract():
    verify_restore_allocation_sources()


def test_cpu_witness_expands_kernel_pages_and_checks_exact_command_without_readback(monkeypatch):
    runner, lease = allocation()
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda *_: pytest.fail('GPU page readback'))
    monkeypatch.setattr(torch.Tensor, 'item', lambda *_: pytest.fail('GPU scalar readback'))
    lease.validate(runner, expected_block_ids=((2, 3), (4, 5)))
    assert lease.rows == ((2, 3), (8, 9, 10, 11))
    with pytest.raises(ValueError, match='native allocation witness'):
        lease.validate(runner, expected_block_ids=((2, 3), (5, 4)))
    with pytest.raises(ValueError, match='native cache arena'):
        lease.group_rows(1, page_count=11)


@pytest.mark.parametrize('change', ['generation', 'index', 'count', 'registration'])
def test_stale_cpu_witness_rejected(change):
    runner, lease = allocation()
    if change == 'generation':
        runner._kvmem_request_generations['restore'] += 1
    elif change == 'index':
        runner.req_states.req_id_to_index['restore'] = 1
    elif change == 'count':
        runner.block_tables.num_blocks.np[1, 0] = 3
    else:
        runner._kvmem_restore_allocations.pop('restore')
    with pytest.raises(ValueError):
        lease.validate(runner)


@pytest.mark.parametrize('blocks', [((0, 3), (4, 5)), ((2, 2), (4, 5)), ((2, 3), (4.0, 5)),
                                   ((2, 3),), ((2, 3), ())])
def test_invalid_native_pages_cannot_form_a_witness(blocks):
    runner, _ = allocation()
    with pytest.raises(ValueError):
        FreshRestoreLease.capture(runner, 'restore', blocks)


def prepared_state(monkeypatch):
    from vllm_kvmem.serving.native_state import NativeRecurrentState
    from vllm_kvmem.serving.session_state import PromptStateAdapter, PromptModelState
    runner, lease = allocation()
    conv, ssm = torch.arange(8*4.).reshape(8, 4), torch.arange(8*6.).reshape(8, 2, 3)
    recurrent = NativeRecurrentState.__new__(NativeRecurrentState)
    recurrent.runner, recurrent.groups = runner, {0: {'gdn': (conv, ssm)}}
    monkeypatch.setattr(recurrent, 'borrow', lambda *_: pytest.fail('Generic GPU row readback'))
    adapter = PromptStateAdapter.__new__(PromptStateAdapter)
    adapter.runner, adapter.recurrent, adapter._identity = runner, recurrent, object()
    adapter.contract = SimpleNamespace(method='mtp', validate_native_groups=lambda _: (1,))
    snapshot = PromptModelState(adapter._identity, 8, 8,
        {('gdn', 'conv'): torch.full((4,), 123.), ('gdn', 'ssm'): torch.full((2, 3), 456.)}, {})
    return adapter, snapshot, lease, conv, ssm


def test_state_restore_uses_cpu_allocation_rows_and_preserves_peer_without_readback(monkeypatch):
    adapter, snapshot, lease, conv, ssm = prepared_state(monkeypatch)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda *_: pytest.fail('GPU page readback'))
    monkeypatch.setattr(torch.Tensor, 'item', lambda *_: pytest.fail('GPU scalar readback'))
    prepared = adapter.prepare_restore(snapshot, 'restore', lease=lease)
    assert adapter.submit_restore(prepared) is None
    adapter.validate_restore(prepared)
    assert torch.equal(conv[2], snapshot.recurrent['gdn', 'conv'])
    assert torch.equal(ssm[2], snapshot.recurrent['gdn', 'ssm'])
    assert torch.equal(conv[3:], before_conv[3:]) and torch.equal(ssm[3:], before_ssm[3:])
    assert adapter.runner.model_state.num_accepted_tokens_gpu.tolist() == [1, 9]


def test_recycled_state_lease_is_rejected_before_first_destination_write(monkeypatch):
    adapter, snapshot, lease, conv, ssm = prepared_state(monkeypatch)
    prepared = adapter.prepare_restore(snapshot, 'restore', lease=lease)
    before = conv.clone(), ssm.clone()
    adapter.runner._kvmem_restore_allocations.clear()
    with pytest.raises(ValueError, match='no longer leased'):
        adapter.submit_restore(prepared)
    with pytest.raises(ValueError, match='no longer leased'):
        adapter.validate_restore(prepared)
    assert torch.equal(conv, before[0]) and torch.equal(ssm, before[1])
    assert adapter.runner.model_state.num_accepted_tokens_gpu.tolist() == [7, 9]


def test_lease_path_does_not_silently_change_dflash_state_semantics(monkeypatch):
    adapter, snapshot, lease, *_ = prepared_state(monkeypatch)
    adapter.contract.method = 'dflash'
    with pytest.raises(ValueError, match='target-only or native MTP'):
        adapter.prepare_restore(snapshot, 'restore', lease=lease)


def test_assembler_clones_leased_native_table_without_scalar_gpu_validation(monkeypatch):
    from vllm_kvmem.serving import window_assembly as window
    runner, lease = allocation()
    key = torch.zeros((16, 4, 1, 4), dtype=torch.float8_e4m3fn)
    value = key.clone()
    module = SimpleNamespace(attn=SimpleNamespace(kv_cache_dtype='fp8_e4m3', layer_name='attn',
        kv_cache=(key, value), _k_scale=torch.tensor(1.)), num_kv_heads=1, head_dim=4,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(64, 4)))
    runner.model = SimpleNamespace(named_modules=lambda: [('layer', module)])
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['gdn']),
        SimpleNamespace(layer_names=['attn'])])
    monkeypatch.setattr(window, 'native_vllm_fp8_views', lambda cache, *_: cache)
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda *_: pytest.fail('GPU table readback'))
    monkeypatch.setattr(torch, 'any', lambda *_: pytest.fail('GPU scalar validation'))
    assembler = window.NativeWindowAssembler.from_runner(runner, 'restore', ('layer',), 4, lease=lease)
    assert assembler.layers[0].kernel_page_table.tolist() == [8, 9, 10, 11]
    assert assembler.layers[0].kernel_page_table.data_ptr() != runner.block_tables.block_tables[1].gpu.data_ptr()
    # A stale allocation fails before assembly claims shared scratch or writes KV.
    runner._kvmem_restore_allocations.clear()
    with pytest.raises(ValueError, match='no longer leased'):
        assembler.assemble(None, None)
    assert assembler.staging.owner is None and assembler.close()


def test_generic_assembler_retains_its_gpu_table_validation():
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler, LayerWindow
    key = torch.zeros((4, 4, 1, 4), dtype=torch.float8_e4m3fn)
    layer = LayerWindow(key, key.clone(), torch.zeros(16, 4), torch.tensor(1.), torch.tensor([0, 2]))
    with pytest.raises(ValueError, match='null or out-of-arena'):
        NativeWindowAssembler((layer,), 4)


def test_direct_restore_layer_borrows_leased_view_without_gpu_validation(monkeypatch):
    """KVMEM_RESTORE_DIRECT_LAYERS: a per-layer assembler built from the restore
    job's own lease-validated view needs no table clone or scalar GPU check."""
    from vllm_kvmem.serving import window_assembly as window
    runner, lease = allocation()
    key = torch.zeros((16, 4, 1, 4), dtype=torch.float8_e4m3fn)
    module = SimpleNamespace(attn=SimpleNamespace(kv_cache_dtype='fp8_e4m3', layer_name='attn',
        kv_cache=(key, key.clone()), _k_scale=torch.tensor(1.)), num_kv_heads=1, head_dim=4,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(64, 4)))
    runner.model = SimpleNamespace(named_modules=lambda: [('layer', module)])
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['gdn']),
        SimpleNamespace(layer_names=['attn'])])
    monkeypatch.setattr(window, 'native_vllm_fp8_views', lambda cache, *_: cache)
    parent = window.NativeWindowAssembler.from_runner(runner, 'restore', ('layer',), 4, lease=lease)
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda *_: pytest.fail('GPU table readback'))
    monkeypatch.setattr(torch, 'any', lambda *_: pytest.fail('GPU scalar validation'))
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *_: pytest.fail('table clone'))
    child = window.NativeWindowAssembler((parent.layers[0],), 4, 8, staging=parent.staging,
        borrowed_staging=True, raw_dtype=torch.bfloat16, lease=lease)
    assert child.layers[0] is parent.layers[0] and child.staging is parent.staging
    assert child.close()
    runner._kvmem_restore_allocations.clear()     # a recycled allocation is still rejected
    with pytest.raises(ValueError, match='no longer leased'):
        window.NativeWindowAssembler((parent.layers[0],), 4, 8, staging=parent.staging,
            borrowed_staging=True, raw_dtype=torch.bfloat16, lease=lease)


def test_sliding_group_may_hold_null_pages_only_when_declared():
    from vllm_kvmem.serving.restore_lease import _blocks
    blocks = ((2, 3), (0, 0, 6, 7))
    assert _blocks(blocks, sliding=(1,)) == blocks
    with pytest.raises(ValueError):
        _blocks(blocks)                              # the strict rule without a sliding group
    for bad in (((0, 3), (0, 6)), ((2, 3), (0, 6, 6)), ((2, 3), (0, 0)), ((2, 3), (0, -1))):
        with pytest.raises(ValueError):
            _blocks(bad, sliding=(1,))               # null target page, alias, all-null, negative


def test_dflash_async_lease_keeps_null_sliding_pages_and_checks_the_command(monkeypatch):
    from vllm_kvmem.serving import restore_lease
    runner, _ = allocation()
    runner.block_tables.num_blocks.np[1, 0] = 4
    monkeypatch.setattr(restore_lease, 'sliding_groups', lambda _runner: (1,))
    lease = FreshRestoreLease.capture(runner, 'restore', ((2, 3), (0, 5)))
    runner._kvmem_restore_allocations['restore'] = lease
    assert lease.sliding == (1,) and lease.rows == ((2, 3), (0, 1, 10, 11))
    lease.validate(runner, expected_block_ids=((2, 3), (0, 5)))
    with pytest.raises(ValueError, match='native allocation witness'):
        lease.validate(runner, expected_block_ids=((2, 3), (5, 0)))


def test_sliding_groups_require_the_dflash_async_opt_in(monkeypatch):
    from vllm_kvmem.serving.restore_lease import sliding_groups
    monkeypatch.delenv('KVMEM_DFLASH_ASYNC_REFRESH', raising=False)
    assert sliding_groups(SimpleNamespace()) == ()
