from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.native_pages import kernel_table, kernel_expansions, stored_expansions


def logical_runner():
    return NS(kernel_block_sizes=[4], block_tables=NS(block_sizes=[8],
        num_blocks=NS(np=torch.tensor([[3]]).numpy()),
        block_tables=[NS(gpu=torch.tensor([[0, 3, 8, 99]], dtype=torch.int32))]))


def test_logical_and_expanded_tables_address_the_same_kv_pages():
    logical = logical_runner()
    old = NS(block_tables=NS(blocks_per_kv_block=[2], kernel_block_sizes=[4],
        num_blocks=NS(np=torch.tensor([[6]]).numpy()),
        block_tables=[NS(gpu=torch.tensor([[0, 1, 6, 7, 16, 17, 99]], dtype=torch.int32))]))
    assert kernel_table(logical, 0, 0).tolist() == kernel_table(old, 0, 0).tolist()
    assert stored_expansions(logical.block_tables) == (1,)
    assert stored_expansions(old.block_tables) == (2,)
    assert kernel_expansions(logical) == kernel_expansions(old) == (2,)


def test_latest_restore_witness_counts_allocator_pages_but_returns_kernel_pages(monkeypatch):
    from vllm_kvmem.restore_lease import FreshRestoreLease
    runner = logical_runner()
    runner.block_tables.num_blocks.np[0, 0] = 2
    runner.req_states = NS(req_id_to_index={'r': 0})
    runner._kvmem_request_generations = {'r': 7}
    runner._kvmem_restore_allocations = {}
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda *_: pytest.fail('GPU readback'))
    lease = FreshRestoreLease.capture(runner, 'r', ((3, 8),))
    runner._kvmem_restore_allocations['r'] = lease
    lease.validate(runner, expected_block_ids=((3, 8),))
    assert lease.group_rows(0) == (6, 7, 16, 17)
    runner.block_tables.num_blocks.np[0, 0] = 4
    with pytest.raises(ValueError, match='page count changed'):
        lease.validate(runner)


def test_unknown_geometry_and_redirected_warmup_cannot_be_borrowed():
    runner = logical_runner()
    runner.kernel_block_sizes = [3]
    with pytest.raises(ValueError, match='incompatible'):
        kernel_table(runner, 0, 0)
    runner.block_tables.redirect_writes_to_null_block = True
    with pytest.raises(ValueError, match='warmup'):
        stored_expansions(runner.block_tables)
