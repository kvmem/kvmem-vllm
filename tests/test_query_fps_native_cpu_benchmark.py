"""Benchmark guards must not trade away fixed inputs or exhaustive equality."""
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('benchmark_query_fps_native_cpu')


def operands():
    # Two original blocks separated by a logical hole; backing bytes remain
    # compact. The reader must use metadata, not invent intervening history.
    keys = (torch.arange(2*64*2*4).reshape(2,64,2,4)%17-8).float().to(torch.float8_e4m3fn)
    keys.view(torch.uint8)[0, 0, 0, 0] = 128  # real negative zero
    metadata = dict(history=128, starts=[0,96], counts=[32,32])
    offsets = np.broadcast_to(np.arange(4,dtype=np.int8)[None,None,:,None],(2,2,4,2)).copy()
    return keys.view(torch.uint8).numpy().copy(),metadata,offsets


def values_for(keys, metadata, offsets, blocks):
    begins = np.r_[0,np.cumsum(metadata['counts'])]
    layers=[]
    for layer in range(keys.shape[0]):
        rows=[]
        for block in blocks:
            chosen=offsets[layer,block].astype(np.int64)
            raw=np.array(keys[layer,begins[block]+chosen,np.arange(keys.shape[2])[None,:],:],copy=True)
            rows.append(torch.from_numpy(raw).view(torch.float8_e4m3fn).half())
        layers.append(torch.stack(rows))
    return torch.stack(layers)


def test_owned_reader_keeps_logical_holes_and_returns_immutable_independent_copy(bench):
    keys,metadata,_=operands()
    authority=bench.OwnedKeys(keys,metadata)
    slab,stats=authority.read_score_keys([authority.records[1].handle,authority.records[0].handle])
    assert slab.shape==(2,2,32,2,4)
    assert np.array_equal(slab[0].view(torch.uint8).numpy(),keys[:,32:64])
    assert [r.history_begin for r in authority.records]==[0,96]
    keys[:]=0
    assert bool(slab.float().abs().sum()>0)
    assert stats['key_read_bytes']==slab.numel() and stats['value_read_bytes']==0


@pytest.mark.parametrize('fault',['generation','duplicate','pinned','count'])
def test_reader_rejects_nonoriginal_or_unbounded_access(bench,fault):
    keys,metadata,_=operands()
    if fault=='count':metadata['counts']=[31,33]
    with pytest.raises(ValueError):
        authority=bench.OwnedKeys(keys,metadata)
        handle=authority.records[0].handle
        if fault=='generation':handle=NS(slot=0,generation=2)
        authority.read_score_keys([handle,handle] if fault=='duplicate' else [handle],
            pin_memory=fault=='pinned')


def test_every_selected_offset_and_original_key_byte_is_checked(bench):
    keys,metadata,offsets=operands()
    blocks=[1,0]
    values=values_for(keys,metadata,offsets,blocks)
    indices=torch.from_numpy(offsets[:,blocks].copy())
    result=bench.validate_every_representative(values,indices,keys,metadata,offsets,blocks)
    assert result['checked_offsets']==2*2*4*2
    assert result['exact_original_fp8_bytes'] and result['exact_cache_bytes']
    # Preserve floating equality while corrupting signed zero in the LAST
    # logical block, proving validation is exhaustive and byte-exact.
    values[0,1,0,0,0]=0.
    assert values[0,1,0,0,0]==0
    with pytest.raises(ValueError,match='cache bytes'):
        bench.validate_every_representative(values,indices,keys,metadata,offsets,blocks)


def test_a_single_changed_offset_fails_before_key_validation(bench):
    keys,metadata,offsets=operands()
    values=values_for(keys,metadata,offsets,[0,1])
    indices=torch.from_numpy(offsets.copy());indices[-1,-1,-1,-1]=6
    with pytest.raises(ValueError,match='offset differs'):
        bench.validate_every_representative(values,indices,keys,metadata,offsets,[0,1])


def test_declared_workers_and_interleaved_confirmation_are_bounded(bench):
    assert bench.initial_arms([4,8,16])==[('torch_1',None),('native_4',4),('native_8',8),('native_16',16)]
    assert bench.confirmation_order('native_8',2)==[(0,'native_8'),(0,'torch_1'),(1,'torch_1'),(1,'native_8')]
    for workers in ([],[0],[17],[4,4],[True]):
        with pytest.raises(ValueError):bench.initial_arms(workers)


def test_summary_includes_all_trials_and_does_not_claim_model_throughput(bench):
    rows=[]
    for name,seconds in [('torch_1',10),('native_8',2),('native_8',4),('torch_1',14)]:
        rows.append(dict(arm=name,seconds=seconds,tokens=120,average_cpu_cores=3,
            validation=dict(checked_offsets=48)))
    result=bench.summarize(rows)
    assert result['arms']['native_8']['trials']==2
    assert result['arms']['native_8']['speedup_vs_torch1']==4
    assert result['arms']['native_8']['tokens_per_second']==40
    assert result['model_or_gpu_started'] is False
    assert 'no model throughput' in result['limitation']
