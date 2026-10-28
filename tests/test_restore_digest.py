import hashlib
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import restore_digest as d


def test_streaming_fp8_reference_matches_original_full_byte_oracle(monkeypatch):
    from vllm_kvmem.window_assembly import NativeWindowAssembler
    from vllm_kvmem.session_worker import SessionWorker
    torch.manual_seed(19)
    block,heads,dim,layers_count=32,2,8,2
    counts=[32]*34+[7];tokens=sum(counts)
    records=[];payloads={};position=0
    for i,count in enumerate(counts):
        records.append((i,1,i*64,count,position));position+=count
        payloads[i]=(torch.randn(layers_count,count,heads,dim).to(torch.float8_e4m3fn),
                     torch.randint(0,255,(layers_count,count,heads,dim),dtype=torch.uint8))
    trig=torch.randn(8192,dim)
    layers=[NS(cos_sin=trig,key_scale=torch.tensor(.1)) for _ in range(layers_count)]
    monkeypatch.setattr(NativeWindowAssembler,'from_runner',lambda *a,**k:NS(layers=layers,close=lambda:None))
    worker=NS(runner=NS(device='cpu'),runtime=NS(names=['a','b'],heads=heads,dim=dim,_wait_archive_progress=None),
        options=NS(memory=NS(block_tokens=block)))
    authority=NS(publish_sealed=lambda *a:None,memory=NS(_store=NS(working_records=lambda:records)),
        records={(i,1):NS(handle=i) for i in range(len(records))},read_records=lambda handles:[payloads[i] for i in handles])
    expected=SessionWorker._rebuilt_target_bytes(worker,'r',authority)
    size=tokens*heads*dim
    expected_hashes=[hashlib.sha256(expected[i:i+size].numpy().tobytes()).hexdigest() for i in range(0,len(expected),size)]
    assert d.rebuilt_digest(worker,'r',authority)['sha256']==expected_hashes


def test_target_stream_covers_every_byte_and_partial_final_page(monkeypatch):
    from vllm_kvmem.window_assembly import NativeWindowAssembler
    tokens=2051;block=32;heads=2;width=4
    layers=[];expected=[]
    for i in range(2):
        key=(torch.arange(80*block*heads*width)%251).byte().reshape(80,block,heads,width)
        value=(key+i+1).clone();table=torch.arange(80-1,-1,-1)
        layers.append(NS(key=key,value=value,kernel_page_table=table))
        positions=torch.arange(tokens);pages=table[positions//block]
        for array in [key,value]:
            expected.append(hashlib.sha256(array[pages,positions%block].numpy().tobytes()).hexdigest())
    closed=[]
    monkeypatch.setattr(NativeWindowAssembler,'from_runner',lambda *a,**k:NS(layers=layers,close=lambda:closed.append(True)))
    worker=NS(runner=NS(device='cpu'),runtime=NS(names=['a','b'],heads=heads,dim=width),
              options=NS(memory=NS(block_tokens=block)))
    result=d.target_digest(worker,'r',tokens)
    assert result['sha256']==expected and result['bytes']==tokens*heads*width*2*2 and closed==[True]
    layers[-1].value[table[(tokens-1)//block],(tokens-1)%block,0,0]^=1
    assert d.target_digest(worker,'r',tokens)['sha256'][-1]!=expected[-1]


@pytest.mark.parametrize('bad',['none','kv','recurrent','missing'])
def test_digest_gate_requires_all_kv_hashes_and_exact_model_bytes(tmp_path,monkeypatch,bad):
    oracle=dict(tokens=2,names=['a'],bytes=16,sha256=['k','v'])
    actual={**oracle,'sha256':['k','wrong']} if bad=='kv' else oracle
    monkeypatch.setattr(d,'target_digest',lambda *args:actual)
    state=torch.arange(4);restored=state+1 if bad=='recurrent' else state
    worker=NS(bindings={'r':NS(namespace='s',window=2,pressure_restore=False,history=99)},
        restore_oracles={} if bad=='missing' else {'s':(99,oracle)},
        runner=NS(_kvmem_probe_output=tmp_path,_kvmem_mtp_modules={}),runtime=NS(names=['a']),
        model_adapter=NS(_recurrent_views=lambda *a:{'x':restored},_draft_views=lambda *a:{}))
    cp=NS(position=99,window=NS(window_tokens=2),authority=NS(raw_dtype=torch.float8_e4m3fn),
          model=NS(draft_begin=0,recurrent={'x':state},draft={}))
    if bad!='none':
        with pytest.raises(RuntimeError):d.verify(worker,'r',cp,True)
        assert not (tmp_path/'verified_restores_digest.jsonl').exists()
    else:
        d.verify(worker,'r',cp,True)
        assert 'sha256_equal' in (tmp_path/'verified_restores_digest.jsonl').read_text()
