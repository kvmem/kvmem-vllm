from copy import copy
import struct
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem import _prefill_state_native as native
from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory, accept_buffered_batch
from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure, _LayerwiseAcceptance


@pytest.fixture
def batch_native():
    if not hasattr(native,'committed_batch'):
        pytest.skip('Build the isolated native batch acceptance extension')
    return native


def options(layer=0):
    return NS(native_prefill_commit=True,prefill_attention_half_life=8192.+layer,
        pressure_policy='attention',guided_query_tokens=0,query_history=False,prefill_attention=True)


def controller(monkeypatch,*,batch=True,layers=16):
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY','1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT','1' if batch else '0')
    query=NS(native_batches=True,optimized=True,names=tuple(range(layers)),
        runner=NS(model_config=NS(enforce_eager=True)))
    return LayerwisePrefillPressure(query)


def bits(values):
    return [(key,struct.pack('<dd',*pair)) for key,pair in values.items()]


def history_state(history):
    return (bits(history.values),history.end,history.samples,history.chunks,
        [(key,struct.pack('<d',score)) for key,score in history.scores().items()],
        history.successful_buffered_commits,history.successful_buffered_restores,
        getattr(history,'last_commit_backend',None))


def layout_view(array,layout):
    if layout=='strided':
        backing=np.zeros(tuple(2*size+3 for size in array.shape),dtype=array.dtype)
        view=backing[1:1+2*array.shape[0]:2,1:1+2*array.shape[1]:2,1:1+2*array.shape[2]:2]
        view[...] = array
        return view
    if layout=='reversed':
        return array[::-1,::-1,::-1]
    return array


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','strided','reversed'])
def test_sixteen_layer_batch_matches_scalar_every_double(batch_native,dtype,layout):
    rng=np.random.default_rng(931)
    scalar=[BufferedPrefillHistory(options(layer)) for layer in range(16)]
    batch=[BufferedPrefillHistory(options(layer)) for layer in range(16)]
    for layer,(left,right) in enumerate(zip(scalar,batch)):
        values={9000:[-0.,1.],9001:[1.e-320,1.e-310],layer:[.123456789,.875]}
        left.values=values;right.values=values
        left.end=right.end=layer
    for step in range(1,17):
        logical=[rng.integers(0,90,size=67).tolist() for _ in range(16)]
        if step==1:
            for keys in logical:keys[:2]=[9000,9001]
        starts=list(range(0,67*32,32))
        if step%3==0:starts=starts[:53]
        masses=rng.random((16,6,67))
        masses/=masses.sum(axis=2,keepdims=True)
        masses=layout_view(masses.astype(dtype),layout)
        rows=list(range(31,192,32))
        expected=[h.accept(step*512,512,512,rows,starts,logical[layer],1792,masses[layer])
                  for layer,h in enumerate(scalar)]
        actual=accept_buffered_batch(batch,step*512,512,512,rows,starts,logical,1792,masses)
        assert actual==expected==[True]*16
        assert [history_state(h) for h in batch]==[history_state(h) for h in scalar]
        assert [h.successful_buffered_batch_commits for h in batch]==[step]*16
        assert all(h.successful_buffered_batch_commits==0 for h in scalar)


@pytest.mark.parametrize('value',[np.nextafter(.995,-np.inf),.995,np.nextafter(.995,np.inf),
    np.nextafter(1.005,-np.inf),1.005,np.nextafter(1.005,np.inf)])
def test_final_layer_validation_boundary_matches_scalar(batch_native,value):
    states=[native.State.from_values({10:[.5,1.]}) for _ in range(16)]
    masses=np.ones((16,1,1));masses[-1,0,0]=value
    saved=[bits(state.values()) for state in states]
    outcomes=[]
    for batched in (False,True):
        try:
            if batched:
                updated=native.committed_batch(states,[.75]*16,[31],[0],[[10]]*16,0,masses)
            else:
                updated=[state.committed(.75,[31],[0],[10],0,masses[layer])
                         for layer,state in enumerate(states)]
        except (ValueError,RuntimeError) as error:
            outcomes.append((type(error),str(error)))
        else:
            outcomes.append([bits(state.values()) for state in updated])
    assert outcomes[0]==outcomes[1]
    assert [bits(state.values()) for state in states]==saved


@pytest.mark.parametrize('fault',['nan','negative','sum','rows','history','dtype','overflow'])
def test_failed_batch_keeps_all_prior_states_and_counters(batch_native,fault):
    histories=[BufferedPrefillHistory(options(layer)) for layer in range(16)]
    logical=[[0,32] for _ in histories]
    rows=[31,63];local=0
    masses=np.tile(np.array([[[.3,.7],[.6,.4]]]),(16,1,1))
    accept_buffered_batch(histories,512,512,512,rows,[0,32],logical,local,masses)
    before=[history_state(h) for h in histories]
    staged=[copy(h) for h in histories]
    if fault=='nan':masses[-1,0,0]=np.nan
    elif fault=='negative':masses[-1,0,0]=-.3
    elif fault=='sum':masses[-1,0,0]=1.
    elif fault=='rows':rows=[31]
    elif fault=='history':logical[-1]=[0]
    elif fault=='dtype':masses=masses.astype(np.float16)
    else:local=2**63-1
    with pytest.raises((ValueError,RuntimeError)):
        accept_buffered_batch(staged,1024,512,512,rows,[0,32],logical,local,masses)
    assert [history_state(h) for h in histories]==before
    assert [bits(h.values) for h in staged]==[bits(h.values) for h in histories]
    assert all(h.successful_buffered_batch_commits==1 and h.successful_buffered_commits==1
               for h in staged)


def test_validation_order_keeps_earlier_bad_mass_before_later_bad_history(batch_native):
    states=[native.State.from_values({}) for _ in range(2)]
    masses=np.array([[[np.nan,0.]],[[.5,.5]]])
    with pytest.raises(RuntimeError,match='Invalid prefill attention mass'):
        native.committed_batch(states,[1.,1.],[31],[0,32],[[0,32],[0]],0,masses)


def test_controller_batches_once_without_per_layer_mass_views_and_publishes_on_commit(
        batch_native,monkeypatch):
    import torch
    c=controller(monkeypatch)
    histories=c._history('owner',options())
    backing=torch.full((16,3,8),.25,dtype=torch.float32)
    masses=backing[:,1:2,::2]
    pending=NS(prefill=(512,512,[0,32,64,96],0),rows=[511],
        layerwise_pressure=(c,'owner',[[layer*1024+i*32 for i in range(4)] for layer in range(16)],masses))
    call=native.committed_batch
    seen=[]
    def observed(*args):
        array=args[6]    # (states, decays, rows, starts, histories, local_begin, masses, history_shift)
        assert args[7]==512-512-0    # end - count - local_begin
        assert array.__array_interface__['data'][0]==masses.data_ptr()
        assert array.strides==tuple(stride*masses.element_size() for stride in masses.stride())
        seen.append(array.shape)
        return call(*args)
    monkeypatch.setattr(native,'committed_batch',observed)
    monkeypatch.setattr(BufferedPrefillHistory,'accept',lambda *args:pytest.fail('Unexpected scalar acceptance'))
    staged=c.prepare_accept(pending,512)
    assert seen==[(16,1,4)]
    assert all(h.chunks==0 for h in histories)
    assert all(c.history_execution(h)['buffered_batch_commits']==1 for h in staged[1])
    transactions=[]
    transaction=NS(commit=lambda:transactions.append('commit'),abort=lambda:transactions.append('abort'))
    abandoned=_LayerwiseAcceptance(transaction,c,staged)
    abandoned.abort()
    assert transactions==['abort'] and c.histories['owner'] is histories
    staged=c.prepare_accept(pending,512)
    accepted=_LayerwiseAcceptance(transaction,c,staged)
    accepted.commit()
    assert transactions==['abort','commit']
    assert all(h.chunks==1 for h in c.histories['owner'])
    assert all(c.history_execution(h)['buffered_batch_commits']==1 for h in c.histories['owner'])


@pytest.mark.parametrize('skip',['partial','empty','stale','mixed'])
def test_skipped_and_mixed_boundaries_preserve_scalar_behavior(batch_native,monkeypatch,skip):
    c=controller(monkeypatch,layers=2)
    histories=c._history('owner',options())
    accepted=511 if skip=='partial' else 512
    rows=[] if skip=='empty' else [511]
    if skip in ('stale','mixed'):
        histories[0].end=512
        if skip=='stale':histories[1].end=512
    masses=np.ones((2,1,1))
    pending=NS(prefill=(512,512,[0],0),rows=rows,
        layerwise_pressure=(c,'owner',[[0],[32]],masses))
    monkeypatch.setattr(native,'committed_batch',lambda *args:pytest.fail('Skipped boundary must remain scalar'))
    before=[history_state(h) for h in histories]
    if skip=='mixed':
        with pytest.raises(RuntimeError,match='disagree on accepted sampling boundary'):
            c.prepare_accept(pending,accepted)
    else:
        staged=c.prepare_accept(pending,accepted)
        assert all(h.successful_buffered_batch_commits==0 and h.chunks==0 for h in staged[1])
    assert [history_state(h) for h in histories]==before


def test_batch_option_is_off_by_default_and_telemetry_stays_unchanged(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT',raising=False)
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY','1')
    query=NS(native_batches=True,optimized=True,names=('a',),runner=NS(model_config=NS(enforce_eager=True)))
    c=LayerwisePrefillPressure(query)
    history=c._history('owner',options())[0]
    history.accept(512,512,512,[511],[0],[0],0,np.ones((1,1)))
    assert not c.batch_accept
    assert c.history_execution(history)==dict(attention_state_backend='native_buffered',
        buffered_commits=1,buffered_restores=0)


def test_enabled_batch_requires_native_capability(monkeypatch):
    monkeypatch.delattr(native,'committed_batch',raising=False)
    with pytest.raises(ValueError,match='rebuilt prefill state extension'):
        controller(monkeypatch)


@pytest.mark.parametrize('flag,buffered,match',[
    ('yes','1','must be 0 or 1'),('1','0','native buffered histories')])
def test_batch_option_rejects_invalid_configuration(monkeypatch,flag,buffered,match):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT',flag)
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY',buffered)
    query=NS(native_batches=True,optimized=True,names=('a',),runner=NS(model_config=NS(enforce_eager=True)))
    with pytest.raises(ValueError,match=match):LayerwisePrefillPressure(query)


def test_native_batch_rejects_mismatched_layer_vectors_and_null_state(batch_native):
    state=native.State.from_values({})
    for states,decays,logical in [([state],[],[[0]]),([state],[1.],[]),([None],[1.],[[0]])]:
        with pytest.raises((ValueError,TypeError)):
            native.committed_batch(states,decays,[31],[0],logical,0,np.ones((1,1,1)))


def test_native_batch_owns_fresh_states_yielded_by_generator(batch_native):
    # Only the native call retains these states after each generator yield.
    # Repeated allocations expose stale-pointer reuse if the converter drops
    # its temporary owner tuple before the GIL-free numerical body starts.
    for repeat in range(32):
        states=(native.State.from_values({layer:[repeat+layer+.5,2.]}) for layer in range(16))
        updated=native.committed_batch(states,[.5]*16,[31],[0],
            [[layer] for layer in range(16)],0,np.ones((16,1,1)))
        assert [state.values() for state in updated]==[
            {layer:[.5*(repeat+layer+.5)+1.,2.]} for layer in range(16)]


def test_native_batch_retains_states_when_another_thread_clears_input_list(batch_native):
    for _ in range(8):
        original=[native.State.from_values({layer:[.5,2.]}) for layer in range(16)]
        ready=threading.Event()
        def state_objects():
            yield from original
            ready.set()
        def clear():
            assert ready.wait(5.)
            original.clear()
        worker=threading.Thread(target=clear)
        worker.start()
        try:
            updated=native.committed_batch(state_objects(),[.5]*16,list(range(32)),[0],
                [[layer] for layer in range(16)],0,np.ones((16,32,1)))
        finally:
            worker.join(5.)
        assert not worker.is_alive() and not original
        assert [state.values() for state in updated]==[{layer:[1.25,2.]} for layer in range(16)]


def test_restore_and_later_batch_preserve_snapshot_and_counter(batch_native):
    original=BufferedPrefillHistory(options())
    original.accept(512,512,512,[511],[0],[7],0,np.ones((1,1)))
    restored=BufferedPrefillHistory(options())
    restored.restore(original.snapshot(),512)
    assert restored.snapshot()==original.snapshot()
    assert restored.successful_buffered_batch_commits==0
    accept_buffered_batch([restored],1024,512,512,[511],[0],[[7]],0,np.ones((1,1,1)))
    original.accept(1024,512,512,[511],[0],[7],0,np.ones((1,1)))
    assert bits(restored.values)==bits(original.values)
    assert restored.successful_buffered_batch_commits==1
