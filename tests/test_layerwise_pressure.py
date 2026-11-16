from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure, extend_records
from vllm_kvmem.serving.prefill_attention import PrefillHistory
from vllm_kvmem.core.block_runs import candidate_starts
from test_recall_experiment import native_authority, finish


def options(memory=None):
    return NS(memory=memory, graph_mode='off', pressure_policy='attention', guided_query_tokens=0,
        query_history=False, prefill_attention=True, prefill_attention_half_life=8192.)


def controller():
    q=NS(native_batches=False, optimized=True, names=('a','b'),
        runner=NS(model_config=NS(enforce_eager=True),device='cpu'))
    return LayerwisePrefillPressure(q)


def test_extension_keeps_each_layers_original_blocks_and_merges_partial_tail():
    a=[(1,1,0,32,0),(4,1,96,16,32)]
    b=[(1,1,0,32,0),(7,1,192,16,32)]
    assert extend_records(a,112,153)==[(1,1,0,32,0),(4,1,96,32,32),(-1,-1,128,25,64)]
    assert extend_records(b,208,249)==[(1,1,0,32,0),(7,1,192,32,32),(-1,-1,224,25,64)]
    assert a[-1][3]==16
    with pytest.raises(ValueError):extend_records(a,112,100)


def test_pressure_scores_target_layers_only_excluding_archived_mtp_draft():
    c=controller()
    action=NS(runtime=NS(names=('a','b','draft'),layers=['target0','target1','draft']))
    assert c.scoring_layers(action)==[(0,'target0'),(1,'target1')]
    action.runtime.names=('a','draft','b')
    with pytest.raises(RuntimeError,match='archive order'):c.scoring_layers(action)


def test_native_batch_pressure_keeps_each_owners_attention_history_separate():
    q=NS(native_batches=True,optimized=True,names=('a','b'),
        runner=NS(model_config=NS(enforce_eager=False),device='cpu'))
    c=LayerwisePrefillPressure(q)
    a=c._history('first',options());b=c._history('second',options())
    a[0].values={32:[1.,1.]};a[1].values={64:[1.,1.]}
    assert all(not h.values for h in b)
    assert a[0] is not a[1] and a[0] is not b[0]
    c.commit_accept(('second',[PrefillHistory(options()),PrefillHistory(options())]))
    assert c.histories['first'][1].scores()=={64:1.}


def test_memory_graph_metadata_does_not_change_cumulative_pressure_scores():
    c=controller();o=options();o.graph_mode='multi';o.graph_alpha=.2
    histories=c._history('r',o)
    histories[0].values={0:[.8,1.]};histories[1].values={32:[.9,1.]}
    assert histories[0].scores()=={0:.8} and histories[1].scores()=={32:.9}
    assert o.graph_mode=='multi' and o.graph_alpha==.2


def packet(c):
    o=options()
    # Same dense positions, different original text at the second slot.
    c.maps['r']=(256,(((0,1,0,32,0),(1,1,64,32,32)),
                     ((0,1,0,32,0),(2,1,128,32,32))))
    a=NS(memory=NS(_store=NS(working_records=lambda:None)))
    pending=NS(state=NS(options=o),prefill=(288,32,[0,32,64],64),
        rows=[31],history=[0,64,256])
    masses=torch.tensor([[[.1,.8,.1]],[[.1,.2,.7]]])
    c.attach_packet(pending,'r',a,[0,32,64],[32,32,32],masses)
    return pending


def test_accepted_scores_keep_per_layer_original_coordinates_without_cross_layer_mean():
    c=controller();p=packet(c)
    staged=c.prepare_accept(p,32)
    assert not c.histories['r'][0].values
    c.commit_accept(staged)
    assert set(c.histories['r'][0].values)=={0,64,256}
    assert set(c.histories['r'][1].values)=={0,128,256}
    assert c.histories['r'][0].scores()[64]==pytest.approx(.8)
    assert c.histories['r'][1].scores()[128]==pytest.approx(.2)
    assert [h.end for h in c.histories['r']]==[288,288]


def test_rejected_or_partial_chunk_does_not_publish_layer_scores():
    c=controller();p=packet(c)
    c.commit_accept(c.prepare_accept(p,31))
    assert all(h.end==0 and not h.values for h in c.histories['r'])


def test_bad_second_layer_keeps_all_accepted_histories_unchanged():
    c=controller();p=packet(c)
    p.layerwise_pressure[3][1,0,0]=float('nan')
    before=[h.snapshot() for h in c.histories['r']]
    with pytest.raises((RuntimeError,ValueError)):c.prepare_accept(p,32)
    assert [h.snapshot() for h in c.histories['r']]==before


def test_native_pressure_picker_uses_each_layers_scores_with_same_budget_and_sink():
    c=controller();a=native_authority(count=256,budget=96,sink=1)
    a.publish_sealed=lambda wait:None
    o=options(a.memory.config)
    hs=c._history('r',o)
    rows=list(a.records.values())
    hs[0].values={r.history_begin:[1. if i in (2,3) else .01,1.] for i,r in enumerate(rows)}
    hs[1].values={r.history_begin:[1. if i in (5,6) else .01,1.] for i,r in enumerate(rows)}
    a.action_history=NS(prefill=hs[0])
    job=NS(command=NS(request_id='r',pressure_stage='prefill',mandatory_spans=(),
        groups=((0,NS(window_tokens=96)),)),history=256,authority=a)
    try:
        c.prepare(NS(options=o),job)
        chosen,evidence=job.layerwise_pressure_plan
        assert [{r.history_begin for r in group} for group in chosen]==[{0,64,96},{0,160,192}]
        assert all(e['charged_tokens']==e['selected_tokens']==96 for e in evidence)
        assert all(e['fallback_reason'] is None for e in evidence)
    finally:assert a.memory.close()


def test_pressure_candidates_never_reopen_other_layers_cold_high_score():
    c=controller();a=native_authority(count=256,budget=96,sink=1)
    a.publish_sealed=lambda wait:None
    rows=list(a.records.values());o=options(a.memory.config)
    hs=c._history('r',o)
    a.memory.select({},pressure_scores=[(r.handle.slot,1. if i in (1,2) else 0.) for i,r in enumerate(rows)],max_tokens=96)
    finish(a.memory)
    def mapping(indices):return tuple((rows[i].handle.slot,rows[i].handle.generation,rows[i].history_begin,32,j*32) for j,i in enumerate(indices))
    c.maps['r']=(256,(mapping((0,1,2)),mapping((0,5,6))))
    hs[0].values={0:[.1,1.],32:[.5,1.],64:[.4,1.]}
    hs[1].values={0:[.1,1.],32:[999.,1.],160:[.5,1.],192:[.4,1.]}
    a.action_history=NS(prefill=hs[0])
    job=NS(command=NS(request_id='r',pressure_stage='prefill',mandatory_spans=(),
        groups=((0,NS(window_tokens=96)),)),history=256,authority=a)
    try:
        c.prepare(NS(options=o),job)
        assert {r.history_begin for r in job.layerwise_pressure_plan[0][1]}=={0,160,192}
    finally:assert a.memory.close()


def test_missing_layer_feedback_fails_without_a_silent_recency_fallback():
    c=controller();a=native_authority(count=256,budget=96)
    a.publish_sealed=lambda wait:None;o=options(a.memory.config)
    hs=c._history('r',o);hs[0].values={0:[1.,1.]};a.action_history=NS(prefill=hs[0])
    job=NS(command=NS(request_id='r',pressure_stage='prefill',mandatory_spans=(),
        groups=((0,NS(window_tokens=96)),)),history=256,authority=a)
    try:
        with pytest.raises(RuntimeError,match='no accepted attention'):c.prepare(NS(options=o),job)
    finally:assert a.memory.close()


@pytest.mark.parametrize('stage', ['prefill', 'decode'])
@pytest.mark.parametrize('seed', [0, 1, 7, 20260722])
def test_slot_cache_preserves_native_preview_inputs_and_reads_each_archive_slot_once(stage, seed):
    import random
    from collections import Counter
    c=controller();c.names=tuple(f'layer{i}' for i in range(16));c.rolling_decode=True
    a=native_authority(count=256,budget=96,sink=1)
    a.publish_sealed=lambda wait:None
    o=options(a.memory.config);hs=c._history('r',o)
    rows=list(a.records.values());rng=random.Random(seed)
    # Include exact score ties; no ranking, candidate or mandatory policy changes.
    for history in hs:
        history.values={r.history_begin:[rng.choice([.01,.2,.2,1.]),1.] for r in rows}
    a.action_history=NS(prefill=hs[0]);c.decode_histories['r']=hs
    expected=[a.memory.preview_selection({},(rows[0].handle.slot,),action=False,
        pressure_scores=[(r.handle.slot,h.scores()[r.history_begin]) for r in rows],max_tokens=96)
        for h in hs]
    calls=Counter()
    class Handle:
        def __init__(self, slot):self._slot=slot
        @property
        def slot(self):
            calls[self._slot]+=1
            return self._slot
    records=a.records
    a.records={key:NS(history_begin=r.history_begin,count=r.count,handle=Handle(r.handle.slot))
               for key,r in records.items()}
    job=NS(command=NS(request_id='r',pressure_stage=stage,mandatory_spans=(),
        groups=((0,NS(window_tokens=96)),)),history=256,authority=a)
    try:
        c.prepare(NS(options=o),job)
        chosen,evidence=job.layerwise_pressure_plan
        actual=[[r.handle._slot for r in group] for group in chosen]
        assert [set(group) for group in actual]==[set(group) for group in expected]
        order={r.handle.slot:i for i,r in enumerate(sorted(rows,key=lambda r:r.history_begin))}
        assert actual==[sorted(group,key=order.__getitem__) for group in expected]
        expected_calls=Counter({r.handle.slot:1 for r in rows})
        if stage=='decode':expected_calls.update(actual[0])
        assert calls==expected_calls
        assert all(candidate_starts(e)==[r.history_begin for r in rows] and 'candidate_starts' not in e for e in evidence)
        assert all(e['fallback_reason'] is None and e['charged_tokens']==96 for e in evidence)
    finally:
        a.records=records
        assert a.memory.close()
