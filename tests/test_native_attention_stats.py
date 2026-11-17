"""Native stats contracts that do not require a CUDA device."""
from types import SimpleNamespace

import pytest

from vllm_kvmem.serving.native_attention import piece_layout, _layout_array
from vllm_kvmem.serving.native_attention_build import UnsupportedNativeStats, patch_prefill, patch_xqa
from vllm_kvmem.core.options import RuntimeOptions
from test_options import config


def test_native_scores_are_off_and_require_action_when_requested():
    assert RuntimeOptions.from_vllm(config({})).action_native_scores == 'off'
    for mode in ('verify', 'reuse'):
        with pytest.raises(ValueError, match='require action attention'):
            RuntimeOptions.from_vllm(config({'kvmem': {'action_native_scores': mode}}))


@pytest.mark.parametrize('setting,value', [('action_native_scores', 'fast'),
    ('action_native_max_rows', 0), ('action_native_max_rows', True), ('action_native_max_bytes', -1)])
def test_native_score_configuration_rejects_invalid_bounds(setting, value):
    with pytest.raises(ValueError, match='Native Action'):
        RuntimeOptions.from_vllm(config({'kvmem': {setting: value}}))


def test_pieces_partition_partial_logical_segments_without_physical_page_assumptions():
    starts, counts = [0, 4, 9, 11], [4, 5, 2, 13]
    layout, pieces = piece_layout(starts, counts, 24)
    assert layout == [(0, 4, 0, 1), (4, 5, 1, 1), (9, 2, 2, 1), (11, 13, 3, 2)]
    assert pieces == 5
    occupied = {}
    for segment, (start, count, first, _) in enumerate(layout):
        for token in range(start, start+count):
            piece = first + token//16 - start//16
            previous = occupied.setdefault(piece, (segment, token//16))
            assert previous == (segment, token//16)
    assert len(occupied) == pieces
    array, total = _layout_array(starts, counts, 24)
    assert [tuple(row) for row in array.tolist()] == layout and total == pieces


@pytest.mark.parametrize('starts,counts,length', [([1], [4], 5), ([0, 4], [4, 0], 4),
    ([0], [3], 4), ([0], [3, 4], 3)])
def test_pieces_cannot_normalize_over_a_subset(starts, counts, length):
    with pytest.raises(ValueError):
        piece_layout(starts, counts, length)
    with pytest.raises(ValueError):
        _layout_array(starts, counts, length)


def test_source_overlay_fails_closed_on_unknown_kernel_structure():
    with pytest.raises(UnsupportedNativeStats):
        patch_prefill('an unrelated prefill header')
    with pytest.raises(UnsupportedNativeStats):
        patch_xqa('an unrelated kernel', 'mha.cu')


def test_native_planning_prepares_scope_once_and_keeps_unaccepted_rows():
    from vllm_kvmem.serving.action_attention import ActionHistory, ActionRuntime
    from vllm_kvmem import _native
    options = SimpleNamespace(action_scope='non-thinking', action_interval=1, action_decay=.5)
    state = ActionHistory(options, [10, 11, 12, 13, 14])
    state.begin('r', [14])
    authority = SimpleNamespace(action_history=state, block_graph=object())
    runtime = ActionRuntime.__new__(ActionRuntime)
    runtime.runner = SimpleNamespace()
    runtime.runtime = SimpleNamespace(drain_requests=lambda *a, **kw: True, ensure_authority=lambda r: authority)
    runtime.begin_response = lambda *a: None
    plan = runtime._plan_owner('r', False, [20, 21, 22, 23, 24], list(range(100, 105)), list(range(4, 9)), 5)
    assert plan['rows'] == [0, 1, 2, 3, 4]
    assert plan['sources'] == tuple(range(100, 105))
    assert state.parser.state == (False, False, 0)
    state.accept(2, plan['rows'], [0], [[1.]] * 5)
    assert state.samples == 2 and state.mass == {0: 2.}
    assert state.parser.state == (False, False, 2)


def test_native_fallback_reports_actual_reason_without_a_tensor():
    from vllm_kvmem.serving.native_attention import NativeAttentionStats
    from collections import Counter
    bank = NativeAttentionStats.__new__(NativeAttentionStats)
    bank.disabled, bank.counts = 'source_mismatch:test', Counter()
    assert bank.score('layer', 'r', False) is None
    assert bank.counts == {'fallback:decode:source_mismatch:test': 1}


def test_sm80_leaves_a_non_flashinfer_forward_on_its_own_backend():
    from vllm_kvmem.serving.native_attention import NativeAttentionStats
    bank = NativeAttentionStats.__new__(NativeAttentionStats)
    bank.fa2_only, bank.reasons = True, {}
    impl = SimpleNamespace(forward=lambda *args: 'original output')
    bank._wrap(impl, 'layer')
    assert impl.forward(None, None, None, None, None, SimpleNamespace(), None) == 'original output'
    assert bank.reasons == {('layer', 'prefill'): 'backend_not_observed',
                            ('layer', 'decode'): 'backend_not_observed'}


def test_capture_discards_invalid_native_packets_and_reuses_original_metadata():
    from vllm_kvmem.serving.action_attention import ActionRuntime
    action = ActionRuntime.__new__(ActionRuntime)
    metadata, batch, plans = object(), object(), object()
    action.metadata, action.native_plans = metadata, plans
    action.candidate_samples, action.packets, action.capture_seconds = 0, 0, 0.
    action.native = SimpleNamespace(validate=lambda:False, disarm=lambda:None, report=lambda:None)
    calls = []
    def capture(captured, actual_batch, trace):
        assert actual_batch is batch and action.metadata is metadata
        assert action.native_plans is plans
        action.metadata = None
        action.candidate_samples += 5
        action.packets += 1
        calls.append(True)
        return {'r': 'invalid' if len(calls)==1 else 'reference'}
    action._capture = capture
    assert action.capture([],batch) == {'r':'reference'}
    assert len(calls)==2 and action.candidate_samples==5 and action.packets==1
    assert action.native_plans is None


def test_decode_candidates_use_immutable_frame_and_invalidate_every_reorder():
    from vllm_kvmem.serving.action_attention import ActionRuntime
    from vllm_kvmem.serving.window_batch import WindowBatchAdapter
    adapter=WindowBatchAdapter((0,),128)
    adapter.commit('r',100,64)
    action=ActionRuntime.__new__(ActionRuntime)
    action.runner=SimpleNamespace(_kvmem_window_adapter=adapter)
    action.runtime=SimpleNamespace(config=SimpleNamespace(block_tokens=32),
        ensure_authority=lambda r:pytest.fail('candidate planning must not read mutable authority'))
    armed=[]
    action.native=SimpleNamespace(disabled=None,arm=armed.append)
    from vllm_kvmem.serving.native_attention_planning import make_decode_frame
    action.native_cache={'r':dict(generation=1,offset=36,
        **make_decode_frame([0,32,64],[32,32,5],[0,1024,96],100,64,5,128,32))}
    batch=SimpleNamespace(req_ids=['r'],is_prefilling_np=[False],num_computed_tokens_np=[100],
        seq_lens_cpu_upper_bound=[105],query_start_loc_np=[0,5])
    assert action._prepare_decode_candidates(batch)
    assert armed[0]['r']['rows']==[0,1,2,3,4]
    assert armed[0]['r']['local_begin']+armed[0]['r']['count']==69
    adapter.commit('r',100,64)  # same offset/extent, different selected view
    assert adapter.generations['r']==2
    assert not action._prepare_decode_candidates(batch)
    adapter.free('r')
    assert 'r' not in adapter.generations
    adapter.commit('r',100,64)
    assert adapter.generations['r']>2  # released/reused IDs cannot reuse an epoch


@pytest.mark.parametrize('native_commit',[False,True])
def test_logical_page_coalescing_preserves_output_prefill_and_graph_checkpoint(native_commit):
    import numpy as np
    from vllm_kvmem.serving.native_attention_planning import canonical_segments
    from vllm_kvmem.serving.action_attention import ActionHistory
    from vllm_kvmem.serving.prefill_attention import PrefillHistory
    from vllm_kvmem.serving.block_graph import BlockGraph
    starts,counts,history=[0,32,37,64],[32,5,27,3],[0,32,32,64]
    compact=canonical_segments(starts,counts,history,67,32)
    assert compact.tolist()==[[0,32,0],[32,32,32],[64,3,64]]
    values=np.array([[.25,.25,.5,0],[.125,.25,.375,.25]],dtype=np.float32)
    merged=np.stack((values[:,0],values[:,1]+values[:,2],values[:,3]),axis=1)
    results=[]
    for s,h,v in ((starts,history,values),(compact[:,0],compact[:,2].tolist(),merged)):
        options=SimpleNamespace(action_scope='non-thinking',action_interval=1,action_decay=.5,
            native_prefill_commit=native_commit,prefill_attention_half_life=8192)
        output=ActionHistory(options,[10,11,12,13,14]);output.begin('r',[14])
        output.parser.prepare([20,21]);output.accept(2,[0,1],h,v.tolist())
        prefill=PrefillHistory(options)
        assert prefill.accept(67,5,5,[0,4],s,h,62,v)
        graph=BlockGraph('coalesce',32,topk=2)
        with graph.prepare_accept(0,96,[62,66],h,v) as update:update.commit()
        snapshot=graph.snapshot();before=graph.rows
        with graph.prepare_accept(96,128,[100],[0],np.ones((1,1),dtype=np.float32)) as update:update.commit()
        graph.restore(snapshot)
        assert graph.rows==before
        results.append((output.mass,output.samples,prefill.values,before))
        snapshot.close();graph.close()
    assert results[0]==results[1]


def test_canonical_segments_bound_repeated_page_fragments_without_changing_mass_domain():
    from vllm_kvmem.serving.native_attention_planning import canonical_segments
    result=canonical_segments([0,32,64],[32,32,7],[0,0,0],71,32)
    assert result.tolist()==[[0,32,0],[32,32,0],[64,7,0]]


def test_overlay_tree_publication_survives_concurrent_tp_workers(tmp_path):
    """Both TP ranks build the same content-addressed overlay on a fresh cache."""
    import threading
    from vllm_kvmem.serving.native_attention_build import _publish_tree
    source = tmp_path / 'xqa-src'
    (source / 'nested').mkdir(parents=True)
    for i in range(64):
        (source / f'f{i}.cu').write_text(str(i) * 1000)
    (source / 'nested' / 'h.h').write_text('header')
    target = tmp_path / 'cache' / 'xqa'
    target.parent.mkdir()
    barrier, errors = threading.Barrier(8), []

    def worker():
        try:
            barrier.wait()
            _publish_tree(source, target)
        except BaseException as error:    # pragma: no cover - reported below
            errors.append(error)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert sorted(p.name for p in target.iterdir()) == sorted(p.name for p in source.iterdir())
    assert all((target / f'f{i}.cu').read_text() == str(i) * 1000 for i in range(64))
    assert (target / 'nested' / 'h.h').read_text() == 'header'
    # No private staging copy is left behind, and a second call is a no-op.
    assert [p.name for p in target.parent.iterdir()] == ['xqa']
    _publish_tree(source, target)
