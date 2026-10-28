from copy import deepcopy

import pytest

from run_hybrid_efficiency_gate import evaluate,LIMITS


def reports():
    v=dict(graph_branch_seconds=dict(mean_k=.02,key_fps_4=.02),graph_selection_seconds=.4,
        scoring_scratch_bound_bytes=32*1024**2,scoring_path='tiled_layer_cache',
        cache_rebuilt_blocks=32,ttft_seconds=100.,prefix_reused=False,
        cache_bytes=1024,cache_capacity_bytes=12*1024**3,output_tokens=100)
    case=dict(id='cold',off=deepcopy(v),on=deepcopy(v))
    cold=dict(matched_inputs_seeds_sampling=True,no_accuracy_gate=True,cases=1,per_case=[case])
    warm=deepcopy(cold);warm['per_case'][0]['id']='warm'
    for arm in ('off','on'):
        warm['per_case'][0][arm].update(prefix_reused=True,ttft_seconds=4.)
    return cold,warm


def test_efficiency_gate_uses_timings_and_resources_not_answers():
    cold,warm=reports()
    cold['per_case'][0]['off'].update(output_tokens=12,accuracy=1)
    cold['per_case'][0]['on'].update(output_tokens=1000,accuracy=0)
    result=evaluate(cold,warm)
    assert result['status']=='passed' and result['no_accuracy_gate']
    assert result['limits']==LIMITS


@pytest.mark.parametrize('fault',['graph','picker','bulk_rebuild','scratch','path','cache','cold_ttft','warm_ttft'])
def test_efficiency_gate_blocks_regressions_before_full_suite(fault):
    cold,warm=reports();v=cold['per_case'][0]['on']
    if fault=='graph':v['graph_branch_seconds']['mean_k']=2.
    elif fault=='picker':v['graph_selection_seconds']=2.
    elif fault=='bulk_rebuild':v['cache_rebuilt_blocks']=1000
    elif fault=='scratch':v['scoring_scratch_bound_bytes']=1024**3
    elif fault=='path':v['scoring_path']='repeated_uploads'
    elif fault=='cache':v['cache_bytes']=13*1024**3
    elif fault=='cold_ttft':v['ttft_seconds']=200.
    else:warm['per_case'][0]['on']['ttft_seconds']=8.
    assert evaluate(cold,warm)['status']=='blocked_performance_review'


def test_efficiency_gate_rejects_unmatched_input_or_missing_warm_prefix():
    cold,warm=reports();cold['matched_inputs_seeds_sampling']=False
    with pytest.raises(ValueError,match='input proof'):evaluate(cold,warm)
    cold,warm=reports();warm['per_case'][0]['on']['prefix_reused']=False
    with pytest.raises(ValueError,match='lifecycle'):evaluate(cold,warm)


def test_dense_cache_needs_hash_bound_geometry_proof_and_same_memory_limit():
    cold,warm=reports();v=cold['per_case'][0]['on']
    v['scoring_path']='dense_layer_cache'
    assert evaluate(cold,warm)['status']=='blocked_performance_review'
    v['scoring_allocation_proof']=dict(kind='derived_from_frozen_source_and_recorded_geometry',
        source_sha256='451e9fc3048823765471e7f91dd4e4ca45910327c8017305552493f421ef9b68')
    assert evaluate(cold,warm)['status']=='passed'
    v['scoring_scratch_bound_bytes']=1024**3
    assert evaluate(cold,warm)['status']=='blocked_performance_review'
