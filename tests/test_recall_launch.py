from copy import deepcopy

import pytest

from vllm_kvmem.serve import build_launch


def profile():
    return dict(model='model', tokenizer='tokenizer', hf_config_path='model',
        model_format='safetensors', draft_method='mtp', draft_tokens=4,
        max_num_seqs=1, prefill_chunk_tokens=512, query_replay_budget=512,
        recall_experiment=dict(method='mean_k', pressure_query_window=128, device='cuda',
                               query_batch=4, block_batch=16),
        kvmem=dict(block_tokens=32, select_budget=36864, gen_budget=16384,
            sink_blocks=41, recent_blocks=0, request_planning=True, async_refresh=True,
            graph_mode='off', graph_scoring_version='v2', graph_alpha=0., guided_query_tokens=0,
            query_history=False, query_score_mode='mean_k',
            query_fps_halo_blocks=0, query_mean_quota_fraction=0., query_fps_prebuild=False))


@pytest.mark.parametrize('method', ['mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass'])
def test_recall_profile_freezes_worker_environment_through_relaunch(tmp_path, method):
    config = profile()
    config['recall_experiment']['method'] = method
    plan = build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert plan['environment']['KVMEM_RECALL_METHOD'] == method
    assert plan['environment']['KVMEM_PRESSURE_QUERY_WINDOW'] == '128'
    assert plan['environment']['KVMEM_RECALL_DEVICE'] == 'cuda'
    assert plan['environment']['KVMEM_RECALL_QUERY_BATCH'] == '4'
    assert plan['environment']['KVMEM_RECALL_BLOCK_BATCH'] == '16'
    repeated = build_launch(plan['profile'], tmp_path, tmp_path/'run2', check_paths=False)
    for key, value in plan['environment'].items():
        if key.startswith('KVMEM_RECALL_') or key == 'KVMEM_PRESSURE_QUERY_WINDOW':
            assert repeated['environment'][key] == value
    config['recall_experiment']['method'] = 'changed-after-build'
    assert plan['profile']['recall_experiment']['method'] == method


@pytest.mark.parametrize('change', [dict(method='unknown'), dict(pressure_query_window=32),
    dict(pressure_query_window=True), dict(query_batch=0), dict(query_batch=17),
    dict(block_batch=0), dict(device='automatic'), dict(extra='silent')])
def test_recall_profile_rejects_unfrozen_settings(tmp_path, change):
    config = profile()
    config['recall_experiment'].update(change)
    with pytest.raises(ValueError):
        build_launch(config, tmp_path, tmp_path/'run', check_paths=False)


def test_recall_profile_is_opt_in(tmp_path, monkeypatch):
    config = profile()
    del config['recall_experiment']
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'fixed4')
    plan = build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert 'KVMEM_RECALL_METHOD' not in plan['environment']


def test_recall_profile_does_not_mutate_input(tmp_path):
    config = profile()
    original = deepcopy(config)
    build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert config == original
