"""Asynchronous FPS preparation is explicit and survives both public launchers."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from test_graph_options import model_profile, runtime, settings, worker_payload
from vllm_kvmem.elastic_serve import build_launch as build_elastic
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def fps_settings(**updates):
    values = settings(query_score_mode='key_fps_4', query_fps_optimized=True,
        graph_scoring_version='v2', graph_aggregation='sum', graph_alpha=0.,
        cpu_read_cache_bytes=4*1024**3, query_window_cache_bytes=1048576)
    values.update(updates)
    return values


def elastic_profile(tmp_path, local=None):
    local = model_profile(tmp_path) if local is None else local
    users = tmp_path/'users.json'
    users.write_text(json.dumps({'users': [{'id': 'fps-prebuild', 'api_key': 'local-test-only'}]}))
    return dict(model=local['model'], tokenizer=local['tokenizer'], users_file=str(users),
        context_mode='meank', draft_method='mtp', draft_tokens=4, block_tokens=32,
        cpu_kv_bytes=4*1024**3, active_context_tokens=8192, gen_budget_tokens=4096,
        max_output_tokens=512, request_planning=True, action_attention=True,
        action_fraction=0., action_scope='non-thinking', prefill_attention=True,
        prefill_chunk_tokens=1024, graph_mode='multi', graph_scoring_version='v2',
        graph_aggregation='sum', graph_alpha=0., query_score_mode='key_fps_4',
        query_window_cache_bytes=1048576, query_fps_optimized=True)


def test_prebuild_defaults_off_and_toggle_changes_only_execution_option():
    assert RuntimeOptions().query_fps_prebuild is False
    assert RuntimeOptions().query_fps_prebuild_device == 'cpu'
    assert runtime().query_fps_prebuild is False
    assert runtime(fps_settings()).query_fps_prebuild is False
    assert runtime(fps_settings(query_fps_optimized=False)).query_fps_prebuild is False
    off = runtime(fps_settings())
    on = replace(off, query_fps_prebuild=True)
    before, after = asdict(off), asdict(on)
    assert before.pop('query_fps_prebuild') is False
    assert after.pop('query_fps_prebuild') is True
    assert before == after


@pytest.mark.parametrize('halo', [0, 1])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_direct_profile_and_serve_cli_roundtrip_without_policy_or_budget_changes(tmp_path, capsys, halo, enabled, device):
    from vllm_kvmem import serve
    profile = model_profile(tmp_path)
    profile['kvmem'] = fps_settings(query_fps_prebuild=enabled, query_fps_halo_blocks=halo,
        query_fps_prebuild_device=device)
    original = deepcopy(profile)
    plan = build_launch(profile, tmp_path, tmp_path/'run')
    worker = worker_payload(plan)
    assert worker['query_fps_prebuild'] is enabled
    assert worker['query_fps_halo_blocks'] == halo
    assert worker['query_fps_prebuild_device'] == device
    assert runtime(worker).query_fps_prebuild is enabled
    assert worker_payload(build_launch(plan['profile'], tmp_path, tmp_path/'replay')) == worker
    config = tmp_path/'profile.json'; config.write_text(json.dumps(profile))
    serve.main(['--config', str(config), '--run-dir', str(tmp_path/'cli'), '--dry-run'])
    cli_plan = json.loads(capsys.readouterr().out)
    assert worker_payload(cli_plan)['query_fps_prebuild'] is enabled
    assert worker_payload(cli_plan)['query_fps_prebuild_device'] == device
    assert profile == original
    assert not (tmp_path/'run').exists() and not (tmp_path/'cli').exists()


@pytest.mark.parametrize('updates,match', [
    ({'query_fps_prebuild': 1}, 'query_fps_prebuild must be boolean'),
    ({'query_fps_prebuild': 'true'}, 'query_fps_prebuild must be boolean'),
    ({'query_fps_prebuild': None}, 'query_fps_prebuild must be boolean'),
    ({'query_fps_prebuild_device': 'cuda:0'}, 'query_fps_prebuild_device must be cpu or cuda'),
    ({'query_fps_prebuild_device': None}, 'query_fps_prebuild_device must be cpu or cuda'),
    ({'query_fps_prebuild_device': True}, 'query_fps_prebuild_device must be cpu or cuda'),
    ({'query_fps_prebuild': True, 'query_fps_optimized': False}, 'query_fps_optimized=true'),
    ({'query_fps_prebuild': True, 'query_fps_optimized': False, 'query_score_mode': 'mean_k'}, 'query_score_mode=key_fps_4'),
    ({'query_fps_prebuild': True, 'query_fps_optimized': False, 'query_score_mode': 'overlap_16_8'}, 'query_score_mode=key_fps_4'),
])
def test_invalid_prebuild_settings_fail_before_worker_launch(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        runtime(fps_settings(**updates))
    profile = model_profile(tmp_path); profile['kvmem'] = fps_settings(**updates)
    with pytest.raises(ValueError, match=match):
        build_launch(profile, tmp_path, tmp_path/'bad')
    elastic = elastic_profile(tmp_path, profile); elastic.update(updates)
    with pytest.raises(ValueError, match=match):
        build_elastic(elastic, tmp_path, tmp_path/'bad-elastic')
    assert not (tmp_path/'bad').exists() and not (tmp_path/'bad-elastic').exists()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_elastic_default_and_explicit_prebuild_survive_backend_profile_replay(tmp_path, device):
    profile = elastic_profile(tmp_path)
    disabled, _ = build_elastic(profile, tmp_path, tmp_path/'disabled')
    assert disabled['query_fps_prebuild'] is False
    assert disabled['mean_k_backend_profile']['kvmem']['query_fps_prebuild'] is False
    assert disabled['mean_k_backend_profile']['kvmem']['query_fps_prebuild_device'] == 'cpu'
    profile.update(query_fps_prebuild=True, query_fps_halo_blocks=1, query_fps_prebuild_device=device)
    original = deepcopy(profile)
    resolved, _ = build_elastic(profile, tmp_path, tmp_path/'enabled')
    backend = resolved['mean_k_backend_profile']
    worker = worker_payload(build_launch(backend, tmp_path, tmp_path/'native'))
    assert resolved['query_fps_prebuild'] is backend['kvmem']['query_fps_prebuild'] is worker['query_fps_prebuild'] is True
    assert resolved['query_fps_prebuild_device'] == backend['kvmem']['query_fps_prebuild_device'] == worker['query_fps_prebuild_device'] == device
    replay, _ = build_elastic(resolved, tmp_path, tmp_path/'replay')
    assert replay['mean_k_backend_profile']['kvmem'] == backend['kvmem']
    assert runtime(worker).query_fps_prebuild is True
    assert profile == original
