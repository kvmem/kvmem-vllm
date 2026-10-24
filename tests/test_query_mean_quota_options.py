"""MeanK quota is explicit, default-off and qualified for FPS4/halo1 only."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from test_graph_options import model_profile, runtime, worker_payload
from test_query_fps_prebuild_options import fps_settings, elastic_profile
from vllm_kvmem.elastic_serve import build_launch as build_elastic
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def quota_settings(**updates):
    values = fps_settings(query_fps_halo_blocks=1, query_mean_quota_fraction=.5)
    values.update(updates)
    return values


def test_default_preserves_existing_modes_and_only_explicit_quota_changes_setting():
    assert RuntimeOptions().query_mean_quota_fraction == 0.
    assert runtime().query_mean_quota_fraction == 0.
    assert runtime(fps_settings()).query_mean_quota_fraction == 0.
    old = runtime(fps_settings(query_fps_halo_blocks=1))
    new = replace(old, query_mean_quota_fraction=.5)
    before, after = asdict(old), asdict(new)
    assert before.pop('query_mean_quota_fraction') == 0.
    assert after.pop('query_mean_quota_fraction') == .5
    assert before == after
    assert runtime(fps_settings(query_score_mode='overlap_16_8',
        query_fps_optimized=False, query_mean_quota_fraction=0.)).query_mean_quota_fraction == 0.


@pytest.mark.parametrize('fraction', [0, .25, .5, 1.])
def test_quota_profile_survives_direct_cli_worker_and_elastic_replay(tmp_path, capsys, fraction):
    from vllm_kvmem import serve
    profile = model_profile(tmp_path)
    profile['kvmem'] = quota_settings(query_mean_quota_fraction=fraction)
    original = deepcopy(profile)
    plan = build_launch(profile, tmp_path, tmp_path/'direct')
    worker = worker_payload(plan)
    assert worker['query_mean_quota_fraction'] == fraction
    assert runtime(worker).query_mean_quota_fraction == fraction
    assert worker_payload(build_launch(plan['profile'], tmp_path, tmp_path/'direct-replay')) == worker
    config = tmp_path/'profile.json'; config.write_text(json.dumps(profile))
    serve.main(['--config', str(config), '--run-dir', str(tmp_path/'cli'), '--dry-run'])
    assert worker_payload(json.loads(capsys.readouterr().out))['query_mean_quota_fraction'] == fraction
    elastic = elastic_profile(tmp_path, profile)
    elastic.update(query_fps_halo_blocks=1, query_mean_quota_fraction=fraction)
    resolved, _ = build_elastic(elastic, tmp_path, tmp_path/'elastic')
    backend = resolved['mean_k_backend_profile']
    assert resolved['query_mean_quota_fraction'] == backend['kvmem']['query_mean_quota_fraction'] == fraction
    actual = worker_payload(build_launch(backend, tmp_path, tmp_path/'backend'))
    assert runtime(actual).query_mean_quota_fraction == fraction
    replay, _ = build_elastic(resolved, tmp_path, tmp_path/'elastic-replay')
    assert replay['mean_k_backend_profile'] == backend
    assert profile == original
    assert not any((tmp_path/name).exists() for name in ('direct', 'cli', 'elastic', 'backend'))


@pytest.mark.parametrize('fraction', [True, False, None, '0.5', -.01, 1.01, float('nan'), float('inf'), -float('inf')])
def test_invalid_fraction_rejected_before_either_launcher_starts(tmp_path, fraction):
    settings = quota_settings(query_mean_quota_fraction=fraction)
    with pytest.raises(ValueError, match='query_mean_quota_fraction'):
        runtime(settings)
    profile = model_profile(tmp_path); profile['kvmem'] = settings
    with pytest.raises(ValueError, match='query_mean_quota_fraction'):
        build_launch(profile, tmp_path, tmp_path/'direct')
    elastic = elastic_profile(tmp_path, profile)
    elastic.update(query_fps_halo_blocks=1, query_mean_quota_fraction=fraction)
    with pytest.raises(ValueError, match='query_mean_quota_fraction'):
        build_elastic(elastic, tmp_path, tmp_path/'elastic')
    assert not (tmp_path/'direct').exists() and not (tmp_path/'elastic').exists()


@pytest.mark.parametrize('updates,match', [
    ({'query_fps_halo_blocks': 0}, 'query_mean_quota_fraction'),
    ({'query_score_mode': 'mean_k', 'query_fps_optimized': False}, 'query_mean_quota_fraction'),
    ({'query_score_mode': 'overlap_16_8', 'query_fps_optimized': False}, 'query_mean_quota_fraction'),
    ({'graph_alpha': .3}, 'graph_alpha=0 or 0.2'),
    ({'guided_query_tokens': 128}, 'guided_query_tokens=0'),
])
def test_quota_requires_fps_halo_and_explicit_qualified_graph_without_guidance(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        runtime(quota_settings(**updates))
    profile = model_profile(tmp_path); profile['kvmem'] = quota_settings(**updates)
    with pytest.raises(ValueError, match=match):
        build_launch(profile, tmp_path, tmp_path/'direct')
    elastic = elastic_profile(tmp_path, profile)
    elastic.update(query_fps_halo_blocks=1, query_mean_quota_fraction=.5)
    elastic.update(updates)
    with pytest.raises(ValueError, match=match):
        build_elastic(elastic, tmp_path, tmp_path/'elastic')


def test_quota_roundtrip_keeps_native_cpu_and_all_budget_settings(tmp_path):
    profile = model_profile(tmp_path)
    profile['kvmem'] = quota_settings(query_fps_prebuild=True,
        query_fps_prebuild_device='cpu', query_fps_cpu_backend='native', query_fps_cpu_workers=16,
        sink_blocks=41, recent_blocks=0, pressure_policy='attention')
    worker = worker_payload(build_launch(profile, tmp_path, tmp_path/'direct'))
    assert runtime(worker).query_mean_quota_fraction == .5
    for name in ('query_fps_cpu_backend', 'query_fps_cpu_workers', 'query_window_cache_bytes',
            'select_budget', 'gen_budget', 'sink_blocks', 'recent_blocks', 'pressure_policy'):
        assert worker[name] == profile['kvmem'][name]
