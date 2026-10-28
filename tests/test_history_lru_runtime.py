"""Production launch contract and graph-free metadata lifecycle for hybrid LRU."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from test_graph_options import runtime, model_profile, worker_payload
from test_query_fps_prebuild_options import fps_settings
from vllm_kvmem.serve import build_launch
from vllm_kvmem.prompt_spans import install_selection_metadata, selection_metadata
from vllm_kvmem.query_boundary_history import QueryBoundaryHistory


def settings(mode='delete'):
    return fps_settings(graph_mode='off', graph_alpha=0.,
        archive_backend='cpu', mean_k_backend='qw3_cuda',
        async_seal=True, cpu_history_coordination=True,
        historical_gc=dict(mode=mode, policy='lru', policy_params=dict(keep_mandatory=True)),
        cpu_release_sealed_sums=True, query_fps_cache_fp8=True,
        query_fps_halo_blocks=1, query_mean_quota_fraction=.5,
        query_fps_prebuild=True, query_fps_prebuild_device='cpu',
        query_fps_cpu_backend='native', query_fps_cpu_workers=16,
        pressure_policy='attention')


@pytest.mark.parametrize('mode', ['off', 'shadow', 'delete'])
def test_compact_hybrid_history_configuration_survives_public_launcher(tmp_path, mode):
    cfg = settings(mode)
    options = runtime(cfg)
    assert options.historical_gc.mode == mode
    assert options.graph_mode == 'off' and options.graph_alpha == 0
    assert options.query_mean_quota_fraction == .5 and options.query_fps_halo_blocks == 1
    assert options.pressure_policy == 'attention'
    assert options.cpu_release_sealed_sums and options.query_fps_cache_fp8
    profile = model_profile(tmp_path)
    profile['kvmem'] = cfg
    launch = build_launch(profile, tmp_path, tmp_path/'run')
    restored = runtime(worker_payload(launch))
    assert restored.historical_gc == options.historical_gc
    assert restored.query_fps_cpu_workers == 16
    assert restored.cpu_release_sealed_sums and restored.query_fps_cache_fp8
    assert worker_payload(build_launch(launch['profile'], tmp_path, tmp_path/'replay')) == worker_payload(launch)


@pytest.mark.parametrize('change,match', [
    (dict(graph_mode='multi'), 'block graph'),
    (dict(graph_alpha=.2), 'query_score_mode'),
    (dict(historical_gc=dict(mode='delete', policy_params=dict(keep_mandatory=False))), 'keep_mandatory'),
    (dict(query_history=True), 'query_history'),
    (dict(cpu_direct_record_seal=True, cpu_seal_serial_allocate=True), 'serial_allocate'),
])
def test_unsafe_combinations_are_rejected_before_startup(change, match):
    with pytest.raises(ValueError, match=match):
        replace(runtime(settings()), **change)


def test_metadata_install_keeps_future_input_and_prunes_only_missing_history():
    history = QueryBoundaryHistory('s', max_spans=3)
    authority = NS(query_boundaries=history, records={}, history_end=0)
    command = NS(message_spans=[(0, 64, 'user'), (64, 128, 'tool')], tokens=list(range(128)))
    install_selection_metadata(authority, command)
    assert history.message_spans == tuple(command.message_spans)
    assert selection_metadata(authority) is history
    history.advance(0, 128)
    authority.history_end = 128
    authority.records = {1: NS(history_begin=64, count=32)}
    command.message_spans.append((128, 160, 'user'))
    command.tokens = list(range(160))
    install_selection_metadata(authority, command, validated_prefix=128)
    assert history.message_spans == ((64, 128, 'tool'), (128, 160, 'user'))
    assert history.accepted_through == 128
    history.close()
