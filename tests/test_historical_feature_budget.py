from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from vllm_kvmem.historical_feature_budget import (
    REDUNDANCY_Q_STAGING_BYTES, redundancy_policy_metadata_bytes,
    split_historical_feature_budget,
)
from vllm_kvmem.historical_gc import HistoricalGcSettings
from vllm_kvmem.query_window_budget import split_cpu_history_budget
from test_historical_options import settings


def feature_options(scopes=3):
    return SimpleNamespace(historical_gc=HistoricalGcSettings(policy='redundancy'),
        memory=SimpleNamespace(repository_blocks=16384), session_capacity=scopes,
        cpu_read_cache_bytes=100*1024**3, query_score_mode='key_fps_4',
        query_window_cache_bytes=12*1024**3)


def test_both_caches_and_all_live_session_owners_fit_the_same_total_budget():
    options = feature_options()
    history, retrieval = split_cpu_history_budget(options)
    available, reserved = split_historical_feature_budget(options, history, 11)
    params = options.historical_gc.policy_params
    metadata = redundancy_policy_metadata_bytes(16384, params)
    assert reserved == 11*((128+8)*1024**2+metadata)+REDUNDANCY_Q_STAGING_BYTES
    assert available + reserved + retrieval == options.cpu_read_cache_bytes
    # The staging reservation is shared rather than multiplied by all owners.
    _, one = split_historical_feature_budget(options, history, 1)
    assert reserved-REDUNDANCY_Q_STAGING_BYTES == 11*(one-REDUNDANCY_Q_STAGING_BYTES)


@pytest.mark.parametrize('historical', [None, HistoricalGcSettings(policy='lru')])
def test_existing_policies_do_not_reserve_redundancy_memory(historical):
    assert split_historical_feature_budget(SimpleNamespace(historical_gc=historical), 42, 0) == (42, 0)


def test_reservation_fails_before_capacity_can_be_overadmitted():
    options = feature_options()
    _, needed = split_historical_feature_budget(options, options.cpu_read_cache_bytes, 8)
    for total in (needed, needed-1, 0):
        with pytest.raises(ValueError, match='CPU history budget'):
            split_historical_feature_budget(options, total, 8)
    assert split_historical_feature_budget(options, needed+1, 8) == (1, needed)


@pytest.mark.parametrize('flag', ['offline_prefill_coalescing', 'prefill_physical_coalescing',
    'prefill_publication_pipeline', 'early_capture_slot_release', 'prefill_capture_ahead'])
def test_unqualified_publication_modes_fail_explicitly(flag):
    with pytest.raises(ValueError, match='redundancy does not yet support'):
        replace(settings(), historical_gc={'policy': 'redundancy'}, **{flag: True})


def test_native_archive_and_scheduler_charge_identical_feature_reservations(tmp_path):
    from vllm_kvmem.historical_admission import HistoricalHistoryAdmission
    from vllm_kvmem.historical_gc import historical_record_charge
    from vllm_kvmem.storage import make_shared_archive
    options = replace(settings('off'), cpu_read_cache_bytes=4*1024**3,
        historical_gc={'policy': 'redundancy', 'mode': 'off',
                       'policy_params': {'feature_cache_bytes': 1024**2,
                                         'query_cache_bytes': 1024**2}})
    _, reserved = split_historical_feature_budget(options, options.cpu_read_cache_bytes, 2)
    options = replace(options, cpu_read_cache_bytes=reserved+1200)
    history, _ = split_cpu_history_budget(options)
    history, expected = split_historical_feature_budget(options, history, 2)
    charge = historical_record_charge(24, 4)
    admission = HistoricalHistoryAdmission(history, charge, 32, 4)
    archive = make_shared_archive(options, tmp_path, 24, 1)
    try:
        assert archive.tier.capacity == admission.ledger.capacity_records == 1200//charge
        assert archive.io_usage()['historical_feature_reserved_bytes'] == expected == reserved
        assert archive.tier.capacity*charge+reserved <= options.cpu_read_cache_bytes
    finally:
        deadline = time.monotonic()+5
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)


def test_redundancy_public_profile_roundtrip_preserves_policy_knobs(tmp_path):
    import json
    from copy import deepcopy
    from test_graph_options import model_profile, worker_payload, runtime
    from vllm_kvmem.serve import build_launch
    profile = model_profile(tmp_path)
    gc = dict(mode='shadow', policy='redundancy', policy_params={
        'representative_similarity': 0.92, 'candidate_blocks': 96,
        'importance_weight': 0.2, 'allow_lossy_fallback': False})
    profile['kvmem'].update(graph_mode='off', archive_backend='cpu',
        cpu_read_cache_bytes=16*1024**3, async_seal=True,
        cpu_history_coordination=True, historical_gc=gc)
    original = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'serve')
    worker = worker_payload(launch)
    assert worker['historical_gc'] == gc
    parsed = runtime(worker)
    native = parsed.historical_gc.native(parsed.memory, 2)
    assert native.policy == 'redundancy'
    assert native.policy_params['representative_similarity'] == .92
    assert native.policy_params['candidate_blocks'] == 96
    replay = build_launch(json.loads(json.dumps(launch['profile'])), tmp_path, tmp_path/'replay')
    assert worker_payload(replay)['historical_gc'] == gc and profile == original
