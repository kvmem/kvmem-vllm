"""Explicit execution switches cannot silently change selection or budgets."""
from copy import deepcopy

import pytest

from test_graph_options import runtime, settings, model_profile, worker_payload
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def configured(**updates):
    return settings(async_refresh=True,async_restore=True,cpu_history_coordination=True,
        archive_backend="cpu",maintenance_max_steps=8,async_seal=True,cpu_read_cache_bytes=4*1024**3,
        **updates)


def test_defaults_off_and_worker_roundtrip(tmp_path):
    assert RuntimeOptions().session_gpu_checkpoint_bytes==0
    assert RuntimeOptions().restore_prepare_split is False
    profile=model_profile(tmp_path)
    profile['kvmem']=configured(session_gpu_checkpoint_bytes=2*1024**3,restore_prepare_split=True)
    before=deepcopy(profile)
    worker=worker_payload(build_launch(profile,tmp_path,tmp_path/'run'))
    value=runtime(worker)
    assert value.session_gpu_checkpoint_bytes==2*1024**3 and value.restore_prepare_split
    assert worker['select_budget']==8192 and worker['gen_budget']==4096
    assert profile==before and not (tmp_path/'run').exists()


@pytest.mark.parametrize('updates', [
    {'session_gpu_checkpoint_bytes':-1}, {'session_gpu_checkpoint_bytes':True},
    {'session_gpu_checkpoint_bytes':17*1024**3}, {'restore_prepare_split':1},
])
def test_invalid_limits_rejected(updates):
    with pytest.raises(ValueError):runtime(configured(**updates))


@pytest.mark.parametrize('updates', [
    {'session_gpu_checkpoint_bytes':1}, {'restore_prepare_split':True},
])
def test_new_paths_require_async_restore(updates):
    with pytest.raises(ValueError):runtime(settings(**updates))
