"""Native construction is explicit, bounded, and survives both launch paths."""
from dataclasses import replace

import pytest

from test_graph_options import model_profile,runtime,worker_payload
from test_query_fps_prebuild_options import fps_settings,elastic_profile
from vllm_kvmem.elastic_serve import build_launch as build_elastic
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def native_settings(**updates):
    result=fps_settings(query_fps_prebuild=True,query_fps_cpu_backend='native',query_fps_cpu_workers=8)
    result.update(updates)
    return result


def test_native_cpu_defaults_preserve_existing_torch_path():
    assert RuntimeOptions().query_fps_cpu_backend=='torch'
    assert RuntimeOptions().query_fps_cpu_workers==8
    assert runtime(fps_settings()).query_fps_cpu_backend=='torch'
    assert runtime(fps_settings(query_fps_prebuild=True,query_fps_prebuild_device='cuda')).query_fps_cpu_backend=='torch'


@pytest.mark.parametrize('workers',[1,8,16,32])
def test_native_cpu_backend_and_bounded_pool_survive_direct_and_elastic_launch(tmp_path,workers):
    local=model_profile(tmp_path);local['kvmem']=native_settings(query_fps_cpu_workers=workers)
    worker=worker_payload(build_launch(local,tmp_path,tmp_path/'native'))
    options=runtime(worker)
    assert options.query_fps_cpu_backend=='native' and options.query_fps_cpu_workers==workers
    assert replace(options,query_fps_cpu_backend='torch').query_fps_cpu_workers==workers
    elastic=elastic_profile(tmp_path,local)
    elastic.update(query_fps_prebuild=True,query_fps_cpu_backend='native',query_fps_cpu_workers=workers)
    resolved,_=build_elastic(elastic,tmp_path,tmp_path/'elastic')
    actual=worker_payload(build_launch(resolved['mean_k_backend_profile'],tmp_path,tmp_path/'backend'))
    assert actual['query_fps_cpu_backend']=='native' and actual['query_fps_cpu_workers']==workers
    replay,_=build_elastic(resolved,tmp_path,tmp_path/'replay')
    assert replay['mean_k_backend_profile']==resolved['mean_k_backend_profile']


@pytest.mark.parametrize('update',[
    dict(query_fps_cpu_backend=None),dict(query_fps_cpu_backend=True),dict(query_fps_cpu_backend='auto'),
    dict(query_fps_cpu_workers=0),dict(query_fps_cpu_workers=33),dict(query_fps_cpu_workers=True),
    dict(query_fps_cpu_workers=8.),dict(query_fps_cpu_workers='8'),
    dict(query_fps_prebuild=False),dict(query_fps_prebuild_device='cuda'),
    dict(query_fps_optimized=False),dict(query_score_mode='mean_k'),
])
def test_invalid_native_cpu_settings_fail_in_runtime_and_both_launchers(tmp_path,update):
    settings=native_settings(**update)
    with pytest.raises(ValueError,match='query_fps_cpu_'):runtime(settings)
    local=model_profile(tmp_path);local['kvmem']=settings
    with pytest.raises(ValueError,match='query_fps_cpu_'):build_launch(local,tmp_path,tmp_path/'native')
    elastic=elastic_profile(tmp_path,local)
    elastic.update(query_fps_prebuild=True,query_fps_cpu_backend='native',query_fps_cpu_workers=8)
    elastic.update(update)
    with pytest.raises(ValueError,match='query_fps_cpu_'):build_elastic(elastic,tmp_path,tmp_path/'elastic')
    assert not (tmp_path/'native').exists() and not (tmp_path/'elastic').exists()
