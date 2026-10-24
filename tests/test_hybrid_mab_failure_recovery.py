from copy import deepcopy
from pathlib import Path

import pytest

from recover_hybrid_mab_failure import KIND, validate_request, verify_recovery
from benchmark_niah import save
import run_hybrid_memory_campaign as campaign


def test_recovery_keeps_native_full_cohort_concurrency():
    fixture=dict(suite='memoryagentbench',qualification=False,reference='/original',
        recovery_source='/failed',recovery_kind=KIND)
    command=campaign.client_command(Path('/recovery'),fixture)
    assert '--resume' in command and '--case-id' not in command
    assert command[command.index('--workers')+1] == '8'
    assert command[command.index('--active-groups')+1] == '8'
    fixture['recovery_kind']='unqualified'
    with pytest.raises(ValueError,match='Unsupported recovery'):campaign.client_command(Path('/r'),fixture)


def test_request_guard_rejects_seed_caps_and_boundary_changes():
    sampling=dict(temperature=.6,top_p=.95,top_k=20,min_p=0,presence_penalty=0,
        repetition_penalty=1,seed=42,max_tokens=1200)
    row=dict(token_sha256='tokens',input_tokens=100000,query_begin=99000,query_end=99990)
    request=dict(sampling,token_sha256='tokens',prompt_tokens=100000,thinking_token_budget=0,
        prefix_tokens=98304,kvmem_query_span=[99000,99990],kvmem_replay_span=[98304,99990])
    validate_request(request,row,sampling,98304)
    for key,value in [('seed',43),('max_tokens',100),('prompt_tokens',99999),
        ('kvmem_query_span',[99001,99990]),('kvmem_replay_span',[98000,99990])]:
        with pytest.raises(ValueError):validate_request({**request,key:value},row,sampling,98304)


def test_failure_recovery_refuses_engine_or_profile_changes(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    fixture=dict(suite='memoryagentbench',qualification=False,recovery_kind=KIND,
        rows=[dict(id='a')],case_ids=['a'],cases=1,source_cohort_sha256='fixed',reference='/ref',
        source_fixture='/fixture',source_fixture_sha256='f',sampling='same',workers=8,active_groups=8,cpu_headroom_gib=16)
    profile=dict(kvmem=dict(select_budget=36864,graph_alpha=.2))
    save(old/'fixture.json',fixture);save(old/'profile.json',profile)
    save(old/'status.json',dict(status='failed'));save(old/'run/cleanup.json',dict(remaining_owned_groups={},gpu_processes=[]))
    files={'src/vllm_kvmem/runner.py':'same','scripts/run_hybrid_memory_campaign.py':'old'}
    save(old/'runtime-proof.json',dict(files=files))
    save(new/'runtime-proof.json',dict(files={**files,'scripts/run_hybrid_memory_campaign.py':'new'}))
    proof=dict(source=str(old),kind=KIND,engine_sources_identical=True,regenerated_answers=0,
        pending_ids=['a'],recovered_ids=[],source_files={})
    verify_recovery(new,fixture,profile,proof)
    bad=deepcopy(profile);bad['kvmem']['graph_alpha']=0
    with pytest.raises(ValueError,match='profile'):verify_recovery(new,fixture,bad,proof)
    bad=deepcopy(fixture);bad['workers']=1
    with pytest.raises(ValueError,match='concurrency'):verify_recovery(new,bad,profile,proof)
    save(new/'runtime-proof.json',dict(files={**files,'src/vllm_kvmem/runner.py':'changed'}))
    with pytest.raises(ValueError,match='engine sources'):verify_recovery(new,fixture,profile,proof)
