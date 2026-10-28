from copy import deepcopy
from pathlib import Path

import pytest

import graph_v2_protocol as original
import hybrid_memory_protocol as policy
import run_hybrid_memory_campaign as campaign
from test_graph_v2_ablation import base_profile


def profile(suite):
    result = original.profiles(base_profile(),18871,suite)['C']
    result['kvmem'].update(query_mean_quota_fraction=.5,query_score_mode='key_fps_4',
        query_fps_halo_blocks=1,query_fps_optimized=True,query_fps_prebuild=True,
        query_fps_prebuild_device='cpu',query_fps_cpu_backend='native',query_fps_cpu_workers=16,
        query_window_cache_bytes=12*1024**3,graph_alpha=0.,graph_aggregation='sum',prefill_physical_coalescing=False)
    return result


@pytest.mark.parametrize('suite',['longmemeval_s','memoryagentbench'])
def test_hybrid_profile_retains_original_memory_geometry_and_requires_exact_quota(suite):
    value = profile(suite)
    policy.validate_profile(value,policy.infer_arm(value),suite)
    for key,bad in dict(query_mean_quota_fraction=.25,graph_alpha=.3,query_fps_cpu_workers=8,
            guided_query_tokens=128,query_history=True,select_budget=32768).items():
        changed = deepcopy(value); changed['kvmem'][key] = bad
        with pytest.raises(ValueError): policy.validate_profile(changed,policy.infer_arm(changed),suite)


def test_longest_prefix_qualification_uses_both_original_questions_and_full_eight_worker_limit():
    command = campaign.client_command(Path('/tmp/frozen-campaign'),dict(suite='memoryagentbench',qualification=True,
        reference='/tmp/original',case_ids=['original_q0','original_q1']))
    assert command.count('--case-id') == 2
    assert command[command.index('--workers')+1] == command[command.index('--active-groups')+1] == '8'
    assert command[command.index('--cpu-headroom-gib')+1] == '16'
    assert '--group-limit' not in command and '--per-group-limit' not in command


def test_existing_memory_campaign_and_unfrozen_runner_refuse_overwrite(tmp_path):
    output = tmp_path/'response.json'; output.write_text('original result')
    with pytest.raises(ValueError,match='fresh memory'): campaign.prepare(tmp_path,'memoryagentbench')
    with pytest.raises(ValueError,match='frozen controller'): campaign.run(tmp_path)
    assert output.read_text() == 'original result'


@pytest.mark.parametrize('query,prompt,expected',[
    ([108328,108361],108368,36768),
    ([102919,102956],102963,36768),
    ([109956,109983],109990,36864),
    ([108000,108100],108107,36704),
])
def test_native_budget_reserves_unaligned_tail_without_shortening_output(query,prompt,expected):
    value = profile('longmemeval_s')
    start = query[0]//32*32
    event = dict(query_span=query,history_begin=start,replay_span=[start,query[1]])
    assert policy.expected_query_budget(value,event,prompt_tokens=prompt,output_tokens=16384) == expected
    # MAB's smaller per-question output reserve leaves the full K budget.
    assert policy.expected_query_budget(value,event,prompt_tokens=prompt,output_tokens=1024) == 36864
    event['history_begin'] += 32
    with pytest.raises(ValueError,match='coordinates'):
        policy.expected_query_budget(value,event,prompt_tokens=prompt,output_tokens=16384)


def test_mab_frozen_prefix_replay_is_bound_to_original_request_coordinates():
    value = profile('memoryagentbench')
    event = dict(query_span=[100100,100200],history_begin=99328,replay_span=[99328,100200])
    assert policy.expected_query_budget(value,event,prompt_tokens=100207,output_tokens=16384,
        replay_span=[99328,100200]) == 35936
    assert policy.expected_query_budget(value,event,prompt_tokens=100207,output_tokens=1024,
        replay_span=[99328,100200]) == 36864
    with pytest.raises(ValueError,match='coordinates'):
        policy.expected_query_budget(value,event,prompt_tokens=100207,output_tokens=16384)
    with pytest.raises(ValueError,match='replay bounds'):
        policy.expected_query_budget(value,event,prompt_tokens=100207,output_tokens=16384,
            replay_span=[99328,100201])
