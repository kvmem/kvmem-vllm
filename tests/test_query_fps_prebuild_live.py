"""CPU protocol tests; never starts a server or reads CUDA state."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Barrier, Lock

import numpy as np
import pytest

import benchmark_query_fps_prebuild_live as bench
from test_ruler_fps_answers import original_profile


def profile():
    result = original_profile()
    result['kvmem'].update(query_score_mode='key_fps_4', query_fps_halo_blocks=1, query_fps_optimized=True)
    return result


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('suite', ['single', 'eight'])
def test_profiles_change_only_prebuild_within_pair_and_reserve_cache_inside_existing_cpu_budget(device, suite):
    original = profile(); saved = deepcopy(original)
    trials = bench.schedule(suite)[:2]
    left, right = [bench.profile_for_trial(original, trial, device=device) for trial in trials]
    bench.assert_paired_profiles(left, right)
    assert original == saved
    assert left['kvmem']['query_window_cache_bytes'] == right['kvmem']['query_window_cache_bytes'] == 12*1024**3
    assert left['kvmem']['cpu_read_cache_bytes'] == right['kvmem']['cpu_read_cache_bytes'] == 100000000000
    assert left['kvmem']['query_fps_prebuild_device'] == right['kvmem']['query_fps_prebuild_device'] == device
    assert left['max_num_seqs'] == (1 if suite == 'single' else 8)
    assert set(bench.profile_diff(left, right)) == {'kvmem.query_fps_prebuild'}
    right['kvmem']['gen_budget'] *= 2
    with pytest.raises(ValueError, match='differ beyond'): bench.assert_paired_profiles(left, right)


def test_schedule_is_fixed_ab_ba_and_eight_ab_without_answer_selection():
    trials = bench.schedule('both')
    assert [row['prebuild'] for row in trials] == [False, True, True, False, False, True]
    assert all(row['cases'] == list(bench.halo.CASE_IDS[:2]) for row in trials[:4])
    assert all(row['cases'] == list(bench.halo.CASE_IDS[:8]) for row in trials[4:])
    assert sum(len(row['cases']) for row in trials) == 24
    assert len({row['id'] for row in trials}) == 6


def test_eight_request_trace_limit_is_checked_before_launch(tmp_path):
    for case in bench.EIGHT_CASES:
        path=tmp_path/bench.halo.ARM/'cases'/case/'query-inputs.jsonl'
        path.parent.mkdir(parents=True); path.write_bytes(b'x'*100)
    assert bench.verify_trace_capacity(tmp_path,bench.schedule('single'),300)['previous_largest_trial_bytes']==200
    with pytest.raises(ValueError,match='trace capacity is too small'):
        bench.verify_trace_capacity(tmp_path,bench.schedule('eight'),800)
    assert bench.verify_trace_capacity(tmp_path,bench.schedule('eight'),900)['previous_largest_trial_bytes']==800


def test_shared_query_log_is_split_by_exact_native_request_identity_without_mutation(tmp_path):
    events = [dict(request_id='cmpl-other-0', phase='query', raw_scores=[99]),
        dict(request_id='cmpl-a-0-worker', phase='query', raw_scores=[.2,.8]),
        dict(request_id='cmpl-ab-0', phase='query', raw_scores=[88])]
    path = tmp_path/'block_graph_inputs.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in events))
    original = path.read_bytes(); target = tmp_path/'case-query.jsonl'
    assert bench.save_request_trace(tmp_path, dict(response_id='cmpl-a'), target) == events[1]
    assert path.read_bytes() == original
    assert json.loads(target.read_text()) == events[1]
    with pytest.raises(FileExistsError): bench.save_request_trace(tmp_path, dict(response_id='cmpl-a'), target)
    with pytest.raises(ValueError, match='Missing or duplicate'):
        bench.save_request_trace(tmp_path, dict(response_id='unknown'), tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def sample_result():
    response = dict(response_id='cmpl-a', chunks=[[20.,4],[22.,3],[23.,2]],
        output_ids=list(range(9)), seconds=23.1, ttft_seconds=20.)
    query = dict(selection='query', refresh_started_at=110., refresh_finished_at=118.,
        selected_history_ranges=[[0,32]], query_key_fps_scoring=dict(cache_build_seconds=2.,scoring_seconds=1.))
    record = dict(starts=[0,32], counts=[32,32], score_mask=[True,True], transport_mask=[True,True],
        mandatory_starts=[0], budget=32, raw_scores=[.3,.7])
    evidence = dict(bounded=[query], query_inputs=[record])
    return dict(response=response, evidence=evidence, metrics={}, result=dict(id='fixed',official_case_score=100,value_covered=False))


def test_metrics_separate_query_wait_from_prefill_and_use_accepted_chunks_for_mtp_decode():
    item = sample_result()
    steps = [dict(model_submitted_at=101.,batch_geometry=[dict(request_id='other',prefill=True)]),
        dict(model_submitted_at=103.,batch_geometry=[dict(request_id='cmpl-a-0',prefill=True)]),
        dict(model_submitted_at=119.,batch_geometry=[dict(request_id='cmpl-a-0',prefill=True)])]
    metrics = bench.phase_metrics(item['response'],item['evidence'],client_started_at=100.,steps=steps)
    assert metrics['native_prefill_before_query_wall_seconds'] == 7.
    assert metrics['client_to_query_seconds'] == 10.
    assert metrics['query_refresh_seconds'] == 8.
    assert metrics['query_to_first_token_seconds'] == 2.
    assert metrics['decode_stream_seconds'] == 3.
    assert metrics['accepted_tokens_after_first_chunk'] == 5
    assert metrics['decode_stream_tokens_per_second'] == 5/3
    assert metrics['completion_tokens'] == 9


def test_pair_comparison_keeps_answer_accuracy_coverage_and_seed_differences_distinct():
    left = sample_result(); right = deepcopy(left)
    result = bench.compare_pair(left,right)
    assert result['seeds_exactly_equal'] and result['picker_equal'] and result['generated_tokens_equal']
    assert result['after']['score'] == 100 and result['after']['value_covered'] is False
    right['evidence']['query_inputs'][0]['raw_scores'][0] += 1e-5
    right['response']['output_ids'].append(9)
    result = bench.compare_pair(left,right)
    assert not result['seeds_exactly_equal'] and result['picker_equal'] and not result['generated_tokens_equal']
    assert result['maximum_absolute_seed_difference'] == pytest.approx(1e-5)


def test_actual_prebuild_backend_and_fallback_counters_are_audited():
    profile = {'kvmem': dict(query_fps_prebuild=True, query_fps_prebuild_device='cpu')}
    state = dict(enabled=True, device='cpu', last_error='preserved diagnostic')
    for key in ('queued_records', 'published_records', 'skipped_records', 'failed_records',
            'deferred_records', 'cancelled_records', 'stale_records', 'query_wait_seconds', 'build_seconds'):
        state[key] = 0
    state['failed_records'] = 2
    query = dict(query_key_fps_scoring=dict(prebuild=state,cache_prebuilt_hit_blocks=10))
    audit = bench.audit_prebuild(query,profile)
    assert audit['failed_records'] == 2 and audit['last_error'] == 'preserved diagnostic'
    assert audit['cache_prebuilt_hit_blocks'] == 10
    profile['kvmem']['query_fps_prebuild_device']='cuda'
    with pytest.raises(ValueError,match='backend differs'): bench.audit_prebuild(query,profile)


def test_eight_submission_barrier_preserves_independent_original_inputs_and_failure_evidence(tmp_path, monkeypatch):
    inputs = tmp_path/'inputs'; inputs.mkdir()
    directory = tmp_path/'trial'; directory.mkdir()
    bench.save(directory/'profile.json', dict(only='cpu mock'))
    fixture = dict(input_directory=str(inputs), sampling=dict(max_tokens=8192,thinking_token_budget=4096))
    rows=[]
    for i in range(8):
        name=f'case-{i}'; file=inputs/(name+'.i32'); np.asarray([i,10],dtype='<i4').tofile(file)
        rows.append(dict(id=name, served_sha256=bench.sha(file), seed=42+i, served_query=[1,2]))
    seen=[]; lock=Lock(); arrived=Barrier(8)
    def fake_stream(profile, session, tokens, query, sampling, timeout):
        with lock: seen.append((tokens, query, sampling.copy(), session))
        arrived.wait(timeout=3)
        if tokens[0] == 7: raise RuntimeError('injected stream failure')
        return dict(seconds=.001, output_ids=[9])
    monkeypatch.setattr(bench.halo.client,'stream',fake_stream)
    barrier=Barrier(8)
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks=[executor.submit(bench.submit_case,tmp_path,directory,fixture,row,{},barrier) for row in rows]
        for task in tasks[:7]: assert task.result()[3]['output_ids'] == [9]
        with pytest.raises(RuntimeError,match='injected'): tasks[-1].result()
    assert len(seen) == len({row[3] for row in seen}) == 8
    assert sorted(row[2]['seed'] for row in seen) == list(range(42,50))
    assert all(row[2]['max_tokens']==8192 and row[2]['thinking_token_budget']==4096 for row in seen)
    assert (directory/'cases/case-7/failure.json').exists()
    assert all((directory/f'cases/case-{i}/response.json').exists() for i in range(7))


def test_existing_root_is_never_overwritten(tmp_path):
    marker=tmp_path/'old-response.json'; marker.write_text('immutable')
    with pytest.raises(ValueError,match='exclusive'):
        bench.prepare(tmp_path,tmp_path/'missing',suite='single',device='cpu')
    assert marker.read_text() == 'immutable'
