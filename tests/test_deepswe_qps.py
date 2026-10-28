"""Protect paired measurement design and distinguish delivered and partial output."""
import json
import pytest

from scripts import benchmark_deepswe_qps as bench


def test_pair_order_reverses_without_changing_workload():
    cells = bench.cell_plan([1, 2, 4, 8], 2, 300)
    assert len(cells) == 16
    for concurrency in (1, 2, 4, 8):
        first = [c for c in cells if c['concurrency'] == concurrency and c['repeat'] == 1]
        second = [c for c in cells if c['concurrency'] == concurrency and c['repeat'] == 2]
        assert [c['backend'] for c in first] == list(reversed([c['backend'] for c in second]))
        for pair in (first, second):
            assert pair[0]['seed'] == pair[1]['seed']
            assert pair[0]['seconds'] == pair[1]['seconds']


def test_qps_counts_completed_turns_and_keeps_inflight_tokens_separate(tmp_path, monkeypatch):
    import scripts.analyze_deepswe_interference as analysis
    monkeypatch.setattr(analysis, 'analyze', lambda path: dict(
        begin=100., end=110., seconds=10., completed_request_output_tokens=20,
        accepted_output_tokens=45, agent_tokens_per_second=4.5))
    bench.save(tmp_path/'run.json', dict(tasks={'a':dict(state='window_truncated')}))
    (tmp_path/'bridge').mkdir()
    events = [dict(event='start', task='a', turn=1, at=100.),
              dict(event='finish', task='a', turn=1, at=102.),
              dict(event='start', task='a', turn=2, at=103.),
              dict(event='finish', task='a', turn=2, at=111.)]
    (tmp_path/'bridge/requests.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    result = bench.summarize_cell(tmp_path)
    assert result['qps'] == .1
    assert result['delivered_output_tps'] == 2.
    assert result['agent_tokens_per_second'] == 4.5
    assert result['outstanding_at_boundary'] == 1
    assert result['official_results'] == {}
    assert result['request_wall_seconds']['mean'] == 2.

    events[0]['request_started_at']=99.
    events[1]['seconds']=3.
    (tmp_path/'bridge/requests.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    result=bench.summarize_cell(tmp_path)
    assert result['request_wall_seconds']['mean']==3.
    assert result['backend_roundtrip_seconds']['mean']==2.


def test_unpaired_and_zero_request_windows_do_not_invent_speedup(tmp_path):
    cells = bench.cell_plan([8], 1, 300)
    bench.save(tmp_path/'manifest.json', dict(cells=cells, measurement={}))
    directory=tmp_path/cells[0]['name'];directory.mkdir()
    bench.save(directory/'summary.json', dict(qps=0., agent_tokens_per_second=10.))
    assert bench.summarize(tmp_path)['pairs'] == []
    directory=tmp_path/cells[1]['name'];directory.mkdir()
    bench.save(directory/'summary.json', dict(qps=1., agent_tokens_per_second=20.))
    result=bench.summarize(tmp_path)
    assert result['complete']
    assert result['pairs'][0]['qps_gain_percent'] is None
    assert result['pairs'][0]['output_tps_gain_percent'] == 100.


def test_carry_refuses_changed_duration_or_incomplete_cleanup(tmp_path):
    source=tmp_path/'r1-native-c4';source.mkdir()
    bench.save(source/'manifest.json',dict(workers=4,seed=1000,backend_kind='native',duration_seconds=300))
    bench.save(source/'run.json',dict(status='window_complete',cleanup_errors=[]))
    bench.save(source/'summary.json',dict(errors=[]))
    bench.save(source/'profile.json',dict(max_model_len=262144))
    cell=dict(concurrency=4,seed=1000,backend='native',seconds=600)
    with pytest.raises(ValueError,match='differs'):
        bench.validate_carry(source,cell,dict(max_model_len=262144),tmp_path/'runtime')
    cell['seconds']=300
    bench.save(source/'run.json',dict(status='window_complete',cleanup_errors=['remaining container']))
    with pytest.raises(ValueError,match='cleanly'):
        bench.validate_carry(source,cell,dict(max_model_len=262144),tmp_path/'runtime')
