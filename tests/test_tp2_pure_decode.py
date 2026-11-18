import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('native', [False, True])
def test_cold_sweep_starts_each_case_without_retained_sessions(tmp_path, monkeypatch, native):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarks/tp2'))
    import pure_decode_workload as workload

    active = set()
    releases = []

    def build(length):
        assert not active
        return [length], []

    def request(directory, ids, span, count, barrier, use_kvmem):
        assert use_kvmem == (not native)
        directory.mkdir()
        if use_kvmem:
            active.add(directory)
        return dict(session_id=str(directory), output_tokens=count, token_accounting_passed=True,
                    ttft_seconds=.1, start_perf_counter=1., chunks=[(.1, 1), (.2, 1)])

    def release(row, directory):
        assert row['session_id'] == str(directory)
        assert (directory.parent / 'summary.json').is_file()
        active.remove(directory)
        releases.append(directory)

    monkeypatch.setattr(workload, 'request_one', request)
    monkeypatch.setattr(workload, 'release_session', release)
    monkeypatch.setattr(workload, 'metrics', lambda: {'owners': len(active)})
    args = SimpleNamespace(sweep_cases='8:2,16:2', seqs=2, max_len=32, pure_decode=2, native=native)
    workload.run(tmp_path, args, build)
    assert not active
    assert len(releases) == (0 if native else 4)
    for path in (tmp_path / 'cases').glob('*/metrics.json'):
        data = json.loads(path.read_text())
        assert data['before']['owners'] == 0
        assert data['after']['owners'] == (0 if native else 2)
