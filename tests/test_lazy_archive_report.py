"""Opt-in KVMEM_LAZY_ARCHIVE_REPORT: rate-limited, off-thread archive reports."""
import json
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

import vllm_kvmem.serving.capture_runtime as capture
from vllm_kvmem.serving.capture_runtime import CaptureRuntime
from vllm_kvmem.serving.report_writer import LatestReportWriter, lazy_archive_report


def _runtime(tmp_path, monkeypatch):
    archive = NS(tier=NS(used_slots=0), capacity_bytes=0, preallocated=False, io_usage=lambda: {})
    layer = NS(num_kv_heads=1, head_dim=2,
               attn=NS(kv_cache_dtype='fp8_e4m3', _v_scale=torch.tensor(1.0)))
    runner = NS(_kvmem_serving=True, _kvmem_layers=('layer',),
        _kvmem_prepared_archive=archive, model=NS(named_modules=lambda: [('layer', layer)]),
        device=torch.device('cpu'), decode_query_len=1,
        vllm_config=NS(kv_transfer_config=None, cache_config=NS(cache_dtype='fp8_e4m3'),
            max_concurrent_batches=1, scheduler_config=NS(max_num_batched_tokens=128, max_num_seqs=4)))
    monkeypatch.setattr(capture, 'PackedDownloads', lambda *args: NS(pending=[]))
    runtime = CaptureRuntime(runner, tmp_path)
    runtime.retirement = NS(poll=lambda _: {'history_end': 1})
    return runtime


def _retire(runtime, request_id):
    runtime.retiring[request_id] = {'request_id': request_id}
    assert runtime.retirement_done(request_id)


def test_flag_validation(monkeypatch):
    monkeypatch.delenv('KVMEM_LAZY_ARCHIVE_REPORT', raising=False)
    assert not lazy_archive_report()
    monkeypatch.setenv('KVMEM_LAZY_ARCHIVE_REPORT', '1')
    assert lazy_archive_report()
    monkeypatch.setenv('KVMEM_LAZY_ARCHIVE_REPORT', 'yes')
    with pytest.raises(ValueError):
        lazy_archive_report()


def test_flag_off_writes_every_report_synchronously_and_indented(tmp_path, monkeypatch):
    monkeypatch.delenv('KVMEM_LAZY_ARCHIVE_REPORT', raising=False)
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.retiring['other'] = {'request_id': 'other'}          # still busy
    _retire(runtime, 'a')
    text = (tmp_path / 'archive.json').read_text()
    assert text.startswith('{\n  "scope"')
    assert json.loads(text)['completed'][-1]['request_id'] == 'a'
    assert 'lazy_archive_reports_skipped' not in json.loads(text)
    assert '_report_writer' not in runtime.__dict__


def test_lazy_skips_only_while_busy_and_final_state_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_LAZY_ARCHIVE_REPORT', '1')
    runtime = _runtime(tmp_path, monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(capture.time, 'monotonic', lambda: clock[0])
    runtime.retiring['other'] = {'request_id': 'other'}          # keeps the runtime busy
    _retire(runtime, 'a')                                        # first report: written
    runtime._report_writer.flush()
    assert json.loads((tmp_path / 'archive.json').read_text())['completed'][-1]['request_id'] == 'a'
    for index in range(5):                                       # within the interval: skipped
        _retire(runtime, f'b{index}')
    runtime._report_writer.flush()
    report = json.loads((tmp_path / 'archive.json').read_text())
    assert report['completed'][-1]['request_id'] == 'a'
    assert runtime._archive_report_skipped == 5
    clock[0] += 1.01                                             # interval elapsed: written while busy
    _retire(runtime, 'c')
    runtime._report_writer.flush()
    report = json.loads((tmp_path / 'archive.json').read_text())
    assert report['completed'][-1]['request_id'] == 'c'
    assert report['lazy_archive_reports_skipped'] == 5 and report['retiring_requests'] == 1
    _retire(runtime, 'd')                                        # skipped again
    _retire(runtime, 'other')                                    # runtime idle: never skipped
    runtime._report_writer.flush()
    report = json.loads((tmp_path / 'archive.json').read_text())
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert report['completed'][-1]['request_id'] == 'other' and report['retiring_requests'] == 0
    assert report['completed_requests'] == progress['completed_requests'] == 9
    assert runtime._archive_report_skipped == 0


def test_shutdown_flushes_a_skipped_report(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_LAZY_ARCHIVE_REPORT', '1')
    runtime = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(capture.time, 'monotonic', lambda: 5.0)
    runtime.retiring['other'] = {'request_id': 'other'}
    _retire(runtime, 'a')
    _retire(runtime, 'b')                                        # skipped, runtime still busy
    writer = runtime._report_writer
    monkeypatch.setattr(runtime, '_shutdown_retirement', lambda: None)
    runtime.shutdown_retirement()
    report = json.loads((tmp_path / 'archive.json').read_text())
    assert report['completed'][-1]['request_id'] == 'b'
    assert '_report_writer' not in runtime.__dict__ and not writer._thread.is_alive()


def test_worker_feedback_continues_while_resource_snapshot_write_is_blocked(tmp_path, monkeypatch):
    from vllm_kvmem.serving import report_writer
    monkeypatch.setenv('KVMEM_LAZY_ARCHIVE_REPORT', '1')
    entered, release = Event(), Event()
    write = report_writer.write_report

    def slow_write(output, name, report, **kwargs):
        if name == 'archive_resources':
            entered.set()
            assert release.wait(5)
        write(output, name, report, **kwargs)

    monkeypatch.setattr(report_writer, 'write_report', slow_write)
    runtime = _runtime(tmp_path, monkeypatch)
    reporter = runtime._archive_resource_reporter
    reporter.interval = 0
    try:
        runtime.publication_blocked_requests()
        assert entered.wait(5)
        reporter.publish(2, 1)
        assert not (tmp_path / 'archive_resources.json').exists()
        release.set()
        reporter.publish(0, 0, force=True)
        terminal = json.loads((tmp_path / 'archive_resources.json').read_text())
        assert (terminal['resource_sample'], terminal['active_requests'], terminal['retiring_requests']) == (3, 0, 0)
    finally:
        release.set()
        runtime._flush_reports()


def test_writer_keeps_the_latest_snapshot_and_reports_errors(tmp_path):
    writer = LatestReportWriter(tmp_path)
    for index in range(50):
        writer.submit('archive', {'value': index})
    writer.flush()
    assert json.loads((tmp_path / 'archive.json').read_text()) == {'value': 49}
    assert writer.written + writer.replaced == writer.submitted == 50
    writer.submit('archive', {'value': object()})                # not serializable
    with pytest.raises(TypeError):
        writer.flush()
    writer.submit('progress', {'ok': 1})
    writer.close()
    assert json.loads((tmp_path / 'progress.json').read_text()) == {'ok': 1}
    with pytest.raises(RuntimeError):
        writer.submit('progress', {})
