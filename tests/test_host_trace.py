import json
from types import SimpleNamespace
import pytest
from vllm_kvmem.host_trace import HostTrace


def test_nested_timing_retains_result_exception_and_no_payload(tmp_path):
    obj = SimpleNamespace(request_id='a')
    obj.inner = lambda value: value
    obj.outer = lambda value: obj.inner(value)
    obj.fail = lambda: (_ for _ in ()).throw(ValueError('original'))
    trace = HostTrace(tmp_path/'trace.jsonl')
    for name in ('outer', 'inner', 'fail'):
        trace.wrap(obj, name, name)
    value = object()
    assert obj.outer(value) is value
    with pytest.raises(ValueError, match='original'): obj.fail()
    trace.close()
    rows = [json.loads(l) for l in trace.path.read_text().splitlines()]
    inner, outer, failure = rows
    assert inner['parent'] == outer['span'] and outer['parent'] is None
    assert outer['started_at'] <= inner['started_at'] <= inner['finished_at'] <= outer['finished_at']
    assert failure['error'] == 'ValueError'
    status = json.loads(trace.path.with_suffix('.status.json').read_text())
    assert status['written'] == 3 and status['dropped'] == 0 and status['error'] is None
    assert obj.outer(value) is value  # Closed recorder is a no-op.


def test_queue_saturation_never_blocks_caller(tmp_path):
    trace = HostTrace(tmp_path/'trace.jsonl', capacity=1)
    obj = SimpleNamespace(call=lambda: 5)
    trace.wrap(obj, 'call', 'call')
    for _ in range(20000): assert obj.call() == 5
    trace.close()
    assert trace.dropped > 0 and trace.recorded + trace.dropped == 20000
    assert trace.written == trace.recorded
