import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analyze_deepswe_interference import prefix_rows,tool_counts,analyze


def test_package_import_does_not_require_scripts_on_pythonpath(tmp_path):
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-c','from scripts.analyze_deepswe_interference import analyze'],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(root)},check=True,capture_output=True)


def test_boundary_ignores_later_and_incomplete_records(tmp_path):
    p=tmp_path/'events';one=json.dumps(dict(type='user'))+'\n';two=json.dumps(dict(type='assistant'))+'\n'
    p.write_text(one+two)
    assert prefix_rows(p,len(one)+5)==[dict(type='user')]


def test_tool_results_deduplicate_and_keep_pending_separate():
    rows=[dict(type='assistant',message=dict(content=[dict(type='tool_use',id='1',name='Bash'),dict(type='tool_use',id='2',name='Read')])),
          dict(type='user',message=dict(content=[dict(type='tool_result',tool_use_id='1',is_error=True)]))]
    value=tool_counts(rows+rows)
    assert value==dict(calls=2,results=1,errors=1,errors_by_tool={'Bash':1},pending=1)


def test_accepted_counter_excludes_warmup_and_cleanup(tmp_path):
    (tmp_path/'server/capture').mkdir(parents=True);(tmp_path/'bridge').mkdir()
    def dump(name,value):(tmp_path/name).write_text(json.dumps(value))
    def metrics(n):return f'vllm:generation_tokens_total {n}\nvllm:prompt_tokens_by_source_total{{source="local_compute"}} 100\n'
    dump('run.json',dict(status='window_complete',cleanup_errors=[],model_ready_at=100,model_ready_monotonic=10,ended_at=300))
    dump('window-end-metrics.json',dict(started_at=199,finished_at=201,raw_metrics=metrics(1010),stream_bytes={},bridge_bytes=0))
    (tmp_path/'metrics-before-formal.txt').write_text(metrics(10))
    (tmp_path/'resources.jsonl').write_text(json.dumps(dict(at=250,raw_metrics=metrics(5000)))+'\n')
    (tmp_path/'bridge/requests.jsonl').write_text('')
    value=analyze(tmp_path)
    assert value['accepted_output_tokens']==1000 and value['seconds']==100
    assert value['agent_tokens_per_second']==10
    assert value['metric_capture_uncertainty_seconds']==2
    dump('run.json',dict(status='needs_attention'))
    with pytest.raises(ValueError,match='did not pass'):analyze(tmp_path)
