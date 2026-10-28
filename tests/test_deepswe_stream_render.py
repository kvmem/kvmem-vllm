import json

import pytest

pytest.importorskip('PIL')
from scripts.render_deepswe_stream_gif import Trace, visible_lines


def test_trace_shows_only_arrived_chunks_and_actual_completion(tmp_path):
    d=tmp_path/'native';d.mkdir()
    (d/'phase-times.json').write_text(json.dumps(dict(resume_begin=100)))
    (d/'resume-0.json').write_text(json.dumps(dict(end=110.1,response=dict(
        usage=dict(completion_tokens=4096),choices=[dict(text='first\nsecond')]))))
    rows=[]
    for t,text in ((2,'first\n'),(10,'second')):
        rows.append(dict(phase_elapsed=t,data=json.dumps(dict(choices=[dict(text=text,token_ids=[7]*2048)]))))
    rows.append(dict(phase_elapsed=10.1,data='[DONE]'))
    (d/'resume-0.sse.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    trace=Trace(tmp_path,'native',0)
    assert trace.state(1.999)==(0,'',0,False)
    assert trace.state(2)==(2048,'first\n',1,False)
    assert trace.state(9.99)==(2048,'first\n',1,False)
    assert trace.state(10)==(4096,'first\nsecond',2,False)
    assert trace.state(10.1)==(4096,'first\nsecond',2,True)


def test_blank_lines_are_explicitly_counted_not_replaced_by_fabricated_output():
    assert visible_lines('actual\n\n\n\ntext')==['actual','[连续空行 × 3]','text']
    assert visible_lines('raw output\n\n\n')==['raw output','[连续空行 × 3]']
