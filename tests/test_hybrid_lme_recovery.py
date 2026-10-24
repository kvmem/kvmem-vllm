import json
from pathlib import Path

import pytest

from benchmark_niah import sha
from recover_hybrid_lme import preserve_case
import run_hybrid_memory_campaign as campaign


def saved_case(tmp_path, *, completed):
    source = tmp_path/'original'; source.mkdir()
    record = dict(id='original-case',status='completed' if completed else 'error',seconds=4.)
    if completed: record.update(answer='saved answer',raw_completion='original saved answer')
    else: record['error'] = 'audit failed'
    (source/'result.json').write_text(json.dumps(record))
    (source/'response.json').write_text('immutable raw response')
    (source/'request.json').write_text('immutable input seed and sampling')
    rebuilt = dict(id='original-case',status='completed',answer='saved answer',raw_completion='original saved answer')
    return source,rebuilt


@pytest.mark.parametrize('completed',[False,True])
def test_recovery_preserves_source_responses_requests_and_successes_byte_for_byte(tmp_path,completed):
    source,rebuilt = saved_case(tmp_path,completed=completed)
    before = {p.name:p.read_bytes() for p in source.iterdir()}
    target = tmp_path/'recovered'
    proof = preserve_case(source,target,rebuilt,dict(native_replay='passed'))
    assert proof == {p.name:sha(p) for p in source.iterdir()}
    assert before == {p.name:p.read_bytes() for p in source.iterdir()}
    for name in ('response.json','request.json'):
        assert (target/name).read_bytes() == before[name]
    if completed:
        assert (target/'result.json').read_bytes() == before['result.json']
    else:
        assert (target/'source-failure-result.json').read_bytes() == before['result.json']
        assert json.loads((target/'result.json').read_text()) == rebuilt
    with pytest.raises(ValueError,match='overwrite'): preserve_case(source,target,rebuilt,{})


def test_resume_is_explicit_and_unsupported_memory_resume_is_refused():
    fixture = dict(suite='longmemeval_s',reference='/tmp/original',recovery_source='/tmp/old')
    command = campaign.client_command(Path('/tmp/recovery'),fixture)
    assert '--resume' in command and '--case-id' not in command
    fixture.update(suite='memoryagentbench',qualification=False)
    with pytest.raises(ValueError,match='Unsupported recovery'): campaign.client_command(Path('/tmp/recovery'),fixture)
