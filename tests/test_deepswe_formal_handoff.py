from copy import deepcopy
import json

import pytest

from scripts import continue_deepswe_formal as handoff
from test_deepswe_lru_10m import workload


def pair():
    _, before = workload()
    before['differences'] = ['Original config']
    after = deepcopy(before)
    after.pop('long_gc_qualification')
    after['preflight_reuse'] = 'preflight-reuse-proof.json'
    after['differences'].append('User stopped the incomplete synthetic10M trial and requested direct formal tasks; passed preflights reused, not rerun.')
    return before, after


def test_only_user_requested_qualification_change():
    before, after = pair()
    handoff.validate_workload(before, after)
    for key in before.keys()-{'long_gc_qualification', 'differences'}:
        invalid = deepcopy(after); invalid[key] = 'different'
        with pytest.raises(ValueError):
            handoff.validate_workload(before, invalid)
    assert before['long_gc_qualification']['target_tokens'] == 9999968


def test_cannot_overwrite_prior_attempt(tmp_path):
    (tmp_path/'answer.json').write_text('original')
    with pytest.raises(ValueError, match='never overwrite'):
        handoff.prepare(tmp_path/'source', tmp_path)
    assert (tmp_path/'answer.json').read_text() == 'original'


def test_fresh_engine_cannot_reuse_public_forward_history(tmp_path):
    assert handoff.fresh_engine_archive(tmp_path) == {}
    for phase in ('submitted', 'completed'):
        (tmp_path/'public_work.json').write_text(json.dumps({phase: {'public_batches': 1}}))
        with pytest.raises(ValueError, match='earlier public requests'):
            handoff.fresh_engine_archive(tmp_path)


def test_existing_empty_archive_checks_checkpoint_release(tmp_path, monkeypatch):
    from scripts import benchmark_deepswe_first16 as runner
    (tmp_path/'archive.json').write_text('{}')
    called=[]
    def wait(path):
        called.append(path);return {'session_checkpoints':0}
    monkeypatch.setattr(runner, 'wait_preflight_release', wait)
    assert handoff.fresh_engine_archive(tmp_path) == {'session_checkpoints':0}
    assert called == [tmp_path]
