import hashlib
import tomllib

import pytest

from scripts.terminalbench_policy import (FIXED_AGENT_TIMEOUT_SEC, configure_task_deadlines,
                                          fix_agent_deadline)

TASK = b'''schema_version = "1.1"

[task]
name = "terminal-bench/example"

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0  # official

[environment]
build_timeout_sec = 600.0
'''


def test_fixed_deadline_changes_only_the_agent_timeout():
    effective, audit = fix_agent_deadline(TASK)
    before, after = tomllib.loads(TASK.decode()), tomllib.loads(effective.decode())
    assert after['agent']['timeout_sec'] == FIXED_AGENT_TIMEOUT_SEC == 3600.0
    after['agent']['timeout_sec'] = before['agent']['timeout_sec']
    assert after == before
    assert b'# official' in effective
    assert audit['original_agent_timeout_sec'] == 900.0
    assert audit['effective_agent_timeout_sec'] == 3600.0
    assert audit['original_sha256'] == hashlib.sha256(TASK).hexdigest()
    assert audit['effective_sha256'] == hashlib.sha256(effective).hexdigest()


def test_fixed_deadline_requires_an_original_agent_deadline():
    without = TASK.replace(b'timeout_sec = 900.0  # official\n', b'')
    with pytest.raises(ValueError, match='original Agent deadline'):
        fix_agent_deadline(without)


def test_configure_fixed_policy(tmp_path):
    for name, value in (('a', '15.0'), ('b', '12000.0')):
        (tmp_path/name).mkdir()
        (tmp_path/name/'task.toml').write_bytes(TASK.replace(b'= 900.0  # official', f'= {value}'.encode()))
    policy = configure_task_deadlines(tmp_path, 'fixed')
    assert policy['agent_time_limit'] == 'fixed' and policy['agent_timeout_sec'] == 3600.0
    assert not policy['official_timing_comparable']
    for name in ('a', 'b'):
        data = tomllib.loads((tmp_path/name/'task.toml').read_text())
        assert data['agent']['timeout_sec'] == 3600.0 and data['verifier']['timeout_sec'] == 900.0
        assert policy['tasks'][name]['effective_agent_timeout_sec'] == 3600.0
