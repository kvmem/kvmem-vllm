from pathlib import Path

import pytest

from scripts.deepswe_tool_cpu_limits import effective_cpu_count, freeze_recovery_setup


@pytest.mark.parametrize('quota,affinity,expected',[
    ('200000 100000',64,2), ('150000 100000',64,1),
    ('400000 100000',2,2), ('max 100000',8,8), ('50000 100000',64,1),
])
def test_cgroup_v2_and_affinity(tmp_path,quota,affinity,expected):
    (tmp_path/'cpu.max').write_text(quota)
    assert effective_cpu_count(tmp_path,affinity)==expected


@pytest.mark.parametrize('quota,expected',[(200000,2),(-1,64)])
def test_cgroup_v1(tmp_path,quota,expected):
    d=tmp_path/'cpu';d.mkdir()
    (d/'cpu.cfs_quota_us').write_text(str(quota))
    (d/'cpu.cfs_period_us').write_text('100000')
    assert effective_cpu_count(tmp_path,64)==expected


def test_invalid_quota_fails_closed(tmp_path):
    (tmp_path/'cpu.max').write_text('0 100000')
    with pytest.raises(ValueError):effective_cpu_count(tmp_path,64)


def test_setup_source_preserved_and_only_auto_workers_changed(tmp_path):
    source=tmp_path/'source';source.mkdir()
    old='#!/bin/bash\nexport FORCE_COLOR=3\nclaude --seed 1004\n'
    (source/'container_claude_agent.sh').write_text(old)
    (source/'run_deepswe_cell.sh').write_text('unchanged original command')
    output=tmp_path/'recovery'
    audit=freeze_recovery_setup(source,output)
    assert (source/'container_claude_agent.sh').read_text()==old
    assert (output/'run_deepswe_cell.sh').read_bytes()==(source/'run_deepswe_cell.sh').read_bytes()
    new=(output/'container_claude_agent.sh').read_text()
    assert new.count('PYTEST_XDIST_AUTO_NUM_WORKERS')==1
    assert new.endswith('claude --seed 1004\n')
    assert audit['status']=='prepared_not_launched'
    with pytest.raises(ValueError):freeze_recovery_setup(source,output)


def test_unrecognized_script_not_copied(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'container_claude_agent.sh').write_text('unknown')
    with pytest.raises(ValueError):freeze_recovery_setup(source,tmp_path/'output')
    assert not (tmp_path/'output').exists()
