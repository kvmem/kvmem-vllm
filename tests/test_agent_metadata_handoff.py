from pathlib import Path

import pytest

from scripts.benchmark_agent_metadata_handoff import fixed_input, verify_change


def test_fixed_input_never_depends_on_generated_output():
    item = dict(tokens=[1, 2, 3], output_token=123)
    assert fixed_input(item, 0) == [1, 2, 3]
    assert fixed_input(item, 2) == [1, 2, 3] + [123]*130
    fixed_input(item, 0).append(9)
    assert item['tokens'] == [1, 2, 3]


def test_only_merge_body_may_change(tmp_path):
    a, b = [tmp_path/name for name in ('before.py', 'after.py')]
    a.write_text('CAP = 32\ndef _merge_spans(a):\n return a\ndef score(x):\n return x\n')
    b.write_text('CAP = 32\ndef _merge_spans(a):\n return sorted(a)\ndef score(x):\n return x\n')
    verify_change(a, b)
    b.write_text(b.read_text().replace('CAP = 32', 'CAP = 64'))
    with pytest.raises(ValueError, match='outside'):
        verify_change(a, b)


def test_removed_helper_is_rejected(tmp_path):
    p = tmp_path/'broken.py'
    p.write_text('x = 1\n')
    with pytest.raises(ValueError, match='unique'):
        verify_change(p, p)
