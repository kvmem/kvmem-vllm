"""The default-on optimization stack (vllm_kvmem/defaults.py)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_kvmem.core import defaults

ROOT = Path(__file__).resolve().parents[1]
NOT_DEFAULTED = ('KVMEM_PUBLICATION_TRACE', 'KVMEM_QUERY_STAGE_PROFILE', 'KVMEM_HD_ALLOW_PTRACE',
                 'KVMEM_LAYERWISE_HISTORICAL_GC', 'KVMEM_SESSION_IDLE_LRU_EVICTION',
                 'KVMEM_LEGACY_DEFAULTS')


def test_apply_sets_every_missing_default_and_keeps_explicit_values():
    env = {'KVMEM_GIL_GATE_MS': '0', 'KVMEM_REQUEST_PLAN_CACHE': '0', 'OTHER': 'x'}
    applied = defaults.apply(env)
    assert set(applied) == set(defaults.DEFAULTS) - {'KVMEM_GIL_GATE_MS', 'KVMEM_REQUEST_PLAN_CACHE'}
    assert env['KVMEM_GIL_GATE_MS'] == '0' and env['KVMEM_REQUEST_PLAN_CACHE'] == '0'
    assert all(env[k] == v for k, v in defaults.DEFAULTS.items() if k in applied)
    assert defaults.apply(env) == ()                                    # idempotent


def test_legacy_switch_skips_the_table_and_is_validated():
    env = {'KVMEM_LEGACY_DEFAULTS': '1'}
    assert defaults.apply(env) == () and env == {'KVMEM_LEGACY_DEFAULTS': '1'}
    with pytest.raises(ValueError, match='KVMEM_LEGACY_DEFAULTS'):
        defaults.apply({'KVMEM_LEGACY_DEFAULTS': 'yes'})


def test_table_excludes_diagnostics_unvalidated_and_policy_flags():
    assert not set(NOT_DEFAULTED) & set(defaults.DEFAULTS)
    assert all(k.startswith('KVMEM_') and type(v) is str and v for k, v in defaults.DEFAULTS.items())


def test_every_default_is_read_by_the_code():
    sources = ''.join(p.read_text(errors='ignore') for p in (ROOT / 'src' / 'vllm_kvmem').rglob('*.py')
                      if p.name != 'defaults.py')
    sources += ''.join(p.read_text(errors='ignore') for p in (ROOT / 'csrc').rglob('*.*'))
    missing = [k for k in defaults.DEFAULTS if k not in sources]
    assert not missing, missing


@pytest.mark.parametrize('legacy', ['0', '1'])
def test_package_import_applies_the_table(legacy):
    env = {k: v for k, v in os.environ.items() if not k.startswith('KVMEM_')}
    env.update(KVMEM_LEGACY_DEFAULTS=legacy, PYTHONPATH=str(ROOT / 'src'), KVMEM_GIL_GATE_MS='7')
    out = subprocess.run([sys.executable, '-c',
                          'import os, json, vllm_kvmem; '
                          'print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("KVMEM_")}))'],
                         env=env, capture_output=True, text=True, check=True).stdout
    import json
    seen = json.loads(out)
    assert seen['KVMEM_GIL_GATE_MS'] == '7'                             # explicit value wins
    if legacy == '1':
        assert set(seen) == {'KVMEM_LEGACY_DEFAULTS', 'KVMEM_GIL_GATE_MS'}
    else:
        assert all(seen[k] == v for k, v in defaults.DEFAULTS.items() if k != 'KVMEM_GIL_GATE_MS')


def test_summary_reports_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv('KVMEM_LEGACY_DEFAULTS', '0')
    monkeypatch.setattr(defaults, 'APPLIED', tuple(k for k in defaults.DEFAULTS if k != 'KVMEM_GIL_GATE_MS'))
    for k, v in defaults.DEFAULTS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '0')
    line = defaults.summary()
    assert line.startswith(f'KVMem default-on optimizations: {len(defaults.DEFAULTS) - 1}/{len(defaults.DEFAULTS)}')
    assert line.endswith('explicitly overridden: KVMEM_GIL_GATE_MS')
    monkeypatch.setenv('KVMEM_LEGACY_DEFAULTS', '1')
    assert 'off' in defaults.summary()


def test_non_thinking_rolling_decode_scoring_is_the_default_policy():
    env = {}
    defaults.apply(env)
    assert (env['KVMEM_LAYERWISE_DECODE_SCOPE'], env['KVMEM_LAYERWISE_DECODE_INTERVAL'],
            env['KVMEM_LAYERWISE_DECODE_PREFILL']) == ('non-thinking', '4', '1')
