"""Explicit, audited Agent timing policy for local Terminal-Bench runs."""
from copy import deepcopy
import hashlib
import re
import tomllib


def disable_agent_deadline(data):
    """Remove only [agent].timeout_sec; Harbor resolves an absent value to None.

    Preserve all other bytes, and reject unfamiliar TOML layouts rather than
    silently changing another phase. This helper is for the single-step TB2.1
    tasks; multi-step tasks need a separate timing policy.
    """
    original = tomllib.loads(data.decode('utf-8'))
    if original.get('steps'):
        raise ValueError('Multi-step Agent deadlines require explicit handling')
    expected = deepcopy(original)
    before = expected.get('agent', {}).pop('timeout_sec', None)
    lines = []
    in_agent = False
    removed = 0
    for line in data.decode('utf-8').splitlines(keepends=True):
        if line.lstrip().startswith('['):
            in_agent = bool(re.fullmatch(r'\s*\[agent\]\s*(?:#.*)?', line.strip()))
        if in_agent and re.match(r'^\s*timeout_sec\s*=', line):
            removed += 1
            continue
        lines.append(line)
    result = ''.join(lines).encode('utf-8')
    if tomllib.loads(result.decode('utf-8')) != expected:
        raise ValueError('Could not remove only the Agent deadline')
    if removed != int('timeout_sec' in original.get('agent', {})):
        raise ValueError('Ambiguous Agent deadline layout')
    return result, dict(original_agent_timeout_sec=before,
                       effective_agent_timeout_sec=None,
                       original_sha256=hashlib.sha256(data).hexdigest(),
                       effective_sha256=hashlib.sha256(result).hexdigest(),
                       verifier_unchanged=True, environment_unchanged=True)


def configure_task_deadlines(tasks_dir, mode):
    if mode not in ('none', 'official'):
        raise ValueError('Unknown Agent time limit policy')
    rows = {}
    for path in sorted(tasks_dir.glob('*/task.toml')):
        original = path.read_bytes()
        if mode == 'none':
            effective, audit = disable_agent_deadline(original)
            path.write_bytes(effective)
        else:
            value = tomllib.loads(original.decode('utf-8')).get('agent', {}).get('timeout_sec')
            digest = hashlib.sha256(original).hexdigest()
            audit = dict(original_agent_timeout_sec=value, effective_agent_timeout_sec=value,
                         original_sha256=digest, effective_sha256=digest,
                         verifier_unchanged=True, environment_unchanged=True)
        rows[path.parent.name] = audit
    return dict(agent_time_limit=mode, official_timing_comparable=(mode == 'official'),
                scope='Agent execution only; verifier/setup/tool timeouts retained', tasks=rows)
