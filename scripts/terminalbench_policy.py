"""Explicit, audited Agent timing policy for local Terminal-Bench runs."""
from copy import deepcopy
import hashlib
import math
import re
import tomllib


RELAXED_AGENT_MULTIPLIER = 2.0
# One wall-clock Agent deadline for every task (user 11-18): bounds long-running tasks.
FIXED_AGENT_TIMEOUT_SEC = 3600.0


def relaxed_agent_timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Relaxed timing requires a finite positive original Agent deadline')
    result = value * RELAXED_AGENT_MULTIPLIER
    if not math.isfinite(result):
        raise ValueError('Relaxed Agent deadline overflow')
    return result


def relax_agent_deadline(data):
    """Multiply only the original Agent deadline; never scale verifier/setup."""
    # Reuse the strict single-step/layout validation before making any change.
    _, audit = disable_agent_deadline(data)
    return set_agent_deadline(data, relaxed_agent_timeout(audit['original_agent_timeout_sec']))


def fix_agent_deadline(data):
    """Replace only the Agent deadline with FIXED_AGENT_TIMEOUT_SEC; never verifier/setup."""
    return set_agent_deadline(data, FIXED_AGENT_TIMEOUT_SEC)


def set_agent_deadline(data, value):
    """Set only [agent].timeout_sec to ``value``, preserving every other byte."""
    _, audit = disable_agent_deadline(data)
    if audit['original_agent_timeout_sec'] is None:
        raise ValueError('An Agent deadline change requires an original Agent deadline')
    original = tomllib.loads(data.decode('utf-8'))
    expected = deepcopy(original)
    expected['agent']['timeout_sec'] = value
    lines, in_agent = [], False
    for line in data.decode('utf-8').splitlines(keepends=True):
        if line.lstrip().startswith('['):
            in_agent = bool(re.fullmatch(r'\s*\[agent\]\s*(?:#.*)?', line.strip()))
        if in_agent and re.match(r'^\s*timeout_sec\s*=', line):
            # Keep indentation, inline comments and line endings.
            line = re.sub(r'(^\s*timeout_sec\s*=\s*)[^#\r\n]*',
                          lambda m: m[1] + str(value) + ' ', line)
        lines.append(line)
    result = ''.join(lines).encode('utf-8')
    if tomllib.loads(result.decode('utf-8')) != expected:
        raise ValueError('Could not change only the Agent deadline')
    return result, dict(audit, effective_agent_timeout_sec=value,
                       effective_sha256=hashlib.sha256(result).hexdigest())


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
    if mode not in ('none', 'official', 'relaxed', 'fixed'):
        raise ValueError('Unknown Agent time limit policy')
    rows, updates = {}, []
    for path in sorted(tasks_dir.glob('*/task.toml')):
        original = path.read_bytes()
        if mode == 'none':
            effective, audit = disable_agent_deadline(original)
        elif mode == 'relaxed':
            effective, audit = relax_agent_deadline(original)
        elif mode == 'fixed':
            effective, audit = fix_agent_deadline(original)
        else:
            value = tomllib.loads(original.decode('utf-8')).get('agent', {}).get('timeout_sec')
            digest = hashlib.sha256(original).hexdigest()
            effective = original
            audit = dict(original_agent_timeout_sec=value, effective_agent_timeout_sec=value,
                         original_sha256=digest, effective_sha256=digest,
                         verifier_unchanged=True, environment_unchanged=True)
        updates.append((path, effective))
        rows[path.parent.name] = audit
    # Validate the whole input before modifying the copied dataset.
    for path, effective in updates:
        path.write_bytes(effective)
    policy = dict(agent_time_limit=mode, official_timing_comparable=(mode == 'official'),
                  scope='Agent execution only; verifier/setup/tool timeouts retained', tasks=rows)
    if mode == 'fixed':
        policy.update(agent_timeout_sec=FIXED_AGENT_TIMEOUT_SEC,
                      clock='Harbor Agent wall time including model and tool waits')
    if mode == 'relaxed':
        policy.update(agent_multiplier=RELAXED_AGENT_MULTIPLIER,
                      clock='Agent elapsed time including model/tool waits, excluding recorded pauses')
    return policy
