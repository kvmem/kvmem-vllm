"""KVMEM_* settings: the one accessor, where they come from, and what they may change (layering step 6).

Reading. Every KVMEM_* value the package reads goes through get() or
require(). Values are read at call time from the process environment, exactly
as os.getenv and os.environ[...] read them; a test forbids reading KVMEM_*
from os.environ anywhere else in the package.

Sources, in the order a later one wins:
1. the preset: the default-on table in defaults.py ("default-on"), or none
   ("legacy");
2. the serving profile's "settings" object, in two namespaces:
     "system":   settings that never change model output (exact optimizations,
                 diagnostics);
     "strategy": settings that may change which blocks are kept or retrieved;
3. explicit environment values given to the launcher for one run (the
   execution flags of an A/B arm), which override the profile.
Names the launcher derives from the profile's own options and flags are not
settings and are refused in the profile.

    "settings": {"preset": "default-on",
                 "system": {"KVMEM_GIL_GATE_MS": "50"},
                 "strategy": {"KVMEM_LAYERWISE_DECODE_SCOPE": "all"}}

    python -m vllm_kvmem.core.settings table             # every setting, its namespace and preset value
    python -m vllm_kvmem.core.settings effective LAUNCH  # a launch.json's effective settings and sources
"""
import os
import re
import sys
from pathlib import Path

PREFIX = 'KVMEM_'


def get(name, default=None):
    """os.getenv(name, default) for a KVMEM_* setting."""
    return os.environ.get(name, default)


def require(name):
    """os.environ[name] for a KVMEM_* setting; KeyError when it is not set."""
    return os.environ[name]


# Derived by serve.build_launch / serve.main from the profile's options and the
# --layerwise-* flags; a profile sets them through those, never directly.
LAUNCHER = frozenset((
    'KVMEM_SERVING', 'KVMEM_RUNNER_PROBE_DIR', 'KVMEM_ARCHIVE_PROBE', 'KVMEM_LIFECYCLE_TRACE',
    'KVMEM_BOUNDED_TARGET', 'KVMEM_WINDOW_METADATA', 'KVMEM_DRAFT_LOAD_COMPAT', 'KVMEM_SPLIT_CAPTURE_DMA',
    'KVMEM_NATIVE_PREFILL_COMMIT', 'KVMEM_ASYNC_CHECKPOINT_CAPTURE', 'KVMEM_REQUEST_PLAN_CONFIG',
    'KVMEM_MULTI_USER_CONFIG', 'KVMEM_REQUIRE_OWNED_SESSION', 'KVMEM_LOGICAL_KV_CAPACITY',
    'KVMEM_FULL_HISTORY_BENCHMARK', 'KVMEM_PREFILL_CHUNK_TOKENS', 'KVMEM_Q8_TENSORCORE', 'KVMEM_QSA_ADAPTER',
    'KVMEM_FULL_GRAPH', 'KVMEM_ELASTIC_ADMIN_FILE', 'KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED',
    'KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES', 'KVMEM_RETRIEVAL_DIAGNOSTICS', 'KVMEM_VALUE_RECALL_DIAGNOSTICS',
    'KVMEM_VALUE_RECALL_PROFILE', 'KVMEM_VERIFY_GUIDED_PRESSURE', 'KVMEM_VERIFY_REQUEST_SWAP',
    'KVMEM_VERIFY_SESSION_RESTORE', 'KVMEM_VERIFY_SESSION_RESTORE_DIGEST', 'KVMEM_LEGACY_DEFAULTS',
    'KVMEM_SYSTEMONE_BRANCHES', 'KVMEM_SYSTEMONE_FORK', 'KVMEM_SYSTEMONE_QUANTUM', 'KVMEM_SYSTEMONE_TOKEN_BUDGET',
    'KVMEM_SYSTEMONE_VERIFY_REFRESH', 'KVMEM_LAYERWISE_QUERY_MODE', 'KVMEM_LAYERWISE_QUERY_OPTIMIZED',
    'KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES', 'KVMEM_LAYERWISE_QUERY_DIAGNOSTICS', 'KVMEM_LAYERWISE_QUERY_NO_PROBE',
    'KVMEM_LAYERWISE_QUERY_MTP_SELECTION', 'KVMEM_LAYERWISE_ROLLING_DECODE', 'KVMEM_LAYERWISE_ORIGINAL_ROPE',
    'KVMEM_LAYERWISE_PREFILL_PRESSURE', 'KVMEM_LAYERWISE_BUFFERED_HISTORY', 'KVMEM_LAYERWISE_PAIRED_READS',
    'KVMEM_LAYERWISE_BATCHED_READS', 'KVMEM_LAYERWISE_PREPARED_INDEX', 'KVMEM_LAYERWISE_PACKED_ASSEMBLY',
    'KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY', 'KVMEM_LAYERWISE_PRESSURE_FAST_METADATA',
    'KVMEM_LAYERWISE_PRESSURE_PREPARED_READS', 'KVMEM_LAYERWISE_PRESSURE_POST_FORWARD_REAP',
    'KVMEM_LAYERWISE_PRESSURE_SUBMIT_ON_PREPARE', 'KVMEM_LAYERWISE_PRESSURE_REUSE_LAYER_VIEWS',
    'KVMEM_LAYERWISE_PRESSURE_RESIDENT_VALUES', 'KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION',
    'KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT',
))

# Known to change which blocks are kept, retrieved or retained.
STRATEGY = frozenset((
    'KVMEM_LAYERWISE_DECODE_SCOPE', 'KVMEM_LAYERWISE_DECODE_INTERVAL', 'KVMEM_LAYERWISE_DECODE_PREFILL',
    'KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT', 'KVMEM_RECALL_METHOD', 'KVMEM_PREFILL_PRESSURE_METHOD',
    'KVMEM_DECODE_PRESSURE_METHOD', 'KVMEM_PRESSURE_QUERY_WINDOW', 'KVMEM_VALUE_RECALL_REFERENCE',
    'KVMEM_LAYERWISE_HISTORICAL_GC', 'KVMEM_SESSION_HISTORY_LRU_EVICTION', 'KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS',
))

# Diagnostics observe and never steer: timing, traces, profiles, verification.
_DIAGNOSTIC = re.compile(r'(TRACE|PROFILE|DIAG|TIMING|MARKS|VERIFY|CHECK$|THREAD_NAMES)')


def namespace(name):
    """'launcher', 'strategy' or 'system' for a known setting; 'unverified' when no run showed it exact."""
    from .defaults import DEFAULTS
    if name in LAUNCHER:
        return 'launcher'
    if name in STRATEGY:
        return 'strategy'
    if name in DEFAULTS or _DIAGNOSTIC.search(name):
        # The default-on table holds only flags measured exact (defaults.py),
        # apart from the rolling-decode scoring flags listed in STRATEGY.
        return 'system'
    return 'unverified'


_KNOWN = None


def known():
    """Every KVMEM_* name the package reads or the preset sets."""
    global _KNOWN
    if _KNOWN is None:
        from .defaults import DEFAULTS
        found = set(DEFAULTS) | LAUNCHER | STRATEGY
        pattern = re.compile(r"_settings\.(?:get|require)\(\s*['\"](KVMEM_[A-Z0-9_]+)['\"]")
        for path in Path(__file__).resolve().parents[1].rglob('*.py'):
            found.update(pattern.findall(path.read_text()))
        _KNOWN = frozenset(found)
    return _KNOWN


def _value(name, value):
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, (int, float, str)):
        return str(value)
    raise ValueError('settings: %s must be a string, number or boolean' % name)


def profile_environment(spec):
    """The environment a profile's "settings" object contributes, validated."""
    if spec is None:
        return {}
    if not isinstance(spec, dict) or set(spec) - {'preset', 'system', 'strategy'}:
        raise ValueError('settings must be an object with preset, system and strategy')
    env = {}
    preset = spec.get('preset', 'default-on')
    if preset not in ('default-on', 'legacy'):
        raise ValueError('settings.preset must be default-on or legacy')
    if preset == 'legacy':
        env['KVMEM_LEGACY_DEFAULTS'] = '1'
    for group in ('system', 'strategy'):
        values = spec.get(group, {})
        if not isinstance(values, dict):
            raise ValueError('settings.%s must be an object' % group)
        for name, value in values.items():
            if not isinstance(name, str) or not name.startswith(PREFIX) or name not in known():
                raise ValueError('settings.%s: unknown setting %r' % (group, name))
            kind = namespace(name)
            if kind == 'launcher':
                raise ValueError('settings.%s: %s is derived from the profile options and launch flags'
                                 % (group, name))
            if group == 'system' and kind != 'system':
                raise ValueError('settings.system: %s may change model output (%s); put it under strategy'
                                 % (name, kind))
            if group == 'strategy' and kind == 'system':
                raise ValueError('settings.strategy: %s never changes model output; put it under system' % name)
            env[name] = _value(name, value)
    return env


def effective(environment):
    """{name: (value, source)} for every known setting a launch environment leaves in effect."""
    from .defaults import DEFAULTS
    out = {}
    legacy = environment.get('KVMEM_LEGACY_DEFAULTS', '0') == '1'
    for name in sorted(known()):
        if name in environment:
            out[name] = (environment[name], 'launch')
        elif name in DEFAULTS and not legacy:
            out[name] = (DEFAULTS[name], 'preset')
    return out


def table():
    from .defaults import DEFAULTS
    lines = ['| Setting | Namespace | Preset value |', '| --- | --- | --- |']
    for name in sorted(known()):
        lines.append('| `%s` | %s | %s |' % (name, namespace(name),
                                              '`%s`' % DEFAULTS[name] if name in DEFAULTS else '-'))
    return '\n'.join(lines)


def main(argv=None):
    import json
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['table']:
        print(table())
        return 0
    if len(argv) == 2 and argv[0] == 'effective':
        launch = json.loads(Path(argv[1]).read_text())
        print(json.dumps({name: dict(value=v, source=s, namespace=namespace(name))
                          for name, (v, s) in effective(launch['environment']).items()}, indent=1))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
