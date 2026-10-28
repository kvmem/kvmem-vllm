"""Match pytest's automatic worker count to an existing container CPU quota.

This does not change the command, timeout, tests, CPU quota, or memory limit.
Use only in a newly frozen recovery setup; never patch a running experiment.
"""
import argparse
import math
import os
from pathlib import Path


def effective_cpu_count(cgroup=Path('/sys/fs/cgroup'), affinity=None):
    """Round fractional quotas down so automatic workers cannot exceed them."""
    limits = [len(os.sched_getaffinity(0)) if affinity is None else affinity]
    if limits[0] < 1:
        raise ValueError('Empty CPU affinity')
    v2 = cgroup / 'cpu.max'
    if v2.exists():
        quota, period = v2.read_text().split()
        if quota != 'max':
            if int(quota) <= 0 or int(period) <= 0:
                raise ValueError('Invalid CPU quota')
            limits.append(max(1, math.floor(int(quota) / int(period))))
    else:
        for base in (cgroup / 'cpu', cgroup):
            q, p = base / 'cpu.cfs_quota_us', base / 'cpu.cfs_period_us'
            if not q.exists():
                continue
            quota, period = int(q.read_text()), int(p.read_text())
            if quota != -1:
                if quota <= 0 or period <= 0:
                    raise ValueError('Invalid CPU quota')
                limits.append(max(1, math.floor(quota / period)))
            break
    return min(limits)


def freeze_recovery_setup(source, output):
    """Copy an original setup, changing only pytest's implicit auto fan-out."""
    import hashlib
    import shutil
    import json
    import inspect
    if output.exists():
        raise ValueError('Recovery setup already exists')
    source = source.resolve()
    script = source / 'container_claude_agent.sh'
    text = script.read_text()
    anchor = 'export FORCE_COLOR=3\n'
    if text.count(anchor) != 1:
        raise ValueError('Unrecognized frozen container setup')
    shutil.copytree(source, output)
    helper = output / 'deepswe_tool_cpu_limits.py'
    shutil.copy2(Path(__file__), helper)
    # Only /runner.sh is mounted by the original supervisor. Embed the small
    # stdlib-only reader instead of depending on a new mount or installation.
    addition = ('\n# Keep automatic pytest fan-out within the existing container CPU quota.\n'
                "deepswe_pytest_workers=$(python3 - <<'DEEPSWE_CPU_QUOTA'\n"
                'import math, os\nfrom pathlib import Path\n'
                + inspect.getsource(effective_cpu_count)
                + '\nprint(effective_cpu_count())\nDEEPSWE_CPU_QUOTA\n)\n'
                'export PYTEST_XDIST_AUTO_NUM_WORKERS=$deepswe_pytest_workers\n')
    (output / script.name).write_text(text.replace(anchor, anchor + addition))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = dict(status='prepared_not_launched', source=str(source),
                 source_script_sha256=digest(script), script_sha256=digest(output/script.name),
                 helper_sha256=digest(helper),
                 change='Only pytest automatic worker count follows existing cgroup quota and affinity.',
                 unchanged='Task prompts, actual tool commands, seeds, model parameters, CPU/memory limits, and timeouts.')
    (output/'cpu-auto-limit-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-from',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.freeze_from:
        if args.output is None:parser.error('--output is required')
        import json
        print(json.dumps(freeze_recovery_setup(args.freeze_from,args.output)))
    else:
        print(effective_cpu_count())
