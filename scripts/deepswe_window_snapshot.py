"""Preserve only this observation window's unfinished container files.

No model requests, resumption or result retries. The caller keeps frozen
supervisors/containers on failure, rather than destroying unpreserved work.
"""
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time

import psutil


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def snapshot(out, children):
    out = Path(out).resolve()
    audit = out/'window-workspaces.json'
    if audit.exists():
        raise ValueError('Snapshot already attempted; retain original audit')
    record = dict(status='started', at=time.time(), supervisors=[], containers=[])
    save(audit, record)
    try:
        # A cell wrapper's EXIT trap removes its --rm container. Hold only
        # wrappers whose exact script/cell paths belong to this new window.
        for child in list(children):
            if child.poll() is not None:
                continue
            try:
                process = psutil.Process(child.pid)
                args = process.cmdline()
                if len(args) < 4 or args[:2] != ['bash', str(out/'setup/run_deepswe_cell.sh')]:
                    continue
                cell = Path(args[3])
                if cell.parent != out/'cases' or cell.name != args[2]:
                    raise ValueError('Cell wrapper ownership mismatch')
                row = dict(pid=process.pid, created=process.create_time(), argv=args)
                process.send_signal(signal.SIGSTOP)
                record['supervisors'].append(row)
                save(audit, record)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        for cidfile in sorted((out/'cases').glob('*/*.container.cid')):
            role = cidfile.name.split('.')[0]
            if role not in ('agent', 'verifier'):
                continue
            cid = cidfile.read_text().strip()
            result = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, timeout=20)
            if result.returncode:
                # A normally finished cell already extracted its patch/result.
                if not (cidfile.parent/'cell_summary.json').exists():
                    raise RuntimeError('Unfinished container missing: '+str(cidfile))
                continue
            info = json.loads(result.stdout)[0]
            owner = hashlib.sha256(str(cidfile.parent).encode()).hexdigest()[:16]
            labels = info['Config'].get('Labels') or {}
            if labels.get('com.qw3.deepswe.run-owner') != owner or labels.get('com.qw3.deepswe.role') != role:
                raise ValueError('Foreign container left untouched: '+cid)
            row = dict(cid=cid, task=cidfile.parent.name, role=role, owner=owner)
            record['containers'].append(row)
            save(audit, record)
            if info['State']['Running'] and not info['State']['Paused']:
                subprocess.run(['docker', 'pause', cid], check=True, capture_output=True, timeout=20)
            row['paused_or_stopped'] = True
            save(audit, record)
        # Pause all owned tasks first, so committing one does not let others
        # continue generating unmeasured answers while copying filesystem data.
        for row in record['containers']:
            tag = 'kvmem-window:'+row['owner']+'-'+row['role']
            row['commit_stdout'] = subprocess.check_output(
                ['docker','commit','--pause=false',row['cid'],tag], text=True, timeout=180)
            # Docker versions can print deprecation warnings to stdout. Bind
            # the tag to its inspected immutable ID, never the whole stdout.
            row['image'] = subprocess.check_output(
                ['docker','image','inspect','--format','{{.Id}}',tag],text=True,timeout=20).strip()
            if not re.fullmatch(r'sha256:[0-9a-f]{64}',row['image']):
                raise ValueError('Invalid committed image identity')
            row['tag'] = tag
            save(audit, record)
        record.update(status='preserved', finished_at=time.time(),
            scope='Container filesystem snapshot, not a process or model KV checkpoint. Bind-mounted evidence stays in the experiment directory.')
        save(audit, record)
        return record
    except BaseException as error:
        record.update(status='failed_preserved_in_place', error=repr(error), finished_at=time.time())
        save(audit, record)
        raise


def release_supervisors(record):
    """After window_done is set, allow normal cleanup of snapshotted tasks."""
    if record['status'] != 'preserved':
        raise ValueError('Never release wrappers after an incomplete snapshot')
    for row in record['supervisors']:
        try:
            p = psutil.Process(row['pid'])
            if abs(p.create_time()-row['created']) > .01 or p.cmdline() != row['argv']:
                raise ValueError('Supervisor identity changed')
            p.send_signal(signal.SIGCONT)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
