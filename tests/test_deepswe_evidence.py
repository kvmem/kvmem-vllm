import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.deepswe_evidence import SCAN


def test_scan_includes_private_files_but_does_not_follow_links(tmp_path):
    root=tmp_path/'evidence';root.mkdir();private=root/'client';private.mkdir(mode=0o700)
    file=private/'private.json';file.write_text('private test fixture');file.chmod(0o600)
    external=tmp_path/'external';external.mkdir();(external/'outside').write_text('not evidence')
    (root/'external-link').symlink_to(external,target_is_directory=True)
    (root/'evidence-hashes.json').write_text('old hash file is excluded')
    code=SCAN.replace("Path('/evidence')",f'Path({str(root)!r})')
    value=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
    assert value['files']=={'client/private.json':hashlib.sha256(file.read_bytes()).hexdigest()}
    assert value['symlink_targets_sha256']=={'external-link':hashlib.sha256(str(external).encode()).hexdigest()}
