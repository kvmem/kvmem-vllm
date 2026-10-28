"""Hash a finished observation through a read-only mount, including root-owned client files."""
import json
from pathlib import Path
import subprocess

EXCLUDED=('evidence-hashes.json','evidence-hash-scope.json')
SCAN=r'''
import hashlib,json,os,stat
from pathlib import Path
root=Path('/evidence');files={};links={}
def onerror(error):raise error
for base,dirs,names in os.walk(root,followlinks=False,onerror=onerror):
 for name in dirs+names:
  p=Path(base)/name;rel=str(p.relative_to(root));mode=p.lstat().st_mode
  if stat.S_ISLNK(mode):links[rel]=hashlib.sha256(os.readlink(p).encode()).hexdigest()
  elif stat.S_ISREG(mode) and rel not in ('evidence-hashes.json','evidence-hash-scope.json'):
   h=hashlib.sha256()
   with p.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
   files[rel]=h.hexdigest()
print(json.dumps(dict(files=files,symlink_targets_sha256=links)))
'''


def hash_finished_window(directory):
    directory=Path(directory).resolve()
    state=json.loads((directory/'run.json').read_text())
    snapshot=json.loads((directory/'window-workspaces.json').read_text())
    if state['status']!='window_complete' or state.get('cleanup_errors') or snapshot['status']!='preserved':
        raise ValueError('Only a completed, cleaned and snapshotted observation may be hashed')
    manifest=json.loads((directory/'manifest.json').read_text())
    task=manifest['tasks'][0];image=task['images'][task['agent_image']]
    result=subprocess.run(['docker','run','--rm','--network','none','--read-only',
        '--cap-drop','ALL','--cap-add','DAC_READ_SEARCH','--security-opt','no-new-privileges','--pids-limit','32','--memory','1g',
        '--user','0','--mount',f'type=bind,source={directory},target=/evidence,readonly',
        '--entrypoint','python3',image,'-c',SCAN],capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError('Read-only evidence hashing failed: '+result.stderr)
    value=json.loads(result.stdout)
    if not value['files'] or 'run.json' not in value['files']:
        raise ValueError('Incomplete evidence scan')
    return value


def write_evidence(directory):
    from scripts import run_agent_handoff_queue as q
    directory=Path(directory)
    if (directory/'evidence-hashes.json').exists():raise ValueError('Evidence already recorded')
    value=hash_finished_window(directory)
    q.save(directory/'evidence-hashes.json',value['files'])
    q.save(directory/'evidence-hash-scope.json',dict(
        scope='All regular files under this completed observation, including private container-owned client files. '
              'Read-only mount, no network, no host permission changes. Symbolic links are hashed as targets and not followed.',
        excluded=list(EXCLUDED),symlink_targets_sha256=value['symlink_targets_sha256']))


def verify_evidence(directory):
    from scripts import run_agent_handoff_queue as q
    directory=Path(directory);value=hash_finished_window(directory)
    if value['files']!=q.read(directory/'evidence-hashes.json'):
        raise ValueError('Preserved observation evidence changed')
    if value['symlink_targets_sha256']!=q.read(directory/'evidence-hash-scope.json')['symlink_targets_sha256']:
        raise ValueError('Preserved observation links changed')
