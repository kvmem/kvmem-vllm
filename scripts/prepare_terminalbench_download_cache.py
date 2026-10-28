"""Fetch exact public dependencies from GitHub's official asset API, proxy-free.

Only published release archives are cached. No task data, answers, installed
Python environments, package indexes or hidden verifier code are included.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from urllib.parse import urlparse, unquote

import httpx


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    files={}
    with httpx.Client(trust_env=False, follow_redirects=True, timeout=60) as client:
        def release(repo, version):
            response=client.get(f'https://api.github.com/repos/{repo}/releases/tags/{version}')
            response.raise_for_status()
            return response.json()

        def download(release, filename, relative):
            asset=next(a for a in release['assets'] if a['name']==filename)
            digest=asset.get('digest') or ''
            if not digest.startswith('sha256:'):
                raise ValueError('Official release has no SHA256 for '+filename)
            path=root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            with client.stream('GET',asset['url']+'?download=1',
                               headers={'Accept':'application/octet-stream'}) as response:
                response.raise_for_status()
                with path.open('xb') as f:
                    for chunk in response.iter_bytes(): f.write(chunk)
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if 'sha256:'+actual!=digest or path.stat().st_size!=asset['size']:
                raise ValueError('Official dependency digest/size mismatch')
            files[relative]=dict(sha256=actual,bytes=path.stat().st_size,
                official_digest=digest,official_asset_url=asset['url'],
                upstream_download_url=asset['browser_download_url'])
            print('Verified',filename,flush=True)
            return path

        uvrel=release('astral-sh/uv','0.9.5')
        uvname='uv-x86_64-unknown-linux-gnu.tar.gz'
        uvarchive=download(uvrel,uvname,'astral-sh/uv/releases/download/0.9.5/'+uvname)
        # Extract outside the read-only archive tree used by benchmark tasks.
        tool_dir=root.parent/(root.name+'-catalog-tools')
        tool_dir.mkdir(exist_ok=False)
        with tarfile.open(uvarchive) as f:f.extractall(tool_dir,filter='data')
        binary=tool_dir/'uv-x86_64-unknown-linux-gnu/uv'
        catalog=json.loads(subprocess.check_output([str(binary.resolve()),'--cache-dir',
            str(tool_dir/'cache'),'python','list','--only-downloads','--output-format','json'],text=True))
        rels={}
        for minor in (11,12,13):
            item=next(p for p in catalog if p['implementation']=='cpython' and
                p['variant']=='default' and p['version_parts']['major']==3 and
                p['version_parts']['minor']==minor)
            relative=unquote(urlparse(item['url']).path).lstrip('/')
            version=relative.split('/')[-2]
            if version not in rels:rels[version]=release('astral-sh/python-build-standalone',version)
            download(rels[version],relative.split('/')[-1],relative)
    manifest=dict(passed=True,proxy_disabled=True,files=files,
        scope='Official uv 0.9.5 and its default CPython 3.11/3.12/3.13 Linux x86_64 GNU downloads',
        container_path='/opt/kvmem-download-cache',read_only=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    prepare(p.parse_args().root.resolve())
