"""Verify exported GIF timing, provenance and composited frames against measured traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(root,output):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.render_deepswe_loop_gif import Renderer,W,H
    provenance=json.loads((output/'gif-provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items():assert sha(root/name)==digest,name
    path=output/'deepswe-throughput-vram-loop.gif'
    assert sha(path)==provenance['gif_sha256']
    assert sha(output/'render.py')==provenance['renderer_sha256']
    renderer=Renderer(root,provenance['speed'],provenance.get('layout','grid'))
    assert renderer.gain==provenance['measured_resume_tps_ratio']
    indices={0,1,len(provenance['frame_real_seconds'])-1}
    indices.update(int((len(provenance['frame_real_seconds'])-1)*fraction) for fraction in (.2,.4,.6,.8))
    # Inspect both sides of the first KVMem repeat, when it falls in this GIF.
    repeat=renderer.durations['kvmem']/provenance['speed']*provenance['fps']
    indices.update(i for i in (int(repeat)-1,int(repeat)+1) if 0<=i<len(provenance['frame_real_seconds']))
    checked=[];duration=0
    with Image.open(path) as gif:
        assert gif.info['loop']==0 and gif.size==(W,H)
        assert gif.n_frames==len(provenance['frame_real_seconds'])
        for i in range(gif.n_frames):
            gif.seek(i);frame_duration=gif.info['duration'];duration+=frame_duration
            assert frame_duration==provenance['frame_duration_ms'][i]
            if i not in indices:continue
            at=provenance['frame_real_seconds'][i]
            expected=np.array(renderer.frame(at),dtype=np.int16)
            actual=np.array(gif.convert('RGB'),dtype=np.int16)
            error=float(np.abs(expected-actual).mean())
            assert error<1.,(i,error)
            checked.append(dict(frame=i,real_seconds=at,mean_rgb_error=error))
    result=dict(passed=True,frames=len(provenance['frame_real_seconds']),duration_seconds=duration/1000,
        infinite_loop=True,source_hashes_verified=True,checked_frames=checked)
    (output/'gif-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.root.resolve(),a.output.resolve())
