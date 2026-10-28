"""One fixed eight-request full-input qualification before the 130-case campaign.

Uses exactly the first admission wave of that campaign, original seeds and
sampling, and the same CPU/native FPS4 + atomic halo1 execution settings.
No shortened input/output, quality-dependent retries or prior-answer changes.
"""
import argparse
from pathlib import Path

import run_ruler_fps_cpu_campaign as campaign

ROOT=Path('/data/chaidi/kvmem-vllm-experiments/query-fps-native-cpu-live-20261023-r1')


def prepare(root,source):
    return campaign.prepare(root,source,qualification=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--source',type=Path,default=campaign.SOURCE)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root.resolve(),args.source.resolve())
    else:campaign.run(args.root.resolve())
