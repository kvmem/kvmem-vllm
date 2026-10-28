"""Render measured SSE arrivals on one shared clock; no interpolated output."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

W,H=1800,1240
BG='#090e16';PANEL='#101923';BORDER='#263647';WHITE='#e6eef6';MUTED='#90a4b8'
AMBER='#ffbc6e';GREEN='#60e3bd';BLUE='#8fbbff'
CJK='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
MONO='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONTS={}
def font(size,mono=False):
    key=(size,mono)
    if key not in FONTS:FONTS[key]=ImageFont.truetype(MONO if mono else CJK,size)
    return FONTS[key]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def txt(draw,xy,text,size=18,fill=WHITE,mono=False,anchor=None):
    draw.text(xy,str(text),font=font(size,mono),fill=fill,anchor=anchor)

class Trace:
    def __init__(self,root,backend,index):
        self.record=read(root/backend/f'resume-{index}.json')
        phase=read(root/backend/'phase-times.json')['resume_begin']
        self.end=self.record['end']-phase
        self.times=[];self.counts=[];self.parts=[];self.chunk_counts=[]
        count=0;chunks=0
        with (root/backend/f'resume-{index}.sse.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['data']=='[DONE]':continue
                payload=json.loads(row['data']);part='';ids=[]
                for c in payload.get('choices',[]):
                    part+=c.get('text','');ids.extend(c.get('token_ids') or [])
                if not ids and not part:continue
                count+=len(ids);chunks+=1
                self.times.append(row['phase_elapsed']);self.counts.append(count)
                self.parts.append(part);self.chunk_counts.append(chunks)
        assert count==self.record['response']['usage']['completion_tokens']==4096
        assert ''.join(self.parts)==self.record['response']['choices'][0]['text']
        assert all(a<=b for a,b in zip(self.times,self.times[1:]))
    def state(self,t):
        n=bisect_right(self.times,t)
        return (self.counts[n-1] if n else 0,''.join(self.parts[:n]),n,t>=self.end)


def visible_lines(text,cols=45,rows=6):
    # Exact text, line wrapping, with repeated blank lines visibly folded.
    lines=[];blank=0
    for line in text.replace('\r\n','\n').split('\n'):
        if not line.strip():blank+=1;continue
        if blank:
            lines.extend([''] if blank==1 else [f'[连续空行 × {blank}]'])
            blank=0
        line=line.expandtabs(4)
        lines.extend(textwrap.wrap(line,width=cols,replace_whitespace=False,
            drop_whitespace=False,break_long_words=True,break_on_hyphens=False) or [''])
    if blank:lines.extend([''] if blank==1 else [f'[连续空行 × {blank}]'])
    return lines[-rows:]

class Renderer:
    def __init__(self,root,speed):
        self.root=root;self.speed=speed
        self.summary=read(root/'summary.json');self.samples=read(root/'history-manifest.json')['samples']
        self.traces={b:[Trace(root,b,i) for i in range(8)] for b in ('native','kvmem')}
        self.duration=max(self.summary['arms'][b]['resume_seconds'] for b in self.traces)
        self.gain=self.summary['arms']['kvmem']['resume_output_tps']/self.summary['arms']['native']['resume_output_tps']
        self.coldgain=self.summary['arms']['kvmem']['cold_plus_resume_output_tps']/self.summary['arms']['native']['cold_plus_resume_output_tps']
        self.base=self.static()
    def static(self):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        txt(d,(30,17),'长历史 · 8 并发',34)
        txt(d,(31,67),'DeepSWE · 同一块 96 GB GPU · 显存预算 72% · 上限 256K',19,MUTED)
        d.rounded_rectangle((768,16,1032,105),radius=16,fill='#12352e',outline='#2a705c')
        txt(d,(900,18),f'{self.gain:.2f}×',43,GREEN,anchor='mt')
        txt(d,(900,71),'本轮实测续接吞吐',17,GREEN,anchor='mt')
        txt(d,(1768,24),'8 × 约 229K 历史',24,WHITE,anchor='rt')
        txt(d,(1768,64),'每请求固定生成 4,096 token',18,MUTED,anchor='rt')
        d.line((900,128,900,1107),fill=BORDER,width=2)
        for col,(backend,title,color) in enumerate((('native','原生 vLLM',AMBER),('kvmem','vLLM + KVMem',GREEN))):
            x=30+col*900;r=self.summary['arms'][backend]
            d.rounded_rectangle((x,121,x+840,211),radius=12,fill=PANEL,outline=BORDER)
            txt(d,(x+18,128),title,27,color)
            txt(d,(x+822,126),f'{r["resume_output_tps"]:.1f} tok/s',28,color,mono=True,anchor='rt')
            txt(d,(x+18,174),f'整批耗时 {r["resume_seconds"]:.1f}s',17,MUTED)
            txt(d,(x+822,174),f'{r["qps"]:.4f} req/s  ·  总输出 32,768 token',16,MUTED,anchor='rt')
        d.line((30,1120,1770,1120),fill=BORDER,width=1)
        txt(d,(30,1132),f'真实 SSE 到达时序 · 两侧统一 {self.speed:g}× 加速 · 等待时间完整保留',18,WHITE)
        txt(d,(30,1167),'4 个真实任务 × 各 2 个历史回合 · 回放模型续接输出，不执行工具或评判任务成功率 · 历史准备不在回放内',16,MUTED)
        txt(d,(30,1198),f'含历史准备的吞吐比：{self.coldgain:.2f}×  |  空行折叠为计数，其他输出按原文显示  |  单次 8 请求批次',16,MUTED)
        return im
    def frame(self,t):
        im=self.base.copy();d=ImageDraw.Draw(im)
        txt(d,(1768,1132),f'共同时间  {t:06.1f}s / {self.duration:.1f}s',18,WHITE,mono=False,anchor='rt')
        for col,(backend,color) in enumerate((('native',AMBER),('kvmem',GREEN))):
            states=[tr.state(t) for tr in self.traces[backend]]
            total=sum(s[0] for s in states);done=sum(s[3] for s in states);x0=30+col*900
            txt(d,(x0,222),f'已收到 {total:,} / 32,768 token',19,color)
            txt(d,(x0+840,222),f'完成 {done} / 8',19,color,anchor='rt')
            d.rounded_rectangle((x0,254,x0+840,260),radius=3,fill=BORDER)
            if total:d.rounded_rectangle((x0,254,x0+840*total/32768,260),radius=3,fill=color)
            for i,(count,text,chunks,complete) in enumerate(states):
                x=x0+(i%2)*426;y=277+(i//2)*207;w=414;h=193
                d.rounded_rectangle((x,y,x+w,y+h),radius=10,fill=PANEL,outline=color if complete else BORDER,width=1)
                for k,c in enumerate(('#ff6d77','#dfb65c','#5bc694')):d.ellipse((x+12+13*k,y+12,x+18+13*k,y+18),fill=c)
                sample=self.samples[i];short=sample['task'].split('-')[0]
                txt(d,(x+60,y+4),f'{i+1:02}  {short} / turn {sample["turn"]}',15,WHITE,mono=True)
                stalled=bool(count and not complete and t-self.traces[backend][i].times[chunks-1]>5)
                state='完成' if complete else ('等待新段' if stalled else ('输出中' if count else '等待首段'))
                txt(d,(x+w-12,y+4),state,14,color if count else MUTED,anchor='rt')
                d.line((x+1,y+31,x+w-1,y+31),fill=BORDER)
                lines=visible_lines(text)
                if not count:
                    txt(d,(x+13,y+61),'等待模型输出…',17,MUTED)
                    txt(d,(x+13,y+91),f'输入 {sample["prompt_tokens"]+1:,} token',14,MUTED)
                else:
                    for j,line in enumerate(lines):
                        is_fold=line.startswith('[连续空行')
                        c=MUTED if is_fold else (BLUE if line.lstrip().startswith(('<','```')) else WHITE)
                        txt(d,(x+12,y+39+j*18),line,14,c,mono=not is_fold)
                    # A cursor indicates an active stream, without inventing text.
                    if not complete and not stalled and int(t*3)%2==0:d.rectangle((x+w-15,y+141,x+w-9,y+153),fill=color)
                d.line((x+12,y+159,x+w-12,y+159),fill=BORDER)
                txt(d,(x+12,y+163),f'{count:,} / 4,096 tok',14,color,mono=True)
                timing=f'{self.traces[backend][i].end:.1f}s 完成' if complete else f'{chunks:,} 段已到达'
                txt(d,(x+w-12,y+162),timing,14,MUTED,anchor='rt')
        return im


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--speed',type=float,default=20);p.add_argument('--fps',type=int,default=10)
    p.add_argument('--poster-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);r=Renderer(args.root,args.speed)
    t=min(r.duration,r.summary['arms']['kvmem']['resume_seconds']+1)
    r.frame(t).save(args.output/'deepswe-8way-poster.png')
    r.frame(0).save(args.output/'deepswe-8way-start.png')
    r.frame(r.duration).save(args.output/'deepswe-8way-final.png')
    if args.poster_only:return
    # Derive one fixed palette from actual representative frames, then quantize
    # all frames identically so terminal text and colors do not shimmer.
    atlas=Image.new('RGB',(W,H*3))
    for j,at in enumerate((0,t,r.duration)):atlas.paste(r.frame(at),(0,j*H))
    palette=atlas.quantize(colors=96,method=Image.Quantize.MEDIANCUT)
    frames=[];durations=[];frame_times=[]
    n=math.ceil(r.duration/args.speed*args.fps)
    for i in range(n+1):
        at=min(i*args.speed/args.fps,r.duration)
        frame=r.frame(at).quantize(palette=palette,dither=Image.Dither.NONE)
        frames.append(frame);frame_times.append(at)
        durations.append(1200 if i==0 else (3500 if i==n else round(1000/args.fps)))
        if i%50==0:print(f'render {i}/{n}',flush=True)
    path=args.output/'deepswe-8way-real-stream.gif'
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,
        loop=0,optimize=True,disposal=1)
    shutil.copy2(Path(__file__),args.output/'render.py')
    provenance=dict(source=str(args.root.resolve()),timing='Both resume phase origins aligned to zero; each output chunk appears only after its actual complete SSE event arrival. No token timing interpolation.',
        speed=args.speed,fps=args.fps,frame_real_seconds=frame_times,frame_duration_ms=durations,
        display='Raw generated text with line wrap and consecutive blank lines folded to an explicit count. Tools in model text are not executed by this replay.',
        waiting_for_next_chunk_label_after_seconds=5,
        measured_resume_tps_ratio=r.gain,measured_cold_inclusive_tps_ratio=r.coldgain,
        data_sha256={str(f.relative_to(args.root)):sha(f) for f in sorted(args.root.rglob('*'))
            if f.is_file() and (f.name.endswith('.sse.jsonl') or f.name in ('summary.json','manifest.json','history-manifest.json','phase-times.json','profile.json','hardware.csv','stream-audit.json'))},
        renderer_sha256=sha(Path(__file__)),gif_sha256=sha(path),gif_bytes=path.stat().st_size)
    (args.output/'gif-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(gif=str(path),bytes=path.stat().st_size,frames=len(frames),ratio=r.gain)))

if __name__=='__main__':main()
