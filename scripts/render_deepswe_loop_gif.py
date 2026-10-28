"""English terminal replay with measured NVML VRAM, looping each full capture."""
import argparse
from bisect import bisect_right
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
MONO='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
SANS='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTS={}
def font(size,mono=False):
    key=(size,mono)
    if key not in FONTS:FONTS[key]=ImageFont.truetype(MONO if mono else SANS,size)
    return FONTS[key]
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def txt(draw,xy,text,size=18,fill=WHITE,mono=False,anchor=None):
    draw.text(xy,str(text),font=font(size,mono),fill=fill,anchor=anchor)

class Trace:
    def __init__(self,root,backend,index):
        self.record=read(root/backend/f'resume-{index}.json')
        phase=read(root/backend/'phase-times.json')['resume_begin']
        self.end=self.record['end']-phase;self.times=[];self.counts=[];self.parts=[]
        count=0;ids=[]
        with (root/backend/f'resume-{index}.sse.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['data']=='[DONE]':continue
                payload=json.loads(row['data']);part='';new_ids=[]
                for c in payload.get('choices',[]):
                    part+=c.get('text','');new_ids.extend(c.get('token_ids') or [])
                if not new_ids and not part:continue
                count+=len(new_ids);ids.extend(new_ids)
                self.times.append(row['phase_elapsed']);self.counts.append(count);self.parts.append(part)
        assert count==4096 and ids==self.record['response']['choices'][0]['token_ids']
        assert ''.join(self.parts)==self.record['response']['choices'][0]['text']
        assert all(a<=b for a,b in zip(self.times,self.times[1:]))
    def state(self,t):
        n=bisect_right(self.times,t)
        return (self.counts[n-1] if n else 0,''.join(self.parts[:n]),n,t>=self.end)

class MemorySeries:
    def __init__(self,root,backend):
        directory=root/backend;self.origin=read(directory/'phase-times.json')['resume_begin']
        self.rows=[json.loads(line) for line in (directory/'gpu-telemetry.jsonl').read_text().splitlines()]
        self.times=[r['at']-self.origin for r in self.rows]
        assert self.times[0]<=0
        assert all(a<=b for a,b in zip(self.times,self.times[1:]))
        assert all(r['memory_total_bytes']==r['memory_used_bytes']+r['memory_free_bytes']+r['memory_reserved_bytes'] for r in self.rows)
    def at(self,t):
        index=bisect_right(self.times,t)-1
        if index<0:raise ValueError('No measured memory sample precedes this frame')
        return self.rows[index],t-self.times[index]


class MergedStream:
    """Group complete received chunks into comparable, roughly 32-token rows."""
    def __init__(self,traces,target_tokens=32):
        self.segments=[]
        for index,trace in enumerate(traces):
            count=0;previous=0;parts=[];own=[]
            for at,cumulative,part in zip(trace.times,trace.counts,trace.parts):
                count+=cumulative-previous;previous=cumulative;parts.append(part)
                if count>=target_tokens:
                    own.append(dict(at=at,request=index,tokens=count,text=''.join(parts)))
                    count=0;parts=[]
            if count or parts:
                own.append(dict(at=trace.times[-1],request=index,tokens=count,text=''.join(parts)))
            assert sum(s['tokens'] for s in own)==4096
            assert ''.join(s['text'] for s in own)==''.join(trace.parts)
            self.segments.extend(own)
        self.segments.sort(key=lambda s:(s['at'],s['request']))
        self.times=[s['at'] for s in self.segments]
        assert sum(s['tokens'] for s in self.segments)==32768

    def visible(self,t,cycle,rows=14):
        end=bisect_right(self.times,t)
        visible=self.segments[max(0,end-rows):end]
        if cycle and len(visible)<rows:
            visible=self.segments[-(rows-len(visible)):]+visible
        return visible


def segment_lines(text,columns=68):
    # Formatting markers expose newline output without scrolling empty pages.
    value=text.replace('\r\n','\n').replace('\r','\\r').expandtabs(4)
    value=re.sub(r'\n{2,}',lambda m:f' [{len(m[0])} newlines] ',value).replace('\n',' ↵ ')
    lines=textwrap.wrap(value,width=columns,replace_whitespace=False,
        drop_whitespace=False,break_long_words=True,break_on_hyphens=False)
    if not lines:return ['[no text in this segment]']
    if len(lines)>2:return [lines[0],lines[1][:columns-1]+'…']
    return lines


def loop_position(elapsed,duration):
    if duration<=0 or elapsed<0:raise ValueError('Invalid replay clock')
    return int(elapsed//duration),elapsed%duration


def visible_lines(text,cols=45,rows=6):
    lines=[];blank=0
    for line in text.replace('\r\n','\n').split('\n'):
        if not line.strip():blank+=1;continue
        if blank:lines.extend([''] if blank==1 else [f'[{blank} blank lines]']);blank=0
        lines.extend(textwrap.wrap(line.expandtabs(4),width=cols,replace_whitespace=False,
            drop_whitespace=False,break_long_words=True,break_on_hyphens=False) or [''])
    if blank:lines.extend([''] if blank==1 else [f'[{blank} blank lines]'])
    return lines[-rows:]

class Renderer:
    def __init__(self,root,speed=5,layout='grid'):
        if layout not in ('grid','merged'):raise ValueError('Unknown terminal layout')
        self.root=root;self.speed=speed;self.layout=layout;self.summary=read(root/'summary.json')
        self.samples=read(root/'history-manifest.json')['samples']
        self.traces={b:[Trace(root,b,i) for i in range(8)] for b in ('native','kvmem')}
        self.merged={b:MergedStream(traces) for b,traces in self.traces.items()} if layout=='merged' else {}
        self.memory={b:MemorySeries(root,b) for b in self.traces}
        self.durations={b:self.summary['arms'][b]['resume_seconds'] for b in self.traces}
        self.duration=max(self.durations.values())
        self.gain=self.summary['arms']['kvmem']['resume_output_tps']/self.summary['arms']['native']['resume_output_tps']
        self.base=self.static()
    def static(self):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        txt(d,(30,19),'Long-context serving',34)
        txt(d,(31,73),'RTX PRO 6000 / 96 GB / 256K context limit',18,MUTED)
        d.rounded_rectangle((768,13,1032,108),radius=16,fill='#12352e',outline='#2a705c')
        txt(d,(900,19),f'{self.gain:.2f}x',43,GREEN,anchor='mt')
        txt(d,(900,77),'GENERATION THROUGHPUT',13,GREEN,anchor='mt')
        txt(d,(1768,27),'8 concurrent requests',24,WHITE,anchor='rt')
        txt(d,(1768,73),'Real DeepSWE histories',18,MUTED,anchor='rt')
        d.line((900,126,900,1116),fill=BORDER,width=2)
        for col,(backend,title,color) in enumerate((('native','Native vLLM',AMBER),('kvmem','vLLM + KVMem',GREEN))):
            x=30+col*900;r=self.summary['arms'][backend]
            d.rounded_rectangle((x,125,x+840,295),radius=12,fill=PANEL,outline=BORDER)
            txt(d,(x+18,138),title,27,color)
            txt(d,(x+820,133),f'{r["resume_output_tps"]:.1f} tok/s',35,color,mono=True,anchor='rt')
            txt(d,(x+18,181),'GENERATION THROUGHPUT',13,MUTED)
            txt(d,(x+820,183),'8 x 4,096 = 32,768 output tokens',14,MUTED,mono=True,anchor='rt')
            d.line((x+18,209,x+822,209),fill=BORDER)
            txt(d,(x+18,221),'GPU VRAM',14,MUTED)
            txt(d,(x+18,272),'NVML / 1 Hz',12,MUTED)
        d.line((30,1130,1770,1130),fill=BORDER)
        txt(d,(30,1146),'RECORDED REPLAY',16,GREEN)
        txt(d,(262,1146),f'Each side repeats its full measured trace. Same {self.speed:g}x playback speed.',16,WHITE)
        txt(d,(1768,1146),'VRAM follows the recorded trace',15,MUTED,anchor='rt')
        policy=read(self.root/'manifest.json').get('memory_policy')
        budget_label='Near-full GPU memory on both sides' if policy=='near-full-physical' else 'Measured GPU memory shown above'
        txt(d,(30,1180),f'8 histories / 4 real tasks / about 229K input tokens per request / 4,096 output tokens per request / {budget_label}',14,MUTED)
        note=('History preparation excluded; waits included. Output grouped into ~32-token segments; newlines condensed; long segments clipped.'
            if self.layout=='merged' else 'History preparation excluded. Generation throughput includes waits across all 8 continuation requests. Loops replay captured output.')
        txt(d,(30,1208),note,13,MUTED)
        return im

    def draw_merged(self,d,backend,x,cycle,t,color):
        y=332;w=840;bottom=1118
        d.rounded_rectangle((x,y,x+w,bottom),radius=12,fill=PANEL,outline=BORDER)
        for i,c in enumerate(('#ff6d77','#dfb65c','#5bc694')):
            d.ellipse((x+18+18*i,y+19,x+27+18*i,y+28),fill=c)
        title='vllm' if backend=='native' else 'kvmem-vllm'
        txt(d,(x+94,y+10),f'{title} / combined output',21,color,mono=True)
        txt(d,(x+w-20,y+16),'8 REQUESTS',13,MUTED,mono=True,anchor='rt')
        d.line((x+1,y+45,x+w-1,y+45),fill=BORDER)
        txt(d,(x+18,y+56),'REQ',12,MUTED,mono=True)
        txt(d,(x+96,y+56),'GENERATED OUTPUT / ~32 TOKENS PER SEGMENT',12,MUTED,mono=True)
        visible=self.merged[backend].visible(t,cycle)
        if not visible:txt(d,(x+96,419),'Waiting for streamed output...',17,MUTED,mono=True)
        for index,segment in enumerate(visible):
            row_y=410+index*44
            txt(d,(x+18,row_y),f'R{segment["request"]+1:02}',15,color,mono=True)
            txt(d,(x+18,row_y+21),f'+{segment["tokens"]}',12,MUTED,mono=True)
            lines=segment_lines(segment['text'])
            for line_index,line in enumerate(lines):
                txt(d,(x+96,row_y+20*line_index),line,17,WHITE,mono=True)
        states=[trace.state(t) for trace in self.traces[backend]]
        total=sum(state[0] for state in states)
        d.line((x+18,1042,x+w-18,1042),fill=BORDER)
        txt(d,(x+18,1054),'TOTAL OUTPUT',14,MUTED)
        txt(d,(x+w-18,1049),f'{total:,} / 32,768 tokens',26,color,mono=True,anchor='rt')
        d.rounded_rectangle((x+18,1092,x+w-18,1101),radius=4,fill=BORDER)
        if total:d.rounded_rectangle((x+18,1092,x+18+(w-36)*total/32768,1101),radius=4,fill=color)

    def frame(self,elapsed):
        im=self.base.copy();d=ImageDraw.Draw(im)
        for col,(backend,color) in enumerate((('native',AMBER),('kvmem',GREEN))):
            cycle,t=loop_position(elapsed,self.durations[backend]);x0=30+col*900
            measured,age=self.memory[backend].at(t)
            used=measured['memory_used_bytes']/1024**3;total=measured['memory_total_bytes']/1024**3
            txt(d,(x0+172,217),f'{used:.1f} / {total:.1f} GiB',24,WHITE,mono=True)
            txt(d,(x0+820,220),f'GPU {measured["gpu_utilization_percent"]:3d}%',20,WHITE,mono=True,anchor='rt')
            d.rounded_rectangle((x0+172,258,x0+820,267),radius=4,fill=BORDER)
            d.rounded_rectangle((x0+172,258,x0+172+648*used/total,267),radius=4,fill=BLUE)
            txt(d,(x0+820,274),f'sample age {age:.1f}s',11,MUTED,anchor='rt')
            txt(d,(x0,305),f'LOOP {cycle+1:02d}',13,color,mono=True)
            txt(d,(x0+840,305),f'Trace {t:06.1f}s / {self.durations[backend]:.1f}s',13,MUTED,mono=True,anchor='rt')
            if self.layout=='merged':
                self.draw_merged(d,backend,x0,cycle,t,color)
                continue
            for i,tr in enumerate(self.traces[backend]):
                count,text,chunks,complete=tr.state(t)
                x=x0+(i%2)*426;y=332+(i//2)*197;w=414;h=185
                stalled=bool(count and not complete and t-tr.times[chunks-1]>5)
                active=bool(count and not complete and not stalled)
                d.rounded_rectangle((x,y,x+w,y+h),radius=10,fill=PANEL,outline=color if active else BORDER)
                for k,c in enumerate(('#ff6d77','#dfb65c','#5bc694')):d.ellipse((x+12+13*k,y+12,x+18+13*k,y+18),fill=c)
                sample=self.samples[i];short=sample['task'].split('-')[0]
                txt(d,(x+60,y+5),f'{i+1:02} {short} / {sample["turn"]}',14,WHITE,mono=True)
                status='STREAMING' if active else ('NEXT LOOP' if complete else 'WAITING')
                txt(d,(x+w-12,y+6),status,12,color if active else MUTED,mono=True,anchor='rt')
                d.line((x+1,y+30,x+w-1,y+30),fill=BORDER)
                if not count:
                    # Retain the terminal tail between replay loops, but add no
                    # new text until the actual first captured chunk arrives.
                    if cycle>0:lines=visible_lines(''.join(tr.parts))
                    else:lines=['Waiting for the first chunk...',f'Input: {sample["prompt_tokens"]+1:,} tokens']
                else:lines=visible_lines(text)
                for j,line in enumerate(lines):
                    folded=line.startswith('[') and 'blank lines]' in line
                    color_line=MUTED if folded or not count else (BLUE if line.lstrip().startswith(('<','```')) else WHITE)
                    txt(d,(x+12,y+39+j*17),line,14,color_line,mono=True)
                if active and int(elapsed*3)%2==0:d.rectangle((x+w-16,y+132,x+w-10,y+144),fill=color)
                d.line((x+12,y+151,x+w-12,y+151),fill=BORDER)
                txt(d,(x+12,y+162),f'{count:,} / 4,096 tokens',13,color,mono=True)
                txt(d,(x+w-12,y+162),f'{chunks:,} chunks',12,MUTED,mono=True,anchor='rt')
        return im


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--speed',type=float,default=2.5)
    p.add_argument('--fps',type=int,default=5);p.add_argument('--poster-only',action='store_true')
    p.add_argument('--layout',choices=('grid','merged'),default='grid')
    args=p.parse_args()
    if args.speed<=0 or args.fps<=0:raise ValueError('Playback speed and frame rate must be positive')
    assert read(args.root/'stream-audit.json')['passed']
    args.output.mkdir(parents=True,exist_ok=True);r=Renderer(args.root,args.speed,args.layout)
    preview_t=min(r.durations['kvmem']*.55,r.duration*.5)
    r.frame(preview_t).save(args.output/'deepswe-loop-poster.png')
    if args.poster_only:return
    atlas=Image.new('RGB',(W,H*3))
    for j,at in enumerate((0,preview_t,r.duration*.8)):atlas.paste(r.frame(at),(0,j*H))
    palette=atlas.quantize(colors=96,method=Image.Quantize.MEDIANCUT)
    frames=[];frame_times=[];durations=[]
    n=math.ceil(r.duration/args.speed*args.fps)
    for i in range(n):
        t=i*args.speed/args.fps;frame_times.append(t)
        frames.append(r.frame(t).quantize(palette=palette,dither=Image.Dither.NONE))
        durations.append(round(1000/args.fps))
        if i%100==0:print(f'render {i}/{n}',flush=True)
    path=args.output/'deepswe-throughput-vram-loop.gif'
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=1)
    shutil.copy2(Path(__file__),args.output/'render.py')
    source_files=[args.root/'summary.json',args.root/'manifest.json',args.root/'history-manifest.json',args.root/'stream-audit.json']
    for b in ('native','kvmem'):
        source_files.extend(args.root/b/name for name in ('phase-times.json','gpu-telemetry.jsonl','telemetry-device.json','telemetry-status.json','profile.json'))
        source_files.extend(args.root/b/f'resume-{i}.sse.jsonl' for i in range(8))
    provenance=dict(source=str(args.root.resolve()),speed=args.speed,fps=args.fps,layout=args.layout,frame_real_seconds=frame_times,
        frame_duration_ms=durations,loop_periods_real_seconds=r.durations,language='English UI; original generated output unchanged',
        output_timing='Each backend repeats its complete measured resume capture. Actual SSE arrivals and waits are preserved within each cycle. No completed-request or throughput claims are inferred from repeated playback.',
        vram_timing='NVML v2 memory.used, excluding driver-reserved bytes. Most recent actual sample at or before the backend trace time; no interpolation. Each display includes sample age.',
        measured_resume_tps_ratio=r.gain,
        generation_throughput_definition='32768 output tokens / elapsed seconds from submission of all 8 continuation requests to final response, including waits; history preparation excluded.',
        merged_output_format=('Complete SSE chunks grouped per request until at least 32 tokens, then sorted by the final chunk arrival timestamp. No early display or simulated arrivals. All segment text and counts reconcile with the original streams. Newlines are marked, long text is clipped to two lines. Total output counters use every actually arrived token, including pending partial segments.' if args.layout=='merged' else None),
        gpu_memory_utilization={b:read(args.root/b/'profile.json')['gpu_memory_utilization'] for b in ('native','kvmem')},
        source_sha256={str(f.relative_to(args.root)):sha(f) for f in source_files},renderer_sha256=sha(Path(__file__)),
        gif_sha256=sha(path),gif_bytes=path.stat().st_size)
    (args.output/'gif-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(gif=str(path),bytes=path.stat().st_size,frames=len(frames),seconds=sum(durations)/1000,ratio=r.gain)))

if __name__=='__main__':main()
