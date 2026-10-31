"""Isolated generic KV assembly component; synthetic bytes, real selection.

Defaults to CPU-only validation. --execute requires the GPU owner to have ended
all live workloads. Does not launch or stop servers. Results are neither model
throughput nor a semantic correctness claim. All archive publication, allocation,
JIT warmup, source reset, and byte checks are outside measured samples.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from functools import wraps


LAYERS, RECORD, HEADS, DIM, PAGE = 17, 32, 4, 256, 1616
RECORD_BYTES = LAYERS * RECORD * HEADS * DIM * 2
SHARED = ('assembly.py', 'authority.py', 'codec.py', 'config.py', 'memory.py',
          'memory_storage.py', 'storage.py', 'transfer.py', 'packed_assembly.py',
          'fp8_compat.py')


class StageTimes:
    """Opt-in host-only instrumentation; never records CUDA kernel durations."""
    def __init__(self):
        self.active = False
        self.rows = {}
        self.local = threading.local()
        self.lock = threading.Lock()
        self.originals = []

    def wrap(self, name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            stack = getattr(self.local, 'stack', None)
            if stack is None:
                stack = self.local.stack = []
            frame = [0.0, 0.0]
            stack.append(frame)
            cpu, wall = time.thread_time(), time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                wall_elapsed, cpu_elapsed = time.perf_counter()-wall, time.thread_time()-cpu
                stack.pop()
                if stack:
                    stack[-1][0] += wall_elapsed
                    stack[-1][1] += cpu_elapsed
                with self.lock:
                    row = self.rows.setdefault(name, dict(calls=0,wall_ms=0.0,thread_cpu_ms=0.0,
                        exclusive_wall_ms=0.0,exclusive_thread_cpu_ms=0.0))
                    row['calls'] += 1
                    row['wall_ms'] += wall_elapsed*1000
                    row['thread_cpu_ms'] += cpu_elapsed*1000
                    row['exclusive_wall_ms'] += (wall_elapsed-frame[0])*1000
                    row['exclusive_thread_cpu_ms'] += (cpu_elapsed-frame[1])*1000
        return timed

    def patch(self, owner, attribute, name):
        original = getattr(owner,attribute)
        self.originals.append((owner,attribute,original))
        setattr(owner,attribute,self.wrap(name,original))

    def start(self):
        self.rows = {}
        self.active = True

    def finish(self):
        self.active = False
        return self.rows

    def restore(self):
        for owner,attribute,original in reversed(self.originals):
            setattr(owner,attribute,original)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_threads():
    """Linux per-thread accounting; tick resolution is recorded, not hidden."""
    names = {t.native_id: t.name for t in threading.enumerate()}
    answer = {}
    for path in Path('/proc/self/task').glob('*/stat'):
        try:
            parts = path.read_text().rsplit(')', 1)[1].split()
            tid = int(path.parent.name)
            answer[str(tid)] = dict(name=names.get(tid, 'native-thread'),
                ticks=int(parts[11])+int(parts[12]))
        except FileNotFoundError:
            pass
    return answer


def inspect(args):
    runtimes = {}
    for item in args.runtime:
        label, path = item.split('=', 1)
        if not label.replace('_', '').replace('-', '').isalnum() or label in runtimes:
            raise ValueError('Runtime labels must be unique letters/numbers/dash/underscore')
        runtimes[label] = Path(path).resolve()
    if len(runtimes) < 2:
        raise ValueError('Provide a reference and at least one candidate runtime')
    if args.output.exists():
        raise ValueError('--output must not already exist')
    trace = json.loads(args.selection.read_text())
    history = trace['history_begin']
    ranges = trace['selected_history_ranges']
    if (history != 49088 or trace['selected_tokens'] != 36864 or
            len(ranges) != 1152 or any(n != RECORD or begin % RECORD for begin, n in ranges)):
        raise ValueError('Require the frozen first formal complete-record selection')
    if ranges != sorted(ranges) or len({r[0] for r in ranges}) != len(ranges):
        raise ValueError('Selection must be unique and history ordered')
    skipped = sum(n for i, (begin, n) in enumerate(ranges) if begin == i*RECORD)
    rebuilt = 36864-skipped
    if (skipped != trace['assembly_skipped_tokens'] or rebuilt != trace['assembly_rebuilt_tokens']
            or rebuilt * LAYERS * HEADS * DIM * 2 != trace['assembly_uploaded_bytes']):
        raise ValueError('First uncompressed-window remap does not match captured work')
    hashes, modules = {}, {}
    for label, runtime in runtimes.items():
        directory = runtime/'src/vllm_kvmem'
        native = list(directory.glob('_native*.so'))
        if len(native) != 1:
            raise ValueError('Require one frozen native extension per runtime')
        files = {name:directory/name for name in (*SHARED, 'window_assembly.py')}
        files['native_extension'] = native[0]
        modules[label] = files
        hashes[label] = {name:digest(path) for name, path in files.items()}
    reference = next(iter(runtimes))
    for label in runtimes:
        for name in (*SHARED, 'native_extension'):
            if hashes[label][name] != hashes[reference][name]:
                raise ValueError(f'{label}:{name} differs; this component isolates window_assembly only')
    if any(v not in (4, 7, 16) for v in args.chunks) or any(v not in (1, 8) for v in args.read_workers):
        raise ValueError('Supported isolated factors: chunks 4/7/16, read/gather workers 1/8')
    if not 1 <= args.warmup <= 4 or not 2 <= args.rounds <= 20:
        raise ValueError('Use 1..4 warmups and 2..20 measured rounds')
    evidence = dict(scope=__doc__, status='validated_cpu_only', source_hashes=hashes,
        source_paths={label:{name:str(path) for name,path in files.items()} for label,files in modules.items()},
        selection_sha256=digest(args.selection), selection_source=trace.get('_component_source'),
        script_sha256=digest(Path(__file__)), reference=reference,
        geometry=dict(layers=LAYERS, record_tokens=RECORD, heads=HEADS, dim=DIM,
            native_page_tokens=PAGE, history_tokens=history, select_budget=36864,
            gen_budget=12288, selected_records=1152, skipped_tokens=skipped,
            rebuilt_tokens=rebuilt, uploaded_bytes=trace['assembly_uploaded_bytes'],
            record_bytes=RECORD_BYTES),
        factors=dict(chunks=args.chunks, read_workers=args.read_workers,
            read_workers_meaning='MemoryArchive.gather_workers; archive I/O executor stays at 2',
            gather_parallel_threshold_bytes=8*1024**2, raw_cache_bytes=0,
            reuse_resident_v=False, packed=True, direct_pinned_reads=True,
            read_prefetch=False, fused_layers=False),
        instrument_stages=args.instrument_stages,
        instrumentation_note='Opt-in Python host-stage timing has overhead and is separate from normal timing. '
            'Inclusive stages nest; do not sum their wall/CPU values. Exclusive values subtract only '
            'instrumented children on the same thread. Neither wall nor CPU stages measure GPU kernels or pure DMA.',
        data='Deterministic synthetic FP8 K/V fill varies by token/layer/head/channel. '
             'Only the selection and cache geometry come from the real DeepSWE run.',
        oracle='All physical K/V bytes equal the warmed reference generic assembler; '
               'unwritten token slots, null/unused pages and page-tail padding equal the seed. '
               'Not semantic model correctness.', warmup=[], samples=[], byte_checks=[])
    return runtimes, modules, trace, evidence


def execute(args, runtimes, modules, trace, evidence):
    reference = evidence['reference']
    sys.path.insert(0, str(runtimes[reference]/'src'))
    import torch
    from vllm_kvmem import _native
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.codec import native_vllm_fp8_views
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    if Path(_native.__file__).resolve() != modules[reference]['native_extension']:
        raise ValueError('Imported native extension is outside the frozen reference runtime')
    arms = {}
    for i, (label, files) in enumerate(modules.items()):
        name = f'vllm_kvmem._generic_diagnostic_arm_{i}'
        spec = importlib.util.spec_from_file_location(name, files['window_assembly.py'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        arms[label] = module
    torch.set_num_threads(1)
    torch.manual_seed(20261031)
    args.output.mkdir(parents=True)
    setup = time.perf_counter()
    history = trace['history_begin']
    archive = MemoryArchive(str(args.output/'cpu-only'), RECORD_BYTES, 16384,
        gather_workers=8, workers=2, cpu_direct_record_seal=True, cpu_seal_hash_workers=1)
    owner = RequestAuthority('generic-component', KVMemConfig(block_tokens=RECORD,
        select_budget=36864, gen_budget=12288, repository_blocks=16384,
        sink_blocks=41, recent_blocks=0), archive, LAYERS, HEADS, DIM,
        raw_dtype=torch.float8_e4m3fn, resident_records=64)

    def values(begin, count, layer, device):
        position = torch.arange(begin, begin+count, device=device, dtype=torch.int32)[:, None, None]
        head = torch.arange(HEADS, device=device, dtype=torch.int32)[None, :, None]
        channel = torch.arange(DIM, device=device, dtype=torch.int32)[None, None, :]
        raw = ((position*3 + layer*19 + head*11 + channel*5) % 80 + 16).to(torch.uint8)
        value = ((position*7 + layer*13 + head*17 + channel*3) % 127).to(torch.uint8)
        return raw, value

    for begin in range(0, history, RECORD*32):
        count = min(RECORD*32, history-begin)
        payloads = [values(begin, count, layer, 'cpu') for layer in range(LAYERS)]
        raw = torch.stack([r for r,v in payloads])
        value = torch.stack([v for r,v in payloads])
        assert owner.append(raw.view(torch.float8_e4m3fn), value,
            history_begin=begin, local_rope_begin=begin)
        owner.publish_sealed(lambda: archive.wait_for_capacity(.05))
    del payloads, raw, value
    owner.poll()
    assert all(record.archived for record in owner.records.values())
    chosen = {begin for begin,count in trace['selected_history_ranges']}
    plan = owner.memory.select({r.handle.slot:float(r.history_begin in chosen)
        for r in owner.records.values()}, action=False)
    by_slot = {r.handle.slot:r for r in owner.records.values()}
    actual = [[by_slot[r.block_id].history_begin,r.n_tokens] for r in plan.remaps]
    assert actual == trace['selected_history_ranges'], 'Native picker changed frozen selected order'
    assert sum(r.n_tokens for r in plan.remaps if r.skip) == trace['assembly_skipped_tokens']
    pending = [r for r in plan.remaps if not r.skip]
    page_count = math.ceil(history/PAGE)
    physical_pages = page_count+7
    table = (torch.randperm(physical_pages-1, device='cuda')[:page_count]+1).to(torch.int32)
    positions = torch.arange(history, device='cuda', dtype=torch.int64)
    slots = table[positions//PAGE].long()*PAGE + positions%PAGE
    angles = torch.randn(page_count*PAGE, 16, device='cuda')
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1)
    seeds, frame, layers = [], [], []
    for i in range(LAYERS):
        seed = torch.full((physical_pages, HEADS, PAGE, DIM*2), 165,
                          dtype=torch.uint8, device='cuda')
        key, value = native_vllm_fp8_views(seed, HEADS, DIM)
        scale = torch.tensor(.5+i*.03125, device='cuda')
        raw, val = values(0, history, i, 'cuda')
        assemble_fp8(raw.view(torch.float8_e4m3fn), val.view(torch.float8_e4m3fn),
                     positions, slots, trig, key, value, scale)
        target = seed.clone()
        k,v = native_vllm_fp8_views(target, HEADS, DIM)
        seeds.append(seed)
        frame.append(target)
        layers.append(arms[reference].LayerWindow(k,v,trig,scale,table))
    del raw, val
    changed = torch.zeros(physical_pages*PAGE, dtype=torch.bool, device='cuda')
    destinations = torch.tensor([p for r in pending for p in range(r.to_base,r.to_base+r.n_tokens)],
                                dtype=torch.int64,device='cuda')
    changed[table[destinations//PAGE].long()*PAGE+destinations%PAGE] = True
    changed = changed.view(physical_pages,PAGE)
    torch.cuda.synchronize()
    evidence.update(setup_seconds=time.perf_counter()-setup, device=torch.cuda.get_device_name(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_tick_hz=os.sysconf('SC_CLK_TCK'),
        physical_pages=physical_pages, logical_pages=page_count,
        source_order_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest(),
        page_table=table.cpu().tolist())
    pools, pointers, oracle = {}, {}, None
    last_result = None
    stages = StageTimes() if args.instrument_stages else None
    if stages:
        from vllm_kvmem import packed_assembly, transfer
        stages.patch(archive,'read','archive.read_submit')
        stages.patch(packed_assembly.PackedRead,'result','reader.result_wait')
        stages.patch(transfer.PackedTransfers,'upload','transfer.upload')
        stages.patch(transfer.PackedTransfers,'upload_from_reader','transfer.upload_from_reader')
        stages.patch(packed_assembly,'assemble_packed_layer','assemble_packed_layer.submit')
        for module in arms.values():
            stages.patch(module.NativeWindowAssembler,'_upload_metadata','metadata.upload_submit')
            stages.patch(module,'_validate_destinations','metadata.validate_destinations')
            stages.patch(module._AssemblyStaging,'destination','staging.destination')
            stages.patch(module._AssemblyStaging,'consumed','staging.consumer_fence_submit')

    def sample(label, chunk, workers):
        nonlocal last_result, oracle
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        # The same initialized executor and authority serve every arm. Quiescent
        # policy changes only how many gather shards a read submits.
        archive.gather_workers = workers
        for destination, seed in zip(frame, seeds):
            destination.copy_(seed)
        torch.cuda.synchronize()
        before_threads = cpu_threads()
        begin_event, end_event = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if stages:
            stages.start()
        main0, cpu0, start = time.thread_time(), time.process_time(), time.perf_counter()
        begin_event.record()
        key = (label,chunk)
        constructor = arms[label].NativeWindowAssembler
        if stages:
            constructor = stages.wrap('assembler.construct',constructor)
        assembler = constructor(tuple(layers), RECORD, chunk_blocks=chunk,
            packed=True, direct_pinned_reads=True, raw_cache=None, reuse_resident_v=False,
            raw_dtype=torch.float8_e4m3fn, read_prefetch=False, fused_layers=False,
            staging=pools.get(key))
        pools.setdefault(key,assembler.staging)
        assembler._shared_staging = True
        result = assembler.assemble(owner,plan)
        if stages:
            stages.wrap('finish.final_event_submit',end_event.record)()
        else:
            end_event.record()
        submitted = time.perf_counter()
        main1,cpu1 = time.thread_time(),time.process_time()
        def finish_fences():
            result.event.synchronize()
            end_event.synchronize()
        if stages:
            stages.wrap('finish.completion_wait',finish_fences)()
        else:
            finish_fences()
        finished = time.perf_counter()
        main2,cpu2 = time.thread_time(),time.process_time()
        after_threads = cpu_threads()
        close_begin = time.perf_counter()
        assert (stages.wrap('finish.assembler_close',assembler.close)() if stages else assembler.close())
        close_ms = (time.perf_counter()-close_begin)*1000
        stage_rows = stages.finish() if stages else None
        current_pointers = [t.data_ptr() for t in assembler.transfers.buffers+assembler.staging.buffers]
        if key in pointers:
            assert current_pointers == pointers[key], 'Staging addresses changed after warmup'
        pointers[key] = current_pointers
        assert result.rebuilt_tokens == trace['assembly_rebuilt_tokens']
        assert result.skipped_tokens == trace['assembly_skipped_tokens']
        assert result.uploaded_bytes == trace['assembly_uploaded_bytes']
        assert result.chunks == math.ceil(len(pending)/chunk)
        assert not (result.split_tokens or result.partial_tokens or result.resident_v_reused_tokens
                    or result.raw_cache_hit_tokens or result.raw_cache_capacity_bytes)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        tasks = sum(min(workers,min(chunk,len(pending)-i))
                    if min(chunk,len(pending)-i)*RECORD_BYTES >= 8*1024**2 else 1
                    for i in range(0,len(pending),chunk))
        assert result.read_stats['cache_gather_tasks'] == tasks, 'Gather branch differs from declared factor'
        if oracle is None:
            assert label == reference
            oracle = [t.clone() for t in frame]
            # Check every unwritten token's K/V bytes, including physical null
            # and unused pages, skipped records, and final-page padding.
            for seed, actual_frame in zip(seeds, frame):
                for p in range(physical_pages):
                    mask = ~changed[p]
                    assert torch.equal(seed[p,:,mask,:],actual_frame[p,:,mask,:]), 'Generic reference overwrote guard bytes'
        equal = all(torch.equal(a,b) for a,b in zip(oracle,frame))
        assert equal, f'{label} changed physical K/V bytes versus reference'
        last_result = result
        tick_hz = evidence['cpu_tick_hz']
        threads = {tid:dict(name=row['name'],cpu_ms=(row['ticks']-before_threads.get(tid,{'ticks':0})['ticks'])*1000/tick_hz)
                   for tid,row in after_threads.items()
                   if row['ticks']-before_threads.get(tid,{'ticks':0})['ticks'] > 0}
        return dict(arm=label,chunk_blocks=chunk,read_workers=workers,
            wall_ms=(finished-start)*1000,submit_wall_ms=(submitted-start)*1000,
            completion_wait_wall_ms=(finished-submitted)*1000,close_wall_ms=close_ms,
            submit_main_thread_cpu_ms=(main1-main0)*1000,total_main_thread_cpu_ms=(main2-main0)*1000,
            submit_process_cpu_ms=(cpu1-cpu0)*1000,total_process_cpu_ms=(cpu2-cpu0)*1000,
            per_thread_cpu_tick_deltas=threads,current_stream_span_ms=begin_event.elapsed_time(end_event),
            host_stages=stage_rows,instrumented=bool(stages),
            reference_expected_layer_submissions=result.chunks*LAYERS,
            host_prepare_ms=result.host_prepare_seconds*1000,read_stats=result.read_stats,
            rebuilt_tokens=result.rebuilt_tokens,skipped_tokens=result.skipped_tokens,
            uploaded_bytes=result.uploaded_bytes,chunks=result.chunks,all_arena_bytes_equal=True,
            timing_note='CUDA span includes host submission gaps; completion wait includes remaining DMA and kernels, not isolated DMA duration.')

    try:
        factors = [(c,w) for c in args.chunks for w in args.read_workers]
        names = list(arms)
        # All loading, JIT and slot allocation finish before any measured row.
        for warmup in range(args.warmup):
            for chunk,workers in factors:
                for label in names:
                    evidence['warmup'].append(dict(round=warmup,**sample(label,chunk,workers)))
        evidence['warmup_end_monotonic'] = time.monotonic()
        for trial in range(args.rounds):
            # For two runtimes the flattened order is ABAB then BABA, balancing
            # which runtime leads each fixed-factor comparison. Factor order
            # also reverses; every runtime runs immediately beside its control.
            order = names if trial%4 < 2 else list(reversed(names))
            for chunk,workers in (factors if trial%2==0 else list(reversed(factors))):
                for label in order:
                    row = dict(round=trial,**sample(label,chunk,workers))
                    evidence['samples'].append(row)
                    with (args.output/'samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
        summary = {}
        for label in names:
            for chunk,workers in factors:
                rows = [r for r in evidence['samples'] if (r['arm'],r['chunk_blocks'],r['read_workers'])==(label,chunk,workers)]
                summary[f'{label}/chunk{chunk}/workers{workers}'] = dict(samples=len(rows),**{
                    'median_'+key:statistics.median(r[key] for r in rows) for key in
                    ('wall_ms','submit_wall_ms','completion_wait_wall_ms','total_main_thread_cpu_ms',
                     'total_process_cpu_ms','host_prepare_ms','current_stream_span_ms')})
        evidence['summary'] = summary
        evidence['memory'] = dict(archive=archive.io_usage(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 cuda_reserved_bytes=torch.cuda.memory_reserved())
        for label, files in modules.items():
            assert {name:digest(path) for name,path in files.items()} == evidence['source_hashes'][label]
        assert digest(args.selection) == evidence['selection_sha256']
        owner.memory.attach_completion(last_result)
        assert owner.memory.commit_selection()
        for pool in pools.values():
            assert pool.close()
        del plan,last_result
        assert owner.close() and archive.close()
        evidence.update(status='completed',cleanup_complete=True,
            byte_checks=dict(samples=len(evidence['samples'])+len(evidence['warmup']),
                layers=LAYERS,bytes_per_sample=sum(t.numel() for t in frame),
                whole_arena_equal=True,reference_unwritten_guard_equal=True))
        (args.output/'result.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(dict(status='completed',summary=summary),indent=2))
    except BaseException as error:
        evidence.update(status='failed',error=repr(error))
        (args.output/'result.json').write_text(json.dumps(evidence,indent=2)+'\n')
        raise
    finally:
        if stages:
            stages.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',action='append',required=True,help='LABEL=/path/to/runtime; first is reference')
    parser.add_argument('--selection',type=Path,default=Path(__file__).with_name('pressure-selection.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunks',type=lambda s:[int(x) for x in s.split(',')],default=[4,7,16])
    parser.add_argument('--read-workers',type=lambda s:[int(x) for x in s.split(',')],default=[1,8])
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--instrument-stages',action='store_true',
        help='Separate host-stage diagnostic with measurement overhead; never mix with normal timing')
    parser.add_argument('--execute',action='store_true',help='Actually use the GPU after the owner ended live runs')
    args = parser.parse_args()
    runtimes,modules,trace,evidence = inspect(args)
    if not args.execute:
        print(json.dumps({key:evidence[key] for key in ('status','geometry','factors','data','oracle')},indent=2))
        return
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args,runtimes,modules,trace,evidence)


if __name__ == '__main__':
    main()
