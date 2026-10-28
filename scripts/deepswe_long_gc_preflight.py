"""Eight real, incremental synthetic histories; no forged positions or task answers.

Binary append-only journals retain every input/output token. HTTP requests carry
the complete prefix, but evidence stores offsets and hashes instead of making a
quadratic number of on-disk copies. This is capacity/lifecycle qualification,
not a quality test or a measurement of natural Agent throughput.
"""
import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

import psutil


SPEC = dict(schema=1, owners=8, target_tokens=9_999_968, append_tokens=32768,
            output_tokens=64, minimum_gc_cycles=8, minimum_free_host_bytes=8*2**30)


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def token_bytes(tokens):
    data = array.array('I', tokens)
    if data.itemsize != 4:
        raise ValueError('Expected uint32 token journal')
    if sys.byteorder != 'little':
        data.byteswap()
    return data.tobytes()


def next_input_count(history, spec):
    remaining = spec['target_tokens'] - history
    if remaining <= 0:
        return 0
    count = min(spec['append_tokens'], remaining-spec['output_tokens'])
    if count <= 0:
        raise ValueError('Final turn must leave room for input and fixed output')
    # Avoid leaving a final turn consisting solely of output.
    if 0 < remaining-count-spec['output_tokens'] <= spec['output_tokens']:
        count -= spec['output_tokens']+1
    return count


def validate_spec(cfg, spec):
    if spec != SPEC:
        raise ValueError('Long-history qualification must match its frozen specification')
    kv = cfg['kvmem']
    if (cfg['max_model_len'] != 10_000_000 or cfg['max_num_seqs'] != 8 or
            kv['select_budget'] != 98304 or kv['gen_budget'] != 16384 or
            kv['historical_gc']['mode'] != 'delete' or kv['historical_gc']['policy'] != 'lru'):
        raise ValueError('Expected 96K/16K, eight owners, deleting LRU, decimal 10M')


def check_snapshot(archive, scheduler, cfg):
    history = archive.get('historical_gc') or {}
    if history.get('policy_failures', 0) or history.get('stale', 0):
        raise RuntimeError('Historical policy/stale-generation failure')
    if archive.get('session_checkpoint_deferrals', 0):
        raise RuntimeError('A complete accepted prefix was not retained')
    if archive.get('cpu_authority_bytes', 0) > cfg['kvmem']['cpu_read_cache_bytes']:
        raise RuntimeError('CPU authority exceeded the declared budget')
    admission = scheduler.get('cpu_history_admission') or {}
    if admission and admission['reserved_bytes'] > admission['capacity_bytes']:
        raise RuntimeError('Physical history reservations exceeded their budget')
    if archive.get('disk_read_bytes', 0) or archive.get('disk_write_bytes', 0):
        raise RuntimeError('Qualification must not spill KV to SSD')


def run(cfg, out, spec, tokenizer, *, namespace, server, bridge):
    validate_spec(cfg, spec)
    directory = out/'long-gc-preflight'
    directory.mkdir(exist_ok=False)
    capture = out/'server/capture'
    period = tokenizer.encode('A synthetic record describes an ordinary object, its location and a short observation. ',
                              add_special_tokens=False)
    output_token = next(t for t in period if t not in tokenizer.all_special_ids)
    release_token = tokenizer.convert_tokens_to_ids('<|im_end|>')
    sessions = [namespace+f'-long-gc-{i}' for i in range(spec['owners'])]
    save(directory/'fixture.json', dict(spec=spec, period=period, output_token=output_token,
        sessions=sessions, profile=cfg,
        scope='Synthetic actual full-prefix growth, constrained outputs; not Agent quality/throughput.'))
    before = read(capture/'archive.json')
    retired_before = (before.get('historical_gc') or {}).get('retired', 0)
    work_before = read(capture/'public_work.json')
    stop, lock = threading.Event(), threading.Lock()
    state = [dict(history=0, turns=0, done=False) for _ in sessions]
    started = time.time()
    minimum_free = psutil.virtual_memory().available
    peak_authority = peak_index = 0

    def post(body):
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",
            data=json.dumps(body, separators=(',', ':')).encode(),
            headers={'Content-Type':'application/json'})
        try:
            with http.open(request, timeout=1800) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Long-history HTTP {error.code}: '+error.read().decode(errors='replace')) from error

    def owner(index):
        sid = sessions[index]
        prefix, digest = array.array('I'), hashlib.sha256()
        folder = directory/f'owner-{index:02d}'
        folder.mkdir()
        with (folder/'tokens.u32le').open('xb') as journal, (folder/'requests.jsonl').open('x') as events:
            turn = 0
            while len(prefix) < spec['target_tokens']:
                if stop.is_set():
                    raise InterruptedError('Another owner or the resource guard failed')
                count = next_input_count(len(prefix), spec)
                appended = [period[(len(prefix)+i+index)%len(period)] for i in range(count)]
                raw = token_bytes(appended)
                prefix.extend(appended); journal.write(raw); journal.flush(); digest.update(raw)
                input_hash = digest.copy().hexdigest()
                body = dict(model=cfg['served_model_name'], prompt=prefix.tolist(),
                    session_id=sid, cache_salt=sid, max_tokens=spec['output_tokens'],
                    temperature=0, seed=1000, ignore_eos=True, return_token_ids=True,
                    allowed_token_ids=[output_token], kvmem_reselect='force',
                    kvmem_query_span=[max(0,len(prefix)-64),len(prefix)], kvmem_mandatory_spans=[])
                began = time.time()
                event = dict(turn=turn, input_tokens=len(prefix), appended_tokens=count,
                    input_sha256=input_hash, journal_input_bytes=4*len(prefix), started_at=began,
                    request_options={k:v for k,v in body.items() if k!='prompt'})
                try:
                    response = post(body)
                    tokens = response['choices'][0]['token_ids']
                    if (tokens != [output_token]*spec['output_tokens'] or
                            response['usage']['prompt_tokens'] != len(prefix) or
                            response['usage']['completion_tokens'] != spec['output_tokens']):
                        raise RuntimeError('Fixed output or full logical prompt accounting differs')
                    event.update(response=response, finished_at=time.time())
                except BaseException as error:
                    event.update(error=repr(error), finished_at=time.time())
                    events.write(json.dumps(event)+'\n'); events.flush()
                    stop.set()
                    raise
                events.write(json.dumps(event)+'\n'); events.flush()
                raw = token_bytes(tokens); prefix.extend(tokens)
                journal.write(raw); journal.flush(); digest.update(raw)
                turn += 1
                with lock:
                    state[index].update(history=len(prefix), turns=turn)
            result = dict(owner=index, history_tokens=len(prefix), turns=turn,
                journal_sha256=digest.hexdigest(), journal_bytes=4*len(prefix))
            save(folder/'result.json', result)
            with lock:
                state[index]['done'] = True
            return result

    error = None
    results = []
    executor = ThreadPoolExecutor(max_workers=spec['owners'])
    futures = [executor.submit(owner, i) for i in range(spec['owners'])]
    try:
        while True:
            if server.poll() is not None or bridge.poll() is not None:
                raise RuntimeError('Owned backend exited during long-history qualification')
            if (out.parent/'STOP').exists():
                raise InterruptedError('User stopped this finite queue')
            minimum_free = min(minimum_free, psutil.virtual_memory().available)
            if minimum_free < spec['minimum_free_host_bytes']:
                raise RuntimeError('Long-history qualification host headroom below 8GiB')
            archive, scheduler = read(capture/'archive.json'), read(capture/'scheduler_resources.json')
            check_snapshot(archive, scheduler, cfg)
            peak_authority = max(peak_authority, archive.get('cpu_authority_bytes', 0))
            peak_index = max(peak_index, archive.get('cpu_index_bytes', 0))
            retired = (archive.get('historical_gc') or {}).get('retired', 0)-retired_before
            with lock:
                owners = [dict(v) for v in state]
            progress = dict(status='running', at=time.time(), started_at=started,
                target_tokens=spec['target_tokens'], min_history=min(v['history'] for v in owners),
                max_history=max(v['history'] for v in owners), completed_owners=sum(v['done'] for v in owners),
                owners=owners, retired_blocks=retired, min_host_available_bytes=minimum_free,
                peak_cpu_authority_bytes=peak_authority, peak_cpu_index_bytes=peak_index)
            save(directory/'progress.json', progress)
            with (directory/'samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(**progress,
                    archive={k:v for k,v in archive.items() if type(v) in (int,float,bool)},
                    historical_gc=archive.get('historical_gc'),
                    cpu_admission=scheduler.get('cpu_history_admission'),
                    free_blocks=scheduler.get('free_blocks'), requests=scheduler.get('requests')))+'\n')
            for future in futures:
                if future.done() and future.exception() is not None:
                    raise future.exception()
            if all(f.done() for f in futures):
                results = [f.result() for f in futures]
                break
            time.sleep(2)
        # Let terminal publication and its capacity settlement finish before audit.
        deadline = time.monotonic()+120
        while True:
            scheduler = read(capture/'scheduler_resources.json')
            archive = read(capture/'archive.json')
            settled = {tuple(row['namespace']):row for row in scheduler.get('session_views', [])}
            if (not any(scheduler.get(k) for k in ('requests','reservations','refresh_pending','connector_pending'))
                    and all(settled.get((sid,sid),{}).get('history',0)>=spec['target_tokens']-1 for sid in sessions)):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Final accepted history did not settle')
            time.sleep(.1)
        owners = {tuple(row['namespace']):row for row in scheduler.get('session_views', [])}
        for sid in sessions:
            if owners.get((sid,sid),{}).get('history',0) < spec['target_tokens']-1:
                raise RuntimeError('Full logical terminal prefix was not retained: '+sid)
        for index, sid in enumerate(sessions):
            row = owners[(sid,sid)]
            with (directory/f'owner-{index:02d}'/'tokens.u32le').open('rb') as stream:
                raw = stream.read(row['history']*4)
            if (len(raw)!=row['history']*4 or hashlib.sha256(raw).hexdigest()!=row['prefix_sha256']):
                raise RuntimeError('Accepted prefix bytes differ from the append-only token journal')
        check_snapshot(archive, scheduler, cfg)
        gc = archive['historical_gc']
        retired = gc['retired']-retired_before
        minimum_retired = spec['owners']*spec['minimum_gc_cycles']*cfg['kvmem']['repository_blocks']
        if retired < minimum_retired:
            raise RuntimeError('Not enough actual repeated background collection')
        work_after = read(capture/'public_work.json')
        if work_before.get('counter_id')!=work_after.get('counter_id'):
            raise RuntimeError('Model engine changed during qualification')
        computed=(work_after['completed']['public_prefill_tokens']-
                  work_before['completed']['public_prefill_tokens'])
        incoming=sum(r['history_tokens']-r['turns']*spec['output_tokens'] for r in results)
        if not incoming <= computed <= incoming*1.01:
            raise RuntimeError('Public prefill indicates missing input or unexpected historical recomputation')
        audit = dict(status='passed', results=results, retired_blocks=retired,
            public_computed_prefill= computed, declared_new_input_tokens=incoming,
            min_host_available_bytes=minimum_free, peak_cpu_authority_bytes=peak_authority,
            peak_cpu_index_bytes=peak_index, before_work=work_before, after_work=work_after,
            settled_archive=archive, settled_scheduler=scheduler,
            seconds=time.time()-started, scope='Real eight-owner near-10M incremental lifecycle, fixed outputs only')
        save(directory/'execution-audit.json', audit)
        # The caller requires zero remaining checkpoints/history before tasks start.
        cleanup = []
        for sid in sessions:
            cleanup.append(post(dict(model=cfg['served_model_name'], prompt=[release_token],
                session_id=sid, cache_salt=sid, max_tokens=1, temperature=0,
                kvmem_reselect='off', kvmem_mandatory_spans=[],
                kv_transfer_params={'kvmem_release_session':True})))
        save(directory/'release-responses.json', cleanup)
        from scripts.benchmark_deepswe_first16 import wait_preflight_release
        save(directory/'cleanup.json', wait_preflight_release(capture))
        progress.update(status='passed', at=time.time(), retired_blocks=retired)
        save(directory/'progress.json', progress)
    except BaseException as exc:
        error = exc; stop.set()
        save(directory/'failure.json', dict(error=repr(exc), at=time.time(), original_attempt_preserved=True))
        progress = read(directory/'progress.json')
        progress.update(status='failed', error=repr(exc), at=time.time())
        save(directory/'progress.json', progress)
        raise
    finally:
        stop.set()
        # The owning controller will tear down its services on failure, unblocking
        # any other in-flight HTTP requests. No retry and no namespace takeover.
        executor.shutdown(wait=error is None, cancel_futures=True)
