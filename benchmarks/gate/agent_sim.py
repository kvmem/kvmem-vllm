"""Agent simulator: S sessions alternating model turns and tool waits on one server.

Phase 1 (grow): every session sends one request that brings its history to --history tokens
(a user message; for KVMem it is also the agent Query span), then all sessions meet at a barrier.
Phase 2 (steady, --duration seconds): each session loops
    tool wait ~ lognormal(--tool-median, --tool-sigma) seconds (capped at --tool-cap)
    -> one model request: previous context + tool result (~ lognormal(--append-median, --append-sigma)
       tokens) -> --output generated tokens (ignore_eos, greedy).
Generated token ids are appended verbatim, so a prefix cache can always reuse the previous turn.
--backend kvmem sends the KVMem session fields (session_id/cache_salt/query span) and releases the
sessions at the end; --backend native sends plain completions (prefix caching does the reuse).
Writes OUT/client.json: per-request rows and a summary of the steady window.

--shared-prefix P (kvmem, docs/prefix-sharing.md): every session's grow prompt starts with one shared
system prompt of P tokens (rounded down to --block) and all sessions use the cache_salt --tag. The
per-session text after it is unchanged, so arms differ only in how the shared prompt is reached:
  cold    each session prefills it itself (the baseline);
  warm    each session first sends the registration-shaped request (shared prompt + --query tokens,
          max_tokens 1) on its own session, then its grow request resumes from that frozen P;
  import  one registration session '<tag>-prefix' sends that request with kvmem_share_prefix = P
          before the sessions start (then --register-wait seconds); each grow request imports it.
The registration and warm requests are not rows (summary.prefix has their timings), so the rows of
any two arms compare request by request.
"""
import argparse, json, math, random, threading, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True); ap.add_argument('--model', required=True)
ap.add_argument('--tokenizer', required=True); ap.add_argument('--backend', choices=('kvmem', 'native'), required=True)
ap.add_argument('--sessions', type=int, default=8); ap.add_argument('--history', type=int, default=64000)
ap.add_argument('--duration', type=float, default=600); ap.add_argument('--output', type=int, default=320)
ap.add_argument('--tool-median', type=float, default=0.9); ap.add_argument('--tool-sigma', type=float, default=1.81)
ap.add_argument('--tool-cap', type=float, default=120.0)
ap.add_argument('--tool-mix', default='')   # 'weight:median:sigma[:cap],...' tool classes (overrides median/sigma)
ap.add_argument('--tool-mix-sticky', action='store_true')   # each session keeps one tool class for the run
ap.add_argument('--append-median', type=float, default=300); ap.add_argument('--append-sigma', type=float, default=1.0)
ap.add_argument('--append-cap', type=int, default=6000); ap.add_argument('--query', type=int, default=32)
ap.add_argument('--grow-concurrency', type=int, default=4); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--tag', default='sim'); ap.add_argument('--out', required=True)
ap.add_argument('--save-tokens', action='store_true')   # exactness checks: keep generated ids per request
ap.add_argument('--max-turns', type=int, default=0)    # stop each session after this many turns (0: duration only)
ap.add_argument('--shared-prefix', type=int, default=0)   # tokens of one system prompt shared by every session
ap.add_argument('--prefix-mode', choices=('cold', 'warm', 'import'), default='cold')
ap.add_argument('--block', type=int, default=32)          # KVMem block_tokens: the shared prompt is aligned to it
ap.add_argument('--register-wait', type=float, default=3.0)
args = ap.parse_args()
if args.shared_prefix and args.backend != 'kvmem':
    ap.error('--shared-prefix needs --backend kvmem')
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.tokenizer)
words = tok.encode('An agent inspects a repository, runs a test, reads the failing traceback, edits a '
                   'function and reports what changed in the module and why the fix is correct. ',
                   add_special_tokens=False)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
rows, lock, errors = [], threading.Lock(), []
grow_gate = threading.Semaphore(args.grow_concurrency)
barrier = threading.Barrier(args.sessions)
window = {}
prefix_info = {}


def post(body):
    req = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=7200) as response:
        return json.load(response)


def text(rnd, n, salt):
    marker = tok.encode(f' [{salt}] ', add_special_tokens=False)
    start = rnd.randrange(len(words))
    return marker + [words[(start + i) % len(words)] for i in range(max(0, n - len(marker)))]


def kvmem_fields(sid, query, query_id):
    return dict(session_id=sid, cache_salt=args.tag if shared else sid, kvmem_reselect='auto',
                kvmem_mandatory_spans=[], kvmem_mandatory_parts='none', kvmem_query_span=list(query),
                kvmem_query_mode='agent', kvmem_query_id=query_id)


def register_request(sid, share):
    """The registration shape: the shared prompt, then a short Query starting at P; max_tokens 1."""
    body = dict(model=args.model, prompt=shared + register_query, max_tokens=1, temperature=0, ignore_eos=True,
                **kvmem_fields(sid, [len(shared), len(shared) + len(register_query)], f'{sid}-prefix'))
    if share:
        body['kv_transfer_params'] = {'kvmem_share_prefix': len(shared)}
    started = time.time()
    post(body)
    return time.time() - started


def release(sid, last):
    post(dict(model=args.model, prompt=last, session_id=sid, cache_salt=args.tag if shared else sid, max_tokens=1,
              temperature=0, kvmem_reselect='off', kvmem_mandatory_spans=[],
              kv_transfer_params={'kvmem_release_session': True}))


def session(index):
    sid = f'{args.tag}-{index}'
    rnd = random.Random(args.seed * 1000 + index)
    prefix, query = [], None

    def request(new, output, kind):
        nonlocal prefix, query
        prefix = prefix + new
        if kind == 'grow':
            query = [len(prefix) - min(args.query, len(new)), len(prefix)]
        body = dict(model=args.model, prompt=prefix, max_tokens=output, temperature=0, ignore_eos=True,
                    return_token_ids=True)
        if args.backend == 'kvmem':
            body.update(kvmem_fields(sid, query, f'{sid}-q0'))
        started = time.time()
        choice = post(body)['choices'][0]
        ended = time.time()
        generated = list(choice.get('token_ids') or [])
        with lock:
            rows.append(dict(session=index, kind=kind, prompt_tokens=len(prefix), new_tokens=len(new),
                             output_tokens=len(generated), started=started, ended=ended,
                             **(dict(tokens=generated) if args.save_tokens else {})))
        prefix = prefix + generated

    try:
        with grow_gate:
            if shared and args.prefix_mode == 'warm':
                seconds = register_request(sid, False)
                with lock:
                    prefix_info.setdefault('warm_seconds', {})[index] = seconds
            own = text(rnd, args.history - len(shared), f'system s{index}')
            request(shared + own if shared else own, 16, 'grow')
        barrier.wait(timeout=7200)
        with lock:
            window.setdefault('start', time.time())
        stop = window['start'] + args.duration
        turns = 0
        sticky = []
        while True:
            if args.max_turns and turns >= args.max_turns:
                break
            turns += 1
            if args.tool_mix:
                classes = [tuple(float(x) for x in c.replace('~', ':').split(':')) for c in args.tool_mix.replace('/', ',').split(',')]
                if args.tool_mix_sticky and sticky:
                    c = sticky[0]
                else:
                    pick, acc = rnd.random() * sum(c[0] for c in classes), 0.0
                    for c in classes:
                        acc += c[0]
                        if pick <= acc:
                            break
                    sticky[:] = [c]
                cap = c[3] if len(c) > 3 else args.tool_cap
                wait = min(cap, math.exp(math.log(c[1]) + c[2] * rnd.gauss(0, 1)))
            else:
                wait = min(args.tool_cap, math.exp(math.log(args.tool_median) + args.tool_sigma * rnd.gauss(0, 1))) \
                    if args.tool_median > 0 else 0.0
            if time.time() + wait >= stop:
                break
            time.sleep(wait)
            size = int(min(args.append_cap, max(16, math.exp(math.log(args.append_median) +
                                                             args.append_sigma * rnd.gauss(0, 1)))))
            request(text(rnd, size, f'tool s{index}'), args.output, 'turn')
            if time.time() >= stop:
                break
    except Exception as error:  # noqa: BLE001 - record and keep the other sessions running
        errors.append(f'{sid}: {error!r}')
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        if args.backend == 'kvmem' and prefix:
            try:
                release(sid, prefix[-1:])
            except Exception as error:  # noqa: BLE001
                errors.append(f'{sid} release: {error!r}')


shared, register_query = [], []
if args.shared_prefix:
    shared = text(random.Random(args.seed * 1000 - 1), args.shared_prefix // args.block * args.block, 'system shared')
    register_query = text(random.Random(args.seed * 1000 - 2), args.query, 'register')
    if not 0 < len(shared) < args.history:
        ap.error('--shared-prefix must hold at least one block and stay below --history')
    prefix_info.update(tokens=len(shared), mode=args.prefix_mode)
    if args.prefix_mode == 'import':
        prefix_info['register_seconds'] = register_request(f'{args.tag}-prefix', True)
        time.sleep(args.register_wait)   # the worker registers the image when the request has retired
threads = [threading.Thread(target=session, args=(i,)) for i in range(args.sessions)]
began = time.time()
for t in threads: t.start()
for t in threads: t.join()
if shared and args.prefix_mode == 'import':
    try:
        release(f'{args.tag}-prefix', register_query[-1:])
    except Exception as error:  # noqa: BLE001
        errors.append(f'prefix release: {error!r}')
w0 = window.get('start', began); w1 = w0 + args.duration
turns = [r for r in rows if r['kind'] == 'turn']
inside = [r for r in turns if w0 <= r['started'] and r['ended'] <= w1 + 0.0]
lat = sorted(r['ended'] - r['started'] for r in inside)
q = lambda p: lat[int(p * (len(lat) - 1))] if lat else None
summary = dict(backend=args.backend, sessions=args.sessions, history=args.history, duration=args.duration,
               tool_median=args.tool_median, tool_sigma=args.tool_sigma, tool_mix=args.tool_mix, tool_mix_sticky=args.tool_mix_sticky, append_median=args.append_median,
               output=args.output, turns=len(inside), errors=errors,
               output_tok_s=sum(r['output_tokens'] for r in inside) / args.duration,
               new_input_tok_s=sum(r['new_tokens'] for r in inside) / args.duration,
               turns_per_min=60 * len(inside) / args.duration,
               turn_latency_p50=q(.5), turn_latency_p90=q(.9), turn_latency_p99=q(.99),
               grow_seconds=w0 - began, window=[w0, w1])
if shared:
    summary['prefix'] = prefix_info
(out / 'client.json').write_text(json.dumps(dict(summary=summary, rows=rows)))
print('SUMMARY', json.dumps(summary, indent=1), flush=True)
