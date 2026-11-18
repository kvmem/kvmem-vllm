"""Serving check for historical_gc.policy "tool_result_age" against an owned, idle server.

One session (stable session_id and cache_salt) receives one complete tool reply per
request, each with an exact `tool` locality span, and keeps the model's accepted output
between turns so every turn reuses the exact session prefix. Reply 1 is long, so part of
it lies outside both the protected sink prefix and the GPU window.

Checks, with free capacity (no pressure):
- through reply `steps` (12): no candidate (shadow_decisions and tool_result_age_retired
  stay 0);
- after reply steps+1 (13): reply-1 pages become candidates (shadow: shadow_decisions > 0;
  delete: tool_result_age_retired > 0 and cpu_released follows);
- delete mode: every retired page lies wholly inside an expired tool reply, outside the sink
  prefix (thinking, assistant and boundary-straddling pages stay);
- later replies move the window: reply-1 pages that were protected by the window or a
  checkpoint are retired after that protection ends.
Excluded from benchmark scores.
"""
import argparse
import json
from pathlib import Path
import time
import uuid

import httpx

STATS = ('published', 'decisions', 'retired', 'shadow_decisions', 'tool_result_age_retired', 'cpu_released',
         'rejected_ineligible', 'policy_failures', 'lru_fallback_retired', 'redundancy_retired')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def gc_stats(capture):
    gc = read(capture/'archive.json').get('historical_gc') or {}
    return {key: gc.get(key, 0) for key in STATS}


def evictions(capture):
    rows = []
    path = capture/'historical_gc_evictions.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def probe(client, model, end_token):
    """One engine step that creates no capture: a release on a fresh session id.

    With KVMEM_LAZY_ARCHIVE_REPORT (default on) archive.json is written only on
    engine steps, so background retirements that finish after a request become
    visible at the next step. The probe forces that step.
    """
    name = 'tool-age-probe-' + uuid.uuid4().hex
    client.post('/v1/completions', json=dict(model=model, session_id=name, cache_salt=name, prompt=[end_token],
                max_tokens=1, kv_transfer_params=dict(kvmem_release_session=True))).raise_for_status()


def settle(client, model, end_token, capture, seconds):
    """Probe until the GC counters are unchanged over three probes 1 s apart (bounded)."""
    probe(client, model, end_token)
    last, same, deadline = gc_stats(capture), 0, time.monotonic() + seconds
    while time.monotonic() < deadline and same < 3:
        time.sleep(1)
        probe(client, model, end_token)
        now = gc_stats(capture)
        same = same + 1 if now == last else 0
        last = now
    return last


def reply_text(index, tokens, encode):
    unit = (f'[tool reply {index}] build log line: module_{index}.c compiled; checksum {index * 7919 % 100003}; '
            f'warnings 0; elapsed {index}.{index % 7}s\n')
    repeats = max(1, tokens // len(encode(unit)))
    return unit * repeats


def main(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    block, sink = args.block_tokens, args.sink_blocks * args.block_tokens
    args.output.mkdir(parents=True, exist_ok=False)
    session = 'tool-age-check-' + uuid.uuid4().hex
    question = 'Build the project, inspect each tool result, and report which modules changed.'
    tokens = encode('<|im_start|>system\nYou are a coding agent that runs tools.<|im_end|>\n<|im_start|>user\n')
    query_begin = len(tokens)
    tokens += encode(question)
    spans = [dict(begin=0, end=len(tokens) + len(encode('<|im_end|>\n')), role='user')]
    query = [query_begin, len(tokens)]
    tokens += encode('<|im_end|>\n<|im_start|>assistant\n')
    tool_spans, rows, retired_begins = [], [], set()
    end_token = encode('<|im_end|>')[0]
    total = args.steps + 1 + args.extra_replies
    with httpx.Client(base_url=args.base, trust_env=False, timeout=3600) as client:
        before = settle(client, args.model, end_token, args.capture, args.settle_seconds)   # fresh baseline
        seen_evictions = len(evictions(args.capture))
        for k in range(1, total + 1):
            # Accepted assistant output of the previous turn, closed; then tool reply k.
            if k > 1:
                closure = encode('<|im_end|>\n')
                spans.append(dict(begin=assistant_begin, end=len(tokens) + len(closure), role='assistant'))
                tokens += closure
            begin = len(tokens)
            tokens += encode('<|im_start|>user\n<tool_response>\n')
            tokens += encode(reply_text(k, args.first_reply_tokens if k == 1 else args.reply_tokens, encode))
            tokens += encode('\n</tool_response><|im_end|>\n')
            spans.append(dict(begin=begin, end=len(tokens), role='tool'))
            tool_spans.append((begin, len(tokens)))
            tokens += encode('<|im_start|>assistant\n')
            assistant_begin = len(tokens)
            request = dict(model=args.model, session_id=session, cache_salt=session, prompt=tokens,
                           max_tokens=args.max_tokens, temperature=0, seed=1000, return_token_ids=True,
                           kvmem_locality_spans=spans, kvmem_query_span=query, kvmem_reselect='auto',
                           kvmem_mandatory_parts='none', kvmem_mandatory_spans=[])
            started = time.perf_counter()
            response = client.post('/v1/completions', json=request)
            if response.status_code != 200:
                save(args.output/'failure.json', dict(step=k, status=response.status_code, body=response.text[:4000]))
                raise RuntimeError(f'Request {k} failed: {response.status_code}')
            choice = response.json()['choices'][0]
            output = choice.get('token_ids') or []
            if not output:
                raise RuntimeError('Response omitted accepted token IDs')
            tokens += output
            stats = settle(client, args.model, end_token, args.capture, args.settle_seconds)
            new = evictions(args.capture)[seen_evictions:]
            for row in new:
                retired_begins.update(row.get('history_begins') or ())
            seen_evictions += len(new)
            delta = {key: stats[key] - before[key] for key in STATS}
            rows.append(dict(reply=k, history=len(tokens), seconds=round(time.perf_counter() - started, 2), **delta))
            print(json.dumps(rows[-1]), flush=True)
        release = client.post('/v1/completions', json=dict(model=args.model, session_id=session, cache_salt=session,
            prompt=encode('<|im_end|>'), max_tokens=1, kv_transfer_params=dict(kvmem_release_session=True)))
    # Expected eligible pages: whole pages inside each expired reply, after the sink prefix.
    def pages(index):
        b, e = tool_spans[index]
        b = max((b + block - 1) // block * block, (sink + block - 1) // block * block)
        e = e // block * block
        return set(range(b, e, block)) if b < e else set()
    expired_after = lambda k: max(0, k - args.steps)   # replies expired after reply k
    first = pages(0)
    at = {row['reply']: row for row in rows}
    candidates = (lambda row: row['shadow_decisions']) if args.mode == 'shadow' else (lambda row: row['tool_result_age_retired'])
    eligible_now = set().union(*(pages(i) for i in range(expired_after(total)))) if expired_after(total) else set()
    checks = dict(
        no_candidate_through_reply_12=all(candidates(at[k]) == 0 for k in range(1, args.steps + 1)),
        candidates_after_reply_13=candidates(at[args.steps + 1]) > 0,
        reply1_has_eligible_pages=len(first) > 0,
        no_pressure_deletions=all(at[k]['lru_fallback_retired'] == 0 and at[k]['redundancy_retired'] == 0 for k in at),
        no_policy_failures=all(at[k]['policy_failures'] == 0 for k in at))
    if args.mode == 'shadow':
        checks.update(nothing_retired=all(at[k]['retired'] == 0 for k in at))
    else:
        checks.update(
            retired_only_expired_tool_pages=retired_begins <= eligible_now,
            retired_counts_match=at[total]['retired'] == at[total]['tool_result_age_retired'],
            cpu_released_followed=at[total]['cpu_released'] >= at[total]['tool_result_age_retired'],
            reply1_retired_grows_after_13=len(retired_begins & first) >= 1)
    result = dict(passed=all(checks.values()), checks=checks, mode=args.mode, steps=args.steps,
                  session=session, prompt_tokens=len(tokens), tool_spans=tool_spans,
                  reply1_eligible_pages=len(first), retired_reply1_pages=len(retired_begins & first),
                  retired_pages=len(retired_begins), eligible_after_last=len(eligible_now),
                  release_status=release.status_code, rows=rows)
    save(args.output/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    if not result['passed']:
        raise SystemExit('tool_result_age check failed; see result.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--model', required=True, help='served model name')
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['shadow', 'delete'], required=True)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--extra-replies', type=int, default=3, help='replies after steps+1 (window moves)')
    p.add_argument('--first-reply-tokens', type=int, default=30000)
    p.add_argument('--reply-tokens', type=int, default=1500)
    p.add_argument('--max-tokens', type=int, default=32)
    p.add_argument('--block-tokens', type=int, default=32)
    p.add_argument('--sink-blocks', type=int, default=41)
    p.add_argument('--settle-seconds', type=float, default=30)
    main(p.parse_args())
