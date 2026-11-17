"""Greedy verbatim-copy check at several concurrencies (verify-graph corruption detector).

Each owner asks the model to copy a distinct ~1,500-token passage verbatim. A stale
verification plan shows up as doubled pieces ("ReadRead", "the the") and a broken copy,
so the per-request copy accuracy separates a correct verify path from a stale one.
Concurrencies between captured graph sizes (3, 5, 6, 7, ...) exercise FULL-graph
request padding. Writes cases/copy-c{N}/summary.json and copy-summary.json.
"""
import concurrent.futures
import json
import random
import re
import threading
import time
import urllib.error
import uuid

from benchmark import TOKENIZER, http, save

WORDS = ['cache', 'request', 'scheduler', 'memory', 'latency', 'kernel', 'archive', 'service',
         'worker', 'capacity', 'queue', 'session', 'prompt', 'output', 'batch', 'Read', 'parameter',
         'function', 'value', 'thread', 'buffer', 'page', 'graph', 'token']
DOUBLED = re.compile(r'\b(\w{2,})\1\b|\b(\w+) \2\b')


def passage(seed, lines):
    rng = random.Random(seed)
    out = []
    for i in range(lines):
        w = rng.sample(WORDS, 6)
        out.append(f'Entry {seed}-{i}: <{w[0]}> {w[1]}={rng.randrange(1, 99999)}; the {w[2]} '
                   f'handler read {rng.randrange(2, 512)} {w[3]} items, then {w[4]} called {w[5]}().')
    return '\n'.join(out)


def run(root, args, build_unused):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    overall = []
    for concurrency in map(int, args.copy_check.split(',')):
        directory = root/'cases'/f'copy-c{concurrency}'
        directory.mkdir(parents=True, exist_ok=False)
        barrier = threading.Barrier(concurrency)

        def one(index):
            text = passage(1000 * concurrency + index, args.copy_lines)
            messages = [{'role': 'system', 'content': 'You copy text exactly.'},
                        {'role': 'user', 'content': 'Copy the following text exactly, character for '
                         'character, with no other words before or after it.\n\n' + text}]
            template = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            ids = tok.encode(template, add_special_tokens=False)
            assert ids and all(isinstance(i, int) for i in ids)
            target = len(tok.encode(text, add_special_tokens=False))
            session = 'copy-' + uuid.uuid4().hex
            body = dict(model='kvmem-throughput', prompt=ids, max_tokens=target + 64, temperature=0,
                        cache_salt=session, session_id=session, kvmem_query_span=[len(ids) - 8, len(ids)],
                        kvmem_reselect='auto', kvmem_mandatory_parts='none')
            barrier.wait()
            start = time.perf_counter()
            try:
                with http('/v1/completions', body, timeout=1200) as response:
                    reply = json.loads(response.read())
            except urllib.error.HTTPError as error:
                raise RuntimeError(f'HTTP {error.code}: {error.read().decode()[:2000]}') from None
            got = reply['choices'][0]['text'].strip()
            common = next((i for i, (a, b) in enumerate(zip(got, text)) if a != b), min(len(got), len(text)))
            doubled = max(0, len(DOUBLED.findall(got)) - len(DOUBLED.findall(text)))
            row = dict(index=index, exact=got == text, prefix_fraction=common / len(text),
                       doubled_markers=doubled, output_chars=len(got), source_chars=len(text),
                       completion_tokens=reply['usage']['completion_tokens'], seconds=time.perf_counter() - start,
                       first_divergence=dict(source=text[common:common + 80], output=got[common:common + 80]))
            save(directory/f'c{index + 1}.json', dict(row, output=got))
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, range(concurrency)))
        summary = dict(concurrency=concurrency, exact=sum(r['exact'] for r in rows),
                       mean_prefix_fraction=sum(r['prefix_fraction'] for r in rows) / len(rows),
                       min_prefix_fraction=min(r['prefix_fraction'] for r in rows),
                       doubled_markers=sum(r['doubled_markers'] for r in rows), rows=rows)
        save(directory/'summary.json', summary)
        overall.append({k: v for k, v in summary.items() if k != 'rows'})
        print(json.dumps(overall[-1]), flush=True)
    save(root/'copy-summary.json', overall)
