"""Pure accepted decode with every owner decoding together (no re-prefill rounds).

One cold submission per owner with a long fixed output. The common window starts when
the last owner emits its first token (no prefill remains anywhere) and ends when the
first owner finishes; tokens accepted inside it over its length is the aggregate pure
decode rate at that concurrency.
"""
import concurrent.futures
import json
import threading
import time

from benchmark import metrics, release_session, request_one, save
from concurrency_workload import window


def run(root, args, build):
    for item in args.sweep_cases.split(','):
        length, concurrency = map(int, item.split(':'))
        assert 0 < concurrency <= args.seqs and length + 1 + args.pure_decode <= args.max_len
        directory = root/'cases'/f'pure{length}-out{args.pure_decode}-c{concurrency}'
        directory.mkdir(parents=True, exist_ok=False)
        ids, span = build(length)
        barrier = threading.Barrier(concurrency)
        before = metrics()
        begin = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request_one, directory/f'c{i+1}', ids, span, args.pure_decode, barrier,
                                   use_kvmem=not args.native)
                       for i in range(concurrency)]
            rows = [f.result() for f in futures]
        seconds = time.perf_counter() - begin
        after = metrics()
        save(directory/'metrics.json', dict(before=before, after=after,
            delta={k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}))
        result = dict(case=directory.name, initial_input_tokens=length, submitted_requests=concurrency,
            wall_seconds=seconds, output_tokens=sum(r['output_tokens'] for r in rows),
            token_accounting_passed=all(r['token_accounting_passed'] for r in rows),
            mean_ttft_seconds=sum(r['ttft_seconds'] for r in rows) / len(rows),
            max_ttft_seconds=max(r['ttft_seconds'] for r in rows), **window(rows),
            scope='Cold submission with a long output; common window = every owner decoding, '
                  'after the last first token and before the first finish (no prefill inside).')
        save(directory/'summary.json', result)
        print(json.dumps(result), flush=True)
        if not args.native:
            for i, row in enumerate(rows, start=1):
                release_session(row, directory/f'c{i}')
