#!/usr/bin/env python3
"""Re-score original RULER indices 0..9 per task without generating answers.

Historical full-context/native and sum/max/query-only results remain read-only.
This is a matched 130-case quality comparison, not a throughput comparison or
an estimate of the 260-case full-cohort score. No answer determines inclusion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import ruler_protocol as protocol


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = {
    'native_full_context': REPO/'artifacts/ruler-v1-256k-native-20261022-r1',
    'sum_graph': REPO/'artifacts/ruler-v1-256k-query-replay-20261022-r1',
    'max_graph': REPO/'artifacts/ruler-pooling-queue-20261022-r1/max',
    'query_only': REPO/'artifacts/ruler-pooling-queue-20261022-r1/query_only',
}
SEMANTIC_FIELDS = ('target_length', 'seed', 'tokenizer_sha256', 'upstream',
    'starts_in_thinking', 'sampling', 'prompt_protocol', 'corpus_manifest_sha256')
ROW_FIELDS = ('id', 'task', 'index', 'input', 'outputs', 'upstream_index',
    'served_input_tokens', 'served_query', 'question', 'served_sha256',
    'rendered_sha256', 'upstream_row_sha256', 'seed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def first10(fixture):
    """Use original per-task indices, never completion order or answer quality."""
    rows = fixture['rows']
    keys = [(r['task'], r['index']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate source case identity')
    chosen = [r for r in rows if type(r['index']) is int and 0 <= r['index'] < 10]
    protocol.validate_cohort(chosen, 10)
    require({(r['task'], r['index']) for r in chosen} ==
        {(task, i) for task in protocol.TASK_FAMILIES for i in range(10)},
        'Expected the original first ten indices of every task')
    return sorted(chosen, key=lambda r: (list(protocol.TASK_FAMILIES).index(r['task']), r['index']))


def fixture_signature(fixture):
    rows = first10(fixture)
    return dict(metadata={key: fixture[key] for key in SEMANTIC_FIELDS},
        rows={r['id']: json_sha({key: r[key] for key in ROW_FIELDS}) for r in rows})


def validate_request(request, row, fixture):
    expected = dict(fixture['sampling'], seed=row['seed'])
    if 'sampling' in request:
        require(request['sampling'] == expected, 'Recorded sampling changed: '+row['id'])
        if 'prompt_sha256' in request:
            # The CPU-FPS controller records the same explicit nested sampling
            # as native, plus the original public Query span used by KVMem.
            require('input_sha256' not in request and 'input_tokens' not in request,
                'Ambiguous candidate input proof')
            prompt_sha = request['prompt_sha256']
            prompt_tokens = request['prompt_tokens']
            require(request['query_span'] == row['served_query'], 'Public Query changed: '+row['id'])
        else:
            prompt_sha = request['input_sha256']
            prompt_tokens = request['input_tokens']
    else:
        require(all(request.get(k) == v for k, v in expected.items()),
            'KVMem sampling changed: '+row['id'])
        prompt_sha = request['prompt_sha256']
        prompt_tokens = request['prompt_tokens']
        require(request['kvmem_query_span'] == row['served_query'], 'Public Query changed: '+row['id'])
    require(prompt_sha == row['served_sha256'] and prompt_tokens == row['served_input_tokens'],
        'Actual served input differs: '+row['id'])


def validate_answer(row, fixture, request, response, result, completed_result, scorer):
    validate_request(request, row, fixture)
    require(result == completed_result, 'Per-case result and completed manifest differ: '+row['id'])
    require(result['status'] == 'completed' and result['id'] == row['id'] and
        result['task'] == row['task'] and result['index'] == row['index'], 'Incomplete/misbound result')
    require(result['input_sha256'] == row['served_sha256'] and
        result['input_tokens'] == row['served_input_tokens'], 'Result input differs')
    require(result['text'] == response['text'] and
        result['completion_tokens'] == len(response['output_ids']) and
        result['response_id'] == response['response_id'], 'Response/result text or token count differs')
    answer = protocol.final_answer(response['text'], starts_in_thinking=fixture['starts_in_thinking'])
    require(result['answer'] == answer['text'] and
        result['thinking_complete'] == answer['thinking_complete'], 'Final answer extraction differs')
    metrics, clean = scorer
    metric = metrics[protocol.TASK_FAMILIES[row['task']]]
    cleaned = clean(answer['text'], metric)
    score = metric['metric_fn']([cleaned], [row['outputs']])
    require(score == result['official_case_score'], 'Historical official case score differs')
    return dict(id=row['id'], task=row['task'], index=row['index'],
        official_case_score=score, thinking_complete=answer['thinking_complete'],
        answer_sha256=hashlib.sha256(answer['text'].encode()).hexdigest(),
        input_sha256=row['served_sha256'], seed=row['seed'])


def extract_reference(root, expected_signature=None):
    root = Path(root)
    fixture = read(root/'fixture.json')
    signature = fixture_signature(fixture)
    if expected_signature is not None:
        require(signature == expected_signature, 'Historical fixture differs from the fixed 130-case cohort')
    rows = first10(fixture)
    status, completed = read(root/'status.json'), read(root/'run/answers/completed.json')
    require(status['status'] == completed['status'] == 'completed' and completed['protocol_passed'] is True,
        'Historical campaign has not completed its execution audit')
    fixture_hash = sha(root/'fixture.json')
    require(completed['fixture_sha256'] == fixture_hash, 'Historical fixture hash changed')
    require(completed['cases'] == len(fixture['rows']) == len(completed['results']), 'Incomplete historical cohort')
    indexed = {r['id']: r for r in completed['results']}
    require(len(indexed) == len(completed['results']), 'Duplicate completed result identity')
    scorer = protocol.load_scorer(fixture['upstream_path'])
    full_scores = protocol.score_suite(fixture['rows'],
        {(r['task'], r['index']): r['text'] for r in completed['results']}, scorer,
        samples_per_task=fixture['samples_per_task'], starts_in_thinking=fixture['starts_in_thinking'])
    require(full_scores == read(root/'run/answers/scores.json'), 'Stored full-cohort scores fail re-scoring')
    require(full_scores['macro_score'] == completed['score'] == status['score'], 'Full-cohort score summary differs')
    files = {name: sha(root/name) for name in ('fixture.json', 'status.json',
        'run/answers/completed.json', 'run/answers/scores.json')}
    predictions, cases = {}, []
    for row in rows:
        directory = root/'run/answers/cases'/row['id']
        request, response, result = (read(directory/name) for name in ('request.json', 'response.json', 'result.json'))
        item = validate_answer(row, fixture, request, response, result, indexed[row['id']], scorer)
        if (directory/'hashes.json').exists():
            manifest = read(directory/'hashes.json')
            require({'request.json', 'response.json', 'result.json', 'evidence.json'} <= set(manifest),
                'Case immutable evidence manifest is incomplete')
            for name, expected in manifest.items():
                require(Path(name).name == name and sha(directory/name) == expected,
                    'Case immutable evidence hash changed: '+row['id']+'/'+name)
        # Snapshot every existing JSON evidence file without rewriting any source.
        item['files'] = {str(path.relative_to(root)): sha(path) for path in sorted(directory.glob('*.json'))}
        cases.append(item)
        predictions[(row['task'], row['index'])] = response['text']
    scores = protocol.score_suite(rows, predictions, scorer, samples_per_task=10,
        starts_in_thinking=fixture['starts_in_thinking'])
    profile_name = 'reference-profile.json' if (root/'reference-profile.json').exists() else 'profile.json'
    profile = read(root/profile_name)
    files[profile_name] = sha(root/profile_name)
    return dict(root=str(root.resolve()), scores=scores, cases=cases, source_files=files,
        full_cohort_macro_score=full_scores['macro_score'], profile=profile,
        actual_historical_workers=completed.get('workers', completed.get('queue_peak_inflight'))), signature


def compare(references, *, fixture_path=None):
    require(bool(references), 'No historical references provided')
    expected = fixture_signature(read(fixture_path)) if fixture_path is not None else None
    result = dict(scope='Original indices 0..9 per each of 13 RULER v1 tasks; 130 cases. '
        'Frozen official final-answer metrics. No generation, answer-based selection, or retries.',
        caveats=['Historical runtime revisions and batching differ; this is not a controlled throughput comparison.',
            'Native uses full context; KVMem methods use bounded selected history.',
            'Original sum prefill coalescing differs from the max/query-only campaign.'],
        fixture_proof=({'path': str(Path(fixture_path).resolve()), 'sha256': sha(fixture_path)}
            if fixture_path is not None else None), methods={})
    for label, root in references.items():
        method, signature = extract_reference(root, expected)
        if expected is None:
            expected = signature
        result['methods'][label] = method
    result['cohort_signature'] = expected
    return result


def report(comparison):
    methods = comparison['methods']
    rows = ['# Original first 10 cases per RULER task', '', comparison['scope'], '',
        '| Task | '+' | '.join(methods)+' |', '|---|'+'---:|'*len(methods)]
    for task in protocol.TASK_FAMILIES:
        rows.append('| '+task+' | '+' | '.join(f"{m['scores']['per_task'][task]['score']:.2f}"
            for m in methods.values())+' |')
    rows.append('| **13-task macro** | '+' | '.join(f"**{m['scores']['macro_score']:.4f}**"
        for m in methods.values())+' |')
    rows.extend(['', *comparison['caveats'], ''])
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, help='Optional new 130-case fixture; verify exact rows, seeds and sampling')
    parser.add_argument('--reference', action='append', default=[], metavar='NAME=ROOT')
    parser.add_argument('--candidate', type=Path, help='Append the completed CPU FPS + halo 130-case campaign')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; existing directories are refused')
    args = parser.parse_args()
    references = dict(DEFAULT_REFERENCES)
    if args.reference:
        references = {}
        for item in args.reference:
            name, separator, path = item.partition('=')
            require(separator and name and path and name not in references, 'Use unique NAME=ROOT references')
            references[name] = Path(path)
    if args.candidate:
        require('cpu_fps_halo' not in references, 'Duplicate candidate reference')
        references['cpu_fps_halo'] = args.candidate
    require(not args.output.exists(), 'Use a fresh comparison output directory')
    comparison = compare(references, fixture_path=args.fixture)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')
    (args.output/'report.md').write_text(report(comparison))
    print(json.dumps({name: item['scores']['macro_score'] for name, item in comparison['methods'].items()}))


if __name__ == '__main__':
    main()
