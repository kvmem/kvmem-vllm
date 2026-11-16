"""Score frozen guided pairs with the same Pro judge and original task prompts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import analyze_memory
from benchmark_memory import source
from benchmark_client_io import prompt_matches
from benchmark_niah import save, sha
from graph_v2_evaluation import validate_evaluation
from guided_benchmark_protocol import require, validate_pair


def load_lme(root):
    folder = root/'evaluation/evaluators'
    path = folder/'historical_lme_score.py'
    spec = importlib.util.spec_from_file_location('frozen_lme_pro_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OFFICIAL = folder/'historical_lme_evaluate_qa.py'
    return module


def score_lme(root, arm, key):
    directory = root/f'longmemeval_s-{arm}/answers'
    output = directory/'scoring-pro'
    manifest = json.loads((directory/'manifest.json').read_text())
    profile = root/f'longmemeval_s-{arm}.json'
    require(sha(profile) == manifest['server_profile_sha256'], 'Profile changed after inference')
    judge = load_lme(root)
    check = judge.prompt_function()
    parameters = {**judge.PARAMETERS, 'model': 'deepseek-v4-pro'}
    client = judge.Client(key)
    identity = dict(api_base=judge.API_BASE, parameters=parameters,
        prompt_sha256=sha(judge.OFFICIAL),
        original_scorer_sha256=sha(root/'evaluation/evaluators/historical_lme_score.py'),
        adapter_sha256=sha(Path(__file__)),
        difference_from_historical_lme='Pro replaces Flash in both arms; no Flash judgments reused')
    if (output/'identity.json').exists():
        require(json.loads((output/'identity.json').read_text()) == identity, 'Judge identity changed')
    save(output/'identity.json', identity)

    def assess(body, path):
        save(path.with_suffix('.request.json'), body)
        raw, timing = client.request('/chat/completions', body)
        save(path.with_suffix('.response.json'), dict(response=raw, **timing))
        model = raw.get('model', '').lower()
        require(model in ('deepseek-v4-pro', 'deepseek-v4.1-pro'), 'Unexpected Pro judge version: '+model)
        return judge.verdict(raw), model

    calibration = output/'calibration.json'
    if not calibration.exists():
        fixtures = json.loads((root/'evaluation/evaluators/historical_lme_calibration.json').read_text())['cases']
        checks = []
        for i, fixture in enumerate(fixtures):
            ref = fixture['fixture']
            prompt = check(ref['question_type'], ref['question'], ref['answer'],
                           fixture['hypothesis'], abstention=ref['question_id'].endswith('_abs'))
            result, model = assess({**parameters, 'messages':[dict(role='user', content=prompt)]},
                                   output/'calibration'/f'case-{i}')
            checks.append(dict(expected=fixture['expected'], observed=result, model=model))
        save(calibration, dict(passed=all(c['expected'] == c['observed'] for c in checks), cases=checks))
    require(json.loads(calibration.read_text())['passed'], 'Pro judge calibration failed')

    def grade(row):
        path = directory/'cases'/row['id']
        result = json.loads((path/'result.json').read_text())
        response = json.loads((path/'response.json').read_text())
        request = json.loads((path/'request.json').read_text())
        src = source(row)
        require(result['status'] == 'completed'
                and prompt_matches(response['choices'][0], src['tokens'])
                and result['token_sha256'] == row['token_sha256']
                and result['prompt_sha256'] == row['prompt_sha256']
                and result['raw_completion'] == response['choices'][0]['text']
                and result['answer'] == result['raw_completion'].split('</think>', 1)[-1].strip(),
                'Answer/input binding changed: '+row['id'])
        original = json.loads((Path(manifest['reference'])/'results/attention/cases/longmemeval_s'/
                               row['id']/'request_config.json').read_text())
        require(all(request[k] == original[k] for k in (
            'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty', 'seed'))
            and request['max_tokens'] == 16384 and request['thinking_token_budget'] == 4096
            and request['kvmem_query_span'] == [row['query_begin'], row['query_end']],
            'Sampling or original Query changed')
        require(all(result[k] == v for k, v in src['info'].items()), 'Original gold metadata changed')
        body = {**parameters, 'messages':[dict(role='user', content=check(
            result['question_type'], result['question'], result['gold_answer'], result['answer'],
            abstention=result['question_id'].endswith('_abs')))]}
        binding = dict(id=row['id'], request_sha256=judge.digest(body), result_sha256=sha(path/'result.json'),
                       source_sha256=row['source_cache_sha256'], question_type=result['question_type'])
        destination = output/'scores'/(row['id']+'.json')
        if destination.exists():
            value = json.loads(destination.read_text())
            require(all(value[k] == v for k, v in binding.items()), 'Cached judgment binding changed')
            return value
        correct, model = assess(body, output/'requests'/row['id'])
        value = dict(**binding, correct=correct, response_model=model)
        save(destination, value)
        return value

    require(len(manifest['rows']) == 102, 'Expected complete LME102')
    with ThreadPoolExecutor(max_workers=4) as pool:
        judgments = list(pool.map(grade, manifest['rows']))
    kinds = sorted({j['question_type'] for j in judgments})
    report = dict(complete=True, correct=sum(j['correct'] for j in judgments), count=len(judgments),
        accuracy=statistics.fmean(j['correct'] for j in judgments),
        by_type={kind:statistics.fmean(j['correct'] for j in judgments if j['question_type'] == kind)
                 for kind in kinds}, response_models=sorted({j['response_model'] for j in judgments}))
    save(output/'summary.json', report)
    return report


def main(args):
    root = args.campaign.resolve()
    evaluation_hash = sha(root/'evaluation/manifest.json')
    identity = validate_evaluation(root, evaluation_hash)
    for suite in ('niah', 'longmemeval_s', 'memoryagentbench'):
        validate_pair(*(json.loads((root/f'{suite}-{arm}.json').read_text())
                        for arm in ('off', 'guided')), suite)
    key = args.credential_file.read_text().strip()
    require(bool(key), 'No judge credential')
    pending = {(suite, arm) for suite in ('longmemeval_s', 'memoryagentbench') for arm in ('off', 'guided')}
    save(root/'grading-status.json', dict(status='waiting', pending=sorted(pending),
         evaluation_manifest_sha256=evaluation_hash, adapter_sha256=sha(Path(__file__))))
    while pending:
        for suite, arm in sorted(pending.copy()):
            directory = root/f'{suite}-{arm}/answers'
            if not (directory/'completed.json').exists():
                continue
            save(root/'grading-status.json', dict(status='grading', suite=suite, arm=arm,
                                                  pending=sorted(pending)))
            if suite == 'longmemeval_s':
                score_lme(root, arm, key)
            else:
                analyze_memory.FROZEN = root/'evaluation/evaluators'
                analyze_memory.PYTHON = identity['python']
                analyze_memory.OFFICIAL_REPO = root/'evaluation/official'
                os.environ['DEEPSEEK_API_KEY'] = key
                try:
                    analyze_memory.analyze(NS(campaign=directory, external=True))
                finally:
                    os.environ.pop('DEEPSEEK_API_KEY', None)
                report = json.loads((directory/'scoring/summary.json').read_text())
                require(report['complete'], 'MAB still has missing judgments')
            pending.remove((suite, arm))
        if not args.watch:
            break
        status = json.loads((root/'status.json').read_text())
        require(status['status'] != 'failed', 'Inference failed; retain answers and scores')
        if pending:
            time.sleep(30)
    save(root/'grading-status.json', dict(status='completed' if not pending else 'pending',
                                          pending=sorted(pending), at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    main(parser.parse_args())
