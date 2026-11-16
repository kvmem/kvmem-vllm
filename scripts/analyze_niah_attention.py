"""Audit output-attention sampling and paired quality on cold frozen NIAH cases."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer

from benchmark_niah import save, TASKS
from benchmark_client_io import prompt_matches
from vllm_kvmem.serving.action_attention import ActionHistory


def audit(campaign, config, baseline):
    cfg = json.loads(config.read_text())
    manifest = json.loads((campaign / 'manifest.json').read_text())
    summary = json.loads((campaign / 'summary.json').read_text())
    old = {r['id']: r for r in json.loads((baseline / 'summary.json').read_text())['results']}
    frozen_old = {r['id']: r for r in json.loads((baseline / 'manifest.json').read_text())['rows']}
    tok = Tokenizer.from_file(str(Path(cfg['tokenizer']) / 'tokenizer.json'))
    markers = [tok.token_to_id(s) for s in ('<tool_call>', '</tool_call>', '<think>', '</think>', '<|im_start|>')]
    options = SimpleNamespace(**cfg['kvmem'])
    rows = []
    for result in summary['results']:
        assert result['status'] == 'completed', result['id']
        case = result['id']
        meta = next(r for r in manifest['rows'] if r['id'] == case)
        assert meta == frozen_old[case], case
        directory = campaign / 'cases' / case
        response = json.loads((directory / 'response.json').read_text())
        action = json.loads((directory / 'action_attention.json').read_text())
        prompt = np.fromfile(campaign / 'inputs' / (case + '.i32'), dtype='<i4').tolist()
        assert prompt_matches(response['choices'][0], prompt)
        generated = response['choices'][0]['token_ids']
        assert len(generated) == response['usage']['completion_tokens']
        state = ActionHistory(options, markers)
        state.begin('offline-oracle', prompt)
        expected = 0
        # The final emitted token has not been consumed by a target forward.
        consumed = generated[:-1]
        for at in range(0, len(consumed), 32):
            chunk = consumed[at:at + 32]
            state.parser.prepare(chunk)
            expected += state.parser.commit(len(chunk)).bit_count()
        assert action['samples'] == expected, (case, action, expected)
        assert action['scope'] == cfg['kvmem']['action_scope']
        assert action['rounds'] == int(expected > 0)
        assert (action['positive_pages'] > 0) == (expected > 0)
        if expected:
            assert abs(action['score_sum'] - 1) < .005
        events = json.loads((directory / 'events.json').read_text())
        query = [e for e in events['bounded_refresh.jsonl'] if e['selection'] == 'query']
        assert len(query) == 1
        assert query[0]['action_attention_enabled'] and query[0]['action_history_pages'] == 0
        if cfg['kvmem'].get('prefill_attention'):
            assert query[0]['attention_source'] == 'prefill'
            assert query[0]['prefill_attention_samples'] > 0
            assert action['prefill_samples'] == query[0]['prefill_attention_samples'], 'Query replay entered input feedback'
            pressure = [e for e in events['bounded_refresh.jsonl'] if e['selection'] == 'pressure']
            assert pressure and all(e['attention_source'] == 'prefill' for e in pressure)
        rows.append({'id': case, 'task': meta['task'], 'depth': meta['position_percent'],
                     'samples': expected, 'positive_pages': action['positive_pages'],
                     'prefill_samples': query[0].get('prefill_attention_samples', 0),
                     'prefill_chunks': query[0].get('prefill_attention_chunks', 0),
                     'query_action_pages': query[0]['action_history_pages'],
                     'old_coverage': old[case]['evaluation']['normalized_coverage'],
                     'new_coverage': result['evaluation']['normalized_coverage'],
                     'old_exact': old[case]['evaluation']['normalized_exact_numeric_set'],
                     'new_exact': result['evaluation']['normalized_exact_numeric_set'],
                     'old_seconds': old[case]['seconds'], 'new_seconds': result['seconds']})
    groups = []
    for task in ['all'] + TASKS:
        subset = [r for r in rows if task == 'all' or r['task'] == task]
        if subset:
            groups.append({'task': task, 'cases': len(subset),
                **{k: statistics.fmean(r[k] for r in subset) for k in ('old_coverage', 'new_coverage')},
                **{k: sum(r[k] for r in subset) for k in ('old_exact', 'new_exact')},
                'sampled_cases': sum(r['samples'] > 0 for r in subset),
                'accepted_action_samples': sum(r['samples'] for r in subset)})
    ledger = json.loads((campaign / 'cpu_admission.json').read_text())
    wall = max(r['at'] for r in ledger['events']) - min(r['at'] for r in ledger['events'])
    inputs = sum(r['served_input_tokens'] for r in summary['results'])
    outputs = sum(r['completion_tokens'] for r in summary['results'])
    report = {'completed': len(rows), 'expected': manifest['cases'], 'groups': groups,
              'action_sample_counts_verified': len(rows), 'cold_query_zero_action_history_verified': len(rows),
              'accepted_action_samples': sum(r['samples'] for r in rows),
              'wall_seconds': wall, 'input_tokens': inputs, 'output_tokens': outputs,
              'end_to_end_input_tokens_per_second': inputs / wall,
              'end_to_end_output_tokens_per_second': outputs / wall,
              'throughput_scope': 'Whole campaign mixed prefill/guided replay/assembly/decode; not isolated phase throughput.',
              'comparison_scope': 'Same frozen inputs and sampling; Action plus async refresh and GPU allocation differ from old baseline.',
              'selection_scope': 'Cold single-turn requests: Action scores committed after Query selection, cannot improve that selection.',
              'rows': rows}
    if cfg['kvmem'].get('prefill_attention'):
        report.update(cold_prefill_attention_verified=len(rows),
            selection_scope='Cold input attention contributes to pressure and first Query selection; output attention is independently verified.',
            comparison_scope='Same frozen inputs and sampling. Complete input-attention policy and K/G/sink/recent/guided settings differ from the prior output-only vLLM campaign.')
    save(campaign / 'attention-audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    a = p.parse_args()
    audit(a.campaign, a.config, a.baseline)
