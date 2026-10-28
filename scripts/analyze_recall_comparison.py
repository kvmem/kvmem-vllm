"""Analyze all five fresh complete RULER runs without rescoring old answers."""
import argparse
from pathlib import Path

import recall_comparison_protocol as p
import ruler_protocol as ruler
from run_recall_comparison import resolve, saved_result, verify


def analyze(root):
    fixture, profile = verify(root)
    scorer = ruler.load_scorer(resolve(root, fixture['upstream_path']))
    arms, official = {}, {}
    for method in p.METHODS:
        output = root / 'arms' / method / 'formal'
        values = []
        for row in fixture['rows']:
            case = output / 'cases' / row['id']
            result = saved_result(case, row, method, root)
            p.require(result is not None, 'Incomplete method; do not report a paired full-cohort comparison')
            attempt = case / result['attempt']
            response = p.read(attempt / 'response.json')
            recalculated = p.score_case(row, response, scorer, starts_in_thinking=fixture['starts_in_thinking'])
            p.require(all(result[key] == value for key, value in recalculated.items()), 'Stored score differs from official scorer')
            evidence = p.read(attempt / 'evidence.json')
            p.audit_events(evidence['events'], method, profile, query_span=row['served_query'],
                           required_types=('prefill_pressure', 'query'))
            values.append(result)
        arms[method] = values
        official[method] = ruler.score_suite(fixture['rows'],
            {(row['task'], row['index']): row['text'] for row in values}, scorer,
            samples_per_task=20, starts_in_thinking=fixture['starts_in_thinking'])
    comparison = p.compare_results(fixture['rows'], arms)
    comparison['official_batch_scores'] = official
    for method in p.METHODS:
        p.require(all(comparison['per_task'][task][method] == official[method]['per_task'][task]['score']
                      for task in ruler.TASK_FAMILIES), 'Task means disagree with frozen official batch scoring')
    comparison['rounding_note'] = ('Task table and oracle average unrounded substring coverage per task, '
        'then apply the same two-decimal task rounding as the frozen official scorer before macro averaging.')
    target = root / 'analysis'
    p.save(target / 'comparison.json', comparison)
    columns = '| Task | Mean-K | Fixed4 | FPS-4 | Min-Max | ExactMass |'
    lines = ['# Fixed-rule recall comparison', '', 'Original RULER 260 cases; 128 accepted tail queries for pressure.',
        'Five independent runs per original input; ExactMass is a diagnostic reference.', '', columns,
        '|---|---:|---:|---:|---:|---:|']
    for task in ruler.TASK_FAMILIES:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{comparison["per_task"][task][m]:.4f}' for m in p.METHODS) + ' |')
    lines.append('| Macro | ' + ' | '.join(f'{comparison["macro_scores"][m]:.4f}' for m in p.METHODS) + ' |')
    potential = comparison['potential']
    lines += ['', f'Best fixed candidate: {", ".join(potential["best_fixed_methods"])} ({potential["best_fixed_score"]:.4f}).',
        f'Per-request post-hoc oracle: {potential["per_request_oracle_score"]:.4f}; gain {potential["gain_points"]:.4f} points.',
        'These are task-balanced whole-request scores. No routing implementation or event-wise oracle is evaluated.', '',
        '| Candidate | Unique full-reference successes |', '|---|---:|']
    lines += [f'| {method} | {comparison["unique_successes"][method]["count"]} |' for method in p.CANDIDATES]
    lines += ['', '| Pair (left vs right) | Wins | Losses | Ties |', '|---|---:|---:|---:|']
    for pair in comparison['pairwise'].values():
        lines.append(f'| {pair["left"]} vs {pair["right"]} | {pair["wins"]} | {pair["losses"]} | {pair["ties"]} |')
    lines += ['', comparison['rounding_note'], '', comparison['success_definition']]
    (target / 'report.md').write_text('\n'.join(lines) + '\n')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    print(result['potential'])
