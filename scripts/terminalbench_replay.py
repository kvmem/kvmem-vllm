"""Read-only replay of saved Terminal-Bench Query decisions; no model calls.

This measures Query invalidation eligibility, not GPU speed or task quality.
Recorded initial Query coordinates are reused only with exact-prefix evidence.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
from vllm_kvmem.core.request_records import load_request

try:
    from .terminalbench_query import TaskQuery, digest
except ImportError:
    from terminalbench_query import TaskQuery, digest


def read(path):
    return json.loads(path.read_text())


def instruction_text(path):
    # Same leading-canary rule as the pinned Harbor Task loader. Verify the
    # resulting text against the original on-wire prompt before using it.
    lines = path.read_text().split('\n')
    while lines and re.fullmatch(r'(<!--.*canary.*-->|#.*canary.*)', lines[0].strip(), re.I):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return '\n'.join(lines)


def replay(root, max_owners=8, max_turns=32, task_names=None):
    sessions, tasks = {}, {}
    for line in (root/'bridge/requests.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('event') == 'session_opened':
            sessions[row['owner']] = row['session_id']
    for line in (root/'formal-events/events.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('trial') and row.get('task'):
            tasks[row['trial']+'__agent'] = row['task'].removeprefix('terminal-bench/')
    results, evidence = [], {}
    for owner, session in sorted(sessions.items()):
        task = tasks.get(session)
        if task is None or (task_names and task not in task_names):
            continue
        directories = sorted(p for p in (root/'bridge'/owner).iterdir()
                             if p.name.isdigit() and (p/'accepted.json').exists() and (p/'plan.json').exists())
        if len(directories) < 2 or directories[0].name != '00001':
            continue
        instruction_path = root/'dataset/tasks'/task/'instruction.md'
        instruction = instruction_text(instruction_path)
        first = load_request(directories[0]/'request.json')
        first_users = [m for m in first['messages'] if m['role'] == 'user']
        if len(first_users) != 1 or not isinstance(first_users[0]['content'], str):
            raise ValueError('Initial task source is ambiguous: '+task)
        source = TaskQuery(session, task, instruction, first_users[0]['content'])
        initial_query = read(directories[0]/'plan.json').get('kvmem_query_span')
        if not initial_query:
            raise ValueError('Initial source has no recorded Query coordinates: '+task)
        evidence[str(instruction_path)] = hashlib.sha256(instruction_path.read_bytes()).hexdigest()
        prior_end, prior_query = None, None
        rows = []
        for index, directory in enumerate(directories[:max_turns]):
            if int(directory.name) != index+1:
                break
            body, plan = load_request(directory/'request.json'), read(directory/'plan.json')
            original = copy.deepcopy(body)
            proof = source.proof(body['messages'])
            if body != original:
                raise AssertionError('Query adapter changed model messages')
            query = plan.get('kvmem_query_span')
            exact = bool(plan.get('exact_output_prefix_reused'))
            eligible = (prior_end is not None and exact and initial_query[1] <= prior_end)
            changed = prior_query is not None and query != prior_query
            overlap = bool(changed and query and prior_end is not None and query[0]//32*32 < prior_end)
            rows.append(dict(turn=index+1, prompt_tokens=plan['prompt_tokens'], exact_prefix=exact,
                             legacy_query=query, task_query=initial_query,
                             task_query_id=proof['query_message_id'], legacy_query_changed=changed,
                             task_query_replay_suppressible=eligible,
                             legacy_replay_overlaps_previous_accepted=overlap))
            accepted = read(directory/'accepted.json')['token_ids']
            prior_end, prior_query = plan['prompt_tokens']+len(accepted), query
            for name in ('request.json', 'plan.json', 'accepted.json'):
                path = directory/name
                evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append(dict(task=task, owner=owner, turns=rows,
                            legacy_unique_queries=len({tuple(r['legacy_query'] or []) for r in rows}),
                            task_query_identities=len({r['task_query_id'] for r in rows})))
        if len(results) >= max_owners:
            break
    if not results:
        raise ValueError('No complete original task traces matched')
    counts = Counter()
    for task in results:
        for row in task['turns']:
            counts['requests'] += 1
            for name in ('legacy_query_changed', 'task_query_replay_suppressible', 'legacy_replay_overlaps_previous_accepted'):
                counts[name] += row[name]
    return dict(source=str(root), scope='Offline fixed-message source replay; no GPU, task answer or throughput claim',
                messages_unchanged=True, token_journals_unchanged=True,
                limitation='Suppressible Query replay is necessary but not sufficient for resident restore; capacity and other frame state still apply',
                tasks=len(results), counts=dict(counts), traces=results, evidence_sha256=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-owners', type=int, default=8)
    parser.add_argument('--max-turns', type=int, default=32)
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new offline evidence path')
    result = replay(args.source.resolve(), args.max_owners, args.max_turns, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({key: result[key] for key in ('tasks', 'counts', 'messages_unchanged')}, indent=2))
