"""Read-only evidence check for one pinned, unscored Agent runtime failure."""
import json
from vllm_kvmem.core.request_records import load_request

try:
    from . import recover_terminalbench_network_audit as base
except ImportError:
    import recover_terminalbench_network_audit as base

INCIDENT = dict(trial='schemelike-metacircular-eval__YdgAcEt',
    task='terminal-bench/schemelike-metacircular-eval',
    raw_sha256='997364c440e1d8e8310218cf6e97b47b25d8331615b6c0ffd7a15faef1685fc5',
    event_sha256='ba5c3dd393d9150608a03942e2c660e2963fc8a83161fdf43add84dfc87cc268',
    trajectory_sha256='89d4f8dc640eed925f097df68592f133fd30179b30ed0b025d5e0249f017647d',
    owner='tb21-8c286350dad657b75bf5f4ee4e87e9a57b4f353c',
    network='d87257b7d3912b7d6005068afe00324d1ed02e3433fda647bcd54f532a9677b2')
ERROR = 'Terminal Query history changed without source metadata'
STREAM_ERROR = 'Incomplete model stream or accepted-token/usage mismatch'
require, read, sha = base.require, base.read, base.sha


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate(arm, lifecycle):
    item = INCIDENT; trial, owner = item['trial'], item['owner']
    session = trial+'__agent'
    raw_name, event_name = f'jobs/formal/{trial}/result.json', f'formal-events/{trial}.json'
    trajectory_name = f'jobs/formal/{trial}/agent/trajectory.json'
    require(sha(arm/raw_name) == item['raw_sha256'] and sha(arm/event_name) == item['event_sha256']
            and sha(arm/trajectory_name) == item['trajectory_sha256'], 'Pinned Query incident evidence changed')
    raw = read(arm/raw_name); error = raw.get('exception_info') or {}
    require(raw == read(arm/event_name) and raw.get('trial_name') == trial and raw.get('task_name') == item['task']
            and error.get('exception_type') == 'ValueError' and error.get('exception_message') == ERROR
            and 'source.proof(messages)' in error.get('exception_traceback', '')
            and 'terminalbench_query.py' in error.get('exception_traceback', '')
            and raw.get('verifier') is None and raw.get('verifier_result') is None,
            'Query incident is not the pinned unscored Agent runtime error')
    execution = raw.get('agent_execution') or {}
    agent = raw.get('agent_result') or {}
    require(execution.get('started_at') and execution.get('finished_at') and raw.get('finished_at')
            and agent.get('metadata', {}).get('n_episodes') == 635
            and len(read(arm/trajectory_name).get('steps', [])) == 635,
            'Query incident did not retain the original completed Agent trajectory')
    own = [row for row in lifecycle if row.get('trial') == trial]
    by_kind = {}
    for kind in ('start', 'environment-start', 'agent-start', 'agent-end', 'kv_released', 'end', 'empty_networks_reclaimed'):
        selected = [row for row in own if row.get('event') == kind]
        require(len(selected) == 1, 'Query incident lacks unique '+kind)
        by_kind[kind] = selected[0]
    require(not any(row.get('event') == 'verification-start' for row in own), 'Query incident reached verifier')
    require(by_kind['kv_released'].get('released') is True and by_kind['kv_released'].get('owners') == [owner],
            'Query incident native owner was not released')
    require(all(by_kind[a]['at'] <= by_kind[b]['at'] for a,b in zip(
        ('start','environment-start','agent-start','agent-end','kv_released','end'),
        ('environment-start','agent-start','agent-end','kv_released','end','empty_networks_reclaimed'))),
        'Query lifecycle order differs')
    bridge = [row for row in rows(arm/'bridge/requests.jsonl') if row.get('owner') == owner]
    opened = [row for row in bridge if row.get('event') == 'session_opened']
    released = [row for row in bridge if row.get('event') == 'history_released']
    reports = [row for row in bridge if 'turn' in row]
    failures = [row for row in bridge if 'error' in row]
    require(len(opened) == len(released) == 1 and opened[0].get('session_id') == released[0].get('session_id') == session
            and sorted(row['turn'] for row in reports) == list(range(1, 636))
            and len(failures) == 9 and all(row.get('error') == STREAM_ERROR for row in failures),
            'Query bridge request history differs')
    sources = [row for row in rows(arm/'formal-query-sources.jsonl') if row.get('session_id') == session]
    bound = [row for row in sources if row.get('event') == 'task_query_bound']
    proofs = [row for row in sources if row.get('event') == 'query_source']
    require(len(bound) == 1 and len(proofs) == 638 and all(row.get('query_user_ordinal') == 0 for row in proofs)
            and len({row.get('query_message_id') for row in proofs}) == 1, 'Query source history differs')
    prefix = f'bridge/{owner}'
    last = read(arm/prefix/'00635/native.json')
    accepted = read(arm/prefix/'00635/accepted.json')
    require(last.get('finish_reason') == 'length' and last.get('usage', {}).get('completion_tokens') == 131072
            and len(last.get('token_ids', [])) == 131072 and last['token_ids'] == accepted.get('token_ids'),
            'Original output-limit response is missing or incomplete')
    attempts = sorted((arm/prefix).glob('00636*'))
    require(len(attempts) == 9 and all(p.is_dir() for p in attempts), 'Query recovery request count differs')
    last_request = load_request(arm/prefix/'00635/request.json')
    names = [raw_name, event_name, trajectory_name, prefix+'/00635/native.json', prefix+'/00635/accepted.json',
             prefix+'/00635/request.json', 'runtime/scripts/terminalbench_query.py']
    for directory in attempts:
        request = load_request(directory/'request.json')
        require(request.get('messages', [])[:-2] == last_request.get('messages')
                and request['messages'][-2].get('role') == 'assistant'
                and request['messages'][-1].get('role') == 'user'
                and request['messages'][-1].get('content', '').startswith('ERROR!! NONE of the actions')
                and request.get('kvmem_query_provenance', {}).get('user_messages_sha256') == proofs[-1]['user_messages_sha256']
                and not any((directory/name).exists() for name in ('response.json','native.json','accepted.json')),
                'Query recovery evidence differs or unexpectedly completed')
        names.append(str((directory/'request.json').relative_to(arm)))
        names.append(str((directory/'plan.json').relative_to(arm)))
    adapter_name = 'runtime/scripts/terminalbench_query.py'
    require(read(arm/'manifest.json')['frozen']['runtime']['scripts/terminalbench_query.py'] == sha(arm/adapter_name),
            'Frozen Query adapter differs from manifest')
    network = arm/'formal-events/network-cleanup'/trial
    checked_path, removed_path = network/(item['network']+'.json'), network/(item['network']+'.removed.json')
    checked, removed = read(checked_path), read(removed_path)
    require(checked.get('status') == 'checked' and removed.get('status') == 'removed'
            and removed.get('retained_filesystem_diffs_unchanged') is True
            and removed.get('project') == checked.get('project') == trial.lower()+'__env'
            and checked.get('network') == removed.get('network') and removed['network'].get('Id') == item['network']
            and removed['network'].get('Containers') == {}
            and checked.get('retained_containers') == removed.get('retained_containers')
            and checked.get('environment_dir') == removed.get('environment_dir') ==
                str(arm/'dataset/tasks/schemelike-metacircular-eval/environment')
            and by_kind['end']['at'] <= checked['at'] <= removed['finished_at'] <= by_kind['empty_networks_reclaimed']['at']
            and by_kind['empty_networks_reclaimed'].get('count') == 1,
            'Query incident network cleanup evidence differs')
    names.extend(str(p.relative_to(arm)) for p in (checked_path, removed_path))
    return dict(passed=True, trial=trial, task=item['task'], classification='agent_runtime_error',
        reward=None, exception_type='ValueError', original_outcome_preserved=True,
        no_verifier_execution=True, model_execution_occurred=True, episodes=635,
        causal_scope='Output-limit recovery, nine failed native stream attempts, then Query provenance rejection; native stream failure cause remains unresolved.',
        quality_scope='Missing verifier reward; retained as an unsuccessful full-cohort attempt. Exclusion from scored-only analysis is potentially policy-dependent.',
        pinned_source_hashes={name: sha(arm/name) for name in names})
