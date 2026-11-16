"""Replay recorded bridge prefix misses without submitting model/tool work."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from vllm_kvmem.core.request_records import load_request


def audit(run):
    spec = importlib.util.spec_from_file_location('prefix_audit_bridge',
        Path(__file__).resolve().parents[1] / 'benchmarks/deepswe/claude_bridge.py')
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    protocol = bridge.Protocol(run / 'protocol/protocol.so')
    rows = [json.loads(line) for line in (run/'bridge/requests.jsonl').read_text().splitlines()]
    misses = [r for r in rows if r['event'] == 'start' and r['turn'] > 1
              and not r['exact_output_prefix_reused']]
    results = []
    for row in misses:
        task, turn = row['task'], row['turn']
        folder = run / 'bridge' / task
        previous = load_request(folder/f'{turn-1:05d}/request.json')
        response = json.loads((folder/f'{turn-1:05d}/response.json').read_text())
        current = load_request(folder/f'{turn:05d}/request.json')
        before = protocol(request=previous)
        sent = protocol(request={**previous, 'messages': previous['messages'] +
                               [{'role':'assistant', 'content':response['content']}]})
        after = protocol(request=current)
        n = len(before['messages'])
        control = lambda r: (r['prompt'].encode()[:r['control_end']], r['tools'])
        checks = {'same_control': control(before) == control(after),
            'append_only': len(after['messages']) > n+1,
            'same_prior_messages': [bridge.message_identity(m) for m in before['messages']] ==
                                   [bridge.message_identity(m) for m in after['messages'][:n]],
            'same_assistant': n < len(after['messages']) and
                bridge.message_identity(sent['messages'][-1]) == bridge.message_identity(after['messages'][n])}
        results.append({'task':task, 'turn':turn, 'prompt_tokens':row['prompt_tokens'],
                        'transport_equivalent':all(checks.values()), **checks})
    return {'run':str(run), 'scope':'Identity checks only; no model execution or token/performance claim',
            'checked_misses':len(results), 'equivalent_misses':sum(r['transport_equivalent'] for r in results),
            'still_distinct':sum(not r['transport_equivalent'] for r in results), 'results':results}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}))
