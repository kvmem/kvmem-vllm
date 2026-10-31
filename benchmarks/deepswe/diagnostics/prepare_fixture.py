"""CPU-only exact-token export for a mixed prefill diagnostic, not live tasks."""
import argparse
import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Frozen kvmem-5090-four-agent-regression-20261031-r2 directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    sys.path.insert(0, str(root / 'runtime/src'))
    run = root / 'live-baseline'
    exporter_path = root / 'runtime/scripts/export_deepswe_capacity_history.py'
    bridge_path = root / 'agent-runtime/benchmarks/deepswe/claude_bridge.py'
    exporter = load('pressure_fixture_exporter', exporter_path)
    bridge = load('pressure_fixture_bridge', bridge_path)
    profile = exporter.read(run / 'profile.json')
    state = exporter.read(run / 'run.json')
    if state['status'] != 'window_complete' or state.get('cleanup_errors') != []:
        raise ValueError('Source window must be completed and cleaned')
    chosen = [
        dict(task='awilix-async-container-initialization', turn=5, prompt_tokens=39844, role='peer'),
        dict(task='koota-composite-trait-aspects', turn=9, prompt_tokens=39070, role='peer'),
        dict(task='textual-richlog-follow-state', turn=10, prompt_tokens=38442, role='peer'),
        dict(task='superjson-error-stack-serialization', turn=25, prompt_tokens=67491, role='target'),
    ]
    events = [json.loads(line) for line in (run / 'bridge/requests.jsonl').read_text().splitlines()]
    completed = {(r['task'], r['turn']) for r in events if r.get('event') == 'finish'
                 and state['model_ready_at'] <= r['at'] <= state['window_deadline']}
    if any((r['task'], r['turn']) not in completed for r in chosen):
        raise ValueError('Selected request did not finish inside the formal window')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    planner = bridge.Planner(bridge.Protocol(run / 'protocol/protocol.so'), tokenizer,
                             profile['kvmem']['gen_budget'], optimized_handoff=True)
    args.output.mkdir(parents=True, exist_ok=False)
    samples = []

    def emit(row, tokens, plan, original):
        values = array.array('I', tokens)
        if values.itemsize != 4:
            raise ValueError('Require 32-bit unsigned token IDs')
        if sys.byteorder != 'little':
            values.byteswap()
        raw = values.tobytes()
        filename = f"{row['task']}-{row['turn']:05d}.u32le"
        (args.output / filename).write_bytes(raw)
        diagnostic = {k: v for k, v in plan.items()
                      if k not in ('kvmem_query_span', 'kvmem_query_message_range', 'kvmem_replay_span')}
        diagnostic.update(kvmem_reselect='off', kvmem_prefill_window='pressure')
        samples.append(dict(row, file=filename, sha256=hashlib.sha256(raw).hexdigest(),
            canonical_token_sha256=original['token_sha256'], original_request_plan=plan,
            diagnostic_request_plan=diagnostic,
            continuation_token_id=original['recorded_continuation_token_id']))

    provenance = {}
    for row in chosen:
        provenance.update(exporter.replay_task(run, row['task'], [row], planner,
                                               bridge.Trajectory(), emit))
    for filename, sha in provenance.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha:
            raise ValueError('Source changed while exporting')
    for path in [exporter_path, bridge_path, run/'protocol/protocol.so', run/'profile.json',
                 run/'run.json', run/'bridge/requests.jsonl', Path(__file__).resolve()]:
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(format='deepswe_mixed_pressure_diagnostic_v1', diagnostic_only=True,
        scope='Four exact recorded real DeepSWE prompts. Three fixed 8192-token decode peers '
              'plus one cold prefill request; no tools and no live-task throughput claim. '
              'Query disabled explicitly, original semantic pins/locality retained. '
              'Generated peer content is not teacher forced and may differ between runtimes.',
        source_run=str(run), eos=tokenizer.convert_tokens_to_ids('<|im_end|>'),
        profile_requirements=dict(select_budget=36864, gen_budget=12288, draft_tokens=4,
            max_num_seqs=4, assembly_chunk_blocks=4, query_replay_budget=1856),
        peer_output_tokens=8192, target_output_tokens=1, samples=samples,
        source_sha256=provenance)
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(exported=len(samples), tokens=sum(r['prompt_tokens'] for r in samples),
                         replayed_source_turns=sum(r['turn'] for r in chosen))))


if __name__ == '__main__':
    main()
