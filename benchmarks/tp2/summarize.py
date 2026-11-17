"""Print the decode result of a run_16x32k.sh run.

usage: python summarize.py runs/LABEL [--decoders 16]
Decode tok/s = tokens returned while every session was decoding (after the last session's first
token, before the first one finished) over that window, per round; acceptance = accepted draft
tokens / drafts (vLLM spec-decode counters); step = median model_step_latency step_ms over
decode-only steps with all sessions decoding (first fifth dropped as warmup).
"""
import glob
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
decoders = int(sys.argv[sys.argv.index('--decoders') + 1]) if '--decoders' in sys.argv else 16
for summary_path in sorted(glob.glob(str(run / 'cases/prepared*/summary.json'))):
    case = Path(summary_path).parent
    summary = json.loads(Path(summary_path).read_text())
    if 'rounds' not in summary or 'median_common_decode_tokens_per_second' not in summary:
        print(case.name, 'incomplete:', str(summary)[:300])
        continue
    rounds = [r['common_decode_tokens_per_second'] for r in summary['rounds']]
    print(f"{case.name}: decode tok/s per round {', '.join(f'{v:.1f}' for v in rounds)}; "
          f"median {summary['median_common_decode_tokens_per_second']:.1f}")
    for i, r in enumerate(summary['rounds'], 1):
        delta = json.loads((case / f'r{i}-metrics.json').read_text())['delta']
        total = lambda prefix: sum(v for k, v in delta.items() if k.startswith(prefix))
        acc = total('vllm:spec_decode_num_accepted_tokens_total') / max(1, total('vllm:spec_decode_num_drafts_total'))
        print(f"  round {i}: accepted/draft {acc:.2f}, turn output {r['aggregate_turn_output_tokens_per_second']:.1f} tok/s, "
              f"mean TTFT {r['mean_ttft_seconds']:.2f} s, token accounting {r['token_accounting_passed']}, "
              f"peak overlapping decoders {r['peak_overlapping_decode_requests']}")
steps = []
latency = run / 'server/capture/model_step_latency.jsonl'
if latency.exists():
    for line in latency.open():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not row.get('prefill_tokens', 1) and row.get('decode_requests') == decoders:
            steps.append(row['step_ms'])
    steps = steps[len(steps) // 5:]
if steps:
    print(f'median decode step with {decoders} decoders: {statistics.median(steps):.2f} ms over {len(steps)} steps')
