#!/usr/bin/env python3
"""Write real-text calibration prompts for benchmark_qsa_phases.py --prompt-file.

Reads LongMemEval-S (``longmemeval_s_cleaned.json``) and writes one JSON object
per line with ``context`` (the dated chat history) and ``question``. The
question asks for step-by-step reasoning over the history, so a long decode
keeps reading the context, which is what the QSA selection trace needs.

    python scripts/qsa_calibration_prompts.py LME.json out.jsonl --count 16 --seed 7
"""
import argparse
import json
import random
import sys
from pathlib import Path

INSTRUCTION = ('Using only the chat history above, answer the question. First list every earlier '
               'message that is relevant, quoting it with its session date, then reason step by step, '
               'then give the answer.\nQuestion ({date}): {question}')


def history(item):
    parts = []
    for date, session in zip(item['haystack_dates'], item['haystack_sessions']):
        parts.append(f'### Session on {date}')
        parts.extend(f"{turn['role']}: {turn['content']}" for turn in session)
    return '\n'.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args(argv)
    items = json.loads(Path(args.source).read_text())
    chosen = random.Random(args.seed).sample(items, min(args.count, len(items)))
    with open(args.output, 'w') as out:
        for item in chosen:
            out.write(json.dumps(dict(id=item['question_id'], context=history(item),
                                      question=INSTRUCTION.format(date=item['question_date'],
                                                                  question=item['question']))) + '\n')
    print(f'wrote {len(chosen)} prompts to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
