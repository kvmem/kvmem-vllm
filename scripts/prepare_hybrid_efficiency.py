"""Prepare finite shape-selected pilots from an already frozen MAB runtime.

This changes only which original cases are dispatched, and the explicit graph
alpha. The prepared full benchmark and every frozen runtime stay untouched.
No answer, gold value or previous correctness score participates in selection.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

from benchmark_niah import save,sha
from graph_v2_protocol import require
from run_ruler_hybrid_campaign import verify_files


def choose(rows):
    require(rows and len({r['id'] for r in rows}) == len(rows),'Duplicate or empty source cohort')
    groups = {}
    for row in rows:
        require(type(row['input_tokens']) is int and 0 < row['query_begin'] < row['query_end'] <= row['input_tokens'],
            'Invalid original input/Query shape')
        groups.setdefault(row['id'].rsplit('_q',1)[0],[]).append(row)
    first = [v[0] for v in groups.values()]
    repeat = [v for v in groups.values() if len(v) > 1]
    require(repeat,'No reusable prefix in original cohort')
    longest = max(repeat,key=lambda v:(max(r['input_tokens'] for r in v),v[0]['id']))
    length_order = sorted(rows,key=lambda r:(r['input_tokens'],r['id']))
    median = length_order[len(rows)//2]['input_tokens']
    longmem = sorted((r for r in first if '_longmemeval_' in r['id']),
        key=lambda r:(r['input_tokens'],r['id']))
    require(longmem,'Missing memory QA workload')
    picks = [
        ('longest_reused_prefix_first',longest[0]),
        ('longest_reused_prefix_next',longest[1]),
        ('maximum_input',length_order[-1]),
        ('minimum_input',length_order[0]),
        ('population_median_input',min(first,key=lambda r:(abs(r['input_tokens']-median),r['id']))),
        ('shortest_query_first_in_group',min(first,key=lambda r:(r['query_end']-r['query_begin'],r['id']))),
        ('maximum_query',min(rows,key=lambda r:(-(r['query_end']-r['query_begin']),r['input_tokens'],r['id']))),
        ('median_memory_qa_prefix',longmem[len(longmem)//2]),
    ]
    selected = {}
    for label,row in picks:
        entry = selected.setdefault(row['id'],dict(id=row['id'],input_tokens=row['input_tokens'],
            query_tokens=row['query_end']-row['query_begin'],reasons=[]))
        entry['reasons'].append(label)
    # Only the deliberately selected next question reuses a prefix. Existing
    # frozen campaign audits verify this continuation's entire cache prefix.
    chosen_groups = {}
    for ident in selected:chosen_groups.setdefault(ident.rsplit('_q',1)[0],[]).append(ident)
    require(sorted(map(len,chosen_groups.values())) == [1]*(len(chosen_groups)-1)+[2],
        'Shape selection unexpectedly introduces another warm continuation')
    return list(selected.values())


def prepare(source, target, *, graph_alpha):
    require(not target.exists(),'Use a fresh efficiency pilot; never overwrite attempts')
    require(type(graph_alpha) in (int,float) and graph_alpha in (0.,.2),'Unsupported graph alpha')
    original = json.loads((source/'fixture.json').read_text())
    require(original['suite'] == 'memoryagentbench' and original['qualification'] is True and
        json.loads((source/'status.json').read_text())['status'] == 'prepared' and
        not (source/'attempt.json').exists(), 'Source must be the untouched prepared MAB qualification')
    proof = json.loads((source/'prepared-proof.json').read_text())
    verify_files(source,proof)
    verify_files(source/'runtime',json.loads((source/'runtime-proof.json').read_text())['files'])
    selection = choose(original['rows'])
    shutil.copytree(source,target)
    profile = json.loads((target/'profile.json').read_text());profile['kvmem']['graph_alpha'] = graph_alpha
    save(target/'profile.json',profile)
    fixture = deepcopy(original)
    fixture.update(case_ids=[v['id'] for v in selection],cases=len(selection),
        profile_sha256=sha(target/'profile.json'),
        qualification_scope='Shape-only efficiency pilot: input/Query extrema, median input, short Query, memory QA and two consecutive questions on longest reusable prefix; no answer-dependent selection.',
        efficiency_pilot=True)
    save(target/'fixture.json',fixture)
    save(target/'efficiency-selection.json',dict(source=str(source.resolve()),source_files=proof,
        case_shapes=selection,graph_alpha=graph_alpha,original_rows_unchanged=True,
        runtime_proof_sha256=sha(source/'runtime-proof.json'),
        sampling='Original per-case seeds and output limits; frozen benchmark client unchanged',
        no_accuracy_gate=True))
    names=tuple(proof)+('efficiency-selection.json',)
    save(target/'prepared-proof.json',{n:sha(target/n) for n in names})
    save(target/'status.json',dict(status='prepared',cases=len(selection),suite='memoryagentbench',
        efficiency_pilot=True,graph_alpha=graph_alpha,at=time.time()))
    require(json.loads((target/'runtime-proof.json').read_text()) ==
        json.loads((source/'runtime-proof.json').read_text()),'Pilot changed frozen runtime')
    return selection


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--graph-alpha',type=float,choices=(0.,.2),required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.source.resolve(),args.output.resolve(),graph_alpha=args.graph_alpha),indent=2))
