"""Audit complete natural-agent outcomes; keep trajectory and quality differences."""
import argparse
import json
from pathlib import Path

from scripts import run_agent_handoff_queue as q


def delta(before,after):
    return dict(before=before,after=after,change_percent=(after/before-1)*100 if before else None)


def analyze(root):
    from scripts.benchmark_agent_fastpath_dsh import validate, outcome
    validate(root)
    arms={}
    for name in ['baseline','candidate']:
        directory=root/name
        result=outcome(directory)
        perf=q.read(directory/'performance.json')
        if perf['interrupted'] or not perf['uninterrupted_eight_task_comparison']:
            raise ValueError('Recovery downtime must not enter an uninterrupted A/B comparison')
        obs=q.read(directory/'observations.json')
        browser=q.read(directory/'browser-summary.json')
        cases={}
        for ident,case in obs['cases'].items():
            state=q.read(directory/'cases'/ident/'status.json')
            started=q.read(directory/'cases'/ident/'turn-01/started.json')['at']
            ended=state.get('completed_at')
            cases[ident]=dict(status=case['status'],human_turns=case['completed_human_turns'],
                wall_seconds=ended-started if ended is not None else None,
                max_logical_input_tokens=case['backend']['max_logical_input_tokens'],
                accepted_output_tokens=case['backend']['accepted_output_tokens'],
                completed_model_requests=case['backend']['completed_model_requests'],
                tool_results=case['statistics']['tool_results'],
                tool_errors=case['statistics']['tool_error_results'],
                shell_nonzero=case['statistics']['bash_nonzero_exit_results'],
                browser=browser['reports'].get(ident,{}).get('status'))
        accepted=sum(x['accepted_output_tokens'] for x in cases.values())
        if accepted!=perf['accepted_output_tokens'] or abs(
                accepted/perf['wall_seconds']-perf['aggregate_agent_tokens_per_second'])>1e-8:
            raise ValueError('Accepted output accounting mismatch')
        restores=[json.loads(line) for line in (directory/'server/capture/async_restore.jsonl').read_text().splitlines()]
        post=sum(r.get('completion_poll')=='post_forward' for r in restores)
        if name=='baseline' and post:
            raise ValueError('Disabled fastpath executed')
        arms[name]=dict(outcome=result,performance=perf,cases=cases,
            post_forward_completions=post,restore_count=len(restores),
            browser_checked=browser['checked'],
            browser_basic_passed=sum(x.get('status')=='passed_basic_smoke' for x in browser['reports'].values()),
            browser_mobile_overflow=sum(bool(x.get('mobile_horizontal_overflow')) for x in browser['reports'].values()),
            evidence_sha256=q.sha(directory/'evidence-hashes.json'))
    b,a=[arms[x]['performance'] for x in ['baseline','candidate']]
    full=all(x['outcome']['full_cohort_passed'] for x in arms.values())
    return dict(arms=arms,full_paired_cohort_passed=full,
        aggregate_agent_tokens_per_second=delta(b['aggregate_agent_tokens_per_second'],a['aggregate_agent_tokens_per_second']),
        wall_seconds=delta(b['wall_seconds'],a['wall_seconds']),
        accepted_output_tokens=delta(b['accepted_output_tokens'],a['accepted_output_tokens']),
        tool_result_error_rate=delta(b['counts']['tool_result_error_rate'],a['counts']['tool_result_error_rate']),
        caveats=['Accepted output divided by full arm wall time includes tool execution and follow-ups.',
            'Same tasks and sampling do not guarantee the same generated code, output length or tool path.',
            'Report observed whole-task throughput together with task completion, per-task durations and quality.',
            'Any original resource-limit failures remain in the totals and are not retried.',
            'Single sequential Off/On experiment; no confidence interval from independent repeats.',
            'MTP verification rows are not accepted output tokens; restore intervals may overlap.',
            'Browser checks cover loading and basic interaction, not exhaustive feature correctness.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    q.save(args.output,analyze(args.root.resolve()))
