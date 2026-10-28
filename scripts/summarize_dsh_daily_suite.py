"""Summarize agent events, exact backend contexts, and reselection evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.run_dsh_daily_suite import event_statistics, bridge_observations, save


def lines(path):
    if not path.exists():return []
    result=[]
    for raw in path.read_text().splitlines():
        try:result.append(json.loads(raw))
        except ValueError:continue
    return result


def summarize(root):
    plan=json.loads((root/'task-plan.json').read_text());reports={};all_ids={}
    for case in plan['cases']:
        directory=root/'cases'/case['id'];events=lines(directory/'events.jsonl')
        stats=event_statistics(events)
        obs=bridge_observations(root,case['session_id'])
        requests=[]
        for owner in obs['agent_owners']:
            for request in (root/'bridge'/owner).glob('*/request.json'):
                turn=request.parent
                body=json.loads(request.read_text())
                prepared=turn/'plan.json'
                if not prepared.exists():continue
                p=json.loads(prepared.read_text())
                segments=[]
                for s in sorted(turn.glob('segment-*.json')):
                    seg=json.loads(s.read_text());segments.append({k:seg.get(k) for k in ('id','prompt_tokens','output_tokens','finish_reason')})
                    if seg.get('id'):all_ids[seg['id']]=case['id']
                requests.append(dict(owner=owner,turn=turn.name,prompt_tokens=p.get('prompt_tokens'),
                    query_message_id=body.get('kvmem_query_provenance',{}).get('query_message_id'),
                    native_segments=segments))
        successful_turns=sum(json.loads(p.read_text()).get('terminal',{}).get('data',{}).get('reason',{}).get('kind')
            in ('completed','complete','done','stop','end-turn') for p in directory.glob('turn-*/result.json'))
        reports[case['id']]=dict(session_id=case['session_id'],workspace=case['workspace'],
            statistics=stats,backend=obs,completed_human_turns=successful_turns,
            context_requirement_met=obs['max_logical_input_tokens']>36864,
            minimum_followups_met=successful_turns>=4,
            request_index=requests,query_reselections=[],pressure_reselections=[],committed_refreshes=[])
        state=directory/'status.json'
        if state.exists():reports[case['id']]['status']=json.loads(state.read_text())['status']
        failure=directory/'failure.json'
        if failure.exists():reports[case['id']]['failure']=json.loads(failure.read_text())
        browser=directory/'browser/result.json'
        if browser.exists():reports[case['id']]['browser']=json.loads(browser.read_text())
    # Worker IDs add a sequence suffix to the HTTP response ID.
    def identify(request):
        if request in all_ids:return all_ids[request]
        return next((v for key,v in all_ids.items() if request and request.startswith(key+'-')),None)
    capture_roots=[Path(p) for p in plan.get('prior_capture_roots',[])]+[root/'server/capture']
    graph=[row for capture in capture_roots for row in lines(capture/'block_graph.jsonl')]
    unassigned=[]
    for row in graph:
        ident=identify(row.get('request_id'))
        if ident is None:
            unassigned.append({k:row.get(k) for k in ('request_id','phase','seed_source')});continue
        small={k:row.get(k) for k in ('request_id','phase','seed_source','alpha','identity',
            'propagation_applied','pressure_policy','budget','selected_tokens','query_quota_selection',
            'graph_selection_seconds','accepted_through','scoring_version')}
        if row.get('phase')=='query':reports[ident]['query_reselections'].append(small)
        elif row.get('phase')=='pressure':reports[ident]['pressure_reselections'].append(small)
    for row in [row for capture in capture_roots for row in lines(capture/'bounded_refresh.jsonl')]:
        ident=identify(row.get('request_id'))
        if ident is not None and row.get('cancelled') is False and row.get('refresh_finished_at') is not None:
            reports[ident]['committed_refreshes'].append({k:row.get(k) for k in (
                'request_id','command_id','selection','history_begin','selected_tokens',
                'query_tokens','guided_query_tokens','phase','seed_source','pressure_policy',
                'alpha','refresh_started_at','refresh_finished_at','public_peer_count')})
    counts=Counter()
    for value in reports.values():
        for key in ('tool_calls','tool_results','tool_error_results','bash_nonzero_exit_results',
                'tool_or_command_failure_results','model_steps','repeated_identical_calls'):
            counts[key]+=value['statistics'][key]
        counts['completed_tasks']+=value.get('status')=='completed'
        counts['completed_human_turns']+=value['completed_human_turns']
        counts['tasks_over_36k']+=value['context_requirement_met']
        counts['query_reselections']+=len(value['query_reselections'])
        counts['pressure_reselections']+=len(value['pressure_reselections'])
        counts['bridge_errors']+=len(value['backend']['errors'])
        counts['committed_query_refreshes']+=sum(x['selection']=='query' for x in value['committed_refreshes'])
        counts['committed_pressure_refreshes']+=sum(x['selection']=='pressure' for x in value['committed_refreshes'])
    counts['tool_result_error_rate']=counts['tool_error_results']/counts['tool_results'] if counts['tool_results'] else None
    counts['tool_or_command_failure_rate']=counts['tool_or_command_failure_results']/counts['tool_results'] if counts['tool_results'] else None
    invariants=[]
    for ident,value in reports.items():
        for row in value['query_reselections']:
            if row.get('propagation_applied') or row.get('alpha')!=0:invariants.append(dict(case=ident,error='Unexpected graph propagation',row=row))
        for row in value['pressure_reselections']:
            if row.get('pressure_policy') not in (None,'attention'):invariants.append(dict(case=ident,error='Unexpected pressure policy',row=row))
        for row in value['committed_refreshes']:
            if row.get('guided_query_tokens')!=0:invariants.append(dict(case=ident,error='Unexpected guided generation',row=row))
    report=dict(at=time.time(),counts=dict(counts),cases=reports,unexpected_policy=invariants,
        unassigned_graph_rows=unassigned,expected_tasks=len(plan['cases']),
        minimum_human_turns=plan['minimum_turns']*len(plan['cases']),
        caveats=['Repeated identical tool calls are observations, not automatically errors.',
                 'Tool-result errors include recoverable command/edit failures; inspect their original events.',
                 'DSH can mark a Bash result isError=false despite a nonzero exit; report those separately. A failing development test is not by itself an inference/tool-parser failure.',
                 'Basic browser smoke is distinct from full feature acceptance.',
                 'First-round two-client startup wait included completed smoke owners until explicit cleanup.'])
    save(root/'observations.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(json.dumps(summarize(args.root)['counts'],ensure_ascii=False))
