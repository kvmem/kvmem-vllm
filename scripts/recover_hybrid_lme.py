"""Re-audit saved complete responses and resume only unanswered LME cases.

No inference is performed here. Old attempts, failed results and captures are
read-only; the recovery root contains copied responses with SHA256 bindings.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from benchmark_niah import save, sha
from benchmark_memory import source as load_input
from benchmark_longmemeval import SAMPLING
import hybrid_memory_protocol as policy
import qualify_graph_niah as client
import run_hybrid_memory_campaign as campaign

require = policy.require
CAPTURES = ('bounded_refresh.jsonl','block_graph.jsonl','block_graph_inputs.jsonl',
    'action_attention.jsonl','archive.json','scheduler_resources.json')


def read(path):
    return json.loads(Path(path).read_text())


def preserve_case(source, target, rebuilt, evidence):
    require(not target.exists(),'Never overwrite an inherited case')
    before = {p.name:sha(p) for p in source.iterdir() if p.is_file()}
    original = read(source/'result.json')
    require(rebuilt['status'] == 'completed' and rebuilt['id'] == original['id'],
        'Recovery must keep original case identity')
    target.mkdir(parents=True)
    for name in before:
        destination = 'source-failure-result.json' if name == 'result.json' and original['status'] != 'completed' else name
        shutil.copy2(source/name,target/destination)
    if original['status'] != 'completed':
        save(target/'result.json',rebuilt)
    else:
        require(original['answer'] == rebuilt['answer'] and
            original['raw_completion'] == rebuilt['raw_completion'],
            'Already completed answer differs from original saved response')
    save(target/'recovery-evidence.json',evidence)
    require(before == {p.name:sha(p) for p in source.iterdir() if p.is_file()},'Source case changed during recovery')
    return before


def prepare(root, previous):
    old_fixture, old_profile = campaign.verify(previous)
    require(old_fixture['suite'] == 'longmemeval_s' and not old_fixture['qualification'],
        'Recovery supports the original independent LME cohort only')
    require(read(previous/'status.json')['status'] == 'failed','Only recover a preserved failed attempt')
    cleanup = read(previous/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],'Original GPU cleanup incomplete')
    campaign.prepare(root,'longmemeval_s')
    fixture,profile = campaign.verify(root)
    require(profile == old_profile and fixture['rows'] == old_fixture['rows'] and
        fixture['source_cohort_sha256'] == old_fixture['source_cohort_sha256'],
        'Recovery changed original profile, inputs, cohort or sampling')
    old_runtime = read(previous/'runtime-proof.json')['files']
    new_runtime = read(root/'runtime-proof.json')['files']
    engine = lambda files:{k:v for k,v in files.items() if k.startswith(('src/','csrc/','third_party/'))
        or k in ('setup.py','pyproject.toml')}
    require(engine(old_runtime) == engine(new_runtime),'Inference/scoring runtime differs; audit-only recovery refused')
    changed_scripts = {k:dict(before=old_runtime.get(k),after=new_runtime.get(k))
        for k in sorted(set(old_runtime)|set(new_runtime)) if old_runtime.get(k) != new_runtime.get(k)}
    capture = previous/'run/server/capture'
    logs = {name:[json.loads(line) for line in (capture/name).read_text().splitlines()]
        for name in CAPTURES if name.endswith('.jsonl')}
    terminal_rows = read(capture/'archive.json')['completed']; scheduler = read(capture/'scheduler_resources.json')
    answers = previous/'run/answers'; output = root/'recovered-answers'
    output.mkdir(); shutil.copy2(answers/'manifest.json',output/'manifest.json')
    rows = {r['id']:r for r in fixture['rows']}; originals = {}; recovered = []; repaired = []
    for directory in sorted((answers/'cases').iterdir()):
        require(directory.name in rows and (directory/'response.json').is_file(),
            'Attempt lacks a complete saved response; do not silently regenerate it')
        row = rows[directory.name]; src = load_input(row); response = read(directory/'response.json')
        request = read(directory/'request.json'); prior_result = read(directory/'result.json')
        choice = response['choices'][0]
        original_sampling = read(Path(fixture['reference'])/'results/attention/cases/longmemeval_s'/row['id']/'request_config.json')
        require(all(request[key] == original_sampling[key] for key in SAMPLING) and
            request['max_tokens'] == profile['kvmem']['gen_budget'] and request['thinking_token_budget'] == 4096 and
            request['token_sha256'] == row['token_sha256'] and request['prompt_tokens'] == row['input_tokens'] and
            request['kvmem_query_span'] == [row['query_begin'],row['query_end']],
            'Original request sampling, seed, token identity or Query span differs')
        require(response['usage']['prompt_tokens'] == len(src['tokens']) == row['input_tokens'] and
            choice['prompt_token_ids'] == src['tokens'] and choice['finish_reason'] in ('stop','length') and
            response['usage']['completion_tokens'] == len(choice['token_ids']) > 0,
            'Original saved response is incomplete or bound to different input')
        evidence = policy.audit_case(capture,response['id'],profile,request['kvmem_query_span'],
            require_pressure=True,event_reader=lambda p:logs[p.name],prompt_tokens=len(src['tokens']),
            output_tokens=request['max_tokens'])
        matches = lambda e:e['request_id'].startswith(response['id']+'-')
        terminal = next(e for e in terminal_rows if matches(e))
        require(terminal.get('retirement_finished_at') and terminal.get('retirement_ack_ready_at') and
            client.owner_released(scheduler,request['session_id'],{'response_id':response['id']},terminal),
            'Original saved response lacks complete retirement/release evidence')
        bounded = [e for e in logs['bounded_refresh.jsonl'] if matches(e)]
        actions = [e for e in logs['action_attention.jsonl'] if matches(e)]
        require(len(actions) == 1,'Missing original accepted-attention evidence')
        raw = choice['text']; answer = raw.split('</think>',1)[-1].strip()
        release = dict(method='ordinary_retirement',namespace=[request['session_id']]*2,released=True,
            retirement_ack_ready_at=terminal['retirement_ack_ready_at'],scheduler_observed_at=scheduler['at'],
            other_requests=scheduler.get('requests'))
        record = dict(src['info'],id=row['id'],suite='longmemeval_s',status='completed',answer=answer,
            raw_completion=raw,thinking_closed='</think>' in raw,usage=response['usage'],metrics=response.get('metrics'),
            seconds=prior_result['seconds'],prompt_tokens=len(src['tokens']),token_sha256=row['token_sha256'],
            prompt_sha256=row['prompt_sha256'],query_sha256=row['query_sha256'],original_input_intact=True,
            attention=evidence['query'],pressure_refreshes=len(evidence['pressure']),finish_reason=choice['finish_reason'],
            hit_output_cap=response['usage']['completion_tokens'] >= request['max_tokens'],graph_v2=evidence,release=release,
            recovery_timing_scope='Original saved client duration; failed-case duration includes preparation/audit. '
                'Original server metrics retained. Not a new inference timing measurement.')
        require(not record['hit_output_cap'],'Original saved answer reached its output cap')
        before = preserve_case(directory,output/'cases'/row['id'],record,dict(graph_v2=evidence,
            terminal=terminal,action_attention=actions[0],release=release,source=str(directory.resolve())))
        target = output/'cases'/row['id']
        for name,value in (('events.json',bounded),('terminal.json',terminal),('action_attention.json',actions[0])):
            if not (target/name).exists(): save(target/name,value)
        originals.update({str((directory/name).relative_to(previous)):digest for name,digest in before.items()})
        recovered.append(row['id'])
        if prior_result['status'] != 'completed': repaired.append(row['id'])
    require(recovered and len(set(recovered)) == len(recovered),'No unique original answers to recover')
    originals.update({str((capture/name).relative_to(previous)):sha(capture/name) for name in CAPTURES})
    for name in ('prepared-proof.json','fixture.json','profile.json','runtime-proof.json','run/answers/manifest.json'):
        originals[name] = sha(previous/name)
    proof = dict(source=str(previous.resolve()),source_files=originals,recovered_ids=recovered,
        repaired_audit_ids=repaired,original_successes_unchanged=len(recovered)-len(repaired),
        regenerated_answers=0,engine_sources_identical=True,audit_script_changes=changed_scripts,
        recovered_files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    save(root/'recovery-proof.json',proof)
    save(output/'summary.json',dict(expected=102,completed=len(recovered),errors=0,hit_output_cap=0,
        inherited_answers=len(recovered),regenerated_answers=0))
    # Include the initial summary in the immutable bootstrap proof too.
    proof['recovered_files']['summary.json'] = sha(output/'summary.json'); save(root/'recovery-proof.json',proof)
    fixture['recovery_source'] = str(previous.resolve())
    fixture['recovery_cases'] = recovered
    save(root/'fixture.json',fixture)
    save(root/'prepared-proof.json',{n:sha(root/n) for n in
        ('profile.json','fixture.json','runtime-proof.json','recovery-proof.json')})
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    args = parser.parse_args()
    proof = prepare(args.root.resolve(),args.previous.resolve())
    print(json.dumps(dict(inherited=len(proof['recovered_ids']),repaired_audits=len(proof['repaired_audit_ids']),
        regenerated_answers=proof['regenerated_answers'])))
