#!/usr/bin/env python3
"""Replay bounded, versioned Query and pressure inputs through native selectors.

This is a counterfactual *selection* analysis. The recorded Mean-K vector and
attention graph remain fixed for every arm; no model is run and no answer
correctness or closed-loop quality is inferred from these rankings. Pressure
arms retain the complete recorded candidate window, including zero seeds.
"""
import argparse
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from vllm_kvmem import _native
from vllm_kvmem.block_graph import GraphPartial, GraphRow
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.graph_scoring import score_graph
from vllm_kvmem.memory import RequestMemory


MAX_NODES = 8192
MAX_RECORDS = 256
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
ARMS = ('mean_k', 'collect', 'legacy_attention', 'locality', 'attention', 'multi')
PRESSURE_ARMS = ('attention_seed', 'locality', 'attention', 'multi')


class _Ready:
    def query(self):
        return True


def _pairs(value):
    if value is None:
        return {}
    items = value.items() if isinstance(value, dict) else value
    result = {int(key): float(score) for key, score in items}
    if any(not math.isfinite(score) or score < 0 for score in result.values()):
        raise ValueError('Invalid recorded legacy attention score')
    return result


def _snapshot(record, block_tokens):
    value = record['graph']
    rows = tuple(GraphRow(int(row['source']),tuple(row['targets']),tuple(row['weights']),
        int(row['samples']),int(row.get('prefill_samples',0)),int(row.get('output_samples',0)))
        for row in value.get('rows',()))
    partial = value.get('partial')
    if partial is not None:
        partial = GraphPartial(int(partial['source']),tuple(partial['targets']),tuple(partial['sums']),
            int(partial['samples']),int(partial.get('prefill_samples',0)),int(partial.get('output_samples',0)))
    spans = tuple(tuple(span) for span in value.get('message_spans',()))
    ends_known = value.get('message_end_known')
    return SimpleNamespace(rows=rows,partial=partial,message_spans=spans,
        message_end_known=tuple((True,)*len(spans) if ends_known is None else ends_known),
        accepted_through=int(value['accepted_through']),
        completed_through=int(value['completed_through']),revision=int(value.get('revision',0)),
        epoch=int(value.get('runtime_epoch',value.get('epoch',0))),block_tokens=block_tokens,
        locality_provenance=value.get('locality_provenance', {}),
        closed=False,bytes=int(value.get('bytes',0)))


def _validate_record(record, *, max_nodes=MAX_NODES):
    if type(max_nodes) is not int or not 0 < max_nodes <= 32768:
        raise ValueError('Invalid bounded replay node capacity')
    version = record.get('version')
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported recorded graph input version')
    from vllm_kvmem.options import validate_graph_aggregation
    validate_graph_aggregation(record.get('aggregation', 'sum'), record.get('scoring_version', 'v1'))
    if version == 1 and (record.get('phase', 'query') != 'query' or
                         record.get('scoring_version', 'v1') != 'v1'):
        raise ValueError('Recorded graph input version 1 only supports legacy Query scoring')
    if version == 2:
        required = ('phase', 'scoring_version', 'alpha', 'score_domain_version',
                    'score_mask', 'transport_mask', 'handles', 'accepted_boundary')
        if any(key not in record for key in required):
            raise ValueError('Recorded graph input version 2 is missing replay metadata')
        if record['phase'] not in ('query', 'pressure'):
            raise ValueError('Unsupported recorded graph selection phase')
        if record['scoring_version'] not in ('v1', 'v2'):
            raise ValueError('Unsupported recorded graph scoring version')
        if record['score_domain_version'] != 'qw3-native-v2':
            raise ValueError('Unsupported recorded Mean-K score-domain version')
        alpha = record['alpha']
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Invalid recorded graph alpha')
    starts,counts,raw,eligible = (record[key] for key in ('starts','counts','raw_scores','eligible'))
    if not 0 < len(starts) <= max_nodes or not len(starts) == len(counts) == len(raw) == len(eligible):
        raise ValueError('Recorded graph index is empty, oversized or shape mismatched')
    cfg = KVMemConfig(**record['config'])
    block = cfg.block_tokens
    phase = record.get('phase', 'query') if version == 2 else 'query'
    if any(type(p) is not int or p < 0 or p % block for p in starts) or len(set(starts)) != len(starts):
        raise ValueError('Recorded index contains invalid or duplicate logical starts')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in raw):
        raise ValueError('Recorded raw seeds must be finite and nonnegative')
    if any(type(value) is not bool for value in eligible):
        raise ValueError('Recorded eligible mask must contain booleans')
    if phase == 'query' and starts != list(range(0,len(starts)*block,block)):
        raise ValueError('Offline native replay requires contiguous original history; holes are unsupported')
    boundary = record.get('accepted_boundary', starts[-1] + counts[-1])
    if type(boundary) is not int or boundary <= 0:
        raise ValueError('Invalid recorded accepted boundary')
    if any(type(n) is not int or not 0 < n <= block or
           p + n > boundary or (n != block and p + n != boundary) for p, n in zip(starts, counts)):
        raise ValueError('Recorded history has an incomplete non-tail block')
    if phase == 'query' and starts[-1] + counts[-1] != boundary:
        raise ValueError('Recorded Query index does not cover its accepted boundary')
    repository = None
    if version == 2:
        for name in ('score_mask', 'transport_mask'):
            if len(record[name]) != len(starts) or any(type(v) is not bool for v in record[name]):
                raise ValueError('Recorded scoring/transport mask shape or type differs')
        if list(eligible) != list(record['transport_mask']):
            raise ValueError('Recorded eligible and transport masks differ')
        handles = record['handles']
        if (len(handles) != len(starts) or any(not isinstance(h, (list, tuple)) or len(h) != 2 or type(h[0]) is not int or
                type(h[1]) is not int or not 0 <= h[0] < cfg.repository_blocks or h[1] <= 0 for h in handles) or
                len({h[0] for h in handles}) != len(handles)):
            raise ValueError('Recorded generation-qualified handles are invalid or duplicated')
        if phase == 'query' and any(not keep and value != 0 for keep, value in zip(record['score_mask'], raw)):
            raise ValueError('Recorded Query seeds contain mass outside the score mask')
    if phase == 'pressure':
        if record['scoring_version'] != 'v2' or record.get('pressure_policy') != 'graph':
            raise ValueError('Recorded graph pressure requires V2 graph policy')
        if record.get('seed_source') not in ('output', 'prefill', 'none'):
            raise ValueError('Invalid recorded pressure seed source')
        repository = record.get('repository_records', ())
        if not 0 < len(repository) <= max_nodes or any(len(r) != 4 for r in repository):
            raise ValueError('Recorded pressure repository is missing or exceeds bounded replay capacity')
        if any(any(type(v) is not int for v in r) or not 0 <= r[0] < cfg.repository_blocks or
               r[1] <= 0 for r in repository) or len({r[0] for r in repository}) != len(repository):
            raise ValueError('Recorded pressure repository has invalid generation-qualified identities')
        ordered = sorted(repository, key=lambda r: r[2])
        if [r[2] for r in ordered] != list(range(0, len(ordered) * block, block)):
            raise ValueError('Offline pressure replay requires a contiguous repository directory')
        if any(r[3] != min(block, boundary-r[2]) or r[3] <= 0 for r in ordered) or ordered[-1][2]+ordered[-1][3] != boundary:
            raise ValueError('Recorded pressure repository counts do not match accepted boundary')
        candidates = record.get('candidate_records', ())
        directory = {r[2]: tuple(r) for r in ordered}
        if len(candidates) != len(starts) or any(len(r) != 5 for r in candidates):
            raise ValueError('Recorded pressure requires its complete candidate records')
        for start, count, handle, candidate in zip(starts, counts, record['handles'], candidates):
            if (any(type(v) is not int for v in candidate) or candidate[4] < 0 or
                    tuple(candidate[:4]) != (handle[0], handle[1], start, count) or
                    tuple(candidate[:4]) != directory.get(start)):
                raise ValueError('Recorded pressure candidate generation, position or count differs')
        if not all(record['score_mask']):
            raise ValueError('Pressure seeds must retain the complete scoring candidate set')
    available = set(starts) if repository is None else {r[2] for r in repository}
    if any(type(p) is not int or p not in available for p in record['mandatory_starts']):
        raise ValueError('Recorded mandatory identity is missing from the index')
    mandatory = set(record['mandatory_starts'])
    expected = [start >= cfg.sink_blocks*block and i < len(starts)-cfg.recent_blocks and start not in mandatory
                for i,start in enumerate(starts)]
    if version == 1 and eligible != expected:
        raise ValueError('Recorded eligible mask differs from native keep bands or mandatory identity')
    budget = record['budget']
    if type(budget) is not int or not 0 < budget <= cfg.select_budget or budget % block:
        raise ValueError('Recorded runtime budget must be block aligned and fit the original K')
    snapshot = _snapshot(record,block)
    if snapshot.accepted_through < boundary:
        raise ValueError('Recorded graph accepted watermark precedes its Query index')
    if len(snapshot.rows) > max_nodes or (snapshot.partial is not None and len(snapshot.partial.targets)>max_nodes):
        raise ValueError('Recorded graph exceeds bounded replay capacity')
    # Production capture is strictly causal. Future-directed operator examples
    # must remain isolated math fixtures, never presented as recorded attention.
    for row in snapshot.rows:
        if (row.source % block or row.samples <= 0 or len(row.targets) > 4 or
                row.prefill_samples+row.output_samples != row.samples or
                any(type(target) is not int or target < 0 or target % block or target >= row.source
                    for target in row.targets)):
            raise ValueError('Recorded attention graph has invalid or noncausal logical identities')
    return cfg,snapshot


def _validate_native_masks(record, memory, handles):
    """V2 delegates keep-band and runtime-budget rules to the same native code."""
    if record['version'] == 1:
        return
    mandatory = [handles[start].slot for start in record['mandatory_starts']]
    if record['phase'] == 'pressure':
        fixed = set(memory._store.fixed_selection_blocks(mandatory, True, record['budget']))
        expected = [handles[start].slot not in fixed for start in record['starts']]
    else:
        frame = memory._store.query_score_frame([handles[start] for start in record['starts']],
            record['counts'], record['accepted_boundary'], mandatory, record['budget'])
        try:
            if list(frame.score_mask) != list(record['score_mask']):
                raise ValueError('Recorded score mask differs from native scoring domain')
            expected = list(frame.transport_mask)
        finally:
            frame.close()
    if expected != list(record['transport_mask']):
        raise ValueError('Recorded transport mask differs from native fixed selection')


def _native_history(config, starts, counts):
    """Recreate selector metadata only, respecting original K/G while appending."""
    memory = RequestMemory('offline-graph-query',config)
    handles = {}
    remaining, local = starts[-1]+counts[-1],0
    try:
        while remaining:
            room = config.select_budget+config.gen_budget-local
            if not room:
                plan = memory.select({},action=False)
                local = plan.total_window_tokens
                memory.attach_completion(_Ready())
                assert memory.commit_selection()
                del plan
                room = config.select_budget+config.gen_budget-local
            count = min(remaining,room)
            spans = memory.begin_append(count,local)
            if spans is None:
                raise BufferError('Recorded repository capacity cannot reconstruct its history')
            # Copy values: Span.handle otherwise retains its Append/Store
            # through pybind reference_internal after RequestMemory.close().
            handles.update((span.history_begin, _native.Handle(span.handle.slot, span.handle.generation))
                           for span in spans)
            memory.attach_completion(_Ready())
            assert memory.commit_append(count)
            remaining -= count
            local += count
        if set(handles) != set(starts):
            raise ValueError('Reconstructed native handles do not match recorded logical blocks')
        return memory,handles
    except BaseException:
        memory.close()
        raise


def _quality(record, selected, baseline):
    evidence = record.get('evidence')
    if evidence is None:
        return dict(required_evidence_coverage=None, required_dependency_coselection=None,
            displaced_mean_k_evidence=None, duplicate_blocks=None, outdated_blocks=None,
            irrelevant_blocks=None, source_span_complete=None, dependency_span_complete=None,
            full_evidence_coselection=None,answer_correctness=None)
    required = set(evidence.get('required_starts',()))
    dependencies = tuple(tuple(pair) for pair in evidence.get('dependency_pairs',()))
    groups = evidence.get('duplicate_groups')
    duplicate = (sum(max(0,len(selected&set(group['starts']))-1) for group in groups)
                 if groups is not None else
                 len(selected & set(evidence['duplicate_starts'])) if 'duplicate_starts' in evidence else None)
    spans = {span['name']:set(span['block_starts']) for span in evidence.get('spans',())}
    source = spans.get('source')
    dependency = spans.get('dependency')
    return dict(required_evidence_coverage=(len(selected & required)/len(required) if required else None),
        required_dependency_coselection=(sum(a in selected and b in selected for a,b in dependencies)/len(dependencies)
                                        if dependencies else None),
        displaced_mean_k_evidence=len((baseline-selected)&required) if required else None,
        duplicate_blocks=duplicate,
        outdated_blocks=len(selected & set(evidence['outdated_starts'])) if 'outdated_starts' in evidence else None,
        irrelevant_blocks=len(selected & set(evidence['irrelevant_starts'])) if 'irrelevant_starts' in evidence else None,
        source_span_complete=source <= selected if source else None,
        dependency_span_complete=dependency <= selected if dependency else None,
        full_evidence_coselection=(source|dependency) <= selected if source and dependency else None,
        answer_correctness=None)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()


def unique_token_span(tokens, needle):
    if not needle:
        raise ValueError('Evidence needle is empty')
    matches = [index for index in range(len(tokens)-len(needle)+1)
               if tokens[index:index+len(needle)] == needle]
    if len(matches) != 1:
        raise ValueError(f'Evidence needle must occur exactly once; found {len(matches)} matches')
    return matches[0],matches[0]+len(needle)


def annotate_fixture(records, fixture, traces, tokenizer):
    """Derive auditable evidence labels; never edit recorded inputs or fixture.

    Native request identities are joined to the retained client trace. Exact
    prompt hashes, including actual first-turn output IDs on continuation, must
    match before labels from the frozen initial archive may be attached.
    """
    if (fixture.get('kind') != 'block_graph_qualification_v1' or
            fixture.get('sha256') != canonical_digest({k:v for k,v in fixture.items() if k!='sha256'})):
        raise ValueError('Fixture kind or immutable hash does not match')
    agents = {agent['index']:agent for agent in fixture['agents']}
    if len(agents) != len(fixture['agents']):
        raise ValueError('Fixture has duplicate agent identities')
    by_turn = {}
    for trace in traces:
        key = trace.get('agent'),trace.get('turn')
        if key in by_turn or key[0] not in agents or key[1] not in (0,1,2):
            raise ValueError('Trace has duplicate, foreign or unsupported agent turn')
        by_turn[key] = trace
    rebuilt,agent_labels,proofs = {},{},[]
    for index,agent in agents.items():
        initial = agent['initial_tokens']
        if agent['first_input_sha256'] != canonical_digest(initial):
            raise ValueError('Fixture initial token fingerprint differs')
        first = by_turn.get((index,0))
        if first is None:
            raise ValueError('Trace is missing the exact first-turn identity')
        before_query = initial + first['output_ids'] + agent['tool_tokens']
        prompts = (initial,before_query+agent['header']+agent['second_question']+agent['footer'],
                   before_query+agent['header']+agent['edited_question']+agent['footer'])
        for turn,prompt in enumerate(prompts):
            trace = by_turn.get((index,turn))
            if trace is None:
                continue
            if trace.get('input_sha256') != canonical_digest(prompt) or trace.get('input_tokens') != len(prompt):
                raise ValueError('Trace prompt fingerprint does not match frozen fixture and accepted output')
            rebuilt[index,turn] = prompt
        project,ledger = f'Project Meridian-{index:02d}',f'ledger-{71+index}'
        deployment,recovery,_ = agent['expected_codes']
        statements = dict(
            source=f'{project} depends on access record {ledger}. When asked about this project, '
                   'resolve its deployment or recovery code through that record.\n',
            dependency=f'Validated access record {ledger}: the deployment code is {deployment}; '
                       f'the recovery code is {recovery}. Both exact strings are authoritative.\n')
        located = {}
        for name,text in statements.items():
            needle = list(tokenizer.encode(text,add_special_tokens=False))
            if needle != agent[name+'_tokens']:
                raise ValueError('Local tokenizer encoding differs from frozen evidence needle')
            begin,end = unique_token_span(initial,needle)
            if end > agent['archive_prefix_tokens']:
                raise ValueError('Evidence needle is outside the frozen archive prefix')
            located[name] = (begin,end)
            proofs.append(dict(agent=index,name=name,text=text,token_ids=needle,
                token_ids_sha256=canonical_digest(needle),begin=begin,end=end,
                first_input_sha256=agent['first_input_sha256']))
        agent_labels[index] = located
    annotated,joins = [],[]
    for number,record in enumerate(records):
        matches = [trace for trace in traces if trace.get('response_id') and
            (record['request_id'] == trace['response_id'] or
             record['request_id'].startswith(trace['response_id']+'-'))]
        if len(matches) != 1:
            raise ValueError('Recorded Query cannot be joined uniquely to a client response')
        trace = matches[0]
        index,turn = trace['agent'],trace['turn']
        origin = by_turn[index,0]['response_id']
        if record['authority_id'] != origin and not record['authority_id'].startswith(origin+'-'):
            raise ValueError('Recorded authority belongs to another fixture session')
        prompt = rebuilt[index,turn]
        block = record['config']['block_tokens']
        if block != 32:
            raise ValueError('Real fixture annotation requires the unchanged B32 geometry')
        positions = set(record['starts'])
        available = dict(zip(record['starts'],record['counts']))
        spans = []
        for name,(begin,end) in agent_labels[index].items():
            covered = list(range(begin//block*block,((end-1)//block+1)*block,block))
            if not set(covered) <= positions or any(start+available[start] < min(end,start+block) for start in covered):
                raise ValueError('Recorded Query index is missing part of a labeled evidence span')
            spans.append(dict(name=name,begin=begin,end=end,block_starts=covered))
        required = set().union(*(set(span['block_starts']) for span in spans))
        observed = {}
        for start,count in zip(record['starts'],record['counts']):
            if start in required or count != block or start+block > len(prompt):
                continue
            tokens = tuple(prompt[start:start+block])
            observed.setdefault(tokens,[]).append(start)
        groups = [dict(starts=starts,token_ids=list(tokens),token_ids_sha256=canonical_digest(list(tokens)))
                  for tokens,starts in observed.items() if len(starts)>1]
        groups.sort(key=lambda group:group['starts'][0])
        source = next(span for span in spans if span['name']=='source')['block_starts']
        dependency = next(span for span in spans if span['name']=='dependency')['block_starts']
        derived = deepcopy(record)
        derived['evidence'] = dict(required_starts=sorted(required),spans=spans,
            dependency_pairs=[[s,t] for s in source for t in dependency],duplicate_groups=groups)
        derived['annotation'] = dict(kind='exact_frozen_fixture_evidence',fixture_sha256=fixture['sha256'],
            agent=index,turn=turn,trace_input_sha256=trace['input_sha256'],
            original_record_sha256=canonical_digest(record),
            scope='Exact source/ledger statement blocks and exact full-block duplicate tokens only; '
                  'outdated/irrelevant labels and counterfactual answer correctness are unknown')
        annotated.append(derived)
        joins.append(dict(record=number,request_id=record['request_id'],authority_id=record['authority_id'],
            agent=index,turn=turn,trace_response_id=trace['response_id'],
            trace_input_sha256=trace['input_sha256'],original_record_sha256=canonical_digest(record),
            labeled_record_sha256=canonical_digest(derived),required_starts=sorted(required),
            duplicate_groups=len(groups)))
    return annotated,dict(version=1,fixture_sha256=fixture['sha256'],needle_proofs=proofs,joins=joins,
        limitations=['Exact evidence coverage is a selector metric, not generated-answer correctness.',
            'All arms reuse one recorded trajectory; generated continuations and graphs are not regenerated.',
            'Duplicate counts measure extra selected copies of byte-identical token blocks; semantic redundancy is unknown.',
            'Outdated and irrelevant content are unlabelled and remain null.'])


def summarize_records(results):
    summaries = {}
    metrics = ('required_evidence_coverage','required_dependency_coselection',
        'source_span_complete','dependency_span_complete','full_evidence_coselection',
        'displaced_mean_k_evidence','duplicate_blocks','outdated_blocks','irrelevant_blocks','answer_correctness')
    groups = list(dict.fromkeys((record['phase'], record['scoring_version'], record.get('aggregation','sum'), arm)
                               for record in results for arm in record['arms']))
    mixed = len({(phase, version, aggregation) for phase, version, aggregation, _arm in groups}) > 1
    for phase, version, aggregation, arm in groups:
        data = [record['arms'][arm] for record in results if arm in record['arms'] and
                record['phase'] == phase and record['scoring_version'] == version and
                record.get('aggregation','sum') == aggregation]
        summary = dict(records=len(data),added_blocks=sum(len(row['added_starts']) for row in data),
                       displaced_blocks=sum(len(row['displaced_starts']) for row in data))
        for metric in metrics:
            values = [row['quality'].get(metric) for row in data]
            known = [value for value in values if value is not None]
            summary[metric] = dict(labeled_records=len(known),
                mean=math.fsum(known)/len(known) if known else None,
                total=math.fsum(known) if known else None)
        prefix = f'{phase}:{version}:' + ('max:' if aggregation == 'max' else '')
        summaries[prefix+arm if mixed else arm] = summary
    return summaries


def analyze_record(record, *, locality_weight=None, attention_weight=None,
                   cross_message_weight=None, legacy_fraction=.25):
    config,snapshot = _validate_record(record)
    starts,raw,eligible = (record[key] for key in ('starts','raw_scores','eligible'))
    phase = record.get('phase', 'query') if record['version'] == 2 else 'query'
    scoring_version = record['scoring_version'] if record['version'] == 2 else 'v1'
    if scoring_version == 'v2' and (locality_weight is not None or attention_weight is not None):
        raise ValueError('Legacy weight overrides do not apply to V2; its recorded alpha fixes both shares')
    alpha = record.get('alpha', .2)
    aggregation = record.get('aggregation', 'sum')
    locality_weight = record.get('locality_weight', .1) if locality_weight is None else locality_weight
    attention_weight = record.get('attention_weight', .2) if attention_weight is None else attention_weight
    cross_message_weight = record.get('cross_message_weight', .25) if cross_message_weight is None else cross_message_weight
    if not math.isfinite(legacy_fraction) or not 0 <= legacy_fraction <= 1:
        raise ValueError('Legacy comparison fraction must lie in [0,1]')
    modes = dict(mean_k='off',collect='collect',legacy_attention='off',
                 attention_seed='off', locality='locality',attention='attention',multi='multi')
    arms = PRESSURE_ARMS if phase == 'pressure' else ARMS
    repository = (sorted(record['repository_records'], key=lambda r: r[2]) if phase == 'pressure'
                  else [(None, None, start, count) for start, count in zip(starts, record['counts'])])
    history_starts, history_counts = [r[2] for r in repository], [r[3] for r in repository]
    results,baseline = {},None
    for arm in arms:
        legacy = arm == 'legacy_attention'
        cfg = replace(config,action_attention=legacy,action_fraction=legacy_fraction if legacy else 0.)
        memory,handles = _native_history(cfg,history_starts,history_counts)
        propagated = None
        try:
            _validate_native_masks(record, memory, handles)
            propagated = score_graph(raw,starts,eligible,snapshot,mode=modes[arm],
                scoring_version=scoring_version, alpha=alpha, aggregation=aggregation,
                locality_weight=locality_weight,attention_weight=attention_weight,
                cross_message_weight=cross_message_weight,block_tokens=cfg.block_tokens)
            if legacy:
                memory.set_action_scores(_pairs(record.get('legacy_action_scores',{})))
            mandatory = tuple(handles[start].slot for start in record['mandatory_starts'])
            if phase == 'pressure':
                # Every C node constrains native pressure fallback, including
                # zeros. An empty or positive-only list can reopen cold history.
                candidates = [(handles[start].slot, value) for start, value in zip(starts, propagated.scores)]
                slots = memory.preview_selection({}, mandatory, action=False,
                    pressure_scores=candidates, max_tokens=record['budget'])
            else:
                slots = memory.preview_selection({handles[start].slot:score
                    for start,score in zip(starts,propagated.scores)}, mandatory,
                    action=legacy,prefill_scores=_pairs(record.get('prefill_scores',{})) if legacy else None,
                    max_tokens=record['budget'])
            slot_positions = {handle.slot:start for start,handle in handles.items()}
            selected = {slot_positions[slot] for slot in slots}
            if baseline is None:
                baseline = selected
                recorded_baseline = record.get('baseline_selected_starts')
                if recorded_baseline is not None and sorted(recorded_baseline) != sorted(baseline):
                    raise ValueError('Recorded baseline selection differs from native replay')
            counts = dict(zip(history_starts,history_counts))
            used = sum(counts[start] for start in selected)
            if used > record['budget']:
                raise RuntimeError('Native replay exceeded its fixed token budget')
            observed = [(row.source,target,weight) for row in snapshot.rows
                        for target,weight in zip(row.targets,row.weights)]
            retained_mass = math.fsum(weight for _,_,weight in observed)
            diagnostics = dict(propagated.diagnostics)
            # CPU propagation time excludes attention capture, graph building,
            # model inference, restoration and request scheduling.
            results[arm] = dict(selected_starts=sorted(selected),selected_tokens=used,
                added_starts=sorted(selected-baseline),displaced_starts=sorted(baseline-selected),
                retained_raw_seed_mass=math.fsum(score for start,score in zip(starts,raw) if start in selected),
                retained_raw_mean_k_mass=(math.fsum(score for start,score in zip(starts,raw) if start in selected)
                                          if phase == 'query' else None),
                legacy_fraction=legacy_fraction if legacy else 0.,
                observed_edge_coselection=(sum(a in selected and b in selected for a,b,_ in observed)/len(observed)
                                          if observed else None),
                observed_mass_coselection=(math.fsum(w for a,b,w in observed if a in selected and b in selected)/retained_mass
                                          if retained_mass else None),
                diagnostics=diagnostics,quality=_quality(record,selected,baseline))
        finally:
            if propagated is not None:
                propagated.close()
            assert memory.close()
    return dict(version=2,input_version=record['version'],phase=phase,scoring_version=scoring_version,
        aggregation=aggregation,
        alpha=alpha if scoring_version == 'v2' else None,
        score_domain_version=record.get('score_domain_version', 'legacy-python-v1'),
        seed_source=record.get('seed_source', 'mean_k'),
        authority_id=record.get('authority_id'),request_id=record.get('request_id'),
        captured_mode=record.get('mode'),input_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
        annotation=record.get('annotation'),
        history_blocks=len(history_starts),candidate_blocks=len(starts),
        budget=record['budget'],block_tokens=config.block_tokens,
        locality_provenance=snapshot.locality_provenance,
        recorded_handles=record.get('handles'),
        baseline_selected_starts=sorted(baseline),
        labeled_fixture=record.get('evidence') is not None,
        observed_source_coverage=len(snapshot.rows)/len(starts),
        mean_retained_attention_mass=(math.fsum(row.retained_mass for row in snapshot.rows)/len(snapshot.rows)
                                      if snapshot.rows else 0.),
        arms=results)


def causal_fixture():
    """Ground-truth labels are explicitly synthetic, not measured task quality."""
    return dict(version=1,authority_id='synthetic-causal',request_id='synthetic-query',mode='synthetic',
        starts=[0,4,8,12,16,20],counts=[4]*6,raw_scores=[.01,.10,.15,.12,.02,.60],
        eligible=[True]*6,mandatory_starts=[],budget=12,
        config=dict(block_tokens=4,select_budget=12,gen_budget=12,repository_blocks=16,
                    sink_blocks=0,recent_blocks=0,action_attention=True,action_fraction=0.,action_combine='quota'),
        graph=dict(rows=[dict(source=20,targets=[4],weights=[.9],samples=4,prefill_samples=3,output_samples=1)],
                   partial=None,message_spans=[],message_end_known=[],accepted_through=24,
                   completed_through=24,revision=1),
        legacy_action_scores={0:.7,4:.3},prefill_scores={4:1.},
        evidence=dict(required_starts=[4,20],dependency_pairs=[[20,4]],
                      duplicate_starts=[],outdated_starts=[16],irrelevant_starts=[12]))


def load_records(paths, limit):
    if not 1 <= limit <= MAX_RECORDS:
        raise ValueError('Offline record limit must be between 1 and 256')
    records,sources = [],[]
    for filename in paths:
        path = Path(filename)
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError('Recorded input file exceeds the offline analysis limit')
        payload = path.read_bytes()
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(payload).hexdigest()))
        for line in payload.splitlines():
            if not line.strip():
                continue
            if len(line) > MAX_RECORD_BYTES:
                raise ValueError('Recorded Query exceeds its bounded export size')
            if len(records) >= limit:
                return records,sources
            records.append(json.loads(line))
    return records,sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',action='append',default=[],help='Recorded block_graph_inputs.jsonl; repeatable')
    parser.add_argument('--synthetic',action='store_true',help='Include the explicitly labeled causal fixture')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-records',type=int,default=64)
    parser.add_argument('--locality-weight',type=float,help='Explicit V1 counterfactual; otherwise use recorded value')
    parser.add_argument('--attention-weight',type=float,help='Explicit V1 counterfactual; otherwise use recorded value')
    parser.add_argument('--cross-message-weight',type=float,help='Explicit boundary counterfactual; otherwise use recorded value')
    parser.add_argument('--legacy-fraction',type=float,default=.25)
    parser.add_argument('--fixture',type=Path,help='Unmodified real-model block graph fixture for exact evidence labels')
    parser.add_argument('--traces',type=Path,help='Retained qualification/traces.json for request/agent and prompt hash joins')
    parser.add_argument('--tokenizer',type=Path,help='Local tokenizer directory; no network or model inference')
    args = parser.parse_args(argv)
    if not args.input and not args.synthetic:
        parser.error('Provide recorded --input or explicit --synthetic')
    records,sources = load_records(args.input,args.max_records)
    annotation = None
    if any((args.fixture,args.traces,args.tokenizer)):
        if not all((args.fixture,args.traces,args.tokenizer)) or args.synthetic:
            parser.error('Use --fixture, --traces and --tokenizer together with recorded inputs only')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()),local_files_only=True,
                                                  trust_remote_code=False)
        fixture = json.loads(args.fixture.read_text())
        traces = json.loads(args.traces.read_text())
        records,annotation = annotate_fixture(records,fixture,traces,tokenizer)
        annotation['files'] = {name:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for name,path in (('fixture',args.fixture),('traces',args.traces))}
        annotation['recorded_inputs'] = sources
        annotation['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        annotation['tokenizer'] = dict(path=str(args.tokenizer.resolve()),class_name=type(tokenizer).__name__,
            files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (args.tokenizer/'tokenizer.json',args.tokenizer/'tokenizer_config.json',args.tokenizer/'config.json')
                   if path.is_file()})
    if args.synthetic:
        if len(records) == args.max_records:
            parser.error('Increase --max-records to include the synthetic fixture')
        records.append(causal_fixture())
    results = [analyze_record(record,locality_weight=args.locality_weight,
        attention_weight=args.attention_weight,cross_message_weight=args.cross_message_weight,
        legacy_fraction=args.legacy_fraction) for record in records]
    output = dict(version=2,kind='fixed_recorded_input_selection_analysis',sources=sources,
        limitations=['All arms reuse the same recorded raw seeds, candidate masks and attention observations.',
                     'No model inference, answer correctness, end-to-end throughput or closed-loop quality is measured.',
                     'Evidence metrics are null unless an explicitly labeled fixture was supplied.',
                     'Legacy Query attention comparison uses the stated fraction; K/G and runtime budget are unchanged.',
                     'V1 captures retain their original Mean-K denominator; replay cannot reconstruct aligned V2 Mean-K seeds.',
                     'Pressure arms use native pressure selection over every recorded working candidate, including zeros.'],
        weights=dict(locality=args.locality_weight,attention=args.attention_weight,
                     cross_message=args.cross_message_weight,legacy_fraction=args.legacy_fraction),
        summary=summarize_records(results),records=results)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    if annotation is not None:
        (args.output_dir/'annotated-inputs.jsonl').write_text(''.join(json.dumps(record,allow_nan=False)+'\n' for record in records))
        (args.output_dir/'annotation-provenance.json').write_text(json.dumps(annotation,indent=2,allow_nan=False)+'\n')
    if args.synthetic:
        (args.output_dir/'synthetic-input.jsonl').write_text(json.dumps(causal_fixture(),allow_nan=False)+'\n')
    with (args.output_dir/'summary.csv').open('w',newline='') as stream:
        columns=['record','phase','scoring_version','arm','selected_tokens','added_blocks','displaced_blocks',
                 'retained_raw_seed_mass','retained_raw_mean_k_mass','observed_mass_coselection','required_evidence_coverage',
                 'required_dependency_coselection','source_span_complete','dependency_span_complete',
                 'full_evidence_coselection','displaced_mean_k_evidence','duplicate_blocks',
                 'outdated_blocks','irrelevant_blocks','answer_correctness','propagation_ms']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for index,record in enumerate(results):
            for arm,row in record['arms'].items():
                writer.writerow(dict(record=index,phase=record['phase'],scoring_version=record['scoring_version'],
                    arm=arm,selected_tokens=row['selected_tokens'],
                    added_blocks=len(row['added_starts']),displaced_blocks=len(row['displaced_starts']),
                    retained_raw_mean_k_mass=row['retained_raw_mean_k_mass'],
                    retained_raw_seed_mass=row['retained_raw_seed_mass'],
                    observed_mass_coselection=row['observed_mass_coselection'],
                    required_evidence_coverage=row['quality']['required_evidence_coverage'],
                    required_dependency_coselection=row['quality']['required_dependency_coselection'],
                    **{key:row['quality'][key] for key in ('source_span_complete','dependency_span_complete',
                        'full_evidence_coselection','displaced_mean_k_evidence','duplicate_blocks',
                        'outdated_blocks','irrelevant_blocks','answer_correctness')},
                    propagation_ms=1000*row['diagnostics']['scoring_seconds']))
    with (args.output_dir/'arm-summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','metric','labeled_records','mean','total'])
        writer.writeheader()
        for arm,summary in output['summary'].items():
            for metric,value in summary.items():
                if isinstance(value,dict):
                    writer.writerow(dict(arm=arm,metric=metric,**value))
    print(json.dumps(dict(records=len(results),arms=sorted({arm for result in results for arm in result['arms']}),
                         output=str(args.output_dir.resolve()))))
    return output


if __name__ == '__main__':
    main()
