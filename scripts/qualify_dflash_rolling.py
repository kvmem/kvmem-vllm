"""Real two-owner DFlash qualification for a shared Query/window epoch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid

from qualify_block_graph import (build_fixture, expected_output_samples, native_matches,
    records, require, save, stream_request, wait_released, wait_terminal_publication)


def run(args):
    profile=json.loads(args.config.read_text());kv=profile['kvmem']
    require(profile['draft_method']=='dflash' and profile['draft_tokens'] in (1,3,7) and
        profile['execution_mode']=='full_graph' and profile['verify_session_restore'] and
        kv['rolling_active_window'],'Require verified rolling DFlash k=1/3/7 with FULL graph configuration')
    args.output.mkdir(parents=True,exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    encode=lambda s:tokenizer.encode(s,add_special_tokens=False)
    padding=encode(' ')
    require(len(padding)==1 and 0<=args.tool_padding_tokens<=31,'Tool padding must be 0..31 single-space tokens')
    require(2<=args.concurrency<=profile['max_num_seqs'],'Qualification concurrency must fit the native request limit')
    fixture=build_fixture(tokenizer,prompt_tokens=12288,concurrency=args.concurrency)
    save(args.output/'fixture.json',fixture)
    sessions=['dflash-rolling-'+uuid.uuid4().hex for _ in range(args.concurrency)]
    rows=[];errors=[];report={};started=time.monotonic()

    def client(index):
        agent=fixture['agents'][index];tokens=list(agent['initial_tokens'])
        query=agent['query_span'];spans=list(agent['message_spans'])
        for turn,count in enumerate((257,257,4096,257)):
            row=stream_request(profile,sessions[index],tokens,count,count if turn==2 else 129,
                query=query,spans=spans,reselect='auto')
            row.update(agent=index,turn=turn,expected_output_samples=
                expected_output_samples(tokenizer,profile,tokens,row['output_ids']))
            rows.append(row);save(args.output/f'owner-{index}-turn-{turn}.json',row)
            tokens=tokens+row['output_ids']
            if turn<2:
                prefix=encode('<|im_end|>\n<|im_start|>tool\n')
                begin=len(tokens)+len(encode('<|im_end|>\n'))
                tokens+=prefix+encode('Actual immutable file evidence: '+('field=value; '*160))
                if turn==1:tokens+=padding*args.tool_padding_tokens
                tokens+=encode('<|im_end|>\n')
                spans.append(dict(begin=begin,end=len(tokens),role='tool'))
                tokens+=encode('<|im_start|>assistant\n<think>\n\n</think>\n\n')
            elif turn==2:
                tokens+=encode('<|im_end|>\n');begin=len(tokens)
                tokens+=agent['header'];qb=len(tokens);tokens+=agent['second_question'];qe=len(tokens)
                tokens+=agent['footer'];query=[qb,qe]
                spans.append(dict(begin=begin,end=qe+agent['message_close_tokens'],role='user'))

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures=[pool.submit(client,i) for i in range(args.concurrency)]
            for future in futures:future.result()
        archive=wait_terminal_publication(args.capture,rows)
        refresh=records(args.capture/'bounded_refresh.jsonl')
        rollback=records(args.capture/'verified_rolling_pressure.jsonl')
        actions=records(args.capture/'action_attention.jsonl')
        restores=records(args.capture/'verified_restores.jsonl')
        proofs=[]
        for row in rows:
            own=[q for q in refresh if native_matches(row,q)]
            queries=[q for q in own if q['selection']=='query']
            if row['turn'] in (0,3):require(len(queries)==1,'New real Query did not select exactly once')
            else:require(not queries,'Tool continuation invented a Query')
            if row['turn']==1:
                require(not own,'Tool continuation compacted before actual capacity pressure')
            if row['turn']==2:
                valid=[q for q in rollback if native_matches(row,q)]
                require(valid and all(q['recurrent_byte_equal'] and q['target_kv_byte_equal'] and
                    q['draft_byte_equal'] and q['draft_layers']>=5 for q in valid),
                    'Missing target/GDN/counter/draft bytes at real decode pressure')
                cap=kv['select_budget']+kv['gen_budget']-kv['block_tokens']
                require(own and all(q['selection']=='pressure' and q['local_tokens_before']==cap for q in own),
                    'Decode pressure did not occur exactly at actual capacity boundary')
            own_actions=[q for q in actions if native_matches(row,q)]
            require(len(own_actions)==1 and own_actions[0]['samples']==row['expected_output_samples'],
                    'Private or rejected row contaminated accepted attention')
            if row['turn']:
                own_restore=[q for q in restores if native_matches(row,q)]
                require(own_restore and all(q['byte_equal'] and q['draft_layers']==5 for q in own_restore),
                        'Missing exact session restore before continuation')
            proofs.append(dict(agent=row['agent'],turn=row['turn'],queries=len(queries),refreshes=len(own)))
        require(archive['session_checkpoint_deferrals']==0,'Checkpoint quota fallback')
        require(any(q['previous_accepted_step']['accepted']>1 for q in rollback),
                'No multi-token DFlash acceptance before capacity selection')
        report=dict(status='passed',draft_tokens=profile['draft_tokens'],concurrency=args.concurrency,
            tool_padding_tokens=args.tool_padding_tokens,
            proofs=proofs,rolling_byte_proofs=rollback,
            restored=len(restores),output_tokens=sum(len(r['output_ids']) for r in rows),
            scope='Fixed quotas; exact Query identity, warm continuation, actual decode capacity, byte and accepted-row gates; not a task-quality benchmark')
    except BaseException as error:
        errors.append(repr(error));raise
    finally:
        for session in sessions:
            try:stream_request(profile,session,[fixture['eos_token_id']],1,1,release=True)
            except Exception as error:errors.append('release: '+repr(error))
        try:wait_released(args.capture,sessions,timeout=60)
        except Exception as error:errors.append('release check: '+repr(error))
        report.update(errors=errors,rows=rows,seconds=time.monotonic()-started,
                      all_owned_sessions_released=not any(e.startswith('release') for e in errors))
        if errors:report['status']='failed'
        save(args.output/'results.json',report)
    require(report['status']=='passed','Rolling DFlash gate failed')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','rolling_byte_proofs')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--tool-padding-tokens',type=int,default=0,
        help='Append space tokens inside the second tool result to exercise a different speculative boundary alignment')
    for name in ('config','capture','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
