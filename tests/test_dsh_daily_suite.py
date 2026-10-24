from scripts.run_dsh_daily_suite import event_statistics, history_since, make_plan


def test_tool_failures_and_repetitions_are_distinct_from_failed_turns():
    events = [
        dict(seq=1,type='tool/call',data=dict(name='read',arguments='{"path":"a"}')),
        dict(seq=2,type='tool/result',data=dict(message=dict(content=[dict(isError=False)]))),
        dict(seq=3,type='tool/call',data=dict(name='read',arguments=dict(path='a'))),
        dict(seq=4,type='tool/result',data=dict(message=dict(content=[dict(isError=True)]))),
        dict(seq=5,type='turn/end',data=dict(reason=dict(kind='completed'))),
    ]
    stats=event_statistics(events)
    assert stats['tool_calls']==2 and stats['tool_results']==2
    assert stats['tool_error_results']==1 and stats['tool_result_error_rate']==.5
    assert stats['repeated_identical_calls']==1
    assert stats['turn_endings'][0]['reason']['kind']=='completed'


def test_history_paginates_before_old_boundary_without_loss_or_duplicates(monkeypatch):
    import scripts.run_dsh_daily_suite as suite
    pages=[dict(events=[dict(event=dict(seq=i)) for i in [9,10]],hasMore=True,projections={'x':1}),
           dict(events=[dict(event=dict(seq=i)) for i in [6,7,8]],hasMore=True),
           dict(events=[dict(event=dict(seq=i)) for i in [3,4,5]],hasMore=True)]
    cursors=[]
    def call(url,method,payload):
        cursors.append(payload.get('beforeSeq'));return pages.pop(0)
    monkeypatch.setattr(suite,'rpc',call)
    rows,projection=history_since('local','s',4)
    assert [r['seq'] for r in rows]==list(range(5,11))
    assert cursors==[None,9,6] and projection=={'x':1}


def test_every_attempt_has_a_distinct_workspace_and_at_least_three_followups(tmp_path):
    plan=make_plan(tmp_path)
    assert len(plan['cases'])==24
    assert len({c['session_id'] for c in plan['cases']})==24
    assert len({c['workspace'] for c in plan['cases']})==24
    assert len({c['preview_port'] for c in plan['cases']})==24
    assert all(len(c['prompts'])==4 for c in plan['cases'])
    assert plan['minimum_logical_input_tokens_exclusive']==36864


def test_bash_nonzero_exit_is_counted_even_when_dsh_tool_envelope_succeeded():
    events=[dict(seq=1,type='tool/call',data=dict(callId='b',name='bash',arguments={})),
        dict(seq=2,type='tool/result',data=dict(message=dict(content=[dict(
            toolCallId='b',isError=False,content=[dict(type='text',text='Assertion failed\n[exit code: 1]')])])))]
    stats=event_statistics(events)
    assert stats['tool_error_results']==0
    assert stats['bash_nonzero_exit_results']==1 and stats['tool_or_command_failure_rate']==1
    events[-1]['data']['message']['content'][0]['isError']=True
    assert event_statistics(events)['tool_or_command_failure_results']==1
