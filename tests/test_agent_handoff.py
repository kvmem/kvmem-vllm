"""Fast handoff preserves full parser output and exact history identities."""
from copy import deepcopy

import pytest

from benchmarks.deepswe.claude_bridge import _identity_key, message_identity
from scripts.dsh_bridge import QwenStream, to_anthropic
from test_dsh_bridge import protocol, parsed_response, assemble


def test_repeated_cancellation_keeps_host_slot_until_producer_finishes():
    import asyncio
    import threading
    from scripts.dsh_bridge import bounded_host_work
    async def check():
        entered=threading.Event();release=threading.Event();committed=[]
        slots=asyncio.Semaphore(1)
        def produce():
            entered.set();assert release.wait(5);committed.append('done')
        task=asyncio.create_task(bounded_host_work(slots,produce))
        while not entered.is_set():await asyncio.sleep(.001)
        task.cancel();await asyncio.sleep(.01)
        task.cancel();await asyncio.sleep(.01)
        assert not task.done() and slots.locked() and not committed
        release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert committed==['done'] and not slots.locked()
    asyncio.run(check())


@pytest.mark.parametrize('left,right', [(True, 1), (1, 1.0), (0.0, -0.0),
    ({'x': [True]}, {'x': [1]}), ('{"x":1}', '{"x":1.0}')])
def test_identity_cache_cannot_hide_json_type_or_argument_edits(left, right):
    assert _identity_key(left) != _identity_key(right)


def test_identity_cache_key_owns_nested_structure():
    obj = {'tools': [{'id': 'x', 'args': ['a']}]}
    key = _identity_key(obj)
    assert key == _identity_key(deepcopy(obj))
    obj['tools'][0]['args'].append('b')
    assert key != _identity_key(obj)


@pytest.mark.parametrize('content', [
    [{'type':'text','text':'中文 answer'}],
    [{'type':'thinking','thinking':'reason','signature':'x'},
     {'type':'tool_use','id':'call-a','name':'edit','input':{'old':'x','new':'y\n中文'}}],
    [{'type':'tool_use','id':'one','name':'read','input':{'path':'x'}},
     {'type':'tool_use','id':'two','name':'read','input':{'path':'y'}}],
])
def test_message_local_assistant_conversion_matches_complete_protocol(protocol, content):
    assistant = {'role':'assistant','content':content}
    payload = dict(model='local',max_tokens=128,system='system',tools=[{'name':'read','input_schema':{'type':'object'}}],
        messages=[{'role':'user','content':'question'}, {'role':'assistant','content':'old answer'},
                  {'role':'user','content':'new query'}])
    complete = protocol(request={**payload,'messages':payload['messages']+[assistant]})
    short = protocol(request={**payload,'messages':[assistant]})
    assert message_identity(complete['messages'][-1]) == message_identity(short['messages'][-1])


def test_tool_body_does_not_reparse_unchanged_reasoning_prefix(protocol):
    tools=[dict(type='function',function=dict(name='write',parameters=dict(type='object',
        properties={'content':dict(type='string')})))]
    raw='reason</think>\n<tool_call><function=write><parameter=content>'+('x中文\n'*100)+'\n</parameter></function></tool_call><|im_end|>'
    counts=[0,0]; outputs=[]
    for which in range(2):
        def counted(**kwargs):
            counts[which]+=1
            return protocol(**kwargs)
        wire=QwenStream(counted,tools,cache_parse=bool(which));deltas=[]
        for c in raw:deltas.append(wire.feed(c))
        output,tail=wire.finish(parsed_response(protocol,raw,tools));deltas.append(tail)
        got=assemble(deltas)
        # Parser deliberately creates fresh tool IDs; compare the function.
        outputs.append((got['content'],got['reasoning_content'],
                        [v['function'] for v in got['tool_calls']]))
    assert outputs[0] == outputs[1]
    assert counts[1] < counts[0] / 4
