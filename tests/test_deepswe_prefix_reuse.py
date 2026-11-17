"""Tool transport equivalence must preserve tokens without masking real edits."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location('deepswe_prefix_bridge',
    Path(__file__).resolve().parents[1] / 'benchmarks/deepswe/claude_bridge.py')
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


def message(arguments, name='Edit', call_id='call-1'):
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'reason',
            'tool_calls': [{'id': call_id, 'type': 'function',
                           'function': {'name': name, 'arguments': arguments}}]}


@pytest.mark.parametrize('name,left,right', [
    ('Edit', '{"old_string":"x","new_string":"y"}',
             '{"replace_all":false,"new_string":"y","old_string":"x"}'),
    ('Bash', '{"command":"true", "timeout":120000.0}',
             '{"timeout":1.2e5,"command":"true"}'),
    ('TaskUpdate', '{"taskId":"1", "status":"completed"}',
                   '{"status":"completed","taskId":"1"}'),
    ('Bash', '{"x":0.0}', '{"x":-0}'),
])
def test_equivalent_tool_transports_have_identical_identity_without_mutation(name, left, right):
    original = message(left, name)
    saved = copy.deepcopy(original)
    assert bridge.message_identity(original) == bridge.message_identity(message(right, name))
    assert original == saved


@pytest.mark.parametrize('name,left,right', [
    ('Edit', '{"replace_all":false}', '{"replace_all":true}'),
    ('Edit', '{}', '{"replace_all":0}'),
    ('Other', '{}', '{"replace_all":false}'),
    ('Edit', '{"old_string":"\\\\u0000"}', '{"old_string":"\\u0000"}'),
    ('Bash', '{"command":" x"}', '{"command":"x"}'),
    ('Bash', '{"x":9007199254740993}', '{"x":9007199254740992.0}'),
    ('Bash', '{"x":1,"x":2}', '{"x":2}'),
    ('Bash', '{"x":NaN}', '{ "x":NaN}'),
    ('Bash', '{"x":[1,2]}', '{"x":[2,1]}'),
])
def test_actual_edits_and_ambiguous_json_remain_distinct(name, left, right):
    assert bridge.message_identity(message(left, name)) != bridge.message_identity(message(right, name))


def test_call_identity_reasoning_and_content_remain_part_of_match():
    first = message('{}')
    for change in (message('{}', call_id='other'), first | {'reasoning_content': 'changed'},
                   first | {'content': 'changed'}, message('{}', name='Bash')):
        assert bridge.message_identity(first) != bridge.message_identity(change)


def test_planner_appends_only_new_tool_result_and_retains_exact_output_ids():
    # Character tokenizer makes any accidental old-prefix re-encoding visible.
    class Tokenizer:
        def convert_tokens_to_ids(self, token): return 999
        def encode(self, text, **kw): return list(text.encode())
    old = {'role': 'user', 'content': 'question'}
    assistant = message('{"old_string":"x"}')
    transported = message('{"old_string":"x","replace_all":false}')
    rendered = {'prompt': 'PREFIX-NEW', 'control_end': 0, 'tools': [],
                'messages': [old, transported, {'role':'tool','content':'NEW'}],
                'spans': [{'index':2, 'role':'tool', 'begin':7, 'end':10,
                           'content_begin':7, 'content_end':10}], 'pins': [], 'query_index':-1}
    planner = bridge.Planner(lambda **kw: rendered, Tokenizer())
    state = bridge.Trajectory(messages=[bridge.message_identity(old)],
        control=bridge.canonical({'prefix':'', 'tools':[]}), tokens=[100, 101],
        output=[901, 902, 999], assistant=bridge.message_identity(assistant))
    prepared = planner.prepare({}, state)
    assert prepared[5] is True
    assert prepared[0] == [100, 101, 901, 902, 999] + list(b'\nNEW')
    rendered['messages'][1] = message('{"old_string":"different"}')
    changed = planner.prepare({}, state)
    assert changed[5] is False and changed[0] == list(b'PREFIX-NEW')


@pytest.fixture(params=('none','NFC'))
def boundary_tokenizer(request):
    from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    backend=Tokenizer(models.BPE(unk_token='<unk>'))
    if request.param=='NFC':backend.normalizer=normalizers.NFC()
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    special=[AddedToken(s,special=True,normalized=False) for s in ('<unk>','<|im_start|>','<|im_end|>')]
    backend.train_from_iterator(['hello hell helping testing tested test 测试汉字🙂 café aaaaa ababab '*10],
        trainers.BpeTrainer(vocab_size=400,special_tokens=special,initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize('text',[
    'hello helping tested café 🙂',
    '<|im_start|>user\n测试汉字🙂 café cafe\u0301 helping a\u0308<|im_end|>\n<|im_start|>assistant\n',
    'hello<|im_start|><|im_end|><|im_start|>user\n<|im_end|>\n',
    'hello<|im_start|>user\n<|im_en',
    '',
])
def test_every_character_boundary_matches_exact_prefix_tokenization(boundary_tokenizer,text):
    encoded=bridge.PrefixTokens(boundary_tokenizer,text)
    assert encoded.ids==boundary_tokenizer.encode(text,add_special_tokens=False)
    for end in range(len(text)+1):
        prefix=text[:end]
        assert encoded.count(len(prefix.encode()))==len(boundary_tokenizer.encode(prefix,add_special_tokens=False))


def test_delimiters_with_stripping_or_normalization_are_not_safe_anchors(boundary_tokenizer):
    from tokenizers import AddedToken
    for marker in ('<|im_start|>','<|im_end|>'):
        boundary_tokenizer.add_tokens([AddedToken(marker,special=True,normalized=True,lstrip=True)])
    text='hello <|im_start|>user\nhello <|im_end|>\n'
    encoded=bridge.PrefixTokens(boundary_tokenizer,text)
    assert encoded.positions==[0]
    for end in range(len(text)+1):
        assert encoded.count(end)==len(boundary_tokenizer.encode(text[:end],add_special_tokens=False))


def test_message_coordinate_work_is_linear_and_duplicate_points_are_cached(boundary_tokenizer):
    class CountingTokenizer:
        def __init__(self, tokenizer):self.tokenizer=tokenizer;self.characters=0
        def __getattr__(self,name):return getattr(self.tokenizer,name)
        def __call__(self,text,**kw):self.characters+=len(text);return self.tokenizer(text,**kw)
        def encode(self,text,**kw):self.characters+=len(text);return self.tokenizer.encode(text,**kw)
    tokenizer=CountingTokenizer(boundary_tokenizer)
    parts=['<|im_start|>user\n'+('hello test café 🙂 '*80)+'<|im_end|>\n' for _ in range(80)]
    text=''.join(parts);encoded=bridge.PrefixTokens(tokenizer,text)
    begin=0
    for part in parts:
        for end in (0,len('<|im_start|>user\n'.encode()),len(part.encode())-len('<|im_end|>\n'),len(part.encode())):
            point=begin+end
            value=encoded.count(point);cost=tokenizer.characters
            assert encoded.count(point)==value and tokenizer.characters==cost
        begin+=len(part.encode())
    assert tokenizer.characters < 3*len(text)


def renderer(*, request):
    """An authoritative structured renderer; no substring search for offsets."""
    text, spans = '', []
    for index, message in enumerate(request['messages']):
        begin = len(text.encode())
        role, content = message['role'], message['content']
        text += '<|im_start|>' + role + '\n'
        content_begin = len(text.encode())
        text += content + (request.get('rewrite', '') if index == 0 else '')
        content_end = len(text.encode())
        text += '<|im_end|>\n'
        spans.append(dict(index=index, begin=begin, end=len(text.encode()), role=role,
                          content_begin=content_begin, content_end=content_end))
    text += request.get('generation_header', '<|im_start|>assistant\n<think>\n')
    return dict(prompt=text, spans=spans, messages=request['messages'], control_end=0,
                tools=request.get('tools', []), pins=[], query_index=-1)


@pytest.mark.parametrize('terminator', ['<|im_end|>', '<|endoftext|>'])
def test_structured_producer_keeps_actual_output_and_suffix_coordinates(boundary_tokenizer, terminator):
    if terminator == '<|endoftext|>':
        boundary_tokenizer.add_special_tokens({'additional_special_tokens': [terminator]})
    planner = bridge.Planner(renderer, boundary_tokenizer)
    state = bridge.Trajectory()
    payload = dict(messages=[dict(role='user', content='测试🙂重复')], tools=['declaration'])
    prepared = planner.prepare(payload, state)
    assert prepared[0] == planner.encode(prepared[2]['prompt'])
    assert prepared[1]['kvmem_locality_spans']
    assert prepared[1]['kvmem_locality_provenance']['status'] == 'exact'
    # Deliberately choose accepted IDs which cannot be recovered by rendering
    # the returned assistant text. Only the actual accepted list is authoritative.
    output = [17, 18, 19, boundary_tokenizer.convert_tokens_to_ids(terminator)]
    response = dict(content='displayed answer')
    planner.commit(state, prepared, payload, response, output)
    accepted_end = len(prepared[0]) + len(output)
    assert state.locality_spans[-1] == (prepared.assistant_begin, accepted_end, 'assistant')
    following = dict(messages=payload['messages'] + [dict(role='assistant', content=response['content']),
        dict(role='tool', content='测试🙂重复'), dict(role='tool', content='')], tools=['declaration'])
    continued = planner.prepare(following, state)
    assert continued[5] is True
    assert continued[0][:accepted_end] == prepared[0] + output
    spans = continued[1]['kvmem_locality_spans']
    assert [(s['begin'],s['end'],s['role']) for s in spans[:2]] == list(state.locality_spans)
    assert [s['role'] for s in spans[-2:]] == ['tool','tool']
    assert spans[-2]['begin'] >= accepted_end
    raw = continued[2]['prompt'].encode()
    first = next(s for s in continued[2]['spans'] if s['index'] == 2)
    suffix = '\n' + raw[first['begin']:].decode()
    offsets = boundary_tokenizer(suffix, add_special_tokens=False, return_offsets_mapping=True)
    first_span = spans[-2]
    relative_begin = first_span['begin'] - accepted_end
    assert suffix[offsets['offset_mapping'][relative_begin][0]:].startswith('<|im_start|>tool\n')
    assert continued[0][accepted_end:] == offsets['input_ids']
    assert continued[1]['kvmem_mandatory_spans'] == prepared[1]['kvmem_mandatory_spans'] == []
    assert 'kvmem_message_spans' not in continued[1]


def test_renderer_prefix_change_and_control_options_disable_stale_reuse(boundary_tokenizer):
    planner = bridge.Planner(renderer, boundary_tokenizer)
    state = bridge.Trajectory()
    payload = dict(messages=[dict(role='user', content='unchanged')])
    first = planner.prepare(payload, state)
    planner.commit(state, first, payload, dict(content='answer'), [7, planner.end])
    following = dict(messages=payload['messages'] + [dict(role='assistant',content='answer'),
                                                    dict(role='tool',content='new')])
    assert planner.prepare(following, state)[5]
    for changes in (dict(rewrite=' TEMPLATE CHANGED'), dict(tools=['new declaration'])):
        edited = planner.prepare({**following, **changes}, state)
        assert not edited[5]
        assert edited[0] == planner.encode(edited[2]['prompt'])
        assert edited[1]['kvmem_locality_provenance']['history'] == 'canonical_full_render'


def test_incomplete_header_or_nonterminal_output_does_not_invent_assistant_end(boundary_tokenizer):
    planner = bridge.Planner(renderer, boundary_tokenizer)
    for header, output in (('<|im_sta', [planner.end]),
                           ('<|im_start|>assistant\n<think>\n', [17,18])):
        state = bridge.Trajectory()
        payload = dict(messages=[dict(role='user',content='x')], generation_header=header)
        first = planner.prepare(payload, state)
        planner.commit(state, first, payload, dict(content='answer'), output)
        assert all(role != 'assistant' for _,_,role in state.locality_spans)


@pytest.mark.parametrize('optimized', [False, True])
def test_length_finish_preserves_accepted_ids_and_closes_only_in_next_input(boundary_tokenizer, optimized):
    planner = bridge.Planner(renderer, boundary_tokenizer, optimized_handoff=optimized)
    state = bridge.Trajectory()
    payload = dict(messages=[dict(role='user',content='question')])
    first = planner.prepare(payload,state)
    output = [17,18,19]  # Deliberately unlike re-encoded display text, with no EOS.
    response = dict(content='truncated answer',stop_reason='max_tokens')
    planner.commit(state,first,payload,response,output)
    before = copy.deepcopy(state)
    following = dict(messages=payload['messages']+[
        dict(role='assistant',content=response['content']),dict(role='user',content='continue')])
    new = planner.prepare(following,state)
    accepted = first[0]+output
    assert new[5] and new[0][:len(accepted)]==accepted
    assert new.input_closure_tokens==1 and new[0][len(accepted)]==planner.end
    assert state.output==output and state==before
    spans=new[1]['kvmem_locality_spans']
    assistant=[s for s in spans if s['role']=='assistant']
    assert assistant==[dict(begin=first.assistant_begin,end=len(accepted)+1,role='assistant')]
    assert new[1]['kvmem_locality_provenance']['unknown_spans']==0
    # Edits still invalidate history; closure is never permission to ignore them.
    following['messages'][1]['content']='modified old answer'
    assert not planner.prepare(following,state)[5]
    assert not planner.prepare(dict(following,rewrite='changed history'),state)[5]


def test_unproven_partial_stream_never_gets_a_template_closure(boundary_tokenizer):
    planner=bridge.Planner(renderer,boundary_tokenizer);state=bridge.Trajectory()
    payload=dict(messages=[dict(role='user',content='question')]);first=planner.prepare(payload,state)
    planner.commit(state,first,payload,dict(content='partial'),[17,18])
    following=dict(messages=payload['messages']+[dict(role='assistant',content='partial'),dict(role='user',content='next')])
    assert not planner.prepare(following,state)[5]


def test_canonical_fallback_agrees_with_accepted_assistant_eos(boundary_tokenizer):
    from vllm_kvmem.serving.block_graph import _merge_spans
    from vllm_kvmem.serving.prompt_spans import normalize_spans
    planner = bridge.Planner(renderer, boundary_tokenizer)
    state = bridge.Trajectory()
    payload = dict(messages=[dict(role='user', content='hello')],
                   generation_header='<|im_start|>assistant\n')
    first = planner.prepare(payload, state)
    output = planner.encode('test<|im_end|>')
    planner.commit(state, first, payload, dict(content='test'), output)
    following = {**payload, 'messages': payload['messages'] + [
        dict(role='assistant', content='test'), dict(role='user', content='help')]}
    second = planner.prepare(following, state)
    assert second[5]
    old = normalize_spans(second[1]['kvmem_locality_spans'])
    planner.commit(state, second, following, dict(content='hello'),
                   planner.encode('hello<|im_end|>'))
    # The newest returned text is edited by a client, forcing a canonical
    # render. Earlier exact message boundaries and token prefix stay valid.
    changed = {**following, 'messages': following['messages'] + [
        dict(role='assistant', content='helping'), dict(role='user', content='test')]}
    third = planner.prepare(changed, state)
    assert not third[5]
    assert third[0][:len(second[0])] == second[0]
    incoming = normalize_spans(third[1]['kvmem_locality_spans'])
    assert incoming[:len(old)] == old
    _merge_spans(old, (True,) * len(old), incoming, len(second[0]) - 1)
    # Keep the inter-message newline unclassified in both render paths;
    # accepted history ends at EOS and does not contain that suffix token.
    assistant = incoming[1]
    assert third[0][assistant[1] - 1] == planner.end
    assert third[0][assistant[1]:incoming[2][0]] == planner.encode('\n')


def _two_turns(planner, header='<|im_start|>assistant\n'):
    state = bridge.Trajectory()
    payload = dict(messages=[dict(role='user', content='hello helping')], generation_header=header)
    first = planner.prepare(payload, state)
    # Accepted IDs deliberately unlike any re-encoding of the displayed text.
    planner.commit(state, first, payload, dict(content='displayed answer'), [17, 18, 19, planner.end])
    second_payload = dict(payload, messages=payload['messages'] + [
        dict(role='assistant', content='displayed answer'), dict(role='tool', content='tested')])
    second = planner.prepare(second_payload, state)
    assert second[5]
    planner.commit(state, second, second_payload, dict(content='café answer'), [21, 22, planner.end])
    return state, second_payload, second


def test_changed_last_assistant_keeps_the_previous_prompt_ids(boundary_tokenizer):
    planner = bridge.Planner(renderer, boundary_tokenizer)
    state, payload, second = _two_turns(planner)
    before = copy.deepcopy(state)
    # The client returns the newest assistant message changed (as Claude Code
    # does when it decodes escapes in a tool input).
    changed = dict(payload, messages=payload['messages'] + [
        dict(role='assistant', content='café answer\u0000'), dict(role='user', content='testing')])
    third = planner.prepare(changed, state)
    assert state == before
    assert not third[5] and third.prompt_prefix_reused
    assert third[0][:len(second[0])] == second[0]
    # A canonical render would re-encode the first accepted answer and lose it.
    assert planner.encode(third[2]['prompt'])[:len(second[0])] != second[0]
    raw = third[2]['prompt'].encode()
    boundary = len(state.rendered_prefix) + len(state.generation_header)
    assert raw[:boundary] == state.rendered_prefix + state.generation_header
    assert third[0][len(second[0]):] == planner.encode(raw[boundary:].decode())
    provenance = third[1]['kvmem_locality_provenance']
    assert provenance['history'] == 'actual_prompt_ids' and provenance['encoding_scope'] == 'append_suffix'
    spans = third[1]['kvmem_locality_spans']
    old = [s for s in spans if s['end'] <= len(second[0])]
    assert old == [s for s in second[1]['kvmem_locality_spans']]
    new = [s for s in spans if s['end'] > len(second[0])]
    assert [s['role'] for s in new] == ['assistant', 'user'] and new[0]['begin'] >= len(second[0])
    from vllm_kvmem.serving.request_plan import RequestPlan
    RequestPlan.parse(third[1], len(third[0]), 32, 16384)


@pytest.mark.parametrize('edit', ['earlier_message', 'control', 'header'])
def test_prompt_prefix_reuse_requires_an_unchanged_prefix(boundary_tokenizer, edit):
    planner = bridge.Planner(renderer, boundary_tokenizer)
    state, payload, second = _two_turns(planner)
    messages = payload['messages'] + [dict(role='assistant', content='edited'), dict(role='user', content='x')]
    changed = dict(payload, messages=messages)
    if edit == 'earlier_message':
        changed['messages'] = [dict(role='user', content='hello helping!')] + messages[1:]
    elif edit == 'control':
        changed['tools'] = ['new declaration']
    else:
        state.generation_header = b'<|im_start|>assistant\n<think>\n'
    third = planner.prepare(changed, state)
    assert not third[5] and not third.prompt_prefix_reused
    assert third[0] == planner.encode(third[2]['prompt'])
    assert third[1]['kvmem_locality_provenance']['history'] == 'canonical_full_render'
