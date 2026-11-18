from scripts.dsh_bridge import observation_tool_spans


def plan(*roles):
    spans, begin = [], 0
    for role in roles:
        spans.append(dict(begin=begin, end=begin + 10, role=role))
        begin += 10
    return {'kvmem_locality_spans': spans, 'kvmem_query_span': [2, 8], 'kvmem_mandatory_parts': 'none'}


def roles(value):
    return [span['role'] for span in value['kvmem_locality_spans']]


def test_later_user_messages_become_tool_spans_and_the_root_stays_user():
    value = observation_tool_spans(plan('system', 'user', 'assistant', 'user', 'assistant', 'user'))
    assert roles(value) == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool']
    assert value['kvmem_query_span'] == [2, 8] and value['kvmem_mandatory_parts'] == 'none'
    assert [(s['begin'], s['end']) for s in value['kvmem_locality_spans']] == [(i, i + 10) for i in range(0, 60, 10)]


def test_the_rule_is_stable_across_turns():
    first = roles(observation_tool_spans(plan('user', 'assistant', 'user')))
    second = roles(observation_tool_spans(plan('user', 'assistant', 'user', 'assistant', 'user')))
    assert second[:len(first)] == first
    # Applying it again to an already labelled plan changes nothing.
    again = observation_tool_spans(observation_tool_spans(plan('user', 'assistant', 'user')))
    assert roles(again) == first


def test_existing_tool_spans_and_plans_without_spans_are_unchanged():
    assert roles(observation_tool_spans(plan('user', 'assistant', 'tool'))) == ['user', 'assistant', 'tool']
    assert observation_tool_spans({}) == {}
