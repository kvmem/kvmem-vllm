import asyncio
import pytest
from scripts.monitor_deepswe_delivery import deliver, delivered


PREFIX = '【DeepSWE 64 次实验定时巡检】'
MESSAGE = PREFIX + '\n快照：\n{"at":100}'


def history(key='key', **extra):
    return dict(data=[dict(id='turn', items=[dict(type='userMessage', id='item',
                clientId=key)], **extra)])


def queued(key='key'):
    return dict(id='queue-id', clientUserMessageId=key,
                input=[dict(type='text', text=MESSAGE)])


def state(value):
    return dict(thread=dict(status=dict(type=value)))


class RPC:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    async def __call__(self, method, params):
        self.calls.append((method, params))
        expected, result = self.steps.pop(0)
        assert method == expected
        assert params.get('threadId') == 'thread'
        if isinstance(result, Exception):
            raise result
        return result


def run(steps):
    rpc = RPC(steps)
    result = asyncio.run(deliver(rpc, 'thread', MESSAGE, 'key', PREFIX))
    assert not rpc.steps
    return result, rpc.calls


def test_consumed_message_retry_never_adds_again():
    result, calls = run([('thread/turns/list', history())])
    assert result['delivery'] == 'persisted_user_message'
    assert len(calls) == 1


def test_queue_empty_after_desktop_consumes_and_finishes_is_delivered():
    result, calls = run([
        ('thread/turns/list', dict(data=[])),
        ('thread/queue/list', dict(data=[])),
        ('thread/queue/add', dict(queuedSubmission=queued())),
        ('thread/read', state('idle')),
        ('thread/queue/start', RuntimeError('queue is empty')),
        ('thread/turns/list', history(completedAt=101)),
    ])
    assert result['delivered_turn_id'] == 'turn'
    assert sum(method == 'thread/queue/add' for method, _ in calls) == 1


def test_active_thread_keeps_existing_exact_submission():
    result, calls = run([
        ('thread/turns/list', dict(data=[])),
        ('thread/queue/list', dict(data=[queued()])),
        ('thread/read', state('active')),
    ])
    assert result['queued_submission_id'] == 'queue-id'
    assert not any(method in ('thread/queue/add', 'thread/queue/start') for method, _ in calls)


def test_unproven_empty_queue_remains_failure():
    rpc = RPC([
        ('thread/turns/list', dict(data=[])),
        ('thread/queue/list', dict(data=[queued()])),
        ('thread/read', state('idle')),
        ('thread/queue/start', RuntimeError('queue is empty')),
        ('thread/turns/list', dict(data=[])),
        ('thread/read', state('idle')),
        ('thread/queue/list', dict(data=[])),
    ])
    with pytest.raises(RuntimeError, match='queue is empty'):
        asyncio.run(deliver(rpc, 'thread', MESSAGE, 'key', PREFIX))


def test_different_monitor_event_is_not_silently_acknowledged():
    rpc = RPC([('thread/turns/list', dict(data=[])),
               ('thread/queue/list', dict(data=[queued('older-key')]))])
    with pytest.raises(RuntimeError, match='retain this pending event'):
        asyncio.run(deliver(rpc, 'thread', MESSAGE, 'key', PREFIX))
    assert not any(method == 'thread/queue/add' for method, _ in rpc.calls)


def test_starts_existing_queue_order_without_reordering_user_message():
    user = dict(id='user', clientUserMessageId='user-key', input=[dict(type='text', text='hello')])
    result, calls = run([
        ('thread/turns/list', history('different-key', completedAt=99)),
        ('thread/queue/list', dict(data=[user])),
        ('thread/queue/add', dict(queuedSubmission=queued())),
        ('thread/read', state('idle')),
        ('thread/queue/start', {}),
    ])
    assert result['started']
    assert calls[-1][1] == dict(threadId='thread')


def test_pagination_uses_client_id_and_handles_older_active_turn():
    rpc = RPC([
        ('thread/turns/list', dict(data=[dict(id='recent', items=[], startedAt=200)], nextCursor='next')),
        ('thread/turns/list', history(startedAt=50, completedAt=None)),
    ])
    assert asyncio.run(delivered(rpc, 'thread', 'key', MESSAGE))['delivered_item_id'] == 'item'
    assert rpc.calls[-1][1]['cursor'] == 'next'


def test_history_lookup_failure_does_not_risk_duplicate_add():
    rpc = RPC([('thread/turns/list', TimeoutError('unavailable'))])
    with pytest.raises(TimeoutError):
        asyncio.run(deliver(rpc, 'thread', MESSAGE, 'key', PREFIX))
    assert len(rpc.calls) == 1


def test_start_race_with_other_active_turn_requires_queued_proof():
    result, _ = run([
        ('thread/turns/list', dict(data=[])),
        ('thread/queue/list', dict(data=[queued()])),
        ('thread/read', state('idle')),
        ('thread/queue/start', RuntimeError('another turn started')),
        ('thread/turns/list', dict(data=[])),
        ('thread/read', state('active')),
        ('thread/queue/list', dict(data=[queued()])),
    ])
    assert result['thread_status'] == 'active'


def test_only_finished_history_before_event_can_end_lookup():
    rpc = RPC([('thread/turns/list', history('older', completedAt=97) | dict(nextCursor='unused'))])
    assert asyncio.run(delivered(rpc, 'thread', 'key', MESSAGE)) is None
    assert len(rpc.calls) == 1
