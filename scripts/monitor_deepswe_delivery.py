"""Deliver monitor events once, including when the desktop drains the queue."""
import asyncio
import json


def event_time(message):
    try:
        return float(json.loads(message.rsplit('\n快照：\n', 1)[1])['at'])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


async def delivered(rpc, thread, key, message):
    """Queue IDs cease to deduplicate after consumption; check persisted clientId."""
    cursor = None
    seen = set()
    cutoff = event_time(message)
    while True:
        params = dict(threadId=thread, limit=8, itemsView='summary', sortDirection='desc')
        if cursor is not None:
            params['cursor'] = cursor
        page = await rpc('thread/turns/list', params)
        for turn in page['data']:
            for item in turn['items']:
                if item.get('type') == 'userMessage' and item.get('clientId') == key:
                    return dict(delivered_turn_id=turn['id'], delivered_item_id=item['id'],
                                started=False, delivery='persisted_user_message')
        # Check every item first: a message can arrive inside an older active turn.
        if cutoff is not None and any(
                turn.get('completedAt') is not None and turn['completedAt'] < cutoff - 2
                for turn in page['data']):
            return None
        cursor = page.get('nextCursor')
        if cursor is None:
            return None
        if cursor in seen:
            raise RuntimeError('Repeated thread history cursor; retain pending event')
        seen.add(cursor)


async def queue_items(rpc, thread):
    cursor = None
    seen = set()
    items = []
    while True:
        params = dict(threadId=thread, limit=100)
        if cursor is not None:
            params['cursor'] = cursor
        page = await rpc('thread/queue/list', params)
        items.extend(page['data'])
        cursor = page.get('nextCursor')
        if cursor is None:
            return items
        if cursor in seen:
            raise RuntimeError('Repeated queue cursor; retain pending event')
        seen.add(cursor)


async def deliver(rpc, thread, message, key, message_prefix):
    receipt = await delivered(rpc, thread, key, message)
    if receipt:
        return receipt
    queued = await queue_items(rpc, thread)
    existing = next((item for item in queued if item['clientUserMessageId'] == key), None)
    if existing is None:
        if any(any(part.get('text', '').startswith(message_prefix)
                   for part in item['input']) for item in queued):
            # Keep this distinct event pending instead of dropping it or piling
            # another monitor turn behind the existing one.
            raise RuntimeError('Earlier monitor event is queued; retain this pending event')
        result = await rpc('thread/queue/add', dict(threadId=thread,
                           clientUserMessageId=key, input=[dict(type='text', text=message)]))
        existing = result['queuedSubmission']
    state = (await rpc('thread/read', dict(threadId=thread, includeTurns=False)))['thread']['status']['type']
    started = False
    if state == 'idle':
        try:
            # Start the oldest queued input, including any user input ahead of us.
            await rpc('thread/queue/start', dict(threadId=thread))
            started = True
        except RuntimeError:
            receipt = await delivered(rpc, thread, key, message)
            if receipt:
                return receipt
            fresh = (await rpc('thread/read', dict(threadId=thread, includeTurns=False)))['thread']['status']['type']
            remaining = await queue_items(rpc, thread)
            if fresh != 'active' or not any(item['clientUserMessageId'] == key for item in remaining):
                raise
            state = fresh
    elif state != 'active':
        raise RuntimeError('Chat is not loaded/available: ' + state)
    return dict(queued_submission_id=existing['id'], started=started, thread_status=state,
                delivery='queued_or_started')


async def wake(thread, socket, message, submission_key, *, message_prefix):
    import websockets
    async with websockets.unix_connect(str(socket), open_timeout=10, close_timeout=3,
                                       max_size=16 * 1024**2) as ws:
        sequence = 0

        async def rpc(method, params):
            nonlocal sequence
            sequence += 1
            request_id = sequence
            await ws.send(json.dumps(dict(id=request_id, method=method, params=params)))
            async with asyncio.timeout(20):
                while True:
                    response = json.loads(await ws.recv())
                    if response.get('id') == request_id:
                        if 'error' in response:
                            raise RuntimeError(method + ': ' + json.dumps(response['error']))
                        return response['result']

        await rpc('initialize', dict(clientInfo=dict(name='kvmem_experiment_monitor', version='2'),
                                    capabilities=dict(experimentalApi=True)))
        await ws.send(json.dumps(dict(method='initialized')))
        return await deliver(rpc, thread, message, submission_key, message_prefix)
