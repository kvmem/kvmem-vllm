"""Native HTTP routing and disposable sessions, without a GPU or listening port."""
import asyncio
import math
from importlib import import_module
from types import SimpleNamespace as NS

import pytest

httpx = pytest.importorskip('httpx')
FastAPI = pytest.importorskip('fastapi').FastAPI
pytest.importorskip('vllm.entrypoints.generate.structured_decisions.protocol')
from vllm.entrypoints.generate.structured_decisions import api_router as api
from vllm.inputs import tokens_input
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm_kvmem.systemone import SystemOnePlugin


@pytest.fixture
def native_app(monkeypatch):
    monkeypatch.setenv('KVMEM_SERVING', '1')
    monkeypatch.delenv('KVMEM_REQUEST_PLAN_CONFIG', raising=False)
    strategy_path = 'vllm.entrypoints.generate.structured_decisions.strategies'
    try:
        strategies = import_module(strategy_path)
    except ModuleNotFoundError as error:
        if error.name != strategy_path:
            raise
        strategies = import_module('vllm.entrypoints.openai.decisions.strategies')

    class Tokenizer:
        labels = {label: i + 10 for i, label in enumerate(sorted(
            {label for qtype in strategies.QUESTION_TYPES.values() for label in qtype.label_set}))}
        def get_added_vocab(self):
            return {'<reply>': 999, '<|im_start|>': 1, '<|im_end|>': 3, '</think>': 4}
        def apply_chat_template(self, *args, **kwargs): return [999]
        def decode(self, ids): return ''
        def encode(self, text, **kwargs):
            if text == '<|im_start|>user\n': return [1, 2]
            if text == '</think>\n<|im_end|>\n': return [4, 3]
            return [self.labels[text]]

    class Renderer:
        renderer = NS(get_tokenizer=lambda: Tokenizer())
        async def preprocess_chat(self, request, messages, **kwargs):
            return messages, (tokens_input([1, 2, 999], cache_salt=request.cache_salt),)

    class Engine:
        errored = False
        model_config = NS(supports_multimodal_inputs=False, is_hybrid=True, is_encoder_decoder=False)
        vllm_config = NS(scheduler_config=NS(max_num_seqs=1),
                         cache_config=NS(enable_prefix_caching=False))
        def __init__(self):
            self.calls, self.sessions = [], set()
        def check_admission(self, *args): pass
        async def abort(self, request_id):
            pytest.fail(f'Completed HTTP request should not abort: {request_id}')
        async def generate(self, prompt, params, request_id, **kwargs):
            release = bool((params.extra_args or {}).get('kv_transfer_params', {}).get(
                'kvmem_release_session'))
            session = kwargs['session_id']
            self.calls.append(NS(session=session, release=release, prompt=prompt))
            if release:
                self.sessions.remove(session)
            else:
                self.sessions.add(session)
            labels = params.logprob_token_ids or [0]
            yield RequestOutput(request_id=request_id, prompt=None,
                prompt_token_ids=prompt['prompt_token_ids'], prompt_logprobs=None,
                outputs=[CompletionOutput(index=0, text='', token_ids=[labels[0]],
                    cumulative_logprob=None, finish_reason='length',
                    logprobs=[{label: Logprob(math.log(1 / (len(labels) + 1))) for label in labels}])],
                finished=True)

    async def build():
        app, engine = FastAPI(), Engine()
        models = NS(is_base_model=lambda name: True, lora_requests={}, model_name=lambda lora: 'test')
        strategy = strategies.NextTokenStrategy(strategies.ReadContext(engine, Renderer(), None, 'auto', {}))
        app.state.serving_structured_decisions = api.ServingStructuredDecisions(models, strategy)
        api.register_structured_decisions_api_router(app)
        await SystemOnePlugin().init_state(engine, app.state, NS())
        return app, engine
    return build


def test_native_http_success_releases_each_temporary_session(native_app):
    async def run():
        app, engine = await native_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url='http://test') as client:
            payload = {'state': 'A damaged package.', 'cache_salt': 'tenant-a', 'questions': {
                'damaged': {'type': 'noul', 'instructions': 'Is it damaged?'},
                'action': {'type': 'choice', 'criteria': {'refund': 'Refund', 'keep': 'Keep'}},
            }}
            for _ in range(2):
                response = await client.post('/v1/systemone', json=payload,
                                             headers={'X-Request-Id': 'reused-client-id'})
                assert response.status_code == 200, response.text
                body = response.json()
                assert body['object'] == 'structured_decision'
                assert set(body['answers']) == {'damaged', 'action'}
                assert body['answers']['damaged']['noul'] == pytest.approx(.5)
                assert body['usage']['output_tokens'] == 2
                assert not engine.sessions
            reads = [call for call in engine.calls if not call.release]
            releases = [call for call in engine.calls if call.release]
            assert len({call.session for call in reads}) == 4
            assert {call.session for call in releases} == {call.session for call in reads}
            assert len(releases) == 4
            assert all(call.prompt['cache_salt'] == 'tenant-a' for call in engine.calls)
            assert all(call.prompt['prompt_token_ids'] == [1] for call in releases)
            assert not app.state.serving_structured_decisions.strategy.context.engine_client.cleanups
    asyncio.run(run())


def test_active_http_uses_one_worker_snapshot_and_native_answers(native_app):
    async def run():
        app, engine = await native_app()
        calls, reads = [], []
        snapshot = {'mode': 'active_branch', 'history_end': 500, 'active_tokens': 200}
        async def rpc(method, kwargs):
            assert method == 'kvmem_systemone_active'
            calls.append(kwargs)
            if kwargs['action'] == 'submit':
                reads.extend(kwargs['reads'])
                return [{'status': 'pending'}]
            if kwargs['action'] == 'cancel': return [{'status': 'cancelled'}]
            return [{'status': 'done', 'snapshot': snapshot, 'reads': [
                {'argmax': read['labels'][0], 'logprobs': [-2.] * len(read['labels'])}
                for read in reads]}]
        engine.collective_rpc = rpc
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/v1/systemone', json={'state': '', 'cache_salt': 'tenant',
                'kvmem_session': {'session_id': 'parent'},
                'questions': {'a': {'type': 'noul'}, 'b': {'type': 'noul'}}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['kvmem'] == snapshot
        assert body['answers']['a']['noul'] == .5
        assert body['answers']['b']['noul'] == .5
        assert body['usage'] == {'input_tokens': 10, 'output_tokens': 2}
        assert [c['action'] for c in calls] == ['submit', 'poll', 'cancel']
        assert len({c['operation'] for c in calls}) == 1
        assert calls[0]['reference']['namespace'] == ['tenant', 'parent']
        assert all(r['tokens'] == [4, 3, 1, 2, 999] for r in reads)
        assert not engine.calls and not engine.sessions
    asyncio.run(run())


@pytest.mark.parametrize('reference,state,status', [
    ({'session_id': 'parent'}, 'ambiguous additional state', 400),
    ({'session_id': 'parent', 'history_end': True}, '', 400),
    ({'session_id': 'parent'}, '', 409),
])
def test_active_http_rejects_invalid_or_unavailable_parent(native_app, reference, state, status):
    async def run():
        app, engine = await native_app()
        async def rpc(method, kwargs):
            return [{'status': 'cancelled'}] if kwargs['action'] == 'cancel' else [{'error': 'No active owner'}]
        engine.collective_rpc = rpc
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/v1/systemone', json={'state': state,
                'kvmem_session': reference, 'questions': {'a': {'type': 'noul'}}})
        assert response.status_code == status, response.text
        assert not engine.calls and not engine.sessions
    asyncio.run(run())


def test_active_http_cancel_removes_queued_operation(native_app, monkeypatch):
    async def run():
        app, engine = await native_app()
        submitted, cancelled, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def disconnect(request):
            await disconnected.wait()
        monkeypatch.setattr('vllm.entrypoints.serve.utils.api_utils.listen_for_disconnect', disconnect)
        async def rpc(method, kwargs):
            if kwargs['action'] == 'submit': submitted.set()
            if kwargs['action'] == 'cancel': cancelled.set()
            return [{'status': 'pending'}]
        engine.collective_rpc = rpc
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            task = asyncio.create_task(client.post('/v1/systemone', json={'state': '',
                'kvmem_session': {'session_id': 'parent'}, 'questions': {'a': {'type': 'noul'}}}))
            await asyncio.wait_for(submitted.wait(), 2)
            disconnected.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.wait_for(cancelled.wait(), 2)
        assert not engine.calls and not engine.sessions
    asyncio.run(run())


@pytest.mark.parametrize('payload,status', [
    ({'questions': {'q': {'type': 'noul'}}}, 422),
    ({'state': 'text', 'questions': {'q': {'type': 'unknown-question-type'}}}, 400),
])
def test_native_http_rejects_invalid_request_before_opening_session(native_app, payload, status):
    async def run():
        app, engine = await native_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url='http://test') as client:
            response = await client.post('/v1/systemone', json=payload)
            assert response.status_code == status, response.text
            assert response.json()
            assert not engine.calls and not engine.sessions
    asyncio.run(run())
