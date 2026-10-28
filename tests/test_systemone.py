import asyncio
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm.sampling_params import SamplingParams
from vllm_kvmem.systemone import DecisionEngine, SystemOnePlugin


class Engine:
    def __init__(self):
        self.calls = []
        self.aborted = []
        self.read_gate = None
        self.release_gate = None
        self.failure = None

    async def generate(self, prompt, params, request_id, **kwargs):
        release = (params.extra_args or {}).get('kv_transfer_params', {}).get('kvmem_release_session')
        self.calls.append((prompt, params, request_id, kwargs, release))
        gate = self.release_gate if release else self.read_gate
        if gate:
            await gate.wait()
        if self.failure and not release:
            raise self.failure
        yield NS(finished=True, probabilities=[.25, .75])

    async def abort(self, request_id):
        self.aborted.append(request_id)


async def consume(engine, prompt=None, params=None):
    return [out async for out in engine.generate(
        prompt or {'type': 'token', 'prompt_token_ids': [12, 34], 'cache_salt': 'tenant-a'},
        params or SamplingParams(max_tokens=1, logprob_token_ids=[4, 8]), 'client-reused-id')]


def test_native_probabilities_and_parameters_survive_disposable_session(monkeypatch):
    monkeypatch.delenv('KVMEM_REQUEST_PLAN_CONFIG', raising=False)
    async def run():
        native = Engine()
        proxy = DecisionEngine(native, 2)
        params = SamplingParams(max_tokens=1, logprob_token_ids=[4, 8], extra_args={'custom': 7})
        out = await consume(proxy, params=params)
        assert out[0].probabilities == [.25, .75]
        read, close = native.calls
        assert read[1].logprob_token_ids == [4, 8]
        assert read[1].extra_args['custom'] == 7
        assert params.extra_args == {'custom': 7}
        assert read[3]['session_id'] == close[3]['session_id']
        assert close[0]['cache_salt'] == 'tenant-a'
        assert close[0]['prompt_token_ids'] == [12] and close[4] is True
        assert not native.aborted
        await consume(proxy)
        assert native.calls[2][3]['session_id'] != read[3]['session_id']
        assert native.calls[2][2] != read[2]
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_failure_and_disconnect_close_session_before_reusing_concurrency_slot(cancel):
    async def run():
        native = Engine()
        native.release_gate = asyncio.Event()
        if cancel:
            native.read_gate = asyncio.Event()
        else:
            native.failure = ValueError('failed native read')
        proxy = DecisionEngine(native, 1)
        first = asyncio.create_task(consume(proxy))
        await asyncio.sleep(0)
        if cancel:
            first.cancel()
        # Allow the failed/cancelled read to enter its shielded release.
        for _ in range(10):
            if len(native.calls) == 2:
                break
            await asyncio.sleep(0)
        assert len(native.calls) == 2 and native.calls[-1][4]
        assert native.aborted == [native.calls[0][2]]
        native.failure, native.read_gate = None, None
        second = asyncio.create_task(consume(proxy))
        await asyncio.sleep(0)
        assert len(native.calls) == 2  # release still owns the only slot
        native.release_gate.set()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await first
        await second
        assert len(native.calls) == 4 and not proxy.cleanups
    asyncio.run(run())


def test_plugin_uses_native_handler_and_is_idempotent(monkeypatch):
    import vllm_kvmem.engine_api as api
    monkeypatch.setenv('KVMEM_SERVING', '1')
    monkeypatch.setattr(api, 'require_contract', lambda name: None)
    @dataclass
    class Context:
        engine_client: object
    async def run():
        native = Engine()
        native.vllm_config = NS(scheduler_config=NS(max_num_seqs=2))
        context = Context(native)
        strategy = NS(context=context)
        handler = NS(strategy=strategy)
        state = NS(serving_structured_decisions=handler)
        plugin = SystemOnePlugin()
        app = NS(routes=['upstream /v1/systemone'])
        plugin.attach_router(app)
        await plugin.init_state(native, state, NS())
        proxy = handler.strategy.context.engine_client
        await plugin.init_state(native, state, NS())
        assert handler.strategy.context.engine_client is proxy
        assert isinstance(proxy, DecisionEngine) and context.engine_client is native
        assert state.serving_structured_decisions is handler
        assert app.routes == ['upstream /v1/systemone']
    asyncio.run(run())


def test_real_upstream_decision_handler_runs_through_plugin(monkeypatch):
    """Exercise native schema, strategy, label reads and answer computation."""
    protocol = pytest.importorskip('vllm.entrypoints.generate.structured_decisions.protocol')
    from importlib import import_module
    from vllm.entrypoints.generate.structured_decisions.serving import ServingStructuredDecisions
    # Upstream moved the shared strategy out of the SystemOne-only package.
    strategy_path = 'vllm.entrypoints.generate.structured_decisions.strategies'
    try:
        strategies = import_module(strategy_path)
    except ModuleNotFoundError as error:
        if error.name != strategy_path:
            raise
        strategies = import_module('vllm.entrypoints.openai.decisions.strategies')
    NextTokenStrategy, ReadContext = strategies.NextTokenStrategy, strategies.ReadContext
    QUESTION_TYPES = strategies.QUESTION_TYPES
    from vllm.inputs import tokens_input
    from vllm.logprobs import Logprob
    from vllm.outputs import CompletionOutput, RequestOutput
    import math
    monkeypatch.setenv('KVMEM_SERVING', '1')
    monkeypatch.delenv('KVMEM_REQUEST_PLAN_CONFIG', raising=False)

    class Tokenizer:
        labels = {label: i + 10 for i, label in enumerate(sorted(
            {label for qtype in QUESTION_TYPES.values() for label in qtype.label_set}))}
        def get_added_vocab(self): return {'<reply>': 999}
        def apply_chat_template(self, *args, **kwargs): return [999]
        def decode(self, ids): return ''
        def encode(self, text, **kwargs): return [self.labels[text]]

    class Renderer:
        renderer = NS(get_tokenizer=lambda: Tokenizer())
        async def preprocess_chat(self, request, messages, **kwargs):
            return messages, (tokens_input([1, 2, 999], cache_salt=request.cache_salt),)

    class Native(Engine):
        errored = False
        model_config = NS(supports_multimodal_inputs=False, is_hybrid=True, is_encoder_decoder=False)
        vllm_config = NS(scheduler_config=NS(max_num_seqs=1),
                         cache_config=NS(enable_prefix_caching=False))
        def check_admission(self, *args): pass
        async def generate(self, prompt, params, request_id, **kwargs):
            release = (params.extra_args or {}).get('kv_transfer_params', {}).get('kvmem_release_session')
            self.calls.append((request_id, kwargs['session_id'], release))
            labels = params.logprob_token_ids or [0]
            yield RequestOutput(request_id=request_id, prompt=None,
                prompt_token_ids=prompt['prompt_token_ids'], prompt_logprobs=None,
                outputs=[CompletionOutput(index=0, text='', token_ids=[labels[0]],
                    cumulative_logprob=None, finish_reason='length',
                    logprobs=[{label: Logprob(math.log(1 / (len(labels) + 1))) for label in labels}])],
                finished=True)

    async def run():
        engine = Native()
        models = NS(is_base_model=lambda name: True, lora_requests={}, model_name=lambda lora: 'test')
        strategy = NextTokenStrategy(ReadContext(engine, Renderer(), None, 'auto', {}))
        handler = ServingStructuredDecisions(models, strategy)
        await SystemOnePlugin().init_state(engine, NS(serving_structured_decisions=handler), NS())
        request = protocol.StructuredDecisionRequest.model_validate({
            'state': 'A damaged package.', 'questions': {
                'damaged': {'type': 'noul', 'instructions': 'Is it damaged?'},
                'action': {'type': 'choice', 'criteria': {'refund': 'Refund', 'keep': 'Keep'}},
            }})
        response = await handler.create_decision(request)
        assert set(response.answers) == {'damaged', 'action'}
        assert response.answers['damaged']['noul'] == pytest.approx(.5)
        assert response.usage.output_tokens == 2
        assert sum(bool(call[2]) for call in engine.calls) == 2
        assert len({call[1] for call in engine.calls}) == 2
    asyncio.run(run())
