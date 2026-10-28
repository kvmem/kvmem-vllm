"""Give upstream decision reads bounded, disposable KVMem sessions.

The native router, renderer, question types and probability computation remain
upstream-owned. This endpoint plugin only wraps the decision strategy's engine.
"""
import asyncio
import copy
import logging
import os
from uuid import uuid4

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self, engine, max_concurrency):
        if type(max_concurrency) is not int or max_concurrency <= 0:
            raise ValueError('Decision reads require a positive concurrency limit')
        self.engine = engine
        self.slots = asyncio.Semaphore(max_concurrency)
        self.cleanups = set()

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _cleanup_done(self, task):
        self.cleanups.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error('Decision session cleanup failed', exc_info=task.exception())

    async def _release(self, iterator, request_id, session_id, prompt, complete):
        try:
            try:
                await iterator.aclose()
            finally:
                if not complete:
                    await self.engine.abort(request_id)
            from vllm.sampling_params import SamplingParams
            params = SamplingParams(max_tokens=1, temperature=0,
                extra_args={'kv_transfer_params': {'kvmem_release_session': True}})
            # Use the existing fenced close protocol, including on cancellation.
            close_prompt = dict(prompt, prompt_token_ids=prompt['prompt_token_ids'][:1])
            async for _ in self.engine.generate(close_prompt, params,
                    request_id + '-release', session_id=session_id):
                pass
        finally:
            self.slots.release()

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        from .systemone_active import active_read
        batch = active_read.get()
        if batch is not None:
            async for output in batch.generate(prompt, sampling_params, request_id, **kwargs):
                yield output
            return
        tokens = prompt.get('prompt_token_ids') if isinstance(prompt, dict) else None
        if (not tokens or any(type(t) is not int or t < 0 for t in tokens)
                or prompt.get('type') not in (None, 'token')
                or 'multi_modal_data' in prompt or 'mm_kwargs' in prompt):
            raise ValueError('KVMem decision reads require a rendered text token prompt')
        # Caller-supplied IDs cannot collide with an agent's persistent session.
        identity = 'kvmem-decision-' + uuid4().hex
        params = copy.deepcopy(sampling_params)
        extras = dict(params.extra_args or {})
        from .http_sessions import normalize_query_plan
        payload = {'prompt': list(tokens), 'max_tokens': params.max_tokens,
                   'session_id': identity,
                   'kv_transfer_params': dict(extras.get('kv_transfer_params') or {})}
        normalize_query_plan(payload)
        extras['kv_transfer_params'] = payload['kv_transfer_params']
        params.extra_args = extras
        kwargs = {**kwargs, 'session_id': identity}
        await self.slots.acquire()
        complete = False
        try:
            iterator = self.engine.generate(prompt, params, identity, **kwargs)
        except BaseException:
            self.slots.release()
            raise
        try:
            async for output in iterator:
                complete = bool(getattr(output, 'finished', False))
                yield output
        finally:
            task = asyncio.create_task(self._release(iterator, identity, identity, prompt, complete))
            self.cleanups.add(task)
            task.add_done_callback(self._cleanup_done)
            # A second disconnect must not cancel fenced cleanup or free its slot early.
            await asyncio.shield(task)


class SystemOnePlugin:
    name = 'kvmem'
    required_tasks = ('generate',)

    def attach_router(self, app):
        pass  # vLLM owns /v1/systemone.

    async def init_state(self, engine_client, state, args):
        if os.getenv('KVMEM_SERVING') != '1':
            return
        serving = getattr(state, 'serving_structured_decisions', None)
        if serving is None:
            return  # Older vLLM or a model without native decisions.
        from .engine_api import require_contract
        require_contract('systemone')
        context = serving.strategy.context
        if isinstance(context.engine_client, DecisionEngine):
            return
        from dataclasses import replace
        concurrency = engine_client.vllm_config.scheduler_config.max_num_seqs
        serving.strategy.context = replace(context,
            engine_client=DecisionEngine(engine_client, concurrency))
        if hasattr(serving, 'create_decision'):
            from .systemone_active import install_active_handler
            install_active_handler(serving, serving.strategy.context.engine_client)
