"""Optional native SystemOne reads against one existing KVMem session.

The upstream handler still owns validation, rendering and answer calculation.
All label reads in an HTTP request enter one worker transaction, so every
question observes the same accepted parent state. Ordinary reads are unchanged.
"""
import asyncio
from contextvars import ContextVar
from http import HTTPStatus
from uuid import uuid4


active_read = ContextVar('kvmem_systemone_active_read', default=None)


class ActiveDecisionError(ValueError):
    pass


def session_reference(request):
    reference = (request.model_extra or {}).get('kvmem_session')
    if reference is None:
        return None
    if (not isinstance(reference, dict) or
            set(reference) - {'session_id', 'generation', 'history_end'} or
            not isinstance(reference.get('session_id'), str) or not reference['session_id']):
        raise ValueError('kvmem_session requires session_id and optional generation/history_end')
    from .session_control import session_namespace
    namespace = session_namespace(reference['session_id'], request.cache_salt)
    for key in ('generation', 'history_end'):
        if key in reference and (type(reference[key]) is not int or reference[key] <= 0):
            raise ValueError(f'kvmem_session.{key} must be a positive integer')
    if request.state != '' or request.instructions:
        raise ValueError('Active-session decisions require state="" and no top-level instructions; '
                         'put the question instructions in questions')
    if any(getattr(request, name, None) for name in ('images', 'audio', 'videos')):
        raise ValueError('Active-session decisions currently support text only')
    if not 1 <= len(request.questions) <= 8:
        raise ValueError('Active-session decisions support 1 to 8 questions per snapshot')
    return {**reference, 'namespace': list(namespace)}


class ActiveReadBatch:
    def __init__(self, proxy, reference, count, tokenizer):
        self.proxy, self.reference, self.count = proxy, reference, count
        self.calls = []
        self.ready = asyncio.Event()
        self.task = None
        self.snapshot = None
        vocab = tokenizer.get_added_vocab()
        if not {'<|im_start|>', '<|im_end|>', '</think>'} <= vocab.keys():
            raise ValueError('Active decisions currently require the Qwen chat template')
        self.user_header = tokenizer.encode('<|im_start|>user\n', add_special_tokens=False)
        self.turn_end = tokenizer.encode('</think>\n<|im_end|>\n', add_special_tokens=False)

    async def generate(self, prompt, params, request_id, **kwargs):
        tokens = prompt.get('prompt_token_ids') if isinstance(prompt, dict) else None
        if (not tokens or prompt.get('type') not in (None, 'token') or
                any(type(t) is not int or t < 0 for t in tokens) or
                'multi_modal_data' in prompt or 'mm_kwargs' in prompt or
                kwargs.get('lora_request') is not None or params.max_tokens != 1 or
                not params.logprob_token_ids):
            raise ValueError('Active decisions require native text-only one-token label reads')
        if len(tokens) > 512:
            raise ValueError('An active decision question exceeds the 512-token scratch limit')
        if tokens[:len(self.user_header)] != self.user_header:
            raise ValueError('Active decision template must start with the question user turn')
        # Continue the existing assistant turn; the question body and reply tail
        # still come from the native renderer and label strategy.
        tokens = self.turn_end + list(tokens)
        index = len(self.calls)
        if index >= self.count:
            raise ValueError('Native decision strategy produced too many reads')
        self.calls.append({'tokens': list(tokens), 'labels': list(params.logprob_token_ids)})
        if len(self.calls) == self.count:
            self.task = asyncio.create_task(self._execute())
            self.ready.set()
        await self.ready.wait()
        result = (await asyncio.shield(self.task))['reads'][index]
        from vllm.logprobs import Logprob
        from vllm.outputs import CompletionOutput, RequestOutput
        # Feed the original label-read strategy its native output type. It owns
        # normalization, label_mass, confidence and all question-type semantics.
        yield RequestOutput(request_id=request_id, prompt=None,
            prompt_token_ids=tokens, prompt_logprobs=None, finished=True,
            outputs=[CompletionOutput(index=0, text='', token_ids=[result['argmax']],
                cumulative_logprob=None, finish_reason='length',
                logprobs=[{label: Logprob(score) for label, score in
                           zip(params.logprob_token_ids, result['logprobs'])}])])

    async def _execute(self):
        operation = uuid4().hex
        engine = self.proxy.engine
        async def rpc(action, **kwargs):
            result = await engine.collective_rpc('kvmem_systemone_active',
                kwargs={'action': action, 'operation': operation, **kwargs})
            if len(result) != 1:
                raise ActiveDecisionError('Active decisions require one worker')
            result = result[0]
            if result.get('error'):
                raise ActiveDecisionError(result['error'])
            return result
        async with self.proxy.slots:
            try:
                await rpc('submit', reference=self.reference, reads=self.calls)
                async with asyncio.timeout(30):
                    while True:
                        result = await rpc('poll')
                        if result.get('status') == 'done':
                            self.snapshot = result['snapshot']
                            return result
                        await asyncio.sleep(.02)
            except TimeoutError as error:
                raise ActiveDecisionError('No eligible active parent step within 30 seconds') from error
            finally:
                # Cancellation cannot leave an operation queued for a later turn.
                await asyncio.shield(rpc('cancel'))

    async def close(self):
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


def install_active_handler(serving, proxy):
    original = serving.create_decision

    async def create_decision(request, raw_request=None):
        try:
            reference = session_reference(request)
        except ValueError as error:
            return serving.create_error_response(error)
        if reference is None:
            return await original(request, raw_request)
        from .engine_api import require_contract
        try:
            require_contract('systemone_active')
            tokenizer = serving.strategy.context.online_renderer.renderer.get_tokenizer()
            batch = ActiveReadBatch(proxy, reference, len(request.questions), tokenizer)
        except ValueError as error:
            return serving.create_error_response(error)
        token = active_read.set(batch)
        try:
            result = await original(request, raw_request)
            if batch.snapshot is not None and hasattr(result, 'answers'):
                result = result.model_copy(update={'kvmem': batch.snapshot})
            return result
        except ActiveDecisionError as error:
            return serving.create_error_response(str(error), status_code=HTTPStatus.CONFLICT)
        except ValueError as error:
            return serving.create_error_response(error)
        finally:
            active_read.reset(token)
            cleanup = asyncio.create_task(batch.close())
            proxy.cleanups.add(cleanup)
            cleanup.add_done_callback(proxy._cleanup_done)
            await asyncio.shield(cleanup)

    serving.create_decision = create_decision
