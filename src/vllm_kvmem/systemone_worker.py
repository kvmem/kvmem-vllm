"""Fenced, target-only decision reads on existing active GPU pages.

RPC only queues work. The runner executes it at a scheduler-confirmed boundary,
before building public attention metadata. This is a serialized temporary branch,
not a second scheduler request or an independent page owner.
"""
import os
import time

from .systemone_active import ActiveDecisionError


class SystemOneWorker:
    def kvmem_systemone_active(self, action, operation, reference=None, reads=None):
        queue = getattr(self.model_runner, '_kvmem_active_decisions', None)
        if queue is None:
            queue = ActiveDecisionQueue(self.model_runner)
            self.model_runner._kvmem_active_decisions = queue
        return queue.rpc(action, operation, reference, reads)


class ActiveDecisionQueue:
    def __init__(self, runner):
        self.runner, self.pending = runner, {}

    def _binding(self, reference):
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        namespace = tuple(reference['namespace'])
        bindings = [] if sessions is None else [b for b in sessions.bindings.values()
            if b.namespace == namespace and b.request_id not in sessions.retiring]
        if len(bindings) != 1:
            raise ActiveDecisionError('Session has no single active GPU owner')
        binding = bindings[0]
        if reference.get('generation', binding.generation) != binding.generation:
            raise ActiveDecisionError('Session generation changed')
        return binding

    def rpc(self, action, operation, reference=None, reads=None):
        now = time.monotonic()
        self.pending = {key: job for key, job in self.pending.items() if job['expires'] > now}
        if action == 'cancel':
            self.pending.pop(operation, None)
            return {'status': 'cancelled'}
        if action == 'poll':
            job = self.pending.get(operation)
            return job['result'] if job else {'error': 'Decision operation expired or missing'}
        if action != 'submit' or operation in self.pending:
            return {'error': 'Invalid or duplicate decision operation'}
        try:
            if len(self.pending) >= 16:
                raise ActiveDecisionError('Active decision queue is full')
            if not reads or len(reads) > 8 or any(not r['tokens'] or len(r['tokens']) > 544
                    or not r['labels'] for r in reads):
                raise ActiveDecisionError('Invalid bounded native label reads')
            runner = self.runner
            parallel = getattr(getattr(runner, 'vllm_config', None), 'parallel_config', None)
            if (not getattr(runner, '_kvmem_bounded_target', False) or
                    any(getattr(parallel, name, 1) != 1 for name in (
                        'tensor_parallel_size', 'pipeline_parallel_size',
                        'decode_context_parallel_size', 'prefill_context_parallel_size')) or
                    getattr(runner, '_kvmem_async_refresh', None) is not None or
                    getattr(runner, '_kvmem_native_attention', None) is not None or
                    getattr(runner, '_kvmem_graph', False) or
                    getattr(getattr(runner, 'speculative_config', None), 'method', None)
                    not in (None, 'mtp')):
                raise ActiveDecisionError('Active decisions currently require eager bounded '
                    'Qwen execution, optional MTP, synchronous refresh, and no native attention scorer')
            binding = self._binding(reference)
            self.pending[operation] = dict(binding=binding, reference=reference, reads=reads,
                expires=now + 30, result={'status': 'pending'})
            return {'status': 'pending'}
        except ActiveDecisionError as error:
            return {'error': str(error)}

    def before_attention(self, batch):
        for job in list(self.pending.values()):
            if job['result'].get('status') != 'pending':
                continue
            try:
                if job['expires'] <= time.monotonic():
                    raise ActiveDecisionError('Decision operation expired')
                binding = self._binding(job['reference'])
                if binding != job['binding']:
                    raise ActiveDecisionError('Active session owner changed before read')
                rid = binding.request_id
                if rid not in batch.req_ids:
                    continue
                index = batch.req_ids.index(rid)
                history = int(batch.num_computed_tokens_np[index])
                if history < int(batch.prefill_len_np[index]) or not history:
                    continue  # Do not expose a partly prefetched prompt.
                sessions = self.runner._kvmem_sessions
                if rid in sessions.imports:
                    continue
                runtime = self.runner._kvmem_capture_runtime
                if not runtime.drain_requests([rid], wait=True):
                    continue  # RPC never substitutes for scheduler acceptance.
                if runtime.authorities[rid].history_end != history:
                    raise ActiveDecisionError('Accepted history and GPU boundary disagree')
                if job['reference'].get('history_end', history) != history:
                    raise ActiveDecisionError('Requested history boundary is no longer active')
                job['result'] = execute_reads(self.runner, batch, index, binding, job['reads'])
            except ActiveDecisionError as error:
                job['result'] = {'error': str(error)}
            # Unexpected GPU/rollback failures propagate: continuing an uncertain
            # parent would violate the ownership contract.


def tail_views(runner, rid, begin, count):
    """Only already allocated, exclusive target pages may be used as scratch."""
    import torch
    from .codec import native_vllm_fp8_views
    from .native_pages import kernel_table
    modules = dict(runner.model.named_modules())
    groups = {name: gid for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups)
              for name in group.layer_names}
    index = runner.req_states.req_id_to_index[rid]
    views = []
    for name in runner._kvmem_layers:
        module = modules[name]
        attn = module.attn
        if attn.kv_cache_dtype != 'fp8_e4m3':
            raise ActiveDecisionError('Active decision scratch requires qualified FP8 KV')
        table = kernel_table(runner, groups[attn.layer_name], index).clone()
        key, value = native_vllm_fp8_views(attn.kv_cache, module.num_kv_heads, module.head_dim)
        if begin + count > table.numel() * key.shape[1]:
            raise ActiveDecisionError('Active parent has insufficient allocated KV tail space')
        if begin + count > module.rotary_emb.cos_sin_cache.shape[0]:
            raise ActiveDecisionError('Decision exceeds the native position table')
        positions = torch.arange(begin, begin + count, device=key.device)
        pages, offsets = table[positions // key.shape[1]].long(), positions % key.shape[1]
        if bool(torch.any(pages <= 0)) or bool(torch.any(pages >= key.shape[0])):
            raise ActiveDecisionError('Decision scratch contains a null or invalid page')
        for tensor in (key, value):
            views.append((tensor.view(torch.uint8), pages, offsets))
    return views


class TailSnapshot:
    def __init__(self, views):
        self.rows = [(tensor, pages, offsets, tensor[pages, offsets].clone())
                     for tensor, pages, offsets in views]

    def restore(self):
        for tensor, pages, offsets, saved in self.rows:
            tensor[pages, offsets] = saved

    def verify(self):
        import torch
        return all(torch.equal(tensor[pages, offsets], saved)
                   for tensor, pages, offsets, saved in self.rows)


def execute_reads(runner, source, index, binding, reads):
    import torch
    from .cache_groups import full_attention_groups
    from .native_state import NativeRecurrentState
    from .publication_step import native_publication_step
    from .state import StateCheckpoints
    from .window_batch import WindowBatchAdapter
    rid = binding.request_id
    history = int(source.num_computed_tokens_np[index])
    adapter = runner._kvmem_window_adapter
    offset = adapter.offsets.get(rid, 0) if adapter is not None else 0
    window = history - offset
    runtime = runner._kvmem_capture_runtime
    capacity = runtime.config.select_budget + runtime.config.gen_budget
    count = max(len(read['tokens']) for read in reads)
    if not 0 < window <= capacity - count or history + count > runner.model_config.max_model_len:
        raise ActiveDecisionError('Decision scratch exceeds the active window or model limit')
    steps = runtime.last_steps.get(rid, ())
    step = native_publication_step(steps[-1]) if steps else None
    if step is None or step['history_begin'] + step['accepted'] != history:
        raise ActiveDecisionError('No exact accepted recurrent-state boundary')
    views = tail_views(runner, rid, window, count)
    state = NativeRecurrentState(runner)
    bundle = state.borrow([rid])
    tail_bytes = sum(count * t.shape[2:].numel() * t.element_size() for t, _, _ in views)
    if bundle.bytes + tail_bytes > 512 * 1024**2:
        raise ActiveDecisionError('Decision rollback snapshot exceeds 512 MiB')
    checkpoints = StateCheckpoints(rid, bundle.bytes, capacity=1)
    checkpoint = checkpoints.capture(history, bundle.tensors)
    tail = TailSnapshot(views)
    audit = os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1'
    prefix = TailSnapshot(tail_views(runner, rid, 0, window)) if audit else None
    private_adapter = WindowBatchAdapter(full_attention_groups(runner.kv_cache_config), capacity)
    private_adapter.offsets = {rid: offset}
    attributes = ('_kvmem_window_adapter', '_kvmem_last_batch',
                  '_kvmem_window_batch', '_kvmem_projecting_query')
    saved = {name: getattr(runner, name, None) for name in attributes}
    started = time.perf_counter()
    results = []
    try:
        runner._kvmem_projecting_query = True
        runner._kvmem_window_adapter = private_adapter
        for read in reads:
            state.validate(bundle)
            checkpoints.restore(checkpoint, bundle.tensors)
            tail.restore()
            counter = runner.model_state.num_accepted_tokens_gpu
            if int(counter[bundle.request_indices[rid]].item()) != 1:
                state.canonicalize_accepted(rid, retained_tokens=step['accepted'],
                    query_tokens=step['scheduled'], speculative=step['history_begin'] +
                    step['scheduled'] > int(source.prefill_len_np[index]))
            results.append(private_read(runner, source, index, history, read))
    finally:
        # Restore every overwritten byte, including speculative lookahead rows.
        state.validate(bundle)
        checkpoints.restore(checkpoint, bundle.tensors)
        tail.restore()
        torch.cuda.current_stream(runner.device).synchronize()
        for name, value in saved.items():
            setattr(runner, name, value)
        if audit:
            from .guided_query import verify_guided_rollback
            verify_guided_rollback(checkpoint, bundle.tensors, [])
            if not tail.verify() or not prefix.verify():
                raise RuntimeError('Active decision changed parent KV bytes')
        if not checkpoints.release(history):
            raise RuntimeError('Active decision rollback did not finish')
    return {'status': 'done', 'reads': results, 'snapshot': {
        'session_id': binding.namespace[1], 'generation': binding.generation,
        'history_end': history, 'active_tokens': window,
        'question_tokens': sum(len(read['tokens']) for read in reads),
        'rollback_bytes': bundle.bytes + tail_bytes,
        'rollback_verified': audit,
        'execution_ms': (time.perf_counter() - started) * 1000,
        'mode': 'active_branch'}}


def private_read(runner, source, index, history, read):
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
    from .capture import capture_to
    from .guided_query import private_batch
    tokens = read['tokens']
    batch = private_batch(source, index, tokens, history, runner.device)
    tables, slots = runner.prepare_attn(batch)
    metadata = runner.model_state.prepare_attn(batch, CUDAGraphMode.NONE, tables, slots,
                                              runner.attn_groups, runner.kv_cache_config)
    draft_groups = getattr(runner.speculator, 'draft_kv_cache_group_ids', ())
    draft_names = {name for group in draft_groups
                  for name in runner.kv_cache_config.kv_cache_groups[group].layer_names}
    metadata = {name: value for name, value in metadata.items() if name not in draft_names}
    inputs = {'input_ids': batch.input_ids, 'positions': batch.positions,
              'inputs_embeds': None, 'intermediate_tensors': None,
              **runner.model_state.prepare_inputs(batch, runner.req_states)}
    mapping = build_slot_mappings_by_layer(slots, runner.kv_cache_config)
    with set_forward_context(metadata, runner.vllm_config, num_tokens=len(tokens),
            slot_mapping=mapping, skip_compiled=True, is_padding=batch.is_padding), capture_to(lambda _: None):
        result = runner._kvmem_private_forward(**inputs)
    hidden = result[0] if isinstance(result, tuple) else result
    logits = runner.model.compute_logits(hidden[-1:])
    labels = torch.tensor([read['labels']], device=logits.device, dtype=torch.int64)
    return {'argmax': int(logits.argmax(-1).item()),
            'logprobs': compute_token_logprobs(logits, labels)[0].cpu().tolist()}
