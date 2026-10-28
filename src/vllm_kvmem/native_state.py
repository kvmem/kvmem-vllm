"""Borrow exact native Qwen GDN state pages for fenced query replay.

No reconstructed or approximated recurrent state: both convolution and SSM
tensors, every allocated speculation row, and the native acceptance counter are
included. Prefix-cached/aligned and context-parallel layouts need separate adapters.
"""
from dataclasses import dataclass
from functools import lru_cache

import torch

from .checkpoint_timing import async_checkpoint_capture, checkpoint_stage
from .publication_step import native_publication_step
from .native_pages import stored_expansions


@lru_cache(maxsize=1)
def verify_state_sources():
    from .engine_api import require_contract
    require_contract('state')


@dataclass
class StateBundle:
    tensors: dict[str, torch.Tensor]
    request_indices: dict[str, int]
    request_generations: dict[str, int]
    page_rows: dict[tuple[str, int], tuple[int, ...]]

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())


class NativeRecurrentState:
    def __init__(self, runner):
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        verify_state_sources()
        config = runner.vllm_config
        if config.cache_config.mamba_cache_mode != "none" or config.cache_config.enable_prefix_caching:
            raise ValueError("Native GDN replay currently requires non-prefix-cached 'none' state layout")
        parallel = config.parallel_config
        if any(getattr(parallel, name, 1) != 1 for name in (
                "tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size",
                "prefill_context_parallel_size")):
            raise ValueError("Native GDN replay currently requires a single worker rank")
        if getattr(runner.model_state, "recoverssm", None) is not None:
            raise ValueError("RecoverSSM state is not part of the qualified replay schema")
        self.runner = runner
        self.groups = {}
        self.conv_widths = {}
        self.conv_dim_first = is_conv_state_dim_first()
        modules = config.compilation_config.static_forward_context
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            spec = spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            if not isinstance(spec, MambaSpec):
                continue
            layers = {}
            for name in group.layer_names:
                module = modules[name]
                if not isinstance(module, QwenGatedDeltaNetAttention):
                    raise ValueError(f"Unqualified recurrent-state layer: {name}")
                if not isinstance(module.kv_cache, (tuple, list)) or len(module.kv_cache) != 2:
                    raise ValueError("Expected native convolution and recurrent cache tensors")
                layers[name] = module.kv_cache
                self.conv_widths[name] = module.conv1d.weight.shape[-1]
            if stored_expansions(runner.block_tables)[group_id] != 1:
                raise ValueError("Recurrent state rows must not use attention page expansion")
            self.groups[group_id] = layers
        if not self.groups:
            raise ValueError("No native Qwen GDN state groups found")

    def _rows(self, request_id, group_id):
        index = self.runner.req_states.req_id_to_index[request_id]
        count = int(self.runner.block_tables.num_blocks.np[group_id, index])
        if count <= 0:
            raise ValueError("Request has no allocated native recurrent state")
        # The CPU witness is populated from the actual native writes,
        # never from num_blocks or an assumed persistent CPU table mirror.
        journal = getattr(self.runner, '_kvmem_recurrent_table_journal', None)
        rows = journal.rows(request_id, group_id) if journal is not None and async_checkpoint_capture(self.runner) else None
        if rows is None:
            # Called only at a quiescent boundary, after native staged writes.
            with checkpoint_stage(self.runner, 'page_table_gpu_read', sync=True):
                rows = tuple(self.runner.block_tables.block_tables[group_id].gpu[index, :count].cpu().tolist())
        else:
            trace = getattr(self.runner, '_kvmem_checkpoint_trace', None)
            if trace is not None:
                trace['cpu_page_table_reads'] = trace.get('cpu_page_table_reads', 0) + 1
        if any(row <= 0 for row in rows) or len(set(rows)) != count:
            raise ValueError("Null or duplicate recurrent state page")
        return index, rows

    def borrow(self, request_ids):
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Duplicate request in a recurrent checkpoint")
        tensors, indices, generations, pages = {}, {}, {}, {}
        used = set()
        for request_id in request_ids:
            generations[request_id] = self.runner._kvmem_request_generations[request_id]
            for group_id, layers in self.groups.items():
                index, rows = self._rows(request_id, group_id)
                indices[request_id] = index
                pages[request_id, group_id] = rows
                for row in rows:
                    if (group_id, row) in used:
                        raise ValueError("Different requests share a writable recurrent state page")
                    used.add((group_id, row))
                    for layer_name, caches in layers.items():
                        for kind, cache in zip(("conv", "ssm"), caches):
                            if row >= cache.shape[0]:
                                raise ValueError("Native recurrent row exceeds its cache arena")
                            tensors[f"{request_id}/{layer_name}/{kind}/{row}"] = cache[row]
            counters = self.runner.model_state.num_accepted_tokens_gpu
            tensors[f"{request_id}/num_accepted"] = counters[index:index+1]
        return StateBundle(tensors, indices, generations, pages)

    def validate(self, bundle):
        for (request_id, group_id), rows in bundle.page_rows.items():
            if self.runner._kvmem_request_generations.get(request_id) != bundle.request_generations[request_id]:
                raise ValueError("Request incarnation changed since checkpoint")
            if self.runner.req_states.req_id_to_index.get(request_id) != bundle.request_indices[request_id]:
                raise ValueError("Request state slot was freed or reused since checkpoint")
            _, current = self._rows(request_id, group_id)
            if current != rows:
                raise ValueError("Native recurrent page table changed since checkpoint")

    def canonicalize_accepted(self, request_id, *, retained_tokens, query_tokens, speculative):
        """Move a verified prefix's native state to base rows before continuation.

        Caller must supply the actual last native query and scheduler-retained
        input count at a quiescent boundary, and hold its physical page leases.
        Zero retention and partial non-speculative prefill require checkpoint
        replay; those earlier states are no longer present in these cache rows.
        This does not restore the independent draft KV/hidden-state lifecycle.
        """
        if (type(speculative) is not bool or type(retained_tokens) is not int or
                type(query_tokens) is not int or not 0 < retained_tokens <= query_tokens):
            raise ValueError('Retained prefix requires positive query rows; zero needs checkpoint replay')
        if not speculative and retained_tokens != query_tokens:
            raise ValueError('Partial non-speculative query requires checkpoint replay')
        bundle = self.borrow([request_id])
        copies = []
        for group_id, layers in self.groups.items():
            rows = bundle.page_rows[request_id, group_id]
            if speculative and query_tokens > len(rows):
                raise ValueError('Speculative query exceeds allocated recurrent rows')
            offset = retained_tokens - 1 if speculative else 0
            for name, (conv, ssm) in layers.items():
                width = self.conv_widths[name] - 1
                base_conv = conv[rows[0]]
                # Normalize both native SD and DS representations to [time, dim].
                temporal = base_conv.transpose(0, 1) if self.conv_dim_first else base_conv
                if temporal.ndim != 2 or width <= 0 or offset + width > temporal.shape[0]:
                    raise ValueError('Native convolution state cannot represent retained prefix')
                if offset:
                    copies.append((temporal[:width], temporal[offset:offset+width]))
                    copies.append((ssm[rows[0]], ssm[rows[offset]]))
        # Validate every group before modifying any state. Cloning handles
        # overlapping convolution windows; scratch is bounded to one copy.
        self.validate(bundle)
        for destination, source in copies:
            destination.copy_(source.clone())
        counter = self.runner.model_state.num_accepted_tokens_gpu
        counter[bundle.request_indices[request_id]].fill_(1)
        if counter.device.type == 'cuda':
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(counter.device))
            return event
        return None


def initialize_cold_single_token_state(runner, batch):
    """A fresh one-token prompt uses native decode kernels, including SD.

    Those kernels consume recurrent state unconditionally. Recycled pages must
    have the same zero initial state that the multi-token prefill path supplies.
    Never clear an imported or continuing prefix, or another request's pages.
    """
    ids=[rid for i,rid in enumerate(batch.req_ids)
         if batch.num_computed_tokens_np[i]==0 and batch.prefill_len_np[i]==1]
    if not ids:
        return
    adapter=NativeRecurrentState(runner)
    bundle=adapter.borrow(ids)
    adapter.validate(bundle)
    for name,tensor in bundle.tensors.items():
        if name.endswith('/num_accepted'):
            tensor.fill_(1)
        else:
            tensor.zero_()


def canonicalize_before_non_speculative_batch(runner, batch):
    """Preserve accepted SD rows before a clipped batch uses native base rows."""
    runtime = getattr(runner, '_kvmem_capture_runtime', None)
    if runtime is None or runner.speculative_config is None:
        return
    drafts = batch.num_draft_tokens_per_req
    candidates = [i for i in range(batch.num_reqs)
                  if (drafts is None or drafts[i] == 0) and not batch.is_prefilling_np[i]
                  and batch.num_computed_tokens_np[i] > 0]
    if not candidates:
        return
    if not runtime.drain_requests((batch.req_ids[i] for i in candidates), wait=True):
        raise RuntimeError('Native continuation preceded scheduler publication confirmation')
    adapter = None
    for i in candidates:
        request_id = batch.req_ids[i]
        steps = runtime.last_steps.get(request_id, ())
        if not steps:
            continue
        step = native_publication_step(steps[-1])
        grouped_prefill = step is not steps[-1]
        if grouped_prefill and step['accepted'] != step['scheduled']:
            raise ValueError('Partial non-speculative query requires checkpoint replay')
        if step['accepted'] <= 1:
            continue
        index = runner.req_states.req_id_to_index[request_id]
        if int(runner.model_state.num_accepted_tokens_gpu[index].item()) == 1:
            continue
        if step['history_begin'] + step['accepted'] != int(batch.num_computed_tokens_np[i]):
            raise ValueError('Non-speculative continuation differs from its last accepted prefix')
        if adapter is None:
            adapter = NativeRecurrentState(runner)
        adapter.canonicalize_accepted(request_id, retained_tokens=step['accepted'],
                                     query_tokens=step['scheduled'], speculative=not grouped_prefill)
