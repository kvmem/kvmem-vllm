"""Split each ordinary B12X QSA run into select, KVMem hooks, attend.

The vendor QSA layer calls ``qsa.run`` through the module-level accessor
``get_b12x_qsa``. ``install_split`` wraps that accessor once per process and
returns the hook list. While the list is not empty, an ordinary run becomes:

1. the vendor's ``run(..., selection_only=True)``: selector state updates,
   ``binding.selected_positions`` and MTP draft anchors, as in a combined run;
2. every hook, in order: ``hook(binding, inputs, query_positions)`` may rewrite
   ``binding.selected_positions`` in place and may return the
   ``query_positions`` attention should mask against;
3. the vendor's own attention launch on the same binding.

Selection and attention read and write the same buffers in the same order as
the combined call, so hooks that change nothing leave the outputs equal.
Draft reuse, cache-only, DCP and overlapped (selection-stream) runs pass
through unchanged.
"""
import importlib
import sys

VENDOR_MODULE = 'vllm.models.qwen4_exp.nvidia.b12x_qsa'


class SplitQSA:
    def __init__(self, api, hooks):
        self._api = api
        self._attention = sys.modules[api.run.__module__]._run_attention
        self.hooks = hooks

    def __getattr__(self, name):
        return getattr(self._api, name)

    def run(self, binding, **kwargs):
        if (not self.hooks or kwargs.get('cache_only') or kwargs.get('selection_only')
                or kwargs.get('reuse') is not None or kwargs.get('index_ready') is not None
                or getattr(binding, 'selection_stream', None) is not None):
            return self._api.run(binding, **kwargs)
        self._api.run(binding, **kwargs, selection_only=True)
        positions = kwargs['query_positions']
        for hook in self.hooks:
            result = hook(binding, kwargs, positions)
            if result is not None:
                positions = result
        return self._attention(binding, query=kwargs['query'], request_ids=kwargs['request_ids'],
                               query_positions=positions)


def install_split(runner):
    """The process's hook list (the accessor is wrapped on first call)."""
    hooks = getattr(runner, '_kvmem_qsa_split_hooks', None)
    if hooks is not None:
        return hooks
    vendor = sys.modules.get(VENDOR_MODULE) or importlib.import_module(VENDOR_MODULE)
    original = vendor.get_b12x_qsa
    hooks = runner._kvmem_qsa_split_hooks = []
    wrapped = {}

    def accessor():
        api = original()
        if api is None:
            return None
        split = wrapped.get(id(api))
        if split is None or split._api is not api:
            split = wrapped[id(api)] = SplitQSA(api, hooks)
        return split

    vendor.get_b12x_qsa = accessor
    return hooks
