"""Keep plugin discovery separate from opting into engine changes."""
from importlib.metadata import version
import os
from ..core import settings as _settings

_registered = False


def register_plugin() -> None:
    global _registered
    if _registered:
        return
    if _settings.get('KVMEM_QSA_ADAPTER') == '1':
        from ..adapters.qsa_config_compat import enable
        enable()
    if _settings.get('KVMEM_SERVING') == '1':
        import sys
        from ..core.defaults import summary
        print(summary(), file=sys.stderr, flush=True)
        from ..adapters.compat import enable_label_logprobs
        enable_label_logprobs()
        if _settings.get('KVMEM_LOGICAL_KV_CAPACITY') == '1':
            from .logical_capacity import enable
            enable()
    if _settings.get("KVMEM_DRAFT_LOAD_COMPAT") == "1":
        from ..adapters.compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    if _settings.get("KVMEM_RUNNER_PROBE_DIR"):
        from ..adapters.compat import enable_runner_probe
        enable_runner_probe()
    if _settings.get("KVMEM_WINDOW_METADATA") == "1":
        from ..adapters.compat import enable_window_metadata
        enable_window_metadata()
    if _settings.get("KVMEM_Q8_TENSORCORE") == "1":
        from ..adapters.q8_adapter import enable_q8_tensorcore
        enable_q8_tensorcore()
    from ..adapters import fa2_verify_graph
    if fa2_verify_graph.enabled():
        fa2_verify_graph.enable()
    # Discovery alone changes nothing and initializes no CUDA. Explicit options
    # above install capability-checked import adapters in this worker process.
    _registered = True


# Defined with the engine contract it checks (layering step 7).
from ..adapters.engine_api import check_engine_version  # noqa: E402,F401
