"""Keep plugin discovery separate from opting into engine changes."""
from importlib.metadata import version
import os

_registered = False


def register_plugin() -> None:
    global _registered
    if _registered:
        return
    if os.getenv('KVMEM_SERVING') == '1':
        from .compat import enable_label_logprobs
        enable_label_logprobs()
        if os.getenv('KVMEM_LOGICAL_KV_CAPACITY') == '1':
            from .logical_capacity import enable
            enable()
    if os.getenv("KVMEM_DRAFT_LOAD_COMPAT") == "1":
        from .compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    if os.getenv("KVMEM_RUNNER_PROBE_DIR"):
        from .compat import enable_runner_probe
        enable_runner_probe()
    if os.getenv("KVMEM_WINDOW_METADATA") == "1":
        from .compat import enable_window_metadata
        enable_window_metadata()
    if os.getenv("KVMEM_Q8_TENSORCORE") == "1":
        from .q8_adapter import enable_q8_tensorcore
        enable_q8_tensorcore()
    # Discovery alone changes nothing and initializes no CUDA. Explicit options
    # above install capability-checked import adapters in this worker process.
    _registered = True


def check_engine_version() -> str:
    """Retain the public diagnostic entry, without an exact-version gate."""
    from .engine_api import require_contract
    require_contract('runner')
    return version("vllm")
