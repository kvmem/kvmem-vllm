"""Capability-checked import adapters; prefer native upstream implementations.

Adapted from vllm-project/vllm commit c723a831a81cb4ff89ea6d61b2d15109306ed1bc.
Only the explicit draft LoadConfig selection is backported. No kernel or model
math changes. Installed source files remain unchanged.
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import linecache
import sys

TARGET = "vllm.v1.worker.gpu.spec_decode.dflash.utils"
OLD = "load_config=get_pp_safe_draft_load_config(vllm_config.load_config),"
NEW = """load_config=get_pp_safe_draft_load_config(
            speculative_config.draft_load_config
            if speculative_config.draft_load_config is not None
            else vllm_config.load_config
        ),"""
RUNNER_TARGET = "vllm.v1.worker.gpu.model_runner"


def patched_source(source: bytes) -> str:
    text = source.decode()
    from .engine_api import find_symbol
    loader = find_symbol(text, 'load_dflash_model')
    configs = [node.value for node in ast.walk(loader)
               if isinstance(node, ast.keyword) and node.arg == 'load_config']
    if len(configs) != 1:
        raise ImportError('Unrecognized vLLM DFlash load_config selection')
    expression = ast.unparse(configs[0])
    # Upstream now owns the fix. Leave its loader entirely intact.
    if expression in (
            'get_pp_safe_draft_load_config(get_draft_load_config(vllm_config))',
            'get_pp_safe_draft_load_config(speculative_config.draft_load_config if '
            'speculative_config.draft_load_config is not None else vllm_config.load_config)'):
        return text
    if text.count(OLD) != 1:
        raise ImportError("Unrecognized or ambiguous draft loader patch")
    return text.replace(OLD, NEW)


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(patched_source(self.get_data(self.path)), self.path, "exec")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError("Cannot locate native vLLM DFlash loader")
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def enable_draft_load_config_fix() -> None:
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return
    from .engine_api import engine_root
    source = (engine_root() / 'v1/worker/gpu/spec_decode/dflash/utils.py').read_bytes()
    if patched_source(source) == source.decode():
        return
    if TARGET in sys.modules:
        raise RuntimeError("DFlash loader already imported; activate compatibility before loading vLLM")
    sys.meta_path.insert(0, _Finder())


class _RunnerLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path)
        from .engine_api import require_contract
        require_contract('runner')
        extension = "\nfrom vllm_kvmem.runner import make_probe_runner\nGPUModelRunner = make_probe_runner(GPUModelRunner)\n"
        return compile(source.decode() + extension, self.path, "exec")


class _RunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != RUNNER_TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError("Cannot locate native V2 runner")
        spec.loader = _RunnerLoader(fullname, spec.origin)
        return spec


def enable_runner_probe() -> None:
    if any(isinstance(f, _RunnerFinder) for f in sys.meta_path):
        return
    if RUNNER_TARGET in sys.modules:
        raise RuntimeError("Runner extension must be enabled before V2 runner import")
    from .engine_api import require_contract
    require_contract('runner')
    sys.meta_path.insert(0, _RunnerFinder())


class _MetadataLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        from .metadata_patch import patch_metadata_source
        return compile(patch_metadata_source(self.get_data(self.path)), self.path, "exec")


class _MetadataFinder(importlib.abc.MetaPathFinder):
    target = "vllm.v1.worker.gpu.attn_utils"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError("Cannot locate native attention metadata builder")
        spec.loader = _MetadataLoader(fullname, spec.origin)
        return spec


def enable_window_metadata() -> None:
    if any(isinstance(f, _MetadataFinder) for f in sys.meta_path):
        return
    if _MetadataFinder.target in sys.modules:
        raise RuntimeError("Window metadata must be enabled before native module import")
    sys.meta_path.insert(0, _MetadataFinder())


class _LabelLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        from .logprob_patch import patch_rejection_source
        original = self.get_data(self.path)
        source = patch_rejection_source(original)
        if source == original.decode():
            return compile(source, self.path, 'exec')
        # AST rewriting changes line numbers. Triton reads a decorated kernel's
        # source through inspect during module execution, so compiling against
        # the installed file would silently give it a different function. Keep
        # an immutable, content-addressed source entry without changing that
        # file or its ordinary linecache entry. mtime=None survives checkcache.
        digest = hashlib.sha256(source.encode()).hexdigest()
        filename = f'<kvmem-label-logprobs:{digest}:{self.path}>'
        code = compile(source, filename, 'exec')
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        return code


class _LabelFinder(importlib.abc.MetaPathFinder):
    target = 'vllm.v1.worker.gpu.spec_decode.rejection_sampler'

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError('Cannot locate native speculative sampler')
        spec.loader = _LabelLoader(fullname, spec.origin)
        return spec


def enable_label_logprobs():
    if any(isinstance(f, _LabelFinder) for f in sys.meta_path):
        return
    if _LabelFinder.target in sys.modules:
        raise RuntimeError('Label logprob adaptation must precede sampler import')
    sys.meta_path.insert(0, _LabelFinder())
