"""CPU regressions for source introspection in the native sampler adapter."""
import ast
import inspect
import json
import linecache
import os
from pathlib import Path
import subprocess
import sys
import types
from importlib.metadata import PackageNotFoundError

import pytest

from vllm_kvmem.compat import _LabelLoader
from vllm_kvmem.engine_api import engine_root


def rewrite(source):
    return ast.unparse(ast.parse(source.decode())) + '\n'


def execute(loader):
    captured = []

    def fake_jit(fn):
        # The real Triton decorator inspects source while the module executes.
        captured.append(inspect.getsource(fn))
        return fn

    scope = {'__name__': loader.name, 'jit': fake_jit}
    exec(loader.get_code(loader.name), scope)
    return scope, captured


def native_source():
    if source := os.getenv('KVMEM_TEST_VLLM_SOURCE'):
        root = Path(source)
    else:
        try:
            root = engine_root()
        except PackageNotFoundError:
            pytest.skip('Install vLLM or set KVMEM_TEST_VLLM_SOURCE')
    return root / 'v1/worker/gpu/spec_decode/rejection_sampler.py'


def test_fake_triton_reads_compiled_source_without_changing_original(tmp_path, monkeypatch):
    from vllm_kvmem import logprob_patch
    monkeypatch.setattr(logprob_patch, 'patch_rejection_source', rewrite)
    path = tmp_path / 'sampler.py'
    original = '# original comments shift the kernel\n\n\ndef helper():\n    return 7\n\n\n@jit\ndef kernel(x):\n    return x + 1\n'
    path.write_text(original)
    original_lines = linecache.getlines(str(path))
    loader = _LabelLoader('source_identity', str(path))
    scope, captured = execute(loader)

    assert len(captured) == 1
    node = ast.parse(captured[0]).body[0]
    assert node.name == 'kernel'
    assert scope['kernel'](2) == 3
    assert scope['kernel'].__code__.co_filename != str(path)
    assert path.read_text() == original
    assert linecache.getlines(str(path)) == original_lines
    linecache.checkcache()
    assert inspect.getsource(scope['kernel']) == captured[0]


def test_source_versions_keep_independent_immutable_inspection(tmp_path, monkeypatch):
    from vllm_kvmem import logprob_patch
    monkeypatch.setattr(logprob_patch, 'patch_rejection_source', rewrite)
    path = tmp_path / 'sampler.py'
    path.write_text('# old source\n@jit\ndef kernel():\n    return 1\n')
    loader = _LabelLoader('source_identity_versions', str(path))
    old, old_source = execute(loader)
    repeated, _ = execute(loader)
    assert old['kernel'].__code__.co_filename == repeated['kernel'].__code__.co_filename

    path.write_text('# new source\n@jit\ndef kernel():\n    return 2\n')
    new, new_source = execute(loader)
    assert old['kernel'].__code__.co_filename != new['kernel'].__code__.co_filename
    linecache.checkcache()
    assert inspect.getsource(old['kernel']) == old_source[0]
    assert inspect.getsource(new['kernel']) == new_source[0]
    assert old['kernel']() == 1
    assert new['kernel']() == 2


def test_native_support_passthrough_keeps_native_filename(tmp_path, monkeypatch):
    from vllm_kvmem import logprob_patch
    monkeypatch.setattr(logprob_patch, 'patch_rejection_source', bytes.decode)
    path = tmp_path / 'sampler.py'
    path.write_text('@jit\ndef kernel():\n    return 1\n')
    loader = _LabelLoader('source_identity_passthrough', str(path))
    scope, captured = execute(loader)
    assert scope['kernel'].__code__.co_filename == str(path)
    assert captured == [path.read_text()]


def test_real_sampler_compiled_function_sources_match_ast_without_import():
    """Check every native module-level function without importing torch/vLLM."""
    from vllm_kvmem.logprob_patch import patch_rejection_source
    path = native_source()
    original = path.read_bytes()
    expected = {n.name: n for n in ast.parse(patch_rejection_source(original)).body
                if isinstance(n, ast.FunctionDef)}
    loader = _LabelLoader('native_source_identity', str(path))
    code = loader.get_code(loader.name)
    count = 0
    for value in code.co_consts:
        if not isinstance(value, types.CodeType) or value.co_name not in expected:
            continue
        fn = types.FunctionType(value, {'__name__': loader.name})
        actual = ast.parse(inspect.getsource(fn)).body[0]
        assert ast.dump(actual) == ast.dump(expected[value.co_name])
        count += 1
    assert count == len(expected)
    assert '_flatten_sampled_kernel' in expected
    assert path.read_bytes() == original


def test_real_triton_sampler_import_preserves_all_jit_source_identities():
    """Import real CPU-side JIT objects; never initialize CUDA or launch kernels."""
    try:
        root = engine_root()
    except PackageNotFoundError:
        pytest.skip('Native import test requires installed vLLM')
    path = root / 'v1/worker/gpu/spec_decode/rejection_sampler.py'
    before = path.read_bytes()
    script = r'''
import ast
import inspect
import json
import linecache
import torch
from vllm_kvmem.compat import enable_label_logprobs
from vllm_kvmem.engine_api import engine_root
from vllm_kvmem.logprob_patch import patch_rejection_source
path = engine_root() / 'v1/worker/gpu/spec_decode/rejection_sampler.py'
original = path.read_bytes()
adapted = ast.parse(patch_rejection_source(original))
expected = {node.name: node for node in adapted.body
            if isinstance(node, ast.FunctionDef) and
            any(ast.unparse(d) == 'triton.jit' for d in node.decorator_list)}
enable_label_logprobs()
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as module
from triton.runtime.jit import JITFunction
local = []
all_jit = []
for name, value in vars(module).items():
    if not isinstance(value, JITFunction):
        continue
    linecache.checkcache()
    actual = ast.parse(inspect.getsource(value.fn)).body[0]
    assert actual.name == value.fn.__name__, (name, actual.name)
    jit_ast = ast.parse(value.src).body[0]
    assert jit_ast.name == value.fn.__name__, (name, jit_ast.name)
    all_jit.append(name)
    if value.fn.__module__ == module.__name__:
        assert ast.dump(actual) == ast.dump(expected[name])
        no_decorators = expected[name]
        no_decorators.decorator_list = []
        assert ast.dump(jit_ast) == ast.dump(no_decorators)
        local.append(name)
assert set(local) == set(expected)
assert local
assert path.read_bytes() == original
assert not torch.cuda.is_initialized()
print(json.dumps({'native_jit': local, 'all_jit': all_jit, 'cuda_initialized': False}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    repo = Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = os.pathsep.join([str(repo / 'src'), env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert '_flatten_sampled_kernel' in report['native_jit']
    assert report['cuda_initialized'] is False
    assert path.read_bytes() == before
