from importlib import metadata
from pathlib import Path
import ast

import pytest

from vllm_kvmem.compat import OLD, NEW, patched_source


def test_loader_patch_changes_only_load_config_and_accepts_unrelated_edits():
    try:
        root = Path(metadata.distribution("vllm").locate_file("vllm"))
    except metadata.PackageNotFoundError:
        pytest.skip("vLLM optional dependency is not installed")
    source = (root / "v1/worker/gpu/spec_decode/dflash/utils.py").read_bytes()
    patched = patched_source(source)
    ast.parse(patched)
    assert patched == source.decode() or patched.replace(NEW, OLD) == source.decode()
    assert patched_source(source + b'\n# upstream comment') == patched + '\n# upstream comment'
    with pytest.raises(ImportError, match="Unrecognized"):
        patched_source(source.replace(b'load_config=', b'unknown_config='))
