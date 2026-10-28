import ast
import os
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from vllm_kvmem.engine_api import (
    CONTRACTS, UnsupportedEngine, check_contract, check_source, engine_root,
)


def source_root():
    if source := os.getenv('KVMEM_TEST_VLLM_SOURCE'):
        return Path(source)
    try:
        return engine_root()
    except PackageNotFoundError:
        pytest.skip('Install vLLM or set KVMEM_TEST_VLLM_SOURCE to an upstream vllm directory')


@pytest.mark.parametrize('feature', tuple(CONTRACTS))
def test_upstream_capabilities(feature):
    root = source_root()
    if feature == 'systemone' and not (root / CONTRACTS[feature][1][0]).exists():
        if os.getenv('KVMEM_TEST_REQUIRE_SYSTEMONE') == '1':
            pytest.fail('Native SystemOne API is required')
        pytest.skip('This baseline predates native SystemOne')
    check_contract(feature, root)


def test_contracts_ignore_versions_comments_and_unrelated_methods():
    source = 'class Engine:\n    def generate(self, prompt, *, session_id=None): pass\n'
    check_source(source + '\n# changed upstream\n__version__ = "99.0.dev1"\n',
                 'Engine.generate', ('prompt', 'session_id'))
    with pytest.raises(UnsupportedEngine, match='session_id'):
        check_source(source.replace('session_id', 'renamed'), 'Engine.generate', ('session_id',))
    with pytest.raises(UnsupportedEngine, match='missing'):
        check_source(source, 'Engine.abort')


def test_compatibility_adapters_compile_against_upstream():
    from vllm_kvmem.compat import patched_source
    from vllm_kvmem.metadata_patch import patch_metadata_source
    from vllm_kvmem.logprob_patch import patch_rejection_source
    root = source_root()
    for relative, patch in (
        ('v1/worker/gpu/spec_decode/dflash/utils.py', patched_source),
        ('v1/worker/gpu/attn_utils.py', patch_metadata_source),
        ('v1/worker/gpu/spec_decode/rejection_sampler.py', patch_rejection_source),
    ):
        source = (root / relative).read_bytes()
        compile(patch(source), str(root / relative), 'exec')
        compile(patch(source + b'\n# upstream documentation update'), relative, 'exec')


def test_native_loader_fix_and_logprob_support_are_preserved():
    from vllm_kvmem.compat import patched_source
    from vllm_kvmem.logprob_patch import patch_rejection_source
    source = b'''def load_dflash_model(target_model, vllm_config):
    return replace(vllm_config,
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)))
'''
    assert patched_source(source) == source.decode()
    source = (source_root() / 'v1/worker/gpu/spec_decode/rejection_sampler.py').read_bytes()
    patched = patch_rejection_source(source)
    assert patch_rejection_source(patched.encode()) == patched


def test_metadata_patch_rejects_missing_semantic_hook():
    from vllm_kvmem.metadata_patch import patch_metadata_source
    source = (source_root() / 'v1/worker/gpu/attn_utils.py').read_bytes()
    with pytest.raises(ImportError, match='metadata patch'):
        patch_metadata_source(source.replace(b'seq_lens=seq_lens,', b'seq_lens=other,'))


def test_state_contract_rejects_changed_initial_acceptance_counter(tmp_path):
    from vllm_kvmem.engine_api import INVARIANTS
    root = source_root()
    files = {item[0] for item in (*CONTRACTS['state'], *INVARIANTS['state'])}
    for relative in files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = (root / relative).read_text()
        path.write_text(source.replace('self.num_accepted_tokens_gpu[req_index].fill_(1)',
                                       'self.num_accepted_tokens_gpu[req_index].fill_(2)'))
    with pytest.raises(UnsupportedEngine, match='ownership contract changed'):
        check_contract('state', tmp_path)


def test_speculative_label_ids_follow_the_correct_request_rows():
    """Execute upstream extraction with CPU stand-ins for the two GPU kernels."""
    from types import SimpleNamespace
    torch = pytest.importorskip('torch')
    from vllm_kvmem.logprob_patch import patch_rejection_source
    from vllm_kvmem.engine_api import find_symbol
    source = (source_root() / 'v1/worker/gpu/spec_decode/rejection_sampler.py').read_bytes()
    patched = patch_rejection_source(source)
    method = find_symbol(patched, 'RejectionSampler._get_logprobs_tensors')
    method.returns = None
    for arg in (*method.args.args, *method.args.kwonlyargs):
        arg.annotation = None
    # Local annotations are not evaluated; preserve the native extraction body.
    tree = ast.Module(body=[method], type_ignores=[])
    seen = []
    class Flatten:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None
    def scores(*args, **kwargs):
        seen.append((args, kwargs))
        return 'labels'
    scope = dict(torch=torch, NO_LOGPROBS=-2, compute_topk_scores=scores,
                 _flatten_sampled_kernel=Flatten())
    exec(compile(ast.fix_missing_locations(tree), 'native_extraction', 'exec'), scope)
    state = object()
    sampler = SimpleNamespace(logprobs_mode='raw_logprobs', logprob_token_ids_state=state)
    owner = SimpleNamespace(sampler=sampler, enable_adaptive_verification=False)
    mapping = torch.tensor([9, 4, 4])  # a label read mixed with a draft verification
    result = scope[method.name](owner, torch.zeros((2, 2), dtype=torch.int64),
        torch.ones(2, dtype=torch.int32), torch.zeros((3, 12)), torch.tensor([0, 1, 3]),
        torch.tensor([0, 1, 3]).numpy(), 0,
        expanded_idx_mapping=mapping, max_per_req_token_ids=2)
    assert result == 'labels'
    args, kwargs = seen[0]
    assert args[1] == 0  # label IDs need no top-k request or full-vocabulary output
    assert kwargs['logprob_token_ids_state'] is state
    assert kwargs['expanded_idx_mapping'] is mapping
    assert kwargs['max_per_req_token_ids'] == 2
