"""Rolling tool-result retirement through accepted request history."""
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.core.config import KVMemConfig
from vllm_kvmem.core.historical_gc import HistoricalGcSettings
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.serving.capture_runtime import CaptureRuntime
from vllm_kvmem.serving.session_control import SessionBinding
from vllm_kvmem.serving.tool_result_history import ToolResultHistory

from test_history_lru_policy import Ready, advance, append, eventually, records_by_start


SPANS = ((0, 5, 'system'), (5, 23, 'tool'), (23, 32, 'assistant')) + tuple(
    (start, start + 4, 'tool') for start in range(32, 80, 4))


def binding(spans=SPANS, *, history=0, mandatory=((12, 13, 'explicit'),)):
    return SessionBinding(('salt', 'agent'), 1, 'r', (42,) * 80, history=history,
                          locality_spans=spans, mandatory_spans=mandatory)


def test_expiration_requires_thirteen_completed_tool_replies_and_exact_whole_pages():
    published = []
    history = ToolResultHistory(SimpleNamespace(set_expired_tool_ranges=published.append), 4, 12)
    history.bind(binding(), 0)
    history.advance(79)
    assert history.completed == 12 and published == []
    history.advance(80)
    assert history.completed == 13 and published == [((8, 12), (16, 20))]
    history.advance(81)  # Generation does not count as another tool reply.
    assert len(published) == 1

    history.rewind(76)
    assert history.completed == 12 and published[-1] == ()
    history.bind(binding(((76, 80, 'tool'),), history=76), 76)
    history.advance(80)
    # Supplied suffix metadata may inherit only the verified, accepted prefix.
    assert history.completed == 13 and published[-1] == ((8, 12), (16, 20))

    with pytest.raises(ValueError, match='Conflicting message boundaries'):
        history.bind(binding(((23, 32, 'tool'),), history=80), 80)
    assert published[-1] == ((8, 12), (16, 20))


def test_unknown_and_split_tool_boundaries_do_not_expire_a_page():
    published = []
    history = ToolResultHistory(SimpleNamespace(set_expired_tool_ranges=published.append), 4, 1)
    history.bind(binding((), mandatory=()), 80)
    history.advance(100)
    assert history.completed == 0
    history.bind(binding(((0, 2, 'tool'), (2, 4, 'tool'), (4, 8, 'tool')), mandatory=()), 8)
    assert history.completed == 3
    assert published == []  # Two adjacent tool replies share page zero.


def test_settings_require_explicit_tool_policy_and_mandatory_protection():
    assert HistoricalGcSettings().policy == 'lru'
    settings = HistoricalGcSettings(policy='tool_result_age')
    assert settings.policy_params == {'keep_mandatory': True, 'tool_result_age_steps': 12, 'lru_fallback': False}
    assert HistoricalGcSettings(policy='tool_result_age',
                                policy_params={'lru_fallback': True}).policy_params['lru_fallback'] is True
    for params in ({'tool_result_age_steps': 0}, {'tool_result_age_steps': True},
                   {'tool_result_age_steps': 1_000_001}, {'keep_mandatory': False},
                   {'thinking_age_steps': 1}, {'lru_fallback': 1}):
        with pytest.raises(ValueError):
            HistoricalGcSettings(policy='tool_result_age', policy_params=params)


def select_window(authority, preferred):
    scores = {record.handle.slot: float(position in preferred)
              for position, record in records_by_start(authority).items()}
    authority.memory.select(scores, max_tokens=16)
    authority.memory.attach_completion(Ready())
    assert authority.memory.commit_selection()
    return {row[2] for row in authority.memory._store.working_records()}


@pytest.mark.parametrize('mode', ['delete', 'shadow', 'off'])
def test_serving_authority_retires_only_expired_tool_pages_and_retries_holds(tmp_path, mode):
    config = KVMemConfig(block_tokens=4, select_budget=96, gen_budget=96,
                         repository_blocks=64, sink_blocks=1, recent_blocks=0)
    settings = HistoricalGcSettings(mode=mode, policy='tool_result_age')
    native = settings.native(config, 1)
    native.background = False
    archive = MemoryArchive(str(tmp_path), 48, 64, native_payloads=True)
    worker = archive.enable_history_gc(native)
    runtime = SimpleNamespace(config=config, runner=SimpleNamespace(_kvmem_layers=('a', 'b')),
        options=SimpleNamespace(historical_gc=settings, query_score_mode='mean_k',
                                mean_k_backend='torch_reference', query_history=False),
        archive=archive, layers=(None, None), heads=1, dim=2, raw_dtype=torch.bfloat16,
        resident_records=32, gc_config=None, gc_progress=None)
    authority = CaptureRuntime.build_authority(runtime, 'r', binding())
    scope = authority.gc.scope
    hold = None
    try:
        append(authority, 76, 0)
        assert select_window(authority, {0, 8, 24, 72}) == {0, 8, 24, 72}
        handles = {position: r.handle for position, r in records_by_start(authority).items()}
        hold = scope.protect([handles[8]])
        assert hold is not None
        append(authority, 3, 16)
        advance(authority, worker)
        assert scope.stats['decisions'] == 0
        assert authority.memory.repository.ready_slots > native.ready_high

        append(authority, 1, 19)
        advance(authority, worker)
        stats = scope.stats
        if mode != 'delete':
            assert stats['retired'] == 0
            assert stats['shadow_decisions'] == (1 if mode == 'shadow' else 0)
            assert set(records_by_start(authority)) == set(range(0, 80, 4))
            return

        assert stats['retired'] == stats['tool_result_age_retired'] == 1
        assert set(records_by_start(authority)) == set(range(0, 80, 4)) - {16}
        assert select_window(authority, {0, 24, 72, 76}) == {0, 24, 72, 76}
        advance(authority, worker)
        assert 8 in records_by_start(authority)  # Exact saved frame still owns it.
        hold = None
        advance(authority, worker)
        assert scope.stats['tool_result_age_retired'] == 2
        assert set(records_by_start(authority)) == set(range(0, 80, 4)) - {8, 16}
    finally:
        hold = None
        eventually(lambda: (worker.advance(), authority.close())[1])
        eventually(lambda: (worker.advance(), archive.close())[1])
