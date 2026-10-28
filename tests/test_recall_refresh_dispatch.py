"""Exercise live refresh dispatch without launching CUDA or a model."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.async_refresh import AsyncRefresh
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.pressure_query_capture import PressureQueryWindow
from vllm_kvmem import recall_experiment
from vllm_kvmem import prefill_attention
from vllm_kvmem.window_assembly import NativeWindowAssembler


def setup_dispatch(monkeypatch, tmp_path, *, selection='pressure', stage='prefill',
                   history=512, budget=256, method='fps4'):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', method)
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', '128')
    monkeypatch.delenv('KVMEM_RETRIEVAL_DIAGNOSTICS', raising=False)
    # Mirrors the existing AsyncRefresh tests: model/device work is replaced by
    # explicit ready/assembly sentinels, while the actual dispatch runs intact.
    monkeypatch.setattr(torch.cuda, 'device', lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *_: nullcontext())
    calls = NS(reads=[], selections=[], noops=[], waits=[], publication=[], assembled=[], legacy=[])

    def legacy(*args, **kwargs):
        calls.legacy.append((args, kwargs))
        pytest.fail('Experiment dispatch reached a legacy scorer')

    monkeypatch.setattr(prefill_attention, 'select_pressure', legacy)
    plan, no_op_plan, completed = object(), object(), object()
    config = KVMemConfig(block_tokens=32, select_budget=budget, gen_budget=128,
                        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    records = {}
    for index in range((history+31)//32):
        handle = NS(slot=index+100, generation=7)
        records[(handle.slot, handle.generation)] = NS(handle=handle, history_begin=index*32,
            count=min(32, history-index*32))
    memory = NS(select=lambda scores, **kwargs: calls.noops.append((scores, kwargs)) or no_op_plan)
    window = PressureQueryWindow()
    tail = torch.arange(min(128, history), dtype=torch.float32).view(1, -1, 1, 1).expand(2, -1, 4, 3)
    window.accept(0, history, tail, row_offset=max(0, history-128), prefill=True)
    original_read = window.read

    def read(end):
        calls.reads.append(end)
        return original_read(end)

    window.read = read
    authority = NS(history_end=history, records=records, memory=memory,
        archive=NS(wait_for_capacity=lambda _: True), pressure_query_window=window,
        publish_sealed=lambda waiter: calls.publication.append(True), select_for_queries=legacy)
    query_span = (history+3, history+9) if selection == 'query' else None
    command = NS(request_id='functional-owner', selection=selection, pressure_stage=stage,
        groups=((0, NS(window_tokens=budget)),), mandatory_spans=((64, 96, 'explicit'),),
        query_span=query_span, input_tokens=tuple(range(16)))
    job = NS(command=command, history=history, authority=authority, graph_binding=None,
             guided_pressure=False, guided_rows=[])
    manager = AsyncRefresh.__new__(AsyncRefresh)
    manager.runner = NS(device='mock-cuda', _kvmem_probe_output=tmp_path,
        _kvmem_capture_runtime=NS(names=('layer0', 'layer1')))
    manager.options = NS(memory=config, pressure_policy='attention', action_profiling=False,
        packed_assembly=False, direct_pinned_reads=False)
    manager.stream = NS(wait_event=lambda ready: calls.waits.append(ready))
    manager.maintenance = None
    manager.emit = lambda *args, **kwargs: None
    assembler = NS(assemble=lambda owner, actual: calls.assembled.append((owner, actual)) or completed)
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *args, **kwargs: assembler)

    def select(owner, queries, mandatory, **kwargs):
        calls.selections.append(dict(authority=owner, queries=queries,
                                     mandatory=list(mandatory), **kwargs))
        return plan

    monkeypatch.setattr(recall_experiment, 'select_for_event', select)
    ready = object()
    public_q = torch.ones(2, 6, 4, 3)
    return NS(manager=manager, job=job, calls=calls, plan=plan, no_op_plan=no_op_plan,
        completed=completed, assembler=assembler, public_q=public_q, tail=tail,
        ready=ready, authority=authority)


@pytest.mark.parametrize('method', recall_experiment.METHODS)
@pytest.mark.parametrize('stage', ['prefill', 'decode'])
def test_pressure_dispatch_uses_accepted_tail_for_every_method(monkeypatch, tmp_path, method, stage):
    setup = setup_dispatch(monkeypatch, tmp_path, method=method, stage=stage)
    result = setup.manager._assemble(setup.job, setup.public_q, setup.ready)
    calls = setup.calls
    assert calls.reads == [512] and not calls.noops and not calls.legacy
    assert calls.waits == [setup.ready] and calls.publication == [True]
    selected, = calls.selections
    assert selected['authority'] is setup.authority
    assert selected['event_type'] == stage+'_pressure' and selected['max_tokens'] == 256
    assert selected['request_id'] == 'functional-owner' and selected['output'] == tmp_path
    torch.testing.assert_close(selected['queries'], setup.tail.to(torch.bfloat16))
    assert selected['queries'] is not setup.public_q
    metadata = selected['query_metadata']
    assert metadata['query_source'] == 'accepted_public_tail'
    assert metadata['query_begin'] == 384 and metadata['query_end'] == metadata['accepted_through'] == 512
    assert metadata['query_rows'] == 128
    assert {handle.slot for handle in selected['mandatory']} >= {100, 102}
    assert calls.assembled == [(setup.authority, setup.plan)]
    assert result[:3] == (setup.assembler, setup.completed, setup.plan)


@pytest.mark.parametrize('method', recall_experiment.METHODS)
def test_query_dispatch_preserves_private_public_input_q_and_boundaries(monkeypatch, tmp_path, method):
    setup = setup_dispatch(monkeypatch, tmp_path, method=method, selection='query')
    setup.manager._assemble(setup.job, setup.public_q, setup.ready)
    selected, = setup.calls.selections
    assert not setup.calls.reads and not setup.calls.legacy and not setup.calls.noops
    assert selected['queries'] is setup.public_q and selected['event_type'] == 'query'
    metadata = selected['query_metadata']
    assert metadata == dict(query_source='public_query', query_begin=515, query_end=521,
                           query_processed_through=521, public_input_end=528)
    # The private query was processed against saved state; accepted KV still
    # ends at 512 until the subsequent ordinary public replay commits it.
    assert setup.authority.history_end == 512 < metadata['query_end']
    assert metadata['query_end'] <= metadata['query_processed_through'] <= metadata['public_input_end']
    assert selected['max_tokens'] == 256


@pytest.mark.parametrize('guided', ['pressure', 'rows'])
def test_guided_inputs_fail_instead_of_falling_back(monkeypatch, tmp_path, guided):
    setup = setup_dispatch(monkeypatch, tmp_path)
    if guided == 'pressure':
        setup.job.guided_pressure = True
    else:
        setup.job.guided_rows = [setup.public_q]
    with pytest.raises(ValueError, match='prohibits guided'):
        setup.manager._assemble(setup.job, setup.public_q, setup.ready)
    assert not setup.calls.reads and not setup.calls.selections and not setup.calls.legacy
    assert not setup.calls.assembled


@pytest.mark.parametrize('stage', ['', 'tool', None])
def test_pressure_without_original_stage_cannot_be_mislabelled(monkeypatch, tmp_path, stage):
    setup = setup_dispatch(monkeypatch, tmp_path, stage=stage)
    with pytest.raises(ValueError, match='original execution stage'):
        setup.manager._assemble(setup.job, None, setup.ready)
    assert not setup.calls.reads and not setup.calls.selections and not setup.calls.legacy


def test_missing_or_stale_pressure_q_never_uses_accumulated_attention(monkeypatch, tmp_path):
    setup = setup_dispatch(monkeypatch, tmp_path)
    window = setup.authority.pressure_query_window
    setup.authority.pressure_query_window = None
    with pytest.raises(ValueError, match='captured accepted Q'):
        setup.manager._assemble(setup.job, None, setup.ready)
    setup.authority.pressure_query_window = window
    window.rewind(480)
    with pytest.raises(RuntimeError, match='accepted boundary'):
        setup.manager._assemble(setup.job, None, setup.ready)
    assert not setup.calls.selections and not setup.calls.legacy and not setup.calls.assembled


@pytest.mark.parametrize('selection', ['chunk', 'query'])
def test_unqualified_selection_entry_is_rejected(monkeypatch, tmp_path, selection):
    setup = setup_dispatch(monkeypatch, tmp_path, selection=selection)
    setup.job.command.query_span = None
    with pytest.raises(ValueError, match='unqualified selection entry'):
        setup.manager._assemble(setup.job, setup.public_q, setup.ready)
    assert not setup.calls.selections and not setup.calls.legacy


@pytest.mark.parametrize('selection,stage', [('query', ''), ('pressure', 'prefill'), ('pressure', 'decode')])
@pytest.mark.parametrize('history', [128, 256])
def test_fitting_history_is_noop_for_every_entry_without_legacy_fallback(monkeypatch, tmp_path, selection, stage, history):
    setup = setup_dispatch(monkeypatch, tmp_path, selection=selection, stage=stage, history=history)
    setup.authority.recall_selection_report = {'actual_method': 'stale_method'}
    result = setup.manager._assemble(setup.job, setup.public_q, setup.ready)
    assert setup.calls.noops == [({}, {'action': False, 'max_tokens': 256})]
    assert not setup.calls.reads and not setup.calls.selections
    assert not setup.calls.legacy
    assert setup.authority.recall_selection_report is None
    assert setup.calls.assembled == [(setup.authority, setup.no_op_plan)]
    assert result[2] is setup.no_op_plan
