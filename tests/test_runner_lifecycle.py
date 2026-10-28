from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


@pytest.mark.parametrize('failed', [False, True])
def test_runner_shutdown_drains_retirement_before_releasing_native_gpu_state(failed):
    from vllm_kvmem.runner import make_probe_runner
    calls = []
    class Base:
        def shutdown(self):
            calls.append('native')
    def retire():
        calls.append('retire')
        if failed:
            raise OSError('retirement failure')
    runner = make_probe_runner(Base)()
    runner._kvmem_capture_runtime = SimpleNamespace(shutdown_retirement=retire)
    if failed:
        with pytest.raises(OSError, match='retirement'):
            runner.shutdown()
    else:
        runner.shutdown()
    assert calls == ['retire', 'native']


def test_unarmed_native_warmup_does_not_access_temporarily_disabled_connector(monkeypatch, tmp_path):
    from vllm_kvmem.runner import make_probe_runner
    import vllm.distributed.kv_transfer as transfer

    native_calls = []
    class Base:
        def execute_model(self, *args, **kwargs):
            native_calls.append(kwargs)
            return "native"

    runner = make_probe_runner(Base)()
    runner._kvmem_bounded_target = True
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_query_refresh = lambda command: command
    batch = SimpleNamespace(total_num_scheduled_tokens=1, finished_req_ids=set(),
                            kv_connector_metadata=None)

    def disabled():
        raise AssertionError("Native warmup disables the global connector agent")

    monkeypatch.setattr(transfer, "get_kv_transfer_group", disabled)
    assert runner.execute_model(batch, randomize_inputs=True) == "native"
    assert native_calls[-1]['randomize_inputs'] is True
    connector = SimpleNamespace(runtime=None)
    monkeypatch.setattr(transfer, "get_kv_transfer_group", lambda: connector)
    (tmp_path / "armed").touch()
    assert runner.execute_model(batch) == "native"
    assert connector.runtime is runner._kvmem_query_refresh


def test_startup_reservation_closes_even_without_a_first_request():
    from vllm_kvmem.runner import make_probe_runner
    closed = []
    class Base:
        def shutdown(self): closed.append('native')
    runner = make_probe_runner(Base)()
    runner._kvmem_prepared_archive = SimpleNamespace(close=lambda: closed.append('archive') or True)
    runner.shutdown()
    assert closed == ['archive', 'native']


def test_prompt_session_controls_start_after_native_warmup(tmp_path):
    from vllm_kvmem.runner import make_probe_runner
    from vllm_kvmem.connector import ControlMetadata
    from vllm_kvmem.session_control import SessionCommands

    class Base:
        def execute_model(self, *args, **kwargs):
            return 'native'

    received = []
    runner = make_probe_runner(Base)()
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_sessions = SimpleNamespace(receive=received.append)
    batch = SimpleNamespace(total_num_scheduled_tokens=1, finished_req_ids=set(),
                            kv_connector_metadata=None)
    assert runner.execute_model(batch) == 'native'
    assert not received
    (tmp_path / 'armed').touch()
    with pytest.raises(RuntimeError, match='connector metadata'):
        runner.execute_model(batch)
    batch.kv_connector_metadata = ControlMetadata()
    assert runner.execute_model(batch) == 'native'
    assert len(received) == 1 and isinstance(received[0], SessionCommands)


def test_serving_arms_at_first_native_metadata_before_the_forward(tmp_path):
    from vllm_kvmem.runner import make_probe_runner
    from vllm_kvmem.connector import ControlMetadata

    armed_at_forward = []
    received = []
    class Base:
        def execute_model(self, *args, **kwargs):
            armed_at_forward.append((tmp_path/'armed').is_file())
            return 'native'

    runner = make_probe_runner(Base)()
    runner._kvmem_serving = True
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_sessions = SimpleNamespace(receive=received.append)
    batch = SimpleNamespace(total_num_scheduled_tokens=1, finished_req_ids=set(), kv_connector_metadata=None)
    assert runner.execute_model(batch) == 'native'
    assert armed_at_forward == [False] and not received
    batch.kv_connector_metadata = ControlMetadata()
    assert runner.execute_model(batch) == 'native'
    assert armed_at_forward == [False, True] and len(received) == 1


def test_terminal_clipping_precedes_both_deferrals_and_only_ready_owners_retire(monkeypatch, tmp_path):
    from dataclasses import replace
    from vllm_kvmem.runner import make_probe_runner
    from vllm_kvmem.connector import ControlMetadata
    import vllm.distributed.kv_transfer as transfer

    calls = []
    original_prefix, resumed_prefix = object(), object()
    original = SimpleNamespace(total_num_scheduled_tokens=1,
        num_scheduled_tokens={'a': 1}, finished_req_ids={'b'},
        kv_connector_metadata=ControlMetadata(terminal_prefixes={'b': original_prefix}))

    def changed(output, finished, prefixes):
        return SimpleNamespace(**{**vars(output), 'finished_req_ids': finished,
            'kv_connector_metadata': replace(output.kv_connector_metadata, terminal_prefixes=prefixes)})

    def confirm(finished, prefixes):
        assert finished == {'b'} and prefixes == {'b': original_prefix}
        calls.append('confirm-original')

    def refresh_defer(output):
        assert output is original and calls == ['confirm-original']
        calls.append('refresh-defer')
        return changed(output, set(), {})

    def capture_defer(output):
        assert not output.finished_req_ids
        calls.append('capture-defer')
        # A previously deferred owner is now ready; b must stay allocated.
        return changed(output, {'old'}, {'old': resumed_prefix})

    def finish(finished, prefixes):
        assert finished == {'old'} and prefixes == {'old': resumed_prefix}
        calls.append('finish-ready')

    class Base:
        def execute_model(self, output, **kwargs):
            assert output.num_scheduled_tokens == {'a': 1}
            assert output.finished_req_ids == {'old'}
            calls.append('native-forward')
            return 'native'

    runner = make_probe_runner(Base)()
    runner._kvmem_bounded_target = True
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_query_refresh = lambda command: command
    runner._kvmem_capture_runtime = SimpleNamespace(begin_step=confirm,
        defer_terminals=capture_defer, finish=finish,
        advance_publication=lambda *_: calls.append('advance'),
        retirement_done=lambda _: False, publication_blocked_requests=lambda: {'b'})
    runner._kvmem_async_refresh = SimpleNamespace(defer_terminals=refresh_defer, maintenance=None,
                                                 options=SimpleNamespace())
    runner._kvmem_sessions = SimpleNamespace(receive=lambda _: calls.append('sessions'))
    runner._kvmem_window_adapter = SimpleNamespace(free=lambda rid: calls.append(('free', rid)))
    connector = SimpleNamespace()
    monkeypatch.setattr(transfer, 'get_kv_transfer_group', lambda: connector)
    (tmp_path/'armed').touch()
    assert runner.execute_model(original) == 'native'
    assert calls == ['confirm-original', 'refresh-defer', 'capture-defer', 'sessions',
                     'finish-ready', 'advance', ('free', 'old'), 'native-forward']
    assert connector.publication_snapshot() == {'b'}
