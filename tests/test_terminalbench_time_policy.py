import asyncio
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
import tomllib

import pytest

from scripts.terminalbench_policy import configure_task_deadlines, disable_agent_deadline


TASK = b'''version = "1.0"
[agent]
timeout_sec = 0.01 # short deadline for regression test
user = "root"
[verifier]
timeout_sec = 7.0
[environment]
build_timeout_sec = 9.0
cpus = 2
memory_mb = 2048
'''


def test_only_agent_timeout_bytes_change():
    changed, audit = disable_agent_deadline(TASK)
    assert changed == TASK.replace(b'timeout_sec = 0.01 # short deadline for regression test\n', b'')
    assert audit['original_agent_timeout_sec'] == .01
    assert audit['effective_agent_timeout_sec'] is None
    assert audit['original_sha256'] != audit['effective_sha256']
    assert disable_agent_deadline(changed)[0] == changed


@pytest.mark.parametrize('mode', ['none', 'official'])
def test_task_copy_policy_retains_other_phases(tmp_path, mode):
    task = tmp_path/'example/task.toml'
    task.parent.mkdir()
    task.write_bytes(TASK)
    policy = configure_task_deadlines(tmp_path, mode)
    actual = tomllib.loads(task.read_text())
    expected = tomllib.loads(TASK.decode())
    if mode == 'none':
        expected['agent'].pop('timeout_sec')
    assert actual == expected
    assert policy['official_timing_comparable'] == (mode == 'official')
    assert policy['tasks']['example']['verifier_unchanged']


@pytest.mark.parametrize('data', [
    b'[agent]\n"timeout_sec" = 1\n',
    b'agent = {timeout_sec = 1}\n',
    TASK + b'[[steps]]\nname = "extra"\n',
])
def test_unrecognized_or_multistep_layout_fails_closed(data):
    with pytest.raises(ValueError):
        disable_agent_deadline(data)


def test_bad_policy_fails_before_mutation(tmp_path):
    task = tmp_path/'example/task.toml'
    task.parent.mkdir()
    task.write_bytes(TASK)
    with pytest.raises(ValueError):
        configure_task_deadlines(tmp_path, 'unexpected')
    assert task.read_bytes() == TASK


def harbor_trial(data, tmp_path):
    pytest.importorskip('harbor')
    from harbor.models.task.config import TaskConfig
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.single_step import SingleStepTrial
    trial = object.__new__(SingleStepTrial)
    trial.task = SimpleNamespace(config=TaskConfig.model_validate(tomllib.loads(data.decode())))
    trial.config = TrialConfig.model_validate(dict(task=dict(path=str(tmp_path)),
        agent=dict(name='terminus-2', override_setup_timeout_sec=11)))
    return trial


def test_pinned_harbor_resolves_true_none_and_preserves_other_timeouts(tmp_path):
    original = harbor_trial(TASK, tmp_path)
    unlimited = harbor_trial(disable_agent_deadline(TASK)[0], tmp_path)
    assert original._compute_agent_timeout_sec() == .01
    assert unlimited._compute_agent_timeout_sec() is None
    for method in ('_compute_verifier_timeout_sec', '_compute_agent_setup_timeout_sec',
                   '_compute_environment_build_timeout_sec'):
        assert getattr(unlimited, method)() == getattr(original, method)()


@pytest.mark.parametrize('mode', ['official', 'none', 'cancel'])
def test_real_harbor_agent_phase_deadline_and_cancellation(tmp_path, monkeypatch, mode):
    pytest.importorskip('harbor')
    import harbor.trial.sync_trajectory as sync_module
    from harbor.trial.errors import AgentTimeoutError
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield

    async def check():
        data = TASK if mode == 'official' else disable_agent_deadline(TASK)[0]
        trial = harbor_trial(data, tmp_path)
        entered = asyncio.Event()
        finished = []
        events = []

        async def work(**kwargs):
            entered.set()
            await asyncio.sleep(0.06)
            finished.append(True)

        async def emit(event):
            events.append(event)

        trial.agent = SimpleNamespace(extra_env={}, run=work)
        trial.user_agent = None
        trial.agent_environment = SimpleNamespace(with_default_user=lambda _: nullcontext(),
            scoped_exec_env=lambda _: nullcontext())
        trial._emit = emit
        trial._network_plan = lambda _: SimpleNamespace(agent_env_baseline=None, agent_phase=None)
        trial._phase_network_policy = scope
        trial._log_context = lambda *args: nullcontext()
        trial.paths = SimpleNamespace(agent_dir=tmp_path)
        monkeypatch.setattr(sync_module, 'sync_trajectory', scope)
        target = SimpleNamespace()
        task = asyncio.create_task(Trial._run_agent_phase(trial, target=target,
            instruction='original task', timeout_sec=trial._compute_agent_timeout_sec(), user=None))
        if mode == 'official':
            with pytest.raises(AgentTimeoutError):
                await task
            assert not finished
        elif mode == 'cancel':
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not finished
        else:
            await task
            assert finished == [True]
        assert events == [TrialEvent.AGENT_START, TrialEvent.AGENT_END]
        assert target.agent_execution.finished_at is not None

    asyncio.run(asyncio.wait_for(check(), 2))
