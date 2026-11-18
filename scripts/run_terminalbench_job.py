"""Run the pinned, unmodified Harbor Terminus agent with local lifecycle hooks.

Hooks account resources, persist evidence, release finished KV owners, and may
stage a qualified public package cache before the original verifier timer.
They never change a prompt, verifier, reward, or agent response. The frozen
task copy may use audited relaxed or disabled Agent deadlines. Relaxed runs
exclude explicitly journaled pause intervals from the Agent timer.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib

import httpx
import psutil
try:
    from .terminalbench_network import reclaim_trial_networks, assert_direct_docker, container_env
except ImportError:
    from terminalbench_network import reclaim_trial_networks, assert_direct_docker, container_env


GIB = 1024**3
DEFAULT_CONTAINER_MEMORY_BYTES = 48 * GIB
HOST_MEMORY_RESERVE_BYTES = 12 * GIB


def validate_memory_budget(value, *, total=None):
    if type(value) is not int or value <= 0:
        raise ValueError('Container memory budget must be a positive integer number of bytes')
    if total is not None and value > total-HOST_MEMORY_RESERVE_BYTES:
        raise ValueError('Container memory budget must leave the 12-GiB host reserve')
    return value


class MemoryAdmission:
    """Change only task admission; each official container limit is untouched."""
    def __init__(self, root, manifest, log):
        self.path = root/'memory-admission-control.json'
        self.total = psutil.virtual_memory().total
        self.base = validate_memory_budget(manifest.get('container_memory_admission_bytes',
            DEFAULT_CONTAINER_MEMORY_BYTES), total=self.total)
        self.budget = self.base
        self.log, self.bad_digest, self.history = log, None, []
        self._record('manifest', None)

    def _record(self, source, previous):
        row = dict(event='memory_admission_policy', source=source, previous_budget_bytes=previous,
                   budget_bytes=self.budget, host_reserve_bytes=HOST_MEMORY_RESERVE_BYTES)
        self.history.append(dict(at=time.time(), **row))
        self.log(row)

    def refresh(self):
        raw = b''
        try:
            try:
                with self.path.open('rb') as stream:
                    raw = stream.read(4097)
            except FileNotFoundError:
                value = self.base
            else:
                if len(raw)>4096:
                    raise ValueError('Memory admission control exceeds 4096 bytes')
                control = json.loads(raw)
                if (not isinstance(control,dict) or set(control)!={'schema_version','budget_bytes'} or
                        type(control['schema_version']) is not int or control['schema_version']!=1):
                    raise ValueError('Invalid memory admission control schema')
                value = validate_memory_budget(control['budget_bytes'],total=self.total)
        except (OSError,ValueError,TypeError) as error:
            # A partial/invalid control update must not cancel any Agent or
            # verifier. Retain the last validated policy and report it once.
            digest = hashlib.sha256(raw+str(error).encode()).hexdigest()
            if digest!=self.bad_digest:
                self.log(dict(event='memory_admission_control_rejected',error=str(error),
                              retained_budget_bytes=self.budget))
                self.bad_digest=digest
            return
        self.bad_digest=None
        if value!=self.budget:
            previous,self.budget=self.budget,value
            self._record('control' if raw else 'manifest',previous)

    def admits(self, reserved, requested, available):
        self.refresh()
        return reserved+requested<=self.budget and available>=requested+HOST_MEMORY_RESERVE_BYTES


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def classify(result):
    verifier = result.get('verifier_result') or {}
    rewards = verifier.get('rewards')
    error = (result.get('exception_info') or {}).get('exception_type')
    # A task timeout remains a benchmark outcome, never a hidden retry.
    return dict(task=result['task_name'], trial=result['trial_name'],
                rewards=rewards, exception=error,
                status='scored' if rewards is not None else
                'agent_timeout' if error == 'AgentTimeoutError' else 'error',
                passed=bool(rewards and rewards.get('reward') == 1))


async def run(root, config_path):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    from harbor.trial.hooks import TrialEvent
    config = JobConfig.model_validate_json(config_path.read_text())
    for key in list(os.environ):
        if 'proxy' in key.lower():
            del os.environ[key]
    download_policy = assert_direct_docker()
    expected = container_env(json.loads((root/'manifest.json').read_text()))
    if any(config.environment.env.get(key) != value for key, value in expected.items()):
        raise ValueError('Job container proxies must be explicitly disabled, or exactly the declared relay')
    job_dir = config.jobs_dir / config.job_name
    if job_dir.exists():
        raise ValueError('Refuse an attempted job directory; recovery requires a separate audit')
    if config.n_attempts != 1 or config.retry.max_retries != 0:
        raise ValueError('Exactly one attempt per task, no task retries')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('agent_time_limit') == 'relaxed' and not config_path.name.startswith('smoke'):
        try:
            from .terminalbench_deadline import install as install_deadline, validate_job_deadlines
        except ImportError:
            from terminalbench_deadline import install as install_deadline, validate_job_deadlines
        validate_job_deadlines(root, config)
        install_deadline(root, config.job_name)
    if manifest.get('length_finish_repair'):
        from terminalbench_length_finish import install as install_length_finish
        install_length_finish(root/(config.job_name+'-length-finish-repair.json'),
                              manifest['length_finish_repair']['upstream_sources'])
    if manifest.get('query_provenance'):
        from terminalbench_query import install as install_query
        sources = {str(root / path): value for path, value in manifest['query_provenance']['tasks'].items()}
        install_query(sources, root/(config.job_name+'-query-sources.jsonl'),
                      manifest['query_provenance']['upstream_sources'])
    if manifest.get('harness_repair'):
        from terminalbench_harness import install
        bundles = {task: root / relative for task, relative in manifest.get('terminal_tool_bundles', {}).items()}
        if any(not path.resolve().is_relative_to(root.resolve()) for path in bundles.values()):
            raise ValueError('Dependency bundle outside frozen experiment')
        install(root/(config.job_name+'-harness.jsonl'), manifest['harness_repair']['upstream_sources'], bundles,
                recorder_transport=manifest['harness_repair'].get('recorder_transport', False),
                tool_install_timeout_sec=manifest['harness_repair'].get('tool_install_timeout_sec'))
    job = await Job.create(config)
    expected = 1 if config_path.name.startswith('smoke') else len(manifest.get('launched_task_names', range(89)))
    if len(job) != expected:
        raise ValueError(f'Expected {expected} tasks, got {len(job)}')
    output = root / (config.job_name + '-events')
    output.mkdir(exist_ok=False)
    save(output/'direct-environment.json', download_policy)
    reservations = {}
    completed = {}
    released = set()
    release_errors = []
    network_errors = []
    verifier_cache_errors = []
    verifier_cache_installed = set()
    changed = asyncio.Condition()
    bridge = json.loads((root / 'connection.json').read_text())
    async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
        def log(value):
            with (output / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=time.time(), **value), ensure_ascii=False) + '\n')

        admission = MemoryAdmission(root, manifest, log)

        async def lifecycle(event):
            name = event.trial_name
            kind = event.event.value
            log(dict(event=kind, task=event.task_name, trial=name))
            if event.event == TrialEvent.START:
                task = tomllib.loads((event.config.task.path / 'task.toml').read_text())
                memory = task['environment'].get('memory_mb', 2048) * 1024**2
                # Preserve official per-task limits while independently
                # controlling their aggregate admission budget.
                async with changed:
                    while not admission.admits(sum(reservations.values()), memory,
                                               psutil.virtual_memory().available):
                        log(dict(event='waiting_for_memory', trial=name, requested=memory,
                                 reserved=sum(reservations.values()),
                                 budget_bytes=admission.budget,
                                 available=psutil.virtual_memory().available))
                        try:
                            await asyncio.wait_for(changed.wait(), 10)
                        except TimeoutError:
                            pass
                    reservations[name] = memory
            if event.event in (TrialEvent.AGENT_END, TrialEvent.END) and name not in released:
                # Harbor assigns this session ID before Terminus.run().
                key = name + '__agent'
                try:
                    response = await client.post(bridge['base'] + '/v1/sessions/release',
                        headers={'x-control-token': bridge['control_token']},
                        json={'session_id': key})
                    response.raise_for_status()
                    value = response.json()
                    if not value.get('released'):
                        raise ValueError('Missing native release acknowledgement')
                    released.add(name)
                    log(dict(event='kv_released', trial=name, **value))
                except Exception as error:
                    release_errors.append(dict(trial=name, error=repr(error)))
                    log(dict(event='release_failed', trial=name, error=repr(error)))
            if event.event == TrialEvent.VERIFICATION_START and name not in verifier_cache_installed:
                task_name = event.config.task.path.name
                entry = manifest.get('verifier_dependency_caches', {}).get(task_name)
                if entry:
                    from terminalbench_verifier_cache import install_for_verification, set_trial_cache_environment
                    if event.config.verifier.env.get('UV_CACHE_DIR'):
                        raise ValueError('Do not replace an explicit verifier cache directory')
                    try:
                        cache_env = await asyncio.to_thread(install_for_verification,
                            root, entry, task_name, name, event.config.task.path,
                            output/(name+'-verifier-cache.json'))
                        set_trial_cache_environment(event.config, cache_env)
                        verifier_cache_installed.add(name)
                        log(dict(event='verifier_cache_ready', trial=name, task=task_name, **cache_env))
                    except Exception as error:
                        verifier_cache_errors.append(dict(trial=name, error=repr(error)))
                        log(dict(event='verifier_cache_failed', trial=name, error=repr(error)))
                        raise
            if event.event == TrialEvent.END:
                result = event.result.model_dump(mode='json')
                completed[name] = classify(result)
                save(output / (name + '.json'), result)
                # END is emitted after Harbor has stopped the environment.
                # Retain stopped containers, images, volumes and answer files.
                try:
                    records = await asyncio.to_thread(reclaim_trial_networks,
                        name.lower()+'__env', event.config.task.path/'environment',
                        output/'network-cleanup'/name)
                    log(dict(event='empty_networks_reclaimed', trial=name, count=len(records)))
                except Exception as error:
                    network_errors.append(dict(trial=name, error=repr(error)))
                    log(dict(event='network_reclamation_failed', trial=name, error=repr(error)))
                async with changed:
                    reservations.pop(name, None)
                    changed.notify_all()
            save(output / 'status.json', dict(at=time.time(), expected=expected,
                completed=len(completed), passed=sum(r['passed'] for r in completed.values()),
                active=list(reservations), released=len(released),
                release_errors=release_errors, network_errors=network_errors,
                verifier_cache_errors=verifier_cache_errors,
                outcomes=list(completed.values())))

        for event in TrialEvent:
            job.add_hook(event, lifecycle)
        result = await job.run()
        save(output / 'harbor-result.json', result.model_dump(mode='json'))
        if (len(completed) != expected or release_errors or network_errors or verifier_cache_errors
                or len(released) != expected):
            raise RuntimeError('Incomplete task classification or KV cleanup; preserve all attempts')
        rows = list(completed.values())
        save(output / 'summary.json', dict(expected=expected, outcomes=rows,
            scored=sum(r['status'] == 'scored' for r in rows), passed=sum(r['passed'] for r in rows),
            task_timeouts=sum(r['status'] == 'agent_timeout' for r in rows),
            errors=sum(r['status'] == 'error' for r in rows), retries=0,
            memory_admission_history=admission.history))
        if config_path.name.startswith('smoke') and not rows[0]['passed']:
            raise RuntimeError('Terminus smoke did not pass; formal tasks have not started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    os.environ['HARBOR_TELEMETRY'] = '0'
    asyncio.run(run(args.root.resolve(), args.config.resolve()))
