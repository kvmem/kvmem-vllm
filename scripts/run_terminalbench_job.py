"""Run the pinned, unmodified Harbor Terminus agent with local lifecycle hooks.

Hooks only account resources, persist evidence, and release finished KV owners.
They never change a task, prompt, verifier, reward, or agent response. The frozen
task copy may explicitly omit Agent deadlines under agent-time-policy.json.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import tomllib

import httpx
import psutil
try:
    from .terminalbench_network import reclaim_trial_networks, assert_direct_docker, DIRECT_CONTAINER_ENV
except ImportError:
    from terminalbench_network import reclaim_trial_networks, assert_direct_docker, DIRECT_CONTAINER_ENV


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
    if any(config.environment.env.get(key) != value for key, value in DIRECT_CONTAINER_ENV.items()):
        raise ValueError('New job config must explicitly disable container proxies')
    job_dir = config.jobs_dir / config.job_name
    if job_dir.exists():
        raise ValueError('Refuse an attempted job directory; recovery requires a separate audit')
    if config.n_attempts != 1 or config.retry.max_retries != 0:
        raise ValueError('Exactly one attempt per task, no task retries')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('harness_repair'):
        from terminalbench_harness import install
        bundles = {task: root / relative for task, relative in manifest.get('terminal_tool_bundles', {}).items()}
        if any(not path.resolve().is_relative_to(root.resolve()) for path in bundles.values()):
            raise ValueError('Dependency bundle outside frozen experiment')
        install(root/(config.job_name+'-harness.jsonl'), manifest['harness_repair']['upstream_sources'], bundles,
                recorder_transport=manifest['harness_repair'].get('recorder_transport', False))
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
    changed = asyncio.Condition()
    bridge = json.loads((root / 'connection.json').read_text())
    async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
        def log(value):
            with (output / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=time.time(), **value), ensure_ascii=False) + '\n')

        async def lifecycle(event):
            name = event.trial_name
            kind = event.event.value
            log(dict(event=kind, task=event.task_name, trial=name))
            if event.event == TrialEvent.START:
                task = tomllib.loads((event.config.task.path / 'task.toml').read_text())
                memory = task['environment'].get('memory_mb', 2048) * 1024**2
                # Preserve each official resource allocation. Admit fewer than
                # eight only if large tasks would exceed this host reserve.
                async with changed:
                    while (sum(reservations.values()) + memory > 24 * 1024**3 or
                           psutil.virtual_memory().available < memory + 12 * 1024**3):
                        log(dict(event='waiting_for_memory', trial=name, requested=memory,
                                 reserved=sum(reservations.values()),
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
                outcomes=list(completed.values())))

        for event in TrialEvent:
            job.add_hook(event, lifecycle)
        result = await job.run()
        save(output / 'harbor-result.json', result.model_dump(mode='json'))
        if len(completed) != expected or release_errors or network_errors or len(released) != expected:
            raise RuntimeError('Incomplete task classification or KV cleanup; preserve all attempts')
        rows = list(completed.values())
        save(output / 'summary.json', dict(expected=expected, outcomes=rows,
            scored=sum(r['status'] == 'scored' for r in rows), passed=sum(r['passed'] for r in rows),
            task_timeouts=sum(r['status'] == 'agent_timeout' for r in rows),
            errors=sum(r['status'] == 'error' for r in rows), retries=0))
        if expected == 1 and not rows[0]['passed']:
            raise RuntimeError('Terminus smoke did not pass; formal tasks have not started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    os.environ['HARBOR_TELEMETRY'] = '0'
    asyncio.run(run(args.root.resolve(), args.config.resolve()))
