"""CPU validation by default; --execute submits a fixed mixed prefill diagnostic.

This is not a live DeepSWE run or an agent-throughput measurement. Run beside
an already started isolated localhost server and point --capture at its files.
The GPU execution path has not been validated. No server is launched/managed.
"""
import argparse
import array
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

import httpx


def read(path):
    return json.loads(path.read_text())


def fixture(path):
    manifest = read(path / 'manifest.json')
    if manifest['format'] != 'deepswe_mixed_pressure_diagnostic_v1':
        raise ValueError('Wrong fixture format')
    for sample in manifest['samples']:
        raw = (path / sample['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sample['sha256']:
            raise ValueError('Fixture bytes changed')
        values = array.array('I')
        values.frombytes(raw)
        if values.itemsize != 4:
            raise ValueError('Require 32-bit token IDs')
        if sys.byteorder != 'little':
            values.byteswap()
        sample['tokens'] = list(values)
        canonical = json.dumps(list(values), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if len(values) != sample['prompt_tokens'] or hashlib.sha256(canonical.encode()).hexdigest() != sample['canonical_token_sha256']:
            raise ValueError('Fixture token count or canonical SHA changed')
    peers = [s for s in manifest['samples'] if s['role'] == 'peer']
    targets = [s for s in manifest['samples'] if s['role'] == 'target']
    if len(peers) != 3 or len(targets) != 1 or manifest['peer_output_tokens'] != 8192:
        raise ValueError('Require three fixed 8192-output peers and one target')
    if any(s['prompt_tokens'] + 8192 >= 48992 for s in peers):
        raise ValueError('Peer could reach pressure refresh during diagnostic')
    return manifest, peers, targets[0]


def validate_profile(profile):
    expected = dict(max_num_seqs=4, draft_tokens=4, draft_method='mtp',
                    max_model_len=262144, max_num_batched_tokens=1876,
                    query_replay_budget=1856, prefill_chunk_tokens=256, weight_offload=None)
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f'Profile must have {key}={value!r}')
    for key, value in dict(select_budget=36864, gen_budget=12288,
                           assembly_chunk_blocks=4, reuse_resident_v=False,
                           gpu_raw_k_cache_bytes=0, step_latency_interval=1).items():
        if profile['kvmem'].get(key) != value:
            raise ValueError(f'Profile must have kvmem.{key}={value!r}')
    if profile.get('native_vllm', False) or profile.get('host') not in ('127.0.0.1', 'localhost'):
        raise ValueError('Require a localhost KVMem server')


class Tail:
    """Only consume newly appended complete JSONL rows; leave server files intact."""
    def __init__(self, path):
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0

    def poll(self):
        if not self.path.exists():
            return []
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            raw = stream.read()
        complete = raw.rfind(b'\n') + 1
        self.offset += complete
        return [json.loads(line) for line in raw[:complete].splitlines()]


def target_shape_evidence(steps, requests, response_id, begin, end):
    """Separate GPU batch evidence from client-stream liveness."""
    valid_interval = begin <= end
    interval = [r for r in steps if begin <= r['started_at'] <= end]
    matched, native_ids = [], set()
    for row in interval:
        targets = [g for g in row.get('batch_geometry', []) if g.get('prefill')
                   and response_id and (g.get('request_id') == response_id
                       or g.get('request_id', '').startswith(response_id+'-'))]
        if targets:
            native_ids.update(g['request_id'] for g in targets)
            matched.append((row,sum(g['query_tokens'] for g in targets)))

    def histogram(pairs):
        groups = {}
        for row, tokens in pairs:
            shape = (row.get('prefill_requests'),row.get('decode_requests'),row.get('prefill_tokens'))
            group = groups.setdefault(shape,dict(prefill_requests=shape[0],decode_requests=shape[1],
                prefill_tokens=shape[2],steps=0,target_public_prefill_tokens=0,model_ms=0.0,step_ms=0.0))
            group['steps'] += 1
            group['target_public_prefill_tokens'] += tokens
            group['model_ms'] += row.get('model_ms',0.0)
            group['step_ms'] += row.get('step_ms',0.0)
        return sorted(groups.values(),key=lambda r:repr((r['prefill_requests'],r['decode_requests'],r['prefill_tokens'])))

    mixed = [(row,tokens) for row,tokens in matched if row.get('decode_requests') == 3]
    target_tokens = sum(tokens for row,tokens in matched)
    peer_counts = []
    for row in requests:
        if row['kind'] != 'decode':
            continue
        at_begin = sum(n for at,n in row['timeline'] if at <= begin) if valid_interval else None
        at_end = sum(n for at,n in row['timeline'] if at <= end) if valid_interval else None
        peer_counts.append(dict(agent=row['agent'],session=row['session'],request_id=row['request_id'],
            accepted_tokens_at_target_begin=at_begin,accepted_tokens_at_target_end=at_end,
            accepted_tokens_between_boundaries=at_end-at_begin if valid_interval else None))
    return dict(target_interval_histogram=histogram([(row,0) for row in interval]),
        target_prefill_histogram=histogram(matched),target_native_request_ids=sorted(native_ids),
        target_interval_valid=valid_interval,target_interval_steps=len(interval),target_interval_steps_without_batch_geometry=sum(
            not row.get('batch_geometry') for row in interval),target_prefill_steps=len(matched),
        target_public_prefill_tokens=target_tokens,mixed_target_steps=len(mixed),
        three_decode_step_fraction=len(mixed)/len(matched) if matched else None,
        three_decode_target_token_fraction=sum(tokens for row,tokens in mixed)/target_tokens if target_tokens else None,
        peer_accepted_tokens=sorted(peer_counts,key=lambda r:r['agent']),
        scope='GPU rows use started_at inside the target interval and exact target request identity in '
              'batch_geometry. Coverage denominators include only target public-prefill rows. '
              'Peer counts are client streaming observations at or before each boundary; '
              'unfinished HTTP streams do not prove a constant three-decode GPU batch.')


async def run(args):
    profile = read(args.profile)
    validate_profile(profile)
    data, peers, target = fixture(args.fixture)
    spec = importlib.util.spec_from_file_location('diagnostic_stream_client', args.runtime/'scripts/timed_agent_workload.py')
    timed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timed)
    args.output.mkdir(parents=True, exist_ok=False)
    recorder = timed.Recorder(args.output, 4)
    prefix = f'pressure-diag-{args.tag}-{uuid.uuid4().hex[:10]}'
    samples = peers + [target]
    sessions = [f'{prefix}-{i}' for i in range(4)]
    touched, jobs, cleanup_errors = [], [], []
    tails = {name: Tail(args.capture/f'{name}.jsonl') for name in
             ('model_step_latency', 'bounded_refresh', 'restore_host_stages', 'host_stages',
              'verified_restores', 'verified_state_restores', 'verified_rolling_pressure',
              'refresh_control_allocations')}
    collected = {name: [] for name in tails}
    result = dict(diagnostic_only=True, status='running', namespace=prefix,
        fixture_sha256=hashlib.sha256((args.fixture/'manifest.json').read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        scope=data['scope'], tag=args.tag, begin_monotonic=time.monotonic(), begin_wall=time.time())
    timed.save(args.output/'profile.json', profile)

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), trust_env=False,
                                limits=httpx.Limits(max_keepalive_connections=0)) as client:
        async def snapshot(name):
            at = dict(monotonic=time.monotonic(), wall=time.time())
            response = await client.get(f"http://127.0.0.1:{profile['port']}/metrics")
            response.raise_for_status()
            (args.output/f'{name}-metrics.txt').write_text(response.text)
            for item in ('scheduler_resources', 'archive', 'cache-geometry'):
                path = args.capture/f'{item}.json'
                if path.exists():
                    timed.save(args.output/f'{name}-{item}.json', read(path))
            at['completed_monotonic'] = time.monotonic()
            timed.save(args.output/f'{name}-time.json', at)
            return at

        def poll():
            for name, tail in tails.items():
                collected[name].extend(tail.poll())

        async def call(i, phase, tokens, count, release=False):
            if i not in touched:
                touched.append(i)
            recorder.phase(i, phase, 0, 0)
            row = await timed.stream_call(client, profile, tokens, sessions[i], count, recorder,
                agent=i, epoch=0, turn=0, kind='cleanup' if release else phase,
                request_id=f'{sessions[i]}-{phase}', new_input_tokens=len(tokens) if phase in ('prime','target') else 1,
                query=not release, query_plan=None if release else samples[i]['diagnostic_request_plan'],
                release_session=release, source_metadata=dict(task=samples[i]['task'],
                    turn=samples[i]['turn'], original_token_sha256=samples[i]['canonical_token_sha256']))
            if row['usage'].get('prompt_tokens') != len(tokens):
                raise ValueError('Server logical prompt count differs from exact input')
            recorder.phase(i, phase+'_finished', 0, 0)
            return row

        try:
            before = await timed.wait_retirement(args.capture)
            if before.get('session_views'):
                raise RuntimeError('Isolated diagnostic requires no existing sessions; will not clear unrelated sessions')
            await snapshot('before')
            # Keep each primed session: its existing authority is reused on resume,
            # so three primes consume three namespaces, leaving one for the target.
            for i, sample in enumerate(peers):
                await call(i, 'prime', sample['tokens'], 1)
                await timed.wait_retirement(args.capture,
                    expected_views={sessions[j]:peers[j]['prompt_tokens'] for j in range(i+1)})
            await snapshot('primed')
            result['resume_begin'] = time.monotonic()
            jobs = [asyncio.create_task(call(i, 'decode', sample['tokens']+[sample['continuation_token_id']], 8192))
                    for i, sample in enumerate(peers)]
            consecutive, stable_at = 0, None
            deadline = time.monotonic() + 120
            while stable_at is None:
                if any(job.done() for job in jobs):
                    raise RuntimeError('Peer ended or failed before target admission')
                if time.monotonic() >= deadline:
                    raise TimeoutError('No stable three-peer decode evidence within 120 seconds')
                poll()
                rows = collected['model_step_latency']
                # A minimum of 32 accepted tokens per peer avoids treating a
                # one-off speculative warm-up step as the steady cohort.
                outputs = [r for r in recorder.rows if r['kind'] == 'decode']
                live = len(outputs) == 3 and all(len(r['output_ids']) >= 32 for r in outputs)
                recent = [r for r in rows if r['started_at'] >= result['resume_begin']]
                consecutive = 0
                for row in reversed(recent):
                    if row['decode_requests'] != 3 or row['prefill_tokens'] != 0:
                        break
                    consecutive += 1
                if live and consecutive >= 3:
                    stable_at = time.monotonic()
                    result['stable_three_decode'] = dict(at=stable_at, consecutive_steps=consecutive,
                        last_step=recent[-1], peer_accepted_tokens=[len(r['output_ids']) for r in outputs])
                else:
                    await asyncio.sleep(.02)
            await snapshot('target_before')
            result['target_begin'] = time.monotonic()
            target_row = await call(3, 'target', target['tokens'], 1)
            result['target_end'] = time.monotonic()
            result['target_ttft_seconds'] = target_row['timeline'][0][0]-target_row['begin']
            result['peers_live_until_target_finished'] = all(not job.done() for job in jobs)
            await snapshot('target_after')
            rows = await asyncio.gather(*jobs)
            await timed.wait_retirement(args.capture, expected_views={
                **{sessions[i]:peers[i]['prompt_tokens']+8192 for i in range(3)},
                sessions[3]:target['prompt_tokens']})
            await snapshot('drained')
            poll()
            result['fixed_peer_output_tokens'] = sum(len(row['output_ids']) for row in rows)
            result['target_response_id'] = target_row['response_id']
            result['status'] = 'completed'
        except BaseException as error:
            result.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            try:
                await timed.wait_retirement(args.capture)
                for i in touched:
                    await call(i, 'release', [data['eos']], 1, release=True)
                after = await timed.wait_retirement(args.capture)
                if any(v.get('namespace', [None])[0] in sessions for v in after.get('session_views', [])):
                    raise RuntimeError('Owned session remained after explicit release')
                await snapshot('after_cleanup')
            except BaseException as error:
                cleanup_errors.append(f'{type(error).__name__}: {error}')
            poll()
            for name, rows in collected.items():
                (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            response_id = result.get('target_response_id')
            def target_matches(row):
                request_id = row.get('request_id', '')
                return bool(response_id and (request_id == response_id or request_id.startswith(response_id+'-')))
            target_rows = [r for r in collected['bounded_refresh'] if target_matches(r)]
            result['target_refresh_rows'] = target_rows
            result['target_batch_shapes'] = target_shape_evidence(collected['model_step_latency'],
                recorder.rows,response_id,result.get('target_begin',float('inf')),result.get('target_end',-1))
            result['mixed_target_steps'] = result['target_batch_shapes']['mixed_target_steps']
            result['private_query_forward_steps'] = sum(r.get('private_forward_steps', 0) for r in target_rows)
            # A cold target's pressure rebuild uses RollingPressureOracle;
            # it does not import a session checkpoint and cannot emit the
            # session-restoration oracle records used by resumed peers.
            proofs = [r for r in collected['verified_rolling_pressure'] if target_matches(r)]
            result['kv_byte_oracle'] = dict(enabled=profile.get('verify_session_restore', False),
                kind='verified_rolling_pressure', verified_target_rows=len(proofs),
                passed=bool(len(proofs) == len(target_rows) == 2 and all(
                    r.get('target_kv_byte_equal') is True and r.get('recurrent_byte_equal') is True
                    and r.get('draft_byte_equal') is True and r.get('mtp_attention_layers', 0) > 0
                    and not r.get('cancelled') and r.get('private_forward_steps') == 0 for r in proofs)))
            result['cleanup_errors'] = cleanup_errors
            result['end_monotonic'] = time.monotonic()
            if result['status'] == 'completed' and (cleanup_errors or not result['mixed_target_steps']
                    or not result.get('peers_live_until_target_finished') or len(target_rows) != 2
                    or result['private_query_forward_steps']
                    or (profile.get('verify_session_restore') and not result['kv_byte_oracle']['passed'])):
                result.update(status='failed', error='Diagnostic shape/accounting assertion failed; inspect rows')
            timed.save(args.output/'result.json', result)
            recorder.stream.close()
    print(json.dumps({k: result[k] for k in ('status','target_ttft_seconds','mixed_target_steps','cleanup_errors') if k in result}))
    return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True, help='Frozen client runtime containing scripts/timed_agent_workload.py')
    parser.add_argument('--profile', type=Path, required=True, help='Actual running server profile')
    parser.add_argument('--capture', type=Path, required=True, help='Actual running server capture directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--execute',action='store_true',
        help='Submit inference requests to the existing localhost GPU server; default only validates local inputs')
    args = parser.parse_args()
    if not args.tag.replace('-', '').replace('_', '').isalnum():
        parser.error('--tag must contain only letters, numbers, dash, underscore')
    if not args.execute:
        validate_profile(read(args.profile))
        data,peers,target = fixture(args.fixture)
        if not (args.runtime/'scripts/timed_agent_workload.py').is_file():
            parser.error('The frozen runtime must contain scripts/timed_agent_workload.py')
        if args.output.exists():
            parser.error('--output must not already exist')
        print(json.dumps(dict(status='validated_cpu_only',server_contacted=False,
            peer_prompt_tokens=[s['prompt_tokens'] for s in peers],
            peer_output_tokens=data['peer_output_tokens'],target_prompt_tokens=target['prompt_tokens'],
            scope=data['scope'],gpu_execution_validated=False)))
        return
    raise SystemExit(asyncio.run(run(args)))


if __name__ == '__main__':
    main()
