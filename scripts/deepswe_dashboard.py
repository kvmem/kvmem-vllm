"""Local, read-only DeepSWE dashboard backed by the existing benchmark monitors."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

PAGE = Path(__file__).resolve().parents[1] / 'benchmarks/deepswe/dashboard/index.html'
PROMPT = 'vllm:prompt_tokens_total'
CACHED = 'vllm:prompt_tokens_cached_total'
OUTPUT = 'vllm:generation_tokens_total'
METRICS = {PROMPT, CACHED, OUTPUT, 'vllm:num_requests_running', 'vllm:num_requests_waiting'}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def user_pause(root):
    """A display-only pause record never rewrites benchmark outcomes."""
    pause = read_json(root / 'user-pause.json')
    if (not isinstance(pause, dict) or pause.get('status') != 'paused'
            or pause.get('reason') != 'user_requested'):
        return {}
    at = pause.get('at')
    if type(at) not in (int, float) or not math.isfinite(at) or at <= 0:
        return {}
    return {key: pause[key] for key in ('at', 'reason', 'audit') if key in pause}


def paused_task_state(state, record, case, pause_at):
    if pause_at is None or state in ('completed', 'context_ceiling', 'queued'):
        return state
    ended_at = record.get('ended_at', case.get('ended_at'))
    if state == 'infrastructure_error':
        # Unknown error time is not evidence that our stop caused the error.
        return 'paused' if ended_at is not None and ended_at >= pause_at else state
    if (state in ('running', 'model', 'tools', 'verifying', 'interrupted')
            and record.get('started_at') is not None
            and (ended_at is None or ended_at >= pause_at)):
        return 'paused'
    return state


def gc_progress_view(progress, workers, now):
    """Completed synthetic history is progress, never a scored DeepSWE task."""
    if not progress or not progress.get('status'):
        return None
    target = progress.get('target_tokens')
    target = target if type(target) is int and target > 0 else None
    rows = progress.get('owners', [])
    owners = []
    for index in range(workers):
        row = rows[index] if index < len(rows) else {}
        history = row.get('history')
        history = history if type(history) is int and history >= 0 else None
        owners.append(dict(number=index + 1, history=history, turns=row.get('turns'),
            done=bool(row.get('done')), percent=min(history / target, 1) * 100
            if history is not None and target else None))
    total = sum(row['history'] for row in owners) if owners and all(
        row['history'] is not None for row in owners) else None
    at, start = progress.get('at'), progress.get('started_at')
    age = max(0, now-at) if at is not None else None
    return dict(status=progress['status'], owners=owners, target_tokens=target,
        history_tokens=total, target_total=target * workers if target else None,
        percent=sum(row['percent'] for row in owners) / workers
        if total is not None and target else None,
        completed_owners=sum(row['done'] for row in owners),
        retired_blocks=progress.get('retired_blocks'), at=at, age_seconds=age,
        elapsed_seconds=max(0, at-start) if at is not None and start is not None else None,
        stale=progress['status']=='running' and (age is None or age > 35))


class JsonLines:
    """Incremental reads; an unfinished final line remains unread until published."""
    def __init__(self, path, limit=2200, transform=None, initial_tail_bytes=None):
        self.path, self.offset, self.identity = path, 0, None
        self.rows = deque(maxlen=limit)
        self.transform, self.initial_tail_bytes = transform, initial_tail_bytes

    def update(self):
        added = []
        try:
            stat = self.path.stat()
            if self.identity != stat.st_ino or stat.st_size < self.offset:
                self.offset, self.identity = 0, stat.st_ino
                self.rows.clear()
                if self.initial_tail_bytes and stat.st_size > self.initial_tail_bytes:
                    with self.path.open('rb') as stream:
                        stream.seek(stat.st_size-self.initial_tail_bytes)
                        stream.readline()
                        self.offset=stream.tell()
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                while line := stream.readline():
                    if not line.endswith(b'\n'):
                        break
                    self.offset = stream.tell()
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row,dict):
                        continue
                    if self.transform:
                        row=self.transform(row)
                    self.rows.append(row)
                    added.append(row)
        except OSError:
            pass
        return added


def delta_rate(before, after, keys):
    seconds = after['at'] - before['at']
    if not 0 < seconds <= 45:
        return None
    changes = []
    for key, sign in keys:
        a, b = before.get('counters', {}).get(key), after.get('counters', {}).get(key)
        if a is None or b is None or b < a:
            return None  # Missing metrics or a restarted counter are not zero throughput.
        changes.append(sign * (b - a))
    total = sum(changes)
    return total / seconds if total >= 0 else None


def throughput_series(rows, formal_start, phase='running'):
    result, previous = [], None
    for row in rows:
        if (not formal_start or row.get('run_phase') != phase
                or row.get('at', 0) < formal_start or 'error' in row):
            previous = None
            continue
        counters = row.get('counters', {})
        point = dict(at=row['at'], prefill=None, decode=None,
                     running=counters.get('vllm:num_requests_running'),
                     waiting=counters.get('vllm:num_requests_waiting'))
        if previous:
            point['prefill'] = delta_rate(previous, row, [(PROMPT, 1), (CACHED, -1)])
            point['decode'] = delta_rate(previous, row, [(OUTPUT, 1)])
        result.append(point)
        previous = row
    return result


def resource_sample(row):
    """Reuse existing telemetry without another model/GPU sampler."""
    result={k:row[k] for k in ('at','archive','scheduler_resources','available_bytes','disk_free_bytes','telemetry_error') if k in row}
    counters={}
    for line in row.get('raw_metrics','').splitlines():
        if not line or line.startswith('#'):continue
        fields=line.split();name=fields[0].split('{',1)[0]
        if name in METRICS:
            try:counters[name]=counters.get(name,0)+float(fields[-1])
            except (ValueError,IndexError):pass
    result['counters']=counters
    if 'raw_metrics' not in row:result['error']=row.get('telemetry_error','Missing native counters')
    try:
        memory,utilization=row['gpu'].split(',')
        result['gpu']=dict(memory_mib=float(memory),utilization=float(utilization))
    except (KeyError,ValueError,AttributeError):result['gpu']={}
    return result


class ToolActivity:
    def __init__(self,path):
        self.events=JsonLines(path,limit=1)
        self.calls,self.results={},{}
        self.last_tool=None

    def snapshot(self):
        for row in self.events.update():
            if row.get('type') not in ('assistant','user'):continue
            content=row.get('message',{}).get('content',[])
            if not isinstance(content,list):continue
            for block in content:
                if not isinstance(block,dict):continue
                if block.get('type')=='tool_use' and block.get('id') not in self.calls:
                    self.calls[block['id']]=block.get('name','tool')
                    self.last_tool=dict(name=block.get('name','tool'),phase='requested')
                elif block.get('type')=='tool_result' and block.get('tool_use_id') not in self.results:
                    self.results[block['tool_use_id']]=bool(block.get('is_error'))
                    self.last_tool=dict(name=self.calls.get(block['tool_use_id'],'tool'),phase='completed',error=bool(block.get('is_error')))
        return dict(tool_calls=len(self.calls),tool_results=len(self.results),
                    tool_errors=sum(self.results.values()),last_tool=self.last_tool)


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.metrics = JsonLines(root / 'native-metrics.jsonl')
        self.resources = JsonLines(root/'resources.jsonl', limit=4000, transform=resource_sample,
                                   initial_tail_bytes=16*1024**2)
        self.events = JsonLines(root / 'bridge/requests.jsonl', limit=1)
        self.gc_samples = JsonLines(root / 'long-gc-preflight/samples.jsonl',
                                    initial_tail_bytes=4*1024**2)
        self.activity, self.statistics, self.finished, self.tools = {}, {}, {}, {}
        self.activity_history = []
        self.lock, self.cached_at, self.cached = threading.Lock(), 0, None

    def snapshot(self):
        with self.lock:
            now = time.time()
            if self.cached and now - self.cached_at < 1:
                return self.cached
            root = self.root
            manifest = read_json(root / 'manifest.json')
            run = read_json(root / 'run.json')
            status = read_json(root / 'status.json')
            pause = user_pause(root)
            pause_at = pause.get('at')
            ended_at = run.get('ended_at')
            if pause_at is not None:
                ended_at = min(ended_at, pause_at) if ended_at is not None else pause_at
            cutoff = ended_at if pause else None
            summary = read_json(root / 'live-summary.json')
            resources = read_json(root / 'server/capture/scheduler_resources.json')
            history = read_json(root / 'server/capture/archive.json')
            self.metrics.update()
            self.resources.update()
            resource_rows = [row for row in self.resources.rows
                             if cutoff is None or row.get('at', float('inf')) <= cutoff]
            resource = resource_rows[-1] if resource_rows else {}
            if pause:
                # Live capture files may already include shutdown work; only a
                # timestamped telemetry row can supply pause-time resources.
                resources = resource.get('scheduler_resources', {})
                history = resource.get('archive', {})
                if summary.get('at', float('inf')) > cutoff:
                    summary = {}
            formal_start = run.get('model_ready_at')
            for event in self.events.update():
                task = event.get('task')
                if not task or not formal_start or event.get('at',0) < formal_start:
                    continue
                activity = self.activity.setdefault(task, {})
                kind = event.get('event')
                if kind == 'start':
                    activity.update(pending=True, at=event['at'], turn=event.get('turn'),
                                    prompt_tokens=event.get('prompt_tokens'))
                elif kind in ('finish', 'error', 'cancelled', 'history_released'):
                    activity.update(pending=False, at=event['at'])
                if kind in ('start', 'finish', 'error', 'cancelled', 'history_released'):
                    self.activity_history.append((task, dict(activity)))
                if kind == 'finish':
                    identity = event.get('request_id') or (task,event.get('turn'),event['at'])
                    if identity not in self.finished:
                        self.finished[identity] = dict(task=task, at=event['at'],
                            output=event.get('usage',{}).get('completion_tokens',0),
                            generation_ms=(event.get('metrics') or {}).get('generation_time_ms'),
                            ttft_ms=(event.get('metrics') or {}).get('time_to_first_token_ms'))
                        stat = self.statistics.setdefault(task,dict(requests=0,output_tokens=0))
                        stat['requests'] += 1
                        stat['output_tokens'] += self.finished[identity]['output']
            finished = [row for row in self.finished.values() if cutoff is None or row['at'] <= cutoff]
            activity_view, statistics_view = self.activity, self.statistics
            if pause:
                # Reconstruct even when the pause file arrives after this
                # dashboard has already consumed some post-cutoff events.
                activity_view, statistics_view = {}, {}
                for task, event in self.activity_history:
                    if event['at'] <= cutoff:
                        activity_view[task] = event
                for event in finished:
                    stat = statistics_view.setdefault(event['task'], dict(requests=0, output_tokens=0))
                    stat['requests'] += 1
                    stat['output_tokens'] += event['output']
            measurement_phase = 'formal' if formal_start else 'preflight'
            metric_rows = [row for row in self.metrics.rows
                           if cutoff is None or row.get('at', float('inf')) <= cutoff]
            if not metric_rows:
                metric_rows = [dict(row,run_phase='running' if formal_start and row.get('at',0)>=formal_start else 'starting')
                               for row in resource_rows]
            series = throughput_series(metric_rows, formal_start or run.get('started_at'),
                                       'running' if formal_start else 'starting')
            latest = series[-1] if series else {}
            metric_age = now - latest['at'] if latest else None
            state = run.get('status', status.get('status', 'starting'))
            queue_status = read_json(root.parent/'status.json') if root.name == 'first16' else {}
            # A prepared cohort has no model metrics until its finite GPU queue
            # clears the foreign owner and runs component qualification.
            if not run:
                state = {'waiting_for_gpu': 'waiting_for_gpu',
                         'gpu_lru_qualification': 'starting',
                         'qualification_then_testing': 'starting',
                         'prepared': 'prepared', 'failed': 'failed'}.get(
                             queue_status.get('status'), state)
            elif not formal_start and queue_status.get('status') == 'failed':
                state = 'failed'
            if pause:
                state = 'paused'
            active = state in ('starting', 'running', 'prepared', 'waiting_for_gpu')
            stale = active and bool(formal_start or series) and (metric_age is None or metric_age > 35)
            statistics = {r['task']: r for r in summary.get('tasks', [])}
            tasks = []
            for task in manifest.get('tasks', []):
                tid = task['task_id']
                record = run.get('tasks', {}).get(tid, {})
                stats = {**statistics.get(tid, {}), **statistics_view.get(tid,{})}
                if pause:
                    stats.update(statistics_view.get(tid, dict(requests=0, output_tokens=0)))
                case = read_json(root / 'cases' / tid / 'status.json')
                activity = activity_view.get(tid, {})
                task_state = record.get('state', 'queued')
                if task_state == 'running':
                    if case.get('state') in ('verifying', 'finished'):
                        task_state = 'verifying'
                    else:
                        task_state = 'model' if activity.get('pending') else 'tools'
                if not active and task_state in ('model', 'tools', 'verifying'):
                    task_state = 'interrupted'
                task_state = paused_task_state(task_state, record, case, pause_at)
                elapsed = stats.get('total_seconds') or stats.get('agent_seconds')
                if record.get('started_at') is not None:
                    if task_state == 'paused':
                        elapsed = max(0, pause_at - record['started_at'])
                    elif elapsed is None:
                        task_end = record.get('ended_at')
                        if task_end is None:
                            task_end = ended_at if ended_at is not None else now
                        elapsed = max(0, task_end - record['started_at'])
                tool = self.tools.setdefault(tid,ToolActivity(root/'cases'/tid/'claude.stream.jsonl')).snapshot()
                reward = (record.get('result',{}).get('reward') or {}).get('reward',stats.get('reward'))
                tasks.append(dict(id=tid, title=task.get('title', tid), state=task_state,
                    reward=reward, requests=stats.get('requests', 0),
                    output_tokens=stats.get('output_tokens', 0),
                    context_tokens=activity.get('prompt_tokens', stats.get('last_prompt_tokens')),
                    elapsed_seconds=elapsed,
                    pending_seconds=max(0, now-activity['at']) if activity.get('pending') and active else None,
                    termination=record.get('termination',stats.get('termination')),**tool))
            admission = resources.get('cpu_history_admission', {})
            profile = manifest.get('profile', {})
            workers = run.get('workers', manifest.get('workers', 8))
            preflight = dict(reuse_requested=bool(manifest.get('preflight_reuse')),
                query=bool(read_json(root/'atomic-query-preflight.json').get('passed')),
                short_tail=bool(read_json(root/'short-tail-preflight.json').get('passed')),
                long_gc=read_json(root/'long-gc-preflight/progress.json'),
                generation_completed=sum(bool(read_json(path).get('cleanup_passed'))
                    for path in root.glob('generation-preflight-*.json')))
            if pause:
                self.gc_samples.update()
                preflight['long_gc'] = next((row for row in reversed(self.gc_samples.rows)
                    if row.get('at', float('inf')) <= cutoff), {})
            preflight['progress'] = gc_progress_view(preflight['long_gc'], workers,
                                                     cutoff if pause else now)
            output = sum(r['output'] for r in finished)
            elapsed = max(0,(ended_at if ended_at is not None else now)-formal_start) if formal_start else 0
            ttft = [r['ttft_ms']/1000 for r in finished if r['ttft_ms'] is not None]
            decode = [r for r in finished if r['generation_ms'] is not None]
            decode_seconds = sum(r['generation_ms']/1000 for r in decode)
            archive = resource.get('archive',{}) or history
            tool_counts = {key:sum(t[key] for t in tasks) for key in ('tool_calls','tool_results','tool_errors')}
            tool_counts['error_rate'] = tool_counts['tool_errors']/tool_counts['tool_results'] if tool_counts['tool_results'] else None
            control_at = resource.get('at', None if pause else status.get('at'))
            self.cached = dict(at=now, run_name=root.parent.name if root.name=='first16' else root.name, state=state,
                queue_state=queue_status.get('status'),
                measurement_phase=measurement_phase, preflight=preflight,
                formal_start=formal_start, ended_at=ended_at,elapsed_seconds=elapsed,
                user_pause=pause or None, metrics_historical=not active,
                measurement_cutoff=cutoff,
                workers=workers,
                planned_tasks=len(tasks), completed_tasks=sum(t['state']=='completed' for t in tasks),
                passed_tasks=sum(t['state']=='completed' and t['reward']==1 for t in tasks),
                context_limited_tasks=sum(t['state']=='context_ceiling' for t in tasks),
                infrastructure_errors=sum(t['state']=='infrastructure_error' for t in tasks),tool_counts=tool_counts,
                metric_at=latest.get('at'), metric_age_seconds=metric_age, stale=stale,
                summary_age_seconds=now-summary['at'] if summary.get('at') else None,
                control_age_seconds=now-control_at if control_at else None,
                sample_seconds=(series[-1]['at']-series[-2]['at']) if len(series)>1 else None,
                latest=latest, series=series, tasks=tasks,
                gpu=resource.get('gpu', {} if pause else status.get('gpu', {})), gpu_at=control_at, cpu=admission,
                cpu_used_bytes=archive.get('cpu_authority_bytes',summary.get('cpu_authority_bytes')),
                history_gc=history.get('historical_gc'),
                refresh_pending=resources.get('refresh_pending'),restore_pending=resources.get('connector_pending'),
                output_tokens=output,requests=len(finished),e2e_output_rate=output/elapsed if elapsed else None,
                single_stream_decode_rate=sum(max(0,r['output']-1) for r in decode)/decode_seconds if decode_seconds else None,
                ttft_seconds=sum(ttft)/len(ttft) if ttft else None,
                restored_tokens=archive.get('session_restored_tokens',0)-run.get('preflight_native_counters',{}).get('session_restored_tokens',0),
                disk_read_bytes=archive.get('disk_read_bytes'),disk_write_bytes=archive.get('disk_write_bytes'),
                model=dict(draft_tokens=profile.get('draft_tokens'),context_limit=profile.get('max_model_len'),
                    select_budget=profile.get('kvmem', {}).get('select_budget'),
                    gen_budget=profile.get('kvmem', {}).get('gen_budget')))
            self.cached_at = now
            return self.cached

def serve(root, port):
    dashboard = Dashboard(root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                body, content_type = PAGE.read_bytes(), 'text/html; charset=utf-8'
            elif path == '/api/status':
                body = json.dumps(dashboard.snapshot(), ensure_ascii=False, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            elif path == '/health':
                body, content_type = b'{"ok":true}', 'application/json'
            elif path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Local: http://localhost:{server.server_address[1]}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--port', type=int, default=50123)
    args = parser.parse_args()
    if not any((args.root / name).is_file() for name in ('run.json', 'manifest.json')):
        parser.error('root must contain a DeepSWE run.json or prepared manifest.json')
    serve(args.root.resolve(), args.port)
