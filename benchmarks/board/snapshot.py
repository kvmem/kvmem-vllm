"""Snapshot an experiment's progress for the Experiment Run Board dashboard.

usage: snapshot.py SPEC.json OUT_DIR [--unblind]

Writes four JSON files, each {"rows": [...], "meta": [...]}, in the board's fixed schema
(README.md). The board draws any experiment written in this schema, so a new experiment
only needs a spec, or its own writer of the same four files.

SPEC.json:
{
  "experiment": "lme500",                      short id
  "title": "LongMemEval-S, 500 questions",     shown as the board's heading
  "root": "/path/to/experiment",               arm directories are relative to it
  "manifest": "dataset/manifest.json",         optional: {"rows": [{"id", <group_field>}]} for totals and groups
  "group_field": "question_type",              per-case group (task type, suite, ...)
  "grading": "grading",                        optional: grade.py directory (key.json, judgments.jsonl), or
             {"kind": "per_case", "dir": "grading-deepseek"}   per-arm verdicts <dir>/<arm dir>/<id>.json {"correct"}
                                               from a judge that never sees arm names (no unblinding needed)
  "arms": [{"arm": "kvmem", "label": "...", "dir": "kvmem", "settings": {"Select budget K": "36,864", ...},
            "kvmem_capture": "kvmem/server*/capture",  optional: KVMem capture directories (a glob; a resumed
                                               arm writes server-r1, ...): prefill throughput and decision checks
            "layer_checks": true}]             optional: false for a global-path arm (no per-layer Query events)
}

An arm directory holds cases/<id>/result.json with at least status, started, ended (epoch seconds),
prompt_tokens, usage.completion_tokens and finish_reason (benchmarks' usual per-case record);
status.json ({"status": "done"|"incomplete"}) when it ended, process.json ({"server_pid"}) while it runs.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path


def iso(epoch):
    return None if epoch is None else dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec='seconds')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def grades(root, spec, unblind):
    """({(arm, case id): 0|1}, judged): blind A/B sheet marks once judging ends (or --unblind), or per-case
    verdicts of an arm-blind judge (always shown). judged is a count, or {arm: count} for per-case verdicts."""
    if not spec.get('grading'):
        return {}, 0
    if isinstance(spec['grading'], dict):
        if spec['grading'].get('kind') != 'per_case':
            raise ValueError('grading must be a grade.py directory or {"kind": "per_case", "dir": ...}')
        base, out, judged = root / spec['grading']['dir'], {}, {}
        for arm in spec['arms']:
            for path in sorted((base / arm.get('grading_dir', arm['dir'])).glob('*.json')):
                verdict = read(path)
                if verdict and 'correct' in verdict:
                    out[(arm['arm'], path.stem)] = int(bool(verdict['correct']))
            judged[arm['arm']] = sum(1 for key in out if key[0] == arm['arm'])
        return out, judged
    directory = root / spec['grading']
    key = read(directory / 'key.json') or {}
    path = directory / 'judgments.jsonl'
    judged = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []
    out = {}
    for j in judged:
        for label, arm in zip('AB', key.get(j['id'], ())):
            out[(arm, j['id'])] = j[label]
    return (out if unblind else {}), len(judged)


def _lines(captures, name):
    for capture in captures:
        path = Path(capture) / name
        yield from (path.open() if path.is_file() else ())


def kvmem_checks(captures, records, budget):
    """Per-arm health checks from KVMem captures: Query and pressure decisions per finished request."""
    by_rid = {r.get('response_id'): r['id'] for r in records if r.get('response_id')}
    stats = {qid: dict(layers=0, armed=0, prefill=0, decode=0, fallback=0, charged=0) for qid in by_rid.values()}
    for line in _lines(captures, 'layerwise_query.jsonl'):
        e = json.loads(line)
        qid = by_rid.get('-'.join(str(e.get('request_id', '')).split('-')[:2]))
        if qid is None:
            continue
        s = stats[qid]
        if e.get('event') == 'layer_selected':
            s['layers'] += 1
            s['charged'] = max(s['charged'], e.get('charged_tokens', 0))
        elif e.get('event') == 'armed':
            s['armed'] += 1
        elif e.get('event') == 'pressure_selected':
            s['decode' if e.get('event_type') == 'decode_pressure' else 'prefill'] += 1
            s['fallback'] += e.get('fallback_reason') is not None
            for layer in e.get('layers', ()):
                s['charged'] = max(s['charged'], layer.get('charged_tokens', 0))
    values = list(stats.values())
    n = len(values)
    complete = sum(v['layers'] == 16 and v['armed'] == 1 for v in values)
    prefill = sum(v['prefill'] for v in values)
    decode = sum(v['decode'] for v in values)
    fallback = sum(v['fallback'] for v in values)
    charged = max((v['charged'] for v in values), default=0)
    return [
        dict(check='Query selected on all 16 layers', value='%d / %d cases' % (complete, n), target='every case',
             state='ok' if complete == n else 'bad'),
        dict(check='Prefill pressure reselections', value='%d (%.1f per case)' % (prefill, prefill / n if n else 0),
             target='at least one per case', state='ok' if n and min(v['prefill'] for v in values) > 0 else 'warn'),
        dict(check='Rolling-decode pressure reselections', value=str(decode), target='fires once output fills G',
             state='ok' if decode else 'info'),
        dict(check='Selection fallbacks', value=str(fallback), target='0', state='ok' if not fallback else 'bad'),
        dict(check='Largest per-layer selection', value='{:,} tokens'.format(charged), target='≤ {:,}'.format(budget),
             state='ok' if charged <= budget else 'bad'),
    ]


def step_prefill(captures):
    """Prefill tok/s over steps that carried only prefill (model_step_latency.jsonl)."""
    tokens = ms = 0
    for line in _lines(captures, 'model_step_latency.jsonl'):
        s = json.loads(line)
        if s.get('prefill_tokens') and not s.get('decode_requests'):
            tokens += s['prefill_tokens']
            ms += s.get('step_ms', 0)
    return round(1000 * tokens / ms) if ms else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spec')
    p.add_argument('out')
    p.add_argument('--unblind', action='store_true', help='include per-case grades (after grading ends)')
    args = p.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    root = Path(spec['root'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    manifest = read(root / spec['manifest']) if spec.get('manifest') else None
    group_field = spec.get('group_field', 'group')
    groups = {r['id']: r.get(group_field) for r in manifest['rows']} if manifest else {}
    total = len(manifest['rows']) if manifest else None
    marks, judged = grades(root, spec, args.unblind)
    runs, cases, checks, settings = [], [], [], []
    for arm in spec['arms']:
        name, directory = arm['arm'], root / arm['dir']
        records = []
        for path in sorted((directory / 'cases').glob('*/result.json')) if directory.is_dir() else ():
            record = read(path)
            if record:
                records.append(record)
        done = [r for r in records if r.get('status') == 'completed']
        errors = [r for r in records if r.get('status') != 'completed']
        status_file = read(directory / 'status.json')
        process = read(directory / 'process.json')
        # A live server wins over an older status.json (a resumed arm keeps its earlier run's file).
        if process and alive(process.get('server_pid')):
            status = 'running'
        elif status_file:
            status = 'done' if status_file.get('status') == 'done' else 'stopped'
        elif directory.is_dir():
            status = 'stopped'
        else:
            status = 'queued'
        started = min((r['started'] for r in records if r.get('started')), default=None)
        ended = max((r['ended'] for r in done if r.get('ended')), default=None)
        until = now if status == 'running' else ended
        elapsed = (until - started) if started and until else None
        prompt = sum(r.get('prompt_tokens', 0) for r in done)
        output = sum((r.get('usage') or {}).get('completion_tokens', 0) for r in done)
        rate = len(done) / elapsed * 3600 if elapsed and done else None
        expected = total if total is not None else len(records)
        remaining = max(0, expected - len(done))
        eta = now + remaining / rate * 3600 if status == 'running' and rate and remaining else None
        arm_marks = [marks[(name, r['id'])] for r in done if (name, r['id']) in marks]
        capture = sorted(root.glob(arm['kvmem_capture'])) if arm.get('kvmem_capture') else None
        runs.append(dict(
            experiment=spec['experiment'], title=spec['title'], arm=name, label=arm.get('label', name), status=status,
            total=expected, completed=len(done), errors=len(errors), started_at=iso(started),
            last_finished_at=iso(ended), snapshot_at=iso(now), eta_at=iso(eta),
            elapsed_hours=round(elapsed / 3600, 3) if elapsed else None,
            cases_per_hour=round(rate, 1) if rate else None,
            prompt_tok_s=round(prompt / elapsed) if elapsed and done else None,
            output_tok_s=round(output / elapsed, 1) if elapsed and done else None,
            step_prefill_tok_s=step_prefill(capture) if capture else None,
            mean_output_tokens=round(output / len(done)) if done else None,
            output_cap_hits=sum(r.get('finish_reason') == 'length' for r in done),
            graded=len(arm_marks), judged_questions=judged.get(name, 0) if isinstance(judged, dict) else judged,
            accuracy=round(100 * sum(arm_marks) / len(arm_marks), 2) if arm_marks else None))
        for r in records:
            usage = r.get('usage') or {}
            cases.append(dict(arm=name, id=r['id'], group=groups.get(r['id'], r.get(group_field)),
                              status=r.get('status'), started_at=iso(r.get('started')), ended_at=iso(r.get('ended')),
                              seconds=round(r['seconds'], 1) if r.get('seconds') is not None else None,
                              prompt_tokens=r.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'),
                              finish=r.get('finish_reason'), correct=marks.get((name, r['id']))))
        rows = [dict(check='Requests failed', value=str(len(errors)), target='0', state='ok' if not errors else 'bad'),
                dict(check='Answers cut at the output limit', value=str(runs[-1]['output_cap_hits']), target='0',
                     state='ok' if not runs[-1]['output_cap_hits'] else 'warn')]
        if capture and done and arm.get('layer_checks', True):
            rows += kvmem_checks(capture, done, int(arm.get('select_budget', 36864)))
        checks += [dict(arm=name, **row) for row in rows]
        settings += [dict(arm=name, setting=k, value=str(v)) for k, v in arm.get('settings', {}).items()]
    code = dict(label='snapshot.py', value=Path(__file__).read_text(), type='code', lang='python')
    source = dict(label='Input', value='%s: %s' % (spec['experiment'], ', '.join(a['dir'] for a in spec['arms'])))
    taken = dict(label='Snapshot', value=iso(now))
    for name, rows in (('runs', runs), ('cases', cases), ('checks', checks), ('settings', settings)):
        (out / (name + '.json')).write_text(json.dumps(dict(rows=rows, meta=[code, source, taken]), ensure_ascii=False))
    print(json.dumps(dict(out=str(out), runs=[{k: r[k] for k in ('arm', 'status', 'completed', 'errors', 'cases_per_hour')}
                                              for r in runs], cases=len(cases), checks=len(checks))))


if __name__ == '__main__':
    main()
