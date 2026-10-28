"""Read-only local monitor for the finite hybrid RULER/LME/MAB + Flash queue.

Repairs run in the same existing conversation, never in a new agent.
No experiment, answer, runtime, GPU process or judge call is mutated here.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import monitor_ruler_pooling_queue as base


EXPECTED = {"ruler": 130, "lme": 102, "mab": 1116}
TERMINAL = {"completed", "failed"}


def read(path, default=None):
    # Tolerate a transient partial write from older clients, but surface
    # persistent malformed state rather than assuming completion.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.05)


def mtime(path):
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0


def process(record):
    """Check zombies and PID reuse; denied visibility is not proof of death."""
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return dict(pid=pid, state="missing")
    directory = Path("/proc") / str(pid)
    try:
        state = (directory / "stat").read_text().rsplit(")", 1)[1].split()[0]
        if state in ("Z", "X"):
            return dict(pid=pid, state="exited")
        command = (directory / "cmdline").read_bytes().decode().split("\0")[:-1]
        if record.get("command") and command != record["command"]:
            return dict(pid=pid, state="identity_changed")
        return dict(pid=pid, state="alive")
    except PermissionError:
        return dict(pid=pid, state="visibility_denied")
    except FileNotFoundError:
        return dict(pid=pid, state="not_visible_or_exited")


def resources(root):
    result = dict(disk_free_bytes=shutil.disk_usage(root).free,
                  system_disk_free_bytes=shutil.disk_usage("/").free)
    memory = {parts[0].rstrip(":"): int(parts[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if len(parts := line.split()) >= 2 and parts[1].isdigit()}
    result["memory_available_bytes"] = memory.get("MemAvailable")
    result["gpu"] = []
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits"], text=True, timeout=10,
            stderr=subprocess.DEVNULL)
        for row in output.splitlines():
            pid, used, device = (part.strip() for part in row.split(",", 2))
            result["gpu"].append(dict(pid=int(pid), memory_mib=int(used), device=device))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["gpu_inspection_error"] = type(error).__name__
    return result


def inspect(root, *, previous=None, now=None, probe=process, resource_probe=resources):
    now = time.time() if now is None else now
    previous = previous or {}
    plan = read(root / "queue-plan.json", {})
    state = read(root / "status.json", {})
    status, active = state.get("status", "missing"), state.get("active")
    issues, arms, grades, processes = [], {}, {}, {}

    def require_process(label, record):
        value = probe(record)
        processes[label] = value
        if value["state"] != "alive":
            issues.append(label + ": process " + value["state"] +
                          "; verify host visibility/identity before any restart")

    if status in ("failed", "missing"):
        issues.append("Queue status: " + status + "; inspect status.json and queue.log")
    if status not in TERMINAL:
        require_process("queue", read(root / "queue-process.json", {}))
    if {a.get("name") for a in plan.get("arms", [])} != set(EXPECTED):
        issues.append("Queue plan does not contain the expected three suites")

    for arm in plan.get("arms", []):
        name, campaign = arm["name"], Path(arm["root"])
        s = read(campaign / "status.json", {})
        answer = campaign / "run/answers"
        p = read(answer / "progress.json")
        if p is None:
            p = read(answer / "summary.json", {})
        admission = read(answer / "queue.json", {})
        done, expected = p.get("completed", 0), p.get("expected", s.get("cases", EXPECTED.get(name)))
        value = dict(root=str(campaign), status=s.get("status", "missing"),
                     completed=done, expected=expected, counts=admission.get("counts", {}))
        arms[name] = value
        if expected != EXPECTED.get(name):
            issues.append(name + ": cohort size changed")
        if s.get("status") in ("failed", None) or p.get("errors"):
            issues.append(name + ": campaign/client failure; preserve attempted answers")
        for location in (campaign / "failure.json", answer / "failure.json"):
            if location.exists():
                issues.append(name + ": " + str(location.relative_to(campaign)) + " exists")
        if s.get("status") == "completed":
            audit = read(campaign / "execution-audit.json", {})
            cleanup = read(campaign / "run/cleanup.json", {})
            # stop_owned_groups returns a PGID -> surviving PID list mapping.
            # Earlier controllers also emitted lists. Missing/null/false fields
            # are not evidence of cleanup, even though they are falsey.
            remaining = cleanup.get("remaining_owned_groups")
            groups_cleared = isinstance(remaining, (dict, list)) and not remaining
            audited = (audit.get("status") == "passed" and audit.get("cases") == expected and
                       groups_cleared and cleanup.get("gpu_processes") == [] and
                       done == expected)
            if name == "ruler":
                audited = audited and (campaign / "scores.json").exists()
            value["audit_and_cleanup_passed"] = audited
            if not audited:
                issues.append(name + ": claimed completion lacks full cohort, passed audit, scores or GPU cleanup")
        if name != active or status not in ("running", "waiting") or s.get("status") in TERMINAL:
            continue
        require_process(name + "/controller", read(campaign / "controller-process.json", {}))
        if s.get("status") in ("loading", "testing") and done < expected:
            require_process(name + "/server", read(campaign / "run/server-process.json", {}))
            if s.get("status") == "testing":
                require_process(name + "/client", read(campaign / "run/client-process.json", {}))
        for log in ("controller.log", "run/server.log", "run/client.log"):
            if base.FATAL.search(base.log_tail(campaign / log)):
                issues.append(name + ": possible fatal error in " + log + "; inspect before recovery")
        work = read(campaign / "run/server/capture/public_work.json", {})
        counters = {key: work.get("completed", {}).get(key, 0) for key in (
            "public_batches", "public_prefill_tokens", "public_non_prefill_tokens")}
        value["work"] = dict(counter_id=work.get("counter_id"), **counters)
        value["engine_pid"] = work.get("pid")
        # Non-prefill tokens include verification, not just accepted output.
        old = previous.get("arms", {}).get(name, {})
        changed = (old.get("root") != str(campaign) or old.get("status") != value["status"] or
                   old.get("completed") != done or old.get("work") != value["work"])
        last = now if changed else old.get("last_progress_at", now)
        # Recurring zero-throughput messages alone do not count as progress.
        if not work:
            rates = re.findall(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+)",
                               base.log_tail(campaign / "run/server.log"))
            if rates and max(map(float, rates[-1])) > 0 and now - mtime(campaign / "run/server.log") < 60:
                last = now
        value["last_progress_at"] = last
        if now - last >= 1200:
            issues.append(name + ": no answer or forward progress observed for 20 minutes; inspect prefill/decode, "
                          "recall and audit work before deciding recovery; never kill on this signal alone")

    for grade in plan.get("grading", []):
        name, output = grade["suite"], Path(grade["output"])
        s = read(output / "status.json", {})
        gs = s.get("status", "grading" if active == name and status == "grading" else "waiting")
        value = dict(output=str(output), status=gs)
        grades[name] = value
        if gs == "failed":
            issues.append(name + ": Flash grading failed; preserve cached judgments")
        value["progress"] = [read(output / f, {}) for f in ("lme-progress.json", "mab-prefetch-progress.json")]
        if active == name and status in ("grading", "waiting_grade"):
            activity = max([mtime(output / f) for f in ("lme-progress.json", "mab-prefetch-progress.json", "scores.json")] +
                           [mtime(root / ("grading-" + name + ".log")), float(state.get("at", now))])
            value["last_activity_at"] = activity
            if now - activity >= 1200:
                issues.append(name + ": no grading progress for 20 minutes; inspect judge calls and grader process")
            if base.FATAL.search(base.log_tail(root / ("grading-" + name + ".log"))):
                issues.append(name + ": possible fatal error in grading log")
        if gs == "completed":
            scores = read(output / "scores.json", {})
            if name == "longmemeval_s":
                valid = (scores.get("status") == "completed" and not scores.get("partial") and
                         scores.get("arms", {}).get("hybrid", {}).get("count") == EXPECTED["lme"])
            else:
                valid = scores.get("complete") is True and scores.get("count") == EXPECTED["mab"]
            value["full_scores_passed"] = valid and s.get("judge") == "deepseek-flash"
            if not value["full_scores_passed"]:
                issues.append(name + ": incomplete or non-Flash grading verdict")

    res = resource_probe(root)
    for field, limit in (("disk_free_bytes", 10 * 1024**3), ("system_disk_free_bytes", 2 * 1024**3),
                         ("memory_available_bytes", 2 * 1024**3)):
        if res.get(field) is not None and res[field] < limit:
            issues.append("Resource pressure: " + field + " below " + str(limit))
    if res.get("gpu_inspection_error"):
        issues.append("GPU read-only inspection failed: " + res["gpu_inspection_error"])
    for owner in res.get("gpu", []):
        owner["matches_active_engine"] = owner["pid"] == arms.get(active, {}).get("engine_pid")
    # Original gate failures are preserved; queue-process.json binds the later
    # passed efficiency review and resolution. Do not flag old failed attempts.
    complete = (status == "completed" and set(arms) == set(EXPECTED) and
                all(a.get("audit_and_cleanup_passed") for a in arms.values()) and
                set(grades) == {"longmemeval_s", "memoryagentbench"} and
                all(g.get("full_scores_passed") for g in grades.values()))
    if status == "completed" and not complete:
        issues.append("Queue claims completion but inference/Flash/audit/cleanup is incomplete")
    return dict(at=now, root=str(root), queue_status=status, active_arm=active, arms=arms,
                grades=grades, processes=processes, issues=issues, complete=complete, **res)


def findings(snapshot, previous):
    events = list(snapshot["issues"])
    for collection in ("arms", "grades"):
        for name, value in snapshot.get(collection, {}).items():
            old = previous.get(collection, {}).get(name, {})
            if value["status"] == "completed" and old.get("status") != "completed":
                events.append(name + ": completed; verify full results and the next stage")
    if previous.get("issues") and not snapshot["issues"]:
        events.append("Previously reported issues cleared; verify recovery and preserved answers")
    if snapshot["complete"] and not previous.get("complete"):
        events.append("All inference and Flash grading completed; verify final comparison and stop monitoring")
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "state-dir", "instructions", "socket"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--interval", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("The interval must be at least 60 seconds")
    if args.inspect_only:
        print(json.dumps(inspect(args.root), ensure_ascii=False, indent=2))
    else:
        base.run(args, inspector=lambda root: inspect(root, previous=read(args.state_dir / "last-check.json", {})),
                 event_finder=findings)
