import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import monitor_hybrid_suite_queue as monitor
import monitor_ruler_pooling_queue as base


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, active="ruler"):
    plan = dict(arms=[dict(name=n, root=str(root/n)) for n in monitor.EXPECTED],
                grading=[dict(suite=s, output=str(root/("grading-"+n))) for n,s in
                         (("lme", "longmemeval_s"), ("mab", "memoryagentbench"))])
    write(root/"queue-plan.json", plan)
    write(root/"status.json", dict(status="running", active=active, at=100))
    write(root/"queue-process.json", dict(pid=10))
    for n, count in monitor.EXPECTED.items():
        write(root/n/"status.json", dict(status="testing" if n == active else "prepared", cases=count))
    for name in ("controller-process.json", "run/server-process.json", "run/client-process.json"):
        write(root/active/name, dict(pid=20))
    progress = "progress.json" if active == "ruler" else "summary.json"
    write(root/active/"run/answers"/progress, dict(completed=7, expected=monitor.EXPECTED[active], errors=0))
    work(root, active, 200)


def work(root, active, tokens):
    write(root/active/"run/server/capture/public_work.json",
          dict(counter_id="one", pid=30, completed=dict(public_prefill_tokens=tokens)))


def check(root, **kwargs):
    return monitor.inspect(root, resource_probe=lambda _: dict(disk_free_bytes=100*1024**3, gpu=[]),
                           probe=kwargs.pop("probe", lambda r: dict(pid=r.get("pid"), state="alive")), **kwargs)


def finish_arm(root, n):
    count = monitor.EXPECTED[n]
    write(root/n/"status.json", dict(status="completed", cases=count))
    p = "progress.json" if n == "ruler" else "summary.json"
    write(root/n/"run/answers"/p, dict(completed=count, expected=count))
    write(root/n/"execution-audit.json", dict(status="passed", cases=count))
    write(root/n/"run/cleanup.json", dict(remaining_owned_groups=[], gpu_processes=[]))
    if n == "ruler":
        write(root/n/"scores.json", dict(macro_score=80))


@pytest.mark.parametrize("active", ["ruler", "lme", "mab"])
def test_suites_use_their_actual_progress_schemas_and_stay_quiet(tmp_path, active):
    fixture(tmp_path, active)
    write(tmp_path/"efficiency-gate-failure.json", dict(status="failed"))
    snap = check(tmp_path, now=1000)
    assert not snap["issues"]
    assert snap["arms"][active]["completed"] == 7
    assert monitor.findings(snap, {}) == []
    assert not snap["complete"]


def test_long_prefill_or_decode_without_completed_answer_is_not_a_stall(tmp_path):
    fixture(tmp_path)
    old = check(tmp_path, now=1000)
    work(tmp_path, "ruler", 2000)
    current = check(tmp_path, now=2500, previous=old)
    assert not current["issues"]
    assert current["arms"]["ruler"]["last_progress_at"] == 2500


def test_unchanged_work_for_twenty_minutes_requires_investigation_not_restart(tmp_path):
    fixture(tmp_path)
    old = check(tmp_path, now=1000)
    current = check(tmp_path, now=2200, previous=old)
    assert len(current["issues"]) == 1
    assert "never kill" in current["issues"][0]
    assert json.loads((tmp_path/"status.json").read_text())["status"] == "running"


def test_visibility_denied_is_not_reported_as_a_dead_process(tmp_path):
    fixture(tmp_path)
    snap = check(tmp_path, probe=lambda r: dict(state="visibility_denied"))
    assert all("verify host visibility" in i for i in snap["issues"])
    assert not any("not alive" in i for i in snap["issues"])


def test_failures_are_reported_even_when_processes_are_alive(tmp_path):
    fixture(tmp_path, "mab")
    write(tmp_path/"mab/run/answers/summary.json", dict(completed=7, expected=1116, errors=1))
    path = tmp_path/"mab/run/server.log"
    path.write_text("CUDA out of memory")
    snap = check(tmp_path)
    assert len(snap["issues"]) == 2
    assert not snap["complete"]


def test_does_not_probe_released_server_during_final_audit(tmp_path):
    fixture(tmp_path)
    write(tmp_path/"ruler/run/answers/progress.json", dict(completed=130, expected=130))
    snap = check(tmp_path)
    assert set(snap["processes"]) == {"queue", "ruler/controller"}


def test_inference_completion_does_not_end_monitor_before_flash_grading(tmp_path):
    fixture(tmp_path)
    for n in monitor.EXPECTED:
        finish_arm(tmp_path, n)
    write(tmp_path/"status.json", dict(status="grading", active="longmemeval_s", at=1000))
    snap = check(tmp_path, now=1001)
    assert not snap["issues"]
    assert not snap["complete"]
    assert len(monitor.findings(snap, {})) == 3
    assert monitor.findings(snap, snap) == []


@pytest.mark.parametrize("remaining", [{}, []])
def test_cleanup_accepts_empty_mapping_and_legacy_list(tmp_path, remaining):
    fixture(tmp_path)
    finish_arm(tmp_path, "ruler")
    write(tmp_path/"ruler/run/cleanup.json", dict(remaining_owned_groups=remaining, gpu_processes=[]))
    snap = check(tmp_path)
    assert snap["arms"]["ruler"]["audit_and_cleanup_passed"]
    assert not snap["issues"]


@pytest.mark.parametrize("remaining", [None, False, 0, "", {"20": [21]}, [20]])
def test_cleanup_rejects_missing_malformed_or_nonempty_group_record(tmp_path, remaining):
    fixture(tmp_path)
    finish_arm(tmp_path, "ruler")
    write(tmp_path/"ruler/run/cleanup.json", dict(remaining_owned_groups=remaining, gpu_processes=[]))
    assert not check(tmp_path)["arms"]["ruler"]["audit_and_cleanup_passed"]
    write(tmp_path/"ruler/run/cleanup.json", dict(gpu_processes=[]))
    assert not check(tmp_path)["arms"]["ruler"]["audit_and_cleanup_passed"]


def test_full_completion_requires_audit_cleanup_and_full_flash_scores(tmp_path):
    fixture(tmp_path)
    for n in monitor.EXPECTED:
        finish_arm(tmp_path, n)
    write(tmp_path/"status.json", dict(status="completed"))
    assert not check(tmp_path)["complete"]
    for n in ("lme", "mab"):
        write(tmp_path/("grading-"+n)/"status.json", dict(status="completed", judge="deepseek-flash"))
    write(tmp_path/"grading-lme/scores.json", dict(status="completed", arms=dict(hybrid=dict(count=102))))
    write(tmp_path/"grading-mab/scores.json", dict(complete=True, count=1116))
    snap = check(tmp_path)
    assert snap["complete"]
    assert not snap["issues"]
    assert len(monitor.findings(snap, {})) == 6
    assert monitor.findings(snap, snap) == []
    write(tmp_path/"mab/run/cleanup.json", dict(remaining_owned_groups=[99], gpu_processes=[99]))
    assert not check(tmp_path)["complete"]


def test_flash_timeout_requests_investigation_and_recovery_not_new_inference(tmp_path):
    fixture(tmp_path)
    write(tmp_path/"status.json", dict(status="grading", active="longmemeval_s", at=1000))
    snap = check(tmp_path, now=2201)
    assert len(snap["issues"]) == 1
    assert "grading progress" in snap["issues"][0]


def test_recovery_is_announced_once(tmp_path):
    fixture(tmp_path)
    snap = check(tmp_path)
    assert len(monitor.findings(snap, dict(issues=["past failure"]))) == 1
    assert monitor.findings(snap, snap) == []


def test_process_identity_detects_pid_reuse():
    assert monitor.process(dict(pid=os.getpid()))["state"] == "alive"
    assert monitor.process(dict(pid=os.getpid(), command=["wrong"]))["state"] == "identity_changed"


def test_failed_wake_is_retried_with_same_message_and_completion_waits(tmp_path, monkeypatch):
    args = SimpleNamespace(state_dir=tmp_path, root=tmp_path, thread="current",
                           socket=Path("/unused"), instructions=tmp_path/"instructions.txt",
                           interval=1200, once=True)
    args.instructions.write_text("【KVMem RULER 定时巡检】current queue")
    snap = dict(issues=[], arms={}, complete=True)
    keys = []

    async def fail(thread, socket, message, key):
        keys.append(key)
        raise RuntimeError("temporary socket failure")

    monkeypatch.setattr(base, "wake", fail)
    base.run(args, inspector=lambda _: snap, event_finder=lambda *_: ["complete"])
    assert (tmp_path/"pending.json").exists()
    assert json.loads((tmp_path/"status.json").read_text())["status"] == "monitoring"

    async def success(thread, socket, message, key):
        keys.append(key)
        return dict(queued_submission_id="one", started=False, thread_status="active")

    monkeypatch.setattr(base, "wake", success)
    base.run(args, inspector=lambda _: snap, event_finder=lambda *_: [])
    assert len(keys) == 2 and keys[0] == keys[1]
    assert not (tmp_path/"pending.json").exists()
    assert json.loads((tmp_path/"status.json").read_text())["status"] == "completed"
