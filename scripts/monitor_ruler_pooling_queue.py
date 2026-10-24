"""Check a RULER queue periodically; wake its existing Codex chat on findings.

This monitor only reads experiments. Repairs run in the existing chat under its
normal permissions, rather than in a separate unattended model process.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time
import uuid


ARMS = ("max", "query_only", "max_guided")
FATAL = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|"
                   r"Fatal Python error|EngineCore encountered an issue")


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as source:
        source.seek(0, 2)
        source.seek(max(0, source.tell() - 65536))
        return source.read().decode("utf-8", errors="replace")


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def inspect(root, now=None, is_alive=alive, disk_usage=shutil.disk_usage):
    now = time.time() if now is None else now
    queue = read(root / "status.json", {})
    issues, arms = [], {}
    status = queue.get("status", "missing")
    if status in ("failed", "missing"):
        issues.append("Queue status: " + status)
    process = read(root / "queue-process.json", {})
    if status not in ("completed", "failed") and not is_alive(process.get("pid")):
        issues.append("Queue controller is not alive")
    for arm in ARMS:
        campaign = root / arm
        state = read(campaign / "status.json", {})
        progress = read(campaign / "run/answers/progress.json", {})
        admission = read(campaign / "run/answers/queue.json", {})
        arms[arm] = dict(status=state.get("status", "waiting"),
                         completed=progress.get("completed", 0),
                         expected=progress.get("expected", state.get("cases", 260)),
                         counts=admission.get("counts", {}))
        if state.get("status") == "failed":
            issues.append(arm + ": campaign failed; inspect status and controller.log")
        failure = campaign / "run/answers/failure.json"
        if failure.exists():
            issues.append(arm + ": client failure.json exists")
        if queue.get("arm") != arm or status != "running":
            continue
        for log in ("controller.log", "run/server.log", "run/client.log"):
            if FATAL.search(log_tail(campaign / log)):
                issues.append(arm + ": possible fatal error in " + log + "; inspect before restarting")
        if state.get("status") != "completed" and not is_alive(queue.get("controller_pid")):
            issues.append(arm + ": campaign controller is not alive")
        if state.get("status") == "testing":
            if not is_alive(state.get("server_pid")):
                issues.append(arm + ": server is not alive")
            done = progress.get("completed", 0)
            expected = arms[arm]["expected"]
            if done < expected:
                if not is_alive(state.get("client_pid")):
                    issues.append(arm + ": client is not alive")
                latest = progress.get("at", state.get("at", now))
                if now - latest > 1200:
                    issues.append(arm + ": no case completed for over 20 minutes; investigate before restarting")
    free = disk_usage(root).free
    if free < 10 * 1024**3:
        issues.append("Experiment disk has less than 10 GiB free")
    complete = status == "completed" and all(arms[a]["status"] == "completed" for a in ARMS)
    if status == "completed" and not complete:
        issues.append("Queue claims completion but an arm is incomplete")
    return dict(at=now, queue_status=status, active_arm=queue.get("arm"),
                arms=arms, issues=issues, disk_free_bytes=free, complete=complete)


def findings(snapshot, previous):
    events = list(snapshot["issues"])
    for arm, value in snapshot["arms"].items():
        old = previous.get("arms", {}).get(arm, {})
        if value["status"] == "completed" and old.get("status") != "completed":
            events.append(arm + ": completed; verify scores and the next arm")
    if snapshot["complete"] and not previous.get("complete"):
        events.append("All three arms completed; verify final results and stop monitoring")
    return events


async def wake(thread, socket, message, submission_key):
    import websockets

    async with websockets.unix_connect(str(socket), open_timeout=10, close_timeout=3) as ws:
        sequence = 0

        async def rpc(method, params):
            nonlocal sequence
            sequence += 1
            request_id = sequence
            await ws.send(json.dumps(dict(id=request_id, method=method, params=params)))
            async with asyncio.timeout(20):
                while True:
                    response = json.loads(await ws.recv())
                    if response.get("id") == request_id:
                        if "error" in response:
                            raise RuntimeError(method + ": " + json.dumps(response["error"]))
                        return response["result"]

        await rpc("initialize", dict(clientInfo=dict(name="kvmem_experiment_monitor", version="1"),
                                     capabilities=dict(experimentalApi=True)))
        await ws.send(json.dumps(dict(method="initialized")))
        queued = await rpc("thread/queue/list", dict(threadId=thread, limit=100))
        # Do not pile up monitoring turns while the user or a repair is busy.
        existing = next((item for item in queued["data"] if any(
            part.get("text", "").startswith("【KVMem RULER 定时巡检】")
            for part in item["input"])), None)
        # A stable client message ID makes retrying an uncertain response safe.
        if existing:
            queued_id = existing["id"]
        else:
            result = await rpc("thread/queue/add", dict(threadId=thread,
                               clientUserMessageId=submission_key,
                               input=[dict(type="text", text=message)]))
            queued_id = result["queuedSubmission"]["id"]
        info = await rpc("thread/read", dict(threadId=thread, includeTurns=False))
        state = info["thread"]["status"]["type"]
        started = False
        if state == "idle":
            try:
                # Preserve the order of any user messages already queued.
                await rpc("thread/queue/start", dict(threadId=thread))
                started = True
            except RuntimeError:
                # Another user turn may start between read and start. Its normal
                # queue drain will consume this message after that turn finishes.
                fresh = await rpc("thread/read", dict(threadId=thread, includeTurns=False))
                if fresh["thread"]["status"]["type"] != "active":
                    raise
        elif state != "active":
            raise RuntimeError("Chat is not loaded/available: " + state)
        return dict(queued_submission_id=queued_id, started=started, thread_status=state)


def run(args, *, inspector=inspect, event_finder=findings):
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        previous = read(args.state_dir / "last-check.json", {})
        pending = read(args.state_dir / "pending.json")
        while not stopped and not (args.state_dir / "STOP").exists():
            try:
                snapshot = inspector(args.root)
            except Exception as error:
                snapshot = dict(at=time.time(), issues=["Monitor inspection failed: " + repr(error)],
                                arms={}, complete=False, queue_status="inspection_error")
            events = event_finder(snapshot, previous)
            if events and pending is None:
                pending = dict(key=str(uuid.uuid4()), events=events,
                               message=args.instructions.read_text() + "\n\n本次巡检发现：\n" +
                               json.dumps(events, ensure_ascii=False) + "\n\n快照：\n" +
                               json.dumps(snapshot, ensure_ascii=False))
                save(args.state_dir / "pending.json", pending)
            delivery_error = None
            if pending is not None:
                try:
                    receipt = asyncio.run(wake(args.thread, args.socket, pending["message"], pending["key"]))
                    save(args.state_dir / "last-delivery.json", dict(at=time.time(), **receipt,
                         client_message_id=pending["key"], events=pending["events"]))
                    (args.state_dir / "pending.json").unlink()
                    pending = None
                except Exception as error:
                    delivery_error = repr(error)
            save(args.state_dir / "last-check.json", snapshot)
            next_check = time.time() + (60 if delivery_error else args.interval)
            terminal = snapshot["complete"] and pending is None
            save(args.state_dir / "status.json", dict(
                status="completed" if terminal else "monitoring", pid=os.getpid(),
                interval_seconds=args.interval, checked_at=time.time(),
                next_check_at=None if terminal else next_check,
                delivery_error=delivery_error, findings=events,
                active_arm=snapshot.get("active_arm"), arms=snapshot["arms"]))
            with (args.state_dir / "checks.jsonl").open("a") as log:
                log.write(json.dumps(dict(**snapshot, delivery_error=delivery_error), ensure_ascii=False) + "\n")
            previous = snapshot
            if args.once or terminal:
                return
            while not stopped and time.time() < next_check and not (args.state_dir / "STOP").exists():
                time.sleep(min(5, max(0, next_check - time.time())))
        save(args.state_dir / "status.json", dict(status="stopped", at=time.time(), pid=os.getpid()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("The interval must be at least 60 seconds")
    run(args)
