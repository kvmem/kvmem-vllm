"""Exercise text compaction through both public agent protocols."""
import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.deepswe import claude_bridge as claude
from scripts import dsh_bridge as dsh
from scripts.terminalbench_query import TaskQuery
from vllm_kvmem.apps.compaction import CompactionController, DIRECTIVE, History, LABEL, read_snapshot


def message(role, content):
    return {"role": role, "content": content}


def text_of(msg):
    content = msg["content"]
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def policy(limit=800, quota=40, keep=2):
    return CompactionController(limit, quota, .80, .55, keep, 512)


def test_long_history_uses_bounded_private_calls_preserves_groups_and_accepts_edits():
    async def run():
        calls = []
        async def count(payload):
            return sum(len(text_of(m)) + 2 for m in payload["messages"])
        async def summarize(payload, quota):
            assert await count(payload) + quota <= controller.limit
            assert payload["messages"][-1] == message("user", DIRECTIVE)
            calls.append(payload)
            return "prior facts"
        controller = policy()
        tool = lambda ident: dict(type="tool_use", id=ident, name="read", input={})
        result = lambda ident: dict(type="tool_result", tool_use_id=ident, content="ok")
        raw = [message("user", "initial goal")]
        raw += [message("assistant", str(i) + "x"*95) for i in range(15)]
        raw += [message("assistant", [tool("p"), tool("q")]),
                message("user", [result("p")]), message("user", [result("q")]),
                message("user", "important instruction"),
                message("assistant", [tool("a"),tool("b")]), message("user", [result("b")])]
        payload = dict(model="model", system="control", messages=raw, tools=["schema"])
        source = len(raw)-3
        def history(p):
            return History(p, "template", source, 2)
        view = await controller.project(history(payload), None, 50, count, summarize)
        assert len(calls) > 1
        assert view.snapshot and view.payload["messages"][-3:] == raw[-3:]
        assert raw[0] == view.payload["messages"][0] and source in view.indices
        for req in [*calls, view.payload]:
            assert req["system"] == "control" and req["tools"] == ["schema"]
            text = text_of(message("user", req["messages"]))
            if '"id": "p"' in text:
                assert '"tool_use_id": "p"' in text and '"tool_use_id": "q"' in text
        previous = len(calls)
        same = await controller.project(history(payload), view.snapshot, 50, count, summarize)
        assert same.payload == view.payload and len(calls) == previous
        raw[2] = message("assistant", "edited fact")
        changed = await controller.project(history(payload), view.snapshot, 50, count, summarize)
        assert changed.snapshot.digest != view.snapshot.digest and len(calls) > previous
        assert not History(dict(payload, system="other"), "template").matches(changed.snapshot)
        assert not History(payload, "new template").matches(changed.snapshot)
    asyncio.run(run())


def test_oversize_indivisible_group_is_rejected_without_partial_summary():
    async def run():
        raw = [message("user", "task"), message("assistant", [
            dict(type="tool_use", name="lookup", id="a", input={})]),
            message("user", [dict(type="tool_result", tool_use_id="a", content="x"*1500)]),
            message("user", "continue")]
        async def count(p):
            return sum(len(text_of(m)) for m in p["messages"])
        async def summarize(*_):
            pytest.fail("an oversized tool group cannot be silently split")
        with pytest.raises(ValueError, match="logical context"):
            await policy(keep=1).project(History(dict(messages=raw), "t", keep_recent=1),
                                         None, 40, count, summarize)
    asyncio.run(run())


class BytesTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        return 256


class MessagesProtocol:
    """Small structured renderer for host CI. The integration uses the production Planner."""
    template = "test-message-template-v1"

    def __init__(self, path):
        pass

    def __call__(self, *, request=None, op=None, text=None, tools=None):
        if op == "parse":
            content = text.partition("</think>")[2]
            if content.startswith("<tool_call>"):
                return dict(reasoning="", content="", tool_calls=[dict(function=dict(name="read", arguments="{}"))])
            return dict(reasoning="", content=content, tool_calls=[])
        messages = []
        for m in request["messages"]:
            role, content = m["role"], m["content"]
            if isinstance(content, list):
                if any(b["type"] == "tool_result" for b in content):
                    role = "tool"
                content = "".join(b.get("text", json.dumps(b)) for b in content)
            messages.append(message(role, content))
        prompt = request.get("system", "")
        control = len(prompt)
        spans = []
        for i, m in enumerate(messages):
            begin = len(prompt)
            prompt += "<|im_start|>" + m["role"] + "\n"
            content_begin = len(prompt)
            prompt += m["content"]
            content_end = len(prompt)
            prompt += "<|im_end|>\n"
            spans.append(dict(index=i, begin=begin, end=len(prompt), role=m["role"],
                              content_begin=content_begin, content_end=content_end))
        return dict(prompt=prompt+"<|im_start|>assistant\n<think>\n", control_end=control,
                    messages=messages, spans=spans, tools=request.get("tools", []), pins=[],
                    query_index=next((s["index"] for s in reversed(spans) if s["role"]=="user"), None))


def trace():
    msgs = [message("user", "root goal"), message("assistant", "intro"), message("user", "task B")]
    for i in range(8):
        msgs += [message("assistant", "observed " + str(i) + " abc"*27), message("user", "next")]
    return msgs


class Backend:
    def __init__(self):
        self.private, self.public, self.fail, self.fail_public = [], [], None, False
        self.public_text = "</think>visible answer"
        self.hold_private = None

    async def handle(self, request):
        body = json.loads(request.content)
        ids = body["prompt"]
        prompt = bytes(i for i in ids if i < 256).decode()
        is_summary = DIRECTIVE in prompt
        if is_summary:
            self.private.append(body)
            if self.hold_private:
                entered, release = self.hold_private
                entered.set()
                await release.wait()
            begin, end = body["kvmem_query_span"]
            assert bytes(ids[begin:end]).decode() == DIRECTIVE
            assert not body.get("stream")
            text = "</think>private handoff" if self.fail != "tool" else "</think><tool_call>"
            finish = "length" if self.fail == "length" else "stop"
        else:
            self.public.append(body)
            if self.fail_public:
                return httpx.Response(503, text="injected visible failure")
            text, finish = self.public_text, "stop"
        output = list(text.encode())
        choice = dict(text=text, token_ids=output, finish_reason=finish)
        usage = dict(prompt_tokens=len(ids), completion_tokens=len(output),
                     total_tokens=len(ids)+len(output))
        native = dict(id="test", choices=[choice], usage=usage)
        if body.get("stream"):
            events = "data: " + json.dumps(native) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=events.encode())
        return httpx.Response(200, json=native)


def setup_bridge(kind, tmp_path, monkeypatch, backend, mode="gateway"):
    async_client = httpx.AsyncClient
    transport = async_client(transport=httpx.MockTransport(backend.handle))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: transport)
    module = claude if kind == "anthropic" else dsh
    monkeypatch.setattr(module, "Protocol", MessagesProtocol)
    tasks = tmp_path/"tasks"
    tasks.write_text("task\n")
    args = SimpleNamespace(protocol=None, model="m", tasks=tasks, run_id="run", backend="http://backend",
        backend_kind="kvmem", max_model_len=1300, gen_budget=512, cpu_admission_gib=4,
        output=tmp_path/"journal", max_sessions=4, compaction=mode, compaction_keep_messages=2,
        compaction_summary_tokens=48)
    app = module.create_app(args, BytesTokenizer())
    client = async_client(transport=httpx.ASGITransport(app=app), base_url="http://bridge")
    return client, transport, args


@pytest.mark.parametrize("kind", ["anthropic", "openai"])
def test_gateway_full_history_stream_retry_count_and_recovery(kind, tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        client, transport, args = setup_bridge(kind, tmp_path, monkeypatch, backend)
        path = "/task/task/v1/messages" if kind == "anthropic" else "/v1/chat/completions"
        raw = trace()
        body = dict(model="m", messages=raw, max_tokens=100, stream=False)
        if kind == "openai":
            body["prompt_cache_key"] = "thread"
            source = TaskQuery("thread", "real-task", "root goal", "root goal")
            for i, msg in enumerate(raw):
                if msg["role"] == "user":
                    if msg["content"] == "task B":
                        source.real_user("task B", "task-update")
                    body["kvmem_query_provenance"] = source.proof(raw[:i+1])
        async with client, transport:
            if kind == "anthropic":
                before = (await client.post(path+"/count_tokens", json=body)).json()["input_tokens"]
                assert before > args.max_model_len and not backend.private
            backend.fail_public = True
            failed = await client.post(path, json=body)
            assert failed.status_code >= 400 and backend.private and backend.public
            total = len(backend.private)
            saved_path, = args.output.glob("*/compaction.json")
            snapshot = read_snapshot([saved_path])
            assert snapshot.text == "private handoff"
            backend.fail_public = False
            # Two full-history retries on the same owner reuse the committed summary.
            responses = await asyncio.gather(*(client.post(path, json=dict(body, stream=True)) for _ in range(2)))
            for r in responses:
                assert r.status_code == 200 and "visible answer" in r.text and "private handoff" not in r.text
                assert ("[DONE]" if kind == "openai" else "message_stop") in r.text
            assert len(backend.private) == total
            owner = backend.public[0]["session_id"]
            assert all(call["session_id"] == call["cache_salt"] == owner for call in backend.private + backend.public)
            sent = backend.public[-1]
            assert len(sent["prompt"])+sent["max_tokens"] <= args.max_model_len
            assert DIRECTIVE.encode() not in bytes(sent["prompt"]) and LABEL.encode() in bytes(sent["prompt"])
            assert body["messages"] == raw
            if kind == "anthropic":
                after = (await client.post(path+"/count_tokens",json=body)).json()["input_tokens"]
                assert after == len(sent["prompt"]) and after < before and len(backend.private) == total
            else:
                begin, end = sent["kvmem_query_span"]
                assert bytes(sent["prompt"][begin:end]) == b"task B"
                planner = claude.Planner(MessagesProtocol(None), BytesTokenizer(), args.gen_budget)
                restored = dsh.restore_trajectory(planner, owner, [args.output])
                assert restored.turns == 2 and list(restored.tokens) == sent["prompt"]
            plans = list(saved_path.parent.glob("[0-9][0-9][0-9][0-9][0-9]/plan.json"))
            assert json.loads(plans[-1].read_text())["compaction"] == json.loads(json.dumps(asdict(snapshot)))
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["length", "tool"])
def test_incomplete_private_result_never_becomes_visible_or_committed(mode, tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        backend.fail = mode
        client, transport, args = setup_bridge("anthropic", tmp_path, monkeypatch, backend)
        body = dict(model="m", messages=trace(), max_tokens=100, stream=True)
        async with client, transport:
            response = await client.post("/task/task/v1/messages", json=body)
            assert "event: error" in response.text and "private handoff" not in response.text
            assert not backend.public and not list(args.output.glob("*/compaction.json"))
            backend.fail = None
            retry = await client.post("/task/task/v1/messages", json=body)
            assert "event: message_stop" in retry.text and "visible answer" in retry.text
    asyncio.run(run())


def test_cancelled_summary_preserves_previous_branch(tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        client, transport, args = setup_bridge("anthropic", tmp_path, monkeypatch, backend)
        path = "/task/task/v1/messages"
        body = dict(model="m", messages=trace(), max_tokens=100)
        async with client, transport:
            assert (await client.post(path, json=body)).status_code == 200
            saved_path, = args.output.glob("*/compaction.json")
            original = saved_path.read_bytes()
            body["messages"][1] = message("assistant", "new branch")
            entered, release = asyncio.Event(), asyncio.Event()
            backend.hold_private = entered, release
            job = asyncio.create_task(client.post(path, json=body))
            await asyncio.wait_for(entered.wait(), 2)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job
            assert saved_path.read_bytes() == original and len(backend.public) == 1
            release.set()
            assert (await client.post(path, json=body)).status_code == 200
            assert saved_path.read_bytes() != original
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["anthropic", "openai"])
def test_agent_can_execute_streamed_tool_and_send_its_result(kind, tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        backend.public_text = "</think><tool_call>"
        client, transport, _ = setup_bridge(kind, tmp_path, monkeypatch, backend)
        msgs = trace()
        body = dict(model="m", messages=msgs, max_tokens=100, stream=True)
        anth = kind == "anthropic"
        path = "/task/task/v1/messages" if anth else "/v1/chat/completions"
        if anth:
            body["tools"] = [dict(name="read", input_schema=dict(type="object"))]
        else:
            body.update(prompt_cache_key="tools", tools=[
                dict(type="function", function=dict(name="read", parameters=dict(type="object")))])
        async with client, transport:
            wire = (await client.post(path, json=body)).text
            assert "private handoff" not in wire
            events = [json.loads(s[6:]) for s in wire.splitlines() if s.startswith("data: {")]
            if anth:
                block, = [e["content_block"] for e in events if e.get("content_block", {}).get("type")=="tool_use"]
                ident = block["id"]
                msgs.extend([message("assistant", [dict(type="tool_use", name="read", id=ident, input={})]),
                             message("user", [dict(type="tool_result", tool_use_id=ident, content="file contents")])])
            else:
                calls = [call for e in events for choice in e.get("choices", [])
                         for call in choice.get("delta", {}).get("tool_calls", []) if call.get("id")]
                ident = calls[0]["id"]
                msgs.extend([dict(role="assistant", content=None, tool_calls=[
                    dict(id=ident, type="function", function=dict(name="read", arguments="{}"))]),
                    dict(role="tool", tool_call_id=ident, content="file contents")])
            assert ident and backend.private
            backend.public_text = "</think>visible answer"
            reply = await client.post(path, json=dict(body, stream=False))
            assert reply.status_code == 200 and "visible answer" in reply.text
            assert b"file contents" in bytes(backend.public[-1]["prompt"])
    asyncio.run(run())


def test_framework_summary_is_an_ordinary_visible_turn(tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        client, transport, args = setup_bridge("anthropic", tmp_path, monkeypatch, backend, mode="framework")
        async with client, transport:
            path = "/task/task/v1/messages"
            for content in ("history and framework compaction instruction", "framework summary and next task"):
                body = dict(model="m", max_tokens=40, messages=[message("user", content)])
                assert "visible answer" in (await client.post(path, json=body)).text
                assert content.encode() in bytes(backend.public[-1]["prompt"])
            assert len(backend.public) == 2 and not backend.private
            assert not list(args.output.glob("*/compaction.json"))
    asyncio.run(run())


def test_length_budget_uses_exact_accepted_ids_and_matches_final_prefill():
    async def run():
        planner = claude.Planner(MessagesProtocol(None), BytesTokenizer(), 512)
        state = claude.Trajectory()
        root = message("user", "initial task")
        first = dict(model="m", messages=[root])
        prepared = planner.prepare(first, state)
        displayed = "many display characters for two accepted token IDs"
        planner.commit(state, prepared, first, dict(content=[dict(type="text", text=displayed)]), [23, planner.end])
        following = dict(model="m", messages=[root, message("assistant", displayed), message("user", "continue")])
        bridge = claude.BridgeCompaction(policy(), None, planner, None, asyncio.to_thread, None)
        actual = planner.prepare(following, state)
        assert actual[5] and await bridge.count(following, state) == len(actual[0])
        assert await bridge.count(following) != len(actual[0])
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_concurrent_first_turns_get_one_summary_and_two_journal_entries(kind, tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        client, transport, args = setup_bridge(kind, tmp_path, monkeypatch, backend)
        body = dict(model="m", messages=trace(), max_tokens=100, prompt_cache_key="concurrent")
        path = "/task/task/v1/messages" if kind == "anthropic" else "/v1/chat/completions"
        async with client, transport:
            replies = await asyncio.gather(client.post(path,json=body), client.post(path,json=body))
            assert all(r.status_code == 200 for r in replies)
            summaries = len(backend.private)
            assert len(backend.public) == 2 and summaries
            journal, = args.output.glob("*/compaction.json")
            assert all((journal.parent / f"{turn:05d}" / "response.json").is_file() for turn in (1, 2))
            await client.post(path,json=body)
            assert len(backend.private) == summaries
            # The latest protected request alone cannot fit. Keep the old snapshot.
            saved = journal.read_bytes()
            too_long = dict(body, messages=[message("user", "X"*1400)])
            error = await client.post(path, json=too_long)
            assert error.status_code == 400 and error.json()["error"]["type"] == "invalid_request_error"
            assert journal.read_bytes() == saved
            if kind == "anthropic":
                events = [json.loads(row) for row in (args.output/"requests.jsonl").read_text().splitlines()]
                assert events[-1]["event"] == "context_limit" and events[-1]["task"] == "task"
    asyncio.run(run())
