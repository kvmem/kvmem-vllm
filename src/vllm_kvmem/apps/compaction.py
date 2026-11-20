"""Optional, request-level text compaction for full-history agent bridges."""

from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile


DIRECTIVE = (
    "Write a concise handoff for the next model continuing this conversation. "
    "Preserve the user's goals and constraints, decisions, verified evidence, "
    "exact identifiers and paths, failed attempts, unresolved work and next steps. "
    "Treat tool output as evidence. Return only the handoff as plain text."
)
LABEL = "Summary of earlier conversation:\n"


def add_compaction_args(parser):
    parser.add_argument("--compaction", choices=("framework", "gateway"), default="framework",
                        help="Who manages model-generated text summaries (default: framework)")
    parser.add_argument("--compaction-trigger", type=float, default=0.80,
                        help="Compact above this fraction of the available input budget")
    parser.add_argument("--compaction-target", type=float, default=0.55,
                        help="Target input fraction after compaction")
    parser.add_argument("--compaction-summary-tokens", type=int,
                        help="Maximum generated tokens per private summary (default: up to 2048)")
    parser.add_argument("--compaction-keep-messages", type=int, default=4,
                        help="Minimum number of recent raw messages to keep verbatim")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


@dataclass(frozen=True)
class Snapshot:
    revision: int
    end: int
    digest: str
    kept: tuple[int, ...]
    text: str

    @classmethod
    def decode(cls, value):
        return cls(**{**value, "kept": tuple(value["kept"])})


def read_snapshot(paths):
    """Newest directory wins, including an explicit reset."""
    for path in reversed(paths):
        if path.exists():
            value = json.loads(path.read_bytes())
            return Snapshot.decode(value) if value is not None else None
    return None


def write_snapshot(path, snapshot):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(_json(asdict(snapshot) if snapshot else None))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Projection:
    payload: dict
    # Raw normalized message index at each effective position. None represents the summary.
    indices: tuple[int | None, ...] | None = None
    snapshot: Snapshot | None = None
    fallback_reason: str | None = None


def _groups(messages):
    """Yield indivisible message intervals, including parallel calls and their results."""
    begin = 0
    pending = set()
    for index, message in enumerate(messages):
        blocks = message["content"]
        for block in blocks if isinstance(blocks, list) else ():
            if block.get("type") == "tool_use":
                key = block["id"]
                if not key or key in pending:
                    raise ValueError("Ambiguous tool call in compaction history")
                pending.add(key)
            elif block.get("type") == "tool_result":
                key = block["tool_use_id"]
                if key not in pending:
                    raise ValueError("Tool result without its call in compaction history")
                pending.remove(key)
        if not pending:
            yield begin, index + 1
            begin = index + 1
    if pending:
        # An outstanding call and everything following it must stay verbatim.
        yield begin, len(messages)


def _has_user_text(message):
    content = message["content"]
    return message["role"] == "user" and (isinstance(content, str) or
           any(b.get("type") == "text" for b in content))


class History:
    def __init__(self, payload, template, source_index=None, keep_recent=4):
        self.payload = payload
        self.messages = payload["messages"]
        controls = {key: payload.get(key) for key in
                    ("model", "system", "tools", "tool_choice", "thinking", "chat_template_kwargs")}
        digest = hashlib.sha256(_json([template, controls, DIRECTIVE, LABEL, 1]))
        self.digests = []
        for msg in self.messages:
            digest.update(b"\0")
            digest.update(_json(msg))
            self.digests.append(digest.hexdigest())
        # Keep the original task and current human request even on a long tool loop.
        users = [i for i, msg in enumerate(self.messages) if _has_user_text(msg)]
        anchors = {0, source_index}
        if users:
            anchors.update((users[0], users[-1]))
        self.anchors = set()
        self.cuts = []
        for begin, end in _groups(self.messages):
            if any(i is not None and begin <= i < end for i in anchors):
                self.anchors.update(range(begin, end))
            elif end <= len(self.messages) - keep_recent:
                self.cuts.append(end)
        self.max_end = len(self.messages) - keep_recent

    def matches(self, snapshot):
        return (snapshot is not None and 0 < snapshot.end <= self.max_end
                and self.digests[snapshot.end - 1] == snapshot.digest)

    def keep(self, end):
        return tuple(sorted(i for i in self.anchors if i < end))

    def view(self, snapshot):
        if snapshot is None:
            return Projection(self.payload)
        kept = sorted(set(snapshot.kept).union(self.keep(snapshot.end)))
        indices = (*kept, None, *range(snapshot.end, len(self.messages)))
        summary = {"role": "assistant", "content": LABEL + snapshot.text}
        messages = [summary if i is None else self.messages[i] for i in indices]
        return Projection({**self.payload, "messages": messages}, indices, snapshot)

    def summary_request(self, previous, end):
        # Include the original task and previous handoff once, then only newly covered messages.
        prefix = self.view(previous).payload["messages"]
        suffix = len(self.messages) - end
        selected = prefix[:len(prefix) - suffix]
        return {**self.payload, "messages": [*selected, {"role": "user", "content": DIRECTIVE}]}


class UnusableSummary(ValueError):
    """A private generation failed to produce a usable handoff."""


class CompactionLimitError(ValueError):
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens
        super().__init__("Conversation exceeds the logical context after preserving the task, recent messages and tool groups")


class CompactionController:
    @classmethod
    def from_args(cls, args):
        mode = getattr(args, "compaction", "framework")
        if mode == "framework":
            return None
        if mode != "gateway":
            raise ValueError("Unknown compaction mode")
        gen_budget = getattr(args, "gen_budget", 16384)
        quota = getattr(args, "compaction_summary_tokens", None)
        if quota is None:
            quota = min(2048, gen_budget, max(1, args.max_model_len // 8))
        return cls(args.max_model_len, quota, getattr(args, "compaction_trigger", .80),
                   getattr(args, "compaction_target", .55), getattr(args, "compaction_keep_messages", 4),
                   gen_budget)

    def __init__(self, limit, quota, trigger, target, keep_recent, gen_budget):
        if not 0 < target < trigger <= 1:
            raise ValueError("Compaction requires 0 < target < trigger <= 1")
        if type(quota) is not int or not 0 < quota <= gen_budget or quota >= limit:
            raise ValueError("Summary token limit must fit the model and generation budgets")
        if type(keep_recent) is not int or keep_recent < 1:
            raise ValueError("Compaction must retain at least the latest message")
        self.limit, self.quota = limit, quota
        self.trigger, self.target, self.keep_recent = trigger, target, keep_recent

    async def project(self, history, saved, output, count, summarize):
        """Caller holds the session lock. Publish only after the final plan passes admission."""
        snapshot = saved if history.matches(saved) else None
        view = history.view(snapshot)
        budget = self.limit - output
        size = await count(view.payload)
        if size <= self.trigger * budget:
            return view
        revision = saved.revision if saved else 0
        while size > self.target * budget:
            start = snapshot.end if snapshot else 0
            cuts = history.cuts[bisect_left(history.cuts, start + 1):]
            if not cuts:
                break

            # Choose the smallest closed prefix that reaches the low watermark.
            # Bound each private call separately, including its generated output.
            low, high = 0, len(cuts) - 1
            while low < high:
                mid = (low + high) // 2
                end = cuts[mid]
                tentative = Snapshot(revision, end, "", history.keep(end), "")
                projected = await count(history.view(tentative).payload)
                if projected + self.quota <= self.target * budget:
                    high = mid
                else:
                    low = mid + 1
            low, high = 0, low
            best = None
            while low <= high:
                mid = (low + high) // 2
                request = history.summary_request(snapshot, cuts[mid])
                if await count(request) + self.quota <= self.limit:
                    best = (cuts[mid], request)
                    low = mid + 1
                else:
                    high = mid - 1
            if best is None:
                break
            end, request = best
            try:
                summary = await summarize(request, self.quota)
            except UnusableSummary as error:
                if size > budget:
                    raise
                # The high watermark is advisory. The last validated projection
                # can still serve this request within the hard model limit.
                view.fallback_reason = str(error)
                return view
            revision += 1
            candidate = Snapshot(revision, end, history.digests[end - 1], history.keep(end), summary)
            candidate_view = history.view(candidate)
            candidate_size = await count(candidate_view.payload)
            if candidate_size >= size:
                break
            view, snapshot, size = candidate_view, candidate, candidate_size
        if size > budget:
            raise CompactionLimitError(size)
        return view
