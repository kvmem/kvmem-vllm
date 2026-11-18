"""Expire archived tool pages at accepted message boundaries."""
from bisect import bisect_right
from operator import itemgetter

from .block_graph import _merge_spans
from .prompt_spans import graph_metadata


class ToolResultHistory:
    def __init__(self, scope, block_tokens, keep_recent):
        self.scope = scope
        self.block_tokens = block_tokens
        self.keep_recent = keep_recent
        self.spans = self.known_ends = self.tool_spans = self.protected = ()
        self.expired = ()
        self.completed = 0

    def bind(self, binding, accepted_through):
        """Install a validated request's spans after proving its reused prefix."""
        spans, _ = graph_metadata(binding)
        self.spans, self.known_ends = _merge_spans(
            self.spans, self.known_ends, spans, accepted_through)
        block = self.block_tokens
        self.protected = sorted((begin // block * block, (end + block - 1) // block * block)
                                for begin, end, _ in binding.mandatory_spans)
        self._update(accepted_through)

    def rewind(self, position):
        self.spans, self.known_ends = _merge_spans(self.spans, self.known_ends, (), position)
        self._update(position)

    def advance(self, accepted_through):
        if self.completed < len(self.tool_spans) and self.tool_spans[self.completed][1] <= accepted_through:
            self._expire(accepted_through)

    def _update(self, accepted_through):
        self.tool_spans = tuple((begin, end) for (begin, end, role), known
                                in zip(self.spans, self.known_ends) if known and role == 'tool')
        self._expire(accepted_through)

    def _expire(self, accepted_through):
        self.completed = bisect_right(self.tool_spans, accepted_through, key=itemgetter(1))
        stop = max(0, self.completed - self.keep_recent)
        block = self.block_tokens
        expired = []
        pin_index = 0
        for begin, end in self.tool_spans[:stop]:
            begin = (begin + block - 1) // block * block
            end = end // block * block
            if begin >= end:
                continue
            while pin_index < len(self.protected):
                pin_begin, pin_end = self.protected[pin_index]
                if pin_begin >= end or begin >= end:
                    break
                if begin < pin_begin:
                    expired.append((begin, pin_begin))
                begin = max(begin, pin_end)
                if pin_end > end:
                    break
                pin_index += 1
            if begin < end:
                expired.append((begin, end))
        expired = tuple(expired)
        if expired != self.expired:
            self.scope.set_expired_tool_ranges(expired)
            self.expired = expired
