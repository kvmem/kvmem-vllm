"""Accepted-token thinking labels for historical GC, independent of Action/graphs.

Classification owns no token journal. Counts live on the existing bounded Record
objects; pending references live only until their existing capture packet commits.
"""
from collections import OrderedDict
from bisect import bisect_right
import weakref

from .prompt_spans import combine_spans

from .. import _native


THINKING_CAPTURE_BYTES_PER_TOKEN = 16  # int64 forward input ID + logical position


def thinking_token_layout(tokenizer):
    """Resolve a supported Qwen message grammar from the actual tokenizer.

    An unknown template disables classification, leaving the ordinary fallback
    policy available. It must never turn arbitrary assistant output into thinking.
    """
    markers = ('<|im_start|>', '<|im_end|>', '<think>', '</think>')
    try:
        template = tokenizer.get_chat_template()
        if not isinstance(template, str) or any(value not in template for value in markers):
            return None
        ids = [tokenizer.convert_tokens_to_ids(value) for value in markers]
        if any(type(value) is not int or not 0 <= value <= 0xffffffff for value in ids):
            return None
        if len(set(ids)) != 4:
            return None
        if any(list(tokenizer.encode(value, add_special_tokens=False)) != [token]
               for value, token in zip(markers, ids)):
            return None
        headers = []
        for role in ('system', 'developer', 'user', 'assistant', 'tool'):
            encoded = list(tokenizer.encode(markers[0] + role + '\n', add_special_tokens=False))
            if not encoded or encoded[0] != ids[0]:
                return None
            headers.append(encoded[1:])
        # Validate bounded/unambiguous headers before creating any owner state.
        _native.HistoricalThinkingParser(ids, headers)
        return ids, headers
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


class ThinkingHistory:
    def __init__(self, scope, layout, block_tokens, max_pending_tokens, *, identity=None):
        if (type(block_tokens) is not int or block_tokens <= 0 or
                type(max_pending_tokens) is not int or max_pending_tokens <= 0):
            raise ValueError('Thinking history requires positive bounded capacities')
        self.scope = scope
        self.parser = _native.HistoricalThinkingParser(*layout) if layout is not None else None
        self.layout = layout
        self.block_tokens = block_tokens
        self.max_pending_tokens = max_pending_tokens
        self.pending = {}
        self.publications = OrderedDict()
        self.pending_tokens = 0
        self.namespace = identity
        self.through = None
        self.seed_required = True
        self.closed = False
        self.generation_span = None
        self.generation_allowed = False
        # Assistant-message ordinals of one binding: (key, begins, ends, completed).
        self.turns = None
        self.published_turns = None
        self.stats = dict(accepted_tokens=0, thinking_tokens=0, unknown_tokens=0,
            pure_blocks=0, mixed_blocks=0, publications=0, invalidations=0, turn_index_errors=0,
            unsupported_template=int(layout is None))

    def note_append(self, history_begin, count, rows):
        """Called after a successful KV append, before its semantic publication."""
        if not count:
            return
        if self.closed or history_begin in self.pending or self.pending_tokens + count > self.max_pending_tokens:
            raise RuntimeError('Thinking capture exceeded its bounded accepted publication ownership')
        entries = tuple((record, offset, start, n) for _, record, offset, start, n in rows)
        if sum(row[3] for row in entries) != count:
            raise RuntimeError('Thinking capture records do not cover the accepted append')
        self.pending[history_begin] = count, entries
        self.pending_tokens += count

    def _reset_parser(self):
        if self.parser is not None:
            self.parser.reset()
        self.seed_required = True

    def _generation_boundary(self, binding):
        """Trust an outer role, never a message header quoted inside its body.

        Structured renderers already send exact complete-message spans. Their
        remaining generation suffix can be recognized only when it consists
        entirely of a canonical assistant header and optional think controls.
        Raw token prompts without outer boundaries deliberately stay unknown.
        """
        self.generation_span, self.generation_allowed = None, False
        if self.layout is None:
            return
        spans = [items[-1] for items in (binding.message_spans, binding.locality_spans) if items]
        if not spans:
            return
        begin, end, role = max(spans, key=lambda row: row[1])
        length = len(binding.tokens)
        if end == length and role == 'assistant':
            self.generation_allowed = True
            return
        if not 0 <= end < length or length - end > 128:
            return
        markers, headers = self.layout
        # The validated Qwen role encodings share their newline terminator.
        newline = []
        for distance in range(1, min(map(len, headers)) + 1):
            if len({header[-distance] for header in headers}) != 1:
                break
            newline.insert(0, headers[0][-distance])
        if not newline:
            return
        tail = list(binding.tokens[end:])
        skip = 0
        while tail[skip:skip+len(newline)] == newline:
            skip += len(newline)
        header = [markers[0], *headers[3]]
        if tail[skip:skip+len(header)] != header:
            return
        cursor = skip + len(header)
        if cursor < len(tail) and tail[cursor] == markers[2]:
            cursor += 1
            while tail[cursor:cursor+len(newline)] == newline:
                cursor += len(newline)
            if cursor < len(tail) and tail[cursor] == markers[3]:
                cursor += 1
        while tail[cursor:cursor+len(newline)] == newline:
            cursor += len(newline)
        if cursor != len(tail):
            return
        self.generation_span = (end + skip, length, 'assistant')
        self.generation_allowed = True

    def _turn_index(self, binding):
        """Exact assistant messages of this binding, from its outer role spans.

        Ordinals are 1-based and count assistant messages from the start of
        the conversation, so a continued session keeps every earlier ordinal.
        The opt-in thinking age rule reads them; classification does not.
        """
        key = (id(binding), binding.namespace, binding.generation, len(binding.tokens),
               len(binding.message_spans), len(binding.locality_spans), self.generation_span)
        if self.turns is None or self.turns[0] != key:
            try:
                spans = combine_spans(binding.message_spans, binding.locality_spans)
            except ValueError:
                # Contradictory boundaries: no ordinals, so the age rule skips them.
                spans = ()
                self.stats['turn_index_errors'] += 1
            begins = [lo for lo, _, role in spans if role == 'assistant']
            ends = [hi for _, hi, role in spans if role == 'assistant']
            # A prompt ending inside an assistant message continues it.
            open_last = (self.generation_allowed and self.generation_span is None and
                         bool(ends) and ends[-1] == len(binding.tokens))
            self.turns = key, begins, ends, len(begins) - int(open_last)
        return self.turns

    def _turn_at(self, binding, position):
        """Ordinal of the assistant message holding ``position`` (0 = none)."""
        _, begins, ends, _ = self._turn_index(binding)
        if position >= len(binding.tokens):
            if not self.generation_allowed:
                return 0
            if self.generation_span is not None:
                return len(begins) + 1
            return len(begins) if ends and ends[-1] == len(binding.tokens) else 0
        index = bisect_right(begins, position) - 1
        return index + 1 if index >= 0 and position < ends[index] else 0

    def _publish_turns(self, binding):
        completed = self._turn_index(binding)[3]
        if completed != self.published_turns:
            self.scope.publish_turns(completed)
            self.published_turns = completed

    def _classify(self, binding, begin, tokens, *, collect=True):
        """Apply exact outer role boundaries before inspecting body markers."""
        if self.parser is None:
            return [0] * len(tokens) if collect else None
        end = begin + len(tokens)
        intervals = []
        for spans in (binding.message_spans, binding.locality_spans):
            # RequestPlan already normalized and cross-validated these tuples.
            # Reuse them in place; do not retain a second full-history index.
            cursor = bisect_right(spans, begin, key=lambda row: row[1])
            while cursor < len(spans) and spans[cursor][0] < end:
                intervals.append(spans[cursor])
                cursor += 1
        if self.generation_span is not None:
            intervals.append(self.generation_span)
        prompt_end = len(binding.tokens)
        points = {begin, end}
        if begin < prompt_end < end:
            points.add(prompt_end)
        for lo, hi, _ in intervals:
            if max(lo, begin) < min(hi, end):
                points.update((max(lo, begin), min(hi, end)))
        points = sorted(points)
        labels = [] if collect else None
        for lo, hi in zip(points, points[1:]):
            roles = {role for first, last, role in intervals if first <= lo and hi <= last}
            role = next(iter(roles)) if len(roles) == 1 else None
            if lo >= prompt_end:
                role = 'assistant' if self.generation_allowed else None
            if role == 'assistant':
                chunk = tokens[lo-begin:hi-begin]
                if collect:
                    labels.extend(self.parser.feed(chunk))
                else:
                    self.parser.seed(chunk)
            else:
                # A user/tool can quote an entire assistant header. Never feed
                # those tokens, and never let their fake state reach generation.
                self.parser.reset()
                if collect:
                    labels.extend([1 if role is not None else 0] * (hi-lo))
        return labels

    def _begin(self, binding, history_begin):
        namespace = binding.namespace, binding.generation
        if self.namespace is not None and self.namespace != namespace:
            raise RuntimeError('Thinking capture belongs to a different session generation')
        if self.through != history_begin:
            self._reset_parser()
        self.namespace = namespace
        self._generation_boundary(binding)
        if self.seed_required and self.parser is not None:
            # The session machinery has verified only binding.history. Never
            # seed from an unaccepted generated suffix or a coincidental length.
            if 0 <= history_begin <= binding.history and history_begin <= len(binding.tokens):
                for offset in range(0, history_begin, 4096):
                    self._classify(binding, offset, binding.tokens[offset:min(history_begin, offset + 4096)], collect=False)
            self.seed_required = False
        self.through = history_begin
        if self.parser is not None:
            self._publish_turns(binding)

    def accept(self, binding, history_begin, retained, tokens, positions, repository):
        if not retained:
            return
        if self.closed:
            raise RuntimeError('Closed thinking history cannot accept another packet')
        item = self.pending.pop(history_begin, None)
        if item is None or item[0] != retained:
            raise RuntimeError('Thinking labels lack their exact accepted KV append')
        self.pending_tokens -= retained
        self._begin(binding, history_begin)
        tokens, positions = list(tokens[:retained]), list(positions[:retained])
        valid = (len(tokens) == retained and positions == list(range(history_begin, history_begin + retained))
                 and all(type(t) is int and 0 <= t <= 0xffffffff for t in tokens))
        if not valid:
            self._reset_parser()
            self.stats['invalidations'] += 1
        labels = self._classify(binding, history_begin, tokens) if valid else [0] * retained
        self.through = history_begin + retained
        self.stats['accepted_tokens'] += retained
        self.stats['thinking_tokens'] += labels.count(2)
        self.stats['unknown_tokens'] += labels.count(0)
        for record, offset, start, count in item[1]:
            if record.history_begin + offset != history_begin + start:
                raise RuntimeError('Thinking record coordinates differ from its accepted input rows')
            previous = getattr(record, '_thinking_counts', (0, 0, 0))
            if previous is None:
                raise RuntimeError('A sealed thinking classification cannot accept a new suffix')
            covered, known, thinking = previous
            if covered != offset:
                # A restored prefix without classification stays unknown. Do
                # not infer that the newly observed suffix covers its prefix.
                known = thinking = 0
            part = labels[start:start + count]
            if offset == 0:
                record._thinking_turn = self._turn_at(binding, history_begin + start)
            record._thinking_counts = (offset + count,
                known + sum(value != 0 for value in part), thinking + part.count(2))
            self.publish(record, repository)

    def publish(self, record, repository):
        if self.closed or not record.archived or record.count != self.block_tokens:
            return False
        counts = getattr(record, '_thinking_counts', (0, 0, 0))
        if counts is None:
            return True  # This immutable sealed generation was already published.
        covered, known, thinking = counts
        if covered != record.count:
            return False  # Its semantic accepted-token publication is still pending.
        key = record.handle.slot, record.handle.generation
        lease = repository.pin(record.handle)
        if not lease:
            self.publications.pop(key, None)
            return False
        try:
            turn = getattr(record, '_thinking_turn', 0)
            result = self.scope.publish_thinking(lease, thinking, known, turn)
            if result:
                self.stats['publications'] += 1
                self.stats['pure_blocks' if thinking == known == record.count else 'mixed_blocks'] += 1
                self.publications.pop(key, None)
                record._thinking_counts = None
                # Kept for prefix images: an imported session republishes it.
                record._thinking_published = (thinking, known, turn)
            else:
                # Retry metadata contention without keeping a deleted Record or
                # its MeanK storage alive. The number is bounded by live records.
                self.publications[key] = weakref.ref(record)
            return result
        finally:
            lease.reset()

    def retry(self, repository, limit=64):
        for _ in range(min(limit, len(self.publications))):
            _, ref = self.publications.popitem(last=False)
            record = ref()
            if record is not None:
                self.publish(record, repository)

    def rewind(self, position):
        self.pending.clear()
        retained = OrderedDict()
        for key, ref in self.publications.items():
            record = ref()
            if record is not None and record.history_begin + record.count <= position:
                retained[key] = ref
        self.publications = retained
        self.pending_tokens = 0
        self._reset_parser()
        self.through = position
        self.stats['invalidations'] += 1

    def close(self):
        self.rewind(0)
        self.publications.clear()
        self.closed = True
