"""Local Claude Code benchmark adapter, with qw3's frozen protocol helpers.

Only the model transport changes. Exact accepted output IDs survive tool round
trips; a changed transcript falls back to a separately audited canonical render.
This endpoint is intended for the owned, isolated benchmark Docker network.
"""
import argparse
import asyncio
from bisect import bisect_right
import ctypes
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
import uuid
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_kvmem.apps.compaction import (
    CompactionController, CompactionLimitError, DIRECTIVE, History, UnusableSummary,
    add_compaction_args, read_snapshot, write_snapshot,
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def backend_request(*, model, tokens, session, max_tokens, plan, backend_kind, seed=1000):
    """Keep token and sampling transport identical across the two engines."""
    if backend_kind not in ('native', 'kvmem'):
        raise ValueError('Unknown inference backend')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Benchmark seed must be an unsigned 32-bit integer')
    data = dict(model=model, prompt=tokens, cache_salt=session, max_tokens=max_tokens,
        thinking_token_budget=4096, temperature=.6, top_p=.95, top_k=20, min_p=0,
        presence_penalty=0, repetition_penalty=1, seed=seed,
        return_token_ids=True, skip_special_tokens=False)
    if backend_kind == 'kvmem':
        data.update(session_id=session, **plan)
    return data


class Protocol:
    def __init__(self, library):
        self.template = "qw3_claude_code:" + hashlib.sha256(Path(library).read_bytes()).hexdigest()
        self.lib = ctypes.CDLL(str(library))
        self.lib.qw3_protocol.argtypes = [ctypes.c_char_p]
        self.lib.qw3_protocol.restype = ctypes.c_void_p
        self.lib.qw3_protocol_free.argtypes = [ctypes.c_void_p]

    def __call__(self, **payload):
        pointer = self.lib.qw3_protocol(canonical(payload).encode())
        try:
            value = json.loads(ctypes.string_at(pointer))
        finally:
            self.lib.qw3_protocol_free(pointer)
        if 'error' in value:
            raise ValueError(value['error'])
        return value


def _argument_identity(text, tool):
    """Compare transported JSON values without changing accepted model tokens.

    Claude Code's Edit adds its documented replace_all=False default on tool
    round trips. No other tool defaults, string escapes, or content edits are
    inferred. Duplicate keys/non-finite numbers remain exact-string matches.
    Decimal parsing avoids equating distinct large numbers through float rounding.
    """
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate tool argument key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('Non-finite tool argument')

    def identity(value):
        if value is None or isinstance(value, (bool, str)):
            return [type(value).__name__, value]
        if isinstance(value, (int, Decimal)):
            number = Decimal(value)
            sign, digits, exponent = number.as_tuple()
            digits = list(digits)
            if not any(digits):
                return ['number', 0, [0], 0]
            while digits[-1] == 0:
                digits.pop()
                exponent += 1
            return ['number', sign, digits, exponent]
        if isinstance(value, list):
            return ['array', [identity(v) for v in value]]
        return ['object', [[k, identity(v)] for k, v in sorted(value.items())]]

    if not isinstance(text, str):
        return ['unparsed', text]
    try:
        value = json.loads(text, parse_float=Decimal, parse_constant=invalid,
                           object_pairs_hook=pairs)
        if tool == 'Edit' and isinstance(value, dict):
            value.setdefault('replace_all', False)
        return ['parsed', identity(value)]
    except (ValueError, ArithmeticError):
        return ['unparsed', text]


def message_identity(message):
    message = dict(message)
    for key in ('content', 'reasoning_content'):
        if isinstance(message.get(key), str):
            message[key] = message[key].strip(' \n\r\t')
    if isinstance(message.get('tool_calls'), list):
        calls = []
        for call in message['tool_calls']:
            call = dict(call)
            function = dict(call.get('function', {}))
            if 'arguments' in function:
                function['arguments'] = _argument_identity(function['arguments'], function.get('name'))
            call['function'] = function
            calls.append(call)
        message['tool_calls'] = calls
    return canonical(message)


def _identity_key(value):
    """Immutable, type-exact JSON key; tool argument strings remain untouched."""
    if isinstance(value, dict):
        return ('object', tuple(sorted((k, _identity_key(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ('array', tuple(_identity_key(v) for v in value))
    if isinstance(value, float):
        return ('float', value.hex())
    return (type(value).__name__, value)


@dataclass
class Trajectory:
    messages: list = field(default_factory=list)
    control: str = ''
    tokens: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    assistant: str = ''
    output: list = field(default_factory=list)
    turns: int = 0
    locations: dict = field(default_factory=dict)
    # Independent of Query/pin prefix counts: these refer to actual sent IDs.
    locality_spans: tuple = ()
    locality_provenance: tuple = ()
    rendered_prefix: bytes | None = None
    identity_keys: tuple = ()
    output_truncated: bool = False
    pending_assistant_begin: int | None = None
    # Rendered bytes after the last message of the previous prompt: the
    # generation header the previous prompt ended with.
    generation_header: bytes | None = None


class Prepared(tuple):
    """Retain the seven-field transport tuple for existing bridge consumers."""
    def __new__(cls, values, *, assistant_begin=None):
        value = super().__new__(cls, values)
        value.assistant_begin = assistant_begin
        return value


class PrefixTokens:
    """Exact prefix lengths anchored at complete Qwen message delimiters.

    Ordinary token offsets do not prove prefix stability: BPE may tokenize a
    truncated word differently. A complete non-stripping added token is a hard
    tokenizer boundary. Re-encode only the suffix after that boundary, retaining
    the full encoding's preceding count. A fallback still encodes exact prefixes.
    """
    def __init__(self, tokenizer, text):
        self.tokenizer, self.data = tokenizer, text.encode()
        self.positions, self.counts = [0], [0]
        self.offset_mapping, self.offset_ids = None, None
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        markers = {}
        # Qwen uses NFC. Non-normalized delimiters are extracted before that
        # normalizer, so normalization cannot compose characters across them.
        normalizer = getattr(backend, 'normalizer', None)
        normalized = json.loads(normalizer.__getstate__()) if normalizer is not None else None
        if getattr(tokenizer, 'is_fast', False) and backend is not None and normalized in (None, {'type':'NFC'}):
            markers = {token_id:str(token) for token_id,token in tokenizer.added_tokens_decoder.items()
                if str(token) in ('<|im_start|>', '<|im_end|>') and token.special
                and not (token.lstrip or token.rstrip or token.normalized or token.single_word)}
        if markers:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            self.ids = encoded['input_ids']
            self.offset_mapping, self.offset_ids = encoded['offset_mapping'], self.ids
            previous, byte_end = 0, 0
            for index,(token_id,(begin,end)) in enumerate(zip(self.ids,encoded['offset_mapping'])):
                if token_id not in markers:continue
                if text[begin:end] != markers[token_id] or end <= previous:
                    raise ValueError('Added-token boundary differs from original Qwen message delimiter')
                byte_end += len(text[previous:end].encode())
                self.positions.append(byte_end);self.counts.append(index+1)
                previous=end
        else:
            self.ids = tokenizer.encode(text, add_special_tokens=False)
            # Preserve the established production encoding; metadata is admitted
            # only if the complete offset encoding reproduces every actual ID.
            try:
                full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                self.offset_mapping, self.offset_ids = full['offset_mapping'], full['input_ids']
            except (TypeError, ValueError, KeyError, NotImplementedError, AttributeError):
                pass
        self.cache = {0:0, len(self.data):len(self.ids)}

    def count(self, point):
        if not 0 <= point <= len(self.data):raise ValueError('Prefix coordinate outside rendered input')
        if point not in self.cache:
            anchor=bisect_right(self.positions,point)-1
            suffix=self.data[self.positions[anchor]:point].decode()
            self.cache[point]=self.counts[anchor]+len(self.tokenizer.encode(suffix,add_special_tokens=False))
        return self.cache[point]


class Planner:
    def __init__(self, protocol, tokenizer, gen_budget=16384, *, optimized_handoff=False):
        self.protocol, self.tokenizer = protocol, tokenizer
        self.optimized_handoff = optimized_handoff
        self.gen_budget = gen_budget
        self.end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        # Qwen can stop with either terminator. Keep the actual accepted EOS
        # instead of re-rendering a completed endoftext answer on the next turn.
        vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
        self.complete_ends = frozenset([self.end] +
            ([vocabulary['<|endoftext|>']] if '<|endoftext|>' in vocabulary else []))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prepare(self, payload, state, *, finalize_plan=None, count_only=False):
        rendered = self.protocol(request=payload)
        text = rendered['prompt'].encode()
        optimized = getattr(self, 'optimized_handoff', False)
        keys = tuple(_identity_key(m) for m in rendered['messages']) if optimized else ()
        old_keys = getattr(state, 'identity_keys', ())
        messages = [(state.messages[i] if optimized and i < len(old_keys) and
                     i < len(state.messages) and keys[i] == old_keys[i] else message_identity(m))
                    for i, m in enumerate(rendered['messages'])]
        control = canonical({'prefix':text[:rendered['control_end']].decode(), 'tools':rendered['tools']})
        n = len(state.messages)
        # A verified length-limited response has accepted tokens even without
        # a generated EOS. The next request may close that unchanged message
        # with the renderer's delimiter as NEW INPUT, preserving its KV prefix.
        close_truncated = bool(state.output and state.output_truncated and
            state.pending_assistant_begin is not None and state.output[-1] not in self.complete_ends)
        assistant_span = next((s for s in rendered['spans'] if s['index'] == n), None)
        close_truncated = bool(close_truncated and assistant_span and
            text[assistant_span['begin']:assistant_span['end']].endswith(b'<|im_end|>\n') and
            self.encode('<|im_end|>') == [self.end])
        reused = bool(state.output and control == state.control and len(messages) > n+1
                      and messages[:n] == state.messages and messages[n] == state.assistant
                      and (state.output[-1] in self.complete_ends or close_truncated))
        if reused and state.rendered_prefix is not None:
            historical_end = max((s['end'] for s in rendered['spans'] if s['index'] < n),
                                 default=rendered['control_end'])
            reused = text[:historical_end] == state.rendered_prefix
        # The client returned the last assistant message changed (for example
        # tool input escapes decoded), so its accepted IDs cannot be reused.
        # Everything before it is unchanged: keep the previous prompt's exact
        # IDs and render only from that message on. A canonical full render
        # would re-encode all earlier accepted outputs, lose the exact prefix
        # and turn the turn into a cold prefill of the whole conversation.
        prompt_reused = False
        if (not reused and state.tokens and state.rendered_prefix is not None and
                state.generation_header and control == state.control and len(messages) > n
                and messages[:n] == state.messages and assistant_span is not None
                and assistant_span['role'] == 'assistant'):
            anchor = state.rendered_prefix + state.generation_header
            historical_end = max((s['end'] for s in rendered['spans'] if s['index'] < n),
                                 default=rendered['control_end'])
            prompt_reused = (historical_end == len(state.rendered_prefix) ==
                             assistant_span['begin'] and len(text) > len(anchor) and
                             text.startswith(anchor))
        def location(point):
            if point <= rendered['control_end']:return (-1,point)
            for span in rendered['spans']:
                if span['begin'] <= point <= span['end']:
                    return (span['index'],point-span['begin'])
            raise ValueError('Semantic boundary is outside rendered messages')
        locations = dict(state.locations) if reused or prompt_reused else {}
        if prompt_reused:
            boundary = len(anchor)
            prefix = list(state.tokens)
            separator_bytes = 0
            suffix = text[boundary:].decode()
            encoded = PrefixTokens(self.tokenizer, suffix)
            tokens = prefix + encoded.ids
            def coordinate(point):
                if point < boundary:
                    return locations[location(point)]
                value=len(prefix) + encoded.count(point-boundary)
                locations[location(point)]=value
                return value
        elif reused:
            first = next(s for s in rendered['spans'] if s['index'] == n+1)
            boundary = first['begin']
            prefix = state.tokens + state.output
            # The closure is prompt input, never added to accepted output or
            # charged as generated tokens. Ordinary EOS replies add only '\n'.
            separator = '<|im_end|>\n' if close_truncated else '\n'
            separator_bytes = len(separator.encode())
            suffix = separator + text[boundary:].decode()
            encoded = PrefixTokens(self.tokenizer, suffix)
            tokens = prefix + encoded.ids
            def coordinate(point):
                if point < boundary:
                    return locations[location(point)]
                value=len(prefix) + encoded.count(separator_bytes+point-boundary)
                locations[location(point)]=value
                return value
        else:
            boundary = 0
            encoded = PrefixTokens(self.tokenizer, text.decode())
            tokens = encoded.ids
            def coordinate(point):
                value=encoded.count(point)
                locations[location(point)]=value
                return value
        if count_only:
            return len(tokens)
        plan = {'kvmem_mandatory_spans': [], 'kvmem_reselect':'auto'}
        for span in rendered['spans']:
            if span['role'] != 'assistant' and (not (reused or prompt_reused) or span['begin']>=boundary):
                for point in (span['begin'],span['end'],span['content_begin'],span['content_end']):
                    coordinate(point)
        query_index = rendered['query_index']
        q = next((s for s in rendered['spans'] if s['index'] == query_index), None)
        if q is not None:
            plan['kvmem_query_span'] = [coordinate(q['content_begin']), coordinate(q['content_end'])]
        kinds = {0:'system', 1:'query', 2:'tool-result', 3:'policy', 4:'root', 5:'tool-call', 6:'tool-result'}
        pins = []
        for pin in rendered['pins']:
            if pin['end'] <= pin['begin']:
                continue
            kind = kinds[pin['reason']]
            if kind in ('tool-call','tool-result'):
                continue  # Match qw3 default lifetimes: tools remain selectable.
            pins.append({'begin':coordinate(pin['begin']), 'end':coordinate(pin['end']), 'kind':kind})
        plan['kvmem_mandatory_spans'] = list({canonical(p):p for p in pins}.values())
        from vllm_kvmem.serving.prompt_spans import map_prompt_spans, span_payload, normalize_provenance
        # Renderer byte ranges cover each real message/tool result. The new
        # suffix uses its actual encoding and exactly one history offset.
        appended = reused or prompt_reused
        shift = separator_bytes - boundary if appended else 0
        local_text = suffix if appended else text.decode()
        local_spans = []
        for span in rendered['spans']:
            if reused and span['begin'] < boundary:
                continue
            if prompt_reused and span['end'] <= boundary:
                continue
            begin = max(span['begin'], boundary) if prompt_reused else span['begin']
            end = span['end']
            # commit() closes an accepted assistant at EOS. The renderer also
            # includes the following separator newline in its byte range;
            # exclude that separator here so a later canonical fallback agrees
            # with already-validated history. Tokens, Query/pin coordinates and
            # the actual prompt remain untouched. Only recognize the exact
            # frozen-protocol suffix, never whitespace inside message content.
            if (span['role'] == 'assistant' and
                    text[span['begin']:end].endswith(b'<|im_end|>\n')):
                end -= 1
            local_spans.append(dict(begin=begin + shift, end=end + shift,
                                    role=span['role']))
        mapped = map_prompt_spans(local_text, local_spans, encoded.ids,
            offset_mapping=encoded.offset_mapping, encoded_ids=encoded.offset_ids,
            tokenizer=self.tokenizer, source='deepswe_protocol',
            template=getattr(self.protocol, 'template', 'qw3_claude_code'),
            token_offset=len(prefix) if appended else 0)
        inherited = (state.locality_spans if reused else
                     tuple(s for s in state.locality_spans if s[1] <= len(prefix)) if prompt_reused else ())
        if reused and close_truncated:
            inherited += ((state.pending_assistant_begin,
                           len(prefix) + encoded.count(len('<|im_end|>')), 'assistant'),)
        spans = inherited + mapped.spans
        provenance = dict(mapped.provenance)
        provenance.update(exact_spans=len(spans), encoding_scope='append_suffix' if appended else 'prompt',
                          history='actual_accepted_ids' if reused else
                          'actual_prompt_ids' if prompt_reused else 'canonical_full_render')
        if reused:
            previous_provenance = dict(state.locality_provenance)
            if close_truncated:
                previous_provenance['unknown_spans'] = max(0, previous_provenance.get('unknown_spans', 0)-1)
                if not previous_provenance['unknown_spans']:
                    previous_provenance['reason'] = '; '.join(r for r in previous_provenance.get('reason', '').split('; ')
                        if r != 'Accepted assistant interval lacks a proven complete message boundary')
            provenance['unknown_spans'] += previous_provenance.get('unknown_spans', 0)
            reasons = set(previous_provenance.get('reason', '').split('; ')) | set(provenance['reason'].split('; '))
            provenance['reason'] = '; '.join(sorted(reasons - {''}))
        if provenance['unknown_spans']:
            provenance['status'] = 'partial' if spans else 'unknown'
        plan['kvmem_locality_spans'] = span_payload(spans)
        plan['kvmem_locality_provenance'] = dict(normalize_provenance(provenance))
        # The generation header is not yet a complete message. Record its start
        # only when the authoritative Qwen header and full token boundary agree;
        # commit will close it at the actual accepted EOS, never a re-encoding.
        assistant_begin = None
        header = max((span['end'] for span in rendered['spans']), default=rendered['control_end'])
        if text[header:].startswith(b'<|im_start|>assistant\n'):
            pending = map_prompt_spans(local_text,
                [dict(begin=header + shift, end=len(local_text.encode()), role='assistant')],
                encoded.ids, offset_mapping=encoded.offset_mapping, encoded_ids=encoded.offset_ids,
                tokenizer=self.tokenizer, token_offset=len(prefix) if appended else 0)
            if pending.spans:
                assistant_begin = pending.spans[0][0]
        result = Prepared((tokens, plan, rendered, messages, control, reused, locations),
                          assistant_begin=assistant_begin)
        result.coordinate = coordinate
        result.identity_keys = keys
        result.prompt_prefix_reused = prompt_reused
        result.input_closure_tokens = encoded.count(len('<|im_end|>')) if reused and close_truncated else 0
        # An internal adapter may resolve an authoritative Question source
        # after exact token coordinates exist. The final plan always passes the
        # same complete public validation; payload fields cannot disable it.
        if finalize_plan is not None:
            finalize_plan(result)
        from vllm_kvmem.serving.request_plan import RequestPlan
        RequestPlan.parse(plan, len(tokens), 32, self.gen_budget)
        return result

    def response(self, raw, tools, model, usage, finish):
        clean = raw.removesuffix('<|im_end|>').removesuffix('<|endoftext|>')
        parsed = self.protocol(op='parse', text=clean, tools=tools)
        content = []
        if parsed['reasoning']:
            content.append({'type':'thinking','thinking':parsed['reasoning'], 'signature':uuid.uuid4().hex})
        if parsed['content']:
            content.append({'type':'text','text':parsed['content']})
        for call in parsed['tool_calls']:
            content.append({'type':'tool_use','id':'toolu_'+uuid.uuid4().hex,
                            'name':call['function']['name'],'input':json.loads(call['function']['arguments'])})
        if not content:
            content = [{'type':'text','text':''}]
        return {'id':'msg_'+uuid.uuid4().hex,'type':'message','role':'assistant','model':model,
                'content':content,'stop_reason':'tool_use' if parsed['tool_calls'] else 'max_tokens' if finish=='length' else 'end_turn',
                'stop_sequence':None,'usage':{'input_tokens':usage['prompt_tokens'], 'output_tokens':usage['completion_tokens']}}

    def commit(self, state, prepared, payload, response, output):
        tokens, plan, rendered, messages, control, reused, locations = prepared
        assistant = {'role':'assistant','content':response['content']}
        # Only the normalized final assistant is consumed here. In the frozen
        # Qwen protocol its conversion depends on this message and tool schema,
        # not on re-rendering all historical messages.
        local = getattr(self, 'optimized_handoff', False)
        converted = self.protocol(request={**payload,'messages':
            [assistant] if local else payload['messages']+[assistant]})
        state.messages, state.control, state.tokens = messages, control, tokens
        state.identity_keys = getattr(prepared, 'identity_keys', ())
        state.plan, state.output = plan, output
        state.output_truncated = response.get('stop_reason') == 'max_tokens'
        state.locations = locations
        historical_end = max((s['end'] for s in rendered['spans']), default=rendered['control_end'])
        state.rendered_prefix = rendered['prompt'].encode()[:historical_end]
        state.generation_header = rendered['prompt'].encode()[historical_end:]
        from vllm_kvmem.serving.prompt_spans import normalize_spans, normalize_provenance
        state.locality_spans = normalize_spans(plan.get('kvmem_locality_spans', []), len(tokens))
        state.locality_provenance = normalize_provenance(plan.get('kvmem_locality_provenance'))
        assistant_begin = getattr(prepared, 'assistant_begin', None)
        state.pending_assistant_begin = assistant_begin
        if assistant_begin is not None and output and output[-1] in self.complete_ends:
            state.locality_spans += ((assistant_begin, len(tokens) + len(output), 'assistant'),)
        accepted_provenance = dict(state.locality_provenance)
        if accepted_provenance:
            accepted_provenance.update(exact_spans=len(state.locality_spans), history='actual_accepted_ids')
            if output and (assistant_begin is None or output[-1] not in self.complete_ends):
                accepted_provenance['unknown_spans'] = accepted_provenance.get('unknown_spans', 0) + 1
                accepted_provenance['status'] = 'partial' if state.locality_spans else 'unknown'
                reasons = set(accepted_provenance.get('reason', '').split('; '))
                reasons.add('Accepted assistant interval lacks a proven complete message boundary')
                accepted_provenance['reason'] = '; '.join(sorted(reasons - {''}))
            state.locality_provenance = normalize_provenance(accepted_provenance)
        state.assistant = message_identity(converted['messages'][-1])
        state.turns += 1


def event(kind, **payload):
    return 'event: '+kind+'\ndata: '+json.dumps({'type':kind,**payload},ensure_ascii=False)+'\n\n'


def response_events(response):
    for index, block in enumerate(response['content']):
        kind = block['type']
        if kind == 'thinking':
            yield event('content_block_start',index=index,content_block={'type':'thinking','thinking':''})
            yield event('content_block_delta',index=index,delta={'type':'thinking_delta','thinking':block['thinking']})
            yield event('content_block_delta',index=index,delta={'type':'signature_delta','signature':block['signature']})
        elif kind == 'tool_use':
            yield event('content_block_start',index=index,content_block={**block,'input':{}})
            yield event('content_block_delta',index=index,delta={'type':'input_json_delta','partial_json':json.dumps(block['input'],ensure_ascii=False)})
        else:
            yield event('content_block_start',index=index,content_block={'type':'text','text':''})
            yield event('content_block_delta',index=index,delta={'type':'text_delta','text':block['text']})
        yield event('content_block_stop',index=index)
    yield event('message_delta',delta={'stop_reason':response['stop_reason'],'stop_sequence':None},usage=response['usage'])
    yield event('message_stop')


def bounded_task_query(payload, instruction, replay_budget):
    """Use a verified task instruction only when CLI context exceeds Q capacity.

    All prompt tokens and original mandatory pins remain unchanged. The source
    must be an exact standalone text block from the frozen task, never an
    arbitrary shortened suffix or a substring guessed from generated history.
    """
    source = instruction.removesuffix('\n')
    if not source or type(replay_budget) is not int or replay_budget <= 0:
        raise ValueError('A complete task instruction and positive replay budget are required')
    def finalize(prepared):
        _, plan, rendered, _, _, reused, _ = prepared
        old = plan.get('kvmem_query_span')
        if old is None or old[1] - old[0] // 32 * 32 <= replay_budget:
            return  # Preserve the original experiment policy for supported Queries.
        matches = [i for i, message in enumerate(payload['messages']) if message.get('role') == 'user'
            and isinstance(message.get('content'), list)
            and any(b.get('type') == 'text' and b.get('text') == source
                    for b in message['content'] if isinstance(b, dict))]
        if matches != [0]:
            raise ValueError('Oversized Query lacks a unique exact frozen task instruction in the initial user message')
        span = next((s for s in rendered['spans'] if s['index'] == rendered['query_index']), None)
        first_user = next((s for s in rendered['spans'] if s['role'] == 'user'), None)
        if span is None or span != first_user:
            raise ValueError('Task instruction is not the renderer-owned initial Query')
        raw = rendered['prompt'].encode(); needle = source.encode()
        content = raw[span['content_begin']:span['content_end']]
        if content.count(needle) != 1:
            raise ValueError('Frozen task instruction is ambiguous in the rendered Query')
        begin = span['content_begin'] + content.index(needle)
        selected = [prepared.coordinate(begin), prepared.coordinate(begin + len(needle))]
        if selected[1] - selected[0] // 32 * 32 > replay_budget:
            raise ValueError('Complete frozen task instruction exceeds Query replay capacity; no clipping allowed')
        plan['kvmem_query_span'] = selected
        plan['kvmem_query_id'] = 'deepswe-task-instruction:' + hashlib.sha256(needle).hexdigest()
        prepared.query_source_audit = dict(mode='oversized_cli_context_fallback',
            original_query_span=old, task_query_span=selected, replay_budget=replay_budget,
            instruction_sha256=hashlib.sha256(needle).hexdigest(),
            prompt_unchanged=True, mandatory_pins_unchanged=True, exact_prefix_reused=reused)
    return finalize


class BridgeCompaction:
    """Use the same renderer, tokenizer, admission and owner as the visible request."""

    def __init__(self, policy, args, planner, client, host_work, log):
        self.policy, self.args, self.planner = policy, args, planner
        self.client, self.host_work, self.log = client, host_work, log

    @classmethod
    def create(cls, args, planner, client, host_work, log):
        policy = CompactionController.from_args(args)
        return cls(policy, args, planner, client, host_work, log) if policy else None

    async def count(self, payload, state=None):
        def measure():
            if state is not None and state.tokens:
                # A re-encoding may have a different length from accepted output IDs.
                return self.planner.prepare(payload, state, count_only=True)
            return len(self.planner.encode(self.planner.protocol(request=payload)['prompt']))
        return await self.host_work(measure)

    async def history(self, payload, owner, directories, source_index=None):
        history = await self.host_work(History, payload, (owner, self.planner.protocol.template),
                                       source_index, self.policy.keep_recent)
        saved = await self.host_work(read_snapshot, [d/'compaction.json' for d in directories])
        return history, saved

    async def project(self, payload, owner, output, directories, state, *, source_index=None, admit=None):
        history, saved = await self.history(payload, owner, directories, source_index)

        async def summarize(request, quota):
            started = time.monotonic()
            def select_directive(prepared):
                rendered = prepared[2]
                # The harness may select an earlier task as its Query. This private
                # call always queries with the final, bridge-authored directive.
                span = rendered['spans'][-1]
                begin, end = span['content_begin'], span['content_end']
                if span['role'] != 'user' or rendered['prompt'].encode()[begin:end] != DIRECTIVE.encode():
                    raise ValueError('Compaction directive has no exact rendered span')
                prepared[1].update(kvmem_query_span=[prepared.coordinate(begin), prepared.coordinate(end)],
                                   kvmem_mandatory_parts='none', kvmem_mandatory_spans=[])
            prepared = await self.host_work(self.planner.prepare, request, state,
                                            finalize_plan=select_directive)
            tokens, plan, rendered = prepared[:3]
            if len(tokens) + quota > self.policy.limit:
                raise ValueError('Private summary exceeds the logical context')
            if admit:
                await admit(len(tokens), quota)
            data = backend_request(model=self.args.model, tokens=tokens, session=owner, max_tokens=quota,
                                   plan=plan, backend_kind=getattr(self.args, 'backend_kind', 'kvmem'))
            data.update(temperature=0, thinking_token_budget=0)
            try:
                reply = await self.client.post(self.args.backend+'/v1/completions', json=data)
                reply.raise_for_status()
            except httpx.HTTPError as error:
                raise UnusableSummary(f'Private summary request failed: {error}') from error
            native = reply.json()
            choice, usage = native['choices'][0], native['usage']
            if (choice.get('finish_reason') != 'stop' or not isinstance(choice.get('token_ids'), list)
                    or not 0 < len(choice['token_ids']) == usage['completion_tokens'] <= quota):
                raise UnusableSummary('Private summary did not finish within its token budget')
            try:
                parsed = self.planner.response(choice['text'], rendered['tools'], self.args.model, usage, 'stop')
            except ValueError as error:
                raise UnusableSummary('Private summary could not be parsed') from error
            text = ''.join(b['text'] for b in parsed['content'] if b['type'] == 'text').strip()
            if parsed['stop_reason'] != 'end_turn' or not text:
                raise UnusableSummary('Private summary must contain final text and no tool call')
            self.log(dict(event='compaction_summary', owner=owner, seconds=time.monotonic()-started,
                          input_tokens=len(tokens), output_tokens=usage['completion_tokens']))
            return text

        async def count(payload):
            return await self.count(payload, state)

        view = await self.policy.project(history, saved, output, count, summarize)
        if view.fallback_reason:
            self.log(dict(event='compaction_fallback', owner=owner, reason=view.fallback_reason))
        return view, saved


def create_app(args, tokenizer):
    app = FastAPI()
    backend_kind = getattr(args, 'backend_kind', 'kvmem')
    if backend_kind not in ('native', 'kvmem'):
        raise ValueError('Unknown inference backend')
    gen_budget = getattr(args, 'gen_budget', 16384)
    if type(gen_budget) is not int or gen_budget <= 0:
        raise ValueError('Generation budget must be positive')
    output_cap = getattr(args, 'max_output_tokens', None)
    if output_cap is None:
        output_cap = gen_budget  # Preserve frozen benchmark defaults.
    if type(output_cap) is not int or not 0 < output_cap < args.max_model_len:
        raise ValueError('Output cap must be positive and below the logical context limit')
    planner_options = {'optimized_handoff': True} if getattr(args, 'optimized_handoff', False) else {}
    planner = Planner(Protocol(args.protocol), tokenizer, gen_budget, **planner_options)
    # --cases: several runs of one task (e.g. one per seed), each a case with its own
    # trajectory, session, output directory and sampling seed: {case: {"task", "seed"}}.
    tasks = set(args.tasks.read_text().splitlines())
    cases = json.loads(args.cases.read_text()) if getattr(args, 'cases', None) is not None else {}
    for case, entry in cases.items():
        if (entry.get('task') not in tasks or type(entry.get('seed')) is not int or
                not 0 <= entry['seed'] < 2**32 or '/' in case or case == 'preflight'):
            raise ValueError('Invalid case entry: ' + case)
    allowed = (set(cases) if cases else tasks) | {'preflight'}
    locks = {task:asyncio.Lock() for task in allowed}
    states = {task:Trajectory() for task in allowed}
    # Each turn's request.json holds only what changed since the previous turn
    # (vllm_kvmem.core.request_records.load_request rebuilds the whole request).
    from vllm_kvmem.core.request_records import RequestRecords
    request_records = {task:RequestRecords() for task in allowed}
    client = httpx.AsyncClient(trust_env=False,timeout=max(3600, output_cap / 10))
    inflight = 0
    credits = {}
    capacity = getattr(args, 'cpu_admission_bytes', None) or int(args.cpu_admission_gib*1024**3)
    changed = asyncio.Condition()
    long_owner = None
    short_limit = 49152*17*4*256*3
    admission_policy = getattr(args, 'cpu_admission_policy', 'legacy')
    engine_admission_verified = False
    args.output.mkdir(parents=True,exist_ok=True)
    def log(row):
        with (args.output/'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')

    compaction = BridgeCompaction.create(args, planner, client, asyncio.to_thread, log)

    async def reserve(task, amount):
        nonlocal long_owner, engine_admission_verified
        if backend_kind == 'native':
            return 0.  # Native vLLM owns GPU admission and prefix-cache eviction.
        if admission_policy == 'engine':
            if not engine_admission_verified:
                geometry = json.loads(args.cache_geometry.read_text())
                if (not geometry.get('cpu_history_coordination') or
                        not geometry.get('cpu_progress_reserve') or
                        geometry['cpu_offload_bytes'] != capacity):
                    raise ValueError('Engine CPU admission/geometry does not match bridge budget')
                engine_admission_verified = True
            return 0.  # Engine defers only requests which cannot acquire credits.
        if amount > capacity:
            raise ValueError('Requested history/output exceeds legacy CPU admission capacity')
        begin=time.monotonic()
        async with changed:
            reported=False
            while ((amount > short_limit and long_owner not in (None,task)) or
                   sum(credits.values())-credits.get(task,0)+amount > capacity):
                if not reported:
                    log({'event':'cpu_admission_wait','task':task,'at':time.time(),
                         'required_bytes':amount,'used_bytes':sum(credits.values()),'capacity_bytes':capacity,
                         'long_trajectory_owner':long_owner})
                    reported=True
                await changed.wait()
            if amount > short_limit and long_owner is None:
                long_owner=task
                log({'event':'long_trajectory_admitted','task':task,'at':time.time(),
                     'policy':'Reserve one full 1M CPU history plus seven 48K histories; never evict or recompute a parked task'})
            credits[task]=amount
            return time.monotonic()-begin

    @app.post('/control/{task}/finish')
    async def finish(task:str,request:Request):
        nonlocal long_owner
        if request.headers.get('x-control-token') != os.environ.get('DEEPSWE_CONTROL_TOKEN','invalid'):
            return JSONResponse({'error':'unauthorized'},status_code=401)
        if task not in allowed:return JSONResponse({'error':'unknown task'},status_code=404)
        async with locks[task]:
            # Explicit release waits for fenced eviction and must not replace
            # the old history with another model checkpoint for the EOS marker.
            # This cleanup completion is never scored.
            session=args.run_id+'-'+task
            cleanup_id = None
            if backend_kind == 'kvmem':
                result=await client.post(args.backend+'/v1/completions',json={'model':args.model,
                    'prompt':[planner.end], 'session_id':session,'cache_salt':session,
                    'max_tokens':1,'temperature':0,'kvmem_reselect':'off',
                    'kvmem_mandatory_spans':[],
                    'kv_transfer_params':{'kvmem_release_session':True}})
                result.raise_for_status()
                cleanup_id = result.json()['id']
            states[task]=Trajectory()
            if compaction:
                write_snapshot(args.output/task/'compaction.json', None)
            async with changed:
                credits.pop(task,None)
                if long_owner==task:long_owner=None
                changed.notify_all()
            log({'event':'history_released','task':task,'at':time.time(),
                 'cleanup_request_id':cleanup_id, 'backend_kind':backend_kind,
                 'native_prefix_cache':'evictable' if backend_kind == 'native' else None})
        return {'released':True}

    @app.get('/task/{task}/health')
    @app.get('/health')
    async def health(task='preflight'):
        result = await client.get(args.backend+'/health')
        return JSONResponse({'ok':result.status_code==200},status_code=result.status_code)

    @app.post('/task/{task}/v1/messages/count_tokens')
    async def count_tokens(task:str, request:Request):
        if task not in allowed:return JSONResponse({'error':'unknown task'},status_code=404)
        payload = await request.json()
        if compaction:
            async with locks[task]:
                owner = args.run_id+'-'+task
                history, saved = await compaction.history(payload, owner, [args.output/task])
                view = history.view(saved if history.matches(saved) else None)
                return {'input_tokens': await compaction.count(view.payload, states[task])}
        value = await asyncio.to_thread(planner.protocol,request=payload)
        return {'input_tokens':len(await asyncio.to_thread(planner.encode,value['prompt']))}

    @app.post('/task/{task}/v1/messages')
    async def messages(task:str, request:Request):
        if task not in allowed:return JSONResponse({'error':'unknown task'},status_code=404)
        raw_payload = await request.json()
        async def generate():
            nonlocal inflight, long_owner
            async with locks[task]:
                state = states[task]
                started=time.time()
                session=args.run_id+'-'+task
                payload = raw_payload
                requested = payload.get('max_tokens', output_cap)
                if type(requested) is not int or requested <= 0:
                    raise ValueError('max_tokens must be a positive integer')
                max_tokens = min(requested, output_cap)
                if compaction:
                    async def admit(prompt_size, quota):
                        await reserve(task, ((prompt_size+quota+31)//32)*32*17*4*256*3)
                    try:
                        view, saved = await compaction.project(payload, session, max_tokens,
                                                              [args.output/task], state, admit=admit)
                    except CompactionLimitError as error:
                        log(dict(event='context_limit', task=task, turn=state.turns+1, at=time.time(),
                                 prompt_tokens=error.input_tokens, requested_output_tokens=max_tokens,
                                 max_model_len=args.max_model_len))
                        raise
                    payload = view.payload
                query_root = getattr(args, 'task_instructions_root', None)
                options = {}
                if query_root is not None and task != 'preflight':
                    source = cases[task]['task'] if task in cases else task
                    instruction = (Path(query_root)/source/'instruction.md').read_text()
                    options['finalize_plan'] = bounded_task_query(payload, instruction, args.query_replay_budget)
                prepared=await asyncio.to_thread(planner.prepare,payload,state,**options)
                tokens, plan, rendered, _, _, reused, _=prepared
                directory=args.output/task/f'{state.turns+1:05d}'
                directory.mkdir(parents=True,exist_ok=True)
                saved_request = await asyncio.to_thread(request_records[task].record, directory.name, raw_payload,
                    base=f'{state.turns:05d}' if state.turns else None)
                (directory/'request.json').write_text(json.dumps(saved_request,ensure_ascii=False))
                if len(tokens)+max_tokens>args.max_model_len:
                    log({'event':'context_limit','task':task,'turn':state.turns+1,'at':time.time(),
                         'prompt_tokens':len(tokens),'requested_output_tokens':max_tokens,
                         'max_model_len':args.max_model_len})
                    raise ValueError('Complete trajectory exceeds configured logical context; no truncation/compaction applied')
                data=backend_request(model=args.model, tokens=tokens, session=session,
                    max_tokens=max_tokens, plan=plan, backend_kind=backend_kind,
                    seed=cases[task]['seed'] if task in cases else getattr(args, 'seed', 1000))
                prompt_reused = getattr(prepared, 'prompt_prefix_reused', False)
                previous_tokens = (len(state.tokens) + len(state.output) if reused else
                                   len(state.tokens) if prompt_reused else 0)
                amount=((len(tokens)+data['max_tokens']+31)//32)*32*17*4*256*3
                cpu_queue_seconds=await reserve(task,amount)
                if compaction and view.snapshot != saved:
                    write_snapshot(args.output/task/'compaction.json', view.snapshot)
                (directory/'plan.json').write_text(json.dumps({k:v for k,v in data.items() if k!='prompt'}|
                    ({'compaction': asdict(view.snapshot)} if compaction and view.snapshot else {}) |
                    {'prompt_tokens':len(tokens),'exact_output_prefix_reused':reused,
                     'prompt_prefix_reused':prompt_reused,
                     'token_sha256':hashlib.sha256(canonical(tokens).encode()).hexdigest(),
                     'task_query_source':getattr(prepared, 'query_source_audit', None)}))
                inflight+=1
                admitted=time.time()
                log({'event':'start','task':task,'turn':state.turns+1,'at':admitted,
                     'request_started_at':started,'cpu_queue_seconds':cpu_queue_seconds,'prompt_tokens':len(tokens),
                     'inflight':inflight,'exact_output_prefix_reused':reused,'prompt_prefix_reused':prompt_reused,
                     'new_input_tokens':len(tokens)-previous_tokens,
                     'prepare_seconds':admitted-started-cpu_queue_seconds, 'backend_kind':backend_kind})
                try:
                    result=await client.post(args.backend+'/v1/completions',json=data)
                    result.raise_for_status(); native=result.json()
                    choice=native['choices'][0]
                    response=planner.response(choice['text'],rendered['tools'],payload['model'],native['usage'],choice['finish_reason'])
                    output=choice['token_ids']
                    completed = replace(state)
                    await asyncio.to_thread(planner.commit,completed,prepared,payload,response,output)
                    if backend_kind == 'kvmem':
                        async with changed:
                            credits[task]=((len(tokens)+len(output)+31)//32)*32*17*4*256*3
                            if compaction and long_owner == task and credits[task] <= short_limit:
                                long_owner = None
                            changed.notify_all()
                    (directory/'response.json').write_text(json.dumps(response,ensure_ascii=False))
                    (directory/'native.json').write_text(json.dumps({**native,'choices':[{k:v for k,v in choice.items() if k!='prompt_token_ids'}]},ensure_ascii=False))
                    states[task] = state = completed
                    log({'event':'finish','task':task,'turn':state.turns,'at':time.time(),'seconds':time.time()-started,
                         'cpu_queue_seconds':cpu_queue_seconds,'native_seconds':time.time()-admitted,
                         'prompt_tokens':len(tokens),'new_input_tokens':len(tokens)-previous_tokens,
                         'usage':native['usage'],'metrics':native.get('metrics'),'request_id':native['id'],
                         'finish_reason':choice['finish_reason'],'tool_calls':sum(b['type']=='tool_use' for b in response['content']),
                         'inflight':inflight})
                    return response
                except BaseException as error:
                    log({'event':'cancelled' if isinstance(error,asyncio.CancelledError) else 'error',
                         'error_type':type(error).__name__,'task':task,'turn':state.turns+1,'at':time.time(),'error':str(error),
                         **({'backend_status':error.response.status_code,'backend_error':error.response.text[:2048]}
                            if isinstance(error,httpx.HTTPStatusError) else {})})
                    raise
                finally:inflight-=1
        if not raw_payload.get('stream'):
            try:return await generate()
            except CompactionLimitError as error:
                return JSONResponse({'type':'error','error':{'type':'invalid_request_error','message':str(error)}},status_code=400)
            except (ValueError,httpx.HTTPError) as error:
                return JSONResponse({'type':'error','error':{'type':'api_error','message':str(error)}},status_code=500)
        async def stream():
            job=asyncio.create_task(generate())
            yield event('message_start',message={'id':'msg_'+uuid.uuid4().hex,'type':'message','role':'assistant',
                'model':raw_payload['model'],'content':[],'stop_reason':None,'stop_sequence':None,
                'usage':{'input_tokens':0,'output_tokens':0}})
            try:
                while not job.done():
                    done,_=await asyncio.wait({job},timeout=10)
                    if not done:yield event('ping')
                for item in response_events(job.result()):yield item
            except (ValueError,httpx.HTTPError) as error:
                kind = 'invalid_request_error' if isinstance(error, CompactionLimitError) else 'api_error'
                yield event('error',error={'type':kind,'message':str(error)})
            finally:
                if not job.done():job.cancel()
                await asyncio.gather(job,return_exceptions=True)
        return StreamingResponse(stream(),media_type='text/event-stream')
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    add_compaction_args(p)
    for name in ('protocol','tasks','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--backend',required=True)
    p.add_argument('--model',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=18483)
    p.add_argument('--max-model-len',type=int,default=1048576)
    p.add_argument('--gen-budget',type=int,default=16384)
    p.add_argument('--max-output-tokens',type=int,
                   help='Independent reply cap; defaults to G. Values above G require rolling_active_window on the backend and stay in one native request')
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--cases',type=Path,
                   help='JSON {case: {"task": TASK, "seed": SEED}}: serve these cases (several seeds per task) instead of --tasks')
    p.add_argument('--backend-kind',choices=('native','kvmem'),default='kvmem')
    p.add_argument('--cpu-admission-gib',type=int,default=72)
    p.add_argument('--cpu-admission-bytes',type=int)
    p.add_argument('--cpu-admission-policy',choices=('legacy','engine'),default='legacy')
    p.add_argument('--cache-geometry',type=Path)
    p.add_argument('--task-instructions-root',type=Path,
                   help='Opt in to exact frozen task Queries when CLI-added context exceeds replay capacity; prompt and mandatory pins are retained')
    p.add_argument('--query-replay-budget',type=int,default=4096)
    p.add_argument('--optimized-handoff',action='store_true',
                   help='Reuse exact message identities and local spans across tool turns')
    args=p.parse_args()
    if not 0 <= args.seed < 2**32:p.error('Seed must be an unsigned 32-bit integer')
    if args.backend_kind == 'kvmem' and args.cpu_admission_policy == 'engine' and args.cache_geometry is None:
        p.error('Engine admission requires --cache-geometry')
    from transformers import AutoTokenizer
    import uvicorn
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    uvicorn.run(create_app(args,tokenizer),host=args.host,port=args.port,log_level='warning')
