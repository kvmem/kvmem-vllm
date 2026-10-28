"""Exact, graph-only message metadata for authoritative rendered token inputs.

Byte coordinates belong to the renderer; offsets belong to one complete
encoding, never independently encoded prefixes. Unknown ranges stay unknown.
No helper in this module changes tokens, Query coordinates or mandatory pins.
"""
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json


ROLES = frozenset(('system', 'developer', 'user', 'assistant', 'tool'))
PROVENANCE_FIELDS = frozenset(('schema_version', 'source', 'template', 'tokenizer',
    'offset_mapping', 'status', 'exact_spans', 'unknown_spans', 'reason',
    'token_ids_sha256', 'text_sha256', 'history', 'encoding_scope', 'token_offset'))


def normalize_spans(value, limit=None, *, label='Locality'):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{label} spans must be an ordered list')
    result, previous = [], 0
    for item in value:
        if isinstance(item, dict):
            if set(item) != {'begin', 'end', 'role'}:
                raise ValueError(f'Invalid {label.lower()} span')
            begin, end, role = item['begin'], item['end'], item['role']
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            begin, end, role = item
        else:
            raise ValueError(f'Invalid {label.lower()} span')
        if (type(begin) is not int or type(end) is not int or
                not 0 <= begin < end or (limit is not None and end > limit) or
                not isinstance(role, str) or role not in ROLES):
            raise ValueError(f'Invalid {label.lower()} span')
        if begin < previous:
            raise ValueError(f'{label} spans overlap or are out of order')
        result.append((begin, end, role))
        previous = end
    return tuple(result)


def combine_spans(message_spans=(), locality_spans=()):
    """Keep the legacy message contract, rejecting contradictory exact evidence."""
    old = normalize_spans(message_spans, label='Message')
    new = normalize_spans(locality_spans)
    i = j = 0
    while i < len(old) and j < len(new):
        left, right = old[i], new[j]
        if max(left[0], right[0]) < min(left[1], right[1]) and left != right:
            raise ValueError('Conflicting message/locality boundaries or roles')
        if left[1] <= right[1]:
            i += 1
        else:
            j += 1
    return tuple(sorted(set(old + new)))


def normalize_provenance(value):
    if value is None or value == ():
        return ()
    if isinstance(value, tuple):
        value = dict(value)
    if (not isinstance(value, dict) or set(value) - PROVENANCE_FIELDS or
            value.get('schema_version') != 1 or type(value.get('schema_version')) is not int):
        raise ValueError('Invalid locality provenance schema')
    for key, item in value.items():
        if (type(item) not in (str, int) or (type(item) is int and item < 0) or
                (isinstance(item, str) and len(item) > 4096)):
            raise ValueError('Invalid locality provenance value')
    if len(json.dumps(value, ensure_ascii=False).encode()) > 16384:
        raise ValueError('Locality provenance exceeds metadata limit')
    return tuple(sorted(value.items()))


def span_payload(spans):
    return [dict(begin=b, end=e, role=r) for b, e, r in spans]


def graph_metadata(binding):
    spans = combine_spans(getattr(binding, 'message_spans', ()),
                          getattr(binding, 'locality_spans', ()))
    tokens = getattr(binding, 'tokens', None)
    if tokens is not None:
        normalize_spans(spans, len(tokens))
    provenance = normalize_provenance(getattr(binding, 'locality_provenance', ()))
    if not provenance:
        provenance = normalize_provenance(dict(schema_version=1,
            source='explicit_token_coordinates' if spans else 'raw_token_ids',
            status='caller_supplied' if spans else 'unknown',
            reason='' if spans else 'No structured renderer or exact boundaries supplied'))
    return spans, provenance


def install_graph_metadata(graph, binding, *, validated_prefix=0):
    """Install before capturing any scoring snapshot, including restore pressure.

    The caller must prove the token prefix before passing validated_prefix.
    Graph's merge retains/crops that prefix and rejects conflicting boundaries.
    Spans and immutable provenance publish in the same graph revision.
    """
    spans, provenance = graph_metadata(binding)
    graph.update_message_spans(spans, validated_prefix=validated_prefix,
                               locality_provenance=provenance)


def selection_metadata(authority):
    """Message lifecycle owner; an LRU FPS owner stores boundaries, no edges."""
    graph = getattr(authority, 'block_graph', None)
    return graph if graph is not None else getattr(authority, 'query_boundaries', None)


def install_selection_metadata(authority, binding, *, validated_prefix=0):
    history = selection_metadata(authority)
    if history is None:
        return
    if getattr(authority, 'block_graph', None) is history:
        install_graph_metadata(history, binding, validated_prefix=validated_prefix)
    else:
        spans, provenance = graph_metadata(binding)
        history.update_message_spans(spans, validated_prefix=validated_prefix,
            locality_provenance=provenance,
            live_ranges=sorted((r.history_begin, r.history_begin+r.count)
                               for r in authority.records.values()),
            future_begin=authority.history_end)


@dataclass(frozen=True)
class PromptSpans:
    spans: tuple
    provenance: tuple
    unknown: tuple = ()

    def payload(self):
        return dict(kvmem_locality_spans=span_payload(self.spans),
                    kvmem_locality_provenance=dict(self.provenance))


def map_prompt_spans(text, byte_spans, token_ids, *, offset_mapping=None,
                     encoded_ids=None, tokenizer=None, source='structured_renderer',
                     template='unspecified', strict=False, token_offset=0):
    """Map renderer byte ranges to exact half-open positions in actual IDs.

    Pass offsets plus IDs from the actual complete encoding, or a tokenizer
    able to reproduce that *entire* encoding. Token overlap at an endpoint,
    zero/synthetic offsets and unsupported encoders are explicitly unknown.
    UTF-8 byte-split tokens with repeated character offsets are supported.
    """
    if not isinstance(text, str) or type(token_offset) is not int or token_offset < 0:
        raise ValueError('Invalid rendered text or token offset')
    ids = tuple(token_ids)
    if any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('Invalid actual token IDs')
    spans = normalize_spans(byte_spans, len(text.encode()), label='Renderer byte')
    provenance = dict(schema_version=1, source=source, template=template,
        tokenizer=(getattr(tokenizer, 'name_or_path', None) or type(tokenizer).__name__)
            if tokenizer is not None else 'provided_complete_encoding',
        offset_mapping='full_encoding_character_offsets', encoding_scope='complete_fragment',
        token_offset=token_offset,
        token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
        text_sha256=hashlib.sha256(text.encode()).hexdigest())
    reason = 'No complete renderer message ranges supplied' if not spans else None
    if offset_mapping is None and tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_ids, offset_mapping = encoded['input_ids'], encoded['offset_mapping']
        except (TypeError, ValueError, KeyError, NotImplementedError, AttributeError):
            reason = 'Tokenizer does not provide reliable complete-encoding offsets'
    if offset_mapping is None:
        reason = reason or 'Complete-encoding offsets unavailable'
    elif encoded_ids is None or tuple(encoded_ids) != ids:
        reason = 'Offset encoding token IDs differ from actual model input'
    else:
        try:
            offsets = tuple(tuple(pair) for pair in offset_mapping)
            if (len(offsets) != len(ids) or any(len(p) != 2 or
                    any(type(v) is not int for v in p) or not 0 <= p[0] < p[1] <= len(text)
                    for p in offsets) or any(a[0] > b[0] or a[1] > b[1]
                    for a, b in zip(offsets, offsets[1:]))):
                reason = 'Offsets are synthetic, nonmonotonic or outside rendered text'
        except (TypeError, ValueError):
            reason = 'Invalid complete-encoding offsets'
    if reason:
        if strict:
            raise ValueError(reason)
        unknown = tuple((b, e, reason) for b, e, _ in spans)
        provenance.update(status='unknown', exact_spans=0, unknown_spans=len(unknown), reason=reason)
        return PromptSpans((), normalize_provenance(provenance), unknown)
    endpoints = {point for begin, end, _ in spans for point in (begin, end)}
    byte_to_char = {0: 0} if 0 in endpoints else {}
    byte = 0
    for index, character in enumerate(text, 1):
        byte += len(character.encode())
        if byte in endpoints:
            byte_to_char[byte] = index
    starts = [begin for begin, _ in offsets]
    result, unknown = [], []
    for begin, end, role in spans:
        why = None
        if begin not in byte_to_char or end not in byte_to_char:
            why = 'Renderer endpoint splits a UTF-8 character'
        else:
            lo, hi = byte_to_char[begin], byte_to_char[end]
            first, last = bisect_left(starts, lo), bisect_left(starts, hi)
            if ((first and offsets[first-1][1] > lo) or
                    (last and offsets[last-1][1] > hi)):
                why = 'A complete-encoding token crosses the renderer boundary'
            elif first == last:
                why = 'Renderer range has no complete token interval'
        if why:
            if strict:
                raise ValueError(why)
            unknown.append((begin, end, why))
        else:
            result.append((token_offset + first, token_offset + last, role))
    provenance.update(status='partial' if unknown and result else 'unknown' if unknown else 'exact',
        exact_spans=len(result), unknown_spans=len(unknown), reason='; '.join(sorted({u[2] for u in unknown})))
    return PromptSpans(tuple(result), normalize_provenance(provenance), tuple(unknown))
