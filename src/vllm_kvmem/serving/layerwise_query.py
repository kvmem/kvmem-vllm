"""Serial, eager diagnostic: select a layer's history immediately before attention.

All layers retain the original number of native slots. Their historical contents
may differ after the final public Query. This is deliberately a one-turn probe:
another refresh or session continuation is rejected, never silently interpreted
using the shared historical mapping. The ordinary serving path is unaffected.
"""
import hashlib
import json
import os
import time
import threading
from types import MethodType

import torch
from ..adapters.original_rope import enabled as original_rope, record_positions
from ..core import settings as _settings


def top_set(memory, records, scores, fixed, budget, *, record_index=None):
    selected = set(memory.preview_selection(
        ({r.handle.slot: float(s) for r, s in zip(records, scores)} if record_index is None
         else record_index.score_dict(scores)),
        tuple(fixed), action=False, prefill_scores=None, max_tokens=budget))
    if not set(fixed) <= selected or len(selected) * 32 > budget:
        raise RuntimeError('Layer selection lost mandatory content or exceeded its budget')
    return selected



def query_scorer(owner, options):
    """The profile's per-layer Query scorer, or None for the built-in MeanK path."""
    name = getattr(options, 'strategy_query_scorer', 'meank')
    cache = owner.__dict__.setdefault('_strategy_scorers', {})
    if name not in cache:
        from ..strategy import QUERY_SCORER, resolve
        cache[name] = resolve(QUERY_SCORER, name)
    return cache[name]


def strategy_scores(scorer, state, layer_index, q, options):
    """A strategy's scores for one layer's candidates, checked against the contract."""
    from ..strategy import QueryLayerView, check_query_scores
    view = QueryLayerView(layer_index, state['records'], state['score_mask'], q,
                          state.get('mean_index'), state['history'], options)
    return list(check_query_scores(view, scorer.scores(view)))


def decode_inline_reads():
    """Opt-in KVMEM_DECODE_INLINE_READS=1: rolling decode pressure refreshes read
    their CPU history inline on the refresh thread (the prefill path: the same
    archive work() and native GIL-free gather) instead of a future on the archive
    I/O pool, whose per-chunk handoffs contend with the engine for the GIL.
    Same bytes in the same staging slots; only the executing thread changes."""
    import os
    value = _settings.get('KVMEM_DECODE_INLINE_READS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_DECODE_INLINE_READS must be 0 or 1')
    return value == '1'


def query_global_layers():
    """KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS=N (strategy setting, default 0): at the
    final per-layer Query the first N full-attention layers take the shared global
    MeanK selection, the native frame chosen from every layer's Query before the
    per-layer pass, and skip their own scoring; later layers keep their per-layer
    MeanK selection. Shallow layers retrieve poorly on their own (LongMemEval-S,
    K 8,192: 82.4 with N=4 against 81.0 per-layer and 77.4 global)."""
    value = _settings.get('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS', '0')
    if not value.isdigit():
        raise ValueError('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS must be a non-negative integer')
    return int(value)


class LayerwiseQuery:
    def __init__(self, runner):
        self.runner = runner
        self.mode = _settings.require('KVMEM_LAYERWISE_QUERY_MODE')
        if self.mode not in ('global', 'sequential'):
            raise ValueError('Expected global or sequential layerwise diagnostic')
        self.native_batches = _settings.get('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES') == '1'
        if not self.native_batches and (not runner.model_config.enforce_eager or runner._kvmem_graph):
            raise ValueError('Layerwise diagnostic requires an eager native forward')
        if not self.native_batches and runner.vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError('Layerwise diagnostic requires one serial request')
        self.names = tuple(runner._kvmem_layers)
        if original_rope():
            from ..adapters.original_rope import extend_rope_table
            if not runner.model_config.enforce_eager:
                raise ValueError('Original RoPE qualification requires eager execution')
            modules = dict(runner.model.named_modules())
            for name in self.names:
                extend_rope_table(modules[name].rotary_emb, runner.model_config.max_model_len)
        self.pending = {}
        self.active = None
        self.finished = set()
        self.path = None
        self.optimized = _settings.get('KVMEM_LAYERWISE_QUERY_OPTIMIZED') == '1'
        self.no_probe = _settings.get('KVMEM_LAYERWISE_QUERY_NO_PROBE') == '1'
        self.mtp_selection = _settings.get('KVMEM_LAYERWISE_QUERY_MTP_SELECTION',
                                       'layer0' if self.no_probe else 'global')
        if self.mtp_selection not in ('global', 'layer0'):
            raise ValueError('Expected global or layer0 MTP selection')
        if (self.no_probe or self.mtp_selection == 'layer0') and (
                self.mode != 'sequential' or not self.optimized):
            raise ValueError('No-probe/layer0 MTP requires optimized sequential Query')
        if self.no_probe and self.mtp_selection != 'layer0':
            raise ValueError('No-probe Query requires layer0 MTP selection')
        self.packed_assembly = _settings.get('KVMEM_LAYERWISE_PACKED_ASSEMBLY') == '1'
        self.prepared_index = _settings.get('KVMEM_LAYERWISE_PREPARED_INDEX') == '1'
        # Query-local metadata only; native ranking and KV execution stay unchanged.
        record_index = _settings.get('KVMEM_LAYERWISE_QUERY_RECORD_INDEX', '0')
        if record_index not in ('0', '1'):
            raise ValueError('Query record indexing must be 0 or 1')
        self.record_index = record_index == '1'
        # Execution candidate remains disabled until full-prefix byte and
        # output equivalence plus matched serving measurements pass.
        self.direct_prefix_restore = False
        self.batched_reads = _settings.get('KVMEM_LAYERWISE_BATCHED_READS') == '1'
        self.paired_reads = _settings.get('KVMEM_LAYERWISE_PAIRED_READS') == '1'
        paired_gather = _settings.get('KVMEM_LAYERWISE_QUERY_PAIRED_GATHER', '0')
        if paired_gather not in ('0', '1'):
            raise ValueError('Query paired gathering must be 0 or 1')
        self.paired_gather = paired_gather == '1'
        if self.paired_gather and (not self.optimized or not self.paired_reads):
            raise ValueError('Query paired gathering requires optimized paired layer reads')
        if self.paired_reads and not self.optimized:
            raise ValueError('Paired layer reads require optimized Query execution')
        if self.batched_reads and not self.optimized:
            raise ValueError('Batched layer reads require optimized Query execution')
        if self.prepared_index and not self.optimized:
            raise ValueError('Prepared layer index requires optimized Query execution')
        self.score_backend = _settings.get('KVMEM_LAYERWISE_QUERY_SCORE_BACKEND', 'legacy')
        if self.score_backend not in ('legacy', 'tiled'):
            raise ValueError('Query score backend must be legacy or tiled')
        if self.score_backend != 'legacy' and (
                not self.optimized or not self.prepared_index or self.mode != 'sequential'):
            raise ValueError('Query score backend override requires prepared optimized sequential Query')
        for attribute, name in (('resident_values', 'KVMEM_LAYERWISE_QUERY_RESIDENT_V'),
                                ('lazy_reader', 'KVMEM_LAYERWISE_QUERY_LAZY_READER')):
            value = _settings.get(name, '0')
            if value not in ('0', '1'):
                raise ValueError(name+' must be 0 or 1')
            setattr(self, attribute, value == '1')
            if value == '1' and (not self.optimized or not self.paired_reads or
                    self.mode != 'sequential' or self.no_probe or self.mtp_selection != 'global'):
                raise ValueError(name+' requires optimized public sequential Query with global MTP')
        if self.resident_values and (not self.packed_assembly or original_rope()):
            raise ValueError('Query resident V requires packed compact-position assembly')
        if (self.lazy_reader or self.resident_values) and self.paired_gather:
            raise ValueError('Query transport candidates and experimental paired gather require separate qualification')
        if self.packed_assembly and (not self.optimized or original_rope()):
            raise ValueError('Packed layer assembly requires optimized compact-position execution')
        self.diagnostics = not self.optimized or _settings.get('KVMEM_LAYERWISE_QUERY_DIAGNOSTICS') == '1'
        stage_profile = _settings.get('KVMEM_QUERY_STAGE_PROFILE', '0')
        if stage_profile not in ('0', '1'):
            raise ValueError('Query stage profiling must be 0 or 1')
        self.stage_profile = stage_profile == '1'
        if self.stage_profile and (not self.optimized or self.mode != 'sequential'):
            raise ValueError('Query stage profiling requires optimized sequential Query')
        self.buffered_events = []
        self._emit_lock = threading.RLock()
        self.deferred_writer = None
        deferred_evidence = _settings.get('KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE', '0')
        if deferred_evidence not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE must be 0 or 1')
        if deferred_evidence == '1':
            from .deferred_evidence import DeferredEvidence
            self.deferred_writer = DeferredEvidence(self)
        self.active_owners = {}
        self.pressure = None
        if _settings.get('KVMEM_LAYERWISE_PREFILL_PRESSURE') == '1':
            from .layerwise_pressure import LayerwisePrefillPressure
            self.pressure = runner._kvmem_layerwise_pressure = LayerwisePrefillPressure(self)

    @property
    def has_active(self):
        return self.active is not None or bool(self.active_owners)

    def _evidence_path(self):
        if self.path is None:
            self.path = self.runner._kvmem_probe_output / 'layerwise_query.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path

    def _append_lines(self, lines):
        # Caller holds _emit_lock; one complete line per write, in queue order.
        with self._evidence_path().open('a') as stream:
            for line in lines:
                stream.write(line)

    def emit(self, event, **fields):
        self._evidence_path()
        record = dict(event=event, mode=self.mode, at=time.time(), monotonic_at=time.perf_counter(), **fields)
        with self._emit_lock:
            if self.optimized and event == 'layer_selected':
                self.buffered_events.append(record)
                return
            # Opt-in deferred rows are older than anything buffered after them.
            deferred = self.deferred_writer.take_lines() if self.deferred_writer is not None else []
            with self.path.open('a') as stream:
                for line in deferred:
                    stream.write(line)
                for pending in self.buffered_events:
                    stream.write(json.dumps(pending, allow_nan=False) + '\n')
                self.buffered_events.clear()
                stream.write(json.dumps(record, allow_nan=False) + '\n')

    def defer(self, event, **fields):
        """Same row as emit(), timestamped now; written later in the same order."""
        if self.deferred_writer is None:
            return self.emit(event, **fields)
        self._evidence_path()
        record = dict(event=event, mode=self.mode, at=time.time(), monotonic_at=time.perf_counter(), **fields)
        with self._emit_lock:
            buffered = list(self.buffered_events)
            self.buffered_events.clear()
            self.deferred_writer.defer(record, buffered)

    def defer_lazy(self, event, resolve, **fields):
        """defer() for a row whose trailing fields are resolve()'s dict, computed when written."""
        if self.deferred_writer is None:
            return self.emit(event, **fields, **resolve())
        self._evidence_path()
        record = dict(event=event, mode=self.mode, at=time.time(), monotonic_at=time.perf_counter(), **fields)
        with self._emit_lock:
            buffered = list(self.buffered_events)
            self.buffered_events.clear()
            self.deferred_writer.defer(record, buffered, resolve=resolve)

    def prepare(self, refresh, job, queries):
        rid = job.command.request_id
        # Rolling mode keeps every layer's map after the final Query (pressure.maps)
        # and refreshes it through the per-layer pressure path. That covers decode
        # and also the ordinary prefill that follows a mid-prompt Query: an agent
        # Query marks the latest user message, and a cold (unrestored) prompt can
        # carry more than G tokens of that turn's tool history after it.
        rolling_pressure = (self.pressure is not None and self.pressure.rolling_decode and
                            job.command.selection == 'pressure' and
                            job.command.pressure_stage in ('decode', 'prefill'))
        if rid in self.finished and not rolling_pressure:
            raise RuntimeError('Layerwise one-turn diagnostic prohibits another refresh')
        if job.command.selection != 'query':
            if self.pressure is not None:
                self.pressure.prepare(refresh, job)
                return
            if refresh.options.pressure_policy != 'attention':
                raise ValueError('Layerwise diagnostic requires accumulated-attention pressure')
            from .prefill_attention import pressure_seed
            _, job.layerwise_pressure_source = pressure_seed(
                job.authority, tuple(job.authority.memory._store.working_records()))
            return
        if ((not self.native_batches and (self.pending or self.active is not None)) or
                rid in self.pending or rid in self.active_owners):
            raise RuntimeError('Overlapping layerwise Query owners')
        from .authority import query_score_frame
        from .request_plan import RequestPlan, selection_config
        cfg = selection_config(refresh.options.memory, job.command.groups[0][1].window_tokens,
                               selection='query', pressure_policy='attention')
        fit = RequestPlan(None, block_tokens=32, gen_budget=cfg.gen_budget,
                          mandatory=job.command.mandatory_spans).mandatory_fit(job.history, cfg)
        authority = job.authority
        authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
        # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: the per-layer Query windows are
        # chosen from this snapshot during the forward; no retirement until
        # all of them are held (released at the last layer).
        from .layerwise_gc_holds import wait_paused as _gc_wait_paused
        _gc_wait_paused(authority)
        with authority.index_snapshot() as (records, means):
            required = set(fit['hard_blocks'])
            pins = [r.handle.slot for r in records if r.history_begin // 32 in required]
            with query_score_frame(authority.memory, records, pins, cfg.select_budget) as frame:
                score_mask = tuple(frame.score_mask)
                fixed = tuple(authority.memory._store.fixed_selection_blocks(
                    pins, pressure=False, budget=cfg.select_budget))
            state = dict(authority=authority, records=tuple(records), means=means[:len(self.names)].clone(),
                         old_q=queries.detach().clone() if queries is not None else None,
                         score_mask=score_mask, fixed=fixed,
                         budget=cfg.select_budget, history=job.history, span=job.command.query_span,
                         request_id=rid, next_layer=0, query_tokens=len(job.command.input_tokens))
        global_layers = query_global_layers()
        if global_layers:
            if self.mode != 'sequential' or self.no_probe:
                raise ValueError('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS needs the sequential per-layer Query '
                                 'with its probe, which makes the shared selection')
            if global_layers > len(self.names):
                raise ValueError('KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS exceeds the full-attention layers')
            state['global_layers'] = global_layers
        # Strategy layer: None runs the built-in per-layer MeanK scores unchanged.
        state['query_scorer'] = query_scorer(self, refresh.options)
        state['trace_scores'] = getattr(refresh.options, 'strategy_trace_scores', False)
        if queries is None and not self.no_probe:
            raise ValueError('Sequential comparison requires its private Query captures')
        if queries is not None and state['old_q'].shape[0] != len(self.names):
            raise ValueError('Frozen Query does not cover every full-attention layer')
        if self.optimized:
            state['means'] = state['means'].to(self.runner.device)
            state['gpu_mask'] = torch.tensor(state['score_mask'], device=self.runner.device, dtype=torch.bool)
            if self.prepared_index:
                from ..core.cuda_scoring import prepare_mean_key_index
                state['mean_index'] = prepare_mean_key_index(state['means'],state['gpu_mask'])
                # Keep one index copy after preparation, as in the old path.
                state['means'] = None
        self.pending[rid] = state

    def arm_no_probe(self, refresh, job):
        from .layerwise_no_probe import arm
        arm(self, refresh, job)

    def retire(self, request_id):
        for states in (self.pending, self.active_owners):
            self._close_query_resources(states.get(request_id), drain=True)
            states.pop(request_id, None)
        if self.active is not None and self.active['request_id'] == request_id:
            self._close_query_resources(self.active, drain=True)
            self.active = None
        self.finished.discard(request_id)

    def _close_query_resources(self, state, *, drain=False):
        if state is None:
            return
        owners = state.get('query_resident_metadata_owners')
        if drain and owners:
            torch.cuda.current_stream(self.runner.device).synchronize()
        reader = state.get('query_region_reader')
        if reader is not None:
            if not reader.close():
                raise RuntimeError('Query source reader still has an active CPU producer')
            state.pop('query_region_reader')
        if owners:
            owners.clear()

    def _query_reader(self, state):
        """Prepare immutable sources lazily; a single Query owns their lifetime."""
        reader = state.get('query_region_reader')
        if reader is None:
            from .query_region_reader import QueryRegionReader
            rid = state['request_id']
            authority = state['authority']
            def generation():
                runtime = self.runner._kvmem_capture_runtime
                if runtime.authorities.get(rid) is not authority:
                    raise ValueError('Lazy Query reader lost its native authority binding')
                return self.runner._kvmem_request_generations.get(rid)
            reader = QueryRegionReader(authority, request_generation=generation(),
                                       generation_getter=generation)
            state['query_region_reader'] = reader
        return reader

    def committed(self, job):
        if job.command.selection != 'query':
            if self.pressure is not None:
                self.pressure.committed(job)
                return
            self.emit('pressure_selected', request_id=job.command.request_id,
                      actual_method='accumulated_attention',
                      score_source=job.layerwise_pressure_source,
                      accepted_history=job.history, selected_tokens=job.selected_tokens)
            return
        state = self.pending.pop(job.command.request_id)
        authority = state['authority']
        by_slot = {r.handle.slot: r for r in state['records']}
        working = list(authority.memory._store.working_records())
        original = {r[0] for r in working}
        # Partial records must have identical inclusion in every layer so native
        # causal lengths and Query positions remain exactly the baseline's.
        state['fixed'] = tuple(set(state['fixed']) | {
            s for s in original if by_slot[s].count != 32})
        state['partial_excluded'] = {r.handle.slot for r in state['records']
                                     if r.count != 32 and r.handle.slot not in original}
        state['selected_tokens'] = job.selected_tokens
        state['global_slots'] = original
        if getattr(self, 'resident_values', False):
            from .query_resident_values import bind_query_resident_sources
            started = time.perf_counter()
            binding = bind_query_resident_sources(self, state, working,
                                                 global_assembly=job.result[:3])
            proof = binding.value_proof
            self.emit('resident_sources_bound', request_id=state['request_id'],
                      source_prepare_seconds=time.perf_counter()-started,
                      **{key: value for key, value in proof.items() if key != 'records'},
                      records=[dict(slot=key[0], generation=key[1], **value)
                               for key, value in proof['records'].items()])
        if self.native_batches:
            self.active_owners[state['request_id']] = state
        else:
            self.active = state
        self.emit('armed', request_id=state['request_id'], layers=len(self.names),
                  budget=state['budget'], gen_budget=authority.memory.config.gen_budget,
                  selected_tokens=job.selected_tokens, query_span=state['span'],
                  fixed_blocks=len(state['fixed']), candidate_blocks=len(state['records']),
                  actual_method='mean_k_global' if self.mode == 'global' else 'mean_k_sequential_layerwise')

    def projection(self, layer_index, record):
        if self.native_batches:
            batch = self.runner._kvmem_last_batch
            # Every request keeps its own logical history, mandatory set and
            # layer cursor even when native scheduling combines public rows.
            for state in tuple(self.active_owners.values()):
                if state['request_id'] in batch.req_ids:
                    self._projection(state, layer_index, record)
        else:
            self._projection(self.active, layer_index, record)

    def _projection(self, state, layer_index, record):
        try:
            return self._projection_inner(state, layer_index, record)
        except BaseException:
            self._close_query_resources(state, drain=True)
            raise

    def _projection_inner(self, state, layer_index, record):
        if state is None:
            return
        if state.get('no_probe') and self.runner._kvmem_request_generations.get(
                state['request_id']) != state['request_generation']:
            raise RuntimeError('No-probe Query reached a recycled native request')
        batch = self.runner._kvmem_last_batch
        if not self.native_batches and list(batch.req_ids) != [state['request_id']]:
            raise RuntimeError('Layerwise diagnostic reached a foreign batch')
        if layer_index != state['next_layer']:
            raise RuntimeError('Full-attention layers did not execute in causal order')
        owner_index = list(batch.req_ids).index(state['request_id'])
        begin = int(batch.num_computed_tokens_np[owner_index])
        row_begin, row_end = (map(int,batch.query_start_loc_np[owner_index:owner_index+2])
                              if self.native_batches else (0,batch.num_tokens))
        end = begin + row_end-row_begin
        qb, qe = state['span']
        if begin != state['history'] or not begin <= qb < qe <= end:
            raise RuntimeError('The complete public Query must fit in one native batch')
        q = record.query[row_begin+qb-begin:row_begin+qe-begin].unsqueeze(0)
        position_evidence = {}
        if original_rope():
            observed = record.positions[..., row_begin+qb-begin:row_begin+qe-begin]
            expected = torch.arange(qb, qe, device=observed.device, dtype=observed.dtype)
            if not torch.equal(observed, expected.expand_as(observed)):
                raise RuntimeError('Layer Query did not use original-history RoPE positions')
            position_evidence = dict(rope_position_policy='original', query_rope_span=[qb,qe],
                                     query_rope_verified=True)
        old_q = state['old_q'][layer_index:layer_index+1] if state['old_q'] is not None else None
        compare = self.diagnostics and old_q is not None
        if old_q is not None and q.shape != old_q.shape:
            raise RuntimeError('Fresh and frozen layer Queries have different rows')
        from ..core.cuda_scoring import mean_key_scores
        from .query_stage_profile import QueryStageProfile, no_stage
        profile = state.get('query_stage_profile')
        if getattr(self, 'stage_profile', False) and profile is None:
            profile = state['query_stage_profile'] = QueryStageProfile(
                self.runner._kvmem_probe_output / 'query_stage_profile.jsonl', state['request_id'])
        if profile is not None:
            profile.layer = layer_index
        stage = profile.stage if profile is not None else no_stage
        records = state['records']
        started = time.perf_counter()
        scorer = state.get('query_scorer')
        use_global = layer_index < state.get('global_layers', 0)
        if use_global:
            # Shared selection (KVMEM_LAYERWISE_QUERY_GLOBAL_LAYERS): no scoring.
            compare, old_scores, scores = False, None, None
        elif scorer is not None:
            # Frozen-Query comparison is a diagnostic of the built-in scorer only.
            compare, old_scores = False, None
            scores = strategy_scores(scorer, state, layer_index, q,
                                     dict(score_backend=getattr(self, 'score_backend', 'legacy')))
        elif self.prepared_index:
            from ..core.cuda_scoring import prepared_layer_scores
            score_backend = getattr(self, 'score_backend', 'legacy')
            score_kwargs = {} if score_backend == 'legacy' else {'backend': score_backend}
            scores = (profile.score(lambda: prepared_layer_scores(q,state['mean_index'],layer_index,**score_kwargs))
                      if profile is not None else
                      prepared_layer_scores(q,state['mean_index'],layer_index,**score_kwargs).cpu().tolist())
            old_scores = (prepared_layer_scores(old_q,state['mean_index'],layer_index,**score_kwargs).cpu().tolist()
                          if compare else None)
        else:
            means = state['means'][layer_index:layer_index+1].to(q.device)
            mask = state['gpu_mask'] if self.optimized else torch.tensor(state['score_mask'], device=q.device, dtype=torch.bool)
            scores = (profile.score(lambda: mean_key_scores(q, means, mask)) if profile is not None
                      else mean_key_scores(q, means, mask).cpu().tolist())
            old_scores = mean_key_scores(old_q, means, mask).cpu().tolist() if compare else None
        record_index = None
        with stage('selection_cpu', candidates=len(records)):
            if use_global:
                # The native frame: mandatory, sink and partial blocks included, same token count.
                selected, frozen_selected = set(state['global_slots']), None
                if not set(state['fixed']) <= selected or len(selected) * 32 > state['budget']:
                    raise RuntimeError('The shared selection lost mandatory content or exceeded its budget')
            elif getattr(self, 'record_index', False):
                from .query_record_index import query_record_index
                record_index = query_record_index(state)
                for i in record_index.excluded_indices:
                    scores[i] = -1.
                    if old_scores is not None:
                        old_scores[i] = -1.
            else:
                for i, r in enumerate(records):
                    if r.handle.slot in state['partial_excluded']:
                        scores[i] = -1.
                        if old_scores is not None:
                            old_scores[i] = -1.
            memory = state['authority'].memory
            if not use_global:
                selected = top_set(memory, records, scores, state['fixed'], state['budget'],
                                   record_index=record_index)
                frozen_selected = (top_set(memory, records, old_scores, state['fixed'], state['budget'],
                                          record_index=record_index)
                                   if old_scores is not None else None)
            chosen = ([r for r in records if r.handle.slot in selected] if record_index is None
                      else record_index.chosen(selected))
            if sum(r.count for r in chosen) != state['selected_tokens']:
                raise RuntimeError('Layer-specific selection changed the native causal length')
        scoring_seconds = time.perf_counter() - started
        actual = selected if self.mode == 'sequential' else state['global_slots']
        if layer_index == 0 and self.mtp_selection == 'layer0':
            state['layer_zero_records'] = chosen
        if state.get('no_probe') or (self.pressure is not None and self.pressure.rolling_decode):
            from .layerwise_decode import record_map
            state.setdefault('rolling_maps', []).append(record_map(chosen))
        from .layerwise_gc_holds import scope_of as _gc_scope
        if _gc_scope(state['authority']) is not None:
            state.setdefault('gc_windows', []).append(tuple(chosen))
        assembly_started = time.perf_counter()
        assembly_stats = {}
        if self.mode == 'sequential':
            if self.optimized:
                assembly_stats = self.assemble_layer_pinned(state, layer_index, chosen)
            else:
                self.assemble_layer(state, layer_index, chosen)
        assembly_seconds = time.perf_counter() - assembly_started
        diagnostic = {}
        if getattr(self, 'score_backend', 'legacy') != 'legacy':
            diagnostic['score_backend'] = self.score_backend
        if compare:
            fresh, old = q.float(), old_q.float()
            norm = old.norm().clamp_min(1e-12)
            cosine = torch.nn.functional.cosine_similarity(fresh.flatten(), old.flatten(), dim=0)
            diagnostic = dict(query_cosine=float(cosine), query_relative_l2=float((fresh-old).norm()/norm),
                fresh_frozen_set_overlap=len(selected & frozen_selected)/max(1,len(selected)),
                frozen_layer_selected_starts=([r.history_begin for r in records if r.handle.slot in frozen_selected]
                    if record_index is None else record_index.starts(frozen_selected)))
        if use_global:
            diagnostic['query_scorer'] = 'global_selection'
        elif scorer is not None:
            diagnostic['query_scorer'] = scorer.name
        if state.get('trace_scores'):
            if layer_index == 0:
                fixed_slots = set(state['fixed'])
                self.emit('query_candidates', request_id=state['request_id'], budget=state['budget'],
                          history=state['history'], selected_tokens=state['selected_tokens'],
                          candidate_starts=[r.history_begin for r in records],
                          candidate_counts=[r.count for r in records],
                          eligible=[bool(e) for e in state['score_mask']],
                          fixed_starts=sorted(r.history_begin for r in records
                                              if r.handle.slot in fixed_slots),
                          excluded_starts=sorted(r.history_begin for r in records
                                                 if r.handle.slot in state.get('partial_excluded', ())))
            if scores is not None:
                diagnostic['candidate_scores'] = [float(v) for v in scores]
        if self.diagnostics:
            diagnostic['q_sha256'] = hashlib.sha256(
                q.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        self.emit('layer_selected', request_id=state['request_id'], layer=layer_index,
                  layer_name=self.names[layer_index], query_source='same_forward_current_layer',
                  predecessor_layers_completed=layer_index, scoring_seconds=scoring_seconds,
                  assembly_host_seconds=assembly_seconds, optimized=self.optimized, diagnostics=self.diagnostics,
                  assembly_backend=assembly_stats,
                  no_probe=state.get('no_probe', False),
                  prepared_candidate_index=self.prepared_index,
                  selected_tokens=state['selected_tokens'], charged_tokens=32*len(actual),
                  selected_starts=([r.history_begin for r in records if r.handle.slot in actual]
                    if record_index is None else [r.history_begin for r in chosen]
                    if self.mode == 'sequential' else record_index.starts(actual)),
                  **diagnostic, **position_evidence)
        state['next_layer'] += 1
        if state['next_layer'] == len(self.names):
            mtp_stats = []
            if self.mtp_selection == 'layer0':
                names = tuple(getattr(self.runner, '_kvmem_archive_layers', self.names))
                if names[:len(self.names)] != self.names:
                    raise RuntimeError('MTP archive order differs from the target layer order')
                for index, name in enumerate(names[len(self.names):], len(self.names)):
                    mtp_stats.append(self.assemble_layer_pinned(
                        state, index, state['layer_zero_records'], layer_name=name))
                self.emit('mtp_selected', request_id=state['request_id'], source='layer0',
                          layers=len(mtp_stats), selected_tokens=state['selected_tokens'],
                          selected_starts=[r.history_begin for r in state['layer_zero_records']],
                          assembly_backend=mtp_stats)
            if state.get('no_probe'):
                from .layerwise_no_probe import commit_layer_zero
                commit_layer_zero(self, state)
            if self.pressure is not None and (state.get('no_probe') or self.pressure.rolling_decode):
                self.pressure.maps[state['request_id']] = state['history'], tuple(state['rolling_maps'])
            if state.get('gc_windows') is not None:
                from .layerwise_gc_holds import holds_of
                holds_of(self.pressure if self.pressure is not None else self).protect(
                    state['request_id'], state['authority'], state['gc_windows'])
            if self.optimized:
                # One final fence also makes the phase timing comparable to
                # the original synchronized path; queued last-layer DMA must
                # not be reported as completed work.
                with stage('final_query_completion_wait'):
                    torch.cuda.current_stream(self.runner.device).synchronize()
            if profile is not None:
                profile.finish()
            if state.get('query_region_reader') is not None:
                reader = state['query_region_reader']
                self.emit('query_reader_completed', request_id=state['request_id'],
                          source_blocks=reader.source_blocks, read_chunks=reader.read_chunks,
                          reused_blocks=reader.reused_blocks)
            self._close_query_resources(state)
            self.finished.add(state['request_id'])
            if state.get('no_probe'):
                from .layerwise_no_probe import report_completed
                report_completed(self, state, mtp_stats)
            self.emit('completed', request_id=state['request_id'], layers=len(self.names))
            if self.native_batches:
                self.active_owners.pop(state['request_id'])
            else:
                self.active = None

    def assemble_layer_pinned(self, state, layer_index, chosen, *, layer_name=None,
                              skip_history_begins=()):
        """Reuse native bounded pinned slots, archive leases, and DMA fences.

        Only this layer's byte regions are read. K/V packing, FP8 rounding,
        dense positions and record order are identical to the diagnostic path.
        The following attention runs on the consuming stream, so it requires
        the DMA event dependency rather than a host synchronization per layer.
        """
        from .gil_gate import checkpoint
        if not state.get('restore_gate_exempt', False):     # KVMEM_RESTORE_GATE_EXEMPT
            checkpoint(state.get('request_id'))  # opt-in KVMEM_GIL_GATE_MS: one layer (never for the step's own requests)
        from ..adapters.assembly import assemble_fp8
        from .packed_assembly import read_packed_records
        from .window_assembly import NativeWindowAssembler
        from .query_stage_profile import no_stage
        profile = state.get('query_stage_profile')
        if profile is not None and state.get('assembly_arena') == 'pressure':
            raise RuntimeError('Final Query stage profile cannot measure pressure assembly')
        if profile is not None:
            profile.layer = layer_index
            read_packed_records = profile.read_packed
        public_query = (state.get('assembly_arena') != 'pressure' and
                        'next_layer' in state and 'span' in state)
        if public_query and getattr(self, 'lazy_reader', False):
            prepared_read = self._query_reader(state).read
            if profile is None:
                read_packed_records = prepared_read
            else:
                def read_packed_records(authority, handles, **kwargs):
                    return profile.read_prepared(prepared_read, authority, handles, **kwargs)
        stage = profile.stage if profile is not None else no_stage
        authority = state['authority']
        if authority.raw_element_bytes != 1:
            raise ValueError('Layerwise assembly requires original FP8 content Keys')
        read_blocks = None
        if self.batched_reads:
            from .layerwise_pressure_efficiency import layer_read_blocks
            options = getattr(getattr(self.runner, '_kvmem_capture_runtime', None), 'options', None)
            read_blocks = layer_read_blocks(getattr(options, 'assembly_chunk_blocks', 32), authority.shape[0])
        with stage('assembler_views_prepare'):
            parent = state.get('pressure_parent_assembler')
            if parent is not None:
                if state.get('assembly_arena') != 'pressure' or state.get('pressure_stage') != 'prefill':
                    raise ValueError('Borrowed pressure views escaped their prefill transaction')
                assembler = NativeWindowAssembler.from_pressure_parent(
                    self.runner, state['request_id'], parent, layer_index, 32, read_blocks or 32)
            elif 'borrowed_layers' in state:
                # KVMEM_RESTORE_DIRECT_LAYERS: the restore job's lease-validated
                # views and private tables (no per-layer table clone or GPU check).
                assembler = NativeWindowAssembler((state['borrowed_layers'][layer_index],), 32,
                    read_blocks or 32, staging=state['borrowed_staging'], borrowed_staging=True,
                    raw_dtype=torch.float8_e4m3fn, lease=state['borrowed_lease'])
            else:
                assembler = NativeWindowAssembler.from_runner(
                    self.runner, state['request_id'],
                    (self.names[layer_index] if layer_name is None else layer_name,), 32,
                    chunk_blocks=read_blocks,
                    arena=state.get('assembly_arena', 'layerwise'),
                    # KVMEM_RESTORE_DIRECT_LAYERS: the restore job's own scratch.
                    **({'staging': state['borrowed_staging']} if 'borrowed_staging' in state else {}))
        layer, staging = assembler.layers[0], assembler.staging
        row_bytes = 32 * authority.shape[2] * authority.shape[3]
        total_raw = authority.shape[0] * row_bytes
        chunk_blocks = min(256, staging.slot_bytes // (2 * row_bytes))
        if chunk_blocks < 1:
            raise ValueError('Bounded native staging cannot hold one layer block')
        from .layerwise_pressure_efficiency import assembly_groups, isolate_partial_records
        with stage('assembly_groups_prepare'):
            # Opt-in: records outside the fill's fast path (below) get their own chunk.
            isolate = ((lambda r: not (r.count == 32 and r.archived))
                       if isolate_partial_records() else None)
            groups, total = assembly_groups(chosen, chunk_blocks, skip_history_begins, isolate=isolate)
        uploaded_bytes = chunks = packed_chunks = paired_read_chunks = inline_read_chunks = prepared_read_chunks = 0
        native_paired_gather_chunks = 0
        # Restore and pressure also call this assembler. The candidate belongs
        # only to the layer-by-layer public Query forward, including its MTP.
        paired_native = (getattr(self, 'paired_gather', False) and
                         state.get('assembly_arena') != 'pressure' and
                         'next_layer' in state and 'span' in state)
        inline_cpu = ((state.get('assembly_arena') == 'pressure' and
                       (state.get('pressure_stage') == 'prefill' or
                        state.get('pressure_stage') == 'decode' and decode_inline_reads()) or
                       state.get('restore_inline_reads', False)) and   # KVMEM_RESTORE_PREPARED_READS
                      getattr(authority.archive, 'memory_only', False))
        staging.claim(assembler)
        try:
            if public_query and getattr(self, 'resident_values', False):
                from .query_resident_values import assemble_query_resident_values
                reused = assemble_query_resident_values(self, state, layer_index, chosen,
                    assembler, chunk_blocks, reader=read_packed_records)
                if reused is not None:
                    return reused
            if 'pressure_value_sources' in state:
                from .pressure_resident_values import assemble_resident_values
                reused = assemble_resident_values(self, state, layer_index, chosen,
                    assembler, groups, total, row_bytes, chunk_blocks)
                if reused is not None:
                    return reused
            from .gil_gate import chunk_checkpoint
            for chunk_index, (cursor, group) in enumerate(groups):
                if chunk_index:
                    chunk_checkpoint(state.get('request_id'))   # opt-in KVMEM_GIL_GATE_CHUNKS
                n = sum(r.count for r in group)
                half = n * authority.shape[2] * authority.shape[3]
                with stage('device_staging_destination', transfer_bytes=2*half):
                    gpu, ready, slot = staging.destination(2 * half)
                def fill(host):
                    nonlocal paired_read_chunks, inline_read_chunks, prepared_read_chunks, native_paired_gather_chunks
                    if all(r.count == 32 and r.archived for r in group):
                        handles = [r.handle for r in group]
                        if getattr(self, 'paired_reads', False):
                            regions = ((layer_index*row_bytes, row_bytes),
                                       (total_raw+layer_index*row_bytes, row_bytes))
                            prepared = state.get('prepared_pressure_reader') if inline_cpu else None
                            if prepared is not None and prepared.contains(group):
                                from .pressure_reader import layer0_parallel_gather
                                parallel = (layer_index == 0 and layer0_parallel_gather() and
                                            state.get('promised_layer0', False) or
                                            state.get('restore_parallel_reads', False))
                                _,stats = prepared.read(group, regions, destination=host,
                                                        **({'parallel': True} if parallel else {}))
                            else:
                                _, stats = read_packed_records(authority, handles, destination=host,
                                    regions=regions, inline_cpu=inline_cpu,
                                    **({'paired_native': True} if paired_native else {}))
                            paired_read_chunks += 1
                            inline_read_chunks += int(stats.get('inline_cpu_region_read', False))
                            prepared_read_chunks += int(stats.get('prepared_cpu_region_read', False))
                            native_paired_gather_chunks += int(stats.get('paired_native_gather', False))
                        else:
                            read_packed_records(authority, handles, destination=host[:half],
                                region=(layer_index*row_bytes, row_bytes))
                            read_packed_records(authority, handles, destination=host[half:],
                                region=(total_raw+layer_index*row_bytes, row_bytes))
                    else:
                        cache = state.get('restore_partial_operands')
                        if cache is None:
                            operands = authority.read_records([r.handle for r in group])
                        else:
                            # Direct restore: a staged record's all-layer copy is
                            # read once and sliced for every layer (immutable here).
                            missing = [r for r in group if id(r) not in cache]
                            if missing:
                                for r, pair in zip(missing, authority.read_records([r.handle for r in missing])):
                                    cache[id(r)] = (r, pair)
                            operands = [cache[id(r)][1] for r in group]
                        raw = torch.cat([k[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                        values = torch.cat([v[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                        host[:half].copy_(raw.contiguous().view(torch.uint8).flatten())
                        host[half:].copy_(values.contiguous().view(torch.uint8).flatten())
                transfer_profile = {'profile': profile} if profile is not None else {}
                with stage('reader_upload_total', transfer_bytes=2*half):
                    transfer = assembler.transfers.upload_from_reader(2*half, gpu, fill, ready=ready,
                                                                     **transfer_profile)
                if transfer is None:
                    with stage('pinned_slot_retry_wait'):
                        next(iter(assembler.transfers.pending.values())).event.synchronize()
                    with stage('reader_upload_total', transfer_bytes=2*half):
                        transfer = assembler.transfers.upload_from_reader(2*half, gpu, fill, ready=ready,
                                                                         **transfer_profile)
                if transfer is None:
                    raise RuntimeError('Pinned layerwise transfer made no progress')
                with stage('consumer_wait_for_h2d', gpu=True):
                    torch.cuda.current_stream(self.runner.device).wait_event(transfer.event)
                packed = False
                if getattr(self, 'packed_assembly', False):
                    from .layerwise_pressure_efficiency import packed_layer_chunk
                    with stage('layout_and_rebuild', gpu=True, tokens=n):
                        packed = packed_layer_chunk(state,layer,gpu,half,cursor,group,row_bytes,chunk_blocks)
                if packed:
                    staging.consumed(slot)
                    uploaded_bytes += 2 * half
                    chunks += 1
                    packed_chunks += 1
                    continue
                with stage('layout_and_rebuild_fallback', gpu=True, tokens=n):
                    raw = gpu[:half].view(torch.float8_e4m3fn).reshape(n, authority.shape[2], authority.shape[3])
                    values = gpu[half:].view(torch.float8_e4m3fn).reshape_as(raw)
                    positions = torch.arange(cursor, cursor+n, device=gpu.device, dtype=torch.int64)
                    page_tokens = layer.key.shape[1]
                    slots = layer.kernel_page_table[positions//page_tokens].long()*page_tokens + positions%page_tokens
                    rp = record_positions(group, gpu.device) if original_rope() else positions
                    if original_rope() and max(r.history_begin+r.count for r in group) > layer.cos_sin.shape[0]:
                        raise ValueError('Original Key position exceeds RoPE table')
                    assemble_fp8(raw, values, rp, slots, layer.cos_sin, layer.key, layer.value, layer.key_scale)
                staging.consumed(slot)
                uploaded_bytes += 2 * half
                chunks += 1
            if total != state['selected_tokens']:
                raise RuntimeError('Layer assembly did not cover the original selected-token extent')
        except BaseException:
            # Native scratch cannot be reused until every submitted copy and
            # consumer has drained, including a partially submitted chunk.
            assembler.transfers.stream.synchronize() if assembler.transfers is not None else None
            torch.cuda.current_stream(self.runner.device).synchronize()
            raise
        finally:
            with stage('assembler_release'):
                staging.release(assembler)
                assembler.close()
        return dict(uploaded_bytes=uploaded_bytes, chunks=chunks, packed_chunks=packed_chunks,
                    chunk_blocks=chunk_blocks, staging_slot_bytes=staging.slot_bytes,
                    paired_read_chunks=paired_read_chunks, inline_read_chunks=inline_read_chunks,
                    prepared_read_chunks=prepared_read_chunks,
                    native_paired_gather_chunks=native_paired_gather_chunks,
                    **({'query_resident_v': False,
                        'query_resident_fallback_reason': state.get('query_resident_fallback_reason')}
                       if public_query and getattr(self, 'resident_values', False) else {}))

    def assemble_layer(self, state, layer_index, chosen):
        from ..adapters.assembly import assemble_fp8
        from .packed_assembly import read_packed_records
        from .window_assembly import NativeWindowAssembler
        authority = state['authority']
        assembler = NativeWindowAssembler.from_runner(
            self.runner, state['request_id'], (self.names[layer_index],), 32, arena='layerwise')
        layer = assembler.layers[0]
        row_bytes = 32*authority.shape[2]*authority.shape[3]
        total_raw = authority.shape[0]*row_bytes*authority.raw_element_bytes
        if authority.raw_element_bytes != 1:
            raise ValueError('Layerwise diagnostic requires original FP8 content Keys')
        cursor = 0
        try:
            for begin in range(0, len(chosen), 32):
                group = chosen[begin:begin+32]
                if all(r.count == 32 and r.archived for r in group):
                    handles = [r.handle for r in group]
                    kb, _ = read_packed_records(authority, handles,
                        region=(layer_index*row_bytes, row_bytes))
                    vb, _ = read_packed_records(authority, handles,
                        region=(total_raw+layer_index*row_bytes, row_bytes))
                    raw = kb.view(torch.float8_e4m3fn).reshape(-1, authority.shape[2], authority.shape[3])
                    values = vb.reshape_as(raw)
                else:
                    operands = authority.read_records([r.handle for r in group])
                    raw = torch.cat([k[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                    values = torch.cat([v[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                raw = raw.to(self.runner.device)
                values = values.view(torch.float8_e4m3fn).to(self.runner.device)
                positions = torch.arange(cursor, cursor+len(raw), device=raw.device, dtype=torch.int64)
                page_tokens = layer.key.shape[1]
                slots = layer.kernel_page_table[positions//page_tokens].long()*page_tokens + positions%page_tokens
                rp = record_positions(chosen[begin:begin+32], raw.device) if original_rope() else positions
                assemble_fp8(raw, values, rp, slots, layer.cos_sin, layer.key, layer.value, layer.key_scale)
                cursor += len(raw)
            torch.cuda.current_stream().synchronize()
            if cursor != state['selected_tokens']:
                raise RuntimeError('Layer assembly did not cover the original selected-token extent')
        finally:
            assembler.close()


def install_layerwise_query(runner):
    from .async_refresh import AsyncRefresh
    from ..adapters.capture import _sink, capture_to
    controller = runner._kvmem_layerwise_query = LayerwiseQuery(runner)
    modules = dict(runner.model.named_modules())
    for index, name in enumerate(controller.names):
        module = modules[name]
        original = module._project_qkv_gate
        def project(self, qkv, positions, _index=index, _original=original):
            if not controller.has_active:
                return _original(qkv, positions)
            buffers = getattr(self, '_kvmem_capture_buffers', None)
            if buffers is not None:
                # Persistent capture deliberately bypasses ContextVar sinks.
                # Its fused native projection has just written these buffers
                # on the consuming stream; use the current rows immediately.
                from ..adapters.capture import CapturedLayer
                result = _original(qkv, positions)
                rows = qkv.shape[0]
                if rows > buffers.query.shape[0]:
                    raise RuntimeError('Current Query exceeds native capture capacity')
                record = CapturedLayer(buffers.name, positions, buffers.query[:rows],
                                       buffers.key[:rows], buffers.value[:rows])
                controller.projection(_index, record)
                return result
            parent = _sink.get()
            captured = []
            def receive(record):
                captured.append(record)
                if parent is not None:
                    parent(record)
            with capture_to(receive):
                result = _original(qkv, positions)
            if len(captured) != 1:
                raise RuntimeError('Layerwise projection did not capture exactly one real Q')
            controller.projection(_index, captured[0])
            return result
        module._project_qkv_gate = MethodType(project, module)
    if not getattr(AsyncRefresh, '_layerwise_diagnostic_installed', False):
        assemble, report = AsyncRefresh._assemble, AsyncRefresh.report
        def assemble_with_probe(self, job, queries, ready):
            from .gil_gate import checkpoint
            checkpoint(job.command.request_id)   # opt-in KVMEM_GIL_GATE_MS: refresh job start
            if (getattr(self.options, 'action_profiling', False) or
                    getattr(self.runner, '_kvmem_step_timing', None) is not None):
                self.emit(job, 'assembly_worker_started')
            diagnostic = getattr(self.runner, '_kvmem_layerwise_query', None)
            if diagnostic is not None:
                with torch.inference_mode(), torch.cuda.device(self.runner.device), torch.cuda.stream(self.stream):
                    self.stream.wait_event(ready)
                    diagnostic.prepare(self, job, queries)
            return assemble(self, job, queries, ready)
        def report_with_probe(self, job, assembly, fit, seconds):
            result = report(self, job, assembly, fit, seconds)
            diagnostic = getattr(self.runner, '_kvmem_layerwise_query', None)
            if diagnostic is not None and not getattr(job, 'layerwise_pipeline_published', False):
                diagnostic.committed(job)
            return result
        AsyncRefresh._assemble, AsyncRefresh.report = assemble_with_probe, report_with_probe
        AsyncRefresh._layerwise_diagnostic_installed = True
    if controller.native_batches:
        install_native_query_dispatch(controller)


def install_native_query_dispatch(controller):
    """Keep native compiled/graph execution except an armed layerwise Query.

    The Query requires Python-controlled archive access between layers. Native
    dispatch and its existing skip_compiled contract provide that boundary;
    normal prefill and decode retain the configured batch/compiled execution.
    """
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import model_runner
    original_dispatch = model_runner.dispatch_cg_and_sync_dp
    original_context = model_runner.set_forward_context
    if getattr(original_dispatch, '_kvmem_layerwise_owner', None) is not None:
        raise RuntimeError('A second layerwise dispatcher cannot share the native worker')
    def dispatch(*args, **kwargs):
        pipeline = getattr(controller.pressure, 'pipeline', None)
        if controller.has_active:
            kwargs['need_eager'] = True
        result = original_dispatch(*args, **kwargs)
        # Native prefill PIECEWISE graphs already exclude the opaque KV
        # writer/attention ops. Preserve those compute graphs and shapes;
        # only a whole-model FULL replay would bypass the per-layer waits.
        if (pipeline is not None and result[0].cg_mode == CUDAGraphMode.FULL and
                pipeline.needs_dispatch_boundary):
            kwargs['need_eager'] = True
            result = original_dispatch(*args, **kwargs)
        return result
    def context(*args, **kwargs):
        if controller.has_active:
            kwargs['skip_compiled'] = True
        return original_context(*args, **kwargs)
    dispatch._kvmem_layerwise_owner = controller
    model_runner.dispatch_cg_and_sync_dp = dispatch
    model_runner.set_forward_context = context
