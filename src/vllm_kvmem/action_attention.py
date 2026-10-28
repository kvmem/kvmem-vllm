"""Request-owned action feedback with qw3's transactional scope parser.

Output feedback uses only scheduler-confirmed, consumed rows. Optional input
feedback separately tracks sampled prefill mass and causal exposure. Private
guided Q, public Query replay, MTP draft layers and rejected rows contribute
neither. Both score histories are checkpointed with the exact historical prefix.
"""
from .native_pages import kernel_table
from dataclasses import dataclass, field, replace
from contextlib import nullcontext
import json
import math
import re
import time

import torch

from . import _native
from .codec import native_vllm_fp8_views


class _AcceptanceParser:
    """Read-only acceptance preview around the unchanged native scope parser.

    Graph reservations must succeed before native commit advances thinking,
    interval and accepted-prefix state. Only enabled graph sessions pay for this
    wrapper; both legacy feedback and graph then consume the same commit mask.
    """
    def __init__(self, parser):
        self.parser = parser
        self.mask = self.count = self.accepted = 0

    @property
    def state(self):
        return self.parser.state

    def prepare(self, tokens):
        mask = self.parser.prepare(tokens)
        self.mask, self.count, self.accepted = mask, len(tokens), 0
        return mask

    def preview(self, accepted):
        if not isinstance(accepted, int) or not self.accepted <= accepted <= self.count:
            raise ValueError('Invalid action attention acceptance prefix')
        return self.mask & ((1 << accepted)-1) & ~((1 << self.accepted)-1)

    def commit(self, accepted):
        expected = self.preview(accepted)
        result = self.parser.commit(accepted)
        if result != expected:
            raise RuntimeError('Action acceptance preview differs from native parser')
        self.accepted = accepted
        return result


class ActionHistory:
    def __init__(self, options, token_ids):
        self.options,self.token_ids=options,token_ids
        self.scores={}
        self.request_id=None
        self.parser=None
        self.mass={}
        self.samples=0
        self.rounds=0
        self.replay_snapshot=None
        from .prefill_attention import PrefillHistory
        self.prefill = PrefillHistory(options) if getattr(options,'prefill_attention',False) else None
        self.prefill_replay = None
        self.prefill_planning_epoch = 0

    def begin(self, request_id, prompt):
        if self.request_id==request_id:
            return
        thinking=False
        # The last message header resets historical assistant thinking markers.
        # Only the last scope marker determines the initial state. Agent
        # prompts usually end in a fresh message header; do not rescan the
        # complete retained history on every tool return.
        for token in reversed(prompt):
            if token in (self.token_ids[4], self.token_ids[2], self.token_ids[3]):
                thinking = token == self.token_ids[2]
                break
        self.parser=_native.ActionParser(self.options.action_scope,self.options.action_interval,
                                        self.token_ids[:4],thinking)
        if (getattr(self.options, 'graph_mode', 'off') != 'off' or
                getattr(self.options, 'action_native_decode_commit', False)):
            self.parser = _AcceptanceParser(self.parser)
        self.request_id=request_id
        self.mass={};self.samples=0

    def accept(self, accepted, rows, history, masses):
        if getattr(self.options, 'action_native_decode_commit', False):
            mask = self.parser.preview(accepted)
            indices = [i for i,row in enumerate(rows) if (mask >> row) & 1]
            if not indices:
                if self.parser.commit(accepted) != mask:
                    raise RuntimeError('Action transaction lost its accepted prefix')
                return
            if hasattr(masses, 'numpy'):
                masses = masses.numpy()
            elif not hasattr(masses, 'shape'):
                import numpy as np
                masses = np.asarray([] if masses is None else masses, dtype='float64').reshape(len(rows),len(history))
            updated, samples = _native.prepare_decode_attention(self.mass,self.samples,indices,
                history,masses,getattr(self.options,'action_commit_max_bytes',16777216))
            if self.parser.commit(accepted) != mask:
                raise RuntimeError('Action transaction lost its accepted prefix')
            self.mass, self.samples = updated, samples
            return
        mask=self.parser.commit(accepted)
        for i,row in enumerate(rows):
            if not (mask>>row)&1:
                continue
            values=masses[i]
            if any(not math.isfinite(v) or v<0 for v in values) or abs(sum(values)-1)>.005:
                raise RuntimeError('Action attention mass is invalid or does not sum to one')
            for key,value in zip(history,values):
                if value: self.mass[key]=self.mass.get(key,0.0)+value
            self.samples+=1

    def finish(self, authority):
        live={r.history_begin for r in authority.records.values()}
        if self.samples:
            self.scores={key:self.options.action_decay*self.scores.get(key,0.0)+
                         self.mass.get(key,0.0)/self.samples for key in live}
            self.rounds+=1
        else:
            self.scores={key:value for key,value in self.scores.items() if key in live}
        authority.memory.set_action_scores(self.scores)
        report={'samples':self.samples,'rounds':self.rounds,
                'positive_pages':sum(v>0 for v in self.scores.values()),
                'score_sum':sum(self.scores.values()),'scope':self.options.action_scope}
        self.parser=None; self.mass={}
        return report

    def snapshot(self, position=None):
        if self.replay_snapshot is not None and position is not None and position<self.replay_snapshot[0]:
            # A replay-only P checkpoint cannot publish feedback whose accepted
            # source extends beyond that checkpoint to a future edited branch.
            from copy import deepcopy
            return deepcopy(self.replay_snapshot[1])
        result = (dict(self.scores),self.rounds)
        return result + (self.prefill.snapshot(),) if self.prefill is not None else result

    def restore(self, snapshot, authority, *, preserve_future=False, validated_end=None):
        # An M->P model rewind can retain feedback through M only when the
        # checkpoint claim compared the entire M token prefix. The physical
        # authority ends at P here; it is not the feedback validity boundary.
        if validated_end is not None:
            if (not preserve_future or type(validated_end) is not int or
                    validated_end < authority.history_end):
                raise ValueError('Invalid validated action-history prefix')
        prefill_snapshot = snapshot[2] if len(snapshot)>2 else ({},0,0,0)
        if self.prefill is not None:
            prefill_end = authority.history_end if validated_end is None else validated_end
            if preserve_future and prefill_snapshot[1] > prefill_end:
                raise ValueError('Future prefill history requires its complete validated prefix')
        scores,self.rounds=snapshot[:2]
        live={r.history_begin for r in authority.records.values()}
        self.scores={key:value for key,value in scores.items() if preserve_future or key in live}
        self.request_id=None;self.parser=None;self.mass={};self.samples=0
        self.replay_snapshot=None
        self.prefill_replay=None
        self.prefill_planning_epoch += 1
        if self.prefill is not None:
            self.prefill.restore(prefill_snapshot, prefill_end)
        authority.memory.set_action_scores(self.scores)


def window_segments(records, history_begin, local_begin, count, block_tokens):
    """Freeze logical identities, including a partial historical tail."""
    starts=[];counts=[];history=[];cursor=0
    for _,_,begin,n,position in records:
        if position>=local_begin: break
        n=min(n,local_begin-position)
        if position!=cursor or n<=0 or n>block_tokens:
            raise ValueError('Action window has a gap or stale dense mapping')
        starts.append(cursor);counts.append(n);history.append(begin-begin%block_tokens)
        cursor+=n
    logical=history_begin+cursor-local_begin
    if logical<0: raise ValueError('Invalid action suffix origin')
    while cursor<local_begin+count:
        n=min(block_tokens-logical%block_tokens,local_begin+count-cursor)
        starts.append(cursor);counts.append(n);history.append(logical-logical%block_tokens)
        cursor+=n;logical+=n
    return starts,counts,history


@dataclass
class ActionPacket:
    state: ActionHistory
    rows: list
    history: list
    masses: object
    event: object
    prefill: tuple | None = None
    # Original consumed-Q coordinates, never predicted-token (+1), dense
    # working positions, GPU page indices or a temporary request identity.
    source_positions: tuple = ()
    history_begin: int | None = None
    candidate_count: int = 0
    # Physical 4K is never itself an acceptance transaction. Only its two
    # validated logical views may reach the existing 2K consumer.
    logical_group: object = None


@dataclass
class _MTPActionFrame:
    batch: object
    captured: tuple
    metadata: object
    samples: int
    packet_count: int
    trace: object
    packets: dict = field(default_factory=dict)
    ticket: object = None
    deferred: object = None


class _PreparedActionAcceptance:
    """A fully validated, reserved graph/legacy publication transaction."""
    def __init__(self, runtime, packet, request_id, pending, graph_update,
                 *, accepted, output=None, prefill=None):
        self.runtime, self.packet, self.request_id = runtime, packet, request_id
        self.pending, self.graph_update = pending, graph_update
        self.accepted, self.output, self.prefill = accepted, output, prefill
        self.done = False

    def commit(self):
        if self.done:
            raise RuntimeError('Action acceptance transaction already finished')
        self.graph_update.resume()
        pending = self.pending
        if self.output is not None:
            mask, mass, samples = self.output
            # All row validation and memory reservation precede this sole
            # parser commit. The graph must not call commit independently.
            if pending.state.parser.commit(self.accepted) != mask:
                raise RuntimeError('Action transaction lost its accepted prefix')
            pending.state.mass, pending.state.samples = mass, samples
        elif self.prefill is not None:
            original, staged, committed = self.prefill
            original.__dict__ = staged.__dict__
        started = time.perf_counter()
        self.graph_update.commit()
        self.runtime.graph_commit_seconds = getattr(self.runtime,'graph_commit_seconds',0.) + time.perf_counter()-started
        self.runtime.graph_commit_calls = getattr(self.runtime,'graph_commit_calls',0) + 1
        self.packet.action_inputs.pop(self.request_id, None)
        if self.output is not None:
            backend = 'native' if getattr(pending.state.options,'action_native_decode_commit',False) else 'python'
            key = 'decode_commit_'+backend+'_calls'
            setattr(self.runtime,key,getattr(self.runtime,key,0)+1)
        self.done = True
        if self.prefill is not None and self.prefill[2]:
            self.runtime._report_prefill(self.request_id, pending)
        self.output = self.prefill = None

    def abort(self):
        if not self.done:
            self.output = self.prefill = None
            self.graph_update.abort()
            self.done = True


@dataclass
class ActionMetadata:
    batch: object
    indices: tuple
    slices: tuple
    host: object
    event: object


def action_metadata(batch, local_positions, prefill_attention, profile=None, pool=None):
    """Snapshot exact GPU coordinates before the target forward is enqueued.

    CPU scheduler counters are only upper bounds under speculative decoding.
    This copies the actual IDs and positions, and records a fence immediately
    after that copy, independently of the later target and draft kernels.
    """
    indices = tuple(i for i in range(batch.num_reqs)
                    if not batch.is_prefilling_np[i] or prefill_attention)
    if not indices:
        return None
    slices = tuple((int(batch.query_start_loc_np[i]), int(batch.query_start_loc_np[i+1]))
                   for i in indices)
    if local_positions.ndim == 2:
        local_positions = local_positions[0]
    with profile.stage('metadata.pack', gpu=True,
                       shape=[3, sum(b-a for a, b in slices)]) if profile else nullcontext():
        packed = torch.cat([torch.stack((batch.input_ids[a:b].long(), batch.positions[a:b],
                                        local_positions[a:b])) for a, b in slices], dim=1)
    event = None
    if packed.device.type == 'cuda' and pool is not None:
        host,event=pool.copy(packed)
    elif packed.device.type == 'cuda':
        with profile.stage('metadata.pinned_allocate', shape=list(packed.shape),
                           allocation_bytes=packed.numel()*packed.element_size()) if profile else nullcontext():
            host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=True)
        with profile.stage('metadata.d2h', gpu=True, transfer_bytes=packed.numel()*packed.element_size(),
                           direction='D2H') if profile else nullcontext():
            host.copy_(packed, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(packed.device))
    else:
        host = packed.clone()
    return ActionMetadata(batch, indices, slices, host, event)


class ActionRuntime:
    def __init__(self, runtime):
        self.runtime=runtime
        self.runner=runtime.runner
        self.options=runtime.options
        self.native = getattr(self.runner, '_kvmem_native_attention', None)
        self.native_plans = None
        self.native_candidates = None
        self.native_cache = {}
        self.prefill_plans = None
        self.prefill_plan_frames = {}
        self.pipeline_requests = set()
        self._mtp_capture_frame = None
        self._deferred_frames = []
        self.deferred_score_pool = None
        self.mtp_stage_stats = {}
        self.capture_seconds=0.0
        self.candidate_samples=0
        self.packets=0
        self.metadata = None
        self.metadata_early_batches = 0
        self.metadata_fallback_batches = 0
        self.metadata_wait_seconds = 0.0
        self.metadata_upload = None
        self.download_pool = None
        if getattr(self.options,'action_download_pool',False):
            from .action_download_pool import ActionDownloadPool
            self.download_pool=ActionDownloadPool(self.options.action_download_max_bytes,
                self.options.action_download_max_slots)
        if getattr(self.options, 'async_action_metadata', False):
            from .action_metadata_upload import PackedActionMetadata
            self.metadata_upload = PackedActionMetadata()
        self.profile = None
        if getattr(self.options, 'action_profiling', False):
            from .action_profile import ActionProfiler
            self.profile = ActionProfiler(runtime.output/'action_profile.jsonl',
                interval=self.options.action_profile_interval,
                max_events=self.options.action_profile_max_events)
        from vllm.tokenizers import get_tokenizer
        cfg=self.runner.vllm_config.model_config
        tokenizer=get_tokenizer(cfg.tokenizer,tokenizer_mode=cfg.tokenizer_mode,
                                trust_remote_code=cfg.trust_remote_code)
        self.ids=[tokenizer.convert_tokens_to_ids(s) for s in
                  ('<tool_call>','</tool_call>','<think>','</think>','<|im_start|>')]
        if any(x is None or x<0 for x in self.ids) or len(set(self.ids))!=5:
            raise ValueError('Action attention requires distinct Qwen scope tokens')
        names=self.runner._kvmem_layers
        ids=[int(re.search(r'layers\.(\d+)\.',name).group(1)) for name in names]
        selected=_native.resolve_action_layers(self.options.action_layers,ids)
        self.layers=[(i,runtime.layers[i]) for i,number in enumerate(ids) if number in selected]
        self.groups={name:group for group,g in enumerate(self.runner.kv_cache_config.kv_cache_groups)
                     for name in g.layer_names}
        self.q_dtypes={}
        self.prefill_q_dtypes={}
        for groups in self.runner.attn_groups:
            for group in groups:
                builder=group.get_metadata_builder(0)
                for name in group.layer_names:
                    if hasattr(builder,'q_data_type_decode'):
                        self.q_dtypes[name]=builder.q_data_type_decode
                        self.prefill_q_dtypes[name]=builder.q_data_type_prefill
        for _,module in self.layers:
            impl=module.attn.impl
            if (module.attn.layer_name not in self.q_dtypes or
                    getattr(impl,'logits_soft_cap',None) not in (None,0) or
                    getattr(impl,'window_left',-1)!=-1 or getattr(impl,'sinks',None) is not None):
                raise ValueError('Action sampling requires qualified full causal FlashInfer attention')
        if (getattr(self.options, 'prefill_deferred_validation', False) and
                self.native is not None and not self.native.disabled):
            from .deferred_action_scoring import FrozenScoreLimits, FrozenScorePool
            modules = [module for _, module in self.layers]
            groups = tuple(dict.fromkeys(self.groups[module.attn.layer_name] for module in modules))
            self._deferred_groups = groups
            geometry = {(module.num_heads, module.head_dim, module.num_kv_heads) for module in modules}
            pages = {native_vllm_fp8_views(module.attn.kv_cache, module.num_kv_heads,
                         module.head_dim)[0].shape[1] for module in modules}
            if len(geometry) != 1 or len(pages) != 1:
                raise ValueError('Deferred prefill scoring requires uniform selected-layer geometry')
            heads, dim, _ = geometry.pop()
            limits = FrozenScoreLimits(layers=len(modules), groups=len(groups), heads=heads,
                dim=dim, page_tokens=pages.pop(), block_tokens=runtime.config.block_tokens,
                max_rows=self.native.max_rows, max_owners=self.runner.max_num_reqs,
                max_segments=self.native.max_segments, max_length=self.native.context)
            self.deferred_score_pool = FrozenScorePool(limits, self.runner.device,
                fused=self.options.action_fused_scoring,
                max_bytes=self.options.prefill_deferred_validation_max_bytes,
                max_scratch_bytes=self.options.prefill_deferred_validation_scratch_bytes)
        (runtime.output/'action_config.json').write_text(json.dumps({
            'layers':selected,'scope':self.options.action_scope,'interval':self.options.action_interval,
            'decay':self.options.action_decay,'fraction':runtime.config.action_fraction,
            'combine':runtime.config.action_combine,'q_dtype':{k:str(v) for k,v in self.q_dtypes.items()},
            'prefill_attention':self.options.prefill_attention,'action_pressure':self.options.action_pressure,
            'prefill_interval':self.options.prefill_attention_interval,
            'prefill_stride':self.options.prefill_attention_stride,
            'prefill_half_life':self.options.prefill_attention_half_life,
            'native_action_metadata':getattr(self.options,'native_action_metadata',False),
            'async_action_metadata':getattr(self.options,'async_action_metadata',False),
            'prefill_publication_pipeline':getattr(self.options,'prefill_publication_pipeline',False),
            'prefill_native_plan_arm':getattr(self.options,'prefill_native_plan_arm',False),
            'prefill_action_before_mtp':getattr(self.options,'prefill_action_before_mtp',False),
            'prefill_deferred_validation':getattr(self.options,'prefill_deferred_validation',False),
            'prefill_q_dtype':{k:str(v) for k,v in self.prefill_q_dtypes.items()},
            'native_scores':getattr(self.options,'action_native_scores','off'),
            'native_decode_commit':getattr(self.options,'action_native_decode_commit',False),
            'download_pool':getattr(self.options,'action_download_pool',False),
            'fused_scoring':getattr(self.options,'action_fused_scoring',False),
            'native_scales':{module.attn.layer_name:{name:getattr(module.attn,'_'+name+'_scale_float',None)
                for name in ('q','k','v')} for _,module in self.layers},
            'implementation':'sampled target Q against native working K; original qw3 parser/fusion'},indent=2))

    def begin_response(self, state, request_id):
        if state.request_id == request_id:
            return
        sessions=getattr(self.runner,'_kvmem_sessions',None)
        binding=sessions.bindings.get(request_id) if sessions is not None else None
        if binding is not None:
            prompt=binding.tokens
        else:
            # Stateless requests and sessions admitted without a cache slot
            # still have an exact native prompt. Do not assume every request
            # owns a retained P/M binding; never include generated draft IDs.
            states=self.runner.req_states
            index=states.req_id_to_index[request_id]
            count=int(states.prompt_len.np[index])
            prompt=states.all_token_ids.gpu[index,:count].cpu().tolist()
        state.begin(request_id,prompt)

    def prepare(self, batch, local_positions):
        # Called only for public forwards. Private Query/guided replays must
        # not overwrite the packet associated with a public native batch.
        if getattr(self, '_mtp_capture_frame', None) is not None:
            raise RuntimeError('Action prepare cannot overwrite an unresolved MTP capture')
        if getattr(self.options, 'prefill_action_before_mtp', False):
            self.runtime._raise_publication_error()
        self._collect_deferred_scores()
        profiler = getattr(self, 'profile', None)
        trace = profiler.begin_batch(batch) if profiler is not None else None
        self.metadata = action_metadata(batch, local_positions, self.options.prefill_attention, trace,
            pool=getattr(self,'download_pool',None))
        self.pipeline_requests = set()
        self.prefill_plans = None
        if getattr(self.options, 'prefill_publication_pipeline', False):
            self._prepare_prefill_pipeline(batch)
        if getattr(self, 'native', None) is not None:
            self.native_plans = None
            self.native_candidates = None
            if self.metadata is None or self.native.disabled:
                self.native.disarm()
                return
            if (getattr(self.options, 'prefill_native_plan_arm', False) and
                    self._arm_prefill_plans(batch)):
                return
            # One owner benefits from direct scope preparation; concurrent
            # decode needs to retain overlap with prior owner publication.
            if len(batch.req_ids)>1 and self._prepare_decode_candidates(batch):
                self.native.counts['prepare:decode_candidates'] += 1
                return
            self.native.counts['prepare:eager'] += 1
            # Native QK runs before capture(). Resolve exactly the SAME scope,
            # stride and logical segments now, without committing acceptance.
            if self.metadata.event is not None:
                self.metadata.event.synchronize()
            tokens, global_positions, local = self.metadata.host.tolist()
            plans = {}; cursor = 0
            for i, (begin, end) in zip(self.metadata.indices, self.metadata.slices):
                request_id, count = batch.req_ids[i], end - begin
                prepared = self.prefill_plans
                plan = prepared[1].get(request_id) if prepared is not None and prepared[0] is batch else None
                if plan is None:
                    plan = self._plan_owner(request_id, bool(batch.is_prefilling_np[i]),
                        tokens[cursor:cursor+count], global_positions[cursor:cursor+count],
                        local[cursor:cursor+count], count)
                else:
                    self._validate_prefill_plan(plan, global_positions[cursor:cursor+count],
                                                local[cursor:cursor+count], count)
                cursor += count
                plan.update(index=i, begin=begin, count=count)
                if plan['rows'] and 'starts' not in plan:
                    starts, counts, history = self._window_segments(plan['authority'],
                        plan['history_begin'], plan['local_begin'], count)
                    plan.update(starts=starts, counts=counts, history=history)
                plans[request_id] = plan
            self.native_plans = (batch, plans)
            self.native.arm(plans)

    def _arm_prefill_plans(self, batch):
        """Use exact ordinary-input plans without fencing their metadata D2H.

        The actual GPU coordinates remain copied and are checked by capture()
        before any score packet can escape. This only moves that check past
        the model forward; native numerical validation and fallback stay intact.
        """
        prepared = getattr(self, 'prefill_plans', None)
        if prepared is None or prepared[0] is not batch or not self.metadata.indices:
            return False
        plans = {}
        for i, (begin, end) in zip(self.metadata.indices, self.metadata.slices):
            request_id = batch.req_ids[i]
            plan = prepared[1].get(request_id)
            if (not bool(batch.is_prefilling_np[i]) or plan is None or not plan.get('pipeline', False) or
                    plan['index'] != i or plan['begin'] != begin or plan['count'] != end - begin):
                return False
            plans[request_id] = plan
        self.native.counts['prepare:prefill_plan'] += 1
        self.native_plans = (batch, plans)
        self.native.arm(plans)
        return True

    def invalidate_prefill_plans(self, request_ids=None):
        frames = getattr(self, 'prefill_plan_frames', {})
        if request_ids is None:
            frames.clear()
        else:
            for request_id in request_ids:
                frames.pop(request_id, None)

    def _prefill_identity(self, request_id, state):
        adapter = getattr(self.runner, '_kvmem_window_adapter', None)
        return (getattr(self.runner, '_kvmem_request_generations', {}).get(request_id, 0),
                getattr(adapter, 'generations', {}).get(request_id, 0),
                getattr(adapter, 'offsets', {}).get(request_id, 0),
                getattr(state, 'prefill_planning_epoch', 0), state.prefill_replay)

    def prefill_pipeline_ready(self, request_id, history_begin, local_begin):
        frame = getattr(self, 'prefill_plan_frames', {}).get(request_id)
        authority = getattr(self.runtime, 'authorities', {}).get(request_id)
        return (frame is not None and authority is frame.authority and
                getattr(authority, 'action_history', None) is frame.state and
                frame.matches(authority, self._prefill_identity(request_id, frame.state),
                              history_begin, local_begin))

    def _prepare_prefill_pipeline(self, batch):
        """Freeze ordinary input plans before forward, without GPU metadata waits.

        Only nonterminal prompt chunks have exact CPU positions and cannot
        include speculative output rows. Decode and public Query replay retain
        their original accepted-prefix fence. The runtime separately limits
        this optimization to one prior confirmed packet per owner.
        """
        if not self.options.prefill_attention or not hasattr(batch, 'num_computed_tokens_np'):
            return
        logical_groups = getattr(self.runner, '_kvmem_logical_prefill_groups', {})
        if logical_groups and (not getattr(self.options, 'prefill_physical_coalescing', False) or
                len(batch.req_ids) != 1 or set(logical_groups) != set(batch.req_ids)):
            raise RuntimeError('Logical prefill receipt requires the qualified single-owner batch')
        plans = {}
        frames = self.__dict__.setdefault('prefill_plan_frames', {})
        states = getattr(self.runner, 'req_states', None)
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        bindings = getattr(sessions, 'bindings', {})
        for i, request_id in enumerate(batch.req_ids):
            begin, end = map(int, batch.query_start_loc_np[i:i+2])
            count = end - begin
            history_begin = int(batch.num_computed_tokens_np[i])
            prompt = None
            native_index = getattr(states, 'req_id_to_index', {}).get(request_id)
            if native_index is not None and hasattr(states, 'prompt_len'):
                prompt = int(states.prompt_len.np[native_index])
            elif request_id in bindings:
                prompt = len(bindings[request_id].tokens)
            if (not bool(batch.is_prefilling_np[i]) or count <= 0 or prompt is None or
                    history_begin + count >= prompt):
                if request_id in logical_groups:
                    raise RuntimeError('Logical prefill receipt includes an ineligible prompt span')
                frames.pop(request_id, None)
                continue
            authority = getattr(self.runtime, 'authorities', {}).get(request_id)
            frame = frames.get(request_id)
            excluded = getattr(getattr(authority, 'action_history', None), 'prefill_replay', None)
            if excluded is not None and history_begin < excluded[1] and history_begin + count > excluded[0]:
                if request_id in logical_groups:
                    raise RuntimeError('Logical prefill receipt overlaps excluded Query replay')
                frames.pop(request_id, None)
                continue
            identity = self._prefill_identity(request_id, frame.state) if frame is not None else None
            offset = (identity[2] if identity is not None else
                getattr(getattr(self.runner, '_kvmem_window_adapter', None), 'offsets', {}).get(request_id, 0))
            local_begin = history_begin - offset
            matches = (frame is not None and getattr(authority, 'action_history', None) is frame.state and
                       frame.matches(authority, identity, history_begin, local_begin))
            can_advance = getattr(self.runtime, 'prefill_pipeline_can_advance', None)
            if not (matches and can_advance is not None and can_advance(request_id)):
                frames.pop(request_id, None)
                if not self.runtime.drain_requests((request_id,), wait=True):
                    raise RuntimeError('Action capture preceded scheduler confirmation')
                authority = self.runtime.ensure_authority(request_id)
                state = getattr(authority, 'action_history', None)
                if state is None:
                    state = authority.action_history = ActionHistory(self.options, self.ids)
                identity = self._prefill_identity(request_id, state)
                local_begin = history_begin - identity[2]
                excluded = state.prefill_replay
                if excluded is not None and history_begin < excluded[1] and history_begin + count > excluded[0]:
                    if request_id in logical_groups:
                        raise RuntimeError('Logical prefill receipt overlaps restored Query replay')
                    # _plan_owner retains the original overlap validation.
                    continue
                from .prefill_planning import freeze_prefill_frame
                frame = freeze_prefill_frame(authority, state, identity, history_begin,
                    local_begin, authority.memory._store.working_records())
            state = frame.state
            logical = logical_groups.get(request_id)
            extra = {}
            if logical is not None:
                logical, score_plans, rows, next_frame = self._plan_logical_prefill(
                    logical, frame, batch, i, begin, count, history_begin, local_begin, prompt)
                final = score_plans[-1]
                rows, starts, counts, history = list(rows), list(final.starts), list(final.counts), list(final.history)
                extra = dict(logical_group=logical, logical_score_plans=score_plans)
            else:
                rows, starts, counts, history, next_frame = frame.extend(
                    history_begin, local_begin, count, self.runtime.config.block_tokens)
            frames[request_id] = next_frame
            sources = (tuple(history_begin + row for row in rows)
                       if getattr(authority, 'block_graph', None) is not None else ())
            plans[request_id] = dict(authority=authority, state=state, history_begin=history_begin,
                local_begin=local_begin, rows=rows, sources=sources, starts=starts,
                counts=counts, history=history, index=i, begin=begin, count=count,
                pipeline=True, **extra)
            self.pipeline_requests.add(request_id)
        self.prefill_plans = (batch, plans)

    def _plan_logical_prefill(self, group, frame, batch, index, begin, count,
                              history_begin, local_begin, prompt):
        """Bind the scheduler receipt to this worker before native forward.

        A rejected receipt is fail-closed: a scheduled physical group cannot be
        silently consumed as one new statistical transaction. The scheduler
        chooses ordinary 2K for requests outside the conservative gates.
        """
        from .logical_action import plan_logical_scores
        from .logical_prefill import LogicalPrefillGroup
        config, native = self.runtime.config, getattr(self, 'native', None)
        step = getattr(self.runner, '_kvmem_logical_prefill_step', None)
        if (not isinstance(group, LogicalPrefillGroup) or group.request_id != batch.req_ids[index] or
                not 2 <= len(group.spans) <= getattr(self.options, 'prefill_physical_max_chunks', 2) or
                group.scheduler_step != step or count != group.count or
                group.history_begin != history_begin or group.local_begin != local_begin or
                group.offset != history_begin-local_begin or group.prompt_tokens != prompt or
                group.capacity != config.select_budget+config.gen_budget or
                group.request_plan.block_tokens != config.block_tokens or
                group.request_plan.gen_budget != config.gen_budget):
            raise RuntimeError('Logical prefill receipt differs from the worker prefix or policy')
        if (native is None or native.disabled or native.mode != 'reuse' or
                not getattr(self.options, 'prefill_native_plan_arm', False)):
            raise RuntimeError('Logical prefill requires available native prefill score capacity')
        group = replace(group, owner_epoch=self._prefill_identity(group.request_id, frame.state))
        plans, rows, next_frame = plan_logical_scores(group, frame)
        final = plans[-1]
        width = sum((start+size+15)//16-start//16 for start, size in zip(final.starts, final.counts))
        if (len(rows) > native.max_rows or len(final.starts) > native.max_segments or
                final.span.local_end > native.context or width > native.width or
                getattr(native, 'rows', None) is not None and begin+count > native.rows.numel()):
            raise RuntimeError('Logical prefill exceeds the existing native score allocation')
        pool = getattr(self, 'download_pool', None)
        def bucket(size):
            return max(256, 1 << (max(1, size)-1).bit_length())
        # Two in-flight physical packets, each with one coordinate and score
        # download. This is an admission bound, not a new allocation or wait.
        required = 2 * (bucket(24*count) + bucket(4*len(rows)*len(final.starts)))
        if pool is None or pool.max_slots < 4 or pool.max_bytes < required:
            raise RuntimeError('Logical prefill exceeds the bounded score download budget')
        upload = getattr(self, 'metadata_upload', None)
        if upload is not None and 8*len(final.starts)+16*len(rows) > upload.capacity:
            raise RuntimeError('Logical prefill exceeds reference metadata staging capacity')
        counters = self.__dict__.setdefault('logical_prefill_stats', {})
        counters['planned_physical_groups'] = counters.get('planned_physical_groups', 0)+1
        counters['planned_logical_chunks'] = counters.get('planned_logical_chunks', 0)+len(plans)
        return group, plans, rows, next_frame

    @staticmethod
    def _logical_packet_views(packets, request_id, plan, pending):
        group = plan.get('logical_group')
        if group is None:
            packets[request_id] = pending
            return packets
        from .logical_action import LogicalActionResult, project_logical_scores
        if not isinstance(packets, LogicalActionResult):
            packets = LogicalActionResult(packets)
        parts = project_logical_scores(group, plan['logical_score_plans'], pending.masses,
            physical_rows=tuple(plan['rows']), event=pending.event)
        pending.logical_group = group
        packets[request_id] = pending
        packets.logical_groups[request_id] = group
        packets.logical_slices[request_id] = parts
        return packets

    def _record_logical_capture(self, result):
        groups = getattr(result, 'logical_groups', {})
        if groups:
            counts = self.__dict__.setdefault('logical_prefill_stats', {})
            for name, value in (('projected_physical_groups', len(groups)),
                                ('projected_logical_chunks', sum(len(g.spans) for g in groups.values())),
                                ('sampled_rows', sum(len(result[rid].rows) for rid in groups))):
                counts[name] = counts.get(name, 0)+value
        return result

    def _logical_validation_fallback(self, result):
        groups = getattr(result, 'logical_groups', {})
        if groups:
            counts = self.__dict__.setdefault('logical_prefill_stats', {})
            counts['validation_fallback_groups'] = counts.get('validation_fallback_groups', 0)+len(groups)

    @staticmethod
    def _validate_prefill_plan(plan, global_positions, local_positions, count):
        if (count != plan['count'] or
                list(global_positions) != list(range(plan['history_begin'], plan['history_begin'] + count)) or
                list(local_positions) != list(range(plan['local_begin'], plan['local_begin'] + count))):
            raise RuntimeError('Pipelined prefill coordinates differ from captured input')

    def _prepare_decode_candidates(self,batch):
        # Decode scope depends on GPU draft IDs. Collect the bounded candidate
        # rows now, and keep the original parser/publication fence after forward.
        if (any(batch.is_prefilling_np) or not hasattr(batch,'num_computed_tokens_np') or
                not hasattr(batch,'seq_lens_cpu_upper_bound') or self.native.disabled):
            return False
        offsets=getattr(getattr(self.runner,'_kvmem_window_adapter',None),'offsets',{})
        generations=getattr(getattr(self.runner,'_kvmem_window_adapter',None),'generations',{})
        cache={request_id:getattr(self,'native_cache',{}).get(request_id) for request_id in batch.req_ids}
        if any(cache[request_id] is None or cache[request_id]['generation']!=generations.get(request_id,0)
               or cache[request_id]['offset']!=offsets.get(request_id,0) for request_id in batch.req_ids):
            return False
        plans={}
        for i,request_id in enumerate(batch.req_ids):
            begin,end=map(int,batch.query_start_loc_np[i:i+2])
            h=int(batch.num_computed_tokens_np[i]);local=h-offsets.get(request_id,0)
            extent=int(batch.seq_lens_cpu_upper_bound[i])-offsets.get(request_id,0)
            if local<0 or extent<local+end-begin:
                return False
            frozen=cache[request_id]
            import numpy as np
            last=int(np.searchsorted(frozen['starts'],extent-1,side='right'))-1
            width=int(frozen['layout'][last,2]+(extent-1)//16-frozen['starts'][last]//16+1)
            plans[request_id]=dict(index=i,begin=begin,rows=list(range(end-begin)),
                local_begin=local,count=extent-local,starts=frozen['starts'],counts=frozen['counts'],
                history=frozen['history'],_layout=frozen['layout'],_width=width,_segments=last+1)
        self.native_candidates=(batch,plans)
        self.native.arm(plans)
        return True

    def _plan_owner(self, request_id, prefill, tokens, global_positions, local_positions, count):
        self.invalidate_prefill_plans((request_id,))
        if not self.runtime.drain_requests((request_id,), wait=True):
            raise RuntimeError('Action capture preceded scheduler confirmation')
        authority = self.runtime.ensure_authority(request_id)
        state = getattr(authority, 'action_history', None)
        if state is None:
            state = authority.action_history = ActionHistory(self.options, self.ids)
        history_begin, local_begin = int(global_positions[0]), int(local_positions[0])
        if prefill:
            excluded = state.prefill_replay
            replay = excluded is not None and history_begin < excluded[1] and history_begin+count > excluded[0]
            if replay and not (excluded[0] <= history_begin and history_begin+count <= excluded[1]):
                raise RuntimeError('Prefill chunk partly overlaps a public Query replay')
            rows = state.prefill.rows(history_begin, count, local_begin, replay=replay)
        else:
            self.begin_response(state, request_id)
            mask = state.parser.prepare(tokens)
            rows = [r for r in range(count) if (mask >> r) & 1]
        sources = (tuple(int(global_positions[r]) for r in rows)
                   if getattr(authority, 'block_graph', None) is not None else ())
        return dict(authority=authority, state=state, history_begin=history_begin,
                    local_begin=local_begin, rows=rows, sources=sources)

    def _window_segments(self, authority, history_begin, local_begin, count):
        store = authority.memory._store
        block_tokens = self.runtime.config.block_tokens
        if getattr(self.options, 'native_action_metadata', False):
            native = getattr(store, 'action_window_segments', None)
            if native is None:
                raise RuntimeError('native_action_metadata requires a rebuilt native extension')
            return native(history_begin, local_begin, count, block_tokens)
        return window_segments(store.working_records(), history_begin, local_begin, count, block_tokens)

    def _mtp_count(self, key):
        counts = self.__dict__.setdefault('mtp_stage_stats', {})
        counts[key] = counts.get(key, 0) + 1

    def _mtp_eligible(self, batch):
        if not getattr(self.options, 'prefill_action_before_mtp', False):
            return False
        native, plans = getattr(self, 'native', None), getattr(self, 'prefill_plans', None)
        native_plans, metadata = getattr(self, 'native_plans', None), getattr(self, 'metadata', None)
        requests = tuple(batch.req_ids)
        if (native is None or native.disabled or native.mode != 'reuse' or not requests or
                not getattr(self.options, 'prefill_publication_pipeline', False) or
                plans is None or plans[0] is not batch or native_plans is None or native_plans[0] is not batch or
                metadata is None or metadata.batch is not batch or
                tuple(metadata.indices) != tuple(range(batch.num_reqs)) or
                set(requests) != getattr(self, 'pipeline_requests', set()) or
                not all(batch.is_prefilling_np)):
            return False
        for i, request_id in enumerate(requests):
            plan = plans[1].get(request_id)
            observed = native_plans[1].get(request_id)
            if (plan is None or observed is None or not plan.get('pipeline', False) or
                    not observed.get('pipeline', False) or plan['index'] != i or
                    plan['count'] <= 0 or plan['count'] != observed['count'] or
                    plan['history_begin'] != observed['history_begin']):
                return False
            # Keep the public nonterminal and replay exclusion checks explicit,
            # even though the immutable planner has already established them.
            index = self.runner.req_states.req_id_to_index.get(request_id)
            prompt = (int(self.runner.req_states.prompt_len.np[index]) if index is not None else None)
            excluded = getattr(plan['state'], 'prefill_replay', None)
            if (prompt is None or plan['history_begin']+plan['count'] >= prompt or
                    (excluded is not None and plan['history_begin'] < excluded[1] and
                     plan['history_begin']+plan['count'] > excluded[0])):
                return False
        return True

    def stage_before_mtp(self, captured, batch):
        """Enqueue a private complete score packet before native MTP proposal.

        The original metadata/coordinate checks still run here. Only numerical
        completion waits move after proposal submission, within this forward.
        """
        if getattr(self, '_mtp_capture_frame', None) is not None:
            raise RuntimeError('Only one same-forward MTP score frame may be staged')
        if not self._mtp_eligible(batch):
            return False
        if (tuple(item.name for item in captured) != tuple(self.runner._kvmem_layers) or
                any(item.key.dtype != torch.bfloat16 for item in captured)):
            raise ValueError('Early MTP scoring requires the complete ordered BF16 target capture')
        self.runtime._raise_publication_error()
        profiler = getattr(self, 'profile', None)
        trace = profiler.begin_batch(batch) if profiler is not None else None
        frame = _MTPActionFrame(batch, tuple(captured), self.metadata,
                               self.candidate_samples, self.packets, trace)
        self._mtp_capture_frame = frame
        frame.packets = self._capture(captured, batch, trace)
        # Always record after all reductions, means and score downloads. A
        # no-dirty/reference-only frame must still fence before any KV append.
        frame.deferred = self._detach_prefill_scores(frame)
        if frame.deferred is None:
            frame.ticket = self.native.enqueue_validation()
        self._mtp_count('staged_batches')
        return True

    def _detach_prefill_scores(self, frame):
        pool = getattr(self, 'deferred_score_pool', None)
        if pool is None or not frame.packets or not self.native.can_detach_validation():
            return None
        from .deferred_action_frame import DeferredActionFrame
        from .deferred_action_scoring import FrozenLayerInput, FrozenOwnerInput
        batch = frame.batch
        prepared = self.native_plans
        if prepared is None or prepared[0] is not batch:
            return None
        groups = self._deferred_groups
        layers = []
        for _, module in self.layers:
            name = module.attn.layer_name
            key, _ = native_vllm_fp8_views(module.attn.kv_cache, module.num_kv_heads, module.head_dim)
            layers.append(FrozenLayerInput(name=name, key=key,
                cos_sin=module.rotary_emb.cos_sin_cache, key_scale=module.attn._k_scale,
                q_scale=module.attn._q_scale, scale=module.attn.impl.scale,
                q_dtype=self.prefill_q_dtypes[name], group=groups.index(self.groups[name])))
        owners = []
        for request_id, pending in frame.packets.items():
            plan = prepared[1][request_id]
            native_index = self.runner.req_states.req_id_to_index[request_id]
            owners.append(FrozenOwnerInput(request_id=request_id,
                queries=tuple(frame.captured[index].query for index, _ in self.layers),
                query_rows=tuple(plan['begin'] + row for row in plan['rows']),
                positions=tuple(plan['local_begin'] + row for row in plan['rows']),
                starts=tuple(int(x) for x in plan['starts']),
                counts=tuple(int(x) for x in plan['counts']),
                length=plan['local_begin'] + plan['count'],
                tables=tuple(kernel_table(self.runner, group, native_index)
                             for group in groups), masses=pending.masses))
        deferred = DeferredActionFrame(pool)
        slot = pool.freeze(tuple(layers), tuple(owners), packet_lease=deferred)
        if slot is None:
            return None
        deferred.slot = slot
        # Register before detach: failures after snapshot enqueue must retain
        # all source leases even if no CapturedPacket has reached the queue.
        self._deferred_frames.append(deferred)
        try:
            ticket = self.native.detach_validation(slot.host_flags, slot.event)
            deferred.seal(slot, ticket)
        except BaseException as error:
            deferred.quarantine(error)
            raise
        self._mtp_count('deferred_batches')
        return deferred

    def _collect_deferred_scores(self):
        pending = []
        for frame in getattr(self, '_deferred_frames', ()):
            if frame.apply(self.native):
                self._mtp_count('deferred_resolved_batches')
                if not frame.outcome.valid:
                    self._mtp_count('deferred_fallback_batches')
            else:
                pending.append(frame)
        self._deferred_frames = pending

    def _finish_mtp_frame(self, frame, *, status):
        self.native.disarm()
        self.native.report()
        self.native_plans = self.native_candidates = self.prefill_plans = None
        self.metadata = None
        profiler = getattr(self, 'profile', None)
        if profiler is not None:
            profiler.finish_batch(status=status)
        self._mtp_capture_frame = None

    def cancel_before_mtp(self):
        """Release private score aliases only after their own complete fence.

        The caller quarantines the whole publication runtime on failure. If a
        fence itself fails, leave the frame/ticket/leases alive and disarm
        nothing; later prepare cannot overwrite them.
        """
        frame = getattr(self, '_mtp_capture_frame', None)
        if frame is None:
            return
        if frame.deferred is not None:
            # The MTP caller has already quarantined publication. Keep this
            # private slot until process teardown; no future prepare or ACK is
            # allowed to reuse the prefix after a failed handoff.
            frame.deferred.quarantine(RuntimeError('Deferred MTP publication was cancelled'))
            self._mtp_count('quarantined_frames')
            return
        ticket = getattr(self.native, '_validation_ticket', None)
        if ticket is None and (frame.ticket is None or not frame.ticket.resolved):
            ticket = self.native.enqueue_validation()
        if ticket is not None:
            self.native.resolve_validation(ticket)
        frame.packets.clear()
        self.candidate_samples, self.packets = frame.samples, frame.packet_count
        self._finish_mtp_frame(frame, status='error')
        self._mtp_count('cancelled_batches')

    def _resolve_mtp_capture(self, batch):
        frame = self._mtp_capture_frame
        if frame.batch is not batch:
            raise RuntimeError('Early MTP score frame belongs to a different public batch')
        if frame.deferred is not None:
            from .deferred_action_frame import DeferredActionResult
            result = DeferredActionResult(frame.packets, frame.deferred)
            self._finish_mtp_frame(frame, status='submitted')
            self._mtp_count('resolved_batches')
            return result
        started = time.perf_counter()
        valid = self.native.resolve_validation(frame.ticket)
        self.capture_seconds += time.perf_counter()-started
        if not valid:
            # Drop every completed tentative host/event lease before acquiring
            # fallback slots. Keep exactly the original plans and metadata.
            self._logical_validation_fallback(frame.packets)
            frame.packets.clear()
            self.metadata = frame.metadata
            self.candidate_samples, self.packets = frame.samples, frame.packet_count
            frame.ticket = None
            frame.packets = self._capture(frame.captured, batch, frame.trace)
            frame.ticket = self.native.enqueue_validation()
            # frame_invalid remains true to force reference scoring. Completion
            # still must be checked before graph-off as well as graph-on append.
            started = time.perf_counter()
            self.native.resolve_validation(frame.ticket)
            self.capture_seconds += time.perf_counter()-started
            self._mtp_count('fallback_batches')
        result = frame.packets
        self._finish_mtp_frame(frame, status='submitted')
        self._mtp_count('resolved_batches')
        return self._record_logical_capture(result)

    def capture(self, captured, batch):
        if getattr(self, '_mtp_capture_frame', None) is not None:
            return self._resolve_mtp_capture(batch)
        profiler = getattr(self, 'profile', None)
        trace = profiler.begin_batch(batch) if profiler is not None else None
        status = 'submitted'
        try:
            metadata = getattr(self, 'metadata', None)
            samples, packets = self.candidate_samples, self.packets
            result = self._capture(captured, batch, trace)
            native = getattr(self, 'native', None)
            if native is not None:
                started = time.perf_counter()
                valid = native.validate()
                self.capture_seconds += time.perf_counter() - started
                if not valid:
                    # No packet has escaped this method. Discard tentative
                    # downloads, keep the exact pre-forward metadata/plans and
                    # recompute this batch through the reference path. Scope
                    # preparation and acceptance must each happen only once.
                    self.metadata = metadata
                    self.candidate_samples, self.packets = samples, packets
                    self._logical_validation_fallback(result)
                    if hasattr(result, 'logical_slices'):
                        result.clear()
                    result = self._capture(captured, batch, trace)
            return self._record_logical_capture(result)
        except BaseException:
            status = 'error'
            raise
        finally:
            if profiler is not None:
                profiler.finish_batch(status=status)
            self.prefill_plans = None
            if getattr(self, 'native', None) is not None:
                self.native.disarm()
                self.native.report()
                self.native_plans = None
                self.native_candidates = None

    def _capture(self, captured, batch, trace=None):
        indices=[i for i in range(batch.num_reqs) if not batch.is_prefilling_np[i] or self.options.prefill_attention]
        if not indices:return {}
        started=time.perf_counter()
        metadata = getattr(self, 'metadata', None)
        self.metadata = None
        if metadata is not None and metadata.batch is batch:
            self.metadata_early_batches += 1
        else:
            # The capture runtime is constructed after the first forward.
            # Standalone diagnostic callers also use this exact fallback.
            metadata = action_metadata(batch, captured[0].positions, self.options.prefill_attention, trace,
                pool=getattr(self,'download_pool',None))
            self.metadata_fallback_batches = getattr(self, 'metadata_fallback_batches', 0) + 1
        waited = time.perf_counter()
        with trace.stage('metadata.existing_fence') if trace else nullcontext():
            if metadata.event is not None:
                metadata.event.synchronize()
        self.metadata_wait_seconds = getattr(self, 'metadata_wait_seconds', 0.0) + time.perf_counter() - waited
        slices = metadata.slices
        with trace.stage('metadata.tolist') if trace else nullcontext():
            data = metadata.host.tolist()
        tokens,global_positions,local_positions=data
        packets={};cursor=0
        candidates=getattr(self,'native_candidates',None)
        if candidates is not None and candidates[0] is batch:
            plans={}
            for i,(begin,end) in zip(indices,slices):
                count=end-begin;request_id=batch.req_ids[i]
                plan=self._plan_owner(request_id,bool(batch.is_prefilling_np[i]),
                    tokens[cursor:cursor+count],global_positions[cursor:cursor+count],
                    local_positions[cursor:cursor+count],count)
                cursor+=count
                plan.update(index=i,begin=begin,count=count)
                if plan['rows']:
                    starts,counts,history=self._window_segments(plan['authority'],plan['history_begin'],
                        plan['local_begin'],count)
                    from .native_attention_planning import canonical_segments
                    canonical=canonical_segments(starts,counts,history,plan['local_begin']+count,
                        self.runtime.config.block_tokens)
                    plan.update(starts=canonical[:,0],counts=canonical[:,1],history=canonical[:,2].tolist())
                plans[request_id]=plan
            self.native.resolve(plans)
            self.native_plans=(batch,plans)
            self.native_candidates=None
            cursor=0
        from .action_scoring import page_attention
        for i,(begin,end) in zip(indices,slices):
            request_id=batch.req_ids[i]
            prefill=bool(batch.is_prefilling_np[i])
            owner = trace.child(request_id=request_id, kind='prefill' if prefill else 'output') if trace else None
            # Parser state and dense page identities must both reflect this
            # request's preceding accepted prefix. A shared DMA packet can
            # still have an unrelated owner publishing CPU history. Admission
            # holds large unfinished owners out of this public batch; tiny
            # decode dependencies may finish through this request-local fence.
            count=end-begin
            prepared = getattr(self, 'native_plans', None)
            if prepared is not None and prepared[0] is batch:
                plan = prepared[1][request_id]
            else:
                prepared = getattr(self, 'prefill_plans', None)
                plan = prepared[1].get(request_id) if prepared is not None and prepared[0] is batch else None
                if plan is None:
                    if request_id in getattr(self.runner, '_kvmem_logical_prefill_groups', {}):
                        raise RuntimeError('Logical prefill capture is missing its pre-forward worker plan')
                    with owner.stage('publication.owner_drain') if owner else nullcontext():
                        plan = self._plan_owner(request_id, prefill, tokens[cursor:cursor+count],
                            global_positions[cursor:cursor+count], local_positions[cursor:cursor+count], count)
            receipt = getattr(self.runner, '_kvmem_logical_prefill_groups', {}).get(request_id)
            if receipt is not None:
                logical = plan.get('logical_group')
                if (logical is None or logical.scheduler_step != receipt.scheduler_step or
                        logical.request_id != receipt.request_id or logical.count != count or
                        logical.history_begin != receipt.history_begin):
                    raise RuntimeError('Logical prefill capture is missing its pre-forward worker plan')
            if plan.get('pipeline', False):
                self._validate_prefill_plan(plan, global_positions[cursor:cursor+count],
                                            local_positions[cursor:cursor+count], count)
            authority, state = plan['authority'], plan['state']
            history_begin, local_begin = plan['history_begin'], plan['local_begin']
            rows, sources = plan['rows'], plan['sources']
            cursor+=count
            if getattr(self,'native',None) is not None and not self.native.disabled:
                cache=self.__dict__.setdefault('native_cache',{})
                generations=getattr(getattr(self.runner,'_kvmem_window_adapter',None),'generations',{})
                generation=generations.get(request_id,0)
                offset=history_begin-local_begin
                previous=cache.get(request_id)
                if previous is None or previous['generation']!=generation or previous['offset']!=offset:
                    if 'starts' not in plan:
                        starts,counts,history=self._window_segments(authority,history_begin,local_begin,count)
                        plan.update(starts=starts,counts=counts,history=history)
                    if len(plan['starts'])<=self.native.max_segments and local_begin+count<=self.native.context:
                        from .native_attention_planning import make_decode_frame
                        cache[request_id]=dict(generation=generation,offset=offset,
                            **make_decode_frame(plan['starts'],plan['counts'],plan['history'],history_begin,
                                local_begin,count,self.native.context,self.runtime.config.block_tokens))
                    else:
                        cache.pop(request_id,None)
            if not rows:
                if plan.get('logical_group') is not None:
                    host = torch.empty((0, len(plan['history'])), dtype=torch.float32)
                    pending = ActionPacket(state, [], plan['history'], host, None,
                        (history_begin+count, count, plan['starts'], local_begin),
                        sources, history_begin, count)
                    packets = self._logical_packet_views(packets, request_id, plan, pending)
                elif not prefill:
                    packets[request_id]=ActionPacket(state,[],[],None,None,
                        source_positions=sources,history_begin=history_begin,candidate_count=count)
                continue
            with owner.stage('metadata.window_segments', active_tokens=local_begin+count,
                             sample_rows=len(rows)) if owner else nullcontext():
                if 'starts' in plan:
                    starts,counts,history=plan['starts'],plan['counts'],plan['history']
                else:
                    starts,counts,history=self._window_segments(authority,history_begin,local_begin,count)
            device=batch.input_ids.device
            if owner:
                owner = owner.child(active_tokens=local_begin+count, sample_rows=len(rows), segments=len(starts))
            native = getattr(self, 'native', None)
            native_masses = [native.score(module.attn.layer_name,request_id,prefill,defer_validation=True)
                             if native else None for _,module in self.layers]
            needs_reference = native is None or native.mode == 'verify' or any(mass is None for mass in native_masses)
            upload = getattr(self, 'metadata_upload', None)
            staging = (upload.lease(starts,counts,rows,begin,local_begin,device,profile=owner)
                       if upload is not None and needs_reference else nullcontext(None))
            with staging as lease:
                if lease is None and needs_reference:
                    with owner.stage('metadata.starts_h2d', gpu=True, transfer_bytes=4*len(starts),
                                     direction='H2D', shape=[len(starts)], dtype='int32') if owner else nullcontext():
                        starts_gpu=torch.tensor(starts,device=device,dtype=torch.int32)
                    with owner.stage('metadata.counts_h2d', gpu=True, transfer_bytes=4*len(counts),
                                     direction='H2D', shape=[len(counts)], dtype='int32') if owner else nullcontext():
                        counts_gpu=torch.tensor(counts,device=device,dtype=torch.int32)
                    with owner.stage('metadata.rows_h2d', gpu=True, transfer_bytes=8*len(rows),
                                     direction='H2D', shape=[len(rows)], dtype='int64') if owner else nullcontext():
                        row_ids=torch.tensor([begin+r for r in rows],device=device,dtype=torch.long)
                    with owner.stage('metadata.positions_h2d', gpu=True, transfer_bytes=8*len(rows),
                                     direction='H2D', shape=[len(rows)], dtype='int64') if owner else nullcontext():
                        positions=torch.tensor([local_begin+r for r in rows],device=device,dtype=torch.int64)
                elif lease is not None:
                    starts_gpu,counts_gpu,row_ids,positions=lease.views
                native_index=self.runner.req_states.req_id_to_index[request_id]
                total=None
                for (index,module),mass in zip(self.layers,native_masses):
                    layer = owner.child(layer=module.attn.layer_name) if owner else None
                    with layer.stage('score.layer_total', gpu=True) if layer else nullcontext():
                        if mass is None or native.mode == 'verify':
                            group=self.groups[module.attn.layer_name]
                            table=kernel_table(self.runner, group, native_index)
                            key,_=native_vllm_fp8_views(module.attn.kv_cache,module.num_kv_heads,module.head_dim)
                            with layer.stage('score.q_gather', gpu=True) if layer else nullcontext():
                                query = captured[index].query.index_select(0,row_ids)
                            if plan.get('logical_group') is not None:
                                from .logical_action import reference_logical_scores
                                reference=reference_logical_scores(plan['logical_group'],
                                    plan['logical_score_plans'],query,positions,key,table,
                                    module.rotary_emb.cos_sin_cache,module.attn._k_scale,starts_gpu,counts_gpu,
                                    scale=module.attn.impl.scale,
                                    q_dtype=self.prefill_q_dtypes[module.attn.layer_name],q_scale=module.attn._q_scale,
                                    block_tokens=self.runtime.config.block_tokens,profile=layer,
                                    fused=getattr(self.options,'action_fused_scoring',False))
                            else:
                                reference=page_attention(query,positions,key,table,
                                    module.rotary_emb.cos_sin_cache,module.attn._k_scale,starts_gpu,counts_gpu,
                                    local_begin+count,scale=module.attn.impl.scale,
                                    q_dtype=(self.prefill_q_dtypes if prefill else self.q_dtypes)[module.attn.layer_name],q_scale=module.attn._q_scale,
                                    block_tokens=self.runtime.config.block_tokens, **({'profile': layer} if layer else {}),
                                    **(dict(fused=True,complete_partition=True)
                                       if getattr(self.options,'action_fused_scoring',False) else {}))
                            counts_by_backend=self.__dict__.setdefault('scoring_backend_counts',{})
                            backend=getattr(reference,'_action_scoring_backend','reference')
                            counts_by_backend[backend]=counts_by_backend.get(backend,0)+1
                            if mass is not None:
                                native.compare(mass,reference)
                            mass=reference
                    with layer.stage('score.layer_add', gpu=True) if layer else nullcontext():
                        total=mass if total is None else total+mass
                with owner.stage('score.layer_mean', gpu=True) if owner else nullcontext():
                    total=total/len(self.layers)
                pool=getattr(self,'download_pool',None)
                if pool is not None:
                    host,event=pool.copy(total)
                else:
                    with owner.stage('mass.pinned_allocate', shape=list(total.shape),
                                 allocation_bytes=total.numel()*total.element_size()) if owner else nullcontext():
                        host=torch.empty(total.shape,dtype=total.dtype,pin_memory=True)
                    with owner.stage('mass.d2h', gpu=True, transfer_bytes=total.numel()*total.element_size(),
                                 direction='D2H') if owner else nullcontext():
                        host.copy_(total,non_blocking=True)
                    event=torch.cuda.Event();event.record()
                if lease is not None:
                    # Covers every layer and recursive scoring chunk, not just
                    # the H2D. The runtime owns staging even if publication fails.
                    lease.complete(event)
                metadata=(history_begin+count,count,starts,local_begin) if prefill else None
                pending=ActionPacket(state,rows,history,host,event,metadata,
                    sources,history_begin,count)
                packets=self._logical_packet_views(packets,request_id,plan,pending)
                self.candidate_samples+=len(rows)
        self.packets+=len(packets)
        self.capture_seconds+=time.perf_counter()-started
        return packets

    def _wait_packet(self, pending):
        if pending is None or pending.event is None:
            return
        waited = time.perf_counter()
        pending.event.synchronize()
        self.accept_event_wait_seconds = getattr(self, 'accept_event_wait_seconds', 0.) + time.perf_counter()-waited

    def _prefill_accept(self, history, pending, accepted):
        end,count,starts,local = pending.prefill
        started = time.perf_counter()
        try:
            return history.accept(end,count,accepted,pending.rows,starts,pending.history,local,pending.masses)
        finally:
            self.prefill_commit_host_seconds = getattr(self, 'prefill_commit_host_seconds', 0.) + time.perf_counter()-started
            self.prefill_commit_calls = getattr(self, 'prefill_commit_calls', 0) + 1
            backend = getattr(history, 'last_commit_backend', None)
            if backend is not None:
                key = 'prefill_commit_' + backend + '_calls'
                setattr(self, key, getattr(self, key, 0)+1)

    def _report_prefill(self, request_id, pending):
        history = pending.state.prefill
        with (self.runtime.output/'prefill_attention.jsonl').open('a') as stream:
            stream.write(json.dumps({'request_id':request_id,'end':pending.prefill[0],
                'query_samples':len(pending.rows),'samples':history.samples,
                'chunks':history.chunks,'pages':len(history.values)})+'\n')

    def prepare_accept(self, packet, request_id, accepted, *, graph, history_begin):
        """Reserve graph state before modifying any accepted feedback.

        Empty and partial-prefill callbacks still prepare an empty graph update,
        so their *accepted* logical boundary can finalize a partial source.
        """
        pending = packet.action_inputs.get(request_id)
        if pending is not None and getattr(pending, 'logical_group', None) is not None:
            raise RuntimeError('Physical prefill scores require ordered logical publication')
        self._wait_packet(pending)
        output = prefill = None
        positions, masses, targets = (), (), ()
        kind = 'output'
        dense_output = None
        if pending is not None:
            if pending.history_begin is not None and pending.history_begin != history_begin:
                raise ValueError('Graph attention packet has stale source coordinates')
            if pending.prefill is not None:
                from copy import copy
                kind = 'prefill'
                original = pending.state.prefill
                staged = copy(original)
                committed = self._prefill_accept(staged, pending, accepted)
                prefill = original, staged, committed
                # Preserve the existing full-chunk rule. An accepted partial
                # prefill advances graph progress with no observed rows.
                end,count,_,_ = pending.prefill
                excluded = getattr(pending.state, 'prefill_replay', None)
                replay = excluded is not None and history_begin < excluded[1] and end > excluded[0]
                if accepted == count and not replay and pending.rows:
                    if len(pending.source_positions) != len(pending.rows):
                        raise ValueError('Graph prefill packet is missing source positions')
                    if any(position != history_begin+row for row,position in
                           zip(pending.rows,pending.source_positions)):
                        raise ValueError('Graph prefill source differs from consumed Q position')
                    positions = pending.source_positions
                    # The captured CPU tensor already owns this storage. Avoid
                    # expanding a large prefill matrix into Python floats
                    # outside graph staging admission.
                    masses = pending.masses.numpy() if hasattr(pending.masses, 'numpy') else pending.masses
                    targets = pending.history
            else:
                parser = pending.state.parser
                if not isinstance(parser, _AcceptanceParser):
                    raise RuntimeError('Graph output requires transactional action parser')
                mask = parser.preview(accepted)
                values = pending.masses.numpy() if hasattr(pending.masses, 'numpy') else pending.masses
                values = [] if values is None else values
                if len(pending.rows) != len(values) or len(pending.source_positions) != len(pending.rows):
                    raise ValueError('Graph output packet is missing attention rows or source positions')
                if any(not isinstance(row,int) or not 0 <= row < parser.count for row in pending.rows):
                    raise ValueError('Graph attention candidate row is out of bounds')
                if mask & ~sum(1 << row for row in set(pending.rows)):
                    raise ValueError('Graph output packet omitted an accepted sampled row')
                native_decode = getattr(getattr(self,'options',pending.state.options),'action_native_decode_commit',False)
                if native_decode:
                    selected = tuple(i for i,row in enumerate(pending.rows) if (mask >> row) & 1)
                    if selected:
                        import numpy as np
                        values = np.asarray(values).reshape(len(pending.rows),len(pending.history))
                        positions = tuple(pending.source_positions[i] for i in selected)
                        if any(pending.source_positions[i] != history_begin+pending.rows[i] for i in selected):
                            raise ValueError('Graph output source differs from consumed Q position')
                        dense_output = selected, (pending.state.mass,pending.state.samples,
                            getattr(pending.state.options,'action_commit_max_bytes',16777216)), mask
                        masses, targets = values, pending.history
                    else:
                        # Most decode frames have no accepted sampled row.
                        # Advancing their graph/parser boundary needs no copy
                        # or validation of the immutable published score map.
                        output=mask,pending.state.mass,pending.state.samples
                updated, samples = (None,None) if native_decode else (dict(pending.state.mass),pending.state.samples)
                kept_positions, kept_masses = [], []
                for row, position, row_mass in (() if native_decode else zip(pending.rows,pending.source_positions,values)):
                    if not (mask >> row) & 1:
                        continue
                    if (len(row_mass) != len(pending.history) or
                            any(not math.isfinite(v) or v < 0 for v in row_mass) or
                            abs(sum(float(v) for v in row_mass)-1) > .005):
                        raise RuntimeError('Action attention mass is invalid or does not sum to one')
                    if position != history_begin + row:
                        raise ValueError('Graph output source differs from consumed Q position')
                    for key,value in zip(pending.history,row_mass):
                        # Tensor.tolist(), used by the graph-off path, creates
                        # Python doubles. Preserve that exact legacy arithmetic
                        # in collect mode instead of NumPy float32 promotion.
                        value = float(value)
                        if value:
                            updated[key] = updated.get(key,0.) + value
                    samples += 1
                    kept_positions.append(position)
                    kept_masses.append(row_mass)
                if not native_decode:
                    output = mask, updated, samples
                    positions, masses, targets = kept_positions, kept_masses, pending.history
        started = time.perf_counter()
        try:
            graph_begin = min(history_begin, graph.accepted_through) if not accepted else history_begin
            update = graph.prepare_accept(graph_begin, graph_begin+accepted,
                source_positions=positions,target_starts=targets,mass_rows=masses,kind=kind,
                **({} if dense_output is None else dict(row_indices=dense_output[0],output_state=dense_output[1])))
            if dense_output is not None:
                mass,samples = getattr(update,'output_state',dense_output[1][:2])
                output = dense_output[2],mass,samples
        finally:
            self.graph_prepare_seconds = getattr(self,'graph_prepare_seconds',0.) + time.perf_counter()-started
            self.graph_prepare_calls = getattr(self,'graph_prepare_calls',0) + 1
        return _PreparedActionAcceptance(self,packet,request_id,pending,update,
            accepted=accepted,output=output,prefill=prefill)

    def accept(self, packet, request_id, accepted):
        pending = packet.action_inputs.get(request_id)
        if pending is not None and getattr(pending, 'logical_group', None) is not None:
            raise RuntimeError('Physical prefill scores require ordered logical publication')
        pending=packet.action_inputs.pop(request_id,None)
        if pending is None:return
        self._wait_packet(pending)
        if pending.prefill is not None:
            history=pending.state.prefill
            committed = self._prefill_accept(history,pending,accepted)
            if committed:
                self._report_prefill(request_id,pending)
            return
        pending.state.accept(accepted,pending.rows,pending.history,
            pending.masses if getattr(pending.state.options,'action_native_decode_commit',False) else
            pending.masses.tolist() if pending.masses is not None else [])
        backend = 'native' if getattr(pending.state.options,'action_native_decode_commit',False) else 'python'
        key = 'decode_commit_'+backend+'_calls'
        setattr(self,key,getattr(self,key,0)+1)

    def finish(self, request_id, authority):
        self._collect_deferred_scores()
        getattr(self,'native_cache',{}).pop(request_id,None)
        self.invalidate_prefill_plans((request_id,))
        native = getattr(self, 'native', None)
        if native is not None:
            native.report(force=True)
        profiler = getattr(self, 'profile', None)
        if profiler is not None:
            # Publication already used its original accepted-score fence.
            # Polling here only recovers completed diagnostic events.
            profiler.flush()
        state=getattr(authority,'action_history',None)
        if state is None:return None
        report=state.finish(authority)
        if state.prefill is not None:
            report.update(prefill_samples=state.prefill.samples,prefill_chunks=state.prefill.chunks,
                          prefill_end=state.prefill.end,prefill_pages=len(state.prefill.values))
        report.update(request_id=request_id,candidate_samples_total=self.candidate_samples,
                      capture_host_seconds_total=self.capture_seconds,
                      metadata_early_batches=getattr(self,'metadata_early_batches',0),
                      metadata_fallback_batches=getattr(self,'metadata_fallback_batches',0),
                      metadata_wait_seconds=getattr(self,'metadata_wait_seconds',0.0))
        if torch.cuda.is_initialized():
            report.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report.update(prefill_commit_host_seconds=getattr(self,'prefill_commit_host_seconds',0.),
                      prefill_commit_calls=getattr(self,'prefill_commit_calls',0),
                      prefill_commit_native_calls=getattr(self,'prefill_commit_native_calls',0),
                      prefill_commit_python_calls=getattr(self,'prefill_commit_python_calls',0),
                      accept_event_wait_seconds=getattr(self,'accept_event_wait_seconds',0.))
        report.update(decode_commit_native_calls=getattr(self,'decode_commit_native_calls',0),
                      decode_commit_python_calls=getattr(self,'decode_commit_python_calls',0))
        if getattr(self,'download_pool',None) is not None:
            report['download_pool_total']=self.download_pool.snapshot()
        report['scoring_backend_counts']=dict(getattr(self,'scoring_backend_counts',{}))
        if getattr(self, 'deferred_score_pool', None) is not None:
            report['deferred_score_pool'] = self.deferred_score_pool.snapshot()
        graph = getattr(authority,'block_graph',None)
        if graph is not None:
            rows = graph.rows
            partial = graph.partial
            report['block_graph'] = dict(graph.stats(),
                prefill_samples=sum(row.prefill_samples for row in rows) +
                    (partial.prefill_samples if partial is not None else 0),
                output_samples=sum(row.output_samples for row in rows) +
                    (partial.output_samples if partial is not None else 0))
            report.update(graph_prepare_seconds=getattr(self,'graph_prepare_seconds',0.),
                graph_prepare_calls=getattr(self,'graph_prepare_calls',0),
                graph_commit_seconds=getattr(self,'graph_commit_seconds',0.),
                graph_commit_calls=getattr(self,'graph_commit_calls',0))
        if getattr(self, 'metadata_upload', None) is not None:
            # Diagnostic snapshot only. Publication/cancellation never retires
            # staging: the engine thread polls its consumer completion events.
            report['metadata_upload_total'] = self.metadata_upload.snapshot()
        with (self.runtime.output/'action_attention.jsonl').open('a') as stream:
            stream.write(json.dumps(report)+'\n')
        return report
