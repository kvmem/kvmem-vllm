"""Worker-side prompt checkpoint capture, import and fenced retirement.

Import runs after native request/table initialization and block zeroing, but
before attention metadata is built. Native sampling and draft proposal retain
their normal per-request lifecycle. CPU checkpoint ownership is bounded separately
from the shared authority arena; the runner must reserve room for idle authorities.
"""
import hashlib
import os
import time
from collections import deque
from itertools import islice

import torch

from .authority import _Ready
from .prompt_checkpoints import PromptCheckpointPool
from .session_control import SessionNotice
from .session_state import PromptStateAdapter


def verify_preserved_authority(authority, timeout=180):
    """Verify all still-retained sealed records without stopping live GC.

    Pin the complete foreground snapshot before reading. GC tombstones can race
    initial pin acquisition, so discard that attempt and retry after draining.
    Holding those leases makes checksum iteration immutable without changing the
    original background policy or counting collected records as corrupt misses.
    """
    deadline = time.monotonic() + timeout
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Preserved authority verification did not drain')
        return value
    while not authority.flush():
        for job, _ in authority.writes:
            job.result(timeout=remaining())
        if not authority.archive.wait_for_capacity(remaining()):
            raise TimeoutError('Preserved authority publication timed out')
    for job, _ in authority.writes:
        job.result(timeout=remaining())
    if getattr(authority.gc, 'native_history', False):
        authority.poll()
        records = tuple(authority.records.values())
        if sum(record.archived for record in records) + authority.gc_reclaimed_records != authority.history_end // authority.shape[1]:
            raise RuntimeError('Preserved authority has unpublished sealed records')
        verified = 0
        for record in records:
            if not record.archived:
                continue
            remaining()
            admitted = authority.gc.scope.admit_read(record.handle)
            if admitted is None:
                continue  # Irreversible retirement won before this new read.
            lease, source = admitted
            if source is None or hashlib.sha256(source).hexdigest() != record.digest:
                raise RuntimeError('Preserved native authority checksum differs from accepted capture')
            verified += 1
        return verified
    while True:
        remaining()
        authority.poll()
        records = list(authority.records.values())
        leases = [authority.memory.repository.pin(record.handle) for record in records]
        if all(leases):
            break
        for lease in leases:
            lease.reset()
    try:
        archived = sum(record.archived for record in records)
        if archived + authority.gc_reclaimed_records != authority.history_end // authority.shape[1]:
            raise RuntimeError('Preserved authority has unpublished sealed records')
        verified = 0
        for record in records:
            if not record.archived:
                continue
            while True:
                try:
                    raw, values = authority.read_record(record.handle)
                    break
                except BlockingIOError:
                    if not authority.archive.wait_for_capacity(remaining()):
                        raise TimeoutError('Preserved authority checksum read timed out')
            digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes() + values.numpy().tobytes()).hexdigest()
            if digest != record.digest:
                raise RuntimeError('Preserved authority checksum differs from accepted capture')
            verified += 1
        return verified
    finally:
        for lease in leases:
            lease.reset()


class SessionWorker:
    def __init__(self, runner, options):
        self.runner, self.options = runner, options
        self.pool = PromptCheckpointPool(options.session_capacity, options.session_checkpoint_bytes,
            retain_prompt=getattr(options,'session_checkpoint_policy','sealed')=='pm',
            native_restore_frame=getattr(options, 'native_restore_frame', False),
            native_token_identity=getattr(options, 'restore_handoff_fastpath', False),
            native_checkpoint_capture=getattr(options, 'native_checkpoint_capture', False))
        self.bindings = {}
        self.identities = {}
        self.imports = set()
        self.captures = {}
        self.retiring = {}
        self.evictions = {}
        self.completed = []
        self.model_adapter = None
        self.terminal_captures = {}
        self.restore_oracles = {}
        self.checkpoint_deferrals = 0
        self.restored_tokens = 0
        self.release_requests = set()
        self._history_idle = deque()
        self._history_idle_queued = set()
        self._history_capacity_sent = {}

    @property
    def runtime(self):
        return self.runner._kvmem_capture_runtime

    def receive(self, commands):
        self.release_requests.update(getattr(commands, 'release_requests', ()))
        self._receive_history_capacity(getattr(commands, 'history_capacity', ()))
        for binding in commands.bindings:
            if binding.request_id in self.bindings:
                raise ValueError('Duplicate worker session binding')
            old = self.identities.get(binding.namespace)
            if old is not None and old != binding.generation:
                raise ValueError('Session namespace reused before worker reclamation')
            if bool(binding.history) != (binding.namespace in self.pool.entries):
                raise ValueError('Worker checkpoint and scheduler import disagree')
            self.identities[binding.namespace] = binding.generation
            self.bindings[binding.request_id] = binding
            if binding.history:
                self.imports.add(binding.request_id)
        for capture in commands.captures:
            binding = self.bindings.get(capture.request_id)
            if binding is None or binding.generation != capture.generation or capture.request_id in self.captures:
                raise ValueError('Stale or duplicate worker checkpoint command')
            self.captures[capture.request_id] = capture
        for capture in getattr(commands, 'terminals', ()):
            binding = self.bindings.get(capture.request_id)
            if binding is None or binding.generation != capture.generation or capture.request_id in self.terminal_captures:
                raise ValueError('Stale or duplicate terminal checkpoint command')
            self.terminal_captures[capture.request_id] = capture

    def before_attention(self, batch):
        if getattr(self.options, 'async_restore', False) and self.imports.intersection(batch.req_ids):
            raise RuntimeError('Unready checkpoint restore entered a public model batch')
        selected = self.imports.intersection(batch.req_ids) | (self.captures.keys() & set(batch.req_ids))
        if not selected:
            return
        if self.runtime is None:
            raise RuntimeError('Prompt checkpoint has no accepted authority runtime')
        if not self.runtime.drain_requests(selected, wait=True):
            raise RuntimeError('Prompt checkpoint preceded scheduler publication confirmation')
        if self.model_adapter is None:
            self.model_adapter = PromptStateAdapter(self.runner)
        for request_id in batch.req_ids:
            if request_id in self.imports:
                self._restore(request_id)
                self.imports.remove(request_id)
            command = self.captures.pop(request_id, None)
            if command is None:
                continue
            binding = self.bindings[request_id]
            authority = self.runtime.authorities[request_id]
            if authority.history_end != command.position:
                raise ValueError('Prompt capture differs from accepted archive history')
            try:
                self.pool.capture(binding.namespace, request_id, authority,
                    command.tokens or binding.tokens, self.model_adapter,
                    replace=getattr(self.options, 'session_capture_decode', False) or self.pool.retain_prompt)
            except BufferError:
                # Cache admission failure preserves ordinary native execution.
                self.checkpoint_deferrals += 1
                if getattr(self.options, 'session_capture_decode', False) or self.pool.retain_prompt:
                    self.completed.append(SessionNotice('capture_deferred', binding.namespace,
                        binding.generation, request_id, command.position))
            else:
                if getattr(self.options, 'session_capture_decode', False) or self.pool.retain_prompt:
                    self.completed.append(SessionNotice('captured', binding.namespace,
                        binding.generation, request_id, command.position))

    def _target_bytes(self, request_id, tokens):
        """Opt-in real-model restoration oracle; unused in ordinary serving."""
        from .window_assembly import NativeWindowAssembler
        assembler = NativeWindowAssembler.from_runner(self.runner, request_id, self.runtime.names,
                                                       self.options.memory.block_tokens)
        try:
            positions = torch.arange(tokens, device=self.runner.device)
            rows = []
            for layer in assembler.layers:
                block = layer.key.shape[1]
                pages = layer.kernel_page_table[positions // block].long()
                offsets = positions % block
                rows.extend((layer.key[pages, offsets].view(torch.uint8).flatten().cpu(),
                             layer.value[pages, offsets].view(torch.uint8).flatten().cpu()))
            return torch.cat(rows)
        finally:
            assembler.close()

    def selection_committed(self, request_id, history):
        binding = self.bindings.get(request_id)
        checkpoint = self.pool.entries.get(binding.namespace) if binding is not None else None
        if checkpoint is not None and checkpoint.position == history:
            if checkpoint.request_id != request_id:
                raise ValueError('Selected frame belongs to another checkpoint request')
            if getattr(self.options,'session_freeze_prompt',False):
                # Independent questions may share an exact input prefix. Keep
                # its pre-selection frame/model/attention state unchanged;
                # the previous answer's selection must not influence the next Q.
                return
            self.pool.update_live_window(checkpoint)

    def _restore(self, request_id):
        from .cache_groups import full_attention_groups
        from .window_assembly import NativeWindowAssembler
        from .window_batch import WindowBatchAdapter
        binding = self.bindings[request_id]
        checkpoint = self.pool.claim(binding.namespace, request_id, binding.tokens,position=binding.history)
        if (checkpoint is None or checkpoint.position != binding.history or
                (checkpoint.window.window_tokens != binding.window and not binding.pressure_restore)):
            raise ValueError('Worker cannot restore the advertised exact prompt prefix')
        if request_id in self.runtime.authorities:
            raise ValueError('Native import already owns a different authority')
        authority = checkpoint.authority
        cfg = self.options.memory
        assembler = NativeWindowAssembler.from_runner(self.runner, request_id, self.runtime.names,
            cfg.block_tokens, packed=self.options.packed_assembly,
            direct_pinned_reads=self.options.direct_pinned_reads)
        try:
            if binding.pressure_restore:
                block = cfg.block_tokens
                plan = self.pool.prepare_claim_window(checkpoint, pressure=True,
                    pressure_budget=(binding.window + block - 1) // block * block,
                    mandatory_spans=binding.mandatory_spans, metadata_binding=binding)
                if plan.total_window_tokens != binding.window:
                    raise ValueError('Pressure restore differs from its reserved window')
            else:
                plan = self.pool.prepare_claim_window(checkpoint, metadata_binding=binding)
            assembly = assembler.assemble(authority, plan)
            assembly.event.synchronize()
            authority.memory.attach_completion(assembly)
            assert authority.memory.commit_selection()
            event = self.model_adapter.restore(checkpoint.model, request_id)
            if event is not None:
                event.synchronize()
            self._verify_restored(request_id, checkpoint)
            restored = self.pool.finish_claim(checkpoint, event or _Ready())
            assert restored is authority
        except BaseException:
            # Import failure is fatal to this worker. Retain all state/page
            # owners through completed device work before normal shutdown cleanup.
            if self.runner.device.type == 'cuda':
                torch.cuda.synchronize(self.runner.device)
            self.pool.evict(checkpoint, _Ready())
            raise
        finally:
            assembler.close()
        self.publish_restore(request_id, authority)

    def _rebuilt_target_bytes(self, request_id, authority):
        """Verification-only CPU oracle from immutable FP8 source records.

        Do not compare a cold FP8-raw rebuild to the original unquantized
        forward's rotated K. The source quantization is intentionally lossy.
        This oracle uses independent Torch math, never the assembly kernels.
        """
        from .window_assembly import NativeWindowAssembler
        from .codec import neox_rope
        authority.publish_sealed(self.runtime._wait_archive_progress)
        records = tuple(authority.memory._store.working_records())
        assembler = NativeWindowAssembler.from_runner(self.runner, request_id, self.runtime.names,
                                                       self.options.memory.block_tokens)
        try:
            keys, values = [[] for _ in assembler.layers], [[] for _ in assembler.layers]
            trigs = [layer.cos_sin.cpu() for layer in assembler.layers]
            scales = [layer.key_scale.cpu() for layer in assembler.layers]
            for start in range(0, len(records), 32):
                chunk = records[start:start+32]
                handles = [authority.records[(slot, generation)].handle
                           for slot, generation, *_ in chunk]
                payloads = authority.read_records(handles)
                for (_, _, _, count, position), (raw, value) in zip(chunk, payloads):
                    positions = torch.arange(position, position+count)
                    for index, layer in enumerate(assembler.layers):
                        trig = trigs[index][positions]
                        half = trig.shape[-1]//2
                        rotated = neox_rope(raw[index], trig[:, :half], trig[:, half:])
                        key = (rotated.float()/scales[index]).clamp(-448,448).to(torch.float8_e4m3fn)
                        keys[index].append(key.view(torch.uint8).flatten())
                        values[index].append(value[index].flatten())
            return torch.cat([tensor for k,v in zip(keys,values)
                              for tensor in (torch.cat(k),torch.cat(v))])
        finally:
            assembler.close()

    def _verify_restored(self, request_id, checkpoint, resident=False):
        if not self.pool.model_ready(checkpoint):
            raise RuntimeError('Restore verification requires a completed model checkpoint')
        if os.getenv('KVMEM_VERIFY_SESSION_RESTORE_DIGEST') == '1':
            from .restore_digest import verify
            return verify(self, request_id, checkpoint, resident)
        binding = self.bindings[request_id]
        working_tokens = binding.window if binding.pressure_restore else checkpoint.window.window_tokens
        # Verification-only: retain one bounded oracle per namespace until a
        # new terminal checkpoint replaces it or the session is evicted. A
        # cancelled successor must prove its return to this same M checkpoint.
        oracle = getattr(self, 'restore_oracles', {}).get(binding.namespace)
        if isinstance(oracle,tuple):
            oracle = oracle[1] if oracle[0]==checkpoint.position else None
        if binding.pressure_restore:
            # The intended selected page set changed; an old-view byte
            # oracle is not a valid comparison for a pressure refresh.
            oracle = None
        oracle_kind = 'native_terminal_window'
        if (os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1' and not resident and
                checkpoint.authority.raw_dtype == torch.float8_e4m3fn):
            oracle = self._rebuilt_target_bytes(request_id, checkpoint.authority)
            oracle_kind = 'fp8_immutable_rebuild'
        if os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1':
            recurrent = self.model_adapter._recurrent_views(request_id)
            if any(not torch.equal(t.cpu(), checkpoint.model.recurrent[name]) for name,t in recurrent.items()):
                raise RuntimeError('Query-boundary recurrent state differs from its CPU checkpoint')
            import json
            with (self.runner._kvmem_probe_output/'verified_state_restores.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_id':request_id,'history_tokens':binding.history,
                    'working_tokens':binding.window,'recurrent_tensors':len(recurrent),
                    'checkpoint_kind':checkpoint.kind,
                    'pressure_restore':binding.pressure_restore,'recurrent_byte_equal':True})+'\n')
        if oracle is not None:
            actual = self._target_bytes(request_id, working_tokens)
            if not torch.equal(actual, oracle):
                import json
                width = self.runtime.heads * self.runtime.dim
                size = working_tokens * width
                differences = []
                for i, name in enumerate(self.runtime.names):
                    row = {'layer': name}
                    for j, kind in enumerate(('key', 'value')):
                        begin = (2 * i + j) * size
                        mismatch = (actual[begin:begin+size] != oracle[begin:begin+size])
                        positions = mismatch.reshape(-1, width).any(dim=1).nonzero().flatten()
                        row[kind] = {'bytes': int(mismatch.sum()), 'rows': positions.numel(),
                                     'first_rows': positions[:8].tolist()}
                    differences.append(row)
                (self.runner._kvmem_probe_output / 'restore_mismatch.json').write_text(
                    json.dumps(differences, indent=2))
                raise RuntimeError('Restored target KV differs from its '+oracle_kind+' oracle')
            recurrent = self.model_adapter._recurrent_views(request_id)
            draft = self.model_adapter._draft_views(request_id, checkpoint.model.draft_begin,
                                                   checkpoint.position)
            if (any(not torch.equal(t.cpu(), checkpoint.model.recurrent[name]) for name, t in recurrent.items())
                    or any(not torch.equal(cache[pages, offsets].cpu(), checkpoint.model.draft[name])
                           for name, (cache, pages, offsets) in draft.items())):
                raise RuntimeError('Restored GDN or draft state differs from its terminal checkpoint')
            import json
            with (self.runner._kvmem_probe_output / 'verified_restores.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_id': request_id, 'history_tokens': binding.history,
                    'working_tokens': working_tokens, 'target_kv_bytes': oracle.numel(),
                    'working_attention_layers': len(self.runtime.names),
                    'mtp_attention_layers': len(getattr(self.runner, '_kvmem_mtp_modules', {})),
                    'recurrent_tensors': len(recurrent), 'draft_layers': len(draft),
                    'oracle_kind': oracle_kind, 'byte_equal': True}) + '\n')

    def prepare_restore_metadata(self, request_id, authority):
        binding = self.bindings[request_id]
        from .prompt_spans import selection_metadata, install_selection_metadata
        graph = selection_metadata(authority)
        if graph is not None:
            # The checkpoint claim has already compared exact token prefixes.
            # Omitted metadata inherits only this verified historical prefix;
            # conflicting boundaries are errors, never silently spliced.
            install_selection_metadata(authority, binding,
                validated_prefix=getattr(authority, '_graph_validated_prefix', binding.history))

    def publish_restore(self, request_id, authority, *, retained=False):
        from .cache_groups import full_attention_groups
        from .window_batch import WindowBatchAdapter
        binding, cfg = self.bindings[request_id], self.options.memory
        from .prompt_spans import selection_metadata
        if selection_metadata(authority) is not None:
            authority.graph_request_id = request_id
        if self.runner._kvmem_window_adapter is None:
            self.runner._kvmem_window_adapter = WindowBatchAdapter(
                full_attention_groups(self.runner.kv_cache_config), cfg.select_budget + cfg.gen_budget)
        self.runner._kvmem_window_adapter.commit(request_id, binding.history, binding.window)
        # Another request may be publishing CPU history while this restore
        # completes. Register the new owner atomically with its initial fence.
        with self.runtime._publication_lock():
            self.runtime._raise_publication_error()
            self.runtime.authorities[request_id] = authority
            self.runtime.accepted[request_id] = binding.history
            self.runtime.rejected[request_id] = 0
            self.runtime.last_steps[request_id] = []
            self.runtime._snapshot_capacity(request_id, authority)
        if authority.gc is not None:
            authority.memory._store.gc_begin_step()
        self.restored_tokens += binding.history
        self.completed.append(SessionNotice('restored', binding.namespace, binding.generation,
                                            request_id, binding.history, binding.window, retained=retained))

    def detach(self, request_id, authority, report):
        binding = self.bindings.get(request_id)
        if binding is None:
            return False
        history = getattr(authority,'query_history',None)
        if history is not None and history.policy.open:
            # A cancelled/incomplete prefill cannot retain provisional relevance
            # or its GPU Q tensor in an idle CPU-only checkpoint.
            history.abort()
        command = getattr(self, 'terminal_captures', {}).pop(request_id, None)
        if command is not None:
            if command.position != authority.history_end:
                raise ValueError('Terminal checkpoint differs from the accepted history')
            if self.model_adapter is None:
                self.model_adapter = PromptStateAdapter(self.runner)
            try:
                self.pool.capture(binding.namespace, request_id, authority, command.tokens,
                                  self.model_adapter, replace=True, terminal=True)
            except BufferError:
                self.checkpoint_deferrals += 1
                kind = 'capture_deferred'
            else:
                kind = 'captured'
            self.completed.append(SessionNotice(kind, binding.namespace, binding.generation,
                                                request_id, command.position))
        checkpoint = self.pool.entries.get(binding.namespace)
        if checkpoint is None:
            return False  # Ordinary retirement will emit an empty checkpoint acknowledgment.
        if checkpoint.request_id != request_id or checkpoint.authority is not authority:
            raise ValueError('Terminal request does not own the retained prompt snapshot')
        if command is not None and kind == 'captured' and os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1':
            if os.getenv('KVMEM_VERIFY_SESSION_RESTORE_DIGEST') == '1':
                from .restore_digest import target_digest
                oracle = target_digest(self, request_id, checkpoint.window.window_tokens)
            else:
                oracle = self._target_bytes(request_id, checkpoint.window.window_tokens)
            self.restore_oracles[binding.namespace] = (checkpoint.position, oracle)
        verified = verify_preserved_authority(authority) if self.options.verify_archive_on_finish else 0
        # Accepted KV capture has drained. The independent model snapshot can
        # still be downloading; pool.idle gates this request's retirement ack.
        self.pool.retire(checkpoint, _Ready())
        self.retiring[request_id] = (checkpoint, {**report, 'history_end': authority.history_end,
            'checkpoint_tokens': checkpoint.position, 'sealed_records_verified': verified,
            'retirement_strategy': 'terminal_checkpoint' if checkpoint.kind=='M' else 'checkpoint_rewind',
            'checkpoint_kind': checkpoint.kind, 'checkpoint_payload_bytes': checkpoint.payload_bytes,
            'checkpoint_pool_bytes': self.pool.payload_bytes,
            'checkpoint_pool_limit_bytes': self.pool.max_payload_bytes,
            'checkpoint_deferrals': self.checkpoint_deferrals,
            'authority_namespace': authority.memory.request_id})
        return True

    def retirement_done(self, request_id):
        pending = self.retiring.get(request_id)
        if pending is None:
            return None
        checkpoint, report = pending
        if not self.pool.idle(checkpoint):
            return False
        binding = self.bindings.pop(request_id)
        prompt = self.pool.prompts.get(binding.namespace)
        historical = getattr(checkpoint.authority.gc, 'native_history', False)
        allocated, protected = self._history_rows(checkpoint) if historical else (0, 0)
        self.completed.append(SessionNotice('retired', binding.namespace, binding.generation,
            request_id, checkpoint.position, checkpoint.window.window_tokens,
            checkpoint.authority.memory.repository.ready_slots,
            prompt.position if prompt else 0,prompt.window.window_tokens if prompt else 0,
            prompt.tokens if prompt else b'',
            tuple((r[2],r[3]) for r in checkpoint.window.records),
            resident_ready=(checkpoint.kind == 'M' and not checkpoint.rewind_started and
                            checkpoint.position == report['history_end'] and
                            (report.get('terminal_prefix') or {}).get('status') in
                            ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED')),
            allocated_records=allocated, protected_records=protected))
        if historical:
            self._history_capacity_sent[binding.namespace] = (binding.generation, checkpoint.position,
                checkpoint.authority.memory.repository.ready_slots, allocated, protected)
            if binding.namespace not in self._history_idle_queued:
                self._history_idle.append(binding.namespace)
                self._history_idle_queued.add(binding.namespace)
        self.runtime.completed.append({**report, 'retirement_ack_ready_at': time.perf_counter()})
        del self.retiring[request_id]
        self.runtime._write_archive_report()
        return True

    def ordinary_retired(self, request_id):
        getattr(self, 'release_requests', set()).discard(request_id)
        binding = self.bindings.pop(request_id, None)
        if binding is not None:
            self.completed.append(SessionNotice('retired', binding.namespace, binding.generation, request_id))
            del self.identities[binding.namespace]

    def evict(self, commands):
        for command in commands:
            if self.identities.get(command.namespace) != command.generation or command.namespace in self.evictions:
                raise ValueError('Stale or duplicate session eviction')
            checkpoint = self.pool.entries[command.namespace]
            authority = checkpoint.authority
            self.evictions[command.namespace] = command
            self.pool.evict(checkpoint)
            from .gpu_raw_cache import retire_raw_cache
            retire_raw_cache(self.runner, authority)
        self._poll_evictions()

    def _poll_evictions(self):
        changed = False
        for namespace, command in list(self.evictions.items()):
            checkpoint = self.pool.entries.get(namespace)
            if checkpoint is not None and not self.pool.poll_eviction(checkpoint):
                continue
            self.completed.append(SessionNotice('evicted', namespace, command.generation, ''))
            del self.evictions[namespace]
            del self.identities[namespace]
            getattr(self, 'restore_oracles', {}).pop(namespace, None)
            self._history_capacity_sent.pop(namespace, None)
            changed = True
        if changed and self.runtime is not None:
            self.runtime._write_archive_report()

    def notices(self):
        self.pool.poll_captures()
        self._poll_evictions()
        self._poll_idle_history()
        result, self.completed = self.completed, []
        return result

    def _poll_idle_history(self, budget=4):
        from .prompt_checkpoints import CheckpointPhase
        for _ in range(min(budget, len(self._history_idle))):
            namespace = self._history_idle.popleft()
            self._history_idle_queued.discard(namespace)
            checkpoint = self.pool.entries.get(namespace)
            if checkpoint is None or checkpoint.phase is not CheckpointPhase.IDLE:
                continue
            authority = checkpoint.authority
            authority.poll()
            allocated, protected = self._history_rows(checkpoint)
            state = (self.identities[namespace], authority.history_end, authority.memory.repository.ready_slots,
                     allocated, protected)
            if self._history_capacity_sent.get(namespace) != state:
                self.completed.append(SessionNotice('history_capacity', namespace, state[0], '',
                    history=state[1], ready_records=state[2], allocated_records=state[3], protected_records=state[4]))
                self._history_capacity_sent[namespace] = state
            self._history_idle.append(namespace)
            self._history_idle_queued.add(namespace)

    def _receive_history_capacity(self, requests):
        from .prompt_checkpoints import CheckpointPhase
        for request in requests:
            checkpoint = self.pool.entries.get(request.namespace)
            if (self.identities.get(request.namespace) != request.generation or checkpoint is None or
                    checkpoint.phase is CheckpointPhase.CLOSING):
                continue  # A late hint cannot change a replacement generation.
            reset = request.ready_records == 0 and request.retained_records is None
            if not reset and checkpoint.phase is not CheckpointPhase.IDLE:
                continue
            authority = checkpoint.authority
            if not getattr(authority.gc, 'native_history', False):
                raise ValueError('Historical pressure requires a native history scope')
            # Scheduler envelopes include index-only tails and pending index
            # owners. Native pressure counts published CPU payloads only.
            # Keeping those envelopes in the payload target leaves one extra
            # sealed record per idle scope and can strand a queued turn.
            index_only = authority._resident_count + len(authority.gc.pending_index)
            limit = (2**64-1 if request.retained_records is None else
                     max(0, request.retained_records-index_only) * authority.record_bytes)
            authority.gc.scope.request_capacity(request.ready_records, limit)

    def _history_rows(self, checkpoint):
        authority = checkpoint.authority
        allocated = authority.archive.namespace_allocated_records(authority.memory.request_id)
        # Retain index-only partial tails and already detached index owners in
        # the same conservative record envelope until their native release.
        index_only = authority._resident_count + len(authority.gc.pending_index)
        allocated += index_only
        frames = [checkpoint]
        prompt = self.pool.prompts.get(checkpoint.namespace)
        if prompt is not None:
            frames.append(prompt)
        protected = {(slot, generation) for frame in frames for slot, generation, *_ in frame.window.records}
        cfg = authority.memory.config
        for record in islice(authority.records.values(), cfg.sink_blocks):
            if record.history_begin < cfg.sink_blocks*cfg.block_tokens:
                protected.add((record.handle.slot, record.handle.generation))
        for record in islice(reversed(authority.records.values()), cfg.recent_blocks):
            if record.history_begin+record.count > authority.history_end-cfg.recent_blocks*cfg.block_tokens:
                protected.add((record.handle.slot, record.handle.generation))
        protected_cpu = sum(key in authority.records and authority.records[key].archived for key in protected)
        # LRU cannot delete an unsealed tail or accelerate an index owner
        # already draining. Never offer these envelopes as reclamation supply
        # for another scope: an older tail could otherwise starve later victims.
        return allocated, protected_cpu+index_only

    def close(self):
        from .prompt_checkpoints import CheckpointPhase
        for checkpoint in list(self.pool.entries.values()):
            if checkpoint.phase is CheckpointPhase.LIVE:
                self.pool.drop_live(checkpoint)  # Runtime still owns active authority.
            elif checkpoint.phase is CheckpointPhase.CLOSING:
                self.pool.poll_eviction(checkpoint)
            else:
                self.pool.evict(checkpoint, _Ready() if checkpoint.phase is CheckpointPhase.CLAIMED else None)
        if not self.pool.poll_captures() or self.pool.entries:
            return False
        self.bindings.clear()
        getattr(self, 'release_requests', set()).clear()
        self.identities.clear()
        self.imports.clear()
        self.captures.clear()
        getattr(self, 'terminal_captures', {}).clear()
        getattr(self, 'restore_oracles', {}).clear()
        self.retiring.clear()
        self.evictions.clear()
        self._history_idle.clear()
        self._history_idle_queued.clear()
        self._history_capacity_sent.clear()
        return True
