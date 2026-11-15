"""Worker-side prompt checkpoint capture, import and fenced retirement.

Import runs after native request/table initialization and block zeroing, but
before attention metadata is built. Native sampling and draft proposal retain
their normal per-request lifecycle. CPU checkpoint ownership is bounded separately
from the shared authority arena; the runner must reserve room for idle authorities.
"""
import hashlib
import os
import time
import weakref
from collections import deque
from itertools import islice

import torch

from .authority import _Ready
from .prompt_checkpoints import PromptCheckpointPool
from .session_control import SessionNotice
from .session_state import PromptStateAdapter
from ..core import settings as _settings


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


def deferred_p_capture():
    """Opt-in KVMEM_DEFERRED_P_CAPTURE=1: a P checkpoint's CPU snapshots run after
    the next forward is enqueued (overlapping it); its model-state copy stays
    before that forward. Same checkpoint contents and admission decisions."""
    value = _settings.get('KVMEM_DEFERRED_P_CAPTURE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_DEFERRED_P_CAPTURE must be 0 or 1')
    return value == '1'


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
        self.prepared_terminal_windows = {}
        self.restore_oracles = {}
        self._swap_fences = {}
        self.checkpoint_deferrals = 0
        self.restored_tokens = 0
        self.release_requests = set()
        self._history_idle = deque()
        self._history_idle_queued = set()
        self._history_capacity_sent = {}
        self.tiering = None          # opt-in cpu_session_ttl_ms (created with the archive)

    def session_tiering(self):
        """Opt-in cpu_session_ttl_ms: CPU -> SSD tiering of idle sessions."""
        ttl = getattr(self.options, 'cpu_session_ttl_ms', 0)
        if not ttl:
            return None
        if self.tiering is None:
            archive = getattr(self.runtime, 'archive', None)
            if archive is None:
                return None
            from .session_tiering import SessionTiering
            archive.tier.enable_spill(self.options.cpu_spill_dir, self.options.cpu_spill_max_bytes)
            self.tiering = SessionTiering(archive, ttl / 1000)
        return self.tiering

    @property
    def runtime(self):
        return self.runner._kvmem_capture_runtime

    def _swap_binding(self, request_id, identity):
        from .session_swap import SessionSwapIdentity
        binding = self.bindings.get(request_id)
        if identity is None:
            if binding is not None:
                raise ValueError('Owned session swap omitted its namespace identity')
            return None
        if (not isinstance(identity, SessionSwapIdentity) or binding is None or
                binding.namespace != identity.namespace or binding.generation != identity.generation or
                self.identities.get(binding.namespace) != identity.generation):
            raise ValueError('Session swap namespace or generation changed')
        return binding

    def swap_ready(self, request_id, identity):
        """Fence existing P/M downloads after accepted-history publication."""
        binding = self._swap_binding(request_id, identity)
        if binding is None:
            return True
        if (request_id in self.imports or request_id in self.captures or
                request_id in self.terminal_captures or request_id in self.prepared_terminal_windows or
                request_id in self.retiring or binding.namespace in self.evictions):
            return False
        authority = self.runtime.authorities.get(request_id)
        if authority is None:
            return False
        from .prompt_checkpoints import CheckpointPhase
        checkpoints = [self.pool.entries.get(binding.namespace), self.pool.prompts.get(binding.namespace)]
        for checkpoint in checkpoints:
            if checkpoint is None:
                continue
            if (checkpoint.request_id != request_id or checkpoint.authority is not authority or
                    checkpoint.phase is not CheckpointPhase.LIVE):
                raise ValueError('Session swap does not own its live P/M checkpoint')
            if not self.pool.model_ready(checkpoint):
                return False
        # A replaced snapshot may still hold a download buffer. Its native
        # source is independent, but fence that owner's copies conservatively.
        return all(self.pool.model_ready(checkpoint) for checkpoint in self.pool._discarded_captures
                   if checkpoint.namespace == binding.namespace)

    def begin_swap(self, request_id, identity):
        from .session_swap import SessionSwapFence
        if not self.swap_ready(request_id, identity):
            raise RuntimeError('Session swap preceded import or checkpoint completion')
        if identity is None:
            return
        if request_id in self._swap_fences:
            raise ValueError('Duplicate active session swap fence')
        binding = self.bindings[request_id]
        self._swap_fences[request_id] = SessionSwapFence(identity, binding,
            self.runtime.authorities[request_id], self.pool.entries.get(binding.namespace),
            self.pool.prompts.get(binding.namespace))

    def validate_swap(self, request_id, identity):
        binding = self._swap_binding(request_id, identity)
        if identity is None:
            if request_id in self._swap_fences:
                raise ValueError('Session swap lost its retained fence')
            return
        fence = self._swap_fences.get(request_id)
        if (fence is None or fence.identity != identity or fence.binding is not binding or
                self.runtime.authorities.get(request_id) is not fence.authority or
                self.pool.entries.get(binding.namespace) is not fence.checkpoint or
                self.pool.prompts.get(binding.namespace) is not fence.prompt or
                not self.swap_ready(request_id, identity)):
            raise ValueError('Retained session swap ownership changed')

    def end_swap(self, request_id, identity):
        """Only after exact restore/drop ACK or drained unpublished failure."""
        self.validate_swap(request_id, identity)
        if identity is not None:
            del self._swap_fences[request_id]

    def _require_not_swapped(self, request_ids):
        if set(request_ids).intersection(getattr(self, '_swap_fences', ())):
            raise RuntimeError('Session lifecycle mutation precedes exact request swap release')

    def receive(self, commands):
        self._require_not_swapped(command.request_id for name in ('bindings', 'captures', 'terminals')
                                 for command in getattr(commands, name, ()))
        retained_namespaces = {fence.identity.namespace for fence in getattr(self, '_swap_fences', {}).values()}
        if any(binding.namespace in retained_namespaces for binding in commands.bindings):
            raise RuntimeError('Namespace reuse precedes exact session swap release')
        self.release_requests.update(getattr(commands, 'release_requests', ()))
        self._receive_history_capacity(getattr(commands, 'history_capacity', ()))
        image_releases = getattr(commands, 'image_releases', ())
        image_imports = getattr(commands, 'image_imports', ())
        if image_releases or image_imports:
            # Prefix images (serving/prefix_images.py): drop released images, then
            # materialize new sessions as idle frozen P checkpoints.
            if any(command.namespace in retained_namespaces for command in image_imports):
                raise RuntimeError('Namespace reuse precedes exact session swap release')
            from .prefix_images import import_commands, release_images
            release_images(self, image_releases)
            import_commands(self, image_imports)
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

    def before_attention(self, batch, *, deferrable=False):
        # deferrable: the runner's public-forward call, whose finish hook runs
        # after the launch. Private maintenance forwards capture synchronously.
        if deferrable and getattr(self, 'deferred_captures', None):
            raise RuntimeError('A deferred P checkpoint was not finished after its forward')
        self._require_not_swapped(batch.req_ids)
        if getattr(self.options, 'async_restore', False) and self.imports.intersection(batch.req_ids):
            raise RuntimeError('Unready checkpoint restore entered a public model batch')
        selected = self.imports.intersection(batch.req_ids) | (self.captures.keys() & set(batch.req_ids))
        if not selected:
            return
        if self.runtime is None:
            raise RuntimeError('Prompt checkpoint has no accepted authority runtime')
        if not self.runtime.drain_requests(selected, wait=True):
            raise RuntimeError('Prompt checkpoint preceded scheduler publication confirmation')
        _marks = getattr(self.runner, '_kvmem_step_timing', None)
        if _marks is not None:
            _marks.mark('sess_drain')
        if self.model_adapter is None:
            self.model_adapter = PromptStateAdapter(self.runner)
        for request_id in batch.req_ids:
            if request_id in self.imports:
                self._restore(request_id)
                self.imports.remove(request_id)
            command = self.captures.pop(request_id, None)
            if command is None:
                continue
            self._take_capture(request_id, command, deferrable, _marks)

    def _take_capture(self, request_id, command, deferrable, _marks=None):
        """One P capture of a drained request (before_attention, held captures)."""
        binding = self.bindings[request_id]
        authority = self.runtime.authorities[request_id]
        if authority.history_end != command.position:
            raise ValueError('Prompt capture differs from accepted archive history')
        from .pressure_promise import pm_sessions, settle_before_checkpoint
        if pm_sessions():
            settle_before_checkpoint(self.runner, request_id)   # opt-in KVMEM_PROMISE_PM_SESSIONS
        replace = getattr(self.options, 'session_capture_decode', False) or self.pool.retain_prompt
        if deferrable and deferred_p_capture() and not getattr(self.options, 'session_freeze_prompt', False):
            # Opt-in: the model-state copy now (it must precede this forward);
            # the CPU snapshots in finish_deferred_captures() after the launch.
            try:
                pending = self.pool.begin_capture(binding.namespace, request_id, authority,
                    command.tokens or binding.tokens, self.model_adapter, replace=replace)
            except BufferError:
                self._capture_not_admitted(binding, request_id, command)
            else:
                self.__dict__.setdefault('deferred_captures', {})[request_id] = [
                    pending, binding, command, False, self._capture_witness(request_id, authority)]
            return
        try:
            checkpoint = self.pool.capture(binding.namespace, request_id, authority,
                command.tokens or binding.tokens, self.model_adapter, replace=replace)
        except BufferError:
            self._capture_not_admitted(binding, request_id, command)
        else:
            if _marks is not None:
                _marks.mark('sess_capture')
            self._capture_installed(binding, request_id, command, checkpoint)

    def _capture_not_admitted(self, binding, request_id, command):
        # Cache admission failure preserves ordinary native execution.
        self.checkpoint_deferrals += 1
        if getattr(self.options, 'session_capture_decode', False) or self.pool.retain_prompt:
            self.completed.append(SessionNotice('capture_deferred', binding.namespace,
                binding.generation, request_id, command.position))

    def _capture_installed(self, binding, request_id, command, checkpoint):
        if checkpoint.layerwise_pressure is not None:
            from .layerwise_checkpoints import emit_prefix_evidence
            emit_prefix_evidence(self.runner._kvmem_layerwise_pressure.query,
                'pressure_prefix_captured', checkpoint.layerwise_pressure,
                request_id=request_id, actual_method='accumulated_attention_per_layer',
                fallback_reason=None)
        if getattr(self.options, 'session_capture_decode', False) or self.pool.retain_prompt:
            self.completed.append(SessionNotice('captured', binding.namespace,
                binding.generation, request_id, command.position))

    def _capture_witness(self, request_id, authority):
        """What a P capture reads, cheaply: an unexpected change between
        begin_capture() and finish_capture() fails loudly instead of silently
        capturing a different state."""
        controller = getattr(self.runner, '_kvmem_layerwise_pressure', None)
        maps = getattr(controller, 'maps', {}).get(request_id) if controller is not None else None
        histories = getattr(controller, 'histories', {}).get(request_id) if controller is not None else None
        action = getattr(authority, 'action_history', None)
        return (authority.history_end, getattr(authority.records, 'version', None),
                tuple(authority.memory._store.working_records()), id(maps), id(histories),
                getattr(action, 'rounds', None), len(getattr(action, 'scores', ()) or ()))

    def finish_deferred_captures(self):
        """Opt-in KVMEM_DEFERRED_P_CAPTURE: runs once the forward that follows
        begin_capture() is enqueued, so these CPU snapshots overlap the GPU."""
        pending = getattr(self, 'deferred_captures', None)
        if not pending:
            return
        self.deferred_captures = {}
        self.deferred_p_captures = getattr(self, 'deferred_p_captures', 0) + len(pending)
        for request_id, (capture, binding, command, committed, witness) in pending.items():
            if not committed and self._capture_witness(request_id, capture.authority) != witness:
                raise RuntimeError('Deferred P checkpoint inputs changed before its capture')
            try:
                checkpoint = self.pool.finish_capture(capture, committed=committed)
            except BufferError:
                self._capture_not_admitted(binding, request_id, command)
            else:
                self._capture_installed(binding, request_id, command, checkpoint)

    def finish_held_captures(self, scheduled, finished=()):
        """Opt-in KVMEM_HELD_P_CAPTURE: the P captures of held final pieces.

        The runner calls this once the step's forward is enqueued, so the
        capture overlaps the GPU instead of delaying a batch launch. A held
        request is in no batch until its capture is reported, so its accepted
        history, model state and frames are exactly what its own final piece
        would have captured before its forward; the capture call is the same.
        """
        held = [request_id for request_id, command in self.captures.items()
                if getattr(command, 'held', False) and request_id not in scheduled]
        if not held:
            return
        done = self.__dict__.setdefault('held_done', [])
        authorities = getattr(self.runtime, 'authorities', {}) if self.runtime is not None else {}
        live = []
        for request_id in held:
            if request_id in finished or request_id not in self.bindings or request_id not in authorities:
                self.captures.pop(request_id)      # cancelled before its final piece ran
                done.append(request_id)
            else:
                live.append(request_id)
        if not live:
            return
        self._require_not_swapped(live)
        if not self.runtime.drain_requests(live, wait=True):
            return      # an unconfirmed sample: the piece stays held, the next step retries
        if self.model_adapter is None:
            self.model_adapter = PromptStateAdapter(self.runner)
        for request_id in live:
            self._take_capture(request_id, self.captures.pop(request_id), False)
            done.append(request_id)
        self.held_p_captures = getattr(self, 'held_p_captures', 0) + len(live)

    def take_held_done(self):
        """Held final pieces whose capture finished (or that were cancelled)."""
        done = getattr(self, 'held_done', None)
        if not done:
            return []
        self.held_done = []
        return done

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
        self._require_not_swapped((request_id,))
        deferred = getattr(self, 'deferred_captures', None)
        if deferred and request_id in deferred and deferred[request_id][0].position == history:
            # The P is not installed yet: finish_capture() takes it as committed,
            # which equals capture() followed by the refresh below.
            deferred[request_id][3] = True
            return
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
            from .layerwise_checkpoints import refresh_committed_snapshot
            refresh_committed_snapshot(self, request_id, checkpoint)
            self.pool.update_live_window(checkpoint)

    def _restore(self, request_id):
        self._require_not_swapped((request_id,))
        restore_started = time.perf_counter()
        restore_stages = {}
        from ..adapters.cache_groups import full_attention_groups
        from .window_assembly import NativeWindowAssembler
        from .window_batch import WindowBatchAdapter
        binding = self.bindings[request_id]
        checkpoint = self.pool.claim(binding.namespace, request_id, binding.tokens,position=binding.history)
        if checkpoint is not None and self.session_tiering() is not None:
            self.tiering.ensure(binding.namespace, checkpoint.authority)     # SSD -> CPU first
        if (checkpoint is None or checkpoint.position != binding.history or
                (checkpoint.window.window_tokens != binding.window and not binding.pressure_restore)):
            raise ValueError('Worker cannot restore the advertised exact prompt prefix')
        if request_id in self.runtime.authorities:
            raise ValueError('Native import already owns a different authority')
        from .layerwise_checkpoints import require_supported_restore, restore_sync
        require_supported_restore(self, request_id, checkpoint)
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
            stage_started = time.perf_counter()
            restore_stages['claim_and_prepare'] = stage_started - restore_started
            from .layerwise_restore_efficiency import install_direct_restore, finish_direct_restore
            direct_restore = install_direct_restore(self, request_id, checkpoint, assembler, plan)
            assembly = assembler.assemble(authority, plan)
            assembly.event.synchronize()
            authority.memory.attach_completion(assembly)
            assert authority.memory.commit_selection()
            now = time.perf_counter()
            restore_stages['shared_kv_restore'] = now - stage_started
            stage_started = now
            if direct_restore:
                finish_direct_restore(self, request_id, checkpoint, assembler)
            elif binding.pressure_restore:
                from .layerwise_checkpoints import restore_pressure_sync
                restore_pressure_sync(self, request_id, checkpoint)
            else:
                restore_sync(self, request_id, checkpoint)
            now = time.perf_counter()
            restore_stages['layerwise_kv_restore'] = now - stage_started
            stage_started = now
            event = self.model_adapter.restore(checkpoint.model, request_id)
            if event is not None:
                event.synchronize()
            now = time.perf_counter()
            restore_stages['recurrent_restore'] = now - stage_started
            stage_started = now
            self._verify_restored(request_id, checkpoint)
            restore_stages['verification'] = time.perf_counter() - stage_started
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
        from .layerwise_checkpoints import realign_after_restore
        realign_after_restore(self, request_id, checkpoint)
        output = getattr(self.runner, '_kvmem_probe_output', None)
        if output is not None:
            import json
            with (output/'sync_restore_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(request_id=request_id,
                    checkpoint_kind=checkpoint.kind, history_tokens=binding.history,
                    window_tokens=binding.window, namespace=list(binding.namespace),
                    generation=binding.generation,
                    started_at=restore_started, finished_at=time.perf_counter(),
                    host_stages_seconds=restore_stages,
                    scope='Host wall intervals including existing CUDA waits; not exclusive GPU kernel time.'))+'\n')

    def _rebuilt_target_bytes(self, request_id, authority, *, record_groups=None):
        """Verification-only CPU oracle from immutable FP8 source records.

        Do not compare a cold FP8-raw rebuild to the original unquantized
        forward's rotated K. The source quantization is intentionally lossy.
        This oracle uses independent Torch math, never the assembly kernels.
        """
        from .window_assembly import NativeWindowAssembler
        from ..core.codec import neox_rope
        authority.publish_sealed(self.runtime._wait_archive_progress)
        records = tuple(authority.memory._store.working_records())
        assembler = NativeWindowAssembler.from_runner(self.runner, request_id, self.runtime.names,
                                                       self.options.memory.block_tokens)
        try:
            keys, values = [[] for _ in assembler.layers], [[] for _ in assembler.layers]
            trigs = [layer.cos_sin.cpu() for layer in assembler.layers]
            scales = [layer.key_scale.cpu() for layer in assembler.layers]
            from .layerwise_checkpoints import oracle_record_groups
            groups = (oracle_record_groups(self, request_id, authority)
                      if record_groups is None else record_groups)
            for frame, indices in groups:
                for start in range(0, len(frame), 32):
                    chunk = frame[start:start+32]
                    handles = [authority.records[(slot, generation)].handle
                               for slot, generation, *_ in chunk]
                    payloads = authority.read_records(handles)
                    for (_, _, _, count, position), (raw, value) in zip(chunk, payloads):
                        positions = torch.arange(position, position+count)
                        for index in indices:
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
        if _settings.get('KVMEM_VERIFY_SESSION_RESTORE_DIGEST') == '1':
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
        if (_settings.get('KVMEM_VERIFY_SESSION_RESTORE') == '1' and not resident and
                checkpoint.authority.raw_dtype == torch.float8_e4m3fn):
            oracle = self._rebuilt_target_bytes(request_id, checkpoint.authority)
            oracle_kind = 'fp8_immutable_rebuild'
        if _settings.get('KVMEM_VERIFY_SESSION_RESTORE') == '1':
            recurrent = self.model_adapter._recurrent_views(request_id)
            if any(not torch.equal(t.cpu(), checkpoint.model.recurrent[name]) for name,t in recurrent.items()):
                raise RuntimeError('Query-boundary recurrent state differs from its CPU checkpoint')
            import json
            with (self.runner._kvmem_probe_output/'verified_state_restores.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_id':request_id,'history_tokens':binding.history,
                    'working_tokens':binding.window,'recurrent_tensors':len(recurrent),
                    'checkpoint_kind':checkpoint.kind,
                    'pressure_restore':binding.pressure_restore,'recurrent_byte_equal':True})+'\n')
        rebuilt_draft = getattr(self, 'layerwise_draft_rebuilt', set())
        if oracle is not None and request_id in rebuilt_draft:
            # KVMEM_LAYERWISE_SESSIONS resident restore after a probe-mode Query:
            # target layers keep their terminal pages; the MTP draft was rebuilt
            # from layer zero's window, so check it against the immutable rebuild.
            rebuilt_draft.discard(request_id)
            width = self.runtime.heads * self.runtime.dim
            size = working_tokens * width
            targets = len(self.runner._kvmem_layerwise_pressure.names)
            rebuilt = self._rebuilt_target_bytes(request_id, checkpoint.authority)
            oracle = oracle.clone()
            oracle[2*targets*size:] = rebuilt[2*targets*size:]
            oracle_kind = 'native_terminal_window_with_rebuilt_draft'
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
        from ..adapters.cache_groups import full_attention_groups
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

    def prepare_terminal_checkpoint(self, request_id, command):
        """Try once; the capture runtime retries busy gates on later steps."""
        self._require_not_swapped((request_id,))
        authority = self.runtime.authorities.get(request_id)
        if authority is None or not getattr(authority.memory._store, 'protect_current_window', False):
            return True
        binding = self.bindings.get(request_id)
        if binding is None or binding.generation != command.generation:
            raise ValueError('Terminal checkpoint preparation has a stale session identity')
        if command.position != authority.history_end:
            raise ValueError('Terminal checkpoint preparation precedes accepted history')
        prepared = self.prepared_terminal_windows.get(request_id)
        if prepared is not None:
            if prepared.position != command.position:
                raise ValueError('Prepared terminal checkpoint changed position')
            return True
        try:
            window = authority.memory.checkpoint_window(allow_partial=True,
                **({'native_validation': True} if self.pool.native_checkpoint_capture else {}))
        except BufferError:
            self.checkpoint_deferrals += 1
            return False
        self.prepared_terminal_windows[request_id] = window
        return True

    def detach(self, request_id, authority, report):
        self._require_not_swapped((request_id,))
        binding = self.bindings.get(request_id)
        if binding is None:
            return False
        history = getattr(authority,'query_history',None)
        if history is not None and history.policy.open:
            # A cancelled/incomplete prefill cannot retain provisional relevance
            # or its GPU Q tensor in an idle CPU-only checkpoint.
            history.abort()
        command = getattr(self, 'terminal_captures', {}).pop(request_id, None)
        prepared = getattr(self, 'prepared_terminal_windows', {}).pop(request_id, None)
        if command is not None:
            if command.position != authority.history_end:
                raise ValueError('Terminal checkpoint differs from the accepted history')
            from .pressure_promise import pm_sessions, settle_before_checkpoint
            if pm_sessions():
                settle_before_checkpoint(self.runner, request_id)   # opt-in KVMEM_PROMISE_PM_SESSIONS
            if self.model_adapter is None:
                self.model_adapter = PromptStateAdapter(self.runner)
            try:
                self.pool.capture(binding.namespace, request_id, authority, command.tokens,
                                  self.model_adapter, replace=True, terminal=True,
                                  **({'prepared_window': prepared} if prepared is not None else {}))
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
        if command is not None and kind == 'captured' and _settings.get('KVMEM_VERIFY_SESSION_RESTORE') == '1':
            if _settings.get('KVMEM_VERIFY_SESSION_RESTORE_DIGEST') == '1':
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
        ready = getattr(getattr(self.runtime, 'archive', None), 'reclamation_ready', None)
        if ready is not None and not ready():
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
        if getattr(binding, 'share_prefix', 0):
            # kvmem_share_prefix: this idle frozen P becomes a prefix image.
            from .prefix_images import register_image
            register_image(self, binding, checkpoint)
        if historical:
            self._history_capacity_sent[binding.namespace] = (binding.generation, checkpoint.position,
                checkpoint.authority.memory.repository.ready_slots, allocated, protected)
            if binding.namespace not in self._history_idle_queued:
                self._history_idle.append(binding.namespace)
                self._history_idle_queued.add(binding.namespace)
        self.runtime.completed.append({**report, 'retirement_ack_ready_at': time.perf_counter()})
        del self.retiring[request_id]
        tiering = self.session_tiering()
        if tiering is not None:
            tiering.idle(binding.namespace, checkpoint.authority)
        self.runtime._write_archive_report()
        return True

    def ordinary_retired(self, request_id):
        self._require_not_swapped((request_id,))
        getattr(self, 'release_requests', set()).discard(request_id)
        binding = self.bindings.pop(request_id, None)
        if binding is not None:
            self.completed.append(SessionNotice('retired', binding.namespace, binding.generation, request_id))
            del self.identities[binding.namespace]
            if getattr(binding, 'share_prefix', 0):
                from .prefix_images import emit
                emit(self, 'refused', reason='the request retained no prompt checkpoint',
                     namespace=list(binding.namespace), generation=binding.generation,
                     request_id=request_id, position=binding.share_prefix)

    def evict(self, commands):
        for command in commands:
            self._require_not_swapped(rid for rid, binding in self.bindings.items()
                                     if binding.namespace == command.namespace)
            if (getattr(command, 'allow_missing', False) and command.namespace not in self.identities
                    and command.namespace not in self.pool.entries and command.namespace not in self.evictions):
                # This rank already retired the uncached request. The scheduler
                # still needs its generation-qualified ACK for the common fence.
                self.completed.append(SessionNotice('evicted', command.namespace, command.generation, ''))
                continue
            if self.identities.get(command.namespace) != command.generation or command.namespace in self.evictions:
                raise ValueError('Stale or duplicate session eviction')
            checkpoint = self.pool.entries[command.namespace]
            authority = checkpoint.authority
            self.evictions[command.namespace] = command
            self.pool.evict(checkpoint)
            if self.tiering is not None:
                self.tiering.forget(command.namespace)
            from .gpu_raw_cache import retire_raw_cache
            retire_raw_cache(self.runner, authority)
        self._poll_evictions()

    def _audit_eviction(self, namespace, blocker, note=None):
        """Diagnostic: one line per pending eviction after 1,000, then every
        100,000 polls; and one line when a cycle collection resolved it."""
        counts = self.__dict__.setdefault('_eviction_polls', {})
        count = counts[namespace] = counts.get(namespace, 0) + (note is None)
        if note is not None or count == 1000 or count % 100000 == 0:
            output = getattr(self.runtime, 'output', None)
            if output is not None:
                import json, time
                with (output/'eviction_audit.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.time(), namespace=list(namespace),
                                                 polls=count, blocker=blocker, **({'resolved': note} if note else {})))+'\n')

    def _poll_evictions(self):
        changed = False
        for namespace, command in list(self.evictions.items()):
            checkpoint = self.pool.entries.get(namespace)
            if checkpoint is not None and not self.pool.poll_eviction(checkpoint):
                self._audit_eviction(namespace, getattr(checkpoint, 'eviction_blocker', None))
                continue
            ready = getattr(getattr(self.runtime, 'archive', None), 'reclamation_ready', None)
            if ready is not None and not ready():
                self._audit_eviction(namespace, 'archive_reclamation')
                continue
            note = getattr(checkpoint, 'eviction_note', None) if checkpoint is not None else None
            if note is not None:
                self._audit_eviction(namespace, None, note=note)
            self.completed.append(SessionNotice('evicted', namespace, command.generation, ''))
            del self.evictions[namespace]
            self.__dict__.get('_eviction_polls', {}).pop(namespace, None)
            del self.identities[namespace]
            getattr(self, 'restore_oracles', {}).pop(namespace, None)
            self._history_capacity_sent.pop(namespace, None)
            changed = True
        if changed and self.runtime is not None:
            self.runtime._write_archive_report()

    def returning(self, namespaces):
        """Opt-in session tiering: these sessions' next turns arrived at the
        scheduler; start their SSD -> CPU promotion now (same bytes, earlier)."""
        tiering = self.session_tiering()
        if tiering is not None:
            for namespace in namespaces:
                tiering.prefetch(namespace)

    def notices(self):
        if self.tiering is not None:
            self.tiering.tick()
        self.pool.poll_captures()
        self._poll_evictions()
        self._poll_idle_history()
        self._pool_notice()
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

    def _protected_private(self, checkpoint, authority, shared):
        """_history_rows' protected archived records, excluding pool-charged ones."""
        frames = [checkpoint]
        prompt = self.pool.prompts.get(checkpoint.namespace)
        if prompt is not None:
            frames.append(prompt)
        protected = {(slot, generation) for frame in frames for slot, generation, *_ in frame.window.records}
        for frame in frames:
            layerwise = getattr(frame, 'layerwise_pressure', None)
            if getattr(layerwise, 'gc_hold', None) is not None:
                protected.update((row[0], row[1]) for rows in layerwise.maps for row in rows)
        cfg = authority.memory.config
        for record in islice(authority.records.values(), cfg.sink_blocks):
            if record.history_begin < cfg.sink_blocks*cfg.block_tokens:
                protected.add((record.handle.slot, record.handle.generation))
        for record in islice(reversed(authority.records.values()), cfg.recent_blocks):
            if record.history_begin+record.count > authority.history_end-cfg.recent_blocks*cfg.block_tokens:
                protected.add((record.handle.slot, record.handle.generation))
        return sum(key not in shared and key in authority.records and authority.records[key].archived
                   for key in protected)

    def _pool_notice(self):
        """Shared-prefix pool size for the scheduler, when it changed."""
        archive = getattr(self.runtime, 'archive', None) if self.runtime is not None else None
        tier = getattr(archive, 'tier', None)
        active = getattr(tier, 'pool_active', None)
        last = self.__dict__.get('_pool_sent')
        if active is None or (last is None and not active()):
            return
        state = tier.pool_state()
        if state != last:
            self._pool_sent = state
            from .prefix_images import HistoryPoolNotice
            self.completed.append(HistoryPoolNotice(records=state[0], draining=state[1]))

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
        archive = authority.archive
        allocated = archive.namespace_allocated_records(authority.memory.request_id)
        # Retain index-only partial tails and already detached index owners in
        # the same conservative record envelope until their native release.
        index_only = authority._resident_count + len(authority.gc.pending_index)
        allocated += index_only
        held = authority.__dict__.get('_prefix_shared_keys')
        if held:
            # Shared prefixes: descriptors charged to the shared-prefix pool are not
            # this session's; its own MeanK index of each shared record (live, or
            # retired with its index owner still pending) is, as a fraction of the
            # per-record charge. Counting by index owner keeps the figure from
            # growing while GC retires a shared record. Protection counts private
            # records only.
            from .prefix_images import index_records
            pending = authority.gc.pending_index
            held.intersection_update(authority.records.keys() | pending.keys())
            shared = archive.namespace_shared_records(authority.memory.request_id)[0]
            pending_shared = sum(key in held for key in pending)
            allocated += index_records(archive, len(held)) - len(shared) - pending_shared
            return allocated, (self._protected_private(checkpoint, authority, held | shared) +
                               index_only - pending_shared)
        frames = [checkpoint]
        prompt = self.pool.prompts.get(checkpoint.namespace)
        if prompt is not None:
            frames.append(prompt)
        from .layerwise_checkpoints import fast_state
        if fast_state():
            # Opt-in KVMEM_CHECKPOINT_FAST_STATE: the same count without
            # rebuilding the frame sets on every idle-history poll.
            protected_cpu = _protected_archived(checkpoint, frames, authority)
            if protected_cpu is not None:
                return allocated, protected_cpu+index_only
        protected = {(slot, generation) for frame in frames for slot, generation, *_ in frame.window.records}
        for frame in frames:
            # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: a per-layer checkpoint holds
            # every layer's records (layerwise_gc_holds.checkpoint_hold).
            layerwise = getattr(frame, 'layerwise_pressure', None)
            if getattr(layerwise, 'gc_hold', None) is not None:
                # Opt-in KVMEM_CHECKPOINT_FAST_STATE: the same pairs, built
                # once per checkpoint instead of on every idle-history poll.
                from .layerwise_checkpoints import map_pairs
                pairs = map_pairs(layerwise)
                protected.update(pairs if pairs is not None else
                                 ((row[0], row[1]) for rows in layerwise.maps for row in rows))
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
        if getattr(self, '_swap_fences', None):
            return False  # The swap owner must drain and release its exact lease.
        # Prepared frames belong to still-live owners, never to an idle pool
        # entry. Drop them before eviction can wait for their repository pins.
        getattr(self, 'prepared_terminal_windows', {}).clear()
        for checkpoint in list(self.pool.entries.values()):
            if checkpoint.phase is CheckpointPhase.LIVE:
                self.pool.drop_live(checkpoint)  # Runtime still owns active authority.
            elif checkpoint.phase is CheckpointPhase.CLOSING:
                self.pool.poll_eviction(checkpoint)
            else:
                self.pool.evict(checkpoint, _Ready() if checkpoint.phase is CheckpointPhase.CLAIMED else None)
        if not self.pool.poll_captures() or self.pool.entries:
            return False
        ready = getattr(getattr(self.runtime, 'archive', None), 'reclamation_ready', None)
        if ready is not None and not ready():
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


def _frame_pairs(frame):
    """(slot, generation) of a frame's window and held layer maps, cached on it.

    The frozenset _history_rows built from ``window.records`` and the layerwise
    maps on every call. Valid while the frame keeps the same records tuple and
    the same (immutable) layerwise snapshot; None when it cannot be cached.
    """
    store = getattr(frame, '__dict__', None)
    window = getattr(frame, 'window', None)
    records = getattr(window, 'records', None)
    if store is None or type(records) is not tuple:
        return None
    layerwise = getattr(frame, 'layerwise_pressure', None)
    held = getattr(layerwise, 'gc_hold', None) is not None
    cached = store.get('_history_pairs')
    if cached is not None and cached[0] is records and cached[2] is held and (
            cached[1] is None if layerwise is None else
            cached[1] is not None and cached[1]() is layerwise):
        return cached[3]
    pairs = {(slot, generation) for slot, generation, *_ in records}
    if held:
        from .layerwise_checkpoints import map_pairs
        rows = map_pairs(layerwise)
        pairs.update(rows if rows is not None else
                     ((row[0], row[1]) for rows in layerwise.maps for row in rows))
    pairs = frozenset(pairs)
    try:
        # A weak reference: the cache must not keep a released checkpoint's
        # layer snapshot (and its native policy hold) alive.
        ref = None if layerwise is None else weakref.ref(layerwise)
    except TypeError:
        return pairs
    store['_history_pairs'] = (records, ref, held, pairs)
    return pairs


def _protected_archived(checkpoint, frames, authority):
    """Archived records among the protected ones, as _history_rows counts them.

    The count over the frames' pairs is reused while the pair sets, the record
    dict (``records.version``) and every archived flag (``archived_epoch``)
    are unchanged; the few sink/recent records are examined on every call.
    None (the caller's original code runs) when a frame cannot be cached.
    """
    records = authority.records
    version = getattr(records, 'version', None)
    epoch = getattr(authority, 'archived_epoch', None)
    store = getattr(checkpoint, '__dict__', None)
    if version is None or epoch is None or store is None:
        return None
    groups = []
    for frame in frames:
        pairs = _frame_pairs(frame)
        if pairs is None:
            return None
        groups.append(pairs)
    cached = store.get('_history_archived')
    if (cached is not None and cached[0] is records and cached[1] == version and cached[2] == epoch and
            len(cached[3]) == len(groups) and all(ref() is pairs for ref, pairs in zip(cached[3], groups))):
        base = cached[4]
    else:
        keys = groups[0] if len(groups) == 1 else groups[0].union(*groups[1:])
        get = records.get
        base = sum(record.archived for record in map(get, keys) if record is not None)
        store['_history_archived'] = (records, version, epoch, tuple(weakref.ref(pairs) for pairs in groups), base)
    cfg = authority.memory.config
    extra = set()
    for record in islice(records.values(), cfg.sink_blocks):
        if record.history_begin < cfg.sink_blocks*cfg.block_tokens:
            extra.add((record.handle.slot, record.handle.generation))
    for record in islice(reversed(records.values()), cfg.recent_blocks):
        if record.history_begin+record.count > authority.history_end-cfg.recent_blocks*cfg.block_tokens:
            extra.add((record.handle.slot, record.handle.generation))
    return base + sum(key in records and records[key].archived for key in extra
                      if not any(key in pairs for pairs in groups))
