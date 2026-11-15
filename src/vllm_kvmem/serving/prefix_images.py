"""Shared prefix images (docs/prefix-sharing.md).

A prefix image is the frozen state of an IDLE prompt checkpoint (kind P) at a
sealed position P: the archive payload of every block below P, each block's
retrieval index (MeanK sums and means), the frozen window frame, the per-layer
windows and score histories, the GDN recurrent and MTP draft state, the action
and graph snapshots and the token identity. Everything in it is immutable.

A new request whose session has no state and whose prompt starts with a
registered image (longest exact match, same cache_salt and model) gets an
ordinary new session B materialized from it: B's own authority, repository
slots, archive allocations and namespace, with the archive payload ``bytes``
objects aliased instead of copied. B then holds an IDLE frozen P checkpoint,
exactly the state a session that prefilled P itself reaches after its turn, and
the existing claim and restore path runs unchanged.

Flow. Registration: a request carrying ``kv_transfer_params.kvmem_share_prefix
= P`` (validated by SessionLedger.lookup) retires with its frozen P; the worker
captures the image (register_image) and reports ``image_registered``; the
scheduler indexes it (PrefixImageIndex). Import: SessionLedger.lookup matches a
new session against the index, holds a placeholder entry ('importing') and
sends SessionImport; the worker materializes B (import_image) and reports
``imported``; the ledger turns B idle and the request is admitted as an
ordinary session hit at P.

Payloads. On the CPU bytes archive the image aliases immutable ``bytes``
(phase 1). On the native payload arena (historical GC, the agent stack) the
image owns each payload as a shared-prefix pool owner, imports alias the
payload handles, and the parent's records move from its private CPU history
charge to the pool (KVMEM_PREFIX_POOL_BYTES); a payload is freed when its last
owner leaves.

Scope (scope_error): the CPU archive, sync or async restore, P checkpoints of
frozen and pm sessions, historical GC with a pool budget, single rank. Refused:
tensor parallel serving, legacy repository GC, Query window/FPS and
query-boundary caches, query history, recall-method Q windows and System One
forks. A refusal is a request-local error at registration and is checked again
at capture and import. Events: capture/prefix_images.jsonl.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import time
import uuid

from ..core import settings as _settings

DEFAULT_MAX_BYTES = 96 * 1024 ** 3


def max_image_bytes():
    """KVMEM_PREFIX_IMAGE_MAX_BYTES: LRU byte budget of the registered images."""
    value = _settings.get('KVMEM_PREFIX_IMAGE_MAX_BYTES', str(DEFAULT_MAX_BYTES))
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if parsed <= 0:
        raise ValueError('KVMEM_PREFIX_IMAGE_MAX_BYTES must be a positive integer')
    return parsed


class ImageRefused(ValueError):
    """A prefix image cannot be captured or imported in this configuration/state."""


def scope_error(options, *, tensor_parallel_size=1):
    """None when this serving configuration can share prefix images, else why not.

    Phase 2 covers the CPU archive with or without historical GC (the native
    payload arena needs a shared-prefix pool budget), P and M checkpoints,
    synchronous and asynchronous restore. Still refused: tensor parallel serving,
    Query window/FPS and query-boundary caches, recall-method Q windows and other
    per-request state that a prompt checkpoint does not carry.
    """
    reasons = []

    def need(condition, reason):
        if not condition:
            reasons.append(reason)
    need(getattr(options, 'session_capacity', 0) > 0, 'prompt sessions are off')
    need(getattr(options, 'request_planning', False), 'request planning is off')
    need(not getattr(options, 'repository_gc', False), 'legacy repository GC is on')
    need(getattr(options, 'archive_backend', None) == 'cpu', 'the archive backend is not the CPU archive')
    if getattr(options, 'historical_gc', None) is not None:
        from ..core.storage import prefix_pool_bytes
        need(prefix_pool_bytes() > 0, 'historical GC needs a shared-prefix pool (KVMEM_PREFIX_POOL_BYTES)')
    need(tensor_parallel_size == 1, 'tensor parallel serving')
    need(not getattr(options, 'query_history', False), 'query_history keeps state outside the checkpoint')
    need(getattr(options, 'query_score_mode', 'mean_k') == 'mean_k', 'a Query window/FPS cache is on')
    need(not getattr(options, 'query_fps_prebuild', False), 'query_fps_prebuild is on')
    need(not _settings.get('KVMEM_RECALL_METHOD'), 'KVMEM_RECALL_METHOD keeps tail-Q state outside the checkpoint')
    need(_settings.get('KVMEM_SYSTEMONE_FORK') != '1', 'System One forks are on')
    return '; '.join(reasons) or None


def archive_scope_error(archive):
    from ..core.memory_storage import MemoryArchive
    if not isinstance(archive, MemoryArchive):
        return 'the archive is not the CPU MemoryArchive'
    if getattr(archive, '_native_payload_arena', None) is not None:
        from ..core.storage import prefix_pool_bytes
        if not prefix_pool_bytes():
            return 'the native payload arena needs a shared-prefix pool (KVMEM_PREFIX_POOL_BYTES)'
    return None


def native(archive):
    return getattr(archive, '_native_payload_arena', None) is not None


def index_records(archive, count):
    """Private charge, in whole records, of ``count`` shared records' MeanK indexes.

    A shared record's payload (and one index copy, the image's) is charged once to
    the pool at the full per-record charge; every session holding the record keeps
    its own index, charged to that session as this fraction of the charge.
    """
    charge = getattr(archive, 'history_record_charge', archive.record_bytes)
    return -(-count * (charge - archive.record_bytes) // charge)


def model_identity(runner):
    config = getattr(runner, 'model_config', None) or getattr(getattr(runner, 'vllm_config', None),
                                                               'model_config', None)
    model = getattr(config, 'model', None)
    return '' if model is None else str(model)


@dataclass(frozen=True)
class ImageKey:
    model: str
    salt: str
    position: int
    sha256: str

    def label(self):
        return '%s@%d:%s' % (self.salt, self.position, self.sha256[:16])


@dataclass(frozen=True, eq=False)
class ImageBlock:
    begin: int
    count: int
    # Bytes archive: the record's immutable bytes (aliased, never copied).
    # Native arena: the payload handle the image owns as a pool owner.
    payload: object
    key_sum: object         # sealed MeanK sums (bytes archive: shared; native: a private copy)
    mean_key: object
    digest: object
    thinking: object = None  # (thinking, known) tokens published for the record, or None


@dataclass(frozen=True, eq=False)
class PrefixImage:
    key: ImageKey
    tokens: bytes            # token identity of tokens[:P] (prompt_checkpoints.token_bytes)
    blocks: tuple            # ImageBlock in history order, covering [0, P)
    window: tuple            # (begin, count, dense) of the frozen frame, dense order
    layerwise: object        # LayerwisePrefix with handle-free maps, or None
    model: object            # PromptModelState: immutable pinned CPU GDN/draft payload
    guided_seed: object
    action_history: object   # private copy of the checkpoint's action snapshot
    graph_data: object       # immutable BlockGraph data at P, or None without a graph
    bytes: int
    source: tuple            # (namespace, generation, request id) of the parent turn
    native: bool = False     # payloads are pool-owned native arena handles

    @property
    def position(self):
        return self.key.position


@dataclass(frozen=True)
class SessionImport:
    namespace: tuple
    generation: int
    key: ImageKey


@dataclass(frozen=True)
class ImageNotice:
    kind: str                # image_registered, image_refused or image_released
    namespace: tuple
    generation: int
    request_id: str
    history: int = 0
    key: ImageKey | None = None
    tokens: bytes = b''
    bytes: int = 0
    reason: str | None = None
    # Native arena: records moved from the parent's private charge to the
    # shared-prefix pool, and the pool's record count after the registration.
    transferred: int = 0
    pool_records: int = 0


@dataclass(frozen=True)
class HistoryPoolNotice:
    """The worker's shared-prefix pool: distinct payloads charged to it, and how
    many of them have lost their last owner but still hold bytes."""
    kind: str = 'history_pool'
    namespace: tuple = ()
    generation: int = 0
    request_id: str = ''
    records: int = 0
    draining: int = 0


# --- scheduler: image index ------------------------------------------------

@dataclass
class IndexedImage:
    key: ImageKey
    tokens: bytes
    bytes: int
    owner: tuple             # (namespace, generation) of the registering session
    ledger: dict             # continuation state the parent's idle ledger entry had at P
    refs: int = 0            # imports in flight
    released: bool = False   # no new matches; dropped when refs reaches 0
    importers: set = field(default_factory=set)   # live sessions imported from this image


class PrefixImageIndex:
    """Scheduler-side registry: matching, in-flight references and the LRU budget.

    The worker holds the payload; it drops an image only on a release from here.
    """
    LEDGER_FIELDS = ('query', 'mandatory', 'query_id', 'guided_anchor', 'prompt_guided_anchor')

    def __init__(self, model, max_bytes):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('Prefix image budget must be a positive byte count')
        self.model, self.max_bytes = '' if model is None else str(model), max_bytes
        self.images = OrderedDict()      # least recently used first
        self.entries = {}                # the session ledger's entries (set by the ledger)
        self.used_bytes = 0
        self.registered = self.matched = self.evicted = 0

    def match(self, salt, prompt):
        """The registered image with the longest exact token prefix of ``prompt``."""
        from .prompt_checkpoints import token_bytes
        best, identities = None, {}
        for image in self.images.values():
            key = image.key
            if (image.released or key.salt != salt or key.model != self.model or
                    len(prompt) <= key.position or (best is not None and key.position <= best.key.position)):
                continue
            identity = identities.get(key.position)
            if identity is None:
                identity = identities[key.position] = token_bytes(prompt, key.position)
            if identity == image.tokens:
                best = image
        if best is not None:
            self.images.move_to_end(best.key)
            self.matched += 1
        return best

    def register(self, notice, entry):
        """Index a worker image; returns the keys the worker must drop."""
        key = notice.key
        if key.model != self.model:
            return [key]
        if key in self.images:
            self.images.move_to_end(key)
            return []
        self.images[key] = IndexedImage(key, notice.tokens, notice.bytes, (notice.namespace, notice.generation),
            {name: getattr(entry, name) for name in self.LEDGER_FIELDS})
        self.used_bytes += notice.bytes
        self.registered += 1
        return self._over_budget()

    def acquire(self, key):
        self.images[key].refs += 1

    def release_ref(self, key):
        image = self.images[key]
        if image.refs <= 0:
            raise ValueError('Prefix image reference released twice')
        image.refs -= 1
        if image.released and not image.refs:
            return [self._drop(key)]
        return self._over_budget()

    def release_owner(self, namespace):
        """An explicit session close releases the images that session registered."""
        dropped = []
        for key, image in list(self.images.items()):
            if image.owner[0] == namespace and not image.released:
                image.released = True
                if not image.refs:
                    dropped.append(self._drop(key))
        return dropped

    def imported(self, key, namespace):
        image = self.images.get(key)
        if image is not None:
            image.importers.add(namespace)

    def forget_namespace(self, namespace):
        """A session left (evicted or retired empty): it no longer references images."""
        for image in self.images.values():
            image.importers.discard(namespace)

    def references(self, image, entries):
        """Live sessions holding the image's payloads: its importers and its owner."""
        owner, generation = image.owner
        entry = entries.get(owner)
        return len(image.importers) + int(entry is not None and entry.generation == generation)

    def references_entry(self, entry):
        """Whether this live session keeps a registered image referenced (owner or importer)."""
        namespace, generation = entry.namespace, entry.generation
        return any(namespace in image.importers or image.owner == (namespace, generation)
                   for image in self.images.values())

    def unreferenced(self, entries):
        """Release every image no session references: its payloads then have no
        other owner, so the release frees their bytes (once they drain)."""
        dropped = []
        for key, image in list(self.images.items()):
            if not image.refs and not image.released and not self.references(image, entries):
                dropped.append(self._drop(key))
        return dropped

    def _over_budget(self):
        """Budget eviction: least referenced first, then least recently used. An
        image with an import in flight stays."""
        dropped = []
        if self.used_bytes <= self.max_bytes:
            return dropped
        entries = self.entries
        order = sorted(enumerate(self.images.items()),
                       key=lambda item: (self.references(item[1][1], entries), item[0]))
        for _, (key, image) in order:
            if self.used_bytes <= self.max_bytes:
                break
            if not image.refs:
                dropped.append(self._drop(key))
        return dropped

    def _drop(self, key):
        image = self.images.pop(key)
        self.used_bytes -= image.bytes
        self.evicted += 1
        return key

    def snapshot(self):
        return dict(images=len(self.images), used_bytes=self.used_bytes, max_bytes=self.max_bytes,
                    registered=self.registered, matched=self.matched, evicted=self.evicted,
                    in_flight=sum(image.refs for image in self.images.values()))


# --- worker: events and store ----------------------------------------------

def emit(worker, event, **fields):
    """One row of capture/prefix_images.jsonl (when the runner has a probe output)."""
    counts = worker.__dict__.setdefault('prefix_image_counts', {})
    counts[event] = counts.get(event, 0) + 1
    output = getattr(getattr(worker, 'runner', None), '_kvmem_probe_output', None)
    if output is None:
        return
    row = dict(event=event, at=time.time(), **fields)
    with (output / 'prefix_images.jsonl').open('a') as stream:
        stream.write(json.dumps(row, default=str) + '\n')


def images_of(worker):
    return worker.__dict__.setdefault('prefix_images', {})


def _tensor_bytes(tensor):
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _rebuilt(prefix, maps, gc_hold=None):
    """A LayerwisePrefix with new maps. replace() leaves the cached handle pairs of
    the parent's repository (_map_pairs) behind; the handle-free sortedness hint
    of its decode rows is kept."""
    result = replace(prefix, maps=maps, gc_hold=gc_hold)
    if getattr(prefix, '__dict__', {}).get('_decode_sorted'):
        object.__setattr__(result, '_decode_sorted', True)
    return result


def _without_handles(prefix):
    """The parent's per-layer snapshot without its repository handles and without
    its historical GC hold (that hold stays with the parent's own checkpoint)."""
    if prefix is None:
        return None
    return _rebuilt(prefix, tuple(tuple((-1, -1, begin, count, dense) for _, _, begin, count, dense in rows)
                                  for rows in prefix.maps))


def _with_handles(prefix, handles):
    if prefix is None:
        return None

    def row(item):
        _, _, begin, count, dense = item
        handle = handles.get(begin)
        if handle is None:
            raise ValueError('Prefix image layer map names a block outside the image')
        return handle.slot, handle.generation, begin, count, dense
    return _rebuilt(prefix, tuple(tuple(row(item) for item in rows) for rows in prefix.maps))


def registration_checkpoint(worker, binding, retired):
    """The idle P checkpoint at kvmem_share_prefix: a terminal (pm) session keeps it
    in pool.prompts beside its M; a frozen session keeps it as its entry."""
    position = binding.share_prefix
    for candidate in (worker.pool.prompts.get(binding.namespace), retired):
        if candidate is not None and getattr(candidate, 'kind', None) == 'P' and candidate.position == position:
            return candidate
    return retired


def image_key(worker, binding, checkpoint):
    return ImageKey(model_identity(worker.runner), binding.namespace[0], binding.share_prefix,
                    hashlib.sha256(checkpoint.tokens).hexdigest())


def capture_image(worker, binding, checkpoint):
    """The prefix image of an IDLE P checkpoint (worker, engine thread).

    Returns the image and, on the native arena, how many of the parent's records
    moved from its private charge to the shared-prefix pool.
    """
    from .prompt_checkpoints import CheckpointPhase
    from .tensor_parallel import active
    position = binding.share_prefix
    reason = (scope_error(worker.options, tensor_parallel_size=2 if active() else 1) or
              archive_scope_error(getattr(checkpoint.authority, 'archive', None)))
    if reason is not None:
        raise ImageRefused(reason)
    if checkpoint.kind != 'P':
        raise ImageRefused('the retained checkpoint is not a prompt (P) checkpoint')
    if checkpoint.position != position:
        raise ImageRefused('kvmem_share_prefix %d differs from the prompt checkpoint position %d'
                           % (position, checkpoint.position))
    if checkpoint.phase is not CheckpointPhase.IDLE or not worker.pool.model_ready(checkpoint):
        raise ImageRefused('the prompt checkpoint is not idle')
    if checkpoint.claim_action_history is not None or checkpoint.claim_graph_history is not None:
        raise ImageRefused('the prompt checkpoint carries claim-only evidence')
    authority = checkpoint.authority
    historical = getattr(authority.gc, 'native_history', False)
    if authority.gc is not None and not historical:
        raise ImageRefused('the authority uses legacy repository GC')
    if authority.history_end < position or authority.writes or authority._unsubmitted or authority.discarded_records:
        raise ImageRefused('the authority has unpublished or rewound records')
    block = authority.shape[1]
    records = sorted((r for r in authority.records.values() if r.history_begin < position),
                     key=lambda record: record.history_begin)
    if ([(r.history_begin, r.count) for r in records] != [(i * block, block) for i in range(position // block)]
            or not all(r.archived for r in records)):
        raise ImageRefused('the authority does not hold every sealed block below P')
    archive = authority.archive
    arena = native(archive)
    if arena:
        from ..core.storage import prefix_pool_bytes
        budget = prefix_pool_bytes() // getattr(archive, 'history_record_charge', archive.record_bytes)
        if archive.pool_records() + len(records) > budget:
            raise ImageRefused('the shared-prefix pool budget (KVMEM_PREFIX_POOL_BYTES) is full')
    repository = authority.memory.repository
    leases = [repository.pin(record.handle) for record in records]
    transferred = 0
    try:
        if not all(leases):
            raise ImageRefused('a prefix block was retired')
        if arena:
            # The image becomes a pool owner of every payload; the parent's records
            # move to the pool charge and its GC scope marks them shared.
            payloads, transferred = archive.share_to_pool(authority.memory.request_id, leases)
            authority.__dict__.setdefault('_prefix_shared_keys', set()).update(
                (r.handle.slot, r.handle.generation) for r in records)
        else:
            payloads = archive.record_payloads(authority.memory.request_id, leases)
    finally:
        for lease in leases:
            lease.reset()
    try:
        # Index tensors of a native-history record live in its session's index
        # owners: copy them, so no owner outlives (or blocks) its session.
        def index(tensor):
            return tensor.clone() if historical and tensor is not None else tensor
        blocks = tuple(ImageBlock(r.history_begin, r.count, payload, index(r.key_sum), index(r.mean_key), r.digest,
                                  getattr(r, '_thinking_published', None))
                       for r, payload in zip(records, payloads))
        window = tuple((begin, count, dense) for _, _, begin, count, dense in checkpoint.window.records)
        layerwise = _without_handles(checkpoint.layerwise_pressure)
        graph = checkpoint.graph_history
        model = checkpoint.model
        payload_bytes = archive.record_bytes if arena else None
        size = (sum((payload_bytes or len(b.payload)) + _tensor_bytes(b.key_sum) + _tensor_bytes(b.mean_key)
                    for b in blocks) +
                model.bytes + len(checkpoint.tokens) + (layerwise.bytes if layerwise is not None else 0) +
                (graph.bytes if graph is not None else 0))
        image = PrefixImage(image_key(worker, binding, checkpoint), checkpoint.tokens, blocks, window, layerwise,
                            model, checkpoint.guided_seed, deepcopy(checkpoint.action_history),
                            graph._data if graph is not None else None, size,
                            (binding.namespace, binding.generation, binding.request_id), arena)
    except BaseException:
        if arena:
            archive.release_pool(payloads)
        raise
    return image, transferred


def register_image(worker, binding, checkpoint):
    """After a kvmem_share_prefix request retired with its P checkpoint (worker)."""
    started = time.perf_counter()
    common = dict(namespace=list(binding.namespace), generation=binding.generation,
                  request_id=binding.request_id, position=binding.share_prefix)
    store = images_of(worker)
    try:
        checkpoint = registration_checkpoint(worker, binding, checkpoint)
        key = image_key(worker, binding, checkpoint)
        if key in store:
            # Already registered; the scheduler knows it. No second payload or pool owner.
            emit(worker, 'duplicate', key=key.label(), **common)
            return store[key]
        image, transferred = capture_image(worker, binding, checkpoint)
    except Exception as error:
        # A refusal (or a defect) leaves the checkpoint and the retirement
        # untouched; later sessions just prefill cold.
        reason = str(error) if isinstance(error, ImageRefused) else '%s: %s' % (type(error).__name__, error)
        emit(worker, 'refused', reason=reason, **common)
        worker.completed.append(ImageNotice('image_refused', binding.namespace, binding.generation,
                                            binding.request_id, binding.share_prefix, reason=reason))
        return None
    store[image.key] = image
    archive = checkpoint.authority.archive
    pool = archive.pool_records() if image.native else 0
    # The parent keeps its own index of the transferred records privately.
    transferred -= index_records(archive, transferred) if transferred else 0
    emit(worker, 'registered', key=image.key.label(), blocks=len(image.blocks), window_tokens=sum(
        r[1] for r in image.window), bytes=image.bytes, images=len(store), native=image.native,
        transferred=transferred, pool_records=pool, seconds=time.perf_counter() - started, **common)
    worker.completed.append(ImageNotice('image_registered', binding.namespace, binding.generation,
        binding.request_id, image.position, image.key, image.tokens, image.bytes,
        transferred=transferred, pool_records=pool))
    return image


def release_images(worker, keys):
    store = images_of(worker)
    for key in keys:
        image = store.pop(key, None)
        # Every release is acknowledged, also for an image this worker no longer
        # holds: the scheduler counts nothing freed until it sees the notice.
        worker.completed.append(ImageNotice('image_released', (), 0, '', key.position, key))
        if image is None:
            continue
        if image.native:
            # The image's pool owner of every payload leaves; payloads no session
            # still owns lose their bytes once readers drain.
            worker.runtime.archive.release_pool([block.payload for block in image.blocks])
        emit(worker, 'evicted', key=key.label(), bytes=image.bytes, images=len(store))


# --- worker: import --------------------------------------------------------

def _index_copy(authority, item):
    """A native-history record's MeanK index in the importing session's own owners."""
    shape = (authority.scoring_layers, authority.shape[2], authority.shape[3])
    mean = authority.mean_k_backend == 'qw3_cuda'
    sums, owner = authority.archive.allocate_index(shape, mean=mean, split_sums=authority.cpu_release_sealed_sums)
    key_sum = mean_key = None
    if item.key_sum is not None:
        sums.copy_(item.key_sum)
        key_sum = sums
    if mean and item.mean_key is not None:
        import torch
        mean_key = (owner.mean_view() if authority.cpu_release_sealed_sums else
                    torch.frombuffer(owner, dtype=torch.float16, count=sums.numel(),
                                     offset=sums.numel() * 4).view(sums.shape))
        mean_key.copy_(item.mean_key)
    if authority.cpu_release_sealed_sums:
        key_sum = None
        owner.release_sums()
    return key_sum, mean_key, owner


def _materialize(authority, image):
    """B's repository slots, store directory, aliased archive records and index."""
    from .. import _native
    from .authority import Record, record_key
    memory = authority.memory
    store, repository, archive = memory._store, memory.repository, authority.archive
    if memory.history_end or len(authority.records):
        raise ValueError('Prefix image import requires a fresh authority')
    if image.native != native(archive):
        raise ImageRefused('the image and this archive disagree on native payloads')
    historical = getattr(authority.gc, 'native_history', False)
    frame = {begin: dense for begin, _, dense in image.window}
    handles = {}
    # Current-window protection stays as the authority was built (serving sets it
    # before the first append). As in a session that prefilled P itself, only the
    # frozen frame's blocks are in the protected current frame, the others not.
    for item in image.blocks:
        reservation = repository.reserve(item.begin, item.count)
        if not reservation:
            raise BufferError('Prefix image exceeds the new repository capacity')
        lease = reservation.commit()
        handle = _native.Handle(lease.handle.slot, lease.handle.generation)
        # Frame blocks are baked at their dense window position, the others at
        # their history position; a restore rebuilds every page from raw K.
        store.register_block(lease, frame.get(item.begin, item.begin), item.begin in frame)
        handles[item.begin] = handle
    if memory.history_end != image.position:
        raise RuntimeError('Imported store history differs from its prefix image')
    for begin, handle in handles.items():
        if begin not in frame:
            store.working_resident(handle, False)      # only the frozen frame is current
    pins = [repository.pin(handles[item.begin]) for item in image.blocks]
    try:
        if not all(pins):
            raise RuntimeError('Imported repository generation is not live')
        # Native arena: descriptors sharing the image's payloads, published to this
        # session's GC scope and marked shared there. Bytes archive: aliased bytes.
        if not archive.alias_records(memory.request_id, pins, [item.payload for item in image.blocks]):
            raise BufferError('The CPU archive has too few free records for the prefix image')
        if image.native:
            # Every imported record is shared: this session holds only its index.
            authority._prefix_shared_keys = {(h.slot, h.generation) for h in handles.values()}
    finally:
        for pin in pins:
            pin.reset()
    flags = _native.GC_RAW_BACKED | _native.GC_VALUE_BACKED | _native.GC_INDEXED
    scope = getattr(authority.gc, 'scope', None)
    thinking = getattr(authority, 'thinking_history', None)
    redundancy = getattr(authority, 'redundancy_history', None)
    for item in image.blocks:
        handle = handles[item.begin]
        if historical:
            key_sum, mean_key, owner = _index_copy(authority, item)
        else:
            key_sum, mean_key, owner = item.key_sum, item.mean_key, None
        record = Record(handle, item.begin, item.count, None, None, key_sum,
                        archived=True, submitted=True, digest=item.digest)
        record.mean_key = mean_key
        record.index_owner = owner
        authority.records[record_key(handle)] = record
        if not historical:
            store.gc_flags(handle, flags, flags)    # as RequestAuthority.poll after a completed write
        if redundancy is not None:
            redundancy.publish_key(record, repository)
        if thinking is not None and scope is not None and item.thinking is not None:
            lease = repository.pin(handle)
            try:
                if lease and scope.publish_thinking(lease, *item.thinking):
                    record._thinking_counts = None
                    record._thinking_published = tuple(item.thinking)
            finally:
                lease.reset()
    authority.archived_epoch += len(image.blocks)
    return handles


def _payload_bytes(identity, window, seed, action, layerwise, graph):
    """PromptCheckpointPool._capture_snapshots' fixed charge, for an imported checkpoint."""
    fixed = len(identity) + len(window.records) * 5 * 8
    if seed is not None:
        fixed += seed.bytes
    if action is not None:
        fixed += len(action[0]) * 16 + 8
        if len(action) > 2:
            fixed += len(action[2][0]) * 24 + 24
    if layerwise is not None:
        fixed += layerwise.bytes
    if graph is not None:
        fixed += graph.bytes
    return fixed


def import_image(worker, image, namespace, generation):
    """Materialize session ``namespace`` from ``image`` as an IDLE frozen P (worker).

    Returns the installed checkpoint and host timings. On failure nothing stays
    installed and the partial authority is closed.
    """
    from .prompt_checkpoints import CheckpointPhase, PromptCheckpoint
    from .session_control import SessionBinding
    from .tensor_parallel import active
    timings, started = {}, time.perf_counter()
    runtime = worker.runtime
    if runtime is None:
        raise ImageRefused('the capture runtime is not initialized')
    reason = (scope_error(worker.options, tensor_parallel_size=2 if active() else 1) or
              archive_scope_error(runtime.archive))
    if reason is not None:
        raise ImageRefused(reason)
    if (namespace in worker.pool.entries or namespace in worker.pool.prompts or
            namespace in worker.identities):
        raise ValueError('Prefix image import into a live session namespace')
    adapter = worker.model_adapter
    if adapter is None or getattr(image.model, 'owner', None) is not getattr(adapter, '_identity', None):
        raise ImageRefused('the image model state belongs to another worker or model load')
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    if (controller is None) != (image.layerwise is None):
        raise ImageRefused('the image and this worker disagree on per-layer pressure')
    authority_id = 'kvmem-prefix-image-' + uuid.uuid4().hex
    authority = runtime.build_authority(authority_id, SessionBinding(namespace, generation, authority_id, ()),
                                        graph_metadata=False)
    timings['authority'] = time.perf_counter() - started
    window = graph_history = checkpoint = None
    try:
        mark = time.perf_counter()
        handles = _materialize(authority, image)
        timings['materialize'] = time.perf_counter() - mark
        mark = time.perf_counter()
        native = getattr(worker.pool, 'native_checkpoint_capture', False)
        window = authority.memory.checkpoint_window(**({'native_validation': True} if native else {}))
        expected = tuple((handles[begin].slot, handles[begin].generation, begin, count, dense)
                         for begin, count, dense in image.window)
        if tuple(window.records) != expected:
            raise RuntimeError('Imported frame differs from the prefix image window')
        graph = getattr(authority, 'block_graph', None)
        if (graph is None) != (image.graph_data is None):
            raise ImageRefused('the image and this worker disagree on the block graph')
        if (getattr(authority, 'action_history', None) is None) != (image.action_history is None):
            raise ImageRefused('the image and this worker disagree on action attention')
        if graph is not None:
            from .block_graph import GraphSnapshot
            graph_history = GraphSnapshot(graph, data=image.graph_data)
        layerwise = _with_handles(image.layerwise, handles)
        from .layerwise_gc_holds import checkpoint_hold, scope_of
        if layerwise is not None and scope_of(authority) is not None:
            # This session's own policy hold on every per-layer window record, as
            # a capture takes (layerwise_checkpoints._capture).
            from .layerwise_checkpoints import _same_identity
            shared = tuple(window.records)
            held = layerwise.maps + (() if _same_identity(shared, layerwise.maps[0]) else (shared,))
            layerwise = _rebuilt(layerwise, layerwise.maps, checkpoint_hold(authority, held))
        fixed = _payload_bytes(image.tokens, window, image.guided_seed, image.action_history, layerwise,
                               graph_history)
        checkpoint = PromptCheckpoint(namespace, authority_id, authority, window, image.model,
                                      image.tokens, fixed + image.model.bytes)
        checkpoint.kind = 'P'
        checkpoint.action_history = image.action_history
        checkpoint.graph_history = graph_history
        checkpoint.guided_seed = image.guided_seed
        checkpoint.layerwise_pressure = layerwise
        checkpoint.phase = CheckpointPhase.IDLE
        worker.pool.install_imported(checkpoint)
        timings['checkpoint'] = time.perf_counter() - mark
    except BaseException:
        if checkpoint is not None:
            checkpoint.window = checkpoint.model = checkpoint.authority = None
            checkpoint.graph_history = checkpoint.layerwise_pressure = None
        checkpoint = window = None          # drop the frame's repository pins before close
        if graph_history is not None:
            graph_history.close()
        if not authority.close():       # closes the graph; retried at the next import command
            worker.__dict__.setdefault('prefix_image_cleanup', []).append(authority)
        raise
    worker.identities[namespace] = generation
    timings['total'] = time.perf_counter() - started
    return checkpoint, timings


def import_commands(worker, commands):
    """SessionImport commands (SessionWorker.receive): one notice each."""
    from .session_control import SessionNotice
    pending = worker.__dict__.get('prefix_image_cleanup')
    if pending:
        worker.prefix_image_cleanup = [authority for authority in pending if not authority.close()]
    for command in commands:
        image = images_of(worker).get(command.key)
        common = dict(namespace=list(command.namespace), generation=command.generation, key=command.key.label())
        try:
            if image is None:
                raise ImageRefused('the prefix image is no longer registered on this worker')
            checkpoint, timings = import_image(worker, image, command.namespace, command.generation)
        except Exception as error:
            # import_image installs nothing on failure; the request prefills cold.
            reason = (str(error) if isinstance(error, (ImageRefused, BufferError)) else
                      '%s: %s' % (type(error).__name__, error))
            emit(worker, 'import_failed', reason=reason, **common)
            worker.completed.append(SessionNotice('import_failed', command.namespace, command.generation, ''))
            continue
        authority = checkpoint.authority
        allocated = protected = 0
        if getattr(authority.gc, 'native_history', False):
            # Private history charge, as an idle session reports it: its aliases
            # are pool records; only their copied index is private.
            allocated, protected = worker._history_rows(checkpoint)
        emit(worker, 'imported', position=checkpoint.position, window_tokens=checkpoint.window.window_tokens,
             blocks=len(image.blocks), authority=authority.memory.request_id, seconds=timings,
             allocated_records=allocated, **common)
        worker.completed.append(SessionNotice('imported', command.namespace, command.generation, '',
            checkpoint.position, checkpoint.window.window_tokens, authority.memory.repository.ready_slots,
            active_records=tuple((r[2], r[3]) for r in checkpoint.window.records),
            allocated_records=allocated, protected_records=protected))
        # The idle-session bookkeeping a retirement sets up (SessionWorker.retirement_done):
        # idle history reports (scheduler reclaim requests and GC progress reach the
        # scheduler as history_capacity notices) and session tiering.
        if getattr(authority.gc, 'native_history', False):
            worker._history_capacity_sent[command.namespace] = (command.generation, checkpoint.position,
                authority.memory.repository.ready_slots, allocated, protected)
            if command.namespace not in worker._history_idle_queued:
                worker._history_idle.append(command.namespace)
                worker._history_idle_queued.add(command.namespace)
        tiering = worker.session_tiering()
        if tiering is not None:
            tiering.idle(command.namespace, authority)
