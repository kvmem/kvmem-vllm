"""CPU scheduling oracles for bounded assembly scratch and its CUDA fences."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm_kvmem import window_assembly as window
from vllm_kvmem.memory import Phase


class Event:
    def __init__(self):
        self.complete = False
        self.stream = None

    def record(self, stream):
        self.stream = stream
        stream.operations.append(("record", self))

    def query(self):
        return self.complete


class Buffer:
    def __init__(self, size, *, parent=None):
        self.size, self.parent = size, parent

    def __getitem__(self, extent):
        return Buffer(extent.stop, parent=self)


@pytest.fixture
def cuda_oracle(monkeypatch):
    stream = SimpleNamespace(operations=[])
    stream.wait_event = lambda event: stream.operations.append(("wait", event))
    allocations, pools = [], []

    def allocate(size, **kwargs):
        result = Buffer(size)
        allocations.append((result, kwargs))
        return result

    class Transfers:
        def __init__(self, slot_bytes, slots, device):
            self.slot_bytes, self.slots, self.device = slot_bytes, slots, device
            self.pending = {}
            self.closed = False
            pools.append(self)

        def close(self):
            if any(not txn.event.query() for txn in self.pending.values()):
                return False
            self.closed = True
            self.pending.clear()
            return True

    monkeypatch.setattr(window, "PackedTransfers", Transfers)
    monkeypatch.setattr(window.torch.cuda, "Event", Event)
    monkeypatch.setattr(window.torch.cuda, "current_stream", lambda device: stream)
    return SimpleNamespace(stream=stream, pools=pools, allocations=allocations,
                           allocate=allocate, monkeypatch=monkeypatch)


def test_two_gpu_slots_wait_for_their_consumer_not_the_previous_chunk(cuda_oracle):
    cuda_oracle.monkeypatch.setattr(window.torch, "empty", cuda_oracle.allocate)
    pool = window._AssemblyStaging(64, torch.device("cuda:0"))
    owner = object()
    pool.claim(owner)
    first, allocated, first_slot = pool.destination(32)
    pool.consumed(first_slot)
    first_consumer = pool.ready[first_slot]
    second, second_ready, second_slot = pool.destination(16)
    # The other device buffer can copy while the first buffer is consumed.
    assert first_slot != second_slot and second_ready is allocated
    assert second_ready is not first_consumer
    pool.consumed(second_slot)
    third, third_ready, third_slot = pool.destination(64)
    # Reusing device storage waits for its kernel readers, not just its DMA.
    assert third_slot == first_slot and third.parent is first.parent
    assert third_ready is first_consumer
    assert second.parent is not first.parent
    assert len(cuda_oracle.allocations) == 2
    assert sum(buffer.size for buffer, _ in cuda_oracle.allocations) == 128
    assert len(cuda_oracle.pools) == 1
    pool.release(owner)
    assert not pool.close()
    for event in pool.ready:
        event.complete = True
    assert pool.close() and not pool.buffers


def test_staging_has_one_cpu_owner_and_never_allocates_for_an_invalid_extent(cuda_oracle):
    pool = window._AssemblyStaging(64, torch.device("cuda:0"))
    owner = object()
    with pytest.raises(ValueError, match="owned bounded"):
        pool.destination(16)
    pool.claim(owner)
    assert not pool.close()
    with pytest.raises(RuntimeError, match="CPU producer"):
        pool.claim(object())
    with pytest.raises(RuntimeError, match="Stale"):
        pool.release(object())
    for size in (0, -1, 65):
        with pytest.raises(ValueError, match="owned bounded"):
            pool.destination(size)
    assert pool.owner is owner and not cuda_oracle.pools
    pool.release(owner)
    assert pool.close()


def test_failed_partial_consumer_fences_both_slots_before_the_next_request(cuda_oracle):
    cuda_oracle.monkeypatch.setattr(window.torch, "empty", cuda_oracle.allocate)
    pool = window._AssemblyStaging(64, torch.device("cuda:0"))
    owner = object()
    pool.claim(owner)
    pool.destination(32)
    failed = Event()
    pool.release(owner, failed)
    assert not pool.close()
    next_owner = object()
    pool.claim(next_owner)
    _, ready, _ = pool.destination(32)
    assert ready is failed
    pool.release(next_owner)
    failed.complete = True
    assert pool.close()


def test_resident_value_scratch_is_shared_bounded_and_failure_fenced(cuda_oracle):
    cuda_oracle.monkeypatch.setattr(window.torch,'empty',cuda_oracle.allocate)
    pool=window._AssemblyStaging(64,torch.device('cuda:0'),value_capacity_bytes=96)
    first=object()
    pool.claim(first)
    value=pool.value_destination(64)
    done=Event()
    pool.value_ready=done
    pool.release(first)
    assert not pool.close()
    second=object()
    pool.claim(second)
    next_value=pool.value_destination(32)
    assert value.parent is next_value.parent and len(cuda_oracle.allocations)==1
    assert cuda_oracle.allocations[0][0].size==96
    assert cuda_oracle.stream.operations[-1]==('wait',done)
    with pytest.raises(ValueError,match='byte budget'):pool.value_destination(97)
    failed=Event()
    pool.release(second,failed)
    assert pool.value_ready is failed and not pool.close()
    failed.complete=True
    assert pool.close() and pool.value_buffer is None


def cpu_layer():
    key = torch.zeros((4, 4, 1, 4), dtype=torch.float8_e4m3fn)
    return window.LayerWindow(key, key.clone(), torch.zeros(16, 4), torch.tensor(1.0),
                              torch.tensor([1, 2], dtype=torch.int32))


def test_all_skipped_window_does_not_allocate_staging_or_metadata(cuda_oracle, monkeypatch):
    assembler = window.NativeWindowAssembler((cpu_layer(),), 4, packed=True,
                                              direct_pinned_reads=True)
    plan = SimpleNamespace(remaps=[SimpleNamespace(block_id=1, n_tokens=4, to_base=0, skip=True)])
    memory = SimpleNamespace(phase=Phase.ASSEMBLING, _plan=plan)
    record = SimpleNamespace(handle=SimpleNamespace(slot=1), count=4)
    authority = SimpleNamespace(memory=memory, shape=(1, 4, 1, 4), records={1: record},raw_dtype=torch.bfloat16)
    monkeypatch.setattr(assembler, "_upload_metadata", lambda _: pytest.fail("all-skip metadata"))
    result = assembler.assemble(authority, plan)
    assert (result.rebuilt_tokens, result.skipped_tokens, result.chunks) == (0, 4, 0)
    assert not cuda_oracle.pools and not assembler.staging.buffers
    assert not assembler.close()
    result.event.complete = True
    assert assembler.close()


def test_assembly_failure_retains_partial_dma_and_kernel_owners(cuda_oracle, monkeypatch):
    assembler = window.NativeWindowAssembler((cpu_layer(),), 4)
    monkeypatch.setattr(window.torch, "empty", cuda_oracle.allocate)
    plan = object()
    memory = SimpleNamespace(phase=Phase.ASSEMBLING, _plan=plan)
    memory.fail_transaction = lambda result: setattr(memory, "fence", result)
    authority = SimpleNamespace(memory=memory)
    dma = Event()

    def fail(*_):
        assembler.staging.destination(16)
        assembler.transfers.pending[0] = SimpleNamespace(event=dma)
        raise OSError("partial assembly")

    monkeypatch.setattr(assembler, "_assemble", fail)
    with pytest.raises(OSError, match="partial assembly"):
        assembler.assemble(authority, plan)
    fence = memory.fence
    assert fence.owners == (assembler, plan, authority)
    assert cuda_oracle.stream.operations[-2:] == [("wait", dma), ("record", fence.event)]
    assert all(event is fence.event for event in assembler.staging.ready)
    assert assembler.staging.owner is None
    assert not assembler.close()
    fence.event.complete = True
    assert not assembler.close()  # The transfer pool must also finish its DMA.
    dma.complete = True
    assert assembler.close()


@pytest.mark.parametrize('fail_tail', [False, True])
def test_hybrid_tail_uses_final_fence_and_invalidates_cache_after_failure(cuda_oracle, monkeypatch, fail_tail):
    cache_events, dropped = [], []
    cache = SimpleNamespace(finish=cache_events.append, drop_namespace=dropped.append,dtype=torch.bfloat16)
    assembler = window.NativeWindowAssembler((cpu_layer(),), 4, packed=True,
        direct_pinned_reads=True, raw_cache=cache)
    remaps = [SimpleNamespace(block_id=i, n_tokens=n, to_base=i*4, skip=False)
              for i,n in enumerate((4,3))]
    plan = SimpleNamespace(remaps=remaps)
    memory = SimpleNamespace(phase=Phase.ASSEMBLING, _plan=plan, request_id='owner')
    memory.fail_transaction = lambda fence:setattr(memory, 'failed', fence)
    authority = SimpleNamespace(memory=memory, shape=(1,4,1,4), raw_dtype=torch.bfloat16,
        records={i:SimpleNamespace(handle=SimpleNamespace(slot=i),count=r.n_tokens)
                 for i,r in enumerate(remaps)})
    full_event, tail_event = Event(), Event()
    full_event.complete = True
    def full(a,p,rs,pending,records):
        assert pending == [remaps[0]]
        return window.AssemblyResult(full_event, (cache,), 4, 0, 1,
            uploaded_bytes=16, raw_cache_hit_tokens=4, split_tokens=4)
    def tail(a,p,rs,pending,records):
        assert not rs and pending == [remaps[1]]
        if fail_tail:
            raise OSError('tail read failed after full-page writes')
        return window.AssemblyResult(tail_event, (authority,), 3, 0, 1,
            uploaded_bytes=36, partial_tokens=3)
    monkeypatch.setattr(assembler, '_assemble_split', full)
    monkeypatch.setattr(assembler, '_assemble_generic', tail)
    if fail_tail:
        with pytest.raises(OSError, match='tail read'):
            assembler.assemble(authority,plan)
        assert dropped == ['owner'] and cache_events == [memory.failed.event]
        assert not assembler.close()
    else:
        result = assembler.assemble(authority,plan)
        assert result.event is tail_event and not result.query() and not assembler.close()
        assert (result.rebuilt_tokens,result.uploaded_bytes,result.split_tokens,result.partial_tokens)==(7,52,4,3)
        assert result.raw_cache_hit_tokens == 4 and authority in result.owners and cache in result.owners
        tail_event.complete = True
        assert assembler.close()
    assert assembler.staging.owner is None


def test_runner_reuses_fixed_staging_across_request_views(cuda_oracle, monkeypatch):
    import numpy as np

    layer = cpu_layer()
    module = SimpleNamespace(num_kv_heads=1, head_dim=4,
        rotary_emb=SimpleNamespace(cos_sin_cache=layer.cos_sin),
        attn=SimpleNamespace(kv_cache_dtype="fp8_e4m3", layer_name="attention",
                             kv_cache=object(), _k_scale=layer.key_scale))
    runner = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [("module", module)]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["attention"])]),
        req_states=SimpleNamespace(req_id_to_index={"a": 0, "b": 1}),
        block_tables=SimpleNamespace(blocks_per_kv_block=(1,), num_blocks=SimpleNamespace(np=np.array([[2, 2]])),
            block_tables=[SimpleNamespace(gpu=torch.tensor([[1, 2], [2, 1]], dtype=torch.int32))]))
    monkeypatch.setattr(window, "native_vllm_fp8_views", lambda *_: (layer.key, layer.value))
    with pytest.raises(ValueError, match="geometry"):
        window.NativeWindowAssembler.from_runner(runner, "a", ["module"], 0)
    assert not hasattr(runner, "_kvmem_assembly_staging")
    first = window.NativeWindowAssembler.from_runner(runner, "a", ["module"], 4)
    assert first.close()
    second = window.NativeWindowAssembler.from_runner(runner, "b", ["module"], 4)
    assert first.staging is second.staging is runner._kvmem_assembly_staging
    assert first.layers[0].kernel_page_table.tolist() == [1, 2]
    assert second.layers[0].kernel_page_table.tolist() == [2, 1]
    with pytest.raises(ValueError, match="geometry changed"):
        window.NativeWindowAssembler.from_runner(runner, "b", ["module"], 4, chunk_blocks=1)
    second.staging.claim(second)
    assert not second.close()
    second.staging.release(second)
    assert second.close() and not cuda_oracle.pools


def test_read_only_oracle_views_do_not_activate_split_assembly(cuda_oracle,monkeypatch):
    import numpy as np
    layer=cpu_layer()
    module=SimpleNamespace(num_kv_heads=1,head_dim=4,
        rotary_emb=SimpleNamespace(cos_sin_cache=layer.cos_sin),
        attn=SimpleNamespace(kv_cache_dtype='fp8_e4m3',layer_name='attention',
                            kv_cache=object(),_k_scale=layer.key_scale))
    options=SimpleNamespace(gpu_raw_k_cache_bytes=4096,reuse_resident_v=True,
                             memory=SimpleNamespace(select_budget=8))
    raw_cache=object()
    staging=window._AssemblyStaging(1*1*4*4*32*2,torch.device('cpu'),value_capacity_bytes=32)
    runner=SimpleNamespace(model=SimpleNamespace(named_modules=lambda:[('module',module)]),
        _kvmem_capture_runtime=SimpleNamespace(options=options),_kvmem_gpu_raw_cache=raw_cache,
        _kvmem_assembly_staging=staging,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['attention'])]),
        req_states=SimpleNamespace(req_id_to_index={'a':0}),
        block_tables=SimpleNamespace(blocks_per_kv_block=(1,), num_blocks=SimpleNamespace(np=np.array([[2]])),
            block_tables=[SimpleNamespace(gpu=torch.tensor([[1,2]],dtype=torch.int32))]))
    monkeypatch.setattr(window,'native_vllm_fp8_views',lambda *_:(layer.key,layer.value))
    observer=window.NativeWindowAssembler.from_runner(runner,'a',['module'],4)
    assert observer.raw_cache is None and not observer.reuse_resident_v
    assert runner._kvmem_gpu_raw_cache is raw_cache and observer.staging is staging
    assert observer.close() and not staging.buffers and staging.value_buffer is None


@pytest.mark.parametrize("direct", [False, True])
def test_metadata_is_submitted_once_before_chunks_and_partial_tail_keeps_dense_positions(
        cuda_oracle, monkeypatch, direct):
    from vllm_kvmem import packed_assembly

    layer = cpu_layer()
    layer.kernel_page_table = torch.tensor([1, 2, 3], dtype=torch.int32)
    assembler = window.NativeWindowAssembler((layer,), 4, chunk_blocks=1, packed=True,
                                              direct_pinned_reads=direct)
    remaps = [SimpleNamespace(block_id=i, n_tokens=count, to_base=i * 4, skip=False)
              for i, count in enumerate((4, 4, 3))]
    plan = SimpleNamespace(remaps=remaps)
    records = {i: SimpleNamespace(handle=SimpleNamespace(slot=i), count=remap.n_tokens)
               for i, remap in enumerate(remaps)}
    authority = SimpleNamespace(memory=SimpleNamespace(phase=Phase.ASSEMBLING, _plan=plan),
        shape=(1, 4, 1, 4), records=records, record_bytes=48, raw_dtype=torch.bfloat16,raw_element_bytes=2,
        read_records=lambda _: [(torch.zeros(1, 3, 1, 4, dtype=torch.bfloat16),
                                  torch.zeros(1, 3, 1, 4, dtype=torch.uint8))])
    operations, packed_positions, tail_positions = [], [], []
    real_empty = torch.empty
    monkeypatch.setattr(window.torch, "empty", lambda *args, **kwargs:
        real_empty(*args, **{**kwargs, "device": "cpu"}))

    def metadata(values):
        operations.append("metadata")
        return torch.tensor(values, dtype=torch.int64)

    class Transfers:
        def __init__(self, *_):
            self.pending = {}

        def upload(self, payload, destination, *, ready):
            assert ready is not None
            operations.append("upload")
            destination.copy_(payload)
            return SimpleNamespace(event=Event())

        def upload_from_reader(self, size, destination, fill, *, ready):
            host = real_empty(size, dtype=torch.uint8)
            fill(host)
            return self.upload(host, destination, ready=ready)

        def close(self):
            return True

    def read_packed(current, handles, *, destination=None):
        assert current is authority and len(handles) == 1
        if destination is None:
            destination = real_empty(48, dtype=torch.uint8)
        destination.zero_()
        return destination, {"bytes": destination.numel()}

    def assemble_tail(raw, values, positions, slots, *_):
        tail_positions.append((positions.tolist(), slots.tolist()))

    monkeypatch.setattr(assembler, "_upload_metadata", metadata)
    monkeypatch.setattr(window, "PackedTransfers", Transfers)
    monkeypatch.setattr(packed_assembly, "read_packed_records", read_packed)
    monkeypatch.setattr(packed_assembly, "assemble_packed_layer", lambda payload, bases, *args, **kwargs:
        packed_positions.append(bases.tolist()))
    monkeypatch.setattr(window, "assemble_fp8", assemble_tail)
    result = assembler.assemble(authority, plan)
    assert operations == ["metadata", "metadata", "upload", "upload", "upload"]
    assert packed_positions == [[0], [4]]
    assert tail_positions == [([8, 9, 10], [12, 13, 14])]
    assert result.rebuilt_tokens == 11 and result.chunks == 3
    assert result.uploaded_bytes == 2 * 48 + 36
    assert len(assembler.staging.buffers) == 2
