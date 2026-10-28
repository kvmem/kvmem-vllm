"""Opt-in bounded-memory verification; hash every KV byte in layout order.

This diagnostic does not affect ordinary serving. GDN/draft tensors are still
compared directly; target KV uses SHA256 instead of retaining gigabyte copies.
"""
import hashlib
import json

import torch


def update(digest, tensor):
    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def finish(keys, values, tokens, names, width):
    return dict(tokens=tokens, names=list(names), bytes=tokens*width*2*len(names),
        sha256=[h.hexdigest() for pair in zip(keys,values) for h in pair])


def target_digest(worker, request_id, tokens):
    from .window_assembly import NativeWindowAssembler
    assembler=NativeWindowAssembler.from_runner(worker.runner,request_id,worker.runtime.names,
                                                 worker.options.memory.block_tokens)
    try:
        keys=[hashlib.sha256() for _ in assembler.layers]
        values=[hashlib.sha256() for _ in assembler.layers]
        for begin in range(0,tokens,1024):
            positions=torch.arange(begin,min(begin+1024,tokens),device=worker.runner.device)
            for i,layer in enumerate(assembler.layers):
                pages=layer.kernel_page_table[positions//layer.key.shape[1]].long()
                offsets=positions%layer.key.shape[1]
                update(keys[i],layer.key[pages,offsets].view(torch.uint8).cpu())
                update(values[i],layer.value[pages,offsets].view(torch.uint8).cpu())
        return finish(keys,values,tokens,worker.runtime.names,worker.runtime.heads*worker.runtime.dim)
    finally:assembler.close()


def rebuilt_digest(worker, request_id, authority):
    from .window_assembly import NativeWindowAssembler
    from .codec import neox_rope
    authority.publish_sealed(worker.runtime._wait_archive_progress)
    records=tuple(authority.memory._store.working_records())
    assembler=NativeWindowAssembler.from_runner(worker.runner,request_id,worker.runtime.names,
                                                 worker.options.memory.block_tokens)
    try:
        keys=[hashlib.sha256() for _ in assembler.layers]
        values=[hashlib.sha256() for _ in assembler.layers]
        # The model advertises 10M logical positions, but this verifier only
        # needs the bounded dense working frame. Never copy an entire logical
        # RoPE table per layer. Shared tables reuse one bounded CPU prefix.
        extent=sum(r[3] for r in records)
        cache={}
        trigs=[]
        for layer in assembler.layers:
            table=layer.cos_sin
            identity=(table.data_ptr(),tuple(table.shape),tuple(table.stride()),table.dtype,table.device)
            if identity not in cache:cache[identity]=table[:extent].cpu()
            trigs.append(cache[identity])
        scales=[layer.key_scale.cpu() for layer in assembler.layers]
        for start in range(0,len(records),32):
            chunk=records[start:start+32]
            payloads=authority.read_records([authority.records[(s,g)].handle for s,g,*_ in chunk])
            for (_,_,_,count,position),(raw,value) in zip(chunk,payloads):
                positions=torch.arange(position,position+count)
                for i,layer in enumerate(assembler.layers):
                    trig=trigs[i][positions];half=trig.shape[-1]//2
                    rotated=neox_rope(raw[i],trig[:,:half],trig[:,half:])
                    key=(rotated.float()/scales[i]).clamp(-448,448).to(torch.float8_e4m3fn)
                    update(keys[i],key);update(values[i],value[i])
        return finish(keys,values,sum(r[3] for r in records),worker.runtime.names,worker.runtime.heads*worker.runtime.dim)
    finally:assembler.close()


def verify(worker,request_id,checkpoint,resident):
    binding=worker.bindings[request_id]
    tokens=binding.window if binding.pressure_restore else checkpoint.window.window_tokens
    saved=worker.restore_oracles.get(binding.namespace)
    expected=saved[1] if isinstance(saved,tuple) and saved[0]==checkpoint.position else None
    kind='native_terminal_window'
    if not resident and checkpoint.authority.raw_dtype==torch.float8_e4m3fn:
        expected=rebuilt_digest(worker,request_id,checkpoint.authority);kind='fp8_immutable_rebuild'
    elif binding.pressure_restore:
        expected=None
    if expected is None:
        raise RuntimeError('Missing bounded target verification oracle')
    actual=target_digest(worker,request_id,tokens)
    if actual!=expected:
        (worker.runner._kvmem_probe_output/'restore-digest-mismatch.json').write_text(
            json.dumps(dict(expected=expected,actual=actual,kind=kind),indent=2)+'\n')
        raise RuntimeError('Restored target KV SHA256 differs from '+kind)
    recurrent=worker.model_adapter._recurrent_views(request_id)
    draft=worker.model_adapter._draft_views(request_id,checkpoint.model.draft_begin,checkpoint.position)
    if (any(not torch.equal(t.cpu(),checkpoint.model.recurrent[name]) for name,t in recurrent.items()) or
        any(not torch.equal(cache[pages,offsets].cpu(),checkpoint.model.draft[name]) for name,(cache,pages,offsets) in draft.items())):
        raise RuntimeError('Restored GDN/draft bytes differ from checkpoint')
    report=dict(request_id=request_id,history_tokens=binding.history,working_tokens=tokens,
        target_kv_bytes=actual['bytes'],sha256_equal=True,sha256=actual['sha256'],
        verification='sha256_all_target_bytes_and_exact_model_tensors',oracle_kind=kind,
        recurrent_byte_equal=True,recurrent_tensors=len(recurrent),draft_layers=len(draft),
        working_attention_layers=len(worker.runtime.names),mtp_attention_layers=len(getattr(worker.runner,'_kvmem_mtp_modules',{})))
    with (worker.runner._kvmem_probe_output/'verified_restores_digest.jsonl').open('a') as stream:
        stream.write(json.dumps(report)+'\n')
