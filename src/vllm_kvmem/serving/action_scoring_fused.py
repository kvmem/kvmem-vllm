"""FP32 sampled QK with shifted segment sums, no token-sized matrices."""
import torch
import triton
import triton.language as tl
from ..adapters.fp8_compat import fp8_storage, fp8_to_float


@triton.jit
def _segment_stats(Q,K,Table,Positions,Scale,Starts,Counts,Maxima,Sums,
                 QS:tl.constexpr,QH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,KH:tl.constexpr,
                 HEADS:tl.constexpr,GROUP:tl.constexpr,DIM:tl.constexpr,PAGE:tl.constexpr,
                 LENGTH,SEGMENTS,ATTENTION_SCALE:tl.constexpr,B:tl.constexpr,D:tl.constexpr,
                 Q_BYTES:tl.constexpr,K_BYTES:tl.constexpr):
    row,head,segment=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    offset,dim=tl.arange(0,B),tl.arange(0,D)
    begin,count=tl.load(Starts+segment),tl.load(Counts+segment)
    pos=begin+offset
    valid=(offset<count)&(pos<LENGTH)&(pos<=tl.load(Positions+row))
    pages=tl.load(Table+pos//PAGE,mask=valid,other=0).to(tl.int64)
    keys=tl.load(K+pages[:,None]*KP+(pos%PAGE)[:,None]*KT+(head//GROUP)*KH+dim[None,:],
        mask=valid[:,None]&(dim[None,:]<DIM),other=0.)
    keys=fp8_to_float(keys,K_BYTES)
    query=fp8_to_float(tl.load(Q+row*QS+head*QH+dim,mask=dim<DIM,other=0.),Q_BYTES)
    # Match the reference FP32 point product/reduction and scale order. This
    # is the existing bypass QK, not native FlashInfer/XQA accumulator reuse.
    logits=tl.sum(keys*query[None,:],axis=1)*tl.load(Scale)*ATTENTION_SCALE
    logits=tl.where(valid,logits,-float('inf'))
    center=tl.max(logits,axis=0)
    # Keep the maximum separate: adding log(sum) to a large FP32 maximum
    # discards normalization bits that subtraction cannot recover later.
    shift=tl.where(center==-float('inf'),0.,center)
    total=tl.sum(tl.exp(logits-shift),axis=0)
    index=(row*HEADS+head)*SEGMENTS+segment
    # Empty causal segments must retain -inf, not zero: zero could dominate
    # the row maximum when every visible logit is large and negative.
    tl.store(Maxima+index,center)
    tl.store(Sums+index,total)


@triton.jit
def _normalize_segments(Maxima,Sums,HEADS:tl.constexpr,SEGMENTS,B:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    i=tl.arange(0,B)
    index=(row*HEADS+head)*SEGMENTS+i
    maxima=tl.load(Maxima+index,mask=i<SEGMENTS,other=-float('inf'))
    sums=tl.load(Sums+index,mask=i<SEGMENTS,other=0.)
    center=tl.max(maxima,axis=0)
    shifted=sums*tl.exp(maxima-center)
    total=tl.sum(shifted,axis=0)
    # Each program owns a complete row/head. Reuse its sum buffer for the
    # normalized masses; the following kernel only averages independent heads.
    tl.store(Sums+index,shifted/total,mask=i<SEGMENTS)


@triton.jit
def _mass(Sums,Out,HEADS:tl.constexpr,SEGMENTS,H:tl.constexpr):
    row,segment=tl.program_id(0),tl.program_id(1)
    h=tl.arange(0,H)
    index=(row*HEADS+h)*SEGMENTS+segment
    probability=tl.load(Sums+index,mask=h<HEADS,other=0.)
    tl.store(Out+row*SEGMENTS+segment,tl.sum(probability,axis=0)/HEADS)


def fused_page_attention(q,positions,key,table,effective_scale,starts,counts,length,*,scale,block_tokens):
    """Caller guarantees segments partition the complete visible key domain.

    Each head uses one common maximum and denominator across all segments.
    Reduction rounding can still differ from the reference token softmax;
    near-tied graph edges are not guaranteed to have identical ordering.
    """
    rows,heads,dim=q.shape;segments=starts.numel()
    maxima=torch.empty((rows,heads,segments),device=q.device,dtype=torch.float32)
    sums=torch.empty_like(maxima)
    mass=torch.empty((rows,segments),device=q.device,dtype=torch.float32)
    q_input,q_bytes=fp8_storage(q)
    key_input,key_bytes=fp8_storage(key)
    _segment_stats[(rows,heads,segments)](q_input,key_input,table,positions,effective_scale,starts,counts,maxima,sums,
        q.stride(0),q.stride(1),*key.stride()[:3],heads,heads//key.shape[2],dim,key.shape[1],
        length,segments,scale,triton.next_power_of_2(block_tokens),triton.next_power_of_2(dim),q_bytes,key_bytes,
        enable_fp_fusion=False)
    _normalize_segments[(rows,heads)](maxima,sums,heads,segments,triton.next_power_of_2(segments),enable_fp_fusion=False)
    _mass[(rows,segments)](sums,mass,heads,segments,triton.next_power_of_2(heads),enable_fp_fusion=False)
    mass._action_scoring_backend='fused'
    return mass


@triton.jit
def _segment_stats_layers(Q,K,KOffsets,Table,Positions,Scales,Starts,Counts,Maxima,Sums,ROWS,
                 QL:tl.constexpr,QS:tl.constexpr,QH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,KH:tl.constexpr,
                 HEADS:tl.constexpr,GROUP:tl.constexpr,DIM:tl.constexpr,PAGE:tl.constexpr,
                 LENGTH,SEGMENTS,ATTENTION_SCALE:tl.constexpr,B:tl.constexpr,D:tl.constexpr,
                 Q_BYTES:tl.constexpr,K_BYTES:tl.constexpr):
    # _segment_stats for several layers in one launch: program 0 spans layer x row.
    # Each layer's keys start KOffsets[layer] elements from K (same shape and
    # strides); its query is Q[layer]; the arithmetic per row/head/segment is
    # the single-layer kernel's, in the same order.
    flat,head,segment=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    layer=flat//ROWS
    row=flat-layer*ROWS
    offset,dim=tl.arange(0,B),tl.arange(0,D)
    begin,count=tl.load(Starts+segment),tl.load(Counts+segment)
    pos=begin+offset
    valid=(offset<count)&(pos<LENGTH)&(pos<=tl.load(Positions+row))
    pages=tl.load(Table+pos//PAGE,mask=valid,other=0).to(tl.int64)
    # Offsets are multiples of 16 elements (checked on the host): the layer's
    # key pointer keeps K's 16-byte alignment, so loads vectorize, and the dot
    # product reduces, exactly as in the single-layer kernel.
    keys=tl.load(K+tl.multiple_of(tl.load(KOffsets+layer),16)+pages[:,None]*KP+(pos%PAGE)[:,None]*KT+(head//GROUP)*KH+dim[None,:],
        mask=valid[:,None]&(dim[None,:]<DIM),other=0.)
    keys=fp8_to_float(keys,K_BYTES)
    query=fp8_to_float(tl.load(Q+layer*QL+row*QS+head*QH+dim,mask=dim<DIM,other=0.),Q_BYTES)
    logits=tl.sum(keys*query[None,:],axis=1)*tl.load(Scales+layer)*ATTENTION_SCALE
    logits=tl.where(valid,logits,-float('inf'))
    center=tl.max(logits,axis=0)
    shift=tl.where(center==-float('inf'),0.,center)
    total=tl.sum(tl.exp(logits-shift),axis=0)
    index=(flat*HEADS+head)*SEGMENTS+segment
    tl.store(Maxima+index,center)
    tl.store(Sums+index,total)


_KEY_OFFSETS = {}


def _key_offsets(keys, base, device):
    """Element offsets of each layer's key view from the first one (cached: KV caches never move)."""
    pointers = tuple(fp8_storage(k)[0].data_ptr() for k in keys)
    cached = _KEY_OFFSETS.get((device, pointers))
    if cached is None:
        size = base.element_size()
        if any((p - pointers[0]) % (16 * size) for p in pointers):
            raise ValueError('Layer key caches are not 16-element aligned to each other')
        cached = torch.tensor([(p - pointers[0]) // size for p in pointers], dtype=torch.int64, device=device)
        if len(_KEY_OFFSETS) > 64:
            _KEY_OFFSETS.clear()
        _KEY_OFFSETS[(device, pointers)] = cached
    return cached


def fused_page_attention_layers(q,positions,keys,table,effective_scales,starts,counts,length,*,scale,block_tokens):
    """fused_page_attention for every layer at once: q [L, rows, heads, dim],
    keys = L key views of one shape/stride/dtype, effective_scales [L].

    Returns [L, rows, segments]; layer l equals fused_page_attention on q[l],
    keys[l] and effective_scales[l] bit for bit (same per-element arithmetic).
    """
    layers,rows,heads,dim=q.shape;segments=starts.numel()
    first=keys[0]
    if (len(keys)!=layers or effective_scales.shape!=(layers,) or
            any(k.shape!=first.shape or k.stride()!=first.stride() or k.dtype!=first.dtype or
                k.device!=first.device for k in keys)):
        raise ValueError('Layer-batched scoring needs one key geometry for every layer')
    maxima=torch.empty((layers*rows,heads,segments),device=q.device,dtype=torch.float32)
    sums=torch.empty_like(maxima)
    mass=torch.empty((layers*rows,segments),device=q.device,dtype=torch.float32)
    q_input,q_bytes=fp8_storage(q)
    key_input,key_bytes=fp8_storage(first)
    offsets=_key_offsets(keys,key_input,q.device)
    _segment_stats_layers[(layers*rows,heads,segments)](q_input,key_input,offsets,table,positions,effective_scales,
        starts,counts,maxima,sums,rows,q.stride(0),q.stride(1),q.stride(2),*first.stride()[:3],heads,
        heads//first.shape[2],dim,first.shape[1],length,segments,scale,triton.next_power_of_2(block_tokens),
        triton.next_power_of_2(dim),q_bytes,key_bytes,enable_fp_fusion=False)
    _normalize_segments[(layers*rows,heads)](maxima,sums,heads,segments,triton.next_power_of_2(segments),
                                             enable_fp_fusion=False)
    _mass[(layers*rows,segments)](sums,mass,heads,segments,triton.next_power_of_2(heads),enable_fp_fusion=False)
    mass=mass.view(layers,rows,segments)
    mass._action_scoring_backend='fused'
    return mass
