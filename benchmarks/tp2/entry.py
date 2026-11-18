"""Per-rank native graph evidence; optional TP=1 embedding UVA experiment."""
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from vllm.entrypoints.cli.main import main
from vllm.plugins import load_general_plugins

load_general_plugins()
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

original = GPUModelRunner.load_model


def load_model(self, *args, **kwargs):
    value = original(self, *args, **kwargs)
    if os.environ.get('BENCH_EMBEDDING_UVA') != '1':
        return value
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    target_model = self.model
    language = target_model.get_language_model() if hasattr(target_model, 'get_language_model') else target_model
    target = language.model.embed_tokens
    draft = self.speculator.model.model.embed_tokens
    assert target is draft and type(target) is VocabParallelEmbedding and target.tp_size == 1
    assert language.lm_head is not target
    weight = target.weight
    assert tuple(weight.shape) == (248320, 5120) and weight.dtype == torch.bfloat16
    params = {id(p): p for model in (target_model, self.speculator.model) for p in model.parameters()}
    others = {key: (p.data_ptr(), str(p.device), str(p.dtype)) for key,p in params.items() if p is not weight}
    before = torch.cuda.memory_allocated()
    cpu = torch.empty(weight.shape, dtype=weight.dtype, device='cpu', pin_memory=True)
    cpu.copy_(weight.detach(), non_blocking=False)
    digest = hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()
    assert digest == 'a158f24c6ad72736067768b4255e70a384e2e931f601b12f853e98ce83ec6bbe', digest
    view = get_accelerator_view_from_cpu_tensor(cpu)
    assert view.device == weight.device and view.shape == weight.shape and view.dtype == weight.dtype
    ids = torch.tensor([0,1,17,124160,248319],device=weight.device)
    assert torch.equal(torch.nn.functional.embedding(ids,weight),torch.nn.functional.embedding(ids,view))
    torch.cuda.synchronize()
    target._benchmark_embedding_cpu_owner = cpu
    weight.data = view
    weight._vllm_is_uva_offloaded = True
    del ids
    torch.cuda.empty_cache()
    reclaimed = before-torch.cuda.memory_allocated()
    assert reclaimed >= cpu.numel()*cpu.element_size(), reclaimed
    assert all((params[key].data_ptr(),str(params[key].device),str(params[key].dtype))==state
               for key,state in others.items())
    evidence = dict(parameter='language_model.model.embed_tokens.weight',
        bytes=cpu.numel()*cpu.element_size(),sha256=digest,shared_target_and_draft=True,
        native_gather_bitwise_equal=True,unchanged_other_parameters=len(others),
        gpu_bytes_reclaimed=reclaimed)
    Path(os.environ['BENCH_EMBEDDING_EVIDENCE']).write_text(json.dumps(evidence,indent=2)+'\n')
    print('BENCHMARK EMBEDDING UVA '+json.dumps(evidence),flush=True)
    return value


GPUModelRunner.load_model = load_model

original_capture_model = GPUModelRunner.capture_model


def capture_model(self, *args, **kwargs):
    value = original_capture_model(self, *args, **kwargs)
    from vllm.compilation.counter import compilation_counter
    target = self.cudagraph_manager
    # DFlash speculators expose query_cudagraph_manager; MTP has its decode manager.
    draft = (getattr(self.speculator, 'query_cudagraph_manager', None) or
             getattr(self.speculator, 'decode_cudagraph_manager', None))
    if draft is None:
        return value
    evidence = dict(target_mode=str(target.cudagraph_mode),
        draft_mode=str(draft.cudagraph_mode),
        target_capture_tokens=target.captured_token_counts(),
        draft_capture_tokens=draft.captured_token_counts(),
        piecewise_graph_count=compilation_counter.num_cudagraph_captured,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved())
    from vllm.distributed import get_tp_group
    rank = get_tp_group().rank_in_group
    evidence.update(tp_rank=rank, tp_size=get_tp_group().world_size)
    path=Path(os.environ['BENCH_GRAPH_DIR'])/f'graph-evidence-rank{rank}.json' 
    path.write_text(json.dumps(evidence,indent=2)+"\n")
    print("BENCHMARK GRAPH EVIDENCE "+json.dumps(evidence),flush=True)
    if os.environ.get("BENCH_GRAPH_TRACE") == "1":
        original_replay = torch.cuda.CUDAGraph.replay
        draft_ids = {id(graph) for graph in draft.graphs.values()}
        counts = dict(target_replays=0,draft_replays=0)
        import atexit
        from vllm_kvmem.serving.report_writer import LatestReportWriter
        # A slow filesystem must not delay the next CUDA graph launch.
        writer = LatestReportWriter(path.parent)
        replay_name = f"graph-replays-rank{rank}"
        last_publish = [0.0]

        def finish_counts():
            writer.submit(replay_name, dict(counts, observed_at=time.monotonic()))
            writer.close()
        atexit.register(finish_counts)

        def replay(graph):
            result=original_replay(graph)
            key="draft_replays" if id(graph) in draft_ids else "target_replays"
            counts[key]+=1
            now = time.monotonic()
            if now - last_publish[0] >= 1.0:
                writer.submit(replay_name, dict(counts, observed_at=now))
                last_publish[0] = now
            return result
        torch.cuda.CUDAGraph.replay=replay
    return value


GPUModelRunner.capture_model = capture_model
if __name__ == '__main__':
    main()
