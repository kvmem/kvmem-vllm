"""Scheduler-owned target semantic refresh using native batched model execution.

Refresh transactions are eager, synchronous, text-only, TP=1, APC off; regular
decode may use FULL native graphs. All batch neighbours are restored after a
provisional query, while only selected requests change attention windows.
DFlash2 preserves independent native draft sliding pages. Exact terminal
checkpoints are handled separately; arbitrary session rollback is unsupported.
"""
from .native_pages import stored_expansions
from dataclasses import dataclass
import json
import os
import time
from types import MethodType

import torch

from .cache_groups import full_attention_groups
from .capture import capture_to
from .native_state import NativeRecurrentState
from .state import StateCheckpoints
from .window_assembly import NativeWindowAssembler
from .window_batch import WindowBatchAdapter


@dataclass
class QueryJob:
    command: object
    event: object = None
    selected_tokens: int | None = None

    def query(self):
        return self.event is not None and self.event.query()


def install_bounded_target(runner):
    from .options import RuntimeOptions
    options = RuntimeOptions.from_vllm(runner.vllm_config)
    draft_contract = None
    if runner.speculative_config is not None:
        if os.getenv("KVMEM_BOUNDED_DFLASH") != "1" and os.getenv("KVMEM_BOUNDED_MTP") != "1":
            raise ValueError("Bounded speculation requires explicit DFlash2 opt-in")
        from .draft_contract import DraftContract
        from .options import RuntimeOptions
        draft_contract = DraftContract.from_vllm(runner.vllm_config, RuntimeOptions.from_vllm(runner.vllm_config).memory)
    runner._kvmem_draft_contract = draft_contract
    if (runner.vllm_config.max_concurrent_batches != 1 or
            runner.vllm_config.cache_config.enable_prefix_caching or
            runner.vllm_config.cache_config.mamba_cache_mode != "none" or
            os.getenv("KVMEM_WINDOW_METADATA") != "1" or not runner._kvmem_continuous):
        raise ValueError("Bounded execution requires synchronous APC-off archive/window execution")
    # Private read transactions must bypass the public refresh-job wrapper.
    runner._kvmem_private_forward = runner.model.forward
    if options.async_refresh:
        if getattr(runner.speculative_config, 'method', None) != 'mtp':
            raise ValueError('Asynchronous refresh currently qualifies native Qwen MTP')
        from .async_refresh import AsyncRefresh
        runner._kvmem_async_refresh = AsyncRefresh(runner, options)
        runner._kvmem_query_refresh = runner._kvmem_async_refresh.enqueue
        return
    jobs = {}
    original = runner.model.forward

    def enqueue(command):
        if command.request_id in jobs:
            raise ValueError("Duplicate scheduled query refresh")
        job = QueryJob(command)
        jobs[command.request_id] = job
        return job

    runner._kvmem_query_refresh = enqueue

    def forward(self, *args, **kwargs):
        if not jobs:
            return original(*args, **kwargs)
        batch = runner._kvmem_last_batch
        started_at = time.perf_counter()
        lifecycle_trace = os.getenv("KVMEM_LIFECYCLE_TRACE") == "1"
        if lifecycle_trace:
            active = {"started_at": started_at, "requests": [
                {"request_id": request_id, "command_id": job.command.command_id,
                 "history_begin": job.command.groups[0][1].history_end,
                 "query_tokens": job.command.groups[0][1].replay_tokens}
                for request_id, job in jobs.items()]}
            temporary = runner._kvmem_probe_output / "bounded_active.tmp"
            temporary.write_text(json.dumps(active))
            temporary.replace(runner._kvmem_probe_output / "bounded_active.json")
        runtime = runner._kvmem_capture_runtime
        if runtime is None or not set(jobs).issubset(batch.req_ids):
            raise ValueError("Scheduled refresh is missing its archived history or query batch")
        draft_groups = draft_contract.validate_native_groups(runner) if draft_contract is not None else ()
        if not runtime.drain_requests(jobs, wait=True):
            raise RuntimeError('Query replay preceded scheduler publication confirmation')
        cfg = runtime.config
        extents = {}
        for request_id, job in jobs.items():
            i = batch.req_ids.index(request_id)
            expected_groups = full_attention_groups(runner.kv_cache_config)
            if tuple(group for group, _ in job.command.groups) != expected_groups:
                raise ValueError("Query command does not cover exactly the native attention groups")
            request_index = runner.req_states.req_id_to_index[request_id]
            for group_id, ticket in job.command.groups:
                expansion = stored_expansions(runner.block_tables)[group_id]
                expected_pages = [block * expansion + page for block in ticket.original_block_ids
                                  for page in range(expansion)]
                count = int(runner.block_tables.num_blocks.np[group_id, request_index])
                actual_pages = runner.block_tables.block_tables[group_id].gpu[request_index, :count].cpu().tolist()
                if actual_pages != expected_pages:
                    raise ValueError("Worker physical pages do not match the scheduler refresh reservation")
            tickets = [ticket for _, ticket in job.command.groups]
            values = {(t.history_end, t.window_tokens, t.replay_tokens) for t in tickets}
            if len(values) != 1:
                raise ValueError("Refresh groups disagree on query extent")
            history, selected_cap, query = values.pop()
            authority = runtime.authorities[request_id]
            from .prompt_spans import install_selection_metadata
            install_selection_metadata(authority, job.command, validated_prefix=history)
            if getattr(authority,'query_history',None) is not None:
                from .query_history import begin_round
                begin_round(runner, authority, request_id)
            if (history != authority.history_end or history != int(batch.num_computed_tokens_np[i]) or
                    not 0 < selected_cap <= cfg.select_budget or
                    query != int(batch.num_scheduled_tokens[i])):
                raise ValueError("Scheduler query does not match accepted authority or native input batch")
            extents[request_id] = (i, history, query)

        if runner._kvmem_window_adapter is None:
            groups = full_attention_groups(runner.kv_cache_config)
            runner._kvmem_window_adapter = WindowBatchAdapter(groups, cfg.select_budget + cfg.gen_budget)
        needs_probe = any(history > jobs[request_id].command.groups[0][1].window_tokens and jobs[request_id].command.selection != 'pressure'
                          and not jobs[request_id].command.guided_reason
                          for request_id, (_, history, _) in extents.items())
        automatic = {rid for rid, job in jobs.items() if job.command.guided_reason}
        needs_state = needs_probe or bool(automatic)
        automatic_queries, automatic_reports = {}, {}
        state_adapter = checkpoints = before = bundle = None
        captured = []
        audit_guided = (options.guided_query_tokens > 0 and
                        os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1')
        guided_draft = []
        rolling_audits = {}
        if options.rolling_active_window and os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1':
            from types import SimpleNamespace
            from .rolling_verification import RollingPressureOracle
            for request_id, job in jobs.items():
                i, history, _ = extents[request_id]
                if job.command.selection == 'pressure' and history >= int(batch.prefill_len_np[i]):
                    audit_job = SimpleNamespace(command=job.command, authority=runtime.authorities[request_id],
                        history=history, cancelled=False, selected_tokens=0, steps=int(needs_probe))
                    rolling_audits[request_id] = (audit_job, RollingPressureOracle(runner, audit_job))
        if needs_state:
            state_adapter = NativeRecurrentState(runner)
            bundle = state_adapter.borrow(batch.req_ids)
            checkpoints = StateCheckpoints("bounded-query", bundle.bytes, capacity=1)
            before = checkpoints.capture(0, bundle.tensors)
            if audit_guided:
                from .guided_query import snapshot_guided_draft
                guided_draft = snapshot_guided_draft(runner,batch,draft_contract)
            for rid in automatic:
                i, history, query = extents[rid]
                command = jobs[rid].command
                authority = runtime.authorities[rid]
                from .guided_seed import original_queries
                fallback = original_queries(authority, history, runner.device)
                diagnostics = dict(guided_query_tokens=0, guided_prompt_tokens=0,
                    guided_seconds=0.0, guided_fallback='emergency-original-query')
                queries = None
                if command.guided_private:
                    reserve = options.guided_reserve
                    local = history-runner._kvmem_window_adapter.offsets.get(rid, 0)
                    private_end = local+len(command.input_tokens)+reserve
                    available = private_end <= cfg.select_budget+cfg.gen_budget
                    for group_id, ticket in command.groups:
                        spec = runner.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
                        available &= private_end <= len(ticket.original_block_ids)*spec.block_size
                    diagnostics['guided_fallback'] = 'capacity'
                    if available:
                        from .guided_query import guided_query
                        started = time.perf_counter()
                        try:
                            generated = guided_query(runner, original, batch, i, history,
                                                     reserve, options.guided_query_tokens,
                                                     prefix=command.input_tokens)
                            queries = generated.queries
                            diagnostics.update(generated.diagnostics,
                                guided_prompt_tokens=generated.prompt_tokens,
                                guided_seconds=time.perf_counter()-started)
                        finally:
                            state_adapter.validate(bundle)
                            checkpoints.restore(before, bundle.tensors)
                if queries is None:
                    queries = fallback
                if queries is None:
                    # Fail before committing any selection. A pressure pick
                    # cannot be reported as a successful semantic refresh or
                    # advance its epoch anchor.
                    raise RuntimeError('Guided refresh has neither a generated nor an accepted original Query: '
                                       + str(diagnostics['guided_fallback']))
                automatic_queries[rid], automatic_reports[rid] = queries, diagnostics
        if needs_probe:
            if getattr(runner, '_kvmem_graph', False):
                original(*args, **kwargs)
                captured = runner._kvmem_graph_captures
            else:
                with capture_to(captured.append):
                    original(*args, **kwargs)
            if tuple(c.name for c in captured) != tuple(runner._kvmem_layers):
                raise RuntimeError("Native provisional batch missed attention layers")

        # The provisional query reads leased native GPU pages, not archived
        # bytes. Let its GPU execution overlap sealed-page packing/SHA/I/O;
        # publication still finishes before selection and any archive read.
        for request_id in jobs:
            authority = runtime.authorities[request_id]
            authority.publish_sealed(runtime._wait_archive_progress)

        owners, reports = [], []
        public_queries = {}
        for request_id, (i, history, query) in extents.items():
            command = jobs[request_id].command
            if (history <= command.groups[0][1].window_tokens or command.selection == 'pressure'
                    or command.guided_reason):
                continue
            begin, end = batch.query_start_loc_np[i:i+2]
            if command.query_span is not None:
                qb, qe = command.query_span
                if not history <= qb < qe <= history + query:
                    raise ValueError('Complete explicit Query is missing from its replay batch')
                begin, end = begin + qb - history, begin + qe - history
            elif draft_contract is not None:
                from .speculative_admission import SpeculativeQueryPolicy
                end = begin + SpeculativeQueryPolicy.authoritative_probe_tokens(
                    history,int(batch.prefill_len_np[i]),query)
            # Private forwards overwrite the persistent capture buffers.
            public_queries[request_id] = torch.stack([c.query[begin:end] for c in captured]).clone()
        for request_id, job in jobs.items():
            i, history, query = extents[request_id]
            local_tokens_before = history - runner._kvmem_window_adapter.offsets.get(request_id, 0)
            authority = runtime.authorities[request_id]
            score_query = query
            guided_tokens = guided_prompt_tokens = 0
            guided_seconds = 0.0
            guided_fallback = None
            guided_diagnostics = {}
            if request_id in automatic:
                guided_diagnostics = automatic_reports[request_id]
                guided_tokens = guided_diagnostics['guided_query_tokens']
                guided_prompt_tokens = guided_diagnostics['guided_prompt_tokens']
                guided_seconds = guided_diagnostics['guided_seconds']
                guided_fallback = guided_diagnostics['guided_fallback']
            from .request_plan import RequestPlan, selection_config
            pressure_policy = getattr(runtime.options, 'pressure_policy', None) or (
                'attention' if getattr(runtime.options, 'action_pressure', False) else 'recency')
            selected_cfg = selection_config(cfg,job.command.groups[0][1].window_tokens,
                selection=job.command.selection,pressure_policy=pressure_policy)
            mandatory_fit = RequestPlan(None,block_tokens=cfg.block_tokens,gen_budget=cfg.gen_budget,
                mandatory=getattr(job.command,'mandatory_spans',())).mandatory_fit(history,selected_cfg)
            required = set(mandatory_fit['hard_blocks'])
            mandatory = [r.handle for r in authority.records.values() if r.history_begin//cfg.block_tokens in required]
            if {authority.records[(h.slot,h.generation)].history_begin//cfg.block_tokens for h in mandatory} != required:
                raise ValueError('Required historical blocks are missing from the durable source')
            if job.command.selection == 'pressure':
                from .prefill_attention import select_pressure
                plan = select_pressure(authority,mandatory,max_tokens=selected_cfg.select_budget)
            elif history <= selected_cfg.select_budget:
                plan = authority.memory.select({})  # Exact all-history bootstrap.
            else:
                queries = (automatic_queries[request_id] if request_id in automatic
                           else public_queries[request_id])
                boundary_guided = (job.command.guided_boundary if job.command.guided_boundary is not None
                    else options.guided_reselect in ('boundary', 'both'))
                if runtime.options.guided_query_tokens and job.command.query_span is not None and boundary_guided:
                    reserve = runtime.options.guided_reserve
                    local_end = history + query - runner._kvmem_window_adapter.offsets.get(request_id,0)
                    available = local_end+reserve <= cfg.select_budget+cfg.gen_budget
                    for group_id, ticket in job.command.groups:
                        spec = runner.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
                        available &= local_end+reserve <= len(ticket.original_block_ids)*spec.block_size
                    if not available:
                        # A warm M can land exactly at the new Query boundary.
                        # Keep the public Query valid without exceeding K+G for
                        # a private rewrite; record this fallback explicitly.
                        guided_fallback = 'capacity'
                    else:
                        from .guided_query import guided_query
                        guided_started = time.perf_counter()
                        try:
                            guided = guided_query(
                                runner,original,batch,i,history+query,reserve,runtime.options.guided_query_tokens)
                        except BaseException:
                            state_adapter.validate(bundle)
                            checkpoints.restore(before,bundle.tensors)
                            raise
                        guided_diagnostics = guided.diagnostics
                        guided_tokens = guided_diagnostics['guided_query_tokens']
                        guided_prompt_tokens = guided.prompt_tokens
                        guided_fallback = guided_diagnostics['guided_fallback']
                        if guided.queries is not None:
                            queries = guided.queries
                        guided_seconds = time.perf_counter() - guided_started
                score_query = queries.shape[1]
                plan = authority.select_for_queries(queries,mandatory,max_tokens=selected_cfg.select_budget)
            selected = plan.total_window_tokens
            if not 0 < selected <= cfg.select_budget:
                raise RuntimeError("Original selector exceeded its reserved prefix")
            assembly_start = time.perf_counter()
            assembler = NativeWindowAssembler.from_runner(runner, request_id, runtime.names, cfg.block_tokens,
                                                          packed=runtime.options.packed_assembly,
                                                          direct_pinned_reads=runtime.options.direct_pinned_reads)
            assembly = assembler.assemble(authority, plan)
            assembly.event.synchronize()
            authority.memory.attach_completion(assembly)
            assert authority.memory.commit_selection()
            sessions = getattr(runner, '_kvmem_sessions', None)
            if sessions is not None:
                sessions.selection_committed(request_id, history)
            runner._kvmem_window_adapter.commit(request_id, history, selected)
            if request_id in rolling_audits:
                audit_job, _ = rolling_audits[request_id]
                audit_job.selected_tokens = selected
                audit_job.result = (assembler, assembly, plan)
            if job.command.selection!='pressure' and not job.command.guided_reason:
                from .prefill_attention import exclude_query_replay
                exclude_query_replay(runner,authority,request_id,history,history+query,job.command.query_span is not None)
            from .prefill_attention import selection_report
            job.selected_tokens = selected
            owners.append((assembler, assembly, plan))
            reports.append({"request_id": request_id, "command_id": job.command.command_id,
                "history_begin": history, "selected_tokens": selected, "query_tokens": query,
                "rolling_active_window": options.rolling_active_window,
                "local_tokens_before": local_tokens_before,
                "rebuilt_tokens": assembly.rebuilt_tokens, "skipped_tokens": assembly.skipped_tokens,
                "batch_requests": list(batch.req_ids), "probe_executed": needs_probe,
                "query_score_tokens": score_query, "draft_groups": list(draft_groups),
                "guided_query_tokens":guided_tokens,"guided_prompt_tokens":guided_prompt_tokens,
                "guided_seconds":guided_seconds,
                "guided_fallback":guided_fallback,
                "guided_trigger_reason":job.command.guided_reason or 'explicit-query' if job.command.selection == 'query' else None,
                "guided_original_seed_bytes":getattr(getattr(authority, 'guided_seed', None), 'bytes', 0),
                **guided_diagnostics,
                "query_history_enabled":getattr(authority,'query_history',None) is not None,
                "action_attention_enabled":cfg.action_attention,
                **selection_report(authority, phase=job.command.selection),
                "action_history_pages":sum(v>0 for v in getattr(getattr(authority,'action_history',None),'scores',{}).values()),
                "action_fraction":cfg.action_fraction if cfg.action_attention else 0.0,
                "action_combine":cfg.action_combine,
                "query_history_pages":len(authority.query_history.policy.scores) if getattr(authority,'query_history',None) is not None else 0,
                "selection": job.command.selection, "query_span": job.command.query_span,
                "mandatory_fit":mandatory_fit,
                "selected_history_ranges":[[r[2],r[3]] for r in authority.memory._store.working_records()],
                "replay_span": [history, history + query], "mean_k_backend": runtime.options.mean_k_backend,
                "assembly_seconds": time.perf_counter() - assembly_start,
                "assembly_host_prepare_seconds": assembly.host_prepare_seconds,
                "assembly_uploaded_bytes": assembly.uploaded_bytes,
                "assembly_read_stats": assembly.read_stats,
                "raw_cache_hit_tokens": assembly.raw_cache_hit_tokens,
                "resident_v_reused_tokens": assembly.resident_v_reused_tokens,
                "raw_cache_capacity_bytes": assembly.raw_cache_capacity_bytes,
                "assembly_device_temp_bytes": assembly.device_temp_bytes,
                "packed_assembly": runtime.options.packed_assembly,
                "direct_pinned_reads": runtime.options.direct_pinned_reads})

        if needs_state:
            state_adapter.validate(bundle)
            checkpoints.restore(before, bundle.tensors)
            if audit_guided and any(row['guided_query_tokens'] for row in reports):
                from .guided_query import verify_guided_rollback
                audit = verify_guided_rollback(before,bundle.tensors,guided_draft)
                with (runner._kvmem_probe_output/'verified_guided_rollback.jsonl').open('a') as stream:
                    for row in reports:
                        if row['guided_query_tokens']:
                            stream.write(json.dumps(dict(request_id=row['request_id'],
                                batch_requests=list(batch.req_ids),**audit))+'\n')
            guided_draft.clear()
        for audit_job, audit in rolling_audits.values():
            audit.verify(audit_job)
        from vllm.forward_context import get_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        context = get_forward_context()
        block_tables, slots = runner.prepare_attn(batch)
        metadata = runner.model_state.prepare_attn(batch, context.cudagraph_runtime_mode,
            block_tables, slots, runner.attn_groups, runner.kv_cache_config)
        context.attn_metadata.clear()
        context.attn_metadata.update(metadata)
        context.slot_mapping.clear()
        context.slot_mapping.update(build_slot_mappings_by_layer(slots, runner.kv_cache_config))
        window = runner._kvmem_window_batch
        if window is not None:
            positions = window.positions
            if kwargs["positions"].ndim == 2:
                positions = positions.unsqueeze(0).expand(kwargs["positions"].shape[0], -1)
            kwargs = {**kwargs, "positions": positions}
        result = original(*args, **kwargs)
        completed = torch.cuda.Event()
        completed.record()
        # First integration retires scratch synchronously. Keep the connector
        # acknowledgment after every replayed neighbour and the query itself.
        completed.synchronize()
        if needs_state:
            assert checkpoints.release(0)
        for assembler, _, _ in owners:
            assert assembler.close()
        finished_at = time.perf_counter()
        if lifecycle_trace:
            (runner._kvmem_probe_output / "bounded_active.json").unlink()
            for report in reports:
                report.update(refresh_started_at=started_at, refresh_finished_at=finished_at)
        with (runner._kvmem_probe_output / "bounded_refresh.jsonl").open("a") as stream:
            for report in reports:
                stream.write(json.dumps(report) + "\n")
        for job in jobs.values():
            job.event = completed
        jobs.clear()
        return result

    runner.model.forward = MethodType(forward, runner.model)
