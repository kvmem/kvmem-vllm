"""Complete refreshed page rows survive native transport without double append."""
from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_kvmem.connector import ControlMetadata
from vllm_kvmem.runner import make_probe_runner


def publication():
    output = SchedulerOutput.make_empty()
    return replace(output, kv_connector_metadata=ControlMetadata(
        block_table_updates={'refreshed': ([3, 4, 5],)}),
        scheduled_cached_reqs=replace(output.scheduled_cached_reqs,
            req_ids=['refreshed', 'peer'], new_block_ids=[([5],), ([11],)],
            num_computed_tokens=[32, 64], num_output_tokens=[1, 1]))


def test_connector_page_rows_survive_worker_pickle_transport():
    output = publication()
    # Native worker MessageQueue uses the highest-protocol pickle transport.
    result = pickle.loads(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(result.kv_connector_metadata, ControlMetadata)
    assert result.kv_connector_metadata.block_table_updates == {'refreshed': ([3, 4, 5],)}


def test_complete_rows_replace_stale_pages_without_duplicate_new_allocations():
    tables = {0: [1, 2], 1: [10]}
    counters = {}

    def append(index, blocks, overwrite=False):
        if overwrite:
            tables[index] = list(blocks[0])
        else:
            tables[index].extend(blocks[0])

    class Native:
        def update_requests(self, output):
            cached = output.scheduled_cached_reqs
            for rid, count, blocks in zip(cached.req_ids, cached.num_computed_tokens, cached.new_block_ids):
                counters[rid] = count
                if blocks is not None:
                    append(self.req_states.req_id_to_index[rid], blocks)
            return 'native-updated'

    runner = make_probe_runner(Native)()
    runner.req_states = SimpleNamespace(req_id_to_index={'refreshed': 0, 'peer': 1})
    runner.block_tables = SimpleNamespace(append_block_ids=append)
    output = publication()
    runner._kvmem_restore_allocations = {'refreshed': object()}
    with pytest.raises(RuntimeError, match='public page/state update'):
        runner.update_requests(output)
    assert tables == {0: [1, 2], 1: [10]} and not counters
    runner._kvmem_restore_allocations.clear()
    assert runner.update_requests(output) == 'native-updated'
    assert tables == {0: [3, 4, 5], 1: [10, 11]}
    assert counters == {'refreshed': 32, 'peer': 64}
    assert output.scheduled_cached_reqs.new_block_ids == [([5],), ([11],)]
