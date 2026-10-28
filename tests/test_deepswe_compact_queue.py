"""New context-cap cohort cannot silently change original task semantics."""
from copy import deepcopy
import pytest
from scripts import queue_deepswe_compact_300k as queue


def test_profile_only_changes_requested_cap_and_storage():
    source = dict(max_model_len=524288, max_num_seqs=8, kvmem=dict(
        select_budget=36864, gen_budget=16384, query_mean_quota_fraction=.5,
        graph_alpha=0., cpu_read_cache_bytes=10**11, query_window_cache_bytes=12*2**30))
    saved = deepcopy(source)
    after = queue.profile_from(source)
    assert source == saved and after['max_model_len'] == 307200
    assert after['kvmem'].pop('cpu_release_sealed_sums') is True
    assert after['kvmem'].pop('query_fps_cache_fp8') is True
    after['max_model_len'] = 524288
    assert after == saved


def workload():
    return dict(tasks=['t'], sampling={'temperature':.6}, seed=1000, workers=8,
        attempts_per_task=1, termination_policies={'repeat':32}, optimized_handoff=True,
        duration_seconds=0, setup_sha256={'a':'b'}, protocol_provenance={'a':'b'},
        claude_sha256='fixed', scoring='frozen', carried_results=[],
        profile={'max_model_len':524288, 'kvmem':{'graph_alpha':0}})


@pytest.mark.parametrize('key', ['tasks','sampling','seed','workers','attempts_per_task',
    'termination_policies','optimized_handoff','duration_seconds','setup_sha256',
    'protocol_provenance','claude_sha256','scoring','carried_results'])
def test_old_workload_and_answers_cannot_be_silently_changed(key):
    before = workload()
    after = deepcopy(before)
    after['profile'] = queue.profile_from(before['profile'])
    queue.validate_workload(before, after)
    after[key] = ['changed']
    with pytest.raises(ValueError): queue.validate_workload(before, after)


def test_recall_change_is_rejected():
    before = workload(); after = deepcopy(before)
    after['profile'] = queue.profile_from(before['profile'])
    after['profile']['kvmem']['graph_alpha'] = .2
    with pytest.raises(ValueError, match='Only the requested'):
        queue.validate_workload(before, after)


def test_prepare_refuses_existing_root_without_mutating_it(tmp_path):
    marker = tmp_path/'evidence';marker.write_text('original')
    with pytest.raises(ValueError, match='no automatic retry'):
        queue.prepare(tmp_path/'missing', tmp_path, tmp_path/'no-report')
    assert marker.read_text() == 'original'


def test_private_transcript_hash_uses_fixed_image_readonly_without_network(tmp_path, monkeypatch):
    private = tmp_path/'first16/cases/task/claude-config/projects/session.jsonl'
    private.parent.mkdir(parents=True); private.write_text('original')
    queue.q.save(tmp_path/'first16/manifest.json', {'tasks':[
        {'agent_image':'image:tag', 'images':{'image:tag':'sha256:fixed'}}]})
    sha = queue.q.sha
    def read(path):
        if path == private: raise PermissionError('container-owned')
        return sha(path)
    monkeypatch.setattr(queue.q, 'sha', read)
    def docker(argv, **kwargs):
        assert argv[:4] == ['docker', 'run', '--rm', '--network']
        assert argv[4] == 'none' and '--read-only' in argv
        stat = private.stat()
        assert argv[argv.index('--user')+1] == f'{stat.st_uid}:{stat.st_gid}'
        assert f'type=bind,src={tmp_path},dst=/evidence,readonly' in argv
        assert argv[-4:] == ['sha256sum','sha256:fixed','--','/evidence/'+str(private.relative_to(tmp_path))]
        return sha(private)+'  '+argv[-1]+'\n'
    monkeypatch.setattr(queue.subprocess, 'check_output', docker)
    result = queue.preserved_hashes(tmp_path)
    assert result[str(private.relative_to(tmp_path))] == sha(private)
    assert private.read_text() == 'original'
