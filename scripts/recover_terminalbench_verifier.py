"""One verifier-only replay of an unchanged finished answer; no Agent runner."""
import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile
import time
import tomllib


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,indent=2)+'\n')


def app_identity(path):
    with tarfile.open(path) as tar:
        return {m.name:[m.type.decode(),m.linkname,m.mode,
                       hashlib.sha256(tar.extractfile(m).read()).hexdigest() if m.isfile() else None]
                for m in tar}


def validate_source(result, verifier_files):
    execution=result.get('agent_execution') or {}
    if ((result.get('exception_info') or {}).get('exception_type')!='VerifierTimeoutError' or
            not execution.get('finished_at') or result.get('verifier_result') is not None):
        raise ValueError('Requires a completed unscored answer with diagnosed verifier timeout')
    if any(Path(p).name in ('reward.txt','reward.json','ctrf.json') for p in verifier_files):
        raise ValueError('Existing verifier result requires review, never replay it blindly')


def expected_verifier_tests(source):
    """Read this task's explicit pytest cases without executing hidden tests."""
    names = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith('test_'):
            continue
        variants = ['']
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and
                    ast.unparse(decorator.func) == 'pytest.mark.parametrize' and
                    len(decorator.args) == 2 and not decorator.keywords and
                    isinstance(ast.literal_eval(decorator.args[0]), str)):
                raise ValueError('Unsupported verifier parametrization needs review')
            values = ast.literal_eval(decorator.args[1])
            if variants != [''] or not isinstance(values, (list, tuple)) or not all(type(v) in (int, str) for v in values):
                raise ValueError('Unsupported verifier cases need review')
            variants = ['[' + str(v) + ']' for v in values]
        names.update('test_outputs.py::' + node.name + v for v in variants)
    if not names:
        raise ValueError('No original verifier cases found')
    return names


def verifier_reward(logs, expected):
    reward = logs / 'reward.txt'
    if not reward.exists() or reward.read_text().strip() not in ('0', '1'):
        raise ValueError('Verifier produced no valid binary reward')
    # test.sh also writes zero after a dependency command fails. Require proof
    # that all original pytest cases actually ran before classifying it.
    report = read(logs / 'ctrf.json')['results']
    tests = report['tests']
    expected = set(expected)
    names = {t['name'] for t in tests}
    if (report['summary']['tests'] != len(tests) or len(names) != len(tests) or
            any(t['status'] not in ('passed', 'failed') for t in tests)):
        raise ValueError('Incomplete verifier execution requires review')
    if names == expected:
        statuses = [t['status'] for t in tests]
    elif names == {n.split('[')[0] for n in expected}:
        # pytest-json-ctrf 0.3.5 strips parameter IDs and treats variants as
        # retries. Its aggregate alone cannot prove each case ran. The original
        # pytest -rA output supplies the complete, distinct case results.
        entries = re.findall(r'^(PASSED|FAILED) (?:\.\./tests/)?(test_outputs\.py::\S+)',
                             (logs/'test-stdout.txt').read_text(), re.M)
        observed = {name:status.lower() for status,name in entries}
        if len(entries) != len(expected) or set(observed) != expected:
            raise ValueError('Incomplete parametrized verifier evidence')
        for t in tests:
            variants = {s for n,s in observed.items() if n.split('[')[0] == t['name']}
            if t['status'] not in variants:
                raise ValueError('CTRF and pytest case outcomes disagree')
        statuses = list(observed.values())
    else:
        raise ValueError('Incomplete verifier execution requires review')
    value = int(reward.read_text())
    if value != int(all(s == 'passed' for s in statuses)):
        raise ValueError('Verifier report and reward disagree')
    return value


def prepare(root, source, trial, preparation):
    if root.exists():raise ValueError('Fresh verifier-only evidence directory required')
    folder=source/'jobs/formal'/trial;result=read(folder/'result.json')
    validate_source(result,(folder/'verifier').glob('*'))
    task=result['task_name'].removeprefix('terminal-bench/')
    if task!='torch-pipeline-parallelism':raise ValueError('Narrow approved verifier-recovery allowlist')
    ready=read(preparation/'verifier-dependencies-ready.json');snapshot=read(preparation/'snapshot.json')
    if not ready['passed'] or not ready['no_tests_executed'] or not ready['app_unchanged']:
        raise ValueError('Missing dependency qualification')
    original_app=app_identity(preparation/'original-app.tar')
    if original_app!=ready['app_content'] or ready['snapshot']!=snapshot['snapshot_image']:
        raise ValueError('Prepared verifier image changed the completed answer')
    config=tomllib.loads((source/'dataset/tasks'/task/'task.toml').read_text())
    if config['verifier']['timeout_sec']!=900 or config['environment']['gpus']!=0:
        raise ValueError('Unexpected frozen verifier limit/resources')
    root.mkdir(parents=True)
    import shutil
    shutil.copytree(source/'dataset/tasks'/task,root/'task')
    shutil.copy2(__file__,root/'controller.py')
    evidence={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    for name in ('snapshot.json','verifier-dependencies-ready.json','original-app.tar'):
        evidence[str(preparation/name)]=sha(preparation/name)
    manifest=dict(source=str(source),trial=trial,task=task,created=time.time(),
                  prepared_image=ready['prepared_image'],snapshot_image=snapshot['snapshot_image'],
                  app_identity=original_app,source_evidence=evidence,
                  files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},
                  config=config,agent_launch_allowed=False,verifier_timeout_sec=900,
                  expected_tests=sorted(expected_verifier_tests((root/'task/tests/test_outputs.py').read_text())),
                  scope='Single verifier retry after dependency-only timeout; original completed Agent answer unchanged; original failure retained')
    save(root/'manifest.json',manifest);save(root/'status.json',dict(status='prepared',at=time.time()))


def run(root):
    m=read(root/'manifest.json')
    for p,h in m['source_evidence'].items():
        if sha(p)!=h:raise ValueError('Original completed evidence changed: '+p)
    for p,h in m['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen verifier recovery changed: '+p)
    with (root/'started').open('x') as f:f.write(str(time.time()))
    env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
    def docker(*args,timeout=60):
        return subprocess.check_output(['docker',*args],text=True,env=env,timeout=timeout).strip()
    import psutil
    if psutil.virtual_memory().available < (m['config']['environment']['memory_mb']+12288)*1024**2:
        raise RuntimeError('Insufficient host reserve for verifier; no answer retry')
    name=root.name
    logs=root/'verifier';logs.mkdir()
    cfg=m['config']['environment']
    args=['create','--name',name,'--label','kvmem.scope=terminalbench-verifier-only-recovery',
          '--cpus',str(cfg['cpus']),'--memory',str(cfg['memory_mb'])+'m',
          '--entrypoint','/bin/sh','--mount','type=bind,source='+str(logs)+',target=/logs/verifier']
    for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY',
                'ftp_proxy','FTP_PROXY','socks_proxy','SOCKS_PROXY','no_proxy','NO_PROXY'):
        args+=['-e',key+'=']
    source=Path(m['source'])
    args+=['--mount','type=bind,source='+str(source/'download-cache')+',target=/opt/kvmem-download-cache,readonly',
           '-e','UV_INSTALLER_GITHUB_BASE_URL=file:///opt/kvmem-download-cache',
           '-e','UV_PYTHON_INSTALL_MIRROR=file:///opt/kvmem-download-cache/astral-sh/python-build-standalone/releases/download']
    cid=docker(*args,m['prepared_image'],'-c','sleep infinity')
    save(root/'container.json',dict(id=cid,image=m['prepared_image'],at=time.time()))
    docker('start',cid)
    try:
        with (root/'app-before.tar').open('wb') as f:
            subprocess.run(['docker','cp',cid+':/app','-'],env=env,stdout=f,check=True)
        if app_identity(root/'app-before.tar')!=m['app_identity']:
            raise ValueError('Answer changed before verifier launch')
        save(root/'status.json',dict(status='verifying',at=time.time(),no_model_requests=True))
        started=time.monotonic()
        docker('cp',str(root/'task/tests')+'/.',cid+':/tests',timeout=60)
        remaining=m['verifier_timeout_sec']-(time.monotonic()-started)
        with (root/'exec.log').open('wb') as f:
            p=subprocess.run(['docker','exec',cid,'bash','-c',
                              'chmod +x /tests/test.sh && /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1'],
                             env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        reward=verifier_reward(logs,m['expected_tests'])
        result=read(source/'jobs/formal'/m['trial']/'result.json')
        result.update(verifier_result={'rewards':{'reward':reward}},exception_info=None,
                      verifier_recovery={'source_result':str(source/'jobs/formal'/m['trial']/'result.json'),
                                         'manifest':str(root/'manifest.json'),'seconds':time.monotonic()-started,
                                         'answer_regenerated':False,'original_failure_preserved':True})
        save(root/'result.json',result)
        save(root/'execution-audit.json',dict(passed=True,answer_unchanged=True,no_model_requests=True,
             verifier_timeout_sec=900,verifier_exit_code=p.returncode,
             reward=result['verifier_result']['rewards']['reward'],source_result_sha256=sha(source/'jobs/formal'/m['trial']/'result.json')))
        save(root/'status.json',dict(status='completed',at=time.time(),reward=result['verifier_result']['rewards']['reward']))
    except BaseException as error:
        save(root/'failure.json',dict(at=time.time(),error=repr(error)))
        save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error)))
        raise
    finally:
        current=read(root/'container.json')
        if current['id']!=cid:raise ValueError('Verifier container identity changed')
        docker('stop','--time','2',cid)
        detail=json.loads(docker('inspect',cid))[0]
        save(root/'cleanup.json',dict(remaining_processes=[cid] if detail['State']['Running'] else [],
                                     remaining_owned_gpu_pids=[],container_preserved=cid))
        save(root/'evidence-hashes.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evidence-hashes.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--trial');p.add_argument('--preparation',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.root.resolve(),a.source.resolve(),a.trial,a.preparation.resolve())
    else:run(a.root.resolve())
