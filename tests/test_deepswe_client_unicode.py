import hashlib
import json
import subprocess

import pytest
from scripts.patch_deepswe_client_unicode import patch, CALL, PATCH


def test_pinned_patch_only_changes_the_known_normalizer_call():
    data = b'prefix'+CALL+b'suffix'
    fixed, audit = patch(data, hashlib.sha256(data).hexdigest())
    assert fixed == b'prefix'+PATCH+b'suffix' and len(data)==len(fixed)
    assert sum(a!=b for a,b in zip(data,fixed)) == audit['changed_bytes'] == 3
    for wrong in ('0'*64, hashlib.sha256(fixed).hexdigest()):
        with pytest.raises(ValueError): patch(data, wrong)
    with pytest.raises(ValueError): patch(CALL*2, hashlib.sha256(CALL*2).hexdigest())


def test_patched_call_preserves_actual_json_values_not_just_rendered_appearance():
    # Execute the exact replacement call with identity schema/tool hooks. It
    # must skip the extra unescaper, while retaining both existing hooks.
    values = [r'\u0000-\u0014', '\u0000-\u0014', r'\\u0041', r'\ud83d\ude00',
              {'x':[r'\u0041', 'A', 1, 1.5, True, None, '\\"']}]
    script = '''const values=JSON.parse(process.argv[1]);
      const rX8=()=>{throw Error("extra decode called")};
      let calls=0;const yW5=x=>(calls++,x),iz4=(z,x,q)=>(calls++,x);
      const z={inputSchema:{},inputJSONSchema:{}},q={};let out=[];
      for(let A of values){''' + PATCH.decode() + ''';out.push(A)}
      console.log(JSON.stringify({out,calls}));'''
    output = json.loads(subprocess.check_output(['node','-e',script,json.dumps(values)], text=True))
    assert output == {'out': values, 'calls': 2*len(values)}
