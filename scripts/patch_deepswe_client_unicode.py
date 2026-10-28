"""Prepare an isolated pinned client that preserves decoded JSON string values.

The caller must provide the independently recorded SHA256 of the unmodified
client. No installed executable is edited. Equal-length replacement preserves
the embedded Bun executable offsets; the new executable needs a launch test.
"""
import argparse
import hashlib
import json
from pathlib import Path

CALL = b'let Y=rX8(yW5(A,z.inputSchema,z.inputJSONSchema));A=Y,A=iz4(z,Y,q)'
PATCH = b'let Y=   (yW5(A,z.inputSchema,z.inputJSONSchema));A=Y,A=iz4(z,Y,q)'


def patch(data, expected_sha256):
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Pinned client hash differs; refusing an unknown client')
    if data.count(CALL) != 1 or len(CALL) != len(PATCH):
        raise ValueError('Expected exactly one known tool-input normalization site')
    offset = data.index(CALL)
    result = data[:offset] + PATCH + data[offset+len(CALL):]
    assert len(result) == len(data)
    return result, dict(original_sha256=expected_sha256,
        patched_sha256=hashlib.sha256(result).hexdigest(), offset=offset,
        changed_bytes=3, bytes=len(result),
        scope='Skip extra Unicode unescaping after JSON parsing, retain schema coercion and tool-specific normalization. '
              'Strict KVMem prefix identity is unchanged. Opt-in isolated client; do not overwrite installed binaries.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    value, audit = patch(a.source.read_bytes(), a.sha256)
    with a.output.open('xb') as f:
        f.write(value)
    a.output.chmod(0o755)
    a.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
