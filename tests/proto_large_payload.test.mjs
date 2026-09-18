import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = fileURLToPath(new URL('..', import.meta.url));
const python = `
import sys
# The generated codec is what this compares; the volvoxai wrapper would pull in
# the engine's NumPy dependency, which the pinned build image does not carry.
sys.path.insert(0, 'runtime/generated/python')
import volvoxai_lite as pb
payload = sys.stdin.buffer.read()
request = pb.ExecuteRequest(context_id=70000000000000003, inputs=[
    pb.Tensor(name='cross_k_α', shape=[len(payload)], dtype=pb.DataType.DATA_TYPE_U8, inline=payload),
    pb.Tensor(name='token', shape=[1], dtype=pb.DataType.DATA_TYPE_I32, inline=b'\\xff\\xff\\xff\\xff'),
])
sys.stdout.buffer.write(request.to_bytes())
`;
const payload = new Uint8Array(3 * 1024 * 1024 + 3);
for (let index = 0; index < payload.length; index++) payload[index] = (index * 17 + (index >>> 8)) & 255;
const expected = execFileSync('python3', ['-c', python], {
  cwd: root, input: payload, maxBuffer: 8 * 1024 * 1024,
});

for (const profile of ['', 'inference/']) {
  test(`${profile || 'full/'} large nested bytes match Python and retain their snapshot`, async () => {
    const pb = await import(`../runtime/generated/typescript/${profile}volvoxai_lite.ts`);
    const backing = new Uint8Array(payload.length + 41).fill(0xa5);
    const view = backing.subarray(17, 17 + payload.length);
    view.set(payload);
    const request = new pb.ExecuteRequest({ contextId: 70000000000000003n, inputs: [
      new pb.Tensor({ name: 'cross_k_α', shape: [BigInt(payload.length)], dtype: pb.DataType.DATA_TYPE_U8, inline: view }),
      new pb.Tensor({ name: 'token', shape: [1n], dtype: pb.DataType.DATA_TYPE_I32, inline: Uint8Array.of(255, 255, 255, 255) }),
    ] });
    const encoded = request.toBinary();
    assert.deepEqual(Buffer.from(encoded), expected);
    const decoded = pb.ExecuteRequest.fromBinary(encoded);
    assert.equal(decoded.contextId, 70000000000000003n);
    assert.deepEqual(decoded.inputs[0].inline, payload);
    assert.deepEqual(decoded.inputs[1].inline, Uint8Array.of(255, 255, 255, 255));
    backing.fill(0x5a);
    request.inputs[0].inline = new Uint8Array();
    request.toBinary();
    assert.deepEqual(Buffer.from(encoded), expected);
    assert.deepEqual(decoded.inputs[0].inline, payload);
    encoded.fill(0);
    assert.deepEqual(decoded.inputs[0].inline, payload);
  });
}
