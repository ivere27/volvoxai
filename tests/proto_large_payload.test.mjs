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
import volvoxai_pb2 as pb
payload = sys.stdin.buffer.read()
request = pb.ExecuteRequest(context_id=70000000000000003, inputs=pb.InputBindings(values=[
    pb.Tensor(name='cross_k_α', shape=[len(payload)], dtype=pb.DataType.DATA_TYPE_U8, inline=payload),
    pb.Tensor(name='token', shape=[1], dtype=pb.DataType.DATA_TYPE_I32, inline=b'\\xff\\xff\\xff\\xff'),
]))
sys.stdout.buffer.write(request.SerializeToString())
`;
const payload = new Uint8Array(3 * 1024 * 1024 + 3);
for (let index = 0; index < payload.length; index++) payload[index] = (index * 17 + (index >>> 8)) & 255;
const expected = execFileSync('python3', ['-c', python], {
  cwd: root, input: payload, maxBuffer: 8 * 1024 * 1024,
});

for (const profile of ['', 'inference/']) {
  test(`${profile || 'full/'} protobuf strings preserve leading BOM, NUL and Unicode`, async () => {
    const pb = await import(`../runtime/generated/typescript/${profile}volvoxai_lite.ts`);
    for (const text of ['', '\ufeff', '\ufeff한글😀\0', 'A\ufeffB\0', '\ufeff\ufeffx', 'ordinary English 123']) {
      const bytes = new TextEncoder().encode(text);
      // An independently assembled one-field protobuf string is also the
      // required encoder output. Every fixture here fits a one-byte length.
      const expected = Uint8Array.from([10, bytes.length, ...bytes]);
      const encoded = new pb.DecodedText({text}).toBinary();
      assert.deepEqual(encoded, expected);
      assert.equal(pb.DecodedText.fromBinary(expected).text, text);
      assert.deepEqual(pb.DecodedText.fromBinary(encoded).toBinary(), expected);
    }
    // The normalization changes BOM handling only; existing nonfatal invalid
    // UTF-8 replacement remains in the generated wire codec.
    assert.equal(pb.DecodedText.fromBinary(Uint8Array.of(10, 1, 255)).text, '\ufffd');
    const strings = ['\ufeffcpu', '\0', '한😀', '\ufeff\ufeff'];
    const repeatedWire = Uint8Array.from(strings.flatMap(text => {
      const value = new TextEncoder().encode(text);
      return [18, value.length, ...value];
    }));
    const policy = new pb.BackendPolicy({backends: strings});
    assert.deepEqual(policy.toBinary(), repeatedWire);
    assert.deepEqual(pb.BackendPolicy.fromBinary(repeatedWire).backends, strings);
    const raw = Uint8Array.of(0xef, 0xbb, 0xbf, 0, 255);
    const nested = new pb.ExecuteRequest({inputs: new pb.InputBindings({values: [
      new pb.Tensor({name: '\ufeffpixels\0한😀', inline: raw}),
    ]})});
    const decoded = pb.ExecuteRequest.fromBinary(nested.toBinary());
    assert.equal(decoded.inputs.values[0].name, '\ufeffpixels\0한😀');
    assert.deepEqual(decoded.inputs.values[0].inline, raw);
  });

  test(`${profile || 'full/'} transport protobuf error strings preserve BOM and NUL`, async () => {
    const runtime = await import(`../runtime/generated/typescript/${profile}synurang_runtime.ts`);
    const message = '\ufefferror 한😀\0', bytes = new TextEncoder().encode(message);
    const details = Uint8Array.from([18, bytes.length, ...bytes]);
    let released = 0, destroyed = 0;
    const host = runtime.createLinkedHost({createInstance: () => ({
      setWakeup() {}, open() {return 1n;}, send() {return 0;}, halfClose() {return 0;},
      receive() {return {kind: 'finished', code: 13, data: details};},
      cancel() {}, release() {released++;}, poll() {return 0;}, hasWork() {return false;},
      destroy() {destroyed++;return 0;},
    })});
    let call;
    try {
      call = await host.open({path: '/volvoxai.v1.VxTextService/DecodeTokens', requestStream: false, responseStream: false});
      await assert.rejects(call.recv(), error => {
        assert.ok(error instanceof runtime.RpcError);
        assert.equal(error.code, 13);
        assert.equal(error.message, message);
        assert.deepEqual(error.details, details);
        return true;
      });
    } finally {
      if (call) await call.close();
      await host.close();
    }
    assert.equal(released, 1);assert.equal(destroyed, 1);
  });

  test(`${profile || 'full/'} large nested bytes match Python and retain their snapshot`, async () => {
    const pb = await import(`../runtime/generated/typescript/${profile}volvoxai_lite.ts`);
    const backing = new Uint8Array(payload.length + 41).fill(0xa5);
    const view = backing.subarray(17, 17 + payload.length);
    view.set(payload);
    const request = new pb.ExecuteRequest({ contextId: 70000000000000003n, inputs: new pb.InputBindings({values: [
      new pb.Tensor({ name: 'cross_k_α', shape: [BigInt(payload.length)], dtype: pb.DataType.DATA_TYPE_U8, inline: view }),
      new pb.Tensor({ name: 'token', shape: [1n], dtype: pb.DataType.DATA_TYPE_I32, inline: Uint8Array.of(255, 255, 255, 255) }),
    ]}) });
    const encoded = request.toBinary();
    assert.deepEqual(Buffer.from(encoded), expected);
    const decoded = pb.ExecuteRequest.fromBinary(encoded);
    assert.equal(decoded.contextId, 70000000000000003n);
    assert.deepEqual(decoded.inputs.values[0].inline, payload);
    assert.deepEqual(decoded.inputs.values[1].inline, Uint8Array.of(255, 255, 255, 255));
    backing.fill(0x5a);
    request.inputs.values[0].inline = new Uint8Array();
    request.toBinary();
    assert.deepEqual(Buffer.from(encoded), expected);
    assert.deepEqual(decoded.inputs.values[0].inline, payload);
    encoded.fill(0);
    assert.deepEqual(decoded.inputs.values[0].inline, payload);
  });
}
